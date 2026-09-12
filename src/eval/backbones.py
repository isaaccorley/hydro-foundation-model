"""Frozen encoders behind one interface for linear-probe evaluation.

Every backbone takes a 12-band Sentinel-2 L2A DN stack in tif order
[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] (float32, DN units, 0 = nodata) and returns
patch tokens (N, g, g, C). Each backbone applies its OWN published preprocessing; bands a
dataset lacks are zero-filled *after* normalization (= band mean), which is the standard
workaround and is reported as a caveat.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import timm
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.pretrain.model import build_encoder  # noqa: E402
from src.pretrain.stats import MAD_STD, MEDIAN, QUANTIFICATION, detect_boa_offset  # noqa: E402

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB_TIF = (3, 2, 1)
PRESET_BANDS = {"all": tuple(range(12)), "landsat6": (1, 2, 3, 7, 10, 11)}
PRESET_RGB_POS = {"all": (3, 2, 1), "landsat6": (2, 1, 0)}


def _offset_corrected_reflectance(dn12: np.ndarray, band_idx, available: set[int]):
    """(N,12,H,W) DN -> (N,len(band_idx),H,W) reflectance with per-patch BOA offset removed.

    Offset is detected only on bands the dataset actually provides.
    """
    out = np.zeros((dn12.shape[0], len(band_idx), *dn12.shape[2:]), np.float32)
    avail_idx = [i for i in range(12) if i in available]
    for n in range(dn12.shape[0]):
        off = detect_boa_offset(dn12[n, avail_idx]) if avail_idx else 0.0
        for j, b in enumerate(band_idx):
            if b in available:
                x = dn12[n, b]
                out[n, j] = np.where(x != 0, (x - off) / QUANTIFICATION, np.nan)
            else:
                out[n, j] = np.nan
    return out


@dataclass
class Backbone:
    name: str
    grid_at_256: int

    def encode(self, dn12: np.ndarray, available: set[int], res: int = 256, device="cpu", bs: int = 8) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def _run(self, fn, x: torch.Tensor, res: int, device, bs: int) -> np.ndarray:
        out = []
        for i in range(0, len(x), bs):
            xb = x[i : i + bs].to(device)
            if res != xb.shape[-1]:
                xb = F.interpolate(xb, size=(res, res), mode="bicubic", align_corners=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                t = fn(xb)
            out.append(t.float().cpu())
        return torch.cat(out).numpy()


class DinoV3SatRGB(Backbone):
    """timm sat493m on hydroloc-style quicklooks clip(DN[B04,B03,B02]/3000)."""

    def __init__(self):
        super().__init__("dinov3-sat (rgb quicklook)", 16)
        self.model = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=True, num_classes=0, img_size=256).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

    def encode(self, dn12, available, res=256, device="cpu", bs=8):
        self.model.to(device)
        x = torch.from_numpy(np.clip(dn12[:, list(RGB_TIF)] / 3000.0, 0, 1).astype(np.float32))
        x = (x - self.mean) / self.std
        g = res // 16
        return self._run(lambda xb: self.model.forward_features(xb)[:, self.model.num_prefix_tokens :].reshape(len(xb), g, g, -1), x, res, device, bs)


class HydroMSI(Backbone):
    """Hydro DINO checkpoint (or untrained inflated DINOv3-SAT when ckpt=None) on MSI input.

    Input pipeline is the training one: per-patch BOA offset -> reflectance -> robust z-score
    (checkpoint's `norm` block, else MEDIAN/MAD_STD) -> clamp -> nodata/missing = 0.
    """

    def __init__(self, preset: str, ckpt: str | None = None):
        tag = f"hydro-{preset}@final" if ckpt else f"sat-inflated@0 ({preset})"
        super().__init__(tag, 16)
        self.band_idx = PRESET_BANDS[preset]
        enc = build_encoder(variant="sat493m", in_chans=len(self.band_idx), img_size=256, rgb_indices=PRESET_RGB_POS[preset])
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = enc.load_state_dict(ck["teacher_encoder"], strict=False)
            assert not unexpected and all(k.startswith("head") for k in missing)
            norm = ck["norm"]
            self.mean = torch.tensor(norm["mean"]).view(1, -1, 1, 1); self.std = torch.tensor(norm["std"]).view(1, -1, 1, 1)
            self.clamp = tuple(norm["clamp"]) if norm.get("clamp") else None
            self.name = f"hydro-{preset}@{ck['step']}"
        else:
            idx = list(self.band_idx)
            self.mean = MEDIAN[idx].view(1, -1, 1, 1); self.std = MAD_STD[idx].view(1, -1, 1, 1); self.clamp = (-5.0, 15.0)
        self.model = enc.eval()

    def encode(self, dn12, available, res=256, device="cpu", bs=8):
        self.model.to(device)
        refl = _offset_corrected_reflectance(dn12, self.band_idx, available)
        x = (torch.from_numpy(refl) - self.mean) / self.std
        if self.clamp:
            x = x.clamp(*self.clamp)
        x = torch.nan_to_num(x, nan=0.0)
        g = res // 16
        return self._run(lambda xb: self.model.forward_features(xb)[:, self.model.num_prefix_tokens :].reshape(len(xb), g, g, -1), x, res, device, bs)


class HydroRGBCkpt(Backbone):
    """Hydro rgb checkpoint: same quicklook pipeline as DinoV3SatRGB, teacher weights loaded."""

    def __init__(self, ckpt: str):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        super().__init__(f"hydro-rgb@{ck['step']}", 16)
        self.model = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=False, num_classes=0, img_size=256)
        missing, unexpected = self.model.load_state_dict(ck["teacher_encoder"], strict=False)
        assert not unexpected and all(k.startswith("head") for k in missing)
        self.model.eval()
        self.mean = torch.tensor(ck["norm"]["mean"]).view(1, 3, 1, 1); self.std = torch.tensor(ck["norm"]["std"]).view(1, 3, 1, 1)

    encode = DinoV3SatRGB.encode


class OlmoEarthBase(Backbone):
    """OlmoEarth v1.2 Base, 12-band S2 L2A, native 8-px patches -> 32x32 tokens at 256."""

    BAND_REORDER = [1, 2, 3, 7, 4, 5, 6, 8, 10, 11, 0, 9]  # tif order -> OlmoEarth S2 L2A order

    def __init__(self):
        super().__init__("olmoearth-v1.2-base", 32)
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
        self.model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE, load_weights=True).eval()
        self.normalizer, self.Modality, self.Sample = Normalizer(std_multiplier=2.0), Modality, MaskedOlmoEarthSample

    @torch.no_grad()
    def encode(self, dn12, available, res=256, device="cpu", bs=8):
        self.model.to(device)
        arr = dn12[:, self.BAND_REORDER].transpose(0, 2, 3, 1)[:, :, :, None, :]  # (N,H,W,1,12) raw DN
        data = self.normalizer.normalize(self.Modality.SENTINEL2_L2A, arr).astype(np.float32)
        missing = [j for j, b in enumerate(self.BAND_REORDER) if b not in available]
        data[..., missing] = 0.0  # normalized space -> band mean
        out = []
        for i in range(0, len(data), bs):
            xb = torch.from_numpy(data[i : i + bs]).to(device)
            n, h, w = xb.shape[:3]
            s = self.Sample(timestamps=torch.zeros(n, 1, 3, dtype=torch.long, device=device), sentinel2_l2a=xb,
                            sentinel2_l2a_mask=torch.zeros(n, h, w, 1, dtype=torch.long, device=device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                tok = self.model.encoder(s, patch_size=8, input_res=10, fast_pass=True)["tokens_and_masks"].sentinel2_l2a
            out.append(tok.reshape(n, h // 8, w // 8, -1).float().cpu())
        return torch.cat(out).numpy()


class PrithviEO2(Backbone):
    """Prithvi-EO-2.0-300M (HLS bands = S2 B02,B03,B04,B8A,B11,B12), standalone loader from the HF repo."""

    TIF_IDX = (1, 2, 3, 8, 10, 11)

    def __init__(self, repo_dir: str):
        super().__init__("prithvi-eo-2.0-300m", 16)
        import importlib.util, json
        spec = importlib.util.spec_from_file_location("prithvi_mae", os.path.join(repo_dir, "prithvi_mae.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cfg = json.load(open(os.path.join(repo_dir, "config.json")))["pretrained_cfg"]
        self.model = mod.PrithviViT(img_size=cfg["img_size"], patch_size=tuple(cfg["patch_size"]), num_frames=cfg["num_frames"],
                                    in_chans=cfg["in_chans"], embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"],
                                    mlp_ratio=cfg["mlp_ratio"], coords_encoding=cfg.get("coords_encoding", []))
        sd = torch.load(os.path.join(repo_dir, "Prithvi_EO_V2_300M.pt"), map_location="cpu", weights_only=False)
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected and not [k for k in missing if "pos_embed" not in k], (missing, unexpected)
        self.model.eval()
        self.mean = torch.tensor(cfg["mean"]).view(1, 6, 1, 1); self.std = torch.tensor(cfg["std"]).view(1, 6, 1, 1)

    def encode(self, dn12, available, res=256, device="cpu", bs=8):
        self.model.to(device)
        x = torch.from_numpy(dn12[:, list(self.TIF_IDX)].astype(np.float32))
        x = (x - self.mean) / self.std  # Prithvi stats are in DN (HLS scaled reflectance) units
        for j, b in enumerate(self.TIF_IDX):
            if b not in available:
                x[:, j] = 0.0
        g = res // 16
        return self._run(lambda xb: self.model.forward_features(xb.unsqueeze(2))[-1][:, 1:].reshape(len(xb), g, g, -1), x, res, device, bs)


class HydroSwinV2Legacy(Backbone):
    """Released Hydro Swin-v2 MSI (SimMIM). Stage-3 features: 16x16 x 512 at 256 px."""

    def __init__(self, config_path="checkpoints/swin-v2-msi/config.yaml"):
        super().__init__("hydro-swinv2-msi (legacy SimMIM)", 16)
        import types
        # src.models/__init__ eagerly imports the lightning/mlflow eval stack; stub it out.
        stub = types.ModuleType("src.models.segmentation"); stub.CustomSemanticSegmentationTask = object
        sys.modules.setdefault("src.models.segmentation", stub)
        from src.models.load_pretrained import swin_v2
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            self.model, self.transforms, _ = swin_v2(config_path)
        finally:
            os.chdir(cwd)
        self.model.eval()

    def encode(self, dn12, available, res=256, device="cpu", bs=8):
        self.model.to(device)
        x = torch.from_numpy(dn12.astype(np.float32))
        for b in range(12):
            if b not in available:
                x[:, b] = float("nan")
        x = self.transforms(x)  # /10000 then MEAN/STD (legacy stats, no offset handling: that IS the legacy pipeline)
        x = torch.nan_to_num(x, nan=0.0)
        def fn(xb):
            f = self.model.get_intermediate_layers(xb, n=(2,), reshape=True)[0]  # (B,512,16,16)
            return f.permute(0, 2, 3, 1)
        return self._run(fn, x, 256, device, bs)


def build_backbones(names: list[str], prithvi_dir: str | None) -> list[Backbone]:
    out = []
    for n in names:
        if n == "dinov3-sat":
            out.append(DinoV3SatRGB())
        elif n.startswith("inflated-"):
            out.append(HydroMSI(n.split("-", 1)[1]))
        elif n.startswith("hydro-rgb:"):
            out.append(HydroRGBCkpt(n.split(":", 1)[1]))
        elif n.startswith("hydro-"):
            preset, ckpt = n.split(":", 1)
            out.append(HydroMSI(preset.split("-", 1)[1], ckpt))
        elif n == "olmoearth":
            out.append(OlmoEarthBase())
        elif n == "prithvi":
            out.append(PrithviEO2(prithvi_dir))
        elif n == "swinv2-legacy":
            out.append(HydroSwinV2Legacy())
        else:
            raise ValueError(n)
    return out
