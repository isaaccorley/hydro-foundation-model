"""Zero-shot dense-feature probe on SWED test chips: frozen encoders vs. SWED water labels.

Works for any Hydro DINO checkpoint (rgb / landsat6 / all). The checkpoint's `encoder.in_chans`
and `norm` block define the input pipeline, so the probe cannot drift from training:
  rgb   : hydroloc quicklook clip(DN[B04,B03,B02]/3000) -> DINOv3-SAT mean/std
  MSI   : band subset -> per-patch BOA offset -> reflectance -> z-score(ckpt stats) -> clamp -> nodata=0

Encoders compared:
  dinov3-sat (rgb)            timm sat493m weights on RGB quicklooks (frozen, untrained)
  sat-inflated@0 (<preset>)   sat493m + inflated patch embed, step 0 -> untrained MSI reference
  hydro-<preset>@<step>       the checkpoint's teacher encoder
For each encoder x resolution (256 native, 512 bicubic-upsampled = 4x tokens):
  * PCA(3) of patch tokens -> false-color maps
  * unsupervised 2-cluster k-means on L2-normalized tokens over the test set; the single global
    cluster->{water,bg} flip maximizing IoU; scored vs labels majority-pooled to the token grid
For MSI encoders, band occlusion @256: zero a band group in normalized space (= band mean),
re-assign tokens to the fitted centroids, report the mIoU drop -> which bands the encoder uses.
Physics reference: MNDWI = (B03 - B11)/(B03 + B11) > 0.

Usage:
  python scripts/zero_shot_swed.py --ckpt output/dino/landsat6/landsat6_step7500.pt
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import timm
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pretrain.model import build_encoder  # noqa: E402
from src.pretrain.stats import QUANTIFICATION, detect_boa_offset  # noqa: E402

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
RGB = (3, 2, 1)  # B04, B03, B02
G, SWIR1 = 2, 10
DIVISOR = 3000.0
PATCH = 16
OCCLUSION_GROUPS = {"SWIR (B11,B12)": {"B11", "B12"}, "NIR (B08,B8A)": {"B08", "B8A"}, "RGB (B02-B04)": {"B02", "B03", "B04"}}


def load_swed_test(root: str, n: int, seed: int = 0):
    img_dir, lab_dir = os.path.join(root, "test", "images"), os.path.join(root, "test", "labels")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"no tifs under {img_dir}")
    if n and n < len(paths):
        paths = sorted(np.random.default_rng(seed).choice(paths, n, replace=False).tolist())
    dns, masks = [], []
    for p in paths:
        with rasterio.open(p) as f:
            dns.append(f.read().astype(np.float32))
        with rasterio.open(os.path.join(lab_dir, os.path.basename(p).replace("_image_", "_label_"))) as f:
            masks.append(np.clip(f.read(1).astype(np.int16), -1, 1))
    return paths, np.stack(dns), np.stack(masks)


def rgb_input(dn: np.ndarray, mean, std) -> torch.Tensor:
    """hydroloc quicklook stretch -> DINOv3-SAT normalization. (N,3,H,W)"""
    x = torch.from_numpy(np.clip(dn[:, list(RGB)] / DIVISOR, 0.0, 1.0))
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def msi_input(dn: np.ndarray, norm: dict) -> torch.Tensor:
    """Exactly MSIHydroDataset.__getitem__, driven by the checkpoint's norm block. (N,C,H,W)"""
    idx = norm["band_indices"]
    mean = torch.tensor(norm["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(norm["std"]).view(1, -1, 1, 1)
    out = []
    for d in dn:
        sub = d[idx]
        valid = torch.from_numpy(sub != 0)
        x = (torch.from_numpy(sub) - detect_boa_offset(sub)) / QUANTIFICATION
        x = (x - mean[0]) / std[0]
        if norm.get("clamp") is not None:
            x = x.clamp(*norm["clamp"])
        out.append(torch.where(valid, x, torch.zeros_like(x)))
    return torch.stack(out)


def build(ckpt_path: str, dn: np.ndarray, device):
    """-> list of (name, model, input tensor, band_names or None), plus preset label."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step, in_chans = ck["step"], ck["encoder"]["in_chans"]
    variant = ck["args"]["variant"]
    preset = ck["args"]["bands"]

    base_rgb = timm.create_model(f"vit_large_patch16_dinov3.{variant}", pretrained=True, num_classes=0, img_size=256)
    cfg = timm.data.resolve_model_data_config(base_rgb)
    x_rgb = rgb_input(dn, cfg["mean"], cfg["std"])
    entries = [("dinov3-sat (rgb)", base_rgb.eval().to(device), x_rgb, None)]

    if in_chans == 3:
        ours = timm.create_model(f"vit_large_patch16_dinov3.{variant}", pretrained=False, num_classes=0, img_size=256)
        x_ours, bands = x_rgb, None
        ref = None
    else:
        norm = ck["norm"]
        bands = [BANDS[i] for i in norm["band_indices"]]
        rgb_pos = tuple(bands.index(b) for b in ("B04", "B03", "B02"))
        ref = build_encoder(variant=variant, in_chans=in_chans, img_size=256, rgb_indices=rgb_pos)
        ours = build_encoder(variant=variant, in_chans=in_chans, img_size=256, rgb_indices=rgb_pos)
        x_ours = msi_input(dn, norm)
        entries.append((f"sat-inflated@0 ({preset})", ref.eval().to(device), x_ours, bands))
    missing, unexpected = ours.load_state_dict(ck["teacher_encoder"], strict=False)
    assert not unexpected and all(k.startswith("head") for k in missing), (missing, unexpected)
    entries.append((f"hydro-{preset}@{step}", ours.eval().to(device), x_ours, bands))
    return entries, preset


@torch.no_grad()
def patch_tokens(model, x: torch.Tensor, res: int, device, bs: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device)
        if res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(res, res), mode="bicubic", align_corners=False)
        t = model.forward_features(xb)[:, model.num_prefix_tokens :]
        g = res // PATCH
        out.append(t.reshape(len(xb), g, g, -1).float().cpu())
    return torch.cat(out).numpy()


def pool_labels(masks: np.ndarray, g: int) -> np.ndarray:
    n, h, _ = masks.shape
    k = h // g
    m = masks.reshape(n, g, k, g, k).transpose(0, 1, 3, 2, 4).reshape(n, g, g, k * k)
    water, valid = (m == 1).sum(-1), (m >= 0).sum(-1)
    return np.where(valid == 0, -1, np.where(water * 2 > valid, 1, 0))


def iou_table(pred: np.ndarray, gt: np.ndarray) -> dict:
    v = gt >= 0
    p, t = pred[v], gt[v]
    res = {}
    for c, name in ((1, "water"), (0, "background")):
        inter = np.logical_and(p == c, t == c).sum()
        union = np.logical_or(p == c, t == c).sum()
        res[f"iou_{name}"] = float(inter / max(union, 1))
    res["miou"] = float((res["iou_water"] + res["iou_background"]) / 2)
    res["acc"] = float((p == t).mean())
    return res


def l2n(feats: np.ndarray) -> np.ndarray:
    return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)


def kmeans_fit(tokens: np.ndarray, gt: np.ndarray, seed: int = 0):
    """Fit k=2, choose the global flip maximizing mIoU. Returns (km, flip, seg, metrics)."""
    n, g, _, c = tokens.shape
    km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(l2n(tokens.reshape(-1, c)))
    labels = km.labels_.reshape(n, g, g)
    a, b = iou_table(labels, gt), iou_table(1 - labels, gt)
    return (km, 0, labels, a) if a["miou"] >= b["miou"] else (km, 1, 1 - labels, b)


def kmeans_apply(km, flip: int, tokens: np.ndarray, gt: np.ndarray) -> dict:
    n, g, _, c = tokens.shape
    labels = km.predict(l2n(tokens.reshape(-1, c))).reshape(n, g, g)
    return iou_table(1 - labels if flip else labels, gt)


def pca_rgb(tokens: np.ndarray, seed: int = 0) -> np.ndarray:
    n, g, _, c = tokens.shape
    z = PCA(n_components=3, random_state=seed).fit_transform(tokens.reshape(-1, c))
    lo, hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
    return np.clip((z - lo) / (hi - lo + 1e-8), 0, 1).reshape(n, g, g, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swed-root", default="benchmark-datasets/SWED")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None, help="default: output/zero_shot_swed/<ckpt stem>")
    ap.add_argument("--n", type=int, default=0, help="subsample test chips (0 = all)")
    ap.add_argument("--resolutions", type=int, nargs="+", default=(256, 512))
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join("output/zero_shot_swed", os.path.splitext(os.path.basename(args.ckpt))[0])
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    paths, dn, masks = load_swed_test(args.swed_root, args.n)
    print(f"SWED test chips: {len(paths)}  water frac={(masks == 1).mean():.2f}", flush=True)
    entries, preset = build(args.ckpt, dn, args.device)

    results, occlusion, segs, pcas, fitted = {}, {}, {}, {}, {}
    for res in args.resolutions:
        g = res // PATCH
        gt = pool_labels(masks, g)
        mndwi = (dn[:, G] - dn[:, SWIR1]) / (dn[:, G] + dn[:, SWIR1] + 1e-6)
        results[f"mndwi>0@{res}"] = iou_table(pool_labels(np.where(masks < 0, -1, (mndwi > 0).astype(np.int16)), g), gt)
        for name, model, x, bands in entries:
            tok = patch_tokens(model, x, res, args.device)
            km, flip, seg, m = kmeans_fit(tok, gt)
            results[f"{name}@{res}"], segs[(name, res)], pcas[(name, res)], fitted[(name, res)] = m, seg, pca_rgb(tok), (km, flip)
            print(f"{name:>28} @{res}: mIoU={m['miou']:.3f} IoU_water={m['iou_water']:.3f} acc={m['acc']:.3f}", flush=True)
        print(f"{'mndwi>0':>28} @{res}: mIoU={results[f'mndwi>0@{res}']['miou']:.3f}", flush=True)

    if not args.no_occlusion:
        res = args.resolutions[0]
        gt = pool_labels(masks, res // PATCH)
        for name, model, x, bands in entries:
            if bands is None:
                continue
            km, flip = fitted[(name, res)]
            full = results[f"{name}@{res}"]["miou"]
            occlusion[name] = {}
            for gname, members in OCCLUSION_GROUPS.items():
                idx = [i for i, b in enumerate(bands) if b in members]
                if not idx:
                    continue
                xo = x.clone()
                xo[:, idx] = 0.0  # normalized space: band mean
                m = kmeans_apply(km, flip, patch_tokens(model, xo, res, args.device), gt)
                occlusion[name][gname] = {"miou": m["miou"], "delta": m["miou"] - full}
                print(f"{name:>28} occlude {gname:>14}: mIoU={m['miou']:.3f} ({m['miou'] - full:+.3f})", flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"segmentation": results, "band_occlusion": occlusion, "ckpt": args.ckpt, "preset": preset}, f, indent=2)

    idx = np.linspace(0, len(paths) - 1, min(args.n_show, len(paths))).astype(int)
    keys = [(e[0], r) for r in args.resolutions for e in entries]
    ncol = 2 + 2 * len(keys)
    fig, axs = plt.subplots(len(idx), ncol, figsize=(1.5 * ncol, 1.5 * len(idx)))
    rgb_vis = np.clip(dn[:, list(RGB)] / DIVISOR, 0, 1)
    for r, i in enumerate(idx):
        axs[r, 0].imshow(rgb_vis[i].transpose(1, 2, 0))
        axs[r, 1].imshow(np.ma.masked_less(masks[i], 0), vmin=0, vmax=1, cmap="Blues", interpolation="none")
        for j, key in enumerate(keys):
            axs[r, 2 + 2 * j].imshow(pcas[key][i], interpolation="nearest")
            axs[r, 3 + 2 * j].imshow(segs[key][i], vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    titles = ["RGB", "SWED label"] + [t for n, r in keys for t in (f"PCA {n}@{r}", f"k-means {n}@{r}")]
    for a, t in zip(axs[0], titles):
        a.set_title(t, fontsize=5)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig_path = os.path.join(out_dir, "swed_zero_shot.png")
    fig.savefig(fig_path, dpi=160)
    print(f"saved {fig_path} and metrics.json", flush=True)


if __name__ == "__main__":
    main()
