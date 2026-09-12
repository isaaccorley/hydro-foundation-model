"""Zero-shot dense-feature probe on hand-labelled water datasets: frozen encoders vs. water labels.

Generalizes scripts/zero_shot_swed.py with `--dataset {swed,s1s2water,sen1floods11}`. Every loader
returns 12-band DN arrays in tif order [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] at
256x256 and masks in {-1 ignore, 0 background, 1 water}. Bands a dataset lacks are zero (= nodata
in the MSI path -> set to the band mean after z-scoring, i.e. occluded).

  swed          SWED test chips, 12-band L2A, labels from thresholded indices (relative use only)
  s1s2water     S1S2-Water (DLR 2023) test-split scenes, S2 **L1C TOA**, 6 bands B02,B03,B04,B08,
                B11,B12 only; hand-labelled water + valid masks; chipped to 256 with a fixed seed
  sen1floods11  Sen1Floods11 hand-labelled test split, S2 **L1C** 13-band 512 chips (B10 dropped),
                tiled to 256; labels {-1,0,1}

Encoders, readouts, occlusion and figure are as in zero_shot_swed.py (see that docstring).

Usage:
  python scripts/zero_shot_water.py --dataset s1s2water --ckpt output/dino/landsat6/landsat6_final.pt
"""

import argparse
import csv
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
from rasterio.windows import Window
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
CHIP = 256
OCCLUSION_GROUPS = {"SWIR (B11,B12)": {"B11", "B12"}, "NIR (B08,B8A)": {"B08", "B8A"}, "RGB (B02-B04)": {"B02", "B03", "B04"}}
BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark-datasets")


# ----------------------------------------------------------------------------- loaders
def load_swed(root, n, seed):
    img_dir, lab_dir = os.path.join(root, "test", "images"), os.path.join(root, "test", "labels")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if n and n < len(paths):
        paths = sorted(np.random.default_rng(seed).choice(paths, n, replace=False).tolist())
    dns, masks = [], []
    for p in paths:
        with rasterio.open(p) as f:
            dns.append(f.read().astype(np.float32))
        with rasterio.open(os.path.join(lab_dir, os.path.basename(p).replace("_image_", "_label_"))) as f:
            masks.append(np.clip(f.read(1).astype(np.int16), -1, 1))
    return [os.path.basename(p) for p in paths], np.stack(dns), np.stack(masks)


def load_sen1floods11(root, n, seed):
    """Hand-labelled test split; 13-band L1C 512 chips -> drop B10, tile to 4x256."""
    with open(os.path.join(root, "splits", "flood_test_data.csv")) as f:
        stems = [row[0].replace("_S1Hand.tif", "") for row in csv.reader(f) if row]
    keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # B1..B9,B8A order in file: B1,B2,...,B8,B8A,B9,B10,B11,B12
    names, dns, masks = [], [], []
    for s in stems:
        with rasterio.open(os.path.join(root, "S2Hand", f"{s}_S2Hand.tif")) as f:
            x = f.read().astype(np.float32)[keep]
        with rasterio.open(os.path.join(root, "LabelHand", f"{s}_LabelHand.tif")) as f:
            m = np.clip(f.read(1).astype(np.int16), -1, 1)
        for i in range(0, x.shape[1] - CHIP + 1, CHIP):
            for j in range(0, x.shape[2] - CHIP + 1, CHIP):
                names.append(f"{s}_{i}_{j}"); dns.append(x[:, i:i + CHIP, j:j + CHIP]); masks.append(m[i:i + CHIP, j:j + CHIP])
    if n and n < len(names):
        idx = np.random.default_rng(seed).choice(len(names), n, replace=False)
        names, dns, masks = [names[i] for i in idx], [dns[i] for i in idx], [masks[i] for i in idx]
    return names, np.stack(dns), np.stack(masks)


def load_s1s2water(root, n, seed, per_scene=16):
    """Test-split scenes (10980^2, 6-band L1C: Blue,Green,Red,NIR,SWIR1,SWIR2) -> random valid 256 chips.

    Chips are drawn where the hand-labelled valid mask is all 1 and the chip has >=2% and <=98%
    water, `per_scene` per scene (fixed seed). Missing bands (B01,B05-B07,B8A,B09) stay 0.
    """
    with open(os.path.join(root, "index.json")) as f:
        test_ids = json.load(f)["test"]
    dest = {0: 1, 1: 2, 2: 3, 3: 7, 4: 10, 5: 11}  # S1S2 band idx -> our 12-band position
    rng = np.random.default_rng(seed)
    names, dns, masks = [], [], []
    for sid in test_ids:
        img_p = os.path.join(root, sid, f"sentinel12_s2_{sid}_img.tif")
        if not all(os.path.exists(img_p.replace("_img", s)) for s in ("_img", "_msk", "_valid")):
            continue
        with rasterio.open(img_p) as fi, rasterio.open(img_p.replace("_img", "_msk")) as fm, rasterio.open(img_p.replace("_img", "_valid")) as fv:
            H, W = fi.height, fi.width
            got, tries = 0, 0
            while got < per_scene and tries < per_scene * 40:
                tries += 1
                r, c = int(rng.integers(0, H - CHIP)), int(rng.integers(0, W - CHIP))
                win = Window(c, r, CHIP, CHIP)
                v = fv.read(1, window=win)
                if v.min() == 0:
                    continue
                m = fm.read(1, window=win).astype(np.int16)
                wf = (m == 1).mean()
                if wf < 0.02 or wf > 0.98:
                    continue
                x6 = fi.read(window=win).astype(np.float32)
                x = np.zeros((12, CHIP, CHIP), np.float32)
                for s_i, d_i in dest.items():
                    x[d_i] = x6[s_i]
                names.append(f"{sid}_{r}_{c}"); dns.append(x); masks.append(m); got += 1
    if n and n < len(names):
        idx = rng.choice(len(names), n, replace=False)
        names, dns, masks = [names[i] for i in idx], [dns[i] for i in idx], [masks[i] for i in idx]
    return names, np.stack(dns), np.stack(masks)


LOADERS = {
    "swed": (load_swed, os.path.join(BENCH, "SWED")),
    "sen1floods11": (load_sen1floods11, os.path.join(BENCH, "Sen1Floods11")),
    "s1s2water": (load_s1s2water, os.path.join(BENCH, "S1S2Water")),
}


# ----------------------------------------------------------------------------- inputs / models
def rgb_input(dn, mean, std):
    x = torch.from_numpy(np.clip(dn[:, list(RGB)] / DIVISOR, 0.0, 1.0))
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def msi_input(dn, norm):
    idx = norm["band_indices"]
    mean = torch.tensor(norm["mean"]).view(-1, 1, 1)
    std = torch.tensor(norm["std"]).view(-1, 1, 1)
    out = []
    for d in dn:
        sub = d[idx]
        valid = torch.from_numpy(sub != 0)
        x = (torch.from_numpy(sub) - detect_boa_offset(sub)) / QUANTIFICATION
        x = (x - mean) / std
        if norm.get("clamp") is not None:
            x = x.clamp(*norm["clamp"])
        out.append(torch.where(valid, x, torch.zeros_like(x)))
    return torch.stack(out)


def build(ckpt_path, dn, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step, in_chans, variant, preset = ck["step"], ck["encoder"]["in_chans"], ck["args"]["variant"], ck["args"]["bands"]
    base_rgb = timm.create_model(f"vit_large_patch16_dinov3.{variant}", pretrained=True, num_classes=0, img_size=256)
    cfg = timm.data.resolve_model_data_config(base_rgb)
    x_rgb = rgb_input(dn, cfg["mean"], cfg["std"])
    entries = [("dinov3-sat (rgb)", base_rgb.eval().to(device), x_rgb, None)]
    if in_chans == 3:
        ours = timm.create_model(f"vit_large_patch16_dinov3.{variant}", pretrained=False, num_classes=0, img_size=256)
        x_ours, bands = x_rgb, None
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
def patch_tokens(model, x, res, device, bs=16):
    out = []
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device)
        if res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(res, res), mode="bicubic", align_corners=False)
        t = model.forward_features(xb)[:, model.num_prefix_tokens :]
        g = res // PATCH
        out.append(t.reshape(len(xb), g, g, -1).float().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- metrics
def pool_labels(masks, g):
    n, h, _ = masks.shape
    k = h // g
    m = masks.reshape(n, g, k, g, k).transpose(0, 1, 3, 2, 4).reshape(n, g, g, k * k)
    water, valid = (m == 1).sum(-1), (m >= 0).sum(-1)
    return np.where(valid == 0, -1, np.where(water * 2 > valid, 1, 0))


def iou_table(pred, gt):
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


def l2n(f):
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)


def kmeans_fit(tokens, gt, seed=0):
    n, g, _, c = tokens.shape
    km = KMeans(n_clusters=2, n_init=4, random_state=seed).fit(l2n(tokens.reshape(-1, c)))
    labels = km.labels_.reshape(n, g, g)
    a, b = iou_table(labels, gt), iou_table(1 - labels, gt)
    return (km, 0, labels, a) if a["miou"] >= b["miou"] else (km, 1, 1 - labels, b)


def kmeans_apply(km, flip, tokens, gt):
    n, g, _, c = tokens.shape
    labels = km.predict(l2n(tokens.reshape(-1, c))).reshape(n, g, g)
    return iou_table(1 - labels if flip else labels, gt)


def pca_rgb(tokens, seed=0):
    n, g, _, c = tokens.shape
    z = PCA(n_components=3, random_state=seed).fit_transform(tokens.reshape(-1, c))
    lo, hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
    return np.clip((z - lo) / (hi - lo + 1e-8), 0, 1).reshape(n, g, g, 3)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(LOADERS), required=True)
    ap.add_argument("--root", default=None, help="override dataset root")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None, help="default: output/zero_shot/<dataset>/<ckpt stem>")
    ap.add_argument("--n", type=int, default=0, help="subsample chips (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolutions", type=int, nargs="+", default=(256, 512))
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    loader, root = LOADERS[args.dataset]
    root = args.root or root
    out_dir = args.out_dir or os.path.join("output/zero_shot", args.dataset, os.path.splitext(os.path.basename(args.ckpt))[0])
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    names, dn, masks = loader(root, args.n, args.seed)
    keep = [i for i in range(len(names)) if (dn[i] != 0).any() and (masks[i] >= 0).any()]
    if len(keep) < len(names):
        print(f"dropping {len(names) - len(keep)} all-nodata / all-ignore chips", flush=True)
        names, dn, masks = [names[i] for i in keep], dn[keep], masks[keep]
    present = [BANDS[b] for b in range(12) if (dn[:, b] != 0).any()]
    lows = [min(np.percentile(d[b][d[b] != 0], 1) for b in range(12) if (d[b] != 0).any()) for d in dn]
    offset_frac = float(np.mean([lo >= 850 for lo in lows]))
    print(f"{args.dataset}: {len(names)} chips  water frac={(masks == 1).mean():.2f}  ignore={(masks < 0).mean():.3f}  "
          f"bands present={present}  BOA-offset detected in {offset_frac:.0%} of chips", flush=True)
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
                idx = [i for i, b in enumerate(bands) if b in members and b in present]
                if not idx:
                    continue
                xo = x.clone()
                xo[:, idx] = 0.0
                m = kmeans_apply(km, flip, patch_tokens(model, xo, res, args.device), gt)
                occlusion[name][gname] = {"miou": m["miou"], "delta": m["miou"] - full}
                print(f"{name:>28} occlude {gname:>14}: mIoU={m['miou']:.3f} ({m['miou'] - full:+.3f})", flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"dataset": args.dataset, "n_chips": len(names), "bands_present": present, "boa_offset_frac": offset_frac,
                   "segmentation": results, "band_occlusion": occlusion, "ckpt": args.ckpt, "preset": preset}, f, indent=2)

    idx = np.linspace(0, len(names) - 1, min(args.n_show, len(names))).astype(int)
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
    titles = ["RGB", "label"] + [t for n_, r_ in keys for t in (f"PCA {n_}@{r_}", f"k-means {n_}@{r_}")]
    for a, t in zip(axs[0], titles):
        a.set_title(t, fontsize=5)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig_path = os.path.join(out_dir, f"{args.dataset}_zero_shot.png")
    fig.savefig(fig_path, dpi=160)
    print(f"saved {fig_path} and metrics.json", flush=True)


if __name__ == "__main__":
    main()
