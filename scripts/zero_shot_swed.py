"""Zero-shot dense-feature probe on SWED test chips: DINOv3-SAT baseline vs. a Hydro DINO teacher.

For each frozen encoder and input resolution (256 native, 512 bicubic-upsampled -> 4x more
patch tokens, the "high-res dense features" regime DINOv3 emphasizes):
  * PCA(3) of all patch tokens -> false-color feature maps
  * unsupervised water segmentation: k-means (k=2) on L2-normalized patch tokens across the
    whole test set; the cluster->{water,bg} mapping is the single global flip that maximizes
    IoU (standard unsupervised-segmentation protocol, no per-image tuning), scored against
    SWED labels majority-pooled to the token grid (label -1 ignored)
Physics reference: MNDWI = (B03 - B11) / (B03 + B11) > 0, pooled the same way.

RGB input replicates hydroloc's quicklooks exactly: clip(DN[B04,B03,B02] / 3000, 0, 1).

Usage:
  python scripts/zero_shot_swed.py --swed-root benchmark-datasets/SWED --ckpt output/dino/rgb/rgb_step5000.pt
"""

import argparse
import glob
import json
import os

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

RGB = (3, 2, 1)  # B04, B03, B02 in the 12-band [B01..B12] stack
G, SWIR1 = 2, 10
DIVISOR = 3000.0
PATCH = 16


def load_swed_test(root: str, n: int, seed: int = 0):
    img_dir, lab_dir = os.path.join(root, "test", "images"), os.path.join(root, "test", "labels")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    if not paths:
        raise FileNotFoundError(f"no tifs under {img_dir}")
    rng = np.random.default_rng(seed)
    if n and n < len(paths):
        paths = sorted(rng.choice(paths, n, replace=False).tolist())
    dns, masks = [], []
    for p in paths:
        with rasterio.open(p) as f:
            dns.append(f.read().astype(np.float32))
        with rasterio.open(os.path.join(lab_dir, os.path.basename(p).replace("_image_", "_label_"))) as f:
            masks.append(np.clip(f.read(1).astype(np.int16), -1, 1))
    return paths, np.stack(dns), np.stack(masks)


def quicklook(dn: np.ndarray) -> np.ndarray:
    """(N,12,H,W) DN -> (N,3,H,W) float in [0,1], same stretch as hydroloc make_jpg.py."""
    return np.clip(dn[:, list(RGB)] / DIVISOR, 0.0, 1.0)


def build_models(ckpt_path: str, device):
    base = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=True, num_classes=0, img_size=256)
    cfg = timm.data.resolve_model_data_config(base)
    ours = timm.create_model("vit_large_patch16_dinov3.sat493m", pretrained=False, num_classes=0, img_size=256)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = ours.load_state_dict(ck["teacher_encoder"], strict=False)
    assert not unexpected and all(k.startswith("head") for k in missing), (missing, unexpected)
    models = {"dinov3-sat": base.eval().to(device), f"hydro-rgb@{ck['step']}": ours.eval().to(device)}
    return models, torch.tensor(cfg["mean"]).view(1, 3, 1, 1), torch.tensor(cfg["std"]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_tokens(model, x: torch.Tensor, res: int, bs: int = 16) -> np.ndarray:
    """x: (N,3,256,256) normalized -> (N, res/16, res/16, C) patch tokens."""
    out = []
    for i in range(0, len(x), bs):
        xb = x[i : i + bs]
        if res != xb.shape[-1]:
            xb = F.interpolate(xb, size=(res, res), mode="bicubic", align_corners=False)
        t = model.forward_features(xb)[:, model.num_prefix_tokens :]
        g = res // PATCH
        out.append(t.reshape(len(xb), g, g, -1).float().cpu())
    return torch.cat(out).numpy()


def pool_labels(masks: np.ndarray, g: int) -> np.ndarray:
    """(N,H,W) in {-1,0,1} -> (N,g,g) majority vote of valid pixels; -1 where no valid pixel."""
    n, h, _ = masks.shape
    k = h // g
    m = masks.reshape(n, g, k, g, k).transpose(0, 1, 3, 2, 4).reshape(n, g, g, k * k)
    water = (m == 1).sum(-1)
    valid = (m >= 0).sum(-1)
    out = np.where(water * 2 > valid, 1, 0)
    return np.where(valid == 0, -1, out)


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


def kmeans_segment(tokens: np.ndarray, gt_grid: np.ndarray, seed: int = 0):
    n, g, _, c = tokens.shape
    feats = tokens.reshape(-1, c)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    labels = KMeans(n_clusters=2, n_init=4, random_state=seed).fit_predict(feats).reshape(n, g, g)
    a, b = iou_table(labels, gt_grid), iou_table(1 - labels, gt_grid)
    return (labels, a) if a["miou"] >= b["miou"] else (1 - labels, b)


def pca_rgb(tokens: np.ndarray, seed: int = 0) -> np.ndarray:
    n, g, _, c = tokens.shape
    z = PCA(n_components=3, random_state=seed).fit_transform(tokens.reshape(-1, c))
    lo, hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
    z = np.clip((z - lo) / (hi - lo + 1e-8), 0, 1)
    return z.reshape(n, g, g, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swed-root", default="benchmark-datasets/SWED")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="output/zero_shot_swed")
    ap.add_argument("--n", type=int, default=0, help="subsample test chips (0 = all)")
    ap.add_argument("--resolutions", type=int, nargs="+", default=(256, 512))
    ap.add_argument("--n-show", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    paths, dn, masks = load_swed_test(args.swed_root, args.n)
    print(f"SWED test chips: {len(paths)}  shape={dn.shape}  water frac={(masks == 1).mean():.2f}  ignore={(masks < 0).mean():.3f}")
    rgb = quicklook(dn)
    models, mean, std = build_models(args.ckpt, args.device)
    x = ((torch.from_numpy(rgb) - mean) / std).to(args.device)

    results, segs, pcas = {}, {}, {}
    for res in args.resolutions:
        g = res // PATCH
        gt = pool_labels(masks, g)
        mndwi = (dn[:, G] - dn[:, SWIR1]) / (dn[:, G] + dn[:, SWIR1] + 1e-6)
        mndwi_pred = pool_labels(np.where(masks < 0, -1, (mndwi > 0).astype(np.int16)), g)
        results[f"mndwi>0@{res}"] = iou_table(mndwi_pred, gt)
        for name, model in models.items():
            tok = patch_tokens(model, x, res)
            seg, m = kmeans_segment(tok, gt)
            results[f"{name}@{res}"] = m
            segs[(name, res)] = seg
            pcas[(name, res)] = pca_rgb(tok)
            print(f"{name:>22} @{res}: mIoU={m['miou']:.3f} IoU_water={m['iou_water']:.3f} acc={m['acc']:.3f}", flush=True)
        print(f"{'mndwi>0':>22} @{res}: mIoU={results[f'mndwi>0@{res}']['miou']:.3f}")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # figure: rows = chips, cols = RGB | GT | per (model,res): PCA, k-means seg
    idx = np.linspace(0, len(paths) - 1, min(args.n_show, len(paths))).astype(int)
    keys = [(n, r) for r in args.resolutions for n in models]
    ncol = 2 + 2 * len(keys)
    fig, axs = plt.subplots(len(idx), ncol, figsize=(1.6 * ncol, 1.6 * len(idx)))
    for r, i in enumerate(idx):
        axs[r, 0].imshow(rgb[i].transpose(1, 2, 0))
        axs[r, 1].imshow(np.ma.masked_less(masks[i], 0), vmin=0, vmax=1, cmap="Blues", interpolation="none")
        for j, (n, res) in enumerate(keys):
            axs[r, 2 + 2 * j].imshow(pcas[(n, res)][i], interpolation="nearest")
            axs[r, 3 + 2 * j].imshow(segs[(n, res)][i], vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    titles = ["RGB", "SWED label"] + [t for n, res in keys for t in (f"PCA {n}@{res}", f"k-means {n}@{res}")]
    for a, t in zip(axs[0], titles):
        a.set_title(t, fontsize=6)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig_path = os.path.join(args.out_dir, "swed_zero_shot.png")
    fig.savefig(fig_path, dpi=160)
    print(f"saved {fig_path} and metrics.json")


if __name__ == "__main__":
    main()
