"""Frozen-encoder linear probes on independently-labelled water benchmarks, vs. spectral baselines.

Protocol: encoder -> patch tokens at --res -> per-token head fit on TRAIN only -> scored on TEST.
  segmentation (ESW):  logistic regression on standardized tokens; token targets = majority vote;
                       metrics = token-grid mIoU and pixel mIoU after bilinear upsampling of P(water)
  regression (MagicBathyNet): ridge on standardized tokens; token target = mean valid depth;
                       metrics = pixel RMSE/MAE (m) on valid pixels after bilinear upsampling
Spectral baselines (no encoder, per pixel): ridge / logistic on offset-corrected reflectance of the
available bands; MNDWI > 0 (water); Stumpf log-ratio ln(B02)/ln(B03) linear fit (depth).

Usage:
  python scripts/linear_probe.py --dataset esw --root benchmark-datasets/earth_surface_water/dset-s2 \
      --backbones dinov3-sat inflated-landsat6 hydro-landsat6:output/dino/landsat6/landsat6_final.pt ...
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.backbones import build_backbones  # noqa: E402
from src.eval.datasets import load_esw, load_mbn  # noqa: E402
from src.pretrain.stats import QUANTIFICATION, detect_boa_offset  # noqa: E402

G, SWIR1, BLUE, GREEN = 2, 10, 1, 2


# ----------------------------------------------------------------------------- targets / metrics
def pool_tokens(target: np.ndarray, valid: np.ndarray, g: int, kind: str):
    """(N,H,W) -> (N,g,g) token targets (+ token validity) by area pooling."""
    n, h, w = target.shape
    t = torch.from_numpy(target.astype(np.float32))[:, None]
    v = torch.from_numpy(valid.astype(np.float32))[:, None]
    tv = F.adaptive_avg_pool2d(t * v, g)[:, 0].numpy()
    vv = F.adaptive_avg_pool2d(v, g)[:, 0].numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tv / vv
    tok_valid = vv >= 0.5
    if kind == "seg":
        return (mean >= 0.5).astype(np.int64), tok_valid
    return mean.astype(np.float32), tok_valid


def upsample(tok: np.ndarray, h: int, w: int) -> np.ndarray:
    """(N,g,g) -> (N,H,W) bilinear."""
    t = torch.from_numpy(tok.astype(np.float32))[:, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[:, 0].numpy()


def seg_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p, t = pred[valid], gt[valid]
    ious = []
    for c in (0, 1):
        inter = np.logical_and(p == c, t == c).sum(); union = np.logical_or(p == c, t == c).sum()
        ious.append(inter / max(union, 1))
    return {"miou": float(np.mean(ious)), "iou_water": float(ious[1]), "acc": float((p == t).mean())}


def reg_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    e = pred[valid] - gt[valid]
    return {"rmse_m": float(np.sqrt(np.mean(e**2))), "mae_m": float(np.mean(np.abs(e))), "n_px": int(valid.sum())}


# ----------------------------------------------------------------------------- heads
def fit_head(kind: str, xtr: np.ndarray, ytr: np.ndarray, seed: int = 0):
    sc = StandardScaler().fit(xtr)
    if kind == "seg":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed).fit(sc.transform(xtr), ytr)
        return lambda x: clf.predict_proba(sc.transform(x))[:, 1]
    reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(xtr), ytr)
    return lambda x: reg.predict(sc.transform(x))


def probe_backbone(bb, kind, train, test, res, device):
    g = bb.grid_at_256 * res // 256
    t0 = time.time()
    ftr = bb.encode(train["dn"], train["available"], res=res, device=device)
    fte = bb.encode(test["dn"], test["available"], res=res, device=device)
    g = ftr.shape[1]
    ytr, vtr = pool_tokens(train["target"], train["valid"], g, kind)
    yte, vte = pool_tokens(test["target"], test["valid"], g, kind)
    head = fit_head(kind, ftr.reshape(-1, ftr.shape[-1])[vtr.ravel()], ytr.ravel()[vtr.ravel()])
    pte = head(fte.reshape(-1, fte.shape[-1])).reshape(fte.shape[:3])
    out = {"grid": int(g), "n_train_tokens": int(vtr.sum()), "seconds": round(time.time() - t0, 1)}
    h, w = test["target"].shape[1:]
    if kind == "seg":
        out.update({"token_" + k: v for k, v in seg_metrics((pte >= 0.5).astype(int), yte, vte).items()})
        out.update({"pixel_" + k: v for k, v in seg_metrics((upsample(pte, h, w) >= 0.5).astype(int), test["target"], test["valid"]).items()})
    else:
        out.update({"token_" + k: v for k, v in reg_metrics(pte, yte, vte).items()})
        out.update({"pixel_" + k: v for k, v in reg_metrics(upsample(pte, h, w), test["target"], test["valid"]).items()})
    return out


# ----------------------------------------------------------------------------- spectral baselines
def reflectance(dn12: np.ndarray, available: set[int]) -> np.ndarray:
    """(N,12,H,W) DN -> (N,k,H,W) offset-corrected reflectance for the available bands (sorted)."""
    idx = sorted(available)
    out = np.zeros((dn12.shape[0], len(idx), *dn12.shape[2:]), np.float32)
    for n in range(len(dn12)):
        off = detect_boa_offset(dn12[n, idx])
        out[n] = np.clip((dn12[n, idx] - off) / QUANTIFICATION, 1e-4, None)
    return out


def spectral_baselines(kind, train, test):
    res = {}
    rtr, rte = reflectance(train["dn"], train["available"]), reflectance(test["dn"], test["available"])
    idx = sorted(train["available"])
    vtr, vte = train["valid"], test["valid"]
    xtr = rtr.transpose(0, 2, 3, 1)[vtr]; ytr = train["target"][vtr]
    xte = rte.transpose(0, 2, 3, 1).reshape(-1, len(idx))
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xtr), min(len(xtr), 300_000), replace=False)
    head = fit_head(kind, xtr[sub], ytr[sub])
    pte = head(xte).reshape(test["target"].shape)
    if kind == "seg":
        res["per-pixel logistic (bands)"] = {"pixel_" + k: v for k, v in seg_metrics((pte >= 0.5).astype(int), test["target"], vte).items()}
        if G in train["available"] and SWIR1 in train["available"]:
            g_, s_ = rte[:, idx.index(G)], rte[:, idx.index(SWIR1)]
            mndwi = (g_ - s_) / (g_ + s_ + 1e-6)
            res["MNDWI > 0"] = {"pixel_" + k: v for k, v in seg_metrics((mndwi > 0).astype(int), test["target"], vte).items()}
    else:
        res["per-pixel ridge (bands)"] = {"pixel_" + k: v for k, v in reg_metrics(pte, test["target"], vte).items()}
        if BLUE in train["available"] and GREEN in train["available"]:
            def stumpf(r):
                return (np.log(1000 * r[:, idx.index(BLUE)]) / np.log(1000 * r[:, idx.index(GREEN)]))[..., None]
            head = fit_head("reg", stumpf(rtr)[vtr], train["target"][vtr])
            p = head(stumpf(rte).reshape(-1, 1)).reshape(test["target"].shape)
            res["Stumpf log-ratio B02/B03 (linear)"] = {"pixel_" + k: v for k, v in reg_metrics(p, test["target"], vte).items()}
        mean_pred = np.full(test["target"].shape, train["target"][vtr].mean(), np.float32)
        res["predict train-mean depth"] = {"pixel_" + k: v for k, v in reg_metrics(mean_pred, test["target"], vte).items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("esw", "mbn"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--site", default="agia_napa", help="mbn only")
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--prithvi-dir", default=None)
    ap.add_argument("--res", type=int, nargs="+", default=(256,))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap chips per split (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if args.dataset == "esw":
        kind, train, test = "seg", load_esw(args.root, "train"), load_esw(args.root, "test")
    else:
        kind, train, test = "reg", load_mbn(args.root, args.site, "train"), load_mbn(args.root, args.site, "test")
    if args.limit:
        for d in (train, test):
            for k in ("dn", "target", "valid", "meta"):
                d[k] = d[k][: args.limit]
    print(f"{args.dataset}: train {train['dn'].shape} test {test['dn'].shape} available bands {sorted(train['available'])}", flush=True)

    results = {"dataset": args.dataset, "site": args.site, "n_train": int(len(train["dn"])), "n_test": int(len(test["dn"])),
               "chip": int(train["dn"].shape[-1]), "available_bands": sorted(train["available"]), "baselines": {}, "backbones": {}}
    results["baselines"] = spectral_baselines(kind, train, test)
    for k, v in results["baselines"].items():
        print(f"{k:>40}: {v}", flush=True)

    for name in args.backbones:
        try:
            bb = build_backbones([name], args.prithvi_dir)[0]
        except Exception as e:  # a missing/broken loader must not kill the other backbones
            results["backbones"][name] = {"error": f"load failed: {type(e).__name__}: {str(e)[:200]}"}
            print(f"{name:>40}: LOAD FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        for res in args.res:
            try:
                r = probe_backbone(bb, kind, train, test, res, args.device)
            except Exception as e:  # keep going; report the failure
                r = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            results["backbones"][f"{bb.name}@{res}"] = r
            print(f"{bb.name + '@' + str(res):>40}: {r}", flush=True)
        bb.model.to("cpu"); torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
