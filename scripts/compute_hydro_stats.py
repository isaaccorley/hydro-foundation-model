"""Per-band normalization stats for the Hydro dataset, with BOA offset harmonization.

Sentinel-2 L2A products from processing baseline >= 04.00 (Jan 2022) store
DN = reflectance * 10000 + 1000; earlier products store DN = reflectance * 10000. The Hydro
tifs are a mix of both, so the offset is detected per patch (from the lowest per-band 1st
percentile of non-zero pixels) and removed before accumulating statistics. Zero pixels are
treated as nodata and excluded.

Outputs into --out-dir:
  hydro_stats.json    mean/std (plain and robust) per band, offset-corrected and raw, plus counts
  hydro_manifest.csv  per patch: detected offset, low percentile, water fraction, nodata fraction

Usage:
  python scripts/compute_hydro_stats.py --root /path/to/data --out-dir output/hydro_stats --workers 32
"""

import argparse
import csv
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import rasterio

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
G, SWIR1 = 2, 10
OFFSET = 1000.0
OFFSET_THRESHOLD = 850.0  # min over bands of p1(non-zero DN); no-offset patches sit at ~0-500
QUANT = 10000.0

HIST_LO, HIST_HI, HIST_BIN = -1000, 16000, 5
HIST_EDGES = np.arange(HIST_LO, HIST_HI + HIST_BIN, HIST_BIN, dtype=np.float64)
N_BINS = len(HIST_EDGES) - 1
ACC_KEYS = ("count", "sum_c", "sumsq_c", "sum_r", "sumsq_r", "hist_c", "hist_r")


def patch_stats(path: str) -> dict:
    with rasterio.open(path) as f:
        x = f.read().astype(np.float64)  # (12, H, W)
    valid = x != 0

    p1 = np.array([np.percentile(x[b][valid[b]], 1) if valid[b].any() else np.nan for b in range(12)])
    lo = np.nanmin(p1)
    offset = OFFSET if lo >= OFFSET_THRESHOLD else 0.0
    xc = x - offset

    out = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "offset": offset,
        "lo_p1": float(lo),
        "nodata_frac": float(1.0 - valid.all(axis=0).mean()),
        "vis_max": float(xc[:4].max()),
        "count": valid.sum(axis=(1, 2)).astype(np.int64),
        "sum_c": np.where(valid, xc, 0).sum(axis=(1, 2)),
        "sumsq_c": np.where(valid, xc**2, 0).sum(axis=(1, 2)),
        "sum_r": np.where(valid, x, 0).sum(axis=(1, 2)),
        "sumsq_r": np.where(valid, x**2, 0).sum(axis=(1, 2)),
        "hist_c": np.stack([np.histogram(xc[b][valid[b]], bins=HIST_EDGES)[0] for b in range(12)]),
        "hist_r": np.stack([np.histogram(x[b][valid[b]], bins=HIST_EDGES)[0] for b in range(12)]),
    }
    both = valid[G] & valid[SWIR1]
    g, s = xc[G][both], xc[SWIR1][both]
    out["water_frac"] = float(((g - s) > 0).mean()) if both.any() else float("nan")
    return out


def hist_percentile(hist: np.ndarray, q: float) -> float:
    cdf = np.cumsum(hist)
    idx = np.searchsorted(cdf, q / 100.0 * cdf[-1])
    return float(HIST_EDGES[min(idx, N_BINS - 1)] + HIST_BIN / 2)


def robust_from_hist(hist: np.ndarray) -> dict:
    """Median, MAD-sigma, p1/p99, and mean/std after clipping to [p0.5, p99.5]."""
    centers = HIST_EDGES[:-1] + HIST_BIN / 2
    median = hist_percentile(hist, 50)
    dev = np.abs(centers - median)
    order = np.argsort(dev)
    cdf = np.cumsum(hist[order])
    mad = float(dev[order][np.searchsorted(cdf, 0.5 * cdf[-1])])
    lo, hi = hist_percentile(hist, 0.5), hist_percentile(hist, 99.5)
    keep = (centers >= lo) & (centers <= hi)
    w = hist[keep].astype(np.float64)
    c = centers[keep]
    mean = float((w * c).sum() / w.sum())
    std = float(np.sqrt((w * (c - mean) ** 2).sum() / w.sum()))
    return {
        "median": median,
        "mad_sigma": 1.4826 * mad,
        "p1": hist_percentile(hist, 1),
        "p99": hist_percentile(hist, 99),
        "clipped_mean": mean,
        "clipped_std": std,
    }


def reduce_stats(acc: dict, manifest: list[dict]) -> dict:
    count = acc["count"]
    stats = {}
    for tag in ("c", "r"):
        mean = acc[f"sum_{tag}"] / count
        std = np.sqrt(np.maximum(acc[f"sumsq_{tag}"] / count - mean**2, 0))
        hist = acc[f"hist_{tag}"]
        per_band = {}
        for b, name in enumerate(BANDS):
            per_band[name] = {"mean": float(mean[b]), "std": float(std[b]), **robust_from_hist(hist[b])}
        stats["offset_corrected" if tag == "c" else "raw"] = {
            "dn": per_band,
            "mean_dn": mean.tolist(),
            "std_dn": std.tolist(),
            "mean_reflectance": (mean / QUANT).tolist(),
            "std_reflectance": (std / QUANT).tolist(),
            "median_dn": [per_band[n]["median"] for n in BANDS],
            "mad_sigma_dn": [per_band[n]["mad_sigma"] for n in BANDS],
            "clipped_mean_dn": [per_band[n]["clipped_mean"] for n in BANDS],
            "clipped_std_dn": [per_band[n]["clipped_std"] for n in BANDS],
        }
    offsets = np.array([m["offset"] for m in manifest])
    los = np.array([m["lo_p1"] for m in manifest])
    stats["meta"] = {
        "n_patches": len(manifest),
        "valid_pixels_per_band": count.tolist(),
        "bands": BANDS,
        "quantification": QUANT,
        "boa_offset": OFFSET,
        "offset_threshold_p1": OFFSET_THRESHOLD,
        "frac_patches_with_offset": float((offsets > 0).mean()),
        "frac_patches_ambiguous_500_850": float(((los >= 500) & (los < OFFSET_THRESHOLD)).mean()),
        "frac_patches_any_nodata": float(np.mean([m["nodata_frac"] > 0 for m in manifest])),
        "water_frac_median": float(np.nanmedian([m["water_frac"] for m in manifest])),
        "convention": "reflectance = (DN - offset) / 10000; normalize with mean_reflectance/std_reflectance",
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default="output/hydro_stats")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.root, "patch_*.tif")))
    if args.limit:
        paths = paths[: args.limit]
    print(f"n={len(paths)} workers={args.workers}", flush=True)

    # Reduce as results stream in; per-patch histograms are ~650 KB each, so never keep them all.
    acc = {k: 0 for k in ACC_KEYS}
    manifest = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(patch_stats, paths, chunksize=64), 1):
            for k in ACC_KEYS:
                acc[k] = acc[k] + r[k]
            manifest.append({k: r[k] for k in ("name", "offset", "lo_p1", "water_frac", "nodata_frac", "vis_max")})
            if i % 10000 == 0:
                print(f"{i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)

    stats = reduce_stats(acc, manifest)
    with open(os.path.join(args.out_dir, "hydro_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out_dir, "hydro_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "offset", "lo_p1", "water_frac", "nodata_frac", "vis_max"])
        for m in sorted(manifest, key=lambda m: m["name"]):
            w.writerow([m["name"], int(m["offset"]), f"{m['lo_p1']:.0f}", f"{m['water_frac']:.4f}",
                        f"{m['nodata_frac']:.5f}", f"{m['vis_max']:.0f}"])

    m = stats["meta"]
    print(f"\ndone in {time.time() - t0:.0f}s | offset patches {m['frac_patches_with_offset']:.1%} | "
          f"ambiguous {m['frac_patches_ambiguous_500_850']:.2%} | nodata patches {m['frac_patches_any_nodata']:.1%} | "
          f"median water frac {m['water_frac_median']:.2f}")
    c, r = stats["offset_corrected"], stats["raw"]
    print(f"\n{'band':5} {'mean_c':>8} {'std_c':>8} {'median':>8} {'madsig':>8} {'clipmean':>9} {'clipstd':>8} | {'mean_raw':>8} {'std_raw':>8}   (DN)")
    for b, name in enumerate(BANDS):
        print(f"{name:5} {c['mean_dn'][b]:8.1f} {c['std_dn'][b]:8.1f} {c['median_dn'][b]:8.1f} {c['mad_sigma_dn'][b]:8.1f} "
              f"{c['clipped_mean_dn'][b]:9.1f} {c['clipped_std_dn'][b]:8.1f} | {r['mean_dn'][b]:8.1f} {r['std_dn'][b]:8.1f}")


if __name__ == "__main__":
    main()
