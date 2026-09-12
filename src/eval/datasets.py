"""Benchmark loaders that emit 12-band DN stacks (tif order, 0 = nodata) + targets, lean deps only.

Each loader returns dict(dn=(N,12,H,W) float32 DN, target=(N,H,W), valid=(N,H,W) bool,
available=set of tif band indices the dataset actually provides, meta=list).
"""

import glob
import os

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def _place(bands_hw: np.ndarray, tif_idx: tuple[int, ...]) -> np.ndarray:
    """(k,H,W) -> (12,H,W) with the k bands at tif positions tif_idx, zeros elsewhere."""
    out = np.zeros((12, *bands_hw.shape[1:]), np.float32)
    for j, b in enumerate(tif_idx):
        out[b] = bands_hw[j]
    return out


def _tile(img: np.ndarray, mask: np.ndarray, chip: int):
    """Non-overlapping chip x chip windows; edge remainders dropped."""
    _, h, w = img.shape
    for r in range(0, h - chip + 1, chip):
        for c in range(0, w - chip + 1, chip):
            yield img[:, r : r + chip, c : c + chip], mask[r : r + chip, c : c + chip]


def _read_split(path: str, key: str) -> list[str]:
    """MagicBathyNet split txt: lines 'train sample:' / 'test sample:' followed by a python-list literal."""
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(f"{key} sample"):
            return [s.strip().strip("'\"[] ") for s in lines[i + 1].split(",")]
    raise ValueError(f"no '{key} sample' in {path}")


# MagicBathyNet S2 chips are 3-band 10 m visible. File band order verified empirically on the
# 82 depth chips: water-pixel means 426/363/93 DN for bands 1/2/3 -> blue, green, red.
MBN_S2_TIF_IDX = (1, 2, 3)  # file band 1,2,3 = B02, B03, B04  (B, G, R)


def load_mbn(root: str, site: str, split: str):
    """MagicBathyNet (Agrafiotis et al. 2024) Sentinel-2 bathymetry: 18x18 px RGB chips (uint16 L2A DN),
    depth tif in metres (negative = below surface, 0 = land/nodata). Official s2_split_bathymetry split."""
    site_dir = os.path.join(root, site)
    ids = _read_split(os.path.join(site_dir, "s2_split_bathymetry.txt"), split)
    dns, tgts, valids, meta = [], [], [], []
    for i in ids:
        with rasterio.open(os.path.join(site_dir, "img", "s2", f"img_{i}.tif")) as f:
            img = f.read().astype(np.float32)
        with rasterio.open(os.path.join(site_dir, "depth", "s2", f"depth_{i}.tif")) as f:
            d = f.read(1).astype(np.float32)
        dns.append(_place(img, MBN_S2_TIF_IDX)); tgts.append(-d); valids.append(d != 0); meta.append(i)
    return dict(dn=np.stack(dns), target=np.stack(tgts), valid=np.stack(valids), available=set(MBN_S2_TIF_IDX), meta=meta)


def load_esw(root: str, split: str, chip: int = 256):
    """Earth Surface Water (Luo et al. 2021): S2 L2A 6-band [B02,B03,B04,B08,B11,B12] scenes + manual water masks.
    train = tra_*, test = val_* (the dataset's own hold-out)."""
    d = {"train": "tra", "test": "val"}[split]
    imgs = sorted(glob.glob(os.path.join(root, f"{d}_scene", "*.tif")))
    dns, tgts, meta = [], [], []
    for p in imgs:
        stem = os.path.basename(p)[:-4].split("_")
        mask_p = os.path.join(root, f"{d}_truth", "_".join(stem[:-2] + [stem[-1], "Truth"]) + ".tif")
        with rasterio.open(p) as f:
            img = f.read().astype(np.float32)
        with rasterio.open(mask_p) as f:
            m = f.read(1)
        for ci, cm in _tile(img, m, chip):
            dns.append(_place(ci, (1, 2, 3, 7, 10, 11))); tgts.append(cm.astype(np.int64)); meta.append(os.path.basename(p))
    dn = np.stack(dns); tgt = np.stack(tgts)
    return dict(dn=dn, target=tgt, valid=np.ones_like(tgt, bool), available={1, 2, 3, 7, 10, 11}, meta=meta)
