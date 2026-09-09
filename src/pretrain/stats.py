"""Per-band normalization constants for the Hydro pretraining set (106,135 patches).

Convention: reflectance = (DN - offset) / 10000, then per-band z-score. Sentinel-2 L2A patches
from processing baseline >= 04.00 carry a +1000 DN BOA offset; older ones don't. The Hydro set
is a mix (56.8% with offset), so the offset is detected per patch from the lowest
per-band 1st percentile of non-zero pixels (see `detect_boa_offset`). Zero pixels are nodata.

Band order (tif): [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12].

Two statistic sets, both offset-corrected, reflectance units, from output/hydro_stats/hydro_stats.json
(scripts/compute_hydro_stats.py, SLURM job 172965):
  MEAN / STD        plain moments. Dominated by a bright minority mode (snow/ice/cloud/glint):
                    water (the ~85% mode) spans only ~0.2 z-units under these.
  MEDIAN / MAD_STD  median and 1.4826*MAD. Water spans ~1.8 z-units; bright pixels land at z~20-35.
"""

import numpy as np
import torch

OFFSET_DN = 1000
OFFSET_THRESHOLD_DN = 850
QUANTIFICATION = 10000

MEAN = torch.tensor([
    0.143512, 0.138491, 0.124300, 0.119969, 0.118488, 0.114186,
    0.111900, 0.114037, 0.107815, 0.088481, 0.039673, 0.037321,
])
STD = torch.tensor([
    0.247166, 0.254039, 0.247700, 0.249921, 0.242846, 0.236904,
    0.229382, 0.241769, 0.220718, 0.192919, 0.056679, 0.051722,
])
MEDIAN = torch.tensor([
    0.054250, 0.043750, 0.030750, 0.024750, 0.025750, 0.023750,
    0.024250, 0.022250, 0.022750, 0.017250, 0.020750, 0.018750,
])
MAD_STD = torch.tensor([
    0.036324, 0.032617, 0.031135, 0.029652, 0.031876, 0.030393,
    0.030393, 0.028169, 0.028911, 0.025204, 0.025204, 0.022239,
])


def detect_boa_offset(dn: np.ndarray) -> float:
    """dn: (C, H, W) raw DN. Returns 1000.0 if the patch carries the BOA offset, else 0.0.

    Same rule as scripts/compute_hydro_stats.py so loader decisions match the manifest.
    """
    lows = [np.percentile(band[band != 0], 1) for band in dn if (band != 0).any()]
    if not lows:
        return 0.0
    return float(OFFSET_DN) if min(lows) >= OFFSET_THRESHOLD_DN else 0.0
