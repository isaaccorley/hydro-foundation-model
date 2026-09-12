#!/bin/bash
#SBATCH --job-name=linprobe
#SBATCH --partition=gpu,gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --exclude=rails11
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Frozen-encoder linear probes on ESW (water seg) and MagicBathyNet (S2 bathymetry) vs. baselines.
# Unmeasured first run; re-size from `sacct -j <id> --format=Elapsed,TotalCPU,MaxRSS`.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/sw/pkgs/anaconda/anaconda3/bin/python3
PRITHVI=/projects/bgtj/isaaccorley/benchmark-datasets/prithvi_eo2_300m  # config.json, prithvi_mae.py, Prithvi_EO_V2_300M.pt from HF ibm-nasa-geospatial/Prithvi-EO-2.0-300M
BB="dinov3-sat inflated-landsat6 inflated-all hydro-rgb:output/dino/rgb/rgb_final.pt hydro-landsat6:output/dino/landsat6/landsat6_final.pt hydro-all:output/dino/all/all_final.pt olmoearth prithvi swinv2-legacy"

$PY -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
$PY scripts/linear_probe.py --dataset esw --root benchmark-datasets/earth_surface_water/dset-s2 \
  --backbones $BB --prithvi-dir "$PRITHVI" --res 256 512 --out output/linear_probe/esw.json
$PY scripts/linear_probe.py --dataset mbn --root benchmark-datasets/magicbathynet/MagicBathyNet --site agia_napa \
  --backbones $BB --prithvi-dir "$PRITHVI" --res 256 --out output/linear_probe/mbn_agia_napa.json
if [ -f benchmark-datasets/magicbathynet/MagicBathyNet/puck_lagoon/s2_split_bathymetry.txt ]; then
  $PY scripts/linear_probe.py --dataset mbn --root benchmark-datasets/magicbathynet/MagicBathyNet --site puck_lagoon \
    --backbones $BB --prithvi-dir "$PRITHVI" --res 256 --out output/linear_probe/mbn_puck_lagoon.json \
    || echo "WARNING: puck_lagoon probe failed (extraction incomplete?)"
else
  echo "puck_lagoon split file not present yet; skipped"
fi
