#!/bin/bash
#SBATCH --job-name=zs_water
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: DATASET=s1s2water CKPT=output/dino/landsat6/landsat6_final.pt sbatch -J zs_s1s2_landsat6 scripts/zero_shot_water.sbatch
# Sized from the SWED probe (98 chips, 3 encoders, occlusion: ~17 min, ~9.5 cores, 12.4 GB).
# Sen1Floods11 test = 360 chips, S1S2-Water test ~200 chips -> allow up to ~4x the SWED time.
set -euo pipefail
: "${DATASET:?set DATASET=swed|s1s2water|sen1floods11}"
: "${CKPT:?set CKPT=path/to/teacher checkpoint}"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
/sw/pkgs/anaconda/anaconda3/bin/python3 scripts/zero_shot_water.py --dataset "$DATASET" --ckpt "$CKPT" --device cpu ${EXTRA_ARGS:-}
