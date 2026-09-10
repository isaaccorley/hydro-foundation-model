#!/bin/bash
#SBATCH --job-name=swed_zeroshot
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: CKPT=output/dino/rgb/rgb_step5000.pt sbatch scripts/zero_shot_swed.sbatch
# CPU-only: ViT-L forward over ~100 SWED test chips x 2 encoders x {256,512}.
# Unmeasured first run; re-size from `sacct -j <id> --format=Elapsed,TotalCPU,MaxRSS`.
set -euo pipefail
: "${CKPT:?set CKPT=path/to/teacher checkpoint}"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
/sw/pkgs/anaconda/anaconda3/bin/python3 scripts/zero_shot_swed.py \
  --swed-root benchmark-datasets/SWED --ckpt "$CKPT" \
  --out-dir "output/zero_shot_swed/$(basename "${CKPT%.pt}")" --device cpu
