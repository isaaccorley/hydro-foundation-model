#!/bin/bash
#SBATCH --job-name=swed_zeroshot
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: CKPT=output/dino/rgb/rgb_step5000.pt sbatch scripts/zero_shot_swed.sbatch
# CPU-only: ViT-L forward over 98 SWED test chips x 2 encoders x {256,512}.
# Measured (job 174019): Elapsed 5:07, TotalCPU 48:33 (~9.5 cores busy), MaxRSS 10.7 GB.
set -euo pipefail
: "${CKPT:?set CKPT=path/to/teacher checkpoint}"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
/sw/pkgs/anaconda/anaconda3/bin/python3 scripts/zero_shot_swed.py \
  --swed-root benchmark-datasets/SWED --ckpt "$CKPT" \
  --out-dir "output/zero_shot_swed/$(basename "${CKPT%.pt}")" --device cpu
