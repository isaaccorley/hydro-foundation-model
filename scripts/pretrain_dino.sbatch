#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --partition=gpu,gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: PRESET=rgb|landsat6|all sbatch -J dino_$PRESET scripts/pretrain_dino.sbatch
# Prefer scripts/submit_dino_chain.sh, which passes measured per-preset --mem/--cpus-per-task.
# One GPU per run; CPUs feed the multi-crop DataLoader (10 crops/image). Measured: rgb 2.7
# cores / 32G, landsat6 5.4 cores / ~50G summed RSS (sacct TotalCPU/Elapsed, MaxRSS).

set -euo pipefail
: "${PRESET:?set PRESET=rgb|landsat6|all}"
DATA=/u/isaaccorley/github/hydroloc
if [ "$PRESET" = "rgb" ]; then ROOT="$DATA/data_jpg"; else ROOT="$DATA/data"; fi

cd "$SLURM_SUBMIT_DIR"
PY=/sw/pkgs/anaconda/anaconda3/bin/python3
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

$PY -m src.pretrain.train \
  --bands "$PRESET" --root "$ROOT" --variant sat493m \
  --steps 15000 --batch-size 32 --accum-steps 8 \
  --freeze-backbone-steps 6000 --backbone-warmup-steps 1000 --warmup-steps 2000 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir "output/dino/$PRESET" \
  --resume auto
