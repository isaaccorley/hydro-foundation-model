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
if [ "$PRESET" = "rgb" ]; then SRC="$DATA/data_jpg"; else SRC="$DATA/data"; fi

# Stage the patch set to node-local scratch: the 12-band loader was NFS-bound at ~200 MB/s
# of small-file reads (16 workers idling at ~65%). 106k files; parallel cp keeps latency down.
STAGE="/tmp/${SLURM_JOB_ID}/$(basename "$SRC")"
trap 'rm -rf "/tmp/${SLURM_JOB_ID}"' EXIT
mkdir -p "$STAGE"
echo "staging $SRC -> $STAGE"; t0=$(date +%s)
if find "$SRC" -maxdepth 1 -type f -name 'patch_*' -print0 \
     | xargs -0 -P "$SLURM_CPUS_PER_TASK" -n 500 cp -t "$STAGE" \
   && [ "$(ls "$STAGE" | wc -l)" -eq "$(ls "$SRC" | wc -l)" ]; then
  ROOT="$STAGE"; echo "staged $(ls "$STAGE" | wc -l) files in $(( $(date +%s) - t0 ))s ($(du -sh "$STAGE" | cut -f1))"
else
  ROOT="$SRC"; echo "WARNING: staging failed or incomplete, falling back to NFS root $SRC"
fi

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
