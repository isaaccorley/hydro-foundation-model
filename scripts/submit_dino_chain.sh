#!/bin/bash
# Submit the three DINO runs to execute sequentially: rgb -> landsat6 -> all.
# `afterany` so a failure in one run does not hold the next one hostage.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# From measured runs (sacct TotalCPU/Elapsed, MaxRSS + headroom):
#   rgb      2.7 cores busy, 32G RSS
#   landsat6 5.4 cores busy, ~50G RSS summed over loader workers
#   all      reads 2x the bands of landsat6 (unmeasured at time of writing)
declare -A MEM=([rgb]=40G [landsat6]=56G [all]=64G)
declare -A CPUS=([rgb]=6 [landsat6]=8 [all]=12)

prev=""
for preset in rgb landsat6 all; do
  dep=""
  [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  jid=$(PRESET=$preset sbatch --parsable $dep -J "dino_$preset" --mem="${MEM[$preset]}" --cpus-per-task="${CPUS[$preset]}" scripts/pretrain_dino.sbatch)
  echo "$preset -> job $jid ${dep:+(after $prev)}"
  prev=$jid
done
