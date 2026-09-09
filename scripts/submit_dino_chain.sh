#!/bin/bash
# Submit the three DINO runs to execute sequentially: rgb -> landsat6 -> all.
# `afterany` so a failure in one run does not hold the next one hostage.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

prev=""
for preset in rgb landsat6 all; do
  dep=""
  [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  jid=$(PRESET=$preset sbatch --parsable $dep -J "dino_$preset" scripts/pretrain_dino.sbatch)
  echo "$preset -> job $jid ${dep:+(after $prev)}"
  prev=$jid
done
