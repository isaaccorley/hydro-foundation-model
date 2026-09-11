#!/bin/bash
# Submit the three DINO runs to execute sequentially: rgb -> landsat6 -> all.
# `afterany` so a failure in one run does not hold the next one hostage.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# From measured runs (sacct TotalCPU/Elapsed, MaxRSS + headroom):
#   rgb      2.7 cores busy, 32G RSS
#   landsat6 5.4 cores busy, ~50G RSS summed over loader workers
#   all      ~10 cores busy but NFS-bound (16 workers at ~65%, 12-band tifs, ~200 MB/s of
#            small reads); more CPUs won't help — staging the tifs to node-local /tmp would.
declare -A MEM=([rgb]=40G [landsat6]=56G [all]=64G)
declare -A CPUS=([rgb]=6 [landsat6]=8 [all]=16)

# EXCLUDE=rails11 ./scripts/submit_dino_chain.sh  -> skip a node that showed unexplained
# step stalls (GPU idle, all our threads asleep) while the same pipeline ran clean elsewhere.
excl=""
[ -n "${EXCLUDE:-}" ] && excl="--exclude=$EXCLUDE"

prev=""
for preset in rgb landsat6 all; do
  dep=""
  [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  jid=$(PRESET=$preset sbatch --parsable $dep $excl -J "dino_$preset" --mem="${MEM[$preset]}" --cpus-per-task="${CPUS[$preset]}" scripts/pretrain_dino.sbatch)
  echo "$preset -> job $jid ${dep:+(after $prev)}"
  prev=$jid
done
