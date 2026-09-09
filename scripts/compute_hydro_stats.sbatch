#!/bin/bash
#SBATCH --job-name=hydro_stats
#SBATCH --partition=cpu_amd,cpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Measured (job 172965, 106k patches): Elapsed 4:58, TotalCPU 2:20:17 (~28 cores busy),
# MaxRSS 16.7 GB. 24G = MaxRSS + ~50% headroom.

cd "$SLURM_SUBMIT_DIR"
/sw/pkgs/anaconda/anaconda3/bin/python3 scripts/compute_hydro_stats.py \
  --root /u/isaaccorley/github/hydroloc/data \
  --out-dir output/hydro_stats \
  --workers "$SLURM_CPUS_PER_TASK"
