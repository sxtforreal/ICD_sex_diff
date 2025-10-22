#!/usr/bin/env bash
#SBATCH --job-name=scmr
#SBATCH --output=logs/scmr_%j.out
#SBATCH --error=logs/scmr_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general

# Adjust modules/env as needed
# module load anaconda/2024.06
# conda activate yourenv

set -euo pipefail
mkdir -p logs results

# Example environment overrides
export SCMR_COX_CORR_THRESHOLD=0.95
export SCMR_COX_PEN=0.1
export SCMR_COX_L1_RATIO=0.2
export SCMR_FS_THRESHOLD=0.3

# Point to data on the cluster
# export ICD_DATA_BASE=/path/to/data
# export NICM_XLSX=/path/to/NICM.xlsx

# Run heldout pipeline (default). You can switch via --mode legacy|heldout|full
python SCMR.py --mode heldout --results_dir results --n_splits 50 --heldout_test_size 0.3 --heldout_seed 42
