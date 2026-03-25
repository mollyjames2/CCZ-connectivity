#!/bin/bash
#SBATCH --job-name=ccz_parcels
#SBATCH --account=n01-SMARTEX
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Array over time periods (0-indexed): 0=Jan2019, 1=Jul2019, 2=Jan2023, 3=Jul2023
#SBATCH --array=0-3
#SBATCH --output=logs/parcels_%A_%a.out
#SBATCH --error=logs/parcels_%A_%a.err

# ── CCZ Particle Tracking — SLURM array job ───────────────────────────────────
# Submits one job per time period.  Each job runs both scenarios (unmined
# and mined) sequentially.  Adjust --time and --mem for production runs.
#
# Usage (from repo root):
#   sbatch workflow/01_run_particle_tracker_slurm.sh
#
# To run a single period:
#   sbatch --array=0 workflow/01_run_particle_tracker_slurm.sh

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_ENV="ccz-connectivity"
CONFIG="config/config.yaml"

# ── Activate conda ─────────────────────────────────────────────────────────────
# Adjust CONDA_BASE to match your HPC conda installation
CONDA_BASE="${CONDA_PREFIX:-/work/n01/n01/shared/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "========================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Array task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Node          : $(hostname)"
echo "Python        : $(which python)"
echo "Config        : ${CONFIG}"
echo "Time period   : ${SLURM_ARRAY_TASK_ID}"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# ── Run unmined scenario ───────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting unmined scenario …"
python workflow/01_run_particle_tracker.py \
    --config "${CONFIG}" \
    --time-period "${SLURM_ARRAY_TASK_ID}" \
    --scenario unmined

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Unmined scenario complete."

# ── Run mined scenario ─────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting mined scenario …"
python workflow/01_run_particle_tracker.py \
    --config "${CONFIG}" \
    --time-period "${SLURM_ARRAY_TASK_ID}" \
    --scenario mined

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Mined scenario complete."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done for array task ${SLURM_ARRAY_TASK_ID}."
