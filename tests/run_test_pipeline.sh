#!/bin/bash
# run_test_pipeline.sh — Run the CCZ connectivity test pipeline.
#
# Generates synthetic test data, runs workflow steps 02–05 using the
# test configuration, checks that all expected output files exist, and
# reports PASS or FAIL.
#
# Usage (from repo root):
#   bash tests/run_test_pipeline.sh
#
# Requirements:
#   conda environment 'ccz-connectivity' must be installed.

set -euo pipefail

CONDA_ENV="ccz-connectivity"
CONFIG="config/config_test.yaml"

# ── Activate conda ─────────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_PREFIX:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "WARNING: Could not source conda.sh — assuming environment is already active"
fi

echo "========================================"
echo "CCZ Connectivity Test Pipeline"
echo "Config: ${CONFIG}"
echo "Python: $(which python)"
echo "========================================"

PASS=0
FAIL=0
FAILED_STEPS=()

_run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "--- ${step_name} ---"
    if "$@"; then
        echo "[OK] ${step_name}"
        PASS=$((PASS + 1))
    else
        echo "[FAILED] ${step_name}"
        FAILED_STEPS+=("${step_name}")
        FAIL=$((FAIL + 1))
        echo "Pipeline aborted after step failure: ${step_name}"
        echo "RESULT: FAIL"
        exit 1
    fi
}

# ── Step 0: Generate synthetic test data ──────────────────────────────────────
_run_step "generate_test_data" python tests/generate_test_data.py

# ── Step 1: Process connectivity (step 02) ───────────────────────────────────
_run_step "02_process_connectivity" \
    python workflow/02_process_connectivity.py \
    --config "${CONFIG}" \
    --scenario both

# ── Step 2: Network analysis (step 03) ────────────────────────────────────────
_run_step "03_network_analysis" \
    python workflow/03_network_analysis.py \
    --config "${CONFIG}"

# ── Step 3: Corridor optimisation (step 04) ───────────────────────────────────
_run_step "04_corridor_optimisation" \
    python workflow/04_corridor_optimisation.py \
    --config "${CONFIG}"

# ── Step 4: Generate figures (step 05) ────────────────────────────────────────
_run_step "05_generate_figures" \
    python workflow/05_generate_figures.py \
    --config "${CONFIG}" \
    --figures 1 2 3 4 5

# ── Step 5: Generate tables (step 06) ─────────────────────────────────────────
#_run_step "06_generate_tables" \
#    python workflow/06_generate_tables.py \
#    --config "${CONFIG}"

# ── Check expected output files ───────────────────────────────────────────────
echo ""
echo "--- Checking expected output files ---"

CONN_DIR="tests/expected_outputs/connectivity"
NET_DIR="tests/expected_outputs/network"
OPT_DIR="tests/expected_outputs/optimisation"
FIG_DIR="tests/expected_outputs/figures"

expected_files=(
    # Connectivity outputs
    "${CONN_DIR}/connectivity_aggregated_unmined.csv"
    "${CONN_DIR}/connectivity_aggregated_mined.csv"
    "${CONN_DIR}/per_particle_unmined.csv"
    "${CONN_DIR}/per_particle_mined.csv"
    "${CONN_DIR}/apei_endpoints_unmined.csv"
    "${CONN_DIR}/apei_endpoints_mined.csv"
    "${CONN_DIR}/comparison_stats_unmined.csv"
    # Network analysis outputs
    "${NET_DIR}/network_metrics_unmined.csv"
    "${NET_DIR}/network_metrics_mined.csv"
    "${NET_DIR}/apei_support_unmined.csv"
    "${NET_DIR}/mining_replenishment.csv"
    "${NET_DIR}/scenario_stable_connectivity.csv"
    "${NET_DIR}/icv_scores.csv"
    # Optimisation outputs
    "${OPT_DIR}/phase1_selection.txt"
    "${OPT_DIR}/minimal_icv_selection.txt"
    "${OPT_DIR}/minimal_icv_selection_with_area.txt"
    "${OPT_DIR}/corridor_edges.csv"
    "${OPT_DIR}/apei_paths.csv"
    # Figures (saved as JPG)
    "${FIG_DIR}/fig3_community_membership.jpg"
    "${FIG_DIR}/fig4_scenario_stable_connectivity.jpg"
    "${FIG_DIR}/fig5_optimised_corridor_network.jpg"
    # Tables
    "tests/expected_outputs/tables/ed_table3_simulation_parameters.csv"
    "tests/expected_outputs/tables/ed_table4_temporal_comparisons.csv"
)

missing_files=()
for f in "${expected_files[@]}"; do
    if [ -f "${f}" ]; then
        echo "  [FOUND] ${f}"
    else
        echo "  [MISSING] ${f}"
        missing_files+=("${f}")
    fi
done

# ── Report ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Test Pipeline Summary"
echo "========================================"
echo "Steps passed : ${PASS}"
echo "Steps failed : ${FAIL}"

if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo "Failed steps : ${FAILED_STEPS[*]}"
fi

echo "Missing files: ${#missing_files[@]}"
if [ ${#missing_files[@]} -gt 0 ]; then
    for f in "${missing_files[@]}"; do
        echo "  - ${f}"
    done
fi

if [ "${FAIL}" -eq 0 ] && [ "${#missing_files[@]}" -eq 0 ]; then
    echo ""
    echo "RESULT: PASS"
    exit 0
else
    echo ""
    echo "RESULT: FAIL"
    exit 1
fi
