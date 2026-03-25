# tests/ — Test Pipeline

This directory contains the test data generator and test pipeline runner
for the CCZ connectivity workflow.

---

## Overview

The test pipeline uses **synthetic data** and does not require HYCOM downloads
or OceanParcels runs.  It enters the main workflow at step 02
(process_connectivity) and exercises the full analysis stack:

```
generate_test_data.py
       ↓
02_process_connectivity.py
       ↓
03_network_analysis.py
       ↓
04_corridor_optimisation.py
       ↓
05_generate_figures.py
```

---

## Test Domain

The test domain differs from production to keep data volumes small:

| Parameter | Test | Production |
|---|---|---|
| Domain | 0–50°E, 0–50°N | 160°W–110°W, 0°–25°N |
| Grid cells | 50 × 50 = 2500 | ~variable (CCZ clipped) |
| Grid resolution | 1° | 100 km |
| Particles per site | 10 | 3648 |
| Reciprocal threshold | 10 | 1000 |
| Figure DPI | 150 | 300 |

---

## Running the Tests

From the repository root:

```bash
# Full test pipeline (activates conda env automatically)
bash tests/run_test_pipeline.sh

# Or step by step:
python tests/generate_test_data.py
python workflow/02_process_connectivity.py --config config/config_test.yaml
python workflow/03_network_analysis.py     --config config/config_test.yaml
python workflow/04_corridor_optimisation.py --config config/config_test.yaml
python workflow/05_generate_figures.py     --config config/config_test.yaml --figures 3 4 5
```

---

## Directory Layout

```
tests/
├── generate_test_data.py    # Generates synthetic shapefiles and Zarr stores
├── run_test_pipeline.sh     # Runs full test pipeline and checks outputs
├── data/                    # Generated test data (not tracked in git)
│   ├── particles/           # 8 Zarr stores (4 periods × 2 scenarios)
│   └── shapefiles/          # Synthetic grid, APEI, AMI, boundary
└── expected_outputs/        # Pipeline outputs written here (not tracked)
    ├── connectivity/
    ├── network/
    ├── optimisation/
    └── figures/
```

---

## Zarr Schema

The synthetic Zarr stores match the OceanParcels 3.0+ schema:

| Array | dtype | shape | chunks | attrs |
|---|---|---|---|---|
| `trajectory` | int64 | (n_particles,) | (n_particles,) | `_ARRAY_DIMENSIONS: ["trajectory"]` |
| `obs` | int32 | (n_timesteps,) | (n_timesteps,) | `_ARRAY_DIMENSIONS: ["obs"]` |
| `lat` | float32 | (n_particles, n_timesteps) | (n_particles, 1) | `_ARRAY_DIMENSIONS: ["trajectory", "obs"]` |
| `lon` | float32 | (n_particles, n_timesteps) | (n_particles, 1) | same |
| `time` | float64 | (n_particles, n_timesteps) | (n_particles, 1) | `units: "seconds since <start_date>"` |
| `z` | float32 | (n_particles, n_timesteps) | (n_particles, 1) | `units: "m"` |
| `distance` | float32 | (n_particles, n_timesteps) | (n_particles, 1) | `units: "km"` |

Root attributes: `Conventions="CF-1.6"`, `parcels_version="3.0.4"`,
`feature_type="trajectory"`.
