# CCZ Larval Dispersal Connectivity

Code supporting James et al.(In Prep) *Larval dispersal modelling reveals connectivity deficits and priority corridors in a deep-sea protected area network*.

---

## Citation

If you use this code, please cite:

> James et al. (in prep.) Larval dispersal modelling reveals connectivity deficits and priority corridors in a deep-sea protected area network *[Journal TBC]*.

A Zenodo DOI will be added upon publication.

---

## Overview

This repository provides a complete, reproducible workflow for:

1. Downloading HYCOM bottom-layer current data for the CCZ domain.
2. Running forward particle-tracking simulations (OceanParcels) for multiple time periods and mining scenarios.
3. Building connectivity matrices from particle trajectories and computing APEI endpoint statistics.
4. Constructing igraph networks, detecting communities, and computing network metrics.
5. Computing the Integrated Connectivity Value (ICV) from four component scores.
6. Optimising a protected-area corridor network via two-phase greedy + Steiner refinement.
7. Generating publication-standard figures.

---

## Repository Structure

```
ccz-connectivity-repo/
├── config/
│   ├── config.yaml            # Production configuration (all manuscript parameters)
│   └── config_test.yaml       # Test configuration (reduced parameters)
├── data/
│   ├── hycom/                 # HYCOM NetCDF downloads (not tracked in git)
│   ├── particles/             # OceanParcels Zarr output (not tracked)
│   └── shapefiles/            # CCZ grid, APEI, AMI, boundary shapefiles
├── src/ccz_connectivity/      # Python library
│   ├── __init__.py
│   ├── grid.py
│   ├── tracking.py
│   ├── connectivity.py
│   ├── network.py
│   ├── icv.py
│   ├── optimisation.py
│   └── plotting.py
├── workflow/
│   ├── 00_download_hycom.py
│   ├── 01_run_particle_tracker.py
│   ├── 01_run_particle_tracker_slurm.sh
│   ├── 02_process_connectivity.py
│   ├── 03_network_analysis.py
│   ├── 04_corridor_optimisation.py
│   └── 05_generate_figures.py
├── analysis/
│   └── hycom_vertical_profile_analysis.py  # Standalone vertical profile characterisation (not part of main workflow)
├── tests/
│   ├── generate_test_data.py
│   ├── run_test_pipeline.sh
│   └── expected_outputs/
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## Key Parameters

| Parameter | Value |
|---|---|
| Grid domain | 160°W–110°W, 0°–25°N |
| Grid resolution | 100 km (Equal Earth, EPSG:8857) |
| PLDs | 19, 35, 69 days |
| Particles per site | 3648 |
| Time periods | Jan 2019, Jul 2019, Jan 2023, Jul 2023 |
| Timestep | 60 min |
| Diffusion (Smagorinsky) | Cs = 0.1, dx = 0.01° |
| Reciprocal threshold | 1000 particles |
| Community detection | fast_greedy (igraph) |
| ICV weights | [0.25, 0.25, 0.25, 0.25] |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/james-et-al/ccz-connectivity.git
cd ccz-connectivity
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate ccz-connectivity
```

### 3. Install the Python package in editable mode

```bash
pip install -e .
```

---

## Data Requirements

Three shapefiles must be obtained and placed in `data/shapefiles/` before running the pipeline.
The CCZ grid is built programmatically from these; HYCOM data are downloaded automatically.

| File | Field used | Source |
|---|---|---|
| `data/shapefiles/ccz_boundary.shp` | geometry | ISA GIS portal or derived from contractor data |
| `data/shapefiles/apei.shp` | `Remarks` (e.g. `"APEI-1"`) | [ISA APEI data](https://www.isa.org.jm/files/files/documents/isa-apei-data.zip) |
| `data/shapefiles/ami.shp` | geometry | [ISA Seabed Contractor Areas](https://www.isa.org.jm/maps-and-gis-data) |

All three are freely available from the International Seabed Authority (ISA) GIS portal.

See `data/README.md` for full directory layout and data provenance notes.

---

## Running the Production Workflow

All workflow scripts accept `--config config/config.yaml`.

### Step 0a — Build the CCZ grid

This only needs to be run once. It creates the 100 km equal-area grid and centroids
CSV that all downstream steps depend on.

```bash
python -c "
import yaml
from pathlib import Path
from ccz_connectivity.grid import make_ccz_grid, save_grid

cfg = yaml.safe_load(open('config/config.yaml'))
gdf = make_ccz_grid(cfg)
save_grid(
    gdf,
    Path(cfg['shapefiles']['ccz_grid']),
    Path(cfg['shapefiles']['ccz_centroids']),
)
print(f'Grid built: {len(gdf)} cells')
"
```

This reprojects the CCZ boundary to EPSG:8857 (Equal Earth), tessellates at 100 km,
clips to the CCZ outline, and saves as WGS84 (EPSG:4326).

### Step 0b — Download HYCOM data

```bash
python workflow/00_download_hycom.py --config config/config.yaml
```

Downloads bottom-layer u/v currents (`water_u_bottom`, `water_v_bottom`) for all four
time periods from HYCOM GOFS 3.1 via OPeNDAP.  Add `--period 0` to download a single
period.  Expect ~2–5 GB per period.

### Step 1 — Run particle tracking

> **HPC required.** Each simulation releases ~68 million particles (3,648 per site ×
> ~1,225 cells × 4 time periods × 2 scenarios).  On ARCHER2 this takes ~12–24 hrs per
> scenario with 128 cores.  The SLURM script submits a 4-element array job (one element
> per time period) and runs both scenarios sequentially within each task.

On ARCHER2 or equivalent (recommended):
```bash
sbatch workflow/01_run_particle_tracker_slurm.sh
```

Single run for testing (one time period, one scenario):
```bash
python workflow/01_run_particle_tracker.py \
    --config config/config.yaml \
    --time-period 0 \
    --scenario unmined
```

Output is one Zarr store per (time period, scenario) in `data/particles/`.

### Step 2 — Process connectivity

```bash
python workflow/02_process_connectivity.py \
    --config config/config.yaml \
    --scenario both
```

### Step 3 — Network analysis

```bash
python workflow/03_network_analysis.py --config config/config.yaml
```

### Step 4 — Corridor optimisation

```bash
python workflow/04_corridor_optimisation.py --config config/config.yaml
```

### Step 5 — Generate figures

```bash
# All figures
python workflow/05_generate_figures.py --config config/config.yaml

# Specific figures
python workflow/05_generate_figures.py --config config/config.yaml --figures 1 3 5 ed
```

---

## Supplementary Analysis

`analysis/hycom_vertical_profile_analysis.py` is a standalone script (not part of the
main workflow) that characterises the vertical structure of HYCOM GOFS 3.1 currents
across the CCZ domain.  It covers the four larval dispersal simulation windows and
produces speed profiles, directional profiles, and larval reachability cross-references
relative to near-bed conditions.

```bash
python analysis/hycom_vertical_profile_analysis.py
```

On first run the script fetches data from HYCOM OPeNDAP and caches the result to
`analysis/hycom_data_cache.npz`; subsequent runs load from the cache.  Outputs (5 PNG
figures) are written to the `analysis/` directory and are not tracked in git.

---

## Test Pipeline

A reduced-scale test pipeline uses synthetic data (no real HYCOM or particle-tracking runs required).  The test domain is 0–50°E, 0–50°N with 2500 1°-resolution cells and 10 particles per site.  The pipeline enters at step 02.

```bash
# From repo root
bash tests/run_test_pipeline.sh
```

Or run individual steps:

```bash
python tests/generate_test_data.py
python workflow/02_process_connectivity.py --config config/config_test.yaml
python workflow/03_network_analysis.py     --config config/config_test.yaml
python workflow/04_corridor_optimisation.py --config config/config_test.yaml
python workflow/05_generate_figures.py     --config config/config_test.yaml --figures 3 4 5
```

See `tests/README.md` for details.

---

## Output Files

After a successful run, outputs are organised under `outputs/`:

```
outputs/
├── connectivity/
│   ├── connectivity_aggregated_{scenario}.csv
│   ├── per_particle_{scenario}.csv
│   ├── apei_endpoints_{scenario}.csv
│   └── comparison_stats_{scenario}.csv
├── network/
│   ├── network_metrics_{scenario}.csv
│   ├── apei_support_{scenario}.csv
│   ├── mining_replenishment.csv
│   ├── scenario_stable_connectivity.csv
│   └── icv_scores.csv
├── optimisation/
│   ├── phase1_selection.txt
│   ├── minimal_icv_selection.txt
│   ├── minimal_icv_selection_with_area.txt
│   ├── corridor_edges.csv
│   └── apei_paths.csv
└── figures/
    ├── fig1_apei_dispersal_density.png
    ├── fig2_ami_particle_density.png
    ├── fig3_community_membership.png
    ├── fig4_scenario_stable_connectivity.png
    ├── fig5_optimised_corridor_network.png
    ├── ed1_domain_overview.png
    ├── ed_betweenness_unmined.png
    ├── ed_betweenness_mined.png
    └── ed_icv_component_distributions.png
```

---

## License

MIT — see `LICENSE`.
