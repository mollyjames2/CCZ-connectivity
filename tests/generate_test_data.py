#!/usr/bin/env python3
"""
generate_test_data.py — Create synthetic test data for the CCZ connectivity pipeline.

Generates:
- 8 Zarr stores (4 time periods × 2 scenarios: unmined / mined) in a
  snapshot schema: release positions + lat/lon at 3 PLD endpoints only.
- CCZ grid shapefile (50 × 50 = 2500 cells, 1° resolution, 0–50°E × 0–50°N).
- CCZ grid centroids CSV.
- APEI shapefile (5 polygons labelled APEI-1 through APEI-5).
- AMI shapefile (6 polygons).
- CCZ boundary shapefile.

Zarr schema (snapshot format)
------------------------------
  trajectory   : (n_particles,)    int64   — unique particle IDs
  release_lon  : (n_particles,)    float32 — grid-cell centroid longitude
  release_lat  : (n_particles,)    float32 — grid-cell centroid latitude
  lon          : (n_particles, 3)  float32 — positions at PLDs 19, 35, 69 d
  lat          : (n_particles, 3)  float32 — positions at PLDs 19, 35, 69 d
  time         : (3,)              int32   — PLD snapshot values in days

This schema differs from OceanParcels' standard continuous-output format.
connectivity.py detects the schema via the presence of the ``release_lon``
variable and adjusts its reading logic accordingly.

All spatial data uses the test domain 0–50°E, 0–50°N.

Usage
-----
    python tests/generate_test_data.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import zarr
from shapely.geometry import Point, box

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
PARTICLES_DIR = DATA_DIR / "particles"
SHAPEFILES_DIR = DATA_DIR / "shapefiles"

DOMAIN_LON = (0.0, 25.0)
DOMAIN_LAT = (0.0, 25.0)
GRID_RES_DEG = 1.0

N_CELLS_X = 25
N_CELLS_Y = 25
N_CELLS = N_CELLS_X * N_CELLS_Y  

PPS = 200                            # particles per site; matches config_test.yaml
N_PARTICLES = N_CELLS * PPS        

PLD_DAYS: list[int] = [19, 35, 69]
PLD_HOURS: list[int] = [d * 24 for d in PLD_DAYS]   # [456, 840, 1656]
MAX_PLD_HOURS: int = max(PLD_HOURS)                   # 1656

STEP_SCALE = 0.05   # degrees per hour (random-walk standard deviation)
RANDOM_SEED = 42    # differs from production (123) to catch hardcoded-seed bugs

TIME_PERIODS = [
    {"label": "Jan2019", "start": "2019-01-01"},
    {"label": "Jul2019", "start": "2019-07-01"},
    {"label": "Jan2023", "start": "2023-01-01"},
    {"label": "Jul2023", "start": "2023-07-01"},
]


# ── Grid helpers ───────────────────────────────────────────────────────────────

def build_centroids() -> tuple[np.ndarray, np.ndarray]:
    """
    Return centroid (lon, lat) arrays for all N_CELLS grid cells.

    Cells are enumerated row-major: ix varies fastest.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(centroids_lon, centroids_lat)``, each shape ``(N_CELLS,)``.
    """
    centroids_lon = np.empty(N_CELLS, dtype=np.float32)
    centroids_lat = np.empty(N_CELLS, dtype=np.float32)
    for cell_idx in range(N_CELLS):
        ix = cell_idx % N_CELLS_X
        iy = cell_idx // N_CELLS_X
        centroids_lon[cell_idx] = DOMAIN_LON[0] + (ix + 0.5) * GRID_RES_DEG
        centroids_lat[cell_idx] = DOMAIN_LAT[0] + (iy + 0.5) * GRID_RES_DEG
    return centroids_lon, centroids_lat


def build_grid(centroids_lon: np.ndarray, centroids_lat: np.ndarray) -> gpd.GeoDataFrame:
    """
    Create a 25 × 25 degree-resolution grid GeoDataFrame.

    Parameters
    ----------
    centroids_lon : np.ndarray
        Cell centroid longitudes, shape (N_CELLS,).
    centroids_lat : np.ndarray
        Cell centroid latitudes, shape (N_CELLS,).

    Returns
    -------
    gpd.GeoDataFrame
        Grid with ``FID`` column in EPSG:4326.
    """
    cells = []
    for cell_idx in range(N_CELLS):
        cx, cy = float(centroids_lon[cell_idx]), float(centroids_lat[cell_idx])
        half = GRID_RES_DEG / 2
        cells.append(box(cx - half, cy - half, cx + half, cy + half))
    gdf = gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:4326")
    gdf["FID"] = gdf.index.astype(int)
    return gdf


def build_apei_shapefile() -> gpd.GeoDataFrame:
    """
    Create 5 synthetic APEI polygons (≈3° squares).

    Returns
    -------
    gpd.GeoDataFrame
        APEI polygons with ``Remarks`` column in EPSG:4326.
    """
    centres = [
        (3.0, 3.0),
        (8.0, 18.0),
        (13.0, 10.0),
        (19.0, 5.0),
        (22.0, 22.0),
    ]
    polygons, labels = [], []
    for i, (cx, cy) in enumerate(centres):
        polygons.append(box(cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5))
        labels.append(f"APEI-{i + 1}")
    return gpd.GeoDataFrame({"Remarks": labels, "geometry": polygons}, crs="EPSG:4326")


def build_ami_shapefile() -> gpd.GeoDataFrame:
    """
    Create 6 synthetic AMI polygons (≈2° squares).

    Returns
    -------
    gpd.GeoDataFrame
        AMI polygons with ``AMI_ID`` column in EPSG:4326.
    """
    centres = [
        (6.0, 4.0),
        (10.0, 22.0),
        (15.0, 14.0),
        (18.0, 19.0),
        (22.0, 9.0),
        (4.0, 20.0),
    ]
    polygons, ids = [], []
    for i, (cx, cy) in enumerate(centres):
        polygons.append(box(cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0))
        ids.append(f"AMI-{i + 1}")
    return gpd.GeoDataFrame({"AMI_ID": ids, "geometry": polygons}, crs="EPSG:4326")


def build_boundary() -> gpd.GeoDataFrame:
    """
    Create the test domain boundary polygon.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with the domain bounding box.
    """
    return gpd.GeoDataFrame(
        {"geometry": [box(DOMAIN_LON[0], DOMAIN_LAT[0], DOMAIN_LON[1], DOMAIN_LAT[1])]},
        crs="EPSG:4326",
    )


def save_shapefiles(
    grid_gdf: gpd.GeoDataFrame,
    apei_gdf: gpd.GeoDataFrame,
    ami_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    centroids_lon: np.ndarray,
    centroids_lat: np.ndarray,
) -> None:
    """
    Save all synthetic shapefiles and the centroids CSV.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame.
    apei_gdf : gpd.GeoDataFrame
        APEI GeoDataFrame.
    ami_gdf : gpd.GeoDataFrame
        AMI GeoDataFrame.
    boundary_gdf : gpd.GeoDataFrame
        Domain boundary GeoDataFrame.
    centroids_lon : np.ndarray
        Cell centroid longitudes.
    centroids_lat : np.ndarray
        Cell centroid latitudes.
    """
    print("Saving shapefiles …")
    grid_gdf.to_file(SHAPEFILES_DIR / "ccz_grid.shp")
    apei_gdf.to_file(SHAPEFILES_DIR / "apei.shp")
    ami_gdf.to_file(SHAPEFILES_DIR / "ami.shp")
    boundary_gdf.to_file(SHAPEFILES_DIR / "ccz_boundary.shp")

    import pandas as pd
    pd.DataFrame({
        "polyid": np.arange(N_CELLS, dtype=int),
        "lon": centroids_lon.astype(float),
        "lat": centroids_lat.astype(float),
    }).to_csv(SHAPEFILES_DIR / "ccz_grid_centroids.csv", index=False)

    print(f"  Grid ({N_CELLS} cells), APEIs ({len(apei_gdf)}), "
          f"AMIs ({len(ami_gdf)}), boundary, centroids saved")


# ── Particle trajectory simulation ────────────────────────────────────────────

def simulate_snapshots(
    release_lons: np.ndarray,
    release_lats: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate hourly random-walk trajectories and return positions at PLD snapshots.

    The simulation runs for MAX_PLD_HOURS (1656 h = 69 d) at hourly steps.
    Positions are sampled at hours 456 (day 19), 840 (day 35), and 1656 (day 69).
    The domain is enforced by clipping at each step.

    Parameters
    ----------
    release_lons : np.ndarray
        Release longitudes, shape (n_particles,).
    release_lats : np.ndarray
        Release latitudes, shape (n_particles,).
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(snap_lon, snap_lat)``, each shape ``(n_particles, 3)`` float32,
        columns corresponding to PLDs 19, 35, 69 days.
    """
    n = len(release_lons)
    lon_walk = release_lons.copy().astype(np.float32)
    lat_walk = release_lats.copy().astype(np.float32)

    snap_lon = np.empty((n, 3), dtype=np.float32)
    snap_lat = np.empty((n, 3), dtype=np.float32)
    snap_at = {h: i for i, h in enumerate(PLD_HOURS)}

    for step in range(1, MAX_PLD_HOURS + 1):
        lon_walk += rng.normal(0.0, STEP_SCALE, n).astype(np.float32)
        lat_walk += rng.normal(0.0, STEP_SCALE, n).astype(np.float32)
        np.clip(lon_walk, DOMAIN_LON[0], DOMAIN_LON[1], out=lon_walk)
        np.clip(lat_walk, DOMAIN_LAT[0], DOMAIN_LAT[1], out=lat_walk)
        if step in snap_at:
            idx = snap_at[step]
            snap_lon[:, idx] = lon_walk
            snap_lat[:, idx] = lat_walk

    return snap_lon, snap_lat


# ── Zarr writers ───────────────────────────────────────────────────────────────

def _write_zarr(
    zarr_path: Path,
    traj_ids: np.ndarray,
    release_lons: np.ndarray,
    release_lats: np.ndarray,
    snap_lon: np.ndarray,
    snap_lat: np.ndarray,
    start_date: str,
) -> None:
    """
    Write a snapshot-schema Zarr store to disk.

    Parameters
    ----------
    zarr_path : Path
        Output path (will be overwritten if it exists).
    traj_ids : np.ndarray
        Particle IDs, shape (n_particles,).
    release_lons : np.ndarray
        Release longitudes (centroids), shape (n_particles,).
    release_lats : np.ndarray
        Release latitudes (centroids), shape (n_particles,).
    snap_lon : np.ndarray
        End-position longitudes at PLDs, shape (n_particles, 3).
    snap_lat : np.ndarray
        End-position latitudes at PLDs, shape (n_particles, 3).
    start_date : str
        Simulation start date string (used for root attrs only).
    """
    n = len(traj_ids)

    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({
        "Conventions": "CF-1.6",
        "feature_type": "trajectory",
        "schema_version": "snapshot_v1",
        "pld_days": PLD_DAYS,
        "start_date": start_date,
    })

    def _ds(name: str, data: np.ndarray, dims: list[str], **attrs: object) -> None:
        chunks = (n,) if data.ndim == 1 else (n, data.shape[1])
        ds = store.create_dataset(name, data=data, dtype=data.dtype, chunks=chunks)
        ds.attrs["_ARRAY_DIMENSIONS"] = dims
        ds.attrs.update(attrs)

    _ds("trajectory", traj_ids, ["trajectory"])
    _ds("release_lon", release_lons, ["trajectory"],
        units="degrees_east", long_name="release longitude (cell centroid)")
    _ds("release_lat", release_lats, ["trajectory"],
        units="degrees_north", long_name="release latitude (cell centroid)")
    _ds("lon", snap_lon, ["trajectory", "pld"],
        units="degrees_east", long_name="longitude at PLD snapshot")
    _ds("lat", snap_lat, ["trajectory", "pld"],
        units="degrees_north", long_name="latitude at PLD snapshot")

    time_arr = np.array(PLD_DAYS, dtype=np.int32)
    ds_time = store.create_dataset(
        "time", data=time_arr, dtype="int32", chunks=(len(PLD_DAYS),)
    )
    ds_time.attrs["_ARRAY_DIMENSIONS"] = ["pld"]
    ds_time.attrs["units"] = "days"
    ds_time.attrs["long_name"] = "planktonic larval duration at snapshot"


def generate_unmined_zarr(
    period: dict,
    centroids_lon: np.ndarray,
    centroids_lat: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the unmined Zarr store for one time period.

    Parameters
    ----------
    period : dict
        Time period dict with keys ``label`` and ``start``.
    centroids_lon : np.ndarray
        Cell centroid longitudes, shape (N_CELLS,).
    centroids_lat : np.ndarray
        Cell centroid latitudes, shape (N_CELLS,).
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(traj_ids, release_lons, release_lats, keep_mask)`` for use by
        :func:`generate_mined_zarr`.
    """
    label = period["label"]
    zarr_path = PARTICLES_DIR / f"particles_{label}_unmined.zarr"
    print(f"  Generating unmined store: {zarr_path.name}")

    # Release positions: PPS copies of each centroid (no jitter — stored as centroids)
    release_lons = np.repeat(centroids_lon, PPS).astype(np.float32)
    release_lats = np.repeat(centroids_lat, PPS).astype(np.float32)
    traj_ids = np.arange(N_PARTICLES, dtype=np.int64)

    snap_lon, snap_lat = simulate_snapshots(release_lons, release_lats, rng)

    _write_zarr(zarr_path, traj_ids, release_lons, release_lats,
                snap_lon, snap_lat, period["start"])

    print(f"    Written: {N_PARTICLES} particles × 3 PLD snapshots {PLD_DAYS}")
    return traj_ids, release_lons, release_lats, snap_lon, snap_lat


def generate_mined_zarr(
    period: dict,
    ami_gdf: gpd.GeoDataFrame,
    traj_ids: np.ndarray,
    release_lons: np.ndarray,
    release_lats: np.ndarray,
    snap_lon: np.ndarray,
    snap_lat: np.ndarray,
) -> None:
    """
    Generate the mined Zarr store by removing AMI-origin particles.

    Particles whose release centroid falls within an AMI polygon are excluded.

    Parameters
    ----------
    period : dict
        Time period dict with key ``label``.
    ami_gdf : gpd.GeoDataFrame
        AMI polygon GeoDataFrame.
    traj_ids : np.ndarray
        All particle trajectory IDs from the unmined store.
    release_lons : np.ndarray
        All release longitudes from the unmined store.
    release_lats : np.ndarray
        All release latitudes from the unmined store.
    snap_lon : np.ndarray
        Snapshot longitudes from the unmined store, shape (n_particles, 3).
    snap_lat : np.ndarray
        Snapshot latitudes from the unmined store, shape (n_particles, 3).
    """
    label = period["label"]
    zarr_path = PARTICLES_DIR / f"particles_{label}_mined.zarr"
    print(f"  Generating mined store (AMI particles removed): {zarr_path.name}")

    ami_union = ami_gdf.unary_union
    keep = np.ones(N_PARTICLES, dtype=bool)
    n_ami_cells = 0
    for cell_idx in range(N_CELLS):
        cx, cy = float(release_lons[cell_idx * PPS]), float(release_lats[cell_idx * PPS])
        if ami_union.contains(Point(cx, cy)):
            keep[cell_idx * PPS : (cell_idx + 1) * PPS] = False
            n_ami_cells += 1

    n_kept = int(keep.sum())
    print(f"    AMI cells excluded: {n_ami_cells}; particles kept: {n_kept}/{N_PARTICLES}")

    _write_zarr(
        zarr_path,
        traj_ids[keep],
        release_lons[keep],
        release_lats[keep],
        snap_lon[keep],
        snap_lat[keep],
        period["start"],
    )
    print(f"    Written: {n_kept} particles × 3 PLD snapshots {PLD_DAYS}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    """Generate all test data."""
    print("=" * 60)
    print("CCZ Connectivity — Synthetic Test Data Generator")
    print("=" * 60)

    PARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    SHAPEFILES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Grid and shapefiles ────────────────────────────────────────────────────
    print("\nBuilding synthetic shapefiles …")
    centroids_lon, centroids_lat = build_centroids()
    grid_gdf = build_grid(centroids_lon, centroids_lat)
    apei_gdf = build_apei_shapefile()
    ami_gdf = build_ami_shapefile()
    boundary_gdf = build_boundary()
    save_shapefiles(grid_gdf, apei_gdf, ami_gdf, boundary_gdf,
                    centroids_lon, centroids_lat)

    # ── Particle trajectories ──────────────────────────────────────────────────
    print("\nGenerating synthetic particle trajectories …")
    print(f"  Schema: snapshot_v1 | PLDs: {PLD_DAYS} days | {N_PARTICLES} particles/period")
    rng = np.random.default_rng(RANDOM_SEED)

    for period in TIME_PERIODS:
        print(f"\n[{period['label']}]")
        traj_ids, release_lons, release_lats, snap_lon, snap_lat = generate_unmined_zarr(
            period, centroids_lon, centroids_lat, rng
        )
        generate_mined_zarr(
            period, ami_gdf, traj_ids, release_lons, release_lats, snap_lon, snap_lat
        )

    print("\n" + "=" * 60)
    print("Test data generation complete.")
    print(f"  Shapefiles : {SHAPEFILES_DIR}/")
    print(f"  Particles  : {PARTICLES_DIR}/")
    print(f"  Zarr schema: snapshot_v1 (release_lon/lat + lon/lat at {PLD_DAYS} d)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
