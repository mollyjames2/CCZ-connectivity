#!/usr/bin/env python3
"""
04_corridor_optimisation.py — Two-phase corridor optimisation.

Loads ICV scores and aggregated connectivity, runs the two-phase
optimisation (Phase 1: greedy ICV selection; Phase 2: Steiner-tree
Dijkstra refinement), and writes selection lists and edge files.

Usage
-----
    python workflow/04_corridor_optimisation.py --config config/config.yaml

Outputs (in <optimisation_dir>/)
---------------------------------
    minimal_icv_selection.txt           — Phase-2 final node list
    minimal_icv_selection_with_area.txt — Final nodes with polygon area (km²)
    phase1_selection.txt                — Phase-1 greedy selection
    corridor_edges.csv                  — Corridor edges (src, dst, flow)
    apei_paths.csv                      — APEI routing paths (src, dst, path)
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Corridor optimisation for CCZ connectivity network"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def _compute_polygon_areas_km2(
    gdf: gpd.GeoDataFrame,
    crs_equal_area: str,
) -> pd.Series:
    """
    Compute polygon areas in km² using an equal-area projection.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Grid GeoDataFrame with ``FID`` column.
    crs_equal_area : str
        EPSG code for the equal-area projection (e.g. ``"EPSG:8857"``).

    Returns
    -------
    pd.Series
        Area in km², indexed by FID.
    """
    gdf_ea = gdf.to_crs(crs_equal_area)
    return gdf_ea.set_index("FID").geometry.area / 1e6  # m² → km²


def main() -> int:
    """Entry point."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    network_dir = Path(cfg["network_dir"])
    conn_dir = Path(cfg["connectivity_dir"])
    optim_dir = Path(cfg["optimisation_dir"])
    optim_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {args.config}")
    logger.info(f"Optimisation output dir: {optim_dir}")

    from ccz_connectivity.optimisation import run_optimisation

    # ── Load inputs ────────────────────────────────────────────────────────────
    icv_path = network_dir / "icv_scores.csv"
    if not icv_path.exists():
        logger.error(f"ICV scores not found: {icv_path}")
        logger.error("Run 03_network_analysis.py first.")
        return 1
    icv_df = pd.read_csv(icv_path)
    logger.info(f"Loaded ICV scores: {len(icv_df)} polygons")

    # Use aggregated unmined connectivity for optimisation routing
    conn_path = conn_dir / "connectivity_aggregated_unmined.csv"
    if not conn_path.exists():
        # Fall back to mined if unmined not available
        conn_path = conn_dir / "connectivity_aggregated_mined.csv"
    if not conn_path.exists():
        logger.error(f"No aggregated connectivity CSV found in {conn_dir}")
        return 1
    conn_df = pd.read_csv(conn_path)
    logger.info(f"Loaded connectivity: {len(conn_df)} pairs from {conn_path.name}")

    grid_gdf = gpd.read_file(cfg["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(cfg["shapefiles"]["apei"])

    # ── Run optimisation ───────────────────────────────────────────────────────
    logger.info("Running two-phase corridor optimisation …")
    results = run_optimisation(icv_df, conn_df, apei_gdf, cfg, grid_gdf=grid_gdf)

    phase1_nodes: list[int] = results["phase1_selection"]
    final_nodes: list[int] = results["final_selection"]
    corridor_edges: pd.DataFrame = results["corridor_edges"]
    apei_paths: pd.DataFrame = results["apei_paths"]

    logger.info(f"Phase 1: {len(phase1_nodes)} selected polygons")
    logger.info(f"Phase 2 final: {len(final_nodes)} polygons (including corridors)")

    # ── Compute polygon areas ──────────────────────────────────────────────────
    crs_ea: str = cfg["grid"]["crs_equal_area"]
    area_series = _compute_polygon_areas_km2(grid_gdf, crs_ea)

    # ── Write outputs ──────────────────────────────────────────────────────────
    # Phase 1 selection
    phase1_path = optim_dir / "phase1_selection.txt"
    phase1_path.write_text("\n".join(str(n) for n in phase1_nodes) + "\n")
    logger.info(f"Saved Phase 1 selection → {phase1_path}")

    # Minimal ICV selection (Phase 2 final)
    minimal_path = optim_dir / "minimal_icv_selection.txt"
    minimal_path.write_text("\n".join(str(n) for n in sorted(final_nodes)) + "\n")
    logger.info(f"Saved final selection → {minimal_path}")

    # Final selection with area
    area_records = [
        {"polygon_ID": fid, "area_km2": float(area_series.get(fid, 0.0))}
        for fid in sorted(final_nodes)
    ]
    area_df = pd.DataFrame(area_records, columns=["polygon_ID", "area_km2"])

    # Merge ICV for context
    area_df = area_df.merge(
        icv_df[["polygon_ID", "ICV"]].rename(columns={"polygon_ID": "polygon_ID"}),
        on="polygon_ID",
        how="left",
    )
    minimal_area_path = optim_dir / "minimal_icv_selection_with_area.txt"
    area_df.to_csv(minimal_area_path, index=False)
    logger.info(f"Saved selection with area → {minimal_area_path}")

    # Corridor edges
    corridor_out = optim_dir / "corridor_edges.csv"
    corridor_edges.to_csv(corridor_out, index=False)
    logger.info(f"Saved corridor edges → {corridor_out} ({len(corridor_edges)} edges)")

    # APEI paths
    apei_out = optim_dir / "apei_paths.csv"
    apei_paths.to_csv(apei_out, index=False)
    logger.info(f"Saved APEI paths → {apei_out} ({len(apei_paths)} paths)")

    # Summary statistics
    total_area = area_df["area_km2"].sum()
    logger.info(f"Total area of selected polygons: {total_area:.1f} km²")
    logger.info("Corridor optimisation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
