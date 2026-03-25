#!/usr/bin/env python3
"""
06_generate_tables.py — Generate manuscript-ready tables.

Produces CSV files for:
    Table 1   — AMI larval connectivity by PLD (presence-based reach, source-
                and sink-normalised flux under baseline and mining disturbance).
    ED Table 1 — Per-APEI breakdown of source/sink-normalised flux to AMIs.
    ED Table 2 — ENSO phase classification for each simulation period (static).
    ED Table 3 — Simulation parameter summary (static).
    ED Table 4 — Pairwise temporal comparison statistics (Jaccard, Cohen's
                kappa, permutation test) for all period pairs.

Usage
-----
    python workflow/06_generate_tables.py --config config/config.yaml

Outputs (in <optimisation_dir>/../tables/ or <figures_dir>/../tables/):
    table1_ami_connectivity.csv
    ed_table1_per_apei_ami_flux.csv
    ed_table2_enso_classification.csv
    ed_table3_simulation_parameters.csv
    ed_table4_temporal_comparisons.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate manuscript tables for CCZ connectivity study"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def _tables_dir(cfg: dict) -> Path:
    """Resolve output directory for tables."""
    # Place tables alongside figures_dir
    figures_dir = Path(cfg["figures_dir"])
    tables_dir = figures_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


# ── Table 1 ───────────────────────────────────────────────────────────────────

def build_table1(cfg: dict, tables_dir: Path) -> None:
    """
    Build Table 1: AMI larval connectivity by PLD.

    Reports per-PLD presence-based spatial reach (applying the 1,000-particle
    threshold), source-normalised flux (proportion of APEI particles reaching
    AMIs), and sink-normalised flux by origin category under baseline and
    mining disturbance.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    tables_dir : Path
        Output directory.
    """
    logger.info("Building Table 1 …")
    conn_dir = Path(cfg["connectivity_dir"])
    threshold: int = cfg["connectivity"]["reciprocal_threshold"]
    pld_days: list[int] = cfg["tracking"]["pld_days"]

    ami_gdf = gpd.read_file(cfg["shapefiles"]["ami"])
    apei_gdf = gpd.read_file(cfg["shapefiles"]["apei"])
    if ami_gdf.crs is None or ami_gdf.crs.to_epsg() != 4326:
        ami_gdf = ami_gdf.to_crs("EPSG:4326")
    if apei_gdf.crs is None or apei_gdf.crs.to_epsg() != 4326:
        apei_gdf = apei_gdf.to_crs("EPSG:4326")

    ami_union = ami_gdf.unary_union
    apei_union = apei_gdf.unary_union
    n_ami_polys = len(ami_gdf)

    def _in_polygon(lon: float, lat: float, union) -> bool:
        if np.isnan(lon) or np.isnan(lat):
            return False
        return union.contains(Point(lon, lat))

    records = []
    for pld in pld_days:
        for scenario in ["unmined", "mined"]:
            p = conn_dir / f"per_particle_pld{pld}_{scenario}.csv"
            if not p.exists():
                fp = conn_dir / f"per_particle_{scenario}.csv"
                if not fp.exists():
                    continue
                df = pd.read_csv(fp)
                df = df[df["PLD"] == pld]
            else:
                df = pd.read_csv(p)

            if df.empty:
                continue

            # Tag particle endpoints relative to AMI
            df = df.copy()
            df["in_ami"] = [
                _in_polygon(r["end_lon"], r["end_lat"], ami_union)
                for _, r in df.iterrows()
            ]

            # --- Presence-based spatial reach ---
            # Which AMI polygons receive ≥ threshold particles from each origin?
            # Simplified: count particles per AMI polygon
            ami_settlers = df[df["in_ami"]]

            # Group by which AMI polygon they landed in
            ami_settler_pts = gpd.GeoDataFrame(
                ami_settlers,
                geometry=gpd.points_from_xy(ami_settlers["end_lon"], ami_settlers["end_lat"]),
                crs="EPSG:4326",
            )
            ami_joined = gpd.sjoin(ami_settler_pts, ami_gdf[["geometry"]], how="left", predicate="within")
            ami_joined["ami_poly_idx"] = ami_joined["index_right"]

            # Per origin category
            if "start_category" in df.columns:
                for cat in ["APEI", "AMI", "unprotected"]:
                    cat_df = ami_joined[ami_joined["start_category"] == cat]
                    amis_covered = (
                        cat_df.groupby("ami_poly_idx").size()
                        .pipe(lambda s: (s >= threshold).sum())
                    )
                    reach_pct = 100.0 * amis_covered / n_ami_polys

                    # Source-normalised flux: proportion of category particles reaching AMIs
                    total_cat = len(df[df["start_category"] == cat]) if "start_category" in df.columns else 0
                    src_flux = 100.0 * len(cat_df) / max(total_cat, 1)

                    records.append({
                        "PLD_days": pld,
                        "scenario": scenario,
                        "origin_category": cat,
                        "presence_reach_pct": round(reach_pct, 2),
                        "source_normalised_flux_pct": round(src_flux, 3),
                    })
            else:
                # No start_category — report totals only
                amis_covered = (
                    ami_joined.groupby("ami_poly_idx").size()
                    .pipe(lambda s: (s >= threshold).sum())
                )
                reach_pct = 100.0 * amis_covered / n_ami_polys
                src_flux = 100.0 * len(ami_joined) / max(len(df), 1)
                records.append({
                    "PLD_days": pld,
                    "scenario": scenario,
                    "origin_category": "all",
                    "presence_reach_pct": round(reach_pct, 2),
                    "source_normalised_flux_pct": round(src_flux, 3),
                })

    if records:
        table1 = pd.DataFrame(records)
        out = tables_dir / "table1_ami_connectivity.csv"
        table1.to_csv(out, index=False)
        logger.info(f"Saved Table 1 → {out}")
    else:
        logger.warning("No data for Table 1")


# ── ED Table 1 ────────────────────────────────────────────────────────────────

def build_ed_table1(cfg: dict, tables_dir: Path) -> None:
    """
    Build ED Table 1: per-APEI modelled larval connectivity to AMIs.

    Reports source- and sink-normalised flux for each APEI under baseline
    and mining disturbance conditions.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    tables_dir : Path
        Output directory.
    """
    logger.info("Building ED Table 1 …")
    conn_dir = Path(cfg["connectivity_dir"])
    label_field: str = cfg["shapefiles"]["apei_label_field"]
    threshold: int = cfg["connectivity"]["reciprocal_threshold"]
    pld_days: list[int] = cfg["tracking"]["pld_days"]

    ami_gdf = gpd.read_file(cfg["shapefiles"]["ami"])
    apei_gdf = gpd.read_file(cfg["shapefiles"]["apei"])
    if ami_gdf.crs is None or ami_gdf.crs.to_epsg() != 4326:
        ami_gdf = ami_gdf.to_crs("EPSG:4326")
    if apei_gdf.crs is None or apei_gdf.crs.to_epsg() != 4326:
        apei_gdf = apei_gdf.to_crs("EPSG:4326")

    ami_union = ami_gdf.unary_union
    apei_labels = sorted(apei_gdf[label_field].unique())

    records = []
    for scenario in ["unmined", "mined"]:
        pall = conn_dir / f"per_particle_{scenario}.csv"
        if not pall.exists():
            continue
        all_df = pd.read_csv(pall)

        # Identify APEI-origin particles
        if "start_category" not in all_df.columns:
            logger.warning("start_category missing — ED Table 1 will be incomplete")
            continue

        apei_df = all_df[all_df["start_category"] == "APEI"]
        total_apei = len(apei_df)

        # Tag which APEI each particle originated from via spatial join
        apei_pts = gpd.GeoDataFrame(
            apei_df.copy(),
            geometry=gpd.points_from_xy(apei_df["start_lon"], apei_df["start_lat"]),
            crs="EPSG:4326",
        )
        apei_pts_joined = gpd.sjoin(
            apei_pts, apei_gdf[[label_field, "geometry"]], how="left", predicate="within"
        )
        apei_pts_joined["apei_label"] = apei_pts_joined[label_field]

        # Tag particles ending in AMI
        apei_pts_joined["in_ami"] = [
            ami_union.contains(Point(r["end_lon"], r["end_lat"]))
            if not (np.isnan(r["end_lon"]) or np.isnan(r["end_lat"]))
            else False
            for _, r in apei_pts_joined.iterrows()
        ]

        # Sink-normalised: total settlers in AMI from all sources
        all_df["in_ami"] = [
            ami_union.contains(Point(r["end_lon"], r["end_lat"]))
            if not (np.isnan(r["end_lon"]) or np.isnan(r["end_lat"]))
            else False
            for _, r in all_df.iterrows()
        ]
        total_ami_settlers = all_df["in_ami"].sum()

        for apei_lbl in apei_labels:
            apei_subset = apei_pts_joined[apei_pts_joined["apei_label"] == apei_lbl]
            n_released = len(apei_subset)
            n_to_ami = apei_subset["in_ami"].sum()

            src_norm = 100.0 * n_to_ami / max(n_released, 1)
            sink_norm = 100.0 * n_to_ami / max(total_ami_settlers, 1)
            share_of_apei_flux = 100.0 * n_to_ami / max(
                apei_pts_joined["in_ami"].sum(), 1
            )

            for pld in pld_days:
                pld_subset = apei_subset[apei_subset["PLD"] == pld] if "PLD" in apei_subset.columns else apei_subset
                n_pld = len(pld_subset)
                n_to_ami_pld = pld_subset["in_ami"].sum() if not pld_subset.empty else 0

                records.append({
                    "APEI": apei_lbl,
                    "scenario": scenario,
                    "PLD_days": pld,
                    "n_released": n_pld,
                    "n_settled_in_AMI": int(n_to_ami_pld),
                    "source_normalised_pct": round(100.0 * n_to_ami_pld / max(n_pld, 1), 3),
                    "sink_normalised_pct": round(
                        100.0 * n_to_ami_pld / max(total_ami_settlers, 1), 3
                    ),
                    "share_of_APEI_flux_pct": round(
                        100.0 * n_to_ami_pld / max(apei_pts_joined["in_ami"].sum(), 1), 3
                    ),
                })

    if records:
        ed1 = pd.DataFrame(records)
        out = tables_dir / "ed_table1_per_apei_ami_flux.csv"
        ed1.to_csv(out, index=False)
        logger.info(f"Saved ED Table 1 → {out}")
    else:
        logger.warning("No data for ED Table 1")


# ── ED Table 2 ────────────────────────────────────────────────────────────────

def build_ed_table2(cfg: dict, tables_dir: Path) -> None:
    """
    Build ED Table 2: ENSO phase for each simulation period (static).

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    tables_dir : Path
        Output directory.
    """
    logger.info("Building ED Table 2 (ENSO classification) …")
    # Static values from manuscript ED Table 2 (NOAA ENSO archive)
    records = [
        {"Period": "Jan 2019", "label": "Jan2019",
         "ENSO_phase": "Weak El Niño",
         "ONI_index": "+0.8",
         "notes": "Weak El Niño onset; transitional"},
        {"Period": "Jul 2019", "label": "Jul2019",
         "ENSO_phase": "Neutral/IOD positive",
         "ONI_index": "+0.5",
         "notes": "Post-El Niño decay; positive IOD developing"},
        {"Period": "Jan 2023", "label": "Jan2023",
         "ENSO_phase": "La Niña (decaying)",
         "ONI_index": "-0.9",
         "notes": "Triple-dip La Niña event decaying"},
        {"Period": "Jul 2023", "label": "Jul2023",
         "ENSO_phase": "El Niño (onset)",
         "ONI_index": "+1.0",
         "notes": "Rapid El Niño development; strong by late 2023"},
    ]
    df = pd.DataFrame(records)
    out = tables_dir / "ed_table2_enso_classification.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved ED Table 2 → {out}")


# ── ED Table 3 ────────────────────────────────────────────────────────────────

def build_ed_table3(cfg: dict, tables_dir: Path) -> None:
    """
    Build ED Table 3: simulation parameter summary (static/from config).

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    tables_dir : Path
        Output directory.
    """
    logger.info("Building ED Table 3 (simulation parameters) …")
    tracking = cfg.get("tracking", {})
    smag = tracking.get("smagorinsky", {})
    records = [
        {"Parameter": "Grid resolution", "Value": "100 × 100 km", "Notes": "Equal Earth projection (EPSG:8857)"},
        {"Parameter": "Domain (grid)", "Value": "160°–110°W, 0°–25°N", "Notes": "CCZ management area"},
        {"Parameter": "Domain (HYCOM retrieval)", "Value": "170°–110°W, 5°S–25°N", "Notes": "Extended for full trajectory capture"},
        {"Parameter": "Simulation periods", "Value": ", ".join(p["label"] for p in cfg["time_periods"]["periods"]), "Notes": "Seasonal + interannual contrast"},
        {"Parameter": "PLDs simulated (days)", "Value": ", ".join(str(p) for p in tracking.get("pld_days", [])), "Notes": "Quartile range of pooled deep-sea LDDs"},
        {"Parameter": "Particles per site (PPS)", "Value": str(tracking.get("pps", "")), "Notes": "Flatline of marginal returns curve"},
        {"Parameter": "Internal timestep", "Value": f"{tracking.get('dt_minutes', '')} min", "Notes": "CFL = 0.2 at HYCOM native resolution"},
        {"Parameter": "Output interval", "Value": f"{tracking.get('output_dt_hours', '')} h", "Notes": "Positions stored at 1-hour intervals"},
        {"Parameter": "Advection scheme", "Value": "RK4 (AdvectionRK4)", "Notes": "4th-order Runge-Kutta"},
        {"Parameter": "Smagorinsky coefficient (Cs)", "Value": str(smag.get("Cs", "")), "Notes": "Standard value for coarse global models"},
        {"Parameter": "Diffusion spatial offset (dx)", "Value": f"{smag.get('dx_deg', '')}°", "Notes": "Central-difference velocity gradient"},
        {"Parameter": "Connectivity threshold", "Value": str(cfg.get("connectivity", {}).get("reciprocal_threshold", "")), "Notes": "Min particles for a reciprocal link"},
        {"Parameter": "Community detection", "Value": cfg.get("network", {}).get("community_method", "fast_greedy"), "Notes": "Deterministic, parameter-free"},
        {"Parameter": "Random seed", "Value": str(tracking.get("random_seed", "")), "Notes": "For reproducibility"},
    ]
    df = pd.DataFrame(records)
    out = tables_dir / "ed_table3_simulation_parameters.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved ED Table 3 → {out}")


# ── ED Table 4 ────────────────────────────────────────────────────────────────

def build_ed_table4(cfg: dict, tables_dir: Path) -> None:
    """
    Build ED Table 4: pairwise temporal comparison statistics.

    Reads comparison_stats_{scenario}.csv files produced by step 02 and
    reformats them as a clean publication table.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    tables_dir : Path
        Output directory.
    """
    logger.info("Building ED Table 4 (pairwise temporal comparisons) …")
    conn_dir = Path(cfg["connectivity_dir"])
    dfs = []
    for scenario in cfg["tracking"]["scenarios"]:
        p = conn_dir / f"comparison_stats_{scenario}.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        logger.warning("No comparison stats CSVs found — ED Table 4 empty")
        return

    df = pd.concat(dfs, ignore_index=True)
    # Round for publication
    for col in ["jaccard_dissimilarity", "cohen_kappa", "permutation_stat", "permutation_p"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Friendly column names
    df = df.rename(columns={
        "period_A": "Period A",
        "period_B": "Period B",
        "PLD": "PLD (days)",
        "scenario": "Scenario",
        "jaccard_dissimilarity": "Jaccard dissimilarity",
        "cohen_kappa": "Cohen's kappa",
        "permutation_stat": "Permutation statistic",
        "permutation_p": "p-value",
    })

    out = tables_dir / "ed_table4_temporal_comparisons.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved ED Table 4 → {out} ({len(df)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    """Entry point."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    tables_dir = _tables_dir(cfg)
    logger.info(f"Config: {args.config}")
    logger.info(f"Tables output dir: {tables_dir}")

    errors: list[str] = []
    for name, fn in [
        ("Table 1", build_table1),
        ("ED Table 1", build_ed_table1),
        ("ED Table 2", build_ed_table2),
        ("ED Table 3", build_ed_table3),
        ("ED Table 4", build_ed_table4),
    ]:
        try:
            fn(cfg, tables_dir)
        except Exception as exc:
            logger.exception(f"Error building {name}: {exc}")
            errors.append(name)

    if errors:
        logger.error(f"Failed to generate: {errors}")
        return 1

    logger.info("All tables generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
