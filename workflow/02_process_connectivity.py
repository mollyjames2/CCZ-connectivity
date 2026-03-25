#!/usr/bin/env python3
"""
02_process_connectivity.py — Build connectivity matrices from particle tracking output.

Reads OceanParcels Zarr stores for all time periods and both scenarios,
builds per-PLD connectivity matrices and per-particle records, computes
APEI endpoint analyses, and produces pairwise Jaccard/Cohen's kappa
comparisons between time periods.

Usage
-----
    python workflow/02_process_connectivity.py \
        --config config/config.yaml \
        --scenario both

Arguments
---------
--config    Path to YAML configuration file.
--scenario  Scenario to process: unmined, mined, or both (default: both).
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
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
        description="Build CCZ connectivity matrices from particle trajectories"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="both",
        choices=["unmined", "mined", "both"],
        help="Scenario to process (default: both)",
    )
    return parser.parse_args()


def process_scenario(cfg: dict, scenario: str) -> None:
    """
    Process all time periods for one scenario.

    Iterates over all configured time periods, builds connectivity matrices
    and per-particle records for each Zarr store, aggregates across time
    periods, and saves CSVs.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    scenario : str
        Scenario label: ``"unmined"`` or ``"mined"``.
    """
    from ccz_connectivity.connectivity import (
        build_connectivity,
        tag_particle_categories,
        compute_apei_endpoints,
        jaccard_dissimilarity,
        cohen_kappa,
        permutation_test,
    )

    particles_dir = Path(cfg["particles_dir"])
    conn_dir = Path(cfg["connectivity_dir"])
    conn_dir.mkdir(parents=True, exist_ok=True)

    periods = cfg["time_periods"]["periods"]
    pld_days: list[int] = cfg["tracking"]["pld_days"]

    grid_gdf = gpd.read_file(cfg["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(cfg["shapefiles"]["apei"])
    ami_gdf = gpd.read_file(cfg["shapefiles"]["ami"])

    all_conn_dfs: list[pd.DataFrame] = []
    all_particle_dfs: list[pd.DataFrame] = []
    period_conn: dict[str, pd.DataFrame] = {}  # label -> connectivity_df

    for period in periods:
        label = period["label"]
        zarr_path = particles_dir / f"particles_{label}_{scenario}.zarr"

        if not zarr_path.exists():
            logger.warning(f"Zarr store not found: {zarr_path} — skipping period {label}")
            continue

        logger.info(f"Processing period {label}, scenario {scenario} …")
        conn_df, particle_df = build_connectivity(zarr_path, grid_gdf, cfg, scenario)
        conn_df["time_period"] = label
        particle_df["time_period"] = label

        all_conn_dfs.append(conn_df)
        all_particle_dfs.append(particle_df)
        period_conn[label] = conn_df

        # Save per-period files
        for pld in pld_days:
            pld_conn = conn_df[conn_df["PLD"] == pld]
            out = conn_dir / f"connectivity_pld{pld}_{label}_{scenario}.csv"
            pld_conn.to_csv(out, index=False)
            logger.info(f"  Saved {out.name}: {len(pld_conn)} pairs")

    if not all_conn_dfs:
        logger.error(f"No connectivity data produced for scenario {scenario}")
        return

    # Aggregated across all time periods
    agg_conn = (
        pd.concat(all_conn_dfs, ignore_index=True)
        .groupby(["Start_polyID", "End_polyID", "PLD", "scenario"])["Count"]
        .sum()
        .reset_index()
    )
    agg_path = conn_dir / f"connectivity_aggregated_{scenario}.csv"
    agg_conn.to_csv(agg_path, index=False)
    logger.info(f"Saved aggregated connectivity: {agg_path}")

    all_particles = pd.concat(all_particle_dfs, ignore_index=True)

    # Tag each particle's release origin as APEI, AMI, or unprotected
    logger.info("Tagging particle origin categories …")
    all_particles = tag_particle_categories(all_particles, apei_gdf, ami_gdf)

    particles_out = conn_dir / f"per_particle_{scenario}.csv"
    all_particles.to_csv(particles_out, index=False)
    logger.info(f"Saved per-particle records: {particles_out}")

    # Per-PLD per-particle files
    for pld in pld_days:
        pld_particles = all_particles[all_particles["PLD"] == pld]
        out = conn_dir / f"per_particle_pld{pld}_{scenario}.csv"
        pld_particles.to_csv(out, index=False)

    # APEI endpoint analysis
    logger.info("Computing APEI endpoint analysis …")
    apei_df = compute_apei_endpoints(all_particles, apei_gdf, cfg)
    apei_out = conn_dir / f"apei_endpoints_{scenario}.csv"
    apei_df.to_csv(apei_out, index=False)
    logger.info(f"Saved APEI endpoints: {apei_out} ({len(apei_df)} records)")

    # Per-PLD APEI endpoints
    for pld in pld_days:
        pld_apei = apei_df[apei_df["PLD"] == pld] if not apei_df.empty else apei_df
        out = conn_dir / f"apei_endpoints_pld{pld}.csv"
        pld_apei.to_csv(out, index=False)

    # Pairwise comparison statistics between time periods
    logger.info("Computing pairwise comparison statistics …")
    _compute_comparisons(period_conn, pld_days, conn_dir, scenario)


def _compute_comparisons(
    period_conn: dict[str, pd.DataFrame],
    pld_days: list[int],
    conn_dir: Path,
    scenario: str,
) -> None:
    """
    Compute Jaccard and Cohen's kappa between all time-period pairs.

    Parameters
    ----------
    period_conn : dict[str, pd.DataFrame]
        Mapping of period label → connectivity DataFrame.
    pld_days : list[int]
        PLD values to compare.
    conn_dir : Path
        Output directory.
    scenario : str
        Scenario label for output filenames.
    """
    from ccz_connectivity.connectivity import jaccard_dissimilarity, cohen_kappa, permutation_test

    labels = list(period_conn.keys())
    records = []

    for pld in pld_days:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                df_a = period_conn[la][period_conn[la]["PLD"] == pld]
                df_b = period_conn[lb][period_conn[lb]["PLD"] == pld]

                # Build common edge index
                all_pairs = set(
                    zip(df_a["Start_polyID"], df_a["End_polyID"])
                ) | set(
                    zip(df_b["Start_polyID"], df_b["End_polyID"])
                )
                pairs = sorted(all_pairs)
                if not pairs:
                    continue

                a_map = df_a.set_index(["Start_polyID", "End_polyID"])["Count"].to_dict()
                b_map = df_b.set_index(["Start_polyID", "End_polyID"])["Count"].to_dict()

                vec_a = np.array([a_map.get(p, 0) for p in pairs], dtype=float)
                vec_b = np.array([b_map.get(p, 0) for p in pairs], dtype=float)

                jac = jaccard_dissimilarity(vec_a, vec_b)
                kap = cohen_kappa(vec_a, vec_b)
                obs, pval = permutation_test(vec_a, vec_b)

                records.append({
                    "period_A": la,
                    "period_B": lb,
                    "PLD": pld,
                    "scenario": scenario,
                    "jaccard_dissimilarity": jac,
                    "cohen_kappa": kap,
                    "permutation_stat": obs,
                    "permutation_p": pval,
                })

    if records:
        comp_df = pd.DataFrame(records)
        out = conn_dir / f"comparison_stats_{scenario}.csv"
        comp_df.to_csv(out, index=False)
        logger.info(f"Saved comparison statistics: {out}")


def main() -> int:
    """Entry point."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    scenarios: list[str] = (
        cfg["tracking"]["scenarios"]
        if args.scenario == "both"
        else [args.scenario]
    )

    logger.info(f"Config: {args.config}")
    logger.info(f"Scenarios: {scenarios}")

    for scenario in scenarios:
        logger.info(f"=== Processing scenario: {scenario} ===")
        try:
            process_scenario(cfg, scenario)
        except Exception as exc:
            logger.exception(f"Error processing scenario {scenario}: {exc}")
            return 1

    logger.info("Connectivity processing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
