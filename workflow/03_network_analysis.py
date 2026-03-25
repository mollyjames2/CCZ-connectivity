#!/usr/bin/env python3
"""
03_network_analysis.py — Build igraph networks and compute connectivity metrics.

Loads aggregated connectivity CSVs for both scenarios, builds directed
weighted igraph networks, detects communities (fast_greedy), computes
per-node network metrics, APEI support scores, mining replenishment scores
(Part A), and scenario-stable connectivity scores.

Usage
-----
    python workflow/03_network_analysis.py --config config/config.yaml
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
        description="Network analysis for CCZ larval connectivity"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    conn_dir = Path(cfg["connectivity_dir"])
    network_dir = Path(cfg["network_dir"])
    network_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {args.config}")
    logger.info(f"Connectivity dir: {conn_dir}")
    logger.info(f"Network output dir: {network_dir}")

    from ccz_connectivity.network import (
        build_graph,
        detect_communities,
        compute_network_metrics,
        compute_apei_support,
        compute_near_miss_distances,
    )
    from ccz_connectivity.icv import (
        compute_mining_replenishment,
        compute_scenario_stable,
        compute_icv,
    )

    grid_gdf = gpd.read_file(cfg["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(cfg["shapefiles"]["apei"])
    ami_gdf = gpd.read_file(cfg["shapefiles"]["ami"])

    reciprocal_threshold: int = cfg["connectivity"]["reciprocal_threshold"]
    pld_days: list[int] = cfg["tracking"]["pld_days"]
    scenarios: list[str] = cfg["tracking"]["scenarios"]

    network_metrics: dict[str, pd.DataFrame] = {}

    # ── Per-scenario network analysis ─────────────────────────────────────────
    for scenario in scenarios:
        logger.info(f"=== Network analysis: scenario={scenario} ===")

        agg_path = conn_dir / f"connectivity_aggregated_{scenario}.csv"
        if not agg_path.exists():
            logger.warning(f"Aggregated connectivity not found: {agg_path} — skipping")
            continue

        conn_df = pd.read_csv(agg_path)
        logger.info(f"Loaded {len(conn_df)} connectivity pairs for scenario {scenario}")

        # Aggregate connectivity across all PLDs and all periods, then build
        # one combined graph — consistent with the manuscript analysis which
        # treats 19, 35, and 69 d PLDs as a single pooled dispersal signal.
        logger.info(
            f"  Aggregating connectivity across PLDs {pld_days} "
            f"({len(conn_df)} pairs before aggregation) …"
        )
        conn_combined = (
            conn_df.groupby(["Start_polyID", "End_polyID"])["Count"]
            .sum()
            .reset_index()
        )
        logger.info(
            f"  Building combined graph ({len(conn_combined)} pairs, "
            f"threshold={reciprocal_threshold}) …"
        )
        g = build_graph(conn_combined, threshold=reciprocal_threshold)
        metrics_final = compute_network_metrics(g, grid_gdf)

        if metrics_final.empty:
            logger.warning(f"No network metrics produced for scenario {scenario}")
            continue

        out_path = network_dir / f"network_metrics_{scenario}.csv"
        metrics_final.to_csv(out_path, index=False)
        logger.info(f"Saved network metrics → {out_path}")
        network_metrics[scenario] = metrics_final

        # ── APEI support scores ────────────────────────────────────────────────
        logger.info(f"  Computing APEI support scores for {scenario} …")
        particles_path = conn_dir / f"per_particle_{scenario}.csv"
        if particles_path.exists():
            particle_df = pd.read_csv(particles_path)
            support_df = compute_apei_support(particle_df, grid_gdf, apei_gdf, cfg)
            support_out = network_dir / f"apei_support_{scenario}.csv"
            support_df.to_csv(support_out, index=False)
            logger.info(f"  Saved APEI support → {support_out}")
        else:
            logger.warning(f"  Per-particle CSV not found: {particles_path}")
            particle_df = pd.DataFrame()

        # ── Near-miss distances ────────────────────────────────────────────────
        apei_endpoints_path = conn_dir / f"apei_endpoints_{scenario}.csv"
        if apei_endpoints_path.exists():
            apei_endpoints = pd.read_csv(apei_endpoints_path)
            # Near-miss: particles from outside APEIs approaching but not settling
            # Here we compute distance of all endpoint records to nearest APEI
            near_miss = compute_near_miss_distances(apei_endpoints, apei_gdf)
            near_miss_out = network_dir / f"near_miss_distances_{scenario}.csv"
            near_miss.to_csv(near_miss_out, index=False)
            logger.info(f"  Saved near-miss distances → {near_miss_out}")

    # ── Mining replenishment (Part A — unmined scenario) ──────────────────────
    if "unmined" in cfg["tracking"]["scenarios"]:
        logger.info("Computing mining replenishment scores …")
        particles_path = conn_dir / "per_particle_unmined.csv"
        if particles_path.exists():
            particle_df = pd.read_csv(particles_path)
            mining_df = compute_mining_replenishment(particle_df, ami_gdf)
            mining_out = network_dir / "mining_replenishment.csv"
            mining_df.to_csv(mining_out, index=False)
            logger.info(f"Saved mining replenishment → {mining_out}")
        else:
            logger.warning("Per-particle unmined CSV not found — skipping mining replenishment")
            mining_df = pd.DataFrame(columns=["PolygonID", "mining_replenishment_score"])
    else:
        mining_df = pd.DataFrame(columns=["PolygonID", "mining_replenishment_score"])

    # ── Scenario-stable connectivity ──────────────────────────────────────────
    if "unmined" in network_metrics and "mined" in network_metrics:
        logger.info("Computing scenario-stable connectivity scores …")
        scenario_df = compute_scenario_stable(
            network_metrics["unmined"], network_metrics["mined"]
        )
        scenario_out = network_dir / "scenario_stable_connectivity.csv"
        scenario_df.to_csv(scenario_out, index=False)
        logger.info(f"Saved scenario-stable scores → {scenario_out}")
    else:
        logger.warning(
            "Cannot compute scenario-stable scores without both scenarios — "
            "using zeros"
        )
        scenario_df = pd.DataFrame(
            columns=["polygon_ID", "Scenario-Stable Connectivity Score"]
        )

    # ── ICV ────────────────────────────────────────────────────────────────────
    logger.info("Computing Integrated Connectivity Value (ICV) …")

    support_path = network_dir / "apei_support_unmined.csv"
    support_df = (
        pd.read_csv(support_path)
        if support_path.exists()
        else pd.DataFrame(columns=["polygonID", "normalised_support_score"])
    )

    weights: list[float] = cfg["icv"]["weights"]
    icv_df = compute_icv(
        support_df=support_df,
        mining_df=mining_df,
        scenario_df=scenario_df,
        network_unmined_df=network_metrics.get(
            "unmined",
            pd.DataFrame(columns=["polygon_ID", "node_betweenness", "community"]),
        ),
        network_mined_df=network_metrics.get(
            "mined",
            pd.DataFrame(columns=["polygon_ID", "node_betweenness", "community"]),
        ),
        weights=weights,
    )
    icv_out = network_dir / "icv_scores.csv"
    icv_df.to_csv(icv_out, index=False)
    logger.info(f"Saved ICV scores → {icv_out} ({len(icv_df)} polygons)")

    logger.info("Network analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
