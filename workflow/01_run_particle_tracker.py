#!/usr/bin/env python3
"""
01_run_particle_tracker.py — Run OceanParcels particle tracking.

Reads the CCZ grid centroids, seeds particles at each site, and runs a
forward particle-tracking simulation using AdvectionRK4 with Smagorinsky
diffusion and distance-accumulation kernels.  Output is written as a
Zarr store compatible with OceanParcels 3.0+ schema.

Usage
-----
    python workflow/01_run_particle_tracker.py \
        --config config/config.yaml \
        --time-period 0 \
        --scenario unmined

Arguments
---------
--config         Path to YAML configuration file.
--time-period    Zero-based index into time_periods.periods (default 0).
--scenario       Scenario label: unmined or mined (default unmined).
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OceanParcels particle tracking for CCZ connectivity"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--time-period",
        type=int,
        default=0,
        dest="time_period",
        help="Zero-based time period index (default: 0)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="unmined",
        choices=["unmined", "mined"],
        help="Scenario: unmined or mined (default: unmined)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the particle tracking runner."""
    args = parse_args()

    cfg = yaml.safe_load(open(args.config))

    periods = cfg["time_periods"]["periods"]
    if args.time_period >= len(periods):
        logger.error(
            f"Time period index {args.time_period} out of range "
            f"(0–{len(periods) - 1})"
        )
        return 1

    period_label = periods[args.time_period]["label"]
    logger.info(
        f"Particle tracking: config={args.config}, "
        f"period={period_label} (index {args.time_period}), "
        f"scenario={args.scenario}"
    )

    # Import here so import errors are reported cleanly
    try:
        from ccz_connectivity.tracking import run_tracking
    except ImportError as exc:
        logger.error(f"Cannot import ccz_connectivity.tracking: {exc}")
        logger.error("Ensure the ccz-connectivity conda environment is active.")
        return 1

    zarr_path = run_tracking(cfg, args.time_period, args.scenario)
    logger.info(f"Output Zarr store: {zarr_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
