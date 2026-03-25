#!/usr/bin/env python3
"""
05_generate_figures.py — Generate all publication figures.

Calls the plotting functions from ccz_connectivity.plotting in the
requested order.  Figures are saved to <figures_dir>/ at the DPI
configured in the YAML.

Usage
-----
    python workflow/05_generate_figures.py --config config/config.yaml
    python workflow/05_generate_figures.py --config config/config.yaml --figures 1 3 5
    python workflow/05_generate_figures.py --config config/config.yaml --figures ed

Arguments
---------
--config    Path to YAML configuration file.
--figures   Space-separated list of figure numbers/labels to generate.
            Use integers (1–5) for main figures, "ed" for all extended-data
            figures, or omit to generate all.
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
        description="Generate CCZ connectivity publication figures"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help=(
            "Figures to generate: 1, 2, 3, 4, 5, ed, or all "
            "(default: all)"
        ),
    )
    return parser.parse_args()


_FIGURE_MAP = {
    "1": "plot_fig1",
    "2": "plot_fig2",
    "3": "plot_fig3",
    "4": "plot_fig4",
    "5": "plot_fig5",
    "ed": "plot_ed_figs",
}


def main() -> int:
    """Entry point."""
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {args.config}")
    logger.info(f"Figures output dir: {figures_dir}")

    # Resolve which figures to generate
    if "all" in args.figures:
        requested = list(_FIGURE_MAP.keys())
    else:
        requested = []
        for f in args.figures:
            f = str(f).strip()
            if f not in _FIGURE_MAP:
                logger.warning(
                    f"Unknown figure identifier '{f}' — valid options: "
                    f"{list(_FIGURE_MAP.keys())}"
                )
            else:
                requested.append(f)

    if not requested:
        logger.error("No valid figures to generate")
        return 1

    logger.info(f"Generating figures: {requested}")

    try:
        import ccz_connectivity.plotting as plotting
    except ImportError as exc:
        logger.error(f"Cannot import ccz_connectivity.plotting: {exc}")
        return 1

    errors: list[str] = []
    for fig_id in requested:
        func_name = _FIGURE_MAP[fig_id]
        func = getattr(plotting, func_name, None)
        if func is None:
            logger.error(f"Plotting function '{func_name}' not found")
            errors.append(fig_id)
            continue

        logger.info(f"--- Generating figure {fig_id} ({func_name}) ---")
        try:
            func(cfg)
        except Exception as exc:
            logger.exception(f"Error generating figure {fig_id}: {exc}")
            errors.append(fig_id)

    if errors:
        logger.error(f"Failed to generate figures: {errors}")
        return 1

    logger.info("All requested figures generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
