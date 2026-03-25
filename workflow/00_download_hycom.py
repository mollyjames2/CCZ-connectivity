#!/usr/bin/env python3
"""
00_download_hycom.py — Download HYCOM bottom-layer u/v via OPeNDAP.

Downloads water_u_bottom and water_v_bottom for each configured time period
from the HYCOM OPeNDAP server and saves as individual NetCDF files.

Usage
-----
    python workflow/00_download_hycom.py --config config/config.yaml

Output
------
    <hycom_data_dir>/water_u_bottom_<label>.nc
    <hycom_data_dir>/water_v_bottom_<label>.nc
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download HYCOM bottom u/v for CCZ connectivity model"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Zero-based period index to download (omit for all)",
    )
    return parser.parse_args()


def lon_to_hycom(lon: float) -> float:
    """
    Convert WGS84 longitude (−180 to 180) to HYCOM 0–360 convention.

    Parameters
    ----------
    lon : float
        Longitude in WGS84 convention.

    Returns
    -------
    float
        Longitude in 0–360 convention.
    """
    return lon % 360.0


def download_period(
    opendap_url: str,
    period: dict,
    hycom_cfg: dict,
    variables: list[str],
    out_dir: Path,
) -> None:
    """
    Download HYCOM bottom-layer current data for one time period via OPeNDAP.

    Subsets by longitude, latitude, and time, extracts only the bottom
    depth level, converts longitudes back to WGS84 (−180 to 180), and
    writes one NetCDF file per variable.

    Parameters
    ----------
    opendap_url : str
        HYCOM OPeNDAP dataset URL.
    period : dict
        Time period dict with keys ``label``, ``start``, ``end``.
    hycom_cfg : dict
        HYCOM configuration block from YAML.
    variables : list[str]
        Variable names to download (e.g. ``["water_u_bottom", "water_v_bottom"]``).
    out_dir : Path
        Directory to write NetCDF files into.
    """
    label: str = period["label"]
    start: str = period["start"]
    end: str = period["end"]
    west: float = float(hycom_cfg["west_hycom"])
    east: float = float(hycom_cfg["east_hycom"])
    south: float = float(hycom_cfg["south"])
    north: float = float(hycom_cfg["north"])

    logger.info(f"Downloading period {label}: {start} → {end}")
    logger.info(f"  Spatial: lon=[{west}, {east}]°E, lat=[{south}, {north}]°N")

    try:
        logger.info(f"  Opening OPeNDAP store: {opendap_url}")
        ds = xr.open_dataset(opendap_url, engine="netcdf4")
    except Exception as exc:
        logger.error(f"Failed to open OPeNDAP dataset: {exc}")
        raise

    # Subset time
    ds_time = ds.sel(time=slice(start, end))
    if len(ds_time.time) == 0:
        logger.error(f"No timesteps found for {start} → {end} — check dataset availability")
        ds.close()
        return

    # Subset longitude (HYCOM 0–360 convention)
    if west < east:
        ds_sub = ds_time.sel(
            lon=slice(west, east),
            lat=slice(south, north),
        )
    else:
        # Wrap-around case (shouldn't occur for CCZ but handle gracefully)
        ds_west = ds_time.sel(lon=slice(west, 360.0), lat=slice(south, north))
        ds_east = ds_time.sel(lon=slice(0.0, east), lat=slice(south, north))
        ds_sub = xr.concat([ds_west, ds_east], dim="lon")

    for var in variables:
        if var not in ds_sub:
            logger.warning(f"Variable {var} not found in dataset — skipping")
            continue

        logger.info(f"  Extracting variable: {var}")
        da = ds_sub[var]

        # If there is a depth dimension, select the bottom level
        depth_dims = [d for d in da.dims if d in ("depth", "lev", "level", "z")]
        if depth_dims:
            depth_dim = depth_dims[0]
            da = da.isel({depth_dim: -1})  # bottom = last index
            logger.info(f"    Selected bottom level (index -1 along '{depth_dim}')")

        # Convert longitudes from 0–360 to −180–180
        lons = da.lon.values.copy()
        lons[lons > 180.0] -= 360.0
        sort_idx = np.argsort(lons)
        da = da.assign_coords(lon=("lon", lons)).isel(lon=sort_idx)

        out_path = out_dir / f"{var}_{label}.nc"
        logger.info(f"  Writing → {out_path}")
        da.to_dataset(name=var).to_netcdf(out_path)
        logger.info(f"  Saved {var} for {label}: shape {da.shape}")

    ds.close()
    logger.info(f"Period {label} complete.")


def main() -> int:
    """Entry point for HYCOM download script."""
    args = parse_args()

    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg["hycom_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    opendap_url: str = cfg["hycom"]["opendap_url"]
    hycom_cfg: dict = cfg["hycom"]
    variables: list[str] = cfg["hycom"]["variables"]
    periods: list[dict] = cfg["time_periods"]["periods"]

    if args.period is not None:
        if args.period >= len(periods):
            logger.error(
                f"Period index {args.period} out of range (0–{len(periods) - 1})"
            )
            return 1
        periods = [periods[args.period]]

    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Periods to download: {[p['label'] for p in periods]}")

    for period in periods:
        try:
            download_period(opendap_url, period, hycom_cfg, variables, out_dir)
        except Exception as exc:
            logger.error(f"Failed to download period {period['label']}: {exc}")
            return 1

    logger.info("All periods downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
