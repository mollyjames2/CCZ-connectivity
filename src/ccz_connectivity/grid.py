"""
grid.py — Build and load the CCZ equal-area grid.

Creates a 100 km × 100 km grid in the Equal Earth projection (EPSG:8857),
clips it to the CCZ boundary shapefile, and exports both a polygon shapefile
and a centroids CSV for use by the particle-tracking and connectivity modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

logger = logging.getLogger(__name__)


def make_ccz_grid(config: dict) -> gpd.GeoDataFrame:
    """
    Create a regular equal-area grid over the CCZ domain and clip to boundary.

    Grid cells are square in the Equal Earth projection (EPSG:8857) at the
    resolution specified in ``config["grid"]["resolution_km"]``.  The result
    is reprojected to WGS84 (EPSG:4326) and clipped to the CCZ boundary
    shapefile.  A sequential integer ``FID`` column is added.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary.  Required keys:
        ``config["grid"]["resolution_km"]``,
        ``config["grid"]["crs_equal_area"]``,
        ``config["shapefiles"]["ccz_boundary"]``.

    Returns
    -------
    gpd.GeoDataFrame
        Polygon GeoDataFrame in EPSG:4326 with columns:
        ``FID`` (int), ``geometry`` (Polygon).
    """
    res_km: float = config["grid"]["resolution_km"]
    res_m: float = res_km * 1_000.0
    crs_ea: str = config["grid"]["crs_equal_area"]
    crs_data: str = config["grid"]["crs_data"]
    boundary_path: Path = Path(config["shapefiles"]["ccz_boundary"])

    logger.info(f"Loading CCZ boundary from {boundary_path}")
    ccz_boundary = gpd.read_file(boundary_path).to_crs(crs_ea)

    xmin, ymin, xmax, ymax = ccz_boundary.total_bounds
    logger.info(
        f"Projected bounds (m): x=[{xmin:.0f}, {xmax:.0f}], y=[{ymin:.0f}, {ymax:.0f}]"
    )

    xs = np.arange(xmin, xmax, res_m)
    ys = np.arange(ymin, ymax, res_m)
    logger.info(f"Creating {len(xs)} × {len(ys)} = {len(xs) * len(ys)} candidate cells")

    cells = [
        box(x, y, x + res_m, y + res_m) for y in ys for x in xs
    ]
    grid_ea = gpd.GeoDataFrame(geometry=cells, crs=crs_ea)

    # Clip to CCZ boundary
    logger.info("Clipping grid to CCZ boundary …")
    grid_clipped = gpd.clip(grid_ea, ccz_boundary)
    grid_clipped = grid_clipped[~grid_clipped.is_empty].reset_index(drop=True)
    grid_clipped["FID"] = grid_clipped.index.astype(int)

    # Reproject to WGS84 for output
    grid_wgs84 = grid_clipped.to_crs(crs_data)
    logger.info(f"Grid contains {len(grid_wgs84)} cells after clipping")
    return grid_wgs84


def load_centroids(path: Path) -> pd.DataFrame:
    """
    Load a centroids CSV file produced by :func:`save_grid`.

    Parameters
    ----------
    path : Path
        Path to a CSV with columns ``polyid``, ``lon``, ``lat``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``polyid`` (int), ``lon`` (float),
        ``lat`` (float).
    """
    path = Path(path)
    logger.debug(f"Loading centroids from {path}")
    df = pd.read_csv(path)
    required = {"polyid", "lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Centroids CSV missing columns: {missing}")
    df["polyid"] = df["polyid"].astype(int)
    return df


def save_grid(
    gdf: gpd.GeoDataFrame,
    out_shp: Path,
    out_centroids: Path,
) -> None:
    """
    Save the grid GeoDataFrame as a shapefile and a centroids CSV.

    The centroids are computed in WGS84 from the polygon geometry.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Grid GeoDataFrame with at least ``FID`` and ``geometry`` columns,
        in EPSG:4326.
    out_shp : Path
        Output path for the polygon shapefile.
    out_centroids : Path
        Output path for the centroids CSV (columns: ``polyid``, ``lon``,
        ``lat``).
    """
    out_shp = Path(out_shp)
    out_centroids = Path(out_centroids)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    out_centroids.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving grid shapefile → {out_shp}")
    gdf.to_file(out_shp)

    centroids = gdf.copy()
    centroids["lon"] = centroids.geometry.centroid.x
    centroids["lat"] = centroids.geometry.centroid.y
    centroids_df = centroids[["FID", "lon", "lat"]].rename(columns={"FID": "polyid"})
    logger.info(f"Saving centroids CSV → {out_centroids}")
    centroids_df.to_csv(out_centroids, index=False)
