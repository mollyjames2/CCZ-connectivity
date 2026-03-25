"""
connectivity.py — Build connectivity matrices from OceanParcels Zarr output.

Reads particle trajectories, spatially joins start/end positions to the CCZ
grid, and produces per-PLD connectivity CSV files.  Also computes APEI
endpoint statistics and pairwise comparison metrics (Jaccard, Cohen's kappa,
permutation tests) between time-period connectivity matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from scipy.stats import contingency
from shapely.geometry import Point

logger = logging.getLogger(__name__)


# ── Spatial helpers ───────────────────────────────────────────────────────────

def prepare_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure the grid GeoDataFrame is in WGS84 and has a spatial index.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input grid GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame reprojected to EPSG:4326 with spatial index built.
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.copy().reset_index(drop=True)
    _ = gdf.sindex  # build spatial index
    return gdf


def join_points_to_polygons(
    lons: np.ndarray,
    lats: np.ndarray,
    gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Spatial join of point coordinates to polygon FIDs.

    Uses a vectorised sjoin for performance.  Points that do not fall within
    any polygon are assigned ``-1``.

    Parameters
    ----------
    lons : np.ndarray
        Array of longitudes, shape (n,).
    lats : np.ndarray
        Array of latitudes, shape (n,).
    gdf : gpd.GeoDataFrame
        Polygon GeoDataFrame with ``FID`` column, in EPSG:4326.

    Returns
    -------
    np.ndarray
        Integer array of FIDs, shape (n,).  ``-1`` where no match.
    """
    pts = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(lons, lats)},
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, gdf[["FID", "geometry"]], how="left", predicate="within")
    # sjoin may produce duplicates if a point touches a shared boundary
    joined = joined[~joined.index.duplicated(keep="first")]
    result = np.full(len(lons), -1, dtype=np.int64)
    valid = joined["FID"].notna()
    result[joined.index[valid]] = joined.loc[valid, "FID"].astype(int).values
    return result


def compute_pld_time_indices(pld_days: list[int], dt_seconds: float) -> list[int]:
    """
    Convert PLD values in days to output timestep indices.

    Used only for OceanParcels continuous-output schema.

    Parameters
    ----------
    pld_days : list[int]
        PLDs in days.
    dt_seconds : float
        Output timestep in seconds (e.g. 3600 for hourly output).

    Returns
    -------
    list[int]
        List of integer time indices corresponding to each PLD.
    """
    return [int(round(pld * 86_400 / dt_seconds)) for pld in pld_days]


def _detect_snapshot_schema(store: zarr.Group) -> bool:
    """
    Return True if the Zarr store uses the snapshot schema (``release_lon`` present).

    The snapshot schema stores only positions at fixed PLD endpoints plus
    ``release_lon`` / ``release_lat`` centroid coordinates.  The OceanParcels
    continuous schema stores hourly trajectories without ``release_lon``.

    Parameters
    ----------
    store : zarr.Group
        Open Zarr group.

    Returns
    -------
    bool
        True for snapshot schema, False for OceanParcels continuous schema.
    """
    return "release_lon" in store


# ── Core connectivity builder ─────────────────────────────────────────────────

def build_connectivity(
    zarr_path: Path,
    gdf: gpd.GeoDataFrame,
    config: dict,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build connectivity matrices from a particle-tracking Zarr store.

    Supports two Zarr schemas:

    **Snapshot schema** (``release_lon`` present):
        Variables ``release_lon``, ``release_lat`` (n_particles,) give the
        exact release centroids.  ``lon`` / ``lat`` have shape
        (n_particles, n_pld) with one column per PLD in ``time`` (days).

    **OceanParcels continuous schema** (no ``release_lon``):
        ``lon`` / ``lat`` have shape (n_particles, n_timesteps) with
        continuous hourly (or sub-hourly) output.  Start position is taken
        from column 0; PLD end positions are looked up by converting PLD
        days to timestep indices using ``output_dt_hours`` from config.

    For each PLD, counts particles that started in polygon A and ended in
    polygon B.  Also returns a per-particle record.

    Parameters
    ----------
    zarr_path : Path
        Path to Zarr store.
    gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame with ``FID`` column, in EPSG:4326.
    config : dict
        Parsed YAML configuration.
    scenario : str
        Scenario label (``"unmined"`` or ``"mined"``), added as a column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(connectivity_df, per_particle_df)``

        connectivity_df columns:
            ``Start_polyID``, ``End_polyID``, ``Count``,
            ``PLD``, ``scenario``

        per_particle_df columns:
            ``Particle_ID``, ``Start_polyID``, ``End_polyID``,
            ``start_lat``, ``start_lon``, ``end_lat``, ``end_lon``,
            ``PLD``, ``scenario``
    """
    zarr_path = Path(zarr_path)
    logger.info(f"Opening Zarr store: {zarr_path}")
    store = zarr.open(str(zarr_path), mode="r")

    snapshot_schema = _detect_snapshot_schema(store)
    pld_days: list[int] = config["tracking"]["pld_days"]

    lat_arr = store["lat"][:]
    lon_arr = store["lon"][:]
    traj_ids = store["trajectory"][:]
    n_particles = len(traj_ids)

    if snapshot_schema:
        # ── Snapshot schema ──────────────────────────────────────────────────
        start_lons = store["release_lon"][:]
        start_lats = store["release_lat"][:]
        pld_times = store["time"][:].tolist()   # e.g. [19, 35, 69] (days)
        logger.info(
            f"Snapshot schema detected: {n_particles} particles, "
            f"PLD snapshots {pld_times} d"
        )
        # Map each config PLD to its column index in the snapshot arrays
        pld_index_map: dict[int, int] = {int(d): i for i, d in enumerate(pld_times)}

        def _get_pld_index(pld: int) -> int | None:
            idx = pld_index_map.get(pld)
            if idx is None:
                logger.warning(
                    f"PLD {pld} d not found in zarr time array {pld_times} — skipping"
                )
            return idx

    else:
        # ── OceanParcels continuous schema ────────────────────────────────────
        n_timesteps = lon_arr.shape[1]
        logger.info(
            f"OceanParcels schema detected: {n_particles} particles × "
            f"{n_timesteps} timesteps"
        )
        start_lons = lon_arr[:, 0]
        start_lats = lat_arr[:, 0]
        dt_output_s: float = config["tracking"]["output_dt_hours"] * 3600.0
        hour_indices = compute_pld_time_indices(pld_days, dt_output_s)

        def _get_pld_index(pld: int) -> int | None:  # type: ignore[misc]
            t_idx = hour_indices[pld_days.index(pld)]
            if t_idx >= n_timesteps:
                logger.warning(
                    f"PLD {pld} d requires index {t_idx} but only "
                    f"{n_timesteps} timesteps available — skipping"
                )
                return None
            return t_idx

    gdf = prepare_polygons(gdf)

    logger.info("Joining start positions to polygons …")
    start_poly = join_points_to_polygons(start_lons, start_lats, gdf)

    conn_records: list[dict] = []
    particle_records: list[dict] = []

    for pld in pld_days:
        t_idx = _get_pld_index(pld)
        if t_idx is None:
            continue

        end_lons = lon_arr[:, t_idx]
        end_lats = lat_arr[:, t_idx]

        # Mask out deleted particles (NaN positions)
        valid = ~(np.isnan(end_lons) | np.isnan(end_lats))
        logger.info(f"PLD {pld} d: {valid.sum()} / {n_particles} particles survived")

        logger.info(f"PLD {pld} d: joining end positions …")
        end_poly = np.full(n_particles, -1, dtype=np.int64)
        end_poly[valid] = join_points_to_polygons(
            end_lons[valid], end_lats[valid], gdf
        )

        # Per-particle records
        for i in np.where(valid)[0]:
            particle_records.append({
                "Particle_ID": int(traj_ids[i]),
                "Start_polyID": int(start_poly[i]),
                "End_polyID": int(end_poly[i]),
                "start_lat": float(start_lats[i]),
                "start_lon": float(start_lons[i]),
                "end_lat": float(end_lats[i]),
                "end_lon": float(end_lons[i]),
                "PLD": pld,
                "scenario": scenario,
            })

        # Aggregate connectivity counts
        valid_both = valid & (start_poly >= 0) & (end_poly >= 0)
        conn_pairs = pd.DataFrame({
            "Start_polyID": start_poly[valid_both].astype(int),
            "End_polyID": end_poly[valid_both].astype(int),
        })
        conn_agg = (
            conn_pairs.groupby(["Start_polyID", "End_polyID"])
            .size()
            .reset_index(name="Count")
        )
        conn_agg["PLD"] = pld
        conn_agg["scenario"] = scenario
        conn_records.append(conn_agg)
        logger.info(f"PLD {pld} d: {len(conn_agg)} connectivity pairs")

    connectivity_df = pd.concat(conn_records, ignore_index=True) if conn_records else pd.DataFrame()
    per_particle_df = pd.DataFrame(particle_records)

    return connectivity_df, per_particle_df


# ── Origin category tagging ───────────────────────────────────────────────────

def tag_particle_categories(
    per_particle_df: pd.DataFrame,
    apei_gdf: gpd.GeoDataFrame,
    ami_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Add a ``start_category`` column classifying each particle's release origin.

    Categories (per manuscript):
    - ``"APEI"`` — released from within an APEI polygon
    - ``"AMI"`` — released from within an AMI polygon (but not APEI)
    - ``"unprotected"`` — released outside both APEIs and AMIs

    Uses start lat/lon coordinates for point-in-polygon classification.

    Parameters
    ----------
    per_particle_df : pd.DataFrame
        Per-particle DataFrame from :func:`build_connectivity`.
        Required columns: ``start_lat``, ``start_lon``.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame.
    ami_gdf : gpd.GeoDataFrame
        AMI polygon GeoDataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional column ``start_category``.
    """
    for gdf_name, gdf in [("APEI", apei_gdf), ("AMI", ami_gdf)]:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

    apei_union = apei_gdf.to_crs("EPSG:4326").unary_union
    ami_union = ami_gdf.to_crs("EPSG:4326").unary_union

    start_pts = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy(
            per_particle_df["start_lon"], per_particle_df["start_lat"]
        )},
        index=per_particle_df.index,
        crs="EPSG:4326",
    )

    in_apei = start_pts.geometry.within(apei_union)
    in_ami = start_pts.geometry.within(ami_union)

    categories = np.where(
        in_apei, "APEI",
        np.where(in_ami, "AMI", "unprotected")
    )

    result = per_particle_df.copy()
    result["start_category"] = categories
    return result


# ── APEI endpoint analysis ────────────────────────────────────────────────────

def compute_apei_endpoints(
    per_particle_df: pd.DataFrame,
    apei_gdf: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Find particles that start inside APEIs and record where they disperse to.

    For each particle released within an APEI polygon, records the
    APEI label, start polygon, end coordinates, and PLD.

    Parameters
    ----------
    per_particle_df : pd.DataFrame
        Per-particle output from :func:`build_connectivity`.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame with a label column.
    config : dict
        Parsed YAML configuration.  Uses
        ``config["shapefiles"]["apei_label_field"]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``StartAPEI``, ``EndLat``, ``EndLon``, ``PLD``,
        ``Start_polyID``, ``scenario``.
    """
    label_field: str = config["shapefiles"]["apei_label_field"]
    if apei_gdf.crs is None or apei_gdf.crs.to_epsg() != 4326:
        apei_gdf = apei_gdf.to_crs("EPSG:4326")

    records: list[dict] = []
    for _, apei_row in apei_gdf.iterrows():
        apei_label = apei_row[label_field]
        poly = apei_row.geometry
        in_apei = per_particle_df.apply(
            lambda r: poly.contains(Point(r["start_lon"], r["start_lat"])), axis=1
        )
        subset = per_particle_df[in_apei]
        for _, row in subset.iterrows():
            records.append({
                "StartAPEI": apei_label,
                "EndLat": row["end_lat"],
                "EndLon": row["end_lon"],
                "PLD": row["PLD"],
                "Start_polyID": row["Start_polyID"],
                "scenario": row["scenario"],
            })

    if records:
        return pd.DataFrame(records)
    return pd.DataFrame(columns=["StartAPEI", "EndLat", "EndLon", "PLD", "Start_polyID", "scenario"])


# ── Comparison statistics ─────────────────────────────────────────────────────

def jaccard_dissimilarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Jaccard dissimilarity between two binary connectivity vectors.

    Treats any non-zero value as a connection present.

    Parameters
    ----------
    a : np.ndarray
        First binary (or count) array, shape (n,).
    b : np.ndarray
        Second binary (or count) array, shape (n,).

    Returns
    -------
    float
        Jaccard dissimilarity in [0, 1].  0 = identical, 1 = no overlap.
    """
    a_bin = (a > 0).astype(bool)
    b_bin = (b > 0).astype(bool)
    intersection = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)
    if union == 0:
        return 0.0
    return float(1.0 - intersection / union)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's kappa between two binary agreement vectors.

    Parameters
    ----------
    a : np.ndarray
        First binary array, shape (n,).
    b : np.ndarray
        Second binary array, shape (n,).

    Returns
    -------
    float
        Cohen's kappa coefficient.
    """
    a_bin = (a > 0).astype(int)
    b_bin = (b > 0).astype(int)

    n = len(a_bin)
    if n == 0:
        return 0.0

    # Confusion matrix elements
    tp = np.sum((a_bin == 1) & (b_bin == 1))
    tn = np.sum((a_bin == 0) & (b_bin == 0))
    fp = np.sum((a_bin == 0) & (b_bin == 1))
    fn = np.sum((a_bin == 1) & (b_bin == 0))

    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n ** 2)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perms: int = 9999,
    seed: int = 123,
) -> tuple[float, float]:
    """
    Permutation test for the difference in mean between two arrays.

    Parameters
    ----------
    a : np.ndarray
        First sample array.
    b : np.ndarray
        Second sample array.
    n_perms : int, optional
        Number of permutations (default 9999).
    seed : int, optional
        Random seed for reproducibility (default 123).

    Returns
    -------
    tuple[float, float]
        ``(observed_statistic, p_value)`` where the statistic is
        ``mean(a) - mean(b)``.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)

    count_extreme = 0
    for _ in range(n_perms):
        rng.shuffle(combined)
        perm_stat = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perms + 1)
    return observed, float(p_value)
