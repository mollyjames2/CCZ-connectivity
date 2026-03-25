"""
icv.py — Integrated Connectivity Value (ICV) computation.

Combines three component scores — APEI support, mining replenishment, and
scenario-stable connectivity — into a single ICV for each CCZ polygon.
Component scores are normalised to [0, 1] before weighting.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


def _normalise_0_1(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    vmin, vmax = series.min(), series.max()
    if vmax == vmin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)


def compute_mining_replenishment(
    per_particle_df: pd.DataFrame,
    ami_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute mining replenishment scores via PCA on AMI-particle features.

    For each source polygon, three features are extracted:
    (1) total particles settling in mining areas,
    (2) number of distinct AMI areas reached,
    (3) percentage of surviving particles that settle in mining areas.

    These are then reduced to a single score via the first PCA component.

    Parameters
    ----------
    per_particle_df : pd.DataFrame
        Per-particle DataFrame from
        :func:`~ccz_connectivity.connectivity.build_connectivity`.
        Required columns: ``Particle_ID``, ``Start_polyID``, ``end_lat``,
        ``end_lon``.
    ami_gdf : gpd.GeoDataFrame
        AMI (Area of Particular Environmental Interest) polygon GeoDataFrame
        with a unique identifier column.

    Returns
    -------
    pd.DataFrame
        Columns: ``PolygonID``, ``mining_replenishment_score``.
    """
    if ami_gdf.crs is None or ami_gdf.crs.to_epsg() != 4326:
        ami_gdf = ami_gdf.to_crs("EPSG:4326")

    # Tag each particle with the AMI polygon it lands in (or None)
    from shapely.geometry import Point

    ami_gdf = ami_gdf.reset_index(drop=True)

    def _tag_ami(row: pd.Series) -> int:
        """Return AMI index or -1."""
        if np.isnan(row["end_lat"]) or np.isnan(row["end_lon"]):
            return -1
        pt = Point(row["end_lon"], row["end_lat"])
        hits = ami_gdf[ami_gdf.contains(pt)]
        if hits.empty:
            return -1
        return int(hits.index[0])

    logger.info("Tagging particles with AMI polygon membership …")
    per_particle_df = per_particle_df.copy()
    per_particle_df["ami_idx"] = per_particle_df.apply(_tag_ami, axis=1)

    feature_records = []
    for poly_id, grp in per_particle_df.groupby("Start_polyID"):
        n_total = len(grp)
        in_ami = grp[grp["ami_idx"] >= 0]
        n_in_ami = len(in_ami)
        n_areas = in_ami["ami_idx"].nunique()
        pct_in_ami = n_in_ami / n_total if n_total > 0 else 0.0
        feature_records.append({
            "PolygonID": poly_id,
            "particles_in_mining": n_in_ami,
            "num_areas_reached": n_areas,
            "pct_in_mining": pct_in_ami,
        })

    feature_df = pd.DataFrame(feature_records)
    if feature_df.empty:
        return pd.DataFrame(columns=["PolygonID", "mining_replenishment_score"])

    feature_cols = ["particles_in_mining", "num_areas_reached", "pct_in_mining"]
    X = feature_df[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    scores = pca.fit_transform(X_scaled)[:, 0]

    # Normalise so higher = more replenishment
    feature_df["mining_replenishment_score"] = _normalise_0_1(pd.Series(scores))
    logger.info(
        f"Mining replenishment PCA: explained variance ratio = "
        f"{pca.explained_variance_ratio_[0]:.3f}"
    )
    return feature_df[["PolygonID", "mining_replenishment_score"]]


def compute_scenario_stable(
    network_unmined: pd.DataFrame,
    network_mined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute scenario-stable connectivity scores.

    Integrates two structural indicators — out-degree (source strength) and
    node betweenness (stepping-stone importance) — from both the unmined and
    mined scenarios.  All four metrics are min-max normalised independently,
    and the final Si is the mean of the four normalised values (per
    manuscript Methods).

    Parameters
    ----------
    network_unmined : pd.DataFrame
        Network metrics for the unmined scenario (output of
        :func:`~ccz_connectivity.network.compute_network_metrics`).
        Required columns: ``polygon_ID``, ``out_degree``, ``node_betweenness``.
    network_mined : pd.DataFrame
        Network metrics for the mined scenario.

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_ID``, ``Scenario-Stable Connectivity Score``.
    """
    unmined = network_unmined[["polygon_ID", "out_degree", "node_betweenness"]].rename(
        columns={"out_degree": "outdeg_unmined", "node_betweenness": "btw_unmined"}
    )
    mined = network_mined[["polygon_ID", "out_degree", "node_betweenness"]].rename(
        columns={"out_degree": "outdeg_mined", "node_betweenness": "btw_mined"}
    )

    merged = unmined.merge(mined, on="polygon_ID", how="outer").fillna(0.0)

    # Normalise each metric independently, then take the mean of all four
    for col in ["outdeg_unmined", "btw_unmined", "outdeg_mined", "btw_mined"]:
        merged[f"{col}_norm"] = _normalise_0_1(merged[col])

    norm_cols = ["outdeg_unmined_norm", "btw_unmined_norm",
                 "outdeg_mined_norm", "btw_mined_norm"]
    merged["Scenario-Stable Connectivity Score"] = merged[norm_cols].mean(axis=1)

    return merged[["polygon_ID", "Scenario-Stable Connectivity Score"]]


def compute_icv(
    support_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    network_unmined_df: pd.DataFrame,
    network_mined_df: pd.DataFrame,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Combine component scores into the Integrated Connectivity Value (ICV).

    ICV = w1*Ai + w2*Mi + w3*Si + w4*Ti

    where:
    - Ai = normalised APEI support score
    - Mi = normalised mining replenishment score (PCA PC1)
    - Si = scenario-stable connectivity score (mean of 4 normalised metrics)
    - Ti = binary transitional indicator (1 if community shifts under mining, 0 otherwise)

    All weights are equal (0.25) by default.

    Parameters
    ----------
    support_df : pd.DataFrame
        APEI support scores.  Columns: ``polygonID``,
        ``normalised_support_score``.
    mining_df : pd.DataFrame
        Mining replenishment scores.  Columns: ``PolygonID``,
        ``mining_replenishment_score``.
    scenario_df : pd.DataFrame
        Scenario-stable scores.  Columns: ``polygon_ID``,
        ``Scenario-Stable Connectivity Score``.
    network_unmined_df : pd.DataFrame
        Unmined network metrics from
        :func:`~ccz_connectivity.network.compute_network_metrics`.
        Required columns: ``polygon_ID``, ``community``.
    network_mined_df : pd.DataFrame
        Mined network metrics.
        Required columns: ``polygon_ID``, ``community``.
    weights : list[float], optional
        Four weights summing to 1.0 (default [0.25, 0.25, 0.25, 0.25]).

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_ID``, ``normalised_support_score``,
        ``mining_replenishment_score``,
        ``Scenario-Stable Connectivity Score``,
        ``community``, ``Ti``, ``ICV``.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    if len(weights) != 4:
        raise ValueError("weights must have exactly 4 elements")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    # Normalise identifiers
    support = support_df.rename(columns={"polygonID": "polygon_ID"}).copy()
    support["polygon_ID"] = support["polygon_ID"].astype(int)

    mining = mining_df.rename(columns={"PolygonID": "polygon_ID"}).copy()
    mining["polygon_ID"] = mining["polygon_ID"].astype(int)

    scenario = scenario_df.copy()
    scenario["polygon_ID"] = scenario["polygon_ID"].astype(int)

    # Derive Ti: binary transitional indicator
    # Ti = 1 if community differs between unmined and mined scenarios, 0 otherwise
    comm_unmined = network_unmined_df[["polygon_ID", "community"]].copy()
    comm_unmined["polygon_ID"] = comm_unmined["polygon_ID"].astype(int)
    comm_unmined = comm_unmined.rename(columns={"community": "community_unmined"})

    comm_mined = network_mined_df[["polygon_ID", "community"]].copy()
    comm_mined["polygon_ID"] = comm_mined["polygon_ID"].astype(int)
    comm_mined = comm_mined.rename(columns={"community": "community_mined"})

    comm = comm_unmined.merge(comm_mined, on="polygon_ID", how="outer")
    # Fill NaN communities with sentinel so mismatches are flagged
    comm["community_unmined"] = comm["community_unmined"].fillna(-1).astype(str)
    comm["community_mined"] = comm["community_mined"].fillna(-2).astype(str)
    comm["Ti"] = (comm["community_unmined"] != comm["community_mined"]).astype(float)
    # Final community label: stable label or 'transitional'
    comm["community"] = comm.apply(
        lambda r: r["community_unmined"] if r["community_unmined"] == r["community_mined"]
        else "transitional",
        axis=1,
    )

    # Merge all components
    df = (
        support
        .merge(mining, on="polygon_ID", how="outer")
        .merge(scenario, on="polygon_ID", how="outer")
        .merge(comm[["polygon_ID", "community", "Ti"]], on="polygon_ID", how="outer")
        .fillna(0.0)
    )

    # Min-max normalise component scores
    df["normalised_support_score"] = _normalise_0_1(df["normalised_support_score"])
    df["mining_replenishment_score"] = _normalise_0_1(df["mining_replenishment_score"])
    df["Scenario-Stable Connectivity Score"] = _normalise_0_1(
        df["Scenario-Stable Connectivity Score"]
    )
    # Ti is already binary [0, 1]; no normalisation needed

    df["ICV"] = (
        weights[0] * df["normalised_support_score"]
        + weights[1] * df["mining_replenishment_score"]
        + weights[2] * df["Scenario-Stable Connectivity Score"]
        + weights[3] * df["Ti"]
    )

    logger.info(
        f"ICV computed for {len(df)} polygons. "
        f"Max={df['ICV'].max():.4f}, Mean={df['ICV'].mean():.4f}, "
        f"Transitional={int(df['Ti'].sum())} cells"
    )

    return df[[
        "polygon_ID",
        "normalised_support_score",
        "mining_replenishment_score",
        "Scenario-Stable Connectivity Score",
        "community",
        "Ti",
        "ICV",
    ]]
