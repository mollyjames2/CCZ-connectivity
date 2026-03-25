"""
network.py — Graph construction, community detection, and network metrics.

Builds weighted directed igraph networks from connectivity DataFrames,
detects communities using the fast_greedy algorithm on an undirected
projection, and computes per-node metrics including betweenness centrality,
degree, and self-recruitment rate.  Also provides APEI support score
computation and near-miss distance analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(
    connectivity_df: pd.DataFrame,
    threshold: int = 0,
) -> ig.Graph:
    """
    Build a weighted directed igraph from a connectivity DataFrame.

    Only edges whose ``Count`` exceeds ``threshold`` are included.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Must contain columns ``Start_polyID``, ``End_polyID``, ``Count``.
    threshold : int, optional
        Minimum particle count for an edge to be included (default 0).

    Returns
    -------
    ig.Graph
        Directed, weighted graph with vertex attribute ``name`` (polygon FID)
        and edge attribute ``weight`` (Count).
    """
    df = connectivity_df[connectivity_df["Count"] > threshold].copy()
    if df.empty:
        logger.warning("No edges survive the threshold — returning empty graph")
        return ig.Graph(directed=True)

    all_nodes = sorted(
        set(df["Start_polyID"].unique()) | set(df["End_polyID"].unique())
    )
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    edges = [
        (node_to_idx[r["Start_polyID"]], node_to_idx[r["End_polyID"]])
        for _, r in df.iterrows()
    ]
    weights = df["Count"].tolist()

    g = ig.Graph(n=len(all_nodes), edges=edges, directed=True)
    g.vs["name"] = all_nodes
    g.es["weight"] = weights
    logger.info(f"Graph built: {g.vcount()} nodes, {g.ecount()} edges")
    return g


# ── Community detection ───────────────────────────────────────────────────────

def detect_communities(g: ig.Graph) -> ig.VertexClustering:
    """
    Detect communities using fast_greedy on an undirected projection.

    The directed graph is converted to undirected (combining reciprocal
    edge weights by summing) before applying the fast_greedy algorithm.

    Parameters
    ----------
    g : ig.Graph
        Directed weighted graph from :func:`build_graph`.

    Returns
    -------
    ig.VertexClustering
        Community membership for each vertex.
    """
    if g.vcount() == 0:
        raise ValueError("Cannot detect communities on an empty graph")

    # Convert to undirected, summing weights of reciprocal edges
    g_und = g.as_undirected(combine_edges={"weight": "sum"})
    dendrogram = g_und.community_fastgreedy(weights="weight")
    clustering = dendrogram.as_clustering()
    logger.info(f"Detected {len(clustering)} communities")
    return clustering


# ── Network metrics ───────────────────────────────────────────────────────────

def compute_network_metrics(
    g: ig.Graph,
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute per-node network metrics and return as a DataFrame.

    Parameters
    ----------
    g : ig.Graph
        Directed weighted graph with vertex attribute ``name`` (polygon FID).
        Community detection is run internally.
    gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame with ``FID`` column, in EPSG:4326.

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_ID``, ``community``, ``centroid_lat``,
        ``centroid_lon``, ``out_degree``, ``in_degree``,
        ``node_betweenness``, ``self_recruitment``.
    """
    if g.vcount() == 0:
        return pd.DataFrame(columns=[
            "polygon_ID", "community", "centroid_lat", "centroid_lon",
            "out_degree", "in_degree", "node_betweenness", "self_recruitment",
        ])

    clustering = detect_communities(g)
    membership = clustering.membership

    # Betweenness (normalised)
    betweenness = g.betweenness(directed=True, weights="weight")
    n = g.vcount()
    norm = (n - 1) * (n - 2) if n > 2 else 1.0
    betweenness_norm = [b / norm for b in betweenness]

    # Degree
    out_deg = g.outdegree()
    in_deg = g.indegree()

    # Self-recruitment: weight of self-loop / total in-weight
    self_rec = []
    for v in g.vs:
        self_edges = g.es.select(_source=v.index, _target=v.index)
        self_w = sum(e["weight"] for e in self_edges)
        in_edges = g.es.select(_target=v.index)
        total_in = sum(e["weight"] for e in in_edges)
        self_rec.append(float(self_w / total_in) if total_in > 0 else 0.0)

    # Centroid lookup
    gdf_indexed = gdf.set_index("FID")
    records = []
    for v in g.vs:
        fid = v["name"]
        if fid in gdf_indexed.index:
            centroid = gdf_indexed.loc[fid, "geometry"].centroid
            clat, clon = centroid.y, centroid.x
        else:
            clat, clon = np.nan, np.nan

        records.append({
            "polygon_ID": fid,
            "community": membership[v.index],
            "centroid_lat": clat,
            "centroid_lon": clon,
            "out_degree": out_deg[v.index],
            "in_degree": in_deg[v.index],
            "node_betweenness": betweenness_norm[v.index],
            "self_recruitment": self_rec[v.index],
        })

    return pd.DataFrame(records)


# ── APEI support score ────────────────────────────────────────────────────────

def compute_apei_support(
    per_particle_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    apei_gdf: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Compute normalised APEI support scores for each CCZ polygon.

    The score integrates both the volume and diversity of larval exchange
    between each polygon and the APEI network, following the manuscript
    formula::

        Ai_raw = (S + R) × (1 + 2×min(S,R)/(S+R)) × (Ns + Nr)

    where S = particles sent from polygon to APEIs, R = particles received
    from APEIs, Ns = distinct APEIs reached, Nr = distinct APEIs that sent
    particles.  Polygons with S+R=0 score zero.  Final scores are
    min-max normalised to [0, 1].

    Parameters
    ----------
    per_particle_df : pd.DataFrame
        Per-particle DataFrame from
        :func:`~ccz_connectivity.connectivity.build_connectivity`.
        Required columns: ``Start_polyID``, ``End_polyID``.
    grid_gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame with ``FID`` column, in EPSG:4326.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame with a label column.
    config : dict
        Parsed YAML configuration (uses ``apei_label_field``).

    Returns
    -------
    pd.DataFrame
        Columns: ``polygonID``, ``normalised_support_score``.
    """
    label_field: str = config["shapefiles"]["apei_label_field"]
    if apei_gdf.crs is None or apei_gdf.crs.to_epsg() != 4326:
        apei_gdf = apei_gdf.to_crs("EPSG:4326")
    if grid_gdf.crs is None or grid_gdf.crs.to_epsg() != 4326:
        grid_gdf = grid_gdf.to_crs("EPSG:4326")

    # Map grid cell FIDs to APEI labels via spatial join
    joined = gpd.sjoin(
        grid_gdf[["FID", "geometry"]],
        apei_gdf[[label_field, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    # FID → APEI label
    fid_to_apei: dict[int, str] = (
        joined.drop_duplicates("FID")
        .set_index("FID")[label_field]
        .to_dict()
    )
    apei_fids: set[int] = set(fid_to_apei.keys())

    if not apei_fids:
        logger.warning("No grid cells intersect APEI polygons — support scores will be zero")

    all_poly_ids: list[int] = sorted(
        set(per_particle_df["Start_polyID"].unique())
        | set(per_particle_df["End_polyID"].unique())
    )

    records: list[dict] = []
    for poly_id in all_poly_ids:
        # Forward: particles from this polygon ending in an APEI cell
        fwd = per_particle_df[
            (per_particle_df["Start_polyID"] == poly_id)
            & (per_particle_df["End_polyID"].isin(apei_fids))
        ]
        S = len(fwd)
        Ns = fwd["End_polyID"].map(fid_to_apei).nunique() if S > 0 else 0

        # Reverse: particles from any APEI cell ending at this polygon
        rev = per_particle_df[
            (per_particle_df["Start_polyID"].isin(apei_fids))
            & (per_particle_df["End_polyID"] == poly_id)
        ]
        R = len(rev)
        Nr = rev["Start_polyID"].map(fid_to_apei).nunique() if R > 0 else 0

        total = S + R
        if total == 0:
            ai_raw = 0.0
        else:
            role_balance = 1.0 + 2.0 * min(S, R) / total
            ai_raw = float(total * role_balance * (Ns + Nr))

        records.append({"polygonID": int(poly_id), "ai_raw": ai_raw})

    support = pd.DataFrame(records)
    vmax = support["ai_raw"].max()
    support["normalised_support_score"] = (
        support["ai_raw"] / vmax if vmax > 0 else 0.0
    )
    logger.info(
        f"APEI support computed for {len(support)} polygons "
        f"(max raw={vmax:.1f})"
    )
    return support[["polygonID", "normalised_support_score"]]


# ── Near-miss distances ───────────────────────────────────────────────────────

def compute_near_miss_distances(
    apei_endpoints_df: pd.DataFrame,
    apei_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute distance from each near-miss particle endpoint to the nearest APEI.

    A "near-miss" particle is one that ends outside all APEIs but within
    a buffer zone.  This function computes the actual distance from each
    endpoint to the nearest APEI polygon boundary.

    Parameters
    ----------
    apei_endpoints_df : pd.DataFrame
        DataFrame with columns ``EndLat``, ``EndLon`` for particle endpoints
        that did *not* settle in an APEI (i.e. near-misses).
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional column ``dist_to_apei_km``.
    """
    if apei_gdf.crs is None or apei_gdf.crs.to_epsg() != 4326:
        apei_gdf = apei_gdf.to_crs("EPSG:4326")

    apei_union = apei_gdf.unary_union

    # Reproject to a metric CRS for accurate distance calculation
    # Use a equidistant cylindrical projection centred on the CCZ
    metric_crs = "EPSG:3857"  # Web Mercator; close enough for km-scale distances
    apei_metric = apei_gdf.to_crs(metric_crs)
    apei_union_metric = apei_metric.unary_union

    pts_gdf = gpd.GeoDataFrame(
        apei_endpoints_df.copy(),
        geometry=gpd.points_from_xy(
            apei_endpoints_df["EndLon"], apei_endpoints_df["EndLat"]
        ),
        crs="EPSG:4326",
    ).to_crs(metric_crs)

    pts_gdf["dist_to_apei_km"] = pts_gdf.geometry.distance(apei_union_metric) / 1000.0

    result = apei_endpoints_df.copy()
    result["dist_to_apei_km"] = pts_gdf["dist_to_apei_km"].values
    return result
