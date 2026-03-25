"""
optimisation.py — Two-phase corridor optimisation for CCZ connectivity.

Phase 1: Greedy ICV-ranked polygon selection ensuring every APEI polygon has
         at least one selected polygon connected to it via a reciprocal link
         above the particle-flow threshold.

Phase 2: Steiner-tree approximation via Dijkstra's algorithm. Routes weighted
         shortest paths between all APEI terminal pairs; corridor = union of
         nodes on any APEI-to-APEI shortest path.  Phase-1 nodes not on any
         such path are discarded as redundant (per manuscript Methods).

Edge weight = 1 / (particle_count + 1e-6) so stronger exchange = lower cost.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def identify_apei_polygon_ids(
    grid_gdf: gpd.GeoDataFrame,
    apei_gdf: gpd.GeoDataFrame,
) -> dict[int, str]:
    """
    Return a mapping of grid cell FID → APEI label for cells that intersect APEIs.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        CCZ grid with ``FID`` column.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame with a label column.

    Returns
    -------
    dict[int, str]
        ``{FID: apei_label}`` for every grid cell that intersects an APEI.
    """
    if apei_gdf.crs is None or apei_gdf.crs != grid_gdf.crs:
        apei_gdf = apei_gdf.to_crs(grid_gdf.crs)

    # Detect label column (first non-geometry string column)
    label_col = next(
        (c for c in apei_gdf.columns if c != "geometry"), None
    )
    joined = gpd.sjoin(
        grid_gdf[["FID", "geometry"]],
        apei_gdf[[label_col, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    return (
        joined.drop_duplicates("FID")
        .set_index("FID")[label_col]
        .to_dict()
    )


def _build_reciprocal_graph(
    connectivity_df: pd.DataFrame,
    threshold: int,
) -> tuple[dict[int, dict[int, float]], set[tuple[int, int]]]:
    """
    Build an adjacency dict for reciprocal links above threshold.

    Edge weight = 1 / (count + 1e-6).

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Connectivity data with columns ``Start_polyID``, ``End_polyID``, ``Count``.
    threshold : int
        Minimum particle count in *each direction* for a reciprocal link.

    Returns
    -------
    tuple[dict, set]
        ``(adjacency, reciprocal_pairs)`` where adjacency maps
        ``node → {neighbour: weight}`` and reciprocal_pairs is the set of
        ``(min_id, max_id)`` pairs with valid reciprocal links.
    """
    # Index by directed pair — sum counts across any duplicate (src, dst) rows
    # (e.g. same edge appearing for multiple PLDs in a combined-PLD dataframe)
    count_map: dict[tuple[int, int], float] = {}
    for _, r in connectivity_df.iterrows():
        key = (int(r["Start_polyID"]), int(r["End_polyID"]))
        count_map[key] = count_map.get(key, 0.0) + float(r["Count"])

    adjacency: dict[int, dict[int, float]] = {}
    reciprocal_pairs: set[tuple[int, int]] = set()

    for (src, dst), count_fwd in count_map.items():
        count_rev = count_map.get((dst, src), 0.0)
        if count_fwd >= threshold and count_rev >= threshold:
            weight = 1.0 / (min(count_fwd, count_rev) + 1e-6)
            adjacency.setdefault(src, {})[dst] = weight
            adjacency.setdefault(dst, {})[src] = weight
            reciprocal_pairs.add((min(src, dst), max(src, dst)))

    return adjacency, reciprocal_pairs


def _dijkstra(
    adjacency: dict[int, dict[int, float]],
    source: int,
) -> dict[int, tuple[float, list[int]]]:
    """
    Dijkstra's algorithm from ``source``.

    Parameters
    ----------
    adjacency : dict[int, dict[int, float]]
        Undirected weighted adjacency dict.
    source : int
        Source node ID.

    Returns
    -------
    dict[int, tuple[float, list[int]]]
        Mapping ``node → (cost, path)`` for all reachable nodes.
    """
    import heapq

    dist: dict[int, float] = {source: 0.0}
    prev: dict[int, int | None] = {source: None}
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist.get(u, float("inf")):
            continue
        for v, w in adjacency.get(u, {}).items():
            alt = cost + w
            if alt < dist.get(v, float("inf")):
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    # Reconstruct paths
    result: dict[int, tuple[float, list[int]]] = {}
    for node, d in dist.items():
        path: list[int] = []
        cur: int | None = node
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        result[node] = (d, path)

    return result


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1_greedy_icv(
    icv_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    apei_gdf: gpd.GeoDataFrame,
    threshold: int,
) -> list[int]:
    """
    Phase 1: greedy ICV-ranked selection until all APEIs are connected.

    Ranks all non-APEI polygons by ICV (descending) and adds each one to the
    selection if it has a reciprocal link ≥ ``threshold`` to any APEI polygon
    not yet covered.  Stops when every APEI polygon ID is connected to ≥ 1
    selected polygon.

    Parameters
    ----------
    icv_df : pd.DataFrame
        ICV DataFrame with columns ``polygon_ID``, ``ICV``.
    connectivity_df : pd.DataFrame
        Connectivity DataFrame with ``Start_polyID``, ``End_polyID``, ``Count``.
    grid_gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame with ``FID`` column (for APEI identification).
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame.
    threshold : int
        Minimum particle count in each direction for a reciprocal link.

    Returns
    -------
    list[int]
        Ordered list of selected polygon FIDs (highest ICV first).
    """
    # Identify APEI grid-cell FIDs
    apei_cell_map = identify_apei_polygon_ids(grid_gdf, apei_gdf)
    apei_fids: set[int] = set(apei_cell_map.keys())

    if not apei_fids:
        logger.warning("No grid cells map to APEI polygons — Phase 1 cannot run")
        return []

    adjacency, _ = _build_reciprocal_graph(connectivity_df, threshold)

    # For each APEI cell, which non-APEI polygons are reciprocally connected to it?
    apei_to_supporters: dict[int, set[int]] = {fid: set() for fid in apei_fids}
    for node, neighbours in adjacency.items():
        if node not in apei_fids:
            for nbr in neighbours:
                if nbr in apei_fids:
                    apei_to_supporters[nbr].add(node)

    covered_apeis: set[int] = set()  # APEI FIDs that now have ≥1 selected supporter
    selected: list[int] = []

    # Sort candidates (non-APEI polygons) by ICV descending
    candidates = (
        icv_df[~icv_df["polygon_ID"].isin(apei_fids)]
        .sort_values("ICV", ascending=False)["polygon_ID"]
        .tolist()
    )

    for poly_id in candidates:
        if covered_apeis >= apei_fids:
            break  # all APEIs covered
        # Check if this polygon supports any uncovered APEI
        supported = {
            apei_fid
            for apei_fid, supporters in apei_to_supporters.items()
            if poly_id in supporters and apei_fid not in covered_apeis
        }
        if supported:
            selected.append(int(poly_id))
            covered_apeis |= supported

    n_uncovered = len(apei_fids - covered_apeis)
    logger.info(
        f"Phase 1: {len(selected)} polygons selected; "
        f"{len(covered_apeis)}/{len(apei_fids)} APEIs covered "
        f"({n_uncovered} uncovered)"
    )
    return selected


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_steiner_refinement(
    connectivity_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    apei_gdf: gpd.GeoDataFrame,
    threshold: int,
) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """
    Phase 2: Dijkstra-based Steiner refinement between APEI terminals.

    Constructs the reciprocal dispersal graph, then computes weighted shortest
    paths between all pairs of APEI terminal nodes.  The final corridor is the
    union of all nodes that appear on at least one APEI-to-APEI shortest path.
    Phase-1 nodes that lie on no shortest path are discarded.

    Edge weight = 1 / (particle_count + 1e-6), so higher-flux paths have
    lower cost and are preferentially selected.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Connectivity DataFrame with ``Start_polyID``, ``End_polyID``, ``Count``.
    grid_gdf : gpd.GeoDataFrame
        CCZ grid GeoDataFrame with ``FID`` column.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame.
    threshold : int
        Minimum particle count in each direction for a reciprocal link.

    Returns
    -------
    tuple[list[int], pd.DataFrame, pd.DataFrame]
        ``(final_nodes, corridor_edges_df, apei_paths_df)``

        corridor_edges_df columns: ``src``, ``dst``, ``flow``
        apei_paths_df columns: ``src``, ``dst``, ``path``
    """
    adjacency, _ = _build_reciprocal_graph(connectivity_df, threshold)

    apei_cell_map = identify_apei_polygon_ids(grid_gdf, apei_gdf)
    apei_fids: list[int] = sorted(
        fid for fid in apei_cell_map if fid in adjacency
    )

    empty_edges = pd.DataFrame(columns=["src", "dst", "flow"])
    empty_paths = pd.DataFrame(columns=["src", "dst", "path"])

    if len(apei_fids) < 2:
        logger.warning("Fewer than 2 APEI terminals in graph — Phase 2 skipped")
        return [], empty_edges, empty_paths

    final_nodes: set[int] = set()
    corridor_edges: list[dict] = []
    apei_paths_records: list[dict] = []

    # Step 1: Dijkstra from each APEI terminal → all-pairs shortest paths
    all_paths: dict[int, dict[int, tuple[float, list[int]]]] = {}
    for src in apei_fids:
        reachable = _dijkstra(adjacency, src)
        all_paths[src] = {
            dst: (cost, path)
            for dst, (cost, path) in reachable.items()
            if dst in apei_fids and dst != src and len(path) >= 2
        }

    # Step 2: Prim's MST on the terminal metric-closure graph
    # (standard 2-approximation for Steiner tree)
    import heapq as _hq

    mst_edges: list[tuple[int, int]] = []
    in_tree: set[int] = {apei_fids[0]}
    heap: list[tuple[float, int, int]] = []
    for dst in apei_fids[1:]:
        if dst in all_paths.get(apei_fids[0], {}):
            cost, _ = all_paths[apei_fids[0]][dst]
            _hq.heappush(heap, (cost, apei_fids[0], dst))

    while heap and len(in_tree) < len(apei_fids):
        cost, u, v = _hq.heappop(heap)
        if v in in_tree:
            continue
        mst_edges.append((u, v))
        in_tree.add(v)
        for w in apei_fids:
            if w not in in_tree and w in all_paths.get(v, {}):
                c, _ = all_paths[v][w]
                _hq.heappush(heap, (c, v, w))

    # Step 3: Collect corridor nodes only from the N-1 MST paths
    for u, v in mst_edges:
        _, path = all_paths[u][v]
        final_nodes |= set(path)
        apei_paths_records.append({
            "src": u,
            "dst": v,
            "path": ">".join(str(n) for n in path),
        })
        for k in range(len(path) - 1):
            corridor_edges.append({
                "src": path[k],
                "dst": path[k + 1],
                "flow": 1,
            })

    # Aggregate corridor edges
    if corridor_edges:
        corridor_df = (
            pd.DataFrame(corridor_edges)
            .groupby(["src", "dst"])["flow"]
            .sum()
            .reset_index()
        )
    else:
        corridor_df = empty_edges

    apei_paths_df = pd.DataFrame(apei_paths_records) if apei_paths_records else empty_paths

    final_list = sorted(final_nodes)
    logger.info(
        f"Phase 2: {len(final_list)} corridor nodes from "
        f"{len(apei_fids)} APEI terminals via {len(mst_edges)}-edge MST "
        f"({len(apei_paths_records)} paths)"
    )
    return final_list, corridor_df, apei_paths_df


# ── Entry point ───────────────────────────────────────────────────────────────

def run_optimisation(
    icv_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    apei_gdf: gpd.GeoDataFrame,
    config: dict,
    grid_gdf: gpd.GeoDataFrame | None = None,
) -> dict:
    """
    Run the full two-phase corridor optimisation and return all results.

    Parameters
    ----------
    icv_df : pd.DataFrame
        ICV DataFrame (output of :func:`~ccz_connectivity.icv.compute_icv`).
    connectivity_df : pd.DataFrame
        Aggregated connectivity DataFrame.
    apei_gdf : gpd.GeoDataFrame
        APEI polygon GeoDataFrame.
    config : dict
        Parsed YAML configuration.  Uses
        ``config["connectivity"]["reciprocal_threshold"]``.
    grid_gdf : gpd.GeoDataFrame, optional
        CCZ grid GeoDataFrame.  Required for APEI cell identification.
        If ``None``, loaded from ``config["shapefiles"]["ccz_grid"]``.

    Returns
    -------
    dict
        Keys:
        ``"phase1_selection"`` (list[int]),
        ``"final_selection"`` (list[int]),
        ``"corridor_edges"`` (pd.DataFrame),
        ``"apei_paths"`` (pd.DataFrame).
    """
    if grid_gdf is None:
        grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])

    threshold: int = config["connectivity"]["reciprocal_threshold"]

    logger.info(f"Phase 1: greedy ICV selection (threshold={threshold}) …")
    phase1 = phase1_greedy_icv(
        icv_df, connectivity_df, grid_gdf, apei_gdf, threshold
    )

    logger.info("Phase 2: Steiner refinement between APEI terminals …")
    final_nodes, corridor_edges, apei_paths = phase2_steiner_refinement(
        connectivity_df, grid_gdf, apei_gdf, threshold
    )

    return {
        "phase1_selection": phase1,
        "final_selection": final_nodes,
        "corridor_edges": corridor_edges,
        "apei_paths": apei_paths,
    }
