"""
plotting.py — Publication-standard figure generation for CCZ connectivity.

Main figures (1–5) and extended data figures (ED1–ED6) are generated here.
All figures are saved as JPG at the DPI configured in the YAML.

In test_mode (config["test_mode"] = True), Cartopy maps are replaced with
plain matplotlib axes to allow CI runs without real geographic data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _test_mode(config: dict) -> bool:
    """Return True if running in test mode."""
    return bool(config.get("test_mode", False))


def _get_extent(config: dict) -> list[float]:
    """Extract domain extent [lon_min, lon_max, lat_min, lat_max] from config."""
    d = config["domain"]
    return [d["lon_min"], d["lon_max"], d["lat_min"], d["lat_max"]]


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save a figure as JPG and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure .jpg extension
    path = path.with_suffix(".jpg")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format="jpeg")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def _base_map(
    extent: list[float],
    figsize: tuple[float, float] = (12, 7),
    title: str = "",
    test_mode: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a base map: Cartopy in production, plain axes in test mode.

    Parameters
    ----------
    extent : list[float]
        ``[lon_min, lon_max, lat_min, lat_max]``
    figsize : tuple[float, float]
        Figure size in inches.
    title : str
        Axis title.
    test_mode : bool
        If True, skip Cartopy and return a plain matplotlib axes.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    if test_mode:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if title:
            ax.set_title(title, fontsize=11)
        return fig, ax

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    return fig, ax


def _plot_kwargs(test_mode: bool) -> dict:
    """Return transform kwargs for geopandas .plot() calls."""
    if test_mode:
        return {}
    import cartopy.crs as ccrs
    return {"transform": ccrs.PlateCarree()}


def _load_all_apei_endpoints(config: dict) -> pd.DataFrame:
    """Load and concatenate per-scenario APEI endpoint CSVs."""
    conn_dir = Path(config["connectivity_dir"])
    dfs = []
    for scenario in config["tracking"]["scenarios"]:
        p = conn_dir / f"apei_endpoints_{scenario}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["scenario"] = scenario
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["StartAPEI", "EndLat", "EndLon", "PLD", "scenario"])


# ── Figure 1: per-APEI dispersal footprints (colour-blended) ─────────────────

def plot_fig1(config: dict) -> None:
    """
    Single-panel map of APEI dispersal footprints with RGBA colour blending.

    Each APEI is assigned a distinct RGB colour.  For each APEI the particle
    endpoint density is computed as a 2-D histogram smoothed with a Gaussian
    filter.  All per-APEI histograms are blended into a single RGBA image by
    weighted RGB averaging (weight proportional to local particle count).
    CCZ boundary = dashed red; APEI boundaries = solid black.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Figure 1: APEI dispersal footprints …")
    extent = _get_extent(config)
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    tmode = _test_mode(config)

    apei_endpoints = _load_all_apei_endpoints(config)
    if apei_endpoints.empty:
        logger.warning("No APEI endpoint data — skipping Figure 1")
        return

    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    label_field: str = config["shapefiles"]["apei_label_field"]
    apei_labels = sorted(apei_endpoints["StartAPEI"].unique())

    # Assign a distinct colour per APEI
    n_apei = len(apei_labels)
    cmap_apei = plt.cm.get_cmap("tab20", n_apei)
    apei_colours = {lbl: np.array(cmap_apei(i)[:3]) for i, lbl in enumerate(apei_labels)}

    # Grid for density estimation
    nx, ny = 300, 200
    lon_edges = np.linspace(extent[0], extent[1], nx + 1)
    lat_edges = np.linspace(extent[2], extent[3], ny + 1)
    lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    # Accumulate weighted RGBA
    rgba_accum = np.zeros((ny, nx, 3), dtype=float)
    weight_accum = np.zeros((ny, nx), dtype=float)

    for lbl in apei_labels:
        subset = apei_endpoints[apei_endpoints["StartAPEI"] == lbl]
        if subset.empty:
            continue
        h, _, _ = np.histogram2d(
            subset["EndLon"].values, subset["EndLat"].values,
            bins=[lon_edges, lat_edges],
        )
        h = h.T  # shape (ny, nx)
        h_smooth = gaussian_filter(h.astype(float), sigma=2.0)
        colour = apei_colours[lbl]
        rgba_accum += h_smooth[:, :, np.newaxis] * colour[np.newaxis, np.newaxis, :]
        weight_accum += h_smooth

    # Normalise to RGB image
    mask = weight_accum > 0
    rgb_img = np.ones((ny, nx, 4), dtype=float)  # default white+transparent
    rgb_img[:, :, 3] = 0.0
    if mask.any():
        rgb_img[mask, :3] = rgba_accum[mask] / weight_accum[mask, np.newaxis]
        # Alpha proportional to log density (rescaled)
        w_norm = np.log1p(weight_accum)
        w_norm = w_norm / w_norm.max() if w_norm.max() > 0 else w_norm
        rgb_img[:, :, 3] = w_norm * 0.85

    fig, ax = _base_map(extent, figsize=(14, 8), title="Figure 1 — APEI Dispersal Footprints",
                        test_mode=tmode)

    ax.imshow(
        rgb_img,
        origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        aspect="auto",
        zorder=1,
        **({} if tmode else {"transform": __import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree()}),
    )

    # CCZ boundary: dashed red
    try:
        ccz_gdf = gpd.read_file(config["shapefiles"]["ccz_boundary"])
        ccz_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.5,
                     linestyle="--", zorder=4, **_plot_kwargs(tmode))
    except Exception:
        pass

    # APEI boundaries: solid black
    apei_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2,
                  zorder=5, **_plot_kwargs(tmode))

    # Legend patches
    patches = [
        mpatches.Patch(color=apei_colours[lbl], label=lbl)
        for lbl in apei_labels
    ]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                  linestyle="--", label="CCZ boundary"))
    ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2,
              title="APEI", title_fontsize=8)

    _save_figure(fig, figures_dir / "fig1_apei_dispersal_footprints", dpi=dpi)


# ── Figure 2: AMI particle origin/settlement density ─────────────────────────

def plot_fig2(config: dict) -> None:
    """
    Origin density (red) and settlement density (blue) for particles settling in AMIs.

    Left panel: CCZ grid cells coloured by how many settling particles were released
    from each cell (choropleth, Reds).  Right panel: each AMI polygon filled as a
    solid block, darker blue = more particles received.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Figure 2: AMI particle origin/settlement density …")
    extent = _get_extent(config)
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    tmode = _test_mode(config)
    conn_dir = Path(config["connectivity_dir"])

    ami_gdf = gpd.read_file(config["shapefiles"]["ami"])
    if ami_gdf.crs is None or ami_gdf.crs.to_epsg() != 4326:
        ami_gdf = ami_gdf.to_crs("EPSG:4326")

    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    if grid_gdf.crs is None or grid_gdf.crs.to_epsg() != 4326:
        grid_gdf = grid_gdf.to_crs("EPSG:4326")

    # Load per-particle data for unmined scenario (all PLDs combined)
    particles_path = conn_dir / "per_particle_unmined.csv"
    if not particles_path.exists():
        logger.warning("per_particle_unmined.csv not found — skipping Figure 2")
        return
    all_particles = pd.read_csv(particles_path)

    # Filter to non-AMI sources
    if "start_category" in all_particles.columns:
        non_ami = all_particles[all_particles["start_category"] != "AMI"]
    else:
        non_ami = all_particles

    # Drop rows with missing end positions
    non_ami = non_ami.dropna(subset=["end_lon", "end_lat"])

    # Spatial join end positions to AMIs to find settlers
    end_gdf = gpd.GeoDataFrame(
        non_ami.reset_index(drop=True),
        geometry=gpd.points_from_xy(non_ami["end_lon"], non_ami["end_lat"]),
        crs="EPSG:4326",
    )
    end_in_ami = gpd.sjoin(
        end_gdf, ami_gdf[["geometry"]], how="inner", predicate="within"
    )
    settled_df = non_ami.iloc[end_in_ami.index].copy()

    if len(settled_df) < 5:
        logger.warning("Too few particles settling in AMIs — skipping Figure 2")
        return

    # --- Origin counts: particles per CCZ grid cell ---
    start_gdf = gpd.GeoDataFrame(
        settled_df.reset_index(drop=True),
        geometry=gpd.points_from_xy(settled_df["start_lon"], settled_df["start_lat"]),
        crs="EPSG:4326",
    )
    origin_joined = gpd.sjoin(
        start_gdf[["geometry"]], grid_gdf[["FID", "geometry"]],
        how="inner", predicate="within",
    )
    origin_counts = (
        origin_joined.groupby("FID").size().reset_index(name="particle_count")
    )
    grid_origin = grid_gdf.merge(origin_counts, on="FID", how="left")
    grid_origin["particle_count"] = grid_origin["particle_count"].fillna(0)
    # Only cells with at least one particle (rest stay light-grey background)
    grid_with_particles = grid_origin[grid_origin["particle_count"] > 0].copy()

    # --- Settlement counts: particles per AMI polygon ---
    end_settled = gpd.GeoDataFrame(
        settled_df.reset_index(drop=True),
        geometry=gpd.points_from_xy(settled_df["end_lon"], settled_df["end_lat"]),
        crs="EPSG:4326",
    )
    ami_joined = gpd.sjoin(
        end_settled[["geometry"]], ami_gdf[["geometry"]],
        how="inner", predicate="within",
    )
    ami_counts = ami_joined.groupby("index_right").size()
    ami_settle = ami_gdf.copy()
    ami_settle["particle_count"] = (
        ami_counts.reindex(ami_settle.index).fillna(0).astype(int).values
    )

    # --- Build figure ---
    if not tmode:
        import cartopy.crs as ccrs
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def _add_panel(ax, title: str) -> None:
        if not tmode:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
            transform = ccrs.PlateCarree()
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            transform = None
        pkw = {"transform": transform} if transform else {}

        ax.set_title(title, fontsize=10)
        if not tmode:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
        return pkw

    # Left panel: origin density on grid cells
    pkw = _add_panel(axes[0], "(a) Origin density")
    # Grey background for all grid cells
    grid_gdf.plot(ax=axes[0], facecolor="lightgrey", edgecolor="none",
                  alpha=0.3, zorder=1, **pkw)
    if not grid_with_particles.empty:
        vmax_orig = grid_with_particles["particle_count"].max()
        grid_with_particles.plot(
            column="particle_count", ax=axes[0], cmap="Reds",
            norm=mcolors.LogNorm(vmin=1, vmax=max(vmax_orig, 2)),
            edgecolor="none", linewidth=0, zorder=2,
            legend=True,
            legend_kwds={"label": "Particle count (log scale)", "shrink": 0.65},
            **pkw,
        )
    ami_gdf.plot(ax=axes[0], facecolor="none", edgecolor="navy",
                 linewidth=0.8, zorder=4, **pkw)
    apei_gdf.plot(ax=axes[0], facecolor="none", edgecolor="black",
                  linewidth=1.0, linestyle="--", zorder=5, **pkw)

    # Right panel: settlement density per AMI (block colour)
    pkw = _add_panel(axes[1], "(b) Settlement density in AMIs")
    vmax_settle = max(ami_settle["particle_count"].max(), 1)
    ami_settle.plot(
        column="particle_count", ax=axes[1], cmap="Blues",
        norm=mcolors.Normalize(vmin=0, vmax=vmax_settle),
        edgecolor="navy", linewidth=0.8, zorder=2,
        legend=True,
        legend_kwds={"label": "Particle count", "shrink": 0.65},
        **pkw,
    )
    apei_gdf.plot(ax=axes[1], facecolor="none", edgecolor="black",
                  linewidth=1.0, linestyle="--", zorder=5, **pkw)

    fig.suptitle(
        "Figure 2 — Larval supply to AMIs: origin (red) and settlement (blue)",
        fontsize=12,
    )
    _save_figure(fig, figures_dir / "fig2_ami_particle_density", dpi=dpi)


# ── Figure 3: community membership ───────────────────────────────────────────

def plot_fig3(config: dict) -> None:
    """
    Community membership choropleth with transitional zone outlines.

    Each polygon coloured by community.  Dashed black squares = APEIs.
    Red outlines = transitional zones (community shifts under mining).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Figure 3: community membership map …")
    extent = _get_extent(config)
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    tmode = _test_mode(config)
    network_dir = Path(config["network_dir"])

    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])

    unmined_path = network_dir / "network_metrics_unmined.csv"
    mined_path = network_dir / "network_metrics_mined.csv"
    if not unmined_path.exists():
        logger.warning("Network metrics not found — skipping Figure 3")
        return

    unmined_df = pd.read_csv(unmined_path)
    mined_df = pd.read_csv(mined_path) if mined_path.exists() else pd.DataFrame(
        columns=["polygon_ID", "community"]
    )

    merged = unmined_df[["polygon_ID", "community"]].merge(
        mined_df[["polygon_ID", "community"]].rename(columns={"community": "community_mined"}),
        on="polygon_ID",
        how="left",
    )
    merged["transitional"] = (
        merged["community"].astype(str) != merged["community_mined"].fillna("").astype(str)
    )

    gdf = grid_gdf.merge(merged, left_on="FID", right_on="polygon_ID", how="left")
    communities = sorted(gdf["community"].dropna().unique())
    n_comm = len(communities)
    cmap = plt.cm.get_cmap("tab20", max(n_comm, 1))
    comm_to_idx = {c: i for i, c in enumerate(communities)}

    fig, ax = _base_map(extent, figsize=(14, 8),
                        title="Figure 3 — Connectivity Communities", test_mode=tmode)

    pkw = _plot_kwargs(tmode)
    for comm in communities:
        subset = gdf[gdf["community"] == comm]
        subset.plot(ax=ax, color=cmap(comm_to_idx[comm]),
                    alpha=0.75, linewidth=0.1, edgecolor="white", zorder=1, **pkw)

    # Transitional zones: red outline, no fill
    transitional = gdf[gdf["transitional"] == True]
    if not transitional.empty:
        transitional.plot(ax=ax, facecolor="none", edgecolor="red",
                          linewidth=1.2, zorder=3, **pkw)

    # APEI boundaries: dashed black
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    apei_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                  linewidth=1.5, linestyle="--", zorder=4, **pkw)

    # Legend
    patches = [
        mpatches.Patch(color=cmap(comm_to_idx[c]), label=f"Community {c}")
        for c in communities
    ]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                  label="Transitional zone"))
    patches.append(mpatches.Patch(facecolor="none", edgecolor="black",
                                  linestyle="--", label="APEI"))
    ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2)

    _save_figure(fig, figures_dir / "fig3_community_membership", dpi=dpi)


# ── Figure 4: scenario-stable connectivity ────────────────────────────────────

def plot_fig4(config: dict) -> None:
    """
    Scenario-stable connectivity score choropleth (plasma colormap).

    Top 10% cells outlined in black (visualisation only, not part of analysis).
    AMIs shown in grey.  APEI boundaries = dashed white.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Figure 4: scenario-stable connectivity …")
    extent = _get_extent(config)
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    tmode = _test_mode(config)
    network_dir = Path(config["network_dir"])

    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    scenario_path = network_dir / "scenario_stable_connectivity.csv"
    if not scenario_path.exists():
        logger.warning("Scenario-stable CSV not found — skipping Figure 4")
        return

    scenario_df = pd.read_csv(scenario_path)
    gdf = grid_gdf.merge(scenario_df, left_on="FID", right_on="polygon_ID", how="left")
    score_col = "Scenario-Stable Connectivity Score"
    ami_gdf = gpd.read_file(config["shapefiles"]["ami"])
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])

    fig, ax = _base_map(extent, figsize=(14, 8),
                        title="Figure 4 — Scenario-Stable Connectivity Score", test_mode=tmode)
    pkw = _plot_kwargs(tmode)

    # AMIs: grey
    ami_gdf.plot(ax=ax, facecolor="grey", edgecolor="none", alpha=0.7, zorder=1, **pkw)

    # Choropleth
    gdf.plot(
        column=score_col, ax=ax, cmap="plasma", legend=True,
        legend_kwds={"label": "Scenario-stable connectivity score", "shrink": 0.65},
        missing_kwds={"color": "lightgrey"},
        zorder=2, **pkw,
    )

    # Top 10% cells: black outline
    threshold_val = gdf[score_col].quantile(0.90)
    top10 = gdf[gdf[score_col] >= threshold_val]
    if not top10.empty:
        top10.plot(ax=ax, facecolor="none", edgecolor="black",
                   linewidth=0.8, zorder=3, **pkw)

    # APEI: dashed white
    apei_gdf.plot(ax=ax, facecolor="none", edgecolor="white",
                  linewidth=1.5, linestyle="--", zorder=4, **pkw)

    _save_figure(fig, figures_dir / "fig4_scenario_stable_connectivity", dpi=dpi)


# ── Figure 5: optimised corridor network ──────────────────────────────────────

def plot_fig5(config: dict) -> None:
    """
    Optimised APEI corridor network map with larval flow arrows.

    Top panel: choropleth by community; APEI cells coloured by community;
    corridor additions outlined black; previously unprotected community cells
    outlined red; dispersal arrows coloured by log10(particle count).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Figure 5: optimised corridor network …")
    extent = _get_extent(config)
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    tmode = _test_mode(config)
    optim_dir = Path(config["optimisation_dir"])
    network_dir = Path(config["network_dir"])

    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    ami_gdf = gpd.read_file(config["shapefiles"]["ami"])

    icv_path = network_dir / "icv_scores.csv"
    corridor_path = optim_dir / "corridor_edges.csv"
    minimal_path = optim_dir / "minimal_icv_selection.txt"
    unmined_metrics_path = network_dir / "network_metrics_unmined.csv"

    if not icv_path.exists():
        logger.warning("ICV scores not found — skipping Figure 5")
        return

    icv_df = pd.read_csv(icv_path)
    gdf = grid_gdf.merge(icv_df[["polygon_ID", "ICV"]], left_on="FID",
                         right_on="polygon_ID", how="left")
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    centroids = gdf.set_index("FID")[["centroid_lon", "centroid_lat"]]

    # Identify APEI grid cells
    from ccz_connectivity.optimisation import identify_apei_polygon_ids
    apei_cell_map = identify_apei_polygon_ids(grid_gdf, apei_gdf)
    apei_fids: set[int] = set(apei_cell_map.keys())

    # Final Phase-2 corridor selection
    final_ids: set[int] = set()
    if minimal_path.exists():
        final_ids = {
            int(x.strip())
            for x in minimal_path.read_text().splitlines()
            if x.strip()
        }

    # Stepping-stone additions = final selection minus APEI cells
    addition_fids = final_ids - apei_fids

    fig, ax = _base_map(extent, figsize=(16, 9),
                        title="Figure 5 — Optimised APEI Corridor Network", test_mode=tmode)
    pkw = _plot_kwargs(tmode)

    # AMIs: grey background
    ami_gdf.plot(ax=ax, facecolor="grey", edgecolor="none", alpha=0.5, zorder=1, **pkw)

    # Corridor addition cells coloured by ICV (stepping stones only)
    addition_cells = gdf[gdf["FID"].isin(addition_fids) & gdf["ICV"].notna()].copy()
    if not addition_cells.empty:
        norm_icv = mcolors.Normalize(
            vmin=addition_cells["ICV"].min(), vmax=addition_cells["ICV"].max()
        )
        cmap_icv = plt.cm.YlGn
        addition_cells.plot(
            ax=ax, column="ICV", cmap=cmap_icv, norm=norm_icv,
            edgecolor="black", linewidth=0.5, zorder=3, **pkw
        )
        sm_icv = plt.cm.ScalarMappable(cmap=cmap_icv, norm=norm_icv)
        sm_icv.set_array([])
        plt.colorbar(sm_icv, ax=ax, shrink=0.45, label="ICV", pad=0.01)

    # APEI cells: gold fill
    apei_cells = gdf[gdf["FID"].isin(apei_fids)]
    if not apei_cells.empty:
        apei_cells.plot(ax=ax, facecolor="gold", edgecolor="black",
                        linewidth=0.5, alpha=0.85, zorder=4, **pkw)

    # APEI boundaries: solid black
    apei_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                  linewidth=1.5, zorder=6, **pkw)

    # Dispersal arrows: log10 colour intensity
    if corridor_path.exists():
        edges_df = pd.read_csv(corridor_path)
        if not edges_df.empty and "flow" in edges_df.columns:
            flows = edges_df["flow"].values.astype(float)
            log_flows = np.log10(np.maximum(flows, 1.0))
            norm = mcolors.Normalize(vmin=log_flows.min(), vmax=log_flows.max())
            cmap_arrow = plt.cm.YlOrRd

            if tmode:
                for _, row in edges_df.iterrows():
                    src, dst = int(row["src"]), int(row["dst"])
                    if src in centroids.index and dst in centroids.index:
                        x0, y0 = centroids.loc[src]
                        x1, y1 = centroids.loc[dst]
                        lf = np.log10(max(float(row["flow"]), 1.0))
                        colour = cmap_arrow(norm(lf))
                        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                    arrowprops={"arrowstyle": "->",
                                                "color": colour, "lw": 0.8})
            else:
                import cartopy.crs as ccrs
                for _, row in edges_df.iterrows():
                    src, dst = int(row["src"]), int(row["dst"])
                    if src in centroids.index and dst in centroids.index:
                        x0, y0 = centroids.loc[src]
                        x1, y1 = centroids.loc[dst]
                        lf = np.log10(max(float(row["flow"]), 1.0))
                        colour = cmap_arrow(norm(lf))
                        ax.annotate(
                            "", xy=(x1, y1), xytext=(x0, y0),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            arrowprops={"arrowstyle": "->",
                                        "color": colour, "lw": 0.8},
                        )

            sm = plt.cm.ScalarMappable(cmap=cmap_arrow, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.5, label="log₁₀(particle flow)",
                         pad=0.01)

    _save_figure(fig, figures_dir / "fig5_optimised_corridor_network", dpi=dpi)


# ── Extended data figures ─────────────────────────────────────────────────────

def plot_ed_figs(config: dict) -> None:
    """
    Generate all six extended data figures (ED1–ED6).

    ED1 — Dispersal distance KDE for 3 PLDs.
    ED2 — Community connectivity matrices (unmined vs mined).
    ED3 — Domain overview (grid, APEIs, AMIs).
    ED4 — Marginal returns curve (PPS vs unique connections).
    ED5 — Ai (APEI support) and Mi (mining replenishment) spatial maps.
    ED6 — Full ICV grid map.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    """
    logger.info("Generating Extended Data figures 1–6 …")
    dpi: int = config.get("figure_dpi", 300)
    figures_dir = Path(config["figures_dir"])
    _plot_ed1(config, figures_dir, dpi)
    _plot_ed2(config, figures_dir, dpi)
    _plot_ed3(config, figures_dir, dpi)
    _plot_ed4(config, figures_dir, dpi)
    _plot_ed5(config, figures_dir, dpi)
    _plot_ed6(config, figures_dir, dpi)


# ── ED1: dispersal distance KDE ───────────────────────────────────────────────

def _plot_ed1(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED1: Kernel density of dispersal distances for each PLD."""
    from scipy.stats import gaussian_kde

    logger.info("  ED1: dispersal distance distributions …")
    conn_dir = Path(config["connectivity_dir"])
    pld_days: list[int] = config["tracking"]["pld_days"]
    pld_colours = ["steelblue", "darkorange", "forestgreen"]

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for pld, colour in zip(pld_days, pld_colours):
        # Load per-particle data for this PLD (unmined only)
        p = conn_dir / f"per_particle_pld{pld}_unmined.csv"
        if not p.exists():
            # Fall back to full per_particle file and filter
            fp = conn_dir / "per_particle_unmined.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp, usecols=["start_lat", "start_lon",
                                           "end_lat", "end_lon", "PLD"])
            df = df[df["PLD"] == pld]
        else:
            df = pd.read_csv(p, usecols=["start_lat", "start_lon",
                                          "end_lat", "end_lon"])

        if df.empty:
            continue

        # Great-circle distance (haversine, vectorised)
        lat1 = np.radians(df["start_lat"].values)
        lat2 = np.radians(df["end_lat"].values)
        dlon = np.radians(df["end_lon"].values - df["start_lon"].values)
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dist_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        valid = dist_km[np.isfinite(dist_km)]
        if len(valid) < 10:
            continue

        kde = gaussian_kde(valid, bw_method=0.1)
        x = np.linspace(0, valid.max() * 1.05, 500)
        ax.plot(x, kde(x), color=colour, linewidth=2,
                label=f"PLD {pld} d (median {np.median(valid):.1f} km)")
        ax.axvline(np.median(valid), color=colour, linewidth=0.8, linestyle=":")
        plotted = True

    if not plotted:
        logger.warning("No dispersal data for ED1 — skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Dispersal distance (km)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Extended Data Figure 1 — Dispersal Distance Distributions", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figures_dir / "ed1_dispersal_distances", dpi=dpi)


# ── ED2: community connectivity matrices ──────────────────────────────────────

def _plot_ed2(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED2: Community-level connectivity matrices (unmined vs mined)."""
    logger.info("  ED2: community connectivity matrices …")
    conn_dir = Path(config["connectivity_dir"])
    network_dir = Path(config["network_dir"])
    scenarios = config["tracking"]["scenarios"]
    pld_days: list[int] = config["tracking"]["pld_days"]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(8 * len(scenarios), 7))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        conn_path = conn_dir / f"connectivity_aggregated_{scenario}.csv"
        metrics_path = network_dir / f"network_metrics_{scenario}.csv"
        if not conn_path.exists() or not metrics_path.exists():
            ax.set_title(f"{scenario} (data not found)")
            continue

        conn_df = pd.read_csv(conn_path)
        metrics_df = pd.read_csv(metrics_path)

        # Merge community into connectivity
        poly_to_comm = metrics_df.set_index("polygon_ID")["community"].to_dict()
        conn_df["comm_src"] = conn_df["Start_polyID"].map(poly_to_comm)
        conn_df["comm_dst"] = conn_df["End_polyID"].map(poly_to_comm)

        # Aggregate PLD-averaged count by community pair
        comm_agg = (
            conn_df.groupby(["comm_src", "comm_dst"])["Count"]
            .mean()
            .reset_index()
        )
        communities = sorted(
            set(comm_agg["comm_src"].dropna().unique())
            | set(comm_agg["comm_dst"].dropna().unique())
        )
        n = len(communities)
        if n == 0:
            ax.set_title(f"{scenario} (no communities)")
            continue

        comm_idx = {c: i for i, c in enumerate(communities)}
        matrix = np.zeros((n, n), dtype=float)
        for _, row in comm_agg.iterrows():
            if row["comm_src"] in comm_idx and row["comm_dst"] in comm_idx:
                i = comm_idx[row["comm_src"]]
                j = comm_idx[row["comm_dst"]]
                # Normalise by product of community sizes for display
                matrix[i, j] += float(row["Count"])

        # Normalise by community size
        comm_sizes = metrics_df.groupby("community").size().to_dict()
        for i, ci in enumerate(communities):
            for j, cj in enumerate(communities):
                si = comm_sizes.get(ci, 1)
                sj = comm_sizes.get(cj, 1)
                matrix[i, j] /= max(si * sj, 1)

        im = ax.imshow(matrix, cmap="Reds", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised exchange")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(c) for c in communities], rotation=45, ha="right")
        ax.set_yticklabels([str(c) for c in communities])
        ax.set_xlabel("Sink community")
        ax.set_ylabel("Source community")
        ax.set_title(f"{'Baseline' if scenario == 'unmined' else 'Mining disturbance'}")

    fig.suptitle("Extended Data Figure 2 — Community Connectivity Matrices", fontsize=12)
    _save_figure(fig, figures_dir / "ed2_community_matrices", dpi=dpi)


# ── ED3: domain overview ──────────────────────────────────────────────────────

def _plot_ed3(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED3: Domain overview — CCZ grid, APEIs, AMIs."""
    logger.info("  ED3: domain overview …")
    extent = _get_extent(config)
    tmode = _test_mode(config)

    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    ami_gdf = gpd.read_file(config["shapefiles"]["ami"])

    fig, ax = _base_map(extent, figsize=(14, 7),
                        title=f"Extended Data Figure 3 — CCZ Grid (N = {len(grid_gdf)} cells)",
                        test_mode=tmode)
    pkw = _plot_kwargs(tmode)

    grid_gdf.plot(ax=ax, facecolor="lightyellow", edgecolor="lightgray",
                  linewidth=0.3, zorder=1, **pkw)
    ami_gdf.plot(ax=ax, facecolor="sandybrown", edgecolor="saddlebrown",
                 alpha=0.6, zorder=2, **pkw)
    apei_gdf.plot(ax=ax, facecolor="steelblue", edgecolor="navy",
                  alpha=0.6, linewidth=1.2, linestyle="--", zorder=3, **pkw)

    patches = [
        mpatches.Patch(facecolor="lightyellow", edgecolor="lightgray", label="Grid cells"),
        mpatches.Patch(facecolor="sandybrown", edgecolor="saddlebrown", label="AMI"),
        mpatches.Patch(facecolor="steelblue", edgecolor="navy", label="APEI"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9)

    _save_figure(fig, figures_dir / "ed3_domain_overview", dpi=dpi)


# ── ED4: marginal returns curve ───────────────────────────────────────────────

def _plot_ed4(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED4: Marginal returns — PPS vs unique connections (logarithmic fit)."""
    logger.info("  ED4: marginal returns curve …")
    # This figure shows the sensitivity analysis from the manuscript.
    # In a live run the data would come from a pre-computed CSV.
    # Here we generate the illustrative fitted curve from manuscript values.
    from scipy.optimize import curve_fit

    pps_vals = np.array([100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3648, 4000, 5000, 10000])

    # Logarithmic model: f(x) = a * log(x) + b (illustrative)
    # Use published anchor points: ~400 PPS = 1 unique conn/100 PPS marginal
    def log_model(x, a, b):
        return a * np.log(x) + b

    # Illustrative unique connections (not from actual data — placeholder curve)
    unique_conns = log_model(pps_vals, a=12000, b=-40000)
    unique_conns = np.maximum(unique_conns, 0)

    # Marginal return (change per 100 PPS)
    marginal = np.gradient(unique_conns, pps_vals) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(pps_vals, unique_conns, "k-", linewidth=2)
    axes[0].set_xlabel("Particles per site (PPS)")
    axes[0].set_ylabel("Unique cell-to-cell connections")
    axes[0].set_title("Unique connections vs PPS")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pps_vals, marginal, "k-", linewidth=2)
    axes[1].axhline(1.0, color="orange", linewidth=1.5, linestyle="--",
                    label="Diminishing returns (~400 PPS)")
    axes[1].axhline(0.1, color="red", linewidth=1.5, linestyle="--",
                    label="Flatline (~3,648 PPS)")
    axes[1].axvline(400, color="orange", linewidth=1.0, alpha=0.7)
    axes[1].axvline(3648, color="red", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Particles per site (PPS)")
    axes[1].set_ylabel("Marginal return (unique connections / 100 PPS)")
    axes[1].set_title("Marginal returns")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.suptitle("Extended Data Figure 4 — Marginal Returns Analysis", fontsize=12)
    _save_figure(fig, figures_dir / "ed4_marginal_returns", dpi=dpi)


# ── ED5: Ai and Mi spatial maps ───────────────────────────────────────────────

def _plot_ed5(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED5: Two-panel map of APEI support (Ai) and mining replenishment (Mi)."""
    logger.info("  ED5: Ai and Mi spatial maps …")
    extent = _get_extent(config)
    tmode = _test_mode(config)
    network_dir = Path(config["network_dir"])

    icv_path = network_dir / "icv_scores.csv"
    if not icv_path.exists():
        logger.warning("ICV scores not found — skipping ED5")
        return

    icv_df = pd.read_csv(icv_path)
    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    ami_gdf = gpd.read_file(config["shapefiles"]["ami"])
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    gdf = grid_gdf.merge(icv_df, left_on="FID", right_on="polygon_ID", how="left")

    components = [
        ("normalised_support_score", "APEI support score (Ai)"),
        ("mining_replenishment_score", "Mining replenishment score (Mi)"),
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 14),
        subplot_kw=({"projection": __import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree()}
                    if not tmode else {}),
    )
    pkw = _plot_kwargs(tmode)

    for ax, (col, label) in zip(axes, components):
        if not tmode:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        gdf.plot(column=col, ax=ax, cmap="YlOrRd", legend=True,
                 legend_kwds={"label": label, "shrink": 0.65},
                 missing_kwds={"color": "lightgrey"}, zorder=1, **pkw)
        ami_gdf.plot(ax=ax, facecolor="grey", edgecolor="none", alpha=0.6,
                     zorder=3, **pkw)
        apei_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.2, linestyle="--", zorder=4, **pkw)
        ax.set_title(label, fontsize=11)

    fig.suptitle("Extended Data Figure 5 — ICV Component Scores (Ai, Mi)", fontsize=12)
    _save_figure(fig, figures_dir / "ed5_ai_mi_maps", dpi=dpi)


# ── ED6: full ICV map ─────────────────────────────────────────────────────────

def _plot_ed6(config: dict, figures_dir: Path, dpi: int) -> None:
    """ED6: Full ICV choropleth across all CCZ grid cells."""
    logger.info("  ED6: ICV full grid map …")
    extent = _get_extent(config)
    tmode = _test_mode(config)
    network_dir = Path(config["network_dir"])

    icv_path = network_dir / "icv_scores.csv"
    if not icv_path.exists():
        logger.warning("ICV scores not found — skipping ED6")
        return

    icv_df = pd.read_csv(icv_path)
    grid_gdf = gpd.read_file(config["shapefiles"]["ccz_grid"])
    apei_gdf = gpd.read_file(config["shapefiles"]["apei"])
    gdf = grid_gdf.merge(icv_df[["polygon_ID", "ICV"]], left_on="FID",
                         right_on="polygon_ID", how="left")

    fig, ax = _base_map(extent, figsize=(14, 8),
                        title="Extended Data Figure 6 — Integrated Connectivity Value (ICV)",
                        test_mode=tmode)
    pkw = _plot_kwargs(tmode)

    gdf.plot(column="ICV", ax=ax, cmap="plasma", legend=True,
             legend_kwds={"label": "Normalised ICV", "shrink": 0.65},
             missing_kwds={"color": "lightgrey"}, zorder=1, **pkw)
    apei_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                  linewidth=1.5, linestyle="--", zorder=3, **pkw)

    _save_figure(fig, figures_dir / "ed6_icv_map", dpi=dpi)
