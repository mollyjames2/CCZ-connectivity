"""
tracking.py — OceanParcels particle-tracking setup and execution.

Builds a FieldSet from HYCOM NetCDF files, defines custom kernels for
Smagorinsky diffusion and distance accumulation, and runs the simulation
using AdvectionRK4 + custom kernels.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# OceanParcels imports deferred so the module can be imported without parcels
# installed (e.g., for documentation builds).
try:
    import parcels
    from parcels import (
        FieldSet,
        ParticleSet,
        JITParticle,
        Variable,
        AdvectionRK4,
        ParcelsRandom,
    )
    from parcels import StatusCode
    _PARCELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PARCELS_AVAILABLE = False
    parcels = None  # type: ignore[assignment]
    FieldSet = None  # type: ignore[assignment]


# ── Kernels ───────────────────────────────────────────────────────────────────

def smagorinsky_kernel(particle, fieldset, time):  # type: ignore[no-untyped-def]
    """
    Smagorinsky sub-grid-scale horizontal diffusion kernel.

    Implements isotropic random-walk diffusion with a diffusivity calculated
    from the local velocity gradient following Smagorinsky (1963).

    Parameters are embedded via fieldset attributes set in
    :func:`build_fieldset`:
    - ``fieldset.Cs``  : Smagorinsky coefficient (default 0.1)
    - ``fieldset.dx``  : grid spacing in metres

    Parameters
    ----------
    particle : parcels.JITParticle
        The particle being advected.
    fieldset : parcels.FieldSet
        The ocean FieldSet containing velocity and diffusion parameters.
    time : float
        Current simulation time (seconds since epoch).
    """
    # Local velocity gradients (finite-difference approximation)
    dx = fieldset.dx
    u0 = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    u1 = fieldset.U[time, particle.depth, particle.lat, particle.lon + dx]
    v0 = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    v1 = fieldset.V[time, particle.depth, particle.lat + dx, particle.lon]

    dudx = (u1 - u0) / dx
    dvdy = (v1 - v0) / dx
    # Shear term (simplified: use off-diagonal gradient estimate)
    dudy = (fieldset.U[time, particle.depth, particle.lat + dx, particle.lon] - u0) / dx
    dvdx = (fieldset.V[time, particle.depth, particle.lat, particle.lon + dx] - v0) / dx

    S = math.sqrt(2.0 * dudx**2 + 2.0 * dvdy**2 + (dudy + dvdx)**2)
    Kh = (fieldset.Cs * fieldset.dx_m) ** 2 * S

    # Random-walk displacement using OceanParcels RNG (required in JIT kernels)
    r = math.sqrt(2.0 * Kh * math.fabs(particle.dt))
    particle.lon += r * ParcelsRandom.uniform(-1.0, 1.0)  # type: ignore[attr-defined]
    particle.lat += r * ParcelsRandom.uniform(-1.0, 1.0)  # type: ignore[attr-defined]


def distance_kernel(particle, fieldset, time):  # type: ignore[no-untyped-def]
    """
    Accumulate great-circle distance travelled by each particle.

    Updates ``particle.distance`` at every timestep using the haversine
    approximation.  The running total is stored in degrees of arc and
    converted to kilometres by downstream analysis.

    Parameters
    ----------
    particle : parcels.JITParticle
        The particle being tracked; must have a ``distance`` Variable.
    fieldset : parcels.FieldSet
        Unused directly, but required by the OceanParcels kernel signature.
    time : float
        Current simulation time (seconds).
    """
    lat_rad = math.pi / 180.0
    dlat = (particle.lat - particle.prev_lat) * lat_rad
    dlon = (particle.lon - particle.prev_lon) * lat_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(particle.prev_lat * lat_rad)
        * math.cos(particle.lat * lat_rad)
        * math.sin(dlon / 2.0) ** 2
    )
    dist_km = 6371.0 * 2.0 * math.asin(math.sqrt(a))
    particle.distance += dist_km
    particle.prev_lat = particle.lat
    particle.prev_lon = particle.lon


def build_fieldset(config: dict, time_period: int) -> "FieldSet":
    """
    Build an OceanParcels FieldSet from HYCOM NetCDF data for one time period.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    time_period : int
        Zero-based index into ``config["time_periods"]["periods"]``.

    Returns
    -------
    parcels.FieldSet
        FieldSet with U and V from HYCOM bottom-layer currents, plus
        ``Cs``, ``dx``, and ``dx_m`` constants for the Smagorinsky kernel.
    """
    if not _PARCELS_AVAILABLE:
        raise ImportError("OceanParcels is required for build_fieldset")

    period = config["time_periods"]["periods"][time_period]
    label = period["label"]
    hycom_dir = Path(config["hycom_data_dir"])

    u_file = hycom_dir / f"water_u_bottom_{label}.nc"
    v_file = hycom_dir / f"water_v_bottom_{label}.nc"

    logger.info(f"Building FieldSet for period {label}")
    logger.info(f"  U: {u_file}")
    logger.info(f"  V: {v_file}")

    filenames = {
        "U": {"lon": str(u_file), "lat": str(u_file), "time": str(u_file), "data": str(u_file)},
        "V": {"lon": str(v_file), "lat": str(v_file), "time": str(v_file), "data": str(v_file)},
    }
    variables = {"U": "water_u_bottom", "V": "water_v_bottom"}
    dimensions = {"U": {"lon": "lon", "lat": "lat", "time": "time"},
                  "V": {"lon": "lon", "lat": "lat", "time": "time"}}

    fieldset = FieldSet.from_netcdf(
        filenames, variables, dimensions, allow_time_extrapolation=True
    )

    # Add Smagorinsky constants as FieldSet attributes
    cs: float = config["tracking"]["smagorinsky"]["Cs"]
    dx_deg: float = config["tracking"]["smagorinsky"]["dx_deg"]
    # Convert dx to approximate metres at domain centre
    lat_centre = (config["domain"]["lat_min"] + config["domain"]["lat_max"]) / 2.0
    dx_m = dx_deg * 111_320.0 * math.cos(math.radians(lat_centre))

    fieldset.add_constant("Cs", cs)
    fieldset.add_constant("dx", dx_deg)
    fieldset.add_constant("dx_m", dx_m)

    logger.info(f"FieldSet built: Cs={cs}, dx_deg={dx_deg}, dx_m={dx_m:.1f} m")
    return fieldset


def run_tracking(config: dict, time_period: int, scenario: str) -> Path:
    """
    Run OceanParcels particle tracking for one time period and scenario.

    Particles are seeded from the CCZ grid centroids at
    ``config["tracking"]["pps"]`` per site.  The simulation runs for
    ``max(pld_days)`` days.  Output is written to Zarr format compatible
    with the OceanParcels 3.0+ schema.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    time_period : int
        Zero-based index into ``config["time_periods"]["periods"]``.
    scenario : str
        One of ``"unmined"`` or ``"mined"``.  Controls which grid shapefile
        is used (mining areas removed for ``"mined"``).

    Returns
    -------
    Path
        Path to the output Zarr store.
    """
    if not _PARCELS_AVAILABLE:
        raise ImportError("OceanParcels is required for run_tracking")

    from ccz_connectivity.grid import load_centroids

    period = config["time_periods"]["periods"][time_period]
    label = period["label"]
    start_date = period["start"]
    pps: int = config["tracking"]["pps"]
    pld_days: list[int] = config["tracking"]["pld_days"]
    dt_min: int = config["tracking"]["dt_minutes"]
    random_seed: int = config["tracking"]["random_seed"]

    particles_dir = Path(config["particles_dir"])
    particles_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = particles_dir / f"particles_{label}_{scenario}.zarr"

    logger.info(f"Running particle tracking: period={label}, scenario={scenario}")

    centroids_path = Path(config["shapefiles"]["ccz_centroids"])
    centroids = load_centroids(centroids_path)

    if scenario == "mined":
        # For mined scenario, exclude sites within AMI polygons
        import geopandas as gpd
        from shapely.geometry import Point
        ami_gdf = gpd.read_file(config["shapefiles"]["ami"])
        ami_union = ami_gdf.unary_union
        mask = centroids.apply(
            lambda row: not ami_union.contains(Point(row["lon"], row["lat"])), axis=1
        )
        centroids = centroids[mask].reset_index(drop=True)
        logger.info(f"Mined scenario: {len(centroids)} sites after excluding AMIs")

    np.random.seed(random_seed)
    lons_seed = np.repeat(centroids["lon"].values, pps)
    lats_seed = np.repeat(centroids["lat"].values, pps)
    # Add small random jitter within ±0.01° to avoid co-located particles
    lons_seed += np.random.uniform(-0.01, 0.01, size=len(lons_seed))
    lats_seed += np.random.uniform(-0.01, 0.01, size=len(lats_seed))

    logger.info(f"Seeding {len(lons_seed)} particles from {len(centroids)} sites")

    fieldset = build_fieldset(config, time_period)

    # Define custom particle class with distance tracking
    class CCZParticle(JITParticle):
        distance = Variable("distance", initial=0.0)
        prev_lat = Variable("prev_lat", to_write=False,
                            initial=parcels.attrgetter("lat"))
        prev_lon = Variable("prev_lon", to_write=False,
                            initial=parcels.attrgetter("lon"))

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=CCZParticle,
        lon=lons_seed,
        lat=lats_seed,
    )

    max_pld = max(pld_days)
    runtime_seconds = max_pld * 86_400
    dt_seconds = dt_min * 60

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(smagorinsky_kernel) + pset.Kernel(distance_kernel)

    output_file = pset.ParticleFile(
        name=str(zarr_path),
        outputdt=config["tracking"]["output_dt_hours"] * 3600,
    )

    logger.info(f"Executing {max_pld}-day simulation (dt={dt_min} min) …")
    pset.execute(
        kernels,
        runtime=runtime_seconds,
        dt=dt_seconds,
        output_file=output_file,
        recovery={StatusCode.ErrorOutOfBounds: parcels.DeletionParticle},
    )
    output_file.close()

    logger.info(f"Particle tracking complete → {zarr_path}")
    return zarr_path
