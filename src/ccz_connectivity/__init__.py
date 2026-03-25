"""
ccz_connectivity
================
Python library for CCZ larval dispersal connectivity analysis.

Modules
-------
grid          : Build and load the equal-area CCZ grid.
tracking      : Configure and run OceanParcels particle tracking.
connectivity  : Build connectivity matrices from particle trajectories.
network       : Graph construction, community detection, and network metrics.
icv           : Integrated Connectivity Value (ICV) computation.
optimisation  : Two-phase corridor optimisation.
plotting      : Publication-standard figure generation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("ccz_connectivity")
except PackageNotFoundError:
    __version__ = "unknown"

# Public API re-exports
from ccz_connectivity.grid import make_ccz_grid, load_centroids, save_grid
from ccz_connectivity.connectivity import (
    build_connectivity,
    compute_apei_endpoints,
    jaccard_dissimilarity,
    cohen_kappa,
    permutation_test,
)
from ccz_connectivity.network import (
    build_graph,
    detect_communities,
    compute_network_metrics,
    compute_apei_support,
)
from ccz_connectivity.icv import compute_icv
from ccz_connectivity.optimisation import run_optimisation

__all__ = [
    "__version__",
    # grid
    "make_ccz_grid",
    "load_centroids",
    "save_grid",
    # connectivity
    "build_connectivity",
    "compute_apei_endpoints",
    "jaccard_dissimilarity",
    "cohen_kappa",
    "permutation_test",
    # network
    "build_graph",
    "detect_communities",
    "compute_network_metrics",
    "compute_apei_support",
    # icv
    "compute_icv",
    # optimisation
    "run_optimisation",
]
