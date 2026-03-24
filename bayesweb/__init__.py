"""
bayesweb — Bayesian belief web with tipping-point detection.

Quick start
-----------
>>> from bayesweb import BeliefWeb, Zone, apply_shock, propagate, fragility, propagation_gain
>>> web = BeliefWeb("demo")
>>> web.add_node("A", "Land use intensifies", alpha=3, beta=1, zone=Zone.NUCLEUS)
>>> web.add_node("B", "Biodiversity declines", alpha=2, beta=2, zone=Zone.PERIPHERY)
>>> web.add_undirected_edge("A", "B", weight=0.7)
>>> import copy
>>> web_s, dl = apply_shock(copy.deepcopy(web), "A", kappa=5, sigma=0.9, inplace=True)
>>> result = propagate(web_s, "A", dl)
>>> print(f"F={fragility(web):.3f}  G={propagation_gain(result, web):.3f}")
"""

from .graph import BeliefWeb, BeliefNode, Zone
from .shock import apply_shock, shock_summary
from .propagation import propagate, calibrate_eta, PropagationResult
from .indicators import (
    fragility,
    fragility_bootstrap,
    propagation_gain,
    scan_tipping_point,
    indicators_summary,
)

__all__ = [
    "BeliefWeb", "BeliefNode", "Zone",
    "apply_shock", "shock_summary",
    "propagate", "calibrate_eta", "PropagationResult",
    "fragility", "fragility_bootstrap", "propagation_gain",
    "scan_tipping_point", "indicators_summary",
]

__version__ = "0.1.0"
