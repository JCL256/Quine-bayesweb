from __future__ import annotations
import copy
import numpy as np
from .graph import BeliefWeb

def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1-1e-12))
    return float(np.log(p/(1-p)))

def apply_shock(web, target, kappa, sigma, inplace=False):
    """
    Apply scenaristic shock (kappa, sigma) to target node.  Eq. (2.4)-(2.6).

    Returns (updated_web, delta_logit).
    inplace=False (default) works on a deep copy.
    """
    if kappa <= 0:         raise ValueError(f"kappa must be > 0 (got {kappa}).")
    if not 0<=sigma<=1:    raise ValueError(f"sigma must be in [0,1] (got {sigma}).")
    result = web if inplace else copy.deepcopy(web)
    node = result.get_node(target)
    alpha_k, beta_k = node.alpha, node.beta
    s_k, p_k = node.solidity, node.belief
    alpha_new = alpha_k + kappa * sigma           # Eq. (2.4)
    beta_new  = beta_k  + kappa * (1.0 - sigma)
    p_new     = (s_k * p_k + kappa * sigma) / (s_k + kappa)  # Eq. (2.5)
    delta_logit = _logit(p_new) - _logit(p_k)    # Eq. (2.6)
    result.update_node_belief(target, alpha_new, beta_new)
    return result, float(delta_logit)

def shock_summary(web, target, kappa, sigma) -> dict:
    """Return shock info dict without modifying the web."""
    node = web.get_node(target)
    s_k, p_k = node.solidity, node.belief
    alpha_new = node.alpha + kappa * sigma
    beta_new  = node.beta  + kappa * (1.0 - sigma)
    p_new     = (s_k * p_k + kappa * sigma) / (s_k + kappa)
    delta_logit = _logit(p_new) - _logit(p_k)
    return {"target": target, "kappa": kappa, "sigma": sigma,
            "p_before": round(p_k,6), "p_after": round(p_new,6),
            "delta_p": round(p_new-p_k,6), "delta_logit": round(delta_logit,6),
            "s_k": round(s_k,4), "alpha_before": node.alpha, "beta_before": node.beta,
            "alpha_after": round(alpha_new,6), "beta_after": round(beta_new,6),
            "neutral": abs(sigma-p_k) < 1e-9}
