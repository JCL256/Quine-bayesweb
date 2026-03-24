from __future__ import annotations
import warnings, copy
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .graph import BeliefWeb, Zone
from .propagation import PropagationResult

def fragility(web: BeliefWeb) -> float:
    """F = s_bar_notN / (s_bar_N + s_bar_notN).  Eq. (2.9)."""
    nuc = web.nucleus; non = [n for n in web.node_ids if n not in nuc]
    if not nuc:  raise ValueError("Nucleus is empty. Assign zone='N' or call auto_partition().")
    if not non:  warnings.warn("All nodes in nucleus — F undefined."); return 0.0
    sN   = np.mean([web.get_node(n).solidity for n in nuc])
    snN  = np.mean([web.get_node(n).solidity for n in non])
    denom = sN + snN
    if denom < 1e-12: warnings.warn("All solidities~0."); return 0.5
    return float(snN / denom)

def fragility_bootstrap(web, n_bootstrap=500, seed=None):
    """95% CI for F by perturbing zone assignments (B=500 resamples)."""
    rng = np.random.default_rng(seed)
    ids = web.node_ids; zones = [web.get_node(n).zone for n in ids]
    solidities = np.array([web.get_node(n).solidity for n in ids])
    samples = []
    for _ in range(n_bootstrap):
        z2 = list(zones)
        for k in range(len(ids)):
            if rng.random() < 0.1:
                z2[k] = rng.choice([Zone.NUCLEUS, Zone.INTERMEDIATE, Zone.PERIPHERY])
        nm = np.array([z==Zone.NUCLEUS for z in z2])
        if nm.sum()==0 or (~nm).sum()==0: continue
        sN  = solidities[nm].mean(); snN = solidities[~nm].mean()
        d   = sN + snN
        if d > 1e-12: samples.append(snN/d)
    if not samples: F=fragility(web); return F,F,F
    a = np.array(samples)
    return float(fragility(web)), float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

def _baseline_delta0(web) -> float:
    """Delta_0 = 1/(2*(s_bar+1)).  Eq. (2.10)."""
    s_bar = np.mean([web.get_node(n).solidity for n in web.node_ids])
    return 1.0/(2.0*(s_bar+1.0))

def propagation_gain(result: PropagationResult, web: BeliefWeb) -> float:
    """G = Delta^(T) / Delta_0.  Eq. (2.11). G->inf signals tipping (Prop. 2)."""
    d0 = _baseline_delta0(web)
    if d0 < 1e-15: warnings.warn("Delta_0~0."); return float("inf")
    return float(result.mean_delta_p / d0)

@dataclass
class TippingPointScan:
    eta_values:    object
    G_values:      object
    eta_critical:  float
    G_at_critical: float
    eta_rho_values: object

def scan_tipping_point(web, target, kappa, sigma, eta_values=None, max_iter=300):
    """Scan eta values and compute G at each step.  Identifies tipping threshold."""
    from .shock import apply_shock
    from .propagation import propagate
    rho_W = web.spectral_radius()
    if eta_values is None:
        eta_max = (0.99/rho_W) if rho_W>1e-12 else 10.0
        eta_values = np.linspace(0.01, eta_max, 50)
    else:
        eta_values = np.asarray(eta_values)
    G_vals = []
    for eta in eta_values:
        wc = copy.deepcopy(web)
        wc, dl = apply_shock(wc, target, kappa, sigma, inplace=True)
        res = propagate(wc, target, dl, eta=eta, max_iter=max_iter)
        G_vals.append(propagation_gain(res, web))
    G_arr = np.array(G_vals)
    over  = np.where(G_arr>10.0)[0]
    eta_c = float(eta_values[over[0]]) if len(over)>0 else float(eta_values[-1])
    G_c   = float(G_arr[over[0]])     if len(over)>0 else float(G_arr[-1])
    return TippingPointScan(eta_values=eta_values, G_values=G_arr,
                            eta_critical=eta_c, G_at_critical=G_c,
                            eta_rho_values=eta_values*rho_W)

def indicators_summary(web, result) -> dict:
    F = fragility(web); G = propagation_gain(result, web)
    return {"F": round(F,6), "G": round(G,6),
            "converged": result.converged, "iterations": result.iterations,
            "eta": round(result.eta,6), "eta_rho": round(result.eta_rho,6),
            "rho_W": round(result.rho_W,6),
            "mean_delta_p": round(result.mean_delta_p,8),
            "F_interpretation": "robust" if F<0.4 else ("vulnerable" if F>0.6 else "intermediate"),
            "G_interpretation": "attenuated" if G<1 else ("amplified" if G<10 else "near tipping point")}
