from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .graph import BeliefWeb

@dataclass
class PropagationResult:
    """Full output of a propagation run.  Eq. (2.7)-(2.8)."""
    node_ids:           list
    p_initial:          np.ndarray
    p_final:            np.ndarray
    delta_logit_cumul:  np.ndarray
    history:            np.ndarray
    iterations:         int
    converged:          bool
    eta:                float
    rho_W:              float
    eta_rho:            float

    @property
    def delta_p(self)      -> np.ndarray: return np.abs(self.p_final - self.p_initial)
    @property
    def mean_delta_p(self) -> float:      return float(np.mean(self.delta_p))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "node_id":           self.node_ids,
            "p_initial":         self.p_initial,
            "p_final":           self.p_final,
            "delta_p":           self.delta_p,
            "delta_logit_cumul": self.delta_logit_cumul,
        })

def _sigmoid(x):  return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
def _logit_v(p):  p=np.clip(p,1e-12,1-1e-12); return np.log(p/(1-p))

def calibrate_eta(web: BeliefWeb, lam: float = 0.9) -> float:
    """eta = lam / rho(W).  Remark 3."""
    if not 0<lam<1: raise ValueError(f"lam must be in (0,1), got {lam}.")
    rho = web.spectral_radius()
    if rho < 1e-12:
        warnings.warn("rho(W) ~ 0. Returning eta=lam."); return float(lam)
    return float(lam/rho)

def propagate(web, target, delta_logit_seed, eta=None, lam=0.9,
              max_iter=500, tol=1e-8) -> PropagationResult:
    """
    Iterative propagation scheme.  x^(t+1) = eta * W^T * x^(t).  Eq. (2.7).
    Convergence requires eta*rho(W) < 1  (Proposition 1).
    """
    W, ids = web.weight_matrix(); n = len(ids)
    if n==0: raise ValueError("BeliefWeb has no nodes.")
    idx = {nid:k for k,nid in enumerate(ids)}
    rho_W = web.spectral_radius()
    if eta is None: eta = calibrate_eta(web, lam=lam)
    eta_rho = eta * rho_W
    if eta_rho >= 1.0:
        warnings.warn(f"eta*rho(W)={eta_rho:.4f} >= 1 — convergence not guaranteed.", RuntimeWarning)
    p_initial  = np.array([web.get_node(nid).belief for nid in ids])
    logit_init = _logit_v(p_initial)
    if target not in idx: raise KeyError(f"Target '{target}' not found.")
    x = np.zeros(n); x[idx[target]] = delta_logit_seed
    WT = W.T; cumul = np.zeros(n); history = [x.copy()]; converged = False
    for _ in range(max_iter):
        cumul += x; x_new = eta*(WT@x); history.append(x_new.copy())
        if np.max(np.abs(x_new)) < tol:
            converged=True; cumul+=x_new; x=x_new; break
        x = x_new
    return PropagationResult(
        node_ids=ids, p_initial=p_initial, p_final=_sigmoid(logit_init+cumul),
        delta_logit_cumul=cumul, history=np.array(history),
        iterations=len(history)-1, converged=converged,
        eta=eta, rho_W=rho_W, eta_rho=eta_rho)

def theoretical_response(web, target, delta_logit_seed, eta) -> np.ndarray:
    """Exact Neumann solution (I - eta*W^T)^{-1} x^(0). For validation."""
    W, ids = web.weight_matrix(); n = len(ids)
    idx = {nid:k for k,nid in enumerate(ids)}
    x0 = np.zeros(n); x0[idx[target]] = delta_logit_seed
    return np.linalg.solve(np.eye(n)-eta*W.T, x0)
