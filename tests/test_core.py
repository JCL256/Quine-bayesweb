"""Unit tests for bayesweb core.  Run: pytest tests/ -v"""
import copy, numpy as np, pytest
from bayesweb import BeliefWeb, Zone, apply_shock, shock_summary, propagate, calibrate_eta
from bayesweb import fragility, propagation_gain, indicators_summary

@pytest.fixture
def web3():
    w = BeliefWeb("t")
    w.add_node("A","core",  alpha=4,beta=1,zone=Zone.NUCLEUS)
    w.add_node("B","inter", alpha=2,beta=2,zone=Zone.INTERMEDIATE)
    w.add_node("C","periph",alpha=1,beta=3,zone=Zone.PERIPHERY)
    w.add_undirected_edge("A","B",0.8); w.add_edge("B","C",0.6)
    return w

def test_belief_formula():
    w=BeliefWeb(); w.add_node("X","t",alpha=3,beta=1)
    assert abs(w.get_node("X").belief - 0.75) < 1e-9
    assert abs(w.get_node("X").solidity - 4.0) < 1e-9

def test_invalid_params():
    w=BeliefWeb()
    with pytest.raises(ValueError): w.add_node("X","t",alpha=0,beta=1)
    w.add_node("A","a"); w.add_node("B","b")
    with pytest.raises(ValueError): w.add_edge("A","B",weight=1.5)

def test_shock_update_eq24_25():
    w=BeliefWeb(); w.add_node("k","t",alpha=4,beta=1,zone=Zone.NUCLEUS)
    info = shock_summary(w,"k",5,0.8)
    p_exp = (5*0.8 + 5*0.8)/10   # (s*p + k*sigma)/(s+k), s=5,p=0.8
    assert abs(info["p_after"]-p_exp) < 1e-9

def test_neutral_shock_remark2():
    w=BeliefWeb(); w.add_node("B","t",alpha=2,beta=2,zone=Zone.NUCLEUS)
    assert abs(shock_summary(w,"B",3,0.5)["delta_logit"]) < 1e-9

def test_no_inplace(web3):
    a0 = web3.get_node("A").alpha
    apply_shock(web3,"A",5,0.9,inplace=False)
    assert web3.get_node("A").alpha == a0

def test_convergence_prop1(web3):
    wc=copy.deepcopy(web3); _, dl = apply_shock(wc,"A",5,0.9,inplace=True)
    res = propagate(wc,"A",dl,lam=0.85)
    assert res.converged
    assert res.eta_rho < 1.0
    assert np.all(res.p_final>0) and np.all(res.p_final<1)

def test_neumann_series(web3):
    from bayesweb.propagation import theoretical_response
    wc=copy.deepcopy(web3); _, dl = apply_shock(wc,"A",5,0.9,inplace=True)
    res  = propagate(wc,"A",dl,lam=0.85,tol=1e-12,max_iter=2000)
    exact = theoretical_response(wc,"A",dl,res.eta)
    np.testing.assert_allclose(res.delta_logit_cumul, exact, atol=1e-5)

def test_isolated_no_propagation():
    w=BeliefWeb(); w.add_node("A","s",alpha=2,beta=2,zone=Zone.NUCLEUS)
    w.add_node("B","i",alpha=2,beta=2,zone=Zone.PERIPHERY)
    wc=copy.deepcopy(w); _, dl = apply_shock(wc,"A",5,0.9,inplace=True)
    res = propagate(wc,"A",dl,lam=0.5,max_iter=10)
    assert abs(res.delta_logit_cumul[res.node_ids.index("B")]) < 1e-12

def test_fragility_robust():
    w=BeliefWeb()
    w.add_node("N","n",alpha=100,beta=10,zone=Zone.NUCLEUS)
    w.add_node("P","p",alpha=1,beta=1,zone=Zone.PERIPHERY)
    assert fragility(w) < 0.05

def test_fragility_vulnerable():
    w=BeliefWeb()
    w.add_node("N","n",alpha=1,beta=1,zone=Zone.NUCLEUS)
    w.add_node("P","p",alpha=100,beta=10,zone=Zone.PERIPHERY)
    assert fragility(w) > 0.9

def test_indicators_summary(web3):
    wc=copy.deepcopy(web3); _, dl = apply_shock(wc,"A",5,0.9,inplace=True)
    res = propagate(wc,"A",dl)
    s   = indicators_summary(web3,res)
    assert all(k in s for k in ["F","G","converged","eta","eta_rho","rho_W"])
    assert s["G"] > 0

def test_json_roundtrip(web3, tmp_path):
    p = str(tmp_path/"w.json"); web3.to_json(p)
    r = BeliefWeb.from_json(p)
    assert len(r.node_ids)==3
    assert r._graph.number_of_edges()==web3._graph.number_of_edges()

def test_from_dataframes():
    import pandas as pd
    ndf = pd.DataFrame([{"node_id":"X","label":"x","alpha":2,"beta":1,"zone":"N"},
                        {"node_id":"Y","label":"y","alpha":1,"beta":1,"zone":"P"}])
    edf = pd.DataFrame([{"source":"X","target":"Y","weight":0.5}])
    w   = BeliefWeb.from_dataframes(ndf,edf)
    assert len(w.node_ids)==2
