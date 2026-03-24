from __future__ import annotations
import json, warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd

class Zone(str, Enum):
    NUCLEUS      = "N"
    INTERMEDIATE = "I"
    PERIPHERY    = "P"

@dataclass
class BeliefNode:
    node_id: str
    label:   str
    alpha:   float = 1.0
    beta:    float = 1.0
    zone:    Zone  = Zone.INTERMEDIATE

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Node '{self.node_id}': alpha and beta must be > 0.")

    @property
    def belief(self)  -> float: return self.alpha / (self.alpha + self.beta)
    @property
    def solidity(self)-> float: return self.alpha + self.beta
    @property
    def logit_belief(self) -> float:
        p = float(np.clip(self.belief, 1e-12, 1-1e-12))
        return float(np.log(p/(1-p)))

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "label": self.label,
                "alpha": self.alpha, "beta": self.beta, "zone": self.zone.value,
                "belief": round(self.belief,6), "solidity": round(self.solidity,4)}

class BeliefWeb:
    """
    Weighted signed belief web  G = (V, E, W).
    Each node carries Beta(alpha, beta).  Eq. (2.1)-(2.3).
    """
    def __init__(self, name: str = "BeliefWeb") -> None:
        self.name = name
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, BeliefNode] = {}

    def add_node(self, node_id, label, alpha=1.0, beta=1.0,
                 zone=Zone.INTERMEDIATE) -> "BeliefWeb":
        if isinstance(zone, str): zone = Zone(zone)
        node = BeliefNode(node_id=node_id, label=label, alpha=alpha, beta=beta, zone=zone)
        self._nodes[node_id] = node
        self._graph.add_node(node_id, **node.to_dict())
        return self

    def get_node(self, node_id: str) -> BeliefNode:
        if node_id not in self._nodes: raise KeyError(f"Node '{node_id}' not found.")
        return self._nodes[node_id]

    def update_node_belief(self, node_id, alpha, beta) -> None:
        node = self.get_node(node_id)
        if alpha <= 0 or beta <= 0: raise ValueError("alpha and beta must be > 0.")
        node.alpha, node.beta = alpha, beta
        nx.set_node_attributes(self._graph, {node_id: {
            "alpha": alpha, "beta": beta,
            "belief": node.belief, "solidity": node.solidity}})

    def add_edge(self, source, target, weight) -> "BeliefWeb":
        if source not in self._nodes: raise KeyError(f"Source '{source}' not found.")
        if target not in self._nodes: raise KeyError(f"Target '{target}' not found.")
        if not -1.0 <= weight <= 1.0: raise ValueError(f"Weight {weight} outside [-1,1].")
        self._graph.add_edge(source, target, weight=weight)
        return self

    def add_undirected_edge(self, i, j, weight) -> "BeliefWeb":
        return self.add_edge(i, j, weight).add_edge(j, i, weight)

    @property
    def node_ids(self) -> list: return list(self._nodes.keys())

    def weight_matrix(self) -> Tuple[np.ndarray, list]:
        ids = self.node_ids; n = len(ids)
        idx = {nid: k for k,nid in enumerate(ids)}
        W   = np.zeros((n,n))
        for u,v,d in self._graph.edges(data=True):
            W[idx[u], idx[v]] = d.get("weight", 0.0)
        return W, ids

    def spectral_radius(self) -> float:
        W, _ = self.weight_matrix()
        return 0.0 if W.size==0 else float(np.max(np.abs(np.linalg.eigvals(W))))

    def nodes_by_zone(self, zone):
        return [nid for nid,n in self._nodes.items() if n.zone==zone]

    @property
    def nucleus(self)       -> list: return self.nodes_by_zone(Zone.NUCLEUS)
    @property
    def intermediate(self)  -> list: return self.nodes_by_zone(Zone.INTERMEDIATE)
    @property
    def periphery(self)     -> list: return self.nodes_by_zone(Zone.PERIPHERY)

    def auto_partition(self, lambda1=1/3, lambda2=1/3, lambda3=1/3) -> "BeliefWeb":
        W, ids = self.weight_matrix(); n = len(ids)
        if n == 0: return self
        deg   = np.sum(np.abs(W),axis=1) + np.sum(np.abs(W),axis=0)
        gamma = lambda1 * (deg / (deg.max() or 1.0))
        t33, t67 = np.percentile(gamma,33.33), np.percentile(gamma,66.67)
        for k,nid in enumerate(ids):
            if   gamma[k] >= t67: self._nodes[nid].zone = Zone.NUCLEUS
            elif gamma[k] <  t33: self._nodes[nid].zone = Zone.PERIPHERY
            else:                 self._nodes[nid].zone = Zone.INTERMEDIATE
        return self

    def to_dict(self) -> dict:
        return {"name": self.name,
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [{"source":u,"target":v,"weight":d["weight"]}
                          for u,v,d in self._graph.edges(data=True)]}

    def to_json(self, path: str) -> None:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BeliefWeb":
        web = cls(name=data.get("name","BeliefWeb"))
        for n in data["nodes"]:
            web.add_node(n["node_id"],n["label"],
                         alpha=n.get("alpha",1.0),beta=n.get("beta",1.0),
                         zone=Zone(n.get("zone","I")))
        for e in data.get("edges",[]):
            web.add_edge(e["source"],e["target"],e["weight"])
        return web

    @classmethod
    def from_json(cls, path: str) -> "BeliefWeb":
        with open(path,"r",encoding="utf-8") as f: return cls.from_dict(json.load(f))

    @classmethod
    def from_dataframes(cls, nodes_df, edges_df, name="BeliefWeb") -> "BeliefWeb":
        web = cls(name=name)
        for _,row in nodes_df.iterrows():
            web.add_node(str(row["node_id"]),str(row["label"]),
                         alpha=float(row.get("alpha",1.0)),beta=float(row.get("beta",1.0)),
                         zone=Zone(str(row.get("zone","I"))))
        for _,row in edges_df.iterrows():
            web.add_edge(str(row["source"]),str(row["target"]),float(row["weight"]))
        return web

    def summary(self) -> str:
        n,e = len(self._nodes), self._graph.number_of_edges()
        rho = self.spectral_radius()
        lines = [f"BeliefWeb '{self.name}'",
                 f"  Nodes : {n}  (N={len(self.nucleus)}, I={len(self.intermediate)}, P={len(self.periphery)})",
                 f"  Edges : {e}", f"  rho(W): {rho:.4f}"]
        for nid,node in self._nodes.items():
            lines.append(f"  [{node.zone.value}] {nid:15s} p={node.belief:.3f}  s={node.solidity:.1f}  «{node.label[:50]}»")
        return "\n".join(lines)

    def __repr__(self): return f"BeliefWeb(name={self.name!r}, nodes={len(self._nodes)}, edges={self._graph.number_of_edges()})"
