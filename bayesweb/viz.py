from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from .graph import BeliefWeb, Zone
from .propagation import PropagationResult

_ZONE_COLOUR = {Zone.NUCLEUS:"#7F77DD", Zone.INTERMEDIATE:"#1D9E75", Zone.PERIPHERY:"#888780"}
_ZONE_SHAPE  = {Zone.NUCLEUS:"s",       Zone.INTERMEDIATE:"o",       Zone.PERIPHERY:"^"}

def plot_web(web, result=None, layout="spring", figsize=(10,7), title=None, save_path=None):
    """Network graph. Node fill = belief p_i (RdYlGn). Border = zone."""
    G = web._graph; ids = web.node_ids
    nodes = [web.get_node(nid) for nid in ids]
    layouts = {"spring":nx.spring_layout,"kamada_kawai":nx.kamada_kawai_layout,
               "circular":nx.circular_layout,"shell":nx.shell_layout}
    pos = layouts.get(layout, nx.spring_layout)(G, seed=42)
    fig, ax = plt.subplots(figsize=figsize)
    beliefs = np.array([n.belief for n in nodes])
    norm = mcolors.Normalize(vmin=0,vmax=1); cmap = cm.RdYlGn
    node_fill = [cmap(norm(p)) for p in beliefs]
    base_size = 900
    if result is not None:
        res_idx = {nid:k for k,nid in enumerate(result.node_ids)}
        dp = np.array([result.delta_p[res_idx[nid]] if nid in res_idx else 0 for nid in ids])
        node_sizes = base_size + 2500*dp/(dp.max()+1e-9)
    else:
        node_sizes = [base_size]*len(ids)
    for zone in [Zone.NUCLEUS, Zone.INTERMEDIATE, Zone.PERIPHERY]:
        znids = [nid for nid in ids if web.get_node(nid).zone==zone]
        if not znids: continue
        nx.draw_networkx_nodes(G, {n:pos[n] for n in znids}, nodelist=znids,
            node_color=[node_fill[ids.index(n)] for n in znids],
            node_size=[node_sizes[ids.index(n)] for n in znids],
            node_shape=_ZONE_SHAPE[zone], edgecolors=_ZONE_COLOUR[zone],
            linewidths=2.5, ax=ax)
    pos_e = [(u,v) for u,v,d in G.edges(data=True) if d.get("weight",0)>=0]
    neg_e = [(u,v) for u,v,d in G.edges(data=True) if d.get("weight",0)<0]
    nx.draw_networkx_edges(G,pos,edgelist=pos_e,width=[abs(G[u][v]["weight"])*3 for u,v in pos_e],
        edge_color="#1D9E75",alpha=0.7,arrows=True,arrowsize=15,ax=ax)
    nx.draw_networkx_edges(G,pos,edgelist=neg_e,width=[abs(G[u][v]["weight"])*3 for u,v in neg_e],
        edge_color="#D85A30",alpha=0.7,style="dashed",arrows=True,arrowsize=15,ax=ax)
    nx.draw_networkx_labels(G,pos,labels={n:n for n in ids},font_size=9,font_weight="bold",ax=ax)
    sm = cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
    fig.colorbar(sm,ax=ax,shrink=0.6,pad=0.02).set_label("Degree of belief p_i",fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker="s",color="w",markerfacecolor="white",markeredgecolor=_ZONE_COLOUR[Zone.NUCLEUS],markersize=10,label="Nucleus V_N"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor=_ZONE_COLOUR[Zone.INTERMEDIATE],markersize=10,label="Intermediate V_I"),
        Line2D([0],[0],marker="^",color="w",markerfacecolor="white",markeredgecolor=_ZONE_COLOUR[Zone.PERIPHERY],markersize=10,label="Periphery V_P"),
        Line2D([0],[0],color="#1D9E75",linewidth=2,label="Support (+)"),
        Line2D([0],[0],color="#D85A30",linewidth=2,linestyle="--",label="Tension (-)"),
    ], loc="upper left", fontsize=9)
    ax.set_title(title or f"Belief web — {web.name}",fontsize=13); ax.axis("off")
    fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    return fig

def plot_propagation(result, web, top_n=20, figsize=(10,5), title=None, save_path=None):
    """Horizontal bar chart of delta_p per node."""
    df = result.to_dataframe()
    df["zone"]  = [web.get_node(n).zone.value for n in df["node_id"]]
    df["label"] = [web.get_node(n).label[:40] for n in df["node_id"]]
    df = df.nlargest(top_n,"delta_p")
    cols = {"N":"#7F77DD","I":"#1D9E75","P":"#888780"}
    fig,ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["label"],df["delta_p"],color=[cols.get(z,"#888780") for z in df["zone"]],
                   edgecolor="white",linewidth=0.5)
    ax.set_xlabel("delta_p (belief shift)",fontsize=11)
    ax.set_title(title or "Belief shift after propagation",fontsize=13)
    ax.invert_yaxis()
    for bar,(_,row) in zip(bars,df.iterrows()):
        ax.text(bar.get_width()+0.002,bar.get_y()+bar.get_height()/2,
                f"{row['p_initial']:.2f} -> {row['p_final']:.2f}",
                va="center",ha="left",fontsize=8,color="#444441")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c,label=f"Zone {z}") for z,c in cols.items()],fontsize=9)
    fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    return fig

def plot_tipping(scan, figsize=(8,5), title=None, save_path=None):
    """G vs eta*rho(W) curve with tipping threshold."""
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(scan.eta_rho_values, scan.G_values, color="#7F77DD", linewidth=2)
    ax.axvline(x=1.0,color="#D85A30",linestyle="--",linewidth=1.5,label="eta*rho=1 (tipping)")
    ax.axhline(y=1.0,color="#888780",linestyle=":",linewidth=1.0,label="G=1 (break-even)")
    ax.set_xlabel("eta * rho(W)",fontsize=11); ax.set_ylabel("Propagation gain G",fontsize=11)
    ax.set_ylim(bottom=0); ax.set_title(title or "Tipping-point scan",fontsize=13)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    return fig

def plot_convergence(result, figsize=(7,4), save_path=None):
    """||x^(t)||_inf over iterations."""
    norms = np.max(np.abs(result.history),axis=1)
    fig,ax = plt.subplots(figsize=figsize)
    ax.semilogy(norms,color="#7F77DD",linewidth=1.8,marker="o",markersize=3)
    ax.axhline(y=1e-8,color="#D85A30",linestyle="--",linewidth=1,label="tol=1e-8")
    ax.set_xlabel("Iteration t",fontsize=11); ax.set_ylabel("||x^(t)||_inf",fontsize=11)
    ax.set_title(f"Convergence ({'OK' if result.converged else 'NOT converged'})",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    return fig
