from __future__ import annotations

import copy
import importlib.util
import io
import math
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import Circle
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

import bayesweb as bw


DEFAULT_MODEL_PATH = "examples/basic_example.py"


def load_web_from_python_file(path_str: str):
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    module_name = f"bayesweb_dynamic_model_{abs(hash((str(path), path.stat().st_mtime_ns)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module depuis : {path}")

    module = importlib.util.module_from_spec(spec)

    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        spec.loader.exec_module(module)

    if hasattr(module, "build_web"):
        web = module.build_web()
    elif hasattr(module, "web"):
        web = module.web
    else:
        raise AttributeError(
            "Le fichier chargé doit exposer soit une fonction build_web(), soit une variable globale web."
        )

    return web, str(path)


def split_edges(web):
    undirected = []
    directed = []
    seen_pairs = set()

    for u, v, data in web._graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        pair = tuple(sorted((u, v)))

        if web._graph.has_edge(v, u):
            w_rev = float(web._graph[v][u].get("weight", 0.0))
            if abs(w - w_rev) < 1e-12:
                if pair not in seen_pairs:
                    undirected.append((pair[0], pair[1], w))
                    seen_pairs.add(pair)
            else:
                directed.append((u, v, w))
        else:
            directed.append((u, v, w))

    return undirected, directed


def circular_zone_positions(web):
    zone_radii = {"N": 0.75, "I": 1.85, "P": 3.0}
    by_zone = {"N": [], "I": [], "P": []}

    for node_id in web.node_ids:
        zone = web.get_node(node_id).zone.value
        by_zone[zone].append(node_id)

    pos = {}

    nucleus_nodes = by_zone["N"]
    if len(nucleus_nodes) == 1:
        pos[nucleus_nodes[0]] = (0.0, 0.0)
    elif len(nucleus_nodes) > 1:
        for i, node_id in enumerate(nucleus_nodes):
            angle = math.pi / 2 + (2 * math.pi * i / len(nucleus_nodes))
            r = zone_radii["N"]
            pos[node_id] = (r * math.cos(angle), r * math.sin(angle))

    for zone in ["I", "P"]:
        nodes = by_zone[zone]
        n = len(nodes)
        if n == 0:
            continue
        start_angle = math.pi / 2
        for i, node_id in enumerate(nodes):
            angle = start_angle - (2 * math.pi * i / n)
            r = zone_radii[zone]
            pos[node_id] = (r * math.cos(angle), r * math.sin(angle))

    return pos


def run_scenario(web, target: str, kappa: float, sigma: float, lam: float = 0.88):
    eta = bw.calibrate_eta(web, lam=lam)

    web_s = copy.deepcopy(web)
    web_s, dl = bw.apply_shock(web_s, target, kappa, sigma, inplace=True)

    shock_info = bw.shock_summary(web, target, kappa, sigma)
    result = bw.propagate(web_s, target, dl, eta=eta)
    summary = bw.indicators_summary(web, result)
    scan = bw.scan_tipping_point(web, target, kappa=kappa, sigma=sigma)

    df = result.to_dataframe().copy()
    df["zone"] = [web.get_node(n).zone.value for n in df["node_id"]]
    df["label"] = [web.get_node(n).label for n in df["node_id"]]

    cols = ["node_id", "label", "zone", "p_initial", "p_final", "delta_p"]
    df = df[cols].sort_values("delta_p", ascending=False)

    return {
        "eta": eta,
        "rho": web.spectral_radius(),
        "shock_info": shock_info,
        "result": result,
        "summary": summary,
        "scan": scan,
        "table": df,
    }


def empty_df():
    return pd.DataFrame(columns=["node_id", "label", "zone", "p_initial", "p_final", "delta_p"])


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Chargement du modèle"),
        ui.input_text("model_path", "Fichier Python modèle", value=DEFAULT_MODEL_PATH),
        ui.input_action_button("reload_btn", "Recharger le fichier"),
        ui.output_text_verbatim("model_status"),
        ui.hr(),
        ui.output_ui("target_ui"),
        ui.input_slider("kappa", "kappa", min=1, max=12, value=8, step=1),
        ui.input_slider("sigma", "sigma", min=0.05, max=0.95, value=0.90, step=0.05),
        ui.input_slider("lam", "lambda (calibrage eta)", min=0.10, max=0.98, value=0.88, step=0.01),
        width="380px",
        style="height: 100vh; overflow-y: auto; padding-bottom: 1rem;",
        open="desktop",
    ),
    ui.tags.div(
        ui.h2("BayesWeb - App pilotée par basic_example.py"),
        ui.row(
            ui.column(6, ui.output_text_verbatim("txt_summary")),
            ui.column(6, ui.output_text_verbatim("txt_shock")),
        ),
        ui.h4("Aide à l'interprétation"),
        ui.markdown(
            """
**F** résume l'intensité interne de la réponse du système.  
Plus **F** est élevé, plus la propagation interne est marquée.

**G** indique si le réseau **amplifie** ou **atténue** le choc initial.  
- **G > 1** : le système amplifie le choc.  
- **G < 1** : le système atténue le choc.

**η critique** est un seuil de bascule théorique.  
Plus il est bas, plus le système peut basculer facilement dans une dynamique forte.
"""
        ),
        ui.h4("Aide sur les paramètres"),
        ui.markdown(
            """
**κ (kappa)** règle la force du choc appliqué au nœud cible.  
Plus **kappa** est élevé, plus le nœud cible est poussé fortement vers une nouvelle valeur.

**σ (sigma)** fixe l'orientation probabiliste du choc.  
Selon sa valeur, le choc peut augmenter ou diminuer la probabilité du nœud cible.

**λ (lambda)** sert à calibrer **η**, c'est-à-dire l'intensité globale de propagation dans le réseau.  
Plus **lambda** est élevé, plus la propagation potentielle entre les nœuds peut être forte.

**p_avant** est la probabilité du nœud cible avant l'application du choc.  
**p_après** est la probabilité du nœud cible juste après le choc, avant la propagation dans tout le réseau.

**delta_logit** mesure l'intensité du changement appliqué au nœud cible dans l'échelle interne du modèle.  
Plus sa valeur absolue est grande, plus le choc est fort.
"""
        ),
        ui.hr(),
        ui.h4("Toile de croyance (zones concentriques)"),
        ui.p("Le réseau, les nœuds, les zones et les liens sont chargés automatiquement depuis le fichier Python indiqué."),
        ui.output_plot("network_plot", height="860px"),
        ui.hr(),
        ui.h4("Résultats par nœud"),
        ui.output_data_frame("tbl_results"),
        style="height: 100vh; overflow-y: auto; padding-right: 12px;",
    ),
    title="BayesWeb Shiny",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def model_data():
        _ = input.reload_btn()
        model_path = input.model_path()

        try:
            web, resolved_path = load_web_from_python_file(model_path)
            node_ids = list(web.node_ids)
            if len(node_ids) == 0:
                raise ValueError("Le modèle chargé ne contient aucun nœud.")

            undirected_edges, directed_edges = split_edges(web)
            return {
                "ok": True,
                "web": web,
                "path": resolved_path,
                "node_ids": node_ids,
                "undirected_edges": undirected_edges,
                "directed_edges": directed_edges,
                "error": "",
            }
        except Exception:
            return {
                "ok": False,
                "web": None,
                "path": model_path,
                "node_ids": [],
                "undirected_edges": [],
                "directed_edges": [],
                "error": traceback.format_exc(),
            }

    @output
    @render.text
    def model_status():
        d = model_data()
        if d["ok"]:
            web = d["web"]
            return (
                f"Modèle chargé : {web.name}\n"
                f"Source : {d['path']}\n"
                f"Nœuds : {len(d['node_ids'])}"
            )
        return f"Erreur de chargement :\n{d['error']}"

    @output
    @render.ui
    def target_ui():
        d = model_data()
        if not d["ok"] or not d["node_ids"]:
            return ui.input_select("target", "Nœud cible", choices={"": "aucun modèle"}, selected="")

        choices = {node_id: node_id for node_id in d["node_ids"]}
        selected = d["node_ids"][0]
        return ui.input_select("target", "Nœud cible", choices=choices, selected=selected)

    @reactive.calc
    def current_target():
        d = model_data()
        if not d["ok"] or not d["node_ids"]:
            return None

        try:
            current = input.target()
        except Exception:
            current = None

        if current in d["node_ids"]:
            return current
        return d["node_ids"][0]

    @reactive.calc
    def scenario_data():
        d = model_data()
        if not d["ok"] or d["web"] is None:
            return {"ok": False, "error": d["error"], "table": empty_df()}

        target = current_target()
        if target is None:
            return {"ok": False, "error": "Le nœud cible n'est pas encore disponible.", "table": empty_df()}

        try:
            out = run_scenario(
                web=d["web"],
                target=target,
                kappa=float(input.kappa()),
                sigma=float(input.sigma()),
                lam=float(input.lam()),
            )
            out["ok"] = True
            out["target"] = target
            out["web"] = d["web"]
            out["undirected_edges"] = d["undirected_edges"]
            out["directed_edges"] = d["directed_edges"]
            return out
        except Exception:
            return {"ok": False, "error": traceback.format_exc(), "table": empty_df()}

    @output
    @render.text
    def txt_summary():
        d = scenario_data()
        if not d["ok"]:
            return f"Simulation impossible.\n\n{d['error']}"

        s = d["summary"]
        r = d["result"]
        web = d["web"]

        return (
            f"Réseau : {web.name}\n"
            f"rho(W) = {d['rho']:.4f}\n"
            f"eta = {d['eta']:.4f}\n"
            f"eta*rho = {d['eta'] * d['rho']:.4f}\n\n"
            f"Convergence : {'oui' if r.converged else 'non'}\n"
            f"Itérations : {r.iterations}\n\n"
            f"F = {s['F']:.4f} ({s['F_interpretation']})\n"
            f"G = {s['G']:.4f} ({s['G_interpretation']})\n"
            f"Seuil de bascule eta_critique ≈ {d['scan'].eta_critical:.4f}"
        )

    @output
    @render.text
    def txt_shock():
        d = scenario_data()
        if not d["ok"]:
            return "Aucun choc calculé."

        i = d["shock_info"]

        return (
            f"Cible : {d['target']}\n"
            f"kappa = {float(input.kappa()):.2f}\n"
            f"sigma = {float(input.sigma()):.2f}\n\n"
            f"p_avant = {i['p_before']:.4f}\n"
            f"p_après = {i['p_after']:.4f}\n"
            f"delta_logit = {i['delta_logit']:+.4f}"
        )

    @output
    @render.plot
    def network_plot():
        d = scenario_data()
        if not d["ok"]:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.text(0.5, 0.5, "Modèle non disponible", ha="center", va="center", fontsize=16)
            ax.text(0.5, 0.42, d["error"], ha="center", va="center", fontsize=9, wrap=True)
            ax.axis("off")
            return fig

        web = d["web"]
        target = d["target"]
        df = d["table"].copy()
        undirected_edges = d["undirected_edges"]
        directed_edges = d["directed_edges"]

        delta_map = {row["node_id"]: float(row["delta_p"]) for _, row in df.iterrows()}
        max_delta = max(delta_map.values()) if delta_map else 0.0

        G = nx.DiGraph()
        for node_id in web.node_ids:
            node = web.get_node(node_id)
            G.add_node(node_id, label=node.label, zone=node.zone.value)

        for u, v, w in directed_edges:
            G.add_edge(u, v, weight=w)

        pos = circular_zone_positions(web)

        fig, ax = plt.subplots(figsize=(10.5, 10.5))

        outer = Circle((0, 0), 3.35, facecolor="#69a7e0", edgecolor="none", alpha=0.35, zorder=0)
        middle = Circle((0, 0), 2.2, facecolor="none", edgecolor="white", linewidth=2, alpha=0.55, zorder=0)
        inner = Circle((0, 0), 1.05, facecolor="#ff8f8f", edgecolor="none", alpha=0.16, zorder=0)

        ax.add_patch(outer)
        ax.add_patch(middle)
        ax.add_patch(inner)

        ax.text(0, 0, "Nucléus", ha="center", va="center", fontsize=12, weight="bold", alpha=0.65)
        ax.text(0, 1.6, "Intermédiaire", ha="center", va="center", fontsize=11, alpha=0.75)
        ax.text(0, 3.0, "Périphérie", ha="center", va="center", fontsize=11, alpha=0.75)

        node_colors = []
        node_sizes = []
        border_widths = []

        for node_id in web.node_ids:
            node = web.get_node(node_id)
            delta = delta_map.get(node_id, 0.0)
            normalized = 0.0 if max_delta == 0 else delta / max_delta
            size = 1100 + normalized * 6200

            if node_id == target:
                node_colors.append("orange")
                size += 1000
                border_widths.append(2.8)
            else:
                if node.zone.value == "N":
                    node_colors.append("#ee7b7b")
                elif node.zone.value == "I":
                    node_colors.append("#9ecae1")
                else:
                    node_colors.append("#8ddf8a")
                border_widths.append(1.8)

            node_sizes.append(size)

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=web.node_ids,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#222222",
            linewidths=border_widths,
            ax=ax,
        )

        label_pos = {node_id: (x, y + 0.14) for node_id, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            labels={node_id: web.get_node(node_id).label for node_id in web.node_ids},
            font_size=10,
            ax=ax,
        )

        positive_undirected = [(u, v) for u, v, w in undirected_edges if w >= 0]
        negative_undirected = [(u, v) for u, v, w in undirected_edges if w < 0]

        positive_widths = [1.6 + abs(w) * 2.5 for u, v, w in undirected_edges if w >= 0]
        negative_widths = [1.6 + abs(w) * 2.5 for u, v, w in undirected_edges if w < 0]

        if positive_undirected:
            nx.draw_networkx_edges(
                nx.Graph(positive_undirected),
                pos,
                edgelist=positive_undirected,
                width=positive_widths,
                edge_color="#333333",
                ax=ax,
            )

        if negative_undirected:
            nx.draw_networkx_edges(
                nx.Graph(negative_undirected),
                pos,
                edgelist=negative_undirected,
                width=negative_widths,
                edge_color="#cc3333",
                style="dashed",
                ax=ax,
            )

        if undirected_edges:
            undirected_graph = nx.Graph([(u, v) for u, v, _ in undirected_edges])
            undirected_labels = {(u, v): f"{w:.2f}" for u, v, w in undirected_edges}
            nx.draw_networkx_edge_labels(
                undirected_graph,
                pos,
                edge_labels=undirected_labels,
                font_size=9,
                ax=ax,
                rotate=False,
                bbox={"alpha": 0.72, "pad": 0.10},
            )

        positive_directed = [(u, v) for u, v, w in directed_edges if w >= 0]
        negative_directed = [(u, v) for u, v, w in directed_edges if w < 0]

        positive_directed_widths = [1.8 + abs(w) * 2.6 for u, v, w in directed_edges if w >= 0]
        negative_directed_widths = [1.8 + abs(w) * 2.6 for u, v, w in directed_edges if w < 0]

        if positive_directed:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=positive_directed,
                width=positive_directed_widths,
                edge_color="#222222",
                arrows=True,
                arrowstyle="->",
                arrowsize=24,
                ax=ax,
                connectionstyle="arc3,rad=0.08",
            )

        if negative_directed:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=negative_directed,
                width=negative_directed_widths,
                edge_color="#cc3333",
                arrows=True,
                arrowstyle="->",
                arrowsize=24,
                style="dashed",
                ax=ax,
                connectionstyle="arc3,rad=0.08",
            )

        if directed_edges:
            directed_labels = {(u, v): f"{w:.2f}" for u, v, w in directed_edges}
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=directed_labels,
                font_size=9,
                ax=ax,
                rotate=False,
                bbox={"alpha": 0.72, "pad": 0.10},
            )

        sorted_rows = list(df.sort_values("delta_p", ascending=False).itertuples(index=False))
        lines = ["Nœuds les plus affectés :"]
        for row in sorted_rows[:3]:
            lines.append(f"- {row.node_id}: Δp = {row.delta_p:.4f}")

        ax.text(
            -3.2,
            -3.2,
            "\n".join(lines),
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"alpha": 0.88, "pad": 0.45},
        )

        ax.set_title(
            f"Toile de croyance — chargée depuis le fichier Python — cible : {target}",
            fontsize=15,
        )
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        return fig

    @output
    @render.data_frame
    def tbl_results():
        d = scenario_data()
        return render.DataGrid(d["table"].copy(), filters=False)


app = App(app_ui, server)
