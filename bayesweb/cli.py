"""cli.py — Command-line interface.  Commands: run | scan | viz | init"""
from __future__ import annotations
import json, sys
from pathlib import Path
import click

@click.group()
@click.version_option("0.1.0")
def main():
    """bayesweb — Bayesian belief web with tipping-point detection."""

@main.command()
@click.argument("output", default="my_web.json")
def init(output):
    """Generate a minimal template JSON belief web file."""
    template = {"name":"MyBeliefWeb","nodes":[
        {"node_id":"A","label":"Urban sprawl reduces agricultural land","alpha":3.0,"beta":1.0,"zone":"N"},
        {"node_id":"B","label":"Biodiversity declines in peri-urban areas","alpha":2.0,"beta":2.0,"zone":"I"},
        {"node_id":"C","label":"Ecosystem services are monetised","alpha":1.0,"beta":3.0,"zone":"P"},
        {"node_id":"D","label":"Policy instruments favour densification","alpha":2.0,"beta":1.5,"zone":"I"},
    ],"edges":[
        {"source":"A","target":"B","weight":0.75},{"source":"A","target":"D","weight":-0.50},
        {"source":"B","target":"C","weight":0.60},{"source":"D","target":"A","weight":-0.40},
        {"source":"D","target":"C","weight":0.30},
    ]}
    Path(output).write_text(json.dumps(template,indent=2,ensure_ascii=False))
    click.echo(f"Template written to {output}")
    click.echo(f"Edit it, then run:  bayesweb run {output} --target A --kappa 5 --sigma 0.9")

@main.command()
@click.argument("web_file")
@click.option("--target","-t",required=True)
@click.option("--kappa", "-k",default=5.0,show_default=True)
@click.option("--sigma", "-s",default=0.9,show_default=True)
@click.option("--lam",   "-l",default=0.9,show_default=True)
@click.option("--max-iter",default=500,show_default=True)
@click.option("--output-json",default=None)
@click.option("--plot",is_flag=True,default=False)
@click.option("--save-plot",default=None)
def run(web_file,target,kappa,sigma,lam,max_iter,output_json,plot,save_plot):
    """Apply a shock to TARGET and propagate through the belief web."""
    import copy
    from .graph import BeliefWeb
    from .shock import apply_shock, shock_summary
    from .propagation import propagate
    from .indicators import fragility, propagation_gain, indicators_summary

    click.echo(f"=== bayesweb run  {web_file} ===")
    try:    web = BeliefWeb.from_json(web_file)
    except Exception as e: click.echo(f"Error: {e}"); sys.exit(1)
    click.echo(web.summary())
    info = shock_summary(web, target, kappa, sigma)
    click.echo(f"\nShock on '{target}':  p {info['p_before']:.4f} -> {info['p_after']:.4f}"
               f"   delta_logit={info['delta_logit']:+.4f}")
    wc = copy.deepcopy(web); wc, dl = apply_shock(wc,target,kappa,sigma,inplace=True)
    res = propagate(wc,target,dl,lam=lam,max_iter=max_iter)
    s   = indicators_summary(web,res)
    click.echo(f"\nF = {s['F']:.4f}  ({s['F_interpretation']})")
    click.echo(f"G = {s['G']:.4f}  ({s['G_interpretation']})")
    click.echo(f"eta*rho(W) = {s['eta_rho']:.4f}   converged={s['converged']}  iter={s['iterations']}")
    df = res.to_dataframe().nlargest(8,"delta_p")
    click.echo("\nTop nodes by |delta_p|:")
    for _,row in df.iterrows():
        z = web.get_node(row["node_id"]).zone.value
        click.echo(f"  [{z}] {row['node_id']:15s} {row['p_initial']:.3f} -> {row['p_final']:.3f}  "
                   f"delta={row['delta_p']:+.4f}")
    if output_json:
        out = {"shock":info,"indicators":s,"nodes":res.to_dataframe().to_dict(orient="records")}
        Path(output_json).write_text(json.dumps(out,indent=2)); click.echo(f"Results saved to {output_json}")
    if plot or save_plot:
        from .viz import plot_propagation
        import matplotlib.pyplot as plt
        fig = plot_propagation(res,web,save_path=save_plot)
        if plot: plt.show()

@main.command()
@click.argument("web_file")
@click.option("--target","-t",required=True)
@click.option("--kappa", "-k",default=5.0,show_default=True)
@click.option("--sigma", "-s",default=0.9,show_default=True)
@click.option("--plot",is_flag=True,default=False)
@click.option("--save-plot",default=None)
def scan(web_file,target,kappa,sigma,plot,save_plot):
    """Scan eta values and find the tipping threshold."""
    from .graph import BeliefWeb
    from .indicators import scan_tipping_point
    web = BeliefWeb.from_json(web_file)
    sc  = scan_tipping_point(web,target,kappa,sigma)
    click.echo(f"rho(W)         = {web.spectral_radius():.4f}")
    click.echo(f"eta_critical   ~ {sc.eta_critical:.4f}")
    click.echo(f"G at critical  ~ {sc.G_at_critical:.2f}")
    if plot or save_plot:
        from .viz import plot_tipping
        import matplotlib.pyplot as plt
        fig = plot_tipping(sc,save_path=save_plot)
        if plot: plt.show()

@main.command()
@click.argument("web_file")
@click.option("--layout",default="spring",
              type=click.Choice(["spring","kamada_kawai","circular","shell"]),show_default=True)
@click.option("--save",default=None)
def viz(web_file,layout,save):
    """Render the belief web as a graph."""
    from .graph import BeliefWeb
    from .viz import plot_web
    import matplotlib.pyplot as plt
    web = BeliefWeb.from_json(web_file)
    fig = plot_web(web,layout=layout,save_path=save)
    if save: click.echo(f"Saved to {save}")
    else:    plt.show()
