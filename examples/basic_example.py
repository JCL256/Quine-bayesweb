"""
basic_example.py — Démonstration end-to-end de bayesweb.

Scénario : toile géoprospective pour un territoire péri-urbain.
Choc     : forte évidence en faveur de "L'étalement urbain s'accélère" (κ=4, σ=0.7).

Ce fichier peut être utilisé de deux façons :
1. comme script Python classique ;
2. comme source de modèle pour l'application Shiny, via la fonction build_web().
"""
import copy
import bayesweb as bw
from bayesweb import BeliefWeb, Zone


def build_web() -> BeliefWeb:
    """Construit et retourne la toile de croyance utilisée par l'exemple."""
    web = BeliefWeb("Territoire péri-urbain")
    web.add_node("sprawl", "L'étalement urbain s'accélère", alpha=3.0, beta=1.0, zone=Zone.NUCLEUS)
    web.add_node("agri", "La surface agricole se réduit", alpha=2.5, beta=1.5, zone=Zone.NUCLEUS)
    web.add_node("biodiv", "L'indice de biodiversité décline", alpha=2.0, beta=2.0, zone=Zone.INTERMEDIATE)
    web.add_node("transit", "L'investissement TC augmente", alpha=1.0, beta=3.0, zone=Zone.INTERMEDIATE)
    web.add_node("ecosystem", "Les services écosystémiques sont reconnus", alpha=1.5, beta=2.5, zone=Zone.PERIPHERY)
    web.add_node("policy", "La politique foncière se durcit", alpha=1.0, beta=4.0, zone=Zone.PERIPHERY)


    web.add_undirected_edge("sprawl", "agri", weight=0.30)
    web.add_undirected_edge("agri", "biodiv", weight=0.70)
    web.add_undirected_edge("sprawl", "transit", weight=-0.55)
    web.add_undirected_edge("transit", "policy", weight=0.65)
    web.add_edge("biodiv", "ecosystem", weight=0.50)
    web.add_edge("policy", "sprawl", weight=-0.40)

    return web


def main() -> None:
    web = build_web()
    print(web.summary())

    # 2. Calibrer η
    eta = bw.calibrate_eta(web, lam=0.88)
    print(f"\nρ(W) = {web.spectral_radius():.4f}   η = {eta:.4f}   η·ρ = {eta*web.spectral_radius():.4f} < 1 ✓")

    # 3. Appliquer le choc
    kappa, sigma = 4.0, 0.70
    web_s = copy.deepcopy(web)
    web_s, dl = bw.apply_shock(web_s, "sprawl", kappa, sigma, inplace=True)
    info = bw.shock_summary(web, "sprawl", kappa, sigma)
    print(f"\nChoc sur 'sprawl'  κ={kappa}  σ={sigma}")
    print(f"  p_avant={info['p_before']:.4f}  →  p_après={info['p_after']:.4f}   Δℓ={info['delta_logit']:+.4f}")

    # 4. Propager
    result = bw.propagate(web_s, "sprawl", dl, eta=eta)
    print(f"\nPropagation : {'convergée' if result.converged else 'NON convergée'} en {result.iterations} itérations")

    # 5. Indicateurs
    s = bw.indicators_summary(web, result)
    print(f"\nF = {s['F']:.4f}  ({s['F_interpretation']})")
    print(f"G = {s['G']:.4f}  ({s['G_interpretation']})")

    # 6. Résultats par nœud
    df = result.to_dataframe()
    df["zone"] = [web.get_node(n).zone.value for n in df["node_id"]]
    df["label"] = [web.get_node(n).label for n in df["node_id"]]
    print("\n" + df.sort_values("delta_p", ascending=False)[["node_id", "zone", "p_initial", "p_final", "delta_p"]].to_string(index=False))

    # 7. Scan seuil de bascule
    scan = bw.scan_tipping_point(web, "sprawl", kappa=8, sigma=0.9)
    print(f"\nSeuil de bascule : η_critique ≈ {scan.eta_critical:.4f}  (G dépasse 10)")


# Compatibilité avec l'application Shiny qui peut lire la variable globale `web`
web = build_web()

if __name__ == "__main__":
    main()
