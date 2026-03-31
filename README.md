# bayesweb

**bayesweb** est un package Python pour construire des **toiles de croyances**, appliquer des **chocs** sur un nœud cible, propager leurs effets dans le réseau, et explorer des **indicateurs de bascule**.

Dans ce dépôt, le **modèle de référence** est défini dans :

```text
examples/basic\\\\\\\_example.py
```

Ce même modèle peut ensuite être :

* exécuté comme script Python,
* chargé dynamiquement dans l’application **Shiny**,
* modifié pour tester d’autres réseaux, nœuds et liens.

\---

## Fondements théoriques

Le cadre théorique du dépôt est présenté dans le document suivant :

* [**Toile de croyances bayésienne, seuils de bascule et géoprospective**](docs/theory/Theorie.pdf)

Ce texte constitue une **note théorique / version auteur d’un préprint en préparation**.
Il fournit l’arrière-plan conceptuel du modèle implémenté dans `bayesweb` ainsi que du prototype d’exploration développé dans `app/app.py`.

La version diffusée dans ce dépôt peut être révisée ultérieurement avant dépôt ou diffusion du préprint.

\---

## Démarrage rapide

### 1\. Installer le package

```bash
pip install -e .\\\\\\\[app]
```

Si l’extra `app` n’est pas disponible dans votre version locale :

```bash
pip install -e .
pip install shiny matplotlib networkx pandas
```

### 2\. Lancer le script d’exemple

```bash
python examples/basic\\\\\\\_example.py
```

### 3\. Lancer l’application Shiny

```bash
shiny run --reload --launch-browser app/app.py
```

### 4\. Modifier le modèle

Éditez :

```text
examples/basic\\\\\\\_example.py
```

Puis, dans l’application Shiny, cliquez sur **Recharger le fichier**.

\---

## Ce que contient ce dépôt

Ce dépôt contient trois éléments principaux.

### 1\. Le package Python `bayesweb`

Il fournit le moteur de calcul :

* définition des nœuds et des zones,
* définition des liens,
* calibration de paramètres,
* propagation des chocs,
* calcul d’indicateurs,
* analyse de seuils de bascule.

### 2\. Un modèle d’exemple

Le fichier :

```text
examples/basic\\\\\\\_example.py
```

sert de **point d’entrée principal** pour définir un réseau.

C’est ici que l’on décrit :

* les nœuds,
* leur zone (`NUCLEUS`, `INTERMEDIATE`, `PERIPHERY`),
* leurs paramètres (`alpha`, `beta`),
* les liens et leurs poids.

### 3\. Une application Shiny

Le fichier :

```text
app/app.py
```

charge automatiquement le modèle défini dans `examples/basic\\\\\\\_example.py` et construit :

* la liste des nœuds,
* le choix du nœud cible,
* le graphe concentrique,
* les indicateurs,
* le tableau des résultats.

\---

## Logique de travail recommandée

Le dépôt est conçu pour suivre ce flux :

1. **Définir ou modifier le réseau** dans `examples/basic\\\\\\\_example.py`
2. **Tester le modèle** avec :

```bash
   python examples/basic\\\\\\\_example.py
   ```

3. **Ouvrir l’interface Shiny** avec :

```bash
   shiny run --reload --launch-browser app/app.py
   ```

4. **Recharger le fichier** dans l’application si nécessaire
5. **Explorer les résultats** dans l’interface

Autrement dit :

* `basic\\\\\\\_example.py` = définition du modèle
* `bayesweb` = moteur de calcul
* `app.py` = interface d’exploration

\---

## Définir le réseau dans `basic\\\\\\\_example.py`

Le fichier `examples/basic\\\\\\\_example.py` est utilisé comme **source du modèle**.

Il est recommandé d’y définir une fonction :

```python
def build\\\\\\\_web():
    ...
    return web
```

puis d’utiliser cette même fonction dans le script principal.

Cette structure est recommandée car elle garantit que :

* le script Python,
* et l’application Shiny

utilisent exactement le **même modèle**.

\---

## Lancer le script Python

Le script d’exemple permet de tester directement le modèle sans passer par l’interface.

```bash
python examples/basic\\\\\\\_example.py
```

Ce script peut afficher :

* un résumé du réseau,
* la calibration de `eta`,
* les informations sur le choc,
* les indicateurs (`F`, `G`),
* le tableau final par nœud,
* un seuil de bascule.

\---

## Lancer l’application Shiny

L’application fournit une interface interactive pour explorer le modèle :

```bash
shiny run --reload --launch-browser app/app.py
```

### L’application permet de :

* charger automatiquement `examples/basic\\\\\\\_example.py`,
* recharger le modèle après modification,
* choisir le nœud cible,
* régler `kappa`, `sigma` et `lambda`,
* visualiser un graphe concentrique de type **toile de croyance de Quine**,
* lire les indicateurs,
* consulter le tableau des résultats par nœud.

\---

## Interprétation des principaux paramètres

### `kappa`

`kappa` règle la **force du choc** appliqué au nœud cible.
Plus `kappa` est élevé, plus le nœud cible est poussé fortement.

### `sigma`

`sigma` règle l’**orientation probabiliste du choc**.
Selon sa valeur, le choc peut augmenter ou diminuer la probabilité du nœud cible.

### `lambda`

`lambda` sert à calibrer `eta`, c’est-à-dire l’intensité globale de propagation dans le réseau.
Plus `lambda` est élevé, plus la propagation potentielle peut être forte.

### `p\\\\\\\_avant`

Probabilité du nœud cible **avant le choc**.

### `p\\\\\\\_après`

Probabilité du nœud cible **juste après le choc**, avant propagation dans l’ensemble du réseau.

### `delta\\\\\\\_logit`

Mesure interne de l’intensité du changement appliqué au nœud cible.

\---

## Interprétation des indicateurs

### `F`

`F` résume l’intensité interne de la réponse du système.
Plus `F` est élevé, plus la propagation interne est marquée.

### `G`

`G` indique si le réseau **amplifie** ou **atténue** le choc initial.

* `G > 1` : amplification
* `G < 1` : atténuation

### `eta\\\\\\\_critique`

Seuil théorique de bascule.
Plus il est bas, plus le système peut entrer facilement dans une dynamique forte.

\---

## Représentation graphique

L’application Shiny dessine le réseau sous forme de **toile concentrique** inspirée de la théorie de la toile de croyance de Quine :

* **Nucléus** au centre
* **Intermédiaire** sur un anneau médian
* **Périphérie** sur l’anneau externe

Le graphe met en évidence :

* la zone de chaque nœud,
* le nœud cible,
* les liens orientés et non orientés,
* les influences négatives en pointillés,
* la taille relative des nœuds selon l’effet observé (`delta\\\\\\\_p`).

\---

## Structure du dépôt

```text
bayesweb/
├── app/
│   └── app.py
├── bayesweb/
│   ├── \\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_.py
│   ├── cli.py
│   └── ...
├── docs/
│   └── theory/
│       └── Théorie.pdf
├── examples/
│   └── basic\\\\\\\_example.py
├── tests/
├── README.md
└── pyproject.toml
```

\---

## Installation en mode développement

Depuis la racine du projet :

```bash
pip install -e .
```

Ou avec les dépendances de l’application :

```bash
pip install -e .\\\\\\\[app]
```

\---

## Utilisation en ligne de commande

Le package propose aussi une interface CLI.

Exemples :

```bash
bayesweb init test.json
bayesweb run test.json --target A --kappa 5 --sigma 0.9
bayesweb scan test.json --target A
bayesweb viz test.json
```

Selon la configuration locale, on peut aussi passer par :

```bash
python -m bayesweb.cli
```

\## Modèle jouet HTML5



Un démonstrateur pédagogique HTML5 est disponible dans :



\- \[Modèle jouet HTML5](docs/toy-model/index.html)



Le modèle HTML5 constitue un démonstrateur pédagogique inspiré du cadre théorique, et non une implémentation intégrale du schéma formel de propagation présenté dans la note.

\---

## Cas d’usage typique

Ce dépôt est particulièrement adapté pour :

* explorer des réseaux de croyances territoriaux,
* tester des scénarios géoprospectifs,
* analyser la propagation d’un choc dans un système de croyances,
* comparer des structures de réseau selon leur robustesse ou leur sensibilité,
* visualiser la structure nucléus / intermédiaire / périphérie d’une toile de croyance.

\---

## Bonnes pratiques

* utiliser `examples/basic\\\\\\\_example.py` comme **fichier de définition du modèle**
* éviter de dupliquer la définition du réseau ailleurs
* privilégier une fonction `build\\\\\\\_web()`
* tester d’abord en script Python
* puis explorer visuellement dans Shiny

\---

## Licence

Le code source de ce dépôt est distribué sous licence \*\*PolyForm Noncommercial 1.0.0\*\*.



Le document théorique disponible dans `docs/theory/Théorie.pdf` est diffusé sous licence \*\*CC BY-NC 4.0\*\*, sauf indication contraire.



Cela signifie que :

\- l’usage, l’étude et la modification sont autorisés,

\- mais l’usage commercial n’est pas autorisé sans accord explicite de l’auteur.

