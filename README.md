---

# Benchmark VRP Agricole

## Problème de Tournées de Véhicules pour Essaims de Robots en Agriculture : Comparaison K-means + GA vs K-means + LNS

---

## Description

Ce projet implémente un **benchmark expérimental complet** comparant deux approches hybrides pour résoudre un **Vehicle Routing Problem (VRP) multi-robots** appliqué à l’agriculture de précision.

L’architecture suit strictement le protocole expérimental décrit dans l'article de (Sinai, 2020)¹.

Les deux approches comparées sont :

* **K-means++ + Algorithme Génétique (GA)**
* **K-means++ + Large Neighborhood Search (LNS)**

---

## Objectif

Optimiser la planification de trajectoires d’une flotte de robots agricoles afin de :

* Minimiser la **distance totale parcourue**
* Minimiser le **makespan** (distance maximale parcourue par un robot)
* Analyser le **temps de calcul**
* Évaluer la **robustesse statistique** (moyenne ± écart-type, coefficient de variation)

---

## Méthodologie

Le solveur VRP fonctionne en **deux phases** :

### 1️⃣ Clustering spatial

* Partitionnement des points via **K-means++**
* Nombre de clusters = nombre de robots
* Implémentation via `scikit-learn`

---

### 2️⃣ Résolution TSP intra-cluster

Chaque cluster est résolu indépendamment :

####  Algorithme Génétique (GA)

Paramètres (strictement conformes au document) :

* Population : 50
* Générations : 100
* Croisement : 0.8 (Order Crossover – OX)
* Mutation : 0.2 (swap)
* Sélection : tournoi (k=3)

---

####  Large Neighborhood Search (LNS)

Paramètres :

* 100 itérations
* Taux de destruction : 30 %
* Température initiale : 100
* Solution initiale : plus proche voisin
* Réparation : insertion gloutonne
* Amélioration locale : 2-opt
* Critère d’acceptation : recuit simulé

---

## Géométries de Champs

Trois types de champs agricoles sont simulés :

### Champ rectangulaire

* Dimensions : 46 × 28

### Champ en L

* Rectangle avec coin supérieur droit retiré

### Champ en H

* Structure composée de trois barres connectées

Les points sont générés par **rejection sampling** pour assurer une distribution uniforme valide.

---

## ⚙️ Configuration Expérimentale

* Nombre de points : `[30, 50, 100]`
* Nombre de robots : `[3, 4, 5]`
* 10 runs par configuration
* Graine aléatoire contrôlée (reproductibilité)

Total :
3 géométries × 3 tailles × 3 nombres de robots × 10 runs

---

## 📊 Métriques Évaluées

Pour chaque configuration :

* Distance totale moyenne ± écart-type
* Makespan moyen ± écart-type
* Temps de calcul moyen
* Coefficient de variation (CV)
* Gain (%) de LNS par rapport à GA

---

## 📂 Structure du Code

Le fichier principal contient :

* `Point` → structure géométrique
* `Solution` → stockage des métriques
* `FieldGenerator` → génération des champs
* `KMeansClustering` → partitionnement spatial
* `GeneticAlgorithm` → résolution TSP par GA
* `LargeNeighborhoodSearch` → résolution TSP par LNS
* `VRPSolver` → orchestration clustering + TSP
* `Benchmark` → exécution complète + agrégation + visualisation

Les résultats sont sauvegardés dans :

```
resultats_benchmark/
├── 1_distance_vs_points.png
├── 2_temps_calcul.png
├── 3_makespan_vs_robots.png
├── 4_gain_lns.png
├── rapport_resultats.txt
```

---

##  Installation

### Dépendances

```bash
pip install numpy matplotlib scikit-learn
```

---

##  Exécution

```bash
python benchmark_vrp.py
```

Le programme :

1. Exécute toutes les configurations
2. Affiche un tableau récapitulatif
3. Génère un rapport texte
4. Sauvegarde tous les graphiques

---

## Graphiques Générés

* Distance totale vs nombre de points
* Temps de calcul vs nombre de points
* Makespan vs nombre de robots
* Gains de LNS
* Comparaison des géométries

---

## Complexité

* **VRP** : NP-difficile
* **GA** : $O(N_{pop} \cdot N_{gen} \cdot n_k)$
* **LNS** : $O(L \cdot n_k^2)$

avec $n_k \approx n/m$


##  Contributions

✔ Implémentation strictement conforme aux paramètres expérimentaux
✔ Comparaison statistique rigoureuse
✔ Analyse multi-géométrie
✔ Visualisations automatiques
✔ Reproductibilité complète

---

## Auteur

Projet académique en Algorithmics, Complexity, and Graph Algorithms I,II.
Année : 2026

## Références
* Sinai, L. (2020). *Efficient path planning for multiple agents in agriculture fields*. Master’s thesis, University of Twente, Netherlands. [Lien vers la thèse](https://purl.utwente.nl)


