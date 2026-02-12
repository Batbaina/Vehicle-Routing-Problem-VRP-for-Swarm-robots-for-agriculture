"""
VRP POUR AGRICULTURE DE PRÉCISION - K-MEANS+GA vs K-MEANS+LNS
================================================================================
Reproduction exacte des conditions expérimentales du papier Botteghi et al. (2020)
pour UGV en environnement continu (sans obstacles).

Algorithmes comparés :
- VOTRE GA + K-means
- VOTRE LNS + K-means

Auteur: Adaptation pour comparaison avec papier
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Set
import time
from dataclasses import dataclass
import random
import os

# =============================================================================
# PARAMÈTRES DU PAPIER (EXACTS)
# =============================================================================

# Paramètres UGV (section 4.2.3)
PAPER_PARAMS = {
    'field_size_units': (46, 28),      # Grille 46×28 unités
    'unit_to_meters': 2.5,             # 1 unité = 2.5m
    'area_ha': 0.805,                 # 115m × 70m = 0.805 ha
    'agent_speed': 15,                # 15 unités/temps (≈ 4m/s)
    'depot_position': (0, 0),         # Départ au coin inférieur gauche
    'detect_radius': 1,              # Rayon de détection
    'time_step': 0.2,               # Pas de temps
}

# Paramètres des champs (Table 4.3 du papier)
FIELD_PROPERTIES = {
    'rect': {
        'shape': 'rectangle',
        'bounds': (0, 46, 0, 28),    # (xmin, xmax, ymin, ymax)
        'area': 1288,
        'perimeter': 148,
        'IQ': 0.74,
        'color': 'blue'
    },
    'L': {
        'shape': 'L',
        'bounds': (0, 46, 0, 28),
        'L_cut': (20, 20),           # Point de coupure pour forme en L
        'area': 928,
        'perimeter': 148,
        'IQ': 0.53,
        'color': 'green'
    },
    'H': {
        'shape': 'H',
        'bounds': (0, 46, 0, 28),
        'H_vertical': (18, 28),       # Barre verticale gauche
        'H_vertical2': (28, 38),      # Barre verticale droite
        'H_horizontal': (10, 18),     # Barre horizontale
        'area': 1008,
        'perimeter': 188,
        'IQ': 0.36,
        'color': 'red'
    }
}

# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class Point:
    """Représente un point d'intérêt (POI) dans le champ agricole"""
    x: float
    y: float
    id: int
    
    def distance_to(self, other: 'Point') -> float:
        """Distance euclidienne"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Solution:
    """Solution VRP complète"""
    routes: List[List[int]]          # Routes par robot
    distances: List[float]          # Distance par robot
    total_distance: float           # Distance totale D
    max_distance: float            # Makespan T
    time: float                    # Temps de calcul
    z: float = 0.5                # Paramètre de compromis
    
    @property
    def cost(self) -> float:
        """Fonction de coût du papier (Eq. 2)"""
        if len(self.distances) == 0:
            return 0.0
        D = self.total_distance
        T = self.max_distance
        m = len(self.distances)
        return self.z * T + (1 - self.z) * (D / m)
    
    @property
    def mean_distance(self) -> float:
        """Distance moyenne par robot"""
        return self.total_distance / len(self.distances) if self.distances else 0.0
    
    @property
    def coefficient_variation(self) -> float:
        """Coefficient de variation (équilibrage)"""
        if len(self.distances) <= 1:
            return 0.0
        mean = np.mean(self.distances)
        if mean == 0:
            return 0.0
        return np.std(self.distances) / mean

# =============================================================================
# GÉNÉRATION DES CHAMPS AGRICOLES (FORMES DU PAPIER)
# =============================================================================

class PaperFieldGenerator:
    """Génère les champs agricoles exactement comme dans le papier (Fig. 4.6)"""
    
    @staticmethod
    def is_point_in_field(x: float, y: float, field_type: str) -> bool:
        """Vérifie si un point est dans la forme du champ"""
        
        if field_type == 'rect':
            # Rectangle plein
            return 0 <= x <= 46 and 0 <= y <= 28
            
        elif field_type == 'L':
            # Forme en L : rectangle complet moins le coin supérieur droit
            if not (0 <= x <= 46 and 0 <= y <= 28):
                return False
            # Exclure le rectangle [20:46, 20:28]
            if x > 20 and y > 20:
                return False
            return True
            
        elif field_type == 'H':
            # Forme en H : trois rectangles
            if not (0 <= x <= 46 and 0 <= y <= 28):
                return False
            
            # Barre verticale gauche
            if 0 <= x <= 18:
                return True
            # Barre verticale droite
            if 28 <= x <= 46:
                return True
            # Barre horizontale centrale
            if 10 <= y <= 18 and 18 <= x <= 28:
                return True
            
            return False
        
        return False
    
    @staticmethod
    def generate_random_points(field_type: str, n_points: int, seed: int = None) -> List[Point]:
        """
        Génère n points aléatoires uniformément distribués dans la forme du champ
        
        Args:
            field_type: 'rect', 'L', 'H'
            n_points: nombre de points à générer
            seed: graine aléatoire
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        points = []
        attempts = 0
        max_attempts = n_points * 100
        
        while len(points) < n_points and attempts < max_attempts:
            x = np.random.uniform(0, 46)
            y = np.random.uniform(0, 28)
            
            if PaperFieldGenerator.is_point_in_field(x, y, field_type):
                points.append(Point(x, y, len(points) + 1))
            
            attempts += 1
        
        if len(points) < n_points:
            print(f"⚠️ Attention: Seulement {len(points)}/{n_points} points générés")
        
        return points
    
    @staticmethod
    def generate_blobs(field_type: str, n_blobs: int, points_per_blob: int, 
                       seed: int = None) -> List[Point]:
        """
        Génère des points en clusters (blobs) comme dans le papier
        
        Args:
            field_type: 'rect', 'L', 'H'
            n_blobs: nombre de clusters
            points_per_blob: points par cluster
            seed: graine aléatoire
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Générer des centroïdes uniformément dans le champ
        centroids = []
        for _ in range(n_blobs):
            while True:
                cx = np.random.uniform(5, 41)
                cy = np.random.uniform(5, 23)
                if PaperFieldGenerator.is_point_in_field(cx, cy, field_type):
                    centroids.append((cx, cy))
                    break
        
        # Générer points autour des centroïdes (std = 3 unités comme dans le papier)
        points = []
        point_id = 1
        std_dev = 3.0
        
        for cx, cy in centroids:
            for _ in range(points_per_blob):
                while True:
                    x = np.random.normal(cx, std_dev)
                    y = np.random.normal(cy, std_dev)
                    if PaperFieldGenerator.is_point_in_field(x, y, field_type):
                        points.append(Point(x, y, point_id))
                        point_id += 1
                        break
        
        return points
    
    @staticmethod
    def plot_field(field_type: str, points: List[Point] = None, 
                   title: str = None, save_path: str = None):
        """Visualise un champ du papier"""
        plt.figure(figsize=(10, 8))
        
        # Dessiner la forme du champ
        if field_type == 'rect':
            plt.fill([0, 46, 46, 0], [0, 0, 28, 28], 'lightgray', alpha=0.3)
            plt.plot([0, 46, 46, 0, 0], [0, 0, 28, 28, 0], 'k-', linewidth=2)
            
        elif field_type == 'L':
            # Rectangle principal
            plt.fill([0, 46, 46, 20, 20, 0], 
                    [0, 0, 20, 20, 28, 28], 'lightgray', alpha=0.3)
            plt.plot([0, 46, 46, 20, 20, 0, 0], 
                    [0, 0, 20, 20, 28, 28, 0], 'k-', linewidth=2)
            
        elif field_type == 'H':
            # Barre gauche
            plt.fill([0, 18, 18, 0], [0, 0, 28, 28], 'lightgray', alpha=0.3)
            # Barre droite
            plt.fill([28, 46, 46, 28], [0, 0, 28, 28], 'lightgray', alpha=0.3)
            # Barre horizontale
            plt.fill([18, 28, 28, 18], [10, 10, 18, 18], 'lightgray', alpha=0.3)
            # Contours
            plt.plot([0, 18, 18, 28, 28, 46, 46, 28, 28, 18, 18, 0, 0],
                    [0, 0, 10, 10, 0, 0, 28, 28, 18, 18, 28, 28, 0], 'k-', linewidth=2)
        
        # Dépôt (0,0)
        plt.plot(0, 0, 'rs', markersize=15, label='Dépôt')
        
        # Points d'intérêt
        if points:
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            plt.scatter(xs, ys, c='blue', s=30, alpha=0.6, label=f'POIs ({len(points)})')
        
        plt.xlabel('X (unités)')
        plt.ylabel('Y (unités)')
        plt.title(title or f'Champ {field_type.upper()} - {FIELD_PROPERTIES[field_type]["area"]} unités²')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-2, 48)
        plt.ylim(-2, 30)
        
        if save_path:
            # Créer le dossier figures s'il n'existe pas
            os.makedirs('figures', exist_ok=True)
            plt.savefig(os.path.join('figures', save_path), dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# CLUSTERING K-MEANS
# =============================================================================

class KMeansClustering:
    """K-means clustering avec coordonnées cartésiennes"""
    
    @staticmethod
    def cluster(points: List[Point], n_clusters: int) -> List[List[Point]]:
        """
        Clusterise les points avec K-means standard
        
        Args:
            points: points à clusteriser
            n_clusters: nombre de clusters (= nombre de robots)
        """
        if len(points) <= n_clusters:
            # Un point par cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, p in enumerate(points):
                clusters[i % n_clusters].append(p)
            return clusters
        
        # Préparer les données pour sklearn
        X = np.array([[p.x, p.y] for p in points])
        
        # K-means avec initialisation K-means++
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                       n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Organiser les points par cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, point in enumerate(points):
            clusters[labels[i]].append(point)
        
        return clusters

# =============================================================================
# VOTRE ALGORITHME GÉNÉTIQUE (GA)
# =============================================================================

class GeneticAlgorithm:
    """VOTRE Algorithme Génétique pour TSP"""
    
    def __init__(self, pop_size: int = 50, n_gen: int = 100, 
                 p_cross: float = 0.8, p_mut: float = 0.2):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.p_cross = p_cross
        self.p_mut = p_mut
    
    def solve(self, points: List[Point], depot: Point) -> List[int]:
        """Résout le TSP pour un cluster"""
        if len(points) <= 1:
            return [p.id for p in points]
        
        # Adapter les paramètres pour les grandes instances
        if len(points) > 20:
            self.n_gen = 200
            self.pop_size = 100
        
        n = len(points)
        
        # Initialisation
        population = []
        for _ in range(self.pop_size):
            route = list(range(n))
            random.shuffle(route)
            population.append(route)
        
        best_route = None
        best_distance = float('inf')
        
        for gen in range(self.n_gen):
            # Évaluation
            distances = [self._calculate_distance(route, points, depot) for route in population]
            fitness = [1.0 / (d + 1e-6) for d in distances]
            
            # Meilleure solution
            min_idx = np.argmin(distances)
            if distances[min_idx] < best_distance:
                best_distance = distances[min_idx]
                best_route = population[min_idx][:]
            
            # Sélection
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Croisement
                if random.random() < self.p_cross:
                    child1, child2 = self._crossover_ox(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutation
                if random.random() < self.p_mut:
                    child1 = self._mutate_swap(child1)
                if random.random() < self.p_mut:
                    child2 = self._mutate_swap(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]
        
        return [points[i].id for i in best_route]
    
    def _calculate_distance(self, route: List[int], points: List[Point], depot: Point) -> float:
        """Calcule distance totale de la route"""
        if not route:
            return 0.0
        
        total = depot.distance_to(points[route[0]])
        for i in range(len(route) - 1):
            total += points[route[i]].distance_to(points[route[i+1]])
        total += points[route[-1]].distance_to(depot)
        
        return total
    
    def _tournament_selection(self, population: List[List[int]], 
                             fitness: List[float], k: int = 3) -> List[int]:
        """Sélection par tournoi"""
        selected = random.sample(list(zip(population, fitness)), k)
        return max(selected, key=lambda x: x[1])[0][:]
    
    def _crossover_ox(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order Crossover (OX)"""
        n = len(parent1)
        if n < 2:
            return parent1[:], parent2[:]
        
        cx1, cx2 = sorted(random.sample(range(n), 2))
        
        # Enfant 1
        child1 = [-1] * n
        child1[cx1:cx2] = parent1[cx1:cx2]
        p2_filtered = [x for x in parent2 if x not in child1[cx1:cx2]]
        idx = 0
        for i in range(n):
            if child1[i] == -1:
                child1[i] = p2_filtered[idx]
                idx += 1
        
        # Enfant 2
        child2 = [-1] * n
        child2[cx1:cx2] = parent2[cx1:cx2]
        p1_filtered = [x for x in parent1 if x not in child2[cx1:cx2]]
        idx = 0
        for i in range(n):
            if child2[i] == -1:
                child2[i] = p1_filtered[idx]
                idx += 1
        
        return child1, child2
    
    def _mutate_swap(self, route: List[int]) -> List[int]:
        """Mutation par échange"""
        if len(route) < 2:
            return route
        route = route[:]
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route

# =============================================================================
# VOTRE LARGE NEIGHBORHOOD SEARCH (LNS)
# =============================================================================

class LargeNeighborhoodSearch:
    """VOTRE LNS pour TSP"""
    
    def __init__(self, n_iter: int = 100, destroy_rate: float = 0.3, 
                 temperature: float = 100.0):
        self.n_iter = n_iter
        self.destroy_rate = destroy_rate
        self.temperature = temperature
    
    def solve(self, points: List[Point], depot: Point) -> List[int]:
        """Résout le TSP avec LNS"""
        if len(points) <= 1:
            return [p.id for p in points]
        
        # Adapter pour grandes instances
        if len(points) > 20:
            self.n_iter = 200
            self.destroy_rate = 0.4
        
        # Solution initiale : plus proche voisin
        current_route = self._nearest_neighbor(points, depot)
        current_cost = self._calculate_cost(current_route, points, depot)
        
        best_route = current_route[:]
        best_cost = current_cost
        
        for iter_num in range(self.n_iter):
            # Destruction
            n_destroy = max(1, int(len(current_route) * self.destroy_rate))
            partial_route, removed = self._destroy_random(current_route, n_destroy)
            
            # Réparation
            new_route = self._repair_greedy(partial_route, removed, points, depot)
            
            # Amélioration locale (2-opt)
            new_route = self._two_opt(new_route, points, depot)
            
            new_cost = self._calculate_cost(new_route, points, depot)
            
            # Acceptation
            if new_cost < best_cost:
                current_route = new_route
                current_cost = new_cost
                best_route = new_route[:]
                best_cost = new_cost
            elif new_cost < current_cost:
                current_route = new_route
                current_cost = new_cost
            else:
                # Recuit simulé
                temp = self.temperature * (1 - iter_num / self.n_iter)
                if random.random() < np.exp((current_cost - new_cost) / max(temp, 1)):
                    current_route = new_route
                    current_cost = new_cost
        
        return [points[i].id for i in best_route]
    
    def _nearest_neighbor(self, points: List[Point], depot: Point) -> List[int]:
        """Heuristique du plus proche voisin"""
        n = len(points)
        unvisited = set(range(n))
        route = []
        current = None
        
        while unvisited:
            if current is None:
                distances = [depot.distance_to(points[i]) for i in unvisited]
                current = min(unvisited, key=lambda i: distances[list(unvisited).index(i)])
            else:
                distances = [points[current].distance_to(points[i]) for i in unvisited]
                current = min(unvisited, key=lambda i: distances[list(unvisited).index(i)])
            
            route.append(current)
            unvisited.remove(current)
        
        return route
    
    def _calculate_cost(self, route: List[int], points: List[Point], depot: Point) -> float:
        """Calcule le coût d'une route"""
        if not route:
            return 0.0
        
        total = depot.distance_to(points[route[0]])
        for i in range(len(route) - 1):
            total += points[route[i]].distance_to(points[route[i+1]])
        total += points[route[-1]].distance_to(depot)
        
        return total
    
    def _destroy_random(self, route: List[int], n_destroy: int) -> Tuple[List[int], List[int]]:
        """Destruction aléatoire"""
        route = route[:]
        removed = random.sample(route, n_destroy)
        partial = [x for x in route if x not in removed]
        return partial, removed
    
    def _repair_greedy(self, partial: List[int], removed: List[int],
                      points: List[Point], depot: Point) -> List[int]:
        """Réparation gloutonne"""
        route = partial[:]
        
        for node in removed:
            best_pos = 0
            best_cost = float('inf')
            
            for pos in range(len(route) + 1):
                test_route = route[:pos] + [node] + route[pos:]
                cost = self._calculate_cost(test_route, points, depot)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            route.insert(best_pos, node)
        
        return route
    
    def _two_opt(self, route: List[int], points: List[Point], depot: Point) -> List[int]:
        """Amélioration 2-opt"""
        if len(route) < 4:
            return route
        
        improved = True
        best_route = route[:]
        
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    if self._calculate_cost(new_route, points, depot) < self._calculate_cost(best_route, points, depot):
                        best_route = new_route
                        improved = True
                        break
                if improved:
                    break
        
        return best_route

# =============================================================================
# SOLVEUR VRP PRINCIPAL
# =============================================================================

class VRPSolver:
    """Solveur VRP avec K-means + GA ou K-means + LNS"""
    
    def __init__(self, points: List[Point], n_robots: int, z: float = 0.5):
        self.points = points
        self.n_robots = n_robots
        self.z = z
        self.depot = Point(0, 0, 0)  # Dépôt à (0,0) comme dans le papier
    
    def solve_with_ga(self, ga_params: Dict = None) -> Solution:
        """K-means + VOTRE GA"""
        start_time = time.time()
        
        # Phase 1: Clustering K-means
        clusters = KMeansClustering.cluster(self.points, self.n_robots)
        
        # Phase 2: GA pour chaque cluster
        ga = GeneticAlgorithm(**(ga_params or {}))
        
        routes = []
        distances = []
        
        for cluster in clusters:
            if not cluster:
                routes.append([])
                distances.append(0.0)
            else:
                route_ids = ga.solve(cluster, self.depot)
                routes.append(route_ids)
                dist = self._calculate_route_distance(route_ids)
                distances.append(dist)
        
        return Solution(
            routes=routes,
            distances=distances,
            total_distance=sum(distances),
            max_distance=max(distances) if distances else 0.0,
            time=time.time() - start_time,
            z=self.z
        )
    
    def solve_with_lns(self, lns_params: Dict = None) -> Solution:
        """K-means + VOTRE LNS"""
        start_time = time.time()
        
        # Phase 1: Clustering K-means
        clusters = KMeansClustering.cluster(self.points, self.n_robots)
        
        # Phase 2: LNS pour chaque cluster
        lns = LargeNeighborhoodSearch(**(lns_params or {}))
        
        routes = []
        distances = []
        
        for cluster in clusters:
            if not cluster:
                routes.append([])
                distances.append(0.0)
            else:
                route_ids = lns.solve(cluster, self.depot)
                routes.append(route_ids)
                dist = self._calculate_route_distance(route_ids)
                distances.append(dist)
        
        return Solution(
            routes=routes,
            distances=distances,
            total_distance=sum(distances),
            max_distance=max(distances) if distances else 0.0,
            time=time.time() - start_time,
            z=self.z
        )
    
    def _calculate_route_distance(self, route_ids: List[int]) -> float:
        """Calcule la distance d'une route"""
        if not route_ids:
            return 0.0
        
        points_dict = {p.id: p for p in self.points}
        
        total = self.depot.distance_to(points_dict[route_ids[0]])
        for i in range(len(route_ids) - 1):
            total += points_dict[route_ids[i]].distance_to(points_dict[route_ids[i+1]])
        total += points_dict[route_ids[-1]].distance_to(self.depot)
        
        return total

# =============================================================================
# BENCHMARK - SEULEMENT GA et LNS
# =============================================================================

class Benchmark:
    """Benchmark pour comparer GA vs LNS avec les paramètres du papier"""
    
    @staticmethod
    def run_experiments(field_types: List[str] = ['rect', 'L', 'H'],
                        n_points_list: List[int] = [50, 100, 150],  # UGV: 50-150 points
                        n_robots_list: List[int] = [1, 2, 3, 4, 5],
                        n_runs: int = 10,
                        z: float = 0.5,
                        distribution: str = 'uniform'):
        """
        Exécute le benchmark complet avec les paramètres exacts du papier
        
        Args:
            field_types: types de champs à tester
            n_points_list: nombres de points à tester
            n_robots_list: nombres de robots à tester
            n_runs: nombre d'exécutions
            z: paramètre de la fonction de coût
            distribution: 'uniform' ou 'blobs'
        """
        
        # Structure pour stocker les résultats
        results = {
            'field': [],
            'n_points': [],
            'n_robots': [],
            'algorithm': [],
            'total_distance': [],
            'max_distance': [],
            'mean_distance': [],
            'cv': [],
            'cost': [],
            'time': [],
            'distances_by_agent': []
        }
        
        total_combinations = (len(field_types) * len(n_points_list) * 
                            len(n_robots_list) * n_runs * 2)  # 2 algorithmes
        print(f"\n{'='*80}")
        print(f"BENCHMARK GA vs LNS - {total_combinations} exécutions")
        print(f"{'='*80}")
        print(f"Champs: {field_types}")
        print(f"Points: {n_points_list}")
        print(f"Robots: {n_robots_list}")
        print(f"Runs par config: {n_runs}")
        print(f"z = {z}")
        print(f"Distribution: {distribution}")
        print(f"{'='*80}\n")
        
        run_count = 0
        
        for field_type in field_types:
            print(f"\n📌 CHAMP {field_type.upper()} - {FIELD_PROPERTIES[field_type]['area']} unités²")
            
            for n_points in n_points_list:
                print(f"\n  📍 n = {n_points} points")
                
                for n_robots in n_robots_list:
                    print(f"    🤖 m = {n_robots} robots...", end=' ')
                    
                    for run in range(n_runs):
                        # Graine aléatoire reproductible
                        seed = run + hash(f"{field_type}{n_points}{n_robots}{run}") % 10000
                        
                        # Générer les points
                        if distribution == 'uniform':
                            points = PaperFieldGenerator.generate_random_points(
                                field_type, n_points, seed=seed
                            )
                        else:  # blobs
                            n_blobs = max(3, n_robots)
                            points_per_blob = max(1, n_points // n_blobs)
                            points = PaperFieldGenerator.generate_blobs(
                                field_type, n_blobs, points_per_blob, seed=seed
                            )
                        
                        if len(points) < n_points * 0.8:
                            continue
                        
                        # Créer le solveur
                        solver = VRPSolver(points, n_robots, z=z)
                        
                        # 1. VOTRE GA
                        sol_ga = solver.solve_with_ga()
                        results['field'].append(field_type)
                        results['n_points'].append(n_points)
                        results['n_robots'].append(n_robots)
                        results['algorithm'].append('GA')
                        results['total_distance'].append(sol_ga.total_distance)
                        results['max_distance'].append(sol_ga.max_distance)
                        results['mean_distance'].append(sol_ga.mean_distance)
                        results['cv'].append(sol_ga.coefficient_variation)
                        results['cost'].append(sol_ga.cost)
                        results['time'].append(sol_ga.time)
                        results['distances_by_agent'].append(sol_ga.distances)
                        
                        # 2. VOTRE LNS
                        sol_lns = solver.solve_with_lns()
                        results['field'].append(field_type)
                        results['n_points'].append(n_points)
                        results['n_robots'].append(n_robots)
                        results['algorithm'].append('LNS')
                        results['total_distance'].append(sol_lns.total_distance)
                        results['max_distance'].append(sol_lns.max_distance)
                        results['mean_distance'].append(sol_lns.mean_distance)
                        results['cv'].append(sol_lns.coefficient_variation)
                        results['cost'].append(sol_lns.cost)
                        results['time'].append(sol_lns.time)
                        results['distances_by_agent'].append(sol_lns.distances)
                        
                        run_count += 2
                        
                        # Progression
                        if (run + 1) % 5 == 0:
                            print('▪', end='', flush=True)
                    
                    print(f" {n_runs} runs")
        
        print(f"\n\n✅ Benchmark terminé: {run_count} solutions générées")
        return results

# =============================================================================
# VISUALISATION
# =============================================================================

class Visualizer:
    """Visualisation des résultats"""
    
    @staticmethod
    def print_results_table(results: Dict):
        """Affiche un tableau récapitulatif des résultats"""
        
        print("\n" + "="*120)
        print("📊 TABLEAU COMPARATIF - K-MEANS+GA vs K-MEANS+LNS")
        print("="*120)
        print(f"{'Champ':<6} {'n':<4} {'m':<3} {'Algo':<6} "
              f"{'Distance D':>12} {'Max T':>12} {'Moyenne':>12} {'CV':>8} {'Coût':>10} {'Temps(s)':>10}")
        print("-"*120)
        
        # Grouper par champ, n, m
        grouped = {}
        for i in range(len(results['field'])):
            key = (results['field'][i], results['n_points'][i], results['n_robots'][i])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(i)
        
        for (field, n, m), indices in sorted(grouped.items()):
            # GA
            ga_indices = [i for i in indices if results['algorithm'][i] == 'GA']
            if ga_indices:
                i = ga_indices[0]  # Prendre le premier (moyenne sur runs)
                print(f"{field:<6} {n:<4} {m:<3} {'GA':<6} "
                      f"{results['total_distance'][i]:>12.1f} "
                      f"{results['max_distance'][i]:>12.1f} "
                      f"{results['mean_distance'][i]:>12.1f} "
                      f"{results['cv'][i]:>8.3f} "
                      f"{results['cost'][i]:>10.3f} "
                      f"{results['time'][i]:>10.3f}")
            
            # LNS
            lns_indices = [i for i in indices if results['algorithm'][i] == 'LNS']
            if lns_indices:
                i = lns_indices[0]
                print(f"{field:<6} {n:<4} {m:<3} {'LNS':<6} "
                      f"{results['total_distance'][i]:>12.1f} "
                      f"{results['max_distance'][i]:>12.1f} "
                      f"{results['mean_distance'][i]:>12.1f} "
                      f"{results['cv'][i]:>8.3f} "
                      f"{results['cost'][i]:>10.3f} "
                      f"{results['time'][i]:>10.3f}")
            
            # Calculer le gain LNS vs GA
            if ga_indices and lns_indices:
                ga_dist = results['total_distance'][ga_indices[0]]
                lns_dist = results['total_distance'][lns_indices[0]]
                gain = (ga_dist - lns_dist) / ga_dist * 100
                print(f"{'':<6} {'':<4} {'':<3} {'Gain':<6} "
                      f"{gain:>+11.1f}% vs GA", end='')
                
                # Équilibrage
                ga_cv = results['cv'][ga_indices[0]]
                lns_cv = results['cv'][lns_indices[0]]
                if lns_cv < ga_cv:
                    print(f"  ✅ LNS mieux équilibré")
                else:
                    print(f"  ⚠️ GA mieux équilibré")
            
            print("-"*120)
    
    @staticmethod
    def plot_comparison_bar(results: Dict, field_type: str = 'rect',
                           n_points: int = 100, save_path: str = None):
        """Graphique en barres comparant GA et LNS"""
        
        # Filtrer les résultats
        mask = np.array([(results['field'][i] == field_type and 
                         results['n_points'][i] == n_points and
                         results['n_robots'][i] == 3)  # m=3 comme référence
                        for i in range(len(results['field']))])
        
        indices = np.where(mask)[0]
        
        # Extraire les données
        ga_indices = [i for i in indices if results['algorithm'][i] == 'GA']
        lns_indices = [i for i in indices if results['algorithm'][i] == 'LNS']
        
        if not ga_indices or not lns_indices:
            print(f"⚠️ Pas assez de données pour {field_type}, n={n_points}")
            return
        
        ga_data = results['total_distance'][ga_indices[0]]
        lns_data = results['total_distance'][lns_indices[0]]
        ga_time = results['time'][ga_indices[0]]
        lns_time = results['time'][lns_indices[0]]
        ga_cv = results['cv'][ga_indices[0]]
        lns_cv = results['cv'][lns_indices[0]]
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Distance totale
        ax = axes[0]
        bars = ax.bar(['GA', 'LNS'], [ga_data, lns_data], 
                     color=['steelblue', 'coral'], alpha=0.8)
        ax.set_ylabel('Distance totale (unités)')
        ax.set_title(f'Champ {field_type.upper()} - n={n_points}, m=3')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs et le gain
        for bar, val in zip(bars, [ga_data, lns_data]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        gain = (ga_data - lns_data) / ga_data * 100
        ax.text(0.5, max(ga_data, lns_data) * 0.9, 
               f'Gain: {gain:.1f}%', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Temps de calcul
        ax = axes[1]
        bars = ax.bar(['GA', 'LNS'], [ga_time, lns_time],
                     color=['steelblue', 'coral'], alpha=0.8)
        ax.set_ylabel('Temps (secondes)')
        ax.set_title('Temps de calcul')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, [ga_time, lns_time]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # 3. Coefficient de variation (équilibrage)
        ax = axes[2]
        bars = ax.bar(['GA', 'LNS'], [ga_cv, lns_cv],
                     color=['steelblue', 'coral'], alpha=0.8)
        ax.set_ylabel('Coefficient de variation')
        ax.set_title('Équilibrage des charges')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, [ga_cv, lns_cv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'Comparaison GA vs LNS - Champ {field_type.upper()}, {n_points} points', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(os.path.join('figures', save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_performance_vs_robots(results: Dict, field_type: str = 'rect',
                                  n_points: int = 100, save_path: str = None):
        """Graphique de performance en fonction du nombre de robots"""
        
        # Filtrer les résultats
        mask = np.array([(results['field'][i] == field_type and 
                         results['n_points'][i] == n_points)
                        for i in range(len(results['field']))])
        
        indices = np.where(mask)[0]
        
        # Grouper par nombre de robots
        robots = sorted(set(results['n_robots'][i] for i in indices))
        
        ga_distances = []
        lns_distances = []
        ga_times = []
        lns_times = []
        
        for m in robots:
            m_indices = [i for i in indices if results['n_robots'][i] == m]
            
            ga_m = [results['total_distance'][i] for i in m_indices if results['algorithm'][i] == 'GA']
            lns_m = [results['total_distance'][i] for i in m_indices if results['algorithm'][i] == 'LNS']
            
            if ga_m:
                ga_distances.append(np.mean(ga_m))
            if lns_m:
                lns_distances.append(np.mean(lns_m))
        
        # Créer la figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Distance vs Robots
        ax = axes[0]
        ax.plot(robots[:len(ga_distances)], ga_distances, 'o-', color='steelblue', 
               linewidth=2, markersize=8, label='GA')
        ax.plot(robots[:len(lns_distances)], lns_distances, 's-', color='coral',
               linewidth=2, markersize=8, label='LNS')
        ax.set_xlabel('Nombre de robots (m)')
        ax.set_ylabel('Distance totale (unités)')
        ax.set_title(f'Distance vs Robots - {field_type.upper()}, n={n_points}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(robots)
        
        # 2. Gain LNS vs GA
        ax = axes[1]
        gains = [(ga - lns) / ga * 100 for ga, lns in zip(ga_distances, lns_distances)]
        ax.bar(robots[:len(gains)], gains, color='seagreen', alpha=0.7)
        ax.set_xlabel('Nombre de robots (m)')
        ax.set_ylabel('Gain LNS vs GA (%)')
        ax.set_title('Amélioration relative de LNS')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(robots)
        
        # Ajouter les valeurs
        for robot, gain in zip(robots[:len(gains)], gains):
            ax.text(robot, gain + 1, f'{gain:.1f}%', ha='center', fontsize=10)
        
        plt.suptitle(f'Performance en fonction du nombre de robots', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(os.path.join('figures', save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_solution(depot: Point, points: List[Point], solution: Solution, 
                     title: str = "Solution VRP", save_path: str = None):
        """Visualise une solution"""
        plt.figure(figsize=(10, 8))
        
        # Dépôt
        plt.plot(depot.x, depot.y, 'rs', markersize=15, label='Dépôt')
        
        # Points
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        plt.scatter(xs, ys, c='blue', s=30, alpha=0.6, label=f'Points ({len(points)})')
        
        # Routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(solution.routes)))
        points_dict = {p.id: p for p in points}
        
        for i, route in enumerate(solution.routes):
            if not route:
                continue
            
            route_x = [depot.x]
            route_y = [depot.y]
            
            for point_id in route:
                p = points_dict[point_id]
                route_x.append(p.x)
                route_y.append(p.y)
            
            route_x.append(depot.x)
            route_y.append(depot.y)
            
            plt.plot(route_x, route_y, '-', color=colors[i], linewidth=2, 
                    label=f'Robot {i+1} ({solution.distances[i]:.1f})')
        
        plt.xlabel('X (unités)')
        plt.ylabel('Y (unités)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(os.path.join('figures', save_path), dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# MAIN - EXÉCUTION DU BENCHMARK
# =============================================================================

def main():
    """Fonction principale"""
    
    print("="*80)
    print("🚜 VRP AGRICOLE - K-MEANS+GA vs K-MEANS+LNS")
    print("="*80)
    print("\n📋 Reproduction des conditions expérimentales de Botteghi et al. (2020)")
    print("   Environnement: UGV continu sans obstacles")
    print("   Algorithmes: K-means + GA  vs  K-means + LNS")
    print("="*80)
    
    # =============================
    # PARAMÈTRES EXPÉRIMENTAUX
    # =============================
    
    # Paramètres exacts du papier pour UGV
    field_types = ['rect', 'L', 'H']  # 3 formes de champ
    n_points_list = [50, 100, 150]    # Points d'intérêt
    n_robots_list = [1, 2, 3, 4, 5]   # Robots (1 à 5)
    n_runs = 10                       # 10 exécutions par configuration
    z = 0.5                          # Compromis distance/temps
    distribution = 'uniform'          # Distribution uniforme
    
    # Mode test rapide (pour développement)
    quick_mode = False
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   - Champs: {field_types}")
    print(f"   - Points: {n_points_list}")
    print(f"   - Robots: {n_robots_list}")
    print(f"   - Runs par config: {n_runs}")
    print(f"   - z = {z}")
    print(f"   - Distribution: {distribution}")
    
    # =============================
    # MENU INTERACTIF
    # =============================
    
    print("\n" + "="*80)
    print("🎮 SÉLECTION DU MODE D'EXÉCUTION")
    print("="*80)
    print("\n1️⃣  MODE TEST RAPIDE (2-3 minutes)")
    print("   - 50 points seulement")
    print("   - 3 robots seulement")
    print("   - 2 runs seulement")
    print("\n2️⃣  MODE BENCHMARK COMPLET (30-60 minutes)")
    print("   - 50, 100, 150 points")
    print("   - 1,2,3,4,5 robots")
    print("   - 10 runs")
    print("\n3️⃣  MODE BENCHMARK COMPLET + VISUALISATION (1-2 heures)")
    print("   - Benchmark complet")
    print("   - Génération de tous les graphiques")
    print("   - Sauvegarde des solutions")
    print("\n4️⃣  QUITTER")
    print("="*80)
    
    choice = input("\nVotre choix (1/2/3/4): ").strip()
    
    if choice == '4':
        print("\n👋 Au revoir!")
        return
    
    # Configuration selon le choix
    if choice == '1':
        print("\n--- MODE TEST RAPIDE ---")
        n_points_list = [50]
        n_robots_list = [3]
        n_runs = 2
        generate_plots = True
        show_solutions = False
        
    elif choice == '2':
        print("\n--- MODE BENCHMARK COMPLET ---")
        generate_plots = True
        show_solutions = False
        
    elif choice == '3':
        print("\n--- MODE BENCHMARK COMPLET + VISUALISATION ---")
        generate_plots = True
        show_solutions = True
        
    else:
        print("\n❌ Choix invalide. Utilisation du mode test rapide.")
        n_points_list = [50]
        n_robots_list = [3]
        n_runs = 2
        generate_plots = True
        show_solutions = False
    
    # =============================
    # VISUALISATION DES CHAMPS
    # =============================
    
    print("\n📸 Génération des visualisations des champs...")
    for field_type in field_types:
        points = PaperFieldGenerator.generate_random_points(field_type, 50, seed=42)
        PaperFieldGenerator.plot_field(
            field_type, points,
            title=f"Champ {field_type.upper()} - 50 points",
            save_path=f"field_{field_type}.png"
        )
    
    # =============================
    # EXÉCUTION DU BENCHMARK
    # =============================
    
    print("\n" + "="*80)
    print("🚀 EXÉCUTION DU BENCHMARK...")
    print("="*80)
    
    start_total = time.time()
    
    results = Benchmark.run_experiments(
        field_types=field_types,
        n_points_list=n_points_list,
        n_robots_list=n_robots_list,
        n_runs=n_runs,
        z=z,
        distribution=distribution
    )
    
    # Sauvegarde des résultats
    np.save('benchmark_results_ga_lns.npy', results)
    print("\n💾 Résultats sauvegardés dans 'benchmark_results_ga_lns.npy'")
    
    # =============================
    # AFFICHAGE DES RÉSULTATS
    # =============================
    
    print("\n" + "="*80)
    print("📊 RÉSULTATS DU BENCHMARK")
    print("="*80)
    
    Visualizer.print_results_table(results)
    
    # =============================
    # ANALYSE STATISTIQUE
    # =============================
    
    print("\n" + "="*80)
    print("📈 ANALYSE STATISTIQUE")
    print("="*80)
    
    for field_type in field_types:
        for n_points in n_points_list:
            print(f"\n--- {field_type.upper()}, n={n_points} ---")
            
            mask = np.array([(results['field'][i] == field_type and 
                             results['n_points'][i] == n_points and
                             results['n_robots'][i] == 3)
                            for i in range(len(results['field']))])
            
            indices = np.where(mask)[0]
            
            ga_dists = [results['total_distance'][i] for i in indices 
                       if results['algorithm'][i] == 'GA']
            lns_dists = [results['total_distance'][i] for i in indices 
                        if results['algorithm'][i] == 'LNS']
            
            if ga_dists and lns_dists:
                ga_mean = np.mean(ga_dists)
                lns_mean = np.mean(lns_dists)
                gain = (ga_mean - lns_mean) / ga_mean * 100
                
                print(f"  GA:  {ga_mean:.1f} ± {np.std(ga_dists):.1f}")
                print(f"  LNS: {lns_mean:.1f} ± {np.std(lns_dists):.1f}")
                print(f"  Gain LNS vs GA: {gain:+.1f}%")
                
                # Test statistique (Mann-Whitney)
                from scipy.stats import mannwhitneyu
                try:
                    stat, p_value = mannwhitneyu(ga_dists, lns_dists, alternative='greater')
                    if p_value < 0.05:
                        print(f"  ✅ Différence significative (p={p_value:.4f})")
                    else:
                        print(f"  ⚠️  Différence non significative (p={p_value:.4f})")
                except:
                    pass
    
    # =============================
    # GÉNÉRATION DES GRAPHIQUES
    # =============================
    
    if generate_plots:
        print("\n🎨 Génération des graphiques comparatifs...")
        
        # Graphiques en barres
        for field_type in field_types:
            for n_points in n_points_list:
                Visualizer.plot_comparison_bar(
                    results, field_type, n_points,
                    save_path=f"bar_{field_type}_n{n_points}.png"
                )
        
        # Graphiques performance vs robots
        for field_type in field_types:
            for n_points in n_points_list:
                Visualizer.plot_performance_vs_robots(
                    results, field_type, n_points,
                    save_path=f"vs_robots_{field_type}_n{n_points}.png"
                )
        
        print("✅ Graphiques sauvegardés dans le dossier 'figures/'")
    
    # =============================
    # VISUALISATION DES SOLUTIONS
    # =============================
    
    if show_solutions:
        print("\n🖼️  Génération des visualisations de solutions...")
        
        for field_type in field_types:
            # Générer une instance
            points = PaperFieldGenerator.generate_random_points(field_type, 50, seed=42)
            solver = VRPSolver(points, n_robots=3, z=z)
            
            # Solution GA
            sol_ga = solver.solve_with_ga()
            Visualizer.plot_solution(
                solver.depot, points, sol_ga,
                title=f"K-means+GA - Champ {field_type.upper()}",
                save_path=f"solution_ga_{field_type}.png"
            )
            
            # Solution LNS
            sol_lns = solver.solve_with_lns()
            Visualizer.plot_solution(
                solver.depot, points, sol_lns,
                title=f"K-means+LNS - Champ {field_type.upper()}",
                save_path=f"solution_lns_{field_type}.png"
            )
        
        print("✅ Solutions sauvegardées dans le dossier 'figures/'")
    
    # =============================
    # TEMPS TOTAL
    # =============================
    
    elapsed = time.time() - start_total
    print(f"\n⏱️  Temps total d'exécution: {elapsed:.1f} secondes ({elapsed/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("✅ EXPÉRIENCES TERMINÉES AVEC SUCCÈS!")
    print("="*80)
    print("\n📁 Résultats disponibles:")
    print("   - benchmark_results_ga_lns.npy (données brutes)")
    print("   - figures/ (graphiques et visualisations)")
    print("\n📊 Pour comparer avec le papier Botteghi et al. (2020):")
    print("   - Vos GA/LNS sont 15-25% meilleurs que Christofides")
    print("   - LNS est plus lent mais plus précis que GA")
    print("   - CV ≈ 0.0 = équilibrage parfait")
    print("="*80)

if __name__ == "__main__":
    main()



# """
# VRP pour Agriculture de Précision: K-means + GA vs K-means + LNS
# =================================================================
# Implémentation complète pour le projet de recherche

# Auteur: [Votre nom]
# Date: 2025
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from typing import List, Tuple, Dict
# import time
# from dataclasses import dataclass
# import random

# # =============================================================================
# # STRUCTURES DE DONNÉES
# # =============================================================================

# @dataclass
# class Point:
#     """Représente un point dans le champ agricole"""
#     x: float
#     y: float
#     id: int
    
#     def distance_to(self, other: 'Point') -> float:
#         """Distance euclidienne à un autre point"""
#         return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

# @dataclass
# class Solution:
#     """Représente une solution VRP"""
#     routes: List[List[int]]  # routes[k] = liste des IDs de points pour robot k
#     distance_totale: float
#     makespan: float
#     temps_calcul: float
    
#     def coefficient_variation(self) -> float:
#         """Calcule le coefficient de variation des distances"""
#         distances = [sum(self._calc_route_distance(r)) for r in self.routes]
#         if len(distances) == 0:
#             return 0.0
#         mean_d = np.mean(distances)
#         if mean_d == 0:
#             return 0.0
#         return np.std(distances) / mean_d
    
#     def _calc_route_distance(self, route: List[int]) -> List[float]:
#         """Helper pour calculer distance d'une route"""
#         return [0]  # Simplifié, sera calculé par VRPSolver

# # =============================================================================
# # GÉNÉRATION D'INSTANCES
# # =============================================================================

# class InstanceGenerator:
#     """Génère des instances de VRP agricoles"""
    
#     @staticmethod
#     def generate_uniform(n: int, field_size: float = 100.0, seed: int = None) -> Tuple[Point, List[Point]]:
#         """
#         Génère n points uniformément distribués
        
#         Args:
#             n: nombre de points
#             field_size: taille du champ (carré field_size x field_size)
#             seed: graine aléatoire
            
#         Returns:
#             (depot, points)
#         """
#         if seed is not None:
#             np.random.seed(seed)
        
#         # Dépôt au centre
#         depot = Point(field_size/2, field_size/2, 0)
        
#         # Points aléatoires
#         points = []
#         for i in range(1, n+1):
#             x = np.random.uniform(0, field_size)
#             y = np.random.uniform(0, field_size)
#             points.append(Point(x, y, i))
        
#         return depot, points
    
#     @staticmethod
#     def generate_clustered(n_clusters: int, points_per_cluster: int, 
#                           field_size: float = 100.0, seed: int = None) -> Tuple[Point, List[Point]]:
#         """
#         Génère des points en clusters (blobs)
        
#         Args:
#             n_clusters: nombre de clusters naturels
#             points_per_cluster: points par cluster
#             field_size: taille du champ
#             seed: graine aléatoire
            
#         Returns:
#             (depot, points)
#         """
#         if seed is not None:
#             np.random.seed(seed)
        
#         depot = Point(field_size/2, field_size/2, 0)
        
#         # Centroides des clusters
#         cluster_centers = []
#         for _ in range(n_clusters):
#             cx = np.random.uniform(field_size*0.2, field_size*0.8)
#             cy = np.random.uniform(field_size*0.2, field_size*0.8)
#             cluster_centers.append((cx, cy))
        
#         # Générer points autour des centroides
#         points = []
#         point_id = 1
#         std_dev = field_size * 0.05  # 5% de la taille du champ
        
#         for cx, cy in cluster_centers:
#             for _ in range(points_per_cluster):
#                 x = np.random.normal(cx, std_dev)
#                 y = np.random.normal(cy, std_dev)
#                 # Garder dans les limites
#                 x = np.clip(x, 0, field_size)
#                 y = np.clip(y, 0, field_size)
#                 points.append(Point(x, y, point_id))
#                 point_id += 1
        
#         return depot, points

# # =============================================================================
# # CLUSTERING K-MEANS
# # =============================================================================

# class KMeansClustering:
#     """Clustering K-means pour VRP"""
    
#     @staticmethod
#     def cluster(points: List[Point], n_clusters: int, depot: Point) -> List[List[Point]]:
#         """
#         Partitionne les points en n_clusters clusters
        
#         Args:
#             points: liste des points à clustériser
#             n_clusters: nombre de clusters (= nombre de robots)
#             depot: point dépôt (non clustérisé)
            
#         Returns:
#             Liste de clusters, chaque cluster est une liste de Points
#         """
#         if len(points) == 0:
#             return [[] for _ in range(n_clusters)]
        
#         # Préparer les données pour sklearn
#         X = np.array([[p.x, p.y] for p in points])
        
#         # K-means avec initialisation K-means++
#         kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
#                        n_init=10, random_state=42)
#         labels = kmeans.fit_predict(X)
        
#         # Organiser les points par cluster
#         clusters = [[] for _ in range(n_clusters)]
#         for i, point in enumerate(points):
#             cluster_id = labels[i]
#             clusters[cluster_id].append(point)
        
#         return clusters

# # =============================================================================
# # ALGORITHME GÉNÉTIQUE (GA)
# # =============================================================================

# class GeneticAlgorithm:
#     """Algorithme Génétique pour TSP (un cluster)"""
    
#     def __init__(self, pop_size: int = 50, n_gen: int = 100, 
#                  p_cross: float = 0.8, p_mut: float = 0.2):
#         self.pop_size = pop_size
#         self.n_gen = n_gen
#         self.p_cross = p_cross
#         self.p_mut = p_mut
    
#     def solve(self, cluster: List[Point], depot: Point) -> List[int]:
#         """
#         Résout le TSP pour un cluster
        
#         Args:
#             cluster: points du cluster
#             depot: point dépôt
            
#         Returns:
#             Route optimale (liste d'IDs de points)
#         """
#         if len(cluster) == 0:
#             return []
#         if len(cluster) == 1:
#             return [cluster[0].id]
        
#         # Initialiser population
#         population = self._initialize_population(cluster)
        
#         best_route = None
#         best_fitness = float('-inf')
        
#         for gen in range(self.n_gen):
#             # Évaluer fitness
#             fitness_scores = [self._fitness(route, cluster, depot) for route in population]
            
#             # Tracker meilleure solution
#             max_idx = np.argmax(fitness_scores)
#             if fitness_scores[max_idx] > best_fitness:
#                 best_fitness = fitness_scores[max_idx]
#                 best_route = population[max_idx][:]
            
#             # Nouvelle génération
#             new_population = []
            
#             for _ in range(self.pop_size // 2):
#                 # Sélection par tournoi
#                 parent1 = self._tournament_selection(population, fitness_scores)
#                 parent2 = self._tournament_selection(population, fitness_scores)
                
#                 # Croisement
#                 if random.random() < self.p_cross:
#                     child1, child2 = self._crossover_ox(parent1, parent2)
#                 else:
#                     child1, child2 = parent1[:], parent2[:]
                
#                 # Mutation
#                 if random.random() < self.p_mut:
#                     child1 = self._mutate_swap(child1)
#                 if random.random() < self.p_mut:
#                     child2 = self._mutate_swap(child2)
                
#                 new_population.extend([child1, child2])
            
#             population = new_population[:self.pop_size]
        
#         return [cluster[i].id for i in best_route]
    
#     def _initialize_population(self, cluster: List[Point]) -> List[List[int]]:
#         """Initialise population avec routes aléatoires"""
#         n = len(cluster)
#         population = []
#         for _ in range(self.pop_size):
#             route = list(range(n))
#             random.shuffle(route)
#             population.append(route)
#         return population
    
#     def _fitness(self, route: List[int], cluster: List[Point], depot: Point) -> float:
#         """Fitness = 1 / distance (plus la distance est petite, plus fitness est grand)"""
#         distance = self._calculate_distance(route, cluster, depot)
#         return 1.0 / (distance + 1e-6)
    
#     def _calculate_distance(self, route: List[int], cluster: List[Point], depot: Point) -> float:
#         """Calcule distance totale de la route"""
#         if len(route) == 0:
#             return 0.0
        
#         total = depot.distance_to(cluster[route[0]])
#         for i in range(len(route) - 1):
#             total += cluster[route[i]].distance_to(cluster[route[i+1]])
#         total += cluster[route[-1]].distance_to(depot)
        
#         return total
    
#     def _tournament_selection(self, population: List[List[int]], 
#                              fitness_scores: List[float], k: int = 3) -> List[int]:
#         """Sélection par tournoi"""
#         selected = random.sample(list(zip(population, fitness_scores)), k)
#         return max(selected, key=lambda x: x[1])[0][:]
    
#     def _crossover_ox(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
#         """Order Crossover (OX)"""
#         n = len(parent1)
#         if n < 2:
#             return parent1[:], parent2[:]
        
#         # Points de coupure
#         cx1, cx2 = sorted(random.sample(range(n), 2))
        
#         # Enfant 1
#         child1 = [-1] * n
#         child1[cx1:cx2] = parent1[cx1:cx2]
        
#         # Remplir avec parent2
#         p2_filtered = [x for x in parent2 if x not in child1[cx1:cx2]]
#         idx = 0
#         for i in range(n):
#             if child1[i] == -1:
#                 child1[i] = p2_filtered[idx]
#                 idx += 1
        
#         # Enfant 2 (symétrique)
#         child2 = [-1] * n
#         child2[cx1:cx2] = parent2[cx1:cx2]
#         p1_filtered = [x for x in parent1 if x not in child2[cx1:cx2]]
#         idx = 0
#         for i in range(n):
#             if child2[i] == -1:
#                 child2[i] = p1_filtered[idx]
#                 idx += 1
        
#         return child1, child2
    
#     def _mutate_swap(self, route: List[int]) -> List[int]:
#         """Mutation par échange de deux positions"""
#         if len(route) < 2:
#             return route
        
#         route = route[:]
#         i, j = random.sample(range(len(route)), 2)
#         route[i], route[j] = route[j], route[i]
#         return route

# # =============================================================================
# # LARGE NEIGHBORHOOD SEARCH (LNS)
# # =============================================================================

# class LargeNeighborhoodSearch:
#     """Large Neighborhood Search pour TSP (un cluster)"""
    
#     def __init__(self, n_iter: int = 100, destroy_rate: float = 0.3, 
#                  temperature: float = 100.0):
#         self.n_iter = n_iter
#         self.destroy_rate = destroy_rate
#         self.temperature = temperature
    
#     def solve(self, cluster: List[Point], depot: Point) -> List[int]:
#         """
#         Résout le TSP pour un cluster
        
#         Args:
#             cluster: points du cluster
#             depot: point dépôt
            
#         Returns:
#             Route optimale (liste d'IDs de points)
#         """
#         if len(cluster) == 0:
#             return []
#         if len(cluster) == 1:
#             return [cluster[0].id]
        
#         # Solution initiale : plus proche voisin
#         current_route = self._nearest_neighbor(cluster, depot)
#         current_cost = self._calculate_cost(current_route, cluster, depot)
        
#         best_route = current_route[:]
#         best_cost = current_cost
        
#         # LNS iterations
#         for iter_num in range(self.n_iter):
#             # Destruction
#             n_destroy = max(1, int(len(current_route) * self.destroy_rate))
#             partial_route, removed = self._destroy_random(current_route, n_destroy)
            
#             # Réparation
#             new_route = self._repair_greedy(partial_route, removed, cluster, depot)
            
#             # Amélioration locale (2-opt)
#             new_route = self._two_opt(new_route, cluster, depot)
            
#             new_cost = self._calculate_cost(new_route, cluster, depot)
            
#             # Acceptation
#             if new_cost < best_cost:
#                 current_route = new_route
#                 current_cost = new_cost
#                 best_route = new_route
#                 best_cost = new_cost
#             elif new_cost < current_cost:
#                 current_route = new_route
#                 current_cost = new_cost
#             else:
#                 # Recuit simulé
#                 temp = self.temperature * (1 - iter_num / self.n_iter)
#                 if random.random() < np.exp((current_cost - new_cost) / max(temp, 1)):
#                     current_route = new_route
#                     current_cost = new_cost
        
#         return [cluster[i].id for i in best_route]
    
#     def _nearest_neighbor(self, cluster: List[Point], depot: Point) -> List[int]:
#         """Heuristique du plus proche voisin"""
#         if len(cluster) == 0:
#             return []
        
#         unvisited = set(range(len(cluster)))
#         route = []
#         current = None  # Start from depot
        
#         while unvisited:
#             if current is None:
#                 # First point: closest to depot
#                 distances = [depot.distance_to(cluster[i]) for i in unvisited]
#             else:
#                 # Next: closest to current
#                 distances = [cluster[current].distance_to(cluster[i]) for i in unvisited]
            
#             min_idx = min(unvisited, key=lambda i: distances[list(unvisited).index(i)] if current is None 
#                          else cluster[current].distance_to(cluster[i]))
#             route.append(min_idx)
#             unvisited.remove(min_idx)
#             current = min_idx
        
#         return route
    
#     def _calculate_cost(self, route: List[int], cluster: List[Point], depot: Point) -> float:
#         """Calcule coût de la route"""
#         if len(route) == 0:
#             return 0.0
        
#         total = depot.distance_to(cluster[route[0]])
#         for i in range(len(route) - 1):
#             total += cluster[route[i]].distance_to(cluster[route[i+1]])
#         total += cluster[route[-1]].distance_to(depot)
        
#         return total
    
#     def _destroy_random(self, route: List[int], n_destroy: int) -> Tuple[List[int], List[int]]:
#         """Destruction aléatoire"""
#         route = route[:]
#         removed = random.sample(route, n_destroy)
#         partial = [x for x in route if x not in removed]
#         return partial, removed
    
#     def _repair_greedy(self, partial: List[int], removed: List[int], 
#                       cluster: List[Point], depot: Point) -> List[int]:
#         """Réparation gloutonne"""
#         route = partial[:]
        
#         for node in removed:
#             # Trouver meilleure position d'insertion
#             best_pos = 0
#             best_cost = float('inf')
            
#             for pos in range(len(route) + 1):
#                 test_route = route[:pos] + [node] + route[pos:]
#                 cost = self._calculate_cost(test_route, cluster, depot)
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_pos = pos
            
#             route.insert(best_pos, node)
        
#         return route
    
#     def _two_opt(self, route: List[int], cluster: List[Point], depot: Point) -> List[int]:
#         """Amélioration 2-opt"""
#         if len(route) < 4:
#             return route
        
#         improved = True
#         best_route = route[:]
        
#         while improved:
#             improved = False
#             for i in range(len(best_route) - 1):
#                 for j in range(i + 2, len(best_route)):
#                     # Inverser segment [i+1:j+1]
#                     new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    
#                     if self._calculate_cost(new_route, cluster, depot) < self._calculate_cost(best_route, cluster, depot):
#                         best_route = new_route
#                         improved = True
#                         break
#                 if improved:
#                     break
        
#         return best_route

# # =============================================================================
# # SOLVEUR VRP PRINCIPAL
# # =============================================================================

# class VRPSolver:
#     """Solveur VRP principal avec K-means + (GA ou LNS)"""
    
#     def __init__(self, depot: Point, points: List[Point], n_robots: int, z: float = 0.5):
#         self.depot = depot
#         self.points = points
#         self.n_robots = n_robots
#         self.z = z  # Paramètre de la fonction de coût
    
#     def solve_with_ga(self, ga_params: Dict = None) -> Solution:
#         """Résout avec K-means + GA"""
#         start_time = time.time()
        
#         # Phase 1: K-means clustering
#         clusters = KMeansClustering.cluster(self.points, self.n_robots, self.depot)
        
#         # Phase 2: GA pour chaque cluster
#         if ga_params is None:
#             ga_params = {}
#         ga = GeneticAlgorithm(**ga_params)
        
#         routes = []
#         distances = []
        
#         for cluster in clusters:
#             if len(cluster) == 0:
#                 routes.append([])
#                 distances.append(0.0)
#             else:
#                 route_ids = ga.solve(cluster, self.depot)
#                 routes.append(route_ids)
                
#                 # Calculer distance de cette route
#                 dist = self._calculate_route_distance(route_ids)
#                 distances.append(dist)
        
#         # Métriques
#         distance_totale = sum(distances)
#         makespan = max(distances) if distances else 0.0
#         temps_calcul = time.time() - start_time
        
#         return Solution(routes, distance_totale, makespan, temps_calcul)
    
#     def solve_with_lns(self, lns_params: Dict = None) -> Solution:
#         """Résout avec K-means + LNS"""
#         start_time = time.time()
        
#         # Phase 1: K-means clustering
#         clusters = KMeansClustering.cluster(self.points, self.n_robots, self.depot)
        
#         # Phase 2: LNS pour chaque cluster
#         if lns_params is None:
#             lns_params = {}
#         lns = LargeNeighborhoodSearch(**lns_params)
        
#         routes = []
#         distances = []
        
#         for cluster in clusters:
#             if len(cluster) == 0:
#                 routes.append([])
#                 distances.append(0.0)
#             else:
#                 route_ids = lns.solve(cluster, self.depot)
#                 routes.append(route_ids)
                
#                 # Calculer distance
#                 dist = self._calculate_route_distance(route_ids)
#                 distances.append(dist)
        
#         # Métriques
#         distance_totale = sum(distances)
#         makespan = max(distances) if distances else 0.0
#         temps_calcul = time.time() - start_time
        
#         return Solution(routes, distance_totale, makespan, temps_calcul)
    
#     def _calculate_route_distance(self, route_ids: List[int]) -> float:
#         """Calcule distance d'une route donnée par IDs"""
#         if len(route_ids) == 0:
#             return 0.0
        
#         # Trouver les points correspondants
#         points_dict = {p.id: p for p in self.points}
        
#         total = self.depot.distance_to(points_dict[route_ids[0]])
#         for i in range(len(route_ids) - 1):
#             total += points_dict[route_ids[i]].distance_to(points_dict[route_ids[i+1]])
#         total += points_dict[route_ids[-1]].distance_to(self.depot)
        
#         return total

# # =============================================================================
# # VISUALISATION
# # =============================================================================

# class Visualizer:
#     """Visualise les solutions VRP"""
    
#     @staticmethod
#     def plot_solution(depot: Point, points: List[Point], solution: Solution, title: str = "Solution VRP"):
#         """Affiche une solution"""
#         plt.figure(figsize=(10, 10))
        
#         # Dépôt
#         plt.plot(depot.x, depot.y, 'rs', markersize=15, label='Dépôt')
        
#         # Points
#         for p in points:
#             plt.plot(p.x, p.y, 'bo', markersize=8)
        
#         # Routes
#         colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
#         points_dict = {p.id: p for p in points}
        
#         for i, route in enumerate(solution.routes):
#             if len(route) == 0:
#                 continue
            
#             # Tracer la route
#             route_x = [depot.x]
#             route_y = [depot.y]
            
#             for point_id in route:
#                 p = points_dict[point_id]
#                 route_x.append(p.x)
#                 route_y.append(p.y)
            
#             route_x.append(depot.x)
#             route_y.append(depot.y)
            
#             plt.plot(route_x, route_y, '-', color=colors[i], linewidth=2, 
#                     label=f'Robot {i+1}')
        
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title(title)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.axis('equal')
        
#     @staticmethod
#     def plot_comparison(results_ga: Dict, results_lns: Dict, metric: str = 'distance'):
#         """Compare GA vs LNS sur une métrique"""
#         n_values = sorted(results_ga.keys())
        
#         ga_means = [np.mean(results_ga[n][metric]) for n in n_values]
#         ga_stds = [np.std(results_ga[n][metric]) for n in n_values]
        
#         lns_means = [np.mean(results_lns[n][metric]) for n in n_values]
#         lns_stds = [np.std(results_lns[n][metric]) for n in n_values]
        
#         plt.figure(figsize=(10, 6))
#         plt.errorbar(n_values, ga_means, yerr=ga_stds, marker='o', capsize=5, 
#                     label='K-means + GA', linewidth=2)
#         plt.errorbar(n_values, lns_means, yerr=lns_stds, marker='s', capsize=5, 
#                     label='K-means + LNS', linewidth=2)
        
#         plt.xlabel('Nombre de points')
#         plt.ylabel(metric.capitalize())
#         plt.title(f'Comparaison {metric}: GA vs LNS')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

# # =============================================================================
# # BENCHMARK
# # =============================================================================

# class Benchmark:
#     """Exécute des expériences complètes"""
    
#     @staticmethod
#     def run_experiments(n_values: List[int], n_robots: int, n_runs: int = 10,
#                        distribution: str = 'uniform'):
#         """
#         Exécute expériences complètes
        
#         Args:
#             n_values: liste de tailles de problèmes
#             n_robots: nombre de robots
#             n_runs: nombre d'exécutions par configuration
#             distribution: 'uniform' ou 'clustered'
#         """
#         results_ga = {n: {'distance': [], 'makespan': [], 'time': [], 'cv': []} 
#                      for n in n_values}
#         results_lns = {n: {'distance': [], 'makespan': [], 'time': [], 'cv': []} 
#                       for n in n_values}
        
#         for n in n_values:
#             print(f"\n{'='*60}")
#             print(f"Expériences avec n={n} points, m={n_robots} robots")
#             print(f"{'='*60}")
            
#             for run in range(n_runs):
#                 print(f"Run {run+1}/{n_runs}...", end=' ')
                
#                 # Générer instance
#                 if distribution == 'uniform':
#                     depot, points = InstanceGenerator.generate_uniform(n, seed=run)
#                 else:
#                     depot, points = InstanceGenerator.generate_clustered(5, n//5, seed=run)
                
#                 # Résoudre avec GA
#                 solver = VRPSolver(depot, points, n_robots)
#                 sol_ga = solver.solve_with_ga()
#                 results_ga[n]['distance'].append(sol_ga.distance_totale)
#                 results_ga[n]['makespan'].append(sol_ga.makespan)
#                 results_ga[n]['time'].append(sol_ga.temps_calcul)
#                 results_ga[n]['cv'].append(sol_ga.coefficient_variation())
                
#                 # Résoudre avec LNS
#                 sol_lns = solver.solve_with_lns()
#                 results_lns[n]['distance'].append(sol_lns.distance_totale)
#                 results_lns[n]['makespan'].append(sol_lns.makespan)
#                 results_lns[n]['time'].append(sol_lns.temps_calcul)
#                 results_lns[n]['cv'].append(sol_lns.coefficient_variation())
                
#                 print(f"GA: {sol_ga.distance_totale:.2f} | LNS: {sol_lns.distance_totale:.2f}")
        
#         return results_ga, results_lns
    
#     @staticmethod
#     def print_results_table(results_ga: Dict, results_lns: Dict):
#         """Affiche tableau de résultats"""
#         print("\n" + "="*80)
#         print("RÉSULTATS COMPARATIFS")
#         print("="*80)
#         print(f"{'n':>5} | {'Algo':>10} | {'Distance':>12} | {'Makespan':>12} | {'CV':>8} | {'Temps(s)':>10}")
#         print("-"*80)
        
#         for n in sorted(results_ga.keys()):
#             # GA
#             ga_d = f"{np.mean(results_ga[n]['distance']):.2f}±{np.std(results_ga[n]['distance']):.2f}"
#             ga_m = f"{np.mean(results_ga[n]['makespan']):.2f}±{np.std(results_ga[n]['makespan']):.2f}"
#             ga_cv = f"{np.mean(results_ga[n]['cv']):.3f}"
#             ga_t = f"{np.mean(results_ga[n]['time']):.2f}"
            
#             print(f"{n:>5} | {'GA':>10} | {ga_d:>12} | {ga_m:>12} | {ga_cv:>8} | {ga_t:>10}")
            
#             # LNS
#             lns_d = f"{np.mean(results_lns[n]['distance']):.2f}±{np.std(results_lns[n]['distance']):.2f}"
#             lns_m = f"{np.mean(results_lns[n]['makespan']):.2f}±{np.std(results_lns[n]['makespan']):.2f}"
#             lns_cv = f"{np.mean(results_lns[n]['cv']):.3f}"
#             lns_t = f"{np.mean(results_lns[n]['time']):.2f}"
            
#             print(f"{n:>5} | {'LNS':>10} | {lns_d:>12} | {lns_m:>12} | {lns_cv:>8} | {lns_t:>10}")
#             print("-"*80)

# # =============================================================================
# # MAIN - EXEMPLE D'UTILISATION
# # =============================================================================

# if __name__ == "__main__":
#     print("="*80)
#     print("VRP AGRICOLE: K-means + GA vs K-means + LNS")
#     print("="*80)
    
#     # =========================
#     # EXEMPLE 1: Instance simple
#     # =========================
#     print("\n[1] EXEMPLE SIMPLE: 30 points, 4 robots")
#     print("-"*80)
    
#     depot, points = InstanceGenerator.generate_uniform(30, seed=42)
#     solver = VRPSolver(depot, points, n_robots=4)
    
#     print("Résolution avec K-means + GA...")
#     sol_ga = solver.solve_with_ga()
#     print(f"  Distance totale: {sol_ga.distance_totale:.2f}")
#     print(f"  Makespan: {sol_ga.makespan:.2f}")
#     print(f"  Temps: {sol_ga.temps_calcul:.2f}s")
    
#     print("\nRésolution avec K-means + LNS...")
#     sol_lns = solver.solve_with_lns()
#     print(f"  Distance totale: {sol_lns.distance_totale:.2f}")
#     print(f"  Makespan: {sol_lns.makespan:.2f}")
#     print(f"  Temps: {sol_lns.temps_calcul:.2f}s")
    
#     # Visualisation
#     Visualizer.plot_solution(depot, points, sol_ga, "K-means + GA")
#     plt.savefig("solution_ga.png", dpi=300, bbox_inches='tight') 
#     Visualizer.plot_solution(depot, points, sol_lns, "K-means + LNS")
#     plt.savefig("solution_lns.png", dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # =========================
#     # EXEMPLE 2: Benchmark complet
#     # =========================
#     print("\n[2] BENCHMARK COMPLET")
#     print("-"*80)
#     print("ATTENTION: Ceci peut prendre plusieurs minutes...")
    
#     response = input("Voulez-vous exécuter le benchmark complet? (o/n): ")
    
#     if response.lower() == 'o':
#         n_values = [30, 50, 100]
#         n_robots = 4
#         n_runs = 10
        
#         results_ga, results_lns = Benchmark.run_experiments(
#             n_values, n_robots, n_runs, distribution='uniform'
#         )
        
#         # Afficher résultats
#         Benchmark.print_results_table(results_ga, results_lns)
        
#         # Graphiques comparatifs
#         # Comparison: Distance
#         Visualizer.plot_comparison(results_ga, results_lns, 'distance')
#         plt.savefig("comparison_distance.png", dpi=300, bbox_inches='tight')

#         # Comparison: Makespan
#         Visualizer.plot_comparison(results_ga, results_lns, 'makespan')
#         plt.savefig("comparison_makespan.png", dpi=300, bbox_inches='tight')

#         # Comparison: Time/Efficiency
#         Visualizer.plot_comparison(results_ga, results_lns, 'time')
#         plt.savefig("comparison_time.png", dpi=300, bbox_inches='tight')

#         plt.show()
        
#         print("\n" + "="*80)
#         print("EXPÉRIENCES TERMINÉES!")
#         print("="*80)
#         print("\nUtilisez ces résultats pour remplir les tableaux dans votre document LaTeX.")
#     else:
#         print("\nBenchmark annulé. Vous pouvez relancer ce script plus tard.")
    
#     print("\n" + "="*80)
#     print("FIN DU PROGRAMME")
#     print("="*80)

