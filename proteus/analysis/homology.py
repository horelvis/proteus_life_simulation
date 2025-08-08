"""
Análisis de homología persistente para trayectorias
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from collections import defaultdict


class PersistentHomology:
    """
    Calcula homología persistente de trayectorias
    Esto captura la "forma" topológica del movimiento
    """
    
    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension
        self.persistence_diagrams = {}
        
    def compute_persistence(self, trajectory: np.ndarray, 
                          max_epsilon: Optional[float] = None) -> Dict:
        """
        Calcula el diagrama de persistencia de una trayectoria
        """
        if len(trajectory) < 3:
            return {'H0': [], 'H1': [], 'betti_numbers': [0, 0]}
            
        # Calcular matriz de distancias
        dist_matrix = squareform(pdist(trajectory))
        
        if max_epsilon is None:
            max_epsilon = np.max(dist_matrix) * 0.5
            
        # Construir filtración
        epsilon_values = np.linspace(0, max_epsilon, 50)
        
        # Calcular homología para cada valor de epsilon
        persistence = {f'H{i}': [] for i in range(self.max_dimension)}
        
        # H0 (componentes conectadas)
        h0_births = []
        h0_deaths = []
        
        for eps in epsilon_values:
            # Construir grafo de Rips
            adjacency = (dist_matrix <= eps).astype(int)
            n_components, labels = connected_components(csr_matrix(adjacency))
            
            # Rastrear nacimientos y muertes
            if len(h0_births) == 0:
                h0_births = [0] * n_components
            
            if n_components < len(h0_births):
                # Algunas componentes murieron
                n_deaths = len(h0_births) - n_components
                h0_deaths.extend([eps] * n_deaths)
                
        # Crear pares de persistencia para H0
        for i in range(len(h0_births)):
            if i < len(h0_deaths):
                persistence['H0'].append((h0_births[i], h0_deaths[i]))
            else:
                persistence['H0'].append((h0_births[i], max_epsilon))
                
        # H1 (loops) - Implementación simplificada
        persistence['H1'] = self._compute_h1_persistence(trajectory, dist_matrix, epsilon_values)
        
        # Calcular números de Betti
        betti_numbers = self._compute_betti_numbers(persistence, max_epsilon/2)
        
        return {
            'persistence': persistence,
            'betti_numbers': betti_numbers,
            'max_epsilon': max_epsilon,
            'persistence_entropy': self._compute_persistence_entropy(persistence)
        }
    
    def _compute_h1_persistence(self, trajectory: np.ndarray, 
                               dist_matrix: np.ndarray,
                               epsilon_values: np.ndarray) -> List[Tuple[float, float]]:
        """
        Calcula persistencia de H1 (loops) - versión simplificada
        """
        h1_features = []
        
        # Detectar loops simples basados en retornos cercanos
        for i in range(len(trajectory) - 10):
            for j in range(i + 10, len(trajectory)):
                if dist_matrix[i, j] < 10:  # Punto de retorno cercano
                    # Verificar que forma un loop real
                    loop_length = j - i
                    if loop_length > 5:
                        # Calcular el diámetro del loop
                        loop_points = trajectory[i:j+1]
                        loop_dists = pdist(loop_points)
                        
                        if len(loop_dists) > 0:
                            birth = np.min(loop_dists)
                            death = np.max(loop_dists)
                            
                            if death > birth * 1.5:  # Loop significativo
                                h1_features.append((birth, death))
                                
        return h1_features[:10]  # Limitar a 10 features más persistentes
    
    def _compute_betti_numbers(self, persistence: Dict, epsilon: float) -> List[int]:
        """
        Calcula los números de Betti para un valor de epsilon dado
        """
        betti = []
        
        for dim in range(self.max_dimension):
            key = f'H{dim}'
            if key in persistence:
                # Contar features vivas en epsilon
                count = sum(1 for (birth, death) in persistence[key] 
                          if birth <= epsilon < death)
                betti.append(count)
            else:
                betti.append(0)
                
        return betti
    
    def _compute_persistence_entropy(self, persistence: Dict) -> float:
        """
        Calcula la entropía de persistencia
        Mide la complejidad de la estructura topológica
        """
        all_lifetimes = []
        
        for dim_features in persistence.values():
            for birth, death in dim_features:
                if death > birth:
                    all_lifetimes.append(death - birth)
                    
        if not all_lifetimes:
            return 0.0
            
        # Normalizar tiempos de vida
        total_lifetime = sum(all_lifetimes)
        if total_lifetime == 0:
            return 0.0
            
        probabilities = [l / total_lifetime for l in all_lifetimes]
        
        # Calcular entropía
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        
        return entropy
    
    def compare_homologies(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compara la homología de dos trayectorias
        Retorna una medida de similitud topológica
        """
        # Calcular persistencia para ambas
        pers1 = self.compute_persistence(traj1)
        pers2 = self.compute_persistence(traj2)
        
        # Comparar números de Betti
        betti_dist = np.linalg.norm(
            np.array(pers1['betti_numbers']) - np.array(pers2['betti_numbers'])
        )
        
        # Comparar entropías
        entropy_dist = abs(pers1['persistence_entropy'] - pers2['persistence_entropy'])
        
        # Comparar diagramas de persistencia usando distancia de Wasserstein simplificada
        wasserstein_dist = self._wasserstein_distance(
            pers1['persistence'], 
            pers2['persistence']
        )
        
        # Combinar métricas
        similarity = 1.0 / (1.0 + betti_dist + entropy_dist + wasserstein_dist)
        
        return similarity
    
    def _wasserstein_distance(self, pers1: Dict, pers2: Dict) -> float:
        """
        Calcula una versión simplificada de la distancia de Wasserstein
        entre diagramas de persistencia
        """
        total_dist = 0
        
        for dim in range(self.max_dimension):
            key = f'H{dim}'
            
            if key in pers1 and key in pers2:
                # Extraer puntos de persistencia
                points1 = np.array(pers1[key]) if pers1[key] else np.array([]).reshape(0, 2)
                points2 = np.array(pers2[key]) if pers2[key] else np.array([]).reshape(0, 2)
                
                if len(points1) > 0 and len(points2) > 0:
                    # Distancia entre centroides
                    centroid1 = np.mean(points1, axis=0)
                    centroid2 = np.mean(points2, axis=0)
                    total_dist += np.linalg.norm(centroid1 - centroid2)
                    
        return total_dist
    
    def visualize_persistence_diagram(self, persistence_data: Dict, 
                                    title: str = "Persistence Diagram") -> Tuple:
        """
        Visualiza el diagrama de persistencia
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = ['blue', 'red', 'green']
        
        for dim in range(min(self.max_dimension, 3)):
            key = f'H{dim}'
            if key in persistence_data['persistence']:
                points = persistence_data['persistence'][key]
                
                if points:
                    points_array = np.array(points)
                    
                    # Diagrama de persistencia
                    axes[0].scatter(points_array[:, 0], points_array[:, 1], 
                                  c=colors[dim], label=f'H{dim}', alpha=0.6, s=50)
                    
                    # Código de barras
                    for i, (birth, death) in enumerate(points):
                        y_pos = dim * 10 + i * 0.5
                        axes[1].plot([birth, death], [y_pos, y_pos], 
                                   c=colors[dim], linewidth=2)
                        
        # Diagonal en diagrama de persistencia
        max_val = persistence_data['max_epsilon']
        axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        axes[0].set_xlabel('Birth')
        axes[0].set_ylabel('Death')
        axes[0].set_title(f'{title} - Persistence Diagram')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epsilon')
        axes[1].set_ylabel('Feature')
        axes[1].set_title(f'{title} - Barcode')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, axes
    
    def extract_topological_summary(self, trajectory: np.ndarray) -> Dict:
        """
        Extrae un resumen de características topológicas
        """
        persistence_data = self.compute_persistence(trajectory)
        
        summary = {
            'betti_0': persistence_data['betti_numbers'][0] if len(persistence_data['betti_numbers']) > 0 else 0,
            'betti_1': persistence_data['betti_numbers'][1] if len(persistence_data['betti_numbers']) > 1 else 0,
            'persistence_entropy': persistence_data['persistence_entropy'],
            'n_significant_loops': 0,
            'max_loop_persistence': 0,
            'trajectory_complexity': 0
        }
        
        # Analizar loops significativos
        if 'H1' in persistence_data['persistence']:
            h1_features = persistence_data['persistence']['H1']
            if h1_features:
                persistences = [death - birth for birth, death in h1_features]
                summary['n_significant_loops'] = len([p for p in persistences if p > 10])
                summary['max_loop_persistence'] = max(persistences) if persistences else 0
                
        # Complejidad de la trayectoria
        if len(trajectory) > 10:
            # Basada en la dimensión fractal de la trayectoria
            summary['trajectory_complexity'] = self._estimate_fractal_dimension(trajectory)
            
        return summary
    
    def _estimate_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """
        Estima la dimensión fractal de una trayectoria usando box-counting
        """
        if len(trajectory) < 10:
            return 1.0
            
        # Normalizar trayectoria
        traj_normalized = trajectory - np.min(trajectory, axis=0)
        max_coord = np.max(traj_normalized)
        
        if max_coord == 0:
            return 1.0
            
        # Box-counting
        box_sizes = [max_coord / (2**i) for i in range(2, 8)]
        counts = []
        
        for box_size in box_sizes:
            if box_size > 0:
                # Discretizar
                boxes = (traj_normalized / box_size).astype(int)
                unique_boxes = len(np.unique(boxes, axis=0))
                counts.append(unique_boxes)
                
        if len(counts) > 1 and len(box_sizes) > 1:
            # Ajuste log-log
            log_sizes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(counts)
            
            # Dimensión fractal es la pendiente negativa
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return -slope
        else:
            return 1.0