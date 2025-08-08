"""
Detección y análisis de propiedades emergentes
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from scipy.spatial import distance_matrix
from scipy.stats import entropy
from collections import defaultdict
import networkx as nx


class EmergenceAnalyzer:
    """
    Analiza propiedades emergentes en el sistema
    Busca patrones y comportamientos no programados explícitamente
    """
    
    def __init__(self):
        self.behavior_history = []
        self.collective_patterns = {}
        self.emergence_metrics = {}
        
    def analyze_collective_behavior(self, organisms: List, time: float) -> Dict:
        """
        Analiza comportamientos colectivos emergentes
        """
        if len(organisms) < 2:
            return {}
            
        # Extraer posiciones y velocidades
        positions = np.array([org.position for org in organisms])
        velocities = np.array([org.velocity for org in organisms])
        
        analysis = {
            'time': time,
            'population_size': len(organisms),
            'spatial_patterns': self._analyze_spatial_patterns(positions),
            'movement_patterns': self._analyze_movement_patterns(velocities),
            'social_structure': self._analyze_social_structure(organisms),
            'information_flow': self._analyze_information_flow(organisms)
        }
        
        # Detectar comportamientos emergentes específicos
        analysis['emergent_behaviors'] = self._detect_emergent_behaviors(analysis)
        
        # Guardar en historia
        self.behavior_history.append(analysis)
        
        return analysis
    
    def _analyze_spatial_patterns(self, positions: np.ndarray) -> Dict:
        """
        Analiza patrones espaciales emergentes
        """
        # Calcular matriz de distancias
        dist_matrix = distance_matrix(positions, positions)
        
        # Clustering espacial
        avg_nearest_neighbor = np.mean([np.min(dist_matrix[i, dist_matrix[i] > 0]) 
                                      for i in range(len(positions)) 
                                      if np.any(dist_matrix[i] > 0)])
        
        # Dispersión
        center = np.mean(positions, axis=0)
        dispersion = np.mean([np.linalg.norm(pos - center) for pos in positions])
        
        # Detectar formaciones
        formations = self._detect_formations(positions, dist_matrix)
        
        return {
            'avg_nearest_neighbor': avg_nearest_neighbor,
            'dispersion': dispersion,
            'clustering_coefficient': self._calculate_clustering_coefficient(dist_matrix),
            'formations': formations,
            'spatial_entropy': self._calculate_spatial_entropy(positions)
        }
    
    def _analyze_movement_patterns(self, velocities: np.ndarray) -> Dict:
        """
        Analiza patrones de movimiento colectivo
        """
        # Alineación (qué tan paralelos son los movimientos)
        if len(velocities) > 1:
            # Normalizar velocidades
            speeds = np.linalg.norm(velocities, axis=1)
            valid_idx = speeds > 0.01
            
            if np.any(valid_idx):
                normalized_vel = velocities[valid_idx] / speeds[valid_idx, np.newaxis]
                
                # Calcular alineación promedio
                alignment = 0
                count = 0
                for i in range(len(normalized_vel)):
                    for j in range(i+1, len(normalized_vel)):
                        alignment += np.dot(normalized_vel[i], normalized_vel[j])
                        count += 1
                        
                avg_alignment = alignment / count if count > 0 else 0
            else:
                avg_alignment = 0
                
            # Velocidad promedio del grupo
            avg_speed = np.mean(speeds)
            
            # Coherencia del movimiento
            if len(velocities) > 0:
                mean_velocity = np.mean(velocities, axis=0)
                coherence = np.linalg.norm(mean_velocity) / (avg_speed + 0.01)
            else:
                coherence = 0
        else:
            avg_alignment = 0
            avg_speed = 0
            coherence = 0
            
        return {
            'alignment': avg_alignment,
            'avg_speed': avg_speed,
            'coherence': coherence,
            'movement_diversity': self._calculate_movement_diversity(velocities)
        }
    
    def _analyze_social_structure(self, organisms: List) -> Dict:
        """
        Analiza estructuras sociales emergentes
        """
        if len(organisms) < 2:
            return {'network_density': 0, 'communities': 0}
            
        # Construir red de interacciones basada en proximidad
        G = nx.Graph()
        
        for i, org1 in enumerate(organisms):
            G.add_node(i, generation=org1.generation, age=org1.age)
            
            for j, org2 in enumerate(organisms[i+1:], i+1):
                dist = np.linalg.norm(org1.position - org2.position)
                if dist < 50:  # Threshold de interacción
                    G.add_edge(i, j, weight=1/dist)
                    
        # Analizar estructura de red
        if G.number_of_edges() > 0:
            density = nx.density(G)
            
            # Detectar comunidades
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                n_communities = len(communities)
                modularity = nx.community.modularity(G, communities)
            except:
                n_communities = 1
                modularity = 0
                
            # Centralidad
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                avg_centrality = np.mean(list(centrality.values()))
            else:
                avg_centrality = 0
        else:
            density = 0
            n_communities = 0
            modularity = 0
            avg_centrality = 0
            
        return {
            'network_density': density,
            'communities': n_communities,
            'modularity': modularity,
            'avg_centrality': avg_centrality
        }
    
    def _analyze_information_flow(self, organisms: List) -> Dict:
        """
        Analiza el flujo de información topológica
        """
        # Rastrear linajes
        lineages = defaultdict(list)
        for org in organisms:
            if org.seed:
                parent_id = org.seed.parent_id.split('_')[0]
                lineages[parent_id].append(org.id)
                
        # Diversidad de linajes
        lineage_diversity = len(lineages)
        
        # Profundidad generacional
        generations = [org.generation for org in organisms]
        generation_spread = max(generations) - min(generations) if generations else 0
        
        # Convergencia de características
        if len(organisms) > 1:
            # Comparar características topológicas
            features = []
            for org in organisms:
                if hasattr(org, 'get_topological_features'):
                    feat = org.get_topological_features()
                    if feat:
                        features.append([
                            feat.get('persistence', 0),
                            feat.get('complexity', 0),
                            feat.get('curvature', 0)
                        ])
                        
            if features:
                features_array = np.array(features)
                feature_variance = np.mean(np.var(features_array, axis=0))
            else:
                feature_variance = 1.0
        else:
            feature_variance = 1.0
            
        return {
            'lineage_diversity': lineage_diversity,
            'generation_spread': generation_spread,
            'feature_convergence': 1 / (feature_variance + 0.01),
            'information_entropy': self._calculate_information_entropy(organisms)
        }
    
    def _detect_formations(self, positions: np.ndarray, 
                          dist_matrix: np.ndarray) -> List[str]:
        """
        Detecta formaciones espaciales específicas
        """
        formations = []
        n = len(positions)
        
        if n < 3:
            return formations
            
        # Detectar líneas
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Verificar colinealidad
                    v1 = positions[j] - positions[i]
                    v2 = positions[k] - positions[i]
                    
                    # Producto cruzado para verificar colinealidad
                    cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if cross < 5.0:  # Threshold de colinealidad
                        formations.append("line_formation")
                        break
                        
        # Detectar círculos
        if n >= 6:
            # Calcular centro geométrico
            center = np.mean(positions, axis=0)
            distances_to_center = [np.linalg.norm(pos - center) for pos in positions]
            
            # Verificar si las distancias son similares
            std_dist = np.std(distances_to_center)
            if std_dist < 10.0:  # Threshold de variación
                formations.append("circle_formation")
                
        # Detectar clusters densos
        density_threshold = 20.0
        for i in range(n):
            neighbors = np.sum(dist_matrix[i] < density_threshold) - 1
            if neighbors >= 4:
                formations.append("dense_cluster")
                break
                
        return list(set(formations))
    
    def _calculate_clustering_coefficient(self, dist_matrix: np.ndarray) -> float:
        """
        Calcula el coeficiente de clustering espacial
        """
        n = len(dist_matrix)
        if n < 3:
            return 0.0
            
        threshold = 30.0  # Distancia para considerar vecinos
        clustering_coeffs = []
        
        for i in range(n):
            # Encontrar vecinos
            neighbors = np.where((dist_matrix[i] > 0) & (dist_matrix[i] < threshold))[0]
            k = len(neighbors)
            
            if k >= 2:
                # Contar conexiones entre vecinos
                connections = 0
                for j in range(k):
                    for l in range(j+1, k):
                        if dist_matrix[neighbors[j], neighbors[l]] < threshold:
                            connections += 1
                            
                # Coeficiente de clustering local
                max_connections = k * (k - 1) / 2
                clustering_coeffs.append(connections / max_connections)
                
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _calculate_spatial_entropy(self, positions: np.ndarray) -> float:
        """
        Calcula la entropía espacial de la distribución
        """
        # Discretizar el espacio en celdas
        grid_size = 50
        max_coord = np.max(positions) if len(positions) > 0 else 1000
        
        # Crear histograma 2D
        hist, _, _ = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=int(max_coord / grid_size),
            range=[[0, max_coord], [0, max_coord]]
        )
        
        # Normalizar
        hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        
        # Calcular entropía
        return entropy(hist_norm.flatten())
    
    def _calculate_movement_diversity(self, velocities: np.ndarray) -> float:
        """
        Calcula la diversidad de patrones de movimiento
        """
        if len(velocities) < 2:
            return 0.0
            
        # Calcular ángulos de movimiento
        angles = []
        for v in velocities:
            if np.linalg.norm(v) > 0.01:
                angle = np.arctan2(v[1], v[0])
                angles.append(angle)
                
        if not angles:
            return 0.0
            
        # Histograma circular
        hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        
        # Entropía del histograma
        hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        return entropy(hist_norm)
    
    def _calculate_information_entropy(self, organisms: List) -> float:
        """
        Calcula la entropía de información en el sistema
        """
        # Usar generaciones como proxy de información
        generations = [org.generation for org in organisms]
        
        if not generations:
            return 0.0
            
        # Calcular distribución de generaciones
        gen_counts = defaultdict(int)
        for gen in generations:
            gen_counts[gen] += 1
            
        # Normalizar
        total = len(generations)
        probs = [count/total for count in gen_counts.values()]
        
        return entropy(probs)
    
    def _detect_emergent_behaviors(self, analysis: Dict) -> List[str]:
        """
        Detecta comportamientos emergentes específicos basados en el análisis
        """
        behaviors = []
        
        # Comportamiento de enjambre
        if (analysis['movement_patterns']['alignment'] > 0.5 and
            analysis['movement_patterns']['coherence'] > 0.7):
            behaviors.append("swarm_behavior")
            
        # Segregación espacial
        if analysis['spatial_patterns']['formations'] and 'dense_cluster' in analysis['spatial_patterns']['formations']:
            if analysis['social_structure']['communities'] > 1:
                behaviors.append("spatial_segregation")
                
        # Migración colectiva
        if (analysis['movement_patterns']['coherence'] > 0.8 and
            analysis['movement_patterns']['avg_speed'] > 5.0):
            behaviors.append("collective_migration")
            
        # Comportamiento evasivo coordinado
        if (analysis['spatial_patterns']['dispersion'] > 100 and
            analysis['movement_patterns']['movement_diversity'] > 1.5):
            behaviors.append("coordinated_evasion")
            
        # Formación de grupos familiares
        if (analysis['social_structure']['communities'] > 0 and
            analysis['information_flow']['generation_spread'] > 2):
            behaviors.append("family_groups")
            
        return behaviors
    
    def track_emergence_over_time(self) -> Dict:
        """
        Rastrea la evolución de propiedades emergentes en el tiempo
        """
        if len(self.behavior_history) < 2:
            return {}
            
        # Extraer series temporales de métricas clave
        times = [b['time'] for b in self.behavior_history]
        
        metrics_over_time = {
            'population': [b['population_size'] for b in self.behavior_history],
            'dispersion': [b['spatial_patterns']['dispersion'] for b in self.behavior_history],
            'alignment': [b['movement_patterns']['alignment'] for b in self.behavior_history],
            'communities': [b['social_structure']['communities'] for b in self.behavior_history],
            'emergent_behaviors': [b['emergent_behaviors'] for b in self.behavior_history]
        }
        
        # Detectar transiciones
        transitions = self._detect_phase_transitions(metrics_over_time)
        
        return {
            'times': times,
            'metrics': metrics_over_time,
            'transitions': transitions,
            'complexity_trend': self._calculate_complexity_trend(metrics_over_time)
        }
    
    def _detect_phase_transitions(self, metrics: Dict) -> List[Dict]:
        """
        Detecta transiciones de fase en el comportamiento del sistema
        """
        transitions = []
        
        # Buscar cambios abruptos en métricas
        for metric_name, values in metrics.items():
            if metric_name == 'emergent_behaviors' or len(values) < 3:
                continue
                
            # Calcular derivada
            derivative = np.diff(values)
            
            # Buscar cambios significativos
            threshold = 2 * np.std(derivative) if np.std(derivative) > 0 else 0.1
            
            for i, d in enumerate(derivative):
                if abs(d) > threshold:
                    transitions.append({
                        'time_index': i+1,
                        'metric': metric_name,
                        'change': float(d),
                        'type': 'increase' if d > 0 else 'decrease'
                    })
                    
        return transitions
    
    def _calculate_complexity_trend(self, metrics: Dict) -> str:
        """
        Calcula la tendencia general de complejidad del sistema
        """
        # Combinar varias métricas en una medida de complejidad
        complexity_scores = []
        
        for i in range(len(metrics['population'])):
            score = 0
            
            # Diversidad de comportamientos
            if i < len(metrics['emergent_behaviors']):
                score += len(metrics['emergent_behaviors'][i]) * 0.3
                
            # Estructura social
            if i < len(metrics['communities']):
                score += metrics['communities'][i] * 0.2
                
            # Coherencia vs caos
            if i < len(metrics['alignment']):
                # Complejidad máxima en valores intermedios
                alignment = metrics['alignment'][i]
                score += (1 - abs(alignment - 0.5) * 2) * 0.5
                
            complexity_scores.append(score)
            
        # Calcular tendencia
        if len(complexity_scores) > 1:
            trend = np.polyfit(range(len(complexity_scores)), complexity_scores, 1)[0]
            
            if trend > 0.01:
                return "increasing_complexity"
            elif trend < -0.01:
                return "decreasing_complexity"
            else:
                return "stable_complexity"
        else:
            return "insufficient_data"