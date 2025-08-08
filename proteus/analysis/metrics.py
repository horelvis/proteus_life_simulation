"""
Métricas no-neuronales para evaluar el sistema
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial import ConvexHull
import networkx as nx
from collections import defaultdict


class NonNeuralMetrics:
    """
    Métricas para evaluar inteligencia sin referencias neuronales
    """
    
    def __init__(self):
        self.metric_history = defaultdict(list)
        
    def compute_all_metrics(self, world_state: Dict, organisms: List) -> Dict:
        """
        Calcula todas las métricas relevantes para el estado actual
        """
        metrics = {
            'survival_metrics': self._compute_survival_metrics(organisms),
            'topological_metrics': self._compute_topological_metrics(organisms),
            'emergence_metrics': self._compute_emergence_metrics(organisms),
            'information_metrics': self._compute_information_metrics(organisms),
            'adaptation_metrics': self._compute_adaptation_metrics(organisms),
            'system_metrics': self._compute_system_metrics(world_state)
        }
        
        # Guardar en historia
        for category, values in metrics.items():
            self.metric_history[category].append(values)
            
        return metrics
    
    def _compute_survival_metrics(self, organisms: List) -> Dict:
        """
        Métricas relacionadas con supervivencia
        """
        if not organisms:
            return {
                'mean_age': 0,
                'max_age': 0,
                'survival_diversity': 0,
                'generational_improvement': 0
            }
            
        ages = [org.age for org in organisms if org.alive]
        generations = [org.generation for org in organisms]
        
        # Agrupar por generación
        age_by_generation = defaultdict(list)
        for org in organisms:
            if org.alive:
                age_by_generation[org.generation].append(org.age)
                
        # Calcular mejora generacional
        gen_improvement = 0
        if len(age_by_generation) > 1:
            sorted_gens = sorted(age_by_generation.keys())
            for i in range(1, len(sorted_gens)):
                prev_gen = sorted_gens[i-1]
                curr_gen = sorted_gens[i]
                
                if age_by_generation[prev_gen] and age_by_generation[curr_gen]:
                    prev_avg = np.mean(age_by_generation[prev_gen])
                    curr_avg = np.mean(age_by_generation[curr_gen])
                    gen_improvement += (curr_avg - prev_avg) / (prev_avg + 1)
                    
        return {
            'mean_age': np.mean(ages) if ages else 0,
            'max_age': np.max(ages) if ages else 0,
            'survival_diversity': np.std(ages) if ages else 0,
            'generational_improvement': gen_improvement / max(1, len(age_by_generation)-1)
        }
    
    def _compute_topological_metrics(self, organisms: List) -> Dict:
        """
        Métricas basadas en propiedades topológicas
        """
        if not organisms:
            return {
                'path_complexity': 0,
                'exploration_efficiency': 0,
                'topological_diversity': 0,
                'movement_sophistication': 0
            }
            
        complexities = []
        efficiencies = []
        curvatures = []
        
        for org in organisms:
            if hasattr(org, 'state') and len(org.state.trajectory_history) > 10:
                # Complejidad del camino
                traj = np.array(org.state.trajectory_history)
                
                # Eficiencia de exploración
                total_distance = sum(np.linalg.norm(traj[i] - traj[i-1]) 
                                   for i in range(1, len(traj)))
                area_covered = self._compute_convex_hull_area(traj)
                
                if total_distance > 0:
                    efficiency = area_covered / total_distance
                    efficiencies.append(efficiency)
                    
                # Complejidad basada en curvatura
                if hasattr(org, 'get_topological_features'):
                    features = org.get_topological_features()
                    if features:
                        complexities.append(features.get('complexity', 0))
                        curvatures.append(features.get('curvature', 0))
                        
        return {
            'path_complexity': np.mean(complexities) if complexities else 0,
            'exploration_efficiency': np.mean(efficiencies) if efficiencies else 0,
            'topological_diversity': np.std(complexities) if complexities else 0,
            'movement_sophistication': np.mean(curvatures) if curvatures else 0
        }
    
    def _compute_emergence_metrics(self, organisms: List) -> Dict:
        """
        Métricas de comportamientos emergentes
        """
        if len(organisms) < 2:
            return {
                'collective_coordination': 0,
                'pattern_formation': 0,
                'behavioral_entropy': 0,
                'emergence_score': 0
            }
            
        # Coordinación colectiva
        positions = np.array([org.position for org in organisms])
        velocities = np.array([org.velocity for org in organisms])
        
        # Alineación de velocidades
        speeds = np.linalg.norm(velocities, axis=1)
        moving_mask = speeds > 0.1
        
        coordination = 0
        if np.any(moving_mask):
            normalized_vel = velocities[moving_mask] / speeds[moving_mask, np.newaxis]
            mean_direction = np.mean(normalized_vel, axis=0)
            coordination = np.linalg.norm(mean_direction)
            
        # Formación de patrones
        pattern_score = self._detect_spatial_patterns(positions)
        
        # Entropía comportamental
        behavior_entropy = self._compute_behavioral_entropy(organisms)
        
        # Score de emergencia combinado
        emergence_score = (coordination + pattern_score + behavior_entropy) / 3
        
        return {
            'collective_coordination': coordination,
            'pattern_formation': pattern_score,
            'behavioral_entropy': behavior_entropy,
            'emergence_score': emergence_score
        }
    
    def _compute_information_metrics(self, organisms: List) -> Dict:
        """
        Métricas de flujo de información topológica
        """
        if not organisms:
            return {
                'information_preservation': 0,
                'trait_heritability': 0,
                'evolutionary_rate': 0,
                'information_diversity': 0
            }
            
        # Preservación de información
        organisms_with_seeds = [org for org in organisms if org.seed is not None]
        preservation = len(organisms_with_seeds) / len(organisms) if organisms else 0
        
        # Heredabilidad de rasgos
        heritability = self._compute_trait_heritability(organisms)
        
        # Tasa evolutiva
        generations = [org.generation for org in organisms]
        evolutionary_rate = (max(generations) - min(generations)) / (len(organisms) + 1) if generations else 0
        
        # Diversidad de información
        if organisms_with_seeds:
            # Usar características de las semillas
            seed_features = []
            for org in organisms_with_seeds:
                if hasattr(org.seed, 'curvature_spectrum'):
                    seed_features.append(org.seed.curvature_spectrum)
                    
            if seed_features:
                # Calcular diversidad como varianza promedio
                seed_array = np.array(seed_features)
                info_diversity = np.mean(np.var(seed_array, axis=0))
            else:
                info_diversity = 0
        else:
            info_diversity = 0
            
        return {
            'information_preservation': preservation,
            'trait_heritability': heritability,
            'evolutionary_rate': evolutionary_rate,
            'information_diversity': info_diversity
        }
    
    def _compute_adaptation_metrics(self, organisms: List) -> Dict:
        """
        Métricas de adaptación al entorno
        """
        if not organisms:
            return {
                'light_avoidance_success': 0,
                'energy_efficiency': 0,
                'environmental_fit': 0,
                'adaptive_flexibility': 0
            }
            
        # Éxito en evitar luz
        light_avoidance = []
        for org in organisms:
            if org.alive and hasattr(org, 'light_exposures'):
                # Ratio de supervivencia vs exposiciones
                if org.age > 0:
                    avoidance_rate = 1 - (org.light_exposures / (org.age + 1))
                    light_avoidance.append(max(0, avoidance_rate))
                    
        # Eficiencia energética
        energy_efficiency = []
        for org in organisms:
            if hasattr(org, 'distance_traveled') and org.distance_traveled > 0:
                efficiency = org.age / (org.distance_traveled + 1)
                energy_efficiency.append(efficiency)
                
        # Ajuste ambiental
        env_fit = self._compute_environmental_fit(organisms)
        
        # Flexibilidad adaptativa
        flexibility = self._compute_adaptive_flexibility(organisms)
        
        return {
            'light_avoidance_success': np.mean(light_avoidance) if light_avoidance else 0,
            'energy_efficiency': np.mean(energy_efficiency) if energy_efficiency else 0,
            'environmental_fit': env_fit,
            'adaptive_flexibility': flexibility
        }
    
    def _compute_system_metrics(self, world_state: Dict) -> Dict:
        """
        Métricas del sistema completo
        """
        return {
            'system_energy': world_state.get('field_energy', 0),
            'population_stability': self._compute_population_stability(),
            'phase_coherence': self._compute_phase_coherence(),
            'kolmogorov_complexity': self._estimate_kolmogorov_complexity()
        }
    
    def _compute_convex_hull_area(self, points: np.ndarray) -> float:
        """
        Calcula el área del casco convexo de un conjunto de puntos
        """
        if len(points) < 3:
            return 0.0
            
        try:
            hull = ConvexHull(points)
            return hull.volume  # En 2D, volume es el área
        except:
            return 0.0
            
    def _detect_spatial_patterns(self, positions: np.ndarray) -> float:
        """
        Detecta y puntúa patrones espaciales
        """
        if len(positions) < 3:
            return 0.0
            
        # Calcular distribución radial
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        
        # Verificar uniformidad de distribución
        if distances:
            # Normalizar
            max_dist = max(distances) + 1
            normalized_dists = [d / max_dist for d in distances]
            
            # Calcular entropía de la distribución radial
            hist, _ = np.histogram(normalized_dists, bins=10)
            if np.sum(hist) > 0:
                hist_norm = hist / np.sum(hist)
                pattern_entropy = entropy(hist_norm)
                
                # Normalizar entropía
                max_entropy = np.log(10)
                pattern_score = pattern_entropy / max_entropy
            else:
                pattern_score = 0
        else:
            pattern_score = 0
            
        return pattern_score
    
    def _compute_behavioral_entropy(self, organisms: List) -> float:
        """
        Calcula la entropía del comportamiento colectivo
        """
        if not organisms:
            return 0.0
            
        # Discretizar comportamientos
        behaviors = []
        
        for org in organisms:
            # Crear vector de comportamiento
            behavior = []
            
            # Velocidad
            speed = np.linalg.norm(org.velocity)
            if speed < 1:
                behavior.append(0)  # Quieto
            elif speed < 5:
                behavior.append(1)  # Lento
            else:
                behavior.append(2)  # Rápido
                
            # Dirección
            if speed > 0.1:
                angle = np.arctan2(org.velocity[1], org.velocity[0])
                behavior.append(int((angle + np.pi) / (np.pi / 4)))  # 8 direcciones
            else:
                behavior.append(-1)
                
            behaviors.append(tuple(behavior))
            
        # Calcular distribución
        unique_behaviors = list(set(behaviors))
        if len(unique_behaviors) > 1:
            counts = [behaviors.count(b) for b in unique_behaviors]
            probs = [c / len(behaviors) for c in counts]
            return entropy(probs)
        else:
            return 0.0
            
    def _compute_trait_heritability(self, organisms: List) -> float:
        """
        Calcula qué tan bien se heredan los rasgos
        """
        parent_child_pairs = []
        
        for org in organisms:
            if org.seed and hasattr(org, 'get_topological_features'):
                # Comparar con características de la semilla parental
                child_features = org.get_topological_features()
                
                if child_features and hasattr(org.seed, 'path_homology'):
                    # Correlación simplificada
                    parent_persistence = org.seed.path_homology.get('persistence', 0)
                    child_persistence = child_features.get('persistence', 0)
                    
                    if parent_persistence > 0 and child_persistence > 0:
                        correlation = 1 - abs(parent_persistence - child_persistence) / max(parent_persistence, child_persistence)
                        parent_child_pairs.append(correlation)
                        
        return np.mean(parent_child_pairs) if parent_child_pairs else 0
    
    def _compute_environmental_fit(self, organisms: List) -> float:
        """
        Calcula qué tan bien adaptados están al entorno
        """
        if not organisms:
            return 0.0
            
        fit_scores = []
        
        for org in organisms:
            if org.alive:
                # Factores de ajuste
                age_factor = min(1.0, org.age / 100)  # Normalizar edad
                energy_factor = org.energy  # Ya está en [0, 1]
                
                # Penalización por exposiciones a luz
                light_penalty = 1.0 / (1 + org.light_exposures * 0.1)
                
                fit_score = (age_factor + energy_factor + light_penalty) / 3
                fit_scores.append(fit_score)
                
        return np.mean(fit_scores) if fit_scores else 0
    
    def _compute_adaptive_flexibility(self, organisms: List) -> float:
        """
        Mide la flexibilidad adaptativa del comportamiento
        """
        if not organisms:
            return 0.0
            
        flexibility_scores = []
        
        for org in organisms:
            if hasattr(org, 'state') and len(org.state.trajectory_history) > 20:
                # Analizar cambios en patrones de movimiento
                traj = np.array(org.state.trajectory_history)
                
                # Dividir trayectoria en segmentos
                segment_length = 10
                n_segments = len(traj) // segment_length
                
                if n_segments >= 2:
                    segment_patterns = []
                    
                    for i in range(n_segments):
                        segment = traj[i*segment_length:(i+1)*segment_length]
                        
                        # Calcular patrón del segmento
                        if len(segment) > 1:
                            velocities = np.diff(segment, axis=0)
                            avg_velocity = np.mean(velocities, axis=0)
                            avg_speed = np.linalg.norm(avg_velocity)
                            segment_patterns.append(avg_speed)
                            
                    # Variabilidad entre segmentos
                    if segment_patterns:
                        flexibility = np.std(segment_patterns) / (np.mean(segment_patterns) + 0.01)
                        flexibility_scores.append(min(1.0, flexibility))
                        
        return np.mean(flexibility_scores) if flexibility_scores else 0
    
    def _compute_population_stability(self) -> float:
        """
        Calcula la estabilidad de la población en el tiempo
        """
        if 'survival_metrics' not in self.metric_history or len(self.metric_history['survival_metrics']) < 2:
            return 0.0
            
        # Extraer historia de edades medias
        mean_ages = [m['mean_age'] for m in self.metric_history['survival_metrics']]
        
        if len(mean_ages) > 1:
            # Calcular variabilidad
            stability = 1.0 / (1.0 + np.std(mean_ages) / (np.mean(mean_ages) + 0.01))
            return stability
        else:
            return 0.0
            
    def _compute_phase_coherence(self) -> float:
        """
        Mide la coherencia de fase del sistema
        """
        if 'emergence_metrics' not in self.metric_history or len(self.metric_history['emergence_metrics']) < 2:
            return 0.0
            
        # Usar coordinación colectiva como proxy
        coordinations = [m['collective_coordination'] for m in self.metric_history['emergence_metrics']]
        
        if len(coordinations) > 1:
            # Coherencia como 1 - variabilidad
            coherence = 1.0 - np.std(coordinations)
            return max(0, coherence)
        else:
            return 0.0
            
    def _estimate_kolmogorov_complexity(self) -> float:
        """
        Estima la complejidad de Kolmogorov del sistema
        """
        # Usar la longitud de la descripción comprimida de la historia
        # Como proxy, usar entropía de múltiples métricas
        
        if not self.metric_history:
            return 0.0
            
        # Concatenar todas las métricas recientes
        all_values = []
        for category_history in self.metric_history.values():
            if category_history:
                latest = category_history[-1]
                if isinstance(latest, dict):
                    all_values.extend([v for v in latest.values() if isinstance(v, (int, float))])
                    
        if all_values:
            # Discretizar valores
            discretized = [int(v * 100) for v in all_values]
            
            # Calcular entropía como proxy de complejidad
            unique_values = set(discretized)
            if len(unique_values) > 1:
                counts = [discretized.count(v) for v in unique_values]
                probs = [c / len(discretized) for c in counts]
                complexity = entropy(probs)
                
                # Normalizar
                max_entropy = np.log(len(discretized))
                return complexity / max_entropy if max_entropy > 0 else 0
                
        return 0.0
    
    def generate_report(self) -> str:
        """
        Genera un reporte de todas las métricas
        """
        if not self.metric_history:
            return "No metrics available yet."
            
        report = "=== NON-NEURAL INTELLIGENCE METRICS REPORT ===\n\n"
        
        # Obtener métricas más recientes
        for category, history in self.metric_history.items():
            if history:
                latest = history[-1]
                report += f"{category.upper()}:\n"
                
                for metric, value in latest.items():
                    if isinstance(value, float):
                        report += f"  {metric}: {value:.3f}\n"
                    else:
                        report += f"  {metric}: {value}\n"
                        
                report += "\n"
                
        # Análisis de tendencias
        report += "TRENDS:\n"
        
        # Tendencia de supervivencia
        if 'survival_metrics' in self.metric_history and len(self.metric_history['survival_metrics']) > 1:
            mean_ages = [m['mean_age'] for m in self.metric_history['survival_metrics']]
            trend = "increasing" if mean_ages[-1] > mean_ages[0] else "decreasing"
            report += f"  Survival trend: {trend}\n"
            
        # Tendencia de emergencia
        if 'emergence_metrics' in self.metric_history and len(self.metric_history['emergence_metrics']) > 1:
            emergence_scores = [m['emergence_score'] for m in self.metric_history['emergence_metrics']]
            trend = "increasing" if emergence_scores[-1] > emergence_scores[0] else "decreasing"
            report += f"  Emergence trend: {trend}\n"
            
        return report