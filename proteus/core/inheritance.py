"""
Sistema de herencia topológica - Transmisión sin genes
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from scipy.fft import fft, ifft
from scipy.signal import find_peaks


@dataclass
class TopologicalSeed:
    """
    Una semilla topológica - NO es ADN, es la forma del viaje
    Contiene la esencia topológica de la experiencia del padre
    """
    # Características del camino recorrido
    path_homology: Dict[str, float]
    curvature_spectrum: np.ndarray
    avoided_zones: List[Tuple[float, float]]
    survival_time: float
    field_interactions: Dict[str, np.ndarray]
    
    # Metadatos
    generation: int
    parent_id: str
    creation_time: float
    
    def to_json(self) -> str:
        """Serializa la semilla para almacenamiento"""
        data = {
            'path_homology': self.path_homology,
            'curvature_spectrum': self.curvature_spectrum.tolist(),
            'avoided_zones': self.avoided_zones,
            'survival_time': self.survival_time,
            'field_interactions': {k: v.tolist() for k, v in self.field_interactions.items()},
            'generation': self.generation,
            'parent_id': self.parent_id,
            'creation_time': self.creation_time
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TopologicalSeed':
        """Deserializa una semilla desde JSON"""
        data = json.loads(json_str)
        data['curvature_spectrum'] = np.array(data['curvature_spectrum'])
        data['field_interactions'] = {k: np.array(v) for k, v in data['field_interactions'].items()}
        return cls(**data)


class TopologicalInheritance:
    """
    Sistema de herencia no-genética basado en topología
    La información se transmite a través de patrones geométricos
    """
    
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        
    def extract_seed(self, trajectory: List[np.ndarray], 
                    field_history: List[Dict],
                    survival_time: float,
                    generation: int,
                    parent_id: str) -> TopologicalSeed:
        """
        Extrae una semilla topológica de la experiencia vivida
        """
        # Convertir trayectoria a array
        traj_array = np.array(trajectory)
        
        # 1. Homología del camino
        homology = self._compute_path_homology(traj_array)
        
        # 2. Espectro de curvatura
        curvature_spectrum = self._compute_curvature_spectrum(traj_array)
        
        # 3. Zonas evitadas (donde hubo alta luz)
        avoided_zones = self._extract_avoided_zones(traj_array, field_history)
        
        # 4. Interacciones con el campo
        field_interactions = self._compute_field_interactions(field_history)
        
        return TopologicalSeed(
            path_homology=homology,
            curvature_spectrum=curvature_spectrum,
            avoided_zones=avoided_zones,
            survival_time=survival_time,
            field_interactions=field_interactions,
            generation=generation,
            parent_id=parent_id,
            creation_time=float(len(trajectory))
        )
    
    def _compute_path_homology(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Calcula características homológicas del camino
        Estas son invariantes topológicos que capturan la "forma" del viaje
        """
        if len(trajectory) < 3:
            return {'loops': 0, 'persistence': 0, 'dimension': 0}
        
        # Detectar loops (cruces consigo mismo)
        loops = 0
        for i in range(len(trajectory) - 10):
            for j in range(i + 10, len(trajectory)):
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                if dist < 5.0:  # Threshold para considerar un loop
                    loops += 1
        
        # Persistencia (qué tan "directo" fue el camino)
        total_distance = np.sum([np.linalg.norm(trajectory[i] - trajectory[i-1]) 
                               for i in range(1, len(trajectory))])
        direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        persistence = direct_distance / (total_distance + 0.01)
        
        # Dimensión fractal aproximada
        scales = [2, 4, 8, 16]
        counts = []
        for scale in scales:
            grid_traj = (trajectory / scale).astype(int)
            unique_cells = len(np.unique(grid_traj, axis=0))
            counts.append(unique_cells)
        
        # Estimar dimensión usando box-counting
        if len(counts) > 1:
            log_counts = np.log(counts)
            log_scales = np.log(scales)
            dimension = -np.polyfit(log_scales, log_counts, 1)[0]
        else:
            dimension = 1.0
        
        return {
            'loops': float(loops),
            'persistence': persistence,
            'dimension': dimension
        }
    
    def _compute_curvature_spectrum(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Calcula el espectro de frecuencias de la curvatura
        Esto captura los "ritmos" del movimiento
        """
        if len(trajectory) < 10:
            return np.zeros(8)
        
        # Calcular curvatura en cada punto
        curvatures = []
        for i in range(2, len(trajectory)):
            v1 = trajectory[i-1] - trajectory[i-2]
            v2 = trajectory[i] - trajectory[i-1]
            
            # Ángulo entre vectores consecutivos
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0.01 and norm2 > 0.01:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
            else:
                curvatures.append(0)
        
        # Transformada de Fourier para obtener espectro
        if len(curvatures) > 8:
            curvature_fft = fft(curvatures)
            spectrum = np.abs(curvature_fft[:8])  # Primeras 8 frecuencias
        else:
            spectrum = np.zeros(8)
            spectrum[:len(curvatures)] = curvatures
        
        return spectrum / (np.max(spectrum) + 0.01)  # Normalizar
    
    def _extract_avoided_zones(self, trajectory: np.ndarray, 
                              field_history: List[Dict]) -> List[Tuple[float, float]]:
        """
        Identifica zonas que fueron consistentemente evitadas
        Estas son "memorias de peligro"
        """
        avoided = []
        
        # Analizar dónde hubo alta luz y el organismo se alejó
        for i, field_state in enumerate(field_history):
            if 'light_intensity' in field_state and field_state['light_intensity'] > 0.5:
                # Esta zona tenía peligro
                if i < len(trajectory):
                    avoided.append((float(trajectory[i][0]), float(trajectory[i][1])))
        
        # Clustear zonas cercanas
        if len(avoided) > 0:
            clustered = []
            for zone in avoided:
                added = False
                for cluster in clustered:
                    dist = np.sqrt((zone[0] - cluster[0])**2 + (zone[1] - cluster[1])**2)
                    if dist < 20:  # Threshold de clustering
                        added = True
                        break
                if not added:
                    clustered.append(zone)
            
            return clustered[:10]  # Máximo 10 zonas
        
        return []
    
    def _compute_field_interactions(self, field_history: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Calcula cómo el organismo interactuó con diferentes aspectos del campo
        """
        interactions = {
            'light_response': np.zeros(4),
            'flow_alignment': np.zeros(4),
            'nutrient_seeking': np.zeros(4)
        }
        
        if not field_history:
            return interactions
        
        # Respuesta a la luz
        light_intensities = [f.get('light_intensity', 0) for f in field_history]
        if light_intensities:
            # Estadísticas de exposición a luz
            interactions['light_response'][0] = np.mean(light_intensities)
            interactions['light_response'][1] = np.std(light_intensities)
            interactions['light_response'][2] = np.max(light_intensities)
            interactions['light_response'][3] = len([l for l in light_intensities if l > 0.5])
        
        # Alineación con el flujo
        flow_alignments = []
        for f in field_history:
            if 'flow_average' in f:
                flow_alignments.append(np.linalg.norm(f['flow_average']))
        
        if flow_alignments:
            interactions['flow_alignment'][0] = np.mean(flow_alignments)
            interactions['flow_alignment'][1] = np.std(flow_alignments)
            interactions['flow_alignment'][2] = np.max(flow_alignments)
            interactions['flow_alignment'][3] = np.min(flow_alignments)
        
        # Búsqueda de nutrientes
        nutrient_densities = [f.get('nutrient_density', 0) for f in field_history]
        if nutrient_densities:
            interactions['nutrient_seeking'][0] = np.mean(nutrient_densities)
            interactions['nutrient_seeking'][1] = np.std(nutrient_densities)
            interactions['nutrient_seeking'][2] = np.max(nutrient_densities)
            interactions['nutrient_seeking'][3] = len([n for n in nutrient_densities if n > 0.1])
        
        return interactions
    
    def apply_inheritance(self, seed: Optional[TopologicalSeed]) -> Dict[str, np.ndarray]:
        """
        Convierte una semilla topológica en modulaciones para el nuevo organismo
        """
        if seed is None:
            # Sin herencia - comportamiento base
            return {
                'movement_bias': np.zeros(2),
                'field_sensitivity': np.ones(4),
                'danger_memory': np.array([])
            }
        
        # La semilla modula el comportamiento
        modulations = {}
        
        # 1. Sesgo de movimiento basado en el espectro de curvatura
        # Las frecuencias dominantes en el padre influyen el movimiento del hijo
        dominant_freq = np.argmax(seed.curvature_spectrum)
        movement_bias = np.array([
            np.sin(dominant_freq * 0.5),
            np.cos(dominant_freq * 0.5)
        ]) * 0.3
        
        # Mutación
        if np.random.random() < self.mutation_rate:
            movement_bias += np.random.normal(0, 0.1, size=2)
        
        modulations['movement_bias'] = movement_bias
        
        # 2. Sensibilidad al campo basada en homología
        field_sensitivity = np.ones(4)
        field_sensitivity[0] = 1.0 + seed.path_homology['persistence'] * 0.5
        field_sensitivity[1] = 1.0 + seed.path_homology['loops'] * 0.01
        field_sensitivity[2] = 1.0 + seed.path_homology['dimension'] * 0.2
        field_sensitivity[3] = 1.0 + seed.survival_time * 0.001
        
        modulations['field_sensitivity'] = field_sensitivity
        
        # 3. Memoria de peligro (zonas a evitar)
        danger_memory = np.array(seed.avoided_zones) if seed.avoided_zones else np.array([])
        modulations['danger_memory'] = danger_memory
        
        # 4. Respuestas aprendidas
        modulations['light_avoidance'] = seed.field_interactions['light_response'][0] * 2.0
        modulations['flow_following'] = seed.field_interactions['flow_alignment'][0]
        modulations['nutrient_attraction'] = seed.field_interactions['nutrient_seeking'][0]
        
        return modulations
    
    def crossover(self, seed1: TopologicalSeed, seed2: TopologicalSeed) -> TopologicalSeed:
        """
        Combina dos semillas topológicas
        No es reproducción sexual - es mezcla de experiencias topológicas
        """
        # Interpolar homologías
        new_homology = {}
        for key in seed1.path_homology:
            alpha = np.random.random()
            new_homology[key] = (alpha * seed1.path_homology[key] + 
                               (1-alpha) * seed2.path_homology[key])
        
        # Combinar espectros de curvatura
        new_spectrum = (seed1.curvature_spectrum + seed2.curvature_spectrum) / 2
        
        # Unir zonas evitadas
        new_avoided = list(set(seed1.avoided_zones + seed2.avoided_zones))[:10]
        
        # Promediar tiempo de supervivencia
        new_survival = (seed1.survival_time + seed2.survival_time) / 2
        
        # Combinar interacciones de campo
        new_interactions = {}
        for key in seed1.field_interactions:
            new_interactions[key] = (seed1.field_interactions[key] + 
                                   seed2.field_interactions[key]) / 2
        
        return TopologicalSeed(
            path_homology=new_homology,
            curvature_spectrum=new_spectrum,
            avoided_zones=new_avoided,
            survival_time=new_survival,
            field_interactions=new_interactions,
            generation=max(seed1.generation, seed2.generation) + 1,
            parent_id=f"{seed1.parent_id}x{seed2.parent_id}",
            creation_time=(seed1.creation_time + seed2.creation_time) / 2
        )