"""
Motor de dinámicas topológicas - El corazón de TopoLife
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class TopologicalState:
    """Estado topológico de un agente en el espacio"""
    position: np.ndarray
    velocity: np.ndarray
    trajectory_history: List[np.ndarray]
    field_memory: Optional[np.ndarray] = None
    
    def add_position(self, pos: np.ndarray):
        """Agrega una posición a la historia de trayectorias"""
        self.trajectory_history.append(pos.copy())
        if len(self.trajectory_history) > 1000:
            self.trajectory_history.pop(0)


class TopologyEngine:
    """Motor principal para computación topológica"""
    
    def __init__(self, world_size: Tuple[int, int], topology_type: str = "toroidal"):
        self.world_size = world_size  # (width, height)
        self.topology_type = topology_type
        # NumPy arrays use (height, width) order, so we need to reverse
        self.field = np.zeros((world_size[1], world_size[0]))
        self.time = 0
        
    def compute_field_gradient(self, position: np.ndarray) -> np.ndarray:
        """Calcula el gradiente del campo topológico en una posición"""
        x, y = int(position[0]), int(position[1])
        
        # Manejar bordes según la topología
        if self.topology_type == "toroidal":
            x = x % self.world_size[0]
            y = y % self.world_size[1]
            
            # Calcular gradiente con diferencias finitas
            # Nota: field[y, x] porque NumPy usa (row, col) order
            dx = (self.field[y, (x+1)%self.world_size[0]] - 
                  self.field[y, (x-1)%self.world_size[0]]) / 2.0
            dy = (self.field[(y+1)%self.world_size[1], x] - 
                  self.field[(y-1)%self.world_size[1], x]) / 2.0
        else:
            # Topología euclidiana con bordes
            x = np.clip(x, 1, self.world_size[0]-2)
            y = np.clip(y, 1, self.world_size[1]-2)
            
            # field[y, x] porque NumPy usa (row, col) order
            dx = (self.field[y, x+1] - self.field[y, x-1]) / 2.0
            dy = (self.field[y+1, x] - self.field[y-1, x]) / 2.0
            
        return np.array([dx, dy])
    
    def apply_toroidal_boundary(self, position: np.ndarray) -> np.ndarray:
        """Aplica condiciones de frontera toroidales"""
        if self.topology_type == "toroidal":
            return np.array([
                position[0] % self.world_size[0],
                position[1] % self.world_size[1]
            ])
        else:
            return np.clip(position, [0, 0], 
                         [self.world_size[0]-1, self.world_size[1]-1])
    
    def compute_topological_flow(self, state: TopologicalState, 
                                inherited_signal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula el flujo topológico que determina el movimiento
        Sin neuronas, sin pesos - pura dinámica topológica
        """
        # Gradiente del campo potencial
        gradient = self.compute_field_gradient(state.position)
        
        # Ruido browniano
        noise = np.random.normal(0, 0.1, size=2)
        
        # Señal heredada modula la dinámica
        if inherited_signal is not None:
            modulation = inherited_signal * 0.3
        else:
            modulation = np.zeros(2)
        
        # La ecuación fundamental: flujo = -gradiente + ruido + herencia
        flow = -gradient + noise + modulation
        
        # Viscosidad del medio
        flow *= 0.8
        
        return flow
    
    def update_field(self, perturbations: List[Dict]):
        """Actualiza el campo topológico con perturbaciones"""
        # Decaimiento natural del campo
        self.field *= 0.99
        
        # Aplicar perturbaciones (ej: luz de depredadores)
        for pert in perturbations:
            pos = pert['position']
            amplitude = pert['amplitude']
            radius = pert['radius']
            
            # Crear máscara circular para la perturbación
            y, x = np.ogrid[:self.world_size[1], :self.world_size[0]]
            
            if self.topology_type == "toroidal":
                # Distancia toroidal
                dx = np.minimum(np.abs(x - pos[0]), 
                               self.world_size[0] - np.abs(x - pos[0]))
                dy = np.minimum(np.abs(y - pos[1]), 
                               self.world_size[1] - np.abs(y - pos[1]))
                dist_sq = dx**2 + dy**2
            else:
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
            
            # Aplicar perturbación gaussiana
            mask = np.exp(-dist_sq / (2 * radius**2))
            self.field += amplitude * mask
        
        self.time += 1
    
    @staticmethod
    def extract_topological_features(trajectory: List[np.ndarray]) -> Dict:
        """
        Extrae características topológicas de una trayectoria
        Estas son las "memorias" que se heredan
        """
        if len(trajectory) < 3:
            return {
                'curvature': 0.0,
                'winding_number': 0,
                'persistence': 0.0,
                'complexity': 0.0
            }
        
        trajectory_array = np.array(trajectory)
        
        # Curvatura promedio del camino
        if len(trajectory) >= 3:
            velocities = np.diff(trajectory_array, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            curvatures = []
            for i in range(len(accelerations)):
                v = velocities[i]
                a = accelerations[i]
                v_norm = np.linalg.norm(v)
                if v_norm > 0.01:
                    curvature = np.linalg.norm(np.cross(v, a)) / (v_norm**3)
                    curvatures.append(curvature)
            
            avg_curvature = np.mean(curvatures) if curvatures else 0.0
        else:
            avg_curvature = 0.0
        
        # Número de vueltas (winding number simplificado)
        angles = []
        for i in range(1, len(trajectory_array)):
            v = trajectory_array[i] - trajectory_array[i-1]
            if np.linalg.norm(v) > 0.01:
                angle = np.arctan2(v[1], v[0])
                angles.append(angle)
        
        if len(angles) > 1:
            angle_changes = np.diff(np.unwrap(angles))
            total_rotation = np.sum(angle_changes)
            winding_number = int(total_rotation / (2 * np.pi))
        else:
            winding_number = 0
        
        # Persistencia (qué tan lejos viajó vs distancia directa)
        if len(trajectory_array) > 1:
            total_distance = np.sum([np.linalg.norm(trajectory_array[i] - trajectory_array[i-1]) 
                                   for i in range(1, len(trajectory_array))])
            direct_distance = np.linalg.norm(trajectory_array[-1] - trajectory_array[0])
            persistence = direct_distance / (total_distance + 0.01)
        else:
            persistence = 0.0
        
        # Complejidad (entropía aproximada del camino)
        if len(trajectory_array) > 10:
            # Discretizar el espacio
            grid_size = 20
            discretized = (trajectory_array / grid_size).astype(int)
            unique_cells = len(np.unique(discretized, axis=0))
            complexity = unique_cells / len(trajectory_array)
        else:
            complexity = 0.0
        
        return {
            'curvature': avg_curvature,
            'winding_number': winding_number,
            'persistence': persistence,
            'complexity': complexity
        }