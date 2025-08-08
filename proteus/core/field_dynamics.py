"""
Dinámicas de campos - Cómo el entorno computa sin neuronas
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.ndimage import gaussian_filter


class FieldDynamics:
    """
    Gestiona los campos potenciales y sus dinámicas
    El campo ES la computación - no hay separación entre datos y procesamiento
    """
    
    def __init__(self, size: Tuple[int, int], dt: float = 0.01):
        self.size = size
        self.dt = dt
        
        # Campos múltiples para diferentes aspectos del entorno
        self.potential_field = np.zeros(size)  # Campo potencial principal
        self.light_field = np.zeros(size)      # Campo de luz (peligro)
        self.nutrient_field = np.random.random(size) * 0.1  # Nutrientes
        self.flow_field = np.zeros((*size, 2))  # Campo vectorial de flujo
        
        # Parámetros de dinámica
        self.viscosity = 0.8
        self.diffusion_rate = 0.1
        self.decay_rate = 0.01
        
    def update(self):
        """Actualiza todos los campos según sus dinámicas intrínsecas"""
        # Difusión del campo potencial
        self.potential_field = gaussian_filter(self.potential_field, sigma=1.0)
        
        # Decaimiento natural
        self.potential_field *= (1 - self.decay_rate)
        self.light_field *= 0.95  # La luz decae más rápido
        
        # Regeneración lenta de nutrientes
        self.nutrient_field += np.random.random(self.size) * 0.001
        self.nutrient_field = np.clip(self.nutrient_field, 0, 1)
        
        # Actualizar campo de flujo basado en gradientes
        self._update_flow_field()
        
    def _update_flow_field(self):
        """Calcula el campo de flujo a partir de los potenciales"""
        # Gradiente del potencial
        gy, gx = np.gradient(self.potential_field)
        
        # El flujo es perpendicular al gradiente (rotación 90°) + deriva
        self.flow_field[:, :, 0] = -gy * 0.5 + np.random.normal(0, 0.01, self.size)
        self.flow_field[:, :, 1] = gx * 0.5 + np.random.normal(0, 0.01, self.size)
        
        # Aplicar viscosidad
        self.flow_field *= self.viscosity
        
    def add_light_burst(self, position: Tuple[float, float], 
                       intensity: float, radius: float):
        """
        Añade un estallido de luz al campo (usado por depredadores)
        La luz es una perturbación masiva en el campo topológico
        """
        x, y = int(position[0]), int(position[1])
        
        # Crear máscara gaussiana para el estallido
        yy, xx = np.ogrid[:self.size[1], :self.size[0]]
        dist_sq = (xx - x)**2 + (yy - y)**2
        
        # Perturbación gaussiana
        light_burst = intensity * np.exp(-dist_sq / (2 * radius**2))
        
        # Actualizar campos
        self.light_field = np.maximum(self.light_field, light_burst)
        self.potential_field += light_burst * 10  # La luz altera el potencial
        
    def consume_nutrients(self, position: Tuple[float, float], amount: float):
        """Consume nutrientes en una posición"""
        x, y = int(position[0]), int(position[1])
        
        # Consumir en un área pequeña
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx = (x + dx) % self.size[0]
                ny = (y + dy) % self.size[1]
                
                consumed = min(self.nutrient_field[nx, ny], amount / 25)
                self.nutrient_field[nx, ny] -= consumed
                
    def get_local_field_state(self, position: Tuple[float, float], 
                             radius: int = 10) -> dict:
        """
        Obtiene el estado local del campo alrededor de una posición
        Esto es lo que "percibe" una criatura sin órganos sensoriales
        """
        x, y = int(position[0]), int(position[1])
        
        # Extraer parches locales de cada campo
        local_potential = self._extract_local_patch(self.potential_field, x, y, radius)
        local_light = self._extract_local_patch(self.light_field, x, y, radius)
        local_nutrients = self._extract_local_patch(self.nutrient_field, x, y, radius)
        local_flow = self._extract_local_patch_vector(self.flow_field, x, y, radius)
        
        # Calcular estadísticas locales
        return {
            'potential_gradient': np.gradient(local_potential),
            'light_intensity': np.mean(local_light),
            'light_gradient': np.gradient(local_light),
            'nutrient_density': np.sum(local_nutrients),
            'flow_average': np.mean(local_flow, axis=(0, 1)),
            'field_curvature': self._compute_local_curvature(local_potential)
        }
        
    def _extract_local_patch(self, field: np.ndarray, x: int, y: int, 
                           radius: int) -> np.ndarray:
        """Extrae un parche local del campo con manejo toroidal"""
        patch = np.zeros((2*radius+1, 2*radius+1))
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                nx = (x + i) % self.size[0]
                ny = (y + j) % self.size[1]
                patch[i+radius, j+radius] = field[nx, ny]
                
        return patch
        
    def _extract_local_patch_vector(self, field: np.ndarray, x: int, y: int, 
                                  radius: int) -> np.ndarray:
        """Extrae un parche local de un campo vectorial"""
        patch = np.zeros((2*radius+1, 2*radius+1, 2))
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                nx = (x + i) % self.size[0]
                ny = (y + j) % self.size[1]
                patch[i+radius, j+radius] = field[nx, ny]
                
        return patch
        
    def _compute_local_curvature(self, patch: np.ndarray) -> float:
        """
        Calcula la curvatura local del campo
        Una medida topológica fundamental
        """
        if patch.size < 9:
            return 0.0
            
        # Aproximación de la curvatura usando el laplaciano
        gy, gx = np.gradient(patch)
        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        
        # Curvatura gaussiana aproximada
        curvature = np.mean(gxx * gyy - gxy * gyx)
        
        return float(curvature)
    
    def compute_field_energy(self) -> float:
        """Calcula la energía total del sistema de campos"""
        potential_energy = np.sum(self.potential_field**2)
        light_energy = np.sum(self.light_field**2)
        flow_energy = np.sum(self.flow_field**2)
        
        return potential_energy + light_energy + flow_energy
    
    def apply_external_force(self, force_function: Callable):
        """
        Aplica una fuerza externa al campo
        La fuerza puede depender de la posición y el tiempo
        """
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                force = force_function(i, j, self.potential_field[i, j])
                self.potential_field[i, j] += force * self.dt