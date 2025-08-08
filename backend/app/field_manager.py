"""
Gestor de campos ambientales para la simulación web
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class FieldManager:
    """Gestiona los campos de luz, nutrientes y perturbaciones"""
    
    def __init__(self, world_size: Tuple[int, int], physics_params: Dict[str, float]):
        self.width, self.height = world_size
        self.physics_params = physics_params
        
        # Resolución reducida para rendimiento web
        self.field_resolution = 50
        self.grid_width = self.field_resolution
        self.grid_height = int(self.field_resolution * self.height / self.width)
        
        # Campos
        self.light_field = np.zeros((self.grid_height, self.grid_width))
        self.nutrient_field = np.ones((self.grid_height, self.grid_width)) * 0.5
        self.perturbation_field = np.zeros((self.grid_height, self.grid_width))
        
        # Parámetros de actualización
        self.light_decay = physics_params.get("light_decay", 0.95)
        self.nutrient_regen = physics_params.get("nutrient_regeneration", 0.001)
        self.diffusion_rate = physics_params.get("diffusion_rate", 0.1)
        
        # Cache de gradientes
        self.light_gradient_x = None
        self.light_gradient_y = None
        self.nutrient_gradient_x = None
        self.nutrient_gradient_y = None
        self._gradient_updated = False
    
    def update(self, predators: List):
        """Actualiza todos los campos"""
        # Resetear campo de luz
        self.light_field *= self.light_decay
        
        # Añadir luz de depredadores atacando
        for predator in predators:
            if predator.is_attacking:
                self._add_light_source(
                    predator.x, predator.y, 
                    predator.light_radius, 
                    intensity=1.0
                )
        
        # Difundir luz
        self._diffuse_field(self.light_field, self.diffusion_rate)
        
        # Regenerar nutrientes (más en ciertas zonas)
        base_regen = self.nutrient_field + self.nutrient_regen
        
        # Zonas de alta regeneración (correspondientes a las zonas seguras)
        safe_zones = [
            {"x": 100, "y": 100, "radius": 80},
            {"x": self.width - 100, "y": self.height - 100, "radius": 80},
            {"x": self.width / 2, "y": 50, "radius": 60}
        ]
        
        for zone in safe_zones:
            # Convertir a coordenadas de grilla
            grid_x = int(zone["x"] * self.grid_width / self.width)
            grid_y = int(zone["y"] * self.grid_height / self.height)
            grid_radius = int(zone["radius"] * self.grid_width / self.width)
            
            # Crear máscara circular para zona segura
            y_indices, x_indices = np.ogrid[:self.grid_height, :self.grid_width]
            distances = np.sqrt((x_indices - grid_x)**2 + (y_indices - grid_y)**2)
            mask = distances <= grid_radius
            
            # Mayor regeneración en zonas seguras
            base_regen[mask] += self.nutrient_regen * 3
        
        self.nutrient_field = np.clip(base_regen, 0, 1)
        
        # Difundir nutrientes
        self._diffuse_field(self.nutrient_field, self.diffusion_rate * 0.5)
        
        # Actualizar perturbaciones (decaen con el tiempo)
        self.perturbation_field *= 0.98
        
        # Calcular gradientes
        self._update_gradients()
    
    def _add_light_source(self, x: float, y: float, radius: float, intensity: float):
        """Añade una fuente de luz al campo"""
        # Convertir coordenadas del mundo a la grilla
        grid_x = int(x * self.grid_width / self.width)
        grid_y = int(y * self.grid_height / self.height)
        grid_radius = int(radius * self.grid_width / self.width)
        
        # Crear máscara circular
        y_indices, x_indices = np.ogrid[:self.grid_height, :self.grid_width]
        distances = np.sqrt(
            (x_indices - grid_x)**2 + (y_indices - grid_y)**2
        )
        
        # Aplicar luz con falloff
        mask = distances <= grid_radius
        falloff = 1 - (distances / (grid_radius + 1))
        self.light_field[mask] += intensity * falloff[mask]
        self.light_field = np.clip(self.light_field, 0, 2)
    
    def _diffuse_field(self, field: np.ndarray, rate: float):
        """Aplica difusión a un campo"""
        # Difusión simple usando convolución
        kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05]
        ])
        
        # Padding para bordes
        padded = np.pad(field, 1, mode='edge')
        
        # Aplicar convolución manualmente (más rápido que scipy para grillas pequeñas)
        new_field = np.zeros_like(field)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                new_field[i, j] = np.sum(
                    padded[i:i+3, j:j+3] * kernel
                )
        
        # Mezclar con campo original
        field[:] = field * (1 - rate) + new_field * rate
    
    def _update_gradients(self):
        """Calcula gradientes de los campos"""
        # Gradientes de luz
        self.light_gradient_y, self.light_gradient_x = np.gradient(self.light_field)
        
        # Gradientes de nutrientes
        self.nutrient_gradient_y, self.nutrient_gradient_x = np.gradient(self.nutrient_field)
        
        self._gradient_updated = True
    
    def get_local_state(self, x: float, y: float) -> Dict:
        """Obtiene el estado local del campo en una posición"""
        # Convertir a coordenadas de grilla
        grid_x = int(x * self.grid_width / self.width)
        grid_y = int(y * self.grid_height / self.height)
        
        # Clamp a los límites
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        # Obtener valores locales
        light_level = self.light_field[grid_y, grid_x]
        nutrient_level = self.nutrient_field[grid_y, grid_x]
        perturbation = self.perturbation_field[grid_y, grid_x]
        
        # Obtener gradientes locales
        light_grad_x = self.light_gradient_x[grid_y, grid_x] if self.light_gradient_x is not None else 0
        light_grad_y = self.light_gradient_y[grid_y, grid_x] if self.light_gradient_y is not None else 0
        nutrient_grad_x = self.nutrient_gradient_x[grid_y, grid_x] if self.nutrient_gradient_x is not None else 0
        nutrient_grad_y = self.nutrient_gradient_y[grid_y, grid_x] if self.nutrient_gradient_y is not None else 0
        
        # Escalar gradientes a coordenadas del mundo
        scale_x = self.width / self.grid_width
        scale_y = self.height / self.grid_height
        
        return {
            "light_level": light_level,
            "nutrient_level": nutrient_level,
            "perturbation": perturbation,
            "light_gradient": np.array([light_grad_x * scale_x, light_grad_y * scale_y]),
            "nutrient_gradient": np.array([nutrient_grad_x * scale_x, nutrient_grad_y * scale_y])
        }
    
    def consume_nutrients(self, x: float, y: float, amount: float):
        """Consume nutrientes en una posición"""
        grid_x = int(x * self.grid_width / self.width)
        grid_y = int(y * self.grid_height / self.height)
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.nutrient_field[grid_y, grid_x] -= amount
            self.nutrient_field[grid_y, grid_x] = max(0, self.nutrient_field[grid_y, grid_x])
    
    def add_perturbation(self, x: float, y: float, strength: float):
        """Añade una perturbación local"""
        grid_x = int(x * self.grid_width / self.width)
        grid_y = int(y * self.grid_height / self.height)
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.perturbation_field[grid_y, grid_x] += strength
            self.perturbation_field = np.clip(self.perturbation_field, -1, 1)
    
    def get_light_field_preview(self) -> List[List[float]]:
        """Obtiene una versión simplificada del campo de luz para visualización"""
        # Reducir resolución si es necesario
        preview_size = 25
        if self.grid_width > preview_size:
            # Downsample
            step_x = self.grid_width // preview_size
            step_y = self.grid_height // preview_size
            preview = self.light_field[::step_y, ::step_x]
        else:
            preview = self.light_field
        
        return preview.tolist()
    
    def get_nutrient_field_preview(self) -> List[List[float]]:
        """Obtiene una versión simplificada del campo de nutrientes"""
        preview_size = 25
        if self.grid_width > preview_size:
            step_x = self.grid_width // preview_size
            step_y = self.grid_height // preview_size
            preview = self.nutrient_field[::step_y, ::step_x]
        else:
            preview = self.nutrient_field
        
        return preview.tolist()
    
    def get_field_stats(self) -> Dict:
        """Obtiene estadísticas de los campos"""
        return {
            "light": {
                "mean": float(np.mean(self.light_field)),
                "max": float(np.max(self.light_field)),
                "min": float(np.min(self.light_field))
            },
            "nutrients": {
                "mean": float(np.mean(self.nutrient_field)),
                "max": float(np.max(self.nutrient_field)),
                "min": float(np.min(self.nutrient_field)),
                "total": float(np.sum(self.nutrient_field))
            },
            "perturbation": {
                "mean": float(np.mean(np.abs(self.perturbation_field))),
                "max": float(np.max(np.abs(self.perturbation_field)))
            }
        }