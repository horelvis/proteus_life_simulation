"""
Sistema de nutrientes para alimentación de organismos
"""

import numpy as np
import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Nutrient:
    """Partícula de nutriente que los organismos pueden consumir"""
    id: str
    x: float
    y: float
    energy_value: float = 0.3
    size: float = 3.0
    alive: bool = True
    age: float = 0.0
    
    def update(self, dt: float):
        """Actualiza el nutriente"""
        self.age += dt
        
        # Los nutrientes se degradan con el tiempo
        if self.age > 100:
            self.energy_value *= 0.99
            
        # Mueren si pierden todo su valor
        if self.energy_value < 0.01:
            self.alive = False


class NutrientField:
    """Gestiona el campo de nutrientes en el mundo"""
    
    def __init__(self, world_size: Tuple[int, int], density: float = 0.001):
        self.world_size = world_size
        self.density = density  # Nutrientes por unidad de área
        self.nutrients: List[Nutrient] = []
        self.spawn_rate = 0.1  # Probabilidad de generar nutrientes por frame
        self.max_nutrients = int(world_size[0] * world_size[1] * density)
        
        # Generar nutrientes iniciales
        self._spawn_initial_nutrients()
    
    def _spawn_initial_nutrients(self):
        """Genera la población inicial de nutrientes"""
        initial_count = self.max_nutrients // 2
        
        for _ in range(initial_count):
            x = np.random.uniform(20, self.world_size[0] - 20)
            y = np.random.uniform(20, self.world_size[1] - 20)
            
            nutrient = Nutrient(
                id=str(uuid.uuid4()),
                x=x,
                y=y,
                energy_value=np.random.uniform(0.2, 0.5)
            )
            self.nutrients.append(nutrient)
    
    def update(self, dt: float, safe_zones: List[Dict] = None):
        """Actualiza todos los nutrientes y genera nuevos"""
        # Actualizar nutrientes existentes
        dead_nutrients = []
        for i, nutrient in enumerate(self.nutrients):
            nutrient.update(dt)
            if not nutrient.alive:
                dead_nutrients.append(i)
        
        # Eliminar nutrientes muertos
        for idx in reversed(dead_nutrients):
            del self.nutrients[idx]
        
        # Generar nuevos nutrientes si hay espacio
        if len(self.nutrients) < self.max_nutrients and np.random.random() < self.spawn_rate:
            self._spawn_nutrient(safe_zones)
    
    def _spawn_nutrient(self, safe_zones: List[Dict] = None):
        """Genera un nuevo nutriente, preferentemente en zonas seguras"""
        if safe_zones and np.random.random() < 0.7:  # 70% en zonas seguras
            # Elegir una zona segura aleatoria
            zone = np.random.choice(safe_zones)
            
            # Generar posición dentro de la zona
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, zone['radius'])
            x = zone['x'] + radius * np.cos(angle)
            y = zone['y'] + radius * np.sin(angle)
            
            # Mayor valor energético en zonas seguras
            energy_value = np.random.uniform(0.4, 0.6)
        else:
            # Generar en posición aleatoria
            x = np.random.uniform(20, self.world_size[0] - 20)
            y = np.random.uniform(20, self.world_size[1] - 20)
            energy_value = np.random.uniform(0.2, 0.4)
        
        nutrient = Nutrient(
            id=str(uuid.uuid4()),
            x=x,
            y=y,
            energy_value=energy_value
        )
        self.nutrients.append(nutrient)
    
    def consume_nutrient(self, nutrient_id: str) -> float:
        """Consume un nutriente y retorna su valor energético"""
        for i, nutrient in enumerate(self.nutrients):
            if nutrient.id == nutrient_id:
                energy = nutrient.energy_value
                del self.nutrients[i]
                return energy
        return 0.0
    
    def find_nearby_nutrients(self, x: float, y: float, radius: float) -> List[Nutrient]:
        """Encuentra nutrientes cercanos a una posición"""
        nearby = []
        for nutrient in self.nutrients:
            if nutrient.alive:
                dist = np.sqrt((nutrient.x - x)**2 + (nutrient.y - y)**2)
                if dist <= radius:
                    nearby.append(nutrient)
        return nearby
    
    def get_nutrient_gradient(self, x: float, y: float, sensing_radius: float = 50) -> np.ndarray:
        """Calcula el gradiente de nutrientes en una posición"""
        gradient = np.array([0.0, 0.0])
        total_weight = 0.0
        
        for nutrient in self.nutrients:
            if nutrient.alive:
                dx = nutrient.x - x
                dy = nutrient.y - y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < sensing_radius and dist > 0:
                    # Peso inversamente proporcional a la distancia
                    weight = nutrient.energy_value / (dist + 1.0)
                    gradient += np.array([dx, dy]) / dist * weight
                    total_weight += weight
        
        # Normalizar el gradiente
        if total_weight > 0:
            gradient /= total_weight
        
        return gradient