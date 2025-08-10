"""
Protozoa - La criatura topológica sin cerebro
"""

import numpy as np
from typing import Optional, Dict, List
import uuid

from ..core.topology_engine import TopologicalState, TopologyEngine
from ..core.inheritance import TopologicalSeed, TopologicalInheritance
from .organs import OrganSystem, TopologicalOrgan


class Protozoa:
    """
    Un organismo que computa a través del movimiento
    Sin neuronas, sin decisiones - solo flujo topológico
    """
    
    def __init__(self, position: np.ndarray, seed: Optional[TopologicalSeed] = None):
        # Identidad
        self.id = str(uuid.uuid4())
        self.generation = seed.generation + 1 if seed else 0
        
        # Estado físico
        self.position = position.astype(float)
        self.velocity = np.zeros(2)
        self.state = TopologicalState(
            position=position.copy(),
            velocity=self.velocity.copy(),
            trajectory_history=[position.copy()]
        )
        
        # Propiedades vitales
        self.alive = True
        self.age = 0.0
        self.energy = 1.0
        self.size = 5.0
        
        # Herencia topológica
        self.seed = seed
        self.inheritance_system = TopologicalInheritance()
        self.inherited_modulations = self.inheritance_system.apply_inheritance(seed)
        
        # Parámetros modulados por herencia
        self.movement_bias = self.inherited_modulations.get('movement_bias', np.zeros(2))
        self.field_sensitivity = self.inherited_modulations.get('field_sensitivity', np.ones(4))
        self.danger_memory = self.inherited_modulations.get('danger_memory', np.array([]))
        
        # Sistema de órganos emergentes
        inherited_organs = None
        if seed and hasattr(seed, 'organs_data'):
            inherited_organs = seed.organs_data
        self.organ_system = OrganSystem(inherited_organs)
        
        # Umbrales y parámetros (ahora influenciados por órganos)
        self.light_tolerance = 0.7 - self.inherited_modulations.get('light_avoidance', 0) * 0.1
        self.speed = 10.0 * (1 + self.inherited_modulations.get('flow_following', 0) * 0.2)
        self.reproduction_threshold = 2.0
        self.reproduction_cooldown = 0
        
        # Historia de campos percibidos
        self.field_history = []
        
        # Estadísticas de vida
        self.distance_traveled = 0.0
        self.light_exposures = 0
        self.nutrients_consumed = 0.0
        
    def sense_environment(self, local_field: Dict):
        """
        Percibe el entorno local
        La percepción ahora es modulada por órganos emergentes
        """
        # Desarrollar órganos con la energía disponible
        organ_cost = self.organ_system.develop_organs(local_field, self.energy)
        self.energy -= organ_cost * 0.1  # Costo de mantenimiento
        # Guardar en historia
        self.field_history.append(local_field)
        if len(self.field_history) > 100:
            self.field_history.pop(0)
            
        # El campo modula directamente el estado interno
        self.state.field_memory = np.array([
            local_field['light_intensity'],
            local_field['nutrient_density'],
            np.linalg.norm(local_field['flow_average']),
            local_field['field_curvature']
        ])
        
        # Contar exposiciones a luz
        if local_field['light_intensity'] > 0.3:
            self.light_exposures += 1
            
    def move(self, topology_engine: TopologyEngine, dt: float):
        """
        Movimiento basado en dinámicas topológicas
        El movimiento ES la computación
        """
        # Calcular flujo topológico
        flow = topology_engine.compute_topological_flow(
            self.state,
            self.movement_bias  # Señal heredada
        )
        
        # Modular por sensibilidad al campo
        if self.state.field_memory is not None:
            # La sensibilidad heredada afecta cómo responde al campo
            field_influence = self.state.field_memory * self.field_sensitivity
            
            # Evitar luz (inversamente proporcional a la intensidad)
            light_avoidance = -field_influence[0] * 5.0
            
            # Buscar nutrientes
            nutrient_seeking = field_influence[1] * 2.0
            
            # Seguir flujo
            flow_following = field_influence[2] * 1.0
            
            # Responder a curvatura
            curvature_response = field_influence[3] * 0.5
            
            # Combinar influencias clásicas
            total_influence = np.array([
                light_avoidance * np.cos(self.age * 0.1),  # Oscilación temporal
                light_avoidance * np.sin(self.age * 0.1) + nutrient_seeking
            ])
            
            flow += total_influence * 0.3
            
        # Añadir modulación de órganos
        organ_dx, organ_dy = self.organ_system.get_movement_modulation({
            'light_intensity': self.state.field_memory[0] if self.state.field_memory is not None else 0,
            'light_gradient': (-flow[0], -flow[1]),  # Gradiente inferido
            'chemical_gradient': (flow[0] * 0.5, flow[1] * 0.5),  # Gradiente químico aproximado
            'viscosity': 0.8
        })
        
        flow[0] += organ_dx * 0.5
        flow[1] += organ_dy * 0.5
            
        # Evitar zonas peligrosas memorizadas
        if len(self.danger_memory) > 0:
            for danger_zone in self.danger_memory:
                dist_to_danger = np.linalg.norm(self.position - danger_zone)
                if dist_to_danger < 50:
                    # Repulsión de la zona peligrosa
                    repulsion = (self.position - danger_zone) / (dist_to_danger + 0.1)
                    flow += repulsion * (50 - dist_to_danger) * 0.1
        
        # Actualizar velocidad con inercia
        self.velocity = self.velocity * 0.9 + flow * 0.1
        
        # Limitar velocidad
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed:
            self.velocity = self.velocity / speed * self.speed
            
        # Actualizar posición
        self.position += self.velocity * dt
        
        # Aplicar condiciones de frontera
        self.position = topology_engine.apply_toroidal_boundary(self.position)
        
        # Actualizar estado topológico
        self.state.position = self.position.copy()
        self.state.velocity = self.velocity.copy()
        self.state.add_position(self.position)
        
        # Actualizar estadísticas
        self.distance_traveled += np.linalg.norm(self.velocity) * dt
        
        # Costo energético del movimiento
        self.energy -= speed * 0.0001
        
    def can_reproduce(self) -> bool:
        """Verifica si puede reproducirse"""
        return (self.alive and 
                self.energy > self.reproduction_threshold and
                self.age > 50 and
                self.reproduction_cooldown <= 0)
    
    def generate_topological_seed(self) -> TopologicalSeed:
        """
        Genera una semilla topológica con la experiencia vivida
        Esta es la herencia - no genes, sino patrones de vida
        """
        seed = self.inheritance_system.extract_seed(
            trajectory=self.state.trajectory_history,
            field_history=self.field_history,
            survival_time=self.age,
            generation=self.generation,
            parent_id=self.id
        )
        
        # Añadir información de órganos a la semilla
        if hasattr(self, 'organ_system'):
            seed.organs_data = [organ for organ in self.organ_system.organs 
                               if organ.functionality > 0.1]  # Solo órganos funcionales
        
        return seed
    
    def die(self, cause: str):
        """Marca la muerte del organismo"""
        self.alive = False
        self.death_cause = cause
        self.death_age = self.age
        
    def get_state(self) -> Dict:
        """Retorna el estado completo del organismo"""
        return {
            'id': self.id,
            'generation': self.generation,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'age': self.age,
            'energy': self.energy,
            'alive': self.alive,
            'distance_traveled': self.distance_traveled,
            'light_exposures': self.light_exposures,
            'has_seed': self.seed is not None,
            'trajectory_length': len(self.state.trajectory_history),
            'phenotype': self.organ_system.describe_phenotype() if hasattr(self, 'organ_system') else 'Primitive',
            'capabilities': self.organ_system.get_total_capabilities() if hasattr(self, 'organ_system') else {}
        }
    
    def get_topological_features(self) -> Dict:
        """Extrae características topológicas actuales"""
        if len(self.state.trajectory_history) < 3:
            return {}
            
        return TopologyEngine.extract_topological_features(
            self.state.trajectory_history
        )