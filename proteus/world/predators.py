"""
Depredadores luminosos - El peligro que emite luz
"""

import numpy as np
from typing import Tuple, Dict
import uuid


class LuminousPredator:
    """
    Depredador que caza emitiendo estallidos de luz
    La luz es una perturbación masiva en el campo topológico
    """
    
    def __init__(self, position: np.ndarray, 
                 attack_frequency: float = 0.1,
                 light_radius: float = 50.0):
        
        # Identidad y posición
        self.id = str(uuid.uuid4())
        self.position = position.astype(float)
        self.velocity = np.zeros(2)
        
        # Parámetros de caza
        self.speed = 15.0  # Más rápido que los protozoos
        self.attack_range = 30.0  # Distancia para atacar
        self.attack_frequency = attack_frequency  # Probabilidad de ataque por paso
        self.attack_cooldown = 0
        self.cooldown_duration = 50  # Pasos entre ataques
        
        # Parámetros de luz
        self.light_radius = light_radius
        self.light_intensity = 1.0
        self.light_duration = 10  # Duración del estallido
        self.light_timer = 0
        
        # Estado
        self.is_attacking = False
        self.is_luring = False  # Para subclases que usen señuelos
        self.hunting_mode = "patrol"  # "patrol" o "pursuit"
        self.target = None
        self.age = 0
        
        # Patrón de patrulla
        self.patrol_center = position.copy()
        self.patrol_radius = 200.0
        self.patrol_angle = 0.0
        
    def update_position(self, world_size: Tuple[int, int], topology: str = "toroidal"):
        """Actualiza la posición según el modo de caza"""
        if self.hunting_mode == "patrol":
            # Movimiento circular de patrulla
            self.patrol_angle += 0.02
            
            new_x = self.patrol_center[0] + self.patrol_radius * np.cos(self.patrol_angle)
            new_y = self.patrol_center[1] + self.patrol_radius * np.sin(self.patrol_angle)
            
            self.position = np.array([new_x, new_y])
            
            # Añadir algo de aleatoriedad
            self.position += np.random.normal(0, 2, size=2)
            
        # Aplicar condiciones de frontera
        if topology == "toroidal":
            self.position[0] = self.position[0] % world_size[0]
            self.position[1] = self.position[1] % world_size[1]
        else:
            self.position = np.clip(self.position, [0, 0], 
                                  [world_size[0]-1, world_size[1]-1])
    
    def attack(self):
        """Ejecuta un ataque con estallido de luz"""
        if self.attack_cooldown <= 0:
            self.is_attacking = True
            self.light_timer = self.light_duration
            self.attack_cooldown = self.cooldown_duration
            
            # El ataque consume energía del depredador también
            # (no implementado aquí, pero podría añadirse)
            
    def update(self, dt: float = 0.1):
        """Actualiza el estado del depredador"""
        self.age += dt
        
        # Actualizar timers
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            
        if self.light_timer > 0:
            self.light_timer -= 1
            if self.light_timer <= 0:
                self.is_attacking = False
                
        # Decidir si atacar (si hay presa cerca)
        if self.target is not None and self.attack_cooldown <= 0:
            if np.random.random() < self.attack_frequency:
                self.attack()
                
    def sense_prey(self, prey_positions: np.ndarray) -> int:
        """
        Detecta la presa más cercana
        Retorna el índice de la presa objetivo
        """
        if len(prey_positions) == 0:
            self.target = None
            self.hunting_mode = "patrol"
            return -1
            
        # Calcular distancias
        distances = np.linalg.norm(prey_positions - self.position, axis=1)
        
        # Encontrar la más cercana
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Cambiar a modo persecución si hay presa cerca
        if closest_distance < 200:  # Radio de detección
            self.target = prey_positions[closest_idx]
            self.hunting_mode = "pursuit"
            return closest_idx
        else:
            self.target = None
            self.hunting_mode = "patrol"
            return -1
            
    def get_light_field_contribution(self) -> Dict:
        """
        Retorna la contribución de este depredador al campo de luz
        Para ser usado por el sistema de campos
        """
        if self.is_attacking:
            return {
                'position': self.position.copy(),
                'intensity': self.light_intensity,
                'radius': self.light_radius,
                'falloff': 'gaussian'
            }
        else:
            return None
            
    def get_state(self) -> Dict:
        """Retorna el estado del depredador"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'is_attacking': self.is_attacking,
            'hunting_mode': self.hunting_mode,
            'age': self.age,
            'attack_cooldown': self.attack_cooldown,
            'light_timer': self.light_timer
        }
        

class AmbushPredator(LuminousPredator):
    """
    Variante de depredador que espera emboscado
    Emite pulsos de luz periódicos para atraer presas curiosas
    """
    
    def __init__(self, position: np.ndarray, pulse_period: float = 100):
        super().__init__(position, attack_frequency=0.3, light_radius=70.0)
        
        self.pulse_period = pulse_period
        self.pulse_timer = 0
        self.ambush_position = position.copy()
        self.lure_intensity = 0.3  # Luz tenue para atraer
        
        # Override algunos parámetros
        self.speed = 5.0  # Mucho más lento
        self.attack_range = 20.0  # Ataca más cerca
        
    def update(self, dt: float = 0.1):
        """Actualiza con comportamiento de emboscada"""
        super().update(dt)
        
        # Pulsos de luz como señuelo
        self.pulse_timer += dt
        if self.pulse_timer >= self.pulse_period:
            self.emit_lure_pulse()
            self.pulse_timer = 0
            
        # Volver lentamente a posición de emboscada
        if self.hunting_mode == "patrol":
            diff = self.ambush_position - self.position
            self.velocity = diff * 0.01
            self.position += self.velocity
            
    def emit_lure_pulse(self):
        """Emite un pulso de luz tenue como señuelo"""
        # Esto se manejará a través del sistema de campos
        # Por ahora solo marcamos el estado
        self.is_luring = True
        
        
class SwarmPredator(LuminousPredator):
    """
    Depredador que caza en grupo coordinado
    Múltiples individuos sincronizan sus ataques de luz
    """
    
    def __init__(self, position: np.ndarray, swarm_id: int):
        super().__init__(position, attack_frequency=0.05, light_radius=30.0)
        
        self.swarm_id = swarm_id
        self.swarm_cohesion = 0.5
        self.swarm_separation = 20.0
        self.sync_threshold = 0.8
        
        # Los miembros del enjambre son más débiles individualmente
        self.light_intensity = 0.6
        self.speed = 12.0
        
    def update_swarm_behavior(self, swarm_positions: np.ndarray, 
                            swarm_velocities: np.ndarray):
        """
        Actualiza comportamiento basado en el enjambre
        Implementa reglas tipo boids pero para depredadores
        """
        if len(swarm_positions) <= 1:
            return
            
        # Cohesión - moverse hacia el centro del grupo
        center = np.mean(swarm_positions, axis=0)
        cohesion_force = (center - self.position) * self.swarm_cohesion
        
        # Separación - evitar colisiones
        separation_force = np.zeros(2)
        for pos in swarm_positions:
            diff = self.position - pos
            dist = np.linalg.norm(diff)
            if 0 < dist < self.swarm_separation:
                separation_force += diff / (dist + 0.1)
                
        # Alineación - igualar velocidades
        avg_velocity = np.mean(swarm_velocities, axis=0)
        alignment_force = (avg_velocity - self.velocity) * 0.3
        
        # Combinar fuerzas
        total_force = cohesion_force + separation_force * 2 + alignment_force
        self.velocity += total_force * 0.1
        
        # Limitar velocidad
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed:
            self.velocity = self.velocity / speed * self.speed