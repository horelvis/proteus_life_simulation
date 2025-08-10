"""
Organismos optimizados para simulación web con evolución topológica real
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import uuid
from dataclasses import dataclass, field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proteus.core.topology_engine import TopologyEngine, TopologicalState


@dataclass
class TopologicalGenome:
    """Genoma topológico simplificado para web"""
    topology: np.ndarray  # Matriz de conexiones topológicas
    organ_expressions: Dict[str, float] = field(default_factory=dict)
    mutation_history: List[str] = field(default_factory=list)
    # Base traits that determine organ expression
    base_motility: float = 0.5
    base_sensitivity: float = 0.5
    base_resilience: float = 0.5
    mutability: float = 0.1
    
    def mutate(self, mutation_rate: float = 0.1):
        """Aplica mutaciones al genoma"""
        # Mutar topología
        if np.random.random() < mutation_rate:
            i, j = np.random.randint(0, self.topology.shape[0], 2)
            self.topology[i, j] += np.random.normal(0, 0.1)
            self.topology[i, j] = np.clip(self.topology[i, j], -1, 1)
            self.mutation_history.append(f"topology_{i}_{j}")
        
        # Mutar rasgos base (que controlan expresión de órganos)
        if np.random.random() < mutation_rate:
            # Mutar motilidad
            self.base_motility = np.clip(
                self.base_motility + np.random.normal(0, 0.1), 0.1, 1.0
            )
            self.mutation_history.append("base_motility")
            
        if np.random.random() < mutation_rate:
            # Mutar sensibilidad
            self.base_sensitivity = np.clip(
                self.base_sensitivity + np.random.normal(0, 0.1), 0.1, 1.0
            )
            self.mutation_history.append("base_sensitivity")
            
        if np.random.random() < mutation_rate:
            # Mutar resiliencia
            self.base_resilience = np.clip(
                self.base_resilience + np.random.normal(0, 0.1), 0.1, 1.0
            )
            self.mutation_history.append("base_resilience")
        
        # Mutar tasa de mutación en sí misma
        if np.random.random() < mutation_rate * 0.5:
            self.mutability = np.clip(
                self.mutability * (0.5 + np.random.random()), 0.01, 0.5
            )
            self.mutation_history.append("mutability")
        
        # Recalcular expresión de órganos basada en nuevos rasgos
        self._update_organ_expressions()
    
    def _update_organ_expressions(self):
        """Actualiza expresiones de órganos basado en rasgos base"""
        self.organ_expressions = {
            # Órganos sensoriales de sensibilidad
            "photosensor": self.base_sensitivity,
            "chemoreceptor": self.base_sensitivity,
            
            # Órganos de movimiento de motilidad
            "flagellum": self.base_motility,
            "speed_boost": (self.base_motility - 0.5) * 2 if self.base_motility > 0.7 else 0,
            
            # Órganos de defensa de resiliencia
            "membrane": self.base_resilience,
            "armor_plates": (self.base_resilience - 0.4) * 2 if self.base_resilience > 0.6 else 0,
            "toxin_gland": (self.base_resilience - 0.5) * 2 if self.base_resilience > 0.7 else 0,
            
            # Órganos especiales de combinaciones
            "electric_organ": (self.base_motility + self.base_resilience - 1.0) 
                if (self.base_motility > 0.6 and self.base_resilience > 0.6) else 0,
            "regeneration": (self.base_resilience - 0.6) * 2 if self.base_resilience > 0.8 else 0,
            "camouflage": (self.base_sensitivity + self.base_resilience - 0.8) * 0.5
                if (self.base_sensitivity > 0.5 and self.base_resilience > 0.5) else 0,
            "vacuole": 0.3 + self.base_resilience * 0.4,
            "pheromone_emitter": self.base_sensitivity * 0.5 if self.base_sensitivity > 0.6 else 0
        }
    
    def copy(self):
        """Crea una copia del genoma"""
        return TopologicalGenome(
            topology=self.topology.copy(),
            organ_expressions=self.organ_expressions.copy(),
            mutation_history=self.mutation_history.copy(),
            base_motility=self.base_motility,
            base_sensitivity=self.base_sensitivity,
            base_resilience=self.base_resilience,
            mutability=self.mutability
        )


@dataclass
class Organ:
    """Órgano emergente simplificado"""
    type: str
    expression: float = 0.0
    development_stage: float = 0.0
    functionality: float = 0.0
    cost: float = 0.1
    
    def develop(self, dt: float, energy: float):
        """Desarrolla el órgano"""
        if energy > self.cost * dt:
            self.development_stage = min(1.0, self.development_stage + dt * 0.1)
            self.functionality = self.expression * self.development_stage
            return self.cost * dt * self.functionality
        return 0
    
    def get_capability(self) -> float:
        """Retorna la capacidad que otorga el órgano"""
        return self.functionality


class WebOrganism:
    """Organismo con verdadera evolución topológica"""
    
    def __init__(self, x: float, y: float, generation: int = 0, 
                 parent_genome: Optional[TopologicalGenome] = None,
                 topology_engine: Optional[TopologyEngine] = None):
        self.id = str(uuid.uuid4())
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        
        # Estado vital
        self.generation = generation
        self.age = 0.0
        self.max_age = 80 + np.random.random() * 40  # 80-120 unidades como en frontend
        self.senescence_start_age = self.max_age * 0.6  # Empieza a envejecer al 60%
        self.energy = 0.8  # Empiezan con menos energía
        self.alive = True
        self.death_cause = None
        
        # Estado de alimentación
        self.hunger = 0.5  # 0 = saciado, 1 = hambriento
        self.feeding_cooldown = 0
        self.nutrients_consumed = 0
        self.last_meal_age = 0
        
        # Estado topológico para computación sin neuronas
        self.topological_state = TopologicalState(
            position=np.array([x, y]),
            velocity=np.array([self.vx, self.vy]),
            trajectory_history=[np.array([x, y])],
            field_memory=None
        )
        
        # Motor topológico (compartido entre organismos)
        self.topology_engine = topology_engine
        
        # Genoma y órganos
        if parent_genome:
            self.genome = parent_genome.copy()
            self.genome.mutate()
        else:
            # Genoma inicial con rasgos base aleatorios
            base_motility = 0.3 + np.random.random() * 0.4
            base_sensitivity = 0.2 + np.random.random() * 0.6
            base_resilience = 0.4 + np.random.random() * 0.4
            
            # Expresión de órganos basada en genes heredados
            self.genome = TopologicalGenome(
                topology=np.random.randn(5, 5) * 0.1,
                organ_expressions={
                    # Órganos sensoriales de sensibilidad
                    "photosensor": base_sensitivity,
                    "chemoreceptor": base_sensitivity,
                    
                    # Órganos de movimiento de motilidad
                    "flagellum": base_motility,
                    "speed_boost": (base_motility - 0.5) * 2 if base_motility > 0.7 else 0,
                    
                    # Órganos de defensa de resiliencia
                    "membrane": base_resilience,
                    "armor_plates": (base_resilience - 0.4) * 2 if base_resilience > 0.6 else 0,
                    "toxin_gland": (base_resilience - 0.5) * 2 if base_resilience > 0.7 else 0,
                    
                    # Órganos especiales de combinaciones
                    "electric_organ": (base_motility + base_resilience - 1.0) 
                        if (base_motility > 0.6 and base_resilience > 0.6) else 0,
                    "regeneration": (base_resilience - 0.6) * 2 if base_resilience > 0.8 else 0,
                    "camouflage": (base_sensitivity + base_resilience - 0.8) * 0.5
                        if (base_sensitivity > 0.5 and base_resilience > 0.5) else 0,
                    "vacuole": 0.3 + base_resilience * 0.4,
                    "pheromone_emitter": base_sensitivity * 0.5 if base_sensitivity > 0.6 else 0
                }
            )
            
            # Guardar rasgos base en el genoma para herencia
            self.genome.base_motility = base_motility
            self.genome.base_sensitivity = base_sensitivity
            self.genome.base_resilience = base_resilience
            self.genome.mutability = 0.05 + np.random.random() * 0.10  # 5-15% mutation rate
        
        # Inicializar órganos
        self.organs: List[Organ] = []
        self._initialize_organs()
        
        # Capacidades derivadas
        self.capabilities = {
            "vision": 0.0,
            "chemotaxis": 0.0,
            "motility": 0.5,
            "protection": 0.0,
            "efficiency": 1.0
        }
        
        # Historia para trayectoria
        self.trajectory = []
        self.last_reproduction = 0.0
    
    def _initialize_organs(self):
        """Inicializa órganos basados en el genoma"""
        for organ_type, expression in self.genome.organ_expressions.items():
            if expression > 0.1:  # Umbral de expresión
                organ = Organ(
                    type=organ_type,
                    expression=expression,
                    development_stage=0.0,
                    functionality=0.0
                )
                self.organs.append(organ)
    
    def update(self, field_state: Dict, dt: float):
        """Actualiza el estado del organismo usando evolución topológica"""
        if not self.alive:
            return
        
        # Desarrollar órganos
        organ_cost = 0
        for organ in self.organs:
            cost = organ.develop(dt, self.energy)
            organ_cost += cost
        
        # Actualizar capacidades basadas en órganos
        self._update_capabilities()
        
        # Actualizar posición en el estado topológico
        self.topological_state.position = np.array([self.x, self.y])
        self.topological_state.velocity = np.array([self.vx, self.vy])
        
        # Percepción del entorno
        light_gradient = field_state.get("light_gradient", np.array([0, 0]))
        nutrient_gradient = field_state.get("nutrient_gradient", np.array([0, 0]))
        light_level = field_state.get("light_level", 0)
        
        # Actualizar hambre
        self.hunger = min(1.0, self.hunger + 0.01 * dt)  # Aumenta gradualmente
        if self.feeding_cooldown > 0:
            self.feeding_cooldown -= dt
        
        # COMPUTACIÓN TOPOLÓGICA PURA - Sin neuronas
        # Crear señal heredada basada en características topológicas
        inherited_signal = self._compute_inherited_signal()
        
        # MOTIVACIÓN DE HAMBRE: Modula la señal heredada
        # Cuando tienen hambre, son más sensibles a nutrientes
        hunger_modulation = np.array([self.hunger, self.hunger])
        inherited_signal = inherited_signal * (1 + hunger_modulation)
        
        # Calcular flujo topológico (la esencia de PROTEUS)
        if self.topology_engine:
            topological_flow = self.topology_engine.compute_topological_flow(
                self.topological_state, 
                inherited_signal
            )
        else:
            # Fallback si no hay motor topológico - usar movimiento aleatorio
            topological_flow = np.random.randn(2) * 0.5
        
        # Comportamiento emergente de la topología + capacidades + hambre
        # Suavizar el flujo topológico con el movimiento anterior
        if not hasattr(self, 'prev_flow'):
            self.prev_flow = np.array([0.0, 0.0])
        
        # Mezclar con flujo anterior para suavidad (70% nuevo, 30% anterior)
        smooth_flow = topological_flow * 0.7 + self.prev_flow * 0.3
        self.prev_flow = smooth_flow.copy()
        
        dx, dy = smooth_flow[0], smooth_flow[1]
        
        # Detección de depredadores por luz (si tiene visión)
        if self.capabilities["vision"] > 0 and light_level > 0.1:
            # La fotosensibilidad permite detectar el peligro
            danger_detection = self.capabilities["vision"] * light_level
            
            # Respuesta de escape proporcional a la capacidad visual
            escape_force = danger_detection * 3.0
            
            # Huir en dirección opuesta al gradiente de luz
            if np.linalg.norm(light_gradient) > 0:
                escape_direction = -light_gradient / np.linalg.norm(light_gradient)
                dx += escape_direction[0] * escape_force
                dy += escape_direction[1] * escape_force
                
                # Aumentar velocidad temporalmente (pánico)
                self.capabilities["motility"] = min(2.0, self.capabilities["motility"] * 1.2)
        
        # Quimiotaxis (si tiene quimiorreceptores) - FUERTEMENTE modulado por HAMBRE
        if self.capabilities["chemotaxis"] > 0:
            # El hambre amplifica dramáticamente la búsqueda de nutrientes
            hunger_amplification = 1 + self.hunger * 3  # Hasta 4x cuando hambriento
            chemotaxis_force = nutrient_gradient * self.capabilities["chemotaxis"] * hunger_amplification * 0.5
            dx += chemotaxis_force[0]
            dy += chemotaxis_force[1]
            
            # Si está muy hambriento, ignora parcialmente otros estímulos
            if self.hunger > 0.8:
                dx = dx * 0.3 + chemotaxis_force[0] * 0.7
                dy = dy * 0.3 + chemotaxis_force[1] * 0.7
        
        # El movimiento "aleatorio" es realmente el flujo topológico
        # No hay aleatoriedad real en PROTEUS - todo es determinista topológicamente
        
        # Aplicar motilidad con movimiento más suave
        motility = self.capabilities["motility"]
        # Mayor inercia (0.95) para movimiento más fluido
        self.vx = self.vx * 0.95 + dx * motility * dt * 0.8
        self.vy = self.vy * 0.95 + dy * motility * dt * 0.8
        
        # Límite de velocidad más suave
        speed = np.sqrt(self.vx**2 + self.vy**2)
        max_speed = 3.0 * motility
        if speed > max_speed:
            # Reducción gradual de velocidad
            reduction_factor = 0.95
            self.vx = self.vx * reduction_factor
            self.vy = self.vy * reduction_factor
        
        # Actualizar posición
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Debug: log movement occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        
        # Consumo de energía (más costoso)
        base_cost = 0.02 * dt  # Duplicado
        movement_cost = speed * 0.003 * dt  # Triplicado
        total_cost = (base_cost + movement_cost + organ_cost) / self.capabilities["efficiency"]
        
        # Ganancia de energía por nutrientes (más difícil)
        nutrient_level = field_state.get("nutrient_level", 0)
        energy_gain = nutrient_level * 0.03 * dt * self.capabilities["efficiency"]  # Reducido
        
        # Penalización adicional por luz
        light_level = field_state.get("light_level", 0)
        if light_level > 0:
            # La luz daña gradualmente
            light_damage = light_level * 0.1 * dt * (1 - self.capabilities["protection"])
            total_cost += light_damage
        
        self.energy += energy_gain - total_cost
        self.energy = max(0, min(1.5, self.energy))  # Límite más bajo
        
        # Envejecimiento
        self.age += dt
        
        # Verificar muerte
        if self.energy <= 0:
            self.die("starvation")
        elif self.age > self.max_age:
            self.die("old_age")
        
        # Efectos del envejecimiento (senescencia)
        if self.age > self.senescence_start_age:
            aging_factor = (self.age - self.senescence_start_age) / (self.max_age - self.senescence_start_age)
            # Reducir capacidades gradualmente
            self.capabilities["motility"] *= (1 - aging_factor * 0.3)
            self.capabilities["chemotaxis"] *= (1 - aging_factor * 0.2)
            # Mayor consumo de energía con la edad
            self.energy -= aging_factor * 0.001
        
        # Guardar posición para trayectoria
        if len(self.trajectory) < 100:
            self.trajectory.append({"x": self.x, "y": self.y})
        
        # Actualizar historia topológica
        self.topological_state.add_position(np.array([self.x, self.y]))
    
    def _update_capabilities(self):
        """Actualiza capacidades basadas en órganos funcionales"""
        # Reset capacidades
        self.capabilities = {
            "vision": 0.0,
            "chemotaxis": 0.0,
            "motility": 0.5,
            "protection": 0.0,
            "efficiency": 1.0
        }
        
        # Agregar contribuciones de órganos
        for organ in self.organs:
            if organ.functionality > 0:
                if organ.type == "photosensor":
                    self.capabilities["vision"] += organ.functionality
                elif organ.type == "chemoreceptor":
                    self.capabilities["chemotaxis"] += organ.functionality
                elif organ.type == "flagellum":
                    self.capabilities["motility"] += organ.functionality * 0.5
                elif organ.type == "cilia":
                    self.capabilities["motility"] += organ.functionality * 0.3
                elif organ.type == "membrane":
                    self.capabilities["protection"] += organ.functionality * 0.5
                    self.capabilities["efficiency"] += organ.functionality * 0.2
                elif organ.type == "vacuole":
                    self.capabilities["efficiency"] += organ.functionality * 0.3
                elif organ.type == "crystallin":
                    self.capabilities["vision"] += organ.functionality * 1.5
                elif organ.type == "pigment_spot":
                    self.capabilities["vision"] += organ.functionality * 0.5
        
        # Limitar capacidades
        for key in self.capabilities:
            if key == "motility" or key == "efficiency":
                self.capabilities[key] = max(0.1, min(2.0, self.capabilities[key]))
            else:
                self.capabilities[key] = max(0.0, min(1.0, self.capabilities[key]))
    
    def can_reproduce(self) -> bool:
        """Verifica si puede reproducirse - necesita estar bien alimentado"""
        return (
            self.alive and 
            self.energy > 1.3 and  # Necesita más energía
            self.hunger < 0.3 and  # No puede estar hambriento
            self.age > 10 and     # Debe ser más viejo
            self.age - self.last_reproduction > 8 and  # Más tiempo entre reproducciones
            self.nutrients_consumed > 3  # Debe haberse alimentado bien
        )
    
    def feed(self, nutrient_energy: float):
        """El organismo se alimenta de un nutriente"""
        if self.feeding_cooldown <= 0:
            # Ganar energía
            self.energy += nutrient_energy * self.capabilities["efficiency"]
            self.energy = min(self.energy, 2.0)  # Límite máximo
            
            # Reducir hambre significativamente
            self.hunger = max(0, self.hunger - nutrient_energy * 2)
            
            # Registrar alimentación
            self.nutrients_consumed += 1
            self.last_meal_age = self.age
            self.feeding_cooldown = 5  # Tiempo entre comidas
            
            return True
        return False
    
    def die(self, cause: str):
        """Marca el organismo como muerto"""
        self.alive = False
        self.death_cause = cause
    
    def get_genome(self) -> TopologicalGenome:
        """Obtiene el genoma para herencia"""
        return self.genome
    
    def mutate(self):
        """Aplica mutaciones basadas en invariantes topológicos"""
        # Extraer características topológicas de la trayectoria
        invariants = self._extract_topological_invariants()
        
        # Mutación base del genoma - usar tasa heredada
        base_rate = self.genome.mutability if hasattr(self.genome, 'mutability') else 0.1
        
        # Modular tasa de mutación según comportamiento topológico
        if invariants['persistence'] > 0.8:
            # Organismos persistentes mutan menos
            mutation_rate = base_rate * 0.5
        elif invariants['complexity'] > 0.7:
            # Organismos con trayectorias complejas mutan más
            mutation_rate = base_rate * 1.5
        else:
            mutation_rate = base_rate
        
        self.genome.mutate(mutation_rate)
        
        # EVOLUCIÓN TOPOLÓGICA: Los órganos emergen según patrones de movimiento
        # No es aleatorio - es determinado por la topología
        
        # Organismos que giran mucho desarrollan sensores
        if invariants['winding_number'] != 0 and np.random.random() < 0.1:
            if "photosensor" in self.genome.organ_expressions:
                self.genome.organ_expressions["photosensor"] += 0.1
            else:
                self.genome.organ_expressions["photosensor"] = 0.2
        
        # Organismos persistentes desarrollan motilidad
        if invariants['persistence'] > 0.7 and np.random.random() < 0.1:
            if "flagellum" in self.genome.organ_expressions:
                self.genome.organ_expressions["flagellum"] += 0.1
            else:
                self.genome.organ_expressions["flagellum"] = 0.2
        
        # Organismos con trayectorias complejas desarrollan quimiotaxis
        if invariants['complexity'] > 0.6 and np.random.random() < 0.1:
            if "chemoreceptor" in self.genome.organ_expressions:
                self.genome.organ_expressions["chemoreceptor"] += 0.1
            else:
                self.genome.organ_expressions["chemoreceptor"] = 0.2
        
        # Actualizar matriz topológica basada en invariantes
        if hasattr(self.genome, 'topology'):
            # La topología se ajusta según el comportamiento observado
            if invariants['curvature'] > 0.5:
                # Alta curvatura -> más conexiones cruzadas
                i, j = np.random.randint(0, self.genome.topology.shape[0], 2)
                if i != j:
                    self.genome.topology[i, j] += invariants['curvature'] * 0.1
                    self.genome.topology[i, j] = np.clip(self.genome.topology[i, j], -1, 1)
    
    def _compute_inherited_signal(self) -> np.ndarray:
        """
        Calcula la señal heredada basada en la topología del genoma
        Esta es la clave de la evolución sin genes - pura topología
        """
        # Usar la matriz topológica del genoma para modular el comportamiento
        if hasattr(self.genome, 'topology'):
            # Proyectar el estado actual a través de la topología heredada
            state_vector = np.array([
                self.energy,
                self.age / 50.0,  # Normalizado
                self.capabilities.get("vision", 0),
                self.capabilities.get("chemotaxis", 0),
                self.capabilities.get("motility", 0.1),
            ])
            
            # Reducir dimensiones si es necesario
            if len(state_vector) > self.genome.topology.shape[0]:
                state_vector = state_vector[:self.genome.topology.shape[0]]
            elif len(state_vector) < self.genome.topology.shape[0]:
                state_vector = np.pad(state_vector, 
                                    (0, self.genome.topology.shape[0] - len(state_vector)))
            
            # Computación topológica: proyectar estado a través de la matriz heredada
            signal = np.dot(self.genome.topology, state_vector)
            
            # Tomar las primeras 2 componentes para movimiento 2D
            return signal[:2] * 0.5
        else:
            return np.zeros(2)
    
    def _extract_topological_invariants(self) -> Dict:
        """
        Extrae invariantes topológicos de la trayectoria del organismo
        Estos se heredan y definen el comportamiento futuro
        """
        if self.topology_engine and len(self.topological_state.trajectory_history) > 10:
            return self.topology_engine.extract_topological_features(
                self.topological_state.trajectory_history
            )
        else:
            return {
                'curvature': 0.0,
                'winding_number': 0,
                'persistence': 0.0,
                'complexity': 0.0
            }
    
    def get_phenotype(self) -> str:
        """Retorna descripción del fenotipo basado en topología"""
        traits = []
        
        # Rasgos basados en capacidades
        if self.capabilities["vision"] > 0.5:
            traits.append("Visual")
        if self.capabilities["chemotaxis"] > 0.5:
            traits.append("Chemical")
        if self.capabilities["motility"] > 1.5:
            traits.append("Fast")
        if self.capabilities["protection"] > 0.5:
            traits.append("Armored")
        if self.capabilities["efficiency"] > 1.3:
            traits.append("Efficient")
        
        # Rasgos topológicos
        invariants = self._extract_topological_invariants()
        if invariants['persistence'] > 0.7:
            traits.append("Persistent")
        if invariants['winding_number'] != 0:
            traits.append("Spiral")
        if invariants['complexity'] > 0.5:
            traits.append("Complex")
        
        return "-".join(traits) if traits else "Basic"


class WebPredator:
    """Depredador luminoso simplificado para web"""
    
    def __init__(self, x: float, y: float):
        self.id = str(uuid.uuid4())
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        
        # Ciclo de vida finito como en frontend
        self.age = 0
        self.max_age = 100 + np.random.random() * 50  # 100-150 unidades
        self.size = 9  # Tamaño inicial (reducido 10%)
        self.energy = 100
        self.max_energy = 150
        self.alive = True
        self.death_cause = None
        self.successful_hunts = 0
        
        # Propiedades de ataque mejoradas
        self.light_radius = 80.0  # Mayor alcance
        self.is_attacking = False
        self.attack_cooldown = 0
        self.attack_duration = 30  # Ataques más largos
        
        # Comportamiento
        self.target_organism = None
        self.patrol_angle = np.random.random() * 2 * np.pi
        self.hunt_radius = 200  # Radio de búsqueda más amplio
        
        # Sistema de memoria como en frontend
        self.memory = {
            'last_hunt_position': {'x': x, 'y': y},
            'time_since_last_hunt': 0,
            'failed_hunt_attempts': 0,
            'visited_areas': [],  # Lista de áreas visitadas recientemente
            'current_patrol_target': None
        }
        self.memory_duration = 500  # Tiempo que recuerda las áreas
        self.area_radius = 100  # Radio de cada área recordada
    
    def update(self, organisms: List[WebOrganism], dt: float, activity_level: float = 1.0):
        """Actualiza el depredador"""
        if not self.alive:
            return
            
        # Envejecimiento
        self.age += dt
        
        # Consumo de energía base (reducido como en frontend)
        base_consumption = 0.02 if not self.is_attacking else 0.05
        self.energy -= base_consumption * dt
        
        # Crecimiento con caza exitosa
        if self.successful_hunts > 0 and self.size < 25:
            self.size = min(22.5, 9 + self.successful_hunts * 0.5)  # Reducido 10%
        
        # Muerte por edad o hambre
        if self.age > self.max_age:
            self.die("old_age")
            return
        elif self.energy <= 0:
            self.die("starvation")
            return
        elif self.memory['time_since_last_hunt'] > 1000:
            # Muerte por hambre prolongada
            self.die("prolonged_starvation")
            return
        
        # Actualizar memoria
        self.memory['time_since_last_hunt'] += 1
        
        # Limpiar áreas visitadas antiguas
        self.memory['visited_areas'] = [
            area for area in self.memory['visited_areas']
            if area['time'] < self.memory_duration
        ]
        
        # Incrementar tiempo en áreas visitadas
        for area in self.memory['visited_areas']:
            area['time'] += 1
        
        # Reducir cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
            self.is_attacking = self.attack_cooldown > (100 - self.attack_duration)
        
        # Buscar presa más cercana
        if not self.is_attacking and organisms:
            closest_dist = float('inf')
            closest_organism = None
            
            for org in organisms:
                if org.alive:
                    dist = np.sqrt((org.x - self.x)**2 + (org.y - self.y)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_organism = org
            
            self.target_organism = closest_organism
            
            # Perseguir si está dentro del radio de caza
            if closest_organism and closest_dist < self.hunt_radius:
                dx = closest_organism.x - self.x
                dy = closest_organism.y - self.y
                norm = np.sqrt(dx**2 + dy**2)
                
                if norm > 0:
                    # Velocidad proporcional a la distancia y nivel de actividad
                    base_speed = 2.5 if closest_dist > 100 else 1.8
                    speed = base_speed * activity_level
                    self.vx = (dx / norm) * speed
                    self.vy = (dy / norm) * speed
                
                # Atacar si está dentro del radio de luz
                if closest_dist < self.light_radius * 0.8 and self.attack_cooldown == 0:
                    self.is_attacking = True
                    self.attack_cooldown = 80  # Menos cooldown = más ataques
                    # Actualizar memoria de caza exitosa
                    self.memory['last_hunt_position'] = {'x': self.x, 'y': self.y}
                    self.memory['time_since_last_hunt'] = 0
                    self.memory['failed_hunt_attempts'] = 0
            else:
                # Sin presas cerca - usar patrullaje inteligente con memoria
                self._intelligent_patrol(activity_level)
        
        # Actualizar posición
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Fricción
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Registrar posición actual si ha pasado suficiente tiempo
        if len(self.memory['visited_areas']) == 0 or \
           all(self._distance_to_area({'x': self.x, 'y': self.y}, area) > self.area_radius 
               for area in self.memory['visited_areas']):
            self.memory['visited_areas'].append({
                'x': self.x,
                'y': self.y,
                'time': 0
            })
    
    def _intelligent_patrol(self, activity_level: float):
        """Patrullaje inteligente que evita áreas ya visitadas"""
        # Si ha pasado mucho tiempo sin cazar, buscar nuevas áreas
        if self.memory['time_since_last_hunt'] > 200:
            # Elegir objetivo de patrullaje si no hay uno
            if not self.memory['current_patrol_target'] or \
               self._distance_to_area({'x': self.x, 'y': self.y}, self.memory['current_patrol_target']) < 50:
                # Generar nuevo objetivo evitando áreas visitadas
                attempts = 0
                while attempts < 10:
                    target_x = np.random.uniform(50, 750)  # Evitar bordes
                    target_y = np.random.uniform(50, 550)
                    new_target = {'x': target_x, 'y': target_y}
                    
                    # Verificar que no esté cerca de áreas visitadas
                    if all(self._distance_to_area(new_target, area) > self.area_radius * 1.5 
                           for area in self.memory['visited_areas']):
                        self.memory['current_patrol_target'] = new_target
                        break
                    attempts += 1
                
                # Si no encuentra buen objetivo, ir a posición aleatoria
                if attempts >= 10:
                    self.memory['current_patrol_target'] = {
                        'x': np.random.uniform(50, 750),
                        'y': np.random.uniform(50, 550)
                    }
            
            # Moverse hacia el objetivo
            if self.memory['current_patrol_target']:
                dx = self.memory['current_patrol_target']['x'] - self.x
                dy = self.memory['current_patrol_target']['y'] - self.y
                norm = np.sqrt(dx**2 + dy**2)
                
                if norm > 0:
                    patrol_speed = 1.5 * activity_level
                    self.vx = (dx / norm) * patrol_speed
                    self.vy = (dy / norm) * patrol_speed
        else:
            # Patrullaje normal aleatorio
            self.patrol_angle += np.random.normal(0, 0.15)
            patrol_speed = 1.2 * activity_level
            self.vx = np.cos(self.patrol_angle) * patrol_speed
            self.vy = np.sin(self.patrol_angle) * patrol_speed
    
    def _distance_to_area(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calcula distancia entre dos posiciones"""
        return np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
    
    def die(self, cause: str):
        """Marca al depredador como muerto"""
        self.alive = False
        self.death_cause = cause
        self.is_attacking = False
    
    def feed(self, energy_gained: float):
        """Alimentarse después de caza exitosa"""
        self.energy = min(self.max_energy, self.energy + energy_gained)
        self.successful_hunts += 1
        self.memory['time_since_last_hunt'] = 0
        
    def can_reproduce(self) -> bool:
        """Verifica si puede reproducirse"""
        return (
            self.alive and
            self.age > 20 and
            self.energy > self.max_energy * 0.8 and
            self.successful_hunts >= 3
        )
    
    def reproduce(self) -> 'WebPredator':
        """Crea un descendiente"""
        # Reducir energía por reproducción
        self.energy *= 0.6
        
        # Crear descendiente cerca
        offset_x = np.random.uniform(-30, 30)
        offset_y = np.random.uniform(-30, 30)
        
        offspring = WebPredator(self.x + offset_x, self.y + offset_y)
        # Heredar algunas características
        offspring.hunt_radius = self.hunt_radius * (0.9 + np.random.random() * 0.2)
        offspring.light_radius = self.light_radius * (0.9 + np.random.random() * 0.2)
        
        return offspring