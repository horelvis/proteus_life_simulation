"""
Motor de simulación optimizado para web
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid
import time
from dataclasses import dataclass
import threading

from .models import (
    SimulationConfig, 
    SimulationState, 
    OrganismData,
    PredatorData,
    FieldData,
    WorldUpdate,
    OrganData,
    CapabilitiesData
)
from .organisms import WebOrganism, WebPredator
from .field_manager import FieldManager
from .nutrients import NutrientField
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from proteus.core.topology_engine import TopologyEngine


class SimulationEngine:
    """Motor de simulación optimizado para tiempo real web"""
    
    def __init__(self, sim_id: str, config: SimulationConfig):
        self.id = sim_id
        self.config = config
        self.time = 0.0
        self.dt = 1.0  # Increased from 0.1 to make movement more visible
        self.current_generation = 0
        
        # Estado
        self.status = "created"  # created, running, paused, stopped
        self.is_running = False
        
        # Entidades
        self.organisms: List[WebOrganism] = []
        self.predators: List[WebPredator] = []
        self.field_manager = FieldManager(config.world_size, config.physics_params)
        
        # Sistema de nutrientes para alimentación
        self.nutrient_field = NutrientField(
            world_size=config.world_size,
            density=0.0005  # Ajustar densidad de nutrientes
        )
        
        # Motor topológico compartido - El corazón de PROTEUS
        self.topology_engine = TopologyEngine(
            world_size=config.world_size,
            topology_type="euclidean"  # No toroidal para mantener bordes
        )
        
        # Zonas seguras (refugios)
        self.safe_zones = [
            {"x": 100, "y": 100, "radius": 80},  # Esquina superior izquierda
            {"x": config.world_size[0] - 100, "y": config.world_size[1] - 100, "radius": 80},  # Esquina inferior derecha
            {"x": config.world_size[0] / 2, "y": 50, "radius": 60},  # Centro superior
        ]
        
        # Estadísticas
        self.stats = {
            "births": 0,
            "deaths": 0,
            "mutations": 0,
            "max_age": 0,
            "total_organisms": 0,
            "extinctions": 0
        }
        
        # Eventos para enviar
        self.event_queue = []
        
        # Thread de simulación
        self.simulation_thread = None
        self._lock = threading.Lock()
        
        # Inicializar población
        self._initialize_population()
    
    def _initialize_population(self):
        """Crea la población inicial"""
        # Crear organismos
        for i in range(self.config.initial_organisms):
            x = np.random.uniform(50, self.config.world_size[0] - 50)
            y = np.random.uniform(50, self.config.world_size[1] - 50)
            
            organism = WebOrganism(x, y, generation=0, topology_engine=self.topology_engine)
            self.organisms.append(organism)
        
        # Crear depredadores
        for i in range(self.config.initial_predators):
            x = np.random.uniform(100, self.config.world_size[0] - 100)
            y = np.random.uniform(100, self.config.world_size[1] - 100)
            
            predator = WebPredator(x, y)
            self.predators.append(predator)
        
        self.stats["total_organisms"] = len(self.organisms)
    
    def start(self):
        """Inicia la simulación en un thread separado"""
        if self.status == "running":
            return
        
        self.status = "running"
        self.is_running = True
        
        # Iniciar thread de simulación
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def pause(self):
        """Pausa la simulación"""
        self.is_running = False
        self.status = "paused"
    
    def stop(self):
        """Detiene la simulación"""
        self.is_running = False
        self.status = "stopped"
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
    
    def _simulation_loop(self):
        """Loop principal de simulación"""
        while self.status != "stopped":
            if self.is_running:
                start_time = time.time()
                
                with self._lock:
                    self._update_step()
                
                # Mantener 20 FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, 0.05 - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.1)
    
    def _update_step(self):
        """Un paso de simulación"""
        
        # Actualizar campos
        self.field_manager.update(self.predators)
        
        # Actualizar nutrientes
        self.nutrient_field.update(self.dt, self.safe_zones)
        
        # Actualizar campo topológico con perturbaciones
        perturbations = []
        for predator in self.predators:
            if predator.is_attacking:
                perturbations.append({
                    'position': np.array([predator.x, predator.y]),
                    'amplitude': 1.0,
                    'radius': predator.light_radius
                })
        
        # Añadir perturbaciones de zonas seguras (atractores)
        for zone in self.safe_zones:
            perturbations.append({
                'position': np.array([zone['x'], zone['y']]),
                'amplitude': -0.5,  # Negativo = atractor
                'radius': zone['radius']
            })
        
        # Añadir perturbaciones de nutrientes (atractores débiles)
        for nutrient in self.nutrient_field.nutrients[:20]:  # Limitar para rendimiento
            if nutrient.alive:
                perturbations.append({
                    'position': np.array([nutrient.x, nutrient.y]),
                    'amplitude': -0.1 * nutrient.energy_value,  # Atracción proporcional al valor
                    'radius': 20
                })
        
        self.topology_engine.update_field(perturbations)
        
        # Calcular ciclo día/noche (los depredadores son más activos de noche)
        day_cycle = (self.time / 100) % 1.0  # Ciclo completo cada 100 unidades de tiempo
        is_night = day_cycle > 0.5
        predator_activity = 1.3 if is_night else 0.6  # Más activos de noche
        
        # Actualizar depredadores
        for predator in self.predators:
            # Verificar si está en zona segura
            in_safe_zone = self._is_in_safe_zone(predator.x, predator.y)
            
            if not in_safe_zone:
                predator.update(self.organisms, self.dt, activity_level=predator_activity)
                
                # Verificar ataques
                if predator.is_attacking:
                    self._check_light_damage(predator)
            else:
                # Los depredadores no pueden entrar a zonas seguras, rebotan
                predator.vx *= -0.5
                predator.vy *= -0.5
        
        # Actualizar organismos
        dead_organisms = []
        for i, organism in enumerate(self.organisms):
            if not organism.alive:
                continue
            
            # Obtener estado del campo local
            field_state = self.field_manager.get_local_state(
                organism.x, organism.y
            )
            
            # Añadir gradiente de nutrientes reales
            nutrient_gradient = self.nutrient_field.get_nutrient_gradient(
                organism.x, organism.y, 
                sensing_radius=50 * (1 + organism.capabilities.get("chemotaxis", 0))
            )
            field_state["nutrient_gradient"] = field_state.get("nutrient_gradient", np.array([0, 0])) + nutrient_gradient
            
            # Bonus de nutrientes en zonas seguras
            if self._is_in_safe_zone(organism.x, organism.y):
                field_state["nutrient_level"] = min(1.0, field_state.get("nutrient_level", 0) * 2.0)
                field_state["light_level"] = 0  # Sin luz dañina en zonas seguras
            
            # Actualizar organismo
            organism.update(field_state, self.dt)
            
            # Verificar alimentación - colisión con nutrientes
            self._check_feeding(organism)
            
            # Verificar muerte
            if not organism.alive:
                dead_organisms.append(i)
                self._record_death(organism)
            
            # Verificar reproducción
            elif organism.can_reproduce():
                self._reproduce_organism(organism)
        
        # Eliminar muertos
        for idx in reversed(dead_organisms):
            del self.organisms[idx]
        
        # Incrementar tiempo
        self.time += self.dt
        
        # Actualizar estadísticas
        self._update_statistics()
    
    def _check_light_damage(self, predator: WebPredator):
        """Verifica daño por luz a organismos cercanos"""
        for organism in self.organisms:
            if not organism.alive:
                continue
            
            distance = np.sqrt(
                (organism.x - predator.x)**2 + 
                (organism.y - predator.y)**2
            )
            
            if distance < predator.light_radius:
                # Daño proporcional a la cercanía (más letal)
                damage_factor = (1 - distance / predator.light_radius)
                base_damage = 0.8  # Daño base más alto
                
                # Los organismos con protección resisten mejor
                protection = organism.capabilities.get("protection", 0)
                damage = base_damage * damage_factor * (1 - protection * 0.5)
                
                organism.energy -= damage
                
                if organism.energy <= 0:
                    organism.die("light_damage")
                    self._add_event({
                        "type": "death",
                        "cause": "predator_attack",
                        "organism_id": organism.id,
                        "predator_id": predator.id
                    })
    
    def _reproduce_organism(self, parent: WebOrganism):
        """Reproduce un organismo"""
        # Crear descendiente
        child_x = parent.x + np.random.uniform(-30, 30)
        child_y = parent.y + np.random.uniform(-30, 30)
        
        # Limitar a los bordes del mundo
        child_x = np.clip(child_x, 0, self.config.world_size[0])
        child_y = np.clip(child_y, 0, self.config.world_size[1])
        
        # Heredar con mutación
        child = WebOrganism(
            child_x, child_y,
            generation=parent.generation + 1,
            parent_genome=parent.get_genome(),
            topology_engine=self.topology_engine
        )
        
        # Aplicar mutaciones
        if np.random.random() < self.config.mutation_rate:
            child.mutate()
            self.stats["mutations"] += 1
        
        self.organisms.append(child)
        parent.energy *= 0.4  # Mayor costo de reproducción
        parent.last_reproduction = parent.age  # Actualizar tiempo de última reproducción
        
        self.stats["births"] += 1
        self.current_generation = max(self.current_generation, child.generation)
        
        self._add_event({
            "type": "birth",
            "parent_id": parent.id,
            "child_id": child.id,
            "generation": child.generation
        })
    
    def _check_feeding(self, organism: WebOrganism):
        """Verifica si el organismo puede alimentarse de nutrientes cercanos"""
        # Radio de detección para alimentación
        feeding_radius = 10.0
        
        # Buscar nutrientes cercanos
        nearby_nutrients = self.nutrient_field.find_nearby_nutrients(
            organism.x, organism.y, feeding_radius
        )
        
        # Intentar alimentarse del nutriente más cercano
        if nearby_nutrients and organism.feeding_cooldown <= 0:
            closest_nutrient = min(
                nearby_nutrients,
                key=lambda n: np.sqrt((n.x - organism.x)**2 + (n.y - organism.y)**2)
            )
            
            # Verificar colisión real
            distance = np.sqrt(
                (closest_nutrient.x - organism.x)**2 + 
                (closest_nutrient.y - organism.y)**2
            )
            
            if distance < feeding_radius:
                # Alimentarse
                if organism.feed(closest_nutrient.energy_value):
                    # Consumir el nutriente
                    self.nutrient_field.consume_nutrient(closest_nutrient.id)
                    
                    # Evento de alimentación
                    self._add_event({
                        "type": "feeding",
                        "organism_id": organism.id,
                        "nutrient_id": closest_nutrient.id,
                        "energy_gained": closest_nutrient.energy_value,
                        "position": {"x": organism.x, "y": organism.y}
                    })
    
    def _record_death(self, organism: WebOrganism):
        """Registra una muerte"""
        self.stats["deaths"] += 1
        
        if organism.age > self.stats["max_age"]:
            self.stats["max_age"] = organism.age
        
        self._add_event({
            "type": "death",
            "organism_id": organism.id,
            "age": organism.age,
            "cause": organism.death_cause
        })
    
    def _update_statistics(self):
        """Actualiza estadísticas globales"""
        alive_count = len([o for o in self.organisms if o.alive])
        
        if alive_count == 0 and self.stats["total_organisms"] > 0:
            self.stats["extinctions"] += 1
            self._add_event({"type": "extinction", "generation": self.current_generation})
        
        self.stats["total_organisms"] = alive_count
    
    def _add_event(self, event: Dict):
        """Añade un evento a la cola"""
        event["time"] = self.time
        self.event_queue.append(event)
        
        # Limitar tamaño de la cola
        if len(self.event_queue) > 100:
            self.event_queue = self.event_queue[-50:]
    
    def _is_in_safe_zone(self, x: float, y: float) -> bool:
        """Verifica si una posición está en una zona segura"""
        for zone in self.safe_zones:
            distance = np.sqrt((x - zone["x"])**2 + (y - zone["y"])**2)
            if distance < zone["radius"]:
                return True
        return False
    
    def get_state(self) -> SimulationState:
        """Obtiene el estado completo de la simulación"""
        with self._lock:
            # Convertir organismos a datos serializables
            organisms_data = [
                self._organism_to_data(org) 
                for org in self.organisms if org.alive
            ]
            
            predators_data = [
                self._predator_to_data(pred)
                for pred in self.predators
            ]
            
            # Campo simplificado para visualización
            field_data = FieldData(
                light_field=self.field_manager.get_light_field_preview(),
                nutrient_field=self.field_manager.get_nutrient_field_preview(),
                field_resolution=50
            )
            
            return SimulationState(
                id=self.id,
                time=self.time,
                generation=self.current_generation,
                status=self.status,
                organisms=organisms_data,
                predators=predators_data,
                field_data=field_data,
                statistics=self.stats.copy()
            )
    
    def get_update(self) -> WorldUpdate:
        """Obtiene una actualización incremental"""
        with self._lock:
            # Solo organismos vivos
            organisms_data = [
                self._organism_to_data(org) 
                for org in self.organisms if org.alive
            ]
            
            predators_data = [
                self._predator_to_data(pred)
                for pred in self.predators
            ]
            
            # Nutrientes activos
            nutrients_data = [
                {
                    "id": nutrient.id,
                    "x": nutrient.x,
                    "y": nutrient.y,
                    "energy_value": nutrient.energy_value,
                    "size": nutrient.size
                }
                for nutrient in self.nutrient_field.nutrients
                if nutrient.alive
            ]
            
            # Eventos desde la última actualización
            events = self.event_queue.copy()
            self.event_queue.clear()
            
            return WorldUpdate(
                time=self.time,
                organisms=organisms_data,
                predators=predators_data,
                nutrients=nutrients_data,
                events=events,
                statistics_delta={"alive_count": len(organisms_data)}
            )
    
    def _organism_to_data(self, organism: WebOrganism) -> OrganismData:
        """Convierte un organismo a datos serializables"""
        # Órganos
        organs_data = []
        if hasattr(organism, 'organs'):
            organs_data = [
                OrganData(
                    type=organ.type,
                    expression=organ.expression,
                    development_stage=organ.development_stage,
                    functionality=organ.functionality,
                    cost=organ.cost
                )
                for organ in organism.organs
            ]
        
        # Capacidades
        capabilities = CapabilitiesData(
            vision=organism.capabilities.get("vision", 0),
            chemotaxis=organism.capabilities.get("chemotaxis", 0),
            motility=organism.capabilities.get("motility", 0.1),
            protection=organism.capabilities.get("protection", 0),
            efficiency=organism.capabilities.get("efficiency", 1.0)
        )
        
        # Color basado en fenotipo
        color = self._get_organism_color(organism)
        
        return OrganismData(
            id=organism.id,
            type="protozoa",
            position={"x": organism.x, "y": organism.y},
            velocity={"x": organism.vx, "y": organism.vy},
            generation=organism.generation,
            age=organism.age,
            energy=organism.energy,
            alive=organism.alive,
            phenotype=organism.get_phenotype(),
            organs=organs_data,
            capabilities=capabilities,
            color=color
        )
    
    def _predator_to_data(self, predator: WebPredator) -> PredatorData:
        """Convierte un depredador a datos serializables"""
        return PredatorData(
            id=predator.id,
            position={"x": predator.x, "y": predator.y},
            velocity={"x": predator.vx, "y": predator.vy},
            is_attacking=predator.is_attacking,
            light_radius=predator.light_radius,
            attack_cooldown=predator.attack_cooldown
        )
    
    def _get_organism_color(self, organism: WebOrganism) -> str:
        """Determina el color del organismo basado en su fenotipo"""
        phenotype = organism.get_phenotype()
        
        # Colores más naturales para protozoos
        if "Visual" in phenotype:
            return "#B0C4DE"  # Light steel blue (como Euglena)
        elif "Chemical" in phenotype:
            return "#8FBC8F"  # Dark sea green (como Paramecium)
        elif "Fast" in phenotype:
            return "#DDA0DD"  # Plum (protozoos rápidos)
        elif "Armored" in phenotype:
            return "#BC8F8F"  # Rosy brown (con protección)
        elif "Efficient" in phenotype:
            return "#B0E0E6"  # Powder blue (eficientes)
        else:
            # Color base translúcido para protozoos simples
            return "#B0C4DE"  # Light steel blue por defecto
    
    def spawn_organism(self, x: float, y: float, organism_type: str = "protozoa") -> str:
        """Añade un nuevo organismo en tiempo real"""
        organism = WebOrganism(x, y, generation=self.current_generation, 
                              topology_engine=self.topology_engine)
        
        with self._lock:
            self.organisms.append(organism)
        
        return organism.id
    
    def add_predator(self, x: float, y: float) -> str:
        """Añade un nuevo depredador"""
        predator = WebPredator(x, y)
        
        with self._lock:
            self.predators.append(predator)
        
        return predator.id
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas detalladas"""
        with self._lock:
            # Estadísticas por fenotipo
            phenotype_counts = {}
            organ_counts = {}
            
            for org in self.organisms:
                if org.alive:
                    # Contar fenotipos
                    phenotype = org.get_phenotype()
                    phenotype_counts[phenotype] = phenotype_counts.get(phenotype, 0) + 1
                    
                    # Contar órganos
                    if hasattr(org, 'organs'):
                        for organ in org.organs:
                            if organ.functionality > 0.1:
                                organ_counts[organ.type] = organ_counts.get(organ.type, 0) + 1
            
            return {
                **self.stats,
                "phenotype_distribution": phenotype_counts,
                "organ_distribution": organ_counts,
                "average_age": np.mean([o.age for o in self.organisms if o.alive]) if self.organisms else 0,
                "average_energy": np.mean([o.energy for o in self.organisms if o.alive]) if self.organisms else 0
            }
    
    def get_organisms(self) -> List[OrganismData]:
        """Obtiene lista de organismos"""
        with self._lock:
            return [
                self._organism_to_data(org) 
                for org in self.organisms if org.alive
            ]