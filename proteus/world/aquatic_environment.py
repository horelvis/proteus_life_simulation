"""
Entorno acuático 2D - El mundo donde viven las criaturas topológicas
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time

from ..core.topology_engine import TopologyEngine
from ..core.field_dynamics import FieldDynamics


class World:
    """
    El mundo acuático visto desde arriba
    Un espacio continuo donde emergen comportamientos sin programación
    """
    
    def __init__(self, size: Tuple[int, int] = (1000, 1000), 
                 viscosity: float = 0.8,
                 temperature: float = 20.0,
                 topology: str = "toroidal"):
        
        self.size = size
        self.viscosity = viscosity
        self.temperature = temperature
        self.topology = topology
        
        # Motores principales
        self.topology_engine = TopologyEngine(size, topology)
        self.field_dynamics = FieldDynamics(size)
        self.field_dynamics.viscosity = viscosity
        
        # Habitantes del mundo
        self.protozoa = []
        self.predators = []
        
        # Estado y estadísticas
        self.time = 0
        self.dt = 0.1
        self.stats = {
            'births': 0,
            'deaths': 0,
            'total_distance_traveled': 0,
            'max_survival_time': 0,
            'current_population': 0
        }
        
        # Configuración visual
        self.fig = None
        self.ax = None
        self.visualization_enabled = False
        
    def add_protozoa(self, protozoa_list: List):
        """Añade protozoos al mundo"""
        self.protozoa.extend(protozoa_list)
        self.stats['current_population'] = len(self.protozoa)
        
    def add_predators(self, predator_list: List):
        """Añade depredadores al mundo"""
        self.predators.extend(predator_list)
        
    def simulate(self, protozoa: List, predators: List, 
                time_steps: int = 10000) -> List:
        """
        Simula el mundo por un número de pasos de tiempo
        Retorna la lista de supervivientes
        """
        self.protozoa = protozoa
        self.predators = predators
        self.stats['current_population'] = len(self.protozoa)
        
        survivors = []
        
        for step in range(time_steps):
            # Actualizar campos
            self._update_fields()
            
            # Mover depredadores
            self._update_predators()
            
            # Mover protozoos y verificar supervivencia
            self._update_protozoa()
            
            # Verificar nacimientos
            self._check_reproduction()
            
            # Incrementar tiempo
            self.time += self.dt
            
            # Log periódico
            if step % 1000 == 0:
                print(f"Step {step}: {len(self.protozoa)} protozoa alive")
                
        # Retornar supervivientes
        survivors = [p for p in self.protozoa if p.alive]
        
        # Actualizar estadísticas finales
        for p in survivors:
            if p.age > self.stats['max_survival_time']:
                self.stats['max_survival_time'] = p.age
                
        return survivors
    
    def _update_fields(self):
        """Actualiza todos los campos dinámicos"""
        # Actualizar dinámicas intrínsecas
        self.field_dynamics.update()
        
        # Crear perturbaciones de los depredadores
        perturbations = []
        for pred in self.predators:
            if pred.is_attacking:
                perturbations.append({
                    'position': pred.position,
                    'amplitude': pred.light_intensity,
                    'radius': pred.light_radius
                })
                
        # Actualizar campo topológico con perturbaciones
        self.topology_engine.update_field(perturbations)
        
    def _update_predators(self):
        """Actualiza el comportamiento de los depredadores"""
        for pred in self.predators:
            # Buscar presa más cercana
            if self.protozoa:
                distances = [np.linalg.norm(pred.position - p.position) 
                           for p in self.protozoa if p.alive]
                if distances:
                    closest_idx = np.argmin(distances)
                    closest_prey = self.protozoa[closest_idx]
                    
                    # Moverse hacia la presa
                    direction = closest_prey.position - pred.position
                    direction_norm = direction / (np.linalg.norm(direction) + 0.01)
                    
                    pred.velocity = direction_norm * pred.speed
                    pred.position += pred.velocity * self.dt
                    
                    # Aplicar fronteras
                    pred.position = self.topology_engine.apply_toroidal_boundary(pred.position)
                    
                    # Atacar si está cerca
                    if distances[closest_idx] < pred.attack_range:
                        pred.attack()
                        # Añadir estallido de luz
                        self.field_dynamics.add_light_burst(
                            pred.position, 
                            pred.light_intensity,
                            pred.light_radius
                        )
    
    def _update_protozoa(self):
        """Actualiza el comportamiento de los protozoos"""
        dead_protozoa = []
        
        for i, proto in enumerate(self.protozoa):
            if not proto.alive:
                continue
                
            # Obtener estado local del campo
            local_field = self.field_dynamics.get_local_field_state(proto.position)
            
            # Verificar muerte por luz
            if local_field['light_intensity'] > proto.light_tolerance:
                proto.die("light_exposure")
                dead_protozoa.append(i)
                self.stats['deaths'] += 1
                continue
            
            # Actualizar movimiento basado en topología
            proto.sense_environment(local_field)
            proto.move(self.topology_engine, self.dt)
            
            # Consumir nutrientes
            self.field_dynamics.consume_nutrients(proto.position, 0.01)
            proto.energy += 0.01  # Ganancia de energía
            
            # Actualizar edad y energía
            proto.age += self.dt
            proto.energy -= 0.005  # Costo metabólico
            
            # Verificar muerte por energía
            if proto.energy <= 0:
                proto.die("starvation")
                dead_protozoa.append(i)
                self.stats['deaths'] += 1
                continue
                
            # Actualizar estadísticas
            distance = np.linalg.norm(proto.velocity) * self.dt
            self.stats['total_distance_traveled'] += distance
        
        # Eliminar muertos
        for idx in sorted(dead_protozoa, reverse=True):
            del self.protozoa[idx]
            
        self.stats['current_population'] = len(self.protozoa)
    
    def _check_reproduction(self):
        """Verifica condiciones de reproducción"""
        new_protozoa = []
        
        for proto in self.protozoa:
            if proto.can_reproduce():
                # Generar semilla topológica
                seed = proto.generate_topological_seed()
                
                # Crear descendencia
                child_pos = proto.position + np.random.normal(0, 10, size=2)
                child_pos = self.topology_engine.apply_toroidal_boundary(child_pos)
                
                # El hijo hereda la semilla topológica
                from .protozoa import Protozoa
                child = Protozoa(position=child_pos, seed=seed)
                new_protozoa.append(child)
                
                # Reset energía del padre
                proto.energy *= 0.5
                proto.reproduction_cooldown = 100
                
                self.stats['births'] += 1
        
        # Añadir nuevos protozoos
        self.protozoa.extend(new_protozoa)
        self.stats['current_population'] = len(self.protozoa)
    
    def visualize(self, save: Optional[str] = None, show: bool = True):
        """Visualiza el estado actual del mundo"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            
        self.ax.clear()
        
        # Mostrar campo de luz como heatmap
        light_field = self.field_dynamics.light_field
        im = self.ax.imshow(light_field.T, cmap='hot', alpha=0.6, 
                           extent=[0, self.size[0], 0, self.size[1]],
                           origin='lower', vmin=0, vmax=1)
        
        # Dibujar protozoos
        for proto in self.protozoa:
            if proto.alive:
                circle = Circle(proto.position, 5, color='cyan', alpha=0.8)
                self.ax.add_patch(circle)
                
                # Mostrar trayectoria reciente
                if len(proto.state.trajectory_history) > 10:
                    recent_traj = np.array(proto.state.trajectory_history[-50:])
                    self.ax.plot(recent_traj[:, 0], recent_traj[:, 1], 
                               'c-', alpha=0.3, linewidth=0.5)
        
        # Dibujar depredadores
        for pred in self.predators:
            if pred.is_attacking:
                # Mostrar zona de luz
                light_circle = Circle(pred.position, pred.light_radius, 
                                    color='yellow', alpha=0.3)
                self.ax.add_patch(light_circle)
            
            # El depredador mismo
            pred_circle = Circle(pred.position, 10, color='red', alpha=0.9)
            self.ax.add_patch(pred_circle)
        
        # Configurar ejes
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        self.ax.set_aspect('equal')
        self.ax.set_title(f'TopoLife World - Time: {self.time:.1f} - Population: {len(self.protozoa)}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Añadir estadísticas
        stats_text = (f"Births: {self.stats['births']}\n"
                     f"Deaths: {self.stats['deaths']}\n"
                     f"Max Survival: {self.stats['max_survival_time']:.1f}")
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return self.fig, self.ax
    
    def animate(self, frames: int = 1000, interval: int = 50):
        """Crea una animación del mundo"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            
        def update(frame):
            # Simular un paso
            self._update_fields()
            self._update_predators()
            self._update_protozoa()
            self._check_reproduction()
            self.time += self.dt
            
            # Redibujar
            self.visualize(show=False)
            
            return self.ax.artists
        
        anim = animation.FuncAnimation(self.fig, update, frames=frames,
                                     interval=interval, blit=False)
        
        return anim
    
    def get_world_state(self) -> Dict:
        """Retorna el estado completo del mundo para análisis"""
        return {
            'time': self.time,
            'population': len(self.protozoa),
            'predators': len(self.predators),
            'field_energy': self.field_dynamics.compute_field_energy(),
            'stats': self.stats.copy(),
            'protozoa_states': [p.get_state() for p in self.protozoa if p.alive],
            'average_age': np.mean([p.age for p in self.protozoa]) if self.protozoa else 0,
            'average_energy': np.mean([p.energy for p in self.protozoa]) if self.protozoa else 0
        }