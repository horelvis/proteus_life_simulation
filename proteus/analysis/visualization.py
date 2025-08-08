"""
Visualización del mundo y análisis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import networkx as nx
from pathlib import Path


class ProteusVisualizer:
    """
    Sistema de visualización para el mundo Proteus
    """
    
    def __init__(self, world_size: Tuple[int, int] = (1000, 1000)):
        self.world_size = world_size
        self.fig = None
        self.axes = None
        self.color_palette = {
            'protozoa': '#00CED1',  # Dark turquoise
            'predator': '#DC143C',  # Crimson
            'light': '#FFD700',     # Gold
            'nutrient': '#90EE90',  # Light green
            'trajectory': '#4169E1', # Royal blue
            'danger_zone': '#FF6347' # Tomato
        }
        
    def create_world_visualization(self, world) -> Tuple:
        """
        Crea una visualización completa del mundo
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        # Panel 1: Vista principal del mundo
        self._draw_world_state(axes[0], world)
        
        # Panel 2: Campo de luz y peligro
        self._draw_field_visualization(axes[1], world)
        
        # Panel 3: Análisis de trayectorias
        self._draw_trajectory_analysis(axes[2], world)
        
        # Panel 4: Estadísticas
        self._draw_statistics(axes[3], world)
        
        plt.tight_layout()
        
        self.fig = fig
        self.axes = axes
        
        return fig, axes
    
    def _draw_world_state(self, ax, world):
        """
        Dibuja el estado principal del mundo
        """
        ax.clear()
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        ax.set_aspect('equal')
        ax.set_title(f'Proteus World - Time: {world.time:.1f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Fondo con gradiente de nutrientes
        if hasattr(world.field_dynamics, 'nutrient_field'):
            nutrient_field = world.field_dynamics.nutrient_field
            im = ax.imshow(nutrient_field.T, cmap='Greens', alpha=0.3,
                          extent=[0, self.world_size[0], 0, self.world_size[1]],
                          origin='lower', vmin=0, vmax=0.5)
        
        # Dibujar protozoos
        for proto in world.protozoa:
            if proto.alive:
                # Color basado en fenotipo emergente
                phenotype = proto.get_state().get('phenotype', 'Primitive')
                if 'Visual' in phenotype:
                    color = self.color_palette['protozoa']  # Azul turquesa para visual
                elif 'Chemical' in phenotype:
                    color = '#90EE90'  # Verde claro para químico
                elif 'Fast' in phenotype:
                    color = '#FF69B4'  # Rosa para rápido
                elif 'Armored' in phenotype:
                    color = '#8B4513'  # Marrón para blindado
                else:
                    # Color por generación para primitivos
                    color_intensity = min(1.0, proto.generation / 10)
                    color = plt.cm.Blues(0.3 + color_intensity * 0.7)
                
                # Tamaño basado en capacidades totales
                capabilities = proto.get_state().get('capabilities', {})
                total_capability = sum(capabilities.values()) if capabilities else 1
                size = 5 + min(15, total_capability * 3)
                
                circle = Circle(proto.position, size, 
                              color=color, alpha=0.8, 
                              edgecolor='darkblue', linewidth=1)
                ax.add_patch(circle)
                
                # Dirección de movimiento
                if np.linalg.norm(proto.velocity) > 0.1:
                    dx, dy = proto.velocity / np.linalg.norm(proto.velocity) * size * 2
                    ax.arrow(proto.position[0], proto.position[1], dx, dy,
                           head_width=size/2, head_length=size/3, 
                           fc=color, ec=color, alpha=0.6)
        
        # Dibujar depredadores
        for pred in world.predators:
            # Depredador
            pred_circle = Circle(pred.position, 12, 
                               color=self.color_palette['predator'], 
                               alpha=0.9, edgecolor='darkred', linewidth=2)
            ax.add_patch(pred_circle)
            
            # Zona de ataque
            if pred.is_attacking:
                light_circle = Circle(pred.position, pred.light_radius,
                                    color=self.color_palette['light'],
                                    alpha=0.2, linestyle='--', fill=True)
                ax.add_patch(light_circle)
                
                # Rayos de luz
                for angle in np.linspace(0, 2*np.pi, 8):
                    x_end = pred.position[0] + pred.light_radius * np.cos(angle)
                    y_end = pred.position[1] + pred.light_radius * np.sin(angle)
                    ax.plot([pred.position[0], x_end], [pred.position[1], y_end],
                          color=self.color_palette['light'], alpha=0.5, linewidth=1)
        
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.color_palette['protozoa'], 
                      markersize=10, label='Protozoa'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.color_palette['predator'], 
                      markersize=10, label='Predator'),
            plt.Line2D([0], [0], color=self.color_palette['light'], 
                      linewidth=3, label='Light attack')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Grid sutil
        ax.grid(True, alpha=0.2)
        
    def _draw_field_visualization(self, ax, world):
        """
        Visualiza los campos del entorno
        """
        ax.clear()
        ax.set_title('Environmental Fields', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Campo combinado: luz + potencial
        combined_field = (world.field_dynamics.light_field * 2 + 
                         world.field_dynamics.potential_field * 0.5)
        
        # Mostrar como heatmap
        im = ax.imshow(combined_field.T, cmap='hot', 
                      extent=[0, self.world_size[0], 0, self.world_size[1]],
                      origin='lower', vmin=0, vmax=2)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Field Intensity', rotation=270, labelpad=15)
        
        # Contornos de peligro
        danger_levels = [0.5, 1.0, 1.5]
        contours = ax.contour(combined_field.T, levels=danger_levels,
                            colors='white', alpha=0.5, linewidths=1,
                            extent=[0, self.world_size[0], 0, self.world_size[1]])
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Superponer posiciones de organismos
        if world.protozoa:
            positions = np.array([p.position for p in world.protozoa if p.alive])
            if len(positions) > 0:
                ax.scatter(positions[:, 0], positions[:, 1], 
                         c='cyan', s=20, alpha=0.6, edgecolors='blue')
        
    def _draw_trajectory_analysis(self, ax, world):
        """
        Analiza y visualiza trayectorias
        """
        ax.clear()
        ax.set_title('Movement Patterns & Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        
        # Seleccionar organismos representativos
        if world.protozoa:
            # Tomar hasta 5 organismos con las trayectorias más largas
            sorted_orgs = sorted(world.protozoa, 
                               key=lambda x: len(x.state.trajectory_history), 
                               reverse=True)[:5]
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_orgs)))
            
            for i, org in enumerate(sorted_orgs):
                if len(org.state.trajectory_history) > 10:
                    traj = np.array(org.state.trajectory_history[-100:])  # Últimos 100 puntos
                    
                    # Crear gradiente de color para mostrar tiempo
                    points = traj.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    # Color con gradiente temporal
                    lc = LineCollection(segments, cmap='viridis', alpha=0.7)
                    lc.set_array(np.linspace(0, 1, len(segments)))
                    lc.set_linewidth(2)
                    ax.add_collection(lc)
                    
                    # Marcar posición actual
                    # Mostrar fenotipo en lugar de generación
                    phenotype = org.get_state().get('phenotype', 'Primitive')
                    ax.scatter(org.position[0], org.position[1], 
                             color=colors[i], s=100, marker='*', 
                             edgecolors='black', linewidth=1,
                             label=f'{phenotype}')
                    
                    # Mostrar zonas evitadas si existen
                    if hasattr(org, 'danger_memory') and len(org.danger_memory) > 0:
                        for danger_zone in org.danger_memory[:5]:  # Máximo 5 zonas
                            danger_circle = Circle(danger_zone, 30,
                                                 color=self.color_palette['danger_zone'],
                                                 alpha=0.2, linestyle='--', fill=False)
                            ax.add_patch(danger_circle)
            
            ax.legend(loc='upper right', fontsize=8)
        
        ax.grid(True, alpha=0.2)
        
    def _draw_statistics(self, ax, world):
        """
        Muestra estadísticas del mundo
        """
        ax.clear()
        ax.set_title('World Statistics', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Recopilar estadísticas
        stats = world.stats.copy()
        
        # Estadísticas adicionales
        if world.protozoa:
            ages = [p.age for p in world.protozoa if p.alive]
            generations = [p.generation for p in world.protozoa]
            energies = [p.energy for p in world.protozoa]
            
            stats.update({
                'avg_age': np.mean(ages) if ages else 0,
                'max_generation': max(generations) if generations else 0,
                'avg_energy': np.mean(energies) if energies else 0,
                'organisms_with_seeds': sum(1 for p in world.protozoa if p.seed is not None)
            })
        
        # Mostrar como tabla
        y_pos = 0.95
        for key, value in stats.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                text = f"{label}: {value:.2f}"
            else:
                text = f"{label}: {value}"
                
            ax.text(0.05, y_pos, text, fontsize=12, 
                   transform=ax.transAxes, verticalalignment='top')
            y_pos -= 0.08
            
        # Mini gráfico de población
        if hasattr(world, 'population_history'):
            ax2 = ax.inset_axes([0.5, 0.1, 0.45, 0.4])
            ax2.plot(world.population_history, 'b-', linewidth=2)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Population')
            ax2.grid(True, alpha=0.3)
            
    def create_evolution_dashboard(self, evolution_data: Dict) -> Tuple:
        """
        Crea un dashboard de evolución multigeneracional
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Panel 1: Supervivencia por generación
        self._plot_survival_trends(axes[0], evolution_data)
        
        # Panel 2: Características topológicas
        self._plot_topological_evolution(axes[1], evolution_data)
        
        # Panel 3: Árbol filogenético
        self._plot_phylogenetic_tree(axes[2], evolution_data)
        
        # Panel 4: Comportamientos emergentes
        self._plot_emergent_behaviors(axes[3], evolution_data)
        
        # Panel 5: Métricas de información
        self._plot_information_metrics(axes[4], evolution_data)
        
        # Panel 6: Predicciones
        self._plot_predictions(axes[5], evolution_data)
        
        plt.tight_layout()
        return fig, axes
    
    def _plot_survival_trends(self, ax, data):
        """
        Grafica tendencias de supervivencia
        """
        ax.clear()
        ax.set_title('Survival Trends Across Generations', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Survival Time')
        
        if 'generation_stats' in data:
            generations = sorted(data['generation_stats'].keys())
            avg_survival = [data['generation_stats'][g].get('avg_survival_time', 0) 
                          for g in generations]
            max_survival = [data['generation_stats'][g].get('max_survival_time', 0) 
                          for g in generations]
            
            ax.plot(generations, avg_survival, 'b-', linewidth=2, 
                   label='Average', marker='o')
            ax.plot(generations, max_survival, 'r--', linewidth=2, 
                   label='Maximum', marker='^')
            
            # Área de desviación estándar
            if 'std_survival_time' in data['generation_stats'][generations[0]]:
                std_survival = [data['generation_stats'][g].get('std_survival_time', 0) 
                              for g in generations]
                ax.fill_between(generations,
                              [avg_survival[i] - std_survival[i] for i in range(len(generations))],
                              [avg_survival[i] + std_survival[i] for i in range(len(generations))],
                              alpha=0.2, color='blue')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    def _plot_topological_evolution(self, ax, data):
        """
        Evolución de características topológicas
        """
        ax.clear()
        ax.set_title('Topological Feature Evolution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Feature Value')
        
        if 'generation_stats' in data:
            generations = sorted(data['generation_stats'].keys())
            
            # Extraer métricas topológicas
            features = ['avg_homology_persistence', 'avg_homology_loops', 
                       'curvature_diversity']
            colors = ['blue', 'green', 'red']
            
            for feat, color in zip(features, colors):
                values = []
                for g in generations:
                    val = data['generation_stats'][g].get(feat, 0)
                    values.append(val)
                    
                if any(v != 0 for v in values):
                    label = feat.replace('avg_homology_', '').replace('_', ' ').title()
                    ax.plot(generations, values, color=color, linewidth=2, 
                           label=label, marker='o', markersize=4)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    def _plot_phylogenetic_tree(self, ax, data):
        """
        Visualiza el árbol filogenético
        """
        ax.clear()
        ax.set_title('Phylogenetic Relationships', fontsize=12, fontweight='bold')
        
        if 'phylogeny' in data and data['phylogeny']:
            # Crear grafo
            G = nx.DiGraph()
            
            for parent, children in data['phylogeny'].items():
                for child in children:
                    G.add_edge(parent, child)
            
            if G.number_of_nodes() > 0:
                # Layout jerárquico
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Dibujar
                nx.draw(G, pos, ax=ax, 
                       with_labels=False,
                       node_color='lightblue',
                       node_size=300,
                       edge_color='gray',
                       arrows=True,
                       arrowsize=10)
                
                # Añadir etiquetas de generación si es posible
                if G.number_of_nodes() < 50:
                    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            else:
                ax.text(0.5, 0.5, 'No phylogenetic data available', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No phylogenetic data available', 
                   ha='center', va='center', transform=ax.transAxes)
                   
    def _plot_emergent_behaviors(self, ax, data):
        """
        Visualiza comportamientos emergentes
        """
        ax.clear()
        ax.set_title('Emergent Behaviors Over Time', fontsize=12, fontweight='bold')
        
        if 'behavior_timeline' in data:
            behaviors = defaultdict(list)
            times = []
            
            for entry in data['behavior_timeline']:
                times.append(entry['time'])
                for behavior in entry.get('behaviors', []):
                    behaviors[behavior].append(entry['time'])
            
            # Crear gráfico de eventos
            y_labels = list(behaviors.keys())
            y_pos = range(len(y_labels))
            
            for i, behavior in enumerate(y_labels):
                ax.scatter(behaviors[behavior], [i] * len(behaviors[behavior]),
                         s=50, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'No behavior data available', 
                   ha='center', va='center', transform=ax.transAxes)
                   
    def _plot_information_metrics(self, ax, data):
        """
        Métricas de información y herencia
        """
        ax.clear()
        ax.set_title('Information Transfer Metrics', fontsize=12, fontweight='bold')
        
        if 'information_metrics' in data:
            metrics = data['information_metrics']
            
            # Crear gráfico de barras
            metric_names = list(metrics.keys())
            values = list(metrics.values())
            
            bars = ax.bar(range(len(metric_names)), values, 
                         color=['blue', 'green', 'red', 'orange'][:len(metric_names)])
            
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels([name.replace('_', ' ').title() 
                               for name in metric_names], rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
            
            # Añadir valores encima de las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No information metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
                   
    def _plot_predictions(self, ax, data):
        """
        Visualiza predicciones evolutivas
        """
        ax.clear()
        ax.set_title('Evolutionary Predictions', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        if 'predictions' in data:
            pred = data['predictions']
            
            y_pos = 0.9
            
            # Tendencias de movimiento
            if 'movement_trend' in pred:
                ax.text(0.05, y_pos, 'Movement Trends:', fontsize=12, 
                       fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.1
                
                for key, value in pred['movement_trend'].items():
                    direction = "↑" if value > 0 else "↓" if value < 0 else "→"
                    text = f"  {key.replace('_', ' ').title()}: {direction} {abs(value):.2f}"
                    ax.text(0.1, y_pos, text, fontsize=10, transform=ax.transAxes)
                    y_pos -= 0.08
                    
            # Comportamientos emergentes predichos
            if 'emerging_behaviors' in pred and pred['emerging_behaviors']:
                y_pos -= 0.05
                ax.text(0.05, y_pos, 'Predicted Behaviors:', fontsize=12, 
                       fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.1
                
                for behavior in pred['emerging_behaviors']:
                    ax.text(0.1, y_pos, f"  • {behavior}", fontsize=10, 
                           transform=ax.transAxes)
                    y_pos -= 0.08
        else:
            ax.text(0.5, 0.5, 'No predictions available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def save_animation(self, world, filename: str, frames: int = 100, 
                      interval: int = 50):
        """
        Guarda una animación del mundo
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            # Simular un paso
            world._update_fields()
            world._update_predators()
            world._update_protozoa()
            world._check_reproduction()
            world.time += world.dt
            
            # Redibujar
            self._draw_world_state(ax, world)
            
            return ax.artists
        
        anim = animation.FuncAnimation(fig, update, frames=frames,
                                     interval=interval, blit=False)
        
        # Guardar
        anim.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        
        print(f"Animation saved to {filename}")
    
    def create_analysis_report(self, world, metrics: Dict, 
                             save_path: Optional[Path] = None):
        """
        Crea un reporte visual completo
        """
        # Crear figura con múltiples páginas
        from matplotlib.backends.backend_pdf import PdfPages
        
        if save_path is None:
            save_path = Path(f"proteus_report_gen_{world.time:.0f}.pdf")
            
        with PdfPages(save_path) as pdf:
            # Página 1: Estado del mundo
            fig1, _ = self.create_world_visualization(world)
            pdf.savefig(fig1)
            plt.close(fig1)
            
            # Página 2: Dashboard de evolución
            if 'evolution_data' in metrics:
                fig2, _ = self.create_evolution_dashboard(metrics['evolution_data'])
                pdf.savefig(fig2)
                plt.close(fig2)
                
            # Página 3: Métricas detalladas
            fig3 = plt.figure(figsize=(12, 16))
            ax = fig3.add_subplot(111)
            ax.axis('off')
            
            # Convertir métricas a texto
            report_text = self._format_metrics_report(metrics)
            ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig3)
            plt.close(fig3)
            
        print(f"Analysis report saved to {save_path}")
        
    def _format_metrics_report(self, metrics: Dict) -> str:
        """
        Formatea las métricas como texto
        """
        report = "PROTEUS ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        def format_dict(d, indent=0):
            text = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    text += " " * indent + f"{key}:\n"
                    text += format_dict(value, indent + 2)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    text += " " * indent + f"{key}: {value[:3]}...\n"
                elif isinstance(value, float):
                    text += " " * indent + f"{key}: {value:.3f}\n"
                else:
                    text += " " * indent + f"{key}: {value}\n"
            return text
            
        return report + format_dict(metrics)