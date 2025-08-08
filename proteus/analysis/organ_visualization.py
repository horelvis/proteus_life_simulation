"""
Visualización de la evolución de órganos topológicos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from typing import List, Dict, Tuple
import networkx as nx


class OrganEvolutionVisualizer:
    """
    Visualiza la emergencia y evolución de órganos topológicos
    """
    
    def __init__(self):
        self.organ_colors = {
            'photosensor': '#FFD700',      # Oro
            'pigment_spot': '#FF8C00',     # Naranja oscuro
            'crystallin': '#1E90FF',       # Azul brillante
            'chemoreceptor': '#32CD32',    # Verde lima
            'flagellum': '#FF1493',        # Rosa profundo
            'cilia': '#FF69B4',            # Rosa claro
            'pseudopod': '#8A2BE2',        # Violeta
            'membrane': '#8B4513',         # Marrón
            'vacuole': '#00CED1',          # Turquesa oscuro
            'nerve_net': '#DC143C'         # Carmesí
        }
        
    def create_organ_evolution_chart(self, population_history: List[Dict]) -> Tuple:
        """
        Crea un gráfico de la evolución de órganos a través del tiempo
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Extraer datos de órganos
        generations = []
        organ_prevalence = {organ: [] for organ in self.organ_colors.keys()}
        avg_functionalities = {organ: [] for organ in self.organ_colors.keys()}
        phenotype_counts = {}
        
        for gen_data in population_history:
            generations.append(gen_data['generation'])
            
            # Contar órganos por tipo
            organ_count = {organ: 0 for organ in self.organ_colors.keys()}
            organ_functionality = {organ: [] for organ in self.organ_colors.keys()}
            phenotypes = {}
            
            for organism in gen_data.get('organisms', []):
                if hasattr(organism, 'organ_system'):
                    # Contar órganos
                    for organ in organism.organ_system.organs:
                        if organ.functionality > 0.1:
                            organ_count[organ.type] += 1
                            organ_functionality[organ.type].append(organ.functionality)
                    
                    # Contar fenotipos
                    phenotype = organism.organ_system.describe_phenotype()
                    phenotypes[phenotype] = phenotypes.get(phenotype, 0) + 1
            
            # Guardar prevalencia
            total_organisms = len(gen_data.get('organisms', []))
            for organ_type in self.organ_colors.keys():
                prevalence = organ_count[organ_type] / total_organisms if total_organisms > 0 else 0
                organ_prevalence[organ_type].append(prevalence)
                
                # Funcionalidad promedio
                if organ_functionality[organ_type]:
                    avg_func = np.mean(organ_functionality[organ_type])
                else:
                    avg_func = 0
                avg_functionalities[organ_type].append(avg_func)
            
            phenotype_counts[gen_data['generation']] = phenotypes
        
        # Gráfico 1: Prevalencia de órganos
        ax1.set_title('Organ Prevalence Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Prevalence (%)')
        
        for organ_type, prevalences in organ_prevalence.items():
            if any(p > 0 for p in prevalences):
                ax1.plot(generations, [p * 100 for p in prevalences], 
                        color=self.organ_colors[organ_type],
                        label=organ_type.replace('_', ' ').title(),
                        linewidth=2, marker='o', markersize=4)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Funcionalidad promedio
        ax2.set_title('Average Organ Functionality', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Functionality')
        
        for organ_type, functionalities in avg_functionalities.items():
            if any(f > 0 for f in functionalities):
                ax2.plot(generations, functionalities,
                        color=self.organ_colors[organ_type],
                        label=organ_type.replace('_', ' ').title(),
                        linewidth=2, alpha=0.7)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Distribución de fenotipos
        ax3.set_title('Phenotype Distribution Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Count')
        
        # Preparar datos para gráfico de áreas apiladas
        all_phenotypes = set()
        for phenotypes in phenotype_counts.values():
            all_phenotypes.update(phenotypes.keys())
        
        phenotype_data = {ph: [] for ph in all_phenotypes}
        for gen in generations:
            gen_phenotypes = phenotype_counts.get(gen, {})
            for ph in all_phenotypes:
                phenotype_data[ph].append(gen_phenotypes.get(ph, 0))
        
        # Crear gráfico de áreas apiladas
        bottom = np.zeros(len(generations))
        phenotype_colors = plt.cm.Set3(np.linspace(0, 1, len(all_phenotypes)))
        
        for i, (phenotype, counts) in enumerate(phenotype_data.items()):
            ax3.bar(generations, counts, bottom=bottom, 
                   label=phenotype, color=phenotype_colors[i],
                   width=0.8)
            bottom += np.array(counts)
        
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, (ax1, ax2, ax3)
    
    def visualize_organism_organs(self, organism) -> Tuple:
        """
        Visualiza los órganos de un organismo individual
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Dibujar el organismo como círculo base
        organism_circle = Circle((0, 0), 1, color='lightblue', 
                               alpha=0.5, edgecolor='darkblue', linewidth=2)
        ax.add_patch(organism_circle)
        
        # Dibujar órganos
        if hasattr(organism, 'organ_system'):
            active_organs = [o for o in organism.organ_system.organs 
                           if o.functionality > 0.1]
            
            n_organs = len(active_organs)
            if n_organs > 0:
                # Distribuir órganos alrededor del organismo
                angles = np.linspace(0, 2*np.pi, n_organs, endpoint=False)
                
                for i, organ in enumerate(active_organs):
                    angle = angles[i]
                    
                    # Posición del órgano
                    r = 0.7 + organ.development_stage * 0.5
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    
                    # Tamaño basado en funcionalidad
                    size = 0.1 + organ.functionality * 0.3
                    
                    # Dibujar órgano
                    organ_circle = Circle((x, y), size, 
                                        color=self.organ_colors.get(organ.type, 'gray'),
                                        alpha=0.7, edgecolor='black', linewidth=1)
                    ax.add_patch(organ_circle)
                    
                    # Etiqueta
                    ax.text(x * 1.3, y * 1.3, organ.type.replace('_', '\n'),
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.7))
        
        # Información del organismo
        phenotype = organism.organ_system.describe_phenotype() if hasattr(organism, 'organ_system') else 'Primitive'
        ax.set_title(f'Organism Phenotype: {phenotype}', 
                    fontsize=14, fontweight='bold')
        
        # Mostrar capacidades
        if hasattr(organism, 'organ_system'):
            capabilities = organism.organ_system.get_total_capabilities()
            y_pos = -2
            for cap, value in capabilities.items():
                ax.text(0, y_pos, f'{cap}: {value:.2f}', 
                       ha='center', fontsize=10)
                y_pos -= 0.2
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def create_organ_network(self, population) -> Tuple:
        """
        Crea una red que muestra las relaciones entre órganos
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Construir grafo de co-ocurrencia de órganos
        G = nx.Graph()
        
        # Contar co-ocurrencias
        organ_pairs = {}
        for organism in population:
            if hasattr(organism, 'organ_system'):
                active_organs = [o.type for o in organism.organ_system.organs 
                               if o.functionality > 0.1]
                
                # Añadir nodos
                for organ in active_organs:
                    if organ not in G:
                        G.add_node(organ)
                
                # Contar pares
                for i in range(len(active_organs)):
                    for j in range(i+1, len(active_organs)):
                        pair = tuple(sorted([active_organs[i], active_organs[j]]))
                        organ_pairs[pair] = organ_pairs.get(pair, 0) + 1
        
        # Añadir aristas con peso
        for (organ1, organ2), count in organ_pairs.items():
            if count > 2:  # Solo mostrar conexiones significativas
                G.add_edge(organ1, organ2, weight=count)
        
        if G.number_of_nodes() > 0:
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Dibujar nodos
            for node in G.nodes():
                nx.draw_networkx_nodes(G, pos, [node], 
                                     node_color=self.organ_colors.get(node, 'gray'),
                                     node_size=1000)
            
            # Dibujar aristas
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edges, 
                                 width=[w/2 for w in weights],
                                 alpha=0.5)
            
            # Etiquetas
            labels = {node: node.replace('_', '\n') for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            ax.set_title('Organ Co-occurrence Network', 
                        fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No organ data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
        return fig, ax