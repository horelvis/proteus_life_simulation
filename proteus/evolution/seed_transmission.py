"""
Transmisión de semillas topológicas entre generaciones
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import pickle
from pathlib import Path

from ..core.inheritance import TopologicalSeed, TopologicalInheritance


class SeedTransmission:
    """
    Gestiona la transmisión de información topológica entre generaciones
    Sin genes, sin ADN - solo patrones de experiencia
    """
    
    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.3):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.inheritance_system = TopologicalInheritance()
        
        # Banco de semillas (memoria generacional)
        self.seed_bank = []
        self.generation_stats = {}
        
    def collect_seeds(self, survivors: List) -> List[TopologicalSeed]:
        """
        Colecta semillas de todos los supervivientes
        """
        seeds = []
        
        for organism in survivors:
            if organism.alive and organism.age > 10:  # Solo organismos maduros
                seed = organism.generate_topological_seed()
                seeds.append(seed)
                
                # Añadir al banco de semillas
                self.seed_bank.append({
                    'seed': seed,
                    'fitness': organism.age,  # Longevidad como proxy de fitness
                    'generation': organism.generation
                })
                
        return seeds
    
    def select_parents(self, seeds: List[TopologicalSeed], 
                      selection_pressure: float = 0.5) -> List[TopologicalSeed]:
        """
        Selecciona semillas para la próxima generación
        No es selección genética - es selección de experiencias exitosas
        """
        if not seeds:
            return []
            
        # Calcular "fitness" basado en tiempo de supervivencia
        fitness_scores = [seed.survival_time for seed in seeds]
        
        # Normalizar scores
        min_fitness = min(fitness_scores)
        max_fitness = max(fitness_scores)
        if max_fitness > min_fitness:
            normalized_fitness = [(f - min_fitness) / (max_fitness - min_fitness) 
                                for f in fitness_scores]
        else:
            normalized_fitness = [1.0] * len(seeds)
            
        # Aplicar presión de selección
        selection_weights = [f ** selection_pressure for f in normalized_fitness]
        total_weight = sum(selection_weights)
        
        if total_weight > 0:
            probabilities = [w / total_weight for w in selection_weights]
        else:
            probabilities = [1.0 / len(seeds)] * len(seeds)
            
        # Seleccionar padres con reemplazo
        n_parents = max(2, int(len(seeds) * 0.5))
        selected_indices = np.random.choice(
            len(seeds), 
            size=n_parents, 
            p=probabilities,
            replace=True
        )
        
        return [seeds[i] for i in selected_indices]
    
    def generate_offspring_seeds(self, parent_seeds: List[TopologicalSeed], 
                               n_offspring: int) -> List[TopologicalSeed]:
        """
        Genera semillas para la descendencia combinando y mutando semillas parentales
        """
        if not parent_seeds:
            return []
            
        offspring_seeds = []
        
        for _ in range(n_offspring):
            # Decidir tipo de reproducción
            if len(parent_seeds) > 1 and np.random.random() < self.crossover_rate:
                # Crossover - combinar dos padres
                parent1 = np.random.choice(parent_seeds)
                parent2 = np.random.choice(parent_seeds)
                
                if parent1.parent_id != parent2.parent_id:  # Evitar auto-crossover
                    child_seed = self.inheritance_system.crossover(parent1, parent2)
                else:
                    child_seed = self._mutate_seed(parent1)
            else:
                # Reproducción asexual con mutación
                parent = np.random.choice(parent_seeds)
                child_seed = self._mutate_seed(parent)
                
            offspring_seeds.append(child_seed)
            
        return offspring_seeds
    
    def _mutate_seed(self, seed: TopologicalSeed) -> TopologicalSeed:
        """
        Aplica mutaciones a una semilla topológica
        Las mutaciones alteran la interpretación de la experiencia
        """
        # Copiar la semilla
        mutated = TopologicalSeed(
            path_homology=seed.path_homology.copy(),
            curvature_spectrum=seed.curvature_spectrum.copy(),
            avoided_zones=seed.avoided_zones.copy(),
            survival_time=seed.survival_time,
            field_interactions={k: v.copy() for k, v in seed.field_interactions.items()},
            generation=seed.generation + 1,
            parent_id=seed.parent_id + "_mut",
            creation_time=seed.creation_time
        )
        
        # Mutar homología
        if np.random.random() < self.mutation_rate:
            for key in mutated.path_homology:
                mutated.path_homology[key] *= np.random.normal(1.0, 0.1)
                
        # Mutar espectro de curvatura
        if np.random.random() < self.mutation_rate:
            noise = np.random.normal(0, 0.05, size=len(mutated.curvature_spectrum))
            mutated.curvature_spectrum += noise
            mutated.curvature_spectrum = np.clip(mutated.curvature_spectrum, 0, 1)
            
        # Mutar zonas evitadas
        if np.random.random() < self.mutation_rate:
            if len(mutated.avoided_zones) > 0:
                # Eliminar una zona aleatoria
                if np.random.random() < 0.3:
                    idx = np.random.randint(len(mutated.avoided_zones))
                    mutated.avoided_zones.pop(idx)
                # O añadir ruido a las posiciones
                else:
                    for i in range(len(mutated.avoided_zones)):
                        x, y = mutated.avoided_zones[i]
                        mutated.avoided_zones[i] = (
                            x + np.random.normal(0, 10),
                            y + np.random.normal(0, 10)
                        )
                        
        # Mutar interacciones de campo
        if np.random.random() < self.mutation_rate:
            for key in mutated.field_interactions:
                noise = np.random.normal(0, 0.1, size=len(mutated.field_interactions[key]))
                mutated.field_interactions[key] += noise
                
        return mutated
    
    def analyze_generation(self, generation: int, seeds: List[TopologicalSeed]) -> Dict:
        """
        Analiza las características de una generación
        """
        if not seeds:
            return {}
            
        stats = {
            'generation': generation,
            'population_size': len(seeds),
            'avg_survival_time': np.mean([s.survival_time for s in seeds]),
            'max_survival_time': np.max([s.survival_time for s in seeds]),
            'min_survival_time': np.min([s.survival_time for s in seeds]),
            'std_survival_time': np.std([s.survival_time for s in seeds])
        }
        
        # Analizar homología promedio
        homology_keys = list(seeds[0].path_homology.keys())
        for key in homology_keys:
            values = [s.path_homology[key] for s in seeds]
            stats[f'avg_homology_{key}'] = np.mean(values)
            stats[f'std_homology_{key}'] = np.std(values)
            
        # Analizar convergencia del espectro de curvatura
        spectra = np.array([s.curvature_spectrum for s in seeds])
        stats['curvature_diversity'] = np.mean(np.std(spectra, axis=0))
        
        # Guardar estadísticas
        self.generation_stats[generation] = stats
        
        return stats
    
    def save_seed_bank(self, filepath: Path):
        """Guarda el banco de semillas para análisis posterior"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'seed_bank': self.seed_bank,
                'generation_stats': self.generation_stats,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }, f)
            
    def load_seed_bank(self, filepath: Path):
        """Carga un banco de semillas guardado"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.seed_bank = data['seed_bank']
            self.generation_stats = data['generation_stats']
            self.mutation_rate = data['mutation_rate']
            self.crossover_rate = data['crossover_rate']
            
    def get_elite_seeds(self, n: int = 10) -> List[TopologicalSeed]:
        """
        Obtiene las mejores semillas del banco
        Las "élites" son aquellas con mayor tiempo de supervivencia
        """
        if not self.seed_bank:
            return []
            
        # Ordenar por fitness (tiempo de supervivencia)
        sorted_bank = sorted(self.seed_bank, key=lambda x: x['fitness'], reverse=True)
        
        # Retornar las mejores n semillas
        return [entry['seed'] for entry in sorted_bank[:n]]
    
    def compute_phylogeny(self) -> Dict[str, List[str]]:
        """
        Construye un árbol filogenético basado en las relaciones parent_id
        """
        phylogeny = {}
        
        for entry in self.seed_bank:
            seed = entry['seed']
            parent_id = seed.parent_id.split('_')[0]  # Remover sufijos de mutación
            
            if parent_id not in phylogeny:
                phylogeny[parent_id] = []
                
            # Extraer ID único del organismo
            if 'x' in seed.parent_id:  # Crossover
                parents = seed.parent_id.split('x')
                for p in parents:
                    if p.split('_')[0] not in phylogeny:
                        phylogeny[p.split('_')[0]] = []
            else:
                child_id = seed.parent_id
                if parent_id != child_id:
                    phylogeny[parent_id].append(child_id)
                    
        return phylogeny