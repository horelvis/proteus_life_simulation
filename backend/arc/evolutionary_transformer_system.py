#!/usr/bin/env python3
"""
Sistema de Transformadores Evolutivos para ARC
Los transformadores son semillas evolutivas que se transmiten, mutan y evolucionan
No hay código hardcodeado - todo emerge de la evolución
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import random
import logging
from copy import deepcopy
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Gene:
    """Gen individual que codifica una micro-operación"""
    operation: str  # 'shift', 'rotate', 'color', 'copy', 'fill', 'mirror', etc
    parameters: Dict[str, Any]  # Parámetros específicos de la operación
    strength: float = 1.0  # Fuerza del gen (0-1)
    mutation_rate: float = 0.1
    
    def mutate(self) -> 'Gene':
        """Muta el gen con cierta probabilidad"""
        if random.random() < self.mutation_rate:
            mutated = deepcopy(self)
            
            # Mutar operación (raro)
            if random.random() < 0.1:
                mutated.operation = random.choice([
                    'shift', 'rotate', 'color', 'copy', 'fill', 
                    'mirror', 'expand', 'contract', 'filter', 'blend'
                ])
            
            # Mutar parámetros (común)
            if random.random() < 0.5:
                for key in mutated.parameters:
                    if isinstance(mutated.parameters[key], (int, float)):
                        # Mutación gaussiana
                        mutated.parameters[key] += random.gauss(0, 0.2)
                    elif isinstance(mutated.parameters[key], bool):
                        # Flip ocasional
                        if random.random() < 0.2:
                            mutated.parameters[key] = not mutated.parameters[key]
            
            # Mutar fuerza
            mutated.strength = max(0, min(1, mutated.strength + random.gauss(0, 0.1)))
            
            return mutated
        return self

@dataclass
class TransformerSeed:
    """
    Semilla transformadora - contiene el código genético para una transformación
    Se transmite entre generaciones y evoluciona
    """
    generation: int
    genome: List[Gene]  # Secuencia de genes que define la transformación
    fitness: float = 0.0  # Qué tan bien resuelve puzzles
    parent_ids: List[str] = field(default_factory=list)
    mutations: int = 0
    successful_applications: int = 0
    
    def __post_init__(self):
        self.id = f"seed_{self.generation}_{random.randint(1000, 9999)}"
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Aplica la transformación codificada en el genoma"""
        result = grid.copy()
        
        for gene in self.genome:
            if gene.strength < random.random():
                continue  # Gen inactivo esta vez
            
            result = self._apply_gene(result, gene)
        
        return result
    
    def _apply_gene(self, grid: np.ndarray, gene: Gene) -> np.ndarray:
        """Aplica un gen individual al grid"""
        result = grid.copy()
        
        if gene.operation == 'shift':
            # Desplazamiento
            dx = int(gene.parameters.get('dx', 0))
            dy = int(gene.parameters.get('dy', 0))
            result = np.roll(result, (dy, dx), axis=(0, 1))
            
        elif gene.operation == 'rotate':
            # Rotación
            k = int(gene.parameters.get('k', 1)) % 4
            result = np.rot90(result, k)
            
        elif gene.operation == 'color':
            # Transformación de color
            source = gene.parameters.get('source', 1)
            target = gene.parameters.get('target', 2)
            mask = result == source
            result[mask] = target
            
        elif gene.operation == 'copy':
            # Copiar región
            src_x = int(gene.parameters.get('src_x', 0))
            src_y = int(gene.parameters.get('src_y', 0))
            dst_x = int(gene.parameters.get('dst_x', 0))
            dst_y = int(gene.parameters.get('dst_y', 0))
            size = int(gene.parameters.get('size', 1))
            
            h, w = grid.shape
            if (0 <= src_y < h and 0 <= src_x < w and 
                0 <= dst_y < h and 0 <= dst_x < w):
                try:
                    region = result[src_y:src_y+size, src_x:src_x+size]
                    if region.size > 0:
                        end_y = min(dst_y+size, h)
                        end_x = min(dst_x+size, w)
                        result[dst_y:end_y, dst_x:end_x] = region[:end_y-dst_y, :end_x-dst_x]
                except:
                    pass
                    
        elif gene.operation == 'fill':
            # Relleno condicional
            target_value = int(gene.parameters.get('target', 0))
            fill_value = int(gene.parameters.get('fill', 1))
            mode = gene.parameters.get('mode', 'exact')
            
            if mode == 'exact':
                mask = result == target_value
            elif mode == 'greater':
                mask = result > target_value
            elif mode == 'less':
                mask = result < target_value
            else:
                mask = result != 0
            
            result[mask] = fill_value
            
        elif gene.operation == 'mirror':
            # Espejo
            axis = gene.parameters.get('axis', 'horizontal')
            if axis == 'horizontal':
                result = np.fliplr(result)
            elif axis == 'vertical':
                result = np.flipud(result)
            elif axis == 'both':
                result = np.fliplr(np.flipud(result))
                
        elif gene.operation == 'expand':
            # Expansión desde puntos
            value = int(gene.parameters.get('value', 1))
            direction = gene.parameters.get('direction', 'cross')
            
            h, w = result.shape
            expansion = np.zeros_like(result)
            
            for y in range(h):
                for x in range(w):
                    if result[y, x] == value:
                        if direction == 'cross':
                            # Cruz
                            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    expansion[ny, nx] = value
                        elif direction == 'diagonal':
                            # Diagonal
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < h and 0 <= nx < w:
                                        expansion[ny, nx] = value
            
            result = np.maximum(result, expansion)
            
        elif gene.operation == 'contract':
            # Contracción/erosión
            value = int(gene.parameters.get('value', 1))
            h, w = result.shape
            contracted = result.copy()
            
            for y in range(h):
                for x in range(w):
                    if result[y, x] == value:
                        # Verificar si está en el borde
                        is_border = False
                        for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and 
                                result[ny, nx] != value):
                                is_border = True
                                break
                        if is_border:
                            contracted[y, x] = 0
            
            result = contracted
            
        elif gene.operation == 'filter':
            # Filtro por condición
            condition = gene.parameters.get('condition', 'isolated')
            
            if condition == 'isolated':
                # Eliminar píxeles aislados
                h, w = result.shape
                filtered = result.copy()
                
                for y in range(h):
                    for x in range(w):
                        if result[y, x] != 0:
                            neighbors = 0
                            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < h and 0 <= nx < w and 
                                    result[ny, nx] != 0):
                                    neighbors += 1
                            if neighbors == 0:
                                filtered[y, x] = 0
                
                result = filtered
                
        elif gene.operation == 'blend':
            # Mezcla con el original
            blend_factor = gene.parameters.get('factor', 0.5)
            result = result * blend_factor + grid * (1 - blend_factor)
            result = np.round(result).astype(grid.dtype)
        
        return result
    
    def reproduce(self, other: 'TransformerSeed') -> 'TransformerSeed':
        """Reproduce con otra semilla (crossover)"""
        # Crossover de genomas
        min_length = min(len(self.genome), len(other.genome))
        
        if min_length <= 1:
            # Si los genomas son muy pequeños, simplemente combinar
            if random.random() < 0.5:
                new_genome = self.genome + other.genome[:1]
            else:
                new_genome = other.genome + self.genome[:1]
        else:
            cut_point = random.randint(1, min_length - 1)
            
            if random.random() < 0.5:
                new_genome = self.genome[:cut_point] + other.genome[cut_point:]
            else:
                new_genome = other.genome[:cut_point] + self.genome[cut_point:]
        
        # Crear descendencia
        child = TransformerSeed(
            generation=max(self.generation, other.generation) + 1,
            genome=[gene.mutate() for gene in new_genome],
            parent_ids=[self.id, other.id]
        )
        
        child.mutations = sum(1 for i, gene in enumerate(child.genome) 
                             if i < len(new_genome) and gene != new_genome[i])
        
        return child
    
    def mutate(self) -> 'TransformerSeed':
        """Crea una versión mutada de sí misma"""
        mutated_genome = []
        
        for gene in self.genome:
            # Duplicación de gen (raro)
            if random.random() < 0.05:
                mutated_genome.append(gene.mutate())
                mutated_genome.append(gene.mutate())
            # Eliminación de gen (raro)
            elif random.random() < 0.05:
                continue
            # Mutación normal
            else:
                mutated_genome.append(gene.mutate())
        
        # Inserción de nuevo gen (raro)
        if random.random() < 0.1:
            new_gene = self._create_random_gene()
            position = random.randint(0, len(mutated_genome))
            mutated_genome.insert(position, new_gene)
        
        mutant = TransformerSeed(
            generation=self.generation + 1,
            genome=mutated_genome,
            parent_ids=[self.id]
        )
        mutant.mutations = self.mutations + 1
        
        return mutant
    
    def _create_random_gene(self) -> Gene:
        """Crea un gen aleatorio"""
        operations = ['shift', 'rotate', 'color', 'copy', 'fill', 
                     'mirror', 'expand', 'contract', 'filter', 'blend']
        
        operation = random.choice(operations)
        parameters = {}
        
        if operation == 'shift':
            parameters = {'dx': random.randint(-3, 3), 'dy': random.randint(-3, 3)}
        elif operation == 'rotate':
            parameters = {'k': random.randint(0, 3)}
        elif operation == 'color':
            parameters = {'source': random.randint(0, 9), 'target': random.randint(0, 9)}
        elif operation == 'copy':
            parameters = {
                'src_x': random.randint(0, 10), 'src_y': random.randint(0, 10),
                'dst_x': random.randint(0, 10), 'dst_y': random.randint(0, 10),
                'size': random.randint(1, 5)
            }
        elif operation == 'fill':
            parameters = {
                'target': random.randint(0, 9),
                'fill': random.randint(0, 9),
                'mode': random.choice(['exact', 'greater', 'less', 'nonzero'])
            }
        elif operation == 'mirror':
            parameters = {'axis': random.choice(['horizontal', 'vertical', 'both'])}
        elif operation == 'expand':
            parameters = {
                'value': random.randint(1, 9),
                'direction': random.choice(['cross', 'diagonal'])
            }
        elif operation == 'contract':
            parameters = {'value': random.randint(1, 9)}
        elif operation == 'filter':
            parameters = {'condition': 'isolated'}
        elif operation == 'blend':
            parameters = {'factor': random.random()}
        
        return Gene(operation=operation, parameters=parameters)

class EvolutionaryPool:
    """Pool genético de transformadores que evoluciona con el tiempo"""
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.population: List[TransformerSeed] = []
        self.generation = 0
        self.hall_of_fame: List[TransformerSeed] = []  # Mejores de todos los tiempos
        
        # Inicializar población
        self._initialize_population()
    
    def _initialize_population(self):
        """Crea población inicial con diversidad genética"""
        for _ in range(self.pool_size):
            # Genoma aleatorio de 3-7 genes
            genome_size = random.randint(3, 7)
            genome = []
            
            for _ in range(genome_size):
                gene = self._create_random_gene()
                genome.append(gene)
            
            seed = TransformerSeed(generation=0, genome=genome)
            self.population.append(seed)
    
    def _create_random_gene(self) -> Gene:
        """Crea un gen aleatorio para la población inicial"""
        operations = ['shift', 'rotate', 'color', 'copy', 'fill', 
                     'mirror', 'expand', 'contract', 'filter', 'blend']
        
        operation = random.choice(operations)
        parameters = {}
        
        # Parámetros aleatorios según operación
        if operation == 'shift':
            parameters = {'dx': random.randint(-3, 3), 'dy': random.randint(-3, 3)}
        elif operation == 'rotate':
            parameters = {'k': random.randint(0, 3)}
        elif operation == 'color':
            parameters = {'source': random.randint(0, 9), 'target': random.randint(0, 9)}
        elif operation == 'copy':
            parameters = {
                'src_x': random.randint(0, 10), 'src_y': random.randint(0, 10),
                'dst_x': random.randint(0, 10), 'dst_y': random.randint(0, 10),
                'size': random.randint(1, 5)
            }
        elif operation == 'fill':
            parameters = {
                'target': random.randint(0, 9),
                'fill': random.randint(0, 9),
                'mode': random.choice(['exact', 'greater', 'less', 'nonzero'])
            }
        elif operation == 'mirror':
            parameters = {'axis': random.choice(['horizontal', 'vertical', 'both'])}
        elif operation == 'expand':
            parameters = {
                'value': random.randint(1, 9),
                'direction': random.choice(['cross', 'diagonal'])
            }
        elif operation == 'contract':
            parameters = {'value': random.randint(1, 9)}
        elif operation == 'filter':
            parameters = {'condition': 'isolated'}
        elif operation == 'blend':
            parameters = {'factor': random.random()}
        
        return Gene(
            operation=operation,
            parameters=parameters,
            strength=random.uniform(0.5, 1.0),
            mutation_rate=random.uniform(0.05, 0.2)
        )
    
    def evolve_for_puzzle(self, train_examples: List[Dict], 
                          max_generations: int = 50) -> TransformerSeed:
        """
        Evoluciona la población para resolver un puzzle específico
        Retorna el mejor transformador encontrado
        """
        best_overall = None
        best_fitness = -1
        
        for gen in range(max_generations):
            # Evaluar fitness de cada semilla
            for seed in self.population:
                seed.fitness = self._evaluate_fitness(seed, train_examples)
            
            # Ordenar por fitness
            self.population.sort(key=lambda s: s.fitness, reverse=True)
            
            # Guardar el mejor
            if self.population[0].fitness > best_fitness:
                best_overall = deepcopy(self.population[0])
                best_fitness = best_overall.fitness
                
                # Añadir al hall of fame si es excepcional
                if best_fitness > 0.9:
                    self.hall_of_fame.append(deepcopy(best_overall))
            
            # Terminar si encontramos solución perfecta
            if best_fitness >= 0.99:
                logger.info(f"Solución perfecta encontrada en generación {gen}")
                break
            
            # Selección y reproducción
            self._next_generation()
            self.generation += 1
        
        return best_overall
    
    def _evaluate_fitness(self, seed: TransformerSeed, 
                         train_examples: List[Dict]) -> float:
        """Evalúa qué tan bien una semilla resuelve los ejemplos"""
        if not train_examples:
            return 0.0
        
        total_score = 0.0
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            expected_output = np.array(example['output'])
            
            # Aplicar transformación
            predicted = seed.apply(input_grid)
            
            # Calcular similitud
            if predicted.shape == expected_output.shape:
                # Accuracy de píxeles
                pixel_acc = np.mean(predicted == expected_output)
                
                # Bonus por patrones correctos
                pattern_bonus = 0.0
                if np.array_equal(predicted, expected_output):
                    pattern_bonus = 0.2  # Bonus por match perfecto
                
                score = pixel_acc + pattern_bonus
            else:
                # Penalización por cambio de dimensiones
                score = 0.0
            
            total_score += score
        
        return total_score / len(train_examples)
    
    def _next_generation(self):
        """Crea la siguiente generación mediante selección y reproducción"""
        new_population = []
        
        # Elitismo: mantener los mejores
        elite_size = self.pool_size // 10
        new_population.extend(deepcopy(self.population[:elite_size]))
        
        # Reproducción
        while len(new_population) < self.pool_size:
            # Selección por torneo
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            if random.random() < 0.7:  # Crossover
                child = parent1.reproduce(parent2)
            else:  # Mutación
                child = parent1.mutate()
            
            new_population.append(child)
        
        # Añadir algunos individuos completamente nuevos (inmigración)
        immigration_size = self.pool_size // 20
        for i in range(immigration_size):
            if i < len(new_population):
                genome_size = random.randint(3, 7)
                genome = [self._create_random_gene() for _ in range(genome_size)]
                new_population[-(i+1)] = TransformerSeed(
                    generation=self.generation + 1,
                    genome=genome
                )
        
        self.population = new_population[:self.pool_size]
    
    def _tournament_select(self, tournament_size: int = 5) -> TransformerSeed:
        """Selección por torneo"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda s: s.fitness)

class EvolutionaryTransformerSolver:
    """
    Solver principal que usa transformadores evolutivos
    NO tiene funciones hardcodeadas - todo emerge de la evolución
    """
    
    def __init__(self, pool_size: int = 50):
        self.pool = EvolutionaryPool(pool_size=pool_size)
        self.learned_transformers: Dict[str, TransformerSeed] = {}
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Resuelve un puzzle evolucionando transformadores"""
        
        # Crear firma del puzzle para cache
        puzzle_signature = self._get_puzzle_signature(train_examples)
        
        # Verificar si ya aprendimos este tipo de puzzle
        if puzzle_signature in self.learned_transformers:
            transformer = self.learned_transformers[puzzle_signature]
            logger.info(f"Usando transformador aprendido: {transformer.id}")
        else:
            # Evolucionar nuevo transformador
            logger.info("Evolucionando nuevo transformador...")
            transformer = self.pool.evolve_for_puzzle(train_examples, max_generations=30)
            
            # Guardar si es bueno
            if transformer.fitness > 0.8:
                self.learned_transformers[puzzle_signature] = transformer
                transformer.successful_applications += 1
        
        # Aplicar transformación al test
        result = transformer.apply(test_input)
        
        return result
    
    def _get_puzzle_signature(self, train_examples: List[Dict]) -> str:
        """Crea una firma única para el tipo de puzzle"""
        if not train_examples:
            return "empty"
        
        signatures = []
        for example in train_examples[:2]:  # Usar solo los primeros 2
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Características del puzzle
            sig = f"{input_grid.shape}_{output_grid.shape}"
            sig += f"_{np.unique(input_grid).tolist()}"
            sig += f"_{np.unique(output_grid).tolist()}"
            
            signatures.append(sig)
        
        return "_".join(signatures)