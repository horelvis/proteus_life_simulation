#!/usr/bin/env python3
"""
ARC Swarm Solver - Sistema de enjambre con votación
Múltiples agentes resuelven puzzles y votan por la mejor solución
Los que fallan mueren, los exitosos continúan
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from .arc_solver_python import ARCSolverPython, TransformationType
from .arc_augmentation import ARCAugmentation, AugmentationType

logger = logging.getLogger(__name__)

@dataclass
class SolverAgent:
    """Agente individual del enjambre"""
    id: int
    solver: ARCSolverPython
    fitness: float = 0.0
    solutions: List[np.ndarray] = None
    alive: bool = True
    mutations: Dict[str, Any] = None
    detected_rules: List[Dict[str, Any]] = None  # Reglas que detectó exitosamente
    
    def __post_init__(self):
        if self.solutions is None:
            self.solutions = []
        if self.mutations is None:
            self.mutations = {}
        if self.detected_rules is None:
            self.detected_rules = []

class ARCSwarmSolver:
    """
    Sistema de enjambre para resolver puzzles ARC
    - Múltiples agentes con diferentes configuraciones
    - Votación para seleccionar mejor solución
    - Evolución: los que fallan mueren
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 generations: int = 5,
                 mutation_rate: float = 0.2,
                 seed: Optional[int] = None,
                 max_workers: Optional[int] = None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.agents: List[SolverAgent] = []
        self.best_solution = None
        self.voting_history = []
        
        # Control de aleatoriedad para reproducibilidad
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            
        # NUEVO: Configurar número de workers para paralelización
        if max_workers is None:
            # Por defecto: min(población/2, 10, núcleos disponibles)
            import os
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(self.population_size // 2, 10, cpu_count)
        else:
            self.max_workers = max_workers
            
        logger.info(f"Enjambre configurado con max_workers={self.max_workers}")
        
    def _create_agent(self, agent_id: int) -> SolverAgent:
        """Crea un agente con configuración aleatoria"""
        solver = ARCSolverPython()
        
        # Mutar configuración aleatoriamente usando RNG controlado
        mutations = {
            'use_augmentation': self.rng.random() > 0.3,
            'verification_strict': self.rng.random() > 0.5,
            'confidence_threshold': 0.5 + self.rng.random() * 0.4,
            'rule_priorities': self._random_rule_priorities()
        }
        
        # Aplicar mutaciones
        solver.use_augmentation = mutations['use_augmentation']
        solver.verificar_reglas = mutations['verification_strict']
        
        agent = SolverAgent(
            id=agent_id,
            solver=solver,
            mutations=mutations
        )
        
        return agent
    
    def _random_rule_priorities(self) -> List[TransformationType]:
        """Genera orden aleatorio de prioridad de reglas"""
        rules = list(TransformationType)
        self.rng.shuffle(rules)
        return rules
    
    def _evaluate_solution(self, 
                          solution: np.ndarray, 
                          train_examples: List[Dict],
                          test_input: np.ndarray = None) -> float:
        """Evalúa qué tan buena es una solución comparando con outputs correctos"""
        if solution is None:
            return 0.0
            
        fitness = 0.0
        total_weight = 0.0
        
        # 1. NUEVO: Comparación directa con outputs de entrenamiento
        # Si podemos aplicar la transformación a los inputs de entrenamiento
        if len(train_examples) > 0:
            cell_accuracy_scores = []
            
            for example in train_examples:
                input_grid = np.array(example['input'])
                expected_output = np.array(example['output'])
                
                # Si la solución tiene el mismo tamaño que el output esperado
                if solution.shape == expected_output.shape:
                    # Calcular precisión a nivel de celda
                    matching_cells = np.sum(solution == expected_output)
                    total_cells = expected_output.size
                    cell_accuracy = matching_cells / total_cells
                    cell_accuracy_scores.append(cell_accuracy)
                    
            if cell_accuracy_scores:
                # Peso principal: precisión directa de celdas (60%)
                avg_cell_accuracy = np.mean(cell_accuracy_scores)
                fitness += avg_cell_accuracy * 0.6
                total_weight += 0.6
        
        # 2. Consistencia con tamaños de output (10%)
        size_matches = 0
        for example in train_examples:
            output = np.array(example['output'])
            if solution.shape == output.shape:
                size_matches += 1
        
        if train_examples:
            size_consistency = size_matches / len(train_examples)
            fitness += size_consistency * 0.1
            total_weight += 0.1
        
        # 3. Uso de colores consistentes (15%)
        train_colors = set()
        for example in train_examples:
            train_colors.update(np.unique(example['output']))
        
        solution_colors = set(np.unique(solution))
        if solution_colors.issubset(train_colors):
            fitness += 0.15
            total_weight += 0.15
        elif train_colors:
            # Penalizar por colores fuera del conjunto esperado
            extra_colors = len(solution_colors - train_colors)
            color_penalty = max(0, 1 - (extra_colors / len(train_colors)))
            fitness += color_penalty * 0.15
            total_weight += 0.15
        
        # 4. Patrones de densidad (10%)
        if len(train_examples) > 0:
            train_densities = [
                np.count_nonzero(ex['output']) / np.array(ex['output']).size 
                for ex in train_examples
            ]
            expected_density = np.mean(train_densities)
            solution_density = np.count_nonzero(solution) / solution.size
            density_diff = abs(expected_density - solution_density)
            density_score = max(0, 1 - density_diff)
            fitness += density_score * 0.1
            total_weight += 0.1
        
        # 5. Complejidad razonable (5%)
        unique_values = len(np.unique(solution))
        expected_unique = np.mean([len(np.unique(ex['output'])) for ex in train_examples])
        complexity_diff = abs(unique_values - expected_unique) / max(expected_unique, 1)
        complexity_score = max(0, 1 - complexity_diff)
        fitness += complexity_score * 0.05
        total_weight += 0.05
        
        # Normalizar si no se alcanzó el peso total
        if total_weight > 0:
            fitness = fitness / total_weight
        
        return fitness
    
    def solve_with_swarm(self, 
                        train_examples: List[Dict], 
                        test_input: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resuelve usando el enjambre con votación"""
        logger.info(f"Starting swarm with {self.population_size} agents")
        
        # Crear población inicial
        self.agents = [
            self._create_agent(i) for i in range(self.population_size)
        ]
        
        best_overall_solution = None
        best_overall_fitness = -1
        
        # Evolución por generaciones
        for generation in range(self.generations):
            logger.info(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Cada agente intenta resolver
            generation_solutions = []
            
            # Resolver en paralelo para mayor velocidad
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_agent = {
                    executor.submit(
                        self._agent_solve, 
                        agent, 
                        train_examples, 
                        test_input
                    ): agent 
                    for agent in self.agents if agent.alive
                }
                
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        solution, reasoning = future.result()
                        if solution is not None:
                            agent.solutions.append(solution)
                            agent.fitness = self._evaluate_solution(solution, train_examples)
                            generation_solutions.append((agent, solution))
                            logger.info(f"   Agente {agent.id}: fitness={agent.fitness:.2f}")
                    except Exception as e:
                        logger.error(f"   Agente {agent.id} falló: {e}")
                        agent.alive = False
            
            # Votación: encontrar la solución más común
            if generation_solutions:
                solution_votes = {}
                
                for agent, solution in generation_solutions:
                    # Convertir solución a string para comparación
                    sol_key = solution.tobytes()
                    
                    if sol_key not in solution_votes:
                        solution_votes[sol_key] = {
                            'solution': solution,
                            'votes': 0,
                            'total_fitness': 0,
                            'agents': []
                        }
                    
                    solution_votes[sol_key]['votes'] += 1
                    solution_votes[sol_key]['total_fitness'] += agent.fitness
                    solution_votes[sol_key]['agents'].append(agent.id)
                
                # Seleccionar solución con más votos (o mejor fitness en caso de empate)
                best_voted = max(
                    solution_votes.values(),
                    key=lambda x: (x['votes'], x['total_fitness'])
                )
                
                logger.info(f"   Voting results: {best_voted['votes']} votes, "
                          f"average fitness: {best_voted['total_fitness']/best_voted['votes']:.2f}")
                
                # Actualizar mejor solución global
                avg_fitness = best_voted['total_fitness'] / best_voted['votes']
                if avg_fitness > best_overall_fitness:
                    best_overall_fitness = avg_fitness
                    best_overall_solution = best_voted['solution']
                
                self.voting_history.append({
                    'generation': generation,
                    'votes': best_voted['votes'],
                    'fitness': avg_fitness,
                    'agents': best_voted['agents']
                })
            
            # Selección natural: matar agentes con bajo fitness
            if generation < self.generations - 1:  # No matar en la última generación
                self._natural_selection()
                
                # Reproducir para mantener población
                self._reproduce()
        
        # Preparar reporte final
        alive_count = sum(1 for agent in self.agents if agent.alive)
        
        report = {
            'solution': best_overall_solution,
            'fitness': best_overall_fitness,
            'alive_agents': alive_count,
            'dead_agents': self.population_size - alive_count,
            'voting_history': self.voting_history,
            'best_agents': [
                {
                    'id': agent.id,
                    'fitness': agent.fitness,
                    'mutations': agent.mutations
                }
                for agent in sorted(self.agents, key=lambda a: a.fitness, reverse=True)[:5]
                if agent.alive
            ]
        }
        
        logger.info(f"\n✅ Enjambre completado:")
        logger.info(f"   Agentes vivos: {alive_count}/{self.population_size}")
        logger.info(f"   Mejor fitness: {best_overall_fitness:.2f}")
        
        return best_overall_solution, report
    
    def _agent_solve(self, 
                    agent: SolverAgent, 
                    train_examples: List[Dict], 
                    test_input: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """Un agente individual intenta resolver el puzzle"""
        try:
            # Aplicar mutaciones de prioridad de reglas
            if 'rule_priorities' in agent.mutations:
                # Aplicar el orden de detección personalizado del agente
                agent.solver.set_rule_priority(agent.mutations['rule_priorities'])
            
            # Resolver
            solution, reasoning = agent.solver.solve_with_steps(train_examples, test_input)
            
            # Capturar reglas detectadas exitosamente
            for step in reasoning:
                if step.get('type') == 'rule_detection' and step.get('rule'):
                    rule = step['rule']
                    if rule not in agent.detected_rules:
                        agent.detected_rules.append(rule)
            
            # Aplicar threshold de confianza
            if 'confidence_threshold' in agent.mutations:
                # Filtrar soluciones de baja confianza
                if agent.solver.confidence < agent.mutations['confidence_threshold']:
                    return None, reasoning
            
            return solution, reasoning
            
        except Exception as e:
            logger.error(f"Agente {agent.id} error: {e}")
            return None, []
    
    def _natural_selection(self):
        """Mata agentes con bajo fitness"""
        # Calcular fitness promedio
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return
            
        avg_fitness = np.mean([a.fitness for a in alive_agents])
        
        # Matar agentes por debajo del promedio
        for agent in alive_agents:
            if agent.fitness < avg_fitness * 0.7:  # 70% del promedio
                agent.alive = False
                logger.info(f"   ☠️  Agente {agent.id} eliminado (fitness={agent.fitness:.2f})")
    
    def _reproduce(self):
        """Crea nuevos agentes para reemplazar a los muertos"""
        alive_agents = [a for a in self.agents if a.alive]
        dead_count = self.population_size - len(alive_agents)
        
        if dead_count > 0 and alive_agents:
            logger.info(f"   🐣 Reproduciendo {dead_count} nuevos agentes")
            
            # Seleccionar padres (los mejores)
            parents = sorted(alive_agents, key=lambda a: a.fitness, reverse=True)
            
            for i in range(dead_count):
                # Crear nuevo agente basado en un padre exitoso
                parent = parents[i % len(parents)]
                new_id = max(a.id for a in self.agents) + 1
                
                new_agent = self._create_agent(new_id)
                
                # Heredar algunas características del padre con mutación
                if self.rng.random() > 0.5:
                    new_agent.mutations['use_augmentation'] = parent.mutations.get('use_augmentation', True)
                if self.rng.random() > 0.5:
                    new_agent.mutations['verification_strict'] = parent.mutations.get('verification_strict', True)
                
                # Mutar threshold de confianza
                parent_threshold = parent.mutations.get('confidence_threshold', 0.7)
                new_agent.mutations['confidence_threshold'] = parent_threshold + (self.rng.random() - 0.5) * 0.2
                new_agent.mutations['confidence_threshold'] = max(0.5, min(0.95, new_agent.mutations['confidence_threshold']))
                
                # NUEVO: Heredar prioridades de reglas con mutación
                if 'rule_priorities' in parent.mutations and self.rng.random() > 0.3:
                    # 70% de probabilidad de heredar prioridades
                    parent_priorities = parent.mutations['rule_priorities'].copy()
                    
                    # Mutar ligeramente el orden (intercambiar 2 reglas aleatorias)
                    if self.rng.random() < self.mutation_rate and len(parent_priorities) > 2:
                        idx1, idx2 = self.rng.choice(len(parent_priorities), size=2, replace=False)
                        parent_priorities[idx1], parent_priorities[idx2] = parent_priorities[idx2], parent_priorities[idx1]
                    
                    new_agent.mutations['rule_priorities'] = parent_priorities
                
                # NUEVO: Heredar reglas detectadas exitosamente
                if parent.detected_rules and self.rng.random() > 0.4:
                    # 60% de probabilidad de heredar conocimiento de reglas
                    new_agent.detected_rules = parent.detected_rules.copy()
                    logger.info(f"   Agente {new_agent.id} hereda {len(new_agent.detected_rules)} reglas del padre {parent.id}")
                
                self.agents.append(new_agent)


def test_swarm_solver():
    """Prueba el solver de enjambre"""
    print("🐝 Probando ARC Swarm Solver")
    print("="*60)
    
    # Ejemplo simple de color mapping
    train_examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[2, 3], [4, 5]]  # +1 a cada color
        }
    ]
    
    test_input = np.array([[1, 3], [2, 4]])
    
    # Crear enjambre
    swarm = ARCSwarmSolver(population_size=10, generations=3)
    
    # Resolver
    solution, report = swarm.solve_with_swarm(train_examples, test_input)
    
    print("\n📊 Resultados del enjambre:")
    print(f"Solución final:\n{solution}")
    print(f"Fitness: {report['fitness']:.2f}")
    print(f"Agentes vivos: {report['alive_agents']}/{swarm.population_size}")
    
    print("\n🏆 Mejores agentes:")
    for agent in report['best_agents']:
        print(f"   Agente {agent['id']}: fitness={agent['fitness']:.2f}")
        print(f"      Mutaciones: {agent['mutations']}")
    
    print("\n🗳️  Historia de votación:")
    for vote in report['voting_history']:
        print(f"   Gen {vote['generation']+1}: {vote['votes']} votos, fitness={vote['fitness']:.2f}")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    test_swarm_solver()