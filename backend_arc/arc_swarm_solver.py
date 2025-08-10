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
from arc_solver_python import ARCSolverPython, TransformationType
from arc_augmentation import ARCAugmentation, AugmentationType

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
    
    def __post_init__(self):
        if self.solutions is None:
            self.solutions = []
        if self.mutations is None:
            self.mutations = {}

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
                 mutation_rate: float = 0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.agents: List[SolverAgent] = []
        self.best_solution = None
        self.voting_history = []
        
    def _create_agent(self, agent_id: int) -> SolverAgent:
        """Crea un agente con configuración aleatoria"""
        solver = ARCSolverPython()
        
        # Mutar configuración aleatoriamente
        mutations = {
            'use_augmentation': random.random() > 0.3,
            'verification_strict': random.random() > 0.5,
            'confidence_threshold': 0.5 + random.random() * 0.4,
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
        random.shuffle(rules)
        return rules
    
    def _evaluate_solution(self, 
                          solution: np.ndarray, 
                          train_examples: List[Dict]) -> float:
        """Evalúa qué tan buena es una solución basándose en patrones"""
        if solution is None:
            return 0.0
            
        fitness = 0.0
        
        # 1. Consistencia con tamaños de output en ejemplos
        for example in train_examples:
            output = np.array(example['output'])
            if solution.shape == output.shape:
                fitness += 0.2
        
        # 2. Uso de colores consistentes
        train_colors = set()
        for example in train_examples:
            train_colors.update(np.unique(example['output']))
        
        solution_colors = set(np.unique(solution))
        if solution_colors.issubset(train_colors):
            fitness += 0.3
        
        # 3. Patrones similares (simplificado)
        if len(train_examples) > 0:
            # Comparar densidad de píxeles no-cero
            train_density = np.mean([
                np.count_nonzero(ex['output']) / np.array(ex['output']).size 
                for ex in train_examples
            ])
            solution_density = np.count_nonzero(solution) / solution.size
            density_diff = abs(train_density - solution_density)
            fitness += max(0, 0.3 * (1 - density_diff))
        
        # 4. Complejidad razonable
        unique_values = len(np.unique(solution))
        if 2 <= unique_values <= 5:
            fitness += 0.2
        
        return fitness
    
    def solve_with_swarm(self, 
                        train_examples: List[Dict], 
                        test_input: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resuelve usando el enjambre con votación"""
        logger.info(f"🐝 Iniciando enjambre con {self.population_size} agentes")
        
        # Crear población inicial
        self.agents = [
            self._create_agent(i) for i in range(self.population_size)
        ]
        
        best_overall_solution = None
        best_overall_fitness = -1
        
        # Evolución por generaciones
        for generation in range(self.generations):
            logger.info(f"\n📊 Generación {generation + 1}/{self.generations}")
            
            # Cada agente intenta resolver
            generation_solutions = []
            
            # Resolver en paralelo para mayor velocidad
            with ThreadPoolExecutor(max_workers=min(10, self.population_size)) as executor:
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
                
                logger.info(f"   🗳️  Votación: {best_voted['votes']} votos, "
                          f"fitness promedio: {best_voted['total_fitness']/best_voted['votes']:.2f}")
                
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
                # Modificar el orden de detección basado en las prioridades del agente
                # (Esto requeriría modificar ARCSolverPython para aceptar orden custom)
                pass
            
            # Resolver
            solution, reasoning = agent.solver.solve_with_steps(train_examples, test_input)
            
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
                if random.random() > 0.5:
                    new_agent.mutations['use_augmentation'] = parent.mutations.get('use_augmentation', True)
                if random.random() > 0.5:
                    new_agent.mutations['verification_strict'] = parent.mutations.get('verification_strict', True)
                
                # Mutar threshold de confianza
                parent_threshold = parent.mutations.get('confidence_threshold', 0.7)
                new_agent.mutations['confidence_threshold'] = parent_threshold + (random.random() - 0.5) * 0.2
                new_agent.mutations['confidence_threshold'] = max(0.5, min(0.95, new_agent.mutations['confidence_threshold']))
                
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