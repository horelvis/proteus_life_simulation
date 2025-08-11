#!/usr/bin/env python3
"""
ARCSwarmSolver Mejorado - Versión con correcciones críticas
Basado en el análisis de fallos del solver original
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from copy import deepcopy
import random
from dataclasses import dataclass, field
from .arc_solver_python import ARCSolverPython

logger = logging.getLogger(__name__)

@dataclass
class SolverAgent:
    """Agente individual del enjambre con capacidades mejoradas"""
    id: int
    solver: ARCSolverPython
    mutations: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    alive: bool = True
    generation: int = 0
    detected_rules: List[Dict] = field(default_factory=list)
    successful_chains: List[List[str]] = field(default_factory=list)  # Cadenas de transformaciones exitosas
    specialization: Optional[str] = None  # Tipo de especialización del agente
    parent_ids: List[int] = field(default_factory=list)  # IDs de los padres (para crossover)
    
class SharedMemory:
    """Memoria compartida entre agentes del enjambre"""
    def __init__(self):
        self.successful_rules = {}  # {rule_type: [(rule, fitness, agent_id)]}
        self.successful_chains = []  # [(chain, fitness, agent_id)]
        self.puzzle_characteristics = {}  # Características del puzzle actual
        
    def add_successful_rule(self, rule_type: str, rule: Dict, fitness: float, agent_id: int):
        """Añade una regla exitosa a la memoria"""
        if rule_type not in self.successful_rules:
            self.successful_rules[rule_type] = []
        self.successful_rules[rule_type].append((rule, fitness, agent_id))
        # Mantener solo las 10 mejores por tipo
        self.successful_rules[rule_type].sort(key=lambda x: x[1], reverse=True)
        self.successful_rules[rule_type] = self.successful_rules[rule_type][:10]
        
    def add_successful_chain(self, chain: List[str], fitness: float, agent_id: int):
        """Añade una cadena de transformaciones exitosa"""
        self.successful_chains.append((chain, fitness, agent_id))
        self.successful_chains.sort(key=lambda x: x[1], reverse=True)
        self.successful_chains = self.successful_chains[:20]
        
    def get_best_rules(self, rule_type: str = None, top_k: int = 5):
        """Obtiene las mejores reglas de un tipo o todas"""
        if rule_type:
            return self.successful_rules.get(rule_type, [])[:top_k]
        else:
            all_rules = []
            for rules in self.successful_rules.values():
                all_rules.extend(rules)
            all_rules.sort(key=lambda x: x[1], reverse=True)
            return all_rules[:top_k]

class ARCSwarmSolverImproved:
    """
    Solver mejorado basado en enjambre con las siguientes mejoras:
    1. Evaluación correcta aplicando reglas a inputs de entrenamiento
    2. Soporte para cadenas de transformaciones
    3. Memoria compartida entre agentes
    4. Crossover real en reproducción
    5. Agentes especializados
    """
    
    # Tipos de especialización para agentes
    SPECIALIZATIONS = [
        'symmetry',      # Especializado en simetrías y rotaciones
        'replication',   # Especializado en replicación y escalado
        'color',         # Especializado en transformaciones de color
        'pattern',       # Especializado en detección de patrones
        'topology',      # Especializado en transformaciones topológicas
        'counting',      # Especializado en conteo y aritmética
        'generalist'     # Sin especialización
    ]
    
    def __init__(self, 
                 population_size: int = 30,
                 generations: int = 10,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elite_size: int = 5):
        """
        Inicializa el solver mejorado
        
        Args:
            population_size: Tamaño de la población
            generations: Número de generaciones
            mutation_rate: Probabilidad de mutación
            crossover_rate: Probabilidad de crossover
            elite_size: Número de mejores agentes a preservar
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.agents = []
        self.shared_memory = SharedMemory()
        self.voting_history = []
        
    def solve_with_swarm(self, 
                        train_examples: List[Dict], 
                        test_input: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Resuelve usando el enjambre mejorado
        """
        logger.info(f"\n🐝 Iniciando enjambre mejorado con {self.population_size} agentes")
        
        # Analizar características del puzzle
        self._analyze_puzzle_characteristics(train_examples)
        
        # Inicializar población con agentes especializados
        self._initialize_population()
        
        best_overall_solution = None
        best_overall_fitness = 0
        
        # Evolución por generaciones
        for generation in range(self.generations):
            logger.info(f"\n📊 Generación {generation + 1}/{self.generations}")
            
            solutions = []
            
            # Cada agente intenta resolver
            for agent in self.agents:
                if not agent.alive:
                    continue
                    
                solution, reasoning, rule_chain = self._agent_solve_with_chains(
                    agent, train_examples, test_input
                )
                
                if solution is not None:
                    # CORRECCIÓN CRÍTICA: Evaluar aplicando la regla a TODOS los inputs de entrenamiento
                    fitness = self._evaluate_solution_correctly(
                        agent, train_examples, test_input, solution, rule_chain
                    )
                    
                    agent.fitness = fitness
                    solutions.append((agent.id, solution, fitness, rule_chain))
                    
                    # Actualizar memoria compartida con reglas exitosas
                    if fitness > 0.5:
                        for rule in agent.detected_rules:
                            self.shared_memory.add_successful_rule(
                                rule.get('type', 'unknown'),
                                rule,
                                fitness,
                                agent.id
                            )
                        if rule_chain:
                            self.shared_memory.add_successful_chain(
                                rule_chain, fitness, agent.id
                            )
                    
                    # Actualizar mejor solución global
                    if fitness > best_overall_fitness:
                        best_overall_solution = solution
                        best_overall_fitness = fitness
                        logger.info(f"   ✨ Nueva mejor solución: fitness={fitness:.3f}")
            
            # Votación y consenso
            if solutions:
                voted_solution = self._voting_consensus(solutions)
                if voted_solution is not None:
                    vote_fitness = self._evaluate_solution_correctly(
                        None, train_examples, test_input, voted_solution, []
                    )
                    if vote_fitness > best_overall_fitness:
                        best_overall_solution = voted_solution
                        best_overall_fitness = vote_fitness
                        logger.info(f"   🗳️ Solución por consenso mejoró: {vote_fitness:.3f}")
            
            # Reproducción con crossover mejorado
            if generation < self.generations - 1:
                self._reproduce_with_crossover()
        
        # Preparar reporte
        report = self._generate_report(best_overall_fitness)
        
        return best_overall_solution, report
    
    def _analyze_puzzle_characteristics(self, train_examples: List[Dict]):
        """Analiza las características del puzzle para guiar la búsqueda"""
        characteristics = {
            'size_changes': False,
            'color_changes': False,
            'pattern_extraction': False,
            'symmetry_present': False,
            'counting_involved': False
        }
        
        for example in train_examples:
            input_arr = np.array(example['input'])
            output_arr = np.array(example['output'])
            
            # Cambios de tamaño
            if input_arr.shape != output_arr.shape:
                characteristics['size_changes'] = True
            
            # Cambios de color
            if set(np.unique(input_arr)) != set(np.unique(output_arr)):
                characteristics['color_changes'] = True
            
            # Posible extracción de patrón
            if output_arr.size < input_arr.size * 0.5:
                characteristics['pattern_extraction'] = True
        
        self.shared_memory.puzzle_characteristics = characteristics
        logger.info(f"📋 Características del puzzle: {characteristics}")
    
    def _initialize_population(self):
        """Inicializa población con agentes especializados"""
        self.agents = []
        
        # Distribuir especializaciones basándose en características del puzzle
        specs_distribution = self._get_specialization_distribution()
        
        for i in range(self.population_size):
            agent = SolverAgent(
                id=i,
                solver=ARCSolverPython(),
                generation=0,
                specialization=specs_distribution[i % len(specs_distribution)]
            )
            
            # Configurar según especialización
            self._configure_agent_by_specialization(agent)
            
            # Mutaciones iniciales aleatorias
            agent.mutations = self._generate_mutations(agent.specialization)
            
            self.agents.append(agent)
        
        logger.info(f"🧬 Población inicializada con especializaciones: {specs_distribution}")
    
    def _get_specialization_distribution(self):
        """Determina la distribución de especializaciones según el puzzle"""
        chars = self.shared_memory.puzzle_characteristics
        distribution = []
        
        # Asignar más agentes a especializaciones relevantes
        if chars.get('symmetry_present'):
            distribution.extend(['symmetry'] * 5)
        if chars.get('color_changes'):
            distribution.extend(['color'] * 4)
        if chars.get('pattern_extraction'):
            distribution.extend(['pattern'] * 4)
        if chars.get('size_changes'):
            distribution.extend(['replication', 'topology'] * 3)
        if chars.get('counting_involved'):
            distribution.extend(['counting'] * 3)
        
        # Añadir generalistas
        distribution.extend(['generalist'] * 5)
        
        # Completar con aleatorios si es necesario
        while len(distribution) < self.population_size:
            distribution.append(random.choice(self.SPECIALIZATIONS))
        
        return distribution[:self.population_size]
    
    def _configure_agent_by_specialization(self, agent: SolverAgent):
        """Configura un agente según su especialización"""
        spec = agent.specialization
        
        if spec == 'symmetry':
            agent.mutations['rule_priorities'] = [
                'symmetry', 'rotation', 'reflection', 'translation'
            ]
        elif spec == 'replication':
            agent.mutations['rule_priorities'] = [
                'replication', 'scaling', 'tiling', 'pattern_repeat'
            ]
        elif spec == 'color':
            agent.mutations['rule_priorities'] = [
                'color_mapping', 'color_fill', 'color_replace', 'color_count'
            ]
        elif spec == 'pattern':
            agent.mutations['rule_priorities'] = [
                'pattern_extraction', 'pattern_completion', 'shape_detection'
            ]
        elif spec == 'topology':
            agent.mutations['rule_priorities'] = [
                'connectivity', 'boundary', 'fill_shape', 'outline'
            ]
        elif spec == 'counting':
            agent.mutations['rule_priorities'] = [
                'count_objects', 'arithmetic', 'size_based'
            ]
        # Los generalistas no tienen prioridades específicas
    
    def _agent_solve_with_chains(self, 
                                 agent: SolverAgent,
                                 train_examples: List[Dict],
                                 test_input: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict], List[str]]:
        """
        Resuelve permitiendo cadenas de transformaciones
        """
        # Intentar primero con regla única (más rápido)
        solution, reasoning = agent.solver.solve_with_steps(train_examples, test_input)
        rule_chain = []
        
        # Si falla o tiene baja confianza, intentar cadenas
        if solution is None or agent.solver.confidence < 0.5:
            # Obtener reglas exitosas de la memoria compartida
            best_rules = self.shared_memory.get_best_rules(top_k=3)
            best_chains = self.shared_memory.successful_chains[:3]
            
            # Intentar aplicar cadenas conocidas
            for chain, _, _ in best_chains:
                chain_solution = self._apply_rule_chain(
                    test_input, chain, train_examples
                )
                if chain_solution is not None:
                    solution = chain_solution
                    rule_chain = chain
                    break
            
            # Si aún no hay solución, intentar combinar reglas
            if solution is None and len(agent.detected_rules) > 1:
                for i, rule1 in enumerate(agent.detected_rules):
                    for rule2 in agent.detected_rules[i+1:]:
                        chain = [rule1['type'], rule2['type']]
                        chain_solution = self._apply_rule_chain(
                            test_input, chain, train_examples
                        )
                        if chain_solution is not None:
                            solution = chain_solution
                            rule_chain = chain
                            agent.successful_chains.append(chain)
                            break
        
        # Capturar reglas detectadas
        for step in reasoning:
            if step.get('type') == 'rule_detection' and step.get('rule'):
                rule = step['rule']
                if rule not in agent.detected_rules:
                    agent.detected_rules.append(rule)
        
        return solution, reasoning, rule_chain
    
    def _apply_rule_chain(self, 
                          input_grid: np.ndarray,
                          rule_chain: List[str],
                          train_examples: List[Dict]) -> Optional[np.ndarray]:
        """Aplica una cadena de transformaciones"""
        if not rule_chain:
            return input_grid
            
        current = input_grid.copy()
        
        for rule_type in rule_chain:
            # Aplicar cada transformación en secuencia
            transform = self._get_transformation(rule_type, train_examples)
            if transform:
                try:
                    transformed = transform(current)
                    if transformed is not None:
                        current = transformed
                    else:
                        logger.debug(f"Transformación {rule_type} retornó None")
                        return None
                except Exception as e:
                    logger.debug(f"Error aplicando transformación {rule_type}: {e}")
                    return None
            else:
                logger.debug(f"No se pudo crear transformación para {rule_type}")
                # Continuar con las siguientes transformaciones si una falla
                pass
        
        return current
    
    def _get_transformation(self, rule_type: str, train_examples: List[Dict]):
        """Obtiene una función de transformación para un tipo de regla"""
        
        # Analizar ejemplos para aprender parámetros de la transformación
        if rule_type == 'color_mapping':
            return self._create_color_mapping_transform(train_examples)
        elif rule_type == 'rotation':
            return self._create_rotation_transform(train_examples)
        elif rule_type == 'reflection':
            return self._create_reflection_transform(train_examples)
        elif rule_type == 'pattern_replication':
            return self._create_replication_transform(train_examples)
        elif rule_type == 'scaling':
            return self._create_scaling_transform(train_examples)
        elif rule_type == 'symmetry':
            return self._create_symmetry_transform(train_examples)
        elif rule_type == 'pattern_extraction':
            return self._create_extraction_transform(train_examples)
        elif rule_type == 'fill_shape':
            return self._create_fill_transform(train_examples)
        elif rule_type == 'count_objects':
            return self._create_counting_transform(train_examples)
        elif rule_type == 'translation':
            return self._create_translation_transform(train_examples)
        else:
            return None
    
    def _evaluate_solution_correctly(self,
                                    agent: Optional[SolverAgent],
                                    train_examples: List[Dict],
                                    test_input: np.ndarray,
                                    solution: np.ndarray,
                                    rule_chain: List[str]) -> float:
        """
        CORRECCIÓN CRÍTICA: Evalúa correctamente aplicando la transformación
        a CADA input de entrenamiento y comparando con su output esperado
        """
        if solution is None:
            return 0.0
        
        fitness = 0.0
        total_weight = 0.0
        
        # 1. EVALUACIÓN CORRECTA: Aplicar la regla/transformación a inputs de entrenamiento
        if agent and len(train_examples) > 0:
            transformation_scores = []
            
            for example in train_examples:
                input_grid = np.array(example['input'])
                expected_output = np.array(example['output'])
                
                # Aplicar la misma transformación que se usó para el test
                if rule_chain:
                    predicted = self._apply_rule_chain(input_grid, rule_chain, train_examples)
                else:
                    # Usar el solver del agente para aplicar su regla
                    predicted, _ = agent.solver.solve_with_steps([example], input_grid)
                
                if predicted is not None:
                    # Comparar predicción con output esperado
                    if predicted.shape == expected_output.shape:
                        matching_cells = np.sum(predicted == expected_output)
                        total_cells = expected_output.size
                        accuracy = matching_cells / total_cells
                        transformation_scores.append(accuracy)
                    else:
                        # Penalizar si el tamaño no coincide
                        transformation_scores.append(0.0)
                else:
                    transformation_scores.append(0.0)
            
            if transformation_scores:
                # Peso principal: qué tan bien la regla transforma los ejemplos (70%)
                avg_transformation_accuracy = np.mean(transformation_scores)
                fitness += avg_transformation_accuracy * 0.7
                total_weight += 0.7
                
                # Bonus por consistencia (10%)
                if all(s > 0.8 for s in transformation_scores):
                    fitness += 0.1
                    total_weight += 0.1
        
        # 2. Características estructurales de la solución (20%)
        structural_score = self._evaluate_structural_properties(
            solution, train_examples
        )
        fitness += structural_score * 0.2
        total_weight += 0.2
        
        # 3. Bonus por usar cadenas de transformaciones exitosas (10%)
        if rule_chain and len(rule_chain) > 1:
            # Premiar soluciones que usan múltiples transformaciones
            chain_bonus = min(0.1, len(rule_chain) * 0.03)
            fitness += chain_bonus
            total_weight += chain_bonus
        
        # Normalizar fitness
        if total_weight > 0:
            fitness = fitness / total_weight
        
        return min(1.0, fitness)
    
    def _evaluate_structural_properties(self, 
                                       solution: np.ndarray,
                                       train_examples: List[Dict]) -> float:
        """Evalúa propiedades estructurales de la solución"""
        score = 0.0
        
        # Analizar outputs de entrenamiento
        train_outputs = [np.array(ex['output']) for ex in train_examples]
        
        # 1. Consistencia de colores
        expected_colors = set()
        for output in train_outputs:
            expected_colors.update(np.unique(output))
        
        solution_colors = set(np.unique(solution))
        if solution_colors.issubset(expected_colors):
            score += 0.3
        elif expected_colors:
            overlap = len(solution_colors & expected_colors) / len(expected_colors)
            score += overlap * 0.3
        
        # 2. Densidad similar
        if train_outputs:
            expected_density = np.mean([
                np.count_nonzero(out) / out.size for out in train_outputs
            ])
            solution_density = np.count_nonzero(solution) / solution.size
            density_diff = abs(expected_density - solution_density)
            score += max(0, 1 - density_diff) * 0.3
        
        # 3. Patrones de conectividad (simplificado)
        if train_outputs:
            # Verificar si la solución mantiene patrones similares
            score += 0.4  # Placeholder - requiere análisis más complejo
        
        return score
    
    def _voting_consensus(self, solutions: List[Tuple]) -> Optional[np.ndarray]:
        """Genera consenso mediante votación mejorada"""
        if not solutions:
            return None
        
        # Ordenar por fitness
        solutions.sort(key=lambda x: x[2], reverse=True)
        
        # Si la mejor solución tiene fitness > 0.8, usarla directamente
        if solutions[0][2] > 0.8:
            return solutions[0][1]
        
        # Intentar consenso entre las mejores soluciones
        top_solutions = solutions[:min(5, len(solutions))]
        
        # Votación por elemento si tienen el mismo tamaño
        shapes = [sol[1].shape for _, sol, _, _ in top_solutions]
        if len(set(shapes)) == 1:
            # Todas tienen el mismo tamaño, hacer votación por celda
            consensus = np.zeros_like(top_solutions[0][1])
            weights = [fit for _, _, fit, _ in top_solutions]
            total_weight = sum(weights)
            
            for (_, solution, fitness, _), weight in zip(top_solutions, weights):
                consensus += solution * (weight / total_weight)
            
            # Redondear al entero más cercano
            return np.round(consensus).astype(int)
        
        # Si no, devolver la mejor
        return solutions[0][1]
    
    def _reproduce_with_crossover(self):
        """Reproducción mejorada con crossover real"""
        # Ordenar por fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        
        # Preservar élite
        new_agents = self.agents[:self.elite_size]
        
        # Selección de padres por torneo
        while len(new_agents) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Clonación con mutación
                parent = self._tournament_selection()
                child = self._clone_and_mutate(parent)
            
            new_agents.append(child)
        
        self.agents = new_agents[:self.population_size]
        
        # Incrementar generación
        for agent in self.agents:
            if agent.generation == 0:  # Nuevo agente
                agent.generation = 1
    
    def _tournament_selection(self, tournament_size: int = 3) -> SolverAgent:
        """Selección por torneo"""
        tournament = random.sample(self.agents, min(tournament_size, len(self.agents)))
        return max(tournament, key=lambda a: a.fitness)
    
    def _crossover(self, parent1: SolverAgent, parent2: SolverAgent) -> SolverAgent:
        """Crossover real entre dos padres"""
        child = SolverAgent(
            id=len(self.agents) + random.randint(1000, 9999),
            solver=ARCSolverPython(),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        # Heredar especialización (50% de cada padre)
        if random.random() < 0.5:
            child.specialization = parent1.specialization
        else:
            child.specialization = parent2.specialization
        
        # Combinar mutaciones
        child.mutations = {}
        
        # Combinar prioridades de reglas
        if 'rule_priorities' in parent1.mutations and 'rule_priorities' in parent2.mutations:
            p1_rules = parent1.mutations['rule_priorities']
            p2_rules = parent2.mutations['rule_priorities']
            # Tomar mitad de cada uno, alternando
            child_rules = []
            for i in range(max(len(p1_rules), len(p2_rules))):
                if i < len(p1_rules) and (i % 2 == 0 or i >= len(p2_rules)):
                    child_rules.append(p1_rules[i])
                elif i < len(p2_rules):
                    child_rules.append(p2_rules[i])
            child.mutations['rule_priorities'] = child_rules
        
        # Combinar otros parámetros
        for key in set(parent1.mutations.keys()) | set(parent2.mutations.keys()):
            if key not in child.mutations:
                if key in parent1.mutations and key in parent2.mutations:
                    # Promediar valores numéricos
                    if isinstance(parent1.mutations[key], (int, float)):
                        child.mutations[key] = (parent1.mutations[key] + parent2.mutations[key]) / 2
                    else:
                        # Elegir aleatoriamente para otros tipos
                        child.mutations[key] = random.choice([
                            parent1.mutations[key],
                            parent2.mutations[key]
                        ])
                elif key in parent1.mutations:
                    child.mutations[key] = parent1.mutations[key]
                else:
                    child.mutations[key] = parent2.mutations[key]
        
        # Heredar reglas detectadas exitosas (evitar duplicados por tipo)
        seen_types = set()
        combined_rules = []
        for rule in parent1.detected_rules + parent2.detected_rules:
            rule_type = rule.get('type', 'unknown')
            if rule_type not in seen_types:
                combined_rules.append(rule)
                seen_types.add(rule_type)
        child.detected_rules = combined_rules
        
        # Heredar cadenas exitosas (convertir a lista de listas)
        seen_chains = set()
        combined_chains = []
        for chain in parent1.successful_chains + parent2.successful_chains:
            chain_tuple = tuple(chain) if isinstance(chain, list) else chain
            if chain_tuple not in seen_chains:
                combined_chains.append(list(chain_tuple))
                seen_chains.add(chain_tuple)
        child.successful_chains = combined_chains
        
        # Aplicar mutación con probabilidad
        if random.random() < self.mutation_rate:
            self._mutate_agent(child)
        
        return child
    
    def _clone_and_mutate(self, parent: SolverAgent) -> SolverAgent:
        """Clona un agente y lo muta"""
        child = SolverAgent(
            id=len(self.agents) + random.randint(1000, 9999),
            solver=ARCSolverPython(),
            generation=parent.generation + 1,
            specialization=parent.specialization,
            parent_ids=[parent.id]
        )
        
        # Copiar características del padre
        child.mutations = deepcopy(parent.mutations)
        child.detected_rules = deepcopy(parent.detected_rules)
        child.successful_chains = deepcopy(parent.successful_chains)
        
        # Mutar
        self._mutate_agent(child)
        
        return child
    
    def _mutate_agent(self, agent: SolverAgent):
        """Aplica mutaciones a un agente"""
        # Mutar prioridades de reglas
        if 'rule_priorities' in agent.mutations and random.random() < 0.3:
            rules = agent.mutations['rule_priorities']
            if len(rules) > 1:
                # Intercambiar dos reglas
                i, j = random.sample(range(len(rules)), 2)
                rules[i], rules[j] = rules[j], rules[i]
        
        # Mutar threshold de confianza
        if random.random() < 0.2:
            agent.mutations['confidence_threshold'] = random.uniform(0.3, 0.9)
        
        # Mutar parámetros de augmentación
        if random.random() < 0.2:
            agent.mutations['use_augmentation'] = random.choice([True, False])
        
        # Posibilidad de cambiar especialización (raro)
        if random.random() < 0.05:
            agent.specialization = random.choice(self.SPECIALIZATIONS)
            self._configure_agent_by_specialization(agent)
    
    def _generate_mutations(self, specialization: str) -> Dict[str, Any]:
        """Genera mutaciones iniciales para un agente"""
        mutations = {}
        
        # Threshold de confianza
        mutations['confidence_threshold'] = random.uniform(0.4, 0.8)
        
        # Uso de augmentación
        mutations['use_augmentation'] = random.choice([True, False])
        
        # Parámetros específicos por especialización
        if specialization == 'replication':
            mutations['max_replication_factor'] = random.randint(2, 10)
        elif specialization == 'color':
            mutations['color_sensitivity'] = random.uniform(0.5, 1.0)
        elif specialization == 'pattern':
            mutations['pattern_threshold'] = random.uniform(0.6, 0.9)
        
        return mutations
    
    def _generate_report(self, best_fitness: float) -> Dict:
        """Genera reporte detallado del enjambre"""
        alive_count = sum(1 for agent in self.agents if agent.alive)
        
        # Estadísticas de especialización
        spec_stats = {}
        for spec in self.SPECIALIZATIONS:
            spec_agents = [a for a in self.agents if a.specialization == spec]
            if spec_agents:
                spec_stats[spec] = {
                    'count': len(spec_agents),
                    'avg_fitness': np.mean([a.fitness for a in spec_agents]),
                    'best_fitness': max([a.fitness for a in spec_agents])
                }
        
        report = {
            'best_fitness': best_fitness,
            'alive_agents': alive_count,
            'dead_agents': self.population_size - alive_count,
            'specialization_stats': spec_stats,
            'shared_memory_rules': len(self.shared_memory.successful_rules),
            'shared_memory_chains': len(self.shared_memory.successful_chains),
            'voting_history': self.voting_history,
            'best_agents': [
                {
                    'id': agent.id,
                    'fitness': agent.fitness,
                    'specialization': agent.specialization,
                    'mutations': agent.mutations,
                    'detected_rules': len(agent.detected_rules),
                    'successful_chains': len(agent.successful_chains)
                }
                for agent in sorted(self.agents, key=lambda a: a.fitness, reverse=True)[:5]
            ]
        }
        
        return report
    
    def _create_color_mapping_transform(self, train_examples: List[Dict]):
        """Crea transformación de mapeo de colores"""
        # Aprender mapeo de colores de los ejemplos
        color_map = {}
        
        for example in train_examples:
            input_arr = np.array(example['input'])
            output_arr = np.array(example['output'])
            
            if input_arr.shape == output_arr.shape:
                # Analizar mapeo de colores pixel a pixel
                for i in range(input_arr.shape[0]):
                    for j in range(input_arr.shape[1]):
                        input_color = input_arr[i, j]
                        output_color = output_arr[i, j]
                        
                        if input_color not in color_map:
                            color_map[input_color] = output_color
                        elif color_map[input_color] != output_color:
                            # Mapeo inconsistente, puede no ser color mapping simple
                            pass
        
        def transform(grid):
            result = grid.copy()
            for old_color, new_color in color_map.items():
                result[grid == old_color] = new_color
            return result
        
        return transform if color_map else None
    
    def _create_rotation_transform(self, train_examples: List[Dict]):
        """Crea transformación de rotación"""
        # Detectar ángulo de rotación
        for k in [1, 2, 3]:  # 90, 180, 270 grados
            matches = 0
            for example in train_examples:
                input_arr = np.array(example['input'])
                output_arr = np.array(example['output'])
                rotated = np.rot90(input_arr, k)
                
                if rotated.shape == output_arr.shape and np.array_equal(rotated, output_arr):
                    matches += 1
            
            if matches == len(train_examples):
                def transform(grid):
                    return np.rot90(grid, k)
                return transform
        
        return None
    
    def _create_reflection_transform(self, train_examples: List[Dict]):
        """Crea transformación de reflexión"""
        # Probar reflexión horizontal y vertical
        for axis in [0, 1]:  # 0: vertical, 1: horizontal
            matches = 0
            for example in train_examples:
                input_arr = np.array(example['input'])
                output_arr = np.array(example['output'])
                reflected = np.flip(input_arr, axis=axis)
                
                if reflected.shape == output_arr.shape and np.array_equal(reflected, output_arr):
                    matches += 1
            
            if matches == len(train_examples):
                def transform(grid):
                    return np.flip(grid, axis=axis)
                return transform
        
        return None
    
    def _create_replication_transform(self, train_examples: List[Dict]):
        """Crea transformación de replicación de patrones"""
        # Detectar factor de replicación
        for factor in [2, 3, 4]:
            matches = 0
            for example in train_examples:
                input_arr = np.array(example['input'])
                output_arr = np.array(example['output'])
                
                # Probar replicación en ambas dimensiones
                replicated = np.tile(input_arr, (factor, factor))
                
                if replicated.shape == output_arr.shape and np.array_equal(replicated, output_arr):
                    matches += 1
            
            if matches == len(train_examples):
                def transform(grid):
                    return np.tile(grid, (factor, factor))
                return transform
        
        return None
    
    def _create_scaling_transform(self, train_examples: List[Dict]):
        """Crea transformación de escalado"""
        # Detectar factor de escala
        scale_factor = None
        
        for example in train_examples:
            input_arr = np.array(example['input'])
            output_arr = np.array(example['output'])
            
            # Calcular factor de escala
            if output_arr.shape[0] % input_arr.shape[0] == 0:
                factor = output_arr.shape[0] // input_arr.shape[0]
                if scale_factor is None:
                    scale_factor = factor
                elif scale_factor != factor:
                    return None  # Factor inconsistente
        
        if scale_factor:
            def transform(grid):
                # Escalar repitiendo cada pixel
                return np.repeat(np.repeat(grid, scale_factor, axis=0), scale_factor, axis=1)
            return transform
        
        return None
    
    def _create_symmetry_transform(self, train_examples: List[Dict]):
        """Crea transformación basada en simetría"""
        def transform(grid):
            h, w = grid.shape
            result = grid.copy()
            
            # Completar simetría horizontal
            for i in range(h):
                for j in range(w // 2):
                    if result[i, j] != 0 and result[i, w - 1 - j] == 0:
                        result[i, w - 1 - j] = result[i, j]
                    elif result[i, w - 1 - j] != 0 and result[i, j] == 0:
                        result[i, j] = result[i, w - 1 - j]
            
            return result
        
        return transform
    
    def _create_extraction_transform(self, train_examples: List[Dict]):
        """Crea transformación de extracción de patrones"""
        # Buscar subregión común
        def transform(grid):
            # Encontrar bounding box del contenido no-cero
            non_zero = np.argwhere(grid != 0)
            if len(non_zero) == 0:
                return grid
            
            min_row, min_col = non_zero.min(axis=0)
            max_row, max_col = non_zero.max(axis=0)
            
            # Extraer subregión
            return grid[min_row:max_row+1, min_col:max_col+1]
        
        return transform
    
    def _create_fill_transform(self, train_examples: List[Dict]):
        """Crea transformación de relleno de formas"""
        # Detectar color de relleno
        fill_color = None
        
        for example in train_examples:
            output_arr = np.array(example['output'])
            unique_colors = np.unique(output_arr)
            
            # El color más común podría ser el de relleno
            if len(unique_colors) > 1:
                counts = [np.sum(output_arr == c) for c in unique_colors if c != 0]
                if counts:
                    max_idx = np.argmax(counts)
                    potential_fill = unique_colors[unique_colors != 0][max_idx]
                    
                    if fill_color is None:
                        fill_color = potential_fill
        
        if fill_color:
            def transform(grid):
                from scipy import ndimage
                result = grid.copy()
                
                # Rellenar regiones cerradas
                binary = (grid != 0).astype(int)
                filled = ndimage.binary_fill_holes(binary)
                result[filled & (grid == 0)] = fill_color
                
                return result
            
            return transform
        
        return None
    
    def _create_counting_transform(self, train_examples: List[Dict]):
        """Crea transformación basada en conteo"""
        def transform(grid):
            # Contar objetos únicos
            from scipy import ndimage
            
            # Etiquetar componentes conectados
            labeled, num_features = ndimage.label(grid != 0)
            
            # Crear grid de salida con el conteo
            result = np.full((1, 1), num_features)
            
            return result
        
        return transform
    
    def _create_translation_transform(self, train_examples: List[Dict]):
        """Crea transformación de traslación"""
        # Detectar desplazamiento
        shift_y, shift_x = 0, 0
        
        for example in train_examples:
            input_arr = np.array(example['input'])
            output_arr = np.array(example['output'])
            
            if input_arr.shape == output_arr.shape:
                # Buscar desplazamiento
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        shifted = np.roll(np.roll(input_arr, dy, axis=0), dx, axis=1)
                        if np.array_equal(shifted, output_arr):
                            shift_y, shift_x = dy, dx
                            break
        
        if shift_y != 0 or shift_x != 0:
            def transform(grid):
                return np.roll(np.roll(grid, shift_y, axis=0), shift_x, axis=1)
            return transform
        
        return None