#!/usr/bin/env python3
"""
Hyperparameter Optimizer - Optimización de hiperparámetros para PROTEUS AGI
Grid search y optimización adaptativa para mejorar rendimiento
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Resultado de optimización de hiperparámetros"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    total_evaluations: int

class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para el sistema PROTEUS AGI
    Implementa grid search, random search y optimización adaptativa
    """
    
    def __init__(self):
        self.optimization_history = []
        self.best_configurations = []
        
    def optimize_swarm_parameters(self, 
                                 solver_class: type,
                                 evaluation_data: List[Dict],
                                 method: str = 'grid_search',
                                 max_evaluations: int = 50,
                                 n_jobs: int = 1,
                                 **kwargs) -> OptimizationResult:
        """
        Optimiza parámetros del enjambre evolutivo
        
        Args:
            solver_class: Clase del solver a optimizar
            evaluation_data: Datos para evaluación (puzzles)
            method: Método de optimización ('grid_search', 'random_search', 'adaptive')
            max_evaluations: Máximo número de evaluaciones
            n_jobs: Número de procesos paralelos
            
        Returns:
            Resultado de optimización
        """
        start_time = time.time()
        
        # Definir espacio de búsqueda
        param_space = self._get_swarm_parameter_space()
        
        logger.info(f"Iniciando optimización de hiperparámetros - Método: {method}")
        logger.info(f"Espacio de parámetros: {len(param_space)} dimensiones")
        logger.info(f"Máximo evaluaciones: {max_evaluations}")
        
        if method == 'grid_search':
            results = self._grid_search_optimization(
                solver_class, param_space, evaluation_data, max_evaluations, n_jobs
            )
        elif method == 'random_search':
            results = self._random_search_optimization(
                solver_class, param_space, evaluation_data, max_evaluations, n_jobs
            )
        elif method == 'adaptive':
            results = self._adaptive_optimization(
                solver_class, param_space, evaluation_data, max_evaluations, n_jobs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Encontrar mejores parámetros
        best_result = max(results, key=lambda x: x['score'])
        
        optimization_time = time.time() - start_time
        
        opt_result = OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results,
            optimization_time=optimization_time,
            total_evaluations=len(results)
        )
        
        # Guardar en historial
        self.optimization_history.append({
            'timestamp': time.time(),
            'method': method,
            'result': opt_result
        })
        
        logger.info(f"Optimización completada en {optimization_time:.2f}s")
        logger.info(f"Mejores parámetros: {best_result['params']}")
        logger.info(f"Mejor puntuación: {best_result['score']:.4f}")
        
        return opt_result
    
    def _get_swarm_parameter_space(self) -> Dict[str, List]:
        """Define espacio de búsqueda para parámetros del enjambre"""
        return {
            'population_size': [10, 20, 30, 40, 50],
            'generations': [3, 5, 8, 10, 15],
            'mutation_rate': [0.1, 0.2, 0.3, 0.35, 0.4, 0.5],
            'crossover_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
            'elite_size': [2, 3, 5, 8, 10],
            'specialization_ratios': [
                {'color': 0.3, 'pattern': 0.2, 'topology': 0.1, 
                 'replication': 0.1, 'symmetry': 0.1, 'counting': 0.1, 'generalist': 0.1},
                {'color': 0.4, 'pattern': 0.15, 'topology': 0.15, 
                 'replication': 0.1, 'symmetry': 0.1, 'counting': 0.05, 'generalist': 0.05},
                {'color': 0.2, 'pattern': 0.2, 'topology': 0.2, 
                 'replication': 0.15, 'symmetry': 0.15, 'counting': 0.05, 'generalist': 0.05},
                {'generalist': 1.0},  # Solo generalistas
                {'color': 0.5, 'generalist': 0.5},  # Solo color y generalistas
            ]
        }
    
    def _grid_search_optimization(self, 
                                 solver_class: type,
                                 param_space: Dict[str, List],
                                 evaluation_data: List[Dict],
                                 max_evaluations: int,
                                 n_jobs: int) -> List[Dict[str, Any]]:
        """Optimización por grid search"""
        
        # Generar todas las combinaciones posibles
        param_names = list(param_space.keys())
        param_values = [param_space[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limitar número de evaluaciones
        if len(all_combinations) > max_evaluations:
            # Muestrear aleatoriamente
            import random
            selected_combinations = random.sample(all_combinations, max_evaluations)
            logger.info(f"Muestreando {max_evaluations} de {len(all_combinations)} combinaciones")
        else:
            selected_combinations = all_combinations
        
        # Crear configuraciones de parámetros
        param_configs = []
        for combination in selected_combinations:
            config = dict(zip(param_names, combination))
            param_configs.append(config)
        
        # Evaluar configuraciones
        if n_jobs == 1:
            results = []
            for i, config in enumerate(param_configs):
                logger.info(f"Evaluando configuración {i+1}/{len(param_configs)}")
                score = self._evaluate_parameter_configuration(
                    solver_class, config, evaluation_data
                )
                results.append({
                    'params': config,
                    'score': score,
                    'evaluation_id': i
                })
        else:
            results = self._parallel_evaluation(
                solver_class, param_configs, evaluation_data, n_jobs
            )
        
        return results
    
    def _random_search_optimization(self,
                                   solver_class: type,
                                   param_space: Dict[str, List],
                                   evaluation_data: List[Dict],
                                   max_evaluations: int,
                                   n_jobs: int) -> List[Dict[str, Any]]:
        """Optimización por random search"""
        import random
        
        param_configs = []
        
        for i in range(max_evaluations):
            # Generar configuración aleatoria
            config = {}
            for param_name, param_values in param_space.items():
                config[param_name] = random.choice(param_values)
            
            param_configs.append(config)
        
        logger.info(f"Generadas {len(param_configs)} configuraciones aleatorias")
        
        # Evaluar configuraciones
        if n_jobs == 1:
            results = []
            for i, config in enumerate(param_configs):
                logger.info(f"Evaluando configuración aleatoria {i+1}/{len(param_configs)}")
                score = self._evaluate_parameter_configuration(
                    solver_class, config, evaluation_data
                )
                results.append({
                    'params': config,
                    'score': score,
                    'evaluation_id': i
                })
        else:
            results = self._parallel_evaluation(
                solver_class, param_configs, evaluation_data, n_jobs
            )
        
        return results
    
    def _adaptive_optimization(self,
                              solver_class: type,
                              param_space: Dict[str, List],
                              evaluation_data: List[Dict],
                              max_evaluations: int,
                              n_jobs: int) -> List[Dict[str, Any]]:
        """Optimización adaptativa (Bayesian-like)"""
        
        results = []
        
        # Fase 1: Exploración inicial con random search
        initial_evaluations = min(max_evaluations // 3, 15)
        initial_results = self._random_search_optimization(
            solver_class, param_space, evaluation_data, initial_evaluations, n_jobs
        )
        results.extend(initial_results)
        
        # Fase 2: Explotación de regiones prometedoras
        remaining_evaluations = max_evaluations - len(results)
        
        if remaining_evaluations > 0:
            # Identificar mejores configuraciones
            sorted_results = sorted(initial_results, key=lambda x: x['score'], reverse=True)
            top_configs = sorted_results[:min(5, len(sorted_results))]
            
            # Generar variaciones de las mejores configuraciones
            exploitation_configs = []
            for top_config in top_configs:
                variations = self._generate_parameter_variations(
                    top_config['params'], param_space, remaining_evaluations // len(top_configs)
                )
                exploitation_configs.extend(variations)
            
            # Limitar a evaluaciones restantes
            exploitation_configs = exploitation_configs[:remaining_evaluations]
            
            logger.info(f"Fase de explotación: {len(exploitation_configs)} variaciones")
            
            # Evaluar variaciones
            if n_jobs == 1:
                for i, config in enumerate(exploitation_configs):
                    logger.info(f"Evaluando variación {i+1}/{len(exploitation_configs)}")
                    score = self._evaluate_parameter_configuration(
                        solver_class, config, evaluation_data
                    )
                    results.append({
                        'params': config,
                        'score': score,
                        'evaluation_id': len(results)
                    })
            else:
                exploitation_results = self._parallel_evaluation(
                    solver_class, exploitation_configs, evaluation_data, n_jobs
                )
                results.extend(exploitation_results)
        
        return results
    
    def _generate_parameter_variations(self,
                                      base_config: Dict[str, Any],
                                      param_space: Dict[str, List],
                                      num_variations: int) -> List[Dict[str, Any]]:
        """Genera variaciones de una configuración base"""
        import random
        
        variations = []
        
        for _ in range(num_variations):
            variation = base_config.copy()
            
            # Modificar aleatoriamente 1-2 parámetros
            params_to_modify = random.sample(
                list(param_space.keys()), 
                random.randint(1, min(2, len(param_space)))
            )
            
            for param_name in params_to_modify:
                # Seleccionar valor cercano al actual si es numérico
                current_value = base_config[param_name]
                available_values = param_space[param_name]
                
                if isinstance(current_value, (int, float)):
                    # Encontrar índice del valor actual
                    try:
                        current_idx = available_values.index(current_value)
                        # Seleccionar valor adyacente
                        neighbor_indices = [
                            max(0, current_idx - 1),
                            min(len(available_values) - 1, current_idx + 1)
                        ]
                        neighbor_idx = random.choice(neighbor_indices)
                        variation[param_name] = available_values[neighbor_idx]
                    except ValueError:
                        # Valor actual no está en la lista, seleccionar aleatoriamente
                        variation[param_name] = random.choice(available_values)
                else:
                    # Para valores no numéricos, seleccionar aleatoriamente
                    variation[param_name] = random.choice(available_values)
            
            variations.append(variation)
        
        return variations
    
    def _parallel_evaluation(self,
                            solver_class: type,
                            param_configs: List[Dict[str, Any]],
                            evaluation_data: List[Dict],
                            n_jobs: int) -> List[Dict[str, Any]]:
        """Evaluación paralela de configuraciones"""
        
        results = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Enviar trabajos
            future_to_config = {}
            for i, config in enumerate(param_configs):
                future = executor.submit(
                    self._evaluate_parameter_configuration,
                    solver_class, config, evaluation_data
                )
                future_to_config[future] = (config, i)
            
            # Recoger resultados
            completed = 0
            for future in as_completed(future_to_config):
                config, evaluation_id = future_to_config[future]
                
                try:
                    score = future.result()
                    results.append({
                        'params': config,
                        'score': score,
                        'evaluation_id': evaluation_id
                    })
                    completed += 1
                    logger.info(f"Completada evaluación {completed}/{len(param_configs)}")
                    
                except Exception as e:
                    logger.error(f"Error evaluando configuración {evaluation_id}: {e}")
                    results.append({
                        'params': config,
                        'score': 0.0,  # Penalizar configuraciones que fallan
                        'evaluation_id': evaluation_id,
                        'error': str(e)
                    })
        
        return results
    
    def _evaluate_parameter_configuration(self,
                                         solver_class: type,
                                         config: Dict[str, Any],
                                         evaluation_data: List[Dict]) -> float:
        """
        Evalúa una configuración específica de parámetros
        
        Returns:
            Puntuación de la configuración (mayor = mejor)
        """
        try:
            # Crear instancia del solver con configuración específica
            solver_params = self._extract_solver_parameters(config)
            solver = solver_class(**solver_params)
            
            # Si hay parámetros de especialización, aplicarlos
            if 'specialization_ratios' in config:
                solver.specialization_ratios = config['specialization_ratios']
            
            # Evaluar en datos de prueba
            scores = []
            max_puzzles = min(3, len(evaluation_data))  # Limitar para velocidad
            
            for puzzle_data in evaluation_data[:max_puzzles]:
                try:
                    # Resolver puzzle
                    solution, metadata = solver.solve_with_swarm(
                        puzzle_data['train'], puzzle_data['test_input']
                    )
                    
                    # Extraer puntuación
                    if metadata and 'best_fitness' in metadata:
                        score = metadata['best_fitness']
                    else:
                        score = 0.0
                    
                    scores.append(score)
                    
                except Exception as e:
                    logger.debug(f"Error resolviendo puzzle: {e}")
                    scores.append(0.0)
            
            # Calcular puntuación promedio
            avg_score = np.mean(scores) if scores else 0.0
            
            # Penalizar configuraciones muy costosas computacionalmente
            penalty_factor = self._calculate_computational_penalty(config)
            
            final_score = avg_score * penalty_factor
            
            return float(final_score)
            
        except Exception as e:
            logger.error(f"Error evaluando configuración: {e}")
            return 0.0
    
    def _extract_solver_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae parámetros válidos para el constructor del solver"""
        solver_params = {}
        
        valid_params = [
            'population_size', 'generations', 'mutation_rate', 
            'crossover_rate', 'elite_size'
        ]
        
        for param in valid_params:
            if param in config:
                solver_params[param] = config[param]
        
        return solver_params
    
    def _calculate_computational_penalty(self, config: Dict[str, Any]) -> float:
        """
        Calcula penalización por costo computacional
        
        Returns:
            Factor de penalización (0.5 - 1.0)
        """
        # Factores de costo
        population_size = config.get('population_size', 30)
        generations = config.get('generations', 5)
        
        # Costo estimado (normalizado)
        computational_cost = (population_size * generations) / (50 * 15)  # Max normalizado
        
        # Penalización suave
        penalty = max(0.5, 1.0 - computational_cost * 0.3)
        
        return penalty
    
    def optimize_puzzle_specific_parameters(self,
                                          solver_class: type,
                                          puzzle_characteristics: Dict[str, Any],
                                          evaluation_data: List[Dict]) -> Dict[str, Any]:
        """
        Optimiza parámetros específicos para características de puzzle
        
        Args:
            puzzle_characteristics: Características del puzzle (tamaño, colores, etc.)
            
        Returns:
            Parámetros optimizados para este tipo de puzzle
        """
        logger.info("Optimizando parámetros específicos para puzzle")
        logger.info(f"Características: {puzzle_characteristics}")
        
        # Determinar parámetros base según características
        base_params = self._get_puzzle_specific_base_parameters(puzzle_characteristics)
        
        # Refinar con búsqueda local
        refined_params = self._local_parameter_refinement(
            solver_class, base_params, evaluation_data
        )
        
        return refined_params
    
    def _get_puzzle_specific_base_parameters(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene parámetros base según características del puzzle"""
        
        params = {
            'population_size': 30,
            'generations': 5,
            'mutation_rate': 0.35,
            'crossover_rate': 0.7,
            'elite_size': 5
        }
        
        # Ajustar según tamaño del puzzle
        puzzle_size = characteristics.get('size', 100)
        if puzzle_size > 400:  # Puzzles grandes
            params['population_size'] = 40
            params['generations'] = 8
        elif puzzle_size < 50:  # Puzzles pequeños
            params['population_size'] = 20
            params['generations'] = 3
        
        # Ajustar según número de colores
        num_colors = characteristics.get('num_colors', 5)
        if num_colors > 7:  # Muchos colores
            params['mutation_rate'] = 0.4
            # Mayor proporción de agentes de color
            params['specialization_ratios'] = {
                'color': 0.5, 'pattern': 0.15, 'topology': 0.1,
                'replication': 0.1, 'symmetry': 0.05, 'counting': 0.05, 'generalist': 0.05
            }
        elif num_colors <= 3:  # Pocos colores
            # Mayor proporción de agentes de patrón y topología
            params['specialization_ratios'] = {
                'color': 0.2, 'pattern': 0.25, 'topology': 0.25,
                'replication': 0.1, 'symmetry': 0.1, 'counting': 0.05, 'generalist': 0.05
            }
        
        # Ajustar según complejidad
        complexity = characteristics.get('complexity', 0.5)
        if complexity > 0.7:  # Alta complejidad
            params['generations'] = max(params['generations'], 8)
            params['elite_size'] = max(params['elite_size'], 8)
        
        return params
    
    def _local_parameter_refinement(self,
                                   solver_class: type,
                                   base_params: Dict[str, Any],
                                   evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Refinamiento local de parámetros"""
        
        current_params = base_params.copy()
        current_score = self._evaluate_parameter_configuration(
            solver_class, current_params, evaluation_data
        )
        
        logger.info(f"Puntuación base: {current_score:.4f}")
        
        # Parámetros a refinar
        refinement_params = {
            'mutation_rate': [current_params['mutation_rate'] * 0.8, 
                             current_params['mutation_rate'] * 1.2],
            'crossover_rate': [max(0.5, current_params['crossover_rate'] - 0.1),
                              min(0.9, current_params['crossover_rate'] + 0.1)],
            'population_size': [max(10, current_params['population_size'] - 10),
                               current_params['population_size'] + 10]
        }
        
        best_params = current_params.copy()
        best_score = current_score
        
        # Probar variaciones
        for param_name, param_range in refinement_params.items():
            for param_value in param_range:
                test_params = current_params.copy()
                test_params[param_name] = param_value
                
                test_score = self._evaluate_parameter_configuration(
                    solver_class, test_params, evaluation_data
                )
                
                if test_score > best_score:
                    best_score = test_score
                    best_params = test_params.copy()
                    logger.info(f"Mejora encontrada: {param_name}={param_value}, score={test_score:.4f}")
        
        logger.info(f"Puntuación final refinada: {best_score:.4f}")
        
        return best_params
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Genera recomendaciones basadas en historial de optimización
        
        Returns:
            Recomendaciones de configuración
        """
        if not self.optimization_history:
            return self._get_default_recommendations()
        
        # Analizar historial
        all_results = []
        for entry in self.optimization_history:
            all_results.extend(entry['result'].all_results)
        
        # Encontrar mejores configuraciones
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:10]
        
        # Analizar patrones en mejores configuraciones
        recommendations = self._analyze_best_configurations(top_results)
        
        return recommendations
    
    def _analyze_best_configurations(self, top_results: List[Dict]) -> Dict[str, Any]:
        """Analiza patrones en mejores configuraciones"""
        
        param_analysis = defaultdict(list)
        
        # Recopilar valores de parámetros en mejores configuraciones
        for result in top_results:
            params = result['params']
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    param_analysis[param_name].append(param_value)
        
        # Calcular estadísticas
        recommendations = {}
        
        for param_name, values in param_analysis.items():
            if values:
                recommendations[param_name] = {
                    'recommended_value': float(np.median(values)),
                    'range': [float(np.min(values)), float(np.max(values))],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        # Añadir recomendaciones generales
        recommendations['general'] = {
            'best_score_achieved': float(max(r['score'] for r in top_results)),
            'avg_score_top10': float(np.mean([r['score'] for r in top_results])),
            'optimization_runs': len(self.optimization_history)
        }
        
        return recommendations
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Recomendaciones por defecto cuando no hay historial"""
        return {
            'population_size': {'recommended_value': 30, 'range': [20, 50]},
            'generations': {'recommended_value': 5, 'range': [3, 10]},
            'mutation_rate': {'recommended_value': 0.35, 'range': [0.2, 0.5]},
            'crossover_rate': {'recommended_value': 0.7, 'range': [0.6, 0.8]},
            'elite_size': {'recommended_value': 5, 'range': [3, 8]},
            'general': {
                'note': 'Estas son recomendaciones por defecto. Ejecute optimización para obtener parámetros específicos.'
            }
        }
    
    def save_optimization_results(self, filepath: str):
        """Guarda resultados de optimización"""
        data = {
            'optimization_history': self.optimization_history,
            'recommendations': self.get_optimization_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Resultados de optimización guardados en: {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """Carga resultados de optimización previos"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.optimization_history = data.get('optimization_history', [])
            logger.info(f"Cargados {len(self.optimization_history)} resultados previos")
            
        except Exception as e:
            logger.error(f"Error cargando resultados de optimización: {e}")


def main():
    """Función de prueba del optimizador"""
    print("\n" + "="*70)
    print("🔧 HYPERPARAMETER OPTIMIZER - PRUEBA")
    print("="*70)
    
    # Ejemplo de uso
    optimizer = HyperparameterOptimizer()
    
    # Simular datos de evaluación
    evaluation_data = [
        {
            'train': [{'input': [[1, 2]], 'output': [[2, 3]]}],
            'test_input': [[3, 4]]
        }
    ]
    
    # Mostrar recomendaciones por defecto
    recommendations = optimizer.get_optimization_recommendations()
    print("\n📋 Recomendaciones de parámetros:")
    for param, info in recommendations.items():
        if param != 'general':
            print(f"  {param}: {info.get('recommended_value', 'N/A')}")
    
    print("\n✅ Optimizador inicializado correctamente")
    print("="*70)


if __name__ == "__main__":
    main()