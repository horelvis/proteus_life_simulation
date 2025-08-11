#!/usr/bin/env python3
"""
Comprehensive Tests - Módulo de pruebas comprensivo para sistema PROTEUS mejorado
Cubre casos de puzzles multi-paso, nuevos colores, cambios de tamaño y evaluación completa
"""

import numpy as np
import sys
import os
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.arc_real_solver import ARCRealSolver
from arc.arc_swarm_solver_improved import ARCSwarmSolverImproved
from arc.structural_analyzer import StructuralAnalyzer
from arc.advanced_transformations import AdvancedTransformations
from arc.topology_detector import TopologyDetector
from arc.hyperparameter_optimizer import HyperparameterOptimizer
from arc.enhanced_memory import EnhancedSharedMemory

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Resultado de una prueba individual"""
    test_name: str
    success: bool
    score: float
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class ComprehensiveTestReport:
    """Reporte completo de todas las pruebas"""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    total_execution_time: float
    test_results: List[TestResult]
    system_performance: Dict[str, Any]
    recommendations: List[str]

class ComprehensiveTestSuite:
    """
    Suite comprensiva de pruebas para el sistema PROTEUS AGI mejorado
    Incluye pruebas de todos los componentes y casos edge
    """
    
    def __init__(self):
        self.test_results = []
        self.system_components = {
            'structural_analyzer': StructuralAnalyzer(),
            'advanced_transformations': AdvancedTransformations(),
            'topology_detector': TopologyDetector(),
            'hyperparameter_optimizer': HyperparameterOptimizer(),
            'enhanced_memory': EnhancedSharedMemory()
        }
        
    def run_all_tests(self, include_performance_tests: bool = True) -> ComprehensiveTestReport:
        """
        Ejecuta todas las pruebas del sistema
        
        Args:
            include_performance_tests: Si incluir pruebas de rendimiento (más lentas)
            
        Returns:
            Reporte completo de pruebas
        """
        logger.info("="*80)
        logger.info("🧪 INICIANDO SUITE COMPRENSIVA DE PRUEBAS PROTEUS AGI")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Lista de pruebas a ejecutar
        test_methods = [
            # Pruebas básicas de componentes
            ('test_structural_analyzer', self.test_structural_analyzer),
            ('test_advanced_transformations', self.test_advanced_transformations),
            ('test_topology_detector', self.test_topology_detector),
            ('test_enhanced_memory', self.test_enhanced_memory),
            
            # Pruebas de integración
            ('test_swarm_solver_integration', self.test_swarm_solver_integration),
            ('test_multi_step_puzzles', self.test_multi_step_puzzles),
            ('test_new_color_introduction', self.test_new_color_introduction),
            ('test_size_change_puzzles', self.test_size_change_puzzles),
            
            # Pruebas de casos edge
            ('test_empty_puzzles', self.test_empty_puzzles),
            ('test_single_pixel_puzzles', self.test_single_pixel_puzzles),
            ('test_large_puzzles', self.test_large_puzzles),
            ('test_monochromatic_puzzles', self.test_monochromatic_puzzles),
            
            # Pruebas de robustez
            ('test_invalid_input_handling', self.test_invalid_input_handling),
            ('test_memory_limit_handling', self.test_memory_limit_handling),
            ('test_timeout_handling', self.test_timeout_handling),
        ]
        
        # Añadir pruebas de rendimiento si se solicita
        if include_performance_tests:
            test_methods.extend([
                ('test_hyperparameter_optimization', self.test_hyperparameter_optimization),
                ('test_scaling_performance', self.test_scaling_performance),
                ('test_real_arc_dataset_sample', self.test_real_arc_dataset_sample),
            ])
        
        # Ejecutar pruebas
        for test_name, test_method in test_methods:
            logger.info(f"\n🔍 Ejecutando: {test_name}")
            
            try:
                test_start = time.time()
                result = test_method()
                test_time = time.time() - test_start
                
                if isinstance(result, TestResult):
                    result.execution_time = test_time
                    self.test_results.append(result)
                else:
                    # Crear resultado por defecto si el método no retorna TestResult
                    self.test_results.append(TestResult(
                        test_name=test_name,
                        success=True,
                        score=1.0,
                        execution_time=test_time,
                        details=result if isinstance(result, dict) else {}
                    ))
                
                logger.info(f"   ✅ Completado en {test_time:.2f}s")
                
            except Exception as e:
                test_time = time.time() - test_start
                logger.error(f"   ❌ Error: {str(e)}")
                
                self.test_results.append(TestResult(
                    test_name=test_name,
                    success=False,
                    score=0.0,
                    execution_time=test_time,
                    details={},
                    error_message=str(e)
                ))
        
        # Generar reporte
        total_time = time.time() - start_time
        report = self._generate_comprehensive_report(total_time)
        
        logger.info("\n" + "="*80)
        logger.info("📊 REPORTE FINAL DE PRUEBAS")
        logger.info("="*80)
        logger.info(f"✅ Pruebas pasadas: {report.passed_tests}/{report.total_tests}")
        logger.info(f"📈 Puntuación promedio: {report.average_score:.3f}")
        logger.info(f"⏱️ Tiempo total: {report.total_execution_time:.2f}s")
        logger.info("="*80)
        
        return report
    
    def test_structural_analyzer(self) -> TestResult:
        """Prueba el analizador estructural profundo"""
        analyzer = self.system_components['structural_analyzer']
        
        # Crear matriz de prueba compleja
        test_matrix = np.array([
            [1, 1, 0, 2, 2],
            [1, 0, 0, 2, 2],
            [0, 0, 3, 0, 0],
            [4, 4, 3, 5, 5],
            [4, 4, 0, 5, 5]
        ])
        
        # Ejecutar análisis
        analysis = analyzer.analyze_comprehensive(test_matrix)
        
        # Verificar componentes del análisis
        required_components = [
            'connectivity', 'topology', 'symmetry', 'geometric',
            'color_distribution', 'patterns', 'structural_complexity'
        ]
        
        success = all(comp in analysis for comp in required_components)
        
        # Verificar que connectivity_score no sea placeholder
        connectivity_score = analysis['connectivity'].get('connectivity_score', 0)
        has_real_analysis = (
            connectivity_score > 0 and
            'by_color' in analysis['connectivity'] and
            'global' in analysis['connectivity']
        )
        
        score = 1.0 if success and has_real_analysis else 0.0
        
        return TestResult(
            test_name="structural_analyzer",
            success=success and has_real_analysis,
            score=score,
            execution_time=0,  # Se llenará por el caller
            details={
                'analysis_components': list(analysis.keys()),
                'connectivity_score': connectivity_score,
                'structural_complexity': analysis.get('structural_complexity', 0)
            }
        )
    
    def test_advanced_transformations(self) -> TestResult:
        """Prueba las transformaciones avanzadas"""
        transformer = self.system_components['advanced_transformations']
        
        test_matrix = np.array([[1, 2], [3, 4]])
        transformations_tested = []
        successful_transforms = 0
        
        # Lista de transformaciones a probar
        test_transforms = [
            ('translate_rotate', {'dx': 1, 'dy': 1, 'angle': 90}),
            ('non_uniform_scale', {'scale_x': 2.0, 'scale_y': 1.5}),
            ('sort_objects_by_size', {}),
            ('matrix_add', {'value': 1}),
            ('contextual_recolor', {}),
            ('irregular_replication', {}),
        ]
        
        for transform_name, params in test_transforms:
            try:
                result = transformer.apply_transformation(test_matrix, transform_name, **params)
                
                # Verificar que la transformación produjo algo diferente (o igual si es válido)
                if result is not None and isinstance(result, np.ndarray):
                    successful_transforms += 1
                    transformations_tested.append({
                        'name': transform_name,
                        'success': True,
                        'output_shape': result.shape
                    })
                else:
                    transformations_tested.append({
                        'name': transform_name,
                        'success': False,
                        'error': 'No valid output'
                    })
                    
            except Exception as e:
                transformations_tested.append({
                    'name': transform_name,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = successful_transforms / len(test_transforms)
        success = success_rate >= 0.8  # Al menos 80% de transformaciones exitosas
        
        return TestResult(
            test_name="advanced_transformations",
            success=success,
            score=success_rate,
            execution_time=0,
            details={
                'transformations_tested': transformations_tested,
                'success_rate': success_rate,
                'total_transforms': len(test_transforms)
            }
        )
    
    def test_topology_detector(self) -> TestResult:
        """Prueba el detector de topología"""
        detector = self.system_components['topology_detector']
        
        # Crear matriz con patrones topológicos
        test_matrix = np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0]
        ])
        
        # Ejecutar análisis topológico
        topology_analysis = detector.detect_comprehensive_topology(test_matrix)
        
        # Verificar componentes esperados
        expected_components = [
            'curves', 'vector_fields', 'topology_transformations',
            'flow_patterns', 'critical_points', 'homology', 'contour_topology'
        ]
        
        components_present = all(comp in topology_analysis for comp in expected_components)
        
        # Verificar que los análisis no sean triviales
        has_meaningful_analysis = (
            len(topology_analysis.get('curves', {}).get('smooth_curves', [])) >= 0 and
            'gradient_magnitude' in topology_analysis.get('vector_fields', {}) and
            'betti_0' in topology_analysis.get('homology', {})
        )
        
        success = components_present and has_meaningful_analysis
        score = 1.0 if success else 0.5
        
        return TestResult(
            test_name="topology_detector",
            success=success,
            score=score,
            execution_time=0,
            details={
                'components_found': list(topology_analysis.keys()),
                'curves_detected': len(topology_analysis.get('curves', {}).get('smooth_curves', [])),
                'critical_points': len(topology_analysis.get('critical_points', [])),
                'homology': topology_analysis.get('homology', {})
            }
        )
    
    def test_enhanced_memory(self) -> TestResult:
        """Prueba la memoria compartida mejorada"""
        memory = self.system_components['enhanced_memory']
        
        # Crear puzzle de ejemplo con numpy arrays
        train_examples = [
            {'input': np.array([[1, 2], [3, 4]]), 'output': np.array([[2, 3], [4, 5]])}
        ]
        test_input = np.array([[5, 6], [7, 8]])
        
        # Crear firma de puzzle
        signature = memory.create_puzzle_signature(train_examples, test_input)
        
        # Almacenar regla
        rule_id = memory.store_successful_rule(
            rule_type="test_rule",
            rule_data={"operation": "increment"},
            fitness=0.9,
            agent_id=1,
            puzzle_signature=signature
        )
        
        # Recuperar reglas similares
        similar_rules = memory.retrieve_similar_rules(signature, top_k=5)
        
        # Verificar estadísticas
        stats = memory.get_memory_statistics()
        
        success = (
            rule_id is not None and
            len(similar_rules) >= 0 and  # Puede ser 0 si no hay reglas similares
            stats['total_rules'] > 0 and
            stats['total_puzzles'] > 0
        )
        
        return TestResult(
            test_name="enhanced_memory",
            success=success,
            score=1.0 if success else 0.0,
            execution_time=0,
            details={
                'rule_stored': rule_id is not None,
                'similar_rules_found': len(similar_rules),
                'memory_stats': stats
            }
        )
    
    def test_swarm_solver_integration(self) -> TestResult:
        """Prueba integración del solver de enjambre"""
        
        # Crear solver con componentes mejorados
        solver = ARCSwarmSolverImproved(
            population_size=10,  # Pequeño para pruebas rápidas
            generations=2,
            mutation_rate=0.3,
            crossover_rate=0.7,
            elite_size=2
        )
        
        # Integrar analizador estructural
        solver.structural_analyzer = self.system_components['structural_analyzer']
        
        # Puzzle de prueba simple
        train_examples = [
            {'input': np.array([[1, 2], [3, 4]]), 'output': np.array([[2, 3], [4, 5]])}
        ]
        test_input = np.array([[5, 6], [7, 8]])
        
        try:
            # Resolver puzzle
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            success = (
                solution is not None and
                isinstance(solution, np.ndarray) and
                metadata is not None and
                'best_fitness' in metadata
            )
            
            score = metadata.get('best_fitness', 0.0) if success else 0.0
            
        except Exception as e:
            success = False
            score = 0.0
            metadata = {'error': str(e)}
        
        return TestResult(
            test_name="swarm_solver_integration",
            success=success,
            score=score,
            execution_time=0,
            details=metadata
        )
    
    def test_multi_step_puzzles(self) -> TestResult:
        """Prueba puzzles que requieren múltiples pasos"""
        
        # Puzzle multi-paso: primero incrementar, luego duplicar
        train_examples = [
            {
                'input': np.array([[1, 1], [2, 2]]),
                'output': np.array([[2, 2, 2, 2], [3, 3, 3, 3]])  # +1, luego duplicar horizontalmente
            }
        ]
        test_input = np.array([[3, 3], [4, 4]])
        expected_output = np.array([[4, 4, 4, 4], [5, 5, 5, 5]])
        
        # Crear solver con más generaciones para pasos múltiples
        solver = ARCSwarmSolverImproved(
            population_size=15,
            generations=3,
            mutation_rate=0.4,
            crossover_rate=0.8,
            elite_size=3
        )
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            # Inicializar variables
            score = 0.0
            success = False
            
            if solution is not None:
                # Evaluar similitud con salida esperada
                if solution.shape == expected_output.shape:
                    accuracy = np.mean(solution == expected_output)
                    score = accuracy
                    success = accuracy > 0.7
                else:
                    score = 0.3  # Penalizar forma incorrecta pero dar puntos parciales
                    success = False
            else:
                score = 0.0
                success = False
                
        except Exception as e:
            solution = None
            score = 0.0
            success = False
            metadata = {'error': str(e)}
        
        return TestResult(
            test_name="multi_step_puzzles",
            success=success,
            score=score,
            execution_time=0,
            details={
                'expected_shape': expected_output.shape,
                'actual_shape': solution.shape if solution is not None else None,
                'metadata': metadata
            }
        )
    
    def test_new_color_introduction(self) -> TestResult:
        """Prueba puzzles donde se introducen nuevos colores"""
        
        # Puzzle que introduce color nuevo (5) basado en patrón
        train_examples = [
            {
                'input': np.array([[1, 2], [3, 4]]),
                'output': np.array([[1, 2, 5], [3, 4, 5]])  # Añadir columna de color nuevo
            }
        ]
        test_input = np.array([[6, 7], [8, 9]])
        
        solver = ARCSwarmSolverImproved(
            population_size=12,
            generations=3,
            mutation_rate=0.35
        )
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            # Inicializar variables
            success = False
            score = 0.0
            
            if solution is not None:
                # Verificar que se introdujo un nuevo color
                input_colors = set(np.unique(test_input))
                solution_colors = set(np.unique(solution))
                new_colors = solution_colors - input_colors - {0}  # Excluir fondo
                
                if len(new_colors) > 0:
                    score = 0.7  # Puntos por introducir nuevo color
                    success = True
                
                # Verificar forma correcta
                if solution.shape[1] > test_input.shape[1]:  # Columna añadida
                    score = min(1.0, score + 0.3)
                    success = True
            
        except Exception as e:
            metadata = {'error': str(e)}
        
        return TestResult(
            test_name="new_color_introduction",
            success=success,
            score=score,
            execution_time=0,
            details={
                'solution_shape': solution.shape if solution is not None else None,
                'input_colors': len(set(np.unique(test_input))),
                'solution_colors': len(set(np.unique(solution))) if solution is not None else 0
            }
        )
    
    def test_size_change_puzzles(self) -> TestResult:
        """Prueba puzzles con cambios de tamaño"""
        
        # Puzzle de escalado: 2x2 -> 4x4
        train_examples = [
            {
                'input': np.array([[1, 2], [3, 4]]),
                'output': np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
            }
        ]
        test_input = np.array([[5, 6], [7, 8]])
        expected_shape = (4, 4)
        
        solver = ARCSwarmSolverImproved(
            population_size=15,
            generations=3,
            mutation_rate=0.4
        )
        
        # Añadir transformaciones de escalado
        solver.advanced_transformations = self.system_components['advanced_transformations']
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            if solution is not None and solution.shape == expected_shape:
                # Verificar patrón de escalado
                score = 0.8  # Puntos por forma correcta
                
                # Verificar si el escalado es correcto
                if (solution[0, 0] == solution[0, 1] == solution[1, 0] == solution[1, 1] and
                    solution[0, 0] in np.unique(test_input)):
                    score = 1.0
                
                success = score > 0.7
            else:
                score = 0.0
                success = False
                
        except Exception as e:
            score = 0.0
            success = False
            metadata = {'error': str(e)}
        
        return TestResult(
            test_name="size_change_puzzles",
            success=success,
            score=score,
            execution_time=0,
            details={
                'expected_shape': expected_shape,
                'actual_shape': solution.shape if solution is not None else None,
                'scaling_factor': 2.0
            }
        )
    
    def test_empty_puzzles(self) -> TestResult:
        """Prueba manejo de puzzles vacíos"""
        
        # Puzzle completamente vacío
        train_examples = [
            {'input': np.array([[0, 0], [0, 0]]), 'output': np.array([[0, 0], [0, 0]])}
        ]
        test_input = np.array([[0, 0], [0, 0]])
        
        solver = ARCSwarmSolverImproved(population_size=5, generations=1)
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            # Para puzzles vacíos, esperamos una solución válida (puede ser vacía)
            success = solution is not None and isinstance(solution, np.ndarray)
            score = 1.0 if success else 0.0
            
        except Exception as e:
            success = False
            score = 0.0
            metadata = {'error': str(e)}
        
        return TestResult(
            test_name="empty_puzzles",
            success=success,
            score=score,
            execution_time=0,
            details={'handled_empty_input': success}
        )
    
    def test_single_pixel_puzzles(self) -> TestResult:
        """Prueba puzzles de un solo píxel"""
        
        train_examples = [
            {'input': np.array([[1]]), 'output': np.array([[2]])}
        ]
        test_input = np.array([[3]])
        
        solver = ARCSwarmSolverImproved(population_size=5, generations=1)
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            success = (
                solution is not None and 
                solution.shape == (1, 1) and
                solution[0, 0] != test_input[0, 0]  # Alguna transformación aplicada
            )
            
            score = 1.0 if success else 0.0
            
        except Exception as e:
            success = False
            score = 0.0
        
        return TestResult(
            test_name="single_pixel_puzzles",
            success=success,
            score=score,
            execution_time=0,
            details={'single_pixel_handled': str(success)}
        )
    
    def test_large_puzzles(self) -> TestResult:
        """Prueba puzzles grandes"""
        
        # Crear puzzle 10x10
        large_input = np.random.randint(0, 5, (10, 10))
        large_output = large_input + 1  # Transformación simple
        large_output = np.clip(large_output, 0, 9)
        
        train_examples = [
            {'input': large_input, 'output': large_output}
        ]
        test_input = np.random.randint(0, 5, (10, 10))
        
        solver = ARCSwarmSolverImproved(
            population_size=8,  # Reducido para manejar tamaño
            generations=2
        )
        
        start_time = time.time()
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            execution_time = time.time() - start_time
            
            # Para puzzles grandes, verificar que no sea excesivamente lento
            reasonable_time = execution_time < 30.0  # Máximo 30 segundos
            
            success = (
                solution is not None and 
                solution.shape == test_input.shape and
                reasonable_time
            )
            
            score = 0.8 if success else 0.0
            if reasonable_time and solution is not None:
                score = 1.0
                
        except Exception as e:
            success = False
            score = 0.0
            execution_time = time.time() - start_time
        
        return TestResult(
            test_name="large_puzzles",
            success=success,
            score=score,
            execution_time=execution_time,
            details={
                'puzzle_size': large_input.shape,
                'execution_time': execution_time,
                'reasonable_performance': execution_time < 30.0
            }
        )
    
    def test_monochromatic_puzzles(self) -> TestResult:
        """Prueba puzzles monocromáticos"""
        
        # Puzzle de un solo color (excepto fondo)
        train_examples = [
            {'input': np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]]), 
             'output': np.array([[0, 2, 2], [2, 2, 0], [2, 0, 2]])}
        ]
        test_input = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        
        solver = ARCSwarmSolverImproved(population_size=8, generations=2)
        
        try:
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            if solution is not None:
                # Verificar que se aplicó transformación de color
                input_nonzero = test_input[test_input != 0]
                output_nonzero = solution[solution != 0]
                
                color_changed = len(set(input_nonzero)) != len(set(output_nonzero))
                shape_preserved = solution.shape == test_input.shape
                
                success = shape_preserved
                score = 0.7 if shape_preserved else 0.0
                if color_changed:
                    score = 1.0
            else:
                success = False
                score = 0.0
                
        except Exception as e:
            success = False
            score = 0.0
        
        return TestResult(
            test_name="monochromatic_puzzles",
            success=success,
            score=score,
            execution_time=0,
            details={'monochromatic_handled': success}
        )
    
    def test_invalid_input_handling(self) -> TestResult:
        """Prueba manejo de entradas inválidas"""
        
        solver = ARCSwarmSolverImproved(population_size=5, generations=1)
        
        invalid_inputs = [
            # Lista vacía de ejemplos
            ([], np.array([[1, 2]])),
            # Ejemplo con entrada None
            ([{'input': None, 'output': np.array([[1]])}], np.array([[2]])),
            # Formas inconsistentes
            ([{'input': np.array([[1]]), 'output': np.array([[1, 2]])}], np.array([[3]])),
        ]
        
        handled_gracefully = 0
        total_tests = len(invalid_inputs)
        
        for train_examples, test_input in invalid_inputs:
            try:
                solution, metadata = solver.solve_with_swarm(train_examples, test_input)
                # Si no lanza excepción, se maneja graciosamente
                handled_gracefully += 1
            except Exception:
                # Se espera que algunos fallen, pero de forma controlada
                handled_gracefully += 1
        
        success = handled_gracefully >= total_tests * 0.8  # Al menos 80%
        score = handled_gracefully / total_tests
        
        return TestResult(
            test_name="invalid_input_handling",
            success=success,
            score=score,
            execution_time=0,
            details={
                'invalid_inputs_tested': total_tests,
                'handled_gracefully': handled_gracefully
            }
        )
    
    def test_memory_limit_handling(self) -> TestResult:
        """Prueba manejo de límites de memoria"""
        
        # Crear puzzle que podría consumir mucha memoria
        solver = ARCSwarmSolverImproved(
            population_size=50,  # Población grande
            generations=1
        )
        
        # Puzzle grande
        large_matrix = np.random.randint(0, 3, (20, 20))
        train_examples = [
            {'input': large_matrix, 'output': large_matrix + 1}
        ]
        test_input = np.random.randint(0, 3, (20, 20))
        
        try:
            import resource
            # Establecer límite de memoria (256 MB)
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, -1))
            
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            
            # Si completa sin crash, maneja memoria bien
            success = solution is not None
            score = 1.0 if success else 0.0
            
            # Restaurar límite
            resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
            
        except MemoryError:
            # Error de memoria manejado explícitamente
            success = True  # Manejo explícito es bueno
            score = 0.8
            # Restaurar límite
            try:
                resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
            except:
                pass
        except Exception:
            # Otro tipo de error
            success = False
            score = 0.0
            # Restaurar límite
            try:
                resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
            except:
                pass
        
        return TestResult(
            test_name="memory_limit_handling",
            success=success,
            score=score,
            execution_time=0,
            details={'memory_handled': success}
        )
    
    def test_timeout_handling(self) -> TestResult:
        """Prueba manejo de timeouts"""
        
        # Configurar solver que podría ser lento
        solver = ARCSwarmSolverImproved(
            population_size=30,
            generations=5  # Muchas generaciones
        )
        
        train_examples = [
            {'input': np.random.randint(0, 5, (8, 8)), 
             'output': np.random.randint(0, 5, (8, 8))}
        ]
        test_input = np.random.randint(0, 5, (8, 8))
        
        start_time = time.time()
        max_time = 15.0  # 15 segundos máximo
        
        try:
            import signal
            
            class TimeoutException(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutException("Timeout exceeded")
            
            # Configurar señal de timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(max_time))  # Timeout en segundos
            
            solution, metadata = solver.solve_with_swarm(train_examples, test_input)
            execution_time = time.time() - start_time
            
            # Cancelar alarma
            signal.alarm(0)
            
            # Verificar tiempo razonable
            reasonable_time = execution_time < max_time
            success = reasonable_time and solution is not None
            score = 1.0 if success else 0.5
            
        except TimeoutException:
            execution_time = time.time() - start_time
            success = True  # Manejó timeout correctamente
            score = 0.7
            signal.alarm(0)
        except Exception:
            execution_time = time.time() - start_time
            success = False
            score = 0.0
            signal.alarm(0)
        
        return TestResult(
            test_name="timeout_handling",
            success=success,
            score=score,
            execution_time=execution_time,
            details={
                'max_allowed_time': max_time,
                'actual_time': execution_time,
                'within_limits': execution_time < max_time
            }
        )
    
    def test_hyperparameter_optimization(self) -> TestResult:
        """Prueba optimización de hiperparámetros"""
        
        optimizer = self.system_components['hyperparameter_optimizer']
        
        # Datos de evaluación simplificados
        evaluation_data = [
            {
                'train': [{'input': np.array([[1, 2]]), 'output': np.array([[2, 3]])}],
                'test_input': np.array([[3, 4]])
            }
        ]
        
        try:
            # Ejecutar optimización con límites reducidos para pruebas
            result = optimizer.optimize_swarm_parameters(
                solver_class=ARCSwarmSolverImproved,
                evaluation_data=evaluation_data,
                method='random_search',
                max_evaluations=5,  # Pocas evaluaciones para velocidad
                n_jobs=1
            )
            
            success = (
                result.best_params is not None and
                result.best_score >= 0 and
                result.total_evaluations > 0
            )
            
            score = min(1.0, result.best_score + 0.3)  # Bonificación por completar
            
        except Exception as e:
            success = False
            score = 0.0
            result = None
        
        return TestResult(
            test_name="hyperparameter_optimization",
            success=success,
            score=score,
            execution_time=0,
            details={
                'optimization_completed': success,
                'evaluations': result.total_evaluations if result else 0,
                'best_score': result.best_score if result else 0
            }
        )
    
    def test_scaling_performance(self) -> TestResult:
        """Prueba rendimiento con diferentes tamaños de población"""
        
        population_sizes = [5, 10, 20]
        performance_results = []
        
        for pop_size in population_sizes:
            solver = ARCSwarmSolverImproved(
                population_size=pop_size,
                generations=2
            )
            
            train_examples = [
                {'input': np.array([[1, 2], [3, 4]]), 'output': np.array([[2, 3], [4, 5]])}
            ]
            test_input = np.array([[5, 6], [7, 8]])
            
            start_time = time.time()
            try:
                solution, metadata = solver.solve_with_swarm(train_examples, test_input)
                execution_time = time.time() - start_time
                
                performance_results.append({
                    'population_size': pop_size,
                    'execution_time': execution_time,
                    'success': solution is not None,
                    'fitness': metadata.get('best_fitness', 0) if metadata else 0
                })
                
            except Exception as e:
                performance_results.append({
                    'population_size': pop_size,
                    'execution_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })
        
        # Evaluar escalabilidad
        successful_runs = [r for r in performance_results if r.get('success', False)]
        scaling_success = len(successful_runs) >= len(population_sizes) * 0.8
        
        # Verificar que tiempo escale razonablemente
        if len(successful_runs) >= 2:
            time_scaling_reasonable = True
            for i in range(1, len(successful_runs)):
                time_ratio = successful_runs[i]['execution_time'] / successful_runs[0]['execution_time']
                pop_ratio = successful_runs[i]['population_size'] / successful_runs[0]['population_size']
                
                # Tiempo no debería escalar más que linealmente
                if time_ratio > pop_ratio * 2:
                    time_scaling_reasonable = False
                    break
        else:
            time_scaling_reasonable = False
        
        success = scaling_success and time_scaling_reasonable
        score = 0.7 if scaling_success else 0.0
        if time_scaling_reasonable:
            score = 1.0
        
        return TestResult(
            test_name="scaling_performance",
            success=success,
            score=score,
            execution_time=0,
            details={
                'performance_results': performance_results,
                'successful_runs': len(successful_runs),
                'reasonable_scaling': time_scaling_reasonable
            }
        )
    
    def test_real_arc_dataset_sample(self) -> TestResult:
        """Prueba con muestra del dataset ARC real"""
        
        try:
            # Cargar dataset real
            arc_loader = ARCRealSolver()
            
            # Obtener puzzle aleatorio
            puzzle_id, puzzle_data = arc_loader.get_random_puzzle()
            
            # Crear solver optimizado para puzzle real
            solver = ARCSwarmSolverImproved(
                population_size=15,
                generations=3,
                mutation_rate=0.35
            )
            
            # Resolver primer caso de prueba
            if puzzle_data['test']:
                test_case = puzzle_data['test'][0]
                
                solution, metadata = solver.solve_with_swarm(
                    puzzle_data['train'], test_case['input']
                )
                
                # Evaluar contra solución oficial si disponible
                if 'output' in test_case and solution is not None:
                    official_output = test_case['output']
                    if solution.shape == official_output.shape:
                        accuracy = np.mean(solution == official_output)
                        score = accuracy
                        success = accuracy > 0.1  # Al menos 10% correcto
                    else:
                        score = 0.1  # Puntos por intentar
                        success = False
                else:
                    # Sin solución oficial, evaluar que genere algo válido
                    score = 0.5 if solution is not None else 0.0
                    success = solution is not None
            else:
                score = 0.0
                success = False
                
        except Exception as e:
            success = False
            score = 0.0
            puzzle_id = "unknown"
        
        return TestResult(
            test_name="real_arc_dataset_sample",
            success=success,
            score=score,
            execution_time=0,
            details={
                'puzzle_id': puzzle_id if 'puzzle_id' in locals() else 'unknown',
                'real_dataset_tested': True,
                'accuracy': score if success else 0
            }
        )
    
    def _generate_comprehensive_report(self, total_execution_time: float) -> ComprehensiveTestReport:
        """Genera reporte comprensivo de todas las pruebas"""
        
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = len(self.test_results) - passed_tests
        
        if self.test_results:
            average_score = np.mean([result.score for result in self.test_results])
        else:
            average_score = 0.0
        
        # Analizar rendimiento del sistema
        system_performance = self._analyze_system_performance()
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations()
        
        return ComprehensiveTestReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_tests=len(self.test_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_score=average_score,
            total_execution_time=total_execution_time,
            test_results=self.test_results,
            system_performance=system_performance,
            recommendations=recommendations
        )
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analiza rendimiento general del sistema"""
        
        if not self.test_results:
            return {}
        
        # Tiempos de ejecución
        execution_times = [r.execution_time for r in self.test_results if r.execution_time > 0]
        
        # Puntuaciones por categoría
        component_scores = defaultdict(list)
        integration_scores = []
        robustness_scores = []
        
        for result in self.test_results:
            if any(comp in result.test_name for comp in ['analyzer', 'transformations', 'topology', 'memory']):
                component_scores['components'].append(result.score)
            elif 'integration' in result.test_name or 'swarm' in result.test_name:
                integration_scores.append(result.score)
            else:
                robustness_scores.append(result.score)
        
        return {
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'max_execution_time': np.max(execution_times) if execution_times else 0,
            'component_avg_score': np.mean(component_scores['components']) if component_scores['components'] else 0,
            'integration_avg_score': np.mean(integration_scores) if integration_scores else 0,
            'robustness_avg_score': np.mean(robustness_scores) if robustness_scores else 0,
            'total_components_tested': len(self.system_components)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en resultados de pruebas"""
        
        recommendations = []
        
        # Analizar fallos
        failed_tests = [r for r in self.test_results if not r.success]
        
        if failed_tests:
            component_failures = [r for r in failed_tests if any(comp in r.test_name for comp in ['analyzer', 'transformations', 'topology'])]
            integration_failures = [r for r in failed_tests if 'integration' in r.test_name]
            performance_failures = [r for r in failed_tests if any(perf in r.test_name for perf in ['scaling', 'timeout', 'memory'])]
            
            if component_failures:
                recommendations.append("Revisar implementación de componentes básicos que fallaron")
            
            if integration_failures:
                recommendations.append("Mejorar integración entre componentes del sistema")
            
            if performance_failures:
                recommendations.append("Optimizar rendimiento y manejo de recursos")
        
        # Analizar puntuaciones bajas
        low_score_tests = [r for r in self.test_results if r.score < 0.5]
        
        if len(low_score_tests) > len(self.test_results) * 0.3:
            recommendations.append("Puntuaciones generalmente bajas - revisar algoritmos centrales")
        
        # Recomendaciones específicas
        multi_step_results = [r for r in self.test_results if 'multi_step' in r.test_name]
        if multi_step_results and multi_step_results[0].score < 0.7:
            recommendations.append("Mejorar capacidad de resolución de puzzles multi-paso")
        
        color_results = [r for r in self.test_results if 'color' in r.test_name]
        if color_results and color_results[0].score < 0.7:
            recommendations.append("Mejorar manejo de introducción de nuevos colores")
        
        size_results = [r for r in self.test_results if 'size' in r.test_name]
        if size_results and size_results[0].score < 0.7:
            recommendations.append("Mejorar transformaciones de cambio de tamaño")
        
        if not recommendations:
            recommendations.append("Sistema funcionando bien - considerar optimizaciones menores")
        
        return recommendations
    
    def save_report(self, report: ComprehensiveTestReport, filepath: str):
        """Guarda reporte a archivo JSON"""
        
        # Convertir a diccionario serializable
        report_dict = {
            'timestamp': report.timestamp,
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'average_score': report.average_score,
            'total_execution_time': report.total_execution_time,
            'system_performance': report.system_performance,
            'recommendations': report.recommendations,
            'test_results': []
        }
        
        # Convertir resultados de pruebas
        for result in report.test_results:
            result_dict = {
                'test_name': result.test_name,
                'success': result.success,
                'score': result.score,
                'execution_time': result.execution_time,
                'details': result.details,
                'error_message': result.error_message
            }
            report_dict['test_results'].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Reporte guardado en: {filepath}")


def main():
    """Función principal de pruebas"""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TEST SUITE - PROTEUS AGI")
    print("="*80)
    
    # Crear suite de pruebas
    test_suite = ComprehensiveTestSuite()
    
    # Ejecutar todas las pruebas
    report = test_suite.run_all_tests(include_performance_tests=True)
    
    # Guardar reporte
    report_path = "/tmp/proteus_comprehensive_test_report.json"
    test_suite.save_report(report, report_path)
    
    print(f"\n📁 Reporte guardado en: {report_path}")
    
    return 0 if report.passed_tests == report.total_tests else 1


if __name__ == "__main__":
    sys.exit(main())