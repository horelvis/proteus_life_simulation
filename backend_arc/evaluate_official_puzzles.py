#!/usr/bin/env python3
"""
Evaluador para puzzles oficiales de ARC Prize
Mide el rendimiento real del sistema
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import logging

from arc_solver_python import ARCSolverPython
from arc_official_loader import ARCOfficialLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARCOfficialEvaluator:
    """
    Evalúa el solver con puzzles oficiales de ARC
    """
    
    def __init__(self, solver: ARCSolverPython):
        self.solver = solver
        self.loader = ARCOfficialLoader()
        self.results_dir = Path("official_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_puzzles(self, 
                        puzzle_ids: List[str] = None,
                        dataset: str = 'training',
                        save_results: bool = True) -> Dict[str, Any]:
        """
        Evalúa el solver con puzzles oficiales
        
        Args:
            puzzle_ids: Lista de IDs específicos o None para muestra
            dataset: 'training' o 'evaluation'
            save_results: Si guardar resultados detallados
            
        Returns:
            Resultados de evaluación
        """
        # Usar puzzles de muestra si no se especifican
        if puzzle_ids is None:
            puzzle_ids = self.loader.get_sample_puzzles()
        
        logger.info(f"🎯 Evaluando {len(puzzle_ids)} puzzles oficiales...")
        
        # Cargar puzzles
        puzzles = self.loader.load_specific_puzzles(puzzle_ids, dataset)
        
        if not puzzles:
            logger.error("No se pudieron cargar puzzles")
            return {}
        
        # Evaluar cada puzzle
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_puzzles': len(puzzles),
            'correct': 0,
            'partially_correct': 0,
            'failed': 0,
            'by_puzzle': {},
            'by_rule_type': {},
            'execution_times': []
        }
        
        for puzzle in puzzles:
            logger.info(f"\n📊 Evaluando puzzle {puzzle['id']}...")
            
            # Evaluar puzzle individual
            puzzle_result = self._evaluate_single_puzzle(puzzle)
            results['by_puzzle'][puzzle['id']] = puzzle_result
            
            # Actualizar contadores
            if puzzle_result['correct']:
                results['correct'] += 1
            elif puzzle_result['accuracy'] > 0.5:
                results['partially_correct'] += 1
            else:
                results['failed'] += 1
            
            # Agrupar por tipo de regla
            rule_type = puzzle_result.get('rule_detected', 'none')
            if rule_type not in results['by_rule_type']:
                results['by_rule_type'][rule_type] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy_sum': 0
                }
            
            results['by_rule_type'][rule_type]['total'] += 1
            if puzzle_result['correct']:
                results['by_rule_type'][rule_type]['correct'] += 1
            results['by_rule_type'][rule_type]['accuracy_sum'] += puzzle_result['accuracy']
            
            results['execution_times'].append(puzzle_result['execution_time_ms'])
        
        # Calcular métricas agregadas
        results['overall_accuracy'] = results['correct'] / results['total_puzzles']
        results['avg_execution_time_ms'] = np.mean(results['execution_times'])
        
        # Calcular accuracy por tipo de regla
        for rule_type, stats in results['by_rule_type'].items():
            stats['accuracy'] = stats['accuracy_sum'] / stats['total']
        
        # Guardar resultados si se solicita
        if save_results:
            self._save_results(results)
        
        # Imprimir resumen
        self._print_summary(results)
        
        return results
    
    def _evaluate_single_puzzle(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa un puzzle individual
        """
        start_time = datetime.now()
        
        result = {
            'puzzle_id': puzzle['id'],
            'correct': False,
            'accuracy': 0.0,
            'rule_detected': None,
            'confidence': 0.0,
            'error': None,
            'reasoning_steps': []
        }
        
        try:
            # Resolver puzzle
            test_input = np.array(puzzle['testExample']['input'])
            solution, steps = self.solver.solve_with_steps(
                puzzle['trainExamples'],
                test_input
            )
            
            # Guardar pasos de razonamiento
            result['reasoning_steps'] = steps
            
            # Detectar regla principal
            if puzzle['trainExamples']:
                rule = self.solver.detect_rule(
                    np.array(puzzle['trainExamples'][0]['input']),
                    np.array(puzzle['trainExamples'][0]['output'])
                )
                if rule:
                    result['rule_detected'] = rule['type']
                    result['confidence'] = rule['confidence']
            
            # Evaluar solución
            eval_metrics = self.loader.evaluate_solution(puzzle, solution.tolist())
            result.update(eval_metrics)
            
            # Análisis adicional si falló
            if not result['correct']:
                result['failure_analysis'] = self._analyze_failure(
                    puzzle, solution, steps
                )
            
        except Exception as e:
            logger.error(f"Error evaluando puzzle {puzzle['id']}: {e}")
            result['error'] = str(e)
        
        # Tiempo de ejecución
        result['execution_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
        
        return result
    
    def _analyze_failure(self, 
                        puzzle: Dict[str, Any],
                        predicted: np.ndarray,
                        steps: List[Dict]) -> Dict[str, Any]:
        """
        Analiza por qué falló una predicción
        """
        expected = np.array(puzzle['testExample']['output'])
        
        analysis = {
            'shape_matches': predicted.shape == expected.shape,
            'colors_used_predicted': sorted(np.unique(predicted).tolist()),
            'colors_used_expected': sorted(np.unique(expected).tolist()),
            'pixel_diff_count': np.sum(predicted != expected) if predicted.shape == expected.shape else -1,
            'rules_tried': []
        }
        
        # Extraer reglas intentadas de los pasos
        for step in steps:
            if step.get('type') == 'rule_detection':
                analysis['rules_tried'].append(step.get('rule', {}).get('type', 'unknown'))
        
        # Intentar detectar qué tipo de transformación era realmente
        analysis['possible_correct_rule'] = self._guess_correct_rule(puzzle)
        
        return analysis
    
    def _guess_correct_rule(self, puzzle: Dict[str, Any]) -> str:
        """
        Intenta adivinar cuál era la regla correcta analizando los ejemplos
        """
        # Análisis simple basado en cambios de tamaño
        for example in puzzle['trainExamples']:
            input_shape = np.array(example['input']).shape
            output_shape = np.array(example['output']).shape
            
            if output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1]:
                return "possibly_pattern_replication"
            elif output_shape[0] < input_shape[0] or output_shape[1] < input_shape[1]:
                if output_shape == (1, 1):
                    return "possibly_counting_or_detection"
                else:
                    return "possibly_pattern_extraction"
        
        return "unknown_complex_transformation"
    
    def _print_summary(self, results: Dict[str, Any]):
        """
        Imprime resumen de resultados
        """
        print("\n" + "="*60)
        print("📊 RESUMEN DE EVALUACIÓN OFICIAL")
        print("="*60)
        
        print(f"\n✅ Correctos: {results['correct']}/{results['total_puzzles']} "
              f"({results['overall_accuracy']*100:.1f}%)")
        print(f"⚡ Parcialmente correctos: {results['partially_correct']}")
        print(f"❌ Fallidos: {results['failed']}")
        print(f"⏱️  Tiempo promedio: {results['avg_execution_time_ms']:.1f}ms")
        
        print("\n📈 Rendimiento por tipo de regla:")
        for rule_type, stats in results['by_rule_type'].items():
            print(f"   {rule_type}: {stats['correct']}/{stats['total']} "
                  f"({stats['accuracy']*100:.1f}%)")
        
        print("\n🔍 Detalles por puzzle:")
        for puzzle_id, result in results['by_puzzle'].items():
            status = "✅" if result['correct'] else "❌"
            print(f"   {status} {puzzle_id}: {result['accuracy']*100:.0f}% "
                  f"(regla: {result.get('rule_detected', 'none')})")
            
            if not result['correct'] and 'failure_analysis' in result:
                print(f"      → Posible regla: {result['failure_analysis']['possible_correct_rule']}")
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Guarda resultados detallados
        """
        filename = f"official_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        # Convertir numpy types a Python types para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)
        
        logger.info(f"💾 Resultados guardados en: {filepath}")
    
    def compare_with_baseline(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara resultados con baselines conocidos
        """
        baselines = {
            'random': 0.0,  # ~0% (1/10^n donde n es número de píxeles)
            'gpt4': 0.12,   # ~12% según reportes oficiales
            'human_average': 0.84,  # ~84% humanos promedio
            'human_expert': 0.95    # ~95% expertos
        }
        
        comparison = {
            'vs_random': results['overall_accuracy'] / max(baselines['random'], 0.001),
            'vs_gpt4': results['overall_accuracy'] / baselines['gpt4'],
            'vs_human_average': results['overall_accuracy'] / baselines['human_average'],
            'vs_human_expert': results['overall_accuracy'] / baselines['human_expert']
        }
        
        print("\n🏆 Comparación con baselines:")
        print(f"   vs Random: {comparison['vs_random']:.1f}x mejor")
        print(f"   vs GPT-4: {comparison['vs_gpt4']:.1f}x "
              f"({'mejor' if comparison['vs_gpt4'] > 1 else 'peor'})")
        print(f"   vs Humano promedio: {comparison['vs_human_average']*100:.1f}% del rendimiento")
        print(f"   vs Humano experto: {comparison['vs_human_expert']*100:.1f}% del rendimiento")
        
        return comparison


def run_comprehensive_evaluation():
    """
    Ejecuta evaluación comprehensiva con puzzles oficiales
    """
    print("🚀 Iniciando evaluación con puzzles oficiales de ARC Prize\n")
    
    # Crear solver y evaluador
    solver = ARCSolverPython()
    evaluator = ARCOfficialEvaluator(solver)
    
    # Evaluar con puzzles de muestra (sin guardar por ahora)
    results = evaluator.evaluate_puzzles(save_results=False)
    
    # Comparar con baselines
    if results:
        evaluator.compare_with_baseline(results)
    
    return results


if __name__ == "__main__":
    # Ejecutar evaluación
    results = run_comprehensive_evaluation()
    
    # Análisis adicional
    if results and results['failed'] > 0:
        print("\n💡 Recomendaciones para mejorar:")
        print("1. Implementar más tipos de transformaciones")
        print("2. Mejorar detección de patrones complejos")
        print("3. Añadir razonamiento jerárquico")
        print("4. Considerar transformaciones compuestas")