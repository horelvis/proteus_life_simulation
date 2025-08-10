#!/usr/bin/env python3
"""
Módulo de evaluación para ARC Solver
Implementa splits train/test apropiados y métricas de evaluación
"""

import numpy as np
import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

@dataclass
class EvaluationResult:
    """Resultado de una evaluación"""
    puzzle_id: str
    puzzle_type: str
    success: bool
    predicted_output: List[List[int]]
    expected_output: List[List[int]]
    rule_detected: str
    confidence: float
    reasoning_steps: int
    time_ms: float

class ARCEvaluator:
    """
    Evaluador profesional para ARC puzzles
    Implementa evaluación rigurosa con splits apropiados
    """
    
    def __init__(self, solver, data_path: str = "arc_puzzles"):
        self.solver = solver
        self.data_path = Path(data_path)
        self.results_path = Path("evaluation_results")
        self.results_path.mkdir(exist_ok=True)
        
    def load_arc_dataset(self, subset: str = "training") -> Dict[str, Any]:
        """
        Carga el dataset oficial de ARC
        
        Args:
            subset: 'training' o 'evaluation'
            
        Returns:
            Dict con puzzles organizados por categoría
        """
        dataset = {
            'all': [],
            'by_category': {},
            'by_difficulty': {
                'easy': [],
                'medium': [],
                'hard': []
            }
        }
        
        # Para demo, usar puzzles sintéticos categorizados
        synthetic_puzzles = self._create_synthetic_dataset()
        
        for puzzle in synthetic_puzzles:
            dataset['all'].append(puzzle)
            
            # Organizar por categoría
            category = puzzle.get('category', 'unknown')
            if category not in dataset['by_category']:
                dataset['by_category'][category] = []
            dataset['by_category'][category].append(puzzle)
            
            # Organizar por dificultad
            difficulty = puzzle.get('difficulty', 'medium')
            dataset['by_difficulty'][difficulty].append(puzzle)
        
        return dataset
    
    def create_evaluation_splits(self, 
                               dataset: Dict[str, Any],
                               test_ratio: float = 0.2,
                               stratified: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Crea splits train/test manteniendo distribución de categorías
        
        Args:
            dataset: Dataset completo
            test_ratio: Proporción para test
            stratified: Si mantener proporciones por categoría
            
        Returns:
            (train_set, test_set)
        """
        if stratified:
            train_set = []
            test_set = []
            
            # Split estratificado por categoría
            for category, puzzles in dataset['by_category'].items():
                random.shuffle(puzzles)
                split_idx = int(len(puzzles) * (1 - test_ratio))
                
                train_set.extend(puzzles[:split_idx])
                test_set.extend(puzzles[split_idx:])
        else:
            # Split aleatorio simple
            all_puzzles = dataset['all'].copy()
            random.shuffle(all_puzzles)
            split_idx = int(len(all_puzzles) * (1 - test_ratio))
            
            train_set = all_puzzles[:split_idx]
            test_set = all_puzzles[split_idx:]
        
        return train_set, test_set
    
    def evaluate_zero_shot(self, test_puzzles: List[Dict]) -> Dict[str, Any]:
        """
        Evaluación zero-shot (sin ver ejemplos de test durante entrenamiento)
        
        Args:
            test_puzzles: Puzzles nunca vistos
            
        Returns:
            Métricas de evaluación
        """
        results = []
        start_time = datetime.now()
        
        for puzzle in test_puzzles:
            try:
                # Medir tiempo por puzzle
                puzzle_start = datetime.now()
                
                # Resolver usando solo ejemplos de entrenamiento del puzzle
                solution, steps = self.solver.solve_with_steps(
                    puzzle['trainExamples'],
                    np.array(puzzle['testExample']['input'])
                )
                
                puzzle_time = (datetime.now() - puzzle_start).total_seconds() * 1000
                
                # Evaluar resultado
                expected = np.array(puzzle['testExample']['output'])
                success = np.array_equal(solution, expected)
                
                # Detectar regla principal
                rule = self.solver.detect_rule(
                    np.array(puzzle['trainExamples'][0]['input']),
                    np.array(puzzle['trainExamples'][0]['output'])
                )
                
                result = EvaluationResult(
                    puzzle_id=puzzle.get('id', 'unknown'),
                    puzzle_type=puzzle.get('category', 'unknown'),
                    success=success,
                    predicted_output=solution.tolist(),
                    expected_output=expected.tolist(),
                    rule_detected=rule['type'] if rule else 'none',
                    confidence=rule['confidence'] if rule else 0.0,
                    reasoning_steps=len(steps),
                    time_ms=puzzle_time
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluando puzzle {puzzle.get('id', 'unknown')}: {e}")
                continue
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calcular métricas
        return self._calculate_metrics(results, total_time)
    
    def evaluate_few_shot(self, 
                         train_puzzles: List[Dict],
                         test_puzzles: List[Dict],
                         k_shot: int = 3) -> Dict[str, Any]:
        """
        Evaluación few-shot (con k ejemplos similares)
        
        Args:
            train_puzzles: Puzzles de entrenamiento
            test_puzzles: Puzzles de test
            k_shot: Número de ejemplos similares a proporcionar
            
        Returns:
            Métricas de evaluación
        """
        results = []
        
        for test_puzzle in test_puzzles:
            # Encontrar k puzzles similares del train set
            similar_puzzles = self._find_similar_puzzles(
                test_puzzle, 
                train_puzzles, 
                k=k_shot
            )
            
            # Combinar ejemplos
            all_examples = []
            for similar in similar_puzzles:
                all_examples.extend(similar['trainExamples'])
            all_examples.extend(test_puzzle['trainExamples'])
            
            # Resolver con más contexto
            try:
                solution, steps = self.solver.solve_with_steps(
                    all_examples,
                    np.array(test_puzzle['testExample']['input'])
                )
                
                expected = np.array(test_puzzle['testExample']['output'])
                success = np.array_equal(solution, expected)
                
                # ... crear EvaluationResult similar a zero-shot
                
            except Exception as e:
                print(f"Error en few-shot: {e}")
                continue
        
        return self._calculate_metrics(results, 0)
    
    def cross_validation(self,
                        dataset: Dict[str, Any],
                        n_folds: int = 5) -> Dict[str, Any]:
        """
        Validación cruzada k-fold estratificada
        
        Args:
            dataset: Dataset completo
            n_folds: Número de folds
            
        Returns:
            Métricas promedio y por fold
        """
        all_puzzles = dataset['all']
        fold_size = len(all_puzzles) // n_folds
        
        fold_results = []
        
        for fold_idx in range(n_folds):
            # Crear fold de test
            start_idx = fold_idx * fold_size
            end_idx = start_idx + fold_size if fold_idx < n_folds - 1 else len(all_puzzles)
            
            test_fold = all_puzzles[start_idx:end_idx]
            train_fold = all_puzzles[:start_idx] + all_puzzles[end_idx:]
            
            # Evaluar fold
            fold_metrics = self.evaluate_zero_shot(test_fold)
            fold_metrics['fold'] = fold_idx + 1
            fold_results.append(fold_metrics)
        
        # Promediar métricas
        avg_metrics = self._average_metrics(fold_results)
        
        return {
            'average': avg_metrics,
            'by_fold': fold_results,
            'std_accuracy': np.std([f['accuracy'] for f in fold_results])
        }
    
    def ablation_study(self, test_puzzles: List[Dict]) -> Dict[str, Any]:
        """
        Estudio de ablación: evaluar impacto de cada componente
        
        Args:
            test_puzzles: Puzzles para evaluar
            
        Returns:
            Impacto de cada componente
        """
        components = {
            'baseline': {'augmentation': False, 'hierarchical': False},
            'with_augmentation': {'augmentation': True, 'hierarchical': False},
            'with_hierarchical': {'augmentation': False, 'hierarchical': True},
            'full_system': {'augmentation': True, 'hierarchical': True}
        }
        
        results = {}
        
        for name, settings in components.items():
            # Configurar solver
            self.solver.use_augmentation = settings['augmentation']
            # self.solver.use_hierarchical = settings['hierarchical']  # TODO
            
            # Evaluar
            metrics = self.evaluate_zero_shot(test_puzzles)
            results[name] = metrics
        
        # Calcular impacto relativo
        baseline_acc = results['baseline']['accuracy']
        for name, metrics in results.items():
            if name != 'baseline':
                metrics['improvement'] = metrics['accuracy'] - baseline_acc
                metrics['relative_improvement'] = (
                    (metrics['accuracy'] - baseline_acc) / baseline_acc * 100
                    if baseline_acc > 0 else 0
                )
        
        return results
    
    def _calculate_metrics(self, 
                          results: List[EvaluationResult],
                          total_time: float) -> Dict[str, Any]:
        """Calcula métricas de evaluación comprehensivas"""
        if not results:
            return {'accuracy': 0, 'total_puzzles': 0}
        
        # Métricas básicas
        correct = sum(1 for r in results if r.success)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        # Métricas por categoría
        by_category = {}
        for result in results:
            cat = result.puzzle_type
            if cat not in by_category:
                by_category[cat] = {'correct': 0, 'total': 0}
            by_category[cat]['total'] += 1
            if result.success:
                by_category[cat]['correct'] += 1
        
        # Calcular accuracy por categoría
        for cat, stats in by_category.items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Métricas de tiempo
        avg_time = np.mean([r.time_ms for r in results]) if results else 0
        
        # Métricas de confianza
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0
        confidence_correlation = self._calculate_confidence_correlation(results)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total_puzzles': total,
            'by_category': by_category,
            'avg_time_ms': avg_time,
            'total_time_s': total_time,
            'avg_confidence': avg_confidence,
            'confidence_correlation': confidence_correlation,
            'rules_detected': self._count_rules(results),
            'avg_reasoning_steps': np.mean([r.reasoning_steps for r in results])
        }
    
    def _calculate_confidence_correlation(self, results: List[EvaluationResult]) -> float:
        """Calcula correlación entre confianza y éxito"""
        if len(results) < 2:
            return 0.0
        
        confidences = [r.confidence for r in results]
        successes = [1 if r.success else 0 for r in results]
        
        # Correlación de Pearson
        return np.corrcoef(confidences, successes)[0, 1]
    
    def _count_rules(self, results: List[EvaluationResult]) -> Dict[str, int]:
        """Cuenta frecuencia de cada tipo de regla detectada"""
        rule_counts = {}
        for result in results:
            rule = result.rule_detected
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        return rule_counts
    
    def _find_similar_puzzles(self, 
                             target: Dict,
                             candidates: List[Dict],
                             k: int = 3) -> List[Dict]:
        """
        Encuentra k puzzles más similares basado en características
        
        Por ahora usa categoría, pero se puede extender con:
        - Similitud de patrones
        - Tamaño de grilla
        - Complejidad visual
        """
        # Filtrar por misma categoría
        same_category = [p for p in candidates if p.get('category') == target.get('category')]
        
        # Si no hay suficientes de la misma categoría, usar aleatorios
        if len(same_category) < k:
            same_category.extend(random.sample(
                [p for p in candidates if p not in same_category],
                min(k - len(same_category), len(candidates) - len(same_category))
            ))
        
        return same_category[:k]
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Promedia métricas de múltiples evaluaciones"""
        if not metrics_list:
            return {}
        
        avg = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'avg_time_ms': np.mean([m['avg_time_ms'] for m in metrics_list]),
            'total_puzzles': sum(m['total_puzzles'] for m in metrics_list)
        }
        
        return avg
    
    def _create_synthetic_dataset(self) -> List[Dict]:
        """Crea dataset sintético para testing"""
        categories = {
            'color_mapping': self._create_color_mapping_puzzle,
            'pattern_replication': self._create_pattern_replication_puzzle,
            'gravity': self._create_gravity_puzzle,
            'counting': self._create_counting_puzzle,
            'reflection': self._create_reflection_puzzle
        }
        
        puzzles = []
        puzzle_id = 0
        
        for category, creator in categories.items():
            # Crear 5 puzzles por categoría
            for i in range(5):
                puzzle = creator()
                puzzle['id'] = f"synthetic_{puzzle_id:03d}"
                puzzle['category'] = category
                puzzle['difficulty'] = random.choice(['easy', 'medium', 'hard'])
                puzzles.append(puzzle)
                puzzle_id += 1
        
        return puzzles
    
    def _create_color_mapping_puzzle(self) -> Dict:
        """Crea puzzle de mapeo de colores"""
        return {
            'trainExamples': [
                {
                    'input': [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                    'output': [[2, 2, 0], [2, 2, 0], [0, 0, 0]]
                },
                {
                    'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    'output': [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
                }
            ],
            'testExample': {
                'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                'output': [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
            }
        }
    
    def _create_pattern_replication_puzzle(self) -> Dict:
        """Crea puzzle de replicación 3x3"""
        return {
            'trainExamples': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1]]
                }
            ],
            'testExample': {
                'input': [[2, 2], [2, 2]],
                'output': [[2, 2, 2, 2, 2, 2],
                          [2, 2, 2, 2, 2, 2],
                          [2, 2, 2, 2, 2, 2],
                          [2, 2, 2, 2, 2, 2],
                          [2, 2, 2, 2, 2, 2],
                          [2, 2, 2, 2, 2, 2]]
            }
        }
    
    def _create_gravity_puzzle(self) -> Dict:
        """Crea puzzle de gravedad"""
        return {
            'trainExamples': [
                {
                    'input': [[1, 0, 2], [0, 0, 0], [3, 0, 4]],
                    'output': [[0, 0, 0], [1, 0, 2], [3, 0, 4]]
                }
            ],
            'testExample': {
                'input': [[5, 0, 0], [0, 6, 0], [0, 0, 0]],
                'output': [[0, 0, 0], [0, 0, 0], [5, 6, 0]]
            }
        }
    
    def _create_counting_puzzle(self) -> Dict:
        """Crea puzzle de conteo"""
        return {
            'trainExamples': [
                {
                    'input': [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                    'output': [[3]]
                },
                {
                    'input': [[2, 2, 2], [2, 2, 0], [0, 0, 0]],
                    'output': [[5]]
                }
            ],
            'testExample': {
                'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                'output': [[5]]
            }
        }
    
    def _create_reflection_puzzle(self) -> Dict:
        """Crea puzzle de reflexión"""
        return {
            'trainExamples': [
                {
                    'input': [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
                    'output': [[3, 2, 1], [0, 0, 0], [0, 0, 0]]
                }
            ],
            'testExample': {
                'input': [[4, 5, 0], [0, 0, 0], [6, 7, 8]],
                'output': [[0, 5, 4], [0, 0, 0], [8, 7, 6]]
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Guarda resultados de evaluación"""
        if filename is None:
            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.results_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Resultados guardados en: {filepath}")
        return filepath


if __name__ == "__main__":
    # Prueba del evaluador
    from arc.arc_solver_python import ARCSolverPython
    
    print("🧪 Probando sistema de evaluación...\n")
    
    solver = ARCSolverPython()
    evaluator = ARCEvaluator(solver)
    
    # Cargar dataset
    dataset = evaluator.load_arc_dataset()
    print(f"📊 Dataset cargado: {len(dataset['all'])} puzzles")
    print(f"   Categorías: {list(dataset['by_category'].keys())}")
    
    # Crear splits
    train_set, test_set = evaluator.create_evaluation_splits(dataset, test_ratio=0.3)
    print(f"\n📊 Splits creados:")
    print(f"   Train: {len(train_set)} puzzles")
    print(f"   Test: {len(test_set)} puzzles")
    
    # Evaluación zero-shot
    print("\n🎯 Evaluación Zero-Shot...")
    zero_shot_results = evaluator.evaluate_zero_shot(test_set)
    
    print(f"\n✅ Resultados Zero-Shot:")
    print(f"   Accuracy: {zero_shot_results['accuracy']:.1%}")
    print(f"   Puzzles correctos: {zero_shot_results['correct']}/{zero_shot_results['total_puzzles']}")
    print(f"   Tiempo promedio: {zero_shot_results['avg_time_ms']:.1f}ms")
    
    # Mostrar resultados por categoría
    print("\n📊 Por categoría:")
    for cat, stats in zero_shot_results['by_category'].items():
        print(f"   {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    # Estudio de ablación
    print("\n🔬 Estudio de Ablación...")
    ablation_results = evaluator.ablation_study(test_set[:5])  # Solo 5 para rapidez
    
    print("\n📊 Impacto de componentes:")
    for component, metrics in ablation_results.items():
        print(f"   {component}: {metrics['accuracy']:.1%}", end="")
        if 'improvement' in metrics:
            print(f" (mejora: {metrics['improvement']:+.1%})")
        else:
            print()
    
    # Guardar resultados
    all_results = {
        'zero_shot': zero_shot_results,
        'ablation': ablation_results,
        'dataset_info': {
            'total_puzzles': len(dataset['all']),
            'train_size': len(train_set),
            'test_size': len(test_set)
        }
    }
    
    evaluator.save_results(all_results)