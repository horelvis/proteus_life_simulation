#!/usr/bin/env python3
"""
Script de comparación con métrica unificada
Compara: Baseline vs Atención Actual vs Atención Integrada
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from arc import HybridProteusARCSolver, HierarchicalAttentionSolver

# Intentar importar el solver de Deep Learning
try:
    from arc import DeepLearningARCSolver
    DL_AVAILABLE = True
except ImportError:
    DeepLearningARCSolver = None
    DL_AVAILABLE = False

class MetricEvaluator:
    """Evaluador con métrica unificada para comparación justa"""
    
    def __init__(self):
        self.test_puzzles = self._create_test_suite()
        
    def _create_test_suite(self):
        """Suite de pruebas estándar"""
        return [
            {
                "name": "cross_expansion",
                "train": [
                    {"input": [[0,0,0],[0,1,0],[0,0,0]], 
                     "output": [[0,1,0],[1,1,1],[0,1,0]]},
                    {"input": [[0,0,0],[0,2,0],[0,0,0]], 
                     "output": [[0,2,0],[2,2,2],[0,2,0]]}
                ],
                "test": {"input": [[0,0,0],[0,3,0],[0,0,0]], 
                         "output": [[0,3,0],[3,3,3],[0,3,0]]}
            },
            {
                "name": "fill_shape",
                "train": [
                    {"input": [[1,1,1],[1,0,1],[1,1,1]], 
                     "output": [[1,1,1],[1,1,1],[1,1,1]]},
                    {"input": [[2,2,2,2],[2,0,0,2],[2,0,0,2],[2,2,2,2]], 
                     "output": [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]}
                ],
                "test": {"input": [[3,3,3],[3,0,3],[3,3,3]], 
                         "output": [[3,3,3],[3,3,3],[3,3,3]]}
            },
            {
                "name": "color_mapping",
                "train": [
                    {"input": [[1,2],[3,4]], 
                     "output": [[2,3],[4,5]]},
                    {"input": [[2,3],[4,5]], 
                     "output": [[3,4],[5,6]]}
                ],
                "test": {"input": [[3,4],[5,6]], 
                         "output": [[4,5],[6,7]]}
            },
            {
                "name": "rotation_90",
                "train": [
                    {"input": [[1,0],[0,0]], 
                     "output": [[0,1],[0,0]]},
                    {"input": [[2,0],[0,0]], 
                     "output": [[0,2],[0,0]]}
                ],
                "test": {"input": [[3,0],[0,0]], 
                         "output": [[0,3],[0,0]]}
            },
            {
                "name": "diagonal_fill",
                "train": [
                    {"input": [[1,0,0],[0,1,0],[0,0,1]], 
                     "output": [[1,1,1],[1,1,1],[1,1,1]]},
                    {"input": [[2,0,0],[0,2,0],[0,0,2]], 
                     "output": [[2,2,2],[2,2,2],[2,2,2]]}
                ],
                "test": {"input": [[3,0,0],[0,3,0],[0,0,3]], 
                         "output": [[3,3,3],[3,3,3],[3,3,3]]}
            },
            {
                "name": "gravity",
                "train": [
                    {"input": [[1,0,2],[0,0,0],[3,0,4]], 
                     "output": [[0,0,0],[1,0,2],[3,0,4]]},
                    {"input": [[5,0,0],[0,6,0],[0,0,0]], 
                     "output": [[0,0,0],[0,0,0],[5,6,0]]}
                ],
                "test": {"input": [[7,0,0],[0,8,0],[0,0,0]], 
                         "output": [[0,0,0],[0,0,0],[7,8,0]]}
            }
        ]
    
    def calculate_metrics(self, prediction: np.ndarray, expected: np.ndarray) -> Dict:
        """Calcula métricas detalladas"""
        
        # Exactitud completa
        exact_match = np.array_equal(prediction, expected)
        
        # Accuracy por píxeles
        if prediction.shape == expected.shape:
            correct_pixels = np.sum(prediction == expected)
            total_pixels = prediction.size
            pixel_accuracy = correct_pixels / total_pixels
        else:
            pixel_accuracy = 0.0
            correct_pixels = 0
            total_pixels = max(prediction.size, expected.size)
        
        # Accuracy por color (ignorando posición)
        pred_colors = np.unique(prediction)
        exp_colors = np.unique(expected)
        color_overlap = len(set(pred_colors) & set(exp_colors)) / max(len(pred_colors), len(exp_colors))
        
        # Accuracy estructural (formas preservadas)
        structural_score = self._calculate_structural_similarity(prediction, expected)
        
        return {
            'exact_match': exact_match,
            'pixel_accuracy': pixel_accuracy,
            'correct_pixels': correct_pixels,
            'total_pixels': total_pixels,
            'color_accuracy': color_overlap,
            'structural_score': structural_score,
            'composite_score': (pixel_accuracy * 0.5 + color_overlap * 0.2 + structural_score * 0.3)
        }
    
    def _calculate_structural_similarity(self, pred: np.ndarray, exp: np.ndarray) -> float:
        """Calcula similitud estructural"""
        if pred.shape != exp.shape:
            return 0.0
        
        # Comparar patrones binarios (0 vs no-0)
        pred_binary = (pred != 0).astype(int)
        exp_binary = (exp != 0).astype(int)
        
        return np.sum(pred_binary == exp_binary) / pred_binary.size
    
    def evaluate_solver(self, solver, solver_name: str) -> Dict:
        """Evalúa un solver en toda la suite de pruebas"""
        results = {
            'solver_name': solver_name,
            'puzzles': [],
            'total_exact': 0,
            'total_pixel_accuracy': 0.0,
            'total_composite_score': 0.0
        }
        
        for puzzle in self.test_puzzles:
            test_input = np.array(puzzle['test']['input'])
            expected = np.array(puzzle['test']['output'])
            
            try:
                # Resolver
                if hasattr(solver, 'solve_with_steps'):
                    solution, _ = solver.solve_with_steps(puzzle['train'], test_input)
                else:
                    solution = solver.solve(puzzle['train'], test_input)
                
                # Calcular métricas
                metrics = self.calculate_metrics(solution, expected)
                
                puzzle_result = {
                    'name': puzzle['name'],
                    'metrics': metrics,
                    'solution_shape': solution.shape,
                    'expected_shape': expected.shape
                }
                
                results['puzzles'].append(puzzle_result)
                
                if metrics['exact_match']:
                    results['total_exact'] += 1
                results['total_pixel_accuracy'] += metrics['pixel_accuracy']
                results['total_composite_score'] += metrics['composite_score']
                
            except Exception as e:
                print(f"Error en {solver_name} con {puzzle['name']}: {e}")
                results['puzzles'].append({
                    'name': puzzle['name'],
                    'error': str(e)
                })
        
        # Promediar métricas
        num_puzzles = len(self.test_puzzles)
        results['avg_pixel_accuracy'] = results['total_pixel_accuracy'] / num_puzzles
        results['avg_composite_score'] = results['total_composite_score'] / num_puzzles
        results['exact_percentage'] = (results['total_exact'] / num_puzzles) * 100
        
        return results

def print_comparison_table(results_list: List[Dict]):
    """Imprime tabla comparativa de resultados"""
    print("\n" + "="*80)
    print("📊 TABLA COMPARATIVA DE SISTEMAS")
    print("="*80)
    
    # Encabezados
    print(f"\n{'Sistema':<30} {'Exactos':<12} {'Pixel Acc':<12} {'Score':<12}")
    print("-"*66)
    
    for result in results_list:
        name = result['solver_name']
        exact = f"{result['total_exact']}/{len(result['puzzles'])} ({result['exact_percentage']:.0f}%)"
        pixel = f"{result['avg_pixel_accuracy']:.1%}"
        score = f"{result['avg_composite_score']:.1%}"
        
        print(f"{name:<30} {exact:<12} {pixel:<12} {score:<12}")
    
    # Detalles por puzzle
    print("\n" + "="*80)
    print("📋 DETALLES POR PUZZLE")
    print("="*80)
    
    puzzle_names = [p['name'] for p in results_list[0]['puzzles']]
    
    print(f"\n{'Puzzle':<20}", end="")
    for result in results_list:
        print(f" {result['solver_name'][:15]:<15}", end="")
    print()
    print("-"*(20 + 15*len(results_list)))
    
    for puzzle_name in puzzle_names:
        print(f"{puzzle_name:<20}", end="")
        for result in results_list:
            puzzle_data = next((p for p in result['puzzles'] if p['name'] == puzzle_name), None)
            if puzzle_data and 'metrics' in puzzle_data:
                if puzzle_data['metrics']['exact_match']:
                    print(f" {'✅ 100%':<15}", end="")
                else:
                    acc = puzzle_data['metrics']['pixel_accuracy']
                    print(f" {f'❌ {acc:.0%}':<15}", end="")
            else:
                print(f" {'ERROR':<15}", end="")
        print()

def main():
    print("="*80)
    print("🔬 COMPARACIÓN DE SISTEMAS DE ATENCIÓN PARA ARC")
    print("="*80)
    
    evaluator = MetricEvaluator()
    results = []
    
    # 1. Evaluar Baseline (sin atención)
    print("\n1️⃣ Evaluando sistema BASELINE (sin atención)...")
    baseline_solver = HybridProteusARCSolver()
    baseline_results = evaluator.evaluate_solver(baseline_solver, "Baseline (sin atención)")
    results.append(baseline_results)
    
    # 2. Evaluar Sistema HAMS (Hierarchical Attention Multi-Scale)
    print("\n2️⃣ Evaluando sistema HAMS (Hierarchical Attention Multi-Scale)...")
    hams_solver = HierarchicalAttentionSolver()
    hams_results = evaluator.evaluate_solver(hams_solver, "HAMS")
    results.append(hams_results)
    
    # 3. Evaluar Sistema Deep Learning (si está disponible)
    if DL_AVAILABLE:
        print("\n3️⃣ Evaluando sistema DEEP LEARNING (A-MHA con CNN)...")
        dl_solver = DeepLearningARCSolver(device='cpu')
        dl_results = evaluator.evaluate_solver(dl_solver, "Deep Learning A-MHA")
        results.append(dl_results)
    else:
        print("\n⚠️ PyTorch no disponible, saltando evaluación de Deep Learning")
    
    # Mostrar comparación
    print_comparison_table(results)
    
    # Análisis de mejora
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE MEJORA")
    print("="*80)
    
    baseline_score = baseline_results['avg_composite_score']
    
    for result in results[1:]:
        improvement = ((result['avg_composite_score'] - baseline_score) / baseline_score) * 100
        if improvement > 0:
            print(f"\n✅ {result['solver_name']}: +{improvement:.1f}% mejora sobre baseline")
        elif improvement < 0:
            print(f"\n❌ {result['solver_name']}: {improvement:.1f}% peor que baseline")
        else:
            print(f"\n➡️ {result['solver_name']}: Sin cambios respecto a baseline")
    
    # Guardar resultados (convertir numpy int64 a int nativo)
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convierte tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_json = convert_numpy_types(results)
    with open('comparison_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("\n💾 Resultados guardados en comparison_results.json")
    
    # Recomendación final
    print("\n" + "="*80)
    print("🎯 RECOMENDACIÓN")
    print("="*80)
    
    best_system = max(results, key=lambda x: x['avg_composite_score'])
    print(f"\n🏆 El mejor sistema es: {best_system['solver_name']}")
    print(f"   Score compuesto: {best_system['avg_composite_score']:.1%}")
    print(f"   Puzzles exactos: {best_system['total_exact']}/{len(best_system['puzzles'])}")
    
    if best_system['solver_name'] != "Baseline (sin atención)":
        print("\n💡 El sistema de atención SÍ mejora el rendimiento")
        print("   Recomendación: Usar el sistema HAMS para mejores resultados")
    else:
        print("\n⚠️ El baseline sigue siendo mejor")
        print("   Recomendación: Revisar la implementación de atención")

if __name__ == "__main__":
    main()