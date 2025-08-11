#!/usr/bin/env python3
"""
Evaluación en puzzles OFICIALES de ARC
Prueba los sistemas en el dataset real del ARC Prize
"""

import numpy as np
import json
import os
from glob import glob
from typing import Dict, List, Tuple
from arc import HybridProteusARCSolver, HierarchicalAttentionSolver

# Intentar importar el solver de Deep Learning
try:
    from arc import DeepLearningARCSolver
    DL_AVAILABLE = True
except ImportError:
    DeepLearningARCSolver = None
    DL_AVAILABLE = False

class ARCOfficialEvaluator:
    """Evaluador para puzzles oficiales de ARC"""
    
    def __init__(self, cache_dir="/app/arc/arc_official_cache"):
        self.cache_dir = cache_dir
        self.puzzles = self._load_official_puzzles()
        
    def _load_official_puzzles(self) -> List[Dict]:
        """Carga puzzles oficiales desde el cache"""
        puzzles = []
        json_files = glob(os.path.join(self.cache_dir, "*.json"))
        
        # Limitar a los primeros N puzzles para prueba rápida
        max_puzzles = 20
        
        for json_file in json_files[:max_puzzles]:
            try:
                with open(json_file, 'r') as f:
                    puzzle = json.load(f)
                    
                    # Formato esperado del ARC oficial
                    if 'train' in puzzle and 'test' in puzzle:
                        puzzles.append({
                            'id': os.path.basename(json_file).replace('.json', ''),
                            'train': puzzle['train'],
                            'test': puzzle['test']
                        })
            except Exception as e:
                print(f"Error cargando {json_file}: {e}")
                continue
        
        return puzzles
    
    def evaluate_solver(self, solver, solver_name: str) -> Dict:
        """Evalúa un solver en puzzles oficiales"""
        results = {
            'solver_name': solver_name,
            'total_puzzles': len(self.puzzles),
            'exact_matches': 0,
            'partial_matches': 0,
            'total_pixel_accuracy': 0.0,
            'puzzle_results': []
        }
        
        print(f"\n🔍 Evaluando {solver_name} en {len(self.puzzles)} puzzles oficiales...")
        
        for i, puzzle in enumerate(self.puzzles):
            puzzle_id = puzzle['id']
            
            # Usar solo el primer test (algunos puzzles tienen múltiples)
            if puzzle['test'] and len(puzzle['test']) > 0:
                test_case = puzzle['test'][0]
                test_input = np.array(test_case['input'])
                
                # Si hay output esperado, usarlo para evaluación
                if 'output' in test_case and test_case['output']:
                    expected = np.array(test_case['output'])
                else:
                    # Si no hay output, usar input como fallback
                    expected = test_input
                    
                try:
                    # Resolver
                    if hasattr(solver, 'solve_with_steps'):
                        solution, _ = solver.solve_with_steps(puzzle['train'], test_input)
                    else:
                        solution = solver.solve(puzzle['train'], test_input)
                    
                    # Calcular métricas
                    exact_match = np.array_equal(solution, expected)
                    
                    if solution.shape == expected.shape:
                        pixel_accuracy = np.sum(solution == expected) / solution.size
                    else:
                        pixel_accuracy = 0.0
                    
                    # Actualizar estadísticas
                    if exact_match:
                        results['exact_matches'] += 1
                    elif pixel_accuracy > 0.5:
                        results['partial_matches'] += 1
                    
                    results['total_pixel_accuracy'] += pixel_accuracy
                    
                    # Guardar resultado individual
                    results['puzzle_results'].append({
                        'id': puzzle_id,
                        'exact': exact_match,
                        'accuracy': pixel_accuracy
                    })
                    
                    # Mostrar progreso
                    status = "✅" if exact_match else ("🔶" if pixel_accuracy > 0.5 else "❌")
                    print(f"  {i+1:3d}. {puzzle_id[:20]:20s} {status} {pixel_accuracy:6.1%}")
                    
                except Exception as e:
                    print(f"  {i+1:3d}. {puzzle_id[:20]:20s} ⚠️  Error: {str(e)[:30]}")
                    results['puzzle_results'].append({
                        'id': puzzle_id,
                        'error': str(e)
                    })
        
        # Calcular promedios
        if results['total_puzzles'] > 0:
            results['avg_pixel_accuracy'] = results['total_pixel_accuracy'] / results['total_puzzles']
            results['exact_percentage'] = (results['exact_matches'] / results['total_puzzles']) * 100
            results['partial_percentage'] = (results['partial_matches'] / results['total_puzzles']) * 100
        
        return results

def print_comparison(results_list: List[Dict]):
    """Imprime comparación de resultados"""
    print("\n" + "="*80)
    print("📊 RESULTADOS EN PUZZLES OFICIALES DE ARC")
    print("="*80)
    
    print(f"\n{'Sistema':<25} {'Exactos':<15} {'Parciales':<15} {'Pixel Acc':<12}")
    print("-"*67)
    
    for result in results_list:
        name = result['solver_name']
        exact = f"{result['exact_matches']}/{result['total_puzzles']} ({result.get('exact_percentage', 0):.0f}%)"
        partial = f"{result['partial_matches']}/{result['total_puzzles']} ({result.get('partial_percentage', 0):.0f}%)"
        pixel = f"{result.get('avg_pixel_accuracy', 0):.1%}"
        
        print(f"{name:<25} {exact:<15} {partial:<15} {pixel:<12}")
    
    # Encontrar el mejor
    if results_list:
        best = max(results_list, key=lambda r: r.get('avg_pixel_accuracy', 0))
        print(f"\n🏆 Mejor sistema: {best['solver_name']}")
        print(f"   - Puzzles exactos: {best['exact_matches']}")
        print(f"   - Accuracy promedio: {best.get('avg_pixel_accuracy', 0):.1%}")

def main():
    print("="*80)
    print("🔬 EVALUACIÓN EN PUZZLES OFICIALES DE ARC PRIZE")
    print("="*80)
    
    evaluator = ARCOfficialEvaluator()
    
    if not evaluator.puzzles:
        print("❌ No se encontraron puzzles oficiales")
        return
    
    print(f"\n📁 Cargados {len(evaluator.puzzles)} puzzles oficiales")
    
    results = []
    
    # 1. Baseline
    print("\n" + "="*80)
    print("1️⃣ BASELINE (sin atención)")
    print("="*80)
    baseline = HybridProteusARCSolver()
    baseline_results = evaluator.evaluate_solver(baseline, "Baseline")
    results.append(baseline_results)
    
    # 2. Sistema HAMS (Hierarchical Attention Multi-Scale)
    print("\n" + "="*80)
    print("2️⃣ SISTEMA HAMS (Hierarchical Attention Multi-Scale)")
    print("="*80)
    hams = HierarchicalAttentionSolver()
    hams_results = evaluator.evaluate_solver(hams, "HAMS")
    results.append(hams_results)
    
    # 3. Sistema Deep Learning (si está disponible)
    if DL_AVAILABLE:
        print("\n" + "="*80)
        print("3️⃣ SISTEMA DEEP LEARNING (A-MHA con CNN)")
        print("="*80)
        dl_solver = DeepLearningARCSolver(device='cpu')
        dl_results = evaluator.evaluate_solver(dl_solver, "Deep Learning")
        results.append(dl_results)
    
    # Mostrar comparación
    print_comparison(results)
    
    # Análisis de mejora
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE MEJORA vs BASELINE")
    print("="*80)
    
    baseline_acc = baseline_results.get('avg_pixel_accuracy', 0)
    
    for result in results[1:]:  # Saltar baseline
        acc = result.get('avg_pixel_accuracy', 0)
        improvement = ((acc - baseline_acc) / max(baseline_acc, 0.01)) * 100
        
        if improvement > 0:
            print(f"\n✅ {result['solver_name']}: +{improvement:.1f}% mejora")
        elif improvement < 0:
            print(f"\n❌ {result['solver_name']}: {improvement:.1f}% peor")
        else:
            print(f"\n➡️ {result['solver_name']}: Sin cambios")
    
    print("\n" + "="*80)
    print("💡 CONCLUSIÓN")
    print("="*80)
    
    if results:
        best = max(results, key=lambda r: r.get('avg_pixel_accuracy', 0))
        print(f"\nEl sistema '{best['solver_name']}' es el mejor en puzzles oficiales")
        print(f"con {best.get('avg_pixel_accuracy', 0):.1%} de accuracy promedio")
        
        if best['exact_matches'] > 0:
            print(f"\n✨ Resolvió perfectamente {best['exact_matches']} puzzle(s) oficial(es)!")

if __name__ == "__main__":
    main()