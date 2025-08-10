#!/usr/bin/env python3
"""
Compara el solver basado en reglas vs el solver PROTEUS topológico
"""

import numpy as np
import time
from typing import Dict, List, Any

from arc_solver_python import ARCSolverPython
from proteus_arc_solver import ProteusARCSolver, ProteusAdapter
from arc_official_loader import ARCOfficialLoader

def print_grid(grid, title=""):
    """Imprime una grilla en formato texto"""
    if title:
        print(f"\n{title}:")
    
    char_map = {
        0: '.',  1: '1',  2: '2',  3: '3',  4: '4',
        5: '5',  6: '6',  7: '7',  8: '8',  9: '9'
    }
    
    for row in grid:
        print('   ', ' '.join(char_map.get(cell, str(cell)) for cell in row))

def evaluate_solver(solver_name: str, solver, puzzles: List[Dict]) -> Dict[str, Any]:
    """Evalúa un solver con puzzles oficiales"""
    results = {
        'solver_name': solver_name,
        'total': len(puzzles),
        'correct': 0,
        'partially_correct': 0,
        'failed': 0,
        'accuracies': [],
        'times': [],
        'by_puzzle': {}
    }
    
    print(f"\n{'='*60}")
    print(f"🔬 Evaluando {solver_name}")
    print(f"{'='*60}")
    
    for puzzle in puzzles:
        print(f"\n📊 Puzzle {puzzle['id']}...")
        start_time = time.time()
        
        try:
            # Resolver según el tipo de solver
            test_input = np.array(puzzle['testExample']['input'])
            
            if hasattr(solver, 'solve_with_steps'):
                solution, steps = solver.solve_with_steps(
                    puzzle['trainExamples'], 
                    test_input
                )
            else:
                # Para PROTEUS solver
                solution = solver.solve(
                    puzzle['trainExamples'], 
                    test_input
                )
            
            # Evaluar resultado
            expected = np.array(puzzle['testExample']['output'])
            
            # Manejar diferencias de tamaño
            if solution.shape != expected.shape:
                accuracy = 0.0
                print(f"   ❌ Error de tamaño: {solution.shape} vs {expected.shape}")
            else:
                accuracy = np.sum(solution == expected) / expected.size
                
                if accuracy == 1.0:
                    results['correct'] += 1
                    print(f"   ✅ Correcto!")
                elif accuracy > 0.5:
                    results['partially_correct'] += 1
                    print(f"   ⚠️  Parcialmente correcto: {accuracy*100:.1f}%")
                else:
                    results['failed'] += 1
                    print(f"   ❌ Fallo: {accuracy*100:.1f}%")
            
            results['accuracies'].append(accuracy)
            results['by_puzzle'][puzzle['id']] = {
                'accuracy': accuracy,
                'time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"   💥 Error: {e}")
            results['failed'] += 1
            results['accuracies'].append(0.0)
            results['by_puzzle'][puzzle['id']] = {
                'accuracy': 0.0,
                'time': time.time() - start_time,
                'error': str(e)
            }
        
        results['times'].append(time.time() - start_time)
    
    # Calcular métricas agregadas
    results['avg_accuracy'] = np.mean(results['accuracies'])
    results['avg_time'] = np.mean(results['times'])
    
    return results

def compare_solvers():
    """Compara ambos solvers con puzzles oficiales"""
    print("🧬 COMPARACIÓN: Solver de Reglas vs PROTEUS Topológico")
    print("="*60)
    
    # Cargar puzzles oficiales
    loader = ARCOfficialLoader()
    puzzle_ids = ['00d62c1b', '0a938d79', '0ca9ddb6', '0d3d703e', '0e206a2e']
    puzzles = loader.load_specific_puzzles(puzzle_ids)
    
    if not puzzles:
        print("❌ No se pudieron cargar puzzles")
        return
    
    print(f"\n📋 Evaluando con {len(puzzles)} puzzles oficiales")
    
    # Crear solvers
    rule_solver = ARCSolverPython()
    rule_solver.use_augmentation = False  # Para comparación justa
    
    proteus_solver = ProteusARCSolver(population_size=30)
    proteus_adapter = ProteusAdapter()
    
    # Evaluar solver de reglas
    rule_results = evaluate_solver("Solver de Reglas Fijas", rule_solver, puzzles)
    
    # Evaluar solver PROTEUS
    proteus_results = evaluate_solver("Solver PROTEUS Topológico", proteus_solver, puzzles)
    
    # Comparar resultados
    print(f"\n{'='*60}")
    print("📊 COMPARACIÓN DE RESULTADOS")
    print(f"{'='*60}")
    
    print(f"\n{'Métrica':<30} {'Reglas':<15} {'PROTEUS':<15} {'Diferencia':<15}")
    print("-"*75)
    
    # Comparar métricas
    metrics = [
        ('Puzzles correctos', rule_results['correct'], proteus_results['correct']),
        ('Parcialmente correctos', rule_results['partially_correct'], proteus_results['partially_correct']),
        ('Fallidos', rule_results['failed'], proteus_results['failed']),
        ('Accuracy promedio', f"{rule_results['avg_accuracy']*100:.1f}%", f"{proteus_results['avg_accuracy']*100:.1f}%"),
        ('Tiempo promedio', f"{rule_results['avg_time']*1000:.1f}ms", f"{proteus_results['avg_time']*1000:.1f}ms")
    ]
    
    for metric, rule_val, proteus_val in metrics:
        if isinstance(rule_val, int):
            diff = proteus_val - rule_val
            diff_str = f"+{diff}" if diff > 0 else str(diff)
        else:
            diff_str = "-"
        
        print(f"{metric:<30} {str(rule_val):<15} {str(proteus_val):<15} {diff_str:<15}")
    
    # Comparación puzzle por puzzle
    print(f"\n📊 Accuracy por puzzle:")
    print(f"{'Puzzle ID':<15} {'Reglas':<15} {'PROTEUS':<15} {'Mejor':<15}")
    print("-"*60)
    
    for puzzle_id in rule_results['by_puzzle']:
        rule_acc = rule_results['by_puzzle'][puzzle_id]['accuracy']
        proteus_acc = proteus_results['by_puzzle'].get(puzzle_id, {}).get('accuracy', 0)
        
        if rule_acc > proteus_acc:
            mejor = "Reglas"
        elif proteus_acc > rule_acc:
            mejor = "PROTEUS"
        else:
            mejor = "Empate"
        
        print(f"{puzzle_id:<15} {rule_acc*100:>6.1f}%{'':<8} {proteus_acc*100:>6.1f}%{'':<8} {mejor:<15}")
    
    # Análisis cualitativo
    print(f"\n💡 Análisis cualitativo:")
    
    if proteus_results['avg_accuracy'] > rule_results['avg_accuracy']:
        print("   ✅ PROTEUS muestra mejor accuracy promedio")
        print("   → La evolución topológica podría capturar patrones más complejos")
    else:
        print("   ⚠️  El solver de reglas muestra mejor accuracy")
        print("   → Las reglas fijas son más efectivas para estos puzzles")
    
    if proteus_results['avg_time'] > rule_results['avg_time'] * 10:
        print("   ⏰ PROTEUS es significativamente más lento")
        print("   → El costo computacional de la evolución es alto")
    
    # Test específico con un puzzle simple
    print(f"\n🧪 Test detallado con puzzle de mapeo de color:")
    test_color_mapping_example(rule_solver, proteus_solver)

def test_color_mapping_example(rule_solver, proteus_solver):
    """Prueba específica con un ejemplo simple de mapeo de color"""
    # Ejemplo simple: 1→2, 2→3, 3→1
    train_examples = [
        {
            'input': [[1, 2, 3], [3, 1, 2], [2, 3, 1]],
            'output': [[2, 3, 1], [1, 2, 3], [3, 1, 2]]
        }
    ]
    test_input = np.array([[1, 1, 2], [2, 3, 3], [3, 1, 2]])
    expected = np.array([[2, 2, 3], [3, 1, 1], [1, 2, 3]])
    
    print("\n   Input de test:")
    print_grid(test_input, "")
    
    print("\n   Output esperado:")
    print_grid(expected, "")
    
    # Probar con reglas
    print("\n   Solver de reglas:")
    solution_rules, _ = rule_solver.solve_with_steps(train_examples, test_input)
    print_grid(solution_rules, "")
    accuracy_rules = np.sum(solution_rules == expected) / expected.size
    print(f"   Accuracy: {accuracy_rules*100:.1f}%")
    
    # Probar con PROTEUS
    print("\n   Solver PROTEUS:")
    solution_proteus = proteus_solver.solve(train_examples, test_input)
    print_grid(solution_proteus, "")
    accuracy_proteus = np.sum(solution_proteus == expected) / expected.size
    print(f"   Accuracy: {accuracy_proteus*100:.1f}%")
    
    # Análisis
    if accuracy_rules > accuracy_proteus:
        print("\n   → Las reglas fijas funcionan mejor para transformaciones simples")
    elif accuracy_proteus > accuracy_rules:
        print("\n   → PROTEUS encuentra la transformación sin reglas predefinidas!")
    else:
        print("\n   → Ambos métodos tienen rendimiento similar")

if __name__ == "__main__":
    compare_solvers()