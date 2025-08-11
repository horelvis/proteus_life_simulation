#!/usr/bin/env python3
"""
Prueba del sistema con atención bidireccional mejorada
Comparación antes/después de aplicar el sistema de atención
"""

import numpy as np
from arc.enhanced_solver_attention import EnhancedSolverWithAttention
from arc.hierarchical_analyzer import HierarchicalAnalyzer
from arc.emergent_rule_system import EmergentRuleSystem
from arc.transformations_fixed import RealTransformations

class AttentionARCSolver(EnhancedSolverWithAttention):
    """
    Solver que usa el sistema de atención bidireccional completo
    """
    
    def __init__(self):
        super().__init__()
        
    def solve_with_steps(self, train_examples, test_input):
        """
        Resuelve usando atención bidireccional
        """
        # Usar el método con atención
        solution, analysis = self.solve_with_attention(train_examples, test_input)
        
        # Crear pasos descriptivos del análisis
        steps = []
        if 'insights' in analysis:
            insights = analysis['insights']
            steps.append({
                'description': f"Detectados {insights.get('num_objects', 0)} objetos, "
                              f"{insights.get('num_relations', 0)} relaciones, "
                              f"{insights.get('num_patterns', 0)} patrones"
            })
            steps.append({
                'description': f"Coherencia bidireccional: {insights.get('coherence_score', 0):.2%}"
            })
            
        return solution, steps

def test_puzzle_with_attention(name, train_examples, test_input, expected):
    """
    Prueba un puzzle usando el sistema de atención
    """
    print(f"\n{'='*60}")
    print(f"🧪 {name}")
    print('='*60)
    
    print(f"\nInput de test:")
    print(test_input)
    
    print(f"\nOutput esperado:")
    print(expected)
    
    # Probar con el solver con atención
    solver = AttentionARCSolver()
    solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    print(f"\nSolución con atención:")
    print(solution)
    
    # Mostrar pasos del análisis
    print("\n📊 Análisis de atención:")
    for step in steps:
        print(f"  • {step['description']}")
    
    # Verificación
    correct = np.array_equal(solution, expected)
    
    if correct:
        print("\n✅ CORRECTO - La solución es exactamente la esperada")
    else:
        print("\n❌ INCORRECTO - La solución NO coincide con lo esperado")
        
        if solution.shape == expected.shape:
            diff = (solution != expected)
            num_diff = np.sum(diff)
            accuracy = 1.0 - (num_diff / solution.size)
            print(f"   Accuracy: {accuracy:.1%} ({solution.size - num_diff}/{solution.size} píxeles correctos)")
        else:
            print(f"   Dimensiones diferentes: {solution.shape} vs {expected.shape}")
    
    return correct

def main():
    print("="*70)
    print("🔬 PRUEBA CON SISTEMA DE ATENCIÓN BIDIRECCIONAL")
    print("Comparando resultados con el sistema mejorado")
    print("="*70)
    
    results = []
    
    # Puzzle 1: Expansión de cruz
    result1 = test_puzzle_with_attention(
        "Expansión de Cruz",
        train_examples=[
            {"input": [[0,0,0],[0,1,0],[0,0,0]], 
             "output": [[0,1,0],[1,1,1],[0,1,0]]},
            {"input": [[0,0,0],[0,2,0],[0,0,0]], 
             "output": [[0,2,0],[2,2,2],[0,2,0]]}
        ],
        test_input=np.array([[0,0,0],[0,3,0],[0,0,0]]),
        expected=np.array([[0,3,0],[3,3,3],[0,3,0]])
    )
    results.append(("Cruz", result1))
    
    # Puzzle 2: Relleno
    result2 = test_puzzle_with_attention(
        "Relleno de Forma",
        train_examples=[
            {"input": [[1,1,1],[1,0,1],[1,1,1]], 
             "output": [[1,1,1],[1,1,1],[1,1,1]]},
            {"input": [[2,2,2,2],[2,0,0,2],[2,0,0,2],[2,2,2,2]], 
             "output": [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]}
        ],
        test_input=np.array([[3,3,3],[3,0,3],[3,3,3]]),
        expected=np.array([[3,3,3],[3,3,3],[3,3,3]])
    )
    results.append(("Relleno", result2))
    
    # Puzzle 3: Mapeo de colores
    result3 = test_puzzle_with_attention(
        "Mapeo de Colores",
        train_examples=[
            {"input": [[1,2],[3,4]], 
             "output": [[2,3],[4,5]]},
            {"input": [[2,3],[4,5]], 
             "output": [[3,4],[5,6]]}
        ],
        test_input=np.array([[3,4],[5,6]]),
        expected=np.array([[4,5],[6,7]])
    )
    results.append(("Mapeo", result3))
    
    # Puzzle 4: Rotación
    result4 = test_puzzle_with_attention(
        "Rotación 90°",
        train_examples=[
            {"input": [[1,0],[0,0]], 
             "output": [[0,1],[0,0]]},
            {"input": [[2,0],[0,0]], 
             "output": [[0,2],[0,0]]}
        ],
        test_input=np.array([[3,0],[0,0]]),
        expected=np.array([[0,3],[0,0]])
    )
    results.append(("Rotación", result4))
    
    # Puzzle 5: Simetría (nuevo)
    result5 = test_puzzle_with_attention(
        "Detección de Simetría",
        train_examples=[
            {"input": [[1,0,1],[0,2,0],[1,0,1]], 
             "output": [[1,1,1],[1,2,1],[1,1,1]]},
            {"input": [[2,0,2],[0,3,0],[2,0,2]], 
             "output": [[2,2,2],[2,3,2],[2,2,2]]}
        ],
        test_input=np.array([[3,0,3],[0,4,0],[3,0,3]]),
        expected=np.array([[3,3,3],[3,4,3],[3,3,3]])
    )
    results.append(("Simetría", result5))
    
    print("\n" + "="*70)
    print("📊 RESUMEN CON SISTEMA DE ATENCIÓN")
    print("="*70)
    
    correct_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n✅ Puzzles resueltos correctamente: {correct_count}/{total_count} ({100*correct_count/total_count:.0f}%)")
    
    print("\n📋 Detalle:")
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  • {name}: {status}")
    
    print("\n🎯 COMPARACIÓN CON BASELINE:")
    print("  • Baseline (sin atención): 2/4 (50%)")
    print(f"  • Con atención: {correct_count}/{total_count} ({100*correct_count/total_count:.0f}%)")
    
    improvement = (correct_count/total_count - 0.5) * 100
    if improvement > 0:
        print(f"\n📈 MEJORA: +{improvement:.0f}% con el sistema de atención")
    elif improvement == 0:
        print("\n➡️ Sin cambios respecto al baseline")
    else:
        print(f"\n📉 Empeoramiento: {improvement:.0f}%")

if __name__ == "__main__":
    main()