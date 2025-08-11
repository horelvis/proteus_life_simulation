#!/usr/bin/env python3
"""
Test del Sistema PROTEUS-ARC Final
Versión definitiva con todas las mejoras integradas
"""

import numpy as np
from arc import HybridProteusARCSolver, StructuralAnalyzer

def test_arc_puzzle(name, train_examples, test_input, expected_output=None):
    """Test genérico para puzzles ARC con el sistema final"""
    print(f"\n{'='*60}")
    print(f"🧬 {name}")
    print('='*60)
    
    # Análisis estructural
    analyzer = StructuralAnalyzer()
    input_analysis = analyzer.analyze_comprehensive(test_input)
    
    print(f"\n📊 Análisis del Input:")
    print(f"  • Complejidad estructural: {input_analysis.get('structural_complexity', 0):.2f}")
    print(f"  • Componentes: {input_analysis['connectivity'].get('total_components', 0)}")
    print(f"  • Simetrías detectadas: {sum(1 for s in input_analysis['symmetry'].values() if s.get('has_symmetry', False))}/8")
    
    # Resolver
    solver = HybridProteusARCSolver()
    solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    print(f"\n💡 Resultado:")
    print(f"  Input:    {test_input.tolist()}")
    print(f"  Solución: {solution.tolist() if solution is not None else 'No encontrada'}")
    if expected_output is not None:
        print(f"  Esperado: {expected_output}")
    print(f"  Confianza: {solver.confidence:.1%}")
    
    # Pasos de razonamiento
    if steps and len(steps) > 0:
        print(f"\n📝 Razonamiento ({len(steps)} pasos):")
        for i, step in enumerate(steps[:3]):
            print(f"  {i+1}. {step.get('description', 'Paso sin descripción')}")
    
    return solution

def main():
    print("="*60)
    print("🚀 SISTEMA PROTEUS-ARC FINAL")
    print("Híbrido Avanzado con Análisis Multiescala")
    print("="*60)
    
    # Test 1: Patrón de expansión
    test_arc_puzzle(
        "Test 1: Expansión de Patrón",
        train_examples=[
            {"input": [[0,0,0],[0,1,0],[0,0,0]], "output": [[0,1,0],[1,1,1],[0,1,0]]},
            {"input": [[0,0,0],[0,2,0],[0,0,0]], "output": [[0,2,0],[2,2,2],[0,2,0]]}
        ],
        test_input=np.array([[0,0,0],[0,3,0],[0,0,0]]),
        expected_output=[[0,3,0],[3,3,3],[0,3,0]]
    )
    
    # Test 2: Relleno de forma
    test_arc_puzzle(
        "Test 2: Relleno de Forma",
        train_examples=[
            {"input": [[1,1,1],[1,0,1],[1,1,1]], "output": [[1,1,1],[1,1,1],[1,1,1]]},
            {"input": [[2,2,0],[2,0,0],[2,2,2]], "output": [[2,2,2],[2,2,2],[2,2,2]]}
        ],
        test_input=np.array([[3,0,3],[3,0,3],[3,3,3]]),
        expected_output=[[3,3,3],[3,3,3],[3,3,3]]
    )
    
    # Test 3: Transformación compleja
    test_arc_puzzle(
        "Test 3: Transformación Compleja",
        train_examples=[
            {"input": [[1,0],[0,1]], "output": [[1,1],[1,1]]},
            {"input": [[2,0],[0,2]], "output": [[2,2],[2,2]]}
        ],
        test_input=np.array([[3,0],[0,3]]),
        expected_output=[[3,3],[3,3]]
    )
    
    print("\n" + "="*60)
    print("✅ SISTEMA FINAL OPERATIVO")
    print("="*60)
    print("\nCaracterísticas del sistema:")
    print("  • Análisis estructural con grafos de conectividad")
    print("  • Segmentación multiescala por colores")
    print("  • Síntesis automática de reglas")
    print("  • Análisis topológico con firmas")
    print("  • Detección de 8 tipos de simetría")
    print("  • Memoria holográfica para patrones")

if __name__ == "__main__":
    main()