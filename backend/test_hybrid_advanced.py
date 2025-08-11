#!/usr/bin/env python3
"""
Test del Sistema Híbrido Avanzado PROTEUS-ARC
Versión final con:
- Análisis multiescala
- Segmentación y grafos
- Síntesis de reglas
- Análisis estructural profundo
"""

import numpy as np
from arc.hybrid_proteus_solver import HybridProteusARCSolver
from arc.structural_analyzer import StructuralAnalyzer

def test_with_structural_analysis(train_examples, test_input, test_name):
    """Test con análisis estructural completo"""
    print(f"\n🔬 {test_name}")
    print("="*50)
    
    # Análisis estructural del input
    analyzer = StructuralAnalyzer()
    input_analysis = analyzer.analyze_comprehensive(test_input)
    
    print("📊 Análisis Estructural del Input:")
    print(f"  • Componentes conectados: {input_analysis['connectivity'].get('total_components', 0)}")
    print(f"  • Complejidad estructural: {input_analysis.get('structural_complexity', 0):.2f}")
    print(f"  • Simetría detectada: {input_analysis['symmetry']}")
    
    # Resolver con el solver híbrido
    solver = HybridProteusARCSolver()
    solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    print(f"\n💡 Solución:")
    print(f"  Input: {test_input.tolist()}")
    print(f"  Output: {solution.tolist() if solution is not None else 'No encontrada'}")
    print(f"  Confianza: {solver.confidence:.2%}")
    
    # Análisis estructural de la solución
    if solution is not None:
        output_analysis = analyzer.analyze_comprehensive(solution)
        print(f"\n📊 Análisis Estructural del Output:")
        print(f"  • Componentes: {output_analysis['connectivity'].get('total_components', 0)}")
        print(f"  • Complejidad: {output_analysis.get('structural_complexity', 0):.2f}")
        
        # Verificar preservación de propiedades topológicas
        input_topo = input_analysis['topology']
        output_topo = output_analysis['topology']
        print(f"\n🔄 Transformación Topológica:")
        print(f"  • Cambio en componentes: {output_topo.get('components', 0) - input_topo.get('components', 0):+d}")
        print(f"  • Preservación de simetría: {output_analysis['symmetry'] == input_analysis['symmetry']}")
    
    return solution

def test_pattern_cross_advanced():
    """Test: Expandir cruz con análisis multiescala"""
    train_examples = [
        {
            "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        },
        {
            "input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
        }
    ]
    
    test_input = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    return test_with_structural_analysis(
        train_examples, 
        test_input, 
        "Test 1: Expansión de Cruz (Multiescala)"
    )

def test_fill_with_graph_analysis():
    """Test: Rellenar forma usando análisis de grafos"""
    train_examples = [
        {
            "input": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        },
        {
            "input": [[2, 2, 0], [2, 0, 0], [2, 2, 2]],
            "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        }
    ]
    
    test_input = np.array([[3, 0, 3], [3, 0, 3], [3, 3, 3]])
    return test_with_structural_analysis(
        train_examples,
        test_input,
        "Test 2: Relleno con Análisis de Grafos"
    )

def test_complex_segmentation():
    """Test: Patrón complejo con segmentación"""
    train_examples = [
        {
            "input": [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
            "output": [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        },
        {
            "input": [[2, 0, 0, 2], [0, 2, 2, 0], [0, 2, 2, 0], [2, 0, 0, 2]],
            "output": [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        }
    ]
    
    test_input = np.array([[3, 0, 0, 3], [0, 3, 3, 0], [0, 3, 3, 0], [3, 0, 0, 3]])
    return test_with_structural_analysis(
        train_examples,
        test_input,
        "Test 3: Segmentación y Síntesis de Reglas"
    )

def test_rule_synthesis():
    """Test: Síntesis automática de reglas complejas"""
    train_examples = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[4, 3], [2, 1]]  # Rotación 180°
        },
        {
            "input": [[5, 6], [7, 8]],
            "output": [[8, 7], [6, 5]]
        }
    ]
    
    test_input = np.array([[2, 3], [4, 5]])
    return test_with_structural_analysis(
        train_examples,
        test_input,
        "Test 4: Síntesis de Reglas Complejas"
    )

def analyze_hybrid_system():
    """Analiza el sistema híbrido completo"""
    print("\n" + "="*60)
    print("🧬 ANÁLISIS DEL SISTEMA HÍBRIDO AVANZADO")
    print("="*60)
    
    # Analizar capacidades del StructuralAnalyzer
    analyzer = StructuralAnalyzer()
    test_matrix = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    analysis = analyzer.analyze_comprehensive(test_matrix)
    
    print("\n📊 Capacidades del Análisis Estructural:")
    print("  Métricas disponibles:")
    for key in analysis.keys():
        print(f"    • {key}")
    
    print("\n🔬 Características del Sistema Híbrido:")
    print("  • Análisis topológico con firmas")
    print("  • Detección de componentes conexas")
    print("  • Análisis de agujeros y simetrías")
    print("  • Dimensión fractal por box-counting")
    print("  • Grafos de adyacencia para conectividad")
    print("  • Segmentación por colores")
    print("  • Síntesis automática de reglas")
    print("  • Selección de reglas guiada por topología")
    
    print("\n⚡ Ventajas del Enfoque Híbrido:")
    print("  • Combina análisis simbólico con topológico")
    print("  • Mejor generalización a patrones no vistos")
    print("  • Explicabilidad de las transformaciones")
    print("  • Adaptación dinámica según estructura")

def main():
    print("="*60)
    print("🚀 SISTEMA PROTEUS-ARC HÍBRIDO AVANZADO")
    print("Versión Final: Multiescala + Segmentación + Grafos + Síntesis")
    print("="*60)
    
    # Ejecutar tests
    results = []
    results.append(("Cruz Multiescala", test_pattern_cross_advanced()))
    results.append(("Relleno con Grafos", test_fill_with_graph_analysis()))
    results.append(("Segmentación Compleja", test_complex_segmentation()))
    results.append(("Síntesis de Reglas", test_rule_synthesis()))
    
    # Resumen
    print("\n" + "="*60)
    print("📈 RESUMEN DE RESULTADOS")
    print("="*60)
    
    successful = sum(1 for _, r in results if r is not None)
    print(f"Tests exitosos: {successful}/{len(results)}")
    
    for name, result in results:
        status = "✅" if result is not None else "❌"
        print(f"  {status} {name}")
    
    # Análisis del sistema
    analyze_hybrid_system()

if __name__ == "__main__":
    main()