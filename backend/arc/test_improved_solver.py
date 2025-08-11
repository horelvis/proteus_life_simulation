#!/usr/bin/env python3
"""
Script de prueba para el solver mejorado
Verifica que las mejoras críticas funcionan correctamente
"""

import numpy as np
import logging
import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.arc_swarm_solver_improved import ARCSwarmSolverImproved

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_color_mapping():
    """Prueba con transformación simple de mapeo de colores"""
    print("\n" + "="*60)
    print("🎨 TEST 1: Mapeo de Colores")
    print("="*60)
    
    # Ejemplos de entrenamiento: cada color se incrementa en 1
    train_examples = [
        {
            'input': [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
            'output': [[2, 3, 4], [3, 4, 2], [4, 2, 3]]
        },
        {
            'input': [[1, 1, 2], [2, 3, 3], [3, 2, 1]],
            'output': [[2, 2, 3], [3, 4, 4], [4, 3, 2]]
        }
    ]
    
    test_input = np.array([[2, 1, 3], [3, 2, 1], [1, 3, 2]])
    expected_output = np.array([[3, 2, 4], [4, 3, 2], [2, 4, 3]])
    
    solver = ARCSwarmSolverImproved(
        population_size=20,
        generations=3,
        mutation_rate=0.3
    )
    
    solution, report = solver.solve_with_swarm(train_examples, test_input)
    
    print(f"\nInput de prueba:\n{test_input}")
    print(f"\nSolución esperada:\n{expected_output}")
    print(f"\nSolución obtenida:\n{solution}")
    
    if solution is not None and np.array_equal(solution, expected_output):
        print("✅ TEST PASADO: La solución es correcta")
    else:
        print("❌ TEST FALLADO: La solución no coincide")
    
    print(f"\nFitness final: {report.get('fitness', 0):.3f}")
    print(f"Agentes vivos: {report.get('alive_agents', 0)}")
    print(f"Reglas en memoria compartida: {report.get('shared_memory_rules', 0)}")
    
    return solution is not None and np.array_equal(solution, expected_output)

def test_rotation():
    """Prueba con transformación de rotación"""
    print("\n" + "="*60)
    print("🔄 TEST 2: Rotación 90°")
    print("="*60)
    
    # Rotación de 90 grados en sentido horario
    train_examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[3, 1], [4, 2]]
        },
        {
            'input': [[5, 6, 7], [8, 9, 10], [11, 12, 13]],
            'output': [[11, 8, 5], [12, 9, 6], [13, 10, 7]]
        }
    ]
    
    test_input = np.array([[1, 0], [2, 3]])
    expected_output = np.array([[2, 1], [3, 0]])
    
    solver = ARCSwarmSolverImproved(
        population_size=15,
        generations=3
    )
    
    solution, report = solver.solve_with_swarm(train_examples, test_input)
    
    print(f"\nInput de prueba:\n{test_input}")
    print(f"\nSolución esperada:\n{expected_output}")
    print(f"\nSolución obtenida:\n{solution}")
    
    if solution is not None and np.array_equal(solution, expected_output):
        print("✅ TEST PASADO: La rotación es correcta")
    else:
        print("❌ TEST FALLADO: La rotación no coincide")
    
    return solution is not None and np.array_equal(solution, expected_output)

def test_pattern_extraction():
    """Prueba extracción de patrones (bounding box)"""
    print("\n" + "="*60)
    print("📦 TEST 3: Extracción de Patrón")
    print("="*60)
    
    # Extraer el rectángulo mínimo que contiene todos los elementos no-cero
    train_examples = [
        {
            'input': [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            'output': [[1, 2], [3, 4]]
        },
        {
            'input': [
                [0, 0, 0],
                [0, 5, 0],
                [0, 0, 0]
            ],
            'output': [[5]]
        }
    ]
    
    test_input = np.array([
        [0, 0, 0, 0],
        [0, 7, 8, 0],
        [0, 9, 10, 0],
        [0, 11, 12, 0],
        [0, 0, 0, 0]
    ])
    expected_output = np.array([[7, 8], [9, 10], [11, 12]])
    
    solver = ARCSwarmSolverImproved(
        population_size=15,
        generations=3
    )
    
    solution, report = solver.solve_with_swarm(train_examples, test_input)
    
    print(f"\nInput de prueba:\n{test_input}")
    print(f"\nSolución esperada:\n{expected_output}")
    print(f"\nSolución obtenida:\n{solution}")
    
    if solution is not None and np.array_equal(solution, expected_output):
        print("✅ TEST PASADO: La extracción es correcta")
    else:
        print("❌ TEST FALLADO: La extracción no coincide")
    
    # Mostrar estadísticas de especialización
    if 'specialization_stats' in report:
        print("\n📊 Estadísticas de especialización:")
        for spec, stats in report['specialization_stats'].items():
            print(f"  {spec}: {stats['count']} agentes, fitness promedio: {stats['avg_fitness']:.3f}")
    
    return solution is not None and np.array_equal(solution, expected_output)

def test_chain_transformations():
    """Prueba cadenas de transformaciones (color mapping + rotación)"""
    print("\n" + "="*60)
    print("⛓️ TEST 4: Cadena de Transformaciones")
    print("="*60)
    
    # Primero mapeo de color (+1), luego rotación 180°
    train_examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[5, 4], [3, 2]]  # +1 y luego rotar 180°
        },
        {
            'input': [[0, 1], [2, 0]],
            'output': [[1, 3], [2, 1]]  # +1 y luego rotar 180°
        }
    ]
    
    test_input = np.array([[2, 3], [1, 0]])
    # Esperado: [[3, 4], [2, 1]] -> rotar 180° -> [[1, 2], [4, 3]]
    expected_output = np.array([[1, 2], [4, 3]])
    
    solver = ARCSwarmSolverImproved(
        population_size=25,
        generations=5,
        mutation_rate=0.4
    )
    
    solution, report = solver.solve_with_swarm(train_examples, test_input)
    
    print(f"\nInput de prueba:\n{test_input}")
    print(f"\nSolución esperada:\n{expected_output}")
    print(f"\nSolución obtenida:\n{solution}")
    
    if solution is not None and np.array_equal(solution, expected_output):
        print("✅ TEST PASADO: La cadena de transformaciones es correcta")
    else:
        print("❌ TEST FALLADO: La cadena no funciona correctamente")
    
    # Mostrar cadenas exitosas
    if 'shared_memory_chains' in report and report['shared_memory_chains'] > 0:
        print(f"\n🔗 Cadenas de transformaciones descubiertas: {report['shared_memory_chains']}")
    
    return solution is not None and np.array_equal(solution, expected_output)

def main():
    """Ejecuta todos los tests"""
    print("\n" + "🚀" * 30)
    print("INICIANDO PRUEBAS DEL SOLVER MEJORADO")
    print("🚀" * 30)
    
    results = {
        "Color Mapping": test_color_mapping(),
        "Rotación": test_rotation(),
        "Extracción de Patrón": test_pattern_extraction(),
        "Cadena de Transformaciones": test_chain_transformations()
    }
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASADO" if passed else "❌ FALLADO"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests pasados")
    
    if total_passed == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON! El solver mejorado funciona correctamente.")
    else:
        print(f"\n⚠️ {total_tests - total_passed} tests fallaron. Revisar implementación.")

if __name__ == "__main__":
    main()