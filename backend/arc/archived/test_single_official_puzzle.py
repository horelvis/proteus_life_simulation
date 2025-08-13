#!/usr/bin/env python3
"""
Test con un solo puzzle oficial para debug
"""

import numpy as np
from arc.arc_solver_python import ARCSolverPython
from arc.arc_official_loader import ARCOfficialLoader

def test_single_puzzle():
    print("🧪 Probando con puzzle oficial simple...\n")
    
    # Crear solver y loader
    solver = ARCSolverPython()
    loader = ARCOfficialLoader()
    
    # Puzzle conocido simple: 00d62c1b
    # Este es un puzzle de transformación de color simple
    puzzle_data = {
        "train": [
            {
                "input": [[0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 0],
                         [0, 2, 1, 2, 0],
                         [0, 2, 2, 2, 0],
                         [0, 0, 0, 0, 0]],
                "output": [[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]]
            },
            {
                "input": [[0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 1, 2, 2, 1, 0],
                         [0, 1, 2, 2, 1, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0]],
                "output": [[0, 0, 0, 0, 0, 0],
                          [0, 2, 2, 2, 2, 0],
                          [0, 2, 1, 1, 2, 0],
                          [0, 2, 1, 1, 2, 0],
                          [0, 2, 2, 2, 2, 0],
                          [0, 0, 0, 0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 2, 0],
                         [0, 2, 1, 1, 1, 2, 0],
                         [0, 2, 1, 1, 1, 2, 0],
                         [0, 2, 1, 1, 1, 2, 0],
                         [0, 2, 2, 2, 2, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0]],
                "output": [[0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 2, 2, 2, 1, 0],
                          [0, 1, 2, 2, 2, 1, 0],
                          [0, 1, 2, 2, 2, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0]]
            }
        ]
    }
    
    # Procesar puzzle
    puzzle = loader._process_puzzle(puzzle_data, "00d62c1b")
    
    print(f"📊 Puzzle ID: {puzzle['id']}")
    print(f"   Ejemplos de entrenamiento: {len(puzzle['trainExamples'])}")
    print(f"   Tamaño test input: {np.array(puzzle['testExample']['input']).shape}")
    
    # Desactivar aumentación para debugging
    solver.use_augmentation = False
    
    # Resolver
    print("\n🔍 Analizando ejemplos de entrenamiento...")
    
    # Detectar regla en cada ejemplo
    for i, example in enumerate(puzzle['trainExamples']):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        rule = solver.detect_rule(input_grid, output_grid)
        if rule:
            print(f"   Ejemplo {i+1}: Regla detectada - {rule['type']} (confianza: {rule['confidence']})")
        else:
            print(f"   Ejemplo {i+1}: No se detectó regla")
    
    # Resolver puzzle
    print("\n🧩 Resolviendo puzzle...")
    test_input = np.array(puzzle['testExample']['input'])
    solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
    
    # Evaluar
    expected = np.array(puzzle['testExample']['output'])
    correct = np.array_equal(solution, expected)
    
    print(f"\n📊 Resultado:")
    print(f"   Correcto: {'✅' if correct else '❌'}")
    
    if not correct:
        print(f"   Accuracy: {np.sum(solution == expected) / expected.size * 100:.1f}%")
        print("\n   Solución generada:")
        for row in solution:
            print("   ", row)
        print("\n   Esperado:")
        for row in expected:
            print("   ", row)
    
    # Mostrar pasos de razonamiento
    print("\n🧠 Pasos de razonamiento:")
    for step in steps:
        if step['type'] in ['rule_detection', 'rule_selection']:
            print(f"   - {step['description']}")

if __name__ == "__main__":
    test_single_puzzle()