#!/usr/bin/env python3
"""
Test del módulo de aumentación
"""

import numpy as np
from arc_augmentation import ARCAugmentation, AugmentationType
from arc_solver_python import ARCSolverPython
import json

def test_augmentation():
    print("🧪 Probando módulo de aumentación...\n")
    
    augmenter = ARCAugmentation()
    
    # Puzzle de ejemplo: mapeo de color simple
    test_puzzle = {
        'input': [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        'output': [
            [0, 0, 2, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
    }
    
    print("📊 Puzzle original:")
    print(f"Input shape: {len(test_puzzle['input'])}x{len(test_puzzle['input'][0])}")
    print("Input:")
    for row in test_puzzle['input']:
        print(' '.join(str(x) for x in row))
    print("\nOutput:")
    for row in test_puzzle['output']:
        print(' '.join(str(x) for x in row))
    
    # Generar aumentaciones
    print("\n🔄 Generando aumentaciones...\n")
    
    augmentation_types = [
        AugmentationType.TRANSLATION,
        AugmentationType.COLOR_PERMUTATION,
        AugmentationType.ROTATION,
        AugmentationType.REFLECTION
    ]
    
    for aug_type in augmentation_types:
        print(f"\n--- {aug_type.value.upper()} ---")
        augmented = augmenter.augment_puzzle(test_puzzle, [aug_type])
        
        if augmented:
            aug_puzzle = augmented[0]
            print(f"Aumentación aplicada: {aug_puzzle.get('augmentation', 'unknown')}")
            
            if 'translation' in aug_puzzle:
                print(f"Translation: dx={aug_puzzle['translation']['dx']}, dy={aug_puzzle['translation']['dy']}")
            elif 'color_map' in aug_puzzle:
                print(f"Color map: {aug_puzzle['color_map']}")
            elif 'rotation' in aug_puzzle:
                print(f"Rotation: {aug_puzzle['rotation']}°")
            elif 'reflection_axis' in aug_puzzle:
                print(f"Reflection: {aug_puzzle['reflection_axis']}")
            
            print("\nInput aumentado:")
            for row in aug_puzzle['input']:
                print(' '.join(str(x) for x in row))
            print("\nOutput aumentado:")
            for row in aug_puzzle['output']:
                print(' '.join(str(x) for x in row))

def test_solver_with_augmentation():
    print("\n\n🤖 Probando solver con aumentación...\n")
    
    solver = ARCSolverPython()
    
    # Crear un puzzle más complejo
    puzzle = {
        'trainExamples': [
            {
                'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                'output': [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
            },
            {
                'input': [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                'output': [[2, 2, 0], [2, 2, 0], [0, 0, 0]]
            }
        ],
        'testExample': {
            'input': [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            'output': [[0, 0, 0], [0, 2, 2], [0, 2, 2]]
        }
    }
    
    # Evaluar con y sin aumentación
    print("📊 Evaluando impacto de la aumentación...")
    results = solver.evaluate_with_augmentation([puzzle])
    
    print(f"\n✅ Resultados:")
    print(f"Sin aumentación: {results['accuracy_without']:.1%}")
    print(f"Con aumentación: {results['accuracy_with']:.1%}")
    print(f"Mejora: {results['improvement']:+.1%}")
    
    # Mostrar pasos de razonamiento
    print("\n🧠 Pasos de razonamiento con aumentación:")
    solver.use_augmentation = True
    test_input = np.array(puzzle['testExample']['input'])
    solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
    
    for step in steps[:5]:  # Mostrar primeros 5 pasos
        print(f"- {step['type']}: {step['description']}")
        if 'details' in step:
            print(f"  Detalles: {step['details']}")

def test_augmentation_validation():
    print("\n\n🔍 Validando aumentaciones...\n")
    
    augmenter = ARCAugmentation()
    solver = ARCSolverPython()
    
    # Puzzle con regla clara (gravedad)
    puzzle = {
        'input': [
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        'output': [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 2, 0, 2, 0]
        ]
    }
    
    print("📊 Validando que las aumentaciones preservan la lógica del puzzle...")
    
    # Detectar regla original
    original_rule = solver.detect_rule(
        np.array(puzzle['input']), 
        np.array(puzzle['output'])
    )
    print(f"Regla original: {original_rule['type'] if original_rule else 'No detectada'}")
    
    # Probar cada tipo de aumentación
    for aug_type in [AugmentationType.TRANSLATION, AugmentationType.ROTATION, AugmentationType.REFLECTION]:
        augmented = augmenter.augment_puzzle(puzzle, [aug_type])
        
        if augmented:
            aug_puzzle = augmented[0]
            is_valid = augmenter.validate_augmentation(puzzle, aug_puzzle, solver)
            
            print(f"\n{aug_type.value}: {'✅ Válida' if is_valid else '❌ Inválida'}")
            
            if not is_valid:
                # Mostrar por qué falló
                aug_rule = solver.detect_rule(
                    np.array(aug_puzzle['input']),
                    np.array(aug_puzzle['output'])
                )
                print(f"  Regla aumentada: {aug_rule['type'] if aug_rule else 'No detectada'}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DE AUMENTACIÓN PARA ARC SOLVER")
    print("=" * 60)
    
    test_augmentation()
    test_solver_with_augmentation()
    test_augmentation_validation()
    
    print("\n\n✅ Tests completados!")