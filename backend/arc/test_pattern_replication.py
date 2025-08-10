#!/usr/bin/env python3
"""
Test específico para pattern replication
"""

import numpy as np
from arc.arc_solver_python import ARCSolverPython

def test_pattern_replication():
    print("🧪 Testing Pattern Replication Detection and Application\n")
    
    solver = ARCSolverPython()
    
    # Test case 1: Simple 2x2 to 6x6 (3x replication)
    test1 = {
        'input': np.array([
            [1, 0],
            [0, 1]
        ]),
        'output': np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]
        ])
    }
    
    print("Test 1: 2x2 → 6x6 (3x replication)")
    print("Input shape:", test1['input'].shape)
    print("Output shape:", test1['output'].shape)
    
    # Detectar regla
    rule = solver._detect_pattern_replication(test1['input'], test1['output'])
    if rule:
        print(f"✅ Rule detected! Confidence: {rule['confidence']}, Factor: {rule['parameters']['factor']}")
    else:
        print("❌ No rule detected")
    
    # Aplicar regla
    if rule:
        predicted = solver._apply_pattern_replication(test1['input'], rule['parameters']['factor'])
        print(f"\nPredicted shape: {predicted.shape}")
        print("Predicted:")
        print(predicted)
        print("\nExpected:")
        print(test1['output'])
        print(f"\nMatch: {np.array_equal(predicted, test1['output'])}")
    
    # Test case 2: Single color 2x2 to 6x6
    print("\n" + "="*50 + "\n")
    test2 = {
        'input': np.array([
            [2, 2],
            [2, 2]
        ]),
        'output': np.array([
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ])
    }
    
    print("Test 2: Uniform 2x2 → 6x6")
    rule2 = solver._detect_pattern_replication(test2['input'], test2['output'])
    if rule2:
        print(f"✅ Rule detected! Factor: {rule2['parameters']['factor']}")
        predicted2 = solver._apply_pattern_replication(test2['input'], rule2['parameters']['factor'])
        print(f"Match: {np.array_equal(predicted2, test2['output'])}")
    else:
        print("❌ No rule detected")
    
    # Test case 3: 2x replication
    print("\n" + "="*50 + "\n")
    test3 = {
        'input': np.array([
            [3, 0, 3],
            [0, 3, 0]
        ]),
        'output': np.array([
            [3, 3, 0, 0, 3, 3],
            [3, 3, 0, 0, 3, 3],
            [0, 0, 3, 3, 0, 0],
            [0, 0, 3, 3, 0, 0]
        ])
    }
    
    print("Test 3: 2x3 → 4x6 (2x replication)")
    rule3 = solver._detect_pattern_replication(test3['input'], test3['output'])
    if rule3:
        print(f"✅ Rule detected! Factor: {rule3['parameters']['factor']}")
        predicted3 = solver._apply_pattern_replication(test3['input'], rule3['parameters']['factor'])
        print(f"Match: {np.array_equal(predicted3, test3['output'])}")
    else:
        print("❌ No rule detected")
    
    # Test full solve_with_steps
    print("\n" + "="*50 + "\n")
    print("Testing full solve_with_steps:")
    
    puzzle = {
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
        ]
    }
    
    test_input = np.array([[2, 2], [2, 2]])
    solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
    
    print(f"Solution shape: {solution.shape}")
    print("Solution:")
    print(solution)
    
    print("\nReasoning steps:")
    for step in steps:
        if step['type'] == 'rule_detection':
            print(f"- {step['description']}")

if __name__ == "__main__":
    test_pattern_replication()