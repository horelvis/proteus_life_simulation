#!/usr/bin/env python3
"""
Test específico para gravity detection
"""

import numpy as np
from arc_solver_python import ARCSolverPython

def test_gravity():
    print("🧪 Testing Gravity Detection and Application\n")
    
    solver = ARCSolverPython()
    
    # Test case 1: Simple gravity
    test1 = {
        'input': np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4]
        ]),
        'output': np.array([
            [0, 0, 0],
            [1, 0, 2],
            [3, 0, 4]
        ])
    }
    
    print("Test 1: Simple gravity")
    print("Input:")
    print(test1['input'])
    print("\nOutput:")
    print(test1['output'])
    
    # Detectar regla
    rule = solver._detect_gravity(test1['input'], test1['output'])
    if rule:
        print(f"\n✅ Gravity rule detected! Confidence: {rule['confidence']}")
    else:
        print("\n❌ No gravity rule detected")
        print("Debugging...")
        
        # Debug columna por columna
        for j in range(test1['input'].shape[1]):
            input_col = test1['input'][:, j]
            output_col = test1['output'][:, j]
            print(f"\nColumn {j}:")
            print(f"  Input:  {input_col}")
            print(f"  Output: {output_col}")
            
            input_elements = input_col[input_col != 0]
            output_elements = output_col[output_col != 0]
            print(f"  Non-zero in:  {input_elements}")
            print(f"  Non-zero out: {output_elements}")
            print(f"  Same elements? {np.array_equal(sorted(input_elements), sorted(output_elements))}")
            
            if len(output_elements) > 0:
                first_non_zero = np.argmax(output_col != 0)
                expected_start = len(output_col) - len(output_elements)
                print(f"  First non-zero at: {first_non_zero}, expected: {expected_start}")
    
    # Aplicar regla
    if rule:
        predicted = solver._apply_gravity(test1['input'])
        print(f"\nPredicted:")
        print(predicted)
        print(f"\nMatch: {np.array_equal(predicted, test1['output'])}")
    
    # Test case 2: Different arrangement
    print("\n" + "="*50 + "\n")
    test2 = {
        'input': np.array([
            [5, 0, 0],
            [0, 6, 0],
            [0, 0, 0]
        ]),
        'output': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [5, 6, 0]
        ])
    }
    
    print("Test 2: Sparse elements")
    rule2 = solver._detect_gravity(test2['input'], test2['output'])
    if rule2:
        print(f"✅ Gravity detected!")
        predicted2 = solver._apply_gravity(test2['input'])
        print(f"Match: {np.array_equal(predicted2, test2['output'])}")
    else:
        print("❌ No gravity detected")
    
    # Test case 3: Multiple elements per column
    print("\n" + "="*50 + "\n")
    test3 = {
        'input': np.array([
            [1, 2, 0],
            [0, 0, 3],
            [4, 0, 0],
            [0, 5, 0]
        ]),
        'output': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 2, 0],
            [4, 5, 3]
        ])
    }
    
    print("Test 3: Multiple elements falling")
    print("Input:")
    print(test3['input'])
    print("\nExpected output:")
    print(test3['output'])
    
    rule3 = solver._detect_gravity(test3['input'], test3['output'])
    if rule3:
        print(f"\n✅ Gravity detected!")
        predicted3 = solver._apply_gravity(test3['input'])
        print("\nPredicted:")
        print(predicted3)
        print(f"\nMatch: {np.array_equal(predicted3, test3['output'])}")
    else:
        print("\n❌ No gravity detected")
    
    # Test full solve_with_steps
    print("\n" + "="*50 + "\n")
    print("Testing full solve_with_steps:")
    
    puzzle = {
        'trainExamples': [
            {
                'input': [[1, 0, 2], [0, 0, 0], [3, 0, 4]],
                'output': [[0, 0, 0], [1, 0, 2], [3, 0, 4]]
            }
        ]
    }
    
    test_input = np.array([[5, 0, 0], [0, 6, 0], [0, 0, 0]])
    
    # Disable augmentation for this test
    solver.use_augmentation = False
    solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
    
    print("Solution:")
    print(solution)
    
    expected = np.array([[0, 0, 0], [0, 0, 0], [5, 6, 0]])
    print("\nExpected:")
    print(expected)
    print(f"\nMatch: {np.array_equal(solution, expected)}")
    
    print("\nReasoning steps:")
    for step in steps:
        if 'rule' in step or step['type'] in ['rule_detection', 'rule_selection']:
            print(f"- {step['type']}: {step.get('description', '')}")
            if 'rule' in step and step['rule']:
                print(f"  Rule type: {step['rule'].get('type', 'unknown')}")

if __name__ == "__main__":
    test_gravity()