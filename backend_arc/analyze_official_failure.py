#!/usr/bin/env python3
"""
Analiza por qué falla con puzzles oficiales
"""

import numpy as np
from arc_solver_python import ARCSolverPython
from arc_official_loader import ARCOfficialLoader

def print_grid(grid, title=""):
    """Imprime una grilla en formato texto"""
    if title:
        print(f"\n{title}:")
    
    # Mapeo de números a caracteres para mejor visualización
    char_map = {
        0: '.',  # Fondo
        1: '1',  # Azul
        2: '2',  # Rojo
        3: '3',  # Verde
        4: '4',  # Amarillo
        5: '5',  # Gris
        6: '6',  # Magenta
        7: '7',  # Naranja
        8: '8',  # Azul claro
        9: '9'   # Marrón
    }
    
    for row in grid:
        print('   ', ' '.join(char_map.get(cell, str(cell)) for cell in row))

def analyze_puzzle_00d62c1b():
    """Analiza específicamente el puzzle 00d62c1b"""
    print("🔍 Analizando puzzle 00d62c1b en detalle...\n")
    
    solver = ARCSolverPython()
    loader = ARCOfficialLoader()
    
    # Cargar puzzle
    puzzles = loader.load_specific_puzzles(['00d62c1b'])
    if not puzzles:
        print("Error cargando puzzle")
        return
    
    puzzle = puzzles[0]
    
    # Visualizar en texto
    print("📊 Visualización del puzzle:")
    
    for i, example in enumerate(puzzle['trainExamples']):
        print(f"\n--- Ejemplo de entrenamiento {i+1} ---")
        print_grid(np.array(example['input']), "Input")
        print_grid(np.array(example['output']), "Output")
    
    print("\n--- Test ---")
    print_grid(np.array(puzzle['testExample']['input']), "Test Input")
    print_grid(np.array(puzzle['testExample']['output']), "Expected Output")
    
    # Analizar cada ejemplo manualmente
    print("\n📐 Análisis manual de transformaciones:")
    
    for i, example in enumerate(puzzle['trainExamples']):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"\n   Ejemplo {i+1}:")
        print(f"   - Input shape: {input_grid.shape}")
        print(f"   - Output shape: {output_grid.shape}")
        print(f"   - Colores en input: {sorted(np.unique(input_grid))}")
        print(f"   - Colores en output: {sorted(np.unique(output_grid))}")
        
        # Contar cambios de color
        color_changes = {}
        for row in range(input_grid.shape[0]):
            for col in range(input_grid.shape[1]):
                in_color = input_grid[row, col]
                out_color = output_grid[row, col]
                if in_color != out_color:
                    key = f"{in_color}→{out_color}"
                    color_changes[key] = color_changes.get(key, 0) + 1
        
        print(f"   - Cambios de color: {color_changes}")
    
    # Probar todas las reglas
    print("\n🧪 Probando todas las reglas disponibles:")
    solver.use_augmentation = False
    
    rules_to_test = [
        'color_mapping', 'pattern_replication', 'reflection', 
        'rotation', 'gravity', 'counting', 'fill_shape',
        'symmetry_detection', 'pattern_extraction', 'line_drawing'
    ]
    
    for rule_type in rules_to_test:
        # Detectar si aplica
        applies = False
        for example in puzzle['trainExamples']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            rule = solver.detect_rule(input_grid, output_grid)
            if rule and rule['type'] == rule_type:
                applies = True
                print(f"   ✓ {rule_type}: Detectado (confianza: {rule['confidence']})")
                break
        
        if not applies:
            print(f"   ✗ {rule_type}: No detectado")
    
    # Resolver con el solver actual
    print("\n🤖 Resolviendo con solver actual:")
    test_input = np.array(puzzle['testExample']['input'])
    solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
    
    expected = np.array(puzzle['testExample']['output'])
    accuracy = np.sum(solution == expected) / expected.size
    
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Regla aplicada: {steps[-2]['description'] if len(steps) > 1 else 'None'}")
    
    # Analizar diferencias píxel por píxel
    if not np.array_equal(solution, expected):
        print("\n❌ Análisis de errores:")
        diff_count = 0
        for row in range(expected.shape[0]):
            for col in range(expected.shape[1]):
                if solution[row, col] != expected[row, col]:
                    diff_count += 1
                    if diff_count <= 5:  # Mostrar solo primeros 5
                        print(f"   - Posición ({row},{col}): "
                              f"predijo {solution[row, col]}, esperaba {expected[row, col]}")
        
        if diff_count > 5:
            print(f"   ... y {diff_count - 5} diferencias más")

def analyze_failure_patterns():
    """Analiza patrones de fallo comunes"""
    print("\n\n📊 Analizando patrones de fallo en puzzles oficiales...\n")
    
    solver = ARCSolverPython()
    loader = ARCOfficialLoader()
    
    # Cargar varios puzzles
    puzzle_ids = ['00d62c1b', '0520fde7', '0a938d79']
    puzzles = loader.load_specific_puzzles(puzzle_ids)
    
    failure_patterns = {
        'no_rule_detected': 0,
        'wrong_rule': 0,
        'partial_success': 0,
        'shape_mismatch': 0
    }
    
    for puzzle in puzzles:
        print(f"\n🧩 Puzzle {puzzle['id']}:")
        
        # Intentar resolver
        test_input = np.array(puzzle['testExample']['input'])
        solution, steps = solver.solve_with_steps(puzzle['trainExamples'], test_input)
        expected = np.array(puzzle['testExample']['output'])
        
        # Analizar resultado
        if solution.shape != expected.shape:
            failure_patterns['shape_mismatch'] += 1
            print("   ❌ Error: Tamaño de salida incorrecto")
        else:
            accuracy = np.sum(solution == expected) / expected.size
            if accuracy == 0:
                failure_patterns['no_rule_detected'] += 1
                print("   ❌ Error: No detectó regla correcta")
            elif accuracy < 1.0:
                failure_patterns['partial_success'] += 1
                print(f"   ⚠️  Parcial: {accuracy*100:.1f}% correcto")
            else:
                print("   ✅ Correcto!")
    
    print("\n📈 Resumen de patrones de fallo:")
    for pattern, count in failure_patterns.items():
        print(f"   {pattern}: {count}")

if __name__ == "__main__":
    # Análisis detallado del primer puzzle
    analyze_puzzle_00d62c1b()
    
    # Análisis de patrones de fallo
    analyze_failure_patterns()