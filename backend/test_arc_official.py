#!/usr/bin/env python3
"""
Pruebas con puzzles oficiales de ARC
Usando solo el modelo de razonamiento lógico
"""

import numpy as np
import json
import os
from glob import glob
from typing import Dict, List
from arc import ARCSolver, LogicalReasoningNetwork

def load_official_puzzles(limit=5):
    """Carga puzzles oficiales de ARC"""
    puzzles = []
    cache_dir = "arc_official_cache"
    
    if os.path.exists(cache_dir):
        files = sorted(glob(os.path.join(cache_dir, "*.json")))[:limit]
        
        for file_path in files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                puzzles.append({
                    'name': os.path.basename(file_path).replace('.json', ''),
                    'data': data
                })
    
    return puzzles

def analyze_puzzle(puzzle_name: str, puzzle_data: Dict):
    """Analiza un puzzle específico con detalles"""
    print(f"\n{'='*60}")
    print(f"📝 PUZZLE: {puzzle_name}")
    print(f"{'='*60}")
    
    train = puzzle_data['train']
    test_input = np.array(puzzle_data['test'][0]['input'])
    
    # Información del puzzle
    print(f"📊 Información:")
    print(f"  - Ejemplos de entrenamiento: {len(train)}")
    print(f"  - Tamaño entrada test: {test_input.shape}")
    
    for i, example in enumerate(train[:2]):  # Primeros 2 ejemplos
        inp = np.array(example['input'])
        out = np.array(example['output'])
        print(f"  - Ejemplo {i+1}: {inp.shape} → {out.shape}")
        
        # Detectar cambio de tamaño
        if inp.shape != out.shape:
            print(f"    ⚠️ Cambio de tamaño detectado!")
        
        # Detectar mapeo de valores
        unique_in = np.unique(inp)
        unique_out = np.unique(out)
        if set(unique_in) != set(unique_out):
            print(f"    🔄 Valores: {unique_in.tolist()} → {unique_out.tolist()}")
    
    # Resolver
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    print(f"\n🧠 Razonamiento:")
    
    # Mostrar inferencias por nivel
    if hasattr(solver, 'inferences'):
        macro_inf = [inf for inf in solver.inferences if inf.level == 'macro']
        meso_inf = [inf for inf in solver.inferences if inf.level == 'meso']
        micro_inf = [inf for inf in solver.inferences if inf.level == 'micro']
        
        if macro_inf:
            print(f"  MACRO ({len(macro_inf)} inferencias):")
            for inf in macro_inf[:2]:
                print(f"    • {inf.conclusion[:60]}")
        
        if meso_inf:
            print(f"  MESO ({len(meso_inf)} inferencias):")
            for inf in meso_inf[:2]:
                print(f"    • {inf.conclusion[:60]}")
        
        if micro_inf:
            print(f"  MICRO ({len(micro_inf)} inferencias):")
            for inf in micro_inf[:2]:
                print(f"    • {inf.conclusion[:60]}")
    
    # Resultado
    if 'output' in puzzle_data['test'][0]:
        expected = np.array(puzzle_data['test'][0]['output'])
        if solution.shape == expected.shape:
            accuracy = np.mean(solution == expected) * 100
            if accuracy == 100:
                print(f"\n✅ PERFECTO: 100% accuracy")
            elif accuracy > 90:
                print(f"\n🔶 MUY BIEN: {accuracy:.1f}% accuracy")
            elif accuracy > 70:
                print(f"\n⚠️ PARCIAL: {accuracy:.1f}% accuracy")
            else:
                print(f"\n❌ BAJO: {accuracy:.1f}% accuracy")
        else:
            print(f"\n❌ ERROR: Tamaño incorrecto {solution.shape} vs {expected.shape}")
            accuracy = 0
    else:
        print(f"\n❓ Sin solución conocida")
        accuracy = None
    
    print(f"\n💾 Solución shape: {solution.shape}")
    
    # Mostrar pequeña muestra de la solución
    if solution.size <= 25:
        print(f"Solución:\n{solution}")
    
    return accuracy

def test_specific_patterns():
    """Prueba patrones específicos conocidos en ARC"""
    print("\n" + "="*80)
    print("🔬 PRUEBAS DE PATRONES ESPECÍFICOS")
    print("="*80)
    
    patterns = {
        "Repetición 3x": {
            'train': [{
                'input': [[1, 2], [3, 4]],
                'output': [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
            }],
            'test': [{'input': [[5, 6], [7, 8]]}],
            'expected_shape': (2, 6)
        },
        "Gravedad": {
            'train': [{
                'input': [[1, 0, 2], [0, 0, 0], [3, 0, 4]],
                'output': [[0, 0, 0], [1, 0, 2], [3, 0, 4]]
            }],
            'test': [{'input': [[5, 0, 0], [0, 6, 0], [0, 0, 7]]}],
            'expected_shape': (3, 3)
        },
        "Mapeo +1": {
            'train': [{
                'input': [[1, 2, 3], [4, 5, 6]],
                'output': [[2, 3, 4], [5, 6, 7]]
            }],
            'test': [{'input': [[7, 8, 9], [1, 2, 3]]}],
            'expected_shape': (2, 3)
        }
    }
    
    for pattern_name, puzzle_data in patterns.items():
        print(f"\n🧪 Patrón: {pattern_name}")
        print("-" * 40)
        
        solver = ARCSolver()
        test_input = np.array(puzzle_data['test'][0]['input'])
        solution = solver.reason(puzzle_data['train'], test_input)
        
        shape_ok = solution.shape == puzzle_data['expected_shape']
        print(f"  Shape esperado: {puzzle_data['expected_shape']}")
        print(f"  Shape obtenido: {solution.shape}")
        
        if shape_ok:
            print(f"  ✅ Dimensiones correctas")
        else:
            print(f"  ❌ Dimensiones incorrectas")
        
        # Mostrar razonamiento
        if hasattr(solver, 'inferences') and solver.inferences:
            print(f"  Inferencia: {solver.inferences[0].conclusion[:50]}")

def main():
    print("="*80)
    print("🧠 PRUEBAS CON PUZZLES OFICIALES DE ARC")
    print("Red de Razonamiento Lógico (Macro→Meso→Micro)")
    print("="*80)
    
    # Cargar puzzles
    puzzles = load_official_puzzles(limit=3)
    
    if not puzzles:
        print("❌ No se encontraron puzzles oficiales")
        return
    
    print(f"\n📁 Cargados {len(puzzles)} puzzles oficiales")
    
    # Analizar cada puzzle
    accuracies = []
    for puzzle in puzzles:
        acc = analyze_puzzle(puzzle['name'], puzzle['data'])
        if acc is not None:
            accuracies.append(acc)
    
    # Resumen
    if accuracies:
        print("\n" + "="*80)
        print("📊 RESUMEN")
        print("="*80)
        print(f"  Puzzles evaluados: {len(accuracies)}")
        print(f"  Accuracy promedio: {np.mean(accuracies):.1f}%")
        print(f"  Accuracy máximo: {max(accuracies):.1f}%")
        print(f"  Accuracy mínimo: {min(accuracies):.1f}%")
        print(f"  Puzzles perfectos: {sum(1 for a in accuracies if a == 100)}")
    
    # Pruebas de patrones específicos
    test_specific_patterns()

if __name__ == "__main__":
    main()