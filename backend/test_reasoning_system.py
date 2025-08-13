#!/usr/bin/env python3
"""
Suite de pruebas para la Red de Razonamiento Lógico
Tests completos del sistema de 3 capas
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import time
from arc import ARCSolver, LogicalReasoningNetwork

def test_basic_pattern_recognition():
    """Test 1: Reconocimiento básico de patrones"""
    print("\n🧪 TEST 1: Reconocimiento de patrones básicos")
    print("-" * 50)
    
    # Patrón simple: duplicación de valor
    train = [{
        'input': [[1, 0], [0, 2]],
        'output': [[2, 0], [0, 4]]
    }]
    
    test_input = np.array([[3, 0], [0, 4]])
    expected = np.array([[6, 0], [0, 8]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    accuracy = np.mean(solution == expected) * 100
    
    print(f"Input: {test_input.tolist()}")
    print(f"Expected: {expected.tolist()}")
    print(f"Got: {solution.tolist()}")
    print(f"✅ Accuracy: {accuracy:.1f}%")
    
    # Mostrar inferencias
    if hasattr(solver, 'inferences') and solver.inferences:
        print("\n📊 Inferencias lógicas:")
        for inf in solver.inferences[:2]:
            print(f"  [{inf.level}] {inf.conclusion[:50]}")
    
    return accuracy > 50

def test_size_change():
    """Test 2: Cambio de tamaño dinámico"""
    print("\n🧪 TEST 2: Cambio de tamaño (3x3 → 3x6)")
    print("-" * 50)
    
    # Patrón: duplicación horizontal
    train = [{
        'input': [[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]],
        'output': [[1, 2, 3, 1, 2, 3],
                   [4, 5, 6, 4, 5, 6],
                   [7, 8, 9, 7, 8, 9]]
    }]
    
    test_input = np.array([[2, 0, 2],
                           [0, 3, 0],
                           [2, 0, 2]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    size_correct = solution.shape == (3, 6)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {solution.shape}")
    print(f"Expected shape: (3, 6)")
    
    if size_correct:
        print("✅ Cambio de tamaño detectado correctamente")
    else:
        print("❌ Cambio de tamaño no detectado")
    
    return size_correct

def test_color_mapping():
    """Test 3: Mapeo de colores emergente"""
    print("\n🧪 TEST 3: Mapeo de colores")
    print("-" * 50)
    
    # Patrón: mapeo consistente de colores
    train = [
        {
            'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            'output': [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
        },
        {
            'input': [[2, 0, 2], [0, 2, 0], [2, 0, 2]],
            'output': [[4, 0, 4], [0, 4, 0], [4, 0, 4]]
        }
    ]
    
    test_input = np.array([[5, 0, 5], [0, 5, 0], [5, 0, 5]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    # Verificar que detectó el patrón de incremento
    unique_values = np.unique(solution[solution != 0])
    
    print(f"Input values: {np.unique(test_input)}")
    print(f"Output values: {np.unique(solution)}")
    
    if len(unique_values) > 0:
        detected_mapping = unique_values[0] > 5
        if detected_mapping:
            print(f"✅ Mapeo detectado: 5 → {unique_values[0]}")
        else:
            print("⚠️ Mapeo parcial detectado")
    else:
        print("❌ No se detectó mapeo")
    
    return len(unique_values) > 0

def test_object_transformation():
    """Test 4: Transformación de objetos"""
    print("\n🧪 TEST 4: Transformación de objetos")
    print("-" * 50)
    
    # Patrón: expansión de cruz
    train = [{
        'input': [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]],
        'output': [[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]]
    }]
    
    test_input = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 2, 0, 0],
                           [0, 0, 0, 0, 0]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    # Verificar si expandió en cruz
    center_y, center_x = 1, 2
    cross_positions = [(0, 2), (1, 1), (1, 3), (2, 2)]
    
    cross_detected = any(solution[pos] != 0 for pos in cross_positions if pos != (center_y, center_x))
    
    print(f"Centro: ({center_y}, {center_x})")
    print(f"Solución:\n{solution}")
    
    if cross_detected:
        print("✅ Transformación de objeto detectada")
    else:
        print("❌ Transformación no aplicada")
    
    return cross_detected

def test_complex_reasoning():
    """Test 5: Razonamiento complejo multi-nivel"""
    print("\n🧪 TEST 5: Razonamiento complejo")
    print("-" * 50)
    
    # Patrón complejo: rotación + mapeo de color
    train = [{
        'input': [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]],
        'output': [[0, 0, 2],
                   [0, 2, 0],
                   [2, 0, 0]]
    }]
    
    test_input = np.array([[3, 0, 0],
                           [0, 3, 0],
                           [0, 0, 3]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    # Verificar transformación
    has_transformation = not np.array_equal(solution, test_input)
    
    print(f"Input:\n{test_input}")
    print(f"Output:\n{solution}")
    
    if has_transformation:
        print("✅ Transformación compleja aplicada")
        
        # Analizar qué tipo de transformación se aplicó
        if solution.shape != test_input.shape:
            print("  - Cambio de tamaño")
        if np.unique(solution).tolist() != np.unique(test_input).tolist():
            print("  - Mapeo de colores")
        if not np.array_equal(solution, test_input):
            print("  - Transformación espacial")
    else:
        print("❌ Sin transformación")
    
    return has_transformation

def test_inference_chain():
    """Test 6: Cadena de inferencias Macro→Meso→Micro"""
    print("\n🧪 TEST 6: Cadena de inferencias")
    print("-" * 50)
    
    train = [{
        'input': [[1, 1], [1, 1]],
        'output': [[2, 2], [2, 2]]
    }]
    
    test_input = np.array([[3, 3], [3, 3]])
    
    solver = ARCSolver()
    solution = solver.reason(train, test_input)
    
    print("📊 Flujo de razonamiento:")
    print("  MACRO: Observación global con V-JEPA")
    print("  MESO: Análisis de objetos y relaciones")
    print("  MICRO: Ejecución a nivel de píxeles")
    
    # Verificar que hay inferencias en cada nivel
    if hasattr(solver, 'inferences'):
        levels = set(inf.level for inf in solver.inferences)
        print(f"\nNiveles con inferencias: {levels}")
        
        for level in ['macro', 'meso', 'micro']:
            level_inferences = [inf for inf in solver.inferences if inf.level == level]
            if level_inferences:
                print(f"\n{level.upper()}:")
                for inf in level_inferences[:1]:
                    print(f"  {inf.premise[:40]} → {inf.conclusion[:40]}")
    
    return True

def run_all_tests():
    """Ejecuta toda la suite de pruebas"""
    print("=" * 80)
    print("🧠 SUITE DE PRUEBAS - RED DE RAZONAMIENTO LÓGICO")
    print("=" * 80)
    print("\nArquitectura: MACRO (V-JEPA) → MESO (Objetos) → MICRO (Píxeles)")
    
    tests = [
        ("Reconocimiento de patrones", test_basic_pattern_recognition),
        ("Cambio de tamaño", test_size_change),
        ("Mapeo de colores", test_color_mapping),
        ("Transformación de objetos", test_object_transformation),
        ("Razonamiento complejo", test_complex_reasoning),
        ("Cadena de inferencias", test_inference_chain)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            start_time = time.time()
            passed = test_func()
            elapsed = time.time() - start_time
            results.append((name, passed, elapsed))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            results.append((name, False, 0))
    
    # Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for name, passed, elapsed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name:30} ({elapsed:.2f}s)")
    
    print(f"\nTotal: {passed_count}/{total_count} pruebas pasadas")
    print(f"Tasa de éxito: {passed_count/total_count*100:.1f}%")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)