#!/usr/bin/env python3
"""
Test Paso a Paso del Sistema de Razonamiento Lógico
Verifica cada componente individualmente con GPU cuando sea posible
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple
import json

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def test_vjepa_observer():
    """Test 1: V-JEPA Observer (Capa MACRO)"""
    print("\n" + "="*60)
    print("🔬 TEST 1: V-JEPA Observer (MACRO)")
    print("="*60)
    
    from arc.vjepa_observer import VJEPAObserver
    
    observer = VJEPAObserver(embedding_dim=64)
    
    # Ejemplo simple
    input_grid = np.array([[1, 0, 1], 
                           [0, 2, 0], 
                           [1, 0, 1]])
    
    output_grid = np.array([[2, 0, 2], 
                            [0, 4, 0], 
                            [2, 0, 2]])
    
    print("\n📥 Input:")
    print(input_grid)
    print("\n📤 Output esperado:")
    print(output_grid)
    
    # Observar transición
    start = time.time()
    observation = observer.observe(input_grid, output_grid)
    elapsed = time.time() - start
    
    print(f"\n⏱️ Tiempo observación: {elapsed*1000:.2f}ms")
    
    # Resultados
    print("\n📊 Observación:")
    print(f"  - Patrón emergente: {observation['emergent_pattern']['type']}")
    print(f"  - Confianza: {observation['emergent_pattern']['confidence']:.2f}")
    
    # Embedding de transformación
    embed = observation['transform_embedding']
    print(f"  - Embedding shape: {embed.shape}")
    print(f"  - Embedding norm: {np.linalg.norm(embed):.2f}")
    
    # Test predicción
    test_input = np.array([[3, 0, 3], 
                          [0, 4, 0], 
                          [3, 0, 3]])
    
    print("\n🎯 Test de predicción:")
    print(f"  Input test:\n{test_input}")
    
    predicted = observer.predict_transformation(test_input)
    print(f"  Predicción:\n{predicted}")
    
    # Verificar con GPU si está disponible
    if device.type == 'cuda':
        print("\n🚀 Test GPU:")
        input_tensor = torch.from_numpy(input_grid).float().to(device)
        output_tensor = torch.from_numpy(output_grid).float().to(device)
        
        # Operación matricial en GPU
        diff = output_tensor - input_tensor
        print(f"  - Diferencia calculada en GPU: {diff.shape}")
        print(f"  - Norma de diferencia: {torch.norm(diff).item():.2f}")
    
    return observation

def test_emergent_rules():
    """Test 2: Sistema de Reglas Emergentes (3 capas)"""
    print("\n" + "="*60)
    print("🔬 TEST 2: Sistema de Reglas Emergentes")
    print("="*60)
    
    from arc.emergent_rule_system import EmergentRuleSystem
    
    rule_system = EmergentRuleSystem()
    
    # Ejemplos de entrenamiento
    train_examples = [
        {
            'input': [[1, 0], [0, 1]],
            'output': [[2, 0], [0, 2]]
        },
        {
            'input': [[3, 0], [0, 3]],
            'output': [[6, 0], [0, 6]]
        }
    ]
    
    print("\n📚 Ejemplos de entrenamiento:")
    for i, ex in enumerate(train_examples):
        print(f"  Ejemplo {i+1}: {np.array(ex['input']).shape} → {np.array(ex['output']).shape}")
    
    # Extraer reglas
    start = time.time()
    rules = rule_system.extract_rules_from_examples(train_examples)
    elapsed = time.time() - start
    
    print(f"\n⏱️ Tiempo extracción: {elapsed*1000:.2f}ms")
    
    # Analizar reglas por nivel
    print("\n📊 Reglas extraídas:")
    print(f"  MICRO (píxel): {len(rules['micro_rules'])} reglas")
    print(f"  MESO (objeto): {len(rules['meso_rules'])} reglas")
    print(f"  MACRO (patrón): {len(rules['macro_rules'])} reglas")
    
    # Mostrar algunas reglas
    if rules['micro_rules']:
        print("\n  Reglas MICRO:")
        for rule in rules['micro_rules'][:2]:
            print(f"    - {rule.pattern}: conf={rule.confidence:.2f}")
    
    if rules['meso_rules']:
        print("\n  Reglas MESO:")
        for rule in rules['meso_rules'][:2]:
            print(f"    - {rule.transformation}: conf={rule.confidence:.2f}")
    
    if rules['macro_rules']:
        print("\n  Reglas MACRO:")
        for rule in rules['macro_rules'][:2]:
            print(f"    - {rule.rule_type}: conf={rule.confidence:.2f}")
    
    return rules

def test_logical_reasoning_flow():
    """Test 3: Flujo completo Macro→Meso→Micro"""
    print("\n" + "="*60)
    print("🔬 TEST 3: Flujo Completo de Razonamiento")
    print("="*60)
    
    from arc.logical_reasoning_network import LogicalReasoningNetwork
    
    network = LogicalReasoningNetwork()
    
    # Ejemplo con cambio de tamaño
    train_examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[1, 2, 1, 2], [3, 4, 3, 4]]  # Duplicación horizontal
        }
    ]
    
    test_input = np.array([[5, 6], [7, 8]])
    
    print("\n📚 Entrenamiento:")
    print(f"  Input shape: {np.array(train_examples[0]['input']).shape}")
    print(f"  Output shape: {np.array(train_examples[0]['output']).shape}")
    print(f"  Cambio: Duplicación horizontal")
    
    print("\n🎯 Test input:")
    print(test_input)
    
    # PASO 1: MACRO
    print("\n" + "-"*40)
    print("📍 PASO 1: MACRO (Observación V-JEPA)")
    start = time.time()
    macro_understanding = network._macro_observation(train_examples)
    macro_time = time.time() - start
    
    print(f"  ⏱️ Tiempo: {macro_time*1000:.2f}ms")
    print(f"  📊 Patrones detectados: {len(macro_understanding['patterns'])}")
    print(f"  🧠 Inferencia global: {macro_understanding['global_inference'].conclusion[:50]}")
    
    # PASO 2: MESO
    print("\n" + "-"*40)
    print("📍 PASO 2: MESO (Razonamiento sobre objetos)")
    start = time.time()
    meso_logic = network._meso_reasoning(macro_understanding, train_examples)
    meso_time = time.time() - start
    
    print(f"  ⏱️ Tiempo: {meso_time*1000:.2f}ms")
    print(f"  📊 Reglas MESO: {len(meso_logic['meso_rules'])}")
    print(f"  🔗 Cadena de razonamiento: {len(meso_logic['reasoning_chain'])} pasos")
    
    # PASO 3: MICRO
    print("\n" + "-"*40)
    print("📍 PASO 3: MICRO (Ejecución píxel a píxel)")
    start = time.time()
    solution = network._micro_execution(meso_logic, test_input, train_examples)
    micro_time = time.time() - start
    
    print(f"  ⏱️ Tiempo: {micro_time*1000:.2f}ms")
    print(f"  📊 Solución shape: {solution.shape}")
    print(f"  📤 Solución:")
    print(solution)
    
    # Resumen de tiempos
    total_time = macro_time + meso_time + micro_time
    print("\n" + "-"*40)
    print("⏱️ TIEMPOS:")
    print(f"  MACRO: {macro_time*1000:.2f}ms ({macro_time/total_time*100:.1f}%)")
    print(f"  MESO:  {meso_time*1000:.2f}ms ({meso_time/total_time*100:.1f}%)")
    print(f"  MICRO: {micro_time*1000:.2f}ms ({micro_time/total_time*100:.1f}%)")
    print(f"  TOTAL: {total_time*1000:.2f}ms")
    
    return solution

def test_gpu_operations():
    """Test 4: Operaciones con GPU"""
    print("\n" + "="*60)
    print("🔬 TEST 4: Operaciones GPU")
    print("="*60)
    
    if device.type != 'cuda':
        print("⚠️ GPU no disponible, saltando test")
        return
    
    # Matrices grandes para ver diferencia
    size = 1000
    print(f"\n📊 Tamaño de matrices: {size}x{size}")
    
    # CPU timing
    np_a = np.random.rand(size, size)
    np_b = np.random.rand(size, size)
    
    start = time.time()
    np_result = np.matmul(np_a, np_b)
    cpu_time = time.time() - start
    
    print(f"\n🖥️ CPU (NumPy):")
    print(f"  Tiempo: {cpu_time*1000:.2f}ms")
    
    # GPU timing
    torch_a = torch.from_numpy(np_a).float().to(device)
    torch_b = torch.from_numpy(np_b).float().to(device)
    
    # Warmup
    _ = torch.matmul(torch_a, torch_b)
    torch.cuda.synchronize()
    
    start = time.time()
    torch_result = torch.matmul(torch_a, torch_b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"\n🚀 GPU (PyTorch):")
    print(f"  Tiempo: {gpu_time*1000:.2f}ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Operaciones específicas para ARC
    print("\n📐 Operaciones ARC en GPU:")
    
    # Convolución para detectar patrones
    grid = torch.rand(1, 1, 30, 30).to(device)
    kernel = torch.ones(1, 1, 3, 3).to(device) / 9.0
    
    start = time.time()
    result = torch.nn.functional.conv2d(grid, kernel, padding=1)
    torch.cuda.synchronize()
    conv_time = time.time() - start
    
    print(f"  Convolución 3x3: {conv_time*1000:.4f}ms")
    
    # Transformación afín
    theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float).to(device)
    grid_affine = torch.nn.functional.affine_grid(theta, (1, 1, 30, 30))
    
    start = time.time()
    result = torch.nn.functional.grid_sample(grid, grid_affine)
    torch.cuda.synchronize()
    affine_time = time.time() - start
    
    print(f"  Transformación afín: {affine_time*1000:.4f}ms")

def test_full_pipeline():
    """Test 5: Pipeline completo con mediciones"""
    print("\n" + "="*60)
    print("🔬 TEST 5: Pipeline Completo")
    print("="*60)
    
    from arc import ARCSolver
    
    # Puzzle real de ARC
    train_examples = [
        {
            'input': [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0]
            ],
            'output': [
                [0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 0, 2, 0],
                [0, 0, 2, 2, 2],
                [0, 0, 0, 2, 0]
            ]
        }
    ]
    
    test_input = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("\n📚 Puzzle: Expansión en cruz")
    print(f"  Train: {len(train_examples)} ejemplos")
    print(f"  Test shape: {test_input.shape}")
    
    solver = ARCSolver()
    
    # Medir tiempo total
    start = time.time()
    solution = solver.reason(train_examples, test_input)
    total_time = time.time() - start
    
    print(f"\n⏱️ Tiempo total: {total_time*1000:.2f}ms")
    
    print(f"\n📤 Solución:")
    print(solution)
    
    # Verificar inferencias
    if hasattr(solver, 'inferences'):
        print(f"\n🧠 Inferencias generadas: {len(solver.inferences)}")
        
        levels = {'macro': 0, 'meso': 0, 'micro': 0}
        for inf in solver.inferences:
            levels[inf.level] = levels.get(inf.level, 0) + 1
        
        print("  Distribución por nivel:")
        for level, count in levels.items():
            print(f"    {level.upper()}: {count}")
    
    # Métricas de calidad
    expected = np.array([
        [0, 0, 3, 0, 0],
        [0, 3, 3, 3, 0],
        [0, 4, 3, 0, 0],
        [4, 4, 4, 0, 0],
        [0, 4, 0, 0, 0]
    ])
    
    if solution.shape == expected.shape:
        accuracy = np.mean(solution == expected) * 100
        print(f"\n📊 Accuracy vs esperado: {accuracy:.1f}%")
    
    return solution

def main():
    print("="*80)
    print("🧪 TEST PASO A PASO - SISTEMA DE RAZONAMIENTO LÓGICO")
    print("="*80)
    
    results = {}
    
    # Test 1: V-JEPA Observer
    try:
        results['vjepa'] = test_vjepa_observer()
        print("✅ V-JEPA Observer OK")
    except Exception as e:
        print(f"❌ V-JEPA Observer: {e}")
        results['vjepa'] = None
    
    # Test 2: Reglas Emergentes
    try:
        results['rules'] = test_emergent_rules()
        print("✅ Sistema de Reglas OK")
    except Exception as e:
        print(f"❌ Sistema de Reglas: {e}")
        results['rules'] = None
    
    # Test 3: Flujo de Razonamiento
    try:
        results['flow'] = test_logical_reasoning_flow()
        print("✅ Flujo de Razonamiento OK")
    except Exception as e:
        print(f"❌ Flujo de Razonamiento: {e}")
        results['flow'] = None
    
    # Test 4: GPU
    try:
        test_gpu_operations()
        print("✅ Operaciones GPU OK")
    except Exception as e:
        print(f"❌ Operaciones GPU: {e}")
    
    # Test 5: Pipeline Completo
    try:
        results['pipeline'] = test_full_pipeline()
        print("✅ Pipeline Completo OK")
    except Exception as e:
        print(f"❌ Pipeline Completo: {e}")
        results['pipeline'] = None
    
    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    print(f"✅ Tests pasados: {passed}/{total}")
    print(f"📈 Tasa de éxito: {passed/total*100:.1f}%")
    
    if device.type == 'cuda':
        print(f"🚀 GPU utilizada: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM usado: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

if __name__ == "__main__":
    main()