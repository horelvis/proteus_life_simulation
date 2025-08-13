#!/usr/bin/env python3
"""
Test Detallado Paso a Paso - Sin GPU por ahora
Analiza cada componente del sistema individualmente
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import json

def print_section(title: str, level: int = 1):
    """Helper para imprimir secciones"""
    if level == 1:
        print("\n" + "="*60)
        print(f"🔬 {title}")
        print("="*60)
    else:
        print("\n" + "-"*40)
        print(f"📍 {title}")

def test_vjepa_observer_detailed():
    """Test detallado del V-JEPA Observer"""
    print_section("TEST 1: V-JEPA Observer (Capa MACRO)")
    
    from arc.vjepa_observer import VJEPAObserver
    
    observer = VJEPAObserver(embedding_dim=64)
    
    # Caso 1: Mapeo simple de valores
    print_section("Caso 1: Mapeo de valores", 2)
    
    input1 = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])
    output1 = np.array([[2, 0, 2], [0, 4, 0], [2, 0, 2]])
    
    print("Input:")
    print(input1)
    print("\nOutput esperado:")
    print(output1)
    
    obs1 = observer.observe(input1, output1)
    
    print(f"\n✓ Patrón detectado: {obs1['emergent_pattern']['type']}")
    print(f"✓ Confianza: {obs1['emergent_pattern']['confidence']:.2f}")
    
    # Analizar embedding
    embed = obs1['transform_embedding']
    print(f"✓ Embedding dimensión: {embed.shape[0]}")
    print(f"✓ Primeros 5 valores: {embed[:5].round(3)}")
    
    # Caso 2: Segunda observación similar
    print_section("Caso 2: Patrón similar", 2)
    
    input2 = np.array([[3, 0, 3], [0, 4, 0], [3, 0, 3]])
    output2 = np.array([[6, 0, 6], [0, 8, 0], [6, 0, 6]])
    
    obs2 = observer.observe(input2, output2)
    
    print(f"✓ Patrón detectado: {obs2['emergent_pattern']['type']}")
    print(f"✓ Confianza: {obs2['emergent_pattern']['confidence']:.2f}")
    
    # Comparar embeddings
    similarity = np.dot(embed, obs2['transform_embedding']) / (
        np.linalg.norm(embed) * np.linalg.norm(obs2['transform_embedding'])
    )
    print(f"✓ Similitud entre transformaciones: {similarity:.3f}")
    
    # Test predicción
    print_section("Predicción basada en observaciones", 2)
    
    test_input = np.array([[5, 0, 5], [0, 6, 0], [5, 0, 5]])
    print("Input de prueba:")
    print(test_input)
    
    predicted = observer.predict_transformation(test_input)
    print("\nPredicción:")
    print(predicted)
    
    # Verificar si aprendió el patrón
    expected = np.array([[10, 0, 10], [0, 12, 0], [10, 0, 10]])
    if np.array_equal(predicted, expected):
        print("✅ Patrón aprendido correctamente!")
    else:
        print(f"⚠️ Predicción diferente de lo esperado")
        print(f"   Esperado: {expected[0,0]}, Obtenido: {predicted[0,0]}")
    
    return observer

def test_emergent_rules_detailed():
    """Test detallado del sistema de reglas emergentes"""
    print_section("TEST 2: Sistema de Reglas Emergentes")
    
    from arc.emergent_rule_system import EmergentRuleSystem
    
    rule_system = EmergentRuleSystem()
    
    # Crear ejemplos con patrones claros
    train_examples = [
        {
            'input': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'output': [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        },
        {
            'input': [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            'output': [[6, 0, 0], [0, 6, 0], [0, 0, 6]]
        }
    ]
    
    print("Ejemplos de entrenamiento:")
    for i, ex in enumerate(train_examples):
        print(f"  Ejemplo {i+1}: valores {np.unique(ex['input'])} → {np.unique(ex['output'])}")
    
    # Extraer reglas
    start_time = time.time()
    rules = rule_system.extract_rules_from_examples(train_examples)
    extraction_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo de extracción: {extraction_time*1000:.2f}ms")
    
    # Analizar reglas MICRO
    print_section("Reglas MICRO (nivel píxel)", 2)
    
    micro_rules = rules['micro_rules']
    print(f"Total: {len(micro_rules)} reglas")
    
    pattern_counts = {}
    for rule in micro_rules:
        pattern_counts[rule.pattern] = pattern_counts.get(rule.pattern, 0) + 1
    
    print("Patrones detectados:")
    for pattern, count in pattern_counts.items():
        print(f"  - {pattern}: {count} veces")
    
    # Mostrar reglas con mayor confianza
    top_micro = sorted(micro_rules, key=lambda r: r.confidence, reverse=True)[:3]
    print("\nTop 3 reglas MICRO:")
    for rule in top_micro:
        print(f"  - {rule.pattern} (conf: {rule.confidence:.2f}, support: {rule.support})")
        if rule.condition:
            print(f"    Condición: {rule.condition}")
        if rule.action:
            print(f"    Acción: {rule.action}")
    
    # Analizar reglas MESO
    print_section("Reglas MESO (nivel objeto)", 2)
    
    meso_rules = rules['meso_rules']
    print(f"Total: {len(meso_rules)} reglas")
    
    for rule in meso_rules[:3]:
        print(f"  - Transformación: {rule.transformation}")
        print(f"    Confianza: {rule.confidence:.2f}")
        if rule.parameters:
            print(f"    Parámetros: {rule.parameters}")
    
    # Analizar reglas MACRO
    print_section("Reglas MACRO (nivel patrón)", 2)
    
    macro_rules = rules['macro_rules']
    print(f"Total: {len(macro_rules)} reglas")
    
    for rule in macro_rules[:3]:
        print(f"  - Tipo: {rule.rule_type}")
        print(f"    Transformación global: {rule.global_transform}")
        print(f"    Confianza: {rule.confidence:.2f}")
    
    return rule_system

def test_reasoning_network_detailed():
    """Test detallado de la red de razonamiento"""
    print_section("TEST 3: Red de Razonamiento Lógico")
    
    from arc.logical_reasoning_network import LogicalReasoningNetwork
    
    network = LogicalReasoningNetwork()
    
    # Ejemplo: expansión en cruz
    train_examples = [
        {
            'input': [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            'output': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        }
    ]
    
    test_input = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    
    print("Patrón: Expansión en cruz")
    print("Input de entrenamiento:")
    print(np.array(train_examples[0]['input']))
    print("\nOutput de entrenamiento:")
    print(np.array(train_examples[0]['output']))
    
    # PASO 1: MACRO
    print_section("PASO 1: Observación MACRO", 2)
    
    start = time.time()
    macro_result = network._macro_observation(train_examples)
    macro_time = time.time() - start
    
    print(f"⏱️ Tiempo: {macro_time*1000:.2f}ms")
    print(f"📊 Patrones detectados: {len(macro_result['patterns'])}")
    
    global_inf = macro_result['global_inference']
    print(f"\n🧠 Inferencia global:")
    print(f"  Premisa: {global_inf.premise}")
    print(f"  Conclusión: {global_inf.conclusion}")
    print(f"  Confianza: {global_inf.confidence:.2f}")
    
    # Analizar espacio de transformación
    transform_space = macro_result['transformation_space']
    print(f"\n📐 Espacio de transformación:")
    print(f"  Dimensión: {transform_space.shape}")
    print(f"  Norma: {np.linalg.norm(transform_space):.2f}")
    
    # PASO 2: MESO
    print_section("PASO 2: Razonamiento MESO", 2)
    
    start = time.time()
    meso_result = network._meso_reasoning(macro_result, train_examples)
    meso_time = time.time() - start
    
    print(f"⏱️ Tiempo: {meso_time*1000:.2f}ms")
    print(f"📊 Reglas MESO: {len(meso_result['meso_rules'])}")
    
    # Mostrar cadena de razonamiento
    print("\n🔗 Cadena de razonamiento:")
    for i, step in enumerate(meso_result['reasoning_chain'][:3]):
        print(f"  Paso {i+1}: {step['action']} (conf: {step.get('confidence', 0):.2f})")
    
    # PASO 3: MICRO
    print_section("PASO 3: Ejecución MICRO", 2)
    
    print("Input de prueba:")
    print(test_input)
    
    start = time.time()
    solution = network._micro_execution(meso_result, test_input, train_examples)
    micro_time = time.time() - start
    
    print(f"\n⏱️ Tiempo: {micro_time*1000:.2f}ms")
    print("\nSolución:")
    print(solution)
    
    # Verificar si aplicó el patrón
    expected = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
    if np.array_equal(solution, expected):
        print("✅ Patrón aplicado correctamente!")
    else:
        accuracy = np.mean(solution == expected) * 100
        print(f"⚠️ Accuracy: {accuracy:.1f}%")
    
    # Resumen de tiempos
    total = macro_time + meso_time + micro_time
    print_section("Resumen de tiempos", 2)
    print(f"MACRO: {macro_time*1000:.2f}ms ({macro_time/total*100:.1f}%)")
    print(f"MESO:  {meso_time*1000:.2f}ms ({meso_time/total*100:.1f}%)")
    print(f"MICRO: {micro_time*1000:.2f}ms ({micro_time/total*100:.1f}%)")
    print(f"TOTAL: {total*1000:.2f}ms")
    
    # Guardar inferencias
    network.inferences.extend([global_inf])
    
    return network

def test_integration():
    """Test de integración completa"""
    print_section("TEST 4: Integración Completa")
    
    from arc import ARCSolver
    
    # Puzzle más complejo
    train = [
        {
            'input': [
                [1, 0, 0, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 0, 0, 4]
            ],
            'output': [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4]
            ]
        }
    ]
    
    test_input = np.array([
        [5, 0, 0, 6],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7, 0, 0, 8]
    ])
    
    print("Puzzle: Expansión de esquinas a cuadrantes")
    print("\nInput de prueba:")
    print(test_input)
    
    solver = ARCSolver()
    
    start = time.time()
    solution = solver.reason(train, test_input)
    total_time = time.time() - start
    
    print(f"\n⏱️ Tiempo total: {total_time*1000:.2f}ms")
    
    print("\nSolución:")
    print(solution)
    
    # Verificar inferencias
    if hasattr(solver, 'inferences'):
        print(f"\n🧠 Total inferencias: {len(solver.inferences)}")
        
        # Contar por nivel
        levels = {}
        for inf in solver.inferences:
            levels[inf.level] = levels.get(inf.level, 0) + 1
        
        print("Distribución:")
        for level, count in sorted(levels.items()):
            print(f"  {level.upper()}: {count}")
        
        # Mostrar algunas inferencias
        print("\nInferencias principales:")
        for inf in solver.inferences[:3]:
            print(f"  [{inf.level}] {inf.conclusion[:50]}...")
    
    return solution

def main():
    print("="*80)
    print("🧪 TEST DETALLADO PASO A PASO")
    print("Sistema de Razonamiento Lógico (sin GPU por ahora)")
    print("="*80)
    
    results = {}
    
    # Test 1: V-JEPA Observer
    try:
        observer = test_vjepa_observer_detailed()
        results['vjepa'] = "✅ OK"
        print("\n✅ V-JEPA Observer completado")
    except Exception as e:
        results['vjepa'] = f"❌ Error: {str(e)[:50]}"
        print(f"\n❌ V-JEPA Observer falló: {e}")
    
    # Test 2: Sistema de Reglas
    try:
        rules = test_emergent_rules_detailed()
        results['rules'] = "✅ OK"
        print("\n✅ Sistema de Reglas completado")
    except Exception as e:
        results['rules'] = f"❌ Error: {str(e)[:50]}"
        print(f"\n❌ Sistema de Reglas falló: {e}")
    
    # Test 3: Red de Razonamiento
    try:
        network = test_reasoning_network_detailed()
        results['network'] = "✅ OK"
        print("\n✅ Red de Razonamiento completada")
    except Exception as e:
        results['network'] = f"❌ Error: {str(e)[:50]}"
        print(f"\n❌ Red de Razonamiento falló: {e}")
    
    # Test 4: Integración
    try:
        solution = test_integration()
        results['integration'] = "✅ OK"
        print("\n✅ Integración completada")
    except Exception as e:
        results['integration'] = f"❌ Error: {str(e)[:50]}"
        print(f"\n❌ Integración falló: {e}")
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    for test, result in results.items():
        print(f"{test:15} : {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print(f"\nTests pasados: {passed}/{total}")
    print(f"Tasa de éxito: {passed/total*100:.1f}%")
    
    # Recomendaciones
    if passed < total:
        print("\n⚠️ Recomendaciones:")
        print("  1. Revisar funciones que devuelven valores por defecto")
        print("  2. Eliminar condicionales innecesarios")
        print("  3. Implementar operaciones con GPU cuando esté disponible")

if __name__ == "__main__":
    main()