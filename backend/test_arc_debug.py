#!/usr/bin/env python3
"""
Test de Debug para ARC - Análisis detallado de fallos
Examina cada capa del sistema con logs exhaustivos
"""

import numpy as np
import torch
import json
import time
import logging
from typing import Dict, List, Tuple
from pathlib import Path

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_official_puzzle(puzzle_id: str = "00d62c1b") -> Dict:
    """Carga un puzzle oficial de ARC"""
    cache_dir = Path("/app/arc_official_cache")
    puzzle_file = cache_dir / f"arc_agi_1_training_{puzzle_id}.json"
    
    if puzzle_file.exists():
        with open(puzzle_file, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Puzzle {puzzle_id} no encontrado en cache")
        # Puzzle de ejemplo si no existe el archivo
        return {
            "train": [
                {
                    "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    "output": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                }
            ],
            "test": [
                {
                    "input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                    "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
                }
            ]
        }

def analyze_puzzle_structure(puzzle: Dict) -> None:
    """Analiza la estructura del puzzle"""
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE ESTRUCTURA DEL PUZZLE")
    print("="*80)
    
    train_examples = puzzle.get("train", [])
    test_examples = puzzle.get("test", [])
    
    print(f"\n📚 Ejemplos de entrenamiento: {len(train_examples)}")
    for i, example in enumerate(train_examples):
        input_shape = np.array(example['input']).shape
        output_shape = np.array(example['output']).shape
        print(f"   Ejemplo {i+1}: Input {input_shape} → Output {output_shape}")
        
        # Analizar cambio de tamaño
        if input_shape != output_shape:
            print(f"      ⚠️ Cambio de tamaño detectado!")
            print(f"      Factor H: {output_shape[0]/input_shape[0]:.1f}x")
            print(f"      Factor W: {output_shape[1]/input_shape[1]:.1f}x")
        
        # Analizar valores únicos
        input_values = np.unique(example['input'])
        output_values = np.unique(example['output'])
        print(f"      Valores input: {input_values}")
        print(f"      Valores output: {output_values}")
        
        # Detectar nuevos colores
        new_colors = set(output_values) - set(input_values)
        if new_colors:
            print(f"      🎨 Nuevos colores en output: {new_colors}")
    
    print(f"\n🎯 Ejemplos de test: {len(test_examples)}")
    for i, example in enumerate(test_examples):
        input_shape = np.array(example['input']).shape
        print(f"   Test {i+1}: Input {input_shape}")

def test_vjepa_layer(train_examples: List[Dict]) -> Dict:
    """Test detallado de la capa V-JEPA"""
    print("\n" + "="*80)
    print("🔬 CAPA 1: V-JEPA OBSERVER (MACRO)")
    print("="*80)
    
    from arc.vjepa_observer import VJEPAObserver
    
    observer = VJEPAObserver(embedding_dim=128)
    observations = []
    
    for i, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"\n📍 Observando ejemplo {i+1}:")
        print(f"   Input shape: {input_grid.shape}")
        print(f"   Output shape: {output_grid.shape}")
        
        # Log detallado de la observación
        logger.debug(f"Input grid:\n{input_grid}")
        logger.debug(f"Output grid:\n{output_grid}")
        
        start = time.time()
        obs = observer.observe(input_grid, output_grid)
        elapsed = time.time() - start
        
        observations.append(obs)
        
        print(f"   ⏱️ Tiempo: {elapsed*1000:.2f}ms")
        print(f"   📊 Patrón emergente: {obs['emergent_pattern']['type']}")
        print(f"   📊 Confianza: {obs['emergent_pattern']['confidence']:.3f}")
        
        # Analizar embedding
        embedding = obs['transform_embedding']
        print(f"   📊 Embedding stats:")
        print(f"      - Dimensión: {embedding.shape}")
        print(f"      - Media: {np.mean(embedding):.3f}")
        print(f"      - Std: {np.std(embedding):.3f}")
        print(f"      - Min/Max: [{np.min(embedding):.3f}, {np.max(embedding):.3f}]")
        
        # Valores más significativos del embedding
        top_indices = np.argsort(np.abs(embedding))[-5:]
        print(f"   📊 Top 5 componentes del embedding:")
        for idx in top_indices:
            print(f"      - Componente {idx}: {embedding[idx]:.3f}")
    
    # Analizar similitudes entre observaciones
    if len(observations) > 1:
        print("\n📊 Análisis de similitud entre observaciones:")
        for i in range(len(observations)):
            for j in range(i+1, len(observations)):
                emb1 = observations[i]['transform_embedding']
                emb2 = observations[j]['transform_embedding']
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                print(f"   Similitud ejemplo {i+1} ↔ {j+1}: {similarity:.3f}")
    
    return {
        'observer': observer,
        'observations': observations
    }

def test_emergent_rules_layer(train_examples: List[Dict]) -> Dict:
    """Test detallado del sistema de reglas emergentes"""
    print("\n" + "="*80)
    print("🔬 CAPA 2: SISTEMA DE REGLAS EMERGENTES (MESO)")
    print("="*80)
    
    from arc.emergent_rule_system import EmergentRuleSystem
    
    rule_system = EmergentRuleSystem()
    
    print("\n📍 Extrayendo reglas de los ejemplos...")
    start = time.time()
    rules = rule_system.extract_rules_from_examples(train_examples)
    elapsed = time.time() - start
    
    print(f"⏱️ Tiempo de extracción: {elapsed*1000:.2f}ms")
    
    # Analizar reglas MICRO
    print(f"\n📊 REGLAS MICRO (nivel píxel): {len(rules['micro_rules'])}")
    pattern_counts = {}
    for rule in rules['micro_rules']:
        pattern_counts[rule.pattern] = pattern_counts.get(rule.pattern, 0) + 1
    
    print("   Distribución de patrones:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(rules['micro_rules']) * 100
        print(f"      - {pattern}: {count} ({percentage:.1f}%)")
    
    # Top reglas micro por confianza
    print("\n   Top 5 reglas MICRO por confianza:")
    for rule in sorted(rules['micro_rules'], key=lambda r: -r.confidence)[:5]:
        print(f"      - {rule.pattern}: conf={rule.confidence:.3f}, support={rule.support}")
        if rule.condition:
            logger.debug(f"        Condición: {rule.condition}")
        if rule.action:
            logger.debug(f"        Acción: {rule.action}")
    
    # Analizar reglas MESO
    print(f"\n📊 REGLAS MESO (nivel objeto): {len(rules['meso_rules'])}")
    if rules['meso_rules']:
        print("   Transformaciones detectadas:")
        transform_counts = {}
        for rule in rules['meso_rules']:
            transform_counts[rule.transformation] = transform_counts.get(rule.transformation, 0) + 1
        
        for transform, count in sorted(transform_counts.items(), key=lambda x: -x[1]):
            print(f"      - {transform}: {count}")
        
        print("\n   Top reglas MESO:")
        for rule in sorted(rules['meso_rules'], key=lambda r: -r.confidence)[:3]:
            print(f"      - {rule.transformation}: conf={rule.confidence:.3f}")
            print(f"        Source shape: {rule.source_shape}")
            print(f"        Target shape: {rule.target_shape}")
            if rule.parameters:
                print(f"        Parámetros: {rule.parameters}")
    
    # Analizar reglas MACRO
    print(f"\n📊 REGLAS MACRO (nivel patrón): {len(rules['macro_rules'])}")
    if rules['macro_rules']:
        print("   Patrones globales detectados:")
        for rule in rules['macro_rules']:
            print(f"      - {rule.rule_type}: {rule.global_transform}")
            print(f"        Confianza: {rule.confidence:.3f}")
            if rule.meso_rules:
                print(f"        Basado en {len(rule.meso_rules)} reglas MESO")
    
    return {
        'rule_system': rule_system,
        'rules': rules
    }

def test_logical_reasoning(train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
    """Test del sistema completo de razonamiento lógico"""
    print("\n" + "="*80)
    print("🔬 CAPA 3: RED DE RAZONAMIENTO LÓGICO COMPLETA")
    print("="*80)
    
    from arc.logical_reasoning_network import LogicalReasoningNetwork
    
    network = LogicalReasoningNetwork()
    
    print("\n📍 Ejecutando razonamiento completo...")
    
    # PASO 1: MACRO
    print("\n" + "-"*60)
    print("PASO 1: Observación MACRO (V-JEPA)")
    start = time.time()
    macro_understanding = network._macro_observation(train_examples)
    macro_time = time.time() - start
    
    print(f"⏱️ Tiempo: {macro_time*1000:.2f}ms")
    print(f"📊 Patrones detectados: {len(macro_understanding['patterns'])}")
    
    # Detalles de la inferencia global
    global_inf = macro_understanding['global_inference']
    print(f"\n🧠 Inferencia Global:")
    print(f"   Nivel: {global_inf.level}")
    print(f"   Premisa: {global_inf.premise}")
    print(f"   Conclusión: {global_inf.conclusion}")
    print(f"   Confianza: {global_inf.confidence:.3f}")
    
    # Analizar espacio de transformación
    transform_space = macro_understanding['transformation_space']
    print(f"\n📐 Espacio de transformación:")
    print(f"   Shape: {transform_space.shape}")
    print(f"   Norma L2: {np.linalg.norm(transform_space):.3f}")
    print(f"   Valores no-cero: {np.count_nonzero(transform_space)}")
    
    # PASO 2: MESO
    print("\n" + "-"*60)
    print("PASO 2: Razonamiento MESO (Objetos)")
    start = time.time()
    meso_logic = network._meso_reasoning(macro_understanding, train_examples)
    meso_time = time.time() - start
    
    print(f"⏱️ Tiempo: {meso_time*1000:.2f}ms")
    print(f"📊 Reglas MESO activas: {len(meso_logic['meso_rules'])}")
    
    # Cadena de razonamiento
    print(f"\n🔗 Cadena de razonamiento: {len(meso_logic['reasoning_chain'])} pasos")
    for i, step in enumerate(meso_logic['reasoning_chain'][:5]):  # Primeros 5 pasos
        print(f"   Paso {i+1}:")
        print(f"      Acción: {step.get('action', 'unknown')}")
        print(f"      Confianza: {step.get('confidence', 0):.3f}")
        if 'details' in step:
            logger.debug(f"      Detalles: {step['details']}")
    
    # PASO 3: MICRO
    print("\n" + "-"*60)
    print("PASO 3: Ejecución MICRO (Píxeles)")
    print(f"\n📥 Input de prueba:")
    print(f"   Shape: {test_input.shape}")
    print(f"   Valores únicos: {np.unique(test_input)}")
    logger.debug(f"Test input:\n{test_input}")
    
    start = time.time()
    solution = network._micro_execution(meso_logic, test_input, train_examples)
    micro_time = time.time() - start
    
    print(f"\n⏱️ Tiempo: {micro_time*1000:.2f}ms")
    print(f"📤 Solución generada:")
    print(f"   Shape: {solution.shape}")
    print(f"   Valores únicos: {np.unique(solution)}")
    logger.debug(f"Solution:\n{solution}")
    
    # Resumen de tiempos
    total_time = macro_time + meso_time + micro_time
    print("\n" + "-"*60)
    print("⏱️ ANÁLISIS DE RENDIMIENTO:")
    print(f"   MACRO: {macro_time*1000:.2f}ms ({macro_time/total_time*100:.1f}%)")
    print(f"   MESO:  {meso_time*1000:.2f}ms ({meso_time/total_time*100:.1f}%)")
    print(f"   MICRO: {micro_time*1000:.2f}ms ({micro_time/total_time*100:.1f}%)")
    print(f"   TOTAL: {total_time*1000:.2f}ms")
    
    # Guardar inferencias
    print(f"\n🧠 Total inferencias generadas: {len(network.inferences)}")
    for inf in network.inferences[:3]:
        print(f"   [{inf.level}] {inf.conclusion[:80]}...")
    
    return solution

def evaluate_solution(solution: np.ndarray, expected: np.ndarray) -> Dict:
    """Evalúa la solución comparando con el resultado esperado"""
    print("\n" + "="*80)
    print("📊 EVALUACIÓN DE LA SOLUCIÓN")
    print("="*80)
    
    # Verificar dimensiones
    print(f"\n📐 Dimensiones:")
    print(f"   Solución: {solution.shape}")
    print(f"   Esperado: {expected.shape}")
    
    if solution.shape != expected.shape:
        print("   ❌ Las dimensiones no coinciden!")
        # Intentar ajustar
        if solution.size == expected.size:
            print("   ⚠️ Mismo número de elementos, intentando reshape...")
            solution = solution.reshape(expected.shape)
        else:
            print("   ❌ No se puede ajustar automáticamente")
            return {'accuracy': 0.0, 'details': 'shape_mismatch'}
    
    # Calcular accuracy
    correct = np.sum(solution == expected)
    total = expected.size
    accuracy = correct / total * 100
    
    print(f"\n✅ Píxeles correctos: {correct}/{total} ({accuracy:.1f}%)")
    
    # Análisis por valor
    print("\n📊 Análisis por valor:")
    for value in np.unique(expected):
        mask = expected == value
        correct_for_value = np.sum(solution[mask] == expected[mask])
        total_for_value = np.sum(mask)
        acc_for_value = correct_for_value / total_for_value * 100 if total_for_value > 0 else 0
        print(f"   Valor {value}: {correct_for_value}/{total_for_value} ({acc_for_value:.1f}%)")
    
    # Análisis de errores
    if accuracy < 100:
        print("\n❌ Análisis de errores:")
        errors = solution != expected
        error_positions = np.argwhere(errors)
        
        print(f"   Total errores: {len(error_positions)}")
        
        # Mostrar primeros errores
        for i, (y, x) in enumerate(error_positions[:5]):
            print(f"   Error {i+1} en posición ({y},{x}):")
            print(f"      Predicho: {solution[y,x]}")
            print(f"      Esperado: {expected[y,x]}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'shape_match': solution.shape == expected.shape
    }

def main():
    """Test principal de debug"""
    print("="*80)
    print("🧪 TEST DE DEBUG PARA PUZZLE ARC")
    print("Sistema de Razonamiento Lógico con Logs Detallados")
    print("="*80)
    
    # Cargar puzzle oficial
    puzzle_id = "0520fde7"  # Puzzle real del cache
    print(f"\n📂 Cargando puzzle oficial: {puzzle_id}")
    
    puzzle = load_official_puzzle(puzzle_id)
    
    # Analizar estructura
    analyze_puzzle_structure(puzzle)
    
    # Preparar datos
    train_examples = puzzle['train']
    test_examples = puzzle['test']
    
    if not test_examples:
        print("⚠️ No hay ejemplos de test, usando el primer ejemplo de entrenamiento")
        test_examples = [train_examples[0]]
    
    test_input = np.array(test_examples[0]['input'])
    test_expected = np.array(test_examples[0].get('output', test_input))
    
    print(f"\n🎯 Resolviendo puzzle con {len(train_examples)} ejemplos de entrenamiento")
    
    # Test capa por capa
    vjepa_results = test_vjepa_layer(train_examples)
    rules_results = test_emergent_rules_layer(train_examples)
    
    # Test sistema completo
    solution = test_logical_reasoning(train_examples, test_input)
    
    # Evaluar solución
    evaluation = evaluate_solution(solution, test_expected)
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    print(f"\n🎯 Puzzle: {puzzle_id}")
    print(f"📚 Ejemplos de entrenamiento: {len(train_examples)}")
    print(f"✅ Accuracy: {evaluation['accuracy']:.1f}%")
    
    if evaluation['accuracy'] < 100:
        print("\n⚠️ PUNTOS DE MEJORA IDENTIFICADOS:")
        
        # Analizar problemas en V-JEPA
        if all(obs['emergent_pattern']['type'] == 'novel' for obs in vjepa_results['observations']):
            print("   1. V-JEPA no encuentra patrones repetidos")
            print("      → Ajustar embedding_dim o mejorar extracción de features")
        
        # Analizar problemas en reglas
        if len(rules_results['rules']['macro_rules']) == 0:
            print("   2. No se detectan patrones globales (MACRO)")
            print("      → Mejorar síntesis de reglas macro desde meso")
        
        if len(rules_results['rules']['meso_rules']) < 2:
            print("   3. Pocas reglas MESO detectadas")
            print("      → Mejorar detección de objetos y transformaciones")
        
        # Analizar cambios de tamaño
        for ex in train_examples:
            if np.array(ex['input']).shape != np.array(ex['output']).shape:
                print("   4. Hay cambios de tamaño en el puzzle")
                print("      → Verificar manejo de resize en emergent_rule_system")
                break
        
        print("\n💡 Recomendaciones:")
        print("   - Aumentar logging en funciones críticas")
        print("   - Verificar manejo de casos edge")
        print("   - Considerar agregar más heurísticas específicas de ARC")
    
    return evaluation['accuracy']

if __name__ == "__main__":
    accuracy = main()
    exit(0 if accuracy == 100 else 1)