#!/usr/bin/env python3
"""
Test del Sistema de Reglas Emergentes
Las reglas surgen desde el nivel de píxeles
"""

import numpy as np
from arc.emergent_rule_system import EmergentRuleSystem

def test_emergent_rules():
    """Prueba el sistema de reglas emergentes"""
    
    print("="*70)
    print("🌱 SISTEMA DE REGLAS EMERGENTES")
    print("Las reglas surgen desde la base (píxeles)")
    print("="*70)
    
    # Ejemplo 1: Expansión de cruz
    print("\n🎯 Test 1: Expansión de Cruz")
    print("-"*50)
    
    train_examples = [
        {
            "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        },
        {
            "input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
        }
    ]
    
    system = EmergentRuleSystem()
    rules = system.extract_rules_from_examples(train_examples)
    
    print("\n📐 NIVEL 0 - MICRO-REGLAS (Píxeles):")
    print(f"  • Total extraídas: {len(rules['micro_rules'])}")
    for i, rule in enumerate(rules['micro_rules'][:3]):
        print(f"  {i+1}. Patrón: {rule.pattern}")
        print(f"     Condición: {rule.condition}")
        print(f"     Acción: {rule.action}")
        print(f"     Confianza: {rule.confidence:.2%} (soporte: {rule.support})")
    
    print("\n🔲 NIVEL 1 - MESO-REGLAS (Objetos):")
    print(f"  • Total construidas: {len(rules['meso_rules'])}")
    for i, rule in enumerate(rules['meso_rules'][:3]):
        print(f"  {i+1}. Transformación: {rule.transformation}")
        print(f"     Forma origen: {rule.source_shape} → {rule.target_shape}")
        print(f"     Confianza: {rule.confidence:.2%}")
        if rule.micro_rules:
            print(f"     Basada en {len(rule.micro_rules)} micro-reglas")
    
    print("\n🌐 NIVEL 2 - MACRO-REGLAS (Patrones):")
    print(f"  • Total sintetizadas: {len(rules['macro_rules'])}")
    for i, rule in enumerate(rules['macro_rules']):
        print(f"  {i+1}. Tipo: {rule.rule_type}")
        print(f"     Transformación global: {rule.global_transform}")
        print(f"     Confianza: {rule.confidence:.2%}")
        if rule.meso_rules:
            print(f"     Compuesta por {len(rule.meso_rules)} meso-reglas")
    
    # Mostrar jerarquía
    hierarchy = rules['rule_hierarchy']
    print("\n🔄 JERARQUÍA DE EMERGENCIA:")
    print("  Micro → Meso:")
    for micro_pattern, meso_transforms in hierarchy['micro_to_meso'].items():
        print(f"    • {micro_pattern} → {meso_transforms}")
    
    print("  Meso → Macro:")
    for meso_transform, macro_types in hierarchy['meso_to_macro'].items():
        print(f"    • {meso_transform} → {macro_types}")
    
    if hierarchy['emergence_path']:
        print("\n  📈 Camino de emergencia:")
        for path in hierarchy['emergence_path']:
            print(f"    {path}")
    
    # Aplicar al test
    test_input = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    print("\n🧪 Aplicando reglas emergentes al test:")
    print(f"Input:\n{test_input}")
    
    result = system.apply_emergent_rules(test_input)
    print(f"\nResultado:\n{result}")
    
    expected = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]])
    print(f"\nEsperado:\n{expected}")
    
    correct = np.array_equal(result, expected)
    print(f"\n¿Correcto? {'✅ SÍ' if correct else '❌ NO'}")
    
    return correct

def test_fill_pattern():
    """Test con patrón de relleno"""
    
    print("\n" + "="*70)
    print("🎯 Test 2: Relleno de Forma")
    print("-"*50)
    
    train_examples = [
        {
            "input": [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        },
        {
            "input": [[2, 2, 0], [2, 0, 0], [2, 2, 2]],
            "output": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        }
    ]
    
    system = EmergentRuleSystem()
    rules = system.extract_rules_from_examples(train_examples)
    
    print("\n📊 Análisis de emergencia:")
    
    # Contar patrones en micro-reglas
    pattern_counts = {}
    for rule in rules['micro_rules']:
        pattern_counts[rule.pattern] = pattern_counts.get(rule.pattern, 0) + 1
    
    print("\n  Distribución de micro-patrones:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"    • {pattern}: {count}")
    
    # Ver transformaciones detectadas
    print("\n  Transformaciones de objetos:")
    for rule in rules['meso_rules']:
        print(f"    • {rule.transformation} (confianza: {rule.confidence:.2%})")
    
    # Ver reglas globales
    print("\n  Reglas globales emergentes:")
    for rule in rules['macro_rules']:
        print(f"    • {rule.rule_type}: {rule.global_transform}")
    
    # Test
    test_input = np.array([[3, 0, 3], [3, 0, 3], [3, 3, 3]])
    print(f"\n🧪 Test input:\n{test_input}")
    
    result = system.apply_emergent_rules(test_input)
    print(f"\nResultado:\n{result}")
    
    expected = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    correct = np.array_equal(result, expected)
    print(f"\n¿Correcto? {'✅ SÍ' if correct else '❌ NO'}")
    
    return correct

def analyze_rule_emergence():
    """Analiza cómo emergen las reglas desde la base"""
    
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DE EMERGENCIA DE REGLAS")
    print("="*70)
    
    # Ejemplo simple para ver la emergencia
    train_examples = [
        {
            "input": [[1, 0], [0, 0]],
            "output": [[1, 1], [1, 0]]
        }
    ]
    
    system = EmergentRuleSystem()
    rules = system.extract_rules_from_examples(train_examples)
    
    print("\n🌱 EMERGENCIA BOTTOM-UP:")
    print("\n1️⃣ PÍXELES (Base):")
    for rule in rules['micro_rules'][:5]:
        print(f"   Píxel con patrón '{rule.pattern}'")
        print(f"   → Condición: {rule.condition}")
        print(f"   → Genera: {rule.action}")
        print()
    
    print("2️⃣ OBJETOS (Emergente de píxeles):")
    for rule in rules['meso_rules']:
        print(f"   Objeto con transformación '{rule.transformation}'")
        if rule.micro_rules:
            patterns = [mr.pattern for mr in rule.micro_rules]
            print(f"   ← Emerge de patrones: {set(patterns)}")
        print()
    
    print("3️⃣ PATRONES GLOBALES (Emergente de objetos):")
    for rule in rules['macro_rules']:
        print(f"   Patrón global '{rule.rule_type}'")
        if rule.meso_rules:
            transforms = [mr.transformation for mr in rule.meso_rules]
            print(f"   ← Emerge de transformaciones: {set(transforms)}")
        print()
    
    print("✨ Las reglas EMERGEN desde los píxeles individuales")
    print("   construyendo patrones cada vez más complejos")

def main():
    print("="*70)
    print("🚀 TEST DEL SISTEMA DE REGLAS EMERGENTES")
    print("="*70)
    
    results = []
    
    # Test 1: Expansión
    result1 = test_emergent_rules()
    results.append(("Expansión", result1))
    
    # Test 2: Relleno
    result2 = test_fill_pattern()
    results.append(("Relleno", result2))
    
    # Análisis de emergencia
    analyze_rule_emergence()
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    correct = sum(1 for _, r in results if r)
    print(f"\n✅ Tests correctos: {correct}/{len(results)}")
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  • {name}: {status}")
    
    print("\n💡 CONCLUSIÓN:")
    if correct > 0:
        print("✨ Las reglas SÍ emergen desde la base (píxeles)")
        print("   y se construyen hacia arriba formando patrones complejos")
    else:
        print("⚠️ Necesita ajustes en la aplicación de reglas emergentes")

if __name__ == "__main__":
    main()