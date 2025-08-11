#!/usr/bin/env python3
"""
Debug: Rastrear qué hace exactamente las transformaciones
"""

import numpy as np
from arc import HybridProteusARCSolver
from arc.arc_solver_python import ARCSolverPython, TransformationType
from arc.emergent_rule_system import EmergentRuleSystem

def debug_transformation_chain():
    """Rastrea toda la cadena de transformaciones"""
    
    print("="*70)
    print("🔍 RASTREO DE TRANSFORMACIONES")
    print("="*70)
    
    # Caso simple: expandir cruz
    train_examples = [
        {
            "input": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "output": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        }
    ]
    
    test_input = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    expected = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]])
    
    print("📊 Input:")
    print(test_input)
    print("\n✅ Esperado:")
    print(expected)
    
    print("\n" + "="*70)
    print("1️⃣ CADENA EN ARCSolverPython")
    print("="*70)
    
    solver = ARCSolverPython()
    
    # Ver qué regla detecta
    print("\n🔍 Detección de regla:")
    rule = solver.detect_rule(train_examples[0]['input'], train_examples[0]['output'])
    print(f"Regla detectada: {rule}")
    
    if rule:
        print(f"\n📝 Aplicando regla '{rule.get('type', 'Unknown')}'...")
        print(f"Parámetros: {rule.get('parameters', {})}")
        
        # Aplicar la regla
        result = solver.apply_rule(rule, test_input)
        print(f"\n📤 Resultado:")
        print(result)
        print(f"¿Correcto? {'✅' if np.array_equal(result, expected) else '❌'}")
    
    # Probar cada tipo de transformación directamente
    print("\n" + "="*70)
    print("2️⃣ PRUEBA DIRECTA DE TRANSFORMACIONES")
    print("="*70)
    
    # Mapear las funciones de transformación
    transformations = {
        'pattern_replication': solver._apply_pattern_replication,
        'fill_shape': solver._apply_fill_shape,
        'color_mapping': solver._apply_color_mapping,
        'rotation': solver._apply_rotation,
        'reflection': solver._apply_reflection,
        'gravity': solver._apply_gravity,
        'line_drawing': solver._apply_line_drawing,
    }
    
    print("\n🧪 Probando cada transformación:")
    for name, func in transformations.items():
        print(f"\n• {name}:")
        try:
            # Diferentes parámetros según la función
            if name == 'pattern_replication':
                result = func(test_input, 3)  # factor 3
            elif name == 'fill_shape':
                result = func(test_input, 3)  # color 3
            elif name == 'color_mapping':
                result = func(test_input, {0: 0, 3: 3})  # mapeo identidad
            elif name == 'rotation':
                result = func(test_input, 90)
            elif name == 'reflection':
                result = func(test_input, 'horizontal')
            elif name == 'gravity':
                result = func(test_input, 'down')
            elif name == 'line_drawing':
                result = func(test_input)
            else:
                result = func(test_input)
            
            print(f"  Resultado:\n{result}")
            if np.array_equal(result, expected):
                print(f"  ✅ ¡ESTA TRANSFORMACIÓN PRODUCE EL RESULTADO CORRECTO!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("3️⃣ CADENA EN EmergentRuleSystem")
    print("="*70)
    
    emergent = EmergentRuleSystem()
    rules = emergent.extract_rules_from_examples(train_examples)
    
    print("\n📊 Reglas emergentes extraídas:")
    print(f"• Micro-reglas: {len(rules['micro_rules'])}")
    print(f"• Meso-reglas: {len(rules['meso_rules'])}")
    print(f"• Macro-reglas: {len(rules['macro_rules'])}")
    
    # Ver qué hace apply_emergent_rules paso a paso
    print("\n🔍 Aplicando reglas emergentes...")
    
    # Rastrear macro-reglas
    if emergent.macro_rules:
        best_macro = max(emergent.macro_rules, key=lambda r: r.confidence)
        print(f"\nMacro-regla seleccionada: {best_macro.rule_type}")
        print(f"Confianza: {best_macro.confidence:.2%}")
        print(f"Transformación global: {best_macro.global_transform}")
    
    # Rastrear meso-reglas
    if emergent.meso_rules:
        print("\nMeso-reglas a aplicar:")
        for meso in emergent.meso_rules[:3]:
            print(f"  • {meso.transformation} (confianza: {meso.confidence:.2%})")
    
    # Aplicar
    result = emergent.apply_emergent_rules(test_input)
    print(f"\n📤 Resultado emergente:")
    print(result)
    print(f"¿Correcto? {'✅' if np.array_equal(result, expected) else '❌'}")
    
    print("\n" + "="*70)
    print("4️⃣ ANÁLISIS DE LA TRANSFORMACIÓN CORRECTA")
    print("="*70)
    
    print("\n🎯 La transformación correcta sería:")
    print("1. Detectar el píxel central no-cero (3 en posición [1,1])")
    print("2. Expandir a las 4 posiciones cardinales:")
    print("   - Arriba: [0,1] = 3")
    print("   - Abajo: [2,1] = 3")
    print("   - Izquierda: [1,0] = 3")
    print("   - Derecha: [1,2] = 3")
    
    # Implementar manualmente la transformación correcta
    print("\n✨ Implementación manual correcta:")
    manual_result = test_input.copy()
    center_val = test_input[1, 1]
    if center_val != 0:
        # Expandir cruz
        manual_result[0, 1] = center_val  # arriba
        manual_result[2, 1] = center_val  # abajo
        manual_result[1, 0] = center_val  # izquierda
        manual_result[1, 2] = center_val  # derecha
    
    print(manual_result)
    print(f"¿Correcto? {'✅' if np.array_equal(manual_result, expected) else '❌'}")

def trace_apply_rule():
    """Rastrea específicamente apply_rule"""
    
    print("\n" + "="*70)
    print("5️⃣ RASTREO DE apply_rule")
    print("="*70)
    
    solver = ARCSolverPython()
    
    # Crear regla manual para expansión de cruz
    rule = {
        'type': 'pattern_replication',
        'confidence': 0.9,
        'parameters': {'factor': 3}
    }
    
    test_input = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    
    print(f"\n📝 Regla: {rule}")
    print(f"📊 Input:\n{test_input}")
    
    # Ver qué hace apply_rule
    print(f"\n🔍 Llamando apply_rule con tipo '{rule['type']}'...")
    
    # Rastrear el flujo
    from arc.arc_solver_python import TransformationType
    
    rule_type = TransformationType(rule['type'])
    print(f"TransformationType: {rule_type}")
    print(f"Es PATTERN_REPLICATION? {rule_type == TransformationType.PATTERN_REPLICATION}")
    
    # Aplicar
    result = solver.apply_rule(rule, test_input)
    print(f"\n📤 Resultado de apply_rule:")
    print(result)
    
    # Probar directamente _apply_pattern_replication
    print(f"\n🔧 Llamada directa a _apply_pattern_replication:")
    direct_result = solver._apply_pattern_replication(test_input, 3)
    print(direct_result)
    
    # Ver dimensiones
    print(f"\nDimensiones:")
    print(f"  Input: {test_input.shape}")
    print(f"  Resultado apply_rule: {result.shape}")
    print(f"  Resultado directo: {direct_result.shape}")

def check_fill_shape():
    """Verifica específicamente fill_shape"""
    
    print("\n" + "="*70)
    print("6️⃣ VERIFICACIÓN DE fill_shape")
    print("="*70)
    
    solver = ARCSolverPython()
    
    # Caso de prueba para fill
    test_input = np.array([
        [3, 0, 3],
        [3, 0, 3],
        [3, 3, 3]
    ])
    
    print("📊 Input (forma a rellenar):")
    print(test_input)
    
    # Aplicar fill_shape
    result = solver._apply_fill_shape(test_input, 3)
    print("\n📤 Resultado de _apply_fill_shape:")
    print(result)
    
    print("\n🔍 Análisis del algoritmo actual:")
    print("El algoritmo actual solo rellena si hay 3+ vecinos no-cero")
    print("Esto NO detecta formas cerradas correctamente")
    
    print("\n✨ Algoritmo correcto debería:")
    print("1. Detectar regiones cerradas por cada color")
    print("2. Usar flood-fill o binary_fill_holes")
    print("3. Rellenar completamente el interior")

def main():
    print("="*70)
    print("🔬 DEBUG COMPLETO DE TRANSFORMACIONES")
    print("="*70)
    
    # Rastrear toda la cadena
    debug_transformation_chain()
    
    # Rastrear apply_rule específicamente
    trace_apply_rule()
    
    # Verificar fill_shape
    check_fill_shape()
    
    print("\n" + "="*70)
    print("📊 RESUMEN DEL PROBLEMA")
    print("="*70)
    
    print("\n❌ PROBLEMAS IDENTIFICADOS:")
    print("1. _apply_pattern_replication espera factor de escala, no expansión de cruz")
    print("2. _apply_fill_shape solo rellena con 3+ vecinos, no detecta formas cerradas")
    print("3. No hay transformación específica para 'expandir cruz'")
    print("4. apply_rule no maneja correctamente algunos tipos de transformación")
    
    print("\n✅ SOLUCIÓN NECESARIA:")
    print("1. Implementar transformación 'expand_cross' específica")
    print("2. Mejorar _apply_fill_shape con flood-fill real")
    print("3. Agregar más transformaciones básicas")
    print("4. Mejorar la detección de patrones")

if __name__ == "__main__":
    main()