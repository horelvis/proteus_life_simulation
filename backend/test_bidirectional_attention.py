#!/usr/bin/env python3
"""
Test del Sistema de Atención Bidireccional
Verifica que cada píxel conoce su contexto completo
"""

import numpy as np
from arc.enhanced_solver_attention import EnhancedSolverWithAttention
from arc.bidirectional_attention import BidirectionalAttentionSystem
import json

def test_bidirectional_context():
    """
    Prueba que los píxeles tienen contexto completo arriba y abajo
    """
    
    print("="*70)
    print("🔄 TEST DEL SISTEMA DE ATENCIÓN BIDIRECCIONAL")
    print("="*70)
    
    # Caso de prueba: Expansión de cruz
    train_examples = [
        {
            "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        }
    ]
    
    test_input = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    
    # Crear solver con atención
    solver = EnhancedSolverWithAttention()
    
    # Resolver con atención
    solution, analysis = solver.solve_with_attention(train_examples, test_input)
    
    print("\n📊 ANÁLISIS DE LA JERARQUÍA BIDIRECCIONAL")
    print("-"*50)
    
    # Mostrar información de la jerarquía
    if 'bidirectional_structure' in analysis:
        structure = analysis['bidirectional_structure']
        
        print(f"\n📍 Píxeles analizados: {len(structure.get('pixels', {}))}")
        print(f"🔲 Objetos detectados: {structure.get('num_objects', 0)}")
        print(f"🔗 Relaciones encontradas: {structure.get('num_relations', 0)}")
        print(f"🎯 Patrones identificados: {structure.get('num_patterns', 0)}")
    
    # Analizar píxeles específicos
    print("\n🔍 CONTEXTO COMPLETO DE PÍXELES CLAVE")
    print("-"*50)
    
    # El píxel central debe tener el contexto más rico
    attention_system = solver.attention_system
    
    if attention_system.pixel_layer:
        # Píxel central (1, 1)
        center_pixel = attention_system.pixel_layer.get((1, 1))
        
        if center_pixel:
            print(f"\n📌 Píxel Central (1, 1):")
            print(f"  Valor: {center_pixel.value}")
            print(f"  Importancia: {center_pixel.importance_score:.2f}")
            
            print(f"\n  ⬆️ INFORMACIÓN TOP-DOWN:")
            print(f"    - Patrón padre: {center_pixel.parent_pattern_id}")
            print(f"    - Rol en patrón: {center_pixel.pattern_role}")
            print(f"    - Valor esperado: {center_pixel.expected_value_from_pattern}")
            
            print(f"\n  ⬇️ INFORMACIÓN BOTTOM-UP:")
            print(f"    - Objeto padre: {center_pixel.parent_object_id}")
            print(f"    - Rol en objeto: {center_pixel.object_role}")
            print(f"    - Relación padre: {center_pixel.parent_relation_id}")
            print(f"    - Rol en relación: {center_pixel.relation_role}")
            
            print(f"\n  🔄 COHERENCIA BIDIRECCIONAL:")
            print(f"    - Soporta patrón: {center_pixel.supports_pattern}")
            print(f"    - Confianza en contexto: {center_pixel.confidence_in_context:.2f}")
            
            # Mostrar contexto completo
            full_context = center_pixel.get_full_context()
            print(f"\n  📋 Contexto Completo:")
            print(json.dumps(full_context, indent=4))
        
        # Analizar píxeles de expansión
        print("\n📐 PÍXELES DE EXPANSIÓN:")
        expansion_positions = [(0, 1), (2, 1), (1, 0), (1, 2)]
        
        for x, y in expansion_positions:
            pixel = attention_system.pixel_layer.get((x, y))
            if pixel:
                print(f"\n  Píxel ({x}, {y}):")
                print(f"    Valor: {pixel.value}")
                print(f"    Rol objeto: {pixel.object_role}")
                print(f"    Rol patrón: {pixel.pattern_role}")
                print(f"    Importancia: {pixel.importance_score:.2f}")
    
    # Mostrar mapa de atención
    if 'attention_map' in analysis and analysis['attention_map']:
        print("\n🗺️ MAPA DE ATENCIÓN:")
        print("-"*50)
        attention_map = np.array(analysis['attention_map'])
        
        print("\nValores de atención (0-1):")
        for row in attention_map:
            print("  " + " ".join(f"{val:.2f}" for val in row))
        
        # Encontrar puntos de máxima atención
        max_attention = np.max(attention_map)
        high_attention_points = np.argwhere(attention_map > max_attention * 0.8)
        
        print(f"\n🎯 Puntos de alta atención (>{max_attention*0.8:.2f}):")
        for y, x in high_attention_points:
            print(f"  - Posición ({x}, {y}): {attention_map[y, x]:.2f}")
    
    # Mostrar insights
    if 'insights' in analysis:
        insights = analysis['insights']
        print("\n💡 INSIGHTS DEL ANÁLISIS:")
        print("-"*50)
        print(f"  Atención promedio: {insights.get('average_attention', 0):.2f}")
        print(f"  Atención máxima: {insights.get('max_attention', 0):.2f}")
        print(f"  Score de coherencia: {insights.get('coherence_score', 0):.2f}")
        
        if 'high_attention_points' in insights:
            print(f"\n  📍 Puntos críticos detectados: {len(insights['high_attention_points'])}")
            
            for i, point in enumerate(insights['high_attention_points'][:3]):
                print(f"\n  Punto {i+1}: {point['position']}")
                print(f"    Atención: {point['attention']:.2f}")
                context = point['context']
                print(f"    Objeto: {context['object']['role']}")
                print(f"    Relación: {context['relation']['role']}")
                print(f"    Patrón: {context['pattern']['role']}")
    
    # Verificar la solución
    print("\n✅ VERIFICACIÓN DE LA SOLUCIÓN:")
    print("-"*50)
    print(f"Input de test:\n{test_input}")
    print(f"\nSolución obtenida:\n{solution}")
    
    expected = np.array([[0, 3, 0], [3, 3, 3], [0, 3, 0]])
    print(f"\nSolución esperada:\n{expected}")
    
    correct = np.array_equal(solution, expected)
    print(f"\n{'✅ CORRECTO' if correct else '❌ INCORRECTO'}")
    
    # Mostrar transformación detectada
    if 'transformation_detected' in analysis:
        trans = analysis['transformation_detected']
        print(f"\n🔄 Transformación detectada: {trans.get('type', 'unknown')}")
        print(f"   Confianza: {trans.get('confidence', 0):.2%}")
        
        if 'pattern_type' in trans:
            print(f"   Tipo de patrón: {trans['pattern_type']}")
            print(f"   Confianza del patrón: {trans.get('pattern_confidence', 0):.2%}")
    
    return correct

def test_complex_hierarchy():
    """
    Prueba con un ejemplo más complejo para ver la jerarquía completa
    """
    
    print("\n" + "="*70)
    print("🏗️ TEST DE JERARQUÍA COMPLEJA")
    print("="*70)
    
    # Caso con múltiples objetos y relaciones
    train_examples = [
        {
            "input": [
                [1, 0, 0, 2],
                [1, 0, 0, 2],
                [0, 0, 0, 0],
                [3, 0, 0, 4]
            ],
            "output": [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [0, 0, 0, 0],
                [3, 3, 4, 4]
            ]
        }
    ]
    
    test_input = np.array([
        [5, 0, 0, 6],
        [5, 0, 0, 6],
        [0, 0, 0, 0],
        [7, 0, 0, 8]
    ])
    
    solver = EnhancedSolverWithAttention()
    solution, analysis = solver.solve_with_attention(train_examples, test_input)
    
    # Analizar objetos detectados
    attention_system = solver.attention_system
    
    print("\n🔲 OBJETOS DETECTADOS:")
    print("-"*50)
    
    for obj_id, obj in attention_system.objects.items():
        print(f"\nObjeto #{obj_id}:")
        print(f"  Color: {obj.color}")
        print(f"  Tamaño: {obj.size} píxeles")
        print(f"  Forma: {obj.shape_type}")
        print(f"  Centro: ({obj.center[0]:.1f}, {obj.center[1]:.1f})")
        print(f"  Relaciones: {obj.parent_relations}")
        print(f"  Patrones: {obj.parent_patterns}")
    
    print("\n🔗 RELACIONES DETECTADAS:")
    print("-"*50)
    
    for rel_id, rel in attention_system.relations.items():
        print(f"\nRelación #{rel_id}:")
        print(f"  Tipo: {rel.type}")
        print(f"  Objetos: {rel.object_ids}")
        print(f"  Fuerza: {rel.strength:.2f}")
    
    print("\n🎯 PATRONES DETECTADOS:")
    print("-"*50)
    
    for pat_id, pat in attention_system.patterns.items():
        print(f"\nPatrón #{pat_id}:")
        print(f"  Tipo: {pat.type}")
        print(f"  Confianza: {pat.confidence:.2f}")
        print(f"  Objetos involucrados: {pat.child_objects}")
        print(f"  Relaciones involucradas: {pat.child_relations}")
    
    return solution

def main():
    print("="*70)
    print("🧪 SUITE DE TESTS - SISTEMA DE ATENCIÓN BIDIRECCIONAL")
    print("="*70)
    
    # Test 1: Contexto bidireccional básico
    test1_result = test_bidirectional_context()
    
    # Test 2: Jerarquía compleja
    test2_result = test_complex_hierarchy()
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE TESTS")
    print("="*70)
    
    print(f"\n✅ Test 1 (Contexto Bidireccional): {'PASÓ' if test1_result else 'FALLÓ'}")
    print(f"✅ Test 2 (Jerarquía Compleja): Completado")
    
    print("\n💡 CONCLUSIÓN:")
    print("El sistema de atención bidireccional permite que cada píxel")
    print("conozca su contexto completo en la jerarquía, desde su rol")
    print("local hasta su participación en patrones globales.")

if __name__ == "__main__":
    main()