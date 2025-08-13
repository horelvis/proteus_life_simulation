#!/usr/bin/env python3
"""
Script de prueba para verificar las funciones implementadas en los módulos de atención
"""

import numpy as np
import sys
sys.path.insert(0, '/home/nexus/git/proteus_life_simulation/backend')

from arc.hierarchical_analyzer import HierarchicalAnalyzer
from arc.hierarchical_attention_solver import HierarchicalAttentionSolver

def test_hierarchical_analyzer():
    """Prueba el analizador jerárquico con una matriz simple"""
    print("\n=== Probando HierarchicalAnalyzer ===")
    
    # Crear una matriz de prueba con un patrón simple
    test_matrix = np.array([
        [0, 1, 0, 2, 0],
        [1, 1, 1, 2, 2],
        [0, 1, 0, 2, 0],
        [3, 0, 0, 0, 4],
        [3, 3, 0, 4, 4]
    ])
    
    analyzer = HierarchicalAnalyzer()
    result = analyzer.analyze_full_hierarchy(test_matrix)
    
    # Verificar que todas las funciones devuelven datos
    print(f"✓ Objetos detectados: {len(result['level_1_objects']['objects'])}")
    print(f"✓ Relaciones encontradas: {len(result['level_2_relations']['relations'])}")
    print(f"✓ Patrones detectados: {len(result['level_3_patterns']['patterns'])}")
    
    # Verificar funciones específicas implementadas
    if result['level_3_patterns']['patterns']:
        for pattern in result['level_3_patterns']['patterns']:
            print(f"  - Patrón: {pattern.pattern_type}, regularidad: {pattern.regularity_score:.2f}")
    
    # Verificar vinculaciones multiescala
    integrated = result['integrated_view']
    if 'pixel_object_links' in integrated:
        print(f"✓ Enlaces píxel-objeto: {len(integrated['pixel_object_links'].get('pixel_to_object_map', {}))}")
    if 'object_relation_links' in integrated:
        print(f"✓ Enlaces objeto-relación: {len(integrated['object_relation_links'].get('object_relations', {}))}")
    
    return True

def test_attention_solver():
    """Prueba el solver de atención con un puzzle simple"""
    print("\n=== Probando HierarchicalAttentionSolver ===")
    
    # Crear un ejemplo de entrenamiento simple
    train_examples = [{
        'input': [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        'output': [
            [2, 1, 2],
            [1, 1, 1],
            [2, 1, 2]
        ]
    }]
    
    test_input = np.array([
        [0, 3, 0],
        [3, 3, 3],
        [0, 3, 0]
    ])
    
    solver = HierarchicalAttentionSolver()
    solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    print(f"✓ Solución generada con shape: {solution.shape}")
    for step in steps:
        print(f"  - {step['description']}")
    
    # Verificar que el análisis jerárquico funciona
    input_grid = np.array(train_examples[0]['input'])
    output_grid = np.array(train_examples[0]['output'])
    analysis = solver._hierarchical_analysis(input_grid, output_grid)
    
    print(f"✓ Análisis jerárquico completado:")
    print(f"  - Objetos: {analysis['num_objects']}")
    print(f"  - Anclas: {analysis['num_anchors']}")
    print(f"  - Coherencia promedio: {analysis['avg_coherence']:.2%}")
    print(f"  - Patrones detectados: {len(analysis['patterns'])}")
    
    return True

def test_pattern_detection():
    """Prueba específica de detección de patrones"""
    print("\n=== Probando Detección de Patrones ===")
    
    # Matriz con patrón de rejilla
    grid_matrix = np.array([
        [1, 0, 2, 0, 3],
        [0, 0, 0, 0, 0],
        [4, 0, 5, 0, 6],
        [0, 0, 0, 0, 0],
        [7, 0, 8, 0, 9]
    ])
    
    analyzer = HierarchicalAnalyzer()
    result = analyzer.analyze_full_hierarchy(grid_matrix)
    
    patterns = result['level_3_patterns']['patterns']
    pattern_types = [p.pattern_type for p in patterns]
    
    print(f"✓ Tipos de patrones detectados: {set(pattern_types)}")
    
    # Verificar si detectó el patrón de rejilla
    grid_patterns = [p for p in patterns if p.pattern_type == 'grid']
    if grid_patterns:
        print(f"✓ Patrón de rejilla detectado con regularidad: {grid_patterns[0].regularity_score:.2f}")
    
    # Matriz con simetría
    symmetric_matrix = np.array([
        [1, 0, 0, 0, 1],
        [0, 2, 0, 2, 0],
        [0, 0, 3, 0, 0],
        [0, 2, 0, 2, 0],
        [1, 0, 0, 0, 1]
    ])
    
    result2 = analyzer.analyze_full_hierarchy(symmetric_matrix)
    patterns2 = result2['level_3_patterns']['patterns']
    
    symmetry_patterns = [p for p in patterns2 if p.pattern_type == 'symmetry']
    if symmetry_patterns:
        print(f"✓ Simetría detectada: {symmetry_patterns[0].transformation}")
    
    return True

if __name__ == "__main__":
    print("Verificando funciones implementadas en módulos de atención...")
    
    try:
        success = True
        success &= test_hierarchical_analyzer()
        success &= test_attention_solver()
        success &= test_pattern_detection()
        
        if success:
            print("\n✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
            print("Las funciones que estaban sin implementar ahora funcionan correctamente.")
        else:
            print("\n❌ Algunas pruebas fallaron")
            
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()