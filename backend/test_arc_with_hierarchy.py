#!/usr/bin/env python3
"""
Test del Sistema ARC con Análisis Jerárquico Completo
Prueba con puzzles reales para ver si las mejoras dan mejores resultados
"""

import numpy as np
from arc import HybridProteusARCSolver, StructuralAnalyzer
from arc.hierarchical_analyzer import HierarchicalAnalyzer

class EnhancedHybridSolver(HybridProteusARCSolver):
    """Solver híbrido mejorado con análisis jerárquico completo"""
    
    def __init__(self):
        super().__init__()
        self.hierarchical_analyzer = HierarchicalAnalyzer()
    
    def detect_rule(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Detecta reglas usando análisis jerárquico completo"""
        
        # Análisis jerárquico del input
        input_hierarchy = self.hierarchical_analyzer.analyze_full_hierarchy(input_grid)
        
        # Análisis jerárquico del output
        output_hierarchy = self.hierarchical_analyzer.analyze_full_hierarchy(output_grid)
        
        # Usar información jerárquica para mejorar detección
        rule = self._detect_with_hierarchy(
            input_grid, output_grid, 
            input_hierarchy, output_hierarchy
        )
        
        # Si no encuentra regla con jerarquía, usar método padre
        if not rule:
            rule = super().detect_rule(input_grid, output_grid)
        
        return rule
    
    def _detect_with_hierarchy(self, input_grid, output_grid, input_h, output_h):
        """Detecta transformaciones usando análisis jerárquico"""
        
        # Comparar objetos entre input y output
        input_objects = input_h['level_1_objects']['objects']
        output_objects = output_h['level_1_objects']['objects']
        
        # Analizar cambios en número de objetos
        obj_change = len(output_objects) - len(input_objects)
        
        # Analizar cambios en relaciones
        input_relations = input_h['level_2_relations']['relations']
        output_relations = output_h['level_2_relations']['relations']
        
        # Detectar tipo de transformación basado en cambios jerárquicos
        if obj_change == 0 and len(input_objects) > 0:
            # Mismo número de objetos - posible transformación geométrica
            
            # Verificar si es relleno
            input_pixels = input_h['level_0_pixels']
            output_pixels = output_h['level_0_pixels']
            
            if output_pixels.get('local_patterns', {}).get('interior', 0) > \
               input_pixels.get('local_patterns', {}).get('interior', 0):
                return {
                    'type': 'fill_shape',
                    'confidence': 0.9,
                    'parameters': {},
                    'hierarchy_based': True
                }
            
            # Verificar rotación/reflexión
            for inp_obj in input_objects:
                for out_obj in output_objects:
                    if inp_obj.color == out_obj.color:
                        if abs(inp_obj.orientation - out_obj.orientation) > 45:
                            return {
                                'type': 'rotation',
                                'confidence': 0.85,
                                'parameters': {'angle': 90},
                                'hierarchy_based': True
                            }
        
        elif obj_change > 0:
            # Más objetos en output - posible replicación
            return {
                'type': 'pattern_replication',
                'confidence': 0.8,
                'parameters': {'factor': 2},
                'hierarchy_based': True
            }
        
        # Analizar patrones globales
        input_patterns = input_h['level_3_patterns']['patterns']
        output_patterns = output_h['level_3_patterns']['patterns']
        
        if output_patterns and not input_patterns:
            # Se creó un patrón nuevo
            return {
                'type': 'pattern_creation',
                'confidence': 0.75,
                'parameters': {},
                'hierarchy_based': True
            }
        
        return None

def test_puzzle(name, train_examples, test_input, expected=None):
    """Prueba un puzzle con ambos solvers para comparar"""
    
    print("\n" + "="*70)
    print(f"🧪 {name}")
    print("="*70)
    
    print("\n📊 Input de test:")
    print(test_input)
    if expected is not None:
        print("\n✅ Output esperado:")
        print(expected)
    
    # Probar con solver original
    print("\n" + "-"*50)
    print("📌 SOLVER ORIGINAL (sin jerarquía completa):")
    print("-"*50)
    
    original_solver = HybridProteusARCSolver()
    original_solution, original_steps = original_solver.solve_with_steps(train_examples, test_input)
    
    print(f"Solución: \n{original_solution}")
    print(f"Confianza: {original_solver.confidence:.1%}")
    print(f"Pasos: {len(original_steps)}")
    
    # Verificar si es correcta
    original_correct = np.array_equal(original_solution, expected) if expected is not None else None
    if original_correct is not None:
        print(f"¿Correcta? {'✅ SÍ' if original_correct else '❌ NO'}")
    
    # Probar con solver mejorado
    print("\n" + "-"*50)
    print("🚀 SOLVER MEJORADO (con análisis jerárquico):")
    print("-"*50)
    
    enhanced_solver = EnhancedHybridSolver()
    enhanced_solution, enhanced_steps = enhanced_solver.solve_with_steps(train_examples, test_input)
    
    print(f"Solución: \n{enhanced_solution}")
    print(f"Confianza: {enhanced_solver.confidence:.1%}")
    print(f"Pasos: {len(enhanced_steps)}")
    
    # Mostrar información jerárquica usada
    for step in enhanced_steps:
        if 'rule' in step and step['rule'] and step['rule'].get('hierarchy_based'):
            print("  ✨ Usó análisis jerárquico")
            break
    
    # Verificar si es correcta
    enhanced_correct = np.array_equal(enhanced_solution, expected) if expected is not None else None
    if enhanced_correct is not None:
        print(f"¿Correcta? {'✅ SÍ' if enhanced_correct else '❌ NO'}")
    
    # Comparación
    print("\n" + "-"*50)
    print("📊 COMPARACIÓN:")
    print("-"*50)
    
    if original_correct is not None and enhanced_correct is not None:
        if enhanced_correct and not original_correct:
            print("🎉 ¡MEJORA! El solver mejorado resolvió correctamente")
        elif original_correct and not enhanced_correct:
            print("⚠️ Regresión: El solver original era mejor")
        elif original_correct and enhanced_correct:
            print("✅ Ambos resolvieron correctamente")
        else:
            print("❌ Ninguno resolvió correctamente")
    
    # Comparar confianza
    conf_diff = enhanced_solver.confidence - original_solver.confidence
    if abs(conf_diff) > 0.1:
        if conf_diff > 0:
            print(f"📈 Confianza mejoró: +{conf_diff:.1%}")
        else:
            print(f"📉 Confianza bajó: {conf_diff:.1%}")
    
    return {
        'original': {'solution': original_solution, 'correct': original_correct},
        'enhanced': {'solution': enhanced_solution, 'correct': enhanced_correct}
    }

def main():
    print("="*70)
    print("🔬 PRUEBA DEL SISTEMA ARC CON MEJORAS JERÁRQUICAS")
    print("="*70)
    
    results = []
    
    # Puzzle 1: Expansión de cruz
    print("\n🎯 Puzzle 1: Expansión de cruz")
    result1 = test_puzzle(
        "Expansión de Cruz",
        train_examples=[
            {"input": np.array([[0,0,0],[0,1,0],[0,0,0]]), 
             "output": np.array([[0,1,0],[1,1,1],[0,1,0]])},
            {"input": np.array([[0,0,0],[0,2,0],[0,0,0]]), 
             "output": np.array([[0,2,0],[2,2,2],[0,2,0]])}
        ],
        test_input=np.array([[0,0,0],[0,3,0],[0,0,0]]),
        expected=np.array([[0,3,0],[3,3,3],[0,3,0]])
    )
    results.append(('Cruz', result1))
    
    # Puzzle 2: Relleno de forma
    print("\n🎯 Puzzle 2: Relleno de forma")
    result2 = test_puzzle(
        "Relleno de Forma",
        train_examples=[
            {"input": np.array([[1,1,1],[1,0,1],[1,1,1]]), 
             "output": np.array([[1,1,1],[1,1,1],[1,1,1]])},
            {"input": np.array([[2,2,0],[2,0,0],[2,2,2]]), 
             "output": np.array([[2,2,2],[2,2,2],[2,2,2]])}
        ],
        test_input=np.array([[3,0,3],[3,0,3],[3,3,3]]),
        expected=np.array([[3,3,3],[3,3,3],[3,3,3]])
    )
    results.append(('Relleno', result2))
    
    # Puzzle 3: Patrón diagonal
    print("\n🎯 Puzzle 3: Patrón diagonal")
    result3 = test_puzzle(
        "Completar Diagonal",
        train_examples=[
            {"input": np.array([[1,0,0],[0,1,0],[0,0,1]]), 
             "output": np.array([[1,1,1],[1,1,1],[1,1,1]])},
            {"input": np.array([[2,0,0],[0,2,0],[0,0,2]]), 
             "output": np.array([[2,2,2],[2,2,2],[2,2,2]])}
        ],
        test_input=np.array([[3,0,0],[0,3,0],[0,0,3]]),
        expected=np.array([[3,3,3],[3,3,3],[3,3,3]])
    )
    results.append(('Diagonal', result3))
    
    # Puzzle 4: Reflexión
    print("\n🎯 Puzzle 4: Reflexión horizontal")
    result4 = test_puzzle(
        "Reflexión Horizontal",
        train_examples=[
            {"input": np.array([[1,0],[0,0]]), 
             "output": np.array([[0,1],[0,0]])},
            {"input": np.array([[2,2],[0,0]]), 
             "output": np.array([[2,2],[0,0]])}
        ],
        test_input=np.array([[3,0],[3,0]]),
        expected=np.array([[0,3],[0,3]])
    )
    results.append(('Reflexión', result4))
    
    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*70)
    
    original_correct = sum(1 for _, r in results if r['original']['correct'])
    enhanced_correct = sum(1 for _, r in results if r['enhanced']['correct'])
    
    print(f"\n✅ Solver Original: {original_correct}/{len(results)} correctos")
    print(f"✅ Solver Mejorado: {enhanced_correct}/{len(results)} correctos")
    
    if enhanced_correct > original_correct:
        improvement = ((enhanced_correct - original_correct) / len(results)) * 100
        print(f"\n🎉 ¡MEJORA SIGNIFICATIVA! +{improvement:.0f}% de precisión")
    elif enhanced_correct == original_correct:
        print(f"\n➡️ Rendimiento similar")
    else:
        print(f"\n⚠️ Rendimiento inferior con las mejoras")
    
    print("\n📋 Detalle por puzzle:")
    for name, result in results:
        orig = "✅" if result['original']['correct'] else "❌"
        enh = "✅" if result['enhanced']['correct'] else "❌"
        print(f"  • {name}: Original {orig} | Mejorado {enh}")
    
    print("\n💡 Conclusión:")
    if enhanced_correct >= original_correct:
        print("Las mejoras jerárquicas están funcionando correctamente")
    else:
        print("Se necesitan ajustes en el análisis jerárquico")

if __name__ == "__main__":
    main()