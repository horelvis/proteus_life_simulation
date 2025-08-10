#!/usr/bin/env python3
"""
Test simple del solver PROTEUS con puzzles básicos
"""

import numpy as np
from proteus_arc_solver import ProteusARCSolver

def print_grid(grid, title=""):
    """Imprime una grilla en formato texto"""
    if title:
        print(f"\n{title}:")
    
    char_map = {0: '.', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    
    for row in grid:
        print('   ', ' '.join(char_map.get(cell, str(cell)) for cell in row))

def test_simple_color_mapping():
    """Prueba con mapeo de color simple"""
    print("🧬 TEST 1: Mapeo de Color Simple (1→2)")
    print("="*50)
    
    # Crear solver con población pequeña
    solver = ProteusARCSolver(population_size=10)
    
    # Ejemplo muy simple: todos los 1 se convierten en 2
    train_examples = [
        {
            'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            'output': [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
        }
    ]
    
    test_input = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    
    print("\nEjemplo de entrenamiento:")
    print_grid(train_examples[0]['input'], "Input")
    print_grid(train_examples[0]['output'], "Output")
    
    print("\nTest:")
    print_grid(test_input, "Input")
    
    # Reducir generaciones para acelerar
    original_solve = solver.solve
    def quick_solve(train_examples, test_input):
        # Guardar configuración original
        solver.generation_limit = 20  # Solo 20 generaciones
        return original_solve(train_examples, test_input)
    
    solver.solve = quick_solve
    
    # Resolver
    print("\n🌀 Evolucionando solución...")
    solution = solver.solve(train_examples, test_input)
    
    print_grid(solution, "Solución PROTEUS")
    
    # Verificar
    expected = np.array([[2, 2, 0], [0, 2, 0], [2, 0, 2]])
    accuracy = np.sum(solution == expected) / expected.size
    print(f"\n✅ Accuracy: {accuracy*100:.1f}%")
    
    # Estadísticas de la población
    if solver.organisms:
        print(f"\n📊 Estadísticas de evolución:")
        print(f"   Organismos supervivientes: {len(solver.organisms)}")
        best = max(solver.organisms, key=lambda o: o.seed.dimension * o.energy)
        print(f"   Mejor dimensión: {best.seed.dimension:.3f}")
        print(f"   Números de Betti: {best.seed.betti_numbers}")
        print(f"   Experiencias codificadas: {best.memory.experience_count}")

def test_pattern_fill():
    """Prueba con relleno de patrón"""
    print("\n\n🧬 TEST 2: Relleno de Región")
    print("="*50)
    
    solver = ProteusARCSolver(population_size=15)
    
    # Rellenar interior con 4
    train_examples = [
        {
            'input': [[3, 3, 3], [3, 0, 3], [3, 3, 3]],
            'output': [[3, 3, 3], [3, 4, 3], [3, 3, 3]]
        }
    ]
    
    test_input = np.array([
        [3, 3, 3, 3, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 3, 3, 3, 3]
    ])
    
    print("\nEjemplo de entrenamiento:")
    print_grid(train_examples[0]['input'], "Input")
    print_grid(train_examples[0]['output'], "Output")
    
    print("\nTest:")
    print_grid(test_input, "Input")
    
    # Modificar para evolución rápida
    solver.generation_limit = 30
    
    print("\n🌀 Evolucionando solución...")
    
    # Hack temporal para limitar generaciones
    def solve_limited(train_examples, test_input):
        from proteus_arc_solver import TopologicalField
        print("   (Limitado a 30 generaciones para test rápido)")
        
        # Inicializar
        solver.field = TopologicalField(test_input.shape)
        solver._initialize_population(test_input.shape)
        
        # Aprender de ejemplos
        for example in train_examples:
            solver._experience_fields(
                np.array(example['input']), 
                np.array(example['output'])
            )
        
        # Evolucionar solo 30 generaciones
        best_solution = test_input
        for gen in range(30):
            solver.field.potential = test_input.astype(float)
            solver.field.update(solver.organisms)
            solver._evolve_generation()
            
            if solver.organisms and gen % 10 == 0:
                best_org = max(solver.organisms, key=lambda o: o.seed.dimension * o.energy)
                best_solution = solver._extract_solution(best_org, test_input)
                print(f"   Generación {gen}: {len(solver.organisms)} organismos vivos")
            
            solver.organisms = [org for org in solver.organisms if org.energy > 0]
            
            # Reproducción mínima
            while len(solver.organisms) < 5 and solver.organisms:
                parent = np.random.choice(solver.organisms)
                from proteus_arc_solver import ProteusOrganism
                child = ProteusOrganism(parent=parent)
                solver.organisms.append(child)
        
        return best_solution
    
    solution = solve_limited(train_examples, test_input)
    
    print_grid(solution, "Solución PROTEUS")
    
    # Verificar patrón (aunque no sea perfecto)
    interior_filled = solution[1:-1, 1:-1]
    if np.any(interior_filled != 0):
        print("\n✅ PROTEUS detectó que debe rellenar el interior")
    else:
        print("\n❌ PROTEUS no detectó el patrón de relleno")

def analyze_topological_approach():
    """Analiza el enfoque topológico"""
    print("\n\n💡 ANÁLISIS: Enfoque Topológico vs Reglas")
    print("="*50)
    
    print("\n📋 Ventajas del enfoque PROTEUS:")
    print("   • No requiere reglas predefinidas")
    print("   • Usa memoria holográfica (resistente a daño)")
    print("   • Evolución sin función de fitness explícita")
    print("   • Basado en invariantes topológicos")
    
    print("\n⚠️  Limitaciones actuales:")
    print("   • Más lento que reglas fijas")
    print("   • Requiere ajuste de hiperparámetros")
    print("   • La extracción de solución es simplificada")
    print("   • Necesita más desarrollo para puzzles complejos")
    
    print("\n🔬 Diferencias clave:")
    print("   • Reglas: if-then explícitos, rápido, limitado")
    print("   • PROTEUS: campos continuos, adaptativo, exploratorio")
    
    print("\n🚀 Mejoras necesarias:")
    print("   1. Mejor mapeo de campos topológicos a grillas discretas")
    print("   2. Evolución más dirigida por la tarea")
    print("   3. Memoria holográfica más estructurada")
    print("   4. Composición de transformaciones topológicas")

if __name__ == "__main__":
    print("🧬 PROTEUS-ARC: Test de Evolución Topológica")
    print("Basado en el paper PROTEUS - sin redes neuronales")
    print("="*60)
    
    # Tests simples
    test_simple_color_mapping()
    test_pattern_fill()
    
    # Análisis
    analyze_topological_approach()
    
    print("\n\n✨ Conclusión:")
    print("PROTEUS implementa los principios del paper pero necesita")
    print("refinamiento para competir con reglas especializadas en ARC.")