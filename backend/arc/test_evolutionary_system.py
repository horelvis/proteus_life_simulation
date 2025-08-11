#!/usr/bin/env python3
"""
Prueba del Sistema de Transformadores Evolutivos
Compara con HAMS y muestra cómo evolucionan las soluciones
"""

import numpy as np
import logging
from evolutionary_transformer_system import EvolutionaryTransformerSolver
from hierarchical_attention_solver import HierarchicalAttentionSolver
from arc_test_puzzles import ARCTestPuzzles
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_grid(grid: np.ndarray, title: str = ""):
    """Visualiza un grid de forma legible"""
    if title:
        print(f"\n{title}:")
    print("-" * (grid.shape[1] * 3 + 1))
    for row in grid:
        row_str = "|"
        for val in row:
            if val == 0:
                row_str += " . "
            else:
                row_str += f" {int(val)} "
        row_str += "|"
        print(row_str)
    print("-" * (grid.shape[1] * 3 + 1))

def test_evolutionary_solver():
    """Prueba el solver evolutivo con puzzles de ejemplo"""
    
    print("="*80)
    print("SISTEMA DE TRANSFORMADORES EVOLUTIVOS")
    print("="*80)
    print("\n🧬 Las transformaciones NO están hardcodeadas")
    print("🧬 Emergen de la evolución de genes")
    print("🧬 Cada generación mejora la solución\n")
    
    # Crear solver evolutivo
    evo_solver = EvolutionaryTransformerSolver(pool_size=30)
    
    # Obtener puzzles de prueba
    puzzles = ARCTestPuzzles.get_all_puzzles()
    
    # Probar con algunos puzzles
    test_puzzles = [
        puzzles[4],  # gravity_simulation
        puzzles[0],  # pattern_completion
        puzzles[2],  # object_counting
    ]
    
    for puzzle in test_puzzles:
        print("\n" + "="*60)
        print(f"🧩 Puzzle: {puzzle['name']}")
        print(f"📝 {puzzle['description']}")
        print("="*60)
        
        # Preparar datos
        train_examples = [
            {'input': ex['input'].tolist(), 'output': ex['output'].tolist()}
            for ex in puzzle['train']
        ]
        test_input = puzzle['test']['input']
        expected_output = puzzle['test']['output']
        
        # Mostrar ejemplo de entrenamiento
        print("\n📚 Ejemplo de entrenamiento:")
        visualize_grid(puzzle['train'][0]['input'], "Input")
        visualize_grid(puzzle['train'][0]['output'], "Output esperado")
        
        print("\n🧪 Test:")
        visualize_grid(test_input, "Input de test")
        
        # Resolver con sistema evolutivo
        print("\n🧬 Evolucionando solución...")
        start_time = time.time()
        
        evo_solution = evo_solver.solve(train_examples, test_input)
        
        evo_time = time.time() - start_time
        
        visualize_grid(evo_solution, "Solución evolutiva")
        
        # Calcular accuracy
        if evo_solution.shape == expected_output.shape:
            accuracy = np.mean(evo_solution == expected_output)
            exact_match = np.array_equal(evo_solution, expected_output)
        else:
            accuracy = 0.0
            exact_match = False
        
        print(f"\n📊 Resultados:")
        print(f"  - Accuracy: {accuracy:.1%}")
        print(f"  - Match exacto: {'✅ SÍ' if exact_match else '❌ NO'}")
        print(f"  - Tiempo: {evo_time:.2f}s")
        print(f"  - Generaciones: {evo_solver.pool.generation}")
        
        # Mostrar genoma del mejor transformador
        if evo_solver.pool.population:
            best = evo_solver.pool.population[0]
            print(f"\n🧬 Genoma del mejor transformador (fitness: {best.fitness:.2%}):")
            for i, gene in enumerate(best.genome[:5]):  # Mostrar primeros 5 genes
                print(f"  Gen {i+1}: {gene.operation} - {gene.parameters}")
            if len(best.genome) > 5:
                print(f"  ... y {len(best.genome)-5} genes más")

def compare_with_hams():
    """Compara el sistema evolutivo con HAMS"""
    
    print("\n" + "="*80)
    print("COMPARACIÓN: EVOLUTIVO vs HAMS")
    print("="*80)
    
    evo_solver = EvolutionaryTransformerSolver(pool_size=20)
    hams_solver = HierarchicalAttentionSolver()
    
    # Puzzle simple para comparación rápida
    puzzle = ARCTestPuzzles.puzzle_gravity_simulation()
    
    train_examples = [
        {'input': ex['input'].tolist(), 'output': ex['output'].tolist()}
        for ex in puzzle['train']
    ]
    test_input = puzzle['test']['input']
    expected_output = puzzle['test']['output']
    
    print(f"\n🧩 Puzzle: {puzzle['name']}")
    visualize_grid(test_input, "Input de test")
    visualize_grid(expected_output, "Output esperado")
    
    # HAMS (con funciones hardcodeadas)
    print("\n1️⃣ HAMS (funciones hardcodeadas):")
    start = time.time()
    hams_solution = hams_solver.solve(train_examples, test_input)
    hams_time = time.time() - start
    
    hams_acc = np.mean(hams_solution == expected_output) if hams_solution.shape == expected_output.shape else 0
    print(f"  - Accuracy: {hams_acc:.1%}")
    print(f"  - Tiempo: {hams_time:.3f}s")
    
    # Evolutivo (sin funciones hardcodeadas)
    print("\n2️⃣ EVOLUTIVO (genes emergentes):")
    start = time.time()
    evo_solution = evo_solver.solve(train_examples, test_input)
    evo_time = time.time() - start
    
    evo_acc = np.mean(evo_solution == expected_output) if evo_solution.shape == expected_output.shape else 0
    print(f"  - Accuracy: {evo_acc:.1%}")
    print(f"  - Tiempo: {evo_time:.3f}s")
    print(f"  - Generaciones evolucionadas: {evo_solver.pool.generation}")
    
    # Conclusión
    print("\n📊 Resumen:")
    if evo_acc > hams_acc:
        print("  🏆 EVOLUTIVO gana - mejor accuracy sin código hardcodeado")
    elif hams_acc > evo_acc:
        print("  🏆 HAMS gana - pero usa funciones hardcodeadas")
    else:
        print("  🤝 Empate en accuracy")
    
    print("\n💡 Diferencia clave:")
    print("  - HAMS: Elige entre 6 funciones predefinidas")
    print("  - EVOLUTIVO: Crea su propia transformación desde cero")

def demonstrate_evolution():
    """Demuestra cómo evoluciona una solución generación por generación"""
    
    print("\n" + "="*80)
    print("DEMOSTRACIÓN DE EVOLUCIÓN")
    print("="*80)
    
    # Crear un puzzle simple
    train_examples = [
        {
            'input': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            'output': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # Expansión en cruz
        }
    ]
    test_input = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    
    print("\n📚 Ejemplo de entrenamiento:")
    visualize_grid(np.array(train_examples[0]['input']), "Input")
    visualize_grid(np.array(train_examples[0]['output']), "Output (expansión en cruz)")
    
    print("\n🧪 A resolver:")
    visualize_grid(test_input, "Test input")
    
    # Evolucionar paso a paso
    from evolutionary_transformer_system import EvolutionaryPool
    
    pool = EvolutionaryPool(pool_size=20)
    
    print("\n🧬 Evolución generación por generación:\n")
    
    for gen in range(10):
        # Evaluar población
        for seed in pool.population:
            seed.fitness = pool._evaluate_fitness(seed, train_examples)
        
        # Ordenar por fitness
        pool.population.sort(key=lambda s: s.fitness, reverse=True)
        best = pool.population[0]
        
        print(f"Generación {gen+1}:")
        print(f"  Mejor fitness: {best.fitness:.2%}")
        print(f"  Genes: {len(best.genome)}")
        
        # Aplicar el mejor
        result = best.apply(test_input)
        
        # Mostrar resultado si mejoró
        if gen == 0 or gen == 4 or gen == 9:
            visualize_grid(result, f"  Resultado en generación {gen+1}")
        
        # Terminar si es perfecto
        if best.fitness >= 0.99:
            print(f"\n✅ ¡Solución perfecta encontrada en generación {gen+1}!")
            break
        
        # Siguiente generación
        pool._next_generation()
        pool.generation += 1
    
    print("\n🧬 Genoma final del mejor transformador:")
    for i, gene in enumerate(best.genome):
        print(f"  Gen {i+1}: {gene.operation} con parámetros {gene.parameters}")

if __name__ == "__main__":
    # Ejecutar pruebas
    print("\n🚀 Iniciando pruebas del Sistema Evolutivo...\n")
    
    # 1. Prueba básica
    test_evolutionary_solver()
    
    # 2. Comparación con HAMS
    compare_with_hams()
    
    # 3. Demostración de evolución
    demonstrate_evolution()
    
    print("\n✅ Pruebas completadas")