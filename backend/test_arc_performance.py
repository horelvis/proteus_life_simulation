#!/usr/bin/env python3
"""
Script para evaluar el rendimiento real del sistema ARC mejorado
"""

import numpy as np
import json
import os
from pathlib import Path
from arc import ARCSolver

def load_arc_puzzles():
    """Carga puzzles de ARC oficiales"""
    puzzles = []
    
    # Buscar archivos JSON en el directorio de caché
    cache_dir = Path("/app/arc_official_cache")
    if cache_dir.exists():
        for file in cache_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    puzzles.append({
                        'name': file.stem,
                        'data': data
                    })
            except:
                pass
    
    # Si no hay caché, crear puzzles de prueba
    if not puzzles:
        puzzles = [
            {
                'name': 'test_cross_expansion',
                'data': {
                    'train': [{
                        'input': [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                        'output': [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                    }],
                    'test': [{'input': [[0, 0, 0], [0, 2, 0], [0, 0, 0]]}]
                }
            },
            {
                'name': 'test_color_mapping',
                'data': {
                    'train': [{
                        'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                        'output': [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
                    }],
                    'test': [{'input': [[3, 0, 3], [0, 3, 0], [3, 0, 3]]}]
                }
            },
            {
                'name': 'test_gravity',
                'data': {
                    'train': [{
                        'input': [[1, 0, 2], [0, 0, 0], [0, 3, 0]],
                        'output': [[0, 0, 0], [0, 0, 0], [1, 3, 2]]
                    }],
                    'test': [{'input': [[4, 0, 0], [0, 5, 0], [0, 0, 6]]}]
                }
            }
        ]
    
    return puzzles

def evaluate_puzzle(solver, puzzle_data):
    """Evalúa un puzzle individual"""
    try:
        train_examples = puzzle_data['train']
        test_input = np.array(puzzle_data['test'][0]['input'])
        
        # Resolver
        solution, steps = solver.solve_with_steps(train_examples, test_input)
        
        # Si hay output esperado, calcular accuracy
        if 'output' in puzzle_data['test'][0]:
            expected = np.array(puzzle_data['test'][0]['output'])
            if solution.shape == expected.shape:
                accuracy = np.mean(solution == expected) * 100
            else:
                accuracy = 0.0
        else:
            accuracy = None
        
        return {
            'success': True,
            'solution': solution.tolist(),
            'steps': steps,
            'accuracy': accuracy
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'accuracy': 0.0
        }

def main():
    print("=" * 80)
    print("🧪 EVALUACIÓN DE RENDIMIENTO DEL SISTEMA ARC MEJORADO")
    print("=" * 80)
    
    # Inicializar solver
    solver = ARCSolver()
    print(f"✅ Usando: {solver.__class__.__name__}")
    print()
    
    # Cargar puzzles
    puzzles = load_arc_puzzles()
    print(f"📁 Evaluando {len(puzzles)} puzzles")
    print()
    
    results = []
    successful = 0
    total_accuracy = []
    
    for i, puzzle in enumerate(puzzles, 1):
        print(f"{i:3}. {puzzle['name'][:30]:30} ", end="")
        
        result = evaluate_puzzle(solver, puzzle['data'])
        results.append(result)
        
        if result['success']:
            if result['accuracy'] is not None:
                print(f"✅ {result['accuracy']:6.1f}%")
                total_accuracy.append(result['accuracy'])
                if result['accuracy'] == 100.0:
                    successful += 1
            else:
                print("✅ Resuelto")
                
            # Mostrar pasos si es interesante
            if len(result['steps']) > 0:
                for step in result['steps'][:2]:  # Solo primeros 2 pasos
                    print(f"     → {step['description']}")
        else:
            print(f"❌ Error: {result['error'][:50]}")
    
    print()
    print("=" * 80)
    print("📊 RESULTADOS FINALES")
    print("=" * 80)
    
    if total_accuracy:
        avg_accuracy = np.mean(total_accuracy)
        print(f"✅ Puzzles perfectos: {successful}/{len(puzzles)} ({successful/len(puzzles)*100:.1f}%)")
        print(f"📈 Accuracy promedio: {avg_accuracy:.1f}%")
        print(f"🎯 Accuracy máxima: {max(total_accuracy):.1f}%")
        print(f"📉 Accuracy mínima: {min(total_accuracy):.1f}%")
    else:
        print("ℹ️ No hay puzzles con solución conocida para evaluar accuracy")
    
    # Análisis de tipos de transformaciones detectadas
    print()
    print("🔍 Transformaciones detectadas:")
    transformations = {}
    for result in results:
        if result['success'] and result['steps']:
            for step in result['steps']:
                if 'Transformación:' in step['description']:
                    trans_type = step['description'].split(':')[1].split('(')[0].strip()
                    transformations[trans_type] = transformations.get(trans_type, 0) + 1
    
    for trans, count in sorted(transformations.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {trans}: {count} veces")

if __name__ == "__main__":
    main()