#!/usr/bin/env python3
"""
Evaluación de la Red de Razonamiento Lógico en ARC
Sistema puro de razonamiento sin simulación
"""

import numpy as np
import json
import os
from glob import glob
from typing import Dict, List, Tuple
from arc import ARCSolver, LogicalReasoningNetwork

def load_official_puzzles() -> List[Dict]:
    """Carga puzzles oficiales de ARC"""
    puzzles = []
    
    # Buscar en el caché oficial
    cache_dir = "arc_official_cache"
    if os.path.exists(cache_dir):
        pattern = os.path.join(cache_dir, "*.json")
        files = glob(pattern)
        
        for file_path in files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                puzzles.append({
                    'name': os.path.basename(file_path).replace('.json', ''),
                    'data': data
                })
    
    return puzzles

def evaluate_puzzle(solver: LogicalReasoningNetwork, puzzle_data: Dict) -> Dict:
    """Evalúa un puzzle con la red de razonamiento"""
    try:
        train = puzzle_data['train']
        test_input = np.array(puzzle_data['test'][0]['input'])
        
        # Resolver con razonamiento lógico
        solution = solver.reason(train, test_input)
        
        # Calcular accuracy si hay output esperado
        if 'output' in puzzle_data['test'][0]:
            expected = np.array(puzzle_data['test'][0]['output'])
            if solution.shape == expected.shape:
                accuracy = np.mean(solution == expected) * 100
                exact_match = np.array_equal(solution, expected)
            else:
                accuracy = 0.0
                exact_match = False
        else:
            accuracy = None
            exact_match = None
        
        # Extraer inferencias
        inferences = []
        if hasattr(solver, 'inferences'):
            for inf in solver.inferences[:3]:
                inferences.append({
                    'level': inf.level,
                    'premise': inf.premise,
                    'conclusion': inf.conclusion,
                    'confidence': inf.confidence
                })
        
        return {
            'success': True,
            'accuracy': accuracy,
            'exact_match': exact_match,
            'inferences': inferences
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'accuracy': 0.0,
            'exact_match': False
        }

def main():
    print("=" * 80)
    print("🧠 EVALUACIÓN DE RED DE RAZONAMIENTO LÓGICO")
    print("=" * 80)
    print()
    print("Arquitectura: MACRO (V-JEPA) → MESO (Objetos) → MICRO (Píxeles)")
    print()
    
    # Cargar puzzles
    puzzles = load_official_puzzles()
    print(f"📁 Evaluando {len(puzzles)} puzzles oficiales de ARC")
    print()
    
    # Inicializar solver
    solver = ARCSolver()
    
    results = []
    exact_matches = 0
    total_accuracy = []
    
    print("🔍 Evaluando puzzles...")
    print("-" * 80)
    
    for i, puzzle in enumerate(puzzles, 1):
        print(f"{i:3}. {puzzle['name'][:30]:30} ", end="")
        
        result = evaluate_puzzle(solver, puzzle['data'])
        results.append(result)
        
        if result['success']:
            if result['accuracy'] is not None:
                if result['exact_match']:
                    print(f"✅ 100.0%")
                    exact_matches += 1
                elif result['accuracy'] >= 90:
                    print(f"🔶 {result['accuracy']:5.1f}%")
                elif result['accuracy'] > 0:
                    print(f"⚠️  {result['accuracy']:5.1f}%")
                else:
                    print(f"❌   0.0%")
                
                total_accuracy.append(result['accuracy'])
                
                # Mostrar primera inferencia
                if result['inferences']:
                    inf = result['inferences'][0]
                    print(f"     [{inf['level']}] {inf['conclusion'][:50]}")
            else:
                print("✅ Resuelto")
        else:
            print(f"❌ Error: {result['error'][:30]}")
    
    print()
    print("=" * 80)
    print("📊 RESULTADOS FINALES")
    print("=" * 80)
    
    if total_accuracy:
        avg_accuracy = np.mean(total_accuracy)
        print(f"✅ Puzzles perfectos: {exact_matches}/{len(puzzles)} ({exact_matches/len(puzzles)*100:.1f}%)")
        print(f"📈 Accuracy promedio: {avg_accuracy:.1f}%")
        print(f"🎯 Accuracy máxima: {max(total_accuracy):.1f}%")
        print(f"📉 Accuracy mínima: {min(total_accuracy):.1f}%")
    
    print()
    print("🔍 Análisis de Razonamiento:")
    
    # Contar tipos de inferencias
    inference_types = {'macro': 0, 'meso': 0, 'micro': 0}
    for result in results:
        if result['success'] and result['inferences']:
            for inf in result['inferences']:
                inference_types[inf['level']] += 1
    
    print(f"  - Inferencias MACRO: {inference_types['macro']}")
    print(f"  - Inferencias MESO: {inference_types['meso']}")
    print(f"  - Inferencias MICRO: {inference_types['micro']}")

if __name__ == "__main__":
    main()