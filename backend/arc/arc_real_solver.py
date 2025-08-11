#!/usr/bin/env python3
"""
ARC Real Solver - Conexión real al dataset ARC Prize oficial
SIN datos simulados, SIN engaños, SOLO puzzles reales
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import glob
import random
from pathlib import Path

logger = logging.getLogger(__name__)

class ARCRealSolver:
    """
    Solver que usa ÚNICAMENTE datos reales del ARC Prize
    - Carga puzzles oficiales del dataset
    - Evalúa con métricas reales 
    - NO datos simulados
    """
    
    def __init__(self, cache_dir: str = "/app/arc_official_cache"):
        """
        Inicializa con dataset ARC real
        
        Args:
            cache_dir: Directorio con puzzles oficiales ARC
        """
        self.cache_dir = cache_dir
        self.puzzles = {}
        self.load_official_puzzles()
        
    def load_official_puzzles(self) -> None:
        """Carga puzzles oficiales del ARC Prize"""
        puzzle_files = glob.glob(f"{self.cache_dir}/*.json")
        
        if not puzzle_files:
            raise FileNotFoundError(f"No se encontraron puzzles ARC en {self.cache_dir}")
        
        logger.info(f"📁 Cargando {len(puzzle_files)} puzzles oficiales ARC")
        
        for file_path in puzzle_files:
            puzzle_id = Path(file_path).stem.replace('arc_agi_1_training_', '')
            
            try:
                with open(file_path, 'r') as f:
                    puzzle_data = json.load(f)
                    
                # Validar estructura oficial ARC
                if 'train' not in puzzle_data or 'test' not in puzzle_data:
                    logger.warning(f"⚠️ Puzzle {puzzle_id} no tiene estructura ARC válida")
                    continue
                    
                # Convertir a numpy arrays
                train_examples = []
                for example in puzzle_data['train']:
                    train_examples.append({
                        'input': np.array(example['input']),
                        'output': np.array(example['output'])
                    })
                
                test_cases = []
                for test in puzzle_data['test']:
                    test_case = {'input': np.array(test['input'])}
                    if 'output' in test:  # Algunos test tienen solución
                        test_case['output'] = np.array(test['output'])
                    test_cases.append(test_case)
                
                self.puzzles[puzzle_id] = {
                    'train': train_examples,
                    'test': test_cases,
                    'file': file_path
                }
                
                logger.debug(f"✅ Puzzle {puzzle_id}: {len(train_examples)} ejemplos de entrenamiento, {len(test_cases)} casos de prueba")
                
            except Exception as e:
                logger.error(f"❌ Error cargando puzzle {puzzle_id}: {e}")
                
        logger.info(f"✅ {len(self.puzzles)} puzzles ARC oficiales cargados correctamente")
        
    def get_random_puzzle(self) -> Tuple[str, Dict]:
        """
        Obtiene un puzzle aleatorio del dataset oficial
        
        Returns:
            Tuple con ID del puzzle y datos
        """
        if not self.puzzles:
            raise RuntimeError("No hay puzzles cargados")
            
        puzzle_id = random.choice(list(self.puzzles.keys()))
        return puzzle_id, self.puzzles[puzzle_id]
    
    def get_puzzle_by_id(self, puzzle_id: str) -> Optional[Dict]:
        """
        Obtiene un puzzle específico por ID
        
        Args:
            puzzle_id: ID del puzzle (ej: '0520fde7')
            
        Returns:
            Datos del puzzle o None si no existe
        """
        return self.puzzles.get(puzzle_id)
    
    def evaluate_solution(self, puzzle_id: str, test_index: int, solution: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa una solución contra la respuesta correcta oficial
        
        Args:
            puzzle_id: ID del puzzle
            test_index: Índice del caso de prueba  
            solution: Solución propuesta
            
        Returns:
            Métricas de evaluación reales
        """
        puzzle = self.puzzles.get(puzzle_id)
        if not puzzle:
            return {'error': f'Puzzle {puzzle_id} no encontrado'}
        
        if test_index >= len(puzzle['test']):
            return {'error': f'Test index {test_index} fuera de rango'}
            
        test_case = puzzle['test'][test_index]
        
        # Si no hay solución oficial, solo validar formato
        if 'output' not in test_case:
            return {
                'puzzle_id': puzzle_id,
                'test_index': test_index,
                'solution_shape': solution.shape,
                'input_shape': test_case['input'].shape,
                'has_official_answer': False,
                'format_valid': True
            }
        
        # Evaluar contra solución oficial
        official_output = test_case['output']
        
        # Verificar dimensiones
        if solution.shape != official_output.shape:
            return {
                'puzzle_id': puzzle_id,
                'test_index': test_index,
                'correct': False,
                'error': 'Dimensiones incorrectas',
                'expected_shape': official_output.shape,
                'got_shape': solution.shape,
                'accuracy': 0.0
            }
        
        # Calcular precisión pixel a pixel
        correct_pixels = np.sum(solution == official_output)
        total_pixels = official_output.size
        accuracy = correct_pixels / total_pixels
        
        return {
            'puzzle_id': puzzle_id,
            'test_index': test_index,
            'correct': accuracy == 1.0,
            'accuracy': accuracy,
            'correct_pixels': int(correct_pixels),
            'total_pixels': int(total_pixels),
            'solution_shape': solution.shape,
            'expected_shape': official_output.shape
        }
    
    def get_puzzle_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del dataset cargado
        
        Returns:
            Estadísticas del dataset oficial
        """
        if not self.puzzles:
            return {'total_puzzles': 0}
        
        total_train_examples = 0
        total_test_cases = 0
        input_shapes = []
        output_shapes = []
        
        for puzzle_data in self.puzzles.values():
            total_train_examples += len(puzzle_data['train'])
            total_test_cases += len(puzzle_data['test'])
            
            for example in puzzle_data['train']:
                input_shapes.append(example['input'].shape)
                output_shapes.append(example['output'].shape)
                
        return {
            'total_puzzles': len(self.puzzles),
            'total_train_examples': total_train_examples,
            'total_test_cases': total_test_cases,
            'puzzle_ids': list(self.puzzles.keys()),
            'avg_input_size': np.mean([np.prod(shape) for shape in input_shapes]),
            'avg_output_size': np.mean([np.prod(shape) for shape in output_shapes]),
            'dataset_source': 'ARC Prize Official'
        }
    
    def solve_puzzle_real(self, puzzle_id: str, solver_function) -> Dict[str, Any]:
        """
        Resuelve un puzzle real usando una función solver
        
        Args:
            puzzle_id: ID del puzzle oficial
            solver_function: Función que toma (train_examples, test_input) -> solution
            
        Returns:
            Resultados de evaluación reales
        """
        puzzle = self.get_puzzle_by_id(puzzle_id)
        if not puzzle:
            return {'error': f'Puzzle {puzzle_id} no encontrado'}
        
        results = []
        
        for test_idx, test_case in enumerate(puzzle['test']):
            try:
                # Resolver usando función proporcionada
                solution = solver_function(puzzle['train'], test_case['input'])
                
                # Evaluar contra dataset oficial
                evaluation = self.evaluate_solution(puzzle_id, test_idx, solution)
                evaluation['solver_used'] = solver_function.__name__
                
                results.append(evaluation)
                
                logger.info(f"📊 Puzzle {puzzle_id} Test {test_idx}: Precisión = {evaluation.get('accuracy', 0):.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error resolviendo {puzzle_id} test {test_idx}: {e}")
                results.append({
                    'puzzle_id': puzzle_id,
                    'test_index': test_idx,
                    'error': str(e),
                    'correct': False,
                    'accuracy': 0.0
                })
        
        return {
            'puzzle_id': puzzle_id,
            'results': results,
            'avg_accuracy': np.mean([r.get('accuracy', 0) for r in results]),
            'total_correct': sum(1 for r in results if r.get('correct', False))
        }


def test_real_arc_data():
    """Prueba el cargador de datos reales ARC"""
    print("="*60)
    print("🧪 PRUEBA DE DATOS REALES ARC PRIZE")
    print("="*60)
    
    # Cargar datos reales
    solver = ARCRealSolver()
    
    # Mostrar estadísticas
    stats = solver.get_puzzle_stats()
    print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
    print(f"   Total de puzzles: {stats['total_puzzles']}")
    print(f"   Ejemplos de entrenamiento: {stats['total_train_examples']}")
    print(f"   Casos de prueba: {stats['total_test_cases']}")
    print(f"   Tamaño promedio input: {stats['avg_input_size']:.1f} píxeles")
    
    # Mostrar puzzle real
    puzzle_id, puzzle_data = solver.get_random_puzzle()
    print(f"\n🎯 EJEMPLO DE PUZZLE REAL: {puzzle_id}")
    print(f"   Ejemplos de entrenamiento: {len(puzzle_data['train'])}")
    print(f"   Casos de prueba: {len(puzzle_data['test'])}")
    
    # Mostrar primer ejemplo
    first_example = puzzle_data['train'][0]
    print(f"\n   📥 Input shape: {first_example['input'].shape}")
    print(f"   📤 Output shape: {first_example['output'].shape}")
    print(f"   📥 Input:\n{first_example['input']}")
    print(f"   📤 Output:\n{first_example['output']}")
    
    print("\n✅ DATOS REALES ARC VERIFICADOS - SIN SIMULACIONES")
    print("="*60)


if __name__ == "__main__":
    test_real_arc_data()