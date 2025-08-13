#!/usr/bin/env python3
"""
Prueba del detector de objetos con puzzles ARC reales
"""

import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List

from object_detector_yolo import ARCObjectDetector, DetectedObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARCSolverWithObjectDetection:
    """
    Solver ARC que usa detección de objetos REAL
    """
    
    def __init__(self):
        self.detector = ARCObjectDetector()
        self.learned_transformations = []
        
    def analyze_puzzle(self, train_examples: List[Dict]) -> Dict:
        """
        Analiza un puzzle detectando objetos en cada ejemplo
        """
        analysis = {
            'object_counts': [],
            'shape_types': [],
            'transformations': []
        }
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Detectar objetos en input y output
            input_objects = self.detector.detect(input_grid)
            output_objects = self.detector.detect(output_grid)
            
            logger.info(f"\nEjemplo {i+1}:")
            logger.info(f"  Input: {len(input_objects)} objetos detectados")
            logger.info(f"  Output: {len(output_objects)} objetos detectados")
            
            # Analizar tipos de formas
            input_shapes = [obj.shape_type for obj in input_objects]
            output_shapes = [obj.shape_type for obj in output_objects]
            
            logger.info(f"  Formas en input: {input_shapes}")
            logger.info(f"  Formas en output: {output_shapes}")
            
            # Detectar transformación
            transformation = self._infer_transformation(
                input_objects, output_objects,
                input_grid, output_grid
            )
            
            analysis['object_counts'].append({
                'input': len(input_objects),
                'output': len(output_objects)
            })
            analysis['shape_types'].append({
                'input': input_shapes,
                'output': output_shapes
            })
            analysis['transformations'].append(transformation)
            
        return analysis
    
    def _infer_transformation(self, input_objects: List[DetectedObject],
                             output_objects: List[DetectedObject],
                             input_grid: np.ndarray,
                             output_grid: np.ndarray) -> str:
        """
        Infiere qué transformación ocurrió basándose en los objetos detectados
        """
        # Caso 1: Mismo número de objetos - posible transformación 1-a-1
        if len(input_objects) == len(output_objects):
            # Verificar si los objetos cambiaron de posición
            input_centers = [obj.center for obj in input_objects]
            output_centers = [obj.center for obj in output_objects]
            
            # Verificar si es una rotación
            if input_grid.shape == output_grid.shape:
                for k in [1, 2, 3]:
                    if np.array_equal(np.rot90(input_grid, k), output_grid):
                        return f"rotation_{k*90}"
                
                if np.array_equal(np.fliplr(input_grid), output_grid):
                    return "flip_horizontal"
                if np.array_equal(np.flipud(input_grid), output_grid):
                    return "flip_vertical"
            
            # Verificar si los objetos cambiaron de color
            input_colors = sorted([obj.color for obj in input_objects])
            output_colors = sorted([obj.color for obj in output_objects])
            
            if input_colors != output_colors:
                return "color_change"
            
            # Verificar si los objetos se movieron
            if input_centers != output_centers:
                return "object_movement"
        
        # Caso 2: Diferente número de objetos
        elif len(output_objects) > len(input_objects):
            return "object_duplication"
        elif len(output_objects) < len(input_objects):
            return "object_removal"
        
        # Caso 3: Cambio de dimensiones
        if input_grid.shape != output_grid.shape:
            h_ratio = output_grid.shape[0] / input_grid.shape[0]
            w_ratio = output_grid.shape[1] / input_grid.shape[1]
            
            if h_ratio == 2 and w_ratio == 2:
                return "scale_2x"
            elif h_ratio == 0.5 and w_ratio == 0.5:
                return "scale_half"
            else:
                return f"resize_to_{output_grid.shape}"
        
        return "complex_transformation"
    
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """
        Resuelve el puzzle usando detección de objetos
        """
        # Analizar ejemplos de entrenamiento
        analysis = self.analyze_puzzle(train_examples)
        
        # Detectar transformación más común
        transformations = analysis['transformations']
        most_common = max(set(transformations), key=transformations.count)
        
        logger.info(f"\nTransformación detectada: {most_common}")
        
        # Detectar objetos en test
        test_objects = self.detector.detect(test_input)
        logger.info(f"Test input: {len(test_objects)} objetos detectados")
        
        # Aplicar transformación
        if most_common.startswith("rotation"):
            angle = int(most_common.split("_")[1])
            return np.rot90(test_input, angle // 90)
        elif most_common == "flip_horizontal":
            return np.fliplr(test_input)
        elif most_common == "flip_vertical":
            return np.flipud(test_input)
        elif most_common == "scale_2x":
            return np.repeat(np.repeat(test_input, 2, axis=0), 2, axis=1)
        elif most_common == "color_change":
            # Aplicar mapeo de color del primer ejemplo
            if train_examples:
                return self._apply_color_mapping(
                    test_input,
                    np.array(train_examples[0]['input']),
                    np.array(train_examples[0]['output'])
                )
        
        # Si no podemos determinar la transformación, devolver input
        return test_input
    
    def _apply_color_mapping(self, test_grid: np.ndarray,
                            ref_input: np.ndarray,
                            ref_output: np.ndarray) -> np.ndarray:
        """Aplica mapeo de colores"""
        result = test_grid.copy()
        
        # Aprender mapeo
        color_map = {}
        for color in np.unique(ref_input):
            mask = ref_input == color
            if np.any(mask) and mask.shape == ref_output.shape:
                output_colors = ref_output[mask]
                if len(output_colors) > 0:
                    unique, counts = np.unique(output_colors, return_counts=True)
                    color_map[color] = unique[np.argmax(counts)]
        
        # Aplicar
        for old_color, new_color in color_map.items():
            result[test_grid == old_color] = new_color
        
        return result


def test_with_arc_puzzles():
    """Prueba con puzzles ARC reales"""
    
    logger.info("="*60)
    logger.info("PRUEBA DE DETECCIÓN DE OBJETOS EN PUZZLES ARC")
    logger.info("="*60)
    
    solver = ARCSolverWithObjectDetection()
    
    # Cargar algunos puzzles
    cache_dir = Path("/app/arc_official_cache")
    puzzle_files = list(cache_dir.glob("arc_agi_1_training_*.json"))[:3]
    
    results = []
    
    for puzzle_file in puzzle_files:
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        
        puzzle_id = puzzle_file.stem.replace("arc_agi_1_training_", "")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PUZZLE: {puzzle_id}")
        logger.info("="*60)
        
        train_examples = puzzle_data.get('train', [])
        test_cases = puzzle_data.get('test', [])
        
        if test_cases:
            test_input = np.array(test_cases[0]['input'])
            
            # Resolver
            solution = solver.solve(train_examples, test_input)
            
            # Verificar
            if 'output' in test_cases[0]:
                correct_output = np.array(test_cases[0]['output'])
                is_correct = np.array_equal(solution, correct_output)
                
                if solution.shape == correct_output.shape:
                    accuracy = np.mean(solution == correct_output)
                else:
                    accuracy = 0.0
                
                logger.info(f"\n✅ Correcto: {'SÍ' if is_correct else 'NO'}")
                logger.info(f"📊 Precisión: {accuracy:.1%}")
                
                results.append({
                    'puzzle_id': puzzle_id,
                    'correct': is_correct,
                    'accuracy': accuracy
                })
    
    # Resumen
    if results:
        logger.info(f"\n{'='*60}")
        logger.info("RESUMEN:")
        correct_count = sum(1 for r in results if r['correct'])
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        
        logger.info(f"Puzzles correctos: {correct_count}/{len(results)}")
        logger.info(f"Precisión promedio: {avg_accuracy:.1%}")


if __name__ == "__main__":
    test_with_arc_puzzles()