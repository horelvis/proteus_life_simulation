#!/usr/bin/env python3
"""
Módulo de aumentación de datos para ARC puzzles
Mejora la generalización aplicando transformaciones que preservan la lógica
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from enum import Enum

class AugmentationType(Enum):
    """Tipos de aumentación disponibles"""
    TRANSLATION = "translation"
    COLOR_PERMUTATION = "color_permutation"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    NOISE = "noise"

class ARCAugmentation:
    """
    Aplica aumentaciones a puzzles ARC para mejorar generalización
    """
    
    def __init__(self):
        self.augmentation_history = []
    
    def augment_puzzle(self, 
                      puzzle: Dict[str, Any], 
                      augmentation_types: List[AugmentationType] = None) -> List[Dict[str, Any]]:
        """
        Genera versiones aumentadas de un puzzle
        
        Args:
            puzzle: Puzzle original con 'input' y 'output'
            augmentation_types: Lista de tipos a aplicar
            
        Returns:
            Lista de puzzles aumentados
        """
        if augmentation_types is None:
            augmentation_types = [
                AugmentationType.TRANSLATION,
                AugmentationType.COLOR_PERMUTATION,
                AugmentationType.ROTATION,
                AugmentationType.REFLECTION
            ]
        
        augmented_puzzles = []
        
        for aug_type in augmentation_types:
            try:
                if aug_type == AugmentationType.TRANSLATION:
                    augmented = self._apply_translation(puzzle)
                elif aug_type == AugmentationType.COLOR_PERMUTATION:
                    augmented = self._apply_color_permutation(puzzle)
                elif aug_type == AugmentationType.ROTATION:
                    augmented = self._apply_rotation(puzzle)
                elif aug_type == AugmentationType.REFLECTION:
                    augmented = self._apply_reflection(puzzle)
                elif aug_type == AugmentationType.NOISE:
                    augmented = self._apply_noise(puzzle)
                else:
                    continue
                
                if augmented:
                    augmented['augmentation'] = aug_type.value
                    augmented_puzzles.append(augmented)
                    
            except Exception as e:
                print(f"Error aplicando {aug_type.value}: {e}")
                continue
        
        return augmented_puzzles
    
    def _apply_translation(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica traslación a un puzzle
        """
        input_grid = np.array(puzzle['input'])
        output_grid = np.array(puzzle['output'])
        
        # Calcular máximo desplazamiento posible
        h, w = input_grid.shape
        max_dx = h // 4
        max_dy = w // 4
        
        # Generar desplazamientos aleatorios
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        # Aplicar traslación
        translated_input = self._translate_grid(input_grid, dx, dy)
        translated_output = self._translate_grid(output_grid, dx, dy)
        
        return {
            'input': translated_input.tolist(),
            'output': translated_output.tolist(),
            'original_puzzle': puzzle,
            'translation': {'dx': dx, 'dy': dy}
        }
    
    def _translate_grid(self, grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """
        Traslada una grilla por (dx, dy)
        """
        h, w = grid.shape
        translated = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                new_i = i + dx
                new_j = j + dy
                
                if 0 <= new_i < h and 0 <= new_j < w:
                    translated[new_i, new_j] = grid[i, j]
        
        return translated
    
    def _apply_color_permutation(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica permutación de colores preservando el 0 (fondo)
        """
        input_grid = np.array(puzzle['input'])
        output_grid = np.array(puzzle['output'])
        
        # Obtener colores únicos (excluyendo 0)
        unique_colors = set()
        unique_colors.update(input_grid.flatten())
        unique_colors.update(output_grid.flatten())
        unique_colors.discard(0)  # Mantener 0 como fondo
        
        if len(unique_colors) < 2:
            return None  # No hay suficientes colores para permutar
        
        # Crear mapeo de colores
        color_list = list(unique_colors)
        shuffled_colors = color_list.copy()
        random.shuffle(shuffled_colors)
        
        color_map = {0: 0}  # El fondo siempre es 0
        for original, new in zip(color_list, shuffled_colors):
            color_map[original] = new
        
        # Aplicar permutación
        permuted_input = self._apply_color_map(input_grid, color_map)
        permuted_output = self._apply_color_map(output_grid, color_map)
        
        return {
            'input': permuted_input.tolist(),
            'output': permuted_output.tolist(),
            'original_puzzle': puzzle,
            'color_map': color_map
        }
    
    def _apply_color_map(self, grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
        """
        Aplica un mapeo de colores a una grilla
        """
        result = np.zeros_like(grid)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                original_color = grid[i, j]
                result[i, j] = color_map.get(original_color, original_color)
        
        return result
    
    def _apply_rotation(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica rotación de 90, 180 o 270 grados
        """
        input_grid = np.array(puzzle['input'])
        output_grid = np.array(puzzle['output'])
        
        # Elegir rotación aleatoria
        k = random.choice([1, 2, 3])  # 90, 180, 270 grados
        
        # Aplicar rotación
        rotated_input = np.rot90(input_grid, k)
        rotated_output = np.rot90(output_grid, k)
        
        return {
            'input': rotated_input.tolist(),
            'output': rotated_output.tolist(),
            'original_puzzle': puzzle,
            'rotation': k * 90
        }
    
    def _apply_reflection(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica reflexión horizontal o vertical
        """
        input_grid = np.array(puzzle['input'])
        output_grid = np.array(puzzle['output'])
        
        # Elegir tipo de reflexión
        axis = random.choice([0, 1])  # 0: vertical, 1: horizontal
        
        # Aplicar reflexión
        reflected_input = np.flip(input_grid, axis)
        reflected_output = np.flip(output_grid, axis)
        
        return {
            'input': reflected_input.tolist(),
            'output': reflected_output.tolist(),
            'original_puzzle': puzzle,
            'reflection_axis': 'vertical' if axis == 0 else 'horizontal'
        }
    
    def _apply_noise(self, puzzle: Dict[str, Any], noise_level: float = 0.05) -> Dict[str, Any]:
        """
        Aplica ruido aleatorio (cambiar algunos píxeles)
        Útil para evaluar robustez
        """
        input_grid = np.array(puzzle['input'])
        output_grid = np.array(puzzle['output'])
        
        # Obtener colores disponibles
        unique_colors = list(set(input_grid.flatten()))
        
        # Aplicar ruido solo al input
        noisy_input = input_grid.copy()
        h, w = noisy_input.shape
        num_pixels = int(h * w * noise_level)
        
        for _ in range(num_pixels):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            # No cambiar si ya es el color de fondo
            if noisy_input[i, j] != 0:
                noisy_input[i, j] = random.choice(unique_colors)
        
        return {
            'input': noisy_input.tolist(),
            'output': output_grid.tolist(),  # Output sin cambios
            'original_puzzle': puzzle,
            'noise_level': noise_level
        }
    
    def validate_augmentation(self, 
                            original_puzzle: Dict[str, Any],
                            augmented_puzzle: Dict[str, Any],
                            solver) -> bool:
        """
        Valida que la aumentación preserve la lógica del puzzle
        
        Args:
            original_puzzle: Puzzle original
            augmented_puzzle: Puzzle aumentado
            solver: Instancia del solver ARC
            
        Returns:
            True si la aumentación es válida
        """
        try:
            # El solver debe detectar la misma regla en ambos
            original_rule = solver.detect_rule(
                np.array(original_puzzle['input']), 
                np.array(original_puzzle['output'])
            )
            
            augmented_rule = solver.detect_rule(
                np.array(augmented_puzzle['input']),
                np.array(augmented_puzzle['output'])
            )
            
            # Las reglas deben ser del mismo tipo
            if original_rule and augmented_rule:
                return original_rule['type'] == augmented_rule['type']
            
            return False
            
        except Exception as e:
            print(f"Error validando aumentación: {e}")
            return False
    
    def create_augmented_dataset(self,
                               puzzles: List[Dict[str, Any]],
                               augmentations_per_puzzle: int = 3) -> List[Dict[str, Any]]:
        """
        Crea un dataset aumentado a partir de puzzles originales
        
        Args:
            puzzles: Lista de puzzles originales
            augmentations_per_puzzle: Número de aumentaciones por puzzle
            
        Returns:
            Dataset aumentado
        """
        augmented_dataset = []
        
        for puzzle in puzzles:
            # Agregar puzzle original
            augmented_dataset.append(puzzle)
            
            # Generar aumentaciones
            for _ in range(augmentations_per_puzzle):
                # Elegir tipo de aumentación aleatoriamente
                aug_type = random.choice(list(AugmentationType))
                augmented = self.augment_puzzle(puzzle, [aug_type])
                
                if augmented:
                    augmented_dataset.extend(augmented)
        
        # Mezclar dataset
        random.shuffle(augmented_dataset)
        
        print(f"Dataset original: {len(puzzles)} puzzles")
        print(f"Dataset aumentado: {len(augmented_dataset)} puzzles")
        
        return augmented_dataset
    
    def analyze_augmentation_impact(self,
                                  original_puzzles: List[Dict[str, Any]],
                                  solver,
                                  augmentation_types: List[AugmentationType] = None) -> Dict[str, Any]:
        """
        Analiza el impacto de diferentes tipos de aumentación
        
        Args:
            original_puzzles: Puzzles originales
            solver: Instancia del solver
            augmentation_types: Tipos a analizar
            
        Returns:
            Análisis del impacto
        """
        if augmentation_types is None:
            augmentation_types = list(AugmentationType)
        
        results = {
            'baseline_accuracy': 0,
            'augmentation_impact': {}
        }
        
        # Evaluar baseline (sin aumentación)
        correct = 0
        for puzzle in original_puzzles:
            try:
                if 'trainExamples' in puzzle:
                    solver.aprender(puzzle['trainExamples'])
                    solution = solver.resolver(puzzle['input'])
                    if np.array_equal(solution, puzzle['output']):
                        correct += 1
            except:
                pass
        
        results['baseline_accuracy'] = correct / len(original_puzzles) if original_puzzles else 0
        
        # Evaluar cada tipo de aumentación
        for aug_type in augmentation_types:
            correct_augmented = 0
            total_augmented = 0
            
            for puzzle in original_puzzles:
                augmented_puzzles = self.augment_puzzle(puzzle, [aug_type])
                
                for aug_puzzle in augmented_puzzles:
                    total_augmented += 1
                    try:
                        if 'trainExamples' in puzzle:
                            # Entrenar con ejemplos aumentados
                            aug_examples = []
                            for example in puzzle['trainExamples']:
                                aug_ex = self.augment_puzzle(example, [aug_type])
                                if aug_ex:
                                    aug_examples.extend(aug_ex)
                            
                            if aug_examples:
                                solver.aprender(aug_examples)
                                solution = solver.resolver(aug_puzzle['input'])
                                if np.array_equal(solution, aug_puzzle['output']):
                                    correct_augmented += 1
                    except:
                        pass
            
            accuracy = correct_augmented / total_augmented if total_augmented > 0 else 0
            results['augmentation_impact'][aug_type.value] = {
                'accuracy': accuracy,
                'improvement': accuracy - results['baseline_accuracy'],
                'total_samples': total_augmented
            }
        
        return results


if __name__ == "__main__":
    # Prueba del módulo
    augmenter = ARCAugmentation()
    
    # Puzzle de ejemplo
    test_puzzle = {
        'input': [
            [0, 0, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0]
        ],
        'output': [
            [0, 0, 3, 0],
            [0, 3, 3, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0]
        ]
    }
    
    print("🧪 Probando aumentaciones...")
    
    # Generar aumentaciones
    augmented = augmenter.augment_puzzle(test_puzzle)
    
    for i, aug_puzzle in enumerate(augmented):
        print(f"\n📊 Aumentación {i+1}: {aug_puzzle.get('augmentation', 'unknown')}")
        print(f"Input shape: {len(aug_puzzle['input'])}x{len(aug_puzzle['input'][0])}")
        
        if 'translation' in aug_puzzle:
            print(f"Translation: dx={aug_puzzle['translation']['dx']}, dy={aug_puzzle['translation']['dy']}")
        elif 'color_map' in aug_puzzle:
            print(f"Color map: {aug_puzzle['color_map']}")
        elif 'rotation' in aug_puzzle:
            print(f"Rotation: {aug_puzzle['rotation']}°")
        elif 'reflection_axis' in aug_puzzle:
            print(f"Reflection: {aug_puzzle['reflection_axis']}")