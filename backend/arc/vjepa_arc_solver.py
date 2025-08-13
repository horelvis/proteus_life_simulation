#!/usr/bin/env python3
"""
V-JEPA para ARC - Versión que cumple las reglas de competencia
NO pre-entrenamiento, aprende solo del puzzle actual
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PuzzleTransformation:
    """Transformación aprendida del puzzle actual"""
    example_idx: int
    input_pattern: np.ndarray
    output_pattern: np.ndarray
    transformation_vector: np.ndarray
    confidence: float

class VJEPAARCSolver:
    """
    V-JEPA para resolver puzzles ARC siguiendo las reglas:
    - NO pre-entrenamiento
    - Solo aprende de los ejemplos del puzzle actual
    - Se reinicia para cada puzzle nuevo
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reinicia completamente para un nuevo puzzle"""
        self.current_puzzle_id = None
        self.learned_transformations = []
        self.embedding_dim = 128
        self.transformation_embeddings = []
        self.confidence = 0.0
        logger.info("V-JEPA reiniciado - memoria limpia para nuevo puzzle")
    
    def start_new_puzzle(self, puzzle_id: str):
        """Inicia un nuevo puzzle - borra todo lo anterior"""
        if self.current_puzzle_id != puzzle_id:
            self.reset()
            self.current_puzzle_id = puzzle_id
            logger.info(f"Iniciando nuevo puzzle: {puzzle_id}")
    
    def learn_from_examples(self, train_examples: List[Dict]) -> Dict:
        """
        Aprende SOLO de los ejemplos de entrenamiento del puzzle actual
        Cumple con las reglas de ARC - no usa conocimiento previo
        """
        results = {
            'examples_learned': 0,
            'pattern_detected': None,
            'confidence': 0.0,
            'transformations': []
        }
        
        if not train_examples:
            return results
        
        # Analizar cada ejemplo de entrenamiento
        for idx, example in enumerate(train_examples):
            input_grid = np.array(example.get('input', []))
            output_grid = np.array(example.get('output', []))
            
            if input_grid.size == 0 or output_grid.size == 0:
                continue
            
            # Extraer características y transformación
            transformation = self._analyze_transformation(input_grid, output_grid, idx)
            self.learned_transformations.append(transformation)
            results['examples_learned'] += 1
        
        # Detectar patrón común en las transformaciones
        if self.learned_transformations:
            pattern = self._find_common_pattern()
            results['pattern_detected'] = pattern
            results['confidence'] = self.confidence
            
            # Información sobre las transformaciones aprendidas
            results['transformations'] = [
                {
                    'example': t.example_idx,
                    'confidence': t.confidence,
                    'type': self._classify_transformation_type(t)
                }
                for t in self.learned_transformations
            ]
        
        return results
    
    def predict(self, test_input: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predice la salida basándose SOLO en lo aprendido del puzzle actual
        """
        if not self.learned_transformations:
            # Sin ejemplos de entrenamiento, no podemos predecir
            return test_input.copy(), 0.0
        
        # Encontrar la transformación más consistente
        best_transformation = self._get_best_transformation()
        
        if best_transformation is None:
            return test_input.copy(), 0.0
        
        # Aplicar la transformación al test
        predicted_output = self._apply_transformation(
            test_input, 
            best_transformation
        )
        
        return predicted_output, self.confidence
    
    def _analyze_transformation(self, input_grid: np.ndarray, 
                               output_grid: np.ndarray, idx: int) -> PuzzleTransformation:
        """Analiza una transformación individual"""
        
        # Crear embedding de la transformación
        input_features = self._extract_features(input_grid)
        output_features = self._extract_features(output_grid)
        
        # Vector de transformación en espacio de características
        transformation_vector = output_features - input_features
        
        # Calcular confianza inicial basada en la consistencia
        confidence = self._calculate_confidence(input_grid, output_grid)
        
        return PuzzleTransformation(
            example_idx=idx,
            input_pattern=input_grid,
            output_pattern=output_grid,
            transformation_vector=transformation_vector,
            confidence=confidence
        )
    
    def _extract_features(self, grid: np.ndarray) -> np.ndarray:
        """
        Extrae características de un grid
        Inspirado en V-JEPA pero sin pre-entrenamiento
        """
        features = np.zeros(self.embedding_dim)
        h, w = grid.shape if len(grid.shape) == 2 else (grid.shape[0], grid.shape[1])
        
        # Características básicas (0-9)
        features[0] = h / 30.0  # Altura normalizada
        features[1] = w / 30.0  # Ancho normalizado
        features[2] = np.mean(grid)  # Valor medio
        features[3] = np.std(grid)  # Desviación estándar
        features[4] = np.count_nonzero(grid) / grid.size  # Densidad
        
        # Distribución de valores (10-19)
        unique_vals, counts = np.unique(grid, return_counts=True)
        for i, (val, count) in enumerate(zip(unique_vals[:10], counts[:10])):
            features[10 + i] = count / grid.size
        
        # Patrones espaciales - dividir en regiones (20-35)
        regions = self._divide_into_regions(grid, 4)
        for i, region in enumerate(regions[:4]):
            if region.size > 0:
                features[20 + i*4] = np.mean(region)
                features[21 + i*4] = np.std(region)
                features[22 + i*4] = np.count_nonzero(region) / region.size
                features[23 + i*4] = len(np.unique(region))
        
        # Simetría y patrones (36-45)
        features[36] = self._check_horizontal_symmetry(grid)
        features[37] = self._check_vertical_symmetry(grid)
        features[38] = self._check_diagonal_symmetry(grid)
        features[39] = self._check_rotation_symmetry(grid)
        
        # Conectividad y estructura (46-55)
        features[46] = self._count_connected_components(grid)
        features[47] = self._measure_edge_density(grid)
        
        return features
    
    def _divide_into_regions(self, grid: np.ndarray, n: int) -> List[np.ndarray]:
        """Divide el grid en n×n regiones"""
        h, w = grid.shape
        regions = []
        
        region_h = max(1, h // n)
        region_w = max(1, w // n)
        
        for i in range(n):
            for j in range(n):
                r_start = i * region_h
                r_end = min((i + 1) * region_h, h)
                c_start = j * region_w
                c_end = min((j + 1) * region_w, w)
                
                if r_start < h and c_start < w:
                    regions.append(grid[r_start:r_end, c_start:c_end])
        
        return regions
    
    def _check_horizontal_symmetry(self, grid: np.ndarray) -> float:
        """Verifica simetría horizontal"""
        flipped = np.fliplr(grid)
        return np.sum(grid == flipped) / grid.size
    
    def _check_vertical_symmetry(self, grid: np.ndarray) -> float:
        """Verifica simetría vertical"""
        flipped = np.flipud(grid)
        return np.sum(grid == flipped) / grid.size
    
    def _check_diagonal_symmetry(self, grid: np.ndarray) -> float:
        """Verifica simetría diagonal"""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        return np.sum(grid == grid.T) / grid.size
    
    def _check_rotation_symmetry(self, grid: np.ndarray) -> float:
        """Verifica simetría rotacional"""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        rotated = np.rot90(grid)
        return np.sum(grid == rotated) / grid.size
    
    def _count_connected_components(self, grid: np.ndarray) -> int:
        """Cuenta componentes conectados (simplificado)"""
        # Implementación simplificada
        unique_nonzero = len(np.unique(grid[grid != 0]))
        return min(unique_nonzero, 10) / 10.0  # Normalizado
    
    def _measure_edge_density(self, grid: np.ndarray) -> float:
        """Mide densidad de bordes"""
        h, w = grid.shape
        edges = 0
        
        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0:
                    # Contar transiciones con vecinos
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if grid[ni, nj] != grid[i, j]:
                                edges += 1
        
        max_edges = 2 * h * w  # Máximo teórico
        return edges / max_edges if max_edges > 0 else 0
    
    def _calculate_confidence(self, input_grid: np.ndarray, 
                            output_grid: np.ndarray) -> float:
        """Calcula confianza inicial de una transformación"""
        # Confianza basada en la regularidad de la transformación
        confidence = 0.5
        
        # Bonus si mantienen el mismo tamaño
        if input_grid.shape == output_grid.shape:
            confidence += 0.1
        
        # Bonus si hay una relación clara en los valores
        input_vals = set(input_grid.flatten())
        output_vals = set(output_grid.flatten())
        
        if len(output_vals) <= len(input_vals):
            confidence += 0.1
        
        return min(confidence, 0.9)
    
    def _find_common_pattern(self) -> str:
        """Encuentra el patrón común en las transformaciones aprendidas"""
        if not self.learned_transformations:
            return "no_pattern"
        
        # Analizar vectores de transformación
        vectors = [t.transformation_vector for t in self.learned_transformations]
        
        # Calcular similitud entre vectores
        avg_vector = np.mean(vectors, axis=0)
        variances = np.var(vectors, axis=0)
        
        # Determinar consistencia
        low_variance_ratio = np.sum(variances < 0.1) / len(variances)
        
        if low_variance_ratio > 0.7:
            self.confidence = 0.8
            return "consistent_transformation"
        elif low_variance_ratio > 0.4:
            self.confidence = 0.5
            return "partial_pattern"
        else:
            self.confidence = 0.3
            return "variable_transformation"
    
    def _get_best_transformation(self) -> Optional[PuzzleTransformation]:
        """Obtiene la mejor transformación aprendida"""
        if not self.learned_transformations:
            return None
        
        # Retornar la transformación con mayor confianza
        return max(self.learned_transformations, key=lambda t: t.confidence)
    
    def _apply_transformation(self, input_grid: np.ndarray, 
                             transformation: PuzzleTransformation) -> np.ndarray:
        """Aplica una transformación aprendida a un nuevo input"""
        
        # Estrategia 1: Si es del mismo tamaño, aplicar transformación directa
        if input_grid.shape == transformation.input_pattern.shape:
            # Buscar mapeo directo
            return self._apply_direct_mapping(
                input_grid, 
                transformation.input_pattern,
                transformation.output_pattern
            )
        
        # Estrategia 2: Aplicar transformación por características
        input_features = self._extract_features(input_grid)
        predicted_features = input_features + transformation.transformation_vector
        
        # Reconstruir desde características (simplificado)
        return self._reconstruct_from_features(
            input_grid, 
            predicted_features,
            transformation.output_pattern
        )
    
    def _apply_direct_mapping(self, input_grid: np.ndarray,
                            ref_input: np.ndarray,
                            ref_output: np.ndarray) -> np.ndarray:
        """Aplica mapeo directo basado en ejemplo de referencia"""
        output = input_grid.copy()
        
        # Crear mapeo de colores
        color_map = {}
        for val in np.unique(ref_input):
            # Encontrar a qué se transforma este valor
            mask = ref_input == val
            if np.any(mask):
                output_vals = ref_output[mask]
                if len(output_vals) > 0:
                    # Usar el valor más común
                    unique, counts = np.unique(output_vals, return_counts=True)
                    color_map[val] = unique[np.argmax(counts)]
        
        # Aplicar mapeo
        for old_val, new_val in color_map.items():
            output[input_grid == old_val] = new_val
        
        return output
    
    def _reconstruct_from_features(self, input_grid: np.ndarray,
                                  features: np.ndarray,
                                  reference: np.ndarray) -> np.ndarray:
        """Reconstruye un grid desde características (simplificado)"""
        # Por ahora, retornar una copia del reference si tiene el mismo tamaño
        if reference.shape == input_grid.shape:
            return reference.copy()
        
        # Si no, retornar el input sin cambios
        return input_grid.copy()
    
    def _classify_transformation_type(self, transformation: PuzzleTransformation) -> str:
        """Clasifica el tipo de transformación detectada"""
        input_g = transformation.input_pattern
        output_g = transformation.output_pattern
        
        # Verificar tipos comunes
        if input_g.shape != output_g.shape:
            return "size_change"
        
        if set(input_g.flatten()) != set(output_g.flatten()):
            return "color_change"
        
        if np.array_equal(np.rot90(input_g), output_g):
            return "rotation"
        
        if np.array_equal(np.fliplr(input_g), output_g):
            return "horizontal_flip"
        
        if np.array_equal(np.flipud(input_g), output_g):
            return "vertical_flip"
        
        return "complex"