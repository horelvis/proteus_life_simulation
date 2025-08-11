#!/usr/bin/env python3
"""
Advanced Transformations - Transformaciones complejas para puzzles ARC
Traslaciones+rotaciones, escalado no uniforme, reordenamiento de objetos, 
operaciones aritméticas sobre matrices
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from scipy import ndimage
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)

class AdvancedTransformations:
    """
    Transformaciones avanzadas para resolver puzzles ARC complejos
    Combina múltiples operaciones y maneja casos específicos del corpus ARC
    """
    
    def __init__(self):
        self.transformation_registry = self._build_transformation_registry()
        
    def _build_transformation_registry(self) -> Dict[str, Callable]:
        """Construye registro de transformaciones disponibles"""
        return {
            # Transformaciones combinadas
            'translate_rotate': self.translate_and_rotate,
            'scale_translate': self.scale_and_translate,
            'rotate_scale': self.rotate_and_scale,
            'flip_rotate_translate': self.flip_rotate_translate,
            
            # Escalado avanzado
            'non_uniform_scale': self.non_uniform_scale,
            'object_wise_scale': self.object_wise_scale,
            'conditional_scale': self.conditional_scale,
            
            # Reordenamiento de objetos
            'sort_objects_by_size': self.sort_objects_by_size,
            'sort_objects_by_color': self.sort_objects_by_color,
            'sort_objects_by_position': self.sort_objects_by_position,
            'rearrange_by_pattern': self.rearrange_by_pattern,
            
            # Operaciones aritméticas
            'matrix_add': self.matrix_arithmetic_add,
            'matrix_subtract': self.matrix_arithmetic_subtract,
            'matrix_multiply': self.matrix_arithmetic_multiply,
            'modular_arithmetic': self.modular_arithmetic,
            
            # Transformaciones topológicas
            'connect_nearest': self.connect_nearest_objects,
            'bridge_objects': self.bridge_objects,
            'fill_enclosed': self.fill_enclosed_areas,
            
            # Cambios de paleta avanzados
            'contextual_recolor': self.contextual_recolor,
            'gradient_recolor': self.gradient_recolor,
            'pattern_based_recolor': self.pattern_based_recolor,
            
            # Inserción/eliminación de objetos
            'insert_at_pattern': self.insert_objects_at_pattern,
            'remove_by_criteria': self.remove_objects_by_criteria,
            'duplicate_and_arrange': self.duplicate_and_arrange,
            
            # Transformaciones irregulares
            'irregular_replication': self.irregular_replication,
            'spiral_arrangement': self.spiral_arrangement,
            'fractal_pattern': self.fractal_pattern
        }
    
    def apply_transformation(self, matrix: np.ndarray, transform_name: str, 
                           **params) -> np.ndarray:
        """
        Aplica una transformación específica con parámetros
        
        Args:
            matrix: Matriz de entrada
            transform_name: Nombre de la transformación
            **params: Parámetros específicos de la transformación
            
        Returns:
            Matriz transformada
        """
        if transform_name not in self.transformation_registry:
            logger.warning(f"Unknown transformation: {transform_name}")
            return matrix.copy()
        
        try:
            transform_func = self.transformation_registry[transform_name]
            return transform_func(matrix, **params)
        except Exception as e:
            logger.error(f"Error applying {transform_name}: {e}")
            return matrix.copy()
    
    # TRANSFORMACIONES COMBINADAS
    
    def translate_and_rotate(self, matrix: np.ndarray, 
                           dx: int = 0, dy: int = 0, 
                           angle: int = 90, **kwargs) -> np.ndarray:
        """Combina traslación y rotación"""
        # Primero trasladar
        translated = self._translate_matrix(matrix, dx, dy)
        
        # Luego rotar
        if angle == 90:
            return np.rot90(translated)
        elif angle == 180:
            return np.rot90(translated, 2)
        elif angle == 270:
            return np.rot90(translated, 3)
        else:
            return translated
    
    def scale_and_translate(self, matrix: np.ndarray,
                           scale_x: float = 2.0, scale_y: float = 2.0,
                           dx: int = 0, dy: int = 0, **kwargs) -> np.ndarray:
        """Combina escalado y traslación"""
        # Primero escalar
        scaled = self._scale_matrix(matrix, scale_x, scale_y)
        
        # Luego trasladar
        return self._translate_matrix(scaled, dx, dy)
    
    def rotate_and_scale(self, matrix: np.ndarray,
                        angle: int = 90, scale: float = 2.0, **kwargs) -> np.ndarray:
        """Combina rotación y escalado"""
        # Primero rotar
        if angle == 90:
            rotated = np.rot90(matrix)
        elif angle == 180:
            rotated = np.rot90(matrix, 2)
        elif angle == 270:
            rotated = np.rot90(matrix, 3)
        else:
            rotated = matrix.copy()
        
        # Luego escalar
        return self._scale_matrix(rotated, scale, scale)
    
    def flip_rotate_translate(self, matrix: np.ndarray,
                             flip_axis: int = 0, angle: int = 90,
                             dx: int = 0, dy: int = 0, **kwargs) -> np.ndarray:
        """Combina flip, rotación y traslación"""
        # Primero flip
        if flip_axis == 0:
            flipped = np.flipud(matrix)
        elif flip_axis == 1:
            flipped = np.fliplr(matrix)
        else:
            flipped = matrix.copy()
        
        # Luego rotar
        if angle == 90:
            rotated = np.rot90(flipped)
        elif angle == 180:
            rotated = np.rot90(flipped, 2)
        elif angle == 270:
            rotated = np.rot90(flipped, 3)
        else:
            rotated = flipped
        
        # Finalmente trasladar
        return self._translate_matrix(rotated, dx, dy)
    
    # ESCALADO AVANZADO
    
    def non_uniform_scale(self, matrix: np.ndarray,
                         scale_x: float = 2.0, scale_y: float = 1.5,
                         **kwargs) -> np.ndarray:
        """Escalado no uniforme en X e Y"""
        h, w = matrix.shape
        new_h = int(h * scale_y)
        new_w = int(w * scale_x)
        
        # Crear nueva matriz
        result = np.zeros((new_h, new_w), dtype=matrix.dtype)
        
        # Mapeo de píxeles con interpolación nearest neighbor
        for r in range(new_h):
            for c in range(new_w):
                orig_r = int(r / scale_y)
                orig_c = int(c / scale_x)
                
                if orig_r < h and orig_c < w:
                    result[r, c] = matrix[orig_r, orig_c]
        
        return result
    
    def object_wise_scale(self, matrix: np.ndarray, scale_factor: float = 2.0,
                         **kwargs) -> np.ndarray:
        """Escala cada objeto independientemente"""
        unique_colors = np.unique(matrix)
        result = np.zeros_like(matrix)
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Extraer objetos de este color
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                
                # Escalar objeto individual
                scaled_obj = self._scale_object(obj_mask, scale_factor)
                
                # Colocar en resultado
                result = np.maximum(result, scaled_obj * color)
        
        return result
    
    def conditional_scale(self, matrix: np.ndarray, 
                         condition: str = 'size', threshold: int = 5,
                         scale_factor: float = 2.0, **kwargs) -> np.ndarray:
        """Escala objetos que cumplen una condición"""
        unique_colors = np.unique(matrix)
        result = matrix.copy()
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                
                # Evaluar condición
                if self._evaluate_condition(obj_mask, condition, threshold):
                    # Escalar objeto
                    result = self._replace_object(result, obj_mask, 
                                                self._scale_object(obj_mask, scale_factor) * color)
        
        return result
    
    # REORDENAMIENTO DE OBJETOS
    
    def sort_objects_by_size(self, matrix: np.ndarray, ascending: bool = True,
                            **kwargs) -> np.ndarray:
        """Ordena objetos por tamaño"""
        objects = self._extract_objects(matrix)
        
        # Ordenar por área
        objects.sort(key=lambda obj: obj['area'], reverse=not ascending)
        
        return self._reconstruct_matrix(objects, matrix.shape)
    
    def sort_objects_by_color(self, matrix: np.ndarray, ascending: bool = True,
                             **kwargs) -> np.ndarray:
        """Ordena objetos por color"""
        objects = self._extract_objects(matrix)
        
        # Ordenar por color
        objects.sort(key=lambda obj: obj['color'], reverse=not ascending)
        
        return self._reconstruct_matrix(objects, matrix.shape)
    
    def sort_objects_by_position(self, matrix: np.ndarray, axis: int = 0,
                                ascending: bool = True, **kwargs) -> np.ndarray:
        """Ordena objetos por posición (eje X o Y)"""
        objects = self._extract_objects(matrix)
        
        # Ordenar por centroide
        if axis == 0:  # Ordenar por Y
            objects.sort(key=lambda obj: obj['centroid'][0], reverse=not ascending)
        else:  # Ordenar por X
            objects.sort(key=lambda obj: obj['centroid'][1], reverse=not ascending)
        
        return self._reconstruct_matrix(objects, matrix.shape)
    
    def rearrange_by_pattern(self, matrix: np.ndarray, pattern: str = 'grid',
                            **kwargs) -> np.ndarray:
        """Reorganiza objetos según un patrón específico"""
        objects = self._extract_objects(matrix)
        
        if pattern == 'grid':
            return self._arrange_in_grid(objects, matrix.shape)
        elif pattern == 'circle':
            return self._arrange_in_circle(objects, matrix.shape)
        elif pattern == 'line':
            return self._arrange_in_line(objects, matrix.shape)
        else:
            return matrix.copy()
    
    # OPERACIONES ARITMÉTICAS
    
    def matrix_arithmetic_add(self, matrix: np.ndarray, value: int = 1,
                             **kwargs) -> np.ndarray:
        """Suma aritmética con saturación"""
        result = matrix + value
        result = np.clip(result, 0, 9)  # ARC usa colores 0-9
        return result
    
    def matrix_arithmetic_subtract(self, matrix: np.ndarray, value: int = 1,
                                  **kwargs) -> np.ndarray:
        """Resta aritmética con saturación"""
        result = matrix - value
        result = np.clip(result, 0, 9)
        return result
    
    def matrix_arithmetic_multiply(self, matrix: np.ndarray, value: int = 2,
                                  **kwargs) -> np.ndarray:
        """Multiplicación aritmética con saturación"""
        result = matrix * value
        result = np.clip(result, 0, 9)
        return result
    
    def modular_arithmetic(self, matrix: np.ndarray, modulus: int = 10,
                          **kwargs) -> np.ndarray:
        """Aritmética modular"""
        return matrix % modulus
    
    # TRANSFORMACIONES TOPOLÓGICAS
    
    def connect_nearest_objects(self, matrix: np.ndarray, connection_color: int = 1,
                               **kwargs) -> np.ndarray:
        """Conecta objetos más cercanos con líneas"""
        objects = self._extract_objects(matrix)
        result = matrix.copy()
        
        if len(objects) < 2:
            return result
        
        # Encontrar pares más cercanos
        centroids = [obj['centroid'] for obj in objects]
        distances = cdist(centroids, centroids)
        
        # Conectar cada objeto con el más cercano
        for i, obj in enumerate(objects):
            distances[i, i] = np.inf  # Excluir self-distance
            nearest_idx = np.argmin(distances[i, :])
            
            # Dibujar línea entre centroides
            start = tuple(map(int, obj['centroid']))
            end = tuple(map(int, objects[nearest_idx]['centroid']))
            
            result = self._draw_line(result, start, end, connection_color)
        
        return result
    
    def bridge_objects(self, matrix: np.ndarray, bridge_color: int = 2,
                      **kwargs) -> np.ndarray:
        """Crea puentes entre objetos del mismo color"""
        unique_colors = np.unique(matrix)
        result = matrix.copy()
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            if num_objects > 1:
                # Encontrar centroides de objetos del mismo color
                centroids = []
                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    centroid = ndimage.center_of_mass(obj_mask)
                    centroids.append(centroid)
                
                # Conectar todos con el primero (star topology)
                for i in range(1, len(centroids)):
                    start = tuple(map(int, centroids[0]))
                    end = tuple(map(int, centroids[i]))
                    result = self._draw_line(result, start, end, bridge_color)
        
        return result
    
    def fill_enclosed_areas(self, matrix: np.ndarray, fill_color: int = 1,
                           **kwargs) -> np.ndarray:
        """Rellena áreas encerradas por objetos"""
        result = matrix.copy()
        
        # Encontrar agujeros usando binary_fill_holes
        for color in np.unique(matrix):
            if color == 0:
                continue
            
            mask = (matrix == color)
            filled = ndimage.binary_fill_holes(mask)
            
            # Los agujeros son donde filled es True pero mask es False
            holes = filled & (~mask)
            result[holes] = fill_color
        
        return result
    
    # CAMBIOS DE PALETA AVANZADOS
    
    def contextual_recolor(self, matrix: np.ndarray, color_map: Dict[int, int] = None,
                          **kwargs) -> np.ndarray:
        """Recoloreo basado en contexto local"""
        if color_map is None:
            color_map = {1: 2, 2: 3, 3: 1}  # Default rotation
        
        result = matrix.copy()
        h, w = matrix.shape
        
        # Recolorar basado en vecindario
        for r in range(h):
            for c in range(w):
                current_color = matrix[r, c]
                if current_color in color_map:
                    # Verificar contexto (vecinos)
                    neighbors = self._get_neighbors(matrix, r, c)
                    if self._should_recolor_based_on_context(current_color, neighbors):
                        result[r, c] = color_map[current_color]
        
        return result
    
    def gradient_recolor(self, matrix: np.ndarray, start_color: int = 1,
                        end_color: int = 9, direction: str = 'horizontal',
                        **kwargs) -> np.ndarray:
        """Crea gradiente de colores"""
        result = matrix.copy()
        h, w = matrix.shape
        
        # Aplicar gradiente solo a píxeles no-cero
        non_zero_mask = (matrix != 0)
        
        if direction == 'horizontal':
            for c in range(w):
                progress = c / (w - 1) if w > 1 else 0
                gradient_color = int(start_color + (end_color - start_color) * progress)
                gradient_color = np.clip(gradient_color, 0, 9)
                
                result[non_zero_mask[:, c], c] = gradient_color
                
        elif direction == 'vertical':
            for r in range(h):
                progress = r / (h - 1) if h > 1 else 0
                gradient_color = int(start_color + (end_color - start_color) * progress)
                gradient_color = np.clip(gradient_color, 0, 9)
                
                result[r, non_zero_mask[r, :]] = gradient_color
        
        return result
    
    def pattern_based_recolor(self, matrix: np.ndarray, pattern: str = 'checkerboard',
                             colors: List[int] = [1, 2], **kwargs) -> np.ndarray:
        """Recolorea basado en patrones geométricos"""
        result = matrix.copy()
        h, w = matrix.shape
        
        non_zero_mask = (matrix != 0)
        
        if pattern == 'checkerboard':
            for r in range(h):
                for c in range(w):
                    if non_zero_mask[r, c]:
                        color_idx = (r + c) % len(colors)
                        result[r, c] = colors[color_idx]
                        
        elif pattern == 'stripes':
            for r in range(h):
                color_idx = r % len(colors)
                result[r, non_zero_mask[r, :]] = colors[color_idx]
        
        return result
    
    # INSERCIÓN/ELIMINACIÓN DE OBJETOS
    
    def insert_objects_at_pattern(self, matrix: np.ndarray, object_color: int = 5,
                                 pattern_positions: List[Tuple[int, int]] = None,
                                 **kwargs) -> np.ndarray:
        """Inserta objetos en posiciones específicas del patrón"""
        result = matrix.copy()
        
        if pattern_positions is None:
            # Usar patrón por defecto (esquinas)
            h, w = matrix.shape
            pattern_positions = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        
        for r, c in pattern_positions:
            if 0 <= r < matrix.shape[0] and 0 <= c < matrix.shape[1]:
                result[r, c] = object_color
        
        return result
    
    def remove_objects_by_criteria(self, matrix: np.ndarray, criteria: str = 'size',
                                  threshold: int = 3, **kwargs) -> np.ndarray:
        """Elimina objetos que cumplen criterios específicos"""
        result = matrix.copy()
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                
                if self._should_remove_object(obj_mask, criteria, threshold):
                    result[obj_mask] = 0  # Remove object
        
        return result
    
    def duplicate_and_arrange(self, matrix: np.ndarray, num_copies: int = 2,
                             arrangement: str = 'horizontal', **kwargs) -> np.ndarray:
        """Duplica la matriz y la organiza"""
        if arrangement == 'horizontal':
            return np.hstack([matrix] * num_copies)
        elif arrangement == 'vertical':
            return np.vstack([matrix] * num_copies)
        elif arrangement == 'grid':
            rows = int(np.ceil(np.sqrt(num_copies)))
            cols = int(np.ceil(num_copies / rows))
            
            grid_rows = []
            for r in range(rows):
                grid_cols = []
                for c in range(cols):
                    if r * cols + c < num_copies:
                        grid_cols.append(matrix)
                    else:
                        grid_cols.append(np.zeros_like(matrix))
                grid_rows.append(np.hstack(grid_cols))
            
            return np.vstack(grid_rows)
        
        return matrix.copy()
    
    # TRANSFORMACIONES IRREGULARES
    
    def irregular_replication(self, matrix: np.ndarray, 
                             positions: List[Tuple[int, int]] = None,
                             **kwargs) -> np.ndarray:
        """Replica objetos en posiciones irregulares"""
        if positions is None:
            positions = [(0, 0), (2, 3), (1, 4)]  # Posiciones por defecto
        
        objects = self._extract_objects(matrix)
        if not objects:
            return matrix.copy()
        
        result = np.zeros_like(matrix)
        
        # Replicar primer objeto en posiciones especificadas
        first_obj = objects[0]
        for pos in positions:
            displaced_obj = self._displace_object(first_obj, pos)
            result = self._add_object_to_matrix(result, displaced_obj)
        
        return result
    
    def spiral_arrangement(self, matrix: np.ndarray, center: Tuple[int, int] = None,
                          **kwargs) -> np.ndarray:
        """Organiza objetos en espiral"""
        objects = self._extract_objects(matrix)
        if not objects:
            return matrix.copy()
        
        if center is None:
            center = (matrix.shape[0] // 2, matrix.shape[1] // 2)
        
        result = np.zeros_like(matrix)
        
        # Generar posiciones en espiral
        spiral_positions = self._generate_spiral_positions(center, len(objects))
        
        for obj, pos in zip(objects, spiral_positions):
            displaced_obj = self._displace_object(obj, pos)
            result = self._add_object_to_matrix(result, displaced_obj)
        
        return result
    
    def fractal_pattern(self, matrix: np.ndarray, iterations: int = 2,
                       scale_factor: float = 0.5, **kwargs) -> np.ndarray:
        """Crea patrón fractal simple"""
        result = matrix.copy()
        
        current_matrix = matrix
        current_scale = 1.0
        
        for i in range(iterations):
            current_scale *= scale_factor
            scaled_matrix = self._scale_matrix(current_matrix, current_scale, current_scale)
            
            # Colocar copias escaladas en esquinas
            h, w = scaled_matrix.shape
            positions = [
                (0, 0),
                (0, max(0, result.shape[1] - w)),
                (max(0, result.shape[0] - h), 0),
                (max(0, result.shape[0] - h), max(0, result.shape[1] - w))
            ]
            
            for pos in positions:
                result = self._overlay_matrix(result, scaled_matrix, pos)
        
        return result
    
    # MÉTODOS AUXILIARES
    
    def _translate_matrix(self, matrix: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Traslada matriz con wrap-around"""
        if dx == 0 and dy == 0:
            return matrix.copy()
        
        return np.roll(np.roll(matrix, dy, axis=0), dx, axis=1)
    
    def _scale_matrix(self, matrix: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
        """Escala matriz usando nearest neighbor"""
        h, w = matrix.shape
        new_h = max(1, int(h * scale_y))
        new_w = max(1, int(w * scale_x))
        
        result = np.zeros((new_h, new_w), dtype=matrix.dtype)
        
        for r in range(new_h):
            for c in range(new_w):
                orig_r = min(h - 1, int(r / scale_y))
                orig_c = min(w - 1, int(c / scale_x))
                result[r, c] = matrix[orig_r, orig_c]
        
        return result
    
    def _scale_object(self, obj_mask: np.ndarray, scale_factor: float) -> np.ndarray:
        """Escala un objeto individual"""
        positions = np.argwhere(obj_mask)
        if len(positions) == 0:
            return obj_mask
        
        centroid = np.mean(positions, axis=0)
        scaled_positions = []
        
        for pos in positions:
            # Escalar relative to centroid
            relative_pos = pos - centroid
            scaled_relative = relative_pos * scale_factor
            scaled_pos = centroid + scaled_relative
            scaled_positions.append(scaled_pos.astype(int))
        
        # Crear nueva máscara
        result = np.zeros_like(obj_mask)
        for pos in scaled_positions:
            if 0 <= pos[0] < result.shape[0] and 0 <= pos[1] < result.shape[1]:
                result[pos[0], pos[1]] = 1
        
        return result
    
    def _extract_objects(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Extrae objetos como diccionarios con metadatos"""
        objects = []
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                positions = np.argwhere(obj_mask)
                
                if len(positions) > 0:
                    centroid = np.mean(positions, axis=0)
                    
                    objects.append({
                        'color': int(color),
                        'mask': obj_mask,
                        'positions': positions,
                        'centroid': centroid,
                        'area': len(positions)
                    })
        
        return objects
    
    def _reconstruct_matrix(self, objects: List[Dict[str, Any]], 
                           shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruye matriz a partir de objetos"""
        result = np.zeros(shape, dtype=int)
        
        for obj in objects:
            result[obj['mask']] = obj['color']
        
        return result
    
    def _evaluate_condition(self, obj_mask: np.ndarray, condition: str, 
                          threshold: int) -> bool:
        """Evalúa condición para transformación condicional"""
        if condition == 'size':
            return np.sum(obj_mask) >= threshold
        elif condition == 'width':
            positions = np.argwhere(obj_mask)
            if len(positions) == 0:
                return False
            width = np.max(positions[:, 1]) - np.min(positions[:, 1]) + 1
            return width >= threshold
        elif condition == 'height':
            positions = np.argwhere(obj_mask)
            if len(positions) == 0:
                return False
            height = np.max(positions[:, 0]) - np.min(positions[:, 0]) + 1
            return height >= threshold
        
        return False
    
    def _arrange_in_grid(self, objects: List[Dict[str, Any]], 
                        shape: Tuple[int, int]) -> np.ndarray:
        """Organiza objetos en grilla"""
        result = np.zeros(shape, dtype=int)
        
        if not objects:
            return result
        
        # Calcular grilla
        n_objects = len(objects)
        grid_size = int(np.ceil(np.sqrt(n_objects)))
        
        h_step = shape[0] // (grid_size + 1)
        w_step = shape[1] // (grid_size + 1)
        
        for i, obj in enumerate(objects):
            grid_r = i // grid_size
            grid_c = i % grid_size
            
            center_r = (grid_r + 1) * h_step
            center_c = (grid_c + 1) * w_step
            
            # Colocar objeto centrado en posición de grilla
            displaced_obj = self._displace_object(obj, (center_r, center_c))
            result = self._add_object_to_matrix(result, displaced_obj)
        
        return result
    
    def _arrange_in_circle(self, objects: List[Dict[str, Any]], 
                          shape: Tuple[int, int]) -> np.ndarray:
        """Organiza objetos en círculo"""
        result = np.zeros(shape, dtype=int)
        
        if not objects:
            return result
        
        center_r, center_c = shape[0] // 2, shape[1] // 2
        radius = min(center_r, center_c) * 0.7
        
        n_objects = len(objects)
        angle_step = 2 * np.pi / n_objects
        
        for i, obj in enumerate(objects):
            angle = i * angle_step
            pos_r = int(center_r + radius * np.cos(angle))
            pos_c = int(center_c + radius * np.sin(angle))
            
            displaced_obj = self._displace_object(obj, (pos_r, pos_c))
            result = self._add_object_to_matrix(result, displaced_obj)
        
        return result
    
    def _arrange_in_line(self, objects: List[Dict[str, Any]], 
                        shape: Tuple[int, int]) -> np.ndarray:
        """Organiza objetos en línea horizontal"""
        result = np.zeros(shape, dtype=int)
        
        if not objects:
            return result
        
        n_objects = len(objects)
        step = shape[1] // (n_objects + 1)
        center_r = shape[0] // 2
        
        for i, obj in enumerate(objects):
            pos_c = (i + 1) * step
            
            displaced_obj = self._displace_object(obj, (center_r, pos_c))
            result = self._add_object_to_matrix(result, displaced_obj)
        
        return result
    
    def _draw_line(self, matrix: np.ndarray, start: Tuple[int, int], 
                  end: Tuple[int, int], color: int) -> np.ndarray:
        """Dibuja línea entre dos puntos usando algoritmo de Bresenham"""
        result = matrix.copy()
        
        x0, y0 = start[1], start[0]  # Swap for matrix indexing
        x1, y1 = end[1], end[0]
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while True:
            if 0 <= y0 < matrix.shape[0] and 0 <= x0 < matrix.shape[1]:
                result[y0, x0] = color
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x0 += sx
            
            if e2 < dx:
                err += dx
                y0 += sy
        
        return result
    
    def _get_neighbors(self, matrix: np.ndarray, r: int, c: int) -> List[int]:
        """Obtiene vecinos de una celda"""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < matrix.shape[0] and 0 <= nc < matrix.shape[1]:
                neighbors.append(matrix[nr, nc])
        
        return neighbors
    
    def _should_recolor_based_on_context(self, color: int, neighbors: List[int]) -> bool:
        """Determina si recolorear basado en contexto de vecinos"""
        # Ejemplo: recolorear si mayoría de vecinos son de color diferente
        if not neighbors:
            return False
        
        different_color_count = sum(1 for n in neighbors if n != color and n != 0)
        return different_color_count > len(neighbors) // 2
    
    def _should_remove_object(self, obj_mask: np.ndarray, criteria: str, 
                             threshold: int) -> bool:
        """Determina si eliminar objeto basado en criterios"""
        if criteria == 'size':
            return np.sum(obj_mask) < threshold
        elif criteria == 'isolated':
            # Objeto aislado si no tiene vecinos
            return not self._has_adjacent_objects(obj_mask)
        
        return False
    
    def _has_adjacent_objects(self, obj_mask: np.ndarray) -> bool:
        """Verifica si objeto tiene objetos adyacentes"""
        # Expandir máscara y verificar si hay solapamiento con otros objetos
        from scipy.ndimage import binary_dilation
        expanded = binary_dilation(obj_mask)
        return np.any(expanded & (~obj_mask))  # Simplificado
    
    def _displace_object(self, obj: Dict[str, Any], target_pos: Tuple[int, int]) -> Dict[str, Any]:
        """Desplaza objeto a nueva posición"""
        if len(obj['positions']) == 0:
            return obj
        
        current_centroid = obj['centroid']
        displacement = np.array(target_pos) - current_centroid
        
        new_positions = obj['positions'] + displacement.astype(int)
        
        return {
            'color': obj['color'],
            'positions': new_positions,
            'centroid': np.array(target_pos),
            'area': obj['area']
        }
    
    def _add_object_to_matrix(self, matrix: np.ndarray, obj: Dict[str, Any]) -> np.ndarray:
        """Añade objeto a matriz"""
        result = matrix.copy()
        
        for pos in obj['positions']:
            if 0 <= pos[0] < matrix.shape[0] and 0 <= pos[1] < matrix.shape[1]:
                result[pos[0], pos[1]] = obj['color']
        
        return result
    
    def _generate_spiral_positions(self, center: Tuple[int, int], 
                                  count: int) -> List[Tuple[int, int]]:
        """Genera posiciones en espiral"""
        positions = []
        r, c = center
        
        for i in range(count):
            # Espiral simple
            angle = i * 0.5
            radius = i * 0.3
            
            pos_r = int(r + radius * np.cos(angle))
            pos_c = int(c + radius * np.sin(angle))
            
            positions.append((pos_r, pos_c))
        
        return positions
    
    def _overlay_matrix(self, base: np.ndarray, overlay: np.ndarray, 
                       position: Tuple[int, int]) -> np.ndarray:
        """Superpone matriz en posición específica"""
        result = base.copy()
        r, c = position
        
        oh, ow = overlay.shape
        bh, bw = base.shape
        
        # Calcular región válida
        r_end = min(r + oh, bh)
        c_end = min(c + ow, bw)
        
        if r < bh and c < bw and r_end > r and c_end > c:
            overlay_slice = overlay[:r_end-r, :c_end-c]
            mask = overlay_slice != 0
            result[r:r_end, c:c_end][mask] = overlay_slice[mask]
        
        return result
    
    def _replace_object(self, matrix: np.ndarray, old_mask: np.ndarray, 
                       new_object: np.ndarray) -> np.ndarray:
        """Reemplaza objeto en matriz"""
        result = matrix.copy()
        result[old_mask] = 0  # Eliminar objeto original
        result = np.maximum(result, new_object)  # Añadir nuevo objeto
        return result