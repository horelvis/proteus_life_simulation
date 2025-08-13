#!/usr/bin/env python3
"""
Sistema de Atención Jerárquica Multiescala (HAMS - Hierarchical Attention Multi-Scale)
Fusiona análisis jerárquico, atención por anclas y coherencia local en un solo sistema
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from arc.transformations_fixed import RealTransformations
from arc.hierarchical_analyzer import HierarchicalAnalyzer
from arc.emergent_rule_system import EmergentRuleSystem

logger = logging.getLogger(__name__)

@dataclass
class HierarchicalPixel:
    """Píxel con información jerárquica multiescala"""
    x: int
    y: int
    value: int
    # Nivel objeto
    object_id: Optional[int] = None
    object_role: Optional[str] = None  # 'center', 'edge', 'corner'
    # Nivel patrón
    pattern_role: Optional[str] = None  # 'critical', 'supporting', 'neutral'
    # Scores de atención
    importance_score: float = 0.0
    coherence: float = 0.0
    # Atención multiescala
    is_anchor: bool = False
    anchor_type: Optional[str] = None  # 'center', 'top', 'bottom', 'left', 'right'
    local_attention: float = 0.0  # Atención en ventana local
    global_attention: float = 0.0  # Atención global

class HierarchicalAttentionSolver:
    """
    HAMS: Sistema de Atención Jerárquica Multiescala
    
    Características principales:
    - Análisis jerárquico de 3 niveles: píxel → objeto → patrón
    - Atención multiescala: local (ventanas) + global (anclas)
    - Detección robusta de transformaciones
    - Integración con transformaciones probadas
    """
    
    def __init__(self):
        self.transformations = RealTransformations()
        self.hierarchical = HierarchicalAnalyzer()
        self.emergent = EmergentRuleSystem()
        
        # Parámetros optimizados
        self.num_anchors = 5  # Anclas por objeto
        self.window_size = 3  # Tamaño de ventana para atención local
        
        # Estado interno
        self.pixel_map: Dict[Tuple[int, int], HierarchicalPixel] = {}
        self.anchors: List[HierarchicalPixel] = []
        self.object_map: Optional[np.ndarray] = None
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Interfaz simple para resolver puzzles"""
        solution, _ = self.solve_with_steps(train_examples, test_input)
        return solution
        
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve un puzzle ARC usando atención jerárquica multiescala
        
        Args:
            train_examples: Ejemplos de entrenamiento
            test_input: Entrada de test a resolver
            
        Returns:
            solution: Grilla resuelta
            steps: Pasos del proceso de resolución
        """
        steps = []
        
        if not train_examples:
            return test_input, [{"description": "No hay ejemplos de entrenamiento"}]
        
        # Analizar ejemplos de entrenamiento
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # PASO 1: Análisis jerárquico multiescala
        analysis = self._hierarchical_analysis(input_grid, output_grid)
        
        steps.append({
            "description": f"Análisis jerárquico: {analysis['num_objects']} objetos, "
                          f"{analysis['num_anchors']} anclas críticas, "
                          f"coherencia {analysis['avg_coherence']:.0%}"
        })
        
        # PASO 2: Detectar transformación
        transformation = self._detect_transformation(
            input_grid, output_grid, analysis, train_examples
        )
        
        steps.append({
            "description": f"Transformación: {transformation['type']} "
                          f"(confianza: {transformation['confidence']:.0%})"
        })
        
        # PASO 3: Aplicar transformación
        solution = self._apply_transformation(
            test_input, transformation, analysis, train_examples
        )
        
        return solution, steps
    
    def _hierarchical_analysis(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Análisis jerárquico completo con atención multiescala
        """
        h, w = input_grid.shape
        
        # 1. Detectar objetos (componentes conectados)
        self.object_map = self._detect_objects(input_grid)
        num_objects = self.object_map.max()
        
        # 2. Crear mapa de píxeles jerárquicos
        self.pixel_map = {}
        for y in range(h):
            for x in range(w):
                pixel = HierarchicalPixel(
                    x=x, y=y, 
                    value=int(input_grid[y, x]),
                    object_id=int(self.object_map[y, x])
                )
                self.pixel_map[(x, y)] = pixel
        
        # 3. Identificar anclas (puntos críticos de cada objeto)
        self._identify_anchors(input_grid)
        
        # 4. Calcular atención local (ventanas)
        self._compute_local_attention(input_grid)
        
        # 5. Detectar patrones y roles
        patterns = self._analyze_patterns(input_grid, output_grid)
        
        # 6. Calcular coherencia
        coherence_scores = self._compute_coherence(input_grid, output_grid)
        
        # 7. Fusionar atención global
        self._compute_global_attention()
        
        return {
            'num_objects': num_objects,
            'num_anchors': len(self.anchors),
            'patterns': patterns,
            'avg_coherence': np.mean(coherence_scores) if coherence_scores else 0,
            'pixel_map': self.pixel_map,
            'anchors': self.anchors,
            'object_map': self.object_map
        }
    
    def _detect_objects(self, grid: np.ndarray) -> np.ndarray:
        """Detecta objetos como componentes conectados"""
        binary = grid != 0
        labeled_array, _ = label(binary)
        return labeled_array
    
    def _identify_anchors(self, grid: np.ndarray):
        """
        Identifica anclas (puntos críticos) para cada objeto
        Las anclas son puntos clave como centro, extremos, esquinas
        """
        self.anchors = []
        
        for obj_id in range(1, self.object_map.max() + 1):
            positions = np.argwhere(self.object_map == obj_id)
            if len(positions) == 0:
                continue
            
            # Centro del objeto (ancla principal)
            center_y, center_x = positions.mean(axis=0).astype(int)
            if self.object_map[center_y, center_x] == obj_id:
                pixel = self.pixel_map[(center_x, center_y)]
                pixel.is_anchor = True
                pixel.anchor_type = 'center'
                pixel.importance_score = 1.0
                pixel.object_role = 'center'
                self.anchors.append(pixel)
            
            # Extremos del objeto (anclas secundarias)
            for anchor_type, selector in [
                ('top', lambda p: p[p[:, 0] == p[:, 0].min()]),
                ('bottom', lambda p: p[p[:, 0] == p[:, 0].max()]),
                ('left', lambda p: p[p[:, 1] == p[:, 1].min()]),
                ('right', lambda p: p[p[:, 1] == p[:, 1].max()])
            ]:
                extremes = selector(positions)
                if len(extremes) > 0:
                    y, x = extremes[len(extremes)//2]
                    pixel = self.pixel_map[(x, y)]
                    pixel.is_anchor = True
                    pixel.anchor_type = anchor_type
                    pixel.importance_score = 0.8
                    pixel.object_role = 'edge'
                    self.anchors.append(pixel)
                    
                    if len(self.anchors) >= self.num_anchors * obj_id:
                        break
    
    def _compute_local_attention(self, grid: np.ndarray):
        """
        Calcula atención local usando ventanas deslizantes
        Cada píxel atiende a su vecindario local
        """
        h, w = grid.shape
        half = self.window_size // 2
        
        for (x, y), pixel in self.pixel_map.items():
            if pixel.value == 0:
                continue
            
            # Calcular atención en ventana local
            attention_sum = 0
            count = 0
            
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor = self.pixel_map.get((nx, ny))
                        if neighbor and neighbor.object_id == pixel.object_id:
                            # Atención basada en similitud y distancia
                            similarity = 1.0 if neighbor.value == pixel.value else 0.5
                            distance = max(abs(dy), abs(dx))
                            attention = similarity * np.exp(-distance / self.window_size)
                            attention_sum += attention
                            count += 1
            
            pixel.local_attention = attention_sum / max(count, 1)
    
    def _analyze_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict]:
        """
        Analiza y detecta patrones de transformación
        Orden importa: patrones más específicos primero
        """
        patterns = []
        
        # ORDEN CRÍTICO: Verificar patrones más específicos primero
        
        # 0. Cambio de tamaño (más prioritario si las dimensiones cambian)
        size_change = self._detect_size_change_pattern(input_grid, output_grid)
        if size_change:
            patterns.append(size_change)
            # Si hay cambio de tamaño, analizar dentro del nuevo tamaño
            if size_change['type'] == 'resize':
                # Continuar analizando otros patrones en el contexto del resize
                pass
        
        # 1. Relleno (más específico que gravedad)
        if input_grid.shape == output_grid.shape and self._is_fill_pattern(input_grid, output_grid):
            patterns.append({'type': 'fill', 'confidence': 0.95})
        # 2. Gravedad (solo si no es relleno)
        elif input_grid.shape == output_grid.shape and self._is_gravity_pattern(input_grid, output_grid):
            patterns.append({'type': 'gravity', 'confidence': 0.90})
        
        # 3. Expansión en cruz
        if self._is_cross_pattern(input_grid, output_grid):
            patterns.append({'type': 'cross_expansion', 'confidence': 0.90})
        
        # 4. Mapeo de colores
        mapping = self._detect_color_mapping(input_grid, output_grid)
        if mapping:
            patterns.append({'type': 'color_mapping', 'confidence': 0.85, 'mapping': mapping})
        
        # 5. Patrón diagonal
        if self._is_diagonal_pattern(input_grid, output_grid):
            patterns.append({'type': 'diagonal_fill', 'confidence': 0.90})
        
        # 6. Rotación
        if np.array_equal(np.rot90(input_grid), output_grid):
            patterns.append({'type': 'rotation', 'confidence': 1.0})
        
        return patterns
    
    def _compute_coherence(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[float]:
        """Calcula coherencia entre input y output para cada píxel"""
        coherence_scores = []
        
        for pixel in self.pixel_map.values():
            if pixel.value != 0:
                out_val = output_grid[pixel.y, pixel.x]
                
                # Coherencia basada en rol del píxel
                if pixel.is_anchor:
                    # Las anclas suelen preservarse o ser críticas para la transformación
                    coherence = 1.0 if out_val == pixel.value else 0.5
                else:
                    # Los píxeles normales pueden cambiar más
                    coherence = 0.7
                
                pixel.coherence = coherence
                coherence_scores.append(coherence)
        
        return coherence_scores
    
    def _compute_global_attention(self):
        """
        Fusiona atención local y global en un score único por píxel
        """
        for pixel in self.pixel_map.values():
            # Ponderación basada en rol jerárquico
            anchor_weight = 1.0 if pixel.is_anchor else 0.3
            local_weight = pixel.local_attention
            importance_weight = pixel.importance_score
            coherence_weight = pixel.coherence
            
            # Fusión no-lineal para enfatizar puntos críticos
            pixel.global_attention = (
                0.4 * anchor_weight +
                0.2 * local_weight +
                0.2 * importance_weight +
                0.2 * coherence_weight
            )
    
    def _detect_transformation(self, input_grid: np.ndarray, 
                              output_grid: np.ndarray,
                              analysis: Dict,
                              train_examples: List[Dict]) -> Dict:
        """Detecta la transformación usando el análisis jerárquico"""
        patterns = analysis['patterns']
        
        if not patterns:
            # Si no hay patrones claros, analizar múltiples ejemplos
            if len(train_examples) > 1:
                pattern = self._find_consistent_pattern(train_examples)
                if pattern:
                    return {'type': pattern, 'confidence': 0.7}
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Seleccionar el patrón con mayor confianza
        best_pattern = max(patterns, key=lambda p: p['confidence'])
        return best_pattern
    
    def _apply_transformation(self, test_grid: np.ndarray,
                            transformation: Dict,
                            analysis: Dict,
                            train_examples: List[Dict] = None) -> np.ndarray:
        """Aplica la transformación detectada"""
        trans_type = transformation['type']
        
        if trans_type == 'resize':
            return self._apply_resize(test_grid, transformation, train_examples)
            
        elif trans_type == 'cross_expansion':
            return self._apply_cross_expansion(test_grid)
            
        elif trans_type == 'fill':
            return self.transformations.fill_enclosed_spaces(test_grid)
            
        elif trans_type == 'gravity':
            return self._apply_gravity(test_grid)
            
        elif trans_type == 'color_mapping':
            mapping = transformation.get('mapping', {})
            if mapping:
                return self.transformations.apply_color_mapping(test_grid, mapping)
            
            # Inferir mapeo dinámicamente de los ejemplos
            inferred_mapping = self._infer_dynamic_color_mapping(train_examples)
            if inferred_mapping:
                return self.transformations.apply_color_mapping(test_grid, inferred_mapping)
            
            # Si no se puede inferir, mantener colores originales
            return test_grid
            
        elif trans_type == 'diagonal_fill':
            return self._apply_diagonal_fill(test_grid)
            
        elif trans_type == 'rotation':
            return np.rot90(test_grid)
        
        # Sin transformación detectada
        return test_grid
    
    # === Métodos de detección de patrones ===
    
    def _detect_size_change_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """
        Detecta cambios de tamaño dinámicamente
        Ej: 3x3 -> 3x9 (expansión horizontal por factor 3)
        """
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        if (in_h, in_w) == (out_h, out_w):
            return None
        
        # Calcular factores de escala
        h_factor = out_h / in_h
        w_factor = out_w / in_w
        
        # Detectar tipo de cambio
        if h_factor == 1 and w_factor > 1:
            # Expansión horizontal (ej: 3x3 -> 3x9)
            # Verificar si es repetición del patrón
            if w_factor == int(w_factor):
                factor = int(w_factor)
                # Verificar si el output repite el input
                is_repeat = True
                for i in range(factor):
                    segment = output_grid[:, i*in_w:(i+1)*in_w]
                    if not np.array_equal(segment, input_grid):
                        is_repeat = False
                        break
                
                if is_repeat:
                    return {
                        'type': 'resize',
                        'subtype': 'horizontal_repeat',
                        'factor': factor,
                        'confidence': 1.0
                    }
                else:
                    # Puede ser expansión con transformación
                    return {
                        'type': 'resize', 
                        'subtype': 'horizontal_expand',
                        'factor': factor,
                        'confidence': 0.9
                    }
        
        elif w_factor == 1 and h_factor > 1:
            # Expansión vertical
            if h_factor == int(h_factor):
                factor = int(h_factor)
                is_repeat = True
                for i in range(factor):
                    segment = output_grid[i*in_h:(i+1)*in_h, :]
                    if not np.array_equal(segment, input_grid):
                        is_repeat = False
                        break
                
                if is_repeat:
                    return {
                        'type': 'resize',
                        'subtype': 'vertical_repeat', 
                        'factor': factor,
                        'confidence': 1.0
                    }
                else:
                    return {
                        'type': 'resize',
                        'subtype': 'vertical_expand',
                        'factor': factor,
                        'confidence': 0.9
                    }
        
        elif h_factor > 1 and w_factor > 1:
            # Expansión en ambas dimensiones
            return {
                'type': 'resize',
                'subtype': 'scale',
                'h_factor': h_factor,
                'w_factor': w_factor,
                'confidence': 0.85
            }
        
        elif h_factor < 1 or w_factor < 1:
            # Reducción
            return {
                'type': 'resize',
                'subtype': 'shrink',
                'h_factor': h_factor,
                'w_factor': w_factor,
                'confidence': 0.8
            }
        
        return None
    
    def _is_cross_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta expansión en cruz"""
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                if input_grid[y, x] != 0:
                    value = input_grid[y, x]
                    cross_count = 0
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < input_grid.shape[0] and 
                            0 <= nx < input_grid.shape[1]):
                            if input_grid[ny, nx] == 0 and output_grid[ny, nx] == value:
                                cross_count += 1
                    
                    if cross_count >= 3:
                        return True
        return False
    
    def _is_fill_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón de relleno"""
        zeros_filled = 0
        values_changed = 0
        
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                if input_grid[y, x] == 0 and output_grid[y, x] != 0:
                    # Verificar si está rodeado
                    neighbors = []
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < input_grid.shape[0] and 
                            0 <= nx < input_grid.shape[1] and
                            input_grid[ny, nx] != 0):
                            neighbors.append(input_grid[ny, nx])
                    
                    if len(neighbors) >= 2:
                        if all(n == neighbors[0] for n in neighbors):
                            if output_grid[y, x] == neighbors[0]:
                                zeros_filled += 1
                
                elif input_grid[y, x] != 0 and input_grid[y, x] != output_grid[y, x]:
                    values_changed += 1
        
        return zeros_filled > 0 and values_changed == 0
    
    def _is_gravity_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón de gravedad"""
        h = input_grid.shape[0]
        gravity_columns = 0
        total_columns_with_values = 0
        
        for col in range(input_grid.shape[1]):
            input_col = input_grid[:, col]
            output_col = output_grid[:, col]
            
            input_nonzero = input_col[input_col != 0]
            output_nonzero = output_col[output_col != 0]
            
            if len(input_nonzero) > 0:
                total_columns_with_values += 1
                
                if len(output_nonzero) == len(input_nonzero):
                    if np.array_equal(sorted(input_nonzero), sorted(output_nonzero)):
                        first_nonzero = np.argmax(output_col != 0)
                        expected_start = h - len(output_nonzero)
                        if first_nonzero == expected_start:
                            if np.all(output_col[:expected_start] == 0):
                                gravity_columns += 1
        
        return gravity_columns > 0 and gravity_columns == total_columns_with_values
    
    def _detect_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta mapeo de colores"""
        mapping = {}
        
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                in_val = input_grid[y, x]
                out_val = output_grid[y, x]
                
                if in_val != 0:
                    if in_val in mapping:
                        if mapping[in_val] != out_val:
                            return None
                    else:
                        mapping[in_val] = out_val
        
        if mapping and any(k != v for k, v in mapping.items()):
            return mapping
        
        return None
    
    def _is_diagonal_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón diagonal"""
        h, w = input_grid.shape
        if h != w:
            return False
        
        diagonal_values = []
        for i in range(min(h, w)):
            if input_grid[i, i] != 0:
                diagonal_values.append(input_grid[i, i])
        
        if diagonal_values:
            expected = np.full_like(input_grid, diagonal_values[0])
            return np.array_equal(output_grid, expected)
        
        return False
    
    # === Métodos de aplicación de transformaciones ===
    
    def _apply_cross_expansion(self, grid: np.ndarray) -> np.ndarray:
        """Aplica expansión en cruz"""
        result = grid.copy()
        h, w = grid.shape
        
        for y in range(h):
            for x in range(w):
                if grid[y, x] != 0:
                    value = grid[y, x]
                    
                    # Verificar si es un punto aislado (candidato a expansión)
                    neighbors_empty = 0
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0):
                            neighbors_empty += 1
                    
                    # Si tiene al menos 3 vecinos vacíos, expandir
                    if neighbors_empty >= 3:
                        for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0):
                                result[ny, nx] = value
        
        return result
    
    def _apply_gravity(self, grid: np.ndarray) -> np.ndarray:
        """Aplica gravedad (objetos caen)"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for col in range(w):
            column = grid[:, col]
            non_zero = column[column != 0]
            
            if len(non_zero) > 0:
                result[h-len(non_zero):, col] = non_zero
        
        return result
    
    def _apply_resize(self, grid: np.ndarray, transformation: Dict, train_examples: List[Dict] = None) -> np.ndarray:
        """
        Aplica cambio de tamaño dinámicamente aprendido
        """
        subtype = transformation.get('subtype', '')
        
        if subtype == 'horizontal_repeat':
            # Repetir horizontalmente
            factor = transformation.get('factor', 2)
            return np.tile(grid, (1, factor))
            
        elif subtype == 'vertical_repeat':
            # Repetir verticalmente
            factor = transformation.get('factor', 2)
            return np.tile(grid, (factor, 1))
            
        elif subtype == 'horizontal_expand':
            # Expansión horizontal con posible transformación
            factor = transformation.get('factor', 2)
            h, w = grid.shape
            result = np.zeros((h, w * factor), dtype=grid.dtype)
            
            # Si hay ejemplos, aprender el patrón de expansión
            if train_examples:
                for example in train_examples:
                    in_grid = np.array(example['input'])
                    out_grid = np.array(example['output'])
                    
                    if out_grid.shape[1] == in_grid.shape[1] * factor:
                        # Analizar cómo se expande
                        for i in range(factor):
                            segment = out_grid[:, i*w:(i+1)*w]
                            if i == 0:
                                result[:, :w] = grid
                            else:
                                # Aplicar transformación detectada en el segmento
                                # Por ejemplo, podría ser rotación, espejo, etc.
                                if np.array_equal(segment, np.fliplr(in_grid)):
                                    result[:, i*w:(i+1)*w] = np.fliplr(grid)
                                elif np.array_equal(segment, np.rot90(in_grid)):
                                    result[:, i*w:(i+1)*w] = np.rot90(grid)
                                else:
                                    result[:, i*w:(i+1)*w] = grid
                        break
            else:
                # Sin ejemplos, repetir simplemente
                result = np.tile(grid, (1, factor))
            
            return result
            
        elif subtype == 'vertical_expand':
            # Similar a horizontal pero en vertical
            factor = transformation.get('factor', 2)
            h, w = grid.shape
            result = np.zeros((h * factor, w), dtype=grid.dtype)
            
            if train_examples:
                for example in train_examples:
                    in_grid = np.array(example['input'])
                    out_grid = np.array(example['output'])
                    
                    if out_grid.shape[0] == in_grid.shape[0] * factor:
                        for i in range(factor):
                            segment = out_grid[i*h:(i+1)*h, :]
                            if i == 0:
                                result[:h, :] = grid
                            else:
                                if np.array_equal(segment, np.flipud(in_grid)):
                                    result[i*h:(i+1)*h, :] = np.flipud(grid)
                                elif np.array_equal(segment, np.rot90(in_grid)):
                                    result[i*h:(i+1)*h, :] = np.rot90(grid)[:h, :w]
                                else:
                                    result[i*h:(i+1)*h, :] = grid
                        break
            else:
                result = np.tile(grid, (factor, 1))
            
            return result
            
        elif subtype == 'scale':
            # Escalar en ambas dimensiones
            h_factor = int(transformation.get('h_factor', 2))
            w_factor = int(transformation.get('w_factor', 2))
            
            # Repetir el patrón
            return np.repeat(np.repeat(grid, h_factor, axis=0), w_factor, axis=1)
            
        elif subtype == 'shrink':
            # Reducir tamaño (submuestreo)
            h_factor = transformation.get('h_factor', 0.5)
            w_factor = transformation.get('w_factor', 0.5)
            
            new_h = int(grid.shape[0] * h_factor)
            new_w = int(grid.shape[1] * w_factor)
            
            # Submuestrear
            step_h = max(1, grid.shape[0] // new_h)
            step_w = max(1, grid.shape[1] // new_w)
            
            return grid[::step_h, ::step_w][:new_h, :new_w]
        
        # Por defecto, devolver sin cambios
        return grid
    
    def _apply_diagonal_fill(self, grid: np.ndarray) -> np.ndarray:
        """Rellena con valor diagonal"""
        h, w = grid.shape
        diagonal_values = []
        
        for i in range(min(h, w)):
            if grid[i, i] != 0:
                diagonal_values.append(grid[i, i])
        
        if diagonal_values:
            return np.full_like(grid, diagonal_values[0])
        
        return grid
    
    def _infer_dynamic_color_mapping(self, train_examples: List[Dict]) -> Optional[Dict]:
        """
        Infiere dinámicamente el mapeo de colores de los ejemplos
        Sin hardcodear incrementos arbitrarios
        """
        if not train_examples:
            return None
        
        all_mappings = []
        
        for example in train_examples:
            input_g = np.array(example['input'])
            output_g = np.array(example['output'])
            
            # Solo considerar si tienen la misma forma
            if input_g.shape != output_g.shape:
                continue
            
            # Detectar mapeo para este ejemplo
            example_mapping = {}
            for y in range(input_g.shape[0]):
                for x in range(input_g.shape[1]):
                    in_val = int(input_g[y, x])
                    out_val = int(output_g[y, x])
                    
                    if in_val != 0:  # Ignorar fondo
                        if in_val in example_mapping:
                            # Verificar consistencia
                            if example_mapping[in_val] != out_val:
                                # Mapeo inconsistente en este ejemplo
                                example_mapping = None
                                break
                        else:
                            example_mapping[in_val] = out_val
                
                if example_mapping is None:
                    break
            
            if example_mapping:
                all_mappings.append(example_mapping)
        
        # Si todos los ejemplos tienen el mismo mapeo, usarlo
        if all_mappings:
            # Verificar si todos los mapeos son iguales
            first_mapping = all_mappings[0]
            if all(m == first_mapping for m in all_mappings):
                return first_mapping
            
            # Si no son iguales, buscar patrón común
            # Por ejemplo, todos incrementan en la misma cantidad
            increments = []
            for mapping in all_mappings:
                if mapping:
                    inc_set = set()
                    for k, v in mapping.items():
                        if k != 0:
                            inc_set.add(v - k)
                    if len(inc_set) == 1:  # Incremento consistente
                        increments.append(list(inc_set)[0])
            
            if increments and all(i == increments[0] for i in increments):
                # Crear mapeo basado en incremento común
                increment = increments[0]
                dynamic_mapping = {}
                for val in range(1, 10):  # Colores ARC típicos
                    new_val = val + increment
                    if 0 <= new_val <= 9:  # Mantener en rango válido
                        dynamic_mapping[val] = new_val
                return dynamic_mapping
        
        return None
    
    def _find_consistent_pattern(self, train_examples: List[Dict]) -> Optional[str]:
        """Encuentra patrón consistente en múltiples ejemplos"""
        patterns_found = []
        
        for example in train_examples:
            input_g = np.array(example['input'])
            output_g = np.array(example['output'])
            
            analysis = self._hierarchical_analysis(input_g, output_g)
            patterns = analysis['patterns']
            
            if patterns:
                best = max(patterns, key=lambda p: p['confidence'])
                patterns_found.append(best['type'])
        
        if patterns_found:
            return max(set(patterns_found), key=patterns_found.count)
        
        return None