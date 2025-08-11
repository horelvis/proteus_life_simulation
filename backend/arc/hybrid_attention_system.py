#!/usr/bin/env python3
"""
Sistema Híbrido de Atención: Anclas + Window Attention
Combina eficiencia O(N·K) con coherencia local O(N·W²)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import logging
from scipy.ndimage import label
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class AnchorPoint:
    """Punto ancla para atención eficiente"""
    x: int
    y: int
    anchor_type: str  # 'center', 'top', 'bottom', 'left', 'right', 'corner'
    object_id: int
    importance: float
    features: np.ndarray  # Vector de características del ancla

@dataclass
class WindowAttentionMask:
    """Máscara para window attention"""
    window_size: int
    region_mask: np.ndarray  # Máscara de regiones/objetos
    attention_weights: np.ndarray

class HybridAttentionSystem:
    """
    Sistema híbrido que combina:
    1. Atención por anclas (O(N·K))
    2. Window attention enmascarada (O(N·W²))
    3. Jerarquía bidireccional existente
    """
    
    def __init__(self, num_anchors: int = 5, window_size: int = 3):
        self.num_anchors = num_anchors
        self.window_size = window_size
        self.anchors: Dict[int, List[AnchorPoint]] = {}
        self.window_masks: List[WindowAttentionMask] = []
        self.attention_map: Optional[np.ndarray] = None
        self.object_map: Optional[np.ndarray] = None
        
    def compute_hybrid_attention(self, grid: np.ndarray) -> np.ndarray:
        """
        Calcula atención híbrida combinando anclas y windows
        """
        h, w = grid.shape
        attention = np.zeros((h, w))
        
        # PASO 1: Detectar objetos/regiones
        self.object_map = self._detect_objects(grid)
        
        # PASO 2: Calcular anclas para cada objeto
        self._compute_anchors(grid, self.object_map)
        
        # PASO 3: Atención por anclas
        anchor_attention = self._compute_anchor_attention(grid)
        
        # PASO 4: Window attention con máscaras
        window_attention = self._compute_window_attention(grid)
        
        # PASO 5: Combinar ambas atenciones
        attention = 0.6 * anchor_attention + 0.4 * window_attention
        
        self.attention_map = attention
        return attention
    
    def _detect_objects(self, grid: np.ndarray) -> np.ndarray:
        """Detecta objetos/regiones en la grilla"""
        # Usar componentes conectados
        binary = grid != 0
        labeled_array, num_features = label(binary)
        return labeled_array
    
    def _compute_anchors(self, grid: np.ndarray, object_map: np.ndarray):
        """
        Calcula K anclas por objeto
        Anclas: centro + 4 puntos cardinales o esquinas
        """
        self.anchors = {}
        
        for obj_id in range(1, object_map.max() + 1):
            # Encontrar píxeles del objeto
            positions = np.argwhere(object_map == obj_id)
            if len(positions) == 0:
                continue
            
            anchors = []
            
            # 1. Ancla central (centroide)
            center_y, center_x = positions.mean(axis=0)
            center_y, center_x = int(center_y), int(center_x)
            
            # Encontrar el píxel más cercano al centroide
            distances = cdist([[center_y, center_x]], positions)[0]
            nearest_idx = distances.argmin()
            actual_center = positions[nearest_idx]
            
            anchors.append(AnchorPoint(
                x=actual_center[1],
                y=actual_center[0],
                anchor_type='center',
                object_id=obj_id,
                importance=1.0,
                features=self._extract_features(grid, actual_center[1], actual_center[0])
            ))
            
            # 2. Anclas en los extremos (top, bottom, left, right)
            # Top (mínimo y)
            top_positions = positions[positions[:, 0] == positions[:, 0].min()]
            if len(top_positions) > 0:
                top = top_positions[len(top_positions)//2]  # Medio del borde superior
                anchors.append(AnchorPoint(
                    x=top[1], y=top[0],
                    anchor_type='top',
                    object_id=obj_id,
                    importance=0.8,
                    features=self._extract_features(grid, top[1], top[0])
                ))
            
            # Bottom (máximo y)
            bottom_positions = positions[positions[:, 0] == positions[:, 0].max()]
            if len(bottom_positions) > 0:
                bottom = bottom_positions[len(bottom_positions)//2]
                anchors.append(AnchorPoint(
                    x=bottom[1], y=bottom[0],
                    anchor_type='bottom',
                    object_id=obj_id,
                    importance=0.8,
                    features=self._extract_features(grid, bottom[1], bottom[0])
                ))
            
            # Left (mínimo x)
            left_positions = positions[positions[:, 1] == positions[:, 1].min()]
            if len(left_positions) > 0:
                left = left_positions[len(left_positions)//2]
                anchors.append(AnchorPoint(
                    x=left[1], y=left[0],
                    anchor_type='left',
                    object_id=obj_id,
                    importance=0.8,
                    features=self._extract_features(grid, left[1], left[0])
                ))
            
            # Right (máximo x)
            right_positions = positions[positions[:, 1] == positions[:, 1].max()]
            if len(right_positions) > 0:
                right = right_positions[len(right_positions)//2]
                anchors.append(AnchorPoint(
                    x=right[1], y=right[0],
                    anchor_type='right',
                    object_id=obj_id,
                    importance=0.8,
                    features=self._extract_features(grid, right[1], right[0])
                ))
            
            # Limitar a num_anchors
            if len(anchors) > self.num_anchors:
                # Ordenar por importancia y tomar los mejores
                anchors.sort(key=lambda a: a.importance, reverse=True)
                anchors = anchors[:self.num_anchors]
            
            self.anchors[obj_id] = anchors
    
    def _extract_features(self, grid: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Extrae vector de características para un punto
        """
        h, w = grid.shape
        features = []
        
        # 1. Valor del píxel
        features.append(grid[y, x])
        
        # 2. Valores de vecindario 3x3
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    features.append(grid[ny, nx])
                else:
                    features.append(0)
        
        # 3. Posición normalizada
        features.append(x / w)
        features.append(y / h)
        
        # 4. Distancia al centro de la grilla
        center_x, center_y = w / 2, h / 2
        dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        features.append(dist_to_center / np.sqrt(w**2 + h**2))
        
        return np.array(features)
    
    def _compute_anchor_attention(self, grid: np.ndarray) -> np.ndarray:
        """
        Calcula atención basada en anclas
        Cada píxel atiende solo a las anclas de su objeto
        
        α_x,a = softmax_a(q(x)ᵀk(a)/√d - λ·dist(x,a))
        """
        h, w = grid.shape
        attention = np.zeros((h, w))
        
        # Parámetros
        lambda_dist = 0.1  # Factor de penalización por distancia
        temperature = 0.5  # Temperatura del softmax
        
        for y in range(h):
            for x in range(w):
                # Obtener objeto al que pertenece el píxel
                obj_id = self.object_map[y, x]
                
                if obj_id == 0:  # Fondo
                    continue
                
                if obj_id not in self.anchors:
                    continue
                
                # Obtener anclas del objeto
                obj_anchors = self.anchors[obj_id]
                
                if not obj_anchors:
                    continue
                
                # Features del píxel actual (query)
                pixel_features = self._extract_features(grid, x, y)
                
                # Calcular atención a cada ancla
                attention_scores = []
                
                for anchor in obj_anchors:
                    # Similitud de características (producto punto normalizado)
                    similarity = np.dot(pixel_features, anchor.features) / (
                        np.linalg.norm(pixel_features) * np.linalg.norm(anchor.features) + 1e-8
                    )
                    
                    # Distancia espacial
                    distance = np.sqrt((x - anchor.x)**2 + (y - anchor.y)**2)
                    
                    # Score combinado
                    score = similarity / temperature - lambda_dist * distance
                    attention_scores.append(score * anchor.importance)
                
                # Softmax sobre las anclas
                if attention_scores:
                    attention_scores = np.array(attention_scores)
                    attention_scores = np.exp(attention_scores - attention_scores.max())
                    attention_scores = attention_scores / attention_scores.sum()
                    
                    # Atención final del píxel
                    attention[y, x] = attention_scores.max()
        
        return attention
    
    def _compute_window_attention(self, grid: np.ndarray) -> np.ndarray:
        """
        Calcula window attention con máscaras por región
        Ventanas de tamaño W×W, máscara para mantener coherencia
        """
        h, w = grid.shape
        attention = np.zeros((h, w))
        half_window = self.window_size // 2
        
        # Para cada posición, calcular atención en su ventana
        for y in range(h):
            for x in range(w):
                # Definir ventana
                y_start = max(0, y - half_window)
                y_end = min(h, y + half_window + 1)
                x_start = max(0, x - half_window)
                x_end = min(w, x + half_window + 1)
                
                # Extraer ventana
                window = grid[y_start:y_end, x_start:x_end]
                window_objects = self.object_map[y_start:y_end, x_start:x_end]
                
                # Posición relativa del píxel en la ventana
                rel_y = y - y_start
                rel_x = x - x_start
                
                # Objeto del píxel central
                center_obj = self.object_map[y, x]
                
                if center_obj == 0:  # Fondo
                    continue
                
                # Crear máscara: 1 para mismo objeto, 0 para diferente
                mask = (window_objects == center_obj).astype(float)
                
                # Query del píxel central
                center_value = window[rel_y, rel_x]
                
                # Calcular scores de atención
                attention_scores = np.zeros_like(window, dtype=float)
                
                for wy in range(window.shape[0]):
                    for wx in range(window.shape[1]):
                        if mask[wy, wx] > 0:  # Solo atender a mismo objeto
                            # Similitud basada en valor
                            value_sim = 1.0 - abs(window[wy, wx] - center_value) / 10.0
                            
                            # Distancia espacial normalizada
                            dist = np.sqrt((wy - rel_y)**2 + (wx - rel_x)**2)
                            dist_factor = np.exp(-dist / self.window_size)
                            
                            attention_scores[wy, wx] = value_sim * dist_factor
                
                # Normalizar scores
                if attention_scores.sum() > 0:
                    attention_scores = attention_scores / attention_scores.sum()
                    
                    # La atención del píxel es el máximo score en su ventana
                    attention[y, x] = attention_scores.max()
        
        return attention
    
    def apply_transformation_with_hybrid_attention(self, 
                                                  test_grid: np.ndarray,
                                                  transformation_type: str,
                                                  reference_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica transformación guiada por el sistema híbrido de atención
        """
        # Calcular atención híbrida
        attention = self.compute_hybrid_attention(test_grid)
        
        result = test_grid.copy()
        
        if transformation_type == "cross_expansion":
            # Usar anclas para expansión precisa
            result = self._expand_using_anchors(test_grid, attention)
            
        elif transformation_type == "fill":
            # Usar window attention para relleno coherente
            result = self._fill_using_windows(test_grid, attention)
            
        elif transformation_type == "gravity":
            # Usar anclas bottom para gravedad
            result = self._apply_gravity_with_anchors(test_grid)
            
        elif transformation_type == "rotation":
            # Detectar eje de rotación con anclas
            result = self._rotate_using_anchors(test_grid)
        
        return result
    
    def _expand_using_anchors(self, grid: np.ndarray, attention: np.ndarray) -> np.ndarray:
        """
        Expansión precisa usando anclas centrales
        """
        result = grid.copy()
        
        for obj_id, anchors in self.anchors.items():
            # Encontrar ancla central
            center_anchor = next((a for a in anchors if a.anchor_type == 'center'), None)
            
            if center_anchor:
                # Expandir desde el centro en cruz
                value = grid[center_anchor.y, center_anchor.x]
                
                # Expansión controlada basada en atención
                for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    dy, dx = direction
                    ny = center_anchor.y + dy
                    nx = center_anchor.x + dx
                    
                    if (0 <= ny < grid.shape[0] and 
                        0 <= nx < grid.shape[1] and
                        grid[ny, nx] == 0 and
                        attention[ny, nx] > 0.3):  # Solo expandir si hay suficiente atención
                        result[ny, nx] = value
        
        return result
    
    def _fill_using_windows(self, grid: np.ndarray, attention: np.ndarray) -> np.ndarray:
        """
        Relleno coherente usando window attention
        """
        result = grid.copy()
        
        # Para cada objeto, rellenar huecos internos
        for obj_id in range(1, self.object_map.max() + 1):
            obj_mask = (self.object_map == obj_id)
            
            # Encontrar bounding box del objeto
            positions = np.argwhere(obj_mask)
            if len(positions) == 0:
                continue
            
            y_min, x_min = positions.min(axis=0)
            y_max, x_max = positions.max(axis=0)
            
            # Valor dominante del objeto
            obj_values = grid[obj_mask]
            if len(obj_values) > 0:
                fill_value = np.bincount(obj_values[obj_values != 0]).argmax() if np.any(obj_values != 0) else 0
                
                # Rellenar huecos dentro del bounding box
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        if grid[y, x] == 0 and attention[y, x] > 0.4:
                            # Verificar si está rodeado por el objeto
                            neighbors = 0
                            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < grid.shape[0] and 
                                    0 <= nx < grid.shape[1] and
                                    self.object_map[ny, nx] == obj_id):
                                    neighbors += 1
                            
                            if neighbors >= 2:  # Al menos 2 vecinos del mismo objeto
                                result[y, x] = fill_value
        
        return result
    
    def _apply_gravity_with_anchors(self, grid: np.ndarray) -> np.ndarray:
        """
        Aplica gravedad usando anclas bottom
        """
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for col in range(w):
            # Obtener valores no-cero de la columna
            column = grid[:, col]
            non_zero = column[column != 0]
            
            if len(non_zero) > 0:
                # Colocar al fondo
                result[h-len(non_zero):, col] = non_zero
        
        return result
    
    def _rotate_using_anchors(self, grid: np.ndarray) -> np.ndarray:
        """
        Rotación basada en anclas
        """
        # Por simplicidad, rotación 90° antihorario
        return np.rot90(grid)
    
    def get_attention_stats(self) -> Dict[str, any]:
        """
        Obtiene estadísticas del sistema de atención
        """
        if self.attention_map is None:
            return {}
        
        return {
            'num_objects': len(self.anchors),
            'total_anchors': sum(len(a) for a in self.anchors.values()),
            'max_attention': float(self.attention_map.max()),
            'mean_attention': float(self.attention_map.mean()),
            'high_attention_ratio': float((self.attention_map > 0.7).sum() / self.attention_map.size)
        }