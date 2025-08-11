#!/usr/bin/env python3
"""
Sistema de Atención Bidireccional con Propagación Vertical
Cada elemento conoce su contexto completo en la jerarquía
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from scipy.ndimage import label, binary_dilation
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

@dataclass
class EnrichedPixel:
    """Píxel con conocimiento completo de su contexto jerárquico"""
    x: int
    y: int
    value: int
    
    # Enlaces hacia arriba
    parent_object_id: Optional[int] = None
    parent_relation_id: Optional[int] = None
    parent_pattern_id: Optional[int] = None
    
    # Propiedades heredadas de capas superiores
    object_role: Optional[str] = None      # "corner", "edge", "center", "interior"
    relation_role: Optional[str] = None    # "connector", "separator", "bridge"
    pattern_role: Optional[str] = None     # "critical", "regular", "noise"
    
    # Información bidireccional
    expected_value_from_pattern: Optional[int] = None
    supports_pattern: bool = False
    confidence_in_context: float = 0.0
    importance_score: float = 0.0
    
    # Contexto local
    neighborhood_3x3: Optional[np.ndarray] = None
    local_entropy: float = 0.0
    
    def get_full_context(self) -> Dict[str, Any]:
        """Devuelve descripción completa del contexto del píxel"""
        return {
            'position': (self.x, self.y),
            'value': self.value,
            'object': {
                'id': self.parent_object_id,
                'role': self.object_role
            },
            'relation': {
                'id': self.parent_relation_id,
                'role': self.relation_role
            },
            'pattern': {
                'id': self.parent_pattern_id,
                'role': self.pattern_role,
                'expected_value': self.expected_value_from_pattern,
                'supports': self.supports_pattern
            },
            'importance': self.importance_score,
            'confidence': self.confidence_in_context
        }

@dataclass
class ObjectInfo:
    """Información de un objeto detectado"""
    id: int
    pixels: List[Tuple[int, int]]
    color: int
    shape_type: str  # "square", "line", "L-shape", "irregular"
    size: int
    center: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    
    # Enlaces bidireccionales
    child_pixels: List[EnrichedPixel] = field(default_factory=list)
    parent_relations: List[int] = field(default_factory=list)
    parent_patterns: List[int] = field(default_factory=list)
    
    def contains_position(self, x: int, y: int) -> bool:
        """Verifica si una posición está dentro del objeto"""
        return (x, y) in self.pixels
    
    def get_pixel_role(self, x: int, y: int) -> str:
        """Determina el rol de un píxel dentro del objeto"""
        if not self.contains_position(x, y):
            return "none"
        
        x_min, y_min, x_max, y_max = self.bounding_box
        
        # Esquinas
        if (x, y) in [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]:
            return "corner"
        
        # Bordes
        if x == x_min or x == x_max or y == y_min or y == y_max:
            return "edge"
        
        # Centro
        if abs(x - self.center[0]) < 1 and abs(y - self.center[1]) < 1:
            return "center"
        
        return "interior"

@dataclass
class RelationInfo:
    """Información sobre relaciones entre objetos"""
    id: int
    type: str  # "aligned", "symmetric", "adjacent", "nested"
    object_ids: List[int]
    strength: float
    
    # Enlaces bidireccionales
    child_objects: List[ObjectInfo] = field(default_factory=list)
    parent_patterns: List[int] = field(default_factory=list)
    
    def involves_object(self, obj_id: int) -> bool:
        """Verifica si un objeto participa en esta relación"""
        return obj_id in self.object_ids

@dataclass
class PatternInfo:
    """Información sobre patrones globales"""
    id: int
    type: str  # "symmetry", "progression", "repetition", "transformation"
    confidence: float
    transformation_rule: Optional[Dict[str, Any]] = None
    
    # Enlaces bidireccionales
    child_relations: List[int] = field(default_factory=list)
    child_objects: List[int] = field(default_factory=list)
    
    def predict_value_at(self, x: int, y: int, context: Dict) -> Optional[int]:
        """Predice el valor esperado en una posición según el patrón"""
        if self.type == "symmetry":
            # Buscar punto simétrico
            if "axis" in context:
                mirror_x = 2 * context["axis"]["x"] - x if context["axis"]["vertical"] else x
                mirror_y = 2 * context["axis"]["y"] - y if context["axis"]["horizontal"] else y
                if "grid" in context:
                    return context["grid"][mirror_y, mirror_x] if 0 <= mirror_y < len(context["grid"]) and 0 <= mirror_x < len(context["grid"][0]) else None
        
        return None

class BidirectionalAttentionSystem:
    """Sistema de atención con propagación bidireccional entre capas"""
    
    def __init__(self):
        self.pixel_layer: Dict[Tuple[int, int], EnrichedPixel] = {}
        self.objects: Dict[int, ObjectInfo] = {}
        self.relations: Dict[int, RelationInfo] = {}
        self.patterns: Dict[int, PatternInfo] = {}
        
        self.attention_map: Optional[np.ndarray] = None
        self.coherence_map: Optional[np.ndarray] = None
        
    def build_bidirectional_hierarchy(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Construye la jerarquía completa con enlaces bidireccionales"""
        
        logger.info("Construyendo jerarquía bidireccional...")
        
        # PASO 1: Crear capa de píxeles enriquecidos
        self._build_pixel_layer(input_grid)
        
        # PASO 2: Detectar objetos y establecer enlaces descendentes
        self._detect_objects_with_backlinks(input_grid)
        
        # PASO 3: Detectar relaciones y propagar información
        self._detect_relations_with_propagation()
        
        # PASO 4: Detectar patrones y propagar hacia abajo
        self._detect_patterns_with_full_propagation(input_grid, output_grid)
        
        # PASO 5: Calcular importancia de cada píxel con contexto completo
        self._calculate_pixel_importance()
        
        return {
            'pixels': self.pixel_layer,
            'objects': self.objects,
            'relations': self.relations,
            'patterns': self.patterns,
            'hierarchy_built': True
        }
    
    def _build_pixel_layer(self, grid: np.ndarray):
        """Crea la capa base de píxeles enriquecidos"""
        h, w = grid.shape
        
        for y in range(h):
            for x in range(w):
                pixel = EnrichedPixel(x=x, y=y, value=int(grid[y, x]))
                
                # Calcular contexto local
                pixel.neighborhood_3x3 = self._get_neighborhood(grid, x, y, size=3)
                pixel.local_entropy = self._calculate_entropy(pixel.neighborhood_3x3)
                
                self.pixel_layer[(x, y)] = pixel
    
    def _detect_objects_with_backlinks(self, grid: np.ndarray):
        """Detecta objetos y establece enlaces bidireccionales con píxeles"""
        # Usar scipy.ndimage.label para encontrar componentes conectados
        labeled_array, num_features = label(grid != 0)
        
        for obj_id in range(1, num_features + 1):
            # Encontrar todos los píxeles del objeto
            positions = np.argwhere(labeled_array == obj_id)
            pixels = [(int(x), int(y)) for y, x in positions]
            
            if not pixels:
                continue
            
            # Determinar color del objeto
            colors = [grid[y, x] for x, y in pixels]
            main_color = max(set(colors), key=colors.count)
            
            # Calcular propiedades del objeto
            xs = [x for x, y in pixels]
            ys = [y for x, y in pixels]
            
            obj = ObjectInfo(
                id=obj_id,
                pixels=pixels,
                color=main_color,
                shape_type=self._classify_shape(pixels),
                size=len(pixels),
                center=(np.mean(xs), np.mean(ys)),
                bounding_box=(min(xs), min(ys), max(xs), max(ys))
            )
            
            self.objects[obj_id] = obj
            
            # ENLACE BIDIRECCIONAL: Objeto → Píxeles
            for x, y in pixels:
                pixel = self.pixel_layer[(x, y)]
                
                # El píxel conoce su objeto padre
                pixel.parent_object_id = obj_id
                pixel.object_role = obj.get_pixel_role(x, y)
                
                # El objeto conoce sus píxeles hijos
                obj.child_pixels.append(pixel)
    
    def _detect_relations_with_propagation(self):
        """Detecta relaciones entre objetos y propaga información"""
        relation_id = 0
        
        # Analizar pares de objetos
        obj_ids = list(self.objects.keys())
        for i, obj1_id in enumerate(obj_ids):
            for obj2_id in obj_ids[i+1:]:
                obj1 = self.objects[obj1_id]
                obj2 = self.objects[obj2_id]
                
                # Detectar tipo de relación
                relation_type, strength = self._analyze_object_relation(obj1, obj2)
                
                if strength > 0.5:  # Umbral de confianza
                    relation = RelationInfo(
                        id=relation_id,
                        type=relation_type,
                        object_ids=[obj1_id, obj2_id],
                        strength=strength
                    )
                    
                    self.relations[relation_id] = relation
                    
                    # ENLACES BIDIRECCIONALES
                    # Relación → Objetos
                    relation.child_objects = [obj1, obj2]
                    
                    # Objetos → Relación
                    obj1.parent_relations.append(relation_id)
                    obj2.parent_relations.append(relation_id)
                    
                    # PROPAGACIÓN A PÍXELES
                    for pixel in obj1.child_pixels + obj2.child_pixels:
                        pixel.parent_relation_id = relation_id
                        pixel.relation_role = self._determine_pixel_relation_role(
                            pixel, relation, obj1, obj2
                        )
                    
                    relation_id += 1
    
    def _detect_patterns_with_full_propagation(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Detecta patrones globales y propaga información a todas las capas inferiores"""
        pattern_id = 0
        
        # Detectar diferentes tipos de patrones
        patterns_found = []
        
        # 1. Detectar simetrías
        symmetry = self._detect_symmetry(input_grid, output_grid)
        if symmetry['confidence'] > 0.7:
            patterns_found.append({
                'type': 'symmetry',
                'confidence': symmetry['confidence'],
                'data': symmetry
            })
        
        # 2. Detectar progresiones
        progression = self._detect_progression(input_grid, output_grid)
        if progression['confidence'] > 0.7:
            patterns_found.append({
                'type': 'progression',
                'confidence': progression['confidence'],
                'data': progression
            })
        
        # 3. Detectar transformaciones
        transformation = self._detect_transformation_pattern(input_grid, output_grid)
        if transformation['confidence'] > 0.7:
            patterns_found.append({
                'type': 'transformation',
                'confidence': transformation['confidence'],
                'data': transformation
            })
        
        # Crear objetos de patrón y propagar información
        for pattern_data in patterns_found:
            pattern = PatternInfo(
                id=pattern_id,
                type=pattern_data['type'],
                confidence=pattern_data['confidence'],
                transformation_rule=pattern_data.get('data')
            )
            
            self.patterns[pattern_id] = pattern
            
            # Vincular con relaciones y objetos relevantes
            for relation in self.relations.values():
                if self._pattern_involves_relation(pattern, relation):
                    pattern.child_relations.append(relation.id)
                    relation.parent_patterns.append(pattern_id)
            
            for obj in self.objects.values():
                if self._pattern_involves_object(pattern, obj):
                    pattern.child_objects.append(obj.id)
                    obj.parent_patterns.append(pattern_id)
            
            # PROPAGACIÓN COMPLETA A PÍXELES
            for pixel in self.pixel_layer.values():
                if self._pixel_supports_pattern(pixel, pattern, input_grid):
                    pixel.parent_pattern_id = pattern_id
                    pixel.pattern_role = self._determine_pixel_pattern_role(pixel, pattern)
                    pixel.supports_pattern = True
                    
                    # El píxel puede tener un valor esperado según el patrón
                    expected = pattern.predict_value_at(
                        pixel.x, pixel.y, 
                        {'grid': input_grid, 'pattern': pattern}
                    )
                    if expected is not None:
                        pixel.expected_value_from_pattern = expected
            
            pattern_id += 1
    
    def perform_bidirectional_scan(self, grid: np.ndarray) -> np.ndarray:
        """Realiza barrido topográfico con propagación bidireccional"""
        h, w = grid.shape
        attention_map = np.zeros((h, w))
        coherence_map = np.zeros((h, w))
        
        # Barrido en espiral desde el centro
        center_x, center_y = w // 2, h // 2
        max_radius = max(h, w)
        
        for radius in range(max_radius):
            points = self._generate_spiral_points(center_x, center_y, radius, h, w)
            
            for x, y in points:
                pixel = self.pixel_layer.get((x, y))
                if not pixel:
                    continue
                
                # MIRAR ARRIBA: Información top-down
                top_down_info = self._propagate_top_down(pixel)
                
                # MIRAR ABAJO: Información bottom-up
                bottom_up_info = self._propagate_bottom_up(pixel, grid)
                
                # Calcular coherencia bidireccional
                coherence = self._calculate_bidirectional_coherence(
                    top_down_info, bottom_up_info
                )
                coherence_map[y, x] = coherence
                
                # La atención es alta donde hay incoherencia (conflictos interesantes)
                # o donde hay alta importancia
                if coherence < 0.5:  # Incoherencia
                    attention_map[y, x] = 1.0 - coherence
                else:  # Coherencia con alta importancia
                    attention_map[y, x] = coherence * pixel.importance_score
                
                # Propagar lateralmente si es un punto crítico
                if attention_map[y, x] > 0.7:
                    self._propagate_lateral_attention(x, y, attention_map, pixel)
        
        self.attention_map = attention_map
        self.coherence_map = coherence_map
        
        return attention_map
    
    def _propagate_top_down(self, pixel: EnrichedPixel) -> Dict[str, Any]:
        """Propaga información desde capas superiores hacia el píxel"""
        info = {
            'expected_value': None,
            'constraints': [],
            'confidence': 0.0,
            'role_expectations': []
        }
        
        # Desde el patrón
        if pixel.parent_pattern_id is not None:
            pattern = self.patterns[pixel.parent_pattern_id]
            
            if pattern.type == 'symmetry':
                info['constraints'].append('must_be_symmetric')
                info['confidence'] = pattern.confidence
            elif pattern.type == 'transformation':
                if pattern.transformation_rule:
                    info['expected_value'] = pixel.expected_value_from_pattern
                    info['confidence'] = pattern.confidence
        
        # Desde la relación
        if pixel.parent_relation_id is not None:
            relation = self.relations[pixel.parent_relation_id]
            
            if relation.type == 'aligned':
                info['constraints'].append('must_align')
            elif relation.type == 'symmetric':
                info['constraints'].append('must_mirror')
        
        # Desde el objeto
        if pixel.parent_object_id is not None:
            obj = self.objects[pixel.parent_object_id]
            
            if pixel.object_role == 'corner':
                info['role_expectations'].append('structural_anchor')
            elif pixel.object_role == 'edge':
                info['role_expectations'].append('boundary')
        
        return info
    
    def _propagate_bottom_up(self, pixel: EnrichedPixel, grid: np.ndarray) -> Dict[str, Any]:
        """Propaga información desde el píxel hacia capas superiores"""
        info = {
            'actual_value': pixel.value,
            'local_pattern': pixel.neighborhood_3x3,
            'structural_role': pixel.object_role,
            'supports_pattern': pixel.supports_pattern,
            'anomaly_score': 0.0
        }
        
        # Calcular si el píxel es anómalo en su contexto local
        if pixel.neighborhood_3x3 is not None:
            neighbors = pixel.neighborhood_3x3.flatten()
            neighbors = neighbors[neighbors != -1]  # Excluir bordes
            if len(neighbors) > 0:
                most_common = np.bincount(neighbors.astype(int)).argmax()
                if pixel.value != most_common:
                    info['anomaly_score'] = 1.0 - (np.sum(neighbors == pixel.value) / len(neighbors))
        
        # Verificar si cumple con expectativas del objeto
        if pixel.parent_object_id:
            obj = self.objects[pixel.parent_object_id]
            if pixel.value != obj.color:
                info['anomaly_score'] += 0.5
        
        return info
    
    def _calculate_bidirectional_coherence(self, top_down: Dict, bottom_up: Dict) -> float:
        """Calcula la coherencia entre información top-down y bottom-up"""
        coherence = 1.0
        
        # Verificar si el valor esperado coincide con el actual
        if top_down['expected_value'] is not None:
            if top_down['expected_value'] != bottom_up['actual_value']:
                coherence *= 0.5
        
        # Verificar si cumple con las restricciones
        if 'must_be_symmetric' in top_down['constraints']:
            if not bottom_up['supports_pattern']:
                coherence *= 0.7
        
        # Penalizar anomalías
        coherence *= (1.0 - bottom_up['anomaly_score'] * 0.3)
        
        # Bonus por cumplir expectativas de rol
        if bottom_up['structural_role'] in ['corner', 'edge']:
            if 'structural_anchor' in top_down['role_expectations'] or \
               'boundary' in top_down['role_expectations']:
                coherence = min(1.0, coherence * 1.2)
        
        return max(0.0, min(1.0, coherence))
    
    def _propagate_lateral_attention(self, x: int, y: int, attention_map: np.ndarray, pixel: EnrichedPixel):
        """Propaga atención lateralmente a píxeles similares"""
        # Buscar píxeles con contexto similar
        search_radius = 3
        h, w = attention_map.shape
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) != (x, y):
                    neighbor_pixel = self.pixel_layer.get((nx, ny))
                    
                    if neighbor_pixel:
                        # Calcular similitud de contexto
                        similarity = self._calculate_context_similarity(pixel, neighbor_pixel)
                        
                        if similarity > 0.7:
                            # Aumentar atención proporcionalmente
                            attention_map[ny, nx] = max(
                                attention_map[ny, nx],
                                attention_map[y, x] * similarity * 0.8
                            )
    
    def _calculate_pixel_importance(self):
        """Calcula la importancia de cada píxel en el contexto completo"""
        for pixel in self.pixel_layer.values():
            importance = 0.0
            
            # Importancia por valor no-cero
            if pixel.value != 0:
                importance += 0.1
            
            # Importancia por rol en objeto
            if pixel.object_role == 'corner':
                importance += 0.3
            elif pixel.object_role == 'edge':
                importance += 0.2
            elif pixel.object_role == 'center':
                importance += 0.25
            
            # Importancia por rol en relación
            if pixel.relation_role in ['connector', 'bridge']:
                importance += 0.2
            
            # Importancia por rol en patrón
            if pixel.pattern_role == 'critical':
                importance += 0.4
            elif pixel.supports_pattern:
                importance += 0.2
            
            # Importancia por entropía local
            importance += pixel.local_entropy * 0.1
            
            pixel.importance_score = min(1.0, importance)
            pixel.confidence_in_context = self._calculate_confidence(pixel)
    
    def _calculate_confidence(self, pixel: EnrichedPixel) -> float:
        """Calcula la confianza en el contexto del píxel"""
        confidence = 0.5  # Base
        
        # Aumentar confianza si tiene múltiples roles definidos
        if pixel.parent_object_id is not None:
            confidence += 0.15
        if pixel.parent_relation_id is not None:
            confidence += 0.15
        if pixel.parent_pattern_id is not None:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    # Métodos auxiliares
    
    def _get_neighborhood(self, grid: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
        """Obtiene el vecindario de un píxel"""
        h, w = grid.shape
        half = size // 2
        neighborhood = np.full((size, size), -1)
        
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighborhood[dy + half, dx + half] = grid[ny, nx]
        
        return neighborhood
    
    def _calculate_entropy(self, neighborhood: np.ndarray) -> float:
        """Calcula la entropía de un vecindario"""
        values = neighborhood[neighborhood != -1]
        if len(values) == 0:
            return 0.0
        
        unique, counts = np.unique(values, return_counts=True)
        probabilities = counts / len(values)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _classify_shape(self, pixels: List[Tuple[int, int]]) -> str:
        """Clasifica la forma de un conjunto de píxeles"""
        if len(pixels) == 1:
            return "point"
        elif len(pixels) == 2:
            return "pair"
        
        xs = [x for x, y in pixels]
        ys = [y for x, y in pixels]
        
        # Verificar si es una línea
        if len(set(xs)) == 1:
            return "vertical_line"
        elif len(set(ys)) == 1:
            return "horizontal_line"
        
        # Verificar si es un rectángulo
        if len(set(xs)) * len(set(ys)) == len(pixels):
            return "rectangle"
        
        # Verificar si es una L
        if self._is_l_shape(pixels):
            return "L_shape"
        
        return "irregular"
    
    def _is_l_shape(self, pixels: List[Tuple[int, int]]) -> bool:
        """Verifica si los píxeles forman una L"""
        # Simplificado: una L tiene píxeles que forman dos líneas perpendiculares
        xs = [x for x, y in pixels]
        ys = [y for x, y in pixels]
        
        # Debe tener exactamente 2 valores únicos en x y en y
        unique_x = set(xs)
        unique_y = set(ys)
        
        if len(unique_x) == 2 and len(unique_y) == 2:
            # Verificar que forma una L y no un rectángulo
            return len(pixels) < len(unique_x) * len(unique_y)
        
        return False
    
    def _analyze_object_relation(self, obj1: ObjectInfo, obj2: ObjectInfo) -> Tuple[str, float]:
        """Analiza la relación entre dos objetos"""
        # Verificar alineación
        if abs(obj1.center[0] - obj2.center[0]) < 1:
            return "vertically_aligned", 0.9
        elif abs(obj1.center[1] - obj2.center[1]) < 1:
            return "horizontally_aligned", 0.9
        
        # Verificar adyacencia
        for x1, y1 in obj1.pixels:
            for x2, y2 in obj2.pixels:
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    return "adjacent", 0.8
        
        # Verificar simetría
        center_x = (obj1.center[0] + obj2.center[0]) / 2
        center_y = (obj1.center[1] + obj2.center[1]) / 2
        
        if self._are_symmetric(obj1.pixels, obj2.pixels, center_x, center_y):
            return "symmetric", 0.85
        
        return "separate", 0.3
    
    def _are_symmetric(self, pixels1: List[Tuple[int, int]], pixels2: List[Tuple[int, int]], 
                       center_x: float, center_y: float) -> bool:
        """Verifica si dos conjuntos de píxeles son simétricos"""
        # Simplificado: verificar simetría respecto al centro
        for x1, y1 in pixels1:
            mirror_x = 2 * center_x - x1
            mirror_y = 2 * center_y - y1
            
            found = False
            for x2, y2 in pixels2:
                if abs(x2 - mirror_x) < 1 and abs(y2 - mirror_y) < 1:
                    found = True
                    break
            
            if not found:
                return False
        
        return True
    
    def _determine_pixel_relation_role(self, pixel: EnrichedPixel, relation: RelationInfo,
                                      obj1: ObjectInfo, obj2: ObjectInfo) -> str:
        """Determina el rol de un píxel en una relación"""
        x, y = pixel.x, pixel.y
        
        # Verificar si conecta los objetos
        is_between = False
        if relation.type in ["horizontally_aligned", "vertically_aligned"]:
            if obj1.center[0] <= x <= obj2.center[0] or obj2.center[0] <= x <= obj1.center[0]:
                if obj1.center[1] <= y <= obj2.center[1] or obj2.center[1] <= y <= obj1.center[1]:
                    is_between = True
        
        if is_between:
            return "connector"
        
        # Verificar si está en el borde de la relación
        if pixel.object_role == "edge":
            return "boundary"
        
        return "participant"
    
    def _detect_symmetry(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detecta patrones de simetría"""
        h, w = input_grid.shape
        
        # Verificar simetría horizontal
        horizontal_score = 0.0
        for y in range(h // 2):
            for x in range(w):
                if input_grid[y, x] == input_grid[h - 1 - y, x]:
                    horizontal_score += 1
        horizontal_score /= (h * w / 2)
        
        # Verificar simetría vertical
        vertical_score = 0.0
        for y in range(h):
            for x in range(w // 2):
                if input_grid[y, x] == input_grid[y, w - 1 - x]:
                    vertical_score += 1
        vertical_score /= (h * w / 2)
        
        best_score = max(horizontal_score, vertical_score)
        axis_type = "horizontal" if horizontal_score > vertical_score else "vertical"
        
        return {
            'confidence': best_score,
            'axis': {
                'horizontal': axis_type == "horizontal",
                'vertical': axis_type == "vertical",
                'x': w // 2,
                'y': h // 2
            }
        }
    
    def _detect_progression(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detecta patrones de progresión"""
        # Simplificado: detectar si hay un patrón incremental
        changes = output_grid - input_grid
        unique_changes = np.unique(changes[changes != 0])
        
        if len(unique_changes) == 1:
            # Cambio uniforme
            return {
                'confidence': 0.8,
                'type': 'uniform',
                'delta': int(unique_changes[0])
            }
        
        return {'confidence': 0.0}
    
    def _detect_transformation_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detecta patrones de transformación"""
        # Detectar expansión
        if output_grid.shape == input_grid.shape:
            non_zero_input = np.count_nonzero(input_grid)
            non_zero_output = np.count_nonzero(output_grid)
            
            if non_zero_output > non_zero_input * 1.5:
                return {
                    'confidence': 0.75,
                    'type': 'expansion',
                    'factor': non_zero_output / max(non_zero_input, 1)
                }
        
        return {'confidence': 0.0}
    
    def _pattern_involves_relation(self, pattern: PatternInfo, relation: RelationInfo) -> bool:
        """Verifica si un patrón involucra una relación"""
        # Simplificado
        if pattern.type == "symmetry" and relation.type == "symmetric":
            return True
        return False
    
    def _pattern_involves_object(self, pattern: PatternInfo, obj: ObjectInfo) -> bool:
        """Verifica si un patrón involucra un objeto"""
        # Todos los objetos participan en patrones globales
        return True
    
    def _pixel_supports_pattern(self, pixel: EnrichedPixel, pattern: PatternInfo, grid: np.ndarray) -> bool:
        """Verifica si un píxel soporta un patrón"""
        if pattern.type == "symmetry":
            # Verificar si tiene un píxel simétrico correspondiente
            h, w = grid.shape
            if pattern.transformation_rule and 'axis' in pattern.transformation_rule:
                axis = pattern.transformation_rule['axis']
                if axis['vertical']:
                    mirror_x = 2 * axis['x'] - pixel.x
                    if 0 <= mirror_x < w:
                        return grid[pixel.y, mirror_x] == pixel.value
                if axis['horizontal']:
                    mirror_y = 2 * axis['y'] - pixel.y
                    if 0 <= mirror_y < h:
                        return grid[mirror_y, pixel.x] == pixel.value
        
        return pixel.value != 0  # Por defecto, píxeles no-cero soportan patrones
    
    def _determine_pixel_pattern_role(self, pixel: EnrichedPixel, pattern: PatternInfo) -> str:
        """Determina el rol de un píxel en un patrón"""
        if pixel.object_role in ["corner", "center"]:
            return "critical"
        elif pixel.supports_pattern:
            return "supporting"
        else:
            return "neutral"
    
    def _generate_spiral_points(self, cx: int, cy: int, radius: int, h: int, w: int) -> List[Tuple[int, int]]:
        """Genera puntos en espiral desde el centro"""
        points = []
        
        if radius == 0:
            if 0 <= cx < w and 0 <= cy < h:
                points.append((cx, cy))
            return points
        
        # Generar puntos en el perímetro del cuadrado de radio dado
        for dx in range(-radius, radius + 1):
            # Línea superior
            y = cy - radius
            x = cx + dx
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))
            
            # Línea inferior
            y = cy + radius
            x = cx + dx
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))
        
        for dy in range(-radius + 1, radius):
            # Línea izquierda
            x = cx - radius
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))
            
            # Línea derecha
            x = cx + radius
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))
        
        return points
    
    def _calculate_context_similarity(self, pixel1: EnrichedPixel, pixel2: EnrichedPixel) -> float:
        """Calcula la similitud entre los contextos de dos píxeles"""
        similarity = 0.0
        factors = 0
        
        # Similitud de valor
        if pixel1.value == pixel2.value:
            similarity += 0.3
        factors += 0.3
        
        # Similitud de rol en objeto
        if pixel1.object_role == pixel2.object_role:
            similarity += 0.25
        factors += 0.25
        
        # Similitud de rol en relación
        if pixel1.relation_role == pixel2.relation_role:
            similarity += 0.25
        factors += 0.25
        
        # Similitud de importancia
        importance_diff = abs(pixel1.importance_score - pixel2.importance_score)
        similarity += 0.2 * (1.0 - importance_diff)
        factors += 0.2
        
        return similarity / factors if factors > 0 else 0.0
    
    def get_attention_insights(self) -> Dict[str, Any]:
        """Obtiene insights del análisis de atención"""
        if self.attention_map is None:
            return {}
        
        # Encontrar puntos de máxima atención
        high_attention_points = []
        threshold = np.percentile(self.attention_map, 90)
        
        for y in range(self.attention_map.shape[0]):
            for x in range(self.attention_map.shape[1]):
                if self.attention_map[y, x] >= threshold:
                    pixel = self.pixel_layer.get((x, y))
                    if pixel:
                        high_attention_points.append({
                            'position': (x, y),
                            'attention': float(self.attention_map[y, x]),
                            'context': pixel.get_full_context()
                        })
        
        return {
            'high_attention_points': high_attention_points,
            'average_attention': float(np.mean(self.attention_map)),
            'max_attention': float(np.max(self.attention_map)),
            'coherence_score': float(np.mean(self.coherence_map)) if self.coherence_map is not None else 0.0,
            'num_objects': len(self.objects),
            'num_relations': len(self.relations),
            'num_patterns': len(self.patterns)
        }