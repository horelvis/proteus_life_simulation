#!/usr/bin/env python3
"""
Analizador Jerárquico Multiescala
Análisis completo desde píxeles hasta relaciones entre formas
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy.ndimage import label, distance_transform_edt
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

@dataclass
class PixelLevel:
    """Nivel 0: Información de píxeles individuales"""
    color: int
    position: Tuple[int, int]
    neighbors: List[int]  # Colores de vecinos
    local_pattern: str  # corner, edge, interior, isolated

@dataclass  
class ObjectLevel:
    """Nivel 1: Objetos/formas detectadas"""
    id: int
    color: int
    pixels: List[Tuple[int, int]]
    area: int
    perimeter: int
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # min_r, min_c, max_r, max_c
    shape_type: str  # rectangle, line, L-shape, irregular, etc.
    orientation: float  # Ángulo principal
    compactness: float
    
@dataclass
class RelationLevel:
    """Nivel 2: Relaciones entre objetos"""
    object1_id: int
    object2_id: int
    relation_type: str  # adjacent, aligned, contains, overlaps, etc.
    distance: float
    relative_position: str  # above, below, left, right, diagonal
    alignment: str  # horizontal, vertical, diagonal, none
    size_ratio: float
    
@dataclass
class PatternLevel:
    """Nivel 3: Patrones y estructuras globales"""
    pattern_type: str  # grid, symmetry, repetition, progression
    objects_involved: List[int]
    regularity_score: float
    transformation: Optional[str]  # rotation, reflection, translation
    parameters: Dict[str, Any]

class HierarchicalAnalyzer:
    """
    Analizador multiescala jerárquico que procesa desde píxeles hasta patrones globales
    """
    
    def __init__(self):
        self.levels = {}
        
    def analyze_full_hierarchy(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Análisis completo en 4 niveles jerárquicos
        """
        logger.info("Iniciando análisis jerárquico multiescala")
        
        # Nivel 0: Análisis de píxeles
        pixel_analysis = self._analyze_pixel_level(matrix)
        
        # Nivel 1: Detección y análisis de objetos
        object_analysis = self._analyze_object_level(matrix, pixel_analysis)
        
        # Nivel 2: Relaciones entre objetos
        relation_analysis = self._analyze_relation_level(object_analysis)
        
        # Nivel 3: Patrones globales
        pattern_analysis = self._analyze_pattern_level(object_analysis, relation_analysis)
        
        # Integración: Síntesis de información multiescala
        integrated = self._integrate_scales(
            pixel_analysis, object_analysis, 
            relation_analysis, pattern_analysis
        )
        
        return {
            'level_0_pixels': pixel_analysis,
            'level_1_objects': object_analysis,
            'level_2_relations': relation_analysis,
            'level_3_patterns': pattern_analysis,
            'integrated_view': integrated,
            'hierarchy_summary': self._generate_hierarchy_summary(integrated)
        }
    
    def _analyze_pixel_level(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Nivel 0: Análisis detallado de píxeles individuales
        """
        h, w = matrix.shape
        pixel_info = []
        color_clusters = defaultdict(list)
        
        for i in range(h):
            for j in range(w):
                color = int(matrix[i, j])
                
                # Obtener vecinos
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbors.append(int(matrix[ni, nj]))
                
                # Clasificar patrón local
                local_pattern = self._classify_local_pattern(neighbors, color)
                
                pixel = PixelLevel(
                    color=color,
                    position=(i, j),
                    neighbors=neighbors,
                    local_pattern=local_pattern
                )
                
                pixel_info.append(pixel)
                color_clusters[color].append((i, j))
        
        return {
            'pixels': pixel_info,
            'color_clusters': dict(color_clusters),
            'local_patterns': self._summarize_local_patterns(pixel_info),
            'edge_pixels': self._find_edge_pixels(pixel_info),
            'corner_pixels': self._find_corner_pixels(pixel_info)
        }
    
    def _analyze_object_level(self, matrix: np.ndarray, pixel_analysis: Dict) -> Dict[str, Any]:
        """
        Nivel 1: Detección y caracterización de objetos/formas
        """
        objects = []
        object_id = 0
        
        # Para cada color, encontrar objetos conectados
        for color, positions in pixel_analysis['color_clusters'].items():
            if color == 0:  # Skip background
                continue
                
            # Crear máscara para este color
            mask = (matrix == color)
            labeled, num_objects = label(mask)
            
            # Analizar cada objeto
            for obj_label in range(1, num_objects + 1):
                obj_mask = (labeled == obj_label)
                obj_pixels = np.argwhere(obj_mask)
                
                if len(obj_pixels) == 0:
                    continue
                
                # Calcular propiedades
                area = len(obj_pixels)
                centroid = np.mean(obj_pixels, axis=0)
                min_r, min_c = obj_pixels.min(axis=0)
                max_r, max_c = obj_pixels.max(axis=0)
                
                # Perímetro usando transformada de distancia
                dist_transform = distance_transform_edt(obj_mask)
                perimeter = np.sum(dist_transform == 1)
                
                # Clasificar forma
                shape_type = self._classify_shape(obj_mask, obj_pixels)
                
                # Orientación principal
                if len(obj_pixels) > 2:
                    _, _, angle = self._compute_principal_axis(obj_pixels)
                    orientation = angle
                else:
                    orientation = 0.0
                
                # Compacidad
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                obj = ObjectLevel(
                    id=object_id,
                    color=color,
                    pixels=obj_pixels.tolist(),
                    area=area,
                    perimeter=int(perimeter),
                    centroid=tuple(centroid),
                    bounding_box=(min_r, min_c, max_r, max_c),
                    shape_type=shape_type,
                    orientation=orientation,
                    compactness=compactness
                )
                
                objects.append(obj)
                object_id += 1
        
        return {
            'objects': objects,
            'num_objects': len(objects),
            'shape_distribution': self._get_shape_distribution(objects),
            'size_distribution': self._get_size_distribution(objects),
            'color_to_objects': self._map_color_to_objects(objects)
        }
    
    def _analyze_relation_level(self, object_analysis: Dict) -> Dict[str, Any]:
        """
        Nivel 2: Análisis de relaciones espaciales entre objetos
        """
        objects = object_analysis['objects']
        relations = []
        
        # Analizar pares de objetos
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]
                
                # Calcular distancia entre centroides
                dist = np.linalg.norm(
                    np.array(obj1.centroid) - np.array(obj2.centroid)
                )
                
                # Determinar posición relativa
                relative_pos = self._get_relative_position(obj1.centroid, obj2.centroid)
                
                # Verificar adyacencia
                is_adjacent = self._check_adjacency(obj1, obj2)
                
                # Verificar alineación
                alignment = self._check_alignment(obj1, obj2)
                
                # Ratio de tamaños
                size_ratio = obj1.area / obj2.area if obj2.area > 0 else float('inf')
                
                # Determinar tipo de relación
                if is_adjacent:
                    relation_type = "adjacent"
                elif self._check_containment(obj1, obj2):
                    relation_type = "contains"
                elif self._check_containment(obj2, obj1):
                    relation_type = "contained_by"
                elif alignment != "none":
                    relation_type = "aligned"
                else:
                    relation_type = "separated"
                
                relation = RelationLevel(
                    object1_id=obj1.id,
                    object2_id=obj2.id,
                    relation_type=relation_type,
                    distance=dist,
                    relative_position=relative_pos,
                    alignment=alignment,
                    size_ratio=size_ratio
                )
                
                relations.append(relation)
        
        # Construir grafo de relaciones
        relation_graph = self._build_relation_graph(objects, relations)
        
        return {
            'relations': relations,
            'num_relations': len(relations),
            'relation_graph': relation_graph,
            'spatial_structure': self._analyze_spatial_structure(relations),
            'clustering': self._find_object_clusters(objects, relations)
        }
    
    def _analyze_pattern_level(self, object_analysis: Dict, relation_analysis: Dict) -> Dict[str, Any]:
        """
        Nivel 3: Detección de patrones y estructuras globales
        """
        objects = object_analysis['objects']
        relations = relation_analysis['relations']
        patterns = []
        
        # Detectar grid/rejilla
        grid_pattern = self._detect_grid_pattern(objects)
        if grid_pattern:
            patterns.append(grid_pattern)
        
        # Detectar simetrías
        symmetry_patterns = self._detect_symmetry_patterns(objects)
        patterns.extend(symmetry_patterns)
        
        # Detectar repeticiones
        repetition_patterns = self._detect_repetition_patterns(objects)
        patterns.extend(repetition_patterns)
        
        # Detectar progresiones
        progression_patterns = self._detect_progression_patterns(objects)
        patterns.extend(progression_patterns)
        
        # Detectar transformaciones
        transformation_patterns = self._detect_transformation_patterns(objects, relations)
        patterns.extend(transformation_patterns)
        
        return {
            'patterns': patterns,
            'num_patterns': len(patterns),
            'dominant_pattern': self._get_dominant_pattern(patterns),
            'pattern_complexity': self._calculate_pattern_complexity(patterns),
            'structural_regularities': self._find_structural_regularities(objects, relations)
        }
    
    def _integrate_scales(self, pixel_level, object_level, relation_level, pattern_level) -> Dict[str, Any]:
        """
        Integración de información multiescala
        """
        return {
            'micro_to_macro': {
                'pixels_to_objects': self._link_pixels_to_objects(pixel_level, object_level),
                'objects_to_relations': self._link_objects_to_relations(object_level, relation_level),
                'relations_to_patterns': self._link_relations_to_patterns(relation_level, pattern_level)
            },
            'cross_scale_features': {
                'local_global_consistency': self._check_local_global_consistency(pixel_level, pattern_level),
                'hierarchical_decomposition': self._hierarchical_decomposition(object_level, relation_level),
                'emergent_properties': self._find_emergent_properties(pixel_level, object_level, pattern_level)
            },
            'scale_invariant_features': self._extract_scale_invariant_features(
                pixel_level, object_level, relation_level, pattern_level
            )
        }
    
    # Métodos auxiliares
    
    def _classify_local_pattern(self, neighbors: List[int], center_color: int) -> str:
        """Clasifica el patrón local de un píxel"""
        if not neighbors:
            return "isolated"
        
        same_color = sum(1 for n in neighbors if n == center_color)
        
        if same_color == len(neighbors):
            return "interior"
        elif same_color >= len(neighbors) * 0.5:
            return "edge"
        elif same_color == 2 and len(neighbors) == 8:
            return "corner"
        else:
            return "boundary"
    
    def _classify_shape(self, mask: np.ndarray, pixels: np.ndarray) -> str:
        """Clasifica el tipo de forma de un objeto"""
        if len(pixels) == 1:
            return "point"
        elif len(pixels) == 2:
            return "pair"
        
        # Verificar si es línea
        if self._is_line(pixels):
            return "line"
        
        # Verificar si es rectángulo
        if self._is_rectangle(mask):
            return "rectangle"
        
        # Verificar formas específicas
        if self._is_L_shape(mask):
            return "L-shape"
        elif self._is_T_shape(mask):
            return "T-shape"
        elif self._is_cross(mask):
            return "cross"
        
        return "irregular"
    
    def _get_relative_position(self, pos1: Tuple, pos2: Tuple) -> str:
        """Determina la posición relativa entre dos objetos"""
        dy = pos2[0] - pos1[0]
        dx = pos2[1] - pos1[1]
        
        if abs(dy) < 0.5:
            return "right" if dx > 0 else "left"
        elif abs(dx) < 0.5:
            return "below" if dy > 0 else "above"
        else:
            if dy > 0 and dx > 0:
                return "diagonal-br"
            elif dy > 0 and dx < 0:
                return "diagonal-bl"
            elif dy < 0 and dx > 0:
                return "diagonal-tr"
            else:
                return "diagonal-tl"
    
    def _check_alignment(self, obj1: ObjectLevel, obj2: ObjectLevel) -> str:
        """Verifica la alineación entre dos objetos"""
        # Alineación horizontal
        if abs(obj1.centroid[0] - obj2.centroid[0]) < 0.5:
            return "horizontal"
        # Alineación vertical
        elif abs(obj1.centroid[1] - obj2.centroid[1]) < 0.5:
            return "vertical"
        # Alineación diagonal
        elif abs((obj1.centroid[0] - obj2.centroid[0]) - 
                (obj1.centroid[1] - obj2.centroid[1])) < 0.5:
            return "diagonal-main"
        elif abs((obj1.centroid[0] - obj2.centroid[0]) + 
                (obj1.centroid[1] - obj2.centroid[1])) < 0.5:
            return "diagonal-anti"
        
        return "none"
    
    def _generate_hierarchy_summary(self, integrated: Dict) -> str:
        """Genera un resumen textual de la jerarquía detectada"""
        summary = []
        
        # Resumen de micro a macro
        micro_macro = integrated.get('micro_to_macro', {})
        if micro_macro:
            summary.append("Estructura jerárquica detectada:")
            summary.append(f"  • Píxeles → {len(micro_macro.get('pixels_to_objects', {}))} objetos")
            summary.append(f"  • Objetos → {len(micro_macro.get('objects_to_relations', {}))} relaciones")
            summary.append(f"  • Relaciones → {len(micro_macro.get('relations_to_patterns', {}))} patrones")
        
        return "\n".join(summary)
    
    # Stubs para métodos complejos (implementar según necesidad)
    
    def _is_line(self, pixels: np.ndarray) -> bool:
        """Verifica si los píxeles forman una línea"""
        if len(pixels) < 3:
            return True
        # Implementar detección de colinearidad
        return False
    
    def _is_rectangle(self, mask: np.ndarray) -> bool:
        """Verifica si la forma es un rectángulo"""
        # Implementar detección de rectángulo
        return False
    
    def _is_L_shape(self, mask: np.ndarray) -> bool:
        """Verifica si la forma es una L"""
        return False
    
    def _is_T_shape(self, mask: np.ndarray) -> bool:
        """Verifica si la forma es una T"""
        return False
    
    def _is_cross(self, mask: np.ndarray) -> bool:
        """Verifica si la forma es una cruz"""
        return False
    
    def _check_adjacency(self, obj1: ObjectLevel, obj2: ObjectLevel) -> bool:
        """Verifica si dos objetos son adyacentes"""
        # Implementar verificación de adyacencia
        return False
    
    def _check_containment(self, obj1: ObjectLevel, obj2: ObjectLevel) -> bool:
        """Verifica si obj1 contiene a obj2"""
        # Implementar verificación de contención
        return False
    
    def _compute_principal_axis(self, pixels: np.ndarray) -> Tuple[float, float, float]:
        """Calcula el eje principal de un conjunto de píxeles"""
        # Implementar PCA o similar
        return 0.0, 0.0, 0.0
    
    # Métodos para resúmenes y estadísticas
    
    def _summarize_local_patterns(self, pixels: List[PixelLevel]) -> Dict[str, int]:
        """Resume los patrones locales encontrados"""
        pattern_counts = defaultdict(int)
        for pixel in pixels:
            pattern_counts[pixel.local_pattern] += 1
        return dict(pattern_counts)
    
    def _find_edge_pixels(self, pixels: List[PixelLevel]) -> List[Tuple[int, int]]:
        """Encuentra píxeles de borde"""
        return [p.position for p in pixels if p.local_pattern in ["edge", "boundary"]]
    
    def _find_corner_pixels(self, pixels: List[PixelLevel]) -> List[Tuple[int, int]]:
        """Encuentra píxeles de esquina"""
        return [p.position for p in pixels if p.local_pattern == "corner"]
    
    def _get_shape_distribution(self, objects: List[ObjectLevel]) -> Dict[str, int]:
        """Obtiene distribución de tipos de forma"""
        shape_counts = defaultdict(int)
        for obj in objects:
            shape_counts[obj.shape_type] += 1
        return dict(shape_counts)
    
    def _get_size_distribution(self, objects: List[ObjectLevel]) -> Dict[str, Any]:
        """Obtiene estadísticas de tamaño"""
        if not objects:
            return {}
        
        areas = [obj.area for obj in objects]
        return {
            'min': min(areas),
            'max': max(areas),
            'mean': np.mean(areas),
            'std': np.std(areas),
            'median': np.median(areas)
        }
    
    def _map_color_to_objects(self, objects: List[ObjectLevel]) -> Dict[int, List[int]]:
        """Mapea colores a IDs de objetos"""
        color_map = defaultdict(list)
        for obj in objects:
            color_map[obj.color].append(obj.id)
        return dict(color_map)
    
    def _build_relation_graph(self, objects: List[ObjectLevel], relations: List[RelationLevel]) -> Dict:
        """Construye grafo de relaciones entre objetos"""
        graph = {obj.id: [] for obj in objects}
        for rel in relations:
            graph[rel.object1_id].append((rel.object2_id, rel.relation_type))
            graph[rel.object2_id].append((rel.object1_id, rel.relation_type))
        return graph
    
    def _analyze_spatial_structure(self, relations: List[RelationLevel]) -> Dict[str, Any]:
        """Analiza la estructura espacial global"""
        return {
            'dominant_alignment': self._find_dominant_alignment(relations),
            'spatial_density': self._calculate_spatial_density(relations),
            'clustering_coefficient': self._calculate_clustering_coefficient(relations)
        }
    
    def _find_object_clusters(self, objects: List[ObjectLevel], relations: List[RelationLevel]) -> List[List[int]]:
        """Encuentra clusters de objetos relacionados"""
        # Implementar clustering basado en relaciones
        return []
    
    def _detect_grid_pattern(self, objects: List[ObjectLevel]) -> Optional[PatternLevel]:
        """Detecta patrones de rejilla"""
        # Implementar detección de grid
        return None
    
    def _detect_symmetry_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de simetría"""
        # Implementar detección de simetría
        return []
    
    def _detect_repetition_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de repetición"""
        # Implementar detección de repetición
        return []
    
    def _detect_progression_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de progresión"""
        # Implementar detección de progresión
        return []
    
    def _detect_transformation_patterns(self, objects: List[ObjectLevel], relations: List[RelationLevel]) -> List[PatternLevel]:
        """Detecta patrones de transformación"""
        # Implementar detección de transformaciones
        return []
    
    def _get_dominant_pattern(self, patterns: List[PatternLevel]) -> Optional[PatternLevel]:
        """Obtiene el patrón dominante"""
        if not patterns:
            return None
        return max(patterns, key=lambda p: p.regularity_score)
    
    def _calculate_pattern_complexity(self, patterns: List[PatternLevel]) -> float:
        """Calcula la complejidad de los patrones"""
        if not patterns:
            return 0.0
        return sum(len(p.objects_involved) for p in patterns) / len(patterns)
    
    def _find_structural_regularities(self, objects: List[ObjectLevel], relations: List[RelationLevel]) -> Dict:
        """Encuentra regularidades estructurales"""
        return {
            'regular_spacing': self._check_regular_spacing(objects),
            'size_consistency': self._check_size_consistency(objects),
            'color_patterns': self._find_color_patterns(objects)
        }
    
    # Métodos de integración multiescala
    
    def _link_pixels_to_objects(self, pixel_level: Dict, object_level: Dict) -> Dict:
        """Vincula información de píxeles con objetos"""
        return {}
    
    def _link_objects_to_relations(self, object_level: Dict, relation_level: Dict) -> Dict:
        """Vincula objetos con sus relaciones"""
        return {}
    
    def _link_relations_to_patterns(self, relation_level: Dict, pattern_level: Dict) -> Dict:
        """Vincula relaciones con patrones"""
        return {}
    
    def _check_local_global_consistency(self, pixel_level: Dict, pattern_level: Dict) -> float:
        """Verifica consistencia entre niveles local y global"""
        return 0.0
    
    def _hierarchical_decomposition(self, object_level: Dict, relation_level: Dict) -> Dict:
        """Descomposición jerárquica de la estructura"""
        return {}
    
    def _find_emergent_properties(self, pixel_level: Dict, object_level: Dict, pattern_level: Dict) -> List[str]:
        """Encuentra propiedades emergentes"""
        return []
    
    def _extract_scale_invariant_features(self, *levels) -> Dict:
        """Extrae características invariantes a la escala"""
        return {}
    
    # Métodos auxiliares adicionales
    
    def _find_dominant_alignment(self, relations: List[RelationLevel]) -> str:
        """Encuentra la alineación dominante"""
        if not relations:
            return "none"
        alignments = [r.alignment for r in relations]
        return max(set(alignments), key=alignments.count)
    
    def _calculate_spatial_density(self, relations: List[RelationLevel]) -> float:
        """Calcula la densidad espacial"""
        if not relations:
            return 0.0
        return np.mean([1.0 / r.distance if r.distance > 0 else 1.0 for r in relations])
    
    def _calculate_clustering_coefficient(self, relations: List[RelationLevel]) -> float:
        """Calcula el coeficiente de clustering"""
        # Implementar cálculo de clustering
        return 0.0
    
    def _check_regular_spacing(self, objects: List[ObjectLevel]) -> bool:
        """Verifica si hay espaciado regular"""
        # Implementar verificación de espaciado
        return False
    
    def _check_size_consistency(self, objects: List[ObjectLevel]) -> bool:
        """Verifica consistencia de tamaños"""
        if not objects:
            return False
        areas = [obj.area for obj in objects]
        return np.std(areas) / np.mean(areas) < 0.2 if np.mean(areas) > 0 else False
    
    def _find_color_patterns(self, objects: List[ObjectLevel]) -> List[str]:
        """Encuentra patrones de color"""
        # Implementar detección de patrones de color
        return []