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
        if not objects:
            return []
        
        # Crear grafo de adyacencia basado en relaciones
        adjacency = defaultdict(set)
        for rel in relations:
            if rel.relation_type in ['adjacent', 'aligned', 'contains']:
                adjacency[rel.object1_id].add(rel.object2_id)
                adjacency[rel.object2_id].add(rel.object1_id)
        
        # Encontrar componentes conectados
        visited = set()
        clusters = []
        
        for obj in objects:
            if obj.id not in visited:
                # BFS para encontrar cluster
                cluster = []
                queue = [obj.id]
                
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        queue.extend(adjacency[current] - visited)
                
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _detect_grid_pattern(self, objects: List[ObjectLevel]) -> Optional[PatternLevel]:
        """Detecta patrones de rejilla"""
        if len(objects) < 4:
            return None
        
        # Extraer centroides
        centroids = np.array([obj.centroid for obj in objects])
        
        # Verificar alineación en filas y columnas
        y_coords = centroids[:, 0]
        x_coords = centroids[:, 1]
        
        # Encontrar valores únicos con tolerancia
        y_unique = np.unique(np.round(y_coords, 1))
        x_unique = np.unique(np.round(x_coords, 1))
        
        # Verificar si forma una rejilla regular
        if len(y_unique) > 1 and len(x_unique) > 1:
            # Calcular espaciado
            y_spacing = np.diff(sorted(y_unique))
            x_spacing = np.diff(sorted(x_unique))
            
            # Verificar regularidad
            y_regular = np.std(y_spacing) < 0.5 if len(y_spacing) > 0 else False
            x_regular = np.std(x_spacing) < 0.5 if len(x_spacing) > 0 else False
            
            if y_regular and x_regular:
                regularity = 1.0 - (np.std(y_spacing) + np.std(x_spacing)) / 2
                return PatternLevel(
                    pattern_type='grid',
                    objects_involved=[obj.id for obj in objects],
                    regularity_score=float(regularity),
                    transformation=None,
                    parameters={
                        'rows': len(y_unique),
                        'cols': len(x_unique),
                        'y_spacing': float(np.mean(y_spacing)) if len(y_spacing) > 0 else 0,
                        'x_spacing': float(np.mean(x_spacing)) if len(x_spacing) > 0 else 0
                    }
                )
        
        return None
    
    def _detect_symmetry_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de simetría"""
        patterns = []
        if len(objects) < 2:
            return patterns
        
        # Calcular centro de masa global
        all_positions = []
        for obj in objects:
            all_positions.extend(obj.pixels)
        
        if not all_positions:
            return patterns
        
        center = np.mean(all_positions, axis=0)
        
        # Verificar simetría horizontal
        h_symmetric = True
        for obj in objects:
            mirrored_y = 2 * center[0] - obj.centroid[0]
            mirrored_x = obj.centroid[1]
            
            # Buscar objeto espejo
            found_mirror = False
            for other in objects:
                if other.id != obj.id:
                    dist = np.sqrt((other.centroid[0] - mirrored_y)**2 + 
                                  (other.centroid[1] - mirrored_x)**2)
                    if dist < 1.5 and other.color == obj.color:
                        found_mirror = True
                        break
            
            if not found_mirror:
                h_symmetric = False
                break
        
        if h_symmetric:
            patterns.append(PatternLevel(
                pattern_type='symmetry',
                objects_involved=[obj.id for obj in objects],
                regularity_score=0.9,
                transformation='horizontal_reflection',
                parameters={'axis': float(center[0])}
            ))
        
        # Verificar simetría vertical
        v_symmetric = True
        for obj in objects:
            mirrored_y = obj.centroid[0]
            mirrored_x = 2 * center[1] - obj.centroid[1]
            
            # Buscar objeto espejo
            found_mirror = False
            for other in objects:
                if other.id != obj.id:
                    dist = np.sqrt((other.centroid[0] - mirrored_y)**2 + 
                                  (other.centroid[1] - mirrored_x)**2)
                    if dist < 1.5 and other.color == obj.color:
                        found_mirror = True
                        break
            
            if not found_mirror:
                v_symmetric = False
                break
        
        if v_symmetric:
            patterns.append(PatternLevel(
                pattern_type='symmetry',
                objects_involved=[obj.id for obj in objects],
                regularity_score=0.9,
                transformation='vertical_reflection',
                parameters={'axis': float(center[1])}
            ))
        
        return patterns
    
    def _detect_repetition_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de repetición"""
        patterns = []
        if len(objects) < 2:
            return patterns
        
        # Agrupar objetos por forma y color
        shape_groups = defaultdict(list)
        for obj in objects:
            key = (obj.color, obj.area, obj.shape_type)
            shape_groups[key].append(obj)
        
        # Buscar grupos con múltiples instancias
        for key, group in shape_groups.items():
            if len(group) >= 2:
                # Calcular distancias entre objetos del grupo
                positions = np.array([obj.centroid for obj in group])
                
                if len(positions) >= 2:
                    # Calcular todas las distancias por pares
                    distances = cdist(positions, positions)
                    
                    # Excluir diagonal (distancia a sí mismo)
                    mask = ~np.eye(len(positions), dtype=bool)
                    valid_distances = distances[mask]
                    
                    if len(valid_distances) > 0:
                        # Verificar si hay distancias regulares
                        mean_dist = np.mean(valid_distances)
                        std_dist = np.std(valid_distances)
                        
                        # Si la desviación es baja, hay repetición regular
                        if mean_dist > 0 and std_dist / mean_dist < 0.3:
                            regularity = 1.0 - (std_dist / mean_dist)
                            patterns.append(PatternLevel(
                                pattern_type='repetition',
                                objects_involved=[obj.id for obj in group],
                                regularity_score=float(regularity),
                                transformation='translation',
                                parameters={
                                    'count': len(group),
                                    'mean_distance': float(mean_dist),
                                    'color': key[0],
                                    'area': key[1]
                                }
                            ))
        
        return patterns
    
    def _detect_progression_patterns(self, objects: List[ObjectLevel]) -> List[PatternLevel]:
        """Detecta patrones de progresión"""
        patterns = []
        if len(objects) < 3:
            return patterns
        
        # Verificar progresión por tamaño
        objects_sorted = sorted(objects, key=lambda o: o.centroid[0])  # Ordenar por posición Y
        areas = [obj.area for obj in objects_sorted]
        
        if len(areas) >= 3:
            # Verificar si las áreas forman una progresión
            diffs = np.diff(areas)
            
            # Progresión aritmética
            if len(diffs) > 0 and np.std(diffs) < 0.5:
                patterns.append(PatternLevel(
                    pattern_type='progression',
                    objects_involved=[obj.id for obj in objects_sorted],
                    regularity_score=0.8,
                    transformation='size_progression',
                    parameters={
                        'type': 'arithmetic',
                        'step': float(np.mean(diffs)),
                        'start_size': float(areas[0])
                    }
                ))
            
            # Progresión geométrica
            elif len(areas) >= 2 and all(a > 0 for a in areas):
                ratios = [areas[i+1] / areas[i] for i in range(len(areas)-1)]
                if np.std(ratios) < 0.1:
                    patterns.append(PatternLevel(
                        pattern_type='progression',
                        objects_involved=[obj.id for obj in objects_sorted],
                        regularity_score=0.8,
                        transformation='size_progression',
                        parameters={
                            'type': 'geometric',
                            'ratio': float(np.mean(ratios)),
                            'start_size': float(areas[0])
                        }
                    ))
        
        # Verificar progresión por color
        colors = [obj.color for obj in objects_sorted]
        if len(set(colors)) == len(colors):  # Todos colores diferentes
            color_diffs = np.diff(colors)
            if len(color_diffs) > 0 and np.std(color_diffs) < 0.5:
                patterns.append(PatternLevel(
                    pattern_type='progression',
                    objects_involved=[obj.id for obj in objects_sorted],
                    regularity_score=0.7,
                    transformation='color_progression',
                    parameters={
                        'step': float(np.mean(color_diffs)),
                        'start_color': colors[0]
                    }
                ))
        
        return patterns
    
    def _detect_transformation_patterns(self, objects: List[ObjectLevel], relations: List[RelationLevel]) -> List[PatternLevel]:
        """Detecta patrones de transformación"""
        patterns = []
        if len(objects) < 2:
            return patterns
        
        # Agrupar objetos por color
        color_groups = defaultdict(list)
        for obj in objects:
            color_groups[obj.color].append(obj)
        
        # Buscar rotaciones
        for color, group in color_groups.items():
            if len(group) >= 2:
                # Comparar formas para detectar rotaciones
                base_obj = group[0]
                for obj in group[1:]:
                    if obj.area == base_obj.area:
                        # Verificar si es una rotación
                        angle_diff = abs(obj.orientation - base_obj.orientation)
                        if angle_diff > 0 and (angle_diff % 90 < 5 or angle_diff % 90 > 85):
                            patterns.append(PatternLevel(
                                pattern_type='transformation',
                                objects_involved=[base_obj.id, obj.id],
                                regularity_score=0.85,
                                transformation='rotation',
                                parameters={
                                    'angle': float(angle_diff),
                                    'center': list(base_obj.centroid)
                                }
                            ))
        
        # Buscar escalado
        for rel in relations:
            if rel.size_ratio != 1.0:
                # Verificar si mantienen la misma forma pero diferente tamaño
                obj1 = next((o for o in objects if o.id == rel.object1_id), None)
                obj2 = next((o for o in objects if o.id == rel.object2_id), None)
                
                if obj1 and obj2 and obj1.shape_type == obj2.shape_type:
                    patterns.append(PatternLevel(
                        pattern_type='transformation',
                        objects_involved=[rel.object1_id, rel.object2_id],
                        regularity_score=0.8,
                        transformation='scaling',
                        parameters={
                            'scale_factor': float(rel.size_ratio),
                            'center': list(obj1.centroid)
                        }
                    ))
        
        return patterns
    
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
        links = {
            'pixel_to_object_map': {},
            'object_pixel_stats': {},
            'boundary_pixels': {},
            'interior_pixels': {}
        }
        
        # Crear un mapa de posición a píxel para búsqueda rápida
        pixel_map = {}
        for pixel in pixel_level.get('pixels', []):
            pixel_map[pixel.position] = pixel
        
        # Mapear cada píxel a su objeto
        for obj in object_level.get('objects', []):
            obj_pixels = set(map(tuple, obj.pixels))
            boundary = []
            interior = []
            
            for px_pos in obj.pixels:
                px_key = tuple(px_pos)
                links['pixel_to_object_map'][px_key] = obj.id
                
                # Determinar si es borde o interior
                px_info = pixel_map.get(px_key)
                if px_info and px_info.local_pattern in ['edge', 'corner']:
                    boundary.append(px_key)
                else:
                    interior.append(px_key)
            
            links['boundary_pixels'][obj.id] = boundary
            links['interior_pixels'][obj.id] = interior
            
            # Estadísticas de píxeles por objeto
            links['object_pixel_stats'][obj.id] = {
                'total_pixels': len(obj.pixels),
                'boundary_count': len(boundary),
                'interior_count': len(interior),
                'boundary_ratio': len(boundary) / max(len(obj.pixels), 1)
            }
        
        return links
    
    def _link_objects_to_relations(self, object_level: Dict, relation_level: Dict) -> Dict:
        """Vincula objetos con sus relaciones"""
        links = {
            'object_relations': defaultdict(list),
            'relation_counts': {},
            'relation_types_per_object': defaultdict(set),
            'strongly_connected': []
        }
        
        # Indexar relaciones por objeto
        for rel in relation_level.get('relations', []):
            links['object_relations'][rel.object1_id].append(rel)
            links['object_relations'][rel.object2_id].append(rel)
            
            links['relation_types_per_object'][rel.object1_id].add(rel.relation_type)
            links['relation_types_per_object'][rel.object2_id].add(rel.relation_type)
        
        # Contar relaciones por objeto
        for obj in object_level.get('objects', []):
            obj_relations = links['object_relations'][obj.id]
            links['relation_counts'][obj.id] = len(obj_relations)
            
            # Identificar objetos fuertemente conectados (>3 relaciones)
            if len(obj_relations) > 3:
                links['strongly_connected'].append(obj.id)
        
        return links
    
    def _link_relations_to_patterns(self, relation_level: Dict, pattern_level: Dict) -> Dict:
        """Vincula relaciones con patrones"""
        links = {
            'pattern_supporting_relations': defaultdict(list),
            'relation_pattern_participation': defaultdict(list),
            'critical_relations': []
        }
        
        # Para cada patrón, identificar relaciones que lo soportan
        for pattern in pattern_level.get('patterns', []):
            pattern_objects = set(pattern.objects_involved)
            
            for rel in relation_level.get('relations', []):
                # Si ambos objetos están en el patrón
                if rel.object1_id in pattern_objects and rel.object2_id in pattern_objects:
                    links['pattern_supporting_relations'][pattern.pattern_type].append(rel)
                    links['relation_pattern_participation'][(
                        rel.object1_id, rel.object2_id
                    )].append(pattern.pattern_type)
                    
                    # Relaciones críticas para patrones de alta regularidad
                    if pattern.regularity_score > 0.8:
                        links['critical_relations'].append(rel)
        
        return links
    
    def _check_local_global_consistency(self, pixel_level: Dict, pattern_level: Dict) -> float:
        """Verifica consistencia entre niveles local y global"""
        if not pixel_level or not pattern_level:
            return 0.0
        
        consistency_scores = []
        
        # Obtener posiciones de píxeles
        pixel_positions = [p.position for p in pixel_level.get('pixels', [])]
        
        # Verificar que los patrones globales se reflejan en características locales
        for pattern in pattern_level.get('patterns', []):
            if pattern.pattern_type == 'symmetry':
                # Verificar simetría en distribución de píxeles
                axis = pattern.parameters.get('axis', 0)
                
                if pixel_positions:
                    # Contar píxeles a cada lado del eje
                    left_count = sum(1 for pos in pixel_positions if pos[0] < axis)
                    right_count = sum(1 for pos in pixel_positions if pos[0] > axis)
                    
                    if left_count + right_count > 0:
                        balance = 1.0 - abs(left_count - right_count) / (left_count + right_count)
                        consistency_scores.append(balance)
            
            elif pattern.pattern_type == 'grid':
                # Verificar regularidad en distribución de píxeles
                spacing = pattern.parameters.get('y_spacing', 1)
                if spacing > 0:
                    regularity = pattern.regularity_score
                    consistency_scores.append(regularity)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _hierarchical_decomposition(self, object_level: Dict, relation_level: Dict) -> Dict:
        """Descomposición jerárquica de la estructura"""
        decomposition = {
            'hierarchy_levels': [],
            'parent_child_relations': defaultdict(list),
            'hierarchy_depth': 0
        }
        
        objects = object_level.get('objects', [])
        if not objects:
            return decomposition
        
        # Nivel 0: Objetos individuales
        level_0 = [{'id': obj.id, 'type': 'atomic', 'size': obj.area} for obj in objects]
        decomposition['hierarchy_levels'].append(level_0)
        
        # Nivel 1: Grupos basados en proximidad
        clusters = self._find_object_clusters(objects, relation_level.get('relations', []))
        if clusters:
            level_1 = []
            for i, cluster in enumerate(clusters):
                cluster_id = f"cluster_{i}"
                level_1.append({
                    'id': cluster_id,
                    'type': 'cluster',
                    'members': cluster,
                    'size': len(cluster)
                })
                
                # Establecer relaciones padre-hijo
                for obj_id in cluster:
                    decomposition['parent_child_relations'][cluster_id].append(obj_id)
            
            decomposition['hierarchy_levels'].append(level_1)
            decomposition['hierarchy_depth'] = len(decomposition['hierarchy_levels'])
        
        return decomposition
    
    def _find_emergent_properties(self, pixel_level: Dict, object_level: Dict, pattern_level: Dict) -> List[str]:
        """Encuentra propiedades emergentes"""
        properties = []
        
        # Propiedad emergente: Formación de estructura compleja desde elementos simples
        num_pixels = len(pixel_level.get('pixels', {}))
        num_objects = len(object_level.get('objects', []))
        num_patterns = len(pattern_level.get('patterns', []))
        
        if num_pixels > 0 and num_objects > 0:
            complexity_ratio = num_objects / num_pixels
            if complexity_ratio < 0.2:
                properties.append('sparse_structure')
            elif complexity_ratio > 0.8:
                properties.append('dense_structure')
        
        # Propiedad emergente: Auto-organización
        if num_patterns > 0:
            max_regularity = max(p.regularity_score for p in pattern_level.get('patterns', []))
            if max_regularity > 0.9:
                properties.append('highly_organized')
            elif max_regularity > 0.7:
                properties.append('moderately_organized')
        
        # Propiedad emergente: Modularidad
        if object_level.get('objects'):
            unique_shapes = len(set(obj.shape_type for obj in object_level['objects']))
            if unique_shapes == 1:
                properties.append('homogeneous_modules')
            elif unique_shapes > 3:
                properties.append('heterogeneous_modules')
        
        # Propiedad emergente: Jerarquía
        if 'symmetry' in [p.pattern_type for p in pattern_level.get('patterns', [])]:
            properties.append('symmetrical_hierarchy')
        
        if 'grid' in [p.pattern_type for p in pattern_level.get('patterns', [])]:
            properties.append('regular_lattice')
        
        return properties
    
    def _extract_scale_invariant_features(self, *levels) -> Dict:
        """Extrae características invariantes a la escala"""
        features = {
            'density': 0.0,
            'connectivity': 0.0,
            'regularity': 0.0,
            'complexity': 0.0,
            'symmetry_score': 0.0
        }
        
        # Extraer de todos los niveles pasados
        for level in levels:
            if isinstance(level, dict):
                # Densidad: proporción de espacio ocupado
                if 'pixels' in level:
                    total_pixels = len(level['pixels'])
                    if 'bounds' in level:
                        area = level['bounds']['width'] * level['bounds']['height']
                        features['density'] = total_pixels / max(area, 1)
                
                # Conectividad: promedio de conexiones por objeto
                if 'relations' in level:
                    num_relations = len(level['relations'])
                    num_objects = len(level.get('objects', [1]))
                    features['connectivity'] = num_relations / max(num_objects, 1)
                
                # Regularidad: de los patrones detectados
                if 'patterns' in level:
                    regularities = [p.regularity_score for p in level['patterns']]
                    if regularities:
                        features['regularity'] = float(np.mean(regularities))
                
                # Complejidad: diversidad de elementos
                if 'objects' in level:
                    unique_colors = len(set(obj.color for obj in level['objects']))
                    unique_shapes = len(set(obj.shape_type for obj in level['objects']))
                    features['complexity'] = (unique_colors + unique_shapes) / 2.0
                
                # Simetría
                if 'patterns' in level:
                    symmetry_patterns = [p for p in level['patterns'] 
                                       if p.pattern_type == 'symmetry']
                    if symmetry_patterns:
                        features['symmetry_score'] = max(p.regularity_score 
                                                        for p in symmetry_patterns)
        
        return features
    
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