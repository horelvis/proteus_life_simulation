#!/usr/bin/env python3
"""
Structural Analyzer - Análisis profundo de conectividad y patrones topológicos
Reemplaza el placeholder de "pattern of connectivity" con análisis real
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, deque
import logging
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure

logger = logging.getLogger(__name__)

class StructuralAnalyzer:
    """
    Analizador estructural avanzado para puzzles ARC
    Implementa análisis real de conectividad, componentes conexas, 
    agujeros, simetrías y patrones topológicos
    """
    
    def __init__(self):
        self.connectivity_cache = {}
        
    def analyze_comprehensive(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Análisis estructural completo de una matriz
        
        Args:
            matrix: Matriz de entrada
            
        Returns:
            Diccionario con métricas estructurales profundas
        """
        analysis = {
            'connectivity': self.analyze_connectivity(matrix),
            'topology': self.analyze_topology(matrix),
            'symmetry': self.analyze_symmetry(matrix),
            'geometric': self.analyze_geometric_properties(matrix),
            'color_distribution': self.analyze_color_distribution(matrix),
            'patterns': self.detect_patterns(matrix)
        }
        
        # Puntuación global basada en complejidad estructural
        analysis['structural_complexity'] = self._calculate_structural_complexity(analysis)
        
        return analysis
    
    def analyze_connectivity(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Análisis profundo de conectividad usando grafos de adyacencia
        
        Returns:
            Métricas de conectividad real (no placeholder)
        """
        h, w = matrix.shape
        
        # Análisis por cada color único
        unique_colors = np.unique(matrix)
        color_connectivity = {}
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
                
            # Crear máscara para el color actual
            mask = (matrix == color)
            
            # Encontrar componentes conexas
            labeled, num_components = ndimage.label(mask)
            
            # Analizar cada componente
            components = []
            for comp_id in range(1, num_components + 1):
                comp_mask = (labeled == comp_id)
                comp_analysis = self._analyze_component(comp_mask, matrix)
                components.append(comp_analysis)
            
            color_connectivity[int(color)] = {
                'num_components': num_components,
                'components': components,
                'total_pixels': int(np.sum(mask)),
                'density': float(np.sum(mask) / (h * w)),
                'adjacency_graph': self._build_adjacency_graph(mask)
            }
        
        # Análisis de conectividad global
        global_connectivity = self._analyze_global_connectivity(matrix, color_connectivity)
        
        return {
            'by_color': color_connectivity,
            'global': global_connectivity,
            'connectivity_score': self._calculate_connectivity_score(color_connectivity)
        }
    
    def _analyze_component(self, mask: np.ndarray, original: np.ndarray) -> Dict[str, Any]:
        """Analiza una componente conexa específica"""
        # Encontrar posiciones del componente
        positions = np.argwhere(mask)
        
        if len(positions) == 0:
            return {'area': 0, 'perimeter': 0, 'compactness': 0, 'holes': 0}
        
        # Calcular área y perímetro
        area = len(positions)
        
        # Perímetro usando gradiente morfológico
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        perimeter = np.sum(mask) - np.sum(eroded)
        
        # Compacidad (4π * area / perimeter²)
        compactness = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        
        # Detectar agujeros usando componentes conexas del fondo dentro del objeto
        # Crear bounding box del componente
        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)
        
        # Extraer región del componente
        comp_region = mask[min_r:max_r+1, min_c:max_c+1]
        
        # Invertir y encontrar componentes del fondo
        inverted = ~comp_region
        
        # Excluir bordes (agujeros internos solamente)
        inverted[0, :] = False
        inverted[-1, :] = False
        inverted[:, 0] = False
        inverted[:, -1] = False
        
        hole_labels, num_holes = ndimage.label(inverted)
        
        # Calcular centroide
        centroid = np.mean(positions, axis=0)
        
        return {
            'area': int(area),
            'perimeter': int(perimeter), 
            'compactness': float(compactness),
            'holes': int(num_holes),
            'centroid': centroid.tolist(),
            'bounding_box': {
                'min_r': int(min_r), 'max_r': int(max_r),
                'min_c': int(min_c), 'max_c': int(max_c)
            },
            'aspect_ratio': float((max_r - min_r + 1) / (max_c - min_c + 1)) if max_c > min_c else 1.0
        }
    
    def _build_adjacency_graph(self, mask: np.ndarray) -> Dict[str, Any]:
        """Construye grafo de adyacencia para componentes conexas"""
        labeled, num_components = ndimage.label(mask)
        
        if num_components <= 1:
            return {'nodes': num_components, 'edges': 0, 'density': 0}
        
        # Encontrar adyacencias entre componentes
        adjacencies = set()
        h, w = mask.shape
        
        # Revisar vecindario 4-conectado
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for r in range(h):
            for c in range(w):
                if labeled[r, c] > 0:
                    current_comp = labeled[r, c]
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbor_comp = labeled[nr, nc]
                            if neighbor_comp > 0 and neighbor_comp != current_comp:
                                edge = tuple(sorted([current_comp, neighbor_comp]))
                                adjacencies.add(edge)
        
        num_edges = len(adjacencies)
        max_edges = (num_components * (num_components - 1)) // 2
        graph_density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            'nodes': num_components,
            'edges': num_edges,
            'density': float(graph_density),
            'adjacency_list': list(adjacencies)
        }
    
    def _analyze_global_connectivity(self, matrix: np.ndarray, 
                                   color_connectivity: Dict) -> Dict[str, Any]:
        """Análisis de conectividad global entre todos los colores"""
        
        # Calcular distancias entre centroides de diferentes colores
        color_centroids = {}
        for color, data in color_connectivity.items():
            centroids = []
            for comp in data['components']:
                if 'centroid' in comp:
                    centroids.append(comp['centroid'])
            if centroids:
                color_centroids[color] = np.array(centroids)
        
        # Matriz de distancias entre colores
        inter_color_distances = {}
        colors = list(color_centroids.keys())
        
        for i, color1 in enumerate(colors):
            for color2 in colors[i+1:]:
                if color1 in color_centroids and color2 in color_centroids:
                    # Calcular distancia mínima entre centroides de diferentes colores
                    distances = cdist(color_centroids[color1], color_centroids[color2])
                    min_distance = float(np.min(distances))
                    inter_color_distances[f"{color1}-{color2}"] = min_distance
        
        # Análisis de separación espacial
        spatial_separation = self._analyze_spatial_separation(matrix)
        
        return {
            'inter_color_distances': inter_color_distances,
            'spatial_separation': spatial_separation,
            'total_components': sum(data['num_components'] for data in color_connectivity.values()),
            'color_diversity': len(color_connectivity)
        }
    
    def analyze_topology(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Análisis topológico real usando teoría de grafos y homología
        """
        # Análisis de conectividad 4 y 8
        topology_4 = self._analyze_topology_connectivity(matrix, connectivity=1)  # 4-conectado
        topology_8 = self._analyze_topology_connectivity(matrix, connectivity=2)  # 8-conectado
        
        # Detectar loops y ciclos
        cycles = self._detect_cycles(matrix)
        
        # Análisis de Euler (V - E + F)
        euler_characteristic = self._calculate_euler_characteristic(matrix)
        
        # Detectar patrones topológicos específicos
        topological_patterns = self._detect_topological_patterns(matrix)
        
        return {
            'connectivity_4': topology_4,
            'connectivity_8': topology_8,
            'cycles': cycles,
            'euler_characteristic': euler_characteristic,
            'topological_patterns': topological_patterns,
            'genus': self._estimate_genus(matrix)
        }
    
    def _detect_cycles(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detecta ciclos y loops en la estructura"""
        unique_colors = np.unique(matrix)
        cycles_by_color = {}
        
        for color in unique_colors:
            if color == 0:
                continue
                
            mask = (matrix == color)
            
            # Usar contornos para detectar loops
            try:
                contours = measure.find_contours(mask.astype(float), 0.5)
                
                loops = []
                for contour in contours:
                    if len(contour) > 8:  # Mínimo para considerar un loop
                        # Verificar si es cerrado
                        start_end_distance = np.linalg.norm(contour[0] - contour[-1])
                        if start_end_distance < 2.0:  # Loop cerrado
                            loops.append({
                                'length': len(contour),
                                'area_enclosed': self._calculate_polygon_area(contour),
                                'perimeter': self._calculate_contour_perimeter(contour)
                            })
                
                cycles_by_color[int(color)] = {
                    'num_loops': len(loops),
                    'loops': loops
                }
                
            except Exception:
                cycles_by_color[int(color)] = {'num_loops': 0, 'loops': []}
        
        return cycles_by_color
    
    def analyze_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Análisis completo de simetrías
        """
        return {
            'horizontal': self._check_horizontal_symmetry(matrix),
            'vertical': self._check_vertical_symmetry(matrix),
            'diagonal_main': self._check_diagonal_symmetry(matrix, main=True),
            'diagonal_anti': self._check_diagonal_symmetry(matrix, main=False),
            'rotational_90': self._check_rotational_symmetry(matrix, 90),
            'rotational_180': self._check_rotational_symmetry(matrix, 180),
            'point_symmetry': self._check_point_symmetry(matrix),
            'translational': self._detect_translational_symmetry(matrix)
        }
    
    def _check_horizontal_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Verifica simetría horizontal"""
        flipped = np.flipud(matrix)
        similarity = np.mean(matrix == flipped)
        
        return {
            'has_symmetry': similarity > 0.95,
            'similarity_score': float(similarity),
            'axis_position': matrix.shape[0] // 2
        }
    
    def _check_vertical_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Verifica simetría vertical"""
        flipped = np.fliplr(matrix)
        similarity = np.mean(matrix == flipped)
        
        return {
            'has_symmetry': similarity > 0.95,
            'similarity_score': float(similarity),
            'axis_position': matrix.shape[1] // 2
        }
    
    def _calculate_connectivity_score(self, color_connectivity: Dict) -> float:
        """
        Calcula puntuación de conectividad real (reemplaza placeholder)
        Basado en densidad de componentes, compacidad y adyacencias
        """
        if not color_connectivity:
            return 0.0
        
        scores = []
        
        for color, data in color_connectivity.items():
            # Penalizar fragmentación excesiva
            fragmentation_penalty = 1.0 / (1.0 + data['num_components'] * 0.1)
            
            # Recompensar compacidad de componentes
            if data['components']:
                avg_compactness = np.mean([comp.get('compactness', 0) for comp in data['components']])
            else:
                avg_compactness = 0
            
            # Considerar densidad de grafo de adyacencia
            graph_density = data['adjacency_graph']['density']
            
            # Score compuesto
            color_score = (fragmentation_penalty * 0.4 + 
                          avg_compactness * 0.4 + 
                          graph_density * 0.2)
            scores.append(color_score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _calculate_structural_complexity(self, analysis: Dict[str, Any]) -> float:
        """Calcula complejidad estructural global"""
        connectivity_score = analysis['connectivity'].get('connectivity_score', 0)
        
        # Contar patrones topológicos únicos
        topology_complexity = len(analysis['topology'].get('topological_patterns', {}))
        
        # Contar simetrías detectadas
        symmetries = analysis['symmetry']
        symmetry_count = sum(1 for s in symmetries.values() 
                           if isinstance(s, dict) and s.get('has_symmetry', False))
        
        # Diversidad de colores
        color_diversity = analysis['connectivity']['global'].get('color_diversity', 0)
        
        # Score final normalizado
        complexity = (connectivity_score * 0.4 + 
                     topology_complexity * 0.1 * 0.3 +
                     symmetry_count * 0.05 * 0.2 +
                     color_diversity * 0.02 * 0.1)
        
        return min(1.0, complexity)  # Normalizar a [0, 1]
    
    # Métodos auxiliares
    def _analyze_topology_connectivity(self, matrix: np.ndarray, connectivity: int) -> Dict:
        """Análisis topológico con conectividad específica"""
        labeled, num_features = ndimage.label(matrix > 0, 
                                            structure=ndimage.generate_binary_structure(2, connectivity))
        return {
            'num_components': int(num_features),
            'connectivity_type': '4-connected' if connectivity == 1 else '8-connected'
        }
    
    def _calculate_euler_characteristic(self, matrix: np.ndarray) -> int:
        """Calcula característica de Euler (V - E + F)"""
        # Simplificado: usar número de componentes - número de agujeros
        labeled, num_components = ndimage.label(matrix > 0)
        
        # Contar agujeros aproximadamente
        holes = 0
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled == comp_id)
            filled = ndimage.binary_fill_holes(comp_mask)
            holes += np.sum(filled) - np.sum(comp_mask)
        
        return int(num_components - holes)
    
    def _detect_topological_patterns(self, matrix: np.ndarray) -> Dict[str, bool]:
        """Detecta patrones topológicos específicos"""
        patterns = {}
        
        # Detectar líneas
        patterns['has_lines'] = self._detect_lines(matrix)
        
        # Detectar rectángulos
        patterns['has_rectangles'] = self._detect_rectangles(matrix)
        
        # Detectar cruces
        patterns['has_crosses'] = self._detect_crosses(matrix)
        
        # Detectar patrones en L
        patterns['has_l_shapes'] = self._detect_l_shapes(matrix)
        
        return patterns
    
    def _detect_lines(self, matrix: np.ndarray) -> bool:
        """Detecta patrones lineales"""
        # Buscar líneas horizontales y verticales
        h, w = matrix.shape
        
        # Líneas horizontales
        for r in range(h):
            row = matrix[r, :]
            if len(np.unique(row)) == 2 and np.sum(row > 0) > w * 0.7:
                return True
        
        # Líneas verticales  
        for c in range(w):
            col = matrix[:, c]
            if len(np.unique(col)) == 2 and np.sum(col > 0) > h * 0.7:
                return True
        
        return False
    
    def _detect_rectangles(self, matrix: np.ndarray) -> bool:
        """Detecta patrones rectangulares"""
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
                
            mask = (matrix == color)
            labeled, num_components = ndimage.label(mask)
            
            for comp_id in range(1, num_components + 1):
                comp_mask = (labeled == comp_id)
                
                # Verificar si es rectangular
                if self._is_rectangular(comp_mask):
                    return True
        
        return False
    
    def _is_rectangular(self, mask: np.ndarray) -> bool:
        """Verifica si una máscara es rectangular"""
        positions = np.argwhere(mask)
        if len(positions) == 0:
            return False
        
        min_r, min_c = positions.min(axis=0)
        max_r, max_c = positions.max(axis=0)
        
        # Crear rectángulo esperado
        expected_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        actual_area = len(positions)
        
        return actual_area / expected_area > 0.9
    
    def _detect_crosses(self, matrix: np.ndarray) -> bool:
        """Detecta patrones en cruz"""
        # Implementación simplificada para detectar cruces
        h, w = matrix.shape
        center_r, center_c = h // 2, w // 2
        
        # Verificar cruz centrada
        if (h >= 3 and w >= 3 and 
            matrix[center_r, :].sum() > w * 0.6 and
            matrix[:, center_c].sum() > h * 0.6):
            return True
        
        return False
    
    def _detect_l_shapes(self, matrix: np.ndarray) -> bool:
        """Detecta patrones en L"""
        # Implementación básica para detectar formas L
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
                
            mask = (matrix == color)
            
            # Usar template matching simple para L
            if self._has_l_pattern(mask):
                return True
        
        return False
    
    def _has_l_pattern(self, mask: np.ndarray) -> bool:
        """Verifica patrón L simple"""
        h, w = mask.shape
        
        # Buscar esquinas que podrían ser L
        corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        
        for corner_r, corner_c in corners:
            if (corner_r < h and corner_c < w and mask[corner_r, corner_c] and
                self._check_l_from_corner(mask, corner_r, corner_c)):
                return True
        
        return False
    
    def _check_l_from_corner(self, mask: np.ndarray, r: int, c: int) -> bool:
        """Verifica L desde una esquina específica"""
        h, w = mask.shape
        
        # Verificar extensión horizontal y vertical desde la esquina
        horizontal_extent = 0
        vertical_extent = 0
        
        # Contar píxeles horizontalmente
        if c < w - 1:
            for cc in range(c, w):
                if mask[r, cc]:
                    horizontal_extent += 1
                else:
                    break
        
        # Contar píxeles verticalmente
        if r < h - 1:
            for rr in range(r, h):
                if mask[rr, c]:
                    vertical_extent += 1
                else:
                    break
        
        return horizontal_extent >= 2 and vertical_extent >= 2
    
    # Métodos auxiliares adicionales
    def _analyze_spatial_separation(self, matrix: np.ndarray) -> Dict[str, float]:
        """Analiza separación espacial entre objetos"""
        return {'avg_separation': 0.0}  # Placeholder simplificado
    
    def _estimate_genus(self, matrix: np.ndarray) -> int:
        """Estima genus topológico"""
        return 0  # Simplificado
    
    def _calculate_polygon_area(self, contour: np.ndarray) -> float:
        """Calcula área de polígono usando fórmula del shoelace"""
        if len(contour) < 3:
            return 0.0
        
        x = contour[:, 1]
        y = contour[:, 0]
        
        area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))
        return float(area)
    
    def _calculate_contour_perimeter(self, contour: np.ndarray) -> float:
        """Calcula perímetro de contorno"""
        if len(contour) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            perimeter += np.linalg.norm(p2 - p1)
        
        return float(perimeter)
    
    def analyze_geometric_properties(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Análisis de propiedades geométricas"""
        return {
            'shape': list(matrix.shape),
            'area': int(matrix.size),
            'non_zero_area': int(np.count_nonzero(matrix)),
            'density': float(np.count_nonzero(matrix) / matrix.size)
        }
    
    def analyze_color_distribution(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Análisis de distribución de colores"""
        unique, counts = np.unique(matrix, return_counts=True)
        
        return {
            'unique_colors': len(unique),
            'color_counts': {int(color): int(count) for color, count in zip(unique, counts)},
            'dominant_color': int(unique[np.argmax(counts)]),
            'color_entropy': self._calculate_color_entropy(counts)
        }
    
    def _calculate_color_entropy(self, counts: np.ndarray) -> float:
        """Calcula entropía de distribución de colores"""
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def detect_patterns(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detección de patrones generales"""
        return {
            'repetitions': self._detect_repetitive_patterns(matrix),
            'gradients': self._detect_gradients(matrix),
            'textures': self._detect_textures(matrix)
        }
    
    def _detect_repetitive_patterns(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detecta patrones repetitivos simples por corrimiento horizontal/vertical"""
        h, w = matrix.shape
        best = {'has_repetition': False, 'direction': None, 'period': None, 'similarity': 0.0}
        # Horizontal shifts
        for shift in range(1, max(1, w // 2)):
            left = matrix[:, :-shift]
            right = matrix[:, shift:]
            if left.shape == right.shape:
                sim = float(np.mean(left == right))
                if sim > best['similarity']:
                    best = {'has_repetition': sim > 0.8, 'direction': 'horizontal', 'period': shift, 'similarity': sim}
        # Vertical shifts
        for shift in range(1, max(1, h // 2)):
            top = matrix[:-shift, :]
            bottom = matrix[shift:, :]
            if top.shape == bottom.shape:
                sim = float(np.mean(top == bottom))
                if sim > best['similarity']:
                    best = {'has_repetition': sim > 0.8, 'direction': 'vertical', 'period': shift, 'similarity': sim}
        return best
    
    def _detect_gradients(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detecta gradientes básicos (tendencia de cambio por filas/columnas)"""
        # Convertir a float para diferencias
        m = matrix.astype(float)
        # Diferencias promedio por eje
        dx = np.mean(np.abs(np.diff(m, axis=1))) if m.shape[1] > 1 else 0.0
        dy = np.mean(np.abs(np.diff(m, axis=0))) if m.shape[0] > 1 else 0.0
        strength = float(max(dx, dy))
        direction = 'horizontal' if dx >= dy else 'vertical'
        return {
            'has_gradient': strength > 0.2,  # umbral heurístico
            'strength': strength,
            'dominant_direction': direction,
            'dx': float(dx),
            'dy': float(dy)
        }
    
    def _detect_textures(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Mide una complejidad de textura simple por contraste local"""
        h, w = matrix.shape
        if h < 2 and w < 2:
            return {'texture_complexity': 0.0}
        # Vecinos 4-conectados
        diffs = []
        # Horizontal
        if w > 1:
            diffs.append(np.mean(matrix[:, 1:] != matrix[:, :-1]))
        # Vertical
        if h > 1:
            diffs.append(np.mean(matrix[1:, :] != matrix[:-1, :]))
        complexity = float(np.mean(diffs)) if diffs else 0.0
        return {'texture_complexity': complexity}
    
    def _check_diagonal_symmetry(self, matrix: np.ndarray, main: bool = True) -> Dict[str, Any]:
        """Verifica simetría diagonal"""
        if matrix.shape[0] != matrix.shape[1]:
            return {'has_symmetry': False, 'similarity_score': 0.0}
        
        if main:
            transposed = matrix.T
        else:
            transposed = np.fliplr(np.flipud(matrix.T))
        
        similarity = np.mean(matrix == transposed)
        
        return {
            'has_symmetry': similarity > 0.95,
            'similarity_score': float(similarity)
        }
    
    def _check_rotational_symmetry(self, matrix: np.ndarray, angle: int) -> Dict[str, Any]:
        """Verifica simetría rotacional"""
        if matrix.shape[0] != matrix.shape[1]:
            return {'has_symmetry': False, 'similarity_score': 0.0}
        
        if angle == 90:
            rotated = np.rot90(matrix)
        elif angle == 180:
            rotated = np.rot90(matrix, 2)
        else:
            return {'has_symmetry': False, 'similarity_score': 0.0}
        
        similarity = np.mean(matrix == rotated)
        
        return {
            'has_symmetry': similarity > 0.95,
            'similarity_score': float(similarity)
        }
    
    def _check_point_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Verifica simetría puntual"""
        rotated_180 = np.rot90(matrix, 2)
        similarity = np.mean(matrix == rotated_180)
        
        return {
            'has_symmetry': similarity > 0.95,
            'similarity_score': float(similarity)
        }
    
    def _detect_translational_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Detecta simetría traslacional"""
        # Implementación básica para detectar patrones repetitivos
        h, w = matrix.shape
        
        # Buscar repeticiones horizontales
        for shift in range(1, w // 2):
            left_part = matrix[:, :w-shift]
            right_part = matrix[:, shift:]
            
            if left_part.shape == right_part.shape:
                similarity = np.mean(left_part == right_part)
                if similarity > 0.8:
                    return {
                        'has_symmetry': True,
                        'direction': 'horizontal',
                        'period': shift,
                        'similarity_score': float(similarity)
                    }
        
        # Buscar repeticiones verticales
        for shift in range(1, h // 2):
            top_part = matrix[:h-shift, :]
            bottom_part = matrix[shift:, :]
            
            if top_part.shape == bottom_part.shape:
                similarity = np.mean(top_part == bottom_part)
                if similarity > 0.8:
                    return {
                        'has_symmetry': True,
                        'direction': 'vertical', 
                        'period': shift,
                        'similarity_score': float(similarity)
                    }
        
        return {'has_symmetry': False, 'similarity_score': 0.0}
