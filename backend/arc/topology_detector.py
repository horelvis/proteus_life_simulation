#!/usr/bin/env python3
"""
Topology Detector - Detector de patrones topológicos con curvas y campos
Implementa análisis topológico avanzado para especialización "topology"
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage, spatial
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class TopologyDetector:
    """
    Detector avanzado de patrones topológicos para puzzles ARC
    Incluye análisis de curvas, campos vectoriales, y transformaciones topológicas
    """
    
    def __init__(self):
        self.field_cache = {}
        self.curve_cache = {}
        
    def detect_comprehensive_topology(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Detección comprensiva de patrones topológicos
        
        Returns:
            Diccionario con análisis topológico completo
        """
        return {
            'curves': self.analyze_curves(matrix),
            'vector_fields': self.analyze_vector_fields(matrix),
            'topology_transformations': self.detect_topology_transforms(matrix),
            'flow_patterns': self.analyze_flow_patterns(matrix),
            'critical_points': self.find_critical_points(matrix),
            'homology': self.compute_basic_homology(matrix),
            'contour_topology': self.analyze_contour_topology(matrix)
        }
    
    def analyze_curves(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analiza curvas y trayectorias en la matriz
        """
        curves_analysis = {
            'smooth_curves': self._detect_smooth_curves(matrix),
            'spline_fits': self._fit_splines_to_objects(matrix),
            'curve_continuity': self._analyze_curve_continuity(matrix),
            'curve_intersections': self._find_curve_intersections(matrix),
            'curvature_analysis': self._analyze_curvature(matrix)
        }
        
        return curves_analysis
    
    def _detect_smooth_curves(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta curvas suaves en objetos"""
        curves = []
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            
            # Encontrar contornos de objetos
            try:
                from skimage import measure
                contours = measure.find_contours(mask.astype(float), 0.5)
                
                for i, contour in enumerate(contours):
                    if len(contour) > 10:  # Mínimo para análisis de curvatura
                        curve_analysis = self._analyze_single_curve(contour)
                        curve_analysis['color'] = int(color)
                        curve_analysis['contour_id'] = i
                        curves.append(curve_analysis)
                        
            except Exception as e:
                logger.debug(f"Error detecting curves for color {color}: {e}")
        
        return curves
    
    def _analyze_single_curve(self, curve: np.ndarray) -> Dict[str, Any]:
        """Analiza una curva individual"""
        # Calcular longitud de curva
        curve_length = self._calculate_curve_length(curve)
        
        # Calcular curvatura local
        curvatures = self._calculate_local_curvature(curve)
        
        # Detectar puntos críticos de curvatura
        critical_points = self._find_curvature_critical_points(curvatures)
        
        # Analizar simetría de curva
        symmetry = self._analyze_curve_symmetry(curve)
        
        # Clasificar tipo de curva
        curve_type = self._classify_curve_type(curve, curvatures)
        
        return {
            'length': float(curve_length),
            'points_count': len(curve),
            'avg_curvature': float(np.mean(np.abs(curvatures))),
            'max_curvature': float(np.max(np.abs(curvatures))),
            'critical_points': critical_points,
            'symmetry_score': symmetry,
            'curve_type': curve_type,
            'smoothness': self._calculate_smoothness(curvatures)
        }
    
    def _calculate_curve_length(self, curve: np.ndarray) -> float:
        """Calcula longitud de curva usando diferencias finitas"""
        if len(curve) < 2:
            return 0.0
        
        diffs = np.diff(curve, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(segment_lengths)
    
    def _calculate_local_curvature(self, curve: np.ndarray) -> np.ndarray:
        """Calcula curvatura local usando diferencias finitas"""
        if len(curve) < 3:
            return np.array([0.0])
        
        # Calcular primera y segunda derivadas
        dx = np.gradient(curve[:, 1])  # x coordinates
        dy = np.gradient(curve[:, 0])  # y coordinates
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvatura: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)
        
        # Evitar división por cero
        denominator = np.maximum(denominator, 1e-10)
        
        curvature = numerator / denominator
        return curvature
    
    def _find_curvature_critical_points(self, curvatures: np.ndarray) -> List[Dict[str, Any]]:
        """Encuentra puntos críticos de curvatura"""
        if len(curvatures) < 3:
            return []
        
        # Encontrar máximos locales de curvatura
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(curvatures, height=np.mean(curvatures))
        
        critical_points = []
        for i, peak_idx in enumerate(peaks):
            critical_points.append({
                'index': int(peak_idx),
                'curvature': float(curvatures[peak_idx]),
                'type': 'high_curvature'
            })
        
        return critical_points
    
    def _analyze_curve_symmetry(self, curve: np.ndarray) -> float:
        """Analiza simetría de curva"""
        if len(curve) < 4:
            return 0.0
        
        # Calcular centro de masa
        centroid = np.mean(curve, axis=0)
        
        # Probar simetría reflection sobre diferentes ejes
        symmetry_scores = []
        
        # Simetría horizontal
        reflected_h = curve.copy()
        reflected_h[:, 0] = 2 * centroid[0] - reflected_h[:, 0]
        
        # Encontrar mejor alineación
        min_distances = []
        for point in curve:
            distances = np.sqrt(np.sum((reflected_h - point)**2, axis=1))
            min_distances.append(np.min(distances))
        
        symmetry_h = 1.0 / (1.0 + np.mean(min_distances))
        symmetry_scores.append(symmetry_h)
        
        # Simetría vertical
        reflected_v = curve.copy()
        reflected_v[:, 1] = 2 * centroid[1] - reflected_v[:, 1]
        
        min_distances = []
        for point in curve:
            distances = np.sqrt(np.sum((reflected_v - point)**2, axis=1))
            min_distances.append(np.min(distances))
        
        symmetry_v = 1.0 / (1.0 + np.mean(min_distances))
        symmetry_scores.append(symmetry_v)
        
        return float(np.max(symmetry_scores))
    
    def _classify_curve_type(self, curve: np.ndarray, curvatures: np.ndarray) -> str:
        """Clasifica tipo de curva basado en características"""
        if len(curve) < 5:
            return 'point'
        
        # Analizar linearidad
        start_point = curve[0]
        end_point = curve[-1]
        
        # Calcular distancia total de puntos a la línea recta
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 1e-10:
            return 'point'
        
        line_unit = line_vector / line_length
        
        total_deviation = 0.0
        for point in curve[1:-1]:
            point_vector = point - start_point
            projection_length = np.dot(point_vector, line_unit)
            projection_point = start_point + projection_length * line_unit
            deviation = np.linalg.norm(point - projection_point)
            total_deviation += deviation
        
        avg_deviation = total_deviation / max(1, len(curve) - 2)
        
        if avg_deviation < 1.0:
            return 'line'
        
        # Analizar curvatura promedio
        avg_curvature = np.mean(np.abs(curvatures))
        
        if avg_curvature < 0.1:
            return 'nearly_straight'
        elif avg_curvature > 0.5:
            return 'highly_curved'
        else:
            return 'moderate_curve'
    
    def _calculate_smoothness(self, curvatures: np.ndarray) -> float:
        """Calcula suavidad de curva basada en variación de curvatura"""
        if len(curvatures) < 2:
            return 1.0
        
        curvature_variation = np.std(curvatures)
        smoothness = 1.0 / (1.0 + curvature_variation)
        return float(smoothness)
    
    def _fit_splines_to_objects(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Ajusta splines a objetos para análisis de suavidad"""
        splines = []
        unique_colors = np.unique(matrix)
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (matrix == color)
            labeled, num_objects = ndimage.label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                positions = np.argwhere(obj_mask)
                
                if len(positions) > 5:  # Mínimo para spline fitting
                    try:
                        spline_analysis = self._fit_spline_to_points(positions)
                        spline_analysis['color'] = int(color)
                        spline_analysis['object_id'] = obj_id
                        splines.append(spline_analysis)
                    except Exception as e:
                        logger.debug(f"Error fitting spline to object {obj_id}, color {color}: {e}")
        
        return splines
    
    def _fit_spline_to_points(self, points: np.ndarray) -> Dict[str, Any]:
        """Ajusta spline a conjunto de puntos"""
        try:
            # Ordenar puntos por distancia desde el primero
            distances = np.sqrt(np.sum((points - points[0])**2, axis=1))
            sorted_indices = np.argsort(distances)
            sorted_points = points[sorted_indices]
            
            # Fitting spline
            if len(sorted_points) >= 4:
                tck, u = splprep([sorted_points[:, 0], sorted_points[:, 1]], 
                                s=len(sorted_points), k=min(3, len(sorted_points)-1))
                
                # Evaluar spline en puntos finos
                u_fine = np.linspace(0, 1, len(sorted_points) * 2)
                spline_points = splev(u_fine, tck)
                
                # Calcular error de fitting
                fitting_error = self._calculate_spline_fitting_error(sorted_points, tck, u)
                
                return {
                    'fitting_error': float(fitting_error),
                    'spline_length': self._calculate_spline_length(tck),
                    'control_points': len(sorted_points),
                    'smoothness_factor': len(sorted_points),  # Higher = smoother
                    'has_good_fit': fitting_error < 2.0
                }
            else:
                return {
                    'fitting_error': float('inf'),
                    'spline_length': 0.0,
                    'control_points': len(sorted_points),
                    'smoothness_factor': 0.0,
                    'has_good_fit': False
                }
                
        except Exception:
            return {
                'fitting_error': float('inf'),
                'spline_length': 0.0,
                'control_points': len(points),
                'smoothness_factor': 0.0,
                'has_good_fit': False
            }
    
    def analyze_vector_fields(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analiza campos vectoriales derivados de la estructura
        """
        # Calcular gradiente de la matriz
        grad_y, grad_x = np.gradient(matrix.astype(float))
        
        # Magnitud del gradiente
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Dirección del gradiente
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Análisis de divergencia y rotacional
        div_x, _ = np.gradient(grad_x)
        _, div_y = np.gradient(grad_y)
        divergence = div_x + div_y
        
        # Rotacional (curl) para campo 2D
        curl_z = np.gradient(grad_y, axis=1) - np.gradient(grad_x, axis=0)
        
        # Detectar puntos críticos del campo vectorial
        critical_points = self._find_vector_field_critical_points(grad_x, grad_y)
        
        # Análisis de flujo
        flow_analysis = self._analyze_flow_characteristics(grad_x, grad_y)
        
        return {
            'gradient_magnitude': {
                'mean': float(np.mean(gradient_magnitude)),
                'max': float(np.max(gradient_magnitude)),
                'std': float(np.std(gradient_magnitude))
            },
            'gradient_direction': {
                'mean': float(np.mean(gradient_direction)),
                'std': float(np.std(gradient_direction))
            },
            'divergence': {
                'mean': float(np.mean(divergence)),
                'max': float(np.max(np.abs(divergence))),
                'sources': int(np.sum(divergence > np.std(divergence))),
                'sinks': int(np.sum(divergence < -np.std(divergence)))
            },
            'curl': {
                'mean': float(np.mean(curl_z)),
                'max': float(np.max(np.abs(curl_z))),
                'vorticity': float(np.mean(np.abs(curl_z)))
            },
            'critical_points': critical_points,
            'flow_characteristics': flow_analysis
        }
    
    def _find_vector_field_critical_points(self, vx: np.ndarray, vy: np.ndarray) -> List[Dict[str, Any]]:
        """Encuentra puntos críticos del campo vectorial"""
        critical_points = []
        h, w = vx.shape
        
        # Buscar puntos donde la velocidad es cercana a cero
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        threshold = np.mean(velocity_magnitude) * 0.1
        
        # Encontrar candidatos a puntos críticos
        candidates = np.where(velocity_magnitude < threshold)
        
        for i in range(len(candidates[0])):
            r, c = candidates[0][i], candidates[1][i]
            
            # Analizar vecindario para clasificar el punto crítico
            if 1 <= r < h-1 and 1 <= c < w-1:
                # Calcular Jacobiano local
                jacobian = self._calculate_local_jacobian(vx, vy, r, c)
                eigenvals = np.linalg.eigvals(jacobian)
                
                # Clasificar tipo de punto crítico
                critical_type = self._classify_critical_point(eigenvals)
                
                critical_points.append({
                    'position': (int(r), int(c)),
                    'type': critical_type,
                    'eigenvalues': eigenvals.tolist(),
                    'strength': float(velocity_magnitude[r, c])
                })
        
        return critical_points
    
    def _calculate_local_jacobian(self, vx: np.ndarray, vy: np.ndarray, 
                                 r: int, c: int) -> np.ndarray:
        """Calcula Jacobiano local del campo vectorial"""
        # Derivadas parciales usando diferencias centrales
        dvx_dx = (vx[r, c+1] - vx[r, c-1]) / 2.0
        dvx_dy = (vx[r+1, c] - vx[r-1, c]) / 2.0
        dvy_dx = (vy[r, c+1] - vy[r, c-1]) / 2.0
        dvy_dy = (vy[r+1, c] - vy[r-1, c]) / 2.0
        
        return np.array([[dvx_dx, dvx_dy],
                        [dvy_dx, dvy_dy]])
    
    def _classify_critical_point(self, eigenvals: np.ndarray) -> str:
        """Clasifica punto crítico basado en eigenvalores"""
        real_parts = np.real(eigenvals)
        imag_parts = np.imag(eigenvals)
        
        if np.all(np.abs(imag_parts) < 1e-10):  # Eigenvalores reales
            if np.all(real_parts > 0):
                return 'source'
            elif np.all(real_parts < 0):
                return 'sink'
            else:
                return 'saddle'
        else:  # Eigenvalores complejos
            if np.all(real_parts > 0):
                return 'unstable_spiral'
            elif np.all(real_parts < 0):
                return 'stable_spiral'
            else:
                return 'center'
    
    def _analyze_flow_characteristics(self, vx: np.ndarray, vy: np.ndarray) -> Dict[str, Any]:
        """Analiza características generales del flujo"""
        # Velocidad promedio
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Coherencia direccional
        angles = np.arctan2(vy, vx)
        
        # Calcular coherencia usando desviación circular
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        angular_deviation = np.abs(angles - mean_angle)
        angular_deviation = np.minimum(angular_deviation, 2*np.pi - angular_deviation)
        coherence = 1.0 - np.mean(angular_deviation) / np.pi
        
        # Detectar regiones de flujo laminar vs turbulento
        velocity_gradient = np.gradient(velocity_magnitude)
        turbulence_indicator = np.std(velocity_gradient)
        
        return {
            'avg_speed': float(np.mean(velocity_magnitude)),
            'max_speed': float(np.max(velocity_magnitude)),
            'directional_coherence': float(coherence),
            'turbulence_level': float(turbulence_indicator),
            'flow_type': 'laminar' if turbulence_indicator < 0.5 else 'turbulent'
        }
    
    def detect_topology_transforms(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta transformaciones topológicas específicas
        """
        transforms = []
        
        # Detectar deformaciones continuas
        deformations = self._detect_continuous_deformations(matrix)
        transforms.extend(deformations)
        
        # Detectar cambios de conectividad
        connectivity_changes = self._detect_connectivity_changes(matrix)
        transforms.extend(connectivity_changes)
        
        # Detectar inversiones topológicas
        inversions = self._detect_topological_inversions(matrix)
        transforms.extend(inversions)
        
        return transforms
    
    def _detect_continuous_deformations(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta deformaciones continuas potenciales"""
        deformations = []
        
        # Analizar distorsiones locales usando análisis de strain
        strain_tensor = self._compute_strain_tensor(matrix)
        
        # Identificar regiones de alta deformación
        strain_magnitude = np.sqrt(np.sum(strain_tensor**2, axis=(2, 3)))
        high_strain_regions = strain_magnitude > np.mean(strain_magnitude) + np.std(strain_magnitude)
        
        if np.any(high_strain_regions):
            labeled, num_regions = ndimage.label(high_strain_regions)
            
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled == region_id)
                region_analysis = self._analyze_deformation_region(strain_tensor, region_mask)
                
                deformations.append({
                    'type': 'continuous_deformation',
                    'region_size': int(np.sum(region_mask)),
                    'deformation_type': region_analysis['type'],
                    'magnitude': region_analysis['magnitude']
                })
        
        return deformations
    
    def _compute_strain_tensor(self, matrix: np.ndarray) -> np.ndarray:
        """Computa tensor de strain local"""
        # Calcular gradiente de desplazamiento
        grad_y, grad_x = np.gradient(matrix.astype(float))
        
        # Tensor de strain (versión simplificada para 2D)
        h, w = matrix.shape
        strain_tensor = np.zeros((h, w, 2, 2))
        
        # Componentes del tensor de deformación
        dux_dx, dux_dy = np.gradient(grad_x)
        duy_dx, duy_dy = np.gradient(grad_y)
        
        # Tensor de strain simétrico
        strain_tensor[:, :, 0, 0] = dux_dx
        strain_tensor[:, :, 1, 1] = duy_dy
        strain_tensor[:, :, 0, 1] = 0.5 * (dux_dy + duy_dx)
        strain_tensor[:, :, 1, 0] = strain_tensor[:, :, 0, 1]
        
        return strain_tensor
    
    def _analyze_deformation_region(self, strain_tensor: np.ndarray, 
                                   region_mask: np.ndarray) -> Dict[str, Any]:
        """Analiza región de deformación"""
        # Extraer tensor de strain en la región
        region_strain = strain_tensor[region_mask]
        
        # Calcular eigenvalores promedio para clasificar deformación
        eigenvals = []
        for i in range(len(region_strain)):
            evals = np.linalg.eigvals(region_strain[i])
            eigenvals.append(evals)
        
        eigenvals = np.array(eigenvals)
        mean_eigenvals = np.mean(eigenvals, axis=0)
        
        # Clasificar tipo de deformación
        if np.all(mean_eigenvals > 0):
            deform_type = 'expansion'
        elif np.all(mean_eigenvals < 0):
            deform_type = 'compression'
        else:
            deform_type = 'shear'
        
        magnitude = float(np.linalg.norm(mean_eigenvals))
        
        return {
            'type': deform_type,
            'magnitude': magnitude
        }
    
    def analyze_flow_patterns(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analiza patrones de flujo en la estructura
        """
        # Crear campo vectorial basado en gradiente
        grad_y, grad_x = np.gradient(matrix.astype(float))
        
        # Detectar streamlines
        streamlines = self._trace_streamlines(grad_x, grad_y)
        
        # Analizar patrones de flujo
        flow_patterns = self._classify_flow_patterns(streamlines, grad_x, grad_y)
        
        # Detectar recirculaciones
        recirculations = self._detect_recirculation_zones(grad_x, grad_y)
        
        return {
            'streamlines': streamlines,
            'flow_patterns': flow_patterns,
            'recirculations': recirculations,
            'flow_complexity': self._calculate_flow_complexity(streamlines)
        }
    
    def _trace_streamlines(self, vx: np.ndarray, vy: np.ndarray, 
                          num_seeds: int = 10) -> List[List[Tuple[float, float]]]:
        """Traza streamlines del campo vectorial"""
        streamlines = []
        h, w = vx.shape
        
        # Crear puntos semilla
        seed_points = []
        for i in range(num_seeds):
            r = np.random.randint(1, h-1)
            c = np.random.randint(1, w-1)
            seed_points.append((r, c))
        
        # Trazar streamline desde cada semilla
        for seed in seed_points:
            streamline = self._integrate_streamline(vx, vy, seed)
            if len(streamline) > 5:  # Mínimo para ser significativo
                streamlines.append(streamline)
        
        return streamlines
    
    def _integrate_streamline(self, vx: np.ndarray, vy: np.ndarray, 
                             start: Tuple[int, int], max_steps: int = 100) -> List[Tuple[float, float]]:
        """Integra una streamline usando método de Euler"""
        streamline = []
        h, w = vx.shape
        
        # Posición actual
        r, c = float(start[0]), float(start[1])
        dt = 0.1  # Paso de integración
        
        for _ in range(max_steps):
            # Verificar límites
            if r < 1 or r >= h-1 or c < 1 or c >= w-1:
                break
            
            streamline.append((r, c))
            
            # Interpolar velocidad en posición actual
            r_int, c_int = int(r), int(c)
            
            # Interpolación bilineal simple
            v_r = vx[r_int, c_int]
            v_c = vy[r_int, c_int]
            
            # Verificar velocidad mínima
            vel_magnitude = np.sqrt(v_r**2 + v_c**2)
            if vel_magnitude < 1e-10:
                break
            
            # Integrar posición
            r += v_r * dt
            c += v_c * dt
        
        return streamline
    
    def find_critical_points(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Encuentra puntos críticos topológicos
        """
        critical_points = []
        
        # Puntos críticos del campo escalar (extremos locales)
        scalar_critical = self._find_scalar_critical_points(matrix)
        critical_points.extend(scalar_critical)
        
        # Puntos críticos del campo vectorial derivado
        grad_y, grad_x = np.gradient(matrix.astype(float))
        vector_critical = self._find_vector_field_critical_points(grad_x, grad_y)
        critical_points.extend(vector_critical)
        
        return critical_points
    
    def _find_scalar_critical_points(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Encuentra extremos locales del campo escalar"""
        from scipy.ndimage import maximum_filter, minimum_filter
        
        critical_points = []
        
        # Filtros para encontrar máximos y mínimos locales
        local_max = (matrix == maximum_filter(matrix, size=3))
        local_min = (matrix == minimum_filter(matrix, size=3))
        
        # Excluir bordes y puntos de fondo
        mask = np.ones_like(matrix, dtype=bool)
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        mask = mask & (matrix != 0)
        
        # Máximos locales
        max_points = np.where(local_max & mask)
        for i in range(len(max_points[0])):
            r, c = max_points[0][i], max_points[1][i]
            critical_points.append({
                'position': (int(r), int(c)),
                'type': 'local_maximum',
                'value': float(matrix[r, c]),
                'category': 'scalar'
            })
        
        # Mínimos locales
        min_points = np.where(local_min & mask)
        for i in range(len(min_points[0])):
            r, c = min_points[0][i], min_points[1][i]
            critical_points.append({
                'position': (int(r), int(c)),
                'type': 'local_minimum',
                'value': float(matrix[r, c]),
                'category': 'scalar'
            })
        
        return critical_points
    
    def compute_basic_homology(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Computa homología básica (números de Betti simplificados)
        """
        # Betti 0: Número de componentes conexas
        binary_matrix = (matrix > 0).astype(int)
        labeled, num_components = ndimage.label(binary_matrix)
        betti_0 = num_components
        
        # Betti 1: Número de agujeros (aproximado)
        betti_1 = 0
        for component_id in range(1, num_components + 1):
            component_mask = (labeled == component_id)
            
            # Usar binary_fill_holes para detectar agujeros
            filled = ndimage.binary_fill_holes(component_mask)
            holes = filled & (~component_mask)
            
            if np.any(holes):
                hole_labeled, num_holes = ndimage.label(holes)
                betti_1 += num_holes
        
        # Característica de Euler simplificada
        euler_characteristic = betti_0 - betti_1
        
        return {
            'betti_0': int(betti_0),
            'betti_1': int(betti_1),
            'euler_characteristic': int(euler_characteristic),
            'genus': max(0, (2 - euler_characteristic) // 2)  # Aproximación para superficies
        }
    
    def analyze_contour_topology(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analiza topología de contornos
        """
        from skimage import measure
        
        contour_analysis = {
            'contours_by_level': {},
            'nested_structure': [],
            'contour_tree': {}
        }
        
        # Analizar contornos para diferentes niveles
        unique_levels = np.unique(matrix)
        
        for level in unique_levels:
            if level == 0:
                continue
            
            try:
                # Encontrar contornos para este nivel
                binary_mask = (matrix >= level)
                contours = measure.find_contours(binary_mask.astype(float), 0.5)
                
                level_analysis = {
                    'num_contours': len(contours),
                    'total_length': sum(self._calculate_curve_length(c) for c in contours),
                    'avg_length': np.mean([self._calculate_curve_length(c) for c in contours]) if contours else 0
                }
                
                contour_analysis['contours_by_level'][int(level)] = level_analysis
                
            except Exception:
                contour_analysis['contours_by_level'][int(level)] = {
                    'num_contours': 0,
                    'total_length': 0,
                    'avg_length': 0
                }
        
        return contour_analysis
    
    # Métodos auxiliares adicionales
    
    def _calculate_spline_fitting_error(self, points: np.ndarray, tck: tuple, u: np.ndarray) -> float:
        """Calcula error de fitting del spline"""
        try:
            # Evaluar spline en parámetros originales
            spline_points = np.array(splev(u, tck)).T
            
            # Calcular error RMS
            errors = np.sqrt(np.sum((points - spline_points)**2, axis=1))
            return float(np.mean(errors))
        except:
            return float('inf')
    
    def _calculate_spline_length(self, tck: tuple, num_points: int = 100) -> float:
        """Calcula longitud de spline"""
        try:
            u_fine = np.linspace(0, 1, num_points)
            points = np.array(splev(u_fine, tck)).T
            
            diffs = np.diff(points, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            return float(np.sum(segment_lengths))
        except:
            return 0.0
    
    def _analyze_curve_continuity(self, matrix: np.ndarray) -> Dict[str, float]:
        """Analiza continuidad de curvas"""
        # Implementación simplificada
        return {
            'c0_continuity': 0.8,  # Placeholder
            'c1_continuity': 0.6,  # Placeholder
            'smoothness_score': 0.7  # Placeholder
        }
    
    def _find_curve_intersections(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Encuentra intersecciones entre curvas"""
        # Implementación simplificada
        return []  # Placeholder
    
    def _analyze_curvature(self, matrix: np.ndarray) -> Dict[str, float]:
        """Análisis general de curvatura"""
        # Implementación simplificada
        return {
            'mean_curvature': 0.5,  # Placeholder
            'gaussian_curvature': 0.0,  # Placeholder
            'curvature_variation': 0.3  # Placeholder
        }
    
    def _detect_connectivity_changes(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta cambios de conectividad"""
        # Placeholder - implementación completa requeriría comparación temporal
        return []
    
    def _detect_topological_inversions(self, matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta inversiones topológicas"""
        # Placeholder - implementación compleja
        return []
    
    def _classify_flow_patterns(self, streamlines: List, vx: np.ndarray, vy: np.ndarray) -> List[str]:
        """Clasifica patrones de flujo"""
        patterns = []
        
        # Analizar convergencia/divergencia de streamlines
        if len(streamlines) > 3:
            # Simplificado: verificar si streamlines convergen o divergen
            convergence_score = self._calculate_streamline_convergence(streamlines)
            
            if convergence_score > 0.7:
                patterns.append('convergent_flow')
            elif convergence_score < -0.7:
                patterns.append('divergent_flow')
            else:
                patterns.append('parallel_flow')
        
        return patterns
    
    def _calculate_streamline_convergence(self, streamlines: List) -> float:
        """Calcula convergencia de streamlines"""
        if len(streamlines) < 2:
            return 0.0
        
        # Calcular distancias entre puntos finales vs iniciales
        start_distances = []
        end_distances = []
        
        for i in range(len(streamlines)):
            for j in range(i+1, len(streamlines)):
                if len(streamlines[i]) > 0 and len(streamlines[j]) > 0:
                    # Distancia entre puntos iniciales
                    start_dist = np.linalg.norm(np.array(streamlines[i][0]) - np.array(streamlines[j][0]))
                    start_distances.append(start_dist)
                    
                    # Distancia entre puntos finales
                    end_dist = np.linalg.norm(np.array(streamlines[i][-1]) - np.array(streamlines[j][-1]))
                    end_distances.append(end_dist)
        
        if not start_distances or not end_distances:
            return 0.0
        
        # Convergencia: distancias finales menores que iniciales
        avg_start = np.mean(start_distances)
        avg_end = np.mean(end_distances)
        
        if avg_start > 0:
            convergence = (avg_start - avg_end) / avg_start
        else:
            convergence = 0.0
        
        return float(convergence)
    
    def _detect_recirculation_zones(self, vx: np.ndarray, vy: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta zonas de recirculación"""
        # Calcular vorticidad
        dvy_dx = np.gradient(vy, axis=1)
        dvx_dy = np.gradient(vx, axis=0)
        vorticity = dvy_dx - dvx_dy
        
        # Identificar regiones de alta vorticidad
        vorticity_threshold = np.std(vorticity)
        high_vorticity = np.abs(vorticity) > vorticity_threshold
        
        # Etiquetar regiones
        labeled, num_regions = ndimage.label(high_vorticity)
        
        recirculations = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled == region_id)
            if np.sum(region_mask) > 5:  # Mínimo tamaño
                center = ndimage.center_of_mass(region_mask)
                avg_vorticity = np.mean(vorticity[region_mask])
                
                recirculations.append({
                    'center': [float(center[0]), float(center[1])],
                    'size': int(np.sum(region_mask)),
                    'strength': float(avg_vorticity),
                    'type': 'clockwise' if avg_vorticity > 0 else 'counterclockwise'
                })
        
        return recirculations
    
    def _calculate_flow_complexity(self, streamlines: List) -> float:
        """Calcula complejidad del flujo"""
        if not streamlines:
            return 0.0
        
        # Complejidad basada en:
        # 1. Número de streamlines
        # 2. Longitud promedio
        # 3. Variabilidad en direcciones
        
        num_streamlines = len(streamlines)
        lengths = [len(s) for s in streamlines if len(s) > 1]
        
        if not lengths:
            return 0.0
        
        avg_length = np.mean(lengths)
        length_std = np.std(lengths)
        
        # Normalizar complejidad
        complexity = min(1.0, (num_streamlines * avg_length * (1 + length_std)) / 1000)
        
        return float(complexity)