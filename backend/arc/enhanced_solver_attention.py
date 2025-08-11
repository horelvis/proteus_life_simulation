#!/usr/bin/env python3
"""
Solver Mejorado con Sistema de Atención Bidireccional
Integra el análisis jerárquico con barridos topográficos
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from arc.hybrid_proteus_solver import HybridProteusARCSolver
from arc.bidirectional_attention import BidirectionalAttentionSystem, EnrichedPixel
from arc.transformations_fixed import RealTransformations
from arc.hierarchical_analyzer import HierarchicalAnalyzer
from arc.emergent_rule_system import EmergentRuleSystem

logger = logging.getLogger(__name__)

class EnhancedSolverWithAttention(HybridProteusARCSolver):
    """
    Solver que utiliza el sistema de atención bidireccional
    para mejorar la detección y aplicación de transformaciones
    """
    
    def __init__(self):
        super().__init__()
        self.attention_system = BidirectionalAttentionSystem()
        self.transformations = RealTransformations()
        self.hierarchical = HierarchicalAnalyzer()
        self.emergent = EmergentRuleSystem()
        
        # Cache de análisis
        self.last_attention_map = None
        self.last_hierarchy = None
        self.last_insights = None
        
    def solve_with_attention(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Resuelve usando el sistema de atención bidireccional
        Retorna la solución y un diccionario con el análisis completo
        """
        
        logger.info("Iniciando resolución con sistema de atención bidireccional")
        
        # PASO 1: Análisis jerárquico estándar
        hierarchy_analysis = self.hierarchical.analyze_full_hierarchy(test_input)
        self.last_hierarchy = hierarchy_analysis
        
        # PASO 2: Construir jerarquía bidireccional desde ejemplos de entrenamiento
        if train_examples:
            # Usar el primer ejemplo para aprender la estructura
            first_example = train_examples[0]
            input_grid = np.array(first_example['input'])
            output_grid = np.array(first_example['output'])
            
            # Construir jerarquía con enlaces bidireccionales
            bidirectional_hierarchy = self.attention_system.build_bidirectional_hierarchy(
                input_grid, output_grid
            )
            
            # PASO 3: Realizar barrido topográfico con atención
            attention_map = self.attention_system.perform_bidirectional_scan(input_grid)
            self.last_attention_map = attention_map
            
            # PASO 4: Obtener insights del análisis de atención
            insights = self.attention_system.get_attention_insights()
            self.last_insights = insights
            
            # PASO 5: Identificar transformación basada en puntos de alta atención
            transformation = self._identify_transformation_from_attention(
                insights['high_attention_points'],
                input_grid,
                output_grid
            )
            
            # PASO 6: Aplicar transformación informada por atención al test
            solution = self._apply_attention_informed_transformation(
                test_input,
                transformation,
                insights
            )
            
            # Preparar análisis completo para retorno
            analysis = {
                'hierarchy': hierarchy_analysis,
                'bidirectional_structure': bidirectional_hierarchy,
                'attention_map': attention_map.tolist() if attention_map is not None else None,
                'insights': insights,
                'transformation_detected': transformation,
                'confidence': self._calculate_solution_confidence(insights)
            }
            
            return solution, analysis
        
        # Si no hay ejemplos, usar método padre
        return super().solve(train_examples, test_input), {'method': 'fallback'}
    
    def _identify_transformation_from_attention(self, high_attention_points: List[Dict], 
                                               input_grid: np.ndarray, 
                                               output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Identifica la transformación analizando los puntos de alta atención
        """
        
        if not high_attention_points:
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Analizar cambios en puntos de alta atención
        changes = []
        for point_info in high_attention_points:
            x, y = point_info['position']
            context = point_info['context']
            
            in_val = input_grid[y, x]
            out_val = output_grid[y, x]
            
            if in_val != out_val:
                changes.append({
                    'position': (x, y),
                    'from': in_val,
                    'to': out_val,
                    'context': context,
                    'attention_score': point_info['attention']
                })
        
        # Analizar patrones en los cambios
        transformation = self._analyze_change_pattern(changes, input_grid, output_grid)
        
        # Enriquecer con información de la jerarquía
        if self.attention_system.patterns:
            # Tomar el patrón con mayor confianza
            best_pattern = max(
                self.attention_system.patterns.values(),
                key=lambda p: p.confidence,
                default=None
            )
            
            if best_pattern:
                transformation['pattern_type'] = best_pattern.type
                transformation['pattern_confidence'] = best_pattern.confidence
        
        return transformation
    
    def _analyze_change_pattern(self, changes: List[Dict], 
                               input_grid: np.ndarray, 
                               output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Analiza el patrón de cambios para identificar la transformación
        """
        
        if not changes:
            return {'type': 'identity', 'confidence': 1.0}
        
        # Agrupar cambios por tipo
        expansions = []
        fills = []
        color_changes = []
        
        for change in changes:
            context = change['context']
            
            # Verificar si es expansión
            if change['from'] == 0 and change['to'] != 0:
                # Ver si hay un valor adyacente igual
                x, y = change['position']
                neighbors = self._get_neighbors(input_grid, x, y)
                if change['to'] in neighbors:
                    expansions.append(change)
                else:
                    fills.append(change)
            
            # Cambio de color
            elif change['from'] != 0 and change['to'] != 0:
                color_changes.append(change)
        
        # Determinar transformación dominante
        if len(expansions) > len(fills) and len(expansions) > len(color_changes):
            # Patrón de expansión detectado
            
            # Verificar si es expansión en cruz
            is_cross = self._is_cross_expansion(expansions, input_grid)
            
            return {
                'type': 'cross_expansion' if is_cross else 'expansion',
                'confidence': len(expansions) / max(len(changes), 1),
                'details': {'points': len(expansions)}
            }
        
        elif len(fills) > len(expansions) and len(fills) > len(color_changes):
            return {
                'type': 'fill',
                'confidence': len(fills) / max(len(changes), 1),
                'details': {'points': len(fills)}
            }
        
        elif len(color_changes) > 0:
            # Analizar el mapeo de colores
            mapping = {}
            for change in color_changes:
                if change['from'] not in mapping:
                    mapping[change['from']] = change['to']
            
            return {
                'type': 'color_mapping',
                'confidence': len(color_changes) / max(len(changes), 1),
                'mapping': mapping
            }
        
        return {'type': 'complex', 'confidence': 0.5}
    
    def _is_cross_expansion(self, expansions: List[Dict], input_grid: np.ndarray) -> bool:
        """
        Verifica si las expansiones forman un patrón de cruz
        """
        
        # Buscar centros de expansión
        centers = {}
        
        for exp in expansions:
            x, y = exp['position']
            
            # Verificar en las 4 direcciones cardinales
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                cx, cy = x - dx, y - dy
                
                if (0 <= cy < input_grid.shape[0] and 
                    0 <= cx < input_grid.shape[1] and
                    input_grid[cy, cx] == exp['to']):
                    
                    center_key = (cx, cy)
                    if center_key not in centers:
                        centers[center_key] = []
                    centers[center_key].append((x, y))
        
        # Verificar si algún centro tiene expansiones en las 4 direcciones
        for center, expanded_positions in centers.items():
            cx, cy = center
            
            has_up = any(x == cx and y == cy - 1 for x, y in expanded_positions)
            has_down = any(x == cx and y == cy + 1 for x, y in expanded_positions)
            has_left = any(x == cx - 1 and y == cy for x, y in expanded_positions)
            has_right = any(x == cx + 1 and y == cy for x, y in expanded_positions)
            
            if sum([has_up, has_down, has_left, has_right]) >= 3:
                return True
        
        return False
    
    def _apply_attention_informed_transformation(self, test_input: np.ndarray,
                                                transformation: Dict[str, Any],
                                                insights: Dict[str, Any]) -> np.ndarray:
        """
        Aplica la transformación guiada por el análisis de atención
        """
        
        result = test_input.copy()
        
        # Aplicar según el tipo de transformación detectada
        trans_type = transformation.get('type', 'unknown')
        
        if trans_type == 'cross_expansion':
            result = self.transformations.expand_cross(result)
            
        elif trans_type == 'expansion':
            # Expansión general
            result = self._apply_general_expansion(result)
            
        elif trans_type == 'fill':
            result = self.transformations.fill_enclosed_spaces(result)
            
        elif trans_type == 'color_mapping':
            mapping = transformation.get('mapping', {})
            if mapping:
                result = self.transformations.apply_color_mapping(result, mapping)
        
        elif trans_type == 'complex':
            # Intentar aplicar transformaciones en puntos de alta atención
            result = self._apply_complex_transformation(
                result,
                insights.get('high_attention_points', [])
            )
        
        return result
    
    def _apply_general_expansion(self, grid: np.ndarray) -> np.ndarray:
        """
        Aplica expansión general (no necesariamente en cruz)
        """
        
        result = grid.copy()
        h, w = grid.shape
        
        # Expandir cada valor no-cero a celdas vacías adyacentes
        for y in range(h):
            for x in range(w):
                if grid[y, x] != 0:
                    value = grid[y, x]
                    
                    # Expandir a vecinos vacíos
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        
                        if (0 <= ny < h and 0 <= nx < w and 
                            grid[ny, nx] == 0):
                            result[ny, nx] = value
        
        return result
    
    def _apply_complex_transformation(self, grid: np.ndarray, 
                                     high_attention_points: List[Dict]) -> np.ndarray:
        """
        Aplica transformaciones complejas basadas en puntos de atención
        """
        
        result = grid.copy()
        
        # Por ahora, aplicar transformaciones locales en puntos de alta atención
        for point_info in high_attention_points:
            x, y = point_info['position']
            context = point_info['context']
            
            # Si el píxel tiene un valor esperado del patrón, usarlo
            if context['pattern']['expected_value'] is not None:
                result[y, x] = context['pattern']['expected_value']
            
            # Si el píxel es crítico para un patrón, preservarlo o enfatizarlo
            elif context['pattern']['role'] == 'critical':
                # Podría expandir o reforzar este punto
                if result[y, x] != 0:
                    # Opcional: expandir puntos críticos
                    pass
        
        return result
    
    def _calculate_solution_confidence(self, insights: Dict[str, Any]) -> float:
        """
        Calcula la confianza en la solución basada en los insights
        """
        
        confidence = 0.5  # Base
        
        # Aumentar confianza si hay alta coherencia
        if 'coherence_score' in insights:
            confidence = confidence * 0.5 + insights['coherence_score'] * 0.5
        
        # Aumentar si se detectaron patrones claros
        if insights.get('num_patterns', 0) > 0:
            confidence += 0.2
        
        # Aumentar si hay puntos de alta atención bien definidos
        if len(insights.get('high_attention_points', [])) > 3:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_neighbors(self, grid: np.ndarray, x: int, y: int) -> List[int]:
        """
        Obtiene los valores de los vecinos de una posición
        """
        
        neighbors = []
        h, w = grid.shape
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append(grid[ny, nx])
        
        return neighbors
    
    def get_visual_analysis(self) -> Dict[str, Any]:
        """
        Obtiene el análisis completo para visualización en la UI
        """
        
        return {
            'attention_map': self.last_attention_map.tolist() if self.last_attention_map is not None else None,
            'hierarchy': self.last_hierarchy,
            'insights': self.last_insights,
            'pixel_contexts': self._get_pixel_contexts_for_ui()
        }
    
    def _get_pixel_contexts_for_ui(self) -> List[Dict]:
        """
        Prepara los contextos de píxeles para visualización
        """
        
        contexts = []
        
        if self.attention_system.pixel_layer:
            # Tomar los 10 píxeles más importantes
            sorted_pixels = sorted(
                self.attention_system.pixel_layer.values(),
                key=lambda p: p.importance_score,
                reverse=True
            )[:10]
            
            for pixel in sorted_pixels:
                contexts.append({
                    'position': (pixel.x, pixel.y),
                    'value': pixel.value,
                    'importance': pixel.importance_score,
                    'object_role': pixel.object_role,
                    'relation_role': pixel.relation_role,
                    'pattern_role': pixel.pattern_role,
                    'full_context': pixel.get_full_context()
                })
        
        return contexts