#!/usr/bin/env python3
"""
Unified ARC Solver
Sistema unificado que usa V-JEPA Observer para aprender patrones emergentemente
Elimina todo hardcodeo de patrones específicos
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

from arc.vjepa_observer import VJEPAObserver
from arc.hierarchical_analyzer import HierarchicalAnalyzer
from arc.transformations_fixed import RealTransformations

logger = logging.getLogger(__name__)

class UnifiedARCSolver:
    """
    Solver unificado que aprende por observación
    Sin hardcodeo de patrones o transformaciones
    """
    
    def __init__(self):
        self.observer = VJEPAObserver(embedding_dim=128)
        self.analyzer = HierarchicalAnalyzer()
        self.transformations = RealTransformations()
        
        # Estado de aprendizaje
        self.learned_patterns = []
        self.confidence_threshold = 0.7
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """
        Resuelve un puzzle ARC aprendiendo de los ejemplos
        """
        solution, _ = self.solve_with_steps(train_examples, test_input)
        return solution
    
    def solve_with_steps(self, train_examples: List[Dict], 
                         test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve con registro de pasos, aprendiendo emergentemente
        """
        steps = []
        
        if not train_examples:
            return test_input, [{"description": "Sin ejemplos de entrenamiento"}]
        
        # PASO 1: Observar y aprender de los ejemplos
        steps.append({"description": "🔍 Observando ejemplos..."})
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Observar transición sin asumir tipo
            observation = self.observer.observe(input_grid, output_grid)
            
            pattern_type = observation['emergent_pattern']['type']
            confidence = observation['emergent_pattern'].get('confidence', 0)
            
            steps.append({
                "description": f"  Ejemplo {i+1}: Patrón {pattern_type} "
                              f"(confianza: {confidence:.2f})"
            })
        
        # PASO 2: Análisis jerárquico para contexto adicional
        if len(train_examples) > 0:
            first_input = np.array(train_examples[0]['input'])
            hierarchical_analysis = self.analyzer.analyze_full_hierarchy(first_input)
            
            num_objects = len(hierarchical_analysis['level_1_objects']['objects'])
            num_patterns = len(hierarchical_analysis['level_3_patterns']['patterns'])
            
            steps.append({
                "description": f"📊 Análisis: {num_objects} objetos, "
                              f"{num_patterns} patrones estructurales"
            })
        
        # PASO 3: Predecir transformación basada en observaciones
        steps.append({"description": "🎯 Aplicando transformación aprendida..."})
        
        solution = self.observer.predict_transformation(test_input)
        
        # Si la predicción no cambió nada, intentar con análisis más profundo
        if np.array_equal(solution, test_input):
            steps.append({"description": "🔄 Refinando predicción..."})
            solution = self._refine_prediction(train_examples, test_input)
        
        # Validar coherencia de la solución
        coherence = self._validate_solution_coherence(
            train_examples, test_input, solution
        )
        
        steps.append({
            "description": f"✅ Solución generada (coherencia: {coherence:.2f})"
        })
        
        return solution, steps
    
    def _refine_prediction(self, train_examples: List[Dict], 
                          test_input: np.ndarray) -> np.ndarray:
        """
        Refina la predicción cuando el método directo no produce cambios
        """
        if not train_examples:
            return test_input
        
        # Analizar consistencia entre ejemplos
        consistent_changes = self._find_consistent_changes(train_examples)
        
        if consistent_changes:
            # Aplicar cambios consistentes encontrados
            result = test_input.copy()
            
            for change in consistent_changes:
                if change['type'] == 'value_mapping':
                    # Mapeo de valores consistente
                    for old_val, new_val in change['mapping'].items():
                        result[result == old_val] = new_val
                        
                elif change['type'] == 'spatial_transform':
                    # Transformación espacial consistente
                    result = self._apply_spatial_transform(
                        result, change['parameters']
                    )
            
            return result
        
        return test_input
    
    def _find_consistent_changes(self, train_examples: List[Dict]) -> List[Dict]:
        """
        Encuentra cambios consistentes en todos los ejemplos
        Sin asumir tipos específicos de cambios
        """
        if not train_examples:
            return []
        
        changes = []
        
        # Analizar cada ejemplo
        all_mappings = []
        shape_changes = []
        
        for example in train_examples:
            input_g = np.array(example['input'])
            output_g = np.array(example['output'])
            
            # Detectar mapeo de valores
            mapping = {}
            if input_g.shape == output_g.shape:
                for val in np.unique(input_g):
                    if val != 0:  # Ignorar fondo
                        mask = (input_g == val)
                        output_vals = output_g[mask]
                        if len(np.unique(output_vals)) == 1:
                            mapping[val] = output_vals[0]
                
                if mapping:
                    all_mappings.append(mapping)
            
            # Detectar cambios de forma
            if input_g.shape != output_g.shape:
                shape_changes.append({
                    'from': input_g.shape,
                    'to': output_g.shape,
                    'factor': (
                        output_g.shape[0] / input_g.shape[0],
                        output_g.shape[1] / input_g.shape[1]
                    )
                })
        
        # Ver si hay mapeo consistente
        if all_mappings and all(m == all_mappings[0] for m in all_mappings):
            changes.append({
                'type': 'value_mapping',
                'mapping': all_mappings[0]
            })
        
        # Ver si hay cambio de forma consistente
        if shape_changes and all(
            s['factor'] == shape_changes[0]['factor'] for s in shape_changes
        ):
            changes.append({
                'type': 'spatial_transform',
                'parameters': {
                    'scale': shape_changes[0]['factor']
                }
            })
        
        return changes
    
    def _apply_spatial_transform(self, grid: np.ndarray, 
                                parameters: Dict) -> np.ndarray:
        """
        Aplica transformación espacial basada en parámetros observados
        """
        if 'scale' in parameters:
            h_scale, w_scale = parameters['scale']
            
            if h_scale == int(h_scale) and w_scale == int(w_scale):
                # Escala entera - repetir
                return np.repeat(
                    np.repeat(grid, int(h_scale), axis=0),
                    int(w_scale), axis=1
                )
            else:
                # Escala fraccionaria - interpolar
                from scipy.ndimage import zoom
                return zoom(grid, (h_scale, w_scale), order=0)
        
        return grid
    
    def _validate_solution_coherence(self, train_examples: List[Dict],
                                    test_input: np.ndarray,
                                    solution: np.ndarray) -> float:
        """
        Valida la coherencia de la solución con los ejemplos
        """
        if not train_examples:
            return 0.5
        
        coherence_scores = []
        
        # Comparar características de la solución con los outputs de ejemplo
        solution_features = self.observer._extract_latent_features(solution)
        
        for example in train_examples:
            output_g = np.array(example['output'])
            output_features = self.observer._extract_latent_features(output_g)
            
            # Similitud en espacio de características
            similarity = self.observer._cosine_similarity(
                solution_features, output_features
            )
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0