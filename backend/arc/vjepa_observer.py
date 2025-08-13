#!/usr/bin/env python3
"""
V-JEPA2 Observer for ARC
Aprende patrones emergentemente por observación, sin hardcodear
Basado en el concepto de Joint Embedding Predictive Architecture
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObservedTransition:
    """Transición observada entre estados"""
    input_state: np.ndarray
    output_state: np.ndarray
    latent_features: Optional[np.ndarray] = None
    transformation_embedding: Optional[np.ndarray] = None

class VJEPAObserver:
    """
    Observador basado en V-JEPA2 que aprende transformaciones
    sin hardcodear patrones específicos
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.observed_transitions = []
        self.learned_embeddings = {}
        
        # Embeddings aprendidos dinámicamente
        self.shape_embeddings = {}  # Mapeo de formas a embeddings
        self.transform_embeddings = {}  # Mapeo de transformaciones a embeddings
        
    def observe(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Observa una transición y extrae características emergentes
        SIN hardcodear tipos de transformación
        """
        # Extraer características latentes (no píxeles directamente)
        input_features = self._extract_latent_features(input_grid)
        output_features = self._extract_latent_features(output_grid)
        
        # Calcular embedding de transformación como diferencia en espacio latente
        transform_embedding = self._compute_transform_embedding(
            input_features, output_features, input_grid.shape, output_grid.shape
        )
        
        # Almacenar observación
        transition = ObservedTransition(
            input_state=input_grid,
            output_state=output_grid,
            latent_features=output_features - input_features,
            transformation_embedding=transform_embedding
        )
        self.observed_transitions.append(transition)
        
        # Detectar patrones emergentes sin categorías predefinidas
        emergent_pattern = self._detect_emergent_pattern(transition)
        
        return {
            'observed': True,
            'latent_difference': transition.latent_features,
            'transform_embedding': transform_embedding,
            'emergent_pattern': emergent_pattern
        }
    
    def _extract_latent_features(self, grid: np.ndarray) -> np.ndarray:
        """
        Extrae características latentes de una grilla
        Similar a como V-JEPA procesa patches de video
        """
        h, w = grid.shape
        features = []
        
        # 1. Características globales
        features.extend([
            np.mean(grid),  # Densidad promedio
            np.std(grid),   # Variabilidad
            np.count_nonzero(grid) / grid.size,  # Ocupación
        ])
        
        # 2. Distribución espacial (sin asumir patrones específicos)
        # Dividir en patches como V-JEPA
        patch_size = min(3, min(h, w))
        patches = []
        
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = grid[i:i+patch_size, j:j+patch_size]
                # Embedding del patch (simplificado)
                patch_features = [
                    np.mean(patch),
                    np.max(patch),
                    np.count_nonzero(patch) / patch.size
                ]
                patches.append(patch_features)
        
        if patches:
            # Agregar estadísticas de patches
            patches_array = np.array(patches)
            features.extend(np.mean(patches_array, axis=0))
            features.extend(np.std(patches_array, axis=0))
        
        # 3. Relaciones espaciales emergentes
        if np.any(grid):
            # Centro de masa
            positions = np.argwhere(grid > 0)
            if len(positions) > 0:
                center = np.mean(positions, axis=0)
                features.extend(center / np.array([h, w]))  # Normalizado
        
        # Pad o truncar a dimensión fija
        feature_vector = np.zeros(self.embedding_dim)
        feature_vector[:min(len(features), self.embedding_dim)] = features[:self.embedding_dim]
        
        return feature_vector
    
    def _compute_transform_embedding(self, input_features: np.ndarray, 
                                   output_features: np.ndarray,
                                   input_shape: Tuple, output_shape: Tuple) -> np.ndarray:
        """
        Calcula embedding de transformación en espacio latente
        """
        embedding = np.zeros(self.embedding_dim)
        
        # Diferencia de características
        feature_diff = output_features - input_features
        
        # Información de cambio de forma (si existe)
        shape_change = np.array([
            output_shape[0] / input_shape[0],  # Factor de cambio en altura
            output_shape[1] / input_shape[1],  # Factor de cambio en anchura
        ])
        
        # Combinar en embedding
        embedding[:len(feature_diff)] = feature_diff
        embedding[-2:] = shape_change
        
        return embedding
    
    def _detect_emergent_pattern(self, transition: ObservedTransition) -> Dict:
        """
        Detecta patrones emergentes comparando con observaciones previas
        SIN categorías hardcodeadas
        """
        if len(self.observed_transitions) < 2:
            return {'type': 'novel', 'confidence': 0.0}
        
        # Comparar con transiciones previas en espacio de embeddings
        similarities = []
        for prev_transition in self.observed_transitions[:-1]:
            if prev_transition.transformation_embedding is not None:
                # Similitud coseno entre embeddings de transformación
                sim = self._cosine_similarity(
                    transition.transformation_embedding,
                    prev_transition.transformation_embedding
                )
                similarities.append(sim)
        
        if similarities:
            max_similarity = max(similarities)
            
            if max_similarity > 0.9:
                # Patrón muy similar a uno observado antes
                return {
                    'type': 'repeated',
                    'confidence': max_similarity,
                    'similar_to_observation': similarities.index(max_similarity)
                }
            elif max_similarity > 0.7:
                # Patrón relacionado pero no idéntico
                return {
                    'type': 'variant',
                    'confidence': max_similarity,
                    'similar_to_observation': similarities.index(max_similarity)
                }
        
        # Patrón nuevo no visto antes
        return {
            'type': 'novel',
            'confidence': 1.0 - (max(similarities) if similarities else 0)
        }
    
    def predict_transformation(self, test_input: np.ndarray) -> np.ndarray:
        """
        Predice la transformación basándose en observaciones previas
        Sin usar reglas hardcodeadas
        """
        if not self.observed_transitions:
            return test_input
        
        # Extraer características del test
        test_features = self._extract_latent_features(test_input)
        
        # Encontrar la transición observada más similar
        best_match = None
        best_similarity = -1
        
        for transition in self.observed_transitions:
            input_features = self._extract_latent_features(transition.input_state)
            similarity = self._cosine_similarity(test_features, input_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = transition
        
        if best_match and best_similarity > 0.5:
            # Aplicar transformación similar
            return self._apply_learned_transformation(
                test_input, best_match
            )
        
        return test_input
    
    def _apply_learned_transformation(self, grid: np.ndarray, 
                                     reference_transition: ObservedTransition) -> np.ndarray:
        """
        Aplica una transformación aprendida por observación
        """
        ref_input = reference_transition.input_state
        ref_output = reference_transition.output_state
        
        # Si las dimensiones cambiaron, aplicar cambio proporcional
        if ref_input.shape != ref_output.shape:
            h_factor = ref_output.shape[0] / ref_input.shape[0]
            w_factor = ref_output.shape[1] / ref_input.shape[1]
            
            new_h = int(grid.shape[0] * h_factor)
            new_w = int(grid.shape[1] * w_factor)
            
            # Aplicar cambio de tamaño observado
            if h_factor == int(h_factor) and w_factor == int(w_factor):
                # Repetición exacta
                result = np.zeros((new_h, new_w), dtype=grid.dtype)
                
                # Analizar cómo se llenó el espacio en la referencia
                for i in range(int(h_factor)):
                    for j in range(int(w_factor)):
                        h_start = i * ref_input.shape[0]
                        h_end = (i + 1) * ref_input.shape[0]
                        w_start = j * ref_input.shape[1]
                        w_end = (j + 1) * ref_input.shape[1]
                        
                        ref_segment = ref_output[h_start:h_end, w_start:w_end]
                        
                        # Aplicar transformación observada en el segmento
                        grid_h_start = i * grid.shape[0]
                        grid_h_end = min((i + 1) * grid.shape[0], new_h)
                        grid_w_start = j * grid.shape[1]
                        grid_w_end = min((j + 1) * grid.shape[1], new_w)
                        
                        # Copiar patrón observado
                        if np.array_equal(ref_segment, ref_input):
                            result[grid_h_start:grid_h_end, grid_w_start:grid_w_end] = grid
                        else:
                            # Aplicar transformación observada en el segmento
                            transformed = self._infer_pixel_transformation(
                                grid, ref_input, ref_segment
                            )
                            result[grid_h_start:grid_h_end, grid_w_start:grid_w_end] = transformed
                
                return result
        
        # Si no hay cambio de tamaño, aplicar transformación pixel a pixel
        return self._infer_pixel_transformation(grid, ref_input, ref_output)
    
    def _infer_pixel_transformation(self, grid: np.ndarray, 
                                   ref_input: np.ndarray, 
                                   ref_output: np.ndarray) -> np.ndarray:
        """
        Infiere transformación a nivel de píxel desde ejemplos
        """
        # Si las formas no coinciden, intentar mapeo de valores solamente
        result = grid.copy()
        
        # Analizar mapeo de valores observado
        value_mapping = {}
        
        # Usar solo la región común para el análisis
        min_h = min(ref_input.shape[0], ref_output.shape[0])
        min_w = min(ref_input.shape[1], ref_output.shape[1])
        
        for y in range(min_h):
            for x in range(min_w):
                if y < ref_input.shape[0] and x < ref_input.shape[1]:
                    in_val = ref_input[y, x]
                    if y < ref_output.shape[0] and x < ref_output.shape[1]:
                        out_val = ref_output[y, x]
                        
                        if in_val != 0:  # Ignorar fondo
                            if in_val not in value_mapping:
                                value_mapping[in_val] = out_val
                            elif value_mapping[in_val] != out_val:
                                # Mapeo inconsistente, usar el más frecuente
                                pass
        
        # Aplicar mapeo aprendido
        for old_val, new_val in value_mapping.items():
            result[grid == old_val] = new_val
        
        return result
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))