#!/usr/bin/env python3
"""
V-JEPA Observer Contrastive
Usa el modelo V-JEPA contrastivo pre-entrenado para observación visual real
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Importar el encoder contrastivo entrenado
import sys
sys.path.append('/app/arc')
from train_vjepa_contrastive import SimpleEncoder


class VJEPAObserverContrastive:
    """
    Observer que usa V-JEPA Contrastive pre-entrenado
    Aprende representaciones visuales REALES sin hardcodeo
    """
    
    def __init__(self, 
                 weights_path: str = "/app/arc/vjepa_contrastive_weights/vjepa_contrastive_final.pth",
                 embedding_dim: int = 128):
        
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar encoder pre-entrenado
        self.encoder = self._load_pretrained_encoder(weights_path)
        
        # Memoria episódica para el puzzle actual
        self.episodic_memory = []
        self.pattern_memory = {}
        
        logger.info(f"V-JEPA Observer Contrastive inicializado en {self.device}")
        
    def _load_pretrained_encoder(self, weights_path: str):
        """Carga el encoder V-JEPA contrastivo pre-entrenado"""
        encoder = SimpleEncoder(input_size=32, output_dim=self.embedding_dim)
        
        if Path(weights_path).exists():
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                encoder.eval()
                logger.info(f"✅ Encoder Contrastive cargado - Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando encoder: {e}")
                logger.info("Usando encoder aleatorio")
        else:
            logger.warning(f"⚠️ No existe {weights_path}")
            logger.info("Entrena primero con: python train_vjepa_contrastive.py --train")
        
        encoder.to(self.device)
        return encoder
    
    def reset_memory(self):
        """Resetea memoria entre puzzles (cumple reglas ARC)"""
        self.episodic_memory = []
        self.pattern_memory = {}
        logger.info("Memoria reseteada para nuevo puzzle")
    
    def observe(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Observa una transición y aprende patrones emergentes
        """
        # Extraer representaciones usando V-JEPA
        input_features = self._extract_features(input_grid)
        output_features = self._extract_features(output_grid)
        
        # Vector de transformación en espacio latente
        transform_vector = output_features - input_features
        
        # Analizar el tipo de transformación
        pattern_info = self._analyze_transformation(
            input_grid, output_grid, 
            input_features, output_features,
            transform_vector
        )
        
        # Guardar en memoria episódica
        self.episodic_memory.append({
            'input': input_grid,
            'output': output_grid,
            'input_features': input_features,
            'output_features': output_features,
            'transform_vector': transform_vector,
            'pattern': pattern_info
        })
        
        # Actualizar memoria de patrones
        pattern_type = pattern_info['type']
        if pattern_type not in self.pattern_memory:
            self.pattern_memory[pattern_type] = []
        self.pattern_memory[pattern_type].append(transform_vector)
        
        return {
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape,
            'emergent_pattern': pattern_info,
            'latent_similarity': float(self._cosine_similarity(input_features, output_features)),
            'transform_magnitude': float(np.linalg.norm(transform_vector))
        }
    
    def _extract_features(self, grid: np.ndarray) -> np.ndarray:
        """Extrae características usando el encoder V-JEPA"""
        # Preparar grid para 32x32
        h, w = grid.shape
        padded = np.zeros((32, 32), dtype=np.float32)
        padded[:min(h, 32), :min(w, 32)] = grid[:min(h, 32), :min(w, 32)] / 10.0  # Normalizar
        
        # Convertir a tensor
        tensor = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Extraer características
        with torch.no_grad():
            features = self.encoder(tensor)
        
        return features.cpu().numpy().squeeze()
    
    def _analyze_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray,
                               input_features: np.ndarray, output_features: np.ndarray,
                               transform_vector: np.ndarray) -> Dict:
        """
        Analiza el tipo de transformación basándose en características aprendidas
        """
        # Calcular métricas de similitud
        feature_similarity = self._cosine_similarity(input_features, output_features)
        transform_magnitude = np.linalg.norm(transform_vector)
        
        # Analizar cambios estructurales
        shape_changed = input_grid.shape != output_grid.shape
        colors_changed = not np.array_equal(np.unique(input_grid), np.unique(output_grid))
        
        # Determinar tipo de patrón basado en métricas
        if transform_magnitude < 0.1:
            pattern_type = 'identity'
            confidence = 0.95
        elif feature_similarity > 0.9 and not shape_changed:
            pattern_type = 'local_modification'
            confidence = feature_similarity
        elif feature_similarity > 0.7:
            if shape_changed:
                pattern_type = 'scaling_transformation'
            else:
                pattern_type = 'color_mapping'
            confidence = feature_similarity
        elif feature_similarity > 0.5:
            pattern_type = 'structural_reorganization'
            confidence = feature_similarity
        else:
            pattern_type = 'complex_transformation'
            confidence = 0.3 + feature_similarity * 0.2
        
        # Si tenemos múltiples ejemplos, verificar consistencia
        if len(self.episodic_memory) > 0:
            consistency = self._check_pattern_consistency(transform_vector, pattern_type)
            confidence = (confidence + consistency) / 2
        
        return {
            'type': pattern_type,
            'confidence': float(confidence),
            'feature_similarity': float(feature_similarity),
            'transform_magnitude': float(transform_magnitude),
            'shape_changed': shape_changed,
            'colors_changed': colors_changed
        }
    
    def _check_pattern_consistency(self, current_transform: np.ndarray, 
                                  current_type: str) -> float:
        """
        Verifica consistencia con patrones previos
        """
        if current_type not in self.pattern_memory:
            return 0.5
        
        previous_transforms = self.pattern_memory[current_type]
        if not previous_transforms:
            return 0.5
        
        # Calcular similitud con transformaciones previas del mismo tipo
        similarities = []
        for prev_transform in previous_transforms[-3:]:  # Últimas 3
            sim = self._cosine_similarity(current_transform, prev_transform)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def predict_transformation(self, test_input: np.ndarray) -> np.ndarray:
        """
        Predice la transformación para un nuevo input
        """
        if not self.episodic_memory:
            return test_input.copy()
        
        # Extraer características del test
        test_features = self._extract_features(test_input)
        
        # Encontrar el ejemplo más similar en memoria
        best_match = None
        best_similarity = -1
        
        for memory in self.episodic_memory:
            sim = self._cosine_similarity(test_features, memory['input_features'])
            if sim > best_similarity:
                best_similarity = sim
                best_match = memory
        
        if best_match is None or best_similarity < 0.3:
            return test_input.copy()
        
        # Aplicar transformación similar
        if best_match['pattern']['type'] == 'identity':
            return test_input.copy()
        
        elif best_match['pattern']['type'] == 'scaling_transformation':
            # Aplicar escala similar
            scale_factor = (
                best_match['output'].shape[0] / best_match['input'].shape[0],
                best_match['output'].shape[1] / best_match['input'].shape[1]
            )
            return self._apply_scaling(test_input, scale_factor)
        
        elif best_match['pattern']['type'] in ['color_mapping', 'local_modification']:
            # Aplicar mapeo de colores
            return self._apply_color_mapping(
                test_input,
                best_match['input'],
                best_match['output']
            )
        
        elif best_match['pattern']['type'] == 'structural_reorganization':
            # Intentar aplicar reorganización similar
            return self._apply_structural_change(
                test_input,
                best_match['input'],
                best_match['output']
            )
        
        else:  # complex_transformation
            # Para transformaciones complejas, intentar múltiples estrategias
            result = self._apply_complex_transformation(
                test_input,
                best_match,
                test_features
            )
            return result if result is not None else test_input.copy()
    
    def _apply_scaling(self, grid: np.ndarray, scale_factor: Tuple[float, float]) -> np.ndarray:
        """Aplica escalado a un grid"""
        h_scale, w_scale = scale_factor
        
        if h_scale == int(h_scale) and w_scale == int(w_scale):
            # Escala entera - repetir píxeles
            return np.repeat(
                np.repeat(grid, int(h_scale), axis=0),
                int(w_scale), axis=1
            )
        else:
            # Escala fraccionaria - necesita interpolación
            new_h = int(grid.shape[0] * h_scale)
            new_w = int(grid.shape[1] * w_scale)
            
            # Implementación simple de nearest neighbor
            result = np.zeros((new_h, new_w), dtype=grid.dtype)
            for i in range(new_h):
                for j in range(new_w):
                    src_i = int(i / h_scale)
                    src_j = int(j / w_scale)
                    if src_i < grid.shape[0] and src_j < grid.shape[1]:
                        result[i, j] = grid[src_i, src_j]
            
            return result
    
    def _apply_color_mapping(self, test_grid: np.ndarray,
                            ref_input: np.ndarray,
                            ref_output: np.ndarray) -> np.ndarray:
        """Aplica mapeo de colores aprendido"""
        result = test_grid.copy()
        
        # Solo si las dimensiones son compatibles
        if ref_input.shape != ref_output.shape:
            return result
        
        # Aprender mapeo de colores
        color_map = {}
        for color in np.unique(ref_input):
            mask = ref_input == color
            output_colors = ref_output[mask]
            if len(output_colors) > 0:
                # Color más frecuente en el output para este color de input
                unique, counts = np.unique(output_colors, return_counts=True)
                color_map[color] = unique[np.argmax(counts)]
        
        # Aplicar mapeo
        for old_color, new_color in color_map.items():
            result[test_grid == old_color] = new_color
        
        return result
    
    def _apply_structural_change(self, test_grid: np.ndarray,
                                ref_input: np.ndarray,
                                ref_output: np.ndarray) -> np.ndarray:
        """
        Aplica cambios estructurales detectados
        """
        # Detectar si es rotación
        for k in range(4):
            if np.array_equal(np.rot90(ref_input, k), ref_output):
                return np.rot90(test_grid, k)
        
        # Detectar si es reflexión
        if np.array_equal(np.fliplr(ref_input), ref_output):
            return np.fliplr(test_grid)
        if np.array_equal(np.flipud(ref_input), ref_output):
            return np.flipud(test_grid)
        
        # Si no es transformación geométrica simple, mantener original
        return test_grid.copy()
    
    def _apply_complex_transformation(self, test_grid: np.ndarray,
                                     best_match: Dict,
                                     test_features: np.ndarray) -> Optional[np.ndarray]:
        """
        Maneja transformaciones complejas usando múltiples estrategias
        """
        # Estrategia 1: Interpolar en espacio latente
        if len(self.episodic_memory) >= 2:
            # Encontrar transformación promedio
            avg_transform = np.mean([m['transform_vector'] for m in self.episodic_memory], axis=0)
            
            # Aplicar transformación promedio (esto es conceptual)
            # En la práctica, necesitaríamos un decoder
            pass
        
        # Estrategia 2: Composición de transformaciones simples
        result = test_grid.copy()
        
        # Intentar aplicar múltiples transformaciones en secuencia
        if best_match['pattern']['colors_changed']:
            result = self._apply_color_mapping(
                result,
                best_match['input'],
                best_match['output']
            )
        
        if best_match['pattern']['shape_changed']:
            scale = (
                best_match['output'].shape[0] / best_match['input'].shape[0],
                best_match['output'].shape[1] / best_match['input'].shape[1]
            )
            result = self._apply_scaling(result, scale)
        
        return result
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre vectores"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
        
        return float(np.dot(a, b) / (a_norm * b_norm))
    
    def get_learned_patterns(self) -> Dict:
        """
        Retorna resumen de patrones aprendidos
        """
        summary = {
            'total_observations': len(self.episodic_memory),
            'pattern_types': {},
            'average_confidence': 0.0
        }
        
        if self.episodic_memory:
            # Contar tipos de patrones
            for memory in self.episodic_memory:
                pattern_type = memory['pattern']['type']
                if pattern_type not in summary['pattern_types']:
                    summary['pattern_types'][pattern_type] = 0
                summary['pattern_types'][pattern_type] += 1
            
            # Confianza promedio
            confidences = [m['pattern']['confidence'] for m in self.episodic_memory]
            summary['average_confidence'] = np.mean(confidences)
        
        return summary