#!/usr/bin/env python3
"""
V-JEPA para ARC usando pesos PUROS pre-entrenados
Combina el conocimiento visual aprendido con las reglas de ARC
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Importar arquitectura del encoder puro
import sys
sys.path.append('/app/arc')
from train_vjepa_pure import PureVJEPAEncoder


class VJEPAPureARC:
    """
    V-JEPA para ARC usando encoder pre-entrenado PURO
    - Usa pesos aprendidos de forma auto-supervisada
    - NO tiene conceptos hardcodeados
    - Cumple reglas ARC (resetea entre puzzles)
    """
    
    def __init__(self, weights_path: str = "/app/arc/vjepa_pure_weights/vjepa_pure_final.pth"):
        # Cargar encoder pre-entrenado
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = self._load_pretrained_encoder(weights_path)
        
        # Memoria temporal del puzzle actual
        self.reset()
        
    def _load_pretrained_encoder(self, weights_path: str) -> PureVJEPAEncoder:
        """Carga el encoder pre-entrenado"""
        encoder = PureVJEPAEncoder(embedding_dim=256)
        
        if Path(weights_path).exists():
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                encoder.eval()  # Modo evaluación
                encoder.to(self.device)
                logger.info(f"✅ Encoder PURO cargado desde {weights_path}")
                logger.info(f"   Entrenado por {checkpoint.get('epoch', 'N/A')} épocas")
                logger.info(f"   Loss final: {checkpoint.get('loss', 'N/A'):.6f}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo cargar encoder pre-entrenado: {e}")
                logger.info("   Usando encoder aleatorio (no entrenado)")
        else:
            logger.warning(f"⚠️ No existe {weights_path}")
            logger.info("   Usando encoder aleatorio - entrena primero con train_vjepa_pure.py")
        
        encoder.to(self.device)
        return encoder
    
    def reset(self):
        """Resetea memoria del puzzle (requerido entre puzzles)"""
        self.current_puzzle_id = None
        self.puzzle_examples = []
        self.learned_patterns = []
        self.confidence = 0.0
        logger.info("Memoria del puzzle reseteada")
    
    def start_new_puzzle(self, puzzle_id: str):
        """Inicia nuevo puzzle"""
        if self.current_puzzle_id != puzzle_id:
            self.reset()
            self.current_puzzle_id = puzzle_id
            logger.info(f"Nuevo puzzle: {puzzle_id}")
    
    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Codifica un grid usando el encoder pre-entrenado
        El encoder ya "sabe ver" patrones (aprendió solo)
        """
        # Preparar grid
        h, w = grid.shape
        padded = np.zeros((30, 30), dtype=np.float32)
        padded[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        
        # Convertir a tensor
        tensor = torch.FloatTensor(padded).unsqueeze(0).to(self.device)
        
        # Codificar con encoder pre-entrenado
        with torch.no_grad():
            embedding = self.encoder(tensor)
        
        return embedding.cpu().numpy().squeeze()
    
    def learn_from_examples(self, train_examples: List[Dict]) -> Dict:
        """
        Aprende del puzzle actual usando representaciones del encoder
        """
        results = {
            'examples_learned': 0,
            'pattern_detected': None,
            'confidence': 0.0,
            'embeddings_similarity': 0.0
        }
        
        if not train_examples:
            return results
        
        all_input_embeddings = []
        all_output_embeddings = []
        transformations = []
        
        # Codificar todos los ejemplos
        for idx, example in enumerate(train_examples):
            input_grid = np.array(example.get('input', []))
            output_grid = np.array(example.get('output', []))
            
            if input_grid.size == 0 or output_grid.size == 0:
                continue
            
            # Usar encoder pre-entrenado para obtener representaciones
            input_emb = self.encode_grid(input_grid)
            output_emb = self.encode_grid(output_grid)
            
            all_input_embeddings.append(input_emb)
            all_output_embeddings.append(output_emb)
            
            # Calcular vector de transformación en espacio latente
            transform_vector = output_emb - input_emb
            transformations.append(transform_vector)
            
            # Guardar ejemplo procesado
            self.puzzle_examples.append({
                'input': input_grid,
                'output': output_grid,
                'input_embedding': input_emb,
                'output_embedding': output_emb,
                'transform': transform_vector
            })
            
            results['examples_learned'] += 1
        
        # Analizar consistencia de transformaciones
        if len(transformations) > 1:
            # Calcular similitud entre vectores de transformación
            similarities = []
            for i in range(len(transformations)):
                for j in range(i+1, len(transformations)):
                    sim = self._cosine_similarity(transformations[i], transformations[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            results['embeddings_similarity'] = float(avg_similarity)
            
            # Determinar tipo de patrón basado en similitud
            if avg_similarity > 0.9:
                results['pattern_detected'] = 'highly_consistent'
                results['confidence'] = 0.9
            elif avg_similarity > 0.7:
                results['pattern_detected'] = 'consistent'
                results['confidence'] = 0.7
            elif avg_similarity > 0.5:
                results['pattern_detected'] = 'partially_consistent'
                results['confidence'] = 0.5
            else:
                results['pattern_detected'] = 'variable'
                results['confidence'] = 0.3
            
            self.confidence = results['confidence']
            
            # Calcular transformación promedio
            avg_transform = np.mean(transformations, axis=0)
            self.learned_patterns.append({
                'avg_transform': avg_transform,
                'std_transform': np.std(transformations, axis=0),
                'confidence': results['confidence']
            })
        
        return results
    
    def predict(self, test_input: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predice output usando transformaciones aprendidas
        """
        if not self.puzzle_examples:
            return test_input.copy(), 0.0
        
        # Codificar test input
        test_embedding = self.encode_grid(test_input)
        
        # Encontrar ejemplo más similar
        best_match_idx = -1
        best_similarity = -1
        
        for idx, example in enumerate(self.puzzle_examples):
            sim = self._cosine_similarity(test_embedding, example['input_embedding'])
            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = idx
        
        if best_match_idx >= 0 and best_similarity > 0.5:
            # Aplicar transformación del ejemplo más similar
            best_example = self.puzzle_examples[best_match_idx]
            
            # Si los tamaños coinciden, intentar mapeo directo
            if test_input.shape == best_example['input'].shape:
                return self._apply_transformation(
                    test_input,
                    best_example['input'],
                    best_example['output']
                ), self.confidence * best_similarity
            else:
                # Aplicar transformación por color mapping
                return self._apply_color_mapping(
                    test_input,
                    best_example['input'],
                    best_example['output']
                ), self.confidence * best_similarity * 0.8
        
        return test_input.copy(), 0.0
    
    def _apply_transformation(self, test_grid, ref_input, ref_output):
        """Aplica transformación aprendida"""
        result = test_grid.copy()
        
        # Intentar detectar tipo de transformación
        if ref_input.shape == ref_output.shape:
            # Mismo tamaño - posible rotación o reflexión
            if np.array_equal(np.rot90(ref_input, 1), ref_output):
                return np.rot90(test_grid, 1)
            elif np.array_equal(np.rot90(ref_input, 2), ref_output):
                return np.rot90(test_grid, 2)
            elif np.array_equal(np.rot90(ref_input, 3), ref_output):
                return np.rot90(test_grid, 3)
            elif np.array_equal(np.fliplr(ref_input), ref_output):
                return np.fliplr(test_grid)
            elif np.array_equal(np.flipud(ref_input), ref_output):
                return np.flipud(test_grid)
        
        # Si no es transformación geométrica, aplicar mapeo de colores
        return self._apply_color_mapping(test_grid, ref_input, ref_output)
    
    def _apply_color_mapping(self, test_grid, ref_input, ref_output):
        """Aplica mapeo de colores aprendido"""
        result = test_grid.copy()
        
        # Aprender mapeo de colores
        color_map = {}
        for color in np.unique(ref_input):
            mask = ref_input == color
            if np.any(mask):
                # Ver a qué color se transforma más frecuentemente
                if mask.shape == ref_output.shape:
                    output_colors = ref_output[mask]
                    if len(output_colors) > 0:
                        unique, counts = np.unique(output_colors, return_counts=True)
                        color_map[color] = unique[np.argmax(counts)]
        
        # Aplicar mapeo
        for old_color, new_color in color_map.items():
            result[test_grid == old_color] = new_color
        
        return result
    
    def _cosine_similarity(self, a, b):
        """Calcula similitud coseno"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)
    
    def get_info(self) -> Dict:
        """Información sobre el estado actual"""
        return {
            'encoder': 'pure_pretrained',
            'device': str(self.device),
            'current_puzzle': self.current_puzzle_id,
            'examples_loaded': len(self.puzzle_examples),
            'patterns_learned': len(self.learned_patterns),
            'confidence': self.confidence
        }