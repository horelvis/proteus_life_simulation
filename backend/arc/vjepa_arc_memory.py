#!/usr/bin/env python3
"""
V-JEPA mejorado para ARC con memoria persistente
Basado en los conceptos de V-JEPA de Meta pero adaptado para puzzles discretos
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ARCTransformation:
    """Transformación aprendida de ARC"""
    input_pattern: str  # Hash del patrón de entrada
    output_pattern: str  # Hash del patrón de salida
    transformation_type: str  # Tipo detectado
    confidence: float
    examples_seen: int
    embedding: List[float]  # Representación latente

class VJEPAARCMemory:
    """
    V-JEPA adaptado para ARC con memoria persistente
    Aprende y recuerda transformaciones entre sesiones
    """
    
    def __init__(self, memory_path: str = "arc_vjepa_memory.pkl"):
        self.memory_path = Path(memory_path)
        self.embedding_dim = 128
        
        # Memoria persistente
        self.known_transformations = {}  # Hash -> ARCTransformation
        self.pattern_embeddings = {}  # Patrón -> embedding
        self.transformation_types = {
            'color_mapping': [],
            'rotation': [],
            'reflection': [],
            'scaling': [],
            'pattern_fill': [],
            'object_manipulation': [],
            'unknown': []
        }
        
        # Cargar memoria si existe
        self.load_memory()
        
    def load_memory(self):
        """Carga la memoria persistente"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_transformations = data.get('transformations', {})
                    self.pattern_embeddings = data.get('embeddings', {})
                    self.transformation_types = data.get('types', self.transformation_types)
                logger.info(f"Memoria cargada: {len(self.known_transformations)} transformaciones conocidas")
            except Exception as e:
                logger.error(f"Error cargando memoria: {e}")
    
    def save_memory(self):
        """Guarda la memoria persistente"""
        try:
            data = {
                'transformations': self.known_transformations,
                'embeddings': self.pattern_embeddings,
                'types': self.transformation_types
            }
            with open(self.memory_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Memoria guardada: {len(self.known_transformations)} transformaciones")
        except Exception as e:
            logger.error(f"Error guardando memoria: {e}")
    
    def observe(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Observa una transformación y la aprende/reconoce
        """
        # Generar embeddings
        input_embedding = self._create_embedding(input_grid)
        output_embedding = self._create_embedding(output_grid)
        
        # Hash para identificar patrones únicos
        input_hash = self._hash_pattern(input_grid)
        output_hash = self._hash_pattern(output_grid)
        transform_hash = f"{input_hash}->{output_hash}"
        
        # Buscar si ya conocemos esta transformación
        if transform_hash in self.known_transformations:
            transform = self.known_transformations[transform_hash]
            transform.examples_seen += 1
            transform.confidence = min(0.99, transform.confidence + 0.1)
            
            result = {
                'recognized': True,
                'transformation_type': transform.transformation_type,
                'confidence': transform.confidence,
                'examples_seen': transform.examples_seen,
                'message': f"Transformación conocida: {transform.transformation_type}"
            }
        else:
            # Nueva transformación - intentar clasificarla
            transform_type = self._classify_transformation(input_grid, output_grid, input_embedding, output_embedding)
            
            # Crear nueva entrada en memoria
            transform = ARCTransformation(
                input_pattern=input_hash,
                output_pattern=output_hash,
                transformation_type=transform_type,
                confidence=0.3,  # Confianza inicial baja
                examples_seen=1,
                embedding=(output_embedding - input_embedding).tolist()
            )
            
            self.known_transformations[transform_hash] = transform
            self.transformation_types[transform_type].append(transform_hash)
            
            result = {
                'recognized': False,
                'transformation_type': transform_type,
                'confidence': transform.confidence,
                'examples_seen': 1,
                'message': f"Nueva transformación aprendida: {transform_type}"
            }
        
        # Guardar memoria actualizada
        self.save_memory()
        
        # Agregar información sobre patrones similares
        similar_patterns = self._find_similar_transformations(transform.embedding)
        result['similar_patterns'] = len(similar_patterns)
        result['memory_size'] = len(self.known_transformations)
        
        return result
    
    def _create_embedding(self, grid: np.ndarray) -> np.ndarray:
        """
        Crea un embedding rico de un grid
        Similar a como V-JEPA procesa frames de video
        """
        h, w = grid.shape if len(grid.shape) == 2 else (grid.shape[0], grid.shape[1])
        embedding = np.zeros(self.embedding_dim)
        
        # Características globales (primeros 10 dims)
        embedding[0] = np.mean(grid)
        embedding[1] = np.std(grid)
        embedding[2] = np.count_nonzero(grid) / grid.size
        embedding[3] = h / 30.0  # Normalizado a tamaño máximo esperado
        embedding[4] = w / 30.0
        
        # Distribución de colores (dims 5-14)
        unique_vals, counts = np.unique(grid, return_counts=True)
        for i, val in enumerate(unique_vals[:10]):
            if 5 + i < 15:
                embedding[5 + i] = counts[i] / grid.size
        
        # Patrones espaciales - dividir en cuadrantes (dims 15-30)
        quad_h, quad_w = h // 2, w // 2
        quadrants = [
            grid[:quad_h, :quad_w],
            grid[:quad_h, quad_w:],
            grid[quad_h:, :quad_w],
            grid[quad_h:, quad_w:]
        ]
        
        for i, quad in enumerate(quadrants):
            if quad.size > 0:
                embedding[15 + i*4] = np.mean(quad)
                embedding[16 + i*4] = np.std(quad)
                embedding[17 + i*4] = np.count_nonzero(quad) / quad.size
                embedding[18 + i*4] = len(np.unique(quad))
        
        # Detección de estructuras (dims 31-50)
        # Líneas horizontales/verticales
        for i in range(min(h, 10)):
            embedding[31 + i] = np.mean(grid[i, :]) if i < h else 0
        for i in range(min(w, 10)):
            embedding[41 + i] = np.mean(grid[:, i]) if i < w else 0
        
        # El resto del embedding puede ser usado para características más complejas
        
        return embedding
    
    def _hash_pattern(self, grid: np.ndarray) -> str:
        """Genera un hash único para un patrón"""
        return str(hash(grid.tobytes()))
    
    def _classify_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray,
                                input_emb: np.ndarray, output_emb: np.ndarray) -> str:
        """
        Clasifica el tipo de transformación basándose en características
        """
        # Análisis básico de la transformación
        same_shape = input_grid.shape == output_grid.shape
        same_values = set(input_grid.flatten()) == set(output_grid.flatten())
        
        if same_shape:
            if not same_values:
                # Cambio de colores
                return 'color_mapping'
            else:
                # Posible rotación o reflexión
                if np.array_equal(np.rot90(input_grid), output_grid):
                    return 'rotation'
                elif np.array_equal(np.fliplr(input_grid), output_grid) or \
                     np.array_equal(np.flipud(input_grid), output_grid):
                    return 'reflection'
        
        # Cambio de tamaño
        if input_grid.shape != output_grid.shape:
            if output_grid.shape[0] > input_grid.shape[0] or \
               output_grid.shape[1] > input_grid.shape[1]:
                return 'scaling'
        
        # Si no podemos clasificarlo
        return 'unknown'
    
    def _find_similar_transformations(self, embedding: List[float], threshold: float = 0.8) -> List[str]:
        """
        Encuentra transformaciones similares en la memoria
        """
        similar = []
        emb_array = np.array(embedding)
        
        for trans_hash, transform in self.known_transformations.items():
            trans_emb = np.array(transform.embedding)
            similarity = 1 - np.linalg.norm(emb_array - trans_emb) / (np.linalg.norm(emb_array) + np.linalg.norm(trans_emb) + 1e-8)
            
            if similarity > threshold:
                similar.append(trans_hash)
        
        return similar
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas de la memoria"""
        stats = {
            'total_transformations': len(self.known_transformations),
            'transformation_types': {}
        }
        
        for t_type, hashes in self.transformation_types.items():
            stats['transformation_types'][t_type] = len(hashes)
        
        # Transformaciones más vistas
        most_seen = sorted(
            self.known_transformations.items(),
            key=lambda x: x[1].examples_seen,
            reverse=True
        )[:5]
        
        stats['most_seen'] = [
            {
                'type': t.transformation_type,
                'examples': t.examples_seen,
                'confidence': t.confidence
            }
            for _, t in most_seen
        ]
        
        return stats