#!/usr/bin/env python3
"""
V-JEPA Pre-entrenado para ARC - Versión con conocimiento visual básico
Pre-entrenado en conceptos visuales generales (NO en puzzles ARC)
Cumple las reglas de competencia
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VisualConcept:
    """Concepto visual pre-aprendido"""
    name: str
    pattern: np.ndarray
    confidence: float

class VJEPAPretrainedARC:
    """
    V-JEPA con conocimiento visual pre-entrenado (válido para ARC)
    
    Pre-entrenado en:
    - Formas geométricas básicas
    - Colores de ARC
    - Transformaciones comunes
    - Relaciones espaciales
    
    NO pre-entrenado en:
    - Puzzles específicos de ARC
    - Soluciones de ARC
    """
    
    def __init__(self):
        # Conocimiento visual pre-entrenado (VÁLIDO)
        self._init_pretrained_knowledge()
        
        # Memoria temporal del puzzle actual (se resetea entre puzzles)
        self.reset_puzzle_memory()
        
    def _init_pretrained_knowledge(self):
        """Inicializa conocimiento visual general pre-entrenado"""
        
        # 1. COLORES DE ARC (conocimiento básico válido)
        self.arc_colors = {
            0: "black/empty",
            1: "blue",
            2: "red", 
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        
        # 2. FORMAS BÁSICAS (conocimiento geométrico general)
        self.basic_shapes = {
            'line_horizontal': self._create_line_pattern(horizontal=True),
            'line_vertical': self._create_line_pattern(horizontal=False),
            'square': self._create_square_pattern(),
            'rectangle': self._create_rectangle_pattern(),
            'L_shape': self._create_L_pattern(),
            'T_shape': self._create_T_pattern(),
            'cross': self._create_cross_pattern(),
            'diagonal': self._create_diagonal_pattern()
        }
        
        # 3. TRANSFORMACIONES BÁSICAS (operaciones visuales generales)
        self.known_transformations = {
            'rotation_90': lambda x: np.rot90(x, 1),
            'rotation_180': lambda x: np.rot90(x, 2),
            'rotation_270': lambda x: np.rot90(x, 3),
            'flip_horizontal': lambda x: np.fliplr(x),
            'flip_vertical': lambda x: np.flipud(x),
            'transpose': lambda x: x.T if x.shape[0] == x.shape[1] else x,
            'inverse_colors': lambda x: self._inverse_colors(x),
        }
        
        # 4. RELACIONES ESPACIALES (conceptos espaciales generales)
        self.spatial_relations = {
            'above': lambda a, b: self._check_above(a, b),
            'below': lambda a, b: self._check_below(a, b),
            'left_of': lambda a, b: self._check_left_of(a, b),
            'right_of': lambda a, b: self._check_right_of(a, b),
            'inside': lambda a, b: self._check_inside(a, b),
            'surrounding': lambda a, b: self._check_surrounding(a, b),
            'adjacent': lambda a, b: self._check_adjacent(a, b),
            'aligned': lambda a, b: self._check_aligned(a, b)
        }
        
        # 5. PATRONES DE MOVIMIENTO (física básica 2D)
        self.movement_patterns = {
            'translation': self._detect_translation,
            'rotation': self._detect_rotation,
            'scaling': self._detect_scaling,
            'reflection': self._detect_reflection,
            'shearing': self._detect_shearing
        }
        
        # Embedding pre-entrenado para codificar características visuales
        self.visual_encoder_dim = 256
        self.pretrained_weights = self._initialize_encoder_weights()
        
    def _initialize_encoder_weights(self):
        """Inicializa pesos del encoder visual (simulado)"""
        # En una implementación real, estos serían pesos de una red pre-entrenada
        # en imágenes generales o patrones geométricos (NO en ARC)
        np.random.seed(42)  # Para reproducibilidad
        return {
            'color_embedding': np.random.randn(10, 32),  # 10 colores -> 32 dims
            'shape_embedding': np.random.randn(100, 64),  # formas -> 64 dims
            'spatial_embedding': np.random.randn(50, 32),  # relaciones -> 32 dims
            'motion_embedding': np.random.randn(20, 32)   # movimientos -> 32 dims
        }
    
    def reset_puzzle_memory(self):
        """Resetea memoria del puzzle actual (REQUERIDO entre puzzles)"""
        self.current_puzzle_id = None
        self.puzzle_examples = []
        self.learned_transformation = None
        self.confidence = 0.0
        logger.info("Memoria del puzzle reseteada - comenzando limpio")
    
    def start_new_puzzle(self, puzzle_id: str):
        """Inicia un nuevo puzzle - resetea memoria específica"""
        if self.current_puzzle_id != puzzle_id:
            self.reset_puzzle_memory()
            self.current_puzzle_id = puzzle_id
            logger.info(f"Nuevo puzzle: {puzzle_id} - usando solo conocimiento visual general")
    
    def encode_visual_features(self, grid: np.ndarray) -> np.ndarray:
        """
        Codifica características visuales usando conocimiento pre-entrenado
        (Como un humano que ya sabe qué son colores y formas)
        """
        features = np.zeros(self.visual_encoder_dim)
        h, w = grid.shape
        
        # 1. Codificar distribución de colores (conocimiento de colores)
        color_features = np.zeros(32)
        unique_colors, counts = np.unique(grid, return_counts=True)
        for color, count in zip(unique_colors, counts):
            if color < 10:
                # Usar embedding pre-entrenado de colores
                color_features += self.pretrained_weights['color_embedding'][color] * (count / grid.size)
        features[:32] = color_features
        
        # 2. Detectar formas conocidas (conocimiento geométrico)
        shape_features = np.zeros(64)
        detected_shapes = self._detect_known_shapes(grid)
        for shape_name, locations in detected_shapes.items():
            shape_idx = hash(shape_name) % 100
            shape_features += self.pretrained_weights['shape_embedding'][shape_idx] * len(locations)
        features[32:96] = shape_features
        
        # 3. Codificar estructura espacial (conocimiento espacial)
        spatial_features = self._encode_spatial_structure(grid)
        features[96:128] = spatial_features[:32]
        
        # 4. Detectar simetrías (conocimiento de patrones)
        symmetry_features = np.array([
            self._check_horizontal_symmetry(grid),
            self._check_vertical_symmetry(grid),
            self._check_rotational_symmetry(grid),
            self._check_diagonal_symmetry(grid)
        ])
        features[128:132] = symmetry_features
        
        # 5. Características de textura/densidad
        texture_features = np.array([
            np.mean(grid > 0),  # Densidad
            np.std(grid),       # Variabilidad
            self._compute_edge_density(grid),  # Bordes
            self._compute_clustering(grid)     # Agrupamiento
        ])
        features[132:136] = texture_features
        
        return features
    
    def learn_from_examples(self, train_examples: List[Dict]) -> Dict:
        """
        Aprende del puzzle actual usando conocimiento visual pre-entrenado
        """
        results = {
            'examples_learned': 0,
            'pattern_detected': None,
            'confidence': 0.0,
            'visual_concepts': [],
            'transformation_type': None
        }
        
        if not train_examples:
            return results
        
        # Analizar cada ejemplo con conocimiento pre-entrenado
        for idx, example in enumerate(train_examples):
            input_grid = np.array(example.get('input', []))
            output_grid = np.array(example.get('output', []))
            
            # Codificar con conocimiento visual pre-entrenado
            input_features = self.encode_visual_features(input_grid)
            output_features = self.encode_visual_features(output_grid)
            
            # Detectar conceptos visuales conocidos
            input_concepts = self._identify_visual_concepts(input_grid)
            output_concepts = self._identify_visual_concepts(output_grid)
            
            # Analizar transformación usando conocimiento de movimientos
            transformation = self._analyze_transformation_with_knowledge(
                input_grid, output_grid, 
                input_features, output_features,
                input_concepts, output_concepts
            )
            
            self.puzzle_examples.append({
                'input': input_grid,
                'output': output_grid,
                'input_features': input_features,
                'output_features': output_features,
                'transformation': transformation,
                'concepts': {
                    'input': input_concepts,
                    'output': output_concepts
                }
            })
            
            results['examples_learned'] += 1
            results['visual_concepts'].extend(input_concepts + output_concepts)
        
        # Encontrar patrón común usando conocimiento pre-entrenado
        if self.puzzle_examples:
            common_pattern = self._find_common_pattern_with_knowledge()
            results['pattern_detected'] = common_pattern['type']
            results['transformation_type'] = common_pattern['transformation']
            results['confidence'] = common_pattern['confidence']
            self.learned_transformation = common_pattern
        
        return results
    
    def predict(self, test_input: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predice usando conocimiento visual + lo aprendido del puzzle
        """
        if not self.learned_transformation:
            return test_input.copy(), 0.0
        
        # Codificar test con conocimiento pre-entrenado
        test_features = self.encode_visual_features(test_input)
        test_concepts = self._identify_visual_concepts(test_input)
        
        # Aplicar transformación aprendida
        prediction = self._apply_learned_transformation(
            test_input, 
            test_features,
            test_concepts,
            self.learned_transformation
        )
        
        return prediction, self.confidence
    
    # === Métodos auxiliares para formas básicas ===
    
    def _create_line_pattern(self, horizontal=True, length=3):
        """Crea patrón de línea"""
        if horizontal:
            return np.ones((1, length))
        return np.ones((length, 1))
    
    def _create_square_pattern(self, size=3):
        """Crea patrón de cuadrado"""
        pattern = np.zeros((size, size))
        pattern[0, :] = 1
        pattern[-1, :] = 1
        pattern[:, 0] = 1
        pattern[:, -1] = 1
        return pattern
    
    def _create_rectangle_pattern(self):
        """Crea patrón de rectángulo"""
        pattern = np.zeros((3, 5))
        pattern[0, :] = 1
        pattern[-1, :] = 1
        pattern[:, 0] = 1
        pattern[:, -1] = 1
        return pattern
    
    def _create_L_pattern(self):
        """Crea patrón en forma de L"""
        pattern = np.zeros((3, 3))
        pattern[:, 0] = 1
        pattern[-1, :] = 1
        return pattern
    
    def _create_T_pattern(self):
        """Crea patrón en forma de T"""
        pattern = np.zeros((3, 3))
        pattern[0, :] = 1
        pattern[:, 1] = 1
        return pattern
    
    def _create_cross_pattern(self):
        """Crea patrón de cruz"""
        pattern = np.zeros((3, 3))
        pattern[1, :] = 1
        pattern[:, 1] = 1
        return pattern
    
    def _create_diagonal_pattern(self):
        """Crea patrón diagonal"""
        return np.eye(3)
    
    # === Métodos de detección ===
    
    def _detect_known_shapes(self, grid: np.ndarray) -> Dict[str, List]:
        """Detecta formas conocidas en el grid"""
        detected = {}
        
        # Buscar cada forma conocida
        for shape_name, shape_pattern in self.basic_shapes.items():
            locations = self._find_pattern_in_grid(grid, shape_pattern)
            if locations:
                detected[shape_name] = locations
        
        return detected
    
    def _find_pattern_in_grid(self, grid: np.ndarray, pattern: np.ndarray) -> List[Tuple[int, int]]:
        """Busca un patrón en el grid"""
        locations = []
        h, w = grid.shape
        ph, pw = pattern.shape
        
        if ph > h or pw > w:
            return locations
        
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                region = grid[i:i+ph, j:j+pw]
                # Verificar si el patrón coincide (ignorando colores específicos)
                if np.array_equal(region > 0, pattern > 0):
                    locations.append((i, j))
        
        return locations
    
    def _identify_visual_concepts(self, grid: np.ndarray) -> List[str]:
        """Identifica conceptos visuales en el grid"""
        concepts = []
        
        # Colores presentes
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color > 0 and color < 10:
                concepts.append(f"color_{self.arc_colors[color]}")
        
        # Formas detectadas
        shapes = self._detect_known_shapes(grid)
        concepts.extend([f"shape_{name}" for name in shapes.keys()])
        
        # Propiedades globales
        if self._check_horizontal_symmetry(grid) > 0.9:
            concepts.append("symmetric_horizontal")
        if self._check_vertical_symmetry(grid) > 0.9:
            concepts.append("symmetric_vertical")
        
        return concepts
    
    def _analyze_transformation_with_knowledge(self, input_grid, output_grid, 
                                              input_features, output_features,
                                              input_concepts, output_concepts):
        """Analiza transformación usando conocimiento pre-entrenado"""
        
        # Verificar transformaciones conocidas
        for trans_name, trans_func in self.known_transformations.items():
            try:
                transformed = trans_func(input_grid)
                if np.array_equal(transformed, output_grid):
                    return {'type': trans_name, 'confidence': 1.0}
            except:
                pass
        
        # Detectar movimientos
        for movement_name, detect_func in self.movement_patterns.items():
            result = detect_func(input_grid, output_grid)
            if result['detected']:
                return {'type': movement_name, 'params': result, 'confidence': result['confidence']}
        
        # Analizar cambio en conceptos
        added_concepts = set(output_concepts) - set(input_concepts)
        removed_concepts = set(input_concepts) - set(output_concepts)
        
        if added_concepts or removed_concepts:
            return {
                'type': 'concept_change',
                'added': list(added_concepts),
                'removed': list(removed_concepts),
                'confidence': 0.7
            }
        
        return {'type': 'unknown', 'confidence': 0.3}
    
    def _find_common_pattern_with_knowledge(self):
        """Encuentra patrón común usando conocimiento pre-entrenado"""
        if not self.puzzle_examples:
            return {'type': 'none', 'transformation': None, 'confidence': 0.0}
        
        # Analizar consistencia de transformaciones
        transformations = [ex['transformation'] for ex in self.puzzle_examples]
        
        # Verificar si todas son del mismo tipo
        trans_types = [t['type'] for t in transformations]
        if len(set(trans_types)) == 1:
            # Transformación consistente
            self.confidence = 0.9
            return {
                'type': 'consistent',
                'transformation': transformations[0],
                'confidence': 0.9
            }
        
        # Buscar patrón mayoritario
        from collections import Counter
        type_counts = Counter(trans_types)
        most_common_type, count = type_counts.most_common(1)[0]
        
        if count > len(transformations) / 2:
            # Patrón mayoritario
            self.confidence = 0.7
            return {
                'type': 'majority',
                'transformation': {'type': most_common_type},
                'confidence': 0.7
            }
        
        # Patrón variable
        self.confidence = 0.4
        return {
            'type': 'variable',
            'transformation': {'type': 'mixed'},
            'confidence': 0.4
        }
    
    def _apply_learned_transformation(self, test_input, test_features, test_concepts, transformation):
        """Aplica transformación aprendida al test"""
        if not transformation or not transformation.get('transformation'):
            return test_input.copy()
        
        trans_type = transformation['transformation'].get('type')
        
        # Aplicar transformación conocida
        if trans_type in self.known_transformations:
            return self.known_transformations[trans_type](test_input)
        
        # Aplicar transformación basada en ejemplos
        if self.puzzle_examples:
            # Encontrar ejemplo más similar
            best_match = self._find_most_similar_example(test_features)
            if best_match:
                # Aplicar transformación del ejemplo más similar
                return self._apply_example_transformation(
                    test_input, 
                    best_match['input'],
                    best_match['output']
                )
        
        return test_input.copy()
    
    def _find_most_similar_example(self, test_features):
        """Encuentra el ejemplo más similar al test"""
        if not self.puzzle_examples:
            return None
        
        best_similarity = -1
        best_example = None
        
        for example in self.puzzle_examples:
            similarity = self._cosine_similarity(test_features, example['input_features'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_example = example
        
        return best_example if best_similarity > 0.5 else None
    
    def _apply_example_transformation(self, test_input, example_input, example_output):
        """Aplica transformación basada en un ejemplo"""
        # Mapeo de colores
        color_map = {}
        for color in np.unique(example_input):
            mask = example_input == color
            if np.any(mask):
                output_colors = example_output[mask]
                if len(output_colors) > 0:
                    # Usar el color más común en la salida
                    unique, counts = np.unique(output_colors, return_counts=True)
                    color_map[color] = unique[np.argmax(counts)]
        
        # Aplicar mapeo
        result = test_input.copy()
        for old_color, new_color in color_map.items():
            result[test_input == old_color] = new_color
        
        return result
    
    # === Métodos auxiliares de análisis ===
    
    def _check_horizontal_symmetry(self, grid):
        """Verifica simetría horizontal"""
        return np.sum(grid == np.fliplr(grid)) / grid.size
    
    def _check_vertical_symmetry(self, grid):
        """Verifica simetría vertical"""  
        return np.sum(grid == np.flipud(grid)) / grid.size
    
    def _check_rotational_symmetry(self, grid):
        """Verifica simetría rotacional"""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        return np.sum(grid == np.rot90(grid, 2)) / grid.size
    
    def _check_diagonal_symmetry(self, grid):
        """Verifica simetría diagonal"""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
        return np.sum(grid == grid.T) / grid.size
    
    def _compute_edge_density(self, grid):
        """Calcula densidad de bordes"""
        h, w = grid.shape
        edges = 0
        for i in range(h-1):
            for j in range(w-1):
                if grid[i,j] != grid[i+1,j]:
                    edges += 1
                if grid[i,j] != grid[i,j+1]:
                    edges += 1
        return edges / (2 * (h-1) * (w-1) + 1)
    
    def _compute_clustering(self, grid):
        """Calcula nivel de agrupamiento"""
        h, w = grid.shape
        same_neighbors = 0
        total_neighbors = 0
        
        for i in range(h):
            for j in range(w):
                if grid[i,j] > 0:
                    for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w:
                            total_neighbors += 1
                            if grid[ni,nj] == grid[i,j]:
                                same_neighbors += 1
        
        return same_neighbors / (total_neighbors + 1)
    
    def _encode_spatial_structure(self, grid):
        """Codifica estructura espacial del grid"""
        h, w = grid.shape
        features = []
        
        # Dividir en cuadrantes
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            grid[:mid_h, :mid_w],
            grid[:mid_h, mid_w:],
            grid[mid_h:, :mid_w],
            grid[mid_h:, mid_w:]
        ]
        
        for quad in quadrants:
            if quad.size > 0:
                features.extend([
                    np.mean(quad > 0),
                    len(np.unique(quad)),
                    np.std(quad)
                ])
        
        # Pad to fixed size
        result = np.zeros(32)
        result[:len(features)] = features[:32]
        return result
    
    def _inverse_colors(self, grid):
        """Invierte colores (mantiene 0 como 0)"""
        result = grid.copy()
        max_color = 9
        result[grid > 0] = max_color - grid[grid > 0] + 1
        return result
    
    # === Detección de movimientos ===
    
    def _detect_translation(self, input_grid, output_grid):
        """Detecta traslación"""
        # Simplificado - en implementación real sería más sofisticado
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_rotation(self, input_grid, output_grid):
        """Detecta rotación"""
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(input_grid, k), output_grid):
                return {'detected': True, 'angle': k*90, 'confidence': 1.0}
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_scaling(self, input_grid, output_grid):
        """Detecta escalado"""
        h_ratio = output_grid.shape[0] / input_grid.shape[0]
        w_ratio = output_grid.shape[1] / input_grid.shape[1]
        
        if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
            return {'detected': True, 'scale': (h_ratio, w_ratio), 'confidence': 0.8}
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_reflection(self, input_grid, output_grid):
        """Detecta reflexión"""
        if np.array_equal(np.fliplr(input_grid), output_grid):
            return {'detected': True, 'axis': 'horizontal', 'confidence': 1.0}
        if np.array_equal(np.flipud(input_grid), output_grid):
            return {'detected': True, 'axis': 'vertical', 'confidence': 1.0}
        return {'detected': False, 'confidence': 0.0}
    
    def _detect_shearing(self, input_grid, output_grid):
        """Detecta cizallamiento"""
        # Simplificado
        return {'detected': False, 'confidence': 0.0}
    
    # === Relaciones espaciales (stubs) ===
    
    def _check_above(self, a, b):
        return False
    
    def _check_below(self, a, b):
        return False
    
    def _check_left_of(self, a, b):
        return False
    
    def _check_right_of(self, a, b):
        return False
    
    def _check_inside(self, a, b):
        return False
    
    def _check_surrounding(self, a, b):
        return False
    
    def _check_adjacent(self, a, b):
        return False
    
    def _check_aligned(self, a, b):
        return False
    
    def _cosine_similarity(self, a, b):
        """Calcula similitud coseno"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)