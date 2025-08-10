"""
ARC Solver Python
Implementación del solucionador de puzzles ARC con transparencia total
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from arc_augmentation import ARCAugmentation, AugmentationType

logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """Tipos de transformaciones detectables"""
    COLOR_MAPPING = "color_mapping"
    PATTERN_REPLICATION = "pattern_replication"
    REFLECTION = "reflection"
    ROTATION = "rotation"
    GRAVITY = "gravity"
    COUNTING = "counting"
    FILL_SHAPE = "fill_shape"
    SYMMETRY_DETECTION = "symmetry_detection"
    PATTERN_EXTRACTION = "pattern_extraction"
    LINE_DRAWING = "line_drawing"

@dataclass
class Rule:
    """Regla de transformación detectada"""
    type: TransformationType
    confidence: float
    parameters: Dict[str, Any]
    
@dataclass
class ReasoningStep:
    """Paso de razonamiento"""
    description: str
    grid_before: np.ndarray
    grid_after: np.ndarray
    changes: List[Dict[str, int]]
    
class ARCSolverPython:
    def __init__(self):
        self.rules = []
        self.reasoning_steps = []
        self.confidence = 0.0
        self.augmenter = ARCAugmentation()
        self.use_augmentation = True
        
    def detect_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detecta la regla de transformación entre input y output"""
        
        # Verificar cada tipo de transformación
        # Orden importante: más específicos primero para evitar falsos positivos
        detectors = [
            # Transformaciones de tamaño primero
            (self._detect_pattern_replication, TransformationType.PATTERN_REPLICATION),
            (self._detect_counting, TransformationType.COUNTING),
            (self._detect_symmetry, TransformationType.SYMMETRY_DETECTION),
            (self._detect_pattern_extraction, TransformationType.PATTERN_EXTRACTION),
            # Transformaciones geométricas
            (self._detect_rotation, TransformationType.ROTATION),
            (self._detect_reflection, TransformationType.REFLECTION),
            (self._detect_gravity, TransformationType.GRAVITY),
            # Transformaciones de contenido
            (self._detect_fill_shape, TransformationType.FILL_SHAPE),
            (self._detect_line_drawing, TransformationType.LINE_DRAWING),
            # Color mapping al final (más general)
            (self._detect_color_mapping, TransformationType.COLOR_MAPPING)
        ]
        
        for detector, transform_type in detectors:
            result = detector(input_grid, output_grid)
            if result is not None:
                return {
                    'type': transform_type.value,
                    'confidence': result['confidence'],
                    'parameters': result.get('parameters', {})
                }
                
        return None
        
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Resuelve el puzzle y devuelve los pasos de razonamiento"""
        self.reasoning_steps = []
        
        # Paso 1: Analizar ejemplos de entrenamiento
        self.reasoning_steps.append({
            'type': 'analysis',
            'description': 'Analizando ejemplos de entrenamiento...',
            'details': f'Procesando {len(train_examples)} ejemplos'
        })
        
        # Aplicar aumentación si está habilitada
        all_examples = train_examples.copy()
        if self.use_augmentation and len(train_examples) > 0:
            self.reasoning_steps.append({
                'type': 'augmentation',
                'description': 'Aplicando aumentación de datos para mejorar generalización',
                'details': 'Generando variaciones con traslación y permutación de colores'
            })
            
            for example in train_examples[:2]:  # Limitar aumentación para no sobrecargar
                augmented = self.augmenter.augment_puzzle(
                    example, 
                    [AugmentationType.TRANSLATION, AugmentationType.COLOR_PERMUTATION]
                )
                all_examples.extend(augmented[:2])  # Máximo 2 aumentaciones por ejemplo
        
        # Detectar reglas de cada ejemplo
        detected_rules = []
        for idx, example in enumerate(all_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            rule = self.detect_rule(input_grid, output_grid)
            if rule:
                detected_rules.append(rule)
                if idx < len(train_examples):  # Solo mostrar ejemplos originales
                    self.reasoning_steps.append({
                        'type': 'rule_detection',
                        'description': f'Regla detectada en ejemplo {idx+1}: {rule["type"]}',
                        'rule': rule,
                        'input': input_grid.tolist(),
                        'output': output_grid.tolist()
                    })
                
        # Paso 2: Consenso de reglas
        if not detected_rules:
            return test_input, self.reasoning_steps
            
        # Usar la regla más común
        rule_types = [r['type'] for r in detected_rules]
        most_common_rule = max(set(rule_types), key=rule_types.count)
        
        # Encontrar la regla con mayor confianza
        best_rule = max(
            [r for r in detected_rules if r['type'] == most_common_rule],
            key=lambda r: r['confidence']
        )
        
        self.reasoning_steps.append({
            'type': 'rule_selection',
            'description': f'Regla seleccionada: {best_rule["type"]}',
            'confidence': best_rule['confidence']
        })
        
        # Paso 3: Aplicar regla al test
        solution = self.apply_rule(best_rule, test_input)
        
        # Registrar pasos de aplicación
        self._generate_application_steps(best_rule, test_input, solution)
        
        return solution, self.reasoning_steps
    
    def evaluate_with_augmentation(self, puzzles: List[Dict]) -> Dict[str, Any]:
        """
        Evalúa el impacto de la aumentación en el rendimiento
        
        Args:
            puzzles: Lista de puzzles para evaluar
            
        Returns:
            Métricas de rendimiento con y sin aumentación
        """
        results = {
            'without_augmentation': {'correct': 0, 'total': 0},
            'with_augmentation': {'correct': 0, 'total': 0},
            'improvement': 0.0
        }
        
        # Evaluar sin aumentación
        self.use_augmentation = False
        for puzzle in puzzles:
            if 'trainExamples' in puzzle and 'testExample' in puzzle:
                test_input = np.array(puzzle['testExample']['input'])
                expected_output = np.array(puzzle['testExample']['output'])
                
                solution, _ = self.solve_with_steps(puzzle['trainExamples'], test_input)
                
                if np.array_equal(solution, expected_output):
                    results['without_augmentation']['correct'] += 1
                results['without_augmentation']['total'] += 1
        
        # Evaluar con aumentación
        self.use_augmentation = True
        for puzzle in puzzles:
            if 'trainExamples' in puzzle and 'testExample' in puzzle:
                test_input = np.array(puzzle['testExample']['input'])
                expected_output = np.array(puzzle['testExample']['output'])
                
                solution, _ = self.solve_with_steps(puzzle['trainExamples'], test_input)
                
                if np.array_equal(solution, expected_output):
                    results['with_augmentation']['correct'] += 1
                results['with_augmentation']['total'] += 1
        
        # Calcular mejora
        acc_without = results['without_augmentation']['correct'] / max(1, results['without_augmentation']['total'])
        acc_with = results['with_augmentation']['correct'] / max(1, results['with_augmentation']['total'])
        results['improvement'] = acc_with - acc_without
        results['accuracy_without'] = acc_without
        results['accuracy_with'] = acc_with
        
        return results
        
    def apply_rule(self, rule: Dict[str, Any], input_grid: np.ndarray) -> np.ndarray:
        """Aplica una regla a un grid de entrada"""
        rule_type = TransformationType(rule['type'])
        parameters = rule.get('parameters', {})
        
        if rule_type == TransformationType.COLOR_MAPPING:
            return self._apply_color_mapping(input_grid, parameters['mapping'])
        elif rule_type == TransformationType.PATTERN_REPLICATION:
            return self._apply_pattern_replication(input_grid, parameters.get('factor', 3))
        elif rule_type == TransformationType.REFLECTION:
            return self._apply_reflection(input_grid, parameters.get('axis', 'horizontal'))
        elif rule_type == TransformationType.ROTATION:
            return self._apply_rotation(input_grid, parameters.get('degrees', 90))
        elif rule_type == TransformationType.GRAVITY:
            return self._apply_gravity(input_grid)
        elif rule_type == TransformationType.COUNTING:
            return self._apply_counting(input_grid)
        elif rule_type == TransformationType.FILL_SHAPE:
            return self._apply_fill_shape(input_grid, parameters.get('fill_color', 3))
        elif rule_type == TransformationType.SYMMETRY_DETECTION:
            return self._apply_symmetry_detection(input_grid)
        elif rule_type == TransformationType.PATTERN_EXTRACTION:
            return self._apply_pattern_extraction(input_grid)
        elif rule_type == TransformationType.LINE_DRAWING:
            return self._apply_line_drawing(input_grid, parameters.get('color', 2))
        else:
            return input_grid
            
    # Detectores de reglas
    
    def _detect_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta mapeo de colores"""
        if input_grid.shape != output_grid.shape:
            return None
            
        color_map = {}
        
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]
                
                if in_color != 0:  # Ignorar fondo
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            return None  # Mapeo inconsistente
                    else:
                        color_map[in_color] = out_color
                        
        if not color_map:
            return None
            
        # Verificar que realmente hay cambios
        if all(k == v for k, v in color_map.items()):
            return None
            
        return {
            'confidence': 0.9,
            'parameters': {'mapping': color_map}
        }
        
    def _detect_gravity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta si los objetos caen por gravedad"""
        if input_grid.shape != output_grid.shape:
            return None
            
        # Verificar cada columna
        for j in range(input_grid.shape[1]):
            input_col = input_grid[:, j]
            output_col = output_grid[:, j]
            
            # Obtener elementos no-cero
            input_elements = input_col[input_col != 0]
            output_elements = output_col[output_col != 0]
            
            # Deben tener los mismos elementos
            if not np.array_equal(sorted(input_elements), sorted(output_elements)):
                return None
                
            # Los elementos deben estar al fondo en output
            if len(output_elements) > 0:
                first_non_zero = np.argmax(output_col != 0)
                expected_start = len(output_col) - len(output_elements)
                if first_non_zero != expected_start:
                    return None
                    
        return {
            'confidence': 0.85,
            'parameters': {'direction': 'down'}
        }
        
    def _detect_reflection(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta reflexión horizontal o vertical"""
        if input_grid.shape != output_grid.shape:
            return None
            
        # Verificar reflexión horizontal
        if np.array_equal(output_grid, np.fliplr(input_grid)):
            return {
                'confidence': 0.95,
                'parameters': {'axis': 'horizontal'}
            }
            
        # Verificar reflexión vertical
        if np.array_equal(output_grid, np.flipud(input_grid)):
            return {
                'confidence': 0.95,
                'parameters': {'axis': 'vertical'}
            }
            
        return None
        
    def _detect_counting(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta si el output es el conteo de elementos"""
        if output_grid.shape != (1, 1):
            return None
            
        non_zero_count = np.count_nonzero(input_grid)
        
        if output_grid[0, 0] == non_zero_count:
            return {
                'confidence': 0.9,
                'parameters': {'count_type': 'non_zero'}
            }
            
        return None
        
    def _detect_pattern_replication(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta replicación de patrón (ej: 3x3)"""
        # Verificar factores comunes
        for factor in [2, 3, 4]:
            if (output_grid.shape[0] == input_grid.shape[0] * factor and
                output_grid.shape[1] == input_grid.shape[1] * factor):
                
                # Verificar que cada celda se replica correctamente
                is_replicated = True
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        # Verificar bloque factor x factor
                        block = output_grid[
                            i*factor:(i+1)*factor,
                            j*factor:(j+1)*factor
                        ]
                        if not np.all(block == input_grid[i, j]):
                            is_replicated = False
                            break
                    if not is_replicated:
                        break
                        
                if is_replicated:
                    return {
                        'confidence': 0.9,
                        'parameters': {'factor': factor}
                    }
                    
        return None
        
    def _detect_rotation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta rotación"""
        # Probar rotaciones de 90, 180, 270 grados
        for k in [1, 2, 3]:  # k=1: 90°, k=2: 180°, k=3: 270°
            rotated = np.rot90(input_grid, k)
            if np.array_equal(output_grid, rotated):
                return {
                    'confidence': 0.95,
                    'parameters': {'degrees': k * 90}
                }
                
        return None
        
    def _detect_fill_shape(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta relleno de formas cerradas"""
        if input_grid.shape != output_grid.shape:
            return None
            
        # Buscar diferencias
        diff_mask = (input_grid == 0) & (output_grid != 0)
        
        if not np.any(diff_mask):
            return None
            
        # Verificar que las diferencias están dentro de formas cerradas
        # Simplificación: verificar que están rodeadas
        filled_positions = np.argwhere(diff_mask)
        
        for pos in filled_positions:
            i, j = pos
            # Verificar vecinos (simplificado)
            neighbors = 0
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < input_grid.shape[0] and 0 <= nj < input_grid.shape[1]:
                    if input_grid[ni, nj] != 0:
                        neighbors += 1
                        
            if neighbors < 3:  # No está bien rodeado
                return None
                
        # Detectar color de relleno
        fill_colors = output_grid[diff_mask]
        fill_color = np.bincount(fill_colors).argmax()
        
        return {
            'confidence': 0.8,
            'parameters': {'fill_color': int(fill_color)}
        }
        
    def _detect_symmetry(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta verificación de simetría"""
        if output_grid.shape != (1, 1):
            return None
            
        # Verificar simetría horizontal
        h_symmetric = np.array_equal(input_grid, np.fliplr(input_grid))
        
        # Verificar simetría vertical
        v_symmetric = np.array_equal(input_grid, np.flipud(input_grid))
        
        is_symmetric = h_symmetric or v_symmetric
        expected = 1 if is_symmetric else 0
        
        if output_grid[0, 0] == expected:
            return {
                'confidence': 0.9,
                'parameters': {
                    'horizontal': h_symmetric,
                    'vertical': v_symmetric
                }
            }
            
        return None
        
    def _detect_pattern_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta extracción de patrón (bounding box de elementos no-cero)"""
        # Encontrar bounding box en input
        non_zero = np.argwhere(input_grid != 0)
        
        if len(non_zero) == 0:
            return None
            
        min_i, min_j = non_zero.min(axis=0)
        max_i, max_j = non_zero.max(axis=0)
        
        extracted = input_grid[min_i:max_i+1, min_j:max_j+1]
        
        if np.array_equal(output_grid, extracted):
            return {
                'confidence': 0.85,
                'parameters': {
                    'bounds': (min_i, min_j, max_i, max_j)
                }
            }
            
        return None
        
    def _detect_line_drawing(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta dibujo de líneas entre puntos"""
        input_points = np.argwhere(input_grid != 0)
        output_points = np.argwhere(output_grid != 0)
        
        if len(input_points) != 2:  # Solo funciona con 2 puntos
            return None
            
        if len(output_points) <= len(input_points):
            return None
            
        # Verificar que hay una línea entre los puntos
        # Simplificación: verificar que hay más puntos en output
        line_color = output_grid[output_points[0][0], output_points[0][1]]
        
        return {
            'confidence': 0.7,
            'parameters': {'color': int(line_color)}
        }
        
    # Aplicadores de reglas
    
    def _apply_color_mapping(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Aplica mapeo de colores"""
        result = grid.copy()
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        return result
        
    def _apply_gravity(self, grid: np.ndarray) -> np.ndarray:
        """Aplica gravedad (elementos caen)"""
        result = np.zeros_like(grid)
        
        for j in range(grid.shape[1]):
            col = grid[:, j]
            non_zero = col[col != 0]
            
            # Colocar al fondo
            if len(non_zero) > 0:
                result[-len(non_zero):, j] = non_zero
                
        return result
        
    def _apply_reflection(self, grid: np.ndarray, axis: str) -> np.ndarray:
        """Aplica reflexión"""
        if axis == 'horizontal':
            return np.fliplr(grid)
        else:
            return np.flipud(grid)
            
    def _apply_rotation(self, grid: np.ndarray, degrees: int) -> np.ndarray:
        """Aplica rotación"""
        k = degrees // 90
        return np.rot90(grid, k)
        
    def _apply_counting(self, grid: np.ndarray) -> np.ndarray:
        """Cuenta elementos no-cero"""
        count = np.count_nonzero(grid)
        return np.array([[count]])
        
    def _apply_pattern_replication(self, grid: np.ndarray, factor: int) -> np.ndarray:
        """Replica el patrón"""
        h, w = grid.shape
        result = np.zeros((h * factor, w * factor), dtype=grid.dtype)
        
        for i in range(h):
            for j in range(w):
                result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]
                
        return result
        
    def _apply_fill_shape(self, grid: np.ndarray, fill_color: int) -> np.ndarray:
        """Rellena formas cerradas"""
        result = grid.copy()
        
        # Buscar espacios vacíos rodeados
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                if grid[i, j] == 0:
                    # Contar vecinos no-cero
                    neighbors = 0
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        if grid[i+di, j+dj] != 0:
                            neighbors += 1
                            
                    if neighbors >= 3:  # Está bien rodeado
                        result[i, j] = fill_color
                        
        return result
        
    def _apply_symmetry_detection(self, grid: np.ndarray) -> np.ndarray:
        """Detecta si el grid es simétrico"""
        h_symmetric = np.array_equal(grid, np.fliplr(grid))
        v_symmetric = np.array_equal(grid, np.flipud(grid))
        
        is_symmetric = h_symmetric or v_symmetric
        return np.array([[1 if is_symmetric else 0]])
        
    def _apply_pattern_extraction(self, grid: np.ndarray) -> np.ndarray:
        """Extrae el patrón (bounding box)"""
        non_zero = np.argwhere(grid != 0)
        
        if len(non_zero) == 0:
            return np.array([[0]])
            
        min_i, min_j = non_zero.min(axis=0)
        max_i, max_j = non_zero.max(axis=0)
        
        return grid[min_i:max_i+1, min_j:max_j+1]
        
    def _apply_line_drawing(self, grid: np.ndarray, color: int) -> np.ndarray:
        """Dibuja línea entre puntos"""
        result = grid.copy()
        points = np.argwhere(grid != 0)
        
        if len(points) != 2:
            return result
            
        # Dibujar línea simple (Bresenham simplificado)
        p1, p2 = points[0], points[1]
        
        # Línea diagonal simple
        steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
        if steps == 0:
            return result
            
        for i in range(steps + 1):
            t = i / steps
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            result[x, y] = color
            
        return result
        
    def _generate_application_steps(self, rule: Dict, input_grid: np.ndarray, output_grid: np.ndarray):
        """Genera pasos detallados de aplicación de regla"""
        rule_type = rule['type']
        
        if rule_type == TransformationType.COLOR_MAPPING.value:
            mapping = rule['parameters']['mapping']
            for old_color, new_color in mapping.items():
                positions = np.argwhere(input_grid == old_color)
                if len(positions) > 0:
                    self.reasoning_steps.append({
                        'type': 'transformation',
                        'description': f'Transformando color {old_color} → {new_color}',
                        'positions': positions.tolist(),
                        'count': len(positions)
                    })
                    
        elif rule_type == TransformationType.GRAVITY.value:
            for j in range(input_grid.shape[1]):
                col_elements = np.count_nonzero(input_grid[:, j])
                if col_elements > 0:
                    self.reasoning_steps.append({
                        'type': 'transformation',
                        'description': f'Aplicando gravedad en columna {j+1}',
                        'elements': col_elements
                    })
                    
        # Agregar más tipos según sea necesario
        
    def get_confidence(self) -> float:
        """Obtiene la confianza actual del solver"""
        return self.confidence
        
    def get_detailed_reasoning(self) -> List[Dict]:
        """Obtiene el razonamiento detallado"""
        return self.reasoning_steps