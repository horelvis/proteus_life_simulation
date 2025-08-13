#!/usr/bin/env python3
"""
Solver Híbrido: PROTEUS + Reglas
Usa principios topológicos para guiar la selección de reglas
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from scipy.ndimage import label

from arc.arc_solver_python import ARCSolverPython, TransformationType
# Importar los nuevos detectores de forma modular
from arc.detectors.base import ObjectDetectorBase, DetectedObject
from arc.detectors.classical_detector import ClassicalObjectDetector

# Intentar importar el detector YOLO, pero no fallar si no está instalado
try:
    from arc.detectors.yolo_detector import YOLOObjectDetector
    YOLO_AVAILABLE = True
except (ImportError, FileNotFoundError):
    YOLO_AVAILABLE = False


@dataclass
class TopologicalSignature:
    """Firma topológica y de objetos de un patrón."""
    dimension: float
    components: int
    holes: int
    density: float
    symmetry_score: float
    edge_ratio: float
    detected_objects: Set[str] = field(default_factory=set)

    def to_vector(self) -> np.ndarray:
        """Convierte los campos numéricos de la firma a un vector numpy."""
        return np.array([
            self.dimension,
            self.components,
            self.holes,
            self.density,
            self.symmetry_score,
            self.edge_ratio
        ])


class TopologicalAnalyzer:
    """Analiza propiedades topológicas y de objetos de las cuadrículas."""

    def __init__(self, detector: ObjectDetectorBase):
        self.detector = detector

    def analyze(self, grid: np.ndarray) -> TopologicalSignature:
        """Extrae la firma completa de una cuadrícula."""
        # Análisis topológico existente
        dimension = self._compute_fractal_dimension(grid)
        binary = grid > 0
        _, components = label(binary)
        holes = self._count_holes(grid)
        density = np.sum(grid > 0) / grid.size if grid.size > 0 else 0
        symmetry_score = self._compute_symmetry(grid)
        edge_ratio = self._compute_edge_ratio(grid)
        
        # Nuevo: Análisis de objetos usando el detector configurado
        detected_objects_data = self.detector.detect(grid)
        # Usamos un conjunto de etiquetas para la comparación
        object_labels = {obj.label for obj in detected_objects_data}

        return TopologicalSignature(
            dimension=dimension,
            components=components,
            holes=holes,
            density=density,
            symmetry_score=symmetry_score,
            edge_ratio=edge_ratio,
            detected_objects=object_labels
        )
    
    def _compute_fractal_dimension(self, grid: np.ndarray) -> float:
        """Calcula dimensión fractal por box-counting"""
        binary = grid > 0
        
        # Tamaños de caja
        sizes = []
        counts = []
        
        for box_size in [1, 2, 4]:
            if box_size > min(grid.shape):
                continue
                
            count = 0
            for i in range(0, grid.shape[0], box_size):
                for j in range(0, grid.shape[1], box_size):
                    box = binary[i:i+box_size, j:j+box_size]
                    if np.any(box):
                        count += 1
            
            if count > 0:
                sizes.append(box_size)
                counts.append(count)
        
        # Calcular dimensión
        if len(sizes) > 1:
            # Regresión log-log
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            
            # Pendiente de la línea
            if len(sizes) > 1:
                slope = -(log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])
                return max(0, min(2, slope))  # Limitar entre 0 y 2
        
        return 1.0  # Default
    
    def _count_holes(self, grid: np.ndarray) -> int:
        """Cuenta agujeros en el patrón"""
        # Invertir para detectar regiones de 0s
        inverted = grid == 0
        
        # Añadir borde de 1s
        padded = np.pad(inverted, 1, constant_values=0)
        
        # Componentes conectados de 0s
        labeled, num_components = label(padded)
        
        # El componente 1 es el fondo exterior
        # Los demás son agujeros
        return max(0, num_components - 1)
    
    def _compute_symmetry(self, grid: np.ndarray) -> float:
        """Calcula puntuación de simetría (0-1)"""
        scores = []
        
        # Simetría horizontal
        h_sym = np.sum(grid == np.fliplr(grid)) / grid.size
        scores.append(h_sym)
        
        # Simetría vertical
        v_sym = np.sum(grid == np.flipud(grid)) / grid.size
        scores.append(v_sym)
        
        # Simetría diagonal
        if grid.shape[0] == grid.shape[1]:
            d_sym = np.sum(grid == grid.T) / grid.size
            scores.append(d_sym)
        
        return max(scores)
    
    def _compute_edge_ratio(self, grid: np.ndarray) -> float:
        """Calcula ratio de píxeles en los bordes"""
        if grid.size == 0:
            return 0.0
            
        edge_pixels = np.sum(grid[0, :] > 0) + np.sum(grid[-1, :] > 0)
        edge_pixels += np.sum(grid[1:-1, 0] > 0) + np.sum(grid[1:-1, -1] > 0)
        
        total_pixels = np.sum(grid > 0)
        
        if total_pixels == 0:
            return 0.0
            
        return edge_pixels / total_pixels

class HybridProteusARCSolver(ARCSolverPython):
    """
    Solver híbrido que usa un detector de objetos configurable y análisis topológico
    para aprender de los ejemplos y guiar la selección de reglas.
    """
    
    def __init__(self, detector_type: str = 'classical'):
        super().__init__()
        
        # Seleccionar el detector de objetos de forma modular
        if detector_type == 'yolo' and YOLO_AVAILABLE:
            print("👁️ Usando detector de objetos YOLO.")
            detector = YOLOObjectDetector()
        else:
            if detector_type == 'yolo' and not YOLO_AVAILABLE:
                print("⚠️  Detector YOLO no disponible, usando detector clásico como fallback.")
            print("👁️ Usando detector de objetos clásico.")
            detector = ClassicalObjectDetector()
            
        self.analyzer = TopologicalAnalyzer(detector=detector)
        # Almacenará las firmas aprendidas: {rule_type: [TopologicalSignature, ...]}
        self.learned_signatures = {}

    def solve_with_steps(self, 
                        train_examples: List[Dict[str, Any]], 
                        test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Resuelve usando un enfoque de aprendizaje topológico y de objetos:
        1. Aprende las firmas de los ejemplos de entrenamiento.
        2. Encuentra la regla que mejor se ajusta a la firma del caso de prueba.
        """
        steps = []
        self.learned_signatures = {} # Reset for each puzzle

        # Paso 1: Aprender de los ejemplos de entrenamiento
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            rule = super().detect_rule(input_grid, output_grid)
            
            if rule and rule.get('type'):
                rule_type = rule['type']
                input_signature = self.analyzer.analyze(input_grid)

                if rule_type not in self.learned_signatures:
                    self.learned_signatures[rule_type] = []
                self.learned_signatures[rule_type].append(input_signature)

                steps.append({
                    'type': 'learning',
                    'description': f'Ejemplo {i+1}: Aprendida firma para regla "{rule_type}" '
                                 f'con objetos: {input_signature.detected_objects or "ninguno"}'
                })

        if not self.learned_signatures:
            steps.append({
                'type': 'fallback',
                'description': 'No se aprendió ninguna firma. Usando el solver base.'
            })
            return super().solve_with_steps(train_examples, test_input)

        # Paso 2: Analizar la entrada de prueba y encontrar la mejor regla
        test_signature = self.analyzer.analyze(test_input)
        steps.append({
            'type': 'analysis',
            'description': f'Firma del test: objetos={test_signature.detected_objects or "ninguno"}, '
                           f'comp={test_signature.components}, agujeros={test_signature.holes}'
        })

        # Paso 3: Encontrar la mejor regla usando cálculo de distancia vectorizado

        # Aplanar la lista de firmas aprendidas para el cálculo vectorizado
        all_learned_sigs = []
        all_learned_rules = []
        for rule_type, signatures in self.learned_signatures.items():
            for sig in signatures:
                all_learned_sigs.append(sig)
                all_learned_rules.append(rule_type)

        best_rule_type = None
        if all_learned_sigs:
            # Convertir firmas a una matriz numérica
            learned_matrix = np.array([s.to_vector() for s in all_learned_sigs])
            test_vector = test_signature.to_vector()
            
            # Pesos para cada característica topológica
            weights = np.array([1.0, 0.5, 0.8, 1.0, 0.7, 0.4])
            
            # Calcular todas las distancias topológicas a la vez
            topo_distances = np.sum(np.abs(learned_matrix - test_vector) * weights, axis=1)

            # Calcular distancias de Jaccard (esto todavía requiere un bucle)
            jaccard_distances = np.array([
                1.0 - (len(test_signature.detected_objects.intersection(s.detected_objects)) /
                       len(test_signature.detected_objects.union(s.detected_objects)))
                if len(test_signature.detected_objects.union(s.detected_objects)) > 0 else 0.0
                for s in all_learned_sigs
            ])

            # Distancia total combinada
            total_distances = topo_distances + (jaccard_distances * 2.0)

            # Encontrar el índice de la distancia mínima
            best_idx = np.argmin(total_distances)
            min_distance = total_distances[best_idx]
            best_rule_type = all_learned_rules[best_idx]

            steps.append({
                'type': 'selection',
                'description': f'Regla seleccionada: "{best_rule_type}" (distancia combinada: {min_distance:.2f})'
            })

        if best_rule_type:

            # Implementar consenso para los parámetros de la regla.
            # En lugar de tomar los parámetros de la primera regla que coincida,
            # recopilamos todos los parámetros y encontramos los más consistentes.
            candidate_rules = []
            for example in train_examples:
                rule = super().detect_rule(np.array(example['input']), np.array(example['output']))
                if rule and rule.get('type') == best_rule_type:
                    candidate_rules.append(rule)

            # Lógica de consenso
            if candidate_rules:
                # Para 'color_mapping', encontrar el mapeo más común.
                if best_rule_type == TransformationType.COLOR_MAPPING.value:
                    mappings = [tuple(sorted(r['parameters']['mapping'].items())) for r in candidate_rules if 'mapping' in r['parameters']]
                    if mappings:
                        most_common_mapping = max(set(mappings), key=mappings.count)
                        rule_params = {'type': best_rule_type, 'parameters': {'mapping': dict(most_common_mapping)}, 'confidence': 1.0}
                    else:
                        rule_params = candidate_rules[0] # Fallback
                else:
                    # Para otras reglas, usar la de mayor confianza como antes.
                    rule_params = max(candidate_rules, key=lambda r: r.get('confidence', 0))
            else:
                rule_params = None

            if rule_params:
                solution = self.apply_rule(rule_params, test_input)
                steps.append({
                    'type': 'transformation',
                    'description': f'Transformación "{best_rule_type}" aplicada.'
                })
            else:
                solution, original_steps = super().solve_with_steps(train_examples, test_input)
                steps.extend(original_steps)
        else:
            solution, original_steps = super().solve_with_steps(train_examples, test_input)
            steps.extend(original_steps)

        return solution, steps

def test_hybrid_solver():
    """Prueba el solver híbrido"""
    print("🧬 Probando Solver Híbrido PROTEUS + Reglas")
    print("="*60)
    
    solver = HybridProteusARCSolver()
    
    # Ejemplo: Rellenar forma
    train_examples = [
        {
            'input': [[3, 3, 3, 3, 3],
                     [3, 0, 0, 0, 3],
                     [3, 0, 0, 0, 3],
                     [3, 0, 0, 0, 3],
                     [3, 3, 3, 3, 3]],
            'output': [[3, 3, 3, 3, 3],
                      [3, 4, 4, 4, 3],
                      [3, 4, 4, 4, 3],
                      [3, 4, 4, 4, 3],
                      [3, 3, 3, 3, 3]]
        }
    ]
    
    test_input = np.array([
        [3, 3, 3, 3, 3, 3, 3],
        [3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3],
        [3, 3, 3, 3, 3, 3, 3]
    ])
    
    print("\n📊 Resolviendo puzzle de relleno...")
    solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    print("\n🧠 Pasos de razonamiento:")
    for step in steps:
        print(f"   - {step['description']}")
    
    print("\n✅ Solución:")
    for row in solution:
        print("   ", row)
    
    # Verificar
    expected = test_input.copy()
    expected[1:-1, 1:-1] = 4
    
    accuracy = np.sum(solution == expected) / expected.size
    print(f"\n📊 Accuracy: {accuracy*100:.1f}%")
    
    if accuracy == 1.0:
        print("🎉 ¡El análisis topológico mejoró la detección!")

if __name__ == "__main__":
    test_hybrid_solver()