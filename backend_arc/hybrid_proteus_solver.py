#!/usr/bin/env python3
"""
Solver Híbrido: PROTEUS + Reglas
Usa principios topológicos para guiar la selección de reglas
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from scipy.ndimage import label, gaussian_filter

from arc_solver_python import ARCSolverPython, TransformationType

@dataclass
class TopologicalSignature:
    """Firma topológica simplificada de un patrón"""
    dimension: float          # Dimensión fractal
    components: int          # Número de componentes conectados
    holes: int              # Número de agujeros
    density: float          # Densidad de valores no-cero
    symmetry_score: float   # Puntuación de simetría
    edge_ratio: float       # Ratio de píxeles en bordes
    
    def distance(self, other: 'TopologicalSignature') -> float:
        """Distancia topológica entre firmas"""
        return (
            abs(self.dimension - other.dimension) +
            abs(self.components - other.components) * 0.5 +
            abs(self.holes - other.holes) * 0.3 +
            abs(self.density - other.density) +
            abs(self.symmetry_score - other.symmetry_score) * 0.7 +
            abs(self.edge_ratio - other.edge_ratio) * 0.4
        )

class TopologicalAnalyzer:
    """Analiza propiedades topológicas de grillas"""
    
    def analyze(self, grid: np.ndarray) -> TopologicalSignature:
        """Extrae firma topológica de una grilla"""
        # Dimensión fractal por box-counting
        dimension = self._compute_fractal_dimension(grid)
        
        # Componentes conectados
        binary = grid > 0
        labeled, components = label(binary)
        
        # Agujeros (regiones de 0s completamente rodeadas)
        holes = self._count_holes(grid)
        
        # Densidad
        density = np.sum(grid > 0) / grid.size
        
        # Simetría
        symmetry_score = self._compute_symmetry(grid)
        
        # Ratio de bordes
        edge_ratio = self._compute_edge_ratio(grid)
        
        return TopologicalSignature(
            dimension=dimension,
            components=components,
            holes=holes,
            density=density,
            symmetry_score=symmetry_score,
            edge_ratio=edge_ratio
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
    Solver híbrido que usa análisis topológico para guiar la selección de reglas
    """
    
    def __init__(self):
        super().__init__()
        self.analyzer = TopologicalAnalyzer()
        self.rule_signatures = self._initialize_rule_signatures()
        
    def _initialize_rule_signatures(self) -> Dict[TransformationType, List[TopologicalSignature]]:
        """
        Define firmas topológicas típicas para cada tipo de regla
        """
        return {
            TransformationType.COLOR_MAPPING: [
                TopologicalSignature(1.0, 1, 0, 0.5, 0.5, 0.3),  # Uniforme
            ],
            TransformationType.PATTERN_REPLICATION: [
                TopologicalSignature(1.5, 1, 0, 0.2, 0.8, 0.1),  # Patrón pequeño
            ],
            TransformationType.FILL_SHAPE: [
                TopologicalSignature(1.8, 1, 1, 0.6, 0.7, 0.8),  # Forma con agujero
            ],
            TransformationType.LINE_DRAWING: [
                TopologicalSignature(1.2, 2, 0, 0.1, 0.3, 0.0),  # Puntos dispersos
            ],
            TransformationType.SYMMETRY_DETECTION: [
                TopologicalSignature(1.5, 1, 0, 0.5, 0.9, 0.5),  # Alta simetría
            ],
            TransformationType.GRAVITY: [
                TopologicalSignature(1.3, 3, 0, 0.3, 0.2, 0.0),  # Objetos separados
            ],
            TransformationType.ROTATION: [
                TopologicalSignature(1.4, 1, 0, 0.4, 0.6, 0.4),  # Forma rotable
            ],
            TransformationType.REFLECTION: [
                TopologicalSignature(1.4, 1, 0, 0.4, 0.8, 0.4),  # Forma reflejable
            ],
            TransformationType.PATTERN_EXTRACTION: [
                TopologicalSignature(1.6, 3, 0, 0.7, 0.4, 0.3),  # Múltiples patrones
            ],
            TransformationType.COUNTING: [
                TopologicalSignature(1.1, 5, 0, 0.2, 0.1, 0.0),  # Muchos componentes
            ]
        }
    
    def detect_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Detecta regla usando análisis topológico para priorizar candidatos
        """
        # Análisis topológico de la transformación
        input_sig = self.analyzer.analyze(input_grid)
        output_sig = self.analyzer.analyze(output_grid)
        
        # Calcular cambio topológico
        topology_change = {
            'dimension_delta': output_sig.dimension - input_sig.dimension,
            'components_delta': output_sig.components - input_sig.components,
            'holes_delta': output_sig.holes - input_sig.holes,
            'density_delta': output_sig.density - input_sig.density,
            'symmetry_delta': output_sig.symmetry_score - input_sig.symmetry_score
        }
        
        # Priorizar reglas basado en firma topológica
        rule_scores = {}
        
        for rule_type, signatures in self.rule_signatures.items():
            # Calcular distancia a firmas típicas
            min_distance = min(
                input_sig.distance(sig) for sig in signatures
            ) if signatures else float('inf')
            
            # Ajustar por cambios topológicos esperados
            score = 1.0 / (1.0 + min_distance)
            
            # Bonificaciones específicas
            if rule_type == TransformationType.FILL_SHAPE and topology_change['holes_delta'] < 0:
                score *= 2.0  # Reducir agujeros sugiere relleno
            elif rule_type == TransformationType.LINE_DRAWING and topology_change['density_delta'] > 0:
                score *= 1.5  # Aumentar densidad sugiere dibujo
            elif rule_type == TransformationType.PATTERN_REPLICATION and output_grid.size > input_grid.size:
                score *= 2.0  # Crecimiento sugiere replicación
            elif rule_type == TransformationType.SYMMETRY_DETECTION and input_sig.symmetry_score > 0.8:
                score *= 1.5  # Alta simetría inicial
            
            rule_scores[rule_type] = score
        
        # Ordenar reglas por puntuación
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Probar reglas en orden de probabilidad topológica
        for rule_type, score in sorted_rules:
            detector = self._get_detector_for_type(rule_type)
            if detector:
                rule = detector(input_grid, output_grid)
                if rule and rule['confidence'] > 0.7:
                    # Añadir información topológica
                    rule['topology_score'] = score
                    rule['topology_analysis'] = {
                        'input_signature': input_sig,
                        'output_signature': output_sig,
                        'change': topology_change
                    }
                    return rule
        
        # Si ninguna regla específica funciona, usar la detección original
        return super().detect_rule(input_grid, output_grid)
    
    def _get_detector_for_type(self, rule_type: TransformationType):
        """Obtiene el detector para un tipo de regla específico"""
        detector_map = {
            TransformationType.COLOR_MAPPING: self._detect_color_mapping,
            TransformationType.PATTERN_REPLICATION: self._detect_pattern_replication,
            TransformationType.FILL_SHAPE: self._detect_fill_shape,
            TransformationType.LINE_DRAWING: self._detect_line_drawing,
            TransformationType.SYMMETRY_DETECTION: self._detect_symmetry,
            TransformationType.GRAVITY: self._detect_gravity,
            TransformationType.ROTATION: self._detect_rotation,
            TransformationType.REFLECTION: self._detect_reflection,
            TransformationType.PATTERN_EXTRACTION: self._detect_pattern_extraction,
            TransformationType.COUNTING: self._detect_counting
        }
        return detector_map.get(rule_type)
    
    def solve_with_steps(self, 
                        train_examples: List[Dict[str, Any]], 
                        test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Resuelve usando análisis topológico para guiar la búsqueda
        """
        steps = []
        
        # Paso 1: Análisis topológico del input de test
        test_signature = self.analyzer.analyze(test_input)
        steps.append({
            'type': 'topological_analysis',
            'description': f'Análisis topológico: dim={test_signature.dimension:.2f}, '
                          f'componentes={test_signature.components}, '
                          f'agujeros={test_signature.holes}'
        })
        
        # Paso 2: Analizar transformaciones en ejemplos
        transformation_patterns = []
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Detectar regla con guía topológica
            rule = self.detect_rule(input_grid, output_grid)
            
            if rule:
                transformation_patterns.append(rule)
                steps.append({
                    'type': 'rule_detection',
                    'description': f'Ejemplo {i+1}: {rule["type"]} '
                                  f'(confianza: {rule["confidence"]:.2f}, '
                                  f'topología: {rule.get("topology_score", 0):.2f})',
                    'rule': rule
                })
        
        # Paso 3: Consenso topológico
        if transformation_patterns:
            # Usar la regla con mejor puntuación topológica
            best_rule = max(transformation_patterns, 
                           key=lambda r: r.get('topology_score', 0) * r['confidence'])
            
            steps.append({
                'type': 'rule_selection',
                'description': f'Seleccionada: {best_rule["type"]} basado en análisis topológico'
            })
            
            # Aplicar transformación
            solution = self._apply_rule(test_input, best_rule)
            
            steps.append({
                'type': 'transformation',
                'description': f'Transformación aplicada con guía topológica'
            })
        else:
            # Fallback al método original
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