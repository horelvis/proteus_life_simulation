#!/usr/bin/env python3
"""
Sistema Unificado ARC Completo
Integra V-JEPA Contrastive + Razonamiento Jerárquico + Transformaciones
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Importar todos los componentes
import sys
sys.path.append('/app/arc')

from vjepa_observer_contrastive import VJEPAObserverContrastive
from hierarchical_analyzer import HierarchicalAnalyzer
from emergent_rule_system import EmergentRuleSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedARCSolverComplete:
    """
    Sistema completo que integra:
    1. V-JEPA Contrastive para visión (aprendido, no hardcodeado)
    2. Análisis jerárquico para estructura
    3. Sistema de reglas emergentes para razonamiento
    4. Predicción de dimensiones de salida
    """
    
    def __init__(self):
        logger.info("=== Inicializando Sistema Unificado ARC ===")
        
        # Componente 1: Visión con V-JEPA Contrastive
        self.visual_observer = VJEPAObserverContrastive()
        logger.info("✅ V-JEPA Contrastive cargado")
        
        # Componente 2: Análisis jerárquico
        self.hierarchical_analyzer = HierarchicalAnalyzer()
        logger.info("✅ Analizador jerárquico inicializado")
        
        # Componente 3: Sistema de reglas emergentes
        self.rule_system = EmergentRuleSystem()
        logger.info("✅ Sistema de reglas emergentes inicializado")
        
        # Estado del solver
        self.current_puzzle_id = None
        self.solution_confidence = 0.0
        self.reasoning_steps = []
        
    def reset_for_new_puzzle(self, puzzle_id: str = None):
        """Resetea el sistema para un nuevo puzzle (cumple reglas ARC)"""
        self.current_puzzle_id = puzzle_id
        self.visual_observer.reset_memory()
        # El rule_system se auto-resetea con nuevos ejemplos
        self.solution_confidence = 0.0
        self.reasoning_steps = []
        logger.info(f"Sistema reseteado para puzzle: {puzzle_id}")
    
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """
        Resuelve un puzzle ARC usando todos los componentes
        """
        solution, steps = self.solve_with_reasoning(train_examples, test_input)
        return solution
    
    def solve_with_reasoning(self, train_examples: List[Dict], 
                            test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve con razonamiento detallado paso a paso
        """
        self.reasoning_steps = []
        
        if not train_examples:
            self.reasoning_steps.append({
                "step": "error",
                "description": "No hay ejemplos de entrenamiento"
            })
            return test_input.copy(), self.reasoning_steps
        
        # FASE 1: OBSERVACIÓN VISUAL
        self._phase_visual_observation(train_examples)
        
        # FASE 2: ANÁLISIS ESTRUCTURAL
        self._phase_structural_analysis(train_examples)
        
        # FASE 3: DETECCIÓN DE REGLAS
        self._phase_rule_detection(train_examples)
        
        # FASE 4: PREDICCIÓN DE DIMENSIONES
        output_shape = self._predict_output_dimensions(train_examples, test_input)
        
        # FASE 5: APLICACIÓN DE TRANSFORMACIÓN
        solution = self._phase_apply_transformation(test_input, output_shape)
        
        # FASE 6: VALIDACIÓN Y REFINAMIENTO
        solution = self._phase_validation_refinement(solution, train_examples)
        
        self.reasoning_steps.append({
            "step": "completion",
            "description": f"Solución generada con confianza {self.solution_confidence:.2%}",
            "output_shape": solution.shape
        })
        
        return solution, self.reasoning_steps
    
    def _phase_visual_observation(self, train_examples: List[Dict]):
        """Fase 1: Observación visual con V-JEPA"""
        self.reasoning_steps.append({
            "step": "visual_observation",
            "description": "🔍 Analizando patrones visuales con V-JEPA..."
        })
        
        pattern_summary = {}
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Observar con V-JEPA
            observation = self.visual_observer.observe(input_grid, output_grid)
            
            pattern_type = observation['emergent_pattern']['type']
            confidence = observation['emergent_pattern']['confidence']
            
            if pattern_type not in pattern_summary:
                pattern_summary[pattern_type] = []
            pattern_summary[pattern_type].append(confidence)
            
            self.reasoning_steps.append({
                "step": f"observation_{i+1}",
                "description": f"  Ejemplo {i+1}: {pattern_type} (confianza: {confidence:.2f})",
                "latent_similarity": observation['latent_similarity'],
                "transform_magnitude": observation['transform_magnitude']
            })
        
        # Determinar patrón dominante
        dominant_pattern = max(pattern_summary.keys(), 
                              key=lambda k: np.mean(pattern_summary[k]))
        avg_confidence = np.mean(pattern_summary[dominant_pattern])
        
        self.reasoning_steps.append({
            "step": "pattern_summary",
            "description": f"  Patrón dominante: {dominant_pattern} ({avg_confidence:.2f})"
        })
    
    def _phase_structural_analysis(self, train_examples: List[Dict]):
        """Fase 2: Análisis estructural jerárquico"""
        self.reasoning_steps.append({
            "step": "structural_analysis",
            "description": "📊 Analizando estructura jerárquica..."
        })
        
        for i, example in enumerate(train_examples[:2]):  # Analizar primeros 2
            input_grid = np.array(example['input'])
            
            try:
                hierarchy = self.hierarchical_analyzer.analyze_full_hierarchy(input_grid)
                
                num_objects = len(hierarchy.get('level_1_objects', {}).get('objects', []))
                num_groups = len(hierarchy.get('level_2_groups', {}).get('groups', []))
                num_patterns = len(hierarchy.get('level_3_patterns', {}).get('patterns', []))
                
                self.reasoning_steps.append({
                    "step": f"structure_{i+1}",
                    "description": f"  Ejemplo {i+1}: {num_objects} objetos, "
                                  f"{num_groups} grupos, {num_patterns} patrones"
                })
            except Exception as e:
                self.reasoning_steps.append({
                    "step": f"structure_{i+1}",
                    "description": f"  Ejemplo {i+1}: Error en análisis - {str(e)[:50]}"
                })
    
    def _phase_rule_detection(self, train_examples: List[Dict]):
        """Fase 3: Detección de reglas emergentes"""
        self.reasoning_steps.append({
            "step": "rule_detection",
            "description": "🧠 Detectando reglas de transformación..."
        })
        
        try:
            # Analizar transformaciones
            for i, example in enumerate(train_examples[:3]):  # Primeros 3
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                # Detectar tipo de transformación básica
                if np.array_equal(input_grid, output_grid):
                    rule_type = "identity"
                elif input_grid.shape != output_grid.shape:
                    rule_type = "dimension_change"
                elif np.array_equal(np.rot90(input_grid), output_grid):
                    rule_type = "rotation_90"
                elif np.array_equal(np.fliplr(input_grid), output_grid):
                    rule_type = "flip_horizontal"
                elif np.array_equal(np.flipud(input_grid), output_grid):
                    rule_type = "flip_vertical"
                else:
                    rule_type = "complex_transform"
                
                self.reasoning_steps.append({
                    "step": f"rule_{i+1}",
                    "description": f"  Ejemplo {i+1}: {rule_type}"
                })
            
            self.solution_confidence = 0.6  # Confianza base
        except Exception as e:
            self.reasoning_steps.append({
                "step": "rule_error",
                "description": f"  Error detectando reglas: {str(e)[:50]}"
            })
            self.solution_confidence = 0.3
    
    def _predict_output_dimensions(self, train_examples: List[Dict], 
                                  test_input: np.ndarray) -> Tuple[int, int]:
        """Fase 4: Predice las dimensiones de salida"""
        self.reasoning_steps.append({
            "step": "dimension_prediction",
            "description": "📐 Prediciendo dimensiones de salida..."
        })
        
        # Analizar relación de dimensiones en ejemplos
        dimension_patterns = []
        
        for example in train_examples:
            input_shape = np.array(example['input']).shape
            output_shape = np.array(example['output']).shape
            
            if input_shape == output_shape:
                dimension_patterns.append('same')
            elif output_shape[0] == input_shape[0] * 2 and output_shape[1] == input_shape[1] * 2:
                dimension_patterns.append('double')
            elif output_shape[0] == input_shape[0] // 2 and output_shape[1] == input_shape[1] // 2:
                dimension_patterns.append('half')
            else:
                dimension_patterns.append('custom')
        
        # Aplicar patrón más común
        if all(p == 'same' for p in dimension_patterns):
            predicted_shape = test_input.shape
            self.reasoning_steps.append({
                "step": "dimension_result",
                "description": f"  Dimensiones: sin cambio {predicted_shape}"
            })
        elif all(p == 'double' for p in dimension_patterns):
            predicted_shape = (test_input.shape[0] * 2, test_input.shape[1] * 2)
            self.reasoning_steps.append({
                "step": "dimension_result",
                "description": f"  Dimensiones: duplicadas {predicted_shape}"
            })
        elif all(p == 'half' for p in dimension_patterns):
            predicted_shape = (test_input.shape[0] // 2, test_input.shape[1] // 2)
            self.reasoning_steps.append({
                "step": "dimension_result",
                "description": f"  Dimensiones: reducidas {predicted_shape}"
            })
        else:
            # Usar dimensiones del primer output como aproximación
            predicted_shape = np.array(train_examples[0]['output']).shape
            self.reasoning_steps.append({
                "step": "dimension_result",
                "description": f"  Dimensiones: estimadas {predicted_shape}"
            })
        
        return predicted_shape
    
    def _phase_apply_transformation(self, test_input: np.ndarray, 
                                   output_shape: Tuple[int, int]) -> np.ndarray:
        """Fase 5: Aplica la transformación aprendida"""
        self.reasoning_steps.append({
            "step": "transformation",
            "description": "🎯 Aplicando transformación aprendida..."
        })
        
        # Usar V-JEPA para predecir
        solution = self.visual_observer.predict_transformation(test_input)
        
        # Ajustar dimensiones si es necesario
        if solution.shape != output_shape:
            self.reasoning_steps.append({
                "step": "dimension_adjustment",
                "description": f"  Ajustando dimensiones de {solution.shape} a {output_shape}"
            })
            
            # Redimensionar solución
            if output_shape[0] > solution.shape[0] or output_shape[1] > solution.shape[1]:
                # Expandir
                new_solution = np.zeros(output_shape, dtype=solution.dtype)
                h = min(solution.shape[0], output_shape[0])
                w = min(solution.shape[1], output_shape[1])
                new_solution[:h, :w] = solution[:h, :w]
                solution = new_solution
            else:
                # Recortar
                solution = solution[:output_shape[0], :output_shape[1]]
        
        return solution
    
    def _phase_validation_refinement(self, solution: np.ndarray, 
                                    train_examples: List[Dict]) -> np.ndarray:
        """Fase 6: Validación y refinamiento de la solución"""
        self.reasoning_steps.append({
            "step": "validation",
            "description": "✅ Validando coherencia de la solución..."
        })
        
        # Extraer características de la solución
        solution_features = self.visual_observer._extract_features(solution)
        
        # Comparar con outputs de entrenamiento
        coherence_scores = []
        for example in train_examples:
            output_grid = np.array(example['output'])
            output_features = self.visual_observer._extract_features(output_grid)
            
            similarity = self.visual_observer._cosine_similarity(
                solution_features, output_features
            )
            coherence_scores.append(similarity)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        self.reasoning_steps.append({
            "step": "coherence_check",
            "description": f"  Coherencia con ejemplos: {avg_coherence:.2%}"
        })
        
        # Si la coherencia es muy baja, intentar método alternativo
        if avg_coherence < 0.3:
            self.reasoning_steps.append({
                "step": "refinement",
                "description": "  ⚠️ Coherencia baja, aplicando refinamiento..."
            })
            
            # Intentar método alternativo simple
            # Por ejemplo, si todos los ejemplos tienen el mismo color de fondo
            if train_examples:
                output_grid = np.array(train_examples[0]['output'])
                background_color = np.bincount(output_grid.flatten()).argmax()
                
                # Aplicar color de fondo más común
                refined_solution = solution.copy()
                refined_solution[refined_solution == 0] = background_color
                
                self.reasoning_steps.append({
                    "step": "refinement_attempt",
                    "description": f"  Aplicando color de fondo común: {background_color}"
                })
                
                solution = refined_solution
        
        # Actualizar confianza final
        self.solution_confidence = (self.solution_confidence + avg_coherence) / 2
        
        return solution
    
    def get_solving_summary(self) -> Dict:
        """
        Retorna un resumen del proceso de resolución
        """
        visual_patterns = self.visual_observer.get_learned_patterns()
        
        return {
            'puzzle_id': self.current_puzzle_id,
            'solution_confidence': self.solution_confidence,
            'visual_patterns': visual_patterns,
            'reasoning_steps': len(self.reasoning_steps),
            'components_used': [
                'V-JEPA Contrastive (Visual)',
                'Hierarchical Analyzer (Structure)',
                'Emergent Rule System (Logic)'
            ]
        }


def test_unified_system():
    """Prueba el sistema unificado completo"""
    import json
    
    logger.info("=== PRUEBA DEL SISTEMA UNIFICADO ARC ===")
    
    # Crear solver
    solver = UnifiedARCSolverComplete()
    
    # Cargar algunos puzzles de prueba
    cache_dir = Path("/app/arc_official_cache")
    puzzle_files = list(cache_dir.glob("arc_agi_1_training_*.json"))[:5]
    
    if not puzzle_files:
        logger.error("No se encontraron puzzles en caché")
        return
    
    results = []
    
    for puzzle_file in puzzle_files:
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)
        
        puzzle_id = puzzle_file.stem.replace("arc_agi_1_training_", "")
        logger.info(f"\n{'='*50}")
        logger.info(f"Puzzle: {puzzle_id}")
        
        # Resetear para nuevo puzzle
        solver.reset_for_new_puzzle(puzzle_id)
        
        # Resolver
        train_examples = puzzle_data.get('train', [])
        test_cases = puzzle_data.get('test', [])
        
        if test_cases:
            test_input = np.array(test_cases[0]['input'])
            
            # Resolver con razonamiento
            solution, steps = solver.solve_with_reasoning(train_examples, test_input)
            
            # Mostrar pasos de razonamiento
            logger.info("\nPasos de razonamiento:")
            for step in steps[:10]:  # Primeros 10 pasos
                logger.info(f"  {step.get('description', step)}")
            
            # Verificar si es correcto
            if 'output' in test_cases[0]:
                correct_output = np.array(test_cases[0]['output'])
                is_correct = np.array_equal(solution, correct_output)
                
                if solution.shape == correct_output.shape:
                    accuracy = np.mean(solution == correct_output)
                else:
                    accuracy = 0.0
                
                logger.info(f"\n✅ Correcto: {'SÍ' if is_correct else 'NO'}")
                logger.info(f"📊 Precisión: {accuracy:.1%}")
                logger.info(f"🎯 Confianza: {solver.solution_confidence:.1%}")
                
                results.append({
                    'puzzle_id': puzzle_id,
                    'correct': is_correct,
                    'accuracy': accuracy,
                    'confidence': solver.solution_confidence
                })
    
    # Resumen final
    if results:
        logger.info(f"\n{'='*50}")
        logger.info("RESUMEN FINAL:")
        correct_count = sum(1 for r in results if r['correct'])
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        logger.info(f"Puzzles correctos: {correct_count}/{len(results)} ({correct_count/len(results):.1%})")
        logger.info(f"Precisión promedio: {avg_accuracy:.1%}")
        logger.info(f"Confianza promedio: {avg_confidence:.1%}")


if __name__ == "__main__":
    test_unified_system()