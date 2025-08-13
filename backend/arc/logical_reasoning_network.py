#!/usr/bin/env python3
"""
Red de Razonamiento Lógico para ARC
Arquitectura de 3 capas: Macro (V-JEPA) → Meso → Micro
Sistema puro de razonamiento sin simulación de vida
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from arc.vjepa_observer import VJEPAObserver
from arc.emergent_rule_system import EmergentRuleSystem, MacroRule, MesoRule, MicroRule

logger = logging.getLogger(__name__)

@dataclass
class LogicalInference:
    """Inferencia lógica derivada del análisis"""
    premise: str
    conclusion: str
    confidence: float
    evidence: List[Any]
    level: str  # "macro", "meso", "micro"

class LogicalReasoningNetwork:
    """
    Red de razonamiento lógico puro
    Flujo: Observación Macro (V-JEPA) → Razonamiento Meso → Ejecución Micro
    """
    
    def __init__(self):
        # Capa MACRO: Observación de alto nivel con V-JEPA
        self.macro_observer = VJEPAObserver(embedding_dim=256)
        
        # Sistema de reglas emergentes (3 capas)
        self.rule_system = EmergentRuleSystem()
        
        # Estado de razonamiento
        self.inferences = []
        self.logical_patterns = {}
        
    def reason(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """
        Razona sobre el problema y genera solución
        """
        # 1. MACRO: Observación de alto nivel
        macro_understanding = self._macro_observation(train_examples)
        
        # 2. MESO: Razonamiento sobre objetos y relaciones
        meso_logic = self._meso_reasoning(macro_understanding, train_examples)
        
        # 3. MICRO: Ejecución detallada a nivel de píxeles
        solution = self._micro_execution(meso_logic, test_input, train_examples)
        
        return solution
    
    def _macro_observation(self, train_examples: List[Dict]) -> Dict:
        """
        CAPA MACRO: Observación de alto nivel con V-JEPA
        Entiende el patrón global sin detalles
        """
        logger.info("🔭 MACRO: Observación de alto nivel")
        
        macro_patterns = []
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # V-JEPA observa la transición en espacio latente
            observation = self.macro_observer.observe(input_grid, output_grid)
            
            # Extraer patrón emergente de alto nivel
            pattern = observation['emergent_pattern']
            
            # Crear inferencia lógica de alto nivel
            if pattern['type'] == 'repeated':
                inference = LogicalInference(
                    premise=f"Transición {i} es similar a patrón previo",
                    conclusion="Existe regularidad global",
                    confidence=pattern['confidence'],
                    evidence=[observation['transform_embedding']],
                    level="macro"
                )
            elif pattern['type'] == 'novel':
                inference = LogicalInference(
                    premise=f"Transición {i} introduce nuevo patrón",
                    conclusion="Requiere nuevo razonamiento",
                    confidence=pattern['confidence'],
                    evidence=[observation['transform_embedding']],
                    level="macro"
                )
            else:
                inference = LogicalInference(
                    premise=f"Transición {i} es variante",
                    conclusion="Patrón base con modificación",
                    confidence=pattern['confidence'],
                    evidence=[observation['transform_embedding']],
                    level="macro"
                )
            
            self.inferences.append(inference)
            macro_patterns.append({
                'observation': observation,
                'inference': inference,
                'embedding': observation['transform_embedding']
            })
        
        # Sintetizar comprensión macro
        return {
            'patterns': macro_patterns,
            'global_inference': self._synthesize_macro_inference(macro_patterns),
            'transformation_space': self._analyze_transformation_space(macro_patterns)
        }
    
    def _meso_reasoning(self, macro_understanding: Dict, train_examples: List[Dict]) -> Dict:
        """
        CAPA MESO: Razonamiento sobre objetos y relaciones
        Traduce comprensión macro a lógica de objetos
        """
        logger.info("🔬 MESO: Razonamiento sobre objetos")
        
        # Extraer reglas desde ejemplos
        rules = self.rule_system.extract_rules_from_examples(train_examples)
        
        # Filtrar y priorizar reglas basándose en comprensión macro
        global_inference = macro_understanding['global_inference']
        
        prioritized_meso_rules = []
        
        for meso_rule in rules['meso_rules']:
            # Evaluar consistencia con inferencia macro
            if self._is_consistent_with_macro(meso_rule, global_inference):
                meso_rule.confidence *= 1.2  # Boost por consistencia
                prioritized_meso_rules.append(meso_rule)
        
        # Crear cadena de razonamiento
        reasoning_chain = self._build_reasoning_chain(
            prioritized_meso_rules,
            macro_understanding['transformation_space']
        )
        
        return {
            'meso_rules': prioritized_meso_rules,
            'reasoning_chain': reasoning_chain,
            'object_logic': self._derive_object_logic(prioritized_meso_rules)
        }
    
    def _micro_execution(self, meso_logic: Dict, test_input: np.ndarray, 
                        train_examples: List[Dict]) -> np.ndarray:
        """
        CAPA MICRO: Ejecución detallada a nivel de píxeles
        Implementa la lógica derivada de capas superiores
        """
        logger.info("⚛️ MICRO: Ejecución a nivel de píxeles")
        
        result = test_input.copy()
        
        # Aplicar cadena de razonamiento
        for step in meso_logic['reasoning_chain']:
            if step['action'] == 'transform_objects':
                result = self._apply_object_transformation(
                    result, step['parameters'], meso_logic['object_logic']
                )
            elif step['action'] == 'apply_rules':
                for rule in step['rules']:
                    result = self._apply_micro_rules(result, rule.micro_rules)
        
        # Si no hay cambios, usar predicción directa de V-JEPA
        if np.array_equal(result, test_input):
            result = self.macro_observer.predict_transformation(test_input)
        
        return result
    
    def _synthesize_macro_inference(self, macro_patterns: List[Dict]) -> LogicalInference:
        """
        Sintetiza una inferencia global desde patrones macro
        """
        if not macro_patterns:
            return LogicalInference(
                premise="Sin patrones observados",
                conclusion="Mantener estado actual",
                confidence=0.0,
                evidence=[],
                level="macro"
            )
        
        # Analizar consistencia entre patrones
        pattern_types = [p['inference'].conclusion for p in macro_patterns]
        
        if all("regularidad" in pt for pt in pattern_types):
            return LogicalInference(
                premise="Todos los ejemplos muestran regularidad",
                conclusion="Aplicar transformación consistente",
                confidence=0.9,
                evidence=pattern_types,
                level="macro"
            )
        elif all("nuevo" in pt for pt in pattern_types):
            return LogicalInference(
                premise="Cada ejemplo es único",
                conclusion="Aprender mapeo específico",
                confidence=0.7,
                evidence=pattern_types,
                level="macro"
            )
        else:
            return LogicalInference(
                premise="Patrones mixtos observados",
                conclusion="Combinar estrategias",
                confidence=0.6,
                evidence=pattern_types,
                level="macro"
            )
    
    def _analyze_transformation_space(self, macro_patterns: List[Dict]) -> np.ndarray:
        """
        Analiza el espacio de transformaciones desde embeddings
        """
        if not macro_patterns:
            return np.zeros(256)
        
        embeddings = [p['embedding'] for p in macro_patterns if p['embedding'] is not None]
        
        if embeddings:
            # Centroide del espacio de transformaciones
            return np.mean(embeddings, axis=0)
        
        return np.zeros(256)
    
    def _is_consistent_with_macro(self, meso_rule: MesoRule, 
                                  global_inference: LogicalInference) -> bool:
        """
        Verifica si una regla meso es consistente con inferencia macro
        """
        if "consistente" in global_inference.conclusion:
            # Favorecer reglas con alta confianza
            return meso_rule.confidence > 0.7
        elif "específico" in global_inference.conclusion:
            # Favorecer reglas específicas
            return meso_rule.transformation != ""
        
        return True
    
    def _build_reasoning_chain(self, meso_rules: List[MesoRule], 
                              transformation_space: np.ndarray) -> List[Dict]:
        """
        Construye cadena de razonamiento desde reglas meso
        """
        chain = []
        
        # Ordenar reglas por confianza
        sorted_rules = sorted(meso_rules, key=lambda r: r.confidence, reverse=True)
        
        for rule in sorted_rules[:3]:  # Top 3 reglas
            chain.append({
                'action': 'apply_rules',
                'rules': [rule],
                'confidence': rule.confidence
            })
        
        return chain
    
    def _derive_object_logic(self, meso_rules: List[MesoRule]) -> Dict:
        """
        Deriva lógica de objetos desde reglas meso
        """
        object_logic = {
            'transformations': {},
            'relationships': {},
            'invariants': []
        }
        
        for rule in meso_rules:
            if rule.transformation:
                object_logic['transformations'][rule.source_shape] = rule.transformation
        
        return object_logic
    
    def _apply_object_transformation(self, grid: np.ndarray, 
                                    parameters: Dict, object_logic: Dict) -> np.ndarray:
        """
        Aplica transformación a objetos según lógica derivada
        """
        # Implementación simplificada
        return grid
    
    def _apply_micro_rules(self, grid: np.ndarray, 
                          micro_rules: List[MicroRule]) -> np.ndarray:
        """
        Aplica reglas micro a nivel de píxeles
        """
        if not micro_rules:
            return grid
        
        result = grid.copy()
        
        for rule in micro_rules:
            if rule.pattern == "spread":
                # Expandir valores
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        if grid[i, j] != 0:
                            # Aplicar a vecinos
                            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                    if grid[ni, nj] == 0:
                                        result[ni, nj] = grid[i, j]
        
        return result