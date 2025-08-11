#!/usr/bin/env python3
"""
Sistema de Reglas Emergentes
Las reglas surgen desde el análisis de píxeles hasta patrones globales
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MicroRule:
    """Regla a nivel de píxel/local"""
    level: str = "pixel"
    pattern: str = ""  # "spread", "preserve", "invert", etc.
    condition: Dict = None  # Condición para aplicar
    action: Dict = None  # Acción a realizar
    confidence: float = 0.0
    support: int = 0  # Cuántas veces se observó

@dataclass
class MesoRule:
    """Regla a nivel de objeto/forma"""
    level: str = "object"
    transformation: str = ""  # "expand", "fill", "rotate", etc.
    source_shape: str = ""
    target_shape: str = ""
    parameters: Dict = None
    confidence: float = 0.0
    micro_rules: List[MicroRule] = None  # Reglas micro que la componen

@dataclass
class MacroRule:
    """Regla a nivel global/patrón"""
    level: str = "pattern"
    rule_type: str = ""  # "replication", "symmetry", "progression"
    global_transform: str = ""
    meso_rules: List[MesoRule] = None  # Reglas meso que la componen
    confidence: float = 0.0

class EmergentRuleSystem:
    """
    Sistema que genera reglas desde la base (píxeles) hasta lo global
    Las reglas emergen del análisis bottom-up
    """
    
    def __init__(self):
        self.micro_rules = []
        self.meso_rules = []
        self.macro_rules = []
        
    def extract_rules_from_examples(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """
        Extrae reglas en todos los niveles desde los ejemplos
        """
        all_micro = []
        all_meso = []
        all_macro = []
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Nivel 0: Extraer micro-reglas (píxel a píxel)
            micro_rules = self._extract_micro_rules(input_grid, output_grid)
            all_micro.extend(micro_rules)
            
            # Nivel 1: Construir meso-reglas desde micro-reglas
            meso_rules = self._build_meso_rules(micro_rules, input_grid, output_grid)
            all_meso.extend(meso_rules)
            
            # Nivel 2: Sintetizar macro-reglas desde meso-reglas
            macro_rules = self._synthesize_macro_rules(meso_rules, input_grid, output_grid)
            all_macro.extend(macro_rules)
        
        # Consolidar y rankear reglas
        self.micro_rules = self._consolidate_micro_rules(all_micro)
        self.meso_rules = self._consolidate_meso_rules(all_meso)
        self.macro_rules = self._consolidate_macro_rules(all_macro)
        
        return {
            'micro_rules': self.micro_rules,
            'meso_rules': self.meso_rules,
            'macro_rules': self.macro_rules,
            'rule_hierarchy': self._build_rule_hierarchy()
        }
    
    def _extract_micro_rules(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[MicroRule]:
        """
        Extrae reglas a nivel de píxel comparando input y output
        """
        micro_rules = []
        h, w = input_grid.shape
        
        for i in range(h):
            for j in range(w):
                in_val = input_grid[i, j]
                out_val = output_grid[i, j]
                
                # Analizar contexto local (3x3)
                context = self._get_local_context(input_grid, i, j)
                
                # Detectar patrón de cambio
                if in_val != out_val:
                    # Cambio de color
                    if in_val == 0 and out_val != 0:
                        # Llenado
                        pattern = "fill"
                        # Ver si es expansión desde vecinos
                        neighbors = self._get_neighbors(input_grid, i, j)
                        if out_val in neighbors:
                            pattern = "spread"
                            condition = {
                                'has_neighbor': out_val,
                                'was_empty': True
                            }
                        else:
                            condition = {
                                'surrounded_by': self._analyze_surrounding(context, in_val)
                            }
                    elif in_val != 0 and out_val == 0:
                        # Borrado
                        pattern = "erase"
                        condition = {'value': in_val}
                    else:
                        # Cambio de color
                        pattern = "recolor"
                        condition = {'from_color': in_val, 'to_color': out_val}
                    
                    action = {'set_value': out_val}
                    
                else:
                    # Sin cambio
                    pattern = "preserve"
                    condition = {'value': in_val}
                    action = {'keep_value': in_val}
                
                # Analizar si el cambio depende de la posición
                position_dependent = self._check_position_dependency(
                    i, j, h, w, in_val, out_val, input_grid, output_grid
                )
                
                if position_dependent:
                    condition['position'] = position_dependent
                
                rule = MicroRule(
                    pattern=pattern,
                    condition=condition,
                    action=action,
                    confidence=1.0,  # Individual siempre 1.0
                    support=1
                )
                
                micro_rules.append(rule)
        
        return micro_rules
    
    def _build_meso_rules(self, micro_rules: List[MicroRule], 
                         input_grid: np.ndarray, output_grid: np.ndarray) -> List[MesoRule]:
        """
        Construye reglas de nivel objeto agrupando micro-reglas
        """
        meso_rules = []
        
        # Agrupar micro-reglas por patrón
        pattern_groups = defaultdict(list)
        for rule in micro_rules:
            pattern_groups[rule.pattern].append(rule)
        
        # Detectar transformaciones de objetos
        from scipy.ndimage import label
        
        # Encontrar objetos en input
        for color in np.unique(input_grid):
            if color == 0:
                continue
            
            input_mask = (input_grid == color)
            input_labeled, num_input = label(input_mask)
            
            # Ver qué pasó con cada objeto
            for obj_id in range(1, num_input + 1):
                obj_mask = (input_labeled == obj_id)
                obj_pixels = np.argwhere(obj_mask)
                
                # Analizar qué micro-reglas afectan este objeto
                obj_micro_rules = []
                for i, j in obj_pixels:
                    for rule in micro_rules:
                        if self._rule_affects_position(rule, i, j):
                            obj_micro_rules.append(rule)
                
                # Determinar transformación del objeto
                transformation = self._determine_object_transformation(
                    obj_pixels, obj_micro_rules, input_grid, output_grid
                )
                
                if transformation:
                    meso_rule = MesoRule(
                        transformation=transformation['type'],
                        source_shape=self._classify_shape(obj_mask),
                        target_shape=transformation.get('target_shape', ''),
                        parameters=transformation.get('parameters', {}),
                        confidence=transformation.get('confidence', 0.5),
                        micro_rules=obj_micro_rules[:5]  # Muestra de micro-reglas
                    )
                    meso_rules.append(meso_rule)
        
        # Detectar reglas de creación de nuevos objetos
        output_only_colors = set(np.unique(output_grid)) - set(np.unique(input_grid))
        for color in output_only_colors:
            # Analizar micro-reglas que crean este color
            creation_rules = [r for r in micro_rules 
                            if r.action.get('set_value') == color]
            
            if creation_rules:
                meso_rule = MesoRule(
                    transformation="create",
                    source_shape="none",
                    target_shape="new_object",
                    parameters={'color': int(color), 'count': len(creation_rules)},
                    confidence=0.7,
                    micro_rules=creation_rules[:5]
                )
                meso_rules.append(meso_rule)
        
        return meso_rules
    
    def _synthesize_macro_rules(self, meso_rules: List[MesoRule], 
                               input_grid: np.ndarray, output_grid: np.ndarray) -> List[MacroRule]:
        """
        Sintetiza reglas globales desde las meso-reglas
        """
        macro_rules = []
        
        # Agrupar meso-reglas por tipo de transformación
        transform_groups = defaultdict(list)
        for rule in meso_rules:
            transform_groups[rule.transformation].append(rule)
        
        # Detectar patrones globales
        
        # 1. Replicación (múltiples objetos con misma transformación)
        for transform, rules in transform_groups.items():
            if len(rules) > 1:
                # Verificar si es replicación uniforme
                if self._is_uniform_transformation(rules):
                    macro_rule = MacroRule(
                        rule_type="uniform_transformation",
                        global_transform=transform,
                        meso_rules=rules,
                        confidence=np.mean([r.confidence for r in rules])
                    )
                    macro_rules.append(macro_rule)
        
        # 2. Expansión/Crecimiento
        if "spread" in [r.pattern for r in sum([m.micro_rules or [] for m in meso_rules], [])]:
            expansion_meso = [r for r in meso_rules if r.transformation in ["expand", "fill"]]
            if expansion_meso:
                macro_rule = MacroRule(
                    rule_type="expansion",
                    global_transform="grow_patterns",
                    meso_rules=expansion_meso,
                    confidence=0.8
                )
                macro_rules.append(macro_rule)
        
        # 3. Transformación posicional
        position_dependent_meso = [r for r in meso_rules 
                                  if any(m.condition.get('position') 
                                        for m in (r.micro_rules or []))]
        if position_dependent_meso:
            macro_rule = MacroRule(
                rule_type="positional",
                global_transform="position_based",
                meso_rules=position_dependent_meso,
                confidence=0.7
            )
            macro_rules.append(macro_rule)
        
        # 4. Detectar si el output es función directa del input
        if self._is_direct_mapping(input_grid, output_grid):
            macro_rule = MacroRule(
                rule_type="direct_mapping",
                global_transform="pixel_wise_function",
                meso_rules=meso_rules,
                confidence=0.9
            )
            macro_rules.append(macro_rule)
        
        return macro_rules
    
    def apply_emergent_rules(self, test_input: np.ndarray) -> np.ndarray:
        """
        Aplica las reglas emergentes al test input
        Empieza desde micro y construye hacia arriba
        """
        result = test_input.copy()
        h, w = test_input.shape
        
        # Primero intentar con macro-reglas (más confiables si existen)
        if self.macro_rules:
            best_macro = max(self.macro_rules, key=lambda r: r.confidence)
            if best_macro.confidence > 0.7:
                result = self._apply_macro_rule(result, best_macro)
                return result
        
        # Si no hay macro-reglas confiables, usar meso-reglas
        if self.meso_rules:
            for meso_rule in sorted(self.meso_rules, key=lambda r: -r.confidence):
                if meso_rule.confidence > 0.6:
                    result = self._apply_meso_rule(result, meso_rule)
        
        # Aplicar micro-reglas para detalles finales
        for i in range(h):
            for j in range(w):
                context = self._get_local_context(result, i, j)
                
                # Encontrar micro-regla aplicable
                applicable_rule = self._find_applicable_micro_rule(
                    result[i, j], context, i, j, h, w
                )
                
                if applicable_rule and applicable_rule.confidence > 0.5:
                    if applicable_rule.action.get('set_value') is not None:
                        result[i, j] = applicable_rule.action['set_value']
        
        return result
    
    # Métodos auxiliares
    
    def _get_local_context(self, grid: np.ndarray, i: int, j: int) -> np.ndarray:
        """Obtiene el contexto 3x3 alrededor de un píxel"""
        h, w = grid.shape
        context = np.zeros((3, 3))
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    context[di+1, dj+1] = grid[ni, nj]
        
        return context
    
    def _get_neighbors(self, grid: np.ndarray, i: int, j: int) -> List[int]:
        """Obtiene los valores de los vecinos"""
        neighbors = []
        h, w = grid.shape
        
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                neighbors.append(int(grid[ni, nj]))
        
        return neighbors
    
    def _analyze_surrounding(self, context: np.ndarray, center_val: int) -> Dict:
        """Analiza el patrón de valores alrededor"""
        unique, counts = np.unique(context, return_counts=True)
        dominant = unique[np.argmax(counts)]
        
        return {
            'dominant_value': int(dominant),
            'surrounded': np.sum(context != center_val) >= 6
        }
    
    def _check_position_dependency(self, i: int, j: int, h: int, w: int,
                                  in_val: int, out_val: int,
                                  input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[str]:
        """Verifica si el cambio depende de la posición"""
        # Centro
        if i == h//2 and j == w//2:
            return "center"
        # Esquinas
        if (i, j) in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            return "corner"
        # Bordes
        if i == 0 or i == h-1 or j == 0 or j == w-1:
            return "edge"
        
        return None
    
    def _rule_affects_position(self, rule: MicroRule, i: int, j: int) -> bool:
        """Verifica si una regla afecta una posición"""
        # Simplificado - en realidad debería verificar la condición completa
        return True
    
    def _determine_object_transformation(self, obj_pixels: np.ndarray, 
                                        micro_rules: List[MicroRule],
                                        input_grid: np.ndarray, 
                                        output_grid: np.ndarray) -> Optional[Dict]:
        """Determina qué transformación sufrió un objeto"""
        if not micro_rules:
            return None
        
        # Contar tipos de patrones en las micro-reglas
        pattern_counts = defaultdict(int)
        for rule in micro_rules:
            pattern_counts[rule.pattern] += 1
        
        # Determinar transformación dominante
        dominant_pattern = max(pattern_counts, key=pattern_counts.get)
        
        transform_map = {
            'spread': {'type': 'expand', 'confidence': 0.8},
            'fill': {'type': 'fill', 'confidence': 0.9},
            'recolor': {'type': 'color_change', 'confidence': 0.85},
            'preserve': {'type': 'identity', 'confidence': 1.0},
            'erase': {'type': 'delete', 'confidence': 0.7}
        }
        
        return transform_map.get(dominant_pattern)
    
    def _classify_shape(self, mask: np.ndarray) -> str:
        """Clasifica la forma de un objeto"""
        # Simplificado
        pixels = np.sum(mask)
        if pixels == 1:
            return "point"
        elif pixels == 2:
            return "pair"
        elif pixels <= 5:
            return "small_shape"
        else:
            return "large_shape"
    
    def _is_uniform_transformation(self, rules: List[MesoRule]) -> bool:
        """Verifica si todas las reglas tienen la misma transformación"""
        if not rules:
            return False
        
        first_transform = rules[0].transformation
        return all(r.transformation == first_transform for r in rules)
    
    def _is_direct_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Verifica si hay mapeo directo píxel a píxel"""
        # Verificar si es una función simple
        if input_grid.shape != output_grid.shape:
            return False
        
        # Ver si hay una relación consistente
        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_val = input_grid[i, j]
                out_val = output_grid[i, j]
                
                if in_val in mapping:
                    if mapping[in_val] != out_val:
                        return False
                else:
                    mapping[in_val] = out_val
        
        return len(mapping) > 0
    
    def _consolidate_micro_rules(self, rules: List[MicroRule]) -> List[MicroRule]:
        """Consolida micro-reglas similares"""
        # Agrupar por patrón y condición
        consolidated = defaultdict(list)
        
        for rule in rules:
            key = (rule.pattern, str(rule.condition))
            consolidated[key].append(rule)
        
        # Crear reglas consolidadas
        final_rules = []
        for (pattern, _), group in consolidated.items():
            if group:
                # Tomar la más común
                consolidated_rule = group[0]
                consolidated_rule.support = len(group)
                consolidated_rule.confidence = len(group) / len(rules)
                final_rules.append(consolidated_rule)
        
        return sorted(final_rules, key=lambda r: -r.confidence)[:20]  # Top 20
    
    def _consolidate_meso_rules(self, rules: List[MesoRule]) -> List[MesoRule]:
        """Consolida meso-reglas similares"""
        # Agrupar por transformación
        consolidated = defaultdict(list)
        
        for rule in rules:
            key = rule.transformation
            consolidated[key].append(rule)
        
        final_rules = []
        for transform, group in consolidated.items():
            if group:
                # Promediar confianza
                avg_confidence = np.mean([r.confidence for r in group])
                best_rule = max(group, key=lambda r: r.confidence)
                best_rule.confidence = avg_confidence
                final_rules.append(best_rule)
        
        return sorted(final_rules, key=lambda r: -r.confidence)
    
    def _consolidate_macro_rules(self, rules: List[MacroRule]) -> List[MacroRule]:
        """Consolida macro-reglas similares"""
        # Agrupar por tipo
        consolidated = defaultdict(list)
        
        for rule in rules:
            consolidated[rule.rule_type].append(rule)
        
        final_rules = []
        for rule_type, group in consolidated.items():
            if group:
                # Tomar la de mayor confianza
                best_rule = max(group, key=lambda r: r.confidence)
                final_rules.append(best_rule)
        
        return sorted(final_rules, key=lambda r: -r.confidence)
    
    def _build_rule_hierarchy(self) -> Dict:
        """Construye la jerarquía de reglas"""
        return {
            'micro_to_meso': self._link_micro_to_meso(),
            'meso_to_macro': self._link_meso_to_macro(),
            'emergence_path': self._trace_emergence_path()
        }
    
    def _link_micro_to_meso(self) -> Dict:
        """Vincula micro-reglas con meso-reglas"""
        links = defaultdict(list)
        for meso in self.meso_rules:
            if meso.micro_rules:
                for micro in meso.micro_rules:
                    links[micro.pattern].append(meso.transformation)
        return dict(links)
    
    def _link_meso_to_macro(self) -> Dict:
        """Vincula meso-reglas con macro-reglas"""
        links = defaultdict(list)
        for macro in self.macro_rules:
            if macro.meso_rules:
                for meso in macro.meso_rules:
                    links[meso.transformation].append(macro.rule_type)
        return dict(links)
    
    def _trace_emergence_path(self) -> List[str]:
        """Traza el camino de emergencia de reglas"""
        paths = []
        
        # Ejemplo de camino: pixel -> objeto -> patrón
        if self.micro_rules and self.meso_rules and self.macro_rules:
            sample_micro = self.micro_rules[0] if self.micro_rules else None
            sample_meso = self.meso_rules[0] if self.meso_rules else None
            sample_macro = self.macro_rules[0] if self.macro_rules else None
            
            if sample_micro and sample_meso and sample_macro:
                path = f"{sample_micro.pattern} -> {sample_meso.transformation} -> {sample_macro.rule_type}"
                paths.append(path)
        
        return paths
    
    def _apply_macro_rule(self, grid: np.ndarray, rule: MacroRule) -> np.ndarray:
        """Aplica una macro-regla"""
        result = grid.copy()
        
        if rule.rule_type == "uniform_transformation":
            # Aplicar transformación uniforme
            for meso in rule.meso_rules or []:
                result = self._apply_meso_rule(result, meso)
        
        elif rule.rule_type == "expansion":
            # Expandir patrones
            result = self._apply_expansion(result)
        
        elif rule.rule_type == "direct_mapping":
            # Mapeo directo
            result = self._apply_direct_mapping(result)
        
        return result
    
    def _apply_meso_rule(self, grid: np.ndarray, rule: MesoRule) -> np.ndarray:
        """Aplica una meso-regla"""
        result = grid.copy()
        
        if rule.transformation == "fill":
            # Rellenar formas
            result = self._fill_shapes(result)
        elif rule.transformation == "expand":
            # Expandir objetos
            result = self._expand_objects(result)
        elif rule.transformation == "color_change":
            # Cambiar colores
            if rule.parameters:
                from_color = rule.parameters.get('from_color')
                to_color = rule.parameters.get('to_color')
                if from_color is not None and to_color is not None:
                    result[result == from_color] = to_color
        
        return result
    
    def _find_applicable_micro_rule(self, value: int, context: np.ndarray, 
                                   i: int, j: int, h: int, w: int) -> Optional[MicroRule]:
        """Encuentra la micro-regla más aplicable"""
        best_rule = None
        best_score = 0
        
        for rule in self.micro_rules:
            # Verificar si la condición se cumple
            if rule.condition:
                score = self._evaluate_condition(rule.condition, value, context, i, j, h, w)
                if score > best_score:
                    best_score = score
                    best_rule = rule
        
        return best_rule if best_score > 0.5 else None
    
    def _evaluate_condition(self, condition: Dict, value: int, context: np.ndarray,
                          i: int, j: int, h: int, w: int) -> float:
        """Evalúa qué tan bien se cumple una condición"""
        score = 0.0
        matches = 0
        total = 0
        
        if 'value' in condition:
            total += 1
            if condition['value'] == value:
                matches += 1
        
        if 'position' in condition:
            total += 1
            pos = condition['position']
            if pos == "center" and i == h//2 and j == w//2:
                matches += 1
            elif pos == "corner" and (i, j) in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
                matches += 1
            elif pos == "edge" and (i == 0 or i == h-1 or j == 0 or j == w-1):
                matches += 1
        
        if 'has_neighbor' in condition:
            total += 1
            neighbors = self._get_neighbors_from_context(context)
            if condition['has_neighbor'] in neighbors:
                matches += 1
        
        if total > 0:
            score = matches / total
        
        return score
    
    def _get_neighbors_from_context(self, context: np.ndarray) -> List[int]:
        """Obtiene vecinos del contexto 3x3"""
        neighbors = []
        # Posiciones de vecinos en contexto 3x3: arriba, abajo, izq, der
        positions = [(0,1), (2,1), (1,0), (1,2)]
        for i, j in positions:
            if i < context.shape[0] and j < context.shape[1]:
                neighbors.append(int(context[i, j]))
        return neighbors
    
    def _apply_expansion(self, grid: np.ndarray) -> np.ndarray:
        """Aplica expansión de patrones"""
        result = grid.copy()
        
        # Expandir cada valor no-cero
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    # Expandir a vecinos vacíos
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            if grid[ni, nj] == 0:
                                result[ni, nj] = grid[i, j]
        
        return result
    
    def _apply_direct_mapping(self, grid: np.ndarray) -> np.ndarray:
        """Aplica mapeo directo basado en reglas aprendidas"""
        # Por ahora, retorna el grid sin cambios
        # En una implementación completa, usaría el mapeo aprendido
        return grid
    
    def _fill_shapes(self, grid: np.ndarray) -> np.ndarray:
        """Rellena formas cerradas"""
        from scipy.ndimage import binary_fill_holes
        result = grid.copy()
        
        for color in np.unique(grid):
            if color == 0:
                continue
            
            mask = (grid == color)
            filled = binary_fill_holes(mask)
            result[filled & (grid == 0)] = color
        
        return result
    
    def _expand_objects(self, grid: np.ndarray) -> np.ndarray:
        """Expande objetos"""
        from scipy.ndimage import binary_dilation
        result = grid.copy()
        
        for color in np.unique(grid):
            if color == 0:
                continue
            
            mask = (grid == color)
            dilated = binary_dilation(mask)
            result[dilated & (grid == 0)] = color
        
        return result