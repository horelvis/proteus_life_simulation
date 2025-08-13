#!/usr/bin/env python3
"""
Sistema Integrado de Razonamiento con Memoria y Atención
Combina todos los módulos disponibles para aprendizaje acumulativo
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Módulos existentes
from arc.vjepa_observer import VJEPAObserver
from arc.emergent_rule_system import EmergentRuleSystem
from arc.enhanced_memory import EnhancedMemory, PuzzleSignature, SolutionRule
from arc.hierarchical_attention_solver import HierarchicalAttentionSolver
from arc.bidirectional_attention import BidirectionalAttention
from arc.hierarchical_analyzer import HierarchicalAnalyzer

logger = logging.getLogger(__name__)

@dataclass 
class IntegratedInference:
    """Inferencia con memoria y contexto"""
    level: str
    premise: str
    conclusion: str
    confidence: float
    evidence: List[Any]
    similar_puzzles: List[str]  # IDs de puzzles similares
    reused_rules: List[str]     # Reglas reutilizadas de memoria

class IntegratedReasoningSystem:
    """
    Sistema unificado que integra:
    1. Memoria persistente (EnhancedMemory)
    2. Atención jerárquica y bidireccional
    3. V-JEPA para observación
    4. Análisis jerárquico completo
    5. Aprendizaje acumulativo
    """
    
    def __init__(self, memory_path: str = "./arc_memory.db"):
        # 1. MEMORIA PERSISTENTE
        self.memory = EnhancedMemory(db_path=memory_path)
        
        # 2. OBSERVACIÓN (V-JEPA)
        self.observer = VJEPAObserver(embedding_dim=256)
        
        # 3. ATENCIÓN JERÁRQUICA
        self.attention_solver = HierarchicalAttentionSolver()
        self.bidirectional = BidirectionalAttention(
            input_dim=256, 
            hidden_dim=512,
            num_heads=8
        )
        
        # 4. ANÁLISIS JERÁRQUICO
        self.analyzer = HierarchicalAnalyzer()
        
        # 5. SISTEMA DE REGLAS
        self.rule_system = EmergentRuleSystem()
        
        # Estado del sistema
        self.current_puzzle_signature = None
        self.inferences = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Sistema integrado inicializado con GPU: {self.device}")
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """
        Resuelve puzzle usando memoria y atención completa
        """
        # 1. GENERAR FIRMA DEL PUZZLE
        self.current_puzzle_signature = self._generate_puzzle_signature(train_examples)
        
        # 2. BUSCAR PUZZLES SIMILARES EN MEMORIA
        similar_puzzles = self.memory.find_similar_puzzles(
            self.current_puzzle_signature,
            top_k=5
        )
        
        logger.info(f"📚 Encontrados {len(similar_puzzles)} puzzles similares en memoria")
        
        # 3. ANÁLISIS JERÁRQUICO COMPLETO
        hierarchical_analysis = self._hierarchical_analysis(train_examples)
        
        # 4. OBSERVACIÓN MACRO CON V-JEPA
        macro_understanding = self._macro_observation_with_memory(
            train_examples, 
            similar_puzzles
        )
        
        # 5. RAZONAMIENTO MESO CON ATENCIÓN
        meso_logic = self._meso_reasoning_with_attention(
            macro_understanding,
            hierarchical_analysis,
            train_examples
        )
        
        # 6. EJECUCIÓN MICRO CON REFINAMIENTO
        solution = self._micro_execution_with_refinement(
            meso_logic,
            test_input,
            train_examples
        )
        
        # 7. GUARDAR EN MEMORIA SI ES EXITOSO
        self._save_to_memory(train_examples, test_input, solution)
        
        return solution
    
    def _generate_puzzle_signature(self, train_examples: List[Dict]) -> PuzzleSignature:
        """Genera firma única del puzzle para búsqueda en memoria"""
        all_inputs = [np.array(ex['input']) for ex in train_examples]
        all_outputs = [np.array(ex['output']) for ex in train_examples]
        
        # Analizar características
        input_shapes = [arr.shape for arr in all_inputs]
        output_shapes = [arr.shape for arr in all_outputs]
        
        has_size_changes = any(i != o for i, o in zip(input_shapes, output_shapes))
        
        # Colores únicos
        input_colors = set()
        output_colors = set()
        for inp, out in zip(all_inputs, all_outputs):
            input_colors.update(np.unique(inp).tolist())
            output_colors.update(np.unique(out).tolist())
        
        has_color_changes = output_colors != input_colors
        
        # Distribución de colores promedio
        color_dist = {}
        for arr in all_inputs:
            unique, counts = np.unique(arr, return_counts=True)
            for val, count in zip(unique, counts):
                color_dist[int(val)] = color_dist.get(int(val), 0) + count
        
        total = sum(color_dist.values())
        color_dist = {k: v/total for k, v in color_dist.items()}
        
        # Detectar patrones básicos
        has_patterns = self._detect_basic_patterns(all_inputs, all_outputs)
        has_symmetry = self._detect_symmetry(all_inputs, all_outputs)
        
        # Complejidad
        complexity = self._calculate_complexity(train_examples)
        
        return PuzzleSignature(
            size=input_shapes[0] if input_shapes else (0, 0),
            num_colors=len(input_colors),
            color_distribution=color_dist,
            has_size_changes=has_size_changes,
            has_color_changes=has_color_changes,
            has_patterns=has_patterns,
            has_symmetry=has_symmetry,
            complexity_score=complexity
        )
    
    def _hierarchical_analysis(self, train_examples: List[Dict]) -> Dict:
        """Análisis jerárquico completo pixel→objeto→relación→patrón"""
        all_analysis = []
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Análisis completo con el HierarchicalAnalyzer
            analysis = self.analyzer.analyze(input_grid, output_grid)
            all_analysis.append(analysis)
        
        # Consolidar análisis
        return {
            'pixel_level': self._consolidate_pixel_analysis(all_analysis),
            'object_level': self._consolidate_object_analysis(all_analysis),
            'relation_level': self._consolidate_relation_analysis(all_analysis),
            'pattern_level': self._consolidate_pattern_analysis(all_analysis),
            'all_analysis': all_analysis
        }
    
    def _macro_observation_with_memory(self, train_examples: List[Dict], 
                                      similar_puzzles: List[Dict]) -> Dict:
        """
        Observación MACRO con V-JEPA + contexto de memoria
        """
        observations = []
        
        # Observar ejemplos actuales
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            obs = self.observer.observe(input_grid, output_grid)
            observations.append(obs)
        
        # Incorporar conocimiento de puzzles similares
        memory_context = []
        for similar in similar_puzzles:
            if 'solution_rules' in similar:
                memory_context.extend(similar['solution_rules'])
        
        # Crear embedding global con contexto
        global_embedding = self._create_contextual_embedding(
            observations, 
            memory_context
        )
        
        # Inferencia global mejorada
        if similar_puzzles:
            inference = IntegratedInference(
                level="macro",
                premise="Patrones similares encontrados en memoria",
                conclusion=f"Reutilizar estrategias de {len(similar_puzzles)} puzzles similares",
                confidence=0.8,
                evidence=observations,
                similar_puzzles=[p.get('id', '') for p in similar_puzzles],
                reused_rules=[r.get('rule_id', '') for r in memory_context[:3]]
            )
        else:
            inference = IntegratedInference(
                level="macro",
                premise="Nuevo tipo de puzzle",
                conclusion="Explorar nuevas estrategias",
                confidence=0.5,
                evidence=observations,
                similar_puzzles=[],
                reused_rules=[]
            )
        
        self.inferences.append(inference)
        
        return {
            'observations': observations,
            'global_embedding': global_embedding,
            'memory_context': memory_context,
            'inference': inference
        }
    
    def _meso_reasoning_with_attention(self, macro_understanding: Dict,
                                      hierarchical_analysis: Dict,
                                      train_examples: List[Dict]) -> Dict:
        """
        Razonamiento MESO con atención bidireccional
        """
        # Extraer reglas emergentes
        rules = self.rule_system.extract_rules_from_examples(train_examples)
        
        # Aplicar atención jerárquica
        attention_features = []
        for example in train_examples:
            # Usar HierarchicalAttentionSolver para análisis con atención
            features = self.attention_solver._extract_hierarchical_features(
                np.array(example['input']),
                np.array(example['output'])
            )
            attention_features.append(features)
        
        # Atención bidireccional para refinamiento
        if attention_features:
            # Convertir a tensores para GPU
            feature_tensor = torch.FloatTensor(np.array(attention_features)).to(self.device)
            
            # Aplicar atención bidireccional
            refined_features, attention_weights = self.bidirectional(
                feature_tensor,
                feature_tensor,
                feature_tensor
            )
            
            # Analizar pesos de atención para identificar patrones importantes
            important_patterns = self._identify_important_patterns(
                attention_weights.cpu().numpy()
            )
        else:
            important_patterns = []
        
        # Combinar con análisis jerárquico
        reasoning_chain = self._build_reasoning_chain(
            rules,
            hierarchical_analysis,
            important_patterns
        )
        
        return {
            'rules': rules,
            'attention_features': attention_features,
            'important_patterns': important_patterns,
            'reasoning_chain': reasoning_chain,
            'hierarchical_context': hierarchical_analysis
        }
    
    def _micro_execution_with_refinement(self, meso_logic: Dict,
                                        test_input: np.ndarray,
                                        train_examples: List[Dict]) -> np.ndarray:
        """
        Ejecución MICRO con múltiples pasadas de refinamiento
        """
        # Primera pasada: aplicar reglas básicas
        solution = self.rule_system.apply_emergent_rules(test_input)
        
        # Segunda pasada: usar atención para refinamiento
        if self.memory.has_successful_rules(self.current_puzzle_signature):
            # Aplicar reglas exitosas de memoria
            successful_rules = self.memory.get_successful_rules(
                self.current_puzzle_signature
            )
            
            for rule in successful_rules:
                solution = self._apply_memory_rule(solution, rule)
        
        # Tercera pasada: ajustar basado en patrones importantes
        for pattern in meso_logic.get('important_patterns', []):
            solution = self._apply_pattern_refinement(solution, pattern)
        
        # Verificar coherencia con ejemplos de entrenamiento
        solution = self._ensure_coherence(solution, train_examples)
        
        return solution
    
    def _save_to_memory(self, train_examples: List[Dict], 
                       test_input: np.ndarray, 
                       solution: np.ndarray):
        """Guarda solución exitosa en memoria para futuro uso"""
        # Crear regla de solución
        rule = SolutionRule(
            rule_id="",
            rule_type="integrated_solution",
            rule_data={
                'train_count': len(train_examples),
                'test_shape': test_input.shape,
                'solution_shape': solution.shape,
                'transformations': self._extract_transformations(train_examples)
            },
            fitness=self._calculate_solution_fitness(solution, train_examples),
            agent_id=0,
            puzzle_signature=self.current_puzzle_signature,
            transformation_chain=[inf.conclusion for inf in self.inferences]
        )
        
        # Guardar en memoria
        self.memory.add_solution(self.current_puzzle_signature, rule)
        logger.info("💾 Solución guardada en memoria para futuro uso")
    
    # Métodos auxiliares
    
    def _detect_basic_patterns(self, inputs: List[np.ndarray], 
                              outputs: List[np.ndarray]) -> bool:
        """Detecta si hay patrones básicos repetitivos"""
        # Simplificado - verificar si hay transformaciones consistentes
        if len(inputs) < 2:
            return False
        
        # Comparar transformaciones
        transforms = []
        for inp, out in zip(inputs, outputs):
            if inp.shape == out.shape:
                diff = out - inp
                transforms.append(diff.flatten())
        
        if len(transforms) >= 2:
            # Ver si las transformaciones son similares
            similarity = np.corrcoef(transforms[0], transforms[1])[0, 1]
            return abs(similarity) > 0.5
        
        return False
    
    def _detect_symmetry(self, inputs: List[np.ndarray], 
                        outputs: List[np.ndarray]) -> bool:
        """Detecta simetría en las transformaciones"""
        for out in outputs:
            # Verificar simetría horizontal
            if np.array_equal(out, np.fliplr(out)):
                return True
            # Verificar simetría vertical
            if np.array_equal(out, np.flipud(out)):
                return True
        return False
    
    def _calculate_complexity(self, train_examples: List[Dict]) -> float:
        """Calcula complejidad del puzzle"""
        complexity = 0.0
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            # Factor 1: Cambio de tamaño
            if inp.shape != out.shape:
                complexity += 2.0
            
            # Factor 2: Nuevos colores
            if set(np.unique(out)) != set(np.unique(inp)):
                complexity += 1.5
            
            # Factor 3: Cantidad de cambios
            if inp.shape == out.shape:
                changes = np.sum(inp != out)
                complexity += changes / inp.size
        
        return complexity / len(train_examples)
    
    def _create_contextual_embedding(self, observations: List[Dict], 
                                    memory_context: List) -> np.ndarray:
        """Crea embedding global con contexto de memoria"""
        # Combinar embeddings de observaciones
        obs_embeddings = [obs['transform_embedding'] for obs in observations]
        
        if obs_embeddings:
            # Promedio ponderado de embeddings
            global_emb = np.mean(obs_embeddings, axis=0)
            
            # Incorporar contexto de memoria si existe
            if memory_context:
                # Ajustar embedding basado en reglas exitosas previas
                for rule in memory_context[:5]:  # Top 5 reglas
                    if 'embedding' in rule:
                        global_emb = 0.7 * global_emb + 0.3 * np.array(rule['embedding'])
            
            return global_emb
        
        return np.zeros(256)
    
    def _consolidate_pixel_analysis(self, all_analysis: List[Dict]) -> Dict:
        """Consolida análisis a nivel de píxel"""
        pixel_patterns = {}
        for analysis in all_analysis:
            if 'pixel_level' in analysis:
                for pattern in analysis['pixel_level'].get('patterns', []):
                    key = pattern.get('type', 'unknown')
                    pixel_patterns[key] = pixel_patterns.get(key, 0) + 1
        return {'patterns': pixel_patterns}
    
    def _consolidate_object_analysis(self, all_analysis: List[Dict]) -> Dict:
        """Consolida análisis a nivel de objeto"""
        object_types = {}
        for analysis in all_analysis:
            if 'object_level' in analysis:
                for obj in analysis['object_level'].get('objects', []):
                    obj_type = obj.get('type', 'unknown')
                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
        return {'object_types': object_types}
    
    def _consolidate_relation_analysis(self, all_analysis: List[Dict]) -> Dict:
        """Consolida análisis a nivel de relación"""
        relations = {}
        for analysis in all_analysis:
            if 'relation_level' in analysis:
                for rel in analysis['relation_level'].get('relations', []):
                    rel_type = rel.get('type', 'unknown')
                    relations[rel_type] = relations.get(rel_type, 0) + 1
        return {'relations': relations}
    
    def _consolidate_pattern_analysis(self, all_analysis: List[Dict]) -> Dict:
        """Consolida análisis a nivel de patrón"""
        patterns = {}
        for analysis in all_analysis:
            if 'pattern_level' in analysis:
                for pattern in analysis['pattern_level'].get('patterns', []):
                    pat_type = pattern.get('type', 'unknown')
                    patterns[pat_type] = patterns.get(pat_type, 0) + 1
        return {'patterns': patterns}
    
    def _identify_important_patterns(self, attention_weights: np.ndarray) -> List[Dict]:
        """Identifica patrones importantes basado en pesos de atención"""
        important = []
        
        # Encontrar picos de atención
        if attention_weights.size > 0:
            mean_attention = np.mean(attention_weights)
            std_attention = np.std(attention_weights)
            threshold = mean_attention + 2 * std_attention
            
            # Índices con alta atención
            high_attention_idx = np.where(attention_weights > threshold)
            
            for idx in zip(*high_attention_idx):
                important.append({
                    'position': idx,
                    'weight': float(attention_weights[idx]),
                    'importance': 'high'
                })
        
        return important[:10]  # Top 10 más importantes
    
    def _build_reasoning_chain(self, rules: Dict, 
                              hierarchical: Dict, 
                              patterns: List) -> List[Dict]:
        """Construye cadena de razonamiento paso a paso"""
        chain = []
        
        # Paso 1: Análisis macro
        chain.append({
            'step': 1,
            'action': 'macro_analysis',
            'description': 'Identificación de patrones globales',
            'confidence': 0.9
        })
        
        # Paso 2: Descomposición en objetos
        if hierarchical.get('object_level'):
            chain.append({
                'step': 2,
                'action': 'object_decomposition',
                'description': f"Identificados {len(hierarchical['object_level'].get('object_types', {}))} tipos de objetos",
                'confidence': 0.85
            })
        
        # Paso 3: Aplicación de reglas
        if rules.get('macro_rules'):
            chain.append({
                'step': 3,
                'action': 'rule_application',
                'description': f"Aplicando {len(rules['macro_rules'])} reglas macro",
                'confidence': 0.8
            })
        
        # Paso 4: Refinamiento con atención
        if patterns:
            chain.append({
                'step': 4,
                'action': 'attention_refinement',
                'description': f"Refinando con {len(patterns)} patrones importantes",
                'confidence': 0.75
            })
        
        return chain
    
    def _apply_memory_rule(self, solution: np.ndarray, rule: Dict) -> np.ndarray:
        """Aplica regla exitosa de memoria"""
        # Simplificado - aplicar transformación si es compatible
        if 'transformation' in rule:
            transform = rule['transformation']
            if transform.get('type') == 'resize' and 'target_shape' in transform:
                # Aplicar cambio de tamaño
                target_shape = tuple(transform['target_shape'])
                if target_shape != solution.shape:
                    # Redimensionar si es necesario
                    from scipy.ndimage import zoom
                    factors = [t/s for t, s in zip(target_shape, solution.shape)]
                    solution = zoom(solution, factors, order=0)
        
        return solution
    
    def _apply_pattern_refinement(self, solution: np.ndarray, 
                                 pattern: Dict) -> np.ndarray:
        """Refina solución basado en patrón importante"""
        # Simplificado - ajustar valores en posiciones importantes
        if 'position' in pattern and pattern['importance'] == 'high':
            pos = pattern['position']
            if len(pos) == 2 and pos[0] < solution.shape[0] and pos[1] < solution.shape[1]:
                # Enfatizar posición importante
                if solution[pos] == 0:
                    # Si está vacío, considerar llenarlo
                    neighbors = self._get_neighbors(solution, pos[0], pos[1])
                    if neighbors:
                        solution[pos] = max(neighbors, key=neighbors.count)
        
        return solution
    
    def _ensure_coherence(self, solution: np.ndarray, 
                         train_examples: List[Dict]) -> np.ndarray:
        """Asegura coherencia con ejemplos de entrenamiento"""
        # Verificar que use colores consistentes
        output_colors = set()
        for ex in train_examples:
            output_colors.update(np.unique(ex['output']).tolist())
        
        # Ajustar colores no válidos
        for val in np.unique(solution):
            if val not in output_colors and val != 0:
                # Reemplazar con color más común
                if output_colors:
                    solution[solution == val] = max(output_colors)
        
        return solution
    
    def _extract_transformations(self, train_examples: List[Dict]) -> List[str]:
        """Extrae lista de transformaciones observadas"""
        transforms = []
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape != out.shape:
                transforms.append(f"resize_{inp.shape}_to_{out.shape}")
            
            new_colors = set(np.unique(out)) - set(np.unique(inp))
            if new_colors:
                transforms.append(f"add_colors_{new_colors}")
            
            if inp.shape == out.shape and not np.array_equal(inp, out):
                transforms.append("pixel_transformation")
        
        return transforms
    
    def _calculate_solution_fitness(self, solution: np.ndarray, 
                                   train_examples: List[Dict]) -> float:
        """Calcula fitness de la solución"""
        fitness = 0.5  # Base
        
        # Verificar consistencia de forma
        output_shapes = [np.array(ex['output']).shape for ex in train_examples]
        if solution.shape in output_shapes:
            fitness += 0.2
        
        # Verificar colores consistentes
        output_colors = set()
        for ex in train_examples:
            output_colors.update(np.unique(ex['output']).tolist())
        
        solution_colors = set(np.unique(solution).tolist())
        if solution_colors.issubset(output_colors):
            fitness += 0.3
        
        return min(fitness, 1.0)
    
    def _get_neighbors(self, grid: np.ndarray, i: int, j: int) -> List[int]:
        """Obtiene valores de vecinos"""
        neighbors = []
        h, w = grid.shape
        
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                neighbors.append(int(grid[ni, nj]))
        
        return neighbors