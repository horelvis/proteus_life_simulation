#!/usr/bin/env python3
"""
Enhanced Memory - Memoria compartida mejorada con historial de puzzles similares
Implementa persistencia, clasificación de reglas y reutilización de conocimiento
"""

import numpy as np
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import sqlite3
import os

logger = logging.getLogger(__name__)

@dataclass
class PuzzleSignature:
    """Firma única de un puzzle para comparación de similitud"""
    size: Tuple[int, int]
    num_colors: int
    color_distribution: Dict[int, float]
    has_size_changes: bool
    has_color_changes: bool
    has_patterns: bool
    has_symmetry: bool
    complexity_score: float
    hash_signature: str = field(default="")
    
    def __post_init__(self):
        if not self.hash_signature:
            self.hash_signature = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calcula hash único del puzzle"""
        content = f"{self.size}_{self.num_colors}_{sorted(self.color_distribution.items())}"
        content += f"_{self.has_size_changes}_{self.has_color_changes}"
        content += f"_{self.has_patterns}_{self.has_symmetry}_{self.complexity_score:.3f}"
        
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class SolutionRule:
    """Regla de solución con metadatos mejorados"""
    rule_id: str
    rule_type: str
    rule_data: Dict[str, Any]
    fitness: float
    agent_id: int
    puzzle_signature: PuzzleSignature
    success_count: int = 0
    failure_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    transformation_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = self.generate_rule_id()
        self.confidence = self.calculate_confidence()
    
    def generate_rule_id(self) -> str:
        """Genera ID único para la regla"""
        content = f"{self.rule_type}_{self.rule_data}_{self.puzzle_signature.hash_signature}"
        return hashlib.sha1(content.encode()).hexdigest()[:12]
    
    def calculate_confidence(self) -> float:
        """Calcula confianza basada en historial de éxito"""
        total_attempts = self.success_count + self.failure_count
        if total_attempts == 0:
            return self.fitness
        
        success_rate = self.success_count / total_attempts
        confidence = (success_rate * 0.7) + (self.fitness * 0.3)
        return confidence

@dataclass
class PuzzleMemory:
    """Memoria específica de un puzzle"""
    puzzle_signature: PuzzleSignature
    successful_rules: List[SolutionRule]
    failed_rules: List[SolutionRule]
    solution_chains: List[List[str]]
    solve_count: int = 0
    avg_fitness: float = 0.0
    last_solved: Optional[datetime] = None

class EnhancedSharedMemory:
    """
    Memoria compartida mejorada con capacidades avanzadas:
    - Historial persistente de puzzles
    - Clasificación automática de reglas
    - Similitud entre puzzles
    - Aprendizaje acumulativo
    """
    
    def __init__(self, memory_dir: str = "/tmp/proteus_memory"):
        self.memory_dir = memory_dir
        self.db_path = os.path.join(memory_dir, "proteus_memory.db")
        
        # Estructuras en memoria
        self.puzzle_memories: Dict[str, PuzzleMemory] = {}
        self.rule_database: Dict[str, SolutionRule] = {}
        self.similarity_clusters: Dict[str, List[str]] = {}
        
        # Cache para búsquedas rápidas
        self.rule_by_type_cache: Dict[str, List[str]] = defaultdict(list)
        self.puzzle_similarity_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Estadísticas
        self.stats = {
            'total_puzzles': 0,
            'total_rules': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'cache_hits': 0
        }
        
        self._initialize_memory_system()
        self._load_existing_memory()
    
    def _initialize_memory_system(self):
        """Inicializa sistema de memoria persistente"""
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Crear base de datos SQLite
        self._initialize_database()
        
        logger.info(f"Sistema de memoria inicializado en: {self.memory_dir}")
    
    def _initialize_database(self):
        """Inicializa base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de puzzles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzles (
                puzzle_hash TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                solve_count INTEGER DEFAULT 0,
                avg_fitness REAL DEFAULT 0.0,
                last_solved TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de reglas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rules (
                rule_id TEXT PRIMARY KEY,
                rule_type TEXT NOT NULL,
                rule_data TEXT NOT NULL,
                fitness REAL NOT NULL,
                puzzle_hash TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (puzzle_hash) REFERENCES puzzles (puzzle_hash)
            )
        ''')
        
        # Tabla de similitudes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS puzzle_similarities (
                puzzle1_hash TEXT,
                puzzle2_hash TEXT,
                similarity_score REAL,
                PRIMARY KEY (puzzle1_hash, puzzle2_hash)
            )
        ''')
        
        # Índices para optimización
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_type ON rules (rule_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_puzzle ON rules (puzzle_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules (confidence DESC)')
        
        conn.commit()
        conn.close()
    
    def _load_existing_memory(self):
        """Carga memoria existente desde persistencia"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Cargar puzzles
            cursor.execute('SELECT puzzle_hash, signature, solve_count, avg_fitness, last_solved FROM puzzles')
            puzzle_rows = cursor.fetchall()
            
            for row in puzzle_rows:
                puzzle_hash, signature_json, solve_count, avg_fitness, last_solved = row
                
                try:
                    signature_data = json.loads(signature_json)
                    signature = PuzzleSignature(**signature_data)
                    
                    self.puzzle_memories[puzzle_hash] = PuzzleMemory(
                        puzzle_signature=signature,
                        successful_rules=[],
                        failed_rules=[],
                        solution_chains=[],
                        solve_count=solve_count,
                        avg_fitness=avg_fitness,
                        last_solved=datetime.fromisoformat(last_solved) if last_solved else None
                    )
                except Exception as e:
                    logger.warning(f"Error cargando puzzle {puzzle_hash}: {e}")
            
            # Cargar reglas
            cursor.execute('''
                SELECT rule_id, rule_type, rule_data, fitness, puzzle_hash, 
                       success_count, failure_count, confidence, timestamp 
                FROM rules
            ''')
            rule_rows = cursor.fetchall()
            
            for row in rule_rows:
                (rule_id, rule_type, rule_data_json, fitness, puzzle_hash,
                 success_count, failure_count, confidence, timestamp) = row
                
                try:
                    rule_data = json.loads(rule_data_json)
                    
                    if puzzle_hash in self.puzzle_memories:
                        puzzle_sig = self.puzzle_memories[puzzle_hash].puzzle_signature
                        
                        rule = SolutionRule(
                            rule_id=rule_id,
                            rule_type=rule_type,
                            rule_data=rule_data,
                            fitness=fitness,
                            agent_id=-1,  # Desconocido al cargar
                            puzzle_signature=puzzle_sig,
                            success_count=success_count,
                            failure_count=failure_count,
                            timestamp=datetime.fromisoformat(timestamp)
                        )
                        
                        self.rule_database[rule_id] = rule
                        self.rule_by_type_cache[rule_type].append(rule_id)
                        
                        # Añadir a memoria de puzzle
                        if fitness > 0.5:
                            self.puzzle_memories[puzzle_hash].successful_rules.append(rule)
                        else:
                            self.puzzle_memories[puzzle_hash].failed_rules.append(rule)
                            
                except Exception as e:
                    logger.warning(f"Error cargando regla {rule_id}: {e}")
            
            conn.close()
            
            self.stats['total_puzzles'] = len(self.puzzle_memories)
            self.stats['total_rules'] = len(self.rule_database)
            
            logger.info(f"Memoria cargada: {self.stats['total_puzzles']} puzzles, {self.stats['total_rules']} reglas")
            
        except Exception as e:
            logger.error(f"Error cargando memoria existente: {e}")
    
    def create_puzzle_signature(self, train_examples: List[Dict], test_input: np.ndarray) -> PuzzleSignature:
        """Crea firma única de un puzzle"""
        
        # Analizar ejemplos de entrenamiento
        all_matrices = []
        for example in train_examples:
            all_matrices.append(example['input'])
            all_matrices.append(example['output'])
        all_matrices.append(test_input)
        
        # Estadísticas generales
        sizes = [matrix.shape for matrix in all_matrices]
        unique_size = len(set(sizes)) == 1
        avg_size = tuple(np.mean([s for s in sizes], axis=0).astype(int))
        
        # Análisis de colores
        all_colors = set()
        color_counts = defaultdict(int)
        
        for matrix in all_matrices:
            unique_colors = np.unique(matrix)
            all_colors.update(unique_colors)
            
            for color in unique_colors:
                color_counts[int(color)] += np.sum(matrix == color)
        
        total_pixels = sum(color_counts.values())
        color_distribution = {
            color: count / total_pixels 
            for color, count in color_counts.items()
        }
        
        # Detectar características
        has_size_changes = self._detect_size_changes(train_examples)
        has_color_changes = self._detect_color_changes(train_examples)
        has_patterns = self._detect_pattern_characteristics(train_examples)
        has_symmetry = self._detect_symmetry_characteristics(train_examples)
        
        # Calcular complejidad
        complexity_score = self._calculate_complexity_score(
            all_matrices, len(all_colors), has_size_changes, has_color_changes, has_patterns
        )
        
        return PuzzleSignature(
            size=avg_size,
            num_colors=len(all_colors),
            color_distribution=color_distribution,
            has_size_changes=has_size_changes,
            has_color_changes=has_color_changes,
            has_patterns=has_patterns,
            has_symmetry=has_symmetry,
            complexity_score=complexity_score
        )
    
    def _detect_size_changes(self, train_examples: List[Dict]) -> bool:
        """Detecta si hay cambios de tamaño input->output"""
        for example in train_examples:
            input_shape = np.array(example['input']).shape
            output_shape = np.array(example['output']).shape
            if input_shape != output_shape:
                return True
        return False
    
    def _detect_color_changes(self, train_examples: List[Dict]) -> bool:
        """Detecta si hay cambios en los colores usados"""
        for example in train_examples:
            input_colors = set(np.unique(example['input']))
            output_colors = set(np.unique(example['output']))
            if input_colors != output_colors:
                return True
        return False
    
    def _detect_pattern_characteristics(self, train_examples: List[Dict]) -> bool:
        """Detecta características de patrones"""
        # Implementación simplificada
        for example in train_examples:
            input_matrix = np.array(example['input'])
            if input_matrix.shape[0] > 5 and input_matrix.shape[1] > 5:
                # Buscar repeticiones simples
                if self._has_repetitive_patterns(input_matrix):
                    return True
        return False
    
    def _has_repetitive_patterns(self, matrix: np.ndarray) -> bool:
        """Detecta patrones repetitivos básicos"""
        h, w = matrix.shape
        
        # Buscar repeticiones horizontales pequeñas
        for period in range(2, min(w//2, 5)):
            if w % period == 0:
                repeated = True
                for r in range(h):
                    for c in range(period, w):
                        if matrix[r, c] != matrix[r, c % period]:
                            repeated = False
                            break
                    if not repeated:
                        break
                if repeated:
                    return True
        
        return False
    
    def _detect_symmetry_characteristics(self, train_examples: List[Dict]) -> bool:
        """Detecta características de simetría"""
        for example in train_examples:
            matrix = np.array(example['input'])
            
            # Simetría horizontal
            if np.array_equal(matrix, np.flipud(matrix)):
                return True
            
            # Simetría vertical
            if np.array_equal(matrix, np.fliplr(matrix)):
                return True
        
        return False
    
    def _calculate_complexity_score(self, matrices: List[np.ndarray], num_colors: int,
                                   has_size_changes: bool, has_color_changes: bool,
                                   has_patterns: bool) -> float:
        """Calcula puntuación de complejidad del puzzle"""
        score = 0.0
        
        # Complejidad por tamaño
        avg_size = np.mean([matrix.size for matrix in matrices])
        score += min(1.0, avg_size / 400)  # Normalizar hasta 400 píxeles
        
        # Complejidad por número de colores
        score += min(1.0, num_colors / 10)
        
        # Complejidad por transformaciones
        if has_size_changes:
            score += 0.3
        if has_color_changes:
            score += 0.2
        if has_patterns:
            score += 0.2
        
        # Complejidad por entropía
        for matrix in matrices:
            unique, counts = np.unique(matrix, return_counts=True)
            if len(counts) > 1:
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                score += entropy / 10  # Normalizar entropía
        
        return min(2.0, score)  # Limitar a máximo 2.0
    
    def store_successful_rule(self, rule_type: str, rule_data: Dict, fitness: float,
                             agent_id: int, puzzle_signature: PuzzleSignature,
                             transformation_chain: List[str] = None) -> str:
        """Almacena una regla exitosa en memoria"""
        
        rule = SolutionRule(
            rule_id="",  # Se generará automáticamente
            rule_type=rule_type,
            rule_data=rule_data,
            fitness=fitness,
            agent_id=agent_id,
            puzzle_signature=puzzle_signature,
            transformation_chain=transformation_chain or []
        )
        
        # Verificar si regla ya existe
        existing_rule = self.rule_database.get(rule.rule_id)
        if existing_rule:
            # Actualizar estadísticas de regla existente
            existing_rule.success_count += 1
            existing_rule.fitness = max(existing_rule.fitness, fitness)
            existing_rule.confidence = existing_rule.calculate_confidence()
            rule = existing_rule
        else:
            # Nueva regla
            rule.success_count = 1
            self.rule_database[rule.rule_id] = rule
            self.rule_by_type_cache[rule_type].append(rule.rule_id)
        
        # Actualizar memoria de puzzle
        puzzle_hash = puzzle_signature.hash_signature
        if puzzle_hash not in self.puzzle_memories:
            self.puzzle_memories[puzzle_hash] = PuzzleMemory(
                puzzle_signature=puzzle_signature,
                successful_rules=[],
                failed_rules=[],
                solution_chains=[]
            )
        
        puzzle_memory = self.puzzle_memories[puzzle_hash]
        
        # Añadir regla si no está ya presente
        if not any(r.rule_id == rule.rule_id for r in puzzle_memory.successful_rules):
            puzzle_memory.successful_rules.append(rule)
        
        # Actualizar estadísticas de puzzle
        puzzle_memory.solve_count += 1
        puzzle_memory.avg_fitness = (
            (puzzle_memory.avg_fitness * (puzzle_memory.solve_count - 1) + fitness) / 
            puzzle_memory.solve_count
        )
        puzzle_memory.last_solved = datetime.now()
        
        # Persistir en base de datos
        self._persist_rule(rule, puzzle_hash)
        
        self.stats['total_rules'] = len(self.rule_database)
        
        logger.debug(f"Regla almacenada: {rule.rule_id}, fitness={fitness:.3f}")
        
        return rule.rule_id
    
    def retrieve_similar_rules(self, puzzle_signature: PuzzleSignature,
                              rule_type: Optional[str] = None,
                              top_k: int = 5) -> List[SolutionRule]:
        """Recupera reglas de puzzles similares"""
        
        # Buscar puzzles similares
        similar_puzzles = self.find_similar_puzzles(puzzle_signature, top_k=top_k*2)
        
        # Recopilar reglas de puzzles similares
        candidate_rules = []
        
        for puzzle_hash, similarity_score in similar_puzzles:
            if puzzle_hash in self.puzzle_memories:
                puzzle_memory = self.puzzle_memories[puzzle_hash]
                
                for rule in puzzle_memory.successful_rules:
                    if rule_type is None or rule.rule_type == rule_type:
                        # Ajustar confianza por similitud de puzzle
                        adjusted_confidence = rule.confidence * similarity_score
                        
                        candidate_rules.append((rule, adjusted_confidence))
        
        # Ordenar por confianza ajustada
        candidate_rules.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top-k reglas
        retrieved_rules = [rule for rule, _ in candidate_rules[:top_k]]
        
        if retrieved_rules:
            self.stats['successful_retrievals'] += 1
            logger.debug(f"Recuperadas {len(retrieved_rules)} reglas similares")
        else:
            self.stats['failed_retrievals'] += 1
        
        return retrieved_rules
    
    def find_similar_puzzles(self, target_signature: PuzzleSignature,
                           top_k: int = 10) -> List[Tuple[str, float]]:
        """Encuentra puzzles similares al target"""
        
        target_hash = target_signature.hash_signature
        
        # Verificar cache
        if target_hash in self.puzzle_similarity_cache:
            self.stats['cache_hits'] += 1
            cached_results = self.puzzle_similarity_cache[target_hash]
            return cached_results[:top_k]
        
        # Calcular similitudes
        similarities = []
        
        for puzzle_hash, puzzle_memory in self.puzzle_memories.items():
            if puzzle_hash == target_hash:
                continue  # Skip mismo puzzle
            
            similarity = self._calculate_puzzle_similarity(
                target_signature, puzzle_memory.puzzle_signature
            )
            
            if similarity > 0.1:  # Umbral mínimo de similitud
                similarities.append((puzzle_hash, similarity))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Actualizar cache
        self.puzzle_similarity_cache[target_hash] = similarities
        
        return similarities[:top_k]
    
    def _calculate_puzzle_similarity(self, sig1: PuzzleSignature, sig2: PuzzleSignature) -> float:
        """Calcula similitud entre dos puzzles"""
        
        similarity_score = 0.0
        
        # Similitud de tamaño (peso: 0.2)
        size_diff = abs(sig1.size[0] - sig2.size[0]) + abs(sig1.size[1] - sig2.size[1])
        size_similarity = max(0, 1.0 - size_diff / 20)  # Normalizar hasta diff=20
        similarity_score += size_similarity * 0.2
        
        # Similitud de número de colores (peso: 0.15)
        color_diff = abs(sig1.num_colors - sig2.num_colors)
        color_similarity = max(0, 1.0 - color_diff / 10)
        similarity_score += color_similarity * 0.15
        
        # Similitud de distribución de colores (peso: 0.25)
        color_dist_similarity = self._calculate_distribution_similarity(
            sig1.color_distribution, sig2.color_distribution
        )
        similarity_score += color_dist_similarity * 0.25
        
        # Similitudes booleanas (peso: 0.3 total)
        boolean_features = [
            ('has_size_changes', 0.1),
            ('has_color_changes', 0.1),
            ('has_patterns', 0.05),
            ('has_symmetry', 0.05)
        ]
        
        for feature, weight in boolean_features:
            if getattr(sig1, feature) == getattr(sig2, feature):
                similarity_score += weight
        
        # Similitud de complejidad (peso: 0.1)
        complexity_diff = abs(sig1.complexity_score - sig2.complexity_score)
        complexity_similarity = max(0, 1.0 - complexity_diff / 2.0)
        similarity_score += complexity_similarity * 0.1
        
        return min(1.0, similarity_score)
    
    def _calculate_distribution_similarity(self, dist1: Dict[int, float], 
                                         dist2: Dict[int, float]) -> float:
        """Calcula similitud entre distribuciones de colores"""
        all_colors = set(dist1.keys()) | set(dist2.keys())
        
        if not all_colors:
            return 1.0
        
        # Crear vectores de distribución
        vec1 = [dist1.get(color, 0.0) for color in all_colors]
        vec2 = [dist2.get(color, 0.0) for color in all_colors]
        
        # Calcular similitud coseno
        try:
            similarity = 1.0 - cosine(vec1, vec2)
            return max(0.0, similarity)
        except:
            return 0.0
    
    def update_rule_performance(self, rule_id: str, success: bool):
        """Actualiza rendimiento de una regla"""
        if rule_id in self.rule_database:
            rule = self.rule_database[rule_id]
            
            if success:
                rule.success_count += 1
            else:
                rule.failure_count += 1
            
            rule.confidence = rule.calculate_confidence()
            
            # Actualizar en base de datos
            self._update_rule_performance_db(rule_id, success)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la memoria"""
        
        # Estadísticas de reglas por tipo
        rule_type_counts = defaultdict(int)
        avg_confidence_by_type = defaultdict(list)
        
        for rule in self.rule_database.values():
            rule_type_counts[rule.rule_type] += 1
            avg_confidence_by_type[rule.rule_type].append(rule.confidence)
        
        # Calcular promedios
        for rule_type in avg_confidence_by_type:
            avg_confidence_by_type[rule_type] = np.mean(avg_confidence_by_type[rule_type])
        
        # Estadísticas de puzzles
        puzzle_complexities = [pm.puzzle_signature.complexity_score 
                              for pm in self.puzzle_memories.values()]
        
        return {
            'total_puzzles': len(self.puzzle_memories),
            'total_rules': len(self.rule_database),
            'rules_by_type': dict(rule_type_counts),
            'avg_confidence_by_type': dict(avg_confidence_by_type),
            'avg_puzzle_complexity': np.mean(puzzle_complexities) if puzzle_complexities else 0,
            'successful_retrievals': self.stats['successful_retrievals'],
            'failed_retrievals': self.stats['failed_retrievals'],
            'cache_hits': self.stats['cache_hits'],
            'memory_size_mb': self._estimate_memory_size()
        }
    
    def cleanup_old_entries(self, days_threshold: int = 30):
        """Limpia entradas antiguas de la memoria"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Identificar reglas antiguas con bajo rendimiento
        rules_to_remove = []
        
        for rule_id, rule in self.rule_database.items():
            if (rule.timestamp < cutoff_date and 
                rule.confidence < 0.3 and 
                rule.success_count == 0):
                rules_to_remove.append(rule_id)
        
        # Eliminar reglas
        for rule_id in rules_to_remove:
            del self.rule_database[rule_id]
            
            # Limpiar cache
            for rule_type, rule_list in self.rule_by_type_cache.items():
                if rule_id in rule_list:
                    rule_list.remove(rule_id)
        
        # Limpiar cache de similitudes
        self.puzzle_similarity_cache.clear()
        
        logger.info(f"Limpiadas {len(rules_to_remove)} reglas antiguas")
    
    def export_knowledge_base(self, filepath: str):
        """Exporta base de conocimiento a archivo"""
        export_data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'puzzle_memories': {},
            'rules': {},
            'statistics': self.get_memory_statistics()
        }
        
        # Exportar memorias de puzzles
        for puzzle_hash, puzzle_memory in self.puzzle_memories.items():
            export_data['puzzle_memories'][puzzle_hash] = {
                'signature': asdict(puzzle_memory.puzzle_signature),
                'solve_count': puzzle_memory.solve_count,
                'avg_fitness': puzzle_memory.avg_fitness,
                'last_solved': puzzle_memory.last_solved.isoformat() if puzzle_memory.last_solved else None
            }
        
        # Exportar reglas
        for rule_id, rule in self.rule_database.items():
            export_data['rules'][rule_id] = asdict(rule)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Base de conocimiento exportada a: {filepath}")
    
    # Métodos auxiliares de persistencia
    
    def _persist_rule(self, rule: SolutionRule, puzzle_hash: str):
        """Persiste regla en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insertar o actualizar puzzle
            cursor.execute('''
                INSERT OR REPLACE INTO puzzles (puzzle_hash, signature, solve_count, avg_fitness, last_solved)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                puzzle_hash,
                json.dumps(asdict(rule.puzzle_signature)),
                self.puzzle_memories[puzzle_hash].solve_count,
                self.puzzle_memories[puzzle_hash].avg_fitness,
                datetime.now().isoformat()
            ))
            
            # Insertar o actualizar regla
            cursor.execute('''
                INSERT OR REPLACE INTO rules 
                (rule_id, rule_type, rule_data, fitness, puzzle_hash, success_count, failure_count, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.rule_type,
                json.dumps(rule.rule_data),
                rule.fitness,
                puzzle_hash,
                rule.success_count,
                rule.failure_count,
                rule.confidence,
                rule.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error persistiendo regla: {e}")
    
    def _update_rule_performance_db(self, rule_id: str, success: bool):
        """Actualiza rendimiento de regla en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if success:
                cursor.execute('''
                    UPDATE rules SET success_count = success_count + 1, 
                                   confidence = ? 
                    WHERE rule_id = ?
                ''', (self.rule_database[rule_id].confidence, rule_id))
            else:
                cursor.execute('''
                    UPDATE rules SET failure_count = failure_count + 1,
                                   confidence = ?
                    WHERE rule_id = ?
                ''', (self.rule_database[rule_id].confidence, rule_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error actualizando rendimiento de regla: {e}")
    
    def _estimate_memory_size(self) -> float:
        """Estima tamaño de memoria en MB"""
        try:
            import sys
            total_size = 0
            
            total_size += sys.getsizeof(self.puzzle_memories)
            total_size += sys.getsizeof(self.rule_database)
            total_size += sys.getsizeof(self.puzzle_similarity_cache)
            
            # Tamaño aproximado del archivo de base de datos
            if os.path.exists(self.db_path):
                total_size += os.path.getsize(self.db_path)
            
            return total_size / (1024 * 1024)  # Convertir a MB
            
        except Exception:
            return 0.0


def main():
    """Función de prueba de memoria mejorada"""
    print("\n" + "="*70)
    print("🧠 ENHANCED SHARED MEMORY - PRUEBA")
    print("="*70)
    
    # Crear instancia de memoria mejorada
    memory = EnhancedSharedMemory()
    
    # Crear firma de puzzle de ejemplo
    train_examples = [
        {'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]},
        {'input': [[5, 6], [7, 8]], 'output': [[6, 7], [8, 9]]}
    ]
    test_input = np.array([[9, 10], [11, 12]])
    
    signature = memory.create_puzzle_signature(train_examples, test_input)
    print(f"\n📝 Firma de puzzle creada:")
    print(f"   Hash: {signature.hash_signature}")
    print(f"   Tamaño: {signature.size}")
    print(f"   Colores: {signature.num_colors}")
    print(f"   Complejidad: {signature.complexity_score:.3f}")
    
    # Almacenar regla de ejemplo
    rule_id = memory.store_successful_rule(
        rule_type="color_increment",
        rule_data={"operation": "add", "value": 1},
        fitness=0.95,
        agent_id=1,
        puzzle_signature=signature
    )
    
    print(f"\n🔧 Regla almacenada: {rule_id}")
    
    # Obtener estadísticas
    stats = memory.get_memory_statistics()
    print(f"\n📊 Estadísticas de memoria:")
    print(f"   Puzzles: {stats['total_puzzles']}")
    print(f"   Reglas: {stats['total_rules']}")
    print(f"   Tamaño: {stats['memory_size_mb']:.2f} MB")
    
    print("\n✅ Memoria mejorada funcionando correctamente")
    print("="*70)


if __name__ == "__main__":
    main()