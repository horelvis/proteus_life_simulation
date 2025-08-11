#!/usr/bin/env python3
"""
Sistema de comparación mejorado con puzzles realistas tipo ARC
Compara: Baseline vs Attention vs Deep Learning
"""

import numpy as np
import json
import time
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Importar puzzles de prueba
from arc_test_puzzles import ARCTestPuzzles

# Importar solvers
from hybrid_proteus_solver import HybridProteusARCSolver
from hierarchical_attention_solver import HierarchicalAttentionSolver

# Intentar importar el solver de Deep Learning
try:
    from deep_learning_solver_fixed import DeepLearningARCSolver
    DL_AVAILABLE = True
except ImportError:
    DeepLearningARCSolver = None
    DL_AVAILABLE = False
    print("⚠️ Deep Learning solver no disponible")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSystemComparator:
    """Comparador mejorado con métricas detalladas"""
    
    def __init__(self):
        self.test_puzzles = ARCTestPuzzles.get_all_puzzles()
        self.results = {}
        
    def pixel_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calcula accuracy de píxeles"""
        if predicted.shape != expected.shape:
            # Intentar redimensionar si es necesario
            try:
                h, w = expected.shape
                if predicted.size == 0:
                    return 0.0
                # Recortar o expandir
                result = np.zeros_like(expected)
                min_h = min(predicted.shape[0], h)
                min_w = min(predicted.shape[1], w) if len(predicted.shape) > 1 else 1
                result[:min_h, :min_w] = predicted[:min_h, :min_w]
                predicted = result
            except:
                return 0.0
        
        return np.mean(predicted == expected)
    
    def structural_similarity(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calcula similitud estructural (objetos detectados correctamente)"""
        # Detectar componentes conectados
        from scipy.ndimage import label
        
        pred_binary = predicted != 0
        exp_binary = expected != 0
        
        pred_labels, pred_count = label(pred_binary)
        exp_labels, exp_count = label(exp_binary)
        
        # Comparar número de objetos
        count_similarity = 1.0 - abs(pred_count - exp_count) / max(pred_count, exp_count, 1)
        
        # Comparar overlap de objetos
        overlap = np.sum(pred_binary & exp_binary) / max(np.sum(exp_binary), 1)
        
        return (count_similarity + overlap) / 2
    
    def pattern_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Evalúa si el patrón general es correcto"""
        # Normalizar colores para comparar patrones
        def normalize_pattern(grid):
            unique_vals = np.unique(grid)
            normalized = np.zeros_like(grid)
            for i, val in enumerate(unique_vals):
                normalized[grid == val] = i
            return normalized
        
        if predicted.shape == expected.shape:
            pred_norm = normalize_pattern(predicted)
            exp_norm = normalize_pattern(expected)
            return np.mean(pred_norm == exp_norm)
        return 0.0
    
    def evaluate_solver(self, solver, solver_name: str) -> Dict:
        """Evalúa un solver en todos los puzzles"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluando: {solver_name}")
        logger.info('='*60)
        
        results = {
            'solver_name': solver_name,
            'puzzles': {},
            'total_exact_matches': 0,
            'total_pixel_accuracy': [],
            'total_structural_similarity': [],
            'total_pattern_accuracy': [],
            'total_time': 0,
            'errors': 0
        }
        
        for puzzle in self.test_puzzles:
            puzzle_name = puzzle['name']
            logger.info(f"\n🧩 Puzzle: {puzzle_name}")
            
            try:
                # Medir tiempo
                start_time = time.time()
                
                # Resolver puzzle
                train_examples = [
                    {'input': ex['input'].tolist(), 'output': ex['output'].tolist()}
                    for ex in puzzle['train']
                ]
                test_input = puzzle['test']['input']
                expected_output = puzzle['test']['output']
                
                # Obtener predicción
                predicted = solver.solve(train_examples, test_input)
                
                elapsed_time = time.time() - start_time
                
                # Calcular métricas
                pixel_acc = self.pixel_accuracy(predicted, expected_output)
                struct_sim = self.structural_similarity(predicted, expected_output)
                pattern_acc = self.pattern_accuracy(predicted, expected_output)
                exact_match = np.array_equal(predicted, expected_output)
                
                # Guardar resultados
                puzzle_results = {
                    'pixel_accuracy': pixel_acc,
                    'structural_similarity': struct_sim,
                    'pattern_accuracy': pattern_acc,
                    'exact_match': exact_match,
                    'time': elapsed_time,
                    'input_shape': test_input.shape,
                    'output_shape': expected_output.shape,
                    'predicted_shape': predicted.shape if isinstance(predicted, np.ndarray) else (0, 0)
                }
                
                results['puzzles'][puzzle_name] = puzzle_results
                results['total_pixel_accuracy'].append(pixel_acc)
                results['total_structural_similarity'].append(struct_sim)
                results['total_pattern_accuracy'].append(pattern_acc)
                if exact_match:
                    results['total_exact_matches'] += 1
                results['total_time'] += elapsed_time
                
                # Log resultado
                logger.info(f"  ✅ Exact Match: {'YES' if exact_match else 'NO'}")
                logger.info(f"  📊 Pixel Accuracy: {pixel_acc:.2%}")
                logger.info(f"  🔍 Structural Similarity: {struct_sim:.2%}")
                logger.info(f"  🎯 Pattern Accuracy: {pattern_acc:.2%}")
                logger.info(f"  ⏱️  Time: {elapsed_time:.3f}s")
                
            except Exception as e:
                logger.error(f"  ❌ Error: {str(e)}")
                results['puzzles'][puzzle_name] = {
                    'error': str(e),
                    'pixel_accuracy': 0,
                    'structural_similarity': 0,
                    'pattern_accuracy': 0,
                    'exact_match': False,
                    'time': 0
                }
                results['errors'] += 1
        
        # Calcular promedios
        results['avg_pixel_accuracy'] = np.mean(results['total_pixel_accuracy']) if results['total_pixel_accuracy'] else 0
        results['avg_structural_similarity'] = np.mean(results['total_structural_similarity']) if results['total_structural_similarity'] else 0
        results['avg_pattern_accuracy'] = np.mean(results['total_pattern_accuracy']) if results['total_pattern_accuracy'] else 0
        results['exact_match_rate'] = results['total_exact_matches'] / len(self.test_puzzles)
        
        return results
    
    def compare_all_systems(self):
        """Compara todos los sistemas disponibles"""
        logger.info("\n" + "="*80)
        logger.info("COMPARACIÓN DE SISTEMAS EN PUZZLES TIPO ARC")
        logger.info("="*80)
        
        # Sistemas a evaluar
        systems = [
            (HybridProteusARCSolver(), "Baseline (Hybrid Proteus)"),
            (HierarchicalAttentionSolver(), "Hierarchical Attention (HAMS)")
        ]
        
        if DL_AVAILABLE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            systems.append((DeepLearningARCSolver(device=device), "Deep Learning (A-MHA)"))
        
        # Evaluar cada sistema
        all_results = {}
        for solver, name in systems:
            results = self.evaluate_solver(solver, name)
            all_results[name] = results
        
        # Generar reporte comparativo
        self.generate_comparison_report(all_results)
        
        # Guardar resultados
        self.save_results(all_results)
        
        return all_results
    
    def generate_comparison_report(self, all_results: Dict):
        """Genera reporte comparativo detallado"""
        logger.info("\n" + "="*80)
        logger.info("📊 REPORTE COMPARATIVO FINAL")
        logger.info("="*80)
        
        # Tabla de métricas principales
        logger.info("\n🎯 Métricas Generales:")
        logger.info("-" * 70)
        logger.info(f"{'Sistema':<30} {'Exact':<10} {'Pixel':<10} {'Struct':<10} {'Pattern':<10}")
        logger.info("-" * 70)
        
        for name, results in all_results.items():
            logger.info(
                f"{name:<30} "
                f"{results['exact_match_rate']:<10.1%} "
                f"{results['avg_pixel_accuracy']:<10.1%} "
                f"{results['avg_structural_similarity']:<10.1%} "
                f"{results['avg_pattern_accuracy']:<10.1%}"
            )
        
        # Análisis por puzzle
        logger.info("\n📋 Desglose por Puzzle:")
        logger.info("-" * 70)
        
        puzzle_names = self.test_puzzles[0]['name'] if self.test_puzzles else []
        for puzzle in self.test_puzzles:
            puzzle_name = puzzle['name']
            logger.info(f"\n🧩 {puzzle_name}:")
            
            for system_name, results in all_results.items():
                if puzzle_name in results['puzzles']:
                    puzzle_res = results['puzzles'][puzzle_name]
                    if 'error' not in puzzle_res:
                        match_str = "✅" if puzzle_res['exact_match'] else "❌"
                        logger.info(
                            f"  {system_name:<25} {match_str} "
                            f"Pixel: {puzzle_res['pixel_accuracy']:.1%} "
                            f"Time: {puzzle_res['time']:.3f}s"
                        )
                    else:
                        logger.info(f"  {system_name:<25} ❌ Error: {puzzle_res['error'][:30]}")
        
        # Resumen de rendimiento
        logger.info("\n⚡ Rendimiento:")
        logger.info("-" * 70)
        for name, results in all_results.items():
            avg_time = results['total_time'] / len(self.test_puzzles)
            logger.info(f"{name:<30} Tiempo promedio: {avg_time:.3f}s")
        
        # Ganador
        logger.info("\n🏆 MEJOR SISTEMA:")
        logger.info("-" * 70)
        
        # Determinar ganador por exact match rate
        best_system = max(all_results.items(), 
                         key=lambda x: (x[1]['exact_match_rate'], x[1]['avg_pixel_accuracy']))
        
        logger.info(f"  {best_system[0]}")
        logger.info(f"  - Exact Match Rate: {best_system[1]['exact_match_rate']:.1%}")
        logger.info(f"  - Pixel Accuracy: {best_system[1]['avg_pixel_accuracy']:.1%}")
        logger.info(f"  - Puzzles resueltos: {best_system[1]['total_exact_matches']}/{len(self.test_puzzles)}")
    
    def save_results(self, all_results: Dict):
        """Guarda resultados en archivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
        
        # Convertir numpy arrays a listas para serialización
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Resultados guardados en: {filename}")

def main():
    """Función principal"""
    comparator = EnhancedSystemComparator()
    results = comparator.compare_all_systems()
    
    # Análisis adicional
    logger.info("\n" + "="*80)
    logger.info("💡 ANÁLISIS DE RESULTADOS")
    logger.info("="*80)
    
    # Identificar fortalezas y debilidades
    for system_name, system_results in results.items():
        logger.info(f"\n📈 {system_name}:")
        
        # Encontrar mejores y peores puzzles
        puzzle_scores = [
            (name, res['pixel_accuracy']) 
            for name, res in system_results['puzzles'].items()
            if 'error' not in res
        ]
        
        if puzzle_scores:
            puzzle_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("  Mejores puzzles:")
            for name, score in puzzle_scores[:3]:
                logger.info(f"    - {name}: {score:.1%}")
            
            logger.info("  Puzzles más difíciles:")
            for name, score in puzzle_scores[-3:]:
                logger.info(f"    - {name}: {score:.1%}")

if __name__ == "__main__":
    main()