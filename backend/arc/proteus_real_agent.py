#!/usr/bin/env python3
"""
PROTEUS Real Agent - Conexión REAL al ARC Prize
SIN simulaciones, SOLO puzzles oficiales del dataset ARC
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.arc_real_solver import ARCRealSolver
from arc.arc_swarm_solver_improved import ARCSwarmSolverImproved

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProteusRealAgent:
    """
    Agente PROTEUS que resuelve puzzles REALES del ARC Prize
    - Usa dataset oficial ARC
    - Métricas de evaluación reales
    - SIN datos simulados
    """
    
    def __init__(self, cache_dir: str = "/app/arc_official_cache"):
        """
        Inicializa con dataset ARC real
        
        Args:
            cache_dir: Directorio con puzzles oficiales
        """
        self.arc_loader = ARCRealSolver(cache_dir)
        
        # Configurar solver evolutivo
        self.swarm_solver = ARCSwarmSolverImproved(
            population_size=30,
            generations=5,
            mutation_rate=0.35,
            crossover_rate=0.7,
            elite_size=5
        )
        
        # Estadísticas del dataset
        self.dataset_stats = self.arc_loader.get_puzzle_stats()
        
        logger.info("🧬 PROTEUS Real Agent inicializado")
        logger.info(f"   Dataset: {self.dataset_stats['total_puzzles']} puzzles reales")
        logger.info(f"   Ejemplos: {self.dataset_stats['total_train_examples']} de entrenamiento")
        logger.info(f"   Casos de prueba: {self.dataset_stats['total_test_cases']}")
    
    def solve_random_puzzle(self) -> Dict[str, Any]:
        """
        Resuelve un puzzle aleatorio del dataset oficial
        
        Returns:
            Resultados de evaluación real
        """
        # Obtener puzzle real aleatorio
        puzzle_id, puzzle_data = self.arc_loader.get_random_puzzle()
        
        logger.info(f"🎯 Resolviendo puzzle REAL: {puzzle_id}")
        logger.info(f"   Ejemplos de entrenamiento: {len(puzzle_data['train'])}")
        logger.info(f"   Casos de prueba: {len(puzzle_data['test'])}")
        
        # Mostrar información del puzzle
        first_example = puzzle_data['train'][0]
        logger.info(f"   Input shape: {first_example['input'].shape}")
        logger.info(f"   Output shape: {first_example['output'].shape}")
        
        results = []
        
        # Resolver cada caso de prueba
        for test_idx, test_case in enumerate(puzzle_data['test']):
            logger.info(f"\n📋 Caso de prueba {test_idx + 1}/{len(puzzle_data['test'])}")
            logger.info(f"   Test input shape: {test_case['input'].shape}")
            
            try:
                # Resolver con enjambre evolutivo
                solution, metadata = self.swarm_solver.solve_with_swarm(
                    puzzle_data['train'], test_case['input']
                )
                
                if solution is not None:
                    # Evaluar contra dataset oficial
                    evaluation = self.arc_loader.evaluate_solution(
                        puzzle_id, test_idx, solution
                    )
                    
                    evaluation['metadata'] = metadata
                    evaluation['solver'] = 'PROTEUS Swarm'
                    
                    results.append(evaluation)
                    
                    logger.info(f"✅ Solución generada:")
                    logger.info(f"   Shape: {solution.shape}")
                    logger.info(f"   Fitness: {metadata.get('best_fitness', 0):.3f}")
                    logger.info(f"   Agentes vivos: {metadata.get('alive_agents', 0)}")
                    
                    if 'accuracy' in evaluation:
                        logger.info(f"   📊 PRECISIÓN OFICIAL: {evaluation['accuracy']:.3f}")
                        logger.info(f"   ✓ Correcto: {'SÍ' if evaluation.get('correct', False) else 'NO'}")
                    
                else:
                    logger.warning("❌ No se pudo generar solución")
                    results.append({
                        'puzzle_id': puzzle_id,
                        'test_index': test_idx,
                        'error': 'No solution generated',
                        'correct': False,
                        'accuracy': 0.0
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error resolviendo caso {test_idx}: {e}")
                results.append({
                    'puzzle_id': puzzle_id,
                    'test_index': test_idx,
                    'error': str(e),
                    'correct': False,
                    'accuracy': 0.0
                })
        
        # Calcular estadísticas finales
        total_accuracy = np.mean([r.get('accuracy', 0) for r in results])
        total_correct = sum(1 for r in results if r.get('correct', False))
        
        final_results = {
            'puzzle_id': puzzle_id,
            'puzzle_source': 'ARC Prize Official Dataset',
            'total_test_cases': len(puzzle_data['test']),
            'results': results,
            'average_accuracy': total_accuracy,
            'cases_correct': total_correct,
            'success_rate': total_correct / len(puzzle_data['test']) if puzzle_data['test'] else 0,
            'dataset_stats': self.dataset_stats
        }
        
        return final_results
    
    def solve_specific_puzzle(self, puzzle_id: str) -> Dict[str, Any]:
        """
        Resuelve un puzzle específico del dataset
        
        Args:
            puzzle_id: ID del puzzle oficial (ej: '0520fde7')
            
        Returns:
            Resultados de evaluación
        """
        puzzle_data = self.arc_loader.get_puzzle_by_id(puzzle_id)
        
        if not puzzle_data:
            return {'error': f'Puzzle {puzzle_id} no encontrado en dataset oficial'}
        
        logger.info(f"🎯 Resolviendo puzzle específico: {puzzle_id}")
        
        # Usar la misma lógica que solve_random_puzzle pero con puzzle específico
        # ... (implementación similar)
        
        return self.solve_puzzle_data(puzzle_id, puzzle_data)
    
    def solve_puzzle_data(self, puzzle_id: str, puzzle_data: Dict) -> Dict[str, Any]:
        """Resuelve datos de puzzle específico"""
        logger.info(f"   Ejemplos de entrenamiento: {len(puzzle_data['train'])}")
        logger.info(f"   Casos de prueba: {len(puzzle_data['test'])}")
        
        results = []
        
        for test_idx, test_case in enumerate(puzzle_data['test']):
            try:
                solution, metadata = self.swarm_solver.solve_with_swarm(
                    puzzle_data['train'], test_case['input']
                )
                
                if solution is not None:
                    evaluation = self.arc_loader.evaluate_solution(
                        puzzle_id, test_idx, solution
                    )
                    evaluation['metadata'] = metadata
                    results.append(evaluation)
                    
                    logger.info(f"📊 Test {test_idx}: Precisión = {evaluation.get('accuracy', 0):.3f}")
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                results.append({'error': str(e), 'accuracy': 0.0})
        
        return {
            'puzzle_id': puzzle_id,
            'results': results,
            'average_accuracy': np.mean([r.get('accuracy', 0) for r in results])
        }
    
    def run_comprehensive_test(self, max_puzzles: int = 5) -> Dict[str, Any]:
        """
        Ejecuta test completo con múltiples puzzles reales
        
        Args:
            max_puzzles: Máximo número de puzzles a probar
            
        Returns:
            Resultados completos del test
        """
        logger.info(f"🚀 INICIANDO TEST COMPLETO PROTEUS REAL")
        logger.info(f"   Puzzles a probar: {min(max_puzzles, len(self.arc_loader.puzzles))}")
        logger.info("="*60)
        
        all_results = []
        puzzle_ids = list(self.arc_loader.puzzles.keys())[:max_puzzles]
        
        for i, puzzle_id in enumerate(puzzle_ids, 1):
            logger.info(f"\n🔹 PUZZLE {i}/{len(puzzle_ids)}: {puzzle_id}")
            
            puzzle_data = self.arc_loader.get_puzzle_by_id(puzzle_id)
            result = self.solve_puzzle_data(puzzle_id, puzzle_data)
            all_results.append(result)
            
            logger.info(f"   📊 Precisión promedio: {result['average_accuracy']:.3f}")
        
        # Estadísticas finales
        global_accuracy = np.mean([r['average_accuracy'] for r in all_results])
        
        summary = {
            'test_type': 'Comprehensive Real ARC Test',
            'puzzles_tested': len(all_results),
            'individual_results': all_results,
            'global_average_accuracy': global_accuracy,
            'dataset_source': 'ARC Prize Official',
            'solver': 'PROTEUS Swarm Intelligence',
            'dataset_stats': self.dataset_stats
        }
        
        logger.info("\n" + "="*60)
        logger.info(f"📈 RESULTADOS FINALES PROTEUS REAL:")
        logger.info(f"   Puzzles probados: {len(all_results)}")
        logger.info(f"   Precisión global: {global_accuracy:.3f}")
        logger.info(f"   Dataset: ARC Prize Oficial")
        logger.info("="*60)
        
        return summary


def main():
    """Función principal de prueba"""
    print("\n" + "="*70)
    print("🧬 PROTEUS REAL AGENT - ARC PRIZE OFICIAL")
    print("="*70)
    
    try:
        # Inicializar agente real
        agent = ProteusRealAgent()
        
        print(f"\n📊 Dataset cargado:")
        print(f"   Puzzles: {agent.dataset_stats['total_puzzles']}")
        print(f"   Ejemplos: {agent.dataset_stats['total_train_examples']}")
        
        # Resolver puzzle aleatorio
        print(f"\n🎯 Resolviendo puzzle aleatorio...")
        result = agent.solve_random_puzzle()
        
        print(f"\n✅ RESULTADO:")
        print(f"   Puzzle: {result['puzzle_id']}")
        print(f"   Precisión: {result['average_accuracy']:.3f}")
        print(f"   Casos correctos: {result['cases_correct']}/{result['total_test_cases']}")
        
        # Test comprensivo con 3 puzzles
        print(f"\n🚀 Ejecutando test comprensivo...")
        comprehensive = agent.run_comprehensive_test(max_puzzles=3)
        
        print(f"\n📈 RESULTADOS COMPRENSIVOS:")
        print(f"   Precisión global: {comprehensive['global_average_accuracy']:.3f}")
        print(f"   Puzzles evaluados: {comprehensive['puzzles_tested']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n" + "="*70)
    print("✅ PRUEBA COMPLETADA - DATOS 100% REALES")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())