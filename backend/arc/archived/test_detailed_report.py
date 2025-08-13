#!/usr/bin/env python3
"""
Script de prueba detallada para generar informe completo con evidencia
del sistema PROTEUS AGI
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/proteus_detailed_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.arc_swarm_solver_improved import ARCSwarmSolverImproved

class ProteusTestReport:
    """Generador de informe detallado con evidencia"""
    
    def __init__(self):
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'system': 'PROTEUS AGI - Swarm Intelligence',
            'tests': [],
            'metrics': {},
            'agent_specialization': {},
            'memory_evolution': []
        }
        
    def run_detailed_test(self):
        """Ejecuta test detallado con múltiples configuraciones"""
        logger.info("="*80)
        logger.info("INICIANDO PRUEBA DETALLADA DEL SISTEMA PROTEUS AGI")
        logger.info("="*80)
        
        # Configuraciones de prueba
        test_configs = [
            {
                'name': 'Small Population - Quick Evolution',
                'population_size': 10,
                'generations': 3,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'elite_size': 2
            },
            {
                'name': 'Standard Configuration',
                'population_size': 30,
                'generations': 5,
                'mutation_rate': 0.35,
                'crossover_rate': 0.7,
                'elite_size': 5
            },
            {
                'name': 'Large Population - Deep Evolution',
                'population_size': 50,
                'generations': 8,
                'mutation_rate': 0.4,
                'crossover_rate': 0.8,
                'elite_size': 8
            }
        ]
        
        # Casos de prueba ARC
        test_cases = [
            {
                'name': 'Simple Color Transformation',
                'train': [
                    {
                        'input': [[1, 2], [3, 4]],
                        'output': [[2, 3], [4, 5]]
                    },
                    {
                        'input': [[5, 6], [7, 8]],
                        'output': [[6, 7], [8, 9]]
                    }
                ],
                'test': [[2, 3], [4, 5]]
            },
            {
                'name': 'Pattern Replication',
                'train': [
                    {
                        'input': [[1, 0], [0, 1]],
                        'output': [[1, 0, 1, 0], [0, 1, 0, 1]]
                    }
                ],
                'test': [[2, 0], [0, 2]]
            },
            {
                'name': 'Symmetry Detection',
                'train': [
                    {
                        'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        'output': [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
                    }
                ],
                'test': [[2, 4, 6], [1, 3, 5], [8, 7, 9]]
            }
        ]
        
        # Ejecutar pruebas para cada configuración
        for config in test_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"CONFIGURACIÓN: {config['name']}")
            logger.info(f"{'='*60}")
            
            solver = ARCSwarmSolverImproved(
                population_size=config['population_size'],
                generations=config['generations'],
                mutation_rate=config['mutation_rate'],
                crossover_rate=config['crossover_rate'],
                elite_size=config['elite_size']
            )
            
            config_results = {
                'config_name': config['name'],
                'parameters': config,
                'test_results': []
            }
            
            for test_case in test_cases:
                logger.info(f"\n📋 Caso de prueba: {test_case['name']}")
                
                # Convertir a numpy arrays
                train_examples = []
                for ex in test_case['train']:
                    train_examples.append({
                        'input': np.array(ex['input']),
                        'output': np.array(ex['output'])
                    })
                test_input = np.array(test_case['test'])
                
                # Resolver con el enjambre
                solution, metadata = solver.solve_with_swarm(train_examples, test_input)
                
                # Recopilar métricas detalladas
                test_result = {
                    'test_name': test_case['name'],
                    'solution_shape': solution.shape if solution is not None else None,
                    'best_fitness': metadata.get('best_fitness', 0),
                    'alive_agents': metadata.get('alive_agents', 0),
                    'total_agents': config['population_size'],
                    'specialization_distribution': self._get_specialization_stats(solver),
                    'memory_size': len(solver.shared_memory.successful_rules),
                    'successful_chains': len(solver.shared_memory.successful_chains),
                    'generations_completed': config['generations']
                }
                
                config_results['test_results'].append(test_result)
                
                # Log detallado
                logger.info(f"   ✅ Fitness alcanzado: {test_result['best_fitness']:.3f}")
                logger.info(f"   👥 Agentes vivos: {test_result['alive_agents']}/{test_result['total_agents']}")
                logger.info(f"   🧬 Distribución de especialización:")
                for spec, count in test_result['specialization_distribution'].items():
                    logger.info(f"      - {spec}: {count} agentes")
                logger.info(f"   💾 Reglas en memoria: {test_result['memory_size']}")
                logger.info(f"   ⛓️ Cadenas exitosas: {test_result['successful_chains']}")
            
            self.report_data['tests'].append(config_results)
        
        # Generar estadísticas globales
        self._generate_global_metrics()
        
        # Guardar informe
        self._save_report()
        
        return self.report_data
    
    def _get_specialization_stats(self, solver) -> Dict[str, int]:
        """Obtiene estadísticas de especialización de agentes"""
        stats = {}
        for agent in solver.agents:
            if agent.alive:
                spec = agent.specialization or 'unknown'
                stats[spec] = stats.get(spec, 0) + 1
        return stats
    
    def _generate_global_metrics(self):
        """Genera métricas globales del sistema"""
        all_fitness = []
        all_alive_ratios = []
        specialization_totals = {}
        
        for test_config in self.report_data['tests']:
            for test_result in test_config['test_results']:
                all_fitness.append(test_result['best_fitness'])
                alive_ratio = test_result['alive_agents'] / test_result['total_agents']
                all_alive_ratios.append(alive_ratio)
                
                for spec, count in test_result['specialization_distribution'].items():
                    specialization_totals[spec] = specialization_totals.get(spec, 0) + count
        
        self.report_data['metrics'] = {
            'average_fitness': np.mean(all_fitness) if all_fitness else 0,
            'max_fitness': max(all_fitness) if all_fitness else 0,
            'min_fitness': min(all_fitness) if all_fitness else 0,
            'std_fitness': np.std(all_fitness) if all_fitness else 0,
            'average_survival_rate': np.mean(all_alive_ratios) if all_alive_ratios else 0,
            'dominant_specialization': max(specialization_totals.items(), key=lambda x: x[1])[0] if specialization_totals else 'none',
            'specialization_diversity': len(specialization_totals)
        }
        
        self.report_data['agent_specialization'] = specialization_totals
    
    def _save_report(self):
        """Guarda el informe en formato JSON y texto"""
        # Guardar JSON
        with open('/tmp/proteus_report.json', 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        # Generar informe en texto
        with open('/tmp/proteus_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("INFORME DETALLADO DEL SISTEMA PROTEUS AGI\n")
            f.write("="*80 + "\n\n")
            f.write(f"Fecha: {self.report_data['timestamp']}\n")
            f.write(f"Sistema: {self.report_data['system']}\n\n")
            
            f.write("MÉTRICAS GLOBALES\n")
            f.write("-"*40 + "\n")
            metrics = self.report_data['metrics']
            f.write(f"Fitness Promedio: {metrics['average_fitness']:.3f}\n")
            f.write(f"Fitness Máximo: {metrics['max_fitness']:.3f}\n")
            f.write(f"Fitness Mínimo: {metrics['min_fitness']:.3f}\n")
            f.write(f"Desviación Estándar: {metrics['std_fitness']:.3f}\n")
            f.write(f"Tasa de Supervivencia Promedio: {metrics['average_survival_rate']:.2%}\n")
            f.write(f"Especialización Dominante: {metrics['dominant_specialization']}\n")
            f.write(f"Diversidad de Especialización: {metrics['specialization_diversity']} tipos\n\n")
            
            f.write("DISTRIBUCIÓN DE ESPECIALIZACIÓN\n")
            f.write("-"*40 + "\n")
            for spec, count in self.report_data['agent_specialization'].items():
                f.write(f"{spec:15} : {count:4} agentes\n")
            f.write("\n")
            
            f.write("RESULTADOS POR CONFIGURACIÓN\n")
            f.write("="*80 + "\n")
            for config in self.report_data['tests']:
                f.write(f"\n{config['config_name']}\n")
                f.write("-"*40 + "\n")
                f.write(f"Población: {config['parameters']['population_size']}\n")
                f.write(f"Generaciones: {config['parameters']['generations']}\n")
                f.write(f"Tasa de Mutación: {config['parameters']['mutation_rate']}\n")
                f.write(f"Tasa de Crossover: {config['parameters']['crossover_rate']}\n")
                f.write(f"Elite: {config['parameters']['elite_size']}\n\n")
                
                for test in config['test_results']:
                    f.write(f"  Prueba: {test['test_name']}\n")
                    f.write(f"    - Fitness: {test['best_fitness']:.3f}\n")
                    f.write(f"    - Agentes Vivos: {test['alive_agents']}/{test['total_agents']}\n")
                    f.write(f"    - Reglas en Memoria: {test['memory_size']}\n")
                    f.write(f"    - Cadenas Exitosas: {test['successful_chains']}\n")
                    f.write("\n")
        
        logger.info("\n📊 Informe guardado en:")
        logger.info("   - /tmp/proteus_report.json (datos completos)")
        logger.info("   - /tmp/proteus_report.txt (resumen legible)")
        logger.info("   - /tmp/proteus_detailed_report.log (log completo)")


def main():
    """Función principal"""
    print("\n" + "="*80)
    print("🧬 PROTEUS AGI - GENERACIÓN DE INFORME DETALLADO")
    print("="*80 + "\n")
    
    tester = ProteusTestReport()
    report = tester.run_detailed_test()
    
    print("\n" + "="*80)
    print("✅ INFORME COMPLETADO")
    print("="*80)
    print(f"\nMétricas Principales:")
    print(f"  - Fitness Promedio: {report['metrics']['average_fitness']:.3f}")
    print(f"  - Fitness Máximo: {report['metrics']['max_fitness']:.3f}")
    print(f"  - Tasa de Supervivencia: {report['metrics']['average_survival_rate']:.2%}")
    print(f"  - Especialización Dominante: {report['metrics']['dominant_specialization']}")
    print(f"\nArchivos generados:")
    print(f"  - /tmp/proteus_report.json")
    print(f"  - /tmp/proteus_report.txt")
    print(f"  - /tmp/proteus_detailed_report.log")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()