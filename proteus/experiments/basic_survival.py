"""
Experimento básico de supervivencia
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from ..world.aquatic_environment import World
from ..world.protozoa import Protozoa
from ..world.predators import LuminousPredator
from ..analysis.visualization import ProteusVisualizer
from ..analysis.metrics import NonNeuralMetrics
from ..evolution.seed_transmission import SeedTransmission
from ..evolution.selection import EmergentSelection
from ..evolution.emergence import EmergenceAnalyzer


def run_basic_survival_experiment(
    n_protozoa: int = 100,
    n_predators: int = 10,
    n_generations: int = 10,
    time_steps_per_generation: int = 1000,
    save_results: bool = True,
    visualize: bool = True
):
    """
    Ejecuta un experimento básico de supervivencia multigeneracional
    """
    print("=== PROTEUS BASIC SURVIVAL EXPERIMENT ===")
    print(f"Initial population: {n_protozoa} protozoa, {n_predators} predators")
    print(f"Generations: {n_generations}")
    print(f"Time steps per generation: {time_steps_per_generation}")
    print()
    
    # Crear mundo
    world = World(size=(1000, 1000), viscosity=0.8, temperature=20.0, topology="toroidal")
    
    # Sistemas de análisis
    metrics_analyzer = NonNeuralMetrics()
    seed_transmission = SeedTransmission(mutation_rate=0.1, crossover_rate=0.3)
    selection_analyzer = EmergentSelection()
    emergence_analyzer = EmergenceAnalyzer()
    visualizer = ProteusVisualizer(world.size)
    
    # Resultados por generación
    results = {
        'parameters': {
            'n_protozoa': n_protozoa,
            'n_predators': n_predators,
            'n_generations': n_generations,
            'time_steps_per_generation': time_steps_per_generation
        },
        'generations': []
    }
    
    # Población inicial sin herencia
    initial_protozoa = [
        Protozoa(
            position=np.random.uniform(0, world.size[0], size=2),
            seed=None
        )
        for _ in range(n_protozoa)
    ]
    
    # Depredadores
    predators = [
        LuminousPredator(
            position=np.random.uniform(0, world.size[0], size=2),
            attack_frequency=0.1,
            light_radius=50
        )
        for _ in range(n_predators)
    ]
    
    # Ejecutar generaciones
    all_seeds = []
    
    for generation in range(n_generations):
        print(f"\n--- Generation {generation} ---")
        
        # Preparar organismos para esta generación
        if generation == 0:
            current_protozoa = initial_protozoa
        else:
            # Generar nueva generación desde semillas
            parent_seeds = seed_transmission.select_parents(all_seeds, selection_pressure=0.5)
            offspring_seeds = seed_transmission.generate_offspring_seeds(parent_seeds, n_protozoa)
            
            current_protozoa = [
                Protozoa(
                    position=np.random.uniform(0, world.size[0], size=2),
                    seed=seed
                )
                for seed in offspring_seeds
            ]
        
        # Simular generación
        survivors = world.simulate(current_protozoa, predators, time_steps_per_generation)
        
        print(f"Survivors: {len(survivors)}/{n_protozoa}")
        
        # Registrar supervivientes
        for survivor in survivors:
            selection_analyzer.record_survival(survivor, world.time)
        
        # Recolectar semillas
        generation_seeds = seed_transmission.collect_seeds(survivors)
        all_seeds.extend(generation_seeds)
        
        # Análisis de la generación
        world_state = world.get_world_state()
        generation_metrics = metrics_analyzer.compute_all_metrics(world_state, survivors)
        
        # Análisis de emergencia
        collective_behavior = emergence_analyzer.analyze_collective_behavior(survivors, world.time)
        
        # Análisis de selección
        selection_analysis = selection_analyzer.analyze_selection_pressure(generation)
        
        # Análisis de semillas
        seed_analysis = seed_transmission.analyze_generation(generation, generation_seeds)
        
        # Guardar resultados de la generación
        generation_result = {
            'generation': generation,
            'n_survivors': len(survivors),
            'metrics': generation_metrics,
            'collective_behavior': collective_behavior,
            'selection_analysis': selection_analysis,
            'seed_analysis': seed_analysis
        }
        
        results['generations'].append(generation_result)
        
        # Visualización intermedia
        if visualize and generation % 5 == 0:
            fig, axes = visualizer.create_world_visualization(world)
            plt.savefig(f"generation_{generation}.png")
            plt.close(fig)
            
        # Resetear el mundo para la siguiente generación
        world.stats['births'] = 0
        world.stats['deaths'] = 0
        
    # Análisis final
    print("\n=== EXPERIMENT COMPLETE ===")
    
    # Estrategias emergentes
    strategies = selection_analyzer.identify_emerging_strategies()
    print("\nEmerging Strategies:")
    for strategy, data in list(strategies.items())[:3]:
        print(f"  {strategy}: {data['avg_survival_time']:.1f} avg survival")
    
    # Predicciones evolutivas
    predictions = selection_analyzer.predict_future_evolution(5)
    print("\nEvolutionary Predictions:")
    for behavior in predictions['emerging_behaviors']:
        print(f"  - {behavior}")
    
    # Tendencia de complejidad
    emergence_trend = emergence_analyzer.track_emergence_over_time()
    print(f"\nComplexity Trend: {emergence_trend.get('complexity_trend', 'unknown')}")
    
    # Guardar resultados
    if save_results:
        results_path = Path("experiment_results")
        results_path.mkdir(exist_ok=True)
        
        # Guardar datos JSON
        with open(results_path / "basic_survival_results.json", 'w') as f:
            # Convertir arrays numpy a listas para JSON
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
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj
                    
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Guardar banco de semillas
        seed_transmission.save_seed_bank(results_path / "seed_bank.pkl")
        
        # Crear reporte visual
        if visualize:
            evolution_data = {
                'generation_stats': seed_transmission.generation_stats,
                'phylogeny': seed_transmission.compute_phylogeny(),
                'behavior_timeline': emergence_trend.get('metrics', {}).get('emergent_behaviors', []),
                'predictions': predictions
            }
            
            final_metrics = {
                'final_generation_metrics': generation_metrics,
                'evolution_data': evolution_data,
                'emerging_strategies': strategies
            }
            
            visualizer.create_analysis_report(world, final_metrics, 
                                            results_path / "experiment_report.pdf")
    
    return results


if __name__ == "__main__":
    # Ejecutar experimento con parámetros por defecto
    results = run_basic_survival_experiment(
        n_protozoa=100,
        n_predators=10,
        n_generations=20,
        time_steps_per_generation=2000,
        save_results=True,
        visualize=True
    )