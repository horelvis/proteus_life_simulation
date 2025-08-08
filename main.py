#!/usr/bin/env python3
"""
PROTEUS - Computación sin Neuronas
Main simulation script
"""

import argparse
import numpy as np
from pathlib import Path
import sys

from proteus import World, Protozoa, LuminousPredator
from proteus.experiments.basic_survival import run_basic_survival_experiment
from proteus.analysis.visualization import TopoLifeVisualizer
from proteus.analysis.metrics import NonNeuralMetrics


def run_quick_demo():
    """
    Ejecuta una demostración rápida del sistema
    """
    print("🧬 PROTEUS QUICK DEMO")
    print("=" * 50)
    print("Creating a world with topological creatures...")
    print()
    
    # Crear mundo pequeño para demo
    world = World(size=(500, 500), viscosity=0.8, temperature=20.0)
    
    # Población inicial
    protozoa = [
        Protozoa(position=np.random.uniform(50, 450, size=2))
        for _ in range(20)
    ]
    
    # Algunos depredadores
    predators = [
        LuminousPredator(
            position=np.random.uniform(100, 400, size=2),
            attack_frequency=0.1,
            light_radius=40
        )
        for _ in range(3)
    ]
    
    # Añadir al mundo
    world.add_protozoa(protozoa)
    world.add_predators(predators)
    
    # Visualizador
    viz = ProteusVisualizer(world.size)
    
    print("Simulating 100 time steps...")
    
    # Simular algunos pasos
    for step in range(100):
        world._update_fields()
        world._update_predators()
        world._update_protozoa()
        
        if step % 20 == 0:
            print(f"Step {step}: {len([p for p in world.protozoa if p.alive])} protozoa alive")
    
    # Visualizar resultado
    print("\nCreating visualization...")
    fig, axes = viz.create_world_visualization(world)
    fig.savefig("demo_world.png")
    print("Visualization saved to demo_world.png")
    
    # Mostrar algunas métricas
    metrics = NonNeuralMetrics()
    world_state = world.get_world_state()
    current_metrics = metrics.compute_all_metrics(world_state, world.protozoa)
    
    print("\nKey Metrics:")
    print(f"  Mean age: {current_metrics['survival_metrics']['mean_age']:.1f}")
    print(f"  Path complexity: {current_metrics['topological_metrics']['path_complexity']:.3f}")
    print(f"  Emergence score: {current_metrics['emergence_metrics']['emergence_score']:.3f}")
    

def run_evolution_experiment(args):
    """
    Ejecuta un experimento evolutivo completo
    """
    print("🧬 PROTEUS EVOLUTION EXPERIMENT")
    print("=" * 50)
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Predators: {args.predators}")
    print(f"Time steps per generation: {args.timesteps}")
    print()
    
    results = run_basic_survival_experiment(
        n_protozoa=args.population,
        n_predators=args.predators,
        n_generations=args.generations,
        time_steps_per_generation=args.timesteps,
        save_results=args.save,
        visualize=args.visualize
    )
    
    if args.save:
        print(f"\nResults saved to experiment_results/")
        

def run_interactive_simulation(args):
    """
    Ejecuta una simulación interactiva
    """
    print("🧬 PROTEUS INTERACTIVE SIMULATION")
    print("=" * 50)
    print("Creating world...")
    
    # Crear mundo
    world = World(size=(1000, 1000))
    
    # Población inicial
    if args.seed_file:
        print(f"Loading seeds from {args.seed_file}")
        # TODO: Implementar carga de semillas
        protozoa = [
            Protozoa(position=np.random.uniform(0, 1000, size=2))
            for _ in range(args.population)
        ]
    else:
        protozoa = [
            Protozoa(position=np.random.uniform(0, 1000, size=2))
            for _ in range(args.population)
        ]
    
    # Depredadores
    predators = [
        LuminousPredator(
            position=np.random.uniform(0, 1000, size=2),
            attack_frequency=0.1
        )
        for _ in range(args.predators)
    ]
    
    world.add_protozoa(protozoa)
    world.add_predators(predators)
    
    # Crear animación
    print("Creating animation...")
    viz = ProteusVisualizer(world.size)
    
    if args.output:
        viz.save_animation(world, args.output, frames=args.frames, interval=50)
    else:
        # Mostrar en vivo
        import matplotlib.pyplot as plt
        anim = world.animate(frames=args.frames)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="PROTEUS - Intelligence without neurons through topological dynamics"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demonstration')
    
    # Evolution experiment
    evo_parser = subparsers.add_parser('evolve', help='Run evolution experiment')
    evo_parser.add_argument('-g', '--generations', type=int, default=20,
                           help='Number of generations (default: 20)')
    evo_parser.add_argument('-p', '--population', type=int, default=100,
                           help='Initial population size (default: 100)')
    evo_parser.add_argument('-d', '--predators', type=int, default=10,
                           help='Number of predators (default: 10)')
    evo_parser.add_argument('-t', '--timesteps', type=int, default=1000,
                           help='Time steps per generation (default: 1000)')
    evo_parser.add_argument('--no-save', dest='save', action='store_false',
                           help='Do not save results')
    evo_parser.add_argument('--no-viz', dest='visualize', action='store_false',
                           help='Disable visualization')
    
    # Interactive simulation
    sim_parser = subparsers.add_parser('simulate', help='Run interactive simulation')
    sim_parser.add_argument('-p', '--population', type=int, default=50,
                           help='Population size (default: 50)')
    sim_parser.add_argument('-d', '--predators', type=int, default=5,
                           help='Number of predators (default: 5)')
    sim_parser.add_argument('-f', '--frames', type=int, default=500,
                           help='Number of frames to simulate (default: 500)')
    sim_parser.add_argument('-o', '--output', type=str,
                           help='Output file for animation (e.g., simulation.gif)')
    sim_parser.add_argument('--seed-file', type=str,
                           help='Load initial seeds from file')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_quick_demo()
    elif args.command == 'evolve':
        run_evolution_experiment(args)
    elif args.command == 'simulate':
        run_interactive_simulation(args)
    else:
        print("PROTEUS - Computation without neurons")
        print()
        print("Usage:")
        print("  python main.py demo              # Quick demonstration")
        print("  python main.py evolve            # Run evolution experiment")
        print("  python main.py simulate          # Interactive simulation")
        print()
        print("Use -h with any command for more options")
        

if __name__ == "__main__":
    main()