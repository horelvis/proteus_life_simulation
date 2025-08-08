#!/usr/bin/env python3
"""
Demo visual de TOPOLIFE - Versión para Docker
Genera imágenes estáticas en lugar de animaciones interactivas
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para Docker
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Importar componentes de PROTEUS
from proteus import World, Protozoa, LuminousPredator
from proteus.analysis.visualization import ProteusVisualizer

def ensure_output_dir():
    """Asegura que existe el directorio de salida"""
    os.makedirs('output', exist_ok=True)

def create_visual_sequence():
    """
    Crea una secuencia de imágenes mostrando la evolución del mundo
    """
    print("🧬 PROTEUS - Demo Visual (Docker)")
    print("=" * 50)
    print("Generando visualizaciones...\n")
    
    ensure_output_dir()
    
    # Crear mundo
    world_size = (600, 600)
    world = World(size=world_size, viscosity=0.8, temperature=20.0)
    
    # Crear población inicial
    n_protozoa = 40
    protozoa = []
    
    # Distribuir en grupos
    for i in range(n_protozoa):
        if i < 15:
            # Grupo superior izquierdo
            pos = np.random.uniform([50, 450], [150, 550])
        elif i < 30:
            # Grupo centro
            pos = np.random.uniform([250, 250], [350, 350])
        else:
            # Grupo inferior derecho
            pos = np.random.uniform([450, 50], [550, 150])
            
        protozoa.append(Protozoa(position=pos))
    
    # Crear depredadores
    predators = [
        LuminousPredator(position=np.array([300, 300]), attack_frequency=0.1, light_radius=60),
        LuminousPredator(position=np.array([150, 150]), attack_frequency=0.08, light_radius=50),
        LuminousPredator(position=np.array([450, 450]), attack_frequency=0.12, light_radius=55),
    ]
    
    world.add_protozoa(protozoa)
    world.add_predators(predators)
    
    # Crear visualizador
    viz = ProteusVisualizer(world_size)
    
    # Generar secuencia de imágenes
    time_points = [0, 50, 100, 200, 500, 1000]
    
    for target_time in time_points:
        print(f"Generando frame en t={target_time}...")
        
        # Simular hasta el tiempo objetivo
        while world.time < target_time * world.dt:
            world._update_fields()
            world._update_predators()
            world._update_protozoa()
            world._check_reproduction()
            world.time += world.dt
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Panel 1: Vista del mundo
        ax1.set_xlim(0, world_size[0])
        ax1.set_ylim(0, world_size[1])
        ax1.set_aspect('equal')
        ax1.set_title(f'PROTEUS - t={target_time} (Población: {len([p for p in world.protozoa if p.alive])})', 
                     fontsize=14, fontweight='bold')
        
        # Fondo de nutrientes
        nutrient_field = world.field_dynamics.nutrient_field
        im1 = ax1.imshow(nutrient_field.T, cmap='Greens', alpha=0.3,
                        extent=[0, world_size[0], 0, world_size[1]],
                        origin='lower', vmin=0, vmax=0.5)
        
        # Dibujar protozoos
        for proto in world.protozoa:
            if proto.alive:
                # Color por generación
                color_intensity = min(1.0, proto.generation / 5)
                color = plt.cm.Blues(0.3 + color_intensity * 0.7)
                
                # Tamaño por edad
                size = 5 + min(10, proto.age / 20)
                
                circle = Circle(proto.position, size, 
                              color=color, alpha=0.8, 
                              edgecolor='darkblue', linewidth=1)
                ax1.add_patch(circle)
                
                # Trayectoria reciente
                if len(proto.state.trajectory_history) > 20:
                    recent_traj = np.array(proto.state.trajectory_history[-40:])
                    ax1.plot(recent_traj[:, 0], recent_traj[:, 1], 
                            color=color, alpha=0.4, linewidth=0.8)
        
        # Dibujar depredadores
        for pred in world.predators:
            pred_circle = Circle(pred.position, 12, 
                               color='red', alpha=0.9, 
                               edgecolor='darkred', linewidth=2)
            ax1.add_patch(pred_circle)
            
            if pred.is_attacking:
                light_circle = Circle(pred.position, pred.light_radius,
                                    color='yellow', alpha=0.2, 
                                    linestyle='--', fill=True)
                ax1.add_patch(light_circle)
        
        # Panel 2: Campo de luz y peligro
        ax2.set_xlim(0, world_size[0])
        ax2.set_ylim(0, world_size[1])
        ax2.set_aspect('equal')
        ax2.set_title('Campo de Luz (Peligro)', fontsize=14, fontweight='bold')
        
        # Campo de luz
        light_field = world.field_dynamics.light_field
        im2 = ax2.imshow(light_field.T, cmap='hot', 
                        extent=[0, world_size[0], 0, world_size[1]],
                        origin='lower', vmin=0, vmax=1)
        
        # Colorbar
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Posiciones de organismos
        if world.protozoa:
            positions = np.array([p.position for p in world.protozoa if p.alive])
            if len(positions) > 0:
                ax2.scatter(positions[:, 0], positions[:, 1], 
                           c='cyan', s=30, alpha=0.8, edgecolors='blue')
        
        # Estadísticas
        n_alive = len([p for p in world.protozoa if p.alive])
        avg_age = np.mean([p.age for p in world.protozoa if p.alive]) if n_alive > 0 else 0
        max_gen = max([p.generation for p in world.protozoa]) if world.protozoa else 0
        
        stats_text = f'Vivos: {n_alive} | Edad promedio: {avg_age:.1f} | Gen máx: {max_gen}'
        fig.suptitle(stats_text, fontsize=12)
        
        # Guardar
        filename = f'output/proteus_t{target_time:04d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  → Guardado: {filename}")
        print(f"     Población: {n_alive}, Generación máx: {max_gen}")
    
    print("\n✓ Secuencia completa generada en output/")
    
    # Crear visualización de resumen
    create_summary_visualization(world, viz)


def create_summary_visualization(world, viz):
    """
    Crea una visualización de resumen con múltiples paneles
    """
    print("\nGenerando visualización de resumen...")
    
    fig, axes = viz.create_world_visualization(world)
    plt.savefig('output/proteus_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Resumen guardado: output/proteus_summary.png")


def create_evolution_timeline():
    """
    Crea una visualización de la evolución a través del tiempo
    """
    print("\nGenerando línea temporal evolutiva...")
    
    ensure_output_dir()
    
    # Configuración
    n_snapshots = 6
    steps_between = 200
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Crear nuevo mundo para cada snapshot
    for i in range(n_snapshots):
        ax = axes[i]
        
        # Crear mundo fresco
        world = World(size=(400, 400))
        
        # Población inicial
        if i == 0:
            # Primera generación sin herencia
            protozoa = [
                Protozoa(position=np.random.uniform(50, 350, size=2))
                for _ in range(30)
            ]
        else:
            # Generaciones posteriores (simulado)
            protozoa = [
                Protozoa(position=np.random.uniform(50, 350, size=2))
                for _ in range(20 + i * 2)
            ]
            # Simular algo de herencia
            for p in protozoa:
                p.generation = i
                p.movement_bias = np.random.normal(0, 0.1 * i, size=2)
        
        predators = [
            LuminousPredator(position=np.random.uniform(100, 300, size=2))
            for _ in range(3)
        ]
        
        world.add_protozoa(protozoa)
        world.add_predators(predators)
        
        # Simular
        for _ in range(steps_between * i):
            world._update_fields()
            world._update_predators()
            world._update_protozoa()
            if np.random.random() < 0.01:  # Reproducción ocasional
                world._check_reproduction()
        
        # Visualizar
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_aspect('equal')
        ax.set_title(f'Generación ~{i}', fontsize=12, fontweight='bold')
        
        # Dibujar organismos
        for proto in world.protozoa[:20]:  # Limitar para claridad
            if proto.alive:
                color = plt.cm.viridis(i / n_snapshots)
                circle = Circle(proto.position, 8, color=color, alpha=0.7)
                ax.add_patch(circle)
                
                # Trayectoria
                if len(proto.state.trajectory_history) > 10:
                    traj = np.array(proto.state.trajectory_history[-30:])
                    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3, linewidth=1)
        
        # Depredadores
        for pred in world.predators:
            ax.add_patch(Circle(pred.position, 10, color='red', alpha=0.8))
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('PROTEUS: Evolución de Comportamiento Sin Neuronas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/proteus_evolution_timeline.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Línea temporal guardada: output/proteus_evolution_timeline.png")


if __name__ == "__main__":
    # Ejecutar todas las visualizaciones
    create_visual_sequence()
    create_evolution_timeline()
    
    print("\n🎉 ¡Demo visual completa!")
    print("Revisa el directorio 'output/' para ver las imágenes generadas.")