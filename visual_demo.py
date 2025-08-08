#!/usr/bin/env python3
"""
Demo visual de TOPOLIFE - Ver el mundo en acción
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

# Importar componentes de PROTEUS
from proteus import World, Protozoa, LuminousPredator
from proteus.analysis.visualization import ProteusVisualizer

def create_visual_demo():
    """
    Crea una demostración visual del mundo TOPOLIFE
    """
    print("🧬 PROTEUS - Demo Visual")
    print("=" * 50)
    print("Creando un mundo acuático con criaturas topológicas...\n")
    
    # Crear mundo más pequeño para mejor visualización
    world_size = (600, 600)
    world = World(size=world_size, viscosity=0.8, temperature=20.0)
    
    # Crear población inicial de protozoos
    n_protozoa = 30
    protozoa = []
    
    for i in range(n_protozoa):
        # Distribuir en diferentes zonas
        if i < 10:
            # Grupo 1: esquina superior izquierda
            pos = np.random.uniform([50, 450], [150, 550])
        elif i < 20:
            # Grupo 2: centro
            pos = np.random.uniform([250, 250], [350, 350])
        else:
            # Grupo 3: esquina inferior derecha
            pos = np.random.uniform([450, 50], [550, 150])
            
        protozoa.append(Protozoa(position=pos))
    
    # Crear depredadores en posiciones estratégicas
    predators = [
        LuminousPredator(position=np.array([300, 300]), attack_frequency=0.1, light_radius=60),
        LuminousPredator(position=np.array([150, 150]), attack_frequency=0.08, light_radius=50),
        LuminousPredator(position=np.array([450, 450]), attack_frequency=0.12, light_radius=55),
    ]
    
    # Añadir al mundo
    world.add_protozoa(protozoa)
    world.add_predators(predators)
    
    # Crear visualizador
    viz = ProteusVisualizer(world_size)
    
    # Crear figura para animación
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Configurar ejes
    ax1.set_xlim(0, world_size[0])
    ax1.set_ylim(0, world_size[1])
    ax1.set_aspect('equal')
    ax1.set_title('Mundo PROTEUS - Vista Principal', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.set_xlim(0, world_size[0])
    ax2.set_ylim(0, world_size[1])
    ax2.set_aspect('equal')
    ax2.set_title('Campo de Luz y Peligro', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Elementos para actualizar
    protozoa_circles = []
    predator_circles = []
    light_circles = []
    trajectory_lines = []
    
    # Texto de estadísticas
    stats_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        """Inicializa la animación"""
        return []
    
    def update_frame(frame):
        """Actualiza cada frame de la animación"""
        # Limpiar elementos anteriores
        for circle in protozoa_circles + predator_circles + light_circles:
            circle.remove()
        protozoa_circles.clear()
        predator_circles.clear()
        light_circles.clear()
        
        for line in trajectory_lines:
            line.remove()
        trajectory_lines.clear()
        
        # Simular un paso
        world._update_fields()
        world._update_predators()
        world._update_protozoa()
        world._check_reproduction()
        world.time += world.dt
        
        # Vista principal (ax1)
        # Dibujar protozoos
        for proto in world.protozoa:
            if proto.alive:
                # Color basado en generación
                color_intensity = min(1.0, proto.generation / 5)
                color = plt.cm.Blues(0.3 + color_intensity * 0.7)
                
                # Tamaño basado en edad
                size = 5 + min(10, proto.age / 20)
                
                circle = Circle(proto.position, size, 
                              color=color, alpha=0.8, 
                              edgecolor='darkblue', linewidth=1)
                ax1.add_patch(circle)
                protozoa_circles.append(circle)
                
                # Mostrar trayectoria reciente
                if len(proto.state.trajectory_history) > 20:
                    recent_traj = np.array(proto.state.trajectory_history[-30:])
                    line = ax1.plot(recent_traj[:, 0], recent_traj[:, 1], 
                                   'c-', alpha=0.3, linewidth=0.8)[0]
                    trajectory_lines.append(line)
        
        # Dibujar depredadores
        for pred in world.predators:
            # Depredador
            pred_circle = Circle(pred.position, 12, 
                               color='red', alpha=0.9, 
                               edgecolor='darkred', linewidth=2)
            ax1.add_patch(pred_circle)
            predator_circles.append(pred_circle)
            
            # Zona de luz si está atacando
            if pred.is_attacking:
                light_circle = Circle(pred.position, pred.light_radius,
                                    color='yellow', alpha=0.2, 
                                    linestyle='--', fill=True)
                ax1.add_patch(light_circle)
                light_circles.append(light_circle)
        
        # Vista de campos (ax2)
        ax2.clear()
        ax2.set_xlim(0, world_size[0])
        ax2.set_ylim(0, world_size[1])
        ax2.set_title('Campo de Luz y Peligro', fontsize=14, fontweight='bold')
        
        # Mostrar campo de luz
        light_field = world.field_dynamics.light_field
        im = ax2.imshow(light_field.T, cmap='hot', alpha=0.8,
                       extent=[0, world_size[0], 0, world_size[1]],
                       origin='lower', vmin=0, vmax=1)
        
        # Superponer posiciones
        if world.protozoa:
            positions = np.array([p.position for p in world.protozoa if p.alive])
            if len(positions) > 0:
                ax2.scatter(positions[:, 0], positions[:, 1], 
                           c='cyan', s=30, alpha=0.8, edgecolors='blue')
        
        # Actualizar estadísticas
        n_alive = len([p for p in world.protozoa if p.alive])
        avg_age = np.mean([p.age for p in world.protozoa if p.alive]) if n_alive > 0 else 0
        max_gen = max([p.generation for p in world.protozoa]) if world.protozoa else 0
        
        stats_text.set_text(
            f'Tiempo: {world.time:.1f}\n'
            f'Población: {n_alive}\n'
            f'Edad promedio: {avg_age:.1f}\n'
            f'Generación máx: {max_gen}\n'
            f'Nacimientos: {world.stats["births"]}\n'
            f'Muertes: {world.stats["deaths"]}'
        )
        
        # Cada 100 frames, mostrar mensaje
        if frame % 100 == 0:
            print(f"Frame {frame}: {n_alive} protozoos vivos, "
                  f"generación máxima: {max_gen}")
        
        return protozoa_circles + predator_circles + light_circles + trajectory_lines + [stats_text]
    
    # Crear animación
    print("Ejecutando simulación visual...")
    print("(Cierra la ventana para terminar)\n")
    
    anim = animation.FuncAnimation(fig, update_frame, init_func=init,
                                  frames=2000, interval=50, blit=True)
    
    # Guardar primeros frames como imágenes
    print("Guardando capturas...")
    
    # Frame inicial
    update_frame(0)
    plt.savefig('proteus_frame_000.png', dpi=150, bbox_inches='tight')
    
    # Simular 50 pasos
    for i in range(50):
        update_frame(i)
    plt.savefig('proteus_frame_050.png', dpi=150, bbox_inches='tight')
    
    # Simular 100 pasos más
    for i in range(50, 150):
        update_frame(i)
    plt.savefig('proteus_frame_150.png', dpi=150, bbox_inches='tight')
    
    print("\nCapturas guardadas:")
    print("  - proteus_frame_000.png (estado inicial)")
    print("  - proteus_frame_050.png (50 pasos)")
    print("  - proteus_frame_150.png (150 pasos)")
    
    # Mostrar la animación
    plt.show()
    
    return world, viz


def create_static_visualization():
    """
    Crea una visualización estática detallada
    """
    print("\nCreando visualización estática detallada...")
    
    # Simular brevemente para tener datos
    world = World(size=(800, 800))
    
    # Crear organismos con diferentes estrategias
    protozoa = []
    
    # Grupo 1: Exploradores (movimiento amplio)
    for i in range(10):
        p = Protozoa(position=np.random.uniform([100, 100], [200, 200]))
        p.movement_bias = np.array([0.5, 0.5])
        protozoa.append(p)
    
    # Grupo 2: Locales (movimiento limitado)
    for i in range(10):
        p = Protozoa(position=np.random.uniform([600, 600], [700, 700]))
        p.movement_bias = np.array([-0.3, -0.3])
        protozoa.append(p)
    
    # Grupo 3: Erráticos
    for i in range(10):
        p = Protozoa(position=np.random.uniform([350, 350], [450, 450]))
        p.movement_bias = np.random.normal(0, 0.5, size=2)
        protozoa.append(p)
    
    # Depredadores
    predators = [
        LuminousPredator(position=np.array([400, 400]), light_radius=80),
        LuminousPredator(position=np.array([200, 600]), light_radius=60),
        LuminousPredator(position=np.array([600, 200]), light_radius=70),
    ]
    
    world.add_protozoa(protozoa)
    world.add_predators(predators)
    
    # Simular 200 pasos
    for _ in range(200):
        world._update_fields()
        world._update_predators() 
        world._update_protozoa()
        
    # Crear visualización completa
    viz = ProteusVisualizer(world.size)
    fig, axes = viz.create_world_visualization(world)
    
    plt.savefig('proteus_analysis.png', dpi=200, bbox_inches='tight')
    print("Visualización guardada como: proteus_analysis.png")
    
    plt.show()
    

if __name__ == "__main__":
    # Ejecutar demo visual
    world, viz = create_visual_demo()
    
    # Crear visualización estática adicional
    create_static_visualization()