#!/usr/bin/env python3
"""
Test de Razonamiento Visual con Atención Iterativa
Muestra cómo el sistema observa y comprende la escena paso a paso
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Para entorno sin display
import matplotlib.pyplot as plt
from pathlib import Path

from arc.iterative_attention_observer import IterativeAttentionObserver

def test_iterative_observation():
    """
    Demuestra el proceso de observación iterativa hasta comprensión
    """
    print("="*80)
    print("🧠 TEST DE RAZONAMIENTO VISUAL ITERATIVO")
    print("="*80)
    
    # Cargar un puzzle real
    puzzle_file = Path("/app/arc_official_cache/arc_agi_1_training_0520fde7.json")
    if puzzle_file.exists():
        with open(puzzle_file, 'r') as f:
            puzzle = json.load(f)
        
        # Usar primer ejemplo de entrenamiento
        example = puzzle['train'][0]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"\n📊 Puzzle cargado:")
        print(f"   Input shape: {input_grid.shape}")
        print(f"   Output shape: {output_grid.shape}")
    else:
        # Puzzle de ejemplo si no hay archivo
        print("\n📊 Usando puzzle de ejemplo")
        input_grid = np.array([
            [0, 0, 0, 5, 0, 1, 0],
            [0, 1, 0, 5, 1, 1, 1],
            [1, 0, 0, 5, 0, 0, 0]
        ])
        output_grid = np.array([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
    
    # Crear observador
    observer = IterativeAttentionObserver(
        max_iterations=8,
        understanding_threshold=0.85
    )
    
    print("\n👁️ INICIANDO OBSERVACIÓN ITERATIVA")
    print("-"*60)
    
    # Observar hasta comprender
    understanding = observer.observe_until_understood(
        input_grid, 
        output_grid,
        visualize=True
    )
    
    # Mostrar cadena de razonamiento
    print("\n🔗 CADENA DE RAZONAMIENTO:")
    print("-"*60)
    for i, step in enumerate(understanding.reasoning_chain, 1):
        print(f"{i}. {step}")
    
    # Mostrar comprensión final
    print("\n✅ COMPRENSIÓN FINAL:")
    print("-"*60)
    print(f"Iteraciones necesarias: {understanding.iterations_needed}")
    print(f"Confianza alcanzada: {understanding.confidence:.1%}")
    print(f"Comprensión global: {understanding.global_understanding}")
    
    # Mostrar comprensiones locales
    print("\n📍 COMPRENSIONES LOCALES:")
    for region, comp in understanding.local_understandings.items():
        print(f"   {region}: {comp}")
    
    # Mostrar relaciones encontradas
    if understanding.relationships_found:
        print("\n🔗 RELACIONES ESPACIALES:")
        for rel in understanding.relationships_found[:5]:
            print(f"   {rel['connection']}")
    
    # Crear visualización final completa
    create_final_visualization(
        input_grid, output_grid, 
        understanding.attention_path,
        understanding.confidence
    )
    
    return understanding

def create_final_visualization(input_grid, output_grid, attention_path, confidence):
    """
    Crea visualización final del proceso de comprensión
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Crear grid de subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Panel principal: Input con todos los focos
    ax_main = fig.add_subplot(gs[:2, :2])
    ax_main.imshow(input_grid, cmap='tab20', interpolation='nearest')
    ax_main.set_title('Proceso de Observación Completo', fontsize=14, fontweight='bold')
    
    # Dibujar todos los focos con colores por iteración
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attention_path)))
    for focus, color in zip(attention_path, colors):
        y, x = focus.position
        circle = plt.Circle((x, y), focus.radius, 
                          fill=False, edgecolor=color,
                          linewidth=2, alpha=0.7)
        ax_main.add_patch(circle)
        
        # Etiqueta con número
        ax_main.text(x, y, str(focus.iteration),
                   color='white', fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax_main.set_xlabel('Cada círculo representa un foco de atención')
    ax_main.grid(True, alpha=0.3)
    
    # Panel de output esperado
    ax_output = fig.add_subplot(gs[0, 2])
    ax_output.imshow(output_grid, cmap='tab20', interpolation='nearest')
    ax_output.set_title('Output Esperado')
    ax_output.grid(True, alpha=0.3)
    
    # Panel de confianza por iteración
    ax_conf = fig.add_subplot(gs[1, 2:])
    iterations = [f.iteration for f in attention_path]
    confidences = [f.confidence for f in attention_path]
    bars = ax_conf.bar(iterations, confidences, color=colors)
    ax_conf.set_xlabel('Iteración')
    ax_conf.set_ylabel('Confianza')
    ax_conf.set_title('Evolución de la Confianza')
    ax_conf.set_ylim([0, 1])
    ax_conf.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Umbral objetivo')
    ax_conf.legend()
    
    # Panel de texto con razonamiento
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    
    # Construir texto de razonamiento
    reasoning_text = "PROCESO DE COMPRENSIÓN:\n\n"
    for i, focus in enumerate(attention_path[:6], 1):  # Primeros 6 pasos
        reasoning_text += f"Paso {i}: Observé {focus.feature} en posición {focus.position}\n"
        reasoning_text += f"        Comprendí: {focus.understanding[:60]}...\n"
        reasoning_text += f"        Confianza: {focus.confidence:.1%}\n\n"
    
    if len(attention_path) > 6:
        reasoning_text += f"... ({len(attention_path)-6} observaciones más)\n\n"
    
    reasoning_text += f"\nCOMPRENSIÓN FINAL: {confidence:.1%}"
    
    ax_text.text(0.05, 0.95, reasoning_text, 
                transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Título general
    fig.suptitle(f'Análisis de Comprensión Visual - Confianza Final: {confidence:.1%}',
                fontsize=16, fontweight='bold')
    
    # Guardar
    output_path = '/tmp/visual_reasoning_complete.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📸 Visualización completa guardada en: {output_path}")
    plt.close()

def main():
    """Test principal"""
    print("\n" + "="*80)
    print("SISTEMA DE RAZONAMIENTO VISUAL CON ATENCIÓN ITERATIVA")
    print("="*80)
    print("\nEste sistema observa la escena múltiples veces,")
    print("enfocándose en diferentes áreas hasta alcanzar")
    print("comprensión completa de todas las relaciones.\n")
    
    # Ejecutar test
    understanding = test_iterative_observation()
    
    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN")
    print("="*80)
    
    print(f"\n🎯 El sistema necesitó {understanding.iterations_needed} observaciones")
    print(f"   para alcanzar {understanding.confidence:.1%} de comprensión")
    
    print("\n💡 INSIGHTS CLAVE:")
    print("   1. La atención iterativa permite comprensión gradual")
    print("   2. Cada observación se enfoca en aspectos no comprendidos")
    print("   3. El razonamiento se construye incrementalmente")
    print("   4. La visualización 2D muestra el proceso de pensamiento")
    
    print("\n✨ El OUTPUT principal NO es la solución del puzzle,")
    print("   sino el PROCESO DE RAZONAMIENTO y COMPRENSIÓN")

if __name__ == "__main__":
    main()