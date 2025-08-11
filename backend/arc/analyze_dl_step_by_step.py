#!/usr/bin/env python3
"""
Análisis paso a paso del funcionamiento del Deep Learning solver
Para entender dónde está fallando
"""

import numpy as np
import torch
import logging
from deep_learning_solver_fixed import DeepLearningARCSolver
from arc_test_puzzles import ARCTestPuzzles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_grid(grid: np.ndarray, title: str = "Grid"):
    """Visualiza un grid de forma legible"""
    logger.info(f"\n{title}:")
    logger.info("-" * (grid.shape[1] * 3 + 1))
    for row in grid:
        row_str = "|"
        for val in row:
            if val == 0:
                row_str += " . "
            else:
                row_str += f" {int(val)} "
        row_str += "|"
        logger.info(row_str)
    logger.info("-" * (grid.shape[1] * 3 + 1))

def analyze_dl_process(puzzle_name: str = "gravity_simulation"):
    """Analiza el proceso completo del DL solver paso a paso"""
    
    logger.info("="*80)
    logger.info("ANÁLISIS PASO A PASO DEL DEEP LEARNING SOLVER")
    logger.info("="*80)
    
    # Obtener el puzzle
    puzzles = ARCTestPuzzles.get_all_puzzles()
    puzzle = next((p for p in puzzles if p['name'] == puzzle_name), None)
    
    if not puzzle:
        logger.error(f"Puzzle {puzzle_name} no encontrado")
        return
    
    logger.info(f"\n🧩 Analizando puzzle: {puzzle_name}")
    logger.info(f"📝 Descripción: {puzzle['description']}")
    
    # Crear solver
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    solver = DeepLearningARCSolver(device=device)
    solver.eval()
    
    # Preparar datos
    train_examples = [
        {'input': ex['input'].tolist(), 'output': ex['output'].tolist()}
        for ex in puzzle['train']
    ]
    test_input = puzzle['test']['input']
    expected_output = puzzle['test']['output']
    
    # Mostrar ejemplos de entrenamiento
    logger.info("\n" + "="*60)
    logger.info("EJEMPLOS DE ENTRENAMIENTO")
    logger.info("="*60)
    
    for i, ex in enumerate(puzzle['train']):
        logger.info(f"\n📚 Ejemplo {i+1}:")
        visualize_grid(ex['input'], "Input")
        visualize_grid(ex['output'], "Output")
    
    # Mostrar test
    logger.info("\n" + "="*60)
    logger.info("PUZZLE DE TEST")
    logger.info("="*60)
    visualize_grid(test_input, "Test Input")
    visualize_grid(expected_output, "Expected Output")
    
    # ANÁLISIS DETALLADO DEL PROCESO
    logger.info("\n" + "="*60)
    logger.info("PROCESO DEL DEEP LEARNING SOLVER")
    logger.info("="*60)
    
    # Paso 1: Conversión a tensores
    logger.info("\n📌 PASO 1: Conversión a Tensores")
    logger.info("-"*40)
    
    first_example = train_examples[0]
    input_grid = np.array(first_example['input'])
    output_grid = np.array(first_example['output'])
    
    with torch.no_grad():
        input_tensor = solver._grid_to_tensor(input_grid)
        output_tensor = solver._grid_to_tensor(output_grid)
    
    logger.info(f"  Input shape: {input_grid.shape} → Tensor shape: {input_tensor.shape}")
    logger.info(f"  (One-hot encoding: 10 canales para valores 0-9)")
    
    # Paso 2: Detección de entidades
    logger.info("\n📌 PASO 2: Detección de Entidades")
    logger.info("-"*40)
    
    entities_input = solver._detect_entities(input_grid)
    entities_output = solver._detect_entities(output_grid)
    
    logger.info(f"  Entidades en input: {len(entities_input)}")
    for i, entity in enumerate(entities_input):
        bbox = entity.bbox
        logger.info(f"    - Entidad {i}: bbox={bbox}, anchors={len(entity.anchors)}")
        # Mostrar máscara de la entidad
        entity_grid = np.zeros_like(input_grid)
        entity_grid[entity.mask] = input_grid[entity.mask]
        visualize_grid(entity_grid, f"    Entidad {i} (input)")
    
    logger.info(f"\n  Entidades en output: {len(entities_output)}")
    for i, entity in enumerate(entities_output):
        bbox = entity.bbox
        logger.info(f"    - Entidad {i}: bbox={bbox}, anchors={len(entity.anchors)}")
    
    # Paso 3: Extracción de features con CNN
    logger.info("\n📌 PASO 3: Extracción de Features con CNN")
    logger.info("-"*40)
    
    with torch.no_grad():
        input_features = solver.encoder(input_tensor)
        output_features = solver.encoder(output_tensor)
    
    logger.info(f"  Features multi-escala extraídas:")
    for scale, feat in input_features.items():
        logger.info(f"    - {scale}: {feat.shape}")
    
    # Paso 4: Fusión multi-escala
    logger.info("\n📌 PASO 4: Fusión Multi-escala")
    logger.info("-"*40)
    
    with torch.no_grad():
        fused_input = solver.fusion(input_features)
        fused_output = solver.fusion(output_features)
    
    logger.info(f"  Features fusionadas: {fused_input.shape}")
    logger.info(f"  (Todas las escalas combinadas en una representación)")
    
    # Paso 5: ROI Pooling
    logger.info("\n📌 PASO 5: ROI Pooling de Entidades")
    logger.info("-"*40)
    
    input_rois = solver._entities_to_rois_batch([entities_input], 1)
    output_rois = solver._entities_to_rois_batch([entities_output], 1)
    
    logger.info(f"  ROIs input: {input_rois.shape if input_rois.numel() > 0 else 'None'}")
    logger.info(f"  ROIs output: {output_rois.shape if output_rois.numel() > 0 else 'None'}")
    
    if input_rois.numel() > 0:
        with torch.no_grad():
            input_pooled = solver.roi_pool(fused_input, input_rois)
            logger.info(f"  Pooled features: {input_pooled.shape}")
    
    # Paso 6: Atención y correspondencias
    logger.info("\n📌 PASO 6: Atención entre Entidades")
    logger.info("-"*40)
    
    with torch.no_grad():
        results = solver.forward(input_tensor, output_tensor)
    
    if 'logits' in results:
        logits = results['logits']
        logger.info(f"  Attention logits shape: {logits.shape}")
        logger.info(f"  (Scores de correspondencia entre entidades)")
        
        # Mostrar matriz de atención si es pequeña
        if logits.numel() > 0 and logits.shape[1] <= 5 and logits.shape[2] <= 5:
            logger.info("\n  Matriz de atención:")
            attention_matrix = torch.softmax(logits[0], dim=-1).cpu().numpy()
            for i, row in enumerate(attention_matrix):
                logger.info(f"    Entity {i}: {row}")
    
    # Paso 7: Predicción de transformación
    logger.info("\n📌 PASO 7: Predicción de Transformación")
    logger.info("-"*40)
    
    if 'transformation' in results:
        trans_logits = results['transformation']
        trans_probs = torch.softmax(trans_logits, dim=-1)
        trans_names = ['cross_expansion', 'fill', 'gravity', 'color_mapping', 
                      'rotation', 'diagonal', 'mirror', 'tile', 'scale', 'unknown']
        
        # Top 3 transformaciones predichas
        top_probs, top_indices = trans_probs[0].topk(3)
        
        logger.info("  Top 3 transformaciones predichas:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            trans_name = trans_names[idx] if idx < len(trans_names) else 'unknown'
            logger.info(f"    {i+1}. {trans_name}: {prob:.2%}")
        
        # Transformación seleccionada
        selected_idx = trans_logits.argmax(dim=-1).item()
        selected_trans = trans_names[selected_idx] if selected_idx < len(trans_names) else 'unknown'
        logger.info(f"\n  ✓ Transformación seleccionada: {selected_trans}")
    
    # Paso 8: Aplicación al test
    logger.info("\n📌 PASO 8: Aplicación al Test")
    logger.info("-"*40)
    
    # Detectar entidades en test
    test_entities = solver._detect_entities(test_input)
    logger.info(f"  Entidades detectadas en test: {len(test_entities)}")
    
    for i, entity in enumerate(test_entities):
        entity_grid = np.zeros_like(test_input)
        entity_grid[entity.mask] = test_input[entity.mask]
        visualize_grid(entity_grid, f"  Entidad {i} (test)")
    
    # Aplicar transformación
    transformation = {'type': selected_trans, 'confidence': top_probs[0].item()}
    
    logger.info(f"\n  Aplicando transformación: {selected_trans}")
    
    # Mostrar el proceso de transformación paso a paso
    if selected_trans == 'gravity':
        logger.info("  → Moviendo objetos hacia abajo...")
        logger.info("  → Apilando en el fondo...")
    elif selected_trans == 'cross_expansion':
        logger.info("  → Expandiendo desde centros de entidades...")
        logger.info("  → Creando patrón de cruz...")
    elif selected_trans == 'fill':
        logger.info("  → Detectando espacios cerrados...")
        logger.info("  → Rellenando con valores vecinos...")
    
    # Obtener predicción final
    with torch.no_grad():
        solution, steps = solver.solve_with_steps(train_examples, test_input)
    
    visualize_grid(solution, "Predicción Final")
    
    # Comparar con esperado
    logger.info("\n📌 EVALUACIÓN")
    logger.info("-"*40)
    
    exact_match = np.array_equal(solution, expected_output)
    pixel_accuracy = np.mean(solution == expected_output) if solution.shape == expected_output.shape else 0
    
    logger.info(f"  Exact Match: {'✅ YES' if exact_match else '❌ NO'}")
    logger.info(f"  Pixel Accuracy: {pixel_accuracy:.1%}")
    
    if not exact_match:
        # Mostrar diferencias
        logger.info("\n  Diferencias:")
        if solution.shape == expected_output.shape:
            diff = (solution != expected_output).astype(int)
            visualize_grid(diff, "  Píxeles diferentes (1=error)")
        else:
            logger.info(f"  Shape mismatch: {solution.shape} vs {expected_output.shape}")
    
    # ANÁLISIS DE PROBLEMAS
    logger.info("\n" + "="*60)
    logger.info("ANÁLISIS DE PROBLEMAS POTENCIALES")
    logger.info("="*60)
    
    problems = []
    
    # Problema 1: Detección de entidades
    if len(entities_input) == 0:
        problems.append("❌ No se detectaron entidades en el input")
    elif len(entities_input) != len(entities_output):
        problems.append(f"⚠️ Número diferente de entidades: {len(entities_input)} vs {len(entities_output)}")
    
    # Problema 2: Transformación incorrecta
    if not exact_match and selected_trans != 'unknown':
        problems.append(f"⚠️ La transformación '{selected_trans}' no es la correcta para este puzzle")
    
    # Problema 3: Sin entrenamiento
    problems.append("⚠️ El modelo NO está entrenado - usa pesos aleatorios")
    
    # Problema 4: Información perdida
    if len(test_entities) == 0:
        problems.append("❌ No se detectaron entidades en el test")
    
    if problems:
        logger.info("\n🔴 Problemas identificados:")
        for problem in problems:
            logger.info(f"  {problem}")
    else:
        logger.info("\n✅ No se detectaron problemas obvios")
    
    # CONCLUSIONES
    logger.info("\n" + "="*60)
    logger.info("CONCLUSIONES")
    logger.info("="*60)
    
    logger.info("""
El Deep Learning solver funciona en estos pasos:

1. **Detección de entidades**: Encuentra objetos conectados
2. **Extracción de features**: CNN procesa el grid completo
3. **ROI Pooling**: Extrae features para cada entidad
4. **Atención**: Calcula correspondencias entre entidades
5. **Predicción**: Clasifica el tipo de transformación
6. **Aplicación**: Aplica la transformación detectada

PROBLEMA PRINCIPAL:
El modelo está tratando de CLASIFICAR transformaciones predefinidas
en lugar de APRENDER la transformación real del puzzle.

Sin entrenamiento, solo puede:
- Detectar patrones simples (gravedad, expansión)
- Aplicar transformaciones hardcodeadas

NO puede:
- Aprender reglas nuevas
- Generalizar a patrones no vistos
- Crear transformaciones complejas
""")

def compare_puzzles():
    """Compara un puzzle exitoso con uno fallido"""
    logger.info("\n" + "="*80)
    logger.info("COMPARACIÓN: PUZZLE EXITOSO vs FALLIDO")
    logger.info("="*80)
    
    # Primero analizar el que funciona
    logger.info("\n✅ PUZZLE QUE FUNCIONA:")
    analyze_dl_process("gravity_simulation")
    
    # Luego uno que falla
    logger.info("\n\n❌ PUZZLE QUE FALLA:")
    analyze_dl_process("pattern_completion")

if __name__ == "__main__":
    # Analizar ambos casos
    compare_puzzles()