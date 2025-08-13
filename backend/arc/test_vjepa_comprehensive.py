#!/usr/bin/env python3
"""
Pruebas exhaustivas para V-JEPA Contrastive
Verifica que el modelo realmente aprendió representaciones útiles
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar el encoder
import sys
sys.path.append('/app/arc')
from train_vjepa_contrastive import SimpleEncoder


def generate_test_patterns():
    """Genera patrones de prueba diversos"""
    patterns = []
    labels = []
    names = []
    
    # Categoría 1: Líneas
    for angle in [0, 45, 90, 135]:
        pattern = create_lines_pattern(angle)
        patterns.append(pattern)
        labels.append(0)  # Categoría líneas
        names.append(f"Líneas_{angle}°")
    
    # Categoría 2: Formas geométricas
    for shape in ['square', 'circle', 'triangle']:
        pattern = create_shape_pattern(shape)
        patterns.append(pattern)
        labels.append(1)  # Categoría formas
        names.append(f"Forma_{shape}")
    
    # Categoría 3: Patrones periódicos
    for freq in [2, 4, 8]:
        pattern = create_periodic_pattern(freq)
        patterns.append(pattern)
        labels.append(2)  # Categoría periódicos
        names.append(f"Periódico_f{freq}")
    
    # Categoría 4: Texturas
    for texture in ['dots', 'grid', 'noise']:
        pattern = create_texture_pattern(texture)
        patterns.append(pattern)
        labels.append(3)  # Categoría texturas
        names.append(f"Textura_{texture}")
    
    # Categoría 5: Gradientes
    for direction in ['horizontal', 'vertical', 'diagonal']:
        pattern = create_gradient_pattern(direction)
        patterns.append(pattern)
        labels.append(4)  # Categoría gradientes
        names.append(f"Gradiente_{direction}")
    
    return patterns, labels, names


def create_lines_pattern(angle):
    """Crea patrón de líneas con ángulo dado"""
    pattern = np.zeros((32, 32))
    angle_rad = np.radians(angle)
    
    for i in range(0, 32, 4):
        if angle == 0:  # Horizontal
            pattern[i:i+2, :] = 1
        elif angle == 90:  # Vertical
            pattern[:, i:i+2] = 1
        elif angle == 45:  # Diagonal /
            for j in range(32):
                y = int(j - i + 16)
                if 0 <= y < 32:
                    pattern[y, j] = 1
                    if y+1 < 32:
                        pattern[y+1, j] = 1
        else:  # Diagonal \
            for j in range(32):
                y = int(j + i - 16)
                if 0 <= y < 32:
                    pattern[y, j] = 1
                    if y+1 < 32:
                        pattern[y+1, j] = 1
    
    return torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(0)


def create_shape_pattern(shape):
    """Crea patrón con forma geométrica"""
    pattern = np.zeros((32, 32))
    
    if shape == 'square':
        pattern[8:24, 8:24] = 1
    elif shape == 'circle':
        center = 16
        for i in range(32):
            for j in range(32):
                if np.sqrt((i-center)**2 + (j-center)**2) < 10:
                    pattern[i, j] = 1
    else:  # triangle
        for i in range(8, 24):
            width = (i - 8) * 2
            start = 16 - width // 2
            end = 16 + width // 2
            if start >= 0 and end <= 32:
                pattern[i, start:end] = 1
    
    return torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(0)


def create_periodic_pattern(freq):
    """Crea patrón periódico"""
    pattern = np.zeros((32, 32))
    
    for i in range(32):
        for j in range(32):
            pattern[i, j] = (np.sin(i * freq * np.pi / 32) + 
                           np.sin(j * freq * np.pi / 32)) / 2 + 0.5
    
    return torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(0)


def create_texture_pattern(texture_type):
    """Crea patrón de textura"""
    pattern = np.zeros((32, 32))
    
    if texture_type == 'dots':
        for i in range(4, 32, 8):
            for j in range(4, 32, 8):
                pattern[i-1:i+2, j-1:j+2] = 1
    elif texture_type == 'grid':
        for i in range(0, 32, 4):
            pattern[i, :] = 0.5
            pattern[:, i] = 0.5
    else:  # noise
        pattern = np.random.rand(32, 32)
    
    return torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(0)


def create_gradient_pattern(direction):
    """Crea patrón de gradiente"""
    pattern = np.zeros((32, 32))
    
    if direction == 'horizontal':
        for i in range(32):
            pattern[:, i] = i / 31
    elif direction == 'vertical':
        for i in range(32):
            pattern[i, :] = i / 31
    else:  # diagonal
        for i in range(32):
            for j in range(32):
                pattern[i, j] = (i + j) / 62
    
    return torch.FloatTensor(pattern).unsqueeze(0).unsqueeze(0)


def test_encoder_comprehensive(weights_path: str = "/app/arc/vjepa_contrastive_weights/vjepa_contrastive_final.pth"):
    """Pruebas exhaustivas del encoder"""
    
    logger.info("=== PRUEBAS EXHAUSTIVAS V-JEPA CONTRASTIVE ===")
    
    # Cargar modelo
    if not Path(weights_path).exists():
        logger.error(f"No se encontró {weights_path}")
        return
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    encoder = SimpleEncoder(input_size=32, output_dim=128)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    
    logger.info(f"Modelo cargado - Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # Generar patrones de prueba
    patterns, labels, names = generate_test_patterns()
    
    # Codificar todos los patrones
    embeddings = []
    with torch.no_grad():
        for pattern in patterns:
            embedding = encoder(pattern)
            embeddings.append(embedding.squeeze().numpy())
    
    embeddings = np.array(embeddings)
    
    # 1. PRUEBA: Matriz de similitud
    logger.info("\n1. ANÁLISIS DE SIMILITUD")
    similarity_matrix = np.zeros((len(patterns), len(patterns)))
    
    for i in range(len(patterns)):
        for j in range(len(patterns)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarity_matrix[i, j] = sim
    
    # Análisis por categorías
    categories = ['Líneas', 'Formas', 'Periódicos', 'Texturas', 'Gradientes']
    category_ranges = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)]
    
    logger.info("\nSimilitud promedio INTRA-categoría (debe ser alta):")
    for cat_idx, (cat_name, (start, end)) in enumerate(zip(categories, category_ranges)):
        if end <= len(patterns):
            intra_sim = similarity_matrix[start:end, start:end]
            avg_sim = (np.sum(intra_sim) - np.trace(intra_sim)) / (intra_sim.size - len(intra_sim))
            logger.info(f"  {cat_name}: {avg_sim:.3f}")
    
    logger.info("\nSimilitud promedio INTER-categoría (debe ser baja):")
    for i, (cat1, (s1, e1)) in enumerate(zip(categories, category_ranges)):
        for j, (cat2, (s2, e2)) in enumerate(zip(categories, category_ranges)):
            if i < j and e1 <= len(patterns) and e2 <= len(patterns):
                inter_sim = similarity_matrix[s1:e1, s2:e2]
                avg_sim = np.mean(inter_sim)
                logger.info(f"  {cat1} vs {cat2}: {avg_sim:.3f}")
    
    # 2. PRUEBA: t-SNE para visualización
    logger.info("\n2. VISUALIZACIÓN t-SNE")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Crear gráfico
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.array(labels))
    
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=[colors[i]], s=100)
        plt.annotate(names[i], (x, y), fontsize=8, alpha=0.7)
    
    plt.title('t-SNE de Embeddings V-JEPA')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/app/arc/vjepa_tsne.png', dpi=100)
    logger.info("  Gráfico guardado en /app/arc/vjepa_tsne.png")
    
    # 3. PRUEBA: Invariancia a transformaciones
    logger.info("\n3. PRUEBA DE INVARIANCIA")
    test_pattern = patterns[0]  # Líneas horizontales
    
    # Aplicar transformaciones
    transforms = {
        'Original': test_pattern,
        'Flip H': torch.flip(test_pattern, [3]),
        'Flip V': torch.flip(test_pattern, [2]),
        'Rotación 90°': torch.rot90(test_pattern, 1, [2, 3]),
        'Ruido +0.1': test_pattern + torch.randn_like(test_pattern) * 0.1,
        'Invertido': 1 - test_pattern
    }
    
    orig_embedding = encoder(test_pattern).squeeze().detach().numpy()
    
    logger.info("Similitud con transformaciones (1.0 = idéntico):")
    for trans_name, trans_pattern in transforms.items():
        trans_embedding = encoder(trans_pattern).squeeze().detach().numpy()
        sim = np.dot(orig_embedding, trans_embedding) / (
            np.linalg.norm(orig_embedding) * np.linalg.norm(trans_embedding)
        )
        logger.info(f"  {trans_name}: {sim:.3f}")
    
    # 4. PRUEBA: Distancia entre categorías
    logger.info("\n4. ANÁLISIS DE CLUSTERING")
    
    # Calcular centroides por categoría
    centroids = []
    for cat_idx, (start, end) in enumerate(category_ranges):
        if end <= len(embeddings):
            centroid = np.mean(embeddings[start:end], axis=0)
            centroids.append(centroid)
    
    # Distancia entre centroides
    logger.info("Distancia euclidiana entre centroides:")
    for i, cat1 in enumerate(categories[:len(centroids)]):
        for j, cat2 in enumerate(categories[:len(centroids)]):
            if i < j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                logger.info(f"  {cat1} - {cat2}: {dist:.3f}")
    
    # 5. EVALUACIÓN FINAL
    logger.info("\n=== EVALUACIÓN FINAL ===")
    
    # Criterios de éxito
    success_criteria = []
    
    # Criterio 1: Similitud intra-categoría > 0.7
    intra_sims = []
    for start, end in category_ranges:
        if end <= len(patterns):
            intra_sim = similarity_matrix[start:end, start:end]
            avg_sim = (np.sum(intra_sim) - np.trace(intra_sim)) / (intra_sim.size - len(intra_sim))
            intra_sims.append(avg_sim)
    
    avg_intra = np.mean(intra_sims) if intra_sims else 0
    success_criteria.append(('Similitud intra-categoría > 0.7', avg_intra > 0.7, avg_intra))
    
    # Criterio 2: Similitud inter-categoría < 0.5
    inter_sims = []
    for i, (s1, e1) in enumerate(category_ranges):
        for j, (s2, e2) in enumerate(category_ranges):
            if i < j and e1 <= len(patterns) and e2 <= len(patterns):
                inter_sim = similarity_matrix[s1:e1, s2:e2]
                inter_sims.append(np.mean(inter_sim))
    
    avg_inter = np.mean(inter_sims) if inter_sims else 1
    success_criteria.append(('Similitud inter-categoría < 0.5', avg_inter < 0.5, avg_inter))
    
    # Criterio 3: Robustez al ruido
    noise_sim = 0
    for pattern in patterns[:3]:  # Probar con algunos patrones
        orig_emb = encoder(pattern).squeeze().detach().numpy()
        noisy = pattern + torch.randn_like(pattern) * 0.1
        noisy_emb = encoder(noisy).squeeze().detach().numpy()
        sim = np.dot(orig_emb, noisy_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(noisy_emb))
        noise_sim += sim
    noise_sim /= 3
    
    success_criteria.append(('Robustez al ruido > 0.8', noise_sim > 0.8, noise_sim))
    
    # Mostrar resultados
    logger.info("\nCRITERIOS DE ÉXITO:")
    all_passed = True
    for criterion, passed, value in success_criteria:
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {criterion}: {value:.3f}")
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("\n🎉 ¡TODAS LAS PRUEBAS PASADAS! El modelo aprendió representaciones útiles.")
    else:
        logger.info("\n⚠️ Algunas pruebas fallaron. El modelo necesita más entrenamiento.")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, 
                       default="/app/arc/vjepa_contrastive_weights/vjepa_contrastive_final.pth",
                       help="Ruta a los pesos del modelo")
    
    args = parser.parse_args()
    
    success = test_encoder_comprehensive(args.weights)
    exit(0 if success else 1)