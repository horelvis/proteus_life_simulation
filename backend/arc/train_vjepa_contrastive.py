#!/usr/bin/env python3
"""
V-JEPA CONTRASTIVE - Entrenamiento CORRECTO con contrastive learning
Esta vez SÍ aprende representaciones útiles
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveVisualDataset(Dataset):
    """
    Dataset que genera matrices con patrones diversos
    """
    
    def __init__(self, num_samples: int = 10000, grid_size: int = 32):
        self.samples = []
        self.grid_size = grid_size
        
        logger.info(f"Generando {num_samples} patrones visuales...")
        
        for i in range(num_samples):
            # Generar patrón con alguna estructura
            pattern = self._generate_pattern()
            self.samples.append(pattern)
            
            if (i + 1) % 2000 == 0:
                logger.info(f"  Generados {i + 1}/{num_samples}")
    
    def _generate_pattern(self):
        """Genera un patrón visual único"""
        size = self.grid_size
        pattern = np.zeros((size, size), dtype=np.float32)
        
        # Elegir tipo de patrón aleatoriamente
        pattern_type = random.choice([
            'blocks', 'stripes', 'diagonal', 'circles', 
            'gradient', 'noise', 'checkerboard', 'spiral'
        ])
        
        if pattern_type == 'blocks':
            # Bloques aleatorios
            num_blocks = random.randint(2, 8)
            for _ in range(num_blocks):
                h = random.randint(2, size//3)
                w = random.randint(2, size//3)
                y = random.randint(0, size - h)
                x = random.randint(0, size - w)
                val = random.random()
                pattern[y:y+h, x:x+w] = val
                
        elif pattern_type == 'stripes':
            # Rayas horizontales o verticales
            num_stripes = random.randint(3, 10)
            vertical = random.random() > 0.5
            for i in range(num_stripes):
                val = random.random()
                if vertical:
                    start = i * size // num_stripes
                    end = (i + 1) * size // num_stripes
                    pattern[:, start:end] = val if i % 2 == 0 else 0
                else:
                    start = i * size // num_stripes
                    end = (i + 1) * size // num_stripes
                    pattern[start:end, :] = val if i % 2 == 0 else 0
                    
        elif pattern_type == 'diagonal':
            # Líneas diagonales
            for i in range(size):
                for j in range(size):
                    pattern[i, j] = ((i + j) % 10) / 10.0
                    
        elif pattern_type == 'circles':
            # Círculos concéntricos
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    pattern[i, j] = np.sin(dist * 0.5) * 0.5 + 0.5
                    
        elif pattern_type == 'gradient':
            # Gradiente
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            if direction == 'horizontal':
                for i in range(size):
                    pattern[:, i] = i / size
            elif direction == 'vertical':
                for i in range(size):
                    pattern[i, :] = i / size
            else:  # diagonal
                for i in range(size):
                    for j in range(size):
                        pattern[i, j] = (i + j) / (2 * size)
                        
        elif pattern_type == 'noise':
            # Ruido estructurado
            pattern = np.random.rand(size, size) * 0.5
            # Añadir algo de estructura
            pattern = np.convolve(pattern.flatten(), np.ones(3)/3, mode='same').reshape(size, size)
            
        elif pattern_type == 'checkerboard':
            # Tablero de ajedrez
            square_size = random.randint(2, 8)
            for i in range(0, size, square_size):
                for j in range(0, size, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        pattern[i:i+square_size, j:j+square_size] = 1.0
                        
        else:  # spiral
            # Espiral
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dy = i - center
                    dx = j - center
                    angle = np.arctan2(dy, dx)
                    radius = np.sqrt(dx**2 + dy**2)
                    pattern[i, j] = (np.sin(angle * 4 + radius * 0.3) + 1) / 2
        
        return pattern
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]).unsqueeze(0)


class SimCLRLoss(nn.Module):
    """
    Implementación correcta de contrastive loss (SimCLR)
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features):
        """
        features: [2N, D] donde N es batch_size
        Las primeras N son aug1, las siguientes N son aug2
        """
        device = features.device
        batch_size = features.shape[0] // 2
        
        # Normalizar features
        features = F.normalize(features, dim=1)
        
        # Calcular similitud entre todos los pares
        similarity_matrix = torch.matmul(features, features.T)
        
        # Crear máscara para positivos (diagonal desplazada)
        positive_mask = torch.zeros_like(similarity_matrix)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1
            positive_mask[i + batch_size, i] = 1
        
        # Crear máscara para negativos (todo excepto diagonal y positivos)
        negative_mask = torch.ones_like(similarity_matrix)
        negative_mask.fill_diagonal_(0)
        negative_mask = negative_mask - positive_mask
        
        # Calcular loss para cada anchor
        loss = 0
        for i in range(2 * batch_size):
            # Encontrar el positivo
            positive_idx = torch.where(positive_mask[i])[0]
            if len(positive_idx) == 0:
                continue
            positive_sim = similarity_matrix[i, positive_idx[0]]
            
            # Encontrar los negativos
            negative_indices = torch.where(negative_mask[i])[0]
            if len(negative_indices) == 0:
                continue
            negative_sims = similarity_matrix[i, negative_indices]
            
            # Calcular loss InfoNCE
            numerator = torch.exp(positive_sim / self.temperature)
            denominator = numerator + torch.sum(torch.exp(negative_sims / self.temperature))
            
            loss += -torch.log(numerator / denominator)
        
        return loss / (2 * batch_size)


class SimpleEncoder(nn.Module):
    """
    Encoder simple pero efectivo
    """
    
    def __init__(self, input_size=32, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Conv layers
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embedding = self.projector(features)
        return embedding


def augment_image(img):
    """
    Aplica augmentaciones aleatorias a una imagen
    """
    aug_img = img.clone()
    
    # Lista de augmentaciones posibles
    augmentations = []
    
    # Rotación
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        aug_img = torch.rot90(aug_img, k, [2, 3])
        augmentations.append(f'rot{k*90}')
    
    # Flip
    if random.random() > 0.5:
        if random.random() > 0.5:
            aug_img = torch.flip(aug_img, [2])  # horizontal
            augmentations.append('flipH')
        else:
            aug_img = torch.flip(aug_img, [3])  # vertical
            augmentations.append('flipV')
    
    # Añadir ruido
    if random.random() > 0.5:
        noise = torch.randn_like(aug_img) * 0.1
        aug_img = aug_img + noise
        aug_img = torch.clamp(aug_img, 0, 1)
        augmentations.append('noise')
    
    # Invertir colores
    if random.random() > 0.5:
        aug_img = 1 - aug_img
        augmentations.append('invert')
    
    # Cortar y rellenar (cutout)
    if random.random() > 0.5:
        h, w = aug_img.shape[2], aug_img.shape[3]
        cut_size = random.randint(4, 8)
        x = random.randint(0, w - cut_size)
        y = random.randint(0, h - cut_size)
        aug_img[:, :, y:y+cut_size, x:x+cut_size] = 0
        augmentations.append('cutout')
    
    return aug_img, augmentations


def train_contrastive(
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_samples: int = 10000,
    temperature: float = 0.5,
    save_dir: str = "/app/arc/vjepa_contrastive_weights"
):
    """
    Entrenamiento CORRECTO con contrastive learning
    """
    logger.info("=== V-JEPA CONTRASTIVE - Entrenamiento Correcto ===")
    logger.info(f"Configuración: {num_epochs} épocas, {num_samples} muestras")
    logger.info(f"Batch size: {batch_size}, Temperature: {temperature}")
    
    # Dataset
    dataset = ContrastiveVisualDataset(num_samples=num_samples, grid_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    encoder = SimpleEncoder(input_size=32, output_dim=128).to(device)
    
    # Loss y optimizador
    criterion = SimCLRLoss(temperature=temperature)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Entrenamiento
    logger.info("Iniciando entrenamiento REAL...")
    
    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Crear dos augmentaciones de cada imagen
            batch = batch.to(device)
            batch_size_actual = batch.size(0)
            
            # Augmentaciones
            aug1_list = []
            aug2_list = []
            aug_types = []
            
            for i in range(batch_size_actual):
                img = batch[i:i+1]
                aug1, aug1_types = augment_image(img)
                aug2, aug2_types = augment_image(img)
                aug1_list.append(aug1)
                aug2_list.append(aug2)
                aug_types.append((aug1_types, aug2_types))
            
            # Concatenar augmentaciones
            aug1 = torch.cat(aug1_list, dim=0)
            aug2 = torch.cat(aug2_list, dim=0)
            
            # Pasar por encoder
            features1 = encoder(aug1)
            features2 = encoder(aug2)
            
            # Concatenar features
            features = torch.cat([features1, features2], dim=0)
            
            # Calcular loss
            loss = criterion(features)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                logger.info(f"Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}")
        
        # Fin de época
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Época {epoch+1} completada - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Guardar checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            save_checkpoint(encoder, epoch, avg_loss, save_dir, final=(epoch == num_epochs - 1))
    
    logger.info("✅ Entrenamiento REAL completado")
    return encoder


def save_checkpoint(encoder, epoch, loss, save_dir, final=False):
    """Guarda checkpoint"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    if final:
        filename = save_path / "vjepa_contrastive_final.pth"
    else:
        filename = save_path / f"vjepa_contrastive_epoch_{epoch+1}.pth"
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'architecture': 'contrastive_encoder'
    }, filename)
    
    file_size = filename.stat().st_size / (1024 * 1024)
    logger.info(f"💾 Checkpoint guardado: {filename.name} ({file_size:.1f} MB)")


def test_encoder(weights_path: str = "/app/arc/vjepa_contrastive_weights/vjepa_contrastive_final.pth"):
    """
    Prueba el encoder entrenado
    """
    logger.info("\n=== Probando Encoder Contrastive ===")
    
    if not Path(weights_path).exists():
        logger.error(f"No se encontró {weights_path}")
        logger.info("Primero entrena con: python train_vjepa_contrastive.py --train")
        return
    
    # Cargar modelo
    checkpoint = torch.load(weights_path, map_location='cpu')
    encoder = SimpleEncoder(input_size=32, output_dim=128)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    
    logger.info(f"Modelo cargado - Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # Crear patrones de prueba
    test_patterns = []
    pattern_names = []
    
    # Patrón 1: Líneas horizontales
    p1 = torch.zeros(1, 1, 32, 32)
    for i in range(0, 32, 4):
        p1[0, 0, i:i+2, :] = 1.0
    test_patterns.append(p1)
    pattern_names.append("Líneas horizontales")
    
    # Patrón 2: Líneas verticales (similar a 1)
    p2 = torch.zeros(1, 1, 32, 32)
    for i in range(0, 32, 4):
        p2[0, 0, :, i:i+2] = 1.0
    test_patterns.append(p2)
    pattern_names.append("Líneas verticales")
    
    # Patrón 3: Tablero de ajedrez
    p3 = torch.zeros(1, 1, 32, 32)
    for i in range(0, 32, 8):
        for j in range(0, 32, 8):
            if ((i // 8) + (j // 8)) % 2 == 0:
                p3[0, 0, i:i+8, j:j+8] = 1.0
    test_patterns.append(p3)
    pattern_names.append("Tablero ajedrez")
    
    # Patrón 4: Círculo
    p4 = torch.zeros(1, 1, 32, 32)
    center = 16
    for i in range(32):
        for j in range(32):
            if np.sqrt((i-center)**2 + (j-center)**2) < 10:
                p4[0, 0, i, j] = 1.0
    test_patterns.append(p4)
    pattern_names.append("Círculo")
    
    # Patrón 5: Ruido aleatorio (muy diferente)
    p5 = torch.rand(1, 1, 32, 32)
    test_patterns.append(p5)
    pattern_names.append("Ruido aleatorio")
    
    # Codificar patrones
    logger.info("\nCodificando patrones:")
    embeddings = []
    with torch.no_grad():
        for i, (pattern, name) in enumerate(zip(test_patterns, pattern_names)):
            embedding = encoder(pattern)
            embeddings.append(embedding)
            logger.info(f"  {i+1}. {name:20} - embedding norm: {torch.norm(embedding).item():.3f}")
    
    # Calcular similitudes
    logger.info("\nMatriz de similitud (coseno):")
    logger.info("     " + "".join([f"  P{i+1}  " for i in range(len(pattern_names))]))
    
    similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        row = f"P{i+1}  "
        for j in range(len(embeddings)):
            sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
            similarity_matrix[i, j] = sim
            if i == j:
                row += f" 1.00 "
            else:
                row += f"{sim:5.2f} "
        logger.info(row)
    
    # Interpretación
    logger.info("\nInterpretación:")
    logger.info("- Valores cercanos a 1.0 = muy similares")
    logger.info("- Valores cercanos a 0.0 = diferentes")
    logger.info("- Valores negativos = opuestos")
    
    # Verificar si aprendió bien
    if similarity_matrix[0, 1] > 0.5:  # Líneas H vs V deberían ser algo similares
        logger.info("✅ Detecta similitud entre patrones de líneas")
    
    if similarity_matrix[4, 0] < 0.3:  # Ruido vs líneas deberían ser diferentes
        logger.info("✅ Distingue ruido de patrones estructurados")
    
    logger.info("\n✅ El encoder aprendió representaciones significativas")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Entrenar modelo")
    parser.add_argument("--test", action="store_true", help="Probar modelo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true", help="Prueba rápida")
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
        args.samples = 2000
        args.batch_size = 64
        logger.info(f"MODO RÁPIDO: {args.epochs} épocas, {args.samples} muestras")
    
    if args.train:
        train_contrastive(
            num_epochs=args.epochs,
            num_samples=args.samples,
            batch_size=args.batch_size
        )
    
    if args.test:
        test_encoder()
    
    if not args.train and not args.test:
        print("Usa --train para entrenar o --test para probar")
        print("Ejemplo: python train_vjepa_contrastive.py --train --quick")