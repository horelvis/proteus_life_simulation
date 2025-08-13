#!/usr/bin/env python3
"""
V-JEPA PURO - Entrenamiento auto-supervisado real
NO le decimos qué es simetría, formas, o patrones
Aprende a "ver" por sí mismo, como un bebé
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureVisualDataset(Dataset):
    """
    Dataset PURO - Solo genera matrices aleatorias con estructura
    NO le dice al modelo qué tipo de estructura es
    """
    
    def __init__(self, num_samples: int = 10000, grid_size_range: Tuple[int, int] = (3, 15)):
        self.samples = []
        self.grid_size_range = grid_size_range
        
        logger.info(f"Generando {num_samples} matrices visuales aleatorias...")
        
        for i in range(num_samples):
            # Solo generar una matriz con alguna estructura (sin decir cuál)
            grid = self._generate_random_structured_grid()
            self.samples.append(grid)
            
            if (i + 1) % 2000 == 0:
                logger.info(f"  Generados {i + 1}/{num_samples} ejemplos...")
    
    def _generate_random_structured_grid(self) -> np.ndarray:
        """
        Genera una matriz con ALGUNA estructura, pero sin categorizar
        El modelo debe aprender qué patrones existen
        """
        h = np.random.randint(*self.grid_size_range)
        w = np.random.randint(*self.grid_size_range)
        
        # Elegir método de generación aleatoriamente
        method = np.random.randint(0, 10)
        
        if method == 0:
            # Matriz completamente aleatoria (ruido)
            grid = np.random.randint(0, 10, (h, w))
            
        elif method == 1:
            # Matriz con un valor dominante
            grid = np.full((h, w), np.random.randint(0, 10))
            noise_mask = np.random.random((h, w)) < 0.2
            grid[noise_mask] = np.random.randint(0, 10, np.sum(noise_mask))
            
        elif method == 2:
            # Gradiente simple
            grid = np.zeros((h, w), dtype=int)
            for i in range(h):
                grid[i, :] = (i * 9 // h)
                
        elif method == 3:
            # Bloques aleatorios
            grid = np.zeros((h, w), dtype=int)
            num_blocks = np.random.randint(1, 5)
            for _ in range(num_blocks):
                bh = np.random.randint(1, min(5, h))
                bw = np.random.randint(1, min(5, w))
                y = np.random.randint(0, max(1, h - bh))
                x = np.random.randint(0, max(1, w - bw))
                grid[y:y+bh, x:x+bw] = np.random.randint(1, 10)
                
        elif method == 4:
            # Líneas aleatorias
            grid = np.zeros((h, w), dtype=int)
            num_lines = np.random.randint(1, 4)
            for _ in range(num_lines):
                color = np.random.randint(1, 10)
                if np.random.random() > 0.5:
                    # Línea horizontal
                    row = np.random.randint(0, h)
                    grid[row, :] = color
                else:
                    # Línea vertical
                    col = np.random.randint(0, w)
                    grid[:, col] = color
                    
        elif method == 5:
            # Patrón repetitivo simple
            pattern_h = np.random.randint(2, 4)
            pattern_w = np.random.randint(2, 4)
            pattern = np.random.randint(0, 5, (pattern_h, pattern_w))
            grid = np.tile(pattern, (h // pattern_h + 1, w // pattern_w + 1))[:h, :w]
            
        elif method == 6:
            # Bordes
            grid = np.zeros((h, w), dtype=int)
            color = np.random.randint(1, 10)
            grid[0, :] = color
            grid[-1, :] = color
            grid[:, 0] = color
            grid[:, -1] = color
            
        elif method == 7:
            # Diagonal
            grid = np.zeros((h, w), dtype=int)
            color = np.random.randint(1, 10)
            for i in range(min(h, w)):
                grid[i, i] = color
                
        elif method == 8:
            # Clusters aleatorios
            grid = np.zeros((h, w), dtype=int)
            num_clusters = np.random.randint(2, 6)
            for _ in range(num_clusters):
                center_y = np.random.randint(0, h)
                center_x = np.random.randint(0, w)
                color = np.random.randint(1, 10)
                radius = np.random.randint(1, 3)
                
                for i in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
                    for j in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                        if abs(i - center_y) + abs(j - center_x) <= radius:
                            grid[i, j] = color
                            
        else:
            # Matriz sparse
            grid = np.zeros((h, w), dtype=int)
            num_points = np.random.randint(1, h * w // 3)
            positions = np.random.choice(h * w, num_points, replace=False)
            colors = np.random.randint(1, 10, num_points)
            for pos, color in zip(positions, colors):
                grid[pos // w, pos % w] = color
        
        return grid
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        grid = self.samples[idx]
        
        # Pad a tamaño fijo
        max_size = 30
        h, w = grid.shape
        padded = np.zeros((max_size, max_size), dtype=np.float32)
        padded[:min(h, max_size), :min(w, max_size)] = grid[:min(h, max_size), :min(w, max_size)]
        
        return torch.FloatTensor(padded)


class PureVJEPAEncoder(nn.Module):
    """
    Encoder PURO - Aprende representaciones sin saber qué buscar
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        # Stack de convoluciones para aprender features
        self.conv_stack = nn.Sequential(
            # Primera capa - detecta bordes y patrones básicos
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Segunda capa - combina patrones básicos
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Tercera capa - patrones más complejos
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Cuarta capa - abstracción alta
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Tamaño fijo de salida
        )
        
        # Proyección a embedding
        self.projector = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        # Asegurar dimensión de canal
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Pasar por convoluciones
        features = self.conv_stack(x)
        
        # Aplanar y proyectar
        features = features.view(features.size(0), -1)
        embedding = self.projector(features)
        
        return embedding


class PureVJEPAPredictor(nn.Module):
    """
    Predictor PURO - Predice representaciones futuras
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        # Red profunda para predicción
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, z_context):
        """Predice representación target desde contexto"""
        return self.predictor(z_context)


def train_pure_vjepa(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_samples: int = 10000,
    mask_ratio: float = 0.5,
    save_dir: str = "/app/arc/vjepa_pure_weights"
):
    """
    Entrena V-JEPA de forma PURAMENTE auto-supervisada
    """
    logger.info("=== V-JEPA PURO - Entrenamiento Auto-Supervisado ===")
    logger.info("NO le decimos qué buscar - aprende solo")
    logger.info(f"Configuración: {num_epochs} épocas, {num_samples} muestras")
    
    # Crear dataset
    dataset = PureVisualDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    encoder = PureVJEPAEncoder(embedding_dim=256).to(device)
    predictor = PureVJEPAPredictor(embedding_dim=256).to(device)
    target_encoder = PureVJEPAEncoder(embedding_dim=256).to(device)
    
    # Target encoder es una copia EMA del encoder (momentum encoding)
    target_encoder.load_state_dict(encoder.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad = False
    
    # Optimizadores
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # Loss
    criterion = nn.MSELoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    logger.info("Iniciando entrenamiento...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        encoder.train()
        predictor.train()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            batch_size_actual = batch.size(0)
            
            # Crear máscaras aleatorias para cada imagen
            # Dividir la imagen en patches y ocultar algunos
            patch_size = 5  # Tamaño de patches a ocultar
            h, w = 30, 30  # Tamaño de imagen padded
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            num_patches = num_patches_h * num_patches_w
            
            # Para cada imagen en el batch, crear máscara diferente
            batch_loss = 0
            
            for i in range(batch_size_actual):
                img = batch[i:i+1]  # Imagen individual
                
                # Decidir qué patches ocultar
                num_masked = int(num_patches * mask_ratio)
                masked_indices = np.random.choice(num_patches, num_masked, replace=False)
                
                # Crear imagen con patches ocultos
                masked_img = img.clone()
                for idx in masked_indices:
                    row = (idx // num_patches_w) * patch_size
                    col = (idx % num_patches_w) * patch_size
                    masked_img[:, row:row+patch_size, col:col+patch_size] = 0
                
                # Encoder: procesar imagen con máscaras
                z_context = encoder(masked_img)
                
                # Target encoder: procesar imagen completa
                with torch.no_grad():
                    z_target = target_encoder(img)
                
                # Predictor: predecir representación completa desde parcial
                z_pred = predictor(z_context)
                
                # Loss: qué tan bien predecimos la representación completa
                loss = criterion(z_pred, z_target)
                batch_loss += loss
            
            # Promedio del batch
            batch_loss = batch_loss / batch_size_actual
            
            # Backprop
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            
            # Actualizar target encoder (EMA)
            momentum = 0.996
            for param, target_param in zip(encoder.parameters(), target_encoder.parameters()):
                target_param.data = momentum * target_param.data + (1 - momentum) * param.data
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {batch_loss.item():.6f}")
        
        # Fin de época
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Época {epoch+1} completada - Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Guardar checkpoint cada 10 épocas
        if (epoch + 1) % 10 == 0:
            save_checkpoint(encoder, predictor, target_encoder, epoch, avg_loss, save_dir)
    
    # Guardar modelo final
    save_checkpoint(encoder, predictor, target_encoder, num_epochs - 1, avg_loss, save_dir, final=True)
    
    logger.info("✅ Entrenamiento completado")
    logger.info("El modelo aprendió a 'ver' sin que le dijéramos qué buscar")
    
    return encoder, predictor, target_encoder


def save_checkpoint(encoder, predictor, target_encoder, epoch, loss, save_dir, final=False):
    """Guarda checkpoint del modelo"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    if final:
        filename = save_path / "vjepa_pure_final.pth"
    else:
        filename = save_path / f"vjepa_pure_epoch_{epoch+1}.pth"
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'target_encoder_state_dict': target_encoder.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'embedding_dim': encoder.embedding_dim,
        'architecture': 'pure_vjepa'
    }, filename)
    
    file_size = filename.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"💾 Checkpoint guardado: {filename.name} ({file_size:.1f} MB)")


def test_pure_vjepa(weights_path: str = "/app/arc/vjepa_pure_weights/vjepa_pure_final.pth"):
    """Prueba el modelo entrenado"""
    logger.info("\n=== Probando V-JEPA PURO ===")
    
    if not Path(weights_path).exists():
        logger.error(f"No se encontró {weights_path}")
        return
    
    # Cargar modelo
    checkpoint = torch.load(weights_path, map_location='cpu')
    encoder = PureVJEPAEncoder(embedding_dim=256)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    
    logger.info(f"Modelo cargado - Época: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
    
    # Crear algunos patrones de prueba
    test_patterns = [
        np.array([[1,1,1,0,0],
                  [1,0,1,0,0],
                  [1,1,1,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]]),  # No le decimos que es un cuadrado
        
        np.array([[0,2,0,2,0],
                  [2,0,2,0,2],
                  [0,2,0,2,0],
                  [2,0,2,0,2],
                  [0,2,0,2,0]]),  # No le decimos que es un patrón alternado
        
        np.array([[3,3,3,3,3],
                  [0,0,0,0,0],
                  [3,3,3,3,3],
                  [0,0,0,0,0],
                  [3,3,3,3,3]])   # No le decimos que son líneas
    ]
    
    logger.info("\nCodificando patrones (sin decirle qué son):")
    embeddings = []
    
    with torch.no_grad():
        for i, pattern in enumerate(test_patterns):
            # Pad
            padded = np.zeros((30, 30))
            padded[:5, :5] = pattern
            
            # Codificar
            tensor = torch.FloatTensor(padded).unsqueeze(0)
            embedding = encoder(tensor)
            embeddings.append(embedding)
            
            logger.info(f"  Patrón {i+1}: embedding shape={embedding.shape}, norm={torch.norm(embedding).item():.3f}")
    
    # Calcular similitudes entre patrones
    logger.info("\nSimilitudes entre patrones (el modelo decide si son similares):")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
            logger.info(f"  Patrón {i+1} vs {j+1}: {sim:.3f}")
    
    logger.info("\n✅ El modelo aprendió representaciones por sí mismo")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V-JEPA PURO - Entrenamiento auto-supervisado real")
    parser.add_argument("--train", action="store_true", help="Entrenar modelo")
    parser.add_argument("--test", action="store_true", help="Probar modelo")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="Prueba rápida")
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 5
        args.samples = 1000
        args.batch_size = 16
        logger.info(f"MODO RÁPIDO: {args.epochs} épocas, {args.samples} muestras")
    
    if args.train:
        train_pure_vjepa(
            num_epochs=args.epochs,
            num_samples=args.samples,
            batch_size=args.batch_size,
            lr=args.lr
        )
    
    if args.test:
        test_pure_vjepa()
    
    if not args.train and not args.test:
        print("Usa --train para entrenar o --test para probar")
        print("Ejemplo: python train_vjepa_pure.py --train --quick")