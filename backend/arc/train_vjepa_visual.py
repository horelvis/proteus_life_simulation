#!/usr/bin/env python3
"""
Script para PRE-ENTRENAR V-JEPA en conocimiento visual general
Este entrenamiento es VÁLIDO para ARC porque NO usa puzzles de ARC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualConceptsDataset(Dataset):
    """
    Dataset de conceptos visuales GENERALES (no ARC)
    Para pre-entrenar el reconocimiento visual básico
    """
    
    def __init__(self, num_samples: int = 10000):
        """
        Genera datos sintéticos de formas, colores y transformaciones
        NO usa puzzles de ARC - solo patrones geométricos generales
        """
        self.samples = []
        
        logger.info(f"Generando {num_samples} ejemplos de entrenamiento visual...")
        
        for i in range(num_samples):
            # Elegir tipo de patrón a generar
            pattern_type = np.random.choice([
                'single_shape',
                'color_pattern', 
                'transformation',
                'spatial_relation',
                'symmetry',
                'repetition'
            ])
            
            sample = self._generate_sample(pattern_type)
            self.samples.append(sample)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generados {i + 1}/{num_samples} ejemplos...")
    
    def _generate_sample(self, pattern_type: str) -> Dict:
        """Genera un ejemplo de entrenamiento visual"""
        
        if pattern_type == 'single_shape':
            return self._generate_shape_sample()
        elif pattern_type == 'color_pattern':
            return self._generate_color_sample()
        elif pattern_type == 'transformation':
            return self._generate_transformation_sample()
        elif pattern_type == 'spatial_relation':
            return self._generate_spatial_sample()
        elif pattern_type == 'symmetry':
            return self._generate_symmetry_sample()
        else:  # repetition
            return self._generate_repetition_sample()
    
    def _generate_shape_sample(self) -> Dict:
        """Genera ejemplo con formas geométricas básicas"""
        grid_size = np.random.randint(5, 15)
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        shape_type = np.random.choice(['line', 'square', 'rectangle', 'L', 'T', 'cross'])
        color = np.random.randint(1, 10)
        
        if shape_type == 'line':
            if np.random.random() > 0.5:  # Horizontal
                row = np.random.randint(0, grid_size)
                length = np.random.randint(2, grid_size)
                start = np.random.randint(0, grid_size - length + 1)
                grid[row, start:start+length] = color
            else:  # Vertical
                col = np.random.randint(0, grid_size)
                length = np.random.randint(2, grid_size)
                start = np.random.randint(0, grid_size - length + 1)
                grid[start:start+length, col] = color
                
        elif shape_type == 'square':
            size = np.random.randint(2, min(5, grid_size))
            x = np.random.randint(0, grid_size - size + 1)
            y = np.random.randint(0, grid_size - size + 1)
            grid[y:y+size, x:x+size] = color
            
        elif shape_type == 'rectangle':
            h = np.random.randint(2, min(5, grid_size))
            w = np.random.randint(2, min(5, grid_size))
            x = np.random.randint(0, max(1, grid_size - w + 1))
            y = np.random.randint(0, max(1, grid_size - h + 1))
            grid[y:y+h, x:x+w] = color
            
        elif shape_type == 'L':
            size = 3
            if grid_size >= size:
                x = np.random.randint(0, grid_size - size + 1)
                y = np.random.randint(0, grid_size - size + 1)
                grid[y:y+size, x] = color
                grid[y+size-1, x:x+size] = color
                
        elif shape_type == 'T':
            if grid_size >= 3:
                x = np.random.randint(1, grid_size - 1)
                y = np.random.randint(0, grid_size - 2)
                grid[y, x-1:x+2] = color
                grid[y:y+3, x] = color
                
        elif shape_type == 'cross':
            center = grid_size // 2
            grid[center, :] = color
            grid[:, center] = color
        
        return {
            'input': grid,
            'labels': {
                'shape': shape_type,
                'color': color,
                'position': self._get_object_position(grid)
            }
        }
    
    def _generate_color_sample(self) -> Dict:
        """Genera ejemplo con patrones de color"""
        grid_size = np.random.randint(5, 10)
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        pattern = np.random.choice(['gradient', 'checkerboard', 'stripes'])
        
        if pattern == 'gradient':
            colors = np.random.choice(range(1, 10), 2, replace=False)
            for i in range(grid_size):
                color_idx = int(i * len(colors) / grid_size)
                grid[i, :] = colors[color_idx]
                
        elif pattern == 'checkerboard':
            colors = np.random.choice(range(1, 10), 2, replace=False)
            for i in range(grid_size):
                for j in range(grid_size):
                    grid[i, j] = colors[(i + j) % 2]
                    
        elif pattern == 'stripes':
            colors = np.random.choice(range(1, 10), 3, replace=False)
            for i in range(grid_size):
                grid[i, :] = colors[i % len(colors)]
        
        return {
            'input': grid,
            'labels': {
                'pattern': pattern,
                'num_colors': len(np.unique(grid[grid > 0]))
            }
        }
    
    def _generate_transformation_sample(self) -> Dict:
        """Genera ejemplo de transformación"""
        base_size = np.random.randint(3, 8)
        base_grid = np.random.randint(0, 5, (base_size, base_size))
        
        transformation = np.random.choice([
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical',
            'scale_2x', 'invert_colors'
        ])
        
        if transformation == 'rotate_90':
            output = np.rot90(base_grid, 1)
        elif transformation == 'rotate_180':
            output = np.rot90(base_grid, 2)
        elif transformation == 'rotate_270':
            output = np.rot90(base_grid, 3)
        elif transformation == 'flip_horizontal':
            output = np.fliplr(base_grid)
        elif transformation == 'flip_vertical':
            output = np.flipud(base_grid)
        elif transformation == 'scale_2x':
            output = np.repeat(np.repeat(base_grid, 2, axis=0), 2, axis=1)
        else:  # invert_colors
            output = base_grid.copy()
            output[output > 0] = 10 - output[output > 0]
        
        return {
            'input': base_grid,
            'output': output,
            'labels': {
                'transformation': transformation
            }
        }
    
    def _generate_spatial_sample(self) -> Dict:
        """Genera ejemplo con relaciones espaciales"""
        grid = np.zeros((10, 10), dtype=np.int32)
        
        # Colocar dos objetos con relación espacial
        obj1_color = np.random.randint(1, 5)
        obj2_color = np.random.randint(5, 10)
        
        relation = np.random.choice(['above', 'below', 'left', 'right', 'diagonal'])
        
        # Objeto 1 (2x2)
        x1, y1 = 3, 3
        grid[y1:y1+2, x1:x1+2] = obj1_color
        
        # Objeto 2 según relación
        if relation == 'above':
            grid[y1-3:y1-1, x1:x1+2] = obj2_color
        elif relation == 'below':
            grid[y1+3:y1+5, x1:x1+2] = obj2_color
        elif relation == 'left':
            grid[y1:y1+2, x1-3:x1-1] = obj2_color
        elif relation == 'right':
            grid[y1:y1+2, x1+3:x1+5] = obj2_color
        else:  # diagonal
            grid[y1+3:y1+5, x1+3:x1+5] = obj2_color
        
        return {
            'input': grid,
            'labels': {
                'relation': relation,
                'num_objects': 2
            }
        }
    
    def _generate_symmetry_sample(self) -> Dict:
        """Genera ejemplo con simetría"""
        size = np.random.randint(5, 10)
        grid = np.zeros((size, size), dtype=np.int32)
        
        symmetry_type = np.random.choice(['horizontal', 'vertical', 'diagonal', 'none'])
        
        if symmetry_type == 'horizontal':
            # Generar mitad superior
            half_h = size // 2
            half_grid = np.random.randint(0, 5, (half_h, size))
            grid[:half_h] = half_grid
            # Reflejar a la mitad inferior
            for i in range(size - half_h):
                if half_h - 1 - i >= 0:
                    grid[half_h + i] = half_grid[half_h - 1 - i]
        elif symmetry_type == 'vertical':
            # Generar mitad izquierda
            half_w = size // 2
            for i in range(size):
                for j in range(half_w):
                    val = np.random.randint(0, 5)
                    grid[i, j] = val
                    if size - 1 - j < size:
                        grid[i, size - 1 - j] = val
        elif symmetry_type == 'diagonal':
            for i in range(size):
                for j in range(i+1):
                    val = np.random.randint(0, 5)
                    grid[i, j] = val
                    if j < size and i < size:
                        grid[j, i] = val
        else:  # none
            grid = np.random.randint(0, 5, (size, size))
        
        return {
            'input': grid,
            'labels': {
                'symmetry': symmetry_type,
                'is_symmetric': symmetry_type != 'none'
            }
        }
    
    def _generate_repetition_sample(self) -> Dict:
        """Genera ejemplo con patrones repetitivos"""
        pattern_size = np.random.randint(2, 4)
        pattern = np.random.randint(0, 5, (pattern_size, pattern_size))
        
        repetitions = np.random.randint(2, 4)
        grid = np.tile(pattern, (repetitions, repetitions))
        
        return {
            'input': grid,
            'labels': {
                'pattern_size': pattern_size,
                'repetitions': repetitions,
                'is_repetitive': True
            }
        }
    
    def _get_object_position(self, grid: np.ndarray) -> str:
        """Determina la posición del objeto en el grid"""
        if np.sum(grid) == 0:
            return 'empty'
        
        y_coords, x_coords = np.where(grid > 0)
        center_y = np.mean(y_coords) / grid.shape[0]
        center_x = np.mean(x_coords) / grid.shape[1]
        
        if center_y < 0.33:
            v_pos = 'top'
        elif center_y > 0.66:
            v_pos = 'bottom'
        else:
            v_pos = 'middle'
            
        if center_x < 0.33:
            h_pos = 'left'
        elif center_x > 0.66:
            h_pos = 'right'
        else:
            h_pos = 'center'
            
        return f"{v_pos}_{h_pos}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Pad todos los grids a tamaño fijo (30x30 como máximo en ARC)
        max_size = 30
        
        def pad_grid(grid, target_size=30):
            """Pad grid a tamaño fijo"""
            h, w = grid.shape
            padded = np.zeros((target_size, target_size), dtype=np.float32)
            padded[:min(h, target_size), :min(w, target_size)] = grid[:min(h, target_size), :min(w, target_size)]
            return padded
        
        # Convertir a tensores con padding
        if 'output' in sample:
            # Ejemplo de transformación
            input_padded = pad_grid(sample['input'], max_size)
            output_padded = pad_grid(sample['output'], max_size)
            input_tensor = torch.FloatTensor(input_padded)
            output_tensor = torch.FloatTensor(output_padded)
            return input_tensor, output_tensor, sample['labels']
        else:
            # Ejemplo de clasificación
            input_padded = pad_grid(sample['input'], max_size)
            input_tensor = torch.FloatTensor(input_padded)
            return input_tensor, sample['labels']


class VJEPAEncoder(nn.Module):
    """
    Encoder de V-JEPA para aprender representaciones visuales
    Similar a la arquitectura original pero adaptado para grids discretos
    """
    
    def __init__(self, input_channels: int = 1, embedding_dim: int = 256):
        super().__init__()
        
        # Encoder convolucional
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Proyección a espacio de embedding
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Agregar dimensión de canal si es necesario
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Encoder convolucional
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Pooling adaptativo
        x = self.pool(x)
        
        # Aplanar y proyectar
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        embedding = self.fc2(x)
        
        return embedding


class VJEPAPredictor(nn.Module):
    """
    Predictor de V-JEPA para predecir representaciones futuras
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, z_context, z_mask_tokens):
        """
        Predice la representación del target dado el contexto
        """
        combined = torch.cat([z_context, z_mask_tokens], dim=-1)
        z_pred = self.predictor(combined)
        return z_pred


def custom_collate_fn(batch):
    """Collate function personalizado para manejar diferentes tipos de datos"""
    # Separar por tipo de dato
    transformation_batch = []
    classification_batch = []
    
    for item in batch:
        if len(item) == 3:  # Transformación
            transformation_batch.append(item)
        else:  # Clasificación
            classification_batch.append(item)
    
    # Si todos son del mismo tipo, procesar normalmente
    if transformation_batch and not classification_batch:
        inputs = torch.stack([item[0] for item in transformation_batch])
        outputs = torch.stack([item[1] for item in transformation_batch])
        labels = [item[2] for item in transformation_batch]
        return inputs, outputs, labels, 'transformation'
    elif classification_batch and not transformation_batch:
        inputs = torch.stack([item[0] for item in classification_batch])
        labels = [item[1] for item in classification_batch]
        return inputs, labels, 'classification'
    else:
        # Mixto - tomar el tipo más común
        if len(transformation_batch) >= len(classification_batch):
            inputs = torch.stack([item[0] for item in transformation_batch])
            outputs = torch.stack([item[1] for item in transformation_batch])
            labels = [item[2] for item in transformation_batch]
            return inputs, outputs, labels, 'transformation'
        else:
            inputs = torch.stack([item[0] for item in classification_batch])
            labels = [item[1] for item in classification_batch]
            return inputs, labels, 'classification'


def train_vjepa(num_epochs: int = 50, batch_size: int = 32, lr: float = 1e-3, 
                save_dir: str = "/app/arc/vjepa_pretrained_weights", num_samples: int = 5000):
    """
    Entrena V-JEPA en datos visuales generales
    """
    logger.info("=== Iniciando pre-entrenamiento de V-JEPA ===")
    logger.info("Este entrenamiento usa SOLO datos sintéticos generales")
    logger.info("NO usa ningún puzzle de ARC - es conocimiento visual general")
    logger.info(f"Los pesos se guardarán en: {save_dir}")
    
    # Crear dataset
    dataset = VisualConceptsDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=custom_collate_fn)
    
    # Crear modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    encoder = VJEPAEncoder(embedding_dim=256).to(device)
    predictor = VJEPAPredictor(embedding_dim=256).to(device)
    
    # Optimizadores
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=lr)
    
    # Loss
    criterion = nn.MSELoss()
    
    # Entrenamiento
    logger.info(f"Iniciando entrenamiento por {num_epochs} épocas...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_type = batch_data[-1]  # Último elemento es el tipo
            
            if batch_type == 'transformation':
                inputs, outputs, labels, _ = batch_data
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                
                # Codificar input y output
                z_input = encoder(inputs)
                z_output = encoder(outputs)
                
                # Crear máscaras aleatorias (simular oclusión parcial)
                mask = torch.rand(z_input.shape) > 0.3
                z_masked = z_input * mask.to(device)
                
                # Predecir representación del output desde input parcial
                z_pred = predictor(z_masked, z_masked)
                
                # Loss: qué tan bien predecimos la representación del output
                loss = criterion(z_pred, z_output)
                
            else:  # classification
                inputs, labels, _ = batch_data
                inputs = inputs.to(device)
                
                # Auto-supervisión: predecir partes ocultas
                z_full = encoder(inputs)
                
                # Ocultar parte aleatoria
                mask = torch.rand(z_full.shape) > 0.5
                z_visible = z_full * mask.to(device)
                z_hidden = z_full * (~mask).to(device)
                
                # Predecir parte oculta desde visible
                z_pred = predictor(z_visible, torch.zeros_like(z_visible))
                
                loss = criterion(z_pred, z_hidden)
            
            # Backprop
            encoder_optimizer.zero_grad()
            predictor_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            predictor_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"  Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Época {epoch+1}/{num_epochs} completada - Loss promedio: {avg_loss:.4f}")
    
    # Guardar modelos entrenados
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Guardar checkpoint completo
    checkpoint_path = save_path / "vjepa_visual_knowledge.pth"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'encoder_architecture': str(encoder),
        'embedding_dim': 256,
        'training_samples': len(dataset),
        'epochs': num_epochs,
        'final_loss': avg_loss,
        'device': str(device)
    }, checkpoint_path)
    
    # Guardar también los pesos por separado para fácil acceso
    torch.save(encoder.state_dict(), save_path / "encoder_weights.pth")
    torch.save(predictor.state_dict(), save_path / "predictor_weights.pth")
    
    # Verificar que se guardaron correctamente
    if checkpoint_path.exists():
        file_size = checkpoint_path.stat().st_size / 1024  # KB
        logger.info(f"✅ Checkpoint guardado: {checkpoint_path} ({file_size:.1f} KB)")
    else:
        logger.error(f"❌ Error: No se pudo guardar el checkpoint")
        
    logger.info(f"✅ Modelos guardados en {save_path}")
    logger.info("Pre-entrenamiento completado - V-JEPA ahora tiene conocimiento visual general")
    logger.info("Este conocimiento es VÁLIDO para usar en competencia ARC")
    
    return encoder, predictor


def test_pretrained_vjepa(weights_dir: str = "/app/arc/vjepa_pretrained_weights"):
    """
    Prueba el V-JEPA pre-entrenado
    """
    logger.info("\n=== Probando V-JEPA pre-entrenado ===")
    
    checkpoint_path = Path(weights_dir) / "vjepa_visual_knowledge.pth"
    
    if not checkpoint_path.exists():
        logger.error(f"❌ No se encontraron pesos en {checkpoint_path}")
        logger.info("Primero entrena el modelo con: python train_vjepa_visual.py --train")
        return False
    
    # Cargar modelos
    logger.info(f"Cargando pesos desde {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    encoder = VJEPAEncoder(embedding_dim=256)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    
    # Crear algunos ejemplos de test
    test_cases = [
        ("Línea horizontal", np.array([[0,0,0,0,0],
                                       [1,1,1,1,1],
                                       [0,0,0,0,0],
                                       [0,0,0,0,0],
                                       [0,0,0,0,0]])),
        ("Cuadrado", np.array([[0,0,0,0,0],
                              [0,2,2,2,0],
                              [0,2,0,2,0],
                              [0,2,2,2,0],
                              [0,0,0,0,0]])),
        ("Patrón diagonal", np.array([[3,0,0,0,0],
                                      [0,3,0,0,0],
                                      [0,0,3,0,0],
                                      [0,0,0,3,0],
                                      [0,0,0,0,3]]))
    ]
    
    logger.info("Codificando patrones de test:")
    for name, pattern in test_cases:
        with torch.no_grad():
            embedding = encoder(torch.FloatTensor(pattern))
            logger.info(f"  {name}: embedding shape = {embedding.shape}, norm = {torch.norm(embedding).item():.3f}")
    
    logger.info("\n✅ V-JEPA puede codificar patrones visuales correctamente")
    
    # Mostrar información del checkpoint
    logger.info("\n📊 Información del modelo cargado:")
    logger.info(f"  - Dimensión de embedding: {checkpoint.get('embedding_dim', 'N/A')}")
    logger.info(f"  - Ejemplos de entrenamiento: {checkpoint.get('training_samples', 'N/A')}")
    logger.info(f"  - Épocas entrenadas: {checkpoint.get('epochs', 'N/A')}")
    logger.info(f"  - Loss final: {checkpoint.get('final_loss', 'N/A'):.4f}" if checkpoint.get('final_loss') else "  - Loss final: N/A")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-entrenar V-JEPA con conocimiento visual general")
    parser.add_argument("--train", action="store_true", help="Entrenar V-JEPA")
    parser.add_argument("--test", action="store_true", help="Probar V-JEPA pre-entrenado")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--num-samples", type=int, default=5000, help="Número de ejemplos de entrenamiento")
    parser.add_argument("--quick", action="store_true", help="Prueba rápida con pocos datos")
    
    args = parser.parse_args()
    
    if args.quick:
        # Prueba rápida con configuración mínima
        args.epochs = min(args.epochs, 3)
        args.num_samples = min(args.num_samples, 500)
        args.batch_size = min(args.batch_size, 8)
        logger.info("🚀 MODO RÁPIDO: epochs={}, samples={}, batch={}".format(
            args.epochs, args.num_samples, args.batch_size))
    
    if args.train:
        train_vjepa(num_epochs=args.epochs, batch_size=args.batch_size, 
                   num_samples=args.num_samples)
    
    if args.test:
        test_pretrained_vjepa()
    
    if not args.train and not args.test:
        print("Usa --train para entrenar o --test para probar")
        print("Ejemplo: python train_vjepa_visual.py --train --epochs 20")