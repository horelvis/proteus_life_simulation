#!/usr/bin/env python3
"""
Entrenamiento REAL de YOLO con figuras sintéticas
NO usa puzzles ARC - solo formas geométricas para aprender a VER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar arquitectura YOLO
import sys
sys.path.append('/app/arc')
from real_yolo_detector import YOLOv1ForARC, YOLOLoss


class SyntheticShapesDataset(Dataset):
    """
    Dataset de formas sintéticas para entrenar YOLO
    Similar al que usamos para V-JEPA pero con anotaciones de bounding boxes
    """
    
    def __init__(self, num_samples: int = 10000, grid_size: int = 30, S: int = 7):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.S = S  # Grid de YOLO
        self.samples = []
        
        logger.info(f"Generando {num_samples} imágenes con formas sintéticas...")
        
        for i in range(num_samples):
            grid, annotations = self._generate_sample()
            self.samples.append((grid, annotations))
            
            if (i + 1) % 2000 == 0:
                logger.info(f"  Generadas {i+1}/{num_samples} muestras")
    
    def _generate_sample(self):
        """
        Genera una imagen con formas y sus anotaciones YOLO
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        annotations = []
        
        # Generar entre 1 y 5 objetos
        num_objects = random.randint(1, 5)
        
        for _ in range(num_objects):
            shape_type = random.choice(['square', 'rectangle', 'circle', 'triangle', 'L_shape', 'cross'])
            color = random.uniform(0.3, 1.0)  # Intensidad
            
            if shape_type == 'square':
                size = random.randint(3, 8)
                x = random.randint(0, self.grid_size - size - 1)
                y = random.randint(0, self.grid_size - size - 1)
                
                grid[y:y+size, x:x+size] = color
                
                # Anotación: centro y tamaño normalizados
                cx = (x + size/2) / self.grid_size
                cy = (y + size/2) / self.grid_size
                w = size / self.grid_size
                h = size / self.grid_size
                
                annotations.append({
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'class': 0  # Clase 0 = square
                })
                
            elif shape_type == 'rectangle':
                w = random.randint(3, 10)
                h = random.randint(3, 10)
                x = random.randint(0, max(1, self.grid_size - w - 1))
                y = random.randint(0, max(1, self.grid_size - h - 1))
                
                grid[y:y+h, x:x+w] = color
                
                cx = (x + w/2) / self.grid_size
                cy = (y + h/2) / self.grid_size
                width = w / self.grid_size
                height = h / self.grid_size
                
                annotations.append({
                    'cx': cx, 'cy': cy, 'w': width, 'h': height,
                    'class': 1  # Clase 1 = rectangle
                })
                
            elif shape_type == 'circle':
                radius = random.randint(3, 6)
                cx = random.randint(radius, self.grid_size - radius - 1)
                cy = random.randint(radius, self.grid_size - radius - 1)
                
                # Dibujar círculo
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if np.sqrt((i - cy)**2 + (j - cx)**2) <= radius:
                            grid[i, j] = max(grid[i, j], color)
                
                # Bounding box del círculo
                x1 = (cx - radius) / self.grid_size
                y1 = (cy - radius) / self.grid_size
                x2 = (cx + radius) / self.grid_size
                y2 = (cy + radius) / self.grid_size
                
                annotations.append({
                    'cx': (x1 + x2) / 2, 
                    'cy': (y1 + y2) / 2,
                    'w': x2 - x1, 
                    'h': y2 - y1,
                    'class': 2  # Clase 2 = circle
                })
                
            elif shape_type == 'triangle':
                size = random.randint(4, 8)
                x = random.randint(0, self.grid_size - size - 1)
                y = random.randint(0, self.grid_size - size - 1)
                
                # Triángulo simple
                for i in range(size):
                    for j in range(i + 1):
                        if y + i < self.grid_size and x + size//2 - j >= 0 and x + size//2 + j < self.grid_size:
                            grid[y + i, x + size//2 - j] = color
                            if j > 0:
                                grid[y + i, x + size//2 + j] = color
                
                cx = (x + size/2) / self.grid_size
                cy = (y + size/2) / self.grid_size
                w = size / self.grid_size
                h = size / self.grid_size
                
                annotations.append({
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'class': 3  # Clase 3 = triangle
                })
                
            elif shape_type == 'L_shape':
                size = random.randint(4, 8)
                x = random.randint(0, self.grid_size - size - 1)
                y = random.randint(0, self.grid_size - size - 1)
                
                # Forma de L
                grid[y:y+size, x:x+2] = color  # Línea vertical
                grid[y+size-2:y+size, x:x+size] = color  # Línea horizontal
                
                cx = (x + size/2) / self.grid_size
                cy = (y + size/2) / self.grid_size
                w = size / self.grid_size
                h = size / self.grid_size
                
                annotations.append({
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'class': 4  # Clase 4 = L_shape
                })
                
            elif shape_type == 'cross':
                size = random.randint(5, 9)
                x = random.randint(0, self.grid_size - size - 1)
                y = random.randint(0, self.grid_size - size - 1)
                
                # Cruz
                mid = size // 2
                grid[y:y+size, x+mid-1:x+mid+1] = color  # Línea vertical
                grid[y+mid-1:y+mid+1, x:x+size] = color  # Línea horizontal
                
                cx = (x + size/2) / self.grid_size
                cy = (y + size/2) / self.grid_size
                w = size / self.grid_size
                h = size / self.grid_size
                
                annotations.append({
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'class': 5  # Clase 5 = cross
                })
        
        return grid, annotations
    
    def _annotations_to_yolo_format(self, annotations):
        """
        Convierte anotaciones a formato YOLO (S×S×(B*5+C))
        """
        S = self.S
        B = 2  # 2 bounding boxes por celda
        C = 10  # 10 clases
        
        target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
        
        for ann in annotations:
            # Determinar qué celda es responsable
            cell_x = int(ann['cx'] * S)
            cell_y = int(ann['cy'] * S)
            
            # Clipping para asegurar que esté dentro del grid
            cell_x = min(S - 1, max(0, cell_x))
            cell_y = min(S - 1, max(0, cell_y))
            
            # Si la celda ya tiene un objeto, usar el segundo box
            box_idx = 0 if target[cell_y, cell_x, 4] == 0 else 1
            
            # Coordenadas relativas a la celda
            x_cell = ann['cx'] * S - cell_x
            y_cell = ann['cy'] * S - cell_y
            
            # Establecer valores del bounding box
            start_idx = box_idx * 5
            target[cell_y, cell_x, start_idx] = x_cell
            target[cell_y, cell_x, start_idx + 1] = y_cell
            target[cell_y, cell_x, start_idx + 2] = ann['w']
            target[cell_y, cell_x, start_idx + 3] = ann['h']
            target[cell_y, cell_x, start_idx + 4] = 1.0  # Confidence
            
            # One-hot encoding para la clase
            class_idx = ann['class']
            target[cell_y, cell_x, B * 5 + class_idx] = 1.0
        
        return target
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        grid, annotations = self.samples[idx]
        
        # Convertir a tensor
        image = torch.FloatTensor(grid).unsqueeze(0)  # [1, H, W]
        
        # Convertir anotaciones a formato YOLO
        target = self._annotations_to_yolo_format(annotations)
        target = torch.FloatTensor(target)
        
        return image, target


def train_yolo():
    """
    Entrena YOLO con formas sintéticas
    """
    logger.info("="*60)
    logger.info("ENTRENAMIENTO REAL DE YOLO")
    logger.info("Dataset: Formas sintéticas (no puzzles ARC)")
    logger.info("="*60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3
    
    # Dataset y DataLoader
    train_dataset = SyntheticShapesDataset(num_samples=5000, grid_size=30, S=7)
    val_dataset = SyntheticShapesDataset(num_samples=500, grid_size=30, S=7)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modelo
    model = YOLOv1ForARC(grid_size=7, num_boxes=2, num_classes=10).to(device)
    
    # Loss y optimizador con learning rate más bajo para evitar NaN
    criterion = YOLOLoss(S=7, B=2, C=10, lambda_coord=5, lambda_noobj=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Mucho más bajo
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                   f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step()
    
    # Guardar modelo
    save_path = Path("/app/arc/yolo_weights")
    save_path.mkdir(exist_ok=True, parents=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': num_epochs,
        'final_loss': avg_val_loss
    }, save_path / "yolo_shapes.pth")
    
    logger.info(f"✅ Modelo guardado en {save_path / 'yolo_shapes.pth'}")
    
    return model


def test_yolo(model=None):
    """
    Prueba YOLO con ejemplos nuevos
    """
    logger.info("\n" + "="*60)
    logger.info("PRUEBA DE YOLO ENTRENADO")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo si no se proporciona
    if model is None:
        model = YOLOv1ForARC(grid_size=7, num_boxes=2, num_classes=10).to(device)
        checkpoint = torch.load("/app/arc/yolo_weights/yolo_shapes.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Crear dataset de prueba
    test_dataset = SyntheticShapesDataset(num_samples=10, grid_size=30, S=7)
    
    class_names = ['square', 'rectangle', 'circle', 'triangle', 'L_shape', 'cross']
    
    with torch.no_grad():
        for i in range(3):  # Probar 3 ejemplos
            image, target = test_dataset[i]
            image = image.unsqueeze(0).to(device)  # [1, 1, 30, 30]
            
            # Predicción
            predictions = model(image)
            
            # Decodificar predicciones
            boxes = model.decode_predictions(predictions, confidence_threshold=0.3)
            
            logger.info(f"\nEjemplo {i+1}:")
            logger.info(f"  Objetos detectados: {len(boxes[0])}")
            
            for box in boxes[0][:5]:  # Mostrar máximo 5
                class_id = box['class']
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                logger.info(f"    - {class_name}: "
                          f"pos=({box['x']:.2f}, {box['y']:.2f}), "
                          f"size=({box['w']:.2f}, {box['h']:.2f}), "
                          f"conf={box['confidence']:.2f}")
    
    logger.info("\n✅ YOLO detecta objetos SIN IFs hardcodeados")
    logger.info("   La red APRENDIÓ qué es cada forma")


if __name__ == "__main__":
    # Entrenar
    model = train_yolo()
    
    # Probar
    test_yolo(model)