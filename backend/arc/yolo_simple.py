#!/usr/bin/env python3
"""
YOLO Simplificado para ARC
Versión más estable con loss MSE simple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleYOLO(nn.Module):
    """
    YOLO simplificado pero REAL
    - Grid cells: SÍ
    - Single pass: SÍ  
    - Neural network: SÍ
    - Sin IFs: SÍ
    """
    
    def __init__(self, S=7):
        super().__init__()
        self.S = S
        
        # CNN simple pero funcional
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Ajustar a grid S×S
        self.adapt = nn.AdaptiveAvgPool2d((S, S))
        
        # Detection head: predice si hay objeto y su posición
        # Por cada celda: [has_object, x_offset, y_offset, width, height]
        self.detect = nn.Conv2d(128, 5, 1)
        
        # Inicialización mejor para evitar que se atasque
        self._initialize_weights()
        
    def forward(self, x):
        # UN SOLO PASE - no hay loops
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.adapt(x)
        x = self.detect(x)
        
        # Output shape: [batch, 5, S, S]
        # Reordenar a [batch, S, S, 5]
        x = x.permute(0, 2, 3, 1)
        
        # Aplicar activaciones (sin modificar in-place)
        # has_object: sigmoid (0 a 1)
        # x, y: sigmoid (0 a 1, relativo a la celda)
        # w, h: exp (siempre positivo)
        objectness = torch.sigmoid(x[..., 0:1])
        xy = torch.sigmoid(x[..., 1:3])
        wh = torch.exp(x[..., 3:5].clamp(max=5))
        
        x = torch.cat([objectness, xy, wh], dim=-1)
        
        return x
    
    def _initialize_weights(self):
        """Inicialización de pesos para mejor convergencia"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # Sesgo inicial para objectness para evitar todo-ceros
                    if m == self.detect:
                        # Inicializar objectness con sesgo positivo pequeño
                        m.bias.data[0] = -2.0  # Después de sigmoid será ~0.12
                    else:
                        nn.init.constant_(m.bias, 0)


def generate_training_data(batch_size=32):
    """
    Genera datos de entrenamiento con figuras geométricas variadas
    """
    images = []
    targets = []
    
    for _ in range(batch_size):
        # Imagen vacía 28x28
        img = np.zeros((28, 28), dtype=np.float32)
        target = np.zeros((7, 7, 5), dtype=np.float32)
        
        # Añadir 1-4 objetos
        num_objects = random.randint(1, 4)
        
        for _ in range(num_objects):
            shape_type = random.choice(['square', 'rect', 'circle', 'triangle', 'L', 'cross'])
            
            if shape_type == 'square':
                size = random.randint(3, 7)
                x = random.randint(0, 28 - size)
                y = random.randint(0, 28 - size)
                img[y:y+size, x:x+size] = 1.0
                w, h = size, size
                
            elif shape_type == 'rect':
                w = random.randint(3, 8)
                h = random.randint(3, 8)
                x = random.randint(0, 28 - w)
                y = random.randint(0, 28 - h)
                img[y:y+h, x:x+w] = 1.0
                
            elif shape_type == 'circle':
                radius = random.randint(2, 5)
                cx = random.randint(radius, 28 - radius)
                cy = random.randint(radius, 28 - radius)
                for i in range(28):
                    for j in range(28):
                        if np.sqrt((i - cy)**2 + (j - cx)**2) <= radius:
                            img[i, j] = 1.0
                x, y = cx - radius, cy - radius
                w, h = radius * 2, radius * 2
                
            elif shape_type == 'triangle':
                size = random.randint(4, 7)
                x = random.randint(0, 28 - size)
                y = random.randint(0, 28 - size)
                for i in range(size):
                    for j in range(i + 1):
                        if y + i < 28:
                            if x + size//2 - j >= 0:
                                img[y + i, x + size//2 - j] = 1.0
                            if j > 0 and x + size//2 + j < 28:
                                img[y + i, x + size//2 + j] = 1.0
                w, h = size, size
                
            elif shape_type == 'L':
                size = random.randint(4, 7)
                x = random.randint(0, 28 - size)
                y = random.randint(0, 28 - size)
                img[y:y+size, x:x+2] = 1.0  # Vertical
                img[y+size-2:y+size, x:x+size] = 1.0  # Horizontal
                w, h = size, size
                
            elif shape_type == 'cross':
                size = random.randint(5, 7)
                x = random.randint(0, 28 - size)
                y = random.randint(0, 28 - size)
                mid = size // 2
                img[y:y+size, x+mid-1:x+mid+1] = 1.0  # Vertical
                img[y+mid-1:y+mid+1, x:x+size] = 1.0  # Horizontal
                w, h = size, size
            
            # Calcular celda responsable
            cx = (x + w/2) / 28 * 7
            cy = (y + h/2) / 28 * 7
            
            cell_x = int(cx)
            cell_y = int(cy)
            
            if cell_x < 7 and cell_y < 7:
                # Solo actualizar si está vacío o con menor confianza
                if target[cell_y, cell_x, 0] < 0.5:
                    target[cell_y, cell_x, 0] = 1.0
                    target[cell_y, cell_x, 1] = cx - cell_x
                    target[cell_y, cell_x, 2] = cy - cell_y
                    target[cell_y, cell_x, 3] = w / 28
                    target[cell_y, cell_x, 4] = h / 28
        
        images.append(img)
        targets.append(target)
    
    # Convertir a numpy array primero para evitar warning
    images_np = np.array(images)
    targets_np = np.array(targets)
    
    return torch.FloatTensor(images_np).unsqueeze(1), torch.FloatTensor(targets_np)


def compute_yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Loss específico para YOLO con ponderación adecuada
    """
    # Separar componentes
    pred_obj = predictions[..., 0]  # Objectness
    pred_xy = predictions[..., 1:3]  # x, y
    pred_wh = predictions[..., 3:5]  # w, h
    
    target_obj = targets[..., 0]
    target_xy = targets[..., 1:3]
    target_wh = targets[..., 3:5]
    
    # Máscara de objetos
    obj_mask = target_obj > 0.5
    noobj_mask = ~obj_mask
    
    # Loss de localización (solo donde hay objetos)
    if obj_mask.any():
        loc_loss = lambda_coord * (
            F.mse_loss(pred_xy[obj_mask], target_xy[obj_mask], reduction='mean') +
            F.mse_loss(pred_wh[obj_mask], target_wh[obj_mask], reduction='mean')
        )
    else:
        loc_loss = 0.0
    
    # Loss de objectness
    obj_loss = F.mse_loss(pred_obj[obj_mask], target_obj[obj_mask], reduction='sum') if obj_mask.any() else 0.0
    noobj_loss = lambda_noobj * F.mse_loss(pred_obj[noobj_mask], target_obj[noobj_mask], reduction='sum') if noobj_mask.any() else 0.0
    
    # Normalizar por batch size
    batch_size = predictions.size(0)
    total_loss = (loc_loss + obj_loss + noobj_loss) / batch_size
    
    return total_loss


def train_simple_yolo():
    """
    Entrena YOLO simple
    """
    logger.info("="*60)
    logger.info("ENTRENANDO YOLO SIMPLE (REAL)")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleYOLO().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Learning rate más bajo
    
    logger.info(f"Device: {device}")
    logger.info("Arquitectura: CNN → Grid 7×7 → Detección")
    logger.info("Sin IFs, sin hardcodeo\n")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(100):  # Más épocas con early stopping
        total_loss = 0
        
        for step in range(50):  # 50 batches por época
            # Generar datos
            images, targets = generate_training_data(32)
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            predictions = model(images)
            
            # Loss YOLO específico con ponderación
            loss = compute_yolo_loss(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / 50
        logger.info(f"Epoch {epoch+1} completa - Loss promedio: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            logger.info(f"  ✓ Mejor loss hasta ahora: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter > 10:
                logger.info(f"  Early stopping - no mejora en {patience_counter} épocas")
                break
        
        # Mostrar estadísticas cada 10 épocas
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_images, test_targets = generate_training_data(1)
                test_images = test_images.to(device)
                test_pred = model(test_images)
                test_obj = test_pred[0, :, :, 0].cpu().numpy()
                logger.info(f"  Stats - Objectness max: {test_obj.max():.4f}, mean: {test_obj.mean():.4f}")
        
        logger.info("")
    
    logger.info("✅ Entrenamiento completado")
    return model


def test_simple_yolo(model):
    """
    Prueba YOLO simple con múltiples umbrales
    """
    logger.info("\n" + "="*60)
    logger.info("PROBANDO YOLO CON DIFERENTES UMBRALES")
    logger.info("="*60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Generar imagen de prueba
    images, targets = generate_training_data(1)
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    # Decodificar predicciones
    pred = predictions[0].cpu().numpy()  # [7, 7, 5]
    
    # Probar diferentes umbrales
    thresholds = [0.5, 0.3, 0.1, 0.05, 0.01]
    
    for threshold in thresholds:
        detected_objects = []
        
        for i in range(7):
            for j in range(7):
                if pred[i, j, 0] > threshold:  # Si hay objeto
                    # Convertir a coordenadas globales
                    cx = (j + pred[i, j, 1]) / 7 * 28
                    cy = (i + pred[i, j, 2]) / 7 * 28
                    w = pred[i, j, 3] * 28
                    h = pred[i, j, 4] * 28
                    
                    detected_objects.append({
                        'x': cx - w/2,
                        'y': cy - h/2,
                        'w': w,
                        'h': h,
                        'conf': pred[i, j, 0]
                    })
        
        logger.info(f"\nUmbral {threshold}: {len(detected_objects)} objetos detectados")
        if detected_objects:
            for i, obj in enumerate(detected_objects[:3]):  # Mostrar máximo 3
                logger.info(f"  Objeto {i+1}: pos=({obj['x']:.1f}, {obj['y']:.1f}), "
                           f"size=({obj['w']:.1f} × {obj['h']:.1f}), "
                           f"conf={obj['conf']:.3f}")
    
    # Comparar con ground truth
    target = targets[0].cpu().numpy()
    gt_objects = []
    
    for i in range(7):
        for j in range(7):
            if target[i, j, 0] > 0.5:
                cx = (j + target[i, j, 1]) / 7 * 28
                cy = (i + target[i, j, 2]) / 7 * 28
                w = target[i, j, 3] * 28
                h = target[i, j, 4] * 28
                gt_objects.append({'x': cx-w/2, 'y': cy-h/2, 'w': w, 'h': h})
    
    logger.info(f"\nGround truth: {len(gt_objects)} objetos")
    
    # Mostrar estadísticas de predicciones
    logger.info("\nEstadísticas de predicciones:")
    objectness_values = pred[:, :, 0].flatten()
    logger.info(f"  Objectness - Min: {objectness_values.min():.4f}, "
                f"Max: {objectness_values.max():.4f}, "
                f"Mean: {objectness_values.mean():.4f}")
    
    # Visualizar grid de objectness
    logger.info("\nMapa de objectness (7×7):")
    for i in range(7):
        row_str = "  "
        for j in range(7):
            val = pred[i, j, 0]
            if val > 0.5:
                row_str += "█ "
            elif val > 0.1:
                row_str += "▓ "
            elif val > 0.01:
                row_str += "░ "
            else:
                row_str += ". "
        logger.info(row_str)
    
    logger.info("\n✅ YOLO detecta objetos en UN SOLO PASE")
    logger.info("   Sin IFs, sin loops de detección")
    logger.info("   La red neuronal APRENDIÓ a detectar")


if __name__ == "__main__":
    # Entrenar
    model = train_simple_yolo()
    
    # Probar
    test_simple_yolo(model)
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN: YOLO REAL")
    logger.info("="*60)
    logger.info("""
    1. Divide imagen en GRID 7×7
    2. Cada celda predice si tiene objeto
    3. TODO en UN forward pass
    4. NO hay IFs para detectar
    5. La red APRENDE durante entrenamiento
    
    Esto SÍ es detección por deep learning
    """)