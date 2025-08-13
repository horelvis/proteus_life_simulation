#!/usr/bin/env python3
"""
V-JEPA AVANZADO - Entrenamiento con formas más complejas y variadas
Incluye círculos, elipses, polígonos, curvas Bezier, etc.
Opción de usar ImageNet si está disponible
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import logging
from typing import Tuple, Optional
import cv2
from scipy import ndimage
from skimage import draw
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisualDataset(Dataset):
    """
    Dataset con formas geométricas avanzadas y transformaciones complejas
    """
    
    def __init__(self, num_samples: int = 10000, grid_size: int = 64, use_imagenet: bool = False):
        self.samples = []
        self.grid_size = grid_size
        self.use_imagenet = use_imagenet
        
        logger.info(f"Generando {num_samples} imágenes con formas avanzadas...")
        logger.info(f"Tamaño de grid: {grid_size}x{grid_size}")
        
        # Si ImageNet está disponible, cargar algunas texturas
        self.textures = self._load_textures() if use_imagenet else None
        
        for i in range(num_samples):
            # Generar imagen con formas complejas
            img = self._generate_complex_image()
            self.samples.append(img)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generados {i + 1}/{num_samples} ejemplos...")
    
    def _load_textures(self):
        """Intenta cargar texturas de ImageNet o genera sintéticas"""
        try:
            # Aquí podrías cargar ImageNet si está disponible
            # Por ahora generamos texturas sintéticas
            logger.info("Generando texturas sintéticas (ImageNet no configurado)")
            textures = []
            for _ in range(10):
                # Generar textura procedural
                texture = self._generate_procedural_texture()
                textures.append(texture)
            return textures
        except:
            return None
    
    def _generate_procedural_texture(self):
        """Genera texturas procedurales (Perlin noise, patrones, etc.)"""
        size = self.grid_size
        texture = np.zeros((size, size))
        
        pattern_type = np.random.choice(['perlin', 'checkerboard', 'stripes', 'dots', 'waves'])
        
        if pattern_type == 'perlin':
            # Perlin noise simplificado
            freq = np.random.uniform(0.05, 0.2)
            for i in range(size):
                for j in range(size):
                    texture[i, j] = np.sin(i * freq) * np.cos(j * freq)
                    
        elif pattern_type == 'checkerboard':
            square_size = np.random.randint(4, 16)
            for i in range(size):
                for j in range(size):
                    texture[i, j] = ((i // square_size) + (j // square_size)) % 2
                    
        elif pattern_type == 'stripes':
            angle = np.random.uniform(0, np.pi)
            freq = np.random.uniform(0.1, 0.5)
            for i in range(size):
                for j in range(size):
                    texture[i, j] = np.sin((i * np.cos(angle) + j * np.sin(angle)) * freq)
                    
        elif pattern_type == 'dots':
            dot_spacing = np.random.randint(8, 16)
            dot_radius = np.random.randint(2, 4)
            for i in range(0, size, dot_spacing):
                for j in range(0, size, dot_spacing):
                    rr, cc = draw.disk((i, j), dot_radius, shape=(size, size))
                    texture[rr, cc] = 1
                    
        else:  # waves
            freq_x = np.random.uniform(0.1, 0.3)
            freq_y = np.random.uniform(0.1, 0.3)
            for i in range(size):
                for j in range(size):
                    texture[i, j] = np.sin(i * freq_x) + np.sin(j * freq_y)
        
        # Normalizar a [0, 1]
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        return texture
    
    def _generate_complex_image(self):
        """Genera una imagen con formas geométricas complejas"""
        img = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Número de formas a dibujar
        num_shapes = np.random.randint(1, 6)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice([
                'circle', 'ellipse', 'triangle', 'polygon', 
                'star', 'bezier', 'spiral', 'fractal_tree',
                'sine_wave', 'heart', 'arrow', 'cross'
            ])
            
            # Color/intensidad de la forma
            intensity = np.random.uniform(0.3, 1.0)
            
            if shape_type == 'circle':
                self._draw_circle(img, intensity)
            elif shape_type == 'ellipse':
                self._draw_ellipse(img, intensity)
            elif shape_type == 'triangle':
                self._draw_triangle(img, intensity)
            elif shape_type == 'polygon':
                self._draw_polygon(img, intensity)
            elif shape_type == 'star':
                self._draw_star(img, intensity)
            elif shape_type == 'bezier':
                self._draw_bezier_curve(img, intensity)
            elif shape_type == 'spiral':
                self._draw_spiral(img, intensity)
            elif shape_type == 'fractal_tree':
                self._draw_fractal_tree(img, intensity)
            elif shape_type == 'sine_wave':
                self._draw_sine_wave(img, intensity)
            elif shape_type == 'heart':
                self._draw_heart(img, intensity)
            elif shape_type == 'arrow':
                self._draw_arrow(img, intensity)
            else:  # cross
                self._draw_cross(img, intensity)
        
        # Aplicar transformación global aleatoria
        if np.random.random() > 0.5:
            img = self._apply_transformation(img)
        
        # Aplicar textura si está disponible
        if self.textures and np.random.random() > 0.7:
            texture_idx = np.random.randint(len(self.textures))
            texture = self.textures[texture_idx]
            img = img * 0.7 + texture * 0.3
        
        # Añadir ruido
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 0.05, img.shape)
            img = img + noise
        
        # Clip a [0, 1]
        img = np.clip(img, 0, 1)
        
        return img
    
    def _draw_circle(self, img, intensity):
        """Dibuja un círculo"""
        center_y = np.random.randint(10, self.grid_size - 10)
        center_x = np.random.randint(10, self.grid_size - 10)
        radius = np.random.randint(3, min(20, self.grid_size // 4))
        
        rr, cc = draw.disk((center_y, center_x), radius, shape=img.shape)
        img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_ellipse(self, img, intensity):
        """Dibuja una elipse"""
        center_y = np.random.randint(10, self.grid_size - 10)
        center_x = np.random.randint(10, self.grid_size - 10)
        radius_y = np.random.randint(3, min(15, self.grid_size // 4))
        radius_x = np.random.randint(3, min(15, self.grid_size // 4))
        rotation = np.random.uniform(0, np.pi)
        
        rr, cc = draw.ellipse(center_y, center_x, radius_y, radius_x, 
                              shape=img.shape, rotation=rotation)
        img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_triangle(self, img, intensity):
        """Dibuja un triángulo"""
        # Generar 3 puntos aleatorios
        points = []
        for _ in range(3):
            y = np.random.randint(5, self.grid_size - 5)
            x = np.random.randint(5, self.grid_size - 5)
            points.append([y, x])
        
        triangle = np.array(points)
        rr, cc = draw.polygon(triangle[:, 0], triangle[:, 1], shape=img.shape)
        img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_polygon(self, img, intensity):
        """Dibuja un polígono regular"""
        num_vertices = np.random.randint(5, 9)
        center_y = self.grid_size // 2 + np.random.randint(-15, 15)
        center_x = self.grid_size // 2 + np.random.randint(-15, 15)
        radius = np.random.randint(5, min(20, self.grid_size // 3))
        
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        vertices_y = center_y + radius * np.sin(angles)
        vertices_x = center_x + radius * np.cos(angles)
        
        rr, cc = draw.polygon(vertices_y, vertices_x, shape=img.shape)
        img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_star(self, img, intensity):
        """Dibuja una estrella"""
        num_points = np.random.choice([5, 6, 8])
        center_y = self.grid_size // 2 + np.random.randint(-10, 10)
        center_x = self.grid_size // 2 + np.random.randint(-10, 10)
        outer_radius = np.random.randint(8, min(20, self.grid_size // 3))
        inner_radius = outer_radius // 2
        
        angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
        vertices_y = []
        vertices_x = []
        
        for i, angle in enumerate(angles):
            if i % 2 == 0:
                r = outer_radius
            else:
                r = inner_radius
            vertices_y.append(center_y + r * np.sin(angle))
            vertices_x.append(center_x + r * np.cos(angle))
        
        rr, cc = draw.polygon(vertices_y, vertices_x, shape=img.shape)
        img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_bezier_curve(self, img, intensity):
        """Dibuja una curva de Bezier"""
        # Puntos de control
        p0 = [np.random.randint(5, self.grid_size-5), np.random.randint(5, self.grid_size-5)]
        p1 = [np.random.randint(5, self.grid_size-5), np.random.randint(5, self.grid_size-5)]
        p2 = [np.random.randint(5, self.grid_size-5), np.random.randint(5, self.grid_size-5)]
        p3 = [np.random.randint(5, self.grid_size-5), np.random.randint(5, self.grid_size-5)]
        
        # Generar puntos de la curva
        t = np.linspace(0, 1, 100)
        curve_points = []
        
        for ti in t:
            # Fórmula de Bezier cúbica
            point = ((1-ti)**3 * np.array(p0) + 
                    3*(1-ti)**2*ti * np.array(p1) + 
                    3*(1-ti)*ti**2 * np.array(p2) + 
                    ti**3 * np.array(p3))
            curve_points.append(point)
        
        # Dibujar la curva
        curve_points = np.array(curve_points).astype(int)
        for i in range(len(curve_points) - 1):
            rr, cc = draw.line(curve_points[i][0], curve_points[i][1],
                              curve_points[i+1][0], curve_points[i+1][1])
            # Validar índices
            valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
            rr = rr[valid_mask]
            cc = cc[valid_mask]
            img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_spiral(self, img, intensity):
        """Dibuja una espiral"""
        center_y = self.grid_size // 2
        center_x = self.grid_size // 2
        
        theta = np.linspace(0, 4 * np.pi, 200)
        r = np.linspace(0, min(25, self.grid_size // 3), 200)
        
        points = []
        for ti, ri in zip(theta, r):
            y = int(center_y + ri * np.sin(ti))
            x = int(center_x + ri * np.cos(ti))
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                points.append([y, x])
        
        # Dibujar la espiral
        for i in range(len(points) - 1):
            rr, cc = draw.line(points[i][0], points[i][1],
                              points[i+1][0], points[i+1][1])
            valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
            rr = rr[valid_mask]
            cc = cc[valid_mask]
            img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_fractal_tree(self, img, intensity, depth=4):
        """Dibuja un árbol fractal simple"""
        def draw_branch(y1, x1, angle, length, depth_remaining):
            if depth_remaining == 0 or length < 2:
                return
            
            # Calcular punto final
            y2 = int(y1 - length * np.cos(angle))
            x2 = int(x1 + length * np.sin(angle))
            
            # Dibujar línea
            if (0 <= y1 < self.grid_size and 0 <= x1 < self.grid_size and
                0 <= y2 < self.grid_size and 0 <= x2 < self.grid_size):
                rr, cc = draw.line(y1, x1, y2, x2)
                valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
                rr = rr[valid_mask]
                cc = cc[valid_mask]
                img[rr, cc] = np.maximum(img[rr, cc], intensity * (depth_remaining / depth))
                
                # Dibujar ramas
                angle_delta = np.pi / 6
                draw_branch(y2, x2, angle - angle_delta, length * 0.7, depth_remaining - 1)
                draw_branch(y2, x2, angle + angle_delta, length * 0.7, depth_remaining - 1)
        
        # Iniciar desde la parte inferior
        start_y = self.grid_size - 5
        start_x = self.grid_size // 2
        initial_length = min(15, self.grid_size // 4)
        
        draw_branch(start_y, start_x, np.pi/2, initial_length, depth)
    
    def _draw_sine_wave(self, img, intensity):
        """Dibuja una onda sinusoidal"""
        amplitude = np.random.randint(5, min(15, self.grid_size // 4))
        frequency = np.random.uniform(0.05, 0.2)
        phase = np.random.uniform(0, 2 * np.pi)
        y_offset = self.grid_size // 2
        
        points = []
        for x in range(self.grid_size):
            y = int(y_offset + amplitude * np.sin(frequency * x + phase))
            if 0 <= y < self.grid_size:
                points.append([y, x])
        
        # Dibujar la onda
        for i in range(len(points) - 1):
            rr, cc = draw.line(points[i][0], points[i][1],
                              points[i+1][0], points[i+1][1])
            valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
            rr = rr[valid_mask]
            cc = cc[valid_mask]
            img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_heart(self, img, intensity):
        """Dibuja un corazón"""
        center_y = self.grid_size // 2
        center_x = self.grid_size // 2
        size = min(15, self.grid_size // 3)
        
        t = np.linspace(0, 2 * np.pi, 100)
        # Ecuación paramétrica del corazón
        x = size * (16 * np.sin(t)**3) / 16
        y = -size * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) / 16
        
        # Trasladar al centro
        x = x + center_x
        y = y + center_y
        
        # Crear polígono
        vertices = np.column_stack((y, x)).astype(int)
        valid_vertices = []
        for v in vertices:
            if 0 <= v[0] < self.grid_size and 0 <= v[1] < self.grid_size:
                valid_vertices.append(v)
        
        if len(valid_vertices) > 2:
            valid_vertices = np.array(valid_vertices)
            rr, cc = draw.polygon(valid_vertices[:, 0], valid_vertices[:, 1], shape=img.shape)
            img[rr, cc] = np.maximum(img[rr, cc], intensity)
    
    def _draw_arrow(self, img, intensity):
        """Dibuja una flecha"""
        # Punto de inicio y dirección aleatorios
        start_y = np.random.randint(10, self.grid_size - 10)
        start_x = np.random.randint(10, self.grid_size - 10)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(10, min(25, self.grid_size // 2))
        
        # Línea principal
        end_y = int(start_y + length * np.sin(angle))
        end_x = int(start_x + length * np.cos(angle))
        
        if 0 <= end_y < self.grid_size and 0 <= end_x < self.grid_size:
            rr, cc = draw.line(start_y, start_x, end_y, end_x)
            valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
            img[rr[valid_mask], cc[valid_mask]] = np.maximum(
                img[rr[valid_mask], cc[valid_mask]], intensity)
            
            # Punta de la flecha
            arrow_length = length // 3
            arrow_angle = np.pi / 6
            
            # Rama izquierda
            arrow_y1 = int(end_y - arrow_length * np.sin(angle - arrow_angle))
            arrow_x1 = int(end_x - arrow_length * np.cos(angle - arrow_angle))
            
            # Rama derecha
            arrow_y2 = int(end_y - arrow_length * np.sin(angle + arrow_angle))
            arrow_x2 = int(end_x - arrow_length * np.cos(angle + arrow_angle))
            
            if 0 <= arrow_y1 < self.grid_size and 0 <= arrow_x1 < self.grid_size:
                rr, cc = draw.line(end_y, end_x, arrow_y1, arrow_x1)
                valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
                img[rr[valid_mask], cc[valid_mask]] = np.maximum(
                    img[rr[valid_mask], cc[valid_mask]], intensity)
            
            if 0 <= arrow_y2 < self.grid_size and 0 <= arrow_x2 < self.grid_size:
                rr, cc = draw.line(end_y, end_x, arrow_y2, arrow_x2)
                valid_mask = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
                img[rr[valid_mask], cc[valid_mask]] = np.maximum(
                    img[rr[valid_mask], cc[valid_mask]], intensity)
    
    def _draw_cross(self, img, intensity):
        """Dibuja una cruz"""
        center_y = np.random.randint(10, self.grid_size - 10)
        center_x = np.random.randint(10, self.grid_size - 10)
        size = np.random.randint(5, min(20, self.grid_size // 3))
        thickness = np.random.randint(1, 3)
        
        # Línea horizontal
        for t in range(-thickness, thickness + 1):
            y = center_y + t
            if 0 <= y < self.grid_size:
                x_start = max(0, center_x - size)
                x_end = min(self.grid_size, center_x + size)
                img[y, x_start:x_end] = np.maximum(img[y, x_start:x_end], intensity)
        
        # Línea vertical
        for t in range(-thickness, thickness + 1):
            x = center_x + t
            if 0 <= x < self.grid_size:
                y_start = max(0, center_y - size)
                y_end = min(self.grid_size, center_y + size)
                img[y_start:y_end, x] = np.maximum(img[y_start:y_end, x], intensity)
    
    def _apply_transformation(self, img):
        """Aplica transformaciones globales a la imagen"""
        transform_type = np.random.choice(['rotate', 'flip', 'swirl', 'elastic', 'blur'])
        
        if transform_type == 'rotate':
            angle = np.random.uniform(0, 360)
            img = ndimage.rotate(img, angle, reshape=False, mode='constant')
            
        elif transform_type == 'flip':
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
                
        elif transform_type == 'swirl':
            # Transformación swirl simple
            center = self.grid_size // 2
            strength = np.random.uniform(0.5, 2.0)
            
            y, x = np.ogrid[:self.grid_size, :self.grid_size]
            dy = y - center
            dx = x - center
            r = np.sqrt(dy**2 + dx**2)
            angle = strength * np.exp(-r / (self.grid_size / 3))
            
            new_y = dy * np.cos(angle) - dx * np.sin(angle) + center
            new_x = dy * np.sin(angle) + dx * np.cos(angle) + center
            
            # Mapear coordenadas
            from scipy.interpolate import griddata
            points = np.column_stack((new_y.ravel(), new_x.ravel()))
            values = img.ravel()
            grid_y, grid_x = np.mgrid[0:self.grid_size, 0:self.grid_size]
            img = griddata(points, values, (grid_y, grid_x), method='linear', fill_value=0)
            
        elif transform_type == 'elastic':
            # Deformación elástica
            alpha = self.grid_size * 2
            sigma = self.grid_size * 0.08
            
            random_state = np.random.RandomState(None)
            shape = img.shape
            
            dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
            dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
            
            y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            img = ndimage.map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
            
        else:  # blur
            sigma = np.random.uniform(0.5, 2.0)
            img = ndimage.gaussian_filter(img, sigma)
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples[idx]
        # Convertir a tensor
        return torch.FloatTensor(img).unsqueeze(0)  # Añadir canal


class AdvancedVJEPAEncoder(nn.Module):
    """
    Encoder más profundo y potente para características complejas
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        # ResNet-like blocks
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Proyección final
        self.fc = nn.Linear(512, embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final projection
        x = self.fc(x)
        
        return x


def train_advanced_vjepa(
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_samples: int = 20000,
    grid_size: int = 64,
    save_dir: str = "/app/arc/vjepa_advanced_weights"
):
    """
    Entrena V-JEPA avanzado con formas complejas
    """
    logger.info("=== V-JEPA AVANZADO - Entrenamiento con formas complejas ===")
    logger.info(f"Configuración: {num_epochs} épocas, {num_samples} muestras, grid {grid_size}x{grid_size}")
    
    # Dataset
    dataset = AdvancedVisualDataset(num_samples=num_samples, grid_size=grid_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    encoder = AdvancedVJEPAEncoder(embedding_dim=512).to(device)
    
    # Optimizador con warmup
    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader)
    )
    
    # Loss - contrastive learning
    criterion = nn.CosineEmbeddingLoss()
    
    logger.info("Iniciando entrenamiento...")
    
    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            batch_size_actual = batch.size(0)
            
            # Crear pares positivos y negativos
            # Positivos: misma imagen con diferentes augmentaciones
            # Negativos: diferentes imágenes
            
            # Augmentaciones
            aug1 = batch
            aug2 = torch.flip(batch, dims=[2])  # Flip horizontal
            
            # Codificar
            z1 = encoder(aug1)
            z2 = encoder(aug2)
            
            # Labels: 1 para pares positivos
            labels = torch.ones(batch_size_actual).to(device)
            
            # Loss
            loss = criterion(z1, z2, labels)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Época {epoch+1} completada - Loss promedio: {avg_loss:.6f}")
        
        # Guardar checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint_advanced(encoder, epoch, avg_loss, save_dir)
    
    # Guardar modelo final
    save_checkpoint_advanced(encoder, num_epochs - 1, avg_loss, save_dir, final=True)
    logger.info("✅ Entrenamiento completado")
    
    return encoder


def save_checkpoint_advanced(encoder, epoch, loss, save_dir, final=False):
    """Guarda checkpoint del modelo avanzado"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    if final:
        filename = save_path / "vjepa_advanced_final.pth"
    else:
        filename = save_path / f"vjepa_advanced_epoch_{epoch+1}.pth"
    
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'embedding_dim': encoder.embedding_dim,
        'architecture': 'advanced_vjepa'
    }, filename)
    
    file_size = filename.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"💾 Checkpoint guardado: {filename.name} ({file_size:.1f} MB)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 5
        args.samples = 1000
        args.batch_size = 16
        args.grid_size = 32
        logger.info(f"MODO RÁPIDO: {args.epochs} épocas, {args.samples} muestras, grid {args.grid_size}")
    
    if args.train:
        train_advanced_vjepa(
            num_epochs=args.epochs,
            num_samples=args.samples,
            batch_size=args.batch_size,
            grid_size=args.grid_size
        )