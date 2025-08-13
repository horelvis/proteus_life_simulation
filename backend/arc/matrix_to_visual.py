#!/usr/bin/env python3
"""
Conversor de Matrices ARC a Imágenes Visuales
Convierte matrices numéricas en imágenes reales que el sistema puede VER y procesar
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Paleta de colores oficial de ARC
ARC_COLORS = {
    0: (0, 0, 0),        # Negro
    1: (0, 116, 217),    # Azul
    2: (255, 65, 54),    # Rojo
    3: (46, 204, 64),    # Verde
    4: (255, 220, 0),    # Amarillo
    5: (170, 170, 170),  # Gris
    6: (240, 18, 190),   # Magenta
    7: (255, 133, 27),   # Naranja
    8: (127, 219, 255),  # Cyan
    9: (135, 12, 37)     # Marrón
}

class MatrixToVisual:
    """
    Convierte matrices numéricas en imágenes visuales que el sistema puede VER
    """
    
    def __init__(self, cell_size: int = 30, grid_lines: bool = True):
        """
        Args:
            cell_size: Tamaño de cada celda en píxeles
            grid_lines: Si dibujar líneas de grilla
        """
        self.cell_size = cell_size
        self.grid_lines = grid_lines
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transformación para convertir PIL a tensor
        self.to_tensor = transforms.ToTensor()
        
    def matrix_to_image(self, matrix: np.ndarray, 
                       save_path: Optional[str] = None) -> Image.Image:
        """
        Convierte una matriz numérica en una imagen visual real
        
        Args:
            matrix: Matriz numérica de ARC (valores 0-9)
            save_path: Ruta opcional para guardar la imagen
            
        Returns:
            Imagen PIL que el sistema puede VER
        """
        if isinstance(matrix, list):
            matrix = np.array(matrix)
            
        h, w = matrix.shape
        
        # Crear imagen con el tamaño correcto
        img_width = w * self.cell_size
        img_height = h * self.cell_size
        
        # Añadir espacio para líneas de grilla si están habilitadas
        if self.grid_lines:
            img_width += (w + 1)
            img_height += (h + 1)
        
        # Crear imagen RGB
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Dibujar cada celda con su color
        for i in range(h):
            for j in range(w):
                value = int(matrix[i, j])
                color = ARC_COLORS.get(value, (200, 200, 200))
                
                # Calcular posición de la celda
                if self.grid_lines:
                    x1 = j * (self.cell_size + 1) + 1
                    y1 = i * (self.cell_size + 1) + 1
                else:
                    x1 = j * self.cell_size
                    y1 = i * self.cell_size
                    
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Dibujar rectángulo con el color
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Dibujar líneas de grilla si están habilitadas
        if self.grid_lines:
            # Líneas horizontales
            for i in range(h + 1):
                y = i * (self.cell_size + 1)
                draw.line([(0, y), (img_width, y)], fill=(128, 128, 128), width=1)
            
            # Líneas verticales
            for j in range(w + 1):
                x = j * (self.cell_size + 1)
                draw.line([(x, 0), (x, img_height)], fill=(128, 128, 128), width=1)
        
        # Guardar si se especifica ruta
        if save_path:
            image.save(save_path)
            logger.info(f"📸 Imagen guardada en: {save_path}")
        
        return image
    
    def puzzle_to_visual_pair(self, input_matrix: np.ndarray, 
                            output_matrix: np.ndarray,
                            save_path: Optional[str] = None) -> Tuple[Image.Image, Image.Image]:
        """
        Convierte un par input/output en imágenes visuales
        
        Returns:
            Tupla de (imagen_input, imagen_output)
        """
        input_img = self.matrix_to_image(input_matrix)
        output_img = self.matrix_to_image(output_matrix)
        
        if save_path:
            # Crear imagen combinada
            total_width = input_img.width + output_img.width + 20
            max_height = max(input_img.height, output_img.height)
            
            combined = Image.new('RGB', (total_width, max_height), color='white')
            combined.paste(input_img, (0, 0))
            combined.paste(output_img, (input_img.width + 20, 0))
            
            combined.save(save_path)
            logger.info(f"📸 Par visual guardado en: {save_path}")
        
        return input_img, output_img
    
    def create_visual_dataset(self, puzzle_json: Dict, 
                            output_dir: str = "/tmp/arc_visual"):
        """
        Crea un dataset visual completo desde un puzzle JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        puzzle_id = puzzle_json.get('id', 'unknown')
        
        # Procesar ejemplos de entrenamiento
        for i, example in enumerate(puzzle_json.get('train', [])):
            input_matrix = np.array(example['input'])
            output_matrix = np.array(example['output'])
            
            # Guardar imágenes individuales
            input_path = output_dir / f"{puzzle_id}_train_{i}_input.png"
            output_path = output_dir / f"{puzzle_id}_train_{i}_output.png"
            combined_path = output_dir / f"{puzzle_id}_train_{i}_combined.png"
            
            self.matrix_to_image(input_matrix, input_path)
            self.matrix_to_image(output_matrix, output_path)
            self.puzzle_to_visual_pair(input_matrix, output_matrix, combined_path)
        
        # Procesar ejemplos de test
        for i, example in enumerate(puzzle_json.get('test', [])):
            input_matrix = np.array(example['input'])
            
            input_path = output_dir / f"{puzzle_id}_test_{i}_input.png"
            self.matrix_to_image(input_matrix, input_path)
            
            # Si hay output esperado
            if 'output' in example and example['output']:
                output_matrix = np.array(example['output'])
                output_path = output_dir / f"{puzzle_id}_test_{i}_output.png"
                self.matrix_to_image(output_matrix, output_path)
        
        logger.info(f"✅ Dataset visual creado en: {output_dir}")
        return output_dir
    
    def visualize_attention_on_image(self, matrix: np.ndarray,
                                    attention_map: np.ndarray,
                                    save_path: Optional[str] = None) -> Image.Image:
        """
        Visualiza un mapa de atención sobre la imagen
        """
        # Crear imagen base
        base_img = self.matrix_to_image(matrix)
        
        # Redimensionar mapa de atención al tamaño de la imagen
        from scipy.ndimage import zoom
        
        h, w = matrix.shape
        img_h, img_w = base_img.height, base_img.width
        
        # Escalar mapa de atención
        scale_h = img_h / attention_map.shape[0]
        scale_w = img_w / attention_map.shape[1]
        
        attention_resized = zoom(attention_map, (scale_h, scale_w), order=1)
        
        # Normalizar atención a 0-255
        attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        attention_uint8 = (attention_norm * 255).astype(np.uint8)
        
        # Crear overlay de atención
        attention_img = Image.fromarray(attention_uint8, mode='L')
        attention_colored = Image.new('RGBA', base_img.size, (255, 0, 0, 0))
        attention_colored.putalpha(attention_img)
        
        # Combinar imagen base con atención
        base_rgba = base_img.convert('RGBA')
        combined = Image.alpha_composite(base_rgba, attention_colored)
        
        if save_path:
            combined.save(save_path)
            logger.info(f"📸 Atención visual guardada en: {save_path}")
        
        return combined
    
    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convierte imagen PIL a tensor para procesamiento en GPU
        """
        tensor = self.to_tensor(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def compare_visual_similarity(self, img1: Image.Image, 
                                 img2: Image.Image) -> float:
        """
        Compara similitud visual entre dos imágenes
        """
        # Convertir a tensores
        tensor1 = self.image_to_tensor(img1)
        tensor2 = self.image_to_tensor(img2)
        
        # Redimensionar si es necesario
        if tensor1.shape != tensor2.shape:
            # Redimensionar al tamaño más pequeño
            min_h = min(tensor1.shape[2], tensor2.shape[2])
            min_w = min(tensor1.shape[3], tensor2.shape[3])
            
            resize = transforms.Resize((min_h, min_w))
            tensor1 = resize(tensor1)
            tensor2 = resize(tensor2)
        
        # Calcular similitud coseno
        tensor1_flat = tensor1.flatten()
        tensor2_flat = tensor2.flatten()
        
        similarity = torch.nn.functional.cosine_similarity(
            tensor1_flat.unsqueeze(0),
            tensor2_flat.unsqueeze(0)
        )
        
        return similarity.item()


def demonstrate_visual_conversion():
    """
    Demuestra la conversión de matrices a imágenes visuales
    """
    print("="*60)
    print("🎨 CONVERSIÓN DE MATRICES A IMÁGENES VISUALES")
    print("="*60)
    
    # Crear conversor
    converter = MatrixToVisual(cell_size=40, grid_lines=True)
    
    # Ejemplo 1: Matriz simple
    matrix1 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    
    print("\n📊 Matriz numérica:")
    print(matrix1)
    
    # Convertir a imagen
    img1 = converter.matrix_to_image(matrix1, "/tmp/arc_visual_example1.png")
    print("✅ Convertido a imagen visual: /tmp/arc_visual_example1.png")
    
    # Ejemplo 2: Matriz más compleja
    matrix2 = np.array([
        [0, 0, 2, 2, 0],
        [0, 3, 3, 3, 0],
        [4, 4, 4, 4, 4],
        [0, 5, 0, 5, 0],
        [6, 0, 7, 0, 8]
    ])
    
    print("\n📊 Matriz compleja:")
    print(matrix2)
    
    img2 = converter.matrix_to_image(matrix2, "/tmp/arc_visual_example2.png")
    print("✅ Convertido a imagen visual: /tmp/arc_visual_example2.png")
    
    # Calcular similitud visual
    similarity = converter.compare_visual_similarity(img1, img2)
    print(f"\n🔍 Similitud visual entre imágenes: {similarity:.2%}")
    
    # Cargar un puzzle real si existe
    try:
        puzzle_path = Path("/app/arc_official_cache/arc_agi_1_training_0520fde7.json")
        if puzzle_path.exists():
            with open(puzzle_path, 'r') as f:
                puzzle = json.load(f)
            
            print("\n📂 Convirtiendo puzzle real a imágenes...")
            output_dir = converter.create_visual_dataset(
                puzzle, 
                output_dir="/tmp/arc_visual_dataset"
            )
            print(f"✅ Dataset visual creado en: {output_dir}")
            
            # Mostrar primera imagen
            first_train = puzzle['train'][0]
            input_matrix = np.array(first_train['input'])
            output_matrix = np.array(first_train['output'])
            
            print(f"\n🖼️ Puzzle shape: Input {input_matrix.shape} → Output {output_matrix.shape}")
            print("   Las matrices numéricas ahora son IMÁGENES VISUALES")
            print("   El sistema puede VER colores, formas y patrones")
            print("   No procesa números abstractos")
    except Exception as e:
        print(f"⚠️ No se pudo cargar puzzle real: {e}")
    
    print("\n" + "="*60)
    print("💡 CONCLUSIÓN:")
    print("   - Las matrices se convierten en imágenes REALES")
    print("   - El sistema VE colores y formas, no números")
    print("   - Procesa visualmente como un humano")
    print("   - Puede aplicar atención visual sobre las imágenes")
    print("="*60)


if __name__ == "__main__":
    demonstrate_visual_conversion()