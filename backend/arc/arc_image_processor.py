#!/usr/bin/env python3
"""
Procesador de imágenes para puzzles ARC
Convierte imágenes a grids de ARC detectando colores
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ARCImageProcessor:
    """
    Procesa imágenes para convertirlas en grids de ARC
    """
    
    def __init__(self):
        # Paleta de colores ARC estándar (RGB)
        self.arc_colors = {
            0: (0, 0, 0),       # Negro
            1: (0, 116, 217),   # Azul
            2: (255, 65, 54),   # Rojo
            3: (46, 204, 64),   # Verde
            4: (255, 220, 0),   # Amarillo
            5: (170, 170, 170), # Gris
            6: (240, 18, 190),  # Magenta
            7: (255, 133, 27),  # Naranja
            8: (127, 219, 255), # Cyan
            9: (135, 12, 37),   # Marrón
        }
        
        # Invertir para búsqueda rápida
        self.rgb_to_arc = {v: k for k, v in self.arc_colors.items()}
        
    def image_to_grid(self, image_data: str, cell_size: int = 30) -> Dict[str, Any]:
        """
        Convierte una imagen base64 a un grid de ARC
        
        Args:
            image_data: Imagen en base64
            cell_size: Tamaño estimado de cada celda en píxeles
            
        Returns:
            Dict con el grid y metadatos
        """
        try:
            # Decodificar imagen base64
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detectar el grid
            grid = self._detect_grid(image, cell_size)
            
            return {
                'success': True,
                'grid': grid,
                'dimensions': (len(grid), len(grid[0]) if grid else 0),
                'original_size': image.size
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_grid(self, image: Image, cell_size: int) -> List[List[int]]:
        """
        Detecta el grid en la imagen
        
        Args:
            image: Imagen PIL
            cell_size: Tamaño estimado de celda
            
        Returns:
            Grid de valores ARC
        """
        width, height = image.size
        
        # Estimar dimensiones del grid
        cols = width // cell_size
        rows = height // cell_size
        
        # Ajustar si es necesario
        if cols == 0 or rows == 0:
            # Intentar detectar automáticamente
            cols, rows, cell_size = self._auto_detect_grid_size(image)
        
        grid = []
        
        for row in range(rows):
            grid_row = []
            for col in range(cols):
                # Obtener el color dominante en esta celda
                x1 = col * cell_size + cell_size // 4
                y1 = row * cell_size + cell_size // 4
                x2 = min((col + 1) * cell_size - cell_size // 4, width)
                y2 = min((row + 1) * cell_size - cell_size // 4, height)
                
                cell_region = image.crop((x1, y1, x2, y2))
                dominant_color = self._get_dominant_color(cell_region)
                arc_value = self._match_arc_color(dominant_color)
                
                grid_row.append(arc_value)
            
            grid.append(grid_row)
        
        return grid
    
    def _auto_detect_grid_size(self, image: Image) -> Tuple[int, int, int]:
        """
        Intenta detectar automáticamente el tamaño del grid
        
        Args:
            image: Imagen PIL
            
        Returns:
            (columnas, filas, tamaño_celda)
        """
        width, height = image.size
        pixels = np.array(image)
        
        # Detectar líneas de grid buscando cambios bruscos
        # Simplificado: asumimos grids comunes de ARC
        common_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
        
        best_size = 30
        best_score = 0
        
        for size in common_sizes:
            if width % size == 0 and height % size == 0:
                cols = width // size
                rows = height // size
                
                # Verificar si parece un grid válido
                if 3 <= cols <= 30 and 3 <= rows <= 30:
                    score = 1.0 / abs(10 - cols) + 1.0 / abs(10 - rows)
                    if score > best_score:
                        best_score = score
                        best_size = size
        
        cols = width // best_size
        rows = height // best_size
        
        return cols, rows, best_size
    
    def _get_dominant_color(self, image_region: Image) -> Tuple[int, int, int]:
        """
        Obtiene el color dominante en una región
        
        Args:
            image_region: Región de la imagen
            
        Returns:
            Color RGB dominante
        """
        # Redimensionar para acelerar el proceso
        small = image_region.resize((10, 10))
        pixels = np.array(small)
        
        # Obtener el color más común
        pixels_flat = pixels.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels_flat, axis=0, return_counts=True)
        
        if len(unique_colors) > 0:
            dominant_idx = np.argmax(counts)
            return tuple(unique_colors[dominant_idx])
        
        return (0, 0, 0)  # Negro por defecto
    
    def _match_arc_color(self, rgb: Tuple[int, int, int]) -> int:
        """
        Encuentra el color ARC más cercano
        
        Args:
            rgb: Color RGB
            
        Returns:
            Valor ARC (0-9)
        """
        # Si es exacto, devolverlo
        if rgb in self.rgb_to_arc:
            return self.rgb_to_arc[rgb]
        
        # Encontrar el más cercano
        min_distance = float('inf')
        best_match = 0
        
        for arc_value, arc_rgb in self.arc_colors.items():
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, arc_rgb))
            if distance < min_distance:
                min_distance = distance
                best_match = arc_value
        
        return best_match
    
    def create_puzzle_from_images(self, 
                                 train_inputs: List[str],
                                 train_outputs: List[str],
                                 test_input: str,
                                 cell_size: int = 30) -> Dict[str, Any]:
        """
        Crea un puzzle completo desde imágenes
        
        Args:
            train_inputs: Lista de imágenes de entrada de entrenamiento
            train_outputs: Lista de imágenes de salida de entrenamiento
            test_input: Imagen de entrada de test
            cell_size: Tamaño de celda
            
        Returns:
            Puzzle en formato ARC
        """
        puzzle = {
            'id': f'image_puzzle_{np.random.randint(10000)}',
            'category': 'image_imported',
            'difficulty': 'unknown',
            'train': [],
            'test': []
        }
        
        # Procesar ejemplos de entrenamiento
        for inp_img, out_img in zip(train_inputs, train_outputs):
            inp_result = self.image_to_grid(inp_img, cell_size)
            out_result = self.image_to_grid(out_img, cell_size)
            
            if inp_result['success'] and out_result['success']:
                puzzle['train'].append({
                    'input': inp_result['grid'],
                    'output': out_result['grid']
                })
        
        # Procesar test
        test_result = self.image_to_grid(test_input, cell_size)
        if test_result['success']:
            puzzle['test'].append({
                'input': test_result['grid'],
                'output': []  # A resolver
            })
        
        return puzzle
    
    def analyze_image_puzzle(self, image_data: str) -> Dict[str, Any]:
        """
        Analiza una imagen que contiene múltiples grids de un puzzle
        
        Args:
            image_data: Imagen completa del puzzle
            
        Returns:
            Análisis y puzzle detectado
        """
        try:
            # Decodificar imagen
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detectar regiones de grids
            grids = self._detect_multiple_grids(image)
            
            # Intentar identificar patrones de entrada/salida
            puzzle = self._grids_to_puzzle(grids)
            
            return {
                'success': True,
                'puzzle': puzzle,
                'grids_detected': len(grids),
                'analysis': {
                    'image_size': image.size,
                    'estimated_grids': len(grids)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analizando imagen de puzzle: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_multiple_grids(self, image: Image) -> List[Dict[str, Any]]:
        """
        Detecta múltiples grids en una imagen
        
        Args:
            image: Imagen completa
            
        Returns:
            Lista de grids detectados
        """
        # Simplificado: dividir la imagen en regiones
        # En una implementación completa, usaríamos detección de contornos
        
        width, height = image.size
        grids = []
        
        # Intentar patrones comunes de ARC
        # Usualmente hay ejemplos de entrenamiento arriba y test abajo
        
        # Por ahora, simplemente detectar el grid completo
        grid = self._detect_grid(image, 30)
        if grid:
            grids.append({
                'grid': grid,
                'position': (0, 0, width, height),
                'type': 'unknown'
            })
        
        return grids
    
    def _grids_to_puzzle(self, grids: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convierte grids detectados en un puzzle
        
        Args:
            grids: Lista de grids detectados
            
        Returns:
            Puzzle en formato ARC
        """
        puzzle = {
            'id': f'image_puzzle_{np.random.randint(10000)}',
            'category': 'image_imported',
            'difficulty': 'unknown',
            'train': [],
            'test': []
        }
        
        # Por ahora, simplemente usar el primer grid como ejemplo
        if grids:
            puzzle['train'].append({
                'input': grids[0]['grid'],
                'output': grids[0]['grid']  # Placeholder
            })
        
        return puzzle