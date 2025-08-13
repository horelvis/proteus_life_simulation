#!/usr/bin/env python3
"""
Detector de Objetos REAL para ARC
Basado en los principios de YOLO pero adaptado para grids discretos de ARC
NO es un nombre falso - realmente detecta objetos
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.ndimage import label, find_objects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Objeto detectado en el grid"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Máscara binaria del objeto
    color: int  # Color dominante
    confidence: float  # Confianza de detección
    area: int  # Área en píxeles
    center: Tuple[float, float]  # Centro del objeto
    shape_type: str  # Tipo de forma detectada


class ARCObjectDetector:
    """
    Detector de objetos REAL para ARC
    
    Diferencias con YOLO tradicional:
    1. YOLO usa CNN profunda, nosotros usamos análisis conectado
    2. YOLO predice bounding boxes, nosotros detectamos regiones exactas
    3. YOLO trabaja con imágenes RGB, nosotros con grids discretos
    
    Pero el PRINCIPIO es el mismo: detectar y localizar objetos
    """
    
    def __init__(self):
        self.min_object_size = 3  # Mínimo 3 píxeles para considerar objeto (no píxeles sueltos)
        self.background_color = 0  # Negro es fondo por defecto
        self.connectivity = 4  # Conectividad 4-vecinos (no diagonal) para componentes
        logger.info("Detector de objetos inicializado (tipo YOLO para grids discretos)")
    
    def detect(self, grid: np.ndarray) -> List[DetectedObject]:
        """
        Detecta todos los objetos en un grid
        
        ESTO SÍ DETECTA OBJETOS REALMENTE:
        1. Encuentra componentes conectados
        2. Extrae bounding boxes
        3. Calcula propiedades
        4. Clasifica formas
        """
        objects = []
        
        # Primero detectar regiones conectadas multicolor
        all_non_background = (grid != self.background_color).astype(np.uint8)
        
        # Estructura de conectividad (4-vecinos, no diagonal)
        structure = np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]])
        
        # Encontrar todas las regiones conectadas (multicolor)
        labeled_regions, num_regions = label(all_non_background, structure=structure)
        
        # Analizar cada región
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            
            # Saltar si es muy pequeña
            area = np.sum(region_mask)
            if area < self.min_object_size:
                continue
            
            # Obtener colores en esta región
            region_colors = grid[region_mask]
            unique_colors_in_region = np.unique(region_colors)
            
            # Si la región tiene un solo color, es un objeto simple
            if len(unique_colors_in_region) == 1:
                color = unique_colors_in_region[0]
                bbox = self._get_bbox(region_mask)
                y_coords, x_coords = np.where(region_mask)
                center = (np.mean(x_coords), np.mean(y_coords))
                shape_type = self._classify_shape(region_mask, bbox)
                
                detected_obj = DetectedObject(
                    bbox=bbox,
                    mask=region_mask,
                    color=int(color),
                    confidence=1.0,
                    area=int(area),
                    center=center,
                    shape_type=shape_type
                )
                objects.append(detected_obj)
            
            # Si tiene múltiples colores, es un objeto compuesto
            else:
                # Tratar toda la región multicolor como un objeto
                bbox = self._get_bbox(region_mask)
                y_coords, x_coords = np.where(region_mask)
                center = (np.mean(x_coords), np.mean(y_coords))
                
                # Color dominante
                color_counts = np.bincount(region_colors.flatten())
                dominant_color = np.argmax(color_counts)
                
                detected_obj = DetectedObject(
                    bbox=bbox,
                    mask=region_mask,
                    color=int(dominant_color),
                    confidence=0.9,  # Menor confianza para objetos multicolor
                    area=int(area),
                    center=center,
                    shape_type="composite"  # Tipo especial para objetos multicolor
                )
                objects.append(detected_obj)
        
        return objects
    
    def _get_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Obtiene el bounding box de una máscara"""
        y_coords, x_coords = np.where(mask)
        x1, y1 = x_coords.min(), y_coords.min()
        x2, y2 = x_coords.max(), y_coords.max()
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _classify_shape(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Clasifica el tipo de forma del objeto
        Esta es la parte más "inteligente" del detector
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        area = np.sum(mask)
        
        # Extraer región del objeto
        obj_region = mask[y1:y2+1, x1:x2+1]
        
        # Calcular fill ratio
        bbox_area = width * height
        fill_ratio = area / bbox_area if bbox_area > 0 else 0
        
        # Clasificación basada en propiedades
        if width == height and fill_ratio > 0.9:
            return "square"
        elif width == height and fill_ratio < 0.5:
            return "hollow_square"
        elif abs(width - height) <= 1 and fill_ratio > 0.7:
            return "near_square"
        elif width > height * 2:
            return "horizontal_line"
        elif height > width * 2:
            return "vertical_line"
        elif fill_ratio < 0.3:
            return "sparse"
        elif fill_ratio > 0.8:
            return "solid_rect"
        elif self._is_L_shape(obj_region):
            return "L_shape"
        elif self._is_T_shape(obj_region):
            return "T_shape"
        elif self._is_cross(obj_region):
            return "cross"
        else:
            return "irregular"
    
    def _is_L_shape(self, region: np.ndarray) -> bool:
        """Detecta si es forma de L"""
        h, w = region.shape
        if h < 3 or w < 3:
            return False
        
        # Verificar patrones de L
        # L normal
        if np.sum(region[:, 0]) > h * 0.7 and np.sum(region[-1, :]) > w * 0.7:
            return True
        # L rotada
        if np.sum(region[0, :]) > w * 0.7 and np.sum(region[:, 0]) > h * 0.7:
            return True
        # L invertida
        if np.sum(region[:, -1]) > h * 0.7 and np.sum(region[-1, :]) > w * 0.7:
            return True
        # L rotada invertida  
        if np.sum(region[0, :]) > w * 0.7 and np.sum(region[:, -1]) > h * 0.7:
            return True
            
        return False
    
    def _is_T_shape(self, region: np.ndarray) -> bool:
        """Detecta si es forma de T"""
        h, w = region.shape
        if h < 3 or w < 3:
            return False
        
        # T vertical
        mid_col = w // 2
        mid_row = h // 2
        
        # Verificar línea horizontal arriba y línea vertical en medio
        if np.sum(region[0, :]) > w * 0.7 and np.sum(region[:, mid_col]) > h * 0.5:
            return True
        # T horizontal
        if np.sum(region[:, 0]) > h * 0.7 and np.sum(region[mid_row, :]) > w * 0.5:
            return True
            
        return False
    
    def _is_cross(self, region: np.ndarray) -> bool:
        """Detecta si es una cruz"""
        h, w = region.shape
        if h < 3 or w < 3:
            return False
        
        mid_col = w // 2
        mid_row = h // 2
        
        # Verificar líneas vertical y horizontal que se cruzan
        vert_line = np.sum(region[:, mid_col]) > h * 0.6
        horiz_line = np.sum(region[mid_row, :]) > w * 0.6
        
        return vert_line and horiz_line
    
    def visualize_detections(self, grid: np.ndarray, objects: List[DetectedObject]) -> str:
        """
        Visualiza las detecciones en formato texto
        """
        h, w = grid.shape
        output = []
        
        output.append(f"Detectados {len(objects)} objetos:")
        output.append("=" * 50)
        
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj.bbox
            output.append(f"\nObjeto {i+1}:")
            output.append(f"  - Tipo: {obj.shape_type}")
            output.append(f"  - Color: {obj.color}")
            output.append(f"  - BBox: ({x1},{y1}) -> ({x2},{y2})")
            output.append(f"  - Área: {obj.area} píxeles")
            output.append(f"  - Centro: ({obj.center[0]:.1f}, {obj.center[1]:.1f})")
        
        return "\n".join(output)
    
    def get_spatial_relations(self, objects: List[DetectedObject]) -> Dict:
        """
        Analiza relaciones espaciales entre objetos detectados
        Esto es lo que YOLO no hace pero nosotros sí necesitamos para ARC
        """
        relations = {
            'alignments': [],
            'containments': [],
            'adjacencies': [],
            'symmetries': []
        }
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Verificar alineación
                if abs(obj1.center[0] - obj2.center[0]) < 1:
                    relations['alignments'].append((i, j, 'vertical'))
                if abs(obj1.center[1] - obj2.center[1]) < 1:
                    relations['alignments'].append((i, j, 'horizontal'))
                
                # Verificar contención
                x1_1, y1_1, x2_1, y2_1 = obj1.bbox
                x1_2, y1_2, x2_2, y2_2 = obj2.bbox
                
                if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
                    relations['containments'].append((i, j))
                elif x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1:
                    relations['containments'].append((j, i))
                
                # Verificar adyacencia
                if self._are_adjacent(obj1.bbox, obj2.bbox):
                    relations['adjacencies'].append((i, j))
        
        return relations
    
    def _are_adjacent(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Verifica si dos bounding boxes son adyacentes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Adyacente horizontalmente
        if (x2_1 + 1 == x1_2 or x2_2 + 1 == x1_1) and \
           not (y2_1 < y1_2 or y2_2 < y1_1):
            return True
        
        # Adyacente verticalmente
        if (y2_1 + 1 == y1_2 or y2_2 + 1 == y1_1) and \
           not (x2_1 < x1_2 or x2_2 < x1_1):
            return True
        
        return False


class NeuralObjectDetector(nn.Module):
    """
    Versión con red neuronal para aprender a detectar objetos
    Más cercano al YOLO original pero adaptado para ARC
    """
    
    def __init__(self, grid_size: int = 30, num_anchors: int = 9):
        super().__init__()
        
        # Backbone CNN para extraer features
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Detection head (tipo YOLO)
        # Para cada celda predice: 
        # - objectness (hay objeto?)
        # - bbox (4 valores: x, y, w, h)
        # - class (qué tipo de forma)
        self.detection = nn.Conv2d(128, num_anchors * (5 + 10), kernel_size=1)
        # 5 = objectness + 4 bbox coords
        # 10 = número de clases de formas
        
        self.num_anchors = num_anchors
        
    def forward(self, x):
        """
        Input: grid de shape [batch, 1, H, W]
        Output: detecciones de shape [batch, grid_h, grid_w, anchors, 5+classes]
        """
        features = self.features(x)
        detections = self.detection(features)
        
        # Reshape para formato YOLO
        batch, _, h, w = detections.shape
        detections = detections.view(batch, self.num_anchors, -1, h, w)
        detections = detections.permute(0, 3, 4, 1, 2)  # [batch, h, w, anchors, predictions]
        
        return detections


def test_detector():
    """Prueba el detector con un grid de ejemplo"""
    
    # Crear grid de prueba con varios objetos
    grid = np.zeros((20, 20), dtype=int)
    
    # Objeto 1: Cuadrado sólido
    grid[2:5, 2:5] = 1
    
    # Objeto 2: L shape
    grid[7:12, 7] = 2
    grid[11, 7:12] = 2
    
    # Objeto 3: Línea horizontal
    grid[15, 5:15] = 3
    
    # Objeto 4: Cruz
    grid[2:8, 15] = 4
    grid[5, 13:18] = 4
    
    print("Grid de prueba:")
    print(grid)
    print("\n" + "="*50)
    
    # Detectar objetos
    detector = ARCObjectDetector()
    objects = detector.detect(grid)
    
    # Mostrar resultados
    print(detector.visualize_detections(grid, objects))
    
    # Analizar relaciones
    print("\n" + "="*50)
    print("Relaciones espaciales:")
    relations = detector.get_spatial_relations(objects)
    
    for rel_type, rel_list in relations.items():
        if rel_list:
            print(f"\n{rel_type.upper()}:")
            for rel in rel_list:
                print(f"  - {rel}")
    
    return objects


if __name__ == "__main__":
    print("DETECTOR DE OBJETOS REAL PARA ARC")
    print("="*50)
    print("Esto SÍ detecta objetos, no es un nombre falso")
    print("Principios de YOLO adaptados a grids discretos")
    print("="*50 + "\n")
    
    objects = test_detector()
    
    print("\n" + "="*50)
    print("DIFERENCIA CON YOLO TRADICIONAL:")
    print("- YOLO: CNN profunda -> Bounding boxes en imágenes")
    print("- Nuestro: Análisis conectado -> Objetos exactos en grids")
    print("- Pero ambos DETECTAN Y LOCALIZAN objetos realmente")