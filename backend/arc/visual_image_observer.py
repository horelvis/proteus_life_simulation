#!/usr/bin/env python3
"""
Observador Visual de Imágenes Reales
Procesa imágenes PNG/JPG en lugar de matrices numéricas
Usa visión por computadora para entender la escena
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImageObservation:
    """Observación de una imagen real"""
    image_path: str
    detected_objects: List[Dict]
    detected_colors: Dict[str, float]
    detected_shapes: List[str]
    detected_patterns: List[str]
    spatial_relationships: List[Dict]
    scene_description: str
    confidence: float

class VisualImageObserver:
    """
    Observador que procesa imágenes reales (PNG, JPG, etc)
    Usa CNN preentrenadas y técnicas de visión por computadora
    """
    
    def __init__(self, use_pretrained: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖼️ Visual Image Observer inicializado en {self.device}")
        
        # Transformaciones para imágenes
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Red neuronal para extracción de características
        if use_pretrained:
            # Usar ResNet preentrenada
            self.feature_extractor = models.resnet50(pretrained=True)
            self.feature_extractor.eval()
            self.feature_extractor = self.feature_extractor.to(self.device)
            
            # Remover la última capa para obtener features
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )
        
        # Red de atención visual personalizada
        self.attention_network = self._build_visual_attention()
        
    def _build_visual_attention(self) -> nn.Module:
        """Construye red de atención visual para imágenes"""
        class ImageAttention(nn.Module):
            def __init__(self):
                super().__init__()
                # Procesamiento de características visuales
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                
                # Atención espacial
                self.spatial_attention = nn.Conv2d(256, 1, kernel_size=1)
                
                # Atención por canal
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                # Extraer características
                f1 = torch.relu(self.conv1(x))
                f2 = torch.relu(self.conv2(f1))
                f3 = torch.relu(self.conv3(f2))
                
                # Aplicar atención espacial
                spatial_att = torch.sigmoid(self.spatial_attention(f3))
                f3_spatial = f3 * spatial_att
                
                # Aplicar atención por canal
                channel_att = self.channel_attention(f3)
                f3_final = f3_spatial * channel_att
                
                return f3_final, spatial_att
        
        return ImageAttention().to(self.device)
    
    def observe_image(self, image_path: Union[str, Path], 
                     visualize: bool = True) -> ImageObservation:
        """
        Observa y analiza una imagen completa
        """
        image_path = Path(image_path)
        logger.info(f"📸 Observando imagen: {image_path}")
        
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Análisis de bajo nivel (OpenCV)
        colors = self._analyze_colors(image_np)
        shapes = self._detect_shapes(image_np)
        edges = self._detect_edges(image_np)
        patterns = self._detect_patterns(image_np)
        
        # Análisis de alto nivel (Deep Learning)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extraer características profundas
            if hasattr(self, 'feature_extractor'):
                deep_features = self.feature_extractor(image_tensor)
            
            # Aplicar atención visual
            attended_features, attention_map = self.attention_network(image_tensor)
        
        # Detectar objetos y relaciones
        objects = self._detect_objects(image_np, attention_map)
        relationships = self._analyze_spatial_relationships(objects)
        
        # Generar descripción de la escena
        scene_desc = self._generate_scene_description(
            colors, shapes, patterns, objects, relationships
        )
        
        # Calcular confianza
        confidence = self._calculate_confidence(
            len(objects), len(shapes), len(patterns)
        )
        
        # Visualizar si está habilitado
        if visualize:
            self._visualize_analysis(
                image_np, attention_map, objects, 
                colors, shapes, image_path
            )
        
        return ImageObservation(
            image_path=str(image_path),
            detected_objects=objects,
            detected_colors=colors,
            detected_shapes=shapes,
            detected_patterns=patterns,
            spatial_relationships=relationships,
            scene_description=scene_desc,
            confidence=confidence
        )
    
    def compare_images(self, image1_path: Union[str, Path], 
                      image2_path: Union[str, Path],
                      visualize: bool = True) -> Dict:
        """
        Compara dos imágenes y encuentra transformaciones
        Similar a comparar input/output en ARC
        """
        logger.info("🔄 Comparando dos imágenes")
        
        # Observar ambas imágenes
        obs1 = self.observe_image(image1_path, visualize=False)
        obs2 = self.observe_image(image2_path, visualize=False)
        
        # Encontrar diferencias
        color_changes = self._compare_colors(obs1.detected_colors, obs2.detected_colors)
        shape_changes = self._compare_shapes(obs1.detected_shapes, obs2.detected_shapes)
        object_changes = self._compare_objects(obs1.detected_objects, obs2.detected_objects)
        
        # Inferir transformación
        transformation = self._infer_transformation(
            color_changes, shape_changes, object_changes
        )
        
        # Generar razonamiento
        reasoning = self._generate_comparison_reasoning(
            obs1, obs2, transformation
        )
        
        if visualize:
            self._visualize_comparison(image1_path, image2_path, transformation)
        
        return {
            'observation1': obs1,
            'observation2': obs2,
            'transformation': transformation,
            'reasoning': reasoning,
            'confidence': (obs1.confidence + obs2.confidence) / 2
        }
    
    def _analyze_colors(self, image_np: np.ndarray) -> Dict[str, float]:
        """Analiza distribución de colores en la imagen"""
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Definir rangos de colores básicos
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'white': [(0, 0, 200), (180, 30, 255)]
        }
        
        color_percentages = {}
        total_pixels = image_np.shape[0] * image_np.shape[1]
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = (np.sum(mask > 0) / total_pixels) * 100
            if percentage > 1:  # Solo colores con más del 1%
                color_percentages[color_name] = round(percentage, 2)
        
        return color_percentages
    
    def _detect_shapes(self, image_np: np.ndarray) -> List[str]:
        """Detecta formas geométricas en la imagen"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Clasificar por número de vértices
            vertices = len(approx)
            if vertices == 3:
                shapes.append("triangle")
            elif vertices == 4:
                # Verificar si es cuadrado o rectángulo
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shapes.append("square")
                else:
                    shapes.append("rectangle")
            elif vertices == 5:
                shapes.append("pentagon")
            elif vertices > 5:
                # Verificar si es círculo
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:
                        shapes.append("circle")
                    else:
                        shapes.append(f"polygon_{vertices}")
        
        return list(set(shapes))  # Eliminar duplicados
    
    def _detect_edges(self, image_np: np.ndarray) -> np.ndarray:
        """Detecta bordes en la imagen"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def _detect_patterns(self, image_np: np.ndarray) -> List[str]:
        """Detecta patrones visuales en la imagen"""
        patterns = []
        
        # Detectar simetría horizontal
        flipped_h = cv2.flip(image_np, 1)
        if np.mean(np.abs(image_np - flipped_h)) < 10:
            patterns.append("horizontal_symmetry")
        
        # Detectar simetría vertical
        flipped_v = cv2.flip(image_np, 0)
        if np.mean(np.abs(image_np - flipped_v)) < 10:
            patterns.append("vertical_symmetry")
        
        # Detectar repetición (usando FFT)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Buscar picos en el espectro de frecuencia
        peaks = np.where(magnitude > np.mean(magnitude) + 3 * np.std(magnitude))
        if len(peaks[0]) > 10:
            patterns.append("repetitive_pattern")
        
        # Detectar gradientes
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        if np.mean(np.abs(gradient_x)) > np.mean(np.abs(gradient_y)) * 1.5:
            patterns.append("vertical_lines")
        elif np.mean(np.abs(gradient_y)) > np.mean(np.abs(gradient_x)) * 1.5:
            patterns.append("horizontal_lines")
        
        return patterns
    
    def _detect_objects(self, image_np: np.ndarray, 
                       attention_map: torch.Tensor) -> List[Dict]:
        """Detecta objetos usando el mapa de atención"""
        objects = []
        
        # Convertir mapa de atención a numpy
        att_map = attention_map.squeeze().cpu().numpy()
        
        # Redimensionar al tamaño de la imagen
        att_map_resized = cv2.resize(att_map, (image_np.shape[1], image_np.shape[0]))
        
        # Umbralizar para encontrar regiones de alta atención
        threshold = np.mean(att_map_resized) + np.std(att_map_resized)
        binary_map = (att_map_resized > threshold).astype(np.uint8) * 255
        
        # Encontrar contornos de objetos
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:  # Filtrar objetos muy pequeños
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extraer región del objeto
                object_region = image_np[y:y+h, x:x+w]
                
                # Analizar color dominante
                avg_color = np.mean(object_region.reshape(-1, 3), axis=0)
                
                objects.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour),
                    'center': (x + w//2, y + h//2),
                    'avg_color': avg_color.tolist(),
                    'attention_score': float(np.mean(att_map_resized[y:y+h, x:x+w]))
                })
        
        return objects
    
    def _analyze_spatial_relationships(self, objects: List[Dict]) -> List[Dict]:
        """Analiza relaciones espaciales entre objetos"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calcular relación espacial
                c1 = obj1['center']
                c2 = obj2['center']
                
                # Dirección relativa
                if abs(c1[0] - c2[0]) < 20:
                    direction = "vertical"
                elif abs(c1[1] - c2[1]) < 20:
                    direction = "horizontal"
                else:
                    direction = "diagonal"
                
                # Distancia
                distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                
                # Verificar si se tocan
                bbox1 = obj1['bbox']
                bbox2 = obj2['bbox']
                touching = self._check_touching(bbox1, bbox2)
                
                relationships.append({
                    'object1_id': obj1['id'],
                    'object2_id': obj2['id'],
                    'direction': direction,
                    'distance': float(distance),
                    'touching': touching
                })
        
        return relationships
    
    def _check_touching(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Verifica si dos bounding boxes se tocan"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Verificar solapamiento
        return not (x1 + w1 < x2 or x2 + w2 < x1 or 
                   y1 + h1 < y2 or y2 + h2 < y1)
    
    def _generate_scene_description(self, colors: Dict, shapes: List,
                                   patterns: List, objects: List,
                                   relationships: List) -> str:
        """Genera descripción textual de la escena"""
        desc = "Observo una escena con "
        
        # Describir colores dominantes
        if colors:
            main_colors = list(colors.keys())[:3]
            desc += f"colores predominantes: {', '.join(main_colors)}. "
        
        # Describir formas
        if shapes:
            desc += f"Detecto formas: {', '.join(shapes[:3])}. "
        
        # Describir objetos
        if objects:
            desc += f"Identifico {len(objects)} objetos principales. "
        
        # Describir patrones
        if patterns:
            desc += f"Patrones visuales: {', '.join(patterns)}. "
        
        # Describir relaciones
        if relationships:
            desc += f"Los objetos tienen {len(relationships)} relaciones espaciales."
        
        return desc
    
    def _calculate_confidence(self, num_objects: int, 
                            num_shapes: int, 
                            num_patterns: int) -> float:
        """Calcula confianza del análisis"""
        # Más elementos detectados = mayor confianza
        base_confidence = 0.5
        
        if num_objects > 0:
            base_confidence += 0.2
        if num_shapes > 0:
            base_confidence += 0.15
        if num_patterns > 0:
            base_confidence += 0.15
        
        return min(base_confidence, 1.0)
    
    def _visualize_analysis(self, image_np: np.ndarray, 
                           attention_map: torch.Tensor,
                           objects: List[Dict],
                           colors: Dict, shapes: List,
                           image_path: Path):
        """Visualiza el análisis de la imagen"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Imagen original
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # 2. Mapa de atención
        att_map = attention_map.squeeze().cpu().numpy()
        att_map_resized = cv2.resize(att_map, (image_np.shape[1], image_np.shape[0]))
        axes[0, 1].imshow(att_map_resized, cmap='hot')
        axes[0, 1].set_title('Mapa de Atención')
        axes[0, 1].axis('off')
        
        # 3. Objetos detectados
        img_objects = image_np.copy()
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(img_objects, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_objects, f"ID:{obj['id']}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        axes[0, 2].imshow(img_objects)
        axes[0, 2].set_title(f'Objetos Detectados ({len(objects)})')
        axes[0, 2].axis('off')
        
        # 4. Bordes detectados
        edges = self._detect_edges(image_np)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Bordes Detectados')
        axes[1, 0].axis('off')
        
        # 5. Distribución de colores
        if colors:
            axes[1, 1].bar(colors.keys(), colors.values())
            axes[1, 1].set_title('Distribución de Colores (%)')
            axes[1, 1].set_xlabel('Color')
            axes[1, 1].set_ylabel('Porcentaje')
        else:
            axes[1, 1].axis('off')
        
        # 6. Información textual
        axes[1, 2].axis('off')
        info_text = f"Formas: {', '.join(shapes[:5])}\n"
        info_text += f"Objetos: {len(objects)}\n"
        info_text += f"Colores principales: {list(colors.keys())[:3]}\n"
        axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, 
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Análisis')
        
        plt.suptitle(f'Análisis Visual: {image_path.name}', fontsize=14)
        plt.tight_layout()
        
        output_path = f'/tmp/image_analysis_{image_path.stem}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📸 Análisis visual guardado en: {output_path}")
    
    def _compare_colors(self, colors1: Dict, colors2: Dict) -> Dict:
        """Compara distribución de colores entre dos imágenes"""
        all_colors = set(colors1.keys()) | set(colors2.keys())
        
        changes = {}
        for color in all_colors:
            val1 = colors1.get(color, 0)
            val2 = colors2.get(color, 0)
            if abs(val1 - val2) > 5:  # Cambio significativo > 5%
                changes[color] = val2 - val1
        
        return changes
    
    def _compare_shapes(self, shapes1: List, shapes2: List) -> Dict:
        """Compara formas entre dos imágenes"""
        return {
            'added': list(set(shapes2) - set(shapes1)),
            'removed': list(set(shapes1) - set(shapes2)),
            'common': list(set(shapes1) & set(shapes2))
        }
    
    def _compare_objects(self, objects1: List, objects2: List) -> Dict:
        """Compara objetos entre dos imágenes"""
        return {
            'count_change': len(objects2) - len(objects1),
            'image1_objects': len(objects1),
            'image2_objects': len(objects2)
        }
    
    def _infer_transformation(self, color_changes: Dict, 
                            shape_changes: Dict,
                            object_changes: Dict) -> str:
        """Infiere el tipo de transformación entre imágenes"""
        transformations = []
        
        if color_changes:
            transformations.append("color_change")
        
        if shape_changes['added']:
            transformations.append("shape_addition")
        if shape_changes['removed']:
            transformations.append("shape_removal")
        
        if object_changes['count_change'] > 0:
            transformations.append("object_increase")
        elif object_changes['count_change'] < 0:
            transformations.append("object_decrease")
        
        if not transformations:
            return "no_change"
        
        return "_".join(transformations)
    
    def _generate_comparison_reasoning(self, obs1: ImageObservation,
                                      obs2: ImageObservation,
                                      transformation: str) -> List[str]:
        """Genera razonamiento sobre la comparación"""
        reasoning = []
        
        reasoning.append(f"Observo imagen 1: {obs1.scene_description}")
        reasoning.append(f"Observo imagen 2: {obs2.scene_description}")
        reasoning.append(f"Detecto transformación tipo: {transformation}")
        
        if obs1.detected_colors != obs2.detected_colors:
            reasoning.append("Los colores han cambiado entre las imágenes")
        
        if len(obs1.detected_objects) != len(obs2.detected_objects):
            reasoning.append(f"El número de objetos cambió de {len(obs1.detected_objects)} a {len(obs2.detected_objects)}")
        
        return reasoning
    
    def _visualize_comparison(self, image1_path: Union[str, Path],
                            image2_path: Union[str, Path],
                            transformation: str):
        """Visualiza comparación entre dos imágenes"""
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img1)
        axes[0].set_title('Imagen 1 (Input)')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title('Imagen 2 (Output)')
        axes[1].axis('off')
        
        # Panel de transformación
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, f"Transformación:\n{transformation.replace('_', ' ').title()}",
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.suptitle('Comparación de Imágenes', fontsize=16)
        plt.tight_layout()
        
        output_path = '/tmp/image_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📸 Comparación guardada en: {output_path}")