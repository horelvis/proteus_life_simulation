"""
Detector de Objetos usando un modelo YOLO pre-entrenado.
"""

import numpy as np
from typing import List
import os
from PIL import Image

# Importar la clase base
from .base import ObjectDetectorBase, DetectedObject

# --- NOTA DE IMPLEMENTACIÓN ---
# Este detector requiere la biblioteca 'ultralytics'.
# Asegúrate de que esté instalada en tu entorno (`pip install ultralytics`).
# El código está diseñado para cargar el modelo, pero no se puede ejecutar
# en este entorno limitado por falta de espacio para instalar la biblioteca.
try:
    from ultralytics import YOLO
except ImportError:
    print("ADVERTENCIA: La biblioteca 'ultralytics' no está instalada. YOLOObjectDetector no funcionará.")
    YOLO = None

# Paleta de colores para convertir la cuadrícula ARC a una imagen RGB
# Los colores se eligen para ser visualmente distintos.
ARC_COLOR_MAP = {
    0: (0, 0, 0),       # Negro
    1: (0, 0, 255),     # Azul
    2: (255, 0, 0),     # Rojo
    3: (0, 255, 0),     # Verde
    4: (255, 255, 0),   # Amarillo
    5: (128, 0, 128),   # Púrpura
    6: (255, 165, 0),   # Naranja
    7: (0, 255, 255),   # Cian
    8: (255, 0, 255),   # Magenta
    9: (211, 211, 211), # Gris claro
}

class YOLOObjectDetector(ObjectDetectorBase):
    """
    Detecta objetos en cuadrículas ARC utilizando un modelo YOLO.
    """
    def __init__(self, model_path: str = "backend/arc/yolo_weights/yolo_shapes.pth"):
        if YOLO is None:
            raise ImportError("La biblioteca 'ultralytics' es necesaria para usar YOLOObjectDetector.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo de pesos del modelo YOLO no se encontró en: {model_path}")

        # Cargar el modelo YOLO pre-entrenado.
        self.model = YOLO(model_path)
        # Suponemos que el modelo fue entrenado en clases como 'cuadrado', 'círculo', etc.
        # Los nombres de las clases se obtienen del propio modelo.
        self.class_names = self.model.names

    def _grid_to_image(self, grid: np.ndarray) -> Image.Image:
        """
        Convierte una cuadrícula ARC en una imagen RGB de PIL.
        """
        # Escalar la imagen para que sea más fácil de procesar por el modelo
        scale = 20
        height, width = grid.shape
        image_array = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

        for r in range(height):
            for c in range(width):
                color_index = grid[r, c]
                color_rgb = ARC_COLOR_MAP.get(color_index, (255, 255, 255)) # Blanco por defecto

                # Rellenar el bloque escalado
                image_array[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color_rgb

        return Image.fromarray(image_array)

    def detect(self, grid: np.ndarray) -> List[DetectedObject]:
        """
        Detecta objetos en la cuadrícula usando el modelo YOLO.
        """
        # 1. Convertir la cuadrícula a una imagen
        image = self._grid_to_image(grid)

        # 2. Ejecutar la predicción del modelo
        # El modelo de ultralytics puede tomar una imagen de PIL directamente.
        results = self.model(image, verbose=False) # verbose=False para una salida limpia

        detected_objects = []

        # 3. Procesar los resultados
        # results es una lista, tomamos el primer elemento para la única imagen que enviamos
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                # Obtener la clase y la confianza
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Obtener la etiqueta del nombre de la clase
                label = self.class_names.get(class_id, "desconocido")

                # Obtener las coordenadas del bounding box y des-escalarlas
                # xyxy es un tensor [x_min, y_min, x_max, y_max]
                scale = 20 # El mismo factor de escala usado en _grid_to_image
                x_min, y_min, x_max, y_max = [int(coord / scale) for coord in box.xyxy[0]]

                detected_objects.append(
                    DetectedObject(
                        label=label,
                        confidence=confidence,
                        bbox=(x_min, y_min, x_max, y_max)
                    )
                )

        return detected_objects
