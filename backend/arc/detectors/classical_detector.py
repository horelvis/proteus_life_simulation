"""
Detector de Objetos Clásico usando Scipy.
"""

import numpy as np
from typing import List
from scipy.ndimage import label, find_objects

# Importar la clase base
from .base import ObjectDetectorBase, DetectedObject

class ClassicalObjectDetector(ObjectDetectorBase):
    """
    Detecta objetos en cuadrículas ARC utilizando etiquetado de componentes conectados.
    Este detector sirve como una base y un fallback si no hay modelos de DL disponibles.
    """
    def detect(self, grid: np.ndarray) -> List[DetectedObject]:
        """
        Detecta objetos como componentes conectados.
        No realiza clasificación de formas, solo las localiza.
        """
        detected_objects = []

        # Etiquetar componentes conectados para cada color por separado
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == 0:  # Ignorar el fondo
                continue

            # Crear una máscara binaria para el color actual
            mask = (grid == color)

            # Encontrar componentes conectados en la máscara
            labeled_array, num_features = label(mask)

            if num_features > 0:
                # Encontrar las cajas delimitadoras (bounding boxes) para cada componente
                slices = find_objects(labeled_array)

                for i, s in enumerate(slices):
                    # El nombre de la etiqueta es genérico, ya que no clasificamos la forma
                    label_name = f"object_color_{color}"

                    # Extraer las coordenadas del bounding box
                    y_slice, x_slice = s
                    x_min, y_min = x_slice.start, y_slice.start
                    x_max, y_max = x_slice.stop, y_slice.stop

                    detected_objects.append(
                        DetectedObject(
                            label=label_name,
                            confidence=1.0,  # La confianza es 1.0 ya que es una detección determinista
                            bbox=(x_min, y_min, x_max, y_max)
                        )
                    )

        return detected_objects
