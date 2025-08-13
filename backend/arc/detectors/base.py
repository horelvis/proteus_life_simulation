from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class DetectedObject:
    """
    Estructura de datos para un objeto detectado.
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

class ObjectDetectorBase(ABC):
    """
    Clase base abstracta para los detectores de objetos.
    Define la interfaz que todos los detectores deben implementar.
    """

    @abstractmethod
    def detect(self, grid: np.ndarray) -> List[DetectedObject]:
        """
        Toma una cuadrícula ARC y devuelve una lista de objetos detectados.

        Args:
            grid: El array de numpy que representa la cuadrícula ARC.

        Returns:
            Una lista de dataclasses DetectedObject.
        """
        pass
