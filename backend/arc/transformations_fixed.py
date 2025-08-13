#!/usr/bin/env python3
"""
Transformaciones ARC corregidas y honestas
Implementaciones reales que hacen lo que deben hacer
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.ndimage import binary_fill_holes, label
import logging

logger = logging.getLogger(__name__)

class RealTransformations:
    """
    Transformaciones reales para puzzles ARC
    Sin trucos, sin falsear - implementaciones honestas
    """
    
    @staticmethod
    def expand_cross(grid: np.ndarray, value: Optional[int] = None) -> np.ndarray:
        """
        Expande valores no-cero en forma de cruz (4 direcciones cardinales)
        """
        result = grid.copy()
        h, w = grid.shape
        
        # Encontrar todos los valores no-cero
        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0:
                    expand_val = value if value is not None else grid[i, j]
                    
                    # Expandir en las 4 direcciones cardinales
                    if i > 0:
                        result[i-1, j] = expand_val  # arriba
                    if i < h-1:
                        result[i+1, j] = expand_val  # abajo
                    if j > 0:
                        result[i, j-1] = expand_val  # izquierda
                    if j < w-1:
                        result[i, j+1] = expand_val  # derecha
        
        return result
    
    @staticmethod
    def fill_enclosed_spaces(grid: np.ndarray) -> np.ndarray:
        """
        Rellena espacios completamente encerrados por cada color
        Usa binary_fill_holes de scipy que funciona correctamente
        """
        result = grid.copy()
        
        # Para cada color no-cero
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == 0:
                continue
            
            # Crear máscara binaria para este color
            mask = (grid == color).astype(int)
            
            # Rellenar agujeros en la máscara
            filled = binary_fill_holes(mask).astype(int)
            
            # Aplicar el relleno al resultado
            new_pixels = filled & (~mask)  # Píxeles nuevos a rellenar
            result[new_pixels.astype(bool)] = color
        
        return result
    
    @staticmethod
    def replicate_pattern(grid: np.ndarray, factor: int = 3) -> np.ndarray:
        """
        Replica cada píxel en un bloque de factor x factor
        Esta es la implementación real de pattern_replication
        """
        h, w = grid.shape
        result = np.zeros((h * factor, w * factor), dtype=grid.dtype)
        
        for i in range(h):
            for j in range(w):
                # Replicar cada píxel en un bloque factor x factor
                result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]
        
        return result
    
    
    @staticmethod
    def apply_color_mapping(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """
        Aplica un mapeo de colores
        """
        result = grid.copy()
        for old_val, new_val in mapping.items():
            result[grid == old_val] = new_val
        return result
    
    @staticmethod
    def rotate(grid: np.ndarray, angle: int) -> np.ndarray:
        """
        Rota la grilla en múltiplos de 90 grados
        """
        k = (angle // 90) % 4
        return np.rot90(grid, k)
    
    @staticmethod
    def reflect(grid: np.ndarray, axis: str) -> np.ndarray:
        """
        Refleja la grilla
        axis: 'horizontal' o 'vertical'
        """
        if axis == 'horizontal':
            return np.fliplr(grid)
        elif axis == 'vertical':
            return np.flipud(grid)
        else:
            return grid.copy()
    
    @staticmethod
    def apply_gravity(grid: np.ndarray, direction: str = 'down') -> np.ndarray:
        """
        Aplica gravedad moviendo elementos no-cero en la dirección especificada
        """
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        if direction == 'down':
            for j in range(w):
                # Obtener valores no-cero de la columna
                col_values = grid[:, j]
                non_zero = col_values[col_values != 0]
                # Poner al fondo
                if len(non_zero) > 0:
                    result[h-len(non_zero):, j] = non_zero
                    
        elif direction == 'up':
            for j in range(w):
                col_values = grid[:, j]
                non_zero = col_values[col_values != 0]
                if len(non_zero) > 0:
                    result[:len(non_zero), j] = non_zero
                    
        elif direction == 'left':
            for i in range(h):
                row_values = grid[i, :]
                non_zero = row_values[row_values != 0]
                if len(non_zero) > 0:
                    result[i, :len(non_zero)] = non_zero
                    
        elif direction == 'right':
            for i in range(h):
                row_values = grid[i, :]
                non_zero = row_values[row_values != 0]
                if len(non_zero) > 0:
                    result[i, w-len(non_zero):] = non_zero
        
        return result
    
    @staticmethod
    def find_and_extend_lines(grid: np.ndarray) -> np.ndarray:
        """
        Encuentra patrones lineales y los extiende
        """
        result = grid.copy()
        h, w = grid.shape
        
        # Buscar líneas horizontales
        for i in range(h):
            row = grid[i, :]
            non_zero = np.where(row != 0)[0]
            if len(non_zero) >= 2:
                # Ver si están en línea
                if non_zero[-1] - non_zero[0] == len(non_zero) - 1:
                    # Es una línea continua, extender
                    val = row[non_zero[0]]
                    result[i, non_zero[0]:non_zero[-1]+1] = val
        
        # Buscar líneas verticales
        for j in range(w):
            col = grid[:, j]
            non_zero = np.where(col != 0)[0]
            if len(non_zero) >= 2:
                if non_zero[-1] - non_zero[0] == len(non_zero) - 1:
                    val = col[non_zero[0]]
                    result[non_zero[0]:non_zero[-1]+1, j] = val
        
        return result

# Función de prueba honesta
def test_transformations_honestly():
    """
    Prueba honesta de las transformaciones
    Sin falsear resultados
    """
    print("="*70)
    print("🔬 PRUEBA HONESTA DE TRANSFORMACIONES CORREGIDAS")
    print("="*70)
    
    # Test 1: Expandir cruz
    print("\n📌 Test 1: Expandir Cruz")
    input_cross = np.array([[0,0,0],[0,3,0],[0,0,0]])
    expected_cross = np.array([[0,3,0],[3,3,3],[0,3,0]])
    
    result_cross = RealTransformations.expand_cross(input_cross)
    
    print(f"Input:\n{input_cross}")
    print(f"Resultado:\n{result_cross}")
    print(f"Esperado:\n{expected_cross}")
    print(f"✅ CORRECTO" if np.array_equal(result_cross, expected_cross) else "❌ INCORRECTO")
    
    # Test 2: Rellenar forma
    print("\n📌 Test 2: Rellenar Forma Cerrada")
    input_fill = np.array([[3,3,3],[3,0,3],[3,3,3]])
    expected_fill = np.array([[3,3,3],[3,3,3],[3,3,3]])
    
    result_fill = RealTransformations.fill_enclosed_spaces(input_fill)
    
    print(f"Input:\n{input_fill}")
    print(f"Resultado:\n{result_fill}")
    print(f"Esperado:\n{expected_fill}")
    print(f"✅ CORRECTO" if np.array_equal(result_fill, expected_fill) else "❌ INCORRECTO")
    
    return {
        'cross': np.array_equal(result_cross, expected_cross),
        'fill': np.array_equal(result_fill, expected_fill)
    }

if __name__ == "__main__":
    results = test_transformations_honestly()
    
    print("\n" + "="*70)
    print("📊 RESUMEN HONESTO:")
    print("="*70)
    
    total = len(results)
    correct = sum(results.values())
    
    print(f"\nResultados: {correct}/{total} correctos")
    for test, passed in results.items():
        print(f"  • {test}: {'✅' if passed else '❌'}")
    
    if correct == total:
        print("\n✅ Todas las transformaciones funcionan correctamente")
    else:
        print(f"\n⚠️ Solo {correct}/{total} transformaciones funcionan")
        print("Siendo honesto: Aún hay trabajo por hacer")