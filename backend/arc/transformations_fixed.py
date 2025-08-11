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
    def detect_and_complete_pattern(input_grid: np.ndarray, output_grid: np.ndarray, 
                                   test_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Intenta detectar el patrón de transformación y aplicarlo
        Retorna None si no puede detectar el patrón
        """
        # Verificar si las dimensiones son compatibles
        if input_grid.shape != output_grid.shape:
            # Podría ser replicación
            if (output_grid.shape[0] % input_grid.shape[0] == 0 and 
                output_grid.shape[1] % input_grid.shape[1] == 0):
                factor = max(output_grid.shape[0] // input_grid.shape[0],
                           output_grid.shape[1] // input_grid.shape[1])
                return RealTransformations.replicate_pattern(test_grid, factor)
            else:
                return None
        
        # Analizar qué cambió entre input y output
        changes = []
        h, w = input_grid.shape
        
        for i in range(h):
            for j in range(w):
                if input_grid[i, j] != output_grid[i, j]:
                    changes.append({
                        'pos': (i, j),
                        'from': input_grid[i, j],
                        'to': output_grid[i, j],
                        'context': RealTransformations._get_context(input_grid, i, j)
                    })
        
        if not changes:
            # No hay cambios, retornar copia
            return test_grid.copy()
        
        # Analizar el patrón de cambios
        pattern_type = RealTransformations._analyze_changes(changes, input_grid, output_grid)
        
        # Aplicar el patrón detectado
        if pattern_type == 'cross_expansion':
            return RealTransformations.expand_cross(test_grid)
        elif pattern_type == 'fill':
            return RealTransformations.fill_enclosed_spaces(test_grid)
        elif pattern_type == 'replication':
            # Detectar factor
            factor = max(output_grid.shape[0] // input_grid.shape[0],
                        output_grid.shape[1] // input_grid.shape[1])
            return RealTransformations.replicate_pattern(test_grid, factor)
        else:
            # No pudimos detectar el patrón
            logger.warning(f"No se pudo detectar el patrón. Tipo: {pattern_type}")
            return None
    
    @staticmethod
    def _get_context(grid: np.ndarray, i: int, j: int) -> Dict[str, Any]:
        """Obtiene el contexto alrededor de una posición"""
        h, w = grid.shape
        context = {
            'neighbors': [],
            'is_edge': i == 0 or i == h-1 or j == 0 or j == w-1,
            'is_corner': (i, j) in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)],
            'is_center': i == h//2 and j == w//2
        }
        
        # Vecinos cardinales
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                context['neighbors'].append(grid[ni, nj])
        
        return context
    
    @staticmethod
    def _analyze_changes(changes: list, input_grid: np.ndarray, 
                        output_grid: np.ndarray) -> str:
        """
        Analiza los cambios para determinar el tipo de patrón
        """
        if not changes:
            return 'identity'
        
        # Ver si todos los cambios son de 0 a algún valor
        all_from_zero = all(c['from'] == 0 for c in changes)
        
        if all_from_zero:
            # Ver si es expansión en cruz
            # Un píxel no-cero expandido crea 4 nuevos píxeles
            non_zero_count = np.count_nonzero(input_grid)
            expected_cross_changes = non_zero_count * 4  # máximo posible
            
            # Verificar si los cambios están en posiciones de cruz
            is_cross = True
            for change in changes:
                i, j = change['pos']
                # Verificar si hay un valor no-cero adyacente en el input
                has_adjacent = False
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < input_grid.shape[0] and 
                        0 <= nj < input_grid.shape[1] and
                        input_grid[ni, nj] == change['to']):
                        has_adjacent = True
                        break
                
                if not has_adjacent:
                    is_cross = False
                    break
            
            if is_cross:
                return 'cross_expansion'
            
            # Ver si es relleno
            # Los cambios deberían estar rodeados por el mismo valor
            is_fill = True
            for change in changes:
                neighbors = change['context']['neighbors']
                if not neighbors or change['to'] not in neighbors:
                    is_fill = False
                    break
            
            if is_fill:
                return 'fill'
        
        # Ver si el output es más grande (replicación)
        if output_grid.shape != input_grid.shape:
            return 'replication'
        
        # Patrón no reconocido
        return 'unknown'
    
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
    
    # Test 3: Detección automática
    print("\n📌 Test 3: Detección Automática de Patrón")
    train_input = np.array([[0,0,0],[0,1,0],[0,0,0]])
    train_output = np.array([[0,1,0],[1,1,1],[0,1,0]])
    test_input = np.array([[0,0,0],[0,2,0],[0,0,0]])
    expected_auto = np.array([[0,2,0],[2,2,2],[0,2,0]])
    
    result_auto = RealTransformations.detect_and_complete_pattern(
        train_input, train_output, test_input
    )
    
    if result_auto is not None:
        print(f"Resultado:\n{result_auto}")
        print(f"Esperado:\n{expected_auto}")
        print(f"✅ CORRECTO" if np.array_equal(result_auto, expected_auto) else "❌ INCORRECTO")
    else:
        print("❌ No pudo detectar el patrón")
    
    return {
        'cross': np.array_equal(result_cross, expected_cross),
        'fill': np.array_equal(result_fill, expected_fill),
        'auto': np.array_equal(result_auto, expected_auto) if result_auto is not None else False
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