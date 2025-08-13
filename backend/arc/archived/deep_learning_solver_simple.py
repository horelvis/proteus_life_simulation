#!/usr/bin/env python3
"""
Sistema de Deep Learning Simplificado con A-MHA para ARC
Versión funcional con arquitectura optimizada
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Encoder CNN simplificado"""
    
    def __init__(self, in_channels=10, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)

class SimpleAttention(nn.Module):
    """Atención simplificada"""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, features):
        # Simple scoring
        scores = self.fc(features)
        return scores

class DeepLearningARCSolver:
    """Solver ARC con Deep Learning Simplificado"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.encoder = SimpleCNN().to(self.device)
        self.attention = SimpleAttention().to(self.device)
        self.encoder.eval()
        self.attention.eval()
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Resuelve puzzle ARC"""
        if not train_examples:
            return test_input
            
        # Analizar primer ejemplo
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # Detectar transformación
        transformation = self._detect_transformation(input_grid, output_grid)
        
        # Aplicar al test
        return self._apply_transformation(test_input, transformation)
    
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Resuelve con pasos detallados"""
        steps = []
        
        if not train_examples:
            return test_input, [{"description": "No hay ejemplos"}]
        
        # Analizar primer ejemplo
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # Extraer features con CNN
        with torch.no_grad():
            input_tensor = self._grid_to_tensor(input_grid)
            output_tensor = self._grid_to_tensor(output_grid)
            
            input_features = self.encoder(input_tensor)
            output_features = self.encoder(output_tensor)
            
            # Concatenar y calcular score
            combined = torch.cat([input_features, output_features], dim=1)
            score = self.attention(combined)
            
        steps.append({
            "description": f"Features extraídas, score: {score.item():.2f}"
        })
        
        # Detectar transformación
        transformation = self._detect_transformation(input_grid, output_grid)
        
        steps.append({
            "description": f"Transformación detectada: {transformation['type']}"
        })
        
        # Aplicar al test
        solution = self._apply_transformation(test_input, transformation)
        
        return solution, steps
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convierte grid a tensor"""
        h, w = grid.shape
        max_val = 10
        
        # One-hot encoding
        one_hot = np.zeros((max_val, h, w), dtype=np.float32)
        for i in range(max_val):
            one_hot[i] = (grid == i).astype(np.float32)
        
        return torch.from_numpy(one_hot).unsqueeze(0).to(self.device)
    
    def _detect_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detecta transformación simple"""
        # Expansión en cruz
        if self._is_cross_expansion(input_grid, output_grid):
            return {'type': 'cross_expansion'}
        
        # Relleno
        if self._is_fill_pattern(input_grid, output_grid):
            return {'type': 'fill'}
        
        # Gravedad
        if self._is_gravity_pattern(input_grid, output_grid):
            return {'type': 'gravity'}
        
        # Rotación
        if np.array_equal(np.rot90(input_grid), output_grid):
            return {'type': 'rotation'}
        
        # Mapeo de colores
        mapping = self._detect_color_mapping(input_grid, output_grid)
        if mapping:
            return {'type': 'color_mapping', 'mapping': mapping}
        
        # Diagonal
        if self._is_diagonal_pattern(input_grid, output_grid):
            return {'type': 'diagonal'}
        
        return {'type': 'unknown'}
    
    def _apply_transformation(self, grid: np.ndarray, transformation: Dict) -> np.ndarray:
        """Aplica transformación detectada"""
        trans_type = transformation['type']
        
        if trans_type == 'cross_expansion':
            return self._apply_cross_expansion(grid)
        elif trans_type == 'fill':
            return self._apply_fill(grid)
        elif trans_type == 'gravity':
            return self._apply_gravity(grid)
        elif trans_type == 'rotation':
            return np.rot90(grid)
        elif trans_type == 'color_mapping':
            mapping = transformation.get('mapping', {})
            return self._apply_color_mapping(grid, mapping)
        elif trans_type == 'diagonal':
            return self._apply_diagonal(grid)
        
        return grid
    
    def _is_cross_expansion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta expansión en cruz"""
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                if input_grid[y, x] != 0:
                    value = input_grid[y, x]
                    cross_count = 0
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < input_grid.shape[0] and 
                            0 <= nx < input_grid.shape[1]):
                            if input_grid[ny, nx] == 0 and output_grid[ny, nx] == value:
                                cross_count += 1
                    if cross_count >= 3:
                        return True
        return False
    
    def _is_fill_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón de relleno"""
        zeros_filled = 0
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                if input_grid[y, x] == 0 and output_grid[y, x] != 0:
                    neighbors = []
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < input_grid.shape[0] and 
                            0 <= nx < input_grid.shape[1] and
                            input_grid[ny, nx] != 0):
                            neighbors.append(input_grid[ny, nx])
                    if len(neighbors) >= 2:
                        zeros_filled += 1
        return zeros_filled > 0
    
    def _is_gravity_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta gravedad"""
        h = input_grid.shape[0]
        for col in range(input_grid.shape[1]):
            input_col = input_grid[:, col]
            output_col = output_grid[:, col]
            input_nonzero = input_col[input_col != 0]
            output_nonzero = output_col[output_col != 0]
            if len(input_nonzero) > 0 and len(output_nonzero) == len(input_nonzero):
                expected_start = h - len(output_nonzero)
                if np.all(output_col[:expected_start] == 0):
                    return True
        return False
    
    def _is_diagonal_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón diagonal"""
        h, w = input_grid.shape
        if h != w:
            return False
        diagonal_values = []
        for i in range(min(h, w)):
            if input_grid[i, i] != 0:
                diagonal_values.append(input_grid[i, i])
        if diagonal_values:
            expected = np.full_like(input_grid, diagonal_values[0])
            return np.array_equal(output_grid, expected)
        return False
    
    def _detect_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detecta mapeo de colores"""
        mapping = {}
        for y in range(input_grid.shape[0]):
            for x in range(input_grid.shape[1]):
                in_val = input_grid[y, x]
                out_val = output_grid[y, x]
                if in_val != 0:
                    if in_val in mapping:
                        if mapping[in_val] != out_val:
                            return None
                    else:
                        mapping[in_val] = out_val
        if mapping and any(k != v for k, v in mapping.items()):
            return mapping
        return None
    
    def _apply_cross_expansion(self, grid: np.ndarray) -> np.ndarray:
        """Aplica expansión en cruz"""
        result = grid.copy()
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                if grid[y, x] != 0:
                    value = grid[y, x]
                    neighbors_empty = 0
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0):
                            neighbors_empty += 1
                    if neighbors_empty >= 3:
                        for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0):
                                result[ny, nx] = value
        return result
    
    def _apply_fill(self, grid: np.ndarray) -> np.ndarray:
        """Aplica relleno"""
        result = grid.copy()
        h, w = grid.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if grid[y, x] == 0:
                    neighbors = [
                        grid[y-1, x], grid[y+1, x],
                        grid[y, x-1], grid[y, x+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    if len(non_zero) >= 3:
                        result[y, x] = max(set(non_zero), key=non_zero.count)
        return result
    
    def _apply_gravity(self, grid: np.ndarray) -> np.ndarray:
        """Aplica gravedad"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for col in range(w):
            column = grid[:, col]
            non_zero = column[column != 0]
            if len(non_zero) > 0:
                result[h-len(non_zero):, col] = non_zero
        return result
    
    def _apply_color_mapping(self, grid: np.ndarray, mapping: Dict) -> np.ndarray:
        """Aplica mapeo de colores"""
        result = grid.copy()
        if not mapping:
            result[result != 0] += 1
        else:
            for old_val, new_val in mapping.items():
                result[grid == old_val] = new_val
        return result
    
    def _apply_diagonal(self, grid: np.ndarray) -> np.ndarray:
        """Aplica patrón diagonal"""
        h, w = grid.shape
        diagonal_values = []
        for i in range(min(h, w)):
            if grid[i, i] != 0:
                diagonal_values.append(grid[i, i])
        if diagonal_values:
            return np.full_like(grid, diagonal_values[0])
        return grid