#!/usr/bin/env python3
"""
Sistema de Atención Jerárquica Multiescala con soporte GPU (HAMS-GPU)
Versión optimizada de HAMS que usa PyTorch para acelerar cálculos en GPU
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

# Importar el solver HAMS original
from arc.hierarchical_attention_solver import HierarchicalAttentionSolver, HierarchicalPixel

logger = logging.getLogger(__name__)

class HierarchicalAttentionSolverGPU(HierarchicalAttentionSolver):
    """
    HAMS-GPU: Versión acelerada por GPU del sistema HAMS
    
    Usa PyTorch para operaciones matriciales pesadas mientras mantiene
    la misma lógica de análisis jerárquico.
    """
    
    def __init__(self, device=None):
        super().__init__()
        
        # Configurar dispositivo
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"🚀 HAMS-GPU inicializado en {self.device}")
        
        # Pre-allocate some tensors for efficiency
        self.kernel_cache = {}
        
    def _compute_local_attention(self, grid: np.ndarray):
        """
        Calcula atención local usando convoluciones GPU
        """
        h, w = grid.shape
        
        # Convertir a tensor y mover a GPU
        grid_tensor = torch.from_numpy(grid).float().to(self.device)
        grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Kernels para detectar patrones locales
        if 'local_kernels' not in self.kernel_cache:
            # Crear kernels una sola vez
            kernels = []
            
            # Kernel de detección de bordes
            edge_kernel = torch.tensor([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], dtype=torch.float32).to(self.device)
            kernels.append(edge_kernel)
            
            # Kernel de detección de esquinas
            corner_kernels = [
                torch.tensor([[2, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=torch.float32),
                torch.tensor([[-1, -1, 2], [-1, 0, -1], [0, -1, -1]], dtype=torch.float32),
                torch.tensor([[-1, -1, 0], [-1, 0, -1], [2, -1, -1]], dtype=torch.float32),
                torch.tensor([[0, -1, -1], [-1, 0, -1], [-1, -1, 2]], dtype=torch.float32),
            ]
            for k in corner_kernels:
                kernels.append(k.to(self.device))
            
            # Stack kernels
            self.kernel_cache['local_kernels'] = torch.stack(kernels).unsqueeze(1)
        
        kernels = self.kernel_cache['local_kernels']
        
        # Aplicar convoluciones en paralelo
        with torch.no_grad():
            responses = F.conv2d(grid_tensor, kernels, padding=1)
            
            # Calcular atención combinada
            attention = torch.abs(responses).mean(dim=1, keepdim=True)
            attention = F.softmax(attention.view(1, -1), dim=-1).view(1, 1, h, w)
            
        # Actualizar pixel map con scores de atención
        attention_np = attention.squeeze().cpu().numpy()
        
        for y in range(h):
            for x in range(w):
                if (x, y) in self.pixel_map:
                    self.pixel_map[(x, y)].local_attention = float(attention_np[y, x])
                    
    def _compute_coherence(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[float]:
        """
        Calcula coherencia usando operaciones tensoriales en GPU
        """
        # Convertir a tensores
        input_tensor = torch.from_numpy(input_grid).float().to(self.device)
        output_tensor = torch.from_numpy(output_grid).float().to(self.device)
        
        h, w = input_grid.shape
        scores = {}
        
        with torch.no_grad():
            # Coherencia de transformación (qué tan consistente es el cambio)
            diff = output_tensor - input_tensor
            change_mask = (diff != 0).float()
            
            # Calcular consistencia local usando convolución
            kernel = torch.ones(1, 1, 3, 3).to(self.device) / 9.0
            input_padded = input_tensor.unsqueeze(0).unsqueeze(0)
            output_padded = output_tensor.unsqueeze(0).unsqueeze(0)
            
            local_mean_input = F.conv2d(input_padded, kernel, padding=1)
            local_mean_output = F.conv2d(output_padded, kernel, padding=1)
            
            # Coherencia = qué tan similar es el cambio local
            local_coherence = 1.0 - torch.abs(local_mean_output - local_mean_input) / 10.0
            local_coherence = torch.clamp(local_coherence, 0, 1)
            
            scores['transformation'] = local_coherence.squeeze().cpu().numpy()
            
            # Coherencia espacial (qué tan agrupados están los cambios)
            if change_mask.sum() > 0:
                # Encontrar componentes conectados de cambios
                change_np = change_mask.cpu().numpy()
                from scipy.ndimage import label
                labeled, num_components = label(change_np)
                
                # Menos componentes = más coherente
                spatial_coherence = 1.0 / (1.0 + num_components)
                scores['spatial'] = spatial_coherence
            else:
                scores['spatial'] = 1.0
                
            # Coherencia de color (cambios consistentes de color)
            if change_mask.sum() > 0:
                unique_changes = torch.unique(diff[change_mask.bool()])
                color_coherence = 1.0 / (1.0 + len(unique_changes))
                scores['color'] = float(color_coherence)
            else:
                scores['color'] = 1.0
            
        # Convertir scores dict a lista de coherencias para compatibilidad
        coherence_list = []
        if 'transformation' in scores:
            transform_coherence = scores['transformation']
            if isinstance(transform_coherence, np.ndarray):
                coherence_list.extend(transform_coherence.flatten().tolist())
        
        # Agregar coherencias espacial y de color como valores únicos
        if 'spatial' in scores:
            coherence_list.append(scores['spatial'])
        if 'color' in scores:
            coherence_list.append(scores['color'])
            
        return coherence_list if coherence_list else [1.0]
        
    def _detect_objects_gpu(self, grid: np.ndarray) -> torch.Tensor:
        """
        Detecta objetos usando operaciones paralelas en GPU
        """
        grid_tensor = torch.from_numpy(grid).to(self.device)
        h, w = grid.shape
        
        # Inicializar etiquetas
        labels = torch.zeros_like(grid_tensor, dtype=torch.int32)
        current_label = 1
        
        # Máscara de píxeles no procesados
        unprocessed = grid_tensor > 0
        
        with torch.no_grad():
            while unprocessed.any():
                # Encontrar siguiente píxel no procesado
                y, x = torch.where(unprocessed)
                if len(y) == 0:
                    break
                    
                seed_y, seed_x = int(y[0]), int(x[0])
                seed_color = grid_tensor[seed_y, seed_x]
                
                # Flood fill paralelo usando dilatación
                mask = torch.zeros_like(grid_tensor, dtype=torch.bool)
                mask[seed_y, seed_x] = True
                
                # Dilatar iterativamente
                for _ in range(max(h, w)):
                    old_mask = mask.clone()
                    
                    # Dilatar en 4 direcciones
                    if seed_y > 0:
                        mask[:-1, :] |= old_mask[1:, :] & (grid_tensor[:-1, :] == seed_color)
                    if seed_y < h-1:
                        mask[1:, :] |= old_mask[:-1, :] & (grid_tensor[1:, :] == seed_color)
                    if seed_x > 0:
                        mask[:, :-1] |= old_mask[:, 1:] & (grid_tensor[:, :-1] == seed_color)
                    if seed_x < w-1:
                        mask[:, 1:] |= old_mask[:, :-1] & (grid_tensor[:, 1:] == seed_color)
                    
                    if (mask == old_mask).all():
                        break
                
                # Asignar etiqueta
                labels[mask] = current_label
                unprocessed &= ~mask
                current_label += 1
                
        return labels
        
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve un puzzle usando análisis jerárquico con GPU
        """
        # Usar implementación base pero con operaciones GPU donde sea posible
        steps = []
        
        # Log uso de GPU
        steps.append({
            "description": f"🚀 Usando HAMS-GPU en {self.device}"
        })
        
        # Continuar con el solver padre
        return super().solve_with_steps(train_examples, test_input)