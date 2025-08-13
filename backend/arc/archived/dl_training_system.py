#!/usr/bin/env python3
"""
Sistema de Entrenamiento para Deep Learning ARC Solver
Incluye pérdidas supervisadas, auxiliares y currículo de entrenamiento
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 500
    
    # Pesos de pérdidas
    policy_weight: float = 1.0
    value_weight: float = 0.5
    compatibility_weight: float = 0.3
    legality_weight: float = 0.2
    
    # Currículo
    curriculum_enabled: bool = True
    initial_difficulty: float = 0.2
    difficulty_increment: float = 0.1
    
    # Augmentación
    augment_flips: bool = True
    augment_rotations: bool = True
    augment_noise: float = 0.05

class ARCLoss(nn.Module):
    """Pérdidas combinadas para entrenamiento ARC"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Pérdidas individuales
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Calcula todas las pérdidas
        
        Args:
            predictions: Dict con 'logits', 'value', 'compatibility'
            targets: Dict con 'action_gt', 'value_gt', 'legal_mask'
        
        Returns:
            Dict con pérdidas individuales y total
        """
        losses = {}
        
        # 1. Policy Loss (supervisada)
        if 'logits' in predictions and 'action_gt' in targets:
            logits = predictions['logits']  # [B, N_actions]
            action_gt = targets['action_gt']  # [B]
            
            # Reshape logits si es necesario
            if len(logits.shape) == 3:  # [B, N_pieces, N_slots]
                B, N_p, N_s = logits.shape
                logits = logits.view(B, N_p * N_s)
            
            losses['policy'] = self.ce_loss(logits, action_gt)
        
        # 2. Value Loss (distancia a solución)
        if 'value' in predictions and 'value_gt' in targets:
            value_pred = predictions['value']  # [B, 1]
            value_gt = targets['value_gt']  # [B, 1]
            losses['value'] = self.mse_loss(value_pred, value_gt)
        
        # 3. Compatibility Loss (bordes compatibles)
        if 'compatibility' in predictions and 'compatibility_gt' in targets:
            compat_pred = predictions['compatibility']  # [B, N_pairs]
            compat_gt = targets['compatibility_gt']  # [B, N_pairs]
            # Pérdida contrastiva
            labels = (compat_gt > 0.5).float() * 2 - 1  # Convertir a +1/-1
            losses['compatibility'] = self.cosine_loss(
                compat_pred[:, :compat_pred.size(1)//2],
                compat_pred[:, compat_pred.size(1)//2:],
                labels
            )
        
        # 4. Legality Loss (penalizar acciones ilegales)
        if 'logits' in predictions and 'legal_mask' in targets:
            logits = predictions['logits']
            legal_mask = targets['legal_mask']  # [B, N_actions]
            
            # Penalizar probabilidad en acciones ilegales
            probs = F.softmax(logits, dim=-1)
            illegal_probs = probs * (1 - legal_mask)
            losses['legality'] = illegal_probs.sum() / legal_mask.size(0)
        
        # Pérdida total ponderada
        total_loss = torch.tensor(0.0, device=predictions.get('logits', torch.zeros(1)).device)
        
        if 'policy' in losses:
            total_loss += self.config.policy_weight * losses['policy']
        if 'value' in losses:
            total_loss += self.config.value_weight * losses['value']
        if 'compatibility' in losses:
            total_loss += self.config.compatibility_weight * losses['compatibility']
        if 'legality' in losses:
            total_loss += self.config.legality_weight * losses['legality']
        
        losses['total'] = total_loss
        
        return losses

class CurriculumScheduler:
    """Maneja el currículo de entrenamiento (fácil → difícil)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_difficulty = config.initial_difficulty
        self.step_count = 0
        
    def get_difficulty(self) -> float:
        """Retorna nivel de dificultad actual [0, 1]"""
        return min(self.current_difficulty, 1.0)
    
    def step(self, success_rate: float):
        """Actualiza dificultad basado en tasa de éxito"""
        self.step_count += 1
        
        # Aumentar dificultad si el modelo está teniendo éxito
        if success_rate > 0.7:
            self.current_difficulty += self.config.difficulty_increment
            self.current_difficulty = min(self.current_difficulty, 1.0)
        # Reducir si está fallando mucho
        elif success_rate < 0.3:
            self.current_difficulty *= 0.9
            self.current_difficulty = max(self.current_difficulty, 0.1)
    
    def sample_puzzle_difficulty(self, puzzles: List[Dict]) -> List[Dict]:
        """Selecciona puzzles según dificultad actual"""
        difficulty = self.get_difficulty()
        
        # Ordenar puzzles por dificultad estimada
        sorted_puzzles = sorted(puzzles, key=lambda p: self._estimate_difficulty(p))
        
        # Seleccionar subset según dificultad
        max_idx = int(len(sorted_puzzles) * difficulty)
        max_idx = max(1, min(max_idx, len(sorted_puzzles)))
        
        return sorted_puzzles[:max_idx]
    
    def _estimate_difficulty(self, puzzle: Dict) -> float:
        """Estima dificultad de un puzzle"""
        input_grid = np.array(puzzle.get('input', []))
        output_grid = np.array(puzzle.get('output', []))
        
        # Métricas de complejidad
        size_complexity = input_grid.size / 100.0
        color_complexity = len(np.unique(input_grid)) / 10.0
        change_complexity = np.sum(input_grid != output_grid) / input_grid.size
        
        return (size_complexity + color_complexity + change_complexity) / 3

class DataAugmentation:
    """Augmentación de datos para ARC"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def augment(self, grid: np.ndarray) -> np.ndarray:
        """Aplica augmentaciones aleatorias"""
        augmented = grid.copy()
        
        # Flips
        if self.config.augment_flips and random.random() < 0.5:
            if random.random() < 0.5:
                augmented = np.fliplr(augmented)
            else:
                augmented = np.flipud(augmented)
        
        # Rotaciones
        if self.config.augment_rotations and random.random() < 0.5:
            k = random.choice([1, 2, 3])
            augmented = np.rot90(augmented, k)
        
        # Ruido
        if self.config.augment_noise > 0:
            noise_mask = np.random.random(augmented.shape) < self.config.augment_noise
            # Solo añadir ruido a píxeles no-cero
            noise_mask = noise_mask & (augmented != 0)
            if np.any(noise_mask):
                # Cambiar ligeramente los valores
                max_val = augmented.max()
                noise = np.random.randint(-1, 2, size=augmented.shape)
                augmented[noise_mask] = np.clip(
                    augmented[noise_mask] + noise[noise_mask],
                    1, max_val
                )
        
        return augmented

class SyntheticDataGenerator:
    """Genera datos sintéticos para entrenamiento"""
    
    def __init__(self, base_puzzles: List[Dict]):
        self.base_puzzles = base_puzzles
        self.transformations = [
            'cross_expansion', 'fill', 'gravity', 
            'color_mapping', 'rotation', 'diagonal'
        ]
        
    def generate_batch(self, batch_size: int) -> List[Dict]:
        """Genera batch de puzzles sintéticos"""
        batch = []
        
        for _ in range(batch_size):
            # Seleccionar puzzle base aleatorio
            base = random.choice(self.base_puzzles)
            
            # Generar variación
            synthetic = self._generate_variation(base)
            batch.append(synthetic)
        
        return batch
    
    def _generate_variation(self, base_puzzle: Dict) -> Dict:
        """Genera variación de un puzzle base"""
        input_grid = np.array(base_puzzle['input'])
        
        # Aplicar transformación aleatoria
        transformation = random.choice(self.transformations)
        output_grid = self._apply_random_transformation(input_grid, transformation)
        
        # Crear estados intermedios (para currículo)
        intermediate_steps = self._generate_intermediate_states(
            input_grid, output_grid, num_steps=random.randint(1, 5)
        )
        
        return {
            'input': input_grid.tolist(),
            'output': output_grid.tolist(),
            'transformation': transformation,
            'intermediate': intermediate_steps
        }
    
    def _apply_random_transformation(self, grid: np.ndarray, trans_type: str) -> np.ndarray:
        """Aplica transformación específica"""
        if trans_type == 'cross_expansion':
            return self._synthetic_cross_expansion(grid)
        elif trans_type == 'fill':
            return self._synthetic_fill(grid)
        elif trans_type == 'gravity':
            return self._synthetic_gravity(grid)
        elif trans_type == 'color_mapping':
            return self._synthetic_color_map(grid)
        elif trans_type == 'rotation':
            return np.rot90(grid, random.choice([1, 2, 3]))
        elif trans_type == 'diagonal':
            return self._synthetic_diagonal(grid)
        else:
            return grid
    
    def _synthetic_cross_expansion(self, grid: np.ndarray) -> np.ndarray:
        """Genera expansión en cruz sintética"""
        result = grid.copy()
        h, w = grid.shape
        
        # Encontrar píxeles no-cero aislados
        for y in range(1, h-1):
            for x in range(1, w-1):
                if grid[y, x] != 0:
                    # Verificar si está aislado
                    neighbors = [
                        grid[y-1, x], grid[y+1, x],
                        grid[y, x-1], grid[y, x+1]
                    ]
                    if all(n == 0 for n in neighbors):
                        # Expandir en cruz
                        value = grid[y, x]
                        result[y-1, x] = value
                        result[y+1, x] = value
                        result[y, x-1] = value
                        result[y, x+1] = value
        
        return result
    
    def _synthetic_fill(self, grid: np.ndarray) -> np.ndarray:
        """Genera relleno sintético"""
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
    
    def _synthetic_gravity(self, grid: np.ndarray) -> np.ndarray:
        """Genera gravedad sintética"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for col in range(w):
            column = grid[:, col]
            non_zero = column[column != 0]
            
            if len(non_zero) > 0:
                result[h-len(non_zero):, col] = non_zero
        
        return result
    
    def _synthetic_color_map(self, grid: np.ndarray) -> np.ndarray:
        """Genera mapeo de colores sintético"""
        result = grid.copy()
        unique_vals = np.unique(grid[grid != 0])
        
        if len(unique_vals) > 0:
            # Crear mapeo aleatorio
            mapping = {val: (val % 9) + 1 for val in unique_vals}
            
            for old_val, new_val in mapping.items():
                result[grid == old_val] = new_val
        
        return result
    
    def _synthetic_diagonal(self, grid: np.ndarray) -> np.ndarray:
        """Genera patrón diagonal sintético"""
        h, w = grid.shape
        if h != w:
            return grid
        
        # Tomar valor de la diagonal
        diagonal_vals = [grid[i, i] for i in range(min(h, w)) if grid[i, i] != 0]
        
        if diagonal_vals:
            return np.full_like(grid, diagonal_vals[0])
        
        return grid
    
    def _generate_intermediate_states(self, start: np.ndarray, 
                                     end: np.ndarray, 
                                     num_steps: int) -> List[np.ndarray]:
        """Genera estados intermedios entre start y end"""
        if num_steps <= 1:
            return [end]
        
        states = []
        diff_mask = (start != end)
        diff_positions = np.argwhere(diff_mask)
        
        if len(diff_positions) > 0:
            # Cambiar gradualmente
            positions_per_step = max(1, len(diff_positions) // num_steps)
            
            current = start.copy()
            for step in range(num_steps):
                # Seleccionar posiciones a cambiar
                start_idx = step * positions_per_step
                end_idx = min(start_idx + positions_per_step, len(diff_positions))
                
                for idx in range(start_idx, end_idx):
                    y, x = diff_positions[idx]
                    current[y, x] = end[y, x]
                
                states.append(current.copy())
        else:
            states = [end] * num_steps
        
        return states

class ARCTrainer:
    """Entrenador principal para el modelo ARC con DL"""
    
    def __init__(self, model, config: TrainingConfig, device='cpu'):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        
        # Optimizador
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Componentes de entrenamiento
        self.loss_fn = ARCLoss(config)
        self.curriculum = CurriculumScheduler(config)
        self.augmentation = DataAugmentation(config)
        
        # Métricas
        self.metrics_history = deque(maxlen=100)
        
    def train_epoch(self, train_data: List[Dict]) -> Dict[str, float]:
        """Entrena una época"""
        self.model.train()
        epoch_losses = []
        epoch_accuracy = []
        
        # Aplicar currículo
        if self.config.curriculum_enabled:
            train_data = self.curriculum.sample_puzzle_difficulty(train_data)
        
        # Crear batches
        for i in range(0, len(train_data), self.config.batch_size):
            batch = train_data[i:i + self.config.batch_size]
            
            # Preparar batch
            batch_inputs, batch_targets = self._prepare_batch(batch)
            
            # Forward pass
            predictions = self.model(batch_inputs)
            
            # Calcular pérdidas
            losses = self.loss_fn(predictions, batch_targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Registrar métricas
            epoch_losses.append(losses['total'].item())
            
            # Calcular accuracy
            if 'logits' in predictions and 'action_gt' in batch_targets:
                preds = predictions['logits'].argmax(dim=-1)
                targets = batch_targets['action_gt']
                accuracy = (preds == targets).float().mean().item()
                epoch_accuracy.append(accuracy)
        
        # Actualizar scheduler
        self.scheduler.step()
        
        # Actualizar currículo
        if epoch_accuracy:
            avg_accuracy = np.mean(epoch_accuracy)
            self.curriculum.step(avg_accuracy)
        
        return {
            'loss': np.mean(epoch_losses) if epoch_losses else 0,
            'accuracy': np.mean(epoch_accuracy) if epoch_accuracy else 0,
            'difficulty': self.curriculum.get_difficulty()
        }
    
    def _prepare_batch(self, batch: List[Dict]) -> Tuple[Dict, Dict]:
        """Prepara batch para entrenamiento"""
        batch_inputs = []
        batch_targets = []
        
        for sample in batch:
            input_grid = np.array(sample['input'])
            output_grid = np.array(sample['output'])
            
            # Aplicar augmentación
            if self.training:
                input_grid = self.augmentation.augment(input_grid)
                output_grid = self.augmentation.augment(output_grid)
            
            # Convertir a tensores
            # ... (conversión específica según el modelo)
            
            batch_inputs.append(input_grid)
            batch_targets.append(output_grid)
        
        # Stack y retornar
        return {'grids': batch_inputs}, {'targets': batch_targets}
    
    def validate(self, val_data: List[Dict]) -> Dict[str, float]:
        """Valida el modelo"""
        self.model.eval()
        val_losses = []
        val_accuracy = []
        
        with torch.no_grad():
            for sample in val_data:
                # Preparar datos
                inputs, targets = self._prepare_batch([sample])
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Calcular pérdidas
                losses = self.loss_fn(predictions, targets)
                val_losses.append(losses['total'].item())
                
                # Calcular accuracy
                if 'logits' in predictions and 'action_gt' in targets:
                    preds = predictions['logits'].argmax(dim=-1)
                    target_actions = targets['action_gt']
                    accuracy = (preds == target_actions).float().mean().item()
                    val_accuracy.append(accuracy)
        
        return {
            'val_loss': np.mean(val_losses) if val_losses else 0,
            'val_accuracy': np.mean(val_accuracy) if val_accuracy else 0
        }