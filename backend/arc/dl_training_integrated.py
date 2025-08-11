#!/usr/bin/env python3
"""
Sistema de Entrenamiento Integrado para Deep Learning ARC Solver
Conecta el solver corregido con el sistema de entrenamiento
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from glob import glob
from dataclasses import dataclass
from collections import deque
import random
from tqdm import tqdm

# Importar el solver corregido
from deep_learning_solver_fixed import DeepLearningARCSolver

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_steps: int = 100
    
    # Pesos de pérdidas
    policy_weight: float = 1.0
    value_weight: float = 0.5
    compatibility_weight: float = 0.3
    transformation_weight: float = 0.8
    
    # Configuración de datos
    train_split: float = 0.8
    val_split: float = 0.2
    max_grid_size: int = 30
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 5

class ARCDataset(Dataset):
    """Dataset para puzzles ARC"""
    
    def __init__(self, data_path: str, transform=None, max_size=30):
        self.data_path = data_path
        self.transform = transform
        self.max_size = max_size
        self.puzzles = self._load_puzzles()
        
    def _load_puzzles(self) -> List[Dict]:
        """Carga puzzles desde archivos JSON"""
        puzzles = []
        
        # Buscar archivos JSON en el directorio
        json_files = glob(os.path.join(self.data_path, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    puzzle_data = json.load(f)
                    
                    # Formato esperado del ARC
                    if 'train' in puzzle_data:
                        for example in puzzle_data['train']:
                            if 'input' in example and 'output' in example:
                                input_grid = np.array(example['input'])
                                output_grid = np.array(example['output'])
                                
                                # Filtrar por tamaño
                                if (input_grid.shape[0] <= self.max_size and 
                                    input_grid.shape[1] <= self.max_size):
                                    puzzles.append({
                                        'id': os.path.basename(json_file).replace('.json', ''),
                                        'input': input_grid,
                                        'output': output_grid
                                    })
                                    
                    # También incluir ejemplos de test si están disponibles
                    if 'test' in puzzle_data:
                        for example in puzzle_data['test']:
                            if 'input' in example and 'output' in example:
                                input_grid = np.array(example['input'])
                                output_grid = np.array(example['output'])
                                
                                if (input_grid.shape[0] <= self.max_size and 
                                    input_grid.shape[1] <= self.max_size):
                                    puzzles.append({
                                        'id': os.path.basename(json_file).replace('.json', '') + '_test',
                                        'input': input_grid,
                                        'output': output_grid
                                    })
            except Exception as e:
                logger.warning(f"Error cargando {json_file}: {e}")
                continue
        
        logger.info(f"Cargados {len(puzzles)} puzzles desde {self.data_path}")
        return puzzles
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        puzzle = self.puzzles[idx]
        
        # Aplicar transformaciones si están definidas
        if self.transform:
            puzzle = self.transform(puzzle)
        
        return puzzle

class ARCCollator:
    """Collator personalizado para manejar grids de diferentes tamaños"""
    
    def __init__(self, max_size=30, device='cpu'):
        self.max_size = max_size
        self.device = device
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convierte batch de puzzles a tensores con padding
        """
        batch_inputs = []
        batch_outputs = []
        batch_masks = []
        
        for sample in batch:
            input_grid = sample['input']
            output_grid = sample['output']
            
            # Padding a tamaño máximo
            input_padded = self._pad_grid(input_grid)
            output_padded = self._pad_grid(output_grid)
            
            # Crear máscara para indicar área válida
            mask = np.zeros((self.max_size, self.max_size))
            mask[:input_grid.shape[0], :input_grid.shape[1]] = 1
            
            batch_inputs.append(input_padded)
            batch_outputs.append(output_padded)
            batch_masks.append(mask)
        
        # Convertir a tensores
        return {
            'input': torch.stack([self._grid_to_tensor(g) for g in batch_inputs]),
            'output': torch.stack([self._grid_to_tensor(g) for g in batch_outputs]),
            'mask': torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=self.device)
        }
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Aplica padding a un grid"""
        h, w = grid.shape
        padded = np.zeros((self.max_size, self.max_size), dtype=grid.dtype)
        padded[:h, :w] = grid
        return padded
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convierte grid a tensor one-hot"""
        max_val = 10
        h, w = grid.shape
        
        # One-hot encoding
        one_hot = np.zeros((max_val, h, w), dtype=np.float32)
        for i in range(max_val):
            one_hot[i] = (grid == i).astype(np.float32)
        
        return torch.from_numpy(one_hot).to(self.device)

class ARCLossIntegrated(nn.Module):
    """Pérdidas adaptadas para el solver integrado"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Pérdidas individuales
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Calcula todas las pérdidas
        
        Args:
            predictions: Dict con outputs del modelo
            targets: Dict con ground truth
        
        Returns:
            Dict con pérdidas individuales y total
        """
        losses = {}
        device = predictions.get('logits', torch.zeros(1)).device
        
        # 1. Policy Loss (atención sobre slots)
        if 'logits' in predictions and 'output' in targets:
            logits = predictions['logits']  # [B, N_pieces, N_slots]
            
            # Para simplificar, usar pérdida de reconstrucción
            # Esto es una aproximación - idealmente necesitaríamos ground truth de acciones
            if len(logits.shape) == 3:
                B, N_p, N_s = logits.shape
                # Flatten para cross entropy
                logits_flat = logits.view(B, -1)
                # Crear target sintético (primera acción válida)
                target_actions = torch.zeros(B, dtype=torch.long, device=device)
                losses['policy'] = self.ce_loss(logits_flat, target_actions)
            else:
                losses['policy'] = torch.tensor(0.0, device=device)
        
        # 2. Value Loss (estimación de calidad)
        if 'value' in predictions:
            value_pred = predictions['value']  # [B, 1]
            # Ground truth: 1.0 si la transformación es correcta
            value_gt = torch.ones_like(value_pred)
            losses['value'] = self.mse_loss(value_pred, value_gt)
        
        # 3. Compatibility Loss
        if 'compatibility' in predictions:
            compat = predictions['compatibility']  # [B, N_pieces, N_slots]
            # Crear target de compatibilidad (todos compatibles por ahora)
            compat_gt = torch.ones_like(compat)
            losses['compatibility'] = self.bce_loss(compat, compat_gt)
        
        # 4. Transformation Loss (clasificación de tipo de transformación)
        if 'transformation' in predictions:
            trans_logits = predictions['transformation']  # [B, 10]
            # Detectar transformación real comparando input/output
            trans_gt = self._detect_transformation_type(targets)
            losses['transformation'] = self.ce_loss(trans_logits, trans_gt)
        
        # Pérdida total ponderada
        total_loss = torch.tensor(0.0, device=device)
        
        if 'policy' in losses:
            total_loss += self.config.policy_weight * losses['policy']
        if 'value' in losses:
            total_loss += self.config.value_weight * losses['value']
        if 'compatibility' in losses:
            total_loss += self.config.compatibility_weight * losses['compatibility']
        if 'transformation' in losses:
            total_loss += self.config.transformation_weight * losses['transformation']
        
        losses['total'] = total_loss
        
        return losses
    
    def _detect_transformation_type(self, targets: Dict) -> torch.Tensor:
        """Detecta el tipo de transformación comparando input/output"""
        # Por ahora, retornar tipo aleatorio
        # En producción, implementar detección real
        B = targets['output'].shape[0]
        return torch.randint(0, 10, (B,), device=targets['output'].device)

class ARCTrainerIntegrated:
    """Entrenador integrado para el modelo ARC con DL"""
    
    def __init__(self, model: DeepLearningARCSolver, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Mover modelo a dispositivo
        self.model.to(self.device)
        
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
        
        # Loss function
        self.loss_fn = ARCLossIntegrated(config)
        
        # Métricas
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Crear directorio de checkpoints
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Entrena una época"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Forward pass
            predictions = self.model(
                batch['input'].to(self.device),
                batch['output'].to(self.device)
            )
            
            # Calcular pérdidas
            losses = self.loss_fn(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Registrar métricas
            loss_val = losses['total'].item()
            epoch_losses.append(loss_val)
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        # Actualizar scheduler
        self.scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Valida el modelo"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Forward pass
                predictions = self.model(
                    batch['input'].to(self.device),
                    batch['output'].to(self.device)
                )
                
                # Calcular pérdidas
                losses = self.loss_fn(predictions, batch)
                val_losses.append(losses['total'].item())
        
        avg_loss = np.mean(val_losses)
        return {'val_loss': avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Loop de entrenamiento completo"""
        logger.info(f"Iniciando entrenamiento por {self.config.num_epochs} épocas")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n--- Época {epoch+1}/{self.config.num_epochs} ---")
            
            # Entrenar
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['train_loss'])
            
            # Validar
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['val_loss'])
            
            # Log métricas
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Guardar mejor modelo
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)
                logger.info("✓ Nuevo mejor modelo guardado")
            
            # Guardar checkpoint regular
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guarda checkpoint del modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint del modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint cargado desde {path}")
        return checkpoint['epoch']

def main():
    """Función principal para entrenar el modelo"""
    
    # Configuración
    config = TrainingConfig(
        batch_size=4,
        num_epochs=50,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info(f"Usando dispositivo: {config.device}")
    
    # Crear datasets
    data_path = "/app/arc/arc_official_cache"  # Ajustar según tu configuración
    
    if not os.path.exists(data_path):
        logger.error(f"No se encuentra el directorio de datos: {data_path}")
        return
    
    dataset = ARCDataset(data_path, max_size=config.max_grid_size)
    
    # División train/val
    n_samples = len(dataset)
    n_train = int(n_samples * config.train_split)
    n_val = n_samples - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    logger.info(f"Datasets creados: {n_train} train, {n_val} val")
    
    # Crear data loaders
    collator = ARCCollator(max_size=config.max_grid_size, device=config.device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Cambiar si tienes múltiples CPUs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Crear modelo
    model = DeepLearningARCSolver(device=config.device)
    
    # Crear entrenador
    trainer = ARCTrainerIntegrated(model, config)
    
    # Entrenar
    trainer.train(train_loader, val_loader)
    
    logger.info("Entrenamiento completado")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()