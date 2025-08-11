#!/usr/bin/env python3
"""
Sistema de Deep Learning con Atención Anclada Multi-Head (A-MHA) - VERSIÓN CORREGIDA
Arquitectura completamente diferenciable para ARC Prize
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from torchvision.ops import roi_align

logger = logging.getLogger(__name__)

@dataclass
class AnchorPoint:
    """Punto de anclaje para entidades"""
    x: int
    y: int
    anchor_type: str  # 'center', 'top', 'bottom', 'left', 'right', 'corner'
    entity_id: int
    embedding: Optional[torch.Tensor] = None
    importance: float = 1.0

@dataclass 
class Entity:
    """Entidad detectada (objeto, patrón, región)"""
    id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    anchors: List[AnchorPoint]
    features: Optional[torch.Tensor] = None
    entity_type: str = 'object'  # 'object', 'pattern', 'hole', 'tile'

class LightweightCNN(nn.Module):
    """Encoder visual ligero tipo ConvNeXt-Tiny para features multi-escala"""
    
    def __init__(self, in_channels=10, base_dim=32):
        super().__init__()
        
        # Convolutions multi-escala
        self.conv1 = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1)
        
        # Bloques residuales ligeros (tipo ConvNeXt)
        self.block1 = self._make_convnext_block(base_dim*4)
        self.block2 = self._make_convnext_block(base_dim*4)
        
        # Feature pyramid para multi-escala
        self.pyramid = nn.ModuleDict({
            'fine': nn.Conv2d(base_dim, base_dim, 1),
            'mid': nn.Conv2d(base_dim*2, base_dim, 1),
            'coarse': nn.Conv2d(base_dim*4, base_dim, 1)
        })
        
        self.feature_dim = base_dim
        
    def _make_convnext_block(self, dim):
        """Bloque tipo ConvNeXt (depthwise + pointwise)"""
        return nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),  # Depthwise
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim*2, 1),  # Pointwise expansion
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1),  # Pointwise projection
        )
    
    def forward(self, x):
        """Produce features multi-escala F"""
        # Encoder
        f1 = F.gelu(self.conv1(x))  # Fine scale
        f2 = F.gelu(self.conv2(f1))  # Mid scale
        f3 = F.gelu(self.conv3(f2))  # Coarse scale
        
        # Residual blocks
        f3 = f3 + self.block1(f3)
        f3 = f3 + self.block2(f3)
        
        # Multi-scale features
        features = {
            'fine': self.pyramid['fine'](f1),
            'mid': self.pyramid['mid'](f2),
            'coarse': self.pyramid['coarse'](f3)
        }
        
        return features

class MultiScaleFusion(nn.Module):
    """Módulo para fusión multi-escala de features"""
    
    def __init__(self, in_dim=32, out_dim=128):
        super().__init__()
        # Proyección para fusión de 3 escalas
        self.fusion_conv = nn.Conv2d(in_dim * 3, out_dim, 1)
        self.norm = nn.BatchNorm2d(out_dim)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fusiona features multi-escala en un tensor único"""
        # Obtener tamaño objetivo (escala más fina)
        target_size = features['fine'].shape[-2:]
        
        # Upsampling de escalas más gruesas
        mid_up = F.interpolate(features['mid'], size=target_size, 
                               mode='bilinear', align_corners=False)
        coarse_up = F.interpolate(features['coarse'], size=target_size, 
                                  mode='bilinear', align_corners=False)
        
        # Concatenar en dimensión de canales
        fused = torch.cat([features['fine'], mid_up, coarse_up], dim=1)
        
        # Proyectar y normalizar
        fused = self.fusion_conv(fused)
        fused = self.norm(fused)
        fused = F.relu(fused)
        
        return fused

class DifferentiableROIPool(nn.Module):
    """ROI Pooling diferenciable para entidades"""
    
    def __init__(self, output_size=(1, 1), spatial_scale=1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature maps
            rois: [N, 5] format: [batch_idx, x1, y1, x2, y2]
        Returns:
            pooled: [N, C, output_h, output_w]
        """
        if rois.numel() == 0:
            return torch.zeros((0, features.shape[1], *self.output_size), 
                              device=features.device)
        
        # Usar ROI Align (diferenciable)
        pooled = roi_align(features, rois, self.output_size, self.spatial_scale)
        
        return pooled

class AnchoredMultiHeadAttention(nn.Module):
    """A-MHA: Atención Multi-Head Anclada para acciones puzzle-específicas"""
    
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Heads auxiliares
        self.compatibility_head = nn.Linear(embed_dim*2, 1)  # Compatibilidad local
        self.value_head = nn.Linear(embed_dim, 1)  # Distancia a solución
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, piece_embeds, slot_embeds, mask=None, spatial_bias=None):
        """
        Args:
            piece_embeds: [B, N_pieces, D] embeddings de piezas/tiles
            slot_embeds: [B, N_slots, D] embeddings de huecos/posiciones
            mask: [B, N_pieces, N_slots] máscara de legalidad
            spatial_bias: [B, N_pieces, N_slots] sesgo espacial
        
        Returns:
            logits: [B, N_pieces, N_slots] scores de atención
            value: [B, 1] estimación de valor del estado
            compatibility: [B, N_pieces, N_slots] scores de compatibilidad
        """
        # Asegurar dimensiones correctas
        if len(piece_embeds.shape) == 2:
            piece_embeds = piece_embeds.unsqueeze(0)
        if len(slot_embeds.shape) == 2:
            slot_embeds = slot_embeds.unsqueeze(0)
            
        B, N_p, D = piece_embeds.shape
        _, N_s, _ = slot_embeds.shape
        
        # Multi-head attention
        Q = self.q_proj(piece_embeds).reshape(B, N_p, self.num_heads, self.head_dim)
        K = self.k_proj(slot_embeds).reshape(B, N_s, self.num_heads, self.head_dim)
        V = self.v_proj(slot_embeds).reshape(B, N_s, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('bphd,bshd->bpsh', Q, K) * self.scale
        scores = scores.mean(dim=-1)  # Average over heads [B, N_p, N_s]
        
        # Añadir sesgo espacial
        if spatial_bias is not None:
            scores = scores + spatial_bias
        
        # Aplicar máscara de legalidad
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax para obtener probabilidades
        attn_probs = F.softmax(scores, dim=-1)  # [B, N_p, N_s]
        attn_probs = self.dropout(attn_probs)
        
        # Calcular output - corregir dimensiones
        # V shape: [B, N_s, num_heads, head_dim]
        # attn_probs shape: [B, N_p, N_s]
        # Necesitamos: [B, N_p, D]
        
        # Expandir attn_probs para heads
        attn_probs_expanded = attn_probs.unsqueeze(-1).unsqueeze(-1)  # [B, N_p, N_s, 1, 1]
        attn_probs_expanded = attn_probs_expanded.expand(B, N_p, N_s, self.num_heads, self.head_dim)
        
        # V expandido para match con N_p
        V_expanded = V.unsqueeze(1).expand(B, N_p, N_s, self.num_heads, self.head_dim)
        
        # Weighted sum
        attn_output = (attn_probs_expanded * V_expanded).sum(dim=2)  # [B, N_p, num_heads, head_dim]
        attn_output = attn_output.reshape(B, N_p, D)
        output = self.out_proj(attn_output)
        
        # Head de valor (estima distancia a solución)
        global_features = output.mean(dim=1)  # Pool sobre piezas
        value = self.value_head(global_features)
        
        # Head de compatibilidad
        # Calcular compatibilidad entre pares de embeddings
        piece_expanded = piece_embeds.unsqueeze(2).expand(-1, -1, N_s, -1)
        slot_expanded = slot_embeds.unsqueeze(1).expand(-1, N_p, -1, -1)
        combined = torch.cat([piece_expanded, slot_expanded], dim=-1)
        compatibility = self.compatibility_head(combined).squeeze(-1)
        
        return scores, value, compatibility

class DeepLearningARCSolver(nn.Module):
    """
    Solver ARC con Deep Learning y Atención Anclada - VERSIÓN CORREGIDA
    Completamente diferenciable y entrenable end-to-end
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        
        # Dimensiones
        self.embed_dim = 128
        self.num_anchors = 5
        
        # Módulos del modelo (ahora son parte del grafo computacional)
        self.encoder = LightweightCNN(in_channels=10)
        self.fusion = MultiScaleFusion(in_dim=32, out_dim=self.embed_dim)
        self.roi_pool = DifferentiableROIPool(output_size=(1, 1))
        self.attention = AnchoredMultiHeadAttention(embed_dim=self.embed_dim)
        
        # Proyección de embeddings (ahora es fija y entrenable)
        self.entity_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Head para predicción de transformación
        self.transformation_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 tipos de transformación posibles
        )
        
        # Mover a dispositivo
        self.to(self.device)
        
        # Cache para entidades (no diferenciable)
        self.entities_cache: Dict[str, List[Entity]] = {}
        
    def forward(self, input_tensor: torch.Tensor, 
                target_tensor: Optional[torch.Tensor] = None,
                entities_input: Optional[List[Entity]] = None,
                entities_target: Optional[List[Entity]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass diferenciable end-to-end
        
        Args:
            input_tensor: [B, C, H, W] tensor de entrada
            target_tensor: [B, C, H, W] tensor objetivo (para entrenamiento)
            entities_input: Lista de entidades detectadas en input
            entities_target: Lista de entidades detectadas en target
            
        Returns:
            Dict con 'logits', 'value', 'compatibility', 'transformation'
        """
        B = input_tensor.shape[0]
        
        # 1. Encode input
        input_features = self.encoder(input_tensor)
        fused_input = self.fusion(input_features)
        
        # 2. Detectar entidades si no se proporcionan
        if entities_input is None:
            entities_input = self._detect_entities_batch(input_tensor)
        
        # 3. ROI Pooling diferenciable
        input_rois = self._entities_to_rois_batch(entities_input, B)
        input_pooled = self.roi_pool(fused_input, input_rois)
        
        # 4. Proyectar embeddings
        if input_pooled.numel() > 0:
            input_embeds = input_pooled.squeeze(-1).squeeze(-1)
            input_embeds = self.entity_projection(input_embeds)
            # Reshape para batch
            input_embeds = self._reshape_embeds_for_batch(input_embeds, entities_input, B)
        else:
            input_embeds = torch.zeros((B, 1, self.embed_dim), device=self.device)
        
        results = {'input_embeds': input_embeds}
        
        # 5. Si hay target, calcular atención y pérdidas
        if target_tensor is not None:
            # Encode target
            target_features = self.encoder(target_tensor)
            fused_target = self.fusion(target_features)
            
            # Detectar entidades target
            if entities_target is None:
                entities_target = self._detect_entities_batch(target_tensor)
            
            # ROI Pooling para target
            target_rois = self._entities_to_rois_batch(entities_target, B)
            target_pooled = self.roi_pool(fused_target, target_rois)
            
            # Proyectar embeddings target
            if target_pooled.numel() > 0:
                target_embeds = target_pooled.squeeze(-1).squeeze(-1)
                target_embeds = self.entity_projection(target_embeds)
                target_embeds = self._reshape_embeds_for_batch(target_embeds, entities_target, B)
            else:
                target_embeds = torch.zeros((B, 1, self.embed_dim), device=self.device)
            
            # 6. Calcular atención
            spatial_bias = self._compute_spatial_bias_batch(entities_input, entities_target, B)
            logits, value, compatibility = self.attention(
                input_embeds, target_embeds, spatial_bias=spatial_bias
            )
            
            # 7. Predecir tipo de transformación
            # Concatenar embeddings globales
            input_global = input_embeds.mean(dim=1)  # [B, D]
            target_global = target_embeds.mean(dim=1)  # [B, D]
            combined = torch.cat([input_global, target_global], dim=-1)  # [B, 2*D]
            transformation_logits = self.transformation_head(combined)  # [B, 10]
            
            results.update({
                'logits': logits,
                'value': value,
                'compatibility': compatibility,
                'transformation': transformation_logits,
                'target_embeds': target_embeds
            })
        
        return results
    
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Interfaz de inferencia para resolver puzzles"""
        self.eval()
        with torch.no_grad():
            solution, _ = self.solve_with_steps(train_examples, test_input)
        return solution
    
    def solve_with_steps(self, train_examples: List[Dict], 
                         test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve puzzle ARC usando el modelo entrenado
        
        Args:
            train_examples: Ejemplos de entrenamiento
            test_input: Entrada a resolver
            
        Returns:
            solution: Grid resuelto
            steps: Pasos del proceso
        """
        steps = []
        
        if not train_examples:
            return test_input, [{"description": "No hay ejemplos"}]
        
        # Analizar primer ejemplo para aprender transformación
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # Convertir a tensores
        input_tensor = self._grid_to_tensor(input_grid)
        output_tensor = self._grid_to_tensor(output_grid)
        
        # Forward pass para aprender transformación
        with torch.no_grad():
            results = self.forward(input_tensor, output_tensor)
            
            # Obtener tipo de transformación predicha
            if 'transformation' in results:
                trans_probs = F.softmax(results['transformation'], dim=-1)
                trans_type = trans_probs.argmax(dim=-1).item()
                trans_confidence = trans_probs.max(dim=-1).values.item()
                
                trans_names = ['cross_expansion', 'fill', 'gravity', 'color_mapping', 
                              'rotation', 'diagonal', 'mirror', 'tile', 'scale', 'unknown']
                trans_name = trans_names[trans_type] if trans_type < len(trans_names) else 'unknown'
                
                steps.append({
                    "description": f"Transformación detectada: {trans_name} (confianza: {trans_confidence:.2%})"
                })
        
        # Detectar entidades en test
        test_entities = self._detect_entities(test_input)
        test_tensor = self._grid_to_tensor(test_input)
        
        # Aplicar transformación aprendida
        with torch.no_grad():
            test_results = self.forward(test_tensor)
            test_embeds = test_results['input_embeds']
        
        # Aplicar transformación detectada
        transformation = {'type': trans_name, 'confidence': trans_confidence}
        solution = self._apply_transformation_learned(
            test_input, transformation, test_entities, test_embeds
        )
        
        steps.append({
            "description": f"Aplicada transformación {trans_name} al test"
        })
        
        return solution, steps
    
    def _detect_entities_batch(self, tensor: torch.Tensor) -> List[List[Entity]]:
        """Detecta entidades para un batch de tensores"""
        batch_entities = []
        B = tensor.shape[0]
        
        for b in range(B):
            # Convertir a numpy para detección
            grid = self._tensor_to_grid(tensor[b])
            entities = self._detect_entities(grid)
            batch_entities.append(entities)
        
        return batch_entities
    
    def _detect_entities(self, grid: np.ndarray) -> List[Entity]:
        """Detecta entidades (objetos, patrones, regiones) y sus anclas"""
        entities = []
        
        # Detectar componentes conectados
        binary = grid != 0
        labeled_array, num_features = label(binary)
        
        for i in range(1, num_features + 1):
            mask = (labeled_array == i)
            coords = np.argwhere(mask)
            
            if len(coords) == 0:
                continue
            
            # Bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (x_min, y_min, x_max, y_max)
            
            # Crear entidad
            entity = Entity(
                id=i,
                mask=mask,
                bbox=bbox,
                anchors=[]
            )
            
            # Generar anclas
            entity.anchors = self._generate_anchors(mask, entity.id)
            entities.append(entity)
        
        return entities
    
    def _generate_anchors(self, mask: np.ndarray, entity_id: int) -> List[AnchorPoint]:
        """Genera puntos de anclaje para una entidad"""
        anchors = []
        coords = np.argwhere(mask)
        
        if len(coords) == 0:
            return anchors
        
        # Centro de masa
        center_y, center_x = center_of_mass(mask)
        anchors.append(AnchorPoint(
            x=int(center_x), y=int(center_y),
            anchor_type='center', entity_id=entity_id,
            importance=1.0
        ))
        
        # Extremos
        for anchor_type, selector in [
            ('top', lambda c: c[c[:, 0] == c[:, 0].min()]),
            ('bottom', lambda c: c[c[:, 0] == c[:, 0].max()]),
            ('left', lambda c: c[c[:, 1] == c[:, 1].min()]),
            ('right', lambda c: c[c[:, 1] == c[:, 1].max()])
        ]:
            extremes = selector(coords)
            if len(extremes) > 0:
                y, x = extremes[len(extremes)//2]
                anchors.append(AnchorPoint(
                    x=x, y=y, anchor_type=anchor_type,
                    entity_id=entity_id, importance=0.8
                ))
                
                if len(anchors) >= self.num_anchors:
                    break
        
        return anchors
    
    def _entities_to_rois_batch(self, batch_entities: List[List[Entity]], 
                                batch_size: int) -> torch.Tensor:
        """Convierte entidades a ROIs para todo el batch"""
        all_rois = []
        
        for batch_idx, entities in enumerate(batch_entities):
            for entity in entities:
                x1, y1, x2, y2 = entity.bbox
                # Formato: [batch_idx, x1, y1, x2, y2]
                roi = torch.tensor([batch_idx, x1, y1, x2, y2], 
                                  dtype=torch.float32, device=self.device)
                all_rois.append(roi)
        
        if all_rois:
            return torch.stack(all_rois)
        else:
            # ROI vacío si no hay entidades
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)
    
    def _reshape_embeds_for_batch(self, embeds: torch.Tensor, 
                                  batch_entities: List[List[Entity]], 
                                  batch_size: int) -> torch.Tensor:
        """Reorganiza embeddings para formato batch"""
        # Encuentra el número máximo de entidades en el batch
        max_entities = max(len(entities) for entities in batch_entities)
        if max_entities == 0:
            return torch.zeros((batch_size, 1, self.embed_dim), device=self.device)
        
        # Crear tensor de salida con padding
        batch_embeds = torch.zeros((batch_size, max_entities, self.embed_dim), 
                                   device=self.device)
        
        # Llenar con embeddings reales
        start_idx = 0
        for b, entities in enumerate(batch_entities):
            n_entities = len(entities)
            if n_entities > 0:
                batch_embeds[b, :n_entities] = embeds[start_idx:start_idx+n_entities]
                start_idx += n_entities
        
        return batch_embeds
    
    def _compute_spatial_bias_batch(self, input_entities: List[List[Entity]],
                                    target_entities: List[List[Entity]],
                                    batch_size: int) -> torch.Tensor:
        """Calcula sesgo espacial para todo el batch"""
        max_input = max(len(e) for e in input_entities) if input_entities else 1
        max_target = max(len(e) for e in target_entities) if target_entities else 1
        
        bias = torch.zeros((batch_size, max_input, max_target), device=self.device)
        lambda_dist = 0.1
        
        for b in range(batch_size):
            if b < len(input_entities) and b < len(target_entities):
                for i, src in enumerate(input_entities[b]):
                    for j, tgt in enumerate(target_entities[b]):
                        if src.anchors and tgt.anchors:
                            src_center = src.anchors[0]
                            tgt_center = tgt.anchors[0]
                            
                            dist = np.sqrt((src_center.x - tgt_center.x)**2 + 
                                         (src_center.y - tgt_center.y)**2)
                            bias[b, i, j] = -lambda_dist * dist
        
        return bias
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convierte grid a tensor one-hot para CNN"""
        h, w = grid.shape
        max_val = 10  # ARC usa valores 0-9
        
        # One-hot encoding
        one_hot = np.zeros((max_val, h, w), dtype=np.float32)
        for i in range(max_val):
            one_hot[i] = (grid == i).astype(np.float32)
        
        # Añadir batch dimension
        tensor = torch.from_numpy(one_hot).unsqueeze(0).to(self.device)
        return tensor
    
    def _tensor_to_grid(self, tensor: torch.Tensor) -> np.ndarray:
        """Convierte tensor one-hot a grid numpy"""
        # Remover batch dimension si existe
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Argmax sobre canales para obtener valores
        grid = tensor.argmax(dim=0).cpu().numpy()
        return grid
    
    def _apply_transformation_learned(self, grid: np.ndarray,
                                     transformation: Dict,
                                     entities: List[Entity],
                                     embeddings: torch.Tensor) -> np.ndarray:
        """Aplica transformación aprendida por el modelo"""
        trans_type = transformation['type']
        
        if trans_type == 'cross_expansion':
            return self._apply_cross_expansion(grid, entities)
        elif trans_type == 'fill':
            return self._apply_fill(grid)
        elif trans_type == 'gravity':
            return self._apply_gravity(grid)
        elif trans_type == 'color_mapping':
            return self._apply_color_mapping(grid, {})
        elif trans_type == 'rotation':
            return np.rot90(grid)
        elif trans_type == 'mirror':
            return np.fliplr(grid)
        elif trans_type == 'diagonal':
            return self._apply_diagonal(grid)
        elif trans_type == 'tile':
            return self._apply_tiling(grid)
        elif trans_type == 'scale':
            return self._apply_scaling(grid)
        else:
            return grid
    
    # === Métodos de transformación (igual que antes) ===
    
    def _apply_cross_expansion(self, grid: np.ndarray, entities: List[Entity]) -> np.ndarray:
        """Aplica expansión en cruz usando información de entidades"""
        result = grid.copy()
        h, w = grid.shape
        
        for entity in entities:
            if entity.anchors:
                center = entity.anchors[0]
                value = grid[center.y, center.x]
                
                if value != 0:
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = center.y + dy, center.x + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                            result[ny, nx] = value
        
        return result
    
    def _apply_fill(self, grid: np.ndarray) -> np.ndarray:
        """Rellena espacios cerrados"""
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
        """Aplica gravedad (objetos caen)"""
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
            result[result > 9] = 9  # Limitar al rango válido
        else:
            for old_val, new_val in mapping.items():
                result[grid == old_val] = new_val
        
        return result
    
    def _apply_diagonal(self, grid: np.ndarray) -> np.ndarray:
        """Aplica transformación diagonal"""
        h, w = grid.shape
        if h != w:
            return grid
        
        result = grid.copy()
        for i in range(min(h, w)):
            if grid[i, i] != 0:
                result[i, :] = grid[i, i]
                result[:, i] = grid[i, i]
        
        return result
    
    def _apply_tiling(self, grid: np.ndarray) -> np.ndarray:
        """Aplica patrón de mosaico"""
        h, w = grid.shape
        pattern_h, pattern_w = h // 2, w // 2
        
        if pattern_h > 0 and pattern_w > 0:
            pattern = grid[:pattern_h, :pattern_w]
            result = np.tile(pattern, (2, 2))[:h, :w]
            return result
        
        return grid
    
    def _apply_scaling(self, grid: np.ndarray) -> np.ndarray:
        """Aplica escalado 2x"""
        result = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
        return result[:grid.shape[0], :grid.shape[1]]  # Mantener tamaño original