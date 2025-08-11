#!/usr/bin/env python3
"""
Sistema de Deep Learning con Atención Anclada Multi-Head (A-MHA)
Arquitectura anti-clásica pero eficiente para ARC Prize
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
        scores = scores.mean(dim=-1)  # Average over heads
        
        # Añadir sesgo espacial
        if spatial_bias is not None:
            scores = scores + spatial_bias
        
        # Aplicar máscara de legalidad
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax para obtener probabilidades
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Calcular output
        attn_output = torch.einsum('bps,bshd->bphd', attn_probs, V.transpose(1, 2))
        attn_output = attn_output.reshape(B, N_p, D)
        output = self.out_proj(attn_output)
        
        # Head de valor (estima distancia a solución)
        global_features = output.mean(dim=1)  # Pool sobre piezas
        value = self.value_head(global_features)
        
        return scores, value

class DeepLearningARCSolver:
    """
    Solver ARC con Deep Learning y Atención Anclada
    Implementa la arquitectura propuesta con CNN + A-MHA
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Modelos
        self.encoder = LightweightCNN(in_channels=10).to(self.device)
        self.attention = AnchoredMultiHeadAttention(embed_dim=128).to(self.device)
        
        # Dimensiones
        self.embed_dim = 128
        self.num_anchors = 5  # Anclas por entidad
        
        # Cache
        self.entities: List[Entity] = []
        self.anchors: List[AnchorPoint] = []
        
        # Poner en modo evaluación
        self.encoder.eval()
        self.attention.eval()
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Interfaz simple para resolver puzzles"""
        solution, _ = self.solve_with_steps(train_examples, test_input)
        return solution
    
    def solve_with_steps(self, train_examples: List[Dict], test_input: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Resuelve puzzle ARC usando DL con atención anclada
        
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
        
        # Analizar primer ejemplo
        first_example = train_examples[0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # PASO 1: Detectar entidades y anclas
        entities_input = self._detect_entities(input_grid)
        entities_output = self._detect_entities(output_grid)
        
        steps.append({
            "description": f"Detectadas {len(entities_input)} entidades input, "
                          f"{len(entities_output)} entidades output"
        })
        
        # PASO 2: Extraer features con CNN
        with torch.no_grad():
            input_tensor = self._grid_to_tensor(input_grid)
            output_tensor = self._grid_to_tensor(output_grid)
            
            input_features = self.encoder(input_tensor)
            output_features = self.encoder(output_tensor)
        
        # PASO 3: Calcular embeddings por ROI pooling
        input_embeds = self._compute_entity_embeddings(entities_input, input_features)
        output_embeds = self._compute_entity_embeddings(entities_output, output_features)
        
        # PASO 4: Aplicar A-MHA para encontrar correspondencias
        action_logits, value = self._compute_attention_actions(
            input_embeds, output_embeds, entities_input, entities_output
        )
        
        steps.append({
            "description": f"Atención calculada, valor estimado: {value:.2f}"
        })
        
        # PASO 5: Detectar transformación
        transformation = self._detect_transformation_dl(
            input_grid, output_grid, action_logits, entities_input, entities_output
        )
        
        steps.append({
            "description": f"Transformación detectada: {transformation['type']}"
        })
        
        # PASO 6: Aplicar transformación al test
        test_entities = self._detect_entities(test_input)
        test_tensor = self._grid_to_tensor(test_input)
        
        with torch.no_grad():
            test_features = self.encoder(test_tensor)
            test_embeds = self._compute_entity_embeddings(test_entities, test_features)
        
        solution = self._apply_transformation_dl(
            test_input, transformation, test_entities, test_embeds
        )
        
        return solution, steps
    
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
    
    def _compute_entity_embeddings(self, entities: List[Entity], 
                                   features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calcula embeddings por ROI pooling sobre las máscaras"""
        if not entities:
            return torch.zeros((1, 1, self.embed_dim), device=self.device)
        
        embeddings = []
        
        for entity in entities:
            # ROI pooling sobre la máscara
            x1, y1, x2, y2 = entity.bbox
            
            # Usar features de escala apropiada
            if (x2 - x1) * (y2 - y1) < 9:  # Entidad pequeña
                feat_map = features['fine']
            elif (x2 - x1) * (y2 - y1) < 25:  # Entidad mediana
                feat_map = features['mid']
            else:  # Entidad grande
                feat_map = features['coarse']
            
            # Adaptar coordenadas a la escala del feature map
            h_feat, w_feat = feat_map.shape[-2:]
            h_orig, w_orig = entity.mask.shape
            
            y1_scaled = int(y1 * h_feat / h_orig)
            y2_scaled = int(y2 * h_feat / h_orig)
            x1_scaled = int(x1 * w_feat / w_orig)
            x2_scaled = int(x2 * w_feat / w_orig)
            
            # Extraer región y pool
            roi = feat_map[:, :, y1_scaled:y2_scaled+1, x1_scaled:x2_scaled+1]
            
            if roi.numel() > 0:
                pooled = F.adaptive_avg_pool2d(roi, (1, 1))
                embed = pooled.squeeze(-1).squeeze(-1)
            else:
                # Usar dimensión del feature map, no embed_dim
                embed = torch.zeros((1, feat_map.shape[1]), device=self.device)
            
            embeddings.append(embed)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=1)
        
        # Proyectar a dimensión de embedding
        if embeddings.shape[-1] != self.embed_dim:
            # Crear proyección con dimensiones correctas
            in_dim = embeddings.shape[-1]
            proj = nn.Linear(in_dim, self.embed_dim).to(self.device)
            with torch.no_grad():
                # Inicializar pesos
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
            embeddings = proj(embeddings)
        
        return embeddings
    
    def _compute_attention_actions(self, source_embeds: torch.Tensor,
                                   target_embeds: torch.Tensor,
                                   source_entities: List[Entity],
                                   target_entities: List[Entity]) -> Tuple[torch.Tensor, float]:
        """Calcula scores de atención para posibles acciones"""
        
        # Calcular máscara de legalidad (todas las combinaciones son válidas inicialmente)
        N_s = len(source_entities) if source_entities else 1
        N_t = len(target_entities) if target_entities else 1
        mask = torch.ones((1, N_s, N_t), device=self.device)
        
        # Calcular sesgo espacial basado en distancias
        spatial_bias = self._compute_spatial_bias(source_entities, target_entities)
        
        # Aplicar A-MHA
        logits, value = self.attention(source_embeds, target_embeds, mask, spatial_bias)
        
        return logits, value.item()
    
    def _compute_spatial_bias(self, source_entities: List[Entity],
                              target_entities: List[Entity]) -> torch.Tensor:
        """Calcula sesgo espacial basado en distancias entre anclas"""
        if not source_entities or not target_entities:
            return torch.zeros((1, 1, 1), device=self.device)
        
        N_s = len(source_entities)
        N_t = len(target_entities)
        bias = torch.zeros((1, N_s, N_t), device=self.device)
        
        lambda_dist = 0.1  # Factor de penalización por distancia
        
        for i, src in enumerate(source_entities):
            for j, tgt in enumerate(target_entities):
                # Distancia entre centros
                src_center = src.anchors[0] if src.anchors else None
                tgt_center = tgt.anchors[0] if tgt.anchors else None
                
                if src_center and tgt_center:
                    dist = np.sqrt((src_center.x - tgt_center.x)**2 + 
                                  (src_center.y - tgt_center.y)**2)
                    bias[0, i, j] = -lambda_dist * dist
        
        return bias
    
    def _detect_transformation_dl(self, input_grid: np.ndarray,
                                  output_grid: np.ndarray,
                                  action_logits: torch.Tensor,
                                  input_entities: List[Entity],
                                  output_entities: List[Entity]) -> Dict:
        """Detecta transformación usando los scores de atención"""
        
        # Obtener acción con mayor score
        if action_logits.numel() > 0:
            max_idx = action_logits.argmax().item()
            N_t = action_logits.shape[-1]
            src_idx = max_idx // N_t
            tgt_idx = max_idx % N_t
        else:
            src_idx = tgt_idx = 0
        
        # Analizar patrones básicos
        patterns = []
        
        # Expansión en cruz
        if self._is_cross_expansion(input_grid, output_grid):
            patterns.append({'type': 'cross_expansion', 'confidence': 0.9})
        
        # Relleno
        if self._is_fill_pattern(input_grid, output_grid):
            patterns.append({'type': 'fill', 'confidence': 0.9})
        
        # Gravedad
        if self._is_gravity_pattern(input_grid, output_grid):
            patterns.append({'type': 'gravity', 'confidence': 0.85})
        
        # Mapeo de colores
        mapping = self._detect_color_mapping(input_grid, output_grid)
        if mapping:
            patterns.append({'type': 'color_mapping', 'confidence': 0.8, 'mapping': mapping})
        
        # Rotación
        if np.array_equal(np.rot90(input_grid), output_grid):
            patterns.append({'type': 'rotation', 'confidence': 1.0})
        
        if patterns:
            return max(patterns, key=lambda p: p['confidence'])
        
        return {'type': 'unknown', 'confidence': 0.0}
    
    def _apply_transformation_dl(self, grid: np.ndarray,
                                 transformation: Dict,
                                 entities: List[Entity],
                                 embeddings: torch.Tensor) -> np.ndarray:
        """Aplica transformación detectada con información de DL"""
        trans_type = transformation['type']
        
        if trans_type == 'cross_expansion':
            return self._apply_cross_expansion(grid, entities)
            
        elif trans_type == 'fill':
            return self._apply_fill(grid)
            
        elif trans_type == 'gravity':
            return self._apply_gravity(grid)
            
        elif trans_type == 'color_mapping':
            mapping = transformation.get('mapping', {})
            return self._apply_color_mapping(grid, mapping)
            
        elif trans_type == 'rotation':
            return np.rot90(grid)
        
        return grid
    
    # === Métodos de transformación ===
    
    def _apply_cross_expansion(self, grid: np.ndarray, entities: List[Entity]) -> np.ndarray:
        """Aplica expansión en cruz usando información de entidades"""
        result = grid.copy()
        h, w = grid.shape
        
        for entity in entities:
            # Usar ancla central
            if entity.anchors:
                center = entity.anchors[0]  # Primera ancla es el centro
                value = grid[center.y, center.x]
                
                if value != 0:
                    # Expandir en cruz
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = center.y + dy, center.x + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                            result[ny, nx] = value
        
        return result
    
    def _apply_fill(self, grid: np.ndarray) -> np.ndarray:
        """Rellena espacios cerrados"""
        result = grid.copy()
        h, w = grid.shape
        
        # Detectar espacios cerrados y rellenar
        for y in range(1, h-1):
            for x in range(1, w-1):
                if grid[y, x] == 0:
                    # Verificar si está rodeado
                    neighbors = [
                        grid[y-1, x], grid[y+1, x],
                        grid[y, x-1], grid[y, x+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    
                    if len(non_zero) >= 3:
                        # Rellenar con el valor más común
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
            # Si no hay mapeo, incrementar valores
            result[result != 0] += 1
        else:
            for old_val, new_val in mapping.items():
                result[grid == old_val] = new_val
        
        return result
    
    # === Métodos de detección de patrones ===
    
    def _is_cross_expansion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón de expansión en cruz"""
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
                    zeros_filled += 1
        
        return zeros_filled > 0
    
    def _is_gravity_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Detecta patrón de gravedad"""
        h = input_grid.shape[0]
        
        for col in range(input_grid.shape[1]):
            input_col = input_grid[:, col]
            output_col = output_grid[:, col]
            
            input_nonzero = input_col[input_col != 0]
            output_nonzero = output_col[output_col != 0]
            
            if len(input_nonzero) > 0 and len(output_nonzero) == len(input_nonzero):
                # Verificar si están al fondo
                expected_start = h - len(output_nonzero)
                if np.all(output_col[:expected_start] == 0):
                    return True
        
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