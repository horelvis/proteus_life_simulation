#!/usr/bin/env python3
"""
Observador con Atención Iterativa hasta Comprensión Total
El sistema observa la escena repetidamente, enfocándose en diferentes aspectos
hasta alcanzar comprensión completa de todas las relaciones
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttentionFocus:
    """Representa un foco de atención en la escena"""
    iteration: int
    position: Tuple[int, int]  # (y, x) centro del foco
    radius: int  # Radio del área de atención
    feature: str  # Qué está observando
    understanding: str  # Qué comprende
    confidence: float
    visual_memory: np.ndarray  # Lo que "ve" en este foco

@dataclass
class SceneUnderstanding:
    """Comprensión completa de la escena tras observación iterativa"""
    iterations_needed: int
    attention_path: List[AttentionFocus]
    global_understanding: str
    local_understandings: Dict[str, str]
    relationships_found: List[Dict]
    confidence: float
    reasoning_chain: List[str]

class IterativeAttentionObserver:
    """
    Observador que mira la escena múltiples veces
    enfocándose en diferentes áreas hasta comprenderla completamente
    """
    
    def __init__(self, max_iterations: int = 10, 
                 understanding_threshold: float = 0.9):
        self.max_iterations = max_iterations
        self.understanding_threshold = understanding_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Red de atención visual
        self.attention_network = self._build_attention_network()
        
        # Memoria visual acumulativa
        self.visual_memory = {}
        self.attention_history = []
        
        # Estado de comprensión
        self.current_understanding = 0.0
        self.reasoning_steps = []
        
    def _build_attention_network(self) -> nn.Module:
        """Construye red neuronal para atención visual"""
        class VisualAttention(nn.Module):
            def __init__(self, input_channels=10):
                super().__init__()
                # Encoder visual
                self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                
                # Mecanismo de atención
                self.attention = nn.MultiheadAttention(128, 8)
                
                # Decoder de comprensión
                self.fc1 = nn.Linear(128, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                
            def forward(self, x, previous_attention=None):
                # Procesar visualmente
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                features = torch.relu(self.conv3(x))
                
                # Aplicar atención
                b, c, h, w = features.shape
                features_flat = features.view(b, c, h*w).permute(2, 0, 1)
                
                if previous_attention is not None:
                    attended, weights = self.attention(
                        features_flat, 
                        features_flat, 
                        features_flat,
                        attn_mask=previous_attention
                    )
                else:
                    attended, weights = self.attention(
                        features_flat, 
                        features_flat, 
                        features_flat
                    )
                
                # Decodificar comprensión
                attended_pooled = attended.mean(dim=0)
                understanding = torch.relu(self.fc1(attended_pooled))
                understanding = torch.relu(self.fc2(understanding))
                understanding = self.fc3(understanding)
                
                return understanding, weights
        
        return VisualAttention().to(self.device)
    
    def observe_until_understood(self, input_grid: np.ndarray, 
                                output_grid: np.ndarray,
                                visualize: bool = True) -> SceneUnderstanding:
        """
        Observa la escena iterativamente hasta comprenderla completamente
        """
        logger.info("👁️ Iniciando observación iterativa de la escena")
        
        self.attention_history = []
        self.reasoning_steps = []
        self.current_understanding = 0.0
        
        iteration = 0
        attention_mask = None
        
        while iteration < self.max_iterations and self.current_understanding < self.understanding_threshold:
            iteration += 1
            
            # PASO 1: Decidir dónde mirar
            focus_point = self._decide_where_to_look(
                input_grid, output_grid, attention_mask
            )
            
            # PASO 2: Observar esa región con atención
            local_observation = self._observe_region(
                input_grid, output_grid, focus_point
            )
            
            # PASO 3: Procesar lo observado
            local_understanding = self._process_observation(
                local_observation, iteration
            )
            
            # PASO 4: Integrar con conocimiento previo
            self._integrate_understanding(local_understanding)
            
            # PASO 5: Actualizar comprensión global
            self.current_understanding = self._evaluate_understanding()
            
            # Registrar foco de atención
            attention_focus = AttentionFocus(
                iteration=iteration,
                position=focus_point['center'],
                radius=focus_point['radius'],
                feature=focus_point['looking_for'],
                understanding=local_understanding['description'],
                confidence=local_understanding['confidence'],
                visual_memory=local_observation['region']
            )
            
            self.attention_history.append(attention_focus)
            
            # Agregar paso de razonamiento
            reasoning = f"Iteración {iteration}: Observo {focus_point['looking_for']} en {focus_point['center']}. "
            reasoning += f"Comprendo: {local_understanding['description']} (conf: {local_understanding['confidence']:.2f})"
            self.reasoning_steps.append(reasoning)
            
            logger.info(f"  {reasoning}")
            
            # Actualizar máscara de atención para próxima iteración
            attention_mask = self._update_attention_mask(
                attention_mask, focus_point, input_grid.shape
            )
            
            # Visualizar si está habilitado
            if visualize and iteration % 2 == 0:
                self._visualize_attention_state(
                    input_grid, output_grid, 
                    self.attention_history, iteration
                )
        
        # Construir comprensión final
        scene_understanding = self._build_final_understanding()
        
        logger.info(f"✅ Comprensión alcanzada: {self.current_understanding:.2%} en {iteration} iteraciones")
        
        return scene_understanding
    
    def _decide_where_to_look(self, input_grid: np.ndarray, 
                             output_grid: np.ndarray,
                             previous_mask: Optional[np.ndarray]) -> Dict:
        """
        Decide inteligentemente dónde enfocar la atención
        basándose en lo que aún no comprende
        """
        h, w = input_grid.shape
        
        # Si es primera iteración, mirar al centro
        if previous_mask is None:
            return {
                'center': (h//2, w//2),
                'radius': min(h, w) // 4,
                'looking_for': 'estructura_general'
            }
        
        # Buscar áreas con mayor diferencia input/output
        # Manejar diferentes tamaños
        if input_grid.shape != output_grid.shape:
            # Si tienen diferente tamaño, analizar solo el input
            # y buscar patrones importantes en él
            diff = np.abs(input_grid.astype(float))
        else:
            diff = np.abs(output_grid.astype(float) - input_grid.astype(float))
        
        # Ponderar por áreas no observadas
        if previous_mask is not None:
            diff = diff * (1 - previous_mask)
        
        # Encontrar punto de máxima diferencia
        if np.max(diff) > 0:
            y, x = np.unravel_index(np.argmax(diff), diff.shape)
            looking_for = 'cambio_significativo'
        else:
            # Si no hay diferencias, buscar patrones
            # Buscar bordes o transiciones
            gradient_y = np.abs(np.diff(input_grid, axis=0))
            gradient_x = np.abs(np.diff(input_grid, axis=1))
            
            if np.max(gradient_y) > np.max(gradient_x):
                y, x = np.unravel_index(np.argmax(gradient_y), gradient_y.shape)
                looking_for = 'borde_horizontal'
            else:
                y, x = np.unravel_index(np.argmax(gradient_x), gradient_x.shape)
                looking_for = 'borde_vertical'
        
        # Determinar radio adaptativo
        radius = max(2, min(h, w) // 6)
        
        return {
            'center': (int(y), int(x)),
            'radius': radius,
            'looking_for': looking_for
        }
    
    def _observe_region(self, input_grid: np.ndarray, 
                       output_grid: np.ndarray,
                       focus: Dict) -> Dict:
        """
        Observa una región específica con atención
        """
        y, x = focus['center']
        r = focus['radius']
        h, w = input_grid.shape
        
        # Extraer región con padding si es necesario
        y_min = max(0, y - r)
        y_max = min(h, y + r + 1)
        x_min = max(0, x - r)
        x_max = min(w, x + r + 1)
        
        input_region = input_grid[y_min:y_max, x_min:x_max]
        output_region = output_grid[y_min:y_max, x_min:x_max]
        
        # Analizar la región
        # Calcular cambios solo si tienen mismo tamaño
        if input_region.shape == output_region.shape:
            changes = np.sum(input_region != output_region)
            pattern = self._detect_local_pattern(input_region, output_region)
        else:
            changes = -1  # Indicador de cambio de tamaño
            pattern = 'cambio_dimension'
        
        observation = {
            'region': input_region,
            'output_region': output_region,
            'position': (y_min, y_max, x_min, x_max),
            'values_input': np.unique(input_region),
            'values_output': np.unique(output_region),
            'changes': changes,
            'pattern': pattern
        }
        
        return observation
    
    def _detect_local_pattern(self, input_region: np.ndarray, 
                             output_region: np.ndarray) -> str:
        """
        Detecta el patrón en la región observada
        """
        if input_region.shape != output_region.shape:
            return 'cambio_dimension'
        
        if np.array_equal(input_region, output_region):
            return 'sin_cambio'
        
        # Analizar tipo de transformación
        diff = output_region - input_region
        
        if np.all(diff >= 0):
            return 'incremento'
        elif np.all(diff <= 0):
            return 'decremento'
        elif np.count_nonzero(diff) == 1:
            return 'cambio_puntual'
        elif np.count_nonzero(output_region) > np.count_nonzero(input_region):
            return 'expansion'
        elif np.count_nonzero(output_region) < np.count_nonzero(input_region):
            return 'contraccion'
        else:
            return 'transformacion_compleja'
    
    def _process_observation(self, observation: Dict, iteration: int) -> Dict:
        """
        Procesa la observación para extraer comprensión
        """
        pattern = observation['pattern']
        
        # Generar descripción de comprensión
        if pattern == 'sin_cambio':
            description = "región estable sin transformación"
            confidence = 1.0
        elif pattern == 'expansion':
            description = f"expansión de valores {observation['values_input']} a {observation['values_output']}"
            confidence = 0.9
        elif pattern == 'cambio_puntual':
            description = "cambio localizado en punto específico"
            confidence = 0.95
        elif pattern == 'transformacion_compleja':
            description = "transformación no lineal requiere más análisis"
            confidence = 0.5
        else:
            description = f"patrón {pattern} detectado"
            confidence = 0.7
        
        # Agregar contexto espacial
        pos = observation['position']
        spatial_context = f" en región ({pos[0]}-{pos[1]}, {pos[2]}-{pos[3]})"
        
        return {
            'description': description + spatial_context,
            'confidence': confidence,
            'pattern': pattern,
            'iteration': iteration
        }
    
    def _integrate_understanding(self, local_understanding: Dict):
        """
        Integra comprensión local con conocimiento acumulado
        """
        # Actualizar memoria visual
        key = f"iter_{local_understanding['iteration']}"
        self.visual_memory[key] = local_understanding
        
        # Conectar con observaciones previas
        if len(self.visual_memory) > 1:
            # Buscar relaciones entre observaciones
            self._find_relationships()
    
    def _find_relationships(self):
        """
        Encuentra relaciones entre diferentes observaciones
        """
        patterns = [mem['pattern'] for mem in self.visual_memory.values()]
        
        # Detectar patrones repetitivos
        if len(set(patterns)) == 1:
            self.reasoning_steps.append("Patrón consistente en toda la escena")
        elif len(set(patterns)) < len(patterns):
            self.reasoning_steps.append("Patrones mixtos pero con repeticiones")
        else:
            self.reasoning_steps.append("Cada región tiene transformación única")
    
    def _evaluate_understanding(self) -> float:
        """
        Evalúa el nivel de comprensión actual
        """
        if not self.visual_memory:
            return 0.0
        
        # Factores de comprensión
        confidences = [mem['confidence'] for mem in self.visual_memory.values()]
        avg_confidence = np.mean(confidences)
        
        # Cobertura espacial (cuánto de la escena hemos observado)
        coverage = min(len(self.attention_history) / 5, 1.0)  # Asumimos 5 focos para cobertura completa
        
        # Coherencia (las observaciones tienen sentido juntas)
        coherence = 0.8 if len(self.reasoning_steps) > 2 else 0.5
        
        # Comprensión total es combinación ponderada
        understanding = (avg_confidence * 0.4 + coverage * 0.3 + coherence * 0.3)
        
        return min(understanding, 1.0)
    
    def _update_attention_mask(self, current_mask: Optional[np.ndarray],
                              focus: Dict, shape: Tuple) -> np.ndarray:
        """
        Actualiza máscara de atención para recordar dónde ya miramos
        """
        if current_mask is None:
            current_mask = np.zeros(shape)
        
        y, x = focus['center']
        r = focus['radius']
        
        # Marcar región observada
        y_min = max(0, y - r)
        y_max = min(shape[0], y + r + 1)
        x_min = max(0, x - r)
        x_max = min(shape[1], x + r + 1)
        
        current_mask[y_min:y_max, x_min:x_max] = 1.0
        
        # Suavizar máscara
        from scipy.ndimage import gaussian_filter
        current_mask = gaussian_filter(current_mask, sigma=1.0)
        
        return current_mask
    
    def _visualize_attention_state(self, input_grid: np.ndarray,
                                  output_grid: np.ndarray,
                                  attention_history: List[AttentionFocus],
                                  iteration: int):
        """
        Visualiza el estado actual de atención en 2D
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Input con focos de atención
        axes[0].imshow(input_grid, cmap='tab20', interpolation='nearest')
        axes[0].set_title(f'Input - Iteración {iteration}')
        
        # Dibujar historia de atención
        for i, focus in enumerate(attention_history):
            y, x = focus.position
            circle = patches.Circle((x, y), focus.radius, 
                                   fill=False, 
                                   edgecolor='red' if i == len(attention_history)-1 else 'yellow',
                                   linewidth=2 if i == len(attention_history)-1 else 1,
                                   alpha=0.8 if i == len(attention_history)-1 else 0.3)
            axes[0].add_patch(circle)
            
            # Número de iteración
            axes[0].text(x, y, str(focus.iteration), 
                        color='white', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Panel 2: Output esperado
        axes[1].imshow(output_grid, cmap='tab20', interpolation='nearest')
        axes[1].set_title('Output esperado')
        
        # Panel 3: Mapa de comprensión
        understanding_map = np.zeros_like(input_grid, dtype=float)
        for focus in attention_history:
            y, x = focus.position
            r = focus.radius
            y_min = max(0, y - r)
            y_max = min(input_grid.shape[0], y + r + 1)
            x_min = max(0, x - r)
            x_max = min(input_grid.shape[1], x + r + 1)
            
            understanding_map[y_min:y_max, x_min:x_max] += focus.confidence
        
        im = axes[2].imshow(understanding_map, cmap='YlOrRd', interpolation='nearest')
        axes[2].set_title(f'Mapa de Comprensión ({self.current_understanding:.1%})')
        plt.colorbar(im, ax=axes[2])
        
        # Agregar texto con razonamiento actual
        if self.reasoning_steps:
            fig.text(0.5, 0.02, f"Último: {self.reasoning_steps[-1][:100]}...", 
                    ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout()
        plt.savefig(f'/tmp/attention_iter_{iteration}.png', dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()
        
        logger.info(f"  📸 Visualización guardada: /tmp/attention_iter_{iteration}.png")
    
    def _build_final_understanding(self) -> SceneUnderstanding:
        """
        Construye la comprensión final de la escena
        """
        # Extraer relaciones encontradas
        relationships = []
        for i in range(len(self.attention_history) - 1):
            curr = self.attention_history[i]
            next = self.attention_history[i + 1]
            
            rel = {
                'from': curr.position,
                'to': next.position,
                'connection': f"{curr.feature} → {next.feature}"
            }
            relationships.append(rel)
        
        # Comprensión global
        if self.current_understanding >= self.understanding_threshold:
            global_understanding = "Escena completamente comprendida: transformación sistemática con patrones locales identificados"
        else:
            global_understanding = f"Comprensión parcial ({self.current_understanding:.1%}): requiere más observaciones"
        
        # Comprensiones locales
        local_understandings = {
            f"región_{i}": focus.understanding 
            for i, focus in enumerate(self.attention_history)
        }
        
        return SceneUnderstanding(
            iterations_needed=len(self.attention_history),
            attention_path=self.attention_history,
            global_understanding=global_understanding,
            local_understandings=local_understandings,
            relationships_found=relationships,
            confidence=self.current_understanding,
            reasoning_chain=self.reasoning_steps
        )