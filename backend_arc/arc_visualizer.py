"""
ARC Visualizer
Genera visualizaciones del proceso de razonamiento
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

logger = logging.getLogger(__name__)

class ARCVisualizer:
    def __init__(self):
        self.cell_size = 30
        self.padding = 10
        self.colors = [
            '#000000',  # 0 - Negro (fondo)
            '#0074D9',  # 1 - Azul
            '#2ECC40',  # 2 - Verde  
            '#FFDC00',  # 3 - Amarillo
            '#FF4136',  # 4 - Rojo
            '#B10DC9',  # 5 - Púrpura
            '#FF851B',  # 6 - Naranja
            '#7FDBFF',  # 7 - Celeste
            '#85144b',  # 8 - Marrón
            '#F012BE',  # 9 - Rosa
        ]
        
    def create_step_visualization(self, step: Dict[str, Any]) -> str:
        """Crea visualización de un paso de razonamiento"""
        if step['type'] == 'analysis':
            return self._create_text_visualization(
                step['description'],
                step.get('details', '')
            )
        elif step['type'] == 'rule_detection':
            return self._create_rule_visualization(step)
        elif step['type'] == 'transformation':
            return self._create_transformation_visualization(step)
        else:
            return self._create_generic_visualization(step)
            
    def create_grid_image(self, grid: np.ndarray, title: str = "", highlight_cells: List[Tuple[int, int]] = None) -> Image:
        """Crea una imagen PIL de un grid"""
        h, w = grid.shape
        img_width = w * self.cell_size + 2 * self.padding
        img_height = h * self.cell_size + 2 * self.padding + 30  # Espacio para título
        
        # Crear imagen
        img = Image.new('RGB', (img_width, img_height), color='#1a1a1a')
        draw = ImageDraw.Draw(img)
        
        # Dibujar título
        if title:
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            draw.text((self.padding, 5), title, fill='white', font=font)
        
        # Dibujar grid
        for i in range(h):
            for j in range(w):
                x = self.padding + j * self.cell_size
                y = self.padding + 30 + i * self.cell_size
                
                # Color de la celda
                color = self.colors[grid[i, j] % len(self.colors)]
                
                # Dibujar celda
                draw.rectangle(
                    [x, y, x + self.cell_size - 1, y + self.cell_size - 1],
                    fill=color,
                    outline='#333333'
                )
                
                # Resaltar si es necesario
                if highlight_cells and (i, j) in highlight_cells:
                    draw.rectangle(
                        [x-2, y-2, x + self.cell_size + 1, y + self.cell_size + 1],
                        outline='#FF851B',
                        width=3
                    )
                    
        return img
        
    def create_comparison_image(self, before: np.ndarray, after: np.ndarray, 
                              title: str = "Transformación") -> str:
        """Crea imagen comparando antes y después"""
        # Crear imágenes individuales
        img_before = self.create_grid_image(before, "Antes")
        img_after = self.create_grid_image(after, "Después")
        
        # Combinar horizontalmente
        total_width = img_before.width + img_after.width + 20
        total_height = max(img_before.height, img_after.height)
        
        combined = Image.new('RGB', (total_width, total_height), color='#1a1a1a')
        combined.paste(img_before, (0, 0))
        combined.paste(img_after, (img_before.width + 20, 0))
        
        # Dibujar flecha
        draw = ImageDraw.Draw(combined)
        arrow_x = img_before.width + 10
        arrow_y = total_height // 2
        draw.text((arrow_x - 5, arrow_y - 10), "→", fill='white', font=ImageFont.load_default())
        
        # Convertir a base64
        return self._image_to_base64(combined)
        
    def create_reasoning_diagram(self, reasoning_steps: List[Dict]) -> str:
        """Crea diagrama de flujo del razonamiento"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(reasoning_steps) + 1)
        ax.axis('off')
        
        # Título
        ax.text(5, len(reasoning_steps) + 0.5, 'Flujo de Razonamiento ARC', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Dibujar cada paso
        for i, step in enumerate(reasoning_steps):
            y = len(reasoning_steps) - i
            
            # Caja del paso
            if step['type'] == 'analysis':
                color = '#0074D9'
            elif step['type'] == 'rule_detection':
                color = '#2ECC40'
            elif step['type'] == 'transformation':
                color = '#FF851B'
            else:
                color = '#B10DC9'
                
            rect = patches.FancyBboxPatch(
                (1, y - 0.4), 8, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='white',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Texto del paso
            ax.text(5, y, step.get('description', 'Paso'), 
                   ha='center', va='center', fontsize=12, color='white')
            
            # Flecha al siguiente paso
            if i < len(reasoning_steps) - 1:
                ax.arrow(5, y - 0.5, 0, -0.4, 
                        head_width=0.2, head_length=0.1, 
                        fc='white', ec='white')
                
        # Guardar como imagen
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode()
        
    def create_animated_gif(self, reasoning_steps: List[Dict]) -> str:
        """Crea GIF animado del proceso de razonamiento"""
        # Por ahora devolvemos un placeholder
        # En producción, usaríamos imageio o similar
        return "animated_gif_placeholder"
        
    def _create_text_visualization(self, title: str, details: str) -> str:
        """Crea visualización de texto"""
        img = Image.new('RGB', (600, 150), color='#1a1a1a')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("Arial.ttf", 18)
            font_details = ImageFont.truetype("Arial.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_details = ImageFont.load_default()
            
        draw.text((20, 20), title, fill='white', font=font_title)
        draw.text((20, 60), details, fill='#aaaaaa', font=font_details)
        
        return self._image_to_base64(img)
        
    def _create_rule_visualization(self, step: Dict) -> str:
        """Crea visualización de detección de regla"""
        rule = step.get('rule', {})
        
        # Crear comparación input/output
        if 'input' in step and 'output' in step:
            input_grid = np.array(step['input'])
            output_grid = np.array(step['output'])
            
            title = f"Regla: {rule.get('type', 'Desconocida')}"
            return self.create_comparison_image(input_grid, output_grid, title)
        else:
            return self._create_text_visualization(
                step['description'],
                f"Tipo: {rule.get('type', 'Desconocida')}\nConfianza: {rule.get('confidence', 0):.2f}"
            )
            
    def _create_transformation_visualization(self, step: Dict) -> str:
        """Crea visualización de transformación"""
        # Por ahora, visualización simple de texto
        positions = step.get('positions', [])
        count = step.get('count', 0)
        
        details = f"Posiciones afectadas: {count}"
        return self._create_text_visualization(step['description'], details)
        
    def _create_generic_visualization(self, step: Dict) -> str:
        """Crea visualización genérica"""
        return self._create_text_visualization(
            step.get('description', 'Paso'),
            str(step.get('details', ''))
        )
        
    def _image_to_base64(self, img: Image) -> str:
        """Convierte imagen PIL a base64"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
        
    def export_reasoning_as_png(self, reasoning_steps: List[Dict], filename: str):
        """Exporta el razonamiento completo como PNG"""
        diagram = self.create_reasoning_diagram(reasoning_steps)
        
        # Decodificar base64 y guardar
        img_data = base64.b64decode(diagram)
        with open(filename, 'wb') as f:
            f.write(img_data)
            
    def create_puzzle_summary(self, puzzle_id: str, is_correct: bool, 
                            confidence: float, rule_type: str) -> str:
        """Crea resumen visual del puzzle"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Fondo
        fig.patch.set_facecolor('#1a1a1a')
        
        # Título
        title_color = '#2ECC40' if is_correct else '#FF4136'
        ax.text(0.5, 0.9, f'Puzzle {puzzle_id}', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=24, color=title_color, fontweight='bold')
        
        # Estado
        status = '✅ Resuelto' if is_correct else '❌ Incorrecto'
        ax.text(0.5, 0.7, status,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=20, color='white')
        
        # Detalles
        details = f"Regla: {rule_type}\nConfianza: {confidence:.1%}"
        ax.text(0.5, 0.4, details,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, color='#aaaaaa')
        
        # Guardar
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode()