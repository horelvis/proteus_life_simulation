"""
Sistema de Órganos Topológicos - Inspirado en el prototipo JS
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class TopologicalOrgan:
    """
    Un órgano topológico que emerge y se desarrolla gradualmente
    """
    type: str
    expression: float  # 0-1, qué tan expresado está
    development_stage: float  # 0-1, qué tan desarrollado está
    cost: float  # Costo energético
    functionality: float  # 0-1, qué tan funcional es
    
    # Tipos posibles de órganos
    ORGAN_TYPES = [
        'photosensor',    # Detecta luz (inicio de visión)
        'chemoreceptor',  # Detecta químicos
        'flagellum',      # Movimiento direccional
        'cilia',          # Sensores táctiles
        'membrane',       # Protección
        'vacuole',        # Almacenamiento de energía
        'pseudopod',      # Extensiones para movimiento
        'crystallin',     # Estructura pre-óptica
        'pigment_spot',   # Mancha ocular primitiva
        'nerve_net'       # Red de señalización primitiva
    ]
    
    @classmethod
    def mutate_new_organ(cls) -> 'TopologicalOrgan':
        """Crea un nuevo órgano por mutación"""
        organ_type = random.choice(cls.ORGAN_TYPES)
        return cls(
            type=organ_type,
            expression=0.1,  # Empieza débil
            development_stage=0.0,
            cost=random.uniform(0.1, 0.5),
            functionality=0.0
        )
    
    @classmethod
    def inherit_organ(cls, parent_organ: 'TopologicalOrgan') -> 'TopologicalOrgan':
        """Hereda un órgano con variación"""
        # Variación en la expresión
        new_expression = np.clip(
            parent_organ.expression + np.random.normal(0, 0.1),
            0, 1
        )
        
        return cls(
            type=parent_organ.type,
            expression=new_expression,
            development_stage=0.0,  # Cada generación empieza a desarrollar desde cero
            cost=parent_organ.cost * np.random.uniform(0.9, 1.1),
            functionality=0.0
        )
    
    def develop(self, environment: Dict, energy: float) -> bool:
        """
        Desarrolla el órgano gradualmente
        Retorna True si se desarrolló, False si no
        """
        if self.expression > 0 and energy > self.cost * 10:
            # Desarrollo gradual
            self.development_stage = min(1.0, 
                self.development_stage + 0.01 * self.expression)
            
            # Funcionalidad depende del desarrollo y el ambiente
            self.functionality = self.calculate_functionality(environment)
            
            return True
        return False
    
    def calculate_functionality(self, environment: Dict) -> float:
        """Calcula qué tan funcional es el órgano en el ambiente actual"""
        base = self.development_stage * self.expression
        
        # Funcionalidad específica por tipo
        if self.type == 'photosensor':
            # Más útil con más luz
            light_level = environment.get('light_intensity', 0.5)
            return base * (1 + light_level)
            
        elif self.type == 'flagellum':
            # Más útil en ambientes menos viscosos
            viscosity = environment.get('viscosity', 0.8)
            return base * (2 - viscosity)
            
        elif self.type == 'membrane':
            # Siempre útil para protección
            return base * 1.5
            
        elif self.type == 'chemoreceptor':
            # Útil si hay gradientes químicos
            chemical_gradient = environment.get('chemical_gradient', 0.3)
            return base * (1 + chemical_gradient)
            
        elif self.type == 'pigment_spot':
            # Evolución de photosensor
            light_level = environment.get('light_intensity', 0.5)
            return base * (1 + light_level * 2)
            
        else:
            return base


class OrganSystem:
    """
    Sistema que gestiona los órganos de un organismo
    """
    
    def __init__(self, inherited_organs: Optional[List[TopologicalOrgan]] = None):
        self.organs: List[TopologicalOrgan] = []
        self.capabilities: Dict[str, float] = {
            'vision': 0.0,      # Capacidad visual
            'chemotaxis': 0.0,  # Detección química
            'motility': 0.1,    # Capacidad de movimiento base
            'protection': 0.0,  # Protección
            'efficiency': 1.0   # Eficiencia metabólica
        }
        
        if inherited_organs:
            # Heredar órganos con posible mutación
            for parent_organ in inherited_organs:
                if random.random() < 0.01:  # 1% de nuevo órgano
                    self.organs.append(TopologicalOrgan.mutate_new_organ())
                else:
                    self.organs.append(TopologicalOrgan.inherit_organ(parent_organ))
        
    def develop_organs(self, environment: Dict, available_energy: float) -> float:
        """
        Desarrolla todos los órganos y retorna el costo energético
        """
        total_cost = 0
        
        for organ in self.organs:
            if organ.develop(environment, available_energy):
                cost = organ.cost * organ.functionality
                total_cost += cost
                self.update_capabilities(organ)
        
        return total_cost
    
    def update_capabilities(self, organ: TopologicalOrgan):
        """Actualiza las capacidades basándose en el órgano"""
        
        # Sistema de visión evolutivo
        if organ.type == 'photosensor':
            self.capabilities['vision'] = max(
                self.capabilities['vision'], 
                organ.functionality * 0.3
            )
        elif organ.type == 'pigment_spot':
            self.capabilities['vision'] = max(
                self.capabilities['vision'], 
                organ.functionality * 0.6
            )
        elif organ.type == 'crystallin':
            # Permite formación de lente primitivo
            self.capabilities['vision'] = min(
                1.0, 
                self.capabilities['vision'] + organ.functionality * 0.2
            )
        
        # Sistema químico
        elif organ.type == 'chemoreceptor':
            self.capabilities['chemotaxis'] = max(
                self.capabilities['chemotaxis'],
                organ.functionality * 0.8
            )
        
        # Sistema de movimiento
        elif organ.type == 'flagellum':
            self.capabilities['motility'] += organ.functionality * 0.5
        elif organ.type == 'cilia':
            self.capabilities['motility'] += organ.functionality * 0.3
        elif organ.type == 'pseudopod':
            self.capabilities['motility'] += organ.functionality * 0.4
        
        # Sistema de protección
        elif organ.type == 'membrane':
            self.capabilities['protection'] = max(
                self.capabilities['protection'],
                organ.functionality * 0.7
            )
        
        # Sistema metabólico
        elif organ.type == 'vacuole':
            self.capabilities['efficiency'] *= (1 + organ.functionality * 0.2)
    
    def get_movement_modulation(self, environment: Dict) -> Tuple[float, float]:
        """
        Calcula la modulación del movimiento basada en órganos
        """
        dx, dy = 0.0, 0.0
        
        # Visión: evitar luz si es peligrosa
        if self.capabilities['vision'] > 0:
            light_gradient = environment.get('light_gradient', (0, 0))
            
            if self.capabilities['vision'] < 0.3:
                # Visión primitiva: solo detecta presencia
                if environment.get('light_intensity', 0) > 0.5:
                    dx -= np.random.uniform(-0.5, 0.5)
                    dy -= np.random.uniform(-0.5, 0.5)
            else:
                # Visión direccional
                dx -= light_gradient[0] * self.capabilities['vision']
                dy -= light_gradient[1] * self.capabilities['vision']
        
        # Quimiotaxis: ir hacia nutrientes
        if self.capabilities['chemotaxis'] > 0:
            chemical_gradient = environment.get('chemical_gradient', (0, 0))
            dx += chemical_gradient[0] * self.capabilities['chemotaxis']
            dy += chemical_gradient[1] * self.capabilities['chemotaxis']
        
        return dx, dy
    
    def get_total_capabilities(self) -> Dict[str, float]:
        """Retorna las capacidades totales del organismo"""
        return self.capabilities.copy()
    
    def describe_phenotype(self) -> str:
        """Describe el fenotipo emergente"""
        active_organs = [o for o in self.organs if o.functionality > 0.1]
        
        if not active_organs:
            return "Primitive form"
        
        # Determinar el fenotipo dominante
        vision_level = self.capabilities['vision']
        motility_level = self.capabilities['motility']
        
        if vision_level > 0.7:
            return "Visual hunter"
        elif vision_level > 0.3:
            return "Light-sensitive swimmer"
        elif self.capabilities['chemotaxis'] > 0.5:
            return "Chemical tracker"
        elif motility_level > 0.5:
            return "Fast swimmer"
        elif self.capabilities['protection'] > 0.5:
            return "Armored grazer"
        else:
            return "Basic survivor"