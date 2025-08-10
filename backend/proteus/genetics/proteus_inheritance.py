"""
PROTEUS Tri-Layer Inheritance System
Layer 1: Topological Core (DNA-like)
Layer 2: Holographic Memory (Epigenetic-like) 
Layer 3: Environmental Traces (Culture-like)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

from .holographic_memory import HolographicMemory

@dataclass
class TopologicalCore:
    """Immutable core parameters - the 'DNA' of PROTEUS"""
    # Fundamental topology
    manifold_dimension: float = 2.5
    curvature_tensor: List[float] = field(default_factory=lambda: [0.0] * 4)
    betti_numbers: List[int] = field(default_factory=lambda: [1, 2, 0])
    fundamental_group: str = 'Z2'
    
    # Morphology
    body_symmetry: int = 3
    organ_capacity: int = 3
    
    # Base traits
    base_motility: float = 0.5
    base_sensitivity: float = 0.5
    base_resilience: float = 0.5
    
    # Evolution
    mutability: float = 0.10  # Base rate, will be randomized on creation
    
    def mutate(self, rate: float = None) -> 'TopologicalCore':
        """Create mutated copy"""
        if rate is None:
            rate = self.mutability
            
        new_core = TopologicalCore()
        
        # Copy with mutations
        for attr, value in self.__dict__.items():
            if np.random.random() < rate:
                new_value = self._mutate_parameter(attr, value)
                setattr(new_core, attr, new_value)
            else:
                setattr(new_core, attr, value)
                
        return new_core
        
    def _mutate_parameter(self, name: str, value):
        """Mutate a single parameter"""
        if isinstance(value, float):
            # Gaussian mutation
            return np.clip(value + np.random.randn() * 0.1, 0.1, 1.0)
        elif isinstance(value, int):
            # Integer mutation
            return max(1, value + np.random.choice([-1, 0, 1]))
        elif isinstance(value, list):
            # List mutation
            new_list = value.copy()
            if len(new_list) > 0:
                idx = np.random.randint(len(new_list))
                if isinstance(new_list[idx], float):
                    new_list[idx] += np.random.randn() * 0.1
                elif isinstance(new_list[idx], int):
                    new_list[idx] = max(0, new_list[idx] + np.random.choice([-1, 0, 1]))
            return new_list
        else:
            return value
            
    def crossover(self, other: 'TopologicalCore') -> 'TopologicalCore':
        """Mendelian inheritance with another core"""
        child = TopologicalCore()
        
        for attr in self.__dict__:
            if np.random.random() < 0.5:
                setattr(child, attr, getattr(self, attr))
            else:
                setattr(child, attr, getattr(other, attr))
                
        return child


@dataclass
class EnvironmentalTrace:
    """Information deposited in the world"""
    position: Tuple[float, float]
    timestamp: float
    type: str
    intensity: float
    pheromone: np.ndarray
    decay: float = 0.99
    

class ProteusInheritance:
    """Complete tri-layer inheritance system"""
    
    def __init__(self, parent1=None, parent2=None):
        # Layer 1: Topological Core
        if parent1 and parent2:
            # Sexual reproduction
            self.core = parent1.core.crossover(parent2.core).mutate()
        elif parent1:
            # Asexual reproduction
            self.core = parent1.core.mutate()
        else:
            # Genesis
            self.core = TopologicalCore()
            # Set random mutability between 5-15% to match frontend
            self.core.mutability = 0.05 + np.random.random() * 0.10
            
        # Layer 2: Holographic Memory
        self.memory = HolographicMemory(
            2000,
            parent1.memory if parent1 else None,
            parent2.memory if parent2 else None
        )
        
        # Layer 3: Environmental Traces
        self.environmental_traces = []
        self.pheromone_signature = self._generate_pheromone_signature()
        self.generation = (parent1.generation + 1) if parent1 else 0
        
        # Learn from parent traces
        if parent1:
            self._learn_from_traces(parent1.environmental_traces)
        if parent2:
            self._learn_from_traces(parent2.environmental_traces)
            
        # Current phenotype expression
        self._phenotype = None
        self._update_phenotype()
        
    def _generate_pheromone_signature(self) -> np.ndarray:
        """Generate unique chemical signature"""
        signature = np.zeros(10, dtype=np.float32)
        
        # Core contribution
        signature[0] = self.core.manifold_dimension / 3.0
        signature[1] = self.core.body_symmetry / 6.0
        signature[2] = self.core.base_motility
        signature[3] = self.core.base_sensitivity
        
        # Memory contribution
        memory_sig = self.memory.compress()['signature'][:6]
        signature[4:4+len(memory_sig)] = memory_sig
        
        return signature
        
    def _learn_from_traces(self, traces: List[EnvironmentalTrace]):
        """Learn from environmental traces"""
        if not traces:
            return
            
        # Convert traces to experiences
        for trace in traces[-10:]:  # Recent traces only
            if trace.decay > 0.5:  # Still strong
                experience = {
                    'type': 'environmental_learning',
                    'position': trace.position,
                    'importance': trace.intensity * trace.decay * 0.5,
                    'trajectory': [{'x': trace.position[0], 'y': trace.position[1]}]
                }
                self.memory.encode(experience)
                
    def experience(self, event: Dict):
        """Process an experience"""
        # Encode in holographic memory
        self.memory.encode(event)
        
        # Create environmental trace
        trace = EnvironmentalTrace(
            position=event.get('position', (0, 0)),
            timestamp=event.get('timestamp', 0),
            type=event.get('type', 'unknown'),
            intensity=event.get('importance', 0.5),
            pheromone=self._generate_event_pheromone(event)
        )
        
        self.environmental_traces.append(trace)
        
        # Limit trace history
        if len(self.environmental_traces) > 100:
            self.environmental_traces.pop(0)
            
        # Update phenotype
        self._update_phenotype()
        
    def _generate_event_pheromone(self, event: Dict) -> np.ndarray:
        """Generate pheromone for specific event"""
        pheromone = np.zeros(5, dtype=np.float32)
        
        event_type = event.get('type', '')
        if 'predator' in event_type:
            pheromone[0] = 1.0  # Danger
        elif 'food' in event_type:
            pheromone[1] = 1.0  # Food
        elif 'reproduction' in event_type:
            pheromone[2] = 1.0  # Mating
        elif 'death' in event_type:
            pheromone[3] = 1.0  # Death
        else:
            pheromone[4] = 0.5  # General
            
        return pheromone
        
    def _update_phenotype(self):
        """Express phenotype from core + memory"""
        # Get memory influence
        context = {'danger': 0, 'hunger': 0.5, 'energy': 1.0}
        memory_behavior = self.memory.recall(context)
        
        # Combine with core
        self._phenotype = {
            # Morphology from core
            'symmetry': self.core.body_symmetry,
            'max_organs': self.core.organ_capacity,
            
            # Behavior from core + memory
            'motility': self.core.base_motility * (1 + memory_behavior['exploration_tendency'] * 0.5),
            'sensitivity': self.core.base_sensitivity * (1 + memory_behavior['caution_level'] * 0.3),
            'resilience': self.core.base_resilience,
            
            # Pure memory behaviors
            'curiosity': memory_behavior['exploration_tendency'],
            'fearfulness': memory_behavior['caution_level'],
            'foraging': memory_behavior['forage_intensity'],
            'sociability': memory_behavior['social_attraction'],
            
            # Organ expressions from inherited genes - REAL EVOLUTION
            'organ_expressions': {
                # Sensory organs from sensitivity gene
                'photosensor': self.core.base_sensitivity * (1 + memory_behavior['caution_level'] * 0.3),
                'chemoreceptor': self.core.base_sensitivity * (1 + memory_behavior['forage_intensity'] * 0.3),
                
                # Movement organs from motility gene
                'flagellum': self.core.base_motility,
                'speed_boost': self.core.base_motility if self.core.base_motility > 0.7 else 0,
                
                # Defense organs from resilience gene
                'membrane': self.core.base_resilience,
                'armor_plates': (self.core.base_resilience - 0.4) * 2 if self.core.base_resilience > 0.6 else 0,
                'toxin_gland': (self.core.base_resilience - 0.5) * 2 if self.core.base_resilience > 0.7 else 0,
                
                # Special organs from trait combinations
                'electric_organ': (self.core.base_motility + self.core.base_resilience - 1.0) 
                    if (self.core.base_motility > 0.6 and self.core.base_resilience > 0.6) else 0,
                    
                'regeneration': (self.core.base_resilience - 0.6) * 2 if self.core.base_resilience > 0.8 else 0,
                
                'camouflage': (self.core.base_sensitivity + self.core.base_resilience - 0.8) * 0.5
                    if (self.core.base_sensitivity > 0.5 and self.core.base_resilience > 0.5) else 0,
                    
                'vacuole': 0.3 + self.core.base_resilience * 0.4,
                
                'pheromone_emitter': self.core.base_sensitivity * 0.5 if self.core.base_sensitivity > 0.6 else 0
            }
        }
        
    def expressPhenotype(self) -> Dict:
        """Get current phenotype expression"""
        if self._phenotype is None:
            self._update_phenotype()
        return self._phenotype
        
    def reproduce(self, mate=None) -> 'ProteusInheritance':
        """Create offspring"""
        return ProteusInheritance(self, mate)
        
    def get_memory_size(self) -> Dict[str, int]:
        """Calculate memory usage"""
        return {
            'core': 200,  # bytes
            'holographic': self.memory.size * 4,  # float32
            'environmental': len(self.environmental_traces) * 100,  # approx
            'total': 200 + self.memory.size * 4 + len(self.environmental_traces) * 100
        }
        
    def compress(self) -> Dict:
        """Compress for storage/transmission"""
        return {
            'generation': self.generation,
            'core': {
                'manifold_dimension': self.core.manifold_dimension,
                'body_symmetry': self.core.body_symmetry,
                'base_traits': {
                    'motility': self.core.base_motility,
                    'sensitivity': self.core.base_sensitivity,
                    'resilience': self.core.base_resilience
                }
            },
            'memory': self.memory.compress(),
            'pheromone': self.pheromone_signature.tolist(),
            'recent_traces': len(self.environmental_traces)
        }
        
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.compress())
        
    @classmethod
    def from_compressed(cls, data: Dict) -> 'ProteusInheritance':
        """Create from compressed data"""
        inheritance = cls()
        
        # Restore core
        inheritance.core.manifold_dimension = data['core']['manifold_dimension']
        inheritance.core.body_symmetry = data['core']['body_symmetry']
        inheritance.core.base_motility = data['core']['base_traits']['motility']
        inheritance.core.base_sensitivity = data['core']['base_traits']['sensitivity']
        inheritance.core.base_resilience = data['core']['base_traits']['resilience']
        
        # Generation
        inheritance.generation = data['generation']
        
        # Pheromone
        inheritance.pheromone_signature = np.array(data['pheromone'], dtype=np.float32)
        
        return inheritance