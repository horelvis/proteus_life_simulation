"""
Holographic Memory System for PROTEUS
Each point contains information about the whole experience
"""

import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
import json

class HolographicMemory:
    """
    Holographic memory that stores experiences as interference patterns
    Robust to damage and allows experience inheritance
    """
    
    def __init__(self, size: int = 1000, parent1=None, parent2=None):
        self.size = size
        
        if parent1 and parent2:
            # Inherit through interference
            self.interference_pattern = self._create_interference(parent1, parent2)
            self.phase_relations = self._blend_phases(
                parent1.phase_relations, 
                parent2.phase_relations
            )
        elif parent1:
            # Single parent - mutate
            self.interference_pattern = self._mutate_pattern(parent1.interference_pattern)
            self.phase_relations = self._mutate_pattern(parent1.phase_relations)
        else:
            # Genesis
            self.interference_pattern = np.random.randn(size).astype(np.float32)
            self.phase_relations = np.random.random(size).astype(np.float32) * 2 * np.pi
            
        # Compressed history
        self.critical_moments = []
        self.memory_strength = 1.0
        
    @staticmethod
    @jit(nopython=True)
    def _create_interference_jit(p1, p2, phase1, phase2):
        """JIT-compiled interference creation"""
        size = len(p1)
        pattern = np.zeros(size, dtype=np.float32)
        
        for i in range(size):
            # Wave interference
            pattern[i] = (p1[i] * np.cos(phase1[i]) + 
                         p2[i] * np.cos(phase2[i]) + 
                         2 * np.sqrt(np.abs(p1[i] * p2[i])) * 
                         np.cos(phase1[i] - phase2[i]))
            # Normalize
            pattern[i] = np.tanh(pattern[i] * 0.5)
            
        return pattern
        
    def _create_interference(self, parent1, parent2):
        """Create interference pattern from parents"""
        return self._create_interference_jit(
            parent1.interference_pattern,
            parent2.interference_pattern,
            parent1.phase_relations,
            parent2.phase_relations
        )
        
    @staticmethod
    @jit(nopython=True)
    def _blend_phases_jit(phases1, phases2):
        """JIT-compiled phase blending"""
        size = len(phases1)
        blended = np.zeros(size, dtype=np.float32)
        
        for i in range(size):
            x = np.cos(phases1[i]) + np.cos(phases2[i])
            y = np.sin(phases1[i]) + np.sin(phases2[i])
            blended[i] = np.arctan2(y, x)
            
        return blended
        
    def _blend_phases(self, phases1, phases2):
        """Blend phase relations from parents"""
        return self._blend_phases_jit(phases1, phases2)
        
    @staticmethod
    @jit(nopython=True)
    def _mutate_pattern_jit(pattern, rate):
        """JIT-compiled pattern mutation"""
        mutated = pattern.copy()
        size = len(pattern)
        
        for i in range(size):
            if np.random.random() < rate:
                mutated[i] += (np.random.random() - 0.5) * 0.2
                mutated[i] = np.tanh(mutated[i])
                
        return mutated
        
    def _mutate_pattern(self, pattern, rate=0.1):
        """Mutate a pattern"""
        return self._mutate_pattern_jit(pattern, rate)
        
    def encode(self, experience: Dict):
        """Encode experience into holographic memory"""
        # Extract features
        importance = experience.get('importance', 0.5)
        trajectory = experience.get('trajectory', [])
        
        # Extract frequencies (simplified)
        frequencies = self._extract_frequencies(trajectory)
        
        # Update interference pattern
        self._update_pattern(frequencies, importance)
        
        # Store critical moments
        if importance > 0.7:
            self.critical_moments.append({
                'timestamp': experience.get('timestamp', 0),
                'topology': self._extract_topology(experience),
                'impact': importance,
                'type': experience.get('type', 'unknown')
            })
            
            # Keep only recent
            if len(self.critical_moments) > 20:
                self.critical_moments.pop(0)
                
    @staticmethod
    @jit(nopython=True)
    def _extract_frequencies_jit(trajectory_x, trajectory_y):
        """Extract frequency components from trajectory"""
        if len(trajectory_x) == 0:
            return np.zeros(10, dtype=np.float32)
            
        frequencies = np.zeros(10, dtype=np.float32)
        n = len(trajectory_x)
        
        for f in range(10):
            sum_val = 0.0
            for i in range(n):
                magnitude = np.sqrt(trajectory_x[i]**2 + trajectory_y[i]**2)
                sum_val += magnitude * np.cos(2 * np.pi * f * i / n)
            frequencies[f] = sum_val / n
            
        return frequencies
        
    def _extract_frequencies(self, trajectory):
        """Extract frequencies from trajectory"""
        if not trajectory:
            return np.zeros(10, dtype=np.float32)
            
        # Convert to numpy arrays
        x = np.array([p['x'] for p in trajectory], dtype=np.float32)
        y = np.array([p['y'] for p in trajectory], dtype=np.float32)
        
        return self._extract_frequencies_jit(x, y)
        
    @jit(nopython=True)
    def _update_pattern_jit(pattern, phases, frequencies, learning_rate):
        """JIT-compiled pattern update"""
        size = len(pattern)
        
        for i in range(size):
            contribution = 0.0
            for f in range(len(frequencies)):
                contribution += frequencies[f] * np.cos(f * i * 0.01 + phases[i])
                
            pattern[i] = ((1 - learning_rate) * pattern[i] + 
                         learning_rate * np.tanh(contribution))
                         
    def _update_pattern(self, frequencies, importance):
        """Update interference pattern with new experience"""
        learning_rate = 0.1 * importance
        self._update_pattern_jit(
            self.interference_pattern,
            self.phase_relations,
            frequencies,
            learning_rate
        )
        
    def _extract_topology(self, experience):
        """Extract topological invariants"""
        trajectory = experience.get('trajectory', [])
        
        return {
            'curvature': self._calculate_curvature(trajectory),
            'winding': self._calculate_winding(trajectory),
            'persistence': len(trajectory),
            'danger': 1.0 / experience.get('predator_distance', 1000),
            'attraction': experience.get('nutrient_direction', 0)
        }
        
    def _calculate_curvature(self, trajectory):
        """Calculate average path curvature"""
        if len(trajectory) < 3:
            return 0.0
            
        total_curvature = 0.0
        for i in range(1, len(trajectory) - 1):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            p3 = trajectory[i+1]
            
            v1 = (p2['x'] - p1['x'], p2['y'] - p1['y'])
            v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
            
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            
            total_curvature += np.arctan2(cross, dot)
            
        return total_curvature / (len(trajectory) - 2)
        
    def _calculate_winding(self, trajectory):
        """Calculate winding number"""
        if len(trajectory) < 2:
            return 0.0
            
        total_angle = 0.0
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            
            angle1 = np.arctan2(p1['y'], p1['x'])
            angle2 = np.arctan2(p2['y'], p2['x'])
            
            d_angle = angle2 - angle1
            while d_angle > np.pi:
                d_angle -= 2 * np.pi
            while d_angle < -np.pi:
                d_angle += 2 * np.pi
                
            total_angle += d_angle
            
        return total_angle / (2 * np.pi)
        
    def recall(self, context: Dict) -> Dict:
        """Recall behavior from memory given context"""
        danger = context.get('danger', 0)
        hunger = context.get('hunger', 0)
        energy = context.get('energy', 1)
        
        # Extract behavioral tendencies from pattern
        behavior = self._extract_behavior()
        
        # Modulate by context
        if danger > 0.5:
            behavior['caution_level'] *= 2
            behavior['exploration_tendency'] *= 0.5
            
        if hunger > 0.7:
            behavior['forage_intensity'] *= 2
            
        return behavior
        
    @staticmethod
    @jit(nopython=True)
    def _extract_behavior_jit(pattern):
        """Extract behaviors from interference pattern"""
        # Different regions encode different behaviors
        exploration = np.tanh(np.mean(pattern[0:250]))
        caution = np.tanh(np.mean(pattern[250:500]))
        foraging = np.tanh(np.mean(pattern[500:750]))
        social = np.tanh(np.mean(pattern[750:1000]))
        
        return exploration, caution, foraging, social
        
    def _extract_behavior(self):
        """Extract behavioral tendencies"""
        exp, caut, forage, social = self._extract_behavior_jit(
            self.interference_pattern
        )
        
        return {
            'exploration_tendency': exp,
            'caution_level': caut,
            'forage_intensity': forage,
            'social_attraction': social
        }
        
    def similarity(self, other) -> float:
        """Calculate similarity with another memory"""
        return np.corrcoef(
            self.interference_pattern, 
            other.interference_pattern
        )[0, 1]
        
    def compress(self) -> Dict:
        """Compress memory for storage"""
        # Extract dominant frequencies
        fft = np.fft.fft(self.interference_pattern)
        power = np.abs(fft[:20])
        
        return {
            'dominant_frequencies': power.tolist(),
            'critical_moments': self.critical_moments[-5:],
            'signature': power.tolist()
        }
        
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            'size': self.size,
            'pattern': self.interference_pattern.tolist()[:100],  # Sample
            'phases': self.phase_relations.tolist()[:100],
            'strength': self.memory_strength,
            'critical_moments': self.critical_moments
        })
        
    @classmethod
    def from_json(cls, data: str):
        """Deserialize from JSON"""
        d = json.loads(data)
        memory = cls(size=d['size'])
        # Restore what we can
        memory.memory_strength = d['strength']
        memory.critical_moments = d['critical_moments']
        return memory