#!/usr/bin/env python3
"""
Environmental Field - Collective Memory System
Implements the third layer of PROTEUS inheritance
"""

import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
import json


class EnvironmentalField:
    """
    Environmental memory field that organisms can read and write to
    Represents collective knowledge and experience trails
    """
    
    def __init__(self, world_size: Tuple[int, int], resolution: int = 20):
        self.world_size = world_size
        self.resolution = resolution
        
        # Grid dimensions
        self.grid_width = world_size[0] // resolution
        self.grid_height = world_size[1] // resolution
        
        # Pheromone layers (float32 for GPU compatibility)
        self.pheromones = {
            'danger': np.zeros((self.grid_height, self.grid_width), dtype=np.float32),
            'food': np.zeros((self.grid_height, self.grid_width), dtype=np.float32),
            'mating': np.zeros((self.grid_height, self.grid_width), dtype=np.float32),
            'death': np.zeros((self.grid_height, self.grid_width), dtype=np.float32),
            'activity': np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        }
        
        # Memory anchors - strong persistent memories
        self.memory_anchors = []
        
        # Decay rates
        self.decay_rates = {
            'danger': 0.98,
            'food': 0.99,
            'mating': 0.97,
            'death': 0.995,  # Death memories persist longer
            'activity': 0.96
        }
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _deposit_pheromone_jit(grid, x, y, intensity, radius):
        """JIT-compiled pheromone deposition with diffusion"""
        h, w = grid.shape
        
        # Grid coordinates
        gx = int(x)
        gy = int(y)
        
        # Deposit with gaussian falloff
        for dy in prange(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny = gy + dy
                nx = gx + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= radius:
                        falloff = np.exp(-dist*dist / (radius*radius * 0.5))
                        grid[ny, nx] += intensity * falloff
                        
    def deposit_pheromone(self, position: Tuple[float, float], 
                         pheromone_type: str, intensity: float):
        """Deposit pheromone at a world position"""
        if pheromone_type not in self.pheromones:
            return
            
        # Convert world position to grid
        gx = position[0] / self.resolution
        gy = position[1] / self.resolution
        
        # Deposit with diffusion
        self._deposit_pheromone_jit(
            self.pheromones[pheromone_type],
            gx, gy, intensity, 2
        )
        
    def deposit_trace(self, organism_inheritance, position: Tuple[float, float], 
                     event_type: str, intensity: float = 1.0):
        """Deposit an environmental trace from an organism"""
        # Get organism's pheromone signature
        pheromone = organism_inheritance.pheromone_signature
        
        # Map event types to pheromone layers
        if event_type == 'death':
            self.deposit_pheromone(position, 'death', intensity * 2.0)
            self.deposit_pheromone(position, 'danger', intensity)
            # Create memory anchor for significant deaths
            if intensity > 1.5:
                self.add_memory_anchor(position, 'death', intensity)
                
        elif event_type == 'predator_escape':
            self.deposit_pheromone(position, 'danger', intensity * 1.5)
            
        elif event_type == 'feeding':
            self.deposit_pheromone(position, 'food', intensity)
            
        elif event_type == 'reproduction':
            self.deposit_pheromone(position, 'mating', intensity)
            
        # Always deposit activity
        self.deposit_pheromone(position, 'activity', intensity * 0.5)
        
    def add_memory_anchor(self, position: Tuple[float, float], 
                         anchor_type: str, strength: float):
        """Add a strong memory point that persists longer"""
        anchor = {
            'position': position,
            'type': anchor_type,
            'strength': strength,
            'timestamp': 0,  # Will be set by simulation
            'decay': 0.999  # Very slow decay
        }
        
        self.memory_anchors.append(anchor)
        
        # Limit anchors
        if len(self.memory_anchors) > 100:
            # Remove weakest
            self.memory_anchors.sort(key=lambda a: a['strength'])
            self.memory_anchors.pop(0)
            
    @staticmethod
    @jit(nopython=True)
    def _read_field_jit(grid, gx, gy, radius):
        """JIT-compiled field reading"""
        h, w = grid.shape
        total = 0.0
        count = 0
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny = gy + dy
                nx = gx + dx
                
                if 0 <= ny < h and 0 <= nx < w:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= radius:
                        weight = 1.0 / (1.0 + dist)
                        total += grid[ny, nx] * weight
                        count += 1
                        
        return total / max(count, 1)
        
    def read_field(self, position: Tuple[float, float], 
                   radius: float = 30) -> Dict[str, float]:
        """Read pheromone concentrations at a position"""
        # Convert to grid coordinates
        gx = int(position[0] / self.resolution)
        gy = int(position[1] / self.resolution)
        grid_radius = int(radius / self.resolution)
        
        readings = {}
        for ptype, grid in self.pheromones.items():
            readings[ptype] = self._read_field_jit(grid, gx, gy, grid_radius)
            
        return readings
        
    def get_gradient(self, position: Tuple[float, float], 
                    pheromone_type: str) -> Tuple[float, float]:
        """Get gradient direction of a pheromone type"""
        if pheromone_type not in self.pheromones:
            return (0.0, 0.0)
            
        # Sample around position
        delta = 10  # Sample distance
        center = self.read_field(position)[pheromone_type]
        
        # Calculate gradients
        right = self.read_field((position[0] + delta, position[1]))[pheromone_type]
        left = self.read_field((position[0] - delta, position[1]))[pheromone_type]
        up = self.read_field((position[0], position[1] - delta))[pheromone_type]
        down = self.read_field((position[0], position[1] + delta))[pheromone_type]
        
        gx = (right - left) / (2 * delta)
        gy = (down - up) / (2 * delta)
        
        return (gx, gy)
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _update_field_jit(grid, decay_rate):
        """JIT-compiled field update with decay and diffusion"""
        h, w = grid.shape
        
        # Decay
        for i in prange(h):
            for j in range(w):
                grid[i, j] *= decay_rate
                
                # Clear very small values
                if grid[i, j] < 0.001:
                    grid[i, j] = 0.0
                    
    def update(self, dt: float):
        """Update the environmental field"""
        # Update each pheromone layer
        for ptype, grid in self.pheromones.items():
            self._update_field_jit(grid, self.decay_rates[ptype])
            
        # Update memory anchors
        for anchor in self.memory_anchors:
            anchor['strength'] *= anchor['decay']
            
        # Remove weak anchors
        self.memory_anchors = [a for a in self.memory_anchors if a['strength'] > 0.1]
        
    def get_visualization_data(self) -> List[Dict]:
        """Get data for visualization"""
        cells = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Only include cells with significant activity
                danger = self.pheromones['danger'][y, x]
                food = self.pheromones['food'][y, x]
                activity = self.pheromones['activity'][y, x]
                
                if danger > 0.1 or food > 0.1 or activity > 0.1:
                    cells.append({
                        'x': x * self.resolution + self.resolution / 2,
                        'y': y * self.resolution + self.resolution / 2,
                        'danger': float(danger),
                        'food': float(food),
                        'activity': float(activity),
                        'size': self.resolution
                    })
                    
        return cells
        
    def save_state(self) -> Dict:
        """Save field state for persistence"""
        return {
            'pheromones': {k: v.tolist() for k, v in self.pheromones.items()},
            'memory_anchors': self.memory_anchors,
            'world_size': self.world_size,
            'resolution': self.resolution
        }
        
    def load_state(self, state: Dict):
        """Load field state"""
        for ptype, data in state['pheromones'].items():
            self.pheromones[ptype] = np.array(data, dtype=np.float32)
            
        self.memory_anchors = state['memory_anchors']