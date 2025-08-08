#!/usr/bin/env python3
"""
Topology Engine - Core of PROTEUS
Manages the topological flow fields and environmental dynamics
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Dict, List
import math


class TopologyEngine:
    """
    Manages the fundamental topological structure of the world
    Creates flow fields, attractors, and dynamic topology
    """
    
    def __init__(self, world_size: Tuple[int, int]):
        self.world_size = world_size
        self.time = 0.0
        
        # Topological parameters
        self.manifold_params = {
            'dimension': 2.5,  # Fractal dimension
            'curvature': 0.1,
            'torsion': 0.05
        }
        
        # Flow field parameters
        self.flow_sources = []
        self.attractors = []
        self.repellers = []
        
        # Initialize base topology
        self._initialize_topology()
        
    def _initialize_topology(self):
        """Initialize the base topological structure"""
        # Create some initial attractors and repellers
        cx, cy = self.world_size[0] / 2, self.world_size[1] / 2
        
        # Central attractor
        self.attractors.append({
            'position': (cx, cy),
            'strength': 0.5,
            'radius': 200,
            'type': 'stable'
        })
        
        # Corner repellers
        corners = [
            (0, 0), 
            (self.world_size[0], 0),
            (0, self.world_size[1]), 
            (self.world_size[0], self.world_size[1])
        ]
        
        for corner in corners:
            self.repellers.append({
                'position': corner,
                'strength': 0.3,
                'radius': 150,
                'type': 'corner'
            })
            
        # Rotating flow sources
        for i in range(3):
            angle = i * 2 * np.pi / 3
            x = cx + 200 * np.cos(angle)
            y = cy + 200 * np.sin(angle)
            
            self.flow_sources.append({
                'position': (x, y),
                'strength': 0.2,
                'rotation': angle,
                'type': 'vortex'
            })
            
    @staticmethod
    @jit(nopython=True)
    def _calculate_flow_field_jit(x, y, sources, attractors, repellers, time):
        """JIT-compiled flow field calculation"""
        vx, vy = 0.0, 0.0
        
        # Attractor contributions
        for i in range(len(attractors)):
            ax, ay = attractors[i, 0], attractors[i, 1]
            strength = attractors[i, 2]
            radius = attractors[i, 3]
            
            dx = ax - x
            dy = ay - y
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            
            if dist < radius:
                factor = strength * (1 - dist/radius) / dist
                vx += dx * factor
                vy += dy * factor
                
        # Repeller contributions
        for i in range(len(repellers)):
            rx, ry = repellers[i, 0], repellers[i, 1]
            strength = repellers[i, 2]
            radius = repellers[i, 3]
            
            dx = x - rx
            dy = y - ry
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            
            if dist < radius:
                factor = strength * (1 - dist/radius) / dist
                vx += dx * factor
                vy += dy * factor
                
        # Flow source contributions (vortices)
        for i in range(len(sources)):
            sx, sy = sources[i, 0], sources[i, 1]
            strength = sources[i, 2]
            rotation = sources[i, 3]
            
            dx = x - sx
            dy = y - sy
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            
            if dist < 150:
                # Rotating flow
                angle = np.arctan2(dy, dx) + np.pi/2
                factor = strength * (1 - dist/150)
                vx += factor * np.cos(angle + rotation + time * 0.5)
                vy += factor * np.sin(angle + rotation + time * 0.5)
                
        # Add base flow
        vx += 0.1 * np.sin(x * 0.01 + time)
        vy += 0.1 * np.cos(y * 0.01 + time)
        
        return vx, vy
        
    def get_flow_at(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Get the topological flow at a given position"""
        # Convert lists to numpy arrays for JIT
        sources = np.array([
            [s['position'][0], s['position'][1], s['strength'], s['rotation']]
            for s in self.flow_sources
        ], dtype=np.float32) if self.flow_sources else np.zeros((0, 4), dtype=np.float32)
        
        attractors = np.array([
            [a['position'][0], a['position'][1], a['strength'], a['radius']]
            for a in self.attractors
        ], dtype=np.float32) if self.attractors else np.zeros((0, 4), dtype=np.float32)
        
        repellers = np.array([
            [r['position'][0], r['position'][1], r['strength'], r['radius']]
            for r in self.repellers
        ], dtype=np.float32) if self.repellers else np.zeros((0, 4), dtype=np.float32)
        
        return self._calculate_flow_field_jit(
            position[0], position[1],
            sources, attractors, repellers,
            self.time
        )
        
    def add_dynamic_attractor(self, position: Tuple[float, float], 
                            strength: float, duration: float):
        """Add a temporary attractor (e.g., for food sources)"""
        self.attractors.append({
            'position': position,
            'strength': strength,
            'radius': 100,
            'type': 'temporary',
            'duration': duration,
            'created_at': self.time
        })
        
    def add_dynamic_repeller(self, position: Tuple[float, float], 
                           strength: float, duration: float):
        """Add a temporary repeller (e.g., for danger zones)"""
        self.repellers.append({
            'position': position,
            'strength': strength,
            'radius': 80,
            'type': 'temporary',
            'duration': duration,
            'created_at': self.time
        })
        
    def evolve(self, dt: float):
        """Evolve the topological structure over time"""
        self.time += dt
        
        # Update flow sources (rotating vortices)
        for source in self.flow_sources:
            if source['type'] == 'vortex':
                source['rotation'] += dt * 0.1
                
        # Remove expired temporary features
        self.attractors = [
            a for a in self.attractors
            if a['type'] != 'temporary' or 
            (self.time - a['created_at']) < a['duration']
        ]
        
        self.repellers = [
            r for r in self.repellers
            if r['type'] != 'temporary' or
            (self.time - r['created_at']) < r['duration']
        ]
        
        # Slowly move vortices
        for source in self.flow_sources:
            if source['type'] == 'vortex':
                angle = self.time * 0.05
                radius = 200
                cx, cy = self.world_size[0] / 2, self.world_size[1] / 2
                
                new_x = cx + radius * np.cos(angle + source['rotation'])
                new_y = cy + radius * np.sin(angle + source['rotation'])
                source['position'] = (new_x, new_y)
                
    def get_curvature_at(self, position: Tuple[float, float]) -> float:
        """Calculate local topological curvature"""
        # Sample flow field around position
        delta = 5.0
        flow_center = self.get_flow_at(position)
        flow_right = self.get_flow_at((position[0] + delta, position[1]))
        flow_up = self.get_flow_at((position[0], position[1] + delta))
        
        # Calculate divergence (simplified curvature)
        div_x = (flow_right[0] - flow_center[0]) / delta
        div_y = (flow_up[1] - flow_center[1]) / delta
        
        return div_x + div_y
        
    def get_safe_zones(self) -> List[Dict]:
        """Get current safe zones (low flow areas)"""
        # For now, return predefined safe zones
        # In future, calculate dynamically based on topology
        return [
            {
                'x': self.world_size[0] * 0.2,
                'y': self.world_size[1] * 0.3,
                'radius': 120
            },
            {
                'x': self.world_size[0] * 0.8,
                'y': self.world_size[1] * 0.7,
                'radius': 120
            }
        ]
        
    def get_visualization_data(self) -> Dict:
        """Get topology data for visualization"""
        return {
            'attractors': [{
                'x': a['position'][0],
                'y': a['position'][1],
                'strength': a['strength'],
                'radius': a['radius']
            } for a in self.attractors],
            'repellers': [{
                'x': r['position'][0],
                'y': r['position'][1],
                'strength': r['strength'],
                'radius': r['radius']
            } for r in self.repellers],
            'flow_sources': [{
                'x': s['position'][0],
                'y': s['position'][1],
                'strength': s['strength'],
                'rotation': s['rotation']
            } for s in self.flow_sources]
        }