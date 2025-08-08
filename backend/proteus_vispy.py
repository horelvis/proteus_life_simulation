#!/usr/bin/env python3
"""
PROTEUS Vispy Backend - GPU-Accelerated Visualization
Combines tri-layer inheritance with high-performance OpenGL rendering
"""

import numpy as np
import vispy.app
from vispy import gloo, visuals
from vispy.scene import SceneCanvas, visuals
from vispy.util.transforms import translate, rotate
from numba import jit, cuda, prange
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Import our core systems
from proteus.core.topology_engine import TopologyEngine
from proteus.genetics.holographic_memory import HolographicMemory
from proteus.genetics.proteus_inheritance import ProteusInheritance
from proteus.environment.environmental_field import EnvironmentalField

# Check CUDA availability
CUDA_AVAILABLE = cuda.is_available()
if CUDA_AVAILABLE:
    print("🚀 CUDA GPU acceleration available!")
else:
    print("💻 Running on CPU with Numba optimization")

@dataclass
class OrganismGL:
    """GPU-optimized organism representation"""
    id: int
    position: np.ndarray  # float32[2]
    velocity: np.ndarray  # float32[2]
    color: np.ndarray     # float32[4] RGBA
    energy: float
    age: float
    generation: int
    inheritance: ProteusInheritance
    
    # GPU buffers
    gpu_index: int = -1
    
class ProteusVispy:
    """High-performance PROTEUS simulation with OpenGL"""
    
    def __init__(self, world_size=(1600, 1200), max_organisms=50000):
        self.world_size = world_size
        self.max_organisms = max_organisms
        
        # Initialize core systems
        self.topology_engine = TopologyEngine(world_size)
        self.environmental_field = EnvironmentalField(world_size)
        
        # OpenGL Canvas
        self.canvas = SceneCanvas(
            keys='interactive', 
            size=world_size,
            bgcolor='#0A0C14',  # Dark water background
            title='PROTEUS - Proto-Topological Evolution'
        )
        self.canvas.show()
        
        # 2D view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        # Initialize GPU data structures
        self._init_gpu_buffers()
        
        # Visual components
        self._init_visuals()
        
        # Organisms
        self.organisms: List[OrganismGL] = []
        self.organism_count = 0
        
        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = 0
        
        # Animation timer (60 FPS)
        self.timer = vispy.app.Timer(interval=1/60., connect=self.update)
        
        # WebSocket for frontend communication
        self.websocket_clients = []
        
    def _init_gpu_buffers(self):
        """Initialize GPU memory buffers"""
        # Organism data (Structure of Arrays for GPU efficiency)
        self.gpu_positions = np.zeros((self.max_organisms, 2), dtype=np.float32)
        self.gpu_velocities = np.zeros((self.max_organisms, 2), dtype=np.float32)
        self.gpu_colors = np.zeros((self.max_organisms, 4), dtype=np.float32)
        self.gpu_energies = np.zeros(self.max_organisms, dtype=np.float32)
        self.gpu_ages = np.zeros(self.max_organisms, dtype=np.float32)
        
        # Environmental field as texture
        self.field_resolution = (self.world_size[0]//4, self.world_size[1]//4)
        self.gpu_field = np.zeros((*self.field_resolution, 4), dtype=np.float32)
        
        # Pheromone layers
        self.gpu_pheromones = {
            'danger': np.zeros(self.field_resolution, dtype=np.float32),
            'food': np.zeros(self.field_resolution, dtype=np.float32),
            'mating': np.zeros(self.field_resolution, dtype=np.float32),
            'death': np.zeros(self.field_resolution, dtype=np.float32)
        }
        
    def _init_visuals(self):
        """Initialize Vispy visual components"""
        # Environmental field as image texture
        self.field_visual = visuals.Image(
            self.gpu_field,
            interpolation='linear',
            parent=self.view.scene,
            method='impostor'
        )
        
        # Organisms as point sprites (ultra fast)
        self.organism_visual = visuals.Markers(
            pos=self.gpu_positions[:self.organism_count],
            size=10,
            face_color=self.gpu_colors[:self.organism_count],
            edge_color=None,
            scaling=True,
            antialias=1,
            parent=self.view.scene
        )
        
        # Safe zones
        self.safe_zones = [
            {'center': (self.world_size[0]*0.2, self.world_size[1]*0.3), 'radius': 120},
            {'center': (self.world_size[0]*0.8, self.world_size[1]*0.7), 'radius': 120}
        ]
        
        for zone in self.safe_zones:
            circle = visuals.Ellipse(
                center=zone['center'],
                radius=zone['radius'],
                color=(0, 1, 0, 0.2),
                border_color=(0, 1, 0, 0.5),
                border_width=2,
                parent=self.view.scene
            )
        
        # Grid
        grid = visuals.GridLines(
            scale=(50, 50),
            color=(1, 1, 1, 0.1),
            parent=self.view.scene
        )
        
        # FPS display
        self.fps_text = visuals.Text(
            '',
            pos=(10, 10),
            color='white',
            font_size=12,
            parent=self.canvas.scene
        )
        
    def spawn_organism(self, x: float, y: float, parent1=None, parent2=None) -> OrganismGL:
        """Spawn a new organism"""
        if self.organism_count >= self.max_organisms:
            return None
            
        # Create inheritance
        inheritance = ProteusInheritance(
            parent1.inheritance if parent1 else None,
            parent2.inheritance if parent2 else None
        )
        
        # Express phenotype
        phenotype = inheritance.expressPhenotype()
        
        # Create organism
        organism = OrganismGL(
            id=self.organism_count,
            position=np.array([x, y], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            color=self._phenotype_to_color(phenotype),
            energy=1.0,
            age=0.0,
            generation=inheritance.generation,
            inheritance=inheritance,
            gpu_index=self.organism_count
        )
        
        # Add to GPU buffers
        idx = self.organism_count
        self.gpu_positions[idx] = organism.position
        self.gpu_velocities[idx] = organism.velocity
        self.gpu_colors[idx] = organism.color
        self.gpu_energies[idx] = organism.energy
        self.gpu_ages[idx] = organism.age
        
        self.organisms.append(organism)
        self.organism_count += 1
        
        return organism
        
    def _phenotype_to_color(self, phenotype) -> np.ndarray:
        """Convert phenotype to RGBA color"""
        # Color based on traits
        r = 0.3 + phenotype['motility'] * 0.4
        g = 0.5 + phenotype['sensitivity'] * 0.3
        b = 0.6 + phenotype['resilience'] * 0.3
        a = 0.8
        return np.array([r, g, b, a], dtype=np.float32)
        
    def initialize_population(self, count=100):
        """Create initial population"""
        for _ in range(count):
            x = np.random.random() * self.world_size[0]
            y = np.random.random() * self.world_size[1]
            self.spawn_organism(x, y)
            
    @jit(nopython=True, parallel=True)
    def update_organisms_cpu(positions, velocities, energies, ages, 
                           field_data, world_size, dt):
        """CPU-optimized organism update using Numba"""
        for i in prange(len(positions)):
            if energies[i] <= 0:
                continue
                
            # Age and energy
            ages[i] += dt
            energies[i] -= dt * 0.01
            
            # Simple topological flow (simplified for CPU)
            x, y = positions[i]
            fx = np.sin(x * 0.01) * 0.5
            fy = np.cos(y * 0.01) * 0.5
            
            # Update velocity
            velocities[i, 0] = velocities[i, 0] * 0.95 + fx * dt
            velocities[i, 1] = velocities[i, 1] * 0.95 + fy * dt
            
            # Update position
            positions[i] += velocities[i] * dt
            
            # Wrap around world
            positions[i, 0] = positions[i, 0] % world_size[0]
            positions[i, 1] = positions[i, 1] % world_size[1]
            
    if CUDA_AVAILABLE:
        @cuda.jit
        def update_organisms_gpu(positions, velocities, energies, ages, 
                               field_data, world_size, dt):
            """GPU kernel for organism updates"""
            i = cuda.grid(1)
            if i >= positions.shape[0] or energies[i] <= 0:
                return
                
            # Each thread handles one organism
            ages[i] += dt
            energies[i] -= dt * 0.01
            
            # Topological flow
            x, y = positions[i, 0], positions[i, 1]
            fx = np.sin(x * 0.01) * 0.5
            fy = np.cos(y * 0.01) * 0.5
            
            # Update velocity
            velocities[i, 0] = velocities[i, 0] * 0.95 + fx * dt
            velocities[i, 1] = velocities[i, 1] * 0.95 + fy * dt
            
            # Update position
            positions[i, 0] = (positions[i, 0] + velocities[i, 0] * dt) % world_size[0]
            positions[i, 1] = (positions[i, 1] + velocities[i, 1] * dt) % world_size[1]
            
    @jit(nopython=True)
    def update_field_cpu(field, pheromones, decay_rate):
        """Update environmental field on CPU"""
        h, w = field.shape[:2]
        for i in range(h):
            for j in range(w):
                # Combine pheromone layers
                field[i, j, 0] = pheromones['danger'][i, j]    # R
                field[i, j, 1] = pheromones['food'][i, j]      # G  
                field[i, j, 2] = pheromones['mating'][i, j]    # B
                field[i, j, 3] = pheromones['death'][i, j] * 0.5  # A
                
                # Decay
                for k in range(4):
                    field[i, j, k] *= decay_rate
                    
    def update(self, event):
        """Main update loop - called 60 times per second"""
        dt = event.dt if event else 1/60.
        
        # Update topology engine
        self.topology_engine.evolve(dt)
        
        # Update organisms
        if CUDA_AVAILABLE and self.organism_count > 1000:
            # GPU update for large populations
            threads_per_block = 256
            blocks = (self.organism_count + threads_per_block - 1) // threads_per_block
            
            self.update_organisms_gpu[blocks, threads_per_block](
                self.gpu_positions, self.gpu_velocities, 
                self.gpu_energies, self.gpu_ages,
                self.gpu_field, np.array(self.world_size, dtype=np.float32), 
                np.float32(dt)
            )
        else:
            # CPU update for small populations
            self.update_organisms_cpu(
                self.gpu_positions[:self.organism_count],
                self.gpu_velocities[:self.organism_count],
                self.gpu_energies[:self.organism_count],
                self.gpu_ages[:self.organism_count],
                self.gpu_field, self.world_size, dt
            )
            
        # Update high-level organism logic
        self._update_organism_behaviors(dt)
        
        # Update environmental field
        self.update_field_cpu(
            self.gpu_field,
            self.gpu_pheromones,
            0.99
        )
        
        # Update visuals
        if self.organism_count > 0:
            self.organism_visual.set_data(
                pos=self.gpu_positions[:self.organism_count],
                face_color=self.gpu_colors[:self.organism_count],
                size=10 + self.gpu_energies[:self.organism_count] * 5
            )
            
        self.field_visual.set_data(self.gpu_field)
        
        # Update FPS
        self.fps_counter += 1
        if self.fps_counter % 60 == 0:
            self.fps_text.text = f'FPS: {60/dt:.1f} | Organisms: {self.organism_count}'
            
    def _update_organism_behaviors(self, dt):
        """Update complex behaviors that need inheritance system"""
        dead_indices = []
        
        for i, org in enumerate(self.organisms):
            if org.gpu_index < 0:
                continue
                
            # Check death
            if self.gpu_energies[org.gpu_index] <= 0:
                dead_indices.append(i)
                # Leave death trace
                self._deposit_death_trace(org)
                continue
                
            # Update from GPU data
            org.position = self.gpu_positions[org.gpu_index]
            org.velocity = self.gpu_velocities[org.gpu_index]
            org.energy = self.gpu_energies[org.gpu_index]
            org.age = self.gpu_ages[org.gpu_index]
            
            # Record experiences
            if np.random.random() < dt:  # Sample experiences
                experience = {
                    'type': 'exploration',
                    'position': org.position.tolist(),
                    'importance': 0.5,
                    'energy': org.energy
                }
                org.inheritance.experience(experience)
                
            # Reproduction check
            if org.energy > 1.5 and org.age > 5:
                self._reproduce(org)
                
        # Remove dead organisms
        for idx in reversed(dead_indices):
            self.organisms.pop(idx)
            
    def _reproduce(self, parent):
        """Handle reproduction with inheritance"""
        if self.organism_count >= self.max_organisms:
            return
            
        # Reduce parent energy
        parent.energy *= 0.5
        self.gpu_energies[parent.gpu_index] *= 0.5
        
        # Spawn offspring nearby
        offset = np.random.randn(2) * 20
        x = parent.position[0] + offset[0]
        y = parent.position[1] + offset[1]
        
        self.spawn_organism(x, y, parent)
        
    def _deposit_death_trace(self, organism):
        """Leave environmental memory trace on death"""
        # Grid position
        gx = int(organism.position[0] / 4) % self.field_resolution[0]
        gy = int(organism.position[1] / 4) % self.field_resolution[1]
        
        # Strong death pheromone
        self.gpu_pheromones['death'][gy, gx] += 2.0
        
        # Diffuse to neighbors
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny = (gy + dy) % self.field_resolution[1]
                nx = (gx + dx) % self.field_resolution[0]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    self.gpu_pheromones['death'][ny, nx] += 1.0 / dist
                    
    def start(self):
        """Start the simulation"""
        self.initialize_population(100)
        self.timer.start()
        vispy.app.run()
        
    async def broadcast_state(self):
        """Send state to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        state = {
            'organisms': [
                {
                    'id': org.id,
                    'position': org.position.tolist(),
                    'energy': float(org.energy),
                    'generation': org.generation,
                    'age': float(org.age)
                }
                for org in self.organisms[:100]  # Limit for performance
            ],
            'stats': {
                'total': self.organism_count,
                'fps': 60,
                'avg_generation': np.mean([o.generation for o in self.organisms]) if self.organisms else 0
            }
        }
        
        message = json.dumps({'type': 'update', 'data': state})
        
        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except:
                self.websocket_clients.remove(client)

# FastAPI integration for frontend communication
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulation = None

@app.on_event("startup")
async def startup():
    global simulation
    simulation = ProteusVispy()
    # Run Vispy in separate thread
    import threading
    thread = threading.Thread(target=simulation.start)
    thread.start()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    simulation.websocket_clients.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(0.1)
            await simulation.broadcast_state()
    except:
        simulation.websocket_clients.remove(websocket)

@app.get("/stats")
async def get_stats():
    return {
        "organisms": simulation.organism_count,
        "max_organisms": simulation.max_organisms,
        "world_size": simulation.world_size
    }

if __name__ == "__main__":
    # Run with: python proteus_vispy.py
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run standalone visualization
        sim = ProteusVispy()
        sim.start()