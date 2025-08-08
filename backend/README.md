# PROTEUS Backend - GPU-Accelerated Simulation

This is the high-performance backend for PROTEUS using Vispy for GPU-accelerated visualization and computation.

## Features

- **GPU Acceleration**: Supports up to 50,000 organisms at 60 FPS
- **Tri-Layer Inheritance**: Full implementation of the PROTEUS inheritance system
- **Environmental Field**: Pheromone-based collective memory
- **WebSocket API**: Real-time communication with React frontend
- **Numba JIT**: CPU optimization for smaller simulations
- **CUDA Support**: Optional GPU kernels for massive simulations

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support (optional):
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x  
pip install cupy-cuda12x
```

## Running

### Standalone Visualization
```bash
python run_vispy.py
```

This opens a native OpenGL window with the simulation.

### WebSocket Server (for React frontend)
```bash
python run_server.py
```

This starts the FastAPI server on `http://localhost:8000` with:
- WebSocket endpoint: `ws://localhost:8000/ws/{client_id}`
- Stats endpoint: `http://localhost:8000/stats`

## Architecture

```
proteus/
├── core/
│   └── topology_engine.py      # Topological flow fields
├── genetics/
│   ├── proteus_inheritance.py  # Tri-layer inheritance
│   └── holographic_memory.py   # Experience encoding
├── environment/
│   └── environmental_field.py  # Collective memory
└── proteus_vispy.py           # Main GPU simulation
```

## Performance

- **CPU Mode**: ~1,000 organisms at 60 FPS
- **GPU Mode**: ~50,000 organisms at 60 FPS
- **Memory**: ~200 bytes per organism (topological core) + 8KB (holographic memory)

## Controls (Standalone Mode)

- **Mouse Drag**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Space**: Pause/Resume
- **R**: Reset simulation
- **ESC**: Exit

## WebSocket Protocol

### Client → Server
```json
{
  "type": "spawn",
  "position": [x, y]
}
```

### Server → Client
```json
{
  "type": "update",
  "data": {
    "organisms": [...],
    "stats": {
      "total": 1000,
      "fps": 60,
      "avg_generation": 5.2
    }
  }
}
```

## Development

To extend the backend:

1. Add new behaviors in `proteus_vispy.py`
2. Implement GPU kernels with `@cuda.jit` decorator
3. Use Numba `@jit` for CPU optimization
4. Keep data in Structure of Arrays (SoA) format for GPU efficiency