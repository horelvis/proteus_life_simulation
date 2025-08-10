# 🧬 PROTEUS - Life Simulation with Real Evolution

**⚠️ EXPERIMENTAL RESEARCH PROJECT**: PROTEUS is a highly experimental system exploring novel computational paradigms inspired by topological dynamics. Performance and stability are not guaranteed. Use for research purposes only.

PROTEUS is an advanced artificial life simulation where organisms evolve through natural selection without any parametrization. The system implements real physics, chemistry, and biology to enable the emergence of complex multicellular life from simple unicellular organisms.

## 🚀 Quick Start

### Backend (Python + Docker)

```bash
cd backend
docker compose up
```

The backend API will be available at [http://localhost:8000](http://localhost:8000).

For ARC solver mode, the WebSocket server runs on [ws://localhost:8765](ws://localhost:8765).

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) to see the simulation.

The frontend now connects exclusively to the Python backend - all simulation logic runs server-side.

### Full System with Docker

```bash
# Start both frontend and backend
docker compose up

# Access frontend at http://localhost:3001
# Access backend at http://localhost:8000
```

## 🧪 Key Features

### Real Evolution (No Parametrization!)

- **Physics-Based Movement**: Organisms move based on real forces, not arbitrary rules
- **Chemical Gradients**: Nutrients and pheromones diffuse following real diffusion equations
- **Sensory Systems**: Vision uses photon detection with quantum efficiency
- **Neural Integration**: Signal processing emerges from temporal integration
- **Metabolic Constraints**: Energy limits all actions naturally

### Tri-Layer Inheritance System

1. **Topological Core**: Fundamental body plan and symmetries
2. **Holographic Memory**: Learned behaviors encoded in high-dimensional space
3. **Environmental Traces**: Epigenetic marks from experiences

### Emergent Multicellularity

The system allows natural evolution from single cells to complex multicellular organisms:

1. **Cell Adhesion**: Van der Waals forces and protein binding
2. **Differentiation**: Morphogen gradients create specialized cell types
3. **Communication**: Gap junctions enable electrical and chemical signaling
4. **Pattern Formation**: Turing patterns emerge from reaction-diffusion

### Organ Evolution

Organisms can evolve various organs with real trade-offs:

**Sensory Organs**:
- `photosensor`: Detects light using photon flux calculations
- `chemoreceptor`: Follows chemical gradients via receptor binding kinetics
- `pressure_sensor`: Detects vibrations and movement
- `magnetoreceptor`: Navigation using magnetic fields

**Movement Organs**:
- `flagellum`: Basic propulsion with energy cost
- `speed_boost`: Enhanced movement at higher energy cost
- `cilia`: Coordinated beating for efficient swimming

**Defense Organs**:
- `toxin_gland`: Chemical defense with production cost
- `armor_plates`: Physical protection but reduced speed
- `electric_organ`: Stunning defense with high energy use
- `camouflage`: Avoid detection through transparency

**Metabolic Organs**:
- `vacuole`: Energy storage and efficiency
- `mitochondria`: Enhanced ATP production
- `chloroplast`: Photosynthesis capability

**Multicellular Organs**:
- `cell_adhesion_proteins`: Enable colony formation
- `gap_junctions`: Direct cell-to-cell communication
- `morphogen_producer`: Create developmental patterns
- `stem_cell_factor`: Enable cell differentiation

## 🔬 Real Physics Implementation

### Cellular Physics
```javascript
// Lennard-Jones potential for cell-cell forces
F = -dU/dr where U = 4ε[(σ/r)¹² - (σ/r)⁶]

// Brownian motion from thermal energy
D = kT/6πηr (Einstein relation)
```

### Chemical Detection
```javascript
// Hill equation for receptor binding
θ = C^n/(Kd^n + C^n)

// Steady-state diffusion
C(r) = C0 * (a/r)
```

### Pattern Formation
```javascript
// Gray-Scott reaction-diffusion
∂u/∂t = Du∇²u - uv² + f(1-u)
∂v/∂t = Dv∇²v + uv² - (f+k)v
```

## 📊 Evolution Metrics

The simulation tracks real evolutionary progress:

- **Genetic Diversity**: Shannon entropy of trait distributions
- **Organ Innovation**: Emergence of new organ types
- **Colony Formation**: Transition to multicellularity
- **Lineage Tracking**: Family trees showing mutations
- **Fitness Landscapes**: Survival and reproduction success

## 🎮 Controls

- **Click organism**: View detailed information
- **Space**: Pause/Resume simulation
- **R**: Reset simulation
- **M**: Show mathematics panel
- **E**: Show evolution metrics

## 🏗️ Architecture

```
frontend/
├── src/
│   ├── simulation/
│   │   ├── Organism.js          # Core organism with physics
│   │   ├── CellularPhysics.js  # Real physics calculations
│   │   ├── MulticellularEvolution.js  # Colony dynamics
│   │   ├── ProteusInheritance.js      # Genetic system
│   │   └── PatternFormation.js        # Turing patterns
│   └── components/
│       ├── SimulationCanvas.js  # WebGL rendering
│       ├── EvolutionMetrics.js  # Real-time analysis
│       └── MathematicsPanel.js  # Live equations

backend/
├── app/
│   ├── organisms.py     # Python organism implementation
│   ├── physics.py       # Backend physics engine
│   └── evolution.py     # Evolutionary algorithms
```

## 🧬 Evolutionary Transitions

The simulation can produce major evolutionary transitions:

1. **Chemical Evolution**: Self-organizing chemical gradients
2. **First Replicators**: Simple organisms that reproduce
3. **Sensory Evolution**: Development of eyes and chemoreceptors
4. **Behavioral Complexity**: Memory and learning
5. **Colonial Life**: Cells that stick together
6. **Division of Labor**: Specialized cell types
7. **True Multicellularity**: Integrated colonial organisms
8. **Complex Organs**: Compound eyes, neural networks

## 🔧 Configuration

Create a `CLAUDE.md` file in the project root for custom settings:

```markdown
# PROTEUS Configuration

## Simulation Parameters
- World size: 800x600
- Initial organisms: 50
- Initial nutrients: 200
- Predator count: 10

## Physics Constants
- Temperature: 300K
- Viscosity: 0.001 Pa·s
- Diffusion rates: Custom per molecule
```

## 📈 Performance

- Supports 1000+ organisms in real-time
- WebGL acceleration for rendering
- Efficient spatial hashing for collision detection
- Web Workers for physics calculations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by research in artificial life and evolutionary biology
- Uses WebGL for high-performance rendering
- Physics equations from biophysics literature
- Pattern formation based on Turing's morphogenesis work

---

**Note**: This is a scientific simulation. All evolution emerges naturally from physical laws - there are no "cheat" parameters or predetermined outcomes. Every behavior you see has evolved because it provides a survival advantage in the simulated environment.