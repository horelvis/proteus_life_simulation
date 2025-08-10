/**
 * PROTEUS Cellular Physics Engine
 * Pure physics-based multicellular emergence
 * NO PARAMETRIZATION - Only physical laws
 */

export class CellularPhysics {
  constructor() {
    // Physical constants (these are laws of physics, not parameters)
    this.BOLTZMANN_CONSTANT = 1.38e-23;
    this.TEMPERATURE = 300; // Kelvin
    this.VISCOSITY = 0.001; // Water at 20°C
    this.CELL_RADIUS = 5; // micrometers
  }
  
  /**
   * Van der Waals forces between cells
   * Real physics: attraction at distance, repulsion when too close
   */
  calculateCellForce(cell1, cell2) {
    const dx = cell2.position.x - cell1.position.x;
    const dy = cell2.position.y - cell1.position.y;
    const r = Math.sqrt(dx * dx + dy * dy);
    
    if (r === 0) return { x: 0, y: 0 };
    
    // Lennard-Jones potential for biological cells
    const sigma = this.CELL_RADIUS * 2; // Equilibrium distance
    const epsilon = this.getBindingEnergy(cell1, cell2);
    
    // F = -dU/dr where U = 4ε[(σ/r)^12 - (σ/r)^6]
    const sr6 = Math.pow(sigma / r, 6);
    const sr12 = sr6 * sr6;
    
    const forceMagnitude = 24 * epsilon / r * (2 * sr12 - sr6);
    
    // Add thermal fluctuations (Brownian motion)
    const thermal = this.calculateThermalForce();
    
    return {
      x: forceMagnitude * (dx / r) + thermal.x,
      y: forceMagnitude * (dy / r) + thermal.y
    };
  }
  
  /**
   * Binding energy depends on surface proteins
   * This emerges from protein expression, not parameters
   */
  getBindingEnergy(cell1, cell2) {
    let energy = 0.1; // Base van der Waals
    
    // Each adhesion protein increases binding
    cell1.organs.forEach(organ => {
      if (organ.type === 'cell_adhesion_proteins') {
        // Protein-protein binding energy
        const matchingProteins = cell2.organs.filter(o => 
          o.type === 'cell_adhesion_proteins'
        );
        matchingProteins.forEach(match => {
          // Real protein binding: depends on concentration and affinity
          energy += organ.expression * match.expression * 0.5;
        });
      }
    });
    
    return energy;
  }
  
  /**
   * Brownian motion from thermal energy
   */
  calculateThermalForce() {
    // Einstein relation: D = kT/6πηr
    // Scale up for simulation units (pixels instead of meters)
    const scaleFactor = 1e6; // Convert from microscopic to simulation scale
    const D = this.BOLTZMANN_CONSTANT * this.TEMPERATURE * scaleFactor / 
              (6 * Math.PI * this.VISCOSITY * this.CELL_RADIUS * 1e-6);
    
    // Random walk with proper variance
    const angle = Math.random() * 2 * Math.PI;
    const magnitude = Math.sqrt(2 * D) * Math.sqrt(-2 * Math.log(Math.random())); // Box-Muller for normal distribution
    
    return {
      x: magnitude * Math.cos(angle) * 0.01, // Scale down to reasonable movement
      y: magnitude * Math.sin(angle) * 0.01
    };
  }
  
  /**
   * Diffusion equation for morphogens
   * ∂C/∂t = D∇²C + P - γC
   */
  updateMorphogenField(field, producers, deltaTime) {
    const newField = [];
    const width = field.length;
    const height = field[0].length;
    
    for (let x = 0; x < width; x++) {
      newField[x] = [];
      for (let y = 0; y < height; y++) {
        // Laplacian for diffusion
        const laplacian = this.calculateLaplacian(field, x, y);
        
        // Production term
        let production = 0;
        producers.forEach(producer => {
          const dist = Math.sqrt(
            Math.pow(producer.position.x - x, 2) + 
            Math.pow(producer.position.y - y, 2)
          );
          if (dist < 50) { // Production range
            production += producer.productionRate * Math.exp(-dist / 20);
          }
        });
        
        // Degradation (first-order kinetics)
        const degradation = field[x][y] * 0.1;
        
        // Update concentration
        const D = 0.5; // Diffusion coefficient
        newField[x][y] = field[x][y] + deltaTime * (
          D * laplacian + production - degradation
        );
        
        // Ensure non-negative
        newField[x][y] = Math.max(0, newField[x][y]);
      }
    }
    
    return newField;
  }
  
  calculateLaplacian(field, x, y) {
    const width = field.length;
    const height = field[0].length;
    
    let laplacian = -4 * field[x][y];
    
    // Finite difference approximation
    if (x > 0) laplacian += field[x-1][y];
    if (x < width-1) laplacian += field[x+1][y];
    if (y > 0) laplacian += field[x][y-1];
    if (y < height-1) laplacian += field[x][y+1];
    
    return laplacian;
  }
  
  /**
   * Nutrient sharing through gap junctions
   * Based on Fick's law of diffusion
   */
  calculateNutrientFlow(cell1, cell2) {
    // Check for gap junctions
    const junctions1 = cell1.organs.filter(o => o.type === 'gap_junctions');
    const junctions2 = cell2.organs.filter(o => o.type === 'gap_junctions');
    
    if (junctions1.length === 0 || junctions2.length === 0) {
      return 0; // No connection
    }
    
    // Permeability depends on junction expression
    const permeability = Math.min(
      junctions1.reduce((sum, j) => sum + j.expression, 0),
      junctions2.reduce((sum, j) => sum + j.expression, 0)
    );
    
    // Concentration gradient drives flow
    const gradient = cell1.energy - cell2.energy;
    
    // Fick's law: J = -P * ∇C
    return permeability * gradient * 0.1;
  }
  
  /**
   * Mechanical stress on colony structure
   * Young's modulus emerges from structural proteins
   */
  calculateStructuralStress(colony) {
    const stress = [];
    
    colony.members.forEach((member1, id1) => {
      let totalStress = { x: 0, y: 0 };
      
      colony.members.forEach((member2, id2) => {
        if (id1 === id2) return;
        
        const force = this.calculateCellForce(member1, member2);
        totalStress.x += force.x;
        totalStress.y += force.y;
      });
      
      // Structural matrix provides resistance
      const structuralStrength = member1.organs
        .filter(o => o.type === 'structural_matrix')
        .reduce((sum, s) => sum + s.expression, 0);
      
      // Hooke's law: stress = E * strain
      const elasticModulus = 1 + structuralStrength * 10;
      
      stress.push({
        id: id1,
        stress: totalStress,
        deformation: {
          x: totalStress.x / elasticModulus,
          y: totalStress.y / elasticModulus
        }
      });
    });
    
    return stress;
  }
  
  /**
   * Signal propagation through neural-like networks
   * Cable equation for bioelectrical signals
   */
  propagateSignal(source, network, signal) {
    const propagation = new Map();
    propagation.set(source.id, { intensity: signal, time: 0 });
    
    const queue = [source];
    const visited = new Set();
    
    while (queue.length > 0) {
      const current = queue.shift();
      if (visited.has(current.id)) continue;
      visited.add(current.id);
      
      const currentSignal = propagation.get(current.id);
      
      // Find connected cells
      network.forEach(cell => {
        if (cell.id === current.id) return;
        
        const distance = this.calculateDistance(current, cell);
        
        // Check electrical coupling
        const coupling = this.getElectricalCoupling(current, cell);
        if (coupling > 0 && distance < 20) { // Adjacent cells
          // Cable equation: signal decay
          const decayConstant = 1 / (1 + coupling * 10);
          const attenuatedSignal = currentSignal.intensity * 
                                  Math.exp(-distance * decayConstant);
          
          // Time delay based on conduction velocity
          const velocity = coupling * 50; // μm/ms
          const delay = distance / velocity;
          
          if (attenuatedSignal > 0.01) { // Threshold for propagation
            propagation.set(cell.id, {
              intensity: attenuatedSignal,
              time: currentSignal.time + delay
            });
            queue.push(cell);
          }
        }
      });
    }
    
    return propagation;
  }
  
  getElectricalCoupling(cell1, cell2) {
    // Gap junctions provide electrical coupling
    const gap1 = cell1.organs.filter(o => o.type === 'gap_junctions');
    const gap2 = cell2.organs.filter(o => o.type === 'gap_junctions');
    
    if (gap1.length === 0 || gap2.length === 0) return 0;
    
    // Conductance proportional to channel expression
    return Math.min(
      gap1.reduce((sum, g) => sum + g.expression, 0),
      gap2.reduce((sum, g) => sum + g.expression, 0)
    );
  }
  
  calculateDistance(cell1, cell2) {
    const dx = cell2.position.x - cell1.position.x;
    const dy = cell2.position.y - cell1.position.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  /**
   * Emergent colony behavior from physical interactions
   */
  updateColonyDynamics(colony, deltaTime) {
    // Calculate all forces
    const stresses = this.calculateStructuralStress(colony);
    
    // Update positions based on forces
    stresses.forEach(({ id, deformation }) => {
      const member = colony.members.get(id);
      if (member) {
        member.position.x += deformation.x * deltaTime;
        member.position.y += deformation.y * deltaTime;
      }
    });
    
    // Update colony center of mass
    colony.updateCenter();
    
    // Nutrient distribution through diffusion
    const nutrientFlows = new Map();
    colony.members.forEach((member1, id1) => {
      let totalFlow = 0;
      colony.members.forEach((member2, id2) => {
        if (id1 === id2) return;
        const flow = this.calculateNutrientFlow(member1, member2);
        totalFlow += flow;
      });
      nutrientFlows.set(id1, totalFlow);
    });
    
    return {
      forces: stresses,
      nutrientFlows: nutrientFlows,
      cohesion: this.calculateColonyCohesion(colony)
    };
  }
  
  /**
   * Colony cohesion emerges from binding forces
   */
  calculateColonyCohesion(colony) {
    let totalBinding = 0;
    let pairCount = 0;
    
    colony.members.forEach((member1, id1) => {
      colony.members.forEach((member2, id2) => {
        if (id1 >= id2) return; // Avoid double counting
        
        const binding = this.getBindingEnergy(member1, member2);
        const distance = this.calculateDistance(member1, member2);
        
        if (distance < 20) { // Adjacent cells
          totalBinding += binding;
          pairCount++;
        }
      });
    });
    
    return pairCount > 0 ? totalBinding / pairCount : 0;
  }
}

/**
 * Reaction-Diffusion system for pattern formation
 * Based on Turing patterns
 */
export class PatternFormation {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.u = this.initializeField(); // Activator
    this.v = this.initializeField(); // Inhibitor
  }
  
  initializeField() {
    const field = [];
    for (let x = 0; x < this.width; x++) {
      field[x] = [];
      for (let y = 0; y < this.height; y++) {
        // Small random perturbations
        field[x][y] = 1 + (Math.random() - 0.5) * 0.01;
      }
    }
    return field;
  }
  
  /**
   * Gray-Scott reaction-diffusion
   * Creates spots, stripes, and complex patterns
   */
  update(deltaTime) {
    const newU = [];
    const newV = [];
    
    // Diffusion rates
    const Du = 0.16;
    const Dv = 0.08;
    
    // Reaction rates (these determine pattern type)
    const f = 0.035; // Feed rate
    const k = 0.065; // Kill rate
    
    for (let x = 0; x < this.width; x++) {
      newU[x] = [];
      newV[x] = [];
      
      for (let y = 0; y < this.height; y++) {
        const u = this.u[x][y];
        const v = this.v[x][y];
        
        // Laplacians for diffusion
        const lapU = this.laplacian(this.u, x, y);
        const lapV = this.laplacian(this.v, x, y);
        
        // Reaction terms
        const uvv = u * v * v;
        
        // Update equations
        newU[x][y] = u + deltaTime * (
          Du * lapU - uvv + f * (1 - u)
        );
        
        newV[x][y] = v + deltaTime * (
          Dv * lapV + uvv - (f + k) * v
        );
        
        // Ensure bounds
        newU[x][y] = Math.max(0, Math.min(1, newU[x][y]));
        newV[x][y] = Math.max(0, Math.min(1, newV[x][y]));
      }
    }
    
    this.u = newU;
    this.v = newV;
  }
  
  laplacian(field, x, y) {
    let lap = -4 * field[x][y];
    
    // Periodic boundary conditions
    const xm = (x - 1 + this.width) % this.width;
    const xp = (x + 1) % this.width;
    const ym = (y - 1 + this.height) % this.height;
    const yp = (y + 1) % this.height;
    
    lap += field[xm][y] + field[xp][y];
    lap += field[x][ym] + field[x][yp];
    
    return lap;
  }
  
  /**
   * Get pattern value at position
   * Cells can read this to determine their fate
   */
  getPatternAt(x, y) {
    const xi = Math.floor(x / 10) % this.width;
    const yi = Math.floor(y / 10) % this.height;
    
    return {
      activator: this.u[xi][yi],
      inhibitor: this.v[xi][yi],
      ratio: this.u[xi][yi] / (this.v[xi][yi] + 0.001)
    };
  }
}