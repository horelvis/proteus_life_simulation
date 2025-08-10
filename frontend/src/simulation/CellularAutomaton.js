/**
 * Cellular Automaton (AC) - Internal control system for organisms
 * No neurons, just local rules on a 2D grid
 */

export class CellularAutomaton {
  constructor(size = 16) {
    this.size = size;
    
    // Grid states
    this.grid = [];
    this.nextGrid = [];
    
    // Initialize grids
    for (let i = 0; i < size; i++) {
      this.grid[i] = [];
      this.nextGrid[i] = [];
      for (let j = 0; j < size; j++) {
        this.grid[i][j] = this.createCell();
        this.nextGrid[i][j] = this.createCell();
      }
    }
    
    // Sensor inputs (normalized 0-1)
    this.sensors = {
      lightLevel: 0,
      chemicalGradientX: 0,
      chemicalGradientY: 0,
      damage: 0,
      neighborDensity: 0,
      pulseRhythm: 0,
      preyTrace: 0,
      temperature: 0.5 // Baseline
    };
    
    // Effector outputs
    this.effectors = {
      movementX: 0,
      movementY: 0,
      adhesion: 0,
      chemicalSecretion: 0,
      secretionType: 'FOOD', // Type of pheromone to secrete
      rigidity: 0.5,
      oscillation: 0
    };
    
    // Rule parameters (inheritable)
    this.params = {
      chemotaxisStrength: 1.5,      // Increased for better food tracking
      phototaxisStrength: -0.7,     // Stronger light avoidance
      damageAvoidance: 2.0,
      diffusionRate: 0.15,          // Faster energy distribution
      oscillatorPeriod: 8,          // Faster movement cycles
      refractoryPeriod: 5,
      repairThreshold: 3,
      rigidityThreshold: 0.5        // More flexible for better movement
    };
    
    // Internal timers
    this.time = 0;
    this.oscillatorPhase = 0;
  }
  
  createCell() {
    return {
      // State variables
      type: 'tissue', // tissue, skin, void, scar
      energy: 1.0,
      charge: 0,
      phase: 0,
      memoryTrace: 0,
      heritageMarker: 0,
      
      // Activity
      activation: 0,
      refractoryTimer: 0,
      
      // Position-based gradients
      chemicalSensor: 0,
      lightSensor: 0,
      damageSensor: 0,
      
      // Morphogenesis
      repairTimer: 0,
      skinFormationPotential: 0,
      structuralIntegrity: 1.0
    };
  }
  
  /**
   * Update sensor values from organism's perception
   */
  updateSensors(perception) {
    // Light detection
    if (perception.predators && perception.predators.length > 0) {
      // Predators emit light
      const nearest = perception.predators[0];
      this.sensors.lightLevel = Math.max(0, 1 - nearest.distance / 100);
    } else {
      this.sensors.lightLevel *= 0.9; // Decay
    }
    
    // Chemical gradient (both nutrients and pheromones)
    let totalGradX = 0;
    let totalGradY = 0;
    
    // Nutrient gradient
    if (perception.nutrients && perception.nutrients.length > 0) {
      perception.nutrients.forEach(nutrient => {
        const weight = nutrient.strength || (1 / (1 + nutrient.distance));
        totalGradX += Math.cos(nutrient.gradient_direction || 0) * weight;
        totalGradY += Math.sin(nutrient.gradient_direction || 0) * weight;
      });
    }
    
    // Pheromone gradients
    if (perception.pheromoneGradients) {
      // Food trail following
      if (perception.pheromoneGradients.FOOD) {
        totalGradX += perception.pheromoneGradients.FOOD.x * this.params.chemotaxisStrength;
        totalGradY += perception.pheromoneGradients.FOOD.y * this.params.chemotaxisStrength;
      }
      
      // Danger avoidance
      if (perception.pheromoneGradients.DANGER) {
        totalGradX -= perception.pheromoneGradients.DANGER.x * 2.0; // Strong avoidance
        totalGradY -= perception.pheromoneGradients.DANGER.y * 2.0;
      }
    }
    
    // Normalize combined gradient
    const mag = Math.sqrt(totalGradX * totalGradX + totalGradY * totalGradY);
    if (mag > 0) {
      this.sensors.chemicalGradientX = totalGradX / mag;
      this.sensors.chemicalGradientY = totalGradY / mag;
    }
    
    // Damage sensor (from energy loss)
    // Will be set externally when organism takes damage
    
    // Neighbor density
    if (perception.organisms) {
      const nearbyCount = perception.organisms.filter(o => o.distance < 50).length;
      this.sensors.neighborDensity = Math.min(1, nearbyCount / 10);
    }
    
    // Pulse rhythm (internal oscillator)
    this.sensors.pulseRhythm = Math.sin(this.time * 2 * Math.PI / this.params.oscillatorPeriod);
  }
  
  /**
   * Main update step - apply CA rules
   */
  update(deltaTime) {
    this.time += deltaTime;
    this.oscillatorPhase = (this.oscillatorPhase + deltaTime) % this.params.oscillatorPeriod;
    
    // Copy current state to next
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        const next = this.nextGrid[i][j];
        
        // Copy base properties
        next.type = cell.type;
        next.energy = cell.energy;
        next.charge = cell.charge;
        next.phase = cell.phase;
        next.memoryTrace = cell.memoryTrace;
        next.heritageMarker = cell.heritageMarker;
        next.skinFormationPotential = cell.skinFormationPotential;
        next.structuralIntegrity = cell.structuralIntegrity;
        
        // Update timers
        next.refractoryTimer = Math.max(0, cell.refractoryTimer - 1);
        next.repairTimer = Math.max(0, cell.repairTimer - 1);
      }
    }
    
    // Apply rules
    this.applyPerceptionRules();
    this.applyHomeostasisRules();
    this.applyMorphogenesisRules();
    this.applyOscillatorRules();
    
    // Swap grids
    const temp = this.grid;
    this.grid = this.nextGrid;
    this.nextGrid = temp;
    
    // Calculate effector outputs
    this.calculateEffectors();
  }
  
  /**
   * Rule 1: Perception-oriented movement (chemotaxis/phototaxis)
   */
  applyPerceptionRules() {
    const centerX = this.size / 2;
    const centerY = this.size / 2;
    
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        const next = this.nextGrid[i][j];
        
        if (cell.type === 'void' || cell.refractoryTimer > 0) continue;
        
        // Position relative to center (normalized)
        const relX = (i - centerX) / centerX;
        const relY = (j - centerY) / centerY;
        
        // Chemical sensing - activate cells in gradient direction
        const chemAlignment = relX * this.sensors.chemicalGradientX + 
                            relY * this.sensors.chemicalGradientY;
        
        if (chemAlignment > 0.5) {
          next.activation = Math.min(1, cell.activation + 0.3 * this.params.chemotaxisStrength);
          next.chemicalSensor = chemAlignment;
          next.refractoryTimer = this.params.refractoryPeriod;
        }
        
        // Light avoidance - activate opposite cells
        if (this.sensors.lightLevel > 0.3) {
          const lightAlignment = -(relX * this.sensors.chemicalGradientX + 
                                 relY * this.sensors.chemicalGradientY);
          
          if (lightAlignment > 0.5) {
            next.activation = Math.min(1, cell.activation + 0.3 * this.params.phototaxisStrength);
            next.lightSensor = this.sensors.lightLevel;
            next.refractoryTimer = this.params.refractoryPeriod;
          }
        }
        
        // Damage response - emergency activation
        if (this.sensors.damage > 0) {
          if (cell.damageSensor > 0.5) {
            next.activation = 1.0;
            next.refractoryTimer = this.params.refractoryPeriod * 2;
          }
        }
      }
    }
  }
  
  /**
   * Rule 2: Homeostasis and self-repair
   */
  applyHomeostasisRules() {
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        const next = this.nextGrid[i][j];
        
        // Count neighbors by type
        const neighbors = this.getNeighbors(i, j);
        const healthyCount = neighbors.filter(n => n.type === 'tissue' && n.energy > 0.5).length;
        const skinCount = neighbors.filter(n => n.type === 'skin').length;
        const voidCount = neighbors.filter(n => n.type === 'void').length;
        const scarCount = neighbors.filter(n => n.type === 'scar').length;
        
        // Advanced repair rules
        if (cell.type === 'void') {
          // Void cells can regenerate based on neighbors
          if (healthyCount >= this.params.repairThreshold) {
            // Normal tissue regeneration
            next.type = 'tissue';
            next.energy = 0.5;
            next.repairTimer = 10;
            next.structuralIntegrity = 0.7;
          } else if (skinCount >= 2 && healthyCount >= 1) {
            // Scar formation near skin
            next.type = 'scar';
            next.energy = 0.3;
            next.structuralIntegrity = 0.5;
          }
        }
        
        // Scar tissue can slowly become normal tissue
        if (cell.type === 'scar') {
          next.repairTimer = Math.max(0, cell.repairTimer - 1);
          if (cell.repairTimer === 0 && healthyCount >= 4) {
            next.type = 'tissue';
            next.energy = 0.7;
            next.structuralIntegrity = 0.8;
          }
        }
        
        // Energy diffusion (improved)
        if (cell.type === 'tissue' || cell.type === 'skin') {
          let energySum = cell.energy;
          let count = 1;
          
          neighbors.forEach(n => {
            if (n.type === 'tissue' || n.type === 'skin') {
              // Skin cells share less energy
              const shareFactor = n.type === 'skin' ? 0.5 : 1.0;
              energySum += n.energy * shareFactor;
              count += shareFactor;
            }
          });
          
          // Conservative diffusion with structural integrity factor
          const avgEnergy = energySum / count;
          const diffusionEfficiency = cell.structuralIntegrity * this.params.diffusionRate;
          next.energy = cell.energy + (avgEnergy - cell.energy) * diffusionEfficiency;
          
          // Update structural integrity
          next.structuralIntegrity = Math.min(1.0, 
            cell.structuralIntegrity + (healthyCount / 8 - voidCount / 8) * 0.01
          );
          
          // Cell death from energy depletion or poor structure
          if (next.energy < 0.1 || next.structuralIntegrity < 0.2) {
            next.type = 'void';
            next.repairTimer = 5; // Harder to repair recently dead cells
          }
        }
      }
    }
  }
  
  /**
   * Rule 3: Morphogenesis - skin formation and structure
   */
  applyMorphogenesisRules() {
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        const next = this.nextGrid[i][j];
        
        if (cell.type === 'void' || cell.type === 'scar') continue;
        
        // Check position and neighbors
        const isBoundary = i === 0 || i === this.size - 1 || 
                          j === 0 || j === this.size - 1;
        const neighbors = this.getNeighbors(i, j);
        const tissueCount = neighbors.filter(n => n.type === 'tissue' || n.type === 'skin').length;
        const voidCount = neighbors.filter(n => n.type === 'void').length;
        const skinCount = neighbors.filter(n => n.type === 'skin').length;
        
        // Calculate skin formation potential
        const exposureLevel = voidCount / 8; // How exposed to void
        const supportLevel = tissueCount / 8; // How supported by tissue
        
        // Update skin formation potential
        next.skinFormationPotential = cell.skinFormationPotential * 0.9 + // Decay
          (exposureLevel * supportLevel * 2.0 + // High at interfaces
          (isBoundary ? 0.3 : 0)) * 0.1; // Boundary bonus
        
        // Dynamic skin formation
        if (cell.type === 'tissue') {
          // Form skin based on potential and context
          if (next.skinFormationPotential > 0.4 && exposureLevel > 0.2) {
            next.type = 'skin';
            next.charge = 0.5 + exposureLevel * 0.5; // Dynamic rigidity
            next.structuralIntegrity = Math.min(1.0, cell.structuralIntegrity + 0.2);
          }
          
          // Stress response - emergency skin formation
          if (cell.damageSensor > 0.3 && voidCount >= 3) {
            next.type = 'skin';
            next.charge = 1.0; // Maximum rigidity under stress
          }
        }
        
        // Skin maintenance and adaptation
        if (cell.type === 'skin') {
          // Dynamic rigidity based on local stress
          const stressFactor = (voidCount / 8) + (cell.damageSensor * 0.5);
          next.charge = Math.min(1.0, Math.max(0.3, 
            cell.charge * 0.95 + stressFactor * 0.1
          ));
          
          // Skin can revert to tissue if no longer needed
          if (voidCount === 0 && skinCount < 2 && cell.charge < 0.4) {
            next.type = 'tissue';
            next.skinFormationPotential = 0.2;
          }
          
          // Skin thickening near damage
          if (cell.damageSensor > 0) {
            next.structuralIntegrity = Math.min(1.0, cell.structuralIntegrity + 0.05);
          }
        }
      }
    }
  }
  
  /**
   * Rule 4: Central oscillator for coordinated movement
   */
  applyOscillatorRules() {
    // Create a ring oscillator in the center
    const centerX = Math.floor(this.size / 2);
    const centerY = Math.floor(this.size / 2);
    const radius = 3;
    
    for (let i = centerX - radius; i <= centerX + radius; i++) {
      for (let j = centerY - radius; j <= centerY + radius; j++) {
        if (i < 0 || i >= this.size || j < 0 || j >= this.size) continue;
        
        const dist = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
        if (dist >= radius - 0.5 && dist <= radius + 0.5) {
          const angle = Math.atan2(j - centerY, i - centerX);
          const phase = (angle + Math.PI) / (2 * Math.PI);
          
          // const cell = this.grid[i][j]; // Not needed - using next directly
          const next = this.nextGrid[i][j];
          
          // Oscillating activation
          const targetPhase = (this.oscillatorPhase / this.params.oscillatorPeriod + phase) % 1;
          if (targetPhase < 0.2) {
            next.activation = Math.max(next.activation, 0.8);
            next.phase = targetPhase;
          }
        }
      }
    }
  }
  
  /**
   * Calculate effector outputs from grid state
   */
  calculateEffectors() {
    // Reset effectors
    this.effectors.movementX = 0;
    this.effectors.movementY = 0;
    this.effectors.adhesion = 0;
    this.effectors.rigidity = 0;
    let activeCells = 0;
    
    const centerX = this.size / 2;
    const centerY = this.size / 2;
    
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        
        if (cell.activation > 0.1) {
          // Movement vector from activated peripheral cells
          const relX = (i - centerX) / centerX;
          const relY = (j - centerY) / centerY;
          const isPeripheral = Math.abs(relX) > 0.5 || Math.abs(relY) > 0.5;
          
          if (isPeripheral) {
            this.effectors.movementX += relX * cell.activation;
            this.effectors.movementY += relY * cell.activation;
            activeCells++;
          }
        }
        
        // Skin cells contribute to rigidity
        if (cell.type === 'skin') {
          this.effectors.rigidity += 0.1;
        }
        
        // Adhesion from charged cells
        if (cell.charge > 0.5) {
          this.effectors.adhesion += cell.charge * 0.1;
        }
      }
    }
    
    // Normalize movement
    if (activeCells > 0) {
      this.effectors.movementX /= activeCells;
      this.effectors.movementY /= activeCells;
    }
    
    // Cap values
    this.effectors.rigidity = Math.min(1, this.effectors.rigidity);
    this.effectors.adhesion = Math.min(1, this.effectors.adhesion);
    
    // Oscillation output from central pattern
    this.effectors.oscillation = this.sensors.pulseRhythm;
    
    // Chemical secretion based on state
    this.calculateChemicalSecretion();
  }
  
  /**
   * Calculate chemical secretion based on cellular state
   */
  calculateChemicalSecretion() {
    // Reset secretion
    this.effectors.chemicalSecretion = 0;
    
    // Count cell states
    let foodSensorCount = 0;
    let damageCount = 0;
    let skinCount = 0;
    let energyTotal = 0;
    let cellCount = 0;
    
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        
        if (cell.type !== 'void') {
          cellCount++;
          energyTotal += cell.energy;
          
          if (cell.chemicalSensor > 0.5) foodSensorCount++;
          if (cell.damageSensor > 0.1) damageCount++;
          if (cell.type === 'skin') skinCount++;
        }
      }
    }
    
    const avgEnergy = cellCount > 0 ? energyTotal / cellCount : 0;
    
    // Decide what to secrete based on state
    if (damageCount > 3 || this.sensors.damage > 0.5) {
      // Secrete danger pheromone when damaged
      this.effectors.secretionType = 'DANGER';
      this.effectors.chemicalSecretion = 0.5 + this.sensors.damage;
    } else if (foodSensorCount > 5 && avgEnergy > 0.7) {
      // Secrete food trail when well-fed and detecting food
      this.effectors.secretionType = 'FOOD';
      this.effectors.chemicalSecretion = 0.3 + (avgEnergy - 0.7);
    } else if (this.sensors.neighborDensity > 0.5 && skinCount > 10) {
      // Secrete colony pheromone when in group
      this.effectors.secretionType = 'COLONY';
      this.effectors.chemicalSecretion = 0.2 + this.sensors.neighborDensity * 0.3;
    } else if (avgEnergy > 0.9 && cellCount > 100) {
      // Ready to reproduce - mating signal
      this.effectors.secretionType = 'MATING';
      this.effectors.chemicalSecretion = 0.4;
    }
  }
  
  /**
   * Get neighbors of a cell (Moore neighborhood)
   */
  getNeighbors(x, y) {
    const neighbors = [];
    
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = x + dx;
        const ny = y + dy;
        
        if (nx >= 0 && nx < this.size && ny >= 0 && ny < this.size) {
          neighbors.push(this.grid[nx][ny]);
        }
      }
    }
    
    return neighbors;
  }
  
  /**
   * Inject damage at specific location
   */
  injectDamage(relativeX, relativeY, intensity) {
    const x = Math.floor((relativeX + 1) * this.size / 2);
    const y = Math.floor((relativeY + 1) * this.size / 2);
    const radius = 2;
    
    for (let i = x - radius; i <= x + radius; i++) {
      for (let j = y - radius; j <= y + radius; j++) {
        if (i >= 0 && i < this.size && j >= 0 && j < this.size) {
          const dist = Math.sqrt((i - x) ** 2 + (j - y) ** 2);
          if (dist <= radius) {
            const cell = this.grid[i][j];
            cell.damageSensor = intensity * (1 - dist / radius);
            cell.energy *= (1 - intensity * 0.5);
            
            // Severe damage can kill cells
            if (intensity > 0.8 && Math.random() < intensity) {
              cell.type = 'void';
            }
          }
        }
      }
    }
    
    this.sensors.damage = intensity;
  }
  
  /**
   * Inherit parameters and patterns from parent
   */
  inherit(parentAC, mutationRate = 0.1) {
    // Copy parameters with mutation
    for (const key in parentAC.params) {
      const mutation = (Math.random() - 0.5) * 2 * mutationRate;
      this.params[key] = parentAC.params[key] * (1 + mutation);
      
      // Clamp parameters to reasonable ranges
      if (key === 'chemotaxisStrength') this.params[key] = Math.max(-2, Math.min(2, this.params[key]));
      if (key === 'phototaxisStrength') this.params[key] = Math.max(-2, Math.min(0, this.params[key]));
      if (key === 'diffusionRate') this.params[key] = Math.max(0.01, Math.min(0.5, this.params[key]));
      if (key === 'oscillatorPeriod') this.params[key] = Math.max(5, Math.min(20, this.params[key]));
      if (key === 'repairThreshold') this.params[key] = Math.max(2, Math.min(5, this.params[key]));
    }
    
    // Inherit successful patterns with variation
    const patterns = this.extractSuccessfulPatterns(parentAC);
    this.applySeedPatterns(patterns, mutationRate);
  }
  
  /**
   * Extract successful patterns from parent
   */
  extractSuccessfulPatterns(parentAC) {
    const patterns = {
      core: [],      // Central oscillator pattern
      membrane: [],  // Skin/boundary pattern
      organs: []     // High-energy clusters
    };
    
    const centerX = Math.floor(this.size / 2);
    const centerY = Math.floor(this.size / 2);
    
    // Extract core pattern (central 6x6)
    for (let i = -3; i < 3; i++) {
      for (let j = -3; j < 3; j++) {
        const x = centerX + i;
        const y = centerY + j;
        if (x >= 0 && x < this.size && y >= 0 && y < this.size) {
          const cell = parentAC.grid[x][y];
          if (cell.type !== 'void' && (cell.phase > 0 || cell.activation > 0.5)) {
            patterns.core.push({
              x: i, y: j,
              type: cell.type,
              phase: cell.phase,
              heritageMarker: cell.heritageMarker
            });
          }
        }
      }
    }
    
    // Extract membrane pattern (skin cells)
    parentAC.grid.forEach((row, i) => {
      row.forEach((cell, j) => {
        if (cell.type === 'skin' && cell.structuralIntegrity > 0.7) {
          patterns.membrane.push({
            relX: (i - centerX) / centerX,
            relY: (j - centerY) / centerY,
            charge: cell.charge
          });
        }
      });
    });
    
    // Extract high-energy clusters
    parentAC.grid.forEach((row, i) => {
      row.forEach((cell, j) => {
        if (cell.energy > 0.8 && cell.type === 'tissue') {
          // Calculate average energy from neighbors manually to avoid context issues
          let neighborEnergy = 0;
          let neighborCount = 0;
          for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
              if (di === 0 && dj === 0) continue;
              const ni = i + di;
              const nj = j + dj;
              if (ni >= 0 && ni < parentAC.grid.length && nj >= 0 && nj < parentAC.grid[0].length) {
                neighborEnergy += parentAC.grid[ni][nj].energy;
                neighborCount++;
              }
            }
          }
          const avgEnergy = neighborCount > 0 ? neighborEnergy / neighborCount : 0;
          if (avgEnergy > 0.6) {
            patterns.organs.push({
              relX: (i - centerX) / centerX,
              relY: (j - centerY) / centerY,
              energy: cell.energy
            });
          }
        }
      });
    });
    
    return patterns;
  }
  
  /**
   * Apply seed patterns to new organism
   */
  applySeedPatterns(patterns, mutationRate) {
    const centerX = Math.floor(this.size / 2);
    const centerY = Math.floor(this.size / 2);
    
    // Apply core pattern with slight variations
    patterns.core.forEach(p => {
      const x = centerX + p.x + Math.round((Math.random() - 0.5) * 2 * mutationRate);
      const y = centerY + p.y + Math.round((Math.random() - 0.5) * 2 * mutationRate);
      
      if (x >= 0 && x < this.size && y >= 0 && y < this.size) {
        this.grid[x][y].type = p.type;
        this.grid[x][y].phase = p.phase * (1 + (Math.random() - 0.5) * mutationRate);
        this.grid[x][y].heritageMarker = p.heritageMarker + 1;
        this.grid[x][y].memoryTrace = 1.0;
      }
    });
    
    // Apply membrane pattern
    if (patterns.membrane.length > 0) {
      // Sample subset of membrane pattern
      const sampleSize = Math.min(patterns.membrane.length, 
        Math.floor(patterns.membrane.length * (1 - mutationRate)));
      
      for (let i = 0; i < sampleSize; i++) {
        const p = patterns.membrane[Math.floor(Math.random() * patterns.membrane.length)];
        const x = Math.floor(centerX + p.relX * centerX);
        const y = Math.floor(centerY + p.relY * centerY);
        
        if (x >= 0 && x < this.size && y >= 0 && y < this.size) {
          this.grid[x][y].skinFormationPotential = 0.6;
          this.grid[x][y].charge = p.charge * (1 - mutationRate * 0.5);
        }
      }
    }
    
    // Apply organ patterns
    patterns.organs.forEach(p => {
      if (Math.random() > mutationRate) { // Some organs may not be inherited
        const x = Math.floor(centerX + p.relX * centerX * (1 + (Math.random() - 0.5) * mutationRate));
        const y = Math.floor(centerY + p.relY * centerY * (1 + (Math.random() - 0.5) * mutationRate));
        
        if (x >= 1 && x < this.size - 1 && y >= 1 && y < this.size - 1) {
          // Create energy hotspot
          for (let dx = -1; dx <= 1; dx++) {
            for (let dy = -1; dy <= 1; dy++) {
              const dist = Math.sqrt(dx * dx + dy * dy);
              this.grid[x + dx][y + dy].energy = Math.min(1.0, 
                p.energy * (1 - dist * 0.3) * (1 - mutationRate * 0.2)
              );
            }
          }
        }
      }
    });
  }
  
  /**
   * Get state for visualization
   */
  getVisualizationData() {
    const data = [];
    
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const cell = this.grid[i][j];
        data.push({
          x: i,
          y: j,
          type: cell.type,
          activation: cell.activation,
          energy: cell.energy,
          charge: cell.charge,
          structuralIntegrity: cell.structuralIntegrity
        });
      }
    }
    
    return data;
  }
  
  /**
   * Desactiva una regla específica (para tests de ablación)
   */
  desactivarRegla(regla) {
    if (!this.reglasDesactivadas) {
      this.reglasDesactivadas = new Set();
    }
    this.reglasDesactivadas.add(regla);
  }
  
  /**
   * Reactiva todas las reglas
   */
  activarTodasLasReglas() {
    this.reglasDesactivadas = new Set();
  }
}