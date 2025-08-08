/**
 * Environmental Field - Stores collective memory traces
 * The world itself becomes a memory storage medium
 */

export class EnvironmentalField {
  constructor(worldSize) {
    this.worldSize = worldSize;
    
    // Grid-based pheromone field
    this.gridSize = 20; // Cell size
    this.gridWidth = Math.ceil(worldSize.width / this.gridSize);
    this.gridHeight = Math.ceil(worldSize.height / this.gridSize);
    
    // Multiple pheromone layers
    this.pheromones = {
      danger: this.createGrid(),
      food: this.createGrid(),
      mating: this.createGrid(),
      death: this.createGrid(),
      activity: this.createGrid()
    };
    
    // Memory anchors - strong persistent memories
    this.memoryAnchors = [];
    
    // Decay rates for different pheromones
    this.decayRates = {
      danger: 0.99,
      food: 0.98,
      mating: 0.97,
      death: 0.995,
      activity: 0.96
    };
  }
  
  createGrid() {
    const grid = [];
    for (let y = 0; y < this.gridHeight; y++) {
      grid[y] = new Float32Array(this.gridWidth);
    }
    return grid;
  }
  
  // Deposit trace from organism
  depositTrace(trace) {
    const { position, pheromone, intensity = 1.0 } = trace;
    
    // Validate intensity
    const safeIntensity = Number.isFinite(intensity) ? Math.min(10, Math.max(0, intensity)) : 1.0;
    
    // Convert position to grid coordinates
    const gx = Math.floor(position.x / this.gridSize);
    const gy = Math.floor(position.y / this.gridSize);
    
    if (gx < 0 || gx >= this.gridWidth || gy < 0 || gy >= this.gridHeight) {
      return;
    }
    
    // Deposit pheromones with clamping to prevent overflow
    if (pheromone[0] > 0) {
      this.pheromones.danger[gy][gx] = Math.min(10, this.pheromones.danger[gy][gx] + pheromone[0] * safeIntensity);
    }
    if (pheromone[1] > 0) {
      this.pheromones.food[gy][gx] = Math.min(10, this.pheromones.food[gy][gx] + pheromone[1] * safeIntensity);
    }
    if (pheromone[2] > 0) {
      this.pheromones.mating[gy][gx] = Math.min(10, this.pheromones.mating[gy][gx] + pheromone[2] * safeIntensity);
    }
    if (pheromone[3] > 0) {
      this.pheromones.death[gy][gx] = Math.min(10, this.pheromones.death[gy][gx] + pheromone[3] * safeIntensity);
    }
    if (pheromone[4] > 0) {
      this.pheromones.activity[gy][gx] = Math.min(10, this.pheromones.activity[gy][gx] + pheromone[4] * safeIntensity);
    }
    
    // Diffuse to neighboring cells
    this.diffuse(gx, gy, safeIntensity * 0.3);
  }
  
  diffuse(cx, cy, amount) {
    // Simple diffusion to 8 neighbors
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        
        const nx = cx + dx;
        const ny = cy + dy;
        
        if (nx >= 0 && nx < this.gridWidth && ny >= 0 && ny < this.gridHeight) {
          const distance = Math.sqrt(dx * dx + dy * dy);
          const diffuseAmount = amount / (distance * 2);
          
          for (const [type, grid] of Object.entries(this.pheromones)) {
            grid[ny][nx] += grid[cy][cx] * diffuseAmount;
          }
        }
      }
    }
  }
  
  // Add memory anchor for significant events
  addMemoryAnchor(anchor) {
    this.memoryAnchors.push({
      ...anchor,
      timestamp: Date.now(),
      strength: 1.0
    });
    
    // Limit anchors
    if (this.memoryAnchors.length > 100) {
      // Remove weakest anchors
      this.memoryAnchors.sort((a, b) => b.strength - a.strength);
      this.memoryAnchors = this.memoryAnchors.slice(0, 100);
    }
  }
  
  // Read environmental memory at position
  readEnvironment(position, radius = 50) {
    const gx = Math.floor(position.x / this.gridSize);
    const gy = Math.floor(position.y / this.gridSize);
    
    const memory = {
      danger: 0,
      food: 0,
      mating: 0,
      death: 0,
      activity: 0,
      nearbyAnchors: []
    };
    
    // Sample pheromones in radius
    const gridRadius = Math.ceil(radius / this.gridSize);
    let count = 0;
    
    for (let dy = -gridRadius; dy <= gridRadius; dy++) {
      for (let dx = -gridRadius; dx <= gridRadius; dx++) {
        const nx = gx + dx;
        const ny = gy + dy;
        
        if (nx >= 0 && nx < this.gridWidth && ny >= 0 && ny < this.gridHeight) {
          const dist = Math.sqrt(dx * dx + dy * dy) * this.gridSize;
          if (dist <= radius) {
            const weight = 1 - (dist / radius);
            
            memory.danger += this.pheromones.danger[ny][nx] * weight;
            memory.food += this.pheromones.food[ny][nx] * weight;
            memory.mating += this.pheromones.mating[ny][nx] * weight;
            memory.death += this.pheromones.death[ny][nx] * weight;
            memory.activity += this.pheromones.activity[ny][nx] * weight;
            
            count++;
          }
        }
      }
    }
    
    // Normalize
    if (count > 0) {
      memory.danger /= count;
      memory.food /= count;
      memory.mating /= count;
      memory.death /= count;
      memory.activity /= count;
    }
    
    // Find nearby anchors
    memory.nearbyAnchors = this.memoryAnchors.filter(anchor => {
      const dx = anchor.position.x - position.x;
      const dy = anchor.position.y - position.y;
      return Math.sqrt(dx * dx + dy * dy) < radius;
    });
    
    return memory;
  }
  
  // Update field over time
  update(deltaTime) {
    // Decay pheromones
    for (const [type, grid] of Object.entries(this.pheromones)) {
      const decay = Math.pow(this.decayRates[type], deltaTime);
      
      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          grid[y][x] *= decay;
          
          // Remove very small values
          if (grid[y][x] < 0.001) {
            grid[y][x] = 0;
          }
        }
      }
    }
    
    // Decay memory anchors
    this.memoryAnchors = this.memoryAnchors.filter(anchor => {
      anchor.strength *= Math.pow(0.999, deltaTime);
      return anchor.strength > 0.01;
    });
  }
  
  // Get visualization data
  getVisualizationData() {
    const data = [];
    
    for (let y = 0; y < this.gridHeight; y++) {
      for (let x = 0; x < this.gridWidth; x++) {
        const danger = this.pheromones.danger[y][x];
        const food = this.pheromones.food[y][x];
        const activity = this.pheromones.activity[y][x];
        
        // Validate values are finite
        const safeDanger = Number.isFinite(danger) ? Math.min(10, Math.max(0, danger)) : 0;
        const safeFood = Number.isFinite(food) ? Math.min(10, Math.max(0, food)) : 0;
        const safeActivity = Number.isFinite(activity) ? Math.min(10, Math.max(0, activity)) : 0;
        
        if (safeDanger > 0.1 || safeFood > 0.1 || safeActivity > 0.1) {
          data.push({
            x: x * this.gridSize + this.gridSize / 2,
            y: y * this.gridSize + this.gridSize / 2,
            danger: safeDanger,
            food: safeFood,
            activity: safeActivity,
            size: this.gridSize
          });
        }
      }
    }
    
    return data;
  }
}