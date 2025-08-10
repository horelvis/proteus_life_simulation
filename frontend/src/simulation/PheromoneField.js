/**
 * Pheromone Field - Global chemical communication system
 * Implements stigmergy (indirect coordination through environment modification)
 */

export class PheromoneField {
  constructor(worldSize, gridSize = 20) {
    this.worldSize = worldSize;
    this.gridSize = gridSize;
    
    // Grid dimensions
    this.gridWidth = Math.ceil(worldSize.width / gridSize);
    this.gridHeight = Math.ceil(worldSize.height / gridSize);
    
    // Pheromone types
    this.types = {
      FOOD: 0,      // Food trail (green)
      DANGER: 1,    // Danger/death (red)
      COLONY: 2,    // Colony/home (blue)
      MATING: 3     // Reproduction signal (purple)
    };
    
    // Initialize grids for each pheromone type
    this.grids = {};
    Object.keys(this.types).forEach(type => {
      this.grids[type] = this.createGrid();
    });
    
    // Decay and diffusion parameters
    this.decayRate = 0.98;      // Per frame decay
    this.diffusionRate = 0.1;   // Spatial spread
    this.evaporationThreshold = 0.01; // Below this, set to 0
  }
  
  createGrid() {
    const grid = [];
    for (let i = 0; i < this.gridHeight; i++) {
      grid[i] = new Float32Array(this.gridWidth);
    }
    return grid;
  }
  
  /**
   * Deposit pheromone at world position
   */
  deposit(x, y, type, intensity = 1.0) {
    const gridX = Math.floor(x / this.gridSize);
    const gridY = Math.floor(y / this.gridSize);
    
    if (gridX >= 0 && gridX < this.gridWidth && 
        gridY >= 0 && gridY < this.gridHeight &&
        this.grids[type]) {
      // Add intensity, capping at 10
      this.grids[type][gridY][gridX] = Math.min(10, 
        this.grids[type][gridY][gridX] + intensity
      );
    }
  }
  
  /**
   * Sample pheromone concentration at world position
   */
  sample(x, y, type) {
    const gridX = Math.floor(x / this.gridSize);
    const gridY = Math.floor(y / this.gridSize);
    
    if (gridX >= 0 && gridX < this.gridWidth && 
        gridY >= 0 && gridY < this.gridHeight &&
        this.grids[type]) {
      return this.grids[type][gridY][gridX];
    }
    return 0;
  }
  
  /**
   * Get gradient at position (for chemotaxis)
   */
  getGradient(x, y, type) {
    const h = this.gridSize; // Step size
    
    // Sample in four directions
    const right = this.sample(x + h, y, type);
    const left = this.sample(x - h, y, type);
    const up = this.sample(x, y - h, type);
    const down = this.sample(x, y + h, type);
    
    // Calculate gradient
    const gradX = (right - left) / (2 * h);
    const gradY = (down - up) / (2 * h);
    
    return { x: gradX, y: gradY };
  }
  
  /**
   * Get interpolated value (smoother than grid sampling)
   */
  sampleInterpolated(x, y, type) {
    const gridX = x / this.gridSize;
    const gridY = y / this.gridSize;
    
    const x0 = Math.floor(gridX);
    const x1 = Math.ceil(gridX);
    const y0 = Math.floor(gridY);
    const y1 = Math.ceil(gridY);
    
    // Bounds check
    if (x0 < 0 || x1 >= this.gridWidth || y0 < 0 || y1 >= this.gridHeight) {
      return 0;
    }
    
    // Bilinear interpolation
    const fx = gridX - x0;
    const fy = gridY - y0;
    
    const grid = this.grids[type];
    const v00 = grid[y0][x0];
    const v10 = x1 < this.gridWidth ? grid[y0][x1] : v00;
    const v01 = y1 < this.gridHeight ? grid[y1][x0] : v00;
    const v11 = (x1 < this.gridWidth && y1 < this.gridHeight) ? grid[y1][x1] : v00;
    
    const v0 = v00 * (1 - fx) + v10 * fx;
    const v1 = v01 * (1 - fx) + v11 * fx;
    
    return v0 * (1 - fy) + v1 * fy;
  }
  
  /**
   * Update pheromone field (decay and diffusion)
   */
  update(deltaTime) {
    Object.keys(this.types).forEach(type => {
      const grid = this.grids[type];
      const newGrid = this.createGrid();
      
      // Apply diffusion and decay
      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          let value = grid[y][x] * this.decayRate;
          
          // Diffusion to neighbors
          if (value > this.evaporationThreshold) {
            // Get neighbor values for diffusion
            let neighborSum = 0;
            let neighborCount = 0;
            
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue;
                
                const nx = x + dx;
                const ny = y + dy;
                
                if (nx >= 0 && nx < this.gridWidth && 
                    ny >= 0 && ny < this.gridHeight) {
                  neighborSum += grid[ny][nx];
                  neighborCount++;
                }
              }
            }
            
            // Apply diffusion
            if (neighborCount > 0) {
              const avgNeighbor = neighborSum / neighborCount;
              value = value * (1 - this.diffusionRate) + 
                     avgNeighbor * this.diffusionRate;
            }
          }
          
          // Apply threshold
          newGrid[y][x] = value > this.evaporationThreshold ? value : 0;
        }
      }
      
      // Swap grids
      this.grids[type] = newGrid;
    });
  }
  
  /**
   * Clear all pheromones
   */
  clear() {
    Object.keys(this.types).forEach(type => {
      this.grids[type] = this.createGrid();
    });
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData() {
    const data = [];
    
    Object.entries(this.types).forEach(([typeName, typeIndex]) => {
      const grid = this.grids[typeName];
      
      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          const value = grid[y][x];
          if (value > this.evaporationThreshold) {
            data.push({
              x: x * this.gridSize + this.gridSize / 2,
              y: y * this.gridSize + this.gridSize / 2,
              type: typeName,
              intensity: value / 10 // Normalize to 0-1
            });
          }
        }
      }
    });
    
    return data;
  }
  
  /**
   * Analyze pheromone patterns for metrics
   */
  analyzePatterns() {
    const analysis = {
      totalIntensity: {},
      coverage: {},
      clusters: {}
    };
    
    Object.entries(this.types).forEach(([typeName, typeIndex]) => {
      const grid = this.grids[typeName];
      let total = 0;
      let coverage = 0;
      
      for (let y = 0; y < this.gridHeight; y++) {
        for (let x = 0; x < this.gridWidth; x++) {
          const value = grid[y][x];
          if (value > 0) {
            total += value;
            coverage++;
          }
        }
      }
      
      analysis.totalIntensity[typeName] = total;
      analysis.coverage[typeName] = coverage / (this.gridWidth * this.gridHeight);
    });
    
    return analysis;
  }
}