/**
 * Hierarchical Control System for Topological Dynamics
 * 
 * Implements two-scale control without neurons:
 * - Slow field: Intentional/navigational objectives (updates every ~100-500ms)
 * - Fast field: Reflexive responses (updates every frame ~16ms)
 * 
 * The slow field modulates fast field parameters, creating hierarchical control
 * through coupled field dynamics, not neural networks.
 */

export class HierarchicalControl {
  constructor(worldSize) {
    this.worldSize = worldSize;
    
    // Slow intentional field (navigation/objectives)
    this.slowField = {
      // Current high-level intention (explore, forage, flee, group)
      intention: 'explore',
      
      // Target direction bias (can be null for undirected exploration)
      targetBias: null,
      
      // Field update period in milliseconds
      updatePeriod: 200, // 5Hz update rate
      
      // Time since last update
      timeSinceUpdate: 0,
      
      // Modulation parameters for fast field
      modulation: {
        chemicalSensitivity: 1.0,
        lightAvoidance: 1.0,
        socialAttraction: 1.0,
        explorationNoise: 0.3
      }
    };
    
    // Fast reactive field (reflexes)
    this.fastField = {
      // Immediate sensory responses
      chemicalGradient: { x: 0, y: 0 },
      lightGradient: { x: 0, y: 0 },
      predatorVector: { x: 0, y: 0 },
      socialVector: { x: 0, y: 0 },
      
      // Combined response vector
      responseVector: { x: 0, y: 0 },
      
      // Convergence tracking
      previousVector: { x: 0, y: 0 },
      convergenceThreshold: 0.01,
      iterationCount: 0,
      maxIterations: 5 // Reduced for performance
    };
    
    // Adaptive computation time tracking
    this.computationMetrics = {
      averageIterations: 0,
      totalDecisions: 0,
      difficultScenes: 0 // Scenes requiring max iterations
    };
  }
  
  /**
   * Update hierarchical control system
   * @param {number} deltaTime - Time since last update in seconds
   * @param {Object} perception - Current sensory perception
   * @param {Object} memory - Organism's holographic memory recall
   * @returns {Object} Movement decision vector
   */
  update(deltaTime, perception, memory) {
    // Update slow field if enough time has passed
    this.slowField.timeSinceUpdate += deltaTime * 1000; // Convert to ms
    
    if (this.slowField.timeSinceUpdate >= this.slowField.updatePeriod) {
      this.updateSlowField(perception, memory);
      this.slowField.timeSinceUpdate = 0;
      
      // Reset fast field after slow field update (thermal noise injection)
      this.resetFastField();
    }
    
    // Always update fast field with adaptive computation time
    return this.updateFastFieldAdaptive(perception);
  }
  
  /**
   * Update slow intentional field based on memory and current state
   */
  updateSlowField(perception, memory) {
    const { energy } = perception;
    
    // Determine high-level intention based on state and memory
    if (energy < 0.3) {
      this.slowField.intention = 'forage';
      this.slowField.modulation.chemicalSensitivity = 2.0; // Increase food sensitivity
      this.slowField.modulation.lightAvoidance = 0.5; // Risk more for food
      this.slowField.modulation.socialAttraction = 0.2; // Less grouping when hungry
    } else if (perception.predatorNearby) {
      this.slowField.intention = 'flee';
      this.slowField.modulation.lightAvoidance = 3.0; // Maximum avoidance
      this.slowField.modulation.chemicalSensitivity = 0.2; // Ignore food
      this.slowField.modulation.socialAttraction = 1.5; // Seek group protection
    } else if (memory.socialAttraction > 0.7 && perception.sameSpeciesNearby) {
      this.slowField.intention = 'group';
      this.slowField.modulation.socialAttraction = 2.5;
      this.slowField.modulation.explorationNoise = 0.1; // Reduce random movement
    } else {
      this.slowField.intention = 'explore';
      this.slowField.modulation = {
        chemicalSensitivity: 1.0,
        lightAvoidance: 1.0,
        socialAttraction: 1.0,
        explorationNoise: 0.5 + memory.explorationTendency * 0.5
      };
    }
    
    // Set target bias based on memory of successful locations
    if (memory.successfulLocations && memory.successfulLocations.length > 0) {
      // Bias towards remembered successful areas
      const target = memory.successfulLocations[0]; // Most recent success
      this.slowField.targetBias = {
        x: target.x - perception.position.x,
        y: target.y - perception.position.y
      };
      
      // Normalize
      const mag = Math.sqrt(this.slowField.targetBias.x ** 2 + this.slowField.targetBias.y ** 2);
      if (mag > 0) {
        this.slowField.targetBias.x /= mag;
        this.slowField.targetBias.y /= mag;
      }
    } else {
      this.slowField.targetBias = null;
    }
  }
  
  /**
   * Reset fast field with controlled thermal noise
   */
  resetFastField() {
    // Add thermal noise to break out of local attractors
    const noiseScale = 0.1;
    
    this.fastField.responseVector = {
      x: (Math.random() - 0.5) * noiseScale,
      y: (Math.random() - 0.5) * noiseScale
    };
    
    // Reset convergence tracking
    this.fastField.iterationCount = 0;
    this.fastField.previousVector = { x: 0, y: 0 };
  }
  
  /**
   * Update fast field with adaptive computation time
   * Iterates until convergence or max iterations
   */
  updateFastFieldAdaptive(perception) {
    const mod = this.slowField.modulation;
    let converged = false;
    
    // Reset iteration count
    this.fastField.iterationCount = 0;
    
    while (!converged && this.fastField.iterationCount < this.fastField.maxIterations) {
      // Store previous vector
      this.fastField.previousVector.x = this.fastField.responseVector.x;
      this.fastField.previousVector.y = this.fastField.responseVector.y;
      
      // Update sensory gradients with modulation from slow field
      this.fastField.chemicalGradient = {
        x: perception.chemicalGradient.x * mod.chemicalSensitivity,
        y: perception.chemicalGradient.y * mod.chemicalSensitivity
      };
      
      this.fastField.lightGradient = {
        x: -perception.lightGradient.x * mod.lightAvoidance, // Negative for avoidance
        y: -perception.lightGradient.y * mod.lightAvoidance
      };
      
      this.fastField.predatorVector = {
        x: -perception.predatorDirection.x * mod.lightAvoidance * 2, // Use light avoidance for predators
        y: -perception.predatorDirection.y * mod.lightAvoidance * 2
      };
      
      this.fastField.socialVector = {
        x: perception.speciesDirection.x * mod.socialAttraction,
        y: perception.speciesDirection.y * mod.socialAttraction
      };
      
      // Combine all vectors
      this.fastField.responseVector = {
        x: this.fastField.chemicalGradient.x + 
           this.fastField.lightGradient.x + 
           this.fastField.predatorVector.x + 
           this.fastField.socialVector.x,
        y: this.fastField.chemicalGradient.y + 
           this.fastField.lightGradient.y + 
           this.fastField.predatorVector.y + 
           this.fastField.socialVector.y
      };
      
      // Add target bias if present
      if (this.slowField.targetBias) {
        this.fastField.responseVector.x += this.slowField.targetBias.x * 0.3;
        this.fastField.responseVector.y += this.slowField.targetBias.y * 0.3;
      }
      
      // Add exploration noise
      this.fastField.responseVector.x += (Math.random() - 0.5) * mod.explorationNoise;
      this.fastField.responseVector.y += (Math.random() - 0.5) * mod.explorationNoise;
      
      // Ensure some minimum movement to avoid getting stuck
      if (Math.abs(this.fastField.responseVector.x) < 0.1 && Math.abs(this.fastField.responseVector.y) < 0.1) {
        this.fastField.responseVector.x += (Math.random() - 0.5) * 0.5;
        this.fastField.responseVector.y += (Math.random() - 0.5) * 0.5;
      }
      
      // Check convergence
      const dx = this.fastField.responseVector.x - this.fastField.previousVector.x;
      const dy = this.fastField.responseVector.y - this.fastField.previousVector.y;
      const change = Math.sqrt(dx * dx + dy * dy);
      
      // Force convergence after 3 iterations to prevent blocking
      if (this.fastField.iterationCount >= 3) {
        converged = true;
      } else {
        converged = change < this.fastField.convergenceThreshold;
      }
      this.fastField.iterationCount++;
    }
    
    // Update computation metrics
    this.computationMetrics.totalDecisions++;
    this.computationMetrics.averageIterations = 
      (this.computationMetrics.averageIterations * (this.computationMetrics.totalDecisions - 1) + 
       this.fastField.iterationCount) / this.computationMetrics.totalDecisions;
    
    if (this.fastField.iterationCount === this.fastField.maxIterations) {
      this.computationMetrics.difficultScenes++;
    }
    
    return {
      decision: this.fastField.responseVector,
      intention: this.slowField.intention,
      iterations: this.fastField.iterationCount,
      converged: converged
    };
  }
  
  /**
   * Get current control state for debugging/visualization
   */
  getState() {
    return {
      slowField: {
        intention: this.slowField.intention,
        modulation: { ...this.slowField.modulation },
        targetBias: this.slowField.targetBias ? { ...this.slowField.targetBias } : null
      },
      fastField: {
        responseVector: { ...this.fastField.responseVector },
        iterationCount: this.fastField.iterationCount
      },
      metrics: {
        averageIterations: this.computationMetrics.averageIterations.toFixed(2),
        difficultScenesRatio: (this.computationMetrics.difficultScenes / 
                              Math.max(1, this.computationMetrics.totalDecisions)).toFixed(3)
      }
    };
  }
  
  /**
   * Reset all fields and metrics
   */
  reset() {
    this.slowField.intention = 'explore';
    this.slowField.targetBias = null;
    this.slowField.timeSinceUpdate = 0;
    this.slowField.modulation = {
      chemicalSensitivity: 1.0,
      lightAvoidance: 1.0,
      socialAttraction: 1.0,
      explorationNoise: 0.3
    };
    
    this.resetFastField();
    
    this.computationMetrics = {
      averageIterations: 0,
      totalDecisions: 0,
      difficultScenes: 0
    };
  }
}