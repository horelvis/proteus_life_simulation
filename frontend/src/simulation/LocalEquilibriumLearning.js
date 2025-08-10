/**
 * Local Equilibrium Learning for PROTEUS
 * 
 * Implements learning without backpropagation by adjusting parameters
 * based on local equilibrium states after survival episodes.
 * 
 * When an organism survives an episode, we freeze its quasi-stationary
 * state and apply local rules that maximize stability/robustness.
 */

export class LocalEquilibriumLearning {
  constructor() {
    // Learning parameters
    this.learningRate = 0.1;
    this.stabilityThreshold = 0.8;
    
    // Episode tracking
    this.currentEpisode = {
      startTime: Date.now(),
      states: [],
      survived: true,
      equilibriumState: null
    };
    
    // Learning history
    this.learningHistory = [];
    this.maxHistorySize = 100;
  }
  
  /**
   * Record state during episode
   * @param {Object} state - Current organism state including position, energy, sensory inputs
   */
  recordState(state) {
    this.currentEpisode.states.push({
      timestamp: Date.now(),
      position: { ...state.position },
      energy: state.energy,
      sensoryInputs: {
        lightLevel: state.lightLevel || 0,
        chemicalGradient: state.chemicalGradient ? { ...state.chemicalGradient } : { x: 0, y: 0 },
        predatorDistance: state.predatorDistance || Infinity
      },
      internalState: {
        intention: state.intention || 'explore',
        computationIterations: state.computationIterations || 1
      }
    });
    
    // Keep only recent states (last 100)
    if (this.currentEpisode.states.length > 100) {
      this.currentEpisode.states.shift();
    }
  }
  
  /**
   * Mark episode end and trigger learning if survived
   * @param {boolean} survived - Whether organism survived the episode
   * @param {Object} finalState - Final state at episode end
   * @param {Object} inheritance - Organism's inheritance to modify
   */
  endEpisode(survived, finalState, inheritance) {
    this.currentEpisode.survived = survived;
    this.currentEpisode.endTime = Date.now();
    
    if (survived && this.currentEpisode.states.length > 10) {
      // Find quasi-stationary equilibrium state
      this.currentEpisode.equilibriumState = this.findEquilibriumState();
      
      // Apply local learning rules
      this.applyLocalLearning(inheritance);
      
      // Store in history
      this.learningHistory.push({
        timestamp: Date.now(),
        survived: true,
        equilibrium: this.currentEpisode.equilibriumState,
        adjustments: this.lastAdjustments
      });
      
      if (this.learningHistory.length > this.maxHistorySize) {
        this.learningHistory.shift();
      }
    }
    
    // Reset for next episode
    this.currentEpisode = {
      startTime: Date.now(),
      states: [],
      survived: true,
      equilibriumState: null
    };
  }
  
  /**
   * Find quasi-stationary equilibrium from episode states
   * Uses the last stable period where state changes were minimal
   */
  findEquilibriumState() {
    const states = this.currentEpisode.states;
    if (states.length < 20) return states[states.length - 1];
    
    // Look for period of stability in last half of episode
    const recentStates = states.slice(-50);
    let mostStableIndex = recentStates.length - 1;
    let minVariance = Infinity;
    
    // Sliding window to find most stable period
    const windowSize = 10;
    for (let i = windowSize; i < recentStates.length; i++) {
      const window = recentStates.slice(i - windowSize, i);
      const variance = this.calculateStateVariance(window);
      
      if (variance < minVariance) {
        minVariance = variance;
        mostStableIndex = i - windowSize / 2;
      }
    }
    
    return recentStates[Math.floor(mostStableIndex)];
  }
  
  /**
   * Calculate variance in state window
   */
  calculateStateVariance(states) {
    if (states.length < 2) return 0;
    
    // Calculate variance in position
    const positions = states.map(s => s.position);
    const avgX = positions.reduce((sum, p) => sum + p.x, 0) / positions.length;
    const avgY = positions.reduce((sum, p) => sum + p.y, 0) / positions.length;
    
    const posVariance = positions.reduce((sum, p) => {
      const dx = p.x - avgX;
      const dy = p.y - avgY;
      return sum + dx * dx + dy * dy;
    }, 0) / positions.length;
    
    // Calculate variance in energy
    const energies = states.map(s => s.energy);
    const avgEnergy = energies.reduce((sum, e) => sum + e, 0) / energies.length;
    const energyVariance = energies.reduce((sum, e) => {
      const de = e - avgEnergy;
      return sum + de * de;
    }, 0) / energies.length;
    
    return posVariance + energyVariance * 100; // Weight energy variance higher
  }
  
  /**
   * Apply local learning rules based on equilibrium state
   * Modifies organism's inheritance parameters
   */
  applyLocalLearning(inheritance) {
    const equilibrium = this.currentEpisode.equilibriumState;
    if (!equilibrium) return;
    
    this.lastAdjustments = {
      chemicalSensitivity: 0,
      lightAvoidance: 0,
      frequency: {}
    };
    
    // Rule 1: If survived with high light exposure, reduce light sensitivity
    if (equilibrium.sensoryInputs.lightLevel > 0.5) {
      // Survived despite light - reduce fear response
      const adjustment = -this.learningRate * equilibrium.sensoryInputs.lightLevel;
      this.adjustLightSensitivity(inheritance, adjustment);
      this.lastAdjustments.lightAvoidance = adjustment;
    }
    
    // Rule 2: If found food before episode end, strengthen chemical following
    const recentStates = this.currentEpisode.states.slice(-20);
    const foundFood = recentStates.some(s => s.energy > equilibrium.energy);
    
    if (foundFood) {
      // Strengthen response to chemical gradients that preceded food
      const foodState = recentStates.find(s => s.energy > equilibrium.energy);
      const preFood = recentStates[recentStates.indexOf(foodState) - 1];
      
      if (preFood && preFood.sensoryInputs.chemicalGradient) {
        const gradientStrength = Math.sqrt(
          preFood.sensoryInputs.chemicalGradient.x ** 2 + 
          preFood.sensoryInputs.chemicalGradient.y ** 2
        );
        
        const adjustment = this.learningRate * gradientStrength;
        this.adjustChemicalSensitivity(inheritance, adjustment);
        this.lastAdjustments.chemicalSensitivity = adjustment;
      }
    }
    
    // Rule 3: Frequency-specific adjustments in holographic memory
    this.adjustHolographicFrequencies(inheritance, equilibrium);
  }
  
  /**
   * Adjust light sensitivity in topological core
   */
  adjustLightSensitivity(inheritance, adjustment) {
    if (!inheritance.topologicalCore) return;
    
    // Modify base sensitivity with bounds
    const current = inheritance.topologicalCore.baseSensitivity;
    inheritance.topologicalCore.baseSensitivity = Math.max(0.1, Math.min(1.0, 
      current + adjustment
    ));
  }
  
  /**
   * Adjust chemical sensitivity
   */
  adjustChemicalSensitivity(inheritance, adjustment) {
    if (!inheritance.topologicalCore) return;
    
    // Increase motility as proxy for chemical following
    const current = inheritance.topologicalCore.baseMotility;
    inheritance.topologicalCore.baseMotility = Math.max(0.1, Math.min(1.0,
      current + adjustment * 0.5 // Half rate for motility
    ));
  }
  
  /**
   * Adjust specific frequencies in holographic memory
   * Reduce frequencies associated with damage, enhance those with success
   */
  adjustHolographicFrequencies(inheritance, equilibrium) {
    if (!inheritance.holographicMemory) return;
    
    // Get current frequency spectrum
    const memory = inheritance.holographicMemory;
    // const compressed = memory.compress(); // Not used directly
    
    // Identify frequencies active during high stress
    const stressStates = this.currentEpisode.states.filter(s => 
      s.sensoryInputs.predatorDistance < 50 || s.sensoryInputs.lightLevel > 0.7
    );
    
    if (stressStates.length > 0) {
      // Suppress frequencies associated with stress if we survived
      stressStates.forEach(state => {
        // Create synthetic experience to modify memory
        const suppressionExperience = {
          type: 'stress_suppression',
          position: state.position,
          trajectory: [state.position],
          importance: -0.2, // Negative importance to suppress
          emotionalValue: 'calm'
        };
        
        memory.encode(suppressionExperience);
      });
      
      this.lastAdjustments.frequency.stressSuppressed = stressStates.length;
    }
    
    // Enhance frequencies associated with energy gain
    const gainStates = this.currentEpisode.states.filter((s, i) => 
      i > 0 && s.energy > this.currentEpisode.states[i-1].energy
    );
    
    if (gainStates.length > 0) {
      gainStates.forEach(state => {
        const enhancementExperience = {
          type: 'reward_enhancement',
          position: state.position,
          trajectory: [state.position],
          importance: 0.5,
          emotionalValue: 'attraction'
        };
        
        memory.encode(enhancementExperience);
      });
      
      this.lastAdjustments.frequency.rewardEnhanced = gainStates.length;
    }
  }
  
  /**
   * Get learning metrics for analysis
   */
  getMetrics() {
    const totalEpisodes = this.learningHistory.length;
    const successfulLearning = this.learningHistory.filter(h => 
      h.adjustments && (
        Math.abs(h.adjustments.chemicalSensitivity) > 0.01 ||
        Math.abs(h.adjustments.lightAvoidance) > 0.01 ||
        Object.keys(h.adjustments.frequency).length > 0
      )
    ).length;
    
    return {
      totalEpisodes,
      successfulLearning,
      learningRate: totalEpisodes > 0 ? successfulLearning / totalEpisodes : 0,
      lastAdjustments: this.lastAdjustments || null
    };
  }
}