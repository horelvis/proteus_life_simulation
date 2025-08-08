/**
 * PROTEUS Holographic Memory System
 * Each point contains information about the whole experience
 * Robust to damage, allows superposition of memories
 */

export class HolographicMemory {
  constructor(size = 1000, parent1 = null, parent2 = null) {
    this.size = size;
    
    if (parent1 && parent2) {
      // Inherit through interference pattern
      this.interferencePattern = this.createInterference(parent1, parent2);
      this.phaseRelations = this.blendPhases(parent1.phaseRelations, parent2.phaseRelations);
    } else if (parent1) {
      // Single parent - mutate slightly
      this.interferencePattern = this.mutatePattern(parent1.interferencePattern);
      this.phaseRelations = this.mutatePattern(parent1.phaseRelations);
    } else {
      // Genesis - random initialization
      this.interferencePattern = new Float32Array(size);
      this.phaseRelations = new Float32Array(size);
      this.initializeRandom();
    }
    
    // Compressed history of critical moments
    this.criticalMoments = [];
    this.memoryStrength = 1.0;
  }
  
  initializeRandom() {
    for (let i = 0; i < this.size; i++) {
      this.interferencePattern[i] = Math.random() * 2 - 1;
      this.phaseRelations[i] = Math.random() * Math.PI * 2;
    }
  }
  
  // Create interference pattern from two parents
  createInterference(parent1, parent2) {
    const pattern = new Float32Array(this.size);
    const p1 = parent1.interferencePattern;
    const p2 = parent2.interferencePattern;
    
    for (let i = 0; i < this.size; i++) {
      // Not simple average - true wave interference
      const phase1 = parent1.phaseRelations[i];
      const phase2 = parent2.phaseRelations[i];
      
      // Constructive and destructive interference
      pattern[i] = p1[i] * Math.cos(phase1) + 
                   p2[i] * Math.cos(phase2) + 
                   2 * Math.sqrt(Math.abs(p1[i] * p2[i])) * Math.cos(phase1 - phase2);
      
      // Normalize
      pattern[i] = Math.tanh(pattern[i] * 0.5);
    }
    
    return pattern;
  }
  
  blendPhases(phases1, phases2) {
    const blended = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      // Circular mean for phases
      const x = Math.cos(phases1[i]) + Math.cos(phases2[i]);
      const y = Math.sin(phases1[i]) + Math.sin(phases2[i]);
      blended[i] = Math.atan2(y, x);
    }
    return blended;
  }
  
  mutatePattern(pattern, mutationRate = 0.1) {
    const mutated = new Float32Array(pattern);
    for (let i = 0; i < pattern.length; i++) {
      if (Math.random() < mutationRate) {
        mutated[i] += (Math.random() - 0.5) * 0.2;
        mutated[i] = Math.tanh(mutated[i]);
      }
    }
    return mutated;
  }
  
  // Encode an experience into the hologram
  encode(experience) {
    // Extract frequency components from experience
    const { trajectory, importance, emotionalValue, survivalValue } = experience;
    
    // Simple frequency extraction (in real implementation, use FFT)
    const frequencies = this.extractFrequencies(trajectory);
    
    // Update interference pattern
    const learningRate = 0.1 * importance;
    for (let i = 0; i < this.size; i++) {
      // Each point gets contribution from ALL frequencies
      let contribution = 0;
      for (let f = 0; f < frequencies.length; f++) {
        contribution += frequencies[f] * Math.cos(f * i * 0.01 + this.phaseRelations[i]);
      }
      
      // Update with momentum
      this.interferencePattern[i] = 
        (1 - learningRate) * this.interferencePattern[i] + 
        learningRate * Math.tanh(contribution);
    }
    
    // Store critical moments
    if (importance > 0.7) {
      this.criticalMoments.push({
        timestamp: Date.now(),
        topology: this.extractTopology(experience),
        impact: importance,
        type: experience.type
      });
      
      // Keep only most recent critical moments
      if (this.criticalMoments.length > 20) {
        this.criticalMoments.shift();
      }
    }
  }
  
  extractFrequencies(trajectory) {
    // Simplified frequency extraction
    const frequencies = new Float32Array(10);
    
    if (!trajectory || trajectory.length === 0) {
      return frequencies;
    }
    
    // Extract basic frequency components
    for (let f = 0; f < 10; f++) {
      let sum = 0;
      for (let i = 0; i < trajectory.length; i++) {
        const point = trajectory[i];
        sum += Math.sqrt(point.x * point.x + point.y * point.y) * 
               Math.cos(2 * Math.PI * f * i / trajectory.length);
      }
      frequencies[f] = sum / trajectory.length;
    }
    
    return frequencies;
  }
  
  extractTopology(experience) {
    const { trajectory, predatorDistance, nutrientDirection } = experience;
    
    // Extract topological invariants
    return {
      curvature: this.calculateCurvature(trajectory),
      windingNumber: this.calculateWindingNumber(trajectory),
      persistence: this.calculatePersistence(trajectory),
      danger: predatorDistance ? 1 / predatorDistance : 0,
      attraction: nutrientDirection ? Math.atan2(nutrientDirection.y, nutrientDirection.x) : 0
    };
  }
  
  calculateCurvature(trajectory) {
    if (!trajectory || trajectory.length < 3) return 0;
    
    // Average curvature along path
    let totalCurvature = 0;
    for (let i = 1; i < trajectory.length - 1; i++) {
      const p1 = trajectory[i - 1];
      const p2 = trajectory[i];
      const p3 = trajectory[i + 1];
      
      const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
      const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
      
      const cross = v1.x * v2.y - v1.y * v2.x;
      const dot = v1.x * v2.x + v1.y * v2.y;
      
      totalCurvature += Math.atan2(cross, dot);
    }
    
    return totalCurvature / (trajectory.length - 2);
  }
  
  calculateWindingNumber(trajectory) {
    if (!trajectory || trajectory.length < 2) return 0;
    
    let totalAngle = 0;
    for (let i = 0; i < trajectory.length - 1; i++) {
      const p1 = trajectory[i];
      const p2 = trajectory[i + 1];
      
      const angle1 = Math.atan2(p1.y, p1.x);
      const angle2 = Math.atan2(p2.y, p2.x);
      
      let dAngle = angle2 - angle1;
      while (dAngle > Math.PI) dAngle -= 2 * Math.PI;
      while (dAngle < -Math.PI) dAngle += 2 * Math.PI;
      
      totalAngle += dAngle;
    }
    
    return totalAngle / (2 * Math.PI);
  }
  
  calculatePersistence(trajectory) {
    // Simplified persistence - how long the organism maintained a direction
    if (!trajectory || trajectory.length < 2) return 0;
    
    let persistence = 0;
    let currentDirection = null;
    
    for (let i = 1; i < trajectory.length; i++) {
      const dx = trajectory[i].x - trajectory[i - 1].x;
      const dy = trajectory[i].y - trajectory[i - 1].y;
      const direction = Math.atan2(dy, dx);
      
      if (currentDirection !== null) {
        const similarity = Math.cos(direction - currentDirection);
        persistence += similarity > 0.8 ? 1 : 0;
      }
      currentDirection = direction;
    }
    
    return persistence / trajectory.length;
  }
  
  // Recall behavior from holographic memory
  recall(context = {}) {
    const { danger, hunger, energy } = context;
    
    // Use interference pattern to generate behavior
    const behavior = {
      explorationTendency: 0,
      cautionLevel: 0,
      forageIntensity: 0,
      socialAttraction: 0
    };
    
    // Different regions of the hologram encode different behaviors
    const regions = {
      exploration: { start: 0, end: 250 },
      caution: { start: 250, end: 500 },
      foraging: { start: 500, end: 750 },
      social: { start: 750, end: 1000 }
    };
    
    // Extract behavioral tendencies from interference pattern
    for (const [behaviorType, region] of Object.entries(regions)) {
      let sum = 0;
      for (let i = region.start; i < region.end; i++) {
        sum += this.interferencePattern[i] * Math.cos(this.phaseRelations[i]);
      }
      behavior[behaviorType + 'Tendency'] = Math.tanh(sum / (region.end - region.start));
    }
    
    // Modulate by current context
    if (danger > 0.5) {
      behavior.cautionLevel *= 2;
      behavior.explorationTendency *= 0.5;
    }
    
    if (hunger > 0.7) {
      behavior.forageIntensity *= 2;
    }
    
    return behavior;
  }
  
  // Calculate similarity between two memories
  similarity(other) {
    let correlation = 0;
    for (let i = 0; i < this.size; i++) {
      correlation += this.interferencePattern[i] * other.interferencePattern[i];
    }
    return correlation / this.size;
  }
  
  // Recover from partial damage
  recover(damageRatio = 0.5) {
    // Even with 50% damage, holographic memory can recover
    const damaged = [...this.interferencePattern];
    const damagedIndices = new Set();
    
    // Simulate damage
    for (let i = 0; i < this.size * damageRatio; i++) {
      const idx = Math.floor(Math.random() * this.size);
      damaged[idx] = 0;
      damagedIndices.add(idx);
    }
    
    // Holographic recovery using undamaged parts
    for (const idx of damagedIndices) {
      let reconstructed = 0;
      let count = 0;
      
      // Each undamaged point contributes to reconstruction
      for (let i = 0; i < this.size; i++) {
        if (!damagedIndices.has(i)) {
          const phase = this.phaseRelations[i] - this.phaseRelations[idx];
          reconstructed += damaged[i] * Math.cos(phase);
          count++;
        }
      }
      
      if (count > 0) {
        this.interferencePattern[idx] = reconstructed / count;
      }
    }
  }
  
  compress() {
    // Return only the most significant frequencies
    const compressed = {
      dominantFrequencies: [],
      criticalMoments: this.criticalMoments.slice(-5),
      signature: new Float32Array(20)
    };
    
    // Extract dominant frequencies
    for (let f = 0; f < 20; f++) {
      let power = 0;
      for (let i = 0; i < this.size; i++) {
        power += this.interferencePattern[i] * Math.cos(f * i * 0.01);
      }
      compressed.signature[f] = power / this.size;
    }
    
    return compressed;
  }
}