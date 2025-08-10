/**
 * PROTEUS Tri-Layer Inheritance System
 * Layer 1: Topological Core (like DNA) - Immutable, defines species
 * Layer 2: Holographic Signature (like Epigenetics) - Experience-modifiable, inheritable
 * Layer 3: Environmental Traces (like Culture) - World-stored information
 */

import { HolographicMemory } from './HolographicMemory';

export class ProteusInheritance {
  constructor(parent1 = null, parent2 = null) {
    // LAYER 1: TOPOLOGICAL CORE (200 bytes)
    // Immutable parameters that define the "species"
    this.topologicalCore = this.inheritCore(parent1, parent2);
    
    // LAYER 2: HOLOGRAPHIC SIGNATURE (8KB)
    // Modifiable by experience, inheritable
    this.holographicMemory = new HolographicMemory(
      2000, 
      parent1?.holographicMemory,
      parent2?.holographicMemory
    );
    
    // LAYER 3: ENVIRONMENTAL TRACES
    // Information deposited in the world
    this.environmentalTraces = {
      pheromoneSignature: this.generatePheromoneSignature(),
      fieldModulations: [],
      anchorPoints: [],
      generation: parent1 ? parent1.environmentalTraces.generation + 1 : 0
    };
    
    // Learn from parent traces if available
    if (parent1) {
      this.learnFromEnvironment(parent1.environmentalTraces);
    }
    if (parent2) {
      this.learnFromEnvironment(parent2.environmentalTraces);
    }
    
    // Behavioral expression
    this.phenotype = this.expressPhenotype();
  }
  
  inheritCore(parent1, parent2) {
    if (!parent1) {
      // Genesis - create random core
      return {
        // Fundamental topological parameters
        manifoldDimension: 2 + Math.random() * 0.7,
        curvatureTensor: Array(4).fill(0).map(() => Math.random() - 0.5),
        bettiNumbers: [1, Math.floor(Math.random() * 3), 0],
        fundamentalGroup: ['Z', 'Z2', 'S1'][Math.floor(Math.random() * 3)],
        
        // Morphological parameters
        bodySymmetry: Math.floor(Math.random() * 4) + 1, // 1-4 fold symmetry
        organCapacity: Math.floor(Math.random() * 3) + 2, // 2-4 organs max
        
        // Behavioral base parameters - ensure viable starting values
        baseMotility: 0.5 + Math.random() * 0.3,      // 0.5-0.8 (was 0.3-0.7)
        baseSensitivity: 0.4 + Math.random() * 0.4,   // 0.4-0.8 (was 0.2-0.8)
        baseResilience: 0.5 + Math.random() * 0.3,    // 0.5-0.8 (was 0.4-0.8)
        
        // Mutation rate - higher for faster speciation
        mutability: 0.10 + Math.random() * 0.20  // 10-30% mutation rate for more diversity
      };
    }
    
    // EVOLUTION: Controlled mutations for gradual adaptation
    const core = {};
    const mutationRate = parent1.topologicalCore.mutability; // Use inherited rate (5-15%)
    
    // Copy from parent
    for (const key in parent1.topologicalCore) {
      if (parent2 && Math.random() < 0.5) {
        core[key] = parent2.topologicalCore[key];
      } else {
        core[key] = parent1.topologicalCore[key];
      }
      
      // Mutate based on mutation rate
      if (Math.random() < mutationRate) {
        core[key] = this.mutateParameter(key, core[key]);
      }
    }
    
    // Always has mutation now
    core.hasMutation = true;
    
    return core;
  }
  
  mutateParameter(key, value) {
    switch(key) {
      case 'manifoldDimension':
        return Math.max(1.5, Math.min(3, value + (Math.random() - 0.5) * 0.3));
      
      case 'curvatureTensor':
        return value.map(v => v + (Math.random() - 0.5) * 0.2);
      
      case 'bettiNumbers':
        const mutated = [...value];
        const idx = Math.floor(Math.random() * mutated.length);
        mutated[idx] = Math.max(0, mutated[idx] + (Math.random() < 0.5 ? -1 : 1));
        return mutated;
      
      case 'bodySymmetry':
        // Bigger changes in symmetry for visible mutations and speciation
        const change = Math.random() < 0.5 ? 1 : 2; // More likely to change by 2
        return Math.max(1, Math.min(6, value + (Math.random() < 0.5 ? -change : change)));
      
      case 'organCapacity':
        return Math.max(1, Math.min(5, value + (Math.random() < 0.5 ? -1 : 1)));
      
      case 'baseMotility':
      case 'baseSensitivity':
      case 'baseResilience':
        // Much larger mutations for rapid trait evolution
        const traitChange = (Math.random() - 0.5) * 0.5; // ±0.25 range
        return Math.max(0.1, Math.min(1, value + traitChange));
      
      case 'mutability':
        // Allow mutability to evolve more dramatically
        return Math.max(0.1, Math.min(0.5, value * (0.5 + Math.random() * 1.0)));
      
      default:
        return value;
    }
  }
  
  generatePheromoneSignature() {
    // Unique chemical signature based on core + memory
    const signature = new Float32Array(10);
    
    // Core contribution
    signature[0] = this.topologicalCore.manifoldDimension;
    signature[1] = this.topologicalCore.bodySymmetry;
    
    // Memory contribution (first few frequencies)
    const memoryFreqs = this.holographicMemory.compress().signature;
    for (let i = 0; i < 8 && i < memoryFreqs.length; i++) {
      signature[i + 2] = memoryFreqs[i];
    }
    
    return signature;
  }
  
  // Experience modifies holographic memory AND leaves environmental traces
  experience(event) {
    // Update holographic memory
    this.holographicMemory.encode(event);
    
    // Leave trace in environment
    this.depositTrace(event);
    
    // Update phenotype expression
    this.phenotype = this.expressPhenotype();
  }
  
  depositTrace(event) {
    // Handle position that might be array or object
    let position = event.position;
    if (Array.isArray(position)) {
      position = { x: position[0], y: position[1] };
    }
    
    // Validate position
    if (!position || !Number.isFinite(position.x) || !Number.isFinite(position.y)) {
      console.warn('Invalid position in event:', event);
      return;
    }
    
    const trace = {
      position: position,
      timestamp: Date.now(),
      type: event.type,
      intensity: event.importance || 1.0,
      pheromone: this.generateEventPheromone(event),
      decay: 0.99 // Traces fade over time
    };
    
    this.environmentalTraces.fieldModulations.push(trace);
    
    // Limit trace history
    if (this.environmentalTraces.fieldModulations.length > 100) {
      this.environmentalTraces.fieldModulations.shift();
    }
    
    // Add anchor points for significant events
    if (event.importance > 0.8) {
      this.environmentalTraces.anchorPoints.push({
        position: event.position,
        memory: this.holographicMemory.compress(),
        type: event.type
      });
    }
  }
  
  generateEventPheromone(event) {
    // Event-specific chemical signature
    const pheromone = [0, 0, 0, 0, 0]; // Use regular array instead of Float32Array
    
    switch(event.type) {
      case 'predator_escape':
      case 'predator_encounter':
        pheromone[0] = 1.0; // Danger marker
        break;
      case 'food_found':
      case 'successful_feeding':
        pheromone[1] = 1.0; // Food marker
        break;
      case 'reproduction':
        pheromone[2] = 1.0; // Mating marker
        break;
      case 'death_nearby':
      case 'death':
        pheromone[3] = 1.0; // Death marker
        break;
      default:
        pheromone[4] = 0.5; // General activity
    }
    
    return pheromone;
  }
  
  learnFromEnvironment(traces) {
    if (!traces || !traces.fieldModulations) return;
    
    // Learn from recent traces
    const recentTraces = traces.fieldModulations.filter(
      t => Date.now() - t.timestamp < 60000 // Last minute
    );
    
    // Create synthetic experience from traces
    recentTraces.forEach(trace => {
      const syntheticExperience = {
        type: 'environmental_learning',
        importance: trace.intensity * 0.5, // Reduced importance for indirect learning
        position: trace.position,
        trajectory: [trace.position], // Single point
        emotionalValue: this.interpretPheromone(trace.pheromone)
      };
      
      // Gentle learning from environment
      this.holographicMemory.encode(syntheticExperience);
    });
  }
  
  interpretPheromone(pheromone) {
    // Convert pheromone to emotional value
    const emotions = {
      fear: pheromone[0],
      attraction: pheromone[1] + pheromone[2],
      avoidance: pheromone[3],
      curiosity: pheromone[4]
    };
    
    // Return dominant emotion
    let maxEmotion = 'neutral';
    let maxValue = 0;
    
    for (const [emotion, value] of Object.entries(emotions)) {
      if (value > maxValue) {
        maxValue = value;
        maxEmotion = emotion;
      }
    }
    
    return maxEmotion;
  }
  
  // Express phenotype from core + memory
  expressPhenotype() {
    const core = this.topologicalCore;
    const memoryInfluence = this.holographicMemory.recall();
    
    return {
      // Morphology from core
      symmetry: core.bodySymmetry,
      maxOrgans: core.organCapacity,
      
      // Behavior from core + memory
      motility: core.baseMotility * (1 + memoryInfluence.explorationTendency * 0.2), // Reduced memory influence
      sensitivity: core.baseSensitivity * (1 + memoryInfluence.cautionLevel * 0.3),
      resilience: core.baseResilience,
      
      // Pure memory-driven behaviors
      curiosity: memoryInfluence.explorationTendency,
      fearfulness: memoryInfluence.cautionLevel,
      foraging: memoryInfluence.forageIntensity,
      sociability: memoryInfluence.socialAttraction,
      
      // REAL EVOLUTION: Organ expression from inherited genes
      organExpressions: {
        // Sensory organs from sensitivity gene
        photosensor: core.baseSensitivity * (1 + memoryInfluence.cautionLevel * 0.3),
        chemoreceptor: core.baseSensitivity * (1 + memoryInfluence.forageIntensity * 0.3),
        
        // Movement organs from motility gene
        flagellum: core.baseMotility,
        speed_boost: core.baseMotility > 0.7 ? (core.baseMotility - 0.5) * 2 : 0,
        
        // Defense organs from resilience gene
        membrane: core.baseResilience,
        armor_plates: core.baseResilience > 0.6 ? (core.baseResilience - 0.4) * 2 : 0,
        toxin_gland: core.baseResilience > 0.7 ? (core.baseResilience - 0.5) * 2 : 0,
        
        // Special organs emerge from high trait combinations
        electric_organ: (core.baseMotility > 0.6 && core.baseResilience > 0.6) ? 
          (core.baseMotility + core.baseResilience - 1.0) : 0,
        
        regeneration: core.baseResilience > 0.8 ? (core.baseResilience - 0.6) * 2 : 0,
        
        camouflage: (core.baseSensitivity > 0.5 && core.baseResilience > 0.5) ?
          (core.baseSensitivity + core.baseResilience - 0.8) * 0.5 : 0,
          
        vacuole: 0.3 + core.baseResilience * 0.4,
        
        pheromone_emitter: core.baseSensitivity > 0.6 ? core.baseSensitivity * 0.5 : 0
      }
    };
  }
  
  // Reproduce with Lamarckian inheritance
  reproduce(mate = null) {
    return new ProteusInheritance(this, mate);
  }
  
  // Get memory size in bytes
  getMemorySize() {
    return {
      core: 200, // ~50 parameters * 4 bytes
      holographic: this.holographicMemory.size * 4, // Float32Array
      environmental: JSON.stringify(this.environmentalTraces).length,
      total: 200 + this.holographicMemory.size * 4 + 1000 // Approximate
    };
  }
  
  // Compress for storage/transmission
  compress() {
    return {
      core: this.topologicalCore,
      memory: this.holographicMemory.compress(),
      traces: {
        pheromone: this.environmentalTraces.pheromoneSignature,
        generation: this.environmentalTraces.generation,
        recentAnchors: this.environmentalTraces.anchorPoints.slice(-3)
      }
    };
  }
}