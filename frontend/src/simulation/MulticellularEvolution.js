/**
 * PROTEUS Multicellular Evolution System
 * Enables transition from unicellular to multicellular life
 */

export class MulticellularEvolution {
  constructor() {
    // New organ types for multicellular evolution
    this.multicellularOrgans = {
      // Pre-multicellular (cooperation)
      cell_adhesion_proteins: {
        cost: 0.0008,
        effects: {
          adhesion_strength: 1.0,
          allows_colony_formation: true
        }
      },
      quorum_sensor: {
        cost: 0.0006,
        effects: {
          density_detection_range: 30,
          collective_behavior: true
        }
      },
      metabolic_sharing: {
        cost: 0.0007,
        effects: {
          nutrient_transfer_rate: 0.3,
          colony_efficiency: 1.2
        }
      },
      synchronized_clock: {
        cost: 0.0005,
        effects: {
          cycle_synchronization: true,
          coordinated_reproduction: true
        }
      },
      
      // Early colonial
      stem_cell_factor: {
        cost: 0.001,
        effects: {
          differentiation_potential: 1.0,
          cell_type_switching: true
        }
      },
      morphogen_producer: {
        cost: 0.0012,
        effects: {
          gradient_strength: 1.0,
          pattern_formation: true
        }
      },
      gap_junctions: {
        cost: 0.0009,
        effects: {
          direct_communication: true,
          signal_speed: 10.0
        }
      },
      structural_matrix: {
        cost: 0.0015,
        effects: {
          tissue_strength: 1.5,
          maintains_shape: true
        }
      },
      
      // Advanced multicellular
      biofilm_matrix: {
        cost: 0.002,
        effects: {
          collective_protection: 2.0,
          nutrient_retention: 1.5
        },
        requires: ['cell_adhesion_proteins']
      },
      epithelium: {
        cost: 0.003,
        effects: {
          barrier_protection: 3.0,
          selective_permeability: true
        },
        requires: ['cell_adhesion_proteins', 'gap_junctions']
      },
      muscle_fiber: {
        cost: 0.004,
        effects: {
          coordinated_movement: 3.0,
          contraction_power: 2.0
        },
        requires: ['gap_junctions', 'structural_matrix']
      },
      neural_net: {
        cost: 0.005,
        effects: {
          distributed_processing: true,
          reaction_speed: 0.05,
          learning_capacity: true
        },
        requires: ['gap_junctions', 'morphogen_producer']
      },
      compound_eye: {
        cost: 0.006,
        effects: {
          vision_range: 300,
          resolution: 5.0,
          motion_detection: true
        },
        requires: ['photosensor', 'neural_net']
      }
    };
  }
  
  // Check if organism can form colonies
  canFormColony(organism) {
    return organism.checkOrganFunctionality('cell_adhesion_proteins') > 0;
  }
  
  // Calculate colony benefits
  getColonyBenefits(colonySize, colonyOrgans) {
    const benefits = {
      protection: 1.0,
      efficiency: 1.0,
      specialization: 0
    };
    
    // Size benefits (up to a limit)
    benefits.protection *= Math.min(colonySize / 10, 3.0);
    
    // Organ synergies
    if (colonyOrgans.has('biofilm_matrix')) {
      benefits.protection *= 2.0;
    }
    
    if (colonyOrgans.has('metabolic_sharing')) {
      benefits.efficiency *= 1.5;
    }
    
    if (colonyOrgans.has('gap_junctions')) {
      // Allows specialization
      benefits.specialization = Math.min(colonySize / 5, 1.0);
    }
    
    return benefits;
  }
  
  // Determine cell differentiation in colony
  differentiateCell(organism, colonyContext) {
    if (!organism.checkOrganFunctionality('stem_cell_factor')) {
      return null; // No differentiation possible
    }
    
    // Morphogen gradients determine cell fate
    const position = colonyContext.relativePosition;
    // const morphogens = colonyContext.morphogenGradients; // Not used in simple differentiation
    
    // Simple positional differentiation
    if (position.edge) {
      // Edge cells become protective
      return {
        type: 'epithelial',
        specialization: ['armor_plates', 'toxin_gland']
      };
    } else if (position.center) {
      // Center cells process nutrients
      return {
        type: 'digestive',
        specialization: ['vacuole', 'metabolic_sharing']
      };
    } else {
      // Middle cells handle movement/sensing
      return {
        type: 'sensory_motor',
        specialization: ['photosensor', 'flagellum']
      };
    }
  }
  
  // Evolution pressure towards multicellularity
  getMulticellularFitness(organism, environment) {
    let fitness = 1.0;
    
    // Benefits of size (protection from predators)
    const predatorPressure = environment.predatorDensity || 0;
    if (predatorPressure > 0.5 && organism.colonySize > 1) {
      fitness *= 1 + (organism.colonySize * 0.1);
    }
    
    // Benefits of specialization
    if (organism.colony && organism.colony.hasSpecializedCells) {
      fitness *= 1.5;
    }
    
    // Cost of coordination
    const coordinationCost = Math.log(organism.colonySize || 1) * 0.1;
    fitness *= (1 - coordinationCost);
    
    return fitness;
  }
  
  // Check for evolutionary transitions
  checkEvolutionaryTransition(populationStats) {
    const transitions = [];
    
    // Transition 1: Aggregation
    if (populationStats.averageColonySize > 2) {
      transitions.push({
        type: 'aggregation',
        description: 'Organisms forming temporary groups'
      });
    }
    
    // Transition 2: Obligate multicellularity
    if (populationStats.percentageInColonies > 0.8) {
      transitions.push({
        type: 'obligate_multicellular',
        description: 'Most organisms exist in colonies'
      });
    }
    
    // Transition 3: Cell differentiation
    if (populationStats.specializedCellTypes > 2) {
      transitions.push({
        type: 'differentiation',
        description: 'Cells specializing for different functions'
      });
    }
    
    // Transition 4: True multicellular organism
    if (populationStats.integratedColonies > 0) {
      transitions.push({
        type: 'true_multicellular',
        description: 'Colonies functioning as single organisms'
      });
    }
    
    return transitions;
  }
}

// Colony class for managing multicellular structures
export class Colony {
  constructor(founderId) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.members = new Map([[founderId, { role: 'founder', position: { x: 0, y: 0 } }]]);
    this.center = { x: 0, y: 0 };
    this.morphogenGradients = new Map();
    this.sharedResources = {
      energy: 0,
      nutrients: 0
    };
    this.specializations = new Map();
  }
  
  addMember(organismId, parentId) {
    const parent = this.members.get(parentId);
    if (!parent) return false;
    
    // Position relative to parent
    const angle = Math.random() * Math.PI * 2;
    const distance = 10; // Cell size
    
    const position = {
      x: parent.position.x + Math.cos(angle) * distance,
      y: parent.position.y + Math.sin(angle) * distance
    };
    
    this.members.set(organismId, {
      role: 'member',
      position: position,
      parentId: parentId
    });
    
    this.updateCenter();
    this.updateMorphogens();
    
    return true;
  }
  
  updateCenter() {
    let sumX = 0, sumY = 0;
    this.members.forEach(member => {
      sumX += member.position.x;
      sumY += member.position.y;
    });
    
    this.center.x = sumX / this.members.size;
    this.center.y = sumY / this.members.size;
  }
  
  updateMorphogens() {
    // Simple radial morphogen gradient
    this.morphogenGradients.clear();
    
    this.members.forEach((member, id) => {
      const dist = Math.sqrt(
        Math.pow(member.position.x - this.center.x, 2) +
        Math.pow(member.position.y - this.center.y, 2)
      );
      
      this.morphogenGradients.set(id, {
        centerGradient: 1 - (dist / (this.members.size * 10)),
        edgeGradient: dist / (this.members.size * 10)
      });
    });
  }
  
  shareResources(contributions) {
    // Pool resources
    contributions.forEach((amount, organismId) => {
      if (this.members.has(organismId)) {
        this.sharedResources.energy += amount * 0.8; // 20% sharing cost
      }
    });
    
    // Distribute based on specialization
    const distributions = new Map();
    const sharePerMember = this.sharedResources.energy / this.members.size;
    
    this.members.forEach((member, id) => {
      const specialization = this.specializations.get(id);
      let share = sharePerMember;
      
      // Specialized cells may get different shares
      if (specialization === 'digestive') {
        share *= 0.8; // Digestive cells need less
      } else if (specialization === 'sensory_motor') {
        share *= 1.2; // Active cells need more
      }
      
      distributions.set(id, share);
    });
    
    this.sharedResources.energy = 0;
    return distributions;
  }
  
  get size() {
    return this.members.size;
  }
  
  get isIntegrated() {
    // Check if colony acts as single organism
    const hasSpecializations = this.specializations.size > this.size * 0.5;
    const hasCoordination = this.morphogenGradients.size === this.size;
    
    return hasSpecializations && hasCoordination && this.size > 4;
  }
}