/**
 * PROTEUS Organism - Evolution-First Design
 * No artificial thresholds, pure natural selection
 */

import { ProteusInheritance } from './ProteusInheritance';
import { MulticellularEvolution, Colony } from './MulticellularEvolution';
import { CellularPhysics } from './CellularPhysics';
import { HierarchicalControl } from './HierarchicalControl';
import { LocalEquilibriumLearning } from './LocalEquilibriumLearning';

export class Organism {
  constructor(x, y, topologyEngine, parentInheritance = null) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.position = { x, y };
    // Start with small random velocity to avoid getting stuck
    this.velocity = { 
      x: (Math.random() - 0.5) * 10, 
      y: (Math.random() - 0.5) * 10 
    };
    this.smoothPosition = { x, y };
    
    // Multicellular properties
    this.colony = null;
    this.attachedTo = null;
    this.cellType = 'generalist'; // Can differentiate
    this.multicellularEvolution = new MulticellularEvolution();
    this.cellularPhysics = new CellularPhysics();
    
    // Physical state
    this.mass = 1.0; // Relative mass
    this.boundCells = new Set(); // Physically bound neighbors
    this.morphogenReceptors = new Map(); // Morphogen concentrations sensed
    
    // Core survival stats
    this.age = 0;
    this.energy = 2.0; // Start with more energy to survive initial period
    this.alive = true;
    this.generation = 0;
    
    // Inheritance system
    this.inheritance = parentInheritance || new ProteusInheritance();
    this.topologyEngine = topologyEngine;
    
    // Express phenotype immediately
    this.organs = [];
    this.capabilities = {};
    this.applyPhenotype();
    
    // Hierarchical control system
    this.hierarchicalControl = new HierarchicalControl(topologyEngine.worldSize);
    
    // Local equilibrium learning
    this.learning = new LocalEquilibriumLearning();
    this.episodeStartEnergy = this.energy;
    this.episodeStartTime = 0;
    
    // Behavioral state
    this.trajectory = [];
    this.lastAction = null;
    this.hunger = 0.5;
    this.deathCause = null;
    
    // Movement momentum to prevent circular patterns
    this.movementMomentum = { x: 0, y: 0 };
    this.lastMovementDecision = { x: 0, y: 0 };
    this.movementCommitmentTime = 0;
    this.stuckDetector = { position: {x, y}, time: 0, threshold: 50 };
    
    // Additional properties for compatibility
    this.color = this.generateColor();
    this.maxAge = 15 + this.inheritance.topologicalCore.baseResilience * 15;  // Max 30 years
    this.phenotype = 'evolving'; // Will be updated
    this.isPanicked = false;
    this.environmentalMemory = null;
  }
  
  generateColor() {
    const core = this.inheritance.topologicalCore;
    // More color variation based on body symmetry (species indicator)
    const hue = 140 + core.bodySymmetry * 40 + core.manifoldDimension * 10;
    const saturation = 40 + core.baseSensitivity * 30;
    const lightness = 40 + core.baseResilience * 20;
    return `hsl(${hue % 360}, ${saturation}%, ${lightness}%)`;
  }
  
  applyPhenotype() {
    const phenotype = this.inheritance.expressPhenotype();
    
    // Create organs based on genetic expression - NO THRESHOLDS
    this.organs = [];
    for (const [organType, expression] of Object.entries(phenotype.organExpressions)) {
      if (expression > 0) { // Any expression creates an organ
        this.organs.push({
          type: organType,
          expression: expression,
          functionality: expression, // Direct mapping
          energyCost: this.getOrganCost(organType) * expression
        });
      }
    }
    
    // Capabilities emerge from organs - NO ARTIFICIAL LIMITS
    this.updateCapabilities();
    
    // Update phenotype name based on dominant organ
    let maxOrgan = 'basic';
    let maxValue = 0;
    this.organs.forEach(organ => {
      if (organ.functionality > maxValue) {
        maxValue = organ.functionality;
        maxOrgan = organ.type;
      }
    });
    this.phenotype = maxOrgan;
  }
  
  updateCapabilities() {
    // Reset to base values from genes
    const phenotype = this.inheritance.expressPhenotype();
    this.capabilities = {
      motility: phenotype.motility,
      sensitivity: phenotype.sensitivity,
      resilience: phenotype.resilience
    };
    
    // Each organ modifies capabilities naturally
    this.organs.forEach(organ => {
      const effect = this.getOrganEffect(organ);
      for (const [cap, value] of Object.entries(effect)) {
        this.capabilities[cap] = (this.capabilities[cap] || 0) + value;
      }
    });
  }
  
  getOrganCost(organType) {
    // Energy cost per time unit - reduced costs for survival
    const costs = {
      photosensor: 0.0005,
      chemoreceptor: 0.0005,
      flagellum: 0.001,
      speed_boost: 0.0015,
      membrane: 0.00025,
      armor_plates: 0.001,
      toxin_gland: 0.00075,
      electric_organ: 0.002,
      regeneration: 0.001,
      camouflage: 0.0005,
      vacuole: 0.00025,
      pheromone_emitter: 0.0005,
      // Multicellular organs
      ...this.multicellularEvolution.multicellularOrgans
    };
    return costs[organType]?.cost || costs[organType] || 0.001;
  }
  
  getOrganEffect(organ) {
    // Direct effects on capabilities - no artificial scaling
    const effects = {
      photosensor: { 
        detection_range: organ.functionality * 100,
        reaction_time: -organ.functionality * 0.1 // Negative = faster
      },
      chemoreceptor: { 
        nutrient_detection: organ.functionality * 80,
        gradient_sensitivity: organ.functionality
      },
      flagellum: { 
        max_speed: organ.functionality * 2,
        acceleration: organ.functionality * 0.5
      },
      speed_boost: { 
        max_speed: organ.functionality * 3,
        energy_efficiency: -organ.functionality * 0.2 // Costs more energy
      },
      membrane: { 
        damage_reduction: organ.functionality * 0.3,
        energy_efficiency: organ.functionality * 0.1
      },
      armor_plates: { 
        damage_reduction: organ.functionality * 0.5,
        max_speed: -organ.functionality * 0.2 // Slows down
      },
      toxin_gland: { 
        toxin_damage: organ.functionality,
        toxin_range: organ.functionality * 10
      },
      electric_organ: { 
        stun_power: organ.functionality,
        stun_range: organ.functionality * 20
      },
      regeneration: { 
        healing_rate: organ.functionality * 0.01
      },
      camouflage: { 
        detection_difficulty: organ.functionality
      },
      vacuole: { 
        energy_capacity: organ.functionality * 0.5,
        energy_efficiency: organ.functionality * 0.2
      },
      pheromone_emitter: { 
        signal_range: organ.functionality * 50,
        signal_strength: organ.functionality
      },
      // Multicellular organs
      cell_adhesion_proteins: {
        adhesion_strength: organ.functionality,
        colony_cohesion: organ.functionality * 0.5
      },
      quorum_sensor: {
        density_detection: organ.functionality * 30,
        collective_awareness: organ.functionality
      },
      metabolic_sharing: {
        resource_transfer: organ.functionality * 0.3,
        efficiency_bonus: organ.functionality * 0.2
      },
      synchronized_clock: {
        coordination: organ.functionality,
        sync_range: organ.functionality * 20
      },
      stem_cell_factor: {
        differentiation_potential: organ.functionality,
        adaptability: organ.functionality * 0.5
      },
      morphogen_producer: {
        gradient_strength: organ.functionality,
        pattern_complexity: organ.functionality * 0.8
      },
      gap_junctions: {
        communication_speed: organ.functionality * 10,
        signal_fidelity: organ.functionality
      },
      structural_matrix: {
        tissue_strength: organ.functionality * 1.5,
        shape_maintenance: organ.functionality
      }
    };
    
    // Check for multicellular organ requirements
    const organData = this.multicellularEvolution?.multicellularOrgans[organ.type];
    if (organData?.requires) {
      // Reduce effectiveness if missing prerequisites
      const hasAllRequirements = organData.requires.every(req => 
        this.organs.some(o => o.type === req)
      );
      if (!hasAllRequirements) {
        // Halve all effects if missing requirements
        Object.keys(effects[organ.type] || {}).forEach(key => {
          if (effects[organ.type][key]) {
            effects[organ.type][key] *= 0.5;
          }
        });
      }
    }
    
    return effects[organ.type] || {};
  }
  
  update(deltaTime, nutrients, predators, organisms = []) {
    if (!this.alive) return;
    
    this.age += deltaTime;
    
    // Update movement commitment timer
    if (this.movementCommitmentTime > 0) {
      this.movementCommitmentTime = Math.max(0, this.movementCommitmentTime - deltaTime);
    }
    
    // Energy consumption - base metabolism + organ costs
    let energyCost = 0.0005 * deltaTime; // Reduced base metabolism
    this.organs.forEach(organ => {
      energyCost += organ.energyCost * deltaTime;
    });
    
    // Debug first organism movement (disabled for performance)
    // if (this.id && this.position && Math.random() < 0.001) {
    //   const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    //   console.log(`Organism ${this.id.substring(0,4)} - Pos: (${this.position.x.toFixed(1)}, ${this.position.y.toFixed(1)}) | Vel: (${this.velocity.x.toFixed(3)}, ${this.velocity.y.toFixed(3)}) | Speed: ${speed.toFixed(3)} | Motility: ${this.capabilities.motility.toFixed(2)}`);
    // }
    
    // Physical forces and colony dynamics
    if (this.colony) {
      // Apply cellular physics
      const dynamics = this.cellularPhysics.updateColonyDynamics(this.colony, deltaTime);
      
      // Nutrient sharing through physical diffusion
      const nutrientFlow = dynamics.nutrientFlows.get(this.id) || 0;
      this.energy += nutrientFlow * deltaTime;
      
      // Morphogen sensing (no parameters, just chemistry)
      if (this.checkOrganFunctionality('morphogen_producer') > 0) {
        // Produce morphogens based on metabolic state
        // const productionRate = this.energy * this.checkOrganFunctionality('morphogen_producer'); // Not used
        // This would update a morphogen field
      }
      
      // Cell fate from morphogen concentrations
      if (this.checkOrganFunctionality('stem_cell_factor') > 0) {
        // Read local morphogen concentrations
        const pattern = this.topologyEngine.patternFormation?.getPatternAt(
          this.position.x, 
          this.position.y
        );
        
        if (pattern) {
          // Cell fate emerges from reaction-diffusion pattern
          if (pattern.ratio > 2) {
            // High activator/inhibitor ratio
            this.cellType = 'sensory'; // Develops sensors
          } else if (pattern.ratio < 0.5) {
            // Low ratio
            this.cellType = 'structural'; // Develops structure
          } else if (pattern.activator > 0.7) {
            // High activator
            this.cellType = 'metabolic'; // Processing
          }
          // No else - remains generalist
        }
      }
      
      // Apply forces from cell-cell interactions
      this.boundCells.forEach(neighborId => {
        const neighbor = this.findCellById(neighborId);
        if (neighbor) {
          const force = this.cellularPhysics.calculateCellForce(this, neighbor);
          // Forces affect movement
          this.velocity.x += force.x * deltaTime / this.mass;
          this.velocity.y += force.y * deltaTime / this.mass;
        }
      });
    }
    
    // Create environment object for perception with limits
    const environment = {
      nutrients: nutrients || [],
      predators: predators || [],
      organisms: (organisms || []).slice(0, 20) // Limit to 20 nearest for performance
    };
    
    // Movement decision based on perception
    const perception = this.perceive(environment);
    const decision = this.decide(perception);
    this.act(decision, deltaTime);
    
    // Debug: Log if organism is not moving (disabled for performance)
    // if (Math.random() < 0.001 && Math.abs(this.velocity.x) < 0.01 && Math.abs(this.velocity.y) < 0.01) {
    //   console.log(`🚨 Organism ${this.id.substring(0,4)} NOT MOVING!`);
    // }
    
    // Energy cost of movement
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    energyCost += speed * 0.00005 * deltaTime; // Reduced movement cost
    
    // Apply energy changes
    this.energy -= energyCost;
    
    // Log energy consumption for first organism periodically (disabled for performance)
    // if (this === this.topologyEngine?.worldSize?.debugOrganism && Math.random() < 0.01) {
    //   console.log(`Energy Debug - ID: ${this.id.substring(0,4)} | Energy: ${this.energy.toFixed(3)} | Cost: ${energyCost.toFixed(5)} | Organs: ${this.organs.length} | Speed: ${speed.toFixed(2)}`);
    // }
    
    // Regeneration if has the organ
    const regen = this.organs.find(o => o.type === 'regeneration');
    if (regen && this.energy < 2.0) {
      this.energy += regen.functionality * 0.01 * deltaTime;
    }
    
    // Death is natural - no artificial thresholds
    if (this.energy <= 0) {
      // End learning episode on death
      this.learning.endEpisode(false, {
        position: this.position,
        energy: this.energy
      }, this.inheritance);
      this.die('starvation');
    }
    
    // Aging is also natural - lifespan varies by genetics
    if (this.age > this.maxAge) {
      // End learning episode on death (old age is "successful")
      this.learning.endEpisode(true, {
        position: this.position,
        energy: this.energy
      }, this.inheritance);
      this.die('old_age');
    }
    
    // Update position with NaN protection
    if (Number.isFinite(this.velocity.x) && Number.isFinite(this.velocity.y)) {
      this.position.x += this.velocity.x * deltaTime;
      this.position.y += this.velocity.y * deltaTime;
    } else {
      // Reset velocity if NaN
      console.warn('NaN velocity detected:', this.id, this.velocity);
      this.velocity.x = 0;
      this.velocity.y = 0;
    }
    
    // Ensure position is valid
    if (!Number.isFinite(this.position.x) || !Number.isFinite(this.position.y)) {
      console.error('NaN position detected:', this.id, this.position);
      // Reset to center
      this.position.x = this.topologyEngine.worldSize.width / 2;
      this.position.y = this.topologyEngine.worldSize.height / 2;
    }
    
    // World bounds wrapping
    const worldWidth = this.topologyEngine.worldSize.width;
    const worldHeight = this.topologyEngine.worldSize.height;
    
    // Wrap X coordinate
    if (this.position.x < 0) {
      this.position.x += worldWidth;
    } else if (this.position.x > worldWidth) {
      this.position.x -= worldWidth;
    }
    
    // Wrap Y coordinate
    if (this.position.y < 0) {
      this.position.y += worldHeight;
    } else if (this.position.y > worldHeight) {
      this.position.y -= worldHeight;
    }
    
    // Smooth position for rendering (more responsive)
    this.smoothPosition.x += (this.position.x - this.smoothPosition.x) * 0.8;
    this.smoothPosition.y += (this.position.y - this.smoothPosition.y) * 0.8;
    
    // Detect if stuck in circular pattern
    const distFromStuckPoint = Math.sqrt(
      Math.pow(this.position.x - this.stuckDetector.position.x, 2) +
      Math.pow(this.position.y - this.stuckDetector.position.y, 2)
    );
    
    if (distFromStuckPoint < 5) {
      this.stuckDetector.time += deltaTime;
      if (this.stuckDetector.time > 1.0) { // Stuck for 1 second
        // Break out with random strong impulse
        const breakoutAngle = Math.random() * Math.PI * 2;
        const breakoutForce = 100;
        this.velocity.x = Math.cos(breakoutAngle) * breakoutForce;
        this.velocity.y = Math.sin(breakoutAngle) * breakoutForce;
        this.stuckDetector.time = 0;
      }
    } else if (distFromStuckPoint > 20) {
      // Reset stuck detector
      this.stuckDetector.position = { x: this.position.x, y: this.position.y };
      this.stuckDetector.time = 0;
    }
    
    // Record trajectory
    this.trajectory.push({ x: this.position.x, y: this.position.y });
    if (this.trajectory.length > 100) this.trajectory.shift();
    
    // Update hunger (for compatibility)
    this.hunger = Math.max(0, Math.min(1, 2.0 - this.energy));
  }
  
  perceive(environment) {
    const perception = {
      nutrients: [],
      predators: [],
      organisms: [],
      light: 0,
      chemicals: {}
    };
    
    // PHOTON DETECTION - Physical light sensing
    const photosensors = this.organs.filter(o => o.type === 'photosensor');
    if (photosensors.length > 0) {
      // Each photosensor has quantum efficiency
      const quantumEfficiency = photosensors.reduce((sum, s) => 
        sum + s.expression * 0.3, 0 // 30% max quantum efficiency
      );
      
      // Detect light sources
      environment.predators.forEach(predator => {
        if (!predator.lightFlash) return;
        
        const dist = this.distanceTo(predator.position);
        
        // Inverse square law for light intensity
        const lightIntensity = predator.lightIntensity || 1.0; // Default intensity
        const photonFlux = lightIntensity / (4 * Math.PI * dist * dist);
        
        // Photon detection is probabilistic
        const detectionProbability = 1 - Math.exp(-photonFlux * quantumEfficiency);
        
        if (Math.random() < detectionProbability) {
          // Angular resolution depends on sensor array
          const angularResolution = Math.PI / (1 + photosensors.length);
          
          perception.predators.push({
            position: predator.position,
            distance: dist,
            clarity: detectionProbability,
            angular_uncertainty: angularResolution
          });
        }
      });
      
      // See other organisms for grouping behavior
      if (environment.organisms) {
        environment.organisms.forEach(org => {
          if (org.id === this.id || !org.alive) return;
          const dist = this.distanceTo(org.position);
          
          // Visual range for seeing other organisms
          const visualRange = 50 + quantumEfficiency * 100;
          
          if (dist < visualRange) {
            perception.organisms.push({
              id: org.id,
              position: org.position,
              distance: dist,
              generation: org.generation,
              capabilities: org.capabilities,
              inheritance: org.inheritance
            });
          }
        });
      }
    }
    
    // MOLECULAR DETECTION - Chemical gradients
    const chemoreceptors = this.organs.filter(o => o.type === 'chemoreceptor');
    if (chemoreceptors.length > 0) {
      // Receptor binding kinetics
      const receptorDensity = chemoreceptors.reduce((sum, c) => 
        sum + c.expression, 0
      );
      
      // Detect chemical gradients
      environment.nutrients.forEach(nutrient => {
        const dist = this.distanceTo(nutrient);
        
        // Steady-state diffusion creates concentration gradient
        // C(r) = C0 * (a/r) where a is source radius
        const sourceRadius = 5;
        const concentration = nutrient.energy * (sourceRadius / Math.max(dist, sourceRadius));
        
        // Hill equation for receptor binding
        const Kd = 0.1; // Dissociation constant
        const n = 2; // Hill coefficient (cooperativity)
        const binding = Math.pow(concentration, n) / 
                       (Math.pow(Kd, n) + Math.pow(concentration, n));
        
        // Detection threshold from receptor occupancy
        const signal = binding * receptorDensity;
        
        if (signal > 0.01) { // Minimal signal threshold
          // Gradient detection by comparing sides of cell
          const gradientAngle = Math.atan2(
            nutrient.y - this.position.y,
            nutrient.x - this.position.x
          );
          
          perception.nutrients.push({
            position: nutrient,
            distance: dist,
            strength: signal,
            gradient_direction: gradientAngle,
            concentration: concentration
          });
        }
      });
      
      // Detect pheromones
      environment.organisms.forEach(org => {
        if (org.id === this.id) return;
        const emitter = org.organs.find(o => o.type === 'pheromone_emitter');
        if (emitter) {
          const dist = this.distanceTo(org.position);
          const range = emitter.functionality * 50;
          if (dist < range) {
            perception.chemicals[org.id] = {
              type: 'organism_pheromone',
              strength: emitter.functionality * (1 - dist / range),
              position: org.position
            };
          }
        }
      });
    }
    
    return perception;
  }
  
  decide(perception) {
    // Use hierarchical control system for decision making
    const memoryRecall = this.inheritance.holographicMemory.recall();
    
    // Prepare perception for hierarchical control
    const hierarchicalPerception = {
      position: this.position,
      energy: this.energy,
      age: this.age,
      predatorNearby: perception.predators.length > 0,
      sameSpeciesNearby: perception.sameSpecies && perception.sameSpecies.length > 0,
      chemicalGradient: { x: 0, y: 0 },
      lightGradient: { x: 0, y: 0 },
      predatorDirection: { x: 0, y: 0 },
      speciesDirection: { x: 0, y: 0 },
      recentExperiences: this.trajectory.slice(-10)
    };
    
    // Calculate gradients for hierarchical control
    perception.nutrients.forEach(nutrient => {
      const signalStrength = nutrient.strength * (nutrient.concentration || 0.5);
      // Convert gradient direction angle to vector
      const dirX = Math.cos(nutrient.gradient_direction);
      const dirY = Math.sin(nutrient.gradient_direction);
      hierarchicalPerception.chemicalGradient.x += dirX * signalStrength;
      hierarchicalPerception.chemicalGradient.y += dirY * signalStrength;
    });
    
    if (perception.predators.length > 0) {
      const nearest = perception.predators[0];
      // Calculate direction to predator
      const dx = nearest.position.x - this.position.x;
      const dy = nearest.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 0) {
        hierarchicalPerception.predatorDirection = { x: dx / dist, y: dy / dist };
      }
    }
    
    if (perception.organisms && perception.organisms.length > 0) {
      let avgX = 0, avgY = 0;
      let sameSpeciesCount = 0;
      
      perception.organisms.forEach(other => {
        // Check if same species (using genetic similarity)
        const similarity = this.calculateGeneticSimilarity(other);
        if (similarity > 0.7) {
          const dx = other.position.x - this.position.x;
          const dy = other.position.y - this.position.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0) {
            avgX += dx / dist;
            avgY += dy / dist;
            sameSpeciesCount++;
          }
        }
      });
      
      if (sameSpeciesCount > 0) {
        hierarchicalPerception.speciesDirection = {
          x: avgX / sameSpeciesCount,
          y: avgY / sameSpeciesCount
        };
        hierarchicalPerception.sameSpeciesNearby = true;
      }
    }
    
    // Record state for learning (reduced frequency for performance)
    if (Math.random() < 0.01) { // Only record 1% of the time for better performance
      this.learning.recordState({
        position: this.position,
        energy: this.energy,
        lightLevel: perception.lightLevel || 0,
        chemicalGradient: hierarchicalPerception.chemicalGradient,
        predatorDistance: perception.predators.length > 0 ? perception.predators[0].distance : Infinity,
        intention: this.hierarchicalControl.slowField.intention,
        computationIterations: this.hierarchicalControl.fastField.iterationCount
      });
    }
    
    // Get decision from hierarchical control
    const controlResult = this.hierarchicalControl.update(
      this.lastUpdateTime || 0.016, // deltaTime
      hierarchicalPerception,
      memoryRecall
    );
    
    // Convert control decision to organism action
    const decision = {
      move: controlResult.decision,
      useAbility: null,
      intention: controlResult.intention,
      computationIterations: controlResult.iterations
    };
    
    // Keep defensive responses
    const urgency = perception.predators.length > 0 ? perception.predators[0].clarity : 0;
    if (urgency > 0.7) {
      this.organs.forEach(organ => {
        if (organ.type === 'toxin_gland' && this.energy > 0.5) {
          decision.useAbility = 'release_toxin';
        } else if (organ.type === 'electric_organ' && this.energy > 0.3) {
          decision.useAbility = 'electric_discharge';
        }
      });
    }
    
    return decision;
  }
  
  act(decision, deltaTime) {
    // Store deltaTime for hierarchical control
    this.lastUpdateTime = deltaTime;
    
    // Execute decision
    
    // Movement
    const maxSpeed = this.getMaxSpeed();
    const acceleration = this.getAcceleration();
    
    // Add momentum to maintain direction and prevent oscillation
    const momentumDecay = 0.9;
    this.movementMomentum.x = this.movementMomentum.x * momentumDecay + decision.move.x * 0.3;
    this.movementMomentum.y = this.movementMomentum.y * momentumDecay + decision.move.y * 0.3;
    
    // Combine decision with momentum
    let finalDecision = {
      x: decision.move.x + this.movementMomentum.x,
      y: decision.move.y + this.movementMomentum.y
    };
    
    // Apply acceleration with NaN protection
    if (Number.isFinite(finalDecision.x) && Number.isFinite(finalDecision.y)) {
      this.velocity.x += finalDecision.x * acceleration * deltaTime;
      this.velocity.y += finalDecision.y * acceleration * deltaTime;
    }
    
    // Ensure velocity is finite
    if (!Number.isFinite(this.velocity.x)) this.velocity.x = 0;
    if (!Number.isFinite(this.velocity.y)) this.velocity.y = 0;
    
    // Limit to max speed
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Natural friction (reduced for better movement)
    this.velocity.x *= 0.98;
    this.velocity.y *= 0.98;
    
    // Use abilities
    if (decision.useAbility) {
      this.useAbility(decision.useAbility);
    }
    
    this.lastAction = decision;
  }
  
  getMaxSpeed() {
    // Base speed from genetic motility trait (MASSIVELY INCREASED)
    let speed = this.capabilities.motility * 200.0; // 40x original
    
    // Each movement organ adds to speed
    this.organs.forEach(organ => {
      if (organ.type === 'flagellum') {
        speed += organ.functionality * 80;
      } else if (organ.type === 'speed_boost') {
        speed += organ.functionality * 120;
      } else if (organ.type === 'armor_plates') {
        speed -= organ.functionality * 20; // Heavy armor slows
      }
    });
    
    return Math.max(20.0, speed); // Much higher minimum speed
  }
  
  getAcceleration() {
    // Base acceleration from motility (MASSIVELY INCREASED)
    let accel = 20.0 + this.capabilities.motility * 100.0;
    
    this.organs.forEach(organ => {
      if (organ.type === 'flagellum') {
        accel += organ.functionality * 15.0;
      } else if (organ.type === 'speed_boost') {
        accel += organ.functionality * 25.0;
      }
    });
    
    return accel;
  }
  
  useAbility(ability) {
    // Abilities cost energy but provide advantages
    switch(ability) {
      case 'release_toxin':
        this.energy -= 0.1;
        // Toxin effect handled by predator detection
        break;
      case 'electric_discharge':
        this.energy -= 0.2;
        // Stun effect handled by predator
        break;
      default:
        // Unknown ability - no effect
        break;
    }
  }
  
  feed(nutrientEnergy) {
    // Energy absorption efficiency depends on organs
    let efficiency = 1.0;
    
    this.organs.forEach(organ => {
      if (organ.type === 'vacuole') {
        efficiency += organ.functionality * 0.3;
      } else if (organ.type === 'membrane') {
        efficiency += organ.functionality * 0.1;
      }
    });
    
    const absorbed = nutrientEnergy * efficiency;
    
    // Max energy capacity also depends on organs
    let maxEnergy = 2.0;
    this.organs.forEach(organ => {
      if (organ.type === 'vacuole') {
        maxEnergy += organ.functionality * 0.5;
      }
    });
    
    this.energy = Math.min(maxEnergy, this.energy + absorbed);
    
    // Record feeding success
    this.inheritance.experience({
      type: 'successful_feeding',
      position: [this.position.x, this.position.y],
      importance: absorbed / nutrientEnergy,
      trajectory: this.trajectory.slice(-10)
    });
  }
  
  canReproduce() {
    // Reproduction emerges from metabolic state - reduced cost
    const reproductionCost = 0.3 + this.organs.length * 0.05;
    
    // Cell division requires energy for DNA replication, protein synthesis
    const excessEnergy = this.energy - reproductionCost;
    
    // Limit reproduction rate for performance
    if (Math.random() > 0.1) return false; // Only 10% chance when conditions are met (reduced from 30%)
    if (excessEnergy <= 0) return false;
    
    // Temperature affects reaction rates (Arrhenius equation)
    const T = this.cellularPhysics.TEMPERATURE;
    const activationEnergy = 50; // kJ/mol for cell division
    const R = 8.314; // Gas constant
    const reactionRate = Math.exp(-activationEnergy / (R * T));
    
    // Metabolic rate influences division
    const metabolicRate = this.organs
      .filter(o => o.type === 'vacuole' || o.type === 'metabolic_sharing')
      .reduce((sum, o) => sum + o.expression, 1.0);
    
    // Division probability from biochemical kinetics - increased for better survival
    const divisionProbability = excessEnergy * reactionRate * metabolicRate * 0.005;
    
    // Stochastic division
    return Math.random() < divisionProbability;
  }
  
  reproduce() {
    // Check for colonial reproduction
    const hasAdhesion = this.checkOrganFunctionality('cell_adhesion_proteins') > 0;
    // const colonyChance = hasAdhesion ? this.checkOrganFunctionality('cell_adhesion_proteins') : 0; // Not used
    
    // Create offspring with mutation
    const childX = this.position.x + (Math.random() - 0.5) * 20;
    const childY = this.position.y + (Math.random() - 0.5) * 20;
    
    const child = new Organism(childX, childY, this.topologyEngine, this.inheritance.reproduce());
    child.generation = this.generation + 1;
    
    // Physics determines if cells stay attached
    if (hasAdhesion) {
      // Calculate binding force
      const bindingEnergy = this.cellularPhysics.getBindingEnergy(this, child);
      const thermalEnergy = this.cellularPhysics.BOLTZMANN_CONSTANT * 
                           this.cellularPhysics.TEMPERATURE;
      
      // Binding occurs if energy overcomes thermal fluctuations
      if (bindingEnergy > thermalEnergy * 3) { // 3kT threshold
        // Cells remain bound
        if (!this.colony) {
          this.colony = new Colony(this.id);
        }
        child.colony = this.colony;
        child.attachedTo = this.id;
        child.colony.addMember(child.id, this.id);
        
        // Physics determines position
        const optimalDistance = 10; // Cell diameter
        const angle = Math.random() * Math.PI * 2;
        child.position.x = this.position.x + Math.cos(angle) * optimalDistance;
        child.position.y = this.position.y + Math.sin(angle) * optimalDistance;
        child.smoothPosition = { ...child.position };
        
        // Add to bound cells
        this.boundCells.add(child.id);
        child.boundCells.add(this.id);
      }
    }
    
    // Energy cost
    const reproductionCost = 0.5 + this.organs.length * 0.1;
    this.energy -= reproductionCost;
    
    // Child gets some energy
    child.energy = reproductionCost * 0.8;
    
    // Record reproduction
    this.inheritance.experience({
      type: 'reproduction',
      position: [this.position.x, this.position.y],
      importance: 1.0,
      trajectory: []
    });
    
    return child;
  }
  
  distanceTo(point) {
    const dx = point.x - this.position.x;
    const dy = point.y - this.position.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  calculateGeneticSimilarity(other) {
    if (!other.inheritance || !this.inheritance) return 0;
    
    const myCore = this.inheritance.topologicalCore;
    const otherCore = other.inheritance?.topologicalCore;
    
    if (!myCore || !otherCore) return 0;
    
    // Compare key genetic traits
    let similarity = 0;
    
    // Body symmetry (important for species)
    if (myCore.bodySymmetry === otherCore.bodySymmetry) {
      similarity += 0.3; // 30% weight
    }
    
    // Organ capacity
    const organDiff = Math.abs(myCore.organCapacity - otherCore.organCapacity);
    similarity += (1 - organDiff / 5) * 0.2; // 20% weight
    
    // Behavioral traits
    const motilityDiff = Math.abs(myCore.baseMotility - otherCore.baseMotility);
    similarity += (1 - motilityDiff) * 0.2; // 20% weight
    
    const sensitivityDiff = Math.abs(myCore.baseSensitivity - otherCore.baseSensitivity);
    similarity += (1 - sensitivityDiff) * 0.15; // 15% weight
    
    const resilienceDiff = Math.abs(myCore.baseResilience - otherCore.baseResilience);
    similarity += (1 - resilienceDiff) * 0.15; // 15% weight
    
    return Math.max(0, Math.min(1, similarity));
  }
  
  vectorTo(point) {
    const dx = point.x - this.position.x;
    const dy = point.y - this.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist === 0) return { x: 0, y: 0 };
    return { x: dx / dist, y: dy / dist };
  }
  
  vectorFrom(point) {
    const v = this.vectorTo(point);
    return { x: -v.x, y: -v.y };
  }
  
  // Allow dynamic organ evolution
  mutateOrgans() {
    const mutationRate = this.inheritance.topologicalCore.mutability;
    
    // Chance to develop new organ type
    if (Math.random() < mutationRate * 0.1) {
      const possibleOrgans = [
        'photosensor', 'chemoreceptor', 'flagellum', 'speed_boost',
        'membrane', 'armor_plates', 'toxin_gland', 'electric_organ',
        'regeneration', 'camouflage', 'vacuole', 'pheromone_emitter',
        // New organ types can emerge
        'magnetoreceptor', 'pressure_sensor', 'thermal_sensor',
        'bio_luminescence', 'acid_gland', 'sticky_secretion',
        'filter_feeding', 'symbiotic_chamber', 'neural_ganglion',
        // Multicellular evolution organs
        'cell_adhesion_proteins', 'quorum_sensor', 'metabolic_sharing',
        'synchronized_clock', 'stem_cell_factor', 'morphogen_producer',
        'gap_junctions', 'structural_matrix'
      ];
      
      // Pick a random organ not already expressed
      const currentOrgans = this.organs.map(o => o.type);
      const newOptions = possibleOrgans.filter(o => !currentOrgans.includes(o));
      
      if (newOptions.length > 0) {
        const newOrgan = newOptions[Math.floor(Math.random() * newOptions.length)];
        
        // Add to genetic expression
        this.inheritance.topologicalCore[`expr_${newOrgan}`] = Math.random() * 0.3;
        
        // Re-express phenotype
        this.applyPhenotype();
      }
    }
  }
  
  // Compatibility methods
  die(cause = 'unknown') {
    this.alive = false;
    this.deathCause = cause;
  }
  
  recordExperience(type, importance = 0.5) {
    this.inheritance.experience({
      type,
      position: [this.position.x, this.position.y],
      importance,
      trajectory: this.trajectory.slice(-20)
    });
  }
  
  detectPredatorThreat(predators) {
    // Use the perceive method but return in expected format
    const perception = this.perceive({ predators, nutrients: [], organisms: [] });
    if (perception.predators.length > 0) {
      const nearest = perception.predators[0];
      return {
        predator: { position: nearest.position },
        distance: nearest.distance
      };
    }
    return null;
  }
  
  findNearestNutrient(nutrients) {
    let nearest = null;
    let minDist = Infinity;
    
    nutrients.forEach(nutrient => {
      const dist = this.distanceTo(nutrient);
      if (dist < minDist) {
        minDist = dist;
        nearest = nutrient;
      }
    });
    
    return nearest ? { nutrient: nearest, distance: minDist } : null;
  }
  
  checkOrganFunctionality(organType) {
    const organ = this.organs.find(o => o.type === organType);
    return organ ? organ.functionality : 0;
  }
  
  get topologicalState() {
    return {
      position: [this.position.x, this.position.y],
      momentum: { x: this.velocity.x, y: this.velocity.y }
    };
  }
  
  set topologicalState(value) {
    // For compatibility
  }
  
  distanceToColonyCenter() {
    if (!this.colony || !this.colony.center) return 0;
    
    const dx = this.position.x - this.colony.center.x;
    const dy = this.position.y - this.colony.center.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  // Physics-based colony cohesion
  maintainColonyAttachment() {
    if (!this.colony) return;
    
    // Check each bound cell
    const toRemove = new Set();
    this.boundCells.forEach(neighborId => {
      const neighbor = this.findCellById(neighborId);
      if (!neighbor) {
        toRemove.add(neighborId);
        return;
      }
      
      // Calculate binding force
      const distance = this.cellularPhysics.calculateDistance(this, neighbor);
      const bindingEnergy = this.cellularPhysics.getBindingEnergy(this, neighbor);
      
      // Thermal energy can break bonds
      const thermalEnergy = this.cellularPhysics.BOLTZMANN_CONSTANT * 
                           this.cellularPhysics.TEMPERATURE;
      
      // Bond breaks if stretched too far or thermal fluctuation
      if (distance > 20 || Math.random() < Math.exp(-bindingEnergy / thermalEnergy)) {
        toRemove.add(neighborId);
        neighbor.boundCells.delete(this.id);
      }
    });
    
    // Remove broken bonds
    toRemove.forEach(id => this.boundCells.delete(id));
    
    // If no bonds remain, cell detaches from colony
    if (this.boundCells.size === 0) {
      this.colony = null;
      this.attachedTo = null;
    }
  }
  
  findCellById(id) {
    // In real implementation, would search through simulation organisms
    // For now, return null (would be implemented in Simulation.js)
    return null;
  }
}