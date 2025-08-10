/**
 * PROTEUS Organism - Evolution-First Design
 * No artificial thresholds, pure natural selection
 */

import { ProteusInheritance } from './ProteusInheritance';

export class Organism {
  constructor(x, y, topologyEngine, parentInheritance = null) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.position = { x, y };
    this.velocity = { x: 0, y: 0 };
    this.smoothPosition = { x, y };
    
    // Core survival stats
    this.age = 0;
    this.energy = 1.0; // Start with full energy
    this.alive = true;
    this.generation = 0;
    
    // Inheritance system
    this.inheritance = parentInheritance || new ProteusInheritance();
    this.topologyEngine = topologyEngine;
    
    // Express phenotype immediately
    this.organs = [];
    this.capabilities = {};
    this.applyPhenotype();
    
    // Behavioral state
    this.trajectory = [];
    this.lastAction = null;
    this.hunger = 0.5;
    this.deathCause = null;
    
    // Additional properties for compatibility
    this.color = this.generateColor();
    this.maxAge = 50 + this.inheritance.topologicalCore.baseResilience * 100;
    this.phenotype = 'evolving'; // Will be updated
    this.isPanicked = false;
    this.environmentalMemory = null;
  }
  
  generateColor() {
    const core = this.inheritance.topologicalCore;
    const hue = 170 + core.manifoldDimension * 15 + core.bodySymmetry * 3;
    const saturation = 40 + core.baseSensitivity * 20;
    const lightness = 45 + core.baseResilience * 10;
    return `hsl(${hue % 200}, ${saturation}%, ${lightness}%)`;
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
    // Energy cost per time unit - heavier organs cost more
    const costs = {
      photosensor: 0.001,
      chemoreceptor: 0.001,
      flagellum: 0.002,
      speed_boost: 0.003,
      membrane: 0.0005,
      armor_plates: 0.002,
      toxin_gland: 0.0015,
      electric_organ: 0.004,
      regeneration: 0.002,
      camouflage: 0.001,
      vacuole: 0.0005,
      pheromone_emitter: 0.001
    };
    return costs[organType] || 0.001;
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
      }
    };
    
    return effects[organ.type] || {};
  }
  
  update(deltaTime, nutrients, predators) {
    if (!this.alive) return;
    
    this.age += deltaTime;
    
    // Energy consumption - base metabolism + organ costs
    let energyCost = 0.001 * deltaTime; // Base metabolism
    this.organs.forEach(organ => {
      energyCost += organ.energyCost * deltaTime;
    });
    
    // Create environment object for perception
    const environment = {
      nutrients: nutrients || [],
      predators: predators || [],
      organisms: [] // Will be filled by simulation if needed
    };
    
    // Movement decision based on perception
    const perception = this.perceive(environment);
    const decision = this.decide(perception);
    this.act(decision, deltaTime);
    
    // Energy cost of movement
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    energyCost += speed * 0.0001 * deltaTime;
    
    // Apply energy changes
    this.energy -= energyCost;
    
    // Regeneration if has the organ
    const regen = this.organs.find(o => o.type === 'regeneration');
    if (regen && this.energy < 2.0) {
      this.energy += regen.functionality * 0.01 * deltaTime;
    }
    
    // Death is natural - no artificial thresholds
    if (this.energy <= 0) {
      this.die('starvation');
    }
    
    // Aging is also natural - lifespan varies by genetics
    if (this.age > this.maxAge) {
      this.die('old_age');
    }
    
    // Update position
    this.position.x += this.velocity.x * deltaTime;
    this.position.y += this.velocity.y * deltaTime;
    
    // Smooth position for rendering
    this.smoothPosition.x += (this.position.x - this.smoothPosition.x) * 0.3;
    this.smoothPosition.y += (this.position.y - this.smoothPosition.y) * 0.3;
    
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
    
    // Visual perception - only with photosensors
    const photosensors = this.organs.filter(o => o.type === 'photosensor');
    if (photosensors.length > 0) {
      const visionRange = photosensors.reduce((sum, s) => 
        sum + s.functionality * 100, 0
      );
      
      // See predators by their light
      environment.predators.forEach(predator => {
        const dist = this.distanceTo(predator.position);
        if (dist < visionRange && predator.lightFlash) {
          perception.predators.push({
            position: predator.position,
            distance: dist,
            clarity: 1 - (dist / visionRange)
          });
        }
      });
      
      // See organisms
      environment.organisms.forEach(org => {
        if (org.id === this.id) return;
        const dist = this.distanceTo(org.position);
        if (dist < visionRange * 0.5) { // Organisms are harder to see
          perception.organisms.push({
            position: org.position,
            distance: dist,
            moving: Math.sqrt(org.velocity.x ** 2 + org.velocity.y ** 2) > 0.1
          });
        }
      });
    }
    
    // Chemical perception - only with chemoreceptors
    const chemoreceptors = this.organs.filter(o => o.type === 'chemoreceptor');
    if (chemoreceptors.length > 0) {
      const chemRange = chemoreceptors.reduce((sum, c) => 
        sum + c.functionality * 80, 0
      );
      
      // Smell nutrients
      environment.nutrients.forEach(nutrient => {
        const dist = this.distanceTo(nutrient);
        if (dist < chemRange) {
          perception.nutrients.push({
            position: nutrient,
            distance: dist,
            strength: (1 - dist / chemRange) * nutrient.energy
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
    // Decision making based on perception and state
    const decision = {
      move: { x: 0, y: 0 },
      useAbility: null
    };
    
    // Threat avoidance - highest priority
    if (perception.predators.length > 0) {
      const nearest = perception.predators.reduce((a, b) => 
        a.distance < b.distance ? a : b
      );
      
      // Flee in opposite direction
      const flee = this.vectorFrom(nearest.position);
      decision.move.x = flee.x * 2; // Emergency speed
      decision.move.y = flee.y * 2;
      
      // Use defensive abilities if available
      const toxin = this.organs.find(o => o.type === 'toxin_gland');
      if (toxin && nearest.distance < toxin.functionality * 10) {
        decision.useAbility = 'release_toxin';
      }
      
      const electric = this.organs.find(o => o.type === 'electric_organ');
      if (electric && nearest.distance < electric.functionality * 20) {
        decision.useAbility = 'electric_discharge';
      }
      
      // Record fear experience
      this.inheritance.experience({
        type: 'predator_encounter',
        position: [this.position.x, this.position.y],
        importance: nearest.clarity,
        trajectory: this.trajectory.slice(-10)
      });
    }
    // Food seeking - when safe
    else if (perception.nutrients.length > 0 && this.energy < 1.8) {
      const best = perception.nutrients.reduce((a, b) => 
        a.strength > b.strength ? a : b
      );
      
      const approach = this.vectorTo(best.position);
      decision.move.x = approach.x;
      decision.move.y = approach.y;
    }
    // Social behavior - follow pheromones
    else if (Object.keys(perception.chemicals).length > 0) {
      let strongestSignal = null;
      let maxStrength = 0;
      
      for (const [id, chem] of Object.entries(perception.chemicals)) {
        if (chem.strength > maxStrength) {
          maxStrength = chem.strength;
          strongestSignal = chem;
        }
      }
      
      if (strongestSignal && this.energy > 1.2) { // Only if well-fed
        const approach = this.vectorTo(strongestSignal.position);
        decision.move.x = approach.x * 0.5; // Gentle approach
        decision.move.y = approach.y * 0.5;
      }
    }
    // Exploration - when nothing else to do
    else {
      // Use topological engine for intelligent exploration
      const flow = this.topologyEngine.getFlowAt(this.position);
      decision.move.x = flow.x * 0.3;
      decision.move.y = flow.y * 0.3;
      
      // Add some randomness based on curiosity
      const curiosity = this.inheritance.expressPhenotype().curiosity || 0.5;
      decision.move.x += (Math.random() - 0.5) * curiosity;
      decision.move.y += (Math.random() - 0.5) * curiosity;
    }
    
    return decision;
  }
  
  act(decision, deltaTime) {
    // Execute decision
    
    // Movement
    const maxSpeed = this.getMaxSpeed();
    const acceleration = this.getAcceleration();
    
    // Apply acceleration
    this.velocity.x += decision.move.x * acceleration * deltaTime;
    this.velocity.y += decision.move.y * acceleration * deltaTime;
    
    // Limit to max speed
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Natural friction
    this.velocity.x *= 0.95;
    this.velocity.y *= 0.95;
    
    // Use abilities
    if (decision.useAbility) {
      this.useAbility(decision.useAbility);
    }
    
    this.lastAction = decision;
  }
  
  getMaxSpeed() {
    let speed = 1.0; // Base speed
    
    // Each movement organ adds to speed
    this.organs.forEach(organ => {
      if (organ.type === 'flagellum') {
        speed += organ.functionality * 2;
      } else if (organ.type === 'speed_boost') {
        speed += organ.functionality * 3;
      } else if (organ.type === 'armor_plates') {
        speed -= organ.functionality * 0.5; // Heavy armor slows
      }
    });
    
    return Math.max(0.1, speed); // Never completely immobile
  }
  
  getAcceleration() {
    let accel = 0.5; // Base acceleration
    
    this.organs.forEach(organ => {
      if (organ.type === 'flagellum') {
        accel += organ.functionality * 0.3;
      } else if (organ.type === 'speed_boost') {
        accel += organ.functionality * 0.5;
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
    // Reproduction depends on energy and age - NO FIXED THRESHOLDS
    // Energy cost of reproduction
    const reproductionCost = 0.5 + this.organs.length * 0.1;
    
    // Must have excess energy
    if (this.energy <= reproductionCost + 0.5) return false;
    
    // Age affects reproduction - young and old less likely
    const maturity = Math.sin((this.age / 50) * Math.PI);
    
    // Probability based on energy and maturity
    const probability = (this.energy - 1.0) * maturity;
    
    return Math.random() < probability * 0.1; // 10% max chance per update
  }
  
  reproduce() {
    // Create offspring with mutation
    const childX = this.position.x + (Math.random() - 0.5) * 20;
    const childY = this.position.y + (Math.random() - 0.5) * 20;
    
    const child = new Organism(childX, childY, this.topologyEngine, this.inheritance.reproduce());
    child.generation = this.generation + 1;
    
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
        'filter_feeding', 'symbiotic_chamber', 'neural_ganglion'
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
}