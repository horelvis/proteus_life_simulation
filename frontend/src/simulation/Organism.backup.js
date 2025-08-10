/**
 * PROTEUS Organism - Frontend Implementation
 */

import { ProteusInheritance } from './ProteusInheritance';

export class Organism {
  constructor(x, y, topologyEngine, parent1 = null, parent2 = null) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.position = { x, y };
    this.velocity = { x: 0, y: 0 };
    this.topologyEngine = topologyEngine;
    
    // Smooth movement
    this.smoothPosition = { x, y };
    this.targetVelocity = { x: 0, y: 0 };
    
    // Bio properties
    this.energy = 1.0;
    this.age = 0;
    this.maxAge = 80 + Math.random() * 40; // Lifespan varies 80-120 time units
    this.senescenceStartAge = this.maxAge * 0.6; // Aging starts at 60% of max age
    this.hunger = 0.5;
    this.alive = true;
    this.isPanicked = false;
    this.agingFactor = 1.0; // Multiplier for aging effects
    
    // TRI-LAYER INHERITANCE SYSTEM
    this.inheritance = new ProteusInheritance(
      parent1?.inheritance,
      parent2?.inheritance
    );
    
    // Generation from inheritance
    this.generation = this.inheritance.environmentalTraces.generation;
    
    // Visual based on inheritance
    this.color = this.generateColor();
    
    // Topological state influenced by inheritance
    this.topologicalState = {
      position: [x, y],
      momentum: { x: 0, y: 0 },
      manifold: this.inheritance.topologicalCore.manifoldDimension
    };
    
    // Experience tracking for holographic memory
    this.experienceBuffer = [];
    this.trajectory = [];
    
    // Express phenotype from inheritance
    this.phenotype = null; // Will be set in applyPhenotype
    this.applyPhenotype();
  }
  
  generateColor() {
    // Blue-green color scheme with genetic variation
    const core = this.inheritance.topologicalCore;
    // Base hue in blue-green range (160-200)
    const hue = 170 + core.manifoldDimension * 15 + core.bodySymmetry * 3;
    const saturation = 40 + core.baseSensitivity * 20;
    const lightness = 45 + core.baseResilience * 10;
    return `hsl(${hue % 200}, ${saturation}%, ${lightness}%)`;
  }
  
  applyPhenotype() {
    const phenotype = this.inheritance.phenotype;
    
    // Determine phenotype name based on dominant traits
    this.phenotype = this.determinePhenotypeType(phenotype);
    
    // SIMPLIFIED: Express ALL organs, no thresholds
    this.organs = [];
    const expressions = phenotype.organExpressions;
    
    // Add every organ with its expression level
    for (const [organType, expression] of Object.entries(expressions)) {
      if (expression > 0.01) { // Minimal threshold just to avoid zero
        this.organs.push({
          type: organType,
          functionality: expression
        });
      }
    }
    
    // SIMPLIFIED: Sum all relevant organs for each capability
    this.capabilities = {
      vision: 0,
      chemotaxis: 0,
      motility: phenotype.motility,
      protection: phenotype.resilience,
      efficiency: 1.0,
      toxicity: 0,
      regeneration: 0,
      stealth: 0,
      electric: 0,
      social: 0
    };
    
    // Let organs contribute to multiple capabilities
    this.organs.forEach(organ => {
      switch(organ.type) {
        case 'photosensor': this.capabilities.vision += organ.functionality; break;
        case 'chemoreceptor': this.capabilities.chemotaxis += organ.functionality; break;
        case 'speed_boost': 
        case 'flagellum': this.capabilities.motility += organ.functionality * 0.3; break;
        case 'armor_plates':
        case 'membrane': this.capabilities.protection += organ.functionality * 0.3; break;
        case 'toxin_gland': this.capabilities.toxicity += organ.functionality; break;
        case 'regeneration': this.capabilities.regeneration += organ.functionality; break;
        case 'camouflage': this.capabilities.stealth += organ.functionality; break;
        case 'electric_organ': this.capabilities.electric += organ.functionality; break;
        case 'vacuole': this.capabilities.efficiency += organ.functionality * 0.2; break;
        case 'pheromone_emitter': this.capabilities.social += organ.functionality; break;
      }
    });
  }
  
  // Record experience for holographic memory
  determinePhenotypeType(phenotype) {
    // SIMPLIFIED: Just name by dominant organ
    let maxOrgan = 'basic';
    let maxValue = 0;
    
    for (const [organ, value] of Object.entries(phenotype.organExpressions)) {
      if (value > maxValue) {
        maxValue = value;
        maxOrgan = organ;
      }
    }
    
    return maxOrgan;
  }
  
  recordExperience(type, importance = 0.5) {
    const experience = {
      type,
      importance,
      position: { ...this.position },
      trajectory: this.trajectory.slice(-20), // Last 20 positions
      timestamp: Date.now(),
      energy: this.energy,
      predatorDistance: this.lastPredatorDistance || null,
      nutrientDirection: this.lastNutrientDirection || null,
      emotionalValue: this.isPanicked ? 'fear' : 'neutral',
      survivalValue: this.energy > 0.5 ? 1 : 0
    };
    
    // Store in buffer
    this.experienceBuffer.push(experience);
    
    // Process significant experiences immediately
    if (importance > 0.7) {
      this.inheritance.experience(experience);
      
      // Let natural selection handle evolution, not directed evolution
      
      this.applyPhenotype(); // Update behavior based on new memory
    }
  }
  
  update(deltaTime, nutrients, predators) {
    if (!this.alive) return;
    
    this.age += deltaTime;
    this.hunger = Math.min(1.0, this.hunger + deltaTime * 0.01);
    
    // Calculate aging effects
    this.updateAging();
    
    // Update trajectory for memory
    this.trajectory.push({ x: this.position.x, y: this.position.y });
    if (this.trajectory.length > 100) this.trajectory.shift();
    
    // Get behavioral modulation from holographic memory
    const memoryContext = {
      danger: this.isPanicked ? 1 : 0,
      hunger: this.hunger,
      energy: this.energy
    };
    const behavior = this.inheritance.holographicMemory.recall(memoryContext);
    
    // Topological flow modulated by memory-driven behavior
    const flow = this.topologyEngine.computeTopologicalFlow(
      this.position,
      this.topologicalState
    );
    
    // Smooth acceleration with behavioral influence
    const accel = {
      x: flow.x * this.capabilities.motility * (1 + behavior.explorationTendency * 0.5),
      y: flow.y * this.capabilities.motility * (1 + behavior.explorationTendency * 0.5)
    };
    
    // SURVIVAL INSTINCT: Detect and flee from predators
    const predatorThreat = this.detectPredatorThreat(predators);
    if (predatorThreat) {
      const { predator, distance } = predatorThreat;
      
      // Panic response based on detection quality
      // Organisms with better eyes see danger earlier and can escape better
      const photosensors = this.organs.filter(o => o.type === 'photosensor');
      const visionQuality = photosensors.reduce((sum, s) => sum + s.functionality, 0);
      
      // No eyes = late detection = extreme panic
      const panicLevel = photosensors.length === 0 ? 1.0 : Math.max(0, 1 - distance / 150);
      const escapeForce = (0.5 + visionQuality) * panicLevel * 3;
      
      // Enhanced by memory-based fearfulness
      const memoryFear = behavior.cautionLevel;
      const totalEscapeForce = escapeForce * (1 + memoryFear);
      
      // Calculate escape direction (opposite of predator)
      const dx = this.position.x - predator.position.x;
      const dy = this.position.y - predator.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      this.lastPredatorDistance = dist;
      
      if (dist > 0) {
        // Override other behaviors when in danger
        accel.x = (dx / dist) * totalEscapeForce;
        accel.y = (dy / dist) * totalEscapeForce;
        
        // Increase speed temporarily (adrenaline response)
        this.capabilities.motility = Math.min(1.0, this.capabilities.motility * 1.1); // Reduced panic boost
      }
      
      // Forget about food when life is at stake
      if (panicLevel > 0.5) {
        // Complete override - survival first
        accel.x *= 2;
        accel.y *= 2;
        this.isPanicked = true;
        
        // Record escape experience
        if (Math.random() < deltaTime) { // Sample experiences
          this.recordExperience('predator_escape', panicLevel);
        }
      } else {
        this.isPanicked = false;
      }
    } else {
      // Restore normal motility when safe
      if (this.capabilities.motility > 0.5) {
        this.capabilities.motility *= 0.95;
      }
      this.isPanicked = false;
      
      // Hunger-driven nutrient seeking (only when safe)
      if (this.capabilities.chemotaxis > 0 && nutrients.length > 0) {
        const nearest = this.findNearestNutrient(nutrients);
        if (nearest && nearest.distance < 100) {
          const dx = nearest.nutrient.x - this.position.x;
          const dy = nearest.nutrient.y - this.position.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          if (dist > 0) {
            const hungerBoost = 1 + this.hunger * 3;
            const forageBoost = 1 + behavior.forageIntensity;
            accel.x += (dx / dist) * this.capabilities.chemotaxis * hungerBoost * forageBoost * 0.5;
            accel.y += (dy / dist) * this.capabilities.chemotaxis * hungerBoost * forageBoost * 0.5;
            
            this.lastNutrientDirection = { x: dx / dist, y: dy / dist };
          }
        }
      }
    }
    
    // Smooth velocity update
    const smoothing = 0.92; // High smoothing for natural movement
    this.velocity.x = this.velocity.x * smoothing + accel.x * deltaTime;
    this.velocity.y = this.velocity.y * smoothing + accel.y * deltaTime;
    
    // Speed limit
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    const maxSpeed = 1.2 * this.capabilities.motility; // Reduced from 2.0 for more natural speed
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Update position with NaN protection
    if (!isNaN(this.velocity.x) && !isNaN(this.velocity.y)) {
      this.position.x += this.velocity.x;
      this.position.y += this.velocity.y;
    } else {
      console.warn('NaN velocity detected in organism:', this.id, this.velocity);
      this.velocity.x = 0;
      this.velocity.y = 0;
    }
    
    // Smooth visual position for rendering
    const posSmoothing = 0.3;
    this.smoothPosition.x += (this.position.x - this.smoothPosition.x) * posSmoothing;
    this.smoothPosition.y += (this.position.y - this.smoothPosition.y) * posSmoothing;
    
    // World bounds with smooth wrapping
    const worldWidth = this.topologyEngine.worldSize.width;
    const worldHeight = this.topologyEngine.worldSize.height;
    
    // Handle X wrapping
    if (this.position.x < 0) {
      this.position.x = worldWidth + this.position.x;
      this.smoothPosition.x = worldWidth + this.smoothPosition.x; // Sync smooth position
    }
    if (this.position.x > worldWidth) {
      this.position.x = this.position.x - worldWidth;
      this.smoothPosition.x = this.smoothPosition.x - worldWidth; // Sync smooth position
    }
    
    // Handle Y wrapping
    if (this.position.y < 0) {
      this.position.y = worldHeight + this.position.y;
      this.smoothPosition.y = worldHeight + this.smoothPosition.y; // Sync smooth position
    }
    if (this.position.y > worldHeight) {
      this.position.y = this.position.y - worldHeight;
      this.smoothPosition.y = this.smoothPosition.y - worldHeight; // Sync smooth position
    }
    
    // Energy loss affected by efficiency (vacuoles)
    let energyLoss = deltaTime * 0.015 / (1 + this.capabilities.efficiency * 0.5);
    
    // Regeneration reduces energy loss
    if (this.capabilities.regeneration > 0) {
      energyLoss *= (1 - this.capabilities.regeneration * 0.3);
    }
    
    // Check if organism is in safe zone
    if (this.environmentalMemory && this.environmentalMemory.inSafeZone) {
      // Reduced energy loss in safe zones (50% reduction)
      energyLoss *= 0.5;
      
      // Slow energy regeneration in safe zones if hungry
      if (this.energy < 1.0 && this.hunger > 0.3) {
        this.energy = Math.min(1.0, this.energy + deltaTime * 0.01);
        this.hunger = Math.max(0, this.hunger - deltaTime * 0.02);
      }
    }
    
    this.energy -= energyLoss;
    
    // Death conditions
    if (this.energy <= 0) {
      this.die('starvation');
    } else if (this.age >= this.maxAge) {
      this.die('old_age');
    }
    
    // Update topological state
    this.topologicalState.position = [this.position.x, this.position.y];
    this.topologicalState.momentum = { x: this.velocity.x, y: this.velocity.y };
  }
  
  findNearestNutrient(nutrients) {
    let nearest = null;
    let minDist = Infinity;
    
    nutrients.forEach(nutrient => {
      if (!nutrient.alive) return;
      
      const dx = nutrient.x - this.position.x;
      const dy = nutrient.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < minDist) {
        minDist = dist;
        nearest = { nutrient, distance: dist };
      }
    });
    
    return nearest;
  }
  
  getToxinDefense() {
    // Toxin defense against predators
    return this.capabilities.toxicity || 0;
  }
  
  getElectricDefense() {
    // Electric defense can stun predators
    return this.capabilities.electric || 0;
  }
  
  
  detectPredatorThreat(predators) {
    if (!predators || predators.length === 0) return null;
    
    // Find photosensor organs
    const photosensors = this.organs.filter(o => o.type === 'photosensor');
    if (photosensors.length === 0) {
      // NO EYES = NO LIGHT DETECTION
      // Can only detect by touch/vibration at very close range
      let touchDetection = null;
      predators.forEach(predator => {
        if (!predator.alive) return;
        const dx = predator.position.x - this.position.x;
        const dy = predator.position.y - this.position.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        // Only detect by physical proximity
        if (dist < 15) { // Touch range
          touchDetection = { predator, distance: dist };
        }
      });
      return touchDetection;
    }
    
    // With photosensors, can detect light flashes
    let closestThreat = null;
    let bestDetection = Infinity;
    
    predators.forEach(predator => {
      if (!predator.alive) return;
      
      const dx = predator.position.x - this.position.x;
      const dy = predator.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Each photosensor contributes to detection
      let totalDetectionPower = 0;
      photosensors.forEach(sensor => {
        // Sensor efficiency decreases with distance
        const sensorRange = sensor.functionality * 150; // Max 150 units for perfect sensor
        if (dist < sensorRange) {
          // Can see the light flash
          const clarity = 1 - (dist / sensorRange);
          totalDetectionPower += clarity * sensor.functionality;
        }
      });
      
      // Multiple sensors improve detection
      if (totalDetectionPower > 0 && dist < bestDetection) {
        // Random chance based on sensor quality
        if (Math.random() < totalDetectionPower) {
          bestDetection = dist;
          closestThreat = { predator, distance: dist };
        }
      }
    });
    
    return closestThreat;
  }
  
  feed(amount) {
    this.energy = Math.min(2.0, this.energy + amount);
    this.hunger = Math.max(0, this.hunger - amount * 2);
    
    // Record feeding experience
    this.recordExperience('food_found', 0.6);
  }
  
  canReproduce() {
    // SIMPLIFIED: Just energy threshold
    return this.energy > 1.5 && this.age > 5;
  }
  
  reproduce() {
    this.energy *= 0.5;
    
    // Record reproduction experience
    this.recordExperience('reproduction', 0.8);
    
    // Process all buffered experiences before reproduction
    this.experienceBuffer.forEach(exp => {
      if (exp.importance > 0.3) {
        this.inheritance.experience(exp);
      }
    });
    
    // Find a mate if nearby (for sexual reproduction)
    // For now, asexual reproduction with self as both parents
    return new Organism(
      this.position.x + (Math.random() - 0.5) * 20,
      this.position.y + (Math.random() - 0.5) * 20,
      this.topologyEngine,
      this, // parent1
      null  // parent2 (asexual for now)
    );
  }
  
  updateAging() {
    // No aging effects in youth
    if (this.age < this.senescenceStartAge) {
      this.agingFactor = 1.0;
      return;
    }
    
    // Calculate aging progress (0 to 1)
    const agingProgress = (this.age - this.senescenceStartAge) / 
                         (this.maxAge - this.senescenceStartAge);
    
    // Aging effects increase exponentially
    this.agingFactor = 1 + agingProgress * agingProgress * 2; // Up to 3x at max age
    
    // Reduce capabilities with age
    const capabilityReduction = 1 - (agingProgress * 0.5); // Up to 50% reduction
    
    // Apply aging to capabilities
    this.capabilities.motility *= capabilityReduction;
    this.capabilities.vision *= capabilityReduction;
    this.capabilities.chemotaxis *= capabilityReduction;
    this.capabilities.efficiency *= capabilityReduction;
    
    // Organs degrade with age
    this.organs.forEach(organ => {
      organ.functionality *= (1 - agingProgress * 0.01); // Slow degradation
    });
  }
  
  
  // Death with memory traces
  die(cause = 'unknown') {
    this.alive = false;
    this.deathCause = cause;
    
    // Leave strong environmental trace on death
    const deathExperience = {
      type: 'death',
      importance: 1.0,
      position: { ...this.position },
      trajectory: this.trajectory.slice(-50),
      timestamp: Date.now(),
      energy: 0,
      emotionalValue: 'terminal',
      survivalValue: -1,
      cause: cause,
      age: this.age
    };
    
    this.inheritance.experience(deathExperience);
    
    // Different memory impacts based on death cause
    if (cause === 'old_age') {
      // Natural death leaves peaceful memory
      this.inheritance.experience({
        type: 'natural_death',
        importance: 0.5,
        position: this.position,
        age: this.age
      });
    } else if (cause === 'starvation') {
      // Starvation leaves warning memory
      this.inheritance.experience({
        type: 'starvation_death',
        importance: 0.8,
        position: this.position,
        lastNutrientDirection: this.lastNutrientDirection
      });
    }
    
    // Return environmental traces for the world to absorb
    return this.inheritance.environmentalTraces;
  }
}