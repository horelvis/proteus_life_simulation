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
    this.hunger = 0.5;
    this.alive = true;
    this.isPanicked = false;
    
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
    this.applyPhenotype();
  }
  
  generateColor() {
    // Color influenced by topological core
    const core = this.inheritance.topologicalCore;
    const hue = 150 + core.manifoldDimension * 20 + core.bodySymmetry * 5;
    const saturation = 50 + core.baseSensitivity * 30;
    const lightness = 50 + core.baseResilience * 20;
    return `hsl(${hue % 360}, ${saturation}%, ${lightness}%)`;
  }
  
  applyPhenotype() {
    const phenotype = this.inheritance.phenotype;
    
    // Generate organs based on phenotype
    this.organs = [];
    const expressions = phenotype.organExpressions;
    
    if (expressions.photosensor > 0.3) {
      this.organs.push({
        type: 'photosensor',
        functionality: Math.min(1, expressions.photosensor)
      });
    }
    
    if (expressions.chemoreceptor > 0.3) {
      this.organs.push({
        type: 'chemoreceptor',
        functionality: Math.min(1, expressions.chemoreceptor)
      });
    }
    
    // Update capabilities from phenotype
    this.capabilities = {
      vision: this.organs.find(o => o.type === 'photosensor')?.functionality || 0,
      chemotaxis: this.organs.find(o => o.type === 'chemoreceptor')?.functionality || 0,
      motility: phenotype.motility,
      protection: phenotype.resilience,
      efficiency: 1.0,
      
      // New behavioral traits from memory
      curiosity: phenotype.curiosity,
      fearfulness: phenotype.fearfulness,
      foraging: phenotype.foraging,
      sociability: phenotype.sociability
    };
  }
  
  // Record experience for holographic memory
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
      this.applyPhenotype(); // Update behavior based on new memory
    }
  }
  
  update(deltaTime, nutrients, predators) {
    if (!this.alive) return;
    
    this.age += deltaTime;
    this.hunger = Math.min(1.0, this.hunger + deltaTime * 0.01);
    
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
      
      // Panic response based on proximity and vision capability
      const panicLevel = Math.max(0, 1 - distance / 150);
      const escapeForce = (0.5 + this.capabilities.vision * 2) * panicLevel * 3;
      
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
    
    // Update position
    this.position.x += this.velocity.x;
    this.position.y += this.velocity.y;
    
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
    
    // Energy consumption modulated by efficiency
    const energyLoss = deltaTime * 0.02 * (2 - this.capabilities.efficiency);
    this.energy -= energyLoss;
    
    if (this.energy <= 0) {
      this.die();
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
  
  detectPredatorThreat(predators) {
    if (!predators || predators.length === 0) return null;
    
    // Detection range depends on vision capability
    const detectionRange = 50 + this.capabilities.vision * 100;
    let closestThreat = null;
    let minDist = detectionRange;
    
    predators.forEach(predator => {
      if (!predator.alive) return;
      
      const dx = predator.position.x - this.position.x;
      const dy = predator.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Can detect predator if within range
      if (dist < minDist) {
        // Better vision = better detection at distance
        const detectionChance = this.capabilities.vision > 0 ? 
          1 - (dist / detectionRange) * (1 - this.capabilities.vision) : 
          dist < 30 ? 1 : 0; // Without vision, only detect very close predators
        
        if (Math.random() < detectionChance || dist < 30) {
          minDist = dist;
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
    // Reproduction influenced by phenotype
    const baseThreshold = 1.5 - this.capabilities.efficiency * 0.3;
    const ageThreshold = 5 - this.inheritance.topologicalCore.baseResilience * 2;
    return this.energy > baseThreshold && this.age > ageThreshold;
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
  
  // Death with memory traces
  die() {
    this.alive = false;
    
    // Leave strong environmental trace on death
    const deathExperience = {
      type: 'death',
      importance: 1.0,
      position: { ...this.position },
      trajectory: this.trajectory.slice(-50),
      timestamp: Date.now(),
      energy: 0,
      emotionalValue: 'terminal',
      survivalValue: -1
    };
    
    this.inheritance.experience(deathExperience);
    
    // Return environmental traces for the world to absorb
    return this.inheritance.environmentalTraces;
  }
}