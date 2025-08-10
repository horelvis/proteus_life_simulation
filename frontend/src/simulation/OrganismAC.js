/**
 * Simplified Organism using Cellular Automaton for control
 * No neurons, no complex systems - just AC + physics
 */

import { CellularAutomaton } from './CellularAutomaton';

export class OrganismAC {
  constructor(x, y, inheritance = null) {
    // Basic properties
    this.id = Math.random().toString(36).substr(2, 9);
    this.position = { x, y };
    this.velocity = { x: 0, y: 0 };
    this.alive = true;
    this.age = 0;
    this.generation = 0;
    
    // Energy system
    this.energy = 1.0;
    this.maxEnergy = 2.0;
    
    // Physical properties
    this.size = 8;
    this.mass = 1.0;
    
    // Cellular Automaton - the brain
    this.cellularAutomaton = new CellularAutomaton(16);
    
    // Inheritance
    if (inheritance) {
      this.generation = inheritance.generation + 1;
      this.cellularAutomaton.inherit(inheritance.parentAC, 0.1);
    }
    
    // Damage tracking
    this.recentDamage = 0;
    this.damageDecay = 0.9;
    
    // Death tracking
    this.deathCause = null;
  }
  
  /**
   * Main update loop
   */
  update(deltaTime, environment) {
    if (!this.alive) return;
    
    this.age += deltaTime;
    
    // Energy consumption
    const baseMetabolism = 0.01 * deltaTime;
    const movementCost = (Math.abs(this.velocity.x) + Math.abs(this.velocity.y)) * 0.001 * deltaTime;
    this.energy -= (baseMetabolism + movementCost);
    
    // Create perception from environment
    const perception = this.perceive(environment);
    
    // Update CA sensors
    this.cellularAutomaton.updateSensors(perception);
    
    // Update CA state
    this.cellularAutomaton.update(deltaTime);
    
    // Apply effectors
    this.applyEffectors(deltaTime, environment);
    
    // Update position
    this.position.x += this.velocity.x * deltaTime;
    this.position.y += this.velocity.y * deltaTime;
    
    // World wrapping
    if (this.position.x < 0) this.position.x = 800;
    if (this.position.x > 800) this.position.x = 0;
    if (this.position.y < 0) this.position.y = 600;
    if (this.position.y > 600) this.position.y = 0;
    
    // Apply friction
    this.velocity.x *= 0.95;
    this.velocity.y *= 0.95;
    
    // Decay damage
    this.recentDamage *= this.damageDecay;
    
    // Death conditions
    if (this.energy <= 0) {
      this.alive = false;
      this.deathCause = 'starvation';
    }
  }
  
  /**
   * Perceive environment and convert to AC-compatible format
   */
  perceive(environment) {
    const perception = {
      nutrients: [],
      predators: [],
      organisms: [],
      pheromoneGradients: {}
    };
    
    // Detect nutrients with improved sensing
    if (environment.nutrients) {
      environment.nutrients.forEach(nutrient => {
        const dx = nutrient.x - this.position.x;
        const dy = nutrient.y - this.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 150) { // Increased detection range
          perception.nutrients.push({
            distance: distance,
            gradient_direction: Math.atan2(dy, dx),
            strength: nutrient.energy / (1 + distance * 0.005) // More sensitive
          });
        }
      });
    }
    
    // Detect predators (they emit light)
    if (environment.predators) {
      environment.predators.forEach(predator => {
        const dx = predator.position.x - this.position.x;
        const dy = predator.position.y - this.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 150 && predator.lightFlash > 0) {
          perception.predators.push({
            distance: distance,
            direction: Math.atan2(dy, dx),
            intensity: predator.lightFlash
          });
        }
      });
    }
    
    // Detect other organisms
    if (environment.organisms) {
      environment.organisms.forEach(org => {
        if (org.id === this.id || !org.alive) return;
        
        const dx = org.position.x - this.position.x;
        const dy = org.position.y - this.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 50) {
          perception.organisms.push({
            id: org.id,
            distance: distance,
            position: org.position
          });
        }
      });
    }
    
    // Detect pheromone gradients
    if (environment.pheromoneField) {
      const types = ['FOOD', 'DANGER', 'COLONY', 'MATING'];
      types.forEach(type => {
        const gradient = environment.pheromoneField.getGradient(
          this.position.x, 
          this.position.y, 
          type
        );
        if (gradient.x !== 0 || gradient.y !== 0) {
          perception.pheromoneGradients[type] = gradient;
        }
      });
    }
    
    return perception;
  }
  
  /**
   * Apply CA effector outputs to organism physics
   */
  applyEffectors(deltaTime, environment) {
    const ca = this.cellularAutomaton;
    
    // Movement from CA
    const moveForce = 50.0; // Base movement force
    const rigidityFactor = 1.0 - ca.effectors.rigidity * 0.5; // Rigid = slower
    
    this.velocity.x += ca.effectors.movementX * moveForce * rigidityFactor * deltaTime;
    this.velocity.y += ca.effectors.movementY * moveForce * rigidityFactor * deltaTime;
    
    // Cap velocity
    const maxSpeed = 100 * rigidityFactor;
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Size modulation from oscillation
    this.size = 8 + ca.effectors.oscillation * 2;
    
    // Pheromone secretion
    if (environment.pheromoneField && ca.effectors.chemicalSecretion > 0) {
      environment.pheromoneField.deposit(
        this.position.x,
        this.position.y,
        ca.effectors.secretionType,
        ca.effectors.chemicalSecretion * deltaTime
      );
    }
  }
  
  /**
   * Feed on nutrient
   */
  feed(amount) {
    const absorbed = amount * 0.8; // 80% efficiency
    this.energy = Math.min(this.maxEnergy, this.energy + absorbed);
  }
  
  /**
   * Take damage
   */
  takeDamage(amount, source = 'unknown') {
    this.energy -= amount;
    this.recentDamage += amount;
    
    // Inject damage into CA at random location
    const angle = Math.random() * Math.PI * 2;
    const relX = Math.cos(angle);
    const relY = Math.sin(angle);
    
    this.cellularAutomaton.injectDamage(relX, relY, amount);
    
    if (this.energy <= 0) {
      this.alive = false;
      this.deathCause = source;
    }
  }
  
  /**
   * Check if can reproduce
   */
  canReproduce() {
    // Need excess energy and minimum age
    return this.energy > 1.5 && this.age > 15 && Math.random() < 0.01; // Reduced reproduction rate
  }
  
  /**
   * Create offspring
   */
  reproduce() {
    // Split energy
    this.energy /= 2;
    
    // Create child nearby
    const angle = Math.random() * Math.PI * 2;
    const distance = 20;
    const childX = this.position.x + Math.cos(angle) * distance;
    const childY = this.position.y + Math.sin(angle) * distance;
    
    // Pass inheritance
    const inheritance = {
      generation: this.generation,
      parentAC: this.cellularAutomaton
    };
    
    const child = new OrganismAC(childX, childY, inheritance);
    child.energy = this.energy; // Share energy
    
    return child;
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData() {
    return {
      position: this.position,
      velocity: this.velocity,
      size: this.size,
      energy: this.energy,
      generation: this.generation,
      alive: this.alive,
      cellularState: this.cellularAutomaton.getVisualizationData(),
      effectors: { ...this.cellularAutomaton.effectors }
    };
  }
  
  /**
   * Calculate morphological stability metric
   */
  getMorphologicalStability() {
    const cells = this.cellularAutomaton.grid;
    let tissueCount = 0;
    let centerX = 0, centerY = 0;
    
    // Calculate center of mass
    for (let i = 0; i < cells.length; i++) {
      for (let j = 0; j < cells[i].length; j++) {
        if (cells[i][j].type === 'tissue') {
          tissueCount++;
          centerX += i;
          centerY += j;
        }
      }
    }
    
    if (tissueCount === 0) return 0;
    
    centerX /= tissueCount;
    centerY /= tissueCount;
    
    // Calculate variance
    let variance = 0;
    for (let i = 0; i < cells.length; i++) {
      for (let j = 0; j < cells[i].length; j++) {
        if (cells[i][j].type === 'tissue') {
          const dx = i - centerX;
          const dy = j - centerY;
          variance += dx * dx + dy * dy;
        }
      }
    }
    
    return tissueCount > 0 ? Math.sqrt(variance / tissueCount) : 0;
  }
  
  /**
   * Calculate foraging efficiency
   */
  getForagingEfficiency() {
    const distanceTraveled = Math.sqrt(
      this.velocity.x * this.velocity.x + 
      this.velocity.y * this.velocity.y
    ) * this.age;
    
    return distanceTraveled > 0 ? this.energy / distanceTraveled : 0;
  }
}