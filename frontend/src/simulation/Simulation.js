/**
 * PROTEUS Simulation - Frontend Implementation
 */

import { TopologyEngine } from './TopologyEngine';
import { Organism } from './Organism';
import { Nutrient } from './Nutrient';
import { Predator } from './Predator';
import { EnvironmentalField } from './EnvironmentalField';

export class Simulation {
  constructor(worldSize) {
    this.worldSize = worldSize;
    this.topologyEngine = new TopologyEngine(worldSize);
    this.environmentalField = new EnvironmentalField(worldSize);
    
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    this.safeZones = this.generateSafeZones();
    
    this.time = 0;
    this.generation = 0;
    this.running = false;
    this.speedMultiplier = 1.0;
    
    this.statistics = {
      totalOrganisms: 0,
      births: 0,
      deaths: 0,
      averageAge: 0,
      averageEnergy: 0
    };
    
    this.lastUpdate = Date.now();
  }
  
  generateSafeZones() {
    return [
      {
        x: this.worldSize.width * 0.2,
        y: this.worldSize.height * 0.3,
        radius: 80
      },
      {
        x: this.worldSize.width * 0.8,
        y: this.worldSize.height * 0.7,
        radius: 80
      }
    ];
  }
  
  initialize(config = {}) {
    const {
      initialOrganisms = 30,
      initialNutrients = 50,
      initialPredators = 8
    } = config;
    
    // Create initial organisms
    for (let i = 0; i < initialOrganisms; i++) {
      const x = Math.random() * this.worldSize.width;
      const y = Math.random() * this.worldSize.height;
      this.organisms.push(new Organism(x, y, this.topologyEngine));
    }
    
    // Create initial nutrients
    for (let i = 0; i < initialNutrients; i++) {
      this.spawnNutrient();
    }
    
    // Create initial predators
    for (let i = 0; i < initialPredators; i++) {
      const x = Math.random() * this.worldSize.width;
      const y = Math.random() * this.worldSize.height;
      this.predators.push(new Predator(x, y, this.worldSize));
    }
    
    this.updateStatistics();
  }
  
  start() {
    this.running = true;
    this.lastUpdate = Date.now();
    this.update();
  }
  
  stop() {
    this.running = false;
  }
  
  setSpeed(multiplier) {
    this.speedMultiplier = Math.max(0.1, Math.min(2.0, multiplier));
  }
  
  update() {
    if (!this.running) return;
    
    const now = Date.now();
    const deltaTime = (now - this.lastUpdate) / 1000;
    this.lastUpdate = now;
    
    // Cap deltaTime to prevent huge jumps and apply speed multiplier
    const dt = Math.min(deltaTime, 0.05) * this.speedMultiplier;
    
    this.time += dt;
    
    // Update topology field
    this.topologyEngine.updateField(dt);
    
    // Update organisms
    this.organisms.forEach(organism => {
      // Let organisms read environmental memory
      const envMemory = this.environmentalField.readEnvironment(organism.position);
      organism.environmentalMemory = envMemory;
      
      organism.update(dt, this.nutrients, this.predators);
      
      // Deposit traces from organism's current state
      if (organism.alive && organism.inheritance) {
        const traces = organism.inheritance.environmentalTraces.fieldModulations;
        traces.forEach(trace => {
          this.environmentalField.depositTrace(trace);
        });
      }
      
      // Check feeding
      if (organism.alive) {
        this.checkFeeding(organism);
        
        // Check reproduction
        if (organism.canReproduce()) {
          const offspring = organism.reproduce();
          this.organisms.push(offspring);
          this.statistics.births++;
        }
      }
    });
    
    // Update nutrients
    this.nutrients.forEach(nutrient => {
      nutrient.update(dt);
    });
    
    // Update predators
    this.predators.forEach(predator => {
      predator.update(dt, this.organisms, this.safeZones);
    });
    
    // Handle dead organisms and their memory traces
    const deadOrganisms = this.organisms.filter(o => !o.alive);
    this.statistics.deaths += deadOrganisms.length;
    
    // Process death traces
    deadOrganisms.forEach(organism => {
      // Death leaves strong environmental traces
      const deathTrace = {
        position: organism.position,
        pheromone: new Float32Array([0, 0, 0, 1, 0]), // Death pheromone
        intensity: 2.0
      };
      this.environmentalField.depositTrace(deathTrace);
      
      // Add memory anchor for significant organisms
      if (organism.generation > 5 || organism.age > 20) {
        this.environmentalField.addMemoryAnchor({
          position: organism.position,
          type: 'death',
          memory: organism.inheritance.compress(),
          generation: organism.generation
        });
      }
    });
    
    // Remove dead entities
    this.organisms = this.organisms.filter(o => o.alive);
    this.nutrients = this.nutrients.filter(n => n.alive);
    this.predators = this.predators.filter(p => p.alive);
    
    // Spawn new nutrients periodically
    if (Math.random() < 0.05) {
      this.spawnNutrient();
    }
    
    // Maintain minimum population
    if (this.organisms.length < 5) {
      for (let i = 0; i < 5; i++) {
        const x = Math.random() * this.worldSize.width;
        const y = Math.random() * this.worldSize.height;
        this.organisms.push(new Organism(x, y, this.topologyEngine));
      }
    }
    
    // Maintain predator population
    if (this.predators.length < 3 && this.organisms.length > 10) {
      const x = Math.random() * this.worldSize.width;
      const y = Math.random() * this.worldSize.height;
      this.predators.push(new Predator(x, y, this.worldSize));
    }
    
    this.updateStatistics();
    
    // Update environmental field
    this.environmentalField.update(dt);
    
    // Continue animation loop
    requestAnimationFrame(() => this.update());
  }
  
  checkFeeding(organism) {
    const feedingRadius = 15;
    
    this.nutrients.forEach(nutrient => {
      if (!nutrient.alive) return;
      
      const dx = nutrient.x - organism.position.x;
      const dy = nutrient.y - organism.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < feedingRadius) {
        organism.feed(nutrient.energy);
        nutrient.alive = false;
      }
    });
  }
  
  spawnNutrient() {
    const x = Math.random() * this.worldSize.width;
    const y = Math.random() * this.worldSize.height;
    this.nutrients.push(new Nutrient(x, y));
  }
  
  updateStatistics() {
    this.statistics.totalOrganisms = this.organisms.length;
    
    if (this.organisms.length > 0) {
      const totalAge = this.organisms.reduce((sum, o) => sum + o.age, 0);
      const totalEnergy = this.organisms.reduce((sum, o) => sum + o.energy, 0);
      const totalGeneration = this.organisms.reduce((sum, o) => sum + o.generation, 0);
      const totalMutationRate = this.organisms.reduce((sum, o) => 
        sum + (o.inheritance?.topologicalCore?.mutability || 0), 0);
      
      this.statistics.averageAge = totalAge / this.organisms.length;
      this.statistics.averageEnergy = totalEnergy / this.organisms.length;
      this.statistics.average_generation = totalGeneration / this.organisms.length;
      this.statistics.average_mutation_rate = totalMutationRate / this.organisms.length;
      
      // Find highest generation
      this.statistics.highest_generation = Math.max(...this.organisms.map(o => o.generation));
      
      // Count mutations
      this.statistics.mutations = this.organisms.filter(o => 
        o.inheritance?.topologicalCore?.hasMutation).length;
      
      // Count memory anchors
      this.statistics.memory_anchors = this.environmentalField.memoryAnchors.length;
    } else {
      this.statistics.averageAge = 0;
      this.statistics.averageEnergy = 0;
      this.statistics.average_generation = 0;
      this.statistics.highest_generation = 0;
      this.statistics.average_mutation_rate = 0;
      this.statistics.memory_anchors = 0;
    }
  }
  
  getState() {
    return {
      organisms: this.organisms.map(o => ({
        id: o.id,
        position: o.smoothPosition, // Use smooth position for rendering
        velocity: o.velocity,
        color: o.color,
        energy: o.energy,
        age: o.age,
        generation: o.generation,
        organs: o.organs,
        isPanicked: o.isPanicked,
        inheritance: o.inheritance,
        capabilities: o.capabilities
      })),
      nutrients: this.nutrients.map(n => ({
        id: n.id,
        x: n.x,
        y: n.y,
        energy_value: n.energy,
        size: n.size
      })),
      predators: this.predators.map(p => ({
        id: p.id,
        position: p.smoothPosition,
        velocity: p.velocity,
        size: p.size,
        light_intensity: p.glowIntensity,
        light_flash: p.lightFlash,
        tentacles: p.tentacles
      })),
      safeZones: this.safeZones,
      statistics: this.statistics,
      time: this.time,
      environmentalField: this.environmentalField.getVisualizationData(),
      memoryAnchors: this.environmentalField.memoryAnchors
    };
  }
}