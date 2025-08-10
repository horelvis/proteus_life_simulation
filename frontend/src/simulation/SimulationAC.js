/**
 * Simplified PROTEUS Simulation with AC-based organisms
 */

import { OrganismAC } from './OrganismAC';
import { Nutrient } from './Nutrient';
import { Predator } from './Predator';
import { PheromoneField } from './PheromoneField';

export class SimulationAC {
  constructor(worldSize) {
    this.worldSize = worldSize;
    
    // Entities
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    
    // Pheromone field
    this.pheromoneField = new PheromoneField(worldSize);
    
    // Time
    this.time = 0;
    this.generation = 0;
    this.running = false;
    
    // Statistics
    this.statistics = {
      totalOrganisms: 0,
      births: 0,
      deaths: 0,
      averageGeneration: 0,
      averageEnergy: 0,
      morphologicalStability: 0,
      foragingEfficiency: 0,
      cooperationIndex: 0,
      competitionIndex: 0,
      pheromoneActivity: {}
    };
    
    // Performance tracking
    this.lastUpdate = Date.now();
    this.frameCount = 0;
    this.fps = 0;
  }
  
  /**
   * Initialize simulation
   */
  initialize(config = {}) {
    const {
      initialOrganisms = 20,
      initialNutrients = 50,
      initialPredators = 1
    } = config;
    
    // Clear existing entities
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    
    // Spawn organisms
    for (let i = 0; i < initialOrganisms; i++) {
      const x = Math.random() * this.worldSize.width;
      const y = Math.random() * this.worldSize.height;
      this.organisms.push(new OrganismAC(x, y));
    }
    
    // Spawn nutrients
    for (let i = 0; i < initialNutrients; i++) {
      this.spawnNutrient();
    }
    
    // Spawn predators
    for (let i = 0; i < initialPredators; i++) {
      const x = Math.random() * this.worldSize.width;
      const y = Math.random() * this.worldSize.height;
      this.predators.push(new Predator(x, y, this.worldSize));
    }
    
    console.log('🚀 AC Simulation initialized:', {
      organisms: initialOrganisms,
      nutrients: initialNutrients,
      predators: initialPredators
    });
  }
  
  /**
   * Start simulation
   */
  start() {
    if (!this.running) {
      this.running = true;
      this.lastUpdate = Date.now();
      this.update();
    }
  }
  
  /**
   * Stop simulation
   */
  stop() {
    this.running = false;
  }
  
  /**
   * Main update loop
   */
  update() {
    if (!this.running) return;
    
    const now = Date.now();
    const deltaTime = Math.min((now - this.lastUpdate) / 1000, 0.1); // Cap at 100ms
    this.lastUpdate = now;
    
    this.time += deltaTime;
    this.frameCount++;
    
    // Update FPS every second
    if (this.frameCount % 60 === 0) {
      this.fps = 1 / deltaTime;
    }
    
    // Create environment for organisms
    const environment = {
      nutrients: this.nutrients,
      predators: this.predators,
      organisms: this.organisms,
      pheromoneField: this.pheromoneField
    };
    
    // Update organisms
    const newOrganisms = [];
    this.organisms.forEach(organism => {
      organism.update(deltaTime, environment);
      
      if (organism.alive) {
        // Check feeding
        this.checkFeeding(organism);
        
        // Check reproduction
        if (organism.canReproduce()) {
          const offspring = organism.reproduce();
          newOrganisms.push(offspring);
          this.statistics.births++;
        }
      }
    });
    
    // Add new organisms
    this.organisms.push(...newOrganisms);
    
    // Update nutrients
    this.nutrients.forEach(nutrient => {
      nutrient.update(deltaTime);
    });
    
    // Update predators
    this.predators.forEach(predator => {
      predator.update(deltaTime, this.organisms, []); // Empty safe zones array
      
      // Check predation
      this.checkPredation(predator);
    });
    
    // Remove dead entities
    const deadOrganisms = this.organisms.filter(o => !o.alive).length;
    this.statistics.deaths += deadOrganisms;
    
    this.organisms = this.organisms.filter(o => o.alive);
    this.nutrients = this.nutrients.filter(n => n.alive);
    this.predators = this.predators.filter(p => p.alive);
    
    // Spawn new nutrients more rarely
    if (Math.random() < 0.02) { // Reduced from 0.1
      this.spawnNutrient();
    }
    
    // Maintain minimum nutrients (reduced)
    while (this.nutrients.length < 10) { // Reduced from 20
      this.spawnNutrient();
    }
    
    // Maximum nutrients control
    if (this.nutrients.length > 40) { // Add upper limit
      // Remove oldest nutrients
      this.nutrients.sort((a, b) => a.age - b.age);
      this.nutrients = this.nutrients.slice(0, 40);
    }
    
    // Handle extinction
    if (this.organisms.length === 0) {
      console.log('💀 Extinction! Restarting...');
      this.initialize();
    }
    
    // Population control
    if (this.organisms.length > 50) {
      // Sort by age and remove oldest
      this.organisms.sort((a, b) => b.age - a.age);
      const toRemove = this.organisms.length - 40;
      
      for (let i = 0; i < toRemove; i++) {
        this.organisms[i].alive = false;
        this.organisms[i].deathCause = 'overpopulation';
        this.statistics.deaths++;
      }
      
      this.organisms = this.organisms.filter(o => o.alive);
    }
    
    // Update pheromone field
    this.pheromoneField.update(deltaTime);
    
    // Update statistics
    this.updateStatistics();
    
    // Continue loop
    requestAnimationFrame(() => this.update());
  }
  
  /**
   * Check organism feeding
   */
  checkFeeding(organism) {
    const feedRadius = 15;
    
    this.nutrients.forEach(nutrient => {
      if (!nutrient.alive) return;
      
      const dx = nutrient.x - organism.position.x;
      const dy = nutrient.y - organism.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < feedRadius) {
        organism.feed(nutrient.energy);
        nutrient.alive = false;
      }
    });
  }
  
  /**
   * Check predator attacks
   */
  checkPredation(predator) {
    const attackRadius = 20;
    
    this.organisms.forEach(organism => {
      if (!organism.alive) return;
      
      const dx = organism.position.x - predator.position.x;
      const dy = organism.position.y - predator.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < attackRadius) {
        // Attack with damage
        const damage = 0.5;
        organism.takeDamage(damage, 'predation');
        
        // Predator gains energy
        predator.energy += damage * 0.5;
        predator.lastMealTime = predator.age;
      }
    });
  }
  
  /**
   * Spawn nutrient at random location
   */
  spawnNutrient() {
    const x = Math.random() * this.worldSize.width;
    const y = Math.random() * this.worldSize.height;
    this.nutrients.push(new Nutrient(x, y));
  }
  
  /**
   * Update statistics
   */
  updateStatistics() {
    this.statistics.totalOrganisms = this.organisms.length;
    
    if (this.organisms.length > 0) {
      // Average generation
      const totalGen = this.organisms.reduce((sum, org) => sum + org.generation, 0);
      this.statistics.averageGeneration = totalGen / this.organisms.length;
      
      // Average energy
      const totalEnergy = this.organisms.reduce((sum, org) => sum + org.energy, 0);
      this.statistics.averageEnergy = totalEnergy / this.organisms.length;
      
      // Morphological stability (average)
      const totalStability = this.organisms.reduce(
        (sum, org) => sum + org.getMorphologicalStability(), 0
      );
      this.statistics.morphologicalStability = totalStability / this.organisms.length;
      
      // Foraging efficiency (average)
      const totalEfficiency = this.organisms.reduce(
        (sum, org) => sum + org.getForagingEfficiency(), 0
      );
      this.statistics.foragingEfficiency = totalEfficiency / this.organisms.length;
    }
    
    // Calculate cooperation/competition metrics
    this.calculateSocialMetrics();
  }
  
  /**
   * Calculate cooperation and competition indices
   */
  calculateSocialMetrics() {
    // Analyze pheromone patterns
    const pheromoneAnalysis = this.pheromoneField.analyzePatterns();
    this.statistics.pheromoneActivity = pheromoneAnalysis;
    
    // Cooperation index based on:
    // - Food trail following (shared information)
    // - Colony pheromone clustering
    // - Average distance between organisms
    
    let cooperationScore = 0;
    let competitionScore = 0;
    
    // Food trail sharing
    if (pheromoneAnalysis.totalIntensity.FOOD > 0) {
      cooperationScore += pheromoneAnalysis.coverage.FOOD * 2;
    }
    
    // Colony clustering
    if (pheromoneAnalysis.totalIntensity.COLONY > 0) {
      cooperationScore += pheromoneAnalysis.coverage.COLONY * 3;
    }
    
    // Calculate average distance between organisms
    if (this.organisms.length > 1) {
      let totalDistance = 0;
      let pairCount = 0;
      
      for (let i = 0; i < this.organisms.length; i++) {
        for (let j = i + 1; j < this.organisms.length; j++) {
          const dx = this.organisms[i].position.x - this.organisms[j].position.x;
          const dy = this.organisms[i].position.y - this.organisms[j].position.y;
          totalDistance += Math.sqrt(dx * dx + dy * dy);
          pairCount++;
        }
      }
      
      const avgDistance = totalDistance / pairCount;
      const normalizedDistance = avgDistance / Math.sqrt(this.worldSize.width * this.worldSize.height);
      
      // Closer organisms = more cooperation
      cooperationScore += (1 - normalizedDistance) * 2;
      
      // Competition for resources
      const organismsPerNutrient = this.organisms.length / Math.max(1, this.nutrients.length);
      competitionScore = Math.min(5, organismsPerNutrient);
    }
    
    // Danger signals indicate competition/conflict
    if (pheromoneAnalysis.totalIntensity.DANGER > 0) {
      competitionScore += pheromoneAnalysis.coverage.DANGER * 2;
    }
    
    // Normalize indices to 0-1 range
    this.statistics.cooperationIndex = Math.min(1, cooperationScore / 10);
    this.statistics.competitionIndex = Math.min(1, competitionScore / 5);
  }
  
  /**
   * Get current statistics
   */
  getStatistics() {
    return {
      ...this.statistics,
      time: this.time,
      fps: this.fps,
      nutrients: this.nutrients.length,
      predators: this.predators.length
    };
  }
  
  /**
   * Get all entities for rendering
   */
  getEntities() {
    return {
      organisms: this.organisms.map(o => o.getVisualizationData()),
      nutrients: this.nutrients,
      predators: this.predators,
      pheromones: this.pheromoneField.getVisualizationData()
    };
  }
}