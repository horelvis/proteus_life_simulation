/**
 * PROTEUS Simulation - Frontend Implementation
 */

import { TopologyEngine } from './TopologyEngine';
import { Organism } from './Organism';
import { Nutrient } from './Nutrient';
import { Predator } from './Predator';
import { EnvironmentalField } from './EnvironmentalField';
import { GeneticPool } from './GeneticPool';
import { LineageTracker } from './LineageTracker';
import { ProteusInheritance } from './ProteusInheritance';

export class Simulation {
  constructor(worldSize) {
    this.worldSize = worldSize;
    this.topologyEngine = new TopologyEngine(worldSize);
    this.environmentalField = new EnvironmentalField(worldSize);
    this.geneticPool = new GeneticPool();
    this.lineageTracker = new LineageTracker();
    
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    this.safeZones = this.generateSafeZones();
    
    this.time = 0;
    this.generation = 0;
    this.running = false;
    this.speedMultiplier = 1.0;
    this.lastDeathCauses = [];
    
    this.statistics = {
      total_organisms: 0,
      totalOrganisms: 0, // Backwards compatibility
      births: 0,
      deaths: 0,
      averageAge: 0,
      averageEnergy: 0,
      average_generation: 0,
      average_mutation_rate: 0,
      highest_generation: 0,
      mutations: 0
    };
    
    this.lastUpdate = Date.now();
    this.birthsSinceLastUpdate = 0;
    this.deathsSinceLastUpdate = 0;
    this.updateCount = 0;
    this.lastWatchdogCheck = Date.now();
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
  
  findSpawnPositionAwayFromSafeZones() {
    let x, y;
    let attempts = 0;
    const maxAttempts = 10;
    const safeZoneBuffer = 50;
    
    do {
      x = Math.random() * this.worldSize.width;
      y = Math.random() * this.worldSize.height;
      attempts++;
      
      // Check if position is far enough from all safe zones
      const tooCloseToSafeZone = this.safeZones.some(zone => {
        const dx = x - zone.x;
        const dy = y - zone.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < zone.radius + safeZoneBuffer;
      });
      
      if (!tooCloseToSafeZone || attempts >= maxAttempts) {
        break;
      }
    } while (attempts < maxAttempts);
    
    return { x, y };
  }
  
  initialize(config = {}) {
    const {
      initialOrganisms = 30,   // Reduced from 50
      initialNutrients = 80,   // Reduced from 120
      initialPredators = 2,    // Keep low number of predators
      useGeneticPool = true
    } = config;
    
    console.log('🚀 Initializing simulation with:', { initialOrganisms, initialNutrients, initialPredators });
    
    // Clear existing organisms first
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    
    // Start new simulation tracking
    this.geneticPool.startNewSimulation();
    
    // Get elite genetics from pool
    const eliteGenetics = this.geneticPool.getEliteGenetics();
    const eliteCount = Math.min(eliteGenetics.length, Math.floor(initialOrganisms * 0.3));
    
    // Create organisms from genetic pool
    let organismsCreated = 0;
    
    if (useGeneticPool && eliteCount > 0) {
      // Spawn elite organisms from genetic pool
      for (let i = 0; i < eliteCount; i++) {
        const elite = eliteGenetics[i % eliteGenetics.length];
        const x = Math.random() * this.worldSize.width;
        const y = Math.random() * this.worldSize.height;
        
        const organism = new Organism(x, y, this.topologyEngine);
        
        // Apply preserved genetics
        if (elite.inheritance) {
          // Replace the organism's inheritance with preserved one
          organism.inheritance = this.reconstructInheritance(elite.inheritance);
          organism.generation = elite.generation;
          organism.applyPhenotype();
        }
        
        this.organisms.push(organism);
        organismsCreated++;
      }
    }
    
    // Create remaining organisms randomly
    for (let i = organismsCreated; i < initialOrganisms; i++) {
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
    
    console.log(`✅ Initialized with ${this.organisms.length} organisms, ${this.nutrients.length} nutrients, ${this.predators.length} predators`);
    
    this.updateStatistics();
    
    console.log('📊 Initial statistics:', this.statistics);
    
    // Ensure we have organisms
    if (this.organisms.length === 0) {
      console.error('❌ No organisms created during initialization!');
      // Force create at least one organism
      const emergencyOrganism = new Organism(
        this.worldSize.width / 2, 
        this.worldSize.height / 2, 
        this.topologyEngine
      );
      this.organisms.push(emergencyOrganism);
      console.log('🆘 Created emergency organism');
    }
  }
  
  start() {
    this.running = true;
    this.lastUpdate = Date.now();
    this.update();
  }
  
  stop() {
    this.running = false;
    
    // Preserve elite organisms before stopping
    this.geneticPool.preserveEliteOrganisms(this.organisms);
  }
  
  setSpeed(multiplier) {
    this.speedMultiplier = Math.max(0.1, Math.min(2.0, multiplier));
  }
  
  update() {
    if (!this.running) return;
    
    const now = Date.now();
    const deltaTime = (now - this.lastUpdate) / 1000;
    
    // Prevent huge time jumps that could break physics
    if (deltaTime > 1.0) {
      console.warn('Large deltaTime detected:', deltaTime, 'seconds. Capping to 0.1');
      this.lastUpdate = now - 100; // Reset to 100ms ago
      return requestAnimationFrame(() => this.update());
    }
    
    // Prevent zero or very small deltaTime which could cause blocking
    if (deltaTime < 0.001) {
      this.lastUpdate = now;
      return requestAnimationFrame(() => this.update());
    }
    
    this.lastUpdate = now;
    
    // Cap deltaTime to prevent huge jumps and apply speed multiplier
    const dt = Math.min(deltaTime, 0.05) * this.speedMultiplier;
    
    this.time += dt;
    
    // Update topology field
    this.topologyEngine.updateField(dt);
    
    // Update organisms with performance optimizations
    const newOrganisms = [];
    
    // Pre-compute spatial index for performance
    const spatialGrid = this.createSpatialGrid(this.organisms);
    
    this.organisms.forEach(organism => {
      // Skip expensive environmental memory for now
      organism.environmentalMemory = { inSafeZone: false };
      
      // Get only nearby organisms for perception (100 unit radius)
      const nearbyOrganisms = this.getNearbyOrganisms(organism.position, 100, spatialGrid);
      
      // Update organism with only nearby entities
      organism.update(dt, this.nutrients, this.predators, nearbyOrganisms);
      
      // Skip trace deposits for performance
      
      // Check feeding
      if (organism.alive) {
        this.checkFeeding(organism);
        
        // Check reproduction (limit population for performance)
        if (this.organisms.length < 60 && organism.canReproduce()) { // Reduced from 150 to 60
          const offspring = organism.reproduce();
          newOrganisms.push(offspring);
          this.statistics.births++;
          this.birthsSinceLastUpdate++;
          
          // Skip lineage tracking for performance
        }
      }
    });
    
    // Add new organisms after loop
    this.organisms.push(...newOrganisms);
    
    // Update nutrients
    this.nutrients.forEach(nutrient => {
      nutrient.update(dt);
    });
    
    // Update predators
    const newPredators = [];
    this.predators.forEach(predator => {
      predator.update(dt, this.organisms, this.safeZones);
      
      // Check predator reproduction
      if (predator.alive && predator.canReproduce()) {
        const offspring = predator.reproduce();
        newPredators.push(offspring);
        console.log(`🦈 Predator reproduced! Energy: ${predator.energy.toFixed(2)}, Age: ${predator.age.toFixed(1)}, Last meal: ${(predator.age - predator.lastMealTime).toFixed(1)} ago`);
        
        // Update predator birth stats
        if (this.currentSimulationStats?.predatorStats) {
          this.currentSimulationStats.predatorStats.births++;
        }
      }
    });
    
    // Add new predators
    this.predators.push(...newPredators);
    
    // Handle dead organisms and their memory traces
    const deadOrganisms = this.organisms.filter(o => !o.alive);
    this.statistics.deaths += deadOrganisms.length;
    this.deathsSinceLastUpdate += deadOrganisms.length;
    
    // Track death causes
    if (deadOrganisms.length > 0) {
      this.lastDeathCauses = deadOrganisms.map(o => ({
        cause: o.deathCause,
        age: o.age,
        energy: o.energy,
        time: this.time
      }));
    }
    
    // Process death traces
    deadOrganisms.forEach(organism => {
      // Track lineage death
      this.lineageTracker.recordDeath(organism);
      
      // Death leaves strong environmental traces
      const deathTrace = {
        position: organism.position,
        pheromone: [0, 0, 0, 1, 0], // Death pheromone
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
    
    // Aggressive population control to prevent PC overheating
    if (this.organisms.length > 80) {
      // Sort by age and remove oldest organisms
      this.organisms.sort((a, b) => b.age - a.age);
      const toRemove = this.organisms.length - 60;
      
      // Kill oldest organisms
      for (let i = 0; i < toRemove; i++) {
        const oldOrganism = this.organisms[i];
        oldOrganism.alive = false;
        oldOrganism.deathCause = 'overpopulation';
        this.statistics.deaths++;
        this.deathsSinceLastUpdate++;
      }
      
      // Remove the dead ones
      this.organisms = this.organisms.filter(o => o.alive);
    }
    
    // Spawn new nutrients periodically - increased rate
    if (Math.random() < 0.1) { // Doubled spawn rate
      this.spawnNutrient();
    }
    
    // Ensure minimum nutrients
    if (this.nutrients.length < 30) {
      for (let i = this.nutrients.length; i < 30; i++) {
        this.spawnNutrient();
      }
    }
    
    // Check for total extinction
    if (this.organisms.length === 0) {
      console.log('🚨 Total extinction detected! Initiating automatic restart...');
      console.log('Last death causes:', this.lastDeathCauses);
      console.log('Time since start:', this.time);
      console.log('Nutrients available:', this.nutrients.length);
      console.log('Average organism lifespan:', (this.statistics.deaths > 0 ? this.time / this.statistics.deaths : 0).toFixed(2));
      
      // Record extinction event
      if (!this.statistics.extinctionEvents) this.statistics.extinctionEvents = 0;
      this.statistics.extinctionEvents++;
      
      // Generate final report before restart
      const extinctionReport = this.geneticPool.getFormattedReport();
      console.log('📊 Extinction Report:', extinctionReport);
      
      // Clear statistics to reflect extinction
      this.statistics.totalOrganisms = 0;
      this.statistics.averageAge = 0;
      this.statistics.averageEnergy = 0;
      
      // Automatic restart with preserved genetics only if still running
      if (this.running) {
        setTimeout(() => {
          if (this.running) { // Double check in case stopped during timeout
            this.autoRestart();
          }
        }, 1000); // Small delay to ensure UI updates
      }
      
      return; // Skip the rest of the update
    }
    
    // Let natural selection work - no artificial population maintenance
    // If population drops too low, extinction is a valid outcome
    
    // Predator population scales with prey - natural balance
    const minPredators = Math.max(2, Math.floor(this.organisms.length / 20)); 
    const maxPredators = Math.max(4, Math.floor(this.organisms.length / 10));
    
    if (this.predators.length < minPredators && this.organisms.length > 5) {
      // Spawn new predators away from safe zones
      for (let i = this.predators.length; i < minPredators; i++) {
        const position = this.findSpawnPositionAwayFromSafeZones();
        this.predators.push(new Predator(position.x, position.y, this.worldSize));
      }
    }
    
    // Limit maximum predators
    if (this.predators.length > maxPredators) {
      // Remove weakest predators
      this.predators.sort((a, b) => a.energy - b.energy);
      this.predators = this.predators.slice(-maxPredators);
    }
    
    this.updateStatistics();
    
    // Update environmental field
    this.environmentalField.update(dt);
    
    // Watchdog check - ensure simulation is still running
    this.updateCount++;
    const watchdogNow = Date.now();
    
    if (watchdogNow - this.lastWatchdogCheck > 5000) { // Check every 5 seconds
      const updatesPerSecond = this.updateCount / 5;
      const avgEnergy = this.organisms.length > 0 ? 
        this.organisms.reduce((sum, o) => sum + o.energy, 0) / this.organisms.length : 0;
      
      console.log(`📊 Simulation health: ${updatesPerSecond.toFixed(1)} updates/sec | 🦠 ${this.organisms.length} organisms | ⚡ Avg energy: ${avgEnergy.toFixed(2)} | 🥬 ${this.nutrients.length} nutrients | 🦈 ${this.predators.length} predators`);
      
      if (updatesPerSecond < 10 && this.running) {
        console.warn('⚠️ Simulation running slow:', updatesPerSecond.toFixed(1), 'updates/sec');
        // Don't reset lastUpdate - this causes timing jumps
      }
      
      this.updateCount = 0;
      this.lastWatchdogCheck = watchdogNow;
    }
    
    // Continue animation loop only if still running
    if (this.running) {
      requestAnimationFrame(() => this.update());
    }
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
  
  createSpatialGrid(organisms) {
    const gridSize = 50; // Size of each grid cell
    const grid = {};
    
    organisms.forEach(org => {
      if (!org.alive) return;
      
      const gridX = Math.floor(org.position.x / gridSize);
      const gridY = Math.floor(org.position.y / gridSize);
      const key = `${gridX},${gridY}`;
      
      if (!grid[key]) {
        grid[key] = [];
      }
      grid[key].push(org);
    });
    
    return { grid, gridSize };
  }
  
  getNearbyOrganisms(position, radius, spatialGrid) {
    const { grid, gridSize } = spatialGrid;
    const nearby = [];
    
    // Calculate grid cells to check
    const gridRadius = Math.ceil(radius / gridSize);
    const centerGridX = Math.floor(position.x / gridSize);
    const centerGridY = Math.floor(position.y / gridSize);
    
    for (let dx = -gridRadius; dx <= gridRadius; dx++) {
      for (let dy = -gridRadius; dy <= gridRadius; dy++) {
        const gridX = centerGridX + dx;
        const gridY = centerGridY + dy;
        const key = `${gridX},${gridY}`;
        
        if (grid[key]) {
          grid[key].forEach(org => {
            const dist = Math.sqrt(
              Math.pow(org.position.x - position.x, 2) +
              Math.pow(org.position.y - position.y, 2)
            );
            
            if (dist <= radius) {
              nearby.push(org);
            }
          });
        }
      }
    }
    
    return nearby;
  }
  
  updateStatistics() {
    this.statistics.total_organisms = this.organisms.length;
    this.statistics.totalOrganisms = this.organisms.length; // Backwards compatibility
    
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
    
    // Update genetic pool statistics
    this.geneticPool.updateStats(
      this.organisms,
      this.birthsSinceLastUpdate,
      this.deathsSinceLastUpdate,
      this.predators
    );
    
    // Reset counters
    this.birthsSinceLastUpdate = 0;
    this.deathsSinceLastUpdate = 0;
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
        maxAge: o.maxAge,
        generation: o.generation,
        organs: o.organs,
        isPanicked: o.isPanicked,
        inheritance: o.inheritance,
        capabilities: o.capabilities,
        phenotype: o.phenotype
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
        tentacles: p.tentacles,
        opacity: p.opacity || 1.0
      })),
      safeZones: this.safeZones,
      statistics: this.statistics,
      time: this.time,
      environmentalField: this.environmentalField.getVisualizationData(),
      memoryAnchors: this.environmentalField.memoryAnchors
    };
  }
  
  getSimulationReport() {
    return this.geneticPool.getFormattedReport();
  }
  
  getEvolutionaryProgress() {
    return this.lineageTracker.getEvolutionaryProgress();
  }
  
  getMostSuccessfulLineages(limit = 10) {
    return this.lineageTracker.getMostSuccessfulLineages(limit);
  }
  
  exportGeneticPool() {
    return this.geneticPool.exportGeneticData();
  }
  
  importGeneticPool(data) {
    this.geneticPool.importGeneticData(data);
  }
  
  reconstructInheritance(compressedData) {
    // Create a new ProteusInheritance from compressed data
    const inheritance = new ProteusInheritance();
    
    // Restore core parameters
    if (compressedData.core) {
      Object.assign(inheritance.topologicalCore, compressedData.core);
    }
    
    // Restore memory if available
    if (compressedData.memory) {
      // The memory is already compressed, we'll use it as is
      // The holographic memory will handle it internally
    }
    
    // Restore environmental traces
    if (compressedData.traces) {
      inheritance.environmentalTraces.generation = compressedData.traces.generation;
      inheritance.environmentalTraces.pheromoneSignature = compressedData.traces.pheromone;
    }
    
    // Update phenotype based on restored data
    inheritance.phenotype = inheritance.expressPhenotype();
    
    return inheritance;
  }
  
  autoRestart() {
    console.log('🔄 Auto-restarting simulation with preserved genetics...');
    
    // Clear current entities but preserve genetic pool
    this.organisms = [];
    this.nutrients = [];
    this.predators = [];
    
    // Clear environmental field
    this.environmentalField = new EnvironmentalField(this.worldSize);
    
    // Re-initialize with more organisms to ensure survival
    this.initialize({
      initialOrganisms: 40,  // More than default
      initialNutrients: 120, // Even more nutrients
      initialPredators: 2,   // Fewer predators initially
      useGeneticPool: true   // Use preserved genetics
    });
    
    // Log restart info
    console.log(`✅ Simulation restarted with ${this.organisms.length} organisms from genetic pool`);
    console.log(`🧬 Genetic pool size: ${this.geneticPool.pool.length}`);
    
    // Continue running if it was running before
    if (this.running) {
      this.lastUpdate = Date.now();
      // Animation loop is already running from the original start()
      // No need to start a new one
    }
  }
}