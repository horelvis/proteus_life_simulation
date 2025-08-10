/**
 * PROTEUS Genetic Pool - Preserves genetic advances across simulations
 */

import { EvolutionaryMetrics } from './EvolutionaryMetrics';

export class GeneticPool {
  constructor() {
    this.pool = [];
    this.maxPoolSize = 50;
    this.generationRecords = [];
    this.simulationHistory = [];
    this.evolutionaryMetrics = new EvolutionaryMetrics();
    this.currentSimulationStats = {
      startTime: null,
      endTime: null,
      peakPopulation: 0,
      totalBirths: 0,
      totalDeaths: 0,
      averageLifespan: 0,
      highestGeneration: 0,
      mutationEvents: 0,
      extinctionEvents: 0,
      survivingLineages: new Set(),
      phenotypeDistribution: {},
      organEvolution: {},
      topPerformers: [],
      predatorStats: {
        totalPredators: 0,
        averageSize: 0,
        births: 0,
        deaths: 0,
        averageAge: 0
      }
    };
  }
  
  startNewSimulation() {
    this.currentSimulationStats = {
      startTime: Date.now(),
      endTime: null,
      peakPopulation: 0,
      totalBirths: 0,
      totalDeaths: 0,
      averageLifespan: 0,
      highestGeneration: 0,
      mutationEvents: 0,
      extinctionEvents: 0,
      survivingLineages: new Set(),
      phenotypeDistribution: {},
      organEvolution: {},
      topPerformers: [],
      predatorStats: {
        totalPredators: 0,
        averageSize: 0,
        births: 0,
        deaths: 0,
        averageAge: 0
      }
    };
  }
  
  updateStats(organisms, births, deaths, predators = []) {
    const stats = this.currentSimulationStats;
    
    // Update basic stats
    stats.totalBirths += births;
    stats.totalDeaths += deaths;
    stats.peakPopulation = Math.max(stats.peakPopulation, organisms.length);
    
    if (organisms.length === 0) {
      stats.extinctionEvents++;
      return;
    }
    
    // Calculate current statistics
    const totalAge = organisms.reduce((sum, o) => sum + o.age, 0);
    stats.averageLifespan = totalAge / organisms.length;
    
    stats.highestGeneration = Math.max(...organisms.map(o => o.generation));
    
    // Track mutations
    stats.mutationEvents = organisms.filter(o => 
      o.inheritance?.topologicalCore?.hasMutation).length;
    
    // Track surviving lineages
    organisms.forEach(o => {
      if (o.generation > 0) {
        stats.survivingLineages.add(o.lineageId || o.id.substring(0, 8));
      }
    });
    
    // Phenotype distribution
    stats.phenotypeDistribution = {};
    organisms.forEach(o => {
      const phenotype = o.phenotype || 'basic';
      stats.phenotypeDistribution[phenotype] = (stats.phenotypeDistribution[phenotype] || 0) + 1;
    });
    
    // Species tracking by body symmetry
    stats.speciesDistribution = {};
    stats.speciesGroups = [];
    const speciesMap = new Map();
    
    organisms.forEach(o => {
      if (!o.inheritance?.topologicalCore) return;
      const symmetry = o.inheritance.topologicalCore.bodySymmetry;
      const speciesKey = `Symmetry-${symmetry}`;
      
      if (!speciesMap.has(speciesKey)) {
        speciesMap.set(speciesKey, {
          name: speciesKey,
          symmetry: symmetry,
          count: 0,
          avgMotility: 0,
          avgSensitivity: 0,
          avgResilience: 0,
          members: []
        });
      }
      
      const species = speciesMap.get(speciesKey);
      species.count++;
      species.avgMotility += o.inheritance.topologicalCore.baseMotility;
      species.avgSensitivity += o.inheritance.topologicalCore.baseSensitivity;
      species.avgResilience += o.inheritance.topologicalCore.baseResilience;
      species.members.push({ id: o.id, position: o.position });
    });
    
    // Calculate species averages and check grouping
    speciesMap.forEach((species, key) => {
      if (species.count > 0) {
        species.avgMotility /= species.count;
        species.avgSensitivity /= species.count;
        species.avgResilience /= species.count;
        stats.speciesDistribution[key] = species.count;
        
        // Check for grouping behavior
        if (species.count > 1) {
          const positions = species.members.map(m => m.position).filter(p => p !== null);
          
          // Calculate average distance between members
          let totalDist = 0;
          let pairs = 0;
          for (let i = 0; i < positions.length - 1; i++) {
            for (let j = i + 1; j < positions.length; j++) {
              const dx = positions[i].x - positions[j].x;
              const dy = positions[i].y - positions[j].y;
              totalDist += Math.sqrt(dx * dx + dy * dy);
              pairs++;
            }
          }
          
          species.avgGroupDistance = pairs > 0 ? totalDist / pairs : 0;
          species.isGrouping = species.avgGroupDistance < 100; // Grouped if avg distance < 100
        }
        
        stats.speciesGroups.push(species);
      }
    });
    
    // Organ evolution tracking
    stats.organEvolution = {};
    organisms.forEach(o => {
      o.organs.forEach(organ => {
        if (!stats.organEvolution[organ.type]) {
          stats.organEvolution[organ.type] = {
            count: 0,
            avgFunctionality: 0,
            maxFunctionality: 0
          };
        }
        const organStats = stats.organEvolution[organ.type];
        organStats.count++;
        organStats.avgFunctionality += organ.functionality;
        organStats.maxFunctionality = Math.max(organStats.maxFunctionality, organ.functionality);
      });
    });
    
    // Normalize organ averages
    Object.keys(stats.organEvolution).forEach(organType => {
      const organStats = stats.organEvolution[organType];
      organStats.avgFunctionality /= organStats.count;
    });
    
    // Track top performers
    const topOrganisms = organisms
      .filter(o => o.generation > 2 && o.energy > 0.5)
      .sort((a, b) => {
        // Score based on generation, age, and energy
        const scoreA = a.generation * 10 + a.age + a.energy * 5;
        const scoreB = b.generation * 10 + b.age + b.energy * 5;
        return scoreB - scoreA;
      })
      .slice(0, 10);
    
    stats.topPerformers = topOrganisms.map(o => ({
      id: o.id.substring(0, 8),
      generation: o.generation,
      age: o.age,
      energy: o.energy,
      phenotype: o.phenotype || 'basic',
      organs: o.organs.length,
      capabilities: o.capabilities
    }));
    
    // Calculate evolvability and novelty for top performers
    if (topOrganisms.length > 0) {
      // Measure evolvability of best organism
      const best = topOrganisms[0];
      if (best.inheritance) {
        stats.evolvability = this.evolutionaryMetrics.measureEvolvability(best, 5);
      }
      
      // Calculate novelty for all organisms with sufficient trajectory
      let totalNovelty = 0;
      let noveltyCount = 0;
      
      organisms.forEach(o => {
        if (o.trajectory && o.trajectory.length > 20) {
          const novelty = this.evolutionaryMetrics.calculateNovelty(o.trajectory);
          totalNovelty += novelty.noveltyScore;
          noveltyCount++;
        }
      });
      
      stats.averageNovelty = noveltyCount > 0 ? totalNovelty / noveltyCount : 0;
    }
    
    // Update predator statistics
    if (predators && predators.length > 0) {
      stats.predatorStats.totalPredators = predators.length;
      const totalAge = predators.reduce((sum, p) => sum + p.age, 0);
      stats.predatorStats.averageAge = totalAge / predators.length;
      const totalSize = predators.reduce((sum, p) => sum + p.size, 0);
      stats.predatorStats.averageSize = totalSize / predators.length;
    }
  }
  
  preserveEliteOrganisms(organisms) {
    if (organisms.length === 0) return;
    
    // Select elite organisms based on fitness criteria
    const elites = organisms
      .filter(o => o.alive && o.generation > 0)
      .sort((a, b) => {
        // Fitness score: generation weight + age + energy + organ development
        const fitnessA = a.generation * 2 + a.age * 0.1 + a.energy + a.organs.length * 0.5;
        const fitnessB = b.generation * 2 + b.age * 0.1 + b.energy + b.organs.length * 0.5;
        return fitnessB - fitnessA;
      })
      .slice(0, 20); // Keep top 20
    
    // Store their genetic information
    elites.forEach(organism => {
      const geneticRecord = {
        id: organism.id,
        generation: organism.generation,
        lineageId: organism.lineageId || organism.id.substring(0, 8),
        inheritance: organism.inheritance.compress(),
        phenotype: organism.phenotype,
        organs: organism.organs.map(o => ({
          type: o.type,
          functionality: o.functionality
        })),
        capabilities: { ...organism.capabilities },
        fitness: organism.generation * 2 + organism.age * 0.1 + organism.energy,
        preservedAt: Date.now()
      };
      
      this.addToPool(geneticRecord);
    });
  }
  
  addToPool(geneticRecord) {
    // Check if we already have a similar organism
    const existingIndex = this.pool.findIndex(record => 
      record.lineageId === geneticRecord.lineageId &&
      record.generation <= geneticRecord.generation
    );
    
    if (existingIndex >= 0) {
      // Replace with newer generation
      this.pool[existingIndex] = geneticRecord;
    } else {
      this.pool.push(geneticRecord);
    }
    
    // Maintain pool size limit
    if (this.pool.length > this.maxPoolSize) {
      // Remove least fit organisms
      this.pool.sort((a, b) => b.fitness - a.fitness);
      this.pool = this.pool.slice(0, this.maxPoolSize);
    }
  }
  
  getEliteGenetics(count = 10) {
    // Return top genetic templates for seeding new simulation
    return this.pool
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, count)
      .map(record => ({
        inheritance: record.inheritance,
        phenotype: record.phenotype,
        organs: record.organs,
        capabilities: record.capabilities,
        generation: record.generation
      }));
  }
  
  generateSimulationReport() {
    const stats = this.currentSimulationStats;
    stats.endTime = Date.now();
    const duration = (stats.endTime - stats.startTime) / 1000; // seconds
    
    const report = {
      summary: {
        duration: `${Math.floor(duration / 60)}m ${Math.floor(duration % 60)}s`,
        peakPopulation: stats.peakPopulation,
        totalBirths: stats.totalBirths,
        totalDeaths: stats.totalDeaths,
        survivalRate: stats.totalBirths > 0 ? 
          ((stats.totalBirths - stats.totalDeaths) / stats.totalBirths * 100).toFixed(1) + '%' : '0%',
        averageLifespan: stats.averageLifespan.toFixed(1),
        highestGeneration: stats.highestGeneration,
        extinctionEvents: stats.extinctionEvents
      },
      evolution: {
        mutationEvents: stats.mutationEvents,
        survivingLineages: stats.survivingLineages.size,
        phenotypes: stats.phenotypeDistribution,
        organDevelopment: stats.organEvolution,
        evolvability: stats.evolvability || null,
        averageNovelty: stats.averageNovelty || 0
      },
      metrics: this.evolutionaryMetrics.getSummary(),
      topPerformers: stats.topPerformers,
      geneticPoolSize: this.pool.length,
      preservedElites: this.pool.slice(0, 5).map(r => ({
        generation: r.generation,
        fitness: r.fitness.toFixed(2),
        phenotype: r.phenotype,
        organs: r.organs.length
      })),
      predatorStats: stats.predatorStats
    };
    
    // Store in history
    this.simulationHistory.push({
      timestamp: Date.now(),
      report: report
    });
    
    return report;
  }
  
  getFormattedReport() {
    const report = this.generateSimulationReport();
    const stats = this.currentSimulationStats;
    
    let formatted = `
═══════════════════════════════════════════
        PROTEUS SIMULATION REPORT         
═══════════════════════════════════════════

📊 SUMMARY
──────────────────────────────────────────
Duration: ${report.summary.duration}
Peak Population: ${report.summary.peakPopulation}
Total Births: ${report.summary.totalBirths}
Total Deaths: ${report.summary.totalDeaths}
Survival Rate: ${report.summary.survivalRate}
Average Lifespan: ${report.summary.averageLifespan}
Highest Generation: ${report.summary.highestGeneration}
Extinction Events: ${report.summary.extinctionEvents}

🧬 EVOLUTIONARY PROGRESS
──────────────────────────────────────────
Mutation Events: ${report.evolution.mutationEvents}
Surviving Lineages: ${report.evolution.survivingLineages}
Evolvability: ${report.evolution.evolvability ? report.evolution.evolvability.averageSensitivity.toFixed(3) : 'N/A'}
Average Novelty: ${report.evolution.averageNovelty.toFixed(3)}

Phenotype Distribution:
${Object.entries(report.evolution.phenotypes)
  .map(([type, count]) => `  ${type}: ${count}`)
  .join('\n')}

Organ Evolution:
${Object.entries(report.evolution.organDevelopment)
  .map(([organ, stats]) => 
    `  ${organ}: ${stats.count} organisms (avg: ${stats.avgFunctionality.toFixed(2)}, max: ${stats.maxFunctionality.toFixed(2)})`
  )
  .join('\n')}

🌊 SPECIES EMERGENCE & GROUPING
──────────────────────────────────────────
Distinct Species: ${Object.keys(stats.speciesDistribution || {}).length}

Species Distribution:
${Object.entries(stats.speciesDistribution || {})
  .map(([species, count]) => `  ${species}: ${count} organisms`)
  .join('\n') || '  No distinct species detected'}

Grouping Behavior:
${(stats.speciesGroups || [])
  .filter(species => species.count > 1)
  .map(species => {
    const groupStatus = species.isGrouping ? '✓ Grouping' : '✗ Dispersed';
    return `  ${species.name}: ${groupStatus} (${species.count} members, avg distance: ${species.avgGroupDistance ? species.avgGroupDistance.toFixed(1) : 'N/A'})`;
  })
  .join('\n') || '  No grouping behavior observed'}

Species Characteristics:
${(stats.speciesGroups || [])
  .filter(species => species.count > 2)
  .map(species => 
    `  ${species.name}:\n` +
    `    Motility: ${species.avgMotility.toFixed(2)}\n` +
    `    Sensitivity: ${species.avgSensitivity.toFixed(2)}\n` +
    `    Resilience: ${species.avgResilience.toFixed(2)}`
  )
  .join('\n') || '  Insufficient data for species characteristics'}

🦈 PREDATOR ECOSYSTEM
──────────────────────────────────────────
Total Predators: ${report.predatorStats.totalPredators}
Average Size: ${report.predatorStats.averageSize.toFixed(1)}
Average Age: ${report.predatorStats.averageAge.toFixed(1)}

🏆 TOP PERFORMERS
──────────────────────────────────────────
${report.topPerformers.slice(0, 5)
  .map((p, i) => 
    `${i + 1}. Gen ${p.generation} | Age: ${p.age.toFixed(1)} | Energy: ${p.energy.toFixed(2)} | ${p.phenotype} | ${p.organs} organs`
  )
  .join('\n')}

📊 EVOLUTIONARY METRICS
──────────────────────────────────────────
Evolvability Trend: ${report.metrics ? (report.metrics.evolvability.trend * 100).toFixed(1) : '0.0'}%
Novelty Archive Size: ${report.metrics ? report.metrics.novelty.archiveSize : 0}
Archive Diversity: ${report.metrics ? report.metrics.novelty.archiveDiversity.toFixed(2) : '0.00'}
Trajectories Analyzed: ${report.metrics ? report.metrics.measurements.trajectoriesAnalyzed : 0}

💾 GENETIC PRESERVATION
──────────────────────────────────────────
Genetic Pool Size: ${report.geneticPoolSize}
Elite Organisms Preserved:
${report.preservedElites
  .map((e, i) => 
    `  ${i + 1}. Gen ${e.generation} | Fitness: ${e.fitness} | ${e.phenotype} | ${e.organs} organs`
  )
  .join('\n')}

═══════════════════════════════════════════
`;
    
    return formatted;
  }
  
  exportGeneticData() {
    return {
      pool: this.pool,
      history: this.simulationHistory,
      timestamp: Date.now()
    };
  }
  
  importGeneticData(data) {
    if (data.pool) {
      this.pool = data.pool;
    }
    if (data.history) {
      this.simulationHistory = data.history;
    }
  }
}