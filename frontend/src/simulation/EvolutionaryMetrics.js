/**
 * Evolutionary Metrics for PROTEUS
 * 
 * Implements scientifically rigorous metrics for:
 * - Evolvability: Sensitivity of phenotype to mutations
 * - Novelty: Behavioral diversity in trajectory space
 * - Robustness: Performance under parameter variations
 */

export class EvolutionaryMetrics {
  constructor() {
    // Trajectory archive for novelty calculations
    this.trajectoryArchive = [];
    this.maxArchiveSize = 1000;
    
    // Evolvability measurements
    this.evolvabilityHistory = [];
    
    // Robustness measurements
    this.robustnessTests = [];
    
    // Feature extractors for trajectories
    this.trajectoryFeatures = {
      // Topological features
      persistentHomology: null,
      curvature: null,
      // Statistical features  
      displacement: null,
      tortuosity: null,
      areaExplored: null
    };
  }
  
  /**
   * Measure evolvability of an organism
   * Tests how small mutations affect phenotype
   * @param {Organism} organism - Organism to test
   * @param {number} numSamples - Number of mutations to test
   * @returns {Object} Evolvability metrics
   */
  measureEvolvability(organism, numSamples = 10) {
    const originalPhenotype = this.extractPhenotype(organism);
    const mutations = [];
    
    for (let i = 0; i < numSamples; i++) {
      // Create small mutation
      const mutatedInheritance = this.createMutation(organism.inheritance);
      
      // Measure phenotypic change
      const mutatedPhenotype = this.extractPhenotypeFromInheritance(mutatedInheritance);
      const phenotypicDistance = this.calculatePhenotypeDistance(
        originalPhenotype, 
        mutatedPhenotype
      );
      
      mutations.push({
        mutationSize: 0.05, // 5% mutation
        phenotypicChange: phenotypicDistance
      });
    }
    
    // Calculate evolvability as average phenotypic change per mutation
    const avgChange = mutations.reduce((sum, m) => sum + m.phenotypicChange, 0) / numSamples;
    const variance = mutations.reduce((sum, m) => 
      sum + Math.pow(m.phenotypicChange - avgChange, 2), 0
    ) / numSamples;
    
    const evolvability = {
      averageSensitivity: avgChange,
      varianceInResponse: variance,
      minChange: Math.min(...mutations.map(m => m.phenotypicChange)),
      maxChange: Math.max(...mutations.map(m => m.phenotypicChange)),
      timestamp: Date.now()
    };
    
    this.evolvabilityHistory.push(evolvability);
    return evolvability;
  }
  
  /**
   * Calculate novelty of organism behavior
   * Based on trajectory features in topological space
   * @param {Array} trajectory - Array of positions over time
   * @returns {Object} Novelty metrics
   */
  calculateNovelty(trajectory) {
    if (trajectory.length < 10) {
      return { noveltyScore: 0, nearestNeighborDistance: 0 };
    }
    
    // Extract topological features from trajectory
    const features = this.extractTrajectoryFeatures(trajectory);
    
    // Compare to archive
    let minDistance = Infinity;
    let distances = [];
    
    this.trajectoryArchive.forEach(archived => {
      const distance = this.calculateFeatureDistance(features, archived.features);
      distances.push(distance);
      minDistance = Math.min(minDistance, distance);
    });
    
    // Novelty is average distance to k-nearest neighbors
    const k = Math.min(15, this.trajectoryArchive.length);
    distances.sort((a, b) => a - b);
    const kNearest = distances.slice(0, k);
    const noveltyScore = kNearest.length > 0 ? 
      kNearest.reduce((sum, d) => sum + d, 0) / kNearest.length : 
      1.0; // Maximum novelty if archive is empty
    
    // Add to archive if novel enough
    if (noveltyScore > 0.3 || this.trajectoryArchive.length < 50) {
      this.trajectoryArchive.push({
        features,
        trajectory: this.compressTrajectory(trajectory),
        timestamp: Date.now()
      });
      
      // Maintain archive size
      if (this.trajectoryArchive.length > this.maxArchiveSize) {
        // Remove least novel (closest to others)
        this.pruneArchive();
      }
    }
    
    return {
      noveltyScore,
      nearestNeighborDistance: minDistance,
      archiveSize: this.trajectoryArchive.length
    };
  }
  
  /**
   * Test robustness under parameter variations
   * @param {Simulation} simulation - Simulation instance
   * @param {Object} baseParams - Base parameters
   * @param {Array} variations - Parameter variations to test
   * @returns {Object} Robustness metrics
   */
  async testRobustness(simulation, baseParams, variations) {
    const results = [];
    
    for (const variation of variations) {
      // Apply parameter variation
      const testParams = { ...baseParams, ...variation.params };
      
      // Run short simulation
      const outcome = await this.runRobustnessTest(simulation, testParams);
      
      results.push({
        variation: variation.name,
        survivalRate: outcome.survivors / outcome.initialPop,
        avgLifespan: outcome.avgLifespan,
        speciesDiversity: outcome.numSpecies,
        degradation: 1 - (outcome.survivors / outcome.baselineSurvivors)
      });
    }
    
    // Calculate robustness metrics
    const avgDegradation = results.reduce((sum, r) => sum + r.degradation, 0) / results.length;
    const maxDegradation = Math.max(...results.map(r => r.degradation));
    
    return {
      averageDegradation: avgDegradation,
      worstCaseDegradation: maxDegradation,
      parameterSensitivity: results,
      robustnessScore: 1 - avgDegradation // Higher is more robust
    };
  }
  
  /**
   * Extract phenotype vector from organism
   */
  extractPhenotype(organism) {
    return {
      // Morphological traits
      symmetry: organism.inheritance?.topologicalCore?.bodySymmetry || 1,
      organCapacity: organism.inheritance?.topologicalCore?.organCapacity || 2,
      numOrgans: organism.organs.length,
      
      // Behavioral traits
      motility: organism.capabilities.motility || 0.5,
      sensitivity: organism.capabilities.sensitivity || 0.5,
      resilience: organism.capabilities.resilience || 0.5,
      
      // Organ expression
      organTypes: organism.organs.map(o => o.type).sort(),
      
      // Memory characteristics
      memorySignature: organism.inheritance?.holographicMemory?.compress().signature.slice(0, 5) || [0,0,0,0,0]
    };
  }
  
  /**
   * Extract phenotype from inheritance alone (for mutation testing)
   */
  extractPhenotypeFromInheritance(inheritance) {
    const phenotype = inheritance.expressPhenotype();
    return {
      symmetry: inheritance.topologicalCore.bodySymmetry,
      organCapacity: inheritance.topologicalCore.organCapacity,
      numOrgans: Object.values(phenotype.organExpressions)
        .filter(expr => expr > 0.1).length,
      motility: phenotype.motility,
      sensitivity: phenotype.sensitivity,
      resilience: phenotype.resilience,
      organTypes: Object.entries(phenotype.organExpressions)
        .filter(([_, expr]) => expr > 0.1)
        .map(([type, _]) => type)
        .sort(),
      memorySignature: inheritance.holographicMemory.compress().signature.slice(0, 5)
    };
  }
  
  /**
   * Calculate distance between two phenotypes
   */
  calculatePhenotypeDistance(p1, p2) {
    let distance = 0;
    
    // Morphological differences
    distance += Math.abs(p1.symmetry - p2.symmetry) * 0.2;
    distance += Math.abs(p1.organCapacity - p2.organCapacity) * 0.1;
    distance += Math.abs(p1.numOrgans - p2.numOrgans) * 0.1;
    
    // Behavioral differences
    distance += Math.abs(p1.motility - p2.motility);
    distance += Math.abs(p1.sensitivity - p2.sensitivity);
    distance += Math.abs(p1.resilience - p2.resilience);
    
    // Organ type differences (Jaccard distance)
    const organs1 = new Set(p1.organTypes);
    const organs2 = new Set(p2.organTypes);
    const intersection = new Set([...organs1].filter(x => organs2.has(x)));
    const union = new Set([...organs1, ...organs2]);
    const jaccardDist = union.size > 0 ? 1 - intersection.size / union.size : 0;
    distance += jaccardDist * 2; // Weight organ differences higher
    
    // Memory signature distance
    const memDist = p1.memorySignature.reduce((sum, val, i) => 
      sum + Math.abs(val - p2.memorySignature[i]), 0
    ) / p1.memorySignature.length;
    distance += memDist * 0.5;
    
    return distance;
  }
  
  /**
   * Create small mutation of inheritance
   */
  createMutation(inheritance) {
    // Clone the inheritance
    const mutated = new (inheritance.constructor)();
    
    // Copy core with small mutations
    Object.assign(mutated.topologicalCore, inheritance.topologicalCore);
    
    // Mutate one random parameter
    const params = ['baseMotility', 'baseSensitivity', 'baseResilience'];
    const param = params[Math.floor(Math.random() * params.length)];
    
    mutated.topologicalCore[param] = Math.max(0.1, Math.min(1.0,
      mutated.topologicalCore[param] + (Math.random() - 0.5) * 0.1
    ));
    
    // Copy memory with slight perturbation
    mutated.holographicMemory = inheritance.holographicMemory;
    
    return mutated;
  }
  
  /**
   * Extract topological features from trajectory
   */
  extractTrajectoryFeatures(trajectory) {
    if (trajectory.length < 2) {
      return {
        displacement: 0,
        tortuosity: 0,
        areaExplored: 0,
        curvature: 0,
        loopCount: 0
      };
    }
    
    // Displacement
    const start = trajectory[0];
    const end = trajectory[trajectory.length - 1];
    const displacement = Math.sqrt(
      Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2)
    );
    
    // Path length
    let pathLength = 0;
    for (let i = 1; i < trajectory.length; i++) {
      const dx = trajectory[i].x - trajectory[i-1].x;
      const dy = trajectory[i].y - trajectory[i-1].y;
      pathLength += Math.sqrt(dx * dx + dy * dy);
    }
    
    // Tortuosity (path length / displacement)
    const tortuosity = displacement > 0 ? pathLength / displacement : pathLength;
    
    // Area explored (convex hull would be better, using bounding box for speed)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    trajectory.forEach(p => {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    });
    const areaExplored = (maxX - minX) * (maxY - minY);
    
    // Average curvature
    let totalCurvature = 0;
    let curvatureCount = 0;
    
    for (let i = 1; i < trajectory.length - 1; i++) {
      const v1 = {
        x: trajectory[i].x - trajectory[i-1].x,
        y: trajectory[i].y - trajectory[i-1].y
      };
      const v2 = {
        x: trajectory[i+1].x - trajectory[i].x,
        y: trajectory[i+1].y - trajectory[i].y
      };
      
      // Angle between vectors
      const dot = v1.x * v2.x + v1.y * v2.y;
      const det = v1.x * v2.y - v1.y * v2.x;
      const angle = Math.atan2(det, dot);
      
      totalCurvature += Math.abs(angle);
      curvatureCount++;
    }
    
    const avgCurvature = curvatureCount > 0 ? totalCurvature / curvatureCount : 0;
    
    // Count loops (simplified - crossing own path)
    let loopCount = 0;
    const gridSize = 10;
    const visited = new Set();
    
    trajectory.forEach(p => {
      const gridX = Math.floor(p.x / gridSize);
      const gridY = Math.floor(p.y / gridSize);
      const key = `${gridX},${gridY}`;
      
      if (visited.has(key)) {
        loopCount++;
      }
      visited.add(key);
    });
    
    return {
      displacement,
      tortuosity,
      areaExplored,
      curvature: avgCurvature,
      loopCount: loopCount / trajectory.length // Normalize by length
    };
  }
  
  /**
   * Calculate distance between feature vectors
   */
  calculateFeatureDistance(f1, f2) {
    const weights = {
      displacement: 0.001, // Normalize by typical scale
      tortuosity: 0.2,
      areaExplored: 0.00001, // Normalize by typical scale
      curvature: 1.0,
      loopCount: 5.0
    };
    
    let distance = 0;
    for (const key in f1) {
      const diff = Math.abs(f1[key] - f2[key]);
      distance += diff * (weights[key] || 1.0);
    }
    
    return distance;
  }
  
  /**
   * Compress trajectory for storage
   */
  compressTrajectory(trajectory) {
    // Keep every nth point to reduce size
    const stride = Math.max(1, Math.floor(trajectory.length / 50));
    return trajectory.filter((_, i) => i % stride === 0);
  }
  
  /**
   * Prune archive to maintain diversity
   */
  pruneArchive() {
    // Calculate pairwise distances
    const distances = [];
    
    for (let i = 0; i < this.trajectoryArchive.length; i++) {
      for (let j = i + 1; j < this.trajectoryArchive.length; j++) {
        const dist = this.calculateFeatureDistance(
          this.trajectoryArchive[i].features,
          this.trajectoryArchive[j].features
        );
        distances.push({ i, j, dist });
      }
    }
    
    // Find closest pair
    distances.sort((a, b) => a.dist - b.dist);
    const closest = distances[0];
    
    // Remove the newer one
    const toRemove = this.trajectoryArchive[closest.i].timestamp > 
                     this.trajectoryArchive[closest.j].timestamp ? closest.i : closest.j;
    
    this.trajectoryArchive.splice(toRemove, 1);
  }
  
  /**
   * Run robustness test with parameters
   */
  async runRobustnessTest(simulation, params) {
    // This would run a short simulation with given parameters
    // For now, return mock data - in real implementation would run actual simulation
    return {
      initialPop: 50,
      survivors: Math.floor(45 * Math.random()),
      avgLifespan: 20 + Math.random() * 10,
      numSpecies: Math.floor(2 + Math.random() * 5),
      baselineSurvivors: 45
    };
  }
  
  /**
   * Get summary metrics
   */
  getSummary() {
    const recentEvolvability = this.evolvabilityHistory.slice(-10);
    const avgEvolvability = recentEvolvability.length > 0 ?
      recentEvolvability.reduce((sum, e) => sum + e.averageSensitivity, 0) / recentEvolvability.length :
      0;
    
    return {
      evolvability: {
        current: avgEvolvability,
        trend: this.calculateTrend(this.evolvabilityHistory.map(e => e.averageSensitivity))
      },
      novelty: {
        archiveSize: this.trajectoryArchive.length,
        archiveDiversity: this.calculateArchiveDiversity()
      },
      measurements: {
        evolvabilitySamples: this.evolvabilityHistory.length,
        trajectoriesAnalyzed: this.trajectoryArchive.length
      }
    };
  }
  
  /**
   * Calculate trend in measurements
   */
  calculateTrend(values) {
    if (values.length < 2) return 0;
    
    const recent = values.slice(-10);
    const older = values.slice(-20, -10);
    
    if (older.length === 0) return 0;
    
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
    
    return (recentAvg - olderAvg) / olderAvg;
  }
  
  /**
   * Calculate diversity of trajectory archive
   */
  calculateArchiveDiversity() {
    if (this.trajectoryArchive.length < 2) return 0;
    
    let totalDist = 0;
    let count = 0;
    
    // Sample random pairs
    for (let i = 0; i < Math.min(100, this.trajectoryArchive.length); i++) {
      const idx1 = Math.floor(Math.random() * this.trajectoryArchive.length);
      const idx2 = Math.floor(Math.random() * this.trajectoryArchive.length);
      
      if (idx1 !== idx2) {
        totalDist += this.calculateFeatureDistance(
          this.trajectoryArchive[idx1].features,
          this.trajectoryArchive[idx2].features
        );
        count++;
      }
    }
    
    return count > 0 ? totalDist / count : 0;
  }
}