/**
 * Lineage Tracker - Tracks family relationships and evolutionary branches
 */

export class LineageTracker {
  constructor() {
    this.lineages = new Map(); // organism.id -> lineage data
    this.familyTrees = new Map(); // root ancestor id -> tree structure
    this.generationStats = new Map(); // generation -> statistics
  }
  
  recordBirth(parent, child) {
    // Create lineage entry for child
    const parentLineage = this.lineages.get(parent.id);
    
    const childLineage = {
      id: child.id,
      parentId: parent.id,
      generation: child.generation,
      birthTime: Date.now(),
      traits: {
        motility: child.inheritance?.topologicalCore?.baseMotility || 0,
        sensitivity: child.inheritance?.topologicalCore?.baseSensitivity || 0,
        resilience: child.inheritance?.topologicalCore?.baseResilience || 0
      },
      organs: child.organs?.map(o => o.type) || [],
      mutations: [],
      descendants: []
    };
    
    // Track mutations
    if (parentLineage) {
      // Compare traits to detect mutations
      const traitDiff = {
        motility: Math.abs(childLineage.traits.motility - parentLineage.traits.motility),
        sensitivity: Math.abs(childLineage.traits.sensitivity - parentLineage.traits.sensitivity),
        resilience: Math.abs(childLineage.traits.resilience - parentLineage.traits.resilience)
      };
      
      if (traitDiff.motility > 0.05) childLineage.mutations.push('motility');
      if (traitDiff.sensitivity > 0.05) childLineage.mutations.push('sensitivity');
      if (traitDiff.resilience > 0.05) childLineage.mutations.push('resilience');
      
      // Check for new organs
      const parentOrgans = new Set(parentLineage.organs);
      childLineage.organs.forEach(organ => {
        if (!parentOrgans.has(organ)) {
          childLineage.mutations.push(`new_organ_${organ}`);
        }
      });
      
      // Add to parent's descendants
      parentLineage.descendants.push(child.id);
      
      // Track in same family tree
      const rootId = this.findRootAncestor(parent.id);
      this.addToFamilyTree(rootId, child.id, parent.id);
    } else {
      // New lineage - create new family tree
      this.familyTrees.set(child.id, {
        rootId: child.id,
        nodes: new Map([[child.id, { parentId: null, children: [] }]])
      });
    }
    
    this.lineages.set(child.id, childLineage);
    
    // Update generation stats
    this.updateGenerationStats(child.generation, childLineage);
  }
  
  recordDeath(organism) {
    const lineage = this.lineages.get(organism.id);
    if (lineage) {
      lineage.deathTime = Date.now();
      lineage.lifespan = organism.age;
      lineage.deathCause = organism.deathCause;
      lineage.finalFitness = this.calculateFitness(organism);
    }
  }
  
  calculateFitness(organism) {
    // Comprehensive fitness calculation
    let fitness = 0;
    
    // Survival fitness
    fitness += organism.age * 2;
    fitness += organism.energy * 10;
    
    // Reproductive fitness
    const lineage = this.lineages.get(organism.id);
    if (lineage) {
      fitness += lineage.descendants.length * 20;
    }
    
    // Adaptive fitness (organs)
    if (organism.organs) {
      organism.organs.forEach(organ => {
        if (organ.functionality > 0.5) {
          fitness += 5;
        }
      });
    }
    
    return fitness;
  }
  
  findRootAncestor(organismId) {
    let currentId = organismId;
    let lineage = this.lineages.get(currentId);
    
    while (lineage && lineage.parentId) {
      currentId = lineage.parentId;
      lineage = this.lineages.get(currentId);
    }
    
    return currentId;
  }
  
  addToFamilyTree(rootId, childId, parentId) {
    const tree = this.familyTrees.get(rootId);
    if (!tree) return;
    
    // Add child node
    tree.nodes.set(childId, { parentId, children: [] });
    
    // Update parent's children
    const parentNode = tree.nodes.get(parentId);
    if (parentNode) {
      parentNode.children.push(childId);
    }
  }
  
  updateGenerationStats(generation, lineage) {
    if (!this.generationStats.has(generation)) {
      this.generationStats.set(generation, {
        count: 0,
        avgMotility: 0,
        avgSensitivity: 0,
        avgResilience: 0,
        organTypes: new Set(),
        mutations: []
      });
    }
    
    const stats = this.generationStats.get(generation);
    stats.count++;
    
    // Update averages (running average)
    const n = stats.count;
    stats.avgMotility = ((n - 1) * stats.avgMotility + lineage.traits.motility) / n;
    stats.avgSensitivity = ((n - 1) * stats.avgSensitivity + lineage.traits.sensitivity) / n;
    stats.avgResilience = ((n - 1) * stats.avgResilience + lineage.traits.resilience) / n;
    
    // Track organ types
    lineage.organs.forEach(organ => stats.organTypes.add(organ));
    
    // Track mutations
    if (lineage.mutations.length > 0) {
      stats.mutations.push(...lineage.mutations);
    }
  }
  
  getEvolutionaryProgress() {
    const generations = Array.from(this.generationStats.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([gen, stats]) => ({
        generation: gen,
        population: stats.count,
        traits: {
          motility: stats.avgMotility,
          sensitivity: stats.avgSensitivity,
          resilience: stats.avgResilience
        },
        organDiversity: stats.organTypes.size,
        mutationCount: stats.mutations.length
      }));
    
    return {
      generations,
      totalLineages: this.lineages.size,
      activeFamilies: this.familyTrees.size,
      overallProgress: this.calculateOverallProgress(generations)
    };
  }
  
  calculateOverallProgress(generations) {
    if (generations.length < 2) return 0;
    
    const first = generations[0];
    const last = generations[generations.length - 1];
    
    // Calculate improvement in each dimension
    const traitImprovement = 
      (last.traits.motility - first.traits.motility) +
      (last.traits.sensitivity - first.traits.sensitivity) +
      (last.traits.resilience - first.traits.resilience);
    
    const diversityImprovement = last.organDiversity - first.organDiversity;
    
    // Combined progress score (0-100)
    const progress = Math.min(100, 
      (traitImprovement * 50) + 
      (diversityImprovement * 10) +
      (generations.length * 2)
    );
    
    return Math.max(0, progress);
  }
  
  getMostSuccessfulLineages(limit = 10) {
    const lineageScores = Array.from(this.lineages.entries())
      .map(([id, lineage]) => ({
        id,
        lineage,
        score: this.calculateLineageScore(lineage)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    
    return lineageScores;
  }
  
  calculateLineageScore(lineage) {
    let score = 0;
    
    // Longevity bonus
    if (lineage.lifespan) {
      score += lineage.lifespan * 2;
    }
    
    // Reproductive success
    score += lineage.descendants.length * 20;
    
    // Trait excellence
    score += (lineage.traits.motility + lineage.traits.sensitivity + lineage.traits.resilience) * 10;
    
    // Innovation bonus (mutations that survived)
    score += lineage.mutations.length * 5;
    
    // Organ diversity
    score += lineage.organs.length * 3;
    
    return score;
  }
  
  getLineageTree(organismId) {
    const rootId = this.findRootAncestor(organismId);
    const tree = this.familyTrees.get(rootId);
    
    if (!tree) return null;
    
    // Build visual tree structure
    const buildTreeNode = (nodeId) => {
      const lineage = this.lineages.get(nodeId);
      const node = tree.nodes.get(nodeId);
      
      if (!lineage || !node) return null;
      
      return {
        id: nodeId,
        generation: lineage.generation,
        traits: lineage.traits,
        organs: lineage.organs,
        mutations: lineage.mutations,
        alive: !lineage.deathTime,
        fitness: lineage.finalFitness || 0,
        children: node.children.map(childId => buildTreeNode(childId)).filter(n => n)
      };
    };
    
    return buildTreeNode(rootId);
  }
  
  reset() {
    this.lineages.clear();
    this.familyTrees.clear();
    this.generationStats.clear();
  }
}