/**
 * PROTEUS Predator - Frontend Implementation
 */

export class Predator {
  constructor(x, y, worldSize) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.position = { x, y };
    this.velocity = { x: 0, y: 0 };
    this.worldSize = worldSize;
    
    // Smooth movement
    this.smoothPosition = { x, y };
    
    // Properties - 50% larger
    this.baseSize = 8.424;  // 50% larger than 5.616
    this.size = this.baseSize;
    this.maxSize = 15.795;  // 50% larger than 10.53
    this.speed = 0.4;  // Much slower to match protozoa
    this.huntRadius = 120; // Increased for larger size
    this.attackRadius = 12;  // Increased for larger size
    this.energy = 1.5; // Start with more energy
    this.maxEnergy = 2.5; // Increased max energy
    this.alive = true;
    this.age = 0;
    this.lastMealTime = 0;
    this.growthFactor = 1.0;
    
    // Natural movement variables
    this.movementDirection = { x: Math.random() - 0.5, y: Math.random() - 0.5 };
    this.directionChangeTimer = 0;
    this.directionChangePeriod = 3 + Math.random() * 4; // Change direction every 3-7 seconds
    this.movementMomentum = { x: 0, y: 0 };
    
    // Visual
    this.color = '#FF4444';
    this.glowIntensity = 0.5;
    this.lightFlash = 0;
    this.lightIntensity = 2.0; // Brightness of predator's bioluminescence
    this.tentacles = this.generateTentacles();
    
    // Blinking effect like protozoa
    this.blinkPhase = Math.random() * Math.PI * 2;
    this.blinkFrequency = 0.5 + Math.random() * 1.5; // Variable blink speed
    this.opacity = 1.0;
    
    // Memory and behavior
    this.memory = {
      lastHuntPosition: { x, y },
      timeSinceLastHunt: 0,
      failedHuntAttempts: 0,
      visitedAreas: [],
      currentPatrolTarget: null
    };
    this.patrolRadius = 150;
    this.boredomThreshold = 15; // Time to wait before moving to new area
  }
  
  generateTentacles() {
    const tentacles = [];
    const count = 3; // Reduced from 6 to 3
    
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      tentacles.push({
        angle: angle,
        length: 8 + Math.random() * 5,
        phase: Math.random() * Math.PI * 2,
        amplitude: 2 + Math.random() * 1
      });
    }
    
    return tentacles;
  }
  
  update(deltaTime, organisms, safeZones = []) {
    if (!this.alive) return;
    
    // Update age
    this.age += deltaTime;
    
    // Update memory timers
    this.memory.timeSinceLastHunt += deltaTime;
    
    // Update light flash
    this.lightFlash = Math.max(0, this.lightFlash - deltaTime * 2);
    
    // Update blinking effect
    this.blinkPhase += deltaTime * this.blinkFrequency;
    this.opacity = 0.4 + 0.6 * Math.abs(Math.sin(this.blinkPhase));
    
    // Update size based on energy and age
    this.updateSize();
    
    // Check if in safe zone
    const inSafeZone = this.isInSafeZone(safeZones);
    
    // Find nearest organism (not in safe zone)
    const target = this.findNearestOrganism(organisms, safeZones);
    
    // Movement decision based on target (like organism perception/decision)
    const decision = { x: 0, y: 0 };
    
    if (target && target.distance < this.huntRadius && !inSafeZone) {
      // Hunt mode - similar to organism chemotaxis
      const dx = target.organism.position.x - this.position.x;
      const dy = target.organism.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist > 0 && !this.wouldEnterSafeZone(this.position.x + dx/dist, this.position.y + dy/dist, safeZones)) {
        // Signal strength based on distance (like chemical gradient)
        const signalStrength = (1 - target.distance / this.huntRadius) * 2;
        
        // Movement decision scaled by hunger (like organisms)
        const hungerLevel = Math.min(1, (this.age - this.lastMealTime) / 20);
        const urgency = signalStrength * (1 + hungerLevel);
        
        decision.x = (dx / dist) * urgency * 1.5; // Much slower hunting
        decision.y = (dy / dist) * urgency * 1.5; // Much slower hunting
        
        // Increase glow when hunting
        this.glowIntensity = Math.min(1.0, this.glowIntensity + deltaTime);
        
        // Flash light periodically when hunting
        if (Math.random() < deltaTime * 2) {
          this.lightFlash = 1.0;
        }
      } else {
        // Move away from safe zone
        this.avoidSafeZone(safeZones, deltaTime);
      }
      
      // Attack if close enough (with feeding cooldown)
      if (target.distance < this.attackRadius) {
        // Check feeding cooldown - can't eat constantly
        const timeSinceLastMeal = this.age - this.lastMealTime;
        const feedingCooldown = 8.0; // 8 seconds between meals - more time for organisms
        
        if (timeSinceLastMeal < feedingCooldown) {
          // Still digesting - can chase but not damage/feed
          this.glowIntensity = 0.5; // Dimmer when digesting
          return;
        }
        
        let damage = deltaTime * 0.8 * this.growthFactor; // Reduced damage
        
        // Protection reduces damage
        const protection = target.organism.capabilities.protection || 0;
        damage *= (1 - protection * 0.5); // Up to 50% damage reduction
        
        target.organism.energy -= damage;
        
        // Check if organism has toxin defense
        const toxinDefense = target.organism.getToxinDefense ? target.organism.getToxinDefense() : 0;
        
        // Take damage from toxin
        if (toxinDefense > 0) {
          const toxinDamage = deltaTime * toxinDefense * 0.5;
          this.energy -= toxinDamage;
          // Visual feedback for toxin damage
          this.color = '#FF8844'; // Orange tint when poisoned
        }
        
        // Check for electric defense
        const electricDefense = target.organism.getElectricDefense ? target.organism.getElectricDefense() : 0;
        if (electricDefense > 0 && Math.random() < electricDefense) {
          // Electric shock stuns predator
          this.velocity.x *= -2; // Knocked back
          this.velocity.y *= -2;
          this.energy -= electricDefense * 0.3; // Energy loss from shock
          return; // Can't feed while stunned
        }
        
        // Gain energy from feeding - reduced if prey is toxic
        const energyGain = deltaTime * 0.8 * (1 + this.growthFactor * 0.3) * (1 - toxinDefense * 0.5);
        this.energy = Math.min(this.maxEnergy, this.energy + energyGain);
        
        // Record meal time
        this.lastMealTime = this.age;
        
        // Bright flash on attack
        this.lightFlash = 1.0;
        
        // Kill organism if energy depleted
        if (target.organism.energy <= 0) {
          target.organism.alive = false;
          // Much bigger bonus energy for successful kill
          this.energy = Math.min(this.maxEnergy, this.energy + 1.5); // Big energy boost
          
          // Remember successful hunt location
          this.memory.lastHuntPosition = { ...this.position };
          this.memory.timeSinceLastHunt = 0;
          this.memory.failedHuntAttempts = 0;
        }
      }
    } else {
      // No prey found - intelligent patrol mode
      if (inSafeZone) {
        // Emergency exit from safe zone
        this.avoidSafeZone(safeZones, deltaTime * 2);
      } else {
        // Check if stuck in same area too long
        if (this.memory.timeSinceLastHunt > this.boredomThreshold) {
          // Time to move to a new area
          if (!this.memory.currentPatrolTarget || this.hasReachedTarget()) {
            this.selectNewPatrolTarget();
          }
          
          // Move towards patrol target
          if (this.memory.currentPatrolTarget) {
            const dx = this.memory.currentPatrolTarget.x - this.position.x;
            const dy = this.memory.currentPatrolTarget.y - this.position.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist > 10) {
              this.velocity.x += (dx / dist) * this.speed * deltaTime * 0.5;
              this.velocity.y += (dy / dist) * this.speed * deltaTime * 0.5;
            }
          }
        } else {
          // No prey - random walk like organisms  
          const angle = Math.random() * Math.PI * 2;
          const metabolicRate = Math.max(0.3, this.energy);
          decision.x = Math.cos(angle) * metabolicRate * 0.8; // Very slow wandering
          decision.y = Math.sin(angle) * metabolicRate * 0.8; // Very slow wandering
        }
        
        // Remember this area as visited
        this.updateVisitedAreas();
      }
      
      // Reduce glow when not hunting
      this.glowIntensity = Math.max(0.3, this.glowIntensity - deltaTime * 0.5);
    }
    
    // Apply movement decision like organisms - match protozoa speed
    const maxSpeed = this.speed * 80; // Match organism scale (0.4 * 80 = 32, vs organism 0.6 * 200 = 120)
    const acceleration = 10 + this.speed * 50; // Match organism acceleration
    
    // Add momentum like organisms
    if (!this.movementMomentum) this.movementMomentum = { x: 0, y: 0 };
    const momentumDecay = 0.9;
    this.movementMomentum.x = this.movementMomentum.x * momentumDecay + decision.x * 0.3;
    this.movementMomentum.y = this.movementMomentum.y * momentumDecay + decision.y * 0.3;
    
    // Combine decision with momentum
    const finalDecision = {
      x: decision.x + this.movementMomentum.x,
      y: decision.y + this.movementMomentum.y
    };
    
    // Apply acceleration
    this.velocity.x += finalDecision.x * acceleration * deltaTime;
    this.velocity.y += finalDecision.y * acceleration * deltaTime;
    
    // Natural friction like organisms
    this.velocity.x *= 0.995;
    this.velocity.y *= 0.995;
    
    // Speed limit
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Update position with deltaTime like organisms
    this.position.x += this.velocity.x * deltaTime;
    this.position.y += this.velocity.y * deltaTime;
    
    // Smooth visual position - less smoothing for more responsive movement
    const smoothing = 0.4;
    this.smoothPosition.x += (this.position.x - this.smoothPosition.x) * smoothing;
    this.smoothPosition.y += (this.position.y - this.smoothPosition.y) * smoothing;
    
    // World bounds with smooth wrapping
    const worldWidth = this.worldSize.width;
    const worldHeight = this.worldSize.height;
    
    // Handle X wrapping
    if (this.position.x < 0) {
      this.position.x = worldWidth + this.position.x;
      this.smoothPosition.x = worldWidth + this.smoothPosition.x;
    }
    if (this.position.x > worldWidth) {
      this.position.x = this.position.x - worldWidth;
      this.smoothPosition.x = this.smoothPosition.x - worldWidth;
    }
    
    // Handle Y wrapping
    if (this.position.y < 0) {
      this.position.y = worldHeight + this.position.y;
      this.smoothPosition.y = worldHeight + this.smoothPosition.y;
    }
    if (this.position.y > worldHeight) {
      this.position.y = this.position.y - worldHeight;
      this.smoothPosition.y = this.smoothPosition.y - worldHeight;
    }
    
    // Energy consumption - increases with size and age
    const baseConsumption = 0.008; // Reduced from 0.015
    const sizeMultiplier = this.size / this.baseSize;
    const ageMultiplier = 1 + (this.age / 200) * 0.3; // Slower aging effect
    const energyLoss = deltaTime * baseConsumption * sizeMultiplier * ageMultiplier;
    
    this.energy -= energyLoss;
    
    // Starvation check - much more lenient
    const timeSinceLastMeal = this.age - this.lastMealTime;
    const starvationThreshold = 150; // Much longer survival without food
    
    if (this.energy <= 0 || timeSinceLastMeal > starvationThreshold) {
      this.alive = false;
    }
    
    // Hunger affects behavior
    const hungerLevel = Math.min(1, timeSinceLastMeal / starvationThreshold);
    if (hungerLevel > 0.7) {
      // Desperate hunting - increased speed and hunt radius
      this.huntRadius = 100 + hungerLevel * 50;
      this.speed = 0.8 + hungerLevel * 0.3;  // Keep natural speed scaling
      this.glowIntensity = Math.min(1.0, this.glowIntensity + deltaTime * hungerLevel);
    }
  }
  
  updateSize() {
    // Growth based on energy and age
    const energyFactor = this.energy / this.maxEnergy;
    const ageFactor = Math.min(1, this.age / 50); // Mature at 50 time units
    
    // Calculate growth
    this.growthFactor = 0.5 + energyFactor * 0.5 + ageFactor * 0.5;
    this.size = Math.min(this.maxSize, this.baseSize * this.growthFactor);
    
    // Update attack radius based on size (scaled for 50% larger predator)
    this.attackRadius = 12 + (this.size - this.baseSize) * 0.5;
  }
  
  canReproduce() {
    // Can reproduce when well-fed, mature, and sufficient energy
    return this.energy > 1.2 && // Lowered from 1.5
           this.age > 15 && // Lowered from 20
           this.growthFactor > 0.7 && // Lowered from 0.8
           (this.age - this.lastMealTime) < 20; // More lenient - within last 20 time units
  }
  
  reproduce() {
    // Create offspring at reduced energy cost
    this.energy -= 0.3; // Reduced cost from 0.5
    
    // Offspring spawns near parent
    const angle = Math.random() * Math.PI * 2;
    const distance = 30 + Math.random() * 20;
    const x = this.position.x + Math.cos(angle) * distance;
    const y = this.position.y + Math.sin(angle) * distance;
    
    const offspring = new Predator(x, y, this.worldSize);
    offspring.energy = 0.8; // Start with more energy
    offspring.size = this.baseSize * 0.8; // Start smaller
    offspring.lastMealTime = offspring.age; // Start as if recently fed
    
    return offspring;
  }
  
  findNearestOrganism(organisms, safeZones) {
    let nearest = null;
    let minDist = Infinity;
    
    organisms.forEach(organism => {
      if (!organism.alive) return;
      
      // Skip organisms in safe zones
      const organismInSafeZone = safeZones.some(zone => {
        const dx = organism.position.x - zone.x;
        const dy = organism.position.y - zone.y;
        return Math.sqrt(dx * dx + dy * dy) < zone.radius;
      });
      
      if (organismInSafeZone) return;
      
      const dx = organism.position.x - this.position.x;
      const dy = organism.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Stealth makes organisms harder to detect
      const stealth = organism.capabilities ? organism.capabilities.stealth || 0 : 0;
      const effectiveDistance = dist * (1 + stealth * 2); // Stealth makes organism seem farther
      
      if (effectiveDistance < minDist) {
        minDist = effectiveDistance;
        nearest = { organism, distance: dist }; // Keep real distance for movement
      }
    });
    
    return nearest;
  }
  
  isInSafeZone(safeZones) {
    return safeZones.some(zone => {
      const dx = this.position.x - zone.x;
      const dy = this.position.y - zone.y;
      return Math.sqrt(dx * dx + dy * dy) < zone.radius;
    });
  }
  
  wouldEnterSafeZone(x, y, safeZones) {
    return safeZones.some(zone => {
      const dx = x - zone.x;
      const dy = y - zone.y;
      return Math.sqrt(dx * dx + dy * dy) < zone.radius + 20; // Buffer zone
    });
  }
  
  avoidSafeZone(safeZones, deltaTime) {
    // Find nearest safe zone
    let nearestZone = null;
    let minDist = Infinity;
    
    safeZones.forEach(zone => {
      const dx = this.position.x - zone.x;
      const dy = this.position.y - zone.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < minDist) {
        minDist = dist;
        nearestZone = { zone, dx, dy, dist };
      }
    });
    
    if (nearestZone && nearestZone.dist < nearestZone.zone.radius + 50) {
      // Move away from safe zone with stronger force
      const force = 50.0; // Much stronger repulsion force
      if (nearestZone.dist > 0) {
        // Override velocity completely to escape
        this.velocity.x = (nearestZone.dx / nearestZone.dist) * force;
        this.velocity.y = (nearestZone.dy / nearestZone.dist) * force;
      } else {
        // If at center, move randomly with strong force
        this.velocity.x = (Math.random() - 0.5) * force * 2;
        this.velocity.y = (Math.random() - 0.5) * force * 2;
      }
    }
  }
  
  selectNewPatrolTarget() {
    // Generate a new target away from recently visited areas
    let bestTarget = null;
    let maxDistance = 0;
    
    // Try several random points
    for (let i = 0; i < 10; i++) {
      const angle = Math.random() * Math.PI * 2;
      const distance = this.patrolRadius * (0.5 + Math.random() * 0.5);
      const targetX = this.position.x + Math.cos(angle) * distance;
      const targetY = this.position.y + Math.sin(angle) * distance;
      
      // Wrap around world bounds
      const wrappedX = (targetX + this.worldSize.width) % this.worldSize.width;
      const wrappedY = (targetY + this.worldSize.height) % this.worldSize.height;
      
      // Calculate minimum distance to visited areas
      let minDistToVisited = Infinity;
      this.memory.visitedAreas.forEach(area => {
        const dx = wrappedX - area.x;
        const dy = wrappedY - area.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        minDistToVisited = Math.min(minDistToVisited, dist);
      });
      
      // Prefer targets far from visited areas
      if (minDistToVisited > maxDistance) {
        maxDistance = minDistToVisited;
        bestTarget = { x: wrappedX, y: wrappedY };
      }
    }
    
    this.memory.currentPatrolTarget = bestTarget || {
      x: Math.random() * this.worldSize.width,
      y: Math.random() * this.worldSize.height
    };
    
    // Reset hunt timer when moving to new area
    this.memory.timeSinceLastHunt = 0;
  }
  
  hasReachedTarget() {
    if (!this.memory.currentPatrolTarget) return true;
    
    const dx = this.memory.currentPatrolTarget.x - this.position.x;
    const dy = this.memory.currentPatrolTarget.y - this.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    return dist < 30;
  }
  
  updateVisitedAreas() {
    // Add current position to visited areas
    const currentArea = { x: this.position.x, y: this.position.y, time: this.age };
    
    // Check if already have a nearby visited area
    const nearbyArea = this.memory.visitedAreas.find(area => {
      const dx = area.x - this.position.x;
      const dy = area.y - this.position.y;
      return Math.sqrt(dx * dx + dy * dy) < 50;
    });
    
    if (!nearbyArea) {
      this.memory.visitedAreas.push(currentArea);
      
      // Keep only recent visits (last 10)
      if (this.memory.visitedAreas.length > 10) {
        this.memory.visitedAreas.shift();
      }
    }
  }
}