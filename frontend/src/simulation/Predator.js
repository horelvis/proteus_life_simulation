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
    
    // Properties
    this.baseSize = 7.2;  // Reduced by 10% from 8
    this.size = this.baseSize;
    this.maxSize = 13.5;  // Reduced by 10% from 15
    this.speed = 1.5;
    this.huntRadius = 120; // Increased from 100
    this.attackRadius = 15;
    this.energy = 1.5; // Start with more energy
    this.maxEnergy = 2.5; // Increased max energy
    this.alive = true;
    this.age = 0;
    this.lastMealTime = 0;
    this.growthFactor = 1.0;
    
    // Visual
    this.color = '#FF4444';
    this.glowIntensity = 0.5;
    this.lightFlash = 0;
    this.tentacles = this.generateTentacles();
    
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
    const count = 6;
    
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
    
    // Update size based on energy and age
    this.updateSize();
    
    // Check if in safe zone
    const inSafeZone = this.isInSafeZone(safeZones);
    
    // Find nearest organism (not in safe zone)
    const target = this.findNearestOrganism(organisms, safeZones);
    
    if (target && target.distance < this.huntRadius && !inSafeZone) {
      // Hunt mode
      const dx = target.organism.position.x - this.position.x;
      const dy = target.organism.position.y - this.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist > 0) {
        // Check if target would lead us into safe zone
        const nextX = this.position.x + (dx / dist) * this.speed * deltaTime;
        const nextY = this.position.y + (dy / dist) * this.speed * deltaTime;
        
        if (!this.wouldEnterSafeZone(nextX, nextY, safeZones)) {
          // Accelerate towards target
          const huntSpeed = this.speed * (1 + (1 - target.distance / this.huntRadius));
          this.velocity.x += (dx / dist) * huntSpeed * deltaTime;
          this.velocity.y += (dy / dist) * huntSpeed * deltaTime;
          
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
      }
      
      // Attack if close enough
      if (target.distance < this.attackRadius) {
        let damage = deltaTime * 1.5 * this.growthFactor; // Base damage
        
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
          // Local search pattern
          const searchAngle = this.age * 0.5;
          this.velocity.x += Math.cos(searchAngle) * 0.3 * deltaTime;
          this.velocity.y += Math.sin(searchAngle) * 0.3 * deltaTime;
        }
        
        // Remember this area as visited
        this.updateVisitedAreas();
      }
      
      // Reduce glow when not hunting
      this.glowIntensity = Math.max(0.3, this.glowIntensity - deltaTime * 0.5);
    }
    
    // Apply drag
    this.velocity.x *= 0.95;
    this.velocity.y *= 0.95;
    
    // Speed limit
    const speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
    const maxSpeed = this.speed * 2;
    if (speed > maxSpeed) {
      this.velocity.x = (this.velocity.x / speed) * maxSpeed;
      this.velocity.y = (this.velocity.y / speed) * maxSpeed;
    }
    
    // Update position
    this.position.x += this.velocity.x;
    this.position.y += this.velocity.y;
    
    // Smooth visual position
    const smoothing = 0.2;
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
      this.speed = 1.5 + hungerLevel * 0.5;
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
    
    // Update attack radius based on size
    this.attackRadius = 10 + (this.size - this.baseSize) * 0.5;
  }
  
  canReproduce() {
    // Can reproduce when well-fed, mature, and sufficient energy
    return this.energy > 1.5 && 
           this.age > 20 && 
           this.growthFactor > 0.8 &&
           this.lastMealTime > this.age - 10; // Recently fed
  }
  
  reproduce() {
    // Create offspring at reduced energy cost
    this.energy -= 0.5;
    
    // Offspring spawns near parent
    const angle = Math.random() * Math.PI * 2;
    const distance = 30 + Math.random() * 20;
    const x = this.position.x + Math.cos(angle) * distance;
    const y = this.position.y + Math.sin(angle) * distance;
    
    const offspring = new Predator(x, y, this.worldSize);
    offspring.energy = 0.5; // Start with less energy
    offspring.size = this.baseSize * 0.8; // Start smaller
    
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
      // Move away from safe zone
      const force = 3.0 * deltaTime;
      if (nearestZone.dist > 0) {
        this.velocity.x += (nearestZone.dx / nearestZone.dist) * force;
        this.velocity.y += (nearestZone.dy / nearestZone.dist) * force;
      } else {
        // If at center, move randomly
        this.velocity.x += (Math.random() - 0.5) * force * 2;
        this.velocity.y += (Math.random() - 0.5) * force * 2;
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