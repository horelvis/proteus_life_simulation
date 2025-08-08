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
    this.size = 10;
    this.speed = 1.5;
    this.huntRadius = 100;
    this.attackRadius = 15;
    this.energy = 1.0;
    this.alive = true;
    
    // Visual
    this.color = '#FF4444';
    this.glowIntensity = 0.5;
    this.lightFlash = 0;
    this.tentacles = this.generateTentacles();
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
    
    // Update light flash
    this.lightFlash = Math.max(0, this.lightFlash - deltaTime * 2);
    
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
        target.organism.energy -= deltaTime * 2;
        this.energy = Math.min(2.0, this.energy + deltaTime * 0.5);
        // Bright flash on attack
        this.lightFlash = 1.0;
      }
    } else {
      // Patrol mode or avoiding safe zone
      if (inSafeZone) {
        // Emergency exit from safe zone
        this.avoidSafeZone(safeZones, deltaTime * 2);
      } else {
        // Normal patrol
        this.velocity.x += (Math.random() - 0.5) * 0.5 * deltaTime;
        this.velocity.y += (Math.random() - 0.5) * 0.5 * deltaTime;
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
    
    // Energy consumption
    this.energy -= deltaTime * 0.01;
    if (this.energy <= 0) {
      this.alive = false;
    }
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
      
      if (dist < minDist) {
        minDist = dist;
        nearest = { organism, distance: dist };
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
}