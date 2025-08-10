/**
 * PROTEUS Nutrient - Frontend Implementation
 */

export class Nutrient {
  constructor(x, y) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.x = x;
    this.y = y;
    this.energy = 0.5; // Increased nutrient value
    this.size = 3;
    this.alive = true;
    this.age = 0;
  }
  
  update(deltaTime) {
    this.age += deltaTime;
    
    // Nutrients decay over time
    if (this.age > 30) {
      this.alive = false;
    }
  }
}