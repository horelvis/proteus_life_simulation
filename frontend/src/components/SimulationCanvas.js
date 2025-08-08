import React, { useRef, useEffect, useState } from 'react';
import styled from 'styled-components';

const Canvas = styled.canvas`
  width: 100%;
  height: 100%;
  cursor: crosshair;
  image-rendering: crisp-edges;
`;

const InfoOverlay = styled.div`
  position: absolute;
  top: 1rem;
  left: 1rem;
  background-color: rgba(0, 0, 0, 0.8);
  padding: 0.5rem 1rem;
  border-radius: 4px;
  font-size: 0.875rem;
  color: var(--text-secondary);
  pointer-events: none;
`;

const SimulationCanvas = ({ 
  organisms = [], 
  predators = [], 
  nutrients = [],
  safeZones = [],
  environmentalField = [],
  memoryAnchors = [],
  onOrganismClick, 
  onCanvasClick,
  selectedOrganismId 
}) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [worldSize, setWorldSize] = useState({ width: 800, height: 600 });
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  // Zoom removed for better performance
  const zoom = 1;
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [isHoveringOrganism, setIsHoveringOrganism] = useState(false);
  const feedingEffectsRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Zoom functionality removed for better performance

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      // Wheel event removed
    };
  }, []);

  // Background particles for water effect
  const particlesRef = useRef([]);
  
  useEffect(() => {
    // Initialize background particles
    const particleCount = 50;
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * worldSize.width,
      y: Math.random() * worldSize.height,
      size: Math.random() * 2 + 0.5,
      speed: Math.random() * 0.5 + 0.1,
      opacity: Math.random() * 0.3 + 0.1
    }));
  }, [worldSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    
    const drawSafeZones = (ctx, zones) => {
      // Draw safe zones passed from simulation
      zones.forEach(zone => {
        // Gradiente de la zona segura
        const gradient = ctx.createRadialGradient(
          zone.x, zone.y, 0,
          zone.x, zone.y, zone.radius
        );
        gradient.addColorStop(0, 'rgba(0, 255, 100, 0.15)');
        gradient.addColorStop(0.7, 'rgba(0, 200, 80, 0.08)');
        gradient.addColorStop(1, 'rgba(0, 150, 60, 0.02)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(zone.x, zone.y, zone.radius, 0, Math.PI * 2);
        ctx.fill();

        // Borde de la zona
        ctx.strokeStyle = 'rgba(0, 255, 100, 0.3)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 10]);
        ctx.beginPath();
        ctx.arc(zone.x, zone.y, zone.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Indicador de texto
        ctx.fillStyle = 'rgba(0, 255, 100, 0.5)';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('SAFE ZONE', zone.x, zone.y);
      });
    };
    
    const drawEnvironmentalField = (ctx, field) => {
      // Draw pheromone traces
      field.forEach(cell => {
        const { x, y, danger, food, activity, size } = cell;
        
        ctx.save();
        ctx.globalAlpha = 0.3;
        
        // Danger pheromones - red
        if (danger > 0.1) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
          const dangerAlpha = Math.min(1, Math.max(0, danger * 0.5));
          gradient.addColorStop(0, `rgba(255, 0, 0, ${dangerAlpha})`);
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fillRect(x - size/2, y - size/2, size, size);
        }
        
        // Food pheromones - green
        if (food > 0.1) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
          const foodAlpha = Math.min(1, Math.max(0, food * 0.3));
          gradient.addColorStop(0, `rgba(0, 255, 0, ${foodAlpha})`);
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fillRect(x - size/2, y - size/2, size, size);
        }
        
        // Activity pheromones - blue
        if (activity > 0.1) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, size);
          const activityAlpha = Math.min(1, Math.max(0, activity * 0.2));
          gradient.addColorStop(0, `rgba(100, 150, 255, ${activityAlpha})`);
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fillRect(x - size/2, y - size/2, size, size);
        }
        
        ctx.restore();
      });
    };
    
    const drawMemoryAnchors = (ctx, anchors) => {
      // Draw strong memory points
      anchors.forEach(anchor => {
        const { position, type, strength } = anchor;
        
        ctx.save();
        ctx.globalAlpha = strength * 0.5;
        
        // Memory anchor visualization
        const time = Date.now() * 0.001;
        const pulse = 0.8 + Math.sin(time * 2) * 0.2;
        
        // Outer ring
        ctx.strokeStyle = type === 'death' ? '#FF6666' : '#66CCFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(position.x, position.y, 15 * pulse, 0, Math.PI * 2);
        ctx.stroke();
        
        // Inner dot
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(position.x, position.y, 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Memory waves
        for (let i = 0; i < 3; i++) {
          const waveRadius = 20 + i * 10 + (time * 10) % 30;
          const waveAlpha = Math.max(0, 1 - waveRadius / 50) * strength;
          
          ctx.strokeStyle = type === 'death' ? 
            `rgba(255, 100, 100, ${waveAlpha * 0.3})` : 
            `rgba(100, 200, 255, ${waveAlpha * 0.3})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(position.x, position.y, waveRadius, 0, Math.PI * 2);
          ctx.stroke();
        }
        
        ctx.restore();
      });
    };
    
    const render = () => {
      // Clear canvas with dark water background (20% brighter)
      ctx.fillStyle = '#0A0C14';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add subtle gradient background (20% brighter)
      const bgGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      bgGradient.addColorStop(0, '#0A0C14');
      bgGradient.addColorStop(0.5, '#0B0E1A');
      bgGradient.addColorStop(1, '#0C1020');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Calculate scale with zoom
      const scaleX = canvas.width / worldSize.width;
      const scaleY = canvas.height / worldSize.height;
      const baseScale = Math.min(scaleX, scaleY);
      const scale = baseScale * zoom;
      
      // Center the world with pan
      const offsetX = (canvas.width - worldSize.width * scale) / 2 + pan.x;
      const offsetY = (canvas.height - worldSize.height * scale) / 2 + pan.y;

      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);

      // Draw background particles
      ctx.fillStyle = 'rgba(100, 150, 200, 0.4)';
      particlesRef.current.forEach(particle => {
        ctx.globalAlpha = particle.opacity;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        
        // Update particle position
        particle.y -= particle.speed;
        if (particle.y < -10) {
          particle.y = worldSize.height + 10;
          particle.x = Math.random() * worldSize.width;
        }
      });
      ctx.globalAlpha = 1;

      // Draw grid
      drawGrid(ctx, worldSize);
      
      // Draw environmental field (pheromone traces)
      drawEnvironmentalField(ctx, environmentalField);

      // Draw safe zones
      drawSafeZones(ctx, safeZones);
      
      // Draw memory anchors
      drawMemoryAnchors(ctx, memoryAnchors);

      // Draw nutrients
      nutrients.forEach(nutrient => {
        drawNutrient(ctx, nutrient);
      });

      // Draw organisms with z-ordering (smaller y values drawn first)
      const sortedOrganisms = [...organisms].sort((a, b) => a.position.y - b.position.y);
      sortedOrganisms.forEach(organism => {
        drawOrganism(ctx, organism, selectedOrganismId === organism.id);
      });

      // Draw predators
      predators.forEach(predator => {
        drawPredator(ctx, predator);
      });

      ctx.restore();

      animationRef.current = requestAnimationFrame(render);
    };

    render();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [organisms, predators, nutrients, selectedOrganismId, worldSize, environmentalField, memoryAnchors]);

  const drawGrid = (ctx, size) => {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;

    // Vertical lines
    for (let x = 0; x <= size.width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, size.height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= size.height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size.width, y);
      ctx.stroke();
    }
  };


  // Convert HSL to RGBA
  const hslToRgba = (hslColor, alpha) => {
    const match = hslColor.match(/hsl\((\d+(?:\.\d+)?),\s*(\d+)%,\s*(\d+)%\)/);
    if (!match) return `rgba(100, 200, 255, ${alpha})`;
    
    const h = parseFloat(match[1]) / 360;
    const s = parseFloat(match[2]) / 100;
    const l = parseFloat(match[3]) / 100;
    
    let r, g, b;
    
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p, q, t) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };
      
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    
    return `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, ${alpha})`;
  };

  const drawOrganism = (ctx, organism, isSelected) => {
    const { position, velocity, color, energy, organs = [], age = 0, maxAge = 100, generation = 0, isPanicked = false } = organism;
    
    // Draw trajectory if selected
    if (isSelected && organism.trajectory) {
      ctx.strokeStyle = 'rgba(100, 150, 200, 0.3)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      organism.trajectory.forEach((point, i) => {
        if (i === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      });
      ctx.stroke();
    }

    ctx.save();
    ctx.translate(position.x, position.y);

    // Calculate organism size
    const baseSize = 8;
    const size = baseSize + energy * 2 + Math.sin(age * 0.1) * 0.5;
    
    // Calculate aging effects
    const ageRatio = age / maxAge;
    const isAging = ageRatio > 0.6;
    const agingIntensity = isAging ? (ageRatio - 0.6) / 0.4 : 0;
    
    // Rotation based on movement
    const rotation = Math.atan2(velocity.y, velocity.x);
    ctx.rotate(rotation);

    // Time-based animation
    const time = Date.now() * 0.001;
    const breathingScale = 1 + Math.sin(time * 2 + position.x * 0.01) * 0.02;
    
    // Body dimensions
    const bodyLength = size * 1.4 * breathingScale;
    const bodyWidth = size * breathingScale;
    
    // Panic indicator
    if (isPanicked) {
      const panicFlash = Math.sin(Date.now() * 0.02) > 0;
      if (panicFlash) {
        const panicGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, size * 2);
        panicGradient.addColorStop(0, 'rgba(255, 100, 100, 0.4)');
        panicGradient.addColorStop(1, 'transparent');
        ctx.fillStyle = panicGradient;
        ctx.beginPath();
        ctx.arc(0, 0, size * 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // Create irregular oval shape with narrower center
    ctx.beginPath();
    const points = 16;
    for (let i = 0; i < points; i++) {
      const angle = (i / points) * Math.PI * 2;
      const wobble = Math.sin(angle * 3 + time) * 0.05 + Math.sin(angle * 5) * 0.03;
      
      // Create pinched center effect
      const centerPinch = 1 - Math.abs(Math.cos(angle)) * 0.2; // Narrower at sides
      
      const radiusX = (bodyLength * 0.5) * (1 + wobble);
      const radiusY = (bodyWidth * 0.5) * centerPinch * (1 + wobble * 0.5);
      const x = Math.cos(angle) * radiusX;
      const y = Math.sin(angle) * radiusY;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    
    // Main body color - blue-green tones
    const baseColor = isAging ? 
      `rgba(40, 120, 140, ${0.8 - agingIntensity * 0.3})` : 
      'rgba(50, 150, 170, 0.8)';
    
    // Dark blue outline
    ctx.strokeStyle = isAging ? 
      `rgba(20, 60, 80, ${0.9 - agingIntensity * 0.2})` : 
      'rgba(30, 70, 100, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Fill body with gradient
    const bodyGradient = ctx.createLinearGradient(-bodyLength/2, -bodyWidth/2, bodyLength/2, bodyWidth/2);
    bodyGradient.addColorStop(0, baseColor);
    bodyGradient.addColorStop(0.5, 'rgba(60, 160, 180, 0.7)');
    bodyGradient.addColorStop(1, 'rgba(40, 130, 150, 0.8)');
    ctx.fillStyle = bodyGradient;
    ctx.fill();
    
    // Draw cilia/short spines around the organism
    ctx.save();
    const ciliaCount = 24;
    for (let i = 0; i < ciliaCount; i++) {
      const angle = (i / ciliaCount) * Math.PI * 2;
      const wobblePhase = time * 3 + i * 0.5;
      const ciliaLength = 2 + Math.sin(wobblePhase) * 0.5;
      
      // Position on the edge following the pinched shape
      const centerPinch = 1 - Math.abs(Math.cos(angle)) * 0.2;
      const edgeX = Math.cos(angle) * (bodyLength * 0.5);
      const edgeY = Math.sin(angle) * (bodyWidth * 0.5 * centerPinch);
      
      // Cilia end point
      const endX = edgeX + Math.cos(angle) * ciliaLength;
      const endY = edgeY + Math.sin(angle) * ciliaLength;
      
      ctx.strokeStyle = 'rgba(20, 60, 80, 0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(edgeX, edgeY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
    ctx.restore();
    
    // Draw internal structures
    // Large nucleus
    const nucleusSize = size * 0.3;
    const nucleusX = bodyLength * 0.1;
    const nucleusY = 0;
    
    ctx.fillStyle = 'rgba(30, 80, 100, 0.6)';
    ctx.beginPath();
    ctx.arc(nucleusX, nucleusY, nucleusSize, 0, Math.PI * 2);
    ctx.fill();
    
    // Nucleus inner structure
    ctx.fillStyle = 'rgba(20, 60, 80, 0.8)';
    ctx.beginPath();
    ctx.arc(nucleusX, nucleusY, nucleusSize * 0.5, 0, Math.PI * 2);
    ctx.fill();
    
    // Vacuoles and other organelles
    const vacuolePositions = [
      { x: -bodyLength * 0.2, y: bodyWidth * 0.2, size: size * 0.15 },
      { x: -bodyLength * 0.1, y: -bodyWidth * 0.3, size: size * 0.1 },
      { x: bodyLength * 0.3, y: bodyWidth * 0.1, size: size * 0.12 },
      { x: bodyLength * 0.2, y: -bodyWidth * 0.2, size: size * 0.08 }
    ];
    
    vacuolePositions.forEach(vac => {
      ctx.fillStyle = 'rgba(100, 180, 200, 0.4)';
      ctx.beginPath();
      ctx.arc(vac.x, vac.y, vac.size, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = 'rgba(80, 140, 160, 0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
    
    // Small dots/granules throughout
    ctx.fillStyle = 'rgba(40, 100, 120, 0.5)';
    for (let i = 0; i < 8; i++) {
      const granuleX = (Math.random() - 0.5) * bodyLength * 0.8;
      const granuleY = (Math.random() - 0.5) * bodyWidth * 0.8;
      const granuleSize = 1 + Math.random();
      
      ctx.beginPath();
      ctx.arc(granuleX, granuleY, granuleSize, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Aging effects - darken and add spots
    if (isAging) {
      ctx.fillStyle = `rgba(0, 0, 0, ${agingIntensity * 0.15})`;
      ctx.fillRect(-bodyLength/2, -bodyWidth/2, bodyLength, bodyWidth);
    }
    

    // Visual indicators for special organs
    const photosensor = organs.find(o => o.type === 'photosensor' && o.functionality > 0.1);
    if (photosensor) {
      // Eyespot near front
      const eyespotX = bodyLength * 0.35;
      const eyespotY = -bodyWidth * 0.2;
      
      ctx.fillStyle = `rgba(200, 100, 50, ${photosensor.functionality * 0.8})`;
      ctx.beginPath();
      ctx.arc(eyespotX, eyespotY, size * 0.1, 0, Math.PI * 2);
      ctx.fill();
      
      // Eyespot outline
      ctx.strokeStyle = 'rgba(150, 70, 30, 0.8)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    
    // Chemoreceptor indication
    const chemoreceptor = organs.find(o => o.type === 'chemoreceptor' && o.functionality > 0.1);
    if (chemoreceptor) {
      // Sensory patches at front
      ctx.fillStyle = `rgba(80, 150, 100, ${chemoreceptor.functionality * 0.4})`;
      for (let i = -1; i <= 1; i++) {
        ctx.beginPath();
        ctx.arc(bodyLength * 0.45, i * bodyWidth * 0.2, size * 0.08, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();

    // Selection indicator
    if (isSelected) {
      ctx.save();
      ctx.translate(position.x, position.y);
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(0, 0, size * 2, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  };

  const drawNutrient = (ctx, nutrient) => {
    const { x, y, energy_value, size = 3 } = nutrient;
    
    ctx.save();
    ctx.translate(x, y);
    
    // Time-based animation
    const time = Date.now() * 0.001;
    const pulse = 1 + Math.sin(time * 2 + x * 0.01) * 0.1;
    
    // Nutrient appearance based on energy value
    const opacity = Math.min(1, Math.max(0, 0.6 + energy_value * 0.4));
    const actualSize = size * pulse;
    
    // Outer glow
    const glowGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, actualSize * 3);
    glowGradient.addColorStop(0, `rgba(100, 255, 100, ${opacity * 0.3})`);
    glowGradient.addColorStop(0.5, `rgba(80, 200, 80, ${opacity * 0.15})`);
    glowGradient.addColorStop(1, 'rgba(60, 150, 60, 0)');
    
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(0, 0, actualSize * 3, 0, Math.PI * 2);
    ctx.fill();
    
    // Main nutrient particle
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, actualSize);
    gradient.addColorStop(0, `rgba(150, 255, 150, ${opacity})`);
    gradient.addColorStop(0.6, `rgba(100, 200, 100, ${opacity * 0.8})`);
    gradient.addColorStop(1, `rgba(80, 180, 80, ${opacity * 0.6})`);
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(0, 0, actualSize, 0, Math.PI * 2);
    ctx.fill();
    
    // Inner bright spot
    ctx.fillStyle = `rgba(200, 255, 200, ${opacity * 0.8})`;
    ctx.beginPath();
    ctx.arc(-actualSize * 0.2, -actualSize * 0.2, actualSize * 0.3, 0, Math.PI * 2);
    ctx.fill();
    
    // Sparkle effect for high-value nutrients
    if (energy_value > 0.4) {
      ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.5})`;
      ctx.lineWidth = 1;
      const sparkleLength = actualSize * 0.5;
      
      for (let i = 0; i < 4; i++) {
        const angle = (i / 4) * Math.PI * 2 + time;
        const x1 = Math.cos(angle) * actualSize;
        const y1 = Math.sin(angle) * actualSize;
        const x2 = Math.cos(angle) * (actualSize + sparkleLength);
        const y2 = Math.sin(angle) * (actualSize + sparkleLength);
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    
    ctx.restore();
  };

  const drawPredator = (ctx, predator) => {
    const { position, velocity = {x: 0, y: 0}, size = 10, light_intensity = 0.5, light_flash = 0, tentacles = [] } = predator;
    
    ctx.save();
    ctx.translate(position.x, position.y);

    // Rotation based on movement
    const rotation = Math.atan2(velocity.y || 0, velocity.x || 1);
    ctx.rotate(rotation);

    // Light flash effect
    if (light_flash > 0) {
      // Bright white flash
      const flashGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, size * 4);
      flashGradient.addColorStop(0, `rgba(255, 255, 255, ${light_flash * 0.9})`);
      flashGradient.addColorStop(0.3, `rgba(240, 240, 255, ${light_flash * 0.5})`);
      flashGradient.addColorStop(1, 'transparent');
      
      ctx.fillStyle = flashGradient;
      ctx.beginPath();
      ctx.arc(0, 0, size * 4, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Bioluminescent white glow based on hunt intensity
    const glowRadius = size * (1.5 + light_intensity);
    const glowGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, glowRadius);
    glowGradient.addColorStop(0, `rgba(255, 255, 255, ${light_intensity * 0.6})`);
    glowGradient.addColorStop(0.5, `rgba(220, 220, 255, ${light_intensity * 0.3})`);
    glowGradient.addColorStop(1, 'transparent');
    
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(0, 0, glowRadius, 0, Math.PI * 2);
    ctx.fill();

    // Predator body - stealthy deep sea creature design
    
    // Shadow/camouflage effect
    const shadowGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, size * 1.5);
    shadowGradient.addColorStop(0, 'rgba(0, 0, 0, 0.3)');
    shadowGradient.addColorStop(1, 'transparent');
    ctx.fillStyle = shadowGradient;
    ctx.beginPath();
    ctx.arc(0, 0, size * 1.5, 0, Math.PI * 2);
    ctx.fill();
    
    // Tentacles/appendages
    ctx.strokeStyle = 'rgba(20, 30, 40, 0.8)';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2 + Math.sin(Date.now() * 0.001 + i) * 0.1;
      const tentacleLength = size * (1.2 + Math.sin(Date.now() * 0.002 + i * 2) * 0.2);
      
      ctx.beginPath();
      ctx.moveTo(0, 0);
      
      // Curved tentacle
      const cp1x = Math.cos(angle) * size * 0.5;
      const cp1y = Math.sin(angle) * size * 0.5;
      const cp2x = Math.cos(angle + 0.2) * tentacleLength * 0.8;
      const cp2y = Math.sin(angle + 0.2) * tentacleLength * 0.8;
      const endX = Math.cos(angle + 0.1) * tentacleLength;
      const endY = Math.sin(angle + 0.1) * tentacleLength;
      
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);
      ctx.stroke();
      
      // Sucker dots
      ctx.fillStyle = 'rgba(40, 50, 60, 0.6)';
      ctx.beginPath();
      ctx.arc(endX, endY, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Main body - dark and translucent
    const bodyGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, size);
    bodyGradient.addColorStop(0, 'rgba(30, 40, 50, 0.9)');
    bodyGradient.addColorStop(0.5, 'rgba(20, 30, 40, 0.8)');
    bodyGradient.addColorStop(1, 'rgba(10, 20, 30, 0.7)');
    ctx.fillStyle = bodyGradient;
    ctx.beginPath();
    
    // Irregular body shape
    const points = 12;
    for (let i = 0; i < points; i++) {
      const angle = (i / points) * Math.PI * 2;
      const wobble = Math.sin(angle * 3 + Date.now() * 0.001) * 2;
      const r = size + wobble;
      
      if (i === 0) {
        ctx.moveTo(Math.cos(angle) * r, Math.sin(angle) * r);
      } else {
        ctx.lineTo(Math.cos(angle) * r, Math.sin(angle) * r);
      }
    }
    ctx.closePath();
    ctx.fill();

    // Internal organs visible through translucent body
    ctx.fillStyle = 'rgba(60, 80, 100, 0.4)';
    ctx.beginPath();
    ctx.arc(-3, -2, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(2, 3, 3, 0, Math.PI * 2);
    ctx.fill();

    // Multiple eyes for 360° vision
    const eyePositions = [
      { x: size * 0.4, y: 0, size: 2.5 },
      { x: -size * 0.3, y: -size * 0.3, size: 2 },
      { x: -size * 0.3, y: size * 0.3, size: 2 },
    ];
    
    eyePositions.forEach(eye => {
      // Eye glow
      ctx.fillStyle = light_intensity > 0.7 ? 'rgba(100, 150, 255, 0.6)' : 'rgba(255, 100, 100, 0.3)';
      ctx.beginPath();
      ctx.arc(eye.x, eye.y, eye.size + 1, 0, Math.PI * 2);
      ctx.fill();
      
      // Eye
      ctx.fillStyle = 'rgba(200, 220, 240, 0.9)';
      ctx.beginPath();
      ctx.arc(eye.x, eye.y, eye.size, 0, Math.PI * 2);
      ctx.fill();
      
      // Pupil
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.arc(eye.x, eye.y, eye.size * 0.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.restore();
  };


  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    console.log('Canvas clicked, organisms count:', organisms.length);

    // Convert to world coordinates with zoom and pan
    const scaleX = canvas.width / worldSize.width;
    const scaleY = canvas.height / worldSize.height;
    const baseScale = Math.min(scaleX, scaleY);
    const scale = baseScale * zoom;
    const offsetX = (canvas.width - worldSize.width * scale) / 2 + pan.x;
    const offsetY = (canvas.height - worldSize.height * scale) / 2 + pan.y;

    const worldX = (x - offsetX) / scale;
    const worldY = (y - offsetY) / scale;

    // Check if clicking on an organism
    const clickRadius = 25; // Increased from 15 for easier selection
    const clickedOrganism = organisms.find(org => {
      const dist = Math.sqrt(
        (org.position.x - worldX) ** 2 + 
        (org.position.y - worldY) ** 2
      );
      return dist < clickRadius;
    });

    if (clickedOrganism && onOrganismClick) {
      console.log('Organism clicked:', clickedOrganism.id);
      onOrganismClick(clickedOrganism);
    } else if (onCanvasClick) {
      onCanvasClick(worldX, worldY, e);
    }
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Handle panning
    if (isPanning) {
      const dx = x - lastMousePos.x;
      const dy = y - lastMousePos.y;
      setPan(prevPan => ({
        x: prevPan.x + dx,
        y: prevPan.y + dy
      }));
      setLastMousePos({ x, y });
      return;
    }

    // Update world coordinates
    const scaleX = canvas.width / worldSize.width;
    const scaleY = canvas.height / worldSize.height;
    const baseScale = Math.min(scaleX, scaleY);
    const scale = baseScale * zoom;
    const offsetX = (canvas.width - worldSize.width * scale) / 2 + pan.x;
    const offsetY = (canvas.height - worldSize.height * scale) / 2 + pan.y;

    const worldX = Math.round((x - offsetX) / scale);
    const worldY = Math.round((y - offsetY) / scale);

    setMousePos({ x: worldX, y: worldY });
    
    // Check if hovering over an organism
    const hoveringOrganism = organisms.some(org => {
      const dist = Math.sqrt(
        (org.position.x - worldX) ** 2 + 
        (org.position.y - worldY) ** 2
      );
      return dist < 25;
    });
    setIsHoveringOrganism(hoveringOrganism);
  };

  const handleMouseDown = (e) => {
    // Middle mouse or cmd+click for panning (Mac)
    if (e.button === 1 || (e.button === 0 && (e.metaKey || e.ctrlKey))) {
      e.preventDefault();
      const rect = canvasRef.current.getBoundingClientRect();
      setIsPanning(true);
      setLastMousePos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  return (
    <>
      <Canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isPanning ? 'grabbing' : isHoveringOrganism ? 'pointer' : 'grab' }}
      />
      <InfoOverlay>
        Position: ({mousePos.x}, {mousePos.y}) • 
        Organisms: {organisms.filter(o => o.alive).length} • 
        Predators: {predators.length} • 
        Zoom: {(zoom * 100).toFixed(0)}%
        <br />
        <small>
          Shift+Click: Spawn • Option+Click: Predator • 
          Scroll: Zoom • Cmd+Drag: Pan • 
          Green zones: Safe havens
        </small>
      </InfoOverlay>
    </>
  );
};

export default SimulationCanvas;