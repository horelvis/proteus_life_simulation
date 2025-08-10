/**
 * Simplified visualization for AC-based simulation
 */

import React, { useRef, useEffect, useState } from 'react';

export const SimulationACCanvas = ({ simulation, worldSize }) => {
  const canvasRef = useRef(null);
  const [selectedOrganism, setSelectedOrganism] = useState(null);
  const [showACGrid, setShowACGrid] = useState(true);
  const [showPheromones, setShowPheromones] = useState(true);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simulation) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 800;
    canvas.height = 600;
    
    let animationId;
    
    const render = () => {
      // Clear canvas
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Get scale
      const scaleX = canvas.width / worldSize.width;
      const scaleY = canvas.height / worldSize.height;
      const scale = Math.min(scaleX, scaleY);
      
      // Center the world
      ctx.save();
      ctx.translate(
        (canvas.width - worldSize.width * scale) / 2,
        (canvas.height - worldSize.height * scale) / 2
      );
      ctx.scale(scale, scale);
      
      // Get entities
      const entities = simulation.getEntities();
      
      // Draw pheromones (if enabled)
      if (showPheromones && entities.pheromones) {
        entities.pheromones.forEach(pheromone => {
          ctx.save();
          
          // Set color based on type
          let color;
          switch(pheromone.type) {
            case 'FOOD':
              color = `rgba(0, 255, 0, ${pheromone.intensity * 0.3})`;
              break;
            case 'DANGER':
              color = `rgba(255, 0, 0, ${pheromone.intensity * 0.3})`;
              break;
            case 'COLONY':
              color = `rgba(0, 100, 255, ${pheromone.intensity * 0.3})`;
              break;
            case 'MATING':
              color = `rgba(255, 0, 255, ${pheromone.intensity * 0.3})`;
              break;
            default:
              color = `rgba(255, 255, 255, ${pheromone.intensity * 0.2})`;
          }
          
          ctx.fillStyle = color;
          ctx.fillRect(
            pheromone.x - 10,
            pheromone.y - 10,
            20,
            20
          );
          
          ctx.restore();
        });
      }
      
      // Draw nutrients
      entities.nutrients.forEach(nutrient => {
        ctx.save();
        ctx.translate(nutrient.x, nutrient.y);
        
        // Simple green circle
        ctx.fillStyle = `rgba(0, 255, 0, ${nutrient.energy})`;
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
      });
      
      // Draw predators
      entities.predators.forEach(predator => {
        ctx.save();
        ctx.translate(predator.position.x, predator.position.y);
        
        // Red triangle
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.beginPath();
        ctx.moveTo(0, -10);
        ctx.lineTo(-8, 8);
        ctx.lineTo(8, 8);
        ctx.closePath();
        ctx.fill();
        
        // Flash effect
        if (predator.lightFlash > 0) {
          ctx.fillStyle = `rgba(255, 100, 100, ${predator.lightFlash * 0.3})`;
          ctx.beginPath();
          ctx.arc(0, 0, 30, 0, Math.PI * 2);
          ctx.fill();
        }
        
        ctx.restore();
      });
      
      // Draw organisms
      entities.organisms.forEach(organism => {
        ctx.save();
        ctx.translate(organism.position.x, organism.position.y);
        
        // Color based on generation
        const hue = (organism.generation * 30) % 360;
        const brightness = 50 + organism.energy * 30;
        ctx.fillStyle = `hsl(${hue}, 70%, ${brightness}%)`;
        
        // Body shape affected by oscillation
        const size = organism.size;
        ctx.beginPath();
        ctx.arc(0, 0, size, 0, Math.PI * 2);
        ctx.fill();
        
        // Show movement vector
        const moveScale = 20;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(
          organism.effectors.movementX * moveScale,
          organism.effectors.movementY * moveScale
        );
        ctx.stroke();
        
        // Selection indicator
        if (selectedOrganism === organism) {
          ctx.strokeStyle = 'white';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(0, 0, size + 5, 0, Math.PI * 2);
          ctx.stroke();
        }
        
        ctx.restore();
      });
      
      ctx.restore();
      
      // Draw selected organism's AC grid
      if (selectedOrganism && showACGrid) {
        drawACGrid(ctx, selectedOrganism.cellularState);
      }
      
      animationId = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [simulation, worldSize, selectedOrganism, showACGrid, showPheromones]);
  
  const drawACGrid = (ctx, cellularState) => {
    if (!cellularState) return;
    
    const gridSize = 16;
    const cellSize = 10;
    const margin = 20;
    const totalSize = gridSize * cellSize;
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(margin, margin, totalSize + 20, totalSize + 20);
    
    // Draw cells
    cellularState.forEach(cell => {
      const x = margin + 10 + cell.x * cellSize;
      const y = margin + 10 + cell.y * cellSize;
      
      // Cell color based on state
      if (cell.type === 'void') {
        ctx.fillStyle = 'rgba(50, 50, 50, 0.5)';
      } else if (cell.type === 'skin') {
        // Skin color varies with rigidity (charge)
        const rigidity = cell.charge || 0.5;
        ctx.fillStyle = `rgba(${150 + rigidity * 50}, ${100 - rigidity * 30}, 50, 0.8)`;
      } else if (cell.type === 'scar') {
        // Scar tissue - pinkish
        ctx.fillStyle = 'rgba(200, 150, 150, 0.7)';
      } else {
        // Tissue - color by activation and energy
        const activation = cell.activation || 0;
        const energy = cell.energy || 0;
        const integrity = cell.structuralIntegrity || 1;
        ctx.fillStyle = `rgba(
          ${100 + activation * 155}, 
          ${50 + energy * 100}, 
          ${50 + integrity * 50}, 
          0.8
        )`;
      }
      
      ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
    });
    
    // Label
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    ctx.fillText('Cellular Automaton State', margin + 10, margin - 5);
  };
  
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Convert to world coordinates
    const scaleX = canvas.width / worldSize.width;
    const scaleY = canvas.height / worldSize.height;
    const scale = Math.min(scaleX, scaleY);
    
    const offsetX = (canvas.width - worldSize.width * scale) / 2;
    const offsetY = (canvas.height - worldSize.height * scale) / 2;
    
    const worldX = (x - offsetX) / scale;
    const worldY = (y - offsetY) / scale;
    
    // Find clicked organism
    const entities = simulation.getEntities();
    const clicked = entities.organisms.find(org => {
      const dx = org.position.x - worldX;
      const dy = org.position.y - worldY;
      return Math.sqrt(dx * dx + dy * dy) < org.size + 5;
    });
    
    console.log('Clicked organism:', clicked);
    if (clicked && clicked.cellularState) {
      console.log('Cellular state cells:', clicked.cellularState.length);
    }
    
    setSelectedOrganism(clicked || null);
  };
  
  return (
    <div className="simulation-ac-container">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{
          border: '1px solid #333',
          cursor: 'crosshair',
          backgroundColor: '#000'
        }}
      />
      
      <div className="controls" style={{ marginTop: '10px' }}>
        <label style={{ color: 'white', marginRight: '10px' }}>
          <input
            type="checkbox"
            checked={showACGrid}
            onChange={(e) => setShowACGrid(e.target.checked)}
          />
          Show AC Grid
        </label>
        
        <label style={{ color: 'white', marginRight: '10px' }}>
          <input
            type="checkbox"
            checked={showPheromones}
            onChange={(e) => setShowPheromones(e.target.checked)}
          />
          Show Pheromones
        </label>
        
        {selectedOrganism && (
          <span style={{ color: 'white', marginLeft: '20px' }}>
            Selected: Gen {selectedOrganism.generation} | 
            Energy: {selectedOrganism.energy.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  );
};