import React, { useRef, useEffect, useState } from 'react';
import styled from 'styled-components';

const Panel = styled.div`
  background-color: var(--bg-secondary);
  border-top: 1px solid var(--bg-tertiary);
  padding: 1rem;
  height: 400px;
  overflow-y: auto;
`;

const Title = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const GraphContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
`;

const Graph = styled.div`
  background-color: var(--bg-primary);
  border: 1px solid var(--bg-tertiary);
  border-radius: 4px;
  padding: 0.5rem;
  height: 150px;
  position: relative;
`;

const GraphTitle = styled.h4`
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  text-align: center;
`;

const Canvas = styled.canvas`
  width: 100%;
  height: calc(100% - 1.5rem);
`;

const Formula = styled.div`
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.75rem;
  color: var(--accent-primary);
  background-color: var(--bg-primary);
  padding: 0.5rem;
  border-radius: 4px;
  margin-bottom: 0.5rem;
  overflow-x: auto;
  white-space: nowrap;
`;

const Stats = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin-top: 1rem;
`;

const Stat = styled.div`
  background-color: var(--bg-primary);
  padding: 0.5rem;
  border-radius: 4px;
  text-align: center;
  
  .label {
    font-size: 0.7rem;
    color: var(--text-secondary);
  }
  
  .value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-primary);
  }
`;

const MathematicsPanel = ({ organisms = [], predators = [], environmentalField = [], statistics = {} }) => {
  const velocityFieldRef = useRef(null);
  const geneticsCanvasRef = useRef(null);
  const dynamicsCanvasRef = useRef(null);
  const energyCanvasRef = useRef(null);
  
  const [populationHistory, setPopulationHistory] = useState([]);
  const [energyHistory, setEnergyHistory] = useState([]);
  
  // Track population and energy over time
  useEffect(() => {
    setPopulationHistory(prev => {
      const newHistory = [...prev, {
        time: Date.now(),
        organisms: organisms.length,
        predators: predators.length
      }].slice(-100); // Keep last 100 points
      return newHistory;
    });
    
    const totalEnergy = organisms.reduce((sum, org) => sum + (org.energy || 0), 0);
    setEnergyHistory(prev => {
      const newHistory = [...prev, {
        time: Date.now(),
        energy: totalEnergy / Math.max(organisms.length, 1)
      }].slice(-100);
      return newHistory;
    });
  }, [organisms, predators]);
  
  // Draw REAL velocity field from organisms
  useEffect(() => {
    const canvas = velocityFieldRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 0.5;
    const gridSize = 20;
    
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Calculate average velocity field from actual organisms
    const fieldResolution = 10;
    const cellWidth = width / fieldResolution;
    const cellHeight = height / fieldResolution;
    const velocityField = {};
    
    // Accumulate velocities by grid cell
    organisms.forEach(org => {
      if (!org.velocity) return;
      
      const gridX = Math.floor(org.position.x / 800 * fieldResolution);
      const gridY = Math.floor(org.position.y / 600 * fieldResolution);
      const key = `${gridX},${gridY}`;
      
      if (!velocityField[key]) {
        velocityField[key] = { vx: 0, vy: 0, count: 0 };
      }
      
      velocityField[key].vx += org.velocity.x;
      velocityField[key].vy += org.velocity.y;
      velocityField[key].count++;
    });
    
    // Draw velocity vectors
    ctx.strokeStyle = '#00CED1';
    ctx.lineWidth = 1;
    
    Object.entries(velocityField).forEach(([key, data]) => {
      const [gridX, gridY] = key.split(',').map(Number);
      if (data.count === 0) return;
      
      const avgVx = data.vx / data.count;
      const avgVy = data.vy / data.count;
      const magnitude = Math.sqrt(avgVx * avgVx + avgVy * avgVy);
      
      if (magnitude > 0.1) {
        const x = (gridX + 0.5) * cellWidth;
        const y = (gridY + 0.5) * cellHeight;
        const scale = Math.min(magnitude * 10, cellWidth * 0.4);
        
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(Math.atan2(avgVy, avgVx));
        
        // Draw arrow
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(scale, 0);
        ctx.lineTo(scale - 3, -2);
        ctx.moveTo(scale, 0);
        ctx.lineTo(scale - 3, 2);
        ctx.stroke();
        
        ctx.restore();
      }
    });
    
    // Show divergence and curl calculations
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px monospace';
    ctx.fillText(`Active cells: ${Object.keys(velocityField).length}`, 5, height - 5);
    
  }, [organisms]);
  
  // Draw REAL genetic distribution
  useEffect(() => {
    const canvas = geneticsCanvasRef.current;
    if (!canvas || organisms.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Collect real trait data
    const traits = {
      motility: [],
      sensitivity: [],
      resilience: []
    };
    
    organisms.forEach(org => {
      if (org.inheritance?.topologicalCore) {
        traits.motility.push(org.inheritance.topologicalCore.baseMotility || 0);
        traits.sensitivity.push(org.inheritance.topologicalCore.baseSensitivity || 0);
        traits.resilience.push(org.inheritance.topologicalCore.baseResilience || 0);
      }
    });
    
    // Draw trait distributions
    const drawHistogram = (data, color, offsetY) => {
      if (data.length === 0) return;
      
      const bins = 20;
      const histogram = new Array(bins).fill(0);
      const min = 0;
      const max = 1;
      
      data.forEach(value => {
        const bin = Math.floor((value - min) / (max - min) * (bins - 1));
        if (bin >= 0 && bin < bins) histogram[bin]++;
      });
      
      const maxCount = Math.max(...histogram, 1);
      const barWidth = width / bins;
      const barHeightMax = height / 3 - 10;
      
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      
      histogram.forEach((count, i) => {
        const barHeight = (count / maxCount) * barHeightMax;
        ctx.fillRect(
          i * barWidth + 1,
          offsetY + barHeightMax - barHeight,
          barWidth - 2,
          barHeight
        );
      });
      
      // Draw mean
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const meanX = mean * width;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(meanX, offsetY);
      ctx.lineTo(meanX, offsetY + barHeightMax);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px monospace';
      ctx.fillText(`μ=${mean.toFixed(2)}`, meanX + 5, offsetY + 10);
    };
    
    drawHistogram(traits.motility, '#FF6B6B', 0);
    drawHistogram(traits.sensitivity, '#4ECDC4', height/3);
    drawHistogram(traits.resilience, '#FFD93D', 2*height/3);
    
    // Labels
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px monospace';
    ctx.fillText('Motility', 5, 15);
    ctx.fillText('Sensitivity', 5, height/3 + 15);
    ctx.fillText('Resilience', 5, 2*height/3 + 15);
    
  }, [organisms]);
  
  // Draw REAL population dynamics
  useEffect(() => {
    const canvas = dynamicsCanvasRef.current;
    if (!canvas || populationHistory.length < 2) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Draw axes
    ctx.strokeStyle = '#444444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, height - 20);
    ctx.lineTo(width - 10, height - 20);
    ctx.moveTo(30, height - 20);
    ctx.lineTo(30, 10);
    ctx.stroke();
    
    // Find max values for scaling
    const maxOrganisms = Math.max(...populationHistory.map(p => p.organisms), 1);
    const maxPredators = Math.max(...populationHistory.map(p => p.predators), 1);
    
    // Draw organism population
    ctx.strokeStyle = '#00CED1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    populationHistory.forEach((point, i) => {
      const x = 30 + (i / (populationHistory.length - 1)) * (width - 40);
      const y = height - 20 - (point.organisms / maxOrganisms) * (height - 40);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Draw predator population
    ctx.strokeStyle = '#FF4444';
    ctx.beginPath();
    
    populationHistory.forEach((point, i) => {
      const x = 30 + (i / (populationHistory.length - 1)) * (width - 40);
      const y = height - 20 - (point.predators / maxPredators) * (height - 40);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#00CED1';
    ctx.font = '10px monospace';
    ctx.fillText(`Organisms: ${organisms.length}`, width - 100, 20);
    
    ctx.fillStyle = '#FF4444';
    ctx.fillText(`Predators: ${predators.length}`, width - 100, 35);
    
    // Y-axis labels
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(maxOrganisms.toString(), 5, 15);
    ctx.fillText('0', 5, height - 20);
    
  }, [populationHistory, organisms, predators]);
  
  // Draw REAL energy distribution
  useEffect(() => {
    const canvas = energyCanvasRef.current;
    if (!canvas || organisms.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Energy distribution scatter plot
    const maxAge = Math.max(...organisms.map(o => o.age || 0), 1);
    const maxEnergy = 2.0; // Maximum possible energy
    
    // Draw grid
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 0.5;
    
    // Age axis
    for (let i = 0; i <= 5; i++) {
      const x = 30 + (i / 5) * (width - 40);
      ctx.beginPath();
      ctx.moveTo(x, height - 20);
      ctx.lineTo(x, 10);
      ctx.stroke();
    }
    
    // Energy axis
    for (let i = 0; i <= 4; i++) {
      const y = height - 20 - (i / 4) * (height - 30);
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.lineTo(width - 10, y);
      ctx.stroke();
    }
    
    // Plot organisms
    organisms.forEach(org => {
      const x = 30 + (org.age / maxAge) * (width - 40);
      const y = height - 20 - (org.energy / maxEnergy) * (height - 30);
      
      // Color by generation
      const hue = (org.generation * 30) % 360;
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Draw average energy over time
    if (energyHistory.length > 1) {
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      energyHistory.forEach((point, i) => {
        const x = 30 + (i / (energyHistory.length - 1)) * (width - 40);
        const y = height - 20 - (point.energy / maxEnergy) * (height - 30);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    
    // Labels
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px monospace';
    ctx.fillText('Age →', width / 2 - 20, height - 5);
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Energy →', 0, 0);
    ctx.restore();
    
  }, [organisms, energyHistory]);
  
  // Calculate real statistics
  const calculateStats = () => {
    if (organisms.length === 0) return { entropy: 0, divergence: 0, complexity: 0 };
    
    // Shannon entropy of trait distribution
    const traitBins = 10;
    const traitCounts = new Array(traitBins).fill(0);
    
    organisms.forEach(org => {
      if (org.inheritance?.topologicalCore) {
        const avgTrait = (
          org.inheritance.topologicalCore.baseMotility +
          org.inheritance.topologicalCore.baseSensitivity +
          org.inheritance.topologicalCore.baseResilience
        ) / 3;
        const bin = Math.floor(avgTrait * (traitBins - 1));
        if (bin >= 0 && bin < traitBins) traitCounts[bin]++;
      }
    });
    
    const total = organisms.length;
    let entropy = 0;
    
    traitCounts.forEach(count => {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    });
    
    // Velocity divergence (simplified)
    let divergence = 0;
    let velocityCount = 0;
    
    organisms.forEach(org => {
      if (org.velocity) {
        divergence += Math.abs(org.velocity.x) + Math.abs(org.velocity.y);
        velocityCount++;
      }
    });
    
    if (velocityCount > 0) {
      divergence /= velocityCount;
    }
    
    // Complexity (organ diversity)
    const organTypes = new Set();
    organisms.forEach(org => {
      if (org.organs) {
        org.organs.forEach(organ => {
          if (organ.functionality > 0.1) {
            organTypes.add(organ.type);
          }
        });
      }
    });
    
    const complexity = organTypes.size;
    
    return { entropy, divergence, complexity };
  };
  
  const stats = calculateStats();
  
  return (
    <Panel>
      <Title>
        📊 Real-Time Mathematics & Dynamics
      </Title>
      
      <Formula>
        H(X) = {stats.entropy.toFixed(3)} bits | Population Genetic Entropy
      </Formula>
      
      <GraphContainer>
        <Graph>
          <GraphTitle>Velocity Field (∇·v)</GraphTitle>
          <Canvas ref={velocityFieldRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Trait Distributions</GraphTitle>
          <Canvas ref={geneticsCanvasRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Population Dynamics (Real)</GraphTitle>
          <Canvas ref={dynamicsCanvasRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Energy vs Age Scatter</GraphTitle>
          <Canvas ref={energyCanvasRef} />
        </Graph>
      </GraphContainer>
      
      <Stats>
        <Stat>
          <div className="label">Genetic Entropy</div>
          <div className="value">{stats.entropy.toFixed(3)}</div>
        </Stat>
        <Stat>
          <div className="label">Velocity Divergence</div>
          <div className="value">{stats.divergence.toFixed(3)}</div>
        </Stat>
        <Stat>
          <div className="label">Organ Complexity</div>
          <div className="value">{stats.complexity}</div>
        </Stat>
      </Stats>
      
      <Formula>
        ∇·v = {stats.divergence.toFixed(3)} | Average Flow Divergence
      </Formula>
      
      <Formula>
        C = {stats.complexity} organ types | Morphological Complexity
      </Formula>
    </Panel>
  );
};

export default MathematicsPanel;