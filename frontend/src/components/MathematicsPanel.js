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

const MathematicsPanel = ({ organisms = [], predators = [], environmentalField = [], statistics = {} }) => {
  const topologyCanvasRef = useRef(null);
  const geneticsCanvasRef = useRef(null);
  const dynamicsCanvasRef = useRef(null);
  const fieldCanvasRef = useRef(null);
  
  // Draw Topology Field
  useEffect(() => {
    const canvas = topologyCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Draw manifold curvature visualization
    ctx.strokeStyle = '#00CED1';
    ctx.lineWidth = 1;
    
    // Grid representing the manifold
    const gridSize = 20;
    for (let x = 0; x < width; x += gridSize) {
      for (let y = 0; y < height; y += gridSize) {
        ctx.save();
        ctx.translate(x, y);
        
        // Simulate curvature distortion
        const distX = (x - width/2) / width;
        const distY = (y - height/2) / height;
        const curvature = Math.sin(distX * Math.PI) * Math.cos(distY * Math.PI) * 0.3;
        
        ctx.rotate(curvature);
        ctx.strokeRect(-gridSize/2, -gridSize/2, gridSize, gridSize);
        ctx.restore();
      }
    }
    
    // Draw flow lines
    ctx.strokeStyle = '#00FF00';
    ctx.globalAlpha = 0.5;
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      const startX = Math.random() * width;
      let x = startX;
      let y = 0;
      ctx.moveTo(x, y);
      
      while (y < height) {
        const flow = Math.sin(x / 50) * 20;
        x += flow;
        y += 5;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    
  }, [organisms]);
  
  // Draw Genetic Distribution
  useEffect(() => {
    const canvas = geneticsCanvasRef.current;
    if (!canvas || organisms.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Calculate mutation distribution
    const mutationRates = organisms.map(o => 
      o.inheritance?.topologicalCore?.mutability || 0
    );
    
    // Draw histogram
    const bins = 10;
    const histogram = new Array(bins).fill(0);
    mutationRates.forEach(rate => {
      const bin = Math.floor(rate * bins);
      if (bin >= 0 && bin < bins) histogram[bin]++;
    });
    
    const maxCount = Math.max(...histogram, 1);
    const barWidth = width / bins;
    
    ctx.fillStyle = '#FF69B4';
    histogram.forEach((count, i) => {
      const barHeight = (count / maxCount) * height * 0.8;
      ctx.fillRect(
        i * barWidth + 2,
        height - barHeight,
        barWidth - 4,
        barHeight
      );
      
      // Label
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${(i / bins * 100).toFixed(0)}%`,
        i * barWidth + barWidth/2,
        height - 5
      );
      ctx.fillStyle = '#FF69B4';
    });
    
    // Draw mean line
    const mean = mutationRates.reduce((a, b) => a + b, 0) / mutationRates.length;
    const meanX = mean * width;
    
    ctx.strokeStyle = '#FFD700';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(meanX, 0);
    ctx.lineTo(meanX, height);
    ctx.stroke();
    
  }, [organisms]);
  
  // Draw Population Dynamics
  useEffect(() => {
    const canvas = dynamicsCanvasRef.current;
    if (!canvas) return;
    
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
    ctx.moveTo(20, height - 20);
    ctx.lineTo(width - 10, height - 20);
    ctx.moveTo(20, height - 20);
    ctx.lineTo(20, 10);
    ctx.stroke();
    
    // Simulate population over time
    const timeSteps = 50;
    const data = [];
    let pop = organisms.length;
    
    for (let t = 0; t < timeSteps; t++) {
      // Logistic growth with stochasticity
      const K = 100; // Carrying capacity
      const r = 0.1; // Growth rate
      const noise = (Math.random() - 0.5) * 5;
      
      pop = pop + r * pop * (1 - pop/K) + noise;
      pop = Math.max(0, pop);
      data.push(pop);
    }
    
    // Draw population curve
    ctx.strokeStyle = '#00CED1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((p, i) => {
      const x = 20 + (i / timeSteps) * (width - 30);
      const y = height - 20 - (p / 100) * (height - 30);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Draw predator-prey phase space
    ctx.strokeStyle = '#FF4444';
    ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
      const preyCount = data[i];
      const predCount = predators.length * (1 + Math.sin(i * 0.2) * 0.3);
      
      const x = 20 + (preyCount / 100) * (width - 30);
      const y = height - 20 - (predCount / 10) * (height - 30);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
  }, [organisms, predators]);
  
  // Draw Environmental Field
  useEffect(() => {
    const canvas = fieldCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = canvas.offsetHeight;
    
    // Clear
    ctx.fillStyle = '#0A0C14';
    ctx.fillRect(0, 0, width, height);
    
    // Draw field intensity as heatmap
    const resolution = 20;
    const cellWidth = width / resolution;
    const cellHeight = height / resolution;
    
    for (let x = 0; x < resolution; x++) {
      for (let y = 0; y < resolution; y++) {
        // Simulate field intensity
        const centerX = width / 2;
        const centerY = height / 2;
        const dx = (x * cellWidth - centerX) / width;
        const dy = (y * cellHeight - centerY) / height;
        
        // Gaussian field with multiple centers
        const field1 = Math.exp(-(dx*dx + dy*dy) * 5);
        const field2 = Math.exp(-((dx-0.3)*(dx-0.3) + (dy+0.2)*(dy+0.2)) * 8);
        const intensity = field1 + field2 * 0.5;
        
        // Color based on intensity
        const hue = 180 - intensity * 60; // Blue to green
        const lightness = 20 + intensity * 40;
        
        ctx.fillStyle = `hsl(${hue}, 70%, ${lightness}%)`;
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
      }
    }
    
    // Draw gradient vectors
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    
    for (let x = 0; x < width; x += 30) {
      for (let y = 0; y < height; y += 30) {
        const dx = (x - width/2) / width;
        const dy = (y - height/2) / height;
        
        // Gradient direction
        const gradX = -2 * dx * Math.exp(-(dx*dx + dy*dy));
        const gradY = -2 * dy * Math.exp(-(dx*dx + dy*dy));
        const mag = Math.sqrt(gradX*gradX + gradY*gradY);
        
        if (mag > 0.01) {
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + gradX/mag * 10, y + gradY/mag * 10);
          ctx.stroke();
        }
      }
    }
    
  }, [environmentalField]);
  
  return (
    <Panel>
      <Title>
        📊 Mathematics & Dynamics
      </Title>
      
      <Formula>
        ∇²ψ + k²ψ = 0 | Topological Field Equation
      </Formula>
      
      <GraphContainer>
        <Graph>
          <GraphTitle>Manifold Topology & Flow</GraphTitle>
          <Canvas ref={topologyCanvasRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Mutation Rate Distribution</GraphTitle>
          <Canvas ref={geneticsCanvasRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Population Dynamics (Lotka-Volterra)</GraphTitle>
          <Canvas ref={dynamicsCanvasRef} />
        </Graph>
        
        <Graph>
          <GraphTitle>Environmental Field Gradient</GraphTitle>
          <Canvas ref={fieldCanvasRef} />
        </Graph>
      </GraphContainer>
      
      <Formula>
        dN/dt = rN(1-N/K) - αNP | Population Growth
      </Formula>
      
      <Formula>
        H(X) = -Σ p(x)log₂p(x) | Genetic Entropy
      </Formula>
      
      <Formula>
        ∂ρ/∂t + ∇·(ρv) = D∇²ρ | Pheromone Diffusion
      </Formula>
    </Panel>
  );
};

export default MathematicsPanel;