/**
 * Componente de animación para visualizar transformaciones y procesos ARC
 * Incluye animaciones suaves y efectos visuales para mejor comprensión
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ARC_COLORS = {
  0: '#000000',
  1: '#0074D9',
  2: '#FF4136',
  3: '#2ECC40',
  4: '#FFDC00',
  5: '#AAAAAA',
  6: '#F012BE',
  7: '#FF851B',
  8: '#7FDBFF',
  9: '#870C25',
};

const ARCProcessAnimation = ({ 
  fromGrid, 
  toGrid, 
  transformation,
  duration = 1000,
  onComplete = () => {}
}) => {
  const canvasRef = useRef(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [highlightCells, setHighlightCells] = useState([]);
  const animationRef = useRef(null);

  useEffect(() => {
    if (!fromGrid || !toGrid) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const cellSize = 30;
    const padding = 10;
    
    const height = Math.max(fromGrid.length, toGrid.length);
    const width = Math.max(fromGrid[0]?.length || 0, toGrid[0]?.length || 0);
    
    canvas.width = width * cellSize + padding * 2;
    canvas.height = height * cellSize + padding * 2;

    const startTime = Date.now();
    
    const animate = () => {
      const currentTime = Date.now();
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      setAnimationProgress(progress);
      
      // Limpiar canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Dibujar grid animado
      for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
          const fromValue = fromGrid[i]?.[j] ?? 0;
          const toValue = toGrid[i]?.[j] ?? 0;
          
          const x = j * cellSize + padding;
          const y = i * cellSize + padding;
          
          if (fromValue !== toValue) {
            // Animar cambios con efecto de fade
            const fromColor = ARC_COLORS[fromValue];
            const toColor = ARC_COLORS[toValue];
            
            // Efecto de transición
            if (progress < 0.5) {
              ctx.globalAlpha = 1 - progress * 2;
              ctx.fillStyle = fromColor;
              ctx.fillRect(x, y, cellSize, cellSize);
              
              ctx.globalAlpha = progress * 2;
              ctx.fillStyle = toColor;
              ctx.fillRect(x, y, cellSize, cellSize);
            } else {
              ctx.globalAlpha = 1;
              ctx.fillStyle = toColor;
              ctx.fillRect(x, y, cellSize, cellSize);
              
              // Efecto de brillo para celdas cambiadas
              if (progress < 0.8) {
                ctx.globalAlpha = 0.5 * (1 - (progress - 0.5) * 2);
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(x + 2, y + 2, cellSize - 4, cellSize - 4);
              }
            }
          } else {
            // Celdas sin cambios
            ctx.globalAlpha = 1;
            ctx.fillStyle = ARC_COLORS[fromValue];
            ctx.fillRect(x, y, cellSize, cellSize);
          }
          
          // Dibujar bordes
          ctx.globalAlpha = 1;
          ctx.strokeStyle = '#333';
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cellSize, cellSize);
        }
      }
      
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        onComplete();
      }
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [fromGrid, toGrid, duration, onComplete]);

  return (
    <div style={styles.container}>
      <canvas ref={canvasRef} style={styles.canvas} />
      
      {/* Indicador de progreso */}
      <div style={styles.progressBar}>
        <motion.div 
          style={styles.progressFill}
          animate={{ width: `${animationProgress * 100}%` }}
          transition={{ duration: 0.1 }}
        />
      </div>
      
      {/* Información de transformación */}
      {transformation && (
        <motion.div 
          style={styles.transformInfo}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <span style={styles.transformLabel}>Transformación:</span>
          <span style={styles.transformName}>{transformation}</span>
        </motion.div>
      )}
    </div>
  );
};

// Componente para mostrar partículas animadas durante el proceso
export const ParticleEffect = ({ active, position }) => {
  const particles = Array.from({ length: 10 }, (_, i) => ({
    id: i,
    x: Math.random() * 100 - 50,
    y: Math.random() * 100 - 50,
    delay: Math.random() * 0.5,
  }));

  if (!active) return null;

  return (
    <div style={{ ...styles.particleContainer, ...position }}>
      {particles.map(particle => (
        <motion.div
          key={particle.id}
          style={styles.particle}
          initial={{ x: 0, y: 0, opacity: 1, scale: 0 }}
          animate={{
            x: particle.x,
            y: particle.y,
            opacity: 0,
            scale: 1.5,
          }}
          transition={{
            duration: 1,
            delay: particle.delay,
            ease: "easeOut",
          }}
        />
      ))}
    </div>
  );
};

// Componente para visualizar el flujo de datos entre componentes
export const DataFlowVisualization = ({ nodes, connections, activeConnection }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !connections.length) return;

    const svg = svgRef.current;
    const rect = svg.getBoundingClientRect();
    
    // Limpiar SVG
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }

    // Dibujar conexiones
    connections.forEach((conn, idx) => {
      const fromNode = nodes.find(n => n.id === conn.from);
      const toNode = nodes.find(n => n.id === conn.to);
      
      if (!fromNode || !toNode) return;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const d = `M ${fromNode.x} ${fromNode.y} Q ${(fromNode.x + toNode.x) / 2} ${fromNode.y} ${toNode.x} ${toNode.y}`;
      
      path.setAttribute('d', d);
      path.setAttribute('stroke', activeConnection === idx ? '#FF4136' : '#444');
      path.setAttribute('stroke-width', activeConnection === idx ? '3' : '2');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke-dasharray', activeConnection === idx ? '5,5' : '0');
      
      if (activeConnection === idx) {
        const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        animate.setAttribute('attributeName', 'stroke-dashoffset');
        animate.setAttribute('values', '10;0');
        animate.setAttribute('dur', '0.5s');
        animate.setAttribute('repeatCount', 'indefinite');
        path.appendChild(animate);
      }
      
      svg.appendChild(path);
    });

    // Dibujar nodos
    nodes.forEach(node => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', node.x);
      circle.setAttribute('cy', node.y);
      circle.setAttribute('r', '8');
      circle.setAttribute('fill', node.active ? '#FF4136' : '#666');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('stroke-width', '2');
      
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', node.x);
      text.setAttribute('y', node.y - 15);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#fff');
      text.setAttribute('font-size', '12');
      text.textContent = node.label;
      
      svg.appendChild(circle);
      svg.appendChild(text);
    });
  }, [nodes, connections, activeConnection]);

  return (
    <svg 
      ref={svgRef} 
      style={styles.dataFlowSvg}
      width="100%"
      height="200"
    />
  );
};

// Estilos
const styles = {
  container: {
    position: 'relative',
    padding: '20px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
  },
  canvas: {
    display: 'block',
    margin: '0 auto',
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
  },
  progressBar: {
    height: '4px',
    backgroundColor: '#3a3a3a',
    borderRadius: '2px',
    marginTop: '15px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#0074D9',
    borderRadius: '2px',
  },
  transformInfo: {
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  transformLabel: {
    color: '#888',
    fontSize: '14px',
  },
  transformName: {
    color: '#fff',
    fontSize: '14px',
    fontWeight: 'bold',
  },
  particleContainer: {
    position: 'absolute',
    pointerEvents: 'none',
  },
  particle: {
    position: 'absolute',
    width: '4px',
    height: '4px',
    backgroundColor: '#FF4136',
    borderRadius: '50%',
  },
  dataFlowSvg: {
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    padding: '10px',
  },
};

export default ARCProcessAnimation;