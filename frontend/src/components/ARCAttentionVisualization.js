/**
 * Visualización del Sistema de Atención y Barridos Topográficos
 * Muestra cómo el sistema analiza las capas jerárquicas
 */

import React, { useState, useEffect, useRef } from 'react';
import './ARCAttentionVisualization.css';

export const ARCAttentionVisualization = ({ 
  puzzleData,
  analysisData,
  onAttentionUpdate 
}) => {
  const [currentLayer, setCurrentLayer] = useState(0);
  const [attentionMap, setAttentionMap] = useState(null);
  const [scanPosition, setScanPosition] = useState({ x: 0, y: 0 });
  const [isScanning, setIsScanning] = useState(false);
  const [discoveredPatterns, setDiscoveredPatterns] = useState([]);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  // Capas del análisis jerárquico
  const layers = [
    { name: 'Píxeles', level: 0, color: '#FF6B6B' },
    { name: 'Objetos', level: 1, color: '#4ECDC4' },
    { name: 'Relaciones', level: 2, color: '#45B7D1' },
    { name: 'Patrones', level: 3, color: '#96CEB4' }
  ];

  // Sistema de Atención con Barrido Topográfico
  const performTopographicScan = () => {
    if (!puzzleData || !analysisData) return;

    setIsScanning(true);
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { input, output } = puzzleData;
    const gridSize = input.length;
    const cellSize = canvas.width / gridSize;

    // Crear mapa de atención inicial
    const attention = Array(gridSize).fill(null).map(() => 
      Array(gridSize).fill(0)
    );

    // Barrido topográfico por capas
    let scanStep = 0;
    const maxSteps = gridSize * gridSize * layers.length;

    const scan = () => {
      if (scanStep >= maxSteps) {
        setIsScanning(false);
        analyzeAttentionResults(attention);
        return;
      }

      // Calcular posición actual del barrido
      const layerIndex = Math.floor(scanStep / (gridSize * gridSize));
      const positionInLayer = scanStep % (gridSize * gridSize);
      const x = positionInLayer % gridSize;
      const y = Math.floor(positionInLayer / gridSize);

      setScanPosition({ x, y });
      setCurrentLayer(layerIndex);

      // Analizar según la capa actual
      const layer = layers[layerIndex];
      let attentionValue = 0;

      switch (layer.level) {
        case 0: // Píxeles
          attentionValue = analyzePixelLevel(input, output, x, y);
          break;
        case 1: // Objetos
          attentionValue = analyzeObjectLevel(input, output, x, y);
          break;
        case 2: // Relaciones
          attentionValue = analyzeRelationLevel(input, output, x, y);
          break;
        case 3: // Patrones
          attentionValue = analyzePatternLevel(input, output, x, y);
          break;
      }

      // Actualizar mapa de atención
      attention[y][x] = Math.max(attention[y][x], attentionValue);

      // Visualizar barrido
      drawAttentionScan(ctx, x, y, cellSize, attentionValue, layer.color);

      scanStep++;
      animationRef.current = requestAnimationFrame(scan);
    };

    scan();
  };

  // Análisis por niveles
  const analyzePixelLevel = (input, output, x, y) => {
    if (!input[y] || !output[y]) return 0;
    
    const inVal = input[y][x];
    const outVal = output[y][x];
    
    // Mayor atención donde hay cambios
    if (inVal !== outVal) {
      return 1.0;
    }
    
    // Atención media en bordes de cambios
    const neighbors = getNeighbors(input, x, y);
    const hasChangeNearby = neighbors.some(n => {
      const nOut = output[n.y] && output[n.y][n.x];
      return n.val !== nOut;
    });
    
    return hasChangeNearby ? 0.5 : 0.1;
  };

  const analyzeObjectLevel = (input, output, x, y) => {
    // Detectar si el píxel es parte de un objeto
    const val = input[y][x];
    if (val === 0) return 0;
    
    // Calcular tamaño del objeto conectado
    const objectSize = getConnectedComponentSize(input, x, y, val);
    
    // Mayor atención a objetos que cambian de forma
    const outputObject = getConnectedComponentSize(output, x, y, output[y][x]);
    
    if (objectSize !== outputObject) {
      return Math.min(1.0, objectSize / 10);
    }
    
    return objectSize > 1 ? 0.3 : 0.1;
  };

  const analyzeRelationLevel = (input, output, x, y) => {
    // Analizar relaciones espaciales
    const val = input[y][x];
    if (val === 0) return 0;
    
    // Buscar patrones de alineación
    const alignments = checkAlignments(input, x, y);
    const outputAlignments = checkAlignments(output, x, y);
    
    // Mayor atención donde cambian las relaciones
    if (alignments.horizontal !== outputAlignments.horizontal ||
        alignments.vertical !== outputAlignments.vertical) {
      return 0.8;
    }
    
    return alignments.count > 2 ? 0.4 : 0.2;
  };

  const analyzePatternLevel = (input, output, x, y) => {
    // Detectar patrones globales
    const localPattern = extractLocalPattern(input, x, y, 3);
    const outputPattern = extractLocalPattern(output, x, y, 3);
    
    // Calcular similitud de patrones
    const similarity = calculatePatternSimilarity(localPattern, outputPattern);
    
    // Mayor atención donde los patrones difieren
    return 1.0 - similarity;
  };

  // Funciones auxiliares
  const getNeighbors = (grid, x, y) => {
    const neighbors = [];
    const directions = [[-1,0], [1,0], [0,-1], [0,1]];
    
    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      if (ny >= 0 && ny < grid.length && nx >= 0 && nx < grid[0].length) {
        neighbors.push({ x: nx, y: ny, val: grid[ny][nx] });
      }
    }
    
    return neighbors;
  };

  const getConnectedComponentSize = (grid, x, y, value) => {
    const visited = new Set();
    const queue = [[x, y]];
    let size = 0;
    
    while (queue.length > 0) {
      const [cx, cy] = queue.shift();
      const key = `${cx},${cy}`;
      
      if (visited.has(key)) continue;
      if (cy < 0 || cy >= grid.length || cx < 0 || cx >= grid[0].length) continue;
      if (grid[cy][cx] !== value) continue;
      
      visited.add(key);
      size++;
      
      queue.push([cx+1, cy], [cx-1, cy], [cx, cy+1], [cx, cy-1]);
    }
    
    return size;
  };

  const checkAlignments = (grid, x, y) => {
    const val = grid[y][x];
    if (val === 0) return { horizontal: 0, vertical: 0, count: 0 };
    
    let horizontal = 0;
    let vertical = 0;
    
    // Check horizontal
    for (let i = 0; i < grid[y].length; i++) {
      if (i !== x && grid[y][i] === val) horizontal++;
    }
    
    // Check vertical
    for (let j = 0; j < grid.length; j++) {
      if (j !== y && grid[j][x] === val) vertical++;
    }
    
    return { horizontal, vertical, count: horizontal + vertical };
  };

  const extractLocalPattern = (grid, x, y, size) => {
    const pattern = [];
    const half = Math.floor(size / 2);
    
    for (let dy = -half; dy <= half; dy++) {
      const row = [];
      for (let dx = -half; dx <= half; dx++) {
        const ny = y + dy;
        const nx = x + dx;
        if (ny >= 0 && ny < grid.length && nx >= 0 && nx < grid[0].length) {
          row.push(grid[ny][nx]);
        } else {
          row.push(-1);
        }
      }
      pattern.push(row);
    }
    
    return pattern;
  };

  const calculatePatternSimilarity = (pattern1, pattern2) => {
    if (!pattern1 || !pattern2) return 0;
    
    let matches = 0;
    let total = 0;
    
    for (let i = 0; i < pattern1.length; i++) {
      for (let j = 0; j < pattern1[i].length; j++) {
        if (pattern1[i][j] !== -1 && pattern2[i] && pattern2[i][j] !== -1) {
          total++;
          if (pattern1[i][j] === pattern2[i][j]) {
            matches++;
          }
        }
      }
    }
    
    return total > 0 ? matches / total : 0;
  };

  const drawAttentionScan = (ctx, x, y, cellSize, attention, color) => {
    // Dibujar punto de atención
    ctx.fillStyle = color;
    ctx.globalAlpha = attention;
    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
    
    // Dibujar onda de barrido
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x * cellSize + cellSize/2, y * cellSize + cellSize/2, 
            cellSize * attention * 2, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.globalAlpha = 1;
  };

  const analyzeAttentionResults = (attentionMap) => {
    // Encontrar puntos de alta atención
    const hotspots = [];
    
    for (let y = 0; y < attentionMap.length; y++) {
      for (let x = 0; x < attentionMap[y].length; x++) {
        if (attentionMap[y][x] > 0.7) {
          hotspots.push({ x, y, attention: attentionMap[y][x] });
        }
      }
    }
    
    // Detectar patrones en los hotspots
    const patterns = detectPatternsInHotspots(hotspots);
    setDiscoveredPatterns(patterns);
    
    // Notificar al componente padre
    if (onAttentionUpdate) {
      onAttentionUpdate({
        attentionMap,
        hotspots,
        patterns
      });
    }
  };

  const detectPatternsInHotspots = (hotspots) => {
    const patterns = [];
    
    // Detectar líneas
    const lines = detectLines(hotspots);
    if (lines.length > 0) {
      patterns.push({ type: 'lines', data: lines });
    }
    
    // Detectar clusters
    const clusters = detectClusters(hotspots);
    if (clusters.length > 0) {
      patterns.push({ type: 'clusters', data: clusters });
    }
    
    // Detectar simetrías
    const symmetries = detectSymmetries(hotspots);
    if (symmetries.length > 0) {
      patterns.push({ type: 'symmetries', data: symmetries });
    }
    
    return patterns;
  };

  const detectLines = (hotspots) => {
    // Simplificado: detectar alineaciones horizontales y verticales
    const lines = [];
    
    // Agrupar por filas
    const rows = {};
    hotspots.forEach(h => {
      if (!rows[h.y]) rows[h.y] = [];
      rows[h.y].push(h.x);
    });
    
    // Buscar líneas horizontales
    Object.entries(rows).forEach(([y, xs]) => {
      if (xs.length >= 3) {
        lines.push({ type: 'horizontal', y: parseInt(y), points: xs });
      }
    });
    
    return lines;
  };

  const detectClusters = (hotspots) => {
    // Algoritmo simple de clustering por proximidad
    const clusters = [];
    const used = new Set();
    
    hotspots.forEach((h, i) => {
      if (used.has(i)) return;
      
      const cluster = [h];
      used.add(i);
      
      // Buscar puntos cercanos
      hotspots.forEach((h2, j) => {
        if (used.has(j)) return;
        
        const dist = Math.abs(h.x - h2.x) + Math.abs(h.y - h2.y);
        if (dist <= 2) {
          cluster.push(h2);
          used.add(j);
        }
      });
      
      if (cluster.length >= 2) {
        clusters.push(cluster);
      }
    });
    
    return clusters;
  };

  const detectSymmetries = (hotspots) => {
    // Detectar simetrías simples
    const symmetries = [];
    
    // Calcular centro de masa
    const centerX = hotspots.reduce((sum, h) => sum + h.x, 0) / hotspots.length;
    const centerY = hotspots.reduce((sum, h) => sum + h.y, 0) / hotspots.length;
    
    // Verificar simetría horizontal
    let horizontalSymmetry = true;
    for (const h of hotspots) {
      const mirrorY = centerY * 2 - h.y;
      const hasMirror = hotspots.some(h2 => 
        Math.abs(h2.y - mirrorY) < 0.5 && h2.x === h.x
      );
      if (!hasMirror) {
        horizontalSymmetry = false;
        break;
      }
    }
    
    if (horizontalSymmetry) {
      symmetries.push({ type: 'horizontal', center: centerY });
    }
    
    return symmetries;
  };

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className="attention-visualization">
      <div className="visualization-header">
        <h3>🧠 Sistema de Atención - Barrido Topográfico</h3>
        <button 
          onClick={performTopographicScan}
          disabled={isScanning}
          className="scan-button"
        >
          {isScanning ? '⏸ Escaneando...' : '▶ Iniciar Barrido'}
        </button>
      </div>

      <div className="visualization-content">
        <div className="layers-panel">
          <h4>Capas Jerárquicas</h4>
          {layers.map((layer, idx) => (
            <div 
              key={idx}
              className={`layer-item ${currentLayer === idx ? 'active' : ''}`}
              style={{ borderColor: layer.color }}
            >
              <div className="layer-indicator" style={{ backgroundColor: layer.color }} />
              <span>{layer.name}</span>
              {currentLayer === idx && isScanning && (
                <span className="scanning-indicator">📡</span>
              )}
            </div>
          ))}
        </div>

        <div className="canvas-container">
          <canvas 
            ref={canvasRef}
            width={400}
            height={400}
            className="attention-canvas"
          />
          {isScanning && (
            <div className="scan-position">
              Posición: ({scanPosition.x}, {scanPosition.y})
            </div>
          )}
        </div>

        <div className="patterns-panel">
          <h4>Patrones Descubiertos</h4>
          {discoveredPatterns.length === 0 ? (
            <p className="no-patterns">Esperando análisis...</p>
          ) : (
            <ul className="patterns-list">
              {discoveredPatterns.map((pattern, idx) => (
                <li key={idx} className="pattern-item">
                  <span className="pattern-type">{pattern.type}</span>
                  <span className="pattern-count">
                    {pattern.data.length} encontrados
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="attention-insights">
        <h4>💡 Insights del Barrido</h4>
        {attentionMap && (
          <div className="insights-content">
            <p>• Puntos de alta atención detectados</p>
            <p>• Análisis multinivel completado</p>
            <p>• Patrones emergentes identificados</p>
          </div>
        )}
      </div>
    </div>
  );
};