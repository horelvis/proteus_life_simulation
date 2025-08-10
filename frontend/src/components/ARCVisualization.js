/**
 * Visualización del proceso de razonamiento en puzzles ARC
 */

import React, { useRef, useEffect, useState } from 'react';

export const ARCVisualization = ({ experimento, organismo }) => {
  const canvasRef = useRef(null);
  const [procesoMental, setProcesoMental] = useState([]);
  const [hipótesisActual, setHipótesisActual] = useState(null);
  
  // Colores ARC estándar
  const coloresARC = [
    '#000000', // 0: Negro
    '#0074D9', // 1: Azul
    '#FF4136', // 2: Rojo
    '#2ECC40', // 3: Verde
    '#FFDC00', // 4: Amarillo
    '#AAAAAA', // 5: Gris
    '#F012BE', // 6: Magenta
    '#FF851B', // 7: Naranja
    '#7FDBFF', // 8: Azul claro
    '#870C25'  // 9: Marrón
  ];
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !experimento) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 1400;
    canvas.height = 800;
    
    let animationId;
    
    const render = () => {
      ctx.fillStyle = '#1a1a1a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Dibujar grids ARC
      if (experimento.puzzleActual) {
        // Si tiene ejemplos de entrenamiento, mostrarlos primero
        if (experimento.puzzleActual.trainExamples) {
          ctx.fillStyle = 'white';
          ctx.font = '12px monospace';
          ctx.fillText('Ejemplos de entrenamiento:', 50, 30);
          
          let xOffset = 50;
          experimento.puzzleActual.trainExamples.slice(0, 2).forEach((example, idx) => {
            drawARCGrid(ctx, xOffset, 50, example.input, `Ej ${idx+1} Input`, 15);
            drawARCGrid(ctx, xOffset + 120, 50, example.output, `→ Output`, 15);
            xOffset += 270;
          });
          
          // Test case
          ctx.fillText('Test:', 50, 250);
          drawARCGrid(ctx, 50, 270, experimento.puzzleActual.input, 'Test Input');
          drawARCGrid(ctx, 350, 270, experimento.gridActual, 'Tu solución');
          drawARCGrid(ctx, 650, 270, experimento.puzzleActual.output, 'Esperado');
        } else {
          // Formato simple si no hay ejemplos
          drawARCGrid(ctx, 50, 50, experimento.puzzleActual.input, 'Input');
          drawARCGrid(ctx, 350, 50, experimento.gridActual, 'Estado Actual');
          drawARCGrid(ctx, 650, 50, experimento.puzzleActual.output, 'Output Esperado');
        }
      }
      
      // Visualizar proceso mental del organismo
      if (organismo) {
        drawProcesoMental(ctx, 50, 350, organismo);
      }
      
      // Mostrar hipótesis actual
      if (hipótesisActual) {
        drawHipótesis(ctx, 650, 350, hipótesisActual);
      }
      
      // Métricas en tiempo real
      drawMétricas(ctx, 950, 50, experimento);
      
      animationId = requestAnimationFrame(render);
    };
    
    const drawARCGrid = (ctx, x, y, grid, título, cellSizeOverride) => {
      const cellSize = cellSizeOverride || 20;
      const padding = cellSizeOverride ? 2 : 5;
      
      // Título
      ctx.fillStyle = 'white';
      ctx.font = '14px monospace';
      ctx.fillText(título, x, y - 10);
      
      // Grid
      grid.forEach((row, i) => {
        row.forEach((cell, j) => {
          ctx.fillStyle = coloresARC[cell] || '#000000';
          ctx.fillRect(
            x + j * (cellSize + padding),
            y + i * (cellSize + padding),
            cellSize,
            cellSize
          );
          
          // Borde
          ctx.strokeStyle = '#333';
          ctx.strokeRect(
            x + j * (cellSize + padding),
            y + i * (cellSize + padding),
            cellSize,
            cellSize
          );
        });
      });
    };
    
    const drawProcesoMental = (ctx, x, y, org) => {
      ctx.fillStyle = 'white';
      ctx.font = '14px monospace';
      ctx.fillText('Proceso Mental (CA)', x, y - 10);
      
      // Visualizar el estado del CA
      const ca = org.cellularAutomaton;
      const cellSize = 10;
      
      ca.grid.forEach((row, i) => {
        row.forEach((cell, j) => {
          // Color basado en activación y tipo
          let color;
          if (cell.type === 'void') {
            color = '#222';
          } else {
            const activation = cell.activation || 0;
            const hue = activation * 120; // Rojo a verde
            const brightness = 30 + cell.energy * 40;
            color = `hsl(${hue}, 70%, ${brightness}%)`;
          }
          
          ctx.fillStyle = color;
          ctx.fillRect(
            x + j * cellSize,
            y + i * cellSize,
            cellSize - 1,
            cellSize - 1
          );
        });
      });
      
      // Mostrar zonas de alta activación (hipótesis)
      ctx.strokeStyle = 'yellow';
      ctx.lineWidth = 2;
      ca.grid.forEach((row, i) => {
        row.forEach((cell, j) => {
          if (cell.activation > 0.7) {
            ctx.strokeRect(
              x + j * cellSize - 2,
              y + i * cellSize - 2,
              cellSize + 3,
              cellSize + 3
            );
          }
        });
      });
    };
    
    const drawHipótesis = (ctx, x, y, hipótesis) => {
      ctx.fillStyle = 'white';
      ctx.font = '14px monospace';
      ctx.fillText('Hipótesis Detectada', x, y - 10);
      
      // Visualizar patrón detectado
      ctx.fillStyle = '#333';
      ctx.fillRect(x, y, 200, 100);
      
      ctx.fillStyle = hipótesis.tieneHipótesis ? '#2ECC40' : '#FF4136';
      ctx.font = '12px monospace';
      ctx.fillText(
        `Confianza: ${(hipótesis.confianza * 100).toFixed(0)}%`,
        x + 10,
        y + 20
      );
      
      if (hipótesis.patrón) {
        ctx.fillText(
          `Patrón: ${hipótesis.patrón.length} células activas`,
          x + 10,
          y + 40
        );
      }
    };
    
    const drawMétricas = (ctx, x, y, exp) => {
      ctx.fillStyle = 'white';
      ctx.font = '14px monospace';
      ctx.fillText('Métricas', x, y - 10);
      
      const métricas = [
        `Intentos: ${exp.intentos || 0}`,
        `Tiempo: ${((exp.tiempoTranscurrido || 0) / 1000).toFixed(1)}s`,
        `Generación: ${exp.generaciónActual || 0}`,
        '',
        'Evidencia de Razonamiento:',
        `- Exploración: ${exp.exploración || 'N/A'}`,
        `- Correcciones: ${exp.correcciones || 0}`,
        `- Estrategias: ${exp.estrategiasÚnicas || 0}`,
        '',
        exp.resuelto ? '✅ RESUELTO!' : '🔄 En proceso...'
      ];
      
      métricas.forEach((métrica, i) => {
        ctx.fillText(métrica, x, y + i * 20);
      });
    };
    
    render();
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [experimento, organismo, hipótesisActual]);
  
  // Actualizar proceso mental periódicamente
  useEffect(() => {
    if (!organismo) return;
    
    const interval = setInterval(() => {
      // Capturar estado mental actual
      const nuevoEstado = {
        timestamp: Date.now(),
        activación: calcularActivaciónTotal(organismo),
        zonasFoco: detectarZonasFoco(organismo),
        movimientoPlanificado: organismo.cellularAutomaton.effectors
      };
      
      setProcesoMental(prev => [...prev.slice(-50), nuevoEstado]);
      
      // Detectar hipótesis
      if (experimento && experimento.validador) {
        const hip = experimento.validador.detectarHipótesis(
          organismo, 
          { posición: organismo.position }
        );
        setHipótesisActual(hip);
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, [organismo, experimento]);
  
  const calcularActivaciónTotal = (org) => {
    let total = 0;
    const grid = org.cellularAutomaton.grid;
    
    grid.forEach(row => {
      row.forEach(cell => {
        if (cell.type !== 'void') {
          total += cell.activation || 0;
        }
      });
    });
    
    return total;
  };
  
  const detectarZonasFoco = (org) => {
    const zonas = [];
    const grid = org.cellularAutomaton.grid;
    const umbral = 0.6;
    
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j].activation > umbral) {
          zonas.push({ x: i, y: j, intensidad: grid[i][j].activation });
        }
      }
    }
    
    return zonas;
  };
  
  return (
    <div className="arc-visualization">
      <canvas
        ref={canvasRef}
        style={{
          border: '1px solid #333',
          backgroundColor: '#1a1a1a',
          width: '100%',
          maxWidth: '1200px'
        }}
      />
      
      <div className="controls" style={{ marginTop: '10px', color: 'white' }}>
        <div style={{ display: 'flex', gap: '20px', fontSize: '12px' }}>
          <div>
            <strong>Leyenda Colores ARC:</strong>
            {coloresARC.map((color, i) => (
              <span key={i} style={{ marginLeft: '10px' }}>
                <span style={{ 
                  backgroundColor: color, 
                  width: '15px', 
                  height: '15px', 
                  display: 'inline-block',
                  border: '1px solid white'
                }}></span> {i}
              </span>
            ))}
          </div>
        </div>
        
        {procesoMental.length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <strong>Activación Mental:</strong>
            <div style={{ 
              display: 'inline-block', 
              marginLeft: '10px',
              width: '200px',
              height: '20px',
              backgroundColor: '#333',
              position: 'relative'
            }}>
              <div style={{
                width: `${Math.min(100, procesoMental[procesoMental.length - 1].activación * 10)}%`,
                height: '100%',
                backgroundColor: '#2ECC40',
                transition: 'width 0.3s'
              }}></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};