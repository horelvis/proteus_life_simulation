/**
 * Visualización paso a paso del razonamiento en ARC
 * Muestra cómo el sistema analiza y resuelve puzzles
 */

import React, { useState, useEffect } from 'react';
import { GifGenerator } from '../utils/GifGenerator';

export const ARCReasoningVisualization = ({ puzzle, solver, onComplete }) => {
  const [pasoActual, setPasoActual] = useState(0);
  const [pasos, setPasos] = useState([]);
  const [animando, setAnimando] = useState(false);
  
  useEffect(() => {
    if (puzzle && solver) {
      // Primero hacer que el solver aprenda de los ejemplos
      if (puzzle.trainExamples) {
        solver.aprender(puzzle.trainExamples);
      }
      generarPasosRazonamiento();
    }
  }, [puzzle]);
  
  const generarPasosRazonamiento = () => {
    const nuevosPasos = [];
    
    // Paso 1: Mostrar ejemplos de entrenamiento
    puzzle.trainExamples?.forEach((ejemplo, idx) => {
      nuevosPasos.push({
        tipo: 'analisis_ejemplo',
        titulo: `📚 Analizando Ejemplo ${idx + 1}`,
        descripcion: 'Buscando patrones en la transformación',
        datos: {
          input: ejemplo.input,
          output: ejemplo.output
        }
      });
    });
    
    // Paso 2: Obtener reglas del solver
    if (solver.transformationRules && solver.transformationRules.length > 0) {
      solver.transformationRules.forEach((regla, idx) => {
        nuevosPasos.push({
          tipo: 'deteccion_regla',
          titulo: `🔍 Regla Detectada: ${regla.tipo}`,
          descripcion: solver.obtenerDescripcionRegla ? 
            solver.obtenerDescripcionRegla(regla.tipo) : 
            explicarRegla(regla.tipo),
          datos: {
            tipo: regla.tipo,
            regla: regla
          }
        });
      });
    }
    
    // Paso 3: Aplicar transformación
    try {
      const solucion = solver.resolver(puzzle.input);
      
      // Paso 4: Resultado final
      nuevosPasos.push({
        tipo: 'resultado',
        titulo: '✅ Solución Generada',
        descripcion: 'Transformación completa aplicada',
        datos: {
          input: puzzle.input,
          output: solucion,
          esperado: puzzle.output
        }
      });
    } catch (error) {
      console.error('Error generando solución:', error);
      nuevosPasos.push({
        tipo: 'error',
        titulo: '❌ Error',
        descripcion: 'No se pudo generar una solución',
        datos: {}
      });
    }
    
    setPasos(nuevosPasos);
  };
  
  const explicarRegla = (tipo) => {
    const explicaciones = {
      'color_mapping': 'Cada color se transforma en otro color específico',
      'pattern_replication': 'El patrón se replica múltiples veces',
      'reflection': 'La imagen se refleja como en un espejo',
      'rotation': 'La imagen se rota 90 grados',
      'counting': 'Se cuenta la cantidad de elementos no vacíos',
      'gravity': 'Los elementos caen hacia abajo como por gravedad',
      'symmetry_detection': 'Se detecta si el patrón es simétrico',
      'fill_shape': 'Se rellenan las formas cerradas',
      'pattern_extraction': 'Se extrae un subpatrón específico',
      'line_drawing': 'Se dibujan líneas entre puntos'
    };
    return explicaciones[tipo] || 'Transformación compleja detectada';
  };
  
  const renderGrid = (grid, cambios = []) => {
    if (!grid) return null;
    
    const colores = ['#000', '#0074D9', '#2ECC40', '#FFDC00', '#FF4136', '#B10DC9', '#FF851B', '#7FDBFF', '#85144b', '#F012BE'];
    
    return (
      <div style={{
        display: 'inline-block',
        border: '2px solid #444',
        backgroundColor: '#111',
        padding: '2px'
      }}>
        {grid.map((fila, i) => (
          <div key={i} style={{ display: 'flex' }}>
            {fila.map((celda, j) => {
              const tieneCambio = cambios.some(c => c.x === i && c.y === j);
              return (
                <div
                  key={j}
                  style={{
                    width: '20px',
                    height: '20px',
                    backgroundColor: colores[celda] || '#000',
                    border: '1px solid #333',
                    transition: 'all 0.3s',
                    transform: tieneCambio ? 'scale(1.2)' : 'scale(1)',
                    boxShadow: tieneCambio ? '0 0 10px #FF851B' : 'none'
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
    );
  };
  
  const renderPaso = (paso) => {
    switch (paso.tipo) {
      case 'analisis_ejemplo':
        return (
          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: '12px', marginBottom: '5px' }}>Input:</div>
              {renderGrid(paso.datos.input)}
            </div>
            <div style={{ fontSize: '24px' }}>→</div>
            <div>
              <div style={{ fontSize: '12px', marginBottom: '5px' }}>Output:</div>
              {renderGrid(paso.datos.output)}
            </div>
          </div>
        );
        
      case 'deteccion_regla':
        return (
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '15px',
            borderRadius: '8px',
            border: '2px solid #2ECC40'
          }}>
            <div style={{ fontSize: '18px', marginBottom: '10px' }}>
              Tipo: <strong>{paso.datos.tipo}</strong>
            </div>
            {paso.datos.parametros && (
              <pre style={{ fontSize: '12px', opacity: 0.8 }}>
                {JSON.stringify(paso.datos.parametros, null, 2)}
              </pre>
            )}
          </div>
        );
        
      case 'aplicacion':
        return (
          <div>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '12px', marginBottom: '5px' }}>Antes:</div>
                {renderGrid(paso.datos.gridAntes)}
              </div>
              <div style={{ fontSize: '24px' }}>→</div>
              <div>
                <div style={{ fontSize: '12px', marginBottom: '5px' }}>Después:</div>
                {renderGrid(paso.datos.gridDespues, paso.datos.cambios)}
              </div>
            </div>
            {paso.datos.cambios && paso.datos.cambios.length > 0 && (
              <div style={{ marginTop: '10px', fontSize: '12px', opacity: 0.8 }}>
                Cambios: {paso.datos.cambios.length} celdas modificadas
              </div>
            )}
          </div>
        );
        
      case 'resultado':
        const correcto = JSON.stringify(paso.datos.output) === JSON.stringify(paso.datos.esperado);
        return (
          <div>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '12px', marginBottom: '5px' }}>Input Original:</div>
                {renderGrid(paso.datos.input)}
              </div>
              <div style={{ fontSize: '24px' }}>→</div>
              <div>
                <div style={{ fontSize: '12px', marginBottom: '5px' }}>Solución Generada:</div>
                {renderGrid(paso.datos.output)}
              </div>
              {paso.datos.esperado && (
                <>
                  <div style={{ fontSize: '24px' }}>=</div>
                  <div>
                    <div style={{ fontSize: '12px', marginBottom: '5px' }}>Esperado:</div>
                    {renderGrid(paso.datos.esperado)}
                  </div>
                  <div style={{ fontSize: '32px' }}>
                    {correcto ? '✅' : '❌'}
                  </div>
                </>
              )}
            </div>
          </div>
        );
        
      case 'error':
        return (
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '15px',
            borderRadius: '8px',
            border: '2px solid #FF4136'
          }}>
            <div style={{ fontSize: '18px', color: '#FF4136' }}>
              {paso.descripcion}
            </div>
          </div>
        );
        
      default:
        return (
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '15px',
            borderRadius: '8px'
          }}>
            <div>{paso.descripcion || 'Paso de procesamiento'}</div>
          </div>
        );
    }
  };
  
  const animarPasos = async () => {
    setAnimando(true);
    setPasoActual(0);
    
    for (let i = 0; i < pasos.length; i++) {
      setPasoActual(i);
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
    
    setAnimando(false);
    if (onComplete) onComplete();
  };
  
  if (pasos.length === 0) return null;
  
  const paso = pasos[pasoActual];
  
  // Verificar que el paso existe
  if (!paso) {
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        border: '1px solid #444',
        borderRadius: '8px',
        padding: '20px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <p>No hay pasos de razonamiento disponibles</p>
      </div>
    );
  }
  
  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      border: '1px solid #444',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    }}>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ margin: '0 0 10px 0' }}>{paso.titulo || 'Paso de Razonamiento'}</h3>
        <p style={{ margin: '0', opacity: 0.8, fontSize: '14px' }}>{paso.descripcion || ''}</p>
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        {renderPaso(paso)}
      </div>
      
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        <button
          onClick={() => setPasoActual(Math.max(0, pasoActual - 1))}
          disabled={pasoActual === 0 || animando}
          style={{
            padding: '8px 16px',
            backgroundColor: pasoActual === 0 || animando ? '#444' : '#2ECC40',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: pasoActual === 0 || animando ? 'not-allowed' : 'pointer'
          }}
        >
          ← Anterior
        </button>
        
        <div style={{ flex: 1, textAlign: 'center' }}>
          Paso {pasoActual + 1} de {pasos.length}
        </div>
        
        <button
          onClick={() => setPasoActual(Math.min(pasos.length - 1, pasoActual + 1))}
          disabled={pasoActual === pasos.length - 1 || animando}
          style={{
            padding: '8px 16px',
            backgroundColor: pasoActual === pasos.length - 1 || animando ? '#444' : '#2ECC40',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: pasoActual === pasos.length - 1 || animando ? 'not-allowed' : 'pointer'
          }}
        >
          Siguiente →
        </button>
        
        <button
          onClick={animarPasos}
          disabled={animando}
          style={{
            padding: '8px 16px',
            backgroundColor: animando ? '#FF851B' : '#0074D9',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: animando ? 'not-allowed' : 'pointer'
          }}
        >
          {animando ? '▶ Reproduciendo...' : '▶ Animar'}
        </button>
        
        <button
          onClick={async () => {
            const generator = new GifGenerator();
            const diagrama = generator.generarDiagramaFlujo([
              { tipo: paso.tipo, descripcion: paso.descripcion }
            ]);
            generator.exportarComoPNG(`arc-${puzzle.id}-paso-${pasoActual + 1}.png`);
          }}
          style={{
            padding: '8px 16px',
            backgroundColor: '#F012BE',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          📸 Exportar
        </button>
      </div>
      
      <div style={{ marginTop: '20px' }}>
        <div style={{
          display: 'flex',
          gap: '2px',
          height: '4px',
          backgroundColor: '#333',
          borderRadius: '2px'
        }}>
          {pasos.map((_, idx) => (
            <div
              key={idx}
              style={{
                flex: 1,
                backgroundColor: idx <= pasoActual ? '#2ECC40' : '#444',
                transition: 'background-color 0.3s'
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};