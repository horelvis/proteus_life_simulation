/**
 * Panel de experimento ARC con backend Python
 */

import React, { useState, useEffect, useRef } from 'react';
import arcWebSocketService from '../services/arcWebSocketService';
import { ARCPuzzleVisualization } from './ARCPuzzleVisualization';
import { ARCReasoningDisplay } from './ARCReasoningDisplay';
import { ARCBenchmarkDisplay } from './ARCBenchmarkDisplay';
import { ARCGameMode } from './ARCGameMode';

export const ARCExperimentPythonPanel = () => {
  const [conectado, setConectado] = useState(false);
  const [cargando, setCargando] = useState(false);
  const [cargandoPuzzles, setCargandoPuzzles] = useState(false);
  const [puzzles, setPuzzles] = useState([]);
  const [puzzleActual, setPuzzleActual] = useState(null);
  const [resultados, setResultados] = useState({});
  const [pasosRazonamiento, setPasosRazonamiento] = useState([]);
  const [mostrarRazonamiento, setMostrarRazonamiento] = useState(false);
  const [estadisticas, setEstadisticas] = useState({
    totalPuzzles: 0,
    resueltos: 0,
    porcentajeExito: 0,
    tiempoPromedio: 0
  });
  const [benchmarks, setBenchmarks] = useState(null);
  const [useOfficial, setUseOfficial] = useState(true);
  const [gameMode, setGameMode] = useState(false);

  const wsClient = useRef(null);

  useEffect(() => {
    // Obtener cliente WebSocket del servicio singleton
    const setupWebSocket = async () => {
      try {
        const client = await arcWebSocketService.getConnection();
        wsClient.current = client;
        
        // Verificar si ya está conectado
        if (arcWebSocketService.isConnected()) {
          setConectado(true);
        }
        
        // Registrar manejadores de eventos
        client.on('connected', () => {
          setConectado(true);
          console.log('✅ Conectado al backend Python');
        });
        
        client.on('disconnected', () => {
          setConectado(false);
          console.log('❌ Desconectado del backend Python');
        });

        client.on('loading', (msg) => {
          console.log('Cargando:', msg.message);
          setCargandoPuzzles(true);
        });

        client.on('puzzles_loaded', (msg) => {
          console.log('Puzzles cargados:', msg.count);
          console.log('Estructura de puzzles:', msg.puzzles);
          if (msg.puzzles && msg.puzzles.length > 0) {
            console.log('Primer puzzle:', msg.puzzles[0]);
            console.log('Train del primer puzzle:', msg.puzzles[0].train);
          }
          setPuzzles(msg.puzzles);
          setEstadisticas(prev => ({ ...prev, totalPuzzles: msg.count }));
          if (msg.benchmarks) {
            setBenchmarks(msg.benchmarks);
          }
          setCargandoPuzzles(false);
        });

        client.on('solving_start', (msg) => {
          console.log('Iniciando resolución:', msg.puzzle_id);
          setCargando(true);
        });

        client.on('analyzing_example', (msg) => {
          console.log('Analizando ejemplo:', msg.example_index);
          // Actualizar UI con progreso
        });

        client.on('rule_detected', (msg) => {
          console.log('Regla detectada:', msg.rule);
          setPasosRazonamiento(prev => [...prev, {
            tipo: 'deteccion_regla',
            titulo: `Regla detectada: ${msg.rule.type}`,
            descripcion: `Confianza: ${(msg.rule.confidence * 100).toFixed(1)}%`,
            datos: msg.rule
          }]);
        });

        client.on('reasoning_step', (msg) => {
          console.log('Paso de razonamiento:', msg.step);
          setPasosRazonamiento(prev => [...prev, msg.step]);
        });

        client.on('solving_complete', (msg) => {
          console.log('Resolución completa:', msg);
          setCargando(false);
          
          // Actualizar resultados
          setResultados(prev => ({
            ...prev,
            [msg.puzzle_id]: {
              correcto: msg.is_correct,
              solucion: msg.solution,
              esperado: msg.expected,
              confianza: msg.confidence,
              pasos: msg.reasoning_steps
            }
          }));

          // Actualizar estadísticas
          if (msg.is_correct) {
            setEstadisticas(prev => ({
              ...prev,
              resueltos: prev.resueltos + 1,
              porcentajeExito: ((prev.resueltos + 1) / prev.totalPuzzles) * 100
            }));
          }
        });

        client.on('integrity_check_complete', (msg) => {
          console.log('Verificación de integridad:', msg);
          alert(`Integridad verificada: ${msg.passed ? '✅ PASÓ' : '❌ FALLÓ'}`);
        });

        client.on('export_ready', (msg) => {
          console.log('Exportación lista:', msg.format);
          // Descargar archivo
          const link = document.createElement('a');
          link.href = `data:image/${msg.format};base64,${msg.data}`;
          link.download = `arc-${msg.puzzle_id}.${msg.format}`;
          link.click();
        });

        client.on('error', (data) => {
          setCargando(false);
          setCargandoPuzzles(false);
          console.error('❌ ERROR:', data.message);
          if (data.traceback) {
            console.error('Traceback:', data.traceback);
          }
          alert(`Error: ${data.message}`);
        });
        
        console.log('✅ Todos los handlers registrados');
        
      } catch (error) {
        console.error('Error configurando WebSocket:', error);
        setConectado(false);
      }
    };
    
    // Llamar a la función async
    setupWebSocket();

    // No desconectar al desmontar ya que usamos singleton
    return () => {
      // El servicio singleton mantiene la conexión
      console.log('Componente desmontado, conexión mantenida por singleton');
    };
  }, []);

  const cargarPuzzles = () => {
    if (!wsClient.current.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    wsClient.current.send('load_puzzles', {
      puzzle_set: 'training',
      count: 10,
      use_official: useOfficial
    });
  };

  const resolverPuzzle = (puzzle) => {
    if (!wsClient.current.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    setPuzzleActual(puzzle);
    setPasosRazonamiento([]);
    wsClient.current.solvePuzzle(puzzle.id, puzzle);
  };

  const resolverTodos = async () => {
    for (const puzzle of puzzles) {
      await new Promise(resolve => {
        // Resolver puzzle y esperar a que termine
        const handler = (msg) => {
          if (msg.type === 'solving_complete' && msg.puzzle_id === puzzle.id) {
            wsClient.current.off('solving_complete', handler);
            resolve();
          }
        };
        wsClient.current.on('solving_complete', handler);
        resolverPuzzle(puzzle);
      });
      
      // Pequeña pausa entre puzzles
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  };

  const verificarIntegridad = () => {
    if (!wsClient.current.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    wsClient.current.verifyIntegrity();
  };

  const exportarVisualizacion = (puzzleId, formato) => {
    if (!wsClient.current.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    wsClient.current.exportVisualization(puzzleId, formato);
  };

  if (gameMode) {
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        color: 'white',
        padding: '20px',
        minHeight: '100vh'
      }}>
        <ARCGameMode 
          webSocketClient={wsClient.current}
          onBack={() => setGameMode(false)}
        />
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      color: 'white',
      padding: '20px',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '10px' }}>
          🐍 Laboratorio ARC con Backend Python
        </h1>
        <p style={{ fontSize: '1.2rem', opacity: 0.8 }}>
          Razonamiento transparente con procesamiento en Python vía WebSocket
        </p>
        
        {/* Estado de conexión */}
        <div style={{
          marginTop: '20px',
          padding: '10px',
          backgroundColor: conectado ? '#2ECC40' : '#FF4136',
          borderRadius: '5px',
          display: 'inline-block'
        }}>
          {conectado ? '✅ Conectado al servidor' : '❌ Desconectado'}
        </div>
      </div>

      {/* Selector de puzzles */}
      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <label style={{
          marginRight: '20px',
          fontSize: '14px',
          color: '#7FDBFF'
        }}>
          <input
            type="checkbox"
            checked={useOfficial}
            onChange={(e) => setUseOfficial(e.target.checked)}
            style={{ marginRight: '5px' }}
          />
          Usar Puzzles Oficiales de ARC Prize
        </label>
      </div>

      {/* Controles principales */}
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <button
          onClick={cargarPuzzles}
          disabled={!conectado || cargando || cargandoPuzzles}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#0074D9',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            margin: '5px',
            cursor: conectado && !cargando && !cargandoPuzzles ? 'pointer' : 'not-allowed',
            opacity: conectado && !cargando && !cargandoPuzzles ? 1 : 0.5
          }}
        >
          {cargandoPuzzles ? '⏳ Cargando...' : '📚 Cargar Puzzles'}
        </button>

        <button
          onClick={resolverTodos}
          disabled={!conectado || cargando || puzzles.length === 0}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#2ECC40',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            margin: '5px',
            cursor: conectado && !cargando && puzzles.length > 0 ? 'pointer' : 'not-allowed',
            opacity: conectado && !cargando && puzzles.length > 0 ? 1 : 0.5
          }}
        >
          🚀 Resolver Todos
        </button>

        <button
          onClick={() => setGameMode(true)}
          disabled={!conectado}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#FF851B',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            margin: '5px',
            cursor: conectado ? 'pointer' : 'not-allowed',
            opacity: conectado ? 1 : 0.5
          }}
        >
          🎮 Modo Juego
        </button>

        <button
          onClick={verificarIntegridad}
          disabled={!conectado || cargando}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#FF851B',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            margin: '5px',
            cursor: conectado && !cargando ? 'pointer' : 'not-allowed',
            opacity: conectado && !cargando ? 1 : 0.5
          }}
        >
          🔍 Verificar Integridad
        </button>
      </div>

      {/* Benchmarks */}
      {benchmarks && (
        <ARCBenchmarkDisplay 
          benchmarks={benchmarks} 
          currentScore={estadisticas.porcentajeExito / 100}
        />
      )}

      {/* Estadísticas */}
      {estadisticas.totalPuzzles > 0 && (
        <div style={{
          backgroundColor: '#2a2a2a',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '30px',
          textAlign: 'center'
        }}>
          <h2 style={{ fontSize: '48px', margin: '10px 0' }}>
            {estadisticas.porcentajeExito.toFixed(1)}%
          </h2>
          <p style={{ fontSize: '18px' }}>
            {estadisticas.resueltos} de {estadisticas.totalPuzzles} puzzles resueltos
          </p>
        </div>
      )}

      {/* Lista de puzzles */}
      {puzzles.length > 0 && (
        <div style={{ marginBottom: '30px' }}>
          <h3>Puzzles Cargados:</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '20px' }}>
            {puzzles.map(puzzle => {
              const resultado = resultados[puzzle.id];
              const estado = resultado ? (resultado.correcto ? '✅' : '❌') : '⏳';
              
              // Renderizar mini grid del primer ejemplo de entrenamiento
              const renderMiniGrid = (grid, maxSize = 10) => {
                if (!grid || !Array.isArray(grid)) {
                  console.log('Grid no válido:', grid);
                  return null;
                }
                
                const height = Math.min(grid.length, maxSize);
                const width = Math.min(grid[0]?.length || 0, maxSize);
                const cellSize = 15;
                
                // Paleta de colores ARC
                const colors = {
                  0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40',
                  4: '#FFDC00', 5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B',
                  8: '#7FDBFF', 9: '#870C25'
                };
                
                return (
                  <div style={{
                    display: 'inline-block',
                    border: '1px solid #444',
                    borderRadius: '3px',
                    padding: '2px',
                    backgroundColor: '#1a1a1a'
                  }}>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${width}, ${cellSize}px)`,
                      gridTemplateRows: `repeat(${height}, ${cellSize}px)`,
                      gap: '1px',
                      backgroundColor: '#333'
                    }}>
                      {grid.slice(0, height).map((row, i) => 
                        row.slice(0, width).map((cell, j) => (
                          <div
                            key={`${i}-${j}`}
                            style={{
                              width: cellSize,
                              height: cellSize,
                              backgroundColor: colors[cell] || '#000',
                              border: '0.5px solid rgba(255,255,255,0.05)'
                            }}
                          />
                        ))
                      )}
                    </div>
                  </div>
                );
              };
              
              return (
                <div
                  key={puzzle.id}
                  style={{
                    backgroundColor: '#2a2a2a',
                    padding: '15px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    border: puzzleActual?.id === puzzle.id ? '2px solid #2ECC40' : '2px solid transparent',
                    transition: 'all 0.3s',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '10px'
                  }}
                  onClick={() => resolverPuzzle(puzzle)}
                >
                  {/* Header con estado y ID */}
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: '1px solid #444',
                    paddingBottom: '10px'
                  }}>
                    <div>
                      <div style={{ fontWeight: 'bold', fontSize: '14px' }}>{puzzle.id}</div>
                      <div style={{ fontSize: '12px', opacity: 0.6, marginTop: '2px' }}>
                        {puzzle.category} • {puzzle.difficulty}
                      </div>
                    </div>
                    <div style={{ fontSize: '24px' }}>{estado}</div>
                  </div>
                  
                  {/* Vista previa del puzzle */}
                  {puzzle.train && puzzle.train.length > 0 && (() => {
                    console.log('Renderizando puzzle:', puzzle.id, 'Train:', puzzle.train[0]);
                    return (
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#888', marginBottom: '5px' }}>
                          Ejemplo 1 de {puzzle.train.length}
                        </div>
                        <div style={{ 
                          display: 'flex', 
                          justifyContent: 'center', 
                          alignItems: 'center',
                          gap: '10px'
                        }}>
                          <div>
                            <div style={{ fontSize: '10px', color: '#7FDBFF', marginBottom: '3px' }}>
                              Input
                            </div>
                            {renderMiniGrid(puzzle.train[0].input)}
                          </div>
                        <div style={{ fontSize: '16px', color: '#666' }}>→</div>
                        <div>
                          <div style={{ fontSize: '10px', color: '#2ECC40', marginBottom: '3px' }}>
                            Output
                          </div>
                          {renderMiniGrid(puzzle.train[0].output)}
                        </div>
                      </div>
                    </div>
                  );
                })()}
                  
                  {/* Test info */}
                  {puzzle.test && puzzle.test.length > 0 && (
                    <div style={{ 
                      fontSize: '11px', 
                      color: '#FF851B',
                      textAlign: 'center',
                      borderTop: '1px solid #444',
                      paddingTop: '8px'
                    }}>
                      Test: {puzzle.test[0].input?.length}×{puzzle.test[0].input?.[0]?.length} grid
                    </div>
                  )}
                  
                  {/* Botones de acción */}
                  <div style={{ 
                    display: 'flex', 
                    gap: '5px',
                    marginTop: '5px'
                  }}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        resolverPuzzle(puzzle);
                      }}
                      style={{
                        flex: 1,
                        padding: '6px 10px',
                        fontSize: '11px',
                        backgroundColor: '#0074D9',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: 'pointer'
                      }}
                    >
                      🎯 Resolver
                    </button>
                    
                    {resultado && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setPuzzleActual(puzzle);
                          setMostrarRazonamiento(true);
                        }}
                        style={{
                          flex: 1,
                          padding: '6px 10px',
                          fontSize: '11px',
                          backgroundColor: '#B10DC9',
                          color: 'white',
                          border: 'none',
                          borderRadius: '3px',
                          cursor: 'pointer'
                        }}
                      >
                        🧠 Ver Detalles
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Visualización mejorada de puzzle y razonamiento */}
      {mostrarRazonamiento && puzzleActual && resultados[puzzleActual.id] && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.95)',
          padding: '20px',
          overflow: 'auto',
          zIndex: 1000
        }}>
          <div style={{
            maxWidth: '1400px',
            margin: '0 auto'
          }}>
            {/* Header con controles */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px',
              backgroundColor: '#1a1a1a',
              padding: '15px',
              borderRadius: '8px'
            }}>
              <h2 style={{ color: '#fff', margin: 0 }}>
                🧩 Análisis Detallado: {puzzleActual.id}
              </h2>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={() => exportarVisualizacion(puzzleActual.id, 'png')}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#F012BE',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  📸 Exportar PNG
                </button>
                <button
                  onClick={() => setMostrarRazonamiento(false)}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#FF4136',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  ✕ Cerrar
                </button>
              </div>
            </div>
            
            {/* Layout de dos columnas */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '20px',
              '@media (max-width: 1200px)': {
                gridTemplateColumns: '1fr'
              }
            }}>
              {/* Columna izquierda: Visualización del puzzle */}
              <div>
                <ARCPuzzleVisualization
                  puzzle={puzzleActual}
                  solution={resultados[puzzleActual.id].solucion}
                  expected={resultados[puzzleActual.id].esperado}
                  showComparison={true}
                  cellSize={25}
                />
              </div>
              
              {/* Columna derecha: Razonamiento */}
              <div>
                <ARCReasoningDisplay
                  reasoning={resultados[puzzleActual.id].pasos || []}
                  confidence={resultados[puzzleActual.id].confianza || 0}
                  isCorrect={resultados[puzzleActual.id].correcto}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Loader para carga de puzzles */}
      {cargandoPuzzles && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1001
        }}>
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '40px',
            borderRadius: '12px',
            textAlign: 'center',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
          }}>
            <div style={{
              fontSize: '48px',
              marginBottom: '20px',
              animation: 'spin 2s linear infinite',
              display: 'inline-block'
            }}>
              🧩
            </div>
            <style>
              {`
                @keyframes spin {
                  from { transform: rotate(0deg); }
                  to { transform: rotate(360deg); }
                }
                @keyframes pulse {
                  0% { opacity: 0.4; }
                  50% { opacity: 1; }
                  100% { opacity: 0.4; }
                }
              `}
            </style>
            <h3 style={{ color: '#7FDBFF', marginBottom: '10px' }}>
              {useOfficial ? 'Cargando Puzzles Oficiales de ARC' : 'Cargando Puzzles de Ejemplo'}
            </h3>
            <div style={{ fontSize: '14px', color: '#888', marginBottom: '20px' }}>
              {useOfficial && 'Descargando desde GitHub oficial...'}
              {!useOfficial && 'Preparando puzzles de demostración...'}
            </div>
            <div style={{
              display: 'flex',
              gap: '5px',
              justifyContent: 'center'
            }}>
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  style={{
                    width: '10px',
                    height: '10px',
                    backgroundColor: '#7FDBFF',
                    borderRadius: '50%',
                    animation: `pulse 1.5s ease-in-out ${i * 0.2}s infinite`
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Estado de carga de resolución */}
      {cargando && (
        <div style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: '#2a2a2a',
          padding: '30px',
          borderRadius: '8px',
          textAlign: 'center',
          zIndex: 999
        }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>⏳ Procesando...</div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>El backend Python está analizando el puzzle</div>
        </div>
      )}
    </div>
  );
};