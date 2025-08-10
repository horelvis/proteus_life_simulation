/**
 * Panel de experimento ARC con backend Python
 */

import React, { useState, useEffect, useRef } from 'react';
import { ARCWebSocketClient } from '../services/ARCWebSocketClient';
import { ARCReasoningVisualization } from './ARCReasoningVisualization';

export const ARCExperimentPythonPanel = () => {
  const [conectado, setConectado] = useState(false);
  const [cargando, setCargando] = useState(false);
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

  const wsClient = useRef(null);

  useEffect(() => {
    // Inicializar cliente WebSocket
    wsClient.current = new ARCWebSocketClient();
    
    // Registrar manejadores de eventos
    wsClient.current.on('connection', (msg) => {
      console.log('Conectado:', msg);
      setConectado(true);
    });

    wsClient.current.on('puzzles_loaded', (msg) => {
      console.log('Puzzles cargados:', msg.count);
      setPuzzles(msg.puzzles);
      setEstadisticas(prev => ({ ...prev, totalPuzzles: msg.count }));
    });

    wsClient.current.on('solving_start', (msg) => {
      console.log('Iniciando resolución:', msg.puzzle_id);
      setCargando(true);
    });

    wsClient.current.on('analyzing_example', (msg) => {
      console.log('Analizando ejemplo:', msg.example_index);
      // Actualizar UI con progreso
    });

    wsClient.current.on('rule_detected', (msg) => {
      console.log('Regla detectada:', msg.rule);
      setPasosRazonamiento(prev => [...prev, {
        tipo: 'deteccion_regla',
        titulo: `Regla detectada: ${msg.rule.type}`,
        descripcion: `Confianza: ${(msg.rule.confidence * 100).toFixed(1)}%`,
        datos: msg.rule
      }]);
    });

    wsClient.current.on('reasoning_step', (msg) => {
      console.log('Paso de razonamiento:', msg.step);
      setPasosRazonamiento(prev => [...prev, msg.step]);
    });

    wsClient.current.on('solving_complete', (msg) => {
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

    wsClient.current.on('integrity_check_complete', (msg) => {
      console.log('Verificación de integridad:', msg);
      alert(`Integridad verificada: ${msg.passed ? '✅ PASÓ' : '❌ FALLÓ'}`);
    });

    wsClient.current.on('export_ready', (msg) => {
      console.log('Exportación lista:', msg.format);
      // Descargar archivo
      const link = document.createElement('a');
      link.href = `data:image/${msg.format};base64,${msg.data}`;
      link.download = `arc-${msg.puzzle_id}.${msg.format}`;
      link.click();
    });

    wsClient.current.on('error', (msg) => {
      console.error('Error del servidor:', msg.message);
      alert(`Error: ${msg.message}`);
    });

    // Conectar al servidor
    conectar();

    return () => {
      if (wsClient.current) {
        wsClient.current.disconnect();
      }
    };
  }, []);

  const conectar = async () => {
    try {
      await wsClient.current.connect();
      console.log('✅ Conectado al backend Python');
    } catch (error) {
      console.error('Error conectando:', error);
      setConectado(false);
    }
  };

  const cargarPuzzles = () => {
    if (!wsClient.current.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    wsClient.current.loadPuzzles('training', 10);
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

      {/* Controles principales */}
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <button
          onClick={cargarPuzzles}
          disabled={!conectado || cargando}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#0074D9',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            margin: '5px',
            cursor: conectado && !cargando ? 'pointer' : 'not-allowed',
            opacity: conectado && !cargando ? 1 : 0.5
          }}
        >
          📚 Cargar Puzzles
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
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
            {puzzles.map(puzzle => {
              const resultado = resultados[puzzle.id];
              const estado = resultado ? (resultado.correcto ? '✅' : '❌') : '⏳';
              
              return (
                <div
                  key={puzzle.id}
                  style={{
                    backgroundColor: '#2a2a2a',
                    padding: '15px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    border: puzzleActual?.id === puzzle.id ? '2px solid #2ECC40' : '2px solid transparent',
                    transition: 'all 0.3s'
                  }}
                  onClick={() => resolverPuzzle(puzzle)}
                >
                  <div style={{ fontSize: '24px', marginBottom: '5px' }}>{estado}</div>
                  <div style={{ fontWeight: 'bold' }}>{puzzle.id}</div>
                  <div style={{ fontSize: '14px', opacity: 0.8 }}>{puzzle.category}</div>
                  <div style={{ fontSize: '12px', opacity: 0.6 }}>{puzzle.difficulty}</div>
                  
                  {resultado && (
                    <div style={{ marginTop: '10px' }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setPuzzleActual(puzzle);
                          setMostrarRazonamiento(true);
                        }}
                        style={{
                          padding: '5px 10px',
                          fontSize: '12px',
                          backgroundColor: '#B10DC9',
                          color: 'white',
                          border: 'none',
                          borderRadius: '3px',
                          cursor: 'pointer'
                        }}
                      >
                        🧠 Ver Razonamiento
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Visualización de razonamiento */}
      {mostrarRazonamiento && puzzleActual && resultados[puzzleActual.id] && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.9)',
          padding: '20px',
          overflow: 'auto',
          zIndex: 1000
        }}>
          <div style={{
            maxWidth: '1200px',
            margin: '0 auto',
            backgroundColor: '#1a1a1a',
            padding: '20px',
            borderRadius: '8px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
              <h2>Razonamiento para {puzzleActual.id}</h2>
              <button
                onClick={() => setMostrarRazonamiento(false)}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#FF4136',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                ✕ Cerrar
              </button>
            </div>
            
            {/* Aquí iría la visualización detallada */}
            <div style={{ marginBottom: '20px' }}>
              <h3>Pasos de Razonamiento:</h3>
              {resultados[puzzleActual.id].pasos.map((paso, idx) => (
                <div key={idx} style={{
                  backgroundColor: '#2a2a2a',
                  padding: '15px',
                  marginBottom: '10px',
                  borderRadius: '5px'
                }}>
                  <div style={{ fontWeight: 'bold' }}>{paso.description}</div>
                  {paso.details && <div style={{ fontSize: '14px', opacity: 0.8 }}>{paso.details}</div>}
                </div>
              ))}
            </div>
            
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={() => exportarVisualizacion(puzzleActual.id, 'png')}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#F012BE',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                📸 Exportar PNG
              </button>
              
              <button
                onClick={() => exportarVisualizacion(puzzleActual.id, 'gif')}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#7FDBFF',
                  color: 'black',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer'
                }}
              >
                🎬 Exportar GIF
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Estado de carga */}
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