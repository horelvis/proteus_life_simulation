import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import arcWebSocketService from './services/arcWebSocketService';

const Container = styled.div`
  min-height: 100vh;
  background: #0a0a0a;
  color: #fff;
  padding: 20px;
  font-family: 'Courier New', monospace;
`;

const Header = styled.div`
  background: #1a1a1a;
  border: 1px solid #333;
  padding: 20px;
  margin-bottom: 20px;
  border-radius: 8px;
`;

const Title = styled.h1`
  color: #ff6b6b;
  margin: 0 0 10px 0;
  font-size: 24px;
`;

const Warning = styled.div`
  background: #2a1515;
  border: 1px solid #ff3333;
  padding: 15px;
  margin: 10px 0;
  border-radius: 4px;
  font-size: 14px;
  color: #ff9999;
`;

const MainGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 20px;
  margin-bottom: 20px;
`;

const Panel = styled.div`
  background: #1a1a1a;
  border: 1px solid #333;
  padding: 15px;
  border-radius: 8px;
`;

const PanelTitle = styled.h2`
  color: #7FDBFF;
  font-size: 16px;
  margin: 0 0 15px 0;
  padding-bottom: 10px;
  border-bottom: 1px solid #333;
`;

const PuzzleSelector = styled.div`
  margin-bottom: 20px;
`;

const Select = styled.select`
  width: 100%;
  padding: 8px;
  background: #2a2a2a;
  border: 1px solid #444;
  color: #fff;
  border-radius: 4px;
  font-family: inherit;
`;

const Button = styled.button`
  padding: 10px 20px;
  background: ${props => props.danger ? '#ff3333' : '#0074D9'};
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  margin-right: 10px;
  margin-top: 10px;
  
  &:hover {
    opacity: 0.8;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const GridDisplay = styled.div`
  display: inline-block;
  border: 1px solid #444;
  padding: 5px;
  margin: 10px 10px 10px 0;
  background: #000;
`;

const GridRow = styled.div`
  display: flex;
`;

const Cell = styled.div`
  width: 20px;
  height: 20px;
  background-color: ${props => props.color};
  border: 0.5px solid #222;
`;

const ProcessLog = styled.div`
  background: #0a0a0a;
  border: 1px solid #333;
  padding: 10px;
  border-radius: 4px;
  max-height: 600px;
  overflow-y: auto;
  font-size: 12px;
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
`;

const LogEntry = styled.div`
  padding: 5px 0;
  border-bottom: 1px solid #222;
  color: ${props => {
    if (props.type === 'error') return '#ff6666';
    if (props.type === 'warning') return '#ffcc00';
    if (props.type === 'success') return '#66ff66';
    if (props.type === 'hierarchical') return '#66ffff';
    if (props.type === 'vjepa') return '#ff99ff';
    if (props.type === 'attention') return '#99ccff';
    if (props.type === 'memory') return '#ffff99';
    return '#cccccc';
  }};
`;

const ModuleStatus = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 15px;
`;

const ModuleRow = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
`;

const ExecutionOrder = styled.div`
  background: #2a2a2a;
  color: #7FDBFF;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
`;

const Module = styled.div`
  background: ${props => props.$active ? '#1a2a1a' : '#1a1a1a'};
  border: 1px solid ${props => props.$active ? '#66ff66' : '#333'};
  padding: 10px;
  border-radius: 4px;
  flex: 1;
`;

const ModuleName = styled.div`
  font-weight: bold;
  color: ${props => props.$active ? '#66ff66' : '#888'};
  margin-bottom: 5px;
`;

const ModuleOutput = styled.div`
  font-size: 11px;
  color: #aaa;
  max-height: 60px;
  overflow-y: auto;
`;

const RawDataDisplay = styled.pre`
  background: #0a0a0a;
  border: 1px solid #333;
  padding: 10px;
  border-radius: 4px;
  font-size: 11px;
  color: #888;
  max-height: 200px;
  overflow: auto;
  margin-top: 10px;
`;

// Colores ARC oficiales
const ARC_COLORS = {
  0: '#000000', // Negro
  1: '#0074D9', // Azul
  2: '#FF4136', // Rojo
  3: '#2ECC40', // Verde
  4: '#FFDC00', // Amarillo
  5: '#AAAAAA', // Gris
  6: '#F012BE', // Magenta
  7: '#FF851B', // Naranja
  8: '#7FDBFF', // Cyan
  9: '#870C25', // Marrón
};

function AppARCReal() {
  const [wsClient, setWsClient] = useState(null);
  const [connected, setConnected] = useState(false);
  const [puzzles, setPuzzles] = useState([]);
  const [selectedPuzzle, setSelectedPuzzle] = useState(null);
  const [currentPuzzle, setCurrentPuzzle] = useState(null);
  const [processLog, setProcessLog] = useState([]);
  const [moduleStates, setModuleStates] = useState({
    hierarchical: { active: false, output: 'No iniciado' },
    vjepa: { active: false, output: 'No iniciado' },
    attention: { active: false, output: 'No iniciado' },
    memory: { active: false, output: 'No iniciado' },
    reasoning: { active: false, output: 'No iniciado' }
  });
  const [processing, setProcessing] = useState(false);
  const [showRawData, setShowRawData] = useState(false);
  const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
  const [viewMode, setViewMode] = useState('train'); // 'train' o 'test'
  const logEndRef = useRef(null);

  useEffect(() => {
    // Obtener cliente WebSocket del servicio singleton
    const setupWebSocket = async () => {
      try {
        const client = await arcWebSocketService.getConnection();
        setWsClient(client);
        
        // Verificar si ya está conectado
        if (arcWebSocketService.isConnected()) {
          setConnected(true);
          addLog('✅ Conectado al backend', 'success');
          // Cargar lista de puzzles si ya está conectado
          client.send('list_puzzles');
        }
        
        client.on('connected', () => {
          setConnected(true);
          addLog('✅ Conectado al backend', 'success');
          // Cargar lista de puzzles al conectar
          client.send('list_puzzles');
        });
        
        client.on('disconnected', () => {
          setConnected(false);
          addLog('❌ Desconectado del backend', 'error');
        });
        
        client.on('puzzles_list', (data) => {
          setPuzzles(data.puzzles || []);
          addLog(`📂 ${data.puzzles?.length || 0} puzzles disponibles`, 'info');
        });
        
        client.on('puzzle_loaded', (data) => {
          setCurrentPuzzle(data.puzzle);
          setCurrentExampleIndex(0); // Reset al primer ejemplo
          setViewMode('train'); // Reset a modo entrenamiento
          addLog(`✅ Puzzle cargado: ${data.puzzle.id}`, 'success');
          
          // Debug: verificar qué visualización llegó
          if (data.puzzle.visualization) {
            console.log('Visualización recibida:', data.puzzle.visualization);
            const trainCount = data.puzzle.visualization.train_examples?.length || 0;
            const testCount = data.puzzle.visualization.test_examples?.length || 0;
            addLog(`🖼️ Imágenes generadas: ${trainCount} entrenamiento, ${testCount} test`, 'success');
          } else {
            console.log('No hay visualización en el puzzle:', data.puzzle);
            addLog(`⚠️ No se recibieron imágenes del puzzle`, 'warning');
          }
        });
        
        // Handlers para procesamiento real
        client.on('hierarchical_update', (data) => {
          setModuleStates(prev => ({
            ...prev,
            hierarchical: { 
              active: true, 
              output: data.patterns_found ? 
                `${data.patterns_found.meso}, ${data.patterns_found.micro}` :
                data.message || 'Procesando...'
            }
          }));
          addLog(`[JERÁRQUICO] ${data.message || 'Análisis de capas'}`, 'hierarchical');
          if (data.input_summary) {
            addLog(`[JERÁRQUICO] ${data.input_summary}`, 'hierarchical');
          }
          if (data.output_summary) {
            addLog(`[JERÁRQUICO] ${data.output_summary}`, 'hierarchical');
          }
          if (data.shapes_detected && data.shapes_detected.length > 0) {
            addLog(`[JERÁRQUICO] Formas detectadas: ${data.shapes_detected.join(', ')}`, 'hierarchical');
          }
        });
        
        client.on('vjepa_observation', (data) => {
      let outputText = data.message || 'Procesando...';
      
      if (data.observation) {
        const obs = data.observation;
        
        if (obs.arc_compliant) {
          // V-JEPA ARC compliant
          outputText = `✅ ARC: ${obs.examples_learned} ejemplos | ${obs.pattern_detected} (${(obs.confidence * 100).toFixed(0)}% conf)`;
          
          // Si hay transformaciones detectadas
          if (obs.transformations && obs.transformations.length > 0) {
            const types = obs.transformations.map(t => t.type).filter((v, i, a) => a.indexOf(v) === i);
            if (types.length > 0) {
              outputText += ` | Tipos: ${types.join(', ')}`;
            }
          }
          
          // Si hay predicción de test
          if (obs.test_prediction && obs.test_prediction.has_prediction) {
            outputText += ` | 🎯 Predicción lista`;
          }
        } else if (data.has_memory) {
          // V-JEPA con memoria persistente (no ARC compliant)
          outputText = `${obs.recognized ? '✅ RECONOCIDO' : '🆕 NUEVO'}: ${obs.transformation_type} (${(obs.confidence * 100).toFixed(0)}% conf)`;
          if (obs.examples_seen > 1) {
            outputText += ` | Visto ${obs.examples_seen} veces`;
          }
          if (obs.memory_size) {
            outputText += ` | Memoria: ${obs.memory_size} patrones`;
          }
        } else {
          // V-JEPA básico - mostrar de forma más compacta
          if (obs.observed) {
            const patternType = obs.emergent_pattern?.type || 'unknown';
            const confidence = obs.emergent_pattern?.confidence || 0;
            outputText = `Observado: ${patternType} (${(confidence * 100).toFixed(0)}% conf)`;
          } else {
            outputText = JSON.stringify(data.observation, null, 2);
          }
        }
      }
      
      setModuleStates(prev => ({
        ...prev,
        vjepa: { 
          active: true, 
          output: outputText
        }
      }));
      
      addLog(`[V-JEPA] ${data.message || 'Observación'}`, 'vjepa');
      
      if (data.memory_stats) {
        addLog(`[V-JEPA MEMORIA] Total: ${data.memory_stats.total_transformations} transformaciones conocidas`, 'vjepa');
      }
        });
        
        client.on('attention_update', (data) => {
      setModuleStates(prev => ({
        ...prev,
        attention: { 
          active: true, 
          output: `Iteración ${data.iteration}: ${data.focus || 'Sin foco'}`
        }
      }));
      addLog(`[ATENCIÓN] Iter ${data.iteration}: ${data.understanding || 'Procesando...'}`, 'attention');
        });
        
        client.on('memory_access', (data) => {
      setModuleStates(prev => ({
        ...prev,
        memory: { 
          active: true, 
          output: `Acceso: ${data.operation || 'read'}, Items: ${data.count || 0}`
        }
      }));
      addLog(`[MEMORIA] ${data.operation}: ${data.result || 'Sin resultados'}`, 'memory');
        });
        
        client.on('reasoning_step', (data) => {
      setModuleStates(prev => ({
        ...prev,
        reasoning: { 
          active: true, 
          output: data.step || 'Procesando...'
        }
      }));
      addLog(`[RAZONAMIENTO] ${data.step}`, 'info');
        });
        
        client.on('processing_complete', (data) => {
      setProcessing(false);
      addLog(`\n=== RESULTADO FINAL ===`, 'success');
      addLog(`Confianza: ${(data.confidence * 100).toFixed(1)}%`, 'success');
      addLog(`Predicción: ${JSON.stringify(data.prediction)}`, 'success');
      addLog(`======================\n`, 'success');
        });
        
        client.on('error', (data) => {
      setProcessing(false);
      addLog(`❌ ERROR: ${data.message}`, 'error');
      if (data.traceback) {
        addLog(`Traceback:\n${data.traceback}`, 'error');
      }
        });
        
        console.log('✅ Todos los handlers registrados');
        
      } catch (error) {
        console.error('Error configurando WebSocket:', error);
        setConnected(false);
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

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setProcessLog(prev => [...prev, { timestamp, message, type }]);
  };

  const scrollToBottom = () => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [processLog]);

  const loadPuzzle = () => {
    if (!selectedPuzzle || !wsClient) return;
    
    addLog(`📂 Cargando puzzle: ${selectedPuzzle}`, 'info');
    wsClient.send('load_puzzle', { puzzle_id: selectedPuzzle });
  };

  const processPuzzle = () => {
    if (!currentPuzzle || !wsClient) return;
    
    setProcessing(true);
    setProcessLog([]); // Limpiar log
    setModuleStates({
      hierarchical: { active: false, output: 'Esperando...' },
      vjepa: { active: false, output: 'Esperando...' },
      attention: { active: false, output: 'Esperando...' },
      memory: { active: false, output: 'Esperando...' },
      reasoning: { active: false, output: 'Esperando...' }
    });
    
    addLog(`🚀 Iniciando procesamiento REAL de ${currentPuzzle.id}`, 'info');
    addLog(`⚠️ SIN SIMULACIÓN - Procesamiento directo del backend`, 'warning');
    
    wsClient.send('process_puzzle_real', { 
      puzzle_id: currentPuzzle.id,
      verbose: true,
      show_raw: true
    });
  };

  const renderGrid = (grid) => {
    if (!grid || !Array.isArray(grid)) return null;
    
    return (
      <GridDisplay>
        {grid.map((row, i) => (
          <GridRow key={i}>
            {row.map((cell, j) => (
              <Cell key={j} color={ARC_COLORS[cell] || '#000'} />
            ))}
          </GridRow>
        ))}
      </GridDisplay>
    );
  };

  return (
    <Container>
      <Header>
        <Title>🔬 PROTEUS ARC - Testing Real (Sin Simulación)</Title>
        <Warning>
          ⚠️ MODO TRANSPARENTE: Todo lo que ves aquí es procesamiento REAL del backend.
          No hay simulación, no hay hardcode. Si algo falla, lo verás inmediatamente.
        </Warning>
        <div style={{ marginTop: '10px' }}>
          Estado: {connected ? '🟢 Conectado' : '🔴 Desconectado'}
        </div>
      </Header>

      <MainGrid>
        <Panel>
          <PanelTitle>Control de Puzzles</PanelTitle>
          
          <PuzzleSelector>
            <Select 
              value={selectedPuzzle || ''} 
              onChange={(e) => setSelectedPuzzle(e.target.value)}
              disabled={!connected}
            >
              <option value="">Selecciona un puzzle...</option>
              {puzzles.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </Select>
            <Button onClick={loadPuzzle} disabled={!selectedPuzzle || !connected}>
              Cargar Puzzle
            </Button>
          </PuzzleSelector>

          {currentPuzzle && (
            <div>
              <h3 style={{ color: '#2ECC40', fontSize: '14px' }}>
                Puzzle Actual: {currentPuzzle.id}
              </h3>
              
              <div style={{ marginTop: '10px' }}>
                {/* Selector de modo Train/Test */}
                <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                  <Button 
                    onClick={() => { setViewMode('train'); setCurrentExampleIndex(0); }}
                    style={{ 
                      background: viewMode === 'train' ? '#2ECC40' : '#444',
                      fontSize: '12px',
                      padding: '5px 10px'
                    }}
                  >
                    Entrenamiento ({currentPuzzle.train?.length || 0})
                  </Button>
                  <Button 
                    onClick={() => { setViewMode('test'); setCurrentExampleIndex(0); }}
                    style={{ 
                      background: viewMode === 'test' ? '#FF851B' : '#444',
                      fontSize: '12px',
                      padding: '5px 10px'
                    }}
                  >
                    Test ({currentPuzzle.test?.length || 0})
                  </Button>
                </div>

                {/* Navegación entre ejemplos */}
                {viewMode === 'train' && currentPuzzle.visualization?.train_examples?.length > 0 && (
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                      <Button 
                        onClick={() => setCurrentExampleIndex(Math.max(0, currentExampleIndex - 1))}
                        disabled={currentExampleIndex === 0}
                        style={{ fontSize: '12px', padding: '5px 10px' }}
                      >
                        ◀
                      </Button>
                      <span style={{ color: '#7FDBFF', fontSize: '12px' }}>
                        Ejemplo {currentExampleIndex + 1} de {currentPuzzle.visualization.train_examples.length}
                      </span>
                      <Button 
                        onClick={() => setCurrentExampleIndex(Math.min(currentPuzzle.visualization.train_examples.length - 1, currentExampleIndex + 1))}
                        disabled={currentExampleIndex >= currentPuzzle.visualization.train_examples.length - 1}
                        style={{ fontSize: '12px', padding: '5px 10px' }}
                      >
                        ▶
                      </Button>
                    </div>
                    
                    {/* Mostrar imágenes del ejemplo actual */}
                    {currentPuzzle.visualization.train_examples[currentExampleIndex] && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div>
                          <div style={{ color: '#888', fontSize: '10px', marginBottom: '5px' }}>Input:</div>
                          <img 
                            src={`data:image/png;base64,${currentPuzzle.visualization.train_examples[currentExampleIndex].input_image}`}
                            alt="Input"
                            style={{ border: '1px solid #444', maxWidth: '150px' }}
                          />
                        </div>
                        <span style={{ color: '#666' }}>→</span>
                        <div>
                          <div style={{ color: '#888', fontSize: '10px', marginBottom: '5px' }}>Output:</div>
                          <img 
                            src={`data:image/png;base64,${currentPuzzle.visualization.train_examples[currentExampleIndex].output_image}`}
                            alt="Output"
                            style={{ border: '1px solid #444', maxWidth: '150px' }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Visualización de Test */}
                {viewMode === 'test' && currentPuzzle.visualization?.test_examples?.length > 0 && (
                  <div>
                    {currentPuzzle.visualization.test_examples.length > 1 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                        <Button 
                          onClick={() => setCurrentExampleIndex(Math.max(0, currentExampleIndex - 1))}
                          disabled={currentExampleIndex === 0}
                          style={{ fontSize: '12px', padding: '5px 10px' }}
                        >
                          ◀
                        </Button>
                        <span style={{ color: '#FF851B', fontSize: '12px' }}>
                          Test {currentExampleIndex + 1} de {currentPuzzle.visualization.test_examples.length}
                        </span>
                        <Button 
                          onClick={() => setCurrentExampleIndex(Math.min(currentPuzzle.visualization.test_examples.length - 1, currentExampleIndex + 1))}
                          disabled={currentExampleIndex >= currentPuzzle.visualization.test_examples.length - 1}
                          style={{ fontSize: '12px', padding: '5px 10px' }}
                        >
                          ▶
                        </Button>
                      </div>
                    )}
                    
                    {/* Mostrar imagen del test */}
                    {currentPuzzle.visualization.test_examples[currentExampleIndex] && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div>
                          <div style={{ color: '#888', fontSize: '10px', marginBottom: '5px' }}>Test Input:</div>
                          <img 
                            src={`data:image/png;base64,${currentPuzzle.visualization.test_examples[currentExampleIndex].input_image}`}
                            alt="Test Input"
                            style={{ border: '1px solid #444', maxWidth: '150px' }}
                          />
                        </div>
                        {currentPuzzle.visualization.test_examples[currentExampleIndex].output_image && (
                          <>
                            <span style={{ color: '#666' }}>→</span>
                            <div>
                              <div style={{ color: '#888', fontSize: '10px', marginBottom: '5px' }}>Expected Output:</div>
                              <img 
                                src={`data:image/png;base64,${currentPuzzle.visualization.test_examples[currentExampleIndex].output_image}`}
                                alt="Expected Output"
                                style={{ border: '1px solid #444', maxWidth: '150px' }}
                              />
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Fallback si no hay visualización */}
                {!currentPuzzle.visualization && currentPuzzle.train?.[0] && (
                  <div>
                    <div style={{ color: '#aaa', fontSize: '11px' }}>Ejemplo 1 (sin visualización):</div>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      {renderGrid(currentPuzzle.train[0].input)}
                      <span style={{ margin: '0 10px', color: '#666' }}>→</span>
                      {renderGrid(currentPuzzle.train[0].output)}
                    </div>
                  </div>
                )}
              </div>

              <Button 
                onClick={processPuzzle} 
                disabled={processing || !connected}
                style={{ marginTop: '20px' }}
              >
                {processing ? 'Procesando...' : '🧠 Procesar con PROTEUS (Real)'}
              </Button>
              
              <Button 
                onClick={() => setShowRawData(!showRawData)}
                style={{ marginTop: '10px', background: '#444' }}
              >
                {showRawData ? 'Ocultar' : 'Mostrar'} Datos Raw
              </Button>
            </div>
          )}

          <ModuleStatus>
            <div style={{ fontSize: '12px', color: '#7FDBFF', marginBottom: '10px', textAlign: 'center' }}>
              📊 ARQUITECTURA DE PROCESAMIENTO (Orden de Ejecución)
            </div>
            
            {/* Módulo 1: V-JEPA - Identificación del mundo visual */}
            <ModuleRow>
              <ExecutionOrder>1</ExecutionOrder>
              <Module $active={moduleStates.vjepa.active}>
                <ModuleName $active={moduleStates.vjepa.active}>
                  👁️ V-JEPA (Identificación Visual)
                </ModuleName>
                <ModuleOutput>
                  {(moduleStates.vjepa.output || 'No disponible').substring(0, 100)}...
                </ModuleOutput>
              </Module>
            </ModuleRow>
            
            {/* Módulo 2: Análisis Jerárquico - Con contexto de V-JEPA */}
            <ModuleRow>
              <ExecutionOrder>2</ExecutionOrder>
              <Module $active={moduleStates.hierarchical.active}>
                <ModuleName $active={moduleStates.hierarchical.active}>
                  🔍 JERÁRQUICO (Abstracción con contexto)
                </ModuleName>
                <ModuleOutput>
                  {(moduleStates.hierarchical.output || 'No disponible').substring(0, 100)}...
                </ModuleOutput>
              </Module>
            </ModuleRow>
            
            {/* Módulo 3: Atención */}
            <ModuleRow>
              <ExecutionOrder>3</ExecutionOrder>
              <Module $active={moduleStates.attention.active}>
                <ModuleName $active={moduleStates.attention.active}>
                  🎯 ATENCIÓN (Focalización)
                </ModuleName>
                <ModuleOutput>
                  {(moduleStates.attention.output || 'No disponible').substring(0, 100)}...
                </ModuleOutput>
              </Module>
            </ModuleRow>
            
            {/* Módulo 4: Memoria */}
            <ModuleRow>
              <ExecutionOrder>4</ExecutionOrder>
              <Module $active={moduleStates.memory.active}>
                <ModuleName $active={moduleStates.memory.active}>
                  💾 MEMORIA (Patrones)
                </ModuleName>
                <ModuleOutput>
                  {(moduleStates.memory.output || 'No disponible').substring(0, 100)}...
                </ModuleOutput>
              </Module>
            </ModuleRow>
            
            {/* Módulo 5: Razonamiento */}
            <ModuleRow>
              <ExecutionOrder>5</ExecutionOrder>
              <Module $active={moduleStates.reasoning.active}>
                <ModuleName $active={moduleStates.reasoning.active}>
                  🧠 RAZONAMIENTO (Síntesis)
                </ModuleName>
                <ModuleOutput>
                  {(moduleStates.reasoning.output || 'No disponible').substring(0, 100)}...
                </ModuleOutput>
              </Module>
            </ModuleRow>
          </ModuleStatus>

          {showRawData && currentPuzzle && (
            <RawDataDisplay>
              {JSON.stringify(currentPuzzle, null, 2)}
            </RawDataDisplay>
          )}
        </Panel>

        <Panel>
          <PanelTitle>Log de Procesamiento Real</PanelTitle>
          <ProcessLog>
            {processLog.length === 0 ? (
              <LogEntry type="info">
                Esperando acciones... Todo lo que aparezca aquí es procesamiento REAL del backend.
              </LogEntry>
            ) : (
              processLog.map((entry, i) => (
                <LogEntry key={i} type={entry.type}>
                  [{entry.timestamp}] {entry.message}
                </LogEntry>
              ))
            )}
            <div ref={logEndRef} />
          </ProcessLog>
        </Panel>
      </MainGrid>

      <Panel>
        <PanelTitle>⚠️ Notas de Transparencia</PanelTitle>
        <div style={{ fontSize: '13px', color: '#aaa', lineHeight: '1.6' }}>
          <p>• V-JEPA: Identificación visual primaria del mundo (objetos, colores, formas)</p>
          <p>• JERÁRQUICO: Análisis de capas usando el contexto visual de V-JEPA</p>
          <p>• ATENCIÓN: Los pasos que ves son los reales del módulo iterative_attention_observer.py</p>
          <p>• MEMORIA: Accesos reales a SQLite, no simulados</p>
          <p>• RAZONAMIENTO: Lo que el sistema realmente "piensa", sin inventar</p>
          <p>• Si algo falla o da error, lo verás inmediatamente en el log</p>
          <p>• Los porcentajes de confianza son calculados, no hardcodeados</p>
        </div>
      </Panel>
    </Container>
  );
}

export default AppARCReal;