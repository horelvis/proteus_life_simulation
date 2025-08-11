/**
 * Componente para visualizar el proceso de resolución de puzzles ARC en tiempo real
 * Muestra paso a paso cómo el sistema PROTEUS intenta resolver el puzzle
 */

import React, { useState, useEffect, useRef } from 'react';
import { ARCPuzzleVisualization } from './ARCPuzzleVisualization';

// Paleta de colores ARC estándar
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

const ARCSolvingProcess = ({ 
  puzzleId = 'demo', 
  trainExamples = [], 
  testInput = null,
  wsConnection = null 
}) => {
  // Estados con datos demo para visualización
  const [solvingSteps, setSolvingSteps] = useState([
    {
      type: 'analysis',
      description: 'Analizando patrones en los ejemplos de entrenamiento...',
      solution: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      confidence: 0.3
    },
    {
      type: 'pattern_detection',
      description: 'Detectando regla de transformación: Replicación de Patrón',
      solution: [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
      confidence: 0.6,
      rule: 'pattern_replication',
      transformation: 'Expandir cada celda 3x3'
    },
    {
      type: 'rule_application',
      description: 'Aplicando regla detectada al input de test',
      solution: [[2, 2, 2], [2, 3, 2], [2, 2, 2]],
      confidence: 0.85,
      rule: 'pattern_replication'
    }
  ]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [swarmInfo, setSwarmInfo] = useState({
    generation: 3,
    population: 20,
    bestFitness: 0.85,
    agents: [
      { id: 1, specialization: 'Pattern', fitness: 0.85 },
      { id: 2, specialization: 'Color', fitness: 0.72 },
      { id: 3, specialization: 'Symmetry', fitness: 0.68 },
      { id: 4, specialization: 'Rotation', fitness: 0.45 },
      { id: 5, specialization: 'Gravity', fitness: 0.33 }
    ]
  });
  const [analysisData, setAnalysisData] = useState({
    patterns: [
      { type: 'Replicación 3x3', confidence: 0.9 },
      { type: 'Mapeo de Color', confidence: 0.7 },
      { type: 'Simetría Horizontal', confidence: 0.5 }
    ],
    rules: [
      { name: 'Expandir Patrón', score: 0.95 },
      { name: 'Preservar Centro', score: 0.88 },
      { name: 'Aplicar Gradiente', score: 0.62 }
    ],
    transformations: ['scale', 'color_map', 'symmetry'],
    confidence: 0.85
  });
  const playbackRef = useRef(null);
  
  // Datos demo para visualización
  const demoTrainExamples = trainExamples.length > 0 ? trainExamples : [
    {
      input: [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
      output: [[0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 2, 2, 2, 1, 1, 1],
               [1, 1, 1, 2, 2, 2, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0]]
    }
  ];
  
  const demoTestInput = testInput || [[1, 0, 1], [0, 3, 0], [1, 0, 1]];

  // Conectar WebSocket para recibir actualizaciones en tiempo real
  useEffect(() => {
    if (wsConnection) {
      wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleSolvingUpdate(data);
      };
    }
  }, [wsConnection]);

  // Manejar actualizaciones del proceso de resolución
  const handleSolvingUpdate = (data) => {
    switch (data.type) {
      case 'step_update':
        setSolvingSteps(prev => [...prev, data.step]);
        break;
      case 'swarm_update':
        setSwarmInfo(data.swarmData);
        break;
      case 'analysis_update':
        setAnalysisData(data.analysis);
        break;
      case 'solution_found':
        setIsPlaying(false);
        break;
    }
  };

  // Control de reproducción de pasos
  useEffect(() => {
    if (isPlaying && solvingSteps.length > 0) {
      playbackRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= solvingSteps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    } else {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    }
    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, solvingSteps.length]);

  const currentStepData = solvingSteps[currentStep] || {};

  return (
    <div style={styles.container}>
      {/* Header con información del proceso */}
      <div style={styles.header}>
        <h2 style={styles.title}>🧩 Proceso de Resolución ARC - PROTEUS AGI</h2>
        <div style={styles.puzzleInfo}>
          <span>Puzzle ID: {puzzleId}</span>
          <span>Paso: {currentStep + 1}/{solvingSteps.length}</span>
          <span>Generación: {swarmInfo.generation}</span>
          <span>Mejor Fitness: {(swarmInfo.bestFitness * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Panel de Control de Reproducción */}
      <div style={styles.controlPanel}>
        <button 
          onClick={() => setCurrentStep(0)}
          style={styles.button}
          disabled={currentStep === 0}
        >
          ⏮️ Inicio
        </button>
        <button 
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          style={styles.button}
          disabled={currentStep === 0}
        >
          ⏪ Anterior
        </button>
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          style={{...styles.button, ...styles.playButton}}
        >
          {isPlaying ? '⏸️ Pausar' : '▶️ Reproducir'}
        </button>
        <button 
          onClick={() => setCurrentStep(Math.min(solvingSteps.length - 1, currentStep + 1))}
          style={styles.button}
          disabled={currentStep >= solvingSteps.length - 1}
        >
          Siguiente ⏩
        </button>
        <button 
          onClick={() => setCurrentStep(solvingSteps.length - 1)}
          style={styles.button}
          disabled={currentStep >= solvingSteps.length - 1}
        >
          Final ⏭️
        </button>
        
        <div style={styles.speedControl}>
          <label style={styles.label}>Velocidad:</label>
          <select 
            value={playbackSpeed} 
            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            style={styles.select}
          >
            <option value={2000}>0.5x</option>
            <option value={1000}>1x</option>
            <option value={500}>2x</option>
            <option value={250}>4x</option>
          </select>
        </div>
      </div>

      {/* Timeline Visual */}
      <div style={styles.timeline}>
        <div style={styles.timelineBar}>
          <div 
            style={{
              ...styles.timelineProgress,
              width: `${(currentStep / Math.max(1, solvingSteps.length - 1)) * 100}%`
            }}
          />
          {solvingSteps.map((step, idx) => (
            <div
              key={idx}
              style={{
                ...styles.timelineMarker,
                left: `${(idx / Math.max(1, solvingSteps.length - 1)) * 100}%`,
                backgroundColor: idx === currentStep ? '#FF4136' : 
                               step.type === 'solution' ? '#2ECC40' : '#AAAAAA'
              }}
              onClick={() => setCurrentStep(idx)}
              title={`Paso ${idx + 1}: ${step.description || step.type}`}
            />
          ))}
        </div>
      </div>

      {/* Visualización Principal */}
      <div style={styles.mainContent}>
        {/* Panel Izquierdo - Ejemplos de Entrenamiento */}
        <div style={styles.leftPanel}>
          <h3 style={styles.sectionTitle}>📚 Ejemplos de Entrenamiento</h3>
          <div style={styles.trainExamples}>
            {demoTrainExamples.map((example, idx) => (
              <div key={idx} style={styles.examplePair}>
                <ARCPuzzleVisualization 
                  puzzle={example.input}
                  cellSize={15}
                />
                <span style={styles.arrow}>→</span>
                <ARCPuzzleVisualization 
                  puzzle={example.output}
                  cellSize={15}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Panel Central - Estado Actual */}
        <div style={styles.centerPanel}>
          <h3 style={styles.sectionTitle}>🔍 Estado Actual del Proceso</h3>
          
          {/* Input del Test */}
          <div style={styles.puzzleSection}>
            <h4 style={styles.subTitle}>Input del Test</h4>
            <ARCPuzzleVisualization 
              puzzle={demoTestInput}
              cellSize={25}
            />
          </div>

          {/* Solución Actual */}
          <div style={styles.puzzleSection}>
            <h4 style={styles.subTitle}>
              Intento Actual 
              {currentStepData.confidence && 
                ` (Confianza: ${(currentStepData.confidence * 100).toFixed(1)}%)`}
            </h4>
            <ARCPuzzleVisualization 
              puzzle={currentStepData.solution}
              cellSize={25}
            />
          </div>

          {/* Descripción del Paso */}
          {currentStepData.description && (
            <div style={styles.stepDescription}>
              <h4 style={styles.subTitle}>Descripción del Paso</h4>
              <p>{currentStepData.description}</p>
              {currentStepData.rule && (
                <div style={styles.ruleInfo}>
                  <strong>Regla Aplicada:</strong> {currentStepData.rule}
                </div>
              )}
              {currentStepData.transformation && (
                <div style={styles.transformInfo}>
                  <strong>Transformación:</strong> {currentStepData.transformation}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Panel Derecho - Información del Enjambre */}
        <div style={styles.rightPanel}>
          <h3 style={styles.sectionTitle}>🐝 Estado del Enjambre</h3>
          
          {/* Estadísticas Generales */}
          <div style={styles.swarmStats}>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Población:</span>
              <span style={styles.statValue}>{swarmInfo.population}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Generación:</span>
              <span style={styles.statValue}>{swarmInfo.generation}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Mejor Fitness:</span>
              <span style={styles.statValue}>
                {(swarmInfo.bestFitness * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Lista de Agentes */}
          <div style={styles.agentsList}>
            <h4 style={styles.subTitle}>Agentes Activos</h4>
            <div style={styles.agentsContainer}>
              {swarmInfo.agents.slice(0, 10).map((agent, idx) => (
                <div 
                  key={agent.id}
                  style={{
                    ...styles.agentCard,
                    ...(selectedAgent === agent.id ? styles.selectedAgent : {})
                  }}
                  onClick={() => setSelectedAgent(agent.id)}
                >
                  <div style={styles.agentHeader}>
                    <span style={styles.agentId}>#{agent.id}</span>
                    <span style={styles.agentType}>{agent.specialization}</span>
                  </div>
                  <div style={styles.agentFitness}>
                    <div style={styles.fitnessBar}>
                      <div 
                        style={{
                          ...styles.fitnessProgress,
                          width: `${agent.fitness * 100}%`,
                          backgroundColor: agent.fitness > 0.8 ? '#2ECC40' : 
                                         agent.fitness > 0.5 ? '#FFDC00' : '#FF4136'
                        }}
                      />
                    </div>
                    <span style={styles.fitnessText}>
                      {(agent.fitness * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Patrones Detectados */}
          <div style={styles.patternsSection}>
            <h4 style={styles.subTitle}>🔬 Patrones Detectados</h4>
            <div style={styles.patternsList}>
              {analysisData.patterns.map((pattern, idx) => (
                <div key={idx} style={styles.patternItem}>
                  <span style={styles.patternType}>{pattern.type}</span>
                  <span style={styles.patternConfidence}>
                    {(pattern.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Reglas Descubiertas */}
          <div style={styles.rulesSection}>
            <h4 style={styles.subTitle}>📐 Reglas Descubiertas</h4>
            <div style={styles.rulesList}>
              {analysisData.rules.map((rule, idx) => (
                <div key={idx} style={styles.ruleItem}>
                  <span style={styles.ruleName}>{rule.name}</span>
                  <span style={styles.ruleScore}>
                    {rule.score.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Panel de Métricas y Análisis */}
      <div style={styles.metricsPanel}>
        <h3 style={styles.sectionTitle}>📊 Métricas del Proceso</h3>
        <div style={styles.metricsGrid}>
          <div style={styles.metricCard}>
            <span style={styles.metricLabel}>Tiempo Total</span>
            <span style={styles.metricValue}>
              {currentStepData.totalTime ? `${currentStepData.totalTime.toFixed(2)}s` : '-'}
            </span>
          </div>
          <div style={styles.metricCard}>
            <span style={styles.metricLabel}>Intentos</span>
            <span style={styles.metricValue}>{solvingSteps.length}</span>
          </div>
          <div style={styles.metricCard}>
            <span style={styles.metricLabel}>Transformaciones</span>
            <span style={styles.metricValue}>{analysisData.transformations.length}</span>
          </div>
          <div style={styles.metricCard}>
            <span style={styles.metricLabel}>Confianza Promedio</span>
            <span style={styles.metricValue}>
              {analysisData.confidence ? `${(analysisData.confidence * 100).toFixed(1)}%` : '-'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Estilos del componente
const styles = {
  container: {
    backgroundColor: '#1a1a1a',
    color: '#fff',
    padding: '20px',
    borderRadius: '12px',
    fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
  },
  header: {
    marginBottom: '20px',
    borderBottom: '2px solid #333',
    paddingBottom: '15px',
  },
  title: {
    fontSize: '24px',
    marginBottom: '10px',
    color: '#fff',
  },
  puzzleInfo: {
    display: 'flex',
    gap: '20px',
    fontSize: '14px',
    color: '#aaa',
  },
  controlPanel: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '20px',
    padding: '15px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
  },
  button: {
    padding: '8px 16px',
    backgroundColor: '#3a3a3a',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'all 0.3s ease',
  },
  playButton: {
    backgroundColor: '#0074D9',
  },
  speedControl: {
    marginLeft: 'auto',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  label: {
    fontSize: '14px',
    color: '#aaa',
  },
  select: {
    padding: '6px 10px',
    backgroundColor: '#3a3a3a',
    color: '#fff',
    border: '1px solid #555',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  timeline: {
    marginBottom: '20px',
    padding: '10px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
  },
  timelineBar: {
    position: 'relative',
    height: '40px',
    backgroundColor: '#3a3a3a',
    borderRadius: '20px',
    overflow: 'hidden',
  },
  timelineProgress: {
    position: 'absolute',
    height: '100%',
    backgroundColor: '#0074D9',
    transition: 'width 0.3s ease',
  },
  timelineMarker: {
    position: 'absolute',
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    top: '50%',
    transform: 'translate(-50%, -50%)',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  mainContent: {
    display: 'grid',
    gridTemplateColumns: '250px 1fr 300px',
    gap: '20px',
    marginBottom: '20px',
  },
  leftPanel: {
    backgroundColor: '#2a2a2a',
    padding: '15px',
    borderRadius: '8px',
  },
  centerPanel: {
    backgroundColor: '#2a2a2a',
    padding: '20px',
    borderRadius: '8px',
  },
  rightPanel: {
    backgroundColor: '#2a2a2a',
    padding: '15px',
    borderRadius: '8px',
  },
  sectionTitle: {
    fontSize: '16px',
    marginBottom: '15px',
    color: '#fff',
    borderBottom: '1px solid #444',
    paddingBottom: '8px',
  },
  subTitle: {
    fontSize: '14px',
    marginBottom: '10px',
    color: '#ccc',
  },
  trainExamples: {
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
  },
  examplePair: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  arrow: {
    color: '#666',
    fontSize: '20px',
  },
  puzzleSection: {
    marginBottom: '20px',
  },
  stepDescription: {
    padding: '15px',
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    fontSize: '14px',
    lineHeight: '1.6',
  },
  ruleInfo: {
    marginTop: '10px',
    padding: '8px',
    backgroundColor: '#2a2a2a',
    borderRadius: '4px',
    fontSize: '13px',
  },
  transformInfo: {
    marginTop: '10px',
    padding: '8px',
    backgroundColor: '#2a2a2a',
    borderRadius: '4px',
    fontSize: '13px',
  },
  swarmStats: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    marginBottom: '15px',
  },
  statItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 10px',
    backgroundColor: '#1a1a1a',
    borderRadius: '4px',
  },
  statLabel: {
    color: '#888',
    fontSize: '13px',
  },
  statValue: {
    color: '#fff',
    fontSize: '13px',
    fontWeight: 'bold',
  },
  agentsList: {
    marginBottom: '15px',
  },
  agentsContainer: {
    maxHeight: '200px',
    overflowY: 'auto',
  },
  agentCard: {
    padding: '8px',
    marginBottom: '6px',
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  selectedAgent: {
    backgroundColor: '#0074D9',
  },
  agentHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '6px',
  },
  agentId: {
    fontSize: '12px',
    color: '#888',
  },
  agentType: {
    fontSize: '11px',
    padding: '2px 6px',
    backgroundColor: '#3a3a3a',
    borderRadius: '3px',
  },
  agentFitness: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  fitnessBar: {
    flex: 1,
    height: '6px',
    backgroundColor: '#3a3a3a',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  fitnessProgress: {
    height: '100%',
    transition: 'width 0.3s ease',
  },
  fitnessText: {
    fontSize: '11px',
    color: '#aaa',
  },
  patternsSection: {
    marginBottom: '15px',
  },
  patternsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  patternItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 8px',
    backgroundColor: '#1a1a1a',
    borderRadius: '4px',
    fontSize: '12px',
  },
  patternType: {
    color: '#ccc',
  },
  patternConfidence: {
    color: '#2ECC40',
    fontWeight: 'bold',
  },
  rulesSection: {
    marginBottom: '15px',
  },
  rulesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  ruleItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 8px',
    backgroundColor: '#1a1a1a',
    borderRadius: '4px',
    fontSize: '12px',
  },
  ruleName: {
    color: '#ccc',
  },
  ruleScore: {
    color: '#FFDC00',
    fontWeight: 'bold',
  },
  metricsPanel: {
    backgroundColor: '#2a2a2a',
    padding: '20px',
    borderRadius: '8px',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '15px',
  },
  metricCard: {
    display: 'flex',
    flexDirection: 'column',
    padding: '12px',
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: '12px',
    color: '#888',
    marginBottom: '8px',
  },
  metricValue: {
    fontSize: '20px',
    color: '#fff',
    fontWeight: 'bold',
  },
};

export default ARCSolvingProcess;