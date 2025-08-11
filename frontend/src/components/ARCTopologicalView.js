/**
 * Componente para visualizar el proceso del solver topológico de ARC
 * Muestra análisis multi-escala, segmentación de objetos y síntesis de reglas
 */

import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';

// Estilos con tema oscuro
const Container = styled.div`
  background: linear-gradient(135deg, #0a0e27 0%, #151932 100%);
  color: #e0e6ed;
  padding: 2rem;
  min-height: 100vh;
  font-family: 'JetBrains Mono', monospace;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 2rem;
  
  h1 {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }
  
  .subtitle {
    color: #8892b0;
    font-size: 1.1rem;
  }
  
  .gpu-status {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    background: ${props => props.$gpuActive ? '#00ff8850' : '#ff444450'};
    border: 1px solid ${props => props.$gpuActive ? '#00ff88' : '#ff4444'};
    border-radius: 20px;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: ${props => props.$gpuActive ? '#00ff88' : '#ff4444'};
  }
`;

const MainGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;
  
  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
`;

const Section = styled.div`
  background: rgba(30, 40, 80, 0.3);
  border: 1px solid rgba(100, 120, 200, 0.2);
  border-radius: 12px;
  padding: 1.5rem;
  backdrop-filter: blur(10px);
  
  h2 {
    color: #64ffda;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .icon {
    width: 24px;
    height: 24px;
  }
`;

const PerceptionView = styled.div`
  .pyramid-levels {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .level {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    
    .level-label {
      font-weight: bold;
      color: #64ffda;
      min-width: 80px;
    }
    
    .level-grid {
      display: grid;
      gap: 2px;
      background: #000;
      padding: 4px;
      border-radius: 4px;
    }
    
    .cell {
      width: 8px;
      height: 8px;
      background: var(--cell-color);
    }
    
    .features {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      
      .feature-tag {
        padding: 0.2rem 0.5rem;
        background: rgba(100, 255, 218, 0.1);
        border: 1px solid rgba(100, 255, 218, 0.3);
        border-radius: 4px;
        font-size: 0.8rem;
      }
    }
  }
`;

const ObjectGraph = styled.div`
  .objects-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    max-height: 400px;
    overflow-y: auto;
    
    &::-webkit-scrollbar {
      width: 6px;
    }
    
    &::-webkit-scrollbar-track {
      background: rgba(0, 0, 0, 0.3);
    }
    
    &::-webkit-scrollbar-thumb {
      background: rgba(100, 255, 218, 0.3);
      border-radius: 3px;
    }
  }
  
  .object-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    border: 1px solid transparent;
    transition: all 0.3s;
    cursor: pointer;
    
    &:hover {
      border-color: rgba(100, 255, 218, 0.3);
      background: rgba(100, 255, 218, 0.05);
    }
    
    .object-color {
      width: 24px;
      height: 24px;
      border-radius: 4px;
      background: var(--obj-color);
      border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .object-info {
      flex: 1;
      margin: 0 1rem;
      
      .object-id {
        font-weight: bold;
        color: #64ffda;
      }
      
      .object-props {
        font-size: 0.8rem;
        color: #8892b0;
        margin-top: 0.2rem;
      }
    }
    
    .object-topology {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      font-size: 0.8rem;
      
      .euler {
        color: #f07178;
      }
      
      .holes {
        color: #ffcb6b;
      }
    }
  }
  
  .graph-viz {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    text-align: center;
    
    svg {
      max-width: 100%;
      height: 200px;
    }
  }
`;

const RuleSynthesis = styled.div`
  .rules-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .rule-item {
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    border-left: 3px solid var(--rule-color);
    
    .rule-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      
      .rule-type {
        font-weight: bold;
        color: #64ffda;
      }
      
      .confidence {
        padding: 0.2rem 0.5rem;
        background: rgba(100, 255, 218, 0.1);
        border-radius: 4px;
        font-size: 0.8rem;
      }
    }
    
    .rule-desc {
      color: #8892b0;
      font-size: 0.9rem;
    }
    
    .rule-params {
      margin-top: 0.5rem;
      padding: 0.5rem;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.8rem;
      color: #c792ea;
    }
  }
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  overflow: hidden;
  margin: 1rem 0;
  
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    width: ${props => props.$progress}%;
    transition: width 0.5s ease;
  }
`;

const MetricsPanel = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
  
  .metric {
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    text-align: center;
    
    .metric-value {
      font-size: 1.8rem;
      font-weight: bold;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
      color: #8892b0;
      font-size: 0.9rem;
      margin-top: 0.25rem;
    }
  }
`;

// Colores ARC
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

const ARCTopologicalView = ({ wsConnection }) => {
  const [gpuActive, setGpuActive] = useState(true);
  const [solverState, setSolverState] = useState({
    phase: 'perception',
    progress: 35,
    pyramidLevels: [
      { level: 0, resolution: '21x21', features: ['edges', 'density', 'fft'] },
      { level: 1, resolution: '10x10', features: ['edges', 'density'] },
      { level: 2, resolution: '5x5', features: ['density'] }
    ],
    objects: [
      { id: 0, color: 1, area: 45, euler: 1, holes: 0, centroid: [10, 10] },
      { id: 1, color: 2, area: 32, euler: 0, holes: 1, centroid: [15, 5] },
      { id: 2, color: 3, area: 28, euler: 1, holes: 0, centroid: [5, 15] },
      { id: 3, color: 4, area: 18, euler: -1, holes: 2, centroid: [12, 12] }
    ],
    rules: [
      { type: 'recolor', confidence: 0.85, params: { from: 1, to: 2 } },
      { type: 'replicate', confidence: 0.72, params: { vector: [3, 0], times: 2 } },
      { type: 'fill_holes', confidence: 0.68, params: {} },
      { type: 'reflect', confidence: 0.45, params: { axis: 'horizontal' } }
    ],
    metrics: {
      objectsDetected: 4,
      rulesGenerated: 4,
      elapsedTime: 0.046,
      validationScore: 0.85
    }
  });

  // Simulación de actualización en tiempo real
  useEffect(() => {
    const interval = setInterval(() => {
      setSolverState(prev => ({
        ...prev,
        progress: Math.min(100, prev.progress + 5),
        metrics: {
          ...prev.metrics,
          elapsedTime: prev.metrics.elapsedTime + 0.01
        }
      }));
    }, 500);

    return () => clearInterval(interval);
  }, []);

  // Conexión WebSocket para actualizaciones reales
  useEffect(() => {
    if (wsConnection) {
      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'solver_update') {
            setSolverState(data.state);
            setGpuActive(data.gpu_active || false);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    }
  }, [wsConnection]);

  return (
    <Container>
      <Header $gpuActive={gpuActive}>
        <h1>PROTEUS ARC Topological Solver</h1>
        <div className="subtitle">Multi-Scale Analysis with Object-Based Reasoning</div>
        <div className="gpu-status">
          {gpuActive ? '🚀 GPU Acceleration Active (CuPy)' : '💻 CPU Mode'}
        </div>
      </Header>

      <ProgressBar $progress={solverState.progress}>
        <div className="progress-fill" />
      </ProgressBar>

      <MainGrid>
        <Section>
          <h2>
            <span className="icon">🔍</span>
            Multi-Scale Perception
          </h2>
          <PerceptionView>
            <div className="pyramid-levels">
              {solverState.pyramidLevels.map((level, idx) => (
                <div key={idx} className="level">
                  <div className="level-label">Level {level.level}</div>
                  <div className="level-info">
                    <div>Resolution: {level.resolution}</div>
                    <div className="features">
                      {level.features.map(f => (
                        <span key={f} className="feature-tag">{f}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </PerceptionView>
        </Section>

        <Section>
          <h2>
            <span className="icon">🔗</span>
            Object Segmentation & Graph
          </h2>
          <ObjectGraph>
            <div className="objects-list">
              {solverState.objects.map(obj => (
                <div key={obj.id} className="object-item">
                  <div 
                    className="object-color" 
                    style={{'--obj-color': ARC_COLORS[obj.color]}}
                  />
                  <div className="object-info">
                    <div className="object-id">Object #{obj.id}</div>
                    <div className="object-props">
                      Area: {obj.area} | Pos: ({obj.centroid[0]}, {obj.centroid[1]})
                    </div>
                  </div>
                  <div className="object-topology">
                    <div className="euler">χ = {obj.euler}</div>
                    <div className="holes">β₁ = {obj.holes}</div>
                  </div>
                </div>
              ))}
            </div>
          </ObjectGraph>
        </Section>

        <Section>
          <h2>
            <span className="icon">⚙️</span>
            Rule Synthesis
          </h2>
          <RuleSynthesis>
            <div className="rules-list">
              {solverState.rules.map((rule, idx) => (
                <div 
                  key={idx} 
                  className="rule-item"
                  style={{'--rule-color': `hsl(${120 * rule.confidence}, 70%, 50%)`}}
                >
                  <div className="rule-header">
                    <div className="rule-type">{rule.type.toUpperCase()}</div>
                    <div className="confidence">{(rule.confidence * 100).toFixed(0)}%</div>
                  </div>
                  <div className="rule-desc">
                    {rule.type === 'recolor' && `Change color ${rule.params.from} → ${rule.params.to}`}
                    {rule.type === 'replicate' && `Replicate ${rule.params.times}x along (${rule.params.vector})`}
                    {rule.type === 'fill_holes' && 'Fill all topological holes'}
                    {rule.type === 'reflect' && `Reflect along ${rule.params.axis} axis`}
                  </div>
                  {Object.keys(rule.params).length > 0 && (
                    <div className="rule-params">
                      {JSON.stringify(rule.params, null, 2)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </RuleSynthesis>
        </Section>
      </MainGrid>

      <MetricsPanel>
        <div className="metric">
          <div className="metric-value">{solverState.metrics.objectsDetected}</div>
          <div className="metric-label">Objects Detected</div>
        </div>
        <div className="metric">
          <div className="metric-value">{solverState.metrics.rulesGenerated}</div>
          <div className="metric-label">Rules Generated</div>
        </div>
        <div className="metric">
          <div className="metric-value">{solverState.metrics.elapsedTime.toFixed(3)}s</div>
          <div className="metric-label">Processing Time</div>
        </div>
        <div className="metric">
          <div className="metric-value">{(solverState.metrics.validationScore * 100).toFixed(0)}%</div>
          <div className="metric-label">Validation Score</div>
        </div>
      </MetricsPanel>
    </Container>
  );
};

export default ARCTopologicalView;