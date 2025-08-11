/**
 * Componente mejorado para mostrar el razonamiento del solver ARC
 */

import React, { useState } from 'react';

export const ARCReasoningDisplay = ({ 
  reasoning = [], 
  confidence = 0,
  isCorrect = null 
}) => {
  const [expandedSteps, setExpandedSteps] = useState(new Set());

  const toggleStep = (index) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };

  const getStepIcon = (step) => {
    switch (step.type) {
      case 'analysis': return '🔍';
      case 'pattern': return '🎯';
      case 'rule': return '📐';
      case 'transformation': return '🔄';
      case 'validation': return '✓';
      case 'error': return '⚠️';
      case 'success': return '✅';
      case 'hypothesis': return '💡';
      default: return '📝';
    }
  };

  const getStepColor = (step) => {
    switch (step.type) {
      case 'error': return '#FF4136';
      case 'success': return '#2ECC40';
      case 'pattern': return '#0074D9';
      case 'rule': return '#B10DC9';
      case 'transformation': return '#FF851B';
      case 'hypothesis': return '#FFDC00';
      default: return '#7FDBFF';
    }
  };

  const formatConfidence = (conf) => {
    const percentage = (conf * 100).toFixed(1);
    let color = '#FF4136'; // Rojo por defecto
    if (conf > 0.8) color = '#2ECC40'; // Verde
    else if (conf > 0.6) color = '#FFDC00'; // Amarillo
    else if (conf > 0.4) color = '#FF851B'; // Naranja
    
    return { percentage, color };
  };

  const renderStepDetails = (details) => {
    if (!details) return null;
    
    if (typeof details === 'string') {
      return <p style={{ margin: '5px 0', fontSize: '13px', color: '#ccc' }}>{details}</p>;
    }
    
    if (typeof details === 'object') {
      return (
        <div style={{ fontSize: '13px', color: '#ccc' }}>
          {Object.entries(details).map(([key, value]) => (
            <div key={key} style={{ margin: '5px 0' }}>
              <strong style={{ color: '#888' }}>{key}:</strong> 
              <span style={{ marginLeft: '10px' }}>
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    
    return null;
  };

  const { percentage, color: confColor } = formatConfidence(confidence);

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      padding: '20px',
      borderRadius: '8px',
      width: '100%'
    }}>
      {/* Header con resultado y confianza */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#2a2a2a',
        borderRadius: '5px',
        border: isCorrect !== null ? (isCorrect ? '2px solid #2ECC40' : '2px solid #FF4136') : 'none'
      }}>
        <div>
          <h3 style={{ color: '#fff', margin: 0 }}>
            🧠 Análisis del Razonamiento
          </h3>
          {isCorrect !== null && (
            <p style={{ 
              margin: '5px 0 0 0', 
              color: isCorrect ? '#2ECC40' : '#FF4136',
              fontWeight: 'bold'
            }}>
              {isCorrect ? '✅ Solución Correcta' : '❌ Solución Incorrecta'}
            </p>
          )}
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '12px', color: '#888', marginBottom: '5px' }}>
            Confianza
          </div>
          <div style={{
            fontSize: '24px',
            fontWeight: 'bold',
            color: confColor
          }}>
            {percentage}%
          </div>
          <div style={{
            width: '100px',
            height: '4px',
            backgroundColor: '#333',
            borderRadius: '2px',
            marginTop: '5px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${percentage}%`,
              height: '100%',
              backgroundColor: confColor,
              transition: 'width 0.3s ease'
            }} />
          </div>
        </div>
      </div>

      {/* Resumen del proceso */}
      <div style={{
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#2a2a2a',
        borderRadius: '5px'
      }}>
        <h4 style={{ color: '#7FDBFF', marginBottom: '10px' }}>
          📊 Resumen del Proceso
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#fff' }}>
              {reasoning.length}
            </div>
            <div style={{ fontSize: '12px', color: '#888' }}>
              Pasos Totales
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#2ECC40' }}>
              {reasoning.filter(s => s.type === 'pattern').length}
            </div>
            <div style={{ fontSize: '12px', color: '#888' }}>
              Patrones Detectados
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#B10DC9' }}>
              {reasoning.filter(s => s.type === 'rule').length}
            </div>
            <div style={{ fontSize: '12px', color: '#888' }}>
              Reglas Aplicadas
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#FF851B' }}>
              {reasoning.filter(s => s.type === 'transformation').length}
            </div>
            <div style={{ fontSize: '12px', color: '#888' }}>
              Transformaciones
            </div>
          </div>
        </div>
      </div>

      {/* Pasos del razonamiento */}
      <div>
        <h4 style={{ color: '#fff', marginBottom: '15px' }}>
          🔍 Pasos Detallados del Razonamiento
        </h4>
        
        {reasoning.length === 0 ? (
          <div style={{
            padding: '20px',
            backgroundColor: '#2a2a2a',
            borderRadius: '5px',
            textAlign: 'center',
            color: '#888'
          }}>
            No hay pasos de razonamiento disponibles
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {reasoning.map((step, index) => (
              <div
                key={index}
                style={{
                  backgroundColor: '#2a2a2a',
                  borderRadius: '5px',
                  overflow: 'hidden',
                  border: `2px solid ${expandedSteps.has(index) ? getStepColor(step) : 'transparent'}`,
                  transition: 'all 0.3s ease'
                }}
              >
                <div
                  onClick={() => toggleStep(index)}
                  style={{
                    padding: '15px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    backgroundColor: expandedSteps.has(index) ? '#333' : 'transparent',
                    transition: 'background-color 0.3s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                    <span style={{ 
                      fontSize: '20px', 
                      marginRight: '10px',
                      filter: expandedSteps.has(index) ? 'none' : 'grayscale(50%)'
                    }}>
                      {getStepIcon(step)}
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        fontWeight: 'bold', 
                        color: getStepColor(step),
                        marginBottom: '3px'
                      }}>
                        Paso {index + 1}: {step.description || step.title || 'Procesando...'}
                      </div>
                      {step.summary && (
                        <div style={{ fontSize: '12px', color: '#888' }}>
                          {step.summary}
                        </div>
                      )}
                    </div>
                  </div>
                  <div style={{ 
                    color: '#666',
                    fontSize: '12px',
                    transform: expandedSteps.has(index) ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.3s ease'
                  }}>
                    ▼
                  </div>
                </div>
                
                {expandedSteps.has(index) && (
                  <div style={{
                    padding: '15px',
                    borderTop: '1px solid #444',
                    backgroundColor: '#1a1a1a'
                  }}>
                    {step.details && renderStepDetails(step.details)}
                    
                    {step.confidence !== undefined && (
                      <div style={{ 
                        marginTop: '10px',
                        padding: '10px',
                        backgroundColor: '#2a2a2a',
                        borderRadius: '3px'
                      }}>
                        <span style={{ color: '#888', fontSize: '12px' }}>
                          Confianza del paso: 
                        </span>
                        <span style={{ 
                          color: formatConfidence(step.confidence).color,
                          fontWeight: 'bold',
                          marginLeft: '10px'
                        }}>
                          {(step.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    
                    {step.error && (
                      <div style={{
                        marginTop: '10px',
                        padding: '10px',
                        backgroundColor: 'rgba(255, 65, 54, 0.1)',
                        borderLeft: '3px solid #FF4136',
                        borderRadius: '3px'
                      }}>
                        <strong style={{ color: '#FF4136' }}>Error:</strong>
                        <p style={{ margin: '5px 0 0 0', color: '#ccc' }}>{step.error}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Análisis de fallos si la solución es incorrecta */}
      {isCorrect === false && reasoning.length > 0 && (
        <div style={{
          marginTop: '20px',
          padding: '15px',
          backgroundColor: 'rgba(255, 65, 54, 0.1)',
          border: '2px solid #FF4136',
          borderRadius: '5px'
        }}>
          <h4 style={{ color: '#FF4136', marginBottom: '10px' }}>
            ⚠️ Análisis del Fallo
          </h4>
          <div style={{ color: '#ccc', fontSize: '13px' }}>
            <p style={{ marginBottom: '10px' }}>
              La solución no coincide con la esperada. Posibles causas:
            </p>
            <ul style={{ marginLeft: '20px' }}>
              <li>Patrón no detectado correctamente</li>
              <li>Regla de transformación incorrecta</li>
              <li>Falta de ejemplos de entrenamiento</li>
              <li>Caso especial no considerado</li>
            </ul>
            <p style={{ marginTop: '10px', color: '#FF851B' }}>
              💡 Sugerencia: Revisa los pasos marcados en rojo para identificar dónde falló el razonamiento.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};