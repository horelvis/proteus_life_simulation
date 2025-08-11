/**
 * Componente mejorado para visualizar puzzles ARC con grids claros
 */

import React from 'react';

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

export const ARCPuzzleVisualization = ({ 
  puzzle, 
  solution = null, 
  expected = null,
  showComparison = false,
  cellSize = 30 
}) => {
  const renderGrid = (grid, title, isCorrect = null) => {
    if (!grid || !Array.isArray(grid)) return null;
    
    const height = grid.length;
    const width = grid[0]?.length || 0;
    
    return (
      <div style={{ 
        margin: '10px',
        padding: '15px',
        backgroundColor: '#2a2a2a',
        borderRadius: '8px',
        border: isCorrect !== null ? (isCorrect ? '2px solid #2ECC40' : '2px solid #FF4136') : 'none'
      }}>
        <h4 style={{ 
          marginBottom: '10px', 
          color: '#fff',
          fontSize: '14px',
          textAlign: 'center'
        }}>
          {title}
          {isCorrect !== null && (
            <span style={{ marginLeft: '10px' }}>
              {isCorrect ? '✅' : '❌'}
            </span>
          )}
        </h4>
        <div style={{
          display: 'inline-block',
          border: '2px solid #444',
          borderRadius: '4px',
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
            {grid.map((row, i) => 
              row.map((cell, j) => (
                <div
                  key={`${i}-${j}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: ARC_COLORS[cell] || '#000',
                    border: '1px solid rgba(255,255,255,0.1)',
                    position: 'relative'
                  }}
                >
                  {/* Mostrar número si no es 0 */}
                  {cell !== 0 && (
                    <span style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      color: cell > 4 ? '#000' : '#fff',
                      fontSize: '10px',
                      fontWeight: 'bold',
                      textShadow: '0 0 2px rgba(0,0,0,0.5)'
                    }}>
                      {cell}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
        <div style={{ 
          marginTop: '5px', 
          fontSize: '11px', 
          color: '#888',
          textAlign: 'center'
        }}>
          {height}×{width}
        </div>
      </div>
    );
  };

  const renderComparison = () => {
    if (!solution || !expected) return null;
    
    // Calcular diferencias
    const differences = [];
    for (let i = 0; i < Math.min(solution.length, expected.length); i++) {
      for (let j = 0; j < Math.min(solution[i].length, expected[i].length); j++) {
        if (solution[i][j] !== expected[i][j]) {
          differences.push({ row: i, col: j, got: solution[i][j], expected: expected[i][j] });
        }
      }
    }
    
    const isCorrect = differences.length === 0;
    
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        padding: '20px',
        borderRadius: '8px',
        margin: '20px 0'
      }}>
        <h3 style={{ color: isCorrect ? '#2ECC40' : '#FF4136', marginBottom: '15px' }}>
          {isCorrect ? '✅ Solución Correcta' : `❌ Solución Incorrecta (${differences.length} diferencias)`}
        </h3>
        
        <div style={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '20px' }}>
          {renderGrid(solution, 'Tu Solución', false)}
          {renderGrid(expected, 'Solución Esperada', true)}
        </div>
        
        {!isCorrect && differences.length > 0 && (
          <div style={{
            marginTop: '20px',
            padding: '15px',
            backgroundColor: '#2a2a2a',
            borderRadius: '5px'
          }}>
            <h4 style={{ color: '#FF851B', marginBottom: '10px' }}>
              📊 Análisis de Diferencias:
            </h4>
            <div style={{ fontSize: '13px', color: '#ccc' }}>
              {differences.slice(0, 5).map((diff, idx) => (
                <div key={idx} style={{ marginBottom: '5px' }}>
                  • Posición [{diff.row},{diff.col}]: 
                  <span style={{ color: ARC_COLORS[diff.got], marginLeft: '5px' }}>
                    Obtuviste {diff.got}
                  </span>
                  <span style={{ color: '#888', margin: '0 5px' }}>→</span>
                  <span style={{ color: ARC_COLORS[diff.expected] }}>
                    Esperado {diff.expected}
                  </span>
                </div>
              ))}
              {differences.length > 5 && (
                <div style={{ marginTop: '10px', color: '#888' }}>
                  ... y {differences.length - 5} diferencias más
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderTrainingExamples = () => {
    if (!puzzle?.train || puzzle.train.length === 0) return null;
    
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        padding: '20px',
        borderRadius: '8px',
        margin: '20px 0'
      }}>
        <h3 style={{ color: '#7FDBFF', marginBottom: '15px' }}>
          📚 Ejemplos de Entrenamiento ({puzzle.train.length})
        </h3>
        {puzzle.train.map((example, idx) => (
          <div key={idx} style={{
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: '#2a2a2a',
            borderRadius: '5px'
          }}>
            <h4 style={{ color: '#fff', marginBottom: '10px', fontSize: '14px' }}>
              Ejemplo {idx + 1}
            </h4>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
              {renderGrid(example.input, 'Entrada')}
              <div style={{ 
                fontSize: '24px', 
                margin: '0 20px',
                color: '#666'
              }}>
                →
              </div>
              {renderGrid(example.output, 'Salida')}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderTestCase = () => {
    if (!puzzle?.test || puzzle.test.length === 0) return null;
    
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        padding: '20px',
        borderRadius: '8px',
        margin: '20px 0'
      }}>
        <h3 style={{ color: '#FF851B', marginBottom: '15px' }}>
          🎯 Caso de Prueba
        </h3>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
          {renderGrid(puzzle.test[0].input, 'Entrada de Prueba')}
          {solution && (
            <>
              <div style={{ fontSize: '24px', margin: '0 20px', color: '#666' }}>
                →
              </div>
              {renderGrid(solution, 'Tu Solución', expected && JSON.stringify(solution) === JSON.stringify(expected))}
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div style={{ width: '100%' }}>
      {/* Información del puzzle */}
      {puzzle && (
        <div style={{
          backgroundColor: '#2a2a2a',
          padding: '15px',
          borderRadius: '8px',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#fff', marginBottom: '5px' }}>
            Puzzle: {puzzle.id || 'Sin ID'}
          </h2>
          {puzzle.category && (
            <span style={{ 
              backgroundColor: '#0074D9', 
              padding: '3px 10px', 
              borderRadius: '3px',
              fontSize: '12px',
              marginRight: '10px'
            }}>
              {puzzle.category}
            </span>
          )}
          {puzzle.difficulty && (
            <span style={{ 
              backgroundColor: '#FF851B', 
              padding: '3px 10px', 
              borderRadius: '3px',
              fontSize: '12px'
            }}>
              {puzzle.difficulty}
            </span>
          )}
        </div>
      )}

      {/* Ejemplos de entrenamiento */}
      {renderTrainingExamples()}
      
      {/* Caso de prueba */}
      {renderTestCase()}
      
      {/* Comparación si está disponible */}
      {showComparison && renderComparison()}
      
      {/* Leyenda de colores */}
      <div style={{
        backgroundColor: '#2a2a2a',
        padding: '15px',
        borderRadius: '8px',
        marginTop: '20px'
      }}>
        <h4 style={{ color: '#888', marginBottom: '10px', fontSize: '12px' }}>
          Leyenda de Colores:
        </h4>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {Object.entries(ARC_COLORS).map(([value, color]) => (
            <div key={value} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                width: '20px',
                height: '20px',
                backgroundColor: color,
                border: '1px solid #444',
                marginRight: '5px'
              }} />
              <span style={{ fontSize: '11px', color: '#888' }}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};