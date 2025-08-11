/**
 * Componente para mostrar benchmarks de LLMs en ARC
 */

import React from 'react';

export const ARCBenchmarkDisplay = ({ benchmarks, currentScore }) => {
  if (!benchmarks) return null;

  // Ordenar benchmarks por accuracy
  const sortedBenchmarks = Object.entries(benchmarks)
    .sort((a, b) => b[1].accuracy - a[1].accuracy)
    .map(([key, data]) => ({ key, ...data }));

  // Actualizar score de PROTEUS
  const proteusIndex = sortedBenchmarks.findIndex(b => b.key === 'proteus');
  if (proteusIndex >= 0) {
    sortedBenchmarks[proteusIndex].accuracy = currentScore;
  }

  const getBarColor = (key) => {
    switch (key) {
      case 'human_baseline': return '#2ECC40';
      case 'proteus': return '#F012BE';
      case 'gpt-4o': return '#0074D9';
      case 'claude-3-opus': return '#FF851B';
      case 'gemini-1.5-pro': return '#FFDC00';
      default: return '#7FDBFF';
    }
  };

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      padding: '20px',
      borderRadius: '8px',
      marginBottom: '30px'
    }}>
      <h3 style={{ color: '#fff', marginBottom: '20px' }}>
        🏆 Benchmarks ARC - Comparación con LLMs
      </h3>
      
      <div style={{ fontSize: '12px', color: '#888', marginBottom: '15px' }}>
        Según <a href="https://docs.arcprize.org/games" target="_blank" rel="noopener noreferrer" 
                style={{ color: '#7FDBFF' }}>docs.arcprize.org/games</a>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {sortedBenchmarks.map((benchmark) => {
          const isProteus = benchmark.key === 'proteus';
          const isHuman = benchmark.key === 'human_baseline';
          
          return (
            <div key={benchmark.key} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '10px',
              backgroundColor: isProteus ? 'rgba(240, 18, 190, 0.1)' : 'transparent',
              borderRadius: '5px',
              border: isProteus ? '1px solid #F012BE' : 'none'
            }}>
              <div style={{ 
                minWidth: '180px',
                fontWeight: isProteus || isHuman ? 'bold' : 'normal',
                color: isHuman ? '#2ECC40' : (isProteus ? '#F012BE' : '#fff')
              }}>
                {benchmark.model}
                {isProteus && ' 🚀'}
                {isHuman && ' 👤'}
              </div>
              
              <div style={{ flex: 1, position: 'relative' }}>
                <div style={{
                  backgroundColor: '#2a2a2a',
                  height: '24px',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    backgroundColor: getBarColor(benchmark.key),
                    height: '100%',
                    width: `${benchmark.accuracy * 100}%`,
                    transition: 'width 0.5s ease',
                    display: 'flex',
                    alignItems: 'center',
                    paddingLeft: '8px'
                  }}>
                    {benchmark.accuracy > 0.1 && (
                      <span style={{ 
                        fontSize: '11px', 
                        color: '#000',
                        fontWeight: 'bold'
                      }}>
                        {(benchmark.accuracy * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                </div>
                {benchmark.accuracy <= 0.1 && (
                  <span style={{
                    position: 'absolute',
                    left: '8px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    fontSize: '11px',
                    color: '#888'
                  }}>
                    {(benchmark.accuracy * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              
              <div style={{ 
                minWidth: '60px', 
                textAlign: 'right',
                fontSize: '11px',
                color: '#888'
              }}>
                {benchmark.date}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#2a2a2a',
        borderRadius: '5px',
        fontSize: '13px',
        color: '#ccc'
      }}>
        <strong style={{ color: '#7FDBFF' }}>💡 Contexto:</strong>
        <ul style={{ margin: '10px 0 0 20px' }}>
          <li>Los humanos alcanzan ~85% de precisión en promedio</li>
          <li>El mejor LLM (GPT-4o) solo alcanza 21%</li>
          <li>ARC Prize ofrece $1M+ para superar 85%</li>
          <li>PROTEUS usa razonamiento topológico, no pattern matching</li>
        </ul>
      </div>
    </div>
  );
};