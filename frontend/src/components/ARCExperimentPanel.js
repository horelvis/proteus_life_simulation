/**
 * Panel de control para experimentos ARC
 */

import React, { useState } from 'react';
import { ARCExperiment } from '../simulation/ARCExperiment';
import { ARCVisualization } from './ARCVisualization';
import { OrganismAC } from '../simulation/OrganismAC';
import { ARCReasoningVisualization } from './ARCReasoningVisualization';
import { ARCSolver } from '../simulation/ARCSolver';

// CSS para animaciones
const animations = `
  @keyframes pulse {
    0% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.8;
      transform: scale(1.05);
    }
    100% {
      opacity: 1;
      transform: scale(1);
    }
  }
  
  @keyframes progress {
    0% {
      transform: translateX(-100%);
    }
    50% {
      transform: translateX(0);
    }
    100% {
      transform: translateX(100%);
    }
  }
`;

export const ARCExperimentPanel = () => {
  const [experimento, setExperimento] = useState(null);
  const [ejecutando, setEjecutando] = useState(false);
  const [resultados, setResultados] = useState(null);
  const [organismoSeleccionado, setOrganismoSeleccionado] = useState(null);
  const [puzzleSeleccionado, setPuzzleSeleccionado] = useState(null);
  const [mostrarRazonamiento, setMostrarRazonamiento] = useState(false);
  
  const iniciarExperimento = async () => {
    try {
      console.log('🚀 Iniciando experimento...');
      setEjecutando(true);
      setResultados(null);
      
      // Crear experimento
      const exp = new ARCExperiment();
      setExperimento(exp);
      
      // Cargar puzzles
      console.log('📊 Cargando puzzles...');
      await exp.cargarPuzzlesARC();
      
      // Crear población inicial
      console.log('🧬 Creando población...');
      const organismos = [];
      for (let i = 0; i < 20; i++) {
        organismos.push(new OrganismAC(
          Math.random() * 200,
          Math.random() * 200
        ));
      }
      
      // Seleccionar primer organismo para visualización
      setOrganismoSeleccionado(organismos[0]);
      
      // Ejecutar experimento
      console.log('🧪 Ejecutando experimento...');
      const reporte = await exp.ejecutarExperimento(organismos);
      
      console.log('✅ Experimento completado:', reporte);
      setResultados(reporte);
      setEjecutando(false);
      
      // Hacer scroll a los resultados
      setTimeout(() => {
        const resultadosElement = document.querySelector('#resultados-experimento');
        if (resultadosElement) {
          resultadosElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (error) {
      console.error('❌ Error en experimento:', error);
      setEjecutando(false);
      alert(`Error: ${error.message}`);
    }
  };
  
  return (
    <div style={{ 
      backgroundColor: '#1a1a1a', 
      color: 'white', 
      padding: '20px',
      minHeight: '100vh'
    }}>
      <style>{animations}</style>
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '10px' }}>
          🧠 Laboratorio de Razonamiento Transparente
        </h1>
        <p style={{ fontSize: '1.2rem', opacity: 0.8, maxWidth: '800px', margin: '0 auto' }}>
          Observa cómo un sistema basado en reglas locales simples puede resolver puzzles 
          complejos sin usar redes neuronales. Cada paso del razonamiento es visible y explicable.
        </p>
      </div>
      
      {!resultados && !ejecutando && (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}>
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '20px',
            borderRadius: '8px',
            border: '2px solid #0074D9'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#0074D9' }}>📊 ¿Qué es ARC?</h3>
            <p style={{ fontSize: '14px', margin: 0 }}>
              El Abstraction and Reasoning Corpus es un benchmark de IA que evalúa 
              la capacidad de razonamiento abstracto mediante puzzles visuales.
            </p>
          </div>
          
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '20px',
            borderRadius: '8px',
            border: '2px solid #2ECC40'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#2ECC40' }}>🔬 Nuestro Enfoque</h3>
            <p style={{ fontSize: '14px', margin: 0 }}>
              Usamos autómatas celulares con reglas locales simples. 
              No hay pesos ocultos ni capas profundas, solo lógica transparente.
            </p>
          </div>
          
          <div style={{
            backgroundColor: '#2a2a2a',
            padding: '20px',
            borderRadius: '8px',
            border: '2px solid #FF851B'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#FF851B' }}>✨ Resultados</h3>
            <p style={{ fontSize: '14px', margin: 0 }}>
              Sistema experimental con resultados variables. 
              Énfasis en transparencia e interpretabilidad completa.
            </p>
          </div>
        </div>
      )}
      
      <div style={{ marginBottom: '20px', textAlign: 'center' }}>
        <button 
          onClick={iniciarExperimento}
          disabled={ejecutando}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: ejecutando ? '#FF4136' : '#2ECC40',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: ejecutando ? 'not-allowed' : 'pointer',
            transition: 'all 0.3s ease',
            animation: ejecutando ? 'pulse 1.5s infinite' : 'none',
            marginRight: '10px'
          }}
        >
          {ejecutando ? '⏳ Experimento en proceso... (puede tomar unos segundos)' : '🚀 Iniciar Experimento'}
        </button>
        
        <button 
          onClick={async () => {
            const { runTests } = await import('../test/TestARCSolver.js');
            runTests();
          }}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#FF851B',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          🔍 Verificar Integridad
        </button>
      </div>
      
      {ejecutando && (
        <div style={{
          backgroundColor: '#2a2a2a',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <div style={{
            fontSize: '24px',
            marginBottom: '10px',
            color: '#FF851B'
          }}>
            🔬 Analizando puzzles ARC...
          </div>
          <div style={{
            fontSize: '14px',
            color: '#aaa'
          }}>
            El sistema está aplicando reglas de autómatas celulares para resolver puzzles de razonamiento.
          </div>
          <div style={{
            marginTop: '20px',
            height: '4px',
            backgroundColor: '#333',
            borderRadius: '2px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              backgroundColor: '#FF851B',
              width: '100%',
              animation: 'progress 2s ease-in-out infinite'
            }}></div>
          </div>
        </div>
      )}
      
      {experimento && !ejecutando && (
        <ARCVisualization 
          experimento={experimento}
          organismo={organismoSeleccionado}
        />
      )}
      
      {resultados && (
        <div id="resultados-experimento" style={{ marginTop: '40px' }}>
          <h2>📊 Resultados del Experimento</h2>
          
          {/* Botón para mostrar razonamiento */}
          <div style={{ marginBottom: '20px' }}>
            <button
              onClick={() => setMostrarRazonamiento(!mostrarRazonamiento)}
              style={{
                padding: '10px 20px',
                fontSize: '16px',
                backgroundColor: '#B10DC9',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer'
              }}
            >
              {mostrarRazonamiento ? '📊 Ver Resultados' : '🧠 Ver Razonamiento Paso a Paso'}
            </button>
          </div>
          
          {mostrarRazonamiento && experimento?.puzzles && (
            <div style={{ marginBottom: '40px' }}>
              <h3>Selecciona un puzzle para ver su proceso de razonamiento:</h3>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '20px' }}>
                {experimento.puzzles.map((puzzle, idx) => {
                  const resultado = resultados.arc.puzzlesResueltos > 0;
                  return (
                    <button
                      key={puzzle.id}
                      onClick={() => setPuzzleSeleccionado(puzzle)}
                      style={{
                        padding: '8px 16px',
                        backgroundColor: puzzleSeleccionado?.id === puzzle.id ? '#2ECC40' : '#444',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '14px'
                      }}
                    >
                      {puzzle.id} ({puzzle.category})
                    </button>
                  );
                })}
              </div>
              
              {puzzleSeleccionado && (
                <ARCReasoningVisualization
                  puzzle={puzzleSeleccionado}
                  solver={(() => {
                    const solver = new ARCSolver();
                    // Asegurarse de que el solver aprenda de los ejemplos
                    if (puzzleSeleccionado.trainExamples) {
                      solver.aprender(puzzleSeleccionado.trainExamples);
                    }
                    return solver;
                  })()}
                  onComplete={() => console.log('Visualización completada')}
                />
              )}
            </div>
          )}
          
          <div style={{ 
            backgroundColor: '#2a2a2a', 
            padding: '20px', 
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            <h2 style={{ 
              textAlign: 'center',
              fontSize: '48px',
              margin: '20px 0'
            }}>
              {resultados.porcentajeRazonamiento?.toFixed(1) || 0}%
            </h2>
            <h3 style={{ 
              textAlign: 'center',
              color: resultados.porcentajeRazonamiento > 50 ? '#2ECC40' : 
                     resultados.porcentajeRazonamiento > 20 ? '#FF851B' : '#FF4136'
            }}>
              {resultados.nivelRazonamiento || resultados.conclusión}
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <h4>Métricas Generales</h4>
                <ul>
                  <li>Tasa de resolución: {(resultados.tasaResolución * 100).toFixed(1)}%</li>
                  <li>Tiempo promedio: {(resultados.tiempoPromedio / 1000).toFixed(1)}s</li>
                  <li>Puzzles resueltos: {resultados.arc.puzzlesResueltos}/{resultados.arc.totalPuzzles}</li>
                </ul>
              </div>
              
              <div>
                <h4>Comparación con Controles</h4>
                <ul>
                  <li>vs Random: {resultados.vsRandom.toFixed(1)}x mejor</li>
                  <li>vs Sin evolución: {resultados.vsSinEvolución.toFixed(1)}x mejor</li>
                </ul>
              </div>
            </div>
          </div>
          
          {resultados.reglaCríticas && resultados.reglaCríticas.length > 0 && (
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '20px', 
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              <h4>🔬 Reglas Críticas (Test de Ablación)</h4>
              <p>Estas reglas son esenciales para el razonamiento:</p>
              <ul>
                {resultados.reglaCríticas.map((regla, i) => (
                  <li key={i} style={{ color: '#FF851B' }}>{regla}</li>
                ))}
              </ul>
            </div>
          )}
          
          {resultados.capacidadesDetectadas && resultados.capacidadesDetectadas.length > 0 && (
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '20px', 
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              <h4>🧠 Capacidades de Razonamiento Detectadas</h4>
              {resultados.capacidadesDetectadas.map((cap, i) => (
                <div key={i} style={{ marginBottom: '5px', fontSize: '14px' }}>
                  <strong>{cap.capacidad}:</strong> {cap.nivel}
                </div>
              ))}
            </div>
          )}
          
          {resultados.porCategoría && (
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '20px', 
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              <h4>📈 Análisis por Tipo de Razonamiento</h4>
              {Object.entries(resultados.porCategoría).map(([cat, data]) => {
                if (data.total > 0) {
                  const porcentaje = data.porcentaje || 0;
                  return (
                    <div key={cat} style={{ marginBottom: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                        <span style={{ fontSize: '14px' }}>{data.nombre}</span>
                        <span style={{ fontSize: '14px' }}>{porcentaje.toFixed(0)}%</span>
                      </div>
                      <div style={{ 
                        width: '100%', 
                        height: '20px', 
                        backgroundColor: '#333',
                        borderRadius: '10px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${porcentaje}%`,
                          height: '100%',
                          backgroundColor: porcentaje > 60 ? '#2ECC40' : 
                                         porcentaje > 30 ? '#FF851B' : '#FF4136',
                          transition: 'width 0.5s'
                        }}></div>
                      </div>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          )}
          
          {resultados.comparación && (
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '20px', 
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              <h4>🔍 Comparación con Benchmarks</h4>
              <ul style={{ fontSize: '14px' }}>
                <li>vs Random Walk: {resultados.comparación.vsRandom.toFixed(1)}x mejor</li>
                <li>vs GPT-4: {(resultados.comparación.vsGPT4 * 100).toFixed(0)}% del rendimiento</li>
                <li>vs Humano Promedio: {(resultados.comparación.vsHumanoPromedio * 100).toFixed(0)}% del rendimiento</li>
              </ul>
            </div>
          )}
          
          {resultados.recomendaciones && resultados.recomendaciones.length > 0 && (
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '20px', 
              borderRadius: '8px'
            }}>
              <h4>💡 Recomendaciones para Mejorar</h4>
              {resultados.recomendaciones.map((rec, i) => (
                <div key={i} style={{ marginBottom: '5px', fontSize: '14px' }}>
                  • {rec.sugerencia}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      <div style={{ marginTop: '40px', fontSize: '14px', opacity: 0.7 }}>
        <h3>Criterios de Validación</h3>
        <ul>
          <li>✅ Zero-shot: Sin ver las soluciones previamente</li>
          <li>✅ Generalización: Debe resolver puzzles nuevos</li>
          <li>✅ Ablación: Identificar componentes esenciales</li>
          <li>✅ Comparación: Mejor que random y baseline</li>
          <li>✅ Transparencia: Todo el proceso es observable</li>
        </ul>
      </div>
    </div>
  );
};