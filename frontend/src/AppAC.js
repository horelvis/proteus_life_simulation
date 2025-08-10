/**
 * Simplified PROTEUS App with Cellular Automaton
 */

import React, { useState, useEffect, useRef } from 'react';
import { SimulationAC } from './simulation/SimulationAC';
import { SimulationACCanvas } from './components/SimulationACCanvas';
import { ARCExperimentPanel } from './components/ARCExperimentPanel';
import { ARCExperimentPythonPanel } from './components/ARCExperimentPythonPanel';

function AppAC() {
  const [mostrarExperimento, setMostrarExperimento] = useState(true); // Ahora inicia en true
  const [usarBackendPython, setUsarBackendPython] = useState(false);
  const [simulation] = useState(() => new SimulationAC({ width: 800, height: 600 }));
  const [isRunning, setIsRunning] = useState(false);
  const [statistics, setStatistics] = useState({});
  const statsInterval = useRef(null);

  useEffect(() => {
    // Initialize simulation
    simulation.initialize({
      initialOrganisms: 20,
      initialNutrients: 30,  // Reduced from 50
      initialPredators: 1
    });

    // Update statistics periodically
    statsInterval.current = setInterval(() => {
      setStatistics(simulation.getStatistics());
    }, 100);

    return () => {
      if (statsInterval.current) {
        clearInterval(statsInterval.current);
      }
      simulation.stop();
    };
  }, [simulation]);

  const handleStartStop = () => {
    if (isRunning) {
      simulation.stop();
    } else {
      simulation.start();
    }
    setIsRunning(!isRunning);
  };

  const handleReset = () => {
    simulation.stop();
    simulation.initialize({
      initialOrganisms: 20,
      initialNutrients: 30,  // Reduced from 50
      initialPredators: 1
    });
    setIsRunning(false);
  };

  return (
    <div className="App" style={{ 
      backgroundColor: '#1a1a1a', 
      minHeight: '100vh', 
      color: 'white',
      overflow: 'auto'
    }}>
      <header style={{ padding: '10px 20px' }}>
        <h1 style={{ margin: '10px 0' }}>PROTEUS-AC: Razonamiento Emergente sin Redes Neuronales</h1>
        <p style={{ margin: '5px 0' }}>Resolviendo puzzles ARC con autómatas celulares transparentes</p>
        <button
          onClick={() => setMostrarExperimento(!mostrarExperimento)}
          style={{
            marginTop: '10px',
            padding: '8px 16px',
            backgroundColor: mostrarExperimento ? '#FF4136' : '#2ECC40',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          {mostrarExperimento ? '🔬 Ver Simulación de Organismos' : '🧠 Volver a Experimento ARC'}
        </button>
        
        {mostrarExperimento && (
          <button
            onClick={() => setUsarBackendPython(!usarBackendPython)}
            style={{
              marginTop: '10px',
              marginLeft: '10px',
              padding: '8px 16px',
              backgroundColor: usarBackendPython ? '#B10DC9' : '#FF851B',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            {usarBackendPython ? '🐍 Backend Python Activo' : '🌐 Usar Backend Python'}
          </button>
        )}
      </header>

      {mostrarExperimento ? (
        usarBackendPython ? <ARCExperimentPythonPanel /> : <ARCExperimentPanel />
      ) : (
        <>
          <div style={{ display: 'flex', gap: '20px', padding: '0 20px 10px 20px' }}>
            {/* Main simulation canvas */}
            <div style={{ flexShrink: 0 }}>
              <SimulationACCanvas 
                simulation={simulation} 
                worldSize={{ width: 800, height: 600 }} 
              />
              
              {/* Controls */}
              <div style={{ marginTop: '10px' }}>
                <button onClick={handleStartStop} style={{ marginRight: '10px' }}>
                  {isRunning ? 'Pause' : 'Start'}
                </button>
                <button onClick={handleReset}>
                  Reset
                </button>
              </div>
            </div>

            {/* Statistics panel */}
            <div style={{ 
              backgroundColor: '#2a2a2a', 
              padding: '15px', 
              borderRadius: '8px', 
              minWidth: '280px',
              maxHeight: '620px',
              overflowY: 'auto'
            }}>
              <h2 style={{ margin: '0 0 10px 0' }}>Statistics</h2>
          
          <div className="stat-group" style={{ marginBottom: '15px' }}>
            <h3 style={{ margin: '5px 0', fontSize: '16px' }}>Population</h3>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Organisms: {statistics.totalOrganisms || 0}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Nutrients: {statistics.nutrients || 0}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Predators: {statistics.predators || 0}</p>
          </div>

          <div className="stat-group" style={{ marginBottom: '15px' }}>
            <h3 style={{ margin: '5px 0', fontSize: '16px' }}>Evolution</h3>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Avg Generation: {(statistics.averageGeneration || 0).toFixed(1)}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Total Births: {statistics.births || 0}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Total Deaths: {statistics.deaths || 0}</p>
          </div>

          <div className="stat-group" style={{ marginBottom: '15px' }}>
            <h3 style={{ margin: '5px 0', fontSize: '16px' }}>Metrics</h3>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Avg Energy: {(statistics.averageEnergy || 0).toFixed(2)}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Morphological Stability: {(statistics.morphologicalStability || 0).toFixed(2)}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Foraging Efficiency: {(statistics.foragingEfficiency || 0).toFixed(4)}</p>
          </div>

          <div className="stat-group" style={{ marginBottom: '15px' }}>
            <h3 style={{ margin: '5px 0', fontSize: '16px' }}>Social Behavior</h3>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Cooperation Index: {(statistics.cooperationIndex || 0).toFixed(2)}</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Competition Index: {(statistics.competitionIndex || 0).toFixed(2)}</p>
          </div>

          <div className="stat-group">
            <h3 style={{ margin: '5px 0', fontSize: '16px' }}>Performance</h3>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>Time: {(statistics.time || 0).toFixed(1)}s</p>
            <p style={{ margin: '3px 0', fontSize: '14px' }}>FPS: {(statistics.fps || 0).toFixed(1)}</p>
          </div>
        </div>
      </div>

      {/* Info panel */}
      <div style={{ padding: '10px 20px 20px 20px' }}>
        <h2 style={{ margin: '10px 0' }}>How it works</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div>
            <h3 style={{ margin: '10px 0', fontSize: '16px' }}>Visual Elements:</h3>
            <ul style={{ fontSize: '13px', margin: '5px 0', paddingLeft: '20px' }}>
              <li><strong>Green circles:</strong> Nutrients (food)</li>
              <li><strong>Red triangles:</strong> Predators (emit light flashes)</li>
              <li><strong>Colored circles:</strong> Organisms (color = generation, brightness = energy)</li>
              <li><strong>White lines:</strong> Movement vectors from CA decision</li>
            </ul>
            
            <h3 style={{ margin: '10px 0', fontSize: '16px' }}>Pheromone Trails (if visible):</h3>
            <ul style={{ fontSize: '13px', margin: '5px 0', paddingLeft: '20px' }}>
              <li><strong>Green clouds:</strong> Food trails (cooperation)</li>
              <li><strong>Red clouds:</strong> Danger signals (avoidance)</li>
              <li><strong>Blue clouds:</strong> Colony markers (aggregation)</li>
              <li><strong>Purple clouds:</strong> Mating signals (reproduction)</li>
            </ul>
          </div>
          
          <div>
            <h3 style={{ margin: '10px 0', fontSize: '16px' }}>Cellular Automaton Rules:</h3>
            <ul style={{ fontSize: '13px', margin: '5px 0', paddingLeft: '20px' }}>
              <li><strong>Chemotaxis:</strong> Cells activate toward chemical gradients</li>
              <li><strong>Phototaxis:</strong> Cells activate away from light</li>
              <li><strong>Homeostasis:</strong> Energy diffusion, tissue repair</li>
              <li><strong>Morphogenesis:</strong> Dynamic skin formation</li>
              <li><strong>Oscillator:</strong> Central pattern generator</li>
              <li><strong>Stigmergy:</strong> Indirect communication via pheromones</li>
            </ul>
            
            <h3 style={{ margin: '10px 0', fontSize: '16px' }}>Cell States in AC Grid:</h3>
            <ul style={{ fontSize: '13px', margin: '5px 0', paddingLeft: '20px' }}>
              <li><strong>Dark gray:</strong> Void (dead cells)</li>
              <li><strong>Pink:</strong> Scar tissue (healing)</li>
              <li><strong>Brown:</strong> Skin (protective barrier)</li>
              <li><strong>Red-green:</strong> Active tissue (energy/activation)</li>
            </ul>
          </div>
        </div>
        
        <p style={{ marginTop: '15px', fontWeight: 'bold', fontSize: '14px' }}>
          Click on any organism to see its internal 16×16 CA state!
        </p>
      </div>
      </>
      )}
    </div>
  );
}

export default AppAC;