/**
 * PROTEUS ARC Mode - Python Backend Only
 */

import React from 'react';
import { ARCExperimentPythonPanel } from './components/ARCExperimentPythonPanel';

function AppAC() {
  return (
    <div className="App" style={{ 
      backgroundColor: '#1a1a1a', 
      minHeight: '100vh', 
      color: 'white',
      overflow: 'auto'
    }}>
      <header style={{ padding: '10px 20px' }}>
        <h1 style={{ margin: '10px 0' }}>PROTEUS-AC: Razonamiento Emergente sin Redes Neuronales</h1>
        <p style={{ margin: '5px 0' }}>Resolviendo puzzles ARC con dinámicas topológicas transparentes</p>
        <p style={{ 
          margin: '10px 0', 
          padding: '10px', 
          backgroundColor: '#2a2a2a', 
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          <strong>🐍 Backend Python Activo</strong> - Conectando con servidor WebSocket en ws://localhost:8765
        </p>
      </header>

      <ARCExperimentPythonPanel />
    </div>
  );
}

export default AppAC;