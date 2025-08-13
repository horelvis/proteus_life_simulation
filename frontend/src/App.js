import React, { useState } from 'react';
import styled from 'styled-components';
import AppBackend from './AppBackend';
import AppAC from './AppAC';
import AppARCReal from './AppARCReal';
import ARCSolvingProcess from './components/ARCSolvingProcess';
import ARCProcessAnimation from './components/ARCProcessAnimation';
import ARCTopologicalView from './components/ARCTopologicalView';

const ModeSelector = styled.div`
  position: fixed;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  background-color: var(--bg-secondary);
  padding: 0.5rem;
  border-radius: 8px;
  border: 1px solid var(--bg-tertiary);
  display: flex;
  gap: 0.5rem;
  z-index: 1000;
`;

const ModeButton = styled.button`
  padding: 0.5rem 1rem;
  background-color: ${props => props.$active ? 'var(--accent-primary)' : 'transparent'};
  color: ${props => props.$active ? 'var(--bg-primary)' : 'var(--text-primary)'};
  border: 1px solid ${props => props.$active ? 'var(--accent-primary)' : 'var(--bg-tertiary)'};
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 0.875rem;
  
  &:hover {
    background-color: ${props => props.$active ? 'var(--accent-primary)' : 'var(--bg-tertiary)'};
  }
`;

function App() {
  const [mode, setMode] = useState('real'); // Ahora inicia en modo REAL

  return (
    <>
      {mode === 'real' && <AppARCReal />}
      {mode === 'backend' && <AppBackend />}
      {mode === 'ac' && <AppAC />}
      {mode === 'solving' && <ARCSolvingProcess />}
      {mode === 'topological' && <ARCTopologicalView />}
      <ModeSelector>
        <ModeButton 
          $active={mode === 'real'} 
          onClick={() => setMode('real')}
          style={{ backgroundColor: mode === 'real' ? '#ff3333' : 'transparent' }}
        >
          🔬 ARC Real (Sin Simulación)
        </ModeButton>
        <ModeButton 
          $active={mode === 'backend'} 
          onClick={() => setMode('backend')}
        >
          Simulation (Python Backend)
        </ModeButton>
        <ModeButton 
          $active={mode === 'ac'} 
          onClick={() => setMode('ac')}
        >
          AC Mode (ARC)
        </ModeButton>
        <ModeButton 
          $active={mode === 'solving'} 
          onClick={() => setMode('solving')}
        >
          ARC Solving Process
        </ModeButton>
        <ModeButton 
          $active={mode === 'topological'} 
          onClick={() => setMode('topological')}
        >
          Topological Solver
        </ModeButton>
      </ModeSelector>
    </>
  );
}

export default App;