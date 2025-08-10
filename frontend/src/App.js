import React, { useState } from 'react';
import styled from 'styled-components';
import AppLocal from './AppLocal';
import AppBackend from './AppBackend';
import AppAC from './AppAC';

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
  const [mode, setMode] = useState('ac'); // Ahora inicia en modo AC

  return (
    <>
      {mode === 'local' ? <AppLocal /> : mode === 'backend' ? <AppBackend /> : <AppAC />}
      <ModeSelector>
        <ModeButton 
          $active={mode === 'local'} 
          onClick={() => setMode('local')}
        >
          Local Simulation
        </ModeButton>
        <ModeButton 
          $active={mode === 'backend'} 
          onClick={() => setMode('backend')}
        >
          GPU Backend (Vispy)
        </ModeButton>
        <ModeButton 
          $active={mode === 'ac'} 
          onClick={() => setMode('ac')}
        >
          AC Mode (ARC)
        </ModeButton>
      </ModeSelector>
    </>
  );
}

export default App;