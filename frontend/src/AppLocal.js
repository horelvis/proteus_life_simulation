import React, { useState, useEffect, useRef, useCallback } from 'react';
import styled from 'styled-components';
import SimulationCanvas from './components/SimulationCanvas';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';
import OrganismInfo from './components/OrganismInfo';
import SimulationReport from './components/SimulationReport';
import MathematicsPanel from './components/MathematicsPanel';
import { EvolutionMetrics } from './components/EvolutionMetrics';
import { Simulation } from './simulation/Simulation';

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: var(--bg-primary);
  overflow: hidden;
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
`;

const Header = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 2rem;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--bg-tertiary);
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent-primary);
  letter-spacing: 0.05em;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-secondary);
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => props.$active ? 'var(--success)' : 'var(--danger)'};
`;

const CanvasContainer = styled.div`
  flex: 1;
  position: relative;
  background-color: var(--bg-primary);
`;

const Sidebar = styled.aside`
  width: 300px;
  background-color: var(--bg-secondary);
  border-left: 1px solid var(--bg-tertiary);
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const BottomPanel = styled.div`
  position: fixed;
  bottom: 0;
  left: 0;
  right: 300px;
  background-color: var(--bg-secondary);
  border-top: 1px solid var(--bg-tertiary);
  max-height: 400px;
  transition: transform 0.3s ease-out;
  transform: ${props => props.$show ? 'translateY(0)' : 'translateY(100%)'};
  z-index: 100;
`;

const ToggleButton = styled.button`
  position: fixed;
  bottom: 10px;
  left: ${props => props.$left || '10px'};
  background-color: var(--accent-primary);
  color: var(--bg-primary);
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  z-index: 101;
  
  &:hover {
    background-color: var(--accent-secondary);
  }
`;

function AppLocal() {
  const [selectedOrganismId, setSelectedOrganismId] = useState(null);
  const [simulationState, setSimulationState] = useState(null);
  const [status, setStatus] = useState('stopped');
  const [showReport, setShowReport] = useState(false);
  const [simulationReport, setSimulationReport] = useState(null);
  const [showMathPanel, setShowMathPanel] = useState(false);
  const [showEvolutionMetrics, setShowEvolutionMetrics] = useState(false);
  const simulationRef = useRef(null);
  const animationFrameRef = useRef(null);
  const updateLoopRef = useRef(null);
  const initializedRef = useRef(false);

  const worldSize = { width: 800, height: 600 };

  useEffect(() => {
    // Prevent double initialization in StrictMode
    if (initializedRef.current) {
      return;
    }
    initializedRef.current = true;
    
    // Cleanup any existing simulation first
    if (simulationRef.current) {
      simulationRef.current.stop();
      simulationRef.current = null;
    }
    
    // Create simulation instance
    const sim = new Simulation(worldSize);
    simulationRef.current = sim;
    sim.initialize();
    
    // Get initial state to display
    const initialState = sim.getState();
    setSimulationState(initialState);
    
    // Don't auto-start - let user control it
    setStatus('stopped');
    
    return () => {
      // Cleanup on unmount
      if (updateLoopRef.current) {
        updateLoopRef.current.stop();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [worldSize]);

  // Separate update loop that can be started/stopped
  const startUpdateLoop = useCallback(() => {
    let frameCount = 0;
    let running = true;
    
    const updateState = () => {
      if (!running || !simulationRef.current) return;
      
      try {
        frameCount++;
        if (frameCount % 3 === 0) {
          const state = simulationRef.current.getState();
          setSimulationState(state);
        }
      } catch (error) {
        console.error('Error getting simulation state:', error);
      }
      
      if (running) {
        animationFrameRef.current = requestAnimationFrame(updateState);
      }
    };
    
    updateLoopRef.current = {
      stop: () => {
        running = false;
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      }
    };
    
    updateState();
  }, []);

  const handleStart = () => {
    console.log('Start button clicked. Current status:', status);
    if (simulationRef.current && status !== 'running') {
      console.log('Starting simulation...');
      simulationRef.current.start();
      startUpdateLoop(); // Start the UI update loop
      setStatus('running');
    }
  };

  const handlePause = () => {
    console.log('Pause button clicked. Current status:', status);
    if (simulationRef.current && status === 'running') {
      console.log('Pausing simulation...');
      simulationRef.current.stop();
      if (updateLoopRef.current) {
        updateLoopRef.current.stop(); // Stop the UI update loop
      }
      setStatus('paused');
    }
  };

  const handleStop = () => {
    console.log('Stop button clicked');
    if (simulationRef.current) {
      console.log('Stopping simulation...');
      simulationRef.current.stop();
      if (updateLoopRef.current) {
        updateLoopRef.current.stop(); // Stop the UI update loop
      }
      
      // Generate report before clearing
      const report = simulationRef.current.getSimulationReport();
      setSimulationReport(report);
      
      // Don't clear organisms immediately - preserve genetics
      setStatus('stopped');
    }
  };
  
  const handleReset = () => {
    console.log('Reset button clicked');
    if (simulationRef.current) {
      console.log('Resetting simulation...');
      simulationRef.current.stop(); // Stop first
      if (updateLoopRef.current) {
        updateLoopRef.current.stop(); // Stop the UI update loop
      }
      simulationRef.current.organisms = [];
      simulationRef.current.nutrients = [];
      simulationRef.current.predators = [];
      simulationRef.current.initialize();
      setStatus('stopped');
    }
  };

  const handleSpeedChange = (speed) => {
    if (simulationRef.current) {
      simulationRef.current.setSpeed(speed);
    }
  };

  const handleSpawnOrganism = (x, y) => {
    if (simulationRef.current && status === 'running') {
      // Spawn a new organism at the clicked position
      const newOrganism = simulationRef.current.addOrganism(x, y);
      
      // Update simulation state to reflect the new organism
      if (newOrganism) {
        setSimulationState(prevState => ({
          ...prevState,
          organisms: [...(prevState?.organisms || []), newOrganism],
          statistics: {
            ...prevState?.statistics,
            organismCount: (prevState?.statistics?.organismCount || 0) + 1
          }
        }));
      }
    }
  };

  const selectedOrganism = simulationState?.organisms.find(
    o => o.id === selectedOrganismId
  );

  return (
    <AppContainer>
      <MainContent>
        <Header>
          <Title>PROTEUS</Title>
          <StatusIndicator>
            <StatusDot $active={status === 'running'} />
            <span>{status === 'running' ? 'Simulation Running' : 'Simulation Paused'}</span>
          </StatusIndicator>
        </Header>
        <CanvasContainer>
          <SimulationCanvas
            organisms={simulationState?.organisms || []}
            predators={simulationState?.predators || []}
            nutrients={simulationState?.nutrients || []}
            safeZones={simulationState?.safeZones || []}
            environmentalField={simulationState?.environmentalField || []}
            memoryAnchors={simulationState?.memoryAnchors || []}
            worldSize={worldSize}
            selectedOrganismId={selectedOrganismId}
            onOrganismClick={(organism) => setSelectedOrganismId(organism.id)}
            onCanvasClick={handleSpawnOrganism}
          />
        </CanvasContainer>
      </MainContent>
      <Sidebar>
        <ControlPanel
          status={status}
          onStart={handleStart}
          onPause={handlePause}
          onStop={handleStop}
          onReset={handleReset}
          onSpeedChange={handleSpeedChange}
          onShowReport={() => setShowReport(true)}
        />
        <StatsPanel statistics={simulationState?.statistics || {}} />
        {selectedOrganism && (
          <OrganismInfo 
            organism={selectedOrganism} 
            onClose={() => setSelectedOrganismId(null)}
          />
        )}
      </Sidebar>
      
      <ToggleButton onClick={() => setShowMathPanel(!showMathPanel)}>
        {showMathPanel ? '📊 Hide Math' : '📊 Show Math'}
      </ToggleButton>
      
      <ToggleButton $left="150px" onClick={() => setShowEvolutionMetrics(!showEvolutionMetrics)}>
        {showEvolutionMetrics ? '🧬 Hide Evolution' : '🧬 Show Evolution'}
      </ToggleButton>
      
      <BottomPanel $show={showMathPanel}>
        <MathematicsPanel
          organisms={simulationState?.organisms || []}
          predators={simulationState?.predators || []}
          environmentalField={simulationState?.environmentalField || []}
          statistics={simulationState?.statistics || {}}
        />
      </BottomPanel>
      
      <BottomPanel $show={showEvolutionMetrics}>
        <EvolutionMetrics
          simulation={simulationRef.current}
        />
      </BottomPanel>
      
      {showReport && simulationReport && (
        <SimulationReport
          report={simulationReport}
          onClose={() => setShowReport(false)}
        />
      )}
    </AppContainer>
  );
}

export default AppLocal;