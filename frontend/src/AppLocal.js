import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import SimulationCanvas from './components/SimulationCanvas';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';
import OrganismInfo from './components/OrganismInfo';
import SimulationReport from './components/SimulationReport';
import MathematicsPanel from './components/MathematicsPanel';
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
  left: 10px;
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
  const simulationRef = useRef(null);
  const animationFrameRef = useRef(null);

  const worldSize = { width: 800, height: 600 };

  useEffect(() => {
    // Create simulation instance
    const sim = new Simulation(worldSize);
    simulationRef.current = sim;
    sim.initialize();
    
    // Update state periodically
    const updateState = () => {
      if (simulationRef.current) {
        setSimulationState(simulationRef.current.getState());
      }
      animationFrameRef.current = requestAnimationFrame(updateState);
    };
    
    updateState();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, []);

  const handleStart = () => {
    if (simulationRef.current && status !== 'running') {
      simulationRef.current.start();
      setStatus('running');
    }
  };

  const handlePause = () => {
    if (simulationRef.current && status === 'running') {
      simulationRef.current.stop();
      setStatus('paused');
    }
  };

  const handleStop = () => {
    if (simulationRef.current) {
      simulationRef.current.stop();
      
      // Generate report before clearing
      const report = simulationRef.current.getSimulationReport();
      setSimulationReport(report);
      
      // Don't clear organisms immediately - preserve genetics
      setStatus('stopped');
    }
  };
  
  const handleReset = () => {
    if (simulationRef.current) {
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
    // TODO: Implement spawn at click position
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
      
      <BottomPanel $show={showMathPanel}>
        <MathematicsPanel
          organisms={simulationState?.organisms || []}
          predators={simulationState?.predators || []}
          environmentalField={simulationState?.environmentalField || []}
          statistics={simulationState?.statistics || {}}
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