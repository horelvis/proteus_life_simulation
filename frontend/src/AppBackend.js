import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import SimulationCanvas from './components/SimulationCanvas';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';
import OrganismInfo from './components/OrganismInfo';

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

const ConnectionStatus = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  background-color: rgba(0, 0, 0, 0.8);
  padding: 0.5rem 1rem;
  border-radius: 4px;
  font-size: 0.875rem;
  color: ${props => props.$connected ? 'var(--success)' : 'var(--danger)'};
`;

function AppBackend() {
  const [selectedOrganismId, setSelectedOrganismId] = useState(null);
  const [simulationState, setSimulationState] = useState(null);
  const [status, setStatus] = useState('stopped');
  const [connected, setConnected] = useState(false);
  const [backendStats, setBackendStats] = useState(null);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const worldSize = { width: 1600, height: 1200 };

  useEffect(() => {
    // Connect to WebSocket
    const connectWebSocket = () => {
      const clientId = `web-${Date.now()}`;
      const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
      
      ws.onopen = () => {
        console.log('Connected to PROTEUS backend');
        setConnected(true);
        setStatus('running');
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'update') {
            // Convert backend format to frontend format
            const state = {
              organisms: message.data.organisms.map(org => ({
                id: org.id,
                position: { x: org.position[0], y: org.position[1] },
                velocity: { x: 0, y: 0 }, // Backend doesn't send velocity
                energy: org.energy,
                generation: org.generation,
                age: org.age,
                alive: true,
                color: `hsl(${180 + org.generation * 10}, 70%, 60%)`,
                trajectory: []
              })),
              predators: [], // Backend will send if implemented
              nutrients: [], // Backend will send if implemented
              safeZones: [
                { x: worldSize.width * 0.2, y: worldSize.height * 0.3, radius: 120 },
                { x: worldSize.width * 0.8, y: worldSize.height * 0.7, radius: 120 }
              ],
              environmentalField: [],
              memoryAnchors: [],
              statistics: message.data.stats
            };
            
            setSimulationState(state);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.onclose = () => {
        console.log('Disconnected from PROTEUS backend');
        setConnected(false);
        setStatus('stopped');
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current = ws;
    };
    
    // Fetch backend stats periodically
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/stats');
        const stats = await response.json();
        setBackendStats(stats);
      } catch (error) {
        console.error('Error fetching backend stats:', error);
      }
    };
    
    connectWebSocket();
    const statsInterval = setInterval(fetchStats, 1000);
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      clearInterval(statsInterval);
    };
  }, []);

  const handleStart = () => {
    // Backend is always running when connected
    setStatus('running');
  };

  const handlePause = () => {
    // Can't pause backend from frontend yet
    setStatus('paused');
  };

  const handleStop = () => {
    // Can't stop backend from frontend yet
    setStatus('stopped');
  };

  const handleSpawnOrganism = (x, y) => {
    // Send spawn command to backend (not implemented yet)
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'spawn',
        position: [x, y]
      }));
    }
  };

  const selectedOrganism = simulationState?.organisms.find(
    o => o.id === selectedOrganismId
  );

  return (
    <AppContainer>
      <MainContent>
        <Header>
          <Title>PROTEUS - GPU Backend</Title>
          <StatusIndicator>
            <StatusDot $active={connected} />
            <span>{connected ? 'Backend Connected' : 'Backend Disconnected'}</span>
            {backendStats && (
              <span> • {backendStats.organisms}/{backendStats.max_organisms} organisms</span>
            )}
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
            onOrganismClick={setSelectedOrganismId}
            onCanvasClick={handleSpawnOrganism}
          />
          <ConnectionStatus $connected={connected}>
            {connected ? '🟢 Backend Connected' : '🔴 Backend Disconnected'}
          </ConnectionStatus>
        </CanvasContainer>
      </MainContent>
      <Sidebar>
        <ControlPanel
          status={status}
          onStart={handleStart}
          onPause={handlePause}
          onStop={handleStop}
        />
        <StatsPanel statistics={simulationState?.statistics || {}} />
        {selectedOrganism && (
          <OrganismInfo organism={selectedOrganism} />
        )}
      </Sidebar>
    </AppContainer>
  );
}

export default AppBackend;