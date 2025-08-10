import React, { useState } from 'react';
import styled from 'styled-components';

const Panel = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid var(--bg-tertiary);
`;

const Title = styled.h2`
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--text-primary);
`;

const ButtonGroup = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
  margin-bottom: 1rem;
`;

const Button = styled.button`
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  
  ${props => props.$primary && `
    background-color: var(--accent-primary);
    color: #000000;
    
    &:hover {
      background-color: #00B8BB;
    }
  `}
  
  ${props => props.$secondary && `
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
    
    &:hover {
      background-color: #3a3a3a;
    }
  `}
  
  ${props => props.$danger && `
    background-color: var(--danger);
    color: var(--text-primary);
    
    &:hover {
      background-color: #DD3333;
    }
  `}
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const SpeedControl = styled.div`
  margin-top: 1rem;
`;

const SpeedMarkers = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 0.25rem;
  font-size: 0.625rem;
  color: var(--text-tertiary);
  user-select: none;
`;

const SpeedHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
`;

const Label = styled.label`
  font-size: 0.75rem;
  color: var(--text-secondary);
`;

const SpeedValue = styled.span`
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--accent-primary);
`;

const Slider = styled.input`
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: var(--bg-tertiary);
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent-primary);
    cursor: pointer;
  }
  
  &::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent-primary);
    cursor: pointer;
    border: none;
  }
`;

const ControlPanel = ({ status, onStart, onPause, onStop, onReset, onSpeedChange, onShowReport, initialSpeed = 1.0 }) => {
  const isRunning = status === 'running';
  const isStopped = status === 'stopped' || status === 'created';
  
  const [speed, setSpeed] = useState(initialSpeed);
  
  const handleSpeedChange = (e) => {
    const newSpeed = parseFloat(e.target.value);
    setSpeed(newSpeed);
    if (onSpeedChange) {
      onSpeedChange(newSpeed);
    }
  };

  return (
    <Panel>
      <Title>Simulation Controls</Title>
      
      <ButtonGroup>
        <Button 
          $primary
          onClick={onStart}
          disabled={isRunning}
        >
          {isStopped ? 'Start' : 'Resume'}
        </Button>
        
        <Button 
          $secondary
          onClick={onPause}
          disabled={!isRunning}
        >
          Pause
        </Button>
        
        <Button 
          $danger
          onClick={onStop}
          disabled={isStopped}
        >
          Stop
        </Button>
        
        <Button 
          $secondary
          onClick={onReset}
        >
          Reset
        </Button>
      </ButtonGroup>
      
      {onShowReport && (
        <Button 
          $secondary
          onClick={onShowReport}
          style={{ width: '100%', marginTop: '0.5rem' }}
        >
          Show Report 📊
        </Button>
      )}
      
      <SpeedControl>
        <SpeedHeader>
          <Label>Simulation Speed</Label>
          <SpeedValue>{speed.toFixed(1)}x</SpeedValue>
        </SpeedHeader>
        <Slider 
          type="range" 
          min="0.1" 
          max="2" 
          step="0.1" 
          value={speed}
          onChange={handleSpeedChange}
          disabled={!isRunning}
        />
        <SpeedMarkers>
          <span>0.1x</span>
          <span>1.0x</span>
          <span>2.0x</span>
        </SpeedMarkers>
      </SpeedControl>
    </Panel>
  );
};

export default ControlPanel;