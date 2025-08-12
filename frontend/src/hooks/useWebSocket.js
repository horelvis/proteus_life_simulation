import { useEffect, useState, useRef } from 'react';
import { WS_URL as WS_BASE_URL } from '../config';

export const useWebSocket = (simulationId) => {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);

  useEffect(() => {
    if (!simulationId) {
      // Close existing connection if simulationId is cleared
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const connect = () => {
      try {
        const ws = new WebSocket(`${WS_BASE_URL}/${simulationId}`);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('WebSocket connected');
          setConnected(true);
          reconnectAttempts.current = 0;
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleMessage(data);
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = (event) => {
          console.log('WebSocket disconnected', event.code, event.reason);
          setConnected(false);
          wsRef.current = null;

          // Don't reconnect if the server rejected the connection (simulation not found)
          if (event.code === 1000 || event.code === 1001) {
            return;
          }

          // Attempt to reconnect with exponential backoff
          if (reconnectAttempts.current < 5) {
            const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current++;
              connect();
            }, timeout);
          }
        };
      } catch (err) {
        console.error('Failed to create WebSocket:', err);
      }
    };

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [simulationId]);

  const handleMessage = (data) => {
    const { type, data: payload } = data;

    switch (type) {
      case 'initial_state':
        handleInitialState(payload);
        break;
      case 'update':
        handleUpdate(payload);
        break;
      default:
        console.log('Unknown message type:', type);
    }
  };

  const handleInitialState = (state) => {
    // Get the simulation context from the parent component
    const simulationContext = window.__simulationContext;
    if (simulationContext) {
      simulationContext.setOrganisms(state.organisms || []);
      simulationContext.setPredators(state.predators || []);
      simulationContext.setNutrients(state.nutrients || []);
    }
  };

  const handleUpdate = (update) => {
    const simulationContext = window.__simulationContext;
    if (simulationContext) {
      simulationContext.setOrganisms(update.organisms || []);
      simulationContext.setPredators(update.predators || []);
      simulationContext.setNutrients(update.nutrients || []);
      
      // Handle events
      if (update.events && update.events.length > 0) {
        update.events.forEach(event => {
          handleEvent(event);
        });
      }
    }
  };

  const handleEvent = (event) => {
    // Log important events
    switch (event.type) {
      case 'birth':
        console.log(`New organism born: Generation ${event.generation}`);
        break;
      case 'death':
        console.log(`Organism died: ${event.cause}`);
        break;
      case 'extinction':
        console.warn(`EXTINCTION at generation ${event.generation}`);
        break;
      case 'feeding':
        console.log(`Organism fed: gained ${event.energy_gained} energy`);
        break;
      default:
        break;
    }
  };

  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  return {
    connected,
    sendMessage
  };
};
