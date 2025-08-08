import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const useSimulation = () => {
  const [simulationId, setSimulationId] = useState(null);
  const [status, setStatus] = useState('created');
  const [organisms, setOrganisms] = useState([]);
  const [predators, setPredators] = useState([]);
  const [nutrients, setNutrients] = useState([]);
  const [statistics, setStatistics] = useState({});
  const [error, setError] = useState(null);
  const [isCreating, setIsCreating] = useState(false);

  const createSimulation = useCallback(async () => {
    if (isCreating) return; // Prevent multiple simultaneous creations
    
    try {
      setIsCreating(true);
      const config = {
        world_size: [800, 600],
        initial_organisms: 30,  // Menos organismos iniciales
        initial_predators: 8,   // Más depredadores
        mutation_rate: 0.15,    // Más mutaciones
        crossover_rate: 0.3,
        enable_organs: true,
        physics_params: {
          viscosity: 0.8,
          temperature: 20.0,
          light_decay: 0.95,
          nutrient_regeneration: 0.0005  // Menos nutrientes
        }
      };

      const response = await axios.post(`${API_BASE_URL}/simulations/create`, config);
      setSimulationId(response.data.simulation_id);
      setStatus(response.data.status);
      setError(null);
      // Clear previous data
      setOrganisms([]);
      setPredators([]);
      setNutrients([]);
      setStatistics({});
    } catch (err) {
      setError(err.message);
      console.error('Failed to create simulation:', err);
    } finally {
      setIsCreating(false);
    }
  }, [isCreating]);

  const startSimulation = useCallback(async () => {
    if (!simulationId) return;
    
    try {
      await axios.post(`${API_BASE_URL}/simulations/${simulationId}/start`);
      setStatus('running');
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Failed to start simulation:', err);
    }
  }, [simulationId]);

  const pauseSimulation = useCallback(async () => {
    if (!simulationId) return;
    
    try {
      await axios.post(`${API_BASE_URL}/simulations/${simulationId}/pause`);
      setStatus('paused');
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Failed to pause simulation:', err);
    }
  }, [simulationId]);

  const stopSimulation = useCallback(async () => {
    if (!simulationId) return;
    
    try {
      await axios.post(`${API_BASE_URL}/simulations/${simulationId}/stop`);
      setStatus('stopped');
      setSimulationId(null);
      setOrganisms([]);
      setPredators([]);
      setNutrients([]);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Failed to stop simulation:', err);
    }
  }, [simulationId]);

  const spawnOrganism = useCallback(async (x, y, type = 'protozoa') => {
    if (!simulationId) return;
    
    try {
      await axios.post(`${API_BASE_URL}/simulations/${simulationId}/spawn-organism`, {
        x,
        y,
        organism_type: type
      });
    } catch (err) {
      console.error('Failed to spawn organism:', err);
    }
  }, [simulationId]);

  const addPredator = useCallback(async (x, y) => {
    if (!simulationId) return;
    
    try {
      await axios.post(`${API_BASE_URL}/simulations/${simulationId}/add-predator`, {
        x,
        y
      });
    } catch (err) {
      console.error('Failed to add predator:', err);
    }
  }, [simulationId]);

  // Fetch statistics periodically
  useEffect(() => {
    if (!simulationId || status !== 'running') return;

    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/simulations/${simulationId}/statistics`);
        setStatistics(response.data);
      } catch (err) {
        console.error('Failed to fetch statistics:', err);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 2000);

    return () => clearInterval(interval);
  }, [simulationId, status]);

  return {
    simulationId,
    status,
    organisms,
    predators,
    nutrients,
    statistics,
    error,
    createSimulation,
    startSimulation,
    pauseSimulation,
    stopSimulation,
    spawnOrganism,
    addPredator,
    setOrganisms,
    setPredators,
    setNutrients
  };
};