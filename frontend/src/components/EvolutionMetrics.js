import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function EvolutionMetrics({ simulation }) {
  const [metrics, setMetrics] = useState({
    traitHistory: [],
    organDistribution: {},
    lineageTree: [],
    mutationAccumulation: [],
    fitnessScores: []
  });
  
  const [selectedMetric, setSelectedMetric] = useState('traits');
  
  const updateMetrics = () => {
    const state = simulation.getState();
    const organisms = state.organisms;
    
    if (organisms.length === 0) return;
    
    // 1. Track trait evolution over time
    const avgTraits = {
      time: state.time,
      generation: state.statistics.average_generation || 0,
      motility: 0,
      sensitivity: 0,
      resilience: 0,
      organCount: 0
    };
    
    let totalOrgans = {};
    
    organisms.forEach(org => {
      if (org.inheritance?.topologicalCore) {
        avgTraits.motility += org.inheritance.topologicalCore.baseMotility || 0;
        avgTraits.sensitivity += org.inheritance.topologicalCore.baseSensitivity || 0;
        avgTraits.resilience += org.inheritance.topologicalCore.baseResilience || 0;
      }
      
      // Count organs
      if (org.organs) {
        org.organs.forEach(organ => {
          if (organ.functionality > 0.1) {
            totalOrgans[organ.type] = (totalOrgans[organ.type] || 0) + 1;
            avgTraits.organCount++;
          }
        });
      }
    });
    
    // Average traits
    const count = organisms.length;
    avgTraits.motility /= count;
    avgTraits.sensitivity /= count;
    avgTraits.resilience /= count;
    avgTraits.organCount /= count;
    
    // 2. Calculate fitness scores
    const fitnessData = organisms.map(org => {
      // Fitness based on survival indicators
      const age = org.age || 0;
      const energy = org.energy || 0;
      const generation = org.generation || 0;
      const organCount = org.organs?.filter(o => o.functionality > 0.1).length || 0;
      
      // Fitness formula: rewards age, energy, generation, and organ development
      const fitness = (age * 0.3) + (energy * 20) + (generation * 2) + (organCount * 5);
      
      return {
        id: org.id,
        fitness: fitness,
        phenotype: org.phenotype || 'Basic'
      };
    }).sort((a, b) => b.fitness - a.fitness).slice(0, 10); // Top 10
    
    // 3. Track mutation accumulation
    const mutationData = {
      time: state.time,
      totalMutations: organisms.reduce((sum, org) => {
        return sum + (org.inheritance?.topologicalCore?.hasMutation ? 1 : 0);
      }, 0),
      mutationRate: state.statistics.average_mutation_rate || 0
    };
    
    // Update state
    setMetrics(prev => ({
      traitHistory: [...prev.traitHistory.slice(-50), avgTraits], // Keep last 50 points
      organDistribution: totalOrgans,
      lineageTree: prev.lineageTree, // TODO: implement lineage tracking
      mutationAccumulation: [...prev.mutationAccumulation.slice(-50), mutationData],
      fitnessScores: fitnessData
    }));
  };
  
  useEffect(() => {
    if (!simulation) return;
    
    const interval = setInterval(() => {
      updateMetrics();
    }, 2000); // Update every 2 seconds
    
    return () => clearInterval(interval);
  }, [simulation, updateMetrics]);
  
  const renderTraitEvolution = () => (
    <div className="metric-panel">
      <h3>Trait Evolution Over Time</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metrics.traitHistory}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="generation" 
            label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
          />
          <YAxis label={{ value: 'Trait Value', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="motility" stroke="#8884d8" name="Motility" />
          <Line type="monotone" dataKey="sensitivity" stroke="#82ca9d" name="Sensitivity" />
          <Line type="monotone" dataKey="resilience" stroke="#ffc658" name="Resilience" />
          <Line type="monotone" dataKey="organCount" stroke="#ff7c7c" name="Avg Organs" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
  
  const renderOrganDistribution = () => (
    <div className="metric-panel">
      <h3>Organ Distribution in Population</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={Object.entries(metrics.organDistribution).map(([type, count]) => ({
          organ: type,
          count: count
        }))}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="organ" angle={-45} textAnchor="end" height={100} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
  
  const renderFitnessScores = () => (
    <div className="metric-panel">
      <h3>Top 10 Organisms by Fitness</h3>
      <div className="fitness-list">
        {metrics.fitnessScores.map((org, index) => (
          <div key={org.id} className="fitness-item">
            <span className="rank">#{index + 1}</span>
            <span className="phenotype">{org.phenotype}</span>
            <span className="fitness-score">{org.fitness.toFixed(1)}</span>
          </div>
        ))}
      </div>
    </div>
  );
  
  const renderMutationTracking = () => (
    <div className="metric-panel">
      <h3>Mutation Accumulation</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metrics.mutationAccumulation}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="totalMutations" stroke="#8884d8" name="Total Mutations" />
          <Line type="monotone" dataKey="mutationRate" stroke="#82ca9d" name="Mutation Rate" yAxisId="right" />
          <YAxis yAxisId="right" orientation="right" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
  
  return (
    <div className="evolution-metrics">
      <h2>Evolution Metrics</h2>
      
      <div className="metric-tabs">
        <button 
          className={selectedMetric === 'traits' ? 'active' : ''}
          onClick={() => setSelectedMetric('traits')}
        >
          Trait Evolution
        </button>
        <button 
          className={selectedMetric === 'organs' ? 'active' : ''}
          onClick={() => setSelectedMetric('organs')}
        >
          Organ Distribution
        </button>
        <button 
          className={selectedMetric === 'fitness' ? 'active' : ''}
          onClick={() => setSelectedMetric('fitness')}
        >
          Fitness Scores
        </button>
        <button 
          className={selectedMetric === 'mutations' ? 'active' : ''}
          onClick={() => setSelectedMetric('mutations')}
        >
          Mutations
        </button>
      </div>
      
      <div className="metric-content">
        {selectedMetric === 'traits' && renderTraitEvolution()}
        {selectedMetric === 'organs' && renderOrganDistribution()}
        {selectedMetric === 'fitness' && renderFitnessScores()}
        {selectedMetric === 'mutations' && renderMutationTracking()}
      </div>
      
      <style>{`
        .evolution-metrics {
          background: rgba(0, 0, 0, 0.8);
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          margin: 20px;
          color: white;
        }
        
        .metric-tabs {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }
        
        .metric-tabs button {
          padding: 8px 16px;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid #555;
          color: white;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.3s;
        }
        
        .metric-tabs button:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        
        .metric-tabs button.active {
          background: rgba(100, 100, 255, 0.3);
          border-color: #88f;
        }
        
        .metric-panel {
          background: rgba(0, 0, 0, 0.5);
          padding: 15px;
          border-radius: 4px;
        }
        
        .fitness-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-top: 15px;
        }
        
        .fitness-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }
        
        .rank {
          font-weight: bold;
          color: #88f;
          width: 40px;
        }
        
        .phenotype {
          flex: 1;
          text-align: center;
        }
        
        .fitness-score {
          font-weight: bold;
          color: #8f8;
          width: 80px;
          text-align: right;
        }
      `}</style>
    </div>
  );
}