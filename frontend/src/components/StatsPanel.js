import React from 'react';
import styled from 'styled-components';
import { AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const Panel = styled.div`
  flex: 1;
  padding: 1.5rem;
  overflow-y: auto;
`;

const Title = styled.h2`
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--text-primary);
`;

const StatGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const StatCard = styled.div`
  background-color: var(--bg-primary);
  padding: 1rem;
  border-radius: 4px;
  border: 1px solid var(--bg-tertiary);
`;

const StatLabel = styled.div`
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
`;

const StatValue = styled.div`
  font-size: 1.5rem;
  font-weight: 600;
  color: ${props => props.$color || 'var(--text-primary)'};
`;

const ChartContainer = styled.div`
  margin-bottom: 1.5rem;
`;

const ChartTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--text-secondary);
`;

const StatsPanel = ({ statistics = {} }) => {
  const {
    total_organisms = statistics.totalOrganisms || 0, // Fallback for backwards compatibility
    births = 0,
    deaths = 0,
    mutations = 0,
    average_age,
    average_energy,
    phenotype_distribution = {},
    organ_distribution = {},
    average_generation = 0,
    highest_generation = 0,
    // total_experiences = 0, // Removed - unused
    memory_anchors = 0,
    average_mutation_rate = 0
  } = statistics;
  
  // Debug logging
  if (total_organisms === 0 && statistics.totalOrganisms > 0) {
    console.warn('Using totalOrganisms fallback:', statistics.totalOrganisms);
  }
  
  // Handle NaN or null values
  const avgAge = isNaN(average_age) || average_age === null ? 0 : average_age;
  const avgEnergy = isNaN(average_energy) || average_energy === null ? 0 : average_energy;

  // Prepare data for charts
  const phenotypeData = Object.entries(phenotype_distribution).map(([name, count]) => ({
    name: name || 'Basic',
    value: count
  }));

  const organData = Object.entries(organ_distribution).map(([name, count]) => ({
    name: name.replace('_', ' '),
    value: count
  }));

  const COLORS = ['#00CED1', '#32CD32', '#FF69B4', '#FFD700', '#9370DB', '#FF6347'];

  return (
    <Panel>
      <Title>Statistics</Title>
      
      <StatGrid>
        <StatCard>
          <StatLabel>Population</StatLabel>
          <StatValue $color={total_organisms > 20 ? '#32CD32' : '#FFA500'}>
            {total_organisms}
          </StatValue>
        </StatCard>
        
        <StatCard>
          <StatLabel>Births</StatLabel>
          <StatValue $color="var(--accent-primary)">{births}</StatValue>
        </StatCard>
        
        <StatCard>
          <StatLabel>Deaths</StatLabel>
          <StatValue $color="var(--danger)">{deaths}</StatValue>
        </StatCard>
        
        <StatCard>
          <StatLabel>Mutations</StatLabel>
          <StatValue $color="var(--accent-secondary)">{mutations}</StatValue>
        </StatCard>
        
        <StatCard>
          <StatLabel>Avg Age</StatLabel>
          <StatValue>{avgAge.toFixed(1)}</StatValue>
        </StatCard>
        
        <StatCard>
          <StatLabel>Avg Energy</StatLabel>
          <StatValue $color={avgEnergy > 1 ? '#32CD32' : '#FFA500'}>
            {avgEnergy.toFixed(2)}
          </StatValue>
        </StatCard>
      </StatGrid>
      
      {/* Tri-Layer Inheritance Stats */}
      <ChartContainer>
        <ChartTitle>🧬 Inheritance System</ChartTitle>
        <StatGrid>
          <StatCard>
            <StatLabel>Avg Generation</StatLabel>
            <StatValue>{average_generation.toFixed(1)}</StatValue>
          </StatCard>
          
          <StatCard>
            <StatLabel>Highest Gen</StatLabel>
            <StatValue $color="var(--accent-primary)">{highest_generation}</StatValue>
          </StatCard>
          
          <StatCard>
            <StatLabel>Mutation Rate</StatLabel>
            <StatValue $color="#FF69B4">
              {(average_mutation_rate * 100).toFixed(1)}%
            </StatValue>
          </StatCard>
          
          <StatCard>
            <StatLabel>Memory Anchors</StatLabel>
            <StatValue $color="#9370DB">{memory_anchors}</StatValue>
          </StatCard>
        </StatGrid>
      </ChartContainer>
      
      {phenotypeData.length > 0 && (
        <ChartContainer>
          <ChartTitle>Phenotype Distribution</ChartTitle>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie
                data={phenotypeData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={2}
                dataKey="value"
              >
                {phenotypeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-primary)', 
                  border: '1px solid var(--bg-tertiary)',
                  borderRadius: '4px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>
      )}
      
      {organData.length > 0 && (
        <ChartContainer>
          <ChartTitle>Organ Evolution</ChartTitle>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={organData}>
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="var(--accent-primary)" 
                fillOpacity={1} 
                fill="url(#colorGradient)" 
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-primary)', 
                  border: '1px solid var(--bg-tertiary)',
                  borderRadius: '4px'
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>
      )}
    </Panel>
  );
};

export default StatsPanel;