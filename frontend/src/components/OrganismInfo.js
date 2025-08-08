import React from 'react';
import styled from 'styled-components';

const InfoContainer = styled.div`
  background-color: var(--bg-primary);
  border-top: 1px solid var(--bg-tertiary);
  padding: 1rem;
  flex: 1;
  overflow-y: auto;
  animation: fadeIn 0.3s ease-out;
  
  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
`;

const Title = styled.h3`
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 1.25rem;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  
  &:hover {
    color: var(--text-primary);
  }
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
  margin-bottom: 1rem;
`;

const InfoItem = styled.div`
  display: flex;
  flex-direction: column;
`;

const InfoLabel = styled.span`
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
`;

const InfoValue = styled.span`
  font-size: 0.875rem;
  color: var(--text-primary);
  font-weight: 500;
`;

const OrgansList = styled.div`
  margin-top: 1rem;
`;

const OrganItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background-color: var(--bg-secondary);
  border-radius: 4px;
  margin-bottom: 0.5rem;
`;

const OrganName = styled.span`
  font-size: 0.75rem;
  color: var(--text-primary);
  text-transform: capitalize;
`;

const OrganBar = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ProgressBar = styled.div`
  width: 60px;
  height: 4px;
  background-color: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  background-color: ${props => props.$color || 'var(--accent-primary)'};
  width: ${props => props.$value * 100}%;
  transition: width 0.3s ease;
`;

const OrganismInfo = ({ organism, onClose }) => {
  if (!organism) return null;

  const {
    id,
    generation,
    age,
    energy,
    phenotype,
    organs = [],
    capabilities = {},
    inheritance,
    isPanicked = false
  } = organism;

  const getOrganColor = (organType) => {
    const colors = {
      photosensor: '#00CED1',
      chemoreceptor: '#32CD32',
      flagellum: '#FF69B4',
      cilia: '#FF1493',
      membrane: '#8B4513',
      vacuole: '#4169E1',
      pseudopod: '#9370DB',
      crystallin: '#00FFFF',
      pigment_spot: '#FFD700',
      nerve_net: '#FF00FF'
    };
    return colors[organType] || '#FFFFFF';
  };

  return (
    <InfoContainer>
      <Header>
        <Title>Organism Details</Title>
        <CloseButton onClick={onClose}>×</CloseButton>
      </Header>
      
      <InfoGrid>
        <InfoItem>
          <InfoLabel>ID</InfoLabel>
          <InfoValue>{id.substring(0, 8)}...</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Generation</InfoLabel>
          <InfoValue>{generation}</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Age</InfoLabel>
          <InfoValue>{age.toFixed(1)}</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Energy</InfoLabel>
          <InfoValue style={{ color: energy > 1 ? '#32CD32' : '#FFA500' }}>
            {energy.toFixed(2)}
          </InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Phenotype</InfoLabel>
          <InfoValue>{phenotype || 'Basic'}</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Vision</InfoLabel>
          <InfoValue>{(capabilities.vision || 0).toFixed(2)}</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Motility</InfoLabel>
          <InfoValue>{(capabilities.motility || 0.1).toFixed(2)}</InfoValue>
        </InfoItem>
        
        <InfoItem>
          <InfoLabel>Efficiency</InfoLabel>
          <InfoValue>{(capabilities.efficiency || 1).toFixed(2)}</InfoValue>
        </InfoItem>
      </InfoGrid>
      
      {/* Tri-Layer Inheritance Info */}
      {inheritance && (
        <>
          <InfoLabel style={{ marginTop: '1rem', marginBottom: '0.5rem', display: 'block', fontSize: '0.875rem', fontWeight: 600 }}>
            🧬 Tri-Layer Inheritance System
          </InfoLabel>
          
          <InfoGrid>
            <InfoItem>
              <InfoLabel>Topological Core</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem' }}>
                Dimension: {inheritance.topologicalCore?.manifoldDimension?.toFixed(2) || 'N/A'}
              </InfoValue>
            </InfoItem>
            
            <InfoItem>
              <InfoLabel>Body Symmetry</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem' }}>
                {inheritance.topologicalCore?.bodySymmetry || 'N/A'}-fold
              </InfoValue>
            </InfoItem>
            
            <InfoItem>
              <InfoLabel>Mutation Rate</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem', color: inheritance.topologicalCore?.hasMutation ? '#FF69B4' : 'inherit' }}>
                {((inheritance.topologicalCore?.mutability || 0) * 100).toFixed(1)}%
                {inheritance.topologicalCore?.hasMutation && ' 🧬'}
              </InfoValue>
            </InfoItem>
            
            <InfoItem>
              <InfoLabel>Memory Strength</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem' }}>
                {inheritance.holographicMemory ? 
                  `${inheritance.holographicMemory.criticalMoments?.length || 0} moments` : 
                  'No memory'}
              </InfoValue>
            </InfoItem>
          </InfoGrid>
          
          {/* Environmental Influences */}
          <InfoLabel style={{ marginTop: '0.75rem', marginBottom: '0.5rem', display: 'block' }}>
            Environmental Memory
          </InfoLabel>
          <InfoGrid>
            <InfoItem>
              <InfoLabel>Danger Response</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem', color: isPanicked ? '#FF4444' : 'inherit' }}>
                {isPanicked ? '⚠️ Panicked!' : 'Calm'}
              </InfoValue>
            </InfoItem>
            
            <InfoItem>
              <InfoLabel>Experiences</InfoLabel>
              <InfoValue style={{ fontSize: '0.75rem' }}>
                {inheritance.environmentalTraces?.experienceCount || 0} total
              </InfoValue>
            </InfoItem>
          </InfoGrid>
        </>
      )}
      
      {organs.length > 0 && (
        <OrgansList>
          <InfoLabel style={{ marginBottom: '0.5rem', display: 'block' }}>
            Developed Organs
          </InfoLabel>
          {organs.map((organ, index) => (
            <OrganItem key={index}>
              <OrganName>{organ.type.replace('_', ' ')}</OrganName>
              <OrganBar>
                <ProgressBar>
                  <ProgressFill 
                    $value={organ.functionality} 
                    $color={getOrganColor(organ.type)}
                  />
                </ProgressBar>
                <InfoValue style={{ fontSize: '0.75rem', minWidth: '35px' }}>
                  {(organ.functionality * 100).toFixed(0)}%
                </InfoValue>
              </OrganBar>
            </OrganItem>
          ))}
        </OrgansList>
      )}
    </InfoContainer>
  );
};

export default OrganismInfo;