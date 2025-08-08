"""
Selección sin fitness explícito - Emergencia natural
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SelectionEvent:
    """Representa un evento de selección en el mundo"""
    time: float
    type: str  # 'death', 'birth', 'survival'
    organism_id: str
    cause: Optional[str] = None
    metadata: Optional[Dict] = None


class EmergentSelection:
    """
    Sistema de selección emergente
    No hay función de fitness - la supervivencia ES el fitness
    """
    
    def __init__(self):
        self.selection_history = []
        self.survival_patterns = defaultdict(list)
        self.death_causes = defaultdict(int)
        self.generation_survivors = defaultdict(list)
        
    def record_death(self, organism, cause: str, time: float):
        """Registra una muerte y su causa"""
        event = SelectionEvent(
            time=time,
            type='death',
            organism_id=organism.id,
            cause=cause,
            metadata={
                'age': organism.age,
                'generation': organism.generation,
                'distance_traveled': organism.distance_traveled,
                'light_exposures': organism.light_exposures
            }
        )
        
        self.selection_history.append(event)
        self.death_causes[cause] += 1
        
    def record_birth(self, organism, time: float):
        """Registra un nacimiento"""
        event = SelectionEvent(
            time=time,
            type='birth',
            organism_id=organism.id,
            metadata={
                'generation': organism.generation,
                'has_seed': organism.seed is not None
            }
        )
        
        self.selection_history.append(event)
        
    def record_survival(self, organism, time: float):
        """Registra un superviviente al final de una generación"""
        event = SelectionEvent(
            time=time,
            type='survival',
            organism_id=organism.id,
            metadata={
                'age': organism.age,
                'generation': organism.generation,
                'distance_traveled': organism.distance_traveled,
                'energy': organism.energy,
                'topological_features': organism.get_topological_features()
            }
        )
        
        self.selection_history.append(event)
        self.generation_survivors[organism.generation].append(organism.id)
        
        # Analizar patrones de supervivencia
        if organism.seed:
            pattern_key = self._extract_survival_pattern(organism)
            self.survival_patterns[pattern_key].append(organism.age)
            
    def _extract_survival_pattern(self, organism) -> str:
        """
        Extrae un patrón de supervivencia del organismo
        Esto nos ayuda a identificar estrategias emergentes
        """
        features = organism.get_topological_features()
        
        # Categorizar basado en características topológicas
        persistence = features.get('persistence', 0)
        complexity = features.get('complexity', 0)
        
        # Crear categorías simples
        if persistence > 0.7:
            movement_type = "direct"
        elif persistence < 0.3:
            movement_type = "wandering"
        else:
            movement_type = "mixed"
            
        if complexity > 0.5:
            exploration_type = "explorer"
        else:
            exploration_type = "local"
            
        return f"{movement_type}_{exploration_type}"
    
    def analyze_selection_pressure(self, generation: int) -> Dict:
        """
        Analiza las presiones de selección en una generación
        """
        # Filtrar eventos de esta generación
        gen_events = [e for e in self.selection_history 
                     if e.metadata.get('generation') == generation]
        
        deaths = [e for e in gen_events if e.type == 'death']
        births = [e for e in gen_events if e.type == 'birth']
        survivors = [e for e in gen_events if e.type == 'survival']
        
        # Calcular estadísticas
        analysis = {
            'generation': generation,
            'total_deaths': len(deaths),
            'total_births': len(births),
            'total_survivors': len(survivors),
            'survival_rate': len(survivors) / (len(deaths) + len(survivors)) 
                           if (len(deaths) + len(survivors)) > 0 else 0
        }
        
        # Analizar causas de muerte
        gen_death_causes = defaultdict(int)
        for death in deaths:
            gen_death_causes[death.cause] += 1
            
        analysis['death_causes'] = dict(gen_death_causes)
        
        # Analizar edad promedio de muerte vs supervivencia
        if deaths:
            death_ages = [d.metadata['age'] for d in deaths]
            analysis['avg_death_age'] = np.mean(death_ages)
            analysis['std_death_age'] = np.std(death_ages)
        else:
            analysis['avg_death_age'] = 0
            analysis['std_death_age'] = 0
            
        if survivors:
            survival_ages = [s.metadata['age'] for s in survivors]
            analysis['avg_survival_age'] = np.mean(survival_ages)
            analysis['std_survival_age'] = np.std(survival_ages)
        else:
            analysis['avg_survival_age'] = 0
            analysis['std_survival_age'] = 0
            
        return analysis
    
    def identify_emerging_strategies(self) -> Dict[str, Dict]:
        """
        Identifica estrategias de supervivencia emergentes
        """
        strategies = {}
        
        for pattern, ages in self.survival_patterns.items():
            if len(ages) >= 5:  # Necesitamos suficientes datos
                strategies[pattern] = {
                    'count': len(ages),
                    'avg_survival_time': np.mean(ages),
                    'std_survival_time': np.std(ages),
                    'max_survival_time': np.max(ages),
                    'success_score': np.mean(ages) * len(ages)  # Combina longevidad y frecuencia
                }
                
        # Ordenar por éxito
        sorted_strategies = sorted(strategies.items(), 
                                 key=lambda x: x[1]['success_score'], 
                                 reverse=True)
        
        return dict(sorted_strategies)
    
    def compute_selection_gradient(self, feature_extractor: Callable) -> np.ndarray:
        """
        Calcula el gradiente de selección para características específicas
        Esto nos dice qué características están bajo presión selectiva
        """
        # Extraer características de supervivientes y muertos
        survivor_features = []
        death_features = []
        
        for event in self.selection_history:
            if event.type == 'survival':
                features = feature_extractor(event.metadata)
                survivor_features.append(features)
            elif event.type == 'death':
                features = feature_extractor(event.metadata)
                death_features.append(features)
                
        if not survivor_features or not death_features:
            return np.zeros(1)
            
        # Calcular diferencia media
        survivor_mean = np.mean(survivor_features, axis=0)
        death_mean = np.mean(death_features, axis=0)
        
        # El gradiente apunta hacia características favorecidas
        gradient = survivor_mean - death_mean
        
        return gradient
    
    def predict_future_evolution(self, n_generations: int = 10) -> Dict:
        """
        Predice tendencias evolutivas basándose en gradientes actuales
        """
        # Calcular gradientes para diferentes características
        def extract_movement(metadata):
            return np.array([
                metadata.get('topological_features', {}).get('persistence', 0),
                metadata.get('topological_features', {}).get('complexity', 0)
            ])
            
        movement_gradient = self.compute_selection_gradient(extract_movement)
        
        predictions = {
            'movement_trend': {
                'persistence_change': movement_gradient[0] * n_generations * 0.1,
                'complexity_change': movement_gradient[1] * n_generations * 0.1
            },
            'emerging_behaviors': []
        }
        
        # Predecir comportamientos emergentes
        if movement_gradient[0] > 0.1:
            predictions['emerging_behaviors'].append("Increased directional movement")
        elif movement_gradient[0] < -0.1:
            predictions['emerging_behaviors'].append("More exploratory movement")
            
        if movement_gradient[1] > 0.1:
            predictions['emerging_behaviors'].append("Complex path patterns")
        elif movement_gradient[1] < -0.1:
            predictions['emerging_behaviors'].append("Simplified movement strategies")
            
        return predictions
    
    def get_evolutionary_timeline(self) -> List[Dict]:
        """
        Crea una línea temporal de eventos evolutivos importantes
        """
        timeline = []
        
        # Agrupar por generación
        events_by_gen = defaultdict(list)
        for event in self.selection_history:
            gen = event.metadata.get('generation', 0)
            events_by_gen[gen].append(event)
            
        for gen in sorted(events_by_gen.keys()):
            gen_events = events_by_gen[gen]
            
            # Identificar eventos significativos
            births = len([e for e in gen_events if e.type == 'birth'])
            deaths = len([e for e in gen_events if e.type == 'death'])
            survivors = len([e for e in gen_events if e.type == 'survival'])
            
            if survivors > 0:
                avg_age = np.mean([e.metadata['age'] for e in gen_events 
                                 if e.type == 'survival'])
            else:
                avg_age = 0
                
            timeline.append({
                'generation': gen,
                'births': births,
                'deaths': deaths,
                'survivors': survivors,
                'avg_survival_age': avg_age,
                'events': self._identify_key_events(gen_events)
            })
            
        return timeline
    
    def _identify_key_events(self, events: List[SelectionEvent]) -> List[str]:
        """Identifica eventos evolutivos clave en una lista de eventos"""
        key_events = []
        
        # Buscar extinciones masivas
        death_ratio = len([e for e in events if e.type == 'death']) / len(events)
        if death_ratio > 0.8:
            key_events.append("Mass extinction event")
            
        # Buscar boom poblacional
        birth_ratio = len([e for e in events if e.type == 'birth']) / len(events)
        if birth_ratio > 0.5:
            key_events.append("Population boom")
            
        # Buscar longevidad excepcional
        survival_ages = [e.metadata['age'] for e in events if e.type == 'survival']
        if survival_ages and np.max(survival_ages) > np.mean(survival_ages) * 2:
            key_events.append("Exceptional longevity achieved")
            
        return key_events