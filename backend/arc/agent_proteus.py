#!/usr/bin/env python3
"""
PROTEUS Agent para ARC Prize 2025
Agente basado en evolución de enjambre con memoria compartida
"""

import os
import sys
import json
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import logging

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.arc_swarm_solver_improved import ARCSwarmSolverImproved
from arc.arc_api_client import ARCApiClient

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProteusAgent:
    """
    Agente PROTEUS para ARC Prize
    Utiliza evolución de enjambre con memoria compartida y agentes especializados
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el agente PROTEUS
        
        Args:
            api_key: API key para ARC Prize (opcional, se toma de env si no se proporciona)
        """
        # Cargar configuración
        load_dotenv('.env')
        load_dotenv('/app/.env')
        
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        self.api_client = ARCApiClient(self.api_key)
        
        # Configuración del solver
        self.solver = ARCSwarmSolverImproved(
            population_size=30,
            generations=5,
            mutation_rate=0.35,
            crossover_rate=0.7,
            elite_size=5
        )
        
        # Estado del juego
        self.current_game = None
        self.current_puzzle = None
        self.submission_history = []
        
        logger.info("🧬 PROTEUS Agent initialized")
        logger.info(f"   Population: {self.solver.population_size}")
        logger.info(f"   Generations: {self.solver.generations}")
        
    def connect(self, game_id: str = "ls20") -> bool:
        """
        Conecta con un juego específico en ARC Prize
        
        Args:
            game_id: ID del juego (por defecto ls20)
            
        Returns:
            True si la conexión fue exitosa
        """
        try:
            logger.info(f"🔌 Connecting to game: {game_id}")
            
            # Resetear el juego
            frame = self.api_client.reset_game(game_id)
            
            if frame and 'guid' in frame:
                self.current_game = game_id
                self.current_puzzle = frame
                logger.info(f"✅ Connected to game {game_id}")
                logger.info(f"   GUID: {frame['guid'][:20]}...")
                
                if 'state' in frame:
                    logger.info(f"   State: {frame['state']}")
                    
                return True
            else:
                logger.error(f"❌ Failed to connect to game {game_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    def solve_current_puzzle(self) -> Optional[np.ndarray]:
        """
        Resuelve el puzzle actual usando el solver de enjambre
        
        Returns:
            Solución como numpy array o None si falla
        """
        if not self.current_puzzle:
            logger.error("No puzzle loaded")
            return None
        
        try:
            # Extraer ejemplos de entrenamiento y test
            train_examples = self._extract_train_examples()
            test_input = self._extract_test_input()
            
            if not train_examples or test_input is None:
                logger.error("Failed to extract puzzle data")
                return None
            
            logger.info(f"🧩 Solving puzzle with {len(train_examples)} training examples")
            logger.info(f"   Test input shape: {test_input.shape}")
            
            # Resolver usando el enjambre
            solution, report = self.solver.solve_with_swarm(
                train_examples, 
                test_input
            )
            
            if solution is not None:
                logger.info(f"✨ Solution found!")
                logger.info(f"   Shape: {solution.shape}")
                logger.info(f"   Fitness: {report.get('fitness', 0):.3f}")
                logger.info(f"   Alive agents: {report.get('alive_agents', 0)}")
                
                # Registrar estadísticas de especialización
                if 'specialization_stats' in report:
                    logger.info("📊 Agent specialization stats:")
                    for spec, stats in report['specialization_stats'].items():
                        logger.info(f"   {spec}: {stats['count']} agents, "
                                  f"avg fitness: {stats['avg_fitness']:.3f}")
                
                return solution
            else:
                logger.warning("⚠️ No solution found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error solving puzzle: {e}")
            return None
    
    def submit_solution(self, solution: np.ndarray) -> Dict[str, Any]:
        """
        Envía una solución al servidor de ARC Prize
        
        Args:
            solution: Solución como numpy array
            
        Returns:
            Respuesta del servidor
        """
        if not self.current_game:
            logger.error("No active game")
            return {"error": "No active game"}
        
        try:
            # Convertir solución a formato ARC (lista de listas)
            solution_list = solution.tolist()
            
            logger.info(f"📤 Submitting solution...")
            
            # Enviar solución
            response = self.api_client.submit_solution(
                self.current_game,
                solution_list
            )
            
            # Registrar en historial
            self.submission_history.append({
                'game': self.current_game,
                'solution': solution_list,
                'response': response
            })
            
            if response.get('correct'):
                logger.info("🎉 Solution correct!")
            else:
                logger.info("❌ Solution incorrect")
                
            return response
            
        except Exception as e:
            logger.error(f"❌ Error submitting solution: {e}")
            return {"error": str(e)}
    
    def run_game(self, game_id: str = "ls20", max_attempts: int = 3) -> bool:
        """
        Ejecuta un juego completo con múltiples intentos
        
        Args:
            game_id: ID del juego
            max_attempts: Número máximo de intentos
            
        Returns:
            True si resuelve el puzzle
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎮 Starting PROTEUS Agent on game: {game_id}")
        logger.info(f"{'='*60}\n")
        
        # Conectar al juego
        if not self.connect(game_id):
            return False
        
        # Intentar resolver
        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n📍 Attempt {attempt}/{max_attempts}")
            
            # Resolver puzzle
            solution = self.solve_current_puzzle()
            
            if solution is not None:
                # Enviar solución
                response = self.submit_solution(solution)
                
                if response.get('correct'):
                    logger.info(f"\n🏆 PUZZLE SOLVED in {attempt} attempts!")
                    return True
                else:
                    logger.info(f"   Incorrect solution, trying again...")
                    
                    # Ajustar parámetros del solver para el siguiente intento
                    self.solver.mutation_rate = min(0.5, self.solver.mutation_rate * 1.1)
                    self.solver.generations = min(10, self.solver.generations + 1)
            else:
                logger.warning(f"   No solution generated")
        
        logger.info(f"\n😔 Failed to solve puzzle after {max_attempts} attempts")
        return False
    
    def _extract_train_examples(self) -> List[Dict]:
        """Extrae ejemplos de entrenamiento del puzzle actual"""
        if not self.current_puzzle:
            return []
        
        # El formato exacto depende de cómo el servidor envía los datos
        # Esto es un ejemplo basado en el formato típico de ARC
        train_examples = []
        
        if 'train' in self.current_puzzle:
            for example in self.current_puzzle['train']:
                train_examples.append({
                    'input': example.get('input', []),
                    'output': example.get('output', [])
                })
        
        return train_examples
    
    def _extract_test_input(self) -> Optional[np.ndarray]:
        """Extrae el input de test del puzzle actual"""
        if not self.current_puzzle:
            return None
        
        if 'test' in self.current_puzzle and len(self.current_puzzle['test']) > 0:
            test_input = self.current_puzzle['test'][0].get('input', [])
            return np.array(test_input)
        
        return None


def main():
    """Función principal para ejecutar el agente"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PROTEUS Agent for ARC Prize')
    parser.add_argument('--game', default='ls20', help='Game ID to play')
    parser.add_argument('--attempts', type=int, default=3, help='Maximum attempts per puzzle')
    parser.add_argument('--api-key', help='ARC API Key (optional, uses env if not provided)')
    
    args = parser.parse_args()
    
    # Crear y ejecutar agente
    agent = ProteusAgent(api_key=args.api_key)
    
    # Ejecutar juego
    success = agent.run_game(
        game_id=args.game,
        max_attempts=args.attempts
    )
    
    # Mostrar resultados finales
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Game: {args.game}")
    print(f"Success: {'✅ YES' if success else '❌ NO'}")
    print(f"Submissions: {len(agent.submission_history)}")
    
    if agent.submission_history:
        print("\n📝 Submission History:")
        for i, submission in enumerate(agent.submission_history, 1):
            print(f"   {i}. Correct: {submission['response'].get('correct', False)}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())