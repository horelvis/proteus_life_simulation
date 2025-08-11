#!/usr/bin/env python3
"""
Script para ejecutar agentes ARC Prize
Soporta agentes random, proteus y custom
"""

import os
import sys
import argparse
import random
import numpy as np
from typing import Dict, List, Any
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RandomAgent:
    """Agente aleatorio simple para testing"""
    
    def __init__(self):
        logger.info("🎲 Random Agent initialized")
        
    def solve(self, train_examples: List[Dict], test_input: np.ndarray) -> np.ndarray:
        """Genera una solución aleatoria"""
        # Analizar tamaños de output en ejemplos
        output_shapes = []
        for example in train_examples:
            output = np.array(example['output'])
            output_shapes.append(output.shape)
        
        # Usar el tamaño más común o el del input
        if output_shapes:
            # Usar el primer tamaño de output como referencia
            target_shape = output_shapes[0]
        else:
            target_shape = test_input.shape
        
        # Generar solución aleatoria con colores del 0 al 9
        solution = np.random.randint(0, 10, size=target_shape)
        
        logger.info(f"Generated random solution with shape: {target_shape}")
        return solution


def run_random_agent(game_id: str = "ls20"):
    """Ejecuta el agente aleatorio"""
    logger.info(f"Running Random Agent on game: {game_id}")
    
    # Crear agente
    agent = RandomAgent()
    
    # Datos de prueba (simulados)
    train_examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[2, 3], [4, 5]]
        }
    ]
    test_input = np.array([[5, 6], [7, 8]])
    
    # Resolver
    solution = agent.solve(train_examples, test_input)
    
    logger.info(f"Solution:\n{solution}")
    return solution


def run_proteus_agent(game_id: str = "ls20", attempts: int = 3):
    """Ejecuta el agente PROTEUS"""
    try:
        # Importar el agente PROTEUS
        from arc.agent_proteus import ProteusAgent
        
        logger.info(f"Running PROTEUS Agent on game: {game_id}")
        
        # Crear y ejecutar agente
        agent = ProteusAgent()
        success = agent.run_game(game_id=game_id, max_attempts=attempts)
        
        return success
        
    except ImportError as e:
        logger.error(f"Failed to import PROTEUS agent: {e}")
        logger.info("Make sure agent_proteus.py is in the arc/ directory")
        return False
    except Exception as e:
        logger.error(f"Error running PROTEUS agent: {e}")
        return False


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Run ARC Prize agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_agent.py --agent=random --game=ls20
  python3 run_agent.py --agent=proteus --game=ls20 --attempts=5
  
Available agents:
  random   - Random solution generator (for testing)
  proteus  - PROTEUS swarm-based solver with shared memory
        """
    )
    
    parser.add_argument(
        '--agent',
        choices=['random', 'proteus'],
        default='proteus',
        help='Agent type to run (default: proteus)'
    )
    
    parser.add_argument(
        '--game',
        default='ls20',
        help='Game ID to play (default: ls20)'
    )
    
    parser.add_argument(
        '--attempts',
        type=int,
        default=3,
        help='Maximum attempts for PROTEUS agent (default: 3)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Header
    print("\n" + "="*60)
    print("🤖 ARC PRIZE AGENT RUNNER")
    print("="*60)
    print(f"Agent: {args.agent.upper()}")
    print(f"Game: {args.game}")
    if args.agent == 'proteus':
        print(f"Max Attempts: {args.attempts}")
    print("="*60 + "\n")
    
    # Ejecutar agente seleccionado
    if args.agent == 'random':
        result = run_random_agent(args.game)
        success = result is not None
    elif args.agent == 'proteus':
        success = run_proteus_agent(args.game, args.attempts)
    else:
        logger.error(f"Unknown agent: {args.agent}")
        success = False
    
    # Resultado final
    print("\n" + "="*60)
    if success:
        print("✅ EXECUTION COMPLETED SUCCESSFULLY")
    else:
        print("❌ EXECUTION FAILED")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())