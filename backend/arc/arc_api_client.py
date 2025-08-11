"""
Cliente para la API oficial de ARC Prize
Basado en: https://github.com/arcprize/ARC-AGI-3-Agents
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ARCApiClient:
    """Cliente para interactuar con la API oficial de ARC Prize"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el cliente de ARC API
        
        Args:
            api_key: Clave de API de ARC Prize. Si no se proporciona, 
                    se busca en la variable de entorno ARC_API_KEY
        """
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        # Nota: La API real de ARC Prize puede tener una URL diferente
        # Esta es una URL de ejemplo - verificar en la documentación oficial
        self.base_url = os.getenv('ARC_API_BASE_URL', 'https://arcprize.org/api/v1')
        self.session = None
        
        # Juegos disponibles según la documentación
        self.available_games = {
            'ls20': 'Learn & Solve 20 - Juego de práctica con 20 puzzles',
            'arc2024': 'ARC Prize 2024 - Competición oficial',
            'training': 'Training set - 400 puzzles de entrenamiento',
            'evaluation': 'Evaluation set - 400 puzzles de evaluación'
        }
        
    async def __aenter__(self):
        """Contexto manager para manejo de sesión"""
        self.session = aiohttp.ClientSession(headers={
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cierra la sesión al salir del contexto"""
        if self.session:
            await self.session.close()
            
    async def get_game_info(self, game_id: str = 'ls20') -> Dict[str, Any]:
        """
        Obtiene información sobre un juego específico
        
        Args:
            game_id: ID del juego (ls20, arc2024, training, evaluation)
            
        Returns:
            Información del juego
        """
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/games/{game_id}",
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Error obteniendo juego: {response.status}")
                        return {}
        
        async with self.session.get(f"{self.base_url}/games/{game_id}") as response:
            if response.status == 200:
                return await response.json()
            return {}
            
    async def get_puzzle(self, game_id: str, puzzle_index: int) -> Dict[str, Any]:
        """
        Obtiene un puzzle específico de un juego
        
        Args:
            game_id: ID del juego
            puzzle_index: Índice del puzzle
            
        Returns:
            Datos del puzzle
        """
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/games/{game_id}/puzzles/{puzzle_index}",
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Error obteniendo puzzle: {response.status}")
                        return {}
                        
        async with self.session.get(
            f"{self.base_url}/games/{game_id}/puzzles/{puzzle_index}"
        ) as response:
            if response.status == 200:
                return await response.json()
            return {}
            
    async def submit_solution(self, game_id: str, puzzle_index: int, 
                            solution: List[List[int]]) -> Dict[str, Any]:
        """
        Envía una solución para un puzzle
        
        Args:
            game_id: ID del juego
            puzzle_index: Índice del puzzle
            solution: Grid de solución
            
        Returns:
            Resultado de la validación
        """
        data = {
            'solution': solution,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/games/{game_id}/puzzles/{puzzle_index}/submit",
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    json=data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Error enviando solución: {response.status}")
                        return {'error': 'Failed to submit'}
                        
        async with self.session.post(
            f"{self.base_url}/games/{game_id}/puzzles/{puzzle_index}/submit",
            json=data
        ) as response:
            if response.status == 200:
                return await response.json()
            return {'error': 'Failed to submit'}
            
    async def get_leaderboard(self, game_id: str = 'arc2024') -> List[Dict[str, Any]]:
        """
        Obtiene el leaderboard de un juego
        
        Args:
            game_id: ID del juego
            
        Returns:
            Lista de posiciones en el leaderboard
        """
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/games/{game_id}/leaderboard",
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
                    
        async with self.session.get(
            f"{self.base_url}/games/{game_id}/leaderboard"
        ) as response:
            if response.status == 200:
                return await response.json()
            return []
            
    def parse_arc_puzzle(self, puzzle_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parsea un puzzle de la API al formato interno
        
        Args:
            puzzle_data: Datos del puzzle de la API
            
        Returns:
            Puzzle en formato interno
        """
        return {
            'id': puzzle_data.get('id', 'unknown'),
            'train': [
                {
                    'input': example.get('input', []),
                    'output': example.get('output', [])
                }
                for example in puzzle_data.get('train', [])
            ],
            'test': [
                {
                    'input': test.get('input', []),
                    'output': test.get('output', []) if 'output' in test else None
                }
                for test in puzzle_data.get('test', [])
            ],
            'metadata': {
                'difficulty': puzzle_data.get('difficulty', 'unknown'),
                'category': puzzle_data.get('category', 'unknown'),
                'source': 'arc_api'
            }
        }
        
    async def load_game_puzzles(self, game_id: str = 'ls20', 
                               count: int = 10) -> List[Dict[str, Any]]:
        """
        Carga múltiples puzzles de un juego
        
        Args:
            game_id: ID del juego
            count: Número de puzzles a cargar
            
        Returns:
            Lista de puzzles
        """
        puzzles = []
        
        # Obtener información del juego
        game_info = await self.get_game_info(game_id)
        total_puzzles = game_info.get('puzzle_count', 20)
        
        # Cargar puzzles
        for i in range(min(count, total_puzzles)):
            puzzle = await self.get_puzzle(game_id, i)
            if puzzle:
                puzzles.append(self.parse_arc_puzzle(puzzle))
                
        logger.info(f"Cargados {len(puzzles)} puzzles del juego {game_id}")
        return puzzles


class ARCGameAgent:
    """Agente base para jugar ARC"""
    
    def __init__(self, api_client: ARCApiClient):
        self.api_client = api_client
        self.current_puzzle = None
        self.attempts = 0
        
    async def solve_puzzle(self, puzzle: Dict[str, Any]) -> List[List[int]]:
        """
        Método a implementar por cada agente
        
        Args:
            puzzle: Puzzle a resolver
            
        Returns:
            Grid de solución
        """
        raise NotImplementedError("Cada agente debe implementar solve_puzzle")
        
    async def play_game(self, game_id: str = 'ls20'):
        """
        Juega un juego completo
        
        Args:
            game_id: ID del juego a jugar
        """
        async with self.api_client:
            # Obtener información del juego
            game_info = await self.api_client.get_game_info(game_id)
            logger.info(f"Jugando: {game_info.get('name', game_id)}")
            
            # Cargar puzzles
            puzzles = await self.api_client.load_game_puzzles(game_id)
            
            # Resolver cada puzzle
            for i, puzzle in enumerate(puzzles):
                logger.info(f"Resolviendo puzzle {i+1}/{len(puzzles)}")
                
                # Generar solución
                solution = await self.solve_puzzle(puzzle)
                
                # Enviar solución
                result = await self.api_client.submit_solution(
                    game_id, i, solution
                )
                
                if result.get('correct'):
                    logger.info(f"✅ Puzzle {i+1} correcto!")
                else:
                    logger.info(f"❌ Puzzle {i+1} incorrecto")
                    

class RandomAgent(ARCGameAgent):
    """Agente que genera soluciones aleatorias"""
    
    async def solve_puzzle(self, puzzle: Dict[str, Any]) -> List[List[int]]:
        """Genera una solución aleatoria"""
        import random
        
        # Usar el tamaño del test input
        test_input = puzzle['test'][0]['input']
        height = len(test_input)
        width = len(test_input[0]) if height > 0 else 0
        
        # Generar grid aleatorio
        solution = []
        for _ in range(height):
            row = [random.randint(0, 9) for _ in range(width)]
            solution.append(row)
            
        return solution


# Ejemplo de uso
async def main():
    """Función principal de ejemplo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ARC API Client')
    parser.add_argument('--agent', default='random', 
                       choices=['random', 'proteus'],
                       help='Tipo de agente')
    parser.add_argument('--game', default='ls20',
                       help='ID del juego')
    parser.add_argument('--api-key', help='ARC API Key')
    
    args = parser.parse_args()
    
    # Crear cliente
    client = ARCApiClient(api_key=args.api_key)
    
    # Crear agente
    if args.agent == 'random':
        agent = RandomAgent(client)
    else:
        agent = RandomAgent(client)  # Por defecto
        
    # Jugar
    await agent.play_game(args.game)


if __name__ == "__main__":
    asyncio.run(main())