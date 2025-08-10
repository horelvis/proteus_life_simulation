#!/usr/bin/env python3
"""
ARC WebSocket Server
Servidor principal para resolver puzzles ARC con transparencia total
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from arc_solver_python import ARCSolverPython
from arc_visualizer import ARCVisualizer
from arc_dataset_loader import ARCDatasetLoader
from arc_swarm_solver import ARCSwarmSolver

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARCWebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.solver = ARCSolverPython()
        self.visualizer = ARCVisualizer()
        self.dataset_loader = ARCDatasetLoader()
        self.active_sessions = {}
        
    async def register(self, websocket):
        """Registra un nuevo cliente"""
        self.clients.add(websocket)
        session_id = f"session_{datetime.now().timestamp()}"
        self.active_sessions[websocket] = {
            'id': session_id,
            'start_time': datetime.now(),
            'puzzles_solved': 0
        }
        logger.info(f"Cliente conectado: {session_id}")
        
        # Enviar mensaje de bienvenida
        await self.send_message(websocket, {
            'type': 'connection',
            'status': 'connected',
            'session_id': session_id,
            'message': 'Conectado al servidor ARC'
        })
        
    async def unregister(self, websocket):
        """Desregistra un cliente"""
        if websocket in self.clients:
            self.clients.remove(websocket)
            session = self.active_sessions.pop(websocket, {})
            logger.info(f"Cliente desconectado: {session.get('id', 'unknown')}")
            
    async def send_message(self, websocket, data: Dict[str, Any]):
        """Envía un mensaje al cliente"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error enviando mensaje: {e}")
            
    async def broadcast(self, data: Dict[str, Any], exclude=None):
        """Envía un mensaje a todos los clientes"""
        if self.clients:
            tasks = []
            for client in self.clients:
                if client != exclude:
                    tasks.append(self.send_message(client, data))
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def handle_message(self, websocket, message: str):
        """Procesa mensajes del cliente"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'load_puzzles':
                await self.handle_load_puzzles(websocket, data)
            elif msg_type == 'solve_puzzle':
                await self.handle_solve_puzzle(websocket, data)
            elif msg_type == 'solve_with_swarm':
                await self.handle_solve_with_swarm(websocket, data)
            elif msg_type == 'get_reasoning_steps':
                await self.handle_get_reasoning_steps(websocket, data)
            elif msg_type == 'verify_integrity':
                await self.handle_verify_integrity(websocket, data)
            elif msg_type == 'export_visualization':
                await self.handle_export_visualization(websocket, data)
            else:
                await self.send_message(websocket, {
                    'type': 'error',
                    'message': f'Tipo de mensaje desconocido: {msg_type}'
                })
                
        except json.JSONDecodeError:
            await self.send_message(websocket, {
                'type': 'error',
                'message': 'Mensaje JSON inválido'
            })
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            await self.send_message(websocket, {
                'type': 'error',
                'message': str(e)
            })
            
    async def handle_load_puzzles(self, websocket, data: Dict[str, Any]):
        """Carga puzzles ARC"""
        puzzle_set = data.get('puzzle_set', 'training')
        count = data.get('count', 10)
        
        await self.send_message(websocket, {
            'type': 'loading',
            'message': 'Cargando puzzles ARC...'
        })
        
        puzzles = self.dataset_loader.load_puzzles(puzzle_set, count)
        
        await self.send_message(websocket, {
            'type': 'puzzles_loaded',
            'puzzles': [self._serialize_puzzle(p) for p in puzzles],
            'count': len(puzzles)
        })
        
    async def handle_solve_puzzle(self, websocket, data: Dict[str, Any]):
        """Resuelve un puzzle específico"""
        puzzle_id = data.get('puzzle_id')
        puzzle_data = data.get('puzzle')
        
        if not puzzle_data:
            await self.send_message(websocket, {
                'type': 'error',
                'message': 'No se proporcionó puzzle'
            })
            return
            
        # Notificar inicio
        await self.send_message(websocket, {
            'type': 'solving_start',
            'puzzle_id': puzzle_id,
            'message': f'Iniciando resolución de {puzzle_id}'
        })
        
        # Resolver puzzle con pasos detallados
        steps = []
        
        # Paso 1: Analizar ejemplos de entrenamiento
        for idx, example in enumerate(puzzle_data.get('train', [])):
            await self.send_message(websocket, {
                'type': 'analyzing_example',
                'puzzle_id': puzzle_id,
                'example_index': idx,
                'input': example['input'],
                'output': example['output']
            })
            
            # Detectar regla
            rule = self.solver.detect_rule(
                np.array(example['input']), 
                np.array(example['output'])
            )
            
            if rule:
                steps.append({
                    'type': 'rule_detected',
                    'rule_type': rule['type'],
                    'confidence': rule['confidence'],
                    'parameters': rule.get('parameters', {})
                })
                
                await self.send_message(websocket, {
                    'type': 'rule_detected',
                    'puzzle_id': puzzle_id,
                    'rule': rule
                })
            
            # Pequeña pausa para que el cliente pueda visualizar
            await asyncio.sleep(0.5)
        
        # Paso 2: Aplicar regla al test
        test_input = np.array(puzzle_data['test'][0]['input'])
        
        await self.send_message(websocket, {
            'type': 'applying_rule',
            'puzzle_id': puzzle_id,
            'message': 'Aplicando regla detectada...'
        })
        
        solution, reasoning_steps = self.solver.solve_with_steps(
            puzzle_data['train'],
            test_input
        )
        
        # Enviar cada paso de razonamiento
        for step in reasoning_steps:
            await self.send_message(websocket, {
                'type': 'reasoning_step',
                'puzzle_id': puzzle_id,
                'step': step
            })
            await asyncio.sleep(0.3)
        
        # Verificar si es correcto
        expected = np.array(puzzle_data['test'][0]['output'])
        is_correct = np.array_equal(solution, expected)
        
        # Actualizar estadísticas de sesión
        if is_correct:
            self.active_sessions[websocket]['puzzles_solved'] += 1
        
        # Enviar resultado final
        await self.send_message(websocket, {
            'type': 'solving_complete',
            'puzzle_id': puzzle_id,
            'solution': solution.tolist(),
            'expected': expected.tolist(),
            'is_correct': is_correct,
            'reasoning_steps': reasoning_steps,
            'confidence': self.solver.get_confidence()
        })
        
    async def handle_get_reasoning_steps(self, websocket, data: Dict[str, Any]):
        """Obtiene pasos detallados de razonamiento"""
        puzzle_id = data.get('puzzle_id')
        
        # Obtener pasos del solver
        steps = self.solver.get_detailed_reasoning()
        
        # Generar visualizaciones para cada paso
        visualizations = []
        for step in steps:
            viz = self.visualizer.create_step_visualization(step)
            visualizations.append(viz)
        
        await self.send_message(websocket, {
            'type': 'reasoning_steps',
            'puzzle_id': puzzle_id,
            'steps': steps,
            'visualizations': visualizations
        })
        
    async def handle_verify_integrity(self, websocket, data: Dict[str, Any]):
        """Verifica la integridad del sistema"""
        await self.send_message(websocket, {
            'type': 'integrity_check_start',
            'message': 'Iniciando verificación de integridad...'
        })
        
        # Ejecutar tests de integridad
        test_results = []
        
        # Test 1: Color mapping con colores nuevos
        test1 = await self._run_integrity_test(
            'color_mapping',
            {'input': [[3, 0, 3], [0, 3, 0]], 'output': [[8, 0, 8], [0, 8, 0]]},
            {'input': [[0, 3, 3], [3, 0, 0]], 'output': [[0, 8, 8], [8, 0, 0]]}
        )
        test_results.append(test1)
        
        # Test 2: Conteo diferente
        test2 = await self._run_integrity_test(
            'counting',
            {'input': [[7, 0, 7], [0, 7, 0], [7, 7, 0]], 'output': [[5]]},
            {'input': [[0, 7, 0], [7, 7, 7]], 'output': [[4]]}
        )
        test_results.append(test2)
        
        # Más tests...
        
        await self.send_message(websocket, {
            'type': 'integrity_check_complete',
            'results': test_results,
            'passed': all(t['passed'] for t in test_results)
        })
        
    async def handle_export_visualization(self, websocket, data: Dict[str, Any]):
        """Exporta visualización como imagen o GIF"""
        puzzle_id = data.get('puzzle_id')
        export_type = data.get('export_type', 'png')
        
        if export_type == 'gif':
            # Generar GIF animado
            gif_data = self.visualizer.create_animated_gif(
                self.solver.get_detailed_reasoning()
            )
            
            await self.send_message(websocket, {
                'type': 'export_ready',
                'puzzle_id': puzzle_id,
                'format': 'gif',
                'data': gif_data
            })
        else:
            # Generar imagen estática
            png_data = self.visualizer.create_reasoning_diagram(
                self.solver.get_detailed_reasoning()
            )
            
            await self.send_message(websocket, {
                'type': 'export_ready',
                'puzzle_id': puzzle_id,
                'format': 'png',
                'data': png_data
            })
    
    def _serialize_puzzle(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """Serializa un puzzle para enviar por WebSocket"""
        return {
            'id': puzzle.get('id', 'unknown'),
            'category': puzzle.get('category', 'unknown'),
            'difficulty': puzzle.get('difficulty', 'unknown'),
            'train': puzzle.get('train', []),
            'test': puzzle.get('test', [])
        }
        
    async def _run_integrity_test(self, test_name: str, train_example: Dict, test_example: Dict):
        """Ejecuta un test de integridad individual"""
        try:
            # Entrenar con ejemplo
            rule = self.solver.detect_rule(
                np.array(train_example['input']),
                np.array(train_example['output'])
            )
            
            # Aplicar a test
            solution = self.solver.apply_rule(
                rule,
                np.array(test_example['input'])
            )
            
            # Verificar
            expected = np.array(test_example['output'])
            passed = np.array_equal(solution, expected)
            
            return {
                'name': test_name,
                'passed': passed,
                'solution': solution.tolist(),
                'expected': expected.tolist()
            }
        except Exception as e:
            return {
                'name': test_name,
                'passed': False,
                'error': str(e)
            }
    
    async def handle_solve_with_swarm(self, websocket, data: Dict[str, Any]):
        """Resuelve un puzzle usando el enjambre con votación"""
        puzzle_id = data.get('puzzle_id')
        puzzle_data = data.get('puzzle')
        swarm_config = data.get('swarm_config', {})
        
        if not puzzle_data:
            await self.send_message(websocket, {
                'type': 'error',
                'message': 'No se proporcionó puzzle'
            })
            return
            
        # Notificar inicio
        await self.send_message(websocket, {
            'type': 'swarm_start',
            'puzzle_id': puzzle_id,
            'message': f'🐝 Iniciando enjambre para {puzzle_id}'
        })
        
        # Crear enjambre con configuración personalizada
        swarm = ARCSwarmSolver(
            population_size=swarm_config.get('population_size', 20),
            generations=swarm_config.get('generations', 5),
            mutation_rate=swarm_config.get('mutation_rate', 0.2)
        )
        
        # Resolver con enjambre
        test_input = np.array(puzzle_data['test'][0]['input'])
        solution, report = swarm.solve_with_swarm(
            puzzle_data['train'],
            test_input
        )
        
        # Enviar actualizaciones de progreso
        for vote in report['voting_history']:
            await self.send_message(websocket, {
                'type': 'swarm_generation',
                'puzzle_id': puzzle_id,
                'generation': vote['generation'] + 1,
                'votes': vote['votes'],
                'fitness': vote['fitness'],
                'agents': vote['agents']
            })
            await asyncio.sleep(0.2)  # Pequeña pausa para visualización
        
        # Verificar si es correcto (si tenemos la respuesta)
        is_correct = False
        expected = None
        if 'output' in puzzle_data['test'][0]:
            expected = np.array(puzzle_data['test'][0]['output'])
            is_correct = np.array_equal(solution, expected)
        
        # Enviar resultado final
        await self.send_message(websocket, {
            'type': 'swarm_complete',
            'puzzle_id': puzzle_id,
            'solution': solution.tolist() if solution is not None else None,
            'expected': expected.tolist() if expected is not None else None,
            'is_correct': is_correct,
            'fitness': report['fitness'],
            'alive_agents': report['alive_agents'],
            'dead_agents': report['dead_agents'],
            'best_agents': report['best_agents']
        })
            
    async def client_handler(self, websocket, path):
        """Maneja la conexión de un cliente"""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def start(self):
        """Inicia el servidor WebSocket"""
        logger.info(f"Iniciando servidor ARC en ws://{self.host}:{self.port}")
        async with websockets.serve(self.client_handler, self.host, self.port):
            await asyncio.Future()  # Ejecutar para siempre

def main():
    """Punto de entrada principal"""
    server = ARCWebSocketServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")

if __name__ == "__main__":
    main()