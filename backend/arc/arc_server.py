#!/usr/bin/env python3
"""
ARC WebSocket Server
Servidor principal para resolver puzzles ARC con transparencia total
"""

import asyncio
import websockets
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos numpy"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

try:
    # Imports relativos cuando se ejecuta como módulo
    from .arc_solver_python import ARCSolverPython
    from .arc_visualizer import ARCVisualizer
    from .arc_dataset_loader import ARCDatasetLoader
    from .arc_official_loader import ARCOfficialLoader
    from .arc_image_processor import ARCImageProcessor
    from .arc_api_client import ARCApiClient
    from .real_processor import RealARCProcessor
except ImportError:
    # Imports absolutos cuando se ejecuta directamente
    from arc_solver_python import ARCSolverPython
    from arc_visualizer import ARCVisualizer
    from arc_dataset_loader import ARCDatasetLoader
    from arc_official_loader import ARCOfficialLoader
    from arc_image_processor import ARCImageProcessor
    from arc_api_client import ARCApiClient
    try:
        from real_processor import RealARCProcessor
    except ImportError:
        RealARCProcessor = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARCWebSocketServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.solver = ARCSolverPython()
        self.visualizer = ARCVisualizer()
        self.dataset_loader = ARCDatasetLoader()
        self.official_loader = ARCOfficialLoader()
        self.image_processor = ARCImageProcessor()
        self.arc_api_client = ARCApiClient()  # Cliente para API oficial
        self.active_sessions = {}
        self.puzzles = []  # Almacenar puzzles cargados
        self.api_puzzles_cache = {}  # Cache de puzzles de la API
        self.current_puzzles = {}  # Puzzles cargados por ID
        self.real_processor = None  # Procesador real transparente
        
        # Benchmarks de LLMs en ARC (según docs.arcprize.org)
        self.llm_benchmarks = {
            'gpt-4o': {'accuracy': 0.21, 'model': 'GPT-4 Optimized', 'date': '2024'},
            'claude-3-opus': {'accuracy': 0.18, 'model': 'Claude 3 Opus', 'date': '2024'},
            'gemini-1.5-pro': {'accuracy': 0.16, 'model': 'Gemini 1.5 Pro', 'date': '2024'},
            'gpt-4': {'accuracy': 0.13, 'model': 'GPT-4', 'date': '2023'},
            'claude-2': {'accuracy': 0.10, 'model': 'Claude 2', 'date': '2023'},
            'human_baseline': {'accuracy': 0.85, 'model': 'Human Average', 'date': '2019'},
            'proteus': {'accuracy': 0.0, 'model': 'PROTEUS (This)', 'date': '2025'}
        }
        
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
            await websocket.send(json.dumps(data, cls=NumpyEncoder))
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
            elif msg_type == 'process_image':
                await self.handle_process_image(websocket, data)
            elif msg_type == 'load_image_puzzle':
                await self.handle_load_image_puzzle(websocket, data)
            elif msg_type == 'validate_solution':
                await self.handle_validate_solution(websocket, data)
            elif msg_type == 'load_api_puzzles':
                await self.handle_load_api_puzzles(websocket, data)
            elif msg_type == 'process_puzzle_real':
                await self.handle_process_puzzle_real(websocket, data)
            elif msg_type == 'load_puzzle':
                await self.handle_load_puzzle(websocket, data)
            elif msg_type == 'list_puzzles':
                await self.handle_list_puzzles(websocket, data)
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
        use_official = data.get('use_official', True)  # Por defecto usar oficiales
        
        await self.send_message(websocket, {
            'type': 'loading',
            'message': f'Cargando {count} puzzles {"oficiales de ARC" if use_official else "de ejemplo"}...'
        })
        
        if use_official:
            # Cargar puzzles oficiales desde GitHub
            try:
                puzzles = self.official_loader.load_from_github(
                    dataset=puzzle_set,
                    version='arc_agi_1',
                    limit=count
                )
                # Convertir al formato esperado
                formatted_puzzles = []
                for p in puzzles:
                    formatted = {
                        'id': p['id'],
                        'train': p['trainExamples'],
                        'test': [p['testExample']] if p['testExample'] else [],
                        'category': 'official',
                        'difficulty': 'unknown',
                        'source': 'arc_official'
                    }
                    formatted_puzzles.append(formatted)
                puzzles = formatted_puzzles
            except Exception as e:
                logger.error(f"Error cargando puzzles oficiales: {e}")
                # Fallback a puzzles de ejemplo
                puzzles = self.dataset_loader.load_puzzles(puzzle_set, count)
        else:
            puzzles = self.dataset_loader.load_puzzles(puzzle_set, count)
        
        # Guardar puzzles en memoria
        self.puzzles = puzzles
        
        # Agregar info de benchmarks
        await self.send_message(websocket, {
            'type': 'puzzles_loaded',
            'puzzles': [self._serialize_puzzle(p) for p in puzzles],
            'count': len(puzzles),
            'benchmarks': self.llm_benchmarks,
            'official': use_official
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
        
    async def handle_validate_solution(self, websocket, data: Dict[str, Any]):
        """Valida la solución del usuario en el modo juego"""
        puzzle_id = data.get('puzzleId')
        user_solution = data.get('solution')
        
        try:
            # Buscar el puzzle en los puzzles cargados
            puzzle = None
            if self.puzzles:
                for p in self.puzzles:
                    if p.get('id') == puzzle_id:
                        puzzle = p
                        break
            
            # Si no hay puzzles cargados, buscar en los demo
            if not puzzle and puzzle_id.startswith('demo_'):
                demo_puzzles = self._get_demo_puzzles()
                puzzle = demo_puzzles.get(puzzle_id)
            
            if not puzzle:
                # Si no está en memoria ni en demos
                if puzzle_id.startswith('demo_'):
                    # Es un puzzle demo, usar solución hardcodeada
                    expected = self._get_demo_solution(puzzle_id)
                    is_correct = self._compare_grids(user_solution, expected)
                else:
                    # No encontrado
                    expected = None
                    is_correct = False
            else:
                # Obtener la solución esperada
                if puzzle.get('test') and len(puzzle['test']) > 0:
                    if 'output' in puzzle['test'][0]:
                        expected = puzzle['test'][0]['output']
                        is_correct = self._compare_grids(user_solution, expected)
                    else:
                        # No hay solución disponible, usar el solver
                        test_input = np.array(puzzle['test'][0]['input'])
                        solution = self.solver.solve(puzzle['train'], test_input)
                        expected = solution.tolist() if solution is not None else None
                        is_correct = self._compare_grids(user_solution, expected)
                else:
                    expected = None
                    is_correct = False
            
            # Calcular diferencias
            differences = []
            if expected:
                for i in range(min(len(user_solution), len(expected))):
                    for j in range(min(len(user_solution[i]), len(expected[i]))):
                        if user_solution[i][j] != expected[i][j]:
                            differences.append({
                                'row': i,
                                'col': j,
                                'got': user_solution[i][j],
                                'expected': expected[i][j]
                            })
            
            await self.send_message(websocket, {
                'type': 'validation_result',
                'puzzleId': puzzle_id,
                'isCorrect': is_correct,
                'expected': expected,
                'differences': differences,
                'message': '¡Correcto!' if is_correct else f'Incorrecto: {len(differences)} diferencias'
            })
            
        except Exception as e:
            logger.error(f"Error validando solución: {e}")
            await self.send_message(websocket, {
                'type': 'error',
                'message': f'Error al validar solución: {str(e)}'
            })
    
    def _compare_grids(self, grid1, grid2):
        """Compara dos grids para ver si son iguales"""
        if not grid1 or not grid2:
            return False
        if len(grid1) != len(grid2):
            return False
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def _get_demo_solution(self, puzzle_id):
        """Obtiene la solución para puzzles demo - Estilo ARC real"""
        solutions = {
            'demo_1': [[0, 2, 0], [0, 3, 0], [0, 2, 0]],  # Reflexión vertical
            'demo_2': [[4]],  # Contar objetos (4 elementos)
            'demo_3': [[5, 6, 5], [6, 5, 6], [5, 6, 5]],  # Extraer patrón
            'demo_4': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [6, 7, 0]],  # Gravedad
            'arc_001': [[0, 0, 8], [0, 0, 9], [0, 0, 0]],  # Rotación 90°
            'arc_002': [[2, 3, 4, 5], [0, 0, 0, 0], [0, 0, 0, 0]]  # Completar secuencia
        }
        return solutions.get(puzzle_id, None)
    
    def _get_demo_puzzles(self):
        """Devuelve puzzles demo predefinidos"""
        return {
            'demo_1': {
                'id': 'demo_1',
                'train': [
                    {'input': [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
                     'output': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]},
                    {'input': [[0, 2, 0], [0, 0, 0], [0, 0, 0]], 
                     'output': [[2, 2, 2], [2, 2, 2], [2, 2, 2]]}
                ],
                'test': [{'input': [[0, 0, 0], [0, 0, 3], [0, 0, 0]]}]
            },
            'demo_2': {
                'id': 'demo_2',
                'train': [
                    {'input': [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
                     'output': [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]}
                ],
                'test': [{'input': [[5, 0, 0, 0], [0, 6, 0, 0], [0, 0, 7, 0], [0, 0, 0, 8]]}]
            },
            'demo_3': {
                'id': 'demo_3',
                'train': [
                    {'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     'output': [[0, 2, 0], [2, 2, 2], [0, 2, 0]]},
                    {'input': [[3, 3, 3], [3, 0, 3], [3, 3, 3]],
                     'output': [[4, 4, 4], [4, 0, 4], [4, 4, 4]]}
                ],
                'test': [{'input': [[5, 5, 0], [5, 0, 0], [0, 0, 0]]}]
            },
            'demo_4': {
                'id': 'demo_4',
                'train': [
                    {'input': [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                     'output': [[1, 1, 1], [1, 0, 1], [1, 1, 1]]},
                    {'input': [[2, 0, 0], [0, 0, 0], [0, 0, 2]],
                     'output': [[2, 0, 2], [0, 0, 0], [2, 0, 2]]}
                ],
                'test': [{'input': [[3, 0, 0], [0, 0, 0], [0, 0, 3]]}]
            },
            'arc_001': {
                'id': 'arc_001',
                'train': [
                    {'input': [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                     'output': [[1, 1], [1, 1]]},
                    {'input': [[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 2, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]],
                     'output': [[2, 2, 2], [2, 2, 2], [2, 2, 2]]}
                ],
                'test': [{'input': [[0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0], [0, 0, 0, 0, 0, 0]]}]
            },
            'arc_002': {
                'id': 'arc_002',
                'train': [
                    {'input': [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                     'output': [[1, 1, 1], [2, 2, 2], [3, 3, 3]]},
                    {'input': [[4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 7]],
                     'output': [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7]]}
                ],
                'test': [{'input': [[8, 0, 0, 0, 0], [0, 9, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 3]]}]
            }
        }

    async def handle_load_api_puzzles(self, websocket, data: Dict[str, Any]):
        """Carga puzzles desde la API oficial de ARC Prize"""
        game_id = data.get('game_id', 'ls20')
        count = data.get('count', 10)
        
        try:
            # Notificar inicio de carga
            await self.send_message(websocket, {
                'type': 'loading',
                'message': f'Cargando puzzles oficiales del juego {game_id}...'
            })
            
            # Verificar si hay API key
            if not self.arc_api_client.api_key:
                logger.warning("No hay ARC_API_KEY configurada, usando puzzles demo")
                # Usar puzzles demo si no hay API key
                demo_puzzles = self._get_demo_puzzles()
                self.api_puzzles_cache = demo_puzzles
                
                puzzles_list = []
                for puzzle_id, puzzle in demo_puzzles.items():
                    puzzle['id'] = puzzle_id
                    puzzles_list.append(puzzle)
                
                await self.send_message(websocket, {
                    'type': 'api_puzzles_loaded',
                    'puzzles': puzzles_list[:count],
                    'count': len(puzzles_list[:count]),
                    'game_id': 'demo',
                    'message': 'Usando puzzles demo (no hay API key)'
                })
                return
            
            # Cargar puzzles de la API
            async with self.arc_api_client:
                puzzles = await self.arc_api_client.load_game_puzzles(game_id, count)
                
                # Guardar en cache
                for puzzle in puzzles:
                    self.api_puzzles_cache[puzzle['id']] = puzzle
                
                # Enviar puzzles cargados
                await self.send_message(websocket, {
                    'type': 'api_puzzles_loaded',
                    'puzzles': puzzles,
                    'count': len(puzzles),
                    'game_id': game_id,
                    'message': f'Cargados {len(puzzles)} puzzles oficiales de {game_id}'
                })
                
        except Exception as e:
            logger.error(f"Error cargando puzzles de API: {e}")
            await self.send_message(websocket, {
                'type': 'error',
                'message': f'Error cargando puzzles: {str(e)}'
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
        # Ahora siempre usamos formato estándar train/test
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
            
    async def handle_process_image(self, websocket, data: Dict[str, Any]):
        """Procesa una imagen para convertirla en grid ARC"""
        image_data = data.get('image_data')
        cell_size = data.get('cell_size', 30)
        
        if not image_data:
            await self.send_message(websocket, {
                'type': 'error',
                'message': 'No se proporcionó imagen'
            })
            return
        
        # Procesar imagen
        result = self.image_processor.image_to_grid(image_data, cell_size)
        
        if result['success']:
            await self.send_message(websocket, {
                'type': 'image_processed',
                'grid': result['grid'],
                'dimensions': result['dimensions'],
                'original_size': result['original_size']
            })
        else:
            await self.send_message(websocket, {
                'type': 'error',
                'message': f"Error procesando imagen: {result.get('error')}"
            })
    
    async def handle_load_image_puzzle(self, websocket, data: Dict[str, Any]):
        """Carga un puzzle completo desde una imagen"""
        image_data = data.get('image_data')
        
        if not image_data:
            await self.send_message(websocket, {
                'type': 'error',
                'message': 'No se proporcionó imagen'
            })
            return
        
        # Analizar imagen como puzzle
        result = self.image_processor.analyze_image_puzzle(image_data)
        
        if result['success']:
            puzzle = result['puzzle']
            
            # Agregar a la lista de puzzles disponibles
            await self.send_message(websocket, {
                'type': 'image_puzzle_loaded',
                'puzzle': self._serialize_puzzle(puzzle),
                'analysis': result.get('analysis', {}),
                'grids_detected': result.get('grids_detected', 0)
            })
        else:
            await self.send_message(websocket, {
                'type': 'error',
                'message': f"Error analizando imagen: {result.get('error')}"
            })
    
    async def handle_list_puzzles(self, websocket, data):
        """Lista puzzles disponibles"""
        try:
            # Listar archivos de puzzle disponibles
            puzzle_files = self.official_loader.get_sample_puzzles()
            
            await self.send_message(websocket, {
                'type': 'puzzles_list',
                'puzzles': puzzle_files[:20]  # Limitar a 20 para no sobrecargar
            })
        except Exception as e:
            await self.send_message(websocket, {
                'type': 'error',
                'message': f'Error listando puzzles: {str(e)}'
            })
    
    async def handle_load_puzzle(self, websocket, data):
        """Carga un puzzle específico"""
        try:
            puzzle_id = data.get('puzzle_id')
            
            # Cargar puzzle oficial usando load_specific_puzzles con una lista de un elemento
            puzzles = self.official_loader.load_specific_puzzles([puzzle_id])
            
            if puzzles and len(puzzles) > 0:
                puzzle = puzzles[0]
                # Guardar en caché
                self.current_puzzles[puzzle_id] = puzzle
                
                # Generar visualización del puzzle
                visualization_data = {}
                try:
                    # El formato ahora siempre es train/test
                    train_data = puzzle.get('train', [])
                    test_data = puzzle.get('test', [])
                    
                    # Generar imágenes para TODOS los ejemplos de entrenamiento
                    import base64
                    import numpy as np
                    from io import BytesIO
                    
                    visualization_data = {
                        'train_examples': [],
                        'test_examples': []
                    }
                    
                    # Procesar todos los ejemplos de entrenamiento
                    if train_data and len(train_data) > 0:
                        for idx, example in enumerate(train_data):
                            try:
                                # Obtener input y output del ejemplo
                                if hasattr(example, 'input'):
                                    input_grid = example.input
                                    output_grid = example.output
                                else:
                                    input_grid = example.get('input', [])
                                    output_grid = example.get('output', [])
                                
                                # Convertir a numpy array si es necesario
                                if not isinstance(input_grid, np.ndarray):
                                    input_grid = np.array(input_grid)
                                if not isinstance(output_grid, np.ndarray):
                                    output_grid = np.array(output_grid)
                                
                                # Visualizar input y output
                                input_img = self.visualizer.create_grid_image(input_grid, title="Input")
                                output_img = self.visualizer.create_grid_image(output_grid, title="Output")
                                
                                # Input image
                                buffer_input = BytesIO()
                                input_img.save(buffer_input, format='PNG')
                                input_b64 = base64.b64encode(buffer_input.getvalue()).decode('utf-8')
                                
                                # Output image
                                buffer_output = BytesIO()
                                output_img.save(buffer_output, format='PNG')
                                output_b64 = base64.b64encode(buffer_output.getvalue()).decode('utf-8')
                                
                                visualization_data['train_examples'].append({
                                    'index': idx,
                                    'input_image': input_b64,
                                    'output_image': output_b64
                                })
                            except Exception as e:
                                logger.warning(f"Error generando imagen para ejemplo {idx}: {e}")
                    
                    # Procesar ejemplos de test
                    if test_data and len(test_data) > 0:
                        for idx, example in enumerate(test_data):
                            try:
                                if hasattr(example, 'input'):
                                    test_input_grid = example.input
                                    # Test puede no tener output
                                    test_output_grid = example.output if hasattr(example, 'output') else None
                                else:
                                    test_input_grid = example.get('input', [])
                                    test_output_grid = example.get('output')
                                
                                # Convertir a numpy array si es necesario
                                if not isinstance(test_input_grid, np.ndarray):
                                    test_input_grid = np.array(test_input_grid)
                                
                                test_input_img = self.visualizer.create_grid_image(test_input_grid, title="Test Input")
                                buffer_test = BytesIO()
                                test_input_img.save(buffer_test, format='PNG')
                                test_input_b64 = base64.b64encode(buffer_test.getvalue()).decode('utf-8')
                                
                                test_example_data = {
                                    'index': idx,
                                    'input_image': test_input_b64
                                }
                                
                                # Si hay output en el test, generarlo también
                                if test_output_grid:
                                    if not isinstance(test_output_grid, np.ndarray):
                                        test_output_grid = np.array(test_output_grid)
                                    test_output_img = self.visualizer.create_grid_image(test_output_grid, title="Expected Output")
                                    buffer_test_out = BytesIO()
                                    test_output_img.save(buffer_test_out, format='PNG')
                                    test_output_b64 = base64.b64encode(buffer_test_out.getvalue()).decode('utf-8')
                                    test_example_data['output_image'] = test_output_b64
                                
                                visualization_data['test_examples'].append(test_example_data)
                            except Exception as e:
                                logger.warning(f"Error generando imagen para test {idx}: {e}")
                            
                except Exception as e:
                    logger.warning(f"No se pudo generar visualización: {e}")
                
                # Enviar puzzle con visualización
                puzzle_data = self._serialize_puzzle(puzzle)
                puzzle_data['visualization'] = visualization_data
                
                await self.send_message(websocket, {
                    'type': 'puzzle_loaded',
                    'puzzle': puzzle_data
                })
            else:
                await self.send_message(websocket, {
                    'type': 'error',
                    'message': f'Puzzle {puzzle_id} no encontrado'
                })
        except Exception as e:
            await self.send_message(websocket, {
                'type': 'error',
                'message': f'Error cargando puzzle: {str(e)}'
            })
    
    async def handle_process_puzzle_real(self, websocket, data):
        """Procesa un puzzle con el procesador REAL transparente"""
        try:
            import traceback
            puzzle_id = data.get('puzzle_id')
            verbose = data.get('verbose', True)
            
            if puzzle_id not in self.current_puzzles:
                await self.send_message(websocket, {
                    'type': 'error',
                    'message': f'Puzzle {puzzle_id} no está cargado'
                })
                return
            
            # Inicializar procesador real si no existe
            if not self.real_processor and RealARCProcessor:
                self.real_processor = RealARCProcessor(websocket_handler=self)
            elif not RealARCProcessor:
                await self.send_message(websocket, {
                    'type': 'error',
                    'message': 'Procesador real no disponible'
                })
                return
            
            # Procesar puzzle de forma transparente
            puzzle_data = self.current_puzzles[puzzle_id]
            result = await self.real_processor.process_puzzle(puzzle_data, verbose=verbose)
            
            # El procesador ya envía updates en tiempo real
            # Solo enviamos el resultado final si es necesario
            if not result['success']:
                await self.send_message(websocket, {
                    'type': 'error',
                    'message': f"Procesamiento falló: {', '.join(result['errors'])}"
                })
            
        except Exception as e:
            logger.error(f"Error en procesamiento real: {e}")
            logger.error(traceback.format_exc())
            await self.send_message(websocket, {
                'type': 'error',
                'message': f'Error: {str(e)}'
            })
    
    async def send_to_all(self, message):
        """Envía mensaje a todos los clientes conectados"""
        if self.clients:
            await asyncio.gather(
                *[self.send_message(client, message) for client in self.clients],
                return_exceptions=True
            )
    
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