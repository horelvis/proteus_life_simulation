#!/usr/bin/env python3
"""
Cargador de puzzles oficiales de ARC Prize
Descarga y procesa puzzles del repositorio oficial
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ARCOfficialLoader:
    """
    Carga puzzles oficiales de ARC desde GitHub
    """
    
    def __init__(self, cache_dir: str = "arc_official_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # URLs oficiales de ARC-AGI
        self.base_urls = {
            'arc_agi_1': 'https://raw.githubusercontent.com/fchollet/ARC-AGI/master',
            'arc_agi_2': 'https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main'
        }
        
        # Mapeo de colores ARC (0-9) a colores visuales
        self.color_map = {
            0: '#000000',  # Negro (fondo)
            1: '#0074D9',  # Azul
            2: '#FF4136',  # Rojo
            3: '#2ECC40',  # Verde
            4: '#FFDC00',  # Amarillo
            5: '#AAAAAA',  # Gris
            6: '#F012BE',  # Magenta
            7: '#FF851B',  # Naranja
            8: '#7FDBFF',  # Azul claro
            9: '#870C25'   # Marrón
        }
    
    def load_from_github(self, 
                        dataset: str = 'training',
                        version: str = 'arc_agi_1',
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Carga puzzles directamente desde GitHub
        
        Args:
            dataset: 'training' o 'evaluation'
            version: 'arc_agi_1' o 'arc_agi_2'
            limit: Número máximo de puzzles a cargar
            
        Returns:
            Lista de puzzles
        """
        if version not in self.base_urls:
            raise ValueError(f"Versión desconocida: {version}")
        
        # Intentar cargar índice de archivos
        index_url = f"{self.base_urls[version]}/data/{dataset}/"
        puzzles = []
        
        # Para ARC-AGI-1, tenemos una lista conocida de archivos
        if version == 'arc_agi_1':
            # Cargar algunos puzzles de ejemplo conocidos
            sample_puzzles = [
                '0520fde7.json',
                '0a938d79.json', 
                '0b148d64.json',
                '0ca9ddb6.json',
                '0d3d703e.json',
                '0dfd9992.json',
                '0e206a2e.json',
                '103c5fa9.json',
                '10fcaaa3.json',
                '11852cab.json'
            ]
            
            for i, puzzle_file in enumerate(sample_puzzles):
                if limit and i >= limit:
                    break
                    
                puzzle_url = f"{self.base_urls[version]}/data/{dataset}/{puzzle_file}"
                
                try:
                    # Intentar cargar desde caché primero
                    cache_file = self.cache_dir / f"{version}_{dataset}_{puzzle_file}"
                    
                    if cache_file.exists():
                        logger.info(f"Cargando desde caché: {puzzle_file}")
                        with open(cache_file, 'r') as f:
                            puzzle_data = json.load(f)
                    else:
                        logger.info(f"Descargando: {puzzle_url}")
                        response = requests.get(puzzle_url)
                        response.raise_for_status()
                        puzzle_data = response.json()
                        
                        # Guardar en caché
                        with open(cache_file, 'w') as f:
                            json.dump(puzzle_data, f)
                    
                    # Procesar puzzle
                    puzzle = self._process_puzzle(puzzle_data, puzzle_file.replace('.json', ''))
                    puzzles.append(puzzle)
                    
                except Exception as e:
                    logger.error(f"Error cargando {puzzle_file}: {e}")
                    continue
        
        return puzzles
    
    def _process_puzzle(self, raw_data: Dict, puzzle_id: str) -> Dict[str, Any]:
        """
        Procesa un puzzle crudo al formato esperado por nuestro solver
        
        Args:
            raw_data: Datos crudos del JSON
            puzzle_id: ID del puzzle
            
        Returns:
            Puzzle procesado
        """
        processed = {
            'id': puzzle_id,
            'trainExamples': [],
            'testExample': None,
            'category': 'unknown',  # ARC no proporciona categorías
            'difficulty': 'unknown',
            'source': 'arc_official'
        }
        
        # Procesar ejemplos de entrenamiento
        for train_pair in raw_data.get('train', []):
            processed['trainExamples'].append({
                'input': train_pair['input'],
                'output': train_pair['output']
            })
        
        # Procesar ejemplo de test (usualmente solo hay uno)
        test_pairs = raw_data.get('test', [])
        if test_pairs:
            processed['testExample'] = {
                'input': test_pairs[0]['input'],
                'output': test_pairs[0]['output']
            }
        
        return processed
    
    def analyze_puzzle(self, puzzle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza características de un puzzle
        
        Args:
            puzzle: Puzzle a analizar
            
        Returns:
            Análisis del puzzle
        """
        analysis = {
            'id': puzzle['id'],
            'num_train_examples': len(puzzle['trainExamples']),
            'grid_sizes': [],
            'unique_colors': set(),
            'complexity_score': 0
        }
        
        # Analizar tamaños de grilla
        all_examples = puzzle['trainExamples'] + ([puzzle['testExample']] if puzzle['testExample'] else [])
        
        for example in all_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            analysis['grid_sizes'].append({
                'input': input_grid.shape,
                'output': output_grid.shape
            })
            
            # Colores únicos
            analysis['unique_colors'].update(input_grid.flatten())
            analysis['unique_colors'].update(output_grid.flatten())
        
        # Calcular score de complejidad simple
        analysis['complexity_score'] = (
            len(analysis['unique_colors']) * 
            len(analysis['grid_sizes']) * 
            max(max(s['input'][0], s['input'][1], s['output'][0], s['output'][1]) 
                for s in analysis['grid_sizes'])
        )
        
        analysis['unique_colors'] = list(analysis['unique_colors'])
        
        return analysis
    
    def create_submission_format(self, solutions: Dict[str, List[List[int]]]) -> Dict[str, Any]:
        """
        Crea formato de submission para ARC Prize
        
        Args:
            solutions: Diccionario {puzzle_id: solution_grid}
            
        Returns:
            Submission en formato correcto
        """
        submission = {}
        
        for puzzle_id, solution in solutions.items():
            # ARC espera el output como una lista de intentos
            # Podemos proporcionar hasta 3 intentos
            submission[puzzle_id] = [solution]
        
        return submission
    
    def evaluate_solution(self, 
                         puzzle: Dict[str, Any],
                         predicted: List[List[int]]) -> Dict[str, Any]:
        """
        Evalúa una solución contra el output esperado
        
        Args:
            puzzle: Puzzle con respuesta correcta
            predicted: Solución predicha
            
        Returns:
            Métricas de evaluación
        """
        if not puzzle.get('testExample'):
            return {'error': 'No test example available'}
        
        expected = np.array(puzzle['testExample']['output'])
        predicted = np.array(predicted)
        
        # Verificar dimensiones
        if expected.shape != predicted.shape:
            return {
                'correct': False,
                'accuracy': 0.0,
                'shape_mismatch': True,
                'expected_shape': expected.shape,
                'predicted_shape': predicted.shape
            }
        
        # Calcular precisión
        correct_pixels = np.sum(expected == predicted)
        total_pixels = expected.size
        accuracy = correct_pixels / total_pixels
        
        return {
            'correct': np.array_equal(expected, predicted),
            'accuracy': accuracy,
            'correct_pixels': int(correct_pixels),
            'total_pixels': int(total_pixels),
            'shape_mismatch': False
        }
    
    def load_specific_puzzles(self, puzzle_ids: List[str], 
                            dataset: str = 'training',
                            version: str = 'arc_agi_1') -> List[Dict[str, Any]]:
        """
        Carga puzzles específicos por ID
        
        Args:
            puzzle_ids: Lista de IDs de puzzles
            dataset: 'training' o 'evaluation'
            version: Versión de ARC
            
        Returns:
            Lista de puzzles cargados
        """
        puzzles = []
        
        for puzzle_id in puzzle_ids:
            puzzle_url = f"{self.base_urls[version]}/data/{dataset}/{puzzle_id}.json"
            
            try:
                cache_file = self.cache_dir / f"{version}_{dataset}_{puzzle_id}.json"
                
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        puzzle_data = json.load(f)
                else:
                    response = requests.get(puzzle_url)
                    response.raise_for_status()
                    puzzle_data = response.json()
                    
                    with open(cache_file, 'w') as f:
                        json.dump(puzzle_data, f)
                
                puzzle = self._process_puzzle(puzzle_data, puzzle_id)
                puzzles.append(puzzle)
                
            except Exception as e:
                logger.error(f"Error cargando {puzzle_id}: {e}")
                continue
        
        return puzzles
    
    def get_sample_puzzles(self) -> List[str]:
        """
        Retorna una lista de IDs de puzzles de muestra para testing
        """
        # Estos son puzzles conocidos que varían en dificultad
        # Verificados que existen en el repositorio
        return [
            '00d62c1b',  # Simple patterns
            '0520fde7',  # Pattern completion
            '0a938d79',  # Geometric transformations
            '0b148d64',  # Color patterns
            '0ca9ddb6',  # Shape manipulation
            '0d3d703e',  # Spatial reasoning
            '0e206a2e',  # Pattern extraction
            '10fcaaa3',  # Complex pattern
            '11852cab',  # Grid transformations
            '1190e5a7'   # Color transformations
        ]


if __name__ == "__main__":
    # Test del cargador
    loader = ARCOfficialLoader()
    
    print("🎮 Cargando puzzles oficiales de ARC...\n")
    
    # Cargar algunos puzzles de muestra
    puzzles = loader.load_from_github(dataset='training', limit=5)
    
    print(f"✅ Cargados {len(puzzles)} puzzles\n")
    
    # Analizar puzzles
    for puzzle in puzzles:
        analysis = loader.analyze_puzzle(puzzle)
        print(f"📊 Puzzle {analysis['id']}:")
        print(f"   - Ejemplos de entrenamiento: {analysis['num_train_examples']}")
        print(f"   - Colores únicos: {sorted(analysis['unique_colors'])}")
        print(f"   - Score de complejidad: {analysis['complexity_score']}")
        print(f"   - Tamaños de grilla: {analysis['grid_sizes'][0]}")
        print()