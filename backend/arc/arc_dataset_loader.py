"""
ARC Dataset Loader
Carga puzzles del dataset ARC
"""

import json
import os
import random
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ARCDatasetLoader:
    def __init__(self):
        # Los mismos puzzles que en el frontend para consistencia
        self.sample_puzzles = [
            # 1. GRAVEDAD/CAÍDA
            {
                "id": "00d62c1b",
                "category": "physics_gravity",
                "difficulty": "medium",
                "train": [
                    {
                        "input": [[0, 4, 0, 9], [0, 0, 0, 0], [0, 4, 6, 0], [1, 0, 0, 0]],
                        "output": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [1, 4, 6, 9]]
                    },
                    {
                        "input": [[0, 0, 0, 0, 0, 9], [0, 0, 0, 8, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [4, 0, 7, 8, 0, 0], [4, 0, 7, 0, 0, 0]],
                        "output": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0], [4, 0, 7, 8, 0, 0], [4, 0, 7, 8, 0, 9]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 2, 0, 4, 3], [5, 0, 0, 0, 0], [0, 0, 6, 0, 0], [5, 2, 0, 4, 0], [5, 0, 0, 0, 0]],
                        "output": [[0, 0, 0, 0, 0], [5, 0, 0, 0, 0], [5, 2, 0, 0, 0], [5, 2, 6, 4, 0], [5, 2, 6, 4, 3]]
                    }
                ]
            },
            
            # 2. REPLICACIÓN DE PATRONES 3x3
            {
                "id": "007bbfb7",
                "category": "pattern_replication",
                "difficulty": "easy",
                "train": [
                    {
                        "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
                        "output": [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[4, 0, 4], [0, 0, 0], [0, 4, 0]],
                        "output": [[4, 0, 4, 4, 0, 4, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 4, 0, 0, 4, 0], [4, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 0], [4, 0, 4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 0, 0]]
                    }
                ]
            },
            
            # 3. CAMBIO DE COLOR SIMPLE
            {
                "id": "42a50994",
                "category": "color_mapping",
                "difficulty": "very_easy",
                "train": [
                    {
                        "input": [[1, 1, 0], [0, 1, 0], [1, 1, 1]],
                        "output": [[4, 4, 0], [0, 4, 0], [4, 4, 4]]
                    },
                    {
                        "input": [[0, 1, 1], [1, 0, 1], [0, 0, 1]],
                        "output": [[0, 4, 4], [4, 0, 4], [0, 0, 4]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 0, 1], [1, 1, 0], [0, 1, 0]],
                        "output": [[4, 0, 4], [4, 4, 0], [0, 4, 0]]
                    }
                ]
            },
            
            # 4. REFLEXIÓN/ESPEJO
            {
                "id": "4258a5f9",
                "category": "reflection",
                "difficulty": "easy",
                "train": [
                    {
                        "input": [[0, 0, 3], [0, 5, 0], [8, 0, 0]],
                        "output": [[3, 0, 0], [0, 5, 0], [0, 0, 8]]
                    },
                    {
                        "input": [[0, 2, 0], [7, 0, 0], [0, 0, 0]],
                        "output": [[0, 2, 0], [0, 0, 7], [0, 0, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 0, 0], [0, 6, 0], [1, 0, 0]],
                        "output": [[0, 0, 0], [0, 6, 0], [0, 0, 1]]
                    }
                ]
            },
            
            # 5. CONTAR OBJETOS
            {
                "id": "b27ca6d3",
                "category": "counting",
                "difficulty": "medium",
                "train": [
                    {
                        "input": [[0, 2, 0, 2], [0, 0, 2, 0], [0, 2, 0, 0], [2, 0, 0, 2]],
                        "output": [[6]]
                    },
                    {
                        "input": [[5, 0, 0], [0, 5, 0], [0, 0, 0]],
                        "output": [[2]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 3, 3], [3, 0, 0], [0, 3, 0]],
                        "output": [[4]]
                    }
                ]
            },
            
            # 6. RELLENAR FORMA
            {
                "id": "a416b8f3",
                "category": "fill_shape",
                "difficulty": "medium",
                "train": [
                    {
                        "input": [[0, 0, 0, 0, 0], [0, 8, 8, 8, 0], [0, 8, 0, 8, 0], [0, 8, 8, 8, 0], [0, 0, 0, 0, 0]],
                        "output": [[0, 0, 0, 0, 0], [0, 8, 8, 8, 0], [0, 8, 3, 8, 0], [0, 8, 8, 8, 0], [0, 0, 0, 0, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 0], [0, 8, 0, 0, 0, 8, 0], [0, 8, 0, 0, 0, 8, 0], [0, 8, 8, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0]],
                        "output": [[0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 8, 8, 0], [0, 8, 3, 3, 3, 8, 0], [0, 8, 3, 3, 3, 8, 0], [0, 8, 8, 8, 8, 8, 0], [0, 0, 0, 0, 0, 0, 0]]
                    }
                ]
            },
            
            # 7. SIMETRÍA
            {
                "id": "d631b094",
                "category": "symmetry_detection",
                "difficulty": "hard",
                "train": [
                    {
                        "input": [[0, 0, 0, 0, 0], [0, 8, 0, 8, 0], [0, 0, 3, 0, 0], [0, 8, 0, 8, 0], [0, 0, 0, 0, 0]],
                        "output": [[1]]
                    },
                    {
                        "input": [[0, 0, 0, 0], [5, 0, 0, 5], [0, 2, 0, 0], [5, 0, 0, 5]],
                        "output": [[0]]
                    }
                ],
                "test": [
                    {
                        "input": [[7, 0, 7], [0, 4, 0], [7, 0, 7]],
                        "output": [[1]]
                    }
                ]
            },
            
            # 8. ROTACIÓN 90°
            {
                "id": "ed36ccf7",
                "category": "rotation",
                "difficulty": "medium",
                "train": [
                    {
                        "input": [[1, 2, 0], [0, 0, 0], [0, 0, 3]],
                        "output": [[0, 0, 3], [0, 0, 0], [1, 2, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[5, 0, 0], [0, 6, 0], [0, 0, 0]],
                        "output": [[0, 0, 0], [0, 6, 0], [5, 0, 0]]
                    }
                ]
            },
            
            # 9. EXTRACCIÓN DE PATRÓN
            {
                "id": "bb43febb",
                "category": "pattern_extraction",
                "difficulty": "hard",
                "train": [
                    {
                        "input": [[0, 0, 0, 0, 0, 0], [0, 3, 3, 0, 0, 0], [0, 3, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 5, 5, 0], [0, 0, 0, 5, 5, 0]],
                        "output": [[3, 3], [3, 3], [5, 5], [5, 5]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 0, 0, 0], [0, 7, 7, 0], [0, 7, 7, 0], [0, 0, 0, 0]],
                        "output": [[7, 7], [7, 7]]
                    }
                ]
            },
            
            # 10. LÍNEAS Y CONEXIONES
            {
                "id": "ce22a75a",
                "category": "line_drawing",
                "difficulty": "very_hard",
                "train": [
                    {
                        "input": [[2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]],
                        "output": [[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]]
                    }
                ],
                "test": [
                    {
                        "input": [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0]],
                        "output": [[0, 0, 3, 0], [0, 3, 0, 0], [0, 3, 0, 0], [3, 0, 0, 0]]
                    }
                ]
            }
        ]
        
    def load_puzzles(self, puzzle_set: str = 'training', count: int = 10) -> List[Dict[str, Any]]:
        """Carga puzzles del dataset"""
        logger.info(f"Cargando {count} puzzles del conjunto {puzzle_set}")
        
        # Por ahora usar los puzzles de ejemplo
        puzzles = self.sample_puzzles[:count]
        
        # En producción, cargaríamos desde archivos JSON reales
        # if os.path.exists(f"arc_dataset/{puzzle_set}"):
        #     puzzles = self._load_from_files(puzzle_set, count)
        
        return puzzles
        
    def load_puzzle_by_id(self, puzzle_id: str) -> Optional[Dict[str, Any]]:
        """Carga un puzzle específico por ID"""
        for puzzle in self.sample_puzzles:
            if puzzle['id'] == puzzle_id:
                return puzzle
        return None
        
    def get_puzzle_categories(self) -> List[str]:
        """Obtiene todas las categorías disponibles"""
        categories = set()
        for puzzle in self.sample_puzzles:
            if 'category' in puzzle:
                categories.add(puzzle['category'])
        return sorted(list(categories))
        
    def get_puzzles_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Obtiene puzzles de una categoría específica"""
        return [p for p in self.sample_puzzles if p.get('category') == category]
        
    def get_puzzles_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Obtiene puzzles de una dificultad específica"""
        return [p for p in self.sample_puzzles if p.get('difficulty') == difficulty]
        
    def _load_from_files(self, puzzle_set: str, count: int) -> List[Dict[str, Any]]:
        """Carga puzzles desde archivos JSON (para producción)"""
        puzzles = []
        dataset_path = f"arc_dataset/{puzzle_set}"
        
        if os.path.exists(dataset_path):
            files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            random.shuffle(files)
            
            for file in files[:count]:
                with open(os.path.join(dataset_path, file), 'r') as f:
                    puzzle_data = json.load(f)
                    puzzle_data['id'] = file.replace('.json', '')
                    puzzles.append(puzzle_data)
                    
        return puzzles
        
    def validate_puzzle(self, puzzle: Dict[str, Any]) -> bool:
        """Valida que un puzzle tenga el formato correcto"""
        required_fields = ['train', 'test']
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in puzzle:
                logger.error(f"Puzzle sin campo {field}")
                return False
                
        # Verificar que train tenga al menos un ejemplo
        if not puzzle['train'] or len(puzzle['train']) == 0:
            logger.error("Puzzle sin ejemplos de entrenamiento")
            return False
            
        # Verificar formato de ejemplos
        for example in puzzle['train']:
            if 'input' not in example or 'output' not in example:
                logger.error("Ejemplo sin input/output")
                return False
                
            # Verificar que sean arrays
            try:
                np.array(example['input'])
                np.array(example['output'])
            except:
                logger.error("Input/output no son arrays válidos")
                return False
                
        return True