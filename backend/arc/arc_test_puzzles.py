#!/usr/bin/env python3
"""
Puzzles de prueba estilo ARC reales
Basados en patrones comunes del dataset oficial
"""

import numpy as np
from typing import Dict, List

class ARCTestPuzzles:
    """Colección de puzzles realistas estilo ARC para pruebas"""
    
    @staticmethod
    def get_all_puzzles() -> List[Dict]:
        """Retorna todos los puzzles de prueba"""
        return [
            ARCTestPuzzles.puzzle_pattern_completion(),
            ARCTestPuzzles.puzzle_symmetry_detection(),
            ARCTestPuzzles.puzzle_object_counting(),
            ARCTestPuzzles.puzzle_color_abstraction(),
            ARCTestPuzzles.puzzle_gravity_simulation(),
            ARCTestPuzzles.puzzle_shape_transformation(),
            ARCTestPuzzles.puzzle_grid_partition(),
            ARCTestPuzzles.puzzle_pattern_extension(),
            ARCTestPuzzles.puzzle_object_sorting(),
            ARCTestPuzzles.puzzle_boundary_tracing()
        ]
    
    @staticmethod
    def puzzle_pattern_completion():
        """Puzzle 1: Completar patrón en esquinas"""
        return {
            "name": "pattern_completion",
            "description": "Completar el patrón en las esquinas del grid",
            "train": [
                {
                    "input": np.array([
                        [2, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 2]
                    ]),
                    "output": np.array([
                        [2, 0, 0, 0, 2],
                        [0, 2, 0, 2, 0],
                        [0, 0, 2, 0, 0],
                        [0, 2, 0, 2, 0],
                        [2, 0, 0, 0, 2]
                    ])
                },
                {
                    "input": np.array([
                        [3, 0, 0, 3],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [3, 0, 0, 3]
                    ]),
                    "output": np.array([
                        [3, 0, 0, 3],
                        [0, 3, 3, 0],
                        [0, 3, 3, 0],
                        [3, 0, 0, 3]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1]
                ]),
                "output": np.array([
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1]
                ])
            }
        }
    
    @staticmethod
    def puzzle_symmetry_detection():
        """Puzzle 2: Detectar y completar simetría"""
        return {
            "name": "symmetry_detection",
            "description": "Completar la mitad faltante con simetría",
            "train": [
                {
                    "input": np.array([
                        [0, 0, 0, 8, 0],
                        [0, 0, 8, 8, 0],
                        [0, 8, 8, 8, 0],
                        [0, 0, 8, 8, 0],
                        [0, 0, 0, 8, 0]
                    ]),
                    "output": np.array([
                        [0, 8, 0, 8, 0],
                        [0, 8, 8, 8, 0],
                        [0, 8, 8, 8, 0],
                        [0, 8, 8, 8, 0],
                        [0, 8, 0, 8, 0]
                    ])
                },
                {
                    "input": np.array([
                        [0, 0, 4],
                        [0, 4, 4],
                        [4, 4, 4]
                    ]),
                    "output": np.array([
                        [4, 0, 4],
                        [4, 4, 4],
                        [4, 4, 4]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 0, 0, 0, 5],
                    [0, 0, 0, 5, 5],
                    [0, 0, 5, 5, 5],
                    [0, 0, 0, 5, 5],
                    [0, 0, 0, 0, 5]
                ]),
                "output": np.array([
                    [5, 0, 0, 0, 5],
                    [5, 5, 0, 5, 5],
                    [5, 5, 5, 5, 5],
                    [5, 5, 0, 5, 5],
                    [5, 0, 0, 0, 5]
                ])
            }
        }
    
    @staticmethod
    def puzzle_object_counting():
        """Puzzle 3: Contar objetos y marcar el más grande"""
        return {
            "name": "object_counting",
            "description": "Resaltar el objeto más grande",
            "train": [
                {
                    "input": np.array([
                        [0, 1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 2, 0],
                        [0, 0, 0, 0, 2, 0],
                        [3, 3, 3, 0, 2, 0],
                        [3, 3, 3, 0, 0, 0],
                        [3, 3, 3, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [9, 9, 9, 0, 0, 0],
                        [9, 9, 9, 0, 0, 0],
                        [9, 9, 9, 0, 0, 0]
                    ])
                },
                {
                    "input": np.array([
                        [4, 4, 0, 0],
                        [4, 4, 0, 6],
                        [4, 4, 0, 6],
                        [0, 0, 0, 6]
                    ]),
                    "output": np.array([
                        [9, 9, 0, 0],
                        [9, 9, 0, 0],
                        [9, 9, 0, 0],
                        [0, 0, 0, 0]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 7, 0, 0, 0],
                    [0, 7, 0, 5, 5],
                    [0, 0, 0, 5, 5],
                    [2, 2, 0, 5, 5],
                    [2, 2, 0, 5, 5]
                ]),
                "output": np.array([
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 9, 9],
                    [0, 0, 0, 9, 9],
                    [0, 0, 0, 9, 9],
                    [0, 0, 0, 9, 9]
                ])
            }
        }
    
    @staticmethod
    def puzzle_color_abstraction():
        """Puzzle 4: Abstracción de colores a patrones"""
        return {
            "name": "color_abstraction",
            "description": "Convertir colores en patrones específicos",
            "train": [
                {
                    "input": np.array([
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [3, 3, 4, 4],
                        [3, 3, 4, 4]
                    ]),
                    "output": np.array([
                        [1, 0, 0, 2],
                        [0, 1, 2, 0],
                        [0, 3, 4, 0],
                        [3, 0, 0, 4]
                    ])
                },
                {
                    "input": np.array([
                        [5, 5, 5],
                        [6, 6, 6],
                        [7, 7, 7]
                    ]),
                    "output": np.array([
                        [5, 0, 5],
                        [0, 6, 0],
                        [7, 0, 7]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [8, 8, 9, 9],
                    [8, 8, 9, 9],
                    [1, 1, 2, 2],
                    [1, 1, 2, 2]
                ]),
                "output": np.array([
                    [8, 0, 0, 9],
                    [0, 8, 9, 0],
                    [0, 1, 2, 0],
                    [1, 0, 0, 2]
                ])
            }
        }
    
    @staticmethod
    def puzzle_gravity_simulation():
        """Puzzle 5: Simular gravedad con objetos cayendo"""
        return {
            "name": "gravity_simulation",
            "description": "Los objetos caen hasta tocar el fondo o otro objeto",
            "train": [
                {
                    "input": np.array([
                        [0, 2, 0, 0, 3],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 4, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 2, 0, 4, 3]
                    ])
                },
                {
                    "input": np.array([
                        [5, 0, 6],
                        [0, 0, 0],
                        [0, 7, 0],
                        [0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [5, 7, 6]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 8, 0, 0],
                    [9, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                "output": np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [9, 8, 0, 1]
                ])
            }
        }
    
    @staticmethod
    def puzzle_shape_transformation():
        """Puzzle 6: Transformar forma según regla"""
        return {
            "name": "shape_transformation",
            "description": "Rotar o reflejar formas según su color",
            "train": [
                {
                    "input": np.array([
                        [0, 0, 0, 0],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]
                    ])
                },
                {
                    "input": np.array([
                        [0, 2, 2, 0],
                        [0, 0, 2, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0],
                        [0, 2, 2, 2],
                        [0, 2, 0, 0],
                        [0, 0, 0, 0]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ]),
                "output": np.array([
                    [0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0]
                ])
            }
        }
    
    @staticmethod
    def puzzle_grid_partition():
        """Puzzle 7: Particionar grid en regiones"""
        return {
            "name": "grid_partition",
            "description": "Dividir el grid en regiones basadas en objetos",
            "train": [
                {
                    "input": np.array([
                        [1, 0, 0, 2],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [3, 0, 0, 4]
                    ]),
                    "output": np.array([
                        [1, 1, 2, 2],
                        [1, 1, 2, 2],
                        [3, 3, 4, 4],
                        [3, 3, 4, 4]
                    ])
                },
                {
                    "input": np.array([
                        [5, 0, 6],
                        [0, 0, 0],
                        [7, 0, 8]
                    ]),
                    "output": np.array([
                        [5, 6, 6],
                        [5, 0, 8],
                        [7, 7, 8]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [1, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [3, 0, 0, 0, 4]
                ]),
                "output": np.array([
                    [1, 1, 0, 2, 2],
                    [1, 1, 0, 2, 2],
                    [0, 0, 0, 0, 0],
                    [3, 3, 0, 4, 4],
                    [3, 3, 0, 4, 4]
                ])
            }
        }
    
    @staticmethod
    def puzzle_pattern_extension():
        """Puzzle 8: Extender patrón detectado"""
        return {
            "name": "pattern_extension",
            "description": "Continuar el patrón detectado",
            "train": [
                {
                    "input": np.array([
                        [1, 2, 1, 0, 0],
                        [2, 1, 2, 0, 0],
                        [1, 2, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [1, 2, 1, 2, 1],
                        [2, 1, 2, 1, 2],
                        [1, 2, 1, 2, 1],
                        [2, 1, 2, 1, 2],
                        [1, 2, 1, 2, 1]
                    ])
                },
                {
                    "input": np.array([
                        [3, 3, 0, 0],
                        [3, 3, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [3, 3, 3, 3],
                        [3, 3, 3, 3],
                        [3, 3, 3, 3],
                        [3, 3, 3, 3]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [4, 0, 4, 0, 0, 0],
                    [0, 4, 0, 0, 0, 0],
                    [4, 0, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]),
                "output": np.array([
                    [4, 0, 4, 0, 4, 0],
                    [0, 4, 0, 4, 0, 4],
                    [4, 0, 4, 0, 4, 0],
                    [0, 4, 0, 4, 0, 4],
                    [4, 0, 4, 0, 4, 0],
                    [0, 4, 0, 4, 0, 4]
                ])
            }
        }
    
    @staticmethod
    def puzzle_object_sorting():
        """Puzzle 9: Ordenar objetos por tamaño"""
        return {
            "name": "object_sorting",
            "description": "Ordenar objetos de izquierda a derecha por tamaño",
            "train": [
                {
                    "input": np.array([
                        [0, 2, 2, 0, 1, 0, 3],
                        [0, 2, 2, 0, 0, 0, 3],
                        [0, 0, 0, 0, 0, 0, 3],
                        [0, 0, 0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [1, 0, 3, 0, 2, 2, 0],
                        [0, 0, 3, 0, 2, 2, 0],
                        [0, 0, 3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]
                    ])
                },
                {
                    "input": np.array([
                        [5, 5, 0, 4],
                        [5, 5, 0, 0],
                        [5, 5, 0, 0]
                    ]),
                    "output": np.array([
                        [4, 0, 5, 5],
                        [0, 0, 5, 5],
                        [0, 0, 5, 5]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 7, 7, 0, 6, 0, 0],
                    [0, 7, 7, 0, 6, 0, 8],
                    [0, 7, 7, 0, 6, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]
                ]),
                "output": np.array([
                    [8, 0, 6, 0, 7, 7, 0],
                    [0, 0, 6, 0, 7, 7, 0],
                    [0, 0, 6, 0, 7, 7, 0],
                    [0, 0, 0, 0, 0, 0, 0]
                ])
            }
        }
    
    @staticmethod
    def puzzle_boundary_tracing():
        """Puzzle 10: Trazar el borde de objetos"""
        return {
            "name": "boundary_tracing",
            "description": "Marcar solo el borde de los objetos",
            "train": [
                {
                    "input": np.array([
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]
                    ]),
                    "output": np.array([
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0]
                    ])
                },
                {
                    "input": np.array([
                        [2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2]
                    ]),
                    "output": np.array([
                        [2, 2, 2, 2],
                        [2, 0, 0, 2],
                        [2, 0, 0, 2],
                        [2, 2, 2, 2]
                    ])
                }
            ],
            "test": {
                "input": np.array([
                    [0, 0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0]
                ]),
                "output": np.array([
                    [0, 0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 3, 0, 0, 3, 0],
                    [0, 3, 0, 0, 3, 0],
                    [0, 3, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0, 0]
                ])
            }
        }