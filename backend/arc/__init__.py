"""
ARC Solver Package
------------------

Este paquete contiene la implementación del solver de puzzles ARC
basado en el razonamiento topológico adaptativo.

El solver principal es `HybridProteusARCSolver`.
"""

from .hybrid_proteus_solver import HybridProteusARCSolver
from .arc_solver_python import ARCSolverPython
from .transformations_fixed import RealTransformations

# El solver híbrido es la implementación principal y recomendada.
ARCSolver = HybridProteusARCSolver

print("🧠 ARC Solver con Razonamiento Topológico Adaptativo inicializado.")

__version__ = "7.0.0" # Version refactorizada
__all__ = ["HybridProteusARCSolver", "ARCSolverPython", "RealTransformations", "ARCSolver"]
