# ARC Solver Module - Versión Final
"""
Sistema PROTEUS-ARC Híbrido Avanzado

Versión definitiva con:
- Análisis estructural profundo con grafos y segmentación
- Análisis topológico multiescala
- Síntesis automática de reglas
- Memoria holográfica

Componentes principales:
- HybridProteusARCSolver: Solver principal con todas las mejoras
- StructuralAnalyzer: Análisis de conectividad, componentes y simetrías
- TopologicalAnalyzer: Análisis de propiedades topológicas
"""

from .hybrid_proteus_solver import HybridProteusARCSolver
from .structural_analyzer import StructuralAnalyzer

__version__ = "2.0.0"
__all__ = ["HybridProteusARCSolver", "StructuralAnalyzer"]