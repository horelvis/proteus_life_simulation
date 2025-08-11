# ARC Solver Module - Versión 4.0
"""
Sistema A-MHA - Anchored Multi-Head Attention con Deep Learning

Sistema de Deep Learning con Atención Anclada Multi-Head para ARC Prize
Arquitectura anti-clásica eficiente con CNN ligera y atención por anclas

Componentes principales:
- DeepLearningARCSolver: Sistema DL con A-MHA y encoder visual
- HierarchicalAttentionSolver: Sistema HAMS de atención jerárquica
- HybridProteusARCSolver: Solver base legacy (para compatibilidad)
"""

from .hybrid_proteus_solver import HybridProteusARCSolver
from .hierarchical_attention_solver import HierarchicalAttentionSolver

# Intentar importar el solver de Deep Learning (requiere PyTorch)
try:
    # Primero intentar la versión corregida
    from .deep_learning_solver_fixed import DeepLearningARCSolver
    DL_AVAILABLE = True
    print("✓ Usando DeepLearningARCSolver versión corregida con nn.Module")
except ImportError:
    try:
        # Fallback a la versión simple si la fixed no existe
        from .deep_learning_solver_simple import DeepLearningARCSolver
        DL_AVAILABLE = True
        print("⚠️ Usando DeepLearningARCSolver versión simple")
    except ImportError:
        DeepLearningARCSolver = None
        DL_AVAILABLE = False
        print("⚠️ PyTorch no disponible. Usando solver HAMS sin Deep Learning.")

# El nuevo solver principal (DL si está disponible, sino HAMS)
ARCSolver = DeepLearningARCSolver if DL_AVAILABLE else HierarchicalAttentionSolver

__version__ = "4.0.0"
__all__ = ["HierarchicalAttentionSolver", "ARCSolver", "HybridProteusARCSolver"]

if DL_AVAILABLE:
    __all__.append("DeepLearningARCSolver")