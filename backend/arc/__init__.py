# ARC Logical Reasoning Network - Versión 6.0
"""
Red de Razonamiento Lógico Puro

Sistema de 3 capas sin simulación de vida:
- MACRO: Observación con V-JEPA (alto nivel)
- MESO: Razonamiento sobre objetos
- MICRO: Ejecución a nivel de píxeles

Flujo: Macro → Meso → Micro
"""

from .logical_reasoning_network import LogicalReasoningNetwork

# Red de razonamiento lógico como sistema principal
ARCSolver = LogicalReasoningNetwork

print("🧠 Red de Razonamiento Lógico (Macro→Meso→Micro)")

__version__ = "6.0.0"
__all__ = ["LogicalReasoningNetwork", "ARCSolver"]