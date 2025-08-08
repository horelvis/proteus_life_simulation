#!/usr/bin/env python3
"""
PROTEUS - Proto-Topological Evolution System
A revolutionary approach to artificial life without neural networks
"""

__version__ = "0.2.0"
__author__ = "PROTEUS Team"

from . import core
from . import genetics
from . import environment

__all__ = [
    'core',
    'genetics', 
    'environment',
    '__version__'
]