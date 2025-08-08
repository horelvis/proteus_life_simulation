#!/usr/bin/env python3
"""
Run PROTEUS with Vispy GPU-accelerated backend
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proteus_vispy import ProteusVispy

if __name__ == "__main__":
    print("Starting PROTEUS Vispy Backend...")
    print("=" * 50)
    
    # Create and start simulation
    sim = ProteusVispy(world_size=(1600, 1200), max_organisms=10000)
    
    print("Controls:")
    print("- Mouse: Pan camera")
    print("- Scroll: Zoom")
    print("- Space: Pause/Resume")
    print("- R: Reset simulation")
    print("- ESC: Exit")
    print("=" * 50)
    
    sim.start()