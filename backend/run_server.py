#!/usr/bin/env python3
"""
Run PROTEUS backend server with WebSocket support
"""

import sys
import os
import asyncio
import threading
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proteus_vispy import app, ProteusVispy

def run_vispy_in_thread(sim):
    """Run Vispy in a separate thread"""
    sim.start()

if __name__ == "__main__":
    print("Starting PROTEUS Backend Server...")
    print("=" * 50)
    print("WebSocket endpoint: ws://localhost:8000/ws/{client_id}")
    print("Stats endpoint: http://localhost:8000/stats")
    print("=" * 50)
    
    # Run with FastAPI server mode
    uvicorn.run(app, host="0.0.0.0", port=8000)