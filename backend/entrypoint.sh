#!/bin/bash

# Start Xvfb for headless OpenGL support (for legacy compatibility)
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Wait for Xvfb to start
sleep 2

# Start ARC WebSocket server only
python arc/arc_server.py