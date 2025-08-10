#!/bin/bash

# PROTEUS Unified Server Start Script

echo "🧬 PROTEUS Unified Server"
echo "========================"

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Instalar dependencias si es necesario
if [ ! -f ".deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Iniciar servidor unificado
echo "Starting unified server..."
echo "📡 FastAPI: http://localhost:8000"
echo "🔌 ARC WebSocket: ws://localhost:8765"
echo "========================"

python3 unified_server.py