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

# Configurar CORS para desarrollo si no está definido
if [ -z "$CORS_ORIGINS" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        DEV_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
    else
        DEV_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$DEV_IP" ]; then DEV_IP=localhost; fi
    export CORS_ORIGINS='["http://localhost:3000","http://localhost:3001","http://127.0.0.1:3000","http://127.0.0.1:3001","http://'$DEV_IP':3000","http://'$DEV_IP':3001","http://'$DEV_IP'"]'
fi

# Iniciar servidor unificado
echo "Starting unified server..."
echo "📡 FastAPI: http://0.0.0.0:8000"
echo "🔌 ARC WebSocket: ws://0.0.0.0:8765"
echo "CORS_ORIGINS=$CORS_ORIGINS"
echo "========================"

python3 unified_server.py
