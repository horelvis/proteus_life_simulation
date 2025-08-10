#!/bin/bash

echo "🚀 Iniciando servidor ARC WebSocket..."

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Instalar dependencias si no están instaladas
pip install -r requirements.txt

# Iniciar servidor
python arc_server.py