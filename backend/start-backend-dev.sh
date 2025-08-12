#!/bin/bash

set -e

echo "🧪 PROTEUS Backend - Desarrollo (unificado)"
echo "=========================================="

# Verificar Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python3 no encontrado. Instala Python 3.9+"
  exit 1
fi

# Crear venv si no existe
if [ ! -d "venv" ]; then
  echo "📦 Creando entorno virtual..."
  python3 -m venv venv
fi

echo "✅ Activando entorno virtual..."
source venv/bin/activate

# Instalar deps si faltan
if [ ! -f ".deps_installed" ]; then
  echo "🔧 Instalando dependencias..."
  pip install -r requirements.txt
  touch .deps_installed
fi

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

# Configurar CORS para dev (localhost, 127.0.0.1 e IP LAN)
if [[ "$OSTYPE" == "darwin"* ]]; then
  DEV_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
else
  DEV_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
[ -z "$DEV_IP" ] && DEV_IP=localhost

export CORS_ORIGINS='["http://localhost:3000","http://localhost:3001","http://127.0.0.1:3000","http://127.0.0.1:3001","http://'$DEV_IP':3000","http://'$DEV_IP':3001","http://'$DEV_IP'"]'

echo ""
echo "🔗 Endpoints de desarrollo:"
echo "   FastAPI:      http://0.0.0.0:8000 (http://$DEV_IP:8000)"
echo "   ARC WebSocket ws://0.0.0.0:8765 (ws://$DEV_IP:8765)"
echo "   CORS_ORIGINS: $CORS_ORIGINS"
echo ""

echo "🚀 Iniciando servidor unificado..."
exec python3 unified_server.py

