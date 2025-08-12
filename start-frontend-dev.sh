#!/bin/bash

# Script para ejecutar el frontend en modo desarrollo
# Esto permite cambios en tiempo real sin necesidad de rebuild

echo "🚀 Iniciando frontend en modo desarrollo..."
echo "================================================"
echo "Frontend: http://localhost:3001 (o http://<IP-LAN>:3001)"
echo "Backend API: http://localhost:8000"
echo "WebSocket ARC: ws://<IP-LAN>:8765"
echo "================================================"

# Detectar IP LAN para permitir acceso desde otros dispositivos
if [[ "$OSTYPE" == "darwin"* ]]; then
  DEV_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
else
  DEV_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi

if [ -z "$DEV_IP" ]; then
  DEV_IP=localhost
fi

# Configurar variables de entorno
export REACT_APP_BACKEND_URL=${REACT_APP_BACKEND_URL:-http://$DEV_IP:8000}
export REACT_APP_WS_URL=${REACT_APP_WS_URL:-ws://$DEV_IP:8000}
export REACT_APP_ARC_WS_URL=${REACT_APP_ARC_WS_URL:-ws://$DEV_IP:8765}
export PORT=${PORT:-3001}
# Servir CRA en 0.0.0.0 para acceso LAN
export HOST=0.0.0.0
# Evitar que se abra el navegador automáticamente
export BROWSER=none

# Ir al directorio del frontend
cd frontend

# Instalar dependencias si no existen
if [ ! -d "node_modules" ]; then
    echo "📦 Instalando dependencias..."
    npm install
fi

# Iniciar en modo desarrollo
echo "🔧 Iniciando servidor de desarrollo..."
echo "   REACT_APP_BACKEND_URL=$REACT_APP_BACKEND_URL"
echo "   REACT_APP_ARC_WS_URL=$REACT_APP_ARC_WS_URL"
npm start
