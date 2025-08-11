#!/bin/bash

# Script para ejecutar el frontend en modo desarrollo
# Esto permite cambios en tiempo real sin necesidad de rebuild

echo "🚀 Iniciando frontend en modo desarrollo..."
echo "================================================"
echo "Frontend: http://localhost:3001"
echo "Backend API: http://localhost:8000"
echo "WebSocket ARC: ws://localhost:8765"
echo "================================================"

# Configurar variables de entorno
export REACT_APP_BACKEND_URL=http://localhost:8000
export REACT_APP_WS_URL=ws://localhost:8000
export REACT_APP_ARC_WS_URL=ws://localhost:8765
export PORT=3001

# Ir al directorio del frontend
cd frontend

# Instalar dependencias si no existen
if [ ! -d "node_modules" ]; then
    echo "📦 Instalando dependencias..."
    npm install
fi

# Iniciar en modo desarrollo
echo "🔧 Iniciando servidor de desarrollo..."
npm start