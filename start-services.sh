#!/bin/bash

echo "🚀 Iniciando servicios PROTEUS..."

# Verificar que Docker esté corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no está corriendo. Por favor, inicia Docker Desktop primero."
    exit 1
fi

# Limpiar contenedores anteriores
echo "🧹 Limpiando contenedores anteriores..."
docker compose down

# Construir imágenes si es necesario
echo "🔨 Construyendo imágenes..."
docker compose build proteus-backend proteus-frontend

# Iniciar servicios
echo "🏃 Iniciando servicios..."
docker compose up -d proteus-backend proteus-frontend

# Esperar a que los servicios estén listos
echo "⏳ Esperando a que los servicios estén listos..."
sleep 5

# Verificar estado
echo "✅ Verificando estado de los servicios..."
docker compose ps

echo ""
echo "📡 Servicios disponibles en:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - ARC WebSocket: ws://localhost:8765"
echo ""
echo "📋 Para ver los logs: docker compose logs -f"
echo "🛑 Para detener: docker compose down"