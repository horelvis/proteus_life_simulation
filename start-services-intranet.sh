#!/bin/bash

echo "🌐 Configurando PROTEUS para acceso desde intranet..."

# Verificar que Docker esté corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no está corriendo. Por favor, inicia Docker Desktop primero."
    exit 1
fi

# Obtener IP del servidor
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SERVER_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
else
    # Linux
    SERVER_IP=$(hostname -I | awk '{print $1}')
fi

if [ -z "$SERVER_IP" ]; then
    echo "❌ No se pudo obtener la IP del servidor"
    echo "Por favor, configura SERVER_IP manualmente en el archivo .env"
    exit 1
fi

echo "📡 IP del servidor detectada: $SERVER_IP"

# Crear archivo .env si no existe
if [ ! -f .env ]; then
    echo "📝 Creando archivo .env con configuración de red..."
    cat > .env << EOF
# Configuración generada automáticamente
SERVER_IP=$SERVER_IP
BACKEND_PORT=8000
WEBSOCKET_PORT=8765
FRONTEND_PORT=3000
REACT_APP_BACKEND_URL=http://$SERVER_IP:8000
REACT_APP_WS_URL=ws://$SERVER_IP:8000
REACT_APP_ARC_WS_URL=ws://$SERVER_IP:8765
CORS_ORIGINS=["http://localhost:3000","http://$SERVER_IP:3000","http://$SERVER_IP"]
EOF
    echo "✅ Archivo .env creado"
else
    echo "📋 Usando archivo .env existente"
    # Actualizar solo SERVER_IP si cambió
    if grep -q "SERVER_IP=" .env; then
        sed -i.bak "s/SERVER_IP=.*/SERVER_IP=$SERVER_IP/" .env
        # Actualizar las URLs también
        sed -i.bak "s|REACT_APP_BACKEND_URL=.*|REACT_APP_BACKEND_URL=http://$SERVER_IP:8000|" .env
        sed -i.bak "s|REACT_APP_WS_URL=.*|REACT_APP_WS_URL=ws://$SERVER_IP:8000|" .env
        sed -i.bak "s|REACT_APP_ARC_WS_URL=.*|REACT_APP_ARC_WS_URL=ws://$SERVER_IP:8765|" .env
    fi
fi

# Mostrar configuración
echo ""
echo "📋 Configuración actual:"
echo "   - IP del servidor: $SERVER_IP"
echo "   - Backend API: http://$SERVER_IP:8000"
echo "   - WebSocket: ws://$SERVER_IP:8765"
echo "   - Frontend: http://$SERVER_IP:3000"
echo ""

# Limpiar contenedores anteriores
echo "🧹 Limpiando contenedores anteriores..."
docker compose down

# Construir imágenes
echo "🔨 Construyendo imágenes..."
docker compose build proteus-backend proteus-frontend

# Iniciar servicios
echo "🏃 Iniciando servicios..."
docker compose up -d proteus-backend proteus-frontend

# Esperar a que estén listos
echo "⏳ Esperando a que los servicios estén listos..."
sleep 10

# Verificar estado
echo "✅ Verificando estado de los servicios..."
docker compose ps

# Instrucciones finales
echo ""
echo "🎉 ¡Servicios iniciados correctamente!"
echo ""
echo "📱 Acceso desde la red local:"
echo "   - Frontend: http://$SERVER_IP:3000"
echo "   - Backend API: http://$SERVER_IP:8000"
echo "   - API Docs: http://$SERVER_IP:8000/docs"
echo ""
echo "💻 Acceso local:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo ""
echo "🔧 Comandos útiles:"
echo "   - Ver logs: docker compose logs -f"
echo "   - Detener: docker compose down"
echo "   - Reiniciar: docker compose restart"
echo ""
echo "⚠️  Asegúrate de que los puertos 3000, 8000 y 8765 estén abiertos en el firewall"