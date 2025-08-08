#!/bin/bash

# Script to start PROTEUS with network access

echo "🦠 PROTEUS Life Simulation - Network Mode"
echo "========================================="

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    LOCAL_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}' | cut -d/ -f1)
else
    echo "❌ Unsupported OS. Please set LOCAL_IP manually."
    exit 1
fi

if [ -z "$LOCAL_IP" ]; then
    echo "❌ Could not detect local IP address"
    echo "Please set it manually: export LOCAL_IP=your.ip.address"
    exit 1
fi

echo "📡 Detected local IP: $LOCAL_IP"
echo ""
echo "🌐 Services will be available at:"
echo "   Frontend: http://$LOCAL_IP:3000"
echo "   Backend:  http://$LOCAL_IP:8000"
echo "   WebSocket: ws://$LOCAL_IP:8000"
echo ""

# Create override file
cat > docker-compose.override.yml <<EOF
version: '3.8'

services:
  proteus-frontend:
    environment:
      - REACT_APP_BACKEND_URL=http://$LOCAL_IP:8000
      - REACT_APP_WS_URL=ws://$LOCAL_IP:8000
EOF

echo "📝 Created docker-compose.override.yml with network settings"
echo ""
echo "🚀 Starting services..."
echo ""

# Start services
docker compose up -d

echo ""
echo "✅ Services started!"
echo ""
echo "📱 Access from any device on your network:"
echo "   http://$LOCAL_IP:3000"
echo ""
echo "🛑 To stop: docker compose down"
echo "📊 To view logs: docker compose logs -f"