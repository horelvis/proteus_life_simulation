#!/bin/bash

echo "🔧 Fixing Docker buildkit error..."

# Stop all containers
echo "Stopping all containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

# Remove buildkit cache
echo "Removing buildkit cache..."
docker builder prune -a -f 2>/dev/null || true

# Try to restart Docker daemon
echo "Restarting Docker..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'quit app "Docker"' 2>/dev/null || true
    sleep 5
    open -a Docker
    echo "Waiting for Docker to start..."
    sleep 20
else
    # Linux
    sudo systemctl restart docker 2>/dev/null || sudo service docker restart 2>/dev/null || true
    sleep 10
fi

# Clean builder cache manually if possible
echo "Cleaning builder cache manually..."
rm -rf ~/.docker/buildkit 2>/dev/null || true

# Create a new builder
echo "Creating new builder instance..."
docker buildx create --name proteus-builder --use 2>/dev/null || true

echo "✅ Docker cleanup complete!"
echo ""
echo "Now try building with:"
echo "docker compose build --no-cache"