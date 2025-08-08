#!/bin/bash

echo "Building PROTEUS backend without buildkit..."

# Disable buildkit completely
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Change to backend directory
cd "$(dirname "$0")"

# Try to build with minimal Dockerfile
echo "Building image..."
docker build \
  --no-cache \
  --force-rm \
  -f Dockerfile.minimal \
  -t proteus-backend:latest \
  .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Run with:"
    echo "docker run -p 8000:8000 proteus-backend:latest"
else
    echo "❌ Build failed"
    echo ""
    echo "Alternative: Use local development:"
    echo "./run_local.sh"
fi