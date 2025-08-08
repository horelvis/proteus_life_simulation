#!/bin/bash

echo "🔨 Building PROTEUS with simplified Docker setup..."

# Disable buildkit
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Build backend
echo "Building backend..."
cd backend
docker build -f Dockerfile.simple -t proteus-backend:latest . --no-cache
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Backend built successfully${NC}"
else
    echo -e "${RED}✗ Backend build failed${NC}"
    exit 1
fi
cd ..

# Run with simple compose
echo "Starting services..."
docker compose -f docker-compose.simple.yml up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Services started successfully${NC}"
    echo ""
    echo "Access the application at:"
    echo "Frontend: http://localhost:3000"
    echo "Backend: http://localhost:8000"
    echo ""
    echo "View logs with:"
    echo "docker compose -f docker-compose.simple.yml logs -f"
else
    echo -e "${RED}✗ Failed to start services${NC}"
    exit 1
fi