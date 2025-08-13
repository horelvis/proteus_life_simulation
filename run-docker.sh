#!/bin/bash

# PROTEUS Docker Deployment Script

echo "🧬 PROTEUS - Proto-Topological Evolution System"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi
}

# Function to display menu
show_menu() {
    echo ""
    echo "Choose deployment option:"
    echo "1) Start backend container"
    echo "2) Stop backend"
    echo "3) View backend logs"
    echo "4) Restart backend"
    echo "5) Clean up (remove containers and images)"
    echo "0) Exit"
    echo ""
}

# Check Docker
check_docker

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            echo -e "${BLUE}Starting backend container...${NC}"
            docker compose up -d backend
            echo -e "${GREEN}✓ Backend started!${NC}"
            echo "Backend API: http://localhost:8000"
            echo "ARC WebSocket: ws://localhost:8765"
            echo ""
            echo -e "${BLUE}Para iniciar el frontend, ejecuta en otra terminal:${NC}"
            echo "cd frontend && npm install && npm start"
            ;;
        2)
            echo -e "${BLUE}Stopping backend...${NC}"
            docker compose down
            echo -e "${GREEN}✓ Backend stopped!${NC}"
            ;;
        3)
            echo -e "${BLUE}Showing backend logs...${NC}"
            docker compose logs -f backend
            ;;
        4)
            echo -e "${BLUE}Restarting backend...${NC}"
            docker compose restart backend
            echo -e "${GREEN}✓ Backend restarted!${NC}"
            ;;
        5)
            echo -e "${RED}This will remove all PROTEUS containers and images${NC}"
            read -p "Are you sure? (y/N): " confirm
            if [[ $confirm == [yY] ]]; then
                docker compose down -v --rmi all
                echo -e "${GREEN}✓ Cleanup complete!${NC}"
            fi
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
done