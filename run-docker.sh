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
    echo "1) Full stack (Frontend + Backend)"
    echo "2) Backend only (GPU/Vispy)"
    echo "3) Frontend only"
    echo "4) Stop all services"
    echo "5) View logs"
    echo "6) Clean up (remove containers and images)"
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
            echo -e "${BLUE}Starting full stack...${NC}"
            docker compose up -d proteus-backend proteus-frontend
            echo -e "${GREEN}✓ Full stack started!${NC}"
            echo "Frontend: http://localhost:3000"
            echo "Backend: http://localhost:8000"
            echo "WebSocket: ws://localhost:8000/ws/{client_id}"
            ;;
        2)
            echo -e "${BLUE}Starting backend only...${NC}"
            docker compose up -d proteus-backend
            echo -e "${GREEN}✓ Backend started!${NC}"
            echo "Backend: http://localhost:8000"
            ;;
        3)
            echo -e "${BLUE}Starting frontend only...${NC}"
            docker compose up -d proteus-frontend
            echo -e "${GREEN}✓ Frontend started!${NC}"
            echo "Frontend: http://localhost:3000"
            ;;
        4)
            echo -e "${BLUE}Stopping all services...${NC}"
            docker compose down
            echo -e "${GREEN}✓ All services stopped!${NC}"
            ;;
        5)
            echo "View logs for which service?"
            echo "1) Backend"
            echo "2) Frontend"
            echo "3) All"
            read -p "Enter choice: " log_choice
            
            case $log_choice in
                1) docker compose logs -f proteus-backend ;;
                2) docker compose logs -f proteus-frontend ;;
                3) docker compose logs -f ;;
                *) echo -e "${RED}Invalid choice${NC}" ;;
            esac
            ;;
        6)
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