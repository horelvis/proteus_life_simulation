# PROTEUS Docker Deployment

## Quick Start

Run the interactive deployment script:

```bash
./run-docker.sh
```

Or use docker compose directly:

```bash
# Start full stack
docker compose up -d proteus-backend proteus-frontend

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

## Services

### 1. **proteus-backend** (GPU-Accelerated Backend)
- Port: 8000
- WebSocket: ws://localhost:8000/ws/{client_id}
- Features:
  - Vispy GPU visualization (headless mode)
  - Numba JIT optimization
  - Supports 50,000 organisms at 60 FPS
  - Tri-layer inheritance system
  - Environmental pheromone fields

### 2. **proteus-frontend** (React UI)
- Port: 3000
- Nginx production server
- Features:
  - Real-time WebSocket updates
  - Mode switcher (Local/GPU Backend)
  - Interactive organism selection
  - Visual memory traces

## GPU Support (Optional)

For NVIDIA GPU acceleration:

1. Install nvidia-docker:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Uncomment GPU section in docker-compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Run with GPU:
```bash
docker compose up -d proteus-backend
```

## Development Mode

Mount source code for live reloading:

```yaml
volumes:
  - ./backend:/app:rw  # Read-write for development
```

## Environment Variables

### Backend
- `DISPLAY=:99` - Virtual display for headless OpenGL
- `PYTHONUNBUFFERED=1` - Real-time logging

### Frontend
- `REACT_APP_BACKEND_URL` - Backend API URL
- `REACT_APP_WS_URL` - WebSocket URL

## Troubleshooting

### Backend won't start
```bash
# Check logs
docker compose logs proteus-backend

# Verify OpenGL support
docker exec proteus-backend glxinfo | grep OpenGL
```

### Frontend can't connect
```bash
# Ensure backend is running
docker ps | grep proteus-backend

# Check network
docker network ls | grep proteus
```

### Performance issues
- Reduce organism count in backend
- Enable GPU support
- Increase Docker memory allocation

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│                 │ ◄─────────────────► │                 │
│  React Frontend │                     │  Vispy Backend  │
│   (Port 3000)   │                     │   (Port 8000)   │
│                 │                     │                 │
└─────────────────┘                     └─────────────────┘
         │                                       │
         │                                       │
         ▼                                       ▼
   ┌──────────┐                           ┌──────────┐
   │  Nginx   │                           │   Xvfb   │
   │  Server  │                           │ Display  │
   └──────────┘                           └──────────┘
```

## Commands Reference

```bash
# Build images
docker compose build

# Start specific service
docker compose up -d proteus-backend

# Scale services
docker compose up -d --scale proteus-backend=2

# Execute commands in container
docker exec -it proteus-backend python -c "import vispy; print(vispy.__version__)"

# Export/Import data
docker cp proteus-backend:/app/data ./backup
docker cp ./data proteus-backend:/app/data
```