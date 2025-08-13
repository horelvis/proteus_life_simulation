# 🐳 Docker Setup - PROTEUS ARC

## Configuración Simplificada

### Backend (Docker)
El backend se ejecuta en un único contenedor Docker con todos los servicios integrados:
- FastAPI API en puerto 8000
- WebSocket ARC en puerto 8765
- Soporte GPU opcional (si está disponible)

### Frontend (Local)
El frontend se ejecuta localmente con hot reload para desarrollo:
```bash
cd frontend
npm install
npm start
```

## Comandos Rápidos

### Iniciar Backend
```bash
docker compose up -d backend
```

### Ver logs
```bash
docker compose logs -f backend
```

### Detener
```bash
docker compose down
```

### Rebuild completo
```bash
docker compose build --no-cache backend
docker compose up -d backend
```

## Script de Gestión
```bash
./run-docker.sh
```

Menú interactivo para:
1. Iniciar backend
2. Detener backend
3. Ver logs
4. Reiniciar
5. Limpiar todo

## Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:
```env
BACKEND_PORT=8000
WEBSOCKET_PORT=8765
ARC_API_KEY=tu_api_key_aqui
```

## URLs de Acceso

- Backend API: http://localhost:8000
- ARC WebSocket: ws://localhost:8765
- Frontend (local): http://localhost:3000

## Notas

- El frontend NO se ejecuta en Docker para facilitar el desarrollo
- El backend monta los volúmenes para hot reload
- GPU support automático si nvidia-docker está instalado