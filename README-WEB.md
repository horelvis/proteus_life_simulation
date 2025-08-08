# PROTEUS Web Application

Versión web interactiva del sistema de evolución proto-topológica PROTEUS, con visualización en tiempo real y control desde el navegador.

## Arquitectura

- **Backend**: FastAPI con WebSocket para actualizaciones en tiempo real
- **Frontend**: React con Canvas para visualización
- **Docker**: Contenedores para fácil despliegue

## Inicio Rápido

### Con Docker (Recomendado)

1. Construir las imágenes:
```bash
make web-build
```

2. Ejecutar la aplicación:
```bash
make web-run
```

3. Abrir en el navegador:
- Frontend: http://localhost:3000
- API Backend: http://localhost:8000

### Desarrollo Local

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

## Características

### Visualización Interactiva
- Canvas 2D con renderizado en tiempo real
- Visualización de organismos con colores según fenotipo
- Trayectorias de movimiento
- Depredadores luminosos con efectos visuales

### Controles de Simulación
- Iniciar/Pausar/Detener simulación
- Control de velocidad
- Spawning manual de organismos (Shift+Click)
- Añadir depredadores (Alt+Click)

### Panel de Estadísticas
- Población en tiempo real
- Distribución de fenotipos
- Evolución de órganos
- Métricas de supervivencia

### Información de Organismos
- Click en organismo para ver detalles
- Órganos desarrollados
- Capacidades emergentes
- Historia evolutiva

## API Endpoints

### Simulación
- `POST /api/simulations/create` - Crear nueva simulación
- `POST /api/simulations/{id}/start` - Iniciar simulación
- `POST /api/simulations/{id}/pause` - Pausar simulación
- `POST /api/simulations/{id}/stop` - Detener simulación

### Estado
- `GET /api/simulations/{id}/state` - Estado completo
- `GET /api/simulations/{id}/statistics` - Estadísticas
- `GET /api/simulations/{id}/organisms` - Lista de organismos

### WebSocket
- `ws://localhost:8000/ws/{simulation_id}` - Actualizaciones en tiempo real

## Configuración

### Backend (FastAPI)
```python
SimulationConfig:
  world_size: [800, 600]
  initial_organisms: 50
  initial_predators: 5
  mutation_rate: 0.1
  enable_organs: true
```

### Frontend (React)
- Actualización a 20 FPS
- Canvas responsivo
- Tema oscuro por defecto

## Desarrollo

### Estructura del Proyecto
```
backend/
├── main.py              # API principal
├── app/
│   ├── simulation.py    # Motor de simulación
│   ├── organisms.py     # Lógica de organismos
│   ├── field_manager.py # Gestión de campos
│   └── models.py        # Modelos Pydantic

frontend/
├── src/
│   ├── App.js              # Componente principal
│   ├── components/         # Componentes React
│   │   ├── SimulationCanvas.js
│   │   ├── ControlPanel.js
│   │   ├── StatsPanel.js
│   │   └── OrganismInfo.js
│   └── hooks/             # Custom hooks
│       ├── useSimulation.js
│       └── useWebSocket.js
```

### Añadir Nuevas Características

#### Nuevo Órgano
1. Añadir a `OrganType` en `backend/app/models.py`
2. Implementar lógica en `WebOrganism._update_capabilities()`
3. Añadir color en `frontend/src/components/SimulationCanvas.js`

#### Nueva Estadística
1. Calcular en `SimulationEngine.get_statistics()`
2. Mostrar en `frontend/src/components/StatsPanel.js`

## Comandos Útiles

```bash
# Ver logs
make web-logs

# Detener aplicación
make web-stop

# Limpiar todo
docker compose -f docker-compose.web.yml down -v
```

## Troubleshooting

### El frontend no se conecta al backend
- Verificar que ambos contenedores estén corriendo
- Revisar CORS en `backend/main.py`
- Verificar configuración de proxy en `nginx.conf`

### WebSocket no funciona
- Verificar que el simulationId sea válido
- Revisar la consola del navegador
- Verificar logs del backend

### Rendimiento lento
- Reducir número de organismos iniciales
- Ajustar `field_resolution` en FieldManager
- Reducir FPS en el frontend

## Demo

Para una demo rápida:
1. `make web-run`
2. Click en "Start" 
3. Shift+Click para añadir organismos
4. Alt+Click para añadir depredadores
5. Observar la evolución en tiempo real