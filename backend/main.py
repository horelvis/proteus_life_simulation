"""
PROTEUS Backend - FastAPI Server
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from typing import Dict, List, Optional
import uuid
from datetime import datetime

from app.simulation import SimulationEngine
from app.models import (
    SimulationConfig, 
    SimulationState,
    OrganismData,
    WorldUpdate
)
from typing import Any


# Gestión de simulaciones activas
active_simulations: Dict[str, SimulationEngine] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🧬 PROTEUS Backend starting...")
    yield
    # Shutdown
    print("Shutting down PROTEUS Backend...")
    # Limpiar simulaciones activas
    for sim_id in list(active_simulations.keys()):
        active_simulations[sim_id].stop()
        del active_simulations[sim_id]


app = FastAPI(
    title="PROTEUS API",
    description="Proto-Topological Evolution System - Backend API",
    version="0.2.0",
    lifespan=lifespan
)

# CORS para permitir conexiones del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "PROTEUS API",
        "version": "0.2.0",
        "status": "running",
        "active_simulations": len(active_simulations)
    }


@app.get("/api/arc/puzzles")
async def get_arc_puzzles() -> Dict[str, Any]:
    """Obtiene la lista de puzzles ARC disponibles"""
    # Puzzles estilo ARC real
    puzzles = [
        {"id": "demo_1", "name": "Reflexión Vertical", "difficulty": "easy", "category": "symmetry"},
        {"id": "demo_2", "name": "Contar Objetos", "difficulty": "easy", "category": "counting"},
        {"id": "demo_3", "name": "Encontrar Patrón", "difficulty": "medium", "category": "pattern"},
        {"id": "demo_4", "name": "Gravedad", "difficulty": "medium", "category": "physics"},
        {"id": "arc_001", "name": "Rotación 90°", "difficulty": "medium", "category": "rotation"},
        {"id": "arc_002", "name": "Completar Secuencia", "difficulty": "hard", "category": "sequence"}
    ]
    
    return {"puzzles": puzzles}


@app.get("/api/arc/puzzle/{puzzle_id}")
async def get_arc_puzzle(puzzle_id: str) -> Dict[str, Any]:
    """Obtiene un puzzle ARC específico"""
    
    # Puzzles demo hardcodeados - Estilo ARC real
    demo_puzzles = {
        "demo_1": {
            "id": "demo_1",
            "name": "Reflexión Vertical",
            "category": "symmetry",
            "difficulty": "easy",
            "train": [
                {
                    "input": [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                    "output": [[1, 0, 1], [2, 0, 2], [3, 0, 3]]
                },
                {
                    "input": [[0, 0, 4], [0, 0, 5], [0, 0, 6]],
                    "output": [[4, 0, 4], [5, 0, 5], [6, 0, 6]]
                },
                {
                    "input": [[7, 8, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[7, 8, 0], [0, 0, 0], [7, 8, 0]]
                }
            ],
            "test": [
                {"input": [[0, 2, 0], [0, 3, 0], [0, 0, 0]]}
            ]
        },
        "demo_2": {
            "id": "demo_2",
            "name": "Contar Objetos",
            "category": "counting",
            "difficulty": "easy",
            "train": [
                {
                    "input": [[1, 0, 1], [0, 0, 0], [1, 0, 0]],
                    "output": [[3]]
                },
                {
                    "input": [[2, 2, 0], [0, 2, 0], [2, 2, 0]],
                    "output": [[5]]
                },
                {
                    "input": [[0, 3, 3], [0, 0, 0], [0, 0, 0]],
                    "output": [[2]]
                }
            ],
            "test": [
                {"input": [[4, 0, 4], [4, 0, 0], [0, 0, 4]]}
            ]
        },
        "demo_3": {
            "id": "demo_3",
            "name": "Encontrar Patrón",
            "category": "pattern",
            "difficulty": "medium",
            "train": [
                {
                    "input": [[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 1, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]],
                    "output": [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
                },
                {
                    "input": [[0, 0, 0, 0, 0, 0], [0, 3, 4, 0, 0, 0], [0, 4, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    "output": [[3, 4], [4, 3]]
                }
            ],
            "test": [
                {"input": [[0, 0, 0, 0, 0, 0, 0], [0, 0, 5, 6, 5, 0, 0], [0, 0, 6, 5, 6, 0, 0], [0, 0, 5, 6, 5, 0, 0], [0, 0, 0, 0, 0, 0, 0]]}
            ]
        },
        "demo_4": {
            "id": "demo_4",
            "name": "Gravedad",
            "category": "physics",
            "difficulty": "medium",
            "train": [
                {
                    "input": [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]]
                },
                {
                    "input": [[2, 0, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 3]]
                },
                {
                    "input": [[0, 0, 0], [4, 0, 5], [0, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 0, 5]]
                }
            ],
            "test": [
                {"input": [[6, 0, 0], [0, 7, 0], [0, 0, 0], [0, 0, 0]]}
            ]
        },
        "arc_001": {
            "id": "arc_001",
            "name": "Rotación 90°",
            "category": "rotation",
            "difficulty": "medium",
            "train": [
                {
                    "input": [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 1], [0, 0, 2], [0, 0, 0]]
                },
                {
                    "input": [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
                    "output": [[5, 4, 3], [0, 0, 0], [0, 0, 0]]
                },
                {
                    "input": [[0, 0, 6], [0, 0, 7], [0, 0, 0]],
                    "output": [[0, 0, 0], [7, 0, 0], [6, 0, 0]]
                }
            ],
            "test": [
                {"input": [[8, 9, 0], [0, 0, 0], [0, 0, 0]]}
            ]
        },
        "arc_002": {
            "id": "arc_002",
            "name": "Completar Secuencia",
            "category": "sequence",
            "difficulty": "hard",
            "train": [
                {
                    "input": [[1, 0, 3], [0, 0, 0], [0, 0, 0]],
                    "output": [[1, 2, 3], [0, 0, 0], [0, 0, 0]]
                },
                {
                    "input": [[0, 0, 0], [4, 0, 6], [0, 0, 0]],
                    "output": [[0, 0, 0], [4, 5, 6], [0, 0, 0]]
                },
                {
                    "input": [[0, 0, 0], [0, 0, 0], [7, 0, 9]],
                    "output": [[0, 0, 0], [0, 0, 0], [7, 8, 9]]
                }
            ],
            "test": [
                {"input": [[2, 0, 0, 5], [0, 0, 0, 0], [0, 0, 0, 0]]}
            ]
        }
    }
    
    if puzzle_id in demo_puzzles:
        return demo_puzzles[puzzle_id]
    
    # Para otros puzzles, devolver un error o cargar de archivo
    return {"error": "Puzzle not found", "id": puzzle_id}


@app.post("/api/simulations/create")
async def create_simulation(config: SimulationConfig) -> Dict:
    """Crea una nueva simulación y detiene todas las anteriores"""
    
    # Detener y eliminar todas las simulaciones activas
    for old_sim_id in list(active_simulations.keys()):
        try:
            old_engine = active_simulations[old_sim_id]
            old_engine.stop()
            
            # Cerrar conexiones WebSocket de simulaciones anteriores
            if old_sim_id in websocket_connections:
                for ws in websocket_connections[old_sim_id]:
                    try:
                        await ws.close()
                    except:
                        pass
                del websocket_connections[old_sim_id]
            
            del active_simulations[old_sim_id]
        except Exception as e:
            print(f"Error stopping simulation {old_sim_id}: {e}")
    
    # Crear nueva simulación
    sim_id = str(uuid.uuid4())
    
    # Crear motor de simulación
    engine = SimulationEngine(sim_id, config)
    active_simulations[sim_id] = engine
    websocket_connections[sim_id] = []
    
    return {
        "simulation_id": sim_id,
        "status": "created",
        "config": config.dict(),
        "created_at": datetime.now().isoformat()
    }


@app.post("/api/simulations/{sim_id}/start")
async def start_simulation(sim_id: str) -> Dict:
    """Inicia una simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    engine.start()
    
    # Iniciar loop de actualización para WebSockets
    asyncio.create_task(simulation_update_loop(sim_id))
    
    return {
        "simulation_id": sim_id,
        "status": "running",
        "message": "Simulation started"
    }


@app.post("/api/simulations/{sim_id}/pause")
async def pause_simulation(sim_id: str) -> Dict:
    """Pausa una simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    engine.pause()
    
    return {
        "simulation_id": sim_id,
        "status": "paused",
        "message": "Simulation paused"
    }


@app.post("/api/simulations/{sim_id}/stop")
async def stop_simulation(sim_id: str) -> Dict:
    """Detiene y elimina una simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    engine.stop()
    
    # Cerrar conexiones WebSocket
    if sim_id in websocket_connections:
        for ws in websocket_connections[sim_id]:
            await ws.close()
        del websocket_connections[sim_id]
    
    del active_simulations[sim_id]
    
    return {
        "simulation_id": sim_id,
        "status": "stopped",
        "message": "Simulation stopped and removed"
    }


@app.get("/api/simulations/{sim_id}/state")
async def get_simulation_state(sim_id: str) -> SimulationState:
    """Obtiene el estado actual de una simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    return engine.get_state()


@app.get("/api/simulations/{sim_id}/statistics")
async def get_simulation_statistics(sim_id: str) -> Dict:
    """Obtiene estadísticas detalladas de la simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    return engine.get_statistics()


@app.get("/api/simulations/{sim_id}/organisms")
async def get_organisms(sim_id: str) -> List[OrganismData]:
    """Obtiene la lista de organismos vivos"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    engine = active_simulations[sim_id]
    return engine.get_organisms()


@app.websocket("/ws/{sim_id}")
async def websocket_endpoint(websocket: WebSocket, sim_id: str):
    """WebSocket para actualizaciones en tiempo real"""
    await websocket.accept()
    
    # Verificar si la simulación existe
    if sim_id not in active_simulations:
        await websocket.send_text(json.dumps({"error": "Simulation not found"}))
        await websocket.close(code=1008, reason="Simulation not found")
        return
    
    # Inicializar lista de conexiones si no existe
    if sim_id not in websocket_connections:
        websocket_connections[sim_id] = []
    
    websocket_connections[sim_id].append(websocket)
    
    try:
        # Enviar estado inicial
        if sim_id in active_simulations:
            engine = active_simulations[sim_id]
            initial_state = engine.get_state()
            await websocket.send_text(json.dumps({
                "type": "initial_state",
                "data": initial_state.dict()
            }))
            
            # Si la simulación está creada pero no iniciada, iniciar el loop de actualización
            if engine.status == "created":
                asyncio.create_task(simulation_update_loop(sim_id))
        
        # Mantener conexión abierta
        while True:
            data = await websocket.receive_text()
            # Procesar comandos del cliente si es necesario
            
    except WebSocketDisconnect:
        if sim_id in websocket_connections and websocket in websocket_connections[sim_id]:
            websocket_connections[sim_id].remove(websocket)


async def simulation_update_loop(sim_id: str):
    """Loop que envía actualizaciones a todos los clientes conectados"""
    while sim_id in active_simulations:
        engine = active_simulations[sim_id]
        
        if engine.is_running:
            # Obtener actualización
            update = engine.get_update()
            
            # Enviar a todos los clientes conectados
            dead_connections = []
            for ws in websocket_connections.get(sim_id, []):
                try:
                    await ws.send_text(json.dumps({
                        "type": "update",
                        "data": update.dict()
                    }))
                except:
                    dead_connections.append(ws)
            
            # Limpiar conexiones muertas
            for ws in dead_connections:
                if ws in websocket_connections[sim_id]:
                    websocket_connections[sim_id].remove(ws)
        
        # Esperar antes de la siguiente actualización
        await asyncio.sleep(0.05)  # 20 FPS


@app.post("/api/simulations/{sim_id}/spawn-organism")
async def spawn_organism(sim_id: str, request: Dict):
    """Añade un nuevo organismo a la simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    x = request.get("x", 400)
    y = request.get("y", 300)
    organism_type = request.get("organism_type", "protozoa")
    
    engine = active_simulations[sim_id]
    organism_id = engine.spawn_organism(x, y, organism_type)
    
    return {
        "organism_id": organism_id,
        "type": organism_type,
        "position": {"x": x, "y": y}
    }


@app.post("/api/simulations/{sim_id}/add-predator")
async def add_predator(sim_id: str, request: Dict):
    """Añade un depredador a la simulación"""
    if sim_id not in active_simulations:
        return {"error": "Simulation not found"}, 404
    
    x = request.get("x", 400)
    y = request.get("y", 300)
    
    engine = active_simulations[sim_id]
    predator_id = engine.add_predator(x, y)
    
    return {
        "predator_id": predator_id,
        "position": {"x": x, "y": y}
    }


@app.get("/api/simulations")
async def list_simulations() -> List[Dict]:
    """Lista todas las simulaciones activas"""
    return [
        {
            "id": sim_id,
            "status": engine.status,
            "time": engine.time,
            "organisms": len(engine.organisms),
            "generation": engine.current_generation
        }
        for sim_id, engine in active_simulations.items()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)