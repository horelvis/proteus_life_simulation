#!/usr/bin/env python3
"""
Servidor Unificado PROTEUS - Maneja tanto simulación como ARC
"""

import asyncio
import threading
import uvicorn
from typing import Dict, Any

# Importar servidor principal FastAPI
from main import app

# Importar servidor WebSocket de ARC
from arc.arc_server import ARCWebSocketServer


def run_fastapi_server():
    """Ejecuta el servidor FastAPI en un thread separado"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_arc_websocket_server():
    """Ejecuta el servidor WebSocket de ARC en un thread separado"""
    asyncio.set_event_loop(asyncio.new_event_loop())
    server = ARCWebSocketServer()
    asyncio.get_event_loop().run_until_complete(server.start())


def main():
    """Inicia ambos servidores"""
    print("🧬 PROTEUS Unified Server Starting...")
    print("=" * 60)
    print("📡 FastAPI Server: http://localhost:8000")
    print("🔌 ARC WebSocket: ws://localhost:8765")
    print("=" * 60)
    
    # Crear threads para cada servidor
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    websocket_thread = threading.Thread(target=run_arc_websocket_server, daemon=True)
    
    # Iniciar ambos servidores
    fastapi_thread.start()
    websocket_thread.start()
    
    print("✅ Both servers running. Press Ctrl+C to stop.")
    
    try:
        # Mantener el programa principal activo
        while True:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)


if __name__ == "__main__":
    main()