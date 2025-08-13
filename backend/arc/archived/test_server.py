#!/usr/bin/env python3
"""
Test simple del servidor ARC
"""

import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Conectado al servidor")
            
            # Esperar mensaje de bienvenida
            welcome = await websocket.recv()
            print(f"📨 Mensaje recibido: {welcome}")
            
            # Enviar solicitud de puzzles
            await websocket.send(json.dumps({
                "type": "load_puzzles",
                "puzzle_set": "training",
                "count": 3
            }))
            print("📤 Solicitud de puzzles enviada")
            
            # Esperar respuesta
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"📨 Tipo de mensaje: {data['type']}")
                
                if data['type'] == 'puzzles_loaded':
                    print(f"✅ {data['count']} puzzles cargados")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Probando conexión con servidor ARC...")
    asyncio.run(test_connection())