#!/usr/bin/env python3
"""Test de carga de puzzles oficiales de ARC"""

import asyncio
import websockets
import json

async def test_load_puzzles():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Esperar mensaje de conexión
        msg = await websocket.recv()
        print(f"Conectado: {msg}")
        
        # Solicitar puzzles oficiales
        request = {
            "type": "load_puzzles",
            "puzzle_set": "training",
            "count": 3,
            "use_official": True
        }
        
        await websocket.send(json.dumps(request))
        print("Solicitando puzzles oficiales...")
        
        # Esperar respuestas
        for _ in range(3):  # Esperar hasta 3 mensajes
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"\nRespuesta tipo: {data.get('type')}")
                
                if data['type'] == 'puzzles_loaded':
                    print(f"Puzzles cargados: {data.get('count')}")
                    print(f"Oficiales: {data.get('official')}")
                    if 'puzzles' in data and len(data['puzzles']) > 0:
                        print(f"Primer puzzle ID: {data['puzzles'][0].get('id')}")
                        print(f"Estructura: train={len(data['puzzles'][0].get('train', []))}, test={len(data['puzzles'][0].get('test', []))}")
                    break
                elif data['type'] == 'loading':
                    print(f"Mensaje: {data.get('message')}")
                elif data['type'] == 'error':
                    print(f"ERROR: {data.get('message')}")
                    break
                    
            except asyncio.TimeoutError:
                print("Timeout esperando respuesta")
                break

if __name__ == "__main__":
    asyncio.run(test_load_puzzles())