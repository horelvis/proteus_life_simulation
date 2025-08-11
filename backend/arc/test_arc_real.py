#!/usr/bin/env python3
"""
Prueba de conexión real con ARC Prize
Basado en el código oficial de ARC-AGI-3-Agents
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Cargar configuración
load_dotenv('.env')
load_dotenv('/app/.env')

# Configuración del servidor
SCHEME = os.environ.get("SCHEME", "https")
HOST = os.environ.get("HOST", "three.arcprize.org")
PORT = os.environ.get("PORT", "443")

# Construir URL
if (SCHEME == "http" and str(PORT) == "80") or (SCHEME == "https" and str(PORT) == "443"):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"

# Headers con API key
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}

def test_connection():
    """Prueba la conexión con el servidor de ARC Prize"""
    
    print(f"🔌 Conectando a: {ROOT_URL}")
    print(f"🔑 API Key: {HEADERS['X-API-Key'][:10]}..." if HEADERS['X-API-Key'] else "❌ No API Key")
    print("-" * 50)
    
    # Intentar obtener lista de juegos
    try:
        print(f"\n📋 Obteniendo lista de juegos desde: {ROOT_URL}/api/games")
        
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)
            
            print(f"   Status: {r.status_code}")
            
            if r.status_code == 200:
                try:
                    games = r.json()
                    print(f"✅ Juegos disponibles: {len(games)}")
                    
                    # Mostrar primeros juegos
                    for i, game in enumerate(games[:5]):
                        if isinstance(game, dict):
                            game_id = game.get('game_id', game)
                            print(f"   {i+1}. {game_id}")
                        else:
                            print(f"   {i+1}. {game}")
                    
                    if len(games) > 5:
                        print(f"   ... y {len(games)-5} más")
                        
                    return games
                    
                except (ValueError, KeyError) as e:
                    print(f"❌ Error parseando respuesta: {e}")
                    print(f"   Respuesta: {r.text[:200]}")
                    
            elif r.status_code == 401:
                print("❌ Error 401: API Key inválida o no autorizada")
                print("   Verifica tu API key en https://three.arcprize.org")
                
            elif r.status_code == 404:
                print("❌ Error 404: Endpoint no encontrado")
                print(f"   URL: {ROOT_URL}/api/games")
                
            else:
                print(f"❌ Error {r.status_code}: {r.text[:200]}")
                
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Error de conexión: No se pudo conectar a {ROOT_URL}")
        print(f"   Detalles: {e}")
        
    except requests.exceptions.Timeout:
        print(f"❌ Timeout: El servidor no respondió en 10 segundos")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        
    return None

def test_game_state(game_id="ls20"):
    """Prueba obtener el estado de un juego específico"""
    
    print(f"\n🎮 Probando juego: {game_id}")
    print("-" * 50)
    
    try:
        # Primero hacer RESET para iniciar el juego
        print(f"🔄 Iniciando juego con RESET...")
        
        with requests.Session() as session:
            session.headers.update(HEADERS)
            
            # Enviar acción RESET
            reset_data = {
                "card_id": "test_card",
                "game_id": game_id
            }
            
            r = session.post(
                f"{ROOT_URL}/api/cmd/RESET",
                json=reset_data,
                timeout=10
            )
            
            print(f"   Status: {r.status_code}")
            
            if r.status_code == 200:
                try:
                    frame = r.json()
                    print(f"✅ Juego iniciado!")
                    
                    # Mostrar información del frame
                    if 'state' in frame:
                        print(f"   Estado: {frame['state']}")
                    if 'score' in frame:
                        print(f"   Puntuación: {frame['score']}")
                    if 'guid' in frame:
                        print(f"   GUID: {frame['guid'][:20]}...")
                        
                    return frame
                    
                except Exception as e:
                    print(f"❌ Error parseando respuesta: {e}")
                    print(f"   Respuesta: {r.text[:500]}")
                    
            else:
                print(f"❌ Error {r.status_code}")
                error_msg = r.text[:500]
                print(f"   Mensaje: {error_msg}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        
    return None

def main():
    """Función principal de prueba"""
    
    print("🧪 Prueba de conexión con ARC Prize API")
    print("=" * 50)
    
    # Probar conexión básica
    games = test_connection()
    
    if games and len(games) > 0:
        # Si hay juegos, probar el primero
        if isinstance(games[0], dict):
            game_id = games[0].get('game_id', 'ls20')
        else:
            game_id = games[0] if isinstance(games[0], str) else 'ls20'
            
        test_game_state(game_id)
    
    print("\n" + "=" * 50)
    print("✨ Prueba completada")

if __name__ == "__main__":
    main()