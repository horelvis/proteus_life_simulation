#!/usr/bin/env python3
"""
Script para probar la integración con la API de ARC Prize
"""

import asyncio
import os
import sys
from arc.arc_api_client import ARCApiClient, RandomAgent

async def test_api_connection():
    """Prueba la conexión con la API de ARC"""
    
    # Verificar API key
    api_key = os.getenv('ARC_API_KEY')
    if not api_key:
        print("⚠️  No se encontró ARC_API_KEY en las variables de entorno")
        print("📝 Para obtener una API key:")
        print("   1. Visita https://arcprize.org")
        print("   2. Regístrate/Inicia sesión")
        print("   3. Ve a tu perfil -> API Keys")
        print("   4. Genera una nueva key")
        print("   5. Configura: export ARC_API_KEY='tu_key_aqui'")
        print("\n🎮 Usando puzzles demo por ahora...")
        return False
    
    print(f"✅ API Key encontrada: {api_key[:10]}...")
    
    # Crear cliente
    client = ARCApiClient(api_key)
    
    try:
        async with client:
            # Probar obtener info del juego
            print("\n📊 Obteniendo información del juego ls20...")
            game_info = await client.get_game_info('ls20')
            
            if game_info:
                print(f"✅ Juego: {game_info.get('name', 'ls20')}")
                print(f"   Puzzles: {game_info.get('puzzle_count', 'unknown')}")
                print(f"   Dificultad: {game_info.get('difficulty', 'unknown')}")
                
                # Cargar un puzzle de ejemplo
                print("\n🧩 Cargando puzzle de ejemplo...")
                puzzle = await client.get_puzzle('ls20', 0)
                
                if puzzle:
                    print(f"✅ Puzzle cargado:")
                    print(f"   ID: {puzzle.get('id', 'unknown')}")
                    print(f"   Ejemplos de entrenamiento: {len(puzzle.get('train', []))}")
                    print(f"   Casos de prueba: {len(puzzle.get('test', []))}")
                    
                    # Mostrar primer ejemplo
                    if puzzle.get('train'):
                        example = puzzle['train'][0]
                        print(f"\n📐 Primer ejemplo:")
                        print(f"   Input size: {len(example['input'])}x{len(example['input'][0])}")
                        print(f"   Output size: {len(example['output'])}x{len(example['output'][0])}")
                    
                    return True
                else:
                    print("❌ No se pudo cargar el puzzle")
                    return False
            else:
                print("❌ No se pudo obtener información del juego")
                return False
                
    except Exception as e:
        print(f"❌ Error conectando con la API: {e}")
        return False


async def test_random_agent():
    """Prueba el agente aleatorio"""
    
    api_key = os.getenv('ARC_API_KEY')
    if not api_key:
        print("⚠️  Saltando prueba del agente (no hay API key)")
        return
    
    print("\n🤖 Probando agente aleatorio...")
    
    client = ARCApiClient(api_key)
    agent = RandomAgent(client)
    
    try:
        async with client:
            # Cargar un puzzle
            puzzle = await client.get_puzzle('ls20', 0)
            if puzzle:
                parsed = client.parse_arc_puzzle(puzzle)
                
                # Generar solución aleatoria
                solution = await agent.solve_puzzle(parsed)
                
                print(f"✅ Solución generada:")
                print(f"   Tamaño: {len(solution)}x{len(solution[0])}")
                
                # Enviar solución (solo si queremos probar)
                # result = await client.submit_solution('ls20', 0, solution)
                # print(f"   Resultado: {result}")
                
    except Exception as e:
        print(f"❌ Error en agente: {e}")


async def main():
    """Función principal"""
    print("🧪 Probando integración con ARC Prize API\n")
    print("=" * 50)
    
    # Probar conexión
    api_ok = await test_api_connection()
    
    if api_ok:
        # Probar agente
        await test_random_agent()
    
    print("\n" + "=" * 50)
    print("✨ Prueba completada")


if __name__ == "__main__":
    # Cargar variables de entorno si existe .env
    from dotenv import load_dotenv
    
    # Intentar cargar desde varios lugares
    if os.path.exists('.env'):
        load_dotenv('.env')
    elif os.path.exists('../.env'):
        load_dotenv('../.env')
    elif os.path.exists('/app/.env'):
        load_dotenv('/app/.env')
    
    # Depuración
    api_key = os.getenv('ARC_API_KEY')
    if api_key:
        print(f"🔑 API Key cargada: {api_key[:10]}...")
    
    asyncio.run(main())