#!/usr/bin/env python3
"""
Script simple para iniciar el servidor ARC
"""

import subprocess
import sys
import os

def main():
    print("🚀 Iniciando servidor ARC WebSocket...")
    
    # Verificar si las dependencias están instaladas
    try:
        import websockets
        import numpy
        print("✅ Dependencias verificadas")
    except ImportError:
        print("⚠️  Instalando dependencias...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Iniciar el servidor
    print("🔌 Iniciando servidor en ws://localhost:8765")
    os.system(f"{sys.executable} arc_server.py")

if __name__ == "__main__":
    main()