#!/usr/bin/env python3
"""
Script para actualizar imports en archivos ARC
"""

import os
import re

# Directorio de archivos ARC
arc_dir = "./arc"

# Patrones de import a actualizar
import_patterns = [
    # Imports directos
    (r'from arc_solver_python import', 'from arc.arc_solver_python import'),
    (r'from arc_visualizer import', 'from arc.arc_visualizer import'),
    (r'from arc_dataset_loader import', 'from arc.arc_dataset_loader import'),
    (r'from arc_swarm_solver import', 'from arc.arc_swarm_solver import'),
    (r'from arc_augmentation import', 'from arc.arc_augmentation import'),
    (r'from arc_official_loader import', 'from arc.arc_official_loader import'),
    (r'from proteus_arc_solver import', 'from arc.proteus_arc_solver import'),
    (r'from hybrid_proteus_solver import', 'from arc.hybrid_proteus_solver import'),
    (r'from arc_evaluation import', 'from arc.arc_evaluation import'),
    
    # Imports con paréntesis
    (r'import arc_solver_python', 'import arc.arc_solver_python'),
    (r'import arc_visualizer', 'import arc.arc_visualizer'),
    (r'import arc_dataset_loader', 'import arc.arc_dataset_loader'),
    (r'import arc_swarm_solver', 'import arc.arc_swarm_solver'),
]

def update_imports_in_file(filepath):
    """Actualiza imports en un archivo"""
    
    # Leer archivo
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar reemplazos
    modified = False
    for pattern, replacement in import_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            modified = True
            content = new_content
    
    # Escribir si hubo cambios
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated: {filepath}")
    else:
        print(f"⏭️  No changes: {filepath}")

def main():
    """Actualiza todos los archivos Python en el directorio arc"""
    
    print("Actualizando imports en archivos ARC...")
    print("=" * 60)
    
    # Procesar todos los archivos .py
    for filename in os.listdir(arc_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(arc_dir, filename)
            update_imports_in_file(filepath)
    
    print("=" * 60)
    print("✅ Actualización completa!")

if __name__ == "__main__":
    main()