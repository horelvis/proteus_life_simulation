# Backend Python para PROTEUS-AC ARC Solver

Sistema de resolución de puzzles ARC con transparencia total, implementado en Python con comunicación WebSocket.

## 🚀 Instalación

1. **Crear entorno virtual**:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

## 🏃 Ejecución

1. **Iniciar el servidor Python**:
```bash
# Desde la carpeta backend_arc
./start_server.sh
# O directamente:
python arc_server.py
```

2. **En el frontend React**:
- El frontend se conectará automáticamente al servidor WebSocket en `ws://localhost:8765`
- Click en "🌐 Usar Backend Python" para activar el modo Python

## 🔧 Arquitectura

### Servidor WebSocket (`arc_server.py`)
- Maneja conexiones múltiples de clientes
- Procesa mensajes de forma asíncrona
- Envía actualizaciones en tiempo real del progreso

### Solver (`arc_solver_python.py`)
- Implementa 10 tipos de transformaciones ARC:
  - Color mapping
  - Pattern replication (3x3)
  - Reflection (horizontal/vertical)
  - Rotation (90°, 180°, 270°)
  - Gravity (elementos caen)
  - Counting (contar elementos)
  - Fill shape (rellenar formas cerradas)
  - Symmetry detection
  - Pattern extraction
  - Line drawing

### Visualizador (`arc_visualizer.py`)
- Genera imágenes PNG de cada paso
- Crea diagramas de flujo del razonamiento
- Exporta GIFs animados del proceso

### Dataset Loader (`arc_dataset_loader.py`)
- Carga puzzles del dataset ARC
- Valida formato de puzzles
- Organiza por categoría y dificultad

## 📨 Protocolo WebSocket

### Mensajes del Cliente al Servidor:

1. **Cargar puzzles**:
```json
{
  "type": "load_puzzles",
  "puzzle_set": "training",
  "count": 10
}
```

2. **Resolver puzzle**:
```json
{
  "type": "solve_puzzle",
  "puzzle_id": "00d62c1b",
  "puzzle": { ... }
}
```

3. **Verificar integridad**:
```json
{
  "type": "verify_integrity"
}
```

### Mensajes del Servidor al Cliente:

1. **Progreso de análisis**:
```json
{
  "type": "analyzing_example",
  "puzzle_id": "00d62c1b",
  "example_index": 0,
  "input": [[...]],
  "output": [[...]]
}
```

2. **Regla detectada**:
```json
{
  "type": "rule_detected",
  "puzzle_id": "00d62c1b",
  "rule": {
    "type": "color_mapping",
    "confidence": 0.95,
    "parameters": { ... }
  }
}
```

3. **Resultado final**:
```json
{
  "type": "solving_complete",
  "puzzle_id": "00d62c1b",
  "solution": [[...]],
  "expected": [[...]],
  "is_correct": true,
  "confidence": 0.95,
  "reasoning_steps": [ ... ]
}
```

## 🔍 Transparencia

Cada paso del proceso es visible:

1. **Análisis de ejemplos**: Muestra qué patrones busca
2. **Detección de reglas**: Explica qué transformación detectó
3. **Aplicación paso a paso**: Visualiza cada cambio
4. **Verificación**: Compara con la solución esperada

## 🧪 Tests de Integridad

El sistema incluye tests automáticos para verificar que no hay datos hardcodeados:

```python
# Test con colores diferentes
test_color_mapping(3->8)  # Diferente al dataset (1->4)

# Test con números diferentes  
test_counting([7,7,7,7,7])  # Cuenta 5 sietes

# Test con patrones nuevos
test_reflection([[1,2,3],[4,5,6]])
```

## 📊 Métricas

El sistema reporta:
- Porcentaje de puzzles resueltos
- Confianza por tipo de transformación
- Tiempo de procesamiento
- Pasos de razonamiento detallados

## 🐛 Debugging

Para más logs detallados:
```python
# En arc_server.py
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Próximas Mejoras

- [ ] Soporte para puzzles más grandes (30x30)
- [ ] Detección de patrones recursivos
- [ ] Aprendizaje de reglas compuestas
- [ ] Paralelización de análisis
- [ ] Caché de resultados