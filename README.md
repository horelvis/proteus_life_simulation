# 🧬 PROTEUS - Ejecución con Docker

## Inicio Rápido

### 1. Construir y ejecutar la demo visual:

```bash
make demo
```

Esto generará una secuencia de imágenes en el directorio `output/` mostrando:
- La evolución del mundo acuático
- Protozoos (azul) evitando depredadores (rojo)
- Campos de luz y peligro
- Emergencia de comportamientos sin programación

### 2. Ejecutar experimento evolutivo:

```bash
make evolution
```

Simula 10 generaciones con herencia topológica y genera reportes en `experiment_results/`

### 3. Exploración interactiva:

```bash
make shell
```

Abre un shell en el contenedor para experimentar:

```python
from proteus import World, Protozoa, LuminousPredator
import numpy as np

# Crear tu propio mundo
world = World(size=(500, 500))

# Añadir criaturas
protozoa = [Protozoa(position=np.random.rand(2)*500) for _ in range(50)]
predators = [LuminousPredator(position=np.random.rand(2)*500) for _ in range(5)]

world.add_protozoa(protozoa)
world.add_predators(predators)

# Simular
survivors = world.simulate(protozoa, predators, time_steps=1000)
print(f"Supervivientes: {len(survivors)}")
```

## Comandos Disponibles

- `make build` - Construir imagen Docker
- `make demo` - Ejecutar demo visual
- `make evolution` - Experimento evolutivo
- `make shell` - Shell interactivo
- `make clean` - Limpiar archivos generados
- `make help` - Ver todos los comandos

## Estructura de Salida

```
output/
├── proteus_t0000.png    # Estado inicial
├── proteus_t0050.png    # Después de 50 pasos
├── proteus_t0100.png    # Después de 100 pasos
├── proteus_t0200.png    # ...
├── proteus_t0500.png
├── proteus_t1000.png
├── proteus_summary.png  # Resumen con 4 paneles
└── proteus_evolution_timeline.png  # Línea temporal

experiment_results/
├── basic_survival_results.json  # Datos del experimento
├── seed_bank.pkl               # Banco de semillas topológicas
└── experiment_report.pdf       # Reporte visual completo
```

## Sin Docker

Si prefieres ejecutar sin Docker:

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar demo
python visual_demo.py
```

## Notas

- Las visualizaciones se guardan automáticamente en `output/`
- El backend matplotlib está configurado para generar imágenes sin GUI
- Cada imagen muestra el estado del mundo y los campos de peligro
- Los colores de los protozoos indican su generación (más oscuro = más evolucionado)