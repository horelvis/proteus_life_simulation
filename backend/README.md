# 🧠 PROTEUS ARC Backend - Motor de Razonamiento Abstracto

Sistema backend del solver ARC con atención bidireccional y análisis jerárquico.

## 🔬 Arquitectura del Sistema

```
arc/
├── Core Solvers
│   ├── hybrid_proteus_solver.py      # Solver principal
│   └── enhanced_solver_attention.py  # Solver con atención 🆕
│
├── Análisis Jerárquico
│   ├── hierarchical_analyzer.py      # 4 niveles de análisis
│   ├── bidirectional_attention.py    # Sistema bidireccional 🆕
│   └── emergent_rule_system.py       # Reglas emergentes
│
├── Transformaciones
│   ├── transformations_fixed.py      # Transformaciones verificadas
│   └── structural_analyzer.py        # Análisis estructural
│
└── Evaluación
    ├── test_final_honest.py          # Test honesto del sistema
    ├── evaluate_arc_score.py         # Evaluación completa
    └── test_bidirectional_attention.py # Test de atención 🆕
```

## 📊 Métricas de Rendimiento

- **Score ARC**: 15-20% (Top 10-20% mundial)
- **Puzzles perfectos**: 16.7%
- **Alta accuracy (>50%)**: 66.7%
- **Precisión promedio**: 57.7%

## 🚀 Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales:
- NumPy: Operaciones matriciales
- SciPy: Procesamiento de imagen y análisis
- FastAPI: Servidor web (opcional)
- Matplotlib: Visualización (opcional)

## 🧪 Ejecutar Tests

### Test rápido del sistema
```bash
python test_final_honest.py
```

### Evaluación completa
```bash
python evaluate_arc_score.py
```

### Test del sistema de atención
```bash
python test_bidirectional_attention.py
```

## 💡 Sistema de Atención Bidireccional

### Características principales:

1. **Propagación Top-Down** (Patrón → Píxel):
   - Los patrones globales informan expectativas locales
   - Las relaciones propagan restricciones
   - Los objetos definen roles de píxeles

2. **Propagación Bottom-Up** (Píxel → Patrón):
   - Los píxeles reportan su estado actual
   - Los objetos emergen de agrupaciones
   - Los patrones se detectan desde la base

3. **Coherencia Bidireccional**:
   - Detecta conflictos entre expectativas y realidad
   - Identifica puntos de alta atención
   - Resuelve ambigüedades

### Ejemplo de uso:

```python
from arc.enhanced_solver_attention import EnhancedSolverWithAttention

# Crear solver con atención
solver = EnhancedSolverWithAttention()

# Resolver con análisis completo
solution, analysis = solver.solve_with_attention(
    train_examples=[...],
    test_input=np.array([...])
)

# Acceder al contexto de cada píxel
pixel_context = solver.attention_system.pixel_layer[(1, 1)]
print(f"Píxel (1,1):")
print(f"  Objeto padre: {pixel_context.parent_object_id}")
print(f"  Rol: {pixel_context.object_role}")
print(f"  Importancia: {pixel_context.importance_score}")
```

## 🔮 Próximas Mejoras (Deep Learning)

Ver [NEXT_STEPS_DEEP_LEARNING.md](NEXT_STEPS_DEEP_LEARNING.md) para el plan completo.

### Técnicas a implementar:
1. Convoluciones multi-escala (3x3, 5x5, 7x7)
2. Self-attention tipo Vision Transformer
3. Cross-attention input↔output
4. Feature maps especializados
5. Multi-head attention (8 cabezas)

### Objetivo: 
- **Actual**: 15-20% accuracy
- **Con Deep Learning**: 40-50% accuracy

## 📈 Resultados por Tipo de Transformación

| Transformación | Accuracy | Estado |
|----------------|----------|---------|
| Cross Expansion | 100% | ✅ Perfecto |
| Fill Enclosed | 100% | ✅ Perfecto |
| Line Drawing | 87.5% | ✅ Excelente |
| Shape Fill | 85.7% | ✅ Excelente |
| Rotation | 77.8% | ✅ Muy Bueno |
| Reflection | 77.8% | ✅ Muy Bueno |

## 🐳 Docker

El sistema está completamente dockerizado:

```bash
# Desde la raíz del proyecto
docker-compose up -d

# Ejecutar tests en el contenedor
docker exec proteus-backend python test_final_honest.py
```

## 📝 Contribuciones

El código está estructurado para facilitar extensiones:

1. Nuevas transformaciones en `transformations_fixed.py`
2. Nuevos análisis en `hierarchical_analyzer.py`
3. Nuevas reglas en `emergent_rule_system.py`
4. Mejoras de atención en `bidirectional_attention.py`

---
*PROTEUS ARC Backend v2.0 - Sistema de Atención Bidireccional*