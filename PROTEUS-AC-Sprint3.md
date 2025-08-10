# PROTEUS-AC: Sprint 3 Completado ✓

## Estigmergia y Cooperación

### Características Implementadas

#### 1. Campo Global de Feromonas
- Grid espacial para 4 tipos de feromonas
- Decay exponencial (98% por frame)
- Difusión espacial (10% a vecinos)
- Evaporación completa bajo umbral

#### 2. Tipos de Feromonas

```javascript
FOOD: 0,    // Verde - rastro hacia comida
DANGER: 1,  // Rojo - señal de peligro/muerte
COLONY: 2,  // Azul - marca de colonia/hogar
MATING: 3   // Púrpura - señal reproductiva
```

#### 3. Secreción Química por Organismos

Los organismos secretan feromonas según su estado interno:

- **DANGER**: Cuando hay daño (>3 células dañadas)
- **FOOD**: Cuando detectan comida y tienen energía alta
- **COLONY**: En grupos con alta densidad de vecinos
- **MATING**: Con energía alta y muchas células sanas

#### 4. Seguimiento de Gradientes

Los organismos ahora:
- Detectan gradientes de feromonas en su posición
- Siguen rastros de FOOD (quimiotaxis positiva)
- Evitan áreas de DANGER (quimiotaxis negativa)
- Integran gradientes con detección directa de nutrientes

#### 5. Métricas de Cooperación/Competencia

**Índice de Cooperación** (0-1):
- Cobertura de rastros de comida compartidos
- Clustering por feromonas de colonia
- Distancia promedio entre organismos (inversa)

**Índice de Competencia** (0-1):
- Ratio organismos/nutrientes
- Cobertura de señales de peligro
- Escasez de recursos

### Comportamientos Emergentes

1. **Formación de Rastros**
   - Los organismos exitosos dejan rastros de FOOD
   - Otros siguen estos rastros (información compartida)
   - Se forman "autopistas" químicas

2. **Señales de Alarma**
   - Organismos dañados secretan DANGER
   - Otros evitan estas áreas
   - Memoria espacial del peligro

3. **Agregación Colonial**
   - Grupos secretan COLONY
   - Atrae a otros organismos
   - Formación de clusters estables

4. **Coordinación Reproductiva**
   - Señales MATING cuando listos
   - Sincronización espacial de reproducción

### Visualización

- **Verde transparente**: Rastros de comida
- **Rojo transparente**: Zonas de peligro
- **Azul transparente**: Áreas coloniales
- **Púrpura transparente**: Señales de apareamiento

### Parámetros Clave

```javascript
decayRate: 0.98         // Persistencia temporal
diffusionRate: 0.1      // Velocidad de dispersión
evaporationThreshold: 0.01  // Límite de detección
gridSize: 20            // Resolución espacial
```

### Interacciones Emergentes

1. **Cooperación Indirecta**
   - Sin comunicación directa
   - Modificación del ambiente compartido
   - Beneficio mutuo por información

2. **Competencia por Recursos**
   - Rastros llevan a múltiples organismos
   - Agotamiento local de nutrientes
   - Dispersión forzada

3. **Memoria Ambiental**
   - El campo retiene historia reciente
   - Organismos aprenden de experiencias ajenas
   - Adaptación colectiva

### Métricas Sprint 3

- **Eficiencia de Forrajeo**: Mejorada con rastros
- **Supervivencia**: Aumentada por señales de peligro
- **Diversidad de Estrategias**: Seguir vs explorar
- **Robustez del Sistema**: Mayor con estigmergia

### Próximo Sprint (4): Optimización y Análisis

- Vectorización para GPU
- Ablaciones sistemáticas
- Comparación con/sin cada componente
- Informe final con visualizaciones