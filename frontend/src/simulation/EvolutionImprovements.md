# Mejoras Evolutivas para PROTEUS

## Problemas Identificados con Parametrizaciones

1. **Umbrales artificiales** que impiden la selección natural:
   - Edad mínima para reproducción (era 5)
   - Energía mínima fija (era 1.5)
   - Thresholds de expresión de órganos (era 0.01)
   - Límites de velocidad artificiales
   - Probabilidades fijas de detección

2. **Órganos sin ventajas evolutivas reales**:
   - Los órganos solo sumaban a capacidades abstractas
   - No había trade-offs reales (costo vs beneficio)
   - Efectos lineales y predecibles

3. **Comportamientos hardcodeados**:
   - Respuestas de pánico fijas
   - Distancias de detección parametrizadas
   - Decisiones basadas en if-else rígidos

## Soluciones Implementadas

### 1. Eliminación de Umbrales
- **Reproducción**: Ahora es probabilística basada en energía y madurez
- **Muerte**: Solo por energía <= 0 o edad > esperanza de vida genética
- **Expresión de órganos**: Cualquier expresión > 0 crea un órgano funcional

### 2. Órganos con Efectos Reales
Cada órgano tiene:
- **Costo energético**: Proporcional a su expresión
- **Efectos múltiples**: Pueden tener ventajas y desventajas
- **Interacciones**: Los órganos interactúan entre sí

Ejemplos:
- `armor_plates`: Protege pero reduce velocidad
- `speed_boost`: Aumenta velocidad pero consume más energía
- `vacuole`: Aumenta capacidad energética Y eficiencia de absorción

### 3. Percepción Realista
- **Sin ojos = sin visión**: Los organismos sin fotosensores NO pueden ver depredadores
- **Quimiorreceptores**: Necesarios para detectar nutrientes a distancia
- **Rango variable**: Depende de la funcionalidad del órgano

### 4. Nuevos Tipos de Órganos Emergentes
Sistema para permitir evolución de NUEVOS órganos:
- `magnetoreceptor`: Navegación magnética
- `pressure_sensor`: Detecta vibraciones
- `thermal_sensor`: Detecta gradientes de temperatura
- `bio_luminescence`: Produce luz propia
- `acid_gland`: Defensa ácida
- `sticky_secretion`: Atrapa presas pequeñas
- `filter_feeding`: Alimentación pasiva
- `symbiotic_chamber`: Aloja bacterias beneficiosas
- `neural_ganglion`: Mejora procesamiento de información

### 5. Decisiones Emergentes
- Las decisiones emergen de la interacción entre:
  - Percepción (limitada por órganos)
  - Estado interno (energía, edad)
  - Memoria holográfica (experiencias pasadas)
  - Campo topológico (navegación inteligente)

### 6. Trade-offs Evolutivos Reales

| Estrategia | Ventajas | Desventajas |
|------------|----------|-------------|
| Muchos órganos sensoriales | Mejor detección de amenazas/comida | Alto costo energético |
| Armadura pesada | Supervivencia a ataques | Lentitud, alto costo |
| Velocidad extrema | Escape rápido | Consume mucha energía |
| Camuflaje | Evita detección | No protege si es detectado |
| Toxinas | Defensa activa | Costo de producción |
| Regeneración | Recuperación de daño | Costo energético continuo |

## Implementación

1. **Reemplazar Organism.js** con OrganismEvolved.js
2. **Actualizar ProteusInheritance.js** para soportar nuevos genes de órganos
3. **Modificar Predator.js** para interactuar con nuevas defensas
4. **Ajustar costos energéticos** para balance evolutivo

## Resultados Esperados

- **Diversificación**: Múltiples estrategias viables
- **Especialización**: Organismos adaptados a nichos específicos
- **Innovación**: Aparición de nuevos órganos y comportamientos
- **Extinción selectiva**: Solo las estrategias no viables desaparecen
- **Coevolución**: Depredadores y presas evolucionan juntos