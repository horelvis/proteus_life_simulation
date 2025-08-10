# PROTEUS-AC: Cellular Automaton Life

## Sprint 1 Completado ✓

### ¿Qué es PROTEUS-AC?

Una versión simplificada de PROTEUS que usa Autómatas Celulares (AC) en lugar de sistemas neuronales complejos. Cada organismo tiene un AC interno de 16×16 que controla su comportamiento mediante reglas locales simples.

### Arquitectura

```
Capa Mundo (PROTEUS original simplificado)
    ↓
Sensores → AC 16×16 → Efectores
    ↑                      ↓
Percepción            Movimiento
```

### Componentes Implementados

#### 1. CellularAutomaton.js
- Grid 16×16 con vecindario Moore
- Estados celulares: tissue, skin, void
- Propiedades: energía, carga, activación, fase

#### 2. Sensores (Entrada)
- **Luz**: Detecta destellos de depredadores
- **Gradiente químico**: Dirección hacia nutrientes
- **Daño**: Respuesta a ataques
- **Densidad de vecinos**: Para comportamiento social
- **Ritmo de pulso**: Oscilador interno

#### 3. Efectores (Salida)
- **Movimiento (dx, dy)**: Vector de desplazamiento
- **Rigidez**: Afecta velocidad máxima
- **Oscilación**: Patrón rítmico de movimiento

#### 4. Reglas Locales

##### Percepción Orientada (Quimiotaxis/Fototaxis)
```javascript
// Células se activan hacia gradientes químicos (comida)
if (chemAlignment > 0.5) {
    next.activation = aumentar
}

// Células se activan lejos de la luz (depredadores)
if (lightAlignment > 0.5 && light > threshold) {
    next.activation = aumentar (lado opuesto)
}
```

##### Homeostasis/Auto-reparación
```javascript
// Células muertas pueden regenerarse si tienen ≥3 vecinos sanos
if (cell.type === 'void' && healthyNeighbors >= 3) {
    cell.type = 'tissue'
}

// Difusión de energía entre células vecinas
energy = energy + (avgNeighborEnergy - energy) * diffusionRate
```

##### Morfogénesis
```javascript
// Formación de piel en bordes con alta densidad
if (isBoundary && neighborDensity > 0.7) {
    cell.type = 'skin'
    cell.rigidity = 1.0
}
```

##### Oscilador Central
- Anillo de células en el centro que se activan en secuencia
- Genera patrones de movimiento coordinado

### Herencia

Los organismos heredan:
1. **Parámetros de reglas** (umbrales, fuerzas, tasas)
2. **Patrones estructurales** (configuración inicial del AC)
3. **Marcadores de herencia** (memoria de generaciones)

### Métricas

1. **Estabilidad Morfológica**: Varianza del centro de masa
2. **Eficiencia de Forrajeo**: Energía obtenida / distancia recorrida
3. **Generación Promedio**: Evolución de la población

### Cómo Ejecutar

```bash
cd frontend
npm start
```

La aplicación mostrará:
- Canvas principal con organismos, nutrientes y depredadores
- Panel de estadísticas en tiempo real
- Grid del AC al hacer clic en un organismo

### Próximos Sprints

**Sprint 2**: Morfogénesis avanzada y memoria heredada
**Sprint 3**: Estigmergia y cooperación mediante feromonas
**Sprint 4**: Optimización GPU y análisis de ablación

### Observaciones Iniciales

1. Los organismos muestran quimiotaxis clara hacia nutrientes
2. La fototaxis (evitar depredadores) funciona pero necesita ajuste
3. El oscilador central genera movimiento rítmico interesante
4. La herencia de parámetros produce variación evolutiva

### Diferencias con PROTEUS Original

| PROTEUS Original | PROTEUS-AC |
|-----------------|------------|
| Sistemas neuronales complejos | AC 16×16 simple |
| Múltiples órganos especializados | Estados celulares básicos |
| Topología compleja | Reglas locales simples |
| Alto costo computacional | Eficiente y escalable |

### Debug

- Click en organismo muestra su grid AC interno
- Colores en el grid:
  - Gris: void (vacío)
  - Marrón: skin (piel)
  - Rojo-Verde: tissue (activación y energía)
- Líneas blancas: vectores de movimiento