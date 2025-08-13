# Arquitectura del Solver ARC: Razonamiento Topológico Adaptativo

## Filosofía del Diseño

El objetivo de este sistema es resolver puzzles del "Abstraction and Reasoning Corpus" (ARC) a través de un razonamiento basado en propiedades estructurales y topológicas, en lugar de depender de un gran número de reglas de transformación hardcodeadas.

El principio clave es que la *forma* y la *estructura* de los datos de entrada pueden guiar la selección de la transformación correcta. El sistema aprende las características topológicas asociadas a diferentes transformaciones a partir de los ejemplos de entrenamiento de un puzzle y luego aplica este conocimiento para resolver el caso de prueba.

## Arquitectura Principal

El sistema se centra en el `HybridProteusARCSolver`, que implementa la siguiente lógica:

1.  **Análisis Topológico**: Utiliza un `TopologicalAnalyzer` para extraer una "firma" de cada grid. Esta firma es un vector de características que describe la estructura del grid.
    *   **Dimensión Fractal**: Mide la complejidad del patrón.
    *   **Componentes Conectados**: Número de "islas" o grupos de píxeles.
    *   **Agujeros**: Número de espacios vacíos rodeados.
    *   **Densidad**: Porcentaje de píxeles no nulos.
    *   **Simetría**: Puntuación de simetría horizontal/vertical.
    *   **Ratio de Borde**: Proporción de píxeles en los bordes.

2.  **Fase de Aprendizaje**: Para un puzzle dado, el solver itera sobre todos los ejemplos de entrenamiento (`train_examples`).
    *   Para cada par `(input, output)`, utiliza un conjunto de detectores de reglas base (`ARCSolverPython`) para identificar la transformación que convierte la entrada en la salida (p. ej., `FILL_SHAPE`, `COLOR_MAPPING`).
    *   Calcula la firma topológica del grid de **entrada**.
    *   Almacena esta firma, asociándola con la regla de transformación encontrada. Esto crea un mapa dinámico de `regla -> [lista de firmas que la activan]`.

3.  **Fase de Inferencia**: Una vez que el aprendizaje ha finalizado, el solver aborda el grid de prueba (`test_input`).
    *   Calcula la firma topológica del grid de prueba.
    *   Compara la firma del grid de prueba con **todas las firmas aprendidas** de la fase de entrenamiento, calculando la "distancia" topológica.
    *   La regla asociada a la firma aprendida más cercana (con la distancia mínima) se selecciona como la transformación más probable.
    *   Se recuperan los parámetros específicos de esa regla (p. ej., el mapa de colores) a partir del ejemplo de entrenamiento correspondiente.
    *   La regla se aplica al grid de prueba para generar la solución.

## Flujo de Razonamiento

```
PARA CADA puzzle:
  1. INICIALIZAR mapa de firmas aprendidas (vacío)
  2. PARA CADA ejemplo de entrenamiento:
     a. DETECTAR regla de transformación (base)
     b. CALCULAR firma topológica del INPUT
     c. GUARDAR firma en el mapa, asociada a la regla
  3. CALCULAR firma topológica del grid de PRUEBA
  4. BUSCAR la firma aprendida más CERCANA a la firma de prueba
  5. SELECCIONAR la regla asociada a esa firma
  6. APLICAR regla al grid de prueba -> SOLUCIÓN
```

## Componentes Clave

1.  **`hybrid_proteus_solver.py`**: Contiene la lógica principal del `HybridProteusARCSolver` y el `TopologicalAnalyzer`. Es el cerebro del sistema.
2.  **`arc_solver_python.py`**: Proporciona las implementaciones base para la detección y aplicación de reglas de transformación individuales. Actúa como la "caja de herramientas" que el solver híbrido utiliza de forma inteligente.
3.  **`evaluate_arc_score.py`**: Script para evaluar el rendimiento del solver en el dataset ARC de forma honesta.

## Estado Actual

El sistema es funcional y se basa en un principio de razonamiento sólido y honesto. El rendimiento ya no se infla con casos de prueba hardcodeados. Las futuras mejoras se centrarán en enriquecer la firma topológica y mejorar los detectores de reglas base.
