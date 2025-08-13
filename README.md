# 🧠 PROTEUS ARC - Sistema de Razonamiento Abstracto

PROTEUS ARC es un sistema de inteligencia artificial diseñado para resolver puzzles del "Abstraction and Reasoning Corpus" (ARC). El enfoque principal de este proyecto es experimentar con un modelo de razonamiento basado en el análisis topológico y estructural, aprendiendo de los ejemplos de entrenamiento de forma dinámica.

Este proyecto se ha refactorizado para eliminar resultados engañosos y código no funcional, centrándose en un enfoque de IA honesto y transparente.

## 🚀 Quick Start

### Con Docker (Recomendado)

```bash
# Clonar y ejecutar
git clone https://github.com/usuario/proteus_arc_solver.git
cd proteus_arc_solver
docker-compose up -d

# Ver logs del backend
docker-compose logs -f backend
```

### Ejecución Local

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# Evaluar el score real del sistema en el dataset ARC
python evaluate_arc_score.py
```

### Acceso al Sistema

- **Frontend**: [http://localhost:3001](http://localhost:3001)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Documentación API**: [http://localhost:8000/docs](http://localhost:8000/docs)


## 🔬 Arquitectura: Razonamiento Topológico Adaptativo

El núcleo del solver es el `HybridProteusARCSolver`, que funciona de la siguiente manera:

1.  **Análisis Topológico**: Extrae una "firma" estructural de cada grid (dimensión fractal, componentes, agujeros, simetría, etc.).
2.  **Aprendizaje Dinámico**: Para cada puzzle, el sistema analiza los ejemplos de entrenamiento. Aprende qué firmas topológicas de entrada se corresponden con qué tipos de transformación. No utiliza reglas o firmas hardcodeadas.
3.  **Inferencia por Similitud**: Para resolver un grid de prueba, calcula su firma topológica y la compara con las que ha aprendido. La regla asociada a la firma más similar de los ejemplos de entrenamiento es la que se aplica.

Para más detalles, consulta el documento `backend/SYSTEM_OVERVIEW.md`.

## 🏆 Resultados y Filosofía

Tras la refactorización, el sistema ya no utiliza atajos ni puzzles pre-seleccionados para inflar las métricas. El script `evaluate_arc_score.py` ahora proporciona una evaluación honesta del rendimiento del solver en el dataset ARC.

El objetivo no es conseguir un score alto por cualquier medio, sino explorar la viabilidad de un razonamiento abstracto basado en principios matemáticos y estructurales, de una forma transparente y verificable.

## 🤝 Contribución

1.  Fork el repositorio.
2.  Crea una rama para tu feature (`git checkout -b feature/nueva-transformacion`).
3.  Realiza tus cambios.
4.  Añade tests si es aplicable.
5.  Envía un Pull Request.

## 📝 Licencia

MIT License - ver el archivo LICENSE para más detalles.
