/**
 * Sistema de benchmark para evaluar capacidades de razonamiento en ARC
 */

export class ARCBenchmark {
  constructor() {
    this.categorías = {
      // Nivel 1: Transformaciones básicas
      color_mapping: { peso: 1, nombre: "Mapeo de colores" },
      pattern_replication: { peso: 2, nombre: "Replicación de patrones" },
      
      // Nivel 2: Transformaciones espaciales
      reflection: { peso: 3, nombre: "Reflexión/Espejo" },
      rotation: { peso: 3, nombre: "Rotación" },
      fill_shape: { peso: 3, nombre: "Rellenar formas" },
      
      // Nivel 3: Razonamiento lógico
      counting: { peso: 4, nombre: "Contar objetos" },
      pattern_extraction: { peso: 4, nombre: "Extracción de patrones" },
      physics_gravity: { peso: 4, nombre: "Física/Gravedad" },
      
      // Nivel 4: Razonamiento abstracto
      symmetry_detection: { peso: 5, nombre: "Detección de simetría" },
      line_drawing: { peso: 5, nombre: "Dibujar líneas/conexiones" }
    };
    
    this.nivelesDificultad = {
      very_easy: 1,
      easy: 2,
      medium: 3,
      hard: 4,
      very_hard: 5
    };
  }
  
  /**
   * Evalúa los resultados y genera métricas de razonamiento
   */
  evaluarResultados(resultados, puzzles) {
    const evaluación = {
      porCategoría: {},
      porDificultad: {},
      general: {
        totalPuzzles: puzzles.length,
        resueltos: 0,
        intentosPromedio: 0,
        tiempoPromedio: 0
      },
      capacidadesDetectadas: [],
      porcentajeRazonamiento: 0,
      nivelRazonamiento: "",
      recomendaciones: []
    };
    
    // Inicializar contadores
    Object.keys(this.categorías).forEach(cat => {
      evaluación.porCategoría[cat] = {
        total: 0,
        resueltos: 0,
        porcentaje: 0,
        nombre: this.categorías[cat].nombre
      };
    });
    
    Object.keys(this.nivelesDificultad).forEach(dif => {
      evaluación.porDificultad[dif] = {
        total: 0,
        resueltos: 0,
        porcentaje: 0
      };
    });
    
    // Procesar resultados
    resultados.forEach((resultado, idx) => {
      const puzzle = puzzles[idx];
      
      if (resultado.resuelto) {
        evaluación.general.resueltos++;
      }
      
      evaluación.general.intentosPromedio += resultado.intentos || 0;
      evaluación.general.tiempoPromedio += resultado.tiempo || 0;
      
      // Por categoría
      if (puzzle.category) {
        evaluación.porCategoría[puzzle.category].total++;
        if (resultado.resuelto) {
          evaluación.porCategoría[puzzle.category].resueltos++;
        }
      }
      
      // Por dificultad
      if (puzzle.difficulty) {
        evaluación.porDificultad[puzzle.difficulty].total++;
        if (resultado.resuelto) {
          evaluación.porDificultad[puzzle.difficulty].resueltos++;
        }
      }
    });
    
    // Calcular promedios
    if (resultados.length > 0) {
      evaluación.general.intentosPromedio /= resultados.length;
      evaluación.general.tiempoPromedio /= resultados.length;
    }
    
    // Calcular porcentajes por categoría
    let puntajeTotal = 0;
    let puntajeMáximo = 0;
    
    Object.keys(evaluación.porCategoría).forEach(cat => {
      const catData = evaluación.porCategoría[cat];
      if (catData.total > 0) {
        catData.porcentaje = (catData.resueltos / catData.total) * 100;
        
        // Calcular puntaje ponderado
        const peso = this.categorías[cat].peso;
        puntajeTotal += catData.resueltos * peso;
        puntajeMáximo += catData.total * peso;
        
        // Detectar capacidades
        if (catData.porcentaje > 0) {
          evaluación.capacidadesDetectadas.push({
            capacidad: catData.nombre,
            nivel: this.getNivelCapacidad(catData.porcentaje)
          });
        }
      }
    });
    
    // Calcular porcentaje de razonamiento general
    if (puntajeMáximo > 0) {
      evaluación.porcentajeRazonamiento = (puntajeTotal / puntajeMáximo) * 100;
    }
    
    // Determinar nivel de razonamiento
    evaluación.nivelRazonamiento = this.getNivelRazonamiento(evaluación.porcentajeRazonamiento);
    
    // Generar recomendaciones
    evaluación.recomendaciones = this.generarRecomendaciones(evaluación);
    
    // Comparar con benchmarks
    evaluación.comparación = this.compararConBenchmarks(evaluación.porcentajeRazonamiento);
    
    return evaluación;
  }
  
  getNivelCapacidad(porcentaje) {
    if (porcentaje >= 80) return "Excelente";
    if (porcentaje >= 60) return "Bueno";
    if (porcentaje >= 40) return "Básico";
    if (porcentaje >= 20) return "Emergente";
    return "No detectado";
  }
  
  getNivelRazonamiento(porcentaje) {
    if (porcentaje >= 90) return "🏆 Nivel superhuman";
    if (porcentaje >= 75) return "🥇 Nivel humano experto";
    if (porcentaje >= 60) return "🥈 Nivel humano promedio";
    if (porcentaje >= 40) return "🥉 Nivel humano básico";
    if (porcentaje >= 20) return "🔰 Razonamiento emergente";
    if (porcentaje >= 10) return "🌱 Indicios de razonamiento";
    if (porcentaje >= 5) return "⚡ Comportamiento reactivo";
    return "❌ Sin razonamiento detectado";
  }
  
  generarRecomendaciones(evaluación) {
    const recs = [];
    
    // Analizar categorías débiles
    Object.entries(evaluación.porCategoría).forEach(([cat, data]) => {
      if (data.total > 0 && data.porcentaje < 50) {
        recs.push({
          tipo: "mejora",
          área: data.nombre,
          sugerencia: `Mejorar capacidad de ${data.nombre.toLowerCase()}. Actualmente: ${data.porcentaje.toFixed(1)}%`
        });
      }
    });
    
    // Sugerir siguiente nivel
    if (evaluación.porcentajeRazonamiento < 20) {
      recs.push({
        tipo: "fundamental",
        área: "Razonamiento básico",
        sugerencia: "Enfocarse en tareas muy simples (color_mapping) para establecer base"
      });
    } else if (evaluación.porcentajeRazonamiento < 50) {
      recs.push({
        tipo: "progreso",
        área: "Razonamiento espacial",
        sugerencia: "Avanzar a transformaciones espaciales (rotación, reflexión)"
      });
    }
    
    return recs;
  }
  
  compararConBenchmarks(porcentaje) {
    return {
      vsRandom: porcentaje / 5, // Random ~5%
      vsGPT4: porcentaje / 30,  // GPT-4 ~30%
      vsHumanoPromedio: porcentaje / 85, // Humano ~85%
      vsHumanoExperto: porcentaje / 95  // Experto ~95%
    };
  }
  
  /**
   * Genera reporte visual detallado
   */
  generarReporteDetallado(evaluación) {
    let reporte = `
╔══════════════════════════════════════════════════════╗
║        EVALUACIÓN DE RAZONAMIENTO PROTEUS-AC         ║
╚══════════════════════════════════════════════════════╝

📊 RESULTADO GENERAL
${evaluación.nivelRazonamiento}
Porcentaje de Razonamiento: ${evaluación.porcentajeRazonamiento.toFixed(1)}%
Puzzles Resueltos: ${evaluación.general.resueltos}/${evaluación.general.totalPuzzles}

📈 ANÁLISIS POR CATEGORÍA
`;
    
    Object.entries(evaluación.porCategoría).forEach(([cat, data]) => {
      if (data.total > 0) {
        const barras = this.generarBarraProgreso(data.porcentaje);
        reporte += `${data.nombre}: ${barras} ${data.porcentaje.toFixed(0)}%\n`;
      }
    });
    
    reporte += `
🧠 CAPACIDADES DETECTADAS
`;
    
    evaluación.capacidadesDetectadas.forEach(cap => {
      reporte += `✓ ${cap.capacidad}: ${cap.nivel}\n`;
    });
    
    reporte += `
🔍 COMPARACIÓN CON BENCHMARKS
vs Random Walk: ${evaluación.comparación.vsRandom.toFixed(1)}x mejor
vs GPT-4: ${(evaluación.comparación.vsGPT4 * 100).toFixed(0)}% del rendimiento
vs Humano Promedio: ${(evaluación.comparación.vsHumanoPromedio * 100).toFixed(0)}% del rendimiento

💡 RECOMENDACIONES
`;
    
    evaluación.recomendaciones.forEach(rec => {
      reporte += `• ${rec.sugerencia}\n`;
    });
    
    return reporte;
  }
  
  generarBarraProgreso(porcentaje) {
    const longitud = 20;
    const llenos = Math.round(porcentaje / 100 * longitud);
    const vacíos = longitud - llenos;
    return '█'.repeat(llenos) + '░'.repeat(vacíos);
  }
}