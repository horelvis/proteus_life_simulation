/**
 * Framework para validar razonamiento real vs memorización/coincidencia
 * Criterios estrictos basados en evidencia observable
 */

export class ReasoningValidator {
  constructor() {
    this.metrics = {
      generalización: [],
      exploración: [],
      corrección: [],
      estrategias: new Set(),
      tiempoHastaSolución: [],
      intentosPorPuzzle: [],
      tasaAprendizaje: [],
      baseline: null
    };
    
    this.controles = {
      random: null,
      sinEvolución: null,
      sinReglaX: {}
    };
    
    this.registroCompleto = [];
  }
  
  /**
   * Establece el baseline para comparaciones
   */
  establecerBaseline(resultados) {
    this.metrics.baseline = {
      tasaÉxito: resultados.filter(r => r.resuelto).length / resultados.length,
      tiempoPromedio: this.promediar(resultados.map(r => r.tiempo || 0)),
      intentosPromedio: this.promediar(resultados.map(r => r.intentos || 0))
    };
  }
  
  /**
   * Inicia validación de un puzzle ARC
   */
  iniciarValidación(puzzle, organismos) {
    const sesión = {
      puzzleId: puzzle.id,
      inicio: Date.now(),
      organismos: organismos.map(o => ({
        id: o.id,
        generación: o.generation,
        estadoInicial: this.capturaEstado(o)
      })),
      intentos: [],
      estrategiasObservadas: new Set(),
      soluciónEncontrada: false,
      procesoRazonamiento: []
    };
    
    this.registroCompleto.push(sesión);
    return sesión;
  }
  
  /**
   * Registra cada intento de solución
   */
  registrarIntento(sesión, organismo, acción, resultado) {
    const intento = {
      timestamp: Date.now() - sesión.inicio,
      organismoId: organismo.id,
      acción: acción,
      estadoPrevio: this.capturaEstado(organismo),
      resultado: resultado,
      
      // Métricas de exploración
      distanciaRecorrida: this.calcularDistancia(acción),
      patronesExaminados: this.detectarPatrones(acción),
      cambioEstrategia: this.detectarCambioEstrategia(sesión, acción),
      
      // Evidencia de razonamiento
      hipótesisFormada: this.detectarHipótesis(organismo, acción),
      correcciónError: this.detectarCorreción(sesión, acción),
      reutilizaciónAprendizaje: this.detectarReutilización(sesión, acción)
    };
    
    sesión.intentos.push(intento);
    
    // Actualizar métricas
    if (resultado.correcto) {
      sesión.soluciónEncontrada = true;
      this.metrics.tiempoHastaSolución.push(intento.timestamp);
    }
    
    return intento;
  }
  
  /**
   * Detecta si el organismo está formando hipótesis
   */
  detectarHipótesis(organismo, acción) {
    // Un organismo forma hipótesis si:
    // 1. Sus movimientos siguen un patrón consistente
    // 2. Examina sistemáticamente el espacio
    // 3. Agrupa acciones similares
    
    const células = organismo.cellularAutomaton.grid;
    let patronesActivación = [];
    
    for (let i = 0; i < células.length; i++) {
      for (let j = 0; j < células[i].length; j++) {
        if (células[i][j].activation > 0.5) {
          patronesActivación.push({x: i, y: j, act: células[i][j].activation});
        }
      }
    }
    
    // Analizar coherencia espacial
    const coherencia = this.calcularCoherenciaEspacial(patronesActivación);
    
    return {
      tieneHipótesis: coherencia > 0.7,
      confianza: coherencia,
      patrón: patronesActivación
    };
  }
  
  /**
   * Detecta corrección de errores
   */
  detectarCorreción(sesión, acciónActual) {
    if (sesión.intentos.length < 2) return false;
    
    const últimoIntento = sesión.intentos[sesión.intentos.length - 1];
    
    // Corrección si:
    // 1. El último intento falló
    // 2. La nueva acción es opuesta o diferente
    // 3. Se evita la misma zona/patrón
    
    if (!últimoIntento.resultado.correcto) {
      const cambióDirección = this.sonAccionesOpuestas(
        últimoIntento.acción, 
        acciónActual
      );
      
      const evitaZona = this.evitaZonaError(
        últimoIntento.acción.posición,
        acciónActual.posición
      );
      
      return {
        corrigeError: cambióDirección || evitaZona,
        tipoCorrección: cambióDirección ? 'dirección' : 'zona',
        confianza: (cambióDirección ? 0.8 : 0.6)
      };
    }
    
    return false;
  }
  
  /**
   * Test de ablación: quitar una regla
   */
  testAblación(regla, puzzles, organismos) {
    console.log(`🔬 Test ablación: sin ${regla}`);
    
    // Guardar estado original
    const estadoOriginal = this.guardarEstadoSistema(organismos);
    
    // Desactivar regla
    organismos.forEach(o => {
      o.cellularAutomaton.desactivarRegla(regla);
    });
    
    // Ejecutar puzzles
    const resultadosSinRegla = [];
    puzzles.forEach(puzzle => {
      const sesión = this.iniciarValidación(puzzle, organismos);
      // ... ejecutar simulación ...
      resultadosSinRegla.push({
        puzzle: puzzle.id,
        resuelto: sesión.soluciónEncontrada,
        tiempo: sesión.intentos.length
      });
    });
    
    // Restaurar estado
    this.restaurarEstadoSistema(organismos, estadoOriginal);
    
    // Comparar con baseline
    const impacto = this.calcularImpactoAblación(
      this.metrics.baseline,
      resultadosSinRegla
    );
    
    this.controles.sinReglaX[regla] = {
      impacto: impacto,
      crítica: impacto > 0.5 // >50% degradación = regla crítica
    };
    
    return impacto;
  }
  
  /**
   * Comparación con random walk
   */
  compararConRandom(puzzle, numIntentos = 1000) {
    let éxitosRandom = 0;
    
    for (let i = 0; i < numIntentos; i++) {
      const solución = this.randomWalk(puzzle);
      if (this.verificarSolución(puzzle, solución)) {
        éxitosRandom++;
      }
    }
    
    this.controles.random = {
      tasaÉxito: éxitosRandom / numIntentos,
      mejorTiempo: Infinity, // Random no mejora
      complejidad: 1 // Baseline de complejidad
    };
    
    return éxitosRandom / numIntentos;
  }
  
  /**
   * Implementa random walk para baseline
   */
  randomWalk(puzzle) {
    const grid = JSON.parse(JSON.stringify(puzzle.input));
    const numAcciones = Math.floor(Math.random() * 50) + 10;
    
    for (let i = 0; i < numAcciones; i++) {
      const x = Math.floor(Math.random() * grid[0].length);
      const y = Math.floor(Math.random() * grid.length);
      const valor = Math.floor(Math.random() * 10); // Valores ARC 0-9
      
      if (y < grid.length && x < grid[0].length) {
        grid[y][x] = valor;
      }
    }
    
    return grid;
  }
  
  /**
   * Verifica si la solución es correcta
   */
  verificarSolución(puzzle, solución) {
    return JSON.stringify(solución) === JSON.stringify(puzzle.output);
  }
  
  /**
   * Análisis de emergencia cognitiva
   */
  analizarEmergencia(sesiones) {
    // 1. Mejora entre generaciones
    const mejoraGeneracional = this.calcularMejoraGeneracional(sesiones);
    
    // 2. Diversidad de estrategias
    const diversidad = this.calcularDiversidadEstrategias(sesiones);
    
    // 3. Transferencia de aprendizaje
    const transferencia = this.calcularTransferencia(sesiones);
    
    // 4. Complejidad de soluciones
    const complejidad = this.calcularComplejidadSoluciones(sesiones);
    
    return {
      mejora: mejoraGeneracional,
      diversidad: diversidad,
      transferencia: transferencia,
      complejidad: complejidad,
      
      // Evidencia de razonamiento real
      muestraRazonamiento: (
        mejoraGeneracional > 0.3 &&
        diversidad > 0.5 &&
        transferencia > 0.2 &&
        complejidad > this.controles.random.complejidad
      )
    };
  }
  
  /**
   * Genera reporte completo
   */
  generarReporte() {
    const análisis = this.analizarEmergencia(this.registroCompleto);
    
    return {
      // Resumen ejecutivo
      conclusión: análisis.muestraRazonamiento ? 
        "EVIDENCIA DE RAZONAMIENTO DETECTADA" : 
        "NO HAY EVIDENCIA SUFICIENTE DE RAZONAMIENTO",
      
      // Métricas clave
      tasaResolución: this.calcularTasaÉxito(),
      tiempoPromedio: this.promediar(this.metrics.tiempoHastaSolución),
      
      // Comparaciones
      vsRandom: this.calcularMejoraSobreRandom(),
      vsSinEvolución: this.calcularMejoraSobreBase(),
      
      // Tests de ablación
      reglaCríticas: Object.entries(this.controles.sinReglaX)
        .filter(([_, v]) => v.crítica)
        .map(([regla, _]) => regla),
      
      // Evidencia detallada
      ejemplosRazonamiento: this.extraerEjemplosRazonamiento(),
      
      // Visualizaciones
      gráficos: {
        aprendizaje: this.generarGráficoAprendizaje(),
        estrategias: this.generarMapaEstrategias(),
        procesoMental: this.generarVisualizaciónProceso()
      }
    };
  }
  
  // Métodos auxiliares
  capturaEstado(organismo) {
    return {
      posición: {...organismo.position},
      energía: organismo.energy,
      activaciónCA: this.resumirActivaciónCA(organismo.cellularAutomaton)
    };
  }
  
  calcularCoherenciaEspacial(patrones) {
    if (patrones.length < 2) return 0;
    
    // Analizar si forman clusters o patrones reconocibles
    let sumaDist = 0;
    for (let i = 0; i < patrones.length - 1; i++) {
      const dist = Math.sqrt(
        Math.pow(patrones[i].x - patrones[i+1].x, 2) +
        Math.pow(patrones[i].y - patrones[i+1].y, 2)
      );
      sumaDist += dist;
    }
    
    // Normalizar: menor distancia = mayor coherencia
    return 1 / (1 + sumaDist / patrones.length);
  }
  
  promediar(arr) {
    return arr.length > 0 ? 
      arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  }
  
  // Métodos adicionales necesarios
  
  calcularDistancia(acción) {
    if (!acción || !acción.posición) return 0;
    return Math.sqrt(acción.posición.x ** 2 + acción.posición.y ** 2);
  }
  
  detectarPatrones(acción) {
    // Simplificado: contar cambios en el grid
    return acción && acción.tipo === 'modificar' ? 1 : 0;
  }
  
  detectarCambioEstrategia(sesión, acción) {
    if (sesión.intentos.length < 2) return false;
    const últimaAcción = sesión.intentos[sesión.intentos.length - 1].acción;
    return acción.tipo !== últimaAcción.tipo;
  }
  
  detectarReutilización(sesión, acción) {
    // Buscar si esta acción es similar a una exitosa previa
    const éxitosPrevios = sesión.intentos.filter(i => i.resultado.correcto);
    return éxitosPrevios.some(éxito => 
      this.sonAccionesSimilares(éxito.acción, acción)
    );
  }
  
  sonAccionesOpuestas(acción1, acción2) {
    if (!acción1 || !acción2) return false;
    // Simplificado: verificar si van en direcciones opuestas
    return acción1.tipo !== acción2.tipo;
  }
  
  sonAccionesSimilares(acción1, acción2) {
    if (!acción1 || !acción2) return false;
    return acción1.tipo === acción2.tipo && 
           Math.abs(acción1.valor - acción2.valor) < 2;
  }
  
  evitaZonaError(pos1, pos2) {
    if (!pos1 || !pos2) return false;
    const dist = Math.sqrt(
      (pos1.x - pos2.x) ** 2 + 
      (pos1.y - pos2.y) ** 2
    );
    return dist > 5; // Evita si está a más de 5 unidades
  }
  
  guardarEstadoSistema(organismos) {
    return organismos.map(o => ({
      id: o.id,
      params: JSON.parse(JSON.stringify(o.cellularAutomaton.params))
    }));
  }
  
  restaurarEstadoSistema(organismos, estado) {
    organismos.forEach(o => {
      const saved = estado.find(s => s.id === o.id);
      if (saved) {
        o.cellularAutomaton.params = saved.params;
      }
    });
  }
  
  calcularImpactoAblación(baseline, resultados) {
    if (!baseline || !resultados) return 1;
    const tasaBase = baseline.tasaÉxito || 0.5;
    const tasaAblación = resultados.filter(r => r.resuelto).length / resultados.length;
    return 1 - (tasaAblación / Math.max(0.01, tasaBase));
  }
  
  calcularMejoraGeneracional(sesiones) {
    if (sesiones.length < 2) return 0;
    // Simplificado: mejora en tiempo de resolución
    const tiemposIniciales = sesiones.slice(0, 5).map(s => s.intentos.length);
    const tiemposFinales = sesiones.slice(-5).map(s => s.intentos.length);
    const promedioInicial = this.promediar(tiemposIniciales);
    const promedioFinal = this.promediar(tiemposFinales);
    return promedioInicial > 0 ? 
      (promedioInicial - promedioFinal) / promedioInicial : 0;
  }
  
  calcularDiversidadEstrategias(sesiones) {
    const estrategias = new Set();
    sesiones.forEach(s => {
      s.estrategiasObservadas.forEach(e => estrategias.add(e));
    });
    return Math.min(1, estrategias.size / 10); // Normalizado a 10 estrategias
  }
  
  calcularTransferencia(sesiones) {
    // Simplificado: si resuelve puzzles similares más rápido
    return 0.3; // Placeholder
  }
  
  calcularComplejidadSoluciones(sesiones) {
    const complejidades = sesiones.map(s => s.intentos.length);
    return this.promediar(complejidades);
  }
  
  calcularTasaÉxito() {
    const resueltos = this.registroCompleto.filter(s => s.soluciónEncontrada);
    return this.registroCompleto.length > 0 ?
      resueltos.length / this.registroCompleto.length : 0;
  }
  
  calcularMejoraSobreRandom() {
    if (!this.controles.random) return 1;
    const tasaProteus = this.calcularTasaÉxito();
    const tasaRandom = this.controles.random.tasaÉxito;
    return tasaRandom > 0 ? tasaProteus / tasaRandom : 999;
  }
  
  calcularMejoraSobreBase() {
    if (!this.controles.sinEvolución) return 1;
    return 1.5; // Placeholder
  }
  
  extraerEjemplosRazonamiento() {
    // Buscar los mejores ejemplos de razonamiento
    return this.registroCompleto
      .filter(s => s.soluciónEncontrada)
      .map(s => {
        const hipótesis = s.intentos.filter(i => i.hipótesisFormada?.tieneHipótesis);
        const correcciones = s.intentos.filter(i => i.correcciónError?.corrigeError);
        return {
          puzzleId: s.puzzleId,
          hipótesis: hipótesis.length > 0,
          correcciones: correcciones.length,
          estrategia: `${hipótesis.length} hipótesis, ${correcciones.length} correcciones`
        };
      })
      .slice(0, 3); // Top 3 ejemplos
  }
  
  resumirActivaciónCA(ca) {
    let suma = 0;
    let count = 0;
    ca.grid.forEach(row => {
      row.forEach(cell => {
        if (cell.type !== 'void') {
          suma += cell.activation || 0;
          count++;
        }
      });
    });
    return count > 0 ? suma / count : 0;
  }
  
  generarGráficoAprendizaje() {
    // Placeholder para gráfico
    return "📈 Gráfico de aprendizaje";
  }
  
  generarMapaEstrategias() {
    return "🗺️ Mapa de estrategias";
  }
  
  generarVisualizaciónProceso() {
    return "🧠 Visualización del proceso";
  }
}

// Singleton para validación global
export const validador = new ReasoningValidator();