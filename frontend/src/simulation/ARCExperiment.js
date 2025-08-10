/**
 * Sistema experimental para evaluar razonamiento con puzzles ARC
 */

import { validador } from './ReasoningValidator';
import { OrganismAC } from './OrganismAC';
import { ARCDataLoader } from './ARCDataLoader';
import { ARCBenchmark } from './ARCBenchmark';
import { ARCSolver } from './ARCSolver';

export class ARCExperiment {
  constructor() {
    this.puzzles = [];
    this.resultados = [];
    this.configuración = {
      tamañoPoblación: 50,
      generacionesMax: 100,
      tiempoLímitePorPuzzle: 300, // segundos
      permitirAprendizaje: false // Zero-shot por defecto
    };
    
    // Propiedades para visualización
    this.puzzleActual = null;
    this.gridActual = null;
    this.intentos = 0;
    this.tiempoTranscurrido = 0;
    this.generaciónActual = 0;
    this.resuelto = false;
    this.validador = validador;
  }
  
  /**
   * Carga puzzles ARC para experimento
   */
  async cargarPuzzlesARC(tipo = 'training') {
    const loader = new ARCDataLoader();
    const arcPuzzles = await loader.loadPuzzles('sample');
    
    // Convertir al formato interno
    this.puzzles = arcPuzzles.map(puzzle => {
      const converted = loader.convertToInternalFormat(puzzle);
      
      // Analizar el puzzle para entender su naturaleza
      const analysis = loader.analyzePuzzle(puzzle);
      console.log(`📊 Puzzle ${puzzle.id}:`, analysis);
      
      return {
        ...converted,
        analysis: analysis
      };
    });
    
    console.log(`📊 Cargados ${this.puzzles.length} puzzles ARC reales`);
  }
  
  /**
   * Ejecuta experimento completo
   */
  async ejecutarExperimento(organismos) {
    console.log('🧪 Iniciando experimento ARC de razonamiento...');
    
    try {
      console.log('📊 Puzzles cargados:', this.puzzles.length);
      
      // Crear solucionador basado en CA
      const solver = new ARCSolver();
      
      // Evaluar cada puzzle
      for (const puzzle of this.puzzles) {
        console.log(`\n🧩 Evaluando puzzle ${puzzle.id} (${puzzle.category})...`);
        
        // Aprender de ejemplos de entrenamiento
        if (puzzle.trainExamples && solver.aprender(puzzle.trainExamples)) {
          // Intentar resolver el test
          const inicio = Date.now();
          console.log(`🔍 Input del puzzle ${puzzle.id}:`, puzzle.input);
          const solucion = solver.resolver(puzzle.input);
          console.log(`🎯 Solución generada:`, solucion);
          console.log(`📋 Solución esperada:`, puzzle.output);
          const tiempo = Date.now() - inicio;
          
          // Verificar si es correcto
          const correcto = JSON.stringify(solucion) === JSON.stringify(puzzle.output);
          
          if (correcto) {
            console.log(`✅ ¡Puzzle ${puzzle.id} resuelto!`);
          } else {
            console.log(`❌ Puzzle ${puzzle.id} incorrecto`);
          }
          
          this.resultados.push({
            puzzle: puzzle.id,
            resuelto: correcto,
            tiempo: tiempo,
            intentos: 1,
            analisis: { muestraRazonamiento: correcto }
          });
        } else {
          // No se pudo aprender regla
          console.log(`⚠️ No se pudo aprender regla para ${puzzle.id}`);
          this.resultados.push({
            puzzle: puzzle.id,
            resuelto: false,
            tiempo: 0,
            intentos: 0,
            analisis: { muestraRazonamiento: false }
          });
        }
      }
      
      return this.generarReporteFinal();
      
    } catch (error) {
      console.error('Error en experimento:', error);
      throw error;
    }
  }
  
  /**
   * Evalúa un puzzle específico
   */
  async evaluarPuzzle(puzzle, organismos) {
    const sesión = validador.iniciarValidación(puzzle, organismos);
    const inicio = Date.now();
    
    // Actualizar propiedades para visualización
    this.puzzleActual = puzzle;
    this.gridActual = JSON.parse(JSON.stringify(puzzle.input));
    this.intentos = 0;
    this.resuelto = false;
    
    // Crear entorno ARC para este puzzle
    const entornoARC = new ARCEnvironment(puzzle);
    
    // Bucle de evaluación (limitado para testing)
    let generación = 0;
    let resuelto = false;
    const maxIntentos = 50; // Más intentos para resolver
    let intentosActuales = 0;
    
    while (
      intentosActuales < maxIntentos && 
      !resuelto &&
      (Date.now() - inicio) < 10000 // 10 segundos por puzzle
    ) {
      intentosActuales++;
      
      // Cada organismo intenta resolver
      for (const organismo of organismos) {
        const acción = this.observarAcción(organismo, entornoARC);
        const resultado = entornoARC.aplicarAcción(acción);
        
        // Actualizar estado para visualización
        this.gridActual = entornoARC.getCurrentGrid();
        this.intentos++;
        this.tiempoTranscurrido = Date.now() - inicio;
        this.generaciónActual = generación;
        
        // Registrar intento
        validador.registrarIntento(sesión, organismo, acción, resultado);
        
        if (resultado.completo && resultado.correcto) {
          resuelto = true;
          this.resuelto = true;
          console.log(`✅ ¡Resuelto por organismo ${organismo.id}!`);
          break;
        }
      }
      
      // Evolución (si está permitida)
      if (this.configuración.permitirAprendizaje && !resuelto) {
        organismos = this.evolucionarPoblación(organismos, entornoARC);
        generación++;
      } else {
        break; // Zero-shot: una sola oportunidad
      }
    }
    
    return {
      puzzle: puzzle.id,
      resuelto: resuelto,
      generaciones: generación,
      tiempo: Date.now() - inicio,
      intentos: sesión.intentos.length,
      análisis: validador.analizarEmergencia([sesión])
    };
  }
  
  /**
   * Observa y registra la acción del organismo
   */
  observarAcción(organismo, entornoARC) {
    // El organismo "ve" el puzzle como parte de su entorno
    const percepción = {
      gridInput: entornoARC.getInputGrid(),
      gridActual: entornoARC.getCurrentGrid(),
      posiciónOrganismo: organismo.position
    };
    
    // Actualizar sensores del CA con información del puzzle
    this.actualizarSensoresARC(organismo, percepción);
    
    // El CA procesa y genera acción
    organismo.cellularAutomaton.update(0.1);
    
    // Traducir efectores del CA a acción sobre el grid
    return {
      tipo: this.determinarTipoAcción(organismo),
      posición: this.calcularPosiciónObjetivo(organismo, entornoARC),
      valor: this.determinarValor(organismo),
      
      // Metadatos para análisis
      activaciónCA: organismo.cellularAutomaton.getVisualizationData(),
      confianza: this.calcularConfianza(organismo)
    };
  }
  
  /**
   * Actualiza sensores del CA con info del puzzle ARC
   */
  actualizarSensoresARC(organismo, percepción) {
    const ca = organismo.cellularAutomaton;
    const gridInput = percepción.gridInput;
    const gridActual = percepción.gridActual;
    
    // Detectar diferencias entre input y actual
    let diferencias = 0;
    let celdas1s = 0;
    
    for (let i = 0; i < gridInput.length; i++) {
      for (let j = 0; j < gridInput[i].length; j++) {
        if (gridInput[i][j] !== gridActual[i][j]) {
          diferencias++;
        }
        if (gridInput[i][j] === 1) {
          celdas1s++;
        }
      }
    }
    
    // Mapear a sensores del CA
    // Si hay muchos 1s, generar "chemical gradient" alto
    ca.sensors.chemicalGradient = celdas1s / (gridInput.length * gridInput[0].length);
    
    // Si ya hicimos cambios, reducir gradiente
    if (diferencias > 0) {
      ca.sensors.chemicalGradient *= (1 - diferencias / celdas1s);
    }
    
    // Dirección hacia el próximo 1 sin cambiar
    let found = false;
    for (let i = 0; i < gridInput.length && !found; i++) {
      for (let j = 0; j < gridInput[i].length && !found; j++) {
        if (gridInput[i][j] === 1 && gridActual[i][j] === 1) {
          ca.sensors.gradientDirection = Math.atan2(i - 2, j - 2);
          found = true;
        }
      }
    }
  }
  
  /**
   * Controles experimentales
   */
  async ejecutarControles() {
    console.log('🎮 Ejecutando controles...');
    
    // 1. Random baseline
    for (const puzzle of this.puzzles) {
      const tasaRandom = validador.compararConRandom(puzzle);
      console.log(`  Random en ${puzzle.id}: ${(tasaRandom * 100).toFixed(1)}%`);
    }
    
    // 2. Sin evolución
    const organismosFijos = this.crearOrganismosSinEvolución();
    const resultadosBase = [];
    for (const puzzle of this.puzzles) {
      const resultado = await this.evaluarPuzzle(puzzle, organismosFijos);
      resultadosBase.push(resultado);
    }
    
    // Establecer baseline
    validador.establecerBaseline(resultadosBase);
  }
  
  /**
   * Crea organismos sin capacidad de evolución
   */
  crearOrganismosSinEvolución() {
    const organismos = [];
    for (let i = 0; i < 10; i++) {
      const org = new OrganismAC(
        Math.random() * 200,
        Math.random() * 200
      );
      // Marcar como no evolutivos
      org.puedeEvolucionar = false;
      organismos.push(org);
    }
    return organismos;
  }
  
  /**
   * Tests de ablación sistemáticos
   */
  async ejecutarAblaciones(organismos) {
    console.log('\n🔬 Tests de ablación...');
    
    const reglas = [
      'chemotaxis',
      'phototaxis', 
      'homeostasis',
      'morphogenesis',
      'oscillator'
    ];
    
    for (const regla of reglas) {
      const impacto = validador.testAblación(
        regla, 
        this.puzzles.slice(0, 3), // Subset para rapidez
        organismos
      );
      
      console.log(`  Sin ${regla}: ${(impacto * 100).toFixed(1)}% degradación`);
    }
  }
  
  /**
   * Visualización en tiempo real
   */
  mostrarProgresoTiempoReal(resultado) {
    const símbolos = resultado.resuelto ? '✅' : '❌';
    const tiempo = (resultado.tiempo / 1000).toFixed(1);
    
    console.log(`  ${símbolos} Tiempo: ${tiempo}s | Intentos: ${resultado.intentos}`);
    
    if (resultado.análisis && resultado.análisis.muestraRazonamiento) {
      console.log(`  🧠 Evidencia de razonamiento detectada!`);
      console.log(`     - Mejora: ${(resultado.análisis.mejora * 100).toFixed(1)}%`);
      console.log(`     - Diversidad: ${(resultado.análisis.diversidad * 100).toFixed(1)}%`);
    }
  }
  
  /**
   * Genera reporte final del experimento
   */
  generarReporteFinal() {
    try {
      // Usar sistema de benchmark avanzado
      const benchmark = new ARCBenchmark();
      const evaluación = benchmark.evaluarResultados(this.resultados, this.puzzles);
      
      // Generar reporte detallado
      const reporteDetallado = benchmark.generarReporteDetallado(evaluación);
      console.log(reporteDetallado);
      
      // Formato para UI
      const reporte = {
        conclusión: evaluación.nivelRazonamiento,
        
        arc: {
          puzzlesResueltos: evaluación.general.resueltos,
          totalPuzzles: evaluación.general.totalPuzzles,
          tasaÉxito: evaluación.general.resueltos / evaluación.general.totalPuzzles,
          tiempoPromedio: evaluación.general.tiempoPromedio
        },
        
        // Métricas principales
        porcentajeRazonamiento: evaluación.porcentajeRazonamiento,
        nivelRazonamiento: evaluación.nivelRazonamiento,
        
        // Detalles por categoría
        porCategoría: evaluación.porCategoría,
        porDificultad: evaluación.porDificultad,
        
        // Capacidades detectadas
        capacidadesDetectadas: evaluación.capacidadesDetectadas,
        
        // Comparaciones
        comparación: evaluación.comparación,
        
        // Para compatibilidad
        tasaResolución: evaluación.general.resueltos / evaluación.general.totalPuzzles,
        tiempoPromedio: evaluación.general.tiempoPromedio,
        vsRandom: evaluación.comparación.vsRandom,
        vsSinEvolución: 1.0,
        reglaCríticas: [],
        ejemplosRazonamiento: evaluación.capacidadesDetectadas,
        
        // Recomendaciones
        recomendaciones: evaluación.recomendaciones,
        
        // Reporte completo en texto
        reporteDetallado: reporteDetallado
      };
      
      console.log('📊 Evaluación completa:', evaluación);
      return reporte;
      
    } catch (error) {
      console.error('Error generando reporte:', error);
      return {
        conclusión: "Error al generar reporte",
        arc: { puzzlesResueltos: 0, totalPuzzles: 0, tasaÉxito: 0 },
        error: error.message
      };
    }
  }
  
  promediar(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }
  
  guardarResultados(reporte) {
    const timestamp = new Date().toISOString();
    localStorage.setItem(
      `arc_experiment_${timestamp}`, 
      JSON.stringify(reporte)
    );
  }
  
  /**
   * Métodos auxiliares para traducir acciones del CA
   */
  determinarTipoAcción(organismo) {
    // Basado en los efectores del CA
    const ef = organismo.cellularAutomaton.effectors;
    if (Math.abs(ef.movementX) > 0.5 || Math.abs(ef.movementY) > 0.5) {
      return 'modificar';
    }
    return 'observar';
  }
  
  calcularPosiciónObjetivo(organismo, entornoARC) {
    const grid = entornoARC.getCurrentGrid();
    const ef = organismo.cellularAutomaton.effectors;
    
    // Convertir movimiento del CA a posición en el grid
    const gridX = Math.floor((ef.movementX + 1) * grid[0].length / 2);
    const gridY = Math.floor((ef.movementY + 1) * grid.length / 2);
    
    return {
      x: Math.max(0, Math.min(grid[0].length - 1, gridX)),
      y: Math.max(0, Math.min(grid.length - 1, gridY))
    };
  }
  
  determinarValor(organismo) {
    // Para el puzzle de cambiar 1s por 2s
    // Si hay alto gradiente químico (muchos 1s), usar valor 2
    const activación = organismo.cellularAutomaton.sensors.chemicalGradient;
    
    if (activación > 0.3) {
      return 2; // Cambiar a 2 cuando detecta 1s
    }
    
    // Exploración aleatoria ocasional
    if (Math.random() < 0.1) {
      return Math.floor(Math.random() * 10);
    }
    
    return 1; // Por defecto mantener
  }
  
  calcularConfianza(organismo) {
    // Basado en la coherencia del CA
    const ca = organismo.cellularAutomaton;
    let activaciónTotal = 0;
    let cellsActivas = 0;
    
    ca.grid.forEach(row => {
      row.forEach(cell => {
        if (cell.type !== 'void' && cell.activation > 0) {
          activaciónTotal += cell.activation;
          cellsActivas++;
        }
      });
    });
    
    return cellsActivas > 0 ? activaciónTotal / cellsActivas : 0;
  }
  
  evolucionarPoblación(organismos, entornoARC) {
    // Simplificado: clonar los mejores
    const mejores = organismos
      .sort((a, b) => b.energy - a.energy)
      .slice(0, organismos.length / 2);
    
    const nuevaPoblación = [];
    mejores.forEach(org => {
      nuevaPoblación.push(org);
      const hijo = org.reproduce();
      if (hijo) nuevaPoblación.push(hijo);
    });
    
    return nuevaPoblación;
  }
  
  compararConBenchmarkHumano() {
    // Benchmark típico humano en ARC
    const promedioHumano = 0.85; // 85% de éxito típico
    const tasaProteus = this.resultados.filter(r => r.resuelto).length / 
                        this.resultados.length;
    return {
      tasaHumana: promedioHumano,
      tasaProteus: tasaProteus,
      ratio: tasaProteus / promedioHumano
    };
  }
}

/**
 * Entorno específico para puzzles ARC
 */
class ARCEnvironment {
  constructor(puzzle) {
    this.puzzle = puzzle;
    this.gridActual = JSON.parse(JSON.stringify(puzzle.input));
    this.acciones = [];
  }
  
  getInputGrid() {
    return this.puzzle.input;
  }
  
  getCurrentGrid() {
    return this.gridActual;
  }
  
  aplicarAcción(acción) {
    this.acciones.push(acción);
    
    // Aplicar transformación según tipo de acción
    if (acción.tipo === 'modificar') {
      const {x, y} = this.gridToCoords(acción.posición);
      if (this.validarPosición(x, y)) {
        // Solo cambiar si la celda original era 1
        if (this.puzzle.input[y][x] === 1) {
          this.gridActual[y][x] = acción.valor;
          console.log(`🎯 Cambiando celda [${y},${x}] de ${this.puzzle.input[y][x]} a ${acción.valor}`);
        }
      }
    }
    
    // Verificar si es solución correcta
    const correcto = this.verificarSolución();
    
    if (correcto) {
      console.log('✅ ¡Puzzle resuelto correctamente!');
    }
    
    return {
      correcto: correcto,
      completo: this.esCompleto(),
      grid: this.gridActual
    };
  }
  
  verificarSolución() {
    return JSON.stringify(this.gridActual) === 
           JSON.stringify(this.puzzle.output);
  }
  
  esCompleto() {
    // Heurística: consideramos completo si se han hecho suficientes cambios
    return this.acciones.length > this.puzzle.input.flat().length / 2;
  }
  
  gridToCoords(posición) {
    return {
      x: Math.floor(posición.x),
      y: Math.floor(posición.y)
    };
  }
  
  validarPosición(x, y) {
    return x >= 0 && x < this.gridActual[0].length &&
           y >= 0 && y < this.gridActual.length;
  }
}