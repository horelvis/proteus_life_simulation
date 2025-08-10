/**
 * Solucionador ARC basado en los principios del Autómata Celular
 * Usa el core de PROTEUS-AC pero adaptado para puzzles
 */

export class ARCSolver {
  constructor() {
    this.gridSize = 30; // Máximo tamaño ARC
    this.iterations = 0;
    this.maxIterations = 100;
    
    // Estados del CA para representar el puzzle
    this.inputGrid = null;
    this.workingGrid = null;
    this.outputGrid = null;
    
    // Patrones aprendidos de los ejemplos
    this.transformationRules = [];
    
    // Registro de razonamiento para transparencia
    this.razonamientoPasos = [];
  }
  
  /**
   * Analiza ejemplos de entrenamiento para extraer reglas
   */
  aprender(ejemplosEntrenamiento) {
    this.transformationRules = [];
    
    ejemplosEntrenamiento.forEach((ejemplo, idx) => {
      console.log(`📚 Analizando ejemplo ${idx + 1}...`);
      console.log(`   Input:`, ejemplo.input);
      console.log(`   Output:`, ejemplo.output);
      
      const regla = this.extraerRegla(ejemplo.input, ejemplo.output);
      if (regla) {
        this.transformationRules.push(regla);
        console.log(`✓ Regla detectada: ${regla.tipo}`);
        
        // Verificar que la regla funciona con el ejemplo
        const test = regla.aplicar(ejemplo.input);
        const funcionaBien = JSON.stringify(test) === JSON.stringify(ejemplo.output);
        console.log(`   ¿Regla funciona con ejemplo?: ${funcionaBien ? '✅' : '❌'}`);
        if (!funcionaBien) {
          console.log(`   Resultado de aplicar regla:`, test);
        }
      }
    });
    
    return this.transformationRules.length > 0;
  }
  
  /**
   * Extrae regla de transformación comparando input/output
   */
  extraerRegla(input, output) {
    // Orden de verificación mejorado para evitar falsos positivos
    // Más específicos primero, más generales al final
    
    // 1. Verificar transformaciones de tamaño primero
    if (this.esReplicacion(input, output)) {
      return {
        tipo: 'pattern_replication',
        aplicar: (grid) => this.aplicarReplicacion(grid)
      };
    }
    
    // 2. Verificar conteo (output 1x1)
    if (this.esConteo(input, output)) {
      return {
        tipo: 'counting',
        aplicar: (grid) => this.aplicarConteo(grid)
      };
    }
    
    // 3. Verificar detección de simetría (output 1x1)
    if (this.esDeteccionSimetria(input, output)) {
      return {
        tipo: 'symmetry_detection',
        aplicar: (grid) => this.aplicarDeteccionSimetria(grid)
      };
    }
    
    // 4. Verificar extracción de patrones
    if (this.esExtraccionPatron(input, output)) {
      return {
        tipo: 'pattern_extraction',
        aplicar: (grid) => this.aplicarExtraccionPatron(grid)
      };
    }
    
    // 5. Verificar transformaciones geométricas
    if (this.esRotacion(input, output)) {
      return {
        tipo: 'rotation',
        aplicar: (grid) => this.aplicarRotacion(grid)
      };
    }
    
    if (this.esReflexion(input, output)) {
      return {
        tipo: 'reflection',
        aplicar: (grid) => this.aplicarReflexion(grid)
      };
    }
    
    if (this.esGravedad(input, output)) {
      return {
        tipo: 'gravity',
        aplicar: (grid) => this.aplicarGravedad(grid)
      };
    }
    
    // 6. Verificar transformaciones de contenido
    if (this.esRellenarForma(input, output)) {
      return {
        tipo: 'fill_shape',
        aplicar: (grid) => this.aplicarRellenarForma(grid)
      };
    }
    
    if (this.esDibujarLinea(input, output)) {
      return {
        tipo: 'line_drawing',
        aplicar: (grid) => this.aplicarDibujarLinea(grid)
      };
    }
    
    // 7. Color mapping como último recurso (más general)
    const colorMap = this.detectarMapeoColor(input, output);
    if (colorMap && this.esMapeoColorReal(input, output, colorMap)) {
      return {
        tipo: 'color_mapping',
        aplicar: (grid) => this.aplicarMapeoColor(grid, colorMap)
      };
    }
    
    return null;
  }
  
  /**
   * Detecta mapeo simple de colores
   */
  detectarMapeoColor(input, output) {
    if (input.length !== output.length || input[0].length !== output[0].length) {
      return null;
    }
    
    const colorMap = {};
    
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        const colorIn = input[i][j];
        const colorOut = output[i][j];
        
        if (colorIn !== 0) { // Ignorar fondo
          if (colorMap[colorIn] === undefined) {
            colorMap[colorIn] = colorOut;
          } else if (colorMap[colorIn] !== colorOut) {
            return null; // No es mapeo consistente
          }
        }
      }
    }
    
    return Object.keys(colorMap).length > 0 ? colorMap : null;
  }
  
  /**
   * Detecta si es replicación 3x3
   */
  esReplicacion(input, output) {
    if (input.length * 3 !== output.length || input[0].length * 3 !== output[0].length) {
      return false;
    }
    
    // Verificar que cada celda del input se replica 3x3 en output
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        const valor = input[i][j];
        // Verificar bloque 3x3 correspondiente
        for (let di = 0; di < 3; di++) {
          for (let dj = 0; dj < 3; dj++) {
            if (output[i*3 + di][j*3 + dj] !== valor) {
              return false;
            }
          }
        }
      }
    }
    
    return true;
  }
  
  /**
   * Detecta si los objetos caen por gravedad
   */
  esGravedad(input, output) {
    // Primero verificar que tengan el mismo tamaño
    if (input.length !== output.length || input[0].length !== output[0].length) {
      return false;
    }
    
    // Verificar cada columna
    for (let j = 0; j < input[0].length; j++) {
      // Recolectar elementos no-cero de la columna en input
      const elementosInput = [];
      for (let i = 0; i < input.length; i++) {
        if (input[i][j] !== 0) {
          elementosInput.push(input[i][j]);
        }
      }
      
      // Recolectar elementos no-cero de la columna en output
      const elementosOutput = [];
      for (let i = 0; i < output.length; i++) {
        if (output[i][j] !== 0) {
          elementosOutput.push(output[i][j]);
        }
      }
      
      // Deben tener los mismos elementos
      if (elementosInput.length !== elementosOutput.length) {
        return false;
      }
      
      // Los elementos deben estar compactados al fondo
      let esperadoIdx = output.length - elementosOutput.length;
      for (let i = 0; i < output.length; i++) {
        if (i < esperadoIdx && output[i][j] !== 0) {
          return false; // Hay elemento donde no debería
        }
        if (i >= esperadoIdx && output[i][j] === 0) {
          return false; // No hay elemento donde debería
        }
      }
    }
    
    return true;
  }
  
  /**
   * Aplica transformación detectada
   */
  resolver(testInput) {
    console.log('🧩 Resolviendo puzzle...');
    this.razonamientoPasos = [];
    
    if (this.transformationRules.length === 0) {
      console.log('❌ No hay reglas aprendidas');
      return testInput;
    }
    
    // Aplicar la primera regla que funcione
    for (const regla of this.transformationRules) {
      try {
        console.log(`🔄 Aplicando regla: ${regla.tipo}`);
        
        // Registrar paso de razonamiento
        this.razonamientoPasos.push({
          tipo: 'aplicar_regla',
          regla: regla.tipo,
          gridAntes: JSON.parse(JSON.stringify(testInput))
        });
        
        const resultado = regla.aplicar(testInput);
        if (resultado) {
          this.razonamientoPasos.push({
            tipo: 'resultado',
            gridDespues: resultado
          });
          return resultado;
        }
      } catch (e) {
        console.log(`⚠️ Error aplicando regla ${regla.tipo}:`, e);
      }
    }
    
    return testInput;
  }
  
  /**
   * Implementaciones de transformaciones
   */
  aplicarMapeoColor(grid, colorMap) {
    const result = [];
    for (let i = 0; i < grid.length; i++) {
      result[i] = [];
      for (let j = 0; j < grid[i].length; j++) {
        const color = grid[i][j];
        result[i][j] = colorMap[color] !== undefined ? colorMap[color] : color;
      }
    }
    return result;
  }
  
  aplicarReplicacion(grid) {
    const result = [];
    for (let i = 0; i < grid.length * 3; i++) {
      result[i] = [];
      for (let j = 0; j < grid[0].length * 3; j++) {
        const origI = Math.floor(i / 3);
        const origJ = Math.floor(j / 3);
        result[i][j] = grid[origI][origJ];
      }
    }
    return result;
  }
  
  aplicarGravedad(grid) {
    const result = Array(grid.length).fill(null).map(() => Array(grid[0].length).fill(0));
    
    // Por cada columna, mover todo hacia abajo
    for (let j = 0; j < grid[0].length; j++) {
      const columna = [];
      
      // Recolectar no-ceros
      for (let i = 0; i < grid.length; i++) {
        if (grid[i][j] !== 0) {
          columna.push(grid[i][j]);
        }
      }
      
      // Colocar en el fondo
      let idx = grid.length - 1;
      for (let k = columna.length - 1; k >= 0; k--) {
        result[idx--][j] = columna[k];
      }
    }
    
    return result;
  }
  
  aplicarReflexion(grid) {
    const result = [];
    for (let i = 0; i < grid.length; i++) {
      result[i] = [];
      for (let j = 0; j < grid[i].length; j++) {
        // Reflexión horizontal
        result[i][j] = grid[i][grid[i].length - 1 - j];
      }
    }
    return result;
  }
  
  aplicarConteo(grid) {
    let count = 0;
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j] !== 0) {
          count++;
        }
      }
    }
    return [[count]];
  }
  
  /**
   * Utilidades
   */
  obtenerColores(grid) {
    const colores = {};
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j] !== 0) {
          colores[grid[i][j]] = (colores[grid[i][j]] || 0) + 1;
        }
      }
    }
    return colores;
  }
  
  mismosColores(colores1, colores2) {
    const keys1 = Object.keys(colores1).sort();
    const keys2 = Object.keys(colores2).sort();
    
    if (keys1.length !== keys2.length) return false;
    
    for (let i = 0; i < keys1.length; i++) {
      if (keys1[i] !== keys2[i] || colores1[keys1[i]] !== colores2[keys2[i]]) {
        return false;
      }
    }
    
    return true;
  }
  
  esReflexion(input, output) {
    if (input.length !== output.length || input[0].length !== output[0].length) {
      return false;
    }
    
    // Verificar reflexión horizontal
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        if (input[i][j] !== output[i][input[i].length - 1 - j]) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  esConteo(input, output) {
    // Output debe ser 1x1
    if (output.length !== 1 || output[0].length !== 1) {
      return false;
    }
    
    // Contar no-ceros en input
    let count = 0;
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        if (input[i][j] !== 0) {
          count++;
        }
      }
    }
    
    return output[0][0] === count;
  }
  
  /**
   * Detecta si es rellenar forma cerrada
   */
  esRellenarForma(input, output) {
    // Verificar que tengan el mismo tamaño
    if (input.length !== output.length || input[0].length !== output[0].length) {
      return false;
    }
    
    // Verificar que sea al menos 3x3
    if (input.length < 3 || input[0].length < 3) {
      return false;
    }
    
    // Buscar formas cerradas en input que se rellenan en output
    let encontradoRelleno = false;
    for (let i = 1; i < input.length - 1; i++) {
      for (let j = 1; j < input[i].length - 1; j++) {
        if (input[i][j] === 0 && output[i][j] !== 0) {
          // Verificar si está rodeado por no-ceros
          const rodeado = this.estaRodeado(input, i, j);
          if (rodeado) {
            encontradoRelleno = true;
          }
        }
      }
    }
    return encontradoRelleno;
  }
  
  estaRodeado(grid, x, y) {
    // Verificar si la posición está dentro de una forma cerrada
    // Simplificado: verificar vecinos
    const vecinos = [
      [-1, 0], [1, 0], [0, -1], [0, 1]
    ];
    
    let bordesCerrados = 0;
    for (const [dx, dy] of vecinos) {
      const nx = x + dx;
      const ny = y + dy;
      // Verificar límites
      if (nx >= 0 && nx < grid.length && ny >= 0 && ny < grid[0].length) {
        if (grid[nx][ny] !== 0) {
          bordesCerrados++;
        }
      }
    }
    
    return bordesCerrados >= 3; // Al menos 3 lados cerrados
  }
  
  aplicarRellenarForma(grid) {
    const result = JSON.parse(JSON.stringify(grid));
    
    // Encontrar el color más común (para rellenar)
    const colores = {};
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j] !== 0) {
          colores[grid[i][j]] = (colores[grid[i][j]] || 0) + 1;
        }
      }
    }
    
    const colorRelleno = 3; // Color típico de relleno en ARC
    
    // Rellenar espacios cerrados
    for (let i = 1; i < grid.length - 1; i++) {
      for (let j = 1; j < grid[i].length - 1; j++) {
        if (grid[i][j] === 0 && this.estaRodeado(grid, i, j)) {
          result[i][j] = colorRelleno;
        }
      }
    }
    
    return result;
  }
  
  /**
   * Detecta si es verificación de simetría
   */
  esDeteccionSimetria(input, output) {
    return output.length === 1 && output[0].length === 1 && 
           (output[0][0] === 0 || output[0][0] === 1);
  }
  
  aplicarDeteccionSimetria(grid) {
    // Verificar simetría horizontal y vertical
    const esSimetrico = this.verificarSimetria(grid);
    return [[esSimetrico ? 1 : 0]];
  }
  
  verificarSimetria(grid) {
    const n = grid.length;
    const m = grid[0].length;
    
    // Simetría horizontal
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < Math.floor(m / 2); j++) {
        if (grid[i][j] !== grid[i][m - 1 - j]) {
          return false;
        }
      }
    }
    
    // Simetría vertical
    for (let i = 0; i < Math.floor(n / 2); i++) {
      for (let j = 0; j < m; j++) {
        if (grid[i][j] !== grid[n - 1 - i][j]) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  /**
   * Detecta rotación 90 grados
   */
  esRotacion(input, output) {
    // Primero probar rotación 90° antihorario
    if (input.length === output[0].length && input[0].length === output.length) {
      let esRotacionAntihorario = true;
      for (let i = 0; i < input.length && esRotacionAntihorario; i++) {
        for (let j = 0; j < input[i].length && esRotacionAntihorario; j++) {
          if (input[i][j] !== output[input[0].length - 1 - j][i]) {
            esRotacionAntihorario = false;
          }
        }
      }
      if (esRotacionAntihorario) return true;
    }
    
    // También probar rotación 90° horario
    if (input.length === output[0].length && input[0].length === output.length) {
      let esRotacionHorario = true;
      for (let i = 0; i < input.length && esRotacionHorario; i++) {
        for (let j = 0; j < input[i].length && esRotacionHorario; j++) {
          if (input[i][j] !== output[j][input.length - 1 - i]) {
            esRotacionHorario = false;
          }
        }
      }
      return esRotacionHorario;
    }
    
    return false;
  }
  
  aplicarRotacion(grid) {
    const n = grid.length;
    const m = grid[0].length;
    const result = Array(m).fill(null).map(() => Array(n).fill(0));
    
    // Rotar 90° sentido antihorario (como en el ejemplo)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        result[m - 1 - j][i] = grid[i][j];
      }
    }
    
    return result;
  }
  
  /**
   * Detecta extracción de patrones no-cero
   */
  esExtraccionPatron(input, output) {
    // Output debe ser más pequeño y contener solo los no-ceros
    if (output.length >= input.length || output[0].length >= input[0].length) {
      return false;
    }
    
    // Verificar que output contenga solo elementos no-cero de input
    for (let i = 0; i < output.length; i++) {
      for (let j = 0; j < output[i].length; j++) {
        if (output[i][j] === 0) {
          let found = false;
          for (let ii = 0; ii < input.length; ii++) {
            for (let jj = 0; jj < input[ii].length; jj++) {
              if (input[ii][jj] === 0 && ii < output.length && jj < output[i].length) {
                found = true;
                break;
              }
            }
          }
          if (!found) return false;
        }
      }
    }
    
    return true;
  }
  
  aplicarExtraccionPatron(grid) {
    // Encontrar boundingbox de elementos no-cero
    let minI = grid.length, maxI = -1;
    let minJ = grid[0].length, maxJ = -1;
    
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j] !== 0) {
          minI = Math.min(minI, i);
          maxI = Math.max(maxI, i);
          minJ = Math.min(minJ, j);
          maxJ = Math.max(maxJ, j);
        }
      }
    }
    
    if (maxI === -1) return [[0]]; // Grid vacío
    
    // Extraer subgrid
    const result = [];
    for (let i = minI; i <= maxI; i++) {
      result.push([]);
      for (let j = minJ; j <= maxJ; j++) {
        result[result.length - 1].push(grid[i][j]);
      }
    }
    
    return result;
  }
  
  /**
   * Detecta si es dibujar línea entre puntos
   */
  esDibujarLinea(input, output) {
    // Contar puntos en input
    const puntosInput = this.contarPuntos(input);
    const puntosOutput = this.contarPuntos(output);
    
    // Debe haber más puntos en output (la línea)
    return puntosInput.length === 2 && puntosOutput.length > 2;
  }
  
  contarPuntos(grid) {
    const puntos = [];
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (grid[i][j] !== 0) {
          puntos.push({x: i, y: j, color: grid[i][j]});
        }
      }
    }
    return puntos;
  }
  
  aplicarDibujarLinea(grid) {
    const result = JSON.parse(JSON.stringify(grid));
    const puntos = this.contarPuntos(grid);
    
    if (puntos.length !== 2) return grid;
    
    // Dibujar línea entre los dos puntos
    const [p1, p2] = puntos;
    const color = p1.color;
    
    // Línea diagonal simple
    const dx = p2.x > p1.x ? 1 : -1;
    const dy = p2.y > p1.y ? 1 : -1;
    
    let x = p1.x;
    let y = p1.y;
    
    while (x !== p2.x || y !== p2.y) {
      result[x][y] = color;
      if (x !== p2.x) x += dx;
      if (y !== p2.y) y += dy;
    }
    result[p2.x][p2.y] = color;
    
    return result;
  }
  
  /**
   * Verifica si realmente es un mapeo de color
   * (no solo una coincidencia parcial)
   */
  esMapeoColorReal(input, output, colorMap) {
    // Verificar que TODOS los cambios se explican por el mapeo
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        const colorIn = input[i][j];
        const colorOut = output[i][j];
        
        if (colorIn === 0 && colorOut === 0) continue; // Fondo ok
        
        // Si hay un color en output que no estaba en input
        // y no es parte del mapeo, no es color mapping puro
        if (colorIn === 0 && colorOut !== 0) {
          return false;
        }
        
        // Si el mapeo no explica el cambio
        if (colorIn !== 0 && colorMap[colorIn] !== colorOut) {
          return false;
        }
      }
    }
    
    // Verificar que al menos hay un cambio real
    let hayaCambio = false;
    for (const [colorIn, colorOut] of Object.entries(colorMap)) {
      if (colorIn !== colorOut) {
        hayaCambio = true;
        break;
      }
    }
    
    return hayaCambio;
  }
  
  /**
   * Obtiene la regla principal detectada
   */
  detectarReglaPrincipal(ejemplos) {
    if (!ejemplos || ejemplos.length === 0) return null;
    
    // Analizar primer ejemplo
    const regla = this.extraerRegla(ejemplos[0].input, ejemplos[0].output);
    
    if (regla) {
      // Extraer parámetros adicionales
      regla.parametros = this.extraerParametrosRegla(regla.tipo, ejemplos);
    }
    
    return regla;
  }
  
  /**
   * Extrae parámetros específicos de cada tipo de regla
   */
  extraerParametrosRegla(tipo, ejemplos) {
    switch (tipo) {
      case 'color_mapping':
        const colorMap = this.detectarMapeoColor(ejemplos[0].input, ejemplos[0].output);
        return { mapeo: colorMap };
        
      case 'gravity':
        return { direccion: 'abajo' };
        
      case 'reflection':
        return { eje: 'horizontal' };
        
      case 'rotation':
        return { grados: 90, sentido: 'antihorario' };
        
      default:
        return {};
    }
  }
  
  /**
   * Genera pasos detallados de aplicación de una regla
   */
  obtenerPasosAplicacion(input, regla) {
    if (!regla) return [];
    
    const pasos = [];
    
    switch (regla.tipo) {
      case 'color_mapping':
        // Mostrar cada cambio de color
        const gridCopia = JSON.parse(JSON.stringify(input));
        const cambios = [];
        
        for (let i = 0; i < input.length; i++) {
          for (let j = 0; j < input[i].length; j++) {
            const colorOriginal = input[i][j];
            if (colorOriginal !== 0 && regla.parametros.mapeo[colorOriginal]) {
              cambios.push({ x: i, y: j, de: colorOriginal, a: regla.parametros.mapeo[colorOriginal] });
            }
          }
        }
        
        // Crear un paso por cada grupo de cambios
        const cambiosPorColor = {};
        cambios.forEach(c => {
          if (!cambiosPorColor[c.de]) cambiosPorColor[c.de] = [];
          cambiosPorColor[c.de].push(c);
        });
        
        Object.entries(cambiosPorColor).forEach(([color, listaCambios]) => {
          const gridAntes = JSON.parse(JSON.stringify(gridCopia));
          listaCambios.forEach(c => {
            gridCopia[c.x][c.y] = c.a;
          });
          
          pasos.push({
            descripcion: `Transformando color ${color} → ${listaCambios[0].a}`,
            gridAntes: gridAntes,
            gridDespues: JSON.parse(JSON.stringify(gridCopia)),
            cambios: listaCambios
          });
        });
        break;
        
      case 'gravity':
        // Mostrar caída columna por columna
        const gridGravedad = JSON.parse(JSON.stringify(input));
        
        for (let j = 0; j < input[0].length; j++) {
          const gridAntes = JSON.parse(JSON.stringify(gridGravedad));
          const elementos = [];
          
          // Recolectar elementos de la columna
          for (let i = 0; i < input.length; i++) {
            if (gridGravedad[i][j] !== 0) {
              elementos.push(gridGravedad[i][j]);
              gridGravedad[i][j] = 0;
            }
          }
          
          // Colocar al fondo
          let idx = gridGravedad.length - 1;
          for (let k = elementos.length - 1; k >= 0; k--) {
            gridGravedad[idx--][j] = elementos[k];
          }
          
          if (elementos.length > 0) {
            pasos.push({
              descripcion: `Aplicando gravedad en columna ${j + 1}`,
              gridAntes: gridAntes,
              gridDespues: JSON.parse(JSON.stringify(gridGravedad)),
              cambios: []
            });
          }
        }
        break;
        
      default:
        // Para otras reglas, mostrar solo antes y después
        pasos.push({
          descripcion: `Aplicando transformación ${regla.tipo}`,
          gridAntes: input,
          gridDespues: regla.aplicar(input),
          cambios: []
        });
    }
    
    return pasos;
  }
  
  /**
   * Obtiene registro completo del razonamiento
   */
  obtenerRazonamiento() {
    return this.razonamientoPasos;
  }
  
  /**
   * Genera pasos de razonamiento en un formato compatible con la visualización
   */
  generarPasosVisualizacion() {
    const pasos = [];
    
    // Si no hay reglas aprendidas, devolver array vacío
    if (this.transformationRules.length === 0) {
      return pasos;
    }
    
    // Agregar paso para cada regla detectada
    this.transformationRules.forEach((regla, idx) => {
      pasos.push({
        tipo: 'deteccion_regla',
        titulo: `Regla ${idx + 1}: ${regla.tipo}`,
        descripcion: this.obtenerDescripcionRegla(regla.tipo),
        datos: {
          tipo: regla.tipo,
          regla: regla
        }
      });
    });
    
    return pasos;
  }
  
  /**
   * Obtiene descripción legible de una regla
   */
  obtenerDescripcionRegla(tipo) {
    const descripciones = {
      'color_mapping': 'Cada color se transforma en otro color específico',
      'pattern_replication': 'El patrón se replica múltiples veces (3x3)',
      'reflection': 'La imagen se refleja como en un espejo',
      'rotation': 'La imagen se rota 90 grados',
      'counting': 'Se cuenta la cantidad de elementos no vacíos',
      'gravity': 'Los elementos caen hacia abajo por gravedad',
      'symmetry_detection': 'Se detecta si el patrón es simétrico',
      'fill_shape': 'Se rellenan las formas cerradas',
      'pattern_extraction': 'Se extrae un subpatrón específico',
      'line_drawing': 'Se dibujan líneas entre puntos'
    };
    return descripciones[tipo] || 'Transformación detectada';
  }
}