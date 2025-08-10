/**
 * Generador de GIFs para visualizar el proceso de razonamiento
 * Crea animaciones paso a paso del solver
 */

export class GifGenerator {
  constructor() {
    this.frames = [];
    this.currentFrame = null;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.cellSize = 20;
    this.padding = 10;
  }
  
  /**
   * Genera un GIF del proceso de resolución
   */
  async generarGifRazonamiento(pasos) {
    this.frames = [];
    
    for (const paso of pasos) {
      const frame = await this.crearFrame(paso);
      this.frames.push(frame);
    }
    
    return this.crearGifAnimado();
  }
  
  /**
   * Crea un frame individual
   */
  async crearFrame(paso) {
    // Calcular tamaño del canvas
    const grid = paso.datos.gridDespues || paso.datos.output || paso.datos.input;
    const width = grid[0].length * this.cellSize + 2 * this.padding;
    const height = grid.length * this.cellSize + 2 * this.padding + 60; // Espacio para título
    
    this.canvas.width = width;
    this.canvas.height = height;
    
    // Limpiar canvas
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.fillRect(0, 0, width, height);
    
    // Dibujar título
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '16px monospace';
    this.ctx.fillText(paso.titulo, this.padding, 25);
    
    // Dibujar descripción
    this.ctx.font = '12px monospace';
    this.ctx.fillStyle = '#aaaaaa';
    this.ctx.fillText(paso.descripcion, this.padding, 45);
    
    // Dibujar grid
    this.dibujarGrid(grid, this.padding, 60, paso.datos.cambios);
    
    // Convertir a imagen
    return this.canvas.toDataURL('image/png');
  }
  
  /**
   * Dibuja un grid en el canvas
   */
  dibujarGrid(grid, offsetX, offsetY, cambios = []) {
    const colores = [
      '#000000', // 0 - Negro
      '#0074D9', // 1 - Azul
      '#2ECC40', // 2 - Verde
      '#FFDC00', // 3 - Amarillo
      '#FF4136', // 4 - Rojo
      '#B10DC9', // 5 - Púrpura
      '#FF851B', // 6 - Naranja
      '#7FDBFF', // 7 - Celeste
      '#85144b', // 8 - Marrón
      '#F012BE'  // 9 - Rosa
    ];
    
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        const x = offsetX + j * this.cellSize;
        const y = offsetY + i * this.cellSize;
        const valor = grid[i][j];
        
        // Color de fondo
        this.ctx.fillStyle = colores[valor] || '#000000';
        this.ctx.fillRect(x, y, this.cellSize - 1, this.cellSize - 1);
        
        // Resaltar cambios
        const cambio = cambios.find(c => c.x === i && c.y === j);
        if (cambio) {
          this.ctx.strokeStyle = '#FF851B';
          this.ctx.lineWidth = 2;
          this.ctx.strokeRect(x - 1, y - 1, this.cellSize + 1, this.cellSize + 1);
        }
      }
    }
  }
  
  /**
   * Crea el GIF animado (simulado con data URLs)
   */
  crearGifAnimado() {
    // Por ahora devolvemos un array de frames
    // En producción, usaríamos una librería como gif.js
    return {
      frames: this.frames,
      duracion: 1000, // ms por frame
      loop: true
    };
  }
  
  /**
   * Genera un diagrama de flujo del razonamiento
   */
  generarDiagramaFlujo(reglas) {
    const width = 800;
    const height = 400;
    this.canvas.width = width;
    this.canvas.height = height;
    
    // Limpiar
    this.ctx.fillStyle = '#1a1a1a';
    this.ctx.fillRect(0, 0, width, height);
    
    // Título
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = 'bold 20px monospace';
    this.ctx.fillText('Flujo de Razonamiento ARC', width/2 - 150, 30);
    
    // Dibujar flujo
    const pasos = [
      { x: 100, y: 100, texto: 'Entrada', color: '#0074D9' },
      { x: 300, y: 100, texto: 'Análisis', color: '#FF851B' },
      { x: 500, y: 100, texto: 'Regla', color: '#B10DC9' },
      { x: 700, y: 100, texto: 'Salida', color: '#2ECC40' }
    ];
    
    // Conectar pasos
    this.ctx.strokeStyle = '#666666';
    this.ctx.lineWidth = 2;
    for (let i = 0; i < pasos.length - 1; i++) {
      this.dibujarFlecha(
        pasos[i].x + 40, pasos[i].y,
        pasos[i + 1].x - 40, pasos[i + 1].y
      );
    }
    
    // Dibujar nodos
    pasos.forEach(paso => {
      // Círculo
      this.ctx.fillStyle = paso.color;
      this.ctx.beginPath();
      this.ctx.arc(paso.x, paso.y, 30, 0, 2 * Math.PI);
      this.ctx.fill();
      
      // Texto
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = '14px monospace';
      this.ctx.fillText(paso.texto, paso.x - 25, paso.y + 50);
    });
    
    // Agregar reglas detectadas
    if (reglas && reglas.length > 0) {
      this.ctx.font = '12px monospace';
      this.ctx.fillStyle = '#aaaaaa';
      let y = 200;
      reglas.forEach(regla => {
        this.ctx.fillText(`• ${regla.tipo}: ${regla.descripcion}`, 100, y);
        y += 20;
      });
    }
    
    return this.canvas.toDataURL('image/png');
  }
  
  /**
   * Dibuja una flecha
   */
  dibujarFlecha(x1, y1, x2, y2) {
    const headlen = 10;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const angle = Math.atan2(dy, dx);
    
    // Línea
    this.ctx.beginPath();
    this.ctx.moveTo(x1, y1);
    this.ctx.lineTo(x2, y2);
    this.ctx.stroke();
    
    // Punta de flecha
    this.ctx.beginPath();
    this.ctx.moveTo(x2, y2);
    this.ctx.lineTo(
      x2 - headlen * Math.cos(angle - Math.PI / 6),
      y2 - headlen * Math.sin(angle - Math.PI / 6)
    );
    this.ctx.moveTo(x2, y2);
    this.ctx.lineTo(
      x2 - headlen * Math.cos(angle + Math.PI / 6),
      y2 - headlen * Math.sin(angle + Math.PI / 6)
    );
    this.ctx.stroke();
  }
  
  /**
   * Exporta como imagen PNG
   */
  exportarComoPNG(nombreArchivo = 'arc-reasoning.png') {
    const link = document.createElement('a');
    link.download = nombreArchivo;
    link.href = this.canvas.toDataURL();
    link.click();
  }
}