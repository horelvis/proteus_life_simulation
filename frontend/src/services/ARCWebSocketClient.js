/**
 * Cliente WebSocket para comunicación con el backend Python de ARC
 */

import { ARC_WS_URL } from '../config';

export class ARCWebSocketClient {
  constructor(url = null) {
    // Prioridad: parámetro -> localStorage override -> config
    const lsUrl = (() => {
      try { return localStorage.getItem('arc_ws_url'); } catch (_) { return null; }
    })();
    this.url = (url && url.trim()) || (lsUrl && lsUrl.trim()) || ARC_WS_URL;
    this.ws = null;
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.messageHandlers = new Map();
    this.messageQueue = [];
    this.sessionId = null;
  }

  /**
   * Conecta al servidor WebSocket
   */
  async connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`🔌 Conectando a ${this.url}...`);
        this.ws = new WebSocket(this.url);
        let opened = false;
        const timeoutMs = 8000;
        const timeoutId = setTimeout(() => {
          if (!opened) {
            const err = new Error(`Timeout conectando WebSocket en ${this.url}`);
            console.error('⏱️  Timeout WebSocket:', err);
            try { this.ws && this.ws.close(); } catch (_) {}
            this.connected = false;
            reject(err);
          }
        }, timeoutMs);

        this.ws.onopen = () => {
          console.log('✅ Conectado al servidor ARC');
          this.connected = true;
          this.reconnectAttempts = 0;
          opened = true;
          clearTimeout(timeoutId);
          
          // Emitir evento de conexión para componentes que lo esperan
          const connectedHandlers = this.messageHandlers.get('connected') || [];
          connectedHandlers.forEach(handler => {
            try {
              handler({ type: 'connected', message: 'Conectado al servidor ARC' });
            } catch (error) {
              console.error('Error en handler connected:', error);
            }
          });
          
          // Procesar mensajes en cola
          this.processMessageQueue();
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (event) => {
          // Los eventos de error de WebSocket no exponen detalles
          const err = new Error(`Error conectando WebSocket en ${this.url}`);
          console.error('❌ Error WebSocket:', err, event);
          reject(err);
        };

        this.ws.onclose = (ev) => {
          console.log('🔌 Conexión cerrada', ev?.code, ev?.reason || '');
          this.connected = false;
          
          // Emitir evento de desconexión
          const disconnectedHandlers = this.messageHandlers.get('disconnected') || [];
          disconnectedHandlers.forEach(handler => {
            try {
              handler({ type: 'disconnected', message: 'Desconectado del servidor ARC' });
            } catch (error) {
              console.error('Error en handler disconnected:', error);
            }
          });
          
          this.handleDisconnect();
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Desconecta del servidor
   */
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }

  /**
   * Envía un mensaje al servidor
   */
  send(type, data = {}) {
    const message = {
      type,
      ...data,
      timestamp: new Date().toISOString()
    };

    if (this.connected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Agregar a cola si no está conectado
      this.messageQueue.push(message);
      console.log('⏳ Mensaje agregado a cola:', type);
    }
  }

  /**
   * Registra un manejador para un tipo de mensaje
   */
  on(messageType, handler) {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType).push(handler);
    console.log(`✅ Handler registrado para: ${messageType}, Total: ${this.messageHandlers.get(messageType).length}`);
  }

  /**
   * Desregistra un manejador
   */
  off(messageType, handler) {
    if (this.messageHandlers.has(messageType)) {
      const handlers = this.messageHandlers.get(messageType);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * Maneja mensajes entrantes
   */
  handleMessage(data) {
    try {
      const message = JSON.parse(data);
      console.log('📨 Mensaje recibido:', message.type, message);

      // Guardar session ID si viene en el mensaje
      if (message.session_id) {
        this.sessionId = message.session_id;
      }

      // Ejecutar manejadores registrados
      const handlers = this.messageHandlers.get(message.type) || [];
      console.log(`Handlers para ${message.type}:`, handlers.length);
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error(`Error en manejador de ${message.type}:`, error);
        }
      });

      // Manejador genérico para todos los mensajes
      const allHandlers = this.messageHandlers.get('*') || [];
      allHandlers.forEach(handler => handler(message));

    } catch (error) {
      console.error('Error procesando mensaje:', error, 'Data:', data);
    }
  }

  /**
   * Maneja desconexión e intenta reconectar
   */
  handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`🔄 Intentando reconectar (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('Error reconectando:', error);
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('❌ No se pudo reconectar después de varios intentos');
    }
  }

  /**
   * Procesa mensajes en cola
   */
  processMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.ws.send(JSON.stringify(message));
    }
  }

  // Métodos específicos para ARC

  /**
   * Carga puzzles del servidor
   */
  loadPuzzles(puzzleSet = 'training', count = 10) {
    this.send('load_puzzles', { puzzle_set: puzzleSet, count });
  }
  
  /**
   * Envía un mensaje y espera una respuesta
   */
  sendMessage(message) {
    return new Promise((resolve, reject) => {
      // Generar ID único para el mensaje
      const messageId = `msg_${Date.now()}_${Math.random()}`;
      const messageWithId = { ...message, messageId };
      
      // Timeout para la respuesta
      const timeout = setTimeout(() => {
        this.off('validation_result', responseHandler);
        reject(new Error('Timeout esperando respuesta del servidor'));
      }, 10000);
      
      // Handler para la respuesta
      const responseHandler = (response) => {
        if (response.type === 'validation_result') {
          clearTimeout(timeout);
          this.off('validation_result', responseHandler);
          resolve(response);
        }
      };
      
      // Registrar el handler
      this.on('validation_result', responseHandler);
      
      // Enviar el mensaje
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(messageWithId));
      } else {
        clearTimeout(timeout);
        reject(new Error('WebSocket no está conectado'));
      }
    });
  }

  /**
   * Resuelve un puzzle
   */
  solvePuzzle(puzzleId, puzzleData) {
    this.send('solve_puzzle', { puzzle_id: puzzleId, puzzle: puzzleData });
  }

  /**
   * Obtiene pasos detallados de razonamiento
   */
  getReasoningSteps(puzzleId) {
    this.send('get_reasoning_steps', { puzzle_id: puzzleId });
  }

  /**
   * Verifica integridad del sistema
   */
  verifyIntegrity() {
    this.send('verify_integrity');
  }

  /**
   * Exporta visualización
   */
  exportVisualization(puzzleId, exportType = 'png') {
    this.send('export_visualization', { puzzle_id: puzzleId, export_type: exportType });
  }

  /**
   * Obtiene estado de conexión
   */
  isConnected() {
    return this.connected && this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Obtiene ID de sesión
   */
  getSessionId() {
    return this.sessionId;
  }
}

export default ARCWebSocketClient;
