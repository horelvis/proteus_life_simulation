/**
 * Servicio singleton para WebSocket de ARC
 * Una sola instancia compartida por toda la aplicación
 */

import ARCWebSocketClient from './ARCWebSocketClient';

class ARCWebSocketService {
  constructor() {
    if (ARCWebSocketService.instance) {
      return ARCWebSocketService.instance;
    }
    
    this.client = null;
    this.connected = false;
    this.connecting = false;
    this.connectionPromise = null;
    
    ARCWebSocketService.instance = this;
  }
  
  /**
   * Obtiene o crea la conexión WebSocket
   */
  async getConnection() {
    // Si ya está conectado, devolver el cliente
    if (this.connected && this.client && this.client.isConnected()) {
      return this.client;
    }
    
    // Si está conectando, esperar la promesa existente
    if (this.connecting && this.connectionPromise) {
      return this.connectionPromise;
    }
    
    // Crear nueva conexión
    this.connecting = true;
    this.connectionPromise = this.connect();
    
    try {
      const client = await this.connectionPromise;
      this.connecting = false;
      return client;
    } catch (error) {
      this.connecting = false;
      throw error;
    }
  }
  
  /**
   * Conecta al WebSocket
   */
  async connect() {
    if (!this.client) {
      this.client = new ARCWebSocketClient();
      
      // Registrar handlers globales
      this.client.on('connected', () => {
        this.connected = true;
        console.log('✅ Servicio WebSocket conectado');
      });
      
      this.client.on('disconnected', () => {
        this.connected = false;
        console.log('❌ Servicio WebSocket desconectado');
      });
    }
    
    if (!this.client.isConnected()) {
      await this.client.connect();
    }
    
    return this.client;
  }
  
  /**
   * Desconecta el WebSocket
   */
  disconnect() {
    if (this.client) {
      this.client.disconnect();
      this.connected = false;
    }
  }
  
  /**
   * Obtiene el cliente actual (puede ser null)
   */
  getClient() {
    return this.client;
  }
  
  /**
   * Verifica si está conectado
   */
  isConnected() {
    return this.connected && this.client && this.client.isConnected();
  }
}

// Crear instancia singleton
const arcWebSocketService = new ARCWebSocketService();

// No congelar el objeto ya que necesitamos modificar sus propiedades internas
// Object.freeze(arcWebSocketService);

export default arcWebSocketService;