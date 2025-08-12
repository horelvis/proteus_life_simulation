// Centralized configuration for API and WebSocket URLs

// Helper to trim trailing slash
const trimSlash = (s) => (s ? s.replace(/\/$/, '') : s);

// Determine current origin
const getOrigin = () => {
  if (typeof window !== 'undefined') {
    return `${window.location.protocol}//${window.location.host}`;
  }
  return 'http://localhost:8000';
};

// Backend base URL
export const BACKEND_BASE_URL = (() => {
  const env = typeof process !== 'undefined' ? process.env.REACT_APP_BACKEND_URL : undefined;
  return trimSlash(env && env.trim() ? env.trim() : getOrigin());
})();

// REST API base
export const API_BASE_URL = `${BACKEND_BASE_URL}/api`;

// Primary WS for simulation/fastapi WS
export const WS_URL = (() => {
  const env = typeof process !== 'undefined' ? process.env.REACT_APP_WS_URL : undefined;
  if (env && env.trim()) return trimSlash(env.trim());
  // Derivar desde BACKEND_BASE_URL si está disponible
  const backend = typeof process !== 'undefined' ? process.env.REACT_APP_BACKEND_URL : undefined;
  if (backend && backend.trim()) {
    try {
      const u = new URL(backend.trim());
      const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${u.host}/ws`;
    } catch (_) {}
  }
  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Opción B (directo): derivar a backend:8000 cuando no hay env
    return `${proto}//${window.location.hostname}:8000/ws`;
  }
  return 'ws://localhost:8000/ws';
})();

// ARC dedicated WebSocket (proxied via Nginx at /arc-ws)
export const ARC_WS_URL = (() => {
  const env = typeof process !== 'undefined' ? process.env.REACT_APP_ARC_WS_URL : undefined;
  if (env && env.trim()) return trimSlash(env.trim());
  // Derivar desde BACKEND_BASE_URL si está disponible (mismo host, puerto 8765)
  const backend = typeof process !== 'undefined' ? process.env.REACT_APP_BACKEND_URL : undefined;
  if (backend && backend.trim()) {
    try {
      const u = new URL(backend.trim());
      const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${u.hostname}:8765`;
    } catch (_) {}
  }
  // Opción B (directo): derivar a backend:8765 cuando no hay env
  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.hostname}:8765`;
  }
  return 'ws://localhost:8765';
})();

export default {
  BACKEND_BASE_URL,
  API_BASE_URL,
  WS_URL,
  ARC_WS_URL,
};
