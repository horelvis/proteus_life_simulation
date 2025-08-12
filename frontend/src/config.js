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

// Backend base URL (prefer env injected by CRA)
const ENV_BACKEND = process.env.REACT_APP_BACKEND_URL;
export const BACKEND_BASE_URL = trimSlash(ENV_BACKEND && ENV_BACKEND.trim() ? ENV_BACKEND.trim() : getOrigin());

// REST API base
export const API_BASE_URL = `${BACKEND_BASE_URL}/api`;

// Primary WS for simulation/fastapi WS
export const WS_URL = (() => {
  const envWs = process.env.REACT_APP_WS_URL;
  if (envWs && envWs.trim()) return trimSlash(envWs.trim());
  // Derivar desde BACKEND_BASE_URL
  try {
    const u = new URL(BACKEND_BASE_URL);
    const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${u.host}/ws`;
  } catch (_) {
    if (typeof window !== 'undefined') {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.hostname}:8000/ws`;
    }
    return 'ws://localhost:8000/ws';
  }
})();

// ARC dedicated WebSocket (proxied via Nginx at /arc-ws)
export const ARC_WS_URL = (() => {
  const envArc = process.env.REACT_APP_ARC_WS_URL;
  if (envArc && envArc.trim()) return trimSlash(envArc.trim());
  // Derivar desde BACKEND_BASE_URL (mismo host, puerto 8765)
  try {
    const u = new URL(BACKEND_BASE_URL);
    const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${u.hostname}:8765`;
  } catch (_) {
    if (typeof window !== 'undefined') {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.hostname}:8765`;
    }
    return 'ws://localhost:8765';
  }
})();

export default {
  BACKEND_BASE_URL,
  API_BASE_URL,
  WS_URL,
  ARC_WS_URL,
};
