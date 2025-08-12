/* CRA dev proxy to simplify frontend->backend URLs
   - Proxies REST:   /api      -> http://<backend-host>:8000
   - Proxies WS:     /ws       -> http://<backend-host>:8000 (ws)
   - Proxies ARC WS: /arc-ws   -> ws(s)://<backend-host>:8765 (ws)
   Backend host is taken from REACT_APP_BACKEND_URL if provided, else http://localhost:8000
*/

const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  const backendEnv = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
  let backendUrl;
  try {
    backendUrl = new URL(backendEnv);
  } catch {
    backendUrl = new URL('http://localhost:8000');
  }

  const apiTarget = `${backendUrl.protocol}//${backendUrl.host}`;
  const arcWsTarget = `${backendUrl.protocol}//${backendUrl.hostname}:8765`;

  // REST API
  app.use(
    '/api',
    createProxyMiddleware({
      target: apiTarget,
      changeOrigin: true,
      logLevel: 'warn',
    })
  );

  // General WebSocket (/ws)
  app.use(
    '/ws',
    createProxyMiddleware({
      target: apiTarget,
      changeOrigin: true,
      ws: true,
      logLevel: 'warn',
    })
  );

  // ARC dedicated WebSocket (/arc-ws)
  app.use(
    '/arc-ws',
    createProxyMiddleware({
      target: arcWsTarget,
      changeOrigin: true,
      ws: true,
      logLevel: 'warn',
    })
  );
};

