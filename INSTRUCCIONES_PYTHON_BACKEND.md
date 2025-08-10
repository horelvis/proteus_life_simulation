# 🐍 Backend Python para PROTEUS-AC

## Instalación Rápida

### 1. Abrir Terminal 1 - Backend Python:
```bash
cd backend_arc
python3 start_server.py
```

Si es la primera vez, instalará las dependencias automáticamente.

### 2. Abrir Terminal 2 - Frontend React:
```bash
cd frontend
npm start
```

### 3. En el Navegador:
1. La app se abrirá automáticamente
2. Click en "🌐 Usar Backend Python"
3. ¡Listo! El sistema usará Python para resolver puzzles

## 🔍 ¿Qué hace el Backend Python?

- **Resuelve puzzles ARC** con 10 tipos de transformaciones
- **Muestra cada paso** del razonamiento
- **Genera visualizaciones** del proceso
- **NO usa redes neuronales**, solo lógica transparente

## 🧪 Probar el Servidor

Para verificar que el servidor funciona:
```bash
cd backend_arc
python3 test_server.py
```

## 🚨 Solución de Problemas

### Error: "No module named websockets"
```bash
pip3 install websockets numpy pillow matplotlib
```

### Error: "Connection refused"
Asegúrate de que el servidor Python esté corriendo en la Terminal 1

### Error en macOS con matplotlib
Ya está solucionado en el código (usa backend 'Agg')

## 📊 Ventajas del Backend Python

1. **Más poder de cálculo** para puzzles complejos
2. **Mejor debugging** con herramientas Python
3. **Visualizaciones avanzadas** con matplotlib
4. **Procesamiento paralelo** de múltiples puzzles

## 🎯 Características

- ✅ WebSocket para comunicación en tiempo real
- ✅ Transparencia total: cada paso es visible
- ✅ Sin cajas negras ni pesos ocultos
- ✅ Exportación de visualizaciones PNG/GIF
- ✅ Verificación de integridad integrada