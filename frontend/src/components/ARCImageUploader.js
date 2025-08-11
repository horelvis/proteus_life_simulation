/**
 * Componente para cargar imágenes de puzzles ARC
 */

import React, { useState, useRef } from 'react';

export const ARCImageUploader = ({ onImageProcessed, wsClient }) => {
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
  const [processedGrid, setProcessedGrid] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const imageData = e.target.result;
        setPreview(imageData);
        processImage(imageData);
      };
      
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        const imageData = e.target.result;
        setPreview(imageData);
        processImage(imageData);
      };
      
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handlePaste = (event) => {
    const items = event.clipboardData.items;
    
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith('image/')) {
        const blob = items[i].getAsFile();
        const reader = new FileReader();
        
        reader.onload = (e) => {
          const imageData = e.target.result;
          setPreview(imageData);
          processImage(imageData);
        };
        
        reader.readAsDataURL(blob);
        break;
      }
    }
  };

  const processImage = async (imageData) => {
    if (!wsClient || !wsClient.isConnected()) {
      alert('No hay conexión con el servidor');
      return;
    }

    setUploading(true);
    setProcessedGrid(null);

    // Registrar handler temporal para la respuesta
    const handleResponse = (msg) => {
      if (msg.type === 'image_processed') {
        setProcessedGrid(msg.grid);
        setUploading(false);
        
        // Crear puzzle desde la imagen
        const puzzle = {
          id: `img_${Date.now()}`,
          category: 'image_imported',
          difficulty: 'unknown',
          train: [{
            input: msg.grid,
            output: msg.grid // Por ahora, mismo grid
          }],
          test: [{
            input: msg.grid,
            output: []
          }]
        };
        
        if (onImageProcessed) {
          onImageProcessed(puzzle);
        }
        
        // Desregistrar handler
        wsClient.off('image_processed', handleResponse);
      } else if (msg.type === 'image_puzzle_loaded') {
        setProcessedGrid(msg.puzzle.train[0]?.input || null);
        setUploading(false);
        
        if (onImageProcessed) {
          onImageProcessed(msg.puzzle);
        }
        
        wsClient.off('image_puzzle_loaded', handleResponse);
      }
    };

    wsClient.on('image_processed', handleResponse);
    wsClient.on('image_puzzle_loaded', handleResponse);

    // Enviar imagen para procesar
    wsClient.send('load_image_puzzle', {
      image_data: imageData
    });
  };

  // Habilitar paste global
  React.useEffect(() => {
    document.addEventListener('paste', handlePaste);
    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, []);

  return (
    <div style={{
      backgroundColor: '#2a2a2a',
      padding: '20px',
      borderRadius: '8px',
      marginBottom: '20px'
    }}>
      <h3 style={{ color: '#7FDBFF', marginBottom: '15px' }}>
        📷 Cargar Imagen de Puzzle
      </h3>

      {/* Área de drop */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: '2px dashed #7FDBFF',
          borderRadius: '8px',
          padding: '40px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: '#1a1a1a',
          marginBottom: '15px',
          transition: 'all 0.3s ease'
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#333'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#1a1a1a'}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <div style={{ fontSize: '48px', marginBottom: '10px' }}>
          📁
        </div>
        <p style={{ color: '#fff', marginBottom: '10px' }}>
          Arrastra una imagen aquí o haz click para seleccionar
        </p>
        <p style={{ color: '#888', fontSize: '14px' }}>
          También puedes pegar una imagen con Ctrl+V
        </p>
      </div>

      {/* Vista previa */}
      {preview && (
        <div style={{ marginTop: '20px' }}>
          <h4 style={{ color: '#fff', marginBottom: '10px' }}>
            Vista Previa:
          </h4>
          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start' }}>
            {/* Imagen original */}
            <div style={{ flex: 1 }}>
              <p style={{ color: '#888', fontSize: '12px', marginBottom: '5px' }}>
                Original:
              </p>
              <img 
                src={preview} 
                alt="Preview" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '300px',
                  border: '1px solid #444',
                  borderRadius: '4px'
                }}
              />
            </div>

            {/* Grid procesado */}
            {processedGrid && (
              <div style={{ flex: 1 }}>
                <p style={{ color: '#888', fontSize: '12px', marginBottom: '5px' }}>
                  Grid Detectado ({processedGrid.length}×{processedGrid[0]?.length}):
                </p>
                <div style={{
                  display: 'inline-block',
                  border: '1px solid #444',
                  borderRadius: '4px',
                  padding: '5px',
                  backgroundColor: '#000'
                }}>
                  {processedGrid.map((row, i) => (
                    <div key={i} style={{ display: 'flex' }}>
                      {row.map((cell, j) => (
                        <div
                          key={j}
                          style={{
                            width: '15px',
                            height: '15px',
                            backgroundColor: getARCColor(cell),
                            border: '0.5px solid #333'
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Estado de carga */}
      {uploading && (
        <div style={{
          textAlign: 'center',
          padding: '20px',
          color: '#7FDBFF'
        }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>
            ⏳ Procesando imagen...
          </div>
          <div style={{ fontSize: '14px', color: '#888' }}>
            Detectando grid y colores ARC
          </div>
        </div>
      )}

      {/* Instrucciones */}
      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#1a1a1a',
        borderRadius: '5px',
        fontSize: '13px',
        color: '#888'
      }}>
        <strong style={{ color: '#7FDBFF' }}>💡 Tips:</strong>
        <ul style={{ margin: '10px 0 0 20px' }}>
          <li>Puedes cargar screenshots de puzzles ARC</li>
          <li>El sistema detectará automáticamente los grids</li>
          <li>Funciona mejor con imágenes claras y sin distorsión</li>
          <li>Puedes pegar imágenes directamente desde el portapapeles</li>
        </ul>
      </div>
    </div>
  );
};

// Función auxiliar para obtener colores ARC
function getARCColor(value) {
  const colors = {
    0: '#000000', // Negro
    1: '#0074D9', // Azul
    2: '#FF4136', // Rojo
    3: '#2ECC40', // Verde
    4: '#FFDC00', // Amarillo
    5: '#AAAAAA', // Gris
    6: '#F012BE', // Magenta
    7: '#FF851B', // Naranja
    8: '#7FDBFF', // Cyan
    9: '#870C25', // Marrón
  };
  return colors[value] || '#000000';
}