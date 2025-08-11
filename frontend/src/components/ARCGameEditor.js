import React, { useState, useCallback, useEffect, useRef } from 'react';

const ARC_COLORS = {
  0: '#000000',
  1: '#0074D9',
  2: '#FF4136',
  3: '#2ECC40',
  4: '#FFDC00',
  5: '#AAAAAA',
  6: '#F012BE',
  7: '#FF851B',
  8: '#7FDBFF',
  9: '#870C25',
};

export const ARCGameEditor = ({ 
  puzzle,
  onSubmit,
  onSave,
  initialGrid = null,
  cellSize = 30
}) => {
  const [outputGrid, setOutputGrid] = useState(initialGrid || []);
  const [selectedColor, setSelectedColor] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState('paint');
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [copiedSelection, setCopiedSelection] = useState(null);
  const [selection, setSelection] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (puzzle?.test?.[0]?.input) {
      const input = puzzle.test[0].input;
      const initial = initialGrid || input.map(row => [...row]);
      setOutputGrid(initial);
      setHistory([initial]);
      setHistoryIndex(0);
    }
  }, [puzzle, initialGrid]);

  const saveToHistory = useCallback((grid) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(grid.map(row => [...row]));
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [history, historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setOutputGrid(history[historyIndex - 1].map(row => [...row]));
    }
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setOutputGrid(history[historyIndex + 1].map(row => [...row]));
    }
  }, [history, historyIndex]);

  const paintCell = useCallback((row, col) => {
    if (row < 0 || row >= outputGrid.length || col < 0 || col >= outputGrid[0].length) return;
    
    const newGrid = outputGrid.map((r, i) => 
      i === row ? r.map((c, j) => j === col ? selectedColor : c) : [...r]
    );
    setOutputGrid(newGrid);
    return newGrid;
  }, [outputGrid, selectedColor]);

  const fillArea = useCallback((startRow, startCol) => {
    const targetColor = outputGrid[startRow][startCol];
    if (targetColor === selectedColor) return;
    
    const newGrid = outputGrid.map(row => [...row]);
    const queue = [[startRow, startCol]];
    const visited = new Set();
    
    while (queue.length > 0) {
      const [row, col] = queue.shift();
      const key = `${row},${col}`;
      
      if (visited.has(key)) continue;
      if (row < 0 || row >= newGrid.length || col < 0 || col >= newGrid[0].length) continue;
      if (newGrid[row][col] !== targetColor) continue;
      
      visited.add(key);
      newGrid[row][col] = selectedColor;
      
      queue.push([row - 1, col], [row + 1, col], [row, col - 1], [row, col + 1]);
    }
    
    setOutputGrid(newGrid);
    saveToHistory(newGrid);
  }, [outputGrid, selectedColor, saveToHistory]);

  const handleCellClick = useCallback((row, col, e) => {
    e.preventDefault();
    
    if (tool === 'paint') {
      const newGrid = paintCell(row, col);
      if (newGrid) saveToHistory(newGrid);
    } else if (tool === 'fill') {
      fillArea(row, col);
    } else if (tool === 'eyedropper') {
      setSelectedColor(outputGrid[row][col]);
      setTool('paint');
    }
  }, [tool, paintCell, fillArea, outputGrid, saveToHistory]);

  const handleMouseDown = useCallback((row, col, e) => {
    e.preventDefault();
    setIsDrawing(true);
    
    if (tool === 'select') {
      setDragStart({ row, col });
      setIsDragging(true);
      setSelection(null);
    } else if (tool === 'paint') {
      handleCellClick(row, col, e);
    }
  }, [tool, handleCellClick]);

  const handleMouseEnter = useCallback((row, col) => {
    if (isDrawing && tool === 'paint') {
      paintCell(row, col);
    } else if (isDragging && tool === 'select' && dragStart) {
      const minRow = Math.min(dragStart.row, row);
      const maxRow = Math.max(dragStart.row, row);
      const minCol = Math.min(dragStart.col, col);
      const maxCol = Math.max(dragStart.col, col);
      setSelection({ minRow, maxRow, minCol, maxCol });
    }
  }, [isDrawing, isDragging, tool, dragStart, paintCell]);

  const handleMouseUp = useCallback(() => {
    if (isDrawing && tool === 'paint') {
      saveToHistory(outputGrid);
    }
    setIsDrawing(false);
    setIsDragging(false);
    setDragStart(null);
  }, [isDrawing, tool, outputGrid, saveToHistory]);

  const copySelection = useCallback(() => {
    if (!selection) return;
    
    const copied = [];
    for (let i = selection.minRow; i <= selection.maxRow; i++) {
      const row = [];
      for (let j = selection.minCol; j <= selection.maxCol; j++) {
        row.push(outputGrid[i][j]);
      }
      copied.push(row);
    }
    setCopiedSelection(copied);
  }, [selection, outputGrid]);

  const pasteSelection = useCallback(() => {
    if (!copiedSelection || !selection) return;
    
    const newGrid = outputGrid.map(row => [...row]);
    const pasteRow = selection.minRow;
    const pasteCol = selection.minCol;
    
    for (let i = 0; i < copiedSelection.length; i++) {
      for (let j = 0; j < copiedSelection[i].length; j++) {
        const targetRow = pasteRow + i;
        const targetCol = pasteCol + j;
        if (targetRow < newGrid.length && targetCol < newGrid[0].length) {
          newGrid[targetRow][targetCol] = copiedSelection[i][j];
        }
      }
    }
    
    setOutputGrid(newGrid);
    saveToHistory(newGrid);
    setSelection(null);
  }, [copiedSelection, selection, outputGrid, saveToHistory]);

  const clearGrid = useCallback(() => {
    const newGrid = outputGrid.map(row => row.map(() => 0));
    setOutputGrid(newGrid);
    saveToHistory(newGrid);
  }, [outputGrid, saveToHistory]);

  const resetToInput = useCallback(() => {
    if (puzzle?.test?.[0]?.input) {
      const input = puzzle.test[0].input;
      const newGrid = input.map(row => [...row]);
      setOutputGrid(newGrid);
      saveToHistory(newGrid);
    }
  }, [puzzle, saveToHistory]);

  const resizeGrid = useCallback((newHeight, newWidth) => {
    const newGrid = [];
    for (let i = 0; i < newHeight; i++) {
      const row = [];
      for (let j = 0; j < newWidth; j++) {
        if (i < outputGrid.length && j < outputGrid[0].length) {
          row.push(outputGrid[i][j]);
        } else {
          row.push(0);
        }
      }
      newGrid.push(row);
    }
    setOutputGrid(newGrid);
    saveToHistory(newGrid);
  }, [outputGrid, saveToHistory]);

  const renderGrid = (grid, title, isEditable = false) => {
    if (!grid || !Array.isArray(grid)) return null;
    
    const height = grid.length;
    const width = grid[0]?.length || 0;
    
    return (
      <div style={{ 
        margin: '10px',
        padding: '15px',
        backgroundColor: '#2a2a2a',
        borderRadius: '8px'
      }}>
        <h4 style={{ 
          marginBottom: '10px', 
          color: '#fff',
          fontSize: '14px',
          textAlign: 'center'
        }}>
          {title}
        </h4>
        <div 
          style={{
            display: 'inline-block',
            border: '2px solid #444',
            borderRadius: '4px',
            padding: '2px',
            backgroundColor: '#1a1a1a',
            position: 'relative'
          }}
          onMouseUp={isEditable ? handleMouseUp : undefined}
          onMouseLeave={isEditable ? handleMouseUp : undefined}
        >
          <div style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${width}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${height}, ${cellSize}px)`,
            gap: '1px',
            backgroundColor: '#333'
          }}>
            {grid.map((row, i) => 
              row.map((cell, j) => {
                const isSelected = selection && 
                  i >= selection.minRow && i <= selection.maxRow &&
                  j >= selection.minCol && j <= selection.maxCol;
                
                return (
                  <div
                    key={`${i}-${j}`}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: ARC_COLORS[cell] || '#000',
                      border: isSelected ? '2px solid #fff' : '1px solid rgba(255,255,255,0.1)',
                      position: 'relative',
                      cursor: isEditable ? 'pointer' : 'default'
                    }}
                    onMouseDown={isEditable ? (e) => handleMouseDown(i, j, e) : undefined}
                    onMouseEnter={isEditable ? () => handleMouseEnter(i, j) : undefined}
                    onClick={isEditable ? (e) => handleCellClick(i, j, e) : undefined}
                    onContextMenu={isEditable ? (e) => {
                      e.preventDefault();
                      setSelectedColor(cell);
                    } : undefined}
                  >
                    {cell !== 0 && (
                      <span style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        color: cell > 4 ? '#000' : '#fff',
                        fontSize: '10px',
                        fontWeight: 'bold',
                        textShadow: '0 0 2px rgba(0,0,0,0.5)',
                        pointerEvents: 'none'
                      }}>
                        {cell}
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
        <div style={{ 
          marginTop: '5px', 
          fontSize: '11px', 
          color: '#888',
          textAlign: 'center'
        }}>
          {height}×{width}
        </div>
      </div>
    );
  };

  const renderTools = () => (
    <div style={{
      backgroundColor: '#2a2a2a',
      padding: '15px',
      borderRadius: '8px',
      marginBottom: '20px'
    }}>
      <h4 style={{ color: '#fff', marginBottom: '10px' }}>🎨 Herramientas</h4>
      
      <div style={{ marginBottom: '15px' }}>
        <div style={{ marginBottom: '10px' }}>
          <span style={{ color: '#888', fontSize: '12px' }}>Herramienta activa:</span>
        </div>
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
          <button
            onClick={() => setTool('paint')}
            style={{
              padding: '5px 10px',
              backgroundColor: tool === 'paint' ? '#0074D9' : '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            🖌️ Pintar
          </button>
          <button
            onClick={() => setTool('fill')}
            style={{
              padding: '5px 10px',
              backgroundColor: tool === 'fill' ? '#0074D9' : '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            🪣 Rellenar
          </button>
          <button
            onClick={() => setTool('eyedropper')}
            style={{
              padding: '5px 10px',
              backgroundColor: tool === 'eyedropper' ? '#0074D9' : '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            💧 Gotero
          </button>
          <button
            onClick={() => setTool('select')}
            style={{
              padding: '5px 10px',
              backgroundColor: tool === 'select' ? '#0074D9' : '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            ⬚ Seleccionar
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <div style={{ marginBottom: '10px' }}>
          <span style={{ color: '#888', fontSize: '12px' }}>Color seleccionado:</span>
        </div>
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
          {Object.entries(ARC_COLORS).map(([value, color]) => (
            <button
              key={value}
              onClick={() => setSelectedColor(parseInt(value))}
              style={{
                width: '30px',
                height: '30px',
                backgroundColor: color,
                border: selectedColor === parseInt(value) ? '3px solid #fff' : '1px solid #444',
                borderRadius: '3px',
                cursor: 'pointer',
                position: 'relative'
              }}
            >
              <span style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                color: parseInt(value) > 4 ? '#000' : '#fff',
                fontSize: '10px',
                fontWeight: 'bold',
                textShadow: '0 0 2px rgba(0,0,0,0.5)'
              }}>
                {value}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
        <button
          onClick={undo}
          disabled={historyIndex <= 0}
          style={{
            padding: '5px 10px',
            backgroundColor: historyIndex <= 0 ? '#333' : '#444',
            color: historyIndex <= 0 ? '#666' : '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: historyIndex <= 0 ? 'not-allowed' : 'pointer',
            fontSize: '12px'
          }}
        >
          ↩️ Deshacer
        </button>
        <button
          onClick={redo}
          disabled={historyIndex >= history.length - 1}
          style={{
            padding: '5px 10px',
            backgroundColor: historyIndex >= history.length - 1 ? '#333' : '#444',
            color: historyIndex >= history.length - 1 ? '#666' : '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: historyIndex >= history.length - 1 ? 'not-allowed' : 'pointer',
            fontSize: '12px'
          }}
        >
          ↪️ Rehacer
        </button>
        <button
          onClick={copySelection}
          disabled={!selection}
          style={{
            padding: '5px 10px',
            backgroundColor: !selection ? '#333' : '#444',
            color: !selection ? '#666' : '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: !selection ? 'not-allowed' : 'pointer',
            fontSize: '12px'
          }}
        >
          📋 Copiar
        </button>
        <button
          onClick={pasteSelection}
          disabled={!copiedSelection || !selection}
          style={{
            padding: '5px 10px',
            backgroundColor: !copiedSelection || !selection ? '#333' : '#444',
            color: !copiedSelection || !selection ? '#666' : '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: !copiedSelection || !selection ? 'not-allowed' : 'pointer',
            fontSize: '12px'
          }}
        >
          📄 Pegar
        </button>
        <button
          onClick={clearGrid}
          style={{
            padding: '5px 10px',
            backgroundColor: '#444',
            color: '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          🗑️ Limpiar
        </button>
        <button
          onClick={resetToInput}
          style={{
            padding: '5px 10px',
            backgroundColor: '#444',
            color: '#fff',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          🔄 Resetear
        </button>
      </div>

      {onSave && (
        <button
          onClick={() => onSave(outputGrid)}
          style={{
            marginTop: '10px',
            padding: '8px 15px',
            backgroundColor: '#FFDC00',
            color: '#000',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: 'bold',
            width: '100%'
          }}
        >
          💾 Guardar Progreso
        </button>
      )}
    </div>
  );

  const renderExamples = () => {
    if (!puzzle?.train || puzzle.train.length === 0) return null;
    
    return (
      <div style={{
        backgroundColor: '#1a1a1a',
        padding: '20px',
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3 style={{ color: '#7FDBFF', marginBottom: '15px' }}>
          📚 Ejemplos de Entrenamiento
        </h3>
        <div style={{ 
          display: 'flex', 
          overflowX: 'auto',
          gap: '20px',
          paddingBottom: '10px'
        }}>
          {puzzle.train.map((example, idx) => (
            <div key={idx} style={{
              minWidth: '200px',
              padding: '10px',
              backgroundColor: '#2a2a2a',
              borderRadius: '5px'
            }}>
              <h5 style={{ color: '#fff', marginBottom: '10px', fontSize: '12px' }}>
                Ejemplo {idx + 1}
              </h5>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {renderGrid(example.input, 'Entrada')}
                <span style={{ color: '#666' }}>→</span>
                {renderGrid(example.output, 'Salida')}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={{ width: '100%' }}>
      {renderExamples()}
      
      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ flex: '1', minWidth: '300px' }}>
          {renderTools()}
          
          <div style={{ 
            backgroundColor: '#2a2a2a',
            padding: '15px',
            borderRadius: '8px'
          }}>
            <h4 style={{ color: '#888', marginBottom: '10px', fontSize: '12px' }}>
              Atajos de teclado:
            </h4>
            <div style={{ fontSize: '11px', color: '#666' }}>
              <div>Ctrl+Z: Deshacer</div>
              <div>Ctrl+Y: Rehacer</div>
              <div>Ctrl+C: Copiar selección</div>
              <div>Ctrl+V: Pegar</div>
              <div>1-9, 0: Seleccionar color</div>
              <div>P: Herramienta pintar</div>
              <div>F: Herramienta rellenar</div>
              <div>E: Gotero</div>
              <div>S: Seleccionar</div>
            </div>
          </div>
        </div>
        
        <div style={{ flex: '2', minWidth: '400px' }}>
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
            {puzzle?.test?.[0]?.input && renderGrid(puzzle.test[0].input, 'Entrada')}
            <span style={{ fontSize: '24px', margin: '0 20px', color: '#666' }}>→</span>
            {renderGrid(outputGrid, 'Tu Solución', true)}
          </div>
          
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            marginTop: '20px',
            gap: '10px'
          }}>
            <button
              onClick={() => onSubmit(outputGrid)}
              style={{
                padding: '12px 30px',
                backgroundColor: '#2ECC40',
                color: '#fff',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: 'bold'
              }}
            >
              ✅ Enviar Solución
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};