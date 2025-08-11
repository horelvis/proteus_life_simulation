import React, { useState, useEffect, useCallback } from 'react';
import { ARCGameEditor } from './ARCGameEditor';
import { ARCPuzzleVisualization } from './ARCPuzzleVisualization';

export const ARCGameMode = ({ 
  webSocketClient,
  onBack
}) => {
  const [gameState, setGameState] = useState('menu');
  const [currentPuzzle, setCurrentPuzzle] = useState(null);
  const [userSolution, setUserSolution] = useState(null);
  const [expectedSolution, setExpectedSolution] = useState(null);
  const [savedProgress, setSavedProgress] = useState({});
  const [puzzleList, setPuzzleList] = useState([]);
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');
  const [score, setScore] = useState(0);
  const [attempts, setAttempts] = useState({});
  const [validationResult, setValidationResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadPuzzleList();
    loadProgress();
  }, []);

  const loadPuzzleList = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/arc/puzzles');
      if (response.ok) {
        const data = await response.json();
        setPuzzleList(data.puzzles || []);
      }
    } catch (error) {
      console.error('Error loading puzzles:', error);
      setPuzzleList([
        { id: 'demo_1', name: 'Tutorial 1', difficulty: 'easy', category: 'pattern' },
        { id: 'demo_2', name: 'Tutorial 2', difficulty: 'easy', category: 'color' },
        { id: 'demo_3', name: 'Challenge 1', difficulty: 'medium', category: 'transform' },
        { id: 'demo_4', name: 'Challenge 2', difficulty: 'hard', category: 'logic' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadProgress = () => {
    const saved = localStorage.getItem('arc_game_progress');
    if (saved) {
      const progress = JSON.parse(saved);
      setSavedProgress(progress.savedGrids || {});
      setScore(progress.score || 0);
      setAttempts(progress.attempts || {});
    }
  };

  const saveProgress = (puzzleId, grid) => {
    const newProgress = {
      ...savedProgress,
      [puzzleId]: grid
    };
    setSavedProgress(newProgress);
    
    const progressData = {
      savedGrids: newProgress,
      score,
      attempts
    };
    localStorage.setItem('arc_game_progress', JSON.stringify(progressData));
  };

  const loadPuzzle = async (puzzleId) => {
    setIsLoading(true);
    setValidationResult(null);
    try {
      const response = await fetch(`/api/arc/puzzle/${puzzleId}`);
      if (response.ok) {
        const puzzle = await response.json();
        setCurrentPuzzle(puzzle);
        setGameState('playing');
        
        if (savedProgress[puzzleId]) {
          setUserSolution(savedProgress[puzzleId]);
        }
      } else {
        const demoData = generateDemoPuzzle(puzzleId);
        setCurrentPuzzle(demoData);
        setGameState('playing');
      }
    } catch (error) {
      console.error('Error loading puzzle:', error);
      const demoData = generateDemoPuzzle(puzzleId);
      setCurrentPuzzle(demoData);
      setGameState('playing');
    } finally {
      setIsLoading(false);
    }
  };

  const generateDemoPuzzle = (puzzleId) => {
    const demos = {
      'demo_1': {
        id: 'demo_1',
        name: 'Tutorial 1: Patrones',
        category: 'pattern',
        difficulty: 'easy',
        train: [
          {
            input: [
              [0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]
            ],
            output: [
              [1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]
            ]
          },
          {
            input: [
              [0, 2, 0],
              [0, 0, 0],
              [0, 0, 0]
            ],
            output: [
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 2]
            ]
          }
        ],
        test: [
          {
            input: [
              [0, 0, 0],
              [0, 0, 3],
              [0, 0, 0]
            ]
          }
        ]
      },
      'demo_2': {
        id: 'demo_2',
        name: 'Tutorial 2: Colores',
        category: 'color',
        difficulty: 'easy',
        train: [
          {
            input: [
              [1, 0, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 3, 0],
              [0, 0, 0, 4]
            ],
            output: [
              [1, 1, 1, 1],
              [2, 2, 2, 2],
              [3, 3, 3, 3],
              [4, 4, 4, 4]
            ]
          }
        ],
        test: [
          {
            input: [
              [5, 0, 0, 0],
              [0, 6, 0, 0],
              [0, 0, 7, 0],
              [0, 0, 0, 8]
            ]
          }
        ]
      },
      'demo_3': {
        id: 'demo_3',
        name: 'Challenge 1: Transformación',
        category: 'transform',
        difficulty: 'medium',
        train: [
          {
            input: [
              [0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]
            ],
            output: [
              [0, 2, 0],
              [2, 2, 2],
              [0, 2, 0]
            ]
          },
          {
            input: [
              [3, 3, 3],
              [3, 0, 3],
              [3, 3, 3]
            ],
            output: [
              [4, 4, 4],
              [4, 0, 4],
              [4, 4, 4]
            ]
          }
        ],
        test: [
          {
            input: [
              [5, 5, 0],
              [5, 0, 0],
              [0, 0, 0]
            ]
          }
        ]
      }
    };
    
    return demos[puzzleId] || demos['demo_1'];
  };

  const handleSubmit = async (solution) => {
    setUserSolution(solution);
    setIsLoading(true);
    
    const puzzleAttempts = (attempts[currentPuzzle.id] || 0) + 1;
    setAttempts({ ...attempts, [currentPuzzle.id]: puzzleAttempts });
    
    try {
      if (webSocketClient?.ws?.readyState === WebSocket.OPEN) {
        const response = await webSocketClient.sendMessage({
          type: 'validate_solution',
          puzzleId: currentPuzzle.id,
          solution: solution
        });
        
        if (response.isCorrect) {
          const points = calculateScore(currentPuzzle.difficulty, puzzleAttempts);
          setScore(score + points);
          setValidationResult({
            correct: true,
            message: `¡Correcto! +${points} puntos`,
            points
          });
        } else {
          setValidationResult({
            correct: false,
            message: `Incorrecto. Intento ${puzzleAttempts}`,
            differences: response.differences
          });
        }
        
        if (response.expected) {
          setExpectedSolution(response.expected);
        }
      } else {
        const isCorrect = validateLocally(solution);
        if (isCorrect) {
          const points = calculateScore(currentPuzzle.difficulty, puzzleAttempts);
          setScore(score + points);
          setValidationResult({
            correct: true,
            message: `¡Correcto! +${points} puntos`,
            points
          });
        } else {
          setValidationResult({
            correct: false,
            message: `Incorrecto. Intento ${puzzleAttempts}`
          });
        }
      }
    } catch (error) {
      console.error('Error validating solution:', error);
      setValidationResult({
        correct: false,
        message: 'Error al validar la solución'
      });
    } finally {
      setIsLoading(false);
      setGameState('result');
    }
  };

  const validateLocally = (solution) => {
    if (currentPuzzle.id === 'demo_1') {
      const expected = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
      ];
      setExpectedSolution(expected);
      return JSON.stringify(solution) === JSON.stringify(expected);
    }
    return false;
  };

  const calculateScore = (difficulty, attempts) => {
    const baseScore = {
      'easy': 10,
      'medium': 25,
      'hard': 50
    }[difficulty] || 10;
    
    const penalty = Math.max(0, (attempts - 1) * 2);
    return Math.max(5, baseScore - penalty);
  };

  const handleSaveProgress = (grid) => {
    if (currentPuzzle) {
      saveProgress(currentPuzzle.id, grid);
    }
  };

  const renderMenu = () => (
    <div style={{
      backgroundColor: '#1a1a1a',
      padding: '30px',
      borderRadius: '10px',
      maxWidth: '800px',
      margin: '0 auto'
    }}>
      <h2 style={{ color: '#fff', textAlign: 'center', marginBottom: '30px' }}>
        🎮 ARC Prize - Modo Juego
      </h2>
      
      <div style={{
        backgroundColor: '#2a2a2a',
        padding: '20px',
        borderRadius: '8px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <h3 style={{ color: '#FFDC00', marginBottom: '10px' }}>
          🏆 Puntuación: {score}
        </h3>
        <div style={{ color: '#888', fontSize: '14px' }}>
          Puzzles completados: {Object.keys(attempts).filter(id => 
            savedProgress[id] && validationResult?.correct
          ).length}
        </div>
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        <label style={{ color: '#888', marginRight: '10px' }}>Filtrar por dificultad:</label>
        <select
          value={selectedDifficulty}
          onChange={(e) => setSelectedDifficulty(e.target.value)}
          style={{
            padding: '5px 10px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '3px'
          }}
        >
          <option value="all">Todas</option>
          <option value="easy">Fácil</option>
          <option value="medium">Medio</option>
          <option value="hard">Difícil</option>
        </select>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
        {puzzleList
          .filter(p => selectedDifficulty === 'all' || p.difficulty === selectedDifficulty)
          .map(puzzle => {
            const completed = savedProgress[puzzle.id] && validationResult?.correct;
            const attempted = attempts[puzzle.id] > 0;
            
            return (
              <button
                key={puzzle.id}
                onClick={() => loadPuzzle(puzzle.id)}
                style={{
                  padding: '15px',
                  backgroundColor: completed ? '#2ECC40' : (attempted ? '#FF851B' : '#2a2a2a'),
                  color: '#fff',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'transform 0.2s',
                  position: 'relative'
                }}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                  {puzzle.name || puzzle.id}
                </div>
                <div style={{ fontSize: '12px', color: '#ddd' }}>
                  {puzzle.category} • {puzzle.difficulty}
                </div>
                {completed && (
                  <span style={{ position: 'absolute', top: '5px', right: '5px' }}>✅</span>
                )}
                {attempted && !completed && (
                  <div style={{ fontSize: '11px', marginTop: '5px', color: '#ffd' }}>
                    Intentos: {attempts[puzzle.id]}
                  </div>
                )}
              </button>
            );
          })}
      </div>
      
      <button
        onClick={onBack}
        style={{
          marginTop: '30px',
          padding: '10px 20px',
          backgroundColor: '#444',
          color: '#fff',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        ← Volver al menú principal
      </button>
    </div>
  );

  const renderGame = () => (
    <div>
      <div style={{
        backgroundColor: '#2a2a2a',
        padding: '15px',
        borderRadius: '8px',
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{ color: '#fff', margin: 0 }}>
          {currentPuzzle?.name || currentPuzzle?.id}
        </h3>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <span style={{ color: '#FFDC00' }}>
            🏆 {score} pts
          </span>
          <button
            onClick={() => setGameState('menu')}
            style={{
              padding: '5px 15px',
              backgroundColor: '#444',
              color: '#fff',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            ← Menú
          </button>
        </div>
      </div>
      
      <ARCGameEditor
        puzzle={currentPuzzle}
        onSubmit={handleSubmit}
        onSave={handleSaveProgress}
        initialGrid={savedProgress[currentPuzzle?.id]}
        cellSize={25}
      />
    </div>
  );

  const renderResult = () => (
    <div style={{
      backgroundColor: '#1a1a1a',
      padding: '30px',
      borderRadius: '10px',
      maxWidth: '1000px',
      margin: '0 auto'
    }}>
      <div style={{
        backgroundColor: validationResult?.correct ? '#2ECC40' : '#FF4136',
        padding: '20px',
        borderRadius: '8px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        <h2 style={{ color: '#fff', marginBottom: '10px' }}>
          {validationResult?.message}
        </h2>
        {validationResult?.correct && (
          <div style={{ fontSize: '18px', color: '#fff' }}>
            Puntuación total: {score}
          </div>
        )}
      </div>
      
      <ARCPuzzleVisualization
        puzzle={currentPuzzle}
        solution={userSolution}
        expected={expectedSolution}
        showComparison={true}
        cellSize={25}
      />
      
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        gap: '20px',
        marginTop: '30px'
      }}>
        {!validationResult?.correct && (
          <button
            onClick={() => setGameState('playing')}
            style={{
              padding: '12px 30px',
              backgroundColor: '#FF851B',
              color: '#fff',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            🔄 Reintentar
          </button>
        )}
        <button
          onClick={() => setGameState('menu')}
          style={{
            padding: '12px 30px',
            backgroundColor: '#0074D9',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold'
          }}
        >
          📋 Elegir otro puzzle
        </button>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        height: '400px',
        color: '#fff'
      }}>
        <div>Cargando...</div>
      </div>
    );
  }

  switch (gameState) {
    case 'menu':
      return renderMenu();
    case 'playing':
      return renderGame();
    case 'result':
      return renderResult();
    default:
      return renderMenu();
  }
};