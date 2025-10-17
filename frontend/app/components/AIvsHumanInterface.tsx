'use client';

import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import StreetViewPlayer from './StreetViewPlayer';
import TeslaStyleOverlay from './TeslaStyleOverlay';
import './AIvsHumanInterface.css';

interface GameState {
  game_id: string;
  status: string;
  ai_confidence_display: string;
  ai_guess?: [number, number];
  human_guess?: [number, number];
  actual_location?: [number, number];
  streetview_location?: any;
  ai_analysis?: any;
  game_mode?: string;
}

interface GameResult {
  ai_distance_error: number;
  human_distance_error: number;
  ai_score: number;
  human_score: number;
  winner: string;
}

interface TeslaOverlayData {
  detected_objects: Array<{
    type: string;
    confidence: number;
    bounding_box: [number, number, number, number];
    geographic_hints?: string[];
  }>;
  confidence_scores: Record<string, number>;
  overlays: Record<string, any>;
  timestamp: string;
}

const AIvsHumanInterface: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameResult, setGameResult] = useState<GameResult | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [humanGuess, setHumanGuess] = useState<{ lat: number; lon: number } | null>(null);
  const [difficulty, setDifficulty] = useState('medium');
  const [teslaOverlayData, setTeslaOverlayData] = useState<TeslaOverlayData | null>(null);
  const [isPollingOverlays, setIsPollingOverlays] = useState(false);

  const { 
    connectionStatus, 
    analysisProgress,
    lastUpdate,
    error,
    joinGame 
  } = useWebSocket();

  const isConnected = connectionStatus === 'Connected';
  const wsError = error;
  const analysisUpdate = analysisProgress;
  const gameStateUpdate = lastUpdate;

  // Handle game state updates from WebSocket
  useEffect(() => {
    if (gameStateUpdate && gameState?.game_id === gameStateUpdate.game_id) {
      setGameState(prevState => ({
        ...prevState,
        ...gameStateUpdate
      }));
    }
  }, [gameStateUpdate, gameState?.game_id]);

  // Poll for Tesla-style overlays when game is active
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (gameState?.game_id && gameState.status === 'analyzing') {
      setIsPollingOverlays(true);
      intervalId = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:8001/api/v1/games/${gameState.game_id}/live-overlays`);
          if (response.ok) {
            const overlayResponse = await response.json();
            if (overlayResponse.overlays) {
              setTeslaOverlayData({
                detected_objects: overlayResponse.detected_objects || [],
                confidence_scores: overlayResponse.confidence_scores || {},
                overlays: overlayResponse.overlays || {},
                timestamp: overlayResponse.timestamp || new Date().toISOString()
              });
            }
          }
        } catch (error) {
          console.error('Error fetching overlays:', error);
        }
      }, 1000); // Update every second for real-time feel
    } else {
      setIsPollingOverlays(false);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [gameState?.game_id, gameState?.status]);

  const startAIvsHumanGame = async () => {
    setIsStarting(true);
    setGameState(null);
    setGameResult(null);
    setHumanGuess(null);
    setTeslaOverlayData(null);

    try {
      const formData = new FormData();
      formData.append('difficulty', difficulty);
      
      const response = await fetch('http://localhost:8001/api/v1/games/ai-vs-human', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        const newGameState: GameState = {
          game_id: result.game_id,
          status: 'analyzing',
          ai_confidence_display: '🤖 AI initializing enhanced computer vision systems...',
          streetview_location: result.location,
          game_mode: 'ai_vs_human'
        };
        
        setGameState(newGameState);
        
        // Join the game room for WebSocket updates
        if (joinGame) {
          joinGame(result.game_id);
        }
        
        console.log('AI vs Human game started:', result);
      } else {
        const errorData = await response.json();
        console.error('Failed to start AI vs Human game:', errorData);
        alert(`Failed to start game: ${errorData.detail}`);
      }
    } catch (error) {
      console.error('Error starting AI vs Human game:', error);
      alert('Error starting game. Please try again.');
    } finally {
      setIsStarting(false);
    }
  };

  const submitHumanGuess = async () => {
    if (!gameState || !humanGuess) return;

    try {
      const response = await fetch(`http://localhost:8001/api/v1/games/${gameState.game_id}/guess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: humanGuess.lat,
          lon: humanGuess.lon,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setGameResult(result.result);
        setGameState(prev => prev ? { ...prev, status: 'completed' } : null);
        console.log('AI vs Human game completed:', result);
      } else {
        const errorData = await response.json();
        console.error('Failed to submit guess:', errorData);
        alert(`Failed to submit guess: ${errorData.detail}`);
      }
    } catch (error) {
      console.error('Error submitting guess:', error);
      alert('Error submitting guess. Please try again.');
    }
  };

  const resetGame = () => {
    setGameState(null);
    setGameResult(null);
    setHumanGuess(null);
    setTeslaOverlayData(null);
  };

  const getDifficultyDescription = (diff: string) => {
    const descriptions = {
      easy: "AI accuracy ~90% - Training mode",
      medium: "AI accuracy ~70% - Balanced challenge", 
      hard: "AI accuracy ~50% - Competitive mode",
      expert: "AI accuracy ~30% - Ultimate challenge"
    };
    return descriptions[diff as keyof typeof descriptions] || descriptions.medium;
  };

  return (
    <div className="ai-vs-human-interface">
      <header className="competition-header">
        <div className="header-content">
          <h1>🤖 vs 👤 AI vs Human Competition</h1>
          <p>Fair geographic challenge with Tesla-style AI visualization</p>
          <div className="connection-status">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {wsError && (
        <div className="error-banner">
          WebSocket Error: {wsError}
        </div>
      )}

      {!gameState && (
        <div className="game-setup">
          <div className="setup-card">
            <h2>Start New Competition</h2>
            
            <div className="difficulty-selection">
              <h3>Choose Difficulty Level</h3>
              <div className="difficulty-options">
                {['easy', 'medium', 'hard', 'expert'].map((level) => (
                  <label key={level} className={`difficulty-option ${difficulty === level ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="difficulty"
                      value={level}
                      checked={difficulty === level}
                      onChange={(e) => setDifficulty(e.target.value)}
                    />
                    <div className="difficulty-info">
                      <span className="difficulty-name">{level.charAt(0).toUpperCase() + level.slice(1)}</span>
                      <span className="difficulty-desc">{getDifficultyDescription(level)}</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="competition-rules">
              <h3>Competition Rules</h3>
              <div className="rules-grid">
                <div className="rule-item">
                  <span className="rule-icon">🎯</span>
                  <span>Random location - no metadata cheating</span>
                </div>
                <div className="rule-item">
                  <span className="rule-icon">👁️</span>
                  <span>AI uses only computer vision analysis</span>
                </div>
                <div className="rule-item">
                  <span className="rule-icon">⚡</span>
                  <span>Tesla-style real-time AI visualization</span>
                </div>
                <div className="rule-item">
                  <span className="rule-icon">🔄</span>
                  <span>Both players get identical imagery</span>
                </div>
                <div className="rule-item">
                  <span className="rule-icon">⏱️</span>
                  <span>No time pressure - accuracy matters</span>
                </div>
                <div className="rule-item">
                  <span className="rule-icon">📊</span>
                  <span>Distance-based scoring system</span>
                </div>
              </div>
            </div>

            <button
              onClick={startAIvsHumanGame}
              disabled={isStarting}
              className="start-competition-btn"
            >
              {isStarting ? 'Generating Random Location...' : 'Start Competition'}
            </button>
          </div>
        </div>
      )}

      {gameState && (
        <div className="competition-active">
          <div className="game-status-bar">
            <div className="status-info">
              <span className="game-id">Competition ID: {gameState.game_id}</span>
              <span className={`status-indicator status-${gameState.status}`}>
                {gameState.status === 'analyzing' ? '🔄 AI Analyzing' :
                 gameState.status === 'waiting_human' ? '⏳ Your Turn' :
                 gameState.status === 'completed' ? '✅ Complete' : gameState.status}
              </span>
            </div>
          </div>

          <div className="competition-layout">
            {/* Left Panel - AI Tesla-style Vision */}
            <div className="ai-panel">
              <div className="panel-header">
                <h3>🤖 AI Vision System</h3>
                <div className="ai-status-indicator">
                  <div className={`ai-pulse ${gameState.status === 'analyzing' ? 'active' : ''}`}></div>
                  <span>Tesla-style Analysis</span>
                </div>
              </div>

              {gameState.streetview_location && (
                <TeslaStyleOverlay
                  imageUrl={gameState.streetview_location.street_view_urls?.[0]?.url || 
                           `https://maps.googleapis.com/maps/api/streetview?size=640x640&location=${gameState.streetview_location.lat},${gameState.streetview_location.lon}&key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}`}
                  overlayData={teslaOverlayData}
                  isAnalyzing={gameState.status === 'analyzing'}
                  aiStatus={gameState.ai_confidence_display}
                  gameId={gameState.game_id}
                />
              )}

              {analysisUpdate && gameState.status === 'analyzing' && (
                <div className="analysis-progress-panel">
                  <div className="progress-text">{analysisUpdate.step}</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      data-progress={(analysisUpdate.progress || 0) * 100}
                    ></div>
                  </div>
                  <div className="progress-stats">
                    Step {analysisUpdate.step_number} of {analysisUpdate.total_steps}
                  </div>
                </div>
              )}
            </div>

            {/* Right Panel - Human View */}
            <div className="human-panel">
              <div className="panel-header">
                <h3>👤 Human View</h3>
                <div className="human-status">
                  {gameState.status === 'waiting_human' ? 
                    <span className="ready-indicator">Ready for your guess!</span> :
                    <span className="waiting-indicator">AI is analyzing...</span>
                  }
                </div>
              </div>

              {gameState.streetview_location && (
                <div className="streetview-container">
                  {/* Development Mode: Use alternative map view */}
                  <div className="dev-streetview-container">
                    <iframe
                      className="streetview-iframe"
                      title="Location View"
                      src={`https://www.openstreetmap.org/export/embed.html?bbox=${gameState.streetview_location.lon-0.01},${gameState.streetview_location.lat-0.01},${gameState.streetview_location.lon+0.01},${gameState.streetview_location.lat+0.01}&layer=mapnik&marker=${gameState.streetview_location.lat},${gameState.streetview_location.lon}`}
                      allowFullScreen
                    ></iframe>
                    
                    <div className="location-info">
                      <p>📍 Location: {gameState.streetview_location.lat.toFixed(4)}, {gameState.streetview_location.lon.toFixed(4)}</p>
                      <p>🎯 Your task: Guess where this is!</p>
                      <p className="dev-note">💡 Dev Mode: Using OpenStreetMap while Google Maps API is being configured</p>
                    </div>
                  </div>
                  
                  {/* Fallback static image if available */}
                  <div className="streetview-fallback">
                    <img 
                      className="streetview-image"
                      src={gameState.streetview_location.street_view_urls?.[0]?.url ||
                           `https://picsum.photos/640/640?random=${Math.floor(Math.random() * 1000)}`}
                      alt="Location View"
                      onError={(e) => {
                        console.log('Street View image failed to load, using placeholder');
                        e.currentTarget.src = `https://picsum.photos/640/640?random=${Math.floor(Math.random() * 1000)}`;
                      }}
                    />
                  </div>
                </div>
              )}

              {gameState.status === 'waiting_human' && (
                <div className="human-input-section">
                  <h4>Make Your Guess</h4>
                  <p>Click on the world map to guess the location!</p>
                  {humanGuess && (
                    <div className="guess-display">
                      <p>Your guess: {humanGuess.lat.toFixed(6)}, {humanGuess.lon.toFixed(6)}</p>
                    </div>
                  )}
                  <button
                    onClick={submitHumanGuess}
                    disabled={!humanGuess}
                    className="submit-guess-btn"
                  >
                    Submit Final Guess
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          {gameState.status === 'completed' && gameResult && (
            <div className="results-panel">
              <div className="results-header">
                <h3>🏆 Competition Results</h3>
                <div className={`winner-announcement winner-${gameResult.winner}`}>
                  {gameResult.winner === 'ai' && '🤖 AI Wins!'}
                  {gameResult.winner === 'human' && '👤 Human Wins!'}
                  {gameResult.winner === 'tie' && '🤝 It\'s a Tie!'}
                </div>
              </div>

              <div className="results-comparison">
                <div className="result-column ai-result">
                  <div className="result-header">🤖 AI Performance</div>
                  <div className="result-stats">
                    <div className="stat-item">
                      <span className="stat-label">Distance Error:</span>
                      <span className="stat-value">{gameResult.ai_distance_error.toFixed(2)} km</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Score:</span>
                      <span className="stat-value">{gameResult.ai_score}</span>
                    </div>
                  </div>
                </div>

                <div className="result-column human-result">
                  <div className="result-header">👤 Human Performance</div>
                  <div className="result-stats">
                    <div className="stat-item">
                      <span className="stat-label">Distance Error:</span>
                      <span className="stat-value">{gameResult.human_distance_error.toFixed(2)} km</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Score:</span>
                      <span className="stat-value">{gameResult.human_score}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="competition-actions">
            <button onClick={resetGame} className="new-competition-btn">
              Start New Competition
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIvsHumanInterface;
