/**
 * Main GeoGuessr Game Component
 * Manages game state, timer, scoring, and synchronized dual-panel interface
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import StreetView from './StreetView';
import OpenCVMirror from './OpenCVMirror';
import './GeoGuesserGame.css';

interface GameState {
  isActive: boolean;
  currentRound: number;
  totalRounds: number;
  timeRemaining: number;
  score: number;
  roundScores: number[];
  gameId: string;
}

interface Location {
  lat: number;
  lon: number;
  id: string;
  country: string;
  city: string;
}

interface ViewState {
  heading: number;
  pitch: number;
  position?: google.maps.LatLng;
}

const GeoGuesserGame: React.FC = () => {
  // Game state
  const [gameState, setGameState] = useState<GameState>({
    isActive: false,
    currentRound: 0,
    totalRounds: 5,
    timeRemaining: 30,
    score: 0,
    roundScores: [],
    gameId: ''
  });

  // Location and view synchronization
  const [currentLocation, setCurrentLocation] = useState<Location | null>(null);
  const [currentView, setCurrentView] = useState<ViewState>({
    heading: 0,
    pitch: 0
  });

  // Game mechanics
  const [isLoading, setIsLoading] = useState(false);
  const [gameMessage, setGameMessage] = useState('Ready to start GeoGuessr?');
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Timer effect
  useEffect(() => {
    if (gameState.isActive && gameState.timeRemaining > 0) {
      timerRef.current = setTimeout(() => {
        setGameState(prev => ({
          ...prev,
          timeRemaining: prev.timeRemaining - 1
        }));
      }, 1000);
    } else if (gameState.timeRemaining === 0 && gameState.isActive) {
      handleTimeUp();
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [gameState.timeRemaining, gameState.isActive]);

  const startNewGame = async () => {
    setIsLoading(true);
    setGameMessage('Starting new game...');
    
    try {
      setGameState({
        isActive: true,
        currentRound: 1,
        totalRounds: 5,
        timeRemaining: 30,
        score: 0,
        roundScores: [],
        gameId: `game_${Date.now()}`
      });

      await loadNewLocation();
      setGameMessage('Game started! Find this location!');
    } catch (error) {
      setGameMessage('Failed to start game. Please try again.');
      console.error('Failed to start game:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadNewLocation = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8001/geoguessr/new-location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty: 'mixed' })
      });

      if (response.ok) {
        const locationData = await response.json();
        setCurrentLocation({
          lat: locationData.lat,
          lon: locationData.lon,
          id: locationData.id,
          country: locationData.country,
          city: locationData.city
        });

        // Reset view for new location
        setCurrentView({
          heading: 0,
          pitch: 0
        });

        setGameMessage(`Round ${gameState.currentRound} - Find this location!`);
      } else {
        throw new Error('Failed to load location');
      }
    } catch (error) {
      setGameMessage('Failed to load location. Please try again.');
      console.error('Failed to load location:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTimeUp = () => {
    setGameState(prev => ({ ...prev, isActive: false }));
    setGameMessage('Time\'s up! No points for this round.');
    
    setTimeout(() => {
      nextRound();
    }, 3000);
  };

  const handleGuessSubmit = (guessLat: number, guessLon: number) => {
    if (!currentLocation || !gameState.isActive) return;

    // Calculate distance and score
    const distance = calculateDistance(
      currentLocation.lat, 
      currentLocation.lon, 
      guessLat, 
      guessLon
    );

    const roundScore = calculateScore(distance);
    
    setGameState(prev => ({
      ...prev,
      isActive: false,
      score: prev.score + roundScore,
      roundScores: [...prev.roundScores, roundScore]
    }));

    setGameMessage(
      `Round ${gameState.currentRound} complete! Distance: ${distance.toFixed(1)}km, Score: ${roundScore}`
    );

    setTimeout(() => {
      nextRound();
    }, 3000);
  };

  const nextRound = () => {
    if (gameState.currentRound < gameState.totalRounds) {
      setGameState(prev => ({
        ...prev,
        currentRound: prev.currentRound + 1,
        timeRemaining: 30,
        isActive: true
      }));
      loadNewLocation();
    } else {
      endGame();
    }
  };

  const endGame = () => {
    setGameMessage(
      `Game Over! Final Score: ${gameState.score}/${gameState.totalRounds * 5000}`
    );
    setGameState(prev => ({ ...prev, isActive: false }));
  };

  const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371; // Earth's radius in kilometers
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const toRad = (degrees: number): number => degrees * (Math.PI / 180);

  const calculateScore = (distance: number): number => {
    if (distance === 0) return 5000;
    return Math.max(0, Math.round(5000 * Math.exp(-distance / 1000)));
  };

  const handleViewChange = (newView: ViewState) => {
    setCurrentView(newView);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="geoguessr-game">
      {/* Game Header */}
      <div className="game-header">
        <div className="game-info">
          <h1>🌍 GeoGuessr AI vs Human</h1>
          <div className="game-stats">
            <span className="round-counter">
              Round: {gameState.currentRound}/{gameState.totalRounds}
            </span>
            <span className="score">Score: {gameState.score}</span>
            <span className={`timer ${gameState.timeRemaining <= 10 ? 'warning' : ''}`}>
              ⏱️ {formatTime(gameState.timeRemaining)}
            </span>
          </div>
        </div>
        
        <div className="game-controls">
          {!gameState.isActive && gameState.currentRound === 0 && (
            <button 
              onClick={startNewGame} 
              disabled={isLoading}
              className="start-button"
            >
              {isLoading ? 'Loading...' : 'Start Game'}
            </button>
          )}
        </div>
      </div>

      {/* Game Message */}
      <div className="game-message">
        {gameMessage}
      </div>

      {/* Dual Panel Interface */}
      {currentLocation && (
        <div className="dual-panel-container">
          {/* Human Player Panel - Left */}
          <div className="panel human-panel">
            <div className="panel-header">
              <h3>🧠 Human Player</h3>
              <p>Navigate and explore to find clues</p>
            </div>
            <div className="panel-content">
              <StreetView
                location={currentLocation}
                onViewChange={handleViewChange}
                onGuessSubmit={handleGuessSubmit}
                gameActive={gameState.isActive}
              />
            </div>
          </div>

          {/* AI Analysis Panel - Right */}
          <div className="panel ai-panel">
            <div className="panel-header">
              <h3>🤖 OpenCV AI</h3>
              <p>Computer vision analysis in real-time</p>
            </div>
            <div className="panel-content">
              <OpenCVMirror
                location={currentLocation}
                currentView={currentView}
                onAnalysisUpdate={(analysis) => {
                  // Handle AI analysis updates here
                  console.log('AI Analysis:', analysis);
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Round Scores Display */}
      {gameState.roundScores.length > 0 && (
        <div className="round-scores">
          <h4>Round Scores:</h4>
          <div className="scores-list">
            {gameState.roundScores.map((score, index) => (
              <span key={index} className="round-score">
                Round {index + 1}: {score}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GeoGuesserGame;
