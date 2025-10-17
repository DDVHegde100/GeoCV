/**
 * Clean GeoGuessr-style game interface
 * Human vs OpenCV AI competition with synchronized Street View
 */

'use client';

import React, { useState, useEffect, useRef } from 'react';
import './GeoGuesserGame.css';

interface GameState {
  isActive: boolean;
  currentRound: number;
  totalRounds: number;
  timeRemaining: number;
  humanScore: number;
  aiScore: number;
  currentLocation: {
    lat: number;
    lon: number;
    country: string;
    city: string;
  } | null;
  gamePhase: 'loading' | 'playing' | 'guessing' | 'results' | 'finished';
}

interface GameGuess {
  lat: number;
  lon: number;
  confidence: number;
}

const GeoGuesserGame: React.FC = () => {
  const [gameState, setGameState] = useState<GameState>({
    isActive: false,
    currentRound: 0,
    totalRounds: 5,
    timeRemaining: 30,
    humanScore: 0,
    aiScore: 0,
    currentLocation: null,
    gamePhase: 'loading'
  });

  const [humanGuess, setHumanGuess] = useState<GameGuess | null>(null);
  const [aiGuess, setAiGuess] = useState<GameGuess | null>(null);
  const [roundResults, setRoundResults] = useState<any[]>([]);

  const startNewGame = async () => {
    setGameState({
      isActive: true,
      currentRound: 1,
      totalRounds: 5,
      timeRemaining: 30,
      humanScore: 0,
      aiScore: 0,
      currentLocation: null,
      gamePhase: 'loading'
    });
    
    await loadNewLocation();
  };

  const loadNewLocation = async () => {
    setGameState(prev => ({ ...prev, gamePhase: 'loading' }));
    
    try {
      const response = await fetch('http://localhost:8001/api/v1/geoguessr/new-location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty: 'mixed' })
      });
      
      const location = await response.json();
      
      setGameState(prev => ({
        ...prev,
        currentLocation: location,
        gamePhase: 'playing',
        timeRemaining: 30
      }));
      
      startTimer();
    } catch (error) {
      console.error('Failed to load location:', error);
    }
  };

  const startTimer = () => {
    const timer = setInterval(() => {
      setGameState(prev => {
        if (prev.timeRemaining <= 1) {
          clearInterval(timer);
          return { ...prev, timeRemaining: 0, gamePhase: 'guessing' };
        }
        return { ...prev, timeRemaining: prev.timeRemaining - 1 };
      });
    }, 1000);
  };

  const submitGuesses = async () => {
    if (!humanGuess || !aiGuess || !gameState.currentLocation) return;
    
    // Calculate distances and scores
    const humanDistance = calculateDistance(
      humanGuess.lat, humanGuess.lon,
      gameState.currentLocation.lat, gameState.currentLocation.lon
    );
    
    const aiDistance = calculateDistance(
      aiGuess.lat, aiGuess.lon,
      gameState.currentLocation.lat, gameState.currentLocation.lon
    );
    
    const humanPoints = calculatePoints(humanDistance);
    const aiPoints = calculatePoints(aiDistance);
    
    const result = {
      round: gameState.currentRound,
      humanGuess,
      aiGuess,
      actualLocation: gameState.currentLocation,
      humanDistance,
      aiDistance,
      humanPoints,
      aiPoints
    };
    
    setRoundResults(prev => [...prev, result]);
    setGameState(prev => ({
      ...prev,
      humanScore: prev.humanScore + humanPoints,
      aiScore: prev.aiScore + aiPoints,
      gamePhase: 'results'
    }));
  };

  const nextRound = () => {
    if (gameState.currentRound >= gameState.totalRounds) {
      setGameState(prev => ({ ...prev, gamePhase: 'finished' }));
    } else {
      setGameState(prev => ({ ...prev, currentRound: prev.currentRound + 1 }));
      setHumanGuess(null);
      setAiGuess(null);
      loadNewLocation();
    }
  };

  const calculateDistance = (lat1: number, lon1: number, lat2: number, lon2: number): number => {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  const calculatePoints = (distance: number): number => {
    if (distance < 1) return 5000;
    if (distance < 10) return 4000;
    if (distance < 100) return 3000;
    if (distance < 500) return 2000;
    if (distance < 1000) return 1000;
    return 0;
  };

  return (
    <div className="geoguessr-game">
      <div className="game-header">
        <div className="game-info">
          <h1>GeoGuessr: Human vs OpenCV AI</h1>
          <div className="round-info">
            Round {gameState.currentRound} of {gameState.totalRounds}
          </div>
          <div className="timer">
            {gameState.timeRemaining}s
          </div>
        </div>
        
        <div className="scores">
          <div className="score human-score">
            Human: {gameState.humanScore}
          </div>
          <div className="score ai-score">
            AI: {gameState.aiScore}
          </div>
        </div>
      </div>

      {!gameState.isActive ? (
        <div className="start-screen">
          <button onClick={startNewGame} className="start-button">
            Start New Game
          </button>
        </div>
      ) : (
        <div className="game-content">
          <div className="street-view-container">
            <div className="ai-view">
              <h3>🤖 OpenCV AI View</h3>
              <div className="cv-overlay-panel">
                {gameState.currentLocation && (
                  <div>CV Analysis loading...</div>
                )}
              </div>
            </div>
            
            <div className="human-view">
              <h3>👤 Human View</h3>
              <div className="street-view-panel">
                {gameState.currentLocation && (
                  <div>Street View loading...</div>
                )}
              </div>
            </div>
          </div>
          
          <div className="guess-panel">
            <button onClick={submitGuesses} disabled={!humanGuess || !aiGuess}>
              Submit Guesses
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default GeoGuesserGame;
