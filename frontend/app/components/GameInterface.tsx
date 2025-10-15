'use client';

import { useState, useEffect } from 'react';
import { MapPin, Clock, Brain, User } from 'lucide-react';

interface GameInterfaceProps {
  gameId: string;
  imageFile: File | null;
  onGameEnd: () => void;
}

interface GameState {
  game_id: string;
  status: string;
  ai_confidence_display: string;
  ai_features?: {
    detections_count: number;
    confidence_level: string;
    processing_time: number;
    feature_summary: Record<string, number>;
    suggested_regions: string[];
  };
  ai_guess?: {
    lat: number;
    lon: number;
  };
}

export default function GameInterface({ gameId, imageFile, onGameEnd }: GameInterfaceProps) {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [userGuess, setUserGuess] = useState<{ lat: string; lon: string }>({ lat: '', lon: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [gameResult, setGameResult] = useState<any>(null);

  useEffect(() => {
    const pollGameState = async () => {
      try {
        const response = await fetch(`/api/backend/games/${gameId}/status`);
        if (response.ok) {
          const state = await response.json();
          setGameState(state);
        }
      } catch (error) {
        console.error('Failed to fetch game state:', error);
      }
    };

    const interval = setInterval(pollGameState, 1000);
    pollGameState(); // Initial call

    return () => clearInterval(interval);
  }, [gameId]);

  const handleSubmitGuess = async () => {
    if (!userGuess.lat || !userGuess.lon) return;

    setIsSubmitting(true);
    try {
      const response = await fetch(`/api/backend/games/${gameId}/guess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: parseFloat(userGuess.lat),
          lon: parseFloat(userGuess.lon),
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setGameResult(result.result);
      }
    } catch (error) {
      console.error('Failed to submit guess:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (gameResult) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
            <h1 className="text-4xl font-bold mb-8">Game Complete!</h1>
            
            <div className={`p-6 rounded-lg mb-6 ${
              gameResult.winner === 'ai' ? 'winner-ai' : 
              gameResult.winner === 'human' ? 'winner-human' : 'winner-tie'
            }`}>
              <h2 className="text-2xl font-bold mb-4">
                Winner: {gameResult.winner === 'ai' ? '🤖 AI' : 
                        gameResult.winner === 'human' ? '👤 Human' : '🤝 Tie'}
              </h2>
              
              <div className="grid md:grid-cols-2 gap-6 text-left">
                <div className="bg-white/50 p-4 rounded-lg">
                  <h3 className="font-bold text-lg mb-2">🤖 AI Performance</h3>
                  <p>Distance Error: {gameResult.ai_distance_error.toFixed(2)} km</p>
                  <p>Score: {gameResult.ai_score}</p>
                </div>
                
                <div className="bg-white/50 p-4 rounded-lg">
                  <h3 className="font-bold text-lg mb-2">👤 Human Performance</h3>
                  <p>Distance Error: {gameResult.human_distance_error.toFixed(2)} km</p>
                  <p>Score: {gameResult.human_score}</p>
                </div>
              </div>
            </div>

            <button
              onClick={onGameEnd}
              className="px-8 py-4 bg-geocv-blue text-white rounded-lg font-bold text-lg hover:bg-blue-700 transition-colors"
            >
              Play Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-center mb-8">GeoCV Challenge</h1>
          
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Panel - AI Analysis */}
            <div className="cv-analysis-panel">
              <h2 className="text-2xl font-bold mb-4 flex items-center">
                <Brain className="w-6 h-6 mr-2" />
                AI Computer Vision Analysis
              </h2>
              
              {/* Image Display */}
              {imageFile && (
                <div className="mb-6">
                  <img 
                    src={URL.createObjectURL(imageFile)} 
                    alt="Analysis Target" 
                    className="w-full rounded-lg shadow-lg max-h-64 object-cover"
                  />
                </div>
              )}
              
              {/* AI Status */}
              <div className="bg-white rounded-lg p-4 mb-4">
                <h3 className="font-bold mb-2">AI Status</h3>
                <p className="text-sm text-gray-600">
                  {gameState?.ai_confidence_display || 'Initializing...'}
                </p>
                
                {gameState?.status === 'analyzing' && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-geocv-blue h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* AI Features */}
              {gameState?.ai_features && (
                <div className="bg-white rounded-lg p-4 mb-4">
                  <h3 className="font-bold mb-2">Detected Features</h3>
                  <div className="space-y-2">
                    <p><strong>Detections:</strong> {gameState.ai_features.detections_count}</p>
                    <p><strong>Confidence:</strong> {gameState.ai_features.confidence_level}</p>
                    <p><strong>Processing Time:</strong> {gameState.ai_features.processing_time.toFixed(2)}s</p>
                    
                    {gameState.ai_features.suggested_regions.length > 0 && (
                      <div>
                        <strong>Suggested Regions:</strong>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {gameState.ai_features.suggested_regions.map((region, idx) => (
                            <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                              {region}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* AI Guess */}
              {gameState?.ai_guess && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h3 className="font-bold text-green-800 mb-2">🎯 AI's Guess</h3>
                  <p className="text-green-700">
                    Lat: {gameState.ai_guess.lat.toFixed(4)}, 
                    Lon: {gameState.ai_guess.lon.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
            
            {/* Right Panel - Human Input */}
            <div className="bg-white rounded-xl shadow-xl p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center">
                <User className="w-6 h-6 mr-2" />
                Your Turn
              </h2>
              
              <div className="space-y-4">
                <p className="text-gray-600">
                  Analyze the image and make your best guess for the location coordinates.
                </p>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Latitude
                  </label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g., 40.7128"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-geocv-blue focus:border-transparent"
                    value={userGuess.lat}
                    onChange={(e) => setUserGuess(prev => ({ ...prev, lat: e.target.value }))}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Longitude
                  </label>
                  <input
                    type="number"
                    step="any"
                    placeholder="e.g., -74.0060"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-geocv-blue focus:border-transparent"
                    value={userGuess.lon}
                    onChange={(e) => setUserGuess(prev => ({ ...prev, lon: e.target.value }))}
                  />
                </div>
                
                <button
                  onClick={handleSubmitGuess}
                  disabled={!userGuess.lat || !userGuess.lon || isSubmitting || gameState?.status !== 'waiting_human'}
                  className={`w-full py-3 rounded-lg font-bold transition-all ${
                    userGuess.lat && userGuess.lon && !isSubmitting && gameState?.status === 'waiting_human'
                      ? 'bg-geocv-blue text-white hover:bg-blue-700 shadow-lg hover:shadow-xl transform hover:scale-105'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {isSubmitting ? (
                    <span className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Submitting...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center">
                      <MapPin className="w-5 h-5 mr-2" />
                      Submit My Guess
                    </span>
                  )}
                </button>
                
                {gameState?.status !== 'waiting_human' && !gameResult && (
                  <p className="text-sm text-center text-gray-500">
                    {gameState?.status === 'analyzing' ? 'AI is still analyzing...' : 'Waiting for game to be ready...'}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
