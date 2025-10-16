'use client';

import { useState, useRef } from 'react';
import { Upload, Camera, MapPin, Zap, Target } from 'lucide-react';
import axios from 'axios';
import AIvsHumanInterface from './components/AIvsHumanInterface';

interface GameMode {
  id: string;
  name: string;
  description: string;
  time_limit?: number;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [gameMode, setGameMode] = useState<string>('classic');
  const [gameId, setGameId] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [actualLocation, setActualLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [showLocationInput, setShowLocationInput] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const gameModes: GameMode[] = [
    {
      id: 'classic',
      name: 'Classic Mode',
      description: 'Unlimited time for both AI and human player'
    },
    {
      id: 'speed',
      name: 'Speed Mode', 
      description: 'AI gets 30s, you get AI time + 10 seconds',
      time_limit: 30
    },
    {
      id: 'blitz',
      name: 'Blitz Mode',
      description: 'Quick rounds - 15 seconds for AI analysis',
      time_limit: 15
    },
    {
      id: 'training',
      name: 'Training Mode',
      description: 'Watch AI reasoning process step by step'
    }, 
    {
      id: 'expert',
      name: 'Expert Mode',
      description: 'Both AI and human get 10 seconds each',
      time_limit: 10
    },
    {
      id: 'streetview',
      name: 'Street View Challenge',
      description: 'Play with AI-generated Street View locations',
      time_limit: 60
    }
  ];

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    }
  };

  const handleStartGame = async () => {
    if (gameMode === 'streetview') {
      return handleStartStreetViewGame();
    }
    
    if (!selectedFile) return;

    setIsStarting(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('game_mode', gameMode);
      
      if (actualLocation) {
        formData.append('actual_lat', actualLocation.lat.toString());
        formData.append('actual_lon', actualLocation.lon.toString());
      }

      const response = await axios.post('http://localhost:8000/api/v1/games/start', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setGameId(response.data.game_id);
    } catch (error) {
      console.error('Failed to start game:', error);
      alert('Failed to start game. Please try again.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStartStreetViewGame = async () => {
    setIsStarting(true);
    try {
      const response = await axios.post('http://localhost:8000/api/v1/games/start-streetview', {
        difficulty: 'medium' // Default to medium difficulty
      });

      setGameId(response.data.game_id);
    } catch (error) {
      console.error('Failed to start Street View game:', error);
      alert('Failed to start Street View game. Please try again.');
    } finally {
      setIsStarting(false);
    }
  };

  const resetGame = () => {
    setGameId(null);
    setSelectedFile(null);
    setActualLocation(null);
    setShowLocationInput(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <AIvsHumanInterface />
  );

  // Legacy UI for reference
  const legacyUI = (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Camera className="w-12 h-12 text-geocv-blue mr-3" />
            <h1 className="text-5xl font-bold text-gray-900">GeoCV</h1>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Challenge an AI that uses computer vision to guess locations from street view images.
            Upload an image and see if you can beat the machine!
          </p>
        </div>

        {/* Game Setup */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            
            {/* Step 1: Upload Image (Skip for Street View mode) */}
            {gameMode !== 'streetview' && (
              <div className="mb-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center">
                  <Upload className="w-6 h-6 mr-2 text-geocv-blue" />
                  Step 1: Upload Street View Image
                </h2>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-geocv-blue transition-colors">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  {selectedFile ? (
                    <div className="space-y-2">
                      <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg flex items-center justify-center">
                        <img 
                          src={URL.createObjectURL(selectedFile)} 
                          alt="Preview" 
                          className="max-w-full max-h-full object-contain rounded-lg"
                        />
                      </div>
                      <p className="text-geocv-blue font-medium">{selectedFile.name}</p>
                      <p className="text-sm text-gray-500">Click to change image</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Upload className="w-16 h-16 mx-auto text-gray-400" />
                      <p className="text-lg font-medium text-gray-600">
                        Choose a street view image
                      </p>
                      <p className="text-sm text-gray-500">
                        PNG, JPG up to 10MB
                      </p>
                    </div>
                  )}
                </label>
              </div>
              </div>
            )}

            {/* Street View Mode Notice */}
            {gameMode === 'streetview' && (
              <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="text-lg font-semibold text-blue-800 mb-2">Street View Challenge Mode</h3>
                <p className="text-blue-700">
                  We'll generate a random Street View location for you! No need to upload an image - 
                  the AI will analyze the same Street View as you and compete to guess the location.
                </p>
              </div>
            )}

            {/* Step 2: Select Game Mode */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold mb-4 flex items-center">
                <Zap className="w-6 h-6 mr-2 text-geocv-blue" />
                {gameMode === 'streetview' ? 'Step 1: Choose Game Mode' : 'Step 2: Choose Game Mode'}
              </h2>
              
              <div className="grid md:grid-cols-2 gap-4">
                {gameModes.map((mode) => (
                  <label key={mode.id} className="cursor-pointer">
                    <input
                      type="radio"
                      name="gameMode"
                      value={mode.id}
                      checked={gameMode === mode.id}
                      onChange={(e) => setGameMode(e.target.value)}
                      className="sr-only"
                    />
                    <div className={`p-4 rounded-lg border-2 transition-all ${
                      gameMode === mode.id 
                        ? 'border-geocv-blue bg-blue-50 ring-2 ring-geocv-blue ring-opacity-20' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}>
                      <h3 className="font-bold text-lg flex items-center">
                        {mode.name}
                        {mode.time_limit && (
                          <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-xs rounded-full">
                            {mode.time_limit}s
                          </span>
                        )}
                      </h3>
                      <p className="text-gray-600 text-sm mt-1">{mode.description}</p>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Step 3: Optional Actual Location (Only for non-Street View modes) */}
            {gameMode !== 'streetview' && (
              <div className="mb-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center">
                  <MapPin className="w-6 h-6 mr-2 text-geocv-blue" />
                  Step 3: Actual Location (Optional)
                </h2>
              
              <div className="flex items-center space-x-4">
                <button
                  type="button"
                  onClick={() => setShowLocationInput(!showLocationInput)}
                  className={`px-4 py-2 rounded-lg border-2 transition-all ${
                    showLocationInput 
                      ? 'border-geocv-blue bg-blue-50 text-geocv-blue' 
                      : 'border-gray-300 text-gray-600 hover:border-gray-400'
                  }`}
                >
                  {showLocationInput ? 'Hide' : 'Add'} Known Location
                </button>
                <p className="text-sm text-gray-500">
                  For accurate scoring (if you know where the image was taken)
                </p>
              </div>

              {showLocationInput && (
                <div className="mt-4 flex space-x-4">
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Latitude
                    </label>
                    <input
                      type="number"
                      step="any"
                      placeholder="e.g., 40.7128"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-geocv-blue focus:border-transparent"
                      value={actualLocation?.lat || ''}
                      onChange={(e) => setActualLocation(prev => ({
                        ...prev,
                        lat: parseFloat(e.target.value) || 0,
                        lon: prev?.lon || 0
                      }))}
                    />
                  </div>
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Longitude
                    </label>
                    <input
                      type="number"
                      step="any"
                      placeholder="e.g., -74.0060"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-geocv-blue focus:border-transparent"
                      value={actualLocation?.lon || ''}
                      onChange={(e) => setActualLocation(prev => ({
                        ...prev,
                        lat: prev?.lat || 0,
                        lon: parseFloat(e.target.value) || 0
                      }))}
                    />
                  </div>
                </div>
              )}
              </div>
            )}

            {/* Start Game Button */}
            <div className="text-center">
              <button
                onClick={handleStartGame}
                disabled={gameMode !== 'streetview' ? (!selectedFile || isStarting) : isStarting}
                className={`px-8 py-4 rounded-lg font-bold text-lg transition-all ${
                  (gameMode === 'streetview' || selectedFile) && !isStarting
                    ? 'bg-geocv-blue text-white hover:bg-blue-700 shadow-lg hover:shadow-xl transform hover:scale-105'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {isStarting ? (
                  <span className="flex items-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    {gameMode === 'streetview' ? 'Finding Location...' : 'Starting Game...'}
                  </span>
                ) : (
                  <span className="flex items-center">
                    <Target className="w-6 h-6 mr-2" />
                    {gameMode === 'streetview' ? 'Start Street View Challenge' : 'Start GeoCV Challenge'}
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-16 grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="text-center">
            <div className="w-16 h-16 bg-geocv-blue rounded-full flex items-center justify-center mx-auto mb-4">
              <Camera className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold mb-2">Computer Vision Analysis</h3>
            <p className="text-gray-600">
              AI analyzes vehicles, architecture, vegetation, roads, and environmental cues
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-geocv-green rounded-full flex items-center justify-center mx-auto mb-4">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold mb-2">Real-Time Competition</h3>
            <p className="text-gray-600">
              Watch the AI work through its analysis while you make your own guess
            </p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-geocv-purple rounded-full flex items-center justify-center mx-auto mb-4">
              <Target className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold mb-2">Scoring & Analysis</h3>
            <p className="text-gray-600">
              Get detailed breakdowns of what the AI detected and how it made its guess
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
