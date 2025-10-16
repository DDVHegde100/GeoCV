'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import '../styles/StreetViewPlayer.css';

interface StreetViewPlayerProps {
  location: {
    lat: number;
    lon: number;
    panorama_id?: string;
    image_url?: string;
    metadata?: any;
  };
  onNavigationChange?: (heading: number, pitch: number, zoom: number) => void;
  className?: string;
}

export default function StreetViewPlayer({ location, onNavigationChange, className = '' }: StreetViewPlayerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [heading, setHeading] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, heading: 0, pitch: 0 });
  
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (location?.image_url) {
      setIsLoading(false);
    } else {
      setError('No Street View image available for this location');
      setIsLoading(false);
    }
  }, [location]);

  useEffect(() => {
    onNavigationChange?.(heading, pitch, zoom);
  }, [heading, pitch, zoom, onNavigationChange]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX,
      y: e.clientY,
      heading: heading,
      pitch: pitch
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;

    // Calculate new heading and pitch based on drag distance
    const sensitivity = 0.5;
    const newHeading = (dragStart.heading + (deltaX * sensitivity)) % 360;
    const newPitch = Math.max(-90, Math.min(90, dragStart.pitch + (deltaY * sensitivity)));

    setHeading(newHeading < 0 ? newHeading + 360 : newHeading);
    setPitch(newPitch);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const adjustHeading = (delta: number) => {
    setHeading(prev => {
      const newHeading = (prev + delta) % 360;
      return newHeading < 0 ? newHeading + 360 : newHeading;
    });
  };

  const adjustPitch = (delta: number) => {
    setPitch(prev => Math.max(-90, Math.min(90, prev + delta)));
  };

  const adjustZoom = (delta: number) => {
    setZoom(prev => Math.max(0.5, Math.min(3, prev + delta)));
  };

  const resetView = () => {
    setHeading(0);
    setPitch(0);
    setZoom(1);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement && containerRef.current) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else if (document.fullscreenElement) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          adjustHeading(-10);
          e.preventDefault();
          break;
        case 'ArrowRight':
          adjustHeading(10);
          e.preventDefault();
          break;
        case 'ArrowUp':
          adjustPitch(10);
          e.preventDefault();
          break;
        case 'ArrowDown':
          adjustPitch(-10);
          e.preventDefault();
          break;
        case '=':
        case '+':
          adjustZoom(0.1);
          e.preventDefault();
          break;
        case '-':
          adjustZoom(-0.1);
          e.preventDefault();
          break;
        case 'r':
        case 'R':
          resetView();
          e.preventDefault();
          break;
        case 'f':
        case 'F':
          toggleFullscreen();
          e.preventDefault();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  if (error) {
    return (
      <div className={`bg-gray-100 rounded-lg flex items-center justify-center ${className}`}>
        <div className="text-center p-8">
          <div className="text-red-500 text-6xl mb-4">📍</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Street View Unavailable</h3>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  // Dynamic styles for transform (unavoidable inline styles for transform calculations)
  const dynamicImageStyle = {
    transform: `scale(${zoom}) rotateY(${heading}deg) rotateX(${pitch}deg)`,
    transformOrigin: 'center center'
  };

  return (
    <div 
      ref={containerRef}
      className={`streetview-container ${className} ${isFullscreen ? 'fullscreen' : ''}`}
    >
      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900 flex items-center justify-center z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-white text-lg">Loading Street View...</p>
          </div>
        </div>
      )}

      {/* Street View Image */}
      <div 
        className="relative w-full h-full overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <img
          ref={imageRef}
          src={location?.image_url}
          alt="Street View"
          className={`streetview-image ${isDragging ? 'dragging' : ''}`}
          style={dynamicImageStyle}
          onLoad={() => setIsLoading(false)}
          onError={() => {
            setError('Failed to load Street View image');
            setIsLoading(false);
          }}
          draggable={false}
        />
        
        {/* Panoramic View Overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-black/20 via-transparent to-black/20 pointer-events-none" />
      </div>

      {/* Navigation Controls */}
      <div className="absolute top-4 right-4 flex flex-col space-y-2">
        {/* Directional Controls */}
        <div className="bg-black/50 rounded-lg p-2 grid grid-cols-3 gap-1">
          <div></div>
          <button
            onClick={() => adjustPitch(10)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Look Up"
          >
            <ChevronUp className="w-4 h-4 text-white" />
          </button>
          <div></div>
          
          <button
            onClick={() => adjustHeading(-15)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Turn Left"
          >
            <ChevronLeft className="w-4 h-4 text-white" />
          </button>
          <button
            onClick={resetView}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4 text-white" />
          </button>
          <button
            onClick={() => adjustHeading(15)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Turn Right"
          >
            <ChevronRight className="w-4 h-4 text-white" />
          </button>
          
          <div></div>
          <button
            onClick={() => adjustPitch(-10)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Look Down"
          >
            <ChevronDown className="w-4 h-4 text-white" />
          </button>
          <div></div>
        </div>

        {/* Zoom Controls */}
        <div className="bg-black/50 rounded-lg p-2 flex flex-col space-y-1">
          <button
            onClick={() => adjustZoom(0.1)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4 text-white" />
          </button>
          <div className="text-white text-xs text-center py-1">
            {Math.round(zoom * 100)}%
          </div>
          <button
            onClick={() => adjustZoom(-0.1)}
            className="p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4 text-white" />
          </button>
        </div>

        {/* Fullscreen Toggle */}
        <button
          onClick={toggleFullscreen}
          className="p-3 bg-black/50 hover:bg-black/70 rounded-lg transition-colors"
          title="Toggle Fullscreen"
        >
          <Maximize2 className="w-4 h-4 text-white" />
        </button>
      </div>

      {/* View Information */}
      <div className="absolute bottom-4 left-4 bg-black/50 rounded-lg p-3 text-white text-sm">
        <div className="space-y-1">
          <div>Heading: {Math.round(heading)}°</div>
          <div>Pitch: {Math.round(pitch)}°</div>
          <div>Zoom: {Math.round(zoom * 100)}%</div>
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 bg-black/50 rounded-lg p-3 text-white text-xs max-w-xs">
        <div className="space-y-1">
          <div className="font-semibold mb-2">Controls:</div>
          <div>• Drag to look around</div>
          <div>• Arrow keys to navigate</div>
          <div>• +/- to zoom</div>
          <div>• R to reset view</div>
          <div>• F for fullscreen</div>
        </div>
      </div>
    </div>
  );
}
