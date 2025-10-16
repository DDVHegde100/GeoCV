'use client';

import React, { useEffect, useState, useRef } from 'react';
import './TeslaStyleOverlay.css';

interface DetectedObject {
  type: string;
  confidence: number;
  bounding_box: [number, number, number, number]; // [x, y, width, height]
  geographic_hints?: string[];
}

interface OverlayData {
  detected_objects: DetectedObject[];
  confidence_scores: Record<string, number>;
  overlays: Record<string, any>;
  timestamp: string;
}

interface TeslaStyleOverlayProps {
  imageUrl: string;
  overlayData: OverlayData | null;
  isAnalyzing: boolean;
  aiStatus: string;
  gameId: string;
}

const TeslaStyleOverlay: React.FC<TeslaStyleOverlayProps> = ({
  imageUrl,
  overlayData,
  isAnalyzing,
  aiStatus,
  gameId
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  
  // Color coding for different object types (Tesla-style)
  const colorMap: Record<string, string> = {
    'vehicle': '#4ECDC4',
    'car': '#4ECDC4',
    'truck': '#4ECDC4',
    'building': '#FF6B6B',
    'architecture': '#FF6B6B',
    'text': '#FF9F43',
    'sign': '#FF9F43',
    'infrastructure': '#A8E6CF',
    'road': '#A8E6CF',
    'vegetation': '#95E1D3',
    'tree': '#95E1D3',
    'default': '#FFFFFF'
  };

  useEffect(() => {
    if (imageLoaded && overlayData && canvasRef.current && imageRef.current) {
      drawOverlays();
    }
  }, [overlayData, imageLoaded]);

  const drawOverlays = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !overlayData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match image
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes and labels for detected objects
    overlayData.detected_objects.forEach((obj) => {
      const [x, y, width, height] = obj.bounding_box;
      const color = colorMap[obj.type.toLowerCase()] || colorMap.default;
      
      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, width, height);
      
      // Draw confidence indicator
      const confidenceHeight = 6;
      const confidenceWidth = width * obj.confidence;
      ctx.fillStyle = color;
      ctx.fillRect(x, y - confidenceHeight - 2, confidenceWidth, confidenceHeight);
      
      // Draw label background
      const label = `${obj.type} ${Math.round(obj.confidence * 100)}%`;
      const labelWidth = ctx.measureText(label).width + 8;
      const labelHeight = 20;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(x, y - labelHeight - confidenceHeight - 4, labelWidth, labelHeight);
      
      // Draw label text
      ctx.fillStyle = color;
      ctx.font = '14px "SF Mono", "Monaco", "Inconsolata", monospace';
      ctx.fillText(label, x + 4, y - confidenceHeight - 10);
      
      // Draw geographic hints if available
      if (obj.geographic_hints && obj.geographic_hints.length > 0) {
        const hintText = obj.geographic_hints[0];
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = '12px "SF Mono", "Monaco", "Inconsolata", monospace';
        ctx.fillText(hintText, x, y + height + 15);
      }
    });

    // Draw scanning lines effect (Tesla-style)
    if (isAnalyzing) {
      ctx.strokeStyle = 'rgba(76, 205, 196, 0.3)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 10]);
      
      const time = Date.now() % 3000;
      const scanLine = (time / 3000) * canvas.height;
      
      ctx.beginPath();
      ctx.moveTo(0, scanLine);
      ctx.lineTo(canvas.width, scanLine);
      ctx.stroke();
    }
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  return (
    <div className="tesla-overlay-container">
      <div className="image-canvas-wrapper">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Street View Analysis"
          className="streetview-image"
          onLoad={handleImageLoad}
          crossOrigin="anonymous"
        />
        <canvas
          ref={canvasRef}
          className="overlay-canvas"
        />
      </div>
      
      {/* AI Status Panel (Tesla-style) */}
      <div className="ai-status-panel">
        <div className="status-header">
          <div className="ai-indicator">
            <div className={`ai-dot ${isAnalyzing ? 'analyzing' : 'ready'}`}></div>
            <span className="ai-label">AI VISION</span>
          </div>
          <div className="confidence-display">
            {overlayData?.confidence_scores?.overall && (
              <span className="confidence-value">
                {Math.round(overlayData.confidence_scores.overall * 100)}% CONFIDENCE
              </span>
            )}
          </div>
        </div>
        
        <div className="status-message">
          {aiStatus}
        </div>
        
        {/* Object Detection Summary */}
        {overlayData?.detected_objects && overlayData.detected_objects.length > 0 && (
          <div className="detection-summary">
            <div className="summary-header">DETECTED OBJECTS</div>
            <div className="object-list">
              {overlayData.detected_objects.map((obj, index) => (
                <div key={index} className="object-item">
                  <div 
                    className={`object-color-indicator color-${obj.type.toLowerCase()}`}
                  ></div>
                  <span className="object-name">{obj.type}</span>
                  <span className="object-confidence">{Math.round(obj.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Legend */}
        <div className="color-legend">
          <div className="legend-title">OBJECT CLASSIFICATION</div>
          <div className="legend-items">
            <div className="legend-item">
              <div className="legend-color vehicles"></div>
              <span>Vehicles</span>
            </div>
            <div className="legend-item">
              <div className="legend-color buildings"></div>
              <span>Buildings</span>
            </div>
            <div className="legend-item">
              <div className="legend-color text-signs"></div>
              <span>Text/Signs</span>
            </div>
            <div className="legend-item">
              <div className="legend-color infrastructure"></div>
              <span>Infrastructure</span>
            </div>
            <div className="legend-item">
              <div className="legend-color vegetation"></div>
              <span>Vegetation</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeslaStyleOverlay;
