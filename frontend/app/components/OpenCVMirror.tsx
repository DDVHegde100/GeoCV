/**
 * OpenCV Mirror View - synchronized with human Street View
 * Shows real-time computer vision analysis overlays
 */

'use client';

import React, { useRef, useEffect, useState } from 'react';
import './OpenCVMirror.css';

interface CVAnalysisData {
  objects: Array<{
    type: string;
    confidence: number;
    bbox: [number, number, number, number];
    color: string;
  }>;
  features: Array<{
    type: string;
    points: Array<[number, number]>;
    confidence: number;
  }>;
  geographical_hints: string[];
  timestamp: number;
}

interface OpenCVMirrorProps {
  location: {
    lat: number;
    lon: number;
  };
  currentView: {
    heading: number;
    pitch: number;
  };
  onAnalysisUpdate?: (analysis: CVAnalysisData) => void;
  className?: string;
}

const OpenCVMirror: React.FC<OpenCVMirrorProps> = ({ 
  location, 
  currentView, 
  onAnalysisUpdate,
  className = '' 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState<CVAnalysisData | null>(null);

  // Color mapping for different object types
  const colorMap: Record<string, string> = {
    'vehicle': '#4ECDC4',
    'building': '#FF6B6B',
    'sign': '#FF9F43',
    'vegetation': '#95E1D3',
    'person': '#A8E6CF',
    'infrastructure': '#DDA0DD',
    'text': '#FFD93D'
  };

  useEffect(() => {
    if (location && currentView) {
      analyzeCurrentView();
    }
  }, [location, currentView]);

  const analyzeCurrentView = async () => {
    if (!location) return;
    
    setIsAnalyzing(true);
    
    try {
      // Get Street View image for the current view
      const streetViewUrl = `https://maps.googleapis.com/maps/api/streetview?size=640x640&location=${location.lat},${location.lon}&heading=${currentView.heading}&pitch=${currentView.pitch}&key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}`;
      
      // Load the image
      if (imageRef.current) {
        imageRef.current.src = streetViewUrl;
        
        imageRef.current.onload = () => {
          drawImageToCanvas();
          performCVAnalysis(streetViewUrl);
        };
      }
      
    } catch (error) {
      console.error('Failed to analyze current view:', error);
      setIsAnalyzing(false);
    }
  };

  const drawImageToCanvas = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size to match image
    canvas.width = 640;
    canvas.height = 640;
    
    // Draw the Street View image
    ctx.drawImage(image, 0, 0, 640, 640);
    
    // Draw CV overlays if available
    if (analysisData) {
      drawCVOverlays(ctx);
    }
  };

  const performCVAnalysis = async (imageUrl: string) => {
    try {
      const response = await fetch('http://localhost:8001/api/v1/geoguessr/analyze-view', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          location: location,
          view: currentView
        })
      });
      
      if (response.ok) {
        const analysis: CVAnalysisData = await response.json();
        setAnalysisData(analysis);
        
        if (onAnalysisUpdate) {
          onAnalysisUpdate(analysis);
        }
        
        // Redraw canvas with overlays
        drawImageToCanvas();
      }
    } catch (error) {
      console.error('CV analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const drawCVOverlays = (ctx: CanvasRenderingContext2D) => {
    if (!analysisData) return;
    
    // Draw bounding boxes for detected objects
    analysisData.objects.forEach(obj => {
      const [x, y, w, h] = obj.bbox;
      const color = colorMap[obj.type] || '#FFFFFF';
      
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '20'; // Semi-transparent fill
      ctx.lineWidth = 3;
      
      // Draw bounding box
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
      
      // Draw label
      ctx.fillStyle = color;
      ctx.font = '16px Arial';
      ctx.fillText(`${obj.type} (${(obj.confidence * 100).toFixed(0)}%)`, x, y - 5);
    });
    
    // Draw feature points
    analysisData.features.forEach(feature => {
      const color = colorMap[feature.type] || '#FFFFFF';
      ctx.fillStyle = color;
      
      feature.points.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    });
    
    // Draw scanning animation if analyzing
    if (isAnalyzing) {
      ctx.strokeStyle = '#4ECDC4';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      const time = Date.now() / 1000;
      const scanY = (Math.sin(time * 2) + 1) * 320;
      
      ctx.beginPath();
      ctx.moveTo(0, scanY);
      ctx.lineTo(640, scanY);
      ctx.stroke();
      
      ctx.setLineDash([]);
    }
  };

  return (
    <div className={`opencv-mirror ${className}`}>
      <div className="analysis-header">
        <h4>🤖 OpenCV Analysis</h4>
        <div className={`analysis-status ${isAnalyzing ? 'analyzing' : 'idle'}`}>
          {isAnalyzing ? 'Analyzing...' : 'Ready'}
        </div>
      </div>
      
      <div className="mirror-container">
        <canvas 
          ref={canvasRef}
          className="cv-canvas"
        />
        <img 
          ref={imageRef}
          className="hidden-image"
          alt="Street View for analysis"
        />
      </div>
      
      {analysisData && (
        <div className="analysis-sidebar">
          <h5>Detected Objects:</h5>
          <div className="object-list">
            {analysisData.objects.map((obj, index) => (
              <div key={index} className="object-item">
                <span 
                  className="object-color" 
                  style={{ backgroundColor: colorMap[obj.type] }}
                ></span>
                <span>{obj.type}</span>
                <span className="confidence">{(obj.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
          
          {analysisData.geographical_hints.length > 0 && (
            <div className="geo-hints">
              <h5>Geographic Clues:</h5>
              {analysisData.geographical_hints.map((hint, index) => (
                <div key={index} className="hint-item">{hint}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default OpenCVMirror;
