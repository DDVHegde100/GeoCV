/**
 * Google Street View component for GeoGuessr
 * Provides interactive Street View navigation
 */

'use client';

import React, { useRef, useEffect, useState } from 'react';

interface StreetViewProps {
  location: {
    lat: number;
    lon: number;
  };
  onViewChange?: (view: { heading: number; pitch: number; position?: google.maps.LatLng }) => void;
  onGuessSubmit?: (lat: number, lon: number) => void;
  gameActive?: boolean;
  className?: string;
}

declare global {
  interface Window {
    google: typeof google;
    initStreetView: () => void;
  }
}

const StreetView: React.FC<StreetViewProps> = ({ 
  location, 
  onViewChange, 
  onGuessSubmit, 
  gameActive = false, 
  className = '' 
}) => {
  const streetViewRef = useRef<HTMLDivElement>(null);
  const panoramaRef = useRef<google.maps.StreetViewPanorama | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const loadGoogleMaps = () => {
      if (window.google && window.google.maps) {
        initializeStreetView();
        return;
      }

      // Load Google Maps JavaScript API
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}&libraries=geometry`;
      script.async = true;
      script.defer = true;
      
      window.initStreetView = () => {
        initializeStreetView();
      };
      
      script.onload = () => {
        if (window.google) {
          initializeStreetView();
        }
      };
      
      document.head.appendChild(script);
    };

    loadGoogleMaps();
  }, []);

  useEffect(() => {
    if (panoramaRef.current && location) {
      // Update Street View location
      const position = new google.maps.LatLng(location.lat, location.lon);
      panoramaRef.current.setPosition(position);
    }
  }, [location]);

  const initializeStreetView = () => {
    if (!streetViewRef.current || !window.google) return;

    const position = new google.maps.LatLng(location.lat, location.lon);
    
    const panorama = new google.maps.StreetViewPanorama(streetViewRef.current, {
      position: position,
      pov: {
        heading: 0,
        pitch: 0
      },
      zoom: 1,
      addressControl: false,
      linksControl: true,
      panControl: true,
      enableCloseButton: false,
      showRoadLabels: false
    });

    panoramaRef.current = panorama;
    setIsLoaded(true);

    // Listen for view changes
    if (onViewChange) {
      panorama.addListener('position_changed', () => {
        const pos = panorama.getPosition();
        const pov = panorama.getPov();
        if (pos && pov) {
          onViewChange({
            heading: pov.heading || 0,
            pitch: pov.pitch || 0,
            position: pos
          });
        }
      });

      panorama.addListener('pov_changed', () => {
        const pos = panorama.getPosition();
        const pov = panorama.getPov();
        if (pos && pov) {
          onViewChange({
            heading: pov.heading || 0,
            pitch: pov.pitch || 0,
            position: pos
          });
        }
      });
    }
  };

  return (
    <div className={`street-view-container ${className}`}>
      <div 
        ref={streetViewRef} 
        className="street-view-panorama"
      />
      {!isLoaded && (
        <div className="street-view-loading">
          Loading Street View...
        </div>
      )}
    </div>
  );
};

export default StreetView;
