"""
Utility functions for GeoCV backend
"""

import os
import logging
from typing import Tuple, List, Optional
import numpy as np

def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )
    
    # Set specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('cv2').setLevel(logging.WARNING)

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth"""
    import math
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    earth_radius = 6371
    distance = earth_radius * c
    
    return distance

def cleanup_temp_files(directory: str, max_age_hours: int = 1):
    """Clean up temporary files older than max_age_hours"""
    import time
    
    if not os.path.exists(directory):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                except OSError:
                    pass
    
    return cleaned_count

def ensure_directories(*directories):
    """Ensure directories exist, create if they don't"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

class RegionDatabase:
    """Simple database of regional characteristics for geographic inference"""
    
    def __init__(self):
        self.vehicle_models = {
            'compact_car': {
                'regions': ['europe', 'asia', 'urban_areas'],
                'confidence_boost': 0.1
            },
            'truck_or_bus': {
                'regions': ['north_america', 'highways', 'commercial_areas'], 
                'confidence_boost': 0.15
            },
            'yellow_car': {
                'regions': ['new_york', 'mumbai', 'urban_centers'],
                'confidence_boost': 0.3
            }
        }
        
        self.architecture_styles = {
            'high_rise': {
                'regions': ['major_cities', 'developed_countries'],
                'confidence_boost': 0.2
            },
            'low_rise': {
                'regions': ['suburbs', 'small_towns', 'residential'],
                'confidence_boost': 0.1
            }
        }
        
        self.vegetation_types = {
            'tropical': {
                'regions': ['equatorial', 'southeast_asia', 'central_america'],
                'confidence_boost': 0.25
            },
            'temperate_forest': {
                'regions': ['northern_hemisphere', 'canada', 'scandinavia'],
                'confidence_boost': 0.2
            },
            'arid': {
                'regions': ['deserts', 'middle_east', 'australia'],
                'confidence_boost': 0.3
            }
        }
        
        self.climate_indicators = {
            'bright_sunny': ['mediterranean', 'arid_regions', 'summer'],
            'overcast': ['northern_regions', 'temperate_climate', 'winter'],
            'tropical_humidity': ['rainforests', 'southeast_asia', 'equatorial']
        }
    
    def get_region_hints(self, detected_features: List[str]) -> List[Tuple[str, float]]:
        """Get region hints with confidence scores based on detected features"""
        region_scores = {}
        
        for feature in detected_features:
            # Check vehicle models
            if feature in self.vehicle_models:
                for region in self.vehicle_models[feature]['regions']:
                    boost = self.vehicle_models[feature]['confidence_boost']
                    region_scores[region] = region_scores.get(region, 0) + boost
            
            # Check architecture
            if feature in self.architecture_styles:
                for region in self.architecture_styles[feature]['regions']:
                    boost = self.architecture_styles[feature]['confidence_boost']
                    region_scores[region] = region_scores.get(region, 0) + boost
            
            # Check vegetation
            if feature in self.vegetation_types:
                for region in self.vegetation_types[feature]['regions']:
                    boost = self.vegetation_types[feature]['confidence_boost']
                    region_scores[region] = region_scores.get(region, 0) + boost
        
        # Sort by score and return top regions
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_regions[:5]  # Top 5 regions

# Global region database instance
region_db = RegionDatabase()
