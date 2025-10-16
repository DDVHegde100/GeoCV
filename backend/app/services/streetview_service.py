"""
Google Street View API Service
Handles random location generation and Street View API integration
"""

import googlemaps
import random
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from geopy.distance import geodesic
import asyncio
import os
from enum import Enum

logger = logging.getLogger(__name__)

class LocationDifficulty(Enum):
    EASY = "easy"      # Major cities, distinctive landmarks
    MEDIUM = "medium"  # Suburban areas, smaller cities
    HARD = "hard"      # Rural areas, remote locations
    EXPERT = "expert"  # Very obscure locations

@dataclass
class StreetViewLocation:
    """Represents a Street View location with metadata"""
    lat: float
    lon: float
    pano_id: Optional[str]
    heading: float = 0.0
    pitch: float = 0.0
    fov: int = 90
    difficulty: LocationDifficulty = LocationDifficulty.MEDIUM
    country: Optional[str] = None
    region: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None

class StreetViewService:
    """
    Service for generating random Street View locations and managing Google API calls
    """
    
    def __init__(self):
        self.gmaps = None
        self.api_key = None
        self.initialized = False
        
        # Biased location pools for different difficulty levels
        self.location_pools = {
            LocationDifficulty.EASY: [
                # Major world cities with distinctive features
                {"lat": 40.7589, "lon": -73.9851, "country": "USA", "region": "New York"},
                {"lat": 48.8566, "lon": 2.3522, "country": "France", "region": "Paris"},
                {"lat": 51.5074, "lon": -0.1278, "country": "UK", "region": "London"},
                {"lat": 35.6762, "lon": 139.6503, "country": "Japan", "region": "Tokyo"},
                {"lat": -33.8688, "lon": 151.2093, "country": "Australia", "region": "Sydney"},
                {"lat": 37.7749, "lon": -122.4194, "country": "USA", "region": "San Francisco"},
                {"lat": 55.7558, "lon": 37.6173, "country": "Russia", "region": "Moscow"},
                {"lat": 52.5200, "lon": 13.4050, "country": "Germany", "region": "Berlin"},
                {"lat": 41.9028, "lon": 12.4964, "country": "Italy", "region": "Rome"},
                {"lat": 39.9042, "lon": 116.4074, "country": "China", "region": "Beijing"},
            ],
            LocationDifficulty.MEDIUM: [
                # Mid-sized cities and suburbs
                {"lat": 47.6062, "lon": -122.3321, "country": "USA", "region": "Seattle"},
                {"lat": 45.4642, "lon": 9.1900, "country": "Italy", "region": "Milan"},
                {"lat": 50.0755, "lon": 14.4378, "country": "Czech Republic", "region": "Prague"},
                {"lat": 59.9311, "lon": 30.3609, "country": "Russia", "region": "St. Petersburg"},
                {"lat": 41.3851, "lon": 2.1734, "country": "Spain", "region": "Barcelona"},
                {"lat": 43.6532, "lon": -79.3832, "country": "Canada", "region": "Toronto"},
                {"lat": -34.6037, "lon": -58.3816, "country": "Argentina", "region": "Buenos Aires"},
                {"lat": 19.4326, "lon": -99.1332, "country": "Mexico", "region": "Mexico City"},
                {"lat": 1.3521, "lon": 103.8198, "country": "Singapore", "region": "Singapore"},
                {"lat": 13.7563, "lon": 100.5018, "country": "Thailand", "region": "Bangkok"},
            ],
            LocationDifficulty.HARD: [
                # Smaller towns and rural areas
                {"lat": 64.1466, "lon": -21.9426, "country": "Iceland", "region": "Reykjavik"},
                {"lat": 71.0486, "lon": 25.7832, "country": "Norway", "region": "Hammerfest"},
                {"lat": -45.8788, "lon": 170.5028, "country": "New Zealand", "region": "Dunedin"},
                {"lat": 27.1751, "lon": 78.0421, "country": "India", "region": "Agra"},
                {"lat": -22.9068, "lon": -43.1729, "country": "Brazil", "region": "Rio de Janeiro"},
                {"lat": 31.2304, "lon": 121.4737, "country": "China", "region": "Shanghai"},
                {"lat": -26.2041, "lon": 28.0473, "country": "South Africa", "region": "Johannesburg"},
                {"lat": 60.1699, "lon": 24.9384, "country": "Finland", "region": "Helsinki"},
                {"lat": 45.8150, "lon": 15.9819, "country": "Croatia", "region": "Zagreb"},
                {"lat": 50.4501, "lon": 30.5234, "country": "Ukraine", "region": "Kiev"},
            ]
        }
        
        # Country-specific biases for more realistic locations
        self.country_weights = {
            "USA": 0.25,      # High Street View coverage
            "Canada": 0.15,   # Good coverage
            "Australia": 0.12, # Good coverage
            "UK": 0.10,       # Excellent coverage
            "Germany": 0.08,  # Good coverage
            "France": 0.08,   # Good coverage
            "Japan": 0.06,    # Limited but high quality
            "Italy": 0.05,    # Good tourist areas
            "Spain": 0.04,    # Good coverage
            "Others": 0.07    # Rest of world
        }
    
    async def initialize(self):
        """Initialize the Google Maps client"""
        try:
            self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            if not self.api_key:
                logger.warning("Google Maps API key not found. Using mock data.")
                self.initialized = False
                return
            
            self.gmaps = googlemaps.Client(key=self.api_key)
            
            # Test the API connection
            test_result = self.gmaps.streetview_metadata(
                location=(40.7589, -73.9851),
                size="640x640"
            )
            
            if test_result.get('status') == 'OK':
                self.initialized = True
                logger.info("Google Street View API initialized successfully")
            else:
                logger.error("Failed to initialize Google Street View API")
                self.initialized = False
                
        except Exception as e:
            logger.error(f"Error initializing Street View service: {e}")
            self.initialized = False
    
    async def get_random_location(self, difficulty: LocationDifficulty = LocationDifficulty.MEDIUM, 
                                region_preference: Optional[str] = None) -> StreetViewLocation:
        """
        Generate a random Street View location based on difficulty and preferences
        """
        
        if not self.initialized:
            # Return mock location if API not available
            return await self._get_mock_location(difficulty)
        
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Select base location from appropriate difficulty pool
                base_locations = self.location_pools.get(difficulty, self.location_pools[LocationDifficulty.MEDIUM])
                
                # Apply region preference if specified
                if region_preference:
                    filtered_locations = [loc for loc in base_locations if region_preference.lower() in loc.get('country', '').lower()]
                    if filtered_locations:
                        base_locations = filtered_locations
                
                # Select random base location
                base_location = random.choice(base_locations)
                
                # Add some randomization around the base location
                lat_offset = random.uniform(-0.01, 0.01)  # ~1km radius
                lon_offset = random.uniform(-0.01, 0.01)
                
                candidate_lat = base_location["lat"] + lat_offset
                candidate_lon = base_location["lon"] + lon_offset
                
                # Check if Street View is available at this location
                metadata = self.gmaps.streetview_metadata(
                    location=(candidate_lat, candidate_lon),
                    radius=1000  # Search within 1km
                )
                
                if metadata.get('status') == 'OK':
                    # Generate Street View image URL
                    image_url = self._generate_streetview_url(
                        lat=metadata['location']['lat'],
                        lon=metadata['location']['lng'],
                        size="640x640",
                        fov=90,
                        heading=random.randint(0, 359),
                        pitch=random.randint(-20, 20)
                    )
                    
                    return StreetViewLocation(
                        lat=metadata['location']['lat'],
                        lon=metadata['location']['lng'],
                        pano_id=metadata.get('pano_id'),
                        heading=random.randint(0, 359),
                        pitch=random.randint(-20, 20),
                        fov=90,
                        difficulty=difficulty,
                        country=base_location.get("country"),
                        region=base_location.get("region"),
                        description=f"{base_location.get('region')}, {base_location.get('country')}",
                        image_url=image_url
                    )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to generate location: {e}")
                continue
        
        # Fallback to mock location if all attempts failed
        logger.error("Failed to generate valid Street View location, using fallback")
        return await self._get_mock_location(difficulty)
    
    async def _get_mock_location(self, difficulty: LocationDifficulty) -> StreetViewLocation:
        """Generate a mock location when API is not available"""
        base_locations = self.location_pools.get(difficulty, self.location_pools[LocationDifficulty.MEDIUM])
        location = random.choice(base_locations)
        
        return StreetViewLocation(
            lat=location["lat"],
            lon=location["lon"],
            pano_id=None,
            heading=random.randint(0, 359),
            pitch=random.randint(-20, 20),
            fov=90,
            difficulty=difficulty,
            country=location.get("country"),
            region=location.get("region"),
            description=f"Mock: {location.get('region')}, {location.get('country')}",
            image_url="https://via.placeholder.com/640x640?text=Mock+Street+View"
        )
    
    def _generate_streetview_url(self, lat: float, lon: float, size: str = "640x640", 
                               fov: int = 90, heading: int = 0, pitch: int = 0) -> str:
        """Generate Google Street View Static API URL"""
        base_url = "https://maps.googleapis.com/maps/api/streetview"
        
        params = {
            'size': size,
            'location': f"{lat},{lon}",
            'fov': fov,
            'heading': heading,
            'pitch': pitch,
            'key': self.api_key
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}?{param_string}"
    
    async def get_streetview_metadata(self, lat: float, lon: float) -> Dict:
        """Get Street View metadata for a specific location"""
        if not self.initialized:
            return {"status": "MOCK", "available": True}
        
        try:
            return self.gmaps.streetview_metadata(location=(lat, lon))
        except Exception as e:
            logger.error(f"Error getting Street View metadata: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def validate_location(self, lat: float, lon: float, radius: int = 1000) -> bool:
        """Check if Street View is available at a given location"""
        metadata = await self.get_streetview_metadata(lat, lon)
        return metadata.get('status') == 'OK'
    
    async def get_nearby_locations(self, lat: float, lon: float, 
                                 count: int = 5, radius: float = 10.0) -> List[StreetViewLocation]:
        """Get multiple Street View locations near a given point"""
        locations = []
        attempts = 0
        max_attempts = count * 3
        
        while len(locations) < count and attempts < max_attempts:
            # Generate random point within radius (km)
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(0, radius)
            
            # Calculate new coordinates
            new_lat = lat + (distance / 111.32) * random.choice([-1, 1])  # Roughly 111.32 km per degree
            new_lon = lon + (distance / (111.32 * abs(lat))) * random.choice([-1, 1])
            
            if await self.validate_location(new_lat, new_lon):
                locations.append(StreetViewLocation(
                    lat=new_lat,
                    lon=new_lon,
                    pano_id=None,
                    heading=random.randint(0, 359),
                    pitch=random.randint(-20, 20),
                    difficulty=LocationDifficulty.MEDIUM
                ))
            
            attempts += 1
        
        return locations
    
    async def get_curated_challenge_locations(self) -> List[StreetViewLocation]:
        """Get a set of curated locations for special challenges"""
        challenge_locations = [
            # Famous landmarks that are challenging but fair
            {"lat": 27.1751, "lon": 78.0421, "desc": "Taj Mahal area, India"},
            {"lat": -13.1631, "lon": -72.5450, "desc": "Machu Picchu region, Peru"},
            {"lat": 29.9792, "lon": 31.1344, "desc": "Giza Pyramids area, Egypt"},
            {"lat": 41.9029, "lon": 12.4534, "desc": "Colosseum area, Rome"},
            {"lat": 40.6892, "lon": -74.0445, "desc": "Statue of Liberty area, NYC"},
        ]
        
        locations = []
        for loc in challenge_locations:
            if await self.validate_location(loc["lat"], loc["lon"]):
                locations.append(StreetViewLocation(
                    lat=loc["lat"],
                    lon=loc["lon"],
                    pano_id=None,
                    difficulty=LocationDifficulty.EXPERT,
                    description=loc["desc"]
                ))
        
        return locations
    
    def get_weighted_difficulty(self) -> LocationDifficulty:
        """Get a weighted random difficulty level"""
        weights = {
            LocationDifficulty.EASY: 0.3,
            LocationDifficulty.MEDIUM: 0.4,
            LocationDifficulty.HARD: 0.25,
            LocationDifficulty.EXPERT: 0.05
        }
        
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]
    
    async def cleanup(self):
        """Clean up resources"""
        self.gmaps = None
        self.initialized = False
        logger.info("Street View service cleaned up")
