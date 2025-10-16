import random
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class RandomLocationService:
    """Service for generating random Street View locations for fair AI vs Human gameplay"""
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        # Curated locations around the world for interesting gameplay
        self.location_pools = {
            'urban': [
                {'lat': 40.7614, 'lng': -73.9776, 'country': 'USA', 'city': 'New York'},
                {'lat': 51.5074, 'lng': -0.1278, 'country': 'UK', 'city': 'London'},
                {'lat': 35.6762, 'lng': 139.6503, 'country': 'Japan', 'city': 'Tokyo'},
                {'lat': 48.8566, 'lng': 2.3522, 'country': 'France', 'city': 'Paris'},
                {'lat': -33.8688, 'lng': 151.2093, 'country': 'Australia', 'city': 'Sydney'},
                {'lat': 55.7558, 'lng': 37.6176, 'country': 'Russia', 'city': 'Moscow'},
                {'lat': 39.9042, 'lng': 116.4074, 'country': 'China', 'city': 'Beijing'},
                {'lat': 19.4326, 'lng': -99.1332, 'country': 'Mexico', 'city': 'Mexico City'},
                {'lat': -23.5505, 'lng': -46.6333, 'country': 'Brazil', 'city': 'São Paulo'},
                {'lat': 28.6139, 'lng': 77.2090, 'country': 'India', 'city': 'New Delhi'},
            ],
            'rural': [
                {'lat': 46.8182, 'lng': 8.2275, 'country': 'Switzerland', 'city': 'Swiss Alps'},
                {'lat': 64.2008, 'lng': -149.4937, 'country': 'USA', 'city': 'Alaska'},
                {'lat': -45.8788, 'lng': 170.5028, 'country': 'New Zealand', 'city': 'Otago'},
                {'lat': 68.9584, 'lng': 33.0848, 'country': 'Russia', 'city': 'Murmansk'},
                {'lat': 71.0486, 'lng': -8.0426, 'country': 'Norway', 'city': 'Svalbard'},
            ],
            'coastal': [
                {'lat': -34.6037, 'lng': 18.4413, 'country': 'South Africa', 'city': 'Cape Town'},
                {'lat': 41.9028, 'lng': 12.4964, 'country': 'Italy', 'city': 'Rome'},
                {'lat': 25.2048, 'lng': 55.2708, 'country': 'UAE', 'city': 'Dubai'},
                {'lat': -16.2902, 'lng': -67.5253, 'country': 'Bolivia', 'city': 'La Paz'},
                {'lat': 1.3521, 'lng': 103.8198, 'country': 'Singapore', 'city': 'Singapore'},
            ]
        }
    
    async def get_random_location(self, difficulty: str = 'mixed', region_preference: Optional[str] = None) -> Dict:
        """
        Generate a random Street View location for the game
        
        Args:
            difficulty: 'easy' (major cities), 'medium' (mixed), 'hard' (rural/remote)
        """
        
        if difficulty == 'easy':
            pool = self.location_pools['urban']
        elif difficulty == 'hard':
            pool = self.location_pools['rural'] + self.location_pools['coastal']
        else:  # mixed
            all_locations = []
            for category_locations in self.location_pools.values():
                all_locations.extend(category_locations)
            pool = all_locations
        
        # Select base location
        base_location = random.choice(pool)
        
        # Add some randomness around the base location (within ~2km radius)
        lat_offset = random.uniform(-0.02, 0.02)  # ~2km
        lng_offset = random.uniform(-0.02, 0.02)
        
        final_location = {
            'lat': base_location['lat'] + lat_offset,
            'lng': base_location['lng'] + lng_offset,
            'country_hint': base_location['country'],
            'city_hint': base_location['city']
        }
        
        # Verify Street View is available at this location
        streetview_available = await self._check_streetview_availability(
            final_location['lat'], final_location['lng']
        )
        
        if not streetview_available:
            # Fallback to original location if random offset doesn't have Street View
            final_location['lat'] = base_location['lat']
            final_location['lng'] = base_location['lng']
        
        # Generate Street View metadata
        streetview_data = await self._get_streetview_data(
            final_location['lat'], final_location['lng']
        )
        
        return {
            'lat': final_location['lat'],
            'lon': final_location['lng'],  # Convert lng to lon for consistency
            'street_view_urls': streetview_data['images'],
            'country_hint': final_location['country_hint'],
            'city_hint': final_location['city_hint'],
            'difficulty': difficulty,
            'metadata': {
                'region': base_location.get('country'),
                'urban_type': self._classify_location_type(base_location),
                'interactive_url': streetview_data['interactive_url']
            }
        }
    
    async def _check_streetview_availability(self, lat: float, lng: float) -> bool:
        """Check if Street View is available at given coordinates"""
        
        if not self.api_key:
            return True  # Mock mode - assume available
        
        url = f"https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            'location': f"{lat},{lng}",
            'key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('status') == 'OK'
        except Exception as e:
            print(f"Error checking Street View availability: {e}")
            return False
    
    async def _get_streetview_data(self, lat: float, lng: float) -> Dict:
        """Get Street View image and metadata for the location"""
        
        streetview_url = f"https://maps.googleapis.com/maps/api/streetview"
        
        # Multiple angles for comprehensive AI analysis
        angles = [0, 90, 180, 270]  # N, E, S, W
        streetview_images = []
        
        for angle in angles:
            params = {
                'size': '640x640',
                'location': f"{lat},{lng}",
                'heading': angle,
                'pitch': 0,
                'key': self.api_key if self.api_key else 'mock'
            }
            
            if self.api_key:
                image_url = f"{streetview_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            else:
                # Mock URL for development
                image_url = f"https://picsum.photos/640/640?random={angle}"
            
            streetview_images.append({
                'angle': angle,
                'direction': ['North', 'East', 'South', 'West'][angles.index(angle)],
                'url': image_url
            })
        
        return {
            'images': streetview_images,
            'location': {'lat': lat, 'lng': lng},
            'interactive_url': self._generate_interactive_streetview_url(lat, lng)
        }
    
    def _generate_interactive_streetview_url(self, lat: float, lng: float) -> str:
        """Generate URL for interactive Street View embed"""
        base_url = "https://www.google.com/maps/embed/v1/streetview"
        params = {
            'location': f"{lat},{lng}",
            'key': self.api_key if self.api_key else 'YOUR_API_KEY'
        }
        return f"{base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
    
    def _classify_location_type(self, location: Dict) -> str:
        """Classify the type of location for AI analysis hints"""
        city = location.get('city', '').lower()
        
        if any(word in city for word in ['alps', 'alaska', 'svalbard']):
            return 'remote'
        elif any(word in city for word in ['new york', 'tokyo', 'london', 'paris']):
            return 'major_city'
        elif any(word in city for word in ['cape town', 'dubai', 'singapore']):
            return 'coastal_city'
        else:
            return 'urban'

    async def generate_game_session(self, difficulty: str = 'medium') -> Dict:
        """
        Generate a complete game session with random location
        
        Returns:
            Complete game session data for AI vs Human competition
        """
        location_data = await self.get_random_location(difficulty)
        
        game_session = {
            'session_id': self._generate_session_id(),
            'game_mode': 'ai_vs_human',
            'location': location_data['game_location'],
            'streetview': location_data['streetview_data'],
            'difficulty': difficulty,
            'metadata': location_data['metadata'],
            'ai_analysis_target': {
                'images': location_data['streetview_data']['images'],
                'analysis_modes': [
                    'object_detection',
                    'text_recognition',
                    'architectural_analysis',
                    'vegetation_detection',
                    'cultural_indicators',
                    'weather_analysis'
                ]
            },
            'human_interface': {
                'streetview_embed': location_data['streetview_data']['interactive_url'],
                'navigation_enabled': True,
                'guess_interface': 'world_map'
            },
            'scoring': {
                'distance_threshold_km': [1, 25, 100, 1000],  # Excellent, Good, Fair, Poor
                'time_limit_seconds': 300  # 5 minutes
            }
        }
        
        return game_session
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())[:8]
