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
        
        # Countries with limited or no Street View coverage (GeoGuessr-style filtering)
        self.blacklisted_countries = {
            'CN',  # China (limited coverage)
            'KP',  # North Korea
            'IR',  # Iran
            'SD',  # Sudan
            'SO',  # Somalia
            'AF',  # Afghanistan
            'IQ',  # Iraq
            'SY',  # Syria
            'LY',  # Libya
            'YE',  # Yemen
            'MM',  # Myanmar (limited)
            'CF',  # Central African Republic
            'TD',  # Chad
            'ER',  # Eritrea
            'SS',  # South Sudan
            'TJ',  # Tajikistan (limited)
            'TM',  # Turkmenistan (limited)
            'UZ',  # Uzbekistan (limited)
            'KG',  # Kyrgyzstan (limited)
            'BF',  # Burkina Faso (limited)
            'ML',  # Mali (limited)
            'NE',  # Niger (limited)
            'MR',  # Mauritania (limited)
            'DJ',  # Djibouti (limited)
            'KM',  # Comoros (limited)
            'ST',  # São Tomé and Príncipe (limited)
            'CV',  # Cape Verde (limited)
            'SB',  # Solomon Islands (limited)
            'VU',  # Vanuatu (limited)
            'WS',  # Samoa (limited)
            'TV',  # Tuvalu (no coverage)
            'NR',  # Nauru (no coverage)
            'PW',  # Palau (limited)
            'FM',  # Micronesia (limited)
            'MH',  # Marshall Islands (limited)
            'KI',  # Kiribati (no coverage)
            'TO',  # Tonga (limited)
        }
        
        # Curated locations around the world for interesting gameplay
        # Only locations with confirmed good Street View coverage
        self.location_pools = {
            'urban': [
                {'lat': 40.7614, 'lng': -73.9776, 'country': 'USA', 'city': 'New York'},
                {'lat': 51.5074, 'lng': -0.1278, 'country': 'UK', 'city': 'London'},
                {'lat': 35.6762, 'lng': 139.6503, 'country': 'Japan', 'city': 'Tokyo'},
                {'lat': 48.8566, 'lng': 2.3522, 'country': 'France', 'city': 'Paris'},
                {'lat': -33.8688, 'lng': 151.2093, 'country': 'Australia', 'city': 'Sydney'},
                {'lat': 55.7558, 'lng': 37.6176, 'country': 'Russia', 'city': 'Moscow'},
                {'lat': 52.3676, 'lng': 4.9041, 'country': 'Netherlands', 'city': 'Amsterdam'},
                {'lat': 19.4326, 'lng': -99.1332, 'country': 'Mexico', 'city': 'Mexico City'},
                {'lat': -23.5505, 'lng': -46.6333, 'country': 'Brazil', 'city': 'São Paulo'},
                {'lat': 28.6139, 'lng': 77.2090, 'country': 'India', 'city': 'New Delhi'},
                {'lat': 59.3293, 'lng': 18.0686, 'country': 'Sweden', 'city': 'Stockholm'},
                {'lat': 45.4642, 'lng': 9.1900, 'country': 'Italy', 'city': 'Milan'},
                {'lat': 50.0755, 'lng': 14.4378, 'country': 'Czech Republic', 'city': 'Prague'},
                {'lat': 37.5665, 'lng': 126.9780, 'country': 'South Korea', 'city': 'Seoul'},
            ],
            'rural': [
                {'lat': 46.8182, 'lng': 8.2275, 'country': 'Switzerland', 'city': 'Swiss Alps'},
                {'lat': 64.2008, 'lng': -149.4937, 'country': 'USA', 'city': 'Alaska'},
                {'lat': -45.8788, 'lng': 170.5028, 'country': 'New Zealand', 'city': 'Otago'},
                {'lat': 56.1304, 'lng': -106.3468, 'country': 'Canada', 'city': 'Saskatchewan'},
                {'lat': 62.3908, 'lng': -114.3717, 'country': 'Canada', 'city': 'Northwest Territories'},
                {'lat': -25.2744, 'lng': 133.7751, 'country': 'Australia', 'city': 'Australian Outback'},
                {'lat': 70.2676, 'lng': 31.1107, 'country': 'Norway', 'city': 'Northern Norway'},
            ],
            'coastal': [
                {'lat': -34.6037, 'lng': 18.4413, 'country': 'South Africa', 'city': 'Cape Town'},
                {'lat': 41.9028, 'lng': 12.4964, 'country': 'Italy', 'city': 'Rome'},
                {'lat': 25.2048, 'lng': 55.2708, 'country': 'UAE', 'city': 'Dubai'},
                {'lat': 1.3521, 'lng': 103.8198, 'country': 'Singapore', 'city': 'Singapore'},
                {'lat': 60.1282, 'lng': 18.6435, 'country': 'Sweden', 'city': 'Stockholm Archipelago'},
                {'lat': 36.8065, 'lng': -76.2859, 'country': 'USA', 'city': 'Virginia Beach'},
                {'lat': -37.8136, 'lng': 144.9631, 'country': 'Australia', 'city': 'Melbourne'},
                {'lat': 49.2827, 'lng': -123.1207, 'country': 'Canada', 'city': 'Vancouver'},
            ]
        }
    
    async def get_random_location(self, difficulty: str = 'mixed', region_preference: Optional[str] = None) -> Dict:
        """
        Generate a random Street View location for the game with GeoGuessr-style reliability
        Keeps trying until a valid Street View location is found
        
        Args:
            difficulty: 'easy' (major cities), 'medium' (mixed), 'hard' (rural/remote)
        """
        max_attempts = 20  # Maximum attempts to find a valid location
        
        for main_attempt in range(max_attempts):
            try:
                print(f"🎯 Location search attempt {main_attempt + 1}/{max_attempts}")
                
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
                print(f"📍 Trying base location: {base_location['city']}, {base_location['country']}")
                
                # Try multiple strategies to find a working location
                valid_location = await self._find_valid_streetview_location(base_location)
                
                if valid_location:
                    print(f"✅ Valid Street View location found: {valid_location['lat']:.6f}, {valid_location['lng']:.6f}")
                    
                    # Generate Street View metadata
                    streetview_data = await self._get_streetview_data(
                        valid_location['lat'], valid_location['lng']
                    )
                    
                    return {
                        'lat': valid_location['lat'],
                        'lon': valid_location['lng'],  # Convert lng to lon for consistency
                        'street_view_urls': streetview_data['images'],
                        'country_hint': valid_location['country_hint'],
                        'city_hint': valid_location['city_hint'],
                        'difficulty': difficulty,
                        'metadata': {
                            'region': base_location.get('country'),
                            'urban_type': self._classify_location_type(base_location),
                            'interactive_url': streetview_data['interactive_url']
                        }
                    }
                else:
                    print(f"❌ No valid Street View found for {base_location['city']}, trying next location...")
                    
            except Exception as e:
                print(f"⚠️ Error in attempt {main_attempt + 1}: {str(e)}")
                continue
        
        # Final fallback: Use a guaranteed working location
        print("🚨 Using emergency fallback location (Times Square, NYC)")
        fallback = {'lat': 40.7580, 'lng': -73.9855, 'country': 'USA', 'city': 'New York'}
        
        streetview_data = await self._get_streetview_data(fallback['lat'], fallback['lng'])
        
        return {
            'lat': fallback['lat'],
            'lon': fallback['lng'],
            'street_view_urls': streetview_data['images'],
            'country_hint': fallback['country'],
            'city_hint': fallback['city'],
            'difficulty': difficulty,
            'metadata': {
                'region': fallback['country'],
                'urban_type': 'major_city',
                'interactive_url': streetview_data['interactive_url']
            }
        }
    
    async def _find_valid_streetview_location(self, base_location: Dict) -> Optional[Dict]:
        """
        Find a valid Street View location using multiple strategies
        Returns None if no valid location found
        """
        
        # Strategy 1: Try random offset around base location
        for offset_attempt in range(8):  # Try 8 different offsets
            lat_offset = random.uniform(-0.015, 0.015)  # ~1.5km radius
            lng_offset = random.uniform(-0.015, 0.015)
            
            test_location = {
                'lat': base_location['lat'] + lat_offset,
                'lng': base_location['lng'] + lng_offset,
                'country_hint': base_location['country'],
                'city_hint': base_location['city']
            }
            
            if await self._check_streetview_availability_robust(test_location['lat'], test_location['lng']):
                print(f"✅ Found valid offset location (attempt {offset_attempt + 1})")
                return test_location
        
        # Strategy 2: Try exact base location
        if await self._check_streetview_availability_robust(base_location['lat'], base_location['lng']):
            print(f"✅ Base location has Street View")
            return {
                'lat': base_location['lat'],
                'lng': base_location['lng'],
                'country_hint': base_location['country'],
                'city_hint': base_location['city']
            }
        
        # Strategy 3: Try smaller radius around base location
        for close_attempt in range(5):
            lat_offset = random.uniform(-0.005, 0.005)  # ~500m radius
            lng_offset = random.uniform(-0.005, 0.005)
            
            test_location = {
                'lat': base_location['lat'] + lat_offset,
                'lng': base_location['lng'] + lng_offset,
                'country_hint': base_location['country'],
                'city_hint': base_location['city']
            }
            
            if await self._check_streetview_availability_robust(test_location['lat'], test_location['lng']):
                print(f"✅ Found valid close location (attempt {close_attempt + 1})")
                return test_location
        
        print(f"❌ No valid Street View found for {base_location['city']}")
        return None
    
    async def _check_streetview_availability_robust(self, lat: float, lng: float) -> bool:
        """
        Robust Street View availability check with multiple validation methods
        Similar to GeoGuessr's validation process
        """
        
        if not self.api_key:
            print("No API key - assuming Street View available for development")
            return True  # Mock mode - assume available
        
        # Method 1: Check Street View Metadata API
        metadata_available = await self._check_streetview_metadata(lat, lng)
        if not metadata_available:
            return False
        
        # Method 2: Verify image is actually available (not just a default image)
        image_quality = await self._verify_streetview_image_quality(lat, lng)
        if not image_quality:
            return False
        
        return True
    
    async def _check_streetview_metadata(self, lat: float, lng: float) -> bool:
        """Check Street View metadata API for availability with robust error handling"""
        
        url = f"https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {
            'location': f"{lat},{lng}",
            'key': self.api_key
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 403:
                        print(f"⚠️ API key quota exceeded or restricted (403)")
                        return False
                    elif response.status == 404:
                        print(f"⚠️ Street View not found (404)")
                        return False
                    elif response.status != 200:
                        print(f"⚠️ API error: {response.status}")
                        return False
                    
                    data = await response.json()
                    
                    # Check if Street View is actually available
                    if data.get('status') != 'OK':
                        return False
                    
                    # Additional checks for quality
                    location_data = data.get('location', {})
                    if not location_data:
                        return False
                    
                    # Verify we have actual location data
                    if 'lat' not in location_data or 'lng' not in location_data:
                        return False
                    
                    print(f"✅ Street View metadata OK for {lat:.6f}, {lng:.6f}")
                    return True
                    
        except asyncio.TimeoutError:
            print(f"⚠️ Street View API timeout for {lat:.6f}, {lng:.6f}")
            return False
        except aiohttp.ClientError as e:
            print(f"⚠️ Street View API connection error: {str(e)}")
            return False
        except Exception as e:
            print(f"⚠️ Street View API unexpected error: {str(e)}")
            return False
    
    async def _verify_streetview_image_quality(self, lat: float, lng: float) -> bool:
        """
        Verify that the Street View image is of good quality and not a placeholder
        """
        
        # Test with a small image size to save bandwidth
        url = f"https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': '200x200',
            'location': f"{lat},{lng}",
            'key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return False
                    
                    # Read image content
                    image_content = await response.read()
                    
                    # Check if image is large enough (not a tiny placeholder)
                    if len(image_content) < 5000:  # Less than 5KB likely a placeholder
                        print(f"Street View image too small: {len(image_content)} bytes")
                        return False
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if 'image' not in content_type.lower():
                        print(f"Invalid content type: {content_type}")
                        return False
                    
                    print(f"Street View image quality OK: {len(image_content)} bytes")
                    return True
                    
        except Exception as e:
            print(f"Error verifying Street View image quality: {e}")
            return False

    async def _check_streetview_availability(self, lat: float, lng: float) -> bool:
        """Legacy method - kept for backwards compatibility"""
        return await self._check_streetview_availability_robust(lat, lng)
    
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
