"""
Game Service for managing GeoCV gameplay
Coordinates between CV pipeline and game logic
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import logging

from app.core.cv_pipeline import CVPipeline, CVAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Current state of a game session"""
    game_id: str
    image_path: Optional[str]  # Can be None for Street View games
    streetview_location: Optional[Dict]  # Street View location data
    ai_analysis: Optional[CVAnalysisResult]
    ai_guess: Optional[Tuple[float, float]]  # lat, lon
    human_guess: Optional[Tuple[float, float]]
    actual_location: Optional[Tuple[float, float]]
    game_mode: str
    time_limit: Optional[int]
    start_time: float
    status: str  # 'analyzing', 'waiting_human', 'completed'
    ai_confidence_display: str
    difficulty: Optional[str] = None

@dataclass 
class GameResult:
    """Result of a completed game round"""
    game_id: str
    ai_distance_error: float
    human_distance_error: float
    ai_score: int
    human_score: int
    winner: str
    processing_details: Dict

class GameService:
    """Main game coordination service"""
    
    def __init__(self, cv_pipeline: CVPipeline):
        self.cv_pipeline = cv_pipeline
        self.active_games: Dict[str, GameState] = {}
        self.game_results: List[GameResult] = []
        
        # Game configuration
        self.game_modes = {
            'classic': {'time_limit': None, 'description': 'Unlimited time for both players'},
            'speed': {'time_limit': 30, 'description': 'AI gets 30s, human gets AI time + 10s'},
            'blitz': {'time_limit': 15, 'description': 'Quick rounds - 15s for AI'},
            'training': {'time_limit': None, 'description': 'See AI reasoning step by step'}
        }
    
    async def start_new_game(self, image_path: str, game_mode: str = 'classic', 
                            actual_location: Optional[Tuple[float, float]] = None) -> str:
        """Start a new game session with uploaded image"""
        
        game_id = str(uuid.uuid4())
        
        # Validate game mode
        if game_mode not in self.game_modes:
            game_mode = 'classic'
        
        # Initialize game state
        game_state = GameState(
            game_id=game_id,
            image_path=image_path,
            streetview_location=None,
            ai_analysis=None,
            ai_guess=None,
            human_guess=None,
            actual_location=actual_location,
            game_mode=game_mode,
            time_limit=self.game_modes[game_mode]['time_limit'],
            start_time=time.time(),
            status='analyzing',
            ai_confidence_display='Initializing AI analysis...'
        )
        
        self.active_games[game_id] = game_state
        
        # Start AI analysis in background
        asyncio.create_task(self._run_ai_analysis(game_id))
        
        logger.info(f"Started new game: {game_id}, mode: {game_mode}")
        return game_id
    
    async def start_streetview_game(self, streetview_location: Dict, game_mode: str = 'classic') -> str:
        """Start a new game session with Street View location"""
        
        game_id = str(uuid.uuid4())
        
        # Validate game mode
        if game_mode not in self.game_modes:
            game_mode = 'classic'
        
        # Extract actual location from Street View data
        actual_location = (
            streetview_location['location']['lat'],
            streetview_location['location']['lon']
        )
        
        # Initialize game state
        game_state = GameState(
            game_id=game_id,
            image_path=None,
            streetview_location=streetview_location,
            ai_analysis=None,
            ai_guess=None,
            human_guess=None,
            actual_location=actual_location,
            game_mode=game_mode,
            time_limit=self.game_modes[game_mode]['time_limit'],
            start_time=time.time(),
            status='analyzing',
            ai_confidence_display='Initializing Street View analysis...',
            difficulty=streetview_location.get('metadata', {}).get('difficulty', 'medium')
        )
        
        self.active_games[game_id] = game_state
        
        # Start AI analysis in background using Street View image
        asyncio.create_task(self._run_streetview_analysis(game_id))
        
        logger.info(f"Started new Street View game: {game_id}, mode: {game_mode}")
        return game_id
    
    async def _run_ai_analysis(self, game_id: str):
        """Run AI analysis for a game"""
        
        game_state = self.active_games.get(game_id)
        if not game_state:
            return
        
        try:
            # Update status messages during analysis
            analysis_steps = [
                "🔍 Loading and preprocessing image...",
                "🚗 Detecting vehicles and transportation...",
                "🛣️ Analyzing roads and infrastructure...", 
                "🌳 Examining vegetation and terrain...",
                "🏢 Studying architectural features...",
                "☀️ Analyzing lighting and sky conditions...",
                "📝 Processing text and signage...",
                "🧠 Aggregating geographic clues...",
                "🎯 Formulating location guess..."
            ]
            
            for i, step in enumerate(analysis_steps):
                game_state.ai_confidence_display = step
                await asyncio.sleep(0.5)  # Simulate processing time
                
                # Add some realistic processing delays for different steps
                if 'Detecting vehicles' in step:
                    await asyncio.sleep(1.0)
                elif 'Analyzing roads' in step:
                    await asyncio.sleep(0.8)
                elif 'Examining vegetation' in step:
                    await asyncio.sleep(0.7)
            
            # Run actual CV analysis
            game_state.ai_confidence_display = "🔬 Running computer vision analysis..."
            analysis_result = await self.cv_pipeline.analyze_image(game_state.image_path)
            game_state.ai_analysis = analysis_result
            
            # Generate AI guess based on analysis
            game_state.ai_confidence_display = "🤔 Making educated geographic guess..."
            await asyncio.sleep(1.0)
            
            ai_guess = self._generate_ai_guess(analysis_result)
            game_state.ai_guess = ai_guess
            
            # Update final status
            confidence_text = self._get_confidence_text(analysis_result.confidence_level.value)
            game_state.ai_confidence_display = f"✅ Analysis complete! {confidence_text}"
            game_state.status = 'waiting_human'
            
            logger.info(f"AI analysis completed for game {game_id}")
            
        except Exception as e:
            logger.error(f"AI analysis failed for game {game_id}: {e}")
            game_state.ai_confidence_display = f"❌ Analysis failed: {str(e)}"
            game_state.status = 'error'
    
    async def _run_streetview_analysis(self, game_id: str):
        """Run AI analysis for a Street View game"""
        
        game_state = self.active_games.get(game_id)
        if not game_state or not game_state.streetview_location:
            return
        
        try:
            # Download Street View image for analysis
            image_url = game_state.streetview_location.get('image_url')
            if not image_url:
                logger.error(f"No Street View image URL for game {game_id}")
                game_state.status = 'error'
                return
            
            # Update status messages during analysis
            analysis_steps = [
                "🌍 Loading Street View imagery...",
                "🔍 Analyzing geographic visual cues...",
                "🚗 Detecting regional vehicles and infrastructure...",
                "🏢 Examining architectural patterns...",
                "🌳 Analyzing climate and vegetation indicators...",
                "📝 Processing visible text and signage...",
                "🧠 Correlating features with geographic databases...",
                "🎯 Calculating most probable location..."
            ]
            
            for i, step in enumerate(analysis_steps):
                game_state.ai_confidence_display = step
                await asyncio.sleep(0.7)  # Slightly longer for Street View analysis
            
            # For now, simulate CV analysis on Street View image
            # In a real implementation, this would download and process the image
            game_state.ai_confidence_display = "🔬 Running computer vision analysis on Street View..."
            
            # Create mock analysis result based on actual location
            actual_lat, actual_lon = game_state.actual_location
            difficulty = game_state.streetview_location.get('metadata', {}).get('difficulty', 'medium')
            
            # Simulate analysis result with features that would be detected
            mock_analysis = self._create_mock_analysis_result(actual_lat, actual_lon, difficulty)
            game_state.ai_analysis = mock_analysis
            
            # Generate AI guess with appropriate difficulty-based accuracy
            game_state.ai_confidence_display = "🤔 Making educated geographic prediction..."
            await asyncio.sleep(1.0)
            
            ai_guess = self._generate_streetview_ai_guess(mock_analysis, difficulty)
            game_state.ai_guess = ai_guess
            
            # Update final status
            confidence_text = self._get_confidence_text(mock_analysis.confidence_level.value)
            game_state.ai_confidence_display = f"✅ Street View analysis complete! {confidence_text}"
            game_state.status = 'waiting_human'
            
            logger.info(f"Street View AI analysis completed for game {game_id}")
            
        except Exception as e:
            logger.error(f"Street View AI analysis failed for game {game_id}: {e}")
            game_state.ai_confidence_display = f"❌ Analysis failed: {str(e)}"
            game_state.status = 'error'
    
    def _generate_ai_guess(self, analysis: CVAnalysisResult) -> Tuple[float, float]:
        """Generate AI location guess based on CV analysis"""
        
        # This is where the geospatial reasoning happens
        # For now, implementing a basic rule-based system
        
        suggested_regions = analysis.suggested_regions
        confidence = analysis.overall_confidence
        
        # Default to a random location if no strong indicators
        lat, lon = 40.7128, -74.0060  # NYC as default
        
        # Rule-based geographic inference
        if 'tropical_region' in suggested_regions:
            # Bias towards tropical locations
            lat = self._add_noise(10.0, confidence)  # Closer to equator
            lon = self._add_noise(-84.0, confidence)  # Central America bias
        
        elif 'arid_climate' in suggested_regions or 'desert_region' in suggested_regions:
            # Desert regions
            lat = self._add_noise(25.0, confidence)
            lon = self._add_noise(45.0, confidence)  # Middle East bias
        
        elif 'northern_regions' in suggested_regions or 'temperate_forest' in suggested_regions:
            # Northern/temperate regions
            lat = self._add_noise(50.0, confidence)
            lon = self._add_noise(10.0, confidence)  # Europe bias
        
        elif 'urban_center' in suggested_regions and 'highway_system' in suggested_regions:
            # Major cities with good infrastructure
            major_cities = [
                (40.7128, -74.0060),  # NYC
                (51.5074, -0.1278),   # London
                (48.8566, 2.3522),    # Paris
                (35.6762, 139.6503),  # Tokyo
                (37.7749, -122.4194), # San Francisco
            ]
            base_lat, base_lon = major_cities[hash(str(suggested_regions)) % len(major_cities)]
            lat = self._add_noise(base_lat, confidence)
            lon = self._add_noise(base_lon, confidence)
        
        elif 'rural_area' in suggested_regions:
            # Rural areas - less precise
            lat = self._add_noise(45.0, confidence * 0.5)  # Lower confidence for rural
            lon = self._add_noise(0.0, confidence * 0.5)
        
        # Additional refinements based on specific features
        feature_summary = analysis.feature_summary
        
        if 'vehicle' in feature_summary and feature_summary['vehicle'] > 3:
            # Lots of vehicles = urban area, increase precision
            confidence_multiplier = 1.2
        else:
            confidence_multiplier = 0.8
        
        # Apply final noise based on overall confidence
        final_noise = (1.0 - confidence * confidence_multiplier) * 10.0  # Up to 10 degrees noise
        lat = max(-90, min(90, lat + (hash(str(analysis.detections)) % 21 - 10) * final_noise / 10))
        lon = max(-180, min(180, lon + (hash(str(analysis.detections[::-1])) % 21 - 10) * final_noise / 10))
        
        return (lat, lon)
    
    def _add_noise(self, base_value: float, confidence: float) -> float:
        """Add confidence-based noise to coordinate"""
        noise_factor = (1.0 - confidence) * 5.0  # Max 5 degrees of noise
        noise = (hash(str(base_value)) % 11 - 5) * noise_factor / 5
        return base_value + noise
    
    def _get_confidence_text(self, confidence_level: str) -> str:
        """Convert confidence level to human-readable text"""
        confidence_texts = {
            'very_high': 'Very confident in this guess! 🎯',
            'high': 'Pretty confident about this location 💪', 
            'medium': 'Moderate confidence, some good clues 🤔',
            'low': 'Low confidence, limited visual cues 😅'
        }
        return confidence_texts.get(confidence_level, 'Analyzing...')
    
    async def submit_human_guess(self, game_id: str, lat: float, lon: float) -> Optional[GameResult]:
        """Submit human player guess and complete the game"""
        
        game_state = self.active_games.get(game_id)
        if not game_state or game_state.status != 'waiting_human':
            return None
        
        game_state.human_guess = (lat, lon)
        game_state.status = 'completed'
        
        # Calculate results if we have actual location
        if game_state.actual_location and game_state.ai_guess:
            result = self._calculate_game_result(game_state)
            self.game_results.append(result)
            
            # Remove from active games
            del self.active_games[game_id]
            
            logger.info(f"Game {game_id} completed. Winner: {result.winner}")
            return result
        
        return None
    
    def _calculate_game_result(self, game_state: GameState) -> GameResult:
        """Calculate the result of a completed game"""
        
        actual_lat, actual_lon = game_state.actual_location
        ai_lat, ai_lon = game_state.ai_guess
        human_lat, human_lon = game_state.human_guess
        
        # Calculate distances using haversine formula
        ai_distance = self._calculate_distance(actual_lat, actual_lon, ai_lat, ai_lon)
        human_distance = self._calculate_distance(actual_lat, actual_lon, human_lat, human_lon)
        
        # Calculate GeoGuesser-style scores (5000 points max, decreasing with distance)
        ai_score = self._distance_to_score(ai_distance)
        human_score = self._distance_to_score(human_distance)
        
        # Determine winner
        if ai_score > human_score:
            winner = 'ai'
        elif human_score > ai_score:
            winner = 'human'
        else:
            winner = 'tie'
        
        # Prepare processing details
        processing_details = {
            'ai_analysis_time': game_state.ai_analysis.processing_time if game_state.ai_analysis else 0,
            'features_detected': game_state.ai_analysis.feature_summary if game_state.ai_analysis else {},
            'confidence_level': game_state.ai_analysis.confidence_level.value if game_state.ai_analysis else 'unknown',
            'geographic_hints': game_state.ai_analysis.suggested_regions if game_state.ai_analysis else []
        }
        
        return GameResult(
            game_id=game_state.game_id,
            ai_distance_error=ai_distance,
            human_distance_error=human_distance,
            ai_score=ai_score,
            human_score=human_score,
            winner=winner,
            processing_details=processing_details
        )
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using haversine formula"""
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
    
    def _distance_to_score(self, distance_km: float) -> int:
        """Convert distance error to GeoGuesser-style score"""
        if distance_km == 0:
            return 5000
        elif distance_km < 1:
            return int(5000 - distance_km * 1000)
        elif distance_km < 25:
            return int(4000 - (distance_km - 1) * 41.67)
        elif distance_km < 200:
            return int(3000 - (distance_km - 25) * 11.43)
        elif distance_km < 750:
            return int(1000 - (distance_km - 200) * 1.82)
        elif distance_km < 2500:
            return int(500 - (distance_km - 750) * 0.29)
        else:
            return 0
    
    def get_game_state(self, game_id: str) -> Optional[Dict]:
        """Get current game state for frontend"""
        game_state = self.active_games.get(game_id)
        if not game_state:
            return None
        
        # Prepare state for frontend
        state_dict = {
            'game_id': game_state.game_id,
            'status': game_state.status,
            'game_mode': game_state.game_mode,
            'ai_confidence_display': game_state.ai_confidence_display,
            'time_limit': game_state.time_limit,
            'elapsed_time': time.time() - game_state.start_time,
            'difficulty': game_state.difficulty,
            'game_type': 'streetview' if game_state.streetview_location else 'upload'
        }
        
        # Add Street View location if available
        if game_state.streetview_location:
            state_dict['streetview_location'] = game_state.streetview_location
        
        # Add AI analysis details if available
        if game_state.ai_analysis:
            state_dict['ai_features'] = {
                'detections_count': len(game_state.ai_analysis.detections),
                'confidence_level': game_state.ai_analysis.confidence_level.value,
                'processing_time': game_state.ai_analysis.processing_time,
                'feature_summary': game_state.ai_analysis.feature_summary,
                'suggested_regions': game_state.ai_analysis.suggested_regions[:3]  # Top 3
            }
        
        # Add guess info if available
        if game_state.ai_guess:
            state_dict['ai_guess'] = {
                'lat': game_state.ai_guess[0],
                'lon': game_state.ai_guess[1]
            }
        
        return state_dict
    
    def get_game_modes(self) -> Dict:
        """Get available game modes"""
        return self.game_modes
    
    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """Get recent game results"""
        recent_results = self.game_results[-limit:] if self.game_results else []
        return [asdict(result) for result in recent_results]
    
    async def cleanup_inactive_games(self, max_age_minutes: int = 30):
        """Clean up games that have been inactive too long"""
        current_time = time.time()
        inactive_games = []
        
        for game_id, game_state in self.active_games.items():
            age_minutes = (current_time - game_state.start_time) / 60
            if age_minutes > max_age_minutes:
                inactive_games.append(game_id)
        
        for game_id in inactive_games:
            del self.active_games[game_id]
            logger.info(f"Cleaned up inactive game: {game_id}")
        
        return len(inactive_games)
    
    def _create_mock_analysis_result(self, lat: float, lon: float, difficulty: str):
        """Create mock CV analysis result based on known location"""
        from app.core.cv_pipeline import CVAnalysisResult, ConfidenceLevel, DetectionResult
        
        # Create realistic detections based on geographic location
        detections = []
        
        # Add mock detections based on location characteristics
        if abs(lat) < 30:  # Tropical regions
            detections.append(DetectionResult(
                feature_type="vegetation",
                confidence=0.8,
                bounding_box=(100, 50, 200, 150),
                metadata={"vegetation_type": "tropical", "coverage": 0.6},
                geographic_hints=["tropical_region", "equatorial_climate"]
            ))
        
        if abs(lat) > 50:  # Northern regions
            detections.append(DetectionResult(
                feature_type="architecture",
                confidence=0.7,
                bounding_box=(50, 100, 300, 200),
                metadata={"building_type": "northern_style", "roof_type": "steep"},
                geographic_hints=["northern_hemisphere", "cold_climate", "europe_or_canada"]
            ))
        
        # Urban vs rural detection
        if "urban" in difficulty or abs(lat - 40.7) < 1 and abs(lon + 74) < 1:  # NYC area
            detections.append(DetectionResult(
                feature_type="vehicle",
                confidence=0.9,
                bounding_box=(200, 300, 100, 80),
                metadata={"vehicle_type": "yellow_taxi", "urban_density": "high"},
                geographic_hints=["urban_center", "new_york", "north_america"]
            ))
        
        # Determine confidence based on difficulty
        confidence_map = {
            "easy": 0.8,
            "medium": 0.6,
            "hard": 0.4,
            "expert": 0.3
        }
        
        overall_confidence = confidence_map.get(difficulty, 0.6)
        
        if overall_confidence > 0.7:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence > 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        # Aggregate geographic hints
        all_hints = []
        for detection in detections:
            all_hints.extend(detection.geographic_hints)
        
        return CVAnalysisResult(
            detections=detections,
            overall_confidence=overall_confidence,
            processing_time=2.5,
            suggested_regions=list(set(all_hints))[:5],
            confidence_level=confidence_level,
            feature_summary={d.feature_type: 1 for d in detections}
        )
    
    def _generate_streetview_ai_guess(self, analysis, difficulty: str) -> Tuple[float, float]:
        """Generate AI guess for Street View games with difficulty-based accuracy"""
        
        # Base accuracy on difficulty level
        accuracy_map = {
            "easy": 50.0,      # Within 50km
            "medium": 150.0,   # Within 150km
            "hard": 500.0,     # Within 500km
            "expert": 1500.0   # Within 1500km
        }
        
        max_error_km = accuracy_map.get(difficulty, 150.0)
        
        # Use the standard guess generation but with difficulty-based error
        suggested_regions = analysis.suggested_regions
        confidence = analysis.overall_confidence
        
        # Start with some reasonable global location
        lat, lon = 45.0, 10.0  # Central Europe as default
        
        # Adjust based on detected features
        if 'tropical_region' in suggested_regions:
            lat = self._add_noise(15.0, confidence)
            lon = self._add_noise(-90.0, confidence)
        elif 'northern_hemisphere' in suggested_regions:
            lat = self._add_noise(55.0, confidence)
            lon = self._add_noise(15.0, confidence)
        elif 'urban_center' in suggested_regions:
            # Major city locations
            major_cities = [(40.7, -74), (51.5, -0.1), (48.9, 2.3), (35.7, 139.7)]
            city_lat, city_lon = major_cities[hash(str(suggested_regions)) % len(major_cities)]
            lat = self._add_noise(city_lat, confidence)
            lon = self._add_noise(city_lon, confidence)
        
        # Apply difficulty-based final error
        error_factor = max_error_km / 111.32  # Convert km to degrees (roughly)
        final_lat_error = (hash(str(analysis.detections)) % 21 - 10) * error_factor / 10
        final_lon_error = (hash(str(analysis.detections[::-1])) % 21 - 10) * error_factor / 10
        
        final_lat = max(-90, min(90, lat + final_lat_error))
        final_lon = max(-180, min(180, lon + final_lon_error))
        
        return (final_lat, final_lon)
