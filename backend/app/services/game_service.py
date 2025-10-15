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
    image_path: str
    ai_analysis: Optional[CVAnalysisResult]
    ai_guess: Optional[Tuple[float, float]]  # lat, lon
    human_guess: Optional[Tuple[float, float]]
    actual_location: Optional[Tuple[float, float]]
    game_mode: str
    time_limit: Optional[int]
    start_time: float
    status: str  # 'analyzing', 'waiting_human', 'completed'
    ai_confidence_display: str

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
        """Start a new game session"""
        
        game_id = str(uuid.uuid4())
        
        # Validate game mode
        if game_mode not in self.game_modes:
            game_mode = 'classic'
        
        # Initialize game state
        game_state = GameState(
            game_id=game_id,
            image_path=image_path,
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
            'elapsed_time': time.time() - game_state.start_time
        }
        
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
