"""
AI Guesser Service - generates location guesses based on OpenCV analysis
Uses computer vision insights to make educated geographical guesses
"""

import random
import math
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class AIGuesserService:
    """AI service that generates location guesses from OpenCV analysis"""
    
    def __init__(self):
        self.geographical_knowledge = self._load_geographical_knowledge()
        self.confidence_weights = {
            'architecture': 0.3,
            'vegetation': 0.25,
            'infrastructure': 0.2,
            'text_signs': 0.15,
            'vehicles': 0.1
        }
        
    def _load_geographical_knowledge(self) -> Dict[str, Any]:
        """Load geographical patterns and region mappings"""
        return {
            'architecture_regions': {
                'red_tile_roof': ['Mediterranean', 'Spain', 'Italy', 'Portugal', 'Southern France'],
                'wooden_house': ['Scandinavia', 'North America', 'Northern Europe'],
                'concrete_modern': ['Urban areas', 'Developed countries'],
                'traditional_asian': ['East Asia', 'Southeast Asia'],
                'colonial_style': ['Former colonies', 'Americas', 'Australia']
            },
            'vegetation_climate': {
                'palm_trees': ['Tropical', 'Subtropical', 'Coastal warm regions'],
                'pine_forests': ['Northern latitudes', 'Mountainous regions'],
                'deciduous': ['Temperate regions', 'North America', 'Europe'],
                'desert_plants': ['Arid regions', 'Southwest US', 'Australia', 'Middle East'],
                'rice_fields': ['Asia', 'Wet tropical regions']
            },
            'infrastructure_hints': {
                'left_driving': ['UK', 'Australia', 'Japan', 'India', 'South Africa'],
                'right_driving': ['Most of world', 'Europe', 'Americas'],
                'yellow_lines': ['US', 'Canada'],
                'white_lines': ['Europe', 'Most countries'],
                'bollards': ['Europe', 'Specific countries have unique designs']
            },
            'country_coordinates': {
                'United States': (39.8283, -98.5795),
                'Canada': (56.1304, -106.3468),
                'United Kingdom': (55.3781, -3.4360),
                'Australia': (-25.2744, 133.7751),
                'Japan': (36.2048, 138.2529),
                'Germany': (51.1657, 10.4515),
                'France': (46.2276, 2.2137),
                'Spain': (40.4637, -3.7492),
                'Italy': (41.8719, 12.5674),
                'Brazil': (-14.2350, -51.9253),
                'India': (20.5937, 78.9629),
                'China': (35.8617, 104.1954),
                'Russia': (61.5240, 105.3188),
                'Mexico': (23.6345, -102.5528),
                'Argentina': (-38.4161, -63.6167),
                'South Africa': (-30.5595, 22.9375),
                'Sweden': (60.1282, 18.6435),
                'Norway': (60.4720, 8.4689),
                'Finland': (61.9241, 25.7482),
                'Poland': (51.9194, 19.1451),
                'Czech Republic': (49.8175, 15.4730),
                'Netherlands': (52.1326, 5.2913),
                'Belgium': (50.5039, 4.4699),
                'Switzerland': (46.8182, 8.2275),
                'Austria': (47.5162, 14.5501),
                'Portugal': (39.3999, -8.2245),
                'Turkey': (38.9637, 35.2433),
                'Greece': (39.0742, 21.8243),
                'Thailand': (15.8700, 100.9925),
                'Vietnam': (14.0583, 108.2772),
                'Indonesia': (-0.7893, 113.9213),
                'Philippines': (12.8797, 121.7740),
                'Malaysia': (4.2105, 101.9758),
                'Singapore': (1.3521, 103.8198),
                'South Korea': (35.9078, 127.7669),
                'Taiwan': (23.6978, 120.9605),
                'New Zealand': (-40.9006, 174.8860),
                'Chile': (-35.6751, -71.5430),
                'Peru': (-9.1900, -75.0152),
                'Colombia': (4.5709, -74.2973),
                'Venezuela': (6.4238, -66.5897),
                'Ecuador': (-1.8312, -78.1834),
                'Uruguay': (-32.5228, -55.7658),
                'Paraguay': (-23.4425, -58.4438),
                'Bolivia': (-16.2902, -63.5887),
                'Egypt': (26.0975, 30.0444),
                'Morocco': (31.7917, -7.0926),
                'Algeria': (28.0339, 1.6596),
                'Tunisia': (33.8869, 9.5375),
                'Kenya': (-0.0236, 37.9062),
                'Nigeria': (9.0820, 8.6753),
                'Ghana': (7.9465, -1.0232),
                'Ethiopia': (9.1450, 40.4897),
                'Tanzania': (-6.3690, 34.8888),
                'Uganda': (1.3733, 32.2903),
                'Botswana': (-22.3285, 24.6849),
                'Namibia': (-22.9576, 18.4904),
                'Zimbabwe': (-19.0154, 29.1549),
                'Zambia': (-13.1339, 27.8493)
            }
        }
    
    async def generate_guess(self, cv_analysis: Dict[str, Any], time_elapsed: int = 0) -> Dict[str, Any]:
        """Generate an AI guess based on OpenCV analysis"""
        try:
            # Extract features from CV analysis
            detected_objects = cv_analysis.get('objects', [])
            geographical_hints = cv_analysis.get('geographical_hints', [])
            color_analysis = cv_analysis.get('color_analysis', {})
            
            # Analyze patterns and generate confidence scores
            region_probabilities = self._analyze_geographical_patterns(
                detected_objects, geographical_hints, color_analysis
            )
            
            # Generate specific coordinate guess
            guess_lat, guess_lon = self._select_coordinates(region_probabilities)
            
            # Calculate confidence based on analysis quality
            confidence = self._calculate_confidence(
                detected_objects, geographical_hints, time_elapsed
            )
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(
                detected_objects, geographical_hints, region_probabilities
            )
            
            return {
                'lat': guess_lat,
                'lon': guess_lon,
                'confidence': confidence,
                'reasoning': reasoning,
                'analysis_quality': self._assess_analysis_quality(cv_analysis),
                'time_taken': min(time_elapsed, 30),  # AI thinks quickly
                'region_probabilities': region_probabilities
            }
            
        except Exception as e:
            logger.error(f"AI guess generation failed: {e}")
            return self._fallback_guess()
    
    def _analyze_geographical_patterns(self, objects: List[Dict], hints: List[str], colors: Dict) -> Dict[str, float]:
        """Analyze CV data to determine regional probabilities"""
        region_scores = {}
        
        # Analyze detected objects
        for obj in objects:
            obj_type = obj.get('type', '')
            confidence = obj.get('confidence', 0)
            
            if obj_type == 'vegetation':
                # High vegetation suggests temperate/tropical climates
                self._add_region_score(region_scores, 'temperate_regions', confidence * 0.3)
                
            elif obj_type == 'vehicle':
                # Vehicle types can indicate regions
                self._add_region_score(region_scores, 'developed_countries', confidence * 0.2)
                
            elif obj_type == 'building':
                # Building styles indicate architectural regions
                self._add_region_score(region_scores, 'urban_areas', confidence * 0.4)
        
        # Analyze geographical hints
        for hint in hints:
            if 'vegetation' in hint.lower():
                if 'tropical' in hint.lower():
                    self._add_region_score(region_scores, 'tropical_regions', 0.5)
                elif 'temperate' in hint.lower():
                    self._add_region_score(region_scores, 'temperate_regions', 0.4)
                    
            elif 'bright lighting' in hint.lower():
                self._add_region_score(region_scores, 'sunny_climates', 0.3)
                
            elif 'vehicles' in hint.lower():
                self._add_region_score(region_scores, 'developed_countries', 0.3)
                
            elif 'buildings' in hint.lower():
                self._add_region_score(region_scores, 'urban_areas', 0.4)
        
        # Normalize scores
        total_score = sum(region_scores.values()) or 1
        return {region: score / total_score for region, score in region_scores.items()}
    
    def _add_region_score(self, scores: Dict[str, float], region: str, score: float):
        """Add score to region"""
        scores[region] = scores.get(region, 0) + score
    
    def _select_coordinates(self, region_probabilities: Dict[str, float]) -> Tuple[float, float]:
        """Select specific coordinates based on regional analysis"""
        
        # Map region types to actual countries/coordinates
        region_to_countries = {
            'temperate_regions': ['United States', 'Germany', 'France', 'United Kingdom'],
            'tropical_regions': ['Brazil', 'Thailand', 'Indonesia', 'Philippines'],
            'developed_countries': ['United States', 'Germany', 'Japan', 'Australia'],
            'urban_areas': ['United States', 'Germany', 'United Kingdom', 'Japan'],
            'sunny_climates': ['Australia', 'Spain', 'Brazil', 'Mexico']
        }
        
        # Select most likely countries
        country_scores = {}
        for region, probability in region_probabilities.items():
            countries = region_to_countries.get(region, [])
            for country in countries:
                country_scores[country] = country_scores.get(country, 0) + probability
        
        # If no clear winner, use global distribution
        if not country_scores:
            country_scores = {'United States': 0.3, 'Germany': 0.2, 'Australia': 0.15, 'United Kingdom': 0.15, 'Brazil': 0.2}
        
        # Select country probabilistically
        countries = list(country_scores.keys())
        weights = list(country_scores.values())
        selected_country = random.choices(countries, weights=weights)[0]
        
        # Get base coordinates for country
        base_lat, base_lon = self.geographical_knowledge['country_coordinates'].get(
            selected_country, (0, 0)
        )
        
        # Add some random variance to avoid always guessing exact center
        lat_variance = random.uniform(-2, 2)  # ±2 degrees
        lon_variance = random.uniform(-3, 3)  # ±3 degrees
        
        final_lat = max(-90, min(90, base_lat + lat_variance))
        final_lon = max(-180, min(180, base_lon + lon_variance))
        
        return final_lat, final_lon
    
    def _calculate_confidence(self, objects: List[Dict], hints: List[str], time_elapsed: int) -> float:
        """Calculate AI confidence in the guess"""
        base_confidence = 0.3  # Baseline AI confidence
        
        # More objects detected = higher confidence
        object_bonus = min(0.3, len(objects) * 0.05)
        
        # Quality of hints
        hint_bonus = min(0.25, len(hints) * 0.05)
        
        # High-confidence objects boost overall confidence
        high_conf_objects = [obj for obj in objects if obj.get('confidence', 0) > 0.7]
        quality_bonus = min(0.15, len(high_conf_objects) * 0.03)
        
        # Time factor (AI doesn't need much time)
        time_factor = 1.0  # AI is confident regardless of time
        
        total_confidence = (base_confidence + object_bonus + hint_bonus + quality_bonus) * time_factor
        return min(0.95, max(0.1, total_confidence))  # Clamp between 0.1 and 0.95
    
    def _generate_reasoning(self, objects: List[Dict], hints: List[str], regions: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the AI's guess"""
        reasoning_parts = []
        
        # Object-based reasoning
        if objects:
            object_types = [obj['type'] for obj in objects]
            if 'vegetation' in object_types:
                reasoning_parts.append("🌱 Detected vegetation patterns suggest temperate climate")
            if 'building' in object_types:
                reasoning_parts.append("🏢 Building architecture indicates developed urban area")
            if 'vehicle' in object_types:
                reasoning_parts.append("🚗 Vehicle presence suggests developed infrastructure")
        
        # Hint-based reasoning
        if hints:
            reasoning_parts.append(f"🔍 Geographic analysis: {hints[0]}")
        
        # Regional probability reasoning
        if regions:
            top_region = max(regions.items(), key=lambda x: x[1])
            reasoning_parts.append(f"🎯 Primary region indicator: {top_region[0]} ({top_region[1]:.1%} confidence)")
        
        if not reasoning_parts:
            reasoning_parts = ["🤖 Making educated guess based on global location patterns"]
        
        return " | ".join(reasoning_parts[:3])  # Limit to top 3 reasons
    
    def _assess_analysis_quality(self, cv_analysis: Dict[str, Any]) -> str:
        """Assess the quality of the CV analysis"""
        objects = cv_analysis.get('objects', [])
        hints = cv_analysis.get('geographical_hints', [])
        
        total_features = len(objects) + len(hints)
        
        if total_features >= 5:
            return "High"
        elif total_features >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _fallback_guess(self) -> Dict[str, Any]:
        """Fallback guess when analysis fails"""
        # Random guess with low confidence
        lat = random.uniform(-60, 70)  # Avoid extreme poles
        lon = random.uniform(-170, 170)
        
        return {
            'lat': lat,
            'lon': lon,
            'confidence': 0.1,
            'reasoning': "🤖 Fallback guess - analysis unavailable",
            'analysis_quality': "Low",
            'time_taken': 5,
            'region_probabilities': {}
        }
