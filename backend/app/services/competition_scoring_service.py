"""
Competition Scoring Service for Human vs AI GeoGuessr
Handles scoring, comparisons, and round-by-round competition tracking
"""

import math
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class CompetitionScoringService:
    """Manages scoring for Human vs AI competition"""
    
    def __init__(self):
        self.max_score_per_round = 5000
        self.distance_decay_factor = 1000  # Distance in km where score halves
        
    def calculate_score(self, distance_km: float) -> int:
        """Calculate score based on distance from correct location"""
        if distance_km <= 0:
            return self.max_score_per_round
        
        # Exponential decay formula: score = max_score * e^(-distance/decay_factor)
        score = self.max_score_per_round * math.exp(-distance_km / self.distance_decay_factor)
        return max(0, int(round(score)))
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        # Convert latitude and longitude to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    async def score_round(self, round_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a complete round with both human and AI guesses"""
        try:
            actual_location = round_data['actual_location']
            human_guess = round_data.get('human_guess')
            ai_guess = round_data.get('ai_guess')
            round_number = round_data.get('round_number', 1)
            time_taken = round_data.get('time_taken', 30)
            
            results = {
                'round_number': round_number,
                'actual_location': actual_location,
                'human': {},
                'ai': {},
                'comparison': {},
                'round_winner': None
            }
            
            # Score human guess
            if human_guess:
                human_distance = self.calculate_distance(
                    actual_location['lat'], actual_location['lon'],
                    human_guess['lat'], human_guess['lon']
                )
                human_score = self.calculate_score(human_distance)
                
                results['human'] = {
                    'guess': human_guess,
                    'distance_km': round(human_distance, 2),
                    'score': human_score,
                    'time_taken': time_taken,
                    'accuracy_rating': self._get_accuracy_rating(human_distance)
                }
            
            # Score AI guess
            if ai_guess:
                ai_distance = self.calculate_distance(
                    actual_location['lat'], actual_location['lon'],
                    ai_guess['lat'], ai_guess['lon']
                )
                ai_score = self.calculate_score(ai_distance)
                
                results['ai'] = {
                    'guess': {
                        'lat': ai_guess['lat'],
                        'lon': ai_guess['lon']
                    },
                    'distance_km': round(ai_distance, 2),
                    'score': ai_score,
                    'time_taken': ai_guess.get('time_taken', 5),
                    'confidence': ai_guess.get('confidence', 0),
                    'reasoning': ai_guess.get('reasoning', ''),
                    'analysis_quality': ai_guess.get('analysis_quality', 'Unknown'),
                    'accuracy_rating': self._get_accuracy_rating(ai_distance)
                }
            
            # Compare results
            if human_guess and ai_guess:
                results['comparison'] = self._compare_performances(
                    results['human'], results['ai']
                )
                
                # Determine round winner
                human_score = results['human']['score']
                ai_score = results['ai']['score']
                
                if human_score > ai_score:
                    results['round_winner'] = 'human'
                elif ai_score > human_score:
                    results['round_winner'] = 'ai'
                else:
                    results['round_winner'] = 'tie'
            
            return results
            
        except Exception as e:
            logger.error(f"Round scoring failed: {e}")
            return self._empty_round_result(round_data.get('round_number', 1))
    
    def _get_accuracy_rating(self, distance_km: float) -> str:
        """Get human-readable accuracy rating"""
        if distance_km < 25:
            return "🎯 Excellent"
        elif distance_km < 100:
            return "🌟 Very Good"
        elif distance_km < 500:
            return "👍 Good"
        elif distance_km < 1500:
            return "📍 Fair"
        elif distance_km < 5000:
            return "🌍 Poor"
        else:
            return "❌ Very Poor"
    
    def _compare_performances(self, human_result: Dict, ai_result: Dict) -> Dict[str, Any]:
        """Compare human vs AI performance for the round"""
        comparison = {
            'score_difference': human_result['score'] - ai_result['score'],
            'distance_difference': ai_result['distance_km'] - human_result['distance_km'],
            'time_difference': human_result['time_taken'] - ai_result['time_taken']
        }
        
        # Add comparative analysis
        if comparison['score_difference'] > 0:
            comparison['winner'] = 'human'
            comparison['margin'] = f"Human won by {comparison['score_difference']} points"
        elif comparison['score_difference'] < 0:
            comparison['winner'] = 'ai'
            comparison['margin'] = f"AI won by {abs(comparison['score_difference'])} points"
        else:
            comparison['winner'] = 'tie'
            comparison['margin'] = "Perfect tie!"
        
        # Distance comparison
        if comparison['distance_difference'] > 0:
            comparison['closer_guess'] = 'human'
            comparison['distance_advantage'] = f"Human was {comparison['distance_difference']:.1f}km closer"
        elif comparison['distance_difference'] < 0:
            comparison['closer_guess'] = 'ai'
            comparison['distance_advantage'] = f"AI was {abs(comparison['distance_difference']):.1f}km closer"
        else:
            comparison['closer_guess'] = 'tie'
            comparison['distance_advantage'] = "Same distance"
        
        return comparison
    
    async def calculate_game_summary(self, round_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate final game summary and overall winner"""
        try:
            summary = {
                'total_rounds': len(round_results),
                'human_total_score': 0,
                'ai_total_score': 0,
                'human_wins': 0,
                'ai_wins': 0,
                'ties': 0,
                'round_breakdown': [],
                'performance_analysis': {},
                'final_winner': None
            }
            
            human_distances = []
            ai_distances = []
            human_times = []
            ai_times = []
            
            for round_result in round_results:
                # Accumulate scores
                if 'human' in round_result and 'score' in round_result['human']:
                    summary['human_total_score'] += round_result['human']['score']
                    human_distances.append(round_result['human']['distance_km'])
                    human_times.append(round_result['human']['time_taken'])
                
                if 'ai' in round_result and 'score' in round_result['ai']:
                    summary['ai_total_score'] += round_result['ai']['score']
                    ai_distances.append(round_result['ai']['distance_km'])
                    ai_times.append(round_result['ai']['time_taken'])
                
                # Count wins
                winner = round_result.get('round_winner')
                if winner == 'human':
                    summary['human_wins'] += 1
                elif winner == 'ai':
                    summary['ai_wins'] += 1
                else:
                    summary['ties'] += 1
                
                # Add round summary
                summary['round_breakdown'].append({
                    'round': round_result.get('round_number'),
                    'winner': winner,
                    'human_score': round_result.get('human', {}).get('score', 0),
                    'ai_score': round_result.get('ai', {}).get('score', 0),
                    'score_difference': round_result.get('comparison', {}).get('score_difference', 0)
                })
            
            # Determine final winner
            if summary['human_total_score'] > summary['ai_total_score']:
                summary['final_winner'] = 'human'
                summary['victory_margin'] = summary['human_total_score'] - summary['ai_total_score']
            elif summary['ai_total_score'] > summary['human_total_score']:
                summary['final_winner'] = 'ai'
                summary['victory_margin'] = summary['ai_total_score'] - summary['human_total_score']
            else:
                summary['final_winner'] = 'tie'
                summary['victory_margin'] = 0
            
            # Performance analysis
            if human_distances and ai_distances:
                summary['performance_analysis'] = {
                    'human_avg_distance': round(sum(human_distances) / len(human_distances), 2),
                    'ai_avg_distance': round(sum(ai_distances) / len(ai_distances), 2),
                    'human_avg_time': round(sum(human_times) / len(human_times), 1),
                    'ai_avg_time': round(sum(ai_times) / len(ai_times), 1),
                    'human_best_round': min(human_distances),
                    'ai_best_round': min(ai_distances),
                    'human_worst_round': max(human_distances),
                    'ai_worst_round': max(ai_distances)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Game summary calculation failed: {e}")
            return self._empty_game_summary()
    
    def _empty_round_result(self, round_number: int) -> Dict[str, Any]:
        """Return empty round result structure"""
        return {
            'round_number': round_number,
            'actual_location': {},
            'human': {},
            'ai': {},
            'comparison': {},
            'round_winner': None,
            'error': 'Scoring failed'
        }
    
    def _empty_game_summary(self) -> Dict[str, Any]:
        """Return empty game summary structure"""
        return {
            'total_rounds': 0,
            'human_total_score': 0,
            'ai_total_score': 0,
            'human_wins': 0,
            'ai_wins': 0,
            'ties': 0,
            'round_breakdown': [],
            'performance_analysis': {},
            'final_winner': None,
            'error': 'Summary calculation failed'
        }
    
    def get_score_breakdown(self, distance_km: float) -> Dict[str, Any]:
        """Get detailed score breakdown for educational purposes"""
        score = self.calculate_score(distance_km)
        
        return {
            'distance_km': round(distance_km, 2),
            'score': score,
            'max_possible': self.max_score_per_round,
            'percentage': round((score / self.max_score_per_round) * 100, 1),
            'accuracy_rating': self._get_accuracy_rating(distance_km),
            'score_formula': f"5000 × e^(-{distance_km:.1f}/1000) = {score}"
        }
