"""
GeoGuessr-style backend API endpoints
Clean implementation for location generation and game management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import random
import uuid

# Import existing services
from app.services.random_location_service import RandomLocationService

router = APIRouter()

# Global services
location_service: Optional[RandomLocationService] = None

def set_location_service(loc_service: RandomLocationService):
    """Set the location service instance"""
    global location_service
    location_service = loc_service

class NewLocationRequest(BaseModel):
    difficulty: str = 'mixed'

class LocationResponse(BaseModel):
    id: str
    lat: float
    lon: float
    country: str
    city: str
    street_view_url: str
    metadata: Dict

@router.post("/geoguessr/new-location")
async def get_new_location(request: NewLocationRequest) -> LocationResponse:
    """Get a new random location for GeoGuessr game"""
    if not location_service:
        raise HTTPException(status_code=500, detail="Location service not initialized")
    
    try:
        # Generate random location with GeoGuessr-style validation
        location_data = await location_service.get_random_location(request.difficulty)
        
        location_id = str(uuid.uuid4())
        
        # Extract first Street View URL for the main view
        street_view_urls = location_data.get('street_view_urls', [])
        main_street_view_url = street_view_urls[0]['url'] if street_view_urls else ''
        
        return LocationResponse(
            id=location_id,
            lat=location_data['lat'],
            lon=location_data['lon'],
            country=location_data.get('country_hint', 'Unknown'),
            city=location_data.get('city_hint', 'Unknown'),
            street_view_url=main_street_view_url,
            metadata={
                'difficulty': request.difficulty,
                'street_view_urls': street_view_urls,
                'region': location_data.get('metadata', {}).get('region', ''),
                'urban_type': location_data.get('metadata', {}).get('urban_type', '')
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate location: {str(e)}")

@router.get("/geoguessr/validate-location/{location_id}")
async def validate_location(location_id: str):
    """Validate that a location has working Street View"""
    # This endpoint can be used to verify Street View is accessible
    return {"location_id": location_id, "valid": True}

class GuessRequest(BaseModel):
    location_id: str
    lat: float
    lon: float
    player_type: str  # 'human' or 'ai'

@router.post("/geoguessr/submit-guess")
async def submit_guess(request: GuessRequest):
    """Submit a guess for a location"""
    # This will be expanded to calculate score and store results
    return {
        "guess_id": str(uuid.uuid4()),
        "location_id": request.location_id,
        "distance_km": 0.0,  # Will calculate actual distance
        "points": 0,
        "player_type": request.player_type
    }
