"""
API Routes for GeoCV Backend
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import aiofiles
import uuid
from pydantic import BaseModel

# Import services (will be injected)
from app.services.game_service import GameService
from app.services.streetview_service import StreetViewService, LocationDifficulty

router = APIRouter()

# Global service instances (will be set in main.py)
game_service: Optional[GameService] = None
streetview_service: Optional[StreetViewService] = None

def set_services(game_svc: GameService, streetview_svc: StreetViewService):
    """Set the service instances"""
    global game_service, streetview_service
    game_service = game_svc
    streetview_service = streetview_svc

# Pydantic models for request/response
class StartGameRequest(BaseModel):
    game_mode: str = "classic"
    actual_location: Optional[tuple] = None

class SubmitGuessRequest(BaseModel):
    lat: float
    lon: float

@router.get("/game-modes")
async def get_game_modes():
    """Get available game modes"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    return game_service.get_game_modes()

@router.post("/games/start")
async def start_game(
    image: UploadFile = File(...),
    game_mode: str = Form("classic"),
    actual_lat: Optional[float] = Form(None),
    actual_lon: Optional[float] = Form(None)
):
    """Start a new game with uploaded image"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded image
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(upload_dir, filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await image.read()
            await f.write(content)
        
        # Prepare actual location if provided
        actual_location = None
        if actual_lat is not None and actual_lon is not None:
            actual_location = (actual_lat, actual_lon)
        
        # Start the game
        game_id = await game_service.start_new_game(
            image_path=file_path,
            game_mode=game_mode,
            actual_location=actual_location
        )
        
        return {
            "game_id": game_id,
            "status": "started",
            "message": "Game started! AI is analyzing the image..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")

@router.post("/games/start-streetview")
async def start_streetview_game(
    game_mode: str = Form("classic"),
    difficulty: str = Form("medium"),
    region: Optional[str] = Form(None)
):
    """Start a new game with a random Street View location"""
    if not game_service or not streetview_service:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Get random Street View location
        from app.services.streetview_service import LocationDifficulty
        
        try:
            difficulty_enum = LocationDifficulty(difficulty.lower())
        except ValueError:
            difficulty_enum = LocationDifficulty.MEDIUM
        
        streetview_location = await streetview_service.get_random_location(
            difficulty=difficulty_enum,
            region_preference=region
        )
        
        # Convert to dict format for game service
        location_data = {
            "location": {
                "lat": streetview_location.lat,
                "lon": streetview_location.lon,
                "pano_id": streetview_location.pano_id,
                "heading": streetview_location.heading,
                "pitch": streetview_location.pitch,
                "fov": streetview_location.fov
            },
            "metadata": {
                "difficulty": streetview_location.difficulty.value,
                "country": streetview_location.country,
                "region": streetview_location.region,
                "description": streetview_location.description
            },
            "image_url": streetview_location.image_url
        }
        
        # Start the Street View game
        game_id = await game_service.start_streetview_game(
            streetview_location=location_data,
            game_mode=game_mode
        )
        
        return {
            "game_id": game_id,
            "streetview_location": location_data,
            "status": "started",
            "message": "Street View game started! AI is analyzing the location..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start Street View game: {str(e)}")

@router.get("/games/{game_id}/status")
async def get_game_status(game_id: str):
    """Get current status of a game"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    state = game_service.get_game_state(game_id)
    if not state:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return state

@router.post("/games/{game_id}/guess")
async def submit_guess(game_id: str, guess: SubmitGuessRequest):
    """Submit human player guess"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    result = await game_service.submit_human_guess(game_id, guess.lat, guess.lon)
    
    if result is None:
        raise HTTPException(status_code=400, detail="Cannot submit guess for this game")
    
    return {
        "result": result.__dict__,
        "message": f"Game completed! Winner: {result.winner}"
    }

@router.get("/games/results/recent")
async def get_recent_results(limit: int = 10):
    """Get recent game results"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    results = game_service.get_recent_results(limit)
    return {"results": results}

@router.post("/analyze-image")
async def analyze_image_endpoint(image: UploadFile = File(...)):
    """Analyze an image without starting a game (for testing)"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded image temporarily
        upload_dir = "temp"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"temp_{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(upload_dir, filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await image.read()
            await f.write(content)
        
        # Run CV analysis
        analysis_result = await game_service.cv_pipeline.analyze_image(file_path)
        
        # Clean up temp file
        os.remove(file_path)
        
        # Convert result to dict for JSON response
        response_data = {
            "detections": [
                {
                    "feature_type": d.feature_type,
                    "confidence": d.confidence,
                    "bounding_box": d.bounding_box,
                    "metadata": d.metadata,
                    "geographic_hints": d.geographic_hints
                }
                for d in analysis_result.detections
            ],
            "overall_confidence": analysis_result.overall_confidence,
            "processing_time": analysis_result.processing_time,
            "suggested_regions": analysis_result.suggested_regions,
            "confidence_level": analysis_result.confidence_level.value,
            "feature_summary": analysis_result.feature_summary
        }
        
        return response_data
        
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/health/cv-pipeline")
async def check_cv_pipeline():
    """Check CV pipeline health"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    pipeline_status = {
        "initialized": game_service.cv_pipeline.initialized,
        "cascade_classifiers": len(game_service.cv_pipeline.cascade_classifiers),
        "feature_extractors": len(game_service.cv_pipeline.feature_extractors)
    }
    
    return {
        "status": "healthy" if pipeline_status["initialized"] else "not_ready",
        "details": pipeline_status
    }

@router.post("/admin/cleanup")
async def cleanup_inactive_games(max_age_minutes: int = 30):
    """Admin endpoint to cleanup inactive games"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    cleaned_count = await game_service.cleanup_inactive_games(max_age_minutes)
    
    return {
        "message": f"Cleaned up {cleaned_count} inactive games",
        "cleaned_count": cleaned_count
    }

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    if not game_service:
        raise HTTPException(status_code=500, detail="Game service not initialized")
    
    stats = {
        "active_games": len(game_service.active_games),
        "total_games_played": len(game_service.game_results),
        "cv_pipeline_initialized": game_service.cv_pipeline.initialized,
        "streetview_initialized": streetview_service.initialized if streetview_service else False
    }
    
    if game_service.game_results:
        # Calculate win rates
        ai_wins = sum(1 for r in game_service.game_results if r.winner == 'ai')
        human_wins = sum(1 for r in game_service.game_results if r.winner == 'human')
        ties = sum(1 for r in game_service.game_results if r.winner == 'tie')
        
        total_games = len(game_service.game_results)
        stats.update({
            "ai_win_rate": ai_wins / total_games if total_games > 0 else 0,
            "human_win_rate": human_wins / total_games if total_games > 0 else 0,
            "tie_rate": ties / total_games if total_games > 0 else 0,
            "average_ai_score": sum(r.ai_score for r in game_service.game_results) / total_games,
            "average_human_score": sum(r.human_score for r in game_service.game_results) / total_games
        })
    
    return stats

# Street View API endpoints
@router.get("/streetview/random")
async def get_random_streetview_location(
    difficulty: str = "medium",
    region: Optional[str] = None
):
    """Get a random Street View location for the game"""
    if not streetview_service:
        raise HTTPException(status_code=500, detail="Street View service not initialized")
    
    try:
        # Parse difficulty
        difficulty_enum = LocationDifficulty(difficulty.lower())
    except ValueError:
        difficulty_enum = LocationDifficulty.MEDIUM
    
    try:
        location = await streetview_service.get_random_location(
            difficulty=difficulty_enum,
            region_preference=region
        )
        
        return {
            "location": {
                "lat": location.lat,
                "lon": location.lon,
                "pano_id": location.pano_id,
                "heading": location.heading,
                "pitch": location.pitch,
                "fov": location.fov
            },
            "metadata": {
                "difficulty": location.difficulty.value,
                "country": location.country,
                "region": location.region,
                "description": location.description
            },
            "image_url": location.image_url,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate location: {str(e)}")

@router.get("/streetview/validate")
async def validate_streetview_location(lat: float, lon: float):
    """Validate if Street View is available at given coordinates"""
    if not streetview_service:
        raise HTTPException(status_code=500, detail="Street View service not initialized")
    
    try:
        is_valid = await streetview_service.validate_location(lat, lon)
        metadata = await streetview_service.get_streetview_metadata(lat, lon)
        
        return {
            "valid": is_valid,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/streetview/image")
async def get_streetview_image(
    lat: float,
    lon: float,
    size: str = "640x640",
    fov: int = 90,
    heading: int = 0,
    pitch: int = 0
):
    """Get Street View image URL for given parameters"""
    if not streetview_service:
        raise HTTPException(status_code=500, detail="Street View service not initialized")
    
    try:
        image_url = streetview_service._generate_streetview_url(
            lat=lat, lon=lon, size=size, fov=fov, heading=heading, pitch=pitch
        )
        
        return {
            "image_url": image_url,
            "parameters": {
                "lat": lat,
                "lon": lon,
                "size": size,
                "fov": fov,
                "heading": heading,
                "pitch": pitch
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image URL: {str(e)}")

@router.get("/streetview/challenge")
async def get_challenge_locations():
    """Get curated challenge locations"""
    if not streetview_service:
        raise HTTPException(status_code=500, detail="Street View service not initialized")
    
    try:
        locations = await streetview_service.get_curated_challenge_locations()
        
        return {
            "locations": [
                {
                    "lat": loc.lat,
                    "lon": loc.lon,
                    "difficulty": loc.difficulty.value,
                    "description": loc.description,
                    "image_url": loc.image_url
                }
                for loc in locations
            ],
            "count": len(locations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get challenge locations: {str(e)}")
