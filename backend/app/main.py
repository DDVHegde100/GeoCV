from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv

from app.api.routes import router, set_services
from app.core.cv_pipeline import CVPipeline
from app.services.game_service import GameService
from app.services.streetview_service import StreetViewService
from app.services.websocket_service import WebSocketService

# Load environment variables
load_dotenv()

# Global service instances
cv_pipeline = None
game_service = None
streetview_service = None
websocket_service = WebSocketService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global cv_pipeline, game_service, streetview_service
    cv_pipeline = CVPipeline()
    streetview_service = StreetViewService()
    game_service = GameService(cv_pipeline)
    
    # Set websocket service in both game service and CV pipeline for real-time updates
    game_service.set_websocket_service(websocket_service)
    cv_pipeline.set_websocket_service(websocket_service)
    
    # Set the services in routes module (including websocket service)
    set_services(game_service, streetview_service, websocket_service)
    
    # Initialize services
    await cv_pipeline.initialize()
    await streetview_service.initialize()
    
    print("🤖 GeoCV AI initialized and ready!")
    print("🌍 Street View service initialized!")
    print("🔌 WebSocket service ready for real-time connections!")
    
    yield
    
    # Shutdown
    if cv_pipeline:
        await cv_pipeline.cleanup()
    if streetview_service:
        await streetview_service.cleanup()
    print("👋 GeoCV AI shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="GeoCV API", 
    description="Computer Vision meets Geography - AI vs Human guessing game with Street View integration and real-time WebSocket updates",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes first
app.include_router(router, prefix="/api/v1")

# Attach WebSocket service to app (this modifies the app object)
# Temporarily disabled due to compatibility issues
# app = websocket_service.attach_to_app(app)

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )