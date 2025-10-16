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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GeoCV API",
    description="Real-Time Computer Vision GeoGuesser AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global services
cv_pipeline = None
game_service = None
streetview_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global cv_pipeline, game_service, streetview_service
    cv_pipeline = CVPipeline()
    streetview_service = StreetViewService()
    game_service = GameService(cv_pipeline)
    
    # Set the services in routes module
    set_services(game_service, streetview_service)
    
    # Initialize services
    await cv_pipeline.initialize()
    await streetview_service.initialize()
    
    print("🤖 GeoCV AI initialized and ready!")
    print("🌍 Street View service initialized!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if cv_pipeline:
        await cv_pipeline.cleanup()
    if streetview_service:
        await streetview_service.cleanup()
    print("👋 GeoCV AI shutting down...")

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "GeoCV API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "cv_pipeline": cv_pipeline is not None,
        "game_service": game_service is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
