"""
WebSocket Service for Real-Time Communication
Handles Socket.IO connections and real-time game updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import socketio
from fastapi import FastAPI

logger = logging.getLogger(__name__)

class WebSocketService:
    """Real-time WebSocket service using Socket.IO"""
    
    def __init__(self):
        # Create Socket.IO server
        self.sio = socketio.AsyncServer(
            cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            logger=logger,
            engineio_logger=logger
        )
        
        # Track active connections
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.game_rooms: Dict[str, List[str]] = {}  # game_id -> [session_ids]
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection"""
            logger.info(f"Client {sid} connected")
            self.active_connections[sid] = {
                'connected_at': datetime.now(),
                'game_id': None,
                'user_type': 'player'
            }
            
            # Send connection confirmation
            await self.sio.emit('connection_established', {
                'status': 'connected',
                'session_id': sid,
                'timestamp': datetime.now().isoformat()
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            if sid in self.active_connections:
                game_id = self.active_connections[sid].get('game_id')
                if game_id and game_id in self.game_rooms:
                    # Remove from game room
                    if sid in self.game_rooms[game_id]:
                        self.game_rooms[game_id].remove(sid)
                    
                    # Notify other players in the game
                    await self.sio.emit('player_disconnected', {
                        'message': 'Another player disconnected',
                        'timestamp': datetime.now().isoformat()
                    }, room=game_id)
                
                del self.active_connections[sid]
            
            logger.info(f"Client {sid} disconnected")
        
        @self.sio.event
        async def join_game(sid, data):
            """Handle player joining a game"""
            try:
                game_id = data.get('game_id')
                if not game_id:
                    await self.sio.emit('error', {
                        'message': 'game_id is required'
                    }, room=sid)
                    return
                
                # Join the game room
                await self.sio.enter_room(sid, game_id)
                
                # Update connection info
                if sid in self.active_connections:
                    self.active_connections[sid]['game_id'] = game_id
                
                # Track game room
                if game_id not in self.game_rooms:
                    self.game_rooms[game_id] = []
                if sid not in self.game_rooms[game_id]:
                    self.game_rooms[game_id].append(sid)
                
                logger.info(f"Client {sid} joined game {game_id}")
                
                # Confirm join
                await self.sio.emit('game_joined', {
                    'game_id': game_id,
                    'status': 'joined',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
                # Notify other players
                await self.sio.emit('player_joined', {
                    'message': 'New player joined the game',
                    'game_id': game_id,
                    'timestamp': datetime.now().isoformat()
                }, room=game_id, skip_sid=sid)
                
            except Exception as e:
                logger.error(f"Error joining game: {e}")
                await self.sio.emit('error', {
                    'message': f'Failed to join game: {str(e)}'
                }, room=sid)
        
        @self.sio.event
        async def leave_game(sid, data):
            """Handle player leaving a game"""
            try:
                game_id = data.get('game_id')
                if game_id:
                    await self.sio.leave_room(sid, game_id)
                    
                    # Update connection info
                    if sid in self.active_connections:
                        self.active_connections[sid]['game_id'] = None
                    
                    # Remove from game room tracking
                    if game_id in self.game_rooms and sid in self.game_rooms[game_id]:
                        self.game_rooms[game_id].remove(sid)
                    
                    logger.info(f"Client {sid} left game {game_id}")
                    
                    # Notify other players
                    await self.sio.emit('player_left', {
                        'message': 'Player left the game',
                        'game_id': game_id,
                        'timestamp': datetime.now().isoformat()
                    }, room=game_id)
                
            except Exception as e:
                logger.error(f"Error leaving game: {e}")
        
        @self.sio.event
        async def request_game_update(sid, data):
            """Handle request for game state update"""
            try:
                game_id = data.get('game_id')
                if not game_id:
                    return
                
                # This will be called by the game service to get current state
                await self.sio.emit('game_update_requested', {
                    'game_id': game_id,
                    'requesting_sid': sid,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error requesting game update: {e}")
    
    async def broadcast_analysis_update(self, game_id: str, update_data: Dict[str, Any]):
        """Broadcast AI analysis update to all players in a game"""
        try:
            if game_id in self.game_rooms and self.game_rooms[game_id]:
                await self.sio.emit('ai_analysis_update', {
                    'game_id': game_id,
                    'update_type': 'analysis_progress',
                    'data': update_data,
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
                
                logger.debug(f"Broadcasted analysis update for game {game_id}")
        
        except Exception as e:
            logger.error(f"Error broadcasting analysis update: {e}")
    
    async def broadcast_game_state(self, game_id: str, game_state: Dict[str, Any]):
        """Broadcast complete game state update"""
        try:
            if game_id in self.game_rooms and self.game_rooms[game_id]:
                await self.sio.emit('game_state_update', {
                    'game_id': game_id,
                    'state': game_state,
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
                
                logger.debug(f"Broadcasted game state for game {game_id}")
        
        except Exception as e:
            logger.error(f"Error broadcasting game state: {e}")
    
    async def broadcast_game_complete(self, game_id: str, result_data: Dict[str, Any]):
        """Broadcast game completion and results"""
        try:
            if game_id in self.game_rooms and self.game_rooms[game_id]:
                await self.sio.emit('game_completed', {
                    'game_id': game_id,
                    'result': result_data,
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
                
                # Clean up game room after a delay
                asyncio.create_task(self._cleanup_game_room(game_id, delay=30))
                
                logger.info(f"Broadcasted game completion for game {game_id}")
        
        except Exception as e:
            logger.error(f"Error broadcasting game completion: {e}")
    
    async def send_error_to_game(self, game_id: str, error_message: str):
        """Send error message to all players in a game"""
        try:
            if game_id in self.game_rooms and self.game_rooms[game_id]:
                await self.sio.emit('game_error', {
                    'game_id': game_id,
                    'error': error_message,
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
        
        except Exception as e:
            logger.error(f"Error sending game error: {e}")
    
    async def _cleanup_game_room(self, game_id: str, delay: int = 0):
        """Clean up game room after specified delay"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            if game_id in self.game_rooms:
                # Remove all clients from the game room
                for sid in self.game_rooms[game_id].copy():
                    await self.sio.leave_room(sid, game_id)
                    if sid in self.active_connections:
                        self.active_connections[sid]['game_id'] = None
                
                # Delete room tracking
                del self.game_rooms[game_id]
                logger.info(f"Cleaned up game room {game_id}")
        
        except Exception as e:
            logger.error(f"Error cleaning up game room {game_id}: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'active_games': len(self.game_rooms),
            'connections_per_game': {
                game_id: len(sids) for game_id, sids in self.game_rooms.items()
            }
        }
    
    async def broadcast_server_message(self, message: str, message_type: str = 'info'):
        """Broadcast server-wide message to all connected clients"""
        await self.sio.emit('server_message', {
            'message': message,
            'type': message_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def attach_to_app(self, app: FastAPI):
        """Attach Socket.IO to FastAPI application"""
        # Create ASGI app for Socket.IO
        socket_app = socketio.ASGIApp(self.sio, app)
        return socket_app
    
    async def broadcast_tesla_overlays(self, game_id: str, overlays_data: Dict[str, Any]):
        """Broadcast Tesla-style visual overlays to game participants"""
        try:
            if game_id in self.game_rooms:
                await self.sio.emit('tesla_overlays_update', {
                    'game_id': game_id,
                    'overlays': overlays_data.get('overlays', {}),
                    'detected_objects': overlays_data.get('detected_objects', []),
                    'confidence_scores': overlays_data.get('confidence_scores', {}),
                    'bounding_boxes': overlays_data.get('bounding_boxes', []),
                    'color_coding': {
                        'vehicles': '#4ECDC4',
                        'buildings': '#FF6B6B', 
                        'text': '#FF9F43',
                        'infrastructure': '#A8E6CF',
                        'vegetation': '#95E1D3'
                    },
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
                
                logger.info(f"Broadcast Tesla overlays to game {game_id}")
        
        except Exception as e:
            logger.error(f"Failed to broadcast Tesla overlays for game {game_id}: {e}")
    
    async def broadcast_ai_vision_update(self, game_id: str, vision_data: Dict[str, Any]):
        """Broadcast AI vision analysis in real-time like Tesla autopilot"""
        try:
            if game_id in self.game_rooms:
                await self.sio.emit('ai_vision_update', {
                    'game_id': game_id,
                    'vision_status': vision_data.get('status', 'analyzing'),
                    'current_step': vision_data.get('step', ''),
                    'progress': vision_data.get('progress', 0),
                    'detected_objects_count': len(vision_data.get('detected_objects', [])),
                    'active_overlays': vision_data.get('active_overlays', []),
                    'confidence_level': vision_data.get('confidence_level', 0),
                    'processing_fps': vision_data.get('processing_fps', 30),
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
        
        except Exception as e:
            logger.error(f"Failed to broadcast AI vision update for game {game_id}: {e}")
    
    async def broadcast_enhanced_analysis_update(self, game_id: str, analysis_data: Dict[str, Any]):
        """Enhanced analysis updates with more detailed information"""
        try:
            if game_id in self.game_rooms:
                # Prepare enhanced update data
                update_data = {
                    'game_id': game_id,
                    'analysis_type': analysis_data.get('analysis_type', 'standard'),
                    'step': analysis_data.get('step', ''),
                    'progress': analysis_data.get('progress', 0),
                    'step_number': analysis_data.get('step_number', 1),
                    'total_steps': analysis_data.get('total_steps', 10),
                    'processing_details': {
                        'neural_networks_active': analysis_data.get('analysis_type') == 'enhanced_tesla_style',
                        'multi_directional_analysis': True,
                        'object_detection_enabled': True,
                        'text_recognition_enabled': True,
                        'geographic_correlation_enabled': True
                    },
                    'real_time_stats': {
                        'objects_detected_so_far': analysis_data.get('objects_detected', 0),
                        'confidence_building': analysis_data.get('confidence', 0),
                        'regions_analyzed': analysis_data.get('regions_analyzed', 0)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.sio.emit('enhanced_analysis_update', update_data, room=game_id)
                
                # Also send the regular analysis update for compatibility
                await self.broadcast_analysis_update(game_id, analysis_data)
        
        except Exception as e:
            logger.error(f"Failed to broadcast enhanced analysis update for game {game_id}: {e}")
    
    async def broadcast_competition_status(self, game_id: str, status_data: Dict[str, Any]):
        """Broadcast AI vs Human competition status"""
        try:
            if game_id in self.game_rooms:
                await self.sio.emit('competition_status_update', {
                    'game_id': game_id,
                    'competition_type': 'ai_vs_human',
                    'ai_status': status_data.get('ai_status', 'analyzing'),
                    'human_status': status_data.get('human_status', 'waiting'),
                    'ai_progress': status_data.get('ai_progress', 0),
                    'ai_confidence': status_data.get('ai_confidence', 0),
                    'game_phase': status_data.get('game_phase', 'analysis'),
                    'location_fairness': {
                        'random_location': True,
                        'no_metadata_cheating': True,
                        'same_view_for_both': True
                    },
                    'timestamp': datetime.now().isoformat()
                }, room=game_id)
        
        except Exception as e:
            logger.error(f"Failed to broadcast competition status for game {game_id}: {e}")

# Global WebSocket service instance
websocket_service = WebSocketService()
