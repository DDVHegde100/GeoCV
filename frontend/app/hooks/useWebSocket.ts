'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

interface ConnectionStatus {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  sessionId: string | null;
}

interface GameUpdate {
  type: 'analysis_progress' | 'state_update' | 'game_completed' | 'error';
  data: any;
  timestamp: string;
}

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnection?: boolean;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
}

export function useWebSocket(gameId?: string, options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnection = true,
    reconnectionAttempts = 5,
    reconnectionDelay = 1000
  } = options;

  const socketRef = useRef<Socket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    connected: false,
    connecting: false,
    error: null,
    sessionId: null
  });
  
  const [lastUpdate, setLastUpdate] = useState<GameUpdate | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<{
    step: string;
    progress: number;
    step_number: number;
    total_steps: number;
  } | null>(null);

  // Connection handlers
  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;

    setConnectionStatus(prev => ({ ...prev, connecting: true, error: null }));

    const socket = io(process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000', {
      transports: ['websocket', 'polling'],
      reconnection,
      reconnectionAttempts,
      reconnectionDelay
    });

    // Connection events
    socket.on('connect', () => {
      console.log('🔌 Connected to WebSocket server');
      setConnectionStatus({
        connected: true,
        connecting: false,
        error: null,
        sessionId: socket.id || null
      });
    });

    socket.on('disconnect', (reason) => {
      console.log('❌ Disconnected from WebSocket server:', reason);
      setConnectionStatus(prev => ({
        ...prev,
        connected: false,
        connecting: false,
        error: reason === 'transport error' ? 'Connection lost' : null
      }));
    });

    socket.on('connect_error', (error) => {
      console.error('🚫 WebSocket connection error:', error);
      setConnectionStatus(prev => ({
        ...prev,
        connected: false,
        connecting: false,
        error: error.message || 'Connection failed'
      }));
    });

    // Connection confirmation
    socket.on('connection_established', (data) => {
      console.log('✅ Connection established:', data);
      setConnectionStatus(prev => ({
        ...prev,
        sessionId: data.session_id
      }));
    });

    // Game-specific events
    socket.on('ai_analysis_update', (data) => {
      console.log('🧠 AI analysis update:', data);
      setLastUpdate({
        type: 'analysis_progress',
        data: data.data,
        timestamp: data.timestamp
      });
      setAnalysisProgress(data.data);
    });

    socket.on('game_state_update', (data) => {
      console.log('🎮 Game state update:', data);
      setLastUpdate({
        type: 'state_update',
        data: data.state,
        timestamp: data.timestamp
      });
    });

    socket.on('game_completed', (data) => {
      console.log('🏁 Game completed:', data);
      setLastUpdate({
        type: 'game_completed',
        data: data.result,
        timestamp: data.timestamp
      });
    });

    socket.on('game_error', (data) => {
      console.error('❌ Game error:', data);
      setLastUpdate({
        type: 'error',
        data: { error: data.error },
        timestamp: data.timestamp
      });
    });

    // Server messages
    socket.on('server_message', (data) => {
      console.log('📢 Server message:', data);
    });

    socketRef.current = socket;
  }, [reconnection, reconnectionAttempts, reconnectionDelay]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setConnectionStatus({
      connected: false,
      connecting: false,
      error: null,
      sessionId: null
    });
  }, []);

  const joinGame = useCallback((gameId: string) => {
    if (socketRef.current?.connected) {
      console.log(`🎯 Joining game: ${gameId}`);
      socketRef.current.emit('join_game', { game_id: gameId });
    }
  }, []);

  const leaveGame = useCallback((gameId: string) => {
    if (socketRef.current?.connected) {
      console.log(`👋 Leaving game: ${gameId}`);
      socketRef.current.emit('leave_game', { game_id: gameId });
    }
  }, []);

  const requestGameUpdate = useCallback((gameId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('request_game_update', { game_id: gameId });
    }
  }, []);

  // Auto connect/disconnect
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [autoConnect, connect]);

  // Auto join/leave game
  useEffect(() => {
    if (gameId && connectionStatus.connected) {
      joinGame(gameId);
      
      return () => {
        leaveGame(gameId);
      };
    }
  }, [gameId, connectionStatus.connected, joinGame, leaveGame]);

  // Retry connection on errors
  useEffect(() => {
    let retryTimeout: NodeJS.Timeout;
    
    if (connectionStatus.error && !connectionStatus.connected && autoConnect) {
      retryTimeout = setTimeout(() => {
        console.log('🔄 Retrying WebSocket connection...');
        connect();
      }, reconnectionDelay);
    }

    return () => {
      if (retryTimeout) {
        clearTimeout(retryTimeout);
      }
    };
  }, [connectionStatus.error, connectionStatus.connected, autoConnect, connect, reconnectionDelay]);

  return {
    // Connection state
    connectionStatus,
    
    // Updates
    lastUpdate,
    analysisProgress,
    
    // Actions
    connect,
    disconnect,
    joinGame,
    leaveGame,
    requestGameUpdate,
    
    // Socket instance (for custom events)
    socket: socketRef.current
  };
}
