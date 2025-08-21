/**
 * useEngineWebSocket - WebSocket hook for real-time engine status updates
 * 
 * Connects to the existing MessageBus WebSocket endpoint and listens for
 * nautilus_engine_status events as specified in Story 6.1.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface EngineWebSocketState {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  connectionError: string | null;
  connectionAttempts: number;
}

interface UseEngineWebSocketOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export const useEngineWebSocket = (options: UseEngineWebSocketOptions = {}) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000
  } = options;

  const [state, setState] = useState<EngineWebSocketState>({
    isConnected: false,
    lastMessage: null,
    connectionError: null,
    connectionAttempts: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout>();
  const mountedRef = useRef(true);

  // Get WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/messagebus`;
  }, []);

  // Send heartbeat to keep connection alive
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      } catch (error) {
        console.error('Failed to send heartbeat:', error);
      }
    }
  }, []);

  // Start heartbeat interval
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
    }
    
    heartbeatTimeoutRef.current = setInterval(sendHeartbeat, heartbeatInterval);
  }, [sendHeartbeat, heartbeatInterval]);

  // Stop heartbeat interval
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = undefined;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const wsUrl = getWebSocketUrl();
      console.log('Connecting to WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // Connection opened
      ws.onopen = () => {
        if (!mountedRef.current) return;
        
        console.log('WebSocket connected for engine monitoring');
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionError: null,
          connectionAttempts: 0
        }));

        // Subscribe to engine status events
        try {
          ws.send(JSON.stringify({
            type: 'subscribe',
            event_types: ['nautilus_engine_status', 'engine_status', 'resource_metrics'],
            timestamp: new Date().toISOString()
          }));
        } catch (error) {
          console.error('Failed to subscribe to engine events:', error);
        }

        startHeartbeat();
      };

      // Message received
      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Filter for engine-related messages
          if (message.type === 'nautilus_engine_status' || 
              message.type === 'engine_status' ||
              message.type === 'resource_metrics' ||
              message.type === 'pong') {
            
            setState(prev => ({
              ...prev,
              lastMessage: message
            }));
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      // Connection error
      ws.onerror = (error) => {
        if (!mountedRef.current) return;
        
        console.error('WebSocket error:', error);
        setState(prev => ({
          ...prev,
          connectionError: 'WebSocket connection error'
        }));
      };

      // Connection closed
      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        
        console.log('WebSocket closed:', event.code, event.reason);
        stopHeartbeat();
        
        setState(prev => ({
          ...prev,
          isConnected: false,
          connectionError: event.code !== 1000 ? `Connection closed: ${event.reason || event.code}` : null
        }));

        // Auto-reconnect if enabled and not manually closed
        if (autoReconnect && event.code !== 1000 && state.connectionAttempts < maxReconnectAttempts) {
          setState(prev => ({
            ...prev,
            connectionAttempts: prev.connectionAttempts + 1
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              console.log(`Attempting to reconnect... (${state.connectionAttempts + 1}/${maxReconnectAttempts})`);
              connect();
            }
          }, reconnectInterval);
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        connectionError: 'Failed to create WebSocket connection'
      }));
    }
  }, [getWebSocketUrl, autoReconnect, reconnectInterval, maxReconnectAttempts, state.connectionAttempts, startHeartbeat, stopHeartbeat]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }

    stopHeartbeat();

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      connectionError: null,
      connectionAttempts: 0
    }));
  }, [stopHeartbeat]);

  // Send message to WebSocket
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          ...message,
          timestamp: new Date().toISOString()
        }));
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    }
    return false;
  }, []);

  // Subscribe to specific engine events
  const subscribeToEngineEvents = useCallback((eventTypes: string[]) => {
    return sendMessage({
      type: 'subscribe',
      event_types: eventTypes
    });
  }, [sendMessage]);

  // Unsubscribe from engine events
  const unsubscribeFromEngineEvents = useCallback((eventTypes: string[]) => {
    return sendMessage({
      type: 'unsubscribe',
      event_types: eventTypes
    });
  }, [sendMessage]);

  // Request engine status update
  const requestEngineStatus = useCallback(() => {
    return sendMessage({
      type: 'request_engine_status'
    });
  }, [sendMessage]);

  // Initialize connection on mount
  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopHeartbeat();
    };
  }, [stopHeartbeat]);

  return {
    // Connection state
    isConnected: state.isConnected,
    connectionError: state.connectionError,
    connectionAttempts: state.connectionAttempts,
    
    // Message data
    lastMessage: state.lastMessage,
    
    // Connection control
    connect,
    disconnect,
    
    // Message sending
    sendMessage,
    subscribeToEngineEvents,
    unsubscribeFromEngineEvents,
    requestEngineStatus,
    
    // Utilities
    sendHeartbeat
  };
};

export default useEngineWebSocket;