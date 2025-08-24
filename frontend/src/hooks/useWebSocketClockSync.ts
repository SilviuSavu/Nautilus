/**
 * WebSocket Clock Synchronization Integration Hook
 * 
 * Integrates frontend clock sync with Phase 2 WebSocket clock manager
 * for real-time trading operations with synchronized timing.
 * 
 * Features:
 * - WebSocket-based real-time clock synchronization
 * - Heartbeat monitoring with clock precision
 * - Trading message timestamps aligned with server clock
 * - Connection health monitoring with clock accuracy
 * - Automatic failover between HTTP and WebSocket sync
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useClockSync } from './useClockSync';
import { useServerTime } from './useServerTime';

export interface WebSocketClockConfig {
  wsUrl?: string;
  heartbeatInterval?: number; // milliseconds
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  enableFallback?: boolean; // Fallback to HTTP sync if WS fails
  priority?: 'critical' | 'high' | 'normal' | 'low';
}

export interface WebSocketClockState {
  isConnected: boolean;
  connectionId: string | null;
  heartbeatStatus: 'active' | 'missed' | 'timeout' | 'recovered';
  lastHeartbeat: number;
  heartbeatLatency: number;
  messageCount: number;
  clockSyncAccuracy: number;
  failoverActive: boolean;
  subscriptionTopics: string[];
}

export interface TradingMessage {
  type: 'order' | 'position' | 'market_data' | 'heartbeat' | 'sync';
  timestamp: number; // Server timestamp
  clientTimestamp: number; // Local timestamp when sent
  data: any;
  connectionId?: string;
  sequence?: number;
}

export interface UseWebSocketClockSyncReturn {
  wsClockState: WebSocketClockState;
  sendMessage: (message: Omit<TradingMessage, 'timestamp' | 'clientTimestamp'>) => void;
  subscribeToTopic: (topic: string) => void;
  unsubscribeFromTopic: (topic: string) => void;
  forceReconnect: () => void;
  getConnectionHealth: () => number; // 0-100%
  isRealTimeEnabled: boolean;
}

const DEFAULT_CONFIG: Required<WebSocketClockConfig> = {
  wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8001/ws',
  heartbeatInterval: 30000, // 30 seconds
  reconnectDelay: 5000, // 5 seconds
  maxReconnectAttempts: 5,
  enableFallback: true,
  priority: 'normal'
};

export const useWebSocketClockSync = (config: WebSocketClockConfig = {}): UseWebSocketClockSyncReturn => {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  const { clockState, getServerTime, isClockSynced } = useClockSync();
  const { serverTimeState } = useServerTime();

  const [wsClockState, setWsClockState] = useState<WebSocketClockState>({
    isConnected: false,
    connectionId: null,
    heartbeatStatus: 'timeout',
    lastHeartbeat: 0,
    heartbeatLatency: 0,
    messageCount: 0,
    clockSyncAccuracy: 0,
    failoverActive: false,
    subscriptionTopics: []
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const messageQueueRef = useRef<TradingMessage[]>([]);

  // Calculate connection health based on multiple factors
  const getConnectionHealth = useCallback((): number => {
    if (!wsClockState.isConnected) {
      return 0;
    }

    const timeSinceHeartbeat = Date.now() - wsClockState.lastHeartbeat;
    const heartbeatHealth = Math.max(0, 100 - (timeSinceHeartbeat / finalConfig.heartbeatInterval) * 100);
    
    const latencyHealth = Math.max(0, 100 - (wsClockState.heartbeatLatency / 1000) * 100);
    const clockHealth = wsClockState.clockSyncAccuracy;
    
    return Math.min(100, (heartbeatHealth * 0.4 + latencyHealth * 0.3 + clockHealth * 0.3));
  }, [wsClockState, finalConfig.heartbeatInterval]);

  // Send message with precise timestamps
  const sendMessage = useCallback((message: Omit<TradingMessage, 'timestamp' | 'clientTimestamp'>) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // Queue message if not connected
      messageQueueRef.current.push({
        ...message,
        timestamp: 0, // Will be set when sending
        clientTimestamp: Date.now()
      });
      return;
    }

    const timestampedMessage: TradingMessage = {
      ...message,
      timestamp: isClockSynced ? getServerTime() : Date.now(),
      clientTimestamp: Date.now()
    };

    try {
      wsRef.current.send(JSON.stringify(timestampedMessage));
      setWsClockState(prev => ({
        ...prev,
        messageCount: prev.messageCount + 1
      }));
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
      // Add to queue for retry
      messageQueueRef.current.push(timestampedMessage);
    }
  }, [isClockSynced, getServerTime]);

  // Subscribe to topic with WebSocket manager
  const subscribeToTopic = useCallback((topic: string) => {
    sendMessage({
      type: 'sync',
      data: {
        action: 'subscribe',
        topic: topic,
        priority: finalConfig.priority
      }
    });

    setWsClockState(prev => ({
      ...prev,
      subscriptionTopics: [...prev.subscriptionTopics.filter(t => t !== topic), topic]
    }));
  }, [sendMessage, finalConfig.priority]);

  // Unsubscribe from topic
  const unsubscribeFromTopic = useCallback((topic: string) => {
    sendMessage({
      type: 'sync',
      data: {
        action: 'unsubscribe',
        topic: topic
      }
    });

    setWsClockState(prev => ({
      ...prev,
      subscriptionTopics: prev.subscriptionTopics.filter(t => t !== topic)
    }));
  }, [sendMessage]);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data);
      const receiveTime = Date.now();

      // Handle different message types
      switch (message.type) {
        case 'pong':
          // Calculate heartbeat latency
          const latency = receiveTime - message.timestamp;
          setWsClockState(prev => ({
            ...prev,
            heartbeatStatus: 'active',
            lastHeartbeat: receiveTime,
            heartbeatLatency: latency,
            clockSyncAccuracy: isClockSynced ? 100 : 50
          }));
          break;

        case 'connection_established':
          setWsClockState(prev => ({
            ...prev,
            connectionId: message.connection_id,
            isConnected: true,
            heartbeatStatus: 'active'
          }));
          
          // Send queued messages
          const queuedMessages = [...messageQueueRef.current];
          messageQueueRef.current = [];
          queuedMessages.forEach(queuedMsg => sendMessage(queuedMsg));
          break;

        case 'heartbeat_timeout':
          setWsClockState(prev => ({
            ...prev,
            heartbeatStatus: 'timeout'
          }));
          break;

        case 'heartbeat_recovered':
          setWsClockState(prev => ({
            ...prev,
            heartbeatStatus: 'recovered'
          }));
          break;

        case 'sync_response':
          // Handle clock synchronization response
          if (message.data && message.data.server_time) {
            const serverTime = message.data.server_time;
            const clockDrift = Math.abs(serverTime - receiveTime);
            setWsClockState(prev => ({
              ...prev,
              clockSyncAccuracy: Math.max(0, 100 - clockDrift / 100) // Penalize high drift
            }));
          }
          break;

        default:
          // Handle trading messages with timestamp validation
          if (message.timestamp && isClockSynced) {
            const serverTime = getServerTime();
            const timeDiff = Math.abs(message.timestamp - serverTime);
            
            // Log if message timestamp is significantly off
            if (timeDiff > 5000) { // 5 second threshold
              console.warn(`Message timestamp drift: ${timeDiff}ms for ${message.type}`);
            }
          }
          
          // Dispatch custom event for message handling
          window.dispatchEvent(new CustomEvent('websocketTradingMessage', {
            detail: { message, receiveTime, clockSynced: isClockSynced }
          }));
          break;
      }

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [isClockSynced, getServerTime, sendMessage]);

  // WebSocket connection management
  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(finalConfig.wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected for clock sync');
        reconnectAttemptsRef.current = 0;
        
        // Register connection with heartbeat parameters
        const registrationMessage = {
          type: 'sync',
          data: {
            action: 'register',
            heartbeat_interval_ms: finalConfig.heartbeatInterval,
            priority: finalConfig.priority,
            enable_clock_sync: true
          }
        };
        
        ws.send(JSON.stringify(registrationMessage));
      };

      ws.onmessage = handleWebSocketMessage;

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        
        setWsClockState(prev => ({
          ...prev,
          isConnected: false,
          connectionId: null,
          heartbeatStatus: 'timeout'
        }));

        // Attempt reconnection if enabled and under max attempts
        if (reconnectAttemptsRef.current < finalConfig.maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`WebSocket reconnection attempt ${reconnectAttemptsRef.current}`);
            connect();
          }, finalConfig.reconnectDelay * Math.pow(1.5, reconnectAttemptsRef.current - 1));
        } else if (finalConfig.enableFallback) {
          // Enable HTTP fallback
          setWsClockState(prev => ({
            ...prev,
            failoverActive: true
          }));
          console.log('WebSocket max reconnection attempts reached, using HTTP fallback');
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  }, [finalConfig, handleWebSocketMessage]);

  // Force reconnection
  const forceReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    reconnectAttemptsRef.current = 0;
    setWsClockState(prev => ({
      ...prev,
      failoverActive: false
    }));
    
    connect();
  }, [connect]);

  // Initialize WebSocket connection
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatTimeoutRef.current) {
        clearTimeout(heartbeatTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Periodic health check and sync request
  useEffect(() => {
    const healthInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send periodic sync request for clock accuracy
        sendMessage({
          type: 'sync',
          data: {
            action: 'time_sync',
            client_time: Date.now(),
            server_time_offset: clockState.serverTimeOffset
          }
        });
      }
    }, 60000); // Every minute

    return () => clearInterval(healthInterval);
  }, [sendMessage, clockState.serverTimeOffset]);

  // Monitor clock synchronization accuracy
  useEffect(() => {
    const accuracy = isClockSynced ? Math.max(80, 100 - Math.abs(clockState.clockDrift) * 10) : 0;
    setWsClockState(prev => ({
      ...prev,
      clockSyncAccuracy: accuracy
    }));
  }, [isClockSynced, clockState.clockDrift]);

  const isRealTimeEnabled = wsClockState.isConnected && !wsClockState.failoverActive;

  return {
    wsClockState,
    sendMessage,
    subscribeToTopic,
    unsubscribeFromTopic,
    forceReconnect,
    getConnectionHealth,
    isRealTimeEnabled
  };
};

export default useWebSocketClockSync;