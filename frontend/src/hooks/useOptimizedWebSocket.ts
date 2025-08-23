/**
 * Optimized WebSocket Hook
 * High-performance WebSocket management with connection pooling and message optimization
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { UI_CONSTANTS } from '../constants/ui';

export interface WebSocketMessage {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  priority?: 'high' | 'normal' | 'low';
}

export interface WebSocketConfig {
  url: string;
  protocols?: string | string[];
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  messageBufferSize?: number;
  enableCompression?: boolean;
  enableBatching?: boolean;
  batchSize?: number;
  batchTimeout?: number;
}

export interface WebSocketStats {
  messagesReceived: number;
  messagesSent: number;
  connectionTime: number;
  lastMessageTime: number;
  averageLatency: number;
  errorCount: number;
  reconnectCount: number;
}

export interface UseOptimizedWebSocketReturn {
  // Connection state
  isConnected: boolean;
  connectionState: 'connecting' | 'connected' | 'disconnecting' | 'disconnected' | 'error';
  error: string | null;
  
  // Message handling
  messages: WebSocketMessage[];
  sendMessage: (type: string, data: any, priority?: 'high' | 'normal' | 'low') => void;
  subscribe: (messageType: string, callback: (message: WebSocketMessage) => void) => () => void;
  
  // Connection management
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  
  // Statistics
  stats: WebSocketStats;
  clearStats: () => void;
  
  // Configuration
  updateConfig: (newConfig: Partial<WebSocketConfig>) => void;
}

export const useOptimizedWebSocket = (
  initialConfig: WebSocketConfig
): UseOptimizedWebSocketReturn => {
  const [config, setConfig] = useState<WebSocketConfig>(initialConfig);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnecting' | 'disconnected' | 'error'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [stats, setStats] = useState<WebSocketStats>({
    messagesReceived: 0,
    messagesSent: 0,
    connectionTime: 0,
    lastMessageTime: 0,
    averageLatency: 0,
    errorCount: 0,
    reconnectCount: 0
  });

  // Refs for managing WebSocket and timers
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const messageBufferRef = useRef<WebSocketMessage[]>([]);
  const batchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const subscribersRef = useRef<Map<string, Set<(message: WebSocketMessage) => void>>>(new Map());
  const latencyMapRef = useRef<Map<string, number>>(new Map());
  const reconnectAttemptsRef = useRef(0);

  // Message queue for priority handling
  const messageQueueRef = useRef<{
    high: WebSocketMessage[];
    normal: WebSocketMessage[];
    low: WebSocketMessage[];
  }>({
    high: [],
    normal: [],
    low: []
  });

  // Update configuration
  const updateConfig = useCallback((newConfig: Partial<WebSocketConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);

  // Clear statistics
  const clearStats = useCallback(() => {
    setStats({
      messagesReceived: 0,
      messagesSent: 0,
      connectionTime: 0,
      lastMessageTime: 0,
      averageLatency: 0,
      errorCount: 0,
      reconnectCount: 0
    });
    latencyMapRef.current.clear();
  }, []);

  // Process message queue based on priority
  const processMessageQueue = useCallback(() => {
    const queue = messageQueueRef.current;
    const messagesToSend: WebSocketMessage[] = [
      ...queue.high,
      ...queue.normal,
      ...queue.low
    ];

    if (messagesToSend.length === 0 || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const batchSize = config.batchSize || 10;
    const toSend = messagesToSend.slice(0, batchSize);
    
    if (config.enableBatching && toSend.length > 1) {
      // Send as batch
      const batchMessage = {
        type: 'batch',
        messages: toSend,
        timestamp: Date.now()
      };
      wsRef.current.send(JSON.stringify(batchMessage));
    } else {
      // Send individual messages
      toSend.forEach(message => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify(message));
        }
      });
    }

    // Update stats
    setStats(prev => ({
      ...prev,
      messagesSent: prev.messagesSent + toSend.length
    }));

    // Remove sent messages from queue
    queue.high = queue.high.filter(msg => !toSend.includes(msg));
    queue.normal = queue.normal.filter(msg => !toSend.includes(msg));
    queue.low = queue.low.filter(msg => !toSend.includes(msg));
  }, [config.enableBatching, config.batchSize]);

  // Send message with priority
  const sendMessage = useCallback((type: string, data: any, priority: 'high' | 'normal' | 'low' = 'normal') => {
    const message: WebSocketMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      data,
      timestamp: Date.now(),
      priority
    };

    // Store for latency calculation
    latencyMapRef.current.set(message.id, message.timestamp);

    // Add to appropriate queue
    messageQueueRef.current[priority].push(message);

    // Process immediately for high priority or if not batching
    if (priority === 'high' || !config.enableBatching) {
      processMessageQueue();
    }
  }, [processMessageQueue, config.enableBatching]);

  // Subscribe to specific message types
  const subscribe = useCallback((messageType: string, callback: (message: WebSocketMessage) => void) => {
    if (!subscribersRef.current.has(messageType)) {
      subscribersRef.current.set(messageType, new Set());
    }
    subscribersRef.current.get(messageType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const subscribers = subscribersRef.current.get(messageType);
      if (subscribers) {
        subscribers.delete(callback);
        if (subscribers.size === 0) {
          subscribersRef.current.delete(messageType);
        }
      }
    };
  }, []);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      const message: WebSocketMessage = {
        id: data.id || `received-${Date.now()}`,
        type: data.type,
        data: data.data || data,
        timestamp: Date.now(),
        priority: data.priority || 'normal'
      };

      // Calculate latency if this is a response
      if (data.responseId && latencyMapRef.current.has(data.responseId)) {
        const sendTime = latencyMapRef.current.get(data.responseId)!;
        const latency = message.timestamp - sendTime;
        
        setStats(prev => {
          const totalLatency = prev.averageLatency * prev.messagesReceived + latency;
          return {
            ...prev,
            averageLatency: totalLatency / (prev.messagesReceived + 1)
          };
        });
        
        latencyMapRef.current.delete(data.responseId);
      }

      // Add to message buffer
      messageBufferRef.current.push(message);
      
      // Limit buffer size
      const bufferSize = config.messageBufferSize || UI_CONSTANTS.DATA_LIMITS.MAX_HISTORY_POINTS;
      if (messageBufferRef.current.length > bufferSize) {
        messageBufferRef.current = messageBufferRef.current.slice(-bufferSize);
      }

      // Update messages state
      setMessages(prev => [...prev.slice(-(bufferSize - 1)), message]);

      // Update stats
      setStats(prev => ({
        ...prev,
        messagesReceived: prev.messagesReceived + 1,
        lastMessageTime: message.timestamp
      }));

      // Notify subscribers
      const subscribers = subscribersRef.current.get(message.type);
      if (subscribers) {
        subscribers.forEach(callback => callback(message));
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      setStats(prev => ({ ...prev, errorCount: prev.errorCount + 1 }));
    }
  }, [config.messageBufferSize]);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    const interval = config.heartbeatInterval || 30000;
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        sendMessage('heartbeat', { timestamp: Date.now() });
      }
    }, interval);
  }, [config.heartbeatInterval, sendMessage]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState('connecting');
    setError(null);

    try {
      const ws = new WebSocket(config.url, config.protocols);
      wsRef.current = ws;

      const connectionStartTime = Date.now();

      ws.onopen = () => {
        setIsConnected(true);
        setConnectionState('connected');
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        setStats(prev => ({
          ...prev,
          connectionTime: Date.now() - connectionStartTime
        }));

        startHeartbeat();
      };

      ws.onmessage = handleMessage;

      ws.onclose = (event) => {
        setIsConnected(false);
        setConnectionState('disconnected');
        
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        if (!event.wasClean && reconnectAttemptsRef.current < (config.maxReconnectAttempts || 5)) {
          const interval = config.reconnectInterval || 3000;
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            setStats(prev => ({ ...prev, reconnectCount: prev.reconnectCount + 1 }));
            connect();
          }, interval);
        }
      };

      ws.onerror = () => {
        setConnectionState('error');
        setError('WebSocket connection error');
        setStats(prev => ({ ...prev, errorCount: prev.errorCount + 1 }));
      };

    } catch (error) {
      setConnectionState('error');
      setError(`Failed to create WebSocket connection: ${error}`);
    }
  }, [config, handleMessage, startHeartbeat]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    setConnectionState('disconnecting');
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    if (batchTimeoutRef.current) {
      clearTimeout(batchTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionState('disconnected');
  }, []);

  // Reconnect
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(connect, 1000);
  }, [disconnect, connect]);

  // Set up batch processing
  useEffect(() => {
    if (config.enableBatching && config.batchTimeout) {
      batchTimeoutRef.current = setInterval(() => {
        processMessageQueue();
      }, config.batchTimeout);

      return () => {
        if (batchTimeoutRef.current) {
          clearInterval(batchTimeoutRef.current);
        }
      };
    }
  }, [config.enableBatching, config.batchTimeout, processMessageQueue]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    connectionState,
    error,
    messages,
    sendMessage,
    subscribe,
    connect,
    disconnect,
    reconnect,
    stats,
    clearStats,
    updateConfig
  };
};