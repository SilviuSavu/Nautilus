/**
 * useWebSocketManager Hook
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Central WebSocket management hook with comprehensive connection handling,
 * message processing, subscription management, and performance monitoring.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface WebSocketMessage {
  id: string;
  type: string;
  timestamp: string;
  data: any;
  direction: 'incoming' | 'outgoing';
  latency?: number;
  error?: string;
  messageId?: string;
  correlationId?: string;
  version?: string;
  priority?: number;
}

interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';
  error: string | null;
  attempts: number;
  connectedAt: string | null;
  lastActivity: string | null;
  sessionId: string | null;
  protocolVersion: string;
  uptime: number;
}

interface PerformanceMetrics {
  messageLatency: number;
  messagesReceived: number;
  messagesSent: number;
  bytesTransferred: number;
  messagesPerSecond: number;
  errorCount: number;
  reconnectionCount: number;
  subscriptionCount: number;
}

interface WebSocketManagerOptions {
  url?: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  enableDebugLogging?: boolean;
}

interface MessageHandler {
  id: string;
  handler: (message: WebSocketMessage) => void;
  filter?: (message: WebSocketMessage) => boolean;
}

export const useWebSocketManager = (options: WebSocketManagerOptions = {}) => {
  const {
    url = `ws://${window.location.hostname}:8001/ws/realtime`,
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000,
    messageQueueSize = 1000,
    enableDebugLogging = process.env.NODE_ENV === 'development'
  } = options;

  // State management
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'disconnected',
    error: null,
    attempts: 0,
    connectedAt: null,
    lastActivity: null,
    sessionId: null,
    protocolVersion: '2.0',
    uptime: 0
  });

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    messageLatency: 0,
    messagesReceived: 0,
    messagesSent: 0,
    bytesTransferred: 0,
    messagesPerSecond: 0,
    errorCount: 0,
    reconnectionCount: 0,
    subscriptionCount: 0
  });

  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);

  // Refs for WebSocket and intervals
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout>();
  const messageHandlersRef = useRef<Map<string, MessageHandler>>(new Map());
  const messageQueueRef = useRef<WebSocketMessage[]>([]);
  const performanceTimerRef = useRef<NodeJS.Timeout>();
  const mountedRef = useRef(true);
  const autoReconnectEnabledRef = useRef(autoReconnect);

  // Logging utility
  const log = useCallback((message: string, level: 'info' | 'warn' | 'error' = 'info') => {
    if (enableDebugLogging) {
      console[level](`[WebSocketManager] ${message}`);
    }
  }, [enableDebugLogging]);

  // Generate unique message ID
  const generateMessageId = useCallback(() => {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Update performance metrics
  const updatePerformanceMetrics = useCallback((updates: Partial<PerformanceMetrics>) => {
    setPerformanceMetrics(prev => ({
      ...prev,
      ...updates,
      messagesPerSecond: prev.messagesReceived > 0 ? 
        prev.messagesReceived / Math.max(connectionState.uptime / 1000, 1) : 0
    }));
  }, [connectionState.uptime]);

  // Process incoming message
  const processMessage = useCallback((event: MessageEvent) => {
    const receiveTime = Date.now();
    
    try {
      const messageData = JSON.parse(event.data);
      const message: WebSocketMessage = {
        id: generateMessageId(),
        type: messageData.type || 'unknown',
        timestamp: messageData.timestamp || new Date().toISOString(),
        data: messageData.data || messageData,
        direction: 'incoming',
        messageId: messageData.message_id,
        correlationId: messageData.correlation_id,
        version: messageData.version || '2.0',
        priority: messageData.priority || 2
      };

      // Calculate latency if timestamp is available
      if (messageData.timestamp) {
        const messageTime = new Date(messageData.timestamp).getTime();
        message.latency = receiveTime - messageTime;
      }

      // Update metrics
      updatePerformanceMetrics({
        messagesReceived: performanceMetrics.messagesReceived + 1,
        messageLatency: message.latency || 0,
        bytesTransferred: performanceMetrics.bytesTransferred + event.data.length
      });

      // Update connection state
      setConnectionState(prev => ({
        ...prev,
        lastActivity: new Date().toISOString()
      }));

      // Add to message history if recording
      if (isRecording) {
        setMessageHistory(prev => {
          const newHistory = [...prev, message];
          return newHistory.slice(-messageQueueSize);
        });
      }

      // Process message handlers
      messageHandlersRef.current.forEach((handler) => {
        try {
          if (!handler.filter || handler.filter(message)) {
            handler.handler(message);
          }
        } catch (error) {
          log(`Error in message handler ${handler.id}: ${error}`, 'error');
          updatePerformanceMetrics({
            errorCount: performanceMetrics.errorCount + 1
          });
        }
      });

      log(`Received message: ${message.type} (${event.data.length} bytes)`);

    } catch (error) {
      log(`Failed to process message: ${error}`, 'error');
      updatePerformanceMetrics({
        errorCount: performanceMetrics.errorCount + 1
      });
    }
  }, [generateMessageId, updatePerformanceMetrics, performanceMetrics, isRecording, messageQueueSize, log]);

  // Send heartbeat
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const heartbeatMessage = {
        type: 'heartbeat',
        timestamp: new Date().toISOString(),
        client_timestamp: Date.now()
      };

      try {
        wsRef.current.send(JSON.stringify(heartbeatMessage));
        log('Heartbeat sent');
      } catch (error) {
        log(`Failed to send heartbeat: ${error}`, 'error');
      }
    }
  }, [log]);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
    }
    
    heartbeatTimeoutRef.current = setInterval(sendHeartbeat, heartbeatInterval);
    log(`Heartbeat started (${heartbeatInterval}ms interval)`);
  }, [sendHeartbeat, heartbeatInterval, log]);

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = undefined;
    }
    log('Heartbeat stopped');
  }, [log]);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (!mountedRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState(prev => ({ ...prev, status: 'connecting' }));
    log(`Connecting to ${url}`);

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;

        const connectedAt = new Date().toISOString();
        const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        setConnectionState(prev => ({
          ...prev,
          status: 'connected',
          error: null,
          attempts: 0,
          connectedAt,
          sessionId,
          lastActivity: connectedAt
        }));

        updatePerformanceMetrics({
          reconnectionCount: connectionState.attempts > 0 ? 
            performanceMetrics.reconnectionCount + 1 : 
            performanceMetrics.reconnectionCount
        });

        startHeartbeat();
        log('WebSocket connected successfully');
      };

      ws.onmessage = processMessage;

      ws.onclose = (event) => {
        if (!mountedRef.current) return;

        stopHeartbeat();
        log(`WebSocket closed: ${event.code} - ${event.reason}`);

        setConnectionState(prev => ({
          ...prev,
          status: 'disconnected',
          error: event.code !== 1000 ? `Connection closed: ${event.reason || event.code}` : null
        }));

        // Auto-reconnect if enabled
        if (autoReconnectEnabledRef.current && event.code !== 1000 && connectionState.attempts < maxReconnectAttempts) {
          setConnectionState(prev => ({
            ...prev,
            status: 'reconnecting',
            attempts: prev.attempts + 1
          }));

          const delay = Math.min(reconnectInterval * Math.pow(2, connectionState.attempts), 30000);
          log(`Reconnecting in ${delay}ms (attempt ${connectionState.attempts + 1}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, delay);
        }
      };

      ws.onerror = (error) => {
        if (!mountedRef.current) return;

        log(`WebSocket error: ${error}`, 'error');
        setConnectionState(prev => ({
          ...prev,
          status: 'error',
          error: 'WebSocket connection error'
        }));

        updatePerformanceMetrics({
          errorCount: performanceMetrics.errorCount + 1
        });
      };

    } catch (error) {
      log(`Failed to create WebSocket: ${error}`, 'error');
      setConnectionState(prev => ({
        ...prev,
        status: 'error',
        error: `Connection failed: ${error}`
      }));
    }
  }, [url, log, processMessage, startHeartbeat, stopHeartbeat, connectionState.attempts, maxReconnectAttempts, reconnectInterval, updatePerformanceMetrics, performanceMetrics]);

  // Disconnect WebSocket
  const disconnect = useCallback(async () => {
    autoReconnectEnabledRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }

    stopHeartbeat();

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setConnectionState(prev => ({
      ...prev,
      status: 'disconnected',
      error: null,
      attempts: 0
    }));

    log('WebSocket disconnected manually');
  }, [stopHeartbeat, log]);

  // Reconnect WebSocket
  const reconnect = useCallback(async () => {
    await disconnect();
    autoReconnectEnabledRef.current = autoReconnect;
    await connect();
  }, [disconnect, connect, autoReconnect]);

  // Send message
  const sendMessage = useCallback((message: any): Promise<boolean> => {
    return new Promise((resolve) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        log('Cannot send message: WebSocket not connected', 'warn');
        resolve(false);
        return;
      }

      try {
        const messageWithMetadata = {
          ...message,
          timestamp: new Date().toISOString(),
          message_id: generateMessageId(),
          version: '2.0'
        };

        const messageString = JSON.stringify(messageWithMetadata);
        wsRef.current.send(messageString);

        // Update metrics
        updatePerformanceMetrics({
          messagesSent: performanceMetrics.messagesSent + 1,
          bytesTransferred: performanceMetrics.bytesTransferred + messageString.length
        });

        // Add to history if recording
        if (isRecording) {
          const outgoingMessage: WebSocketMessage = {
            id: generateMessageId(),
            type: message.type || 'command',
            timestamp: messageWithMetadata.timestamp,
            data: message,
            direction: 'outgoing',
            messageId: messageWithMetadata.message_id
          };

          setMessageHistory(prev => {
            const newHistory = [...prev, outgoingMessage];
            return newHistory.slice(-messageQueueSize);
          });
        }

        log(`Message sent: ${message.type || 'unknown'} (${messageString.length} bytes)`);
        resolve(true);

      } catch (error) {
        log(`Failed to send message: ${error}`, 'error');
        updatePerformanceMetrics({
          errorCount: performanceMetrics.errorCount + 1
        });
        resolve(false);
      }
    });
  }, [generateMessageId, updatePerformanceMetrics, performanceMetrics, isRecording, messageQueueSize, log]);

  // Add message handler
  const addMessageHandler = useCallback((
    id: string,
    handler: (message: WebSocketMessage) => void,
    filter?: (message: WebSocketMessage) => boolean
  ) => {
    messageHandlersRef.current.set(id, { id, handler, filter });
    log(`Message handler added: ${id}`);

    return () => {
      messageHandlersRef.current.delete(id);
      log(`Message handler removed: ${id}`);
    };
  }, [log]);

  // Subscribe to message type
  const subscribe = useCallback(async (messageType: string, filters?: any): Promise<string> => {
    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const subscriptionMessage = {
      type: 'subscribe',
      subscription_id: subscriptionId,
      message_type: messageType,
      filters: filters || {}
    };

    const success = await sendMessage(subscriptionMessage);
    
    if (success) {
      updatePerformanceMetrics({
        subscriptionCount: performanceMetrics.subscriptionCount + 1
      });
      log(`Subscribed to ${messageType} with ID: ${subscriptionId}`);
      return subscriptionId;
    } else {
      throw new Error(`Failed to subscribe to ${messageType}`);
    }
  }, [sendMessage, updatePerformanceMetrics, performanceMetrics, log]);

  // Unsubscribe
  const unsubscribe = useCallback(async (subscriptionId: string): Promise<boolean> => {
    const unsubscribeMessage = {
      type: 'unsubscribe',
      subscription_id: subscriptionId
    };

    const success = await sendMessage(unsubscribeMessage);
    
    if (success) {
      updatePerformanceMetrics({
        subscriptionCount: Math.max(0, performanceMetrics.subscriptionCount - 1)
      });
      log(`Unsubscribed: ${subscriptionId}`);
    }

    return success;
  }, [sendMessage, updatePerformanceMetrics, performanceMetrics, log]);

  // Control functions
  const enableAutoReconnect = useCallback(() => {
    autoReconnectEnabledRef.current = true;
    log('Auto-reconnect enabled');
  }, [log]);

  const disableAutoReconnect = useCallback(() => {
    autoReconnectEnabledRef.current = false;
    log('Auto-reconnect disabled');
  }, [log]);

  const recordMessages = useCallback(() => {
    setIsRecording(true);
    log('Message recording started');
  }, [log]);

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    log('Message recording stopped');
  }, [log]);

  const clearMessageHistory = useCallback(() => {
    setMessageHistory([]);
    log('Message history cleared');
  }, [log]);

  // Get connection info
  const getConnectionInfo = useCallback(() => {
    return {
      ...connectionState,
      uptime: connectionState.connectedAt ? 
        Date.now() - new Date(connectionState.connectedAt).getTime() : 0,
      ...performanceMetrics
    };
  }, [connectionState, performanceMetrics]);

  // Get message stats
  const getMessageStats = useCallback(() => {
    return {
      totalMessages: performanceMetrics.messagesReceived + performanceMetrics.messagesSent,
      errorCount: performanceMetrics.errorCount,
      averageLatency: performanceMetrics.messageLatency,
      messageTypes: messageHistory.reduce((acc, msg) => {
        acc[msg.type] = (acc[msg.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    };
  }, [performanceMetrics, messageHistory]);

  // Update uptime periodically
  useEffect(() => {
    if (connectionState.status === 'connected' && connectionState.connectedAt) {
      const interval = setInterval(() => {
        setConnectionState(prev => ({
          ...prev,
          uptime: Date.now() - new Date(connectionState.connectedAt!).getTime()
        }));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [connectionState.status, connectionState.connectedAt]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatTimeoutRef.current) {
        clearInterval(heartbeatTimeoutRef.current);
      }
      if (performanceTimerRef.current) {
        clearInterval(performanceTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount');
      }
    };
  }, []);

  return {
    // Connection state
    connectionState: connectionState.status,
    connectionError: connectionState.error,
    connectionAttempts: connectionState.attempts,
    isReconnecting: connectionState.status === 'reconnecting',
    
    // Performance metrics
    messageLatency: performanceMetrics.messageLatency,
    messagesReceived: performanceMetrics.messagesReceived,
    messagesSent: performanceMetrics.messagesSent,
    subscriptionCount: performanceMetrics.subscriptionCount,
    
    // Message management
    messageHistory,
    isRecording,
    
    // Connection control
    connect,
    disconnect,
    reconnect,
    
    // Message handling
    sendMessage,
    addMessageHandler,
    subscribe,
    unsubscribe,
    
    // Recording control
    recordMessages,
    stopRecording,
    clearMessageHistory,
    
    // Auto-reconnect control
    enableAutoReconnect,
    disableAutoReconnect,
    autoReconnectEnabled: autoReconnectEnabledRef.current,
    
    // Information getters
    getConnectionInfo,
    getMessageStats
  };
};

export default useWebSocketManager;