/**
 * React hook for managing MessageBus WebSocket connection and state
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { webSocketService, WebSocketMessage, PerformanceMetrics } from '../services/websocket';
import { MessageBusState, MessageBusMessage, MessageBusConnectionStatus } from '../types/messagebus';

const MAX_MESSAGES_BUFFER = 100; // Keep last 100 messages

export const useMessageBus = () => {
  const [state, setState] = useState<MessageBusState>({
    connectionStatus: 'disconnected',
    messages: [],
    latestMessage: null,
    connectionInfo: null,
    messagesReceived: 0
  });

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>(() => {
    try {
      return webSocketService.getPerformanceMetrics();
    } catch (error) {
      console.warn('Failed to get initial performance metrics:', error);
      return {
        messageLatency: [],
        averageLatency: 0,
        maxLatency: 0,
        minLatency: Infinity,
        messagesProcessed: 0,
        messagesPerSecond: 0,
        lastUpdateTime: Date.now()
      };
    }
  });

  const [isAutoConnect, setIsAutoConnect] = useState(true);
  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  const connectionInfoInterval = useRef<number | null>(null);

  // Fetch MessageBus connection info from backend API
  const fetchConnectionInfo = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/messagebus/status`);
      if (response.ok) {
        const info: MessageBusConnectionStatus = await response.json();
        setState(prev => ({ ...prev, connectionInfo: info }));
      }
    } catch (error) {
      console.error('Failed to fetch MessageBus connection info:', error);
    }
  }, [apiUrl]);

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    try {
      setState(prev => {
        const newState = { ...prev };

        if (message.type === 'messagebus') {
          const messageBusMessage = message as MessageBusMessage;
          
          // Add to messages buffer (keep last N messages)
          const newMessages = [...prev.messages, messageBusMessage];
          if (newMessages.length > MAX_MESSAGES_BUFFER) {
            newMessages.shift(); // Remove oldest message
          }

          newState.messages = newMessages;
          newState.latestMessage = messageBusMessage;
          newState.messagesReceived = prev.messagesReceived + 1;
        }

        return newState;
      });

      // Update performance metrics safely
      try {
        setPerformanceMetrics(webSocketService.getPerformanceMetrics());
      } catch (metricsError) {
        console.warn('Failed to update performance metrics:', metricsError);
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  }, []);

  // Handle WebSocket connection status changes
  const handleStatusChange = useCallback((status: 'connecting' | 'connected' | 'disconnected' | 'error') => {
    setState(prev => ({ ...prev, connectionStatus: status }));

    // Fetch connection info when connected
    if (status === 'connected') {
      fetchConnectionInfo();
    }
  }, [fetchConnectionInfo]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    try {
      webSocketService.connect();
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }, []);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    try {
      webSocketService.disconnect();
      setIsAutoConnect(false);
    } catch (error) {
      console.error('Failed to disconnect WebSocket:', error);
    }
  }, []);

  // Clear message history
  const clearMessages = useCallback(() => {
    setState(prev => ({
      ...prev,
      messages: [],
      latestMessage: null,
      messagesReceived: 0
    }));
  }, []);

  // Send message to backend
  const sendMessage = useCallback((message: any) => {
    try {
      webSocketService.send(message);
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
    }
  }, []);

  // Setup WebSocket handlers and auto-connect
  useEffect(() => {
    try {
      webSocketService.addMessageHandler(handleMessage);
      webSocketService.addStatusHandler(handleStatusChange);

      // Auto-connect if enabled
      if (isAutoConnect) {
        webSocketService.connect();
      }

      // Periodically fetch connection info
      connectionInfoInterval.current = setInterval(fetchConnectionInfo, 10000); // Every 10 seconds

      return () => {
        try {
          webSocketService.removeMessageHandler(handleMessage);
          webSocketService.removeStatusHandler(handleStatusChange);
        } catch (error) {
          console.warn('Error removing WebSocket handlers:', error);
        }
        
        if (connectionInfoInterval.current) {
          clearInterval(connectionInfoInterval.current);
        }
      };
    } catch (error) {
      console.error('Error setting up WebSocket handlers:', error);
    }
  }, [handleMessage, handleStatusChange, isAutoConnect, fetchConnectionInfo]);

  // Get messages by topic
  const getMessagesByTopic = useCallback((topic: string): MessageBusMessage[] => {
    return state.messages.filter(msg => msg.topic === topic);
  }, [state.messages]);

  // Get latest message by topic
  const getLatestMessageByTopic = useCallback((topic: string): MessageBusMessage | null => {
    const messages = getMessagesByTopic(topic);
    return messages.length > 0 ? messages[messages.length - 1] : null;
  }, [getMessagesByTopic]);

  // Get message statistics
  const getStats = useCallback(() => {
    const topicCounts: Record<string, number> = {};
    state.messages.forEach(msg => {
      topicCounts[msg.topic] = (topicCounts[msg.topic] || 0) + 1;
    });

    return {
      totalMessages: state.messagesReceived,
      bufferedMessages: state.messages.length,
      uniqueTopics: Object.keys(topicCounts).length,
      topicCounts
    };
  }, [state.messages, state.messagesReceived]);

  return {
    // State
    connectionStatus: state.connectionStatus,
    messages: state.messages,
    latestMessage: state.latestMessage,
    connectionInfo: state.connectionInfo,
    messagesReceived: state.messagesReceived,
    isAutoConnect,
    performanceMetrics,

    // Actions
    connect,
    disconnect,
    clearMessages,
    sendMessage,
    setIsAutoConnect,

    // Utilities
    getMessagesByTopic,
    getLatestMessageByTopic,
    getStats
  };
};