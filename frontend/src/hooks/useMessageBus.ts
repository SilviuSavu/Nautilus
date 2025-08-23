/**
 * React Hook for Direct Message Bus Integration
 * Provides high-performance real-time data without HTTP proxy overhead
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { messageBusService } from '../services/MessageBusService';

interface UseMessageBusOptions {
  autoConnect?: boolean;
  reconnectOnMount?: boolean;
}

interface MessageBusState {
  connected: boolean;
  subscriptions: number;
  reconnectAttempts: number;
  lastMessage?: any;
  error?: string;
  connectionStatus?: string;
  latestMessage?: any;
  messages?: any[];
  connectionInfo?: any;
  messagesReceived?: number;
}

export function useMessageBus(options: UseMessageBusOptions = {}) {
  const { autoConnect = true, reconnectOnMount = true } = options;
  const [state, setState] = useState<MessageBusState>({
    connected: false,
    subscriptions: 0,
    reconnectAttempts: 0,
    connectionStatus: 'disconnected',
    latestMessage: null,
    messages: [],
    connectionInfo: null,
    messagesReceived: 0
  });
  
  const subscriptionsRef = useRef<Set<string>>(new Set());

  // Update state from service
  const updateState = useCallback(() => {
    const status = messageBusService.getConnectionStatus();
    setState(prev => ({
      ...prev,
      connected: status.connected,
      subscriptions: status.subscriptions,
      reconnectAttempts: status.reconnectAttempts,
      connectionStatus: status.connected ? 'connected' : 'disconnected'
    }));
  }, []);

  // Connect to message bus
  const connect = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: undefined }));
      const success = await messageBusService.connect();
      if (!success) {
        setState(prev => ({ ...prev, error: 'Failed to connect to message bus' }));
      }
      updateState();
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Unknown connection error' 
      }));
    }
  }, [updateState]);

  // Disconnect from message bus
  const disconnect = useCallback(() => {
    // Clean up all subscriptions from this hook
    subscriptionsRef.current.forEach(subId => {
      messageBusService.unsubscribe(subId);
    });
    subscriptionsRef.current.clear();
    
    messageBusService.disconnect();
    updateState();
  }, [updateState]);

  // Subscribe to high-speed data stream
  const subscribe = useCallback((topic: string, callback: (message: any) => void) => {
    const subscriptionId = messageBusService.subscribe(topic, callback);
    subscriptionsRef.current.add(subscriptionId);
    updateState();
    
    return () => {
      messageBusService.unsubscribe(subscriptionId);
      subscriptionsRef.current.delete(subscriptionId);
      updateState();
    };
  }, [updateState]);

  // Send command via message bus (bypasses HTTP)
  const sendCommand = useCallback(async (command: string, data: any) => {
    try {
      setState(prev => ({ ...prev, error: undefined }));
      return await messageBusService.sendCommand(command, data);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Command failed';
      setState(prev => ({ ...prev, error: errorMessage }));
      throw error;
    }
  }, []);

  // Get latest message by topic (for backward compatibility)
  const getLatestMessageByTopic = useCallback((topic: string) => {
    // This is a simplified implementation for backward compatibility
    // In a real implementation, you'd store messages by topic
    return state.latestMessage;
  }, [state.latestMessage]);

  // Additional backward compatibility functions
  const clearMessages = useCallback(() => {
    setState(prev => ({ ...prev, messages: [] }));
  }, []);

  const getStats = useCallback(() => {
    return {
      connected: state.connected,
      subscriptions: state.subscriptions,
      messagesReceived: state.messagesReceived || 0
    };
  }, [state.connected, state.subscriptions, state.messagesReceived]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !state.connected) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      subscriptionsRef.current.forEach(subId => {
        messageBusService.unsubscribe(subId);
      });
      subscriptionsRef.current.clear();
    };
  }, [autoConnect, connect, state.connected]);

  // Set up periodic state updates
  useEffect(() => {
    const interval = setInterval(updateState, 1000);
    return () => clearInterval(interval);
  }, [updateState]);

  return {
    ...state,
    connect,
    disconnect,
    subscribe,
    sendCommand,
    isConnected: state.connected,
    getLatestMessageByTopic,
    clearMessages,
    getStats,
    // Additional backward compatibility properties
    connectionStatus: state.connectionStatus,
    latestMessage: state.latestMessage,
    messages: state.messages,
    connectionInfo: state.connectionInfo,
    messagesReceived: state.messagesReceived
  };
}

/**
 * Hook for real-time market data (high-frequency updates)
 */
export function useMarketDataStream(symbol: string) {
  const { subscribe, isConnected } = useMessageBus();
  const [marketData, setMarketData] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!isConnected || !symbol) return;

    const unsubscribe = subscribe(`market_data.${symbol}`, (data) => {
      setMarketData(data);
      setLastUpdate(new Date());
    });

    return unsubscribe;
  }, [subscribe, isConnected, symbol]);

  return {
    marketData,
    lastUpdate,
    isConnected
  };
}

/**
 * Hook for real-time portfolio updates (bypasses HTTP entirely)
 */
export function useRealtimePortfolio(portfolioId: string) {
  const { subscribe, sendCommand, isConnected } = useMessageBus();
  const [portfolio, setPortfolio] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get initial portfolio data via message bus command
  useEffect(() => {
    if (!isConnected || !portfolioId) return;

    const fetchPortfolio = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await sendCommand('get_realtime_portfolio', { portfolioId });
        setPortfolio(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch portfolio');
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();
  }, [isConnected, portfolioId, sendCommand]);

  // Subscribe to real-time portfolio updates
  useEffect(() => {
    if (!isConnected || !portfolioId) return;

    const unsubscribe = subscribe(`portfolio.${portfolioId}`, (data) => {
      setPortfolio(prev => ({ ...prev, ...data }));
    });

    return unsubscribe;
  }, [subscribe, isConnected, portfolioId]);

  return {
    portfolio,
    loading,
    error,
    isConnected
  };
}

/**
 * Hook for real-time order updates
 */
export function useOrderStream() {
  const { subscribe, isConnected } = useMessageBus();
  const [orders, setOrders] = useState<any[]>([]);
  const [lastOrderUpdate, setLastOrderUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('order_updates', (orderData) => {
      setOrders(prev => {
        // Update existing order or add new one
        const existingIndex = prev.findIndex(o => o.order_id === orderData.order_id);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = { ...updated[existingIndex], ...orderData };
          return updated;
        } else {
          return [...prev, orderData];
        }
      });
      setLastOrderUpdate(new Date());
    });

    return unsubscribe;
  }, [subscribe, isConnected]);

  return {
    orders,
    lastOrderUpdate,
    isConnected
  };
}

/**
 * Hook for real-time position updates
 */
export function usePositionStream() {
  const { subscribe, isConnected } = useMessageBus();
  const [positions, setPositions] = useState<any[]>([]);
  const [lastPositionUpdate, setLastPositionUpdate] = useState<Date | null>(null);

  useEffect(() => {
    if (!isConnected) return;

    const unsubscribe = subscribe('position_updates', (positionData) => {
      setPositions(prev => {
        // Update existing position or add new one
        const existingIndex = prev.findIndex(p => p.position_id === positionData.position_id);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = { ...updated[existingIndex], ...positionData };
          return updated;
        } else {
          return [...prev, positionData];
        }
      });
      setLastPositionUpdate(new Date());
    });

    return unsubscribe;
  }, [subscribe, isConnected]);

  return {
    positions,
    lastPositionUpdate,
    isConnected
  };
}