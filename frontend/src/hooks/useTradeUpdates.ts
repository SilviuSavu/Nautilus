/**
 * useTradeUpdates - WebSocket hook for real-time trade execution updates
 * Sprint 3 Priority 1: WebSocket Streaming Infrastructure
 * 
 * Connects to /ws/trades/updates endpoint for live trade and order execution updates
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface TradeUpdate {
  type: 'trade_executed' | 'order_status' | 'position_update' | 'pnl_update';
  data: {
    trade_id?: string;
    order_id?: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    status?: 'FILLED' | 'PARTIAL' | 'CANCELLED' | 'REJECTED' | 'PENDING';
    filled_quantity?: number;
    remaining_quantity?: number;
    avg_fill_price?: number;
    commission?: number;
    fees?: number;
    pnl?: number;
    unrealized_pnl?: number;
    realized_pnl?: number;
    portfolio_id: string;
    strategy_id?: string;
    timestamp: string;
    execution_id?: string;
    venue?: string;
    slippage_bps?: number;
    execution_time_ms?: number;
  };
  timestamp: string;
  connection_id?: string;
}

interface TradeUpdatesState {
  isConnected: boolean;
  recentTrades: TradeUpdate[];
  recentOrders: TradeUpdate[];
  totalTrades: number;
  totalVolume: number;
  totalPnL: number;
  lastUpdate: Date | null;
  connectionError: string | null;
  connectionAttempts: number;
}

interface UseTradeUpdatesOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  maxRecentItems?: number;
  portfolioFilter?: string[];
  strategyFilter?: string[];
}

export const useTradeUpdates = (options: UseTradeUpdatesOptions = {}) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000,
    maxRecentItems = 100,
    portfolioFilter = [],
    strategyFilter = []
  } = options;

  const [state, setState] = useState<TradeUpdatesState>({
    isConnected: false,
    recentTrades: [],
    recentOrders: [],
    totalTrades: 0,
    totalVolume: 0,
    totalPnL: 0,
    lastUpdate: null,
    connectionError: null,
    connectionAttempts: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout>();
  const mountedRef = useRef(true);

  // Get WebSocket URL for trade updates
  const getWebSocketUrl = useCallback(() => {
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
    const protocol = apiBaseUrl.startsWith('https') ? 'wss:' : 'ws:';
    const host = apiBaseUrl.replace(/^https?:\/\//, '');
    return `${protocol}//${host}/ws/trades/updates`;
  }, []);

  // Send heartbeat to keep connection alive
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString()
        }));
      } catch (error) {
        console.error('Failed to send trade updates heartbeat:', error);
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

  // Check if update should be included based on filters
  const shouldIncludeUpdate = useCallback((update: TradeUpdate) => {
    // Portfolio filter
    if (portfolioFilter.length > 0 && !portfolioFilter.includes(update.data.portfolio_id)) {
      return false;
    }

    // Strategy filter
    if (strategyFilter.length > 0 && update.data.strategy_id && 
        !strategyFilter.includes(update.data.strategy_id)) {
      return false;
    }

    return true;
  }, [portfolioFilter, strategyFilter]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const wsUrl = getWebSocketUrl();
      console.log('Connecting to trade updates WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // Connection opened
      ws.onopen = () => {
        if (!mountedRef.current) return;
        
        console.log('Trade updates WebSocket connected');
        setState(prev => ({
          ...prev,
          isConnected: true,
          connectionError: null,
          connectionAttempts: 0
        }));

        // Subscribe with filters
        try {
          ws.send(JSON.stringify({
            type: 'subscribe',
            event_types: ['trade_executed', 'order_status', 'position_update', 'pnl_update'],
            filters: {
              portfolios: portfolioFilter,
              strategies: strategyFilter
            },
            timestamp: new Date().toISOString()
          }));
        } catch (error) {
          console.error('Failed to subscribe to trade updates:', error);
        }

        startHeartbeat();
      };

      // Message received
      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const update: TradeUpdate = JSON.parse(event.data);
          
          if (!shouldIncludeUpdate(update)) {
            return;
          }

          setState(prev => {
            const newState = { ...prev };
            const updateTime = new Date(update.timestamp);
            
            // Update based on message type
            if (update.type === 'trade_executed') {
              // Add to recent trades
              const newTrades = [update, ...prev.recentTrades].slice(0, maxRecentItems);
              newState.recentTrades = newTrades;
              
              // Update totals
              newState.totalTrades = prev.totalTrades + 1;
              newState.totalVolume = prev.totalVolume + (update.data.quantity || 0);
              
              if (update.data.pnl) {
                newState.totalPnL = prev.totalPnL + update.data.pnl;
              }
              
            } else if (update.type === 'order_status') {
              // Add to recent orders
              const newOrders = [update, ...prev.recentOrders].slice(0, maxRecentItems);
              newState.recentOrders = newOrders;
              
            } else if (update.type === 'pnl_update') {
              // Update P&L totals
              if (update.data.realized_pnl !== undefined) {
                newState.totalPnL = update.data.realized_pnl;
              }
            }
            
            newState.lastUpdate = updateTime;
            return newState;
          });
          
        } catch (error) {
          console.error('Failed to parse trade update message:', error);
        }
      };

      // Connection error
      ws.onerror = (error) => {
        if (!mountedRef.current) return;
        
        console.error('Trade updates WebSocket error:', error);
        setState(prev => ({
          ...prev,
          connectionError: 'Trade updates connection error'
        }));
      };

      // Connection closed
      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        
        console.log('Trade updates WebSocket closed:', event.code, event.reason);
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
              console.log(`Attempting to reconnect trade updates... (${state.connectionAttempts + 1}/${maxReconnectAttempts})`);
              connect();
            }
          }, reconnectInterval);
        }
      };

    } catch (error) {
      console.error('Failed to create trade updates WebSocket connection:', error);
      setState(prev => ({
        ...prev,
        connectionError: 'Failed to create WebSocket connection'
      }));
    }
  }, [getWebSocketUrl, autoReconnect, reconnectInterval, maxReconnectAttempts, state.connectionAttempts, startHeartbeat, stopHeartbeat, shouldIncludeUpdate, portfolioFilter, strategyFilter, maxRecentItems]);

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
        console.error('Failed to send trade update message:', error);
        return false;
      }
    }
    return false;
  }, []);

  // Update subscription filters
  const updateFilters = useCallback((newPortfolios?: string[], newStrategies?: string[]) => {
    return sendMessage({
      type: 'update_filters',
      filters: {
        portfolios: newPortfolios || portfolioFilter,
        strategies: newStrategies || strategyFilter
      }
    });
  }, [sendMessage, portfolioFilter, strategyFilter]);

  // Clear recent updates
  const clearRecentUpdates = useCallback(() => {
    setState(prev => ({
      ...prev,
      recentTrades: [],
      recentOrders: []
    }));
  }, []);

  // Get trade summary statistics
  const getTradesSummary = useCallback(() => {
    const trades = state.recentTrades.filter(t => t.type === 'trade_executed');
    
    if (trades.length === 0) {
      return {
        count: 0,
        totalVolume: 0,
        avgPrice: 0,
        avgSlippage: 0,
        avgExecutionTime: 0
      };
    }

    const totalVolume = trades.reduce((sum, t) => sum + (t.data.quantity || 0), 0);
    const totalValue = trades.reduce((sum, t) => sum + (t.data.price * t.data.quantity || 0), 0);
    const avgPrice = totalVolume > 0 ? totalValue / totalVolume : 0;
    
    const tradesWithSlippage = trades.filter(t => t.data.slippage_bps !== undefined);
    const avgSlippage = tradesWithSlippage.length > 0 
      ? tradesWithSlippage.reduce((sum, t) => sum + (t.data.slippage_bps || 0), 0) / tradesWithSlippage.length 
      : 0;
      
    const tradesWithExecTime = trades.filter(t => t.data.execution_time_ms !== undefined);
    const avgExecutionTime = tradesWithExecTime.length > 0 
      ? tradesWithExecTime.reduce((sum, t) => sum + (t.data.execution_time_ms || 0), 0) / tradesWithExecTime.length 
      : 0;

    return {
      count: trades.length,
      totalVolume,
      avgPrice,
      avgSlippage,
      avgExecutionTime
    };
  }, [state.recentTrades]);

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
    
    // Trade data
    recentTrades: state.recentTrades,
    recentOrders: state.recentOrders,
    totalTrades: state.totalTrades,
    totalVolume: state.totalVolume,
    totalPnL: state.totalPnL,
    lastUpdate: state.lastUpdate,
    
    // Connection control
    connect,
    disconnect,
    
    // Message sending
    sendMessage,
    updateFilters,
    
    // Utilities
    clearRecentUpdates,
    getTradesSummary,
    sendHeartbeat
  };
};

export default useTradeUpdates;