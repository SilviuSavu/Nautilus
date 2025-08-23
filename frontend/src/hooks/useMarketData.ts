/**
 * useMarketData - Enhanced WebSocket hook for real-time market data streaming
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Integrates with the new WebSocket management system for improved performance,
 * reliability, and comprehensive market data streaming capabilities.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useWebSocketManager } from './useWebSocketManager';
import { useRealTimeData } from './useRealTimeData';

interface MarketDataMessage {
  type: string;
  symbol: string;
  data: {
    price: number;
    bid: number;
    ask: number;
    volume: number;
    change?: number;
    change_percent?: number;
    timestamp: string;
  };
  timestamp: string;
  connection_id?: string;
}

interface MarketDataState {
  isConnected: boolean;
  symbol: string | null;
  price: number | null;
  bid: number | null;
  ask: number | null;
  volume: number | null;
  change: number | null;
  changePercent: number | null;
  lastUpdate: Date | null;
  connectionError: string | null;
  connectionAttempts: number;
  venue: string | null;
  dataFreshness: number;
  messageCount: number;
}

interface UseMarketDataOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export const useMarketData = (
  symbol?: string,
  options: UseMarketDataOptions = {}
) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000
  } = options;

  // Use new WebSocket infrastructure
  const {
    connectionState,
    connectionError,
    connectionAttempts,
    subscribe,
    unsubscribe,
    addMessageHandler,
    sendMessage: wsSendMessage,
    connect: wsConnect
  } = useWebSocketManager({ autoReconnect, reconnectInterval, maxReconnectAttempts });

  // Use real-time data hook for market data stream
  const { 
    marketData, 
    subscribeToStream, 
    unsubscribeFromStream,
    isConnected 
  } = useRealTimeData();

  const [state, setState] = useState<MarketDataState>({
    isConnected: false,
    symbol: null,
    price: null,
    bid: null,
    ask: null,
    volume: null,
    change: null,
    changePercent: null,
    lastUpdate: null,
    connectionError: null,
    connectionAttempts: 0,
    venue: null,
    dataFreshness: 0,
    messageCount: 0
  });

  const currentSymbolRef = useRef<string | undefined>(symbol);
  const subscriptionIdRef = useRef<string | null>(null);
  const messageCountRef = useRef(0);

  // Process market data updates from real-time stream
  useEffect(() => {
    if (marketData && marketData.length > 0 && currentSymbolRef.current) {
      const symbolData = marketData.find(data => data.symbol === currentSymbolRef.current);
      
      if (symbolData) {
        setState(prev => ({
          ...prev,
          price: symbolData.price,
          bid: symbolData.bid,
          ask: symbolData.ask,
          volume: symbolData.volume,
          change: symbolData.change,
          changePercent: symbolData.changePercent,
          lastUpdate: new Date(symbolData.timestamp),
          venue: symbolData.venue || null,
          dataFreshness: Date.now() - new Date(symbolData.timestamp).getTime(),
          messageCount: prev.messageCount + 1
        }));
      }
    }
  }, [marketData]);

  // Update connection state
  useEffect(() => {
    setState(prev => ({
      ...prev,
      isConnected: isConnected && connectionState === 'connected',
      connectionError: connectionError,
      connectionAttempts: connectionAttempts
    }));
  }, [isConnected, connectionState, connectionError, connectionAttempts]);

  // Subscribe to symbol market data
  const subscribeToSymbol = useCallback(async (newSymbol: string) => {
    if (!newSymbol || newSymbol === currentSymbolRef.current) {
      return;
    }

    try {
      // Unsubscribe from previous symbol if exists
      if (subscriptionIdRef.current && currentSymbolRef.current) {
        await unsubscribeFromStream('market_data');
        subscriptionIdRef.current = null;
      }

      // Subscribe to new symbol
      const subscriptionId = await subscribeToStream('market_data', {
        symbols: [newSymbol]
      });

      if (subscriptionId) {
        subscriptionIdRef.current = subscriptionId;
        currentSymbolRef.current = newSymbol;
        
        setState(prev => ({
          ...prev,
          symbol: newSymbol,
          messageCount: 0
        }));

        console.log(`Subscribed to market data for ${newSymbol}`);
      }
    } catch (error) {
      console.error(`Failed to subscribe to market data for ${newSymbol}:`, error);
      setState(prev => ({
        ...prev,
        connectionError: `Failed to subscribe to ${newSymbol}`
      }));
    }
  }, [subscribeToStream, unsubscribeFromStream]);

  // Disconnect and unsubscribe
  const disconnect = useCallback(async () => {
    try {
      if (subscriptionIdRef.current) {
        await unsubscribeFromStream('market_data');
        subscriptionIdRef.current = null;
      }

      setState(prev => ({
        ...prev,
        symbol: null,
        price: null,
        bid: null,
        ask: null,
        volume: null,
        change: null,
        changePercent: null,
        lastUpdate: null,
        venue: null,
        messageCount: 0
      }));

      currentSymbolRef.current = undefined;
      console.log('Disconnected from market data');
    } catch (error) {
      console.error('Failed to disconnect from market data:', error);
    }
  }, [unsubscribeFromStream]);

  // Enhanced message sending with market data context
  const sendMessage = useCallback(async (message: any): Promise<boolean> => {
    const marketDataMessage = {
      ...message,
      symbol: currentSymbolRef.current,
      type: message.type || 'market_data_command'
    };

    return await wsSendMessage(marketDataMessage);
  }, [wsSendMessage]);

  // Request market data snapshot
  const requestSnapshot = useCallback(async (): Promise<boolean> => {
    if (!currentSymbolRef.current) {
      return false;
    }

    return await sendMessage({
      type: 'get_snapshot',
      symbol: currentSymbolRef.current
    });
  }, [sendMessage]);

  // Request historical data
  const requestHistoricalData = useCallback(async (
    timeframe: string = '1D',
    periods: number = 30
  ): Promise<boolean> => {
    if (!currentSymbolRef.current) {
      return false;
    }

    return await sendMessage({
      type: 'get_historical',
      symbol: currentSymbolRef.current,
      timeframe,
      periods
    });
  }, [sendMessage]);

  // Get market data statistics
  const getMarketDataStats = useCallback(() => {
    return {
      symbol: state.symbol,
      messageCount: state.messageCount,
      dataFreshness: state.dataFreshness,
      connectionQuality: state.dataFreshness < 5000 ? 'excellent' : 
                        state.dataFreshness < 15000 ? 'good' : 'poor',
      lastUpdate: state.lastUpdate,
      venue: state.venue
    };
  }, [state]);

  // Initialize connection and subscribe to symbol
  useEffect(() => {
    if (symbol) {
      // Ensure WebSocket is connected first
      if (connectionState !== 'connected') {
        wsConnect();
      }
      
      // Subscribe to symbol when connected
      if (connectionState === 'connected') {
        subscribeToSymbol(symbol);
      }
    }

    return () => {
      disconnect();
    };
  }, [symbol, connectionState, subscribeToSymbol, disconnect, wsConnect]);

  // Calculate spread
  const spread = state.bid && state.ask ? state.ask - state.bid : null;
  const spreadBps = spread && state.price ? (spread / state.price) * 10000 : null;

  // Calculate price change indicators
  const isPositive = state.change ? state.change > 0 : false;
  const isNegative = state.change ? state.change < 0 : false;

  return {
    // Connection state
    isConnected: state.isConnected,
    connectionError: state.connectionError,
    connectionAttempts: state.connectionAttempts,
    
    // Market data
    symbol: state.symbol,
    price: state.price,
    bid: state.bid,
    ask: state.ask,
    volume: state.volume,
    change: state.change,
    changePercent: state.changePercent,
    lastUpdate: state.lastUpdate,
    venue: state.venue,
    
    // Enhanced metrics
    dataFreshness: state.dataFreshness,
    messageCount: state.messageCount,
    
    // Calculated values
    spread,
    spreadBps,
    isPositive,
    isNegative,
    
    // Connection control
    connect: subscribeToSymbol,
    disconnect,
    
    // Enhanced message sending
    sendMessage,
    requestSnapshot,
    requestHistoricalData,
    
    // Analytics
    getMarketDataStats,
    
    // Stream management
    subscriptionId: subscriptionIdRef.current
  };
};

export default useMarketData;