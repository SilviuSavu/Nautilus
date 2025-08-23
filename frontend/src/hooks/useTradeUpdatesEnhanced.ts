/**
 * useTradeUpdatesEnhanced - Enhanced WebSocket hook for real-time trade execution updates
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Integrates with the new WebSocket management system for improved trade updates
 * with comprehensive filtering, analytics, and performance monitoring.
 */

import { useState, useEffect, useCallback } from 'react';
import { useRealTimeData } from './useRealTimeData';
import { useWebSocketManager } from './useWebSocketManager';

interface TradeUpdateEnhanced {
  id: string;
  type: 'trade_executed' | 'order_status' | 'position_update' | 'pnl_update';
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  status: string;
  timestamp: string;
  portfolioId?: string;
  strategyId?: string;
  executionId?: string;
  venue?: string;
  slippage?: number;
  executionTime?: number;
  commission?: number;
  fees?: number;
  pnl?: number;
  unrealizedPnl?: number;
  realizedPnl?: number;
}

interface TradeUpdatesStats {
  totalTrades: number;
  totalVolume: number;
  totalPnl: number;
  averageSlippage: number;
  averageExecutionTime: number;
  winRate: number;
  lastTradeTime: string | null;
  tradesPerMinute: number;
}

interface UseTradeUpdatesEnhancedOptions {
  portfolioFilter?: string[];
  strategyFilter?: string[];
  symbolFilter?: string[];
  maxRecentTrades?: number;
  enableAnalytics?: boolean;
}

export const useTradeUpdatesEnhanced = (options: UseTradeUpdatesEnhancedOptions = {}) => {
  const {
    portfolioFilter = [],
    strategyFilter = [],
    symbolFilter = [],
    maxRecentTrades = 100,
    enableAnalytics = true
  } = options;

  // Use new WebSocket infrastructure
  const { 
    tradeUpdates, 
    subscribeToStream, 
    unsubscribeFromStream,
    isConnected 
  } = useRealTimeData();

  const {
    connectionState,
    connectionError,
    sendMessage
  } = useWebSocketManager();

  const [recentTrades, setRecentTrades] = useState<TradeUpdateEnhanced[]>([]);
  const [stats, setStats] = useState<TradeUpdatesStats>({
    totalTrades: 0,
    totalVolume: 0,
    totalPnl: 0,
    averageSlippage: 0,
    averageExecutionTime: 0,
    winRate: 0,
    lastTradeTime: null,
    tradesPerMinute: 0
  });

  // Convert raw trade updates to enhanced format
  const processTradeUpdates = useCallback(() => {
    if (!tradeUpdates || tradeUpdates.length === 0) return;

    const filteredTrades = tradeUpdates
      .filter(trade => {
        // Apply filters
        if (portfolioFilter.length > 0 && trade.portfolioId && !portfolioFilter.includes(trade.portfolioId)) {
          return false;
        }
        if (strategyFilter.length > 0 && trade.strategyId && !strategyFilter.includes(trade.strategyId)) {
          return false;
        }
        if (symbolFilter.length > 0 && !symbolFilter.includes(trade.symbol)) {
          return false;
        }
        return true;
      })
      .map(trade => ({
        id: trade.id,
        type: 'trade_executed' as const,
        symbol: trade.symbol,
        side: trade.side,
        quantity: trade.quantity,
        price: trade.price,
        status: trade.status,
        timestamp: trade.timestamp,
        portfolioId: trade.portfolioId,
        strategyId: trade.strategyId,
        executionId: trade.executionId,
        venue: trade.venue,
        slippage: trade.slippage,
        executionTime: trade.executionTime,
        commission: trade.commission,
        fees: trade.fees,
        pnl: trade.pnl,
        unrealizedPnl: trade.unrealizedPnl,
        realizedPnl: trade.realizedPnl
      }))
      .slice(0, maxRecentTrades);

    setRecentTrades(filteredTrades);

    // Calculate statistics if analytics enabled
    if (enableAnalytics && filteredTrades.length > 0) {
      const totalTrades = filteredTrades.length;
      const totalVolume = filteredTrades.reduce((sum, trade) => sum + trade.quantity, 0);
      const totalPnl = filteredTrades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
      
      const tradesWithSlippage = filteredTrades.filter(trade => trade.slippage !== undefined);
      const averageSlippage = tradesWithSlippage.length > 0 
        ? tradesWithSlippage.reduce((sum, trade) => sum + (trade.slippage || 0), 0) / tradesWithSlippage.length
        : 0;

      const tradesWithExecutionTime = filteredTrades.filter(trade => trade.executionTime !== undefined);
      const averageExecutionTime = tradesWithExecutionTime.length > 0 
        ? tradesWithExecutionTime.reduce((sum, trade) => sum + (trade.executionTime || 0), 0) / tradesWithExecutionTime.length
        : 0;

      const profitableTrades = filteredTrades.filter(trade => (trade.pnl || 0) > 0);
      const winRate = totalTrades > 0 ? (profitableTrades.length / totalTrades) * 100 : 0;

      const lastTradeTime = filteredTrades.length > 0 ? filteredTrades[0].timestamp : null;

      // Calculate trades per minute
      const now = new Date();
      const oneMinuteAgo = new Date(now.getTime() - 60000);
      const recentTrades = filteredTrades.filter(trade => 
        new Date(trade.timestamp) > oneMinuteAgo
      );
      const tradesPerMinute = recentTrades.length;

      setStats({
        totalTrades,
        totalVolume,
        totalPnl,
        averageSlippage,
        averageExecutionTime,
        winRate,
        lastTradeTime,
        tradesPerMinute
      });
    }
  }, [
    tradeUpdates,
    portfolioFilter,
    strategyFilter,
    symbolFilter,
    maxRecentTrades,
    enableAnalytics
  ]);

  // Subscribe to trade updates stream
  useEffect(() => {
    if (connectionState === 'connected') {
      const filters: any = {};
      
      if (portfolioFilter.length > 0) {
        filters.portfolio_ids = portfolioFilter;
      }
      if (strategyFilter.length > 0) {
        filters.strategy_ids = strategyFilter;
      }
      if (symbolFilter.length > 0) {
        filters.symbols = symbolFilter;
      }

      subscribeToStream('trade_updates', filters);
    }

    return () => {
      unsubscribeFromStream('trade_updates');
    };
  }, [connectionState, portfolioFilter, strategyFilter, symbolFilter, subscribeToStream, unsubscribeFromStream]);

  // Process trade updates when they change
  useEffect(() => {
    processTradeUpdates();
  }, [processTradeUpdates]);

  // Get trade by ID
  const getTradeById = useCallback((tradeId: string): TradeUpdateEnhanced | null => {
    return recentTrades.find(trade => trade.id === tradeId) || null;
  }, [recentTrades]);

  // Get trades by symbol
  const getTradesBySymbol = useCallback((symbol: string): TradeUpdateEnhanced[] => {
    return recentTrades.filter(trade => trade.symbol === symbol);
  }, [recentTrades]);

  // Get trades by portfolio
  const getTradesByPortfolio = useCallback((portfolioId: string): TradeUpdateEnhanced[] => {
    return recentTrades.filter(trade => trade.portfolioId === portfolioId);
  }, [recentTrades]);

  // Clear trade history
  const clearTradeHistory = useCallback(() => {
    setRecentTrades([]);
    setStats({
      totalTrades: 0,
      totalVolume: 0,
      totalPnl: 0,
      averageSlippage: 0,
      averageExecutionTime: 0,
      winRate: 0,
      lastTradeTime: null,
      tradesPerMinute: 0
    });
  }, []);

  // Update trade filters
  const updateFilters = useCallback(async (filters: {
    portfolioIds?: string[];
    strategyIds?: string[];
    symbols?: string[];
  }) => {
    // Unsubscribe from current stream
    await unsubscribeFromStream('trade_updates');

    // Subscribe with new filters
    await subscribeToStream('trade_updates', {
      portfolio_ids: filters.portfolioIds,
      strategy_ids: filters.strategyIds,
      symbols: filters.symbols
    });
  }, [subscribeToStream, unsubscribeFromStream]);

  // Request trade history
  const requestTradeHistory = useCallback(async (
    startDate?: string,
    endDate?: string,
    limit: number = 100
  ) => {
    return await sendMessage({
      type: 'get_trade_history',
      start_date: startDate,
      end_date: endDate,
      limit,
      filters: {
        portfolio_ids: portfolioFilter,
        strategy_ids: strategyFilter,
        symbols: symbolFilter
      }
    });
  }, [sendMessage, portfolioFilter, strategyFilter, symbolFilter]);

  return {
    // Connection state
    isConnected: isConnected && connectionState === 'connected',
    connectionError,

    // Trade data
    recentTrades,
    stats,

    // Analytics functions
    getTradeById,
    getTradesBySymbol,
    getTradesByPortfolio,

    // Control functions
    clearTradeHistory,
    updateFilters,
    requestTradeHistory,

    // Current filters
    activeFilters: {
      portfolioFilter,
      strategyFilter,
      symbolFilter
    }
  };
};

export default useTradeUpdatesEnhanced;