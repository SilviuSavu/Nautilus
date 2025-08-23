/**
 * useRealTimeData Hook
 * Sprint 3: Advanced WebSocket Infrastructure
 * 
 * Generic real-time data streaming hook for all Sprint 3 message types
 * with automatic subscription management, data caching, and stream analytics.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketManager } from './useWebSocketManager';

interface MarketDataPoint {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  timestamp: string;
  venue?: string;
}

interface TradeUpdate {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  status: 'pending' | 'partial' | 'filled' | 'cancelled' | 'rejected';
  timestamp: string;
  userId?: string;
  executionId?: string;
}

interface RiskAlert {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  portfolio: string;
  metric: string;
  currentValue: number;
  threshold: number;
  timestamp: string;
}

interface PerformanceUpdate {
  portfolio: string;
  strategy?: string;
  pnl: number;
  returns: number;
  sharpe: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  timestamp: string;
}

interface OrderUpdate {
  orderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  orderType: string;
  status: string;
  filledQuantity: number;
  remainingQuantity: number;
  timestamp: string;
  userId?: string;
}

interface PositionUpdate {
  portfolioId: string;
  symbol: string;
  side: 'long' | 'short' | 'flat';
  quantity: number;
  avgPrice: number;
  marketPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  timestamp: string;
}

interface SystemHealthUpdate {
  component: string;
  status: 'healthy' | 'degraded' | 'error';
  cpu: number;
  memory: number;
  uptime: number;
  timestamp: string;
}

interface StreamStatistics {
  messagesPerSecond: number;
  totalMessages: number;
  errorCount: number;
  averageLatency: number;
  streamHealth: Record<string, number>;
  dataFreshness: Record<string, number>;
}

interface StreamSubscription {
  id: string;
  type: string;
  isActive: boolean;
  messageCount: number;
  lastMessage?: string;
}

export const useRealTimeData = () => {
  const {
    connectionState,
    addMessageHandler,
    subscribe,
    unsubscribe,
    sendMessage
  } = useWebSocketManager();

  // Data states
  const [marketData, setMarketData] = useState<MarketDataPoint[]>([]);
  const [tradeUpdates, setTradeUpdates] = useState<TradeUpdate[]>([]);
  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceUpdate[]>([]);
  const [orderUpdates, setOrderUpdates] = useState<OrderUpdate[]>([]);
  const [positionUpdates, setPositionUpdates] = useState<PositionUpdate[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealthUpdate[]>([]);

  // Stream management
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeStreams, setActiveStreams] = useState<StreamSubscription[]>([]);
  const [streamStatistics, setStreamStatistics] = useState<StreamStatistics>({
    messagesPerSecond: 0,
    totalMessages: 0,
    errorCount: 0,
    averageLatency: 0,
    streamHealth: {},
    dataFreshness: {}
  });

  // Refs for data management
  const messageHandlersRef = useRef<Map<string, string>>(new Map());
  const subscriptionIdsRef = useRef<Map<string, string>>(new Map());
  const dataBuffersRef = useRef<Map<string, any[]>>(new Map());
  const messageCounterRef = useRef(0);
  const lastMessageTimeRef = useRef(Date.now());

  // Data buffer configuration
  const MAX_BUFFER_SIZE = 1000;
  const DATA_RETENTION_TIME = 5 * 60 * 1000; // 5 minutes

  // Initialize data buffers
  useEffect(() => {
    dataBuffersRef.current.set('market_data', []);
    dataBuffersRef.current.set('trade_updates', []);
    dataBuffersRef.current.set('risk_alerts', []);
    dataBuffersRef.current.set('performance_updates', []);
    dataBuffersRef.current.set('order_updates', []);
    dataBuffersRef.current.set('position_updates', []);
    dataBuffersRef.current.set('system_health', []);
  }, []);

  // Clean old data periodically
  const cleanOldData = useCallback(() => {
    const now = Date.now();
    const cutoffTime = now - DATA_RETENTION_TIME;

    // Clean market data
    setMarketData(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));

    // Clean trade updates
    setTradeUpdates(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));

    // Clean performance data
    setPerformanceData(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));

    // Keep alerts longer (30 minutes)
    const alertCutoffTime = now - (30 * 60 * 1000);
    setRiskAlerts(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > alertCutoffTime
    ));

    // Clean other data types
    setOrderUpdates(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));

    setPositionUpdates(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));

    setSystemHealth(prev => prev.filter(item => 
      new Date(item.timestamp).getTime() > cutoffTime
    ));
  }, []);

  // Process market data message
  const processMarketData = useCallback((message: any) => {
    try {
      const marketPoint: MarketDataPoint = {
        symbol: message.data.symbol,
        price: message.data.price,
        change: message.data.change || 0,
        changePercent: message.data.change_percent || 0,
        volume: message.data.volume || 0,
        bid: message.data.bid_price || message.data.bid || 0,
        ask: message.data.ask_price || message.data.ask || 0,
        timestamp: message.timestamp,
        venue: message.data.venue
      };

      setMarketData(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(item => item.symbol === marketPoint.symbol);
        
        if (existingIndex >= 0) {
          updated[existingIndex] = marketPoint;
        } else {
          updated.push(marketPoint);
        }

        return updated.slice(-MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing market data:', error);
    }
  }, []);

  // Process trade updates
  const processTradeUpdate = useCallback((message: any) => {
    try {
      const tradeUpdate: TradeUpdate = {
        id: message.data.trade_id || message.data.id || message.messageId,
        symbol: message.data.symbol,
        side: message.data.side,
        quantity: message.data.quantity,
        price: message.data.price,
        status: message.data.status,
        timestamp: message.timestamp,
        userId: message.data.user_id,
        executionId: message.data.execution_id
      };

      setTradeUpdates(prev => {
        const updated = [tradeUpdate, ...prev];
        return updated.slice(0, MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing trade update:', error);
    }
  }, []);

  // Process risk alerts
  const processRiskAlert = useCallback((message: any) => {
    try {
      const riskAlert: RiskAlert = {
        id: message.data.alert_id || message.messageId,
        type: message.data.risk_type || message.data.type,
        severity: message.data.severity,
        message: message.data.message,
        portfolio: message.data.portfolio_id || message.data.portfolio,
        metric: message.data.metric,
        currentValue: message.data.current_value,
        threshold: message.data.threshold || message.data.limit_value,
        timestamp: message.timestamp
      };

      setRiskAlerts(prev => {
        const updated = [riskAlert, ...prev];
        return updated.slice(0, MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing risk alert:', error);
    }
  }, []);

  // Process performance updates
  const processPerformanceUpdate = useCallback((message: any) => {
    try {
      const performanceUpdate: PerformanceUpdate = {
        portfolio: message.data.portfolio_id,
        strategy: message.data.strategy_id,
        pnl: message.data.pnl || message.data.total_pnl,
        returns: message.data.returns || message.data.total_return,
        sharpe: message.data.sharpe_ratio || 0,
        maxDrawdown: message.data.max_drawdown || 0,
        winRate: message.data.win_rate || 0,
        totalTrades: message.data.trade_count || 0,
        timestamp: message.timestamp
      };

      setPerformanceData(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(item => 
          item.portfolio === performanceUpdate.portfolio && 
          item.strategy === performanceUpdate.strategy
        );
        
        if (existingIndex >= 0) {
          updated[existingIndex] = performanceUpdate;
        } else {
          updated.push(performanceUpdate);
        }

        return updated.slice(-MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing performance update:', error);
    }
  }, []);

  // Process order updates
  const processOrderUpdate = useCallback((message: any) => {
    try {
      const orderUpdate: OrderUpdate = {
        orderId: message.data.order_id,
        symbol: message.data.symbol,
        side: message.data.side,
        quantity: message.data.quantity,
        price: message.data.price,
        orderType: message.data.order_type || 'market',
        status: message.data.status || message.data.order_status,
        filledQuantity: message.data.filled_quantity || 0,
        remainingQuantity: message.data.remaining_quantity || message.data.quantity,
        timestamp: message.timestamp,
        userId: message.data.user_id
      };

      setOrderUpdates(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(item => item.orderId === orderUpdate.orderId);
        
        if (existingIndex >= 0) {
          updated[existingIndex] = orderUpdate;
        } else {
          updated.unshift(orderUpdate);
        }

        return updated.slice(0, MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing order update:', error);
    }
  }, []);

  // Process position updates
  const processPositionUpdate = useCallback((message: any) => {
    try {
      const positionUpdate: PositionUpdate = {
        portfolioId: message.data.portfolio_id,
        symbol: message.data.symbol,
        side: message.data.position_side,
        quantity: message.data.quantity,
        avgPrice: message.data.avg_price,
        marketPrice: message.data.market_price || message.data.current_price,
        unrealizedPnl: message.data.unrealized_pnl,
        realizedPnl: message.data.realized_pnl || 0,
        timestamp: message.timestamp
      };

      setPositionUpdates(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(item => 
          item.portfolioId === positionUpdate.portfolioId && 
          item.symbol === positionUpdate.symbol
        );
        
        if (existingIndex >= 0) {
          updated[existingIndex] = positionUpdate;
        } else {
          updated.push(positionUpdate);
        }

        return updated.slice(-MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing position update:', error);
    }
  }, []);

  // Process system health updates
  const processSystemHealth = useCallback((message: any) => {
    try {
      const healthUpdate: SystemHealthUpdate = {
        component: message.data.component,
        status: message.data.status,
        cpu: message.data.cpu_usage || message.data.cpu || 0,
        memory: message.data.memory_usage || message.data.memory || 0,
        uptime: message.data.uptime || 0,
        timestamp: message.timestamp
      };

      setSystemHealth(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(item => item.component === healthUpdate.component);
        
        if (existingIndex >= 0) {
          updated[existingIndex] = healthUpdate;
        } else {
          updated.push(healthUpdate);
        }

        return updated.slice(-MAX_BUFFER_SIZE);
      });

    } catch (error) {
      console.error('Error processing system health:', error);
    }
  }, []);

  // Update stream statistics
  const updateStreamStatistics = useCallback(() => {
    const now = Date.now();
    const timeDelta = (now - lastMessageTimeRef.current) / 1000;
    
    setStreamStatistics(prev => ({
      ...prev,
      messagesPerSecond: timeDelta > 0 ? messageCounterRef.current / timeDelta : 0,
      totalMessages: messageCounterRef.current,
      dataFreshness: {
        market_data: marketData.length > 0 ? now - new Date(marketData[0].timestamp).getTime() : 0,
        trade_updates: tradeUpdates.length > 0 ? now - new Date(tradeUpdates[0].timestamp).getTime() : 0,
        risk_alerts: riskAlerts.length > 0 ? now - new Date(riskAlerts[0].timestamp).getTime() : 0,
        performance_updates: performanceData.length > 0 ? now - new Date(performanceData[0].timestamp).getTime() : 0
      }
    }));
  }, [marketData, tradeUpdates, riskAlerts, performanceData]);

  // Start streaming
  const startStreaming = useCallback(async () => {
    if (connectionState !== 'connected' || isStreaming) {
      return;
    }

    try {
      setIsStreaming(true);
      
      // Initialize message counters
      messageCounterRef.current = 0;
      lastMessageTimeRef.current = Date.now();

      console.log('Real-time streaming started');
      
    } catch (error) {
      console.error('Failed to start streaming:', error);
      setIsStreaming(false);
    }
  }, [connectionState, isStreaming]);

  // Stop streaming
  const stopStreaming = useCallback(async () => {
    if (!isStreaming) {
      return;
    }

    try {
      // Unsubscribe from all streams
      const unsubscribePromises = Array.from(subscriptionIdsRef.current.entries()).map(
        async ([streamType, subscriptionId]) => {
          try {
            await unsubscribe(subscriptionId);
            subscriptionIdsRef.current.delete(streamType);
          } catch (error) {
            console.error(`Failed to unsubscribe from ${streamType}:`, error);
          }
        }
      );

      await Promise.allSettled(unsubscribePromises);

      // Remove message handlers
      messageHandlersRef.current.forEach((handlerId) => {
        // Handler cleanup is managed by useWebSocketManager
      });
      messageHandlersRef.current.clear();

      setIsStreaming(false);
      setActiveStreams([]);
      
      console.log('Real-time streaming stopped');
      
    } catch (error) {
      console.error('Error stopping streaming:', error);
    }
  }, [isStreaming, unsubscribe]);

  // Subscribe to specific stream
  const subscribeToStream = useCallback(async (
    streamType: string,
    filters?: Record<string, any>
  ): Promise<string | null> => {
    if (connectionState !== 'connected') {
      throw new Error('WebSocket not connected');
    }

    try {
      // Create message handler based on stream type
      let messageProcessor;
      switch (streamType) {
        case 'market_data':
          messageProcessor = processMarketData;
          break;
        case 'trade_updates':
          messageProcessor = processTradeUpdate;
          break;
        case 'risk_alerts':
          messageProcessor = processRiskAlert;
          break;
        case 'performance_updates':
          messageProcessor = processPerformanceUpdate;
          break;
        case 'order_updates':
          messageProcessor = processOrderUpdate;
          break;
        case 'position_updates':
          messageProcessor = processPositionUpdate;
          break;
        case 'system_health':
          messageProcessor = processSystemHealth;
          break;
        default:
          throw new Error(`Unsupported stream type: ${streamType}`);
      }

      // Add message handler
      const handlerId = addMessageHandler(
        `stream_${streamType}`,
        (message) => {
          messageCounterRef.current += 1;
          messageProcessor(message);
        },
        (message) => message.type === streamType
      );

      messageHandlersRef.current.set(streamType, handlerId);

      // Subscribe via WebSocket
      const subscriptionId = await subscribe(streamType, filters);
      subscriptionIdsRef.current.set(streamType, subscriptionId);

      // Add to active streams
      setActiveStreams(prev => [...prev, {
        id: subscriptionId,
        type: streamType,
        isActive: true,
        messageCount: 0,
        lastMessage: new Date().toISOString()
      }]);

      console.log(`Subscribed to ${streamType} stream`);
      return subscriptionId;

    } catch (error) {
      console.error(`Failed to subscribe to ${streamType}:`, error);
      return null;
    }
  }, [
    connectionState,
    addMessageHandler,
    subscribe,
    processMarketData,
    processTradeUpdate,
    processRiskAlert,
    processPerformanceUpdate,
    processOrderUpdate,
    processPositionUpdate,
    processSystemHealth
  ]);

  // Unsubscribe from specific stream
  const unsubscribeFromStream = useCallback(async (streamType: string): Promise<boolean> => {
    try {
      const subscriptionId = subscriptionIdsRef.current.get(streamType);
      if (!subscriptionId) {
        return false;
      }

      await unsubscribe(subscriptionId);
      subscriptionIdsRef.current.delete(streamType);

      // Remove message handler
      const handlerId = messageHandlersRef.current.get(streamType);
      if (handlerId) {
        messageHandlersRef.current.delete(streamType);
      }

      // Remove from active streams
      setActiveStreams(prev => prev.filter(stream => stream.type !== streamType));

      console.log(`Unsubscribed from ${streamType} stream`);
      return true;

    } catch (error) {
      console.error(`Failed to unsubscribe from ${streamType}:`, error);
      return false;
    }
  }, [unsubscribe]);

  // Get stream statistics
  const getStreamStatistics = useCallback(() => {
    updateStreamStatistics();
    return streamStatistics;
  }, [updateStreamStatistics, streamStatistics]);

  // Periodic cleanup and statistics update
  useEffect(() => {
    if (isStreaming) {
      const cleanupInterval = setInterval(cleanOldData, 60000); // Every minute
      const statsInterval = setInterval(updateStreamStatistics, 5000); // Every 5 seconds

      return () => {
        clearInterval(cleanupInterval);
        clearInterval(statsInterval);
      };
    }
  }, [isStreaming, cleanOldData, updateStreamStatistics]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return {
    // Data streams
    marketData,
    tradeUpdates,
    riskAlerts,
    performanceData,
    orderUpdates,
    positionUpdates,
    systemHealth,

    // Stream management
    isStreaming,
    activeStreams,
    streamStatistics,

    // Stream control
    startStreaming,
    stopStreaming,
    subscribeToStream,
    unsubscribeFromStream,

    // Analytics
    getStreamStatistics,

    // Connection status
    isConnected: connectionState === 'connected'
  };
};

export default useRealTimeData;