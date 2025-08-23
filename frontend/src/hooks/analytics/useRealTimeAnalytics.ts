/**
 * useRealTimeAnalytics Hook - Sprint 3 Integration
 * Real-time analytics with streaming updates and sub-second calculations
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useMessageBus } from '../useMessageBus';

export interface RealTimeAnalyticsData {
  portfolio_id: string;
  timestamp: string;
  pnl: {
    realized: number;
    unrealized: number;
    total: number;
    daily_change: number;
    daily_change_pct: number;
  };
  risk_metrics: {
    var_1d: number;
    var_5d: number;
    expected_shortfall: number;
    beta: number;
    volatility: number;
    max_drawdown: number;
  };
  performance: {
    total_return: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    alpha: number;
    information_ratio: number;
  };
  positions: {
    long_exposure: number;
    short_exposure: number;
    net_exposure: number;
    gross_exposure: number;
    leverage: number;
  };
  execution: {
    fill_rate: number;
    avg_slippage: number;
    implementation_shortfall: number;
    market_impact: number;
  };
}

export interface UseRealTimeAnalyticsOptions {
  portfolioId: string;
  updateInterval?: number; // milliseconds, default 250ms for sub-second updates
  enableStreaming?: boolean;
  autoStart?: boolean;
  bufferSize?: number; // number of historical data points to keep
}

export interface UseRealTimeAnalyticsReturn {
  // Current data
  currentData: RealTimeAnalyticsData | null;
  historicalData: RealTimeAnalyticsData[];
  
  // Status
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
  updateCount: number;
  
  // Controls
  start: () => void;
  stop: () => void;
  reset: () => void;
  
  // Analytics
  getTrends: () => {
    pnl_trend: number;
    volatility_trend: number;
    sharpe_trend: number;
    exposure_trend: number;
  };
  
  // Performance stats
  getPerformanceStats: () => {
    avg_latency: number;
    update_frequency: number;
    data_completeness: number;
  };
}

export function useRealTimeAnalytics(
  options: UseRealTimeAnalyticsOptions
): UseRealTimeAnalyticsReturn {
  const { portfolioId, updateInterval = 250, enableStreaming = true, autoStart = true, bufferSize = 1000 } = options;
  
  // State
  const [currentData, setCurrentData] = useState<RealTimeAnalyticsData | null>(null);
  const [historicalData, setHistoricalData] = useState<RealTimeAnalyticsData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [updateCount, setUpdateCount] = useState(0);
  
  // Refs
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  const latencyRef = useRef<number[]>([]);
  const startTimeRef = useRef<number>(0);
  
  // Message bus for WebSocket communication
  const { isConnected: messageBusConnected, subscribe, unsubscribe } = useMessageBus();
  
  // API base URL from environment
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stop();
    };
  }, []);
  
  // Fetch analytics data from API
  const fetchAnalyticsData = useCallback(async () => {
    if (!portfolioId) return;
    
    const requestStart = performance.now();
    setIsLoading(true);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/analytics/portfolio/${portfolioId}/summary`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      const requestEnd = performance.now();
      const latency = requestEnd - requestStart;
      
      // Track latency
      latencyRef.current.push(latency);
      if (latencyRef.current.length > 100) {
        latencyRef.current.shift();
      }
      
      if (isMountedRef.current) {
        setCurrentData(data);
        setLastUpdate(new Date());
        setUpdateCount(prev => prev + 1);
        setError(null);
        
        // Add to historical data
        setHistoricalData(prev => {
          const newData = [...prev, data];
          return newData.length > bufferSize ? newData.slice(-bufferSize) : newData;
        });
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch analytics');
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [portfolioId, API_BASE_URL, bufferSize]);
  
  // WebSocket connection for real-time streaming
  const setupWebSocket = useCallback(() => {
    if (!enableStreaming || !portfolioId) return;
    
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/analytics/realtime/${portfolioId}`;
    
    try {
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        setIsConnected(true);
        setError(null);
      };
      
      websocketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (isMountedRef.current) {
            setCurrentData(data);
            setLastUpdate(new Date());
            setUpdateCount(prev => prev + 1);
            
            // Add to historical data
            setHistoricalData(prev => {
              const newData = [...prev, data];
              return newData.length > bufferSize ? newData.slice(-bufferSize) : newData;
            });
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };
      
      websocketRef.current.onerror = (error) => {
        setError('WebSocket connection failed');
        setIsConnected(false);
      };
      
      websocketRef.current.onclose = () => {
        setIsConnected(false);
        // Attempt to reconnect after 1 second
        setTimeout(() => {
          if (isMountedRef.current && enableStreaming) {
            setupWebSocket();
          }
        }, 1000);
      };
    } catch (err) {
      setError('Failed to establish WebSocket connection');
      setIsConnected(false);
    }
  }, [enableStreaming, portfolioId, API_BASE_URL, bufferSize]);
  
  // Start real-time analytics
  const start = useCallback(() => {
    if (!portfolioId) return;
    
    startTimeRef.current = Date.now();
    
    // Set up WebSocket if streaming is enabled
    if (enableStreaming) {
      setupWebSocket();
    }
    
    // Set up polling fallback or primary method
    intervalRef.current = setInterval(fetchAnalyticsData, updateInterval);
    
    // Initial fetch
    fetchAnalyticsData();
  }, [portfolioId, enableStreaming, updateInterval, setupWebSocket, fetchAnalyticsData]);
  
  // Stop real-time analytics
  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    
    setIsConnected(false);
  }, []);
  
  // Reset all data
  const reset = useCallback(() => {
    setCurrentData(null);
    setHistoricalData([]);
    setUpdateCount(0);
    setLastUpdate(null);
    setError(null);
    latencyRef.current = [];
  }, []);
  
  // Get trends from historical data
  const getTrends = useCallback(() => {
    if (historicalData.length < 2) {
      return {
        pnl_trend: 0,
        volatility_trend: 0,
        sharpe_trend: 0,
        exposure_trend: 0,
      };
    }
    
    const recent = historicalData.slice(-10); // Last 10 data points
    const firstPoint = recent[0];
    const lastPoint = recent[recent.length - 1];
    
    return {
      pnl_trend: lastPoint.pnl.total - firstPoint.pnl.total,
      volatility_trend: lastPoint.risk_metrics.volatility - firstPoint.risk_metrics.volatility,
      sharpe_trend: lastPoint.performance.sharpe_ratio - firstPoint.performance.sharpe_ratio,
      exposure_trend: lastPoint.positions.net_exposure - firstPoint.positions.net_exposure,
    };
  }, [historicalData]);
  
  // Get performance statistics
  const getPerformanceStats = useCallback(() => {
    const avgLatency = latencyRef.current.length > 0 
      ? latencyRef.current.reduce((sum, lat) => sum + lat, 0) / latencyRef.current.length 
      : 0;
    
    const uptime = startTimeRef.current > 0 ? Date.now() - startTimeRef.current : 0;
    const updateFrequency = updateCount > 0 && uptime > 0 ? (updateCount / (uptime / 1000)) : 0;
    
    // Calculate data completeness based on expected vs actual updates
    const expectedUpdates = uptime / updateInterval;
    const dataCompleteness = expectedUpdates > 0 ? (updateCount / expectedUpdates) * 100 : 0;
    
    return {
      avg_latency: avgLatency,
      update_frequency: updateFrequency,
      data_completeness: Math.min(dataCompleteness, 100), // Cap at 100%
    };
  }, [updateCount, updateInterval]);
  
  // Auto-start if enabled
  useEffect(() => {
    if (autoStart && portfolioId) {
      start();
    }
    
    return () => {
      stop();
    };
  }, [autoStart, portfolioId, start, stop]);
  
  return {
    // Current data
    currentData,
    historicalData,
    
    // Status
    isConnected,
    isLoading,
    error,
    lastUpdate,
    updateCount,
    
    // Controls
    start,
    stop,
    reset,
    
    // Analytics
    getTrends,
    
    // Performance stats
    getPerformanceStats,
  };
}

export default useRealTimeAnalytics;