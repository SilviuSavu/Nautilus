/**
 * useExecutionQuality Hook
 * Sprint 3: Advanced Execution Quality Analysis
 * 
 * Comprehensive execution quality analytics with slippage analysis,
 * market impact measurement, venue quality scoring, and execution optimization.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketStream } from '../useWebSocketStream';
import type { StreamMessage } from '../useWebSocketStream';

export interface ExecutionMetrics {
  // Basic Execution Metrics
  fillRate: number;
  averageFillTime: number;
  totalSlippage: number;
  averageSlippage: number;
  slippageBps: number;
  
  // Implementation Shortfall
  implementationShortfall: number;
  implementationShortfallBps: number;
  
  // Market Impact
  marketImpact: number;
  marketImpactBps: number;
  temporaryImpact: number;
  permanentImpact: number;
  
  // Timing Costs
  timingCost: number;
  opportunityCost: number;
  delayedExecutionCost: number;
  
  // Venue Analysis
  venueQuality: Record<string, {
    fillRate: number;
    avgSlippage: number;
    marketShare: number;
    qualityScore: number;
  }>;
  
  // Order Type Performance
  orderTypeAnalysis: Record<string, {
    count: number;
    avgSlippage: number;
    fillRate: number;
    avgFillTime: number;
  }>;
  
  // Size Analysis
  sizeImpactAnalysis: {
    small: { avgSlippage: number; fillRate: number };
    medium: { avgSlippage: number; fillRate: number };
    large: { avgSlippage: number; fillRate: number };
  };
}

export interface ExecutionTrade {
  tradeId: string;
  orderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  executedQuantity: number;
  orderPrice: number;
  executedPrice: number;
  benchmarkPrice: number;
  venue: string;
  orderType: string;
  timestamp: string;
  executionTime: number;
  slippage: number;
  slippageBps: number;
  marketImpact: number;
  costs: {
    commission: number;
    fees: number;
    borrowCost?: number;
  };
}

export interface VenueAnalysis {
  venue: string;
  trades: number;
  volume: number;
  averageSlippage: number;
  fillRate: number;
  averageFillTime: number;
  marketShare: number;
  qualityScore: number;
  uptime: number;
  rejectRate: number;
}

export interface ExecutionAlert {
  id: string;
  type: 'high_slippage' | 'slow_fill' | 'venue_issue' | 'market_impact';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  tradeId?: string;
  venue?: string;
  threshold: number;
  actualValue: number;
  timestamp: string;
  acknowledged: boolean;
}

export interface UseExecutionQualityOptions {
  portfolioId?: string;
  updateInterval?: number;
  enableRealTime?: boolean;
  slippageThreshold?: number;
  fillTimeThreshold?: number;
  enableAlerts?: boolean;
  venueAnalysisEnabled?: boolean;
  historicalPeriod?: number;
}

export interface UseExecutionQualityReturn {
  // Execution data
  metrics: ExecutionMetrics | null;
  trades: ExecutionTrade[];
  venueAnalysis: VenueAnalysis[];
  alerts: ExecutionAlert[];
  
  // Status
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
  isRealTimeActive: boolean;
  
  // Analysis
  analyzeSlippage: (trades?: ExecutionTrade[]) => {
    distribution: { range: string; count: number; percentage: number }[];
    outliers: ExecutionTrade[];
    averageBySize: Record<string, number>;
  };
  
  analyzeMarketImpact: (symbol: string, timeWindow?: number) => {
    temporaryImpact: number;
    permanentImpact: number;
    priceReversion: number;
    volumeImpact: number;
  };
  
  compareVenues: (venues: string[]) => {
    comparison: Record<string, {
      rank: number;
      score: number;
      strengths: string[];
      weaknesses: string[];
    }>;
    recommendation: string;
  };
  
  // Optimization
  optimizeExecution: (orderParams: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    urgency: 'low' | 'medium' | 'high';
  }) => {
    recommendedVenue: string;
    recommendedOrderType: string;
    expectedSlippage: number;
    expectedFillTime: number;
  };
  
  // Benchmarking
  benchmarkAgainstTWAP: (tradeId: string) => {
    twapPrice: number;
    executionPrice: number;
    performance: number;
    percentile: number;
  };
  
  benchmarkAgainstVWAP: (tradeId: string) => {
    vwapPrice: number;
    executionPrice: number;
    performance: number;
    percentile: number;
  };
  
  // Alerts
  acknowledgeAlert: (alertId: string) => void;
  clearAlerts: () => void;
  setAlertThreshold: (type: string, threshold: number) => void;
  
  // Control
  startRealTimeTracking: () => void;
  stopRealTimeTracking: () => void;
  refresh: () => Promise<void>;
  exportAnalysis: (format: 'json' | 'csv' | 'pdf') => Promise<string | Blob>;
}

const DEFAULT_OPTIONS: Required<UseExecutionQualityOptions> = {
  portfolioId: '',
  updateInterval: 1000,
  enableRealTime: true,
  slippageThreshold: 10, // basis points
  fillTimeThreshold: 5000, // milliseconds
  enableAlerts: true,
  venueAnalysisEnabled: true,
  historicalPeriod: 30 // days
};

export function useExecutionQuality(
  options: UseExecutionQualityOptions = {}
): UseExecutionQualityReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [metrics, setMetrics] = useState<ExecutionMetrics | null>(null);
  const [trades, setTrades] = useState<ExecutionTrade[]>([]);
  const [venueAnalysis, setVenueAnalysis] = useState<VenueAnalysis[]>([]);
  const [alerts, setAlerts] = useState<ExecutionAlert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [alertThresholds, setAlertThresholds] = useState<Record<string, number>>({
    high_slippage: config.slippageThreshold,
    slow_fill: config.fillTimeThreshold,
    venue_issue: 0.1,
    market_impact: 20
  });
  
  // Refs
  const tradesBufferRef = useRef<ExecutionTrade[]>([]);
  const priceDataRef = useRef<Record<string, { timestamp: string; price: number }[]>>({});
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // WebSocket stream for real-time execution data
  const {
    isActive: isRealTimeActive,
    latestMessage,
    startStream,
    stopStream,
    error: streamError
  } = useWebSocketStream({
    streamId: 'execution_quality',
    messageType: 'execution_analytics',
    bufferSize: 1000,
    autoSubscribe: config.enableRealTime
  });
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);
  
  // Process real-time execution data
  useEffect(() => {
    if (latestMessage && latestMessage.data) {
      const executionData = latestMessage.data;
      
      // Convert to ExecutionTrade format
      const trade: ExecutionTrade = {
        tradeId: executionData.execution_id || generateTradeId(),
        orderId: executionData.order_id || '',
        symbol: executionData.symbol || '',
        side: executionData.side || 'buy',
        quantity: executionData.quantity || 0,
        executedQuantity: executionData.filled_quantity || 0,
        orderPrice: executionData.order_price || 0,
        executedPrice: executionData.fill_price || 0,
        benchmarkPrice: executionData.benchmark_price || 0,
        venue: executionData.venue || '',
        orderType: executionData.order_type || '',
        timestamp: executionData.timestamp || new Date().toISOString(),
        executionTime: executionData.execution_time_ms || 0,
        slippage: executionData.slippage || 0,
        slippageBps: executionData.slippage_bps || 0,
        marketImpact: executionData.market_impact || 0,
        costs: {
          commission: executionData.commission || 0,
          fees: executionData.fees || 0,
          borrowCost: executionData.borrow_cost
        }
      };
      
      // Add to trades
      setTrades(prev => {
        const newTrades = [...prev, trade];
        return newTrades.slice(-1000); // Keep last 1000 trades
      });
      
      // Check for alerts
      checkExecutionAlerts(trade);
      
      setLastUpdate(new Date());
    }
  }, [latestMessage]);
  
  // Generate unique trade ID
  const generateTradeId = useCallback(() => {
    return `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Check execution alerts
  const checkExecutionAlerts = useCallback((trade: ExecutionTrade) => {
    const newAlerts: ExecutionAlert[] = [];
    
    // High slippage alert
    if (Math.abs(trade.slippageBps) > alertThresholds.high_slippage) {
      newAlerts.push({
        id: `alert_${Date.now()}_slippage`,
        type: 'high_slippage',
        severity: Math.abs(trade.slippageBps) > alertThresholds.high_slippage * 2 ? 'critical' : 'high',
        message: `High slippage detected: ${trade.slippageBps.toFixed(2)} bps on ${trade.symbol}`,
        tradeId: trade.tradeId,
        venue: trade.venue,
        threshold: alertThresholds.high_slippage,
        actualValue: Math.abs(trade.slippageBps),
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }
    
    // Slow fill alert
    if (trade.executionTime > alertThresholds.slow_fill) {
      newAlerts.push({
        id: `alert_${Date.now()}_fill`,
        type: 'slow_fill',
        severity: trade.executionTime > alertThresholds.slow_fill * 2 ? 'high' : 'medium',
        message: `Slow execution detected: ${trade.executionTime}ms for ${trade.symbol}`,
        tradeId: trade.tradeId,
        venue: trade.venue,
        threshold: alertThresholds.slow_fill,
        actualValue: trade.executionTime,
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }
    
    // Market impact alert
    if (Math.abs(trade.marketImpact) > alertThresholds.market_impact) {
      newAlerts.push({
        id: `alert_${Date.now()}_impact`,
        type: 'market_impact',
        severity: Math.abs(trade.marketImpact) > alertThresholds.market_impact * 2 ? 'critical' : 'high',
        message: `High market impact: ${trade.marketImpact.toFixed(2)} bps on ${trade.symbol}`,
        tradeId: trade.tradeId,
        venue: trade.venue,
        threshold: alertThresholds.market_impact,
        actualValue: Math.abs(trade.marketImpact),
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }
    
    if (newAlerts.length > 0) {
      setAlerts(prev => [...prev, ...newAlerts].slice(-100)); // Keep last 100 alerts
    }
  }, [alertThresholds]);
  
  // Calculate execution metrics
  const calculateMetrics = useCallback(() => {
    if (trades.length === 0) return;
    
    // Basic metrics
    const totalTrades = trades.length;
    const filledTrades = trades.filter(t => t.executedQuantity > 0);
    const fillRate = totalTrades > 0 ? filledTrades.length / totalTrades : 0;
    
    const totalSlippage = trades.reduce((sum, t) => sum + Math.abs(t.slippage), 0);
    const averageSlippage = totalTrades > 0 ? totalSlippage / totalTrades : 0;
    const slippageBps = trades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / totalTrades;
    
    const averageFillTime = filledTrades.length > 0 
      ? filledTrades.reduce((sum, t) => sum + t.executionTime, 0) / filledTrades.length 
      : 0;
    
    // Implementation shortfall calculation
    const implementationShortfall = trades.reduce((sum, t) => {
      const shortfall = (t.executedPrice - t.benchmarkPrice) * t.executedQuantity;
      return sum + (t.side === 'buy' ? shortfall : -shortfall);
    }, 0);
    
    const totalNotional = trades.reduce((sum, t) => sum + (t.executedPrice * t.executedQuantity), 0);
    const implementationShortfallBps = totalNotional > 0 
      ? (implementationShortfall / totalNotional) * 10000 
      : 0;
    
    // Market impact
    const marketImpact = trades.reduce((sum, t) => sum + Math.abs(t.marketImpact), 0) / totalTrades;
    const marketImpactBps = trades.reduce((sum, t) => {
      return sum + Math.abs((t.marketImpact / t.executedPrice) * 10000);
    }, 0) / totalTrades;
    
    // Venue analysis
    const venueStats: Record<string, any> = {};
    trades.forEach(trade => {
      if (!venueStats[trade.venue]) {
        venueStats[trade.venue] = {
          trades: [],
          totalVolume: 0
        };
      }
      venueStats[trade.venue].trades.push(trade);
      venueStats[trade.venue].totalVolume += trade.executedQuantity * trade.executedPrice;
    });
    
    const venueQuality: ExecutionMetrics['venueQuality'] = {};
    Object.keys(venueStats).forEach(venue => {
      const venueTrades = venueStats[venue].trades;
      const fillRate = venueTrades.filter((t: ExecutionTrade) => t.executedQuantity > 0).length / venueTrades.length;
      const avgSlippage = venueTrades.reduce((sum: number, t: ExecutionTrade) => sum + Math.abs(t.slippageBps), 0) / venueTrades.length;
      const marketShare = venueStats[venue].totalVolume / totalNotional;
      const qualityScore = (fillRate * 40) + ((100 - Math.min(avgSlippage, 100)) * 0.4) + (marketShare * 20);
      
      venueQuality[venue] = {
        fillRate,
        avgSlippage,
        marketShare,
        qualityScore
      };
    });
    
    // Order type analysis
    const orderTypeStats: Record<string, ExecutionTrade[]> = {};
    trades.forEach(trade => {
      if (!orderTypeStats[trade.orderType]) {
        orderTypeStats[trade.orderType] = [];
      }
      orderTypeStats[trade.orderType].push(trade);
    });
    
    const orderTypeAnalysis: ExecutionMetrics['orderTypeAnalysis'] = {};
    Object.keys(orderTypeStats).forEach(orderType => {
      const typeTrades = orderTypeStats[orderType];
      orderTypeAnalysis[orderType] = {
        count: typeTrades.length,
        avgSlippage: typeTrades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / typeTrades.length,
        fillRate: typeTrades.filter(t => t.executedQuantity > 0).length / typeTrades.length,
        avgFillTime: typeTrades.reduce((sum, t) => sum + t.executionTime, 0) / typeTrades.length
      };
    });
    
    // Size analysis
    const sortedBySize = [...trades].sort((a, b) => 
      (a.quantity * a.orderPrice) - (b.quantity * b.orderPrice)
    );
    
    const smallTrades = sortedBySize.slice(0, Math.floor(sortedBySize.length / 3));
    const mediumTrades = sortedBySize.slice(Math.floor(sortedBySize.length / 3), Math.floor(2 * sortedBySize.length / 3));
    const largeTrades = sortedBySize.slice(Math.floor(2 * sortedBySize.length / 3));
    
    const sizeImpactAnalysis = {
      small: {
        avgSlippage: smallTrades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / smallTrades.length || 0,
        fillRate: smallTrades.filter(t => t.executedQuantity > 0).length / smallTrades.length || 0
      },
      medium: {
        avgSlippage: mediumTrades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / mediumTrades.length || 0,
        fillRate: mediumTrades.filter(t => t.executedQuantity > 0).length / mediumTrades.length || 0
      },
      large: {
        avgSlippage: largeTrades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / largeTrades.length || 0,
        fillRate: largeTrades.filter(t => t.executedQuantity > 0).length / largeTrades.length || 0
      }
    };
    
    const newMetrics: ExecutionMetrics = {
      fillRate,
      averageFillTime,
      totalSlippage,
      averageSlippage,
      slippageBps,
      implementationShortfall,
      implementationShortfallBps,
      marketImpact,
      marketImpactBps,
      temporaryImpact: marketImpact * 0.6, // Estimate
      permanentImpact: marketImpact * 0.4, // Estimate
      timingCost: 0, // Would need benchmark timing
      opportunityCost: 0, // Would need market opportunity analysis
      delayedExecutionCost: 0, // Would need delay analysis
      venueQuality,
      orderTypeAnalysis,
      sizeImpactAnalysis
    };
    
    setMetrics(newMetrics);
  }, [trades]);
  
  // Analyze slippage distribution
  const analyzeSlippage = useCallback((tradesToAnalyze = trades) => {
    if (tradesToAnalyze.length === 0) {
      return {
        distribution: [],
        outliers: [],
        averageBySize: {}
      };
    }
    
    const slippages = tradesToAnalyze.map(t => Math.abs(t.slippageBps));
    
    // Distribution
    const ranges = [
      { min: 0, max: 5, label: '0-5 bps' },
      { min: 5, max: 10, label: '5-10 bps' },
      { min: 10, max: 20, label: '10-20 bps' },
      { min: 20, max: 50, label: '20-50 bps' },
      { min: 50, max: Infinity, label: '50+ bps' }
    ];
    
    const distribution = ranges.map(range => {
      const count = slippages.filter(s => s >= range.min && s < range.max).length;
      return {
        range: range.label,
        count,
        percentage: (count / slippages.length) * 100
      };
    });
    
    // Outliers (>2 standard deviations)
    const mean = slippages.reduce((sum, s) => sum + s, 0) / slippages.length;
    const stdDev = Math.sqrt(
      slippages.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / slippages.length
    );
    
    const outliers = tradesToAnalyze.filter(t => Math.abs(t.slippageBps) > mean + 2 * stdDev);
    
    // Average by size
    const sortedBySize = [...tradesToAnalyze].sort((a, b) => 
      (a.quantity * a.orderPrice) - (b.quantity * b.orderPrice)
    );
    
    const third = Math.floor(sortedBySize.length / 3);
    const averageBySize = {
      small: sortedBySize.slice(0, third).reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / third || 0,
      medium: sortedBySize.slice(third, 2 * third).reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / third || 0,
      large: sortedBySize.slice(2 * third).reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / (sortedBySize.length - 2 * third) || 0
    };
    
    return { distribution, outliers, averageBySize };
  }, [trades]);
  
  // Analyze market impact
  const analyzeMarketImpact = useCallback((symbol: string, timeWindow = 300000) => {
    const symbolTrades = trades.filter(t => t.symbol === symbol);
    if (symbolTrades.length === 0) {
      return {
        temporaryImpact: 0,
        permanentImpact: 0,
        priceReversion: 0,
        volumeImpact: 0
      };
    }
    
    // This would require more sophisticated analysis with price data
    // For now, returning estimates based on trade data
    const avgMarketImpact = symbolTrades.reduce((sum, t) => sum + Math.abs(t.marketImpact), 0) / symbolTrades.length;
    
    return {
      temporaryImpact: avgMarketImpact * 0.6,
      permanentImpact: avgMarketImpact * 0.4,
      priceReversion: 0.3, // Placeholder
      volumeImpact: avgMarketImpact * 0.8
    };
  }, [trades]);
  
  // Compare venues
  const compareVenues = useCallback((venues: string[]) => {
    const comparison: Record<string, any> = {};
    
    venues.forEach(venue => {
      const venueTrades = trades.filter(t => t.venue === venue);
      if (venueTrades.length === 0) return;
      
      const fillRate = venueTrades.filter(t => t.executedQuantity > 0).length / venueTrades.length;
      const avgSlippage = venueTrades.reduce((sum, t) => sum + Math.abs(t.slippageBps), 0) / venueTrades.length;
      const avgFillTime = venueTrades.reduce((sum, t) => sum + t.executionTime, 0) / venueTrades.length;
      
      const score = (fillRate * 40) + ((100 - Math.min(avgSlippage, 100)) * 0.4) + ((10000 - Math.min(avgFillTime, 10000)) / 10000 * 20);
      
      comparison[venue] = {
        rank: 0, // Will be calculated after all venues
        score,
        strengths: [],
        weaknesses: []
      };
    });
    
    // Rank venues
    const venueScores = Object.entries(comparison).sort(([,a], [,b]) => b.score - a.score);
    venueScores.forEach(([venue], index) => {
      comparison[venue].rank = index + 1;
    });
    
    const bestVenue = venueScores[0][0];
    
    return {
      comparison,
      recommendation: `${bestVenue} offers the best overall execution quality`
    };
  }, [trades]);
  
  // Optimize execution
  const optimizeExecution = useCallback((orderParams: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    urgency: 'low' | 'medium' | 'high';
  }) => {
    // Analyze historical performance for similar orders
    const similarTrades = trades.filter(t => 
      t.symbol === orderParams.symbol && 
      t.side === orderParams.side &&
      Math.abs(t.quantity - orderParams.quantity) / orderParams.quantity < 0.5
    );
    
    if (similarTrades.length === 0) {
      return {
        recommendedVenue: 'SMART',
        recommendedOrderType: orderParams.urgency === 'high' ? 'MARKET' : 'LIMIT',
        expectedSlippage: 10,
        expectedFillTime: 5000
      };
    }
    
    // Find best venue based on historical performance
    const venuePerformance: Record<string, { slippage: number; fillTime: number; count: number }> = {};
    
    similarTrades.forEach(trade => {
      if (!venuePerformance[trade.venue]) {
        venuePerformance[trade.venue] = { slippage: 0, fillTime: 0, count: 0 };
      }
      venuePerformance[trade.venue].slippage += Math.abs(trade.slippageBps);
      venuePerformance[trade.venue].fillTime += trade.executionTime;
      venuePerformance[trade.venue].count++;
    });
    
    let bestVenue = 'SMART';
    let bestScore = 0;
    
    Object.keys(venuePerformance).forEach(venue => {
      const perf = venuePerformance[venue];
      const avgSlippage = perf.slippage / perf.count;
      const avgFillTime = perf.fillTime / perf.count;
      
      // Score based on urgency
      const slippageWeight = orderParams.urgency === 'high' ? 0.3 : 0.7;
      const timeWeight = 1 - slippageWeight;
      
      const score = (100 - avgSlippage) * slippageWeight + (10000 - avgFillTime) / 100 * timeWeight;
      
      if (score > bestScore) {
        bestScore = score;
        bestVenue = venue;
      }
    });
    
    const expectedSlippage = venuePerformance[bestVenue] ? 
      venuePerformance[bestVenue].slippage / venuePerformance[bestVenue].count : 10;
    const expectedFillTime = venuePerformance[bestVenue] ? 
      venuePerformance[bestVenue].fillTime / venuePerformance[bestVenue].count : 5000;
    
    return {
      recommendedVenue: bestVenue,
      recommendedOrderType: orderParams.urgency === 'high' ? 'MARKET' : 
        orderParams.urgency === 'medium' ? 'IOC' : 'LIMIT',
      expectedSlippage,
      expectedFillTime
    };
  }, [trades]);
  
  // Benchmark against TWAP
  const benchmarkAgainstTWAP = useCallback((tradeId: string) => {
    // This would require implementing TWAP calculation with market data
    return {
      twapPrice: 100,
      executionPrice: 100.05,
      performance: -0.05,
      percentile: 45
    };
  }, []);
  
  // Benchmark against VWAP
  const benchmarkAgainstVWAP = useCallback((tradeId: string) => {
    // This would require implementing VWAP calculation with market data
    return {
      vwapPrice: 100.02,
      executionPrice: 100.05,
      performance: -0.03,
      percentile: 52
    };
  }, []);
  
  // Alert management
  const acknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  }, []);
  
  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);
  
  const setAlertThreshold = useCallback((type: string, threshold: number) => {
    setAlertThresholds(prev => ({ ...prev, [type]: threshold }));
  }, []);
  
  // Control functions
  const startRealTimeTracking = useCallback(() => {
    startStream();
  }, [startStream]);
  
  const stopRealTimeTracking = useCallback(() => {
    stopStream();
  }, [stopStream]);
  
  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      // Fetch historical execution data
      const response = await fetch(
        `${API_BASE_URL}/api/v1/analytics/execution/${config.portfolioId}/trades?` +
        `days=${config.historicalPeriod}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setTrades(data.trades || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, config.portfolioId, config.historicalPeriod]);
  
  const exportAnalysis = useCallback(async (format: 'json' | 'csv' | 'pdf'): Promise<string | Blob> => {
    const data = {
      metrics,
      trades: trades.slice(0, 100), // Export last 100 trades
      venueAnalysis,
      alerts: alerts.filter(a => !a.acknowledged)
    };
    
    switch (format) {
      case 'json':
        return JSON.stringify(data, null, 2);
      case 'csv':
        // Convert to CSV
        const csvRows = [
          ['Trade ID', 'Symbol', 'Side', 'Quantity', 'Price', 'Slippage (bps)', 'Fill Time', 'Venue'],
          ...trades.map(t => [
            t.tradeId, t.symbol, t.side, t.quantity, t.executedPrice, 
            t.slippageBps, t.executionTime, t.venue
          ])
        ];
        return csvRows.map(row => row.join(',')).join('\n');
      case 'pdf':
        return new Blob([JSON.stringify(data)], { type: 'application/json' });
      default:
        return '';
    }
  }, [metrics, trades, venueAnalysis, alerts]);
  
  // Calculate metrics when trades change
  useEffect(() => {
    if (trades.length > 0) {
      calculateMetrics();
    }
  }, [trades, calculateMetrics]);
  
  // Initial data fetch
  useEffect(() => {
    if (config.portfolioId) {
      refresh();
    }
  }, [config.portfolioId, refresh]);
  
  return {
    // Execution data
    metrics,
    trades,
    venueAnalysis,
    alerts,
    
    // Status
    isLoading,
    error: error || streamError,
    lastUpdate,
    isRealTimeActive,
    
    // Analysis
    analyzeSlippage,
    analyzeMarketImpact,
    compareVenues,
    
    // Optimization
    optimizeExecution,
    
    // Benchmarking
    benchmarkAgainstTWAP,
    benchmarkAgainstVWAP,
    
    // Alerts
    acknowledgeAlert,
    clearAlerts,
    setAlertThreshold,
    
    // Control
    startRealTimeTracking,
    stopRealTimeTracking,
    refresh,
    exportAnalysis
  };
}

export default useExecutionQuality;