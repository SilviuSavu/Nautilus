/**
 * useExecutionAnalytics Hook - Sprint 3 Integration
 * Trade execution quality analysis with slippage monitoring and market impact
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface ExecutionAnalyticsRequest {
  portfolio_id?: string;
  strategy_id?: string;
  start_date?: string;
  end_date?: string;
  symbol?: string;
  order_types?: string[];
  venue?: string;
  include_market_impact?: boolean;
  include_timing_analysis?: boolean;
  benchmark_type?: 'arrival' | 'vwap' | 'twap' | 'close';
}

export interface ExecutionAnalyticsResult {
  summary_metrics: {
    total_trades: number;
    total_volume: number;
    total_notional: number;
    avg_trade_size: number;
    fill_rate: number;
    partial_fill_rate: number;
    cancel_rate: number;
    avg_execution_time: number; // seconds
  };
  
  slippage_analysis: {
    implementation_shortfall: {
      total_is: number;
      market_impact: number;
      timing_risk: number;
      delay_cost: number;
      opportunity_cost: number;
    };
    
    price_impact: {
      temporary_impact: number;
      permanent_impact: number;
      total_impact: number;
      impact_per_participation: number;
    };
    
    slippage_distribution: {
      bucket_range: [number, number];
      trade_count: number;
      volume_percentage: number;
      avg_slippage: number;
    }[];
    
    benchmark_comparison: {
      arrival_price_slippage: number;
      vwap_slippage: number;
      twap_slippage: number;
      close_price_slippage: number;
    };
  };
  
  timing_analysis: {
    time_to_fill: {
      mean: number;
      median: number;
      percentile_75: number;
      percentile_95: number;
      std_dev: number;
    };
    
    intraday_patterns: {
      hour: number;
      avg_slippage: number;
      avg_market_impact: number;
      trade_count: number;
      volume_participation: number;
    }[];
    
    execution_efficiency: {
      optimal_timing_score: number;
      market_timing_alpha: number;
      execution_alpha: number;
    };
  };
  
  market_microstructure: {
    bid_ask_spread_analysis: {
      avg_spread: number;
      spread_at_execution: number;
      spread_capture_ratio: number;
    };
    
    order_book_impact: {
      level_1_impact: number;
      level_2_impact: number;
      level_3_impact: number;
      recovery_time: number; // seconds
    };
    
    liquidity_metrics: {
      participation_rate: number;
      market_share: number;
      volume_weighted_spread: number;
      effective_spread: number;
    };
  };
  
  venue_analysis: {
    venue_name: string;
    trade_count: number;
    volume_percentage: number;
    avg_slippage: number;
    avg_market_impact: number;
    fill_rate: number;
    avg_execution_time: number;
  }[];
  
  order_type_analysis: {
    order_type: string;
    trade_count: number;
    avg_slippage: number;
    fill_rate: number;
    avg_size: number;
    market_impact: number;
  }[];
  
  trade_details: {
    trade_id: string;
    timestamp: string;
    symbol: string;
    side: 'buy' | 'sell';
    order_type: string;
    quantity: number;
    filled_quantity: number;
    avg_price: number;
    benchmark_price: number;
    slippage: number;
    market_impact: number;
    execution_time: number;
    venue: string;
  }[];
  
  performance_attribution: {
    alpha_from_execution: number;
    execution_cost_as_pct_of_returns: number;
    execution_sharpe_impact: number;
    cost_adjusted_returns: number;
  };
  
  risk_metrics: {
    execution_risk: number;
    liquidity_risk_score: number;
    market_impact_risk: number;
    timing_risk: number;
  };
}

export interface UseExecutionAnalyticsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableRealTimeTracking?: boolean;
  alertThresholds?: {
    slippage_threshold?: number;
    fill_rate_threshold?: number;
    execution_time_threshold?: number;
  };
}

export interface UseExecutionAnalyticsReturn {
  // State
  result: ExecutionAnalyticsResult | null;
  isAnalyzing: boolean;
  error: string | null;
  lastAnalyzed: Date | null;
  analysisDuration: number;
  
  // Real-time tracking
  isTracking: boolean;
  currentTrade: ExecutionTrade | null;
  executionAlerts: ExecutionAlert[];
  
  // Actions
  analyzeExecution: (request: ExecutionAnalyticsRequest) => Promise<void>;
  analyzeExecutionAsync: (request: ExecutionAnalyticsRequest) => Promise<ExecutionAnalyticsResult>;
  startRealTimeTracking: (portfolioId: string) => void;
  stopRealTimeTracking: () => void;
  
  // Specific analysis functions
  analyzeSlippage: (
    portfolioId: string,
    timeframe: string,
    benchmark: 'arrival' | 'vwap' | 'twap'
  ) => Promise<{
    avg_slippage: number;
    slippage_std: number;
    worst_slippage: number;
    best_slippage: number;
    slippage_trend: number;
  }>;
  
  getMarketImpactAnalysis: (
    portfolioId: string,
    symbol?: string
  ) => Promise<{
    temporary_impact: number;
    permanent_impact: number;
    participation_rate: number;
    impact_coefficient: number;
    liquidity_score: number;
  }>;
  
  getExecutionAlpha: (
    portfolioId: string,
    benchmarkStrategy: string
  ) => Promise<{
    execution_alpha: number;
    alpha_annualized: number;
    alpha_t_stat: number;
    alpha_p_value: number;
    information_ratio: number;
  }>;
  
  // Optimization functions
  getOptimalExecutionStrategy: (
    symbol: string,
    quantity: number,
    urgency: 'low' | 'medium' | 'high'
  ) => Promise<{
    recommended_strategy: string;
    estimated_slippage: number;
    estimated_market_impact: number;
    execution_horizon: number;
    participation_rate: number;
  }>;
  
  // TCA (Transaction Cost Analysis)
  runTCA: (
    tradeIds: string[]
  ) => Promise<{
    total_cost: number;
    explicit_costs: number;
    implicit_costs: number;
    opportunity_cost: number;
    cost_breakdown: {
      commissions: number;
      fees: number;
      market_impact: number;
      timing_cost: number;
      slippage: number;
    };
  }>;
}

interface ExecutionTrade {
  trade_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  status: 'pending' | 'partial' | 'filled' | 'cancelled';
  slippage: number;
  market_impact: number;
}

interface ExecutionAlert {
  id: string;
  type: 'high_slippage' | 'poor_fill_rate' | 'slow_execution' | 'high_impact';
  severity: 'low' | 'medium' | 'high';
  message: string;
  timestamp: Date;
  trade_id: string;
  current_value: number;
  threshold_value: number;
}

export function useExecutionAnalytics(
  options: UseExecutionAnalyticsOptions = {}
): UseExecutionAnalyticsReturn {
  const { 
    autoRefresh = false, 
    refreshInterval = 30000, 
    enableRealTimeTracking = false,
    alertThresholds = {}
  } = options;
  
  // State
  const [result, setResult] = useState<ExecutionAnalyticsResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAnalyzed, setLastAnalyzed] = useState<Date | null>(null);
  const [analysisDuration, setAnalysisDuration] = useState(0);
  const [isTracking, setIsTracking] = useState(false);
  const [currentTrade, setCurrentTrade] = useState<ExecutionTrade | null>(null);
  const [executionAlerts, setExecutionAlerts] = useState<ExecutionAlert[]>([]);
  
  // Refs
  const lastRequestRef = useRef<ExecutionAnalyticsRequest | null>(null);
  const trackingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopRealTimeTracking();
    };
  }, []);
  
  // Main execution analysis function
  const analyzeExecution = useCallback(async (request: ExecutionAnalyticsRequest) => {
    const startTime = performance.now();
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/analytics/execution/analyze`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...request,
            include_market_impact: request.include_market_impact ?? true,
            include_timing_analysis: request.include_timing_analysis ?? true,
            benchmark_type: request.benchmark_type || 'arrival',
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const executionResult = await response.json();
      const endTime = performance.now();
      
      if (isMountedRef.current) {
        setResult(executionResult);
        setLastAnalyzed(new Date());
        setAnalysisDuration(endTime - startTime);
        lastRequestRef.current = request;
        
        // Check for execution alerts
        checkForExecutionAlerts(executionResult);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Execution analysis failed');
        setResult(null);
      }
    } finally {
      if (isMountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  }, [API_BASE_URL]);
  
  // Async analysis that returns the result
  const analyzeExecutionAsync = useCallback(async (request: ExecutionAnalyticsRequest): Promise<ExecutionAnalyticsResult> => {
    await analyzeExecution(request);
    
    if (error) {
      throw new Error(error);
    }
    
    if (!result) {
      throw new Error('Execution analysis failed to return result');
    }
    
    return result;
  }, [analyzeExecution, error, result]);
  
  // Check for execution alerts
  const checkForExecutionAlerts = useCallback((executionResult: ExecutionAnalyticsResult) => {
    const newAlerts: ExecutionAlert[] = [];
    const now = new Date();
    
    // Check slippage threshold
    if (alertThresholds.slippage_threshold) {
      const avgSlippage = Math.abs(executionResult.slippage_analysis.benchmark_comparison.arrival_price_slippage);
      if (avgSlippage > alertThresholds.slippage_threshold) {
        newAlerts.push({
          id: `slippage-alert-${now.getTime()}`,
          type: 'high_slippage',
          severity: avgSlippage > alertThresholds.slippage_threshold * 2 ? 'high' : 'medium',
          message: `High slippage detected: ${(avgSlippage * 100).toFixed(2)}%`,
          timestamp: now,
          trade_id: 'portfolio-wide',
          current_value: avgSlippage,
          threshold_value: alertThresholds.slippage_threshold,
        });
      }
    }
    
    // Check fill rate threshold
    if (alertThresholds.fill_rate_threshold) {
      const fillRate = executionResult.summary_metrics.fill_rate;
      if (fillRate < alertThresholds.fill_rate_threshold) {
        newAlerts.push({
          id: `fill-rate-alert-${now.getTime()}`,
          type: 'poor_fill_rate',
          severity: fillRate < alertThresholds.fill_rate_threshold * 0.8 ? 'high' : 'medium',
          message: `Low fill rate: ${(fillRate * 100).toFixed(1)}%`,
          timestamp: now,
          trade_id: 'portfolio-wide',
          current_value: fillRate,
          threshold_value: alertThresholds.fill_rate_threshold,
        });
      }
    }
    
    // Check execution time threshold
    if (alertThresholds.execution_time_threshold) {
      const avgExecTime = executionResult.summary_metrics.avg_execution_time;
      if (avgExecTime > alertThresholds.execution_time_threshold) {
        newAlerts.push({
          id: `exec-time-alert-${now.getTime()}`,
          type: 'slow_execution',
          severity: avgExecTime > alertThresholds.execution_time_threshold * 1.5 ? 'high' : 'medium',
          message: `Slow execution time: ${avgExecTime.toFixed(1)}s`,
          timestamp: now,
          trade_id: 'portfolio-wide',
          current_value: avgExecTime,
          threshold_value: alertThresholds.execution_time_threshold,
        });
      }
    }
    
    setExecutionAlerts(prev => [...prev, ...newAlerts]);
  }, [alertThresholds]);
  
  // Start real-time execution tracking
  const startRealTimeTracking = useCallback((portfolioId: string) => {
    if (isTracking) return;
    
    setIsTracking(true);
    
    if (enableRealTimeTracking) {
      // Set up WebSocket for real-time execution updates
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/execution/realtime/${portfolioId}`;
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onmessage = (event) => {
        try {
          const tradeUpdate = JSON.parse(event.data);
          if (isMountedRef.current) {
            setCurrentTrade(tradeUpdate);
            
            // Check if this trade triggers any alerts
            if (alertThresholds.slippage_threshold && 
                Math.abs(tradeUpdate.slippage) > alertThresholds.slippage_threshold) {
              setExecutionAlerts(prev => [...prev, {
                id: `trade-slippage-${tradeUpdate.trade_id}-${Date.now()}`,
                type: 'high_slippage',
                severity: 'high',
                message: `High slippage on trade ${tradeUpdate.trade_id}: ${(tradeUpdate.slippage * 100).toFixed(2)}%`,
                timestamp: new Date(),
                trade_id: tradeUpdate.trade_id,
                current_value: Math.abs(tradeUpdate.slippage),
                threshold_value: alertThresholds.slippage_threshold,
              }]);
            }
          }
        } catch (err) {
          console.error('Failed to parse execution update:', err);
        }
      };
      
      websocketRef.current.onerror = () => {
        console.error('Execution tracking WebSocket error');
        setIsTracking(false);
      };
    }
  }, [isTracking, enableRealTimeTracking, API_BASE_URL, alertThresholds]);
  
  // Stop real-time tracking
  const stopRealTimeTracking = useCallback(() => {
    setIsTracking(false);
    setCurrentTrade(null);
    
    if (trackingIntervalRef.current) {
      clearInterval(trackingIntervalRef.current);
      trackingIntervalRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  }, []);
  
  // Analyze slippage
  const analyzeSlippage = useCallback(async (
    portfolioId: string,
    timeframe: string,
    benchmark: 'arrival' | 'vwap' | 'twap'
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/execution/slippage`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          timeframe: timeframe,
          benchmark: benchmark,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get market impact analysis
  const getMarketImpactAnalysis = useCallback(async (
    portfolioId: string,
    symbol?: string
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/execution/market-impact`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          symbol: symbol,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get execution alpha
  const getExecutionAlpha = useCallback(async (
    portfolioId: string,
    benchmarkStrategy: string
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/execution/alpha`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          benchmark_strategy: benchmarkStrategy,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get optimal execution strategy
  const getOptimalExecutionStrategy = useCallback(async (
    symbol: string,
    quantity: number,
    urgency: 'low' | 'medium' | 'high'
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/execution/optimal-strategy`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: symbol,
          quantity: quantity,
          urgency: urgency,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Run TCA (Transaction Cost Analysis)
  const runTCA = useCallback(async (tradeIds: string[]) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/execution/tca`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          trade_ids: tradeIds,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  return {
    // State
    result,
    isAnalyzing,
    error,
    lastAnalyzed,
    analysisDuration,
    
    // Real-time tracking
    isTracking,
    currentTrade,
    executionAlerts,
    
    // Actions
    analyzeExecution,
    analyzeExecutionAsync,
    startRealTimeTracking,
    stopRealTimeTracking,
    
    // Specific analysis functions
    analyzeSlippage,
    getMarketImpactAnalysis,
    getExecutionAlpha,
    
    // Optimization functions
    getOptimalExecutionStrategy,
    
    // TCA
    runTCA,
  };
}

export default useExecutionAnalytics;