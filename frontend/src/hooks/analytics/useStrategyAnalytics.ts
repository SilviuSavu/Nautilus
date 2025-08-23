/**
 * useStrategyAnalytics Hook - Sprint 3 Integration
 * Strategy performance analysis, comparison, and benchmarking
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface StrategyAnalyticsRequest {
  strategy_ids: string[];
  portfolio_id?: string;
  start_date?: string;
  end_date?: string;
  benchmark?: string;
  include_attribution?: boolean;
  include_factor_analysis?: boolean;
  include_regime_analysis?: boolean;
  comparison_type?: 'peer' | 'benchmark' | 'historical';
  rolling_window?: number;
}

export interface StrategyAnalyticsResult {
  strategy_summaries: {
    strategy_id: string;
    strategy_name: string;
    inception_date: string;
    total_return: number;
    annualized_return: number;
    volatility: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    current_aum: number;
    status: 'active' | 'paused' | 'stopped';
  }[];
  
  comparative_analysis: {
    performance_ranking: {
      strategy_id: string;
      rank: number;
      percentile: number;
      score: number;
    }[];
    
    risk_adjusted_ranking: {
      strategy_id: string;
      sharpe_rank: number;
      sortino_rank: number;
      calmar_rank: number;
      composite_rank: number;
    }[];
    
    correlation_matrix: {
      strategies: string[];
      correlations: number[][];
      avg_correlation: number;
      diversification_benefit: number;
    };
  };
  
  factor_analysis?: {
    strategy_id: string;
    factor_exposures: {
      market_beta: number;
      size_factor: number;
      value_factor: number;
      momentum_factor: number;
      quality_factor: number;
      volatility_factor: number;
      custom_factors: Record<string, number>;
    };
    
    factor_attribution: {
      factor_name: string;
      exposure: number;
      return_contribution: number;
      risk_contribution: number;
    }[];
    
    alpha_decomposition: {
      pure_alpha: number;
      factor_alpha: number;
      total_alpha: number;
      alpha_t_stat: number;
      alpha_p_value: number;
    };
  }[];
  
  regime_analysis?: {
    bull_market_performance: {
      strategy_id: string;
      return: number;
      volatility: number;
      sharpe_ratio: number;
      max_drawdown: number;
      periods_count: number;
    }[];
    
    bear_market_performance: {
      strategy_id: string;
      return: number;
      volatility: number;
      sharpe_ratio: number;
      max_drawdown: number;
      periods_count: number;
    }[];
    
    neutral_market_performance: {
      strategy_id: string;
      return: number;
      volatility: number;
      sharpe_ratio: number;
      max_drawdown: number;
      periods_count: number;
    }[];
    
    regime_detection: {
      current_regime: 'bull' | 'bear' | 'neutral';
      regime_probability: number;
      expected_duration: number;
      regime_history: {
        start_date: string;
        end_date: string;
        regime: string;
        confidence: number;
      }[];
    };
  };
  
  performance_attribution: {
    strategy_id: string;
    attribution_breakdown: {
      asset_allocation: number;
      security_selection: number;
      market_timing: number;
      currency_effect: number;
      interaction_effect: number;
    };
    
    sector_attribution: {
      sector: string;
      allocation_effect: number;
      selection_effect: number;
      total_effect: number;
      weight_difference: number;
    }[];
    
    time_period_attribution: {
      period: string;
      strategy_return: number;
      benchmark_return: number;
      excess_return: number;
      information_ratio: number;
    }[];
  }[];
  
  rolling_analytics: {
    date: string;
    strategy_returns: Record<string, number>;
    strategy_volatilities: Record<string, number>;
    strategy_sharpe_ratios: Record<string, number>;
    strategy_max_drawdowns: Record<string, number>;
    benchmark_return: number;
  }[];
  
  drawdown_analysis: {
    strategy_id: string;
    current_drawdown: number;
    max_drawdown: number;
    avg_drawdown: number;
    drawdown_duration: {
      current: number;
      max: number;
      avg: number;
    };
    
    recovery_analysis: {
      avg_recovery_time: number;
      recovery_rate: number;
      incomplete_recoveries: number;
    };
    
    underwater_curve: {
      date: string;
      drawdown: number;
    }[];
  }[];
  
  risk_decomposition: {
    strategy_id: string;
    total_risk: number;
    systematic_risk: number;
    idiosyncratic_risk: number;
    concentration_risk: number;
    tracking_error: number;
    
    risk_contributions: {
      asset: string;
      weight: number;
      contribution: number;
      marginal_contribution: number;
    }[];
  }[];
}

export interface UseStrategyAnalyticsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableLiveTracking?: boolean;
  benchmarkStrategy?: string;
  alertThresholds?: {
    drawdown_threshold?: number;
    correlation_threshold?: number;
    performance_threshold?: number;
  };
}

export interface UseStrategyAnalyticsReturn {
  // State
  result: StrategyAnalyticsResult | null;
  isAnalyzing: boolean;
  error: string | null;
  lastAnalyzed: Date | null;
  analysisDuration: number;
  
  // Live tracking
  isTracking: boolean;
  livePerformance: Record<string, StrategyLiveMetrics>;
  strategyAlerts: StrategyAlert[];
  
  // Actions
  analyzeStrategies: (request: StrategyAnalyticsRequest) => Promise<void>;
  analyzeStrategiesAsync: (request: StrategyAnalyticsRequest) => Promise<StrategyAnalyticsResult>;
  startLiveTracking: (strategyIds: string[]) => void;
  stopLiveTracking: () => void;
  
  // Comparison functions
  compareStrategies: (
    strategyIds: string[],
    metrics: ('return' | 'risk' | 'sharpe' | 'drawdown')[]
  ) => Promise<{
    comparison_results: {
      metric: string;
      rankings: { strategy_id: string; value: number; rank: number }[];
    }[];
    statistical_significance: {
      strategy_a: string;
      strategy_b: string;
      p_value: number;
      is_significant: boolean;
    }[];
  }>;
  
  // Peer analysis
  getPeerAnalysis: (
    strategyId: string,
    universe: string
  ) => Promise<{
    peer_ranking: number;
    percentile: number;
    peer_statistics: {
      metric: string;
      strategy_value: number;
      peer_median: number;
      peer_mean: number;
      peer_25th: number;
      peer_75th: number;
    }[];
  }>;
  
  // Performance forecasting
  forecastPerformance: (
    strategyId: string,
    horizon: number,
    scenarios: number
  ) => Promise<{
    expected_return: number;
    expected_volatility: number;
    return_distribution: {
      percentile: number;
      return_value: number;
    }[];
    scenario_analysis: {
      scenario: string;
      probability: number;
      expected_return: number;
      max_drawdown: number;
    }[];
  }>;
  
  // Style analysis
  getStyleAnalysis: (
    strategyId: string
  ) => Promise<{
    style_classification: string;
    style_consistency: number;
    style_drift: number;
    style_exposures: Record<string, number>;
    style_evolution: {
      date: string;
      style_weights: Record<string, number>;
    }[];
  }>;
}

interface StrategyLiveMetrics {
  strategy_id: string;
  current_return: number;
  current_drawdown: number;
  current_volatility: number;
  current_sharpe: number;
  last_update: string;
  status: 'active' | 'paused' | 'stopped';
}

interface StrategyAlert {
  id: string;
  type: 'drawdown' | 'underperformance' | 'high_correlation' | 'risk_breach';
  severity: 'low' | 'medium' | 'high';
  strategy_id: string;
  message: string;
  timestamp: Date;
  current_value: number;
  threshold_value: number;
}

export function useStrategyAnalytics(
  options: UseStrategyAnalyticsOptions = {}
): UseStrategyAnalyticsReturn {
  const { 
    autoRefresh = false, 
    refreshInterval = 60000, 
    enableLiveTracking = false,
    benchmarkStrategy = 'SPY',
    alertThresholds = {}
  } = options;
  
  // State
  const [result, setResult] = useState<StrategyAnalyticsResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAnalyzed, setLastAnalyzed] = useState<Date | null>(null);
  const [analysisDuration, setAnalysisDuration] = useState(0);
  const [isTracking, setIsTracking] = useState(false);
  const [livePerformance, setLivePerformance] = useState<Record<string, StrategyLiveMetrics>>({});
  const [strategyAlerts, setStrategyAlerts] = useState<StrategyAlert[]>([]);
  
  // Refs
  const lastRequestRef = useRef<StrategyAnalyticsRequest | null>(null);
  const trackingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopLiveTracking();
    };
  }, []);
  
  // Main strategy analysis function
  const analyzeStrategies = useCallback(async (request: StrategyAnalyticsRequest) => {
    if (!request.strategy_ids || request.strategy_ids.length === 0) {
      setError('At least one strategy ID is required');
      return;
    }
    
    const startTime = performance.now();
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/analytics/strategy/analyze`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...request,
            benchmark: request.benchmark || benchmarkStrategy,
            include_attribution: request.include_attribution ?? true,
            include_factor_analysis: request.include_factor_analysis ?? true,
            include_regime_analysis: request.include_regime_analysis ?? true,
            rolling_window: request.rolling_window || 252, // 1 year
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const strategyResult = await response.json();
      const endTime = performance.now();
      
      if (isMountedRef.current) {
        setResult(strategyResult);
        setLastAnalyzed(new Date());
        setAnalysisDuration(endTime - startTime);
        lastRequestRef.current = request;
        
        // Check for strategy alerts
        checkForStrategyAlerts(strategyResult);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Strategy analysis failed');
        setResult(null);
      }
    } finally {
      if (isMountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  }, [API_BASE_URL, benchmarkStrategy]);
  
  // Async analysis that returns the result
  const analyzeStrategiesAsync = useCallback(async (request: StrategyAnalyticsRequest): Promise<StrategyAnalyticsResult> => {
    await analyzeStrategies(request);
    
    if (error) {
      throw new Error(error);
    }
    
    if (!result) {
      throw new Error('Strategy analysis failed to return result');
    }
    
    return result;
  }, [analyzeStrategies, error, result]);
  
  // Check for strategy alerts
  const checkForStrategyAlerts = useCallback((strategyResult: StrategyAnalyticsResult) => {
    const newAlerts: StrategyAlert[] = [];
    const now = new Date();
    
    strategyResult.strategy_summaries.forEach(strategy => {
      // Check drawdown threshold
      if (alertThresholds.drawdown_threshold) {
        if (Math.abs(strategy.max_drawdown) > alertThresholds.drawdown_threshold) {
          newAlerts.push({
            id: `drawdown-${strategy.strategy_id}-${now.getTime()}`,
            type: 'drawdown',
            severity: Math.abs(strategy.max_drawdown) > alertThresholds.drawdown_threshold * 1.5 ? 'high' : 'medium',
            strategy_id: strategy.strategy_id,
            message: `High drawdown for ${strategy.strategy_name}: ${(strategy.max_drawdown * 100).toFixed(1)}%`,
            timestamp: now,
            current_value: Math.abs(strategy.max_drawdown),
            threshold_value: alertThresholds.drawdown_threshold,
          });
        }
      }
      
      // Check performance threshold
      if (alertThresholds.performance_threshold) {
        if (strategy.sharpe_ratio < alertThresholds.performance_threshold) {
          newAlerts.push({
            id: `performance-${strategy.strategy_id}-${now.getTime()}`,
            type: 'underperformance',
            severity: strategy.sharpe_ratio < alertThresholds.performance_threshold * 0.5 ? 'high' : 'medium',
            strategy_id: strategy.strategy_id,
            message: `Low Sharpe ratio for ${strategy.strategy_name}: ${strategy.sharpe_ratio.toFixed(2)}`,
            timestamp: now,
            current_value: strategy.sharpe_ratio,
            threshold_value: alertThresholds.performance_threshold,
          });
        }
      }
    });
    
    // Check correlation threshold
    if (alertThresholds.correlation_threshold && strategyResult.comparative_analysis.correlation_matrix.avg_correlation) {
      const avgCorrelation = strategyResult.comparative_analysis.correlation_matrix.avg_correlation;
      if (avgCorrelation > alertThresholds.correlation_threshold) {
        newAlerts.push({
          id: `correlation-${now.getTime()}`,
          type: 'high_correlation',
          severity: 'medium',
          strategy_id: 'portfolio',
          message: `High average strategy correlation: ${(avgCorrelation * 100).toFixed(1)}%`,
          timestamp: now,
          current_value: avgCorrelation,
          threshold_value: alertThresholds.correlation_threshold,
        });
      }
    }
    
    setStrategyAlerts(prev => [...prev, ...newAlerts]);
  }, [alertThresholds]);
  
  // Start live tracking
  const startLiveTracking = useCallback((strategyIds: string[]) => {
    if (isTracking) return;
    
    setIsTracking(true);
    
    if (enableLiveTracking) {
      // Set up WebSocket for live strategy updates
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/strategy/live`;
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onopen = () => {
        // Subscribe to strategy updates
        strategyIds.forEach(strategyId => {
          websocketRef.current?.send(JSON.stringify({
            action: 'subscribe',
            strategy_id: strategyId,
          }));
        });
      };
      
      websocketRef.current.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          if (isMountedRef.current) {
            setLivePerformance(prev => ({
              ...prev,
              [update.strategy_id]: update,
            }));
            
            // Check for alerts on live data
            if (alertThresholds.drawdown_threshold && 
                Math.abs(update.current_drawdown) > alertThresholds.drawdown_threshold) {
              setStrategyAlerts(prev => [...prev, {
                id: `live-drawdown-${update.strategy_id}-${Date.now()}`,
                type: 'drawdown',
                severity: 'high',
                strategy_id: update.strategy_id,
                message: `Live drawdown alert: ${(update.current_drawdown * 100).toFixed(1)}%`,
                timestamp: new Date(),
                current_value: Math.abs(update.current_drawdown),
                threshold_value: alertThresholds.drawdown_threshold,
              }]);
            }
          }
        } catch (err) {
          console.error('Failed to parse strategy update:', err);
        }
      };
      
      websocketRef.current.onerror = () => {
        console.error('Strategy tracking WebSocket error');
        setIsTracking(false);
      };
    }
  }, [isTracking, enableLiveTracking, API_BASE_URL, alertThresholds]);
  
  // Stop live tracking
  const stopLiveTracking = useCallback(() => {
    setIsTracking(false);
    setLivePerformance({});
    
    if (trackingIntervalRef.current) {
      clearInterval(trackingIntervalRef.current);
      trackingIntervalRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  }, []);
  
  // Compare strategies
  const compareStrategies = useCallback(async (
    strategyIds: string[],
    metrics: ('return' | 'risk' | 'sharpe' | 'drawdown')[]
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/strategy/compare`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_ids: strategyIds,
          metrics: metrics,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get peer analysis
  const getPeerAnalysis = useCallback(async (
    strategyId: string,
    universe: string
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/strategy/peer-analysis`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_id: strategyId,
          universe: universe,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Forecast performance
  const forecastPerformance = useCallback(async (
    strategyId: string,
    horizon: number,
    scenarios: number
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/strategy/forecast`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_id: strategyId,
          horizon_days: horizon,
          scenarios: scenarios,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get style analysis
  const getStyleAnalysis = useCallback(async (strategyId: string) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/strategy/style-analysis`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_id: strategyId,
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
    
    // Live tracking
    isTracking,
    livePerformance,
    strategyAlerts,
    
    // Actions
    analyzeStrategies,
    analyzeStrategiesAsync,
    startLiveTracking,
    stopLiveTracking,
    
    // Comparison functions
    compareStrategies,
    
    // Analysis functions
    getPeerAnalysis,
    forecastPerformance,
    getStyleAnalysis,
  };
}

export default useStrategyAnalytics;