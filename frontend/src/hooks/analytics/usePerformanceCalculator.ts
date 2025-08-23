/**
 * usePerformanceCalculator Hook - Sprint 3 Integration
 * Advanced performance calculations with real-time updates
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface PerformanceCalculationRequest {
  portfolio_id: string;
  start_date?: string;
  end_date?: string;
  benchmark?: string;
  frequency?: 'daily' | 'weekly' | 'monthly';
  include_rolling?: boolean;
  rolling_window?: number;
  risk_free_rate?: number;
}

export interface PerformanceCalculationResult {
  basic_metrics: {
    total_return: number;
    annualized_return: number;
    annualized_volatility: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    information_ratio: number;
    tracking_error: number;
    max_drawdown: number;
    max_drawdown_duration: number;
  };
  risk_adjusted_metrics: {
    alpha: number;
    beta: number;
    treynor_ratio: number;
    jensen_alpha: number;
    batting_average: number;
    up_capture: number;
    down_capture: number;
    tail_ratio: number;
  };
  advanced_metrics: {
    var_95: number;
    var_99: number;
    expected_shortfall_95: number;
    expected_shortfall_99: number;
    omega_ratio: number;
    kappa_3: number;
    gain_to_pain_ratio: number;
    sterling_ratio: number;
  };
  rolling_metrics?: {
    date: string;
    return: number;
    volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
    beta: number;
    alpha: number;
  }[];
  benchmark_comparison: {
    benchmark_return: number;
    benchmark_volatility: number;
    benchmark_sharpe: number;
    relative_return: number;
    correlation: number;
    r_squared: number;
  };
  attribution: {
    allocation_effect: number;
    selection_effect: number;
    interaction_effect: number;
    total_effect: number;
  };
  periods: {
    period: string;
    portfolio_return: number;
    benchmark_return: number;
    excess_return: number;
    information_ratio: number;
  }[];
}

export interface UsePerformanceCalculatorOptions {
  autoCalculate?: boolean;
  cacheResults?: boolean;
  enableRealTime?: boolean;
  updateInterval?: number;
}

export interface UsePerformanceCalculatorReturn {
  // State
  result: PerformanceCalculationResult | null;
  isCalculating: boolean;
  error: string | null;
  lastCalculated: Date | null;
  calculationDuration: number;
  
  // Actions
  calculate: (request: PerformanceCalculationRequest) => Promise<void>;
  calculateAsync: (request: PerformanceCalculationRequest) => Promise<PerformanceCalculationResult>;
  recalculate: () => Promise<void>;
  clear: () => void;
  
  // Utilities
  comparePerformance: (
    portfolioA: string,
    portfolioB: string,
    options?: Partial<PerformanceCalculationRequest>
  ) => Promise<{
    portfolio_a: PerformanceCalculationResult;
    portfolio_b: PerformanceCalculationResult;
    comparison: {
      better_sharpe: string;
      better_return: string;
      better_volatility: string;
      better_max_drawdown: string;
    };
  }>;
  
  // Stress testing
  stressTesting: (
    request: PerformanceCalculationRequest & {
      stress_scenarios: Array<{
        name: string;
        market_shock: number;
        volatility_spike: number;
        correlation_increase: number;
      }>;
    }
  ) => Promise<{
    base_case: PerformanceCalculationResult;
    stress_results: Array<{
      scenario_name: string;
      stressed_metrics: PerformanceCalculationResult;
      impact_summary: {
        return_impact: number;
        volatility_impact: number;
        sharpe_impact: number;
        max_drawdown_impact: number;
      };
    }>;
  }>;
  
  // Performance attribution
  getDetailedAttribution: (
    request: PerformanceCalculationRequest & {
      attribution_type: 'sector' | 'security' | 'factor' | 'style';
      benchmark_weights?: Record<string, number>;
    }
  ) => Promise<{
    total_active_return: number;
    allocation_effect: number;
    selection_effect: number;
    interaction_effect: number;
    currency_effect?: number;
    breakdown: Array<{
      name: string;
      portfolio_weight: number;
      benchmark_weight: number;
      portfolio_return: number;
      benchmark_return: number;
      allocation_contribution: number;
      selection_contribution: number;
      total_contribution: number;
    }>;
  }>;
}

export function usePerformanceCalculator(
  options: UsePerformanceCalculatorOptions = {}
): UsePerformanceCalculatorReturn {
  const { autoCalculate = false, cacheResults = true, enableRealTime = false, updateInterval = 30000 } = options;
  
  // State
  const [result, setResult] = useState<PerformanceCalculationResult | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastCalculated, setLastCalculated] = useState<Date | null>(null);
  const [calculationDuration, setCalculationDuration] = useState(0);
  
  // Refs
  const lastRequestRef = useRef<PerformanceCalculationRequest | null>(null);
  const cacheRef = useRef<Map<string, { result: PerformanceCalculationResult; timestamp: number }>>(new Map());
  const isMountedRef = useRef(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);
  
  // Generate cache key
  const getCacheKey = useCallback((request: PerformanceCalculationRequest): string => {
    return JSON.stringify({
      portfolio_id: request.portfolio_id,
      start_date: request.start_date,
      end_date: request.end_date,
      benchmark: request.benchmark,
      frequency: request.frequency,
      rolling_window: request.rolling_window,
    });
  }, []);
  
  // Check cache
  const getFromCache = useCallback((request: PerformanceCalculationRequest): PerformanceCalculationResult | null => {
    if (!cacheResults) return null;
    
    const key = getCacheKey(request);
    const cached = cacheRef.current.get(key);
    
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
      return cached.result;
    }
    
    return null;
  }, [cacheResults, getCacheKey]);
  
  // Save to cache
  const saveToCache = useCallback((request: PerformanceCalculationRequest, result: PerformanceCalculationResult) => {
    if (!cacheResults) return;
    
    const key = getCacheKey(request);
    cacheRef.current.set(key, {
      result,
      timestamp: Date.now(),
    });
    
    // Cleanup old cache entries
    if (cacheRef.current.size > 100) {
      const entries = Array.from(cacheRef.current.entries());
      entries.sort((a, b) => b[1].timestamp - a[1].timestamp);
      cacheRef.current.clear();
      entries.slice(0, 50).forEach(([key, value]) => {
        cacheRef.current.set(key, value);
      });
    }
  }, [cacheResults, getCacheKey]);
  
  // Main calculation function
  const calculate = useCallback(async (request: PerformanceCalculationRequest) => {
    if (!request.portfolio_id) {
      setError('Portfolio ID is required');
      return;
    }
    
    // Check cache first
    const cachedResult = getFromCache(request);
    if (cachedResult) {
      setResult(cachedResult);
      setLastCalculated(new Date());
      setError(null);
      return;
    }
    
    const startTime = performance.now();
    setIsCalculating(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/analytics/performance/analyze`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...request,
            calculate_rolling: request.include_rolling ?? true,
            rolling_window_days: request.rolling_window ?? 30,
            risk_free_rate: request.risk_free_rate ?? 0.02,
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const calculationResult = await response.json();
      const endTime = performance.now();
      
      if (isMountedRef.current) {
        setResult(calculationResult);
        setLastCalculated(new Date());
        setCalculationDuration(endTime - startTime);
        lastRequestRef.current = request;
        
        // Save to cache
        saveToCache(request, calculationResult);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Calculation failed');
        setResult(null);
      }
    } finally {
      if (isMountedRef.current) {
        setIsCalculating(false);
      }
    }
  }, [API_BASE_URL, getFromCache, saveToCache]);
  
  // Async calculation that returns the result
  const calculateAsync = useCallback(async (request: PerformanceCalculationRequest): Promise<PerformanceCalculationResult> => {
    await calculate(request);
    
    if (error) {
      throw new Error(error);
    }
    
    if (!result) {
      throw new Error('Calculation failed to return result');
    }
    
    return result;
  }, [calculate, error, result]);
  
  // Recalculate using last request
  const recalculate = useCallback(async () => {
    if (lastRequestRef.current) {
      // Clear cache for this request to force recalculation
      const key = getCacheKey(lastRequestRef.current);
      cacheRef.current.delete(key);
      await calculate(lastRequestRef.current);
    }
  }, [calculate, getCacheKey]);
  
  // Clear results
  const clear = useCallback(() => {
    setResult(null);
    setError(null);
    setLastCalculated(null);
    setCalculationDuration(0);
    lastRequestRef.current = null;
  }, []);
  
  // Compare performance between two portfolios
  const comparePerformance = useCallback(async (
    portfolioA: string,
    portfolioB: string,
    options: Partial<PerformanceCalculationRequest> = {}
  ) => {
    const baseRequest = {
      start_date: options.start_date,
      end_date: options.end_date,
      benchmark: options.benchmark || 'SPY',
      frequency: options.frequency || 'daily' as const,
      include_rolling: options.include_rolling ?? true,
      rolling_window: options.rolling_window ?? 30,
      risk_free_rate: options.risk_free_rate ?? 0.02,
    };
    
    const [resultA, resultB] = await Promise.all([
      calculateAsync({ ...baseRequest, portfolio_id: portfolioA }),
      calculateAsync({ ...baseRequest, portfolio_id: portfolioB }),
    ]);
    
    return {
      portfolio_a: resultA,
      portfolio_b: resultB,
      comparison: {
        better_sharpe: resultA.basic_metrics.sharpe_ratio > resultB.basic_metrics.sharpe_ratio ? portfolioA : portfolioB,
        better_return: resultA.basic_metrics.total_return > resultB.basic_metrics.total_return ? portfolioA : portfolioB,
        better_volatility: resultA.basic_metrics.annualized_volatility < resultB.basic_metrics.annualized_volatility ? portfolioA : portfolioB,
        better_max_drawdown: resultA.basic_metrics.max_drawdown > resultB.basic_metrics.max_drawdown ? portfolioB : portfolioA,
      },
    };
  }, [calculateAsync]);
  
  // Stress testing
  const stressTesting = useCallback(async (
    request: PerformanceCalculationRequest & {
      stress_scenarios: Array<{
        name: string;
        market_shock: number;
        volatility_spike: number;
        correlation_increase: number;
      }>;
    }
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/risk/analyze`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: request.portfolio_id,
          stress_scenarios: request.stress_scenarios,
          include_performance_impact: true,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Detailed attribution analysis
  const getDetailedAttribution = useCallback(async (
    request: PerformanceCalculationRequest & {
      attribution_type: 'sector' | 'security' | 'factor' | 'style';
      benchmark_weights?: Record<string, number>;
    }
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/performance/attribution`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Real-time updates
  useEffect(() => {
    if (enableRealTime && lastRequestRef.current && updateInterval > 0) {
      intervalRef.current = setInterval(() => {
        if (lastRequestRef.current) {
          calculate(lastRequestRef.current);
        }
      }, updateInterval);
      
      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [enableRealTime, updateInterval, calculate]);
  
  return {
    // State
    result,
    isCalculating,
    error,
    lastCalculated,
    calculationDuration,
    
    // Actions
    calculate,
    calculateAsync,
    recalculate,
    clear,
    
    // Utilities
    comparePerformance,
    
    // Advanced features
    stressTesting,
    getDetailedAttribution,
  };
}

export default usePerformanceCalculator;