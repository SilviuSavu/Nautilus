/**
 * usePerformanceMetrics Hook - Story 5.1 Integration
 * React hook for fetching and managing performance analytics data
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { performanceMetricsService } from '../../services/analytics/PerformanceMetricsService';
import { 
  PerformanceAnalyticsResponse, 
  AnalyticsState,
  MonteCarloRequest,
  MonteCarloResponse,
  AttributionAnalysisResponse,
  StatisticalTestsResponse 
} from '../../types/analytics';

export interface UsePerformanceMetricsOptions {
  portfolioId: string;
  benchmark?: string;
  startDate?: string;
  endDate?: string;
  autoRefresh?: boolean;
  refreshInterval?: number; // milliseconds
}

export interface UsePerformanceMetricsReturn {
  // State
  analytics: AnalyticsState;
  monteCarlo: {
    loading: boolean;
    error: string | null;
    data: MonteCarloResponse | null;
  };
  attribution: {
    loading: boolean;
    error: string | null;
    data: AttributionAnalysisResponse | null;
  };
  statisticalTests: {
    loading: boolean;
    error: string | null;
    data: StatisticalTestsResponse | null;
  };
  
  // Actions
  refreshAnalytics: () => Promise<void>;
  runMonteCarloSimulation: (request: MonteCarloRequest) => Promise<void>;
  getAttributionAnalysis: (type?: 'sector' | 'style' | 'security' | 'factor', period?: string) => Promise<void>;
  getStatisticalTests: (testType?: string, significanceLevel?: number) => Promise<void>;
  
  // Computed values
  isLoading: boolean;
  hasError: boolean;
  isDataAvailable: boolean;
}

export function usePerformanceMetrics(options: UsePerformanceMetricsOptions): UsePerformanceMetricsReturn {
  // State management
  const [analytics, setAnalytics] = useState<AnalyticsState>({
    loading: false,
    error: null,
    data: null,
    lastUpdated: null,
  });

  const [monteCarlo, setMonteCarlo] = useState({
    loading: false,
    error: null,
    data: null,
  });

  const [attribution, setAttribution] = useState({
    loading: false,
    error: null,
    data: null,
  });

  const [statisticalTests, setStatisticalTests] = useState({
    loading: false,
    error: null,
    data: null,
  });

  // Refs for cleanup and intervals
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []);

  // Main analytics fetcher
  const refreshAnalytics = useCallback(async () => {
    if (!options.portfolioId || !performanceMetricsService.validatePortfolioId(options.portfolioId)) {
      setAnalytics(prev => ({
        ...prev,
        error: 'Invalid portfolio ID',
        loading: false,
      }));
      return;
    }

    setAnalytics(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await performanceMetricsService.getPerformanceAnalytics(
        options.portfolioId,
        options.startDate,
        options.endDate,
        options.benchmark || 'SPY'
      );

      if (isMountedRef.current) {
        setAnalytics({
          loading: false,
          error: null,
          data,
          lastUpdated: new Date(),
        });
      }
    } catch (error) {
      if (isMountedRef.current) {
        setAnalytics(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to fetch analytics',
        }));
      }
    }
  }, [options.portfolioId, options.startDate, options.endDate, options.benchmark]);

  // Monte Carlo simulation
  const runMonteCarloSimulation = useCallback(async (request: MonteCarloRequest) => {
    setMonteCarlo(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await performanceMetricsService.runMonteCarloSimulation(request);
      
      if (isMountedRef.current) {
        setMonteCarlo({
          loading: false,
          error: null,
          data,
        });
      }
    } catch (error) {
      if (isMountedRef.current) {
        setMonteCarlo(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Monte Carlo simulation failed',
        }));
      }
    }
  }, []);

  // Attribution analysis
  const getAttributionAnalysis = useCallback(async (
    type: 'sector' | 'style' | 'security' | 'factor' = 'sector',
    period: string = '3M'
  ) => {
    if (!options.portfolioId) return;

    setAttribution(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await performanceMetricsService.getAttributionAnalysis(
        options.portfolioId,
        type,
        period
      );

      if (isMountedRef.current) {
        setAttribution({
          loading: false,
          error: null,
          data,
        });
      }
    } catch (error) {
      if (isMountedRef.current) {
        setAttribution(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Attribution analysis failed',
        }));
      }
    }
  }, [options.portfolioId]);

  // Statistical tests
  const getStatisticalTests = useCallback(async (
    testType: string = 'sharpe',
    significanceLevel: number = 0.05
  ) => {
    if (!options.portfolioId) return;

    setStatisticalTests(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await performanceMetricsService.getStatisticalTests(
        options.portfolioId,
        testType,
        significanceLevel
      );

      if (isMountedRef.current) {
        setStatisticalTests({
          loading: false,
          error: null,
          data,
        });
      }
    } catch (error) {
      if (isMountedRef.current) {
        setStatisticalTests(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Statistical tests failed',
        }));
      }
    }
  }, [options.portfolioId]);

  // Initial load and auto-refresh setup
  useEffect(() => {
    if (options.portfolioId) {
      refreshAnalytics();
    }

    // Setup auto-refresh if enabled
    if (options.autoRefresh && options.refreshInterval) {
      refreshIntervalRef.current = setInterval(() => {
        refreshAnalytics();
      }, options.refreshInterval);
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [options.portfolioId, options.autoRefresh, options.refreshInterval, refreshAnalytics]);

  // Computed values
  const isLoading = analytics.loading || monteCarlo.loading || attribution.loading || statisticalTests.loading;
  const hasError = !!(analytics.error || monteCarlo.error || attribution.error || statisticalTests.error);
  const isDataAvailable = !!(analytics.data || monteCarlo.data || attribution.data || statisticalTests.data);

  return {
    // State
    analytics,
    monteCarlo,
    attribution,
    statisticalTests,
    
    // Actions
    refreshAnalytics,
    runMonteCarloSimulation,
    getAttributionAnalysis,
    getStatisticalTests,
    
    // Computed values
    isLoading,
    hasError,
    isDataAvailable,
  };
}

// Helper hook for benchmarks
export function useBenchmarks() {
  const [benchmarks, setBenchmarks] = useState<{
    loading: boolean;
    error: string | null;
    data: Array<{
      symbol: string;
      name: string;
      category: string;
      data_available_from: string;
    }>;
  }>({
    loading: false,
    error: null,
    data: [],
  });

  const fetchBenchmarks = useCallback(async () => {
    setBenchmarks(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await performanceMetricsService.getAvailableBenchmarks();
      setBenchmarks({
        loading: false,
        error: null,
        data: response.benchmarks,
      });
    } catch (error) {
      setBenchmarks(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch benchmarks',
      }));
    }
  }, []);

  useEffect(() => {
    fetchBenchmarks();
  }, [fetchBenchmarks]);

  return {
    benchmarks: benchmarks.data,
    loading: benchmarks.loading,
    error: benchmarks.error,
    refetch: fetchBenchmarks,
  };
}

export default usePerformanceMetrics;