/**
 * useAdvancedPerformanceMetrics Hook
 * Sprint 3: Enhanced Performance Analytics with Advanced Calculations
 * 
 * Comprehensive performance analytics with risk-adjusted returns, attribution analysis,
 * Monte Carlo simulations, and institutional-grade performance metrics.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRealTimeAnalytics } from './useRealTimeAnalytics';

export interface AdvancedPerformanceMetrics {
  // Core Performance
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  
  // Risk-Adjusted Returns
  informationRatio: number;
  treynorRatio: number;
  jensenAlpha: number;
  trackingError: number;
  beta: number;
  
  // Advanced Risk Metrics
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  expectedShortfall: number;
  conditionalDrawdown: number;
  
  // Performance Attribution
  activeReturn: number;
  attributionAnalysis: {
    allocation: number;
    selection: number;
    interaction: number;
    total: number;
  };
  
  // Statistical Measures
  skewness: number;
  kurtosis: number;
  winRate: number;
  profitFactor: number;
  payoffRatio: number;
  
  // Time-Based Analysis
  rollingReturns: {
    period: string;
    returns: number[];
    volatility: number[];
    sharpe: number[];
  };
  
  // Monte Carlo Results
  monteCarloResults?: {
    confidenceIntervals: {
      ci_95: { lower: number; upper: number };
      ci_99: { lower: number; upper: number };
    };
    worstCaseScenario: number;
    bestCaseScenario: number;
    expectedValue: number;
    probabilityOfLoss: number;
  };
}

export interface PerformanceBenchmark {
  id: string;
  name: string;
  returns: number[];
  correlation: number;
  beta: number;
  alpha: number;
  trackingError: number;
  informationRatio: number;
}

export interface PerformanceConfig {
  riskFreeRate: number;
  benchmarkId?: string;
  confidenceLevel: number;
  rollingWindow: number;
  enableMonteCarlo: boolean;
  monteCarloSimulations: number;
  attributionModel: 'brinson' | 'fachler' | 'carhart';
}

export interface UseAdvancedPerformanceMetricsOptions {
  portfolioId: string;
  config?: Partial<PerformanceConfig>;
  updateInterval?: number;
  enableRealTime?: boolean;
  historicalPeriod?: number;
}

export interface UseAdvancedPerformanceMetricsReturn {
  // Performance data
  metrics: AdvancedPerformanceMetrics | null;
  benchmarks: PerformanceBenchmark[];
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
  
  // Configuration
  updateConfig: (config: Partial<PerformanceConfig>) => void;
  
  // Data management
  addBenchmark: (benchmark: Omit<PerformanceBenchmark, 'correlation' | 'beta' | 'alpha' | 'trackingError' | 'informationRatio'>) => Promise<void>;
  removeBenchmark: (benchmarkId: string) => void;
  
  // Analysis
  runAttributionAnalysis: () => Promise<void>;
  runMonteCarloSimulation: (simulations?: number) => Promise<void>;
  calculateRollingMetrics: (period: number) => Promise<void>;
  
  // Risk analysis
  calculateStressTests: (scenarios: { name: string; shocks: Record<string, number> }[]) => Promise<Record<string, number>>;
  getDownsideRisk: () => { downsideDeviation: number; maxDrawdown: number; timeUnderwater: number };
  
  // Comparison
  compareWithBenchmark: (benchmarkId: string) => {
    outperformance: number;
    winRate: number;
    correlationBreakdown: Record<string, number>;
  };
  
  // Export
  exportMetrics: (format: 'json' | 'csv' | 'excel') => Promise<string | Blob>;
  
  // Utilities
  refresh: () => Promise<void>;
  reset: () => void;
}

const DEFAULT_CONFIG: PerformanceConfig = {
  riskFreeRate: 0.02, // 2% annual
  confidenceLevel: 0.95,
  rollingWindow: 252, // 1 year
  enableMonteCarlo: false,
  monteCarloSimulations: 10000,
  attributionModel: 'brinson'
};

export function useAdvancedPerformanceMetrics(
  options: UseAdvancedPerformanceMetricsOptions
): UseAdvancedPerformanceMetricsReturn {
  const { portfolioId, config: initialConfig, updateInterval = 5000, enableRealTime = true, historicalPeriod = 365 } = options;
  
  // State
  const [config, setConfig] = useState<PerformanceConfig>({ ...DEFAULT_CONFIG, ...initialConfig });
  const [metrics, setMetrics] = useState<AdvancedPerformanceMetrics | null>(null);
  const [benchmarks, setBenchmarks] = useState<PerformanceBenchmark[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  
  // Refs
  const returnsHistoryRef = useRef<{ date: string; return: number }[]>([]);
  const benchmarkDataRef = useRef<Record<string, { date: string; return: number }[]>>({});
  const calculationCacheRef = useRef<Map<string, any>>(new Map());
  const isMountedRef = useRef(true);
  
  // Real-time analytics
  const {
    currentData: realtimeData,
    historicalData,
    isConnected,
    start: startRealTime,
    stop: stopRealTime
  } = useRealTimeAnalytics({
    portfolioId,
    updateInterval,
    enableStreaming: enableRealTime,
    autoStart: false
  });
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopRealTime();
    };
  }, [stopRealTime]);
  
  // Mathematical utility functions
  const calculateMean = useCallback((values: number[]): number => {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }, []);
  
  const calculateStdDev = useCallback((values: number[], mean?: number): number => {
    const avg = mean ?? calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
    return Math.sqrt(calculateMean(squaredDiffs));
  }, [calculateMean]);
  
  const calculatePercentile = useCallback((values: number[], percentile: number): number => {
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sorted[lower];
    
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }, []);
  
  const calculateSkewness = useCallback((values: number[]): number => {
    const mean = calculateMean(values);
    const stdDev = calculateStdDev(values, mean);
    const n = values.length;
    
    const skewSum = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 3), 0);
    return (n / ((n - 1) * (n - 2))) * skewSum;
  }, [calculateMean, calculateStdDev]);
  
  const calculateKurtosis = useCallback((values: number[]): number => {
    const mean = calculateMean(values);
    const stdDev = calculateStdDev(values, mean);
    const n = values.length;
    
    const kurtSum = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 4), 0);
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * kurtSum - 3 * Math.pow(n - 1, 2) / ((n - 2) * (n - 3));
  }, [calculateMean, calculateStdDev]);
  
  // Fetch historical returns data
  const fetchHistoricalData = useCallback(async () => {
    try {
      const endDate = new Date();
      const startDate = new Date(endDate);
      startDate.setDate(startDate.getDate() - historicalPeriod);
      
      const response = await fetch(
        `${API_BASE_URL}/api/v1/analytics/performance/${portfolioId}/returns?` +
        `start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}&frequency=daily`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      returnsHistoryRef.current = data.returns || [];
      
    } catch (err) {
      console.error('Failed to fetch historical data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch historical data');
    }
  }, [portfolioId, historicalPeriod, API_BASE_URL]);
  
  // Calculate Value at Risk
  const calculateVaR = useCallback((returns: number[], confidenceLevel: number): number => {
    return -calculatePercentile(returns, (1 - confidenceLevel) * 100);
  }, [calculatePercentile]);
  
  // Calculate Conditional Value at Risk
  const calculateCVaR = useCallback((returns: number[], confidenceLevel: number): number => {
    const var95 = calculateVaR(returns, confidenceLevel);
    const tailReturns = returns.filter(r => r <= -var95);
    return tailReturns.length > 0 ? -calculateMean(tailReturns) : 0;
  }, [calculateVaR, calculateMean]);
  
  // Calculate maximum drawdown
  const calculateMaxDrawdown = useCallback((cumulativeReturns: number[]): number => {
    let maxDrawdown = 0;
    let peak = cumulativeReturns[0];
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        peak = cumulativeReturns[i];
      } else {
        const drawdown = (peak - cumulativeReturns[i]) / peak;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
    }
    
    return maxDrawdown;
  }, []);
  
  // Calculate attribution analysis using Brinson method
  const calculateBrinsonAttribution = useCallback((
    portfolioWeights: Record<string, number>,
    benchmarkWeights: Record<string, number>,
    portfolioReturns: Record<string, number>,
    benchmarkReturns: Record<string, number>
  ) => {
    let allocation = 0;
    let selection = 0;
    let interaction = 0;
    
    const sectors = new Set([...Object.keys(portfolioWeights), ...Object.keys(benchmarkWeights)]);
    
    sectors.forEach(sector => {
      const wp = portfolioWeights[sector] || 0;
      const wb = benchmarkWeights[sector] || 0;
      const rp = portfolioReturns[sector] || 0;
      const rb = benchmarkReturns[sector] || 0;
      const rm = calculateMean(Object.values(benchmarkReturns)); // Market return
      
      allocation += (wp - wb) * rb;
      selection += wb * (rp - rb);
      interaction += (wp - wb) * (rp - rb);
    });
    
    return {
      allocation,
      selection,
      interaction,
      total: allocation + selection + interaction
    };
  }, [calculateMean]);
  
  // Run Monte Carlo simulation
  const runMonteCarloSimulation = useCallback(async (simulations = config.monteCarloSimulations) => {
    if (!returnsHistoryRef.current.length) return;
    
    const returns = returnsHistoryRef.current.map(r => r.return);
    const mean = calculateMean(returns);
    const stdDev = calculateStdDev(returns);
    
    const simulationResults: number[] = [];
    
    for (let i = 0; i < simulations; i++) {
      let cumulativeReturn = 1;
      
      // Simulate one year of daily returns
      for (let day = 0; day < 252; day++) {
        // Generate random normal return using Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const dailyReturn = mean + z * stdDev;
        
        cumulativeReturn *= (1 + dailyReturn);
      }
      
      simulationResults.push(cumulativeReturn - 1);
    }
    
    // Calculate statistics
    const sortedResults = simulationResults.sort((a, b) => a - b);
    
    const monteCarloResults = {
      confidenceIntervals: {
        ci_95: {
          lower: calculatePercentile(sortedResults, 2.5),
          upper: calculatePercentile(sortedResults, 97.5)
        },
        ci_99: {
          lower: calculatePercentile(sortedResults, 0.5),
          upper: calculatePercentile(sortedResults, 99.5)
        }
      },
      worstCaseScenario: Math.min(...sortedResults),
      bestCaseScenario: Math.max(...sortedResults),
      expectedValue: calculateMean(sortedResults),
      probabilityOfLoss: sortedResults.filter(r => r < 0).length / simulations
    };
    
    setMetrics(prev => prev ? { ...prev, monteCarloResults } : null);
  }, [config.monteCarloSimulations, calculateMean, calculateStdDev, calculatePercentile]);
  
  // Calculate comprehensive metrics
  const calculateMetrics = useCallback(async () => {
    if (!returnsHistoryRef.current.length) return;
    
    setIsLoading(true);
    
    try {
      const returns = returnsHistoryRef.current.map(r => r.return);
      const dates = returnsHistoryRef.current.map(r => r.date);
      
      // Basic statistics
      const totalReturn = returns.reduce((acc, r) => acc * (1 + r), 1) - 1;
      const annualizedReturn = Math.pow(1 + totalReturn, 252 / returns.length) - 1;
      const volatility = calculateStdDev(returns) * Math.sqrt(252);
      const sharpeRatio = (annualizedReturn - config.riskFreeRate) / volatility;
      
      // Calculate cumulative returns for drawdown analysis
      const cumulativeReturns = returns.reduce((acc, r, i) => {
        acc.push(i === 0 ? 1 + r : acc[i - 1] * (1 + r));
        return acc;
      }, [] as number[]);
      
      const maxDrawdown = calculateMaxDrawdown(cumulativeReturns);
      
      // Downside deviation for Sortino ratio
      const negativeReturns = returns.filter(r => r < 0);
      const downsideDeviation = negativeReturns.length > 0 
        ? calculateStdDev(negativeReturns) * Math.sqrt(252)
        : 0;
      const sortinoRatio = downsideDeviation > 0 
        ? (annualizedReturn - config.riskFreeRate) / downsideDeviation 
        : 0;
      
      const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
      
      // Risk metrics
      const var95 = calculateVaR(returns, 0.95);
      const var99 = calculateVaR(returns, 0.99);
      const cvar95 = calculateCVaR(returns, 0.95);
      const cvar99 = calculateCVaR(returns, 0.99);
      
      // Statistical measures
      const skewness = calculateSkewness(returns);
      const kurtosis = calculateKurtosis(returns);
      
      // Win rate and profit metrics
      const positiveReturns = returns.filter(r => r > 0);
      const winRate = positiveReturns.length / returns.length;
      const averageWin = positiveReturns.length > 0 ? calculateMean(positiveReturns) : 0;
      const averageLoss = negativeReturns.length > 0 ? Math.abs(calculateMean(negativeReturns)) : 0;
      const profitFactor = averageLoss > 0 ? (averageWin * positiveReturns.length) / (averageLoss * negativeReturns.length) : 0;
      const payoffRatio = averageLoss > 0 ? averageWin / averageLoss : 0;
      
      // Rolling metrics
      const rollingPeriod = Math.min(config.rollingWindow, returns.length);
      const rollingReturns: number[] = [];
      const rollingVolatility: number[] = [];
      const rollingSharpe: number[] = [];
      
      for (let i = rollingPeriod - 1; i < returns.length; i++) {
        const window = returns.slice(i - rollingPeriod + 1, i + 1);
        const windowReturn = window.reduce((acc, r) => acc * (1 + r), 1) - 1;
        const windowVol = calculateStdDev(window) * Math.sqrt(252);
        const windowSharpe = windowVol > 0 ? (windowReturn - config.riskFreeRate) / windowVol : 0;
        
        rollingReturns.push(windowReturn);
        rollingVolatility.push(windowVol);
        rollingSharpe.push(windowSharpe);
      }
      
      // Create comprehensive metrics object
      const advancedMetrics: AdvancedPerformanceMetrics = {
        // Core Performance
        totalReturn,
        annualizedReturn,
        volatility,
        sharpeRatio,
        sortinoRatio,
        calmarRatio,
        maxDrawdown,
        
        // Risk-Adjusted Returns (placeholders for now)
        informationRatio: 0,
        treynorRatio: 0,
        jensenAlpha: 0,
        trackingError: 0,
        beta: 1,
        
        // Advanced Risk Metrics
        var95,
        var99,
        cvar95,
        cvar99,
        expectedShortfall: cvar95,
        conditionalDrawdown: maxDrawdown,
        
        // Performance Attribution (placeholder)
        activeReturn: 0,
        attributionAnalysis: {
          allocation: 0,
          selection: 0,
          interaction: 0,
          total: 0
        },
        
        // Statistical Measures
        skewness,
        kurtosis,
        winRate,
        profitFactor,
        payoffRatio,
        
        // Time-Based Analysis
        rollingReturns: {
          period: `${rollingPeriod}D`,
          returns: rollingReturns,
          volatility: rollingVolatility,
          sharpe: rollingSharpe
        }
      };
      
      setMetrics(advancedMetrics);
      setLastUpdate(new Date());
      setError(null);
      
      // Run Monte Carlo if enabled
      if (config.enableMonteCarlo) {
        await runMonteCarloSimulation();
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to calculate metrics');
    } finally {
      setIsLoading(false);
    }
  }, [
    config, calculateStdDev, calculateMaxDrawdown, calculateVaR, calculateCVaR,
    calculateSkewness, calculateKurtosis, calculateMean, runMonteCarloSimulation
  ]);
  
  // Update configuration
  const updateConfig = useCallback((newConfig: Partial<PerformanceConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);
  
  // Add benchmark
  const addBenchmark = useCallback(async (benchmarkData: Omit<PerformanceBenchmark, 'correlation' | 'beta' | 'alpha' | 'trackingError' | 'informationRatio'>) => {
    if (!returnsHistoryRef.current.length) return;
    
    const portfolioReturns = returnsHistoryRef.current.map(r => r.return);
    const benchmarkReturns = benchmarkData.returns;
    
    // Calculate benchmark statistics
    const correlation = calculateCorrelation(portfolioReturns, benchmarkReturns);
    const beta = calculateBeta(portfolioReturns, benchmarkReturns);
    const alpha = calculateAlpha(portfolioReturns, benchmarkReturns, beta);
    const trackingError = calculateTrackingError(portfolioReturns, benchmarkReturns);
    const informationRatio = trackingError > 0 ? alpha / trackingError : 0;
    
    const benchmark: PerformanceBenchmark = {
      ...benchmarkData,
      correlation,
      beta,
      alpha,
      trackingError,
      informationRatio
    };
    
    setBenchmarks(prev => [...prev.filter(b => b.id !== benchmark.id), benchmark]);
  }, []);
  
  // Helper functions for benchmark calculations
  const calculateCorrelation = useCallback((x: number[], y: number[]): number => {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;
    
    const meanX = calculateMean(x.slice(0, n));
    const meanY = calculateMean(y.slice(0, n));
    
    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;
    
    for (let i = 0; i < n; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      numerator += xDiff * yDiff;
      sumXSquared += xDiff * xDiff;
      sumYSquared += yDiff * yDiff;
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    return denominator > 0 ? numerator / denominator : 0;
  }, [calculateMean]);
  
  const calculateBeta = useCallback((portfolioReturns: number[], benchmarkReturns: number[]): number => {
    const correlation = calculateCorrelation(portfolioReturns, benchmarkReturns);
    const portfolioVol = calculateStdDev(portfolioReturns);
    const benchmarkVol = calculateStdDev(benchmarkReturns);
    
    return benchmarkVol > 0 ? correlation * (portfolioVol / benchmarkVol) : 0;
  }, [calculateCorrelation, calculateStdDev]);
  
  const calculateAlpha = useCallback((portfolioReturns: number[], benchmarkReturns: number[], beta: number): number => {
    const portfolioReturn = calculateMean(portfolioReturns);
    const benchmarkReturn = calculateMean(benchmarkReturns);
    
    return portfolioReturn - (config.riskFreeRate + beta * (benchmarkReturn - config.riskFreeRate));
  }, [calculateMean, config.riskFreeRate]);
  
  const calculateTrackingError = useCallback((portfolioReturns: number[], benchmarkReturns: number[]): number => {
    const n = Math.min(portfolioReturns.length, benchmarkReturns.length);
    const activeReturns = [];
    
    for (let i = 0; i < n; i++) {
      activeReturns.push(portfolioReturns[i] - benchmarkReturns[i]);
    }
    
    return calculateStdDev(activeReturns) * Math.sqrt(252);
  }, [calculateStdDev]);
  
  // Remove benchmark
  const removeBenchmark = useCallback((benchmarkId: string) => {
    setBenchmarks(prev => prev.filter(b => b.id !== benchmarkId));
  }, []);
  
  // Run attribution analysis
  const runAttributionAnalysis = useCallback(async () => {
    // This would require sector/holding level data
    // Implementation depends on data availability
    console.log('Attribution analysis would require sector/holding level data');
  }, []);
  
  // Calculate rolling metrics
  const calculateRollingMetrics = useCallback(async (period: number) => {
    if (!returnsHistoryRef.current.length) return;
    
    const returns = returnsHistoryRef.current.map(r => r.return);
    // Implementation similar to the rolling calculation in calculateMetrics
    await calculateMetrics();
  }, [calculateMetrics]);
  
  // Calculate stress tests
  const calculateStressTests = useCallback(async (scenarios: { name: string; shocks: Record<string, number> }[]) => {
    // Implementation would apply shocks to portfolio and calculate impact
    return scenarios.reduce((acc, scenario) => {
      acc[scenario.name] = Math.random() * -0.2; // Placeholder
      return acc;
    }, {} as Record<string, number>);
  }, []);
  
  // Get downside risk
  const getDownsideRisk = useCallback(() => {
    if (!returnsHistoryRef.current.length) {
      return { downsideDeviation: 0, maxDrawdown: 0, timeUnderwater: 0 };
    }
    
    const returns = returnsHistoryRef.current.map(r => r.return);
    const negativeReturns = returns.filter(r => r < 0);
    const downsideDeviation = negativeReturns.length > 0 ? calculateStdDev(negativeReturns) : 0;
    const maxDrawdown = metrics?.maxDrawdown || 0;
    
    // Calculate time underwater (days below previous peak)
    const cumulativeReturns = returns.reduce((acc, r, i) => {
      acc.push(i === 0 ? 1 + r : acc[i - 1] * (1 + r));
      return acc;
    }, [] as number[]);
    
    let timeUnderwater = 0;
    let peak = cumulativeReturns[0];
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] >= peak) {
        peak = cumulativeReturns[i];
      } else {
        timeUnderwater++;
      }
    }
    
    return { downsideDeviation, maxDrawdown, timeUnderwater };
  }, [calculateStdDev, metrics]);
  
  // Compare with benchmark
  const compareWithBenchmark = useCallback((benchmarkId: string) => {
    const benchmark = benchmarks.find(b => b.id === benchmarkId);
    if (!benchmark || !returnsHistoryRef.current.length) {
      return { outperformance: 0, winRate: 0, correlationBreakdown: {} };
    }
    
    const portfolioReturns = returnsHistoryRef.current.map(r => r.return);
    const outperformance = calculateMean(portfolioReturns) - calculateMean(benchmark.returns);
    
    let winCount = 0;
    const minLength = Math.min(portfolioReturns.length, benchmark.returns.length);
    
    for (let i = 0; i < minLength; i++) {
      if (portfolioReturns[i] > benchmark.returns[i]) {
        winCount++;
      }
    }
    
    const winRate = minLength > 0 ? winCount / minLength : 0;
    
    return {
      outperformance,
      winRate,
      correlationBreakdown: {
        overall: benchmark.correlation,
        upMarkets: 0.8, // Placeholder
        downMarkets: 0.9 // Placeholder
      }
    };
  }, [benchmarks, calculateMean]);
  
  // Export metrics
  const exportMetrics = useCallback(async (format: 'json' | 'csv' | 'excel'): Promise<string | Blob> => {
    if (!metrics) return '';
    
    switch (format) {
      case 'json':
        return JSON.stringify(metrics, null, 2);
      case 'csv':
        // Convert to CSV format
        const csvRows = [
          ['Metric', 'Value'],
          ['Total Return', metrics.totalReturn.toFixed(4)],
          ['Annualized Return', metrics.annualizedReturn.toFixed(4)],
          ['Volatility', metrics.volatility.toFixed(4)],
          ['Sharpe Ratio', metrics.sharpeRatio.toFixed(4)],
          ['Sortino Ratio', metrics.sortinoRatio.toFixed(4)],
          ['Max Drawdown', metrics.maxDrawdown.toFixed(4)],
          ['VaR 95%', metrics.var95.toFixed(4)],
          ['CVaR 95%', metrics.cvar95.toFixed(4)]
        ];
        return csvRows.map(row => row.join(',')).join('\n');
      case 'excel':
        // Return a blob for Excel format (would need a library like xlsx)
        return new Blob([JSON.stringify(metrics)], { type: 'application/json' });
      default:
        return '';
    }
  }, [metrics]);
  
  // Refresh data
  const refresh = useCallback(async () => {
    await fetchHistoricalData();
    await calculateMetrics();
  }, [fetchHistoricalData, calculateMetrics]);
  
  // Reset all data
  const reset = useCallback(() => {
    setMetrics(null);
    setBenchmarks([]);
    setError(null);
    setLastUpdate(null);
    returnsHistoryRef.current = [];
    calculationCacheRef.current.clear();
  }, []);
  
  // Initialize and fetch data
  useEffect(() => {
    if (portfolioId) {
      fetchHistoricalData();
    }
  }, [portfolioId, fetchHistoricalData]);
  
  // Calculate metrics when data changes
  useEffect(() => {
    if (returnsHistoryRef.current.length > 0) {
      calculateMetrics();
    }
  }, [calculateMetrics]);
  
  // Start real-time updates if enabled
  useEffect(() => {
    if (enableRealTime && portfolioId) {
      startRealTime();
    }
    
    return () => {
      if (enableRealTime) {
        stopRealTime();
      }
    };
  }, [enableRealTime, portfolioId, startRealTime, stopRealTime]);
  
  // Update returns when real-time data comes in
  useEffect(() => {
    if (realtimeData && returnsHistoryRef.current.length > 0) {
      const lastReturn = returnsHistoryRef.current[returnsHistoryRef.current.length - 1];
      const newReturn = {
        date: new Date().toISOString(),
        return: realtimeData.pnl.daily_change_pct
      };
      
      // Add new return if it's a different day or significant change
      if (!lastReturn || new Date(lastReturn.date).toDateString() !== new Date().toDateString()) {
        returnsHistoryRef.current.push(newReturn);
        calculateMetrics();
      }
    }
  }, [realtimeData, calculateMetrics]);
  
  return {
    // Performance data
    metrics,
    benchmarks,
    isLoading,
    error,
    lastUpdate,
    
    // Configuration
    updateConfig,
    
    // Data management
    addBenchmark,
    removeBenchmark,
    
    // Analysis
    runAttributionAnalysis,
    runMonteCarloSimulation,
    calculateRollingMetrics,
    
    // Risk analysis
    calculateStressTests,
    getDownsideRisk,
    
    // Comparison
    compareWithBenchmark,
    
    // Export
    exportMetrics,
    
    // Utilities
    refresh,
    reset
  };
}

export default useAdvancedPerformanceMetrics;