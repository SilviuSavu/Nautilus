/**
 * Analytics Type Definitions for Story 5.1
 * Matches backend API response schemas exactly
 */

// Performance Analytics Response Types
export interface PerformanceAnalyticsResponse {
  alpha: number;
  beta: number;
  information_ratio: number;
  tracking_error: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  volatility: number;
  downside_deviation: number;
  rolling_metrics: RollingMetric[];
  period_start: string;
  period_end: string;
  benchmark: string;
}

export interface RollingMetric {
  date: string;
  alpha: number;
  beta: number;
  sharpe_ratio: number;
}

// Monte Carlo Simulation Types
export interface MonteCarloRequest {
  portfolio_id: string;
  scenarios: number;
  time_horizon_days: number;
  confidence_levels: number[];
  stress_scenarios?: string[];
}

export interface MonteCarloResponse {
  scenarios_run: number;
  time_horizon_days: number;
  confidence_intervals: {
    percentile_5: number;
    percentile_25: number;
    percentile_50: number;
    percentile_75: number;
    percentile_95: number;
  };
  expected_return: number;
  probability_of_loss: number;
  value_at_risk_5: number;
  expected_shortfall_5: number;
  worst_case_scenario: number;
  best_case_scenario: number;
  stress_test_results: StressTestResult[];
  simulation_paths: number[][];
}

export interface StressTestResult {
  scenario_name: string;
  probability_of_loss: number;
  expected_loss: number;
  var_95: number;
}

// Attribution Analysis Types
export interface AttributionAnalysisResponse {
  attribution_type: 'sector' | 'style' | 'security' | 'factor';
  period_start: string;
  period_end: string;
  total_active_return: number;
  attribution_breakdown: {
    security_selection: number;
    asset_allocation: number;
    interaction_effect: number;
    currency_effect?: number;
  };
  sector_attribution: SectorAttribution[];
  factor_attribution: FactorAttribution[];
}

export interface SectorAttribution {
  sector: string;
  portfolio_weight: number;
  benchmark_weight: number;
  portfolio_return: number;
  benchmark_return: number;
  allocation_effect: number;
  selection_effect: number;
  total_effect: number;
}

export interface FactorAttribution {
  factor_name: string;
  factor_exposure: number;
  factor_return: number;
  contribution: number;
}

// Statistical Tests Types
export interface StatisticalTestsResponse {
  sharpe_ratio_test: {
    sharpe_ratio: number;
    t_statistic: number;
    p_value: number;
    is_significant: boolean;
    confidence_interval: [number, number];
  };
  alpha_significance_test: {
    alpha: number;
    t_statistic: number;
    p_value: number;
    is_significant: boolean;
    confidence_interval: [number, number];
  };
  beta_stability_test: {
    beta: number;
    rolling_beta_std: number;
    stability_score: number;
    regime_changes_detected: number;
  };
  performance_persistence: {
    persistence_score: number;
    consecutive_winning_periods: number;
    consistency_rating: 'High' | 'Medium' | 'Low';
  };
  bootstrap_results: BootstrapResult[];
}

export interface BootstrapResult {
  metric: string;
  bootstrap_mean: number;
  bootstrap_std: number;
  confidence_interval_95: [number, number];
}

// Benchmarks Types
export interface BenchmarksResponse {
  benchmarks: Benchmark[];
}

export interface Benchmark {
  symbol: string;
  name: string;
  category: string;
  data_available_from: string;
}

// Frontend-specific types for UI state management
export interface AnalyticsState {
  loading: boolean;
  error: string | null;
  data: PerformanceAnalyticsResponse | null;
  lastUpdated: Date | null;
}

export interface MonteCarloState {
  loading: boolean;
  error: string | null;
  data: MonteCarloResponse | null;
  simulationProgress: number;
}

export interface AttributionState {
  loading: boolean;
  error: string | null;
  data: AttributionAnalysisResponse | null;
  attributionType: 'sector' | 'style' | 'security' | 'factor';
}

export interface StatisticalTestsState {
  loading: boolean;
  error: string | null;
  data: StatisticalTestsResponse | null;
  significanceLevel: number;
}

export interface BenchmarksState {
  loading: boolean;
  error: string | null;
  data: Benchmark[];
  selectedBenchmark: string;
}

// Chart data types for visualization
export interface PerformanceChartData {
  date: string;
  portfolio_return: number;
  benchmark_return: number;
  alpha: number;
  beta: number;
  sharpe_ratio: number;
}

export interface MonteCarloChartData {
  scenario: number;
  final_return: number;
  path: number[];
}

export interface AttributionChartData {
  sector: string;
  allocation_effect: number;
  selection_effect: number;
  total_effect: number;
}

// Error handling types
export interface AnalyticsError {
  code: string;
  message: string;
  details?: string;
  timestamp: Date;
}

export interface OfflineState {
  isOnline: boolean;
  lastSyncTime: Date | null;
  cachedData: {
    performance?: PerformanceAnalyticsResponse;
    monteCarlo?: MonteCarloResponse;
    attribution?: AttributionAnalysisResponse;
    statisticalTests?: StatisticalTestsResponse;
  };
}

// Configuration types
export interface AnalyticsConfig {
  defaultBenchmark: string;
  defaultTimeframe: string;
  autoRefresh: boolean;
  refreshInterval: number; // milliseconds
  maxRetries: number;
  offlineMode: boolean;
}

// Performance metrics calculation types
export interface PerformanceMetrics {
  alpha: number;
  beta: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  information_ratio: number;
  tracking_error: number;
  max_drawdown: number;
  volatility: number;
  downside_deviation: number;
}

export interface RiskAdjustedReturns {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  treynor_ratio: number;
  jensen_alpha: number;
}

// Time series data types
export interface TimeSeriesData {
  date: string;
  value: number;
}

export interface PriceData extends TimeSeriesData {
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
}

export interface ReturnData extends TimeSeriesData {
  cumulative_return: number;
  daily_return: number;
}

// Export all types as a namespace for easier imports
export namespace Analytics {
  export type Performance = PerformanceAnalyticsResponse;
  export type MonteCarlo = MonteCarloResponse;
  export type Attribution = AttributionAnalysisResponse;
  export type StatisticalTests = StatisticalTestsResponse;
  export type Benchmarks = BenchmarksResponse;
  export type State = AnalyticsState;
  export type Config = AnalyticsConfig;
  export type Error = AnalyticsError;
  export type OfflineState = OfflineState;
}