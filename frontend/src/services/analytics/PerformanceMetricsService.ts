/**
 * Performance Metrics Service - Story 5.1 Analytics Integration
 * Integrates with backend analytics APIs for comprehensive performance analysis
 */

import axios from 'axios';
import { 
  PerformanceAnalyticsResponse,
  MonteCarloRequest,
  MonteCarloResponse,
  AttributionAnalysisResponse,
  StatisticalTestsResponse,
  BenchmarksResponse 
} from '../../types/analytics';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

class PerformanceMetricsService {
  private apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 second timeout for heavy analytics
    headers: {
      'Content-Type': 'application/json',
    },
  });

  /**
   * Get comprehensive performance analytics for a portfolio
   */
  async getPerformanceAnalytics(
    portfolioId: string,
    startDate?: string,
    endDate?: string,
    benchmark: string = 'SPY'
  ): Promise<PerformanceAnalyticsResponse> {
    try {
      const params = new URLSearchParams({
        benchmark,
        ...(startDate && { start_date: startDate }),
        ...(endDate && { end_date: endDate }),
      });

      const response = await this.apiClient.get<PerformanceAnalyticsResponse>(
        `/api/v1/analytics/performance/${portfolioId}?${params}`
      );

      return response.data;
    } catch (error) {
      console.error('Error fetching performance analytics:', error);
      throw this.handleApiError(error, 'Failed to fetch performance analytics');
    }
  }

  /**
   * Run Monte Carlo simulation for portfolio risk/return projections
   */
  async runMonteCarloSimulation(request: MonteCarloRequest): Promise<MonteCarloResponse> {
    try {
      const response = await this.apiClient.post<MonteCarloResponse>(
        '/api/v1/analytics/monte-carlo',
        request
      );

      return response.data;
    } catch (error) {
      console.error('Error running Monte Carlo simulation:', error);
      throw this.handleApiError(error, 'Failed to run Monte Carlo simulation');
    }
  }

  /**
   * Get performance attribution analysis
   */
  async getAttributionAnalysis(
    portfolioId: string,
    attributionType: 'sector' | 'style' | 'security' | 'factor' = 'sector',
    period: string = '3M'
  ): Promise<AttributionAnalysisResponse> {
    try {
      const params = new URLSearchParams({
        attribution_type: attributionType,
        period,
      });

      const response = await this.apiClient.get<AttributionAnalysisResponse>(
        `/api/v1/analytics/attribution/${portfolioId}?${params}`
      );

      return response.data;
    } catch (error) {
      console.error('Error fetching attribution analysis:', error);
      throw this.handleApiError(error, 'Failed to fetch attribution analysis');
    }
  }

  /**
   * Get statistical significance tests for portfolio performance
   */
  async getStatisticalTests(
    portfolioId: string,
    testType: string = 'sharpe',
    significanceLevel: number = 0.05
  ): Promise<StatisticalTestsResponse> {
    try {
      const params = new URLSearchParams({
        test_type: testType,
        significance_level: significanceLevel.toString(),
      });

      const response = await this.apiClient.get<StatisticalTestsResponse>(
        `/api/v1/analytics/statistical-tests/${portfolioId}?${params}`
      );

      return response.data;
    } catch (error) {
      console.error('Error fetching statistical tests:', error);
      throw this.handleApiError(error, 'Failed to fetch statistical tests');
    }
  }

  /**
   * Get available benchmarks for comparison
   */
  async getAvailableBenchmarks(): Promise<BenchmarksResponse> {
    try {
      const response = await this.apiClient.get<BenchmarksResponse>(
        '/api/v1/analytics/benchmarks'
      );

      return response.data;
    } catch (error) {
      console.error('Error fetching benchmarks:', error);
      throw this.handleApiError(error, 'Failed to fetch available benchmarks');
    }
  }

  /**
   * Calculate Alpha using CAPM model
   * Alpha = Portfolio Return - (Risk-free Rate + Beta * (Market Return - Risk-free Rate))
   */
  calculateAlpha(
    portfolioReturn: number,
    marketReturn: number,
    beta: number,
    riskFreeRate: number = 0.02
  ): number {
    return portfolioReturn - (riskFreeRate + beta * (marketReturn - riskFreeRate));
  }

  /**
   * Calculate Information Ratio
   * Information Ratio = Active Return / Tracking Error
   */
  calculateInformationRatio(portfolioReturn: number, benchmarkReturn: number, trackingError: number): number {
    if (trackingError === 0) return 0;
    return (portfolioReturn - benchmarkReturn) / trackingError;
  }

  /**
   * Calculate Sharpe Ratio
   * Sharpe Ratio = (Portfolio Return - Risk-free Rate) / Portfolio Volatility
   */
  calculateSharpeRatio(portfolioReturn: number, volatility: number, riskFreeRate: number = 0.02): number {
    if (volatility === 0) return 0;
    return (portfolioReturn - riskFreeRate) / volatility;
  }

  /**
   * Calculate Sortino Ratio (downside risk-adjusted return)
   * Sortino Ratio = (Portfolio Return - Risk-free Rate) / Downside Deviation
   */
  calculateSortinoRatio(portfolioReturn: number, downsideDeviation: number, riskFreeRate: number = 0.02): number {
    if (downsideDeviation === 0) return 0;
    return (portfolioReturn - riskFreeRate) / downsideDeviation;
  }

  /**
   * Calculate Calmar Ratio
   * Calmar Ratio = Annual Return / Maximum Drawdown
   */
  calculateCalmarRatio(annualReturn: number, maxDrawdown: number): number {
    if (maxDrawdown === 0) return 0;
    return annualReturn / Math.abs(maxDrawdown);
  }

  /**
   * Calculate Maximum Drawdown from price series
   */
  calculateMaxDrawdown(priceSeries: number[]): number {
    if (priceSeries.length < 2) return 0;

    let maxDrawdown = 0;
    let peak = priceSeries[0];

    for (let i = 1; i < priceSeries.length; i++) {
      if (priceSeries[i] > peak) {
        peak = priceSeries[i];
      }

      const drawdown = (peak - priceSeries[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown * 100; // Return as percentage
  }

  /**
   * Calculate rolling window metrics
   */
  calculateRollingMetrics(
    returns: number[],
    benchmarkReturns: number[],
    windowSize: number = 30
  ): Array<{
    date: string;
    alpha: number;
    beta: number;
    sharpe_ratio: number;
  }> {
    const rollingMetrics = [];
    const minLength = Math.min(returns.length, benchmarkReturns.length);

    for (let i = windowSize; i < minLength; i++) {
      const windowReturns = returns.slice(i - windowSize, i);
      const windowBenchmarkReturns = benchmarkReturns.slice(i - windowSize, i);

      // Calculate beta for the window
      const beta = this.calculateBeta(windowReturns, windowBenchmarkReturns);
      
      // Calculate average returns for the window
      const avgPortfolioReturn = windowReturns.reduce((sum, r) => sum + r, 0) / windowSize;
      const avgBenchmarkReturn = windowBenchmarkReturns.reduce((sum, r) => sum + r, 0) / windowSize;
      
      // Calculate volatility for Sharpe ratio
      const volatility = this.calculateVolatility(windowReturns);
      
      // Calculate metrics
      const alpha = this.calculateAlpha(avgPortfolioReturn * 252, avgBenchmarkReturn * 252, beta);
      const sharpeRatio = this.calculateSharpeRatio(avgPortfolioReturn * 252, volatility);

      rollingMetrics.push({
        date: new Date(Date.now() - (minLength - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        alpha: alpha,
        beta: beta,
        sharpe_ratio: sharpeRatio
      });
    }

    return rollingMetrics;
  }

  /**
   * Calculate Beta (market sensitivity)
   */
  private calculateBeta(portfolioReturns: number[], marketReturns: number[]): number {
    if (portfolioReturns.length !== marketReturns.length || portfolioReturns.length === 0) {
      return 1.0; // Default beta
    }

    const n = portfolioReturns.length;
    const portfolioMean = portfolioReturns.reduce((sum, r) => sum + r, 0) / n;
    const marketMean = marketReturns.reduce((sum, r) => sum + r, 0) / n;

    let covariance = 0;
    let marketVariance = 0;

    for (let i = 0; i < n; i++) {
      const portfolioDiff = portfolioReturns[i] - portfolioMean;
      const marketDiff = marketReturns[i] - marketMean;
      
      covariance += portfolioDiff * marketDiff;
      marketVariance += marketDiff * marketDiff;
    }

    covariance /= n - 1;
    marketVariance /= n - 1;

    return marketVariance === 0 ? 1.0 : covariance / marketVariance;
  }

  /**
   * Calculate volatility (annualized standard deviation)
   */
  private calculateVolatility(returns: number[]): number {
    if (returns.length === 0) return 0;

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const squaredDiffs = returns.map(r => Math.pow(r - mean, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / (returns.length - 1);
    
    return Math.sqrt(variance * 252); // Annualized volatility
  }

  /**
   * Handle API errors with user-friendly messages
   */
  private handleApiError(error: any, defaultMessage: string): Error {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const message = error.response?.data?.detail || error.message;

      switch (status) {
        case 400:
          return new Error(`Invalid request: ${message}`);
        case 401:
          return new Error('Authentication required');
        case 403:
          return new Error('Access denied');
        case 404:
          return new Error('Analytics service not available');
        case 422:
          return new Error(`Validation error: ${message}`);
        case 500:
          return new Error('Analytics service error. Please try again.');
        default:
          return new Error(`${defaultMessage}: ${message}`);
      }
    }

    return new Error(defaultMessage);
  }

  /**
   * Validate portfolio ID format
   */
  validatePortfolioId(portfolioId: string): boolean {
    return portfolioId && portfolioId.length > 0 && portfolioId.trim() !== '';
  }

  /**
   * Validate date range for analytics
   */
  validateDateRange(startDate: string, endDate: string): boolean {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const now = new Date();

    return start <= end && end <= now && start >= new Date(now.getFullYear() - 10, 0, 1);
  }
}

// Export singleton instance
export const performanceMetricsService = new PerformanceMetricsService();
export default PerformanceMetricsService;