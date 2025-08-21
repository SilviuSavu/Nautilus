/**
 * Mock PerformanceMetricsService for Testing
 */
import { vi } from 'vitest';

export const performanceMetricsService = {
  getPerformanceAnalytics: vi.fn(),
  runMonteCarloSimulation: vi.fn(),
  getAttributionAnalysis: vi.fn(),
  getStatisticalTests: vi.fn(),
  getAvailableBenchmarks: vi.fn(() => Promise.resolve([
    { symbol: 'SPY', name: 'S&P 500', category: 'Large Cap', data_available_from: '2020-01-01' },
    { symbol: 'QQQ', name: 'NASDAQ 100', category: 'Tech', data_available_from: '2020-01-01' }
  ]))
};

export default performanceMetricsService;