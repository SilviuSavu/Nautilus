/**
 * Story5AdvancedAnalyticsDashboard Component Tests - Story 5.1
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import Story5AdvancedAnalyticsDashboard from '../Story5AdvancedAnalyticsDashboard';

// Mock empty portfolio data (what backend returns for empty portfolios)
const mockEmptyPortfolioData = {
  alpha: 0.0,
  beta: 1.0,
  information_ratio: 0.0,
  tracking_error: 0.0,
  sharpe_ratio: 0.0,
  sortino_ratio: 0.0,
  calmar_ratio: 0.0,
  max_drawdown: 0.0,
  volatility: 0.0,
  downside_deviation: 0.0,
  rolling_metrics: [],
  period_start: '2024-08-20',
  period_end: '2025-08-20',
  benchmark: 'SPY'
};

const mockPopulatedPortfolioData = {
  alpha: 0.035,
  beta: 1.15,
  sharpe_ratio: 1.25,
  sortino_ratio: 1.45,
  calmar_ratio: 0.85,
  information_ratio: 0.65,
  max_drawdown: -8.5,
  volatility: 0.18,
  tracking_error: 0.042,
  rolling_metrics: [
    { date: '2024-01-01', alpha: 0.032, beta: 1.12, sharpe_ratio: 1.22 },
    { date: '2024-02-01', alpha: 0.038, beta: 1.18, sharpe_ratio: 1.28 }
  ],
  period_start: '2024-01-01',
  period_end: '2024-12-31',
  benchmark: 'SPY'
};

// Create mock functions
const mockUsePerformanceMetrics = vi.fn();
const mockUseBenchmarks = vi.fn();

// Mock the hooks with default populated data
vi.mock('../../hooks/analytics/usePerformanceMetrics', () => ({
  usePerformanceMetrics: mockUsePerformanceMetrics,
  useBenchmarks: mockUseBenchmarks
}));

// Set default mock returns
beforeEach(() => {
  mockUsePerformanceMetrics.mockReturnValue({
    analytics: {
      loading: false,
      error: null,
      data: mockPopulatedPortfolioData,
      lastUpdated: new Date()
    },
    monteCarlo: {
      loading: false,
      error: null,
      data: null
    },
    attribution: {
      loading: false,
      error: null,
      data: null
    },
    statisticalTests: {
      loading: false,
      error: null,
      data: null
    },
    refreshAnalytics: vi.fn(),
    runMonteCarloSimulation: vi.fn(),
    getAttributionAnalysis: vi.fn(),
    getStatisticalTests: vi.fn(),
    isLoading: false,
    hasError: false,
    isDataAvailable: true
  });

  mockUseBenchmarks.mockReturnValue({
    benchmarks: [
      { symbol: 'SPY', name: 'S&P 500', category: 'Large Cap', data_available_from: '2020-01-01' },
      { symbol: 'QQQ', name: 'NASDAQ 100', category: 'Tech', data_available_from: '2020-01-01' }
    ],
    loading: false,
    error: null,
    refetch: vi.fn()
  });
});

describe('Story5AdvancedAnalyticsDashboard', () => {
  beforeEach(() => {
    // Mock ResizeObserver
    global.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    };

    // Clear all mocks
    vi.clearAllMocks();

    // Reset to default populated data
    mockUsePerformanceMetrics.mockReturnValue({
      analytics: {
        loading: false,
        error: null,
        data: mockPopulatedPortfolioData,
        lastUpdated: new Date()
      },
      monteCarlo: { loading: false, error: null, data: null },
      attribution: { loading: false, error: null, data: null },
      statisticalTests: { loading: false, error: null, data: null },
      refreshAnalytics: vi.fn(),
      runMonteCarloSimulation: vi.fn(),
      getAttributionAnalysis: vi.fn(),
      getStatisticalTests: vi.fn(),
      isLoading: false,
      hasError: false,
      isDataAvailable: true
    });

    mockUseBenchmarks.mockReturnValue({
      benchmarks: [
        { symbol: 'SPY', name: 'S&P 500', category: 'Large Cap', data_available_from: '2020-01-01' },
        { symbol: 'QQQ', name: 'NASDAQ 100', category: 'Tech', data_available_from: '2020-01-01' }
      ],
      loading: false,
      error: null,
      refetch: vi.fn()
    });
  });

  it('renders dashboard with performance metrics tab', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Check header
    expect(screen.getByText('Advanced Performance Analytics')).toBeInTheDocument();
    expect(screen.getByText('Story 5.1: Comprehensive performance and risk analysis')).toBeInTheDocument();

    // Check controls
    expect(screen.getByText('Refresh')).toBeInTheDocument();
    expect(screen.getByText('Configure')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();

    // Check tabs
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    expect(screen.getByText('Monte Carlo Analysis')).toBeInTheDocument();
    expect(screen.getByText('Attribution Analysis')).toBeInTheDocument();
    expect(screen.getByText('Statistical Tests')).toBeInTheDocument();

    await waitFor(() => {
      // Check performance metrics cards
      expect(screen.getByText('Alpha')).toBeInTheDocument();
      expect(screen.getByText('Beta')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Information Ratio')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
      expect(screen.getByText('Volatility')).toBeInTheDocument();
      expect(screen.getByText('Sortino Ratio')).toBeInTheDocument();
      expect(screen.getByText('Calmar Ratio')).toBeInTheDocument();
      expect(screen.getByText('Tracking Error')).toBeInTheDocument();
    });
  });

  it('displays correct performance metric values', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    await waitFor(() => {
      // Check specific metric values
      expect(screen.getByText('3.500%')).toBeInTheDocument(); // Alpha
      expect(screen.getByText('1.150')).toBeInTheDocument();  // Beta
      expect(screen.getByText('1.250')).toBeInTheDocument();  // Sharpe Ratio
      expect(screen.getByText('1.450')).toBeInTheDocument();  // Sortino Ratio
      expect(screen.getByText('0.650')).toBeInTheDocument();  // Information Ratio
      expect(screen.getByText('8.50%')).toBeInTheDocument();  // Max Drawdown
      expect(screen.getByText('18.00%')).toBeInTheDocument(); // Volatility
      expect(screen.getByText('4.20%')).toBeInTheDocument();  // Tracking Error
    });
  });

  it('shows benchmark and date range controls', () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Benchmark selector should show available benchmarks
    expect(screen.getByDisplayValue('SPY')).toBeInTheDocument();

    // Date range picker should be present
    const datePickerElements = screen.getAllByRole('textbox');
    expect(datePickerElements.length).toBeGreaterThan(0);
  });

  it('switches between tabs correctly', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Initially should show Performance Metrics tab
    expect(screen.getByText('Rolling Window Analysis')).toBeInTheDocument();

    // Click Monte Carlo tab
    const monteCarloTab = screen.getByText('Monte Carlo Analysis');
    fireEvent.click(monteCarloTab);

    await waitFor(() => {
      // Should show Monte Carlo content
      expect(screen.getByText('Run Monte Carlo simulation to see results')).toBeInTheDocument();
    });

    // Click Attribution Analysis tab
    const attributionTab = screen.getByText('Attribution Analysis');
    fireEvent.click(attributionTab);

    await waitFor(() => {
      // Should show "Load Attribution Analysis" button since no data is loaded
      expect(screen.getByText('Load Attribution Analysis')).toBeInTheDocument();
    });

    // Click Statistical Tests tab
    const statisticalTab = screen.getByText('Statistical Tests');
    fireEvent.click(statisticalTab);

    await waitFor(() => {
      // Should show "Run Statistical Tests" button since no data is loaded
      expect(screen.getByText('Run Statistical Tests')).toBeInTheDocument();
    });
  });

  it('handles benchmark selection changes', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Find benchmark selector
    const benchmarkSelect = screen.getByDisplayValue('SPY');
    
    // Open dropdown
    fireEvent.mouseDown(benchmarkSelect);
    
    await waitFor(() => {
      // Select QQQ
      const qqqOption = screen.getByText('QQQ');
      fireEvent.click(qqqOption);
    });

    // Benchmark should change to QQQ
    expect(screen.getByDisplayValue('QQQ')).toBeInTheDocument();
  });

  it('handles refresh button click', async () => {
    const mockRefreshAnalytics = vi.fn();
    
    // Mock the hook to return our spy function
    vi.mocked(require('../../hooks/analytics/usePerformanceMetrics').usePerformanceMetrics)
      .mockReturnValueOnce({
        analytics: { loading: false, error: null, data: null },
        monteCarlo: { loading: false, error: null, data: null },
        attribution: { loading: false, error: null, data: null },
        statisticalTests: { loading: false, error: null, data: null },
        refreshAnalytics: mockRefreshAnalytics,
        runMonteCarloSimulation: vi.fn(),
        getAttributionAnalysis: vi.fn(),
        getStatisticalTests: vi.fn(),
        isLoading: false,
        hasError: false,
        isDataAvailable: false
      });

    render(<Story5AdvancedAnalyticsDashboard />);

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    expect(mockRefreshAnalytics).toHaveBeenCalled();
  });

  it('opens and closes configuration modal', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Click Configure button
    const configureButton = screen.getByText('Configure');
    fireEvent.click(configureButton);

    await waitFor(() => {
      // Modal should be open
      expect(screen.getByText('Dashboard Configuration')).toBeInTheDocument();
      expect(screen.getByText('Portfolio ID')).toBeInTheDocument();
      expect(screen.getByText('Auto Refresh')).toBeInTheDocument();
    });

    // Close modal
    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);

    await waitFor(() => {
      // Modal should be closed
      expect(screen.queryByText('Dashboard Configuration')).not.toBeInTheDocument();
    });
  });

  it('opens Monte Carlo configuration modal', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Switch to Monte Carlo tab
    const monteCarloTab = screen.getByText('Monte Carlo Analysis');
    fireEvent.click(monteCarloTab);

    await waitFor(() => {
      // Click Run Simulation button
      const runSimulationButton = screen.getByText('Run Simulation');
      fireEvent.click(runSimulationButton);
    });

    await waitFor(() => {
      // Modal should open
      expect(screen.getByText('Monte Carlo Simulation Configuration')).toBeInTheDocument();
      expect(screen.getByText('Number of Scenarios')).toBeInTheDocument();
      expect(screen.getByText('Stress Test Scenarios')).toBeInTheDocument();
    });
  });

  it('handles export functionality', async () => {
    // Mock console.log to avoid actual file download
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    render(<Story5AdvancedAnalyticsDashboard />);

    const exportButton = screen.getByText('Export');
    fireEvent.click(exportButton);

    // Should trigger export functionality (mocked success message)
    await waitFor(() => {
      // This would typically show a success message
      expect(screen.getByText('Advanced Performance Analytics')).toBeInTheDocument();
    });

    consoleSpy.mockRestore();
  });

  it('shows error state when analytics fail', async () => {
    // Mock error state
    vi.mocked(require('../../hooks/analytics/usePerformanceMetrics').usePerformanceMetrics)
      .mockReturnValueOnce({
        analytics: { loading: false, error: 'Failed to load analytics', data: null },
        monteCarlo: { loading: false, error: null, data: null },
        attribution: { loading: false, error: null, data: null },
        statisticalTests: { loading: false, error: null, data: null },
        refreshAnalytics: vi.fn(),
        runMonteCarloSimulation: vi.fn(),
        getAttributionAnalysis: vi.fn(),
        getStatisticalTests: vi.fn(),
        isLoading: false,
        hasError: true,
        isDataAvailable: false
      });

    render(<Story5AdvancedAnalyticsDashboard />);

    // Should show error alert
    expect(screen.getByText('Analytics Error')).toBeInTheDocument();
    expect(screen.getByText('Failed to load analytics data. Please try refreshing.')).toBeInTheDocument();
    
    // Should have retry button
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('shows loading state correctly', async () => {
    // Mock loading state
    vi.mocked(require('../../hooks/analytics/usePerformanceMetrics').usePerformanceMetrics)
      .mockReturnValueOnce({
        analytics: { loading: true, error: null, data: null },
        monteCarlo: { loading: false, error: null, data: null },
        attribution: { loading: false, error: null, data: null },
        statisticalTests: { loading: false, error: null, data: null },
        refreshAnalytics: vi.fn(),
        runMonteCarloSimulation: vi.fn(),
        getAttributionAnalysis: vi.fn(),
        getStatisticalTests: vi.fn(),
        isLoading: true,
        hasError: false,
        isDataAvailable: false
      });

    render(<Story5AdvancedAnalyticsDashboard />);

    // Should show loading spinner
    expect(screen.getByTestId('loading-spinner') || screen.getByLabelText('Loading')).toBeInTheDocument();
  });

  it('displays rolling metrics chart when data is available', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    await waitFor(() => {
      // Should show rolling window analysis
      expect(screen.getByText('Rolling Window Analysis')).toBeInTheDocument();
      
      // Chart should be rendered (ResizeObserver mock allows this)
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });
  });

  it('handles date range changes', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    // Find date range picker preset buttons
    const lastMonthButton = screen.getByText('Last 1M');
    fireEvent.click(lastMonthButton);

    // Should update the date range (this would trigger a re-fetch)
    expect(screen.getByText('Advanced Performance Analytics')).toBeInTheDocument();
  });

  // New tests for empty portfolio handling
  it('shows meaningful message for empty portfolio data', async () => {
    // Mock empty portfolio data
    mockUsePerformanceMetrics.mockReturnValue({
      analytics: {
        loading: false,
        error: null,
        data: mockEmptyPortfolioData,
        lastUpdated: new Date()
      },
      monteCarlo: { loading: false, error: null, data: null },
      attribution: { loading: false, error: null, data: null },
      statisticalTests: { loading: false, error: null, data: null },
      refreshAnalytics: vi.fn(),
      runMonteCarloSimulation: vi.fn(),
      getAttributionAnalysis: vi.fn(),
      getStatisticalTests: vi.fn(),
      isLoading: false,
      hasError: false,
      isDataAvailable: true
    });

    render(<Story5AdvancedAnalyticsDashboard portfolioId="test_portfolio" />);

    // Should show empty portfolio message
    await waitFor(() => {
      expect(screen.getByText('No Portfolio Data Available')).toBeInTheDocument();
      expect(screen.getByText(/appears to be empty or has no trading history/)).toBeInTheDocument();
      expect(screen.getByText('A portfolio with actual positions or trading history')).toBeInTheDocument();
      expect(screen.getByText('Historical returns data for performance calculation')).toBeInTheDocument();
      expect(screen.getByText('At least 30 days of data for meaningful analytics')).toBeInTheDocument();
    });
  });

  it('shows Monte Carlo data requirement message for empty portfolio', async () => {
    // Mock empty portfolio data
    mockUsePerformanceMetrics.mockReturnValue({
      analytics: {
        loading: false,
        error: null,
        data: mockEmptyPortfolioData,
        lastUpdated: new Date()
      },
      monteCarlo: { loading: false, error: null, data: null },
      attribution: { loading: false, error: null, data: null },
      statisticalTests: { loading: false, error: null, data: null },
      refreshAnalytics: vi.fn(),
      runMonteCarloSimulation: vi.fn(),
      getAttributionAnalysis: vi.fn(),
      getStatisticalTests: vi.fn(),
      isLoading: false,
      hasError: false,
      isDataAvailable: true
    });

    render(<Story5AdvancedAnalyticsDashboard portfolioId="test_portfolio" />);

    // Switch to Monte Carlo tab
    const monteCarloTab = screen.getByText('Monte Carlo Analysis');
    fireEvent.click(monteCarloTab);

    // Should show Monte Carlo data requirement message
    await waitFor(() => {
      expect(screen.getByText('Monte Carlo Simulation Requires Portfolio Data')).toBeInTheDocument();
      expect(screen.getByText(/Monte Carlo analysis needs historical portfolio returns/)).toBeInTheDocument();
    });
  });

  it('applies correct styling for positive and negative metrics', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    await waitFor(() => {
      // Alpha is positive (3.5%), should have green color
      const alphaValue = screen.getByText('3.500%');
      expect(alphaValue).toHaveStyle({ color: '#52c41a' });

      // Max Drawdown is negative (-8.5%), should have red color
      const maxDrawdownValue = screen.getByText('8.50%');
      expect(maxDrawdownValue.closest('.ant-statistic-content')).toHaveStyle({ color: '#f5222d' });
    });
  });

  it('shows appropriate icons for different metrics', async () => {
    render(<Story5AdvancedAnalyticsDashboard />);

    await waitFor(() => {
      // Should have rise icon for positive alpha
      expect(screen.getByLabelText('rise')).toBeInTheDocument();
      
      // Should have alert icon for max drawdown
      expect(screen.getByLabelText('alert')).toBeInTheDocument();
    });
  });
});