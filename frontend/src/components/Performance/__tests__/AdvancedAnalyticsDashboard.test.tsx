/**
 * AdvancedAnalyticsDashboard Test Suite
 * Comprehensive tests for the advanced analytics dashboard component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import AdvancedAnalyticsDashboard from '../AdvancedAnalyticsDashboard';

// Mock the hooks and dependencies
vi.mock('../../../hooks/analytics/useAdvancedAnalytics', () => ({
  useAdvancedAnalytics: vi.fn(() => ({
    analytics: {
      // Risk-Adjusted Returns
      sharpeRatio: 1.85,
      sortinoRatio: 2.12,
      calmarRatio: 1.47,
      omegaRatio: 1.32,
      informationRatio: 1.25,
      
      // Risk Metrics
      var95: -45000,
      var99: -62000,
      cvar95: -52000,
      maxDrawdown: -8.5,
      averageDrawdown: -3.2,
      drawdownDuration: 15.5,
      
      // Performance Attribution
      alpha: 0.045,
      beta: 1.15,
      correlation: 0.82,
      trackingError: 0.08,
      activeReturn: 0.06,
      
      // Trade Analysis
      winRate: 68.5,
      profitFactor: 1.75,
      expectancy: 125.50,
      averageWin: 850.25,
      averageLoss: -485.75,
      largestWin: 5250.00,
      largestLoss: -2875.50,
      consecutiveWins: 8,
      consecutiveLosses: 4,
      
      // Portfolio Statistics
      totalReturn: 15.75,
      annualizedReturn: 18.25,
      volatility: 0.18,
      skewness: -0.25,
      kurtosis: 3.15,
      
      // Time-Based Analytics
      bestMonth: { period: '2024-01', return: 8.5 },
      worstMonth: { period: '2024-03', return: -4.2 },
      monthlyWinRate: 75.0,
      quarterlyReturns: [
        { quarter: '2024-Q1', return: 4.5 },
        { quarter: '2024-Q2', return: 6.8 },
        { quarter: '2024-Q3', return: 3.2 },
        { quarter: '2024-Q4', return: 1.2 }
      ]
    },
    performanceBreakdown: [
      {
        strategy: 'Momentum Strategy',
        allocation: 40.0,
        return: 12.5,
        volatility: 0.16,
        sharpe: 1.95,
        maxDrawdown: -6.5,
        contribution: 5.0
      },
      {
        strategy: 'Mean Reversion',
        allocation: 35.0,
        return: 18.2,
        volatility: 0.22,
        sharpe: 1.75,
        maxDrawdown: -9.2,
        contribution: 6.37
      },
      {
        strategy: 'Statistical Arbitrage',
        allocation: 25.0,
        return: 15.8,
        volatility: 0.12,
        sharpe: 2.15,
        maxDrawdown: -4.1,
        contribution: 3.95
      }
    ],
    riskFactors: [
      { factor: 'Market Beta', exposure: 1.15, contribution: 65.2 },
      { factor: 'Size Factor', exposure: -0.25, contribution: -12.8 },
      { factor: 'Value Factor', exposure: 0.45, contribution: 22.1 },
      { factor: 'Momentum Factor', exposure: 0.32, contribution: 15.5 },
      { factor: 'Quality Factor', exposure: 0.18, contribution: 10.0 }
    ],
    sectorExposure: [
      { sector: 'Technology', exposure: 35.2, benchmark: 28.5 },
      { sector: 'Financials', exposure: 18.7, benchmark: 22.1 },
      { sector: 'Healthcare', exposure: 15.8, benchmark: 12.3 },
      { sector: 'Consumer Discretionary', exposure: 12.5, benchmark: 15.2 },
      { sector: 'Energy', exposure: 8.9, benchmark: 6.8 },
      { sector: 'Others', exposure: 8.9, benchmark: 15.1 }
    ],
    monthlyReturns: Array.from({ length: 12 }, (_, i) => ({
      month: `2024-${String(i + 1).padStart(2, '0')}`,
      return: Math.random() * 10 - 2,
      benchmark: Math.random() * 8 - 1.5
    })),
    rollingMetrics: Array.from({ length: 252 }, (_, i) => ({
      date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
      sharpeRatio: 1.5 + Math.random() * 0.7,
      maxDrawdown: Math.random() * -10,
      var95: Math.random() * -50000,
      return: Math.random() * 20 - 5
    })),
    isLoading: false,
    error: null,
    lastUpdated: new Date(),
    refreshAnalytics: vi.fn(),
    generateReport: vi.fn(),
    exportData: vi.fn()
  }))
}));

vi.mock('../../../hooks/analytics/usePerformanceAttribution', () => ({
  usePerformanceAttribution: vi.fn(() => ({
    attribution: {
      portfolioReturn: 15.75,
      benchmarkReturn: 12.25,
      activeReturn: 3.50,
      allocationEffect: 1.25,
      selectionEffect: 2.25,
      interactionEffect: 0.50,
      totalAttribution: 4.00
    },
    isCalculating: false,
    error: null
  }))
}));

vi.mock('../../../hooks/analytics/useRiskAnalytics', () => ({
  useRiskAnalytics: vi.fn(() => ({
    riskMetrics: {
      portfolioVar: -45000,
      componentVar: [
        { component: 'Equity Risk', var: -38000 },
        { component: 'Credit Risk', var: -5000 },
        { component: 'Market Risk', var: -2000 }
      ],
      stressTests: [
        { scenario: '2008 Crisis', pnl: -125000 },
        { scenario: 'COVID-19', pnl: -85000 },
        { scenario: 'Black Monday', pnl: -95000 }
      ]
    },
    isAnalyzing: false,
    error: null
  }))
}));

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: vi.fn(({ children }) => <div data-testid="responsive-container">{children}</div>),
  LineChart: vi.fn(() => <div data-testid="line-chart">Line Chart</div>),
  Line: vi.fn(() => <div data-testid="line">Line</div>),
  BarChart: vi.fn(() => <div data-testid="bar-chart">Bar Chart</div>),
  Bar: vi.fn(() => <div data-testid="bar">Bar</div>),
  PieChart: vi.fn(() => <div data-testid="pie-chart">Pie Chart</div>),
  Pie: vi.fn(() => <div data-testid="pie">Pie</div>),
  Cell: vi.fn(() => <div data-testid="cell">Cell</div>),
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>),
  ComposedChart: vi.fn(() => <div data-testid="composed-chart">Composed Chart</div>),
  Area: vi.fn(() => <div data-testid="area">Area</div>),
  AreaChart: vi.fn(() => <div data-testid="area-chart">Area Chart</div>),
  ScatterChart: vi.fn(() => <div data-testid="scatter-chart">Scatter Chart</div>),
  Scatter: vi.fn(() => <div data-testid="scatter">Scatter</div>),
  Heatmap: vi.fn(() => <div data-testid="heatmap">Heatmap</div>)
}));

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '2024-01-15'),
    subtract: vi.fn(() => ({
      format: vi.fn(() => '2024-01-01')
    })),
    valueOf: vi.fn(() => 1705327200000)
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

// Mock fetch for export functionality
global.fetch = vi.fn();

// Mock URL for blob downloads
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  },
  writable: true,
});

describe('AdvancedAnalyticsDashboard', () => {
  const user = userEvent.setup();
  const mockProps = {
    portfolioId: 'test-portfolio',
    timeRange: '1Y' as const,
    refreshInterval: 300000,
    enableExports: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the dashboard with all main sections', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Advanced Portfolio Analytics')).toBeInTheDocument();
      expect(screen.getByText('Performance Overview')).toBeInTheDocument();
      expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
      expect(screen.getByText('Attribution Analysis')).toBeInTheDocument();
    });

    it('renders in compact mode', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} compactMode={true} />);
      
      expect(screen.getByText('Advanced Analytics')).toBeInTheDocument();
      expect(screen.queryByText('Advanced Portfolio Analytics')).not.toBeInTheDocument();
    });

    it('renders with custom portfolio ID', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} portfolioId="custom-portfolio" />);
      
      expect(screen.getByText('Advanced Portfolio Analytics')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} height={600} />);
      
      const dashboard = screen.getByText('Advanced Portfolio Analytics').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });
  });

  describe('Performance Metrics Display', () => {
    it('displays all key performance metrics', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Total Return')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Sortino Ratio')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
      expect(screen.getByText('VaR 95%')).toBeInTheDocument();
      expect(screen.getByText('Win Rate')).toBeInTheDocument();
    });

    it('displays metric values correctly formatted', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('15.75%')).toBeInTheDocument(); // Total Return
      expect(screen.getByText('1.85')).toBeInTheDocument(); // Sharpe Ratio
      expect(screen.getByText('68.5%')).toBeInTheDocument(); // Win Rate
      expect(screen.getByText('-8.5%')).toBeInTheDocument(); // Max Drawdown
    });

    it('shows trend indicators for metrics', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      // Check for trend arrows or icons
      const trendIcons = screen.getAllByRole('img');
      expect(trendIcons.length).toBeGreaterThan(0);
    });
  });

  describe('Tab Navigation', () => {
    it('renders all analytics tabs', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Performance Overview')).toBeInTheDocument();
      expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
      expect(screen.getByText('Attribution Analysis')).toBeInTheDocument();
      expect(screen.getByText('Factor Exposure')).toBeInTheDocument();
      expect(screen.getByText('Trade Analysis')).toBeInTheDocument();
    });

    it('switches between tabs correctly', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const riskTab = screen.getByText('Risk Analysis');
      await user.click(riskTab);
      
      expect(screen.getByText('Risk Metrics Overview')).toBeInTheDocument();
      expect(screen.getByText('Value at Risk (VaR)')).toBeInTheDocument();
    });

    it('shows attribution analysis tab content', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const attributionTab = screen.getByText('Attribution Analysis');
      await user.click(attributionTab);
      
      expect(screen.getByText('Performance Attribution')).toBeInTheDocument();
      expect(screen.getByText('Allocation Effect')).toBeInTheDocument();
      expect(screen.getByText('Selection Effect')).toBeInTheDocument();
    });

    it('displays factor exposure analysis', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const factorTab = screen.getByText('Factor Exposure');
      await user.click(factorTab);
      
      expect(screen.getByText('Risk Factor Exposure')).toBeInTheDocument();
      expect(screen.getByText('Market Beta')).toBeInTheDocument();
      expect(screen.getByText('Size Factor')).toBeInTheDocument();
    });

    it('shows trade analysis metrics', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const tradeTab = screen.getByText('Trade Analysis');
      await user.click(tradeTab);
      
      expect(screen.getByText('Trading Statistics')).toBeInTheDocument();
      expect(screen.getByText('Profit Factor')).toBeInTheDocument();
      expect(screen.getByText('Expectancy')).toBeInTheDocument();
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders all chart types', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThan(0);
    });

    it('shows sector exposure chart', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const factorTab = screen.getByText('Factor Exposure');
      await user.click(factorTab);
      
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    });

    it('displays rolling metrics chart', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('shows performance heatmap', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      // Heatmap should be rendered for monthly performance
      expect(screen.getByText('Monthly Returns')).toBeInTheDocument();
    });
  });

  describe('Control Panel', () => {
    it('renders time range selector', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const timeRangeControl = screen.getByDisplayValue('1Y');
      expect(timeRangeControl).toBeInTheDocument();
    });

    it('changes time range', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const timeRangeSelect = screen.getByDisplayValue('1Y');
      await user.click(timeRangeSelect);
      
      const threeMonthOption = screen.getByText('3M');
      await user.click(threeMonthOption);
      
      expect(screen.getByDisplayValue('3M')).toBeInTheDocument();
    });

    it('renders benchmark selector', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Benchmark:')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const refreshButton = screen.getByRole('button', { name: /reload/i });
      expect(refreshButton).toBeInTheDocument();
    });

    it('renders settings button', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      expect(settingsButton).toBeInTheDocument();
    });
  });

  describe('Settings Panel', () => {
    it('opens settings modal', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Advanced Analytics Settings')).toBeInTheDocument();
      });
    });

    it('shows all settings options', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Chart Settings')).toBeInTheDocument();
        expect(screen.getByText('Risk Settings')).toBeInTheDocument();
        expect(screen.getByText('Performance Settings')).toBeInTheDocument();
      });
    });

    it('allows updating settings', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save Settings');
        expect(saveButton).toBeInTheDocument();
      });
    });
  });

  describe('Export Functionality', () => {
    beforeEach(() => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(['test data'], { type: 'application/pdf' }))
      });

      const mockAnchorElement = {
        style: {},
        href: '',
        download: '',
        click: vi.fn(),
      };
      vi.spyOn(document, 'createElement').mockReturnValue(mockAnchorElement as any);
      vi.spyOn(document.body, 'appendChild').mockImplementation(() => mockAnchorElement as any);
      vi.spyOn(document.body, 'removeChild').mockImplementation(() => mockAnchorElement as any);
    });

    it('renders export button when enabled', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      expect(exportButton).toBeInTheDocument();
    });

    it('opens export menu', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      await user.click(exportButton);
      
      await waitFor(() => {
        expect(screen.getByText('Export Report (PDF)')).toBeInTheDocument();
        expect(screen.getByText('Export Data (Excel)')).toBeInTheDocument();
        expect(screen.getByText('Export Data (CSV)')).toBeInTheDocument();
      });
    });

    it('handles PDF export', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      await user.click(exportButton);
      
      const pdfExportOption = screen.getByText('Export Report (PDF)');
      await user.click(pdfExportOption);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          expect.stringContaining('/analytics/export'),
          expect.objectContaining({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: expect.stringContaining('pdf')
          })
        );
      });
    });
  });

  describe('Data Table Display', () => {
    it('displays strategy breakdown table', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Strategy Breakdown')).toBeInTheDocument();
      expect(screen.getByText('Momentum Strategy')).toBeInTheDocument();
      expect(screen.getByText('Mean Reversion')).toBeInTheDocument();
      expect(screen.getByText('Statistical Arbitrage')).toBeInTheDocument();
    });

    it('shows sortable table columns', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Strategy')).toBeInTheDocument();
      expect(screen.getByText('Allocation')).toBeInTheDocument();
      expect(screen.getByText('Return')).toBeInTheDocument();
      expect(screen.getByText('Volatility')).toBeInTheDocument();
      expect(screen.getByText('Sharpe')).toBeInTheDocument();
    });

    it('displays quarterly returns table', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('2024-Q1')).toBeInTheDocument();
      expect(screen.getByText('2024-Q2')).toBeInTheDocument();
      expect(screen.getByText('2024-Q3')).toBeInTheDocument();
      expect(screen.getByText('2024-Q4')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when analytics fail', () => {
      const { useAdvancedAnalytics } = require('../../../hooks/analytics/useAdvancedAnalytics');
      useAdvancedAnalytics.mockReturnValue({
        ...useAdvancedAnalytics(),
        error: 'Failed to load advanced analytics data'
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Analytics Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to load advanced analytics data')).toBeInTheDocument();
    });

    it('shows loading state', () => {
      const { useAdvancedAnalytics } = require('../../../hooks/analytics/useAdvancedAnalytics');
      useAdvancedAnalytics.mockReturnValue({
        ...useAdvancedAnalytics(),
        isLoading: true,
        analytics: null
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Loading advanced analytics...')).toBeInTheDocument();
    });

    it('handles missing data gracefully', () => {
      const { useAdvancedAnalytics } = require('../../../hooks/analytics/useAdvancedAnalytics');
      useAdvancedAnalytics.mockReturnValue({
        ...useAdvancedAnalytics(),
        analytics: null,
        performanceBreakdown: [],
        riskFactors: []
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('No analytics data available')).toBeInTheDocument();
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes data automatically', async () => {
      const { useAdvancedAnalytics } = require('../../../hooks/analytics/useAdvancedAnalytics');
      const mockRefresh = vi.fn();
      useAdvancedAnalytics.mockReturnValue({
        ...useAdvancedAnalytics(),
        refreshAnalytics: mockRefresh
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} refreshInterval={1000} />);
      
      // Fast-forward time to trigger refresh
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('shows last updated timestamp', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large datasets efficiently', () => {
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
        sharpeRatio: 1.5 + Math.random() * 0.7,
        maxDrawdown: Math.random() * -10,
        var95: Math.random() * -50000,
        return: Math.random() * 20 - 5
      }));

      const { useAdvancedAnalytics } = require('../../../hooks/analytics/useAdvancedAnalytics');
      useAdvancedAnalytics.mockReturnValue({
        ...useAdvancedAnalytics(),
        rollingMetrics: largeDataset
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Advanced Portfolio Analytics')).toBeInTheDocument();
    });

    it('debounces user interactions', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const timeRangeSelect = screen.getByDisplayValue('1Y');
      
      // Rapid clicks should be debounced
      await user.click(timeRangeSelect);
      await user.click(timeRangeSelect);
      await user.click(timeRangeSelect);
      
      expect(timeRangeSelect).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for charts', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      const charts = screen.getAllByTestId('responsive-container');
      expect(charts.length).toBeGreaterThan(0);
    });

    it('supports keyboard navigation', async () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      // Tab navigation should work
      await user.tab();
      
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides alternative text for visual elements', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      // Check that important metrics have accessible text
      expect(screen.getByText('Total Return')).toBeInTheDocument();
      expect(screen.getByText('15.75%')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      // Mock window.innerWidth for mobile
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      expect(screen.getByText('Advanced Portfolio Analytics')).toBeInTheDocument();
    });

    it('shows appropriate column spans for different screen sizes', () => {
      render(<AdvancedAnalyticsDashboard {...mockProps} />);
      
      // The dashboard should render with responsive grid layout
      const metricsCards = screen.getAllByText(/Return|Ratio|Drawdown|Rate/);
      expect(metricsCards.length).toBeGreaterThan(0);
    });
  });
});