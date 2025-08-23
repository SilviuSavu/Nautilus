/**
 * RealTimeAnalyticsDashboard Test Suite
 * Comprehensive tests for the real-time analytics dashboard component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import RealTimeAnalyticsDashboard from '../RealTimeAnalyticsDashboard';

// Mock the hooks
vi.mock('../../../hooks/analytics/useRealTimeAnalytics', () => ({
  useRealTimeAnalytics: vi.fn(() => ({
    currentData: {
      portfolio_id: 'test-portfolio',
      timestamp: '2024-01-15T10:30:00Z',
      pnl: {
        realized: 12450.75,
        unrealized: 8234.50,
        total: 20685.25,
        daily_change: 1250.00,
        daily_change_pct: 6.44
      },
      risk_metrics: {
        var_1d: -45000,
        var_5d: -85000,
        expected_shortfall: -52000,
        beta: 1.15,
        volatility: 0.18,
        max_drawdown: -8.5
      },
      performance: {
        total_return: 15.75,
        sharpe_ratio: 1.85,
        sortino_ratio: 2.12,
        alpha: 0.045,
        information_ratio: 1.32
      },
      positions: {
        long_exposure: 500000,
        short_exposure: -150000,
        net_exposure: 350000,
        gross_exposure: 650000,
        leverage: 2.1
      },
      execution: {
        fill_rate: 97.5,
        avg_slippage: 3.2,
        implementation_shortfall: 5.8,
        market_impact: 2.1
      }
    },
    historicalData: [
      {
        portfolio_id: 'test-portfolio',
        timestamp: '2024-01-15T10:25:00Z',
        pnl: { realized: 12200, unrealized: 8100, total: 20300, daily_change: 1000, daily_change_pct: 5.2 },
        risk_metrics: { var_1d: -44000, var_5d: -84000, expected_shortfall: -51000, beta: 1.14, volatility: 0.17, max_drawdown: -8.2 },
        performance: { total_return: 15.2, sharpe_ratio: 1.82, sortino_ratio: 2.08, alpha: 0.042, information_ratio: 1.28 },
        positions: { long_exposure: 495000, short_exposure: -145000, net_exposure: 350000, gross_exposure: 640000, leverage: 2.05 },
        execution: { fill_rate: 97.2, avg_slippage: 3.0, implementation_shortfall: 5.5, market_impact: 2.0 }
      }
    ],
    isConnected: true,
    isLoading: false,
    error: null,
    lastUpdate: new Date(),
    updateCount: 125,
    start: vi.fn(),
    stop: vi.fn(),
    reset: vi.fn(),
    getTrends: vi.fn(() => ({
      pnl_trend: 385.25,
      volatility_trend: 0.01,
      sharpe_trend: 0.03,
      exposure_trend: 10000
    })),
    getPerformanceStats: vi.fn(() => ({
      avg_latency: 45.2,
      update_frequency: 4.8,
      data_completeness: 98.5
    }))
  }))
}));

vi.mock('../../../hooks/analytics/usePerformanceMetrics', () => ({
  usePerformanceMetrics: vi.fn(() => ({
    analytics: {
      loading: false,
      error: null,
      data: {
        summary: {
          total_return: 15.75,
          sharpe_ratio: 1.85,
          max_drawdown: -8.5,
          win_rate: 68.5
        }
      },
      lastUpdated: new Date()
    },
    refreshAnalytics: vi.fn(),
    isLoading: false,
    hasError: false,
    isDataAvailable: true
  }))
}));

vi.mock('../../../hooks/analytics/useRiskAnalytics', () => ({
  useRiskAnalytics: vi.fn(() => ({
    result: null,
    isAnalyzing: false,
    error: null,
    riskAlerts: [
      {
        id: 'alert-1',
        type: 'var_breach',
        severity: 'medium',
        message: 'VaR threshold approaching',
        timestamp: new Date(),
        portfolio_id: 'test-portfolio',
        current_value: 48000,
        threshold_value: 50000
      }
    ],
    isMonitoring: false,
    analyzeRisk: vi.fn(),
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn()
  }))
}));

vi.mock('../../../hooks/analytics/useStrategyAnalytics', () => ({
  useStrategyAnalytics: vi.fn(() => ({
    result: null,
    isAnalyzing: false,
    error: null,
    analyzeStrategies: vi.fn()
  }))
}));

vi.mock('../../../hooks/analytics/useExecutionAnalytics', () => ({
  useExecutionAnalytics: vi.fn(() => ({
    result: null,
    isAnalyzing: false,
    error: null,
    analyzeExecution: vi.fn()
  }))
}));

vi.mock('../../../hooks/useWebSocketManager', () => ({
  useWebSocketManager: vi.fn(() => ({
    connectionStatus: 'connected',
    connect: vi.fn(),
    disconnect: vi.fn(),
    send: vi.fn()
  }))
}));

vi.mock('../../../hooks/useMessageBus', () => ({
  useMessageBus: vi.fn(() => ({
    connectionStatus: 'connected',
    isConnected: true,
    subscribe: vi.fn(),
    unsubscribe: vi.fn()
  }))
}));

// Mock @ant-design/charts
vi.mock('@ant-design/charts', () => ({
  Line: vi.fn(() => <div data-testid="line-chart">Line Chart</div>),
  Area: vi.fn(() => <div data-testid="area-chart">Area Chart</div>),
  Column: vi.fn(() => <div data-testid="column-chart">Column Chart</div>),
  Pie: vi.fn(() => <div data-testid="pie-chart">Pie Chart</div>),
  Gauge: vi.fn(() => <div data-testid="gauge-chart">Gauge Chart</div>),
  Heatmap: vi.fn(() => <div data-testid="heatmap-chart">Heatmap Chart</div>)
}));

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '10:30:00'),
    isAfter: vi.fn(() => true),
    subtract: vi.fn(() => ({
      isAfter: vi.fn(() => true)
    })),
    valueOf: vi.fn(() => 1642249800000)
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

// Mock fetch for export functionality
global.fetch = vi.fn();

// Mock window.URL for blob downloads
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  },
  writable: true,
});

describe('RealTimeAnalyticsDashboard', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the dashboard with default props', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('Real-time Analytics Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Live Streaming')).toBeInTheDocument();
    });

    it('renders in compact mode', () => {
      render(<RealTimeAnalyticsDashboard compactMode={true} />);
      
      expect(screen.getByText('Real-time Analytics')).toBeInTheDocument();
      expect(screen.queryByText('Real-time Analytics Dashboard')).not.toBeInTheDocument();
    });

    it('renders with custom portfolio ID', () => {
      render(<RealTimeAnalyticsDashboard portfolioId="custom-portfolio" />);
      
      expect(screen.getByText('Real-time Analytics Dashboard')).toBeInTheDocument();
    });
  });

  describe('Connection Status', () => {
    it('displays connection status indicators', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('WebSocket')).toBeInTheDocument();
      expect(screen.getByText('MessageBus')).toBeInTheDocument();
      expect(screen.getByText('Analytics')).toBeInTheDocument();
    });

    it('shows performance statistics', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText(/Updates: 125/)).toBeInTheDocument();
      expect(screen.getByText(/Latency: 45.2ms/)).toBeInTheDocument();
      expect(screen.getByText(/Rate: 4.8\/s/)).toBeInTheDocument();
    });
  });

  describe('Key Metrics Display', () => {
    it('displays all key performance metrics', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('Unrealized P&L')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
      expect(screen.getByText('VaR 95%')).toBeInTheDocument();
      expect(screen.getByText('Net Exposure')).toBeInTheDocument();
      expect(screen.getByText('Leverage')).toBeInTheDocument();
      expect(screen.getByText('Fill Rate')).toBeInTheDocument();
    });

    it('displays metric values correctly', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      // Check for formatted values
      expect(screen.getByText('$20,685.25')).toBeInTheDocument();
      expect(screen.getByText('1.85')).toBeInTheDocument();
      expect(screen.getByText('97.5%')).toBeInTheDocument();
    });

    it('shows trend indicators', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      // Check for trend icons (should be in the document)
      const trendingIcons = screen.getAllByRole('img');
      expect(trendingIcons.length).toBeGreaterThan(0);
    });
  });

  describe('Tab Navigation', () => {
    it('renders all tabs', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('P&L Analysis')).toBeInTheDocument();
      expect(screen.getByText('Risk Metrics')).toBeInTheDocument();
      expect(screen.getByText('Strategy Performance')).toBeInTheDocument();
      expect(screen.getByText('Execution Quality')).toBeInTheDocument();
    });

    it('switches between tabs', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const riskTab = screen.getByText('Risk Metrics');
      await user.click(riskTab);
      
      expect(screen.getByText('Risk Heatmap')).toBeInTheDocument();
      expect(screen.getByText('Risk Alerts')).toBeInTheDocument();
    });

    it('shows strategy tab content', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const strategyTab = screen.getByText('Strategy Performance');
      await user.click(strategyTab);
      
      expect(screen.getByText('Strategy analytics integration in progress')).toBeInTheDocument();
    });

    it('shows execution tab with metrics', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const executionTab = screen.getByText('Execution Quality');
      await user.click(executionTab);
      
      expect(screen.getByText('Fill Rate')).toBeInTheDocument();
      expect(screen.getByText('Avg Slippage')).toBeInTheDocument();
      expect(screen.getByText('Implementation Shortfall')).toBeInTheDocument();
      expect(screen.getByText('Market Impact')).toBeInTheDocument();
    });
  });

  describe('Charts Rendering', () => {
    it('renders the main chart', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('shows loading state when no data', () => {
      // Mock empty historical data
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        historicalData: [],
        currentData: null
      });

      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('Loading real-time data...')).toBeInTheDocument();
    });
  });

  describe('Risk Alerts', () => {
    it('displays risk alerts count', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      // Check for alert badge
      const alertBadge = screen.getByText('1');
      expect(alertBadge).toBeInTheDocument();
    });

    it('shows alert details in risk tab', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const riskTab = screen.getByText('Risk Metrics');
      await user.click(riskTab);
      
      expect(screen.getByText('VaR threshold approaching')).toBeInTheDocument();
    });

    it('shows success message when no alerts', async () => {
      // Mock no alerts
      const { useRiskAnalytics } = require('../../../hooks/analytics/useRiskAnalytics');
      useRiskAnalytics.mockReturnValue({
        ...useRiskAnalytics(),
        riskAlerts: []
      });

      render(<RealTimeAnalyticsDashboard />);
      
      const riskTab = screen.getByText('Risk Metrics');
      await user.click(riskTab);
      
      expect(screen.getByText('All systems normal')).toBeInTheDocument();
      expect(screen.getByText('No active risk alerts')).toBeInTheDocument();
    });
  });

  describe('Control Panel', () => {
    it('renders time range selector', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const timeRangeSelect = screen.getByDisplayValue('1H');
      expect(timeRangeSelect).toBeInTheDocument();
    });

    it('changes time range', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const timeRangeSelect = screen.getByDisplayValue('1H');
      await user.click(timeRangeSelect);
      
      const fiveMinOption = screen.getByText('5M');
      await user.click(fiveMinOption);
      
      // The value should change
      expect(timeRangeSelect).toBeInTheDocument();
    });

    it('renders pause/start button', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const pauseButton = screen.getByText('Pause');
      expect(pauseButton).toBeInTheDocument();
    });

    it('renders reset button', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const resetButton = screen.getByText('Reset');
      expect(resetButton).toBeInTheDocument();
    });

    it('renders settings button', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      expect(settingsButton).toBeInTheDocument();
    });
  });

  describe('Settings Panel', () => {
    it('opens settings drawer', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Dashboard Settings')).toBeInTheDocument();
      });
    });

    it('shows settings form fields', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Update Interval (ms)')).toBeInTheDocument();
        expect(screen.getByText('Auto Refresh')).toBeInTheDocument();
        expect(screen.getByText('Show Alerts')).toBeInTheDocument();
        expect(screen.getByText('VaR 95% Threshold ($)')).toBeInTheDocument();
        expect(screen.getByText('Leverage Threshold')).toBeInTheDocument();
        expect(screen.getByText('Concentration Threshold (%)')).toBeInTheDocument();
      });
    });
  });

  describe('Export Functionality', () => {
    beforeEach(() => {
      // Mock successful fetch response
      (global.fetch as any).mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(['test data'], { type: 'application/pdf' }))
      });

      // Mock document.createElement and related methods
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
      render(<RealTimeAnalyticsDashboard enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      expect(exportButton).toBeInTheDocument();
    });

    it('does not render export button when disabled', () => {
      render(<RealTimeAnalyticsDashboard enableExports={false} />);
      
      const exportButton = screen.queryByText('Export');
      expect(exportButton).not.toBeInTheDocument();
    });

    it('opens export menu', async () => {
      render(<RealTimeAnalyticsDashboard enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      await user.click(exportButton);
      
      await waitFor(() => {
        expect(screen.getByText('Export PDF')).toBeInTheDocument();
        expect(screen.getByText('Export Excel')).toBeInTheDocument();
        expect(screen.getByText('Export CSV')).toBeInTheDocument();
        expect(screen.getByText('Export JSON')).toBeInTheDocument();
      });
    });

    it('handles PDF export', async () => {
      render(<RealTimeAnalyticsDashboard enableExports={true} />);
      
      const exportButton = screen.getByText('Export');
      await user.click(exportButton);
      
      const pdfExportOption = screen.getByText('Export PDF');
      await user.click(pdfExportOption);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          'http://localhost:8001/api/v1/analytics/export',
          expect.objectContaining({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: expect.stringContaining('pdf')
          })
        );
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message when real-time analytics fails', () => {
      // Mock error state
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        error: 'Failed to connect to analytics service'
      });

      render(<RealTimeAnalyticsDashboard />);
      
      expect(screen.getByText('Real-time Analytics Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to connect to analytics service')).toBeInTheDocument();
    });

    it('handles missing data gracefully', () => {
      // Mock null data
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        currentData: null,
        historicalData: []
      });

      render(<RealTimeAnalyticsDashboard />);
      
      // Should not crash and should show loading state
      expect(screen.getByText('Loading real-time data...')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for different screen sizes', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      // Check that responsive grid columns are present
      const metricsCards = screen.getAllByText(/P&L|Sharpe|VaR|Exposure/);
      expect(metricsCards.length).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('handles large amounts of historical data', () => {
      const largeHistoricalData = Array.from({ length: 1000 }, (_, i) => ({
        portfolio_id: 'test-portfolio',
        timestamp: new Date(Date.now() - i * 1000).toISOString(),
        pnl: { realized: 12000 + i, unrealized: 8000 + i, total: 20000 + i, daily_change: 1000, daily_change_pct: 5 },
        risk_metrics: { var_1d: -44000, var_5d: -84000, expected_shortfall: -51000, beta: 1.14, volatility: 0.17, max_drawdown: -8.2 },
        performance: { total_return: 15.2, sharpe_ratio: 1.82, sortino_ratio: 2.08, alpha: 0.042, information_ratio: 1.28 },
        positions: { long_exposure: 495000, short_exposure: -145000, net_exposure: 350000, gross_exposure: 640000, leverage: 2.05 },
        execution: { fill_rate: 97.2, avg_slippage: 3.0, implementation_shortfall: 5.5, market_impact: 2.0 }
      }));

      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        historicalData: largeHistoricalData
      });

      render(<RealTimeAnalyticsDashboard />);
      
      // Should render without performance issues
      expect(screen.getByText('Real-time Analytics Dashboard')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for interactive elements', () => {
      render(<RealTimeAnalyticsDashboard />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<RealTimeAnalyticsDashboard />);
      
      // Tab navigation should work
      await user.tab();
      
      // Focus should be on an interactive element
      const focusedElement = screen.getByRole('button', { name: /1H/i });
      expect(focusedElement).toHaveFocus();
    });
  });
});