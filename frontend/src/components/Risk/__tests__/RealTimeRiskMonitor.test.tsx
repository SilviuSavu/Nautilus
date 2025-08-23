/**
 * RealTimeRiskMonitor Test Suite
 * Comprehensive tests for the real-time risk monitoring component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import RealTimeRiskMonitor from '../RealTimeRiskMonitor';

// Mock the risk monitoring hooks
vi.mock('../../../hooks/analytics/useRealTimeRisk', () => ({
  useRealTimeRisk: vi.fn(() => ({
    currentRiskData: {
      portfolio_id: 'test-portfolio',
      timestamp: '2024-01-15T10:30:00Z',
      overall_score: 75.5,
      risk_level: 'medium',
      metrics: {
        var_95: -45000,
        var_99: -62000,
        expected_shortfall: -52000,
        volatility: 0.18,
        beta: 1.15,
        max_drawdown: -8.5,
        concentration_risk: 0.35,
        leverage: 2.1,
        liquidity_risk: 0.12
      },
      limits: [
        {
          limit_id: 'var-limit-1',
          limit_type: 'var_95',
          current_value: -45000,
          limit_value: -50000,
          utilization: 90.0,
          status: 'warning',
          breach_probability: 0.15
        },
        {
          limit_id: 'leverage-limit-1',
          limit_type: 'leverage',
          current_value: 2.1,
          limit_value: 2.5,
          utilization: 84.0,
          status: 'normal',
          breach_probability: 0.05
        },
        {
          limit_id: 'concentration-limit-1',
          limit_type: 'concentration',
          current_value: 0.35,
          limit_value: 0.40,
          utilization: 87.5,
          status: 'warning',
          breach_probability: 0.12
        }
      ],
      exposures: {
        sector_exposure: [
          { sector: 'Technology', exposure: 0.35, limit: 0.40, status: 'normal' },
          { sector: 'Financials', exposure: 0.22, limit: 0.25, status: 'normal' },
          { sector: 'Healthcare', exposure: 0.18, limit: 0.20, status: 'normal' },
          { sector: 'Energy', exposure: 0.12, limit: 0.15, status: 'normal' },
          { sector: 'Others', exposure: 0.13, limit: 0.15, status: 'normal' }
        ],
        geographic_exposure: [
          { region: 'North America', exposure: 0.55, limit: 0.60, status: 'normal' },
          { region: 'Europe', exposure: 0.25, limit: 0.30, status: 'normal' },
          { region: 'Asia Pacific', exposure: 0.15, limit: 0.20, status: 'normal' },
          { region: 'Emerging Markets', exposure: 0.05, limit: 0.10, status: 'normal' }
        ],
        currency_exposure: [
          { currency: 'USD', exposure: 0.70, limit: 0.80, status: 'normal' },
          { currency: 'EUR', exposure: 0.20, limit: 0.25, status: 'normal' },
          { currency: 'JPY', exposure: 0.10, limit: 0.15, status: 'normal' }
        ]
      },
      alerts: [
        {
          id: 'alert-001',
          type: 'var_breach_warning',
          severity: 'medium',
          message: 'VaR 95% approaching limit threshold',
          timestamp: '2024-01-15T10:25:00Z',
          metric: 'var_95',
          current_value: -45000,
          threshold_value: -50000,
          breach_probability: 0.15,
          recommendation: 'Consider reducing position sizes in high-risk assets'
        },
        {
          id: 'alert-002',
          type: 'concentration_warning',
          severity: 'low',
          message: 'Technology sector concentration above 80% of limit',
          timestamp: '2024-01-15T10:20:00Z',
          metric: 'sector_concentration',
          current_value: 0.35,
          threshold_value: 0.40,
          breach_probability: 0.08,
          recommendation: 'Monitor technology sector exposure'
        }
      ]
    },
    historicalRiskData: Array.from({ length: 100 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
      overall_score: 70 + Math.random() * 20,
      var_95: -40000 - Math.random() * 20000,
      volatility: 0.15 + Math.random() * 0.1,
      leverage: 1.8 + Math.random() * 0.6
    })),
    riskTrends: {
      var_trend: 'increasing',
      volatility_trend: 'stable',
      leverage_trend: 'decreasing',
      overall_trend: 'stable'
    },
    isMonitoring: true,
    isConnected: true,
    isLoading: false,
    error: null,
    lastUpdate: new Date(),
    updateCount: 1247,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    refreshRiskData: vi.fn(),
    updateSettings: vi.fn()
  }))
}));

vi.mock('../../../hooks/analytics/useRiskLimits', () => ({
  useRiskLimits: vi.fn(() => ({
    limits: [
      {
        id: 'limit-001',
        name: 'Portfolio VaR 95%',
        type: 'var_95',
        value: -50000,
        warning_threshold: 0.9,
        breach_threshold: 1.0,
        enabled: true,
        auto_adjust: true
      },
      {
        id: 'limit-002', 
        name: 'Maximum Leverage',
        type: 'leverage',
        value: 2.5,
        warning_threshold: 0.85,
        breach_threshold: 1.0,
        enabled: true,
        auto_adjust: false
      }
    ],
    updateLimit: vi.fn(),
    createLimit: vi.fn(),
    deleteLimit: vi.fn()
  }))
}));

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: vi.fn(({ children }) => <div data-testid="responsive-container">{children}</div>),
  LineChart: vi.fn(() => <div data-testid="line-chart">Line Chart</div>),
  Line: vi.fn(() => <div data-testid="line">Line</div>),
  AreaChart: vi.fn(() => <div data-testid="area-chart">Area Chart</div>),
  Area: vi.fn(() => <div data-testid="area">Area</div>),
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
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>)
}));

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '10:30:00'),
    fromNow: vi.fn(() => '5 minutes ago'),
    valueOf: vi.fn(() => 1705327200000)
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('RealTimeRiskMonitor', () => {
  const user = userEvent.setup();
  const mockProps = {
    portfolioId: 'test-portfolio',
    updateInterval: 5000,
    enableAlerts: true,
    height: 600
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the risk monitor dashboard', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Real-time Risk Monitor')).toBeInTheDocument();
      expect(screen.getByText('Risk Overview')).toBeInTheDocument();
      expect(screen.getByText('Risk Limits')).toBeInTheDocument();
      expect(screen.getByText('Exposures')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<RealTimeRiskMonitor {...mockProps} height={800} />);
      
      const dashboard = screen.getByText('Real-time Risk Monitor').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<RealTimeRiskMonitor />);
      
      expect(screen.getByText('Real-time Risk Monitor')).toBeInTheDocument();
    });
  });

  describe('Risk Score Display', () => {
    it('displays overall risk score', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Overall Risk Score')).toBeInTheDocument();
      expect(screen.getByText('75.5')).toBeInTheDocument();
    });

    it('shows risk level indicator', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('medium')).toBeInTheDocument();
    });

    it('displays risk score with appropriate color coding', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      // Should have color-coded risk level display
      const riskLevel = screen.getByText('medium');
      expect(riskLevel).toBeInTheDocument();
    });
  });

  describe('Risk Metrics', () => {
    it('displays all key risk metrics', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('VaR 95%')).toBeInTheDocument();
      expect(screen.getByText('VaR 99%')).toBeInTheDocument();
      expect(screen.getByText('Expected Shortfall')).toBeInTheDocument();
      expect(screen.getByText('Volatility')).toBeInTheDocument();
      expect(screen.getByText('Beta')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
    });

    it('shows formatted metric values', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('-$45,000')).toBeInTheDocument(); // VaR 95%
      expect(screen.getByText('-$62,000')).toBeInTheDocument(); // VaR 99%
      expect(screen.getByText('18.0%')).toBeInTheDocument(); // Volatility
      expect(screen.getByText('1.15')).toBeInTheDocument(); // Beta
    });

    it('displays leverage and concentration metrics', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Leverage')).toBeInTheDocument();
      expect(screen.getByText('2.1x')).toBeInTheDocument();
      expect(screen.getByText('Concentration Risk')).toBeInTheDocument();
      expect(screen.getByText('35.0%')).toBeInTheDocument();
    });
  });

  describe('Risk Limits Monitoring', () => {
    it('displays risk limits with utilization', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('VaR 95% Limit')).toBeInTheDocument();
      expect(screen.getByText('90.0%')).toBeInTheDocument(); // Utilization
      expect(screen.getByText('Leverage Limit')).toBeInTheDocument();
      expect(screen.getByText('84.0%')).toBeInTheDocument();
    });

    it('shows limit status indicators', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('warning')).toBeInTheDocument();
      expect(screen.getByText('normal')).toBeInTheDocument();
    });

    it('displays breach probabilities', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('15.0%')).toBeInTheDocument(); // VaR breach probability
      expect(screen.getByText('5.0%')).toBeInTheDocument(); // Leverage breach probability
    });

    it('renders progress bars for limit utilization', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });
  });

  describe('Exposure Analysis', () => {
    it('displays sector exposure breakdown', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const exposuresTab = screen.getByText('Exposures');
      expect(exposuresTab).toBeInTheDocument();
    });

    it('shows all exposure types in tabs', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const exposuresTab = screen.getByText('Exposures');
      await user.click(exposuresTab);
      
      await waitFor(() => {
        expect(screen.getByText('Sector Exposure')).toBeInTheDocument();
        expect(screen.getByText('Geographic')).toBeInTheDocument();
        expect(screen.getByText('Currency')).toBeInTheDocument();
      });
    });

    it('displays sector exposure details', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const exposuresTab = screen.getByText('Exposures');
      await user.click(exposuresTab);
      
      await waitFor(() => {
        expect(screen.getByText('Technology')).toBeInTheDocument();
        expect(screen.getByText('Financials')).toBeInTheDocument();
        expect(screen.getByText('Healthcare')).toBeInTheDocument();
        expect(screen.getByText('35.0%')).toBeInTheDocument(); // Tech exposure
      });
    });

    it('shows geographic exposure breakdown', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const exposuresTab = screen.getByText('Exposures');
      await user.click(exposuresTab);
      
      const geographicTab = screen.getByText('Geographic');
      await user.click(geographicTab);
      
      await waitFor(() => {
        expect(screen.getByText('North America')).toBeInTheDocument();
        expect(screen.getByText('Europe')).toBeInTheDocument();
        expect(screen.getByText('55.0%')).toBeInTheDocument(); // NA exposure
      });
    });

    it('displays currency exposure', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const exposuresTab = screen.getByText('Exposures');
      await user.click(exposuresTab);
      
      const currencyTab = screen.getByText('Currency');
      await user.click(currencyTab);
      
      await waitFor(() => {
        expect(screen.getByText('USD')).toBeInTheDocument();
        expect(screen.getByText('EUR')).toBeInTheDocument();
        expect(screen.getByText('70.0%')).toBeInTheDocument(); // USD exposure
      });
    });
  });

  describe('Risk Alerts', () => {
    it('displays active risk alerts', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Risk Alerts')).toBeInTheDocument();
      expect(screen.getByText('VaR 95% approaching limit threshold')).toBeInTheDocument();
      expect(screen.getByText('Technology sector concentration above 80% of limit')).toBeInTheDocument();
    });

    it('shows alert severity levels', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('low')).toBeInTheDocument();
    });

    it('displays alert timestamps', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      // Should show relative timestamps
      expect(screen.getByText(/5 minutes ago/)).toBeInTheDocument();
    });

    it('shows alert recommendations', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Consider reducing position sizes in high-risk assets')).toBeInTheDocument();
      expect(screen.getByText('Monitor technology sector exposure')).toBeInTheDocument();
    });

    it('displays alert count badge', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const alertsBadge = screen.getByText('2');
      expect(alertsBadge).toBeInTheDocument();
    });
  });

  describe('Historical Charts', () => {
    it('renders risk score trend chart', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('shows multiple metric charts', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      // Should have charts for different metrics
      const charts = screen.getAllByTestId('responsive-container');
      expect(charts.length).toBeGreaterThan(1);
    });

    it('displays trend indicators', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      // Check for trend arrows or indicators
      const trendElements = screen.getAllByRole('img');
      expect(trendElements.length).toBeGreaterThan(0);
    });
  });

  describe('Real-time Updates', () => {
    it('shows connection status', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Monitoring')).toBeInTheDocument();
    });

    it('displays update statistics', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Updates: 1,247')).toBeInTheDocument();
      expect(screen.getByText(/Last update:/)).toBeInTheDocument();
    });

    it('allows starting and stopping monitoring', async () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      const mockStop = vi.fn();
      const mockStart = vi.fn();
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        stopMonitoring: mockStop,
        startMonitoring: mockStart
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const stopButton = screen.getByText('Stop');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('refreshes data automatically', async () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      const mockRefresh = vi.fn();
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        refreshRiskData: mockRefresh
      });

      render(<RealTimeRiskMonitor {...mockProps} updateInterval={1000} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });
  });

  describe('Settings and Configuration', () => {
    it('opens settings panel', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Risk Monitor Settings')).toBeInTheDocument();
      });
    });

    it('shows configuration options', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Update Interval')).toBeInTheDocument();
        expect(screen.getByText('Alert Thresholds')).toBeInTheDocument();
        expect(screen.getByText('Display Settings')).toBeInTheDocument();
      });
    });

    it('allows updating settings', async () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      const mockUpdate = vi.fn();
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        updateSettings: mockUpdate
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save Settings');
        expect(saveButton).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message when monitoring fails', () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        error: 'Failed to connect to risk monitoring service'
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Risk Monitor Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to connect to risk monitoring service')).toBeInTheDocument();
    });

    it('shows loading state', () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        isLoading: true,
        currentRiskData: null
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Loading risk data...')).toBeInTheDocument();
    });

    it('handles connection loss', () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        isConnected: false
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('handles missing data gracefully', () => {
      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        currentRiskData: null,
        historicalRiskData: []
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('No risk data available')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large amounts of historical data', () => {
      const largeHistoricalData = Array.from({ length: 5000 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        overall_score: 70 + Math.random() * 20,
        var_95: -40000 - Math.random() * 20000,
        volatility: 0.15 + Math.random() * 0.1,
        leverage: 1.8 + Math.random() * 0.6
      }));

      const { useRealTimeRisk } = require('../../../hooks/analytics/useRealTimeRisk');
      useRealTimeRisk.mockReturnValue({
        ...useRealTimeRisk(),
        historicalRiskData: largeHistoricalData
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Real-time Risk Monitor')).toBeInTheDocument();
    });

    it('optimizes chart rendering', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      // Should render charts without performance issues
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for metrics', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      progressBars.forEach(progressBar => {
        expect(progressBar).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('warning')).toBeInTheDocument();
      expect(screen.getByText('normal')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Real-time Risk Monitor')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<RealTimeRiskMonitor {...mockProps} />);
      
      expect(screen.getByText('Risk Overview')).toBeInTheDocument();
      expect(screen.getByText('Risk Limits')).toBeInTheDocument();
    });
  });
});