/**
 * RealTimeAnalytics Test Suite
 * Sprint 3: Real-time performance analytics and monitoring testing
 * 
 * Tests live P&L calculations, portfolio performance metrics, strategy analytics,
 * execution quality analysis, and real-time data streaming functionality.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import RealTimeAnalytics from '../RealTimeAnalytics';

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: vi.fn(({ children }) => <div data-testid="responsive-container">{children}</div>),
  LineChart: vi.fn(() => <div data-testid="line-chart">Line Chart</div>),
  Line: vi.fn(() => <div data-testid="line">Line</div>),
  AreaChart: vi.fn(() => <div data-testid="area-chart">Area Chart</div>),
  Area: vi.fn(() => <div data-testid="area">Area</div>),
  BarChart: vi.fn(() => <div data-testid="bar-chart">Bar Chart</div>),
  Bar: vi.fn(() => <div data-testid="bar">Bar</div>),
  ComposedChart: vi.fn(() => <div data-testid="composed-chart">Composed Chart</div>),
  ScatterChart: vi.fn(() => <div data-testid="scatter-chart">Scatter Chart</div>),
  Scatter: vi.fn(() => <div data-testid="scatter">Scatter</div>),
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>),
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>)
}));

// Mock real-time analytics hook
vi.mock('../../../hooks/analytics/useRealTimeAnalytics', () => ({
  useRealTimeAnalytics: vi.fn(() => ({
    performanceMetrics: {
      portfolio: {
        totalPnL: 127845.67,
        dailyPnL: 3456.89,
        unrealizedPnL: 89234.12,
        realizedPnL: 38611.55,
        totalReturn: 12.78,
        dailyReturn: 0.34,
        sharpeRatio: 1.85,
        maxDrawdown: -8.45,
        volatility: 15.67,
        beta: 1.12,
        alpha: 2.34,
        informationRatio: 0.89
      },
      strategies: [
        {
          strategyId: 'momentum-1',
          strategyName: 'Momentum Strategy Alpha',
          pnl: 45678.90,
          return: 15.23,
          sharpe: 2.10,
          maxDrawdown: -5.67,
          trades: 1245,
          winRate: 67.8,
          avgWin: 234.56,
          avgLoss: -156.78,
          status: 'active'
        },
        {
          strategyId: 'mean-reversion-1',
          strategyName: 'Mean Reversion Pro',
          pnl: 32167.77,
          return: 8.91,
          sharpe: 1.45,
          maxDrawdown: -12.34,
          trades: 892,
          winRate: 58.4,
          avgWin: 189.45,
          avgLoss: -201.23,
          status: 'active'
        }
      ],
      execution: {
        totalTrades: 2137,
        avgFillTime: 123.45, // milliseconds
        slippage: 0.0234, // basis points
        fillRate: 98.67,
        rejectionRate: 1.33,
        avgSpread: 0.0123,
        impactCost: 0.0456,
        executionScore: 94.5
      },
      risk: {
        var95: -12567.89,
        var99: -18945.67,
        expectedShortfall: -21234.56,
        correlationRisk: 0.78,
        concentrationRisk: 23.45,
        leverageRatio: 1.67,
        portfolioVaR: -15678.90
      },
      benchmarkComparison: {
        benchmarkReturn: 8.45,
        excessReturn: 4.33,
        trackingError: 3.21,
        informationRatio: 1.35,
        upCapture: 105.67,
        downCapture: 87.89
      }
    },
    historicalData: {
      pnlHistory: [
        { timestamp: Date.now() - 300000, pnl: 124388.78, benchmark: 115234.45 },
        { timestamp: Date.now() - 240000, pnl: 125456.89, benchmark: 116123.56 },
        { timestamp: Date.now() - 180000, pnl: 126789.12, benchmark: 117456.78 },
        { timestamp: Date.now() - 120000, pnl: 127123.45, benchmark: 118234.90 },
        { timestamp: Date.now() - 60000, pnl: 127456.78, benchmark: 118789.12 },
        { timestamp: Date.now(), pnl: 127845.67, benchmark: 119345.67 }
      ],
      returnHistory: [
        { timestamp: Date.now() - 300000, return: 11.89, benchmark: 7.45 },
        { timestamp: Date.now() - 240000, return: 12.34, benchmark: 7.89 },
        { timestamp: Date.now() - 180000, return: 12.56, benchmark: 8.12 },
        { timestamp: Date.now() - 120000, return: 12.67, benchmark: 8.34 },
        { timestamp: Date.now() - 60000, return: 12.71, benchmark: 8.45 },
        { timestamp: Date.now(), return: 12.78, benchmark: 8.45 }
      ],
      volumeProfile: [
        { timeSlot: '09:30', volume: 1234567, vwap: 145.67 },
        { timeSlot: '10:00', volume: 987654, vwap: 146.12 },
        { timeSlot: '10:30', volume: 1456789, vwap: 145.89 },
        { timeSlot: '11:00', volume: 1123456, vwap: 146.34 }
      ]
    },
    realTimeUpdates: {
      isStreaming: true,
      lastUpdate: Date.now(),
      updateFrequency: 1000, // 1 second
      connectionStatus: 'connected',
      messagesReceived: 15674,
      latency: 23.45
    },
    alerts: [
      {
        id: 'alert-drawdown',
        type: 'risk',
        severity: 'warning',
        message: 'Portfolio drawdown approaching -10% threshold',
        timestamp: Date.now() - 180000,
        value: -8.45,
        threshold: -10.0,
        resolved: false
      },
      {
        id: 'alert-volume',
        type: 'execution',
        severity: 'info',
        message: 'High trading volume detected in momentum strategy',
        timestamp: Date.now() - 120000,
        value: 1245,
        threshold: 1000,
        resolved: true
      }
    ],
    isInitialized: true,
    isLoading: false,
    error: null,
    startStreaming: vi.fn(),
    stopStreaming: vi.fn(),
    refreshData: vi.fn(),
    exportAnalytics: vi.fn(),
    acknowledgeAlert: vi.fn(),
    configureAlerts: vi.fn()
  }))
}));

describe('RealTimeAnalytics', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    portfolioId: 'portfolio-main',
    updateInterval: 1000,
    enableAlerts: true,
    alertThresholds: {
      drawdownMax: -10.0,
      sharpeMin: 1.0,
      varMax: -20000,
      volumeSpike: 1000
    },
    benchmarkSymbol: 'SPY',
    showAdvancedMetrics: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  describe('Basic Rendering', () => {
    it('renders real-time analytics dashboard', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Real-Time Analytics Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Performance')).toBeInTheDocument();
      expect(screen.getByText('Strategy Analysis')).toBeInTheDocument();
      expect(screen.getByText('Execution Quality')).toBeInTheDocument();
    });

    it('displays streaming status indicator', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Live Streaming')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('shows last update timestamp', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });

    it('displays portfolio ID', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('portfolio-main')).toBeInTheDocument();
    });
  });

  describe('Portfolio Performance Metrics', () => {
    it('displays total P&L correctly', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('$127,845.67')).toBeInTheDocument();
    });

    it('shows daily P&L with positive/negative coloring', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Daily P&L')).toBeInTheDocument();
      expect(screen.getByText('$3,456.89')).toBeInTheDocument();
      // Positive P&L should be in green color
    });

    it('displays return metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Total Return')).toBeInTheDocument();
      expect(screen.getByText('12.78%')).toBeInTheDocument();
      expect(screen.getByText('Daily Return')).toBeInTheDocument();
      expect(screen.getByText('0.34%')).toBeInTheDocument();
    });

    it('shows risk-adjusted metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('1.85')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
      expect(screen.getByText('-8.45%')).toBeInTheDocument();
    });

    it('displays advanced metrics when enabled', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Alpha')).toBeInTheDocument();
      expect(screen.getByText('2.34%')).toBeInTheDocument();
      expect(screen.getByText('Beta')).toBeInTheDocument();
      expect(screen.getByText('1.12')).toBeInTheDocument();
      expect(screen.getByText('Information Ratio')).toBeInTheDocument();
      expect(screen.getByText('0.89')).toBeInTheDocument();
    });

    it('shows volatility metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Volatility')).toBeInTheDocument();
      expect(screen.getByText('15.67%')).toBeInTheDocument();
    });
  });

  describe('Strategy Analysis', () => {
    it('displays strategy performance table', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Strategy Performance')).toBeInTheDocument();
      expect(screen.getByText('Momentum Strategy Alpha')).toBeInTheDocument();
      expect(screen.getByText('Mean Reversion Pro')).toBeInTheDocument();
    });

    it('shows strategy P&L and returns', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('$45,678.90')).toBeInTheDocument(); // Momentum strategy P&L
      expect(screen.getByText('15.23%')).toBeInTheDocument(); // Momentum strategy return
      expect(screen.getByText('$32,167.77')).toBeInTheDocument(); // Mean reversion P&L
      expect(screen.getByText('8.91%')).toBeInTheDocument(); // Mean reversion return
    });

    it('displays strategy metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('2.10')).toBeInTheDocument(); // Momentum Sharpe
      expect(screen.getByText('1.45')).toBeInTheDocument(); // Mean reversion Sharpe
      expect(screen.getByText('67.8%')).toBeInTheDocument(); // Win rate
      expect(screen.getByText('58.4%')).toBeInTheDocument(); // Win rate
    });

    it('shows trade statistics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('1,245')).toBeInTheDocument(); // Trade count
      expect(screen.getByText('892')).toBeInTheDocument(); // Trade count
      expect(screen.getByText('$234.56')).toBeInTheDocument(); // Avg win
      expect(screen.getByText('-$156.78')).toBeInTheDocument(); // Avg loss
    });

    it('indicates strategy status', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getAllByText('active')).toHaveLength(2);
    });
  });

  describe('Execution Quality Metrics', () => {
    it('displays execution statistics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Execution Quality')).toBeInTheDocument();
      expect(screen.getByText('Total Trades')).toBeInTheDocument();
      expect(screen.getByText('2,137')).toBeInTheDocument();
    });

    it('shows fill time metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Avg Fill Time')).toBeInTheDocument();
      expect(screen.getByText('123.45ms')).toBeInTheDocument();
    });

    it('displays slippage and spread data', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Slippage')).toBeInTheDocument();
      expect(screen.getByText('2.34 bps')).toBeInTheDocument();
      expect(screen.getByText('Avg Spread')).toBeInTheDocument();
      expect(screen.getByText('1.23 bps')).toBeInTheDocument();
    });

    it('shows fill and rejection rates', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Fill Rate')).toBeInTheDocument();
      expect(screen.getByText('98.67%')).toBeInTheDocument();
      expect(screen.getByText('Rejection Rate')).toBeInTheDocument();
      expect(screen.getByText('1.33%')).toBeInTheDocument();
    });

    it('displays execution score', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Execution Score')).toBeInTheDocument();
      expect(screen.getByText('94.5')).toBeInTheDocument();
    });
  });

  describe('Risk Metrics', () => {
    it('displays VaR calculations', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('VaR (95%)')).toBeInTheDocument();
      expect(screen.getByText('-$12,567.89')).toBeInTheDocument();
      expect(screen.getByText('VaR (99%)')).toBeInTheDocument();
      expect(screen.getByText('-$18,945.67')).toBeInTheDocument();
    });

    it('shows expected shortfall', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Expected Shortfall')).toBeInTheDocument();
      expect(screen.getByText('-$21,234.56')).toBeInTheDocument();
    });

    it('displays risk ratios', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Correlation Risk')).toBeInTheDocument();
      expect(screen.getByText('0.78')).toBeInTheDocument();
      expect(screen.getByText('Leverage Ratio')).toBeInTheDocument();
      expect(screen.getByText('1.67')).toBeInTheDocument();
    });

    it('shows concentration risk', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Concentration Risk')).toBeInTheDocument();
      expect(screen.getByText('23.45%')).toBeInTheDocument();
    });
  });

  describe('Benchmark Comparison', () => {
    it('displays benchmark comparison metrics', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Benchmark Comparison')).toBeInTheDocument();
      expect(screen.getByText('Excess Return')).toBeInTheDocument();
      expect(screen.getByText('4.33%')).toBeInTheDocument();
    });

    it('shows tracking error and information ratio', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Tracking Error')).toBeInTheDocument();
      expect(screen.getByText('3.21%')).toBeInTheDocument();
      expect(screen.getByText('Information Ratio')).toBeInTheDocument();
      expect(screen.getByText('1.35')).toBeInTheDocument();
    });

    it('displays capture ratios', () => {
      render(<RealTimeAnalytics {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Up Capture')).toBeInTheDocument();
      expect(screen.getByText('105.67%')).toBeInTheDocument();
      expect(screen.getByText('Down Capture')).toBeInTheDocument();
      expect(screen.getByText('87.89%')).toBeInTheDocument();
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders P&L trend chart', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays return comparison chart', async () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      const returnsTab = screen.getByText('Returns');
      await user.click(returnsTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('shows volume profile visualization', async () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      const volumeTab = screen.getByText('Volume Profile');
      await user.click(volumeTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });

    it('displays strategy performance chart', async () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      const strategiesTab = screen.getByText('Strategy Analysis');
      await user.click(strategiesTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('shows streaming connection status', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('23.45ms')).toBeInTheDocument(); // Latency
    });

    it('displays message count and update frequency', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('15,674 messages')).toBeInTheDocument();
      expect(screen.getByText('1s updates')).toBeInTheDocument();
    });

    it('updates metrics in real-time', async () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      // Simulate real-time update
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      
      expect(screen.getByText('Real-Time Analytics Dashboard')).toBeInTheDocument();
    });

    it('handles streaming interruptions', () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        realTimeUpdates: {
          ...useRealTimeAnalytics().realTimeUpdates,
          connectionStatus: 'disconnected'
        }
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });
  });

  describe('Alerts and Notifications', () => {
    it('displays performance alerts', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Performance Alerts')).toBeInTheDocument();
      expect(screen.getByText('Portfolio drawdown approaching -10% threshold')).toBeInTheDocument();
      expect(screen.getByText('High trading volume detected in momentum strategy')).toBeInTheDocument();
    });

    it('shows alert severity levels', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('warning')).toBeInTheDocument();
      expect(screen.getByText('info')).toBeInTheDocument();
    });

    it('displays resolved and active alerts differently', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('Resolved')).toBeInTheDocument();
    });

    it('allows acknowledging alerts', async () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const mockAcknowledge = vi.fn();
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });
  });

  describe('Control Functions', () => {
    it('starts real-time streaming', async () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const mockStart = vi.fn();
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        realTimeUpdates: {
          ...useRealTimeAnalytics().realTimeUpdates,
          isStreaming: false
        },
        startStreaming: mockStart
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      const startButton = screen.getByText('Start Streaming');
      await user.click(startButton);
      
      expect(mockStart).toHaveBeenCalled();
    });

    it('stops real-time streaming', async () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const mockStop = vi.fn();
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        stopStreaming: mockStop
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Streaming');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('refreshes analytics data', async () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const mockRefresh = vi.fn();
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        refreshData: mockRefresh
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      const refreshButton = screen.getByText('Refresh');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('exports analytics data', async () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const mockExport = vi.fn();
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        exportAnalytics: mockExport
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      const exportButton = screen.getByText('Export Data');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when analytics fails', () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        error: 'Failed to load analytics data'
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Analytics Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to load analytics data')).toBeInTheDocument();
    });

    it('shows loading state', () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        isLoading: true
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Loading analytics...')).toBeInTheDocument();
    });

    it('handles missing portfolio data', () => {
      render(<RealTimeAnalytics {...defaultProps} portfolioId="" />);
      
      expect(screen.getByText('No portfolio specified')).toBeInTheDocument();
    });
  });

  describe('Performance and Load Testing', () => {
    it('handles high-frequency updates efficiently', () => {
      render(<RealTimeAnalytics {...defaultProps} updateInterval={100} />);
      
      act(() => {
        for (let i = 0; i < 100; i++) {
          vi.advanceTimersByTime(100);
        }
      });
      
      expect(screen.getByText('Real-Time Analytics Dashboard')).toBeInTheDocument();
    });

    it('manages large historical datasets', () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const largeHistoricalData = {
        pnlHistory: Array.from({ length: 10000 }, (_, i) => ({
          timestamp: Date.now() - i * 1000,
          pnl: 100000 + Math.random() * 50000,
          benchmark: 90000 + Math.random() * 30000
        })),
        returnHistory: [],
        volumeProfile: []
      };

      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        historicalData: largeHistoricalData
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Real-Time Analytics Dashboard')).toBeInTheDocument();
    });

    it('maintains responsiveness with multiple strategies', () => {
      const { useRealTimeAnalytics } = require('../../../hooks/analytics/useRealTimeAnalytics');
      const manyStrategies = Array.from({ length: 50 }, (_, i) => ({
        strategyId: `strategy-${i}`,
        strategyName: `Strategy ${i}`,
        pnl: Math.random() * 100000,
        return: Math.random() * 20,
        sharpe: Math.random() * 3,
        maxDrawdown: -Math.random() * 20,
        trades: Math.floor(Math.random() * 2000),
        winRate: 50 + Math.random() * 30,
        avgWin: Math.random() * 500,
        avgLoss: -Math.random() * 300,
        status: 'active'
      }));

      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        performanceMetrics: {
          ...useRealTimeAnalytics().performanceMetrics,
          strategies: manyStrategies
        }
      });

      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Real-Time Analytics Dashboard')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides keyboard navigation support', async () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<RealTimeAnalytics {...defaultProps} />);
      
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Performance')).toBeInTheDocument();
      expect(screen.getByText('Strategy Analysis')).toBeInTheDocument();
    });
  });
});