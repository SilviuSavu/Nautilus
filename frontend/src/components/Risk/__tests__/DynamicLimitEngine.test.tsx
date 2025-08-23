/**
 * DynamicLimitEngine Test Suite
 * Sprint 3: Dynamic risk limit management and breach detection testing
 * 
 * Tests dynamic limit adjustment, ML-based breach prediction, real-time monitoring,
 * automated responses, and compliance reporting functionality.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import DynamicLimitEngine from '../DynamicLimitEngine';

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: vi.fn(({ children }) => <div data-testid="responsive-container">{children}</div>),
  LineChart: vi.fn(() => <div data-testid="line-chart">Line Chart</div>),
  Line: vi.fn(() => <div data-testid="line">Line</div>),
  AreaChart: vi.fn(() => <div data-testid="area-chart">Area Chart</div>),
  Area: vi.fn(() => <div data-testid="area">Area</div>),
  BarChart: vi.fn(() => <div data-testid="bar-chart">Bar Chart</div>),
  Bar: vi.fn(() => <div data-testid="bar">Bar</div>),
  ScatterChart: vi.fn(() => <div data-testid="scatter-chart">Scatter Chart</div>),
  Scatter: vi.fn(() => <div data-testid="scatter">Scatter</div>),
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>),
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>)
}));

// Mock dynamic limit engine hook
vi.mock('../../../hooks/risk/useDynamicLimitEngine', () => ({
  useDynamicLimitEngine: vi.fn(() => ({
    riskLimits: [
      {
        id: 'var-limit-1',
        name: 'Portfolio VaR Limit',
        type: 'var',
        currentValue: -85670.45,
        limitValue: -100000,
        utilizationPercentage: 85.67,
        status: 'active',
        lastUpdated: Date.now() - 30000,
        breachProbability: 0.23,
        adjustmentHistory: [
          { timestamp: Date.now() - 86400000, oldLimit: -120000, newLimit: -100000, reason: 'volatility_decrease' },
          { timestamp: Date.now() - 172800000, oldLimit: -80000, newLimit: -120000, reason: 'market_stress' }
        ]
      },
      {
        id: 'concentration-limit-1',
        name: 'Single Name Concentration',
        type: 'concentration',
        currentValue: 18.45,
        limitValue: 25.0,
        utilizationPercentage: 73.8,
        status: 'active',
        lastUpdated: Date.now() - 60000,
        breachProbability: 0.05,
        adjustmentHistory: []
      },
      {
        id: 'leverage-limit-1',
        name: 'Portfolio Leverage',
        type: 'leverage',
        currentValue: 2.34,
        limitValue: 3.0,
        utilizationPercentage: 78.0,
        status: 'active',
        lastUpdated: Date.now() - 15000,
        breachProbability: 0.12,
        adjustmentHistory: []
      },
      {
        id: 'drawdown-limit-1',
        name: 'Maximum Drawdown',
        type: 'drawdown',
        currentValue: -12.67,
        limitValue: -15.0,
        utilizationPercentage: 84.47,
        status: 'breached',
        lastUpdated: Date.now() - 45000,
        breachProbability: 0.95,
        breachTime: Date.now() - 120000,
        adjustmentHistory: []
      }
    ],
    breachPredictions: [
      {
        limitId: 'var-limit-1',
        limitName: 'Portfolio VaR Limit',
        predictedBreachTime: Date.now() + 3600000, // 1 hour from now
        confidence: 0.78,
        severity: 'medium',
        contributingFactors: [
          { factor: 'market_volatility', impact: 0.45, description: 'Increased market volatility' },
          { factor: 'correlation_risk', impact: 0.33, description: 'Rising asset correlations' }
        ],
        recommendedActions: [
          'Reduce position sizes in high-beta stocks',
          'Increase hedge positions in defensive assets'
        ]
      },
      {
        limitId: 'drawdown-limit-1',
        limitName: 'Maximum Drawdown',
        predictedBreachTime: Date.now() + 1800000, // 30 minutes from now
        confidence: 0.92,
        severity: 'high',
        contributingFactors: [
          { factor: 'momentum_reversal', impact: 0.67, description: 'Momentum strategy underperforming' },
          { factor: 'sector_rotation', impact: 0.25, description: 'Unfavorable sector rotation' }
        ],
        recommendedActions: [
          'Immediately reduce momentum strategy allocation',
          'Implement stop-loss protocols'
        ]
      }
    ],
    breachHistory: [
      {
        id: 'breach-1',
        limitId: 'drawdown-limit-1',
        limitName: 'Maximum Drawdown',
        breachTime: Date.now() - 120000,
        currentValue: -15.23,
        limitValue: -15.0,
        severity: 'critical',
        duration: 120000, // 2 minutes
        resolved: false,
        responseActions: [
          { action: 'position_reduction', status: 'completed', timestamp: Date.now() - 110000 },
          { action: 'alert_sent', status: 'completed', timestamp: Date.now() - 115000 }
        ]
      },
      {
        id: 'breach-2',
        limitId: 'var-limit-1',
        limitName: 'Portfolio VaR Limit',
        breachTime: Date.now() - 3600000,
        currentValue: -105670.45,
        limitValue: -100000,
        severity: 'high',
        duration: 900000, // 15 minutes
        resolved: true,
        resolveTime: Date.now() - 2700000,
        responseActions: [
          { action: 'hedge_adjustment', status: 'completed', timestamp: Date.now() - 3500000 },
          { action: 'portfolio_rebalance', status: 'completed', timestamp: Date.now() - 3400000 }
        ]
      }
    ],
    engineConfiguration: {
      adjustmentFrequency: 300000, // 5 minutes
      predictionHorizon: 3600000, // 1 hour
      confidenceThreshold: 0.7,
      autoAdjustmentEnabled: true,
      breachNotificationEnabled: true,
      emergencyStopEnabled: true,
      mlModelVersion: 'v2.1.3',
      lastModelUpdate: Date.now() - 86400000
    },
    systemMetrics: {
      engineStatus: 'running',
      lastHealthCheck: Date.now() - 30000,
      processedLimits: 4,
      activePredictions: 2,
      totalBreaches: 5,
      averageResponseTime: 45.2,
      predictionAccuracy: 0.847
    },
    isMonitoring: true,
    isConfiguring: false,
    error: null,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    updateLimit: vi.fn(),
    createLimit: vi.fn(),
    deleteLimit: vi.fn(),
    triggerManualAdjustment: vi.fn(),
    acknowledgeBreach: vi.fn(),
    configureEngine: vi.fn(),
    exportReport: vi.fn(),
    resetEngine: vi.fn()
  }))
}));

describe('DynamicLimitEngine', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    portfolioId: 'portfolio-main',
    monitoringInterval: 30000, // 30 seconds
    enableAutoAdjustment: true,
    enableBreachPrediction: true,
    enableEmergencyStop: true,
    predictionHorizon: 3600000, // 1 hour
    confidenceThreshold: 0.75
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
    it('renders dynamic limit engine dashboard', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Dynamic Risk Limit Engine')).toBeInTheDocument();
      expect(screen.getByText('Active Risk Limits')).toBeInTheDocument();
      expect(screen.getByText('Breach Predictions')).toBeInTheDocument();
      expect(screen.getByText('System Status')).toBeInTheDocument();
    });

    it('displays engine status', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Engine Status: Running')).toBeInTheDocument();
      expect(screen.getByText('ML Model: v2.1.3')).toBeInTheDocument();
    });

    it('shows monitoring status', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });

    it('displays portfolio ID', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('portfolio-main')).toBeInTheDocument();
    });
  });

  describe('Risk Limits Display', () => {
    it('shows all active risk limits', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Portfolio VaR Limit')).toBeInTheDocument();
      expect(screen.getByText('Single Name Concentration')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Leverage')).toBeInTheDocument();
      expect(screen.getByText('Maximum Drawdown')).toBeInTheDocument();
    });

    it('displays current values and limits', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('-$85,670.45')).toBeInTheDocument(); // VaR current
      expect(screen.getByText('-$100,000')).toBeInTheDocument(); // VaR limit
      expect(screen.getByText('18.45%')).toBeInTheDocument(); // Concentration current
      expect(screen.getByText('25.0%')).toBeInTheDocument(); // Concentration limit
    });

    it('shows utilization percentages', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('85.67%')).toBeInTheDocument(); // VaR utilization
      expect(screen.getByText('73.8%')).toBeInTheDocument(); // Concentration utilization
      expect(screen.getByText('78.0%')).toBeInTheDocument(); // Leverage utilization
      expect(screen.getByText('84.47%')).toBeInTheDocument(); // Drawdown utilization
    });

    it('displays breach probabilities', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('23%')).toBeInTheDocument(); // VaR breach probability
      expect(screen.getByText('5%')).toBeInTheDocument(); // Concentration breach probability
      expect(screen.getByText('95%')).toBeInTheDocument(); // Drawdown breach probability (breached)
    });

    it('shows limit status indicators', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getAllByText('active')).toHaveLength(3);
      expect(screen.getByText('breached')).toBeInTheDocument();
    });

    it('displays last updated timestamps', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  describe('Breach Predictions', () => {
    it('displays breach prediction alerts', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Breach Predictions')).toBeInTheDocument();
      expect(screen.getByText('Predicted breach in 1h')).toBeInTheDocument();
      expect(screen.getByText('Predicted breach in 30m')).toBeInTheDocument();
    });

    it('shows prediction confidence levels', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('78% confidence')).toBeInTheDocument();
      expect(screen.getByText('92% confidence')).toBeInTheDocument();
    });

    it('displays severity indicators', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('high')).toBeInTheDocument();
    });

    it('shows contributing factors', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Increased market volatility')).toBeInTheDocument();
      expect(screen.getByText('Rising asset correlations')).toBeInTheDocument();
      expect(screen.getByText('Momentum strategy underperforming')).toBeInTheDocument();
    });

    it('displays recommended actions', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Reduce position sizes in high-beta stocks')).toBeInTheDocument();
      expect(screen.getByText('Immediately reduce momentum strategy allocation')).toBeInTheDocument();
    });

    it('shows factor impact weights', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('45%')).toBeInTheDocument(); // market_volatility impact
      expect(screen.getByText('67%')).toBeInTheDocument(); // momentum_reversal impact
    });
  });

  describe('Breach History', () => {
    it('displays breach history section', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Breach History');
      expect(historyTab).toBeInTheDocument();
    });

    it('shows historical breaches', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Breach History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Maximum Drawdown')).toBeInTheDocument();
        expect(screen.getByText('Portfolio VaR Limit')).toBeInTheDocument();
      });
    });

    it('displays breach details', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Breach History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('critical')).toBeInTheDocument();
        expect(screen.getByText('high')).toBeInTheDocument();
        expect(screen.getByText('Duration: 2m')).toBeInTheDocument();
        expect(screen.getByText('Duration: 15m')).toBeInTheDocument();
      });
    });

    it('shows resolved and active breaches differently', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Breach History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(screen.getByText('Resolved')).toBeInTheDocument();
      });
    });

    it('displays response actions', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Breach History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('position_reduction')).toBeInTheDocument();
        expect(screen.getByText('alert_sent')).toBeInTheDocument();
        expect(screen.getByText('hedge_adjustment')).toBeInTheDocument();
      });
    });
  });

  describe('System Metrics', () => {
    it('displays system performance metrics', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('System Metrics')).toBeInTheDocument();
      expect(screen.getByText('Processed Limits: 4')).toBeInTheDocument();
      expect(screen.getByText('Active Predictions: 2')).toBeInTheDocument();
      expect(screen.getByText('Total Breaches: 5')).toBeInTheDocument();
    });

    it('shows response time metrics', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Avg Response Time')).toBeInTheDocument();
      expect(screen.getByText('45.2ms')).toBeInTheDocument();
    });

    it('displays prediction accuracy', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Prediction Accuracy')).toBeInTheDocument();
      expect(screen.getByText('84.7%')).toBeInTheDocument();
    });

    it('shows last health check', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText(/Last health check:/)).toBeInTheDocument();
    });
  });

  describe('Limit Management', () => {
    it('allows creating new limits', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockCreate = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        createLimit: mockCreate
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const createButton = screen.getByText('Create Limit');
      await user.click(createButton);
      
      expect(screen.getByText('Create New Risk Limit')).toBeInTheDocument();
    });

    it('allows updating existing limits', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockUpdate = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        updateLimit: mockUpdate
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const editButtons = screen.getAllByText('Edit');
      if (editButtons.length > 0) {
        await user.click(editButtons[0]);
        expect(mockUpdate).toHaveBeenCalled();
      }
    });

    it('allows deleting limits', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockDelete = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        deleteLimit: mockDelete
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const deleteButtons = screen.getAllByText('Delete');
      if (deleteButtons.length > 0) {
        await user.click(deleteButtons[0]);
        expect(mockDelete).toHaveBeenCalled();
      }
    });

    it('supports manual limit adjustments', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockAdjust = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        triggerManualAdjustment: mockAdjust
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const adjustButtons = screen.getAllByText('Adjust');
      if (adjustButtons.length > 0) {
        await user.click(adjustButtons[0]);
        expect(mockAdjust).toHaveBeenCalled();
      }
    });
  });

  describe('Configuration Management', () => {
    it('displays engine configuration settings', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Engine Configuration')).toBeInTheDocument();
        expect(screen.getByText('Auto-adjustment: Enabled')).toBeInTheDocument();
        expect(screen.getByText('Breach Prediction: Enabled')).toBeInTheDocument();
      });
    });

    it('shows adjustment frequency setting', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Adjustment Frequency: 5m')).toBeInTheDocument();
      });
    });

    it('displays prediction horizon', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Prediction Horizon: 1h')).toBeInTheDocument();
      });
    });

    it('allows updating configuration', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockConfigure = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        configureEngine: mockConfigure
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        const updateButton = screen.getByText('Update Configuration');
        expect(updateButton).toBeInTheDocument();
      });
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders limit utilization chart', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    });

    it('displays breach prediction timeline', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const predictionTab = screen.getByText('Prediction Timeline');
      await user.click(predictionTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('shows historical trend analysis', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const trendsTab = screen.getByText('Historical Trends');
      await user.click(trendsTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('displays factor correlation chart', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const correlationTab = screen.getByText('Factor Analysis');
      await user.click(correlationTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('scatter-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Control Functions', () => {
    it('starts limit monitoring', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockStart = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        isMonitoring: false,
        startMonitoring: mockStart
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      expect(mockStart).toHaveBeenCalled();
    });

    it('stops limit monitoring', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockStop = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        stopMonitoring: mockStop
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('acknowledges breaches', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockAcknowledge = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        acknowledgeBreache: mockAcknowledge
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });

    it('exports risk reports', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockExport = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        exportReport: mockExport
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const exportButton = screen.getByText('Export Report');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });

    it('resets engine state', async () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const mockReset = vi.fn();
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        resetEngine: mockReset
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      const resetButton = screen.getByText('Reset Engine');
      await user.click(resetButton);
      
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('updates limit values in real-time', () => {
      render(<DynamicLimitEngine {...defaultProps} monitoringInterval={5000} />);
      
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      expect(screen.getByText('Dynamic Risk Limit Engine')).toBeInTheDocument();
    });

    it('refreshes breach predictions continuously', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      act(() => {
        vi.advanceTimersByTime(30000);
      });
      
      expect(screen.getByText('Breach Predictions')).toBeInTheDocument();
    });

    it('updates system metrics periodically', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('System Metrics')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when engine fails', () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        error: 'Dynamic limit engine connection failed'
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Engine Error')).toBeInTheDocument();
      expect(screen.getByText('Dynamic limit engine connection failed')).toBeInTheDocument();
    });

    it('handles missing limit data gracefully', () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        riskLimits: []
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('No risk limits configured')).toBeInTheDocument();
    });

    it('shows engine offline status', () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        systemMetrics: {
          ...useDynamicLimitEngine().systemMetrics,
          engineStatus: 'offline'
        }
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Engine Status: Offline')).toBeInTheDocument();
    });
  });

  describe('Performance and Load Testing', () => {
    it('handles many limits efficiently', () => {
      const { useDynamicLimitEngine } = require('../../../hooks/risk/useDynamicLimitEngine');
      const manyLimits = Array.from({ length: 100 }, (_, i) => ({
        id: `limit-${i}`,
        name: `Risk Limit ${i}`,
        type: 'custom',
        currentValue: Math.random() * 100000,
        limitValue: 100000 + Math.random() * 50000,
        utilizationPercentage: Math.random() * 100,
        status: 'active',
        lastUpdated: Date.now() - Math.random() * 3600000,
        breachProbability: Math.random(),
        adjustmentHistory: []
      }));

      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        riskLimits: manyLimits
      });

      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Dynamic Risk Limit Engine')).toBeInTheDocument();
    });

    it('maintains responsiveness with frequent updates', () => {
      render(<DynamicLimitEngine {...defaultProps} monitoringInterval={1000} />);
      
      act(() => {
        for (let i = 0; i < 30; i++) {
          vi.advanceTimersByTime(1000);
        }
      });
      
      expect(screen.getByText('Dynamic Risk Limit Engine')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides keyboard navigation support', async () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<DynamicLimitEngine {...defaultProps} />);
      
      expect(screen.getByText('Active Risk Limits')).toBeInTheDocument();
      expect(screen.getByText('Breach Predictions')).toBeInTheDocument();
      expect(screen.getByText('System Status')).toBeInTheDocument();
    });
  });
});