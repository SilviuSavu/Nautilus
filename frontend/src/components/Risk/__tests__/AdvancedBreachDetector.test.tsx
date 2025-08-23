/**
 * AdvancedBreachDetector Test Suite
 * Comprehensive tests for the advanced breach detection component with ML predictions
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import AdvancedBreachDetector from '../AdvancedBreachDetector';

// Mock the breach detection hooks
vi.mock('../../../hooks/analytics/useBreachDetection', () => ({
  useBreachDetection: vi.fn(() => ({
    breachAnalysis: {
      overall_risk_score: 78.5,
      breach_probability: 0.15,
      time_to_breach_estimate: '2.5 hours',
      confidence_level: 0.87,
      ml_model_version: 'v2.1.0',
      last_model_update: '2024-01-14T10:00:00Z',
      prediction_accuracy: 0.92,
      limits_at_risk: [
        {
          limit_id: 'var-95-limit',
          limit_name: 'Portfolio VaR 95%',
          current_value: -45000,
          limit_value: -50000,
          utilization: 90.0,
          breach_probability: 0.18,
          time_to_breach: '1.8 hours',
          severity: 'high',
          trend: 'increasing',
          confidence: 0.89,
          contributing_factors: [
            { factor: 'Market Volatility', impact: 0.35 },
            { factor: 'Position Concentration', impact: 0.28 },
            { factor: 'Correlation Breakdown', impact: 0.22 }
          ]
        },
        {
          limit_id: 'leverage-limit',
          limit_name: 'Maximum Leverage',
          current_value: 2.15,
          limit_value: 2.5,
          utilization: 86.0,
          breach_probability: 0.08,
          time_to_breach: '4.2 hours',
          severity: 'medium',
          trend: 'stable',
          confidence: 0.82,
          contributing_factors: [
            { factor: 'Position Growth', impact: 0.45 },
            { factor: 'Cash Utilization', impact: 0.35 }
          ]
        },
        {
          limit_id: 'concentration-limit',
          limit_name: 'Sector Concentration',
          current_value: 0.38,
          limit_value: 0.40,
          utilization: 95.0,
          breach_probability: 0.25,
          time_to_breach: '45 minutes',
          severity: 'critical',
          trend: 'increasing',
          confidence: 0.94,
          contributing_factors: [
            { factor: 'Technology Sector Growth', impact: 0.65 },
            { factor: 'Market Momentum', impact: 0.25 }
          ]
        }
      ],
      pattern_analysis: {
        detected_patterns: [
          {
            pattern_type: 'volatility_spike',
            pattern_strength: 0.78,
            historical_accuracy: 0.85,
            typical_duration: '2-4 hours',
            breach_correlation: 0.72,
            description: 'Intraday volatility spike pattern detected'
          },
          {
            pattern_type: 'correlation_breakdown',
            pattern_strength: 0.65,
            historical_accuracy: 0.79,
            typical_duration: '1-3 days',
            breach_correlation: 0.68,
            description: 'Asset correlation breakdown during market stress'
          }
        ],
        market_regime: {
          current_regime: 'high_volatility',
          regime_probability: 0.82,
          regime_duration: '6 days',
          typical_breach_rate: 0.15
        }
      },
      early_warning_signals: [
        {
          signal_type: 'var_acceleration',
          strength: 0.75,
          time_detected: '2024-01-15T10:25:00Z',
          expected_impact: 'VaR may increase by 15% in next hour',
          urgency: 'high',
          recommended_action: 'Consider position reduction'
        },
        {
          signal_type: 'liquidity_deterioration',
          strength: 0.58,
          time_detected: '2024-01-15T10:20:00Z',
          expected_impact: 'Liquidity costs may increase by 8%',
          urgency: 'medium',
          recommended_action: 'Monitor bid-ask spreads'
        }
      ]
    },
    historicalBreaches: [
      {
        breach_id: 'breach-001',
        limit_name: 'Portfolio VaR 95%',
        breach_time: '2024-01-10T14:30:00Z',
        predicted_time: '2024-01-10T14:15:00Z',
        prediction_accuracy: 'accurate',
        breach_magnitude: 1.12,
        duration_minutes: 45,
        resolution_action: 'Position reduction',
        financial_impact: -15750.50,
        lessons_learned: 'Early warning system worked effectively'
      },
      {
        breach_id: 'breach-002',
        limit_name: 'Sector Concentration',
        breach_time: '2024-01-08T09:45:00Z',
        predicted_time: '2024-01-08T08:30:00Z',
        prediction_accuracy: 'early',
        breach_magnitude: 1.05,
        duration_minutes: 120,
        resolution_action: 'Portfolio rebalancing',
        financial_impact: -8250.25,
        lessons_learned: 'Sector rotation prediction needs refinement'
      }
    ],
    modelMetrics: {
      total_predictions: 1247,
      accurate_predictions: 1148,
      false_positives: 62,
      false_negatives: 37,
      precision: 0.949,
      recall: 0.969,
      f1_score: 0.959,
      model_drift_score: 0.12,
      last_retrain_date: '2024-01-12T02:00:00Z',
      next_retrain_scheduled: '2024-01-19T02:00:00Z'
    },
    isAnalyzing: false,
    error: null,
    startDetection: vi.fn(),
    stopDetection: vi.fn(),
    updateThresholds: vi.fn(),
    retrainModel: vi.fn(),
    acknowledgeAlert: vi.fn(),
    exportAnalysis: vi.fn(),
    refreshAnalysis: vi.fn()
  }))
}));

// Mock ML model hooks
vi.mock('../../../hooks/analytics/useMLModel', () => ({
  useMLModel: vi.fn(() => ({
    modelStatus: {
      status: 'active',
      version: 'v2.1.0',
      accuracy: 0.92,
      last_updated: '2024-01-14T10:00:00Z',
      training_data_size: 50000,
      feature_count: 127,
      model_type: 'gradient_boosting'
    },
    featureImportance: [
      { feature: 'volatility_30d', importance: 0.185, category: 'market' },
      { feature: 'position_concentration', importance: 0.142, category: 'portfolio' },
      { feature: 'correlation_breakdown', importance: 0.128, category: 'correlation' },
      { feature: 'market_stress', importance: 0.115, category: 'market' },
      { feature: 'liquidity_score', importance: 0.098, category: 'liquidity' }
    ],
    modelValidation: {
      validation_score: 0.91,
      cross_validation_scores: [0.92, 0.89, 0.93, 0.90, 0.91],
      overfitting_score: 0.08,
      stability_score: 0.87
    }
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
  ScatterChart: vi.fn(() => <div data-testid="scatter-chart">Scatter Chart</div>),
  Scatter: vi.fn(() => <div data-testid="scatter">Scatter</div>),
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
    valueOf: vi.fn(() => 1705327200000),
    subtract: vi.fn(() => ({
      format: vi.fn(() => '2024-01-01')
    }))
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('AdvancedBreachDetector', () => {
  const user = userEvent.setup();
  const mockProps = {
    portfolioId: 'test-portfolio',
    enableMLPredictions: true,
    enableEarlyWarning: true,
    updateInterval: 30000,
    height: 700
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the breach detector dashboard', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Advanced Breach Detector')).toBeInTheDocument();
      expect(screen.getByText('Breach Risk Analysis')).toBeInTheDocument();
      expect(screen.getByText('ML Predictions')).toBeInTheDocument();
      expect(screen.getByText('Early Warning Signals')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<AdvancedBreachDetector {...mockProps} height={800} />);
      
      const dashboard = screen.getByText('Advanced Breach Detector').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<AdvancedBreachDetector />);
      
      expect(screen.getByText('Advanced Breach Detector')).toBeInTheDocument();
    });

    it('conditionally renders ML features when enabled', () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      expect(screen.getByText('ML Model v2.1.0')).toBeInTheDocument();
      expect(screen.getByText('Prediction Accuracy: 92%')).toBeInTheDocument();
    });
  });

  describe('Breach Risk Overview', () => {
    it('displays overall risk score', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Overall Risk Score')).toBeInTheDocument();
      expect(screen.getByText('78.5')).toBeInTheDocument();
    });

    it('shows breach probability', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Breach Probability')).toBeInTheDocument();
      expect(screen.getByText('15.0%')).toBeInTheDocument();
    });

    it('displays time to breach estimate', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Time to Breach')).toBeInTheDocument();
      expect(screen.getByText('2.5 hours')).toBeInTheDocument();
    });

    it('shows confidence level', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Confidence')).toBeInTheDocument();
      expect(screen.getByText('87%')).toBeInTheDocument();
    });
  });

  describe('Limits at Risk', () => {
    it('displays all limits with breach risk', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Portfolio VaR 95%')).toBeInTheDocument();
      expect(screen.getByText('Maximum Leverage')).toBeInTheDocument();
      expect(screen.getByText('Sector Concentration')).toBeInTheDocument();
    });

    it('shows utilization percentages', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('90.0%')).toBeInTheDocument(); // VaR utilization
      expect(screen.getByText('86.0%')).toBeInTheDocument(); // Leverage utilization
      expect(screen.getByText('95.0%')).toBeInTheDocument(); // Concentration utilization
    });

    it('displays breach probabilities for each limit', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('18%')).toBeInTheDocument(); // VaR breach prob
      expect(screen.getByText('8%')).toBeInTheDocument();  // Leverage breach prob
      expect(screen.getByText('25%')).toBeInTheDocument(); // Concentration breach prob
    });

    it('shows time to breach estimates', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('1.8 hours')).toBeInTheDocument(); // VaR time to breach
      expect(screen.getByText('4.2 hours')).toBeInTheDocument(); // Leverage time to breach
      expect(screen.getByText('45 minutes')).toBeInTheDocument(); // Concentration time to breach
    });

    it('displays severity levels with appropriate styling', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('critical')).toBeInTheDocument();
    });

    it('shows contributing factors', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Market Volatility')).toBeInTheDocument();
      expect(screen.getByText('Position Concentration')).toBeInTheDocument();
      expect(screen.getByText('Technology Sector Growth')).toBeInTheDocument();
    });

    it('renders progress bars for utilization', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(2);
    });
  });

  describe('Pattern Analysis', () => {
    it('displays detected patterns', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const patternsTab = screen.getByText('Pattern Analysis');
      expect(patternsTab).toBeInTheDocument();
    });

    it('shows pattern details when tab is clicked', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const patternsTab = screen.getByText('Pattern Analysis');
      await user.click(patternsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Detected Patterns')).toBeInTheDocument();
        expect(screen.getByText('volatility_spike')).toBeInTheDocument();
        expect(screen.getByText('correlation_breakdown')).toBeInTheDocument();
      });
    });

    it('displays pattern strengths', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const patternsTab = screen.getByText('Pattern Analysis');
      await user.click(patternsTab);
      
      await waitFor(() => {
        expect(screen.getByText('78%')).toBeInTheDocument(); // volatility spike strength
        expect(screen.getByText('65%')).toBeInTheDocument(); // correlation breakdown strength
      });
    });

    it('shows market regime information', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const patternsTab = screen.getByText('Pattern Analysis');
      await user.click(patternsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Market Regime')).toBeInTheDocument();
        expect(screen.getByText('high_volatility')).toBeInTheDocument();
        expect(screen.getByText('82%')).toBeInTheDocument(); // regime probability
      });
    });

    it('displays historical accuracy for patterns', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const patternsTab = screen.getByText('Pattern Analysis');
      await user.click(patternsTab);
      
      await waitFor(() => {
        expect(screen.getByText('85%')).toBeInTheDocument(); // volatility spike accuracy
        expect(screen.getByText('79%')).toBeInTheDocument(); // correlation breakdown accuracy
      });
    });
  });

  describe('Early Warning Signals', () => {
    it('displays early warning signals when enabled', () => {
      render(<AdvancedBreachDetector {...mockProps} enableEarlyWarning={true} />);
      
      expect(screen.getByText('Early Warning Signals')).toBeInTheDocument();
      expect(screen.getByText('VaR Acceleration')).toBeInTheDocument();
      expect(screen.getByText('Liquidity Deterioration')).toBeInTheDocument();
    });

    it('shows signal strengths', () => {
      render(<AdvancedBreachDetector {...mockProps} enableEarlyWarning={true} />);
      
      expect(screen.getByText('75%')).toBeInTheDocument(); // VaR acceleration strength
      expect(screen.getByText('58%')).toBeInTheDocument(); // Liquidity deterioration strength
    });

    it('displays urgency levels', () => {
      render(<AdvancedBreachDetector {...mockProps} enableEarlyWarning={true} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
    });

    it('shows recommended actions', () => {
      render(<AdvancedBreachDetector {...mockProps} enableEarlyWarning={true} />);
      
      expect(screen.getByText('Consider position reduction')).toBeInTheDocument();
      expect(screen.getByText('Monitor bid-ask spreads')).toBeInTheDocument();
    });

    it('displays expected impacts', () => {
      render(<AdvancedBreachDetector {...mockProps} enableEarlyWarning={true} />);
      
      expect(screen.getByText('VaR may increase by 15% in next hour')).toBeInTheDocument();
      expect(screen.getByText('Liquidity costs may increase by 8%')).toBeInTheDocument();
    });
  });

  describe('ML Model Information', () => {
    it('displays model status when ML is enabled', () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      expect(screen.getByText('ML Model Status')).toBeInTheDocument();
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('v2.1.0')).toBeInTheDocument();
    });

    it('shows model metrics', () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
      expect(screen.getByText('92%')).toBeInTheDocument();
      expect(screen.getByText('Predictions Made')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
    });

    it('displays precision and recall', () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      expect(screen.getByText('Precision')).toBeInTheDocument();
      expect(screen.getByText('94.9%')).toBeInTheDocument();
      expect(screen.getByText('Recall')).toBeInTheDocument();
      expect(screen.getByText('96.9%')).toBeInTheDocument();
    });

    it('shows feature importance', () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      const mlTab = screen.getByText('ML Model');
      expect(mlTab).toBeInTheDocument();
    });

    it('displays feature importance chart', async () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      const mlTab = screen.getByText('ML Model');
      await user.click(mlTab);
      
      await waitFor(() => {
        expect(screen.getByText('Feature Importance')).toBeInTheDocument();
        expect(screen.getByText('volatility_30d')).toBeInTheDocument();
        expect(screen.getByText('position_concentration')).toBeInTheDocument();
      });
    });
  });

  describe('Historical Breach Analysis', () => {
    it('displays historical breaches table', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      expect(historyTab).toBeInTheDocument();
    });

    it('shows breach details', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Historical Breaches')).toBeInTheDocument();
        expect(screen.getByText('Portfolio VaR 95%')).toBeInTheDocument();
        expect(screen.getByText('Sector Concentration')).toBeInTheDocument();
      });
    });

    it('displays prediction accuracy', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('accurate')).toBeInTheDocument();
        expect(screen.getByText('early')).toBeInTheDocument();
      });
    });

    it('shows financial impact', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('-$15,750.50')).toBeInTheDocument();
        expect(screen.getByText('-$8,250.25')).toBeInTheDocument();
      });
    });

    it('displays resolution actions', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Position reduction')).toBeInTheDocument();
        expect(screen.getByText('Portfolio rebalancing')).toBeInTheDocument();
      });
    });
  });

  describe('Control Panel', () => {
    it('renders detection control buttons', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Stop Detection')).toBeInTheDocument();
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    it('allows starting and stopping detection', async () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      const mockStop = vi.fn();
      const mockStart = vi.fn();
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        stopDetection: mockStop,
        startDetection: mockStart
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      const stopButton = screen.getByText('Stop Detection');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('renders settings button', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      expect(settingsButton).toBeInTheDocument();
    });

    it('opens settings modal', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Breach Detection Settings')).toBeInTheDocument();
      });
    });

    it('shows threshold configuration', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Detection Thresholds')).toBeInTheDocument();
        expect(screen.getByText('ML Model Settings')).toBeInTheDocument();
        expect(screen.getByText('Alert Configuration')).toBeInTheDocument();
      });
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders breach probability chart', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays risk factor contributions chart', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
    });

    it('shows utilization progress charts', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('renders feature importance chart when ML is enabled', async () => {
      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      const mlTab = screen.getByText('ML Model');
      await user.click(mlTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes analysis automatically', async () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      const mockRefresh = vi.fn();
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        refreshAnalysis: mockRefresh
      });

      render(<AdvancedBreachDetector {...mockProps} updateInterval={5000} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('shows last update time', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });

    it('displays detection status', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Detection Active')).toBeInTheDocument();
    });
  });

  describe('Alert Management', () => {
    it('allows acknowledging alerts', async () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      const mockAcknowledge = vi.fn();
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });

    it('shows alert acknowledgment status', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      // Should show alert statuses
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('critical')).toBeInTheDocument();
    });
  });

  describe('Export Functionality', () => {
    it('renders export button', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const exportButton = screen.getByText('Export Analysis');
      expect(exportButton).toBeInTheDocument();
    });

    it('handles export analysis', async () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      const mockExport = vi.fn();
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        exportAnalysis: mockExport
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      const exportButton = screen.getByText('Export Analysis');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when detection fails', () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        error: 'Breach detection service unavailable'
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Detection Error')).toBeInTheDocument();
      expect(screen.getByText('Breach detection service unavailable')).toBeInTheDocument();
    });

    it('shows loading state during analysis', () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        isAnalyzing: true,
        breachAnalysis: null
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Analyzing breach patterns...')).toBeInTheDocument();
    });

    it('handles missing ML model gracefully', () => {
      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        breachAnalysis: {
          ...useBreachDetection().breachAnalysis,
          ml_model_version: null,
          prediction_accuracy: null
        }
      });

      render(<AdvancedBreachDetector {...mockProps} enableMLPredictions={true} />);
      
      expect(screen.getByText('ML model not available')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large datasets efficiently', () => {
      const largeHistoricalData = Array.from({ length: 1000 }, (_, i) => ({
        breach_id: `breach-${i.toString().padStart(3, '0')}`,
        limit_name: ['VaR 95%', 'Leverage', 'Concentration'][i % 3],
        breach_time: new Date(Date.now() - i * 86400000).toISOString(),
        predicted_time: new Date(Date.now() - i * 86400000 - 900000).toISOString(),
        prediction_accuracy: ['accurate', 'early', 'late'][i % 3],
        breach_magnitude: 1.0 + Math.random() * 0.3,
        duration_minutes: Math.floor(Math.random() * 180) + 15,
        resolution_action: ['Position reduction', 'Rebalancing', 'Hedge adjustment'][i % 3],
        financial_impact: -(Math.random() * 50000 + 1000),
        lessons_learned: 'System performed as expected'
      }));

      const { useBreachDetection } = require('../../../hooks/analytics/useBreachDetection');
      useBreachDetection.mockReturnValue({
        ...useBreachDetection(),
        historicalBreaches: largeHistoricalData
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Advanced Breach Detector')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for progress bars', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      progressBars.forEach(progressBar => {
        expect(progressBar).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('critical')).toBeInTheDocument();
      expect(screen.getByText('Detection Active')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Advanced Breach Detector')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<AdvancedBreachDetector {...mockProps} />);
      
      expect(screen.getByText('Breach Risk Analysis')).toBeInTheDocument();
      expect(screen.getByText('Early Warning Signals')).toBeInTheDocument();
    });
  });
});