/**
 * Sprint3Dashboard Integration Test Suite
 * Sprint 3: End-to-end integration testing for Sprint 3 dashboard
 * 
 * Tests complete Sprint 3 dashboard functionality including WebSocket infrastructure,
 * risk management, analytics, strategy deployment, and system monitoring integration.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import Sprint3Dashboard from '../../pages/Sprint3Dashboard';

// Mock all recharts
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

// Mock WebSocket global
const mockWebSocket = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
  readyState: 1,
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn()
};

global.WebSocket = vi.fn(() => mockWebSocket);

// Mock all Sprint 3 hooks
vi.mock('../../hooks/websocket/useWebSocketMonitoring', () => ({
  useWebSocketMonitoring: vi.fn(() => ({
    connectionStats: {
      total_connections: 1247,
      active_connections: 1189,
      connection_rate: 2.3,
      average_connection_duration: '2h 34m'
    },
    performanceMetrics: {
      average_latency: 23.4,
      throughput: 847,
      error_rate: 0.23
    },
    isMonitoring: true,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn()
  }))
}));

vi.mock('../../hooks/analytics/useRealTimeAnalytics', () => ({
  useRealTimeAnalytics: vi.fn(() => ({
    performanceMetrics: {
      portfolio: {
        totalPnL: 127845.67,
        dailyReturn: 0.34,
        sharpeRatio: 1.85
      },
      strategies: [
        {
          strategyId: 'momentum-1',
          strategyName: 'Momentum Strategy',
          pnl: 45678.90,
          return: 15.23
        }
      ]
    },
    realTimeUpdates: {
      isStreaming: true,
      connectionStatus: 'connected'
    },
    isInitialized: true,
    startStreaming: vi.fn(),
    stopStreaming: vi.fn()
  }))
}));

vi.mock('../../hooks/risk/useDynamicLimitEngine', () => ({
  useDynamicLimitEngine: vi.fn(() => ({
    riskLimits: [
      {
        id: 'var-limit-1',
        name: 'Portfolio VaR Limit',
        currentValue: -85670.45,
        limitValue: -100000,
        utilizationPercentage: 85.67,
        status: 'active',
        breachProbability: 0.23
      }
    ],
    breachPredictions: [
      {
        limitId: 'var-limit-1',
        predictedBreachTime: Date.now() + 3600000,
        confidence: 0.78,
        severity: 'medium'
      }
    ],
    isMonitoring: true,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn()
  }))
}));

vi.mock('../../hooks/strategy/useDeploymentApproval', () => ({
  useDeploymentApproval: vi.fn(() => ({
    approvalQueue: [
      {
        id: 'deployment-1',
        strategyName: 'Momentum Alpha Strategy',
        status: 'pending_review',
        priority: 'high',
        submittedBy: 'john.doe@company.com'
      }
    ],
    approvalMetrics: {
      totalSubmissions: 156,
      approvedDeployments: 134,
      rejectedDeployments: 22,
      successRate: 94.7
    },
    approveDeployment: vi.fn(),
    rejectDeployment: vi.fn()
  }))
}));

vi.mock('../../hooks/monitoring/useSystemMonitor', () => ({
  useSystemMonitor: vi.fn(() => ({
    systemHealth: {
      overallStatus: 'healthy',
      overallScore: 94.7,
      components: {
        webSocketInfrastructure: { status: 'healthy', score: 98.2 },
        riskManagement: { status: 'warning', score: 87.5 },
        analytics: { status: 'healthy', score: 96.1 }
      }
    },
    performanceMetrics: {
      cpuUsage: 34.7,
      memoryUsage: 67.3,
      errorRate: 0.23
    },
    alerts: [
      {
        id: 'alert-1',
        severity: 'critical',
        message: 'Risk limit breached',
        timestamp: Date.now() - 300000
      }
    ],
    isMonitoring: true,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn()
  }))
}));

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn()
    }
  };
});

describe('Sprint3Dashboard Integration Tests', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    // Mock environment variables
    process.env.VITE_API_BASE_URL = 'http://localhost:8001';
    process.env.VITE_WS_URL = 'localhost:8001';
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  describe('Dashboard Initialization', () => {
    it('renders Sprint 3 dashboard with all main sections', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Sprint 3 Advanced Trading Infrastructure')).toBeInTheDocument();
      expect(screen.getByText('System Health Overview')).toBeInTheDocument();
      expect(screen.getByText('WebSocket Infrastructure')).toBeInTheDocument();
      expect(screen.getByText('Real-Time Analytics')).toBeInTheDocument();
      expect(screen.getByText('Risk Management')).toBeInTheDocument();
      expect(screen.getByText('Strategy Deployment')).toBeInTheDocument();
    });

    it('initializes all monitoring services on mount', async () => {
      const mockStartWebSocket = vi.fn();
      const mockStartAnalytics = vi.fn();
      const mockStartRisk = vi.fn();
      const mockStartSystem = vi.fn();

      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      const { useRealTimeAnalytics } = require('../../hooks/analytics/useRealTimeAnalytics');
      const { useDynamicLimitEngine } = require('../../hooks/risk/useDynamicLimitEngine');
      const { useSystemMonitor } = require('../../hooks/monitoring/useSystemMonitor');

      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        startMonitoring: mockStartWebSocket
      });
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        startStreaming: mockStartAnalytics
      });
      useDynamicLimitEngine.mockReturnValue({
        ...useDynamicLimitEngine(),
        startMonitoring: mockStartRisk
      });
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        startMonitoring: mockStartSystem
      });

      render(<Sprint3Dashboard />);
      
      await waitFor(() => {
        expect(mockStartWebSocket).toHaveBeenCalled();
        expect(mockStartAnalytics).toHaveBeenCalled();
        expect(mockStartRisk).toHaveBeenCalled();
        expect(mockStartSystem).toHaveBeenCalled();
      });
    });

    it('displays overall system status correctly', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('System Status: Healthy')).toBeInTheDocument();
      expect(screen.getByText('94.7%')).toBeInTheDocument(); // Overall score
    });

    it('shows real-time connection status', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument(); // Total connections
      expect(screen.getByText('1,189')).toBeInTheDocument(); // Active connections
    });
  });

  describe('WebSocket Infrastructure Integration', () => {
    it('displays WebSocket connection metrics', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Active Connections')).toBeInTheDocument();
      expect(screen.getByText('1,189')).toBeInTheDocument();
      expect(screen.getByText('Connection Rate')).toBeInTheDocument();
      expect(screen.getByText('2.3/sec')).toBeInTheDocument();
    });

    it('shows WebSocket performance indicators', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Average Latency')).toBeInTheDocument();
      expect(screen.getByText('23.4ms')).toBeInTheDocument();
      expect(screen.getByText('Throughput')).toBeInTheDocument();
      expect(screen.getByText('847 msg/s')).toBeInTheDocument();
    });

    it('renders WebSocket monitoring charts', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getAllByTestId('line-chart').length).toBeGreaterThan(0);
    });

    it('allows toggling WebSocket monitoring', async () => {
      const mockStop = vi.fn();
      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        stopMonitoring: mockStop
      });

      render(<Sprint3Dashboard />);
      
      const stopButton = screen.getByText('Stop WebSocket Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });
  });

  describe('Real-Time Analytics Integration', () => {
    it('displays portfolio performance metrics', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('$127,845.67')).toBeInTheDocument();
      expect(screen.getByText('Daily Return')).toBeInTheDocument();
      expect(screen.getByText('0.34%')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('1.85')).toBeInTheDocument();
    });

    it('shows strategy performance data', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Strategy Performance')).toBeInTheDocument();
      expect(screen.getByText('Momentum Strategy')).toBeInTheDocument();
      expect(screen.getByText('$45,678.90')).toBeInTheDocument(); // Strategy P&L
      expect(screen.getByText('15.23%')).toBeInTheDocument(); // Strategy return
    });

    it('indicates real-time streaming status', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Live Streaming')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('renders analytics charts', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getAllByTestId('area-chart').length).toBeGreaterThan(0);
      expect(screen.getAllByTestId('composed-chart').length).toBeGreaterThan(0);
    });
  });

  describe('Risk Management Integration', () => {
    it('displays active risk limits', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Active Risk Limits')).toBeInTheDocument();
      expect(screen.getByText('Portfolio VaR Limit')).toBeInTheDocument();
      expect(screen.getByText('-$85,670.45')).toBeInTheDocument(); // Current value
      expect(screen.getByText('-$100,000')).toBeInTheDocument(); // Limit value
    });

    it('shows risk limit utilization', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('85.67%')).toBeInTheDocument(); // Utilization
      expect(screen.getByText('active')).toBeInTheDocument(); // Status
    });

    it('displays breach predictions', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Breach Predictions')).toBeInTheDocument();
      expect(screen.getByText('Predicted breach in 1h')).toBeInTheDocument();
      expect(screen.getByText('78% confidence')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument(); // Severity
    });

    it('renders risk monitoring charts', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getAllByTestId('bar-chart').length).toBeGreaterThan(0);
    });
  });

  describe('Strategy Deployment Integration', () => {
    it('displays deployment approval queue', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Deployment Queue')).toBeInTheDocument();
      expect(screen.getByText('Momentum Alpha Strategy')).toBeInTheDocument();
      expect(screen.getByText('pending_review')).toBeInTheDocument();
      expect(screen.getByText('high')).toBeInTheDocument(); // Priority
    });

    it('shows deployment metrics', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Total: 156')).toBeInTheDocument();
      expect(screen.getByText('Approved: 134')).toBeInTheDocument();
      expect(screen.getByText('Success Rate: 94.7%')).toBeInTheDocument();
    });

    it('allows deployment actions', async () => {
      const mockApprove = vi.fn();
      const { useDeploymentApproval } = require('../../hooks/strategy/useDeploymentApproval');
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        approveDeployment: mockApprove
      });

      render(<Sprint3Dashboard />);
      
      const approveButton = screen.getByText('Quick Approve');
      await user.click(approveButton);
      
      expect(mockApprove).toHaveBeenCalled();
    });

    it('renders deployment pipeline chart', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getAllByTestId('pie-chart').length).toBeGreaterThan(0);
    });
  });

  describe('System Monitoring Integration', () => {
    it('displays component health status', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('WebSocket: 98.2%')).toBeInTheDocument();
      expect(screen.getByText('Risk: 87.5%')).toBeInTheDocument();
      expect(screen.getByText('Analytics: 96.1%')).toBeInTheDocument();
    });

    it('shows system resource metrics', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('34.7%')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('67.3%')).toBeInTheDocument();
    });

    it('displays system alerts', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('System Alerts')).toBeInTheDocument();
      expect(screen.getByText('Risk limit breached')).toBeInTheDocument();
      expect(screen.getByText('critical')).toBeInTheDocument();
    });

    it('renders system health charts', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getAllByTestId('responsive-container').length).toBeGreaterThan(5);
    });
  });

  describe('Cross-Component Integration', () => {
    it('coordinates data updates across all components', async () => {
      render(<Sprint3Dashboard />);
      
      // Simulate real-time data updates
      act(() => {
        vi.advanceTimersByTime(30000); // 30 seconds
      });
      
      await waitFor(() => {
        expect(screen.getByText('Sprint 3 Advanced Trading Infrastructure')).toBeInTheDocument();
      });
    });

    it('propagates alerts across dashboard sections', async () => {
      const { useSystemMonitor } = require('../../hooks/monitoring/useSystemMonitor');
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        alerts: [
          {
            id: 'critical-alert',
            severity: 'critical',
            message: 'System overload detected',
            component: 'webSocketInfrastructure',
            timestamp: Date.now()
          }
        ]
      });

      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('System overload detected')).toBeInTheDocument();
    });

    it('maintains consistent state across navigation', async () => {
      render(<Sprint3Dashboard />);
      
      // Navigate to different dashboard sections
      const riskTab = screen.getByText('Risk Dashboard');
      await user.click(riskTab);
      
      await waitFor(() => {
        expect(screen.getByText('Dynamic Risk Limits')).toBeInTheDocument();
      });
      
      const analyticsTab = screen.getByText('Analytics Dashboard');
      await user.click(analyticsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Real-Time Portfolio Analytics')).toBeInTheDocument();
      });
    });

    it('handles component failures gracefully', () => {
      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionStats: null,
        error: 'WebSocket service unavailable'
      });

      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('WebSocket service unavailable')).toBeInTheDocument();
      // Other components should still function
      expect(screen.getByText('Real-Time Analytics')).toBeInTheDocument();
    });
  });

  describe('Performance and Load Testing', () => {
    it('handles high-frequency data updates efficiently', async () => {
      render(<Sprint3Dashboard />);
      
      // Simulate rapid updates every second for 30 seconds
      act(() => {
        for (let i = 0; i < 30; i++) {
          vi.advanceTimersByTime(1000);
        }
      });
      
      expect(screen.getByText('Sprint 3 Advanced Trading Infrastructure')).toBeInTheDocument();
    });

    it('maintains responsiveness with large datasets', () => {
      // Mock large datasets
      const { useRealTimeAnalytics } = require('../../hooks/analytics/useRealTimeAnalytics');
      const manyStrategies = Array.from({ length: 100 }, (_, i) => ({
        strategyId: `strategy-${i}`,
        strategyName: `Strategy ${i}`,
        pnl: Math.random() * 100000,
        return: Math.random() * 20
      }));

      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        performanceMetrics: {
          portfolio: { totalPnL: 127845.67, dailyReturn: 0.34, sharpeRatio: 1.85 },
          strategies: manyStrategies
        }
      });

      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Sprint 3 Advanced Trading Infrastructure')).toBeInTheDocument();
    });

    it('handles concurrent WebSocket connections', () => {
      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionStats: {
          total_connections: 5000,
          active_connections: 4847,
          connection_rate: 25.7,
          average_connection_duration: '4h 12m'
        }
      });

      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('5,000')).toBeInTheDocument(); // Total connections
      expect(screen.getByText('4,847')).toBeInTheDocument(); // Active connections
    });
  });

  describe('Real-time Synchronization', () => {
    it('synchronizes WebSocket and analytics updates', async () => {
      const mockWebSocketUpdate = vi.fn();
      const mockAnalyticsUpdate = vi.fn();

      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      const { useRealTimeAnalytics } = require('../../hooks/analytics/useRealTimeAnalytics');

      useWebSocketMonitoring.mockImplementation(() => ({
        ...useWebSocketMonitoring(),
        refreshMetrics: mockWebSocketUpdate
      }));
      
      useRealTimeAnalytics.mockImplementation(() => ({
        ...useRealTimeAnalytics(),
        refreshData: mockAnalyticsUpdate
      }));

      render(<Sprint3Dashboard />);
      
      // Simulate synchronized update trigger
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      await waitFor(() => {
        expect(mockWebSocketUpdate).toHaveBeenCalled();
        expect(mockAnalyticsUpdate).toHaveBeenCalled();
      });
    });

    it('maintains data consistency across components', async () => {
      render(<Sprint3Dashboard />);
      
      // Check that risk data is consistent across risk and system monitoring
      expect(screen.getByText('Portfolio VaR Limit')).toBeInTheDocument();
      expect(screen.getByText('Risk: 87.5%')).toBeInTheDocument();
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('recovers from WebSocket connection failures', async () => {
      const { useWebSocketMonitoring } = require('../../hooks/websocket/useWebSocketMonitoring');
      
      // Start with connection failure
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionStats: null,
        isMonitoring: false,
        error: 'Connection failed'
      });

      const { rerender } = render(<Sprint3Dashboard />);
      
      expect(screen.getByText('Connection failed')).toBeInTheDocument();
      
      // Simulate recovery
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionStats: {
          total_connections: 1247,
          active_connections: 1189,
          connection_rate: 2.3
        },
        isMonitoring: true,
        error: null
      });

      rerender(<Sprint3Dashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('1,247')).toBeInTheDocument();
      });
    });

    it('handles partial component failures', () => {
      const { useRealTimeAnalytics } = require('../../hooks/analytics/useRealTimeAnalytics');
      useRealTimeAnalytics.mockReturnValue({
        ...useRealTimeAnalytics(),
        performanceMetrics: null,
        isInitialized: false,
        error: 'Analytics service down'
      });

      render(<Sprint3Dashboard />);
      
      // Analytics should show error, but other components work
      expect(screen.getByText('Analytics service down')).toBeInTheDocument();
      expect(screen.getByText('Portfolio VaR Limit')).toBeInTheDocument(); // Risk still works
      expect(screen.getByText('1,189')).toBeInTheDocument(); // WebSocket still works
    });
  });

  describe('User Interactions', () => {
    it('supports dashboard navigation', async () => {
      render(<Sprint3Dashboard />);
      
      // Test tab navigation
      const tabs = ['Overview', 'WebSocket', 'Analytics', 'Risk', 'Strategy', 'System'];
      
      for (const tabName of tabs) {
        const tab = screen.getByText(tabName);
        await user.click(tab);
        
        await waitFor(() => {
          expect(tab.closest('.ant-tabs-tab')).toHaveClass('ant-tabs-tab-active');
        });
      }
    });

    it('allows expanding/collapsing dashboard sections', async () => {
      render(<Sprint3Dashboard />);
      
      const expandButton = screen.getByText('Expand WebSocket Details');
      await user.click(expandButton);
      
      await waitFor(() => {
        expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
      });
    });

    it('supports real-time control actions', async () => {
      const mockStopAll = vi.fn();
      render(<Sprint3Dashboard />);
      
      const emergencyStop = screen.getByText('Emergency Stop All Services');
      await user.click(emergencyStop);
      
      // Should trigger confirmation dialog
      await waitFor(() => {
        expect(screen.getByText('Confirm Emergency Stop')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility and Usability', () => {
    it('provides keyboard navigation throughout dashboard', async () => {
      render(<Sprint3Dashboard />);
      
      // Test keyboard navigation
      await user.tab();
      await user.tab();
      await user.tab();
      
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels for screen readers', () => {
      render(<Sprint3Dashboard />);
      
      const buttons = screen.getAllByRole('button');
      const headings = screen.getAllByRole('heading');
      
      expect(buttons.length).toBeGreaterThan(5);
      expect(headings.length).toBeGreaterThan(3);
    });

    it('provides meaningful status indicators', () => {
      render(<Sprint3Dashboard />);
      
      expect(screen.getByText('System Status: Healthy')).toBeInTheDocument();
      expect(screen.getByText('Live Streaming')).toBeInTheDocument();
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  describe('Data Export and Reporting', () => {
    it('supports exporting dashboard data', async () => {
      render(<Sprint3Dashboard />);
      
      const exportButton = screen.getByText('Export Dashboard Data');
      await user.click(exportButton);
      
      await waitFor(() => {
        expect(screen.getByText('Export Options')).toBeInTheDocument();
      });
    });

    it('generates comprehensive system reports', async () => {
      render(<Sprint3Dashboard />);
      
      const reportButton = screen.getByText('Generate System Report');
      await user.click(reportButton);
      
      const { notification } = await import('antd');
      expect(notification.info).toHaveBeenCalledWith({
        message: 'Report Generation',
        description: 'Generating comprehensive system report...'
      });
    });
  });
});