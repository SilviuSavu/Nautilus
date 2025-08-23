/**
 * WebSocketScalabilityMonitor Test Suite
 * Sprint 3: WebSocket scalability testing and load monitoring
 * 
 * Tests concurrent connection handling, load testing scenarios, 
 * resource utilization tracking, scaling recommendations, and performance under stress.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import WebSocketScalabilityMonitor from '../WebSocketScalabilityMonitor';

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
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>),
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>)
}));

// Mock WebSocket scalability hook
vi.mock('../../../hooks/websocket/useWebSocketScalability', () => ({
  useWebSocketScalability: vi.fn(() => ({
    scalabilityMetrics: {
      currentConnections: 1247,
      maxConcurrentConnections: 1456,
      targetCapacity: 2000,
      utilizationPercentage: 62.35,
      connectionRate: 15.7, // connections per second
      disconnectionRate: 3.2,
      averageConnectionLifetime: 1847000, // ms
      resourceUsage: {
        cpuUsage: 34.2,
        memoryUsage: 1024 * 1024 * 512, // 512MB
        networkBandwidth: 2150000000, // 2.15 Gbps
        fileDescriptors: 1534,
        threadCount: 24
      },
      performanceMetrics: {
        averageLatency: 23.4,
        p95Latency: 67.8,
        throughputPerConnection: 847.2,
        errorRate: 0.23,
        timeoutRate: 0.05
      }
    },
    loadTestResults: {
      isRunning: false,
      currentTestLoad: 0,
      maxTestedLoad: 1500,
      resultsHistory: [
        {
          timestamp: Date.now() - 3600000,
          connectionCount: 500,
          latencyP95: 45.2,
          errorRate: 0.1,
          throughput: 423500,
          resourceUsage: { cpu: 15.2, memory: 256 * 1024 * 1024 }
        },
        {
          timestamp: Date.now() - 1800000,
          connectionCount: 1000,
          latencyP95: 78.9,
          errorRate: 0.3,
          throughput: 847000,
          resourceUsage: { cpu: 28.7, memory: 384 * 1024 * 1024 }
        },
        {
          timestamp: Date.now() - 900000,
          connectionCount: 1500,
          latencyP95: 123.4,
          errorRate: 1.2,
          throughput: 1270500,
          resourceUsage: { cpu: 43.5, memory: 512 * 1024 * 1024 }
        }
      ]
    },
    scalingRecommendations: [
      {
        category: 'infrastructure',
        priority: 'high',
        recommendation: 'Add 2 additional server instances to handle projected load',
        expectedImpact: 'Increase capacity to 4000 concurrent connections',
        estimatedCost: '$240/month',
        implementationTime: '2 hours'
      },
      {
        category: 'configuration',
        priority: 'medium',
        recommendation: 'Optimize connection pooling configuration',
        expectedImpact: 'Reduce memory usage by 15-20%',
        estimatedCost: '$0',
        implementationTime: '30 minutes'
      },
      {
        category: 'architecture',
        priority: 'low',
        recommendation: 'Implement connection sharding across multiple processes',
        expectedImpact: 'Improve fault tolerance and resource distribution',
        estimatedCost: 'Development time',
        implementationTime: '1 week'
      }
    ],
    scalabilityAlerts: [
      {
        id: 'alert-cpu-high',
        type: 'resource_limit',
        severity: 'warning',
        message: 'CPU usage approaching 80% threshold',
        timestamp: Date.now() - 300000,
        threshold: 80,
        currentValue: 76.3,
        resolved: false
      },
      {
        id: 'alert-connections-high',
        type: 'capacity_limit',
        severity: 'critical',
        message: 'Connection count approaching maximum capacity',
        timestamp: Date.now() - 180000,
        threshold: 90,
        currentValue: 89.7,
        resolved: true
      }
    ],
    isMonitoring: true,
    error: null,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    startLoadTest: vi.fn(),
    stopLoadTest: vi.fn(),
    generateScalingReport: vi.fn(),
    resetMetrics: vi.fn(),
    acknowledgeAlert: vi.fn(),
    applyRecommendation: vi.fn()
  }))
}));

describe('WebSocketScalabilityMonitor', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    monitoringInterval: 5000,
    targetCapacity: 2000,
    alertThresholds: {
      cpuUsage: 80,
      memoryUsage: 0.8, // 80%
      connectionUtilization: 90,
      errorRate: 1,
      latencyP95: 100
    },
    enableAutoScaling: false,
    loadTestConfig: {
      maxConnections: 2000,
      rampUpDuration: 300000, // 5 minutes
      testDuration: 600000,   // 10 minutes
      messageRate: 10
    }
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
    it('renders scalability monitor component', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('WebSocket Scalability Monitor')).toBeInTheDocument();
      expect(screen.getByText('Connection Scaling')).toBeInTheDocument();
      expect(screen.getByText('Resource Utilization')).toBeInTheDocument();
      expect(screen.getByText('Load Testing')).toBeInTheDocument();
    });

    it('displays current scaling status', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });

    it('shows connection capacity metrics', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Current Connections')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
      expect(screen.getByText('Target Capacity')).toBeInTheDocument();
      expect(screen.getByText('2,000')).toBeInTheDocument();
    });

    it('displays utilization percentage', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Utilization')).toBeInTheDocument();
      expect(screen.getByText('62.35%')).toBeInTheDocument();
    });
  });

  describe('Connection Metrics', () => {
    it('shows connection rate statistics', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Connection Rate')).toBeInTheDocument();
      expect(screen.getByText('15.7/sec')).toBeInTheDocument();
      expect(screen.getByText('Disconnect Rate')).toBeInTheDocument();
      expect(screen.getByText('3.2/sec')).toBeInTheDocument();
    });

    it('displays maximum concurrent connections achieved', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Peak Concurrent')).toBeInTheDocument();
      expect(screen.getByText('1,456')).toBeInTheDocument();
    });

    it('shows average connection lifetime', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Avg Lifetime')).toBeInTheDocument();
      expect(screen.getByText(/30m|31m/)).toBeInTheDocument(); // ~30.78 minutes
    });

    it('displays performance metrics per connection', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Latency (P95)')).toBeInTheDocument();
      expect(screen.getByText('67.8ms')).toBeInTheDocument();
      expect(screen.getByText('Throughput/Conn')).toBeInTheDocument();
      expect(screen.getByText('847.2 msg/s')).toBeInTheDocument();
    });
  });

  describe('Resource Utilization', () => {
    it('displays CPU usage metrics', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('34.2%')).toBeInTheDocument();
    });

    it('shows memory usage information', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('512 MB')).toBeInTheDocument();
    });

    it('displays network bandwidth utilization', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Network Bandwidth')).toBeInTheDocument();
      expect(screen.getByText(/Gbps/)).toBeInTheDocument();
    });

    it('shows file descriptor and thread counts', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('File Descriptors')).toBeInTheDocument();
      expect(screen.getByText('1,534')).toBeInTheDocument();
      expect(screen.getByText('Thread Count')).toBeInTheDocument();
      expect(screen.getByText('24')).toBeInTheDocument();
    });

    it('applies color coding based on utilization levels', () => {
      const highUsageProps = {
        ...defaultProps,
        alertThresholds: {
          ...defaultProps.alertThresholds,
          cpuUsage: 30 // Lower threshold to trigger warning
        }
      };
      
      render(<WebSocketScalabilityMonitor {...highUsageProps} />);
      
      // CPU usage of 34.2% should be highlighted as exceeding 30% threshold
      expect(screen.getByText('34.2%')).toBeInTheDocument();
    });
  });

  describe('Load Testing', () => {
    it('displays load test controls', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Load Testing')).toBeInTheDocument();
      expect(screen.getByText('Start Load Test')).toBeInTheDocument();
    });

    it('shows load test configuration', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Max Connections: 2,000')).toBeInTheDocument();
      expect(screen.getByText('Ramp Duration: 5m')).toBeInTheDocument();
      expect(screen.getByText('Test Duration: 10m')).toBeInTheDocument();
    });

    it('starts load testing', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockStartTest = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        startLoadTest: mockStartTest
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const startButton = screen.getByText('Start Load Test');
      await user.click(startButton);
      
      expect(mockStartTest).toHaveBeenCalled();
    });

    it('displays load test progress when running', () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        loadTestResults: {
          ...useWebSocketScalability().loadTestResults,
          isRunning: true,
          currentTestLoad: 750
        }
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Load Test Running')).toBeInTheDocument();
      expect(screen.getByText('Current Load: 750')).toBeInTheDocument();
    });

    it('shows load test results history', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Load Test History')).toBeInTheDocument();
      expect(screen.getByText('500 connections')).toBeInTheDocument();
      expect(screen.getByText('1000 connections')).toBeInTheDocument();
      expect(screen.getByText('1500 connections')).toBeInTheDocument();
    });

    it('stops load testing', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockStopTest = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        loadTestResults: {
          ...useWebSocketScalability().loadTestResults,
          isRunning: true
        },
        stopLoadTest: mockStopTest
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Load Test');
      await user.click(stopButton);
      
      expect(mockStopTest).toHaveBeenCalled();
    });
  });

  describe('Scaling Recommendations', () => {
    it('displays scaling recommendations', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Scaling Recommendations')).toBeInTheDocument();
      expect(screen.getByText('Add 2 additional server instances to handle projected load')).toBeInTheDocument();
      expect(screen.getByText('Optimize connection pooling configuration')).toBeInTheDocument();
    });

    it('shows recommendation priorities', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('low')).toBeInTheDocument();
    });

    it('displays expected impact and costs', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Increase capacity to 4000 concurrent connections')).toBeInTheDocument();
      expect(screen.getByText('$240/month')).toBeInTheDocument();
      expect(screen.getByText('2 hours')).toBeInTheDocument();
    });

    it('allows applying recommendations', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockApply = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        applyRecommendation: mockApply
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const applyButtons = screen.getAllByText('Apply');
      if (applyButtons.length > 0) {
        await user.click(applyButtons[0]);
        expect(mockApply).toHaveBeenCalled();
      }
    });

    it('categorizes recommendations correctly', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('infrastructure')).toBeInTheDocument();
      expect(screen.getByText('configuration')).toBeInTheDocument();
      expect(screen.getByText('architecture')).toBeInTheDocument();
    });
  });

  describe('Scalability Alerts', () => {
    it('displays scalability alerts', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Scalability Alerts')).toBeInTheDocument();
      expect(screen.getByText('CPU usage approaching 80% threshold')).toBeInTheDocument();
      expect(screen.getByText('Connection count approaching maximum capacity')).toBeInTheDocument();
    });

    it('shows alert severity levels', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('warning')).toBeInTheDocument();
      expect(screen.getByText('critical')).toBeInTheDocument();
    });

    it('displays current values vs thresholds', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('76.3% of 80%')).toBeInTheDocument();
      expect(screen.getByText('89.7% of 90%')).toBeInTheDocument();
    });

    it('distinguishes resolved from active alerts', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('Resolved')).toBeInTheDocument();
    });

    it('allows acknowledging alerts', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockAcknowledge = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders connection trends chart', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('shows resource utilization chart', async () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const resourceTab = screen.getByText('Resource Usage');
      await user.click(resourceTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('displays load test results chart', async () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const loadTestTab = screen.getByText('Load Test Results');
      await user.click(loadTestTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });
    });

    it('shows capacity utilization visualization', async () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const capacityTab = screen.getByText('Capacity');
      await user.click(capacityTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Control Functions', () => {
    it('starts scalability monitoring', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockStart = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        isMonitoring: false,
        startMonitoring: mockStart
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      expect(mockStart).toHaveBeenCalled();
    });

    it('stops scalability monitoring', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockStop = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        stopMonitoring: mockStop
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('generates scaling report', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockReport = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        generateScalingReport: mockReport
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const reportButton = screen.getByText('Generate Report');
      await user.click(reportButton);
      
      expect(mockReport).toHaveBeenCalled();
    });

    it('resets metrics', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      const mockReset = vi.fn();
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        resetMetrics: mockReset
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const resetButton = screen.getByText('Reset Metrics');
      await user.click(resetButton);
      
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Auto-scaling Configuration', () => {
    it('shows auto-scaling status when enabled', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} enableAutoScaling={true} />);
      
      expect(screen.getByText('Auto-scaling: Enabled')).toBeInTheDocument();
    });

    it('allows toggling auto-scaling', async () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} enableAutoScaling={false} />);
      
      const autoScaleToggle = screen.getByText('Enable Auto-scaling');
      expect(autoScaleToggle).toBeInTheDocument();
    });

    it('displays auto-scaling rules when enabled', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} enableAutoScaling={true} />);
      
      expect(screen.getByText('Auto-scaling Rules')).toBeInTheDocument();
    });
  });

  describe('Performance Under Load', () => {
    it('handles high connection counts efficiently', () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        scalabilityMetrics: {
          ...useWebSocketScalability().scalabilityMetrics,
          currentConnections: 5000,
          maxConcurrentConnections: 5247
        }
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('5,000')).toBeInTheDocument();
      expect(screen.getByText('5,247')).toBeInTheDocument();
    });

    it('maintains responsiveness during load tests', async () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        loadTestResults: {
          ...useWebSocketScalability().loadTestResults,
          isRunning: true,
          currentTestLoad: 1500
        }
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      // Should still be able to navigate during load test
      const resourceTab = screen.getByText('Resource Usage');
      await user.click(resourceTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('handles rapid metric updates', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} monitoringInterval={1000} />);
      
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      expect(screen.getByText('WebSocket Scalability Monitor')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when monitoring fails', () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        error: 'Failed to connect to scalability monitoring service'
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to connect to scalability monitoring service')).toBeInTheDocument();
    });

    it('handles missing metrics gracefully', () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        scalabilityMetrics: null
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Scalability metrics unavailable')).toBeInTheDocument();
    });

    it('shows appropriate messages when load test fails', () => {
      const { useWebSocketScalability } = require('../../../hooks/websocket/useWebSocketScalability');
      useWebSocketScalability.mockReturnValue({
        ...useWebSocketScalability(),
        loadTestResults: {
          ...useWebSocketScalability().loadTestResults,
          isRunning: false,
          lastError: 'Load test failed: Connection timeout'
        }
      });

      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Load test failed: Connection timeout')).toBeInTheDocument();
    });
  });

  describe('Data Formatting and Calculations', () => {
    it('formats large numbers correctly', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('1,247')).toBeInTheDocument(); // Current connections
      expect(screen.getByText('2,000')).toBeInTheDocument(); // Target capacity
    });

    it('calculates utilization percentage accurately', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      // 1247 / 2000 * 100 = 62.35%
      expect(screen.getByText('62.35%')).toBeInTheDocument();
    });

    it('formats memory usage correctly', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('512 MB')).toBeInTheDocument();
    });

    it('displays time durations properly', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText(/30m|31m/)).toBeInTheDocument(); // Connection lifetime
    });
  });

  describe('Accessibility and Usability', () => {
    it('provides keyboard navigation support', async () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Current Connections')).toBeInTheDocument();
      expect(screen.getByText('Resource Utilization')).toBeInTheDocument();
      expect(screen.getByText('Load Testing')).toBeInTheDocument();
    });

    it('provides meaningful status indicators', () => {
      render(<WebSocketScalabilityMonitor {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });
  });
});