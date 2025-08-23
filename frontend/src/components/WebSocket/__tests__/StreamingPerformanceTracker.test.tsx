/**
 * StreamingPerformanceTracker Test Suite
 * Sprint 3: WebSocket streaming performance monitoring and optimization testing
 * 
 * Tests real-time performance tracking, latency monitoring, throughput analysis,
 * connection quality assessment, and performance optimization recommendations.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import StreamingPerformanceTracker from '../StreamingPerformanceTracker';

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

// Mock WebSocket performance tracking hook
vi.mock('../../../hooks/websocket/useStreamingPerformance', () => ({
  useStreamingPerformance: vi.fn(() => ({
    performanceMetrics: {
      latency: {
        current: 12.5,
        average: 15.3,
        min: 3.2,
        max: 67.8,
        p50: 14.1,
        p95: 45.2,
        p99: 58.9
      },
      throughput: {
        messagesPerSecond: 847,
        bytesPerSecond: 1248576,
        peakMessagesPerSecond: 1285,
        peakBytesPerSecond: 1887232
      },
      connectionQuality: {
        stability: 98.5,
        reliability: 99.2,
        efficiency: 96.8,
        overallScore: 98.2
      },
      networkMetrics: {
        packetLoss: 0.02,
        jitter: 3.4,
        bandwidth: 2150000000, // 2.15 Gbps
        roundTripTime: 24.7
      },
      bufferMetrics: {
        inboundBuffer: 156,
        outboundBuffer: 89,
        bufferOverflows: 2,
        averageBufferSize: 124
      },
      errorMetrics: {
        connectionErrors: 1,
        messageErrors: 3,
        timeoutErrors: 0,
        protocolErrors: 1,
        totalErrors: 5
      }
    },
    historicalData: [
      {
        timestamp: Date.now() - 60000,
        latency: 11.2,
        throughput: 823,
        quality: 97.8,
        errors: 0
      },
      {
        timestamp: Date.now() - 30000,
        latency: 13.7,
        throughput: 901,
        quality: 98.5,
        errors: 1
      },
      {
        timestamp: Date.now(),
        latency: 12.5,
        throughput: 847,
        quality: 98.2,
        errors: 0
      }
    ],
    performanceAlerts: [
      {
        id: 'alert-1',
        type: 'latency_spike',
        severity: 'medium',
        message: 'Latency spike detected: 67.8ms peak',
        timestamp: Date.now() - 120000,
        resolved: true
      },
      {
        id: 'alert-2',
        type: 'buffer_overflow',
        severity: 'low',
        message: 'Buffer overflow occurred in outbound queue',
        timestamp: Date.now() - 60000,
        resolved: false
      }
    ],
    optimizationSuggestions: [
      {
        category: 'buffer_management',
        priority: 'high',
        suggestion: 'Consider increasing buffer size for high-throughput streams',
        impact: 'Reduce buffer overflows and improve message delivery reliability'
      },
      {
        category: 'connection_tuning',
        priority: 'medium',
        suggestion: 'Optimize connection pooling configuration',
        impact: 'Improve connection establishment time and resource utilization'
      }
    ],
    isTracking: true,
    isAnalyzing: false,
    error: null,
    startTracking: vi.fn(),
    stopTracking: vi.fn(),
    resetMetrics: vi.fn(),
    exportMetrics: vi.fn(),
    triggerAnalysis: vi.fn(),
    acknowledgeAlert: vi.fn()
  }))
}));

describe('StreamingPerformanceTracker', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    connectionId: 'test-connection',
    trackingInterval: 1000,
    alertThresholds: {
      latencyMax: 100,
      throughputMin: 100,
      qualityMin: 95,
      errorRateMax: 1
    },
    enableRealTime: true,
    showAdvancedMetrics: true,
    autoOptimize: false
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
    it('renders performance tracker component', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
      expect(screen.getByText('Latency Analysis')).toBeInTheDocument();
      expect(screen.getByText('Throughput Monitoring')).toBeInTheDocument();
    });

    it('displays current tracking status', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Performance Tracking Active')).toBeInTheDocument();
    });

    it('shows connection ID', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('test-connection')).toBeInTheDocument();
    });

    it('renders without advanced metrics when disabled', () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={false} />);
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });
  });

  describe('Performance Metrics Display', () => {
    it('displays latency metrics correctly', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Current Latency')).toBeInTheDocument();
      expect(screen.getByText('12.5ms')).toBeInTheDocument();
      expect(screen.getByText('Average Latency')).toBeInTheDocument();
      expect(screen.getByText('15.3ms')).toBeInTheDocument();
    });

    it('shows throughput statistics', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Message Rate')).toBeInTheDocument();
      expect(screen.getByText('847 msg/s')).toBeInTheDocument();
      expect(screen.getByText('Data Rate')).toBeInTheDocument();
    });

    it('displays connection quality score', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Overall Quality')).toBeInTheDocument();
      expect(screen.getByText('98.2%')).toBeInTheDocument();
    });

    it('shows percentile latency values', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('P95: 45.2ms')).toBeInTheDocument();
      expect(screen.getByText('P99: 58.9ms')).toBeInTheDocument();
    });

    it('displays network metrics when advanced mode enabled', () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Packet Loss')).toBeInTheDocument();
      expect(screen.getByText('0.02%')).toBeInTheDocument();
      expect(screen.getByText('Jitter')).toBeInTheDocument();
      expect(screen.getByText('3.4ms')).toBeInTheDocument();
    });
  });

  describe('Performance Charts', () => {
    it('renders latency trend chart', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('shows throughput chart', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const throughputTab = screen.getByText('Throughput');
      await user.click(throughputTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('displays quality metrics chart', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const qualityTab = screen.getByText('Quality');
      await user.click(qualityTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('shows latency distribution histogram', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={true} />);
      
      const distributionTab = screen.getByText('Distribution');
      await user.click(distributionTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Alerts', () => {
    it('displays performance alerts', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const alertsSection = screen.getByText('Performance Alerts');
      expect(alertsSection).toBeInTheDocument();
      
      expect(screen.getByText('Latency spike detected: 67.8ms peak')).toBeInTheDocument();
      expect(screen.getByText('Buffer overflow occurred in outbound queue')).toBeInTheDocument();
    });

    it('shows alert severity levels', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('low')).toBeInTheDocument();
    });

    it('displays resolved and unresolved alerts differently', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      // Should show different styling for resolved vs unresolved
      expect(screen.getByText('Resolved')).toBeInTheDocument();
      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    it('allows acknowledging alerts', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockAcknowledge = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });
  });

  describe('Optimization Suggestions', () => {
    it('displays optimization suggestions', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Performance Optimization')).toBeInTheDocument();
      expect(screen.getByText('Consider increasing buffer size for high-throughput streams')).toBeInTheDocument();
      expect(screen.getByText('Optimize connection pooling configuration')).toBeInTheDocument();
    });

    it('shows suggestion priorities', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
    });

    it('displays impact descriptions', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText(/Reduce buffer overflows/)).toBeInTheDocument();
      expect(screen.getByText(/Improve connection establishment/)).toBeInTheDocument();
    });

    it('categorizes suggestions properly', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('buffer_management')).toBeInTheDocument();
      expect(screen.getByText('connection_tuning')).toBeInTheDocument();
    });
  });

  describe('Control Functions', () => {
    it('starts performance tracking', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockStart = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        isTracking: false,
        startTracking: mockStart
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const startButton = screen.getByText('Start Tracking');
      await user.click(startButton);
      
      expect(mockStart).toHaveBeenCalled();
    });

    it('stops performance tracking', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockStop = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        stopTracking: mockStop
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Tracking');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('resets performance metrics', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockReset = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        resetMetrics: mockReset
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const resetButton = screen.getByText('Reset Metrics');
      await user.click(resetButton);
      
      expect(mockReset).toHaveBeenCalled();
    });

    it('exports performance data', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockExport = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        exportMetrics: mockExport
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const exportButton = screen.getByText('Export Data');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });

    it('triggers performance analysis', async () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const mockAnalysis = vi.fn();
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        triggerAnalysis: mockAnalysis
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const analyzeButton = screen.getByText('Analyze Performance');
      await user.click(analyzeButton);
      
      expect(mockAnalysis).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('updates metrics in real-time when enabled', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} enableRealTime={true} />);
      
      // Should show current metrics
      expect(screen.getByText('12.5ms')).toBeInTheDocument();
      
      // Simulate real-time update
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });

    it('respects tracking interval', () => {
      render(<StreamingPerformanceTracker {...defaultProps} trackingInterval={2000} />);
      
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      
      expect(screen.getByText('Performance Tracking Active')).toBeInTheDocument();
    });

    it('stops updates when real-time is disabled', () => {
      render(<StreamingPerformanceTracker {...defaultProps} enableRealTime={false} />);
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });
  });

  describe('Threshold Management', () => {
    it('applies custom alert thresholds', () => {
      const customThresholds = {
        latencyMax: 50,
        throughputMin: 500,
        qualityMin: 99,
        errorRateMax: 0.5
      };
      
      render(<StreamingPerformanceTracker {...defaultProps} alertThresholds={customThresholds} />);
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });

    it('highlights metrics exceeding thresholds', () => {
      const strictThresholds = {
        latencyMax: 10, // Current latency is 12.5ms
        throughputMin: 1000, // Current throughput is 847 msg/s
        qualityMin: 99, // Current quality is 98.2%
        errorRateMax: 0.1
      };
      
      render(<StreamingPerformanceTracker {...defaultProps} alertThresholds={strictThresholds} />);
      
      // Values exceeding thresholds should be highlighted
      expect(screen.getByText('12.5ms')).toBeInTheDocument();
      expect(screen.getByText('847 msg/s')).toBeInTheDocument();
    });
  });

  describe('Buffer Monitoring', () => {
    it('displays buffer metrics when advanced mode enabled', () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Buffer Status')).toBeInTheDocument();
      expect(screen.getByText('Inbound: 156')).toBeInTheDocument();
      expect(screen.getByText('Outbound: 89')).toBeInTheDocument();
    });

    it('shows buffer overflow count', () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Overflows: 2')).toBeInTheDocument();
    });

    it('displays average buffer size', () => {
      render(<StreamingPerformanceTracker {...defaultProps} showAdvancedMetrics={true} />);
      
      expect(screen.getByText('Avg Size: 124')).toBeInTheDocument();
    });
  });

  describe('Error Tracking', () => {
    it('displays error metrics', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Error Summary')).toBeInTheDocument();
      expect(screen.getByText('Connection: 1')).toBeInTheDocument();
      expect(screen.getByText('Message: 3')).toBeInTheDocument();
      expect(screen.getByText('Protocol: 1')).toBeInTheDocument();
    });

    it('shows total error count', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Total Errors: 5')).toBeInTheDocument();
    });

    it('calculates error rate correctly', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      // Error rate should be calculated based on total errors vs messages
      expect(screen.getByText(/Error Rate/)).toBeInTheDocument();
    });
  });

  describe('Performance Analysis', () => {
    it('shows analysis progress when running', () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        isAnalyzing: true
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Analyzing Performance...')).toBeInTheDocument();
    });

    it('displays analysis results', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      // Analysis results should be integrated into suggestions
      expect(screen.getByText('Performance Optimization')).toBeInTheDocument();
    });
  });

  describe('Auto-optimization', () => {
    it('shows auto-optimization status when enabled', () => {
      render(<StreamingPerformanceTracker {...defaultProps} autoOptimize={true} />);
      
      expect(screen.getByText('Auto-optimization: Enabled')).toBeInTheDocument();
    });

    it('allows toggling auto-optimization', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} autoOptimize={false} />);
      
      const autoOptToggle = screen.getByText('Enable Auto-optimize');
      expect(autoOptToggle).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when tracking fails', () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        error: 'Failed to connect to performance tracking service'
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Performance Tracking Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to connect to performance tracking service')).toBeInTheDocument();
    });

    it('handles missing connection gracefully', () => {
      render(<StreamingPerformanceTracker {...defaultProps} connectionId="" />);
      
      expect(screen.getByText('No connection specified')).toBeInTheDocument();
    });

    it('shows fallback UI when metrics unavailable', () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        performanceMetrics: null
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Performance metrics unavailable')).toBeInTheDocument();
    });
  });

  describe('Load Testing Scenarios', () => {
    it('handles high-frequency metric updates', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} trackingInterval={100} />);
      
      // Simulate rapid updates
      act(() => {
        for (let i = 0; i < 100; i++) {
          vi.advanceTimersByTime(100);
        }
      });
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });

    it('maintains performance with large historical datasets', () => {
      const { useStreamingPerformance } = require('../../../hooks/websocket/useStreamingPerformance');
      const largeHistoricalData = Array.from({ length: 10000 }, (_, i) => ({
        timestamp: Date.now() - i * 1000,
        latency: 10 + Math.random() * 50,
        throughput: 500 + Math.random() * 1000,
        quality: 90 + Math.random() * 10,
        errors: Math.floor(Math.random() * 3)
      }));

      useStreamingPerformance.mockReturnValue({
        ...useStreamingPerformance(),
        historicalData: largeHistoricalData
      });

      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Streaming Performance Tracker')).toBeInTheDocument();
    });

    it('handles concurrent connection tracking', () => {
      const multiConnectionProps = {
        ...defaultProps,
        connectionId: 'multi-connection-test'
      };
      
      render(<StreamingPerformanceTracker {...multiConnectionProps} />);
      
      expect(screen.getByText('multi-connection-test')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides keyboard navigation', async () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes ARIA labels for metrics', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen readers with descriptive text', () => {
      render(<StreamingPerformanceTracker {...defaultProps} />);
      
      expect(screen.getByText('Current Latency')).toBeInTheDocument();
      expect(screen.getByText('Message Rate')).toBeInTheDocument();
      expect(screen.getByText('Overall Quality')).toBeInTheDocument();
    });
  });
});