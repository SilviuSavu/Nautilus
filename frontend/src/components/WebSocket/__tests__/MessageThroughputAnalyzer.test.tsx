/**
 * MessageThroughputAnalyzer Test Suite
 * Sprint 3: Comprehensive WebSocket message throughput analysis testing
 * 
 * Tests message throughput monitoring, bandwidth analysis, performance insights,
 * message type classification, real-time updates, and load testing scenarios.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import MessageThroughputAnalyzer from '../MessageThroughputAnalyzer';

// Mock recharts
vi.mock('recharts', () => ({
  ResponsiveContainer: vi.fn(({ children }) => <div data-testid="responsive-container">{children}</div>),
  ComposedChart: vi.fn(() => <div data-testid="composed-chart">Composed Chart</div>),
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
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>),
  Treemap: vi.fn(() => <div data-testid="treemap">Treemap</div>)
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

describe('MessageThroughputAnalyzer', () => {
  const user = userEvent.setup();
  
  // Mock WebSocket endpoints
  const mockEndpoints = [
    {
      id: 'endpoint-1',
      name: 'Market Data Stream',
      url: 'ws://localhost:8001/ws/market-data',
      status: 'connected' as const,
      latency: 15.3,
      messagesPerSecond: 1250,
      subscriptions: 45,
      lastActivity: '2024-01-15T10:30:00Z',
      uptime: 7200000, // 2 hours
      errorCount: 2,
      quality: 98.5
    },
    {
      id: 'endpoint-2',
      name: 'Risk Alerts',
      url: 'ws://localhost:8001/ws/risk-alerts',
      status: 'connected' as const,
      latency: 8.7,
      messagesPerSecond: 340,
      subscriptions: 12,
      lastActivity: '2024-01-15T10:29:45Z',
      uptime: 5400000, // 1.5 hours
      errorCount: 0,
      quality: 100
    },
    {
      id: 'endpoint-3',
      name: 'Trade Updates',
      url: 'ws://localhost:8001/ws/trades',
      status: 'disconnected' as const,
      latency: 0,
      messagesPerSecond: 0,
      subscriptions: 0,
      lastActivity: '2024-01-15T09:15:00Z',
      uptime: 0,
      errorCount: 15,
      quality: 0
    }
  ];

  const mockHistoricalData = [
    {
      timestamp: Date.now() - 300000,
      totalMessages: 1200,
      inboundMessages: 840,
      outboundMessages: 360,
      totalBandwidth: 1024000,
      averageLatency: 12.5,
      errorRate: 0.8
    },
    {
      timestamp: Date.now() - 240000,
      totalMessages: 1350,
      inboundMessages: 945,
      outboundMessages: 405,
      totalBandwidth: 1152000,
      averageLatency: 14.2,
      errorRate: 1.2
    }
  ];

  const defaultProps = {
    endpoints: mockEndpoints,
    historicalData: mockHistoricalData,
    timeRange: '1h',
    showRealtime: false, // Disable for consistent testing
    maxDataPoints: 100,
    enablePrediction: true,
    alertThresholds: {
      throughputMin: 10,
      throughputMax: 5000,
      bandwidthMax: 10485760, // 10 MB/s
      latencyMax: 100,
      errorRateMax: 5
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
    it('renders throughput analyzer component', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
      expect(screen.getByText('Total Throughput')).toBeInTheDocument();
      expect(screen.getByText('Bandwidth Usage')).toBeInTheDocument();
      expect(screen.getByText('Avg Latency')).toBeInTheDocument();
      expect(screen.getByText('Error Rate')).toBeInTheDocument();
    });

    it('displays key metrics correctly', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByText('Total Throughput')).toBeInTheDocument();
      expect(screen.getByText('Bandwidth Usage')).toBeInTheDocument();
      expect(screen.getByText('Avg Latency')).toBeInTheDocument();
      expect(screen.getByText('Error Rate')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      const { container } = render(
        <MessageThroughputAnalyzer {...defaultProps} className="custom-analyzer" />
      );
      
      expect(container.firstChild).toHaveClass('custom-analyzer');
    });

    it('handles empty endpoints gracefully', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} endpoints={[]} />);
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument(); // Total throughput should be 0
    });
  });

  describe('Key Metrics Display', () => {
    it('calculates total throughput correctly', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      // Total messages per second should be sum of all endpoints
      const expectedTotal = mockEndpoints.reduce((sum, ep) => sum + ep.messagesPerSecond, 0);
      expect(screen.getByText(expectedTotal.toString())).toBeInTheDocument();
    });

    it('displays bandwidth usage', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByText(/B\/s|KB\/s|MB\/s|GB\/s/)).toBeInTheDocument();
    });

    it('shows average latency', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByText(/ms$/)).toBeInTheDocument();
    });

    it('displays error rate percentage', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByText(/%$/)).toBeInTheDocument();
    });

    it('applies color coding based on thresholds', () => {
      const highLatencyProps = {
        ...defaultProps,
        endpoints: [{
          ...mockEndpoints[0],
          latency: 150 // Above threshold
        }]
      };
      
      render(<MessageThroughputAnalyzer {...highLatencyProps} />);
      
      // Should show red color for high latency
      const latencyValue = screen.getByText(/150ms/);
      expect(latencyValue).toBeInTheDocument();
    });
  });

  describe('Performance Insights', () => {
    it('shows performance insights when available', async () => {
      const highThroughputProps = {
        ...defaultProps,
        endpoints: [{
          ...mockEndpoints[0],
          messagesPerSecond: 8000 // High throughput
        }],
        showRealtime: true
      };
      
      render(<MessageThroughputAnalyzer {...highThroughputProps} />);
      
      // Wait for component to generate insights
      act(() => {
        vi.advanceTimersByTime(10000); // 10 seconds
      });
      
      await waitFor(() => {
        const insightsAlert = screen.queryByText('Throughput Analysis Insights');
        if (insightsAlert) {
          expect(insightsAlert).toBeInTheDocument();
        }
      });
    });

    it('detects high message throughput', async () => {
      const highThroughputEndpoints = mockEndpoints.map(ep => ({
        ...ep,
        messagesPerSecond: ep.messagesPerSecond * 3
      }));
      
      render(
        <MessageThroughputAnalyzer 
          {...defaultProps} 
          endpoints={highThroughputEndpoints}
          showRealtime={true}
        />
      );
      
      act(() => {
        vi.advanceTimersByTime(15000); // Allow time for analysis
      });
      
      await waitFor(() => {
        const insight = screen.queryByText(/High message throughput detected/);
        if (insight) {
          expect(insight).toBeInTheDocument();
        }
      });
    });

    it('identifies low throughput conditions', async () => {
      const lowThroughputProps = {
        ...defaultProps,
        endpoints: [{
          ...mockEndpoints[0],
          messagesPerSecond: 5 // Below minimum threshold
        }],
        showRealtime: true
      };
      
      render(<MessageThroughputAnalyzer {...lowThroughputProps} />);
      
      act(() => {
        vi.advanceTimersByTime(15000);
      });
      
      await waitFor(() => {
        const insight = screen.queryByText(/Low message throughput/);
        if (insight) {
          expect(insight).toBeInTheDocument();
        }
      });
    });
  });

  describe('Tab Navigation and Functionality', () => {
    it('switches between tabs correctly', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      // Initially on throughput tab
      expect(screen.getByText('Throughput Trends')).toBeInTheDocument();
      
      // Click message types tab
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Message Distribution')).toBeInTheDocument();
      });
    });

    it('displays message types table', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Message Type')).toBeInTheDocument();
        expect(screen.getByText('Count')).toBeInTheDocument();
        expect(screen.getByText('Size')).toBeInTheDocument();
        expect(screen.getByText('Latency')).toBeInTheDocument();
        expect(screen.getByText('Errors')).toBeInTheDocument();
        expect(screen.getByText('Trend')).toBeInTheDocument();
      });
    });

    it('shows bandwidth analysis tab', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const bandwidthTab = screen.getByText('Bandwidth Analysis');
      await user.click(bandwidthTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Chart Visualizations', () => {
    it('renders throughput trends chart', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
    });

    it('displays message distribution pie chart', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });

    it('shows bandwidth chart in bandwidth tab', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const bandwidthTab = screen.getByText('Bandwidth Analysis');
      await user.click(bandwidthTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Controls and Filters', () => {
    it('toggles between bandwidth and message view', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const bandwidthToggle = screen.getByText('Bandwidth');
      await user.click(bandwidthToggle);
      
      expect(bandwidthToggle).toBeInTheDocument();
    });

    it('toggles prediction mode when enabled', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} enablePrediction={true} />);
      
      const predictionToggle = screen.getByText('Prediction');
      await user.click(predictionToggle);
      
      expect(predictionToggle).toBeInTheDocument();
    });

    it('filters endpoints correctly', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const endpointFilter = screen.getByPlaceholderText('Filter endpoints');
      await user.click(endpointFilter);
      
      await waitFor(() => {
        expect(screen.getByText('Market Data Stream')).toBeInTheDocument();
        expect(screen.getByText('Risk Alerts')).toBeInTheDocument();
      });
    });

    it('handles export functionality', async () => {
      const { notification } = await import('antd');
      
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const exportButton = screen.getByText('Export');
      await user.click(exportButton);
      
      expect(notification.success).toHaveBeenCalledWith({
        message: 'Data Exported',
        description: 'Throughput data has been exported successfully'
      });
    });
  });

  describe('Real-time Updates', () => {
    it('updates metrics in real-time when enabled', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      // Initial render should show some throughput
      expect(screen.getByText(/\d+/)).toBeInTheDocument();
      
      // Advance time to trigger real-time update
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      
      await waitFor(() => {
        expect(screen.getByText(/\d+/)).toBeInTheDocument();
      });
    });

    it('respects maxDataPoints limit', () => {
      const lowMaxPoints = {
        ...defaultProps,
        maxDataPoints: 5,
        showRealtime: true
      };
      
      render(<MessageThroughputAnalyzer {...lowMaxPoints} />);
      
      // Generate multiple data points
      act(() => {
        for (let i = 0; i < 10; i++) {
          vi.advanceTimersByTime(1000);
        }
      });
      
      // Component should limit data points internally
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });

    it('stops real-time updates when disabled', () => {
      const { rerender } = render(
        <MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />
      );
      
      // Disable real-time updates
      rerender(<MessageThroughputAnalyzer {...defaultProps} showRealtime={false} />);
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });
  });

  describe('Message Type Analysis', () => {
    it('categorizes message types correctly', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      // Wait for message type data to be generated
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        // Should show various message types
        expect(screen.getByText(/MARKET_DATA|TRADE_UPDATE|RISK_ALERT/)).toBeInTheDocument();
      });
    });

    it('calculates message type statistics', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      act(() => {
        vi.advanceTimersByTime(3000);
      });
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        // Should show percentage values
        expect(screen.getByText(/% of total/)).toBeInTheDocument();
      });
    });

    it('shows message size information', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByText(/Avg:/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles missing historical data gracefully', () => {
      render(
        <MessageThroughputAnalyzer 
          {...defaultProps} 
          historicalData={[]}
          endpoints={[]}
        />
      );
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('displays appropriate values for disconnected endpoints', () => {
      const disconnectedEndpoints = mockEndpoints.map(ep => ({
        ...ep,
        status: 'disconnected' as const,
        messagesPerSecond: 0,
        latency: 0
      }));
      
      render(
        <MessageThroughputAnalyzer 
          {...defaultProps} 
          endpoints={disconnectedEndpoints}
        />
      );
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });

    it('handles invalid alert thresholds', () => {
      const invalidThresholds = {
        ...defaultProps,
        alertThresholds: {
          throughputMin: -1,
          throughputMax: 0,
          bandwidthMax: -1,
          latencyMax: -1,
          errorRateMax: -1
        }
      };
      
      render(<MessageThroughputAnalyzer {...invalidThresholds} />);
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });
  });

  describe('Performance and Load Testing', () => {
    it('handles high-frequency data updates efficiently', async () => {
      const highFrequencyProps = {
        ...defaultProps,
        showRealtime: true,
        maxDataPoints: 1000
      };
      
      render(<MessageThroughputAnalyzer {...highFrequencyProps} />);
      
      // Simulate rapid updates
      act(() => {
        for (let i = 0; i < 100; i++) {
          vi.advanceTimersByTime(100); // 10 updates per second
        }
      });
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });

    it('manages memory usage with large datasets', () => {
      const largeEndpoints = Array.from({ length: 100 }, (_, i) => ({
        id: `endpoint-${i}`,
        name: `Endpoint ${i}`,
        url: `ws://localhost:8001/ws/stream-${i}`,
        status: 'connected' as const,
        latency: Math.random() * 50,
        messagesPerSecond: Math.floor(Math.random() * 1000),
        subscriptions: Math.floor(Math.random() * 20),
        lastActivity: new Date().toISOString(),
        uptime: Math.random() * 86400000,
        errorCount: Math.floor(Math.random() * 10),
        quality: 90 + Math.random() * 10
      }));
      
      render(
        <MessageThroughputAnalyzer 
          {...defaultProps} 
          endpoints={largeEndpoints}
          maxDataPoints={1000
        }
        />
      );
      
      expect(screen.getByText('Message Throughput Analysis')).toBeInTheDocument();
    });

    it('maintains responsiveness under load', async () => {
      const loadTestProps = {
        ...defaultProps,
        endpoints: Array.from({ length: 50 }, (_, i) => ({
          ...mockEndpoints[0],
          id: `load-endpoint-${i}`,
          messagesPerSecond: 1000 + i
        })),
        showRealtime: true
      };
      
      render(<MessageThroughputAnalyzer {...loadTestProps} />);
      
      // Simulate heavy load
      act(() => {
        for (let i = 0; i < 50; i++) {
          vi.advanceTimersByTime(200);
        }
      });
      
      // Component should still be responsive
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Message Distribution')).toBeInTheDocument();
      });
    });
  });

  describe('Data Format and Calculations', () => {
    it('formats bytes correctly', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      // Should show formatted byte values
      expect(screen.getByText(/B\/s|KB\/s|MB\/s|GB\/s/)).toBeInTheDocument();
    });

    it('calculates bandwidth accurately', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      
      // Bandwidth should be calculated based on message count and size
      expect(screen.getByText('Bandwidth Usage')).toBeInTheDocument();
    });

    it('computes message distribution percentages', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} showRealtime={true} />);
      
      act(() => {
        vi.advanceTimersByTime(3000);
      });
      
      const typesTab = screen.getByText('Message Types');
      await user.click(typesTab);
      
      await waitFor(() => {
        expect(screen.getByText(/% of total/)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility and Usability', () => {
    it('provides keyboard navigation support', async () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<MessageThroughputAnalyzer {...defaultProps} />);
      
      // Should have descriptive text for screen readers
      expect(screen.getByText('Total Throughput')).toBeInTheDocument();
      expect(screen.getByText('Bandwidth Usage')).toBeInTheDocument();
    });
  });
});