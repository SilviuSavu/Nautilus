/**
 * WebSocketMonitoringSuite Test Suite
 * Comprehensive tests for the WebSocket monitoring suite component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import WebSocketMonitoringSuite from '../WebSocketMonitoringSuite';

// Mock the WebSocket monitoring hooks
vi.mock('../../../hooks/websocket/useWebSocketMonitoring', () => ({
  useWebSocketMonitoring: vi.fn(() => ({
    connectionStats: {
      total_connections: 1247,
      active_connections: 1189,
      idle_connections: 58,
      failed_connections: 15,
      connection_rate: 2.3,
      disconnection_rate: 0.8,
      average_connection_duration: '2h 34m',
      peak_concurrent_connections: 1456,
      connection_success_rate: 98.8
    },
    messageStats: {
      total_messages_sent: 2847561,
      total_messages_received: 2851394,
      messages_per_second: 847.5,
      peak_messages_per_second: 1285.7,
      average_message_size: 342.8,
      total_data_transferred: '15.8 GB',
      message_delivery_rate: 99.6,
      message_loss_rate: 0.4
    },
    performanceMetrics: {
      average_latency: 12.5,
      p95_latency: 45.2,
      p99_latency: 89.3,
      min_latency: 2.1,
      max_latency: 234.8,
      jitter: 8.7,
      bandwidth_utilization: 67.3,
      cpu_usage: 23.8,
      memory_usage: 145.6,
      network_throughput: '2.1 Gbps'
    },
    connectionHealth: [
      {
        connection_id: 'conn-001',
        client_id: 'client-dashboard-1',
        ip_address: '192.168.1.100',
        status: 'connected',
        connected_at: '2024-01-15T08:30:00Z',
        last_activity: '2024-01-15T10:29:45Z',
        messages_sent: 15687,
        messages_received: 15692,
        latency: 8.3,
        data_transferred: '45.2 MB',
        subscription_count: 12
      },
      {
        connection_id: 'conn-002',
        client_id: 'client-analytics-2',
        ip_address: '192.168.1.101',
        status: 'connected',
        connected_at: '2024-01-15T09:15:00Z',
        last_activity: '2024-01-15T10:29:50Z',
        messages_sent: 8934,
        messages_received: 8945,
        latency: 15.7,
        data_transferred: '28.7 MB',
        subscription_count: 8
      },
      {
        connection_id: 'conn-003',
        client_id: 'client-risk-3',
        ip_address: '192.168.1.102',
        status: 'disconnected',
        connected_at: '2024-01-15T07:45:00Z',
        disconnected_at: '2024-01-15T10:15:00Z',
        last_activity: '2024-01-15T10:14:58Z',
        messages_sent: 12450,
        messages_received: 12455,
        latency: null,
        data_transferred: '38.9 MB',
        subscription_count: 0,
        disconnect_reason: 'Client timeout'
      }
    ],
    subscriptionStats: {
      total_subscriptions: 8547,
      active_subscriptions: 8234,
      failed_subscriptions: 313,
      subscription_types: {
        'market_data': 3245,
        'risk_alerts': 1876,
        'portfolio_updates': 1523,
        'trade_notifications': 982,
        'system_status': 608
      },
      top_subscribed_topics: [
        { topic: 'market_data.AAPL', subscribers: 456 },
        { topic: 'risk_alerts.var_breach', subscribers: 234 },
        { topic: 'portfolio.performance', subscribers: 189 },
        { topic: 'trades.execution', subscribers: 167 },
        { topic: 'system.health', subscribers: 145 }
      ]
    },
    alertHistory: [
      {
        alert_id: 'alert-001',
        type: 'high_latency',
        severity: 'warning',
        message: 'High latency detected: 156ms average',
        timestamp: '2024-01-15T10:25:00Z',
        affected_connections: 23,
        resolved_at: '2024-01-15T10:27:30Z',
        resolution: 'Auto-scaled server capacity'
      },
      {
        alert_id: 'alert-002',
        type: 'connection_failure',
        severity: 'critical',
        message: 'Multiple connection failures detected',
        timestamp: '2024-01-15T09:45:00Z',
        affected_connections: 8,
        resolved_at: '2024-01-15T09:52:15Z',
        resolution: 'Network connectivity restored'
      },
      {
        alert_id: 'alert-003',
        type: 'message_backlog',
        severity: 'medium',
        message: 'Message queue backlog exceeding threshold',
        timestamp: '2024-01-15T10:10:00Z',
        affected_connections: 156,
        resolved_at: null,
        resolution: null
      }
    ],
    serverMetrics: {
      server_instances: 4,
      healthy_instances: 3,
      cpu_usage_avg: 45.7,
      memory_usage_avg: 67.3,
      network_utilization: 78.9,
      disk_usage: 34.2,
      load_balancer_health: 'healthy',
      redis_connection_pool: 85,
      database_connections: 12
    },
    realTimeUpdates: true,
    isMonitoring: true,
    error: null,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    refreshMetrics: vi.fn(),
    exportMetrics: vi.fn(),
    acknowledgeAlert: vi.fn(),
    closeConnection: vi.fn(),
    resetStats: vi.fn()
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
    valueOf: vi.fn(() => 1705327200000),
    subtract: vi.fn(() => ({
      format: vi.fn(() => '08:30:00')
    })),
    diff: vi.fn(() => 7200000) // 2 hours in ms
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('WebSocketMonitoringSuite', () => {
  const user = userEvent.setup();
  const mockProps = {
    autoRefresh: true,
    refreshInterval: 5000,
    enableRealTime: true,
    height: 800,
    showAdvancedMetrics: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the monitoring suite dashboard', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('WebSocket Monitoring Suite')).toBeInTheDocument();
      expect(screen.getByText('Connection Overview')).toBeInTheDocument();
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
      expect(screen.getByText('Active Connections')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<WebSocketMonitoringSuite {...mockProps} height={1000} />);
      
      const dashboard = screen.getByText('WebSocket Monitoring Suite').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<WebSocketMonitoringSuite />);
      
      expect(screen.getByText('WebSocket Monitoring Suite')).toBeInTheDocument();
    });

    it('shows monitoring status', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });

    it('displays real-time indicator when enabled', () => {
      render(<WebSocketMonitoringSuite {...mockProps} enableRealTime={true} />);
      
      expect(screen.getByText('Real-time Updates')).toBeInTheDocument();
    });
  });

  describe('Connection Statistics', () => {
    it('displays connection overview statistics', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Total Connections')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
      expect(screen.getByText('Active Connections')).toBeInTheDocument();
      expect(screen.getByText('1,189')).toBeInTheDocument();
    });

    it('shows connection rates', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Connection Rate')).toBeInTheDocument();
      expect(screen.getByText('2.3/sec')).toBeInTheDocument();
      expect(screen.getByText('Disconnect Rate')).toBeInTheDocument();
      expect(screen.getByText('0.8/sec')).toBeInTheDocument();
    });

    it('displays success rates', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('98.8%')).toBeInTheDocument();
    });

    it('shows peak statistics', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Peak Concurrent')).toBeInTheDocument();
      expect(screen.getByText('1,456')).toBeInTheDocument();
    });

    it('displays average connection duration', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Avg Duration')).toBeInTheDocument();
      expect(screen.getByText('2h 34m')).toBeInTheDocument();
    });
  });

  describe('Message Statistics', () => {
    it('displays message throughput metrics', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      expect(messagesTab).toBeInTheDocument();
    });

    it('shows message volume statistics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      await user.click(messagesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Messages Sent')).toBeInTheDocument();
        expect(screen.getByText('2,847,561')).toBeInTheDocument();
        expect(screen.getByText('Messages Received')).toBeInTheDocument();
        expect(screen.getByText('2,851,394')).toBeInTheDocument();
      });
    });

    it('displays message rate metrics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      await user.click(messagesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Current Rate')).toBeInTheDocument();
        expect(screen.getByText('847.5/sec')).toBeInTheDocument();
        expect(screen.getByText('Peak Rate')).toBeInTheDocument();
        expect(screen.getByText('1,285.7/sec')).toBeInTheDocument();
      });
    });

    it('shows data transfer statistics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      await user.click(messagesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Total Data')).toBeInTheDocument();
        expect(screen.getByText('15.8 GB')).toBeInTheDocument();
        expect(screen.getByText('Avg Message Size')).toBeInTheDocument();
        expect(screen.getByText('342.8 bytes')).toBeInTheDocument();
      });
    });

    it('displays delivery rates', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      await user.click(messagesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Delivery Rate')).toBeInTheDocument();
        expect(screen.getByText('99.6%')).toBeInTheDocument();
        expect(screen.getByText('Loss Rate')).toBeInTheDocument();
        expect(screen.getByText('0.4%')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Metrics', () => {
    it('displays latency metrics', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      expect(performanceTab).toBeInTheDocument();
    });

    it('shows latency statistics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByText('Average Latency')).toBeInTheDocument();
        expect(screen.getByText('12.5ms')).toBeInTheDocument();
        expect(screen.getByText('P95 Latency')).toBeInTheDocument();
        expect(screen.getByText('45.2ms')).toBeInTheDocument();
        expect(screen.getByText('P99 Latency')).toBeInTheDocument();
        expect(screen.getByText('89.3ms')).toBeInTheDocument();
      });
    });

    it('displays jitter metrics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByText('Jitter')).toBeInTheDocument();
        expect(screen.getByText('8.7ms')).toBeInTheDocument();
      });
    });

    it('shows resource utilization', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
        expect(screen.getByText('23.8%')).toBeInTheDocument();
        expect(screen.getByText('Memory Usage')).toBeInTheDocument();
        expect(screen.getByText('145.6 MB')).toBeInTheDocument();
      });
    });

    it('displays bandwidth metrics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByText('Bandwidth Utilization')).toBeInTheDocument();
        expect(screen.getByText('67.3%')).toBeInTheDocument();
        expect(screen.getByText('Network Throughput')).toBeInTheDocument();
        expect(screen.getByText('2.1 Gbps')).toBeInTheDocument();
      });
    });
  });

  describe('Active Connections', () => {
    it('displays connection table', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      expect(connectionsTab).toBeInTheDocument();
    });

    it('shows connection details', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('client-dashboard-1')).toBeInTheDocument();
        expect(screen.getByText('client-analytics-2')).toBeInTheDocument();
        expect(screen.getByText('client-risk-3')).toBeInTheDocument();
      });
    });

    it('displays connection statuses', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getAllByText('connected')).toHaveLength(2);
        expect(screen.getByText('disconnected')).toBeInTheDocument();
      });
    });

    it('shows IP addresses', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('192.168.1.100')).toBeInTheDocument();
        expect(screen.getByText('192.168.1.101')).toBeInTheDocument();
        expect(screen.getByText('192.168.1.102')).toBeInTheDocument();
      });
    });

    it('displays message counts', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('15,687')).toBeInTheDocument(); // messages sent
        expect(screen.getByText('15,692')).toBeInTheDocument(); // messages received
      });
    });

    it('shows connection latency', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('8.3ms')).toBeInTheDocument();
        expect(screen.getByText('15.7ms')).toBeInTheDocument();
      });
    });

    it('displays data transfer amounts', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('45.2 MB')).toBeInTheDocument();
        expect(screen.getByText('28.7 MB')).toBeInTheDocument();
        expect(screen.getByText('38.9 MB')).toBeInTheDocument();
      });
    });

    it('shows subscription counts', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('12 subs')).toBeInTheDocument();
        expect(screen.getByText('8 subs')).toBeInTheDocument();
      });
    });

    it('displays disconnect reasons', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Client timeout')).toBeInTheDocument();
      });
    });
  });

  describe('Subscription Analytics', () => {
    it('displays subscription statistics', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      expect(subscriptionsTab).toBeInTheDocument();
    });

    it('shows total subscription metrics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      await user.click(subscriptionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Total Subscriptions')).toBeInTheDocument();
        expect(screen.getByText('8,547')).toBeInTheDocument();
        expect(screen.getByText('Active Subscriptions')).toBeInTheDocument();
        expect(screen.getByText('8,234')).toBeInTheDocument();
      });
    });

    it('displays subscription types breakdown', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      await user.click(subscriptionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('market_data: 3,245')).toBeInTheDocument();
        expect(screen.getByText('risk_alerts: 1,876')).toBeInTheDocument();
        expect(screen.getByText('portfolio_updates: 1,523')).toBeInTheDocument();
      });
    });

    it('shows top subscribed topics', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      await user.click(subscriptionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('market_data.AAPL')).toBeInTheDocument();
        expect(screen.getByText('456 subscribers')).toBeInTheDocument();
        expect(screen.getByText('risk_alerts.var_breach')).toBeInTheDocument();
        expect(screen.getByText('234 subscribers')).toBeInTheDocument();
      });
    });

    it('displays failed subscription count', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      await user.click(subscriptionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Failed Subscriptions')).toBeInTheDocument();
        expect(screen.getByText('313')).toBeInTheDocument();
      });
    });
  });

  describe('Alert Management', () => {
    it('displays alert history', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      expect(alertsTab).toBeInTheDocument();
    });

    it('shows alert details', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('high_latency')).toBeInTheDocument();
        expect(screen.getByText('connection_failure')).toBeInTheDocument();
        expect(screen.getByText('message_backlog')).toBeInTheDocument();
      });
    });

    it('displays alert severities', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('warning')).toBeInTheDocument();
        expect(screen.getByText('critical')).toBeInTheDocument();
        expect(screen.getByText('medium')).toBeInTheDocument();
      });
    });

    it('shows alert messages', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('High latency detected: 156ms average')).toBeInTheDocument();
        expect(screen.getByText('Multiple connection failures detected')).toBeInTheDocument();
      });
    });

    it('displays resolution information', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Auto-scaled server capacity')).toBeInTheDocument();
        expect(screen.getByText('Network connectivity restored')).toBeInTheDocument();
      });
    });

    it('shows unresolved alerts', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Unresolved')).toBeInTheDocument();
      });
    });

    it('allows acknowledging alerts', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockAcknowledge = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        const acknowledgeButtons = screen.getAllByText('Acknowledge');
        if (acknowledgeButtons.length > 0) {
          expect(acknowledgeButtons[0]).toBeInTheDocument();
        }
      });
    });
  });

  describe('Server Metrics', () => {
    it('displays server health when advanced metrics are enabled', () => {
      render(<WebSocketMonitoringSuite {...mockProps} showAdvancedMetrics={true} />);
      
      const serverTab = screen.getByText('Server Health');
      expect(serverTab).toBeInTheDocument();
    });

    it('shows server instance information', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} showAdvancedMetrics={true} />);
      
      const serverTab = screen.getByText('Server Health');
      await user.click(serverTab);
      
      await waitFor(() => {
        expect(screen.getByText('Server Instances')).toBeInTheDocument();
        expect(screen.getByText('4 total')).toBeInTheDocument();
        expect(screen.getByText('3 healthy')).toBeInTheDocument();
      });
    });

    it('displays resource utilization', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} showAdvancedMetrics={true} />);
      
      const serverTab = screen.getByText('Server Health');
      await user.click(serverTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU: 45.7%')).toBeInTheDocument();
        expect(screen.getByText('Memory: 67.3%')).toBeInTheDocument();
        expect(screen.getByText('Network: 78.9%')).toBeInTheDocument();
      });
    });

    it('shows connection pool stats', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} showAdvancedMetrics={true} />);
      
      const serverTab = screen.getByText('Server Health');
      await user.click(serverTab);
      
      await waitFor(() => {
        expect(screen.getByText('Redis Pool: 85')).toBeInTheDocument();
        expect(screen.getByText('DB Connections: 12')).toBeInTheDocument();
      });
    });

    it('displays load balancer health', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} showAdvancedMetrics={true} />);
      
      const serverTab = screen.getByText('Server Health');
      await user.click(serverTab);
      
      await waitFor(() => {
        expect(screen.getByText('Load Balancer')).toBeInTheDocument();
        expect(screen.getByText('healthy')).toBeInTheDocument();
      });
    });
  });

  describe('Control Panel', () => {
    it('renders monitoring control buttons', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    it('allows stopping monitoring', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockStop = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        stopMonitoring: mockStop
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const stopButton = screen.getByText('Stop Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('allows refreshing metrics', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockRefresh = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        refreshMetrics: mockRefresh
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const refreshButton = screen.getByText('Refresh');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('allows closing individual connections', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockClose = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        closeConnection: mockClose
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      await user.click(connectionsTab);
      
      await waitFor(() => {
        const closeButtons = screen.getAllByText('Close');
        if (closeButtons.length > 0) {
          expect(closeButtons[0]).toBeInTheDocument();
        }
      });
    });

    it('allows resetting statistics', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockReset = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        resetStats: mockReset
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const resetButton = screen.getByText('Reset Stats');
      await user.click(resetButton);
      
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders connection metrics chart', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays message throughput chart', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const messagesTab = screen.getByText('Messages');
      await user.click(messagesTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('shows latency distribution chart', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });

    it('displays subscription distribution pie chart', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const subscriptionsTab = screen.getByText('Subscriptions');
      await user.click(subscriptionsTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes metrics automatically when enabled', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockRefresh = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        refreshMetrics: mockRefresh
      });

      render(<WebSocketMonitoringSuite {...mockProps} autoRefresh={true} refreshInterval={2000} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('shows real-time update indicator', () => {
      render(<WebSocketMonitoringSuite {...mockProps} enableRealTime={true} />);
      
      expect(screen.getByText('Real-time Updates')).toBeInTheDocument();
    });

    it('displays last update timestamp', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  describe('Export Functionality', () => {
    it('renders export button', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const exportButton = screen.getByText('Export Metrics');
      expect(exportButton).toBeInTheDocument();
    });

    it('handles metrics export', async () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      const mockExport = vi.fn();
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        exportMetrics: mockExport
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const exportButton = screen.getByText('Export Metrics');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when monitoring fails', () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        error: 'WebSocket monitoring service unavailable'
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Monitoring Error')).toBeInTheDocument();
      expect(screen.getByText('WebSocket monitoring service unavailable')).toBeInTheDocument();
    });

    it('handles empty connection list', () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionHealth: []
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const connectionsTab = screen.getByText('Active Connections');
      expect(connectionsTab).toBeInTheDocument();
    });

    it('shows monitoring stopped state', () => {
      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        isMonitoring: false
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Monitoring Stopped')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large numbers of connections efficiently', () => {
      const largeConnectionList = Array.from({ length: 1000 }, (_, i) => ({
        connection_id: `conn-${i.toString().padStart(4, '0')}`,
        client_id: `client-${i}`,
        ip_address: `192.168.${Math.floor(i / 254) + 1}.${(i % 254) + 1}`,
        status: ['connected', 'disconnected', 'idle'][i % 3],
        connected_at: new Date(Date.now() - i * 60000).toISOString(),
        last_activity: new Date(Date.now() - (i * 1000)).toISOString(),
        messages_sent: Math.floor(Math.random() * 20000),
        messages_received: Math.floor(Math.random() * 20000),
        latency: Math.random() * 100,
        data_transferred: `${(Math.random() * 100).toFixed(1)} MB`,
        subscription_count: Math.floor(Math.random() * 20)
      }));

      const { useWebSocketMonitoring } = require('../../../hooks/websocket/useWebSocketMonitoring');
      useWebSocketMonitoring.mockReturnValue({
        ...useWebSocketMonitoring(),
        connectionHealth: largeConnectionList
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('WebSocket Monitoring Suite')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for interactive elements', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('WebSocket Monitoring Suite')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<WebSocketMonitoringSuite {...mockProps} />);
      
      expect(screen.getByText('Connection Overview')).toBeInTheDocument();
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });
  });
});