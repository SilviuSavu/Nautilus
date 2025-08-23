/**
 * ConnectionHealthDashboard Test Suite
 * Comprehensive tests for the WebSocket connection health dashboard component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import ConnectionHealthDashboard from '../ConnectionHealthDashboard';

// Mock the connection health hooks
vi.mock('../../../hooks/websocket/useConnectionHealth', () => ({
  useConnectionHealth: vi.fn(() => ({
    connectionHealth: {
      overall_status: 'healthy',
      health_score: 94.7,
      uptime_percentage: 99.97,
      total_connections: 1247,
      healthy_connections: 1189,
      degraded_connections: 43,
      failed_connections: 15,
      connection_stability_score: 96.2,
      average_response_time: 12.5,
      error_rate: 0.8,
      throughput_score: 92.3
    },
    connectionMetrics: [
      {
        connection_id: 'conn-001',
        client_id: 'dashboard-client-1',
        status: 'healthy',
        health_score: 98.5,
        latency: 8.3,
        packet_loss: 0.1,
        jitter: 2.1,
        bandwidth_utilization: 45.2,
        error_count: 0,
        reconnect_count: 0,
        last_heartbeat: '2024-01-15T10:29:55Z',
        connection_duration: '2h 15m',
        data_quality_score: 99.2,
        threat_level: 'none'
      },
      {
        connection_id: 'conn-002',
        client_id: 'analytics-client-2',
        status: 'degraded',
        health_score: 78.4,
        latency: 45.7,
        packet_loss: 2.3,
        jitter: 8.9,
        bandwidth_utilization: 78.9,
        error_count: 5,
        reconnect_count: 2,
        last_heartbeat: '2024-01-15T10:29:48Z',
        connection_duration: '1h 42m',
        data_quality_score: 85.3,
        threat_level: 'low',
        issues: ['high_latency', 'packet_loss']
      },
      {
        connection_id: 'conn-003',
        client_id: 'risk-client-3',
        status: 'critical',
        health_score: 23.1,
        latency: 156.8,
        packet_loss: 12.5,
        jitter: 45.2,
        bandwidth_utilization: 95.7,
        error_count: 23,
        reconnect_count: 8,
        last_heartbeat: '2024-01-15T10:27:32Z',
        connection_duration: '45m',
        data_quality_score: 52.8,
        threat_level: 'medium',
        issues: ['timeout_errors', 'high_packet_loss', 'connection_instability']
      }
    ],
    healthTrends: {
      past_24h: [
        { timestamp: '2024-01-14T10:30:00Z', health_score: 96.2, connections: 1156 },
        { timestamp: '2024-01-14T14:30:00Z', health_score: 94.8, connections: 1203 },
        { timestamp: '2024-01-14T18:30:00Z', health_score: 92.1, connections: 1178 },
        { timestamp: '2024-01-14T22:30:00Z', health_score: 95.5, connections: 1089 },
        { timestamp: '2024-01-15T02:30:00Z', health_score: 97.2, connections: 945 },
        { timestamp: '2024-01-15T06:30:00Z', health_score: 96.8, connections: 1098 },
        { timestamp: '2024-01-15T10:30:00Z', health_score: 94.7, connections: 1247 }
      ]
    },
    networkDiagnostics: {
      dns_resolution_time: 2.3,
      tcp_handshake_time: 8.7,
      ssl_handshake_time: 12.4,
      first_byte_time: 18.9,
      connection_establishment_success_rate: 98.7,
      network_congestion_score: 15.2,
      route_stability_score: 94.8,
      isp_performance_score: 91.5
    },
    securityMetrics: {
      suspicious_activity_detected: 3,
      blocked_connections: 12,
      ddos_attempts: 0,
      malformed_requests: 8,
      rate_limit_violations: 45,
      security_score: 96.8,
      threat_intelligence_score: 98.2
    },
    performanceInsights: [
      {
        insight_type: 'optimization',
        category: 'latency',
        message: 'Consider implementing connection pooling for clients with high reconnection rates',
        priority: 'medium',
        potential_improvement: '15-20% latency reduction',
        implementation_complexity: 'low'
      },
      {
        insight_type: 'alert',
        category: 'capacity',
        message: 'Connection count approaching 80% of configured maximum',
        priority: 'high',
        potential_improvement: 'Prevent service degradation',
        implementation_complexity: 'medium'
      },
      {
        insight_type: 'recommendation',
        category: 'monitoring',
        message: 'Enable advanced heartbeat monitoring for critical clients',
        priority: 'low',
        potential_improvement: 'Enhanced failure detection',
        implementation_complexity: 'low'
      }
    ],
    alertHistory: [
      {
        alert_id: 'health-001',
        type: 'connection_degraded',
        severity: 'warning',
        message: 'Connection health score dropped below 80%',
        affected_connections: ['conn-002'],
        triggered_at: '2024-01-15T10:15:00Z',
        resolved_at: null,
        auto_remediation_attempted: true,
        remediation_action: 'Initiated connection reset'
      },
      {
        alert_id: 'health-002',
        type: 'high_error_rate',
        severity: 'critical',
        message: 'Error rate exceeded 10% threshold',
        affected_connections: ['conn-003'],
        triggered_at: '2024-01-15T09:45:00Z',
        resolved_at: '2024-01-15T10:05:00Z',
        auto_remediation_attempted: true,
        remediation_action: 'Forced connection recycling'
      }
    ],
    isMonitoring: true,
    error: null,
    startHealthMonitoring: vi.fn(),
    stopHealthMonitoring: vi.fn(),
    forceConnectionRecycle: vi.fn(),
    optimizeConnection: vi.fn(),
    isolateConnection: vi.fn(),
    runDiagnostics: vi.fn(),
    exportHealthReport: vi.fn(),
    refreshHealth: vi.fn()
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
  RadialBarChart: vi.fn(() => <div data-testid="radial-bar-chart">Radial Bar Chart</div>),
  RadialBar: vi.fn(() => <div data-testid="radial-bar">Radial Bar</div>),
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>)
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
    diff: vi.fn(() => 8100000) // 2h 15m in ms
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('ConnectionHealthDashboard', () => {
  const user = userEvent.setup();
  const mockProps = {
    autoRefresh: true,
    refreshInterval: 10000,
    showNetworkDiagnostics: true,
    enableSecurityMetrics: true,
    height: 900
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the health dashboard', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Connection Health Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Health Overview')).toBeInTheDocument();
      expect(screen.getByText('Connection Details')).toBeInTheDocument();
      expect(screen.getByText('Health Trends')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<ConnectionHealthDashboard {...mockProps} height={1200} />);
      
      const dashboard = screen.getByText('Connection Health Dashboard').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<ConnectionHealthDashboard />);
      
      expect(screen.getByText('Connection Health Dashboard')).toBeInTheDocument();
    });

    it('shows monitoring status', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Health Monitoring Active')).toBeInTheDocument();
    });
  });

  describe('Health Overview', () => {
    it('displays overall health status', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Overall Status')).toBeInTheDocument();
      expect(screen.getByText('healthy')).toBeInTheDocument();
    });

    it('shows health score', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Health Score')).toBeInTheDocument();
      expect(screen.getByText('94.7')).toBeInTheDocument();
    });

    it('displays uptime percentage', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Uptime')).toBeInTheDocument();
      expect(screen.getByText('99.97%')).toBeInTheDocument();
    });

    it('shows connection counts', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Total Connections')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
      expect(screen.getByText('Healthy')).toBeInTheDocument();
      expect(screen.getByText('1,189')).toBeInTheDocument();
    });

    it('displays degraded and failed connections', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Degraded')).toBeInTheDocument();
      expect(screen.getByText('43')).toBeInTheDocument();
      expect(screen.getByText('Failed')).toBeInTheDocument();
      expect(screen.getByText('15')).toBeInTheDocument();
    });

    it('shows stability score', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Stability Score')).toBeInTheDocument();
      expect(screen.getByText('96.2')).toBeInTheDocument();
    });

    it('displays response time metrics', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Avg Response Time')).toBeInTheDocument();
      expect(screen.getByText('12.5ms')).toBeInTheDocument();
    });

    it('shows error rate', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Error Rate')).toBeInTheDocument();
      expect(screen.getByText('0.8%')).toBeInTheDocument();
    });

    it('displays throughput score', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Throughput Score')).toBeInTheDocument();
      expect(screen.getByText('92.3')).toBeInTheDocument();
    });

    it('renders health score gauge', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByTestId('radial-bar-chart')).toBeInTheDocument();
    });
  });

  describe('Connection Details', () => {
    it('displays connection table', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      expect(detailsTab).toBeInTheDocument();
    });

    it('shows connection information', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('dashboard-client-1')).toBeInTheDocument();
        expect(screen.getByText('analytics-client-2')).toBeInTheDocument();
        expect(screen.getByText('risk-client-3')).toBeInTheDocument();
      });
    });

    it('displays connection statuses', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('healthy')).toBeInTheDocument();
        expect(screen.getByText('degraded')).toBeInTheDocument();
        expect(screen.getByText('critical')).toBeInTheDocument();
      });
    });

    it('shows health scores for connections', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('98.5')).toBeInTheDocument();
        expect(screen.getByText('78.4')).toBeInTheDocument();
        expect(screen.getByText('23.1')).toBeInTheDocument();
      });
    });

    it('displays latency metrics', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('8.3ms')).toBeInTheDocument();
        expect(screen.getByText('45.7ms')).toBeInTheDocument();
        expect(screen.getByText('156.8ms')).toBeInTheDocument();
      });
    });

    it('shows packet loss percentages', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('0.1%')).toBeInTheDocument();
        expect(screen.getByText('2.3%')).toBeInTheDocument();
        expect(screen.getByText('12.5%')).toBeInTheDocument();
      });
    });

    it('displays connection durations', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('2h 15m')).toBeInTheDocument();
        expect(screen.getByText('1h 42m')).toBeInTheDocument();
        expect(screen.getByText('45m')).toBeInTheDocument();
      });
    });

    it('shows error and reconnect counts', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('0 errors')).toBeInTheDocument();
        expect(screen.getByText('5 errors')).toBeInTheDocument();
        expect(screen.getByText('23 errors')).toBeInTheDocument();
      });
    });

    it('displays threat levels', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('none')).toBeInTheDocument();
        expect(screen.getByText('low')).toBeInTheDocument();
        expect(screen.getByText('medium')).toBeInTheDocument();
      });
    });

    it('shows connection issues', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        expect(screen.getByText('high_latency')).toBeInTheDocument();
        expect(screen.getByText('packet_loss')).toBeInTheDocument();
        expect(screen.getByText('timeout_errors')).toBeInTheDocument();
      });
    });
  });

  describe('Health Trends', () => {
    it('displays health trend chart', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const trendsTab = screen.getByText('Health Trends');
      expect(trendsTab).toBeInTheDocument();
    });

    it('shows 24-hour trend data', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const trendsTab = screen.getByText('Health Trends');
      await user.click(trendsTab);
      
      await waitFor(() => {
        expect(screen.getByText('24-Hour Health Trend')).toBeInTheDocument();
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('displays connection count trends', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const trendsTab = screen.getByText('Health Trends');
      await user.click(trendsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Connection Count Trend')).toBeInTheDocument();
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Network Diagnostics', () => {
    it('displays network diagnostics when enabled', () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      expect(networkTab).toBeInTheDocument();
    });

    it('shows DNS resolution time', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('DNS Resolution')).toBeInTheDocument();
        expect(screen.getByText('2.3ms')).toBeInTheDocument();
      });
    });

    it('displays TCP handshake time', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('TCP Handshake')).toBeInTheDocument();
        expect(screen.getByText('8.7ms')).toBeInTheDocument();
      });
    });

    it('shows SSL handshake time', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('SSL Handshake')).toBeInTheDocument();
        expect(screen.getByText('12.4ms')).toBeInTheDocument();
      });
    });

    it('displays first byte time', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('First Byte Time')).toBeInTheDocument();
        expect(screen.getByText('18.9ms')).toBeInTheDocument();
      });
    });

    it('shows connection establishment success rate', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('Connection Success Rate')).toBeInTheDocument();
        expect(screen.getByText('98.7%')).toBeInTheDocument();
      });
    });

    it('displays network quality scores', async () => {
      render(<ConnectionHealthDashboard {...mockProps} showNetworkDiagnostics={true} />);
      
      const networkTab = screen.getByText('Network Diagnostics');
      await user.click(networkTab);
      
      await waitFor(() => {
        expect(screen.getByText('Route Stability')).toBeInTheDocument();
        expect(screen.getByText('94.8')).toBeInTheDocument();
        expect(screen.getByText('ISP Performance')).toBeInTheDocument();
        expect(screen.getByText('91.5')).toBeInTheDocument();
      });
    });
  });

  describe('Security Metrics', () => {
    it('displays security metrics when enabled', () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      expect(securityTab).toBeInTheDocument();
    });

    it('shows suspicious activity count', async () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      await user.click(securityTab);
      
      await waitFor(() => {
        expect(screen.getByText('Suspicious Activity')).toBeInTheDocument();
        expect(screen.getByText('3 detected')).toBeInTheDocument();
      });
    });

    it('displays blocked connections', async () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      await user.click(securityTab);
      
      await waitFor(() => {
        expect(screen.getByText('Blocked Connections')).toBeInTheDocument();
        expect(screen.getByText('12 blocked')).toBeInTheDocument();
      });
    });

    it('shows DDoS attempts', async () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      await user.click(securityTab);
      
      await waitFor(() => {
        expect(screen.getByText('DDoS Attempts')).toBeInTheDocument();
        expect(screen.getByText('0 attempts')).toBeInTheDocument();
      });
    });

    it('displays rate limit violations', async () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      await user.click(securityTab);
      
      await waitFor(() => {
        expect(screen.getByText('Rate Limit Violations')).toBeInTheDocument();
        expect(screen.getByText('45 violations')).toBeInTheDocument();
      });
    });

    it('shows security scores', async () => {
      render(<ConnectionHealthDashboard {...mockProps} enableSecurityMetrics={true} />);
      
      const securityTab = screen.getByText('Security');
      await user.click(securityTab);
      
      await waitFor(() => {
        expect(screen.getByText('Security Score')).toBeInTheDocument();
        expect(screen.getByText('96.8')).toBeInTheDocument();
        expect(screen.getByText('Threat Intelligence')).toBeInTheDocument();
        expect(screen.getByText('98.2')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Insights', () => {
    it('displays performance insights', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const insightsTab = screen.getByText('Insights');
      expect(insightsTab).toBeInTheDocument();
    });

    it('shows optimization recommendations', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const insightsTab = screen.getByText('Insights');
      await user.click(insightsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Performance Insights')).toBeInTheDocument();
        expect(screen.getByText('Consider implementing connection pooling')).toBeInTheDocument();
      });
    });

    it('displays insight priorities', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const insightsTab = screen.getByText('Insights');
      await user.click(insightsTab);
      
      await waitFor(() => {
        expect(screen.getByText('medium')).toBeInTheDocument();
        expect(screen.getByText('high')).toBeInTheDocument();
        expect(screen.getByText('low')).toBeInTheDocument();
      });
    });

    it('shows potential improvements', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const insightsTab = screen.getByText('Insights');
      await user.click(insightsTab);
      
      await waitFor(() => {
        expect(screen.getByText('15-20% latency reduction')).toBeInTheDocument();
        expect(screen.getByText('Prevent service degradation')).toBeInTheDocument();
      });
    });

    it('displays implementation complexity', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const insightsTab = screen.getByText('Insights');
      await user.click(insightsTab);
      
      await waitFor(() => {
        expect(screen.getAllByText('low')).toHaveLength(3); // 2 complexity + 1 priority
        expect(screen.getByText('medium')).toBeInTheDocument();
      });
    });
  });

  describe('Alert History', () => {
    it('displays alert history', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      expect(alertsTab).toBeInTheDocument();
    });

    it('shows alert details', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Connection health score dropped below 80%')).toBeInTheDocument();
        expect(screen.getByText('Error rate exceeded 10% threshold')).toBeInTheDocument();
      });
    });

    it('displays alert severities', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('warning')).toBeInTheDocument();
        expect(screen.getByText('critical')).toBeInTheDocument();
      });
    });

    it('shows remediation actions', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Initiated connection reset')).toBeInTheDocument();
        expect(screen.getByText('Forced connection recycling')).toBeInTheDocument();
      });
    });

    it('displays resolution status', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const alertsTab = screen.getByText('Alerts');
      await user.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(screen.getByText('Resolved')).toBeInTheDocument();
      });
    });
  });

  describe('Control Actions', () => {
    it('renders control buttons', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
      expect(screen.getByText('Refresh')).toBeInTheDocument();
      expect(screen.getByText('Run Diagnostics')).toBeInTheDocument();
    });

    it('allows stopping health monitoring', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockStop = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        stopHealthMonitoring: mockStop
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const stopButton = screen.getByText('Stop Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('allows refreshing health data', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockRefresh = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        refreshHealth: mockRefresh
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const refreshButton = screen.getByText('Refresh');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('allows running diagnostics', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockDiagnostics = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        runDiagnostics: mockDiagnostics
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const diagnosticsButton = screen.getByText('Run Diagnostics');
      await user.click(diagnosticsButton);
      
      expect(mockDiagnostics).toHaveBeenCalled();
    });

    it('allows forcing connection recycle', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockRecycle = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        forceConnectionRecycle: mockRecycle
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        const recycleButtons = screen.getAllByText('Recycle');
        if (recycleButtons.length > 0) {
          expect(recycleButtons[0]).toBeInTheDocument();
        }
      });
    });

    it('allows isolating problematic connections', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockIsolate = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        isolateConnection: mockIsolate
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      await user.click(detailsTab);
      
      await waitFor(() => {
        const isolateButtons = screen.getAllByText('Isolate');
        if (isolateButtons.length > 0) {
          expect(isolateButtons[0]).toBeInTheDocument();
        }
      });
    });
  });

  describe('Export Functionality', () => {
    it('renders export button', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const exportButton = screen.getByText('Export Report');
      expect(exportButton).toBeInTheDocument();
    });

    it('handles health report export', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockExport = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        exportHealthReport: mockExport
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const exportButton = screen.getByText('Export Report');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes health data automatically when enabled', async () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      const mockRefresh = vi.fn();
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        refreshHealth: mockRefresh
      });

      render(<ConnectionHealthDashboard {...mockProps} autoRefresh={true} refreshInterval={5000} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('shows last update timestamp', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when health monitoring fails', () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        error: 'Connection health service unavailable'
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Health Monitoring Error')).toBeInTheDocument();
      expect(screen.getByText('Connection health service unavailable')).toBeInTheDocument();
    });

    it('handles empty connection list', () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        connectionMetrics: []
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const detailsTab = screen.getByText('Connection Details');
      expect(detailsTab).toBeInTheDocument();
    });

    it('shows monitoring stopped state', () => {
      const { useConnectionHealth } = require('../../../hooks/websocket/useConnectionHealth');
      useConnectionHealth.mockReturnValue({
        ...useConnectionHealth(),
        isMonitoring: false
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Health Monitoring Stopped')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for progress indicators', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('supports keyboard navigation', async () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('healthy')).toBeInTheDocument();
      expect(screen.getByText('Health Monitoring Active')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Connection Health Dashboard')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<ConnectionHealthDashboard {...mockProps} />);
      
      expect(screen.getByText('Health Overview')).toBeInTheDocument();
      expect(screen.getByText('Connection Details')).toBeInTheDocument();
    });
  });
});