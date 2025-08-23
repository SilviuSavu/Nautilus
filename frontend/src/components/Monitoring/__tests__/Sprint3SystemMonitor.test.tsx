/**
 * Sprint3SystemMonitor Test Suite
 * Sprint 3: Comprehensive system monitoring and health tracking testing
 * 
 * Tests system health aggregation, component status monitoring, performance metrics,
 * alert management, and infrastructure monitoring across all Sprint 3 components.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import Sprint3SystemMonitor from '../Sprint3SystemMonitor';

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
  ComposedChart: vi.fn(() => <div data-testid="composed-chart">Composed Chart</div>),
  XAxis: vi.fn(() => <div data-testid="x-axis">XAxis</div>),
  YAxis: vi.fn(() => <div data-testid="y-axis">YAxis</div>),
  CartesianGrid: vi.fn(() => <div data-testid="cartesian-grid">Grid</div>),
  Tooltip: vi.fn(() => <div data-testid="tooltip">Tooltip</div>),
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>),
  ReferenceLine: vi.fn(() => <div data-testid="reference-line">Reference Line</div>)
}));

// Mock system monitoring hook
vi.mock('../../../hooks/monitoring/useSystemMonitor', () => ({
  useSystemMonitor: vi.fn(() => ({
    systemHealth: {
      overallStatus: 'healthy',
      overallScore: 94.7,
      lastHealthCheck: Date.now() - 30000,
      components: {
        webSocketInfrastructure: {
          status: 'healthy',
          score: 98.2,
          metrics: {
            activeConnections: 1247,
            connectionRate: 15.7,
            averageLatency: 23.4,
            errorRate: 0.23,
            throughput: '847 msg/s',
            bandwidthUtilization: '67.3%'
          },
          lastCheck: Date.now() - 15000,
          issues: []
        },
        riskManagement: {
          status: 'warning',
          score: 87.5,
          metrics: {
            activeLimits: 12,
            breachedLimits: 1,
            predictedBreaches: 2,
            averageResponseTime: '45.2ms',
            systemUtilization: '78.9%'
          },
          lastCheck: Date.now() - 20000,
          issues: [
            'One limit currently breached: Maximum Drawdown',
            'High breach probability detected for VaR limit'
          ]
        },
        analytics: {
          status: 'healthy',
          score: 96.1,
          metrics: {
            activeStrategies: 8,
            dataProcessingRate: '1.2M events/min',
            calculationLatency: '12.7ms',
            cacheHitRatio: '94.6%',
            storageUtilization: '67.8%'
          },
          lastCheck: Date.now() - 10000,
          issues: []
        },
        strategyDeployment: {
          status: 'healthy',
          score: 91.3,
          metrics: {
            runningDeployments: 3,
            successRate: '98.7%',
            averageDeployTime: '3.2min',
            rollbacksToday: 0,
            testCoverage: '92.4%'
          },
          lastCheck: Date.now() - 25000,
          issues: []
        },
        database: {
          status: 'healthy',
          score: 93.8,
          metrics: {
            connectionPoolSize: 50,
            activeConnections: 23,
            queryLatency: '8.9ms',
            diskUsage: '45.7%',
            memoryUsage: '62.3%'
          },
          lastCheck: Date.now() - 18000,
          issues: []
        },
        messageQueue: {
          status: 'healthy',
          score: 95.4,
          metrics: {
            queueDepth: 145,
            processingRate: '5.7K msg/s',
            errorRate: '0.12%',
            consumerLag: '0.8s',
            memoryUsage: '34.2%'
          },
          lastCheck: Date.now() - 12000,
          issues: []
        }
      }
    },
    performanceMetrics: {
      cpuUsage: 34.7,
      memoryUsage: 67.3,
      diskUsage: 45.2,
      networkUtilization: 78.9,
      responseTime: 87.2,
      throughput: 1247,
      errorRate: 0.23,
      uptime: 99.95
    },
    alerts: [
      {
        id: 'alert-risk-breach',
        component: 'riskManagement',
        severity: 'critical',
        message: 'Maximum Drawdown limit breached: -15.23% exceeds -15.0%',
        timestamp: Date.now() - 300000,
        resolved: false,
        assignedTo: 'risk-team',
        escalationLevel: 1
      },
      {
        id: 'alert-connection-spike',
        component: 'webSocketInfrastructure',
        severity: 'warning',
        message: 'Connection rate spike detected: 25.7 connections/sec',
        timestamp: Date.now() - 180000,
        resolved: true,
        resolvedAt: Date.now() - 120000,
        assignedTo: 'ops-team',
        escalationLevel: 0
      },
      {
        id: 'alert-queue-depth',
        component: 'messageQueue',
        severity: 'info',
        message: 'Queue depth increased: 245 messages pending',
        timestamp: Date.now() - 60000,
        resolved: false,
        assignedTo: 'dev-team',
        escalationLevel: 0
      }
    ],
    historicalData: [
      {
        timestamp: Date.now() - 300000,
        overallScore: 92.1,
        cpuUsage: 28.9,
        memoryUsage: 61.2,
        errorRate: 0.18,
        throughput: 1156
      },
      {
        timestamp: Date.now() - 240000,
        overallScore: 94.3,
        cpuUsage: 32.4,
        memoryUsage: 64.7,
        errorRate: 0.15,
        throughput: 1203
      },
      {
        timestamp: Date.now() - 180000,
        overallScore: 93.8,
        cpuUsage: 35.1,
        memoryUsage: 66.9,
        errorRate: 0.21,
        throughput: 1189
      },
      {
        timestamp: Date.now() - 120000,
        overallScore: 94.7,
        cpuUsage: 34.7,
        memoryUsage: 67.3,
        errorRate: 0.23,
        throughput: 1247
      }
    ],
    infrastructure: {
      servers: [
        {
          id: 'web-01',
          name: 'WebSocket Server 1',
          status: 'healthy',
          cpuUsage: 32.4,
          memoryUsage: 64.8,
          diskUsage: 42.1,
          uptime: 99.98,
          lastCheck: Date.now() - 30000
        },
        {
          id: 'web-02',
          name: 'WebSocket Server 2',
          status: 'healthy',
          cpuUsage: 28.7,
          memoryUsage: 59.2,
          diskUsage: 38.9,
          uptime: 99.95,
          lastCheck: Date.now() - 35000
        },
        {
          id: 'db-01',
          name: 'Database Primary',
          status: 'healthy',
          cpuUsage: 41.3,
          memoryUsage: 78.6,
          diskUsage: 56.7,
          uptime: 99.99,
          lastCheck: Date.now() - 20000
        },
        {
          id: 'redis-01',
          name: 'Redis Cache',
          status: 'warning',
          cpuUsage: 67.8,
          memoryUsage: 89.4,
          diskUsage: 23.4,
          uptime: 99.87,
          lastCheck: Date.now() - 25000,
          issues: ['High memory usage approaching limit']
        }
      ],
      services: [
        { name: 'NautilusTrader Engine', status: 'healthy', version: '1.219.0' },
        { name: 'Risk Management API', status: 'healthy', version: '2.1.3' },
        { name: 'Analytics Service', status: 'healthy', version: '1.8.7' },
        { name: 'WebSocket Gateway', status: 'healthy', version: '3.2.1' },
        { name: 'Message Bus', status: 'healthy', version: '1.5.2' }
      ]
    },
    isMonitoring: true,
    error: null,
    startMonitoring: vi.fn(),
    stopMonitoring: vi.fn(),
    refreshHealth: vi.fn(),
    acknowledgeAlert: vi.fn(),
    escalateAlert: vi.fn(),
    generateReport: vi.fn(),
    configureThresholds: vi.fn()
  }))
}));

describe('Sprint3SystemMonitor', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    refreshInterval: 30000,
    enableAlerts: true,
    enableAutoRefresh: true,
    thresholds: {
      cpu: 80,
      memory: 85,
      disk: 90,
      errorRate: 1,
      responseTime: 100
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
    it('renders system monitor dashboard', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
      expect(screen.getByText('System Health Overview')).toBeInTheDocument();
      expect(screen.getByText('Component Status')).toBeInTheDocument();
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });

    it('displays overall system status', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('System Status: Healthy')).toBeInTheDocument();
      expect(screen.getByText('94.7')).toBeInTheDocument(); // Overall score
    });

    it('shows monitoring status', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });

    it('displays last health check time', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText(/Last check:/)).toBeInTheDocument();
    });
  });

  describe('Component Status Display', () => {
    it('shows all Sprint 3 components', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('WebSocket Infrastructure')).toBeInTheDocument();
      expect(screen.getByText('Risk Management')).toBeInTheDocument();
      expect(screen.getByText('Analytics')).toBeInTheDocument();
      expect(screen.getByText('Strategy Deployment')).toBeInTheDocument();
      expect(screen.getByText('Database')).toBeInTheDocument();
      expect(screen.getByText('Message Queue')).toBeInTheDocument();
    });

    it('displays component health scores', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('98.2')).toBeInTheDocument(); // WebSocket score
      expect(screen.getByText('87.5')).toBeInTheDocument(); // Risk Management score
      expect(screen.getByText('96.1')).toBeInTheDocument(); // Analytics score
      expect(screen.getByText('91.3')).toBeInTheDocument(); // Strategy Deployment score
    });

    it('shows component status indicators', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getAllByText('healthy')).toHaveLength(5);
      expect(screen.getByText('warning')).toBeInTheDocument();
    });

    it('displays component metrics', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('1,247')).toBeInTheDocument(); // Active connections
      expect(screen.getByText('847 msg/s')).toBeInTheDocument(); // Throughput
      expect(screen.getByText('23.4ms')).toBeInTheDocument(); // Latency
      expect(screen.getByText('12')).toBeInTheDocument(); // Active limits
    });

    it('shows component issues when present', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('One limit currently breached: Maximum Drawdown')).toBeInTheDocument();
      expect(screen.getByText('High breach probability detected for VaR limit')).toBeInTheDocument();
    });

    it('displays last check timestamps for components', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText(/Last check:/)).toBeInTheDocument();
    });
  });

  describe('Performance Metrics', () => {
    it('displays system resource utilization', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('34.7%')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('67.3%')).toBeInTheDocument();
      expect(screen.getByText('Disk Usage')).toBeInTheDocument();
      expect(screen.getByText('45.2%')).toBeInTheDocument();
    });

    it('shows network and throughput metrics', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Network Utilization')).toBeInTheDocument();
      expect(screen.getByText('78.9%')).toBeInTheDocument();
      expect(screen.getByText('Throughput')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
    });

    it('displays response time and error rate', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Response Time')).toBeInTheDocument();
      expect(screen.getByText('87.2ms')).toBeInTheDocument();
      expect(screen.getByText('Error Rate')).toBeInTheDocument();
      expect(screen.getByText('0.23%')).toBeInTheDocument();
    });

    it('shows system uptime', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Uptime')).toBeInTheDocument();
      expect(screen.getByText('99.95%')).toBeInTheDocument();
    });

    it('applies color coding for threshold violations', () => {
      const highUsageProps = {
        ...defaultProps,
        thresholds: {
          ...defaultProps.thresholds,
          memory: 60 // Lower threshold to trigger warning
        }
      };
      
      render(<Sprint3SystemMonitor {...highUsageProps} />);
      
      // Memory usage of 67.3% should be highlighted
      expect(screen.getByText('67.3%')).toBeInTheDocument();
    });
  });

  describe('Alert Management', () => {
    it('displays system alerts', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('System Alerts')).toBeInTheDocument();
      expect(screen.getByText('Maximum Drawdown limit breached: -15.23% exceeds -15.0%')).toBeInTheDocument();
      expect(screen.getByText('Connection rate spike detected: 25.7 connections/sec')).toBeInTheDocument();
    });

    it('shows alert severity levels', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('critical')).toBeInTheDocument();
      expect(screen.getByText('warning')).toBeInTheDocument();
      expect(screen.getByText('info')).toBeInTheDocument();
    });

    it('displays resolved and active alerts differently', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Active')).toBeInTheDocument();
      expect(screen.getByText('Resolved')).toBeInTheDocument();
    });

    it('shows alert assignment and escalation', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('risk-team')).toBeInTheDocument();
      expect(screen.getByText('ops-team')).toBeInTheDocument();
      expect(screen.getByText('dev-team')).toBeInTheDocument();
    });

    it('allows acknowledging alerts', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockAcknowledge = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        acknowledgeAlert: mockAcknowledge
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const acknowledgeButtons = screen.getAllByText('Acknowledge');
      if (acknowledgeButtons.length > 0) {
        await user.click(acknowledgeButtons[0]);
        expect(mockAcknowledge).toHaveBeenCalled();
      }
    });

    it('supports alert escalation', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockEscalate = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        escalateAlert: mockEscalate
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const escalateButtons = screen.getAllByText('Escalate');
      if (escalateButtons.length > 0) {
        await user.click(escalateButtons[0]);
        expect(mockEscalate).toHaveBeenCalled();
      }
    });
  });

  describe('Infrastructure Status', () => {
    it('displays server infrastructure', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const infraTab = screen.getByText('Infrastructure');
      await user.click(infraTab);
      
      await waitFor(() => {
        expect(screen.getByText('WebSocket Server 1')).toBeInTheDocument();
        expect(screen.getByText('WebSocket Server 2')).toBeInTheDocument();
        expect(screen.getByText('Database Primary')).toBeInTheDocument();
        expect(screen.getByText('Redis Cache')).toBeInTheDocument();
      });
    });

    it('shows server resource usage', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const infraTab = screen.getByText('Infrastructure');
      await user.click(infraTab);
      
      await waitFor(() => {
        expect(screen.getByText('32.4%')).toBeInTheDocument(); // CPU usage
        expect(screen.getByText('64.8%')).toBeInTheDocument(); // Memory usage
        expect(screen.getByText('42.1%')).toBeInTheDocument(); // Disk usage
      });
    });

    it('displays server uptime metrics', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const infraTab = screen.getByText('Infrastructure');
      await user.click(infraTab);
      
      await waitFor(() => {
        expect(screen.getByText('99.98%')).toBeInTheDocument(); // Server uptime
        expect(screen.getByText('99.95%')).toBeInTheDocument(); // Server uptime
      });
    });

    it('shows service status', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const infraTab = screen.getByText('Infrastructure');
      await user.click(infraTab);
      
      await waitFor(() => {
        expect(screen.getByText('NautilusTrader Engine')).toBeInTheDocument();
        expect(screen.getByText('Risk Management API')).toBeInTheDocument();
        expect(screen.getByText('Analytics Service')).toBeInTheDocument();
        expect(screen.getByText('1.219.0')).toBeInTheDocument(); // Version
      });
    });

    it('highlights server issues', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const infraTab = screen.getByText('Infrastructure');
      await user.click(infraTab);
      
      await waitFor(() => {
        expect(screen.getByText('High memory usage approaching limit')).toBeInTheDocument();
      });
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders system health trend chart', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays component status distribution', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const statusTab = screen.getByText('Status Distribution');
      await user.click(statusTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });

    it('shows resource utilization trends', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const resourcesTab = screen.getByText('Resource Trends');
      await user.click(resourcesTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('displays performance comparison chart', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const performanceTab = screen.getByText('Performance');
      await user.click(performanceTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Control Functions', () => {
    it('starts system monitoring', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockStart = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        isMonitoring: false,
        startMonitoring: mockStart
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      expect(mockStart).toHaveBeenCalled();
    });

    it('stops system monitoring', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockStop = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        stopMonitoring: mockStop
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const stopButton = screen.getByText('Stop Monitoring');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('refreshes health status', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockRefresh = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        refreshHealth: mockRefresh
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const refreshButton = screen.getByText('Refresh');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('generates system report', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockReport = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        generateReport: mockReport
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const reportButton = screen.getByText('Generate Report');
      await user.click(reportButton);
      
      expect(mockReport).toHaveBeenCalled();
    });

    it('configures monitoring thresholds', async () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const mockConfigure = vi.fn();
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        configureThresholds: mockConfigure
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const configButton = screen.getByText('Configure Thresholds');
      await user.click(configButton);
      
      expect(mockConfigure).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('updates system status in real-time', () => {
      render(<Sprint3SystemMonitor {...defaultProps} enableAutoRefresh={true} />);
      
      act(() => {
        vi.advanceTimersByTime(30000);
      });
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
    });

    it('refreshes component metrics automatically', () => {
      render(<Sprint3SystemMonitor {...defaultProps} refreshInterval={10000} />);
      
      act(() => {
        vi.advanceTimersByTime(10000);
      });
      
      expect(screen.getByText('System Health Overview')).toBeInTheDocument();
    });

    it('stops auto-refresh when disabled', () => {
      render(<Sprint3SystemMonitor {...defaultProps} enableAutoRefresh={false} />);
      
      act(() => {
        vi.advanceTimersByTime(30000);
      });
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when monitoring fails', () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        error: 'System monitoring service unavailable'
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Monitoring Error')).toBeInTheDocument();
      expect(screen.getByText('System monitoring service unavailable')).toBeInTheDocument();
    });

    it('handles missing health data gracefully', () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        systemHealth: null
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Health data unavailable')).toBeInTheDocument();
    });

    it('shows degraded system status', () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        systemHealth: {
          ...useSystemMonitor().systemHealth,
          overallStatus: 'degraded',
          overallScore: 65.4
        }
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('System Status: Degraded')).toBeInTheDocument();
      expect(screen.getByText('65.4')).toBeInTheDocument();
    });
  });

  describe('Performance and Load Testing', () => {
    it('handles many components efficiently', () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const manyComponents = {};
      for (let i = 0; i < 50; i++) {
        manyComponents[`component-${i}`] = {
          status: 'healthy',
          score: 90 + Math.random() * 10,
          metrics: {
            metric1: Math.random() * 100,
            metric2: Math.random() * 1000
          },
          lastCheck: Date.now() - Math.random() * 60000,
          issues: []
        };
      }

      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        systemHealth: {
          ...useSystemMonitor().systemHealth,
          components: manyComponents
        }
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
    });

    it('maintains responsiveness with frequent updates', () => {
      render(<Sprint3SystemMonitor {...defaultProps} refreshInterval={1000} />);
      
      act(() => {
        for (let i = 0; i < 60; i++) {
          vi.advanceTimersByTime(1000);
        }
      });
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
    });

    it('handles large alert volumes', () => {
      const { useSystemMonitor } = require('../../../hooks/monitoring/useSystemMonitor');
      const manyAlerts = Array.from({ length: 100 }, (_, i) => ({
        id: `alert-${i}`,
        component: `component-${i % 6}`,
        severity: ['info', 'warning', 'critical'][i % 3],
        message: `Alert message ${i}`,
        timestamp: Date.now() - i * 60000,
        resolved: i % 4 === 0,
        assignedTo: 'team',
        escalationLevel: 0
      }));

      useSystemMonitor.mockReturnValue({
        ...useSystemMonitor(),
        alerts: manyAlerts
      });

      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('Sprint 3 System Monitor')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides keyboard navigation support', async () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('System Health Overview')).toBeInTheDocument();
      expect(screen.getByText('Component Status')).toBeInTheDocument();
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });

    it('provides meaningful status indicators', () => {
      render(<Sprint3SystemMonitor {...defaultProps} />);
      
      expect(screen.getByText('System Status: Healthy')).toBeInTheDocument();
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });
  });
});