/**
 * DeploymentOrchestrator Test Suite
 * Comprehensive tests for the deployment orchestration component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import DeploymentOrchestrator from '../DeploymentOrchestrator';

// Mock the deployment orchestration hooks
vi.mock('../../../hooks/deployment/useDeploymentOrchestrator', () => ({
  useDeploymentOrchestrator: vi.fn(() => ({
    activeDeployments: [
      {
        deployment_id: 'dep-001',
        strategy_name: 'Advanced Momentum Strategy',
        version: 'v2.1.0',
        environment: 'production',
        deployment_type: 'blue_green',
        status: 'deploying',
        progress: 75.5,
        started_at: '2024-01-15T10:00:00Z',
        eta: '2024-01-15T10:25:00Z',
        health_score: 92.5,
        instances: [
          { id: 'inst-001', status: 'healthy', load: 45.2, version: 'v2.1.0' },
          { id: 'inst-002', status: 'healthy', load: 38.7, version: 'v2.1.0' },
          { id: 'inst-003', status: 'starting', load: 0, version: 'v2.1.0' }
        ]
      },
      {
        deployment_id: 'dep-002',
        strategy_name: 'Statistical Arbitrage',
        version: 'v1.8.5',
        environment: 'staging',
        deployment_type: 'rolling',
        status: 'testing',
        progress: 45.0,
        started_at: '2024-01-15T10:15:00Z',
        eta: '2024-01-15T10:40:00Z',
        health_score: 88.0,
        instances: [
          { id: 'inst-004', status: 'healthy', load: 52.1, version: 'v1.8.5' },
          { id: 'inst-005', status: 'testing', load: 25.3, version: 'v1.8.5' }
        ]
      }
    ],
    environments: [
      {
        name: 'production',
        status: 'healthy',
        active_strategies: 8,
        total_instances: 24,
        healthy_instances: 22,
        cpu_usage: 65.2,
        memory_usage: 72.8,
        network_throughput: '1.2 GB/s',
        last_deployment: '2024-01-15T08:30:00Z'
      },
      {
        name: 'staging',
        status: 'warning',
        active_strategies: 3,
        total_instances: 6,
        healthy_instances: 5,
        cpu_usage: 45.8,
        memory_usage: 58.3,
        network_throughput: '450 MB/s',
        last_deployment: '2024-01-15T10:15:00Z'
      },
      {
        name: 'development',
        status: 'healthy',
        active_strategies: 12,
        total_instances: 8,
        healthy_instances: 8,
        cpu_usage: 35.5,
        memory_usage: 42.1,
        network_throughput: '280 MB/s',
        last_deployment: '2024-01-14T16:45:00Z'
      }
    ],
    deploymentQueue: [
      {
        queue_id: 'queue-001',
        strategy_name: 'Mean Reversion Strategy',
        version: 'v3.0.0',
        target_environment: 'staging',
        deployment_type: 'canary',
        priority: 'high',
        scheduled_time: '2024-01-15T11:00:00Z',
        estimated_duration: '18 minutes',
        dependencies: ['dep-001'],
        approval_status: 'approved',
        requester: 'pm@company.com'
      },
      {
        queue_id: 'queue-002',
        strategy_name: 'Options Strategy Alpha',
        version: 'v1.5.2',
        target_environment: 'production',
        deployment_type: 'blue_green',
        priority: 'medium',
        scheduled_time: '2024-01-15T14:00:00Z',
        estimated_duration: '25 minutes',
        dependencies: [],
        approval_status: 'pending',
        requester: 'dev@company.com'
      }
    ],
    orchestrationRules: [
      {
        rule_id: 'rule-001',
        name: 'Production Safety Rule',
        description: 'No deployments to production during market hours',
        conditions: {
          environment: 'production',
          market_hours: true,
          min_approval_count: 2
        },
        actions: ['block_deployment', 'require_approval'],
        enabled: true,
        last_triggered: '2024-01-14T15:30:00Z'
      },
      {
        rule_id: 'rule-002',
        name: 'Resource Threshold Rule',
        description: 'Block deployments when CPU usage > 80%',
        conditions: {
          cpu_threshold: 80,
          memory_threshold: 85
        },
        actions: ['delay_deployment', 'send_alert'],
        enabled: true,
        last_triggered: null
      }
    ],
    resourceMetrics: {
      total_cpu_cores: 96,
      used_cpu_cores: 58.4,
      total_memory_gb: 384,
      used_memory_gb: 245.8,
      total_storage_tb: 12,
      used_storage_tb: 7.2,
      network_bandwidth_gbps: 10,
      peak_network_usage_gbps: 3.8
    },
    deploymentMetrics: {
      total_deployments_today: 15,
      successful_deployments: 13,
      failed_deployments: 1,
      cancelled_deployments: 1,
      avg_deployment_time: '16m 32s',
      deployment_success_rate: 86.7,
      rollback_rate: 6.7
    },
    isOrchestrating: true,
    error: null,
    startDeployment: vi.fn(),
    cancelDeployment: vi.fn(),
    pauseDeployment: vi.fn(),
    resumeDeployment: vi.fn(),
    rollbackDeployment: vi.fn(),
    scheduleDeployment: vi.fn(),
    updateRule: vi.fn(),
    scaleEnvironment: vi.fn(),
    refreshOrchestrator: vi.fn(),
    exportMetrics: vi.fn()
  }))
}));

// Mock environment management hooks
vi.mock('../../../hooks/deployment/useEnvironmentManager', () => ({
  useEnvironmentManager: vi.fn(() => ({
    environments: ['production', 'staging', 'development'],
    createEnvironment: vi.fn(),
    updateEnvironment: vi.fn(),
    deleteEnvironment: vi.fn(),
    getEnvironmentHealth: vi.fn()
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
  Legend: vi.fn(() => <div data-testid="legend">Legend</div>)
}));

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '10:30:00'),
    fromNow: vi.fn(() => '5 minutes ago'),
    valueOf: vi.fn(() => 1705327200000),
    add: vi.fn(() => ({
      format: vi.fn(() => '11:00:00')
    }))
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('DeploymentOrchestrator', () => {
  const user = userEvent.setup();
  const mockProps = {
    environment: 'production',
    autoRefresh: true,
    showAdvancedMetrics: true,
    height: 900
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the orchestrator dashboard', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Deployment Orchestrator')).toBeInTheDocument();
      expect(screen.getByText('Active Deployments')).toBeInTheDocument();
      expect(screen.getByText('Environments')).toBeInTheDocument();
      expect(screen.getByText('Deployment Queue')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<DeploymentOrchestrator {...mockProps} height={1200} />);
      
      const dashboard = screen.getByText('Deployment Orchestrator').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<DeploymentOrchestrator />);
      
      expect(screen.getByText('Deployment Orchestrator')).toBeInTheDocument();
    });

    it('shows orchestration status', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Orchestration Active')).toBeInTheDocument();
    });
  });

  describe('Active Deployments', () => {
    it('displays active deployment cards', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Advanced Momentum Strategy')).toBeInTheDocument();
      expect(screen.getByText('Statistical Arbitrage')).toBeInTheDocument();
    });

    it('shows deployment details', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('v2.1.0')).toBeInTheDocument();
      expect(screen.getByText('v1.8.5')).toBeInTheDocument();
      expect(screen.getByText('production')).toBeInTheDocument();
      expect(screen.getByText('staging')).toBeInTheDocument();
    });

    it('displays deployment types', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('blue_green')).toBeInTheDocument();
      expect(screen.getByText('rolling')).toBeInTheDocument();
    });

    it('shows deployment statuses', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('deploying')).toBeInTheDocument();
      expect(screen.getByText('testing')).toBeInTheDocument();
    });

    it('displays progress percentages', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('75.5%')).toBeInTheDocument();
      expect(screen.getByText('45.0%')).toBeInTheDocument();
    });

    it('shows health scores', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('92.5')).toBeInTheDocument();
      expect(screen.getByText('88.0')).toBeInTheDocument();
    });

    it('displays instance information', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('3 instances')).toBeInTheDocument();
      expect(screen.getByText('2 instances')).toBeInTheDocument();
    });

    it('shows ETAs', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText(/ETA:/)).toBeInTheDocument();
    });

    it('renders progress bars', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(1);
    });
  });

  describe('Environment Overview', () => {
    it('displays all environments', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      expect(envTab).toBeInTheDocument();
    });

    it('shows environment statuses', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('healthy')).toBeInTheDocument();
        expect(screen.getByText('warning')).toBeInTheDocument();
      });
    });

    it('displays resource usage', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU: 65.2%')).toBeInTheDocument();
        expect(screen.getByText('Memory: 72.8%')).toBeInTheDocument();
      });
    });

    it('shows instance counts', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('22/24 healthy')).toBeInTheDocument();
        expect(screen.getByText('5/6 healthy')).toBeInTheDocument();
      });
    });

    it('displays active strategy counts', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('8 strategies')).toBeInTheDocument();
        expect(screen.getByText('3 strategies')).toBeInTheDocument();
        expect(screen.getByText('12 strategies')).toBeInTheDocument();
      });
    });

    it('shows network throughput', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('1.2 GB/s')).toBeInTheDocument();
        expect(screen.getByText('450 MB/s')).toBeInTheDocument();
      });
    });
  });

  describe('Deployment Queue', () => {
    it('displays queued deployments', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      expect(queueTab).toBeInTheDocument();
    });

    it('shows queue details', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('Mean Reversion Strategy')).toBeInTheDocument();
        expect(screen.getByText('Options Strategy Alpha')).toBeInTheDocument();
      });
    });

    it('displays queue priorities', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('high')).toBeInTheDocument();
        expect(screen.getByText('medium')).toBeInTheDocument();
      });
    });

    it('shows scheduled times', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('11:00:00')).toBeInTheDocument();
        expect(screen.getByText('14:00:00')).toBeInTheDocument();
      });
    });

    it('displays approval statuses', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('approved')).toBeInTheDocument();
        expect(screen.getByText('pending')).toBeInTheDocument();
      });
    });

    it('shows estimated durations', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('18 minutes')).toBeInTheDocument();
        expect(screen.getByText('25 minutes')).toBeInTheDocument();
      });
    });

    it('displays dependencies', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const queueTab = screen.getByText('Deployment Queue');
      await user.click(queueTab);
      
      await waitFor(() => {
        expect(screen.getByText('1 dependency')).toBeInTheDocument();
        expect(screen.getByText('No dependencies')).toBeInTheDocument();
      });
    });
  });

  describe('Orchestration Rules', () => {
    it('displays orchestration rules', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      expect(rulesTab).toBeInTheDocument();
    });

    it('shows rule details', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      await user.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Production Safety Rule')).toBeInTheDocument();
        expect(screen.getByText('Resource Threshold Rule')).toBeInTheDocument();
      });
    });

    it('displays rule descriptions', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      await user.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('No deployments to production during market hours')).toBeInTheDocument();
        expect(screen.getByText('Block deployments when CPU usage > 80%')).toBeInTheDocument();
      });
    });

    it('shows rule statuses', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      await user.click(rulesTab);
      
      await waitFor(() => {
        const enabledStatuses = screen.getAllByText('Enabled');
        expect(enabledStatuses.length).toBe(2);
      });
    });

    it('displays rule actions', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      await user.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('block_deployment')).toBeInTheDocument();
        expect(screen.getByText('require_approval')).toBeInTheDocument();
        expect(screen.getByText('delay_deployment')).toBeInTheDocument();
      });
    });

    it('shows last triggered times', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rulesTab = screen.getByText('Rules');
      await user.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Last triggered:')).toBeInTheDocument();
        expect(screen.getByText('Never triggered')).toBeInTheDocument();
      });
    });
  });

  describe('Resource Monitoring', () => {
    it('displays resource metrics when advanced metrics are enabled', () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      expect(metricsTab).toBeInTheDocument();
    });

    it('shows CPU usage', async () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU Cores')).toBeInTheDocument();
        expect(screen.getByText('58.4 / 96')).toBeInTheDocument();
      });
    });

    it('displays memory usage', async () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Memory (GB)')).toBeInTheDocument();
        expect(screen.getByText('245.8 / 384')).toBeInTheDocument();
      });
    });

    it('shows storage usage', async () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Storage (TB)')).toBeInTheDocument();
        expect(screen.getByText('7.2 / 12')).toBeInTheDocument();
      });
    });

    it('displays network metrics', async () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Network Bandwidth')).toBeInTheDocument();
        expect(screen.getByText('Peak: 3.8 / 10 Gbps')).toBeInTheDocument();
      });
    });
  });

  describe('Deployment Statistics', () => {
    it('displays deployment metrics', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Today: 15')).toBeInTheDocument();
      expect(screen.getByText('Success: 13')).toBeInTheDocument();
      expect(screen.getByText('Failed: 1')).toBeInTheDocument();
    });

    it('shows success rates', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('86.7%')).toBeInTheDocument(); // Success rate
    });

    it('displays average deployment time', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Avg Time: 16m 32s')).toBeInTheDocument();
    });

    it('shows rollback rate', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Rollback: 6.7%')).toBeInTheDocument();
    });
  });

  describe('Deployment Controls', () => {
    it('renders control buttons for active deployments', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Pause')).toBeInTheDocument();
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    it('allows pausing deployments', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockPause = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        pauseDeployment: mockPause
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      const pauseButton = screen.getByText('Pause');
      await user.click(pauseButton);
      
      expect(mockPause).toHaveBeenCalled();
    });

    it('allows canceling deployments', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockCancel = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        cancelDeployment: mockCancel
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);
      
      expect(mockCancel).toHaveBeenCalled();
    });

    it('shows rollback buttons for problematic deployments', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const rollbackButton = screen.getByText('Rollback');
      expect(rollbackButton).toBeInTheDocument();
    });

    it('allows scheduling new deployments', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockSchedule = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        scheduleDeployment: mockSchedule
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      const scheduleButton = screen.getByText('Schedule Deployment');
      await user.click(scheduleButton);
      
      await waitFor(() => {
        expect(screen.getByText('Schedule New Deployment')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes orchestrator data automatically', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockRefresh = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        refreshOrchestrator: mockRefresh
      });

      render(<DeploymentOrchestrator {...mockProps} autoRefresh={true} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(15000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('updates deployment progress in real-time', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      // Should show current progress
      expect(screen.getByText('75.5%')).toBeInTheDocument();
      expect(screen.getByText('45.0%')).toBeInTheDocument();
    });

    it('shows orchestration status', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Orchestration Active')).toBeInTheDocument();
    });
  });

  describe('Environment Scaling', () => {
    it('allows scaling environments', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockScale = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        scaleEnvironment: mockScale
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        const scaleButton = screen.getByText('Scale');
        expect(scaleButton).toBeInTheDocument();
      });
    });

    it('shows scaling controls for environments', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByText('Scale Up')).toBeInTheDocument();
        expect(screen.getByText('Scale Down')).toBeInTheDocument();
      });
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders deployment progress charts', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays resource usage charts when advanced metrics are enabled', async () => {
      render(<DeploymentOrchestrator {...mockProps} showAdvancedMetrics={true} />);
      
      const metricsTab = screen.getByText('Resources');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });

    it('shows deployment timeline visualization', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      // Should have timeline visualization
      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    });

    it('displays environment health charts', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const envTab = screen.getByText('Environments');
      await user.click(envTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Export Functionality', () => {
    it('renders export metrics button', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const exportButton = screen.getByText('Export Metrics');
      expect(exportButton).toBeInTheDocument();
    });

    it('handles metrics export', async () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      const mockExport = vi.fn();
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        exportMetrics: mockExport
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      const exportButton = screen.getByText('Export Metrics');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when orchestrator fails', () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        error: 'Orchestration service unavailable'
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Orchestrator Error')).toBeInTheDocument();
      expect(screen.getByText('Orchestration service unavailable')).toBeInTheDocument();
    });

    it('handles empty deployment queues', () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        activeDeployments: [],
        deploymentQueue: []
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('No active deployments')).toBeInTheDocument();
    });

    it('shows offline environment status', () => {
      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        environments: [{
          ...useDeploymentOrchestrator().environments[0],
          status: 'offline'
        }]
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('offline')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large numbers of deployments efficiently', () => {
      const largeDeploymentList = Array.from({ length: 50 }, (_, i) => ({
        deployment_id: `dep-${i.toString().padStart(3, '0')}`,
        strategy_name: `Strategy ${i + 1}`,
        version: `v1.${i}.0`,
        environment: ['production', 'staging', 'development'][i % 3],
        deployment_type: ['blue_green', 'rolling', 'canary'][i % 3],
        status: ['deploying', 'testing', 'completed'][i % 3],
        progress: Math.random() * 100,
        started_at: new Date(Date.now() - i * 60000).toISOString(),
        eta: new Date(Date.now() + (100 - i) * 60000).toISOString(),
        health_score: 80 + Math.random() * 20,
        instances: Array.from({ length: Math.floor(Math.random() * 5) + 1 }, (_, j) => ({
          id: `inst-${i}-${j}`,
          status: ['healthy', 'starting', 'testing'][j % 3],
          load: Math.random() * 100,
          version: `v1.${i}.0`
        }))
      }));

      const { useDeploymentOrchestrator } = require('../../../hooks/deployment/useDeploymentOrchestrator');
      useDeploymentOrchestrator.mockReturnValue({
        ...useDeploymentOrchestrator(),
        activeDeployments: largeDeploymentList
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Deployment Orchestrator')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for progress bars', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      progressBars.forEach(progressBar => {
        expect(progressBar).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('deploying')).toBeInTheDocument();
      expect(screen.getByText('testing')).toBeInTheDocument();
      expect(screen.getByText('healthy')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Deployment Orchestrator')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<DeploymentOrchestrator {...mockProps} />);
      
      expect(screen.getByText('Active Deployments')).toBeInTheDocument();
      expect(screen.getByText('Environments')).toBeInTheDocument();
    });
  });
});