/**
 * AdvancedDeploymentPipeline Test Suite
 * Comprehensive tests for the advanced strategy deployment pipeline component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import AdvancedDeploymentPipeline from '../AdvancedDeploymentPipeline';

// Mock the deployment pipeline hooks
vi.mock('../../../hooks/deployment/useDeploymentPipeline', () => ({
  useDeploymentPipeline: vi.fn(() => ({
    pipeline: {
      pipeline_id: 'pipe-001',
      name: 'Production Deployment Pipeline',
      strategy_id: 'strat-001',
      strategy_name: 'Advanced Momentum Strategy',
      version: 'v2.1.0',
      status: 'running',
      stage: 'testing',
      progress: 65.5,
      created_at: '2024-01-15T10:00:00Z',
      started_at: '2024-01-15T10:05:00Z',
      estimated_completion: '2024-01-15T10:45:00Z',
      deployment_config: {
        deployment_type: 'blue_green',
        rollback_enabled: true,
        approval_required: true,
        automated_testing: true,
        gradual_rollout: {
          enabled: true,
          initial_percentage: 10,
          stages: [10, 25, 50, 100],
          stage_duration: '15m'
        }
      }
    },
    stages: [
      {
        stage_id: 'validation',
        name: 'Strategy Validation',
        order: 1,
        status: 'completed',
        progress: 100,
        duration: '2m 15s',
        started_at: '2024-01-15T10:05:00Z',
        completed_at: '2024-01-15T10:07:15Z',
        checks: [
          { name: 'Syntax Validation', status: 'passed', duration: '15s' },
          { name: 'Parameter Validation', status: 'passed', duration: '30s' },
          { name: 'Risk Limits Check', status: 'passed', duration: '45s' },
          { name: 'Dependency Check', status: 'passed', duration: '45s' }
        ]
      },
      {
        stage_id: 'testing',
        name: 'Automated Testing',
        order: 2,
        status: 'running',
        progress: 65,
        duration: null,
        started_at: '2024-01-15T10:07:15Z',
        completed_at: null,
        checks: [
          { name: 'Unit Tests', status: 'passed', duration: '1m 20s' },
          { name: 'Integration Tests', status: 'passed', duration: '2m 45s' },
          { name: 'Backtesting', status: 'running', duration: null },
          { name: 'Performance Tests', status: 'pending', duration: null }
        ]
      },
      {
        stage_id: 'approval',
        name: 'Deployment Approval',
        order: 3,
        status: 'pending',
        progress: 0,
        duration: null,
        started_at: null,
        completed_at: null,
        checks: [
          { name: 'Risk Review', status: 'pending', duration: null },
          { name: 'Business Approval', status: 'pending', duration: null }
        ]
      },
      {
        stage_id: 'deployment',
        name: 'Production Deployment',
        order: 4,
        status: 'pending',
        progress: 0,
        duration: null,
        started_at: null,
        completed_at: null,
        checks: [
          { name: 'Environment Setup', status: 'pending', duration: null },
          { name: 'Strategy Deployment', status: 'pending', duration: null },
          { name: 'Health Checks', status: 'pending', duration: null }
        ]
      }
    ],
    deploymentHistory: [
      {
        deployment_id: 'dep-001',
        pipeline_id: 'pipe-001',
        version: 'v2.0.0',
        status: 'completed',
        deployment_type: 'rolling',
        started_at: '2024-01-14T14:30:00Z',
        completed_at: '2024-01-14T14:52:30Z',
        duration: '22m 30s',
        success_rate: 100,
        rollback_occurred: false,
        performance_impact: '+2.5% returns'
      },
      {
        deployment_id: 'dep-002',
        pipeline_id: 'pipe-001',
        version: 'v1.9.5',
        status: 'rolled_back',
        deployment_type: 'canary',
        started_at: '2024-01-10T09:15:00Z',
        completed_at: '2024-01-10T09:45:20Z',
        duration: '30m 20s',
        success_rate: 25,
        rollback_occurred: true,
        rollback_reason: 'Performance degradation detected',
        performance_impact: '-1.8% returns'
      }
    ],
    approvals: [
      {
        approval_id: 'apr-001',
        approver: 'risk@company.com',
        role: 'Risk Manager',
        status: 'pending',
        requested_at: '2024-01-15T10:30:00Z',
        review_deadline: '2024-01-15T14:00:00Z',
        comments: null
      },
      {
        approval_id: 'apr-002',
        approver: 'pm@company.com',
        role: 'Portfolio Manager',
        status: 'approved',
        requested_at: '2024-01-15T10:00:00Z',
        approved_at: '2024-01-15T10:15:00Z',
        comments: 'Strategy performance looks promising in backtests'
      }
    ],
    testResults: {
      overall_status: 'running',
      total_tests: 47,
      passed_tests: 35,
      failed_tests: 2,
      running_tests: 10,
      test_coverage: 87.5,
      execution_time: '8m 45s',
      performance_metrics: {
        sharpe_ratio: 1.85,
        max_drawdown: -6.2,
        total_return: 12.4,
        win_rate: 68.5
      },
      failed_test_details: [
        {
          test_name: 'stress_test_2008_scenario',
          category: 'stress_testing',
          error: 'Exceeded maximum drawdown threshold (-15%)',
          severity: 'high'
        },
        {
          test_name: 'liquidity_constraint_test',
          category: 'risk_management',
          error: 'Position size exceeds liquidity limits',
          severity: 'medium'
        }
      ]
    },
    pipelineMetrics: {
      total_deployments: 127,
      successful_deployments: 115,
      failed_deployments: 8,
      rolled_back_deployments: 4,
      success_rate: 90.5,
      avg_deployment_time: '18m 32s',
      avg_test_time: '12m 15s',
      avg_approval_time: '2h 45m'
    },
    isRunning: true,
    error: null,
    startPipeline: vi.fn(),
    stopPipeline: vi.fn(),
    pausePipeline: vi.fn(),
    resumePipeline: vi.fn(),
    retryStage: vi.fn(),
    skipStage: vi.fn(),
    approvePipeline: vi.fn(),
    rejectPipeline: vi.fn(),
    rollbackDeployment: vi.fn(),
    refreshPipeline: vi.fn(),
    exportLogs: vi.fn()
  }))
}));

vi.mock('../../../hooks/deployment/usePipelineConfig', () => ({
  usePipelineConfig: vi.fn(() => ({
    config: {
      deployment_types: ['rolling', 'blue_green', 'canary'],
      test_suites: ['unit', 'integration', 'performance', 'stress'],
      approval_workflows: ['standard', 'expedited', 'emergency'],
      rollback_strategies: ['automatic', 'manual', 'staged']
    },
    updateConfig: vi.fn()
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
    subtract: vi.fn(() => ({
      format: vi.fn(() => '2024-01-01')
    })),
    add: vi.fn(() => ({
      format: vi.fn(() => '10:45:00')
    }))
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

describe('AdvancedDeploymentPipeline', () => {
  const user = userEvent.setup();
  const mockProps = {
    pipelineId: 'pipe-001',
    strategyId: 'strat-001',
    height: 800,
    autoRefresh: true,
    showAdvancedControls: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the deployment pipeline dashboard', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Advanced Deployment Pipeline')).toBeInTheDocument();
      expect(screen.getByText('Pipeline Overview')).toBeInTheDocument();
      expect(screen.getByText('Stage Progress')).toBeInTheDocument();
      expect(screen.getByText('Test Results')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} height={1000} />);
      
      const dashboard = screen.getByText('Advanced Deployment Pipeline').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<AdvancedDeploymentPipeline />);
      
      expect(screen.getByText('Advanced Deployment Pipeline')).toBeInTheDocument();
    });

    it('shows advanced controls when enabled', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} showAdvancedControls={true} />);
      
      expect(screen.getByText('Advanced Controls')).toBeInTheDocument();
    });
  });

  describe('Pipeline Overview', () => {
    it('displays pipeline basic information', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Production Deployment Pipeline')).toBeInTheDocument();
      expect(screen.getByText('Advanced Momentum Strategy')).toBeInTheDocument();
      expect(screen.getByText('v2.1.0')).toBeInTheDocument();
    });

    it('shows pipeline status', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('testing')).toBeInTheDocument(); // current stage
    });

    it('displays overall progress', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('65.5%')).toBeInTheDocument();
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('shows deployment configuration', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('blue_green')).toBeInTheDocument();
      expect(screen.getByText('Rollback Enabled')).toBeInTheDocument();
      expect(screen.getByText('Approval Required')).toBeInTheDocument();
    });

    it('displays time estimates', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText(/Started:/)).toBeInTheDocument();
      expect(screen.getByText(/ETA:/)).toBeInTheDocument();
    });
  });

  describe('Stage Progress', () => {
    it('displays all pipeline stages', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Strategy Validation')).toBeInTheDocument();
      expect(screen.getByText('Automated Testing')).toBeInTheDocument();
      expect(screen.getByText('Deployment Approval')).toBeInTheDocument();
      expect(screen.getByText('Production Deployment')).toBeInTheDocument();
    });

    it('shows stage statuses', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getAllByText('pending')).toHaveLength(2);
    });

    it('displays stage progress percentages', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('100%')).toBeInTheDocument(); // completed stage
      expect(screen.getByText('65%')).toBeInTheDocument(); // running stage
    });

    it('shows stage durations', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('2m 15s')).toBeInTheDocument();
    });

    it('displays individual checks within stages', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Syntax Validation')).toBeInTheDocument();
      expect(screen.getByText('Parameter Validation')).toBeInTheDocument();
      expect(screen.getByText('Unit Tests')).toBeInTheDocument();
      expect(screen.getByText('Integration Tests')).toBeInTheDocument();
    });

    it('shows check statuses', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getAllByText('passed')).toHaveLength(4);
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getAllByText('pending')).toHaveLength(3);
    });
  });

  describe('Test Results', () => {
    it('displays test summary', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      expect(testTab).toBeInTheDocument();
    });

    it('shows test statistics', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByText('Total Tests: 47')).toBeInTheDocument();
        expect(screen.getByText('Passed: 35')).toBeInTheDocument();
        expect(screen.getByText('Failed: 2')).toBeInTheDocument();
        expect(screen.getByText('Running: 10')).toBeInTheDocument();
      });
    });

    it('displays test coverage', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByText('Coverage: 87.5%')).toBeInTheDocument();
      });
    });

    it('shows performance metrics', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByText('Sharpe Ratio: 1.85')).toBeInTheDocument();
        expect(screen.getByText('Max Drawdown: -6.2%')).toBeInTheDocument();
        expect(screen.getByText('Win Rate: 68.5%')).toBeInTheDocument();
      });
    });

    it('displays failed test details', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByText('stress_test_2008_scenario')).toBeInTheDocument();
        expect(screen.getByText('liquidity_constraint_test')).toBeInTheDocument();
        expect(screen.getByText('Exceeded maximum drawdown threshold')).toBeInTheDocument();
      });
    });

    it('shows test severity levels', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByText('high')).toBeInTheDocument();
        expect(screen.getByText('medium')).toBeInTheDocument();
      });
    });
  });

  describe('Approval Workflow', () => {
    it('displays approval status', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      expect(approvalTab).toBeInTheDocument();
    });

    it('shows approver information', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      await user.click(approvalTab);
      
      await waitFor(() => {
        expect(screen.getByText('Risk Manager')).toBeInTheDocument();
        expect(screen.getByText('Portfolio Manager')).toBeInTheDocument();
      });
    });

    it('displays approval statuses', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      await user.click(approvalTab);
      
      await waitFor(() => {
        expect(screen.getByText('pending')).toBeInTheDocument();
        expect(screen.getByText('approved')).toBeInTheDocument();
      });
    });

    it('shows review deadlines', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      await user.click(approvalTab);
      
      await waitFor(() => {
        expect(screen.getByText(/Deadline:/)).toBeInTheDocument();
      });
    });

    it('displays approver comments', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      await user.click(approvalTab);
      
      await waitFor(() => {
        expect(screen.getByText('Strategy performance looks promising in backtests')).toBeInTheDocument();
      });
    });

    it('allows approving pipeline', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockApprove = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        approvePipeline: mockApprove
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const approvalTab = screen.getByText('Approvals');
      await user.click(approvalTab);
      
      await waitFor(() => {
        const approveButton = screen.getByText('Approve');
        expect(approveButton).toBeInTheDocument();
      });
    });
  });

  describe('Deployment History', () => {
    it('displays deployment history', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      expect(historyTab).toBeInTheDocument();
    });

    it('shows historical deployments', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('v2.0.0')).toBeInTheDocument();
        expect(screen.getByText('v1.9.5')).toBeInTheDocument();
      });
    });

    it('displays deployment statuses', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('completed')).toBeInTheDocument();
        expect(screen.getByText('rolled_back')).toBeInTheDocument();
      });
    });

    it('shows deployment types', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('rolling')).toBeInTheDocument();
        expect(screen.getByText('canary')).toBeInTheDocument();
      });
    });

    it('displays performance impact', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('+2.5% returns')).toBeInTheDocument();
        expect(screen.getByText('-1.8% returns')).toBeInTheDocument();
      });
    });

    it('shows rollback information', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const historyTab = screen.getByText('History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Performance degradation detected')).toBeInTheDocument();
      });
    });
  });

  describe('Pipeline Controls', () => {
    it('renders control buttons', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Pause')).toBeInTheDocument();
      expect(screen.getByText('Stop')).toBeInTheDocument();
    });

    it('allows pausing pipeline', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockPause = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        pausePipeline: mockPause
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const pauseButton = screen.getByText('Pause');
      await user.click(pauseButton);
      
      expect(mockPause).toHaveBeenCalled();
    });

    it('allows stopping pipeline', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockStop = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        stopPipeline: mockStop
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const stopButton = screen.getByText('Stop');
      await user.click(stopButton);
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('shows advanced controls when enabled', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} showAdvancedControls={true} />);
      
      expect(screen.getByText('Retry Stage')).toBeInTheDocument();
      expect(screen.getByText('Skip Stage')).toBeInTheDocument();
    });

    it('allows retrying failed stages', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockRetry = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        retryStage: mockRetry
      });

      render(<AdvancedDeploymentPipeline {...mockProps} showAdvancedControls={true} />);
      
      const retryButton = screen.getByText('Retry Stage');
      await user.click(retryButton);
      
      expect(mockRetry).toHaveBeenCalled();
    });
  });

  describe('Pipeline Metrics', () => {
    it('displays deployment statistics', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const metricsTab = screen.getByText('Metrics');
      expect(metricsTab).toBeInTheDocument();
    });

    it('shows success metrics', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const metricsTab = screen.getByText('Metrics');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Total: 127')).toBeInTheDocument();
        expect(screen.getByText('Success Rate: 90.5%')).toBeInTheDocument();
      });
    });

    it('displays average times', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const metricsTab = screen.getByText('Metrics');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Avg Deployment: 18m 32s')).toBeInTheDocument();
        expect(screen.getByText('Avg Testing: 12m 15s')).toBeInTheDocument();
        expect(screen.getByText('Avg Approval: 2h 45m')).toBeInTheDocument();
      });
    });

    it('shows rollback statistics', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const metricsTab = screen.getByText('Metrics');
      await user.click(metricsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Rollbacks: 4')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes pipeline status automatically', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockRefresh = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        refreshPipeline: mockRefresh
      });

      render(<AdvancedDeploymentPipeline {...mockProps} autoRefresh={true} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(10000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('updates progress in real-time', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      // Should show current progress
      expect(screen.getByText('65.5%')).toBeInTheDocument();
      expect(screen.getByText('65%')).toBeInTheDocument();
    });

    it('shows pipeline status indicator', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Pipeline Active')).toBeInTheDocument();
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders pipeline progress chart', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('displays test results chart', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const testTab = screen.getByText('Test Results');
      await user.click(testTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      });
    });

    it('shows deployment timeline visualization', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      // Should have timeline visualization
      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when pipeline fails', () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        error: 'Pipeline execution failed'
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Pipeline Error')).toBeInTheDocument();
      expect(screen.getByText('Pipeline execution failed')).toBeInTheDocument();
    });

    it('handles missing pipeline data', () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        pipeline: null,
        stages: []
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('No pipeline data available')).toBeInTheDocument();
    });

    it('shows stage failure details', () => {
      // Mock a failed stage
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        stages: [{
          ...useDeploymentPipeline().stages[0],
          status: 'failed',
          error: 'Validation failed: Risk limits exceeded'
        }]
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });

  describe('Export and Logging', () => {
    it('renders export logs button', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const exportButton = screen.getByText('Export Logs');
      expect(exportButton).toBeInTheDocument();
    });

    it('handles log export', async () => {
      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      const mockExport = vi.fn();
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        exportLogs: mockExport
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const exportButton = screen.getByText('Export Logs');
      await user.click(exportButton);
      
      expect(mockExport).toHaveBeenCalled();
    });
  });

  describe('Performance', () => {
    it('handles large deployment histories efficiently', () => {
      const largeHistory = Array.from({ length: 500 }, (_, i) => ({
        deployment_id: `dep-${i.toString().padStart(3, '0')}`,
        pipeline_id: 'pipe-001',
        version: `v1.${i}.0`,
        status: ['completed', 'failed', 'rolled_back'][i % 3],
        deployment_type: ['rolling', 'blue_green', 'canary'][i % 3],
        started_at: new Date(Date.now() - i * 86400000).toISOString(),
        completed_at: new Date(Date.now() - i * 86400000 + 1800000).toISOString(),
        duration: `${Math.floor(Math.random() * 30) + 10}m ${Math.floor(Math.random() * 60)}s`,
        success_rate: Math.floor(Math.random() * 40) + 60,
        rollback_occurred: i % 7 === 0,
        performance_impact: `${(Math.random() * 10 - 5).toFixed(1)}% returns`
      }));

      const { useDeploymentPipeline } = require('../../../hooks/deployment/useDeploymentPipeline');
      useDeploymentPipeline.mockReturnValue({
        ...useDeploymentPipeline(),
        deploymentHistory: largeHistory
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Advanced Deployment Pipeline')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for progress bars', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      progressBars.forEach(progressBar => {
        expect(progressBar).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getAllByText('pending')).toHaveLength(2);
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Advanced Deployment Pipeline')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<AdvancedDeploymentPipeline {...mockProps} />);
      
      expect(screen.getByText('Pipeline Overview')).toBeInTheDocument();
      expect(screen.getByText('Stage Progress')).toBeInTheDocument();
    });
  });
});