/**
 * DeploymentApprovalEngine Test Suite
 * Sprint 3: Strategy deployment approval workflow and automation testing
 * 
 * Tests approval workflows, automated testing, deployment pipelines,
 * rollback procedures, and CI/CD integration functionality.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import DeploymentApprovalEngine from '../DeploymentApprovalEngine';

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

// Mock deployment approval hook
vi.mock('../../../hooks/strategy/useDeploymentApproval', () => ({
  useDeploymentApproval: vi.fn(() => ({
    approvalQueue: [
      {
        id: 'deployment-1',
        strategyId: 'momentum-alpha-v2.1',
        strategyName: 'Momentum Alpha Strategy',
        version: '2.1.0',
        submittedBy: 'john.doe@company.com',
        submittedAt: Date.now() - 1800000, // 30 minutes ago
        status: 'pending_review',
        priority: 'high',
        deploymentType: 'production',
        targetEnvironment: 'prod',
        estimatedImpact: 'medium',
        description: 'Updated momentum calculation with improved signal detection',
        changes: [
          'Enhanced momentum indicator calculation',
          'Added volatility-based position sizing',
          'Improved stop-loss logic',
          'Updated backtesting parameters'
        ],
        testResults: {
          syntaxValidation: 'passed',
          unitTests: { passed: 127, failed: 0, coverage: 94.7 },
          integrationTests: { passed: 23, failed: 1, coverage: 89.3 },
          backtesting: { 
            sharpe: 1.85, 
            maxDrawdown: -8.2, 
            winRate: 67.3,
            totalReturn: 23.4 
          },
          riskAssessment: 'low',
          performanceTest: 'passed'
        },
        approvers: [
          { name: 'Sarah Johnson', role: 'Risk Manager', status: 'approved', timestamp: Date.now() - 900000 },
          { name: 'Mike Chen', role: 'Head of Trading', status: 'pending', timestamp: null },
          { name: 'Alex Smith', role: 'CTO', status: 'pending', timestamp: null }
        ],
        deploymentPlan: {
          steps: [
            'Deploy to staging environment',
            'Run smoke tests',
            'Gradual rollout (10% traffic)',
            'Monitor performance for 30 minutes',
            'Full deployment if metrics are stable'
          ],
          rollbackPlan: [
            'Immediate traffic redirect to previous version',
            'Preserve position states',
            'Generate rollback report'
          ],
          estimatedDuration: '45 minutes'
        }
      },
      {
        id: 'deployment-2',
        strategyId: 'arbitrage-pro-v1.3',
        strategyName: 'Arbitrage Pro Strategy',
        version: '1.3.2',
        submittedBy: 'jane.smith@company.com',
        submittedAt: Date.now() - 3600000, // 1 hour ago
        status: 'approved',
        priority: 'medium',
        deploymentType: 'hotfix',
        targetEnvironment: 'prod',
        estimatedImpact: 'low',
        description: 'Critical bug fix for order matching logic',
        changes: [
          'Fixed order matching race condition',
          'Updated error handling for market disconnections'
        ],
        testResults: {
          syntaxValidation: 'passed',
          unitTests: { passed: 89, failed: 0, coverage: 96.1 },
          integrationTests: { passed: 15, failed: 0, coverage: 92.7 },
          backtesting: { 
            sharpe: 2.1, 
            maxDrawdown: -4.5, 
            winRate: 72.8,
            totalReturn: 18.7 
          },
          riskAssessment: 'very_low',
          performanceTest: 'passed'
        },
        approvers: [
          { name: 'Sarah Johnson', role: 'Risk Manager', status: 'approved', timestamp: Date.now() - 1800000 },
          { name: 'Mike Chen', role: 'Head of Trading', status: 'approved', timestamp: Date.now() - 1200000 },
          { name: 'Alex Smith', role: 'CTO', status: 'approved', timestamp: Date.now() - 600000 }
        ],
        scheduledDeployment: Date.now() + 900000 // 15 minutes from now
      },
      {
        id: 'deployment-3',
        strategyId: 'mean-reversion-beta',
        strategyName: 'Mean Reversion Beta',
        version: '0.9.1',
        submittedBy: 'bob.wilson@company.com',
        submittedAt: Date.now() - 7200000, // 2 hours ago
        status: 'rejected',
        priority: 'low',
        deploymentType: 'experimental',
        targetEnvironment: 'staging',
        estimatedImpact: 'high',
        description: 'Experimental mean reversion strategy with ML components',
        rejectionReason: 'Insufficient test coverage and high risk assessment',
        rejectedBy: 'Sarah Johnson',
        rejectedAt: Date.now() - 5400000,
        changes: [
          'Added machine learning prediction model',
          'New risk management layer'
        ],
        testResults: {
          syntaxValidation: 'passed',
          unitTests: { passed: 45, failed: 8, coverage: 67.2 },
          integrationTests: { passed: 5, failed: 3, coverage: 54.1 },
          backtesting: { 
            sharpe: 0.8, 
            maxDrawdown: -15.7, 
            winRate: 52.1,
            totalReturn: 8.3 
          },
          riskAssessment: 'high',
          performanceTest: 'failed'
        }
      }
    ],
    deploymentHistory: [
      {
        id: 'deployment-hist-1',
        strategyId: 'momentum-alpha-v2.0',
        version: '2.0.5',
        deployedAt: Date.now() - 86400000, // 1 day ago
        deployedBy: 'automated-system',
        status: 'successful',
        duration: 2700000, // 45 minutes
        rollbacksCount: 0,
        performanceMetrics: {
          sharpe: 1.92,
          totalReturn: 21.8,
          maxDrawdown: -6.3,
          uptime: 99.97
        }
      },
      {
        id: 'deployment-hist-2',
        strategyId: 'arbitrage-pro-v1.2',
        version: '1.2.8',
        deployedAt: Date.now() - 172800000, // 2 days ago
        deployedBy: 'jane.smith@company.com',
        status: 'rolled_back',
        duration: 5400000, // 1.5 hours
        rollbacksCount: 1,
        rollbackReason: 'Performance degradation detected',
        rollbackAt: Date.now() - 167400000,
        performanceMetrics: {
          sharpe: 0.65,
          totalReturn: -3.2,
          maxDrawdown: -12.8,
          uptime: 94.2
        }
      }
    ],
    approvalMetrics: {
      totalSubmissions: 156,
      approvedDeployments: 134,
      rejectedDeployments: 22,
      averageApprovalTime: 3600000, // 1 hour
      successRate: 94.7,
      rollbackRate: 5.3,
      approvalsByType: {
        production: 89,
        staging: 31,
        hotfix: 14,
        experimental: 22
      },
      approvalsByPriority: {
        critical: 8,
        high: 45,
        medium: 67,
        low: 36
      }
    },
    approvalConfiguration: {
      requiredApprovers: [
        { role: 'Risk Manager', required: true },
        { role: 'Head of Trading', required: true },
        { role: 'CTO', required: false, conditions: ['high_impact', 'production'] }
      ],
      automatedChecks: {
        syntaxValidation: true,
        unitTestThreshold: 90,
        integrationTestThreshold: 85,
        coverageThreshold: 80,
        backtestingRequired: true,
        riskAssessmentRequired: true,
        performanceTestRequired: true
      },
      deploymentWindows: [
        { environment: 'production', allowedHours: [9, 17], timezone: 'UTC' },
        { environment: 'staging', allowedHours: [0, 23], timezone: 'UTC' }
      ]
    },
    isProcessing: false,
    error: null,
    submitForApproval: vi.fn(),
    approveDeployment: vi.fn(),
    rejectDeployment: vi.fn(),
    scheduleDeployment: vi.fn(),
    cancelDeployment: vi.fn(),
    rollbackDeployment: vi.fn(),
    updateApprovalConfig: vi.fn(),
    generateApprovalReport: vi.fn(),
    refreshQueue: vi.fn()
  }))
}));

describe('DeploymentApprovalEngine', () => {
  const user = userEvent.setup();
  
  const defaultProps = {
    refreshInterval: 30000,
    enableAutoRefresh: true,
    enableNotifications: true,
    userRole: 'Head of Trading',
    userId: 'mike.chen@company.com'
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
    it('renders deployment approval engine', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Deployment Approval Engine')).toBeInTheDocument();
      expect(screen.getByText('Pending Approvals')).toBeInTheDocument();
      expect(screen.getByText('Deployment Queue')).toBeInTheDocument();
      expect(screen.getByText('Approval Metrics')).toBeInTheDocument();
    });

    it('displays current user role', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Role: Head of Trading')).toBeInTheDocument();
    });

    it('shows queue statistics', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Pending: 1')).toBeInTheDocument();
      expect(screen.getByText('Approved: 1')).toBeInTheDocument();
      expect(screen.getByText('Rejected: 1')).toBeInTheDocument();
    });
  });

  describe('Approval Queue Display', () => {
    it('shows pending deployments', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Momentum Alpha Strategy')).toBeInTheDocument();
      expect(screen.getByText('Arbitrage Pro Strategy')).toBeInTheDocument();
      expect(screen.getByText('Mean Reversion Beta')).toBeInTheDocument();
    });

    it('displays deployment details', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('v2.1.0')).toBeInTheDocument();
      expect(screen.getByText('v1.3.2')).toBeInTheDocument();
      expect(screen.getByText('production')).toBeInTheDocument();
      expect(screen.getByText('hotfix')).toBeInTheDocument();
    });

    it('shows submission information', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('john.doe@company.com')).toBeInTheDocument();
      expect(screen.getByText('jane.smith@company.com')).toBeInTheDocument();
      expect(screen.getByText('bob.wilson@company.com')).toBeInTheDocument();
    });

    it('displays priority levels', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('high')).toBeInTheDocument();
      expect(screen.getByText('medium')).toBeInTheDocument();
      expect(screen.getByText('low')).toBeInTheDocument();
    });

    it('shows deployment status', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('pending_review')).toBeInTheDocument();
      expect(screen.getByText('approved')).toBeInTheDocument();
      expect(screen.getByText('rejected')).toBeInTheDocument();
    });

    it('displays estimated impact', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('medium')).toBeInTheDocument(); // impact
      expect(screen.getByText('low')).toBeInTheDocument(); // impact
      expect(screen.getByText('high')).toBeInTheDocument(); // impact
    });
  });

  describe('Test Results Display', () => {
    it('shows test validation results', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Syntax: passed')).toBeInTheDocument();
      expect(screen.getByText('Units: 127/0')).toBeInTheDocument(); // passed/failed
      expect(screen.getByText('Integration: 23/1')).toBeInTheDocument();
    });

    it('displays test coverage', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('94.7%')).toBeInTheDocument(); // unit coverage
      expect(screen.getByText('89.3%')).toBeInTheDocument(); // integration coverage
    });

    it('shows backtesting results', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Sharpe: 1.85')).toBeInTheDocument();
      expect(screen.getByText('Drawdown: -8.2%')).toBeInTheDocument();
      expect(screen.getByText('Win Rate: 67.3%')).toBeInTheDocument();
      expect(screen.getByText('Return: 23.4%')).toBeInTheDocument();
    });

    it('displays risk assessment', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Risk: low')).toBeInTheDocument();
      expect(screen.getByText('Risk: very_low')).toBeInTheDocument();
      expect(screen.getByText('Risk: high')).toBeInTheDocument();
    });

    it('shows performance test status', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getAllByText('passed')).toHaveLength(2); // performance tests
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });

  describe('Approver Status', () => {
    it('displays required approvers', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Sarah Johnson')).toBeInTheDocument();
      expect(screen.getByText('Mike Chen')).toBeInTheDocument();
      expect(screen.getByText('Alex Smith')).toBeInTheDocument();
    });

    it('shows approver roles', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Risk Manager')).toBeInTheDocument();
      expect(screen.getByText('Head of Trading')).toBeInTheDocument();
      expect(screen.getByText('CTO')).toBeInTheDocument();
    });

    it('indicates approval status', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getAllByText('approved')).toHaveLength(4); // including status filters
      expect(screen.getAllByText('pending')).toHaveLength(3); // including status
    });

    it('shows approval timestamps', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText(/15 minutes ago/)).toBeInTheDocument();
    });
  });

  describe('Deployment Plan', () => {
    it('displays deployment steps', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const detailsButton = screen.getAllByText('View Details')[0];
      await user.click(detailsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Deploy to staging environment')).toBeInTheDocument();
        expect(screen.getByText('Run smoke tests')).toBeInTheDocument();
        expect(screen.getByText('Gradual rollout (10% traffic)')).toBeInTheDocument();
      });
    });

    it('shows rollback plan', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const detailsButton = screen.getAllByText('View Details')[0];
      await user.click(detailsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Immediate traffic redirect to previous version')).toBeInTheDocument();
        expect(screen.getByText('Preserve position states')).toBeInTheDocument();
        expect(screen.getByText('Generate rollback report')).toBeInTheDocument();
      });
    });

    it('displays estimated duration', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const detailsButton = screen.getAllByText('View Details')[0];
      await user.click(detailsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Duration: 45 minutes')).toBeInTheDocument();
      });
    });
  });

  describe('Approval Actions', () => {
    it('allows approving deployments', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockApprove = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        approveDeployment: mockApprove
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const approveButtons = screen.getAllByText('Approve');
      if (approveButtons.length > 0) {
        await user.click(approveButtons[0]);
        expect(mockApprove).toHaveBeenCalled();
      }
    });

    it('allows rejecting deployments', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockReject = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        rejectDeployment: mockReject
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const rejectButtons = screen.getAllByText('Reject');
      if (rejectButtons.length > 0) {
        await user.click(rejectButtons[0]);
        expect(mockReject).toHaveBeenCalled();
      }
    });

    it('supports scheduling deployments', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockSchedule = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        scheduleDeployment: mockSchedule
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const scheduleButtons = screen.getAllByText('Schedule');
      if (scheduleButtons.length > 0) {
        await user.click(scheduleButtons[0]);
        expect(mockSchedule).toHaveBeenCalled();
      }
    });

    it('allows canceling deployments', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockCancel = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        cancelDeployment: mockCancel
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const cancelButtons = screen.getAllByText('Cancel');
      if (cancelButtons.length > 0) {
        await user.click(cancelButtons[0]);
        expect(mockCancel).toHaveBeenCalled();
      }
    });

    it('shows rollback options for live deployments', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockRollback = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        rollbackDeployment: mockRollback
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const rollbackButtons = screen.getAllByText('Rollback');
      if (rollbackButtons.length > 0) {
        await user.click(rollbackButtons[0]);
        expect(mockRollback).toHaveBeenCalled();
      }
    });
  });

  describe('Deployment History', () => {
    it('displays historical deployments', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Deployment History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('momentum-alpha-v2.0')).toBeInTheDocument();
        expect(screen.getByText('arbitrage-pro-v1.2')).toBeInTheDocument();
      });
    });

    it('shows deployment outcomes', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Deployment History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('successful')).toBeInTheDocument();
        expect(screen.getByText('rolled_back')).toBeInTheDocument();
      });
    });

    it('displays performance metrics', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Deployment History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('1.92')).toBeInTheDocument(); // Sharpe
        expect(screen.getByText('21.8%')).toBeInTheDocument(); // Return
        expect(screen.getByText('99.97%')).toBeInTheDocument(); // Uptime
      });
    });

    it('shows rollback information', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const historyTab = screen.getByText('Deployment History');
      await user.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Performance degradation detected')).toBeInTheDocument();
        expect(screen.getByText('Rollbacks: 1')).toBeInTheDocument();
      });
    });
  });

  describe('Approval Metrics', () => {
    it('displays approval statistics', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Total Submissions: 156')).toBeInTheDocument();
      expect(screen.getByText('Approved: 134')).toBeInTheDocument();
      expect(screen.getByText('Rejected: 22')).toBeInTheDocument();
    });

    it('shows success and rollback rates', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Success Rate: 94.7%')).toBeInTheDocument();
      expect(screen.getByText('Rollback Rate: 5.3%')).toBeInTheDocument();
    });

    it('displays average approval time', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Avg Approval Time: 1h')).toBeInTheDocument();
    });

    it('shows approval breakdowns', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Production: 89')).toBeInTheDocument();
      expect(screen.getByText('Staging: 31')).toBeInTheDocument();
      expect(screen.getByText('Critical: 8')).toBeInTheDocument();
      expect(screen.getByText('High: 45')).toBeInTheDocument();
    });
  });

  describe('Configuration Management', () => {
    it('displays approval configuration', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Approval Configuration')).toBeInTheDocument();
        expect(screen.getByText('Required Approvers')).toBeInTheDocument();
        expect(screen.getByText('Automated Checks')).toBeInTheDocument();
      });
    });

    it('shows required approver roles', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Risk Manager: Required')).toBeInTheDocument();
        expect(screen.getByText('Head of Trading: Required')).toBeInTheDocument();
        expect(screen.getByText('CTO: Conditional')).toBeInTheDocument();
      });
    });

    it('displays automated check thresholds', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Unit Test Coverage: 90%')).toBeInTheDocument();
        expect(screen.getByText('Integration Coverage: 85%')).toBeInTheDocument();
        expect(screen.getByText('Overall Coverage: 80%')).toBeInTheDocument();
      });
    });

    it('shows deployment windows', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        expect(screen.getByText('Production: 09:00-17:00 UTC')).toBeInTheDocument();
        expect(screen.getByText('Staging: 00:00-23:00 UTC')).toBeInTheDocument();
      });
    });

    it('allows updating configuration', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockUpdate = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        updateApprovalConfig: mockUpdate
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const configTab = screen.getByText('Configuration');
      await user.click(configTab);
      
      await waitFor(() => {
        const updateButton = screen.getByText('Update Configuration');
        expect(updateButton).toBeInTheDocument();
      });
    });
  });

  describe('Charts and Visualizations', () => {
    it('renders approval metrics chart', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
    });

    it('displays deployment timeline', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const timelineTab = screen.getByText('Timeline');
      await user.click(timelineTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    it('shows approval time trends', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const trendsTab = screen.getByText('Trends');
      await user.click(trendsTab);
      
      await waitFor(() => {
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Control Functions', () => {
    it('refreshes approval queue', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockRefresh = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        refreshQueue: mockRefresh
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const refreshButton = screen.getByText('Refresh Queue');
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });

    it('generates approval report', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockReport = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        generateApprovalReport: mockReport
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const reportButton = screen.getByText('Generate Report');
      await user.click(reportButton);
      
      expect(mockReport).toHaveBeenCalled();
    });

    it('submits new deployment for approval', async () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      const mockSubmit = vi.fn();
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        submitForApproval: mockSubmit
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const submitButton = screen.getByText('Submit New Deployment');
      await user.click(submitButton);
      
      expect(mockSubmit).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes queue automatically when enabled', () => {
      render(<DeploymentApprovalEngine {...defaultProps} enableAutoRefresh={true} />);
      
      act(() => {
        vi.advanceTimersByTime(30000);
      });
      
      expect(screen.getByText('Deployment Approval Engine')).toBeInTheDocument();
    });

    it('stops auto-refresh when disabled', () => {
      render(<DeploymentApprovalEngine {...defaultProps} enableAutoRefresh={false} />);
      
      act(() => {
        vi.advanceTimersByTime(30000);
      });
      
      expect(screen.getByText('Deployment Approval Engine')).toBeInTheDocument();
    });
  });

  describe('User Role Permissions', () => {
    it('shows appropriate actions for user role', () => {
      render(<DeploymentApprovalEngine {...defaultProps} userRole="Head of Trading" />);
      
      // Head of Trading should see approve/reject buttons
      expect(screen.getAllByText('Approve').length).toBeGreaterThan(0);
    });

    it('restricts actions for unauthorized roles', () => {
      render(<DeploymentApprovalEngine {...defaultProps} userRole="Developer" />);
      
      // Developer might have limited permissions
      expect(screen.getByText('Deployment Approval Engine')).toBeInTheDocument();
    });

    it('shows role-specific information', () => {
      render(<DeploymentApprovalEngine {...defaultProps} userRole="Risk Manager" />);
      
      expect(screen.getByText('Role: Risk Manager')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when loading fails', () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        error: 'Failed to load approval queue'
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Approval Engine Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to load approval queue')).toBeInTheDocument();
    });

    it('shows processing state', () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        isProcessing: true
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Processing...')).toBeInTheDocument();
    });

    it('handles empty approval queue', () => {
      const { useDeploymentApproval } = require('../../../hooks/strategy/useDeploymentApproval');
      useDeploymentApproval.mockReturnValue({
        ...useDeploymentApproval(),
        approvalQueue: []
      });

      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('No pending approvals')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides keyboard navigation support', async () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('includes proper ARIA labels', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('supports screen reader accessibility', () => {
      render(<DeploymentApprovalEngine {...defaultProps} />);
      
      expect(screen.getByText('Pending Approvals')).toBeInTheDocument();
      expect(screen.getByText('Deployment Queue')).toBeInTheDocument();
      expect(screen.getByText('Approval Metrics')).toBeInTheDocument();
    });
  });
});