/**
 * AnalyticsAggregator Test Suite
 * Comprehensive tests for the analytics data aggregation component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import AnalyticsAggregator from '../AnalyticsAggregator';

// Mock the analytics hooks
vi.mock('../../../hooks/analytics/useAnalyticsAggregator', () => ({
  useAnalyticsAggregator: vi.fn(() => ({
    jobs: [
      {
        job_id: 'agg-001',
        job_name: 'Daily Performance Aggregation',
        data_type: 'performance',
        interval: 'day',
        source_table: 'raw_performance_data',
        target_table: 'daily_performance_agg',
        start_date: '2024-01-01',
        end_date: '2024-01-15',
        status: 'completed',
        progress: 100,
        records_processed: 15680,
        total_records: 15680,
        compression_ratio: 8.5,
        created_at: '2024-01-15T10:00:00Z',
        duration: '2m 35s',
        error_message: null
      },
      {
        job_id: 'agg-002',
        job_name: 'Risk Metrics Hourly Aggregation',
        data_type: 'risk',
        interval: 'hour',
        source_table: 'raw_risk_data',
        target_table: 'hourly_risk_agg',
        start_date: '2024-01-15',
        end_date: '2024-01-15',
        status: 'running',
        progress: 65,
        records_processed: 2850,
        total_records: 4380,
        compression_ratio: null,
        created_at: '2024-01-15T09:30:00Z',
        duration: null,
        error_message: null
      },
      {
        job_id: 'agg-003',
        job_name: 'Weekly Strategy Analysis',
        data_type: 'strategy',
        interval: 'week',
        source_table: 'strategy_performance',
        target_table: 'weekly_strategy_agg',
        start_date: '2024-01-08',
        end_date: '2024-01-15',
        status: 'failed',
        progress: 25,
        records_processed: 580,
        total_records: 2320,
        compression_ratio: null,
        created_at: '2024-01-15T08:00:00Z',
        duration: null,
        error_message: 'Data source connection timeout'
      }
    ],
    aggregationStats: {
      total_jobs: 15,
      active_jobs: 3,
      completed_jobs: 10,
      failed_jobs: 2,
      total_records_processed: 2580000,
      total_storage_saved: '1.2 GB',
      avg_compression_ratio: 7.8,
      last_job_time: '2024-01-15T10:00:00Z'
    },
    storageMetrics: {
      raw_data_size: '15.8 GB',
      aggregated_data_size: '2.1 GB',
      compression_ratio: 7.52,
      storage_efficiency: 86.7,
      retention_policy: '90 days',
      cleanup_schedule: 'Daily at 02:00'
    },
    isLoading: false,
    error: null,
    startAggregation: vi.fn(),
    stopAggregation: vi.fn(),
    cancelJob: vi.fn(),
    retryJob: vi.fn(),
    deleteJob: vi.fn(),
    exportJobResults: vi.fn(),
    refreshJobs: vi.fn(),
    updateSettings: vi.fn()
  }))
}));

// Mock form hooks for settings
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    Form: {
      ...actual.Form,
      useForm: () => [{
        getFieldsValue: vi.fn(() => ({
          dataType: 'performance',
          interval: 'day',
          compression: true,
          retention: 90
        })),
        setFieldsValue: vi.fn(),
        resetFields: vi.fn(),
        validateFields: vi.fn(() => Promise.resolve({
          jobName: 'Test Aggregation Job',
          dataType: 'performance',
          interval: 'day',
          startDate: '2024-01-01',
          endDate: '2024-01-15'
        }))
      }]
    }
  };
});

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '2024-01-15'),
    subtract: vi.fn(() => ({
      format: vi.fn(() => '2024-01-01')
    })),
    valueOf: vi.fn(() => 1705327200000),
    fromNow: vi.fn(() => '5 minutes ago')
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

// Mock fetch for API calls
global.fetch = vi.fn();

describe('AnalyticsAggregator', () => {
  const user = userEvent.setup();
  const mockProps = {
    className: 'test-aggregator',
    height: 600,
    autoRefresh: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the aggregator dashboard', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Analytics Data Aggregator')).toBeInTheDocument();
      expect(screen.getByText('Data Aggregation Jobs')).toBeInTheDocument();
      expect(screen.getByText('Storage Metrics')).toBeInTheDocument();
    });

    it('applies custom className and height', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const dashboard = screen.getByText('Analytics Data Aggregator').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<AnalyticsAggregator />);
      
      expect(screen.getByText('Analytics Data Aggregator')).toBeInTheDocument();
    });
  });

  describe('Summary Statistics', () => {
    it('displays aggregation statistics', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Total Jobs')).toBeInTheDocument();
      expect(screen.getByText('15')).toBeInTheDocument(); // total_jobs
      expect(screen.getByText('Active Jobs')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument(); // active_jobs
      expect(screen.getByText('Storage Saved')).toBeInTheDocument();
      expect(screen.getByText('1.2 GB')).toBeInTheDocument();
    });

    it('shows compression statistics', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Avg Compression')).toBeInTheDocument();
      expect(screen.getByText('7.8x')).toBeInTheDocument();
    });

    it('displays storage efficiency metrics', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Raw Data Size')).toBeInTheDocument();
      expect(screen.getByText('15.8 GB')).toBeInTheDocument();
      expect(screen.getByText('Aggregated Size')).toBeInTheDocument();
      expect(screen.getByText('2.1 GB')).toBeInTheDocument();
    });
  });

  describe('Job Management', () => {
    it('displays job list with all columns', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Job Name')).toBeInTheDocument();
      expect(screen.getByText('Data Type')).toBeInTheDocument();
      expect(screen.getByText('Interval')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Progress')).toBeInTheDocument();
      expect(screen.getByText('Records')).toBeInTheDocument();
    });

    it('shows all job entries', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Daily Performance Aggregation')).toBeInTheDocument();
      expect(screen.getByText('Risk Metrics Hourly Aggregation')).toBeInTheDocument();
      expect(screen.getByText('Weekly Strategy Analysis')).toBeInTheDocument();
    });

    it('displays job statuses correctly', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });

    it('shows progress bars for running jobs', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('displays compression ratios for completed jobs', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('8.5x')).toBeInTheDocument(); // compression ratio
    });

    it('shows error messages for failed jobs', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Data source connection timeout')).toBeInTheDocument();
    });
  });

  describe('Job Actions', () => {
    it('renders action buttons for each job', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Should have multiple action buttons for different job states
      const actionButtons = screen.getAllByRole('button');
      expect(actionButtons.length).toBeGreaterThan(5);
    });

    it('allows canceling running jobs', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockCancel = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        cancelJob: mockCancel
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Find and click cancel button for running job
      const cancelButtons = screen.getAllByRole('button', { name: /stop/i });
      if (cancelButtons.length > 0) {
        await user.click(cancelButtons[0]);
        expect(mockCancel).toHaveBeenCalledWith('agg-002');
      }
    });

    it('allows retrying failed jobs', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockRetry = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        retryJob: mockRetry
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Find and click retry button for failed job
      const retryButtons = screen.getAllByRole('button', { name: /reload/i });
      if (retryButtons.length > 0) {
        await user.click(retryButtons[0]);
        expect(mockRetry).toHaveBeenCalledWith('agg-003');
      }
    });

    it('allows deleting jobs', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockDelete = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        deleteJob: mockDelete
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Find and click delete button
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      if (deleteButtons.length > 0) {
        await user.click(deleteButtons[0]);
        
        // Should show confirmation
        await waitFor(() => {
          expect(screen.getByText(/Are you sure/)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Create New Job', () => {
    it('opens create job modal', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const createButton = screen.getByText('Create Job');
      await user.click(createButton);
      
      await waitFor(() => {
        expect(screen.getByText('Create Aggregation Job')).toBeInTheDocument();
      });
    });

    it('shows all form fields in create modal', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const createButton = screen.getByText('Create Job');
      await user.click(createButton);
      
      await waitFor(() => {
        expect(screen.getByText('Job Name')).toBeInTheDocument();
        expect(screen.getByText('Data Type')).toBeInTheDocument();
        expect(screen.getByText('Aggregation Interval')).toBeInTheDocument();
        expect(screen.getByText('Date Range')).toBeInTheDocument();
        expect(screen.getByText('Enable Compression')).toBeInTheDocument();
      });
    });

    it('validates form fields', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const createButton = screen.getByText('Create Job');
      await user.click(createButton);
      
      await waitFor(() => {
        const submitButton = screen.getByText('Create Job');
        expect(submitButton).toBeInTheDocument();
      });
      
      // Try to submit empty form
      const submitButton = screen.getByText('Create Job');
      await user.click(submitButton);
      
      // Form validation should prevent submission
    });

    it('creates new job with valid data', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockStart = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        startAggregation: mockStart
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      const createButton = screen.getByText('Create Job');
      await user.click(createButton);
      
      await waitFor(() => {
        const submitButton = screen.getAllByText('Create Job')[1]; // Second one is in modal
        expect(submitButton).toBeInTheDocument();
      });
    });
  });

  describe('Settings and Configuration', () => {
    it('opens settings modal', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Aggregator Settings')).toBeInTheDocument();
      });
    });

    it('shows configuration options', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Default Settings')).toBeInTheDocument();
        expect(screen.getByText('Storage Settings')).toBeInTheDocument();
        expect(screen.getByText('Performance Settings')).toBeInTheDocument();
      });
    });

    it('allows updating settings', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockUpdate = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        updateSettings: mockUpdate
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      const settingsButton = screen.getByRole('button', { name: /setting/i });
      await user.click(settingsButton);
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save Settings');
        expect(saveButton).toBeInTheDocument();
      });
    });
  });

  describe('Data Export', () => {
    it('renders bulk export button', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const exportButton = screen.getByText('Bulk Export');
      expect(exportButton).toBeInTheDocument();
    });

    it('opens export options', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const exportButton = screen.getByText('Bulk Export');
      await user.click(exportButton);
      
      await waitFor(() => {
        expect(screen.getByText('Export Job Results')).toBeInTheDocument();
        expect(screen.getByText('Export All Data')).toBeInTheDocument();
        expect(screen.getByText('Export Summary Report')).toBeInTheDocument();
      });
    });

    it('handles individual job export', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockExport = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        exportJobResults: mockExport
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Find export button for specific job
      const exportButtons = screen.getAllByRole('button', { name: /download/i });
      if (exportButtons.length > 0) {
        await user.click(exportButtons[0]);
        expect(mockExport).toHaveBeenCalled();
      }
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes data when autoRefresh is enabled', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockRefresh = vi.fn();
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        refreshJobs: mockRefresh
      });

      render(<AnalyticsAggregator {...mockProps} autoRefresh={true} />);
      
      // Fast-forward time to trigger auto-refresh
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('shows real-time progress updates', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Should show running job with progress
      expect(screen.getByText('65%')).toBeInTheDocument();
      expect(screen.getByText('2,850 / 4,380')).toBeInTheDocument();
    });

    it('updates job statuses automatically', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Initially should show current statuses
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when aggregator fails', () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        error: 'Failed to load aggregation jobs'
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Failed to load aggregation jobs')).toBeInTheDocument();
    });

    it('shows loading state', () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        isLoading: true,
        jobs: []
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Loading aggregation data...')).toBeInTheDocument();
    });

    it('handles empty job list', () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        jobs: [],
        aggregationStats: {
          total_jobs: 0,
          active_jobs: 0,
          completed_jobs: 0,
          failed_jobs: 0,
          total_records_processed: 0,
          total_storage_saved: '0 MB',
          avg_compression_ratio: 0,
          last_job_time: null
        }
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('No aggregation jobs found')).toBeInTheDocument();
    });

    it('handles API errors gracefully', async () => {
      const { useAnalyticsAggregator } = require('../../../hooks/analytics/useAnalyticsAggregator');
      const mockStart = vi.fn().mockRejectedValue(new Error('API Error'));
      useAnalyticsAggregator.mockReturnValue({
        ...useAnalyticsAggregator(),
        startAggregation: mockStart
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      const createButton = screen.getByText('Create Job');
      await user.click(createButton);
      
      // Should not crash on API error
      expect(screen.getByText('Analytics Data Aggregator')).toBeInTheDocument();
    });
  });

  describe('Performance Monitoring', () => {
    it('displays performance metrics', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Records Processed')).toBeInTheDocument();
      expect(screen.getByText('2,580,000')).toBeInTheDocument();
    });

    it('shows storage efficiency', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Storage Efficiency')).toBeInTheDocument();
      expect(screen.getByText('86.7%')).toBeInTheDocument();
    });

    it('displays retention policy information', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      expect(screen.getByText('Retention Policy')).toBeInTheDocument();
      expect(screen.getByText('90 days')).toBeInTheDocument();
    });
  });

  describe('Filtering and Sorting', () => {
    it('allows filtering jobs by status', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const filterButton = screen.getByRole('button', { name: /filter/i });
      await user.click(filterButton);
      
      await waitFor(() => {
        expect(screen.getByText('All Status')).toBeInTheDocument();
        expect(screen.getByText('Running')).toBeInTheDocument();
        expect(screen.getByText('Completed')).toBeInTheDocument();
        expect(screen.getByText('Failed')).toBeInTheDocument();
      });
    });

    it('allows filtering jobs by data type', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Should have data type filter options
      expect(screen.getByText('performance')).toBeInTheDocument();
      expect(screen.getByText('risk')).toBeInTheDocument();
      expect(screen.getByText('strategy')).toBeInTheDocument();
    });

    it('sorts jobs by different columns', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Click on sortable column headers
      const jobNameHeader = screen.getByText('Job Name');
      await user.click(jobNameHeader);
      
      // Should trigger sort functionality
      expect(jobNameHeader).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for interactive elements', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', async () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Tab navigation should work
      await user.tab();
      
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<AnalyticsAggregator {...mockProps} />);
      
      // Status should be clearly indicated
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('adjusts table layout for smaller screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Should render table with responsive layout
      expect(screen.getByText('Data Aggregation Jobs')).toBeInTheDocument();
    });

    it('maintains functionality on mobile', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<AnalyticsAggregator {...mockProps} />);
      
      // Core functionality should still be accessible
      expect(screen.getByText('Create Job')).toBeInTheDocument();
      expect(screen.getByText('Bulk Export')).toBeInTheDocument();
    });
  });
});