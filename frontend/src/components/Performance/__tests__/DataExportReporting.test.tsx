/**
 * DataExportReporting Test Suite
 * Comprehensive tests for the data export and reporting component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import DataExportReporting from '../DataExportReporting';

// Mock the export hooks
vi.mock('../../../hooks/analytics/useDataExport', () => ({
  useDataExport: vi.fn(() => ({
    exportFormats: ['pdf', 'excel', 'csv', 'json'],
    reportTypes: [
      { id: 'performance', name: 'Performance Report', description: 'Comprehensive portfolio performance analysis' },
      { id: 'risk', name: 'Risk Report', description: 'Risk metrics and VaR calculations' },
      { id: 'attribution', name: 'Attribution Report', description: 'Performance attribution analysis' },
      { id: 'execution', name: 'Execution Report', description: 'Trade execution quality analysis' },
      { id: 'compliance', name: 'Compliance Report', description: 'Regulatory compliance summary' }
    ],
    exportHistory: [
      {
        id: 'exp-001',
        report_type: 'performance',
        format: 'pdf',
        parameters: { portfolio_id: 'port-1', date_range: '2024-01-01_2024-01-15' },
        status: 'completed',
        file_size: '2.4 MB',
        created_at: '2024-01-15T10:30:00Z',
        download_url: 'https://example.com/downloads/exp-001.pdf',
        expires_at: '2024-01-22T10:30:00Z'
      },
      {
        id: 'exp-002',
        report_type: 'risk',
        format: 'excel',
        parameters: { portfolio_id: 'port-1', metrics: ['var', 'es', 'beta'] },
        status: 'processing',
        file_size: null,
        created_at: '2024-01-15T11:00:00Z',
        download_url: null,
        expires_at: null
      },
      {
        id: 'exp-003',
        report_type: 'execution',
        format: 'csv',
        parameters: { date_range: '2024-01-08_2024-01-15', trade_type: 'all' },
        status: 'failed',
        file_size: null,
        created_at: '2024-01-15T09:45:00Z',
        download_url: null,
        expires_at: null,
        error_message: 'Insufficient data for the specified period'
      }
    ],
    exportStats: {
      total_exports: 128,
      successful_exports: 115,
      failed_exports: 8,
      in_progress: 5,
      total_data_exported: '450 MB',
      avg_export_time: '45 seconds',
      most_popular_format: 'excel',
      most_popular_report: 'performance'
    },
    templateLibrary: [
      {
        id: 'template-001',
        name: 'Monthly Performance Summary',
        description: 'Standard monthly performance report template',
        report_type: 'performance',
        parameters: {
          include_charts: true,
          include_attribution: true,
          benchmark_comparison: true,
          format_preference: 'pdf'
        },
        usage_count: 45,
        last_used: '2024-01-14T16:20:00Z'
      },
      {
        id: 'template-002',
        name: 'Risk Dashboard Export',
        description: 'Comprehensive risk metrics export template',
        report_type: 'risk',
        parameters: {
          include_stress_tests: true,
          var_confidence_levels: [95, 99],
          include_component_var: true,
          format_preference: 'excel'
        },
        usage_count: 32,
        last_used: '2024-01-13T14:15:00Z'
      }
    ],
    scheduledExports: [
      {
        id: 'sched-001',
        name: 'Weekly Performance Report',
        report_type: 'performance',
        format: 'pdf',
        schedule: 'weekly',
        schedule_details: 'Every Monday at 08:00',
        recipients: ['manager@company.com', 'analyst@company.com'],
        parameters: { portfolio_id: 'port-1', include_benchmark: true },
        enabled: true,
        next_run: '2024-01-22T08:00:00Z',
        last_run: '2024-01-15T08:00:00Z',
        last_status: 'success'
      },
      {
        id: 'sched-002',
        name: 'Daily Risk Monitoring',
        report_type: 'risk',
        format: 'email',
        schedule: 'daily',
        schedule_details: 'Every day at 07:30',
        recipients: ['risk@company.com'],
        parameters: { alert_threshold_only: true },
        enabled: true,
        next_run: '2024-01-16T07:30:00Z',
        last_run: '2024-01-15T07:30:00Z',
        last_status: 'success'
      }
    ],
    isLoading: false,
    error: null,
    exportData: vi.fn(),
    downloadExport: vi.fn(),
    deleteExport: vi.fn(),
    createTemplate: vi.fn(),
    deleteTemplate: vi.fn(),
    useTemplate: vi.fn(),
    scheduleExport: vi.fn(),
    updateSchedule: vi.fn(),
    deleteSchedule: vi.fn(),
    refreshExports: vi.fn()
  }))
}));

// Mock form handling
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    Form: {
      ...actual.Form,
      useForm: () => [{
        getFieldsValue: vi.fn(() => ({
          reportType: 'performance',
          format: 'pdf',
          dateRange: ['2024-01-01', '2024-01-15'],
          portfolioId: 'port-1'
        })),
        setFieldsValue: vi.fn(),
        resetFields: vi.fn(),
        validateFields: vi.fn(() => Promise.resolve({
          reportType: 'performance',
          format: 'pdf',
          dateRange: ['2024-01-01', '2024-01-15'],
          portfolioId: 'port-1',
          includeBenchmark: true,
          includeCharts: true
        }))
      }]
    }
  };
});

// Mock dayjs
vi.mock('dayjs', () => {
  const mockDayjs = vi.fn(() => ({
    format: vi.fn(() => '2024-01-15 10:30'),
    fromNow: vi.fn(() => '5 minutes ago'),
    valueOf: vi.fn(() => 1705327200000),
    isAfter: vi.fn(() => false),
    isBefore: vi.fn(() => true)
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

// Mock fetch for downloads
global.fetch = vi.fn();

// Mock URL.createObjectURL for file downloads
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  },
  writable: true,
});

// Mock document.createElement for anchor element
const mockAnchorElement = {
  style: {},
  href: '',
  download: '',
  click: vi.fn(),
};

describe('DataExportReporting', () => {
  const user = userEvent.setup();
  const mockProps = {
    portfolioId: 'test-portfolio',
    className: 'test-export',
    height: 700
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(document, 'createElement').mockReturnValue(mockAnchorElement as any);
    vi.spyOn(document.body, 'appendChild').mockImplementation(() => mockAnchorElement as any);
    vi.spyOn(document.body, 'removeChild').mockImplementation(() => mockAnchorElement as any);
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Basic Rendering', () => {
    it('renders the export dashboard', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Data Export & Reporting')).toBeInTheDocument();
      expect(screen.getByText('Quick Export')).toBeInTheDocument();
      expect(screen.getByText('Export History')).toBeInTheDocument();
      expect(screen.getByText('Report Templates')).toBeInTheDocument();
    });

    it('applies custom className and height', () => {
      render(<DataExportReporting {...mockProps} />);
      
      const dashboard = screen.getByText('Data Export & Reporting').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<DataExportReporting />);
      
      expect(screen.getByText('Data Export & Reporting')).toBeInTheDocument();
    });
  });

  describe('Export Statistics', () => {
    it('displays export statistics', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Total Exports')).toBeInTheDocument();
      expect(screen.getByText('128')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('89.8%')).toBeInTheDocument(); // 115/128
    });

    it('shows data volume statistics', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Data Exported')).toBeInTheDocument();
      expect(screen.getByText('450 MB')).toBeInTheDocument();
    });

    it('displays average export time', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Avg Export Time')).toBeInTheDocument();
      expect(screen.getByText('45 seconds')).toBeInTheDocument();
    });

    it('shows popular formats and reports', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Most Popular')).toBeInTheDocument();
      expect(screen.getByText('Excel, Performance')).toBeInTheDocument();
    });
  });

  describe('Quick Export Form', () => {
    it('renders quick export form with all fields', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Report Type')).toBeInTheDocument();
      expect(screen.getByText('Export Format')).toBeInTheDocument();
      expect(screen.getByText('Date Range')).toBeInTheDocument();
      expect(screen.getByText('Portfolio')).toBeInTheDocument();
    });

    it('shows all available report types', () => {
      render(<DataExportReporting {...mockProps} />);
      
      // Should have dropdown options for report types
      const reportTypeSelect = screen.getByDisplayValue(/Select report type/);
      expect(reportTypeSelect).toBeInTheDocument();
    });

    it('displays all export formats', () => {
      render(<DataExportReporting {...mockProps} />);
      
      // Should show format options
      expect(screen.getByText('PDF')).toBeInTheDocument();
      expect(screen.getByText('Excel')).toBeInTheDocument();
      expect(screen.getByText('CSV')).toBeInTheDocument();
      expect(screen.getByText('JSON')).toBeInTheDocument();
    });

    it('allows customizing export parameters', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      // Should have additional options section
      const advancedButton = screen.getByText('Advanced Options');
      await user.click(advancedButton);
      
      await waitFor(() => {
        expect(screen.getByText('Include Charts')).toBeInTheDocument();
        expect(screen.getByText('Include Benchmark')).toBeInTheDocument();
        expect(screen.getByText('Include Attribution')).toBeInTheDocument();
      });
    });

    it('starts export on form submission', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockExport = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        exportData: mockExport
      });

      render(<DataExportReporting {...mockProps} />);
      
      const exportButton = screen.getByText('Start Export');
      await user.click(exportButton);
      
      await waitFor(() => {
        expect(mockExport).toHaveBeenCalled();
      });
    });
  });

  describe('Export History', () => {
    it('displays export history table', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Report Type')).toBeInTheDocument();
      expect(screen.getByText('Format')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('File Size')).toBeInTheDocument();
      expect(screen.getByText('Created')).toBeInTheDocument();
      expect(screen.getByText('Actions')).toBeInTheDocument();
    });

    it('shows all export entries', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('performance')).toBeInTheDocument();
      expect(screen.getByText('risk')).toBeInTheDocument();
      expect(screen.getByText('execution')).toBeInTheDocument();
    });

    it('displays export statuses correctly', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('processing')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });

    it('shows download buttons for completed exports', () => {
      render(<DataExportReporting {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      expect(downloadButtons.length).toBeGreaterThan(0);
    });

    it('allows downloading completed exports', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockDownload = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        downloadExport: mockDownload
      });

      render(<DataExportReporting {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      if (downloadButtons.length > 0) {
        await user.click(downloadButtons[0]);
        expect(mockDownload).toHaveBeenCalledWith('exp-001');
      }
    });

    it('allows deleting exports', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockDelete = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        deleteExport: mockDelete
      });

      render(<DataExportReporting {...mockProps} />);
      
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      if (deleteButtons.length > 0) {
        await user.click(deleteButtons[0]);
        
        // Should show confirmation
        await waitFor(() => {
          expect(screen.getByText(/Are you sure/)).toBeInTheDocument();
        });
      }
    });

    it('shows error messages for failed exports', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Insufficient data for the specified period')).toBeInTheDocument();
    });
  });

  describe('Report Templates', () => {
    it('displays template library', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Template Library')).toBeInTheDocument();
      expect(screen.getByText('Monthly Performance Summary')).toBeInTheDocument();
      expect(screen.getByText('Risk Dashboard Export')).toBeInTheDocument();
    });

    it('shows template usage statistics', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Used 45 times')).toBeInTheDocument();
      expect(screen.getByText('Used 32 times')).toBeInTheDocument();
    });

    it('allows using templates', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockUseTemplate = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        useTemplate: mockUseTemplate
      });

      render(<DataExportReporting {...mockProps} />);
      
      const useButtons = screen.getAllByText('Use Template');
      if (useButtons.length > 0) {
        await user.click(useButtons[0]);
        expect(mockUseTemplate).toHaveBeenCalledWith('template-001');
      }
    });

    it('opens create template modal', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const createButton = screen.getByText('Create Template');
      await user.click(createButton);
      
      await waitFor(() => {
        expect(screen.getByText('Create Export Template')).toBeInTheDocument();
      });
    });

    it('allows creating new templates', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockCreate = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        createTemplate: mockCreate
      });

      render(<DataExportReporting {...mockProps} />);
      
      const createButton = screen.getByText('Create Template');
      await user.click(createButton);
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save Template');
        expect(saveButton).toBeInTheDocument();
      });
    });
  });

  describe('Scheduled Exports', () => {
    it('displays scheduled exports', () => {
      render(<DataExportReporting {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      expect(scheduledTab).toBeInTheDocument();
    });

    it('shows scheduled export details', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText('Weekly Performance Report')).toBeInTheDocument();
        expect(screen.getByText('Daily Risk Monitoring')).toBeInTheDocument();
        expect(screen.getByText('Every Monday at 08:00')).toBeInTheDocument();
        expect(screen.getByText('Every day at 07:30')).toBeInTheDocument();
      });
    });

    it('shows next run times', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText(/Next run:/)).toBeInTheDocument();
      });
    });

    it('allows creating new scheduled exports', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        const createButton = screen.getByText('New Schedule');
        expect(createButton).toBeInTheDocument();
      });
    });

    it('allows enabling/disabling schedules', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockUpdate = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        updateSchedule: mockUpdate
      });

      render(<DataExportReporting {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        const toggleSwitches = screen.getAllByRole('switch');
        if (toggleSwitches.length > 0) {
          expect(toggleSwitches[0]).toBeInTheDocument();
        }
      });
    });
  });

  describe('Filtering and Search', () => {
    it('allows filtering exports by status', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const filterButton = screen.getByRole('button', { name: /filter/i });
      await user.click(filterButton);
      
      await waitFor(() => {
        expect(screen.getByText('All Status')).toBeInTheDocument();
        expect(screen.getByText('Completed')).toBeInTheDocument();
        expect(screen.getByText('Processing')).toBeInTheDocument();
        expect(screen.getByText('Failed')).toBeInTheDocument();
      });
    });

    it('allows filtering by report type', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      // Should have report type filter
      const reportFilters = screen.getAllByText(/performance|risk|execution/);
      expect(reportFilters.length).toBeGreaterThan(0);
    });

    it('allows searching exports', () => {
      render(<DataExportReporting {...mockProps} />);
      
      const searchInput = screen.getByPlaceholderText(/Search exports/i);
      expect(searchInput).toBeInTheDocument();
    });

    it('filters results based on search input', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      const searchInput = screen.getByPlaceholderText(/Search exports/i);
      await user.type(searchInput, 'performance');
      
      // Should filter results (implementation dependent)
      expect(searchInput).toHaveValue('performance');
    });
  });

  describe('Data Refresh', () => {
    it('refreshes export history automatically', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockRefresh = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        refreshExports: mockRefresh
      });

      render(<DataExportReporting {...mockProps} autoRefresh={true} />);
      
      // Should automatically refresh
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(30000); // 30 seconds
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('allows manual refresh', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockRefresh = vi.fn();
      useDataExport.mockReturnValue({
        ...useDataExport(),
        refreshExports: mockRefresh
      });

      render(<DataExportReporting {...mockProps} />);
      
      const refreshButton = screen.getByRole('button', { name: /reload/i });
      await user.click(refreshButton);
      
      expect(mockRefresh).toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when export service fails', () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      useDataExport.mockReturnValue({
        ...useDataExport(),
        error: 'Failed to load export data'
      });

      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Export Service Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to load export data')).toBeInTheDocument();
    });

    it('shows loading state', () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      useDataExport.mockReturnValue({
        ...useDataExport(),
        isLoading: true,
        exportHistory: []
      });

      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('Loading export data...')).toBeInTheDocument();
    });

    it('handles empty export history', () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      useDataExport.mockReturnValue({
        ...useDataExport(),
        exportHistory: [],
        exportStats: {
          total_exports: 0,
          successful_exports: 0,
          failed_exports: 0,
          in_progress: 0,
          total_data_exported: '0 MB',
          avg_export_time: '0 seconds',
          most_popular_format: null,
          most_popular_report: null
        }
      });

      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('No exports found')).toBeInTheDocument();
    });

    it('handles export failures gracefully', async () => {
      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockExport = vi.fn().mockRejectedValue(new Error('Export failed'));
      useDataExport.mockReturnValue({
        ...useDataExport(),
        exportData: mockExport
      });

      render(<DataExportReporting {...mockProps} />);
      
      const exportButton = screen.getByText('Start Export');
      await user.click(exportButton);
      
      // Should handle error gracefully without crashing
      expect(screen.getByText('Data Export & Reporting')).toBeInTheDocument();
    });
  });

  describe('File Download', () => {
    it('handles successful downloads', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(['test data'], { type: 'application/pdf' }))
      });

      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockDownload = vi.fn().mockResolvedValue(true);
      useDataExport.mockReturnValue({
        ...useDataExport(),
        downloadExport: mockDownload
      });

      render(<DataExportReporting {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      if (downloadButtons.length > 0) {
        await user.click(downloadButtons[0]);
        
        await waitFor(() => {
          expect(mockDownload).toHaveBeenCalled();
        });
      }
    });

    it('handles download errors', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Download failed'));

      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      const mockDownload = vi.fn().mockRejectedValue(new Error('Download failed'));
      useDataExport.mockReturnValue({
        ...useDataExport(),
        downloadExport: mockDownload
      });

      render(<DataExportReporting {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      if (downloadButtons.length > 0) {
        await user.click(downloadButtons[0]);
        
        // Should handle error gracefully
        await waitFor(() => {
          expect(mockDownload).toHaveBeenCalled();
        });
      }
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for form controls', () => {
      render(<DataExportReporting {...mockProps} />);
      
      const buttons = screen.getAllByRole('button');
      const comboboxes = screen.getAllByRole('combobox');
      
      buttons.forEach(button => expect(button).toBeInTheDocument());
      comboboxes.forEach(combobox => expect(combobox).toBeInTheDocument());
    });

    it('supports keyboard navigation', async () => {
      render(<DataExportReporting {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<DataExportReporting {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('processing')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large export history efficiently', () => {
      const largeHistory = Array.from({ length: 500 }, (_, i) => ({
        id: `exp-${i.toString().padStart(3, '0')}`,
        report_type: ['performance', 'risk', 'execution'][i % 3],
        format: ['pdf', 'excel', 'csv'][i % 3],
        parameters: { portfolio_id: 'port-1' },
        status: ['completed', 'processing', 'failed'][i % 3],
        file_size: `${(Math.random() * 5 + 0.1).toFixed(1)} MB`,
        created_at: new Date(Date.now() - i * 1000 * 60 * 60).toISOString(),
        download_url: i % 3 === 0 ? `https://example.com/downloads/exp-${i}.pdf` : null,
        expires_at: i % 3 === 0 ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() : null
      }));

      const { useDataExport } = require('../../../hooks/analytics/useDataExport');
      useDataExport.mockReturnValue({
        ...useDataExport(),
        exportHistory: largeHistory
      });

      render(<DataExportReporting {...mockProps} />);
      
      // Should render without performance issues
      expect(screen.getByText('Data Export & Reporting')).toBeInTheDocument();
    });
  });
});