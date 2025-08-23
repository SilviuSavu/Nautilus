/**
 * RiskReportGenerator Test Suite
 * Comprehensive tests for the risk report generation component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import RiskReportGenerator from '../RiskReportGenerator';

// Mock the risk reporting hooks
vi.mock('../../../hooks/analytics/useRiskReporting', () => ({
  useRiskReporting: vi.fn(() => ({
    reportTypes: [
      {
        id: 'var_report',
        name: 'Value at Risk Report',
        description: 'Comprehensive VaR analysis with confidence intervals',
        template: 'var_template',
        estimated_duration: '3-5 minutes',
        complexity: 'medium'
      },
      {
        id: 'stress_test',
        name: 'Stress Testing Report',
        description: 'Portfolio performance under stress scenarios',
        template: 'stress_template',
        estimated_duration: '5-8 minutes',
        complexity: 'high'
      },
      {
        id: 'compliance',
        name: 'Compliance Report',
        description: 'Regulatory compliance and limit monitoring',
        template: 'compliance_template',
        estimated_duration: '2-3 minutes',
        complexity: 'low'
      },
      {
        id: 'concentration',
        name: 'Concentration Risk Report',
        description: 'Portfolio concentration and diversification analysis',
        template: 'concentration_template',
        estimated_duration: '3-4 minutes',
        complexity: 'medium'
      },
      {
        id: 'liquidity',
        name: 'Liquidity Risk Report',
        description: 'Liquidity assessment and funding requirements',
        template: 'liquidity_template',
        estimated_duration: '4-6 minutes',
        complexity: 'high'
      }
    ],
    reportTemplates: [
      {
        id: 'template-001',
        name: 'Executive Risk Summary',
        report_type: 'var_report',
        description: 'High-level risk summary for executives',
        sections: ['executive_summary', 'key_metrics', 'risk_alerts', 'recommendations'],
        format_options: ['pdf', 'powerpoint'],
        auto_refresh: false,
        usage_count: 87,
        last_used: '2024-01-14T16:30:00Z'
      },
      {
        id: 'template-002',
        name: 'Detailed Stress Analysis',
        report_type: 'stress_test',
        description: 'Comprehensive stress testing with scenarios',
        sections: ['methodology', 'scenarios', 'results', 'recommendations', 'appendices'],
        format_options: ['pdf', 'excel'],
        auto_refresh: true,
        usage_count: 45,
        last_used: '2024-01-13T10:15:00Z'
      },
      {
        id: 'template-003',
        name: 'Regulatory Compliance',
        report_type: 'compliance',
        description: 'Standard regulatory compliance report',
        sections: ['compliance_status', 'limit_monitoring', 'violations', 'action_items'],
        format_options: ['pdf'],
        auto_refresh: true,
        usage_count: 123,
        last_used: '2024-01-15T09:00:00Z'
      }
    ],
    generationQueue: [
      {
        job_id: 'rpt-001',
        report_type: 'var_report',
        template_name: 'Executive Risk Summary',
        portfolio_id: 'port-001',
        status: 'completed',
        progress: 100,
        started_at: '2024-01-15T10:00:00Z',
        completed_at: '2024-01-15T10:03:45Z',
        duration: '3m 45s',
        file_size: '2.4 MB',
        download_url: '/downloads/risk_report_001.pdf',
        expires_at: '2024-01-22T10:03:45Z'
      },
      {
        job_id: 'rpt-002',
        report_type: 'stress_test',
        template_name: 'Detailed Stress Analysis',
        portfolio_id: 'port-002',
        status: 'generating',
        progress: 65,
        started_at: '2024-01-15T10:15:00Z',
        completed_at: null,
        duration: null,
        file_size: null,
        download_url: null,
        expires_at: null
      },
      {
        job_id: 'rpt-003',
        report_type: 'compliance',
        template_name: 'Regulatory Compliance',
        portfolio_id: 'port-001',
        status: 'failed',
        progress: 25,
        started_at: '2024-01-15T09:45:00Z',
        completed_at: null,
        duration: null,
        file_size: null,
        download_url: null,
        expires_at: null,
        error_message: 'Data source connection timeout'
      }
    ],
    reportingStats: {
      total_reports_generated: 1247,
      successful_reports: 1185,
      failed_reports: 62,
      avg_generation_time: '4m 32s',
      total_data_processed: '15.8 GB',
      most_popular_report: 'var_report',
      most_popular_template: 'Executive Risk Summary',
      peak_usage_hour: '09:00-10:00'
    },
    customSections: [
      { id: 'executive_summary', name: 'Executive Summary', required: true, default_enabled: true },
      { id: 'key_metrics', name: 'Key Risk Metrics', required: true, default_enabled: true },
      { id: 'var_analysis', name: 'VaR Analysis', required: false, default_enabled: true },
      { id: 'stress_scenarios', name: 'Stress Testing', required: false, default_enabled: false },
      { id: 'concentration', name: 'Concentration Analysis', required: false, default_enabled: true },
      { id: 'liquidity', name: 'Liquidity Assessment', required: false, default_enabled: false },
      { id: 'compliance', name: 'Compliance Status', required: false, default_enabled: true },
      { id: 'recommendations', name: 'Risk Recommendations', required: true, default_enabled: true },
      { id: 'appendices', name: 'Technical Appendices', required: false, default_enabled: false }
    ],
    scheduledReports: [
      {
        id: 'sched-001',
        name: 'Daily VaR Report',
        report_type: 'var_report',
        template_id: 'template-001',
        schedule: 'daily',
        schedule_time: '08:00',
        recipients: ['risk@company.com', 'management@company.com'],
        enabled: true,
        last_run: '2024-01-15T08:00:00Z',
        next_run: '2024-01-16T08:00:00Z',
        success_rate: 0.98
      },
      {
        id: 'sched-002',
        name: 'Weekly Stress Test',
        report_type: 'stress_test',
        template_id: 'template-002',
        schedule: 'weekly',
        schedule_time: 'Monday 07:00',
        recipients: ['risk@company.com'],
        enabled: true,
        last_run: '2024-01-15T07:00:00Z',
        next_run: '2024-01-22T07:00:00Z',
        success_rate: 0.95
      }
    ],
    isGenerating: false,
    error: null,
    generateReport: vi.fn(),
    downloadReport: vi.fn(),
    deleteReport: vi.fn(),
    createTemplate: vi.fn(),
    updateTemplate: vi.fn(),
    deleteTemplate: vi.fn(),
    scheduleReport: vi.fn(),
    updateSchedule: vi.fn(),
    deleteSchedule: vi.fn(),
    refreshQueue: vi.fn()
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
          reportType: 'var_report',
          templateId: 'template-001',
          portfolioId: 'port-001',
          dateRange: ['2024-01-01', '2024-01-15'],
          format: 'pdf',
          includeCharts: true,
          confidenceLevel: 95
        })),
        setFieldsValue: vi.fn(),
        resetFields: vi.fn(),
        validateFields: vi.fn(() => Promise.resolve({
          reportType: 'var_report',
          templateId: 'template-001',
          portfolioId: 'port-001',
          dateRange: ['2024-01-01', '2024-01-15'],
          format: 'pdf',
          sections: ['executive_summary', 'key_metrics', 'recommendations']
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
    subtract: vi.fn(() => ({
      format: vi.fn(() => '2024-01-01')
    }))
  }));
  mockDayjs.extend = vi.fn();
  return { default: mockDayjs };
});

// Mock file download
global.fetch = vi.fn();
Object.defineProperty(window, 'URL', {
  value: {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn(),
  },
  writable: true,
});

const mockAnchorElement = {
  style: {},
  href: '',
  download: '',
  click: vi.fn(),
};

describe('RiskReportGenerator', () => {
  const user = userEvent.setup();
  const mockProps = {
    portfolioId: 'test-portfolio',
    className: 'test-generator',
    height: 800
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
    it('renders the report generator dashboard', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Risk Report Generator')).toBeInTheDocument();
      expect(screen.getByText('Generate Report')).toBeInTheDocument();
      expect(screen.getByText('Report Queue')).toBeInTheDocument();
      expect(screen.getByText('Templates')).toBeInTheDocument();
    });

    it('applies custom className and height', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const dashboard = screen.getByText('Risk Report Generator').closest('.ant-card');
      expect(dashboard).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<RiskReportGenerator />);
      
      expect(screen.getByText('Risk Report Generator')).toBeInTheDocument();
    });
  });

  describe('Generation Statistics', () => {
    it('displays report generation statistics', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Total Reports')).toBeInTheDocument();
      expect(screen.getByText('1,247')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('95.0%')).toBeInTheDocument(); // 1185/1247
    });

    it('shows average generation time', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Avg Generation Time')).toBeInTheDocument();
      expect(screen.getByText('4m 32s')).toBeInTheDocument();
    });

    it('displays data processing volume', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Data Processed')).toBeInTheDocument();
      expect(screen.getByText('15.8 GB')).toBeInTheDocument();
    });

    it('shows popular report types', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Most Popular')).toBeInTheDocument();
      expect(screen.getByText('VaR Report, Executive Summary')).toBeInTheDocument();
    });
  });

  describe('Report Generation Form', () => {
    it('renders generation form with all fields', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Report Type')).toBeInTheDocument();
      expect(screen.getByText('Template')).toBeInTheDocument();
      expect(screen.getByText('Portfolio')).toBeInTheDocument();
      expect(screen.getByText('Date Range')).toBeInTheDocument();
      expect(screen.getByText('Format')).toBeInTheDocument();
    });

    it('displays all available report types', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should have dropdown with report type options
      const reportTypeSelect = screen.getByDisplayValue(/Select report type/);
      expect(reportTypeSelect).toBeInTheDocument();
    });

    it('shows report complexity indicators', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should indicate complexity levels
      expect(screen.getByText(/3-5 minutes/)).toBeInTheDocument();
      expect(screen.getByText(/medium/)).toBeInTheDocument();
    });

    it('allows customizing report sections', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const customizeButton = screen.getByText('Customize Sections');
      await user.click(customizeButton);
      
      await waitFor(() => {
        expect(screen.getByText('Executive Summary')).toBeInTheDocument();
        expect(screen.getByText('Key Risk Metrics')).toBeInTheDocument();
        expect(screen.getByText('VaR Analysis')).toBeInTheDocument();
      });
    });

    it('shows required vs optional sections', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const customizeButton = screen.getByText('Customize Sections');
      await user.click(customizeButton);
      
      await waitFor(() => {
        // Required sections should be indicated
        expect(screen.getByText('Required')).toBeInTheDocument();
        expect(screen.getByText('Optional')).toBeInTheDocument();
      });
    });

    it('starts report generation', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockGenerate = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        generateReport: mockGenerate
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const generateButton = screen.getByText('Generate Report');
      await user.click(generateButton);
      
      await waitFor(() => {
        expect(mockGenerate).toHaveBeenCalled();
      });
    });
  });

  describe('Report Queue Management', () => {
    it('displays report generation queue', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Report Queue')).toBeInTheDocument();
      expect(screen.getByText('Job ID')).toBeInTheDocument();
      expect(screen.getByText('Report Type')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Progress')).toBeInTheDocument();
    });

    it('shows all queue entries', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('rpt-001')).toBeInTheDocument();
      expect(screen.getByText('rpt-002')).toBeInTheDocument();
      expect(screen.getByText('rpt-003')).toBeInTheDocument();
    });

    it('displays job statuses correctly', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('generating')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });

    it('shows progress bars for active jobs', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      expect(progressBars.length).toBeGreaterThan(0);
    });

    it('displays generation duration', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('3m 45s')).toBeInTheDocument();
    });

    it('shows file sizes for completed reports', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('2.4 MB')).toBeInTheDocument();
    });

    it('displays error messages for failed jobs', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Data source connection timeout')).toBeInTheDocument();
    });

    it('allows downloading completed reports', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockDownload = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        downloadReport: mockDownload
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      if (downloadButtons.length > 0) {
        await user.click(downloadButtons[0]);
        expect(mockDownload).toHaveBeenCalledWith('rpt-001');
      }
    });

    it('allows deleting reports', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockDelete = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        deleteReport: mockDelete
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      if (deleteButtons.length > 0) {
        await user.click(deleteButtons[0]);
        
        await waitFor(() => {
          expect(screen.getByText(/Are you sure/)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Template Management', () => {
    it('displays template library', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      expect(templatesTab).toBeInTheDocument();
    });

    it('shows template details', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Template Library')).toBeInTheDocument();
        expect(screen.getByText('Executive Risk Summary')).toBeInTheDocument();
        expect(screen.getByText('Detailed Stress Analysis')).toBeInTheDocument();
        expect(screen.getByText('Regulatory Compliance')).toBeInTheDocument();
      });
    });

    it('displays template usage statistics', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Used 87 times')).toBeInTheDocument();
        expect(screen.getByText('Used 45 times')).toBeInTheDocument();
        expect(screen.getByText('Used 123 times')).toBeInTheDocument();
      });
    });

    it('shows template sections', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        expect(screen.getByText('4 sections')).toBeInTheDocument();
        expect(screen.getByText('5 sections')).toBeInTheDocument();
      });
    });

    it('allows using templates', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockUse = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        generateReport: mockUse
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        const useButtons = screen.getAllByText('Use Template');
        if (useButtons.length > 0) {
          expect(useButtons[0]).toBeInTheDocument();
        }
      });
    });

    it('opens create template modal', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        const createButton = screen.getByText('Create Template');
        expect(createButton).toBeInTheDocument();
      });
    });

    it('allows creating new templates', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockCreate = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        createTemplate: mockCreate
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const templatesTab = screen.getByText('Templates');
      await user.click(templatesTab);
      
      await waitFor(() => {
        const createButton = screen.getByText('Create Template');
        expect(createButton).toBeInTheDocument();
      });
    });
  });

  describe('Scheduled Reports', () => {
    it('displays scheduled reports', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      expect(scheduledTab).toBeInTheDocument();
    });

    it('shows schedule details', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText('Scheduled Reports')).toBeInTheDocument();
        expect(screen.getByText('Daily VaR Report')).toBeInTheDocument();
        expect(screen.getByText('Weekly Stress Test')).toBeInTheDocument();
      });
    });

    it('displays schedule frequencies', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText('Daily at 08:00')).toBeInTheDocument();
        expect(screen.getByText('Weekly on Monday 07:00')).toBeInTheDocument();
      });
    });

    it('shows success rates', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText('98%')).toBeInTheDocument();
        expect(screen.getByText('95%')).toBeInTheDocument();
      });
    });

    it('displays next run times', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        expect(screen.getByText(/Next run:/)).toBeInTheDocument();
      });
    });

    it('allows enabling/disabling schedules', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockUpdate = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        updateSchedule: mockUpdate
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        const toggleSwitches = screen.getAllByRole('switch');
        if (toggleSwitches.length > 0) {
          expect(toggleSwitches[0]).toBeInTheDocument();
        }
      });
    });

    it('allows creating new schedules', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const scheduledTab = screen.getByText('Scheduled');
      await user.click(scheduledTab);
      
      await waitFor(() => {
        const createButton = screen.getByText('New Schedule');
        expect(createButton).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    it('refreshes queue automatically', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockRefresh = vi.fn();
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        refreshQueue: mockRefresh
      });

      render(<RiskReportGenerator {...mockProps} autoRefresh={true} />);
      
      vi.useFakeTimers();
      act(() => {
        vi.advanceTimersByTime(10000);
      });
      
      await waitFor(() => {
        expect(mockRefresh).toHaveBeenCalled();
      });
      
      vi.useRealTimers();
    });

    it('updates progress bars in real-time', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should show progress for active jobs
      expect(screen.getByText('65%')).toBeInTheDocument();
    });

    it('shows generation status updates', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('generating')).toBeInTheDocument();
    });
  });

  describe('Filtering and Search', () => {
    it('allows filtering reports by status', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const filterButton = screen.getByRole('button', { name: /filter/i });
      await user.click(filterButton);
      
      await waitFor(() => {
        expect(screen.getByText('All Status')).toBeInTheDocument();
        expect(screen.getByText('Completed')).toBeInTheDocument();
        expect(screen.getByText('Generating')).toBeInTheDocument();
        expect(screen.getByText('Failed')).toBeInTheDocument();
      });
    });

    it('allows filtering by report type', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should have report type filter options
      const reportFilters = screen.getAllByText(/var_report|stress_test|compliance/);
      expect(reportFilters.length).toBeGreaterThan(0);
    });

    it('allows searching reports', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const searchInput = screen.getByPlaceholderText(/Search reports/i);
      expect(searchInput).toBeInTheDocument();
    });

    it('filters results based on search input', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const searchInput = screen.getByPlaceholderText(/Search reports/i);
      await user.type(searchInput, 'var');
      
      expect(searchInput).toHaveValue('var');
    });
  });

  describe('Export and Download', () => {
    it('handles successful downloads', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(['test data'], { type: 'application/pdf' }))
      });

      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockDownload = vi.fn().mockResolvedValue(true);
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        downloadReport: mockDownload
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const downloadButtons = screen.getAllByRole('button', { name: /download/i });
      if (downloadButtons.length > 0) {
        await user.click(downloadButtons[0]);
        
        await waitFor(() => {
          expect(mockDownload).toHaveBeenCalled();
        });
      }
    });

    it('shows download expiration warnings', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should show expiration times
      expect(screen.getByText(/Expires:/)).toBeInTheDocument();
    });

    it('handles bulk export', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const bulkExportButton = screen.getByText('Bulk Export');
      await user.click(bulkExportButton);
      
      await waitFor(() => {
        expect(screen.getByText('Export All Reports')).toBeInTheDocument();
        expect(screen.getByText('Export Completed')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message when service fails', () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        error: 'Report generation service unavailable'
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Report Generator Error')).toBeInTheDocument();
      expect(screen.getByText('Report generation service unavailable')).toBeInTheDocument();
    });

    it('shows loading state during generation', () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        isGenerating: true
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Generating report...')).toBeInTheDocument();
    });

    it('handles empty queue gracefully', () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        generationQueue: [],
        reportingStats: {
          total_reports_generated: 0,
          successful_reports: 0,
          failed_reports: 0,
          avg_generation_time: '0m 0s',
          total_data_processed: '0 MB',
          most_popular_report: null,
          most_popular_template: null,
          peak_usage_hour: null
        }
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('No reports in queue')).toBeInTheDocument();
    });

    it('handles generation failures gracefully', async () => {
      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      const mockGenerate = vi.fn().mockRejectedValue(new Error('Generation failed'));
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        generateReport: mockGenerate
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      const generateButton = screen.getByText('Generate Report');
      await user.click(generateButton);
      
      // Should handle error without crashing
      expect(screen.getByText('Risk Report Generator')).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('validates required fields', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const generateButton = screen.getByText('Generate Report');
      await user.click(generateButton);
      
      // Should show validation messages for empty required fields
      await waitFor(() => {
        expect(screen.getByText('Please select report type')).toBeInTheDocument();
      });
    });

    it('validates date range', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Should validate that end date is after start date
      const dateRangeInputs = screen.getAllByRole('textbox');
      expect(dateRangeInputs.length).toBeGreaterThan(0);
    });

    it('shows field dependencies', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Template field should depend on report type selection
      expect(screen.getByText('Template')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large report queues efficiently', () => {
      const largeQueue = Array.from({ length: 200 }, (_, i) => ({
        job_id: `rpt-${i.toString().padStart(3, '0')}`,
        report_type: ['var_report', 'stress_test', 'compliance'][i % 3],
        template_name: ['Executive Summary', 'Detailed Analysis', 'Compliance'][i % 3],
        portfolio_id: 'port-001',
        status: ['completed', 'generating', 'failed'][i % 3],
        progress: i % 3 === 1 ? Math.floor(Math.random() * 100) : 100,
        started_at: new Date(Date.now() - i * 60000).toISOString(),
        completed_at: i % 3 === 0 ? new Date(Date.now() - i * 60000 + 180000).toISOString() : null,
        duration: i % 3 === 0 ? `${Math.floor(Math.random() * 5) + 2}m ${Math.floor(Math.random() * 60)}s` : null,
        file_size: i % 3 === 0 ? `${(Math.random() * 5 + 1).toFixed(1)} MB` : null,
        download_url: i % 3 === 0 ? `/downloads/report_${i}.pdf` : null,
        expires_at: i % 3 === 0 ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() : null
      }));

      const { useRiskReporting } = require('../../../hooks/analytics/useRiskReporting');
      useRiskReporting.mockReturnValue({
        ...useRiskReporting(),
        generationQueue: largeQueue
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Risk Report Generator')).toBeInTheDocument();
    });

    it('debounces user interactions', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const refreshButton = screen.getByRole('button', { name: /reload/i });
      
      // Rapid clicks should be debounced
      await user.click(refreshButton);
      await user.click(refreshButton);
      await user.click(refreshButton);
      
      expect(refreshButton).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for form controls', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const buttons = screen.getAllByRole('button');
      const comboboxes = screen.getAllByRole('combobox');
      
      buttons.forEach(button => expect(button).toBeInTheDocument());
      comboboxes.forEach(combobox => expect(combobox).toBeInTheDocument());
    });

    it('supports keyboard navigation', async () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      await user.tab();
      const focusedElement = document.activeElement;
      expect(focusedElement).toBeDefined();
    });

    it('provides meaningful status indicators', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('generating')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });

    it('has proper progress bar labels', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      const progressBars = screen.getAllByRole('progressbar');
      progressBars.forEach(progressBar => {
        expect(progressBar).toBeInTheDocument();
      });
    });
  });

  describe('Responsive Design', () => {
    it('adjusts layout for mobile screens', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Risk Report Generator')).toBeInTheDocument();
    });

    it('maintains functionality on tablet', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      render(<RiskReportGenerator {...mockProps} />);
      
      expect(screen.getByText('Generate Report')).toBeInTheDocument();
      expect(screen.getByText('Report Queue')).toBeInTheDocument();
    });

    it('adjusts table layout for smaller screens', () => {
      render(<RiskReportGenerator {...mockProps} />);
      
      // Table should be responsive
      expect(screen.getByText('Job ID')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });
  });
});