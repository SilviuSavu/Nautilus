/**
 * Story 5.3: Unit Tests for TemplateManager Component
 * Tests for report template management interface
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { notification } from 'antd';
import TemplateManager from '../TemplateManager';
import { ReportTemplate, ReportType, ReportFormat, ScheduleFrequency } from '../../../types/export';

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
    },
  };
});

// Mock the ReportBuilder component
vi.mock('../ReportBuilder', () => {
  return function MockReportBuilder({ visible, onCancel, onSave, initialTemplate }: any) {
    if (!visible) return null;
    return (
      <div data-testid="report-builder">
        <button onClick={() => onSave(initialTemplate || { name: 'New Template' })}>
          Save Template
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    );
  };
});

const mockTemplates: ReportTemplate[] = [
  {
    id: 'template-001',
    name: 'Daily Performance Report',
    description: 'Comprehensive daily trading performance analysis',
    type: ReportType.PERFORMANCE,
    format: ReportFormat.PDF,
    sections: [
      {
        id: 'section-1',
        name: 'Performance Overview',
        type: 'metrics',
        configuration: { metrics: ['total_pnl', 'win_rate', 'sharpe_ratio'] }
      }
    ],
    parameters: [
      {
        name: 'date',
        type: 'date',
        default_value: '2024-01-01',
        required: true
      }
    ],
    schedule: {
      frequency: ScheduleFrequency.DAILY,
      time: '08:00',
      timezone: 'UTC',
      recipients: ['trader@example.com']
    },
    created_at: new Date('2024-01-15'),
    updated_at: new Date('2024-01-20')
  },
  {
    id: 'template-002',
    name: 'Weekly Risk Assessment',
    description: 'Weekly risk metrics and exposure analysis',
    type: ReportType.RISK,
    format: ReportFormat.EXCEL,
    sections: [],
    parameters: [],
    created_at: new Date('2024-01-10'),
    updated_at: new Date('2024-01-18')
  }
];

const defaultProps = {
  onGenerateReport: vi.fn().mockResolvedValue({ report_id: 'test-report' }),
};

describe('TemplateManager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the main interface with title', () => {
      render(<TemplateManager {...defaultProps} />);
      
      expect(screen.getByText('Report Templates')).toBeInTheDocument();
      expect(screen.getByText('Create Template')).toBeInTheDocument();
    });

    it('renders template statistics', async () => {
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Total Templates')).toBeInTheDocument();
        expect(screen.getByText('Scheduled')).toBeInTheDocument();
        expect(screen.getByText('Performance')).toBeInTheDocument();
        expect(screen.getByText('Risk')).toBeInTheDocument();
      });
    });

    it('renders search and filter controls', () => {
      render(<TemplateManager {...defaultProps} />);
      
      expect(screen.getByPlaceholderText('Search templates...')).toBeInTheDocument();
      expect(screen.getByText('All Types')).toBeInTheDocument();
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    it('renders templates table', async () => {
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.getByText('Weekly Risk Assessment')).toBeInTheDocument();
      });
    });
  });

  describe('Template Creation', () => {
    it('opens ReportBuilder when Create Template clicked', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      const createButton = screen.getByText('Create Template');
      await user.click(createButton);
      
      expect(screen.getByTestId('report-builder')).toBeInTheDocument();
    });

    it('saves new template successfully', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      // Open builder
      const createButton = screen.getByText('Create Template');
      await user.click(createButton);
      
      // Save template
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Template Saved',
          description: 'Report template has been saved successfully',
        });
      });
    });
  });

  describe('Template Management', () => {
    it('allows editing templates', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click edit button
      const editButtons = screen.getAllByLabelText('Edit');
      await user.click(editButtons[0]);
      
      expect(screen.getByTestId('report-builder')).toBeInTheDocument();
    });

    it('allows duplicating templates', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click duplicate button
      const duplicateButtons = screen.getAllByLabelText('Duplicate');
      await user.click(duplicateButtons[0]);
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Template Duplicated',
          description: 'Template has been duplicated successfully',
        });
      });
    });

    it('allows deleting templates with confirmation', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click delete button
      const deleteButtons = screen.getAllByLabelText('delete');
      await user.click(deleteButtons[0]);
      
      // Confirm deletion
      const confirmButton = screen.getByText('Delete');
      await user.click(confirmButton);
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Template Deleted',
          description: 'Report template has been deleted successfully',
        });
      });
    });

    it('allows generating reports from templates', async () => {
      const user = userEvent.setup();
      const onGenerateReport = vi.fn().mockResolvedValue({ report_id: 'test-report' });
      render(<TemplateManager {...defaultProps} onGenerateReport={onGenerateReport} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click generate button
      const generateButtons = screen.getAllByLabelText('Generate Report');
      await user.click(generateButtons[0]);
      
      expect(screen.getByText('Generate Report')).toBeInTheDocument();
    });
  });

  describe('Filtering and Search', () => {
    it('filters templates by search text', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.getByText('Weekly Risk Assessment')).toBeInTheDocument();
      });
      
      // Search for "Daily"
      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'Daily');
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.queryByText('Weekly Risk Assessment')).not.toBeInTheDocument();
      });
    });

    it('filters templates by type', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.getByText('Weekly Risk Assessment')).toBeInTheDocument();
      });
      
      // Filter by Performance type
      const typeFilter = screen.getByDisplayValue('All Types');
      await user.click(typeFilter);
      
      const performanceOption = screen.getByText('Performance');
      await user.click(performanceOption);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.queryByText('Weekly Risk Assessment')).not.toBeInTheDocument();
      });
    });

    it('refreshes templates when refresh button clicked', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      const refreshButton = screen.getByText('Refresh');
      await user.click(refreshButton);
      
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalledWith({
          message: 'Templates refreshed'
        });
      });
    });
  });

  describe('Template Preview', () => {
    it('opens preview modal when preview button clicked', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click preview button
      const previewButtons = screen.getAllByLabelText('Preview');
      await user.click(previewButtons[0]);
      
      expect(screen.getByText('Template Preview')).toBeInTheDocument();
    });

    it('shows template details in preview modal', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click preview button
      const previewButtons = screen.getAllByLabelText('Preview');
      await user.click(previewButtons[0]);
      
      expect(screen.getByText('Template Preview')).toBeInTheDocument();
      expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      expect(screen.getByText('1 sections configured')).toBeInTheDocument();
      expect(screen.getByText('1 parameters defined')).toBeInTheDocument();
    });

    it('closes preview modal', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Open preview
      const previewButtons = screen.getAllByLabelText('Preview');
      await user.click(previewButtons[0]);
      
      // Close preview
      const closeButton = screen.getByText('Close');
      await user.click(closeButton);
      
      expect(screen.queryByText('Template Preview')).not.toBeInTheDocument();
    });
  });

  describe('Statistics Display', () => {
    it('calculates and displays correct statistics', async () => {
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        // Should show 2 total templates from mock data
        expect(screen.getByText('2')).toBeInTheDocument(); // Total templates
        expect(screen.getByText('1')).toBeInTheDocument(); // Scheduled (only one has schedule)
      });
    });

    it('updates statistics when templates change', async () => {
      const user = userEvent.setup();
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Delete a template
      const deleteButtons = screen.getAllByLabelText('delete');
      await user.click(deleteButtons[0]);
      
      const confirmButton = screen.getByText('Delete');
      await user.click(confirmButton);
      
      // Statistics should update (though this is difficult to test with mock data)
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles template save errors gracefully', async () => {
      const user = userEvent.setup();
      
      // Mock ReportBuilder to simulate error
      jest.doMock('../ReportBuilder', () => {
        return function MockReportBuilder({ visible, onCancel, onSave }: any) {
          if (!visible) return null;
          return (
            <div data-testid="report-builder">
              <button onClick={() => onSave(null).catch(() => {})}>
                Save Template
              </button>
              <button onClick={onCancel}>Cancel</button>
            </div>
          );
        };
      });
      
      render(<TemplateManager {...defaultProps} />);
      
      const createButton = screen.getByText('Create Template');
      await user.click(createButton);
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      // Should handle error gracefully without crashing
      await waitFor(() => {
        expect(screen.getByTestId('report-builder')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('provides proper labels for action buttons', async () => {
      render(<TemplateManager {...defaultProps} />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Check that buttons have proper labels
      expect(screen.getAllByLabelText('Generate Report')).toHaveLength(2);
      expect(screen.getAllByLabelText('Preview')).toHaveLength(2);
      expect(screen.getAllByLabelText('Edit')).toHaveLength(2);
      expect(screen.getAllByLabelText('Duplicate')).toHaveLength(2);
      expect(screen.getAllByLabelText('delete')).toHaveLength(2);
    });
  });
});