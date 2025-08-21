/**
 * Story 5.3: Unit Tests for ReportBuilder Component
 * Tests for visual report template builder interface
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { notification } from 'antd';
import ReportBuilder from '../ReportBuilder';
import { ReportTemplate, ReportType, ReportFormat } from '../../../types/export';

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
    },
  };
});

const mockTemplate: ReportTemplate = {
  id: 'test-template',
  name: 'Test Report Template',
  description: 'Test description',
  type: ReportType.PERFORMANCE,
  format: ReportFormat.PDF,
  sections: [
    {
      id: 'section-1',
      name: 'Performance Overview',
      type: 'metrics',
      configuration: { metrics: ['total_pnl', 'win_rate'] }
    }
  ],
  parameters: [
    {
      name: 'date_range',
      type: 'date_range',
      default_value: { days: 1 },
      required: true
    }
  ]
};

const defaultProps = {
  visible: true,
  onCancel: vi.fn(),
  onSave: vi.fn().mockResolvedValue(undefined),
};

describe('ReportBuilder', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders create modal when no initial template provided', () => {
      render(<ReportBuilder {...defaultProps} />);
      
      expect(screen.getByText('Create Report Template')).toBeInTheDocument();
      expect(screen.getByText('Basic Information')).toBeInTheDocument();
    });

    it('renders edit modal when initial template provided', () => {
      render(<ReportBuilder {...defaultProps} initialTemplate={mockTemplate} />);
      
      expect(screen.getByText('Edit Report Template')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Test Report Template')).toBeInTheDocument();
    });

    it('renders all step tabs', () => {
      render(<ReportBuilder {...defaultProps} />);
      
      expect(screen.getByText('Basic Info')).toBeInTheDocument();
      expect(screen.getByText('Sections')).toBeInTheDocument();
      expect(screen.getByText('Parameters')).toBeInTheDocument();
      expect(screen.getByText('Scheduling')).toBeInTheDocument();
    });

    it('does not render when visible is false', () => {
      render(<ReportBuilder {...defaultProps} visible={false} />);
      
      expect(screen.queryByText('Create Report Template')).not.toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('requires template name', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please enter template name')).toBeInTheDocument();
      });
    });

    it('requires report type', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Fill name but not type
      const nameInput = screen.getByPlaceholderText('e.g., Daily Performance Report');
      await user.type(nameInput, 'Test Template');
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please select report type')).toBeInTheDocument();
      });
    });

    it('requires description', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Fill required fields except description
      const nameInput = screen.getByPlaceholderText('e.g., Daily Performance Report');
      await user.type(nameInput, 'Test Template');
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please enter description')).toBeInTheDocument();
      });
    });
  });

  describe('Section Management', () => {
    it('allows adding sections', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Navigate to sections step
      const sectionsTab = screen.getByText('Sections');
      await user.click(sectionsTab);
      
      const addSectionButton = screen.getByText('Add Section');
      await user.click(addSectionButton);
      
      expect(screen.getByText('Section 1')).toBeInTheDocument();
    });

    it('shows validation error when no sections added', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Daily Performance Report');
      await user.type(nameInput, 'Test Template');
      
      const descriptionInput = screen.getByPlaceholderText('Describe what this report template will generate...');
      await user.type(descriptionInput, 'Test description');
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.error).toHaveBeenCalledWith({
          message: 'Validation Error',
          description: 'Please add at least one section to the report template',
        });
      });
    });
  });

  describe('Parameter Management', () => {
    it('allows adding parameters', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Navigate to parameters step
      const parametersTab = screen.getByText('Parameters');
      await user.click(parametersTab);
      
      const addParameterButton = screen.getByText('Add Parameter');
      await user.click(addParameterButton);
      
      expect(screen.getByPlaceholderText('Parameter name')).toBeInTheDocument();
    });

    it('allows removing parameters', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} initialTemplate={mockTemplate} />);
      
      // Navigate to parameters step
      const parametersTab = screen.getByText('Parameters');
      await user.click(parametersTab);
      
      // Should have initial parameter
      expect(screen.getByDisplayValue('date_range')).toBeInTheDocument();
      
      // Remove parameter
      const deleteButton = screen.getByLabelText('delete');
      await user.click(deleteButton);
      
      expect(screen.queryByDisplayValue('date_range')).not.toBeInTheDocument();
    });
  });

  describe('Scheduling', () => {
    it('enables scheduling when switch is turned on', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Navigate to scheduling step
      const schedulingTab = screen.getByText('Scheduling');
      await user.click(schedulingTab);
      
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);
      
      expect(screen.getByText('Select frequency')).toBeInTheDocument();
      expect(screen.getByText('Enter email addresses')).toBeInTheDocument();
    });

    it('hides scheduling options when disabled', () => {
      render(<ReportBuilder {...defaultProps} />);
      
      // Navigate to scheduling step
      const schedulingTab = screen.getByText('Scheduling');
      fireEvent.click(schedulingTab);
      
      expect(screen.queryByText('Select frequency')).not.toBeInTheDocument();
    });
  });

  describe('Save Functionality', () => {
    it('calls onSave with correct template data', async () => {
      const user = userEvent.setup();
      const onSave = vi.fn().mockResolvedValue(undefined);
      render(<ReportBuilder {...defaultProps} onSave={onSave} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Daily Performance Report');
      await user.type(nameInput, 'Test Template');
      
      const descriptionInput = screen.getByPlaceholderText('Describe what this report template will generate...');
      await user.type(descriptionInput, 'Test description');
      
      // Navigate to sections and add one
      const sectionsTab = screen.getByText('Sections');
      await user.click(sectionsTab);
      
      const addSectionButton = screen.getByText('Add Section');
      await user.click(addSectionButton);
      
      const sectionNameInput = screen.getByPlaceholderText('Enter section name');
      await user.type(sectionNameInput, 'Test Section');
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onSave).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'Test Template',
            description: 'Test description',
            sections: expect.arrayContaining([
              expect.objectContaining({
                name: 'Test Section'
              })
            ])
          })
        );
      });
    });

    it('shows success notification on save', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} initialTemplate={mockTemplate} />);
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Template Saved',
          description: 'Report template has been saved successfully',
        });
      });
    });

    it('calls onCancel after successful save', async () => {
      const user = userEvent.setup();
      const onCancel = vi.fn();
      render(<ReportBuilder {...defaultProps} onCancel={onCancel} initialTemplate={mockTemplate} />);
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onCancel).toHaveBeenCalled();
      });
    });

    it('handles save errors gracefully', async () => {
      const user = userEvent.setup();
      const onSave = vi.fn().mockRejectedValue(new Error('Save failed'));
      render(<ReportBuilder {...defaultProps} onSave={onSave} initialTemplate={mockTemplate} />);
      
      const saveButton = screen.getByText('Save Template');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.error).toHaveBeenCalledWith({
          message: 'Save Failed',
          description: 'Save failed',
        });
      });
    });
  });

  describe('Cancel Functionality', () => {
    it('calls onCancel when cancel button clicked', async () => {
      const user = userEvent.setup();
      const onCancel = vi.fn();
      render(<ReportBuilder {...defaultProps} onCancel={onCancel} />);
      
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);
      
      expect(onCancel).toHaveBeenCalled();
    });

    it('calls onCancel when modal is closed', async () => {
      const onCancel = vi.fn();
      render(<ReportBuilder {...defaultProps} onCancel={onCancel} />);
      
      // Simulate ESC key press to close modal
      fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });
      
      expect(onCancel).toHaveBeenCalled();
    });
  });

  describe('Step Navigation', () => {
    it('allows navigation between steps', async () => {
      const user = userEvent.setup();
      render(<ReportBuilder {...defaultProps} />);
      
      // Click on Sections step
      const sectionsTab = screen.getByText('Sections');
      await user.click(sectionsTab);
      
      expect(screen.getByText('Report Sections')).toBeInTheDocument();
      
      // Click on Parameters step
      const parametersTab = screen.getByText('Parameters');
      await user.click(parametersTab);
      
      expect(screen.getByText('Template Parameters')).toBeInTheDocument();
    });

    it('shows next/previous buttons', () => {
      render(<ReportBuilder {...defaultProps} />);
      
      expect(screen.getByText('Next')).toBeInTheDocument();
      // Previous button should not be visible on first step
      expect(screen.queryByText('Previous')).not.toBeInTheDocument();
    });
  });
});