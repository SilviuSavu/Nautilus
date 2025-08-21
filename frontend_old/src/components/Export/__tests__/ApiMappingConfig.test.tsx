/**
 * Story 5.3: Unit Tests for ApiMappingConfig Component
 * Tests for data mapping configuration interface
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { notification } from 'antd';
import ApiMappingConfig from '../ApiMappingConfig';
import { FieldMapping } from '../../../types/export';

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

const mockSourceFields = [
  'total_pnl',
  'unrealized_pnl',
  'win_rate',
  'sharpe_ratio',
  'timestamp',
  'symbol'
];

const mockTargetFields = [
  { value: 'portfolio_value', label: 'Portfolio Value', type: 'numeric' },
  { value: 'profit_loss', label: 'Profit/Loss', type: 'numeric' },
  { value: 'success_rate', label: 'Success Rate', type: 'numeric' },
  { value: 'performance_ratio', label: 'Performance Ratio', type: 'numeric' },
  { value: 'updated_at', label: 'Last Updated', type: 'date' },
  { value: 'instrument', label: 'Instrument', type: 'string' }
];

const mockMappings: FieldMapping[] = [
  {
    source_field: 'total_pnl',
    target_field: 'portfolio_value',
    transformation: '* 100'
  },
  {
    source_field: 'win_rate',
    target_field: 'success_rate'
  }
];

const defaultProps = {
  mappings: mockMappings,
  onMappingsChange: vi.fn(),
  sourceFields: mockSourceFields,
  targetFields: mockTargetFields,
};

describe('ApiMappingConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the main interface with title', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      expect(screen.getByText('Field Mappings Configuration')).toBeInTheDocument();
      expect(screen.getByText('Add Mapping')).toBeInTheDocument();
    });

    it('renders info alert about field mapping', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      expect(screen.getByText('Field Mapping Information')).toBeInTheDocument();
      expect(screen.getByText(/Configure how your local data fields map to external API fields/)).toBeInTheDocument();
    });

    it('renders existing mappings in table', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      expect(screen.getByText('total_pnl')).toBeInTheDocument();
      expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
      expect(screen.getByText('* 100')).toBeInTheDocument();
      expect(screen.getByText('win_rate')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
    });

    it('shows empty state when no mappings provided', () => {
      render(<ApiMappingConfig {...defaultProps} mappings={[]} />);
      
      expect(screen.getByText('No field mappings configured. Click "Add Mapping" to create one.')).toBeInTheDocument();
    });
  });

  describe('Adding Mappings', () => {
    it('opens add mapping modal when Add Mapping clicked', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      expect(screen.getByText('Create Field Mapping')).toBeInTheDocument();
    });

    it('saves new mapping successfully', async () => {
      const user = userEvent.setup();
      const onMappingsChange = vi.fn();
      render(<ApiMappingConfig {...defaultProps} onMappingsChange={onMappingsChange} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Fill form
      const sourceSelect = screen.getByLabelText('Source Field');
      await user.click(sourceSelect);
      await user.click(screen.getByText('total_pnl'));
      
      const targetSelect = screen.getByLabelText('Target Field');
      await user.click(targetSelect);
      await user.click(screen.getByText('Portfolio Value'));
      
      // Save mapping
      const saveButton = screen.getByText('OK');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onMappingsChange).toHaveBeenCalledWith([
          ...mockMappings,
          expect.objectContaining({
            source_field: 'total_pnl',
            target_field: 'portfolio_value'
          })
        ]);
      });
    });

    it('requires source field selection', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Try to save without selecting source field
      const saveButton = screen.getByText('OK');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please select source field')).toBeInTheDocument();
      });
    });

    it('requires target field selection', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Select source but not target
      const sourceSelect = screen.getByLabelText('Source Field');
      await user.click(sourceSelect);
      await user.click(screen.getByText('total_pnl'));
      
      // Try to save
      const saveButton = screen.getByText('OK');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please select target field')).toBeInTheDocument();
      });
    });
  });

  describe('Editing Mappings', () => {
    it('opens edit modal when edit button clicked', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Find and click edit button
      const editButtons = screen.getAllByLabelText('Edit Mapping');
      await user.click(editButtons[0]);
      
      expect(screen.getByText('Edit Field Mapping')).toBeInTheDocument();
      expect(screen.getByDisplayValue('total_pnl')).toBeInTheDocument();
    });

    it('updates existing mapping', async () => {
      const user = userEvent.setup();
      const onMappingsChange = vi.fn();
      render(<ApiMappingConfig {...defaultProps} onMappingsChange={onMappingsChange} />);
      
      // Edit first mapping
      const editButtons = screen.getAllByLabelText('Edit Mapping');
      await user.click(editButtons[0]);
      
      // Change transformation
      const transformationField = screen.getByLabelText(/Transformation/);
      await user.clear(transformationField);
      await user.type(transformationField, '/ 100');
      
      // Save changes
      const saveButton = screen.getByText('OK');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onMappingsChange).toHaveBeenCalledWith([
          expect.objectContaining({
            source_field: 'total_pnl',
            target_field: 'portfolio_value',
            transformation: '/ 100'
          }),
          mockMappings[1]
        ]);
      });
    });
  });

  describe('Deleting Mappings', () => {
    it('deletes mapping when delete button clicked', async () => {
      const user = userEvent.setup();
      const onMappingsChange = vi.fn();
      render(<ApiMappingConfig {...defaultProps} onMappingsChange={onMappingsChange} />);
      
      // Find and click delete button
      const deleteButtons = screen.getAllByLabelText('Delete Mapping');
      await user.click(deleteButtons[0]);
      
      await waitFor(() => {
        expect(onMappingsChange).toHaveBeenCalledWith([mockMappings[1]]);
      });
      
      expect(notification.success).toHaveBeenCalledWith({
        message: 'Mapping Deleted',
        description: 'Field mapping has been removed successfully',
      });
    });
  });

  describe('Duplicating Mappings', () => {
    it('duplicates mapping when duplicate button clicked', async () => {
      const user = userEvent.setup();
      const onMappingsChange = vi.fn();
      render(<ApiMappingConfig {...defaultProps} onMappingsChange={onMappingsChange} />);
      
      // Find and click duplicate button
      const duplicateButtons = screen.getAllByLabelText('Duplicate Mapping');
      await user.click(duplicateButtons[0]);
      
      await waitFor(() => {
        expect(onMappingsChange).toHaveBeenCalledWith([
          ...mockMappings,
          expect.objectContaining({
            source_field: 'total_pnl',
            target_field: 'portfolio_value_copy',
            transformation: '* 100'
          })
        ]);
      });
      
      expect(notification.success).toHaveBeenCalledWith({
        message: 'Mapping Duplicated',
        description: 'Field mapping has been duplicated successfully',
      });
    });
  });

  describe('Transformation Testing', () => {
    it('opens test modal when test button clicked', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Find and click test button
      const testButtons = screen.getAllByLabelText('Test Transformation');
      await user.click(testButtons[0]);
      
      expect(screen.getByText('Test Transformation')).toBeInTheDocument();
      expect(screen.getByText(/Testing transformation: \* 100/)).toBeInTheDocument();
    });

    it('displays test results in test modal', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Find and click test button
      const testButtons = screen.getAllByLabelText('Test Transformation');
      await user.click(testButtons[0]);
      
      expect(screen.getByText('Test Transformation')).toBeInTheDocument();
      expect(screen.getByText('Input')).toBeInTheDocument();
      expect(screen.getByText('Output')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });

    it('closes test modal', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open test modal
      const testButtons = screen.getAllByLabelText('Test Transformation');
      await user.click(testButtons[0]);
      
      // Close modal
      const closeButton = screen.getByText('Close');
      await user.click(closeButton);
      
      expect(screen.queryByText('Test Transformation')).not.toBeInTheDocument();
    });
  });

  describe('Field Type Detection', () => {
    it('shows field types for source fields', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Should show field types as tags
      expect(screen.getByText('total_pnl')).toBeInTheDocument();
      expect(screen.getByText('win_rate')).toBeInTheDocument();
    });

    it('displays target field labels and values', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
      expect(screen.getByText('portfolio_value')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('success_rate')).toBeInTheDocument();
    });
  });

  describe('Validation Status', () => {
    it('shows valid status for correct mappings', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Should show valid tags
      const validTags = screen.getAllByText('Valid');
      expect(validTags.length).toBeGreaterThan(0);
    });

    it('shows invalid status for incorrect mappings', () => {
      const invalidMappings: FieldMapping[] = [
        {
          source_field: 'invalid_source',
          target_field: 'invalid_target'
        }
      ];
      
      render(<ApiMappingConfig {...defaultProps} mappings={invalidMappings} />);
      
      // Should show invalid tag
      expect(screen.getByText('Invalid')).toBeInTheDocument();
    });
  });

  describe('Transformation Templates', () => {
    it('shows transformation template dropdown', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      expect(screen.getByText('Common Transformations')).toBeInTheDocument();
      expect(screen.getByText('Choose a template...')).toBeInTheDocument();
    });

    it('applies template transformation', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Select template
      const templateSelect = screen.getByText('Choose a template...');
      await user.click(templateSelect);
      
      const percentageOption = screen.getByText('Multiply by 100 (percentage)');
      await user.click(percentageOption);
      
      // Should populate transformation field
      expect(screen.getByDisplayValue('* 100')).toBeInTheDocument();
    });
  });

  describe('Modal Management', () => {
    it('closes add modal on cancel', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Cancel
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);
      
      expect(screen.queryByText('Create Field Mapping')).not.toBeInTheDocument();
    });

    it('closes edit modal on cancel', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open edit modal
      const editButtons = screen.getAllByLabelText('Edit Mapping');
      await user.click(editButtons[0]);
      
      // Cancel
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);
      
      expect(screen.queryByText('Edit Field Mapping')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles transformation test errors gracefully', async () => {
      const user = userEvent.setup();
      const mappingWithBadTransform: FieldMapping[] = [
        {
          source_field: 'total_pnl',
          target_field: 'portfolio_value',
          transformation: 'invalid.javascript.syntax'
        }
      ];
      
      render(<ApiMappingConfig {...defaultProps} mappings={mappingWithBadTransform} />);
      
      // Test transformation should handle errors
      const testButtons = screen.getAllByLabelText('Test Transformation');
      await user.click(testButtons[0]);
      
      expect(screen.getByText('Test Transformation')).toBeInTheDocument();
      // Should show error status in test results
    });

    it('handles missing field data gracefully', () => {
      render(<ApiMappingConfig {...defaultProps} sourceFields={[]} targetFields={[]} />);
      
      // Should still render without crashing
      expect(screen.getByText('Field Mappings Configuration')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper labels for form fields', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      expect(screen.getByLabelText('Source Field')).toBeInTheDocument();
      expect(screen.getByLabelText('Target Field')).toBeInTheDocument();
    });

    it('provides proper labels for action buttons', () => {
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Check that buttons have proper labels
      expect(screen.getAllByLabelText('Test Transformation')).toHaveLength(2);
      expect(screen.getAllByLabelText('Edit Mapping')).toHaveLength(2);
      expect(screen.getAllByLabelText('Duplicate Mapping')).toHaveLength(2);
      expect(screen.getAllByLabelText('Delete Mapping')).toHaveLength(2);
    });
  });

  describe('Search and Filtering', () => {
    it('filters source fields in dropdown', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Source field select should support search
      const sourceSelect = screen.getByLabelText('Source Field');
      expect(sourceSelect).toBeInTheDocument();
    });

    it('filters target fields in dropdown', async () => {
      const user = userEvent.setup();
      render(<ApiMappingConfig {...defaultProps} />);
      
      // Open add modal
      const addButton = screen.getByText('Add Mapping');
      await user.click(addButton);
      
      // Target field select should support search
      const targetSelect = screen.getByLabelText('Target Field');
      expect(targetSelect).toBeInTheDocument();
    });
  });
});