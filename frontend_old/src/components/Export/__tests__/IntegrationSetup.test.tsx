/**
 * Story 5.3: Unit Tests for IntegrationSetup Component
 * Tests for third-party API integration setup interface
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { notification } from 'antd';
import IntegrationSetup from '../IntegrationSetup';
import { ApiIntegration, AuthenticationType, IntegrationStatus } from '../../../types/export';

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
    },
  };
});

const mockIntegration: ApiIntegration = {
  id: 'integration-001',
  name: 'Portfolio Analytics API',
  endpoint: 'https://api.example.com/portfolio',
  authentication: {
    type: AuthenticationType.API_KEY,
    api_key: 'test-key-123',
    header_name: 'X-API-Key'
  },
  data_mapping: [
    {
      source_field: 'total_pnl',
      target_field: 'portfolio_value'
    }
  ],
  schedule: {
    frequency: 'hourly',
    enabled: true
  },
  status: IntegrationStatus.ACTIVE
};

const defaultProps = {
  visible: true,
  onCancel: vi.fn(),
  onSave: vi.fn().mockResolvedValue(undefined),
};

describe('IntegrationSetup', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders create modal when no initial integration provided', () => {
      render(<IntegrationSetup {...defaultProps} />);
      
      expect(screen.getByText('Create API Integration')).toBeInTheDocument();
      expect(screen.getByText('Basic Information')).toBeInTheDocument();
    });

    it('renders edit modal when initial integration provided', () => {
      render(<IntegrationSetup {...defaultProps} initialIntegration={mockIntegration} />);
      
      expect(screen.getByText('Edit API Integration')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Portfolio Analytics API')).toBeInTheDocument();
    });

    it('renders all step tabs', () => {
      render(<IntegrationSetup {...defaultProps} />);
      
      expect(screen.getByText('Basic Info')).toBeInTheDocument();
      expect(screen.getByText('Authentication')).toBeInTheDocument();
      expect(screen.getByText('Field Mapping')).toBeInTheDocument();
      expect(screen.getByText('Sync Settings')).toBeInTheDocument();
    });

    it('does not render when visible is false', () => {
      render(<IntegrationSetup {...defaultProps} visible={false} />);
      
      expect(screen.queryByText('Create API Integration')).not.toBeInTheDocument();
    });
  });

  describe('Basic Information Step', () => {
    it('requires integration name', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please enter integration name')).toBeInTheDocument();
      });
    });

    it('requires API endpoint', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill name but not endpoint
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please enter API endpoint')).toBeInTheDocument();
      });
    });

    it('validates URL format for endpoint', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill with invalid URL
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'invalid-url');
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(screen.getByText('Please enter a valid URL')).toBeInTheDocument();
      });
    });
  });

  describe('Authentication Step', () => {
    it('shows API key fields when API Key selected', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to authentication step
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Select API Key (should be default)
      expect(screen.getByText('API Key')).toBeInTheDocument();
      expect(screen.getByText('Enter your API key')).toBeInTheDocument();
      expect(screen.getByText('Header Name')).toBeInTheDocument();
    });

    it('shows OAuth fields when OAuth selected', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to authentication step
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Select OAuth
      const authTypeSelect = screen.getByRole('combobox');
      await user.click(authTypeSelect);
      
      const oauthOption = screen.getByText('OAuth 2.0');
      await user.click(oauthOption);
      
      expect(screen.getByText('Client ID')).toBeInTheDocument();
      expect(screen.getByText('Client Secret')).toBeInTheDocument();
    });

    it('shows Basic Auth fields when Basic selected', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to authentication step
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Select Basic Auth
      const authTypeSelect = screen.getByRole('combobox');
      await user.click(authTypeSelect);
      
      const basicOption = screen.getByText('Basic Authentication');
      await user.click(basicOption);
      
      expect(screen.getByText('Username')).toBeInTheDocument();
      expect(screen.getByText('Password')).toBeInTheDocument();
    });

    it('allows testing connection', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'https://api.example.com/test');
      
      // Navigate to authentication step
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Fill API key
      const apiKeyInput = screen.getByPlaceholderText('Enter your API key');
      await user.type(apiKeyInput, 'test-key');
      
      // Test connection
      const testButton = screen.getByText('Test Connection');
      await user.click(testButton);
      
      expect(screen.getByText('Testing...')).toBeInTheDocument();
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Connection Test Successful',
          description: 'API endpoint is reachable and authentication is valid',
        });
      }, { timeout: 3000 });
    });
  });

  describe('Field Mapping Step', () => {
    it('allows adding field mappings', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to field mapping step
      const mappingTab = screen.getByText('Field Mapping');
      await user.click(mappingTab);
      
      const addMappingButton = screen.getByText('Add Mapping');
      await user.click(addMappingButton);
      
      expect(screen.getByText('Select source field')).toBeInTheDocument();
      expect(screen.getByText('Select target field')).toBeInTheDocument();
    });

    it('shows validation warning when no field mappings added', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'https://api.example.com/test');
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.warning).toHaveBeenCalledWith({
          message: 'No Field Mappings',
          description: 'Add at least one field mapping for data synchronization',
        });
      });
    });

    it('allows removing field mappings', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} initialIntegration={mockIntegration} />);
      
      // Navigate to field mapping step
      const mappingTab = screen.getByText('Field Mapping');
      await user.click(mappingTab);
      
      // Should have initial mapping
      expect(screen.getByText('total_pnl')).toBeInTheDocument();
      
      // Remove mapping
      const deleteButton = screen.getByLabelText('delete');
      await user.click(deleteButton);
      
      expect(screen.queryByText('total_pnl')).not.toBeInTheDocument();
    });
  });

  describe('Sync Settings Step', () => {
    it('enables sync settings when switch is turned on', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to sync settings step
      const syncTab = screen.getByText('Sync Settings');
      await user.click(syncTab);
      
      const enableSwitch = screen.getByRole('switch');
      await user.click(enableSwitch);
      
      expect(screen.getByText('Select sync frequency')).toBeInTheDocument();
      expect(screen.getByText('Number of retry attempts')).toBeInTheDocument();
    });

    it('hides sync options when disabled', () => {
      render(<IntegrationSetup {...defaultProps} />);
      
      // Navigate to sync settings step
      const syncTab = screen.getByText('Sync Settings');
      fireEvent.click(syncTab);
      
      expect(screen.queryByText('Select sync frequency')).not.toBeInTheDocument();
    });
  });

  describe('Save Functionality', () => {
    it('calls onSave with correct integration data', async () => {
      const user = userEvent.setup();
      const onSave = vi.fn().mockResolvedValue(undefined);
      render(<IntegrationSetup {...defaultProps} onSave={onSave} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'https://api.example.com/test');
      
      // Navigate to field mapping and add mapping
      const mappingTab = screen.getByText('Field Mapping');
      await user.click(mappingTab);
      
      const addMappingButton = screen.getByText('Add Mapping');
      await user.click(addMappingButton);
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onSave).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'Test Integration',
            endpoint: 'https://api.example.com/test',
            status: IntegrationStatus.ACTIVE
          })
        );
      });
    });

    it('shows success notification on save', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} initialIntegration={mockIntegration} />);
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(notification.success).toHaveBeenCalledWith({
          message: 'Integration Saved',
          description: 'API integration has been configured successfully',
        });
      });
    });

    it('calls onCancel after successful save', async () => {
      const user = userEvent.setup();
      const onCancel = vi.fn();
      render(<IntegrationSetup {...defaultProps} onCancel={onCancel} initialIntegration={mockIntegration} />);
      
      const saveButton = screen.getByText('Save Integration');
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(onCancel).toHaveBeenCalled();
      });
    });

    it('handles save errors gracefully', async () => {
      const user = userEvent.setup();
      const onSave = vi.fn().mockRejectedValue(new Error('Save failed'));
      render(<IntegrationSetup {...defaultProps} onSave={onSave} initialIntegration={mockIntegration} />);
      
      const saveButton = screen.getByText('Save Integration');
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
      render(<IntegrationSetup {...defaultProps} onCancel={onCancel} />);
      
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);
      
      expect(onCancel).toHaveBeenCalled();
    });
  });

  describe('Step Navigation', () => {
    it('allows navigation between steps', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Click on Authentication step
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      expect(screen.getByText('Authentication Type')).toBeInTheDocument();
      
      // Click on Field Mapping step
      const mappingTab = screen.getByText('Field Mapping');
      await user.click(mappingTab);
      
      expect(screen.getByText('Field Mappings')).toBeInTheDocument();
    });

    it('shows next/previous buttons', () => {
      render(<IntegrationSetup {...defaultProps} />);
      
      expect(screen.getByText('Next')).toBeInTheDocument();
      // Previous button should not be visible on first step
      expect(screen.queryByText('Previous')).not.toBeInTheDocument();
    });
  });

  describe('Form Population', () => {
    it('populates form with initial integration data', () => {
      render(<IntegrationSetup {...defaultProps} initialIntegration={mockIntegration} />);
      
      expect(screen.getByDisplayValue('Portfolio Analytics API')).toBeInTheDocument();
      expect(screen.getByDisplayValue('https://api.example.com/portfolio')).toBeInTheDocument();
    });

    it('resets form when no initial integration provided', () => {
      render(<IntegrationSetup {...defaultProps} />);
      
      expect(screen.getByPlaceholderText('e.g., Portfolio Analytics API')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('https://api.example.com/v1/data')).toBeInTheDocument();
    });
  });

  describe('Connection Testing', () => {
    it('shows connection test results', async () => {
      const user = userEvent.setup();
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill required fields
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'https://api.example.com/test');
      
      // Navigate to authentication
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Fill API key
      const apiKeyInput = screen.getByPlaceholderText('Enter your API key');
      await user.type(apiKeyInput, 'test-key');
      
      // Test connection
      const testButton = screen.getByText('Test Connection');
      await user.click(testButton);
      
      await waitFor(() => {
        expect(screen.getByText('Connection OK')).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Error Handling', () => {
    it('handles connection test failures gracefully', async () => {
      const user = userEvent.setup();
      
      // Mock a failing test
      const mockFetch = vi.fn().mockRejectedValue(new Error('Connection failed'));
      global.fetch = mockFetch;
      
      render(<IntegrationSetup {...defaultProps} />);
      
      // Fill basic info
      const nameInput = screen.getByPlaceholderText('e.g., Portfolio Analytics API');
      await user.type(nameInput, 'Test Integration');
      
      const endpointInput = screen.getByPlaceholderText('https://api.example.com/v1/data');
      await user.type(endpointInput, 'https://api.example.com/test');
      
      // Navigate to authentication
      const authTab = screen.getByText('Authentication');
      await user.click(authTab);
      
      // Test should handle errors gracefully
      const testButton = screen.getByText('Test Connection');
      expect(testButton).toBeInTheDocument();
    });
  });
});