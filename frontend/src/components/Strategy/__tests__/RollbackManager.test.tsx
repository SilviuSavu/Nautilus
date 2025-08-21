import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { RollbackManager } from '../RollbackManager';
import { strategyService } from '../services/strategyService';
import { 
  StrategyVersion, 
  RollbackPlan, 
  RollbackValidation, 
  RollbackProgress 
} from '../types/strategyTypes';

// Mock the strategy service
vi.mock('../services/strategyService');

const mockStrategyService = vi.mocked(strategyService);

describe('RollbackManager', () => {
  const mockStrategyId = 'test-strategy-123';
  const currentVersion = 3;
  
  const mockTargetVersion: StrategyVersion = {
    id: 'version-2',
    config_id: mockStrategyId,
    version_number: 2,
    config_snapshot: {
      id: mockStrategyId,
      name: 'Test Strategy v2',
      template_id: 'template-1',
      parameters: { fast_period: 8, slow_period: 20 },
      risk_settings: {
        max_position_size: '8000',
        position_sizing_method: 'fixed' as const
      },
      deployment_settings: {
        mode: 'paper' as const,
        venue: 'IB'
      },
      user_id: 'user-1',
      version: 2,
      status: 'archived' as const,
      created_at: new Date(),
      updated_at: new Date(),
      tags: []
    },
    change_summary: 'Reduced position size for risk management',
    created_by: 'trader1',
    created_at: new Date('2024-01-02')
  };

  const mockRollbackPlan: RollbackPlan = {
    strategy_id: mockStrategyId,
    from_version: currentVersion,
    to_version: 2,
    changes_to_revert: [
      {
        id: 'change-1',
        strategy_id: mockStrategyId,
        change_type: 'parameter_change',
        timestamp: new Date('2024-01-03'),
        changed_by: 'trader1',
        description: 'Updated fast period from 8 to 10',
        changed_fields: ['fast_period'],
        auto_generated: false
      }
    ],
    execution_steps: [
      {
        step_id: 'backup',
        description: 'Create configuration backup',
        action_type: 'backup',
        estimated_duration: 5,
        critical: true
      },
      {
        step_id: 'stop_strategy',
        description: 'Stop strategy execution',
        action_type: 'strategy_restart',
        estimated_duration: 10,
        critical: true
      }
    ],
    risk_assessment: {
      risk_level: 'medium',
      warnings: ['Parameter changes may affect performance'],
      blockers: [],
      recommendations: ['Monitor performance after rollback']
    },
    estimated_duration_seconds: 60,
    backup_required: true,
    dependencies: []
  };

  const mockValidation: RollbackValidation = {
    validation_passed: true,
    validation_errors: [],
    warnings: ['This will revert recent parameter optimizations'],
    pre_rollback_checks: [
      {
        check_name: 'strategy_state',
        description: 'Strategy is in safe state for rollback',
        passed: true,
        details: 'Strategy is currently stopped'
      }
    ],
    backup_verification: {
      backup_created: true,
      backup_path: '/backups/test-strategy-123_backup',
      backup_size_mb: 2.5,
      backup_verified: true
    }
  };

  const mockProgress: RollbackProgress = {
    rollback_id: 'rollback-123',
    status: 'running',
    overall_progress: 60,
    current_step: 'Reverting configuration',
    current_operation: 'Applying version 2 configuration',
    completed_steps: 3,
    total_steps: 5,
    elapsed_seconds: 45,
    estimated_remaining_seconds: 30,
    errors: [],
    warnings: []
  };

  const defaultProps = {
    strategyId: mockStrategyId,
    currentVersion,
    targetVersion: mockTargetVersion,
    visible: true,
    onClose: vi.fn(),
    onRollbackComplete: vi.fn()
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockStrategyService.generateRollbackPlan.mockResolvedValue(mockRollbackPlan);
    mockStrategyService.validateRollback.mockResolvedValue(mockValidation);
    mockStrategyService.executeRollback.mockResolvedValue({ success: true });
  });

  it('renders rollback manager modal when visible', () => {
    render(<RollbackManager {...defaultProps} />);
    
    expect(screen.getByText('Rollback to Version 2')).toBeInTheDocument();
    expect(screen.getByText('Plan')).toBeInTheDocument();
  });

  it('generates and displays rollback plan on mount', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    await waitFor(() => {
      expect(mockStrategyService.generateRollbackPlan).toHaveBeenCalledWith(
        mockStrategyId,
        currentVersion,
        2
      );
    });

    await waitFor(() => {
      expect(screen.getByText('Rollback Plan: Version 3 → Version 2')).toBeInTheDocument();
      expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
      expect(screen.getByText('MEDIUM')).toBeInTheDocument();
    });
  });

  it('displays rollback plan details correctly', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    await waitFor(() => {
      // Risk level
      expect(screen.getByText('MEDIUM')).toBeInTheDocument();
      
      // Changes count
      expect(screen.getByText('1')).toBeInTheDocument(); // Changes to Revert
      
      // Estimated duration
      expect(screen.getByText('60')).toBeInTheDocument(); // Duration in seconds
      
      // Warnings
      expect(screen.getByText('Parameter changes may affect performance')).toBeInTheDocument();
    });
  });

  it('shows execution steps in timeline', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Rollback Steps')).toBeInTheDocument();
      expect(screen.getByText('Create configuration backup')).toBeInTheDocument();
      expect(screen.getByText('Stop strategy execution')).toBeInTheDocument();
    });
  });

  it('proceeds to configuration step when configure button clicked', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    await waitFor(() => {
      const configureButton = screen.getByText('Configure');
      fireEvent.click(configureButton);
    });

    await waitFor(() => {
      expect(screen.getByText('Rollback Configuration')).toBeInTheDocument();
      expect(screen.getByText('Create backup before rollback (recommended)')).toBeInTheDocument();
    });
  });

  it('shows rollback settings with default values', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to configure step
    await waitFor(() => {
      const configureButton = screen.getByText('Configure');
      fireEvent.click(configureButton);
    });

    await waitFor(() => {
      // Check default checkboxes
      const backupCheckbox = screen.getByRole('checkbox', { name: /Create backup before rollback/ });
      const stopStrategyCheckbox = screen.getByRole('checkbox', { name: /Stop strategy execution during rollback/ });
      const validateCheckbox = screen.getByRole('checkbox', { name: /Perform validation checks before rollback/ });
      
      expect(backupCheckbox).toBeChecked();
      expect(stopStrategyCheckbox).toBeChecked();
      expect(validateCheckbox).toBeChecked();
    });
  });

  it('allows changing rollback settings', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to configure step
    await waitFor(() => {
      const configureButton = screen.getByText('Configure');
      fireEvent.click(configureButton);
    });

    await waitFor(() => {
      // Uncheck backup option
      const backupCheckbox = screen.getByRole('checkbox', { name: /Create backup before rollback/ });
      fireEvent.click(backupCheckbox);
      expect(backupCheckbox).not.toBeChecked();

      // Add rollback reason
      const reasonTextarea = screen.getByPlaceholderText('Describe why you\'re performing this rollback...');
      fireEvent.change(reasonTextarea, { target: { value: 'Performance regression in v3' } });
      expect(reasonTextarea).toHaveValue('Performance regression in v3');
    });
  });

  it('performs validation when validate button clicked', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate through steps
    await waitFor(() => {
      const configureButton = screen.getByText('Configure');
      fireEvent.click(configureButton);
    });

    await waitFor(() => {
      const validateButton = screen.getByText('Validate');
      fireEvent.click(validateButton);
    });

    await waitFor(() => {
      expect(mockStrategyService.validateRollback).toHaveBeenCalledWith(
        mockStrategyId,
        2,
        mockRollbackPlan
      );
    });
  });

  it('shows validation results with success status', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to validation step
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    
    await waitFor(() => {
      expect(screen.getByText('Validation Successful')).toBeInTheDocument();
      expect(screen.getByText('Rollback is safe to proceed.')).toBeInTheDocument();
    });
  });

  it('shows validation warnings when present', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to validation step
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    
    await waitFor(() => {
      expect(screen.getByText('This will revert recent parameter optimizations')).toBeInTheDocument();
    });
  });

  it('shows pre-rollback checks results', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to validation step
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    
    await waitFor(() => {
      expect(screen.getByText('Pre-rollback Checks')).toBeInTheDocument();
      expect(screen.getByText('Strategy is in safe state for rollback')).toBeInTheDocument();
    });
  });

  it('displays backup verification information', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to validation step
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    
    await waitFor(() => {
      expect(screen.getByText('Backup Status')).toBeInTheDocument();
      expect(screen.getByText('Yes')).toBeInTheDocument(); // Backup Created
      expect(screen.getByText('2.5 MB')).toBeInTheDocument(); // Backup Size
    });
  });

  it('executes rollback when execute button clicked', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate through all steps to execute
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    await waitFor(() => fireEvent.click(screen.getByText('Execute Rollback')));
    
    await waitFor(() => {
      expect(mockStrategyService.executeRollback).toHaveBeenCalledWith(
        mockStrategyId,
        2,
        expect.objectContaining({
          createBackup: true,
          stopStrategy: true,
          validateBeforeRollback: true,
          forceRollback: false,
          reason: ''
        }),
        expect.any(Function)
      );
    });
  });

  it('shows rollback progress during execution', async () => {
    // Mock progress callback
    mockStrategyService.executeRollback.mockImplementation(async (_, __, ___, callback) => {
      if (callback) {
        setTimeout(() => callback(mockProgress), 100);
      }
      return { success: true };
    });
    
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to execute
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    await waitFor(() => fireEvent.click(screen.getByText('Execute Rollback')));
    
    await waitFor(() => {
      expect(screen.getByText('Rollback in Progress')).toBeInTheDocument();
    });
  });

  it('shows completion screen on successful rollback', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate through all steps
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    await waitFor(() => fireEvent.click(screen.getByText('Execute Rollback')));
    
    // Wait for completion
    await waitFor(() => {
      expect(screen.getByText('Rollback Completed Successfully')).toBeInTheDocument();
      expect(defaultProps.onRollbackComplete).toHaveBeenCalledWith(true, 2);
    });
  });

  it('prevents proceeding when validation fails', async () => {
    const failedValidation: RollbackValidation = {
      validation_passed: false,
      validation_errors: ['Strategy is currently running'],
      warnings: [],
      pre_rollback_checks: [
        {
          check_name: 'strategy_state',
          description: 'Strategy must be stopped',
          passed: false
        }
      ]
    };
    
    mockStrategyService.validateRollback.mockResolvedValue(failedValidation);
    
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to validation
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    
    await waitFor(() => {
      expect(screen.getByText('Validation Failed')).toBeInTheDocument();
      expect(screen.getByText('Strategy is currently running')).toBeInTheDocument();
      
      // Execute button should be disabled
      const executeButton = screen.getByText('Execute Rollback');
      expect(executeButton.closest('button')).toBeDisabled();
    });
  });

  it('allows back navigation between steps', async () => {
    render(<RollbackManager {...defaultProps} />);
    
    // Go to configure step
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    
    await waitFor(() => {
      expect(screen.getByText('Rollback Configuration')).toBeInTheDocument();
      
      // Back button should be available
      const backButton = screen.getByText('Back');
      fireEvent.click(backButton);
    });

    await waitFor(() => {
      expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    });
  });

  it('closes modal when cancel button clicked', () => {
    render(<RollbackManager {...defaultProps} />);
    
    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);
    
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it('handles rollback execution errors gracefully', async () => {
    mockStrategyService.executeRollback.mockRejectedValue(new Error('Execution failed'));
    
    render(<RollbackManager {...defaultProps} />);
    
    // Navigate to execute
    await waitFor(() => fireEvent.click(screen.getByText('Configure')));
    await waitFor(() => fireEvent.click(screen.getByText('Validate')));
    await waitFor(() => fireEvent.click(screen.getByText('Execute Rollback')));
    
    await waitFor(() => {
      expect(defaultProps.onRollbackComplete).toHaveBeenCalledWith(false);
    });
  });

  it('shows loading state during plan generation', () => {
    mockStrategyService.generateRollbackPlan.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    
    render(<RollbackManager {...defaultProps} />);
    
    // Should show loading spinner
    expect(document.querySelector('.ant-spin')).toBeInTheDocument();
  });

  it('disables next button when plan is not loaded', async () => {
    mockStrategyService.generateRollbackPlan.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    
    render(<RollbackManager {...defaultProps} />);
    
    // Configure button should be disabled
    const configureButton = screen.getByText('Configure');
    expect(configureButton.closest('button')).toBeDisabled();
  });
});