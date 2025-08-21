import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { VersionControl } from '../VersionControl';
import { strategyService } from '../services/strategyService';
import { StrategyVersion, DeploymentResult } from '../types/strategyTypes';

// Mock the strategy service
vi.mock('../services/strategyService');

const mockStrategyService = vi.mocked(strategyService);

describe('VersionControl', () => {
  const mockStrategyId = 'test-strategy-123';
  const currentVersion = 3;
  
  const mockVersions: StrategyVersion[] = [
    {
      id: 'version-3',
      config_id: mockStrategyId,
      version_number: 3,
      config_snapshot: {
        id: mockStrategyId,
        name: 'Test Strategy v3',
        template_id: 'template-1',
        parameters: { fast_period: 10, slow_period: 20 },
        risk_settings: {
          max_position_size: '10000',
          position_sizing_method: 'fixed' as const
        },
        deployment_settings: {
          mode: 'live' as const,
          venue: 'IB'
        },
        user_id: 'user-1',
        version: 3,
        status: 'deployed' as const,
        created_at: new Date(),
        updated_at: new Date(),
        tags: []
      },
      change_summary: 'Updated fast period parameter',
      created_by: 'trader1',
      created_at: new Date('2024-01-03'),
      deployment_results: [
        {
          deployment_id: 'deploy-3',
          start_time: new Date('2024-01-03'),
          end_time: new Date('2024-01-04'),
          final_pnl: { toNumber: () => 150.50, toFixed: (d: number) => '150.50' },
          trade_count: 12,
          success: true,
          notes: 'Successful deployment with good performance'
        }
      ]
    },
    {
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
      created_at: new Date('2024-01-02'),
      deployment_results: [
        {
          deployment_id: 'deploy-2',
          start_time: new Date('2024-01-02'),
          end_time: new Date('2024-01-02'),
          final_pnl: { toNumber: () => -25.00, toFixed: (d: number) => '-25.00' },
          trade_count: 5,
          success: false,
          notes: 'Deployment failed due to configuration error'
        }
      ]
    },
    {
      id: 'version-1',
      config_id: mockStrategyId,
      version_number: 1,
      config_snapshot: {
        id: mockStrategyId,
        name: 'Test Strategy v1',
        template_id: 'template-1',
        parameters: { fast_period: 5, slow_period: 15 },
        risk_settings: {
          max_position_size: '5000',
          position_sizing_method: 'percentage' as const
        },
        deployment_settings: {
          mode: 'backtest' as const,
          venue: 'SIM'
        },
        user_id: 'user-1',
        version: 1,
        status: 'draft' as const,
        created_at: new Date(),
        updated_at: new Date(),
        tags: []
      },
      change_summary: 'Initial strategy configuration',
      created_by: 'trader1',
      created_at: new Date('2024-01-01')
    }
  ];

  const defaultProps = {
    strategyId: mockStrategyId,
    currentVersion,
    visible: true,
    onClose: vi.fn(),
    onVersionSelect: vi.fn(),
    onRollback: vi.fn()
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockStrategyService.getVersionHistory.mockResolvedValue({ versions: mockVersions });
    mockStrategyService.rollbackToVersion.mockResolvedValue({ success: true, new_version: 2 });
    mockStrategyService.compareVersions.mockResolvedValue({
      version1: mockVersions[0],
      version2: mockVersions[1],
      differences: [],
      configuration_diff: {
        parameter_changes: [],
        total_changes: 0,
        high_impact_changes: 0,
        medium_impact_changes: 0,
        low_impact_changes: 0
      }
    });
  });

  it('renders version control modal when visible', async () => {
    render(<VersionControl {...defaultProps} />);
    
    expect(screen.getByText('Strategy Version Control')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(mockStrategyService.getVersionHistory).toHaveBeenCalledWith(mockStrategyId);
    });
  });

  it('displays version history with correct information', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Check if versions are displayed
      expect(screen.getByText('Version 3')).toBeInTheDocument();
      expect(screen.getByText('Version 2')).toBeInTheDocument();
      expect(screen.getByText('Version 1')).toBeInTheDocument();
    });

    // Check current version tag
    expect(screen.getByText('Current')).toBeInTheDocument();

    // Check change summaries
    expect(screen.getByText('Updated fast period parameter')).toBeInTheDocument();
    expect(screen.getByText('Reduced position size for risk management')).toBeInTheDocument();
  });

  it('shows version status indicators correctly', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Current version should be blue
      const currentTag = screen.getByText('Current');
      expect(currentTag).toBeInTheDocument();
      
      // Failed deployment should show appropriate status
      const failedItems = screen.getAllByText('Failed');
      expect(failedItems.length).toBeGreaterThan(0);
    });
  });

  it('allows version selection for comparison', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Get compare buttons
      const compareButtons = screen.getAllByTitle('Compare');
      expect(compareButtons.length).toBe(3); // One for each version
    });

    // Click compare button for version 2
    const compareButtons = screen.getAllByTitle('Compare');
    fireEvent.click(compareButtons[1]);

    // Should enable version selection
    await waitFor(() => {
      // Button should become primary (selected state)
      expect(compareButtons[1]).toHaveClass('ant-btn-primary');
    });
  });

  it('enables compare button when exactly 2 versions selected', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      const compareButtons = screen.getAllByTitle('Compare');
      
      // Select first version
      fireEvent.click(compareButtons[0]);
      
      // Compare button in footer should still be disabled (only 1 selected)
      const footerCompareButton = screen.getByText(/Compare Selected \(1\/2\)/);
      expect(footerCompareButton.closest('button')).toBeDisabled();
      
      // Select second version
      fireEvent.click(compareButtons[1]);
      
      // Now compare button should be enabled
      const enabledCompareButton = screen.getByText(/Compare Selected \(2\/2\)/);
      expect(enabledCompareButton.closest('button')).not.toBeDisabled();
    });
  });

  it('performs version comparison when compare button clicked', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      const compareButtons = screen.getAllByTitle('Compare');
      
      // Select two versions
      fireEvent.click(compareButtons[0]);
      fireEvent.click(compareButtons[1]);
      
      // Click footer compare button
      const footerCompareButton = screen.getByText(/Compare Selected \(2\/2\)/);
      fireEvent.click(footerCompareButton);
    });

    await waitFor(() => {
      expect(mockStrategyService.compareVersions).toHaveBeenCalledWith(
        mockStrategyId,
        'version-3',
        'version-2'
      );
    });
  });

  it('handles rollback confirmation and execution', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Find rollback button for version 2 (not current version)
      const rollbackButtons = screen.getAllByTitle('Rollback');
      expect(rollbackButtons.length).toBe(2); // Version 2 and 1 should have rollback buttons
    });

    // Click first rollback button
    const rollbackButtons = screen.getAllByTitle('Rollback');
    fireEvent.click(rollbackButtons[0]);

    // Popconfirm should appear
    await waitFor(() => {
      expect(screen.getByText('Rollback to this version?')).toBeInTheDocument();
    });

    // Confirm rollback
    const confirmButton = screen.getByText('Rollback');
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(mockStrategyService.rollbackToVersion).toHaveBeenCalledWith(
        mockStrategyId,
        'version-2'
      );
      expect(defaultProps.onRollback).toHaveBeenCalledWith('version-2');
    });
  });

  it('shows version details in modal when view button clicked', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      const viewButtons = screen.getAllByTitle('View Details');
      fireEvent.click(viewButtons[0]);
    });

    await waitFor(() => {
      expect(screen.getByText('Version 3 Details')).toBeInTheDocument();
      expect(screen.getByText('Updated fast period parameter')).toBeInTheDocument();
      expect(screen.getByText('trader1')).toBeInTheDocument();
    });
  });

  it('displays deployment history for versions with deployment results', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      const viewButtons = screen.getAllByTitle('View Details');
      fireEvent.click(viewButtons[0]);
    });

    await waitFor(() => {
      expect(screen.getByText('Deployment History')).toBeInTheDocument();
      expect(screen.getByText('Successful Deployment')).toBeInTheDocument();
      expect(screen.getByText('P&L: $150.50')).toBeInTheDocument();
      expect(screen.getByText('Trades: 12')).toBeInTheDocument();
    });
  });

  it('handles empty version history gracefully', async () => {
    mockStrategyService.getVersionHistory.mockResolvedValue({ versions: [] });
    
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('No Version History')).toBeInTheDocument();
      expect(screen.getByText(/This strategy has no saved versions yet/)).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    mockStrategyService.getVersionHistory.mockRejectedValue(new Error('API Error'));
    
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        'Failed to load version history:',
        expect.any(Error)
      );
    });

    consoleSpy.mockRestore();
  });

  it('shows performance metrics in version list', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Should show P&L and trade count
      expect(screen.getByText(/P&L: \$150\.50, Trades: 12/)).toBeInTheDocument();
      expect(screen.getByText(/P&L: \$-25\.00, Trades: 5/)).toBeInTheDocument();
      expect(screen.getByText('No deployment data')).toBeInTheDocument(); // Version 1
    });
  });

  it('closes modal when close button clicked', async () => {
    render(<VersionControl {...defaultProps} />);
    
    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);
    
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it('does not show rollback button for current version', async () => {
    render(<VersionControl {...defaultProps} />);
    
    await waitFor(() => {
      // Version 3 (current) should not have a rollback button
      const versionItems = screen.getAllByText(/Version \d+/);
      
      // Check that current version row doesn't have rollback button
      const currentVersionRow = versionItems[0].closest('.ant-list-item');
      expect(currentVersionRow).not.toContainHTML('Rollback');
    });
  });

  it('shows loading state while fetching versions', () => {
    // Mock a pending promise
    mockStrategyService.getVersionHistory.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    
    render(<VersionControl {...defaultProps} />);
    
    expect(screen.getByText('Strategy Version Control')).toBeInTheDocument();
    // Loading spinner should be present
    expect(document.querySelector('.ant-spin')).toBeInTheDocument();
  });
});