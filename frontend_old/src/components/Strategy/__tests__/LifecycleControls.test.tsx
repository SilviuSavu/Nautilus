import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { LifecycleControls } from '../LifecycleControls';
import { StrategyConfig, StrategyInstance } from '../types/strategyTypes';

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock strategyService
vi.mock('../services/strategyService', () => ({
  default: {
    deployStrategy: vi.fn(),
    controlStrategy: vi.fn(),
    getStrategyStatus: vi.fn(),
    deleteConfiguration: vi.fn()
  }
}));

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn()
    }
  };
});

describe('LifecycleControls', () => {
  const mockStrategy: StrategyConfig = {
    id: 'test-strategy-1',
    name: 'Test Strategy',
    template_id: 'moving_average_cross',
    user_id: 'user-1',
    parameters: {
      fast_period: 10,
      slow_period: 20
    },
    risk_settings: {
      max_position_size: 1000,
      position_sizing_method: 'fixed'
    },
    deployment_settings: {
      mode: 'paper',
      venue: 'IB'
    },
    version: 1,
    status: 'draft',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    tags: []
  };

  const mockInstance: StrategyInstance = {
    id: 'instance-1',
    config_id: 'test-strategy-1',
    nautilus_strategy_id: 'nautilus-1',
    deployment_id: 'deploy-1',
    state: 'running',
    performance_metrics: {
      total_pnl: 150.25,
      unrealized_pnl: 25.50,
      total_trades: 10,
      winning_trades: 7,
      win_rate: 0.7,
      max_drawdown: -50.00,
      last_updated: new Date().toISOString()
    },
    runtime_info: {
      orders_placed: 15,
      positions_opened: 8,
      uptime_seconds: 3600
    },
    error_log: [],
    started_at: new Date().toISOString()
  };

  const mockOnInstanceUpdate = vi.fn();
  const mockOnStrategyUpdate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders deployment button when no instance is provided', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('Deploy Strategy')).toBeInTheDocument();
    expect(screen.getByText('Strategy not deployed')).toBeInTheDocument();
  });

  it('shows deployment modal when deploy button is clicked', async () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    fireEvent.click(screen.getByText('Deploy Strategy'));

    await waitFor(() => {
      expect(screen.getByText('Deploy Strategy')).toBeInTheDocument();
      expect(screen.getByText('Deployment Mode')).toBeInTheDocument();
    });
  });

  it('renders control buttons when instance is provided', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={mockInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    // Should show pause button for running strategy
    expect(screen.getByText('Pause')).toBeInTheDocument();
    expect(screen.getByText('Stop')).toBeInTheDocument();
    expect(screen.getByText('Restart')).toBeInTheDocument();
  });

  it('displays strategy status correctly', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={mockInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('$150.25')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('70.0%')).toBeInTheDocument();
  });

  it('shows performance metrics', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={mockInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('Trades: 10')).toBeInTheDocument();
    expect(screen.getByText('Orders: 15')).toBeInTheDocument();
    expect(screen.getByText('Positions: 8')).toBeInTheDocument();
  });

  it('renders different control buttons based on strategy state', () => {
    const stoppedInstance = { ...mockInstance, state: 'stopped' as const };
    
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={stoppedInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    // Should show start button for stopped strategy
    expect(screen.getByText('Start')).toBeInTheDocument();
    expect(screen.queryByText('Pause')).not.toBeInTheDocument();
  });

  it('renders pause and resume buttons correctly', () => {
    const pausedInstance = { ...mockInstance, state: 'paused' as const };
    
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={pausedInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    // Should show resume button for paused strategy
    expect(screen.getByText('Resume')).toBeInTheDocument();
    expect(screen.queryByText('Pause')).not.toBeInTheDocument();
  });

  it('shows error log when present', () => {
    const instanceWithErrors = {
      ...mockInstance,
      error_log: [
        {
          timestamp: new Date().toISOString(),
          message: 'Connection timeout',
          level: 'error' as const
        }
      ]
    };

    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={instanceWithErrors}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('Recent Errors')).toBeInTheDocument();
    expect(screen.getByText('Connection timeout')).toBeInTheDocument();
  });

  it('displays deployment form fields', async () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    fireEvent.click(screen.getByText('Deploy Strategy'));

    await waitFor(() => {
      expect(screen.getByText('Deployment Mode')).toBeInTheDocument();
      expect(screen.getByText('Auto Start')).toBeInTheDocument();
      expect(screen.getByText('Risk Check')).toBeInTheDocument();
      expect(screen.getByText('Paper Trading (Simulated)')).toBeInTheDocument();
    });
  });

  it('shows strategy lifecycle title and description', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('Strategy Lifecycle')).toBeInTheDocument();
    expect(screen.getByText('Deploy and manage strategy execution')).toBeInTheDocument();
  });

  it('renders settings dropdown menu', () => {
    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={mockInstance}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    // The settings button should be present (last button in the controls)
    const settingsButtons = screen.getAllByRole('button');
    const settingsButton = settingsButtons[settingsButtons.length - 1];
    expect(settingsButton).toBeInTheDocument();
  });

  it('handles uptime display correctly', () => {
    const instanceWithLongUptime = {
      ...mockInstance,
      runtime_info: {
        ...mockInstance.runtime_info,
        uptime_seconds: 7200 // 2 hours
      }
    };

    render(
      <LifecycleControls
        strategy={mockStrategy}
        instance={instanceWithLongUptime}
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(screen.getByText('2')).toBeInTheDocument(); // 2 hours uptime
    expect(screen.getByText('hrs')).toBeInTheDocument();
  });

  it('applies custom className when provided', () => {
    const { container } = render(
      <LifecycleControls
        strategy={mockStrategy}
        className="custom-lifecycle-controls"
        onInstanceUpdate={mockOnInstanceUpdate}
        onStrategyUpdate={mockOnStrategyUpdate}
      />
    );

    expect(container.firstChild).toHaveClass('lifecycle-controls');
    expect(container.firstChild).toHaveClass('custom-lifecycle-controls');
  });
});