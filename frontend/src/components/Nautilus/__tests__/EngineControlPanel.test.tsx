/**
 * Tests for EngineControlPanel component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import EngineControlPanel from '../EngineControlPanel';

const mockEngineStatus = {
  state: 'stopped' as const,
  mode: 'paper' as const,
};

const mockConfig = {
  engine_type: 'live',
  log_level: 'INFO',
  instance_id: 'nautilus-001',
  trading_mode: 'paper' as const,
  max_memory: '2g',
  max_cpu: '2.0',
  risk_engine_enabled: true,
  max_position_size: 100000,
  max_order_rate: 100
};

const mockProps = {
  status: mockEngineStatus,
  config: mockConfig,
  onStart: vi.fn(),
  onStop: vi.fn(),
  onRestart: vi.fn(),
  onEmergencyStop: vi.fn(),
  loading: false
};

describe('EngineControlPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all control buttons', () => {
    render(<EngineControlPanel {...mockProps} />);

    expect(screen.getByRole('button', { name: /start engine/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /stop engine/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /restart/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /force stop/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /emergency stop/i })).toBeInTheDocument();
  });

  it('enables start button when engine is stopped', () => {
    render(<EngineControlPanel {...mockProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    expect(startButton).not.toBeDisabled();
  });

  it('disables start button when engine is running', () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    expect(startButton).toBeDisabled();
  });

  it('calls onStart when start button is clicked with paper trading', () => {
    render(<EngineControlPanel {...mockProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    fireEvent.click(startButton);

    expect(mockProps.onStart).toHaveBeenCalledWith(mockConfig, false);
  });

  it('shows live trading confirmation modal for live trading mode', () => {
    const liveConfig = { ...mockConfig, trading_mode: 'live' as const };
    const liveProps = { ...mockProps, config: liveConfig };

    render(<EngineControlPanel {...liveProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    fireEvent.click(startButton);

    expect(screen.getByText(/LIVE TRADING MODE CONFIRMATION/i)).toBeInTheDocument();
    expect(screen.getByText(/Real money trading/i)).toBeInTheDocument();
  });

  it('confirms live trading start when user accepts', async () => {
    const liveConfig = { ...mockConfig, trading_mode: 'live' as const };
    const liveProps = { ...mockProps, config: liveConfig };

    render(<EngineControlPanel {...liveProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    fireEvent.click(startButton);

    const confirmButton = screen.getByRole('button', { name: /START LIVE TRADING/i });
    fireEvent.click(confirmButton);

    expect(mockProps.onStart).toHaveBeenCalledWith(liveConfig, true);
  });

  it('shows stop confirmation modal when stop button is clicked', () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const stopButton = screen.getByRole('button', { name: /stop engine/i });
    fireEvent.click(stopButton);

    expect(screen.getByText(/Stop NautilusTrader Engine/i)).toBeInTheDocument();
  });

  it('shows force stop confirmation with warning', () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const forceStopButton = screen.getByRole('button', { name: /force stop/i });
    fireEvent.click(forceStopButton);

    expect(screen.getByText(/FORCE STOP ENGINE/i)).toBeInTheDocument();
    expect(screen.getByText(/immediately terminate/i)).toBeInTheDocument();
  });

  it('calls onStop when stop is confirmed', async () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const stopButton = screen.getByRole('button', { name: /stop engine/i });
    fireEvent.click(stopButton);

    const confirmButton = screen.getByRole('button', { name: /stop engine/i });
    fireEvent.click(confirmButton);

    expect(mockProps.onStop).toHaveBeenCalledWith(false);
  });

  it('calls onRestart when restart button is clicked', () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const restartButton = screen.getByRole('button', { name: /restart/i });
    fireEvent.click(restartButton);

    expect(mockProps.onRestart).toHaveBeenCalled();
  });

  it('shows emergency stop confirmation', () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const emergencyButton = screen.getByRole('button', { name: /emergency stop/i });
    fireEvent.click(emergencyButton);

    expect(screen.getByText(/Emergency Stop/i)).toBeInTheDocument();
  });

  it('calls onEmergencyStop when emergency stop is confirmed', async () => {
    const runningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const }
    };

    render(<EngineControlPanel {...runningProps} />);

    const emergencyButton = screen.getByRole('button', { name: /emergency stop/i });
    fireEvent.click(emergencyButton);

    const confirmButton = screen.getByRole('button', { name: /emergency stop/i });
    fireEvent.click(confirmButton);

    expect(mockProps.onEmergencyStop).toHaveBeenCalled();
  });

  it('disables buttons when loading', () => {
    const loadingProps = { ...mockProps, loading: true };

    render(<EngineControlPanel {...loadingProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    const stopButton = screen.getByRole('button', { name: /stop engine/i });

    expect(startButton).toBeDisabled();
    expect(stopButton).toBeDisabled();
  });

  it('shows transitioning state alert', () => {
    const transitioningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'starting' as const }
    };

    render(<EngineControlPanel {...transitioningProps} />);

    expect(screen.getByText(/Engine starting.../i)).toBeInTheDocument();
  });

  it('shows live trading warning when engine is running in live mode', () => {
    const liveRunningProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'running' as const, mode: 'live' as const }
    };

    render(<EngineControlPanel {...liveRunningProps} />);

    expect(screen.getByText(/Live Trading Active/i)).toBeInTheDocument();
  });

  it('shows error alert when engine has error state', () => {
    const errorProps = {
      ...mockProps,
      status: { ...mockEngineStatus, state: 'error' as const }
    };

    render(<EngineControlPanel {...errorProps} />);

    expect(screen.getByText(/Engine Error/i)).toBeInTheDocument();
  });

  it('displays risk configuration in live trading confirmation', () => {
    const liveConfig = { 
      ...mockConfig, 
      trading_mode: 'live' as const,
      risk_engine_enabled: true,
      max_position_size: 500000
    };
    const liveProps = { ...mockProps, config: liveConfig };

    render(<EngineControlPanel {...liveProps} />);

    const startButton = screen.getByRole('button', { name: /start engine/i });
    fireEvent.click(startButton);

    expect(screen.getByText(/Risk Engine.*Enabled/)).toBeInTheDocument();
    expect(screen.getByText(/Max Position.*500,000/)).toBeInTheDocument();
  });
});

export { };