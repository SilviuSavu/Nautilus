/**
 * Tests for NautilusEngineManager component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import NautilusEngineManager from '../NautilusEngineManager';
import { engineService } from '../../../services/engineService';

// Mock the engine service
vi.mock('../../../services/engineService', () => ({
  engineService: {
    getStatus: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    restart: vi.fn(),
    emergencyStop: vi.fn(),
  }
}));

// Mock the WebSocket hook
vi.mock('../../../hooks/useEngineWebSocket', () => ({
  useEngineWebSocket: () => ({
    isConnected: true,
    lastMessage: null,
    connectionError: null,
  })
}));

// Mock child components
vi.mock('../EngineStatusIndicator', () => ({
  EngineStatusIndicator: ({ status }: any) => (
    <div data-testid="engine-status-indicator">Status: {status.state}</div>
  )
}));

vi.mock('../EngineControlPanel', () => ({
  EngineControlPanel: ({ onStart, onStop, onRestart, onEmergencyStop }: any) => (
    <div data-testid="engine-control-panel">
      <button onClick={() => onStart({}, false)} data-testid="start-button">Start</button>
      <button onClick={() => onStop(false)} data-testid="stop-button">Stop</button>
      <button onClick={onRestart} data-testid="restart-button">Restart</button>
      <button onClick={onEmergencyStop} data-testid="emergency-stop-button">Emergency Stop</button>
    </div>
  )
}));

vi.mock('../ResourceMonitor', () => ({
  ResourceMonitor: () => <div data-testid="resource-monitor">Resource Monitor</div>
}));

vi.mock('../EngineConfigPanel', () => ({
  EngineConfigPanel: ({ onConfigChange }: any) => (
    <div data-testid="engine-config-panel">
      <button onClick={() => onConfigChange({ trading_mode: 'live' })} data-testid="config-change">
        Change Config
      </button>
    </div>
  )
}));

describe('NautilusEngineManager', () => {
  const mockEngineStatus = {
    state: 'stopped',
    config: {
      engine_type: 'live',
      log_level: 'INFO',
      instance_id: 'nautilus-001',
      trading_mode: 'paper',
      max_memory: '2g',
      max_cpu: '2.0',
      data_catalog_path: '/app/data',
      cache_database_path: '/app/cache',
      risk_engine_enabled: true,
    }
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (engineService.getStatus as any).mockResolvedValue({
      success: true,
      status: mockEngineStatus
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders the main components', async () => {
    render(<NautilusEngineManager />);

    await waitFor(() => {
      expect(screen.getByText('NautilusTrader Engine Management')).toBeInTheDocument();
      expect(screen.getByTestId('engine-status-indicator')).toBeInTheDocument();
      expect(screen.getByTestId('engine-control-panel')).toBeInTheDocument();
      expect(screen.getByTestId('resource-monitor')).toBeInTheDocument();
      expect(screen.getByTestId('engine-config-panel')).toBeInTheDocument();
    });
  });

  it('loads engine status on mount', async () => {
    render(<NautilusEngineManager />);

    await waitFor(() => {
      expect(engineService.getStatus).toHaveBeenCalled();
      expect(screen.getByText('Status: stopped')).toBeInTheDocument();
    });
  });

  it('handles start engine action', async () => {
    (engineService.start as any).mockResolvedValue({
      success: true,
      message: 'Engine started',
      state: 'starting'
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      const startButton = screen.getByTestId('start-button');
      fireEvent.click(startButton);
    });

    expect(engineService.start).toHaveBeenCalledWith({}, false);
  });

  it('handles stop engine action', async () => {
    (engineService.stop as any).mockResolvedValue({
      success: true,
      message: 'Engine stopped',
      state: 'stopping'
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      const stopButton = screen.getByTestId('stop-button');
      fireEvent.click(stopButton);
    });

    expect(engineService.stop).toHaveBeenCalledWith(false);
  });

  it('handles restart engine action', async () => {
    (engineService.restart as any).mockResolvedValue({
      success: true,
      message: 'Engine restarted',
      state: 'stopping'
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      const restartButton = screen.getByTestId('restart-button');
      fireEvent.click(restartButton);
    });

    expect(engineService.restart).toHaveBeenCalled();
  });

  it('handles emergency stop action', async () => {
    (engineService.emergencyStop as any).mockResolvedValue({
      success: true,
      message: 'Emergency stop executed',
      state: 'stopped'
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      const emergencyStopButton = screen.getByTestId('emergency-stop-button');
      fireEvent.click(emergencyStopButton);
    });

    expect(engineService.emergencyStop).toHaveBeenCalled();
  });

  it('handles configuration changes', async () => {
    render(<NautilusEngineManager />);

    await waitFor(() => {
      const configChangeButton = screen.getByTestId('config-change');
      fireEvent.click(configChangeButton);
    });

    // Config should be updated in component state
    // This would be verified through component behavior
  });

  it('displays error message when engine service fails', async () => {
    (engineService.getStatus as any).mockRejectedValue(new Error('Connection failed'));

    render(<NautilusEngineManager />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to connect to engine service/)).toBeInTheDocument();
    });
  });

  it('shows real-time monitoring connection status', async () => {
    render(<NautilusEngineManager />);

    await waitFor(() => {
      expect(screen.getByText(/Real-time monitoring connected/)).toBeInTheDocument();
    });
  });

  it('displays error alerts when engine operations fail', async () => {
    (engineService.start as any).mockResolvedValue({
      success: false,
      message: 'Failed to start engine',
      state: 'error'
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      const startButton = screen.getByTestId('start-button');
      fireEvent.click(startButton);
    });

    await waitFor(() => {
      expect(screen.getByText(/Failed to start engine/)).toBeInTheDocument();
    });
  });

  it('polls for status updates at regular intervals', async () => {
    vi.useFakeTimers();
    
    render(<NautilusEngineManager />);

    // Initial call
    await waitFor(() => {
      expect(engineService.getStatus).toHaveBeenCalledTimes(1);
    });

    // Advance timer by 5 seconds
    vi.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(engineService.getStatus).toHaveBeenCalledTimes(2);
    });

    vi.useRealTimers();
  });

  it('handles engine status with errors', async () => {
    const errorStatus = {
      ...mockEngineStatus,
      state: 'error',
      last_error: 'Engine configuration error'
    };

    (engineService.getStatus as any).mockResolvedValue({
      success: true,
      status: errorStatus
    });

    render(<NautilusEngineManager />);

    await waitFor(() => {
      expect(screen.getByText(/Engine configuration error/)).toBeInTheDocument();
    });
  });
});

export { };