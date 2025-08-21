import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AlertSystem } from '../AlertSystem';

// Mock fetch
global.fetch = vi.fn();
const mockFetch = fetch as vi.MockedFunction<typeof fetch>;

// Mock notification
const mockNotification = {
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn()
};
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: mockNotification
  };
});

const mockStrategies = [
  {
    id: 'strategy-1',
    config_id: 'config-1',
    nautilus_strategy_id: 'nautilus-1',
    deployment_id: 'deploy-1',
    state: 'running' as const,
    performance_metrics: {
      total_pnl: '1250.50',
      unrealized_pnl: '125.25',
      total_trades: 45,
      winning_trades: 28,
      win_rate: 0.622,
      max_drawdown: '8.5',
      sharpe_ratio: 1.85,
      last_updated: new Date()
    },
    runtime_info: {
      orders_placed: 45,
      positions_opened: 12,
      uptime_seconds: 86400
    },
    error_log: [],
    started_at: new Date()
  }
];

const mockAlerts = [
  {
    id: 'alert-1',
    name: 'High Drawdown Alert',
    strategy_id: 'strategy-1',
    alert_type: 'drawdown_limit',
    condition: 'above',
    threshold_value: 10,
    current_value: 8.5,
    is_active: true,
    notification_methods: ['dashboard', 'email'],
    email_addresses: ['trader@example.com'],
    phone_numbers: [],
    trigger_count: 2,
    created_at: new Date(),
    updated_at: new Date()
  }
];

const mockTriggers = [
  {
    id: 'trigger-1',
    alert_id: 'alert-1',
    strategy_id: 'strategy-1',
    triggered_at: new Date(),
    value_at_trigger: 11.2,
    threshold_value: 10,
    alert_type: 'drawdown_limit',
    resolved: false,
    notification_sent: true,
    acknowledgement_required: true
  }
];

describe('AlertSystem', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    mockNotification.success.mockClear();
    mockNotification.error.mockClear();
    mockNotification.warning.mockClear();
    
    // Default successful responses
    mockFetch.mockImplementation((url) => {
      if (url.includes('/api/v1/performance/alerts/triggers')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ triggers: mockTriggers })
        } as Response);
      }
      
      if (url.includes('/api/v1/performance/alerts/check')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ triggered_alerts: [] })
        } as Response);
      }
      
      if (url.includes('/api/v1/performance/alerts')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ alerts: mockAlerts })
        } as Response);
      }
      
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({})
      } as Response);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders alert system with title', () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    expect(screen.getByText('Performance Alert System')).toBeInTheDocument();
    expect(screen.getByText('Configure alerts for performance thresholds and risk limits')).toBeInTheDocument();
  });

  it('loads and displays alerts', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      expect(screen.getByText('High Drawdown Alert')).toBeInTheDocument();
    });
  });

  it('shows create alert button', () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    expect(screen.getByText('Create Alert')).toBeInTheDocument();
  });

  it('opens create alert modal when button is clicked', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    const createButton = screen.getByText('Create Alert');
    fireEvent.click(createButton);
    
    await waitFor(() => {
      expect(screen.getByText('Create Performance Alert')).toBeInTheDocument();
    });
  });

  it('displays active alerts summary', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      expect(screen.getByText(/1 active alerts monitoring 1 strategies/)).toBeInTheDocument();
    });
  });

  it('shows alert configuration table', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      expect(screen.getByText('Alert Configuration')).toBeInTheDocument();
      expect(screen.getByText('High Drawdown Alert')).toBeInTheDocument();
      expect(screen.getByText('DRAWDOWN_LIMIT')).toBeInTheDocument();
    });
  });

  it('displays recent alert triggers', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      expect(screen.getByText('Recent Alert Triggers')).toBeInTheDocument();
      expect(screen.getByText('DRAWDOWN_LIMIT')).toBeInTheDocument();
    });
  });

  it('allows toggling alert active status', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      const toggleSwitch = screen.getByRole('switch');
      expect(toggleSwitch).toBeChecked();
    });
    
    const toggleSwitch = screen.getByRole('switch');
    fireEvent.click(toggleSwitch);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/performance/alerts/alert-1/toggle'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ is_active: false })
        })
      );
    });
  });

  it('creates new alert when form is submitted', async () => {
    mockFetch.mockImplementation((url, options) => {
      if (url.includes('/api/v1/performance/alerts') && options?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ id: 'new-alert' })
        } as Response);
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ alerts: mockAlerts })
      } as Response);
    });
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    // Open create modal
    fireEvent.click(screen.getByText('Create Alert'));
    
    await waitFor(() => {
      expect(screen.getByText('Create Performance Alert')).toBeInTheDocument();
    });
    
    // Fill form
    fireEvent.change(screen.getByPlaceholderText('e.g., High Drawdown Alert'), {
      target: { value: 'Test Alert' }
    });
    
    fireEvent.click(screen.getByText('Select strategy'));
    fireEvent.click(screen.getByText('strategy-1'));
    
    fireEvent.click(screen.getByText('Select alert type'));
    fireEvent.click(screen.getByText('P&L Threshold'));
    
    fireEvent.click(screen.getByText('Select condition'));
    fireEvent.click(screen.getByText('Below'));
    
    fireEvent.change(screen.getByPlaceholderText('Enter threshold'), {
      target: { value: '1000' }
    });
    
    // Submit form
    fireEvent.click(screen.getByText('Create Alert'));
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/performance/alerts',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: expect.stringContaining('Test Alert')
        })
      );
    });
    
    expect(mockNotification.success).toHaveBeenCalledWith({
      message: 'Alert Created',
      description: 'Performance alert has been created successfully',
      duration: 3
    });
  });

  it('acknowledges alerts when acknowledge button is clicked', async () => {
    mockFetch.mockImplementation((url, options) => {
      if (url.includes('/acknowledge') && options?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({})
        } as Response);
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ 
          alerts: mockAlerts,
          triggers: mockTriggers 
        })
      } as Response);
    });
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      const acknowledgeButton = screen.getByText('Acknowledge');
      expect(acknowledgeButton).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Acknowledge'));
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/performance/alerts/triggers/trigger-1/acknowledge'),
        expect.objectContaining({ method: 'POST' })
      );
    });
    
    expect(mockNotification.success).toHaveBeenCalledWith({
      message: 'Alert Acknowledged',
      description: 'Alert has been acknowledged successfully',
      duration: 3
    });
  });

  it('deletes alert when delete button is clicked', async () => {
    mockFetch.mockImplementation((url, options) => {
      if (url.includes('/alert-1') && options?.method === 'DELETE') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({})
        } as Response);
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ alerts: mockAlerts })
      } as Response);
    });
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      const deleteButton = screen.getByLabelText('delete');
      expect(deleteButton).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByLabelText('delete'));
    
    // Confirm deletion in modal
    await waitFor(() => {
      expect(screen.getByText('Delete Alert')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Delete'));
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/performance/alerts/alert-1',
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  it('shows correct alert type colors and icons', async () => {
    const multipleAlerts = [
      ...mockAlerts,
      {
        id: 'alert-2',
        name: 'P&L Alert',
        strategy_id: 'strategy-1',
        alert_type: 'pnl_threshold',
        condition: 'below',
        threshold_value: 1000,
        current_value: 1250,
        is_active: true,
        notification_methods: ['dashboard'],
        email_addresses: [],
        phone_numbers: [],
        trigger_count: 0,
        created_at: new Date(),
        updated_at: new Date()
      }
    ];
    
    mockFetch.mockImplementation(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ alerts: multipleAlerts })
    } as Response));
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      expect(screen.getByText('💰 PNL_THRESHOLD')).toBeInTheDocument();
      expect(screen.getByText('📉 DRAWDOWN_LIMIT')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    mockFetch.mockImplementation(() => Promise.resolve({
      ok: false,
      json: () => Promise.resolve({})
    } as Response));
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    // Should not crash when API fails
    expect(screen.getByText('Performance Alert System')).toBeInTheDocument();
  });

  it('checks for active alerts periodically', async () => {
    vi.useFakeTimers();
    
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    // Initial loads
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
    
    // Fast forward 30 seconds (check interval)
    vi.advanceTimersByTime(30000);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/v1/performance/alerts/check');
    });
    
    vi.useRealTimers();
  });

  it('shows notification methods correctly', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    await waitFor(() => {
      // Should show email and dashboard icons for the alert
      expect(screen.getByLabelText('mail')).toBeInTheDocument();
      expect(screen.getByLabelText('bell')).toBeInTheDocument();
    });
  });

  it('validates form inputs', async () => {
    render(<AlertSystem strategies={mockStrategies} performanceData={null} />);
    
    // Open create modal
    fireEvent.click(screen.getByText('Create Alert'));
    
    await waitFor(() => {
      expect(screen.getByText('Create Performance Alert')).toBeInTheDocument();
    });
    
    // Try to submit without required fields
    fireEvent.click(screen.getByText('Create Alert'));
    
    await waitFor(() => {
      expect(screen.getByText('Please enter alert name')).toBeInTheDocument();
      expect(screen.getByText('Please select strategy')).toBeInTheDocument();
    });
  });
});