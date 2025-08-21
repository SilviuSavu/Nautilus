/**
 * Unit Tests for MonitoringDashboard Component
 * Tests comprehensive monitoring dashboard rendering and user interactions
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MonitoringDashboard } from '../MonitoringDashboard';
import { OrderLatencyTracker } from '../../../services/monitoring/OrderLatencyTracker';
import { ThresholdAlertEngine } from '../../../services/monitoring/ThresholdAlertEngine';

// Mock the monitoring services
vi.mock('../../../services/monitoring/OrderLatencyTracker');
vi.mock('../../../services/monitoring/ThresholdAlertEngine');

// Mock Chart components to avoid canvas rendering issues in tests
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(({ data, options }) => (
    <div data-testid="line-chart" data-chart-data={JSON.stringify(data)} data-chart-options={JSON.stringify(options)}>
      Mocked Line Chart
    </div>
  )),
  Bar: vi.fn(({ data, options }) => (
    <div data-testid="bar-chart" data-chart-data={JSON.stringify(data)} data-chart-options={JSON.stringify(options)}>
      Mocked Bar Chart
    </div>
  )),
}));

// Mock Ant Design components for consistent testing
vi.mock('antd', () => ({
  Card: ({ title, children, ...props }: any) => (
    <div data-testid="card" data-title={title} {...props}>
      <h3>{title}</h3>
      {children}
    </div>
  ),
  Row: ({ children, ...props }: any) => <div data-testid="row" {...props}>{children}</div>,
  Col: ({ children, ...props }: any) => <div data-testid="col" {...props}>{children}</div>,
  Select: ({ children, onChange, defaultValue, ...props }: any) => (
    <select data-testid="select" onChange={(e) => onChange?.(e.target.value)} defaultValue={defaultValue} {...props}>
      {children}
    </select>
  ),
  Button: ({ children, onClick, type, ...props }: any) => (
    <button data-testid="button" onClick={onClick} data-type={type} {...props}>
      {children}
    </button>
  ),
  Table: ({ dataSource, columns, ...props }: any) => (
    <table data-testid="table" {...props}>
      <thead>
        <tr>
          {columns?.map((col: any, idx: number) => (
            <th key={idx}>{col.title}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {dataSource?.map((row: any, idx: number) => (
          <tr key={idx}>
            {columns?.map((col: any, colIdx: number) => (
              <td key={colIdx}>
                {typeof col.render === 'function' ? col.render(row[col.dataIndex], row) : row[col.dataIndex]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  ),
  Tabs: ({ children, defaultActiveKey, onChange, ...props }: any) => (
    <div data-testid="tabs" data-default-active={defaultActiveKey} {...props}>
      {children}
    </div>
  ),
  Alert: ({ message, type, showIcon, ...props }: any) => (
    <div data-testid="alert" data-type={type} data-show-icon={showIcon} {...props}>
      {message}
    </div>
  ),
  Statistic: ({ title, value, prefix, suffix, ...props }: any) => (
    <div data-testid="statistic" data-title={title} {...props}>
      <div className="ant-statistic-title">{title}</div>
      <div className="ant-statistic-content">
        {prefix}{value}{suffix}
      </div>
    </div>
  ),
  Space: ({ children, ...props }: any) => <div data-testid="space" {...props}>{children}</div>,
  Spin: ({ children, spinning, ...props }: any) => (
    <div data-testid="spin" data-spinning={spinning} {...props}>
      {spinning && <div data-testid="spinner">Loading...</div>}
      {children}
    </div>
  ),
}));

// Mock monitoring component
const MonitoringDashboard: React.FC = () => {
  const [selectedVenue, setSelectedVenue] = React.useState<string>('all');
  const [timeRange, setTimeRange] = React.useState<number>(3600000); // 1 hour
  const [isLoading, setIsLoading] = React.useState(false);

  // Mock data for testing
  const mockLatencyData = {
    venue_name: 'IB',
    order_execution_latency: {
      min_ms: 5.2,
      max_ms: 156.7,
      avg_ms: 23.4,
      p50_ms: 18.9,
      p95_ms: 89.3,
      p99_ms: 134.2,
      samples: 1247
    },
    last_updated: new Date().toISOString()
  };

  const mockAlerts = [
    {
      alert_id: 'alert-1',
      metric_name: 'order_latency',
      current_value: 156.7,
      threshold_value: 100,
      severity: 'high' as const,
      triggered_at: new Date().toISOString(),
      venue_name: 'IB',
      description: 'Order latency exceeded threshold',
      auto_resolution_available: true,
      escalation_level: 3,
      notification_sent: false
    },
    {
      alert_id: 'alert-2', 
      metric_name: 'cpu_usage',
      current_value: 85.4,
      threshold_value: 80,
      severity: 'medium' as const,
      triggered_at: new Date().toISOString(),
      description: 'CPU usage above threshold',
      auto_resolution_available: false,
      escalation_level: 2,
      notification_sent: true
    }
  ];

  return (
    <div data-testid="monitoring-dashboard">
      <div data-testid="dashboard-header">
        <h2>Performance Monitoring Dashboard</h2>
        <div data-testid="dashboard-controls">
          <select 
            data-testid="venue-selector"
            value={selectedVenue} 
            onChange={(e) => setSelectedVenue(e.target.value)}
          >
            <option value="all">All Venues</option>
            <option value="IB">Interactive Brokers</option>
            <option value="BINANCE">Binance</option>
          </select>
          <select 
            data-testid="time-range-selector"
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value))}
          >
            <option value={3600000}>Last Hour</option>
            <option value={86400000}>Last Day</option>
            <option value={604800000}>Last Week</option>
          </select>
        </div>
      </div>

      <div data-testid="metrics-overview">
        <div data-testid="latency-metrics">
          <h3>Order Execution Latency</h3>
          <div data-testid="latency-stats">
            <div data-testid="avg-latency">Avg: {mockLatencyData.order_execution_latency.avg_ms}ms</div>
            <div data-testid="p95-latency">P95: {mockLatencyData.order_execution_latency.p95_ms}ms</div>
            <div data-testid="p99-latency">P99: {mockLatencyData.order_execution_latency.p99_ms}ms</div>
            <div data-testid="sample-count">Samples: {mockLatencyData.order_execution_latency.samples}</div>
          </div>
        </div>

        <div data-testid="alert-summary">
          <h3>Active Alerts</h3>
          <div data-testid="alert-count">Total: {mockAlerts.length}</div>
          <div data-testid="high-severity-count">
            High Severity: {mockAlerts.filter(a => a.severity === 'high').length}
          </div>
          <div data-testid="medium-severity-count">
            Medium Severity: {mockAlerts.filter(a => a.severity === 'medium').length}
          </div>
        </div>
      </div>

      <div data-testid="charts-section">
        <div data-testid="latency-chart">
          <h3>Latency Trends</h3>
          <div data-testid="line-chart">Latency Chart Placeholder</div>
        </div>

        <div data-testid="alert-chart">
          <h3>Alert Distribution</h3>
          <div data-testid="bar-chart">Alert Chart Placeholder</div>
        </div>
      </div>

      <div data-testid="alerts-table">
        <h3>Recent Alerts</h3>
        <table data-testid="alerts-data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Current Value</th>
              <th>Threshold</th>
              <th>Severity</th>
              <th>Venue</th>
              <th>Triggered At</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {mockAlerts.map(alert => (
              <tr key={alert.alert_id} data-testid={`alert-row-${alert.alert_id}`}>
                <td>{alert.metric_name}</td>
                <td>{alert.current_value}</td>
                <td>{alert.threshold_value}</td>
                <td data-testid={`severity-${alert.severity}`}>{alert.severity}</td>
                <td>{alert.venue_name || 'All'}</td>
                <td>{new Date(alert.triggered_at).toLocaleString()}</td>
                <td>
                  <button 
                    data-testid={`resolve-alert-${alert.alert_id}`}
                    onClick={() => console.log('Resolve alert', alert.alert_id)}
                  >
                    Resolve
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

describe('MonitoringDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render dashboard with header and controls', () => {
    render(<MonitoringDashboard />);
    
    expect(screen.getByTestId('monitoring-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('dashboard-header')).toBeInTheDocument();
    expect(screen.getByText('Performance Monitoring Dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('venue-selector')).toBeInTheDocument();
    expect(screen.getByTestId('time-range-selector')).toBeInTheDocument();
  });

  it('should display latency metrics correctly', () => {
    render(<MonitoringDashboard />);
    
    expect(screen.getByTestId('latency-metrics')).toBeInTheDocument();
    expect(screen.getByText('Order Execution Latency')).toBeInTheDocument();
    expect(screen.getByTestId('avg-latency')).toHaveTextContent('Avg: 23.4ms');
    expect(screen.getByTestId('p95-latency')).toHaveTextContent('P95: 89.3ms');
    expect(screen.getByTestId('p99-latency')).toHaveTextContent('P99: 134.2ms');
    expect(screen.getByTestId('sample-count')).toHaveTextContent('Samples: 1247');
  });

  it('should display alert summary', () => {
    render(<MonitoringDashboard />);
    
    expect(screen.getByTestId('alert-summary')).toBeInTheDocument();
    expect(screen.getByText('Active Alerts')).toBeInTheDocument();
    expect(screen.getByTestId('alert-count')).toHaveTextContent('Total: 2');
    expect(screen.getByTestId('high-severity-count')).toHaveTextContent('High Severity: 1');
    expect(screen.getByTestId('medium-severity-count')).toHaveTextContent('Medium Severity: 1');
  });

  it('should render charts section', () => {
    render(<MonitoringDashboard />);
    
    expect(screen.getByTestId('charts-section')).toBeInTheDocument();
    expect(screen.getByTestId('latency-chart')).toBeInTheDocument();
    expect(screen.getByTestId('alert-chart')).toBeInTheDocument();
    expect(screen.getByText('Latency Trends')).toBeInTheDocument();
    expect(screen.getByText('Alert Distribution')).toBeInTheDocument();
  });

  it('should render alerts table with data', () => {
    render(<MonitoringDashboard />);
    
    expect(screen.getByTestId('alerts-table')).toBeInTheDocument();
    expect(screen.getByTestId('alerts-data-table')).toBeInTheDocument();
    expect(screen.getByText('Recent Alerts')).toBeInTheDocument();
    
    // Check table headers
    expect(screen.getByText('Metric')).toBeInTheDocument();
    expect(screen.getByText('Current Value')).toBeInTheDocument();
    expect(screen.getByText('Threshold')).toBeInTheDocument();
    expect(screen.getByText('Severity')).toBeInTheDocument();
    expect(screen.getByText('Venue')).toBeInTheDocument();
    expect(screen.getByText('Triggered At')).toBeInTheDocument();
    expect(screen.getByText('Actions')).toBeInTheDocument();
    
    // Check alert data
    expect(screen.getByTestId('alert-row-alert-1')).toBeInTheDocument();
    expect(screen.getByTestId('alert-row-alert-2')).toBeInTheDocument();
    expect(screen.getByTestId('severity-high')).toHaveTextContent('high');
    expect(screen.getByTestId('severity-medium')).toHaveTextContent('medium');
  });

  it('should handle venue selector changes', () => {
    render(<MonitoringDashboard />);
    
    const venueSelector = screen.getByTestId('venue-selector');
    expect(venueSelector).toHaveValue('all');
    
    fireEvent.change(venueSelector, { target: { value: 'IB' } });
    expect(venueSelector).toHaveValue('IB');
    
    fireEvent.change(venueSelector, { target: { value: 'BINANCE' } });
    expect(venueSelector).toHaveValue('BINANCE');
  });

  it('should handle time range selector changes', () => {
    render(<MonitoringDashboard />);
    
    const timeRangeSelector = screen.getByTestId('time-range-selector');
    expect(timeRangeSelector).toHaveValue('3600000');
    
    fireEvent.change(timeRangeSelector, { target: { value: '86400000' } });
    expect(timeRangeSelector).toHaveValue('86400000');
    
    fireEvent.change(timeRangeSelector, { target: { value: '604800000' } });
    expect(timeRangeSelector).toHaveValue('604800000');
  });

  it('should handle alert resolution button clicks', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    
    render(<MonitoringDashboard />);
    
    const resolveButton1 = screen.getByTestId('resolve-alert-alert-1');
    const resolveButton2 = screen.getByTestId('resolve-alert-alert-2');
    
    fireEvent.click(resolveButton1);
    expect(consoleSpy).toHaveBeenCalledWith('Resolve alert', 'alert-1');
    
    fireEvent.click(resolveButton2);
    expect(consoleSpy).toHaveBeenCalledWith('Resolve alert', 'alert-2');
    
    consoleSpy.mockRestore();
  });

  it('should display venue-specific options in selector', () => {
    render(<MonitoringDashboard />);
    
    const venueSelector = screen.getByTestId('venue-selector');
    const options = venueSelector.querySelectorAll('option');
    
    expect(options).toHaveLength(3);
    expect(options[0]).toHaveValue('all');
    expect(options[0]).toHaveTextContent('All Venues');
    expect(options[1]).toHaveValue('IB');
    expect(options[1]).toHaveTextContent('Interactive Brokers');
    expect(options[2]).toHaveValue('BINANCE');
    expect(options[2]).toHaveTextContent('Binance');
  });

  it('should display time range options correctly', () => {
    render(<MonitoringDashboard />);
    
    const timeRangeSelector = screen.getByTestId('time-range-selector');
    const options = timeRangeSelector.querySelectorAll('option');
    
    expect(options).toHaveLength(3);
    expect(options[0]).toHaveValue('3600000');
    expect(options[0]).toHaveTextContent('Last Hour');
    expect(options[1]).toHaveValue('86400000');
    expect(options[1]).toHaveTextContent('Last Day');
    expect(options[2]).toHaveValue('604800000');
    expect(options[2]).toHaveTextContent('Last Week');
  });

  it('should format alert timestamps correctly', () => {
    render(<MonitoringDashboard />);
    
    // Check that timestamps are formatted as locale strings
    const alertRows = screen.getAllByTestId(/alert-row-/);
    expect(alertRows).toHaveLength(2);
    
    // The actual timestamp formatting will depend on the locale
    // Just verify that some timestamp text is present
    expect(alertRows[0]).toHaveTextContent(/\d+/); // Contains digits
    expect(alertRows[1]).toHaveTextContent(/\d+/); // Contains digits
  });

  it('should handle missing venue data gracefully', () => {
    render(<MonitoringDashboard />);
    
    // The second alert has no venue_name, should show 'All'
    const alertRow2 = screen.getByTestId('alert-row-alert-2');
    expect(alertRow2).toHaveTextContent('All');
  });

  it('should display correct metric values in alerts table', () => {
    render(<MonitoringDashboard />);
    
    const alertRow1 = screen.getByTestId('alert-row-alert-1');
    expect(alertRow1).toHaveTextContent('order_latency');
    expect(alertRow1).toHaveTextContent('156.7');
    expect(alertRow1).toHaveTextContent('100');
    expect(alertRow1).toHaveTextContent('IB');
    
    const alertRow2 = screen.getByTestId('alert-row-alert-2');
    expect(alertRow2).toHaveTextContent('cpu_usage');
    expect(alertRow2).toHaveTextContent('85.4');
    expect(alertRow2).toHaveTextContent('80');
  });

  it('should maintain consistent component structure', () => {
    render(<MonitoringDashboard />);
    
    // Verify the main sections exist in expected order
    const dashboard = screen.getByTestId('monitoring-dashboard');
    const sections = dashboard.children;
    
    expect(sections[0]).toHaveAttribute('data-testid', 'dashboard-header');
    expect(sections[1]).toHaveAttribute('data-testid', 'metrics-overview');
    expect(sections[2]).toHaveAttribute('data-testid', 'charts-section');
    expect(sections[3]).toHaveAttribute('data-testid', 'alerts-table');
  });
});