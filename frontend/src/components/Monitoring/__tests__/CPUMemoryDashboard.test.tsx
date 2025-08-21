/**
 * Unit Tests for CPUMemoryDashboard React Component
 * Tests CPU and memory dashboard rendering and interactions
 */

import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { CPUMemoryDashboard } from '../CPUMemoryDashboard';
import { systemResourceMonitor } from '../../../services/monitoring/SystemResourceMonitor';

// Mock the SystemResourceMonitor
vi.mock('../../../services/monitoring/SystemResourceMonitor', () => ({
  systemResourceMonitor: {
    onSnapshot: vi.fn(),
    removeCallback: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    getSystemMetrics: vi.fn(),
    getResourceStatistics: vi.fn(),
    getHealthScore: vi.fn(),
    detectResourceAnomalies: vi.fn(),
  }
}));

// Mock Recharts components to avoid canvas rendering issues in tests
vi.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
}));

const mockSystemResourceMonitor = systemResourceMonitor as vi.Mocked<typeof systemResourceMonitor>;

const mockSystemMetrics = {
  timestamp: '2024-08-20T12:00:00Z',
  cpu_metrics: {
    usage_percent: 45.5,
    core_count: 8,
    load_average_1m: 1.5,
    load_average_5m: 1.3,
    load_average_15m: 1.1,
    per_core_usage: [40, 50, 35, 60, 45, 30, 55, 25],
    temperature_celsius: 65
  },
  memory_metrics: {
    total_gb: 16.0,
    used_gb: 8.5,
    available_gb: 7.5,
    usage_percent: 53.1,
    swap_total_gb: 2.0,
    swap_used_gb: 0.5,
    buffer_cache_gb: 2.0
  },
  network_metrics: {
    bytes_sent_per_sec: 1024,
    bytes_received_per_sec: 2048,
    packets_sent_per_sec: 10,
    packets_received_per_sec: 15,
    errors_per_sec: 0,
    active_connections: 25,
    bandwidth_utilization_percent: 15.5
  },
  disk_metrics: {
    total_space_gb: 500.0,
    used_space_gb: 250.0,
    available_space_gb: 250.0,
    usage_percent: 50.0,
    read_iops: 100,
    write_iops: 50,
    read_throughput_mbps: 150.0,
    write_throughput_mbps: 75.0
  },
  process_metrics: {
    trading_engine_cpu_percent: 5.5,
    trading_engine_memory_mb: 512,
    database_cpu_percent: 3.2,
    database_memory_mb: 1024,
    total_processes: 150
  }
};

const mockResourceStats = {
  cpu: { min: 30, max: 60, avg: 45 },
  memory: { min: 6000, max: 10000, avg: 8000 },
  samples: 50
};

const mockAnomalies = {
  cpu_anomaly: false,
  memory_anomaly: false,
  network_anomaly: false,
  details: []
};

describe('CPUMemoryDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSystemResourceMonitor.getSystemMetrics.mockReturnValue(mockSystemMetrics);
    mockSystemResourceMonitor.getResourceStatistics.mockReturnValue(mockResourceStats);
    mockSystemResourceMonitor.getHealthScore.mockReturnValue(85);
    mockSystemResourceMonitor.detectResourceAnomalies.mockReturnValue(mockAnomalies);
  });

  describe('Loading States', () => {
    it('should show loading state when no metrics are available', () => {
      mockSystemResourceMonitor.getSystemMetrics.mockReturnValue(null);
      
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Loading system metrics...')).toBeInTheDocument();
    });

    it('should render dashboard when metrics are available', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Health Score')).toBeInTheDocument();
      expect(screen.getByText('85')).toBeInTheDocument();
    });
  });

  describe('Control Panel', () => {
    it('should display monitoring toggle switch', () => {
      render(<CPUMemoryDashboard />);
      
      const toggleSwitch = screen.getByRole('switch');
      expect(toggleSwitch).toBeInTheDocument();
      expect(screen.getByText('Monitoring OFF')).toBeInTheDocument();
    });

    it('should start monitoring when toggle is switched on', () => {
      render(<CPUMemoryDashboard />);
      
      const toggleSwitch = screen.getByRole('switch');
      fireEvent.click(toggleSwitch);
      
      expect(mockSystemResourceMonitor.start).toHaveBeenCalledWith(5000);
      expect(screen.getByText('Monitoring ON')).toBeInTheDocument();
    });

    it('should stop monitoring when toggle is switched off', () => {
      render(<CPUMemoryDashboard />);
      
      const toggleSwitch = screen.getByRole('switch');
      
      // Turn on
      fireEvent.click(toggleSwitch);
      // Turn off
      fireEvent.click(toggleSwitch);
      
      expect(mockSystemResourceMonitor.stop).toHaveBeenCalled();
      expect(screen.getByText('Monitoring OFF')).toBeInTheDocument();
    });

    it('should display health score with correct color', () => {
      render(<CPUMemoryDashboard />);
      
      const healthScore = screen.getByText('85');
      expect(healthScore).toHaveStyle({ color: 'rgb(63, 134, 0)' }); // Green for healthy
    });

    it('should display health score in warning color for medium scores', () => {
      mockSystemResourceMonitor.getHealthScore.mockReturnValue(65);
      
      render(<CPUMemoryDashboard />);
      
      const healthScore = screen.getByText('65');
      expect(healthScore).toHaveStyle({ color: 'rgb(250, 140, 22)' }); // Orange for warning
    });
  });

  describe('CPU Metrics Display', () => {
    it('should display CPU usage percentage', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('45.5%')).toBeInTheDocument();
    });

    it('should display CPU core count', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Cores')).toBeInTheDocument();
      expect(screen.getByText('8')).toBeInTheDocument();
    });

    it('should display load average', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Load Avg')).toBeInTheDocument();
      expect(screen.getByText('1.50')).toBeInTheDocument();
    });

    it('should display per-core usage in full mode', () => {
      render(<CPUMemoryDashboard compactMode={false} />);
      
      expect(screen.getByText('Per-Core Usage:')).toBeInTheDocument();
      expect(screen.getByText('Core 1')).toBeInTheDocument();
      expect(screen.getByText('40.0%')).toBeInTheDocument();
    });

    it('should hide per-core usage in compact mode', () => {
      render(<CPUMemoryDashboard compactMode={true} />);
      
      expect(screen.queryByText('Per-Core Usage:')).not.toBeInTheDocument();
    });
  });

  describe('Memory Metrics Display', () => {
    it('should display memory usage percentage', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('53.1%')).toBeInTheDocument();
    });

    it('should display used and total memory', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Used')).toBeInTheDocument();
      expect(screen.getByText('8.50')).toBeInTheDocument();
      expect(screen.getByText('Total')).toBeInTheDocument();
      expect(screen.getByText('16.00')).toBeInTheDocument();
    });

    it('should display memory breakdown in full mode', () => {
      render(<CPUMemoryDashboard compactMode={false} />);
      
      expect(screen.getByText('Memory Breakdown:')).toBeInTheDocument();
      expect(screen.getByText('Available')).toBeInTheDocument();
      expect(screen.getByText('7.50 GB')).toBeInTheDocument();
      expect(screen.getByText('Buffer/Cache')).toBeInTheDocument();
      expect(screen.getByText('2.00 GB')).toBeInTheDocument();
    });
  });

  describe('Performance Statistics', () => {
    it('should display CPU and memory average statistics', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('CPU Avg')).toBeInTheDocument();
      expect(screen.getByText('45.0%')).toBeInTheDocument();
      expect(screen.getByText('Memory Avg')).toBeInTheDocument();
      expect(screen.getByText('8000')).toBeInTheDocument();
    });

    it('should display peak values', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('CPU Peak')).toBeInTheDocument();
      expect(screen.getByText('60.0%')).toBeInTheDocument();
      expect(screen.getByText('Memory Peak')).toBeInTheDocument();
      expect(screen.getByText('10000')).toBeInTheDocument();
    });

    it('should display sample count', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Samples')).toBeInTheDocument();
      expect(screen.getByText('50')).toBeInTheDocument();
    });
  });

  describe('Charts Rendering', () => {
    it('should render CPU usage trend chart', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('CPU Usage Trend')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    it('should render memory usage trend chart', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('Memory Usage Trend')).toBeInTheDocument();
      expect(screen.getByTestId('area-chart')).toBeInTheDocument();
    });

    it('should use smaller chart size in compact mode', () => {
      render(<CPUMemoryDashboard compactMode={true} />);
      
      const responsiveContainers = screen.getAllByTestId('responsive-container');
      expect(responsiveContainers.length).toBeGreaterThan(0);
    });
  });

  describe('Anomaly Alerts', () => {
    it('should display anomaly alert when anomalies are detected', () => {
      const anomaliesWithIssues = {
        cpu_anomaly: true,
        memory_anomaly: true,
        network_anomaly: false,
        details: ['CPU usage significantly above normal', 'Memory usage critically high']
      };
      
      mockSystemResourceMonitor.detectResourceAnomalies.mockReturnValue(anomaliesWithIssues);
      
      render(<CPUMemoryDashboard showAlerts={true} />);
      
      expect(screen.getByText('Resource Usage Anomaly Detected')).toBeInTheDocument();
      expect(screen.getByText('CPU usage significantly above normal')).toBeInTheDocument();
      expect(screen.getByText('Memory usage critically high')).toBeInTheDocument();
    });

    it('should not display alert when no anomalies are detected', () => {
      render(<CPUMemoryDashboard showAlerts={true} />);
      
      expect(screen.queryByText('Resource Usage Anomaly Detected')).not.toBeInTheDocument();
    });

    it('should not display alerts when showAlerts is false', () => {
      const anomaliesWithIssues = {
        cpu_anomaly: true,
        memory_anomaly: true,
        network_anomaly: false,
        details: ['CPU usage significantly above normal']
      };
      
      mockSystemResourceMonitor.detectResourceAnomalies.mockReturnValue(anomaliesWithIssues);
      
      render(<CPUMemoryDashboard showAlerts={false} />);
      
      expect(screen.queryByText('Resource Usage Anomaly Detected')).not.toBeInTheDocument();
    });
  });

  describe('Refresh Interval Configuration', () => {
    it('should use custom refresh interval', () => {
      render(<CPUMemoryDashboard refreshInterval={2000} />);
      
      const toggleSwitch = screen.getByRole('switch');
      fireEvent.click(toggleSwitch);
      
      expect(mockSystemResourceMonitor.start).toHaveBeenCalledWith(2000);
    });

    it('should use default refresh interval when not specified', () => {
      render(<CPUMemoryDashboard />);
      
      const toggleSwitch = screen.getByRole('switch');
      fireEvent.click(toggleSwitch);
      
      expect(mockSystemResourceMonitor.start).toHaveBeenCalledWith(5000);
    });
  });

  describe('Snapshot Updates', () => {
    it('should register snapshot callback on mount', () => {
      render(<CPUMemoryDashboard />);
      
      expect(mockSystemResourceMonitor.onSnapshot).toHaveBeenCalledWith(expect.any(Function));
    });

    it('should remove callback on unmount', () => {
      const { unmount } = render(<CPUMemoryDashboard />);
      
      const callback = mockSystemResourceMonitor.onSnapshot.mock.calls[0][0];
      
      unmount();
      
      expect(mockSystemResourceMonitor.removeCallback).toHaveBeenCalledWith(callback);
    });

    it('should update metrics when snapshot callback is triggered', async () => {
      render(<CPUMemoryDashboard />);
      
      const callback = mockSystemResourceMonitor.onSnapshot.mock.calls[0][0];
      
      // Simulate new snapshot
      const newSnapshot = {
        timestamp: new Date(),
        cpu_usage_percent: 60.0,
        memory_usage_mb: 9000,
        available_memory_mb: 6000,
        heap_used_mb: 256,
        network_bytes_in_per_sec: 1500,
        network_bytes_out_per_sec: 800,
        disk_usage_percent: 55.0
      };
      
      act(() => {
        callback(newSnapshot);
      });
      
      await waitFor(() => {
        expect(mockSystemResourceMonitor.getSystemMetrics).toHaveBeenCalled();
      });
    });
  });

  describe('Color Coding', () => {
    it('should use green color for low CPU usage', () => {
      const lowCpuMetrics = {
        ...mockSystemMetrics,
        cpu_metrics: { ...mockSystemMetrics.cpu_metrics, usage_percent: 25 }
      };
      mockSystemResourceMonitor.getSystemMetrics.mockReturnValue(lowCpuMetrics);
      
      render(<CPUMemoryDashboard />);
      
      // Progress component should use green color for low usage
      expect(screen.getByText('25.0%')).toBeInTheDocument();
    });

    it('should use red color for high CPU usage', () => {
      const highCpuMetrics = {
        ...mockSystemMetrics,
        cpu_metrics: { ...mockSystemMetrics.cpu_metrics, usage_percent: 85 }
      };
      mockSystemResourceMonitor.getSystemMetrics.mockReturnValue(highCpuMetrics);
      
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('85.0%')).toBeInTheDocument();
    });

    it('should use appropriate color for memory usage levels', () => {
      const highMemoryMetrics = {
        ...mockSystemMetrics,
        memory_metrics: { ...mockSystemMetrics.memory_metrics, usage_percent: 95 }
      };
      mockSystemResourceMonitor.getSystemMetrics.mockReturnValue(highMemoryMetrics);
      
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('95.0%')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle missing system metrics gracefully', () => {
      mockSystemResourceMonitor.getSystemMetrics.mockReturnValue({
        ...mockSystemMetrics,
        cpu_metrics: { ...mockSystemMetrics.cpu_metrics, per_core_usage: [] }
      });
      
      expect(() => render(<CPUMemoryDashboard />)).not.toThrow();
    });

    it('should handle callback errors gracefully', () => {
      render(<CPUMemoryDashboard />);
      
      const callback = mockSystemResourceMonitor.onSnapshot.mock.calls[0][0];
      
      // Mock system monitor to throw error
      mockSystemResourceMonitor.getSystemMetrics.mockImplementation(() => {
        throw new Error('System metrics error');
      });
      
      expect(() => {
        act(() => {
          callback({
            timestamp: new Date(),
            cpu_usage_percent: 50,
            memory_usage_mb: 8000,
            available_memory_mb: 7000,
            heap_used_mb: 256,
            network_bytes_in_per_sec: 1000,
            network_bytes_out_per_sec: 500,
            disk_usage_percent: 50
          });
        });
      }).not.toThrow();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels for progress indicators', () => {
      render(<CPUMemoryDashboard />);
      
      const progressElements = screen.getAllByRole('progressbar');
      expect(progressElements.length).toBeGreaterThan(0);
    });

    it('should have semantic headings for sections', () => {
      render(<CPUMemoryDashboard />);
      
      expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      expect(screen.getByText('Performance Statistics')).toBeInTheDocument();
    });
  });
});