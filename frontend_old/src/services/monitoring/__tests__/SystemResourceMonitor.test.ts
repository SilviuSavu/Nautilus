/**
 * Unit Tests for SystemResourceMonitor
 * Tests system resource monitoring and anomaly detection
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SystemResourceMonitor, SystemResourceSnapshot } from '../SystemResourceMonitor';

// Mock the systeminformation library
vi.mock('systeminformation', () => ({
  cpu: vi.fn(),
  mem: vi.fn(),
  networkStats: vi.fn(),
  fsSize: vi.fn(),
  processes: vi.fn(),
  currentLoad: vi.fn(),
}));

import * as si from 'systeminformation';

describe('SystemResourceMonitor', () => {
  let monitor: SystemResourceMonitor;
  const mockCpu = si.cpu as vi.MockedFunction<typeof si.cpu>;
  const mockMem = si.mem as vi.MockedFunction<typeof si.mem>;
  const mockNetworkStats = si.networkStats as vi.MockedFunction<typeof si.networkStats>;
  const mockFsSize = si.fsSize as vi.MockedFunction<typeof si.fsSize>;
  const mockProcesses = si.processes as vi.MockedFunction<typeof si.processes>;
  const mockCurrentLoad = si.currentLoad as vi.MockedFunction<typeof si.currentLoad>;

  beforeEach(() => {
    monitor = new SystemResourceMonitor();
    vi.clearAllMocks();
    
    // Setup default mocks
    mockCpu.mockResolvedValue({
      manufacturer: 'Intel',
      brand: 'Core i7',
      cores: 8,
      physicalCores: 4,
      speed: '2.8',
      speedMin: '1.0',
      speedMax: '4.0'
    } as any);

    mockMem.mockResolvedValue({
      total: 16 * 1024 * 1024 * 1024, // 16GB
      available: 8 * 1024 * 1024 * 1024, // 8GB
      used: 8 * 1024 * 1024 * 1024, // 8GB
      swaptotal: 2 * 1024 * 1024 * 1024, // 2GB
      swapused: 100 * 1024 * 1024 // 100MB
    } as any);

    mockCurrentLoad.mockResolvedValue({
      avgLoad: 1.5,
      currentLoad: 35.5,
      cpus: [
        { load: 30.0 },
        { load: 25.0 },
        { load: 40.0 },
        { load: 45.0 },
        { load: 35.0 },
        { load: 30.0 },
        { load: 20.0 },
        { load: 50.0 }
      ]
    } as any);

    mockNetworkStats.mockResolvedValue([{
      iface: 'en0',
      operstate: 'up',
      rx_bytes: 1000000,
      tx_bytes: 500000,
      rx_errors: 0,
      tx_errors: 0,
      rx_dropped: 0,
      tx_dropped: 0
    }] as any);

    mockFsSize.mockResolvedValue([{
      fs: '/dev/disk1',
      type: 'apfs',
      size: 500 * 1024 * 1024 * 1024, // 500GB
      used: 250 * 1024 * 1024 * 1024, // 250GB
      available: 250 * 1024 * 1024 * 1024, // 250GB
      use: 50.0,
      mount: '/'
    }] as any);

    mockProcesses.mockResolvedValue({
      all: 150,
      running: 5,
      blocked: 0,
      sleeping: 145,
      list: []
    } as any);
  });

  afterEach(() => {
    monitor.stop();
  });

  describe('Initialization and Configuration', () => {
    it('should initialize with default configuration', () => {
      expect(monitor.isRunning()).toBe(false);
      expect(monitor.getHealthScore()).toBeGreaterThanOrEqual(0);
    });

    it('should start monitoring', async () => {
      monitor.start(1000); // 1 second interval
      expect(monitor.isRunning()).toBe(true);
      monitor.stop();
    });

    it('should stop monitoring', () => {
      monitor.start(1000);
      monitor.stop();
      expect(monitor.isRunning()).toBe(false);
    });
  });

  describe('System Metrics Collection', () => {
    it('should collect system metrics snapshot', async () => {
      await monitor.collectSnapshot();
      
      const metrics = monitor.getSystemMetrics();
      
      expect(metrics).toBeTruthy();
      expect(metrics.cpu_metrics.core_count).toBe(8);
      expect(metrics.cpu_metrics.usage_percent).toBeCloseTo(35.5, 1);
      expect(metrics.memory_metrics.total_gb).toBeCloseTo(16, 1);
      expect(metrics.memory_metrics.used_gb).toBeCloseTo(8, 1);
      expect(metrics.process_metrics.total_processes).toBe(150);
    });

    it('should handle system metrics collection errors gracefully', async () => {
      mockCpu.mockRejectedValue(new Error('CPU info unavailable'));
      
      await monitor.collectSnapshot();
      
      const metrics = monitor.getSystemMetrics();
      expect(metrics).toBeTruthy(); // Should still return metrics with default values
    });

    it('should calculate derived metrics correctly', async () => {
      await monitor.collectSnapshot();
      
      const metrics = monitor.getSystemMetrics();
      
      expect(metrics.memory_metrics.usage_percent).toBeCloseTo(50, 1); // 8GB used / 16GB total
      expect(metrics.disk_metrics.usage_percent).toBeCloseTo(50, 1); // 250GB used / 500GB total
    });
  });

  describe('Resource Statistics', () => {
    beforeEach(async () => {
      // Collect some snapshots with varying data
      for (let i = 0; i < 5; i++) {
        mockCurrentLoad.mockResolvedValue({
          avgLoad: 1.5 + i * 0.2,
          currentLoad: 30 + i * 5,
          cpus: Array(8).fill(null).map((_, j) => ({ load: 25 + i * 5 + j * 2 }))
        } as any);

        mockMem.mockResolvedValue({
          total: 16 * 1024 * 1024 * 1024,
          available: (8 - i) * 1024 * 1024 * 1024,
          used: (8 + i) * 1024 * 1024 * 1024,
          swaptotal: 2 * 1024 * 1024 * 1024,
          swapused: (100 + i * 50) * 1024 * 1024
        } as any);

        await monitor.collectSnapshot();
        
        // Small delay to ensure different timestamps
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    });

    it('should calculate resource statistics correctly', () => {
      const stats = monitor.getResourceStatistics(60000); // Last minute
      
      expect(stats.samples).toBe(5);
      expect(stats.cpu.min).toBeGreaterThan(0);
      expect(stats.cpu.max).toBeGreaterThan(stats.cpu.min);
      expect(stats.cpu.avg).toBeGreaterThan(0);
      expect(stats.memory.min).toBeGreaterThan(0);
      expect(stats.memory.max).toBeGreaterThan(stats.memory.min);
    });

    it('should filter statistics by time range', () => {
      const recentStats = monitor.getResourceStatistics(1000); // Last second
      const allStats = monitor.getResourceStatistics(); // All time
      
      expect(recentStats.samples).toBeLessThanOrEqual(allStats.samples);
    });
  });

  describe('Health Score Calculation', () => {
    it('should calculate health score based on resource usage', async () => {
      // Low usage scenario
      mockCurrentLoad.mockResolvedValue({
        avgLoad: 0.5,
        currentLoad: 20,
        cpus: Array(8).fill(null).map(() => ({ load: 20 }))
      } as any);

      mockMem.mockResolvedValue({
        total: 16 * 1024 * 1024 * 1024,
        available: 12 * 1024 * 1024 * 1024, // 75% available
        used: 4 * 1024 * 1024 * 1024, // 25% used
        swaptotal: 2 * 1024 * 1024 * 1024,
        swapused: 0
      } as any);

      await monitor.collectSnapshot();
      
      const healthScore = monitor.getHealthScore();
      expect(healthScore).toBeGreaterThan(80); // Should be high with low usage
    });

    it('should return lower health score for high resource usage', async () => {
      // High usage scenario
      mockCurrentLoad.mockResolvedValue({
        avgLoad: 8.0,
        currentLoad: 95,
        cpus: Array(8).fill(null).map(() => ({ load: 95 }))
      } as any);

      mockMem.mockResolvedValue({
        total: 16 * 1024 * 1024 * 1024,
        available: 1 * 1024 * 1024 * 1024, // Only 6.25% available
        used: 15 * 1024 * 1024 * 1024, // 93.75% used
        swaptotal: 2 * 1024 * 1024 * 1024,
        swapused: 1.5 * 1024 * 1024 * 1024 // High swap usage
      } as any);

      await monitor.collectSnapshot();
      
      const healthScore = monitor.getHealthScore();
      expect(healthScore).toBeLessThan(30); // Should be low with high usage
    });
  });

  describe('Anomaly Detection', () => {
    beforeEach(async () => {
      // Create baseline with normal values
      for (let i = 0; i < 10; i++) {
        mockCurrentLoad.mockResolvedValue({
          avgLoad: 1.0 + Math.random() * 0.5,
          currentLoad: 30 + Math.random() * 10,
          cpus: Array(8).fill(null).map(() => ({ load: 25 + Math.random() * 15 }))
        } as any);

        await monitor.collectSnapshot();
      }
    });

    it('should detect CPU anomalies', async () => {
      // Inject anomalous CPU usage
      mockCurrentLoad.mockResolvedValue({
        avgLoad: 8.0,
        currentLoad: 95,
        cpus: Array(8).fill(null).map(() => ({ load: 95 }))
      } as any);

      await monitor.collectSnapshot();
      
      const anomalies = monitor.detectResourceAnomalies();
      expect(anomalies.cpu_anomaly).toBe(true);
      expect(anomalies.details).toContain('CPU usage significantly above normal');
    });

    it('should detect memory anomalies', async () => {
      // Inject anomalous memory usage
      mockMem.mockResolvedValue({
        total: 16 * 1024 * 1024 * 1024,
        available: 0.5 * 1024 * 1024 * 1024, // Very low available memory
        used: 15.5 * 1024 * 1024 * 1024,
        swaptotal: 2 * 1024 * 1024 * 1024,
        swapused: 1.8 * 1024 * 1024 * 1024 // High swap usage
      } as any);

      await monitor.collectSnapshot();
      
      const anomalies = monitor.detectResourceAnomalies();
      expect(anomalies.memory_anomaly).toBe(true);
      expect(anomalies.details).toContain('Memory usage significantly above normal');
    });

    it('should not detect anomalies for normal usage', async () => {
      // Normal usage within baseline range
      mockCurrentLoad.mockResolvedValue({
        avgLoad: 1.2,
        currentLoad: 35,
        cpus: Array(8).fill(null).map(() => ({ load: 30 }))
      } as any);

      await monitor.collectSnapshot();
      
      const anomalies = monitor.detectResourceAnomalies();
      expect(anomalies.cpu_anomaly).toBe(false);
      expect(anomalies.memory_anomaly).toBe(false);
      expect(anomalies.network_anomaly).toBe(false);
    });
  });

  describe('Snapshot Management', () => {
    it('should maintain snapshot history limit', async () => {
      // Create more snapshots than the limit
      for (let i = 0; i < 110; i++) {
        await monitor.collectSnapshot();
      }
      
      const snapshots = monitor.getRecentSnapshots(150);
      expect(snapshots.length).toBeLessThanOrEqual(100); // Should be limited
    });

    it('should provide recent snapshots', async () => {
      await monitor.collectSnapshot();
      await monitor.collectSnapshot();
      
      const snapshots = monitor.getRecentSnapshots(5);
      expect(snapshots.length).toBe(2);
      expect(snapshots[0].timestamp).toBeInstanceOf(Date);
    });

    it('should clear all snapshots', async () => {
      await monitor.collectSnapshot();
      
      monitor.clearSnapshots();
      
      const snapshots = monitor.getRecentSnapshots();
      expect(snapshots.length).toBe(0);
    });
  });

  describe('Callbacks and Events', () => {
    it('should trigger callbacks on new snapshots', async () => {
      const mockCallback = vi.fn();
      monitor.onSnapshot(mockCallback);
      
      await monitor.collectSnapshot();
      
      expect(mockCallback).toHaveBeenCalledTimes(1);
      expect(mockCallback).toHaveBeenCalledWith(expect.objectContaining({
        timestamp: expect.any(Date),
        cpu_usage_percent: expect.any(Number),
        memory_usage_mb: expect.any(Number)
      }));
    });

    it('should remove callbacks correctly', async () => {
      const mockCallback = vi.fn();
      monitor.onSnapshot(mockCallback);
      monitor.removeCallback(mockCallback);
      
      await monitor.collectSnapshot();
      
      expect(mockCallback).not.toHaveBeenCalled();
    });

    it('should handle callback errors gracefully', async () => {
      const errorCallback = vi.fn().mockImplementation(() => {
        throw new Error('Callback error');
      });
      
      monitor.onSnapshot(errorCallback);
      
      // Should not throw
      await expect(monitor.collectSnapshot()).resolves.not.toThrow();
    });
  });

  describe('Monitoring Loop', () => {
    it('should collect snapshots at regular intervals when started', async () => {
      const mockCallback = vi.fn();
      monitor.onSnapshot(mockCallback);
      
      monitor.start(100); // 100ms interval
      
      // Wait for a few intervals
      await new Promise(resolve => setTimeout(resolve, 350));
      
      monitor.stop();
      
      expect(mockCallback).toHaveBeenCalledTimes(3); // Should have been called ~3 times
    });

    it('should not collect snapshots when stopped', async () => {
      const mockCallback = vi.fn();
      monitor.onSnapshot(mockCallback);
      
      monitor.start(100);
      monitor.stop();
      
      await new Promise(resolve => setTimeout(resolve, 250));
      
      // Should have been called at most once (initial collection on start)
      expect(mockCallback).toHaveBeenCalledTimes(1);
    });
  });

  describe('Error Handling', () => {
    it('should handle missing systeminformation data gracefully', async () => {
      mockCpu.mockResolvedValue({} as any);
      mockMem.mockResolvedValue({} as any);
      mockCurrentLoad.mockResolvedValue({} as any);
      
      await monitor.collectSnapshot();
      
      const metrics = monitor.getSystemMetrics();
      expect(metrics).toBeTruthy();
      expect(metrics.cpu_metrics.core_count).toBe(0); // Should use default values
      expect(metrics.memory_metrics.total_gb).toBe(0);
    });

    it('should continue monitoring after collection errors', async () => {
      const mockCallback = vi.fn();
      monitor.onSnapshot(mockCallback);
      
      // First call succeeds
      mockCpu.mockResolvedValueOnce({
        cores: 4,
        physicalCores: 2
      } as any);
      
      // Second call fails
      mockCpu.mockRejectedValueOnce(new Error('System error'));
      
      // Third call succeeds again
      mockCpu.mockResolvedValueOnce({
        cores: 4,
        physicalCores: 2
      } as any);
      
      await monitor.collectSnapshot();
      await monitor.collectSnapshot();
      await monitor.collectSnapshot();
      
      expect(mockCallback).toHaveBeenCalledTimes(3);
    });
  });

  describe('Performance Optimization', () => {
    it('should not block on slow system calls', async () => {
      // Mock slow system calls
      mockCpu.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({
        cores: 4,
        physicalCores: 2
      } as any), 100)));

      const startTime = Date.now();
      await monitor.collectSnapshot();
      const endTime = Date.now();
      
      // Should complete in reasonable time even with slow system calls
      expect(endTime - startTime).toBeLessThan(500);
    });

    it('should limit memory usage with snapshot history', async () => {
      // Simulate many snapshots
      for (let i = 0; i < 200; i++) {
        await monitor.collectSnapshot();
      }
      
      const snapshots = monitor.getRecentSnapshots();
      expect(snapshots.length).toBeLessThanOrEqual(100); // Should be limited
    });
  });
});