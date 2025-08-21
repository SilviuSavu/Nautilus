/**
 * Unit Tests for OrderLatencyTracker
 * Tests order execution timing measurement and latency analysis
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OrderLatencyTracker, LatencyMeasurement } from '../OrderLatencyTracker';

describe('OrderLatencyTracker', () => {
  let tracker: OrderLatencyTracker;

  beforeEach(() => {
    tracker = new OrderLatencyTracker();
    vi.clearAllMocks();
  });

  afterEach(() => {
    tracker.clear();
  });

  describe('Order Tracking', () => {
    it('should start tracking an order', () => {
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      
      expect(tracker.getPendingOrdersCount()).toBe(1);
      expect(tracker.getPendingOrdersByVenue().get('IB')).toBe(1);
    });

    it('should complete order tracking and calculate latency', () => {
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      
      // Wait a small amount to ensure latency > 0
      setTimeout(() => {
        const measurement = tracker.completeOrder('order-1', true);
        
        expect(measurement).toBeTruthy();
        expect(measurement!.orderId).toBe('order-1');
        expect(measurement!.venue).toBe('IB');
        expect(measurement!.latency_ms).toBeGreaterThan(0);
        expect(measurement!.success).toBe(true);
        expect(tracker.getPendingOrdersCount()).toBe(0);
      }, 10);
    });

    it('should handle order completion for non-existent order', () => {
      const measurement = tracker.completeOrder('non-existent', true);
      expect(measurement).toBeNull();
    });

    it('should cancel order tracking', () => {
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      tracker.cancelOrder('order-1');
      
      expect(tracker.getPendingOrdersCount()).toBe(0);
    });
  });

  describe('Latency Metrics', () => {
    beforeEach(() => {
      // Add some test data
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      tracker.completeOrder('order-1', true);
      
      tracker.startOrder('order-2', 'IB', 'MSFT', 'limit', 'sell');
      tracker.completeOrder('order-2', true);
    });

    it('should calculate venue latency metrics', () => {
      const metrics = tracker.getVenueLatencyMetrics('IB');
      
      expect(metrics.venue_name).toBe('IB');
      expect(metrics.order_execution_latency?.samples).toBe(2);
      expect(metrics.order_execution_latency?.avg_ms).toBeGreaterThan(0);
      expect(metrics.order_execution_latency?.min_ms).toBeGreaterThanOrEqual(0);
      expect(metrics.order_execution_latency?.max_ms).toBeGreaterThanOrEqual(metrics.order_execution_latency?.min_ms);
    });

    it('should return empty metrics for venue with no data', () => {
      const metrics = tracker.getVenueLatencyMetrics('NonExistent');
      
      expect(metrics.venue_name).toBe('NonExistent');
      expect(metrics.order_execution_latency?.samples).toBe(0);
      expect(metrics.order_execution_latency?.avg_ms).toBe(0);
    });

    it('should get all venue metrics', () => {
      const allMetrics = tracker.getAllVenueMetrics();
      
      expect(allMetrics.length).toBe(1); // Only IB has data
      expect(allMetrics[0].venue_name).toBe('IB');
    });

    it('should calculate average latency correctly', () => {
      const avgLatency = tracker.getAverageLatency('IB');
      expect(avgLatency).toBeGreaterThan(0);
      
      const allVenuesAvg = tracker.getAverageLatency();
      expect(allVenuesAvg).toBeGreaterThan(0);
    });
  });

  describe('Percentile Calculations', () => {
    beforeEach(() => {
      // Add orders with known latencies by mocking performance.now()
      const originalNow = performance.now;
      let mockTime = 1000;
      
      performance.now = vi.fn(() => mockTime);
      
      // Create orders with incremental latencies
      for (let i = 0; i < 100; i++) {
        mockTime = 1000;
        tracker.startOrder(`order-${i}`, 'IB', 'TEST', 'market', 'buy');
        mockTime = 1000 + (i + 1) * 10; // 10ms, 20ms, 30ms, ... 1000ms latencies
        tracker.completeOrder(`order-${i}`, true);
      }
      
      performance.now = originalNow;
    });

    it('should calculate percentiles correctly', () => {
      const metrics = tracker.getVenueLatencyMetrics('IB');
      
      expect(metrics.order_execution_latency?.samples).toBe(100);
      expect(metrics.order_execution_latency?.p50_ms).toBeCloseTo(500, 0); // 50th percentile
      expect(metrics.order_execution_latency?.p95_ms).toBeCloseTo(950, 0); // 95th percentile
      expect(metrics.order_execution_latency?.p99_ms).toBeCloseTo(990, 0); // 99th percentile
    });
  });

  describe('Callbacks and Events', () => {
    it('should trigger callbacks on order completion', () => {
      const mockCallback = vi.fn();
      tracker.onLatencyMeasurement(mockCallback);
      
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      const measurement = tracker.completeOrder('order-1', true);
      
      expect(mockCallback).toHaveBeenCalledWith(measurement);
    });

    it('should remove callbacks correctly', () => {
      const mockCallback = vi.fn();
      tracker.onLatencyMeasurement(mockCallback);
      tracker.removeLatencyCallback(mockCallback);
      
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      tracker.completeOrder('order-1', true);
      
      expect(mockCallback).not.toHaveBeenCalled();
    });
  });

  describe('Memory Management', () => {
    it('should maintain history size limit', () => {
      // Create more orders than the max history size
      const maxSize = 10000;
      
      for (let i = 0; i < maxSize + 100; i++) {
        tracker.startOrder(`order-${i}`, 'IB', 'TEST', 'market', 'buy');
        tracker.completeOrder(`order-${i}`, true);
      }
      
      const recentMeasurements = tracker.getRecentMeasurements(maxSize + 50);
      expect(recentMeasurements.length).toBeLessThanOrEqual(maxSize);
    });

    it('should clear all data', () => {
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      tracker.completeOrder('order-1', true);
      
      tracker.clear();
      
      expect(tracker.getPendingOrdersCount()).toBe(0);
      expect(tracker.getRecentMeasurements().length).toBe(0);
      expect(tracker.getAllVenueMetrics().length).toBe(0);
    });
  });

  describe('Statistics', () => {
    beforeEach(() => {
      tracker.startOrder('order-1', 'IB', 'AAPL', 'market', 'buy');
      tracker.completeOrder('order-1', true);
    });

    it('should provide comprehensive statistics', () => {
      const stats = tracker.getStatistics();
      
      expect(stats.totalMeasurements).toBe(1);
      expect(stats.pendingOrders).toBe(0);
      expect(stats.averageLatency).toBeGreaterThan(0);
      expect(stats.venues).toEqual(['IB']);
      expect(stats.oldestMeasurement).toBeInstanceOf(Date);
      expect(stats.newestMeasurement).toBeInstanceOf(Date);
    });
  });

  describe('Time Range Filtering', () => {
    beforeEach(() => {
      // Mock Date to create measurements at different times
      const originalDate = Date;
      let mockTime = Date.now() - 3600000; // 1 hour ago
      
      global.Date = vi.fn(() => new originalDate(mockTime)) as any;
      global.Date.now = vi.fn(() => mockTime);
      
      // Add old measurement
      tracker.startOrder('order-old', 'IB', 'AAPL', 'market', 'buy');
      tracker.completeOrder('order-old', true);
      
      // Add recent measurement
      mockTime = Date.now() - 300000; // 5 minutes ago
      tracker.startOrder('order-recent', 'IB', 'MSFT', 'market', 'buy');
      tracker.completeOrder('order-recent', true);
      
      global.Date = originalDate;
    });

    it('should filter metrics by time range', () => {
      const recentMetrics = tracker.getVenueLatencyMetrics('IB', 600000); // Last 10 minutes
      const allMetrics = tracker.getVenueLatencyMetrics('IB');
      
      expect(recentMetrics.order_execution_latency?.samples).toBe(1); // Only recent order
      expect(allMetrics.order_execution_latency?.samples).toBe(2); // Both orders
    });
  });
});