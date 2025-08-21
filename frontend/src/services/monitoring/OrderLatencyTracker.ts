/**
 * Order Latency Tracker
 * Tracks and measures order execution timing for latency analysis
 */

import { LatencyMetrics } from '../../types/monitoring';

export interface OrderExecution {
  orderId: string;
  venue: string;
  symbol: string;
  startTime: number;
  endTime?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'failed';
  orderType: 'market' | 'limit' | 'stop';
  side: 'buy' | 'sell';
}

export interface LatencyMeasurement {
  orderId: string;
  venue: string;
  latency_ms: number;
  timestamp: Date;
  orderType: string;
  success: boolean;
}

export class OrderLatencyTracker {
  private pendingOrders: Map<string, OrderExecution> = new Map();
  private latencyHistory: LatencyMeasurement[] = [];
  private maxHistorySize: number = 10000;
  private callbackHandlers: ((measurement: LatencyMeasurement) => void)[] = [];

  /**
   * Start tracking an order execution
   */
  startOrder(orderId: string, venue: string, symbol: string, orderType: 'market' | 'limit' | 'stop', side: 'buy' | 'sell'): void {
    const startTime = performance.now();
    
    const execution: OrderExecution = {
      orderId,
      venue,
      symbol,
      startTime,
      status: 'pending',
      orderType,
      side
    };

    this.pendingOrders.set(orderId, execution);
  }

  /**
   * Complete order tracking and calculate latency
   */
  completeOrder(orderId: string, success: boolean = true): LatencyMeasurement | null {
    const execution = this.pendingOrders.get(orderId);
    if (!execution) {
      console.warn(`OrderLatencyTracker: Order ${orderId} not found in pending orders`);
      return null;
    }

    const endTime = performance.now();
    const latency_ms = endTime - execution.startTime;

    execution.endTime = endTime;
    execution.status = success ? 'filled' : 'failed';

    const measurement: LatencyMeasurement = {
      orderId,
      venue: execution.venue,
      latency_ms,
      timestamp: new Date(),
      orderType: execution.orderType,
      success
    };

    // Add to history
    this.latencyHistory.push(measurement);
    
    // Maintain history size limit
    if (this.latencyHistory.length > this.maxHistorySize) {
      this.latencyHistory.shift();
    }

    // Remove from pending orders
    this.pendingOrders.delete(orderId);

    // Notify callbacks
    this.callbackHandlers.forEach(callback => callback(measurement));

    return measurement;
  }

  /**
   * Cancel order tracking
   */
  cancelOrder(orderId: string): void {
    const execution = this.pendingOrders.get(orderId);
    if (execution) {
      execution.status = 'cancelled';
      this.pendingOrders.delete(orderId);
    }
  }

  /**
   * Get latency metrics for a specific venue
   */
  getVenueLatencyMetrics(venue: string, timeRangeMs?: number): Partial<LatencyMetrics> {
    const now = Date.now();
    const cutoffTime = timeRangeMs ? now - timeRangeMs : 0;

    const venueLatencies = this.latencyHistory
      .filter(measurement => 
        measurement.venue === venue && 
        measurement.timestamp.getTime() > cutoffTime &&
        measurement.success
      )
      .map(m => m.latency_ms);

    if (venueLatencies.length === 0) {
      return {
        venue_name: venue,
        order_execution_latency: {
          min_ms: 0,
          max_ms: 0,
          avg_ms: 0,
          p50_ms: 0,
          p95_ms: 0,
          p99_ms: 0,
          samples: 0
        },
        last_updated: new Date().toISOString()
      };
    }

    const sortedLatencies = venueLatencies.sort((a, b) => a - b);
    const min_ms = sortedLatencies[0];
    const max_ms = sortedLatencies[sortedLatencies.length - 1];
    const avg_ms = sortedLatencies.reduce((sum, val) => sum + val, 0) / sortedLatencies.length;
    
    const p50_ms = this.calculatePercentile(sortedLatencies, 0.5);
    const p95_ms = this.calculatePercentile(sortedLatencies, 0.95);
    const p99_ms = this.calculatePercentile(sortedLatencies, 0.99);

    return {
      venue_name: venue,
      order_execution_latency: {
        min_ms: Math.round(min_ms * 100) / 100,
        max_ms: Math.round(max_ms * 100) / 100,
        avg_ms: Math.round(avg_ms * 100) / 100,
        p50_ms: Math.round(p50_ms * 100) / 100,
        p95_ms: Math.round(p95_ms * 100) / 100,
        p99_ms: Math.round(p99_ms * 100) / 100,
        samples: venueLatencies.length
      },
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Get all venue latency metrics
   */
  getAllVenueMetrics(timeRangeMs?: number): Partial<LatencyMetrics>[] {
    const venues = [...new Set(this.latencyHistory.map(m => m.venue))];
    return venues.map(venue => this.getVenueLatencyMetrics(venue, timeRangeMs));
  }

  /**
   * Get recent latency measurements
   */
  getRecentMeasurements(count: number = 100): LatencyMeasurement[] {
    return this.latencyHistory.slice(-count);
  }

  /**
   * Get pending orders count
   */
  getPendingOrdersCount(): number {
    return this.pendingOrders.size;
  }

  /**
   * Get pending orders by venue
   */
  getPendingOrdersByVenue(): Map<string, number> {
    const venueCount = new Map<string, number>();
    
    this.pendingOrders.forEach(order => {
      const currentCount = venueCount.get(order.venue) || 0;
      venueCount.set(order.venue, currentCount + 1);
    });

    return venueCount;
  }

  /**
   * Add callback for new latency measurements
   */
  onLatencyMeasurement(callback: (measurement: LatencyMeasurement) => void): void {
    this.callbackHandlers.push(callback);
  }

  /**
   * Remove callback
   */
  removeLatencyCallback(callback: (measurement: LatencyMeasurement) => void): void {
    const index = this.callbackHandlers.indexOf(callback);
    if (index > -1) {
      this.callbackHandlers.splice(index, 1);
    }
  }

  /**
   * Clear all tracking data
   */
  clear(): void {
    this.pendingOrders.clear();
    this.latencyHistory = [];
  }

  /**
   * Get average latency for a venue over time period
   */
  getAverageLatency(venue?: string, timeRangeMs?: number): number {
    const now = Date.now();
    const cutoffTime = timeRangeMs ? now - timeRangeMs : 0;

    const relevantMeasurements = this.latencyHistory
      .filter(measurement => 
        (!venue || measurement.venue === venue) &&
        measurement.timestamp.getTime() > cutoffTime &&
        measurement.success
      );

    if (relevantMeasurements.length === 0) return 0;

    const totalLatency = relevantMeasurements.reduce((sum, m) => sum + m.latency_ms, 0);
    return Math.round((totalLatency / relevantMeasurements.length) * 100) / 100;
  }

  /**
   * Calculate percentile from sorted array
   */
  private calculatePercentile(sortedArray: number[], percentile: number): number {
    if (sortedArray.length === 0) return 0;
    
    const index = percentile * (sortedArray.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;

    if (upper >= sortedArray.length) return sortedArray[sortedArray.length - 1];
    
    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  /**
   * Clean up old measurements beyond max history
   */
  private maintainHistoryLimit(): void {
    if (this.latencyHistory.length > this.maxHistorySize) {
      this.latencyHistory = this.latencyHistory.slice(-this.maxHistorySize);
    }
  }

  /**
   * Get statistics summary
   */
  getStatistics(): {
    totalMeasurements: number;
    pendingOrders: number;
    averageLatency: number;
    venues: string[];
    oldestMeasurement?: Date;
    newestMeasurement?: Date;
  } {
    return {
      totalMeasurements: this.latencyHistory.length,
      pendingOrders: this.pendingOrders.size,
      averageLatency: this.getAverageLatency(),
      venues: [...new Set(this.latencyHistory.map(m => m.venue))],
      oldestMeasurement: this.latencyHistory[0]?.timestamp,
      newestMeasurement: this.latencyHistory[this.latencyHistory.length - 1]?.timestamp
    };
  }
}

// Singleton instance for global order latency tracking
export const orderLatencyTracker = new OrderLatencyTracker();