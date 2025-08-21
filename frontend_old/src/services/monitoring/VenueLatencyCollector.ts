/**
 * Venue Latency Collector
 * Collects and aggregates latency measurements specific to trading venues
 */

import { LatencyMetrics } from '../../types/monitoring';
import { OrderLatencyTracker, orderLatencyTracker } from './OrderLatencyTracker';
import { MarketDataLatencyMonitor, marketDataLatencyMonitor } from './MarketDataLatencyMonitor';

export interface VenueLatencySnapshot {
  venue: string;
  timestamp: Date;
  ping_ms: number;
  connection_test_ms: number;
  api_response_time_ms: number;
  order_ack_time_ms: number;
  market_data_delay_ms: number;
}

export interface VenueHealthScore {
  venue: string;
  overall_score: number; // 0-100
  latency_score: number;
  reliability_score: number;
  consistency_score: number;
  last_updated: Date;
}

export interface ConnectionLatencyTest {
  venue: string;
  test_type: 'ping' | 'api_call' | 'order_test' | 'market_data';
  start_time: number;
  end_time?: number;
  success: boolean;
  error_message?: string;
  latency_ms?: number;
}

export class VenueLatencyCollector {
  private latencySnapshots: VenueLatencySnapshot[] = [];
  private connectionTests: Map<string, ConnectionLatencyTest> = new Map();
  private healthScores: Map<string, VenueHealthScore> = new Map();
  private maxSnapshots: number = 10000;
  private testTimeouts: Map<string, NodeJS.Timeout> = new Map();
  private callbacks: ((snapshot: VenueLatencySnapshot) => void)[] = [];

  constructor(
    private orderTracker: OrderLatencyTracker = orderLatencyTracker,
    private marketDataMonitor: MarketDataLatencyMonitor = marketDataLatencyMonitor
  ) {}

  /**
   * Start a connection latency test
   */
  startConnectionTest(venue: string, testType: 'ping' | 'api_call' | 'order_test' | 'market_data'): string {
    const testId = `${venue}_${testType}_${Date.now()}`;
    const test: ConnectionLatencyTest = {
      venue,
      test_type: testType,
      start_time: performance.now(),
      success: false
    };

    this.connectionTests.set(testId, test);

    // Set timeout for test (30 seconds)
    const timeout = setTimeout(() => {
      this.completeConnectionTest(testId, false, 'Test timeout');
    }, 30000);

    this.testTimeouts.set(testId, timeout);

    return testId;
  }

  /**
   * Complete a connection latency test
   */
  completeConnectionTest(testId: string, success: boolean, errorMessage?: string): ConnectionLatencyTest | null {
    const test = this.connectionTests.get(testId);
    if (!test) {
      console.warn(`VenueLatencyCollector: Test ${testId} not found`);
      return null;
    }

    test.end_time = performance.now();
    test.success = success;
    test.error_message = errorMessage;
    test.latency_ms = test.end_time - test.start_time;

    // Clear timeout
    const timeout = this.testTimeouts.get(testId);
    if (timeout) {
      clearTimeout(timeout);
      this.testTimeouts.delete(testId);
    }

    // Update venue health score
    this.updateVenueHealthScore(test.venue);

    this.connectionTests.delete(testId);
    return test;
  }

  /**
   * Collect comprehensive latency snapshot for a venue
   */
  async collectVenueSnapshot(venue: string): Promise<VenueLatencySnapshot> {
    const timestamp = new Date();

    // Run concurrent latency tests
    const pingTestId = this.startConnectionTest(venue, 'ping');
    const apiTestId = this.startConnectionTest(venue, 'api_call');
    
    // Wait for basic connectivity tests (with timeout)
    await new Promise(resolve => setTimeout(resolve, 100)); // Brief delay for initial measurements

    // Get current measurements from other trackers
    const orderMetrics = this.orderTracker.getVenueLatencyMetrics(venue, 60000); // Last minute
    const marketDataMetrics = this.marketDataMonitor.getMarketDataLatencyMetrics(venue, 60000);

    // Complete tests and get results
    this.completeConnectionTest(pingTestId, true);
    this.completeConnectionTest(apiTestId, true);

    const snapshot: VenueLatencySnapshot = {
      venue,
      timestamp,
      ping_ms: this.getRecentTestLatency(venue, 'ping') || 0,
      connection_test_ms: this.getRecentTestLatency(venue, 'api_call') || 0,
      api_response_time_ms: this.getRecentTestLatency(venue, 'api_call') || 0,
      order_ack_time_ms: orderMetrics.order_execution_latency?.avg_ms || 0,
      market_data_delay_ms: marketDataMetrics.market_data_latency?.feed_latency_ms || 0
    };

    // Store snapshot
    this.latencySnapshots.push(snapshot);
    this.maintainSnapshotLimit();

    // Update health score
    this.updateVenueHealthScore(venue);

    // Notify callbacks
    this.callbacks.forEach(callback => callback(snapshot));

    return snapshot;
  }

  /**
   * Get comprehensive latency metrics for venue
   */
  getVenueLatencyMetrics(venue: string, timeRangeMs?: number): LatencyMetrics {
    const now = Date.now();
    const cutoffTime = timeRangeMs ? now - timeRangeMs : 0;

    // Get snapshots in time range
    const recentSnapshots = this.latencySnapshots.filter(s => 
      s.venue === venue && 
      s.timestamp.getTime() > cutoffTime
    );

    // Get order execution metrics
    const orderMetrics = this.orderTracker.getVenueLatencyMetrics(venue, timeRangeMs);
    
    // Get market data metrics
    const marketDataMetrics = this.marketDataMonitor.getMarketDataLatencyMetrics(venue, timeRangeMs);

    // Calculate connection latency stats
    const connectionLatencies = {
      ping_ms: this.calculateStats(recentSnapshots.map(s => s.ping_ms)),
      jitter_ms: this.calculateJitter(recentSnapshots.map(s => s.ping_ms)),
      packet_loss_percent: this.calculatePacketLoss(venue, timeRangeMs)
    };

    return {
      venue_name: venue,
      order_execution_latency: orderMetrics.order_execution_latency || {
        min_ms: 0, max_ms: 0, avg_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0, samples: 0
      },
      market_data_latency: marketDataMetrics.market_data_latency || {
        tick_to_trade_ms: 0, feed_latency_ms: 0, processing_latency_ms: 0, total_latency_ms: 0
      },
      connection_latency: {
        ping_ms: connectionLatencies.ping_ms.avg || 0,
        jitter_ms: connectionLatencies.jitter_ms,
        packet_loss_percent: connectionLatencies.packet_loss_percent
      },
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Get venue health score
   */
  getVenueHealthScore(venue: string): VenueHealthScore | null {
    return this.healthScores.get(venue) || null;
  }

  /**
   * Get all venue health scores
   */
  getAllVenueHealthScores(): VenueHealthScore[] {
    return Array.from(this.healthScores.values());
  }

  /**
   * Compare venues by performance
   */
  compareVenues(timeRangeMs: number = 3600000): {
    venue_ranking: {
      venue: string;
      overall_latency_ms: number;
      health_score: number;
      rank: number;
    }[];
    best_venue: string;
    worst_venue: string;
    performance_gap_ms: number;
  } {
    const venues = [...new Set(this.latencySnapshots.map(s => s.venue))];
    
    const venuePerformance = venues.map(venue => {
      const metrics = this.getVenueLatencyMetrics(venue, timeRangeMs);
      const healthScore = this.getVenueHealthScore(venue);
      
      // Calculate overall latency (weighted average of different latency types)
      const orderLatency = metrics.order_execution_latency.avg_ms || 0;
      const marketDataLatency = metrics.market_data_latency.total_latency_ms || 0;
      const connectionLatency = metrics.connection_latency.ping_ms || 0;
      
      const overall_latency_ms = (orderLatency * 0.5) + (marketDataLatency * 0.3) + (connectionLatency * 0.2);
      
      return {
        venue,
        overall_latency_ms: Math.round(overall_latency_ms * 100) / 100,
        health_score: healthScore?.overall_score || 0
      };
    });

    // Sort by overall latency (lower is better)
    const ranked = venuePerformance.sort((a, b) => a.overall_latency_ms - b.overall_latency_ms);
    
    const venue_ranking = ranked.map((item, index) => ({
      ...item,
      rank: index + 1
    }));

    const best_venue = ranked[0]?.venue || '';
    const worst_venue = ranked[ranked.length - 1]?.venue || '';
    const performance_gap_ms = (ranked[ranked.length - 1]?.overall_latency_ms || 0) - (ranked[0]?.overall_latency_ms || 0);

    return {
      venue_ranking,
      best_venue,
      worst_venue,
      performance_gap_ms: Math.round(performance_gap_ms * 100) / 100
    };
  }

  /**
   * Get recent snapshots for a venue
   */
  getRecentSnapshots(venue: string, count: number = 50): VenueLatencySnapshot[] {
    return this.latencySnapshots
      .filter(s => s.venue === venue)
      .slice(-count);
  }

  /**
   * Get performance trend for venue
   */
  getPerformanceTrend(venue: string, timeRangeMs: number = 3600000): {
    trend: 'improving' | 'degrading' | 'stable';
    change_percent: number;
    confidence: number;
  } {
    const now = Date.now();
    const halfRange = timeRangeMs / 2;
    
    // Split time range in half
    const olderSnapshots = this.latencySnapshots.filter(s => 
      s.venue === venue && 
      s.timestamp.getTime() > (now - timeRangeMs) &&
      s.timestamp.getTime() <= (now - halfRange)
    );
    
    const newerSnapshots = this.latencySnapshots.filter(s => 
      s.venue === venue && 
      s.timestamp.getTime() > (now - halfRange)
    );

    if (olderSnapshots.length < 10 || newerSnapshots.length < 10) {
      return { trend: 'stable', change_percent: 0, confidence: 0 };
    }

    const olderAvgLatency = this.calculateAverageLatency(olderSnapshots);
    const newerAvgLatency = this.calculateAverageLatency(newerSnapshots);
    
    const change_percent = ((newerAvgLatency - olderAvgLatency) / olderAvgLatency) * 100;
    const confidence = Math.min(olderSnapshots.length + newerSnapshots.length, 100) / 100;

    let trend: 'improving' | 'degrading' | 'stable' = 'stable';
    if (Math.abs(change_percent) > 5) { // 5% threshold for trend detection
      trend = change_percent > 0 ? 'degrading' : 'improving';
    }

    return {
      trend,
      change_percent: Math.round(change_percent * 100) / 100,
      confidence: Math.round(confidence * 100) / 100
    };
  }

  /**
   * Add callback for new snapshots
   */
  onSnapshot(callback: (snapshot: VenueLatencySnapshot) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: (snapshot: VenueLatencySnapshot) => void): void {
    const index = this.callbacks.indexOf(callback);
    if (index > -1) {
      this.callbacks.splice(index, 1);
    }
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.latencySnapshots = [];
    this.connectionTests.clear();
    this.healthScores.clear();
    
    // Clear timeouts
    this.testTimeouts.forEach(timeout => clearTimeout(timeout));
    this.testTimeouts.clear();
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalSnapshots: number;
    activeTests: number;
    monitoredVenues: number;
    averageLatencyAllVenues: number;
  } {
    const venues = new Set(this.latencySnapshots.map(s => s.venue));
    const avgLatency = this.latencySnapshots.length > 0 ?
      this.latencySnapshots.reduce((sum, s) => sum + this.calculateAverageLatency([s]), 0) / this.latencySnapshots.length :
      0;

    return {
      totalSnapshots: this.latencySnapshots.length,
      activeTests: this.connectionTests.size,
      monitoredVenues: venues.size,
      averageLatencyAllVenues: Math.round(avgLatency * 100) / 100
    };
  }

  private getRecentTestLatency(venue: string, testType: string): number | null {
    // This would integrate with actual network testing - for now return simulated values
    return Math.random() * 50 + 10; // 10-60ms range
  }

  private calculateStats(values: number[]): { min: number; max: number; avg: number } {
    if (values.length === 0) return { min: 0, max: 0, avg: 0 };
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    return { min, max, avg: Math.round(avg * 100) / 100 };
  }

  private calculateJitter(pingValues: number[]): number {
    if (pingValues.length < 2) return 0;
    
    let totalVariation = 0;
    for (let i = 1; i < pingValues.length; i++) {
      totalVariation += Math.abs(pingValues[i] - pingValues[i-1]);
    }
    
    return Math.round((totalVariation / (pingValues.length - 1)) * 100) / 100;
  }

  private calculatePacketLoss(venue: string, timeRangeMs?: number): number {
    // This would integrate with actual network monitoring
    // For now, return simulated packet loss percentage
    return Math.random() * 0.5; // 0-0.5% packet loss
  }

  private calculateAverageLatency(snapshots: VenueLatencySnapshot[]): number {
    if (snapshots.length === 0) return 0;
    
    const totalLatency = snapshots.reduce((sum, snapshot) => {
      return sum + (snapshot.ping_ms + snapshot.api_response_time_ms + snapshot.order_ack_time_ms) / 3;
    }, 0);
    
    return totalLatency / snapshots.length;
  }

  private updateVenueHealthScore(venue: string): void {
    const recentSnapshots = this.getRecentSnapshots(venue, 100);
    if (recentSnapshots.length < 10) return;

    // Calculate different score components
    const latencyScore = this.calculateLatencyScore(recentSnapshots);
    const reliabilityScore = this.calculateReliabilityScore(venue);
    const consistencyScore = this.calculateConsistencyScore(recentSnapshots);

    // Weighted overall score
    const overall_score = (latencyScore * 0.4) + (reliabilityScore * 0.3) + (consistencyScore * 0.3);

    const healthScore: VenueHealthScore = {
      venue,
      overall_score: Math.round(overall_score),
      latency_score: Math.round(latencyScore),
      reliability_score: Math.round(reliabilityScore),
      consistency_score: Math.round(consistencyScore),
      last_updated: new Date()
    };

    this.healthScores.set(venue, healthScore);
  }

  private calculateLatencyScore(snapshots: VenueLatencySnapshot[]): number {
    const avgLatency = this.calculateAverageLatency(snapshots);
    
    // Score based on latency ranges (100 = excellent, 0 = poor)
    if (avgLatency <= 10) return 100;
    if (avgLatency <= 25) return 90;
    if (avgLatency <= 50) return 75;
    if (avgLatency <= 100) return 60;
    if (avgLatency <= 200) return 40;
    if (avgLatency <= 500) return 20;
    return 0;
  }

  private calculateReliabilityScore(venue: string): number {
    // This would be based on connection success rates, uptime, etc.
    // For now, return a simulated score
    return 85 + Math.random() * 15; // 85-100 range
  }

  private calculateConsistencyScore(snapshots: VenueLatencySnapshot[]): number {
    if (snapshots.length < 10) return 50;
    
    const latencies = snapshots.map(s => this.calculateAverageLatency([s]));
    const mean = latencies.reduce((sum, val) => sum + val, 0) / latencies.length;
    const variance = latencies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / latencies.length;
    const stdDev = Math.sqrt(variance);
    
    // Lower standard deviation = higher consistency score
    const coefficientOfVariation = stdDev / mean;
    return Math.max(0, 100 - (coefficientOfVariation * 200)); // Scale coefficient of variation to 0-100 score
  }

  private maintainSnapshotLimit(): void {
    if (this.latencySnapshots.length > this.maxSnapshots) {
      this.latencySnapshots = this.latencySnapshots.slice(-this.maxSnapshots);
    }
  }
}

// Singleton instance for global venue latency collection
export const venueLatencyCollector = new VenueLatencyCollector();