/**
 * Market Data Latency Monitor
 * Monitors latency for market data feeds from various venues
 */

import { LatencyMetrics } from '../../types/monitoring';

export interface MarketDataTick {
  symbol: string;
  venue: string;
  timestamp: number;
  price: number;
  volume: number;
  receivedAt: number;
}

export interface MarketDataLatencyMeasurement {
  symbol: string;
  venue: string;
  tick_to_trade_ms: number;
  feed_latency_ms: number;
  processing_latency_ms: number;
  total_latency_ms: number;
  timestamp: Date;
}

export interface LatencyBenchmark {
  venue: string;
  baseline_feed_latency_ms: number;
  baseline_processing_latency_ms: number;
  last_calibrated: Date;
}

export class MarketDataLatencyMonitor {
  private latencyMeasurements: MarketDataLatencyMeasurement[] = [];
  private benchmarks: Map<string, LatencyBenchmark> = new Map();
  private processingStartTimes: Map<string, number> = new Map();
  private maxMeasurements: number = 50000;
  private callbacks: ((measurement: MarketDataLatencyMeasurement) => void)[] = [];

  /**
   * Record when market data processing starts
   */
  startProcessing(symbol: string, venue: string): void {
    const key = `${venue}:${symbol}`;
    this.processingStartTimes.set(key, performance.now());
  }

  /**
   * Record market data tick and calculate latencies
   */
  recordTick(tick: MarketDataTick, tradeExecutionTime?: number): MarketDataLatencyMeasurement {
    const processingEndTime = performance.now();
    const key = `${tick.venue}:${tick.symbol}`;
    const processingStartTime = this.processingStartTimes.get(key) || processingEndTime;
    
    // Calculate various latency components
    const feed_latency_ms = tick.receivedAt - tick.timestamp;
    const processing_latency_ms = processingEndTime - processingStartTime;
    const tick_to_trade_ms = tradeExecutionTime ? tradeExecutionTime - tick.timestamp : 0;
    const total_latency_ms = feed_latency_ms + processing_latency_ms;

    const measurement: MarketDataLatencyMeasurement = {
      symbol: tick.symbol,
      venue: tick.venue,
      tick_to_trade_ms: Math.max(0, tick_to_trade_ms),
      feed_latency_ms: Math.max(0, feed_latency_ms),
      processing_latency_ms: Math.max(0, processing_latency_ms),
      total_latency_ms: Math.max(0, total_latency_ms),
      timestamp: new Date()
    };

    // Add to measurements
    this.latencyMeasurements.push(measurement);
    this.maintainMeasurementLimit();

    // Clean up processing start time
    this.processingStartTimes.delete(key);

    // Notify callbacks
    this.callbacks.forEach(callback => callback(measurement));

    return measurement;
  }

  /**
   * Get market data latency metrics for a venue
   */
  getMarketDataLatencyMetrics(venue: string, timeRangeMs?: number): Partial<LatencyMetrics> {
    const now = Date.now();
    const cutoffTime = timeRangeMs ? now - timeRangeMs : 0;

    const venueMeasurements = this.latencyMeasurements.filter(m => 
      m.venue === venue && 
      m.timestamp.getTime() > cutoffTime
    );

    if (venueMeasurements.length === 0) {
      return {
        venue_name: venue,
        market_data_latency: {
          tick_to_trade_ms: 0,
          feed_latency_ms: 0,
          processing_latency_ms: 0,
          total_latency_ms: 0
        },
        last_updated: new Date().toISOString()
      };
    }

    // Calculate averages
    const avgTickToTrade = this.calculateAverage(venueMeasurements.map(m => m.tick_to_trade_ms));
    const avgFeedLatency = this.calculateAverage(venueMeasurements.map(m => m.feed_latency_ms));
    const avgProcessingLatency = this.calculateAverage(venueMeasurements.map(m => m.processing_latency_ms));
    const avgTotalLatency = this.calculateAverage(venueMeasurements.map(m => m.total_latency_ms));

    return {
      venue_name: venue,
      market_data_latency: {
        tick_to_trade_ms: Math.round(avgTickToTrade * 100) / 100,
        feed_latency_ms: Math.round(avgFeedLatency * 100) / 100,
        processing_latency_ms: Math.round(avgProcessingLatency * 100) / 100,
        total_latency_ms: Math.round(avgTotalLatency * 100) / 100
      },
      last_updated: new Date().toISOString()
    };
  }

  /**
   * Get latency breakdown by component
   */
  getLatencyBreakdown(venue?: string, timeRangeMs?: number): {
    feed_latency: number[];
    processing_latency: number[];
    total_latency: number[];
    tick_to_trade: number[];
  } {
    const now = Date.now();
    const cutoffTime = timeRangeMs ? now - timeRangeMs : 0;

    const filteredMeasurements = this.latencyMeasurements.filter(m =>
      (!venue || m.venue === venue) &&
      m.timestamp.getTime() > cutoffTime
    );

    return {
      feed_latency: filteredMeasurements.map(m => m.feed_latency_ms),
      processing_latency: filteredMeasurements.map(m => m.processing_latency_ms),
      total_latency: filteredMeasurements.map(m => m.total_latency_ms),
      tick_to_trade: filteredMeasurements.map(m => m.tick_to_trade_ms)
    };
  }

  /**
   * Set baseline latency benchmark for a venue
   */
  setBenchmark(venue: string, feedLatencyMs: number, processingLatencyMs: number): void {
    this.benchmarks.set(venue, {
      venue,
      baseline_feed_latency_ms: feedLatencyMs,
      baseline_processing_latency_ms: processingLatencyMs,
      last_calibrated: new Date()
    });
  }

  /**
   * Auto-calibrate benchmark based on recent measurements
   */
  autoCalibrateBenchmark(venue: string, sampleSize: number = 1000): LatencyBenchmark | null {
    const recentMeasurements = this.latencyMeasurements
      .filter(m => m.venue === venue)
      .slice(-sampleSize);

    if (recentMeasurements.length < 100) {
      console.warn(`MarketDataLatencyMonitor: Insufficient data to calibrate benchmark for ${venue}`);
      return null;
    }

    const feedLatencies = recentMeasurements.map(m => m.feed_latency_ms);
    const processingLatencies = recentMeasurements.map(m => m.processing_latency_ms);

    // Use 25th percentile as baseline (better than average for most normal conditions)
    const baselineFeedLatency = this.calculatePercentile(feedLatencies.sort((a, b) => a - b), 0.25);
    const baselineProcessingLatency = this.calculatePercentile(processingLatencies.sort((a, b) => a - b), 0.25);

    const benchmark: LatencyBenchmark = {
      venue,
      baseline_feed_latency_ms: Math.round(baselineFeedLatency * 100) / 100,
      baseline_processing_latency_ms: Math.round(baselineProcessingLatency * 100) / 100,
      last_calibrated: new Date()
    };

    this.benchmarks.set(venue, benchmark);
    return benchmark;
  }

  /**
   * Get current latency vs benchmark comparison
   */
  getLatencyComparison(venue: string, timeRangeMs: number = 60000): {
    current_vs_baseline: {
      feed_latency_ratio: number;
      processing_latency_ratio: number;
      feed_degradation_percent: number;
      processing_degradation_percent: number;
    };
    status: 'normal' | 'degraded' | 'critical';
    recommendation?: string;
  } {
    const benchmark = this.benchmarks.get(venue);
    const currentMetrics = this.getMarketDataLatencyMetrics(venue, timeRangeMs);

    if (!benchmark || !currentMetrics.market_data_latency) {
      return {
        current_vs_baseline: {
          feed_latency_ratio: 1,
          processing_latency_ratio: 1,
          feed_degradation_percent: 0,
          processing_degradation_percent: 0
        },
        status: 'normal'
      };
    }

    const feedRatio = currentMetrics.market_data_latency.feed_latency_ms / benchmark.baseline_feed_latency_ms;
    const processingRatio = currentMetrics.market_data_latency.processing_latency_ms / benchmark.baseline_processing_latency_ms;
    
    const feedDegradation = ((feedRatio - 1) * 100);
    const processingDegradation = ((processingRatio - 1) * 100);

    let status: 'normal' | 'degraded' | 'critical' = 'normal';
    let recommendation: string | undefined;

    if (feedRatio > 2 || processingRatio > 2) {
      status = 'critical';
      recommendation = 'Immediate investigation required - latency more than 2x baseline';
    } else if (feedRatio > 1.5 || processingRatio > 1.5) {
      status = 'degraded';
      recommendation = 'Monitor closely - latency degradation detected';
    }

    return {
      current_vs_baseline: {
        feed_latency_ratio: Math.round(feedRatio * 100) / 100,
        processing_latency_ratio: Math.round(processingRatio * 100) / 100,
        feed_degradation_percent: Math.round(feedDegradation * 100) / 100,
        processing_degradation_percent: Math.round(processingDegradation * 100) / 100
      },
      status,
      recommendation
    };
  }

  /**
   * Get recent measurements for a symbol/venue
   */
  getRecentMeasurements(symbol?: string, venue?: string, count: number = 100): MarketDataLatencyMeasurement[] {
    let filtered = this.latencyMeasurements;
    
    if (symbol) filtered = filtered.filter(m => m.symbol === symbol);
    if (venue) filtered = filtered.filter(m => m.venue === venue);
    
    return filtered.slice(-count);
  }

  /**
   * Get all venues being monitored
   */
  getMonitoredVenues(): string[] {
    return [...new Set(this.latencyMeasurements.map(m => m.venue))];
  }

  /**
   * Get all symbols being monitored for a venue
   */
  getMonitoredSymbols(venue?: string): string[] {
    const filtered = venue ? 
      this.latencyMeasurements.filter(m => m.venue === venue) : 
      this.latencyMeasurements;
    
    return [...new Set(filtered.map(m => m.symbol))];
  }

  /**
   * Add callback for new latency measurements
   */
  onLatencyMeasurement(callback: (measurement: MarketDataLatencyMeasurement) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: (measurement: MarketDataLatencyMeasurement) => void): void {
    const index = this.callbacks.indexOf(callback);
    if (index > -1) {
      this.callbacks.splice(index, 1);
    }
  }

  /**
   * Clear all measurements and benchmarks
   */
  clear(): void {
    this.latencyMeasurements = [];
    this.benchmarks.clear();
    this.processingStartTimes.clear();
  }

  /**
   * Get performance statistics
   */
  getStatistics(): {
    totalMeasurements: number;
    uniqueVenues: number;
    uniqueSymbols: number;
    benchmarksSet: number;
    averageLatencyAllVenues: number;
    oldestMeasurement?: Date;
    newestMeasurement?: Date;
  } {
    const venues = new Set(this.latencyMeasurements.map(m => m.venue));
    const symbols = new Set(this.latencyMeasurements.map(m => m.symbol));
    
    const avgLatency = this.latencyMeasurements.length > 0 ?
      this.latencyMeasurements.reduce((sum, m) => sum + m.total_latency_ms, 0) / this.latencyMeasurements.length :
      0;

    return {
      totalMeasurements: this.latencyMeasurements.length,
      uniqueVenues: venues.size,
      uniqueSymbols: symbols.size,
      benchmarksSet: this.benchmarks.size,
      averageLatencyAllVenues: Math.round(avgLatency * 100) / 100,
      oldestMeasurement: this.latencyMeasurements[0]?.timestamp,
      newestMeasurement: this.latencyMeasurements[this.latencyMeasurements.length - 1]?.timestamp
    };
  }

  /**
   * Calculate average of number array
   */
  private calculateAverage(numbers: number[]): number {
    if (numbers.length === 0) return 0;
    return numbers.reduce((sum, val) => sum + val, 0) / numbers.length;
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
   * Maintain measurement count limit
   */
  private maintainMeasurementLimit(): void {
    if (this.latencyMeasurements.length > this.maxMeasurements) {
      this.latencyMeasurements = this.latencyMeasurements.slice(-this.maxMeasurements);
    }
  }
}

// Singleton instance for global market data latency monitoring
export const marketDataLatencyMonitor = new MarketDataLatencyMonitor();