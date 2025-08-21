/**
 * Latency Percentile Calculator
 * Provides accurate P50/P95/P99 percentile calculations for latency analysis
 */

export interface PercentileResult {
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  p99_9: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  std_deviation: number;
  sample_count: number;
}

export interface HistogramBucket {
  min: number;
  max: number;
  count: number;
  percentage: number;
}

export interface LatencyDistribution {
  buckets: HistogramBucket[];
  percentiles: PercentileResult;
  outliers: number[];
  outlier_threshold: number;
}

export class LatencyPercentileCalculator {
  private static readonly PERCENTILES = [50, 90, 95, 99, 99.9];
  private static readonly HISTOGRAM_BUCKETS = 50;

  /**
   * Calculate comprehensive percentile statistics
   */
  static calculatePercentiles(latencies: number[]): PercentileResult {
    if (latencies.length === 0) {
      return {
        p50: 0, p90: 0, p95: 0, p99: 0, p99_9: 0,
        min: 0, max: 0, mean: 0, median: 0,
        std_deviation: 0, sample_count: 0
      };
    }

    // Sort the array for percentile calculation
    const sorted = [...latencies].sort((a, b) => a - b);
    const count = sorted.length;

    // Basic statistics
    const min = sorted[0];
    const max = sorted[count - 1];
    const mean = latencies.reduce((sum, val) => sum + val, 0) / count;
    const median = this.calculatePercentile(sorted, 50);

    // Standard deviation
    const variance = latencies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / count;
    const std_deviation = Math.sqrt(variance);

    // Calculate percentiles
    const p50 = this.calculatePercentile(sorted, 50);
    const p90 = this.calculatePercentile(sorted, 90);
    const p95 = this.calculatePercentile(sorted, 95);
    const p99 = this.calculatePercentile(sorted, 99);
    const p99_9 = this.calculatePercentile(sorted, 99.9);

    return {
      p50: this.round(p50),
      p90: this.round(p90),
      p95: this.round(p95),
      p99: this.round(p99),
      p99_9: this.round(p99_9),
      min: this.round(min),
      max: this.round(max),
      mean: this.round(mean),
      median: this.round(median),
      std_deviation: this.round(std_deviation),
      sample_count: count
    };
  }

  /**
   * Calculate a specific percentile from sorted array
   */
  static calculatePercentile(sortedArray: number[], percentile: number): number {
    if (sortedArray.length === 0) return 0;
    if (percentile <= 0) return sortedArray[0];
    if (percentile >= 100) return sortedArray[sortedArray.length - 1];

    const index = (percentile / 100) * (sortedArray.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    if (upper >= sortedArray.length) {
      return sortedArray[sortedArray.length - 1];
    }

    return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
  }

  /**
   * Calculate percentiles using interpolation methods
   */
  static calculatePercentilesLinearInterpolation(latencies: number[], percentiles: number[]): Map<number, number> {
    if (latencies.length === 0) {
      return new Map(percentiles.map(p => [p, 0]));
    }

    const sorted = [...latencies].sort((a, b) => a - b);
    const results = new Map<number, number>();

    for (const percentile of percentiles) {
      const value = this.calculatePercentile(sorted, percentile);
      results.set(percentile, this.round(value));
    }

    return results;
  }

  /**
   * Create latency distribution histogram
   */
  static createLatencyDistribution(latencies: number[], bucketCount: number = this.HISTOGRAM_BUCKETS): LatencyDistribution {
    if (latencies.length === 0) {
      return {
        buckets: [],
        percentiles: this.calculatePercentiles([]),
        outliers: [],
        outlier_threshold: 0
      };
    }

    const percentiles = this.calculatePercentiles(latencies);
    
    // Calculate outlier threshold (values beyond 99.5th percentile)
    const outlier_threshold = this.calculatePercentile([...latencies].sort((a, b) => a - b), 99.5);
    const outliers = latencies.filter(val => val > outlier_threshold);

    // Create histogram buckets
    const min = percentiles.min;
    const max = Math.min(percentiles.max, outlier_threshold); // Exclude outliers from histogram
    const bucketSize = (max - min) / bucketCount;

    const buckets: HistogramBucket[] = [];
    
    for (let i = 0; i < bucketCount; i++) {
      const bucketMin = min + (i * bucketSize);
      const bucketMax = i === bucketCount - 1 ? max : bucketMin + bucketSize;
      
      const count = latencies.filter(val => val >= bucketMin && val < bucketMax).length;
      const percentage = (count / latencies.length) * 100;

      buckets.push({
        min: this.round(bucketMin),
        max: this.round(bucketMax),
        count,
        percentage: this.round(percentage)
      });
    }

    return {
      buckets,
      percentiles,
      outliers,
      outlier_threshold: this.round(outlier_threshold)
    };
  }

  /**
   * Compare percentiles between two datasets
   */
  static comparePercentiles(baseline: number[], current: number[]): {
    baseline_percentiles: PercentileResult;
    current_percentiles: PercentileResult;
    improvements: Map<string, number>; // percentage improvement (negative = degradation)
    statistical_significance: boolean;
  } {
    const baselineStats = this.calculatePercentiles(baseline);
    const currentStats = this.calculatePercentiles(current);

    const improvements = new Map<string, number>();
    
    // Calculate percentage changes
    const metrics = ['p50', 'p90', 'p95', 'p99', 'p99_9', 'mean'] as const;
    
    for (const metric of metrics) {
      const baselineValue = baselineStats[metric];
      const currentValue = currentStats[metric];
      
      if (baselineValue > 0) {
        const improvement = ((baselineValue - currentValue) / baselineValue) * 100;
        improvements.set(metric, this.round(improvement));
      }
    }

    // Simple statistical significance test (Mann-Whitney U test approximation)
    const statistical_significance = this.testStatisticalSignificance(baseline, current);

    return {
      baseline_percentiles: baselineStats,
      current_percentiles: currentStats,
      improvements,
      statistical_significance
    };
  }

  /**
   * Detect latency anomalies using percentile-based thresholds
   */
  static detectAnomalies(latencies: number[], windowSize: number = 100): {
    anomalies: {
      index: number;
      value: number;
      expected_range: { min: number; max: number };
      severity: 'mild' | 'moderate' | 'severe';
    }[];
    anomaly_rate: number;
    detection_threshold: number;
  } {
    if (latencies.length < windowSize) {
      return { anomalies: [], anomaly_rate: 0, detection_threshold: 0 };
    }

    const anomalies: Array<{
      index: number;
      value: number;
      expected_range: { min: number; max: number };
      severity: 'mild' | 'moderate' | 'severe';
    }> = [];

    // Use sliding window to detect anomalies
    for (let i = windowSize; i < latencies.length; i++) {
      const window = latencies.slice(i - windowSize, i);
      const stats = this.calculatePercentiles(window);
      
      const currentValue = latencies[i];
      
      // Define expected range as mean ± 3 standard deviations
      const lowerBound = Math.max(0, stats.mean - (3 * stats.std_deviation));
      const upperBound = stats.mean + (3 * stats.std_deviation);
      
      if (currentValue < lowerBound || currentValue > upperBound) {
        // Determine severity based on how far outside the range
        let severity: 'mild' | 'moderate' | 'severe';
        const distance = Math.max(
          Math.abs(currentValue - lowerBound),
          Math.abs(currentValue - upperBound)
        );
        const threshold = stats.std_deviation;
        
        if (distance > 5 * threshold) severity = 'severe';
        else if (distance > 4 * threshold) severity = 'moderate';
        else severity = 'mild';

        anomalies.push({
          index: i,
          value: currentValue,
          expected_range: { min: this.round(lowerBound), max: this.round(upperBound) },
          severity
        });
      }
    }

    const anomaly_rate = (anomalies.length / (latencies.length - windowSize)) * 100;
    const detection_threshold = this.calculatePercentile([...latencies].sort((a, b) => a - b), 99);

    return {
      anomalies,
      anomaly_rate: this.round(anomaly_rate),
      detection_threshold: this.round(detection_threshold)
    };
  }

  /**
   * Calculate latency SLA compliance
   */
  static calculateSLACompliance(latencies: number[], slaThresholds: { percentile: number; threshold_ms: number }[]): {
    overall_compliance: number;
    sla_results: {
      percentile: number;
      threshold_ms: number;
      actual_ms: number;
      compliant: boolean;
      margin_ms: number;
    }[];
    samples_count: number;
  } {
    if (latencies.length === 0) {
      return {
        overall_compliance: 100,
        sla_results: [],
        samples_count: 0
      };
    }

    const percentiles = this.calculatePercentiles(latencies);
    const sla_results = [];
    let compliantCount = 0;

    for (const sla of slaThresholds) {
      let actual_ms: number;
      
      // Map percentile to calculated value
      switch (sla.percentile) {
        case 50: actual_ms = percentiles.p50; break;
        case 90: actual_ms = percentiles.p90; break;
        case 95: actual_ms = percentiles.p95; break;
        case 99: actual_ms = percentiles.p99; break;
        case 99.9: actual_ms = percentiles.p99_9; break;
        default:
          actual_ms = this.calculatePercentile([...latencies].sort((a, b) => a - b), sla.percentile);
      }

      const compliant = actual_ms <= sla.threshold_ms;
      const margin_ms = sla.threshold_ms - actual_ms;
      
      if (compliant) compliantCount++;

      sla_results.push({
        percentile: sla.percentile,
        threshold_ms: sla.threshold_ms,
        actual_ms: this.round(actual_ms),
        compliant,
        margin_ms: this.round(margin_ms)
      });
    }

    const overall_compliance = slaThresholds.length > 0 ? 
      (compliantCount / slaThresholds.length) * 100 : 
      100;

    return {
      overall_compliance: this.round(overall_compliance),
      sla_results,
      samples_count: latencies.length
    };
  }

  /**
   * Rolling percentile calculation for streaming data
   */
  static createRollingPercentileCalculator(windowSize: number = 1000) {
    const buffer: number[] = [];
    
    return {
      addValue(value: number): PercentileResult | null {
        buffer.push(value);
        
        // Maintain window size
        if (buffer.length > windowSize) {
          buffer.shift();
        }
        
        // Return percentiles if we have enough data
        if (buffer.length >= Math.min(10, windowSize / 10)) {
          return this.calculatePercentiles(buffer);
        }
        
        return null;
      },
      
      getCurrentPercentiles(): PercentileResult {
        return this.calculatePercentiles(buffer);
      },
      
      getBufferSize(): number {
        return buffer.length;
      },
      
      clear(): void {
        buffer.length = 0;
      }
    };
  }

  /**
   * Simple statistical significance test
   */
  private static testStatisticalSignificance(group1: number[], group2: number[]): boolean {
    if (group1.length < 30 || group2.length < 30) return false;

    const mean1 = group1.reduce((sum, val) => sum + val, 0) / group1.length;
    const mean2 = group2.reduce((sum, val) => sum + val, 0) / group2.length;
    
    const variance1 = group1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / group1.length;
    const variance2 = group2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / group2.length;
    
    const standardError = Math.sqrt((variance1 / group1.length) + (variance2 / group2.length));
    const tStat = Math.abs(mean1 - mean2) / standardError;
    
    // Simple t-test threshold for significance (approximately p < 0.05)
    return tStat > 1.96;
  }

  /**
   * Round number to 2 decimal places
   */
  private static round(value: number): number {
    return Math.round(value * 100) / 100;
  }
}

/**
 * Utility class for real-time percentile tracking
 */
export class RealTimePercentileTracker {
  private calculator = LatencyPercentileCalculator.createRollingPercentileCalculator(this.windowSize);
  private callbacks: ((percentiles: PercentileResult) => void)[] = [];
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(
    private windowSize: number = 1000,
    private updateIntervalMs: number = 1000
  ) {}

  /**
   * Start real-time tracking
   */
  start(): void {
    if (this.updateInterval) return;

    this.updateInterval = setInterval(() => {
      if (this.calculator.getBufferSize() > 0) {
        const percentiles = this.calculator.getCurrentPercentiles();
        this.callbacks.forEach(callback => callback(percentiles));
      }
    }, this.updateIntervalMs);
  }

  /**
   * Stop real-time tracking
   */
  stop(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  /**
   * Add new latency measurement
   */
  addMeasurement(latencyMs: number): void {
    this.calculator.addValue(latencyMs);
  }

  /**
   * Get current percentiles
   */
  getCurrentPercentiles(): PercentileResult {
    return this.calculator.getCurrentPercentiles();
  }

  /**
   * Add callback for percentile updates
   */
  onUpdate(callback: (percentiles: PercentileResult) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: (percentiles: PercentileResult) => void): void {
    const index = this.callbacks.indexOf(callback);
    if (index > -1) {
      this.callbacks.splice(index, 1);
    }
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.calculator.clear();
  }
}