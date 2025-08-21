/**
 * Network Throughput Monitor
 * Monitors network bandwidth utilization and throughput patterns
 */

export interface ThroughputMeasurement {
  measurement_id: string;
  timestamp: Date;
  bytes_sent: number;
  bytes_received: number;
  packets_sent: number;
  packets_received: number;
  connection_count: number;
  measurement_interval_ms: number;
  throughput_mbps: number;
  bandwidth_utilization_percent: number;
}

export interface ThroughputStatistics {
  time_period: string;
  average_throughput_mbps: number;
  peak_throughput_mbps: number;
  minimum_throughput_mbps: number;
  average_utilization_percent: number;
  peak_utilization_percent: number;
  data_transferred_gb: number;
  measurement_count: number;
  volatility_score: number;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
}

export interface BandwidthAlert {
  alert_id: string;
  alert_type: 'utilization_high' | 'throughput_low' | 'pattern_anomaly' | 'capacity_warning';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  current_value: number;
  threshold_value: number;
  triggered_at: Date;
  resolved: boolean;
  recommendations: string[];
}

export interface ThroughputPattern {
  pattern_id: string;
  pattern_type: 'peak_hours' | 'baseline_shift' | 'periodic_spike' | 'gradual_increase';
  description: string;
  confidence_score: number;
  detected_at: Date;
  parameters: Record<string, any>;
}

export class NetworkThroughputMonitor {
  private measurements: ThroughputMeasurement[] = new Map();
  private statistics: Map<string, ThroughputStatistics> = new Map();
  private alerts: BandwidthAlert[] = [];
  private patterns: ThroughputPattern[] = [];
  private maxMeasurements: number = 10000;
  
  // Configuration
  private config = {
    max_bandwidth_mbps: 1000, // 1 Gbps default capacity
    utilization_warning_threshold: 75, // 75%
    utilization_critical_threshold: 90, // 90%
    throughput_warning_threshold_mbps: 10, // 10 Mbps minimum
    measurement_interval_ms: 5000, // 5 seconds
    pattern_detection_min_samples: 100
  };

  private callbacks: {
    onMeasurement: ((measurement: ThroughputMeasurement) => void)[];
    onAlert: ((alert: BandwidthAlert) => void)[];
    onPatternDetected: ((pattern: ThroughputPattern) => void)[];
  } = { onMeasurement: [], onAlert: [], onPatternDetected: [] };

  private nextMeasurementId: number = 1;
  private nextAlertId: number = 1;
  private nextPatternId: number = 1;
  private lastMeasurement: ThroughputMeasurement | null = null;

  /**
   * Record a throughput measurement
   */
  recordMeasurement(
    bytesSent: number,
    bytesReceived: number,
    packetsSent: number = 0,
    packetsReceived: number = 0,
    connectionCount: number = 0
  ): ThroughputMeasurement {
    const now = new Date();
    const intervalMs = this.lastMeasurement ? 
      now.getTime() - this.lastMeasurement.timestamp.getTime() : 
      this.config.measurement_interval_ms;

    // Calculate throughput (convert bytes to Mbps)
    const totalBytes = bytesSent + bytesReceived;
    const throughputMbps = (totalBytes * 8) / (1024 * 1024) / (intervalMs / 1000);
    
    // Calculate bandwidth utilization
    const bandwidthUtilization = (throughputMbps / this.config.max_bandwidth_mbps) * 100;

    const measurement: ThroughputMeasurement = {
      measurement_id: `measure_${this.nextMeasurementId++}`,
      timestamp: now,
      bytes_sent: bytesSent,
      bytes_received: bytesReceived,
      packets_sent: packetsSent,
      packets_received: packetsReceived,
      connection_count: connectionCount,
      measurement_interval_ms: intervalMs,
      throughput_mbps: Math.round(throughputMbps * 100) / 100,
      bandwidth_utilization_percent: Math.round(bandwidthUtilization * 100) / 100
    };

    // Store measurement
    this.measurements.push(measurement);
    this.maintainMeasurementLimit();

    // Update statistics
    this.updateStatistics();

    // Check for alerts
    this.checkThroughputAlerts(measurement);

    // Analyze patterns
    if (this.measurements.length >= this.config.pattern_detection_min_samples) {
      this.analyzePatterns();
    }

    this.lastMeasurement = measurement;

    // Notify callbacks
    this.callbacks.onMeasurement.forEach(callback => callback(measurement));

    return measurement;
  }

  /**
   * Get throughput statistics for time period
   */
  getThroughputStatistics(timePeriod: string = '24h'): ThroughputStatistics {
    const measurements = this.getMeasurementsInPeriod(timePeriod);
    
    if (measurements.length === 0) {
      return this.getEmptyStatistics(timePeriod);
    }

    const throughputValues = measurements.map(m => m.throughput_mbps);
    const utilizationValues = measurements.map(m => m.bandwidth_utilization_percent);
    
    const avgThroughput = this.calculateAverage(throughputValues);
    const peakThroughput = Math.max(...throughputValues);
    const minThroughput = Math.min(...throughputValues);
    const avgUtilization = this.calculateAverage(utilizationValues);
    const peakUtilization = Math.max(...utilizationValues);
    
    // Calculate total data transferred
    const totalDataBytes = measurements.reduce((sum, m) => sum + m.bytes_sent + m.bytes_received, 0);
    const dataTransferredGb = (totalDataBytes / (1024 * 1024 * 1024));
    
    // Calculate volatility (coefficient of variation)
    const volatilityScore = this.calculateVolatility(throughputValues);
    
    // Determine trend direction
    const trendDirection = this.calculateTrendDirection(measurements);

    const statistics: ThroughputStatistics = {
      time_period: timePeriod,
      average_throughput_mbps: Math.round(avgThroughput * 100) / 100,
      peak_throughput_mbps: Math.round(peakThroughput * 100) / 100,
      minimum_throughput_mbps: Math.round(minThroughput * 100) / 100,
      average_utilization_percent: Math.round(avgUtilization * 100) / 100,
      peak_utilization_percent: Math.round(peakUtilization * 100) / 100,
      data_transferred_gb: Math.round(dataTransferredGb * 100) / 100,
      measurement_count: measurements.length,
      volatility_score: Math.round(volatilityScore * 100) / 100,
      trend_direction: trendDirection
    };

    this.statistics.set(timePeriod, statistics);
    return statistics;
  }

  /**
   * Get recent measurements
   */
  getRecentMeasurements(count: number = 100): ThroughputMeasurement[] {
    return this.measurements.slice(-count);
  }

  /**
   * Get active bandwidth alerts
   */
  getActiveAlerts(): BandwidthAlert[] {
    return this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Get detected patterns
   */
  getDetectedPatterns(): ThroughputPattern[] {
    return [...this.patterns];
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.alert_id === alertId);
    if (alert) {
      alert.resolved = true;
      return true;
    }
    return false;
  }

  /**
   * Get throughput distribution by time of day
   */
  getThroughputByTimeOfDay(days: number = 7): {
    hour: number;
    average_throughput_mbps: number;
    peak_throughput_mbps: number;
    average_utilization_percent: number;
    measurement_count: number;
  }[] {
    const cutoff = new Date(Date.now() - (days * 24 * 60 * 60 * 1000));
    const relevantMeasurements = this.measurements.filter(m => m.timestamp >= cutoff);
    
    const hourlyData = new Map<number, ThroughputMeasurement[]>();
    
    // Initialize all hours
    for (let hour = 0; hour < 24; hour++) {
      hourlyData.set(hour, []);
    }
    
    // Group measurements by hour
    for (const measurement of relevantMeasurements) {
      const hour = measurement.timestamp.getHours();
      hourlyData.get(hour)!.push(measurement);
    }
    
    const results: any[] = [];
    
    for (let hour = 0; hour < 24; hour++) {
      const hourMeasurements = hourlyData.get(hour)!;
      const throughputValues = hourMeasurements.map(m => m.throughput_mbps);
      const utilizationValues = hourMeasurements.map(m => m.bandwidth_utilization_percent);
      
      results.push({
        hour,
        average_throughput_mbps: throughputValues.length > 0 ? 
          Math.round(this.calculateAverage(throughputValues) * 100) / 100 : 0,
        peak_throughput_mbps: throughputValues.length > 0 ? 
          Math.round(Math.max(...throughputValues) * 100) / 100 : 0,
        average_utilization_percent: utilizationValues.length > 0 ? 
          Math.round(this.calculateAverage(utilizationValues) * 100) / 100 : 0,
        measurement_count: hourMeasurements.length
      });
    }
    
    return results;
  }

  /**
   * Predict future throughput needs
   */
  predictThroughputNeeds(hoursAhead: number = 24): {
    predicted_throughput_mbps: number;
    predicted_utilization_percent: number;
    confidence_score: number;
    capacity_warning: boolean;
    recommendations: string[];
  } {
    const recentMeasurements = this.getRecentMeasurements(200); // Last 200 measurements
    
    if (recentMeasurements.length < 50) {
      return {
        predicted_throughput_mbps: 0,
        predicted_utilization_percent: 0,
        confidence_score: 0,
        capacity_warning: false,
        recommendations: ['Insufficient data for prediction - collect more throughput measurements']
      };
    }
    
    const throughputValues = recentMeasurements.map(m => m.throughput_mbps);
    const trend = this.calculateLinearTrend(throughputValues);
    
    // Simple linear prediction
    const currentAverage = this.calculateAverage(throughputValues.slice(-20)); // Last 20 measurements
    const predictedThroughput = Math.max(0, currentAverage + (trend.slope * (hoursAhead * 12))); // 12 measurements per hour (5s intervals)
    const predictedUtilization = (predictedThroughput / this.config.max_bandwidth_mbps) * 100;
    
    const confidenceScore = Math.abs(trend.correlation);
    const capacityWarning = predictedUtilization > this.config.utilization_warning_threshold;
    
    const recommendations: string[] = [];
    if (capacityWarning) {
      recommendations.push(`Predicted utilization (${predictedUtilization.toFixed(1)}%) may exceed capacity in ${hoursAhead} hours`);
      recommendations.push('Consider bandwidth upgrade or traffic optimization');
    }
    
    if (trend.slope > 0 && confidenceScore > 0.5) {
      recommendations.push('Increasing throughput trend detected - monitor capacity closely');
    }
    
    if (predictedThroughput < this.config.throughput_warning_threshold_mbps) {
      recommendations.push('Low throughput predicted - investigate potential connectivity issues');
    }
    
    return {
      predicted_throughput_mbps: Math.round(predictedThroughput * 100) / 100,
      predicted_utilization_percent: Math.round(predictedUtilization * 100) / 100,
      confidence_score: Math.round(confidenceScore * 100) / 100,
      capacity_warning: capacityWarning,
      recommendations
    };
  }

  /**
   * Set configuration
   */
  setConfiguration(config: Partial<typeof this.config>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfiguration(): typeof this.config {
    return { ...this.config };
  }

  /**
   * Add callbacks
   */
  onMeasurement(callback: (measurement: ThroughputMeasurement) => void): void {
    this.callbacks.onMeasurement.push(callback);
  }

  onAlert(callback: (alert: BandwidthAlert) => void): void {
    this.callbacks.onAlert.push(callback);
  }

  onPatternDetected(callback: (pattern: ThroughputPattern) => void): void {
    this.callbacks.onPatternDetected.push(callback);
  }

  /**
   * Remove callbacks
   */
  removeCallback(callback: Function): void {
    let index = this.callbacks.onMeasurement.indexOf(callback as any);
    if (index > -1) this.callbacks.onMeasurement.splice(index, 1);

    index = this.callbacks.onAlert.indexOf(callback as any);
    if (index > -1) this.callbacks.onAlert.splice(index, 1);

    index = this.callbacks.onPatternDetected.indexOf(callback as any);
    if (index > -1) this.callbacks.onPatternDetected.splice(index, 1);
  }

  /**
   * Clear all monitoring data
   */
  clear(): void {
    this.measurements = [];
    this.statistics.clear();
    this.alerts = [];
    this.patterns = [];
    this.nextMeasurementId = 1;
    this.nextAlertId = 1;
    this.nextPatternId = 1;
    this.lastMeasurement = null;
  }

  // Private methods

  private getMeasurementsInPeriod(timePeriod: string): ThroughputMeasurement[] {
    const now = new Date();
    let cutoffTime: Date;

    switch (timePeriod) {
      case '1h': cutoffTime = new Date(now.getTime() - (60 * 60 * 1000)); break;
      case '6h': cutoffTime = new Date(now.getTime() - (6 * 60 * 60 * 1000)); break;
      case '24h': cutoffTime = new Date(now.getTime() - (24 * 60 * 60 * 1000)); break;
      case '7d': cutoffTime = new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000)); break;
      case '30d': cutoffTime = new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000)); break;
      default: cutoffTime = new Date(now.getTime() - (24 * 60 * 60 * 1000));
    }

    return this.measurements.filter(measurement => measurement.timestamp >= cutoffTime);
  }

  private maintainMeasurementLimit(): void {
    if (this.measurements.length > this.maxMeasurements) {
      this.measurements = this.measurements.slice(-this.maxMeasurements);
    }
  }

  private updateStatistics(): void {
    const periods = ['1h', '6h', '24h', '7d'];
    for (const period of periods) {
      this.getThroughputStatistics(period);
    }
  }

  private checkThroughputAlerts(measurement: ThroughputMeasurement): void {
    // Check utilization alerts
    if (measurement.bandwidth_utilization_percent > this.config.utilization_critical_threshold) {
      this.createAlert(
        'utilization_high',
        'critical',
        `Bandwidth utilization critical: ${measurement.bandwidth_utilization_percent.toFixed(1)}%`,
        measurement.bandwidth_utilization_percent,
        this.config.utilization_critical_threshold,
        [
          'Immediate attention required - network capacity exceeded',
          'Consider traffic shaping or bandwidth upgrade',
          'Monitor for service degradation'
        ]
      );
    } else if (measurement.bandwidth_utilization_percent > this.config.utilization_warning_threshold) {
      this.createAlert(
        'utilization_high',
        'high',
        `Bandwidth utilization high: ${measurement.bandwidth_utilization_percent.toFixed(1)}%`,
        measurement.bandwidth_utilization_percent,
        this.config.utilization_warning_threshold,
        [
          'Monitor bandwidth usage closely',
          'Plan for capacity expansion',
          'Optimize data transmission efficiency'
        ]
      );
    }

    // Check low throughput alerts
    if (measurement.throughput_mbps < this.config.throughput_warning_threshold_mbps) {
      this.createAlert(
        'throughput_low',
        'medium',
        `Low throughput detected: ${measurement.throughput_mbps.toFixed(2)} Mbps`,
        measurement.throughput_mbps,
        this.config.throughput_warning_threshold_mbps,
        [
          'Investigate connectivity issues',
          'Check for network congestion',
          'Verify data feed activity'
        ]
      );
    }
  }

  private analyzePatterns(): void {
    // Analyze for peak hours pattern
    this.detectPeakHoursPattern();
    
    // Analyze for baseline shifts
    this.detectBaselineShiftPattern();
    
    // Analyze for periodic spikes
    this.detectPeriodicSpikePattern();
  }

  private detectPeakHoursPattern(): void {
    const hourlyData = this.getThroughputByTimeOfDay(7);
    const avgThroughput = this.calculateAverage(hourlyData.map(h => h.average_throughput_mbps));
    
    // Find hours with significantly higher than average throughput
    const peakHours = hourlyData.filter(h => h.average_throughput_mbps > avgThroughput * 1.5);
    
    if (peakHours.length >= 2 && peakHours.length <= 8) {
      this.createPattern(
        'peak_hours',
        `Peak usage hours detected: ${peakHours.map(h => h.hour).join(', ')}`,
        0.8,
        { peak_hours: peakHours.map(h => h.hour), average_peak_throughput: this.calculateAverage(peakHours.map(h => h.average_throughput_mbps)) }
      );
    }
  }

  private detectBaselineShiftPattern(): void {
    const recent = this.getRecentMeasurements(100);
    const older = this.measurements.slice(-200, -100);
    
    if (recent.length >= 50 && older.length >= 50) {
      const recentAvg = this.calculateAverage(recent.map(m => m.throughput_mbps));
      const olderAvg = this.calculateAverage(older.map(m => m.throughput_mbps));
      
      const percentChange = ((recentAvg - olderAvg) / olderAvg) * 100;
      
      if (Math.abs(percentChange) > 20) {
        this.createPattern(
          'baseline_shift',
          `Baseline throughput ${percentChange > 0 ? 'increased' : 'decreased'} by ${Math.abs(percentChange).toFixed(1)}%`,
          0.7,
          { previous_baseline: olderAvg, current_baseline: recentAvg, percent_change: percentChange }
        );
      }
    }
  }

  private detectPeriodicSpikePattern(): void {
    const recent = this.getRecentMeasurements(200);
    if (recent.length < 100) return;
    
    const throughputValues = recent.map(m => m.throughput_mbps);
    const avgThroughput = this.calculateAverage(throughputValues);
    const spikes = recent.filter(m => m.throughput_mbps > avgThroughput * 2);
    
    if (spikes.length > 5) {
      // Check for periodic pattern in spike timing
      const spikeIntervals: number[] = [];
      for (let i = 1; i < spikes.length; i++) {
        const interval = (spikes[i].timestamp.getTime() - spikes[i-1].timestamp.getTime()) / 1000 / 60; // minutes
        spikeIntervals.push(interval);
      }
      
      const avgInterval = this.calculateAverage(spikeIntervals);
      const intervalVariance = this.calculateVariance(spikeIntervals);
      
      // If intervals are relatively consistent (low variance), it's a pattern
      if (intervalVariance / avgInterval < 0.3) {
        this.createPattern(
          'periodic_spike',
          `Periodic throughput spikes detected every ~${avgInterval.toFixed(0)} minutes`,
          0.6,
          { spike_count: spikes.length, average_interval_minutes: avgInterval, spike_magnitude: spikes[0].throughput_mbps / avgThroughput }
        );
      }
    }
  }

  private createAlert(
    alertType: 'utilization_high' | 'throughput_low' | 'pattern_anomaly' | 'capacity_warning',
    severity: 'low' | 'medium' | 'high' | 'critical',
    message: string,
    currentValue: number,
    thresholdValue: number,
    recommendations: string[]
  ): void {
    // Check for existing similar alert
    const existingAlert = this.alerts.find(alert => 
      !alert.resolved &&
      alert.alert_type === alertType &&
      alert.severity === severity
    );
    
    if (existingAlert) return;

    const alert: BandwidthAlert = {
      alert_id: `alert_${this.nextAlertId++}`,
      alert_type: alertType,
      severity,
      message,
      current_value: currentValue,
      threshold_value: thresholdValue,
      triggered_at: new Date(),
      resolved: false,
      recommendations
    };

    this.alerts.push(alert);
    this.callbacks.onAlert.forEach(callback => callback(alert));
  }

  private createPattern(
    patternType: 'peak_hours' | 'baseline_shift' | 'periodic_spike' | 'gradual_increase',
    description: string,
    confidenceScore: number,
    parameters: Record<string, any>
  ): void {
    // Check for existing similar pattern
    const existingPattern = this.patterns.find(p => 
      p.pattern_type === patternType &&
      (Date.now() - p.detected_at.getTime()) < 24 * 60 * 60 * 1000 // Within last 24h
    );
    
    if (existingPattern) return;

    const pattern: ThroughputPattern = {
      pattern_id: `pattern_${this.nextPatternId++}`,
      pattern_type: patternType,
      description,
      confidence_score: confidenceScore,
      detected_at: new Date(),
      parameters
    };

    this.patterns.push(pattern);
    this.callbacks.onPatternDetected.forEach(callback => callback(pattern));
  }

  private calculateAverage(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.calculateAverage(values);
    return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  }

  private calculateVolatility(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.calculateAverage(values);
    const variance = this.calculateVariance(values);
    const stdDev = Math.sqrt(variance);
    return mean === 0 ? 0 : stdDev / Math.abs(mean);
  }

  private calculateTrendDirection(measurements: ThroughputMeasurement[]): 'increasing' | 'decreasing' | 'stable' {
    if (measurements.length < 10) return 'stable';
    
    const throughputValues = measurements.map(m => m.throughput_mbps);
    const trend = this.calculateLinearTrend(throughputValues);
    
    if (Math.abs(trend.slope) < 0.1) return 'stable';
    return trend.slope > 0 ? 'increasing' : 'decreasing';
  }

  private calculateLinearTrend(values: number[]): { slope: number; intercept: number; correlation: number } {
    const n = values.length;
    if (n === 0) return { slope: 0, intercept: 0, correlation: 0 };
    
    const x = Array.from({ length: n }, (_, i) => i);
    const y = values;
    
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = y.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + (val * y[i]), 0);
    const sumXX = x.reduce((sum, val) => sum + (val * val), 0);
    const sumYY = y.reduce((sum, val) => sum + (val * val), 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    const correlation = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    return { 
      slope: isNaN(slope) ? 0 : slope,
      intercept: isNaN(intercept) ? 0 : intercept,
      correlation: isNaN(correlation) ? 0 : correlation
    };
  }

  private getEmptyStatistics(timePeriod: string): ThroughputStatistics {
    return {
      time_period: timePeriod,
      average_throughput_mbps: 0,
      peak_throughput_mbps: 0,
      minimum_throughput_mbps: 0,
      average_utilization_percent: 0,
      peak_utilization_percent: 0,
      data_transferred_gb: 0,
      measurement_count: 0,
      volatility_score: 0,
      trend_direction: 'stable'
    };
  }
}

// Global instance for network throughput monitoring
export const networkThroughputMonitor = new NetworkThroughputMonitor();