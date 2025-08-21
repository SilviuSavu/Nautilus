/**
 * Connection Success Rate Tracker
 * Monitors connection success/failure rates and patterns for different venues
 */

export interface ConnectionAttempt {
  attempt_id: string;
  venue_name: string;
  timestamp: Date;
  success: boolean;
  failure_reason?: string;
  response_time_ms?: number;
  retry_count: number;
  connection_type: 'initial' | 'reconnect' | 'heartbeat' | 'data_request';
  metadata?: Record<string, any>;
}

export interface SuccessRateMetrics {
  venue_name: string;
  time_period: string;
  total_attempts: number;
  successful_attempts: number;
  failed_attempts: number;
  success_rate_percent: number;
  average_response_time_ms: number;
  failure_breakdown: {
    timeout: number;
    network_error: number;
    authentication_error: number;
    server_error: number;
    unknown: number;
  };
  trend_direction: 'improving' | 'degrading' | 'stable';
  last_updated: Date;
}

export interface FailurePattern {
  pattern_type: 'time_based' | 'frequency_based' | 'sequential';
  description: string;
  occurrences: number;
  severity: 'low' | 'medium' | 'high';
  recommended_action: string;
  first_detected: Date;
  last_occurrence: Date;
}

export class ConnectionSuccessRateTracker {
  private connectionAttempts: Map<string, ConnectionAttempt[]> = new Map();
  private successRateMetrics: Map<string, SuccessRateMetrics> = new Map();
  private detectedPatterns: Map<string, FailurePattern[]> = new Map();
  private maxAttemptsPerVenue: number = 5000;
  private callbacks: {
    onSuccess: ((venue: string, attempt: ConnectionAttempt) => void)[];
    onFailure: ((venue: string, attempt: ConnectionAttempt) => void)[];
    onPatternDetected: ((venue: string, pattern: FailurePattern) => void)[];
  } = { onSuccess: [], onFailure: [], onPatternDetected: [] };

  private nextAttemptId: number = 1;

  /**
   * Record a connection attempt
   */
  recordConnectionAttempt(
    venue: string,
    success: boolean,
    connectionType: 'initial' | 'reconnect' | 'heartbeat' | 'data_request' = 'initial',
    responseTimeMs?: number,
    failureReason?: string,
    retryCount: number = 0,
    metadata?: Record<string, any>
  ): ConnectionAttempt {
    const attempt: ConnectionAttempt = {
      attempt_id: `attempt_${this.nextAttemptId++}`,
      venue_name: venue,
      timestamp: new Date(),
      success,
      failure_reason: failureReason,
      response_time_ms: responseTimeMs,
      retry_count: retryCount,
      connection_type: connectionType,
      metadata
    };

    // Store attempt
    if (!this.connectionAttempts.has(venue)) {
      this.connectionAttempts.set(venue, []);
    }

    const attempts = this.connectionAttempts.get(venue)!;
    attempts.push(attempt);

    // Maintain attempt limit
    if (attempts.length > this.maxAttemptsPerVenue) {
      attempts.shift();
    }

    // Update metrics
    this.updateSuccessRateMetrics(venue);

    // Check for failure patterns
    if (!success) {
      this.analyzeFailurePatterns(venue);
    }

    // Notify callbacks
    if (success) {
      this.callbacks.onSuccess.forEach(callback => callback(venue, attempt));
    } else {
      this.callbacks.onFailure.forEach(callback => callback(venue, attempt));
    }

    return attempt;
  }

  /**
   * Get success rate metrics for venue
   */
  getSuccessRateMetrics(venue: string, timePeriod: string = '24h'): SuccessRateMetrics | null {
    const attempts = this.getAttemptsInPeriod(venue, timePeriod);
    
    if (attempts.length === 0) {
      return null;
    }

    const successfulAttempts = attempts.filter(a => a.success);
    const failedAttempts = attempts.filter(a => !a.success);
    
    const successRate = (successfulAttempts.length / attempts.length) * 100;
    const avgResponseTime = this.calculateAverageResponseTime(successfulAttempts);
    const failureBreakdown = this.categorizeFailures(failedAttempts);
    const trend = this.calculateTrend(venue, timePeriod);

    const metrics: SuccessRateMetrics = {
      venue_name: venue,
      time_period: timePeriod,
      total_attempts: attempts.length,
      successful_attempts: successfulAttempts.length,
      failed_attempts: failedAttempts.length,
      success_rate_percent: Math.round(successRate * 100) / 100,
      average_response_time_ms: Math.round(avgResponseTime * 100) / 100,
      failure_breakdown: failureBreakdown,
      trend_direction: trend,
      last_updated: new Date()
    };

    this.successRateMetrics.set(`${venue}_${timePeriod}`, metrics);
    return metrics;
  }

  /**
   * Get success rate comparison between time periods
   */
  compareSuccessRates(venue: string, period1: string, period2: string): {
    period1_metrics: SuccessRateMetrics | null;
    period2_metrics: SuccessRateMetrics | null;
    improvement_percent: number;
    statistical_significance: boolean;
    key_differences: string[];
  } {
    const period1Metrics = this.getSuccessRateMetrics(venue, period1);
    const period2Metrics = this.getSuccessRateMetrics(venue, period2);
    
    if (!period1Metrics || !period2Metrics) {
      return {
        period1_metrics: period1Metrics,
        period2_metrics: period2Metrics,
        improvement_percent: 0,
        statistical_significance: false,
        key_differences: ['Insufficient data for comparison']
      };
    }

    const improvement = period1Metrics.success_rate_percent - period2Metrics.success_rate_percent;
    const statSig = this.testStatisticalSignificance(
      period1Metrics.total_attempts,
      period1Metrics.successful_attempts,
      period2Metrics.total_attempts,
      period2Metrics.successful_attempts
    );

    const keyDifferences: string[] = [];
    
    // Analyze key differences
    if (Math.abs(improvement) > 5) {
      keyDifferences.push(`Success rate ${improvement > 0 ? 'improved' : 'degraded'} by ${Math.abs(improvement).toFixed(1)}%`);
    }

    const responseTimeDiff = period1Metrics.average_response_time_ms - period2Metrics.average_response_time_ms;
    if (Math.abs(responseTimeDiff) > 50) {
      keyDifferences.push(`Response time ${responseTimeDiff > 0 ? 'increased' : 'decreased'} by ${Math.abs(responseTimeDiff).toFixed(0)}ms`);
    }

    // Check for failure type changes
    const p1Failures = period1Metrics.failure_breakdown;
    const p2Failures = period2Metrics.failure_breakdown;
    
    for (const [type, count] of Object.entries(p1Failures)) {
      const p2Count = (p2Failures as any)[type] || 0;
      const diff = count - p2Count;
      if (Math.abs(diff) > 5) {
        keyDifferences.push(`${type.replace('_', ' ')} failures ${diff > 0 ? 'increased' : 'decreased'} by ${Math.abs(diff)}`);
      }
    }

    return {
      period1_metrics: period1Metrics,
      period2_metrics: period2Metrics,
      improvement_percent: Math.round(improvement * 100) / 100,
      statistical_significance: statSig,
      key_differences: keyDifferences.length > 0 ? keyDifferences : ['No significant differences detected']
    };
  }

  /**
   * Get detected failure patterns
   */
  getFailurePatterns(venue: string): FailurePattern[] {
    return this.detectedPatterns.get(venue) || [];
  }

  /**
   * Get success rate by connection type
   */
  getSuccessRateByType(venue: string, timePeriod: string = '24h'): {
    connection_type: string;
    total_attempts: number;
    success_rate_percent: number;
    average_response_time_ms: number;
  }[] {
    const attempts = this.getAttemptsInPeriod(venue, timePeriod);
    const byType = new Map<string, ConnectionAttempt[]>();

    // Group by connection type
    for (const attempt of attempts) {
      if (!byType.has(attempt.connection_type)) {
        byType.set(attempt.connection_type, []);
      }
      byType.get(attempt.connection_type)!.push(attempt);
    }

    const results: any[] = [];
    
    for (const [type, typeAttempts] of byType) {
      const successfulAttempts = typeAttempts.filter(a => a.success);
      const successRate = (successfulAttempts.length / typeAttempts.length) * 100;
      const avgResponseTime = this.calculateAverageResponseTime(successfulAttempts);

      results.push({
        connection_type: type,
        total_attempts: typeAttempts.length,
        success_rate_percent: Math.round(successRate * 100) / 100,
        average_response_time_ms: Math.round(avgResponseTime * 100) / 100
      });
    }

    return results.sort((a, b) => b.success_rate_percent - a.success_rate_percent);
  }

  /**
   * Get retry analysis
   */
  getRetryAnalysis(venue: string, timePeriod: string = '24h'): {
    attempts_with_retries: number;
    total_retry_attempts: number;
    average_retries_per_failure: number;
    retry_success_rate: number;
    max_retries_observed: number;
  } {
    const attempts = this.getAttemptsInPeriod(venue, timePeriod);
    const attemptsWithRetries = attempts.filter(a => a.retry_count > 0);
    
    if (attemptsWithRetries.length === 0) {
      return {
        attempts_with_retries: 0,
        total_retry_attempts: 0,
        average_retries_per_failure: 0,
        retry_success_rate: 0,
        max_retries_observed: 0
      };
    }

    const totalRetries = attemptsWithRetries.reduce((sum, a) => sum + a.retry_count, 0);
    const successfulRetries = attemptsWithRetries.filter(a => a.success).length;
    const maxRetries = Math.max(...attemptsWithRetries.map(a => a.retry_count));
    
    return {
      attempts_with_retries: attemptsWithRetries.length,
      total_retry_attempts: totalRetries,
      average_retries_per_failure: Math.round((totalRetries / attemptsWithRetries.length) * 100) / 100,
      retry_success_rate: Math.round((successfulRetries / attemptsWithRetries.length) * 100 * 100) / 100,
      max_retries_observed: maxRetries
    };
  }

  /**
   * Get hourly success rate distribution
   */
  getHourlyDistribution(venue: string, days: number = 7): {
    hour: number;
    success_rate_percent: number;
    total_attempts: number;
    average_response_time_ms: number;
  }[] {
    const now = new Date();
    const cutoff = new Date(now.getTime() - (days * 24 * 60 * 60 * 1000));
    const attempts = (this.connectionAttempts.get(venue) || [])
      .filter(a => a.timestamp >= cutoff);

    const hourlyData = new Map<number, ConnectionAttempt[]>();
    
    // Initialize all hours
    for (let hour = 0; hour < 24; hour++) {
      hourlyData.set(hour, []);
    }

    // Group attempts by hour
    for (const attempt of attempts) {
      const hour = attempt.timestamp.getHours();
      hourlyData.get(hour)!.push(attempt);
    }

    const results: any[] = [];
    
    for (let hour = 0; hour < 24; hour++) {
      const hourAttempts = hourlyData.get(hour)!;
      const successfulAttempts = hourAttempts.filter(a => a.success);
      const successRate = hourAttempts.length > 0 ? (successfulAttempts.length / hourAttempts.length) * 100 : 0;
      const avgResponseTime = this.calculateAverageResponseTime(successfulAttempts);

      results.push({
        hour,
        success_rate_percent: Math.round(successRate * 100) / 100,
        total_attempts: hourAttempts.length,
        average_response_time_ms: Math.round(avgResponseTime * 100) / 100
      });
    }

    return results;
  }

  /**
   * Get recent connection attempts
   */
  getRecentAttempts(venue: string, limit: number = 100): ConnectionAttempt[] {
    const attempts = this.connectionAttempts.get(venue) || [];
    return attempts.slice(-limit);
  }

  /**
   * Get success rate summary for all venues
   */
  getAllVenuesSuccessRates(timePeriod: string = '24h'): {
    venue_name: string;
    success_rate_percent: number;
    total_attempts: number;
    trend_direction: 'improving' | 'degrading' | 'stable';
    health_status: 'healthy' | 'warning' | 'critical';
  }[] {
    const venues = Array.from(this.connectionAttempts.keys());
    const results: any[] = [];

    for (const venue of venues) {
      const metrics = this.getSuccessRateMetrics(venue, timePeriod);
      if (metrics) {
        let healthStatus: 'healthy' | 'warning' | 'critical';
        if (metrics.success_rate_percent >= 95) healthStatus = 'healthy';
        else if (metrics.success_rate_percent >= 85) healthStatus = 'warning';
        else healthStatus = 'critical';

        results.push({
          venue_name: venue,
          success_rate_percent: metrics.success_rate_percent,
          total_attempts: metrics.total_attempts,
          trend_direction: metrics.trend_direction,
          health_status: healthStatus
        });
      }
    }

    return results.sort((a, b) => b.success_rate_percent - a.success_rate_percent);
  }

  /**
   * Add callbacks
   */
  onSuccess(callback: (venue: string, attempt: ConnectionAttempt) => void): void {
    this.callbacks.onSuccess.push(callback);
  }

  onFailure(callback: (venue: string, attempt: ConnectionAttempt) => void): void {
    this.callbacks.onFailure.push(callback);
  }

  onPatternDetected(callback: (venue: string, pattern: FailurePattern) => void): void {
    this.callbacks.onPatternDetected.push(callback);
  }

  /**
   * Remove callbacks
   */
  removeCallback(callback: Function): void {
    let index = this.callbacks.onSuccess.indexOf(callback as any);
    if (index > -1) this.callbacks.onSuccess.splice(index, 1);

    index = this.callbacks.onFailure.indexOf(callback as any);
    if (index > -1) this.callbacks.onFailure.splice(index, 1);

    index = this.callbacks.onPatternDetected.indexOf(callback as any);
    if (index > -1) this.callbacks.onPatternDetected.splice(index, 1);
  }

  /**
   * Clear all tracking data
   */
  clear(): void {
    this.connectionAttempts.clear();
    this.successRateMetrics.clear();
    this.detectedPatterns.clear();
    this.nextAttemptId = 1;
  }

  // Private methods

  private getAttemptsInPeriod(venue: string, timePeriod: string): ConnectionAttempt[] {
    const attempts = this.connectionAttempts.get(venue) || [];
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

    return attempts.filter(attempt => attempt.timestamp >= cutoffTime);
  }

  private calculateAverageResponseTime(attempts: ConnectionAttempt[]): number {
    const attemptsWithResponseTime = attempts.filter(a => a.response_time_ms !== undefined);
    if (attemptsWithResponseTime.length === 0) return 0;

    const totalResponseTime = attemptsWithResponseTime.reduce((sum, a) => sum + (a.response_time_ms || 0), 0);
    return totalResponseTime / attemptsWithResponseTime.length;
  }

  private categorizeFailures(failedAttempts: ConnectionAttempt[]): {
    timeout: number;
    network_error: number;
    authentication_error: number;
    server_error: number;
    unknown: number;
  } {
    const breakdown = {
      timeout: 0,
      network_error: 0,
      authentication_error: 0,
      server_error: 0,
      unknown: 0
    };

    for (const attempt of failedAttempts) {
      const reason = attempt.failure_reason?.toLowerCase() || '';
      
      if (reason.includes('timeout') || reason.includes('timed out')) {
        breakdown.timeout++;
      } else if (reason.includes('network') || reason.includes('connection') || reason.includes('socket')) {
        breakdown.network_error++;
      } else if (reason.includes('auth') || reason.includes('login') || reason.includes('credential')) {
        breakdown.authentication_error++;
      } else if (reason.includes('server') || reason.includes('500') || reason.includes('503')) {
        breakdown.server_error++;
      } else {
        breakdown.unknown++;
      }
    }

    return breakdown;
  }

  private calculateTrend(venue: string, timePeriod: string): 'improving' | 'degrading' | 'stable' {
    // Get two periods for comparison
    const currentPeriod = this.getAttemptsInPeriod(venue, timePeriod);
    
    // Calculate previous period based on current period length
    const now = new Date();
    let periodMs: number;
    
    switch (timePeriod) {
      case '1h': periodMs = 60 * 60 * 1000; break;
      case '6h': periodMs = 6 * 60 * 60 * 1000; break;
      case '24h': periodMs = 24 * 60 * 60 * 1000; break;
      case '7d': periodMs = 7 * 24 * 60 * 60 * 1000; break;
      case '30d': periodMs = 30 * 24 * 60 * 60 * 1000; break;
      default: periodMs = 24 * 60 * 60 * 1000;
    }

    const previousPeriodStart = new Date(now.getTime() - (2 * periodMs));
    const previousPeriodEnd = new Date(now.getTime() - periodMs);
    
    const allAttempts = this.connectionAttempts.get(venue) || [];
    const previousPeriod = allAttempts.filter(a => 
      a.timestamp >= previousPeriodStart && a.timestamp <= previousPeriodEnd
    );

    if (currentPeriod.length < 10 || previousPeriod.length < 10) {
      return 'stable'; // Not enough data for trend analysis
    }

    const currentSuccessRate = (currentPeriod.filter(a => a.success).length / currentPeriod.length) * 100;
    const previousSuccessRate = (previousPeriod.filter(a => a.success).length / previousPeriod.length) * 100;
    
    const difference = currentSuccessRate - previousSuccessRate;
    
    if (Math.abs(difference) < 5) return 'stable'; // Less than 5% change
    return difference > 0 ? 'improving' : 'degrading';
  }

  private updateSuccessRateMetrics(venue: string): void {
    // Update metrics for different time periods
    const periods = ['1h', '6h', '24h', '7d'];
    
    for (const period of periods) {
      this.getSuccessRateMetrics(venue, period);
    }
  }

  private analyzeFailurePatterns(venue: string): void {
    const recentAttempts = this.getRecentAttempts(venue, 100);
    const recentFailures = recentAttempts.filter(a => !a.success);
    
    if (recentFailures.length < 5) return; // Need minimum failures for pattern analysis

    // Analyze time-based patterns
    this.analyzeTimeBasedPatterns(venue, recentFailures);
    
    // Analyze frequency-based patterns
    this.analyzeFrequencyPatterns(venue, recentFailures);
    
    // Analyze sequential patterns
    this.analyzeSequentialPatterns(venue, recentAttempts);
  }

  private analyzeTimeBasedPatterns(venue: string, failures: ConnectionAttempt[]): void {
    // Check for failures clustered in specific time periods
    const hourlyFailures = new Map<number, number>();
    
    for (const failure of failures) {
      const hour = failure.timestamp.getHours();
      hourlyFailures.set(hour, (hourlyFailures.get(hour) || 0) + 1);
    }

    // Find hours with significantly high failure rates
    const avgFailuresPerHour = failures.length / 24;
    const threshold = avgFailuresPerHour * 3; // 3x average

    for (const [hour, count] of hourlyFailures) {
      if (count > threshold) {
        this.recordFailurePattern(venue, {
          pattern_type: 'time_based',
          description: `High failure rate detected between ${hour}:00-${hour + 1}:00`,
          occurrences: count,
          severity: count > threshold * 2 ? 'high' : 'medium',
          recommended_action: 'Investigate system load or external factors during this time period',
          first_detected: new Date(),
          last_occurrence: failures.filter(f => f.timestamp.getHours() === hour)[count - 1].timestamp
        });
      }
    }
  }

  private analyzeFrequencyPatterns(venue: string, failures: ConnectionAttempt[]): void {
    // Analyze failure frequency over time
    const now = new Date();
    const last15min = failures.filter(f => (now.getTime() - f.timestamp.getTime()) <= 15 * 60 * 1000);
    
    if (last15min.length >= 5) {
      this.recordFailurePattern(venue, {
        pattern_type: 'frequency_based',
        description: `High frequency failures: ${last15min.length} failures in last 15 minutes`,
        occurrences: last15min.length,
        severity: last15min.length >= 10 ? 'high' : 'medium',
        recommended_action: 'Check for network issues or service degradation',
        first_detected: last15min[0].timestamp,
        last_occurrence: last15min[last15min.length - 1].timestamp
      });
    }
  }

  private analyzeSequentialPatterns(venue: string, attempts: ConnectionAttempt[]): void {
    // Look for consecutive failures
    let consecutiveFailures = 0;
    let maxConsecutiveFailures = 0;
    let consecutiveFailureStart: Date | null = null;

    for (const attempt of attempts.slice(-50)) { // Last 50 attempts
      if (!attempt.success) {
        if (consecutiveFailures === 0) {
          consecutiveFailureStart = attempt.timestamp;
        }
        consecutiveFailures++;
        maxConsecutiveFailures = Math.max(maxConsecutiveFailures, consecutiveFailures);
      } else {
        consecutiveFailures = 0;
      }
    }

    if (maxConsecutiveFailures >= 5) {
      this.recordFailurePattern(venue, {
        pattern_type: 'sequential',
        description: `${maxConsecutiveFailures} consecutive connection failures detected`,
        occurrences: maxConsecutiveFailures,
        severity: maxConsecutiveFailures >= 10 ? 'high' : 'medium',
        recommended_action: 'Investigate persistent connectivity issues',
        first_detected: consecutiveFailureStart || new Date(),
        last_occurrence: new Date()
      });
    }
  }

  private recordFailurePattern(venue: string, pattern: FailurePattern): void {
    if (!this.detectedPatterns.has(venue)) {
      this.detectedPatterns.set(venue, []);
    }

    const patterns = this.detectedPatterns.get(venue)!;
    
    // Check if similar pattern already exists
    const existingPattern = patterns.find(p => 
      p.pattern_type === pattern.pattern_type && 
      p.description.includes(pattern.description.split(':')[0])
    );

    if (existingPattern) {
      // Update existing pattern
      existingPattern.occurrences++;
      existingPattern.last_occurrence = pattern.last_occurrence;
    } else {
      // Add new pattern
      patterns.push(pattern);
      
      // Notify callbacks
      this.callbacks.onPatternDetected.forEach(callback => callback(venue, pattern));
    }

    // Maintain pattern limit
    if (patterns.length > 20) {
      patterns.shift();
    }
  }

  private testStatisticalSignificance(
    n1: number, s1: number, // period 1: total attempts, successful attempts
    n2: number, s2: number  // period 2: total attempts, successful attempts
  ): boolean {
    if (n1 < 30 || n2 < 30) return false; // Need sufficient sample size

    const p1 = s1 / n1;
    const p2 = s2 / n2;
    const pooledP = (s1 + s2) / (n1 + n2);
    
    const standardError = Math.sqrt(pooledP * (1 - pooledP) * (1/n1 + 1/n2));
    const zScore = Math.abs(p1 - p2) / standardError;
    
    // Two-tailed test at 95% confidence level
    return zScore > 1.96;
  }
}

// Global instance for connection success rate tracking
export const connectionSuccessRateTracker = new ConnectionSuccessRateTracker();