/**
 * Connection Quality Scorer
 * Provides comprehensive connection quality scoring algorithms
 */

import { ConnectionQuality } from '../../types/monitoring';

export interface QualityScore {
  overall_score: number; // 0-100
  component_scores: {
    stability: number;
    latency: number;
    throughput: number;
    reliability: number;
    data_quality: number;
  };
  score_breakdown: {
    base_score: number;
    stability_modifier: number;
    latency_modifier: number;
    throughput_modifier: number;
    reliability_modifier: number;
    data_quality_modifier: number;
  };
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  recommendations: string[];
  calculated_at: Date;
}

export interface ScoringWeights {
  stability: number;    // How much stability affects the score (0-1)
  latency: number;      // How much latency affects the score (0-1)
  throughput: number;   // How much throughput affects the score (0-1)
  reliability: number;  // How much reliability affects the score (0-1)
  data_quality: number; // How much data quality affects the score (0-1)
}

export interface ScoringBenchmarks {
  excellent_threshold: number; // Score above this = excellent (90+)
  good_threshold: number;      // Score above this = good (75+)
  fair_threshold: number;      // Score above this = fair (60+)
  poor_threshold: number;      // Score above this = poor (40+)
  // Below poor_threshold = critical
}

export class ConnectionQualityScorer {
  private defaultWeights: ScoringWeights = {
    stability: 0.25,    // 25% weight
    latency: 0.25,      // 25% weight
    throughput: 0.15,   // 15% weight
    reliability: 0.20,  // 20% weight
    data_quality: 0.15  // 15% weight
  };

  private benchmarks: ScoringBenchmarks = {
    excellent_threshold: 90,
    good_threshold: 75,
    fair_threshold: 60,
    poor_threshold: 40
  };

  // Latency benchmarks (milliseconds)
  private latencyBenchmarks = {
    excellent: 25,  // < 25ms = excellent
    good: 50,       // < 50ms = good
    fair: 100,      // < 100ms = fair
    poor: 200       // < 200ms = poor, >= 200ms = critical
  };

  // Throughput benchmarks (Mbps)
  private throughputBenchmarks = {
    excellent: 100,  // > 100 Mbps = excellent
    good: 50,        // > 50 Mbps = good
    fair: 10,        // > 10 Mbps = fair
    poor: 1          // > 1 Mbps = poor, <= 1 Mbps = critical
  };

  /**
   * Calculate comprehensive quality score for a connection
   */
  calculateQualityScore(
    connectionQuality: ConnectionQuality,
    customWeights?: Partial<ScoringWeights>
  ): QualityScore {
    const weights = { ...this.defaultWeights, ...customWeights };
    
    // Calculate individual component scores
    const stabilityScore = this.calculateStabilityScore(connectionQuality);
    const latencyScore = this.calculateLatencyScore(connectionQuality);
    const throughputScore = this.calculateThroughputScore(connectionQuality);
    const reliabilityScore = this.calculateReliabilityScore(connectionQuality);
    const dataQualityScore = this.calculateDataQualityScore(connectionQuality);

    // Calculate weighted overall score
    const baseScore = 100;
    const stabilityModifier = (stabilityScore - 100) * weights.stability;
    const latencyModifier = (latencyScore - 100) * weights.latency;
    const throughputModifier = (throughputScore - 100) * weights.throughput;
    const reliabilityModifier = (reliabilityScore - 100) * weights.reliability;
    const dataQualityModifier = (dataQualityScore - 100) * weights.data_quality;

    const overallScore = Math.max(0, Math.min(100, baseScore + 
      stabilityModifier + 
      latencyModifier + 
      throughputModifier + 
      reliabilityModifier + 
      dataQualityModifier
    ));

    const grade = this.calculateGrade(overallScore);
    const recommendations = this.generateRecommendations(connectionQuality, {
      stability: stabilityScore,
      latency: latencyScore,
      throughput: throughputScore,
      reliability: reliabilityScore,
      data_quality: dataQualityScore
    });

    return {
      overall_score: Math.round(overallScore * 100) / 100,
      component_scores: {
        stability: Math.round(stabilityScore * 100) / 100,
        latency: Math.round(latencyScore * 100) / 100,
        throughput: Math.round(throughputScore * 100) / 100,
        reliability: Math.round(reliabilityScore * 100) / 100,
        data_quality: Math.round(dataQualityScore * 100) / 100
      },
      score_breakdown: {
        base_score: baseScore,
        stability_modifier: Math.round(stabilityModifier * 100) / 100,
        latency_modifier: Math.round(latencyModifier * 100) / 100,
        throughput_modifier: Math.round(throughputModifier * 100) / 100,
        reliability_modifier: Math.round(reliabilityModifier * 100) / 100,
        data_quality_modifier: Math.round(dataQualityModifier * 100) / 100
      },
      grade,
      recommendations,
      calculated_at: new Date()
    };
  }

  /**
   * Calculate stability score based on uptime and disconnect frequency
   */
  calculateStabilityScore(connection: ConnectionQuality): number {
    const uptime = connection.uptime_percent_24h;
    const disconnects = connection.disconnect_count_24h;
    
    let score = 100;
    
    // Uptime penalty
    if (uptime < 99.9) score -= (99.9 - uptime) * 10; // Heavy penalty for downtime
    if (uptime < 99) score -= (99 - uptime) * 5;      // Additional penalty
    if (uptime < 95) score -= (95 - uptime) * 2;      // More penalty for poor uptime
    
    // Disconnect frequency penalty
    if (disconnects > 0) score -= disconnects * 5;    // 5 points per disconnect
    if (disconnects > 10) score -= (disconnects - 10) * 2; // Extra penalty for frequent disconnects
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate latency score based on response time and jitter
   */
  calculateLatencyScore(connection: ConnectionQuality): number {
    const responseTime = connection.performance_metrics.response_time_ms;
    
    let score = 100;
    
    // Response time scoring
    if (responseTime <= this.latencyBenchmarks.excellent) {
      score = 100; // Excellent latency
    } else if (responseTime <= this.latencyBenchmarks.good) {
      score = 90 - ((responseTime - this.latencyBenchmarks.excellent) / (this.latencyBenchmarks.good - this.latencyBenchmarks.excellent)) * 10;
    } else if (responseTime <= this.latencyBenchmarks.fair) {
      score = 75 - ((responseTime - this.latencyBenchmarks.good) / (this.latencyBenchmarks.fair - this.latencyBenchmarks.good)) * 15;
    } else if (responseTime <= this.latencyBenchmarks.poor) {
      score = 50 - ((responseTime - this.latencyBenchmarks.fair) / (this.latencyBenchmarks.poor - this.latencyBenchmarks.fair)) * 25;
    } else {
      score = Math.max(0, 25 - (responseTime - this.latencyBenchmarks.poor) / 10);
    }
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate throughput score based on bandwidth utilization
   */
  calculateThroughputScore(connection: ConnectionQuality): number {
    const throughputMbps = connection.performance_metrics.throughput_mbps;
    
    let score = 100;
    
    // Throughput scoring (higher is better)
    if (throughputMbps >= this.throughputBenchmarks.excellent) {
      score = 100; // Excellent throughput
    } else if (throughputMbps >= this.throughputBenchmarks.good) {
      score = 90 - ((this.throughputBenchmarks.excellent - throughputMbps) / (this.throughputBenchmarks.excellent - this.throughputBenchmarks.good)) * 10;
    } else if (throughputMbps >= this.throughputBenchmarks.fair) {
      score = 75 - ((this.throughputBenchmarks.good - throughputMbps) / (this.throughputBenchmarks.good - this.throughputBenchmarks.fair)) * 15;
    } else if (throughputMbps >= this.throughputBenchmarks.poor) {
      score = 50 - ((this.throughputBenchmarks.fair - throughputMbps) / (this.throughputBenchmarks.fair - this.throughputBenchmarks.poor)) * 25;
    } else {
      score = Math.max(0, 25 * (throughputMbps / this.throughputBenchmarks.poor));
    }
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate reliability score based on error rates and reconnection stats
   */
  calculateReliabilityScore(connection: ConnectionQuality): number {
    const errorRate = connection.performance_metrics.error_rate_percent;
    const reconnectAttempts = connection.reconnection_stats.reconnect_attempts_24h;
    const autoReconnectEnabled = connection.reconnection_stats.auto_reconnect_enabled;
    
    let score = 100;
    
    // Error rate penalty
    if (errorRate > 0) score -= errorRate * 10; // 10 points per 1% error rate
    if (errorRate > 5) score -= (errorRate - 5) * 5; // Additional penalty for high error rates
    
    // Reconnection attempts penalty
    if (reconnectAttempts > 0) score -= reconnectAttempts * 2; // 2 points per reconnect attempt
    if (reconnectAttempts > 20) score -= (reconnectAttempts - 20); // Extra penalty for frequent reconnects
    
    // Auto-reconnect bonus
    if (autoReconnectEnabled) score += 5; // Small bonus for having auto-reconnect
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate data quality score based on message rates and data integrity
   */
  calculateDataQualityScore(connection: ConnectionQuality): number {
    const messageRate = connection.data_quality.message_rate_per_sec;
    const duplicatePercent = connection.data_quality.duplicate_messages_percent;
    const outOfSequencePercent = connection.data_quality.out_of_sequence_percent;
    const staleDataPercent = connection.data_quality.stale_data_percent;
    
    let score = 100;
    
    // Message rate scoring (higher is generally better for trading data)
    if (messageRate === 0) {
      score -= 30; // Significant penalty for no data
    } else if (messageRate < 1) {
      score -= 15; // Penalty for very low message rate
    } else if (messageRate < 10) {
      score -= 5; // Small penalty for low message rate
    }
    
    // Data quality penalties
    if (duplicatePercent > 0) score -= duplicatePercent * 5; // 5 points per 1% duplicates
    if (outOfSequencePercent > 0) score -= outOfSequencePercent * 8; // 8 points per 1% out of sequence
    if (staleDataPercent > 0) score -= staleDataPercent * 10; // 10 points per 1% stale data
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate letter grade based on overall score
   */
  calculateGrade(overallScore: number): 'A' | 'B' | 'C' | 'D' | 'F' {
    if (overallScore >= this.benchmarks.excellent_threshold) return 'A';
    if (overallScore >= this.benchmarks.good_threshold) return 'B';
    if (overallScore >= this.benchmarks.fair_threshold) return 'C';
    if (overallScore >= this.benchmarks.poor_threshold) return 'D';
    return 'F';
  }

  /**
   * Generate recommendations based on component scores
   */
  generateRecommendations(
    connection: ConnectionQuality, 
    componentScores: { [key: string]: number }
  ): string[] {
    const recommendations: string[] = [];
    
    // Stability recommendations
    if (componentScores.stability < 80) {
      if (connection.uptime_percent_24h < 99) {
        recommendations.push('Improve connection stability - uptime is below 99%');
      }
      if (connection.disconnect_count_24h > 5) {
        recommendations.push('Investigate frequent disconnections - reduce connection instability');
      }
    }
    
    // Latency recommendations
    if (componentScores.latency < 75) {
      const latency = connection.performance_metrics.response_time_ms;
      if (latency > this.latencyBenchmarks.fair) {
        recommendations.push(`Optimize network latency - current response time (${latency}ms) exceeds acceptable threshold`);
      }
      recommendations.push('Consider network optimization or closer server location');
    }
    
    // Throughput recommendations
    if (componentScores.throughput < 70) {
      const throughput = connection.performance_metrics.throughput_mbps;
      if (throughput < this.throughputBenchmarks.fair) {
        recommendations.push(`Increase bandwidth capacity - current throughput (${throughput} Mbps) is insufficient`);
      }
      recommendations.push('Evaluate network infrastructure and bandwidth allocation');
    }
    
    // Reliability recommendations
    if (componentScores.reliability < 75) {
      const errorRate = connection.performance_metrics.error_rate_percent;
      if (errorRate > 2) {
        recommendations.push(`Address high error rate (${errorRate}%) - investigate root cause of connection errors`);
      }
      if (connection.reconnection_stats.reconnect_attempts_24h > 10) {
        recommendations.push('Reduce reconnection frequency - investigate connection persistence issues');
      }
      if (!connection.reconnection_stats.auto_reconnect_enabled) {
        recommendations.push('Enable auto-reconnection for improved reliability');
      }
    }
    
    // Data quality recommendations
    if (componentScores.data_quality < 70) {
      if (connection.data_quality.message_rate_per_sec < 1) {
        recommendations.push('Investigate low data message rate - ensure data feed is active');
      }
      if (connection.data_quality.duplicate_messages_percent > 1) {
        recommendations.push('Address duplicate message issues - implement deduplication logic');
      }
      if (connection.data_quality.out_of_sequence_percent > 0.5) {
        recommendations.push('Fix out-of-sequence message handling - implement proper message ordering');
      }
      if (connection.data_quality.stale_data_percent > 0.5) {
        recommendations.push('Reduce stale data percentage - improve data freshness monitoring');
      }
    }
    
    // Overall recommendations
    const overallScore = (Object.values(componentScores).reduce((sum, score) => sum + score, 0) / Object.keys(componentScores).length);
    
    if (overallScore < this.benchmarks.poor_threshold) {
      recommendations.push('CRITICAL: Connection quality is severely degraded - immediate attention required');
      recommendations.push('Consider failover to backup connection or alternative data source');
    } else if (overallScore < this.benchmarks.fair_threshold) {
      recommendations.push('WARNING: Connection quality is below acceptable standards - prioritize improvements');
    } else if (overallScore < this.benchmarks.good_threshold) {
      recommendations.push('NOTICE: Connection quality has room for improvement - consider optimization');
    }
    
    return recommendations;
  }

  /**
   * Compare quality scores between two connections
   */
  compareConnections(
    connection1: ConnectionQuality,
    connection2: ConnectionQuality,
    customWeights?: Partial<ScoringWeights>
  ): {
    connection1_score: QualityScore;
    connection2_score: QualityScore;
    better_connection: string;
    score_difference: number;
    component_comparison: {
      component: string;
      connection1_score: number;
      connection2_score: number;
      winner: string;
    }[];
  } {
    const score1 = this.calculateQualityScore(connection1, customWeights);
    const score2 = this.calculateQualityScore(connection2, customWeights);
    
    const scoreDifference = Math.abs(score1.overall_score - score2.overall_score);
    const betterConnection = score1.overall_score > score2.overall_score ? connection1.venue_name : connection2.venue_name;
    
    const componentComparison = Object.keys(score1.component_scores).map(component => ({
      component,
      connection1_score: (score1.component_scores as any)[component],
      connection2_score: (score2.component_scores as any)[component],
      winner: (score1.component_scores as any)[component] > (score2.component_scores as any)[component] ? 
        connection1.venue_name : connection2.venue_name
    }));
    
    return {
      connection1_score: score1,
      connection2_score: score2,
      better_connection: betterConnection,
      score_difference: Math.round(scoreDifference * 100) / 100,
      component_comparison: componentComparison
    };
  }

  /**
   * Set custom scoring weights
   */
  setWeights(weights: Partial<ScoringWeights>): void {
    this.defaultWeights = { ...this.defaultWeights, ...weights };
    
    // Normalize weights to sum to 1.0
    const totalWeight = Object.values(this.defaultWeights).reduce((sum, weight) => sum + weight, 0);
    if (totalWeight !== 1.0) {
      for (const key in this.defaultWeights) {
        (this.defaultWeights as any)[key] = (this.defaultWeights as any)[key] / totalWeight;
      }
    }
  }

  /**
   * Set custom scoring benchmarks
   */
  setBenchmarks(benchmarks: Partial<ScoringBenchmarks>): void {
    this.benchmarks = { ...this.benchmarks, ...benchmarks };
  }

  /**
   * Get current scoring configuration
   */
  getScoringConfig(): {
    weights: ScoringWeights;
    benchmarks: ScoringBenchmarks;
    latency_benchmarks: typeof this.latencyBenchmarks;
    throughput_benchmarks: typeof this.throughputBenchmarks;
  } {
    return {
      weights: { ...this.defaultWeights },
      benchmarks: { ...this.benchmarks },
      latency_benchmarks: { ...this.latencyBenchmarks },
      throughput_benchmarks: { ...this.throughputBenchmarks }
    };
  }

  /**
   * Reset to default configuration
   */
  resetToDefaults(): void {
    this.defaultWeights = {
      stability: 0.25,
      latency: 0.25,
      throughput: 0.15,
      reliability: 0.20,
      data_quality: 0.15
    };
    
    this.benchmarks = {
      excellent_threshold: 90,
      good_threshold: 75,
      fair_threshold: 60,
      poor_threshold: 40
    };
  }
}

// Global instance for connection quality scoring
export const connectionQualityScorer = new ConnectionQualityScorer();