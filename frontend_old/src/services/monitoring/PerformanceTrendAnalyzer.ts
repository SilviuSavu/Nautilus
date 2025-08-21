/**
 * Performance Trend Analyzer
 * Analyzes performance trends and provides insights for optimization
 */

export interface PerformanceTrend {
  metric_name: string;
  current_value: number;
  trend_direction: 'improving' | 'degrading' | 'stable';
  change_percent_24h: number;
  change_percent_7d: number;
  predicted_value_24h: number;
  confidence_score: number;
  baseline_value: number;
  volatility_score: number;
}

export interface PerformanceBaseline {
  metric_name: string;
  baseline_value: number;
  baseline_std_dev: number;
  established_at: Date;
  sample_count: number;
  confidence_level: number;
  auto_update: boolean;
}

export interface TrendAnalysisResult {
  trends: PerformanceTrend[];
  overall_performance_score: number;
  degrading_metrics_count: number;
  improving_metrics_count: number;
  stable_metrics_count: number;
  recommendations: string[];
  anomalies_detected: number;
}

export interface MetricDataPoint {
  timestamp: Date;
  value: number;
  metadata?: Record<string, any>;
}

export class PerformanceTrendAnalyzer {
  private metricData: Map<string, MetricDataPoint[]> = new Map();
  private baselines: Map<string, PerformanceBaseline> = new Map();
  private maxDataPoints: number = 10000;
  private trendCallbacks: ((result: TrendAnalysisResult) => void)[] = [];

  // Analysis parameters
  private analysisConfig = {
    min_samples_for_trend: 10,
    short_term_window_hours: 24,
    medium_term_window_hours: 168, // 7 days
    long_term_window_hours: 720, // 30 days
    volatility_threshold: 0.2, // 20%
    significant_change_threshold: 0.05, // 5%
    confidence_threshold: 0.7
  };

  /**
   * Add metric data point
   */
  addDataPoint(metricName: string, value: number, metadata?: Record<string, any>): void {
    if (!this.metricData.has(metricName)) {
      this.metricData.set(metricName, []);
    }

    const dataPoints = this.metricData.get(metricName)!;
    const dataPoint: MetricDataPoint = {
      timestamp: new Date(),
      value,
      metadata
    };

    dataPoints.push(dataPoint);

    // Maintain data point limit
    if (dataPoints.length > this.maxDataPoints) {
      dataPoints.shift();
    }

    // Auto-update baseline if enabled
    const baseline = this.baselines.get(metricName);
    if (baseline?.auto_update) {
      this.updateBaseline(metricName);
    }
  }

  /**
   * Establish baseline for a metric
   */
  establishBaseline(metricName: string, autoUpdate: boolean = true): PerformanceBaseline | null {
    const dataPoints = this.metricData.get(metricName);
    if (!dataPoints || dataPoints.length < 100) {
      console.warn(`PerformanceTrendAnalyzer: Insufficient data to establish baseline for ${metricName}`);
      return null;
    }

    // Use the oldest 30% of data as baseline (stable period)
    const baselineSampleSize = Math.floor(dataPoints.length * 0.3);
    const baselineData = dataPoints.slice(0, baselineSampleSize);
    
    const values = baselineData.map(dp => dp.value);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    const baseline: PerformanceBaseline = {
      metric_name: metricName,
      baseline_value: Math.round(mean * 1000) / 1000,
      baseline_std_dev: Math.round(stdDev * 1000) / 1000,
      established_at: new Date(),
      sample_count: baselineData.length,
      confidence_level: Math.min(1, baselineData.length / 1000), // More samples = higher confidence
      auto_update: autoUpdate
    };

    this.baselines.set(metricName, baseline);
    return baseline;
  }

  /**
   * Update existing baseline with recent data
   */
  updateBaseline(metricName: string): void {
    const baseline = this.baselines.get(metricName);
    const dataPoints = this.metricData.get(metricName);
    
    if (!baseline || !dataPoints || dataPoints.length < 100) return;

    // Use exponential moving average to update baseline
    const recentData = dataPoints.slice(-100); // Last 100 points
    const recentValues = recentData.map(dp => dp.value);
    const recentMean = recentValues.reduce((sum, val) => sum + val, 0) / recentValues.length;
    
    // Smooth update: 90% old baseline, 10% new data
    const smoothingFactor = 0.1;
    baseline.baseline_value = baseline.baseline_value * (1 - smoothingFactor) + recentMean * smoothingFactor;
    baseline.baseline_value = Math.round(baseline.baseline_value * 1000) / 1000;
    
    // Update metadata
    baseline.sample_count += recentData.length;
    baseline.confidence_level = Math.min(1, baseline.confidence_level + 0.01); // Slight confidence increase
  }

  /**
   * Analyze trends for all metrics
   */
  analyzeTrends(): TrendAnalysisResult {
    const trends: PerformanceTrend[] = [];
    let degradingCount = 0;
    let improvingCount = 0;
    let stableCount = 0;
    const recommendations: string[] = [];
    let anomaliesDetected = 0;

    for (const metricName of this.metricData.keys()) {
      const trend = this.analyzeMetricTrend(metricName);
      if (trend) {
        trends.push(trend);
        
        switch (trend.trend_direction) {
          case 'degrading': degradingCount++; break;
          case 'improving': improvingCount++; break;
          case 'stable': stableCount++; break;
        }
        
        if (trend.volatility_score > this.analysisConfig.volatility_threshold) {
          anomaliesDetected++;
        }
      }
    }

    // Generate recommendations
    if (degradingCount > 0) {
      recommendations.push(`${degradingCount} metrics showing performance degradation - investigate root causes`);
    }
    
    if (anomaliesDetected > 0) {
      recommendations.push(`${anomaliesDetected} metrics showing high volatility - monitor for instability`);
    }
    
    const highConfidenceTrends = trends.filter(t => t.confidence_score > this.analysisConfig.confidence_threshold);
    if (highConfidenceTrends.length < trends.length / 2) {
      recommendations.push('Insufficient data for reliable trend analysis - collect more metrics over time');
    }

    // Calculate overall performance score
    const overallScore = this.calculateOverallPerformanceScore(trends);

    return {
      trends,
      overall_performance_score: overallScore,
      degrading_metrics_count: degradingCount,
      improving_metrics_count: improvingCount,
      stable_metrics_count: stableCount,
      recommendations,
      anomalies_detected: anomaliesDetected
    };
  }

  /**
   * Analyze trend for specific metric
   */
  analyzeMetricTrend(metricName: string): PerformanceTrend | null {
    const dataPoints = this.metricData.get(metricName);
    if (!dataPoints || dataPoints.length < this.analysisConfig.min_samples_for_trend) {
      return null;
    }

    const now = new Date();
    const shortTermCutoff = new Date(now.getTime() - (this.analysisConfig.short_term_window_hours * 60 * 60 * 1000));
    const mediumTermCutoff = new Date(now.getTime() - (this.analysisConfig.medium_term_window_hours * 60 * 60 * 1000));

    // Get data for different time windows
    const shortTermData = dataPoints.filter(dp => dp.timestamp >= shortTermCutoff);
    const mediumTermData = dataPoints.filter(dp => dp.timestamp >= mediumTermCutoff);
    
    if (shortTermData.length === 0) return null;

    const currentValue = shortTermData[shortTermData.length - 1].value;
    const baseline = this.baselines.get(metricName);
    const baselineValue = baseline?.baseline_value || this.calculateMean(dataPoints.slice(0, Math.min(100, dataPoints.length)).map(dp => dp.value));

    // Calculate changes
    const change24h = this.calculatePercentChange(shortTermData);
    const change7d = this.calculatePercentChange(mediumTermData);

    // Determine trend direction
    const trendDirection = this.determineTrendDirection(change24h, change7d);

    // Calculate prediction and confidence
    const { prediction, confidence } = this.calculatePrediction(shortTermData);

    // Calculate volatility
    const volatilityScore = this.calculateVolatility(shortTermData.map(dp => dp.value));

    return {
      metric_name: metricName,
      current_value: Math.round(currentValue * 1000) / 1000,
      trend_direction: trendDirection,
      change_percent_24h: Math.round(change24h * 100) / 100,
      change_percent_7d: Math.round(change7d * 100) / 100,
      predicted_value_24h: Math.round(prediction * 1000) / 1000,
      confidence_score: Math.round(confidence * 100) / 100,
      baseline_value: Math.round(baselineValue * 1000) / 1000,
      volatility_score: Math.round(volatilityScore * 100) / 100
    };
  }

  /**
   * Detect performance anomalies
   */
  detectAnomalies(metricName: string, windowSize: number = 50): {
    anomalies: { timestamp: Date; value: number; severity: 'mild' | 'moderate' | 'severe' }[];
    anomaly_rate: number;
  } {
    const dataPoints = this.metricData.get(metricName);
    if (!dataPoints || dataPoints.length < windowSize) {
      return { anomalies: [], anomaly_rate: 0 };
    }

    const baseline = this.baselines.get(metricName);
    if (!baseline) {
      return { anomalies: [], anomaly_rate: 0 };
    }

    const anomalies: { timestamp: Date; value: number; severity: 'mild' | 'moderate' | 'severe' }[] = [];
    
    for (const dataPoint of dataPoints) {
      const deviation = Math.abs(dataPoint.value - baseline.baseline_value);
      const standardDeviations = deviation / baseline.baseline_std_dev;
      
      let severity: 'mild' | 'moderate' | 'severe' | null = null;
      
      if (standardDeviations > 3) severity = 'severe';
      else if (standardDeviations > 2) severity = 'moderate';
      else if (standardDeviations > 1.5) severity = 'mild';
      
      if (severity) {
        anomalies.push({
          timestamp: dataPoint.timestamp,
          value: dataPoint.value,
          severity
        });
      }
    }

    const anomalyRate = (anomalies.length / dataPoints.length) * 100;

    return {
      anomalies,
      anomaly_rate: Math.round(anomalyRate * 100) / 100
    };
  }

  /**
   * Get performance comparison between time periods
   */
  comparePerformancePeriods(metricName: string, period1Hours: number, period2Hours: number): {
    period1_avg: number;
    period2_avg: number;
    improvement_percent: number;
    statistical_significance: boolean;
  } {
    const dataPoints = this.metricData.get(metricName);
    if (!dataPoints) {
      return { period1_avg: 0, period2_avg: 0, improvement_percent: 0, statistical_significance: false };
    }

    const now = new Date();
    const period1Start = new Date(now.getTime() - (period1Hours * 60 * 60 * 1000));
    const period2Start = new Date(now.getTime() - ((period1Hours + period2Hours) * 60 * 60 * 1000));
    const period2End = period1Start;

    const period1Data = dataPoints.filter(dp => dp.timestamp >= period1Start);
    const period2Data = dataPoints.filter(dp => dp.timestamp >= period2Start && dp.timestamp <= period2End);

    if (period1Data.length === 0 || period2Data.length === 0) {
      return { period1_avg: 0, period2_avg: 0, improvement_percent: 0, statistical_significance: false };
    }

    const period1Avg = this.calculateMean(period1Data.map(dp => dp.value));
    const period2Avg = this.calculateMean(period2Data.map(dp => dp.value));
    
    const improvementPercent = ((period1Avg - period2Avg) / period2Avg) * 100;
    const statSig = this.testStatisticalSignificance(
      period1Data.map(dp => dp.value),
      period2Data.map(dp => dp.value)
    );

    return {
      period1_avg: Math.round(period1Avg * 1000) / 1000,
      period2_avg: Math.round(period2Avg * 1000) / 1000,
      improvement_percent: Math.round(improvementPercent * 100) / 100,
      statistical_significance: statSig
    };
  }

  /**
   * Get all baselines
   */
  getBaselines(): PerformanceBaseline[] {
    return Array.from(this.baselines.values());
  }

  /**
   * Set analysis configuration
   */
  setAnalysisConfig(config: Partial<typeof this.analysisConfig>): void {
    this.analysisConfig = { ...this.analysisConfig, ...config };
  }

  /**
   * Add trend analysis callback
   */
  onTrendUpdate(callback: (result: TrendAnalysisResult) => void): void {
    this.trendCallbacks.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: (result: TrendAnalysisResult) => void): void {
    const index = this.trendCallbacks.indexOf(callback);
    if (index > -1) {
      this.trendCallbacks.splice(index, 1);
    }
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.metricData.clear();
    this.baselines.clear();
  }

  // Private methods

  private calculatePercentChange(dataPoints: MetricDataPoint[]): number {
    if (dataPoints.length < 2) return 0;
    
    const firstValue = dataPoints[0].value;
    const lastValue = dataPoints[dataPoints.length - 1].value;
    
    if (firstValue === 0) return 0;
    
    return ((lastValue - firstValue) / firstValue) * 100;
  }

  private determineTrendDirection(change24h: number, change7d: number): 'improving' | 'degrading' | 'stable' {
    const threshold = this.analysisConfig.significant_change_threshold * 100;
    
    // Weight recent changes more heavily
    const weightedChange = (change24h * 0.7) + (change7d * 0.3);
    
    if (Math.abs(weightedChange) < threshold) return 'stable';
    return weightedChange > 0 ? 'degrading' : 'improving'; // Assuming lower values are better
  }

  private calculatePrediction(dataPoints: MetricDataPoint[]): { prediction: number; confidence: number } {
    if (dataPoints.length < 3) {
      const lastValue = dataPoints[dataPoints.length - 1]?.value || 0;
      return { prediction: lastValue, confidence: 0 };
    }

    // Simple linear regression prediction
    const n = dataPoints.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = dataPoints.map(dp => dp.value);
    
    const { slope, intercept, correlation } = this.linearRegression(x, y);
    
    // Predict next value (n)
    const prediction = slope * n + intercept;
    const confidence = Math.abs(correlation);
    
    return { 
      prediction: Math.max(0, prediction), // Ensure non-negative
      confidence 
    };
  }

  private calculateVolatility(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = this.calculateMean(values);
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    // Coefficient of variation as volatility measure
    return mean === 0 ? 0 : stdDev / Math.abs(mean);
  }

  private calculateOverallPerformanceScore(trends: PerformanceTrend[]): number {
    if (trends.length === 0) return 100;
    
    let totalScore = 0;
    
    for (const trend of trends) {
      let score = 100;
      
      // Penalize degrading trends
      if (trend.trend_direction === 'degrading') {
        score -= Math.abs(trend.change_percent_24h) * 2; // Double penalty for recent degradation
        score -= Math.abs(trend.change_percent_7d);
      }
      
      // Reward improving trends
      if (trend.trend_direction === 'improving') {
        score += Math.abs(trend.change_percent_24h);
      }
      
      // Penalize high volatility
      score -= trend.volatility_score * 50;
      
      // Weight by confidence
      score *= trend.confidence_score;
      
      totalScore += Math.max(0, Math.min(100, score));
    }
    
    return Math.round((totalScore / trends.length) * 100) / 100;
  }

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private linearRegression(x: number[], y: number[]): { slope: number; intercept: number; correlation: number } {
    const n = x.length;
    if (n === 0) return { slope: 0, intercept: 0, correlation: 0 };
    
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

  private testStatisticalSignificance(group1: number[], group2: number[]): boolean {
    if (group1.length < 10 || group2.length < 10) return false;
    
    const mean1 = this.calculateMean(group1);
    const mean2 = this.calculateMean(group2);
    
    const var1 = group1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / group1.length;
    const var2 = group2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / group2.length;
    
    const standardError = Math.sqrt((var1 / group1.length) + (var2 / group2.length));
    const tStat = Math.abs(mean1 - mean2) / standardError;
    
    // Simple significance test (approximately p < 0.05)
    return tStat > 1.96;
  }
}

// Global instance for performance trend analysis
export const performanceTrendAnalyzer = new PerformanceTrendAnalyzer();