/**
 * Capacity Utilization Tracker
 * Monitors system capacity usage and predicts when resources will be exhausted
 */

import { SystemResourceSnapshot } from './SystemResourceMonitor';

export interface CapacityMetric {
  resource_type: 'cpu' | 'memory' | 'network' | 'disk' | 'connections';
  current_utilization_percent: number;
  average_utilization_percent: number;
  peak_utilization_percent: number;
  capacity_limit: number;
  current_usage: number;
  available_capacity: number;
  last_updated: Date;
}

export interface CapacityPrediction {
  resource_type: string;
  exhaustion_prediction_days?: number;
  trend_direction: 'increasing' | 'decreasing' | 'stable';
  confidence_score: number; // 0-1
  recommended_actions: string[];
  warning_threshold_days: number;
  critical_threshold_days: number;
}

export interface UtilizationAlert {
  id: string;
  resource_type: string;
  alert_type: 'threshold_breach' | 'capacity_warning' | 'trend_alert';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  current_value: number;
  threshold_value: number;
  triggered_at: Date;
  resolved: boolean;
}

export class CapacityUtilizationTracker {
  private utilizationHistory: Map<string, number[]> = new Map();
  private capacityMetrics: Map<string, CapacityMetric> = new Map();
  private alerts: UtilizationAlert[] = [];
  private maxHistorySize: number = 1000;
  private callbacks: ((alert: UtilizationAlert) => void)[] = [];

  // Capacity limits (configurable)
  private capacityLimits = {
    cpu: 100, // 100%
    memory: 16 * 1024, // 16GB in MB
    network: 1000 * 1024 * 1024, // 1Gbps in bytes/sec
    disk: 1000 * 1024, // 1TB in MB
    connections: 10000 // Max connections
  };

  // Alert thresholds
  private thresholds = {
    warning: 75, // 75%
    critical: 90, // 90%
    prediction_warning_days: 30,
    prediction_critical_days: 7
  };

  /**
   * Update capacity metrics from system snapshot
   */
  updateFromSnapshot(snapshot: SystemResourceSnapshot): void {
    this.updateCPUCapacity(snapshot);
    this.updateMemoryCapacity(snapshot);
    this.updateNetworkCapacity(snapshot);
    this.updateConnectionCapacity(snapshot);
    
    // Check for threshold alerts
    this.checkThresholdAlerts();
    
    // Update predictions
    this.updateCapacityPredictions();
  }

  /**
   * Get current capacity metrics
   */
  getCapacityMetrics(): CapacityMetric[] {
    return Array.from(this.capacityMetrics.values());
  }

  /**
   * Get capacity metric for specific resource
   */
  getCapacityMetric(resourceType: string): CapacityMetric | null {
    return this.capacityMetrics.get(resourceType) || null;
  }

  /**
   * Get capacity predictions
   */
  getCapacityPredictions(): CapacityPrediction[] {
    const predictions: CapacityPrediction[] = [];
    
    for (const [resourceType, metric] of this.capacityMetrics) {
      const prediction = this.calculateCapacityPrediction(resourceType, metric);
      predictions.push(prediction);
    }
    
    return predictions;
  }

  /**
   * Get utilization trend for a resource
   */
  getUtilizationTrend(resourceType: string, timeRangePoints: number = 100): {
    trend: 'increasing' | 'decreasing' | 'stable';
    slope: number;
    correlation: number;
    dataPoints: number[];
  } {
    const history = this.utilizationHistory.get(resourceType) || [];
    const recentHistory = history.slice(-timeRangePoints);
    
    if (recentHistory.length < 10) {
      return {
        trend: 'stable',
        slope: 0,
        correlation: 0,
        dataPoints: recentHistory
      };
    }
    
    // Calculate linear regression
    const n = recentHistory.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = recentHistory;
    
    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = y.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + (val * y[i]), 0);
    const sumXX = x.reduce((sum, val) => sum + (val * val), 0);
    const sumYY = y.reduce((sum, val) => sum + (val * val), 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const correlation = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    let trend: 'increasing' | 'decreasing' | 'stable';
    if (Math.abs(slope) < 0.1) trend = 'stable';
    else trend = slope > 0 ? 'increasing' : 'decreasing';
    
    return {
      trend,
      slope: Math.round(slope * 1000) / 1000,
      correlation: Math.round(Math.abs(correlation) * 1000) / 1000,
      dataPoints: recentHistory
    };
  }

  /**
   * Get active capacity alerts
   */
  getActiveAlerts(severity?: string): UtilizationAlert[] {
    let alerts = this.alerts.filter(alert => !alert.resolved);
    
    if (severity) {
      alerts = alerts.filter(alert => alert.severity === severity);
    }
    
    return alerts;
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.resolved = true;
      return true;
    }
    return false;
  }

  /**
   * Set capacity limits
   */
  setCapacityLimits(limits: Partial<typeof this.capacityLimits>): void {
    this.capacityLimits = { ...this.capacityLimits, ...limits };
  }

  /**
   * Set alert thresholds
   */
  setThresholds(thresholds: Partial<typeof this.thresholds>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
  }

  /**
   * Get capacity summary
   */
  getCapacitySummary(): {
    overall_health_score: number;
    resources_at_risk: number;
    critical_alerts: number;
    predicted_exhaustions: number;
    avg_utilization: number;
  } {
    const metrics = this.getCapacityMetrics();
    const activeAlerts = this.getActiveAlerts();
    const predictions = this.getCapacityPredictions();
    
    if (metrics.length === 0) {
      return {
        overall_health_score: 100,
        resources_at_risk: 0,
        critical_alerts: 0,
        predicted_exhaustions: 0,
        avg_utilization: 0
      };
    }
    
    const avgUtilization = metrics.reduce((sum, m) => sum + m.current_utilization_percent, 0) / metrics.length;
    const resourcesAtRisk = metrics.filter(m => m.current_utilization_percent > this.thresholds.warning).length;
    const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical').length;
    const predictedExhaustions = predictions.filter(p => 
      p.exhaustion_prediction_days && p.exhaustion_prediction_days < this.thresholds.prediction_warning_days
    ).length;
    
    // Calculate health score (0-100)
    let healthScore = 100;
    healthScore -= (avgUtilization / 100) * 30; // Utilization penalty
    healthScore -= resourcesAtRisk * 15; // Resources at risk penalty
    healthScore -= criticalAlerts * 20; // Critical alerts penalty
    healthScore -= predictedExhaustions * 10; // Prediction penalty
    
    return {
      overall_health_score: Math.max(0, Math.round(healthScore)),
      resources_at_risk: resourcesAtRisk,
      critical_alerts: criticalAlerts,
      predicted_exhaustions: predictedExhaustions,
      avg_utilization: Math.round(avgUtilization * 100) / 100
    };
  }

  /**
   * Add callback for alerts
   */
  onAlert(callback: (alert: UtilizationAlert) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: (alert: UtilizationAlert) => void): void {
    const index = this.callbacks.indexOf(callback);
    if (index > -1) {
      this.callbacks.splice(index, 1);
    }
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.utilizationHistory.clear();
    this.capacityMetrics.clear();
    this.alerts = [];
  }

  // Private methods

  private updateCPUCapacity(snapshot: SystemResourceSnapshot): void {
    const utilizationPercent = snapshot.cpu_usage_percent;
    const currentUsage = snapshot.cpu_usage_percent;
    
    this.updateResourceHistory('cpu', utilizationPercent);
    
    const history = this.utilizationHistory.get('cpu') || [];
    const avgUtilization = history.reduce((sum, val) => sum + val, 0) / history.length;
    const peakUtilization = Math.max(...history);
    
    const metric: CapacityMetric = {
      resource_type: 'cpu',
      current_utilization_percent: utilizationPercent,
      average_utilization_percent: avgUtilization,
      peak_utilization_percent: peakUtilization,
      capacity_limit: this.capacityLimits.cpu,
      current_usage: currentUsage,
      available_capacity: this.capacityLimits.cpu - currentUsage,
      last_updated: snapshot.timestamp
    };
    
    this.capacityMetrics.set('cpu', metric);
  }

  private updateMemoryCapacity(snapshot: SystemResourceSnapshot): void {
    const totalMemory = snapshot.memory_usage_mb + snapshot.available_memory_mb;
    const utilizationPercent = (snapshot.memory_usage_mb / totalMemory) * 100;
    
    this.updateResourceHistory('memory', utilizationPercent);
    
    const history = this.utilizationHistory.get('memory') || [];
    const avgUtilization = history.reduce((sum, val) => sum + val, 0) / history.length;
    const peakUtilization = Math.max(...history);
    
    const metric: CapacityMetric = {
      resource_type: 'memory',
      current_utilization_percent: utilizationPercent,
      average_utilization_percent: avgUtilization,
      peak_utilization_percent: peakUtilization,
      capacity_limit: this.capacityLimits.memory,
      current_usage: snapshot.memory_usage_mb,
      available_capacity: snapshot.available_memory_mb,
      last_updated: snapshot.timestamp
    };
    
    this.capacityMetrics.set('memory', metric);
  }

  private updateNetworkCapacity(snapshot: SystemResourceSnapshot): void {
    const utilizationPercent = (snapshot.network_bytes_per_sec / this.capacityLimits.network) * 100;
    
    this.updateResourceHistory('network', utilizationPercent);
    
    const history = this.utilizationHistory.get('network') || [];
    const avgUtilization = history.reduce((sum, val) => sum + val, 0) / history.length;
    const peakUtilization = Math.max(...history);
    
    const metric: CapacityMetric = {
      resource_type: 'network',
      current_utilization_percent: Math.min(100, utilizationPercent),
      average_utilization_percent: avgUtilization,
      peak_utilization_percent: peakUtilization,
      capacity_limit: this.capacityLimits.network,
      current_usage: snapshot.network_bytes_per_sec,
      available_capacity: this.capacityLimits.network - snapshot.network_bytes_per_sec,
      last_updated: snapshot.timestamp
    };
    
    this.capacityMetrics.set('network', metric);
  }

  private updateConnectionCapacity(snapshot: SystemResourceSnapshot): void {
    const utilizationPercent = (snapshot.active_connections / this.capacityLimits.connections) * 100;
    
    this.updateResourceHistory('connections', utilizationPercent);
    
    const history = this.utilizationHistory.get('connections') || [];
    const avgUtilization = history.reduce((sum, val) => sum + val, 0) / history.length;
    const peakUtilization = Math.max(...history);
    
    const metric: CapacityMetric = {
      resource_type: 'connections',
      current_utilization_percent: utilizationPercent,
      average_utilization_percent: avgUtilization,
      peak_utilization_percent: peakUtilization,
      capacity_limit: this.capacityLimits.connections,
      current_usage: snapshot.active_connections,
      available_capacity: this.capacityLimits.connections - snapshot.active_connections,
      last_updated: snapshot.timestamp
    };
    
    this.capacityMetrics.set('connections', metric);
  }

  private updateResourceHistory(resourceType: string, value: number): void {
    if (!this.utilizationHistory.has(resourceType)) {
      this.utilizationHistory.set(resourceType, []);
    }
    
    const history = this.utilizationHistory.get(resourceType)!;
    history.push(value);
    
    // Maintain history size limit
    if (history.length > this.maxHistorySize) {
      history.shift();
    }
  }

  private calculateCapacityPrediction(resourceType: string, metric: CapacityMetric): CapacityPrediction {
    const trend = this.getUtilizationTrend(resourceType);
    const recommended_actions: string[] = [];
    
    let exhaustion_prediction_days: number | undefined;
    
    if (trend.trend === 'increasing' && trend.slope > 0) {
      // Calculate time to exhaustion based on trend
      const remainingCapacity = 100 - metric.current_utilization_percent;
      const dailyIncrease = trend.slope * 24; // Assuming measurements are hourly
      
      if (dailyIncrease > 0) {
        exhaustion_prediction_days = Math.round(remainingCapacity / dailyIncrease);
      }
    }
    
    // Generate recommendations
    if (metric.current_utilization_percent > this.thresholds.critical) {
      recommended_actions.push(`URGENT: ${resourceType.toUpperCase()} usage is critical (${metric.current_utilization_percent.toFixed(1)}%)`);
      recommended_actions.push(`Consider immediate capacity expansion`);
    } else if (metric.current_utilization_percent > this.thresholds.warning) {
      recommended_actions.push(`${resourceType.toUpperCase()} usage is high (${metric.current_utilization_percent.toFixed(1)}%)`);
      recommended_actions.push(`Plan for capacity expansion`);
    }
    
    if (exhaustion_prediction_days && exhaustion_prediction_days < this.thresholds.prediction_critical_days) {
      recommended_actions.push(`Critical: ${resourceType.toUpperCase()} capacity will be exhausted in ${exhaustion_prediction_days} days`);
    } else if (exhaustion_prediction_days && exhaustion_prediction_days < this.thresholds.prediction_warning_days) {
      recommended_actions.push(`Warning: ${resourceType.toUpperCase()} capacity will be exhausted in ${exhaustion_prediction_days} days`);
    }
    
    if (trend.trend === 'increasing') {
      recommended_actions.push(`Monitor ${resourceType.toUpperCase()} usage trend closely`);
    }
    
    return {
      resource_type: resourceType,
      exhaustion_prediction_days,
      trend_direction: trend.trend,
      confidence_score: trend.correlation,
      recommended_actions,
      warning_threshold_days: this.thresholds.prediction_warning_days,
      critical_threshold_days: this.thresholds.prediction_critical_days
    };
  }

  private checkThresholdAlerts(): void {
    for (const [resourceType, metric] of this.capacityMetrics) {
      // Check for threshold breaches
      if (metric.current_utilization_percent > this.thresholds.critical) {
        this.createAlert(
          resourceType,
          'threshold_breach',
          'critical',
          `${resourceType.toUpperCase()} utilization critical: ${metric.current_utilization_percent.toFixed(1)}%`,
          metric.current_utilization_percent,
          this.thresholds.critical
        );
      } else if (metric.current_utilization_percent > this.thresholds.warning) {
        this.createAlert(
          resourceType,
          'threshold_breach',
          'high',
          `${resourceType.toUpperCase()} utilization high: ${metric.current_utilization_percent.toFixed(1)}%`,
          metric.current_utilization_percent,
          this.thresholds.warning
        );
      }
    }
  }

  private updateCapacityPredictions(): void {
    const predictions = this.getCapacityPredictions();
    
    for (const prediction of predictions) {
      if (prediction.exhaustion_prediction_days) {
        if (prediction.exhaustion_prediction_days <= prediction.critical_threshold_days) {
          this.createAlert(
            prediction.resource_type,
            'capacity_warning',
            'critical',
            `${prediction.resource_type.toUpperCase()} capacity will be exhausted in ${prediction.exhaustion_prediction_days} days`,
            prediction.exhaustion_prediction_days,
            prediction.critical_threshold_days
          );
        } else if (prediction.exhaustion_prediction_days <= prediction.warning_threshold_days) {
          this.createAlert(
            prediction.resource_type,
            'capacity_warning',
            'medium',
            `${prediction.resource_type.toUpperCase()} capacity will be exhausted in ${prediction.exhaustion_prediction_days} days`,
            prediction.exhaustion_prediction_days,
            prediction.warning_threshold_days
          );
        }
      }
    }
  }

  private createAlert(
    resourceType: string,
    alertType: 'threshold_breach' | 'capacity_warning' | 'trend_alert',
    severity: 'low' | 'medium' | 'high' | 'critical',
    message: string,
    currentValue: number,
    thresholdValue: number
  ): void {
    // Check if similar alert already exists
    const existingAlert = this.alerts.find(alert => 
      !alert.resolved &&
      alert.resource_type === resourceType &&
      alert.alert_type === alertType &&
      alert.severity === severity
    );
    
    if (existingAlert) return; // Don't create duplicate alerts
    
    const alert: UtilizationAlert = {
      id: `${resourceType}_${alertType}_${Date.now()}`,
      resource_type: resourceType,
      alert_type: alertType,
      severity,
      message,
      current_value: currentValue,
      threshold_value: thresholdValue,
      triggered_at: new Date(),
      resolved: false
    };
    
    this.alerts.push(alert);
    
    // Notify callbacks
    this.callbacks.forEach(callback => callback(alert));
  }
}

// Global instance for capacity utilization tracking
export const capacityUtilizationTracker = new CapacityUtilizationTracker();