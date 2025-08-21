/**
 * Latency Alert Engine
 * Monitors latency metrics and generates alerts based on configurable thresholds
 */

import { PerformanceAlert, AlertSeverity } from '../../types/monitoring';
import { LatencyPercentileCalculator, PercentileResult } from './LatencyPercentileCalculator';

export interface LatencyThreshold {
  id: string;
  name: string;
  metric_type: 'p50' | 'p90' | 'p95' | 'p99' | 'p99_9' | 'mean' | 'max';
  venue?: string;
  threshold_ms: number;
  severity: AlertSeverity;
  enabled: boolean;
  consecutive_breaches_required: number;
  cooldown_minutes: number;
  auto_resolve: boolean;
  auto_resolve_threshold_ms?: number;
}

export interface LatencyAlertContext {
  venue: string;
  metric_type: string;
  current_value_ms: number;
  threshold_ms: number;
  consecutive_breaches: number;
  duration_minutes: number;
  samples_count: number;
  recent_values: number[];
}

export interface AlertRule {
  threshold: LatencyThreshold;
  last_triggered?: Date;
  consecutive_breaches: number;
  in_cooldown: boolean;
  cooldown_until?: Date;
  active_alert_id?: string;
}

export class LatencyAlertEngine {
  private alertRules: Map<string, AlertRule> = new Map();
  private activeAlerts: Map<string, PerformanceAlert> = new Map();
  private alertHistory: PerformanceAlert[] = [];
  private callbacks: {
    onAlert: ((alert: PerformanceAlert) => void)[];
    onResolve: ((alertId: string, resolutionMethod: 'auto' | 'manual' | 'timeout') => void)[];
  } = { onAlert: [], onResolve: [] };

  private nextAlertId: number = 1;

  /**
   * Add or update a latency threshold rule
   */
  addThreshold(threshold: LatencyThreshold): void {
    const rule: AlertRule = {
      threshold,
      consecutive_breaches: 0,
      in_cooldown: false
    };

    this.alertRules.set(threshold.id, rule);
  }

  /**
   * Remove a threshold rule
   */
  removeThreshold(thresholdId: string): void {
    const rule = this.alertRules.get(thresholdId);
    if (rule?.active_alert_id) {
      this.resolveAlert(rule.active_alert_id, 'manual');
    }
    this.alertRules.delete(thresholdId);
  }

  /**
   * Enable or disable a threshold rule
   */
  setThresholdEnabled(thresholdId: string, enabled: boolean): void {
    const rule = this.alertRules.get(thresholdId);
    if (rule) {
      rule.threshold.enabled = enabled;
      if (!enabled && rule.active_alert_id) {
        this.resolveAlert(rule.active_alert_id, 'manual');
      }
    }
  }

  /**
   * Check latency metrics against all configured thresholds
   */
  checkLatencyMetrics(venue: string, percentiles: PercentileResult, recentValues: number[]): PerformanceAlert[] {
    const triggeredAlerts: PerformanceAlert[] = [];

    for (const rule of this.alertRules.values()) {
      if (!rule.threshold.enabled) continue;
      
      // Skip if venue filter doesn't match
      if (rule.threshold.venue && rule.threshold.venue !== venue && rule.threshold.venue !== 'all') {
        continue;
      }

      // Skip if in cooldown
      if (rule.in_cooldown && rule.cooldown_until && new Date() < rule.cooldown_until) {
        continue;
      }

      const alert = this.evaluateThreshold(rule, venue, percentiles, recentValues);
      if (alert) {
        triggeredAlerts.push(alert);
      }
    }

    return triggeredAlerts;
  }

  /**
   * Manually resolve an active alert
   */
  resolveAlert(alertId: string, resolutionMethod: 'manual' | 'auto' | 'timeout' = 'manual'): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (!alert) return false;

    // Update alert record
    this.alertHistory.push({
      ...alert,
      resolution_method: resolutionMethod,
      resolved_at: new Date().toISOString()
    });

    // Remove from active alerts
    this.activeAlerts.delete(alertId);

    // Update associated rule
    const rule = Array.from(this.alertRules.values())
      .find(r => r.active_alert_id === alertId);
    
    if (rule) {
      rule.active_alert_id = undefined;
      rule.consecutive_breaches = 0;
      rule.in_cooldown = true;
      rule.cooldown_until = new Date(Date.now() + (rule.threshold.cooldown_minutes * 60 * 1000));
    }

    // Notify callbacks
    this.callbacks.onResolve.forEach(callback => callback(alertId, resolutionMethod));

    return true;
  }

  /**
   * Get all active alerts
   */
  getActiveAlerts(venue?: string, severity?: AlertSeverity): PerformanceAlert[] {
    let alerts = Array.from(this.activeAlerts.values());
    
    if (venue) {
      alerts = alerts.filter(alert => alert.venue_name === venue);
    }
    
    if (severity) {
      alerts = alerts.filter(alert => alert.severity === severity);
    }
    
    return alerts;
  }

  /**
   * Get alert history
   */
  getAlertHistory(count: number = 100, venue?: string): PerformanceAlert[] {
    let history = [...this.alertHistory];
    
    if (venue) {
      history = history.filter(alert => alert.venue_name === venue);
    }
    
    return history.slice(-count);
  }

  /**
   * Get alert statistics
   */
  getAlertStatistics(timeRangeMs: number = 24 * 60 * 60 * 1000): {
    total_alerts: number;
    active_alerts: number;
    by_severity: Map<AlertSeverity, number>;
    by_venue: Map<string, number>;
    avg_resolution_time_minutes: number;
    most_frequent_metric: string;
  } {
    const now = Date.now();
    const cutoffTime = now - timeRangeMs;

    const recentAlerts = this.alertHistory.filter(alert => 
      new Date(alert.triggered_at).getTime() > cutoffTime
    );

    const by_severity = new Map<AlertSeverity, number>();
    const by_venue = new Map<string, number>();
    const metric_counts = new Map<string, number>();

    let total_resolution_time = 0;
    let resolved_count = 0;

    for (const alert of recentAlerts) {
      // Count by severity
      by_severity.set(alert.severity, (by_severity.get(alert.severity) || 0) + 1);
      
      // Count by venue
      const venue = alert.venue_name || 'unknown';
      by_venue.set(venue, (by_venue.get(venue) || 0) + 1);
      
      // Count by metric
      metric_counts.set(alert.metric_name, (metric_counts.get(alert.metric_name) || 0) + 1);
      
      // Calculate resolution time
      if (alert.resolved_at) {
        const triggerTime = new Date(alert.triggered_at).getTime();
        const resolveTime = new Date(alert.resolved_at).getTime();
        total_resolution_time += (resolveTime - triggerTime);
        resolved_count++;
      }
    }

    const avg_resolution_time_minutes = resolved_count > 0 ? 
      (total_resolution_time / resolved_count) / (60 * 1000) : 0;

    // Find most frequent metric
    let most_frequent_metric = 'none';
    let max_count = 0;
    for (const [metric, count] of metric_counts) {
      if (count > max_count) {
        max_count = count;
        most_frequent_metric = metric;
      }
    }

    return {
      total_alerts: recentAlerts.length,
      active_alerts: this.activeAlerts.size,
      by_severity,
      by_venue,
      avg_resolution_time_minutes: Math.round(avg_resolution_time_minutes * 100) / 100,
      most_frequent_metric
    };
  }

  /**
   * Add alert callback
   */
  onAlert(callback: (alert: PerformanceAlert) => void): void {
    this.callbacks.onAlert.push(callback);
  }

  /**
   * Add resolve callback
   */
  onAlertResolve(callback: (alertId: string, resolutionMethod: 'auto' | 'manual' | 'timeout') => void): void {
    this.callbacks.onResolve.push(callback);
  }

  /**
   * Remove callback
   */
  removeCallback(callback: Function): void {
    let index = this.callbacks.onAlert.indexOf(callback as any);
    if (index > -1) {
      this.callbacks.onAlert.splice(index, 1);
    }

    index = this.callbacks.onResolve.indexOf(callback as any);
    if (index > -1) {
      this.callbacks.onResolve.splice(index, 1);
    }
  }

  /**
   * Get all threshold configurations
   */
  getThresholds(): LatencyThreshold[] {
    return Array.from(this.alertRules.values()).map(rule => rule.threshold);
  }

  /**
   * Test if a value would trigger an alert (without actually triggering)
   */
  testThreshold(thresholdId: string, value: number): boolean {
    const rule = this.alertRules.get(thresholdId);
    if (!rule || !rule.threshold.enabled) return false;

    return value > rule.threshold.threshold_ms;
  }

  /**
   * Bulk update latency data for multiple venues
   */
  checkAllVenues(venueData: Map<string, { percentiles: PercentileResult; recentValues: number[] }>): PerformanceAlert[] {
    const allTriggeredAlerts: PerformanceAlert[] = [];

    for (const [venue, data] of venueData) {
      const alerts = this.checkLatencyMetrics(venue, data.percentiles, data.recentValues);
      allTriggeredAlerts.push(...alerts);
    }

    return allTriggeredAlerts;
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.alertRules.clear();
    this.activeAlerts.clear();
    this.alertHistory = [];
    this.nextAlertId = 1;
  }

  /**
   * Auto-resolve alerts that have crossed back below threshold
   */
  checkAutoResolve(venue: string, percentiles: PercentileResult): string[] {
    const resolvedAlerts: string[] = [];

    for (const alert of this.activeAlerts.values()) {
      if (alert.venue_name !== venue) continue;

      const rule = Array.from(this.alertRules.values())
        .find(r => r.active_alert_id === alert.alert_id);
      
      if (!rule?.threshold.auto_resolve || !rule.threshold.auto_resolve_threshold_ms) {
        continue;
      }

      const currentValue = this.extractMetricValue(percentiles, rule.threshold.metric_type);
      
      if (currentValue <= rule.threshold.auto_resolve_threshold_ms) {
        if (this.resolveAlert(alert.alert_id, 'auto')) {
          resolvedAlerts.push(alert.alert_id);
        }
      }
    }

    return resolvedAlerts;
  }

  private evaluateThreshold(
    rule: AlertRule, 
    venue: string, 
    percentiles: PercentileResult, 
    recentValues: number[]
  ): PerformanceAlert | null {
    const threshold = rule.threshold;
    const currentValue = this.extractMetricValue(percentiles, threshold.metric_type);
    
    const isBreached = currentValue > threshold.threshold_ms;
    
    if (isBreached) {
      rule.consecutive_breaches++;
      rule.in_cooldown = false;
    } else {
      rule.consecutive_breaches = 0;
      return null;
    }

    // Check if we have enough consecutive breaches
    if (rule.consecutive_breaches < threshold.consecutive_breaches_required) {
      return null;
    }

    // Don't create duplicate alerts
    if (rule.active_alert_id) {
      return null;
    }

    // Create new alert
    const alert: PerformanceAlert = {
      alert_id: `lat_${this.nextAlertId++}`,
      metric_name: `${threshold.metric_type}_latency`,
      current_value: Math.round(currentValue * 100) / 100,
      threshold_value: threshold.threshold_ms,
      severity: threshold.severity,
      triggered_at: new Date().toISOString(),
      venue_name: venue,
      description: this.generateAlertDescription(threshold, venue, currentValue),
      auto_resolution_available: threshold.auto_resolve,
      escalation_level: this.mapSeverityToEscalation(threshold.severity),
      notification_sent: false
    };

    // Store alert
    this.activeAlerts.set(alert.alert_id, alert);
    rule.active_alert_id = alert.alert_id;
    rule.last_triggered = new Date();

    // Notify callbacks
    this.callbacks.onAlert.forEach(callback => callback(alert));

    return alert;
  }

  private extractMetricValue(percentiles: PercentileResult, metricType: string): number {
    switch (metricType) {
      case 'p50': return percentiles.p50;
      case 'p90': return percentiles.p90;
      case 'p95': return percentiles.p95;
      case 'p99': return percentiles.p99;
      case 'p99_9': return percentiles.p99_9;
      case 'mean': return percentiles.mean;
      case 'max': return percentiles.max;
      default: return percentiles.p95;
    }
  }

  private generateAlertDescription(threshold: LatencyThreshold, venue: string, currentValue: number): string {
    const metricName = threshold.metric_type.toUpperCase();
    const exceedBy = Math.round((currentValue - threshold.threshold_ms) * 100) / 100;
    const percentage = Math.round(((currentValue - threshold.threshold_ms) / threshold.threshold_ms) * 100);
    
    return `${venue} ${metricName} latency (${currentValue}ms) exceeded threshold (${threshold.threshold_ms}ms) by ${exceedBy}ms (${percentage}%)`;
  }

  private mapSeverityToEscalation(severity: AlertSeverity): number {
    switch (severity) {
      case 'low': return 1;
      case 'medium': return 2;
      case 'high': return 3;
      case 'critical': return 4;
      default: return 2;
    }
  }

  /**
   * Create standard latency thresholds for a venue
   */
  static createStandardThresholds(venue: string): LatencyThreshold[] {
    return [
      {
        id: `${venue}_p95_warning`,
        name: `${venue} P95 Latency Warning`,
        metric_type: 'p95',
        venue,
        threshold_ms: 100,
        severity: 'medium',
        enabled: true,
        consecutive_breaches_required: 3,
        cooldown_minutes: 5,
        auto_resolve: true,
        auto_resolve_threshold_ms: 80
      },
      {
        id: `${venue}_p95_critical`,
        name: `${venue} P95 Latency Critical`,
        metric_type: 'p95',
        venue,
        threshold_ms: 200,
        severity: 'critical',
        enabled: true,
        consecutive_breaches_required: 2,
        cooldown_minutes: 2,
        auto_resolve: true,
        auto_resolve_threshold_ms: 150
      },
      {
        id: `${venue}_p99_warning`,
        name: `${venue} P99 Latency Warning`,
        metric_type: 'p99',
        venue,
        threshold_ms: 500,
        severity: 'high',
        enabled: true,
        consecutive_breaches_required: 2,
        cooldown_minutes: 10,
        auto_resolve: true,
        auto_resolve_threshold_ms: 400
      }
    ];
  }
}

// Global instance for latency alerting
export const latencyAlertEngine = new LatencyAlertEngine();