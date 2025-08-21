/**
 * Threshold Alert Engine
 * Configurable threshold-based alerting system for performance metrics
 */

import { PerformanceAlert, AlertSeverity, AlertConfigurationRequest } from '../../types/monitoring';

export interface ThresholdRule {
  rule_id: string;
  name: string;
  metric_name: string;
  threshold_value: number;
  condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
  severity: AlertSeverity;
  enabled: boolean;
  venue_filter?: string[];
  consecutive_breaches_required: number;
  cooldown_minutes: number;
  auto_resolve: boolean;
  auto_resolve_threshold?: number;
  created_at: Date;
  last_triggered?: Date;
}

export interface AlertState {
  rule_id: string;
  current_breaches: number;
  in_cooldown: boolean;
  cooldown_until?: Date;
  active_alert_id?: string;
  last_check: Date;
}

export class ThresholdAlertEngine {
  private thresholdRules: Map<string, ThresholdRule> = new Map();
  private alertStates: Map<string, AlertState> = new Map();
  private activeAlerts: Map<string, PerformanceAlert> = new Map();
  private alertHistory: PerformanceAlert[] = [];
  private maxHistorySize: number = 10000;

  private callbacks: {
    onAlert: ((alert: PerformanceAlert) => void)[];
    onResolve: ((alertId: string, resolutionMethod: 'auto' | 'manual' | 'timeout') => void)[];
    onRuleChange: ((rule: ThresholdRule, action: 'created' | 'updated' | 'deleted') => void)[];
  } = { onAlert: [], onResolve: [], onRuleChange: [] };

  private nextRuleId: number = 1;
  private nextAlertId: number = 1;

  /**
   * Create a new threshold rule
   */
  createThresholdRule(config: {
    name: string;
    metric_name: string;
    threshold_value: number;
    condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
    severity: AlertSeverity;
    venue_filter?: string[];
    consecutive_breaches_required?: number;
    cooldown_minutes?: number;
    auto_resolve?: boolean;
    auto_resolve_threshold?: number;
  }): ThresholdRule {
    const rule: ThresholdRule = {
      rule_id: `rule_${this.nextRuleId++}`,
      name: config.name,
      metric_name: config.metric_name,
      threshold_value: config.threshold_value,
      condition: config.condition,
      severity: config.severity,
      enabled: true,
      venue_filter: config.venue_filter,
      consecutive_breaches_required: config.consecutive_breaches_required || 1,
      cooldown_minutes: config.cooldown_minutes || 5,
      auto_resolve: config.auto_resolve || false,
      auto_resolve_threshold: config.auto_resolve_threshold,
      created_at: new Date()
    };

    this.thresholdRules.set(rule.rule_id, rule);
    this.alertStates.set(rule.rule_id, {
      rule_id: rule.rule_id,
      current_breaches: 0,
      in_cooldown: false,
      last_check: new Date()
    });

    this.callbacks.onRuleChange.forEach(callback => callback(rule, 'created'));

    return rule;
  }

  /**
   * Update an existing threshold rule
   */
  updateThresholdRule(ruleId: string, updates: Partial<Omit<ThresholdRule, 'rule_id' | 'created_at'>>): ThresholdRule | null {
    const rule = this.thresholdRules.get(ruleId);
    if (!rule) return null;

    const updatedRule = { ...rule, ...updates };
    this.thresholdRules.set(ruleId, updatedRule);

    this.callbacks.onRuleChange.forEach(callback => callback(updatedRule, 'updated'));

    return updatedRule;
  }

  /**
   * Delete a threshold rule
   */
  deleteThresholdRule(ruleId: string): boolean {
    const rule = this.thresholdRules.get(ruleId);
    if (!rule) return false;

    // Resolve any active alert for this rule
    const alertState = this.alertStates.get(ruleId);
    if (alertState?.active_alert_id) {
      this.resolveAlert(alertState.active_alert_id, 'manual');
    }

    this.thresholdRules.delete(ruleId);
    this.alertStates.delete(ruleId);

    this.callbacks.onRuleChange.forEach(callback => callback(rule, 'deleted'));

    return true;
  }

  /**
   * Enable or disable a threshold rule
   */
  setRuleEnabled(ruleId: string, enabled: boolean): boolean {
    const rule = this.thresholdRules.get(ruleId);
    if (!rule) return false;

    rule.enabled = enabled;

    // If disabling, resolve any active alert
    if (!enabled) {
      const alertState = this.alertStates.get(ruleId);
      if (alertState?.active_alert_id) {
        this.resolveAlert(alertState.active_alert_id, 'manual');
      }
    }

    return true;
  }

  /**
   * Check metric value against all applicable threshold rules
   */
  checkMetricValue(
    metricName: string,
    value: number,
    venue?: string,
    metadata?: Record<string, any>
  ): PerformanceAlert[] {
    const triggeredAlerts: PerformanceAlert[] = [];
    const now = new Date();

    for (const rule of this.thresholdRules.values()) {
      if (!rule.enabled || rule.metric_name !== metricName) {
        continue;
      }

      // Check venue filter
      if (rule.venue_filter && rule.venue_filter.length > 0 && venue) {
        if (!rule.venue_filter.includes(venue) && !rule.venue_filter.includes('all')) {
          continue;
        }
      }

      const alertState = this.alertStates.get(rule.rule_id)!;

      // Check if in cooldown
      if (alertState.in_cooldown && alertState.cooldown_until && now < alertState.cooldown_until) {
        continue;
      }

      // Check threshold condition
      const isThresholdBreached = this.evaluateCondition(value, rule.threshold_value, rule.condition);

      if (isThresholdBreached) {
        alertState.current_breaches++;
        alertState.in_cooldown = false;

        // Check if we have enough consecutive breaches
        if (alertState.current_breaches >= rule.consecutive_breaches_required && !alertState.active_alert_id) {
          const alert = this.createAlert(rule, value, venue, metadata);
          triggeredAlerts.push(alert);
          alertState.active_alert_id = alert.alert_id;
          rule.last_triggered = now;
        }
      } else {
        // Reset breaches if threshold is not breached
        alertState.current_breaches = 0;

        // Check for auto-resolve
        if (alertState.active_alert_id && rule.auto_resolve && rule.auto_resolve_threshold !== undefined) {
          const shouldResolve = this.evaluateCondition(value, rule.auto_resolve_threshold, 
            rule.condition === 'greater_than' ? 'less_than' : 'greater_than'
          );

          if (shouldResolve) {
            this.resolveAlert(alertState.active_alert_id, 'auto');
            alertState.active_alert_id = undefined;
          }
        }
      }

      alertState.last_check = now;
    }

    return triggeredAlerts;
  }

  /**
   * Manually resolve an alert
   */
  resolveAlert(alertId: string, resolutionMethod: 'manual' | 'auto' | 'timeout' = 'manual'): boolean {
    const alert = this.activeAlerts.get(alertId);
    if (!alert) return false;

    // Update alert with resolution info
    const resolvedAlert: PerformanceAlert = {
      ...alert,
      resolved_at: new Date().toISOString(),
      resolution_method: resolutionMethod
    };

    // Move to history
    this.alertHistory.push(resolvedAlert);
    this.maintainHistoryLimit();

    // Remove from active alerts
    this.activeAlerts.delete(alertId);

    // Update alert state
    const alertState = Array.from(this.alertStates.values())
      .find(state => state.active_alert_id === alertId);

    if (alertState) {
      alertState.active_alert_id = undefined;
      alertState.current_breaches = 0;
      alertState.in_cooldown = true;
      
      const rule = this.thresholdRules.get(alertState.rule_id);
      if (rule) {
        alertState.cooldown_until = new Date(Date.now() + (rule.cooldown_minutes * 60 * 1000));
      }
    }

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
  getAlertHistory(limit?: number, venue?: string): PerformanceAlert[] {
    let history = [...this.alertHistory];

    if (venue) {
      history = history.filter(alert => alert.venue_name === venue);
    }

    if (limit) {
      history = history.slice(-limit);
    }

    return history.reverse(); // Most recent first
  }

  /**
   * Get all threshold rules
   */
  getThresholdRules(metricName?: string, enabled?: boolean): ThresholdRule[] {
    let rules = Array.from(this.thresholdRules.values());

    if (metricName) {
      rules = rules.filter(rule => rule.metric_name === metricName);
    }

    if (enabled !== undefined) {
      rules = rules.filter(rule => rule.enabled === enabled);
    }

    return rules;
  }

  /**
   * Get alert statistics
   */
  getAlertStatistics(timeRangeMs: number = 24 * 60 * 60 * 1000): {
    total_alerts: number;
    active_alerts: number;
    by_severity: Map<AlertSeverity, number>;
    by_metric: Map<string, number>;
    by_venue: Map<string, number>;
    avg_resolution_time_minutes: number;
    most_frequent_metric: string;
    false_positive_rate: number;
  } {
    const now = Date.now();
    const cutoffTime = now - timeRangeMs;

    const recentAlerts = this.alertHistory.filter(alert =>
      new Date(alert.triggered_at).getTime() > cutoffTime
    );

    const bySeverity = new Map<AlertSeverity, number>();
    const byMetric = new Map<string, number>();
    const byVenue = new Map<string, number>();

    let totalResolutionTime = 0;
    let resolvedCount = 0;
    let quickResolutions = 0; // Resolved within 5 minutes (potential false positives)

    for (const alert of recentAlerts) {
      // Count by severity
      bySeverity.set(alert.severity, (bySeverity.get(alert.severity) || 0) + 1);

      // Count by metric
      byMetric.set(alert.metric_name, (byMetric.get(alert.metric_name) || 0) + 1);

      // Count by venue
      if (alert.venue_name) {
        byVenue.set(alert.venue_name, (byVenue.get(alert.venue_name) || 0) + 1);
      }

      // Calculate resolution time
      if (alert.resolved_at) {
        const triggerTime = new Date(alert.triggered_at).getTime();
        const resolveTime = new Date(alert.resolved_at).getTime();
        const resolutionTime = (resolveTime - triggerTime) / (60 * 1000); // minutes

        totalResolutionTime += resolutionTime;
        resolvedCount++;

        if (resolutionTime < 5) {
          quickResolutions++;
        }
      }
    }

    // Find most frequent metric
    let mostFrequentMetric = 'none';
    let maxCount = 0;
    for (const [metric, count] of byMetric) {
      if (count > maxCount) {
        maxCount = count;
        mostFrequentMetric = metric;
      }
    }

    const avgResolutionTime = resolvedCount > 0 ? totalResolutionTime / resolvedCount : 0;
    const falsePositiveRate = resolvedCount > 0 ? (quickResolutions / resolvedCount) * 100 : 0;

    return {
      total_alerts: recentAlerts.length,
      active_alerts: this.activeAlerts.size,
      by_severity: bySeverity,
      by_metric: byMetric,
      by_venue: byVenue,
      avg_resolution_time_minutes: Math.round(avgResolutionTime * 100) / 100,
      most_frequent_metric: mostFrequentMetric,
      false_positive_rate: Math.round(falsePositiveRate * 100) / 100
    };
  }

  /**
   * Test if a value would trigger any rules (without actually triggering)
   */
  testValue(metricName: string, value: number, venue?: string): {
    would_trigger: boolean;
    matching_rules: ThresholdRule[];
  } {
    const matchingRules: ThresholdRule[] = [];

    for (const rule of this.thresholdRules.values()) {
      if (!rule.enabled || rule.metric_name !== metricName) {
        continue;
      }

      // Check venue filter
      if (rule.venue_filter && rule.venue_filter.length > 0 && venue) {
        if (!rule.venue_filter.includes(venue) && !rule.venue_filter.includes('all')) {
          continue;
        }
      }

      if (this.evaluateCondition(value, rule.threshold_value, rule.condition)) {
        matchingRules.push(rule);
      }
    }

    return {
      would_trigger: matchingRules.length > 0,
      matching_rules: matchingRules
    };
  }

  /**
   * Bulk import rules from configuration
   */
  importRules(configurations: AlertConfigurationRequest[]): {
    imported: number;
    skipped: number;
    errors: string[];
  } {
    let imported = 0;
    let skipped = 0;
    const errors: string[] = [];

    for (const config of configurations) {
      try {
        // Validate configuration
        if (!config.metric_name || config.threshold_value === undefined) {
          errors.push(`Invalid configuration: missing metric_name or threshold_value`);
          skipped++;
          continue;
        }

        this.createThresholdRule({
          name: `${config.metric_name} ${config.condition} ${config.threshold_value}`,
          metric_name: config.metric_name,
          threshold_value: config.threshold_value,
          condition: config.condition,
          severity: config.severity,
          venue_filter: config.venue_filter,
          consecutive_breaches_required: 1,
          cooldown_minutes: config.escalation_rules?.escalate_after_minutes || 5,
          auto_resolve: config.auto_resolution?.enabled || false,
          auto_resolve_threshold: config.auto_resolution?.resolution_threshold
        });

        imported++;
      } catch (error) {
        errors.push(`Error importing rule: ${error}`);
        skipped++;
      }
    }

    return { imported, skipped, errors };
  }

  /**
   * Add callback functions
   */
  onAlert(callback: (alert: PerformanceAlert) => void): void {
    this.callbacks.onAlert.push(callback);
  }

  onAlertResolve(callback: (alertId: string, resolutionMethod: 'auto' | 'manual' | 'timeout') => void): void {
    this.callbacks.onResolve.push(callback);
  }

  onRuleChange(callback: (rule: ThresholdRule, action: 'created' | 'updated' | 'deleted') => void): void {
    this.callbacks.onRuleChange.push(callback);
  }

  /**
   * Remove callbacks
   */
  removeCallback(callback: Function): void {
    let index = this.callbacks.onAlert.indexOf(callback as any);
    if (index > -1) this.callbacks.onAlert.splice(index, 1);

    index = this.callbacks.onResolve.indexOf(callback as any);
    if (index > -1) this.callbacks.onResolve.splice(index, 1);

    index = this.callbacks.onRuleChange.indexOf(callback as any);
    if (index > -1) this.callbacks.onRuleChange.splice(index, 1);
  }

  /**
   * Clear all rules, alerts, and history
   */
  clear(): void {
    this.thresholdRules.clear();
    this.alertStates.clear();
    this.activeAlerts.clear();
    this.alertHistory = [];
    this.nextRuleId = 1;
    this.nextAlertId = 1;
  }

  // Private methods

  private evaluateCondition(value: number, threshold: number, condition: string): boolean {
    switch (condition) {
      case 'greater_than': return value > threshold;
      case 'less_than': return value < threshold;
      case 'equals': return Math.abs(value - threshold) < 0.001; // Small epsilon for float comparison
      case 'not_equals': return Math.abs(value - threshold) >= 0.001;
      default: return false;
    }
  }

  private createAlert(rule: ThresholdRule, currentValue: number, venue?: string, metadata?: Record<string, any>): PerformanceAlert {
    const alert: PerformanceAlert = {
      alert_id: `alert_${this.nextAlertId++}`,
      metric_name: rule.metric_name,
      current_value: Math.round(currentValue * 1000) / 1000,
      threshold_value: rule.threshold_value,
      severity: rule.severity,
      triggered_at: new Date().toISOString(),
      venue_name: venue,
      description: this.generateAlertDescription(rule, currentValue, venue),
      auto_resolution_available: rule.auto_resolve,
      escalation_level: this.mapSeverityToEscalation(rule.severity),
      notification_sent: false,
      metadata
    };

    this.activeAlerts.set(alert.alert_id, alert);
    this.callbacks.onAlert.forEach(callback => callback(alert));

    return alert;
  }

  private generateAlertDescription(rule: ThresholdRule, currentValue: number, venue?: string): string {
    const venuePrefix = venue ? `${venue}: ` : '';
    const conditionText = {
      greater_than: 'exceeded',
      less_than: 'fell below',
      equals: 'equals',
      not_equals: 'does not equal'
    }[rule.condition] || 'triggered';

    return `${venuePrefix}${rule.metric_name} ${conditionText} threshold (${currentValue} ${rule.condition.replace('_', ' ')} ${rule.threshold_value})`;
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

  private maintainHistoryLimit(): void {
    if (this.alertHistory.length > this.maxHistorySize) {
      this.alertHistory = this.alertHistory.slice(-this.maxHistorySize);
    }
  }

  /**
   * Create standard threshold rules for common metrics
   */
  static createStandardRules(): AlertConfigurationRequest[] {
    return [
      {
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        enabled: true,
        notification_channels: { email: [] },
        escalation_rules: { escalate_after_minutes: 5, escalation_contacts: [] },
        auto_resolution: { enabled: true, resolution_threshold: 70 }
      },
      {
        metric_name: 'memory_usage',
        threshold_value: 85,
        condition: 'greater_than',
        severity: 'high',
        enabled: true,
        notification_channels: { email: [] },
        escalation_rules: { escalate_after_minutes: 5, escalation_contacts: [] },
        auto_resolution: { enabled: true, resolution_threshold: 75 }
      },
      {
        metric_name: 'response_time_ms',
        threshold_value: 1000,
        condition: 'greater_than',
        severity: 'medium',
        enabled: true,
        notification_channels: { email: [] },
        escalation_rules: { escalate_after_minutes: 10, escalation_contacts: [] },
        auto_resolution: { enabled: true, resolution_threshold: 500 }
      },
      {
        metric_name: 'error_rate_percent',
        threshold_value: 5,
        condition: 'greater_than',
        severity: 'critical',
        enabled: true,
        notification_channels: { email: [] },
        escalation_rules: { escalate_after_minutes: 2, escalation_contacts: [] },
        auto_resolution: { enabled: true, resolution_threshold: 1 }
      }
    ];
  }
}

// Global instance for threshold-based alerting
export const thresholdAlertEngine = new ThresholdAlertEngine();