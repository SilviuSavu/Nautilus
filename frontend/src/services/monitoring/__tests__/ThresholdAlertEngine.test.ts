/**
 * Unit Tests for ThresholdAlertEngine
 * Tests threshold-based alerting system with configurable rules and auto-resolution
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ThresholdAlertEngine, ThresholdRule, AlertState } from '../ThresholdAlertEngine';
import { PerformanceAlert, AlertSeverity } from '../../types/monitoring';

describe('ThresholdAlertEngine', () => {
  let alertEngine: ThresholdAlertEngine;

  beforeEach(() => {
    alertEngine = new ThresholdAlertEngine();
    vi.clearAllMocks();
  });

  afterEach(() => {
    alertEngine.clear();
  });

  describe('Threshold Rule Management', () => {
    it('should create a threshold rule', () => {
      const rule = alertEngine.createThresholdRule({
        name: 'CPU Usage High',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        consecutive_breaches_required: 2,
        cooldown_minutes: 5
      });

      expect(rule.rule_id).toBeTruthy();
      expect(rule.name).toBe('CPU Usage High');
      expect(rule.metric_name).toBe('cpu_usage');
      expect(rule.threshold_value).toBe(80);
      expect(rule.condition).toBe('greater_than');
      expect(rule.severity).toBe('high');
      expect(rule.consecutive_breaches_required).toBe(2);
      expect(rule.cooldown_minutes).toBe(5);
      expect(rule.enabled).toBe(true);
    });

    it('should update an existing rule', () => {
      const rule = alertEngine.createThresholdRule({
        name: 'Test Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });

      const updatedRule = alertEngine.updateThresholdRule(rule.rule_id, {
        threshold_value: 70,
        severity: 'high'
      });

      expect(updatedRule?.threshold_value).toBe(70);
      expect(updatedRule?.severity).toBe('high');
      expect(updatedRule?.name).toBe('Test Rule'); // Unchanged
    });

    it('should delete a threshold rule', () => {
      const rule = alertEngine.createThresholdRule({
        name: 'Test Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });

      const deleted = alertEngine.deleteThresholdRule(rule.rule_id);
      expect(deleted).toBe(true);

      const rules = alertEngine.getThresholdRules();
      expect(rules).toHaveLength(0);
    });

    it('should enable/disable rules', () => {
      const rule = alertEngine.createThresholdRule({
        name: 'Test Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });

      const disabled = alertEngine.setRuleEnabled(rule.rule_id, false);
      expect(disabled).toBe(true);

      const rules = alertEngine.getThresholdRules();
      expect(rules[0].enabled).toBe(false);
    });
  });

  describe('Threshold Evaluation', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'CPU High',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        consecutive_breaches_required: 1
      });
    });

    it('should trigger alert when threshold is breached', () => {
      const alerts = alertEngine.checkMetricValue('cpu_usage', 85);
      
      expect(alerts).toHaveLength(1);
      expect(alerts[0].metric_name).toBe('cpu_usage');
      expect(alerts[0].current_value).toBe(85);
      expect(alerts[0].threshold_value).toBe(80);
      expect(alerts[0].severity).toBe('high');
    });

    it('should not trigger alert when threshold is not breached', () => {
      const alerts = alertEngine.checkMetricValue('cpu_usage', 75);
      expect(alerts).toHaveLength(0);
    });

    it('should respect consecutive breaches requirement', () => {
      alertEngine.clear();
      alertEngine.createThresholdRule({
        name: 'CPU High',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        consecutive_breaches_required: 3
      });

      // First breach
      let alerts = alertEngine.checkMetricValue('cpu_usage', 85);
      expect(alerts).toHaveLength(0);

      // Second breach
      alerts = alertEngine.checkMetricValue('cpu_usage', 90);
      expect(alerts).toHaveLength(0);

      // Third breach - should trigger
      alerts = alertEngine.checkMetricValue('cpu_usage', 95);
      expect(alerts).toHaveLength(1);
    });

    it('should reset breach count when threshold is no longer breached', () => {
      alertEngine.clear();
      alertEngine.createThresholdRule({
        name: 'CPU High',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        consecutive_breaches_required: 2
      });

      // First breach
      alertEngine.checkMetricValue('cpu_usage', 85);
      
      // Reset with normal value
      alertEngine.checkMetricValue('cpu_usage', 70);
      
      // Should need 2 breaches again
      let alerts = alertEngine.checkMetricValue('cpu_usage', 85);
      expect(alerts).toHaveLength(0);
      
      alerts = alertEngine.checkMetricValue('cpu_usage', 90);
      expect(alerts).toHaveLength(1);
    });
  });

  describe('Alert Conditions', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'Greater Than Test',
        metric_name: 'test_gt',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });

      alertEngine.createThresholdRule({
        name: 'Less Than Test',
        metric_name: 'test_lt',
        threshold_value: 20,
        condition: 'less_than',
        severity: 'medium'
      });

      alertEngine.createThresholdRule({
        name: 'Equals Test',
        metric_name: 'test_eq',
        threshold_value: 100,
        condition: 'equals',
        severity: 'low'
      });
    });

    it('should evaluate greater_than condition correctly', () => {
      const alertsTrue = alertEngine.checkMetricValue('test_gt', 60);
      const alertsFalse = alertEngine.checkMetricValue('test_gt', 40);
      
      expect(alertsTrue).toHaveLength(1);
      expect(alertsFalse).toHaveLength(0);
    });

    it('should evaluate less_than condition correctly', () => {
      const alertsTrue = alertEngine.checkMetricValue('test_lt', 10);
      const alertsFalse = alertEngine.checkMetricValue('test_lt', 30);
      
      expect(alertsTrue).toHaveLength(1);
      expect(alertsFalse).toHaveLength(0);
    });

    it('should evaluate equals condition correctly', () => {
      const alertsTrue = alertEngine.checkMetricValue('test_eq', 100);
      const alertsFalse = alertEngine.checkMetricValue('test_eq', 99.5);
      
      expect(alertsTrue).toHaveLength(1);
      expect(alertsFalse).toHaveLength(0);
    });
  });

  describe('Venue Filtering', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'IB Only Rule',
        metric_name: 'latency',
        threshold_value: 100,
        condition: 'greater_than',
        severity: 'high',
        venue_filter: ['IB']
      });

      alertEngine.createThresholdRule({
        name: 'All Venues Rule',
        metric_name: 'latency',
        threshold_value: 200,
        condition: 'greater_than',
        severity: 'critical',
        venue_filter: ['all']
      });
    });

    it('should apply venue filtering correctly', () => {
      const ibAlerts = alertEngine.checkMetricValue('latency', 150, 'IB');
      const alpacaAlerts = alertEngine.checkMetricValue('latency', 150, 'Alpaca');
      
      expect(ibAlerts).toHaveLength(1); // IB rule triggered
      expect(alpacaAlerts).toHaveLength(0); // No rules for Alpaca at this threshold
    });

    it('should trigger all venues rule for any venue', () => {
      const ibAlerts = alertEngine.checkMetricValue('latency', 250, 'IB');
      const alpacaAlerts = alertEngine.checkMetricValue('latency', 250, 'Alpaca');
      
      expect(ibAlerts).toHaveLength(2); // Both rules triggered for IB
      expect(alpacaAlerts).toHaveLength(1); // All venues rule triggered for Alpaca
    });
  });

  describe('Auto-Resolution', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'Auto Resolve Test',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high',
        auto_resolve: true,
        auto_resolve_threshold: 70
      });
    });

    it('should auto-resolve alert when value drops below resolve threshold', () => {
      // Trigger alert
      const alerts = alertEngine.checkMetricValue('cpu_usage', 85);
      expect(alerts).toHaveLength(1);
      
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts).toHaveLength(1);
      
      // Auto-resolve with lower value
      alertEngine.checkMetricValue('cpu_usage', 65);
      
      const activeAlertsAfter = alertEngine.getActiveAlerts();
      expect(activeAlertsAfter).toHaveLength(0);
    });

    it('should not auto-resolve if value is still above resolve threshold', () => {
      // Trigger alert
      alertEngine.checkMetricValue('cpu_usage', 85);
      
      // Value drops but still above resolve threshold
      alertEngine.checkMetricValue('cpu_usage', 75);
      
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts).toHaveLength(1);
    });
  });

  describe('Cooldown Management', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'Cooldown Test',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium',
        cooldown_minutes: 5
      });
    });

    it('should respect cooldown period after alert resolution', () => {
      // Trigger alert
      const alerts = alertEngine.checkMetricValue('test_metric', 60);
      expect(alerts).toHaveLength(1);
      
      // Resolve alert manually
      const alertId = alerts[0].alert_id;
      alertEngine.resolveAlert(alertId, 'manual');
      
      // Try to trigger again immediately - should be in cooldown
      const newAlerts = alertEngine.checkMetricValue('test_metric', 70);
      expect(newAlerts).toHaveLength(0);
    });
  });

  describe('Alert Management', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'Test Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'high'
      });
    });

    it('should track active alerts', () => {
      alertEngine.checkMetricValue('test_metric', 60);
      
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts).toHaveLength(1);
      expect(activeAlerts[0].metric_name).toBe('test_metric');
    });

    it('should resolve alerts manually', () => {
      const alerts = alertEngine.checkMetricValue('test_metric', 60);
      const alertId = alerts[0].alert_id;
      
      const resolved = alertEngine.resolveAlert(alertId, 'manual');
      expect(resolved).toBe(true);
      
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts).toHaveLength(0);
    });

    it('should maintain alert history', () => {
      const alerts = alertEngine.checkMetricValue('test_metric', 60);
      const alertId = alerts[0].alert_id;
      
      alertEngine.resolveAlert(alertId, 'manual');
      
      const history = alertEngine.getAlertHistory();
      expect(history).toHaveLength(1);
      expect(history[0].alert_id).toBe(alertId);
      expect(history[0].resolution_method).toBe('manual');
    });
  });

  describe('Alert Statistics', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'High CPU',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high'
      });

      alertEngine.createThresholdRule({
        name: 'Low Memory',
        metric_name: 'memory_available',
        threshold_value: 10,
        condition: 'less_than',
        severity: 'critical'
      });
    });

    it('should provide comprehensive alert statistics', () => {
      // Trigger some alerts
      alertEngine.checkMetricValue('cpu_usage', 90, 'IB');
      alertEngine.checkMetricValue('memory_available', 5, 'Alpaca');
      
      const stats = alertEngine.getAlertStatistics();
      
      expect(stats.active_alerts).toBe(2);
      expect(stats.by_severity.get('high')).toBe(1);
      expect(stats.by_severity.get('critical')).toBe(1);
      expect(stats.by_venue.get('IB')).toBe(1);
      expect(stats.by_venue.get('Alpaca')).toBe(1);
    });
  });

  describe('Callbacks and Events', () => {
    it('should trigger callbacks on alert creation', () => {
      const mockCallback = vi.fn();
      alertEngine.onAlert(mockCallback);
      
      alertEngine.createThresholdRule({
        name: 'Test',
        metric_name: 'test',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });
      
      const alerts = alertEngine.checkMetricValue('test', 60);
      
      expect(mockCallback).toHaveBeenCalledWith(alerts[0]);
    });

    it('should trigger callbacks on alert resolution', () => {
      const mockCallback = vi.fn();
      alertEngine.onAlertResolve(mockCallback);
      
      alertEngine.createThresholdRule({
        name: 'Test',
        metric_name: 'test',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });
      
      const alerts = alertEngine.checkMetricValue('test', 60);
      const alertId = alerts[0].alert_id;
      
      alertEngine.resolveAlert(alertId, 'manual');
      
      expect(mockCallback).toHaveBeenCalledWith(alertId, 'manual');
    });
  });

  describe('Test Value Function', () => {
    beforeEach(() => {
      alertEngine.createThresholdRule({
        name: 'Test Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });
    });

    it('should test if value would trigger rules without triggering', () => {
      const testResult = alertEngine.testValue('test_metric', 60);
      
      expect(testResult.would_trigger).toBe(true);
      expect(testResult.matching_rules).toHaveLength(1);
      
      // Should not have triggered actual alert
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts).toHaveLength(0);
    });

    it('should return false for values that would not trigger', () => {
      const testResult = alertEngine.testValue('test_metric', 40);
      
      expect(testResult.would_trigger).toBe(false);
      expect(testResult.matching_rules).toHaveLength(0);
    });
  });

  describe('Rule Import', () => {
    it('should import rules from configuration', () => {
      const configurations = [
        {
          metric_name: 'cpu_usage',
          threshold_value: 80,
          condition: 'greater_than' as const,
          severity: 'high' as AlertSeverity,
          enabled: true,
          notification_channels: { email: ['test@example.com'] },
          escalation_rules: { escalate_after_minutes: 5, escalation_contacts: [] },
          auto_resolution: { enabled: true }
        }
      ];
      
      const result = alertEngine.importRules(configurations);
      
      expect(result.imported).toBe(1);
      expect(result.skipped).toBe(0);
      expect(result.errors).toHaveLength(0);
      
      const rules = alertEngine.getThresholdRules();
      expect(rules).toHaveLength(1);
      expect(rules[0].metric_name).toBe('cpu_usage');
    });

    it('should handle invalid configurations', () => {
      const configurations = [
        {
          // Missing required fields
          metric_name: '',
          threshold_value: undefined as any,
          condition: 'greater_than' as const,
          severity: 'high' as AlertSeverity,
          enabled: true,
          notification_channels: { email: [] },
          escalation_rules: { escalate_after_minutes: 5, escalation_contacts: [] },
          auto_resolution: { enabled: false }
        }
      ];
      
      const result = alertEngine.importRules(configurations);
      
      expect(result.imported).toBe(0);
      expect(result.skipped).toBe(1);
      expect(result.errors).toHaveLength(1);
    });
  });

  describe('Standard Rules', () => {
    it('should provide standard rule configurations', () => {
      const standardRules = ThresholdAlertEngine.createStandardRules();
      
      expect(standardRules).toHaveLength(4);
      expect(standardRules.map(r => r.metric_name)).toContain('cpu_usage');
      expect(standardRules.map(r => r.metric_name)).toContain('memory_usage');
      expect(standardRules.map(r => r.metric_name)).toContain('response_time_ms');
      expect(standardRules.map(r => r.metric_name)).toContain('error_rate_percent');
    });
  });
});