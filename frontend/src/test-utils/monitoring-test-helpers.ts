/**
 * Monitoring Test Utilities
 * Helper functions and mock data for testing monitoring components and services
 */

import { LatencyMeasurement, OrderExecution } from '../services/monitoring/OrderLatencyTracker';
import { ThresholdRule, AlertState } from '../services/monitoring/ThresholdAlertEngine';
import { PerformanceAlert, AlertSeverity } from '../types/monitoring';

// Mock data generators for testing

export const createMockLatencyMeasurement = (overrides?: Partial<LatencyMeasurement>): LatencyMeasurement => ({
  orderId: `order-${Date.now()}`,
  venue: 'IB',
  latency_ms: 25.5,
  timestamp: new Date(),
  orderType: 'market',
  success: true,
  ...overrides
});

export const createMockOrderExecution = (overrides?: Partial<OrderExecution>): OrderExecution => ({
  orderId: `order-${Date.now()}`,
  venue: 'IB',
  symbol: 'AAPL',
  startTime: performance.now(),
  status: 'pending',
  orderType: 'market',
  side: 'buy',
  ...overrides
});

export const createMockThresholdRule = (overrides?: Partial<ThresholdRule>): ThresholdRule => ({
  rule_id: `rule-${Date.now()}`,
  name: 'Test Rule',
  metric_name: 'cpu_usage',
  threshold_value: 80,
  condition: 'greater_than',
  severity: 'high',
  enabled: true,
  consecutive_breaches_required: 1,
  cooldown_minutes: 5,
  auto_resolve: false,
  created_at: new Date(),
  ...overrides
});

export const createMockAlertState = (overrides?: Partial<AlertState>): AlertState => ({
  rule_id: `rule-${Date.now()}`,
  current_breaches: 0,
  in_cooldown: false,
  last_check: new Date(),
  ...overrides
});

export const createMockPerformanceAlert = (overrides?: Partial<PerformanceAlert>): PerformanceAlert => ({
  alert_id: `alert-${Date.now()}`,
  metric_name: 'cpu_usage',
  current_value: 85,
  threshold_value: 80,
  severity: 'high',
  triggered_at: new Date().toISOString(),
  description: 'CPU usage exceeded threshold',
  auto_resolution_available: false,
  escalation_level: 3,
  notification_sent: false,
  ...overrides
});

// Test data sets for different scenarios

export const mockLatencyDatasets = {
  // Low latency performance
  lowLatency: Array.from({ length: 100 }, (_, i) => createMockLatencyMeasurement({
    orderId: `low-latency-${i}`,
    latency_ms: 5 + Math.random() * 10, // 5-15ms
    venue: 'IB',
    success: true
  })),

  // High latency performance
  highLatency: Array.from({ length: 50 }, (_, i) => createMockLatencyMeasurement({
    orderId: `high-latency-${i}`,
    latency_ms: 100 + Math.random() * 200, // 100-300ms
    venue: 'BINANCE',
    success: true
  })),

  // Mixed performance with failures
  mixedPerformance: Array.from({ length: 200 }, (_, i) => createMockLatencyMeasurement({
    orderId: `mixed-${i}`,
    latency_ms: i % 10 === 0 ? 500 + Math.random() * 1000 : 20 + Math.random() * 50, // Occasional spikes
    venue: i % 2 === 0 ? 'IB' : 'BINANCE',
    success: i % 20 !== 0, // 5% failure rate
    timestamp: new Date(Date.now() - (200 - i) * 60000) // Spread over last 200 minutes
  }))
};

export const mockAlertDatasets = {
  // Critical system alerts
  criticalAlerts: [
    createMockPerformanceAlert({
      alert_id: 'critical-1',
      metric_name: 'memory_usage',
      current_value: 95,
      threshold_value: 90,
      severity: 'critical',
      venue_name: 'IB',
      description: 'Memory usage critically high'
    }),
    createMockPerformanceAlert({
      alert_id: 'critical-2',
      metric_name: 'disk_usage',
      current_value: 98,
      threshold_value: 95,
      severity: 'critical',
      description: 'Disk space critically low'
    })
  ],

  // Performance degradation alerts
  performanceAlerts: [
    createMockPerformanceAlert({
      alert_id: 'perf-1',
      metric_name: 'order_latency',
      current_value: 156.7,
      threshold_value: 100,
      severity: 'high',
      venue_name: 'BINANCE',
      description: 'Order execution latency high'
    }),
    createMockPerformanceAlert({
      alert_id: 'perf-2',
      metric_name: 'throughput',
      current_value: 45,
      threshold_value: 100,
      severity: 'medium',
      venue_name: 'IB',
      description: 'Trading throughput below threshold'
    })
  ],

  // Resolved alerts for historical testing
  resolvedAlerts: [
    createMockPerformanceAlert({
      alert_id: 'resolved-1',
      metric_name: 'cpu_usage',
      current_value: 85,
      threshold_value: 80,
      severity: 'medium',
      resolved_at: new Date(Date.now() - 30 * 60000).toISOString(), // 30 minutes ago
      resolution_method: 'auto',
      description: 'CPU usage spike resolved automatically'
    })
  ]
};

export const mockThresholdRules = {
  // Standard system monitoring rules
  systemRules: [
    createMockThresholdRule({
      rule_id: 'sys-cpu',
      name: 'High CPU Usage',
      metric_name: 'cpu_usage',
      threshold_value: 80,
      condition: 'greater_than',
      severity: 'high',
      consecutive_breaches_required: 2,
      cooldown_minutes: 10
    }),
    createMockThresholdRule({
      rule_id: 'sys-memory',
      name: 'High Memory Usage',
      metric_name: 'memory_usage',
      threshold_value: 85,
      condition: 'greater_than',
      severity: 'critical',
      consecutive_breaches_required: 1,
      cooldown_minutes: 5
    })
  ],

  // Trading performance rules
  tradingRules: [
    createMockThresholdRule({
      rule_id: 'trade-latency',
      name: 'Order Latency Threshold',
      metric_name: 'order_latency',
      threshold_value: 100,
      condition: 'greater_than',
      severity: 'high',
      venue_filter: ['IB', 'BINANCE'],
      auto_resolve: true,
      auto_resolve_threshold: 75
    }),
    createMockThresholdRule({
      rule_id: 'trade-success',
      name: 'Order Success Rate',
      metric_name: 'success_rate',
      threshold_value: 95,
      condition: 'less_than',
      severity: 'medium',
      consecutive_breaches_required: 3
    })
  ]
};

// Test utilities for assertions and expectations

export const expectLatencyMeasurement = (measurement: LatencyMeasurement) => ({
  toHaveValidStructure: () => {
    expect(measurement).toHaveProperty('orderId');
    expect(measurement).toHaveProperty('venue');
    expect(measurement).toHaveProperty('latency_ms');
    expect(measurement).toHaveProperty('timestamp');
    expect(measurement).toHaveProperty('orderType');
    expect(measurement).toHaveProperty('success');
    expect(typeof measurement.latency_ms).toBe('number');
    expect(measurement.latency_ms).toBeGreaterThanOrEqual(0);
  },

  toBeSuccessful: () => {
    expect(measurement.success).toBe(true);
    expect(measurement.latency_ms).toBeGreaterThan(0);
  },

  toHaveLatencyInRange: (min: number, max: number) => {
    expect(measurement.latency_ms).toBeGreaterThanOrEqual(min);
    expect(measurement.latency_ms).toBeLessThanOrEqual(max);
  }
});

export const expectPerformanceAlert = (alert: PerformanceAlert) => ({
  toHaveValidStructure: () => {
    expect(alert).toHaveProperty('alert_id');
    expect(alert).toHaveProperty('metric_name');
    expect(alert).toHaveProperty('current_value');
    expect(alert).toHaveProperty('threshold_value');
    expect(alert).toHaveProperty('severity');
    expect(alert).toHaveProperty('triggered_at');
    expect(alert).toHaveProperty('description');
    expect(['low', 'medium', 'high', 'critical']).toContain(alert.severity);
  },

  toBeResolved: () => {
    expect(alert).toHaveProperty('resolved_at');
    expect(alert.resolved_at).toBeTruthy();
    expect(alert).toHaveProperty('resolution_method');
  },

  toHaveSeverity: (expectedSeverity: AlertSeverity) => {
    expect(alert.severity).toBe(expectedSeverity);
  },

  toMatchMetric: (metricName: string) => {
    expect(alert.metric_name).toBe(metricName);
  }
});

export const expectThresholdRule = (rule: ThresholdRule) => ({
  toHaveValidStructure: () => {
    expect(rule).toHaveProperty('rule_id');
    expect(rule).toHaveProperty('name');
    expect(rule).toHaveProperty('metric_name');
    expect(rule).toHaveProperty('threshold_value');
    expect(rule).toHaveProperty('condition');
    expect(rule).toHaveProperty('severity');
    expect(rule).toHaveProperty('enabled');
    expect(['greater_than', 'less_than', 'equals', 'not_equals']).toContain(rule.condition);
    expect(['low', 'medium', 'high', 'critical']).toContain(rule.severity);
  },

  toBeEnabled: () => {
    expect(rule.enabled).toBe(true);
  },

  toHaveAutoResolution: () => {
    expect(rule.auto_resolve).toBe(true);
    expect(rule.auto_resolve_threshold).toBeDefined();
  }
});

// Mock service implementations for testing

export const createMockOrderLatencyTracker = () => {
  const tracker = {
    pendingOrders: new Map(),
    latencyHistory: [] as LatencyMeasurement[],
    callbackHandlers: [] as Array<(measurement: LatencyMeasurement) => void>,

    startOrder: vi.fn((orderId: string, venue: string, symbol: string, orderType: any, side: any) => {
      const execution = createMockOrderExecution({ orderId, venue, symbol, orderType, side });
      tracker.pendingOrders.set(orderId, execution);
    }),

    completeOrder: vi.fn((orderId: string, success: boolean = true) => {
      const execution = tracker.pendingOrders.get(orderId);
      if (!execution) return null;

      const measurement = createMockLatencyMeasurement({
        orderId,
        venue: execution.venue,
        success,
        latency_ms: Math.random() * 100 + 10 // Random latency 10-110ms
      });

      tracker.latencyHistory.push(measurement);
      tracker.pendingOrders.delete(orderId);
      tracker.callbackHandlers.forEach(callback => callback(measurement));

      return measurement;
    }),

    getVenueLatencyMetrics: vi.fn((venue: string) => ({
      venue_name: venue,
      order_execution_latency: {
        min_ms: 5.2,
        max_ms: 156.7,
        avg_ms: 23.4,
        p50_ms: 18.9,
        p95_ms: 89.3,
        p99_ms: 134.2,
        samples: tracker.latencyHistory.filter(m => m.venue === venue).length
      },
      last_updated: new Date().toISOString()
    })),

    getPendingOrdersCount: vi.fn(() => tracker.pendingOrders.size),

    onLatencyMeasurement: vi.fn((callback: (measurement: LatencyMeasurement) => void) => {
      tracker.callbackHandlers.push(callback);
    }),

    clear: vi.fn(() => {
      tracker.pendingOrders.clear();
      tracker.latencyHistory = [];
      tracker.callbackHandlers = [];
    })
  };

  return tracker;
};

export const createMockThresholdAlertEngine = () => {
  const engine = {
    thresholdRules: new Map<string, ThresholdRule>(),
    alertStates: new Map<string, AlertState>(),
    activeAlerts: new Map<string, PerformanceAlert>(),
    alertHistory: [] as PerformanceAlert[],

    createThresholdRule: vi.fn((config: any) => {
      const rule = createMockThresholdRule(config);
      engine.thresholdRules.set(rule.rule_id, rule);
      engine.alertStates.set(rule.rule_id, createMockAlertState({ rule_id: rule.rule_id }));
      return rule;
    }),

    checkMetricValue: vi.fn((metricName: string, value: number, venue?: string) => {
      const triggeredAlerts: PerformanceAlert[] = [];
      
      for (const rule of engine.thresholdRules.values()) {
        if (rule.metric_name === metricName && rule.enabled) {
          let shouldTrigger = false;
          
          switch (rule.condition) {
            case 'greater_than': shouldTrigger = value > rule.threshold_value; break;
            case 'less_than': shouldTrigger = value < rule.threshold_value; break;
            case 'equals': shouldTrigger = Math.abs(value - rule.threshold_value) < 0.001; break;
            case 'not_equals': shouldTrigger = Math.abs(value - rule.threshold_value) >= 0.001; break;
          }

          if (shouldTrigger) {
            const alert = createMockPerformanceAlert({
              metric_name: metricName,
              current_value: value,
              threshold_value: rule.threshold_value,
              severity: rule.severity,
              venue_name: venue,
              description: `${metricName} threshold breached`
            });
            
            engine.activeAlerts.set(alert.alert_id, alert);
            triggeredAlerts.push(alert);
          }
        }
      }
      
      return triggeredAlerts;
    }),

    getActiveAlerts: vi.fn(() => Array.from(engine.activeAlerts.values())),

    resolveAlert: vi.fn((alertId: string, method: 'manual' | 'auto' | 'timeout' = 'manual') => {
      const alert = engine.activeAlerts.get(alertId);
      if (!alert) return false;

      const resolvedAlert = {
        ...alert,
        resolved_at: new Date().toISOString(),
        resolution_method: method
      };

      engine.alertHistory.push(resolvedAlert);
      engine.activeAlerts.delete(alertId);
      return true;
    }),

    getAlertStatistics: vi.fn(() => ({
      total_alerts: engine.alertHistory.length,
      active_alerts: engine.activeAlerts.size,
      by_severity: new Map([
        ['high', 2],
        ['medium', 1],
        ['critical', 1]
      ]),
      by_metric: new Map([
        ['cpu_usage', 2],
        ['memory_usage', 1],
        ['order_latency', 1]
      ]),
      by_venue: new Map([
        ['IB', 2],
        ['BINANCE', 2]
      ]),
      avg_resolution_time_minutes: 15.5,
      most_frequent_metric: 'cpu_usage',
      false_positive_rate: 5.2
    })),

    clear: vi.fn(() => {
      engine.thresholdRules.clear();
      engine.alertStates.clear();
      engine.activeAlerts.clear();
      engine.alertHistory = [];
    })
  };

  return engine;
};

// Time utilities for testing

export const timeUtils = {
  // Create timestamps at specific intervals
  createTimeSequence: (startTime: Date, intervalMs: number, count: number): Date[] => {
    return Array.from({ length: count }, (_, i) => 
      new Date(startTime.getTime() + i * intervalMs)
    );
  },

  // Create timestamps for last N minutes
  createRecentTimestamps: (minutes: number, intervalMinutes: number = 1): Date[] => {
    const now = new Date();
    const count = Math.floor(minutes / intervalMinutes);
    const intervalMs = intervalMinutes * 60 * 1000;
    return Array.from({ length: count }, (_, i) => 
      new Date(now.getTime() - (count - 1 - i) * intervalMs)
    );
  }
};

// Assertion helpers for monitoring tests

export const monitoringAssertions = {
  expectValidLatencyMetrics: (metrics: any) => {
    expect(metrics).toHaveProperty('venue_name');
    expect(metrics).toHaveProperty('order_execution_latency');
    expect(metrics.order_execution_latency).toHaveProperty('min_ms');
    expect(metrics.order_execution_latency).toHaveProperty('max_ms');
    expect(metrics.order_execution_latency).toHaveProperty('avg_ms');
    expect(metrics.order_execution_latency).toHaveProperty('p50_ms');
    expect(metrics.order_execution_latency).toHaveProperty('p95_ms');
    expect(metrics.order_execution_latency).toHaveProperty('p99_ms');
    expect(metrics.order_execution_latency).toHaveProperty('samples');
  },

  expectValidAlertStatistics: (stats: any) => {
    expect(stats).toHaveProperty('total_alerts');
    expect(stats).toHaveProperty('active_alerts');
    expect(stats).toHaveProperty('by_severity');
    expect(stats).toHaveProperty('by_metric');
    expect(stats).toHaveProperty('by_venue');
    expect(stats).toHaveProperty('avg_resolution_time_minutes');
    expect(typeof stats.total_alerts).toBe('number');
    expect(typeof stats.active_alerts).toBe('number');
    expect(typeof stats.avg_resolution_time_minutes).toBe('number');
  }
};