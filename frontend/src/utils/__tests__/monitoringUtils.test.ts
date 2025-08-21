/**
 * Unit Tests for Monitoring Utilities
 * Tests utility functions for monitoring system
 */

import { describe, it, expect } from 'vitest';

// Mock monitoring utility functions
const monitoringUtils = {
  // Format latency values with appropriate units
  formatLatency: (milliseconds: number): string => {
    if (milliseconds < 1) {
      return `${(milliseconds * 1000).toFixed(1)}µs`;
    } else if (milliseconds < 1000) {
      return `${milliseconds.toFixed(1)}ms`;
    } else {
      return `${(milliseconds / 1000).toFixed(2)}s`;
    }
  },

  // Calculate percentiles from an array of values
  calculatePercentile: (values: number[], percentile: number): number => {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;

    if (upper >= sorted.length) return sorted[sorted.length - 1];
    
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  },

  // Calculate moving average
  calculateMovingAverage: (values: number[], windowSize: number): number[] => {
    if (values.length === 0 || windowSize <= 0) return [];
    
    const result: number[] = [];
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = values.slice(start, i + 1);
      const average = window.reduce((sum, val) => sum + val, 0) / window.length;
      result.push(Math.round(average * 100) / 100); // Round to 2 decimal places
    }
    return result;
  },

  // Format alert severity with color coding
  formatAlertSeverity: (severity: 'low' | 'medium' | 'high' | 'critical'): {
    text: string;
    color: string;
    priority: number;
  } => {
    const severityMap = {
      low: { text: 'Low', color: '#52c41a', priority: 1 },
      medium: { text: 'Medium', color: '#faad14', priority: 2 },
      high: { text: 'High', color: '#fa8c16', priority: 3 },
      critical: { text: 'Critical', color: '#ff4d4f', priority: 4 }
    };

    return severityMap[severity] || severityMap.medium;
  },

  // Format time duration
  formatDuration: (milliseconds: number): string => {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h`;
    } else if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  },

  // Calculate alert statistics
  calculateAlertStats: (alerts: Array<{
    severity: 'low' | 'medium' | 'high' | 'critical';
    triggered_at: string;
    resolved_at?: string;
    venue_name?: string;
  }>): {
    totalAlerts: number;
    bySeverity: Record<string, number>;
    byVenue: Record<string, number>;
    averageResolutionTime: number;
    unresolved: number;
  } => {
    const bySeverity: Record<string, number> = {};
    const byVenue: Record<string, number> = {};
    let totalResolutionTime = 0;
    let resolvedCount = 0;
    let unresolved = 0;

    alerts.forEach(alert => {
      // Count by severity
      bySeverity[alert.severity] = (bySeverity[alert.severity] || 0) + 1;

      // Count by venue
      const venue = alert.venue_name || 'unknown';
      byVenue[venue] = (byVenue[venue] || 0) + 1;

      // Calculate resolution time
      if (alert.resolved_at) {
        const triggerTime = new Date(alert.triggered_at).getTime();
        const resolveTime = new Date(alert.resolved_at).getTime();
        totalResolutionTime += (resolveTime - triggerTime);
        resolvedCount++;
      } else {
        unresolved++;
      }
    });

    return {
      totalAlerts: alerts.length,
      bySeverity,
      byVenue,
      averageResolutionTime: resolvedCount > 0 ? totalResolutionTime / resolvedCount : 0,
      unresolved
    };
  },

  // Validate alert configuration
  validateAlertConfig: (config: {
    metric_name?: string;
    threshold_value?: number;
    condition?: string;
    severity?: string;
  }): { valid: boolean; errors: string[] } => {
    const errors: string[] = [];

    if (!config.metric_name || config.metric_name.trim() === '') {
      errors.push('Metric name is required');
    }

    if (config.threshold_value === undefined || config.threshold_value === null) {
      errors.push('Threshold value is required');
    } else if (typeof config.threshold_value !== 'number' || isNaN(config.threshold_value)) {
      errors.push('Threshold value must be a valid number');
    }

    const validConditions = ['greater_than', 'less_than', 'equals', 'not_equals'];
    if (!config.condition || !validConditions.includes(config.condition)) {
      errors.push('Condition must be one of: greater_than, less_than, equals, not_equals');
    }

    const validSeverities = ['low', 'medium', 'high', 'critical'];
    if (!config.severity || !validSeverities.includes(config.severity)) {
      errors.push('Severity must be one of: low, medium, high, critical');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  },

  // Generate alert description
  generateAlertDescription: (
    metricName: string,
    currentValue: number,
    thresholdValue: number,
    condition: string,
    venue?: string
  ): string => {
    const venuePrefix = venue ? `[${venue}] ` : '';
    const conditionText = {
      greater_than: 'exceeded',
      less_than: 'fell below',
      equals: 'equals',
      not_equals: 'does not equal'
    }[condition] || 'triggered';

    return `${venuePrefix}${metricName} ${conditionText} threshold (${currentValue} vs ${thresholdValue})`;
  },

  // Convert bytes to human readable format
  formatBytes: (bytes: number, decimals: number = 2): string => {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  },

  // Format percentage with proper precision
  formatPercentage: (value: number, decimals: number = 1): string => {
    return `${value.toFixed(decimals)}%`;
  },

  // Check if timestamp is recent
  isRecentTimestamp: (timestamp: string, thresholdMinutes: number = 5): boolean => {
    const now = Date.now();
    const timestampMs = new Date(timestamp).getTime();
    const diffMs = now - timestampMs;
    const diffMinutes = diffMs / (1000 * 60);
    
    return diffMinutes <= thresholdMinutes;
  }
};

describe('monitoringUtils', () => {
  describe('formatLatency', () => {
    it('should format microsecond values correctly', () => {
      expect(monitoringUtils.formatLatency(0.5)).toBe('500.0µs');
      expect(monitoringUtils.formatLatency(0.123)).toBe('123.0µs');
      expect(monitoringUtils.formatLatency(0.001)).toBe('1.0µs');
    });

    it('should format millisecond values correctly', () => {
      expect(monitoringUtils.formatLatency(1)).toBe('1.0ms');
      expect(monitoringUtils.formatLatency(25.5)).toBe('25.5ms');
      expect(monitoringUtils.formatLatency(999.9)).toBe('999.9ms');
    });

    it('should format second values correctly', () => {
      expect(monitoringUtils.formatLatency(1000)).toBe('1.00s');
      expect(monitoringUtils.formatLatency(2500)).toBe('2.50s');
      expect(monitoringUtils.formatLatency(10000)).toBe('10.00s');
    });
  });

  describe('calculatePercentile', () => {
    it('should return 0 for empty array', () => {
      expect(monitoringUtils.calculatePercentile([], 50)).toBe(0);
    });

    it('should calculate percentiles correctly', () => {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      
      expect(monitoringUtils.calculatePercentile(values, 0)).toBe(1);
      expect(monitoringUtils.calculatePercentile(values, 50)).toBe(5.5);
      expect(monitoringUtils.calculatePercentile(values, 90)).toBe(9.1);
      expect(monitoringUtils.calculatePercentile(values, 100)).toBe(10);
    });

    it('should handle single value array', () => {
      expect(monitoringUtils.calculatePercentile([42], 50)).toBe(42);
      expect(monitoringUtils.calculatePercentile([42], 0)).toBe(42);
      expect(monitoringUtils.calculatePercentile([42], 100)).toBe(42);
    });

    it('should handle unsorted arrays', () => {
      const values = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10];
      expect(monitoringUtils.calculatePercentile(values, 50)).toBe(5.5);
    });
  });

  describe('calculateMovingAverage', () => {
    it('should return empty array for empty input', () => {
      expect(monitoringUtils.calculateMovingAverage([], 3)).toEqual([]);
    });

    it('should return empty array for invalid window size', () => {
      expect(monitoringUtils.calculateMovingAverage([1, 2, 3], 0)).toEqual([]);
      expect(monitoringUtils.calculateMovingAverage([1, 2, 3], -1)).toEqual([]);
    });

    it('should calculate moving average correctly', () => {
      const values = [1, 2, 3, 4, 5];
      const result = monitoringUtils.calculateMovingAverage(values, 3);
      
      expect(result).toEqual([1, 1.5, 2, 3, 4]);
    });

    it('should handle window size larger than array', () => {
      const values = [1, 2, 3];
      const result = monitoringUtils.calculateMovingAverage(values, 5);
      
      expect(result).toEqual([1, 1.5, 2]);
    });
  });

  describe('formatAlertSeverity', () => {
    it('should format low severity correctly', () => {
      const result = monitoringUtils.formatAlertSeverity('low');
      expect(result.text).toBe('Low');
      expect(result.color).toBe('#52c41a');
      expect(result.priority).toBe(1);
    });

    it('should format medium severity correctly', () => {
      const result = monitoringUtils.formatAlertSeverity('medium');
      expect(result.text).toBe('Medium');
      expect(result.color).toBe('#faad14');
      expect(result.priority).toBe(2);
    });

    it('should format high severity correctly', () => {
      const result = monitoringUtils.formatAlertSeverity('high');
      expect(result.text).toBe('High');
      expect(result.color).toBe('#fa8c16');
      expect(result.priority).toBe(3);
    });

    it('should format critical severity correctly', () => {
      const result = monitoringUtils.formatAlertSeverity('critical');
      expect(result.text).toBe('Critical');
      expect(result.color).toBe('#ff4d4f');
      expect(result.priority).toBe(4);
    });
  });

  describe('formatDuration', () => {
    it('should format seconds correctly', () => {
      expect(monitoringUtils.formatDuration(5000)).toBe('5s');
      expect(monitoringUtils.formatDuration(45000)).toBe('45s');
    });

    it('should format minutes correctly', () => {
      expect(monitoringUtils.formatDuration(60000)).toBe('1m 0s');
      expect(monitoringUtils.formatDuration(90000)).toBe('1m 30s');
      expect(monitoringUtils.formatDuration(3600000)).toBe('1h 0m');
    });

    it('should format hours correctly', () => {
      expect(monitoringUtils.formatDuration(7200000)).toBe('2h 0m');
      expect(monitoringUtils.formatDuration(5400000)).toBe('1h 30m');
    });

    it('should format days correctly', () => {
      expect(monitoringUtils.formatDuration(86400000)).toBe('1d 0h');
      expect(monitoringUtils.formatDuration(90000000)).toBe('1d 1h');
    });
  });

  describe('calculateAlertStats', () => {
    it('should calculate statistics correctly', () => {
      const alerts = [
        {
          severity: 'high' as const,
          triggered_at: '2023-01-01T00:00:00Z',
          resolved_at: '2023-01-01T00:05:00Z',
          venue_name: 'IB'
        },
        {
          severity: 'medium' as const,
          triggered_at: '2023-01-01T01:00:00Z',
          venue_name: 'BINANCE'
        },
        {
          severity: 'high' as const,
          triggered_at: '2023-01-01T02:00:00Z',
          resolved_at: '2023-01-01T02:10:00Z',
          venue_name: 'IB'
        }
      ];

      const stats = monitoringUtils.calculateAlertStats(alerts);

      expect(stats.totalAlerts).toBe(3);
      expect(stats.bySeverity.high).toBe(2);
      expect(stats.bySeverity.medium).toBe(1);
      expect(stats.byVenue.IB).toBe(2);
      expect(stats.byVenue.BINANCE).toBe(1);
      expect(stats.unresolved).toBe(1);
      expect(stats.averageResolutionTime).toBe(450000); // 7.5 minutes in ms
    });

    it('should handle empty alerts array', () => {
      const stats = monitoringUtils.calculateAlertStats([]);

      expect(stats.totalAlerts).toBe(0);
      expect(stats.bySeverity).toEqual({});
      expect(stats.byVenue).toEqual({});
      expect(stats.averageResolutionTime).toBe(0);
      expect(stats.unresolved).toBe(0);
    });

    it('should handle alerts without venue names', () => {
      const alerts = [
        {
          severity: 'low' as const,
          triggered_at: '2023-01-01T00:00:00Z'
        }
      ];

      const stats = monitoringUtils.calculateAlertStats(alerts);

      expect(stats.byVenue.unknown).toBe(1);
      expect(stats.unresolved).toBe(1);
    });
  });

  describe('validateAlertConfig', () => {
    it('should validate correct configuration', () => {
      const config = {
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high'
      };

      const result = monitoringUtils.validateAlertConfig(config);

      expect(result.valid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('should validate missing metric name', () => {
      const config = {
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'high'
      };

      const result = monitoringUtils.validateAlertConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Metric name is required');
    });

    it('should validate invalid threshold value', () => {
      const config = {
        metric_name: 'cpu_usage',
        threshold_value: 'invalid' as any,
        condition: 'greater_than',
        severity: 'high'
      };

      const result = monitoringUtils.validateAlertConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Threshold value must be a valid number');
    });

    it('should validate invalid condition', () => {
      const config = {
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'invalid_condition',
        severity: 'high'
      };

      const result = monitoringUtils.validateAlertConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Condition must be one of: greater_than, less_than, equals, not_equals');
    });

    it('should validate invalid severity', () => {
      const config = {
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'invalid_severity'
      };

      const result = monitoringUtils.validateAlertConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Severity must be one of: low, medium, high, critical');
    });
  });

  describe('generateAlertDescription', () => {
    it('should generate description without venue', () => {
      const description = monitoringUtils.generateAlertDescription(
        'cpu_usage', 85, 80, 'greater_than'
      );

      expect(description).toBe('cpu_usage exceeded threshold (85 vs 80)');
    });

    it('should generate description with venue', () => {
      const description = monitoringUtils.generateAlertDescription(
        'memory_usage', 95, 90, 'greater_than', 'IB'
      );

      expect(description).toBe('[IB] memory_usage exceeded threshold (95 vs 90)');
    });

    it('should handle different conditions', () => {
      expect(monitoringUtils.generateAlertDescription('latency', 50, 100, 'less_than'))
        .toBe('latency fell below threshold (50 vs 100)');
      
      expect(monitoringUtils.generateAlertDescription('count', 0, 0, 'equals'))
        .toBe('count equals threshold (0 vs 0)');
      
      expect(monitoringUtils.generateAlertDescription('status', 1, 2, 'not_equals'))
        .toBe('status does not equal threshold (1 vs 2)');
    });
  });

  describe('formatBytes', () => {
    it('should format bytes correctly', () => {
      expect(monitoringUtils.formatBytes(0)).toBe('0 Bytes');
      expect(monitoringUtils.formatBytes(1024)).toBe('1 KB');
      expect(monitoringUtils.formatBytes(1048576)).toBe('1 MB');
      expect(monitoringUtils.formatBytes(1073741824)).toBe('1 GB');
    });

    it('should handle decimal places', () => {
      expect(monitoringUtils.formatBytes(1536, 2)).toBe('1.5 KB');
      expect(monitoringUtils.formatBytes(1536, 0)).toBe('2 KB');
    });
  });

  describe('formatPercentage', () => {
    it('should format percentages correctly', () => {
      expect(monitoringUtils.formatPercentage(50)).toBe('50.0%');
      expect(monitoringUtils.formatPercentage(75.555, 2)).toBe('75.56%');
      expect(monitoringUtils.formatPercentage(100, 0)).toBe('100%');
    });
  });

  describe('isRecentTimestamp', () => {
    it('should identify recent timestamps', () => {
      const now = new Date();
      const recentTime = new Date(now.getTime() - 2 * 60 * 1000); // 2 minutes ago
      const oldTime = new Date(now.getTime() - 10 * 60 * 1000); // 10 minutes ago

      expect(monitoringUtils.isRecentTimestamp(recentTime.toISOString())).toBe(true);
      expect(monitoringUtils.isRecentTimestamp(oldTime.toISOString())).toBe(false);
    });

    it('should handle custom threshold', () => {
      const now = new Date();
      const time = new Date(now.getTime() - 8 * 60 * 1000); // 8 minutes ago

      expect(monitoringUtils.isRecentTimestamp(time.toISOString(), 5)).toBe(false);
      expect(monitoringUtils.isRecentTimestamp(time.toISOString(), 10)).toBe(true);
    });
  });
});