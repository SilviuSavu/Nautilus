/**
 * Story 5.2: System Performance Monitoring Components Export
 * Central export file for all monitoring components
 * Updated for Sprint 3: Advanced System Monitoring Infrastructure
 */

// Original monitoring components
export { default as SystemPerformanceDashboard } from './SystemPerformanceDashboard';
export { default as CPUMemoryDashboard } from './CPUMemoryDashboard';
export { default as ConnectionHealthDashboard } from './ConnectionHealthDashboard';
export { default as NetworkMonitoringDashboard } from './NetworkMonitoringDashboard';

// Sprint 3: Advanced monitoring components
export { default as Sprint3SystemMonitor } from './Sprint3SystemMonitor';
export { default as PrometheusMetricsDashboard } from './PrometheusMetricsDashboard';
export { default as GrafanaEmbedDashboard } from './GrafanaEmbedDashboard';
export { default as AlertRulesManager } from './AlertRulesManager';
export { default as SystemHealthAggregator } from './SystemHealthAggregator';
export { default as PerformanceTrendAnalyzer } from './PerformanceTrendAnalyzer';
export { default as ComponentStatusMatrix } from './ComponentStatusMatrix';
export { default as SystemResourceMonitor } from './SystemResourceMonitor';

// Export types for external use
export type { 
  LatencyMetrics,
  SystemMetrics,
  ConnectionQuality,
  PerformanceAlert,
  MonitoringState
} from '../../types/monitoring';

// Export services for external use
export { systemMonitoringService } from '../../services/monitoring/SystemMonitoringService';

// Export hooks for external use
export { useSystemMonitoring } from '../../hooks/monitoring/useSystemMonitoring';