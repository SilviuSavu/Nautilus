// Existing components
export { default as RiskDashboard } from './RiskDashboard';
export { default as ExposureAnalysis } from './ExposureAnalysis';
export { default as RiskMetrics } from './RiskMetrics';
export { default as AlertSystem } from './AlertSystem';

// New Sprint 3 components
export { default as DynamicLimitEngine } from './DynamicLimitEngine';
export { default as DynamicRiskLimits } from './DynamicRiskLimits';
export { default as BreachDetector } from './BreachDetector';
export { default as RiskReporter } from './RiskReporter';
export { default as LimitMonitor } from './LimitMonitor';
export { default as RiskAlertCenter } from './RiskAlertCenter';
export { default as ComplianceReporting } from './ComplianceReporting';
export { default as VaRCalculator } from './VaRCalculator';

// Sprint 3 Enhanced Components
export { default as RealTimeRiskMonitor } from './RealTimeRiskMonitor';
export { default as AdvancedBreachDetector } from './AdvancedBreachDetector';
export { default as RiskReportGenerator } from './RiskReportGenerator';
export { default as RiskLimitConfigPanel } from './RiskLimitConfigPanel';
export { default as RiskDashboardSprint3 } from './RiskDashboardSprint3';

// Enhanced Risk Engine - Institutional Grade
export { default as EnhancedRiskDashboard } from './EnhancedRiskDashboard';

// Hook exports
export { useDynamicLimits } from '../../hooks/risk/useDynamicLimits';
export { useBreachDetection } from '../../hooks/risk/useBreachDetection';
export { useRiskReporting } from '../../hooks/risk/useRiskReporting';
export { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';

// Re-export types for convenience
export * from './types/riskTypes';

// Re-export service for convenience
export { riskService } from './services/riskService';