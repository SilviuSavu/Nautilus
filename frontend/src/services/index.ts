/**
 * Services Index
 * =============
 * 
 * Central export point for all Nautilus frontend services.
 * Import services from here for consistency across the application.
 */

// Core services
export { authService } from './auth';
export { persistentApiClient } from './persistentApiClient';
export { MessageBusService } from './MessageBusService';
export { DirectAccessClient } from './DirectAccessClient';

// Trading and market data services
export { backtestService } from './backtestService';
export { engineService } from './engineService';
export { orderBookService } from './orderBookService';
export { positionService } from './positionService';
export { websocketService } from './websocket';

// Data and analytics services
export { chartDataProcessors } from './chartDataProcessors';
export { chartLayoutService } from './chartLayoutService';
export { correlationCalculator } from './correlationCalculator';
export { dataCatalogService } from './dataCatalogService';
export { dbnomicsService } from './dbnomicsService';
export { drawingService } from './drawingService';
export { indicatorEngine } from './indicatorEngine';
export { indicatorOptimization } from './indicatorOptimization';
export { multiDataSourceService } from './multiDataSourceService';
export { patternRecognition } from './patternRecognition';
export { performanceAttributionCalculator } from './performanceAttributionCalculator';
export { pnlCalculationEngine } from './pnlCalculationEngine';
export { portfolioAggregationService } from './portfolioAggregationService';
export { portfolioMetrics } from './portfolioMetrics';
export { tradingEconomicsService } from './tradingEconomicsService';

// 🚨 NEW: Collateral Management Service (Mission Critical)
export { collateralService, default as CollateralService } from './collateralService';
export type {
  Position,
  Portfolio,
  MarginRequirement,
  OptimizationResult,
  CrossMarginBenefit,
  MarginAlert,
  StressTestResult,
  RegulatoryReport,
  MonitoringStatus,
  EngineStatus,
} from './collateralService';

// Analytics services
export { PerformanceMetricsService } from './analytics/PerformanceMetricsService';

// Export services
export { ExportService } from './export/ExportService';

// Monitoring services
export { SystemMonitoringService } from './monitoring/SystemMonitoringService';
export { LatencyAlertEngine } from './monitoring/LatencyAlertEngine';
export { MarketDataLatencyMonitor } from './monitoring/MarketDataLatencyMonitor';
export { NetworkThroughputMonitor } from './monitoring/NetworkThroughputMonitor';
export { OrderLatencyTracker } from './monitoring/OrderLatencyTracker';
export { SystemResourceMonitor } from './monitoring/SystemResourceMonitor';
export { ThresholdAlertEngine } from './monitoring/ThresholdAlertEngine';

/**
 * Service endpoints by category for easy reference
 */
export const API_ENDPOINTS = {
  // Core API
  auth: '/api/auth',
  
  // Trading engines
  trading: '/api/v1/trading',
  execution: '/api/v1/execution',
  portfolio: '/api/v1/portfolio',
  
  // Market data
  market_data: '/api/v1/market-data',
  order_book: '/api/v1/order-book',
  
  // Analytics
  analytics: '/api/v1/analytics',
  performance: '/api/v1/performance',
  risk: '/api/v1/risk',
  
  // Data sources
  edgar: '/api/v1/edgar',
  fred: '/api/v1/fred',
  data_gov: '/api/v1/data-gov',
  dbnomics: '/api/v1/dbnomics',
  trading_economics: '/api/v1/trading-economics',
  
  // Engines
  factor_engine: '/api/v1/factor-engine',
  ml_engine: '/api/v1/ml',
  volatility_engine: '/api/v1/volatility',
  
  // 🚨 NEW: Collateral Management (Port 9000)
  collateral: '/api/v1/collateral',
  
  // Monitoring
  monitoring: '/api/v1/monitoring',
  system: '/api/v1/system',
  
  // Export
  export: '/api/v1/export',
} as const;

/**
 * Service status endpoints for health checks
 */
export const HEALTH_ENDPOINTS = {
  backend: '/health',
  collateral: '/api/v1/collateral/health',
  risk: '/api/v1/risk/health',
  analytics: '/api/v1/analytics/health',
  factor_engine: '/api/v1/factor-engine/health',
  ml_engine: '/api/v1/ml/health',
  volatility: '/api/v1/volatility/health',
  monitoring: '/api/v1/monitoring/health',
} as const;