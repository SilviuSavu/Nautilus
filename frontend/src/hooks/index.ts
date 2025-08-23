/**
 * Hooks Index
 * Sprint 3: Enterprise Trading Platform Hooks
 * 
 * Comprehensive export of all trading platform hooks including:
 * - Advanced WebSocket Infrastructure
 * - Performance Analytics & Monitoring
 * - Risk Management & ML Breach Detection
 * - Strategy Deployment & Testing
 */

// ===== ADVANCED WEBSOCKET INFRASTRUCTURE =====
// Core WebSocket Management
export { useWebSocketManager } from './useWebSocketManager';
export { useSubscriptionManager } from './useSubscriptionManager';
export { useRealTimeData } from './useRealTimeData';
export { useConnectionHealth } from './useConnectionHealth';

// Sprint 3 Advanced WebSocket Hooks
export { useWebSocketStream } from './useWebSocketStream';
export { useWebSocketHealth } from './useWebSocketHealth';
export { useWebSocketSubscriptions } from './useWebSocketSubscriptions';
export { useWebSocketReconnect } from './useWebSocketReconnect';

// Enhanced Market Data Hooks
export { useMarketData } from './useMarketData';
export { useTradeUpdatesEnhanced } from './useTradeUpdatesEnhanced';

// ===== PERFORMANCE ANALYTICS & MONITORING =====
// Existing Analytics
export { useRealTimeAnalytics } from './analytics/useRealTimeAnalytics';

// Sprint 3 Advanced Analytics
export { useAdvancedPerformanceMetrics } from './analytics/useAdvancedPerformanceMetrics';
export { useExecutionQuality } from './analytics/useExecutionQuality';

// ===== RISK MANAGEMENT & ML =====
// Sprint 3 Advanced Risk Management
export { useAdvancedRiskLimits } from './risk/useAdvancedRiskLimits';
export { useMLBreachPrediction } from './risk/useMLBreachPrediction';

// ===== STRATEGY DEPLOYMENT & TESTING =====
// Sprint 3 Strategy Management
export { useAdvancedDeploymentPipeline } from './strategy/useAdvancedDeploymentPipeline';
export { useAdvancedStrategyTesting } from './strategy/useAdvancedStrategyTesting';

// ===== LEGACY HOOKS (Backward Compatibility) =====
export { useEngineWebSocket } from './useEngineWebSocket';
export { useTradeUpdates } from './useTradeUpdates';
export { useRiskAlerts } from './useRiskAlerts';

// ===== DEFAULT EXPORTS FOR CONVENIENCE =====
// Core WebSocket
export { default as useWebSocketManager } from './useWebSocketManager';
export { default as useSubscriptionManager } from './useSubscriptionManager';
export { default as useRealTimeData } from './useRealTimeData';
export { default as useConnectionHealth } from './useConnectionHealth';

// Sprint 3 WebSocket
export { default as useWebSocketStream } from './useWebSocketStream';
export { default as useWebSocketHealth } from './useWebSocketHealth';
export { default as useWebSocketSubscriptions } from './useWebSocketSubscriptions';
export { default as useWebSocketReconnect } from './useWebSocketReconnect';

// Market Data
export { default as useMarketData } from './useMarketData';
export { default as useTradeUpdatesEnhanced } from './useTradeUpdatesEnhanced';

// Analytics
export { default as useRealTimeAnalytics } from './analytics/useRealTimeAnalytics';
export { default as useAdvancedPerformanceMetrics } from './analytics/useAdvancedPerformanceMetrics';
export { default as useExecutionQuality } from './analytics/useExecutionQuality';

// Risk Management
export { default as useAdvancedRiskLimits } from './risk/useAdvancedRiskLimits';
export { default as useMLBreachPrediction } from './risk/useMLBreachPrediction';

// Strategy Management
export { default as useAdvancedDeploymentPipeline } from './strategy/useAdvancedDeploymentPipeline';
export { default as useAdvancedStrategyTesting } from './strategy/useAdvancedStrategyTesting';

// Legacy
export { default as useEngineWebSocket } from './useEngineWebSocket';
export { default as useTradeUpdates } from './useTradeUpdates';
export { default as useRiskAlerts } from './useRiskAlerts';