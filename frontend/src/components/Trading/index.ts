// Advanced Trading Dashboard Components
// Professional institutional-grade trading widgets and visualizations

export { default as AdvancedOrderBookWidget } from './AdvancedOrderBookWidget';
export { default as PnLWaterfallWidget } from './PnLWaterfallWidget';
export { default as RiskHeatmapWidget } from './RiskHeatmapWidget';
export { default as StrategyPerformanceWidget } from './StrategyPerformanceWidget';
export { default as AdvancedTradingChart } from './AdvancedTradingChart';
export { default as AlertNotificationSystem } from './AlertNotificationSystem';
export { default as CustomDashboardBuilder } from './CustomDashboardBuilder';

// Re-export types and interfaces
export type { 
  DashboardWidget,
  DashboardTemplate,
  AlertRule,
  AlertInstance,
  AlertAction,
  TechnicalIndicator,
  DrawingTool,
  StrategyMetrics,
  RiskCell,
  PnLComponent,
  CandlestickData
} from './types';

// Widget configurations and utilities
export const WIDGET_TYPES = [
  'orderbook',
  'pnl-waterfall',
  'risk-heatmap',
  'strategy-performance',
  'trading-chart',
  'alert-system'
] as const;

export const TRADING_THEMES = ['light', 'dark'] as const;

export const DEFAULT_WIDGET_CONFIG = {
  orderbook: {
    symbol: 'AAPL',
    depth: 20,
    showHeatmap: true,
    showVolumeProfile: true,
    showSpreadAnalysis: true
  },
  'pnl-waterfall': {
    portfolioId: 'default',
    granularity: 'hour',
    showBreakdown: true,
    showAnimation: true
  },
  'risk-heatmap': {
    portfolioId: 'default',
    viewDimensions: 'symbol-risk',
    colorScheme: 'red-green',
    showSeverityOnly: false
  },
  'strategy-performance': {
    strategyIds: ['strategy1'],
    benchmarkId: 'SPY',
    showComparison: true,
    showAttribution: true,
    showRiskMetrics: true
  },
  'trading-chart': {
    symbol: 'AAPL',
    timeframe: '1h',
    chartType: 'candlestick',
    showVolume: true,
    enableDrawing: true,
    enableCrosshair: true
  },
  'alert-system': {
    portfolioId: 'default',
    soundEnabled: true,
    popupEnabled: true,
    maxDisplayedAlerts: 50
  }
};