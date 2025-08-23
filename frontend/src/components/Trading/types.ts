// Advanced Trading Components - Type Definitions
// Comprehensive type definitions for institutional-grade trading widgets

import moment from 'moment';

// Dashboard Builder Types
export interface DashboardWidget {
  id: string;
  type: 'orderbook' | 'pnl-waterfall' | 'risk-heatmap' | 'strategy-performance' | 'trading-chart' | 'alert-system' | 'custom';
  title: string;
  component: string;
  config: Record<string, any>;
  layout: {
    x: number;
    y: number;
    w: number;
    h: number;
    minW?: number;
    minH?: number;
    maxW?: number;
    maxH?: number;
  };
  style: {
    backgroundColor?: string;
    borderColor?: string;
    borderWidth?: number;
    borderRadius?: number;
    opacity?: number;
  };
  enabled: boolean;
  locked: boolean;
}

export interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  category: 'trading' | 'risk' | 'performance' | 'analytics' | 'custom';
  widgets: DashboardWidget[];
  layout: 'grid' | 'masonry' | 'tabs' | 'sidebar';
  theme: 'light' | 'dark';
  responsiveBreakpoints: Record<string, number>;
}

// Order Book Types
export interface OrderBookLevel {
  price: number;
  size: number;
  count: number;
  cumulativeSize: number;
  percentage: number;
}

export interface OrderBookData {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  spreadBps: number;
  midPrice: number;
  totalBidVolume: number;
  totalAskVolume: number;
  timestamp: number;
  depth: number;
}

export interface HeatmapCell {
  price: number;
  volume: number;
  intensity: number;
  time: number;
}

export interface VolumeProfileLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentage: number;
}

// P&L Waterfall Types
export interface PnLComponent {
  id: string;
  name: string;
  value: number;
  previousValue: number;
  change: number;
  changePercent: number;
  category: 'trading' | 'fees' | 'funding' | 'other';
  subComponents?: PnLComponent[];
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface PnLWaterfallData {
  portfolioId: string;
  timestamp: number;
  totalPnL: number;
  previousTotalPnL: number;
  components: PnLComponent[];
  cumulativeChanges: number[];
  breakdown: {
    gross: number;
    net: number;
    fees: number;
    slippage: number;
    funding: number;
  };
}

export interface WaterfallBar {
  id: string;
  name: string;
  value: number;
  cumulative: number;
  start: number;
  end: number;
  type: 'positive' | 'negative' | 'total' | 'starting';
  category: string;
  isAnimating: boolean;
}

// Risk Heatmap Types
export interface RiskCell {
  id: string;
  symbol: string;
  riskType: 'var' | 'exposure' | 'concentration' | 'leverage' | 'correlation' | 'liquidity';
  value: number;
  normalizedValue: number;
  limit: number;
  utilization: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  trend: 'increasing' | 'decreasing' | 'stable';
  lastUpdate: number;
  metadata: {
    position: number;
    marketValue: number;
    volatility: number;
    beta: number;
    sector: string;
    geography: string;
  };
  drillDownData?: RiskCell[];
}

export interface RiskHeatmapData {
  portfolioId: string;
  timestamp: number;
  cells: RiskCell[];
  aggregatedRisk: {
    totalVaR: number;
    totalExposure: number;
    riskUtilization: number;
    breachCount: number;
  };
  dimensions: {
    symbols: string[];
    riskTypes: string[];
    sectors: string[];
    geographies: string[];
  };
}

// Strategy Performance Types
export interface StrategyMetrics {
  strategyId: string;
  name: string;
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  sortino: number;
  calmar: number;
  var95: number;
  beta: number;
  alpha: number;
  treynor: number;
  informationRatio: number;
  trackingError: number;
  returns: Array<{
    date: string;
    value: number;
    cumulativeReturn: number;
    drawdown: number;
  }>;
  trades: Array<{
    id: string;
    timestamp: number;
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    pnl: number;
    commission: number;
  }>;
  attribution: {
    alpha: number;
    beta: number;
    sectors: Array<{
      name: string;
      contribution: number;
      weight: number;
    }>;
    factors: Array<{
      name: string;
      exposure: number;
      contribution: number;
    }>;
  };
  riskMetrics: {
    var: number;
    expectedShortfall: number;
    maxDrawdown: number;
    drawdownDuration: number;
    volatility: number;
    downVolatility: number;
  };
  benchmark: {
    name: string;
    totalReturn: number;
    volatility: number;
    correlation: number;
    beta: number;
  };
}

// Trading Chart Types
export interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
}

export interface TechnicalIndicator {
  id: string;
  name: string;
  type: 'overlay' | 'oscillator' | 'volume';
  enabled: boolean;
  parameters: Record<string, any>;
  data: Array<{
    timestamp: number;
    value: number | number[];
    color?: string;
  }>;
  style: {
    lineColor: string;
    lineWidth: number;
    lineStyle: 'solid' | 'dashed' | 'dotted';
    fillColor?: string;
    opacity: number;
  };
}

export interface DrawingTool {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical' | 'rectangle' | 'fibonacci' | 'text';
  points: Array<{ 
    x: number; 
    y: number; 
    timestamp?: number; 
    price?: number; 
  }>;
  style: {
    color: string;
    width: number;
    style: 'solid' | 'dashed' | 'dotted';
    fill?: string;
    opacity?: number;
  };
  text?: string;
  locked: boolean;
  visible: boolean;
}

export interface ChartState {
  zoom: number;
  pan: { x: number; y: number };
  crosshair: { x: number; y: number; visible: boolean };
  selectedDrawing: string | null;
  drawingMode: string | null;
  mousePosition: { x: number; y: number };
  visibleRange: { start: number; end: number };
}

// Alert System Types
export interface AlertCondition {
  id: string;
  field: 'price' | 'volume' | 'pnl' | 'risk' | 'custom';
  operator: '>' | '<' | '=' | '>=' | '<=' | '!=' | 'between' | 'contains';
  value: any;
  value2?: any;
  logicalOperator?: 'AND' | 'OR';
}

export interface AlertAction {
  id: string;
  type: 'notification' | 'email' | 'sms' | 'webhook' | 'sound' | 'popup';
  enabled: boolean;
  config: Record<string, any>;
  priority: 'low' | 'medium' | 'high';
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'price' | 'risk' | 'performance' | 'system' | 'custom';
  conditions: AlertCondition[];
  actions: AlertAction[];
  schedule: {
    enabled: boolean;
    startTime?: string;
    endTime?: string;
    days?: string[];
    timezone?: string;
  };
  cooldown: number;
  maxAlerts: number;
  createdAt: number;
  updatedAt: number;
  lastTriggered?: number;
  triggerCount: number;
  metadata: Record<string, any>;
}

export interface AlertInstance {
  id: string;
  ruleId: string;
  ruleName: string;
  title: string;
  message: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  status: 'active' | 'acknowledged' | 'resolved' | 'dismissed';
  timestamp: number;
  acknowledgedAt?: number;
  acknowledgedBy?: string;
  resolvedAt?: number;
  resolvedBy?: string;
  data: Record<string, any>;
  actions: AlertAction[];
  escalated: boolean;
  escalatedAt?: number;
  snoozedUntil?: number;
}

// Common Widget Props
export interface BaseWidgetProps {
  compactMode?: boolean;
  height?: number;
  theme?: 'light' | 'dark';
  exportEnabled?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface SymbolWidgetProps extends BaseWidgetProps {
  symbol: string;
  onSymbolChange?: (symbol: string) => void;
}

export interface PortfolioWidgetProps extends BaseWidgetProps {
  portfolioId: string;
  onPortfolioChange?: (portfolioId: string) => void;
}

export interface StrategyWidgetProps extends BaseWidgetProps {
  strategyIds: string[];
  onStrategyChange?: (strategyIds: string[]) => void;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  timestamp: number;
  data: any;
  source?: string;
}

export interface AlertWebSocketMessage extends WebSocketMessage {
  type: 'alert_triggered' | 'alert_updated' | 'rule_created' | 'rule_updated' | 'rule_deleted';
  data: {
    alert?: AlertInstance;
    rule?: AlertRule;
    ruleId?: string;
  };
}

export interface MarketDataWebSocketMessage extends WebSocketMessage {
  type: 'price_update' | 'order_book_update' | 'trade_update';
  data: {
    symbol: string;
    price?: number;
    volume?: number;
    orderBook?: OrderBookData;
    trade?: {
      price: number;
      size: number;
      side: 'buy' | 'sell';
      timestamp: number;
    };
  };
}

// Export Configuration Types
export interface ExportConfig {
  format: 'png' | 'svg' | 'pdf' | 'csv' | 'json';
  quality?: number;
  width?: number;
  height?: number;
  includeLogo?: boolean;
  includeTimestamp?: boolean;
  filename?: string;
}

// Theme Configuration
export interface TradingTheme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    grid: string;
    bull: string;
    bear: string;
    neutral: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      small: number;
      medium: number;
      large: number;
      xlarge: number;
    };
    fontWeight: {
      normal: number;
      medium: number;
      bold: number;
    };
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  borderRadius: {
    small: number;
    medium: number;
    large: number;
  };
  shadows: {
    small: string;
    medium: string;
    large: string;
  };
}

// Responsive Breakpoints
export interface ResponsiveBreakpoints {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
}

// Performance Metrics
export interface PerformanceMetrics {
  renderTime: number;
  updateTime: number;
  memoryUsage: number;
  fps: number;
  dataPoints: number;
  lastUpdate: number;
}

// Widget State Management
export interface WidgetState {
  id: string;
  loading: boolean;
  error: string | null;
  data: any;
  lastUpdate: number;
  config: Record<string, any>;
  performance: PerformanceMetrics;
}

// Data Source Configuration
export interface DataSourceConfig {
  type: 'websocket' | 'rest' | 'mock';
  url: string;
  headers?: Record<string, string>;
  authentication?: {
    type: 'bearer' | 'basic' | 'apikey';
    token?: string;
    username?: string;
    password?: string;
    apiKey?: string;
  };
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  timeout?: number;
}

export type WidgetType = typeof WIDGET_TYPES[number];
export type TradingThemeType = typeof TRADING_THEMES[number];