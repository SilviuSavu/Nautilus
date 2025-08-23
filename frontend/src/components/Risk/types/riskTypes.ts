export interface PortfolioRisk {
  portfolio_id: string;
  var_1d: string; // Using string for Decimal representation
  var_1w: string;
  var_1m: string;
  expected_shortfall: string;
  beta: number;
  correlation_matrix: CorrelationData[];
  concentration_risk: ConcentrationMetric[];
  total_exposure: string;
  last_calculated: Date;
}

export interface CorrelationData {
  symbol1: string;
  symbol2: string;
  correlation: number;
  confidence_level: number;
  calculation_period_days: number;
}

export interface ConcentrationMetric {
  category: 'instrument' | 'sector' | 'geography' | 'currency';
  name: string;
  exposure_amount: string;
  exposure_percentage: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface RiskLimit {
  id: string;
  name: string;
  portfolio_id: string;
  limit_type: 'var' | 'concentration' | 'position_size' | 'leverage' | 'correlation';
  threshold_value: string;
  warning_threshold: string;
  action: 'warn' | 'block' | 'reduce' | 'notify';
  active: boolean;
  breach_count: number;
  last_breach?: Date;
  created_at: Date;
  updated_at: Date;
}

export interface RiskAlert {
  id: string;
  portfolio_id: string;
  alert_type: 'limit_breach' | 'concentration_risk' | 'correlation_spike' | 'var_exceeded';
  severity: 'info' | 'warning' | 'critical';
  message: string;
  triggered_at: Date;
  acknowledged: boolean;
  acknowledged_by?: string;
  acknowledged_at?: Date;
  metadata: Record<string, any>;
}

export interface ExposureAnalysis {
  total_exposure: string;
  long_exposure: string;
  short_exposure: string;
  net_exposure: string;
  by_instrument: InstrumentExposure[];
  by_sector: SectorExposure[];
  by_currency: CurrencyExposure[];
  by_geography: GeographyExposure[];
}

export interface InstrumentExposure {
  symbol: string;
  position_size: string;
  market_value: string;
  percentage_of_portfolio: number;
  unrealized_pnl: string;
  risk_contribution: number;
}

export interface SectorExposure {
  sector: string;
  total_exposure: string;
  percentage_of_portfolio: number;
  instrument_count: number;
  top_holdings: InstrumentExposure[];
}

export interface CurrencyExposure {
  currency: string;
  exposure_amount: string;
  percentage_of_portfolio: number;
  hedge_ratio?: number;
}

export interface GeographyExposure {
  region: string;
  country?: string;
  exposure_amount: string;
  percentage_of_portfolio: number;
  political_risk_score?: number;
}

export interface RiskMetrics {
  portfolio_id: string;
  var_1d_95: string;
  var_1d_99: string;
  var_1w_95: string;
  var_1w_99: string;
  var_1m_95: string;
  var_1m_99: string;
  expected_shortfall_95: string;
  expected_shortfall_99: string;
  beta_vs_market: number;
  portfolio_volatility: number;
  sharpe_ratio: number;
  max_drawdown: string;
  correlation_with_market: number;
  tracking_error: number;
  information_ratio: number;
  calculated_at: Date;
}

export interface StressTestScenario {
  id: string;
  name: string;
  description: string;
  scenario_type: 'historical' | 'hypothetical' | 'monte_carlo';
  market_shocks: MarketShock[];
  expected_pnl_impact: string;
  var_impact: string;
  confidence_level: number;
  created_at: Date;
}

export interface MarketShock {
  asset_class: string;
  instrument?: string;
  shock_type: 'price_change' | 'volatility_change' | 'correlation_change';
  shock_magnitude: number; // percentage change
  shock_direction: 'up' | 'down' | 'both';
}

export interface PositionSizingRecommendation {
  portfolio_id: string;
  instrument: string;
  current_position: string;
  recommended_position: string;
  max_position_size: string;
  reasoning: string;
  risk_adjusted_size: string;
  kelly_criterion_size?: string;
  var_based_size: string;
  volatility_adjusted_size: string;
  confidence_score: number;
}

// API Request/Response Types
export interface RiskCalculationRequest {
  portfolio_id: string;
  calculation_type: 'var' | 'correlation' | 'exposure' | 'stress_test' | 'all';
  confidence_levels?: number[];
  time_horizons?: number[]; // days
  include_stress_tests?: boolean;
}

export interface RiskCalculationResponse {
  portfolio_id: string;
  status: 'success' | 'error' | 'partial';
  calculations: {
    risk_metrics?: RiskMetrics;
    exposure_analysis?: ExposureAnalysis;
    stress_test_results?: StressTestResult[];
  };
  calculation_time_ms: number;
  errors?: string[];
  warnings?: string[];
}

export interface StressTestResult {
  scenario: StressTestScenario;
  portfolio_pnl_impact: string;
  var_impact: string;
  position_impacts: Array<{
    instrument: string;
    current_value: string;
    shocked_value: string;
    pnl_impact: string;
  }>;
  risk_metrics_impact: {
    var_change: string;
    beta_change: number;
    correlation_change: number;
  };
}

export interface RiskLimitConfiguration {
  portfolio_id: string;
  limits: RiskLimit[];
  notification_settings: {
    email_alerts: boolean;
    dashboard_alerts: boolean;
    webhook_url?: string;
  };
  escalation_rules: EscalationRule[];
}

export interface EscalationRule {
  trigger_condition: 'multiple_breaches' | 'critical_breach' | 'time_based';
  threshold_count?: number;
  time_window_minutes?: number;
  action: 'email_manager' | 'auto_reduce' | 'halt_trading' | 'emergency_liquidate';
  recipients: string[];
}

export interface PreTradeRiskAssessment {
  portfolio_id: string;
  proposed_trade: {
    instrument: string;
    quantity: string;
    side: 'buy' | 'sell';
    order_type: string;
  };
  risk_impact: {
    var_impact: string;
    concentration_impact: number;
    correlation_impact: number;
    leverage_impact: number;
  };
  limit_violations: RiskLimitViolation[];
  recommendation: 'approve' | 'approve_with_warning' | 'reject';
  max_safe_quantity?: string;
  risk_score: number;
}

export interface RiskLimitViolation {
  limit_id: string;
  limit_name: string;
  current_value: string;
  limit_value: string;
  violation_severity: 'warning' | 'breach';
  recommended_action: string;
}

// Dashboard State Management
export interface RiskDashboardState {
  portfolio_risk: PortfolioRisk | null;
  risk_metrics: RiskMetrics | null;
  exposure_analysis: ExposureAnalysis | null;
  active_alerts: RiskAlert[];
  risk_limits: RiskLimit[];
  loading: {
    risk_calculations: boolean;
    exposure_data: boolean;
    alerts: boolean;
    limits: boolean;
  };
  errors: {
    risk_calculations?: string;
    exposure_data?: string;
    alerts?: string;
    limits?: string;
  };
  last_updated: {
    risk_calculations?: Date;
    exposure_data?: Date;
    alerts?: Date;
    limits?: Date;
  };
  real_time_enabled: boolean;
  auto_refresh_interval: number; // seconds
}

// WebSocket Message Types
export interface RiskWebSocketMessage {
  type: 'risk_update' | 'alert_triggered' | 'limit_breach' | 'calculation_complete';
  portfolio_id: string;
  timestamp: Date;
  data: any;
}

export interface RiskUpdateMessage extends RiskWebSocketMessage {
  type: 'risk_update';
  data: {
    risk_metrics?: Partial<RiskMetrics>;
    exposure_updates?: Partial<ExposureAnalysis>;
    new_alerts?: RiskAlert[];
  };
}

export interface AlertTriggeredMessage extends RiskWebSocketMessage {
  type: 'alert_triggered';
  data: {
    alert: RiskAlert;
    triggered_by: string;
    immediate_action_required: boolean;
  };
}

// Chart and Visualization Types
export interface RiskChartData {
  var_history: Array<{
    date: Date;
    var_95: number;
    var_99: number;
    actual_pnl?: number;
  }>;
  correlation_matrix: Array<{
    symbol1: string;
    symbol2: string;
    correlation: number;
  }>;
  concentration_pie: Array<{
    category: string;
    value: number;
    color: string;
  }>;
  exposure_timeline: Array<{
    date: Date;
    long_exposure: number;
    short_exposure: number;
    net_exposure: number;
  }>;
}

// Sprint 3 Enhanced Risk Management Types

// Enhanced Limit Types for Dynamic Risk Management
export interface DynamicRiskLimit extends RiskLimit {
  auto_adjustment_enabled: boolean;
  ml_prediction_enabled: boolean;
  adjustment_frequency_minutes: number;
  sensitivity_factor: number;
  adjustment_history: LimitAdjustment[];
  ml_confidence_score?: number;
  predicted_breach_time?: Date;
  breach_probability_24h?: number;
}

export interface LimitAdjustment {
  id: string;
  timestamp: Date;
  old_value: string;
  new_value: string;
  adjustment_reason: 'market_volatility' | 'ml_prediction' | 'manual' | 'breach_recovery';
  confidence_score: number;
  triggered_by?: string;
}

// ML-Based Breach Detection
export interface BreachPrediction {
  limit_id: string;
  limit_name: string;
  current_value: string;
  limit_value: string;
  predicted_breach_time: Date;
  breach_probability: number;
  confidence_score: number;
  contributing_factors: string[];
  recommended_actions: RecommendedAction[];
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface RecommendedAction {
  action_type: 'adjust_limit' | 'reduce_position' | 'hedge_exposure' | 'alert_manager';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  description: string;
  estimated_impact: string;
  execution_time_minutes?: number;
}

// Real-Time Monitoring
export interface RealTimeRiskMetrics {
  portfolio_id: string;
  timestamp: Date;
  var_95_current: number;
  var_99_current: number;
  portfolio_value: number;
  total_exposure: number;
  leverage_ratio: number;
  concentration_risk_score: number;
  correlation_risk_score: number;
  liquidity_risk_score: number;
  overall_risk_score: number;
  position_count: number;
  active_orders_count: number;
  margin_utilization: number;
  drawdown_current: number;
  volatility_24h: number;
}

// Multi-Format Reporting
export interface RiskReportRequest {
  portfolio_id: string;
  report_type: 'daily' | 'weekly' | 'monthly' | 'custom' | 'regulatory';
  format: 'json' | 'pdf' | 'csv' | 'excel' | 'html';
  sections: ReportSection[];
  date_range?: { start: Date; end: Date };
  include_charts?: boolean;
  include_recommendations?: boolean;
  regulatory_framework?: 'basel_iii' | 'mifid_ii' | 'dodd_frank' | 'custom';
}

export interface ReportSection {
  section_type: 'executive_summary' | 'var_analysis' | 'concentration' | 'stress_tests' | 'limit_breaches' | 'recommendations';
  enabled: boolean;
  detail_level: 'summary' | 'detailed' | 'full';
}

export interface RiskReport {
  id: string;
  request: RiskReportRequest;
  status: 'generating' | 'completed' | 'failed' | 'cancelled';
  created_at: Date;
  completed_at?: Date;
  file_url?: string;
  file_size_bytes?: number;
  error_message?: string;
  preview_data?: any;
}

// Advanced Alert System
export interface RiskAlertSprint3 extends RiskAlert {
  escalation_level: number;
  auto_escalation_enabled: boolean;
  escalation_rules: EscalationStep[];
  acknowledgment_required: boolean;
  resolution_required: boolean;
  related_alerts: string[];
  impact_assessment: ImpactAssessment;
  recommended_actions: RecommendedAction[];
  ml_generated: boolean;
  prediction_accuracy?: number;
}

export interface EscalationStep {
  level: number;
  trigger_condition: 'time_based' | 'severity_increase' | 'manual';
  delay_minutes: number;
  recipients: string[];
  channels: ('email' | 'sms' | 'slack' | 'webhook')[];
  auto_actions?: string[];
}

export interface ImpactAssessment {
  financial_impact_estimate: string;
  probability_of_loss: number;
  potential_max_loss: string;
  time_to_resolution_hours: number;
  affected_positions: string[];
  regulatory_implications: string[];
}

// Risk Configuration Panel
export interface RiskConfigurationSprint3 {
  portfolio_id: string;
  monitoring_enabled: boolean;
  update_frequency_seconds: number;
  alert_sensitivity: 'low' | 'medium' | 'high' | 'custom';
  ml_predictions_enabled: boolean;
  auto_limit_adjustment: boolean;
  emergency_procedures_enabled: boolean;
  risk_models: ActiveRiskModel[];
  notification_preferences: NotificationPreference[];
  compliance_frameworks: string[];
  custom_thresholds: CustomThreshold[];
}

export interface ActiveRiskModel {
  model_id: string;
  model_name: string;
  model_type: 'var' | 'correlation' | 'concentration' | 'liquidity' | 'credit';
  enabled: boolean;
  confidence_threshold: number;
  update_frequency_minutes: number;
  parameters: Record<string, any>;
}

export interface NotificationPreference {
  event_type: string;
  channels: ('dashboard' | 'email' | 'sms' | 'slack' | 'webhook')[];
  severity_threshold: 'info' | 'warning' | 'critical';
  quiet_hours_enabled: boolean;
  quiet_hours_start?: string;
  quiet_hours_end?: string;
}

export interface CustomThreshold {
  metric_name: string;
  warning_threshold: number;
  critical_threshold: number;
  calculation_method: string;
  enabled: boolean;
}

// Risk Limit Configuration Types
export interface RiskLimitType {
  id: string;
  name: string;
  description: string;
  category: 'position' | 'exposure' | 'var' | 'leverage' | 'concentration' | 'correlation' | 'drawdown' | 'volatility';
  calculation_method: string;
  default_thresholds: {
    warning: number;
    breach: number;
  };
  supports_auto_adjustment: boolean;
  supports_ml_prediction: boolean;
}

// Performance Metrics for Risk Components
export interface RiskComponentPerformance {
  component_name: string;
  last_update: Date;
  update_frequency_ms: number;
  calculation_time_ms: number;
  success_rate: number;
  error_count_24h: number;
  cache_hit_rate?: number;
}

// Utility Types
export type RiskCalculationType = 'var' | 'correlation' | 'exposure' | 'stress_test' | 'all';
export type TimeHorizon = '1d' | '1w' | '1m' | '3m' | '1y';
export type ConfidenceLevel = 90 | 95 | 99;

// Using string for Decimal compatibility
export interface Decimal {
  toNumber(): number;
  toFixed(digits: number): string;
}