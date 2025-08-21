export interface StrategyTemplate {
  id: string;
  name: string;
  category: 'trend_following' | 'mean_reversion' | 'arbitrage' | 'market_making';
  description: string;
  python_class: string; // NautilusTrader strategy class name
  parameters: ParameterDefinition[];
  risk_parameters: RiskParameterDefinition[];
  example_configs: ExampleConfig[];
  documentation_url?: string;
  created_at: Date;
  updated_at: Date;
}

export interface ParameterDefinition {
  name: string;
  display_name: string;
  type: ParameterType;
  required: boolean;
  default_value?: any;
  min_value?: number;
  max_value?: number;
  allowed_values?: any[];
  validation_rules: ValidationRule[];
  help_text: string;
  group: string; // For UI grouping
}

export type ParameterType = 
  | 'string' 
  | 'integer' 
  | 'decimal' 
  | 'boolean' 
  | 'instrument_id' 
  | 'timeframe'
  | 'currency'
  | 'percentage';

export interface ValidationRule {
  type: 'range' | 'regex' | 'custom';
  params: Record<string, any>;
  error_message: string;
}

export interface RiskParameterDefinition extends ParameterDefinition {
  impact_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface ExampleConfig {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface StrategyConfig {
  id: string;
  name: string;
  template_id: string;
  user_id: string;
  parameters: Record<string, any>;
  risk_settings: RiskSettings;
  deployment_settings: DeploymentSettings;
  version: number;
  status: 'draft' | 'validated' | 'deployed' | 'archived';
  created_at: Date;
  updated_at: Date;
  tags: string[];
}

export interface RiskSettings {
  max_position_size: string; // Using string for Decimal representation
  stop_loss_atr?: number;
  take_profit_atr?: number;
  max_daily_loss?: string;
  position_sizing_method: 'fixed' | 'percentage' | 'volatility_adjusted';
}

export interface DeploymentSettings {
  mode: 'live' | 'paper' | 'backtest';
  venue: string;
  start_time?: Date;
  end_time?: Date;
  initial_balance?: string;
}

export interface StrategyInstance {
  id: string;
  config_id: string;
  nautilus_strategy_id: string;
  deployment_id: string;
  state: StrategyState;
  performance_metrics: PerformanceMetrics;
  runtime_info: RuntimeInfo;
  error_log: ErrorEntry[];
  started_at: Date;
  stopped_at?: Date;
}

export type StrategyState = 
  | 'initializing'
  | 'running' 
  | 'paused' 
  | 'stopping' 
  | 'stopped' 
  | 'error' 
  | 'completed';

export interface PerformanceMetrics {
  total_pnl: string;
  unrealized_pnl: string;
  total_trades: number;
  winning_trades: number;
  win_rate: number;
  max_drawdown: string;
  sharpe_ratio?: number;
  last_updated: Date;
}

export interface RuntimeInfo {
  orders_placed: number;
  positions_opened: number;
  last_signal_time?: Date;
  cpu_usage?: number;
  memory_usage?: number;
  uptime_seconds: number;
}

export interface ErrorEntry {
  timestamp: Date;
  level: 'warning' | 'error' | 'critical';
  message: string;
  nautilus_error?: string;
  stack_trace?: string;
}

export interface StrategyError {
  type: 'validation' | 'deployment' | 'runtime' | 'connection';
  severity: 'warning' | 'error' | 'critical';
  message: string;
  nautilus_error?: string;
  recovery_suggestions: string[];
  timestamp: Date;
}

// API Response Types
export interface TemplateResponse {
  templates: StrategyTemplate[];
  categories: string[];
}

export interface ConfigureRequest {
  template_id: string;
  name: string;
  parameters: Record<string, any>;
  risk_settings?: RiskSettings;
}

export interface ConfigureResponse {
  strategy_id: string;
  config: StrategyConfig;
  validation_result: ValidationResult;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface ValidationWarning {
  field: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
}

export interface DeployRequest {
  strategy_id: string;
  deployment_mode: 'live' | 'paper' | 'backtest';
}

export interface DeployResponse {
  deployment_id: string;
  status: 'deploying' | 'running' | 'failed';
  nautilus_strategy_id?: string;
  error_message?: string;
}

export interface ControlRequest {
  action: 'start' | 'stop' | 'pause' | 'resume';
  force?: boolean;
}

export interface ControlResponse {
  status: 'success' | 'error';
  new_state: StrategyState;
  message?: string;
}

export interface StatusResponse {
  strategy_id: string;
  state: StrategyState;
  performance_metrics: PerformanceMetrics;
  last_error?: string;
  runtime_info: RuntimeInfo;
}

export interface StrategySearchFilters {
  category?: string;
  search_text?: string;
  tags?: string[];
  status?: string;
}

export interface TemplateCategory {
  id: string;
  name: string;
  description: string;
  template_count: number;
  icon?: string;
}

export interface DeploymentResult {
  deployment_id: string;
  start_time: Date;
  end_time?: Date;
  final_pnl?: Decimal;
  trade_count: number;
  success: boolean;
  notes?: string;
}

// Using string for Decimal compatibility
export interface Decimal {
  toNumber(): number;
  toFixed(digits: number): string;
}

// Version Control Types
export interface StrategyVersion {
  id: string;
  config_id: string;
  version_number: number;
  config_snapshot: StrategyConfig;
  change_summary: string;
  created_by: string;
  created_at: Date;
  deployment_results?: DeploymentResult[];
}

export interface ConfigurationChange {
  id: string;
  strategy_id: string;
  change_type: 'parameter_change' | 'deployment' | 'pause' | 'stop' | 'save' | 'rollback';
  timestamp: Date;
  changed_by: string;
  description?: string;
  reason?: string;
  version?: number;
  changed_fields?: string[];
  config_diff?: Record<string, any>;
  config_snapshot?: ConfigurationSnapshot;
  auto_generated: boolean;
  deployment_mode?: string;
  rollback_version?: number;
  performance_before?: PerformanceMetrics;
  performance_after?: PerformanceMetrics;
}

export interface ConfigurationSnapshot {
  snapshot_id: string;
  timestamp: Date;
  config_data: StrategyConfig;
  version_number: number;
  checksum: string;
}

export interface ConfigurationAudit {
  id: string;
  timestamp: Date;
  action: string;
  user_id: string;
  details?: string;
  risk_level?: 'low' | 'medium' | 'high';
  warnings?: string[];
}

export interface VersionComparisonResult {
  version1: StrategyVersion;
  version2: StrategyVersion;
  differences: ConfigurationDiff[];
  performance_comparison?: PerformanceComparison;
  configuration_diff: {
    parameter_changes: ParameterChange[];
    total_changes: number;
    high_impact_changes: number;
    medium_impact_changes: number;
    low_impact_changes: number;
  };
}

export interface ConfigurationDiff {
  type: 'added' | 'removed' | 'modified' | 'unchanged';
  path: string;
  old_value?: any;
  new_value?: any;
  impact_level?: 'low' | 'medium' | 'high';
}

export interface ParameterChange {
  parameter_name: string;
  change_type: 'added' | 'removed' | 'modified' | 'unchanged';
  old_value?: any;
  new_value?: any;
  category?: string;
  impact_level?: 'low' | 'medium' | 'high';
  description?: string;
}

export interface PerformanceComparison {
  version1_pnl?: Decimal;
  version1_trades?: number;
  version1_win_rate?: number;
  version1_sharpe?: number;
  version2_pnl?: Decimal;
  version2_trades?: number;
  version2_win_rate?: number;
  version2_sharpe?: number;
  pnl_difference?: Decimal;
  win_rate_change?: number;
  trade_count_change?: number;
  sharpe_improvement?: number;
  statistical_significance?: number;
  performance_breakdown?: Array<{
    metric: string;
    version1: any;
    version2: any;
    change: any;
  }>;
}

// Rollback System Types
export interface RollbackPlan {
  strategy_id: string;
  from_version: number;
  to_version: number;
  changes_to_revert: ConfigurationChange[];
  execution_steps: RollbackStep[];
  risk_assessment: RiskAssessment;
  estimated_duration_seconds: number;
  backup_required: boolean;
  dependencies: string[];
}

export interface RollbackStep {
  step_id: string;
  description: string;
  action_type: 'config_change' | 'database_update' | 'strategy_restart' | 'validation' | 'backup';
  estimated_duration?: number;
  critical: boolean;
  rollback_action?: string;
}

export interface RiskAssessment {
  risk_level: 'low' | 'medium' | 'high';
  warnings: string[];
  blockers: string[];
  recommendations: string[];
}

export interface RollbackValidation {
  validation_passed: boolean;
  validation_errors: string[];
  warnings: string[];
  pre_rollback_checks: Array<{
    check_name: string;
    description: string;
    passed: boolean;
    details?: string;
  }>;
  backup_verification?: {
    backup_created: boolean;
    backup_path: string;
    backup_size_mb: number;
    backup_verified: boolean;
  };
}

export interface RollbackProgress {
  rollback_id: string;
  status: 'initializing' | 'running' | 'completed' | 'failed' | 'rolled_back';
  overall_progress: number;
  current_step: string;
  current_operation?: string;
  completed_steps: number;
  total_steps: number;
  elapsed_seconds: number;
  estimated_remaining_seconds: number;
  errors: string[];
  warnings: string[];
}

export interface BackupSnapshot {
  backup_id: string;
  strategy_id: string;
  timestamp: Date;
  config_backup: StrategyConfig;
  version_backup: number;
  backup_path: string;
  size_bytes: number;
  checksum: string;
  retention_until: Date;
}