/**
 * Collateral Management Service
 * ============================
 * 
 * Frontend service for the Nautilus Collateral Management Engine.
 * Provides real-time margin monitoring, cross-margining optimization,
 * and predictive margin call alerts.
 */

import { persistentApiClient } from './persistentApiClient';

// Types for collateral management
export interface Position {
  id: string;
  symbol: string;
  quantity: number;
  market_value: number;
  asset_class: 'equity' | 'bond' | 'fx' | 'derivative' | 'commodity' | 'crypto';
  currency?: string;
  sector?: string;
  country?: string;
  duration?: number;
  implied_volatility?: number;
  delta?: number;
  gamma?: number;
  theta?: number;
  vega?: number;
}

export interface Portfolio {
  id: string;
  name: string;
  positions: Position[];
  available_cash: number;
  currency?: string;
  leverage_ratio?: number;
}

export interface MarginRequirement {
  total_margin: number;
  net_initial_margin: number;
  variation_margin: number;
  cross_margin_offset: number;
  margin_utilization: number;
  margin_utilization_percent: number;
  margin_excess: number;
  time_to_margin_call_minutes?: number;
  position_margins: { [positionId: string]: { [key: string]: number } };
}

export interface OptimizationResult {
  original_margin: number;
  optimized_margin: number;
  margin_savings: number;
  capital_efficiency_improvement: number;
  cross_margin_benefits: CrossMarginBenefit[];
  computation_time_ms?: number;
  optimization_method: string;
}

export interface CrossMarginBenefit {
  asset_class: string;
  position_count: number;
  gross_margin: number;
  cross_margin_offset: number;
  offset_percentage: number;
  calculation_method: string;
}

export interface MarginAlert {
  portfolio_id: string;
  severity: 'ok' | 'info' | 'warning' | 'critical' | 'emergency';
  message: string;
  margin_utilization: number;
  time_to_margin_call_minutes?: number;
  recommended_action?: string;
  affected_positions?: string[];
  required_action_amount?: number;
  timestamp: string;
}

export interface StressTestResult {
  portfolio_id: string;
  stress_test_results: {
    scenario: string;
    base_margin: number;
    stressed_margin: number;
    margin_increase: number;
    margin_increase_percent: number;
    positions_at_risk: string[];
    estimated_liquidation_value: number;
    passes_test: boolean;
  }[];
  overall_assessment: {
    all_scenarios_passed: boolean;
    max_margin_increase_percent: number;
    recommendation: 'PASS' | 'INCREASE_BUFFER';
  };
  tested_scenarios: string[];
  test_timestamp: string;
}

export interface RegulatoryReport {
  regulatory_requirements: {
    basel_iii: number;
    dodd_frank: number;
    emir: number;
    local_regulatory: number;
    total_requirement: number;
  };
  capital_adequacy: {
    ratio: number;
    minimum_required: number;
    status: 'COMPLIANT' | 'NON_COMPLIANT';
  };
  compliance_status: {
    basel_iii_compliant: boolean;
    dodd_frank_compliant: boolean;
    emir_compliant: boolean;
    overall_compliant: boolean;
  };
  recommendations: string[];
  jurisdiction: string;
  entity_type: string;
  calculation_date: string;
}

export interface MonitoringStatus {
  is_monitoring: boolean;
  last_update?: string;
  calculation_time_ms?: number;
  active_alerts: number;
  current_margin?: {
    total_margin_requirement: number;
    margin_utilization: number;
    margin_utilization_percent: number;
    margin_excess: number;
    time_to_margin_call_minutes?: number;
  };
  config: {
    update_interval_seconds: number;
    warning_threshold: number;
    critical_threshold: number;
    emergency_threshold: number;
  };
}

export interface EngineStatus {
  engine_status: string;
  active_portfolios: number;
  performance_metrics: {
    calculations_performed: number;
    alerts_generated: number;
    optimizations_run: number;
    average_calculation_time_ms: number;
  };
  hardware_acceleration: boolean;
  risk_engine_integration: boolean;
  components: {
    margin_calculator: string;
    collateral_optimizer: string;
    margin_monitor: string;
    regulatory_calculator: string;
  };
}

class CollateralService {
  private baseUrl = '/api/v1/collateral';

  /**
   * Get engine health status
   */
  async getHealth(): Promise<{ status: string; engine_status: string; hardware_acceleration: boolean; timestamp: string }> {
    const response = await persistentApiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  /**
   * Get detailed engine status and performance metrics
   */
  async getEngineStatus(): Promise<EngineStatus> {
    const response = await persistentApiClient.get(`${this.baseUrl}/status`);
    return response.data;
  }

  /**
   * Calculate comprehensive margin requirements for a portfolio
   */
  async calculateMargin(
    portfolio: Portfolio, 
    optimize: boolean = true, 
    includeStressTest: boolean = false
  ): Promise<{
    margin_requirement: MarginRequirement;
    regulatory_capital: any;
    optimization?: OptimizationResult;
    stress_test?: StressTestResult;
    calculation_metadata: any;
  }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/margin/calculate`, {
      portfolio,
      optimize,
      include_stress_test: includeStressTest,
    });
    return response.data.data;
  }

  /**
   * Optimize margin requirements through cross-margining
   */
  async optimizeMargin(portfolio: Portfolio): Promise<OptimizationResult> {
    const response = await persistentApiClient.post(`${this.baseUrl}/margin/optimize`, portfolio);
    return response.data.data;
  }

  /**
   * Start real-time margin monitoring for a portfolio
   */
  async startMonitoring(
    portfolio: Portfolio, 
    config?: Partial<{
      update_interval_seconds: number;
      warning_threshold: number;
      critical_threshold: number;
      emergency_threshold: number;
      predictive_horizon_minutes: number;
      alert_cooldown_minutes: number;
      enable_predictive_alerts: boolean;
      enable_stress_testing: boolean;
    }>
  ): Promise<{ success: boolean; message: string; portfolio_id: string }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/monitoring/start`, {
      portfolio,
      monitoring_config: config,
    });
    return response.data;
  }

  /**
   * Stop real-time margin monitoring for a portfolio
   */
  async stopMonitoring(portfolioId: string): Promise<{ success: boolean; message: string; portfolio_id: string }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/monitoring/stop/${portfolioId}`);
    return response.data;
  }

  /**
   * Get current monitoring status for a portfolio
   */
  async getMonitoringStatus(portfolioId: string): Promise<{ success: boolean; portfolio_id: string; monitoring_status: MonitoringStatus }> {
    const response = await persistentApiClient.get(`${this.baseUrl}/monitoring/status/${portfolioId}`);
    return response.data;
  }

  /**
   * Optimize collateral allocation
   */
  async optimizeCollateralAllocation(
    portfolio: Portfolio,
    availableCollateral: { [collateralType: string]: number }
  ): Promise<{
    total_margin_requirement: number;
    collateral_allocation: { [type: string]: any };
    remaining_shortfall: number;
    total_haircut_cost: number;
    optimization_efficiency: number;
  }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/collateral/optimize-allocation`, {
      portfolio,
      available_collateral: availableCollateral,
    });
    return response.data.data;
  }

  /**
   * Run comprehensive margin stress tests
   */
  async runStressTest(
    portfolio: Portfolio,
    scenarios?: string[]
  ): Promise<StressTestResult> {
    const response = await persistentApiClient.post(`${this.baseUrl}/stress-test`, {
      portfolio,
      scenarios,
    });
    return response.data.data;
  }

  /**
   * Generate regulatory compliance report
   */
  async generateRegulatoryReport(
    portfolio: Portfolio,
    jurisdiction: string = 'US',
    entityType: string = 'hedge_fund'
  ): Promise<RegulatoryReport> {
    const response = await persistentApiClient.get(
      `${this.baseUrl}/regulatory/report/${portfolio.id}?jurisdiction=${jurisdiction}&entity_type=${entityType}`,
      {
        data: portfolio,
      }
    );
    return response.data.data;
  }

  /**
   * Generate comprehensive collateral management report
   */
  async generateComprehensiveReport(portfolio: Portfolio): Promise<{
    portfolio_id: string;
    report_timestamp: string;
    executive_summary: {
      total_margin_requirement: number;
      margin_utilization_percent: number;
      capital_efficiency_improvement: number;
      regulatory_compliant: boolean;
      stress_test_passed: boolean;
    };
    detailed_analysis: any;
    recommendations: string[];
  }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/reports/comprehensive`, portfolio);
    return response.data.data;
  }

  /**
   * Get performance metrics
   */
  async getPerformanceMetrics(): Promise<{
    performance_metrics: any;
    active_portfolios: number;
    hardware_acceleration: boolean;
    components_status: any;
    uptime: number;
  }> {
    const response = await persistentApiClient.get(`${this.baseUrl}/performance/metrics`);
    return response.data.data;
  }

  /**
   * Emergency stop all monitoring
   */
  async emergencyStopAllMonitoring(): Promise<{ success: boolean; message: string; timestamp: string }> {
    const response = await persistentApiClient.post(`${this.baseUrl}/emergency/stop-all-monitoring`);
    return response.data;
  }

  /**
   * Stream real-time margin alerts for a portfolio
   * Returns an EventSource for Server-Sent Events
   */
  streamMarginAlerts(portfolioId: string): EventSource {
    const eventSource = new EventSource(`${this.baseUrl}/alerts/stream/${portfolioId}`);
    return eventSource;
  }

  /**
   * Helper method to process margin alert stream
   */
  subscribeToMarginAlerts(
    portfolioId: string,
    onAlert: (alert: MarginAlert) => void,
    onError?: (error: Event) => void,
    onHeartbeat?: () => void
  ): EventSource {
    const eventSource = this.streamMarginAlerts(portfolioId);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'alert') {
          onAlert(data as MarginAlert);
        } else if (data.type === 'heartbeat') {
          onHeartbeat?.();
        } else if (data.error) {
          console.error('Collateral alert stream error:', data.error);
          onError?.(event);
        }
      } catch (error) {
        console.error('Error parsing alert stream data:', error);
        onError?.(event);
      }
    };

    eventSource.onerror = (error) => {
      console.error('Collateral alert stream connection error:', error);
      onError?.(error);
    };

    return eventSource;
  }

  /**
   * Format currency values for display
   */
  formatCurrency(value: number, currency: string = 'USD'): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  }

  /**
   * Format percentage values for display
   */
  formatPercentage(value: number, decimals: number = 1): string {
    return `${(value * 100).toFixed(decimals)}%`;
  }

  /**
   * Get severity color for margin alerts
   */
  getAlertSeverityColor(severity: string): string {
    switch (severity) {
      case 'emergency': return '#ff0000';
      case 'critical': return '#ff6600';
      case 'warning': return '#ffcc00';
      case 'info': return '#0099ff';
      case 'ok': return '#00cc00';
      default: return '#666666';
    }
  }

  /**
   * Get severity icon for margin alerts
   */
  getAlertSeverityIcon(severity: string): string {
    switch (severity) {
      case 'emergency': return '🚨';
      case 'critical': return '⚠️';
      case 'warning': return '⚡';
      case 'info': return 'ℹ️';
      case 'ok': return '✅';
      default: return '❓';
    }
  }

  /**
   * Calculate time until margin call in human-readable format
   */
  formatTimeToMarginCall(minutes?: number): string {
    if (!minutes) return 'N/A';
    
    if (minutes === 0) return 'Immediate';
    if (minutes < 60) return `${minutes}m`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
    
    const days = Math.floor(minutes / 1440);
    const hours = Math.floor((minutes % 1440) / 60);
    return `${days}d ${hours}h`;
  }

  /**
   * Calculate margin utilization risk level
   */
  getMarginUtilizationRiskLevel(utilization: number): 'low' | 'medium' | 'high' | 'critical' | 'emergency' {
    if (utilization >= 0.95) return 'emergency';
    if (utilization >= 0.90) return 'critical';
    if (utilization >= 0.80) return 'high';
    if (utilization >= 0.60) return 'medium';
    return 'low';
  }
}

export const collateralService = new CollateralService();
export default collateralService;