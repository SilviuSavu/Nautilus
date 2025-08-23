import axios, { AxiosResponse } from 'axios';
import {
  PortfolioRisk,
  RiskMetrics,
  ExposureAnalysis,
  RiskAlert,
  RiskLimit,
  RiskCalculationRequest,
  RiskCalculationResponse,
  StressTestScenario,
  StressTestResult,
  RiskLimitConfiguration,
  PreTradeRiskAssessment,
  PositionSizingRecommendation,
  RiskChartData,
  // Sprint 3 Types
  DynamicRiskLimit,
  BreachPrediction,
  RealTimeRiskMetrics,
  RiskReportRequest,
  RiskReport,
  RiskAlertSprint3,
  RiskConfigurationSprint3,
  RiskLimitType,
  RiskComponentPerformance
} from '../types/riskTypes';

const API_BASE = `/api/v1`;

class RiskService {
  private async apiCall<T>(url: string, options?: any): Promise<T> {
    try {
      const response: AxiosResponse<T> = await axios({
        url: `${API_BASE}${url}`,
        timeout: 10000,
        ...options
      });
      return response.data;
    } catch (error: any) {
      console.error(`Risk API Error [${url}]:`, error);
      throw new Error(
        error.response?.data?.error?.message || 
        error.message || 
        'Risk service error'
      );
    }
  }

  // Portfolio Risk Overview
  async getPortfolioRisk(portfolioId: string): Promise<PortfolioRisk> {
    return this.apiCall<PortfolioRisk>(`/risk/portfolio/${portfolioId}`);
  }

  async calculateRiskMetrics(request: RiskCalculationRequest): Promise<RiskCalculationResponse> {
    return this.apiCall<RiskCalculationResponse>('/risk/calculate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: request
    });
  }

  async getRiskMetrics(portfolioId: string): Promise<RiskMetrics> {
    return this.apiCall<RiskMetrics>(`/risk/metrics/${portfolioId}`);
  }

  // Exposure Analysis
  async getExposureAnalysis(portfolioId: string): Promise<ExposureAnalysis> {
    return this.apiCall<ExposureAnalysis>(`/risk/exposure/${portfolioId}`);
  }

  async getExposureBreakdown(
    portfolioId: string, 
    breakdown_type: 'instrument' | 'sector' | 'currency' | 'geography'
  ): Promise<any> {
    return this.apiCall(`/risk/exposure/${portfolioId}/breakdown?type=${breakdown_type}`);
  }

  // Risk Alerts
  async getRiskAlerts(portfolioId: string): Promise<RiskAlert[]> {
    return this.apiCall<RiskAlert[]>(`/risk/alerts/${portfolioId}`);
  }

  async acknowledgeAlert(alertId: string, acknowledgedBy: string): Promise<void> {
    return this.apiCall<void>(`/risk/alerts/${alertId}/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { acknowledged_by: acknowledgedBy }
    });
  }

  async dismissAlert(alertId: string): Promise<void> {
    return this.apiCall<void>(`/risk/alerts/${alertId}/dismiss`, {
      method: 'DELETE'
    });
  }

  // Risk Limits
  async getRiskLimits(portfolioId: string): Promise<RiskLimit[]> {
    return this.apiCall<RiskLimit[]>(`/risk/limits/${portfolioId}`);
  }

  async createRiskLimit(limit: Omit<RiskLimit, 'id' | 'created_at' | 'updated_at'>): Promise<RiskLimit> {
    return this.apiCall<RiskLimit>('/risk/limits', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: limit
    });
  }

  async updateRiskLimit(limitId: string, updates: Partial<RiskLimit>): Promise<RiskLimit> {
    return this.apiCall<RiskLimit>(`/risk/limits/${limitId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      data: updates
    });
  }

  async deleteRiskLimit(limitId: string): Promise<void> {
    return this.apiCall<void>(`/risk/limits/${limitId}`, {
      method: 'DELETE'
    });
  }

  async configureRiskLimits(config: RiskLimitConfiguration): Promise<void> {
    return this.apiCall<void>('/risk/limits/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: config
    });
  }

  // Stress Testing
  async getStressTestScenarios(): Promise<StressTestScenario[]> {
    return this.apiCall<StressTestScenario[]>('/risk/stress-tests/scenarios');
  }

  async runStressTest(
    portfolioId: string, 
    scenarioId: string
  ): Promise<StressTestResult> {
    return this.apiCall<StressTestResult>('/risk/stress-tests/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { portfolio_id: portfolioId, scenario_id: scenarioId }
    });
  }

  async createCustomStressTest(scenario: Omit<StressTestScenario, 'id' | 'created_at'>): Promise<StressTestScenario> {
    return this.apiCall<StressTestScenario>('/risk/stress-tests/scenarios', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: scenario
    });
  }

  // Pre-Trade Risk Assessment
  async assessPreTradeRisk(assessment: {
    portfolio_id: string;
    instrument: string;
    quantity: string;
    side: 'buy' | 'sell';
    order_type: string;
  }): Promise<PreTradeRiskAssessment> {
    return this.apiCall<PreTradeRiskAssessment>('/risk/assess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: assessment
    });
  }

  // Position Sizing
  async getPositionSizingRecommendation(
    portfolioId: string,
    instrument: string,
    targetRiskPercentage?: number
  ): Promise<PositionSizingRecommendation> {
    const params = new URLSearchParams();
    if (targetRiskPercentage !== undefined) {
      params.append('target_risk_pct', targetRiskPercentage.toString());
    }
    
    return this.apiCall<PositionSizingRecommendation>(
      `/risk/position-sizing/${portfolioId}/${instrument}?${params}`
    );
  }

  async calculateOptimalPositionSize(
    portfolioId: string,
    instrument: string,
    riskParameters: {
      max_risk_per_trade?: number;
      target_volatility?: number;
      kelly_fraction?: number;
      confidence_level?: number;
    }
  ): Promise<PositionSizingRecommendation> {
    return this.apiCall<PositionSizingRecommendation>('/risk/position-sizing/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: {
        portfolio_id: portfolioId,
        instrument,
        risk_parameters: riskParameters
      }
    });
  }

  // Chart Data for Risk Visualization
  async getRiskChartData(
    portfolioId: string,
    chartType: 'var_history' | 'correlation_matrix' | 'concentration' | 'exposure_timeline',
    dateRange?: { start: Date; end: Date }
  ): Promise<RiskChartData> {
    const params = new URLSearchParams();
    params.append('chart_type', chartType);
    if (dateRange) {
      params.append('start_date', dateRange.start.toISOString());
      params.append('end_date', dateRange.end.toISOString());
    }

    return this.apiCall<RiskChartData>(`/risk/charts/${portfolioId}?${params}`);
  }

  // Real-time Risk Monitoring
  async enableRealTimeMonitoring(portfolioId: string): Promise<void> {
    return this.apiCall<void>(`/risk/monitor/${portfolioId}/enable`, {
      method: 'POST'
    });
  }

  async disableRealTimeMonitoring(portfolioId: string): Promise<void> {
    return this.apiCall<void>(`/risk/monitor/${portfolioId}/disable`, {
      method: 'POST'
    });
  }

  async getRealTimeRiskStatus(portfolioId: string): Promise<{
    monitoring_enabled: boolean;
    last_update: Date;
    update_frequency_seconds: number;
    active_alerts_count: number;
  }> {
    return this.apiCall(`/risk/monitor/${portfolioId}/status`);
  }

  // Risk Report Generation
  async generateRiskReport(
    portfolioId: string,
    reportType: 'daily' | 'weekly' | 'monthly' | 'custom',
    options?: {
      include_stress_tests?: boolean;
      include_scenarios?: boolean;
      format?: 'json' | 'pdf' | 'excel';
      date_range?: { start: Date; end: Date };
    }
  ): Promise<any> {
    return this.apiCall('/risk/reports/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: {
        portfolio_id: portfolioId,
        report_type: reportType,
        options: options || {}
      }
    });
  }

  // Portfolio Correlation Analysis
  async getCorrelationMatrix(
    portfolioId: string,
    timeframe: '1d' | '1w' | '1m' | '3m' | '1y' = '1m'
  ): Promise<Array<{ symbol1: string; symbol2: string; correlation: number }>> {
    return this.apiCall(`/risk/correlation/${portfolioId}?timeframe=${timeframe}`);
  }

  async getPortfolioCorrelationWithMarket(
    portfolioId: string,
    marketIndex: string = 'SPY'
  ): Promise<{
    correlation: number;
    beta: number;
    tracking_error: number;
    information_ratio: number;
  }> {
    return this.apiCall(`/risk/correlation/${portfolioId}/market?index=${marketIndex}`);
  }

  // Utility Methods
  async getAvailableRiskModels(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    supported_calculations: string[];
  }>> {
    return this.apiCall('/risk/models/available');
  }

  async getRiskConfiguration(portfolioId: string): Promise<{
    calculation_frequency: number;
    confidence_levels: number[];
    time_horizons: number[];
    stress_test_enabled: boolean;
    real_time_monitoring: boolean;
  }> {
    return this.apiCall(`/risk/config/${portfolioId}`);
  }

  async updateRiskConfiguration(
    portfolioId: string,
    config: {
      calculation_frequency?: number;
      confidence_levels?: number[];
      time_horizons?: number[];
      stress_test_enabled?: boolean;
      real_time_monitoring?: boolean;
    }
  ): Promise<void> {
    return this.apiCall(`/risk/config/${portfolioId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      data: config
    });
  }

  // Health check for risk service
  async healthCheck(): Promise<{ status: string; timestamp: Date; services: any }> {
    return this.apiCall<{ status: string; timestamp: Date; services: any }>('/risk/health');
  }

  // Historical VaR Backtesting
  async backtestVaR(
    portfolioId: string,
    startDate: Date,
    endDate: Date,
    confidenceLevel: number = 95
  ): Promise<{
    total_days: number;
    breaches: number;
    breach_rate: number;
    expected_breach_rate: number;
    statistical_significance: number;
    results: Array<{
      date: Date;
      predicted_var: number;
      actual_pnl: number;
      breach: boolean;
    }>;
  }> {
    return this.apiCall('/risk/backtest/var', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: {
        portfolio_id: portfolioId,
        start_date: startDate.toISOString(),
        end_date: endDate.toISOString(),
        confidence_level: confidenceLevel
      }
    });
  }

  // =================== SPRINT 3 ENHANCED METHODS ===================

  // Dynamic Risk Limits Management
  async getDynamicLimits(portfolioId: string): Promise<DynamicRiskLimit[]> {
    return this.apiCall<DynamicRiskLimit[]>(`/risk/limits/dynamic/${portfolioId}`);
  }

  async createDynamicLimit(limit: Omit<DynamicRiskLimit, 'id' | 'created_at' | 'updated_at'>): Promise<DynamicRiskLimit> {
    return this.apiCall<DynamicRiskLimit>('/risk/limits/dynamic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: limit
    });
  }

  async updateDynamicLimit(limitId: string, updates: Partial<DynamicRiskLimit>): Promise<DynamicRiskLimit> {
    return this.apiCall<DynamicRiskLimit>(`/risk/limits/dynamic/${limitId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      data: updates
    });
  }

  async adjustLimitAutomatically(limitId: string, reason: string): Promise<DynamicRiskLimit> {
    return this.apiCall<DynamicRiskLimit>(`/risk/limits/dynamic/${limitId}/adjust`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { reason }
    });
  }

  // ML-Based Breach Detection
  async getBreachPredictions(portfolioId: string, timeHorizon: '15m' | '30m' | '1h' | '4h' | '24h' = '1h'): Promise<BreachPrediction[]> {
    return this.apiCall<BreachPrediction[]>(`/risk/breaches/predict/${portfolioId}?horizon=${timeHorizon}`);
  }

  async enableBreachPrediction(portfolioId: string, config: {
    update_frequency_minutes?: number;
    confidence_threshold?: number;
    prediction_horizon_hours?: number;
  }): Promise<void> {
    return this.apiCall<void>(`/risk/breaches/predict/${portfolioId}/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: config
    });
  }

  async disableBreachPrediction(portfolioId: string): Promise<void> {
    return this.apiCall<void>(`/risk/breaches/predict/${portfolioId}/disable`, {
      method: 'POST'
    });
  }

  // Real-Time Risk Monitoring
  async startRealTimeMonitoring(portfolioId: string, options?: {
    update_frequency_seconds?: number;
    enable_alerts?: boolean;
    enable_auto_actions?: boolean;
  }): Promise<void> {
    return this.apiCall<void>(`/risk/monitor/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { portfolio_id: portfolioId, ...options }
    });
  }

  async stopRealTimeMonitoring(portfolioId: string): Promise<void> {
    return this.apiCall<void>(`/risk/monitor/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { portfolio_id: portfolioId }
    });
  }

  async getRealTimeMetrics(portfolioId: string): Promise<RealTimeRiskMetrics> {
    return this.apiCall<RealTimeRiskMetrics>(`/risk/monitor/${portfolioId}/metrics`);
  }

  async getMonitoringStatus(portfolioId: string): Promise<{
    monitoring_active: boolean;
    last_update: Date;
    update_frequency_seconds: number;
    alert_count_24h: number;
    breach_count_24h: number;
    performance_metrics: RiskComponentPerformance[];
  }> {
    return this.apiCall(`/risk/monitor/${portfolioId}/status`);
  }

  // Multi-Format Risk Reporting
  async requestRiskReport(request: RiskReportRequest): Promise<RiskReport> {
    return this.apiCall<RiskReport>('/risk/reports/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: request
    });
  }

  async getRiskReports(portfolioId: string, status?: 'generating' | 'completed' | 'failed'): Promise<RiskReport[]> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    return this.apiCall<RiskReport[]>(`/risk/reports/${portfolioId}?${params}`);
  }

  async getRiskReport(reportId: string): Promise<RiskReport> {
    return this.apiCall<RiskReport>(`/risk/reports/report/${reportId}`);
  }

  async downloadRiskReport(reportId: string): Promise<Blob> {
    const response = await axios.get(`${API_BASE}/risk/reports/report/${reportId}/download`, {
      responseType: 'blob'
    });
    return response.data;
  }

  async cancelRiskReport(reportId: string): Promise<void> {
    return this.apiCall<void>(`/risk/reports/report/${reportId}/cancel`, {
      method: 'DELETE'
    });
  }

  // Enhanced Alert Management
  async getEnhancedAlerts(portfolioId: string): Promise<RiskAlertSprint3[]> {
    return this.apiCall<RiskAlertSprint3[]>(`/risk/alerts/enhanced/${portfolioId}`);
  }

  async acknowledgeEnhancedAlert(alertId: string, response: {
    acknowledged_by: string;
    acknowledgment_note?: string;
    action_taken?: string;
  }): Promise<RiskAlertSprint3> {
    return this.apiCall<RiskAlertSprint3>(`/risk/alerts/enhanced/${alertId}/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: response
    });
  }

  async escalateAlert(alertId: string, escalation: {
    escalated_by: string;
    escalation_reason: string;
    override_delay?: boolean;
  }): Promise<RiskAlertSprint3> {
    return this.apiCall<RiskAlertSprint3>(`/risk/alerts/enhanced/${alertId}/escalate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: escalation
    });
  }

  async resolveAlert(alertId: string, resolution: {
    resolved_by: string;
    resolution_note: string;
    resolution_actions: string[];
  }): Promise<RiskAlertSprint3> {
    return this.apiCall<RiskAlertSprint3>(`/risk/alerts/enhanced/${alertId}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: resolution
    });
  }

  // Risk Configuration Management
  async getRiskConfiguration(portfolioId: string): Promise<RiskConfigurationSprint3> {
    return this.apiCall<RiskConfigurationSprint3>(`/risk/config/enhanced/${portfolioId}`);
  }

  async updateRiskConfiguration(portfolioId: string, config: Partial<RiskConfigurationSprint3>): Promise<RiskConfigurationSprint3> {
    return this.apiCall<RiskConfigurationSprint3>(`/risk/config/enhanced/${portfolioId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      data: config
    });
  }

  async getAvailableRiskLimitTypes(): Promise<RiskLimitType[]> {
    return this.apiCall<RiskLimitType[]>('/risk/limits/types');
  }

  // Limit Validation and Testing
  async validateRiskLimit(portfolioId: string, limit: Partial<RiskLimit>): Promise<{
    valid: boolean;
    warnings: string[];
    errors: string[];
    recommended_adjustments: string[];
  }> {
    return this.apiCall('/risk/limits/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { portfolio_id: portfolioId, limit }
    });
  }

  async testRiskLimitImpact(portfolioId: string, limitId: string, testScenarios: string[]): Promise<{
    test_results: Array<{
      scenario: string;
      would_trigger: boolean;
      estimated_impact: string;
      recommended_action: string;
    }>;
  }> {
    return this.apiCall(`/risk/limits/${limitId}/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { portfolio_id: portfolioId, scenarios: testScenarios }
    });
  }

  // System Performance and Health
  async getRiskSystemHealth(): Promise<{
    overall_status: 'healthy' | 'degraded' | 'unhealthy';
    components: RiskComponentPerformance[];
    active_monitoring_sessions: number;
    total_limits_monitored: number;
    alerts_processed_24h: number;
    ml_model_accuracy: number;
    last_health_check: Date;
  }> {
    return this.apiCall('/risk/system/health');
  }

  async getRiskSystemMetrics(timeRange: '1h' | '24h' | '7d' | '30d' = '24h'): Promise<{
    calculation_latency_p95: number;
    calculation_latency_p99: number;
    throughput_calculations_per_second: number;
    error_rate: number;
    cache_hit_rate: number;
    memory_usage_mb: number;
    active_websocket_connections: number;
  }> {
    return this.apiCall(`/risk/system/metrics?range=${timeRange}`);
  }
}

export const riskService = new RiskService();
export default riskService;