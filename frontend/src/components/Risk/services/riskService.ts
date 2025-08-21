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
  RiskChartData
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
}

export const riskService = new RiskService();
export default riskService;