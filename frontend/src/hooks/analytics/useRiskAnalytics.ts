/**
 * useRiskAnalytics Hook - Sprint 3 Integration
 * Risk analytics with VaR calculations, stress testing, and exposure analysis
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface RiskAnalyticsRequest {
  portfolio_id: string;
  start_date?: string;
  end_date?: string;
  confidence_levels?: number[];
  var_methods?: ('historical' | 'parametric' | 'monte_carlo')[];
  stress_scenarios?: StressScenario[];
  include_component_var?: boolean;
  include_marginal_var?: boolean;
  rolling_window?: number;
}

export interface StressScenario {
  name: string;
  description: string;
  market_shock: number;
  volatility_multiplier: number;
  correlation_adjustment: number;
  sector_specific?: Record<string, number>;
}

export interface RiskAnalyticsResult {
  portfolio_metrics: {
    portfolio_value: number;
    net_exposure: number;
    gross_exposure: number;
    leverage: number;
    beta: number;
    correlation_to_market: number;
  };
  
  var_analysis: {
    confidence_level: number;
    time_horizon_days: number;
    historical_var: number;
    parametric_var: number;
    monte_carlo_var: number;
    expected_shortfall: number;
    coherent_risk_measure: number;
  }[];
  
  stress_testing: {
    scenario_name: string;
    description: string;
    base_portfolio_value: number;
    stressed_portfolio_value: number;
    absolute_loss: number;
    percentage_loss: number;
    component_losses: Array<{
      asset: string;
      weight: number;
      base_value: number;
      stressed_value: number;
      loss_contribution: number;
    }>;
  }[];
  
  exposure_analysis: {
    sector_exposure: Record<string, {
      long_exposure: number;
      short_exposure: number;
      net_exposure: number;
      percentage_of_portfolio: number;
    }>;
    
    currency_exposure: Record<string, {
      exposure_amount: number;
      percentage_of_portfolio: number;
      hedged_amount: number;
      unhedged_amount: number;
    }>;
    
    geographic_exposure: Record<string, {
      exposure_amount: number;
      percentage_of_portfolio: number;
    }>;
  };
  
  component_var?: {
    asset: string;
    individual_var: number;
    marginal_var: number;
    component_var: number;
    percentage_contribution: number;
  }[];
  
  risk_decomposition: {
    systematic_risk: number;
    idiosyncratic_risk: number;
    concentration_risk: number;
    liquidity_risk_score: number;
    market_risk_score: number;
    credit_risk_score: number;
  };
  
  risk_metrics_time_series?: {
    date: string;
    var_95: number;
    var_99: number;
    expected_shortfall_95: number;
    beta: number;
    volatility: number;
    max_drawdown: number;
  }[];
  
  correlation_matrix: {
    assets: string[];
    correlation_data: number[][];
    eigenvalues: number[];
    condition_number: number;
  };
  
  scenario_analysis: {
    scenario_name: string;
    probability: number;
    expected_return: number;
    volatility: number;
    var_95: number;
    tail_risk: number;
  }[];
}

export interface UseRiskAnalyticsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  enableRealTimeMonitoring?: boolean;
  alertThresholds?: {
    var_95_threshold?: number;
    concentration_threshold?: number;
    leverage_threshold?: number;
  };
}

export interface UseRiskAnalyticsReturn {
  // State
  result: RiskAnalyticsResult | null;
  isAnalyzing: boolean;
  error: string | null;
  lastAnalyzed: Date | null;
  analysisDuration: number;
  
  // Risk monitoring
  riskAlerts: RiskAlert[];
  isMonitoring: boolean;
  
  // Actions
  analyzeRisk: (request: RiskAnalyticsRequest) => Promise<void>;
  analyzeRiskAsync: (request: RiskAnalyticsRequest) => Promise<RiskAnalyticsResult>;
  startMonitoring: (portfolioId: string) => void;
  stopMonitoring: () => void;
  
  // Risk calculations
  calculatePortfolioVar: (
    portfolioId: string,
    confidenceLevel: number,
    method: 'historical' | 'parametric' | 'monte_carlo'
  ) => Promise<{
    var_value: number;
    expected_shortfall: number;
    calculation_method: string;
    confidence_level: number;
  }>;
  
  runStressTest: (
    portfolioId: string,
    scenarios: StressScenario[]
  ) => Promise<{
    base_portfolio_value: number;
    stress_results: Array<{
      scenario: string;
      portfolio_value: number;
      loss_amount: number;
      loss_percentage: number;
    }>;
  }>;
  
  getExposureAnalysis: (
    portfolioId: string,
    exposureType: 'sector' | 'currency' | 'geographic' | 'all'
  ) => Promise<{
    total_exposure: number;
    net_exposure: number;
    gross_exposure: number;
    exposures: Record<string, {
      amount: number;
      percentage: number;
      risk_contribution: number;
    }>;
  }>;
  
  // Risk limit monitoring
  checkRiskLimits: (portfolioId: string) => Promise<{
    breaches: RiskLimitBreach[];
    warnings: RiskLimitWarning[];
    compliance_status: 'compliant' | 'warning' | 'breach';
  }>;
  
  // Advanced risk measures
  calculateCoherentRiskMeasures: (
    portfolioId: string,
    measures: ('cvar' | 'spectral' | 'distortion')[]
  ) => Promise<{
    measure_name: string;
    value: number;
    confidence_level: number;
    interpretation: string;
  }[]>;
}

interface RiskAlert {
  id: string;
  type: 'var_breach' | 'concentration' | 'leverage' | 'correlation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  portfolio_id: string;
  current_value: number;
  threshold_value: number;
}

interface RiskLimitBreach {
  limit_id: string;
  limit_name: string;
  current_value: number;
  threshold_value: number;
  breach_amount: number;
  severity: 'warning' | 'critical';
}

interface RiskLimitWarning {
  limit_id: string;
  limit_name: string;
  current_value: number;
  warning_threshold: number;
  threshold_value: number;
  distance_to_breach: number;
}

export function useRiskAnalytics(
  options: UseRiskAnalyticsOptions = {}
): UseRiskAnalyticsReturn {
  const { 
    autoRefresh = false, 
    refreshInterval = 60000, 
    enableRealTimeMonitoring = false,
    alertThresholds = {}
  } = options;
  
  // State
  const [result, setResult] = useState<RiskAnalyticsResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAnalyzed, setLastAnalyzed] = useState<Date | null>(null);
  const [analysisDuration, setAnalysisDuration] = useState(0);
  const [riskAlerts, setRiskAlerts] = useState<RiskAlert[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  
  // Refs
  const lastRequestRef = useRef<RiskAnalyticsRequest | null>(null);
  const monitoringIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);
  const websocketRef = useRef<WebSocket | null>(null);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // Cleanup
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopMonitoring();
    };
  }, []);
  
  // Main risk analysis function
  const analyzeRisk = useCallback(async (request: RiskAnalyticsRequest) => {
    if (!request.portfolio_id) {
      setError('Portfolio ID is required');
      return;
    }
    
    const startTime = performance.now();
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/sprint3/analytics/risk/analyze`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...request,
            confidence_levels: request.confidence_levels || [0.95, 0.99],
            var_methods: request.var_methods || ['historical', 'parametric', 'monte_carlo'],
            include_component_var: request.include_component_var ?? true,
            include_marginal_var: request.include_marginal_var ?? true,
            rolling_window: request.rolling_window || 252, // 1 year
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const riskResult = await response.json();
      const endTime = performance.now();
      
      if (isMountedRef.current) {
        setResult(riskResult);
        setLastAnalyzed(new Date());
        setAnalysisDuration(endTime - startTime);
        lastRequestRef.current = request;
        
        // Check for risk alerts
        checkForRiskAlerts(riskResult);
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Risk analysis failed');
        setResult(null);
      }
    } finally {
      if (isMountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  }, [API_BASE_URL]);
  
  // Async analysis that returns the result
  const analyzeRiskAsync = useCallback(async (request: RiskAnalyticsRequest): Promise<RiskAnalyticsResult> => {
    await analyzeRisk(request);
    
    if (error) {
      throw new Error(error);
    }
    
    if (!result) {
      throw new Error('Risk analysis failed to return result');
    }
    
    return result;
  }, [analyzeRisk, error, result]);
  
  // Check for risk alerts based on thresholds
  const checkForRiskAlerts = useCallback((riskResult: RiskAnalyticsResult) => {
    const newAlerts: RiskAlert[] = [];
    const now = new Date();
    
    // Check VaR thresholds
    if (alertThresholds.var_95_threshold) {
      const var95 = riskResult.var_analysis.find(v => v.confidence_level === 0.95);
      if (var95 && Math.abs(var95.historical_var) > alertThresholds.var_95_threshold) {
        newAlerts.push({
          id: `var-breach-${now.getTime()}`,
          type: 'var_breach',
          severity: Math.abs(var95.historical_var) > alertThresholds.var_95_threshold * 1.5 ? 'critical' : 'high',
          message: `95% VaR (${Math.abs(var95.historical_var).toFixed(2)}) exceeds threshold (${alertThresholds.var_95_threshold})`,
          timestamp: now,
          portfolio_id: lastRequestRef.current?.portfolio_id || '',
          current_value: Math.abs(var95.historical_var),
          threshold_value: alertThresholds.var_95_threshold,
        });
      }
    }
    
    // Check leverage threshold
    if (alertThresholds.leverage_threshold) {
      const leverage = riskResult.portfolio_metrics.leverage;
      if (leverage > alertThresholds.leverage_threshold) {
        newAlerts.push({
          id: `leverage-breach-${now.getTime()}`,
          type: 'leverage',
          severity: leverage > alertThresholds.leverage_threshold * 1.2 ? 'critical' : 'medium',
          message: `Portfolio leverage (${leverage.toFixed(2)}) exceeds threshold (${alertThresholds.leverage_threshold})`,
          timestamp: now,
          portfolio_id: lastRequestRef.current?.portfolio_id || '',
          current_value: leverage,
          threshold_value: alertThresholds.leverage_threshold,
        });
      }
    }
    
    // Check concentration risk
    if (alertThresholds.concentration_threshold) {
      Object.entries(riskResult.exposure_analysis.sector_exposure).forEach(([sector, exposure]) => {
        if (Math.abs(exposure.percentage_of_portfolio) > alertThresholds.concentration_threshold!) {
          newAlerts.push({
            id: `concentration-${sector}-${now.getTime()}`,
            type: 'concentration',
            severity: 'medium',
            message: `High concentration in ${sector}: ${exposure.percentage_of_portfolio.toFixed(1)}%`,
            timestamp: now,
            portfolio_id: lastRequestRef.current?.portfolio_id || '',
            current_value: Math.abs(exposure.percentage_of_portfolio),
            threshold_value: alertThresholds.concentration_threshold,
          });
        }
      });
    }
    
    setRiskAlerts(prev => [...prev, ...newAlerts]);
  }, [alertThresholds]);
  
  // Start real-time risk monitoring
  const startMonitoring = useCallback((portfolioId: string) => {
    if (isMonitoring) return;
    
    setIsMonitoring(true);
    
    if (enableRealTimeMonitoring) {
      // Set up WebSocket for real-time risk updates
      const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/risk/realtime/${portfolioId}`;
      
      websocketRef.current = new WebSocket(wsUrl);
      
      websocketRef.current.onmessage = (event) => {
        try {
          const riskUpdate = JSON.parse(event.data);
          if (isMountedRef.current) {
            setResult(prev => prev ? { ...prev, ...riskUpdate } : riskUpdate);
            checkForRiskAlerts(riskUpdate);
          }
        } catch (err) {
          console.error('Failed to parse risk update:', err);
        }
      };
      
      websocketRef.current.onerror = () => {
        console.error('Risk monitoring WebSocket error');
        setIsMonitoring(false);
      };
    }
    
    // Set up polling as backup or primary method
    monitoringIntervalRef.current = setInterval(() => {
      if (lastRequestRef.current) {
        analyzeRisk(lastRequestRef.current);
      }
    }, refreshInterval);
  }, [isMonitoring, enableRealTimeMonitoring, API_BASE_URL, refreshInterval, analyzeRisk, checkForRiskAlerts]);
  
  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
      monitoringIntervalRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
  }, []);
  
  // Calculate portfolio VaR
  const calculatePortfolioVar = useCallback(async (
    portfolioId: string,
    confidenceLevel: number,
    method: 'historical' | 'parametric' | 'monte_carlo'
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/risk/var`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          confidence_level: confidenceLevel,
          method: method,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Run stress test
  const runStressTest = useCallback(async (
    portfolioId: string,
    scenarios: StressScenario[]
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/risk/stress-test`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          scenarios: scenarios,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Get exposure analysis
  const getExposureAnalysis = useCallback(async (
    portfolioId: string,
    exposureType: 'sector' | 'currency' | 'geographic' | 'all'
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/risk/exposure`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          exposure_type: exposureType,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Check risk limits
  const checkRiskLimits = useCallback(async (portfolioId: string) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/risk/breaches/${portfolioId}`
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  // Calculate coherent risk measures
  const calculateCoherentRiskMeasures = useCallback(async (
    portfolioId: string,
    measures: ('cvar' | 'spectral' | 'distortion')[]
  ) => {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/sprint3/analytics/risk/coherent-measures`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          measures: measures,
        }),
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }, [API_BASE_URL]);
  
  return {
    // State
    result,
    isAnalyzing,
    error,
    lastAnalyzed,
    analysisDuration,
    
    // Risk monitoring
    riskAlerts,
    isMonitoring,
    
    // Actions
    analyzeRisk,
    analyzeRiskAsync,
    startMonitoring,
    stopMonitoring,
    
    // Risk calculations
    calculatePortfolioVar,
    runStressTest,
    getExposureAnalysis,
    
    // Risk limit monitoring
    checkRiskLimits,
    
    // Advanced risk measures
    calculateCoherentRiskMeasures,
  };
}

export default useRiskAnalytics;