/**
 * useAdvancedStrategyTesting Hook
 * Sprint 3: Comprehensive Strategy Testing Framework
 * 
 * Advanced testing capabilities with backtesting, paper trading, stress testing,
 * Monte Carlo simulations, and automated validation pipelines.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketStream } from '../useWebSocketStream';

export type TestType = 
  | 'backtest'
  | 'paper_trading' 
  | 'stress_test'
  | 'monte_carlo'
  | 'walk_forward'
  | 'sensitivity'
  | 'regime_test'
  | 'compliance'
  | 'risk_validation';

export type TestStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TestConfiguration {
  id: string;
  name: string;
  type: TestType;
  description: string;
  
  // Test parameters
  parameters: {
    startDate?: string;
    endDate?: string;
    initialCapital?: number;
    benchmark?: string;
    commission?: number;
    slippage?: number;
    
    // Monte Carlo specific
    simulations?: number;
    confidence?: number;
    
    // Stress test specific
    scenarios?: StressScenario[];
    
    // Paper trading specific
    duration?: number; // days
    realTime?: boolean;
    
    // Walk forward specific
    trainingPeriod?: number; // days
    testingPeriod?: number; // days
    stepSize?: number; // days
  };
  
  // Validation criteria
  criteria: {
    minSharpe?: number;
    maxDrawdown?: number;
    minWinRate?: number;
    minProfitFactor?: number;
    maxVaR?: number;
    custom?: CustomCriteria[];
  };
  
  // Runtime configuration
  timeout: number; // minutes
  priority: 'low' | 'medium' | 'high';
  parallel: boolean;
}

export interface StressScenario {
  id: string;
  name: string;
  description: string;
  shocks: {
    asset: string;
    type: 'price' | 'volatility' | 'correlation';
    magnitude: number; // percentage
  }[];
}

export interface CustomCriteria {
  id: string;
  name: string;
  expression: string;
  threshold: number;
  operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
}

export interface TestResult {
  id: string;
  configId: string;
  strategyId: string;
  
  // Execution details
  status: TestStatus;
  startedAt: string;
  completedAt?: string;
  duration?: number;
  progress: number;
  
  // Performance metrics
  metrics: {
    totalReturn: number;
    annualizedReturn: number;
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxDrawdown: number;
    volatility: number;
    
    // Trade statistics
    totalTrades: number;
    winRate: number;
    profitFactor: number;
    averageTrade: number;
    bestTrade: number;
    worstTrade: number;
    
    // Risk metrics
    var95: number;
    cvar95: number;
    beta: number;
    alpha: number;
    informationRatio: number;
    trackingError: number;
    
    // Custom metrics
    customMetrics?: Record<string, number>;
  };
  
  // Detailed analysis
  analysis: {
    equityCurve: { date: string; equity: number; drawdown: number }[];
    monthlyReturns: { month: string; return: number }[];
    rollingMetrics: { date: string; sharpe: number; drawdown: number }[];
    tradeAnalysis: TradeAnalysis;
    riskAnalysis: RiskAnalysis;
  };
  
  // Test-specific results
  typeSpecificResults: {
    // Monte Carlo results
    monteCarlo?: {
      confidenceIntervals: {
        returns: { lower: number; upper: number };
        sharpe: { lower: number; upper: number };
        drawdown: { lower: number; upper: number };
      };
      probabilityOfLoss: number;
      worstCaseScenario: number;
      bestCaseScenario: number;
    };
    
    // Stress test results
    stressTest?: {
      scenarios: {
        scenarioId: string;
        impact: number;
        recovery: number;
        maxLoss: number;
      }[];
      overallRiskScore: number;
    };
    
    // Walk forward results
    walkForward?: {
      periods: {
        period: string;
        inSampleReturn: number;
        outOfSampleReturn: number;
        degradation: number;
      }[];
      averageDegradation: number;
      consistency: number;
    };
  };
  
  // Validation results
  validation: {
    passed: boolean;
    failedCriteria: string[];
    score: number; // 0-100
    recommendations: string[];
  };
  
  // Artifacts
  artifacts: {
    reports: string[];
    charts: string[];
    data: string[];
    logs: string[];
  };
  
  // Metadata
  createdBy: string;
  version: string;
  tags: string[];
}

export interface TradeAnalysis {
  trades: {
    entryDate: string;
    exitDate: string;
    symbol: string;
    side: 'long' | 'short';
    quantity: number;
    entryPrice: number;
    exitPrice: number;
    pnl: number;
    duration: number;
    slippage: number;
    commission: number;
  }[];
  
  distribution: {
    winningTrades: number;
    losingTrades: number;
    averageWin: number;
    averageLoss: number;
    largestWin: number;
    largestLoss: number;
    consecutiveWins: number;
    consecutiveLosses: number;
  };
  
  timing: {
    averageHoldingTime: number;
    holdingTimeDistribution: { range: string; count: number }[];
    entryTimeAnalysis: { hour: number; avgReturn: number }[];
    exitTimeAnalysis: { hour: number; avgReturn: number }[];
  };
}

export interface RiskAnalysis {
  drawdownAnalysis: {
    maxDrawdown: number;
    averageDrawdown: number;
    drawdownDuration: number;
    recovery: number;
    underWaterPeriods: { start: string; end: string; magnitude: number }[];
  };
  
  volatilityAnalysis: {
    dailyVolatility: number;
    weeklyVolatility: number;
    monthlyVolatility: number;
    volatilityRegimes: { period: string; regime: 'low' | 'medium' | 'high'; return: number }[];
  };
  
  correlationAnalysis: {
    benchmark: number;
    market: number;
    sectors: Record<string, number>;
  };
  
  tailRiskAnalysis: {
    skewness: number;
    kurtosis: number;
    leftTailEvents: number;
    rightTailEvents: number;
    extremeReturns: { date: string; return: number }[];
  };
}

export interface TestSuite {
  id: string;
  name: string;
  description: string;
  strategyId: string;
  
  // Suite configuration
  tests: TestConfiguration[];
  runInParallel: boolean;
  continueOnFailure: boolean;
  
  // Execution state
  status: TestStatus;
  currentTestIndex: number;
  totalTests: number;
  progress: number;
  startedAt?: string;
  completedAt?: string;
  
  // Results
  results: TestResult[];
  summary: {
    passed: number;
    failed: number;
    cancelled: number;
    overallScore: number;
    recommendation: 'approve' | 'reject' | 'conditional';
    conditions?: string[];
  };
}

export interface TestTemplate {
  id: string;
  name: string;
  description: string;
  category: 'equity' | 'fixed_income' | 'fx' | 'crypto' | 'multi_asset';
  tests: Omit<TestConfiguration, 'id'>[];
  defaultCriteria: TestConfiguration['criteria'];
}

export interface UseAdvancedStrategyTestingOptions {
  strategyId?: string;
  enableRealTime?: boolean;
  autoSave?: boolean;
  maxConcurrentTests?: number;
  defaultTimeout?: number;
}

export interface UseAdvancedStrategyTestingReturn {
  // Test data
  testResults: TestResult[];
  testSuites: TestSuite[];
  templates: TestTemplate[];
  runningTests: TestResult[];
  
  // Current test state
  currentTest: TestResult | null;
  currentSuite: TestSuite | null;
  
  // Status
  isLoading: boolean;
  isTesting: boolean;
  error: string | null;
  
  // Test execution
  runTest: (config: TestConfiguration) => Promise<string>;
  runTestSuite: (suite: Omit<TestSuite, 'id' | 'status' | 'progress' | 'results' | 'summary'>) => Promise<string>;
  runTestFromTemplate: (templateId: string, strategyId: string, parameters?: Record<string, any>) => Promise<string>;
  
  // Test management
  cancelTest: (testId: string) => Promise<void>;
  retryTest: (testId: string) => Promise<void>;
  pauseTest: (testId: string) => Promise<void>;
  resumeTest: (testId: string) => Promise<void>;
  
  // Suite management
  createSuite: (suite: Omit<TestSuite, 'id' | 'status' | 'progress' | 'results' | 'summary'>) => Promise<string>;
  runSuite: (suiteId: string) => Promise<void>;
  cancelSuite: (suiteId: string) => Promise<void>;
  
  // Configuration
  saveTestConfig: (config: TestConfiguration) => Promise<string>;
  loadTestConfig: (configId: string) => Promise<TestConfiguration>;
  validateConfig: (config: TestConfiguration) => { valid: boolean; errors: string[] };
  
  // Templates
  createTemplate: (template: Omit<TestTemplate, 'id'>) => Promise<string>;
  applyTemplate: (templateId: string, strategyId: string, overrides?: Partial<TestConfiguration>) => Promise<TestConfiguration[]>;
  
  // Analysis and comparison
  compareResults: (resultIds: string[]) => {
    comparison: Record<string, { values: number[]; best: string; worst: string }>;
    ranking: { resultId: string; score: number; rank: number }[];
  };
  
  analyzePerformance: (resultId: string) => Promise<{
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
    optimizationSuggestions: string[];
  }>;
  
  // Optimization
  optimizeParameters: (configId: string, parameters: string[], ranges: Record<string, [number, number]>) => Promise<{
    optimalParameters: Record<string, number>;
    improvementScore: number;
    validationResults: TestResult;
  }>;
  
  // Reporting
  generateReport: (resultId: string, format: 'json' | 'pdf' | 'html') => Promise<string | Blob>;
  generateSuiteReport: (suiteId: string, format: 'json' | 'pdf' | 'html') => Promise<string | Blob>;
  exportResults: (resultIds: string[], format: 'json' | 'csv') => Promise<string>;
  
  // Historical data
  getTestHistory: (strategyId?: string) => Promise<TestResult[]>;
  getPerformanceTrends: (strategyId: string, metric: string) => Promise<{ date: string; value: number }[]>;
  
  // Utilities
  refresh: () => Promise<void>;
  reset: () => void;
}

const DEFAULT_OPTIONS: Required<UseAdvancedStrategyTestingOptions> = {
  strategyId: '',
  enableRealTime: true,
  autoSave: true,
  maxConcurrentTests: 3,
  defaultTimeout: 60 // minutes
};

export function useAdvancedStrategyTesting(
  options: UseAdvancedStrategyTestingOptions = {}
): UseAdvancedStrategyTestingReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [templates, setTemplates] = useState<TestTemplate[]>([]);
  const [currentTest, setCurrentTest] = useState<TestResult | null>(null);
  const [currentSuite, setCurrentSuite] = useState<TestSuite | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Refs
  const testProgressRef = useRef<Map<string, number>>(new Map());
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // WebSocket stream for real-time test updates
  const {
    isActive: isRealTimeActive,
    latestMessage,
    startStream,
    stopStream,
    error: streamError
  } = useWebSocketStream({
    streamId: 'strategy_testing',
    messageType: 'strategy_performance',
    bufferSize: 500,
    autoSubscribe: config.enableRealTime
  });
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);
  
  // Process real-time test updates
  useEffect(() => {
    if (latestMessage && latestMessage.data) {
      const updateData = latestMessage.data;
      
      if (updateData.type === 'test_update' && updateData.test_id) {
        const testId = updateData.test_id;
        
        setTestResults(prev => prev.map(result => {
          if (result.id === testId) {
            return {
              ...result,
              status: updateData.status || result.status,
              progress: updateData.progress || result.progress,
              metrics: updateData.metrics ? { ...result.metrics, ...updateData.metrics } : result.metrics,
              completedAt: updateData.status === 'completed' ? new Date().toISOString() : result.completedAt,
              duration: updateData.duration || result.duration
            };
          }
          return result;
        }));
        
        // Update current test if it matches
        if (currentTest?.id === testId) {
          setCurrentTest(prev => prev ? {
            ...prev,
            status: updateData.status || prev.status,
            progress: updateData.progress || prev.progress,
            metrics: updateData.metrics ? { ...prev.metrics, ...updateData.metrics } : prev.metrics
          } : null);
        }
        
        // Update progress tracking
        if (updateData.progress !== undefined) {
          testProgressRef.current.set(testId, updateData.progress);
        }
      }
      
      // Handle suite updates
      if (updateData.type === 'suite_update' && updateData.suite_id) {
        const suiteId = updateData.suite_id;
        
        setTestSuites(prev => prev.map(suite => {
          if (suite.id === suiteId) {
            return {
              ...suite,
              status: updateData.status || suite.status,
              progress: updateData.progress || suite.progress,
              currentTestIndex: updateData.current_test_index || suite.currentTestIndex
            };
          }
          return suite;
        }));
      }
    }
  }, [latestMessage, currentTest]);
  
  // Utility functions
  const generateId = useCallback((prefix: string) => {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Test execution
  const runTest = useCallback(async (testConfig: TestConfiguration): Promise<string> => {
    setIsTesting(true);
    
    try {
      const testId = generateId('test');
      
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/tests`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: testId,
          config: testConfig,
          strategy_id: config.strategyId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Test execution failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      const newTestResult: TestResult = {
        id: testId,
        configId: testConfig.id,
        strategyId: config.strategyId,
        status: 'running',
        startedAt: new Date().toISOString(),
        progress: 0,
        metrics: {
          totalReturn: 0,
          annualizedReturn: 0,
          sharpeRatio: 0,
          sortinoRatio: 0,
          calmarRatio: 0,
          maxDrawdown: 0,
          volatility: 0,
          totalTrades: 0,
          winRate: 0,
          profitFactor: 0,
          averageTrade: 0,
          bestTrade: 0,
          worstTrade: 0,
          var95: 0,
          cvar95: 0,
          beta: 0,
          alpha: 0,
          informationRatio: 0,
          trackingError: 0
        },
        analysis: {
          equityCurve: [],
          monthlyReturns: [],
          rollingMetrics: [],
          tradeAnalysis: {
            trades: [],
            distribution: {
              winningTrades: 0,
              losingTrades: 0,
              averageWin: 0,
              averageLoss: 0,
              largestWin: 0,
              largestLoss: 0,
              consecutiveWins: 0,
              consecutiveLosses: 0
            },
            timing: {
              averageHoldingTime: 0,
              holdingTimeDistribution: [],
              entryTimeAnalysis: [],
              exitTimeAnalysis: []
            }
          },
          riskAnalysis: {
            drawdownAnalysis: {
              maxDrawdown: 0,
              averageDrawdown: 0,
              drawdownDuration: 0,
              recovery: 0,
              underWaterPeriods: []
            },
            volatilityAnalysis: {
              dailyVolatility: 0,
              weeklyVolatility: 0,
              monthlyVolatility: 0,
              volatilityRegimes: []
            },
            correlationAnalysis: {
              benchmark: 0,
              market: 0,
              sectors: {}
            },
            tailRiskAnalysis: {
              skewness: 0,
              kurtosis: 0,
              leftTailEvents: 0,
              rightTailEvents: 0,
              extremeReturns: []
            }
          }
        },
        typeSpecificResults: {},
        validation: {
          passed: false,
          failedCriteria: [],
          score: 0,
          recommendations: []
        },
        artifacts: {
          reports: [],
          charts: [],
          data: [],
          logs: []
        },
        createdBy: 'current_user',
        version: '1.0.0',
        tags: []
      };
      
      setTestResults(prev => [newTestResult, ...prev]);
      setCurrentTest(newTestResult);
      
      return testId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test execution failed');
      throw err;
    } finally {
      setIsTesting(false);
    }
  }, [generateId, API_BASE_URL, config.strategyId]);
  
  const runTestSuite = useCallback(async (suiteConfig: Omit<TestSuite, 'id' | 'status' | 'progress' | 'results' | 'summary'>): Promise<string> => {
    const suiteId = generateId('suite');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/suites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: suiteId,
          ...suiteConfig
        })
      });
      
      if (!response.ok) {
        throw new Error(`Test suite execution failed: ${response.statusText}`);
      }
      
      const newSuite: TestSuite = {
        id: suiteId,
        ...suiteConfig,
        status: 'running',
        currentTestIndex: 0,
        totalTests: suiteConfig.tests.length,
        progress: 0,
        startedAt: new Date().toISOString(),
        results: [],
        summary: {
          passed: 0,
          failed: 0,
          cancelled: 0,
          overallScore: 0,
          recommendation: 'approve'
        }
      };
      
      setTestSuites(prev => [newSuite, ...prev]);
      setCurrentSuite(newSuite);
      
      return suiteId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test suite execution failed');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const runTestFromTemplate = useCallback(async (templateId: string, strategyId: string, parameters?: Record<string, any>): Promise<string> => {
    const template = templates.find(t => t.id === templateId);
    if (!template) {
      throw new Error('Template not found');
    }
    
    // Apply template with parameter overrides
    const testConfigs = template.tests.map(test => ({
      ...test,
      id: generateId('config'),
      parameters: { ...test.parameters, ...parameters }
    }));
    
    // Create and run suite
    return runTestSuite({
      name: `${template.name} - ${strategyId}`,
      description: `Automated test from template: ${template.description}`,
      strategyId,
      tests: testConfigs,
      runInParallel: false,
      continueOnFailure: true
    });
  }, [templates, generateId, runTestSuite]);
  
  // Test management
  const cancelTest = useCallback(async (testId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/tests/${testId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Test cancellation failed: ${response.statusText}`);
      }
      
      setTestResults(prev => prev.map(test => 
        test.id === testId ? { ...test, status: 'cancelled', completedAt: new Date().toISOString() } : test
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test cancellation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const retryTest = useCallback(async (testId: string): Promise<void> => {
    const existingTest = testResults.find(t => t.id === testId);
    if (!existingTest) {
      throw new Error('Test not found');
    }
    
    // Find the original configuration and run again
    const config = await loadTestConfig(existingTest.configId);
    await runTest(config);
  }, [testResults, runTest]);
  
  const pauseTest = useCallback(async (testId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/tests/${testId}/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Test pause failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test pause failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const resumeTest = useCallback(async (testId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/tests/${testId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Test resume failed: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test resume failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Suite management
  const createSuite = useCallback(async (suiteConfig: Omit<TestSuite, 'id' | 'status' | 'progress' | 'results' | 'summary'>): Promise<string> => {
    const suiteId = generateId('suite');
    
    const newSuite: TestSuite = {
      id: suiteId,
      ...suiteConfig,
      status: 'pending',
      currentTestIndex: 0,
      totalTests: suiteConfig.tests.length,
      progress: 0,
      results: [],
      summary: {
        passed: 0,
        failed: 0,
        cancelled: 0,
        overallScore: 0,
        recommendation: 'approve'
      }
    };
    
    setTestSuites(prev => [newSuite, ...prev]);
    
    if (config.autoSave) {
      try {
        await fetch(`${API_BASE_URL}/api/v1/testing/suites`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newSuite)
        });
      } catch (err) {
        console.warn('Failed to auto-save suite:', err);
      }
    }
    
    return suiteId;
  }, [generateId, config.autoSave, API_BASE_URL]);
  
  const runSuite = useCallback(async (suiteId: string): Promise<void> => {
    const suite = testSuites.find(s => s.id === suiteId);
    if (!suite) {
      throw new Error('Suite not found');
    }
    
    await runTestSuite(suite);
  }, [testSuites, runTestSuite]);
  
  const cancelSuite = useCallback(async (suiteId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/suites/${suiteId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Suite cancellation failed: ${response.statusText}`);
      }
      
      setTestSuites(prev => prev.map(suite => 
        suite.id === suiteId ? { ...suite, status: 'cancelled', completedAt: new Date().toISOString() } : suite
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Suite cancellation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Configuration management
  const saveTestConfig = useCallback(async (testConfig: TestConfiguration): Promise<string> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/configs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testConfig)
      });
      
      if (!response.ok) {
        throw new Error(`Configuration save failed: ${response.statusText}`);
      }
      
      return testConfig.id;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Configuration save failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const loadTestConfig = useCallback(async (configId: string): Promise<TestConfiguration> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/configs/${configId}`);
      
      if (!response.ok) {
        throw new Error(`Configuration load failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.config;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Configuration load failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const validateConfig = useCallback((testConfig: TestConfiguration): { valid: boolean; errors: string[] } => {
    const errors: string[] = [];
    
    // Basic validation
    if (!testConfig.name) errors.push('Test name is required');
    if (!testConfig.type) errors.push('Test type is required');
    
    // Type-specific validation
    switch (testConfig.type) {
      case 'backtest':
        if (!testConfig.parameters.startDate) errors.push('Start date is required for backtesting');
        if (!testConfig.parameters.endDate) errors.push('End date is required for backtesting');
        if (!testConfig.parameters.initialCapital) errors.push('Initial capital is required for backtesting');
        break;
      case 'monte_carlo':
        if (!testConfig.parameters.simulations || testConfig.parameters.simulations < 100) {
          errors.push('Monte Carlo requires at least 100 simulations');
        }
        break;
      case 'paper_trading':
        if (!testConfig.parameters.duration || testConfig.parameters.duration < 1) {
          errors.push('Paper trading requires duration of at least 1 day');
        }
        break;
    }
    
    // Criteria validation
    if (testConfig.criteria.minSharpe && testConfig.criteria.minSharpe < 0) {
      errors.push('Minimum Sharpe ratio cannot be negative');
    }
    
    return {
      valid: errors.length === 0,
      errors
    };
  }, []);
  
  // Template management
  const createTemplate = useCallback(async (template: Omit<TestTemplate, 'id'>): Promise<string> => {
    const templateId = generateId('template');
    
    try {
      const newTemplate: TestTemplate = { ...template, id: templateId };
      
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/templates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTemplate)
      });
      
      if (!response.ok) {
        throw new Error(`Template creation failed: ${response.statusText}`);
      }
      
      setTemplates(prev => [...prev, newTemplate]);
      return templateId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Template creation failed');
      throw err;
    }
  }, [generateId, API_BASE_URL]);
  
  const applyTemplate = useCallback(async (templateId: string, strategyId: string, overrides?: Partial<TestConfiguration>): Promise<TestConfiguration[]> => {
    const template = templates.find(t => t.id === templateId);
    if (!template) {
      throw new Error('Template not found');
    }
    
    return template.tests.map(test => ({
      ...test,
      id: generateId('config'),
      parameters: { ...test.parameters, ...overrides?.parameters },
      criteria: { ...template.defaultCriteria, ...test.criteria, ...overrides?.criteria }
    }));
  }, [templates, generateId]);
  
  // Analysis functions
  const compareResults = useCallback((resultIds: string[]) => {
    const results = testResults.filter(r => resultIds.includes(r.id));
    
    if (results.length < 2) {
      throw new Error('At least 2 results are required for comparison');
    }
    
    const metrics = ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate', 'profitFactor'];
    const comparison: Record<string, { values: number[]; best: string; worst: string }> = {};
    
    metrics.forEach(metric => {
      const values = results.map(r => r.metrics[metric as keyof typeof r.metrics] as number);
      const best = metric === 'maxDrawdown' ? Math.min(...values) : Math.max(...values);
      const worst = metric === 'maxDrawdown' ? Math.max(...values) : Math.min(...values);
      
      const bestIndex = values.indexOf(best);
      const worstIndex = values.indexOf(worst);
      
      comparison[metric] = {
        values,
        best: results[bestIndex].id,
        worst: results[worstIndex].id
      };
    });
    
    // Calculate overall ranking
    const ranking = results.map(result => {
      const scores = metrics.map(metric => {
        const values = comparison[metric].values;
        const value = result.metrics[metric as keyof typeof result.metrics] as number;
        const max = Math.max(...values);
        const min = Math.min(...values);
        
        // Normalize score (0-1) - higher is better except for maxDrawdown
        const normalized = metric === 'maxDrawdown' 
          ? 1 - (value - min) / (max - min)
          : (value - min) / (max - min);
        
        return isNaN(normalized) ? 0.5 : normalized;
      });
      
      const score = scores.reduce((sum, s) => sum + s, 0) / scores.length * 100;
      return { resultId: result.id, score };
    });
    
    ranking.sort((a, b) => b.score - a.score);
    ranking.forEach((item, index) => {
      item.rank = index + 1;
    });
    
    return { comparison, ranking };
  }, [testResults]);
  
  const analyzePerformance = useCallback(async (resultId: string) => {
    const result = testResults.find(r => r.id === resultId);
    if (!result) {
      throw new Error('Test result not found');
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/results/${resultId}/analyze`);
      
      if (!response.ok) {
        throw new Error(`Performance analysis failed: ${response.statusText}`);
      }
      
      const analysis = await response.json();
      
      return {
        strengths: analysis.strengths || [],
        weaknesses: analysis.weaknesses || [],
        recommendations: analysis.recommendations || [],
        optimizationSuggestions: analysis.optimization_suggestions || []
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Performance analysis failed');
      throw err;
    }
  }, [testResults, API_BASE_URL]);
  
  // Optimization
  const optimizeParameters = useCallback(async (configId: string, parameters: string[], ranges: Record<string, [number, number]>) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_id: configId,
          parameters,
          ranges,
          strategy_id: config.strategyId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Parameter optimization failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      return {
        optimalParameters: result.optimal_parameters,
        improvementScore: result.improvement_score,
        validationResults: result.validation_results
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Parameter optimization failed');
      throw err;
    }
  }, [API_BASE_URL, config.strategyId]);
  
  // Reporting functions
  const generateReport = useCallback(async (resultId: string, format: 'json' | 'pdf' | 'html'): Promise<string | Blob> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/results/${resultId}/report?format=${format}`);
      
      if (!response.ok) {
        throw new Error(`Report generation failed: ${response.statusText}`);
      }
      
      if (format === 'json') {
        const data = await response.json();
        return JSON.stringify(data, null, 2);
      } else {
        return await response.blob();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const generateSuiteReport = useCallback(async (suiteId: string, format: 'json' | 'pdf' | 'html'): Promise<string | Blob> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/suites/${suiteId}/report?format=${format}`);
      
      if (!response.ok) {
        throw new Error(`Suite report generation failed: ${response.statusText}`);
      }
      
      if (format === 'json') {
        const data = await response.json();
        return JSON.stringify(data, null, 2);
      } else {
        return await response.blob();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Suite report generation failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const exportResults = useCallback(async (resultIds: string[], format: 'json' | 'csv'): Promise<string> => {
    const results = testResults.filter(r => resultIds.includes(r.id));
    
    if (format === 'json') {
      return JSON.stringify(results, null, 2);
    } else {
      const csvRows = [
        ['Test ID', 'Strategy ID', 'Type', 'Status', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
        ...results.map(r => [
          r.id, r.strategyId, r.configId, r.status, 
          r.metrics.totalReturn, r.metrics.sharpeRatio, 
          r.metrics.maxDrawdown, r.metrics.winRate
        ])
      ];
      return csvRows.map(row => row.join(',')).join('\n');
    }
  }, [testResults]);
  
  // Historical data
  const getTestHistory = useCallback(async (strategyId?: string): Promise<TestResult[]> => {
    try {
      const url = strategyId 
        ? `${API_BASE_URL}/api/v1/testing/results?strategy_id=${strategyId}`
        : `${API_BASE_URL}/api/v1/testing/results`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`History retrieval failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.results || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'History retrieval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  const getPerformanceTrends = useCallback(async (strategyId: string, metric: string): Promise<{ date: string; value: number }[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/testing/trends?strategy_id=${strategyId}&metric=${metric}`);
      
      if (!response.ok) {
        throw new Error(`Trend retrieval failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.trends || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Trend retrieval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Control functions
  const refresh = useCallback(async () => {
    setIsLoading(true);
    
    try {
      const [resultsResponse, suitesResponse, templatesResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/testing/results${config.strategyId ? `?strategy_id=${config.strategyId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/testing/suites${config.strategyId ? `?strategy_id=${config.strategyId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/testing/templates`)
      ]);
      
      if (resultsResponse.ok) {
        const resultsData = await resultsResponse.json();
        setTestResults(resultsData.results || []);
      }
      
      if (suitesResponse.ok) {
        const suitesData = await suitesResponse.json();
        setTestSuites(suitesData.suites || []);
      }
      
      if (templatesResponse.ok) {
        const templatesData = await templatesResponse.json();
        setTemplates(templatesData.templates || []);
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, config.strategyId]);
  
  const reset = useCallback(() => {
    setTestResults([]);
    setTestSuites([]);
    setTemplates([]);
    setCurrentTest(null);
    setCurrentSuite(null);
    setError(null);
    testProgressRef.current.clear();
  }, []);
  
  // Computed values
  const runningTests = testResults.filter(t => t.status === 'running' || t.status === 'pending');
  
  // Initial data load
  useEffect(() => {
    refresh();
  }, [refresh]);
  
  return {
    // Test data
    testResults,
    testSuites,
    templates,
    runningTests,
    
    // Current test state
    currentTest,
    currentSuite,
    
    // Status
    isLoading,
    isTesting,
    error: error || streamError,
    
    // Test execution
    runTest,
    runTestSuite,
    runTestFromTemplate,
    
    // Test management
    cancelTest,
    retryTest,
    pauseTest,
    resumeTest,
    
    // Suite management
    createSuite,
    runSuite,
    cancelSuite,
    
    // Configuration
    saveTestConfig,
    loadTestConfig,
    validateConfig,
    
    // Templates
    createTemplate,
    applyTemplate,
    
    // Analysis and comparison
    compareResults,
    analyzePerformance,
    
    // Optimization
    optimizeParameters,
    
    // Reporting
    generateReport,
    generateSuiteReport,
    exportResults,
    
    // Historical data
    getTestHistory,
    getPerformanceTrends,
    
    // Utilities
    refresh,
    reset
  };
}

export default useAdvancedStrategyTesting;