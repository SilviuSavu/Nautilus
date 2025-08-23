/**
 * Strategy Testing Hook
 * Manages automated testing and validation for strategies
 */

import { useState, useCallback, useEffect } from 'react';
import { message } from 'antd';

export interface TestConfig {
  backtest?: {
    durationDays: number;
    startDate?: string;
    endDate?: string;
    initialCapital: number;
    instruments: string[];
  };
  paperTrading?: {
    durationMinutes: number;
    maxPositions: number;
    riskLimits: Record<string, number>;
  };
  stressTesting?: {
    scenarios: string[];
    volatilityMultiplier: number;
  };
  performanceTargets?: {
    minSharpeRatio: number;
    maxDrawdown: number;
    minWinRate: number;
  };
}

export interface TestResult {
  testId: string;
  testType: 'backtest' | 'paper_trading' | 'stress_test' | 'validation';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  score: number;
  metrics: Record<string, any>;
  errors: string[];
  warnings: string[];
  startedAt: Date;
  completedAt?: Date;
  duration?: number;
}

export interface TestSuite {
  suiteId: string;
  strategyId: string;
  version: string;
  testConfig: TestConfig;
  tests: TestResult[];
  overallScore: number;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'cancelled';
  startedAt: Date;
  completedAt?: Date;
  totalDuration?: number;
}

export interface BacktestResult {
  testId: string;
  summary: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
  };
  timeSeries: {
    timestamp: string;
    equity: number;
    drawdown: number;
    exposure: number;
  }[];
  trades: {
    entryTime: string;
    exitTime: string;
    instrument: string;
    side: 'long' | 'short';
    quantity: number;
    entryPrice: number;
    exitPrice: number;
    pnl: number;
    commission: number;
  }[];
  riskMetrics: {
    var95: number;
    var99: number;
    expectedShortfall: number;
    beta: number;
    alpha: number;
  };
}

export interface ValidationResult {
  testId: string;
  validations: {
    codeQuality: {
      score: number;
      issues: string[];
    };
    riskCompliance: {
      score: number;
      violations: string[];
    };
    performanceConstraints: {
      score: number;
      breaches: string[];
    };
    dependencyCheck: {
      score: number;
      missingDependencies: string[];
    };
  };
  overallScore: number;
  passed: boolean;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

export const useStrategyTesting = () => {
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [backtestResults, setBacktestResults] = useState<Record<string, BacktestResult>>({});
  const [validationResults, setValidationResults] = useState<Record<string, ValidationResult>>({});
  const [activeTests, setActiveTests] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch test suites
  const fetchTestSuites = useCallback(async (strategyId?: string) => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (strategyId) params.append('strategy_id', strategyId);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/suites?${params}`);
      const data = await response.json();
      
      setTestSuites(data.map((suite: any) => ({
        ...suite,
        startedAt: new Date(suite.startedAt),
        completedAt: suite.completedAt ? new Date(suite.completedAt) : undefined,
        tests: suite.tests.map((test: any) => ({
          ...test,
          startedAt: new Date(test.startedAt),
          completedAt: test.completedAt ? new Date(test.completedAt) : undefined
        }))
      })));
    } catch (err) {
      console.error('Failed to fetch test suites:', err);
      setError('Failed to fetch test suites');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch backtest result
  const fetchBacktestResult = useCallback(async (testId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/backtest/${testId}`);
      const data = await response.json();
      
      setBacktestResults(prev => ({
        ...prev,
        [testId]: data
      }));
      
      return data;
    } catch (err) {
      console.error('Failed to fetch backtest result:', err);
      setError('Failed to fetch backtest result');
      return null;
    }
  }, []);

  // Fetch validation result
  const fetchValidationResult = useCallback(async (testId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/validation/${testId}`);
      const data = await response.json();
      
      setValidationResults(prev => ({
        ...prev,
        [testId]: data
      }));
      
      return data;
    } catch (err) {
      console.error('Failed to fetch validation result:', err);
      setError('Failed to fetch validation result');
      return null;
    }
  }, []);

  // Start test suite
  const startTestSuite = useCallback(async (
    strategyId: string,
    version: string,
    testConfig: TestConfig
  ): Promise<TestSuite | null> => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/suites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_id: strategyId,
          version,
          test_config: testConfig
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start test suite: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Test suite started successfully`);
      
      await fetchTestSuites();
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined,
        tests: result.tests.map((test: any) => ({
          ...test,
          startedAt: new Date(test.startedAt),
          completedAt: test.completedAt ? new Date(test.completedAt) : undefined
        }))
      };
    } catch (err) {
      console.error('Failed to start test suite:', err);
      message.error(`Failed to start test suite: ${err}`);
      setError(`Failed to start test suite: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchTestSuites]);

  // Start backtest
  const startBacktest = useCallback(async (
    strategyCode: string,
    strategyConfig: Record<string, any>,
    backtestConfig: TestConfig['backtest']
  ): Promise<TestResult | null> => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_code: strategyCode,
          strategy_config: strategyConfig,
          backtest_config: backtestConfig
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start backtest: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Backtest started successfully`);
      
      setActiveTests(prev => new Set([...prev, result.testId]));
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined
      };
    } catch (err) {
      console.error('Failed to start backtest:', err);
      message.error(`Failed to start backtest: ${err}`);
      setError(`Failed to start backtest: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Start paper trading test
  const startPaperTradingTest = useCallback(async (
    strategyCode: string,
    strategyConfig: Record<string, any>,
    paperTradingConfig: TestConfig['paperTrading']
  ): Promise<TestResult | null> => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/paper-trading`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_code: strategyCode,
          strategy_config: strategyConfig,
          paper_trading_config: paperTradingConfig
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start paper trading test: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Paper trading test started successfully`);
      
      setActiveTests(prev => new Set([...prev, result.testId]));
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined
      };
    } catch (err) {
      console.error('Failed to start paper trading test:', err);
      message.error(`Failed to start paper trading test: ${err}`);
      setError(`Failed to start paper trading test: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Start validation test
  const startValidationTest = useCallback(async (
    strategyCode: string,
    strategyConfig: Record<string, any>
  ): Promise<ValidationResult | null> => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/validation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_code: strategyCode,
          strategy_config: strategyConfig
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to start validation test: ${response.statusText}`);
      }

      const result = await response.json();
      message.success(`Validation test completed successfully`);
      
      setValidationResults(prev => ({
        ...prev,
        [result.testId]: result
      }));
      
      return result;
    } catch (err) {
      console.error('Failed to start validation test:', err);
      message.error(`Failed to start validation test: ${err}`);
      setError(`Failed to start validation test: ${err}`);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Stop test
  const stopTest = useCallback(async (testId: string): Promise<boolean> => {
    try {
      setLoading(true);

      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/tests/${testId}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Failed to stop test: ${response.statusText}`);
      }

      message.success(`Test stopped successfully`);
      
      setActiveTests(prev => {
        const newSet = new Set(prev);
        newSet.delete(testId);
        return newSet;
      });
      
      await fetchTestSuites();
      
      return true;
    } catch (err) {
      console.error('Failed to stop test:', err);
      message.error(`Failed to stop test: ${err}`);
      setError(`Failed to stop test: ${err}`);
      return false;
    } finally {
      setLoading(false);
    }
  }, [fetchTestSuites]);

  // Get test status
  const getTestStatus = useCallback(async (testId: string): Promise<TestResult | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/tests/${testId}/status`);
      if (!response.ok) {
        throw new Error(`Failed to get test status: ${response.statusText}`);
      }

      const result = await response.json();
      
      return {
        ...result,
        startedAt: new Date(result.startedAt),
        completedAt: result.completedAt ? new Date(result.completedAt) : undefined
      };
    } catch (err) {
      console.error('Failed to get test status:', err);
      setError(`Failed to get test status: ${err}`);
      return null;
    }
  }, []);

  // Run quick validation
  const runQuickValidation = useCallback(async (
    strategyCode: string
  ): Promise<{ isValid: boolean; errors: string[]; warnings: string[] }> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategy/testing/quick-validation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy_code: strategyCode })
      });

      if (!response.ok) {
        throw new Error(`Failed to run quick validation: ${response.statusText}`);
      }

      return await response.json();
    } catch (err) {
      console.error('Failed to run quick validation:', err);
      setError(`Failed to run quick validation: ${err}`);
      return { isValid: false, errors: [String(err)], warnings: [] };
    }
  }, []);

  // Get test suite by ID
  const getTestSuite = useCallback((suiteId: string): TestSuite | null => {
    return testSuites.find(suite => suite.suiteId === suiteId) || null;
  }, [testSuites]);

  // Get backtest result by test ID
  const getBacktestResult = useCallback((testId: string): BacktestResult | null => {
    return backtestResults[testId] || null;
  }, [backtestResults]);

  // Get validation result by test ID
  const getValidationResult = useCallback((testId: string): ValidationResult | null => {
    return validationResults[testId] || null;
  }, [validationResults]);

  // Check if test is active
  const isTestActive = useCallback((testId: string): boolean => {
    return activeTests.has(testId);
  }, [activeTests]);

  // Initialize data
  useEffect(() => {
    fetchTestSuites();
  }, [fetchTestSuites]);

  return {
    // State
    testSuites,
    backtestResults,
    validationResults,
    activeTests,
    loading,
    error,

    // Actions
    startTestSuite,
    startBacktest,
    startPaperTradingTest,
    startValidationTest,
    stopTest,
    runQuickValidation,

    // Queries
    getTestSuite,
    getBacktestResult,
    getValidationResult,
    getTestStatus,
    isTestActive,

    // Data fetching
    fetchTestSuites,
    fetchBacktestResult,
    fetchValidationResult
  };
};