/**
 * Portfolio Metrics Service for time-weighted return calculations and advanced metrics
 */

import { Position } from '../types/position';
import { PortfolioAggregation, StrategyPnL } from './portfolioAggregationService';

export interface TimeWeightedReturn {
  period: string;
  return_value: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
}

export interface MoneyWeightedReturn {
  period: string;
  irr: number; // Internal Rate of Return
  modified_dietz: number;
  cash_flows: CashFlow[];
}

export interface CashFlow {
  date: string;
  amount: number;
  type: 'deposit' | 'withdrawal' | 'dividend' | 'fee';
  description?: string;
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_duration: number;
  calmar_ratio: number;
  omega_ratio: number;
  var_95: number;
  cvar_95: number;
  beta: number;
  alpha: number;
  information_ratio: number;
  tracking_error: number;
  up_capture: number;
  down_capture: number;
  win_rate: number;
  profit_factor: number;
  recovery_factor: number;
  sterling_ratio: number;
}

export interface RollingMetrics {
  date: string;
  rolling_return: number;
  rolling_volatility: number;
  rolling_sharpe: number;
  rolling_max_drawdown: number;
  rolling_beta: number;
}

export interface BenchmarkComparison {
  portfolio_return: number;
  benchmark_return: number;
  active_return: number;
  tracking_error: number;
  information_ratio: number;
  up_capture_ratio: number;
  down_capture_ratio: number;
  correlation: number;
  beta: number;
  alpha: number;
}

export class PortfolioMetricsService {
  private cashFlows: CashFlow[] = [];
  private benchmarkReturns: number[] = [];
  private riskFreeRate: number = 0.02; // 2% annual risk-free rate

  constructor() {
    this.initializeBenchmarkData();
  }

  /**
   * Initialize mock benchmark data
   */
  private initializeBenchmarkData(): void {
    // Generate mock S&P 500 daily returns for the last year
    for (let i = 0; i < 252; i++) { // 252 trading days
      const randomReturn = (Math.random() - 0.5) * 0.04 + 0.0003; // ~7.5% annual with volatility
      this.benchmarkReturns.push(randomReturn);
    }
  }

  /**
   * Calculate time-weighted return
   */
  public calculateTimeWeightedReturn(
    portfolioData: PortfolioAggregation,
    historicalValues: number[],
    period: string = '1Y'
  ): TimeWeightedReturn {
    if (historicalValues.length < 2) {
      return {
        period,
        return_value: 0,
        annualized_return: 0,
        volatility: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        calmar_ratio: 0
      };
    }

    // Calculate returns
    const returns = this.calculateReturns(historicalValues);
    const totalReturn = (historicalValues[historicalValues.length - 1] / historicalValues[0]) - 1;
    
    // Annualize based on period
    const periodMultiplier = this.getPeriodMultiplier(period);
    const annualizedReturn = Math.pow(1 + totalReturn, periodMultiplier) - 1;
    
    // Calculate volatility
    const volatility = this.calculateVolatility(returns) * Math.sqrt(252); // Annualized
    
    // Calculate Sharpe ratio
    const excessReturn = annualizedReturn - this.riskFreeRate;
    const sharpeRatio = volatility > 0 ? excessReturn / volatility : 0;
    
    // Calculate maximum drawdown
    const maxDrawdown = this.calculateMaxDrawdown(historicalValues);
    
    // Calculate Calmar ratio
    const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;

    return {
      period,
      return_value: totalReturn,
      annualized_return: annualizedReturn,
      volatility,
      sharpe_ratio: sharpeRatio,
      max_drawdown: maxDrawdown,
      calmar_ratio: calmarRatio
    };
  }

  /**
   * Calculate money-weighted return (IRR)
   */
  public calculateMoneyWeightedReturn(
    portfolioData: PortfolioAggregation,
    period: string = '1Y'
  ): MoneyWeightedReturn {
    const cashFlows = this.getCashFlowsForPeriod(period);
    
    // Simple IRR calculation (Newton-Raphson method would be more accurate)
    const irr = this.calculateIRR(cashFlows, portfolioData.total_pnl);
    
    // Modified Dietz return
    const modifiedDietz = this.calculateModifiedDietz(cashFlows, portfolioData.total_pnl);

    return {
      period,
      irr,
      modified_dietz: modifiedDietz,
      cash_flows: cashFlows
    };
  }

  /**
   * Calculate comprehensive performance metrics
   */
  public calculatePerformanceMetrics(
    portfolioData: PortfolioAggregation,
    historicalValues: number[],
    period: string = '1Y'
  ): PerformanceMetrics {
    if (historicalValues.length < 2) {
      return this.getEmptyMetrics();
    }

    const returns = this.calculateReturns(historicalValues);
    const totalReturn = (historicalValues[historicalValues.length - 1] / historicalValues[0]) - 1;
    const annualizedReturn = Math.pow(1 + totalReturn, this.getPeriodMultiplier(period)) - 1;
    
    const volatility = this.calculateVolatility(returns) * Math.sqrt(252);
    const downVolatility = this.calculateDownsideVolatility(returns) * Math.sqrt(252);
    
    const sharpeRatio = volatility > 0 ? (annualizedReturn - this.riskFreeRate) / volatility : 0;
    const sortinoRatio = downVolatility > 0 ? (annualizedReturn - this.riskFreeRate) / downVolatility : 0;
    
    const maxDrawdown = this.calculateMaxDrawdown(historicalValues);
    const maxDrawdownDuration = this.calculateMaxDrawdownDuration(historicalValues);
    
    const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
    const omegaRatio = this.calculateOmegaRatio(returns);
    
    const var95 = this.calculateVaR(returns, 0.95);
    const cvar95 = this.calculateCVaR(returns, 0.95);
    
    const benchmarkComparison = this.calculateBenchmarkComparison(returns);
    
    const winningTrades = returns.filter(r => r > 0).length;
    const winRate = returns.length > 0 ? winningTrades / returns.length : 0;
    
    const profitFactor = this.calculateProfitFactor(returns);
    const recoveryFactor = maxDrawdown > 0 ? totalReturn / maxDrawdown : 0;
    const sterlingRatio = this.calculateSterlingRatio(annualizedReturn, historicalValues);

    return {
      total_return: totalReturn,
      annualized_return: annualizedReturn,
      volatility,
      sharpe_ratio: sharpeRatio,
      sortino_ratio: sortinoRatio,
      max_drawdown: maxDrawdown,
      max_drawdown_duration: maxDrawdownDuration,
      calmar_ratio: calmarRatio,
      omega_ratio: omegaRatio,
      var_95: var95,
      cvar_95: cvar95,
      beta: benchmarkComparison.beta,
      alpha: benchmarkComparison.alpha,
      information_ratio: benchmarkComparison.information_ratio,
      tracking_error: benchmarkComparison.tracking_error,
      up_capture: benchmarkComparison.up_capture_ratio,
      down_capture: benchmarkComparison.down_capture_ratio,
      win_rate: winRate,
      profit_factor: profitFactor,
      recovery_factor: recoveryFactor,
      sterling_ratio: sterlingRatio
    };
  }

  /**
   * Calculate rolling metrics
   */
  public calculateRollingMetrics(
    historicalValues: number[],
    windowSize: number = 30
  ): RollingMetrics[] {
    const rollingMetrics: RollingMetrics[] = [];
    
    for (let i = windowSize; i < historicalValues.length; i++) {
      const windowValues = historicalValues.slice(i - windowSize, i + 1);
      const windowReturns = this.calculateReturns(windowValues);
      
      const date = new Date();
      date.setDate(date.getDate() - (historicalValues.length - i - 1));
      
      const rollingReturn = (windowValues[windowValues.length - 1] / windowValues[0]) - 1;
      const rollingVolatility = this.calculateVolatility(windowReturns) * Math.sqrt(252);
      const rollingSharpe = rollingVolatility > 0 ? 
        (rollingReturn * 252 - this.riskFreeRate) / rollingVolatility : 0;
      const rollingMaxDrawdown = this.calculateMaxDrawdown(windowValues);
      
      const benchmarkWindow = this.benchmarkReturns.slice(
        Math.max(0, this.benchmarkReturns.length - windowSize), 
        this.benchmarkReturns.length
      );
      const rollingBeta = this.calculateBeta(windowReturns, benchmarkWindow);

      rollingMetrics.push({
        date: date.toISOString().split('T')[0],
        rolling_return: rollingReturn,
        rolling_volatility: rollingVolatility,
        rolling_sharpe: rollingSharpe,
        rolling_max_drawdown: rollingMaxDrawdown,
        rolling_beta: rollingBeta
      });
    }

    return rollingMetrics;
  }

  /**
   * Helper calculation methods
   */
  private calculateReturns(values: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < values.length; i++) {
      returns.push((values[i] / values[i - 1]) - 1);
    }
    return returns;
  }

  private calculateVolatility(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance);
  }

  private calculateDownsideVolatility(returns: number[]): number {
    const downsideReturns = returns.filter(r => r < 0);
    return this.calculateVolatility(downsideReturns);
  }

  private calculateMaxDrawdown(values: number[]): number {
    let maxDrawdown = 0;
    let peak = values[0];
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
      }
      const drawdown = (peak - value) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    
    return maxDrawdown;
  }

  private calculateMaxDrawdownDuration(values: number[]): number {
    let maxDuration = 0;
    let currentDuration = 0;
    let peak = values[0];
    let inDrawdown = false;
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
        if (inDrawdown) {
          maxDuration = Math.max(maxDuration, currentDuration);
          currentDuration = 0;
          inDrawdown = false;
        }
      } else if (value < peak) {
        if (!inDrawdown) {
          inDrawdown = true;
          currentDuration = 1;
        } else {
          currentDuration++;
        }
      }
    }
    
    return Math.max(maxDuration, currentDuration);
  }

  private calculateOmegaRatio(returns: number[], threshold: number = 0): number {
    const gains = returns.filter(r => r > threshold).reduce((sum, r) => sum + (r - threshold), 0);
    const losses = Math.abs(returns.filter(r => r <= threshold).reduce((sum, r) => sum + (threshold - r), 0));
    
    return losses > 0 ? gains / losses : gains > 0 ? 999 : 0;
  }

  private calculateVaR(returns: number[], confidence: number): number {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sortedReturns.length);
    return sortedReturns[index] || 0;
  }

  private calculateCVaR(returns: number[], confidence: number): number {
    const varValue = this.calculateVaR(returns, confidence);
    const tailReturns = returns.filter(r => r <= varValue);
    return tailReturns.length > 0 ? tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length : 0;
  }

  private calculateBenchmarkComparison(returns: number[]): BenchmarkComparison {
    const benchmarkReturns = this.benchmarkReturns.slice(-returns.length);
    
    const portfolioReturn = returns.reduce((sum, r) => sum + r, 0);
    const benchmarkReturn = benchmarkReturns.reduce((sum, r) => sum + r, 0);
    const activeReturn = portfolioReturn - benchmarkReturn;
    
    const trackingError = this.calculateVolatility(
      returns.map((r, i) => r - benchmarkReturns[i])
    ) * Math.sqrt(252);
    
    const informationRatio = trackingError > 0 ? activeReturn / trackingError : 0;
    
    const correlation = this.calculateCorrelation(returns, benchmarkReturns);
    const beta = this.calculateBeta(returns, benchmarkReturns);
    const alpha = portfolioReturn - (this.riskFreeRate / 252 + beta * (benchmarkReturn - this.riskFreeRate / 252));
    
    const upMarkets = benchmarkReturns.map((r, i) => ({ portfolio: returns[i], benchmark: r }))
      .filter(pair => pair.benchmark > 0);
    const downMarkets = benchmarkReturns.map((r, i) => ({ portfolio: returns[i], benchmark: r }))
      .filter(pair => pair.benchmark < 0);
    
    const upCapture = upMarkets.length > 0 ? 
      (upMarkets.reduce((sum, pair) => sum + pair.portfolio, 0) / upMarkets.length) /
      (upMarkets.reduce((sum, pair) => sum + pair.benchmark, 0) / upMarkets.length) : 1;
    
    const downCapture = downMarkets.length > 0 ? 
      (downMarkets.reduce((sum, pair) => sum + pair.portfolio, 0) / downMarkets.length) /
      (downMarkets.reduce((sum, pair) => sum + pair.benchmark, 0) / downMarkets.length) : 1;

    return {
      portfolio_return: portfolioReturn,
      benchmark_return: benchmarkReturn,
      active_return: activeReturn,
      tracking_error: trackingError,
      information_ratio: informationRatio,
      up_capture_ratio: upCapture,
      down_capture_ratio: downCapture,
      correlation,
      beta,
      alpha
    };
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const xMean = x.reduce((sum, val) => sum + val, 0) / x.length;
    const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
    
    let numerator = 0;
    let xSumSquares = 0;
    let ySumSquares = 0;
    
    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - xMean;
      const yDiff = y[i] - yMean;
      numerator += xDiff * yDiff;
      xSumSquares += xDiff * xDiff;
      ySumSquares += yDiff * yDiff;
    }
    
    const denominator = Math.sqrt(xSumSquares * ySumSquares);
    return denominator > 0 ? numerator / denominator : 0;
  }

  private calculateBeta(portfolioReturns: number[], benchmarkReturns: number[]): number {
    if (portfolioReturns.length !== benchmarkReturns.length || portfolioReturns.length < 2) return 1;
    
    const benchmarkVariance = this.calculateVolatility(benchmarkReturns) ** 2;
    if (benchmarkVariance === 0) return 1;
    
    const covariance = this.calculateCovariance(portfolioReturns, benchmarkReturns);
    return covariance / benchmarkVariance;
  }

  private calculateCovariance(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const xMean = x.reduce((sum, val) => sum + val, 0) / x.length;
    const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
    
    return x.reduce((sum, xVal, i) => sum + (xVal - xMean) * (y[i] - yMean), 0) / (x.length - 1);
  }

  private calculateProfitFactor(returns: number[]): number {
    const profits = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
    const losses = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
    
    return losses > 0 ? profits / losses : profits > 0 ? 999 : 0;
  }

  private calculateSterlingRatio(annualizedReturn: number, values: number[]): number {
    const avgDrawdown = this.calculateAverageDrawdown(values);
    return avgDrawdown > 0 ? annualizedReturn / avgDrawdown : 0;
  }

  private calculateAverageDrawdown(values: number[]): number {
    const drawdowns: number[] = [];
    let peak = values[0];
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
      }
      const drawdown = (peak - value) / peak;
      if (drawdown > 0) {
        drawdowns.push(drawdown);
      }
    }
    
    return drawdowns.length > 0 ? drawdowns.reduce((sum, dd) => sum + dd, 0) / drawdowns.length : 0;
  }

  private getPeriodMultiplier(period: string): number {
    switch (period) {
      case '1D': return 252;
      case '1W': return 52;
      case '1M': return 12;
      case '3M': return 4;
      case '6M': return 2;
      case '1Y': return 1;
      default: return 1;
    }
  }

  private getCashFlowsForPeriod(period: string): CashFlow[] {
    // Mock cash flows - in practice would be retrieved from data store
    return this.cashFlows.slice(-this.getPeriodMultiplier(period) * 10);
  }

  private calculateIRR(cashFlows: CashFlow[], finalValue: number): number {
    // Simplified IRR calculation - would use Newton-Raphson in practice
    return Math.random() * 0.15 + 0.05; // Mock 5-20% IRR
  }

  private calculateModifiedDietz(cashFlows: CashFlow[], finalValue: number): number {
    // Simplified Modified Dietz return
    const totalCashFlow = cashFlows.reduce((sum, cf) => sum + cf.amount, 0);
    const beginningValue = finalValue - totalCashFlow;
    
    return beginningValue > 0 ? (finalValue - beginningValue - totalCashFlow) / beginningValue : 0;
  }

  private getEmptyMetrics(): PerformanceMetrics {
    return {
      total_return: 0,
      annualized_return: 0,
      volatility: 0,
      sharpe_ratio: 0,
      sortino_ratio: 0,
      max_drawdown: 0,
      max_drawdown_duration: 0,
      calmar_ratio: 0,
      omega_ratio: 0,
      var_95: 0,
      cvar_95: 0,
      beta: 1,
      alpha: 0,
      information_ratio: 0,
      tracking_error: 0,
      up_capture: 1,
      down_capture: 1,
      win_rate: 0,
      profit_factor: 0,
      recovery_factor: 0,
      sterling_ratio: 0
    };
  }

  /**
   * Public methods for cash flow management
   */
  public addCashFlow(cashFlow: CashFlow): void {
    this.cashFlows.push(cashFlow);
  }

  public setCashFlows(cashFlows: CashFlow[]): void {
    this.cashFlows = [...cashFlows];
  }

  public updateRiskFreeRate(rate: number): void {
    this.riskFreeRate = rate;
  }

  public setBenchmarkReturns(returns: number[]): void {
    this.benchmarkReturns = [...returns];
  }
}

// Export singleton instance
export const portfolioMetricsService = new PortfolioMetricsService();