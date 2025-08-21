/**
 * Correlation Calculator Service for statistical calculations
 */

import { Position } from '../types/position';
import { StrategyPnL } from './portfolioAggregationService';

export interface CorrelationMatrix {
  strategies: string[];
  matrix: number[][];
  pValues: number[][];
  significanceMatrix: boolean[][];
}

export interface MarketCorrelation {
  strategy_id: string;
  sp500_correlation: number;
  nasdaq_correlation: number;
  bond_correlation: number;
  vix_correlation: number;
  commodity_correlation: number;
}

export interface FactorExposure {
  factor_name: string;
  exposure: number;
  confidence_interval: [number, number];
  r_squared: number;
  t_statistic: number;
  p_value: number;
}

export interface RollingCorrelation {
  date: string;
  correlation: number;
  volatility: number;
  beta: number;
}

export class CorrelationCalculatorService {
  private readonly SIGNIFICANCE_LEVEL = 0.05;
  private marketData: Map<string, number[]> = new Map();
  private factorData: Map<string, number[]> = new Map();

  constructor() {
    this.initializeMarketData();
    this.initializeFactorData();
  }

  /**
   * Initialize mock market data
   */
  private initializeMarketData(): void {
    const periods = 252; // 1 year of daily data
    
    // S&P 500 returns (mock)
    this.marketData.set('SP500', this.generateMarketReturns(0.0008, 0.012, periods));
    
    // NASDAQ returns (mock)
    this.marketData.set('NASDAQ', this.generateMarketReturns(0.001, 0.015, periods));
    
    // Bond returns (mock)
    this.marketData.set('BONDS', this.generateMarketReturns(0.0003, 0.005, periods));
    
    // VIX (volatility index) - inverse relationship with markets
    this.marketData.set('VIX', this.generateMarketReturns(-0.0002, 0.2, periods));
    
    // Commodity returns (mock)
    this.marketData.set('COMMODITY', this.generateMarketReturns(0.0004, 0.018, periods));
  }

  /**
   * Initialize factor data (Fama-French factors)
   */
  private initializeFactorData(): void {
    const periods = 252;
    
    // Market factor (broad market return)
    this.factorData.set('Market', this.generateMarketReturns(0.0008, 0.012, periods));
    
    // Size factor (small minus big)
    this.factorData.set('SMB', this.generateMarketReturns(0.0001, 0.008, periods));
    
    // Value factor (high minus low)
    this.factorData.set('HML', this.generateMarketReturns(0.0002, 0.009, periods));
    
    // Momentum factor
    this.factorData.set('MOM', this.generateMarketReturns(0.0003, 0.010, periods));
    
    // Quality factor
    this.factorData.set('QMJ', this.generateMarketReturns(0.0001, 0.007, periods));
    
    // Low volatility factor
    this.factorData.set('BAB', this.generateMarketReturns(0.0002, 0.006, periods));
  }

  /**
   * Generate mock market returns with specified parameters
   */
  private generateMarketReturns(mean: number, volatility: number, periods: number): number[] {
    const returns: number[] = [];
    
    for (let i = 0; i < periods; i++) {
      // Box-Muller transformation for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      
      const dailyReturn = mean + volatility / Math.sqrt(252) * z;
      returns.push(dailyReturn);
    }
    
    return returns;
  }

  /**
   * Calculate correlation matrix between strategies
   */
  public calculateStrategyCorrelationMatrix(strategies: StrategyPnL[]): CorrelationMatrix {
    const strategyNames = strategies.map(s => s.strategy_name);
    const strategyReturns = strategies.map(s => this.generateStrategyReturns(s));
    
    const n = strategies.length;
    const matrix: number[][] = [];
    const pValues: number[][] = [];
    const significanceMatrix: boolean[][] = [];
    
    for (let i = 0; i < n; i++) {
      matrix[i] = [];
      pValues[i] = [];
      significanceMatrix[i] = [];
      
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
          pValues[i][j] = 0.0;
          significanceMatrix[i][j] = true;
        } else {
          const correlation = this.calculatePearsonCorrelation(strategyReturns[i], strategyReturns[j]);
          const pValue = this.calculateCorrelationPValue(correlation, strategyReturns[i].length);
          
          matrix[i][j] = correlation;
          pValues[i][j] = pValue;
          significanceMatrix[i][j] = pValue < this.SIGNIFICANCE_LEVEL;
        }
      }
    }
    
    return {
      strategies: strategyNames,
      matrix,
      pValues,
      significanceMatrix
    };
  }

  /**
   * Calculate market correlations for strategies
   */
  public calculateMarketCorrelations(strategies: StrategyPnL[]): MarketCorrelation[] {
    return strategies.map(strategy => {
      const strategyReturns = this.generateStrategyReturns(strategy);
      
      return {
        strategy_id: strategy.strategy_id,
        sp500_correlation: this.calculatePearsonCorrelation(
          strategyReturns, 
          this.marketData.get('SP500') || []
        ),
        nasdaq_correlation: this.calculatePearsonCorrelation(
          strategyReturns, 
          this.marketData.get('NASDAQ') || []
        ),
        bond_correlation: this.calculatePearsonCorrelation(
          strategyReturns, 
          this.marketData.get('BONDS') || []
        ),
        vix_correlation: this.calculatePearsonCorrelation(
          strategyReturns, 
          this.marketData.get('VIX') || []
        ),
        commodity_correlation: this.calculatePearsonCorrelation(
          strategyReturns, 
          this.marketData.get('COMMODITY') || []
        )
      };
    });
  }

  /**
   * Calculate factor exposures for a strategy
   */
  public calculateFactorExposures(strategy: StrategyPnL): FactorExposure[] {
    const strategyReturns = this.generateStrategyReturns(strategy);
    const exposures: FactorExposure[] = [];
    
    for (const [factorName, factorReturns] of this.factorData.entries()) {
      const regression = this.calculateLinearRegression(strategyReturns, factorReturns);
      
      exposures.push({
        factor_name: factorName,
        exposure: regression.beta,
        confidence_interval: regression.confidenceInterval,
        r_squared: regression.rSquared,
        t_statistic: regression.tStatistic,
        p_value: regression.pValue
      });
    }
    
    return exposures.sort((a, b) => Math.abs(b.exposure) - Math.abs(a.exposure));
  }

  /**
   * Calculate rolling correlations
   */
  public calculateRollingCorrelations(
    strategy: StrategyPnL, 
    benchmark: string = 'SP500',
    windowSize: number = 30
  ): RollingCorrelation[] {
    const strategyReturns = this.generateStrategyReturns(strategy);
    const benchmarkReturns = this.marketData.get(benchmark) || [];
    
    const rollingCorrelations: RollingCorrelation[] = [];
    
    for (let i = windowSize; i < strategyReturns.length; i++) {
      const strategyWindow = strategyReturns.slice(i - windowSize, i);
      const benchmarkWindow = benchmarkReturns.slice(i - windowSize, i);
      
      const correlation = this.calculatePearsonCorrelation(strategyWindow, benchmarkWindow);
      const volatility = this.calculateVolatility(strategyWindow);
      const beta = this.calculateBeta(strategyWindow, benchmarkWindow);
      
      const date = new Date();
      date.setDate(date.getDate() - (strategyReturns.length - i));
      
      rollingCorrelations.push({
        date: date.toISOString().split('T')[0],
        correlation,
        volatility: volatility * Math.sqrt(252), // Annualized
        beta
      });
    }
    
    return rollingCorrelations;
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private calculatePearsonCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Calculate p-value for correlation significance testing
   */
  private calculateCorrelationPValue(correlation: number, sampleSize: number): number {
    if (sampleSize < 3) return 1;
    
    // t-test for correlation significance
    const t = correlation * Math.sqrt((sampleSize - 2) / (1 - correlation * correlation));
    
    // Simplified p-value calculation (would use t-distribution CDF in practice)
    const absT = Math.abs(t);
    if (absT > 2.58) return 0.01;  // 99% confidence
    if (absT > 1.96) return 0.05;  // 95% confidence
    if (absT > 1.64) return 0.1;   // 90% confidence
    return 0.2;
  }

  /**
   * Calculate linear regression for factor analysis
   */
  private calculateLinearRegression(y: number[], x: number[]): {
    beta: number;
    alpha: number;
    rSquared: number;
    tStatistic: number;
    pValue: number;
    confidenceInterval: [number, number];
  } {
    if (x.length !== y.length || x.length < 2) {
      return {
        beta: 0, alpha: 0, rSquared: 0, tStatistic: 0, 
        pValue: 1, confidenceInterval: [0, 0]
      };
    }
    
    const n = x.length;
    const xMean = x.reduce((sum, val) => sum + val, 0) / n;
    const yMean = y.reduce((sum, val) => sum + val, 0) / n;
    
    let xxSum = 0;
    let xySum = 0;
    let yySum = 0;
    
    for (let i = 0; i < n; i++) {
      const xDiff = x[i] - xMean;
      const yDiff = y[i] - yMean;
      xxSum += xDiff * xDiff;
      xySum += xDiff * yDiff;
      yySum += yDiff * yDiff;
    }
    
    const beta = xxSum === 0 ? 0 : xySum / xxSum;
    const alpha = yMean - beta * xMean;
    
    // Calculate R-squared
    const totalSumSquares = yySum;
    const residualSumSquares = y.reduce((sum, yi, i) => {
      const predicted = alpha + beta * x[i];
      return sum + Math.pow(yi - predicted, 2);
    }, 0);
    
    const rSquared = totalSumSquares === 0 ? 0 : 1 - (residualSumSquares / totalSumSquares);
    
    // Calculate t-statistic for beta
    const standardError = Math.sqrt(residualSumSquares / ((n - 2) * xxSum));
    const tStatistic = standardError === 0 ? 0 : beta / standardError;
    const pValue = this.calculateCorrelationPValue(Math.abs(tStatistic) / Math.sqrt(n - 2), n);
    
    // 95% confidence interval
    const criticalValue = 1.96; // Approximate for large samples
    const margin = criticalValue * standardError;
    const confidenceInterval: [number, number] = [beta - margin, beta + margin];
    
    return {
      beta,
      alpha,
      rSquared,
      tStatistic,
      pValue,
      confidenceInterval
    };
  }

  /**
   * Calculate beta coefficient
   */
  private calculateBeta(strategyReturns: number[], marketReturns: number[]): number {
    const covariance = this.calculateCovariance(strategyReturns, marketReturns);
    const marketVariance = this.calculateVariance(marketReturns);
    
    return marketVariance === 0 ? 1 : covariance / marketVariance;
  }

  /**
   * Calculate covariance between two return series
   */
  private calculateCovariance(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const xMean = x.reduce((sum, val) => sum + val, 0) / x.length;
    const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
    
    return x.reduce((sum, xi, i) => sum + (xi - xMean) * (y[i] - yMean), 0) / (x.length - 1);
  }

  /**
   * Calculate variance of return series
   */
  private calculateVariance(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    return returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / (returns.length - 1);
  }

  /**
   * Calculate volatility (standard deviation)
   */
  private calculateVolatility(returns: number[]): number {
    return Math.sqrt(this.calculateVariance(returns));
  }

  /**
   * Generate strategy returns based on strategy performance
   */
  private generateStrategyReturns(strategy: StrategyPnL): number[] {
    const periods = 252;
    const totalReturn = strategy.total_pnl / 100000; // Assume $100k base
    const avgDailyReturn = totalReturn / periods;
    const volatility = Math.max(0.005, Math.abs(totalReturn) * 0.3); // Strategy volatility
    
    const returns: number[] = [];
    for (let i = 0; i < periods; i++) {
      const randomComponent = (Math.random() - 0.5) * volatility * 2;
      returns.push(avgDailyReturn + randomComponent);
    }
    
    return returns;
  }

  /**
   * Update market data
   */
  public updateMarketData(symbol: string, returns: number[]): void {
    this.marketData.set(symbol, returns);
  }

  /**
   * Get available market symbols
   */
  public getAvailableMarkets(): string[] {
    return Array.from(this.marketData.keys());
  }

  /**
   * Get available factors
   */
  public getAvailableFactors(): string[] {
    return Array.from(this.factorData.keys());
  }
}

// Export singleton instance
export const correlationCalculator = new CorrelationCalculatorService();