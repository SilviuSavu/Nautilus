/**
 * Analytics Utilities
 * Sprint 3: Advanced Analytics and Performance Calculation Utilities
 * 
 * Comprehensive utilities for financial calculations, statistical analysis,
 * performance metrics, and data transformation.
 */

// Financial Calculation Utilities
export class FinancialCalculations {
  // Risk-free rate (annual, as decimal)
  static readonly DEFAULT_RISK_FREE_RATE = 0.02;
  static readonly TRADING_DAYS_PER_YEAR = 252;
  static readonly BASIS_POINTS = 10000;

  // Basic returns calculations
  static calculateReturn(startValue: number, endValue: number): number {
    if (startValue === 0) return 0;
    return (endValue - startValue) / startValue;
  }

  static calculateLogReturn(startValue: number, endValue: number): number {
    if (startValue <= 0 || endValue <= 0) return 0;
    return Math.log(endValue / startValue);
  }

  static calculateCumulativeReturn(returns: number[]): number {
    return returns.reduce((cum, ret) => cum * (1 + ret), 1) - 1;
  }

  // Annualized calculations
  static annualizeReturn(totalReturn: number, periods: number, periodsPerYear = this.TRADING_DAYS_PER_YEAR): number {
    if (periods === 0) return 0;
    return Math.pow(1 + totalReturn, periodsPerYear / periods) - 1;
  }

  static annualizeVolatility(volatility: number, periodsPerYear = this.TRADING_DAYS_PER_YEAR): number {
    return volatility * Math.sqrt(periodsPerYear);
  }

  // Risk metrics
  static calculateSharpeRatio(
    returns: number[], 
    riskFreeRate = this.DEFAULT_RISK_FREE_RATE, 
    periodsPerYear = this.TRADING_DAYS_PER_YEAR
  ): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = this.calculateMean(returns);
    const volatility = this.calculateStandardDeviation(returns);
    
    if (volatility === 0) return 0;
    
    const annualizedReturn = avgReturn * periodsPerYear;
    const annualizedVolatility = this.annualizeVolatility(volatility, periodsPerYear);
    
    return (annualizedReturn - riskFreeRate) / annualizedVolatility;
  }

  static calculateSortinoRatio(
    returns: number[], 
    riskFreeRate = this.DEFAULT_RISK_FREE_RATE,
    periodsPerYear = this.TRADING_DAYS_PER_YEAR
  ): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = this.calculateMean(returns);
    const downsideReturns = returns.filter(r => r < 0);
    const downsideDeviation = downsideReturns.length > 0 
      ? this.calculateStandardDeviation(downsideReturns) 
      : 0;
    
    if (downsideDeviation === 0) return 0;
    
    const annualizedReturn = avgReturn * periodsPerYear;
    const annualizedDownsideDeviation = this.annualizeVolatility(downsideDeviation, periodsPerYear);
    
    return (annualizedReturn - riskFreeRate) / annualizedDownsideDeviation;
  }

  static calculateCalmarRatio(returns: number[], periodsPerYear = this.TRADING_DAYS_PER_YEAR): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = this.calculateMean(returns);
    const maxDrawdown = this.calculateMaxDrawdown(returns);
    
    if (maxDrawdown === 0) return 0;
    
    const annualizedReturn = avgReturn * periodsPerYear;
    return annualizedReturn / maxDrawdown;
  }

  // Drawdown calculations
  static calculateDrawdown(cumulativeReturns: number[]): number[] {
    if (cumulativeReturns.length === 0) return [];
    
    const drawdowns: number[] = [];
    let peak = cumulativeReturns[0];
    
    for (let i = 0; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        peak = cumulativeReturns[i];
      }
      
      const drawdown = peak > 0 ? (peak - cumulativeReturns[i]) / peak : 0;
      drawdowns.push(drawdown);
    }
    
    return drawdowns;
  }

  static calculateMaxDrawdown(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const cumulativeReturns = this.calculateCumulativeReturnsArray(returns);
    const drawdowns = this.calculateDrawdown(cumulativeReturns);
    
    return Math.max(...drawdowns);
  }

  // VaR and CVaR calculations
  static calculateVaR(returns: number[], confidenceLevel = 0.95): number {
    if (returns.length === 0) return 0;
    
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
    
    return -sortedReturns[index];
  }

  static calculateCVaR(returns: number[], confidenceLevel = 0.95): number {
    if (returns.length === 0) return 0;
    
    const var95 = this.calculateVaR(returns, confidenceLevel);
    const tailReturns = returns.filter(r => r <= -var95);
    
    return tailReturns.length > 0 ? -this.calculateMean(tailReturns) : 0;
  }

  // Beta and Alpha calculations
  static calculateBeta(
    assetReturns: number[], 
    benchmarkReturns: number[]
  ): number {
    if (assetReturns.length === 0 || benchmarkReturns.length === 0) return 0;
    
    const minLength = Math.min(assetReturns.length, benchmarkReturns.length);
    const assetSlice = assetReturns.slice(0, minLength);
    const benchmarkSlice = benchmarkReturns.slice(0, minLength);
    
    const covariance = this.calculateCovariance(assetSlice, benchmarkSlice);
    const benchmarkVariance = this.calculateVariance(benchmarkSlice);
    
    return benchmarkVariance === 0 ? 0 : covariance / benchmarkVariance;
  }

  static calculateAlpha(
    assetReturns: number[], 
    benchmarkReturns: number[],
    riskFreeRate = this.DEFAULT_RISK_FREE_RATE,
    periodsPerYear = this.TRADING_DAYS_PER_YEAR
  ): number {
    if (assetReturns.length === 0 || benchmarkReturns.length === 0) return 0;
    
    const beta = this.calculateBeta(assetReturns, benchmarkReturns);
    const assetReturn = this.calculateMean(assetReturns) * periodsPerYear;
    const benchmarkReturn = this.calculateMean(benchmarkReturns) * periodsPerYear;
    
    return assetReturn - (riskFreeRate + beta * (benchmarkReturn - riskFreeRate));
  }

  // Information ratio and tracking error
  static calculateTrackingError(
    assetReturns: number[], 
    benchmarkReturns: number[],
    periodsPerYear = this.TRADING_DAYS_PER_YEAR
  ): number {
    if (assetReturns.length === 0 || benchmarkReturns.length === 0) return 0;
    
    const minLength = Math.min(assetReturns.length, benchmarkReturns.length);
    const activeReturns = [];
    
    for (let i = 0; i < minLength; i++) {
      activeReturns.push(assetReturns[i] - benchmarkReturns[i]);
    }
    
    const volatility = this.calculateStandardDeviation(activeReturns);
    return this.annualizeVolatility(volatility, periodsPerYear);
  }

  static calculateInformationRatio(
    assetReturns: number[], 
    benchmarkReturns: number[],
    periodsPerYear = this.TRADING_DAYS_PER_YEAR
  ): number {
    if (assetReturns.length === 0 || benchmarkReturns.length === 0) return 0;
    
    const trackingError = this.calculateTrackingError(assetReturns, benchmarkReturns, periodsPerYear);
    if (trackingError === 0) return 0;
    
    const minLength = Math.min(assetReturns.length, benchmarkReturns.length);
    const activeReturns = [];
    
    for (let i = 0; i < minLength; i++) {
      activeReturns.push(assetReturns[i] - benchmarkReturns[i]);
    }
    
    const averageActiveReturn = this.calculateMean(activeReturns) * periodsPerYear;
    return averageActiveReturn / trackingError;
  }

  // Statistical measures
  static calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  static calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    
    return this.calculateMean(squaredDiffs);
  }

  static calculateStandardDeviation(values: number[]): number {
    return Math.sqrt(this.calculateVariance(values));
  }

  static calculateSkewness(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = this.calculateMean(values);
    const stdDev = this.calculateStandardDeviation(values);
    
    if (stdDev === 0) return 0;
    
    const n = values.length;
    const skewSum = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 3), 0);
    
    return (n / ((n - 1) * (n - 2))) * skewSum;
  }

  static calculateKurtosis(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = this.calculateMean(values);
    const stdDev = this.calculateStandardDeviation(values);
    
    if (stdDev === 0) return 0;
    
    const n = values.length;
    const kurtSum = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 4), 0);
    
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * kurtSum - 
           3 * Math.pow(n - 1, 2) / ((n - 2) * (n - 3));
  }

  static calculateCovariance(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const meanX = this.calculateMean(x);
    const meanY = this.calculateMean(y);
    
    let covariance = 0;
    for (let i = 0; i < x.length; i++) {
      covariance += (x[i] - meanX) * (y[i] - meanY);
    }
    
    return covariance / x.length;
  }

  static calculateCorrelation(x: number[], y: number[]): number {
    const covariance = this.calculateCovariance(x, y);
    const stdDevX = this.calculateStandardDeviation(x);
    const stdDevY = this.calculateStandardDeviation(y);
    
    if (stdDevX === 0 || stdDevY === 0) return 0;
    
    return covariance / (stdDevX * stdDevY);
  }

  // Helper methods
  static calculateCumulativeReturnsArray(returns: number[]): number[] {
    const cumulative = [1];
    
    for (let i = 0; i < returns.length; i++) {
      cumulative.push(cumulative[cumulative.length - 1] * (1 + returns[i]));
    }
    
    return cumulative;
  }

  static calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sorted[lower];
    
    const weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }
}

// Data Processing Utilities
export class DataProcessor {
  // Time series processing
  static resampleTimeSeries(
    data: { timestamp: string; value: number }[],
    frequency: 'minute' | 'hour' | 'day' | 'week' | 'month'
  ): { timestamp: string; value: number }[] {
    if (data.length === 0) return [];

    const grouped = new Map<string, number[]>();
    
    data.forEach(point => {
      const date = new Date(point.timestamp);
      let key: string;
      
      switch (frequency) {
        case 'minute':
          key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${date.getHours()}-${date.getMinutes()}`;
          break;
        case 'hour':
          key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${date.getHours()}`;
          break;
        case 'day':
          key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
          break;
        case 'week':
          const week = this.getWeekNumber(date);
          key = `${date.getFullYear()}-${week}`;
          break;
        case 'month':
          key = `${date.getFullYear()}-${date.getMonth()}`;
          break;
        default:
          key = point.timestamp;
      }
      
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key)!.push(point.value);
    });

    const result: { timestamp: string; value: number }[] = [];
    grouped.forEach((values, key) => {
      const avgValue = FinancialCalculations.calculateMean(values);
      result.push({
        timestamp: this.keyToTimestamp(key, frequency),
        value: avgValue
      });
    });

    return result.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }

  // Rolling calculations
  static calculateRollingMetric(
    values: number[],
    windowSize: number,
    calculator: (window: number[]) => number
  ): number[] {
    if (values.length < windowSize) return [];
    
    const result: number[] = [];
    
    for (let i = windowSize - 1; i < values.length; i++) {
      const window = values.slice(i - windowSize + 1, i + 1);
      result.push(calculator(window));
    }
    
    return result;
  }

  static calculateRollingReturns(values: number[], windowSize: number): number[] {
    return this.calculateRollingMetric(values, windowSize, (window) => {
      return FinancialCalculations.calculateCumulativeReturn(
        window.slice(1).map((val, idx) => FinancialCalculations.calculateReturn(window[idx], val))
      );
    });
  }

  static calculateRollingVolatility(values: number[], windowSize: number): number[] {
    return this.calculateRollingMetric(values, windowSize, (window) => {
      const returns = window.slice(1).map((val, idx) => 
        FinancialCalculations.calculateReturn(window[idx], val)
      );
      return FinancialCalculations.calculateStandardDeviation(returns);
    });
  }

  static calculateRollingSharpe(
    values: number[], 
    windowSize: number,
    riskFreeRate = FinancialCalculations.DEFAULT_RISK_FREE_RATE
  ): number[] {
    return this.calculateRollingMetric(values, windowSize, (window) => {
      const returns = window.slice(1).map((val, idx) => 
        FinancialCalculations.calculateReturn(window[idx], val)
      );
      return FinancialCalculations.calculateSharpeRatio(returns, riskFreeRate);
    });
  }

  // Data cleaning and validation
  static cleanTimeSeries(data: { timestamp: string; value: number }[]): { timestamp: string; value: number }[] {
    return data
      .filter(point => !isNaN(point.value) && isFinite(point.value))
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .filter((point, index, arr) => {
        // Remove duplicates based on timestamp
        return index === 0 || point.timestamp !== arr[index - 1].timestamp;
      });
  }

  static removeOutliers(
    values: number[], 
    method: 'iqr' | 'zscore' = 'iqr',
    threshold = 3
  ): number[] {
    if (values.length === 0) return [];

    if (method === 'iqr') {
      const q1 = FinancialCalculations.calculatePercentile(values, 25);
      const q3 = FinancialCalculations.calculatePercentile(values, 75);
      const iqr = q3 - q1;
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;
      
      return values.filter(val => val >= lowerBound && val <= upperBound);
    } else {
      const mean = FinancialCalculations.calculateMean(values);
      const stdDev = FinancialCalculations.calculateStandardDeviation(values);
      
      return values.filter(val => Math.abs((val - mean) / stdDev) <= threshold);
    }
  }

  static interpolateMissingValues(
    data: { timestamp: string; value: number | null }[],
    method: 'linear' | 'forward_fill' | 'backward_fill' = 'linear'
  ): { timestamp: string; value: number }[] {
    const result: { timestamp: string; value: number }[] = [];
    
    for (let i = 0; i < data.length; i++) {
      if (data[i].value !== null) {
        result.push({ timestamp: data[i].timestamp, value: data[i].value! });
      } else {
        let interpolatedValue: number;
        
        switch (method) {
          case 'forward_fill':
            const prevValue = result[result.length - 1]?.value;
            interpolatedValue = prevValue || 0;
            break;
          case 'backward_fill':
            const nextPoint = data.slice(i + 1).find(point => point.value !== null);
            interpolatedValue = nextPoint?.value || 0;
            break;
          case 'linear':
          default:
            const prevPoint = result[result.length - 1];
            const nextPoint2 = data.slice(i + 1).find(point => point.value !== null);
            
            if (prevPoint && nextPoint2) {
              const prevTime = new Date(prevPoint.timestamp).getTime();
              const currentTime = new Date(data[i].timestamp).getTime();
              const nextTime = new Date(nextPoint2.timestamp).getTime();
              
              const ratio = (currentTime - prevTime) / (nextTime - prevTime);
              interpolatedValue = prevPoint.value + ratio * (nextPoint2.value! - prevPoint.value);
            } else {
              interpolatedValue = prevPoint?.value || nextPoint2?.value || 0;
            }
            break;
        }
        
        result.push({ timestamp: data[i].timestamp, value: interpolatedValue });
      }
    }
    
    return result;
  }

  // Helper methods
  private static getWeekNumber(date: Date): number {
    const onejan = new Date(date.getFullYear(), 0, 1);
    const millisecsInDay = 86400000;
    return Math.ceil(((date.getTime() - onejan.getTime()) / millisecsInDay + onejan.getDay() + 1) / 7);
  }

  private static keyToTimestamp(key: string, frequency: string): string {
    const parts = key.split('-');
    const year = parseInt(parts[0]);
    
    switch (frequency) {
      case 'minute':
        return new Date(year, parseInt(parts[1]), parseInt(parts[2]), parseInt(parts[3]), parseInt(parts[4])).toISOString();
      case 'hour':
        return new Date(year, parseInt(parts[1]), parseInt(parts[2]), parseInt(parts[3])).toISOString();
      case 'day':
        return new Date(year, parseInt(parts[1]), parseInt(parts[2])).toISOString();
      case 'week':
        const week = parseInt(parts[1]);
        const jan4 = new Date(year, 0, 4);
        const weekStart = new Date(jan4.getTime() - (jan4.getDay() - 1) * 86400000 + (week - 1) * 7 * 86400000);
        return weekStart.toISOString();
      case 'month':
        return new Date(year, parseInt(parts[1]), 1).toISOString();
      default:
        return key;
    }
  }
}

// Performance Attribution Utilities
export class PerformanceAttribution {
  // Brinson attribution model
  static calculateBrinsonAttribution(
    portfolioWeights: Record<string, number>,
    benchmarkWeights: Record<string, number>,
    portfolioReturns: Record<string, number>,
    benchmarkReturns: Record<string, number>
  ): {
    allocation: number;
    selection: number;
    interaction: number;
    total: number;
  } {
    let allocation = 0;
    let selection = 0;
    let interaction = 0;

    const sectors = new Set([
      ...Object.keys(portfolioWeights),
      ...Object.keys(benchmarkWeights)
    ]);

    const benchmarkReturn = Object.values(benchmarkReturns).reduce((sum, ret, _, arr) => sum + ret / arr.length, 0);

    sectors.forEach(sector => {
      const wp = portfolioWeights[sector] || 0;
      const wb = benchmarkWeights[sector] || 0;
      const rp = portfolioReturns[sector] || 0;
      const rb = benchmarkReturns[sector] || benchmarkReturn;

      allocation += (wp - wb) * (rb - benchmarkReturn);
      selection += wb * (rp - rb);
      interaction += (wp - wb) * (rp - rb);
    });

    return {
      allocation,
      selection,
      interaction,
      total: allocation + selection + interaction
    };
  }

  // Factor-based attribution
  static calculateFactorAttribution(
    portfolioReturns: number[],
    factorLoadings: Record<string, number[]>,
    factorReturns: Record<string, number[]>
  ): {
    factorContributions: Record<string, number>;
    specificReturn: number;
    totalReturn: number;
  } {
    if (portfolioReturns.length === 0) {
      return { factorContributions: {}, specificReturn: 0, totalReturn: 0 };
    }

    const totalReturn = FinancialCalculations.calculateMean(portfolioReturns);
    const factorContributions: Record<string, number> = {};
    
    let totalFactorContribution = 0;

    Object.keys(factorLoadings).forEach(factor => {
      const loadings = factorLoadings[factor];
      const returns = factorReturns[factor];
      
      if (loadings && returns && loadings.length === returns.length) {
        const contribution = FinancialCalculations.calculateMean(
          loadings.map((loading, i) => loading * returns[i])
        );
        
        factorContributions[factor] = contribution;
        totalFactorContribution += contribution;
      }
    });

    const specificReturn = totalReturn - totalFactorContribution;

    return {
      factorContributions,
      specificReturn,
      totalReturn
    };
  }
}

// Risk Analytics Utilities
export class RiskAnalytics {
  // Regime detection
  static detectVolatilityRegimes(
    returns: number[],
    lookbackWindow = 30
  ): Array<{ start: number; end: number; regime: 'low' | 'medium' | 'high'; volatility: number }> {
    if (returns.length < lookbackWindow * 2) return [];

    const rollingVolatility = DataProcessor.calculateRollingVolatility(returns, lookbackWindow);
    const volThresholds = this.calculateVolatilityThresholds(rollingVolatility);
    
    const regimes: Array<{ start: number; end: number; regime: 'low' | 'medium' | 'high'; volatility: number }> = [];
    let currentRegime: 'low' | 'medium' | 'high' | null = null;
    let regimeStart = 0;

    rollingVolatility.forEach((vol, i) => {
      const regime = this.classifyVolatilityRegime(vol, volThresholds);
      
      if (regime !== currentRegime) {
        if (currentRegime !== null) {
          regimes.push({
            start: regimeStart,
            end: i - 1,
            regime: currentRegime,
            volatility: FinancialCalculations.calculateMean(rollingVolatility.slice(regimeStart, i))
          });
        }
        currentRegime = regime;
        regimeStart = i;
      }
    });

    // Add final regime
    if (currentRegime !== null && regimeStart < rollingVolatility.length) {
      regimes.push({
        start: regimeStart,
        end: rollingVolatility.length - 1,
        regime: currentRegime,
        volatility: FinancialCalculations.calculateMean(rollingVolatility.slice(regimeStart))
      });
    }

    return regimes;
  }

  private static calculateVolatilityThresholds(volatilities: number[]): { low: number; high: number } {
    const sortedVol = [...volatilities].sort((a, b) => a - b);
    return {
      low: FinancialCalculations.calculatePercentile(sortedVol, 33),
      high: FinancialCalculations.calculatePercentile(sortedVol, 67)
    };
  }

  private static classifyVolatilityRegime(
    volatility: number, 
    thresholds: { low: number; high: number }
  ): 'low' | 'medium' | 'high' {
    if (volatility <= thresholds.low) return 'low';
    if (volatility >= thresholds.high) return 'high';
    return 'medium';
  }

  // Stress testing utilities
  static applyStressScenario(
    baseReturns: number[],
    shockMagnitude: number,
    shockType: 'absolute' | 'relative' = 'relative'
  ): number[] {
    return baseReturns.map(ret => {
      if (shockType === 'absolute') {
        return ret + shockMagnitude;
      } else {
        return ret * (1 + shockMagnitude);
      }
    });
  }

  static calculateWorstCaseScenario(
    returns: number[],
    confidenceLevel = 0.01
  ): { scenario: number[]; probability: number } {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const worstCaseIndex = Math.floor(confidenceLevel * sortedReturns.length);
    const worstCaseReturn = sortedReturns[worstCaseIndex];
    
    // Create scenario with all returns at worst case level
    const scenario = returns.map(() => worstCaseReturn);
    
    return {
      scenario,
      probability: confidenceLevel
    };
  }
}

// Export utility functions for common operations
export const formatPercentage = (value: number, decimals = 2): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

export const formatBasisPoints = (value: number, decimals = 0): string => {
  return `${(value * FinancialCalculations.BASIS_POINTS).toFixed(decimals)} bps`;
};

export const formatCurrency = (value: number, currency = 'USD', decimals = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value);
};

export const formatLargeNumber = (value: number, decimals = 1): string => {
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (abs >= 1e12) return `${sign}${(abs / 1e12).toFixed(decimals)}T`;
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(decimals)}B`;
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(decimals)}M`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(decimals)}K`;
  
  return `${sign}${abs.toFixed(decimals)}`;
};

export const calculateCompoundAnnualGrowthRate = (
  beginningValue: number,
  endingValue: number,
  numberOfPeriods: number
): number => {
  if (beginningValue <= 0 || numberOfPeriods <= 0) return 0;
  return Math.pow(endingValue / beginningValue, 1 / numberOfPeriods) - 1;
};

export const calculateTimeWeightedReturn = (
  cashFlows: Array<{ date: string; amount: number }>,
  values: Array<{ date: string; value: number }>
): number => {
  // Simplified TWR calculation
  if (values.length < 2) return 0;
  
  const periods: number[] = [];
  
  for (let i = 1; i < values.length; i++) {
    const startValue = values[i - 1].value;
    const endValue = values[i].value;
    const startDate = new Date(values[i - 1].date);
    const endDate = new Date(values[i].date);
    
    // Find cash flows in this period
    const periodCashFlows = cashFlows.filter(cf => {
      const cfDate = new Date(cf.date);
      return cfDate > startDate && cfDate <= endDate;
    });
    
    const totalCashFlow = periodCashFlows.reduce((sum, cf) => sum + cf.amount, 0);
    const adjustedStartValue = startValue + totalCashFlow;
    
    if (adjustedStartValue > 0) {
      const periodReturn = (endValue - adjustedStartValue) / adjustedStartValue;
      periods.push(periodReturn);
    }
  }
  
  return FinancialCalculations.calculateCumulativeReturn(periods);
};