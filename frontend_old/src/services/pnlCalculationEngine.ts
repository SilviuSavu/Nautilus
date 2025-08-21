/**
 * P&L Calculation Engine for advanced profit and loss calculations
 */

import { Position, PnLCalculation, CurrencyConversion } from '../types/position';

export interface PnLConfiguration {
  baseCurrency: string;
  includeFees: boolean;
  useMarkToMarket: boolean;
  taxRate?: number;
}

export interface HistoricalPnL {
  date: string;
  dailyPnL: number;
  cumulativePnL: number;
  realizedPnL: number;
  unrealizedPnL: number;
  currency: string;
}

export interface PnLBreakdown {
  symbol: string;
  venue: string;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPnL: number;
  pnlPercentage: number;
  contribution: number; // Contribution to overall portfolio P&L
  currency: string;
}

export interface RiskMetrics {
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  beta?: number;
  var95: number; // 95% Value at Risk
  expectedShortfall: number;
}

export class PnLCalculationEngine {
  private config: PnLConfiguration;
  private currencyRates: Map<string, CurrencyConversion> = new Map();
  private historicalPrices: Map<string, number[]> = new Map();
  private historicalPnL: HistoricalPnL[] = [];

  constructor(config: PnLConfiguration) {
    this.config = config;
  }

  /**
   * Calculate real-time unrealized P&L for a position
   */
  public calculateUnrealizedPnL(
    position: Position,
    currentPrice: number,
    fees: number = 0
  ): number {
    const priceDifference = currentPrice - position.averagePrice;
    const multiplier = position.side === 'LONG' ? 1 : -1;
    let pnl = priceDifference * position.quantity * multiplier;

    // Include fees if configured
    if (this.config.includeFees) {
      pnl -= fees;
    }

    // Convert to base currency if needed
    if (position.currency !== this.config.baseCurrency) {
      pnl = this.convertCurrency(pnl, position.currency, this.config.baseCurrency);
    }

    return pnl;
  }

  /**
   * Calculate realized P&L for a closed position
   */
  public calculateRealizedPnL(
    openPrice: number,
    closePrice: number,
    quantity: number,
    side: 'LONG' | 'SHORT',
    currency: string,
    openFees: number = 0,
    closeFees: number = 0
  ): number {
    const priceDifference = closePrice - openPrice;
    const multiplier = side === 'LONG' ? 1 : -1;
    let pnl = priceDifference * quantity * multiplier;

    // Include fees if configured
    if (this.config.includeFees) {
      pnl -= (openFees + closeFees);
    }

    // Convert to base currency if needed
    if (currency !== this.config.baseCurrency) {
      pnl = this.convertCurrency(pnl, currency, this.config.baseCurrency);
    }

    return pnl;
  }

  /**
   * Calculate mark-to-market valuation for all positions
   */
  public calculateMarkToMarket(
    positions: Position[],
    currentPrices: Map<string, number>
  ): Map<string, number> {
    const markToMarketValues = new Map<string, number>();

    for (const position of positions) {
      const currentPrice = currentPrices.get(position.symbol);
      if (currentPrice !== undefined) {
        const marketValue = currentPrice * position.quantity;
        const convertedValue = position.currency !== this.config.baseCurrency
          ? this.convertCurrency(marketValue, position.currency, this.config.baseCurrency)
          : marketValue;
        
        markToMarketValues.set(position.id, convertedValue);
      }
    }

    return markToMarketValues;
  }

  /**
   * Calculate portfolio-level P&L
   */
  public calculatePortfolioPnL(positions: Position[]): PnLCalculation {
    let totalUnrealizedPnL = 0;
    let totalRealizedPnL = 0;
    let totalExposure = 0;

    for (const position of positions) {
      // Convert P&L to base currency
      const unrealizedPnL = position.currency !== this.config.baseCurrency
        ? this.convertCurrency(position.unrealizedPnl, position.currency, this.config.baseCurrency)
        : position.unrealizedPnl;

      const realizedPnL = position.currency !== this.config.baseCurrency
        ? this.convertCurrency(position.realizedPnl, position.currency, this.config.baseCurrency)
        : position.realizedPnl;

      const exposure = position.currency !== this.config.baseCurrency
        ? this.convertCurrency(position.currentPrice * position.quantity, position.currency, this.config.baseCurrency)
        : position.currentPrice * position.quantity;

      totalUnrealizedPnL += unrealizedPnL;
      totalRealizedPnL += realizedPnL;
      totalExposure += Math.abs(exposure);
    }

    const totalPnL = totalUnrealizedPnL + totalRealizedPnL;
    const pnlPercentage = totalExposure > 0 ? (totalPnL / totalExposure) * 100 : 0;

    // Calculate daily P&L (simplified - would need historical data for accuracy)
    const dailyPnL = totalUnrealizedPnL; // This should be calculated from daily changes

    return {
      unrealizedPnl: totalUnrealizedPnL,
      realizedPnl: totalRealizedPnL,
      totalPnl: totalPnL,
      currency: this.config.baseCurrency,
      dailyPnl: dailyPnL,
      pnlPercentage
    };
  }

  /**
   * Calculate P&L breakdown by position
   */
  public calculatePnLBreakdown(
    positions: Position[],
    totalPortfolioPnL: number
  ): PnLBreakdown[] {
    return positions.map(position => {
      const unrealizedPnL = position.currency !== this.config.baseCurrency
        ? this.convertCurrency(position.unrealizedPnl, position.currency, this.config.baseCurrency)
        : position.unrealizedPnl;

      const realizedPnL = position.currency !== this.config.baseCurrency
        ? this.convertCurrency(position.realizedPnl, position.currency, this.config.baseCurrency)
        : position.realizedPnl;

      const totalPnL = unrealizedPnL + realizedPnL;
      const exposure = position.currentPrice * position.quantity;
      const pnlPercentage = exposure > 0 ? (totalPnL / exposure) * 100 : 0;
      const contribution = totalPortfolioPnL !== 0 ? (totalPnL / totalPortfolioPnL) * 100 : 0;

      return {
        symbol: position.symbol,
        venue: position.venue,
        unrealizedPnL,
        realizedPnL,
        totalPnL,
        pnlPercentage,
        contribution,
        currency: this.config.baseCurrency
      };
    });
  }

  /**
   * Calculate risk metrics
   */
  public calculateRiskMetrics(
    positions: Position[],
    historicalReturns: number[]
  ): RiskMetrics {
    if (historicalReturns.length === 0) {
      return {
        sharpeRatio: 0,
        maxDrawdown: 0,
        volatility: 0,
        var95: 0,
        expectedShortfall: 0
      };
    }

    // Calculate basic statistics
    const mean = historicalReturns.reduce((sum, ret) => sum + ret, 0) / historicalReturns.length;
    const variance = historicalReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / historicalReturns.length;
    const volatility = Math.sqrt(variance);

    // Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    const sharpeRatio = volatility > 0 ? mean / volatility : 0;

    // Maximum drawdown
    const maxDrawdown = this.calculateMaxDrawdown(historicalReturns);

    // Value at Risk (95% confidence)
    const sortedReturns = [...historicalReturns].sort((a, b) => a - b);
    const var95Index = Math.floor(0.05 * sortedReturns.length);
    const var95 = sortedReturns[var95Index] || 0;

    // Expected Shortfall (Conditional VaR)
    const tailReturns = sortedReturns.slice(0, var95Index + 1);
    const expectedShortfall = tailReturns.length > 0
      ? tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length
      : 0;

    return {
      sharpeRatio,
      maxDrawdown,
      volatility,
      var95,
      expectedShortfall
    };
  }

  /**
   * Calculate maximum drawdown
   */
  private calculateMaxDrawdown(returns: number[]): number {
    let maxDrawdown = 0;
    let peak = 0;
    let cumulativeReturn = 0;

    for (const ret of returns) {
      cumulativeReturn += ret;
      peak = Math.max(peak, cumulativeReturn);
      const drawdown = (peak - cumulativeReturn) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    return maxDrawdown;
  }

  /**
   * Add historical P&L data point
   */
  public addHistoricalPnL(date: string, pnl: PnLCalculation): void {
    const lastEntry = this.historicalPnL[this.historicalPnL.length - 1];
    const cumulativePnL = lastEntry ? lastEntry.cumulativePnL + pnl.dailyPnl : pnl.dailyPnl;

    this.historicalPnL.push({
      date,
      dailyPnL: pnl.dailyPnl,
      cumulativePnL,
      realizedPnL: pnl.realizedPnl,
      unrealizedPnL: pnl.unrealizedPnl,
      currency: pnl.currency
    });

    // Keep only last 365 days
    if (this.historicalPnL.length > 365) {
      this.historicalPnL = this.historicalPnL.slice(-365);
    }
  }

  /**
   * Get historical P&L data
   */
  public getHistoricalPnL(days: number = 30): HistoricalPnL[] {
    return this.historicalPnL.slice(-days);
  }

  /**
   * Update currency rates
   */
  public updateCurrencyRates(rates: Map<string, CurrencyConversion>): void {
    this.currencyRates = new Map(rates);
  }

  /**
   * Convert currency using current rates
   * @public - exposed for external use via position service
   */
  public convertCurrency(amount: number, fromCurrency: string, toCurrency: string): number {
    if (fromCurrency === toCurrency) return amount;

    const key = `${fromCurrency}_${toCurrency}`;
    const conversion = this.currencyRates.get(key);

    if (conversion) {
      return amount * conversion.rate;
    }

    // Try reverse conversion
    const reverseKey = `${toCurrency}_${fromCurrency}`;
    const reverseConversion = this.currencyRates.get(reverseKey);

    if (reverseConversion) {
      return amount / reverseConversion.rate;
    }

    console.warn(`No currency conversion rate found for ${fromCurrency} to ${toCurrency}`);
    return amount;
  }

  /**
   * Update configuration
   */
  public updateConfiguration(config: Partial<PnLConfiguration>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  public getConfiguration(): PnLConfiguration {
    return { ...this.config };
  }
}

// Export default engine instance
export const pnlEngine = new PnLCalculationEngine({
  baseCurrency: 'USD',
  includeFees: true,
  useMarkToMarket: true
});