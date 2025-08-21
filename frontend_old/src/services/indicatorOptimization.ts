/**
 * Indicator Optimization and Backtesting Service
 * Provides optimization algorithms and backtesting capabilities for technical indicators
 */

import { OHLCVData } from '../components/Chart/types/chartTypes'
import { indicatorEngine, TechnicalIndicator, IndicatorResult } from './indicatorEngine'

export interface OptimizationResult {
  bestParameters: Record<string, any>
  bestScore: number
  allResults: OptimizationTestResult[]
  executionTime: number
  totalTests: number
}

export interface OptimizationTestResult {
  parameters: Record<string, any>
  score: number
  metrics: BacktestMetrics
}

export interface BacktestMetrics {
  totalReturn: number
  winRate: number
  profitFactor: number
  maxDrawdown: number
  sharpeRatio: number
  totalTrades: number
  avgTradeReturn: number
  consecutiveWins: number
  consecutiveLosses: number
}

export interface BacktestConfig {
  indicatorId: string
  data: OHLCVData[]
  parameters: Record<string, any>
  strategy: TradingStrategy
  initialCapital: number
  transactionCost: number
  riskFreeRate: number
}

export interface TradingStrategy {
  type: 'crossover' | 'threshold' | 'divergence' | 'custom'
  buyCondition: (indicatorValue: number, price: number, index: number) => boolean
  sellCondition: (indicatorValue: number, price: number, index: number) => boolean
  stopLoss?: number
  takeProfit?: number
}

export interface Trade {
  type: 'buy' | 'sell'
  price: number
  timestamp: string
  indicatorValue: number
  index: number
}

export interface Position {
  entryPrice: number
  entryTime: string
  entryIndex: number
  size: number
  unrealizedPnL: number
}

class IndicatorOptimizationService {
  
  /**
   * Optimize indicator parameters using grid search
   */
  async optimizeParameters(
    indicatorId: string,
    data: OHLCVData[],
    parameterRanges: Record<string, { min: number; max: number; step: number }>,
    strategy: TradingStrategy,
    config: {
      initialCapital?: number
      transactionCost?: number
      riskFreeRate?: number
      scoringMetric?: 'sharpe' | 'return' | 'profitFactor' | 'winRate'
    } = {}
  ): Promise<OptimizationResult> {
    const startTime = Date.now()
    const {
      initialCapital = 10000,
      transactionCost = 0.001,
      riskFreeRate = 0.02,
      scoringMetric = 'sharpe'
    } = config

    const parameterCombinations = this.generateParameterCombinations(parameterRanges)
    const results: OptimizationTestResult[] = []

    for (const parameters of parameterCombinations) {
      const backtest = await this.runBacktest({
        indicatorId,
        data,
        parameters,
        strategy,
        initialCapital,
        transactionCost,
        riskFreeRate
      })

      const score = this.calculateScore(backtest, scoringMetric)
      
      results.push({
        parameters,
        score,
        metrics: backtest
      })
    }

    // Sort by score (higher is better)
    results.sort((a, b) => b.score - a.score)
    
    const executionTime = Date.now() - startTime

    return {
      bestParameters: results[0]?.parameters || {},
      bestScore: results[0]?.score || 0,
      allResults: results,
      executionTime,
      totalTests: results.length
    }
  }

  /**
   * Run backtest for specific indicator parameters
   */
  async runBacktest(config: BacktestConfig): Promise<BacktestMetrics> {
    const {
      indicatorId,
      data,
      parameters,
      strategy,
      initialCapital,
      transactionCost,
      riskFreeRate
    } = config

    // Calculate indicator values
    const indicatorResult = indicatorEngine.calculate(indicatorId, data, parameters)
    if (!indicatorResult) {
      throw new Error(`Failed to calculate indicator ${indicatorId}`)
    }

    const trades: Trade[] = []
    let position: Position | null = null
    let capital = initialCapital
    let equity = initialCapital
    const equityCurve: number[] = []
    let maxEquity = initialCapital
    let maxDrawdown = 0

    // Simulate trading
    for (let i = 1; i < data.length; i++) {
      const price = data[i].close
      const indicatorValue = indicatorResult.values[i]?.value
      
      if (indicatorValue === null || indicatorValue === undefined) {
        equityCurve.push(equity)
        continue
      }

      // Check for buy signal
      if (!position && strategy.buyCondition(indicatorValue, price, i)) {
        const size = Math.floor((capital * 0.95) / price) // Use 95% of capital, leave some for costs
        const cost = size * price * transactionCost
        
        if (size > 0 && (size * price + cost) <= capital) {
          position = {
            entryPrice: price,
            entryTime: data[i].time,
            entryIndex: i,
            size,
            unrealizedPnL: 0
          }
          
          capital -= (size * price + cost)
          
          trades.push({
            type: 'buy',
            price,
            timestamp: data[i].time,
            indicatorValue,
            index: i
          })
        }
      }
      
      // Check for sell signal
      else if (position && strategy.sellCondition(indicatorValue, price, i)) {
        const proceeds = position.size * price
        const cost = proceeds * transactionCost
        
        capital += (proceeds - cost)
        
        trades.push({
          type: 'sell',
          price,
          timestamp: data[i].time,
          indicatorValue,
          index: i
        })
        
        position = null
      }

      // Update equity
      if (position) {
        const unrealizedValue = position.size * price
        equity = capital + unrealizedValue
        position.unrealizedPnL = unrealizedValue - (position.size * position.entryPrice)
      } else {
        equity = capital
      }

      equityCurve.push(equity)
      
      // Track drawdown
      if (equity > maxEquity) {
        maxEquity = equity
      } else {
        const drawdown = (maxEquity - equity) / maxEquity
        maxDrawdown = Math.max(maxDrawdown, drawdown)
      }
    }

    // Close any remaining position
    if (position && data.length > 0) {
      const finalPrice = data[data.length - 1].close
      const proceeds = position.size * finalPrice
      const cost = proceeds * transactionCost
      capital += (proceeds - cost)
      equity = capital
    }

    return this.calculateBacktestMetrics(trades, equityCurve, initialCapital, riskFreeRate)
  }

  /**
   * Generate all parameter combinations for optimization
   */
  private generateParameterCombinations(
    ranges: Record<string, { min: number; max: number; step: number }>
  ): Record<string, any>[] {
    const parameterNames = Object.keys(ranges)
    const combinations: Record<string, any>[] = []

    const generateRecursive = (index: number, current: Record<string, any>) => {
      if (index >= parameterNames.length) {
        combinations.push({ ...current })
        return
      }

      const paramName = parameterNames[index]
      const range = ranges[paramName]
      
      for (let value = range.min; value <= range.max; value += range.step) {
        current[paramName] = value
        generateRecursive(index + 1, current)
      }
    }

    generateRecursive(0, {})
    return combinations
  }

  /**
   * Calculate backtest performance metrics
   */
  private calculateBacktestMetrics(
    trades: Trade[],
    equityCurve: number[],
    initialCapital: number,
    riskFreeRate: number
  ): BacktestMetrics {
    const finalEquity = equityCurve[equityCurve.length - 1] || initialCapital
    const totalReturn = (finalEquity - initialCapital) / initialCapital

    // Calculate trade-based metrics
    const buyTrades = trades.filter(t => t.type === 'buy')
    const sellTrades = trades.filter(t => t.type === 'sell')
    const completedTrades = Math.min(buyTrades.length, sellTrades.length)
    
    let wins = 0
    let losses = 0
    let totalProfits = 0
    let totalLosses = 0
    let consecutiveWins = 0
    let consecutiveLosses = 0
    let maxConsecutiveWins = 0
    let maxConsecutiveLosses = 0

    for (let i = 0; i < completedTrades; i++) {
      const buyPrice = buyTrades[i].price
      const sellPrice = sellTrades[i].price
      const tradeReturn = (sellPrice - buyPrice) / buyPrice

      if (tradeReturn > 0) {
        wins++
        totalProfits += tradeReturn
        consecutiveWins++
        consecutiveLosses = 0
        maxConsecutiveWins = Math.max(maxConsecutiveWins, consecutiveWins)
      } else {
        losses++
        totalLosses += Math.abs(tradeReturn)
        consecutiveLosses++
        consecutiveWins = 0
        maxConsecutiveLosses = Math.max(maxConsecutiveLosses, consecutiveLosses)
      }
    }

    const winRate = completedTrades > 0 ? wins / completedTrades : 0
    const profitFactor = totalLosses > 0 ? totalProfits / totalLosses : totalProfits > 0 ? Infinity : 0
    const avgTradeReturn = completedTrades > 0 ? totalReturn / completedTrades : 0

    // Calculate Sharpe ratio
    let sharpeRatio = 0
    if (equityCurve.length > 1) {
      const returns = []
      for (let i = 1; i < equityCurve.length; i++) {
        returns.push((equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1])
      }
      
      const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length
      const returnStd = Math.sqrt(
        returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
      )
      
      if (returnStd > 0) {
        const annualizedReturn = avgReturn * 252 // Assuming daily returns
        const annualizedStd = returnStd * Math.sqrt(252)
        sharpeRatio = (annualizedReturn - riskFreeRate) / annualizedStd
      }
    }

    // Calculate max drawdown
    let maxDrawdown = 0
    let peak = initialCapital
    for (const equity of equityCurve) {
      if (equity > peak) {
        peak = equity
      }
      const drawdown = (peak - equity) / peak
      maxDrawdown = Math.max(maxDrawdown, drawdown)
    }

    return {
      totalReturn,
      winRate,
      profitFactor,
      maxDrawdown,
      sharpeRatio,
      totalTrades: completedTrades,
      avgTradeReturn,
      consecutiveWins: maxConsecutiveWins,
      consecutiveLosses: maxConsecutiveLosses
    }
  }

  /**
   * Calculate optimization score based on selected metric
   */
  private calculateScore(metrics: BacktestMetrics, scoringMetric: string): number {
    switch (scoringMetric) {
      case 'sharpe':
        return metrics.sharpeRatio
      case 'return':
        return metrics.totalReturn
      case 'profitFactor':
        return metrics.profitFactor
      case 'winRate':
        return metrics.winRate
      default:
        return metrics.sharpeRatio
    }
  }

  /**
   * Create common trading strategies
   */
  static createCrossoverStrategy(
    fastIndicatorValue: number,
    slowIndicatorValue: number
  ): TradingStrategy {
    return {
      type: 'crossover',
      buyCondition: (indicatorValue, price, index) => {
        // This would need historical values to detect crossover
        return false // Placeholder
      },
      sellCondition: (indicatorValue, price, index) => {
        return false // Placeholder
      }
    }
  }

  static createThresholdStrategy(
    buyThreshold: number,
    sellThreshold: number,
    mode: 'above_below' | 'below_above' = 'below_above'
  ): TradingStrategy {
    return {
      type: 'threshold',
      buyCondition: (indicatorValue, price, index) => {
        return mode === 'below_above' ? 
          indicatorValue < buyThreshold : 
          indicatorValue > buyThreshold
      },
      sellCondition: (indicatorValue, price, index) => {
        return mode === 'below_above' ? 
          indicatorValue > sellThreshold : 
          indicatorValue < sellThreshold
      }
    }
  }

  static createRSIStrategy(oversoldLevel = 30, overboughtLevel = 70): TradingStrategy {
    return {
      type: 'threshold',
      buyCondition: (rsiValue, price, index) => rsiValue < oversoldLevel,
      sellCondition: (rsiValue, price, index) => rsiValue > overboughtLevel
    }
  }
}

// Export singleton instance
export const indicatorOptimization = new IndicatorOptimizationService()

export default indicatorOptimization