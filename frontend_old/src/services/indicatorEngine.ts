/**
 * Technical Indicator Engine
 * Provides calculation and management for technical indicators including custom scripting capabilities
 */

import { OHLCVData } from '../components/Chart/types/chartTypes'

// Core indicator types
export interface IndicatorParameter {
  name: string
  type: 'number' | 'string' | 'boolean' | 'color'
  defaultValue: any
  min?: number
  max?: number
  options?: string[]
}

export interface TechnicalIndicator {
  id: string
  name: string
  type: 'built_in' | 'custom' | 'scripted'
  parameters: IndicatorParameter[]
  calculation: {
    script?: string
    function?: (data: OHLCVData[], params: Record<string, any>) => number[]
    period: number
    source: 'close' | 'open' | 'high' | 'low' | 'volume'
  }
  display: {
    color: string
    lineWidth: number
    style: 'solid' | 'dashed' | 'dotted'
    overlay: boolean
  }
  alerts?: AlertCondition[]
}

export interface AlertCondition {
  id: string
  type: 'crossover' | 'crossunder' | 'greater_than' | 'less_than'
  value?: number
  compareIndicatorId?: string
  enabled: boolean
}

export interface IndicatorResult {
  indicatorId: string
  values: Array<{
    time: string
    value: number | null
  }>
  metadata: {
    name: string
    color: string
    lineWidth: number
    style: string
    overlay: boolean
  }
}

class IndicatorEngine {
  private indicators: Map<string, TechnicalIndicator> = new Map()
  private cache: Map<string, IndicatorResult> = new Map()

  constructor() {
    this.initializeBuiltInIndicators()
  }

  // Initialize built-in technical indicators
  private initializeBuiltInIndicators() {
    // Simple Moving Average
    this.registerIndicator({
      id: 'sma',
      name: 'Simple Moving Average',
      type: 'built_in',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 1,
          max: 200
        }
      ],
      calculation: {
        function: this.calculateSMA,
        period: 20,
        source: 'close'
      },
      display: {
        color: '#FF6B6B',
        lineWidth: 2,
        style: 'solid',
        overlay: true
      }
    })

    // Exponential Moving Average
    this.registerIndicator({
      id: 'ema',
      name: 'Exponential Moving Average',
      type: 'built_in',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 1,
          max: 200
        }
      ],
      calculation: {
        function: this.calculateEMA,
        period: 20,
        source: 'close'
      },
      display: {
        color: '#4ECDC4',
        lineWidth: 2,
        style: 'solid',
        overlay: true
      }
    })

    // RSI
    this.registerIndicator({
      id: 'rsi',
      name: 'Relative Strength Index',
      type: 'built_in',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 14,
          min: 2,
          max: 100
        }
      ],
      calculation: {
        function: this.calculateRSI,
        period: 14,
        source: 'close'
      },
      display: {
        color: '#45B7D1',
        lineWidth: 2,
        style: 'solid',
        overlay: false
      }
    })

    // MACD
    this.registerIndicator({
      id: 'macd',
      name: 'MACD',
      type: 'built_in',
      parameters: [
        {
          name: 'fastPeriod',
          type: 'number',
          defaultValue: 12,
          min: 1,
          max: 50
        },
        {
          name: 'slowPeriod',
          type: 'number',
          defaultValue: 26,
          min: 1,
          max: 100
        },
        {
          name: 'signalPeriod',
          type: 'number',
          defaultValue: 9,
          min: 1,
          max: 50
        }
      ],
      calculation: {
        function: this.calculateMACD,
        period: 26,
        source: 'close'
      },
      display: {
        color: '#96CEB4',
        lineWidth: 2,
        style: 'solid',
        overlay: false
      }
    })

    // Bollinger Bands
    this.registerIndicator({
      id: 'bollinger',
      name: 'Bollinger Bands',
      type: 'built_in',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 2,
          max: 100
        },
        {
          name: 'stdDev',
          type: 'number',
          defaultValue: 2,
          min: 0.1,
          max: 5
        }
      ],
      calculation: {
        function: this.calculateBollingerBands,
        period: 20,
        source: 'close'
      },
      display: {
        color: '#FECA57',
        lineWidth: 1,
        style: 'solid',
        overlay: true
      }
    })
  }

  // Register a new indicator
  registerIndicator(indicator: TechnicalIndicator) {
    this.indicators.set(indicator.id, indicator)
    this.cache.delete(this.getCacheKey(indicator.id, {})) // Clear cache
  }

  // Get available indicators
  getAvailableIndicators(): TechnicalIndicator[] {
    return Array.from(this.indicators.values())
  }

  // Get indicator by ID
  getIndicator(id: string): TechnicalIndicator | undefined {
    return this.indicators.get(id)
  }

  // Calculate indicator values
  calculate(
    indicatorId: string,
    data: OHLCVData[],
    parameters: Record<string, any> = {}
  ): IndicatorResult | null {
    const indicator = this.indicators.get(indicatorId)
    if (!indicator) {
      console.warn(`Indicator ${indicatorId} not found`)
      return null
    }

    const cacheKey = this.getCacheKey(indicatorId, parameters)
    const cached = this.cache.get(cacheKey)
    
    // Simple cache invalidation - could be improved with data hash
    if (cached && cached.values.length === data.length) {
      return cached
    }

    try {
      let values: number[]
      
      if (indicator.type === 'scripted' && indicator.calculation.script) {
        values = this.executeScript(indicator.calculation.script, data, parameters)
      } else if (indicator.calculation.function) {
        values = indicator.calculation.function(data, parameters)
      } else {
        throw new Error(`No calculation method available for indicator ${indicatorId}`)
      }

      const result: IndicatorResult = {
        indicatorId,
        values: values.map((value, index) => ({
          time: data[index]?.time || '',
          value: isFinite(value) ? value : null
        })),
        metadata: {
          name: indicator.name,
          color: indicator.display.color,
          lineWidth: indicator.display.lineWidth,
          style: indicator.display.style,
          overlay: indicator.display.overlay
        }
      }

      this.cache.set(cacheKey, result)
      return result
    } catch (error) {
      console.error(`Error calculating indicator ${indicatorId}:`, error)
      return null
    }
  }

  // Execute custom indicator script (sandboxed)
  private executeScript(script: string, data: OHLCVData[], params: Record<string, any>): number[] {
    // Basic sandboxing - in production should use more robust solution
    const context = {
      data,
      params,
      Math,
      console: { log: (...args: any[]) => console.log('[Indicator Script]', ...args) },
      // Helper functions
      sma: (values: number[], period: number) => this.calculateSMAArray(values, period),
      ema: (values: number[], period: number) => this.calculateEMAArray(values, period),
      rsi: (values: number[], period: number) => this.calculateRSIArray(values, period),
    }

    try {
      const func = new Function(
        'data', 'params', 'Math', 'console', 'sma', 'ema', 'rsi',
        `
        "use strict";
        ${script}
        `
      )
      
      const result = func(
        context.data,
        context.params,
        context.Math,
        context.console,
        context.sma,
        context.ema,
        context.rsi
      )

      if (!Array.isArray(result)) {
        throw new Error('Script must return an array of numbers')
      }

      return result
    } catch (error) {
      console.error('Script execution error:', error)
      throw error
    }
  }

  // Built-in indicator calculations
  private calculateSMA = (data: OHLCVData[], params: Record<string, any>): number[] => {
    const period = params.period || 20
    const source = params.source || 'close'
    const values = data.map(d => d[source as keyof OHLCVData] as number)
    return this.calculateSMAArray(values, period)
  }

  private calculateEMA = (data: OHLCVData[], params: Record<string, any>): number[] => {
    const period = params.period || 20
    const source = params.source || 'close'
    const values = data.map(d => d[source as keyof OHLCVData] as number)
    return this.calculateEMAArray(values, period)
  }

  private calculateRSI = (data: OHLCVData[], params: Record<string, any>): number[] => {
    const period = params.period || 14
    const source = params.source || 'close'
    const values = data.map(d => d[source as keyof OHLCVData] as number)
    return this.calculateRSIArray(values, period)
  }

  private calculateMACD = (data: OHLCVData[], params: Record<string, any>): number[] => {
    const fastPeriod = params.fastPeriod || 12
    const slowPeriod = params.slowPeriod || 26
    const source = params.source || 'close'
    const values = data.map(d => d[source as keyof OHLCVData] as number)
    
    const fastEMA = this.calculateEMAArray(values, fastPeriod)
    const slowEMA = this.calculateEMAArray(values, slowPeriod)
    
    return fastEMA.map((fast, i) => 
      isFinite(fast) && isFinite(slowEMA[i]) ? fast - slowEMA[i] : NaN
    )
  }

  private calculateBollingerBands = (data: OHLCVData[], params: Record<string, any>): number[] => {
    const period = params.period || 20
    const stdDev = params.stdDev || 2
    const source = params.source || 'close'
    const values = data.map(d => d[source as keyof OHLCVData] as number)
    
    const sma = this.calculateSMAArray(values, period)
    const stdDevValues: number[] = []
    
    for (let i = 0; i < values.length; i++) {
      if (i < period - 1) {
        stdDevValues.push(NaN)
        continue
      }
      
      const slice = values.slice(i - period + 1, i + 1)
      const mean = sma[i]
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period
      stdDevValues.push(Math.sqrt(variance))
    }
    
    // Return upper band - could be extended to return middle and lower bands
    return sma.map((ma, i) => 
      isFinite(ma) && isFinite(stdDevValues[i]) ? ma + (stdDev * stdDevValues[i]) : NaN
    )
  }

  // Helper calculation methods
  private calculateSMAArray(values: number[], period: number): number[] {
    const result: number[] = []
    for (let i = 0; i < values.length; i++) {
      if (i < period - 1) {
        result.push(NaN)
      } else {
        const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0)
        result.push(sum / period)
      }
    }
    return result
  }

  private calculateEMAArray(values: number[], period: number): number[] {
    const result: number[] = []
    const multiplier = 2 / (period + 1)
    
    for (let i = 0; i < values.length; i++) {
      if (i === 0) {
        result.push(values[0])
      } else {
        const ema = (values[i] * multiplier) + (result[i - 1] * (1 - multiplier))
        result.push(ema)
      }
    }
    return result
  }

  private calculateRSIArray(values: number[], period: number): number[] {
    const result: number[] = []
    const gains: number[] = []
    const losses: number[] = []
    
    for (let i = 1; i < values.length; i++) {
      const change = values[i] - values[i - 1]
      gains.push(change > 0 ? change : 0)
      losses.push(change < 0 ? -change : 0)
    }
    
    for (let i = 0; i < values.length; i++) {
      if (i < period) {
        result.push(NaN)
      } else {
        const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period
        const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period
        
        if (avgLoss === 0) {
          result.push(100)
        } else {
          const rs = avgGain / avgLoss
          const rsi = 100 - (100 / (1 + rs))
          result.push(rsi)
        }
      }
    }
    
    return result
  }

  private getCacheKey(indicatorId: string, parameters: Record<string, any>): string {
    return `${indicatorId}_${JSON.stringify(parameters)}`
  }

  // Clear cache
  clearCache() {
    this.cache.clear()
  }

  // Create custom indicator
  createCustomIndicator(
    name: string,
    script: string,
    parameters: IndicatorParameter[],
    display: TechnicalIndicator['display']
  ): string {
    const id = `custom_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    const indicator: TechnicalIndicator = {
      id,
      name,
      type: 'scripted',
      parameters,
      calculation: {
        script,
        period: 1,
        source: 'close'
      },
      display
    }
    
    this.registerIndicator(indicator)
    return id
  }
}

// Export singleton instance
export const indicatorEngine = new IndicatorEngine()

// Export types and classes for external use
export { IndicatorEngine }
export default indicatorEngine