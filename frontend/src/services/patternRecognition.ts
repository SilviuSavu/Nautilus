/**
 * Pattern Recognition Service
 * Implements algorithmic chart pattern detection and analysis
 */

import { OHLCVData } from '../components/Chart/types/chartTypes'
import { ChartPattern, PatternDefinition, PatternRule, Point, PatternCoordinates } from '../types/charting'

export interface PatternDetectionConfig {
  minBars: number
  maxBars: number
  minConfidence: number
  enabledPatterns: string[]
  sensitivity: 'low' | 'medium' | 'high'
}

export interface PatternAlert {
  id: string
  patternId: string
  triggeredAt: string
  alertType: 'formation' | 'completion' | 'breakout'
  confidence: number
  message: string
}

class PatternRecognitionService {
  private patternDefinitions: Map<string, PatternDefinition> = new Map()
  private detectedPatterns: Map<string, ChartPattern> = new Map()
  private patternAlerts: PatternAlert[] = []
  private config: PatternDetectionConfig = {
    minBars: 10,
    maxBars: 100,
    minConfidence: 0.6,
    enabledPatterns: [],
    sensitivity: 'medium'
  }

  constructor() {
    this.initializeBuiltInPatterns()
  }

  // Configuration
  setConfig(config: Partial<PatternDetectionConfig>) {
    this.config = { ...this.config, ...config }
  }

  getConfig(): PatternDetectionConfig {
    return { ...this.config }
  }

  // Pattern definition management
  private initializeBuiltInPatterns() {
    // Head and Shoulders
    this.registerPattern({
      id: 'head_shoulders',
      name: 'Head and Shoulders',
      type: 'head_shoulders',
      rules: [
        { type: 'price_action', condition: 'peak_sequence', parameters: { peaks: 3, order: 'low-high-low' } },
        { type: 'volume', condition: 'decreasing', parameters: { fromPeak: 2 } }
      ],
      minBars: 20,
      maxBars: 80,
      minConfidence: 0.7
    })

    // Double Top
    this.registerPattern({
      id: 'double_top',
      name: 'Double Top',
      type: 'double_top',
      rules: [
        { type: 'price_action', condition: 'double_peak', parameters: { tolerance: 0.02 } },
        { type: 'volume', condition: 'lower_on_second', parameters: {} }
      ],
      minBars: 15,
      maxBars: 60,
      minConfidence: 0.65
    })

    // Double Bottom
    this.registerPattern({
      id: 'double_bottom',
      name: 'Double Bottom',
      type: 'double_bottom',
      rules: [
        { type: 'price_action', condition: 'double_valley', parameters: { tolerance: 0.02 } },
        { type: 'volume', condition: 'higher_on_second', parameters: {} }
      ],
      minBars: 15,
      maxBars: 60,
      minConfidence: 0.65
    })

    // Triangle patterns
    this.registerPattern({
      id: 'ascending_triangle',
      name: 'Ascending Triangle',
      type: 'triangle',
      rules: [
        { type: 'price_action', condition: 'ascending_triangle', parameters: { minTouches: 4 } }
      ],
      minBars: 20,
      maxBars: 100,
      minConfidence: 0.6
    })

    // Cup and Handle
    this.registerPattern({
      id: 'cup_handle',
      name: 'Cup and Handle',
      type: 'cup_handle',
      rules: [
        { type: 'price_action', condition: 'cup_formation', parameters: { depth: 0.15, symmetry: 0.8 } },
        { type: 'price_action', condition: 'handle_formation', parameters: { retracement: 0.5 } }
      ],
      minBars: 30,
      maxBars: 150,
      minConfidence: 0.75
    })

    // Flag patterns
    this.registerPattern({
      id: 'bull_flag',
      name: 'Bull Flag',
      type: 'flag',
      rules: [
        { type: 'price_action', condition: 'strong_move_up', parameters: { minMove: 0.1 } },
        { type: 'price_action', condition: 'flag_consolidation', parameters: { slope: 'down', duration: [5, 20] } }
      ],
      minBars: 10,
      maxBars: 40,
      minConfidence: 0.6
    })

    // Wedge patterns
    this.registerPattern({
      id: 'falling_wedge',
      name: 'Falling Wedge',
      type: 'wedge',
      rules: [
        { type: 'price_action', condition: 'converging_lines', parameters: { direction: 'down' } },
        { type: 'volume', condition: 'decreasing', parameters: {} }
      ],
      minBars: 15,
      maxBars: 80,
      minConfidence: 0.65
    })
  }

  registerPattern(definition: Omit<PatternDefinition, 'id'>): string {
    const id = definition.id || `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    this.patternDefinitions.set(id, { id, ...definition })
    return id
  }

  getPatternDefinitions(): PatternDefinition[] {
    return Array.from(this.patternDefinitions.values())
  }

  // Pattern detection
  async detectPatterns(
    data: OHLCVData[], 
    instrument: string,
    timeframe: string
  ): Promise<ChartPattern[]> {
    if (data.length < this.config.minBars) {
      return []
    }

    const patterns: ChartPattern[] = []
    const enabledDefinitions = Array.from(this.patternDefinitions.values())
      .filter(def => 
        this.config.enabledPatterns.length === 0 || 
        this.config.enabledPatterns.includes(def.id)
      )

    for (const definition of enabledDefinitions) {
      const detectedPatterns = await this.detectPattern(data, definition, instrument, timeframe)
      patterns.push(...detectedPatterns)
    }

    // Filter by confidence and store
    const validPatterns = patterns.filter(p => p.confidence >= this.config.minConfidence)
    validPatterns.forEach(pattern => {
      this.detectedPatterns.set(pattern.id, pattern)
    })

    return validPatterns
  }

  private async detectPattern(
    data: OHLCVData[],
    definition: PatternDefinition,
    instrument: string,
    timeframe: string
  ): Promise<ChartPattern[]> {
    const patterns: ChartPattern[] = []
    const windowSize = Math.min(definition.maxBars, data.length)
    const minSize = Math.max(definition.minBars, 10)

    // Sliding window approach
    for (let i = minSize; i <= windowSize; i++) {
      const window = data.slice(-i)
      const confidence = this.calculatePatternConfidence(window, definition)
      
      if (confidence >= definition.minConfidence) {
        const pattern = await this.createPattern(window, definition, confidence, instrument, timeframe)
        if (pattern) {
          patterns.push(pattern)
        }
      }
    }

    return patterns
  }

  private calculatePatternConfidence(data: OHLCVData[], definition: PatternDefinition): number {
    let totalConfidence = 0
    let ruleCount = 0

    for (const rule of definition.rules) {
      const ruleConfidence = this.evaluateRule(data, rule)
      totalConfidence += ruleConfidence
      ruleCount++
    }

    return ruleCount > 0 ? totalConfidence / ruleCount : 0
  }

  private evaluateRule(data: OHLCVData[], rule: PatternRule): number {
    switch (rule.type) {
      case 'price_action':
        return this.evaluatePriceActionRule(data, rule)
      case 'volume':
        return this.evaluateVolumeRule(data, rule)
      case 'indicator':
        return this.evaluateIndicatorRule(data, rule)
      default:
        return 0
    }
  }

  private evaluatePriceActionRule(data: OHLCVData[], rule: PatternRule): number {
    const { condition, parameters } = rule

    switch (condition) {
      case 'peak_sequence':
        return this.detectPeakSequence(data, parameters)
      case 'double_peak':
        return this.detectDoublePeak(data, parameters)
      case 'double_valley':
        return this.detectDoubleValley(data, parameters)
      case 'ascending_triangle':
        return this.detectAscendingTriangle(data, parameters)
      case 'cup_formation':
        return this.detectCupFormation(data, parameters)
      case 'handle_formation':
        return this.detectHandleFormation(data, parameters)
      case 'strong_move_up':
        return this.detectStrongMove(data, parameters, 'up')
      case 'flag_consolidation':
        return this.detectFlagConsolidation(data, parameters)
      case 'converging_lines':
        return this.detectConvergingLines(data, parameters)
      default:
        return 0
    }
  }

  private evaluateVolumeRule(data: OHLCVData[], rule: PatternRule): number {
    const { condition, parameters } = rule

    switch (condition) {
      case 'decreasing':
        return this.detectDecreasingVolume(data, parameters)
      case 'increasing':
        return this.detectIncreasingVolume(data, parameters)
      case 'lower_on_second':
        return this.detectLowerVolumeOnSecond(data, parameters)
      case 'higher_on_second':
        return this.detectHigherVolumeOnSecond(data, parameters)
      default:
        return 0
    }
  }

  private evaluateIndicatorRule(data: OHLCVData[], rule: PatternRule): number {
    // Placeholder for indicator-based rules
    return 0.5
  }

  // Pattern detection algorithms
  private detectPeakSequence(data: OHLCVData[], params: any): number {
    const peaks = this.findPeaks(data)
    if (peaks.length < 3) return 0

    const lastThree = peaks.slice(-3)
    const [first, second, third] = lastThree.map(p => data[p].high)

    // Check for head and shoulders pattern (low-high-low)
    if (second > first && second > third && Math.abs(first - third) / first < 0.05) {
      const neckline = Math.min(first, third)
      const headHeight = second - neckline
      const symmetry = 1 - Math.abs(first - third) / Math.max(first, third)
      
      return Math.min(0.9, symmetry * 0.8 + (headHeight / second) * 0.2)
    }

    return 0
  }

  private detectDoublePeak(data: OHLCVData[], params: any): number {
    const peaks = this.findPeaks(data)
    if (peaks.length < 2) return 0

    const tolerance = params.tolerance || 0.02
    const lastTwo = peaks.slice(-2)
    const [first, second] = lastTwo.map(p => data[p].high)
    
    const difference = Math.abs(first - second) / Math.max(first, second)
    return difference <= tolerance ? 1 - (difference / tolerance) : 0
  }

  private detectDoubleValley(data: OHLCVData[], params: any): number {
    const valleys = this.findValleys(data)
    if (valleys.length < 2) return 0

    const tolerance = params.tolerance || 0.02
    const lastTwo = valleys.slice(-2)
    const [first, second] = lastTwo.map(p => data[p].low)
    
    const difference = Math.abs(first - second) / Math.max(first, second)
    return difference <= tolerance ? 1 - (difference / tolerance) : 0
  }

  private detectAscendingTriangle(data: OHLCVData[], params: any): number {
    const peaks = this.findPeaks(data)
    const valleys = this.findValleys(data)
    
    if (peaks.length < 2 || valleys.length < 2) return 0

    // Check if peaks are relatively flat (resistance)
    const peakPrices = peaks.map(p => data[p].high)
    const peakVariation = this.calculateVariation(peakPrices)
    
    // Check if valleys are ascending (ascending support)
    const valleyPrices = valleys.map(p => data[p].low)
    const isAscending = this.isAscendingSequence(valleyPrices)
    
    if (peakVariation < 0.03 && isAscending) {
      return 0.8
    }
    
    return 0
  }

  private detectCupFormation(data: OHLCVData[], params: any): number {
    if (data.length < 30) return 0

    const prices = data.map(d => d.close)
    const start = prices[0]
    const end = prices[prices.length - 1]
    const lowest = Math.min(...prices)
    
    const depth = (start - lowest) / start
    const requiredDepth = params.depth || 0.15
    
    if (depth < requiredDepth) return 0

    // Check for U-shape (cup formation)
    const midPoint = Math.floor(prices.length / 2)
    const leftSide = prices.slice(0, midPoint)
    const rightSide = prices.slice(midPoint)
    
    const leftTrend = this.calculateTrend(leftSide)
    const rightTrend = this.calculateTrend(rightSide)
    
    // Left should be declining, right should be inclining
    if (leftTrend < -0.001 && rightTrend > 0.001) {
      const symmetry = 1 - Math.abs(start - end) / Math.max(start, end)
      return Math.min(0.8, symmetry * 0.7 + depth * 0.3)
    }
    
    return 0
  }

  private detectHandleFormation(data: OHLCVData[], params: any): number {
    // Simplified handle detection after cup
    const recentData = data.slice(-10) // Last 10 bars for handle
    if (recentData.length < 5) return 0

    const prices = recentData.map(d => d.close)
    const high = Math.max(...prices)
    const low = Math.min(...prices)
    
    const retracement = (high - low) / high
    const maxRetracement = params.retracement || 0.5
    
    return retracement <= maxRetracement ? 0.7 : 0
  }

  private detectStrongMove(data: OHLCVData[], params: any, direction: 'up' | 'down'): number {
    if (data.length < 5) return 0

    const start = data[0].close
    const end = data[data.length - 1].close
    const move = direction === 'up' ? (end - start) / start : (start - end) / start
    
    const minMove = params.minMove || 0.1
    return move >= minMove ? Math.min(0.9, move / minMove * 0.7) : 0
  }

  private detectFlagConsolidation(data: OHLCVData[], params: any): number {
    const prices = data.map(d => d.close)
    const trend = this.calculateTrend(prices)
    
    const expectedSlope = params.slope === 'down' ? -0.002 : 0.002
    const duration = params.duration || [5, 20]
    
    if (data.length >= duration[0] && data.length <= duration[1]) {
      const slopeFit = Math.abs(trend - expectedSlope) < 0.001 ? 0.8 : 0
      return slopeFit
    }
    
    return 0
  }

  private detectConvergingLines(data: OHLCVData[], params: any): number {
    const peaks = this.findPeaks(data)
    const valleys = this.findValleys(data)
    
    if (peaks.length < 2 || valleys.length < 2) return 0

    const peakTrend = this.calculateTrend(peaks.map(p => data[p].high))
    const valleyTrend = this.calculateTrend(valleys.map(p => data[p].low))
    
    const isConverging = Math.abs(peakTrend - valleyTrend) > 0.001
    const direction = params.direction || 'any'
    
    if (isConverging) {
      if (direction === 'down' && peakTrend < 0 && valleyTrend < 0) {
        return 0.7
      } else if (direction === 'up' && peakTrend > 0 && valleyTrend > 0) {
        return 0.7
      } else if (direction === 'any') {
        return 0.6
      }
    }
    
    return 0
  }

  private detectDecreasingVolume(data: OHLCVData[], params: any): number {
    const volumes = data.map(d => d.volume)
    const trend = this.calculateTrend(volumes)
    return trend < -0.001 ? 0.7 : 0
  }

  private detectIncreasingVolume(data: OHLCVData[], params: any): number {
    const volumes = data.map(d => d.volume)
    const trend = this.calculateTrend(volumes)
    return trend > 0.001 ? 0.7 : 0
  }

  private detectLowerVolumeOnSecond(data: OHLCVData[], params: any): number {
    const peaks = this.findPeaks(data)
    if (peaks.length < 2) return 0

    const lastTwo = peaks.slice(-2)
    const firstVolume = data[lastTwo[0]].volume
    const secondVolume = data[lastTwo[1]].volume

    return secondVolume < firstVolume * 0.8 ? 0.7 : 0
  }

  private detectHigherVolumeOnSecond(data: OHLCVData[], params: any): number {
    const valleys = this.findValleys(data)
    if (valleys.length < 2) return 0

    const lastTwo = valleys.slice(-2)
    const firstVolume = data[lastTwo[0]].volume
    const secondVolume = data[lastTwo[1]].volume

    return secondVolume > firstVolume * 1.2 ? 0.7 : 0
  }

  // Utility methods
  private findPeaks(data: OHLCVData[]): number[] {
    const peaks: number[] = []
    const lookback = 3

    for (let i = lookback; i < data.length - lookback; i++) {
      let isPeak = true
      const current = data[i].high

      // Check if current is higher than surrounding points
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && data[j].high >= current) {
          isPeak = false
          break
        }
      }

      if (isPeak) {
        peaks.push(i)
      }
    }

    return peaks
  }

  private findValleys(data: OHLCVData[]): number[] {
    const valleys: number[] = []
    const lookback = 3

    for (let i = lookback; i < data.length - lookback; i++) {
      let isValley = true
      const current = data[i].low

      // Check if current is lower than surrounding points
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && data[j].low <= current) {
          isValley = false
          break
        }
      }

      if (isValley) {
        valleys.push(i)
      }
    }

    return valleys
  }

  private calculateVariation(values: number[]): number {
    if (values.length === 0) return 0
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    
    return Math.sqrt(variance) / mean
  }

  private isAscendingSequence(values: number[]): boolean {
    for (let i = 1; i < values.length; i++) {
      if (values[i] <= values[i - 1]) {
        return false
      }
    }
    return true
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0

    const n = values.length
    const sumX = (n * (n - 1)) / 2
    const sumY = values.reduce((sum, val) => sum + val, 0)
    const sumXY = values.reduce((sum, val, i) => sum + (i * val), 0)
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6

    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
  }

  private async createPattern(
    data: OHLCVData[],
    definition: PatternDefinition,
    confidence: number,
    instrument: string,
    timeframe: string
  ): Promise<ChartPattern | null> {
    const coordinates = this.calculatePatternCoordinates(data, definition.type)
    if (!coordinates) return null

    const pattern: ChartPattern = {
      id: `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: definition.name,
      type: definition.type,
      confidence,
      coordinates,
      timeframe,
      status: 'forming',
      detectedAt: new Date().toISOString(),
      instrument
    }

    return pattern
  }

  private calculatePatternCoordinates(data: OHLCVData[], type: ChartPattern['type']): PatternCoordinates | null {
    // Simplified coordinate calculation
    const points: Point[] = []
    
    switch (type) {
      case 'head_shoulders':
        const peaks = this.findPeaks(data).slice(-3)
        if (peaks.length === 3) {
          points.push(
            ...peaks.map((peakIndex, i) => ({
              x: peakIndex,
              y: data[peakIndex].high,
              time: data[peakIndex].time
            }))
          )
        }
        break
        
      case 'double_top':
        const doublePeaks = this.findPeaks(data).slice(-2)
        if (doublePeaks.length === 2) {
          points.push(
            ...doublePeaks.map(peakIndex => ({
              x: peakIndex,
              y: data[peakIndex].high,
              time: data[peakIndex].time
            }))
          )
        }
        break
        
      default:
        // Generic pattern coordinates
        points.push(
          { x: 0, y: data[0].close, time: data[0].time },
          { x: data.length - 1, y: data[data.length - 1].close, time: data[data.length - 1].time }
        )
    }

    if (points.length === 0) return null

    return {
      points,
      boundingBox: {
        left: Math.min(...points.map(p => p.x)),
        top: Math.max(...points.map(p => p.y)),
        right: Math.max(...points.map(p => p.x)),
        bottom: Math.min(...points.map(p => p.y))
      }
    }
  }

  // Pattern management
  getDetectedPatterns(instrument?: string, timeframe?: string): ChartPattern[] {
    let patterns = Array.from(this.detectedPatterns.values())
    
    if (instrument) {
      patterns = patterns.filter(p => p.instrument === instrument)
    }
    
    if (timeframe) {
      patterns = patterns.filter(p => p.timeframe === timeframe)
    }
    
    return patterns
  }

  clearPatterns(): void {
    this.detectedPatterns.clear()
  }

  // Historical analysis
  analyzeHistoricalPatterns(
    data: OHLCVData[], 
    instrument: string,
    timeframe: string
  ): Promise<{
    totalPatterns: number
    successRate: number
    avgTimeToComplete: number
    patternPerformance: Record<string, number>
  }> {
    // Placeholder for historical analysis
    return Promise.resolve({
      totalPatterns: 0,
      successRate: 0,
      avgTimeToComplete: 0,
      patternPerformance: {}
    })
  }
}

// Export singleton instance
export const patternRecognition = new PatternRecognitionService()

export default patternRecognition