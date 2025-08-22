/**
 * Backtest Service for managing backtest operations and progress monitoring
 * Integrates with existing WebSocket service for real-time updates
 */

import { webSocketService, WebSocketMessage } from './websocket'

export interface BacktestConfig {
  strategyClass: string
  strategyConfig: Record<string, any>
  startDate: string
  endDate: string
  instruments: string[]
  venues: string[]
  initialBalance: number
  baseCurrency: string
  dataConfiguration: {
    dataType: 'tick' | 'bar'
    barType?: string
    resolution?: string
    dataQuality?: 'raw' | 'cleaned' | 'validated'
  }
  executionSettings?: {
    commissionModel?: string
    slippageModel?: string
    fillModel?: string
  }
  riskSettings?: {
    positionSizing?: string
    leverageLimit?: number
    maxPortfolioRisk?: number
  }
}

export interface BacktestProgressData {
  percentage: number
  currentDate: string
  processedBars: number
  totalBars: number
  estimatedTimeRemaining: number
  processingRate: number
  statistics: {
    tradesExecuted: number
    currentBalance: number
    unrealizedPnl: number
    drawdown: number
  }
  performance: {
    memoryUsage: number
    cpuUsage: number
    diskIO: number
  }
}

export interface BacktestErrorData {
  errorType: 'validation' | 'execution' | 'data' | 'timeout' | 'memory'
  errorMessage: string
  errorDetails?: {
    stackTrace?: string
    failedAtDate?: string
    failedInstrument?: string
    recoverable: boolean
  }
}

export interface BacktestCompleteData {
  status: 'completed' | 'cancelled' | 'failed'
  executionTime: number
  summary: {
    totalTrades: number
    finalBalance: number
    totalReturn: number
    maxDrawdown: number
  }
  resultsUrl: string
}

export interface BacktestResult {
  backtestId: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress?: BacktestProgressData
  error?: BacktestErrorData
  completion?: BacktestCompleteData
  startTime?: string
  endTime?: string
  config?: BacktestConfig
  metrics?: PerformanceMetrics
  trades?: TradeResult[]
  equityCurve?: EquityPoint[]
}

export interface PerformanceMetrics {
  totalReturn: number
  annualizedReturn: number
  sharpeRatio: number
  sortinoRatio: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  volatility: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
  averageWin: number
  averageLoss: number
  largestWin: number
  largestLoss: number
  calmarRatio: number
  informationRatio: number
  alpha: number
  beta: number
}

export interface TradeResult {
  tradeId: string
  instrumentId: string
  side: 'buy' | 'sell'
  quantity: number
  entryPrice: number
  exitPrice: number
  entryTime: string
  exitTime: string
  pnl: number
  commission: number
  duration: number
  tags?: string[]
}

export interface EquityPoint {
  timestamp: string
  equity: number
  drawdown: number
  balance: number
  unrealizedPnl: number
}

export interface BacktestValidationResult {
  isValid: boolean
  errors: string[]
  warnings?: string[]
}

export class BacktestConfigValidator {
  validateDateRange(config: BacktestConfig): BacktestValidationResult {
    const errors: string[] = []
    const startDate = new Date(config.startDate)
    const endDate = new Date(config.endDate)
    
    if (startDate >= endDate) {
      errors.push('Start date must be before end date')
    }
    
    if (startDate > new Date()) {
      errors.push('Start date cannot be in the future')
    }
    
    const duration = endDate.getTime() - startDate.getTime()
    const days = duration / (1000 * 60 * 60 * 24)
    
    if (days < 1) {
      errors.push('Backtest duration must be at least 1 day')
    }
    
    if (days > 1825) { // 5 years
      errors.push('Backtest duration cannot exceed 5 years')
    }
    
    return { isValid: errors.length === 0, errors }
  }
  
  validateInstruments(config: BacktestConfig): BacktestValidationResult {
    const errors: string[] = []
    
    if (config.instruments.length === 0) {
      errors.push('At least one instrument must be selected')
    }
    
    if (config.instruments.length > 50) {
      errors.push('Maximum 50 instruments allowed per backtest')
    }
    
    // Validate instrument symbols format
    const invalidSymbols = config.instruments.filter(symbol => 
      !/^[A-Z]{1,6}(\.[A-Z]{1,4})?$/.test(symbol)
    )
    
    if (invalidSymbols.length > 0) {
      errors.push(`Invalid instrument symbols: ${invalidSymbols.join(', ')}`)
    }
    
    return { isValid: errors.length === 0, errors }
  }
  
  validateCapital(config: BacktestConfig): BacktestValidationResult {
    const errors: string[] = []
    
    if (config.initialBalance < 1000) {
      errors.push('Initial balance must be at least $1,000')
    }
    
    if (config.initialBalance > 100000000) {
      errors.push('Initial balance cannot exceed $100,000,000')
    }
    
    const supportedCurrencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
    if (!supportedCurrencies.includes(config.baseCurrency)) {
      errors.push(`Unsupported currency: ${config.baseCurrency}`)
    }
    
    return { isValid: errors.length === 0, errors }
  }
  
  validateStrategy(config: BacktestConfig): BacktestValidationResult {
    const errors: string[] = []
    
    if (!config.strategyClass || config.strategyClass.trim() === '') {
      errors.push('Strategy class is required')
    }
    
    // Basic strategy config validation
    if (!config.strategyConfig || Object.keys(config.strategyConfig).length === 0) {
      errors.push('Strategy configuration is required')
    }
    
    return { isValid: errors.length === 0, errors }
  }
  
  validateComplete(config: BacktestConfig): BacktestValidationResult {
    const dateValidation = this.validateDateRange(config)
    const instrumentsValidation = this.validateInstruments(config)
    const capitalValidation = this.validateCapital(config)
    const strategyValidation = this.validateStrategy(config)
    
    const allErrors = [
      ...dateValidation.errors,
      ...instrumentsValidation.errors,
      ...capitalValidation.errors,
      ...strategyValidation.errors
    ]
    
    return {
      isValid: allErrors.length === 0,
      errors: allErrors
    }
  }
}

export class BacktestProgressMonitor {
  private callbacks: Map<string, (data: BacktestProgressData | BacktestErrorData | BacktestCompleteData) => void>
  private messageHandler: (message: WebSocketMessage) => void
  
  constructor() {
    this.callbacks = new Map()
    this.messageHandler = this.handleWebSocketMessage.bind(this)
    webSocketService.addMessageHandler(this.messageHandler)
  }
  
  subscribeToBacktest(backtestId: string, callback: (data: BacktestProgressData | BacktestErrorData | BacktestCompleteData) => void) {
    this.callbacks.set(backtestId, callback)
    
    // Send subscription message through existing WebSocket
    webSocketService.send({
      type: 'subscribe_backtest_progress',
      backtestId: backtestId
    })
  }
  
  unsubscribeFromBacktest(backtestId: string) {
    this.callbacks.delete(backtestId)
    
    webSocketService.send({
      type: 'unsubscribe_backtest_progress',
      backtestId: backtestId
    })
  }
  
  private handleWebSocketMessage(message: WebSocketMessage) {
    if (message.type?.startsWith('backtest_')) {
      const backtestId = message.payload?.backtestId
      const callback = this.callbacks.get(backtestId)
      
      if (callback && message.payload) {
        switch (message.type) {
          case 'backtest_progress':
            callback(message.payload as BacktestProgressData)
            break
          case 'backtest_error':
            callback(message.payload as BacktestErrorData)
            break
          case 'backtest_complete':
            callback(message.payload as BacktestCompleteData)
            // Auto-unsubscribe on completion
            this.unsubscribeFromBacktest(backtestId)
            break
        }
      }
    }
  }
  
  destroy() {
    webSocketService.removeMessageHandler(this.messageHandler)
    this.callbacks.clear()
  }
}

export class BacktestService {
  private apiUrl: string
  private validator: BacktestConfigValidator
  
  constructor() {
    this.apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'
    this.validator = new BacktestConfigValidator()
  }
  
  validateConfig(config: BacktestConfig): BacktestValidationResult {
    return this.validator.validateComplete(config)
  }
  
  async startBacktest(config: BacktestConfig): Promise<BacktestResult> {
    // Validate configuration before sending
    const validation = this.validateConfig(config)
    if (!validation.isValid) {
      throw new Error(`Configuration validation failed: ${validation.errors.join(', ')}`)
    }
    
    const backtestId = this.generateBacktestId()
    
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/engine/backtest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify({
        backtest_id: backtestId,
        config: config
      })
    })
    
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to start backtest')
    }
    
    const result = await response.json()
    
    return {
      backtestId: result.backtest_id,
      status: result.status || 'queued',
      config: config
    }
  }
  
  async getBacktestStatus(backtestId: string): Promise<BacktestResult> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/engine/backtest/${backtestId}`, {
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to get backtest status')
    }
    
    const result = await response.json()
    return result.backtest
  }
  
  async cancelBacktest(backtestId: string): Promise<boolean> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/engine/backtest/${backtestId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to cancel backtest')
    }
    
    const result = await response.json()
    return result.success
  }
  
  async listBacktests(): Promise<BacktestResult[]> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/engine/backtests`, {
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to list backtests')
    }
    
    const result = await response.json()
    return result.backtests || []
  }
  
  async getBacktestResults(backtestId: string): Promise<BacktestResult> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/backtest/results/${backtestId}`, {
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    })
    
    if (!response.ok) {
      throw new Error('Failed to get backtest results')
    }
    
    return await response.json()
  }
  
  async compareBacktests(backtestIds: string[]): Promise<any> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/backtest/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.getAuthToken()}`
      },
      body: JSON.stringify({ backtest_ids: backtestIds })
    })
    
    if (!response.ok) {
      throw new Error('Failed to compare backtests')
    }
    
    return await response.json()
  }
  
  async deleteBacktest(backtestId: string): Promise<boolean> {
    const response = await fetch(`${this.apiUrl}/api/v1/nautilus/backtest/${backtestId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    })
    
    return response.ok
  }
  
  private generateBacktestId(): string {
    const timestamp = Date.now()
    const random = Math.random().toString(36).substring(2, 8)
    return `backtest-${timestamp}-${random}`
  }
  
  private getAuthToken(): string {
    // In a real implementation, this would get the token from auth service
    return localStorage.getItem('auth_token') || ''
  }
}

// Export singleton instance
export const backtestService = new BacktestService()
export const backtestValidator = new BacktestConfigValidator()