/**
 * Position Management Service for real-time position tracking
 */

import { 
  Position, 
  PositionUpdate, 
  AccountBalance, 
  AccountUpdate, 
  PositionSummary, 
  PnLCalculation,
  CurrencyConversion,
  PositionAlert,
  PositionRisk
} from '../types/position';
import { WebSocketMessage, webSocketService } from './websocket';
import { pnlEngine, PnLBreakdown } from './pnlCalculationEngine';

export class PositionService {
  private positions: Map<string, Position> = new Map();
  private accountBalances: Map<string, AccountBalance> = new Map();
  private currencyRates: Map<string, CurrencyConversion> = new Map();
  private alerts: PositionAlert[] = [];
  
  private positionHandlers: Set<(positions: Position[]) => void> = new Set();
  private accountHandlers: Set<(accounts: AccountBalance[]) => void> = new Set();
  private summaryHandlers: Set<(summary: PositionSummary) => void> = new Set();
  private alertHandlers: Set<(alerts: PositionAlert[]) => void> = new Set();

  constructor() {
    this.initializeWebSocketHandlers();
  }

  /**
   * Initialize WebSocket message handlers for position and account updates
   */
  private initializeWebSocketHandlers(): void {
    webSocketService.addMessageHandler(this.handleWebSocketMessage.bind(this));
  }

  /**
   * Handle incoming WebSocket messages for position and account updates
   */
  private handleWebSocketMessage(message: WebSocketMessage): void {
    try {
      switch (message.type) {
        case 'position_update':
          this.handlePositionUpdate(message as PositionUpdate);
          break;
        case 'account_update':
          this.handleAccountUpdate(message as AccountUpdate);
          break;
        case 'market_data':
          this.handleMarketDataUpdate(message);
          break;
        case 'currency_rate':
          this.handleCurrencyRateUpdate(message);
          break;
        default:
          // Check if it's a NautilusTrader message type
          if (message.message_type?.includes('Position') || message.message_type?.includes('Account')) {
            this.handleNautilusMessage(message);
          }
          break;
      }
    } catch (error) {
      console.error('Error handling position message:', error);
    }
  }

  /**
   * Handle NautilusTrader specific position/account messages
   */
  private handleNautilusMessage(message: WebSocketMessage): void {
    const { message_type, payload } = message;
    
    if (message_type?.includes('PositionOpened') || message_type?.includes('PositionChanged')) {
      this.processNautilusPosition(payload);
    } else if (message_type?.includes('AccountState')) {
      this.processNautilusAccount(payload);
    }
  }

  /**
   * Process NautilusTrader position data
   */
  private processNautilusPosition(payload: any): void {
    if (!payload) return;

    const position: Position = {
      id: payload.position_id || `${payload.symbol}_${payload.venue}`,
      symbol: payload.symbol || payload.instrument_id?.symbol,
      venue: payload.venue || payload.instrument_id?.venue,
      side: payload.side === 'BUY' || payload.quantity > 0 ? 'LONG' : 'SHORT',
      quantity: Math.abs(payload.quantity || payload.size || 0),
      averagePrice: payload.avg_px_open || payload.average_price || 0,
      unrealizedPnl: payload.unrealized_pnl || 0,
      realizedPnl: payload.realized_pnl || 0,
      currentPrice: payload.last_price || payload.current_price || 0,
      currency: payload.settlement_currency || payload.currency || 'USD',
      timestamp: Date.now(),
      openTimestamp: payload.ts_opened ? new Date(payload.ts_opened / 1000000).getTime() : Date.now()
    };

    this.updatePosition(position);
  }

  /**
   * Process NautilusTrader account data
   */
  private processNautilusAccount(payload: any): void {
    if (!payload) return;

    const account: AccountBalance = {
      currency: payload.base_currency || payload.currency || 'USD',
      balance: payload.balance || 0,
      availableBalance: payload.free_balance || payload.available || 0,
      marginUsed: payload.margin_used || 0,
      marginAvailable: payload.margin_available || 0,
      buyingPower: payload.buying_power || 0,
      equity: payload.equity || payload.balance || 0,
      timestamp: Date.now()
    };

    this.updateAccount(account);
  }

  /**
   * Handle position updates
   */
  private handlePositionUpdate(update: PositionUpdate): void {
    this.updatePosition(update.position);
  }

  /**
   * Handle account updates
   */
  private handleAccountUpdate(update: AccountUpdate): void {
    this.updateAccount(update.account);
  }

  /**
   * Handle market data updates to recalculate unrealized P&L
   */
  private handleMarketDataUpdate(message: WebSocketMessage): void {
    const { symbol, last } = message;
    
    if (symbol && last) {
      this.updatePositionPrices(symbol, last);
    }
  }

  /**
   * Handle currency rate updates
   */
  private handleCurrencyRateUpdate(message: WebSocketMessage): void {
    if (message.payload) {
      const conversion: CurrencyConversion = message.payload;
      const key = `${conversion.fromCurrency}_${conversion.toCurrency}`;
      this.currencyRates.set(key, conversion);
    }
  }

  /**
   * Update position data and notify handlers
   */
  private updatePosition(position: Position): void {
    this.positions.set(position.id, position);
    this.checkPositionAlerts(position);
    this.notifyPositionHandlers();
    this.notifySummaryHandlers();
  }

  /**
   * Update account data and notify handlers
   */
  private updateAccount(account: AccountBalance): void {
    this.accountBalances.set(account.currency, account);
    this.notifyAccountHandlers();
    this.notifySummaryHandlers();
  }

  /**
   * Update position prices from market data
   */
  private updatePositionPrices(symbol: string, currentPrice: number): void {
    let updated = false;
    
    for (const [id, position] of this.positions.entries()) {
      if (position.symbol === symbol) {
        const updatedPosition = {
          ...position,
          currentPrice,
          unrealizedPnl: this.calculateUnrealizedPnl(position, currentPrice),
          timestamp: Date.now()
        };
        
        this.positions.set(id, updatedPosition);
        updated = true;
      }
    }
    
    if (updated) {
      this.notifyPositionHandlers();
      this.notifySummaryHandlers();
    }
  }

  /**
   * Calculate unrealized P&L for a position using the P&L engine
   */
  private calculateUnrealizedPnl(position: Position, currentPrice: number): number {
    return pnlEngine.calculateUnrealizedPnL(position, currentPrice);
  }

  /**
   * Check for position alerts
   */
  private checkPositionAlerts(position: Position): void {
    // Price change alerts
    const priceChangePercent = Math.abs((position.currentPrice - position.averagePrice) / position.averagePrice) * 100;
    
    if (priceChangePercent > 5) {
      this.addAlert({
        id: `price_${position.id}_${Date.now()}`,
        positionId: position.id,
        type: 'price_change',
        message: `${position.symbol} price changed by ${priceChangePercent.toFixed(2)}%`,
        severity: priceChangePercent > 10 ? 'warning' : 'info',
        timestamp: Date.now(),
        acknowledged: false
      });
    }

    // P&L threshold alerts
    const pnlPercent = (position.unrealizedPnl / (position.averagePrice * position.quantity)) * 100;
    
    if (Math.abs(pnlPercent) > 5) {
      this.addAlert({
        id: `pnl_${position.id}_${Date.now()}`,
        positionId: position.id,
        type: 'pnl_threshold',
        message: `${position.symbol} P&L at ${pnlPercent.toFixed(2)}%`,
        severity: Math.abs(pnlPercent) > 10 ? 'critical' : 'warning',
        timestamp: Date.now(),
        acknowledged: false
      });
    }
  }

  /**
   * Add position alert
   */
  private addAlert(alert: PositionAlert): void {
    this.alerts.push(alert);
    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }
    this.notifyAlertHandlers();
  }

  /**
   * Calculate position summary using the P&L engine
   */
  private calculatePositionSummary(): PositionSummary {
    const positions = Array.from(this.positions.values());
    
    const longPositions = positions.filter(p => p.side === 'LONG').length;
    const shortPositions = positions.filter(p => p.side === 'SHORT').length;
    
    // Use P&L engine for accurate calculations
    const pnl = pnlEngine.calculatePortfolioPnL(positions);
    
    const totalExposure = positions.reduce((sum, p) => sum + (p.currentPrice * p.quantity), 0);
    
    const netExposure = positions.reduce((sum, p) => {
      const exposure = p.currentPrice * p.quantity;
      return sum + (p.side === 'LONG' ? exposure : -exposure);
    }, 0);
    
    const grossExposure = positions.reduce((sum, p) => {
      return sum + Math.abs(p.currentPrice * p.quantity);
    }, 0);

    return {
      totalPositions: positions.length,
      longPositions,
      shortPositions,
      totalExposure,
      netExposure,
      grossExposure,
      pnl,
      currency: pnl.currency
    };
  }

  /**
   * Get all current positions
   */
  public getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get all account balances
   */
  public getAccountBalances(): AccountBalance[] {
    return Array.from(this.accountBalances.values());
  }

  /**
   * Get position summary
   */
  public getPositionSummary(): PositionSummary {
    return this.calculatePositionSummary();
  }

  /**
   * Get active alerts
   */
  public getAlerts(): PositionAlert[] {
    return this.alerts.filter(alert => !alert.acknowledged);
  }

  /**
   * Acknowledge alert
   */
  public acknowledgeAlert(alertId: string): void {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      this.notifyAlertHandlers();
    }
  }

  /**
   * Convert currency using current rates
   */
  public convertCurrency(amount: number, fromCurrency: string, toCurrency: string): number {
    return pnlEngine.convertCurrency(amount, fromCurrency, toCurrency);
  }

  /**
   * Get P&L breakdown by position
   */
  public getPnLBreakdown(): PnLBreakdown[] {
    const positions = this.getPositions();
    const summary = this.getPositionSummary();
    return pnlEngine.calculatePnLBreakdown(positions, summary.pnl.totalPnl);
  }

  /**
   * Get historical P&L data
   */
  public getHistoricalPnL(days: number = 30) {
    return pnlEngine.getHistoricalPnL(days);
  }

  /**
   * Calculate mark-to-market values
   */
  public getMarkToMarketValues(): Map<string, number> {
    const positions = this.getPositions();
    const currentPrices = new Map<string, number>();
    
    // Build current prices map
    positions.forEach(position => {
      currentPrices.set(position.symbol, position.currentPrice);
    });
    
    return pnlEngine.calculateMarkToMarket(positions, currentPrices);
  }

  /**
   * Event handlers
   */
  public addPositionHandler(handler: (positions: Position[]) => void): void {
    this.positionHandlers.add(handler);
  }

  public removePositionHandler(handler: (positions: Position[]) => void): void {
    this.positionHandlers.delete(handler);
  }

  public addAccountHandler(handler: (accounts: AccountBalance[]) => void): void {
    this.accountHandlers.add(handler);
  }

  public removeAccountHandler(handler: (accounts: AccountBalance[]) => void): void {
    this.accountHandlers.delete(handler);
  }

  public addSummaryHandler(handler: (summary: PositionSummary) => void): void {
    this.summaryHandlers.add(handler);
  }

  public removeSummaryHandler(handler: (summary: PositionSummary) => void): void {
    this.summaryHandlers.delete(handler);
  }

  public addAlertHandler(handler: (alerts: PositionAlert[]) => void): void {
    this.alertHandlers.add(handler);
  }

  public removeAlertHandler(handler: (alerts: PositionAlert[]) => void): void {
    this.alertHandlers.delete(handler);
  }

  /**
   * Notify handlers
   */
  private notifyPositionHandlers(): void {
    const positions = this.getPositions();
    this.positionHandlers.forEach(handler => {
      try {
        handler(positions);
      } catch (error) {
        console.error('Error in position handler:', error);
      }
    });
  }

  private notifyAccountHandlers(): void {
    const accounts = this.getAccountBalances();
    this.accountHandlers.forEach(handler => {
      try {
        handler(accounts);
      } catch (error) {
        console.error('Error in account handler:', error);
      }
    });
  }

  private notifySummaryHandlers(): void {
    const summary = this.getPositionSummary();
    this.summaryHandlers.forEach(handler => {
      try {
        handler(summary);
      } catch (error) {
        console.error('Error in summary handler:', error);
      }
    });
  }

  private notifyAlertHandlers(): void {
    const alerts = this.getAlerts();
    this.alertHandlers.forEach(handler => {
      try {
        handler(alerts);
      } catch (error) {
        console.error('Error in alert handler:', error);
      }
    });
  }
}

// Export singleton instance
export const positionService = new PositionService();