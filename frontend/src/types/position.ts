/**
 * TypeScript types for Position and Account Monitoring
 */

export interface Position {
  id: string;
  symbol: string;
  venue: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  averagePrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  currentPrice: number;
  currency: string;
  timestamp: number;
  openTimestamp: number;
}

export interface PositionUpdate {
  type: 'position_update';
  position: Position;
  timestamp: number;
}

export interface AccountBalance {
  currency: string;
  balance: number;
  availableBalance: number;
  marginUsed: number;
  marginAvailable: number;
  buyingPower: number;
  equity: number;
  timestamp: number;
}

export interface AccountUpdate {
  type: 'account_update';
  account: AccountBalance;
  timestamp: number;
}

export interface PnLCalculation {
  unrealizedPnl: number;
  realizedPnl: number;
  totalPnl: number;
  currency: string;
  dailyPnl: number;
  pnlPercentage: number;
}

export interface PositionSummary {
  totalPositions: number;
  longPositions: number;
  shortPositions: number;
  totalExposure: number;
  netExposure: number;
  grossExposure: number;
  pnl: PnLCalculation;
  currency: string;
}

export interface CurrencyConversion {
  fromCurrency: string;
  toCurrency: string;
  rate: number;
  timestamp: number;
}

export interface PositionRisk {
  positionId: string;
  riskScore: number;
  beta?: number;
  correlation?: number;
  maxDrawdown: number;
  sharpeRatio?: number;
}

export interface PositionAlert {
  id: string;
  positionId: string;
  type: 'price_change' | 'pnl_threshold' | 'margin_call' | 'position_size';
  message: string;
  severity: 'info' | 'warning' | 'critical';
  timestamp: number;
  acknowledged: boolean;
}

export interface PositionState {
  positions: Position[];
  accountBalances: AccountBalance[];
  positionSummary: PositionSummary | null;
  alerts: PositionAlert[];
  isLoading: boolean;
  error: string | null;
  lastUpdate: number;
}