export interface OrderBookLevel {
  price: number;
  quantity: number;
  orderCount?: number;
}

export interface OrderBookData {
  symbol: string;
  venue: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
}

export interface OrderBookMessage {
  type: 'order_book_update';
  symbol: string;
  venue: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
}

export interface OrderBookSpread {
  bestBid: number | null;
  bestAsk: number | null;
  spread: number | null;
  spreadPercentage: number | null;
}

export interface OrderBookAggregationSettings {
  enabled: boolean;
  increment: number;
  maxLevels: number;
}

export interface OrderBookDisplaySettings {
  showSpread: boolean;
  showOrderCount: boolean;
  colorScheme: 'default' | 'dark' | 'light';
  decimals: number;
}

export interface ProcessedOrderBookLevel extends OrderBookLevel {
  cumulative: number;
  percentage: number;
  id: string;
}

export interface ProcessedOrderBookData {
  symbol: string;
  venue: string;
  bids: ProcessedOrderBookLevel[];
  asks: ProcessedOrderBookLevel[];
  spread: OrderBookSpread;
  timestamp: number;
  totalBidVolume: number;
  totalAskVolume: number;
}

export interface OrderBookSubscription {
  symbol: string;
  venue: string;
  active: boolean;
  lastUpdate: number;
}

export interface OrderBookStore {
  currentOrderBook: ProcessedOrderBookData | null;
  subscriptions: OrderBookSubscription[];
  aggregationSettings: OrderBookAggregationSettings;
  displaySettings: OrderBookDisplaySettings;
  isLoading: boolean;
  error: string | null;
  connectionStatus: 'connected' | 'disconnected' | 'error';
  
  // Actions
  subscribeToOrderBook: (symbol: string, venue: string) => void;
  unsubscribeFromOrderBook: (symbol: string, venue: string) => void;
  updateOrderBook: (data: OrderBookData) => void;
  updateAggregationSettings: (settings: Partial<OrderBookAggregationSettings>) => void;
  updateDisplaySettings: (settings: Partial<OrderBookDisplaySettings>) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'error') => void;
  clearOrderBook: () => void;
}

export interface OrderBookError {
  type: 'connection' | 'data' | 'subscription' | 'processing';
  message: string;
  symbol?: string;
  timestamp: number;
}

export interface OrderBookPerformanceMetrics {
  updateLatency: number[];
  averageLatency: number;
  maxLatency: number;
  updatesPerSecond: number;
  lastUpdateTime: number;
}