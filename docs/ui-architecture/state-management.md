# State Management

## Store Structure

```
src/store/
├── index.ts                 # Combined store exports and types
├── marketDataStore.ts       # Real-time market data (prices, orderbooks)
├── orderStore.ts           # Order lifecycle and execution state
├── portfolioStore.ts       # Positions, P&L, risk metrics
├── strategyStore.ts        # Strategy management and performance
├── systemStore.ts          # Connection status, alerts, user preferences
├── uiStore.ts             # UI state (panel layouts, modal visibility)
└── middleware/
    ├── websocketMiddleware.ts  # WebSocket integration
    ├── persistenceMiddleware.ts # Local storage persistence
    └── loggingMiddleware.ts    # Development logging
```

## State Management Template

```typescript
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { MarketDataState, Instrument, OrderBook, Trade } from '../types';

interface MarketDataStore extends MarketDataState {
  // Data getters
  getInstrumentPrice: (instrumentId: string) => number | null;
  getOrderBook: (instrumentId: string) => OrderBook | null;
  getRecentTrades: (instrumentId: string, limit?: number) => Trade[];
  
  // Data setters (called by WebSocket handlers)
  updatePrice: (instrumentId: string, price: number, timestamp: number) => void;
  updateOrderBook: (instrumentId: string, orderBook: OrderBook) => void;
  addTrade: (instrumentId: string, trade: Trade) => void;
  
  // Connection management
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'reconnecting') => void;
  setLastUpdate: (instrumentId: string, timestamp: number) => void;
  
  // Cleanup
  clearInstrumentData: (instrumentId: string) => void;
  reset: () => void;
}

const initialState: MarketDataState = {
  instruments: new Map(),
  orderBooks: new Map(),
  recentTrades: new Map(),
  prices: new Map(),
  lastUpdates: new Map(),
  connectionStatus: 'disconnected',
  subscribedInstruments: new Set(),
};

export const useMarketDataStore = create<MarketDataStore>()(
  subscribeWithSelector(
    immer((set, get) => ({
      ...initialState,

      // Optimized getters
      getInstrumentPrice: (instrumentId: string) => {
        const prices = get().prices;
        return prices.get(instrumentId) ?? null;
      },

      getOrderBook: (instrumentId: string) => {
        const orderBooks = get().orderBooks;
        return orderBooks.get(instrumentId) ?? null;
      },

      getRecentTrades: (instrumentId: string, limit = 100) => {
        const trades = get().recentTrades.get(instrumentId) ?? [];
        return trades.slice(0, limit);
      },

      // High-performance updates (called frequently)
      updatePrice: (instrumentId: string, price: number, timestamp: number) => {
        set((state) => {
          state.prices.set(instrumentId, price);
          state.lastUpdates.set(instrumentId, timestamp);
        });
      },

      updateOrderBook: (instrumentId: string, orderBook: OrderBook) => {
        set((state) => {
          state.orderBooks.set(instrumentId, orderBook);
          state.lastUpdates.set(instrumentId, Date.now());
        });
      },

      addTrade: (instrumentId: string, trade: Trade) => {
        set((state) => {
          const trades = state.recentTrades.get(instrumentId) ?? [];
          trades.unshift(trade);
          // Keep only last 1000 trades per instrument
          if (trades.length > 1000) {
            trades.splice(1000);
          }
          state.recentTrades.set(instrumentId, trades);
        });
      },

      setConnectionStatus: (status) => {
        set((state) => {
          state.connectionStatus = status;
        });
      },

      clearInstrumentData: (instrumentId: string) => {
        set((state) => {
          state.prices.delete(instrumentId);
          state.orderBooks.delete(instrumentId);
          state.recentTrades.delete(instrumentId);
          state.lastUpdates.delete(instrumentId);
          state.subscribedInstruments.delete(instrumentId);
        });
      },

      reset: () => {
        set(() => ({ ...initialState }));
      },
    }))
  )
);

// Selector hooks for optimal re-rendering
export const useInstrumentPrice = (instrumentId: string) =>
  useMarketDataStore((state) => state.getInstrumentPrice(instrumentId));

export const useOrderBook = (instrumentId: string) =>
  useMarketDataStore((state) => state.getOrderBook(instrumentId));

export const useConnectionStatus = () =>
  useMarketDataStore((state) => state.connectionStatus);
```
