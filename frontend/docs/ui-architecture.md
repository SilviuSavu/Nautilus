# NautilusTrader Frontend Architecture Document

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2025-08-16 | 1.0 | Initial frontend architecture creation | Winston (Architect) |

## Template and Framework Selection

### Framework Assessment
Based on the provided frontend specification and NautilusTrader project structure, this architecture uses:

**Framework:** React 18+ with TypeScript
**Build Tool:** Vite (faster than Create React App, better for real-time performance)
**UI Foundation:** Ant Design + custom financial components
**State Management:** Zustand (lightweight, performant for real-time updates)
**WebSocket:** Native WebSocket + reconnection logic
**Charts:** Lightweight Charts (as specified in the UX doc)

**Key Rationale:**
- React + TypeScript provides type safety critical for financial data
- Vite offers superior hot reload and build performance
- Ant Design gives professional trading terminal aesthetics
- Zustand avoids Redux overhead for high-frequency updates
- This approach aligns with the <100ms update requirement

## Frontend Tech Stack

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Framework | React | 18.3+ | Core UI framework | Mature ecosystem, excellent TypeScript support, optimal for real-time updates |
| UI Library | Ant Design | 5.x | Base component library | Professional financial terminal aesthetics, proven in trading applications |
| State Management | Zustand | 4.x | Global state management | Lightweight, minimal boilerplate, excellent performance for high-frequency updates |
| Routing | React Router | 6.x | Client-side routing | Industry standard, supports protected routes and lazy loading |
| Build Tool | Vite | 5.x | Development and build tooling | Superior hot reload, faster builds than CRA, optimized for performance |
| Styling | CSS Modules + Ant Design | - | Component styling | Scoped styles, integrates well with Ant Design theming |
| Testing | Vitest + Testing Library | Latest | Unit and integration testing | Vite-native testing, React Testing Library for component tests |
| Component Library | Custom Financial Components | - | Trading-specific UI components | Specialized components for order books, charts, trading widgets |
| Form Handling | React Hook Form | 7.x | Form validation and handling | Minimal re-renders, excellent performance for trading forms |
| Animation | Framer Motion | 11.x | Micro-interactions and transitions | Smooth animations without blocking real-time data updates |
| Dev Tools | React DevTools, Zustand DevTools | Latest | Development debugging | Essential for debugging state and component behavior |

## Project Structure

```
frontend/
├── public/
│   ├── favicon.ico
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button/
│   │   │   ├── Input/
│   │   │   ├── Modal/
│   │   │   └── index.ts
│   │   ├── trading/
│   │   │   ├── TradingDataGrid/
│   │   │   ├── OrderEntryWidget/
│   │   │   ├── RealTimeChart/
│   │   │   ├── StatusIndicator/
│   │   │   ├── PositionSummaryCard/
│   │   │   ├── AlertBanner/
│   │   │   └── index.ts
│   │   └── layout/
│   │       ├── Header/
│   │       ├── Sidebar/
│   │       ├── DashboardGrid/
│   │       └── index.ts
│   ├── pages/
│   │   ├── Dashboard/
│   │   │   ├── index.tsx
│   │   │   ├── Dashboard.module.css
│   │   │   └── components/
│   │   ├── MarketData/
│   │   ├── StrategyManagement/
│   │   ├── OrderManagement/
│   │   ├── RiskPortfolio/
│   │   └── SystemStatus/
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useRealTimeData.ts
│   │   ├── useMarketData.ts
│   │   ├── useOrderManagement.ts
│   │   └── index.ts
│   ├── store/
│   │   ├── marketDataStore.ts
│   │   ├── orderStore.ts
│   │   ├── portfolioStore.ts
│   │   ├── strategyStore.ts
│   │   ├── systemStore.ts
│   │   └── index.ts
│   ├── services/
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   ├── marketData.ts
│   │   │   ├── orders.ts
│   │   │   ├── portfolio.ts
│   │   │   └── strategies.ts
│   │   ├── websocket/
│   │   │   ├── connection.ts
│   │   │   ├── handlers.ts
│   │   │   └── types.ts
│   │   └── formatters/
│   │       ├── currency.ts
│   │       ├── datetime.ts
│   │       └── numbers.ts
│   ├── types/
│   │   ├── api.ts
│   │   ├── trading.ts
│   │   ├── market.ts
│   │   ├── orders.ts
│   │   └── index.ts
│   ├── utils/
│   │   ├── constants.ts
│   │   ├── helpers.ts
│   │   ├── validation.ts
│   │   └── performance.ts
│   ├── styles/
│   │   ├── globals.css
│   │   ├── variables.css
│   │   ├── antd-overrides.css
│   │   └── themes/
│   │       ├── dark.css
│   │       └── light.css
│   ├── assets/
│   │   ├── icons/
│   │   ├── images/
│   │   └── fonts/
│   ├── App.tsx
│   ├── App.module.css
│   ├── main.tsx
│   └── vite-env.d.ts
├── docs/
│   ├── ui-architecture.md
│   ├── component-library.md
│   └── development-guide.md
├── tests/
│   ├── __mocks__/
│   ├── components/
│   ├── hooks/
│   ├── services/
│   ├── utils/
│   └── setup.ts
├── .env.example
├── .env.local
├── .gitignore
├── eslint.config.js
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
└── vitest.config.ts
```

## Component Standards

### Component Template

```typescript
import React, { memo, useCallback, useMemo } from 'react';
import { Button, Space } from 'antd';
import styles from './TradingButton.module.css';
import { TradingButtonProps, ButtonVariant, ButtonState } from './types';

interface TradingButtonProps {
  variant: ButtonVariant;
  size?: 'small' | 'medium' | 'large';
  state?: ButtonState;
  loading?: boolean;
  disabled?: boolean;
  onClick: (event: React.MouseEvent<HTMLButtonElement>) => void;
  children: React.ReactNode;
  className?: string;
  'data-testid'?: string;
}

const TradingButton: React.FC<TradingButtonProps> = memo(({
  variant,
  size = 'medium',
  state = 'default',
  loading = false,
  disabled = false,
  onClick,
  children,
  className,
  'data-testid': testId,
}) => {
  const buttonClass = useMemo(() => [
    styles.tradingButton,
    styles[variant],
    styles[size],
    styles[state],
    className,
  ].filter(Boolean).join(' '), [variant, size, state, className]);

  const handleClick = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    if (!disabled && !loading) {
      onClick(event);
    }
  }, [onClick, disabled, loading]);

  return (
    <Button
      className={buttonClass}
      loading={loading}
      disabled={disabled}
      onClick={handleClick}
      data-testid={testId}
      size={size}
    >
      {children}
    </Button>
  );
});

TradingButton.displayName = 'TradingButton';

export default TradingButton;
export type { TradingButtonProps };
```

### Naming Conventions

**Files and Directories:**
- **Components**: PascalCase directories with index files (`TradingDataGrid/`, `OrderEntryWidget/`)
- **Component files**: PascalCase matching directory name (`TradingDataGrid.tsx`)
- **Style files**: ComponentName.module.css (`TradingDataGrid.module.css`)
- **Type files**: `types.ts` within component directory
- **Hook files**: camelCase with `use` prefix (`useMarketData.ts`)
- **Service files**: camelCase (`marketDataService.ts`)
- **Store files**: camelCase with `Store` suffix (`marketDataStore.ts`)

**Code Naming:**
- **Components**: PascalCase (`TradingDataGrid`, `OrderEntryWidget`)
- **Props interfaces**: ComponentName + Props (`TradingDataGridProps`)
- **Custom hooks**: camelCase with `use` prefix (`useRealTimeData`)
- **Store actions**: camelCase verbs (`updateMarketData`, `setOrderStatus`)
- **Constants**: SCREAMING_SNAKE_CASE (`WEBSOCKET_ENDPOINTS`, `ORDER_TYPES`)
- **CSS classes**: kebab-case in CSS, camelCase in TypeScript (`trading-button` → `styles.tradingButton`)

## State Management

### Store Structure

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

### State Management Template

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

## API Integration

### Service Template

```typescript
import { ApiClient } from './client';
import { 
  OrderRequest, 
  OrderResponse, 
  OrderStatus, 
  OrderUpdatePayload,
  ApiResponse 
} from '../types/api';

export class OrderService {
  private client: ApiClient;

  constructor(client: ApiClient) {
    this.client = client;
  }

  async placeOrder(orderRequest: OrderRequest): Promise<ApiResponse<OrderResponse>> {
    try {
      const response = await this.client.post<OrderResponse>('/orders', {
        body: orderRequest,
        timeout: 5000,
      });

      return {
        success: true,
        data: response.data,
        timestamp: Date.now(),
      };
    } catch (error) {
      return this.handleOrderError(error, 'PLACE_ORDER_FAILED');
    }
  }

  async cancelOrder(orderId: string): Promise<ApiResponse<void>> {
    try {
      await this.client.delete(`/orders/${orderId}`, {
        timeout: 3000,
      });

      return {
        success: true,
        data: undefined,
        timestamp: Date.now(),
      };
    } catch (error) {
      return this.handleOrderError(error, 'CANCEL_ORDER_FAILED');
    }
  }

  private handleOrderError(error: unknown, operation: string): ApiResponse<never> {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    console.error(`${operation}:`, error);
    
    return {
      success: false,
      error: {
        code: operation,
        message: errorMessage,
        timestamp: Date.now(),
      },
      timestamp: Date.now(),
    };
  }
}
```

### API Client Configuration

```typescript
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { useSystemStore } from '../store/systemStore';

export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  apiKey?: string;
  retryAttempts: number;
  retryDelay: number;
}

export class ApiClient {
  private instance: AxiosInstance;
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = config;
    this.instance = this.createAxiosInstance();
    this.setupInterceptors();
  }

  private createAxiosInstance(): AxiosInstance {
    return axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });
  }

  private setupInterceptors(): void {
    // Request interceptor for authentication
    this.instance.interceptors.request.use(
      (config) => {
        const { apiKey } = this.config;
        if (apiKey) {
          config.headers.Authorization = `Bearer ${apiKey}`;
        }

        config.metadata = { startTime: Date.now() };
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and latency tracking
    this.instance.interceptors.response.use(
      (response: AxiosResponse) => {
        const latency = Date.now() - response.config.metadata?.startTime;
        useSystemStore.getState().updateApiLatency(latency);
        return response;
      },
      async (error) => {
        // Handle authentication errors and retries
        if (error.response?.status === 401) {
          useSystemStore.getState().setAuthenticationError();
          return Promise.reject(error);
        }

        if (this.shouldRetry(error)) {
          return this.retryRequest(error.config);
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  async post<T>(url: string, config?: AxiosRequestConfig & { body?: any }): Promise<AxiosResponse<T>> {
    const { body, ...axiosConfig } = config || {};
    return this.instance.post<T>(url, body, axiosConfig);
  }
}
```

## Routing

### Route Configuration

```typescript
import React, { Suspense } from 'react';
import { 
  createBrowserRouter, 
  RouterProvider, 
  Navigate, 
  Outlet 
} from 'react-router-dom';
import { Spin } from 'antd';
import { useAuthStore } from '../store/authStore';
import { Layout } from '../components/layout';

// Lazy-loaded page components for code splitting
const Dashboard = React.lazy(() => import('../pages/Dashboard'));
const MarketData = React.lazy(() => import('../pages/MarketData'));
const StrategyManagement = React.lazy(() => import('../pages/StrategyManagement'));
const OrderManagement = React.lazy(() => import('../pages/OrderManagement'));
const RiskPortfolio = React.lazy(() => import('../pages/RiskPortfolio'));
const SystemStatus = React.lazy(() => import('../pages/SystemStatus'));
const Login = React.lazy(() => import('../pages/Login'));

// Protected route wrapper
const ProtectedRoute: React.FC = () => {
  const isAuthenticated = useAuthStore(state => state.isAuthenticated);

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <Layout>
      <Outlet />
    </Layout>
  );
};

// Router configuration
export const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '/login',
    element: (
      <Suspense fallback={<Spin size="large" />}>
        <Login />
      </Suspense>
    ),
  },
  {
    path: '/',
    element: <ProtectedRoute />,
    children: [
      {
        path: 'dashboard',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: 'market-data',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <MarketData />
          </Suspense>
        ),
      },
      {
        path: 'strategies',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <StrategyManagement />
          </Suspense>
        ),
      },
      {
        path: 'orders',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <OrderManagement />
          </Suspense>
        ),
      },
      {
        path: 'portfolio',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <RiskPortfolio />
          </Suspense>
        ),
      },
      {
        path: 'system',
        element: (
          <Suspense fallback={<Spin size="large" />}>
            <SystemStatus />
          </Suspense>
        ),
      },
    ],
  },
]);

// Route constants for type-safe navigation
export const ROUTES = {
  DASHBOARD: '/dashboard',
  MARKET_DATA: '/market-data',
  STRATEGIES: '/strategies',
  ORDERS: '/orders',
  PORTFOLIO: '/portfolio',
  SYSTEM: '/system',
  LOGIN: '/login',
} as const;
```

## Styling Guidelines

### Styling Approach

**Hybrid CSS Architecture:**
- **Ant Design**: Base component library for professional trading terminal aesthetics
- **CSS Modules**: Component-scoped styles to prevent conflicts in multi-panel layouts
- **CSS Custom Properties**: Global theme system supporting light/dark modes

### Global Theme Variables

```css
/* variables.css */
:root {
  /* Color System - Financial Trading Optimized */
  --color-primary: #1890ff;
  --color-secondary: #722ed1;
  --color-accent: #13c2c2;
  
  /* Trading Semantic Colors */
  --color-success: #52c41a;        /* Profitable/Buy */
  --color-error: #ff4d4f;          /* Loss/Sell */
  --color-warning: #faad14;        /* Risk/Pending */
  
  /* Typography System */
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', Monaco, monospace;
  
  /* Font Sizes */
  --text-xs: 0.75rem;   /* 12px */
  --text-sm: 0.875rem;  /* 14px */
  --text-base: 1rem;    /* 16px */
  --text-lg: 1.125rem;  /* 18px */
  --text-xl: 1.25rem;   /* 20px */
  
  /* Spacing System (4px base unit) */
  --spacing-xs: 0.25rem;    /* 4px */
  --spacing-sm: 0.5rem;     /* 8px */
  --spacing-md: 0.75rem;    /* 12px */
  --spacing-lg: 1rem;       /* 16px */
  --spacing-xl: 1.5rem;     /* 24px */
  --spacing-2xl: 2rem;      /* 32px */
  
  /* Animation System */
  --animation-duration-fast: 150ms;
  --animation-duration-normal: 250ms;
  --animation-easing: cubic-bezier(0.4, 0, 0.2, 1);
  
  /* Trading-Specific Variables */
  --update-flash-duration: 150ms;
}

/* Dark Mode Overrides */
[data-theme="dark"] {
  --color-text-primary: #ffffff;
  --surface-primary: #141414;
  --surface-secondary: #1f1f1f;
  --border-primary: #434343;
  --color-success: #73d13d;
  --color-error: #ff7875;
}
```

## Testing Requirements

### Component Test Template

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { TradingDataGrid } from './TradingDataGrid';
import { useMarketDataStore } from '../../store/marketDataStore';

// Mock store
vi.mock('../../store/marketDataStore');

describe('TradingDataGrid', () => {
  const mockProps = {
    instruments: ['BTCUSD', 'ETHUSD'],
    columns: ['instrument', 'price', 'change', 'volume'],
    onRowClick: vi.fn(),
    'data-testid': 'trading-data-grid',
  };

  beforeEach(() => {
    vi.mocked(useMarketDataStore).mockReturnValue({
      prices: new Map([['BTCUSD', 50000], ['ETHUSD', 3000]]),
      getInstrumentPrice: vi.fn(),
      connectionStatus: 'connected',
    });
  });

  it('renders with correct test id', () => {
    render(<TradingDataGrid {...mockProps} />);
    expect(screen.getByTestId('trading-data-grid')).toBeInTheDocument();
  });

  it('displays all instrument rows', () => {
    render(<TradingDataGrid {...mockProps} />);
    expect(screen.getByText('BTCUSD')).toBeInTheDocument();
    expect(screen.getByText('ETHUSD')).toBeInTheDocument();
  });

  it('calls onRowClick when row is clicked', async () => {
    const user = userEvent.setup();
    render(<TradingDataGrid {...mockProps} />);
    
    const btcRow = screen.getByText('BTCUSD').closest('[data-testid*="row"]');
    await user.click(btcRow!);
    
    expect(mockProps.onRowClick).toHaveBeenCalledWith('BTCUSD');
  });
});
```

### Testing Best Practices

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **E2E Tests**: Test critical user flows (using Cypress/Playwright)
4. **Coverage Goals**: Aim for 80% code coverage
5. **Test Structure**: Arrange-Act-Assert pattern
6. **Mock External Dependencies**: API calls, routing, state management

## Environment Configuration

```bash
# .env.example - Copy to .env.local and configure

# Application Environment
VITE_ENV=development                          # development | staging | production
VITE_APP_NAME=NautilusTrader Dashboard
VITE_APP_VERSION=1.0.0

# Backend API Configuration
VITE_API_BASE_URL=http://localhost:8000/api   # NautilusTrader FastAPI backend
VITE_API_TIMEOUT=10000                        # API request timeout (milliseconds)
VITE_API_RETRY_ATTEMPTS=3                     # Number of retry attempts

# Real-time Data WebSocket
VITE_WS_BASE_URL=ws://localhost:8000/ws       # WebSocket endpoint
VITE_WS_RECONNECT_INTERVAL=5000               # Reconnection interval (milliseconds)
VITE_WS_MAX_RECONNECT_ATTEMPTS=10             # Maximum reconnection attempts

# Performance Configuration
VITE_UPDATE_THROTTLE_MS=50                    # Minimum time between UI updates
VITE_MAX_TRADE_HISTORY_ITEMS=1000             # Maximum trades to keep in memory
VITE_MAX_ORDER_BOOK_LEVELS=20                 # Maximum orderbook levels to display

# Feature Flags
VITE_ENABLE_LIVE_TRADING=false                # Enable live trading (vs simulation)
VITE_ENABLE_ADVANCED_ORDERS=true              # Enable stop/limit/conditional orders
VITE_ENABLE_DARK_MODE=true                    # Enable dark mode toggle
VITE_ENABLE_DEBUG_PANEL=false                 # Enable debug information panel

# Security Configuration
VITE_CSP_REPORT_URI=                          # CSP violation report endpoint
VITE_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Development Configuration
VITE_DEV_SERVER_PORT=5173                     # Development server port
VITE_DEV_SERVER_HOST=localhost                # Development server host
```

## Frontend Developer Standards

### Critical Coding Rules

1. **Performance**: Always use React.memo() for components displaying real-time data
2. **Type Safety**: Never use `any` type - define proper TypeScript interfaces
3. **Error Handling**: Always handle API errors and connection failures gracefully
4. **Testing**: Write tests for all trading-critical components and user flows
5. **Accessibility**: Implement proper ARIA labels and keyboard navigation
6. **State Management**: Use Zustand selectors to prevent unnecessary re-renders
7. **Code Splitting**: Lazy load all page components for optimal bundle size
8. **Environment Variables**: Use VITE_ prefix for all client-side environment variables

### Quick Reference

**Common Commands:**
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run test         # Run test suite
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

**Key Import Patterns:**
```typescript
// Store imports
import { useMarketDataStore } from '../store/marketDataStore';

// Component imports
import { TradingDataGrid } from '../components/trading';

// Type imports
import type { OrderRequest, ApiResponse } from '../types/api';
```

**File Naming Conventions:**
- Components: `TradingDataGrid.tsx`
- Styles: `TradingDataGrid.module.css`
- Types: `types.ts` (within component directory)
- Tests: `TradingDataGrid.test.tsx`

This architecture provides a solid foundation for building a high-performance financial trading dashboard that meets the stringent requirements outlined in the frontend specification.