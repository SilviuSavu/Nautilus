# Frontend Module Configuration

This file provides frontend-specific configuration and context for the Nautilus trading platform React application.

## Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Styling**: Ant Design components with custom CSS
- **State Management**: Zustand for application state
- **Routing**: React Router for navigation
- **Charts**: Lightweight Charts for trading visualizations

## Development Commands (Docker Only)
**IMPORTANT: Frontend runs ONLY in Docker container. Do NOT run locally.**
**CRITICAL: NO hardcoded values - all components use environment variables.**

### Container Commands
- Start frontend container: `docker-compose up frontend`
- View frontend logs: `docker-compose logs frontend`
- Execute commands in container: `docker exec -it nautilus-frontend [command]`
- Restart frontend: `docker-compose restart frontend`

### Development & Testing
- **Frontend URL**: http://localhost:3000 (containerized only)
- **Backend API**: http://localhost:8001 (containerized only)
- Run unit tests: `cd frontend && npm test` (can run locally)
- Run E2E tests: `cd frontend && npx playwright test` (can run locally)
- Run E2E headed: `cd frontend && npx playwright test --headed`
- Build for production: `docker exec nautilus-frontend npm run build`
- Type checking: `docker exec nautilus-frontend npx tsc --noEmit`

### Environment Variables (Configured in Docker)
- **VITE_API_BASE_URL**: http://localhost:8001 (backend endpoint)
- **VITE_WS_URL**: localhost:8001 (WebSocket endpoint)
- **All components**: Use `import.meta.env.VITE_API_BASE_URL` - NO hardcoded URLs

## Component Patterns
- Use functional components with hooks
- Follow naming convention: PascalCase for components
- Organize by feature in `src/components/[Feature]/`
- Export components through index.ts files
- Use TypeScript interfaces for props
- **NEVER hardcode URLs or ports** - always use `import.meta.env.VITE_API_BASE_URL`

## Testing Approach
- **Unit Tests**: Vitest + React Testing Library
- **E2E Tests**: Playwright for user workflows
- **Test Location**: `src/components/__tests__/` for units, `tests/e2e/` for E2E
- **Coverage**: Aim for >80% on business logic components

## Key Directories
```
src/
├── components/         # React components organized by feature
├── hooks/             # Custom React hooks
├── services/          # API and external service integrations
├── types/             # TypeScript type definitions
├── pages/             # Route-level page components
└── main.tsx           # Application entry point
```

## UI/UX Guidelines
- Use Ant Design components for consistency
- Implement responsive design patterns
- Follow accessibility best practices
- Use semantic HTML structure
- Maintain consistent spacing and typography

## State Management
- Use Zustand for global application state
- Keep component state local when possible
- Use custom hooks for shared stateful logic
- Implement proper error boundaries

## API Integration
- Use axios for HTTP requests
- Implement proper error handling
- Use TypeScript interfaces for API responses
- Handle loading and error states consistently

## Data Architecture Integration (Containerized)
The frontend integrates with containerized multi-source data architecture via port 8001:

### Unified Data Sources (NEW)
All data sources are now unified under `/api/v1/nautilus-data/`:
- **FRED Economic Data**: Real-time macro factors and economic indicators
- **Alpha Vantage**: Market data, quotes, search, and fundamentals
- **EDGAR SEC Data**: Company filings, facts, and regulatory information
- **IBKR Gateway**: Professional trading data and execution

### Key Integration Points
```typescript
// Unified health check endpoint
GET /api/v1/nautilus-data/health
// Returns status for FRED, Alpha Vantage, EDGAR, and IBKR

// FRED macro factors endpoint
GET /api/v1/nautilus-data/fred/macro-factors
// Returns 16+ real-time calculated macro factors

// Alpha Vantage search endpoint
GET /api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL
// Returns symbol search results

// EDGAR company search endpoint
GET /api/v1/edgar/companies/search?q=Apple
// Returns company search with CIK/ticker mapping

// All endpoints accessed via containerized backend at localhost:8001
```

### Status Monitoring Components
- **FactorDashboard.tsx**: Real-time FRED and Alpha Vantage status
- **Dashboard.tsx**: Main system status indicators with unified health checks
- **Data Source Badges**: Visual indicators for all containerized data sources

### Frontend Connection & Latency Architecture

The frontend implements **optimized connection patterns** for different latency requirements across various data flows.

#### Connection Types & Performance Targets

**HTTP API Calls** (Request/Response):
```typescript
// Standard REST API calls via axios
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'
// Target: < 200ms response time
// Usage: Data source health checks, configuration updates, historical queries
```

**WebSocket Real-Time Streaming** (Persistent connections):
```typescript  
// Real-time data streaming
const WS_URL = import.meta.env.VITE_WS_URL || 'localhost:8001'
// Target: < 50ms message latency
// Usage: Live market data, trade executions, system alerts
// Capacity: 1000+ concurrent connections per backend instance
```

**Polling-Based Updates** (Fallback pattern):
```typescript
// Used for non-critical updates or WebSocket fallback
// Interval: 5-30 seconds depending on data criticality
// Usage: System health, background status updates
```

#### Frontend Performance Optimizations

**React Component Patterns** (Latency-aware):
- **Memoization**: `useMemo` and `useCallback` for expensive calculations
- **Virtual Scrolling**: Large data sets (market data, trade history)
- **Lazy Loading**: Components loaded on-demand
- **Code Splitting**: Route-based bundle splitting

**State Management** (Zustand optimized):
- **Selective Updates**: Only re-render components with changed data
- **Background Updates**: Non-blocking state synchronization
- **Connection Pooling**: Reuse WebSocket connections across components

**Data Flow Performance**:
```typescript
// Critical real-time data (< 50ms target)
useWebSocket(endpoint, {
  onMessage: (data) => updateState(data),
  heartbeat: 30000,  // 30-second heartbeat
  reconnectAttempts: 10
})

// Standard API calls (< 200ms target)  
const { data, loading, error } = useQuery(endpoint, {
  refreshInterval: 5000,  // 5-second polling
  timeout: 10000  // 10-second timeout
})

// Background updates (< 5s acceptable)
usePolling(endpoint, {
  interval: 30000,  // 30-second intervals
  background: true
})
```

#### Connection Health Monitoring

**WebSocket Connection Management**:
- **Heartbeat Monitoring**: 30-second intervals
- **Automatic Reconnection**: Exponential backoff
- **Connection State Tracking**: Visual indicators in UI
- **Failover Patterns**: Graceful degradation to HTTP polling

**Performance Monitoring Hooks**:
```typescript
// Real-time connection health
const { connectionStatus, latency, messageCount } = useWebSocketHealth()

// API response time tracking  
const { responseTime, errorRate } = useAPIPerformance()

// System resource monitoring
const { cpuUsage, memoryUsage } = useSystemMetrics()
```

#### Responsive Design & Performance

**Viewport Optimization**:
- **Mobile-first**: Optimized for mobile trading applications
- **Adaptive Rendering**: Reduce complexity on smaller screens
- **Touch-friendly**: Large tap targets for order management
- **Offline Support**: Cache critical data for offline viewing

**Bundle Optimization**:
- **Tree Shaking**: Remove unused code
- **Lazy Routes**: Load pages on-demand
- **Asset Optimization**: Compressed images and fonts
- **CDN Integration**: Static asset delivery optimization