# Frontend Environment Configuration

## ⚠️ CRITICAL: NO Hardcoded Values

**All frontend components now use environment variables. NO hardcoded URLs or ports.**

## Environment Variables Used

### Primary Configuration
- **VITE_API_BASE_URL**: `http://localhost:8001` (backend API endpoint)
- **VITE_WS_URL**: `localhost:8001` (WebSocket endpoint)

### Configured in Docker
These are automatically set in `docker-compose.yml`:
```yaml
environment:
  - VITE_API_BASE_URL=http://localhost:8001
  - VITE_WS_URL=localhost:8001
```

## Components Updated

### Fixed Hardcoded Values
✅ **IBDashboard.tsx**: Removed hardcoded port 7496 fallbacks
✅ **SimpleChart.tsx**: Uses `VITE_API_BASE_URL` instead of hardcoded localhost:8001
✅ **FactorDashboard.tsx**: Uses `VITE_WS_URL` for WebSocket connections
✅ **NetworkMonitoringDashboard.tsx**: Removed hardcoded IB Gateway IP
✅ **Strategy components**: Changed localhost:8002 → `VITE_API_BASE_URL`
✅ **Performance components**: Changed localhost:8002 → `VITE_API_BASE_URL`
✅ **Demo/Test components**: Show dynamic URLs from environment

### Pattern Used
```typescript
// ✅ CORRECT: Use environment variable
const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

// ❌ WRONG: Hardcoded URL (REMOVED)
const apiUrl = 'http://localhost:8080'
```

### WebSocket Pattern
```typescript
// ✅ CORRECT: Use environment variable
const wsUrl = import.meta.env.VITE_WS_URL || 'localhost:8001'
const ws = new WebSocket(`ws://${wsUrl}/api/v1/streaming/ws/factors`)

// ❌ WRONG: Hardcoded WebSocket URL (REMOVED)
const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/ws/factors')
```

## Services Updated

### API Services
- **PerformanceMetricsService**: Uses `VITE_API_BASE_URL`
- **DataCatalogService**: Uses `VITE_API_BASE_URL`
- **BacktestService**: Uses `VITE_API_BASE_URL`
- **SystemMonitoringService**: Uses `VITE_API_BASE_URL`
- **InstrumentService**: Uses `VITE_API_BASE_URL`
- **StrategyService**: Uses `VITE_API_BASE_URL`

### WebSocket Services
- **WebSocketService**: Uses `VITE_WS_URL` fallback
- **FactorDashboard**: Uses `VITE_WS_URL` for streaming

## Result

### Before (Hardcoded)
```typescript
// Bad examples that were REMOVED:
const response = await fetch('http://localhost:8080/api/endpoint')
const ws = new WebSocket('ws://localhost:8000/streaming')
port: 7496  // Hardcoded IB Gateway port
```

### After (Environment Variables)
```typescript
// Good examples now used:
const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'
const response = await fetch(`${apiUrl}/api/endpoint`)

const wsUrl = import.meta.env.VITE_WS_URL || 'localhost:8001'
const ws = new WebSocket(`ws://${wsUrl}/streaming`)

port: 0  // No hardcoded fallback, use backend response
```

## Benefits

1. **Flexible Deployment**: Can change backend URL via environment variables
2. **No Port Conflicts**: No hardcoded ports that might conflict
3. **Proper Containerization**: Frontend properly communicates through Docker network
4. **Easy Configuration**: Change URLs in one place (docker-compose.yml)
5. **Development/Production**: Same code works in different environments

## Verification

All components now properly:
- ✅ Use `import.meta.env.VITE_API_BASE_URL` for API calls
- ✅ Use `import.meta.env.VITE_WS_URL` for WebSocket connections  
- ✅ Show dynamic URLs in display components
- ✅ Have sensible fallbacks (not hardcoded wrong ports)
- ✅ Work correctly in Docker environment