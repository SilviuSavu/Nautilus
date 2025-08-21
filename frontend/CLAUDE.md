# Frontend Module Configuration

This file provides frontend-specific configuration and context for the Nautilus trading platform React application.

## Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Styling**: Ant Design components with custom CSS
- **State Management**: Zustand for application state
- **Routing**: React Router for navigation
- **Charts**: Lightweight Charts for trading visualizations

## Development Commands
- Start dev server: `npm run dev` (runs on port 3000)
- Run unit tests: `npm test` (Vitest)
- Run E2E tests: `npx playwright test`
- Run E2E headed: `npx playwright test --headed`
- Build for production: `npm run build`
- Preview production build: `npm run preview`
- Type checking: `npx tsc --noEmit`

## Component Patterns
- Use functional components with hooks
- Follow naming convention: PascalCase for components
- Organize by feature in `src/components/[Feature]/`
- Export components through index.ts files
- Use TypeScript interfaces for props

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

## Data Architecture Integration
The frontend integrates with a multi-source data architecture:

### Data Source Status Display
- **IBKR Status**: Shows Interactive Brokers Gateway connection status
- **YFinance Status**: Shows historical data supplement availability
- **Data Flow Indicator**: Displays active data source (Database Cache, IBKR, or YFinance)

### Key Integration Points
```typescript
// YFinance status interface (matches backend response)
interface YFinanceStatus {
  service: string;
  status: 'operational' | 'disconnected';
  role: string;
  initialized: boolean;
  connected: boolean;
  error_message?: string;
}

// Unified historical data endpoint
GET /api/v1/market-data/historical/bars
// Returns data from: IBKR → Database Cache → YFinance (fallback hierarchy)
```

### Status Monitoring Components
- **Dashboard.tsx**: Main data source status indicators
- **Historical Data Backfill**: YFinance manual import interface
- **Data Source Badges**: Visual indicators for active data sources