# Frontend Developer Standards

## Critical Coding Rules

1. **Performance**: Always use React.memo() for components displaying real-time data
2. **Type Safety**: Never use `any` type - define proper TypeScript interfaces
3. **Error Handling**: Always handle API errors and connection failures gracefully
4. **Testing**: Write tests for all trading-critical components and user flows
5. **Accessibility**: Implement proper ARIA labels and keyboard navigation
6. **State Management**: Use Zustand selectors to prevent unnecessary re-renders
7. **Code Splitting**: Lazy load all page components for optimal bundle size
8. **Environment Variables**: Use VITE_ prefix for all client-side environment variables

## Quick Reference

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