# Testing Requirements

## Component Test Template

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

## Testing Best Practices

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **E2E Tests**: Test critical user flows (using Cypress/Playwright)
4. **Coverage Goals**: Aim for 80% code coverage
5. **Test Structure**: Arrange-Act-Assert pattern
6. **Mock External Dependencies**: API calls, routing, state management
