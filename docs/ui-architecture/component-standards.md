# Component Standards

## Component Template

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

## Naming Conventions

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
