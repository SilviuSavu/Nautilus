# Styling Guidelines

## Styling Approach

**Hybrid CSS Architecture:**
- **Ant Design**: Base component library for professional trading terminal aesthetics
- **CSS Modules**: Component-scoped styles to prevent conflicts in multi-panel layouts
- **CSS Custom Properties**: Global theme system supporting light/dark modes

## Global Theme Variables

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
