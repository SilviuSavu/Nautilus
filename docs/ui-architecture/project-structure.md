# Project Structure

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
