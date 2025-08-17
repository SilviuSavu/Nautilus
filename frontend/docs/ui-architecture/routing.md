# Routing

## Route Configuration

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
