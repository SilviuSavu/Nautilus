# Sprint 3 Component Usage Guide

## Overview

This comprehensive guide covers all Sprint 3 frontend components, their usage patterns, props, integration examples, and best practices.

## Table of Contents

1. [WebSocket Components](#websocket-components)
2. [Risk Management Components](#risk-management-components)
3. [Analytics & Performance Components](#analytics--performance-components)
4. [Strategy Management Components](#strategy-management-components)
5. [Monitoring Components](#monitoring-components)
6. [Export & Reporting Components](#export--reporting-components)
7. [Hooks & Services](#hooks--services)
8. [Integration Patterns](#integration-patterns)

---

## WebSocket Components

### ConnectionHealthDashboard

**Location:** `src/components/WebSocket/ConnectionHealthDashboard.tsx`

**Description:** Comprehensive WebSocket connection health monitoring dashboard with real-time metrics.

**Props Interface:**
```typescript
interface ConnectionHealthProps {
  connectionId?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  showMetrics?: boolean;
  onConnectionChange?: (status: ConnectionStatus) => void;
}
```

**Usage Example:**
```tsx
import { ConnectionHealthDashboard } from '@/components/WebSocket';

function DashboardPage() {
  const handleConnectionChange = (status: ConnectionStatus) => {
    console.log('Connection status changed:', status);
  };

  return (
    <ConnectionHealthDashboard
      connectionId="main-dashboard"
      autoRefresh={true}
      refreshInterval={5000}
      showMetrics={true}
      onConnectionChange={handleConnectionChange}
    />
  );
}
```

**Features:**
- Real-time connection status monitoring
- Latency tracking and alerts
- Message throughput visualization
- Automatic reconnection handling
- Performance metrics dashboard

### WebSocketMonitoringSuite

**Location:** `src/components/WebSocket/WebSocketMonitoringSuite.tsx`

**Description:** Complete WebSocket monitoring suite with connection management and performance tracking.

**Props Interface:**
```typescript
interface MonitoringSuiteProps {
  connections: Connection[];
  showDetailedMetrics?: boolean;
  enableAlerts?: boolean;
  alertThresholds?: AlertThresholds;
  onAlert?: (alert: Alert) => void;
}
```

**Usage Example:**
```tsx
import { WebSocketMonitoringSuite } from '@/components/WebSocket';

function MonitoringPage() {
  const alertThresholds = {
    maxLatency: 1000,
    minThroughput: 100,
    maxReconnections: 5
  };

  return (
    <WebSocketMonitoringSuite
      connections={activeConnections}
      showDetailedMetrics={true}
      enableAlerts={true}
      alertThresholds={alertThresholds}
      onAlert={(alert) => console.log('Alert:', alert)}
    />
  );
}
```

### MessageThroughputAnalyzer

**Location:** `src/components/WebSocket/MessageThroughputAnalyzer.tsx`

**Description:** Analyzes and visualizes WebSocket message throughput patterns.

**Usage Example:**
```tsx
import { MessageThroughputAnalyzer } from '@/components/WebSocket';

<MessageThroughputAnalyzer
  timeWindow="5m"
  showBreakdown={true}
  messageTypes={['trade_update', 'market_data', 'risk_alert']}
/>
```

---

## Risk Management Components

### RiskDashboardSprint3

**Location:** `src/components/Risk/RiskDashboardSprint3.tsx`

**Description:** Advanced risk dashboard with real-time monitoring, limit management, and breach detection.

**Props Interface:**
```typescript
interface RiskDashboardProps {
  portfolioId: string;
  showLimits?: boolean;
  showBreaches?: boolean;
  refreshInterval?: number;
  enableAlerts?: boolean;
}
```

**Usage Example:**
```tsx
import { RiskDashboardSprint3 } from '@/components/Risk';

function RiskPage() {
  return (
    <RiskDashboardSprint3
      portfolioId="PORTFOLIO_001"
      showLimits={true}
      showBreaches={true}
      refreshInterval={30000}
      enableAlerts={true}
    />
  );
}
```

**Features:**
- Real-time risk metrics visualization
- Dynamic risk limit management
- Breach detection and alerting
- VaR calculations with multiple models
- Stress testing results display

### AdvancedBreachDetector

**Location:** `src/components/Risk/AdvancedBreachDetector.tsx`

**Description:** ML-powered breach detection with predictive analytics.

**Props Interface:**
```typescript
interface BreachDetectorProps {
  portfolioId: string;
  enablePrediction?: boolean;
  alertThreshold?: number;
  onBreachDetected?: (breach: RiskBreach) => void;
}
```

**Usage Example:**
```tsx
import { AdvancedBreachDetector } from '@/components/Risk';

<AdvancedBreachDetector
  portfolioId="PORTFOLIO_001"
  enablePrediction={true}
  alertThreshold={0.8}
  onBreachDetected={(breach) => {
    // Handle breach notification
    notificationService.send({
      type: 'risk_breach',
      data: breach
    });
  }}
/>
```

### RiskLimitConfigPanel

**Location:** `src/components/Risk/RiskLimitConfigPanel.tsx`

**Description:** Comprehensive risk limit configuration and management interface.

**Usage Example:**
```tsx
import { RiskLimitConfigPanel } from '@/components/Risk';

<RiskLimitConfigPanel
  portfolioId="PORTFOLIO_001"
  onLimitCreated={(limit) => console.log('Created:', limit)}
  onLimitUpdated={(limit) => console.log('Updated:', limit)}
/>
```

### DynamicLimitEngine

**Location:** `src/components/Risk/DynamicLimitEngine.tsx`

**Description:** Dynamic risk limit engine with auto-adjustment capabilities.

**Usage Example:**
```tsx
import { DynamicLimitEngine } from '@/components/Risk';

<DynamicLimitEngine
  portfolioId="PORTFOLIO_001"
  enableAutoAdjustment={true}
  adjustmentStrategy="volatility_based"
  adjustmentFrequency="daily"
/>
```

---

## Analytics & Performance Components

### AdvancedAnalyticsDashboard

**Location:** `src/components/Performance/AdvancedAnalyticsDashboard.tsx`

**Description:** Comprehensive analytics dashboard with real-time performance metrics and visualization.

**Props Interface:**
```typescript
interface AnalyticsDashboardProps {
  portfolioId: string;
  timeRange?: TimeRange;
  benchmark?: string;
  showAttribution?: boolean;
  enableRealTime?: boolean;
}
```

**Usage Example:**
```tsx
import { AdvancedAnalyticsDashboard } from '@/components/Performance';

function AnalyticsPage() {
  return (
    <AdvancedAnalyticsDashboard
      portfolioId="PORTFOLIO_001"
      timeRange={{ start: '2024-01-01', end: '2024-12-31' }}
      benchmark="SPY"
      showAttribution={true}
      enableRealTime={true}
    />
  );
}
```

**Features:**
- Real-time performance calculations
- Attribution analysis (sector, style, security)
- Risk-adjusted returns visualization
- Benchmark comparison
- Factor exposure analysis

### RealTimeAnalyticsDashboard

**Location:** `src/components/Performance/RealTimeAnalyticsDashboard.tsx`

**Description:** Real-time analytics with streaming updates and live calculations.

**Usage Example:**
```tsx
import { RealTimeAnalyticsDashboard } from '@/components/Performance';

<RealTimeAnalyticsDashboard
  portfolioId="PORTFOLIO_001"
  updateInterval={1000}
  metrics={['pnl', 'var', 'sharpe_ratio', 'drawdown']}
  enableWebSocket={true}
/>
```

### AnalyticsAggregator

**Location:** `src/components/Performance/AnalyticsAggregator.tsx`

**Description:** Data aggregation service for historical analytics and reporting.

**Usage Example:**
```tsx
import { AnalyticsAggregator } from '@/components/Performance';

<AnalyticsAggregator
  dataSources={['portfolio', 'benchmarks', 'risk_factors']}
  aggregationLevel="daily"
  exportFormats={['json', 'csv', 'excel']}
/>
```

### ExecutionAnalytics

**Location:** `src/components/Performance/ExecutionAnalytics.tsx`

**Description:** Trade execution quality analysis and slippage monitoring.

**Usage Example:**
```tsx
import { ExecutionAnalytics } from '@/components/Performance';

<ExecutionAnalytics
  portfolioId="PORTFOLIO_001"
  timeRange="1M"
  showSlippageAnalysis={true}
  benchmarkVwap={true}
/>
```

---

## Strategy Management Components

### AdvancedDeploymentPipeline

**Location:** `src/components/Strategy/AdvancedDeploymentPipeline.tsx`

**Description:** Complete strategy deployment pipeline with approval workflows and automated testing.

**Props Interface:**
```typescript
interface DeploymentPipelineProps {
  strategyId: string;
  enableApproval?: boolean;
  autoTesting?: boolean;
  rollbackOnFailure?: boolean;
  onDeploymentComplete?: (result: DeploymentResult) => void;
}
```

**Usage Example:**
```tsx
import { AdvancedDeploymentPipeline } from '@/components/Strategy';

function DeploymentPage() {
  return (
    <AdvancedDeploymentPipeline
      strategyId="MOMENTUM_V1"
      enableApproval={true}
      autoTesting={true}
      rollbackOnFailure={true}
      onDeploymentComplete={(result) => {
        console.log('Deployment completed:', result);
      }}
    />
  );
}
```

**Features:**
- Multi-environment deployment support
- Blue-green and canary deployment strategies
- Automated testing integration
- Approval workflow management
- Real-time deployment monitoring

### StrategyVersionControl

**Location:** `src/components/Strategy/StrategyVersionControl.tsx`

**Description:** Git-like version control for trading strategies.

**Usage Example:**
```tsx
import { StrategyVersionControl } from '@/components/Strategy';

<StrategyVersionControl
  strategyId="MOMENTUM_V1"
  showDiff={true}
  enableBranching={true}
  onVersionCreate={(version) => console.log('New version:', version)}
/>
```

### DeploymentApprovalEngine

**Location:** `src/components/Strategy/DeploymentApprovalEngine.tsx`

**Description:** Approval workflow engine for strategy deployments.

**Usage Example:**
```tsx
import { DeploymentApprovalEngine } from '@/components/Strategy';

<DeploymentApprovalEngine
  deploymentId="DEPLOY_001"
  approvers={['risk_manager', 'portfolio_manager']}
  requireAllApprovals={true}
/>
```

### ProductionMonitor

**Location:** `src/components/Strategy/ProductionMonitor.tsx`

**Description:** Real-time monitoring of strategies in production.

**Usage Example:**
```tsx
import { ProductionMonitor } from '@/components/Strategy';

<ProductionMonitor
  strategyId="MOMENTUM_V1"
  environment="production"
  alertOnDeviations={true}
  performanceThresholds={{
    maxDrawdown: 0.05,
    minSharpeRatio: 1.0
  }}
/>
```

---

## Monitoring Components

### Sprint3SystemMonitor

**Location:** `src/components/Monitoring/Sprint3SystemMonitor.tsx`

**Description:** Comprehensive system monitoring dashboard for all Sprint 3 components.

**Props Interface:**
```typescript
interface SystemMonitorProps {
  components?: ComponentType[];
  refreshInterval?: number;
  showAlerts?: boolean;
  enableNotifications?: boolean;
}
```

**Usage Example:**
```tsx
import { Sprint3SystemMonitor } from '@/components/Monitoring';

<Sprint3SystemMonitor
  components={['analytics', 'risk', 'websocket', 'strategy']}
  refreshInterval={30000}
  showAlerts={true}
  enableNotifications={true}
/>
```

### PrometheusMetricsDashboard

**Location:** `src/components/Monitoring/PrometheusMetricsDashboard.tsx`

**Description:** Prometheus metrics visualization and alerting dashboard.

**Usage Example:**
```tsx
import { PrometheusMetricsDashboard } from '@/components/Monitoring';

<PrometheusMetricsDashboard
  metricsEndpoint="http://localhost:9090"
  dashboardConfig="nautilus-trading-overview"
  timeRange="1h"
/>
```

### ComponentStatusMatrix

**Location:** `src/components/Monitoring/ComponentStatusMatrix.tsx`

**Description:** Matrix view of all system components and their health status.

**Usage Example:**
```tsx
import { ComponentStatusMatrix } from '@/components/Monitoring';

<ComponentStatusMatrix
  components={systemComponents}
  updateInterval={10000}
  showDependencies={true}
/>
```

---

## Export & Reporting Components

### DataExportReporting

**Location:** `src/components/Performance/DataExportReporting.tsx`

**Description:** Comprehensive data export and reporting system with multiple formats.

**Props Interface:**
```typescript
interface DataExportProps {
  dataType: 'performance' | 'risk' | 'trades' | 'portfolio';
  portfolioId?: string;
  dateRange: DateRange;
  formats: ExportFormat[];
  onExportComplete?: (result: ExportResult) => void;
}
```

**Usage Example:**
```tsx
import { DataExportReporting } from '@/components/Performance';

<DataExportReporting
  dataType="performance"
  portfolioId="PORTFOLIO_001"
  dateRange={{ start: '2024-01-01', end: '2024-12-31' }}
  formats={['excel', 'pdf', 'csv']}
  onExportComplete={(result) => {
    console.log('Export completed:', result.downloadUrl);
  }}
/>
```

---

## Hooks & Services

### useRiskMonitoring

**Location:** `src/hooks/risk/useRiskMonitoring.ts`

**Description:** Hook for real-time risk monitoring with WebSocket integration.

**Usage Example:**
```tsx
import { useRiskMonitoring } from '@/hooks/risk';

function RiskComponent({ portfolioId }: { portfolioId: string }) {
  const {
    riskMetrics,
    alerts,
    isMonitoring,
    startMonitoring,
    stopMonitoring
  } = useRiskMonitoring(portfolioId);

  useEffect(() => {
    startMonitoring();
    return () => stopMonitoring();
  }, [portfolioId]);

  return (
    <div>
      <div>VaR: {riskMetrics?.var95}</div>
      {alerts.map(alert => (
        <div key={alert.id} className="alert">
          {alert.message}
        </div>
      ))}
    </div>
  );
}
```

### useStrategyDeployment

**Location:** `src/hooks/strategy/useStrategyDeployment.ts`

**Description:** Hook for managing strategy deployments with real-time status updates.

**Usage Example:**
```tsx
import { useStrategyDeployment } from '@/hooks/strategy';

function DeploymentComponent() {
  const {
    deploy,
    deploymentStatus,
    isDeploying,
    rollback,
    deploymentHistory
  } = useStrategyDeployment();

  const handleDeploy = async () => {
    const result = await deploy({
      strategyId: 'MOMENTUM_V1',
      version: '1.2.0',
      environment: 'staging'
    });
    
    console.log('Deployment result:', result);
  };

  return (
    <div>
      <button onClick={handleDeploy} disabled={isDeploying}>
        Deploy Strategy
      </button>
      <div>Status: {deploymentStatus}</div>
    </div>
  );
}
```

### useWebSocketManager

**Location:** `src/hooks/useWebSocketManager.ts`

**Description:** Comprehensive WebSocket management hook with connection pooling and subscription management.

**Usage Example:**
```tsx
import { useWebSocketManager } from '@/hooks';

function RealTimeComponent() {
  const {
    isConnected,
    subscribe,
    unsubscribe,
    send,
    connectionStats
  } = useWebSocketManager();

  useEffect(() => {
    const subscription = subscribe('portfolio.updates', {
      portfolio_id: 'PORTFOLIO_001'
    }, (data) => {
      console.log('Portfolio update:', data);
    });

    return () => unsubscribe(subscription.id);
  }, []);

  return (
    <div>
      <div>Connected: {isConnected ? 'Yes' : 'No'}</div>
      <div>Latency: {connectionStats.latency}ms</div>
    </div>
  );
}
```

### useRealTimeAnalytics

**Location:** `src/hooks/analytics/useRealTimeAnalytics.ts`

**Description:** Hook for real-time analytics with streaming performance calculations.

**Usage Example:**
```tsx
import { useRealTimeAnalytics } from '@/hooks/analytics';

function AnalyticsComponent({ portfolioId }: { portfolioId: string }) {
  const {
    metrics,
    isStreaming,
    startStreaming,
    stopStreaming,
    historicalData
  } = useRealTimeAnalytics(portfolioId);

  return (
    <div>
      <div>Current P&L: ${metrics?.unrealizedPnl?.toFixed(2)}</div>
      <div>Sharpe Ratio: {metrics?.sharpeRatio?.toFixed(3)}</div>
      <button onClick={isStreaming ? stopStreaming : startStreaming}>
        {isStreaming ? 'Stop' : 'Start'} Streaming
      </button>
    </div>
  );
}
```

---

## Integration Patterns

### Component Composition Pattern

```tsx
// Main dashboard composition
import {
  RiskDashboardSprint3,
  AdvancedAnalyticsDashboard,
  Sprint3SystemMonitor,
  ConnectionHealthDashboard
} from '@/components';

function Sprint3Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="sprint3-dashboard">
      <div className="header">
        <ConnectionHealthDashboard connectionId="main-dashboard" />
        <Sprint3SystemMonitor components={['all']} />
      </div>
      
      <div className="content">
        {activeTab === 'risk' && (
          <RiskDashboardSprint3 
            portfolioId="PORTFOLIO_001"
            enableAlerts={true}
          />
        )}
        {activeTab === 'analytics' && (
          <AdvancedAnalyticsDashboard
            portfolioId="PORTFOLIO_001"
            enableRealTime={true}
          />
        )}
      </div>
    </div>
  );
}
```

### Provider Pattern for Context

```tsx
// Sprint3Provider context
import { Sprint3Provider, useSprint3Context } from '@/context/Sprint3Context';

function App() {
  return (
    <Sprint3Provider
      config={{
        websocketUrl: process.env.VITE_WS_URL,
        apiBaseUrl: process.env.VITE_API_BASE_URL,
        enableRealTime: true
      }}
    >
      <Sprint3Dashboard />
    </Sprint3Provider>
  );
}

function AnyComponent() {
  const { 
    systemHealth, 
    connectionStatus, 
    alertsCount 
  } = useSprint3Context();
  
  // Component logic here
}
```

### Error Boundary Pattern

```tsx
import { Sprint3ErrorBoundary } from '@/components/ErrorBoundary';

function ComponentWithErrorHandling() {
  return (
    <Sprint3ErrorBoundary
      fallback={<div>Something went wrong with Sprint 3 components</div>}
      onError={(error, errorInfo) => {
        console.error('Sprint 3 Error:', error, errorInfo);
      }}
    >
      <RiskDashboardSprint3 portfolioId="PORTFOLIO_001" />
      <AdvancedAnalyticsDashboard portfolioId="PORTFOLIO_001" />
    </Sprint3ErrorBoundary>
  );
}
```

## Best Practices

### Performance Optimization

1. **Use React.memo for expensive components:**
```tsx
const AnalyticsDashboard = React.memo(AdvancedAnalyticsDashboard);
```

2. **Implement proper cleanup in useEffect:**
```tsx
useEffect(() => {
  const cleanup = startRiskMonitoring(portfolioId);
  return cleanup;
}, [portfolioId]);
```

3. **Use lazy loading for large components:**
```tsx
const AdvancedAnalytics = lazy(() => 
  import('@/components/Performance/AdvancedAnalyticsDashboard')
);
```

### Error Handling

1. **Implement proper error boundaries**
2. **Handle WebSocket connection errors**
3. **Provide fallback UI for failed components**
4. **Log errors for monitoring**

### Real-time Data Management

1. **Use WebSocket connections efficiently**
2. **Implement proper subscription management**
3. **Handle connection drops gracefully**
4. **Optimize data updates to prevent excessive re-renders**

This comprehensive guide provides detailed information for using all Sprint 3 components effectively in your trading application.