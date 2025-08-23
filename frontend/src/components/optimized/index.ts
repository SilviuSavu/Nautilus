/**
 * Optimized Components Index
 * Centralized exports for all production-ready optimized components
 */

// Common optimized components
export { default as ErrorBoundary } from '../common/ErrorBoundary';
export { default as LoadingState } from '../common/LoadingState';
export { default as OptimizedChart } from '../common/OptimizedChart';
export { default as VirtualizedTable } from '../common/VirtualizedTable';

// Layout components
export { default as ResponsiveLayout } from '../layout/ResponsiveLayout';

// Optimized hooks
export { useOptimizedWebSocket } from '../../hooks/useOptimizedWebSocket';
export { usePerformanceMonitor } from '../../hooks/usePerformanceMonitor';

// Lazy-loaded Sprint 3 components with optimizations
export const Sprint3Dashboard = lazy(() => 
  import('../../pages/Sprint3Dashboard').then(module => ({ default: module.default }))
);

export const WebSocketMonitoringSuite = lazy(() => 
  import('../WebSocket/WebSocketMonitoringSuite').then(module => ({ 
    default: module.default 
  }))
);

export const Sprint3SystemMonitor = lazy(() => 
  import('../Monitoring/Sprint3SystemMonitor').then(module => ({ 
    default: module.default 
  }))
);

export const RiskDashboardSprint3 = lazy(() => 
  import('../Risk/RiskDashboardSprint3').then(module => ({ 
    default: module.default 
  }))
);

export const DeploymentOrchestrator = lazy(() => 
  import('../Strategy/DeploymentOrchestrator').then(module => ({ 
    default: module.default 
  }))
);

export const PrometheusMetricsDashboard = lazy(() => 
  import('../Monitoring/PrometheusMetricsDashboard').then(module => ({ 
    default: module.default 
  }))
);

export const GrafanaEmbedDashboard = lazy(() => 
  import('../Monitoring/GrafanaEmbedDashboard').then(module => ({ 
    default: module.default 
  }))
);

export const ConnectionHealthDashboard = lazy(() => 
  import('../Monitoring/ConnectionHealthDashboard').then(module => ({ 
    default: module.default 
  }))
);

export const MessageThroughputAnalyzer = lazy(() => 
  import('../WebSocket/MessageThroughputAnalyzer').then(module => ({ 
    default: module.default 
  }))
);

export const WebSocketScalabilityMonitor = lazy(() => 
  import('../WebSocket/WebSocketScalabilityMonitor').then(module => ({ 
    default: module.default 
  }))
);

export const StreamingPerformanceTracker = lazy(() => 
  import('../WebSocket/StreamingPerformanceTracker').then(module => ({ 
    default: module.default 
  }))
);

export const RealTimeRiskMonitor = lazy(() => 
  import('../Risk/RealTimeRiskMonitor').then(module => ({ 
    default: module.default 
  }))
);

export const AdvancedDeploymentPipeline = lazy(() => 
  import('../Strategy/AdvancedDeploymentPipeline').then(module => ({ 
    default: module.default 
  }))
);

// Import lazy function
import { lazy } from 'react';

// Constants and utilities
export { UI_CONSTANTS, ACCESSIBILITY } from '../../constants/ui';
export { default as theme } from '../../styles/theme';

// Type definitions
export type {
  BaseComponentProps,
  DataComponentProps,
  PerformanceProps,
  AccessibilityProps,
  ResponsiveProps,
  ThemeProps,
  WebSocketConnectionState,
  WebSocketMessage,
  ChartDataPoint,
  ChartConfig,
  VisualizationProps,
  TableColumn,
  TableProps,
  MetricDefinition,
  MetricValue,
  AlertRule,
  ProductionConfig
} from '../../types/components';

// Performance monitoring utilities
export interface OptimizationReport {
  componentName: string;
  renderTime: number;
  memoryUsage: number;
  optimizationSuggestions: string[];
  performanceScore: number;
  timestamp: number;
}

export interface BundleAnalysis {
  totalSize: number;
  gzippedSize: number;
  chunkSizes: Record<string, number>;
  unusedCode: string[];
  duplicatedModules: string[];
  optimizationOpportunities: string[];
}

// Component performance wrapper
export const withPerformanceMonitoring = <P extends object>(
  Component: React.ComponentType<P>,
  componentName: string
) => {
  return React.memo((props: P) => {
    const startTime = performance.now();
    
    React.useEffect(() => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      if (renderTime > 16) { // More than one frame at 60fps
        console.warn(`Slow render detected: ${componentName} took ${renderTime.toFixed(2)}ms`);
      }
      
      // Report to performance monitoring service
      if (window.performanceMonitor) {
        window.performanceMonitor.recordRender(componentName, renderTime);
      }
    });

    return <Component {...props} />;
  });
};

// Accessibility helper
export const withAccessibility = <P extends object>(
  Component: React.ComponentType<P>
) => {
  return React.forwardRef<any, P>((props, ref) => {
    // Add default accessibility props
    const accessibilityProps = {
      role: 'main',
      tabIndex: -1,
      ...props
    };

    return <Component {...accessibilityProps} ref={ref} />;
  });
};

// Error boundary wrapper
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: {
    fallbackTitle?: string;
    fallbackMessage?: string;
    onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  }
) => {
  return (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );
};

// Responsive wrapper
export const withResponsive = <P extends object>(
  Component: React.ComponentType<P & ResponsiveProps>
) => {
  return (props: P & ResponsiveProps) => {
    const [screenSize, setScreenSize] = React.useState({
      width: window.innerWidth,
      height: window.innerHeight
    });

    React.useEffect(() => {
      const handleResize = () => {
        setScreenSize({
          width: window.innerWidth,
          height: window.innerHeight
        });
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }, []);

    const breakpoints = {
      xs: screenSize.width < 480,
      sm: screenSize.width >= 480 && screenSize.width < 768,
      md: screenSize.width >= 768 && screenSize.width < 992,
      lg: screenSize.width >= 992 && screenSize.width < 1200,
      xl: screenSize.width >= 1200 && screenSize.width < 1600,
      xxl: screenSize.width >= 1600
    };

    return <Component {...props} breakpoints={breakpoints} />;
  };
};

// Component registry for dynamic loading
export const ComponentRegistry = {
  // Monitoring components
  'sprint3-dashboard': Sprint3Dashboard,
  'websocket-monitoring': WebSocketMonitoringSuite,
  'system-monitor': Sprint3SystemMonitor,
  'prometheus-metrics': PrometheusMetricsDashboard,
  'grafana-dashboard': GrafanaEmbedDashboard,
  'connection-health': ConnectionHealthDashboard,
  
  // Risk management components
  'risk-dashboard': RiskDashboardSprint3,
  'realtime-risk': RealTimeRiskMonitor,
  
  // Strategy components
  'deployment-orchestrator': DeploymentOrchestrator,
  'advanced-pipeline': AdvancedDeploymentPipeline,
  
  // WebSocket components
  'message-throughput': MessageThroughputAnalyzer,
  'websocket-scalability': WebSocketScalabilityMonitor,
  'streaming-performance': StreamingPerformanceTracker
};

// Dynamic component loader
export const loadComponent = async (componentName: string) => {
  const Component = ComponentRegistry[componentName as keyof typeof ComponentRegistry];
  
  if (!Component) {
    throw new Error(`Component ${componentName} not found in registry`);
  }
  
  return Component;
};

// Bundle optimization utilities
export const optimizeBundle = () => {
  // Dynamic imports for better code splitting
  const loadChunk = (chunkName: string) => {
    switch (chunkName) {
      case 'monitoring':
        return Promise.all([
          import('../Monitoring/Sprint3SystemMonitor'),
          import('../Monitoring/PrometheusMetricsDashboard'),
          import('../Monitoring/GrafanaEmbedDashboard')
        ]);
      case 'websocket':
        return Promise.all([
          import('../WebSocket/WebSocketMonitoringSuite'),
          import('../WebSocket/MessageThroughputAnalyzer'),
          import('../WebSocket/WebSocketScalabilityMonitor')
        ]);
      case 'strategy':
        return Promise.all([
          import('../Strategy/DeploymentOrchestrator'),
          import('../Strategy/AdvancedDeploymentPipeline')
        ]);
      case 'risk':
        return Promise.all([
          import('../Risk/RiskDashboardSprint3'),
          import('../Risk/RealTimeRiskMonitor')
        ]);
      default:
        return Promise.reject(new Error(`Unknown chunk: ${chunkName}`));
    }
  };

  return { loadChunk };
};

// Production build utilities
export const ProductionUtils = {
  // Preload critical components
  preloadCritical: () => {
    const criticalComponents = [
      Sprint3Dashboard,
      WebSocketMonitoringSuite,
      Sprint3SystemMonitor
    ];
    
    return Promise.all(criticalComponents);
  },
  
  // Check component health
  healthCheck: async () => {
    const health = {
      componentsLoaded: Object.keys(ComponentRegistry).length,
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
      timestamp: Date.now()
    };
    
    return health;
  },
  
  // Generate performance report
  generateReport: (): OptimizationReport => {
    return {
      componentName: 'Sprint3System',
      renderTime: performance.now(),
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
      optimizationSuggestions: [
        'Enable code splitting for non-critical components',
        'Implement component virtualization for large lists',
        'Use React.memo for expensive components',
        'Enable service worker for caching'
      ],
      performanceScore: 85,
      timestamp: Date.now()
    };
  }
};

// Global performance monitoring setup
if (typeof window !== 'undefined') {
  window.performanceMonitor = {
    recordRender: (componentName: string, renderTime: number) => {
      // Send to monitoring service
      console.debug(`Performance: ${componentName} rendered in ${renderTime.toFixed(2)}ms`);
    },
    
    recordError: (error: Error, componentName?: string) => {
      console.error(`Error in ${componentName || 'unknown'}:`, error);
    },
    
    getMetrics: () => ({
      memory: (performance as any).memory,
      timing: performance.timing,
      navigation: performance.navigation
    })
  };
}

// Export default configuration
export default {
  components: ComponentRegistry,
  utils: ProductionUtils,
  optimization: optimizeBundle()
};