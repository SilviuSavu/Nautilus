/**
 * Production Configuration
 * Comprehensive production-ready configuration for Sprint 3 deployment
 */

import type { ProductionConfig } from '../types/components';

// Environment detection
const isDevelopment = import.meta.env.MODE === 'development';
const isProduction = import.meta.env.MODE === 'production';
const isTest = import.meta.env.MODE === 'test';

// Base URLs from environment variables with fallbacks
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'localhost:8001';
const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL || 'http://localhost:3002';
const PROMETHEUS_URL = import.meta.env.VITE_PROMETHEUS_URL || 'http://localhost:9090';

// Feature flags from environment
const FEATURE_FLAGS = {
  enableAdvancedMonitoring: import.meta.env.VITE_ENABLE_ADVANCED_MONITORING === 'true',
  enableWebSocketOptimization: import.meta.env.VITE_ENABLE_WS_OPTIMIZATION === 'true',
  enablePerformanceMonitoring: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING !== 'false',
  enableErrorReporting: import.meta.env.VITE_ENABLE_ERROR_REPORTING !== 'false',
  enableAnalytics: import.meta.env.VITE_ENABLE_ANALYTICS !== 'false',
  enableServiceWorker: import.meta.env.VITE_ENABLE_SERVICE_WORKER === 'true',
  enableOfflineMode: import.meta.env.VITE_ENABLE_OFFLINE_MODE === 'true',
  enableDarkMode: import.meta.env.VITE_ENABLE_DARK_MODE !== 'false',
  enableAccessibilityMode: import.meta.env.VITE_ENABLE_A11Y === 'true',
  enableDebugMode: isDevelopment || import.meta.env.VITE_ENABLE_DEBUG === 'true'
};

// Performance thresholds
export const PERFORMANCE_THRESHOLDS = {
  // Memory thresholds (in MB)
  memory: {
    warning: 100,
    critical: 200
  },
  
  // Render time thresholds (in ms)
  renderTime: {
    warning: 16, // One frame at 60fps
    critical: 33  // Two frames at 60fps
  },
  
  // Bundle size thresholds (in KB)
  bundleSize: {
    warning: 500,
    critical: 1000
  },
  
  // Network thresholds
  network: {
    latency: {
      warning: 100, // ms
      critical: 500
    },
    throughput: {
      warning: 10,   // requests/second
      critical: 5
    }
  },
  
  // WebSocket thresholds
  webSocket: {
    connectionRetries: 5,
    messageQueueSize: 1000,
    maxConnections: 10,
    heartbeatInterval: 30000, // 30 seconds
    reconnectDelay: 3000      // 3 seconds
  }
};

// Monitoring configuration
export const MONITORING_CONFIG = {
  prometheus: {
    enabled: FEATURE_FLAGS.enableAdvancedMonitoring,
    endpoint: `${PROMETHEUS_URL}/api/v1`,
    scrapeInterval: 15000, // 15 seconds
    metrics: [
      'http_requests_total',
      'http_request_duration_seconds',
      'websocket_connections_active',
      'websocket_messages_total',
      'component_render_duration_seconds',
      'javascript_heap_size_bytes',
      'trading_orders_total',
      'risk_alerts_total'
    ]
  },
  
  grafana: {
    enabled: FEATURE_FLAGS.enableAdvancedMonitoring,
    dashboardUrl: `${GRAFANA_URL}/d/nautilus-overview`,
    embedUrl: `${GRAFANA_URL}/dashboard-solo/db/nautilus-overview`,
    orgId: 1,
    refresh: '5s',
    timeRange: {
      from: 'now-1h',
      to: 'now'
    }
  },
  
  errorReporting: {
    enabled: FEATURE_FLAGS.enableErrorReporting,
    dsn: import.meta.env.VITE_SENTRY_DSN,
    environment: import.meta.env.MODE,
    tracesSampleRate: isProduction ? 0.1 : 1.0,
    beforeSend: (event: any) => {
      // Filter out development-only errors
      if (isDevelopment && event.exception) {
        return null;
      }
      return event;
    }
  },
  
  analytics: {
    enabled: FEATURE_FLAGS.enableAnalytics && isProduction,
    trackingId: import.meta.env.VITE_GA_TRACKING_ID,
    events: {
      pageView: true,
      userInteraction: true,
      performance: true,
      errors: true,
      webVitals: true
    }
  }
};

// Security configuration
export const SECURITY_CONFIG = {
  csp: {
    enabled: isProduction,
    directives: {
      'default-src': ["'self'"],
      'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      'font-src': ["'self'", "https://fonts.gstatic.com"],
      'img-src': ["'self'", "data:", "https:"],
      'connect-src': [
        "'self'",
        API_BASE_URL,
        `ws://${WS_BASE_URL}`,
        `wss://${WS_BASE_URL}`,
        GRAFANA_URL,
        PROMETHEUS_URL
      ],
      'frame-src': [GRAFANA_URL],
      'worker-src': ["'self'", "blob:"],
      'manifest-src': ["'self'"]
    }
  },
  
  trustedDomains: [
    new URL(API_BASE_URL).hostname,
    new URL(GRAFANA_URL).hostname,
    new URL(PROMETHEUS_URL).hostname,
    'localhost',
    '127.0.0.1'
  ],
  
  apiKeyRequired: isProduction,
  
  headers: {
    'X-Frame-Options': 'SAMEORIGIN',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin'
  }
};

// WebSocket configuration
export const WEBSOCKET_CONFIG = {
  baseUrl: `${WS_BASE_URL.startsWith('ws') ? '' : 'ws://'}${WS_BASE_URL}`,
  endpoints: {
    engineStatus: '/ws/engine/status',
    marketData: '/ws/market-data',
    tradeUpdates: '/ws/trades/updates',
    systemHealth: '/ws/system/health',
    riskAlerts: '/ws/risk/alerts',
    strategyUpdates: '/ws/strategies/updates'
  },
  
  options: {
    maxReconnectAttempts: PERFORMANCE_THRESHOLDS.webSocket.connectionRetries,
    reconnectInterval: PERFORMANCE_THRESHOLDS.webSocket.reconnectDelay,
    heartbeatInterval: PERFORMANCE_THRESHOLDS.webSocket.heartbeatInterval,
    messageBufferSize: PERFORMANCE_THRESHOLDS.webSocket.messageQueueSize,
    enableCompression: true,
    enableBatching: true,
    batchSize: 10,
    batchTimeout: 100, // ms
    connectionTimeout: 10000, // 10 seconds
    protocols: ['nautilus-v1']
  },
  
  messageTypes: {
    HEARTBEAT: 'heartbeat',
    ENGINE_STATUS: 'engine_status',
    MARKET_DATA: 'market_data',
    TRADE_UPDATE: 'trade_update',
    RISK_ALERT: 'risk_alert',
    SYSTEM_HEALTH: 'system_health',
    STRATEGY_UPDATE: 'strategy_update',
    ERROR: 'error',
    BATCH: 'batch'
  }
};

// API configuration
export const API_CONFIG = {
  baseUrl: API_BASE_URL,
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
  retryDelay: 1000, // 1 second
  
  endpoints: {
    // System endpoints
    systemHealth: '/api/v1/system/health',
    systemMetrics: '/api/v1/system/metrics',
    
    // WebSocket endpoints
    websocketConnections: '/api/v1/websocket/connections',
    websocketSubscriptions: '/api/v1/websocket/subscriptions',
    
    // Monitoring endpoints
    prometheusMetrics: '/api/v1/monitoring/prometheus',
    grafanaDashboards: '/api/v1/monitoring/grafana',
    
    // Risk management
    riskLimits: '/api/v1/risk/limits',
    riskBreaches: '/api/v1/risk/breaches',
    riskReports: '/api/v1/risk/reports',
    
    // Strategy management
    strategies: '/api/v1/strategies',
    deployments: '/api/v1/strategies/deployments',
    pipelines: '/api/v1/strategies/pipelines',
    
    // Analytics
    analytics: '/api/v1/analytics',
    performance: '/api/v1/analytics/performance',
    
    // Data sources
    alphaVantage: '/api/v1/alpha-vantage',
    fred: '/api/v1/fred',
    edgar: '/api/v1/edgar',
    dataGov: '/api/v1/datagov',
    
    // NautilusTrader
    nautilusEngine: '/api/v1/nautilus/engine'
  },
  
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    ...(import.meta.env.VITE_API_KEY && {
      'X-API-Key': import.meta.env.VITE_API_KEY
    })
  }
};

// UI Configuration
export const UI_CONFIG = {
  theme: {
    defaultMode: import.meta.env.VITE_DEFAULT_THEME || 'light',
    enableDarkMode: FEATURE_FLAGS.enableDarkMode,
    enableSystemTheme: true,
    animation: {
      enabled: !import.meta.env.VITE_REDUCE_MOTION,
      duration: isProduction ? 150 : 300
    }
  },
  
  layout: {
    sidebarWidth: 280,
    collapsedSidebarWidth: 80,
    headerHeight: 64,
    footerHeight: 48,
    contentPadding: 24,
    enableResponsive: true,
    breakpoints: {
      mobile: 768,
      tablet: 1024,
      desktop: 1200
    }
  },
  
  virtualization: {
    enabled: true,
    overscan: 10,
    itemHeight: 54,
    threshold: 100 // Items threshold to enable virtualization
  },
  
  notifications: {
    maxCount: 5,
    duration: 4500,
    placement: 'topRight',
    showProgress: true
  },
  
  accessibility: {
    enabled: FEATURE_FLAGS.enableAccessibilityMode,
    focusOutlineWidth: 2,
    minContrastRatio: 4.5,
    enableScreenReader: true,
    enableKeyboardNavigation: true,
    announceChanges: true
  }
};

// Cache configuration
export const CACHE_CONFIG = {
  enabled: isProduction,
  version: import.meta.env.VITE_APP_VERSION || '1.0.0',
  
  strategies: {
    networkFirst: ['api/**'],
    cacheFirst: ['static/**', '*.css', '*.js'],
    staleWhileRevalidate: ['images/**']
  },
  
  ttl: {
    api: 5 * 60 * 1000,    // 5 minutes
    static: 30 * 24 * 60 * 60 * 1000, // 30 days
    images: 7 * 24 * 60 * 60 * 1000   // 7 days
  },
  
  storage: {
    quota: 100 * 1024 * 1024, // 100MB
    cleanup: {
      enabled: true,
      threshold: 0.8, // 80% of quota
      strategy: 'lru' // Least Recently Used
    }
  }
};

// Main production configuration
const productionConfig: ProductionConfig = {
  // Core configuration
  apiBaseUrl: API_BASE_URL,
  wsBaseUrl: WS_BASE_URL,
  
  // Feature toggles
  enableAnalytics: FEATURE_FLAGS.enableAnalytics,
  enableErrorReporting: FEATURE_FLAGS.enableErrorReporting,
  enablePerformanceMonitoring: FEATURE_FLAGS.enablePerformanceMonitoring,
  
  // Logging
  logLevel: isDevelopment ? 'debug' : isProduction ? 'error' : 'info',
  
  // Feature flags
  featureFlags: FEATURE_FLAGS,
  
  // Monitoring setup
  monitoring: {
    prometheus: MONITORING_CONFIG.prometheus,
    grafana: MONITORING_CONFIG.grafana
  },
  
  // Security configuration
  security: SECURITY_CONFIG
};

// Export configurations
export {
  productionConfig as default,
  FEATURE_FLAGS,
  PERFORMANCE_THRESHOLDS,
  MONITORING_CONFIG,
  SECURITY_CONFIG,
  WEBSOCKET_CONFIG,
  API_CONFIG,
  UI_CONFIG,
  CACHE_CONFIG
};

// Runtime environment info
export const RUNTIME_INFO = {
  version: import.meta.env.VITE_APP_VERSION || '1.0.0',
  buildTime: import.meta.env.VITE_BUILD_TIME || new Date().toISOString(),
  commit: import.meta.env.VITE_GIT_COMMIT || 'unknown',
  branch: import.meta.env.VITE_GIT_BRANCH || 'unknown',
  environment: import.meta.env.MODE,
  nodeEnv: import.meta.env.NODE_ENV,
  isDevelopment,
  isProduction,
  isTest
};

// Configuration validation
export const validateConfiguration = (): boolean => {
  const errors: string[] = [];
  
  // Validate required URLs
  if (!API_BASE_URL) errors.push('API_BASE_URL is required');
  if (!WS_BASE_URL) errors.push('WS_BASE_URL is required');
  
  // Validate monitoring configuration
  if (FEATURE_FLAGS.enableAdvancedMonitoring) {
    if (!PROMETHEUS_URL) errors.push('PROMETHEUS_URL is required when monitoring is enabled');
    if (!GRAFANA_URL) errors.push('GRAFANA_URL is required when monitoring is enabled');
  }
  
  // Validate error reporting
  if (FEATURE_FLAGS.enableErrorReporting && isProduction) {
    if (!import.meta.env.VITE_SENTRY_DSN) {
      console.warn('Error reporting is enabled but VITE_SENTRY_DSN is not configured');
    }
  }
  
  // Validate analytics
  if (FEATURE_FLAGS.enableAnalytics && isProduction) {
    if (!import.meta.env.VITE_GA_TRACKING_ID) {
      console.warn('Analytics is enabled but VITE_GA_TRACKING_ID is not configured');
    }
  }
  
  if (errors.length > 0) {
    console.error('Configuration validation failed:', errors);
    return false;
  }
  
  return true;
};

// Initialize configuration
if (!validateConfiguration()) {
  throw new Error('Invalid configuration detected. Check console for details.');
}

console.info('Production configuration loaded:', {
  version: RUNTIME_INFO.version,
  environment: RUNTIME_INFO.environment,
  features: Object.entries(FEATURE_FLAGS)
    .filter(([, enabled]) => enabled)
    .map(([feature]) => feature)
});

export default productionConfig;