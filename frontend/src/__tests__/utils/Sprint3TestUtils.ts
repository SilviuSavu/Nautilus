/**
 * Sprint 3 Test Utilities
 * Comprehensive test utilities and mocks for Sprint 3 components
 * 
 * Provides mock data generators, test helpers, performance testing utilities,
 * WebSocket mocks, and integration test support for all Sprint 3 features.
 */

import { vi } from 'vitest';

// Type definitions for Sprint 3 test data
export interface MockWebSocketConnection {
  id: string;
  clientId: string;
  ipAddress: string;
  status: 'connected' | 'disconnected' | 'connecting' | 'error';
  connectedAt: string;
  lastActivity: string;
  messagesPerSecond: number;
  latency: number;
  subscriptions: number;
  dataTransferred: string;
  errorCount: number;
}

export interface MockRiskLimit {
  id: string;
  name: string;
  type: 'var' | 'concentration' | 'leverage' | 'drawdown' | 'exposure';
  currentValue: number;
  limitValue: number;
  utilizationPercentage: number;
  status: 'active' | 'breached' | 'warning' | 'disabled';
  breachProbability: number;
  lastUpdated: number;
  description?: string;
}

export interface MockStrategy {
  strategyId: string;
  strategyName: string;
  version: string;
  pnl: number;
  return: number;
  sharpe: number;
  maxDrawdown: number;
  trades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  status: 'active' | 'inactive' | 'testing' | 'stopped';
}

export interface MockDeployment {
  id: string;
  strategyId: string;
  strategyName: string;
  version: string;
  status: 'pending_review' | 'approved' | 'rejected' | 'deployed' | 'failed';
  priority: 'critical' | 'high' | 'medium' | 'low';
  submittedBy: string;
  submittedAt: number;
  deploymentType: 'production' | 'staging' | 'hotfix' | 'experimental';
}

export interface MockSystemComponent {
  name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  score: number;
  cpuUsage?: number;
  memoryUsage?: number;
  responseTime?: number;
  errorRate?: number;
  uptime?: number;
  lastCheck: number;
}

/**
 * WebSocket Mock Data Generators
 */
export class WebSocketMockGenerator {
  static generateConnection(overrides: Partial<MockWebSocketConnection> = {}): MockWebSocketConnection {
    const baseConnection: MockWebSocketConnection = {
      id: `conn-${Math.random().toString(36).substr(2, 9)}`,
      clientId: `client-${Math.random().toString(36).substr(2, 6)}`,
      ipAddress: `192.168.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`,
      status: 'connected',
      connectedAt: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      lastActivity: new Date(Date.now() - Math.random() * 300000).toISOString(),
      messagesPerSecond: Math.floor(Math.random() * 1000) + 100,
      latency: Math.random() * 100 + 5,
      subscriptions: Math.floor(Math.random() * 20) + 1,
      dataTransferred: `${(Math.random() * 100).toFixed(1)} MB`,
      errorCount: Math.floor(Math.random() * 5)
    };
    
    return { ...baseConnection, ...overrides };
  }

  static generateConnectionList(count: number): MockWebSocketConnection[] {
    return Array.from({ length: count }, () => this.generateConnection());
  }

  static generateHighLoadScenario(connectionCount: number): MockWebSocketConnection[] {
    return Array.from({ length: connectionCount }, (_, i) => this.generateConnection({
      messagesPerSecond: 800 + Math.random() * 400, // High message rate
      latency: 10 + Math.random() * 20, // Low latency
      subscriptions: Math.floor(Math.random() * 15) + 5, // Multiple subscriptions
      errorCount: Math.floor(Math.random() * 2) // Low error count
    }));
  }

  static generateStressTestScenario(connectionCount: number): MockWebSocketConnection[] {
    return Array.from({ length: connectionCount }, (_, i) => this.generateConnection({
      messagesPerSecond: Math.random() * 100, // Variable load
      latency: 50 + Math.random() * 200, // High latency
      errorCount: Math.floor(Math.random() * 10), // Higher error rate
      status: Math.random() > 0.8 ? 'error' : 'connected'
    }));
  }
}

/**
 * Risk Management Mock Data Generators
 */
export class RiskMockGenerator {
  static generateRiskLimit(overrides: Partial<MockRiskLimit> = {}): MockRiskLimit {
    const types = ['var', 'concentration', 'leverage', 'drawdown', 'exposure'] as const;
    const type = types[Math.floor(Math.random() * types.length)];
    
    const baseLimit: MockRiskLimit = {
      id: `limit-${Math.random().toString(36).substr(2, 9)}`,
      name: `${type.toUpperCase()} Risk Limit`,
      type,
      currentValue: type === 'var' ? -Math.random() * 100000 : Math.random() * 100,
      limitValue: type === 'var' ? -Math.random() * 150000 : Math.random() * 120,
      utilizationPercentage: Math.random() * 100,
      status: 'active',
      breachProbability: Math.random(),
      lastUpdated: Date.now() - Math.random() * 3600000,
      description: `${type} risk limit for portfolio protection`
    };
    
    // Ensure utilization makes sense
    baseLimit.utilizationPercentage = Math.abs(baseLimit.currentValue / baseLimit.limitValue) * 100;
    
    return { ...baseLimit, ...overrides };
  }

  static generateRiskLimitList(count: number): MockRiskLimit[] {
    return Array.from({ length: count }, () => this.generateRiskLimit());
  }

  static generateBreachedLimits(count: number): MockRiskLimit[] {
    return Array.from({ length: count }, () => this.generateRiskLimit({
      status: 'breached',
      utilizationPercentage: 100 + Math.random() * 20,
      breachProbability: 0.9 + Math.random() * 0.1
    }));
  }

  static generateWarningLimits(count: number): MockRiskLimit[] {
    return Array.from({ length: count }, () => this.generateRiskLimit({
      status: 'warning',
      utilizationPercentage: 85 + Math.random() * 10,
      breachProbability: 0.7 + Math.random() * 0.2
    }));
  }
}

/**
 * Strategy Mock Data Generators
 */
export class StrategyMockGenerator {
  static generateStrategy(overrides: Partial<MockStrategy> = {}): MockStrategy {
    const strategyTypes = ['Momentum', 'Mean Reversion', 'Arbitrage', 'Trend Following', 'Market Making'];
    const strategyType = strategyTypes[Math.floor(Math.random() * strategyTypes.length)];
    
    const baseStrategy: MockStrategy = {
      strategyId: `strategy-${Math.random().toString(36).substr(2, 9)}`,
      strategyName: `${strategyType} Strategy ${Math.floor(Math.random() * 100)}`,
      version: `${Math.floor(Math.random() * 3) + 1}.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}`,
      pnl: (Math.random() - 0.3) * 100000, // Can be negative
      return: (Math.random() - 0.2) * 30, // Can be negative
      sharpe: Math.random() * 3,
      maxDrawdown: -Math.random() * 25,
      trades: Math.floor(Math.random() * 2000) + 100,
      winRate: 45 + Math.random() * 30,
      avgWin: Math.random() * 500 + 100,
      avgLoss: -(Math.random() * 300 + 50),
      status: 'active'
    };
    
    return { ...baseStrategy, ...overrides };
  }

  static generateStrategyList(count: number): MockStrategy[] {
    return Array.from({ length: count }, () => this.generateStrategy());
  }

  static generateHighPerformingStrategies(count: number): MockStrategy[] {
    return Array.from({ length: count }, () => this.generateStrategy({
      pnl: Math.random() * 100000 + 50000,
      return: Math.random() * 25 + 10,
      sharpe: Math.random() * 2 + 1.5,
      maxDrawdown: -Math.random() * 8,
      winRate: 60 + Math.random() * 20
    }));
  }

  static generateUnderperformingStrategies(count: number): MockStrategy[] {
    return Array.from({ length: count }, () => this.generateStrategy({
      pnl: -Math.random() * 50000,
      return: -Math.random() * 15,
      sharpe: Math.random() * 0.5,
      maxDrawdown: -Math.random() * 20 - 15,
      winRate: 30 + Math.random() * 20,
      status: Math.random() > 0.5 ? 'testing' : 'stopped'
    }));
  }
}

/**
 * Deployment Mock Data Generators
 */
export class DeploymentMockGenerator {
  static generateDeployment(overrides: Partial<MockDeployment> = {}): MockDeployment {
    const baseDeployment: MockDeployment = {
      id: `deployment-${Math.random().toString(36).substr(2, 9)}`,
      strategyId: `strategy-${Math.random().toString(36).substr(2, 9)}`,
      strategyName: `Strategy ${Math.floor(Math.random() * 100)}`,
      version: `${Math.floor(Math.random() * 3) + 1}.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}`,
      status: 'pending_review',
      priority: 'medium',
      submittedBy: `user${Math.floor(Math.random() * 100)}@company.com`,
      submittedAt: Date.now() - Math.random() * 86400000,
      deploymentType: 'production'
    };
    
    return { ...baseDeployment, ...overrides };
  }

  static generateDeploymentQueue(count: number): MockDeployment[] {
    return Array.from({ length: count }, () => this.generateDeployment());
  }

  static generatePendingDeployments(count: number): MockDeployment[] {
    return Array.from({ length: count }, () => this.generateDeployment({
      status: 'pending_review',
      submittedAt: Date.now() - Math.random() * 3600000 // Within last hour
    }));
  }

  static generateCriticalDeployments(count: number): MockDeployment[] {
    return Array.from({ length: count }, () => this.generateDeployment({
      priority: 'critical',
      deploymentType: 'hotfix',
      submittedAt: Date.now() - Math.random() * 1800000 // Within last 30 minutes
    }));
  }
}

/**
 * System Monitoring Mock Data Generators
 */
export class SystemMockGenerator {
  static generateSystemComponent(overrides: Partial<MockSystemComponent> = {}): MockSystemComponent {
    const componentNames = [
      'WebSocket Infrastructure',
      'Risk Management',
      'Analytics Engine',
      'Strategy Deployment',
      'Database',
      'Message Queue',
      'NautilusTrader Engine'
    ];
    
    const baseComponent: MockSystemComponent = {
      name: componentNames[Math.floor(Math.random() * componentNames.length)],
      status: 'healthy',
      score: 85 + Math.random() * 15,
      cpuUsage: Math.random() * 60 + 10,
      memoryUsage: Math.random() * 70 + 20,
      responseTime: Math.random() * 100 + 10,
      errorRate: Math.random() * 2,
      uptime: 95 + Math.random() * 5,
      lastCheck: Date.now() - Math.random() * 300000
    };
    
    return { ...baseComponent, ...overrides };
  }

  static generateSystemComponents(count: number): MockSystemComponent[] {
    return Array.from({ length: count }, () => this.generateSystemComponent());
  }

  static generateHealthySystem(): MockSystemComponent[] {
    return [
      this.generateSystemComponent({ 
        name: 'WebSocket Infrastructure', 
        status: 'healthy', 
        score: 98.2,
        cpuUsage: 25,
        memoryUsage: 45,
        errorRate: 0.1
      }),
      this.generateSystemComponent({ 
        name: 'Risk Management', 
        status: 'healthy', 
        score: 96.5,
        cpuUsage: 30,
        memoryUsage: 55,
        errorRate: 0.2
      }),
      this.generateSystemComponent({ 
        name: 'Analytics Engine', 
        status: 'healthy', 
        score: 94.8,
        cpuUsage: 40,
        memoryUsage: 60,
        errorRate: 0.15
      })
    ];
  }

  static generateStressedSystem(): MockSystemComponent[] {
    return [
      this.generateSystemComponent({ 
        name: 'WebSocket Infrastructure', 
        status: 'warning', 
        score: 78.2,
        cpuUsage: 85,
        memoryUsage: 90,
        errorRate: 2.5
      }),
      this.generateSystemComponent({ 
        name: 'Risk Management', 
        status: 'critical', 
        score: 65.1,
        cpuUsage: 95,
        memoryUsage: 95,
        errorRate: 5.2
      }),
      this.generateSystemComponent({ 
        name: 'Analytics Engine', 
        status: 'warning', 
        score: 72.3,
        cpuUsage: 80,
        memoryUsage: 85,
        errorRate: 3.1
      })
    ];
  }
}

/**
 * Performance Testing Utilities
 */
export class PerformanceTestUtils {
  static measureRenderTime(renderFn: () => void): number {
    const start = performance.now();
    renderFn();
    const end = performance.now();
    return end - start;
  }

  static async measureAsyncOperation<T>(operation: () => Promise<T>): Promise<{ result: T; duration: number }> {
    const start = performance.now();
    const result = await operation();
    const end = performance.now();
    return { result, duration: end - start };
  }

  static simulateHighLoad(operations: (() => void)[], concurrency: number = 10): Promise<void[]> {
    const chunks = [];
    for (let i = 0; i < operations.length; i += concurrency) {
      chunks.push(operations.slice(i, i + concurrency));
    }

    return chunks.reduce(
      (promise, chunk) => promise.then(() => Promise.all(chunk.map(op => Promise.resolve(op())))),
      Promise.resolve([])
    );
  }

  static generateLargeDataset<T>(generator: () => T, size: number): T[] {
    return Array.from({ length: size }, generator);
  }
}

/**
 * WebSocket Mock Implementation
 */
export class MockWebSocket {
  static create(options: { 
    url?: string; 
    protocols?: string[]; 
    readyState?: number;
    autoConnect?: boolean;
  } = {}) {
    const mockWebSocket = {
      CONNECTING: 0,
      OPEN: 1,
      CLOSING: 2,
      CLOSED: 3,
      readyState: options.readyState ?? 1,
      url: options.url ?? 'ws://localhost:8001/ws',
      protocol: '',
      extensions: '',
      bufferedAmount: 0,
      binaryType: 'blob',
      
      // Event handlers
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
      
      // Methods
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
      
      // Test utilities
      simulateOpen() {
        if (this.onopen) this.onopen(new Event('open'));
        this.readyState = this.OPEN;
      },
      
      simulateClose(code = 1000, reason = '') {
        if (this.onclose) this.onclose({ code, reason, wasClean: true });
        this.readyState = this.CLOSED;
      },
      
      simulateMessage(data: string | object) {
        if (this.onmessage) {
          const messageData = typeof data === 'object' ? JSON.stringify(data) : data;
          this.onmessage({ data: messageData, origin: this.url, lastEventId: '', source: null, ports: [] });
        }
      },
      
      simulateError(error = 'Connection error') {
        if (this.onerror) this.onerror(new Event('error'));
      }
    };
    
    if (options.autoConnect) {
      setTimeout(() => mockWebSocket.simulateOpen(), 10);
    }
    
    return mockWebSocket;
  }
}

/**
 * Test Data Presets
 */
export const TestDataPresets = {
  // WebSocket presets
  webSocket: {
    healthyConnections: () => WebSocketMockGenerator.generateConnectionList(50),
    highLoadConnections: () => WebSocketMockGenerator.generateHighLoadScenario(1000),
    stressTestConnections: () => WebSocketMockGenerator.generateStressTestScenario(2000)
  },
  
  // Risk management presets
  risk: {
    healthyLimits: () => RiskMockGenerator.generateRiskLimitList(10),
    warningLimits: () => RiskMockGenerator.generateWarningLimits(5),
    breachedLimits: () => RiskMockGenerator.generateBreachedLimits(2)
  },
  
  // Strategy presets
  strategies: {
    balanced: () => StrategyMockGenerator.generateStrategyList(20),
    highPerformers: () => StrategyMockGenerator.generateHighPerformingStrategies(10),
    underPerformers: () => StrategyMockGenerator.generateUnderperformingStrategies(5)
  },
  
  // Deployment presets
  deployments: {
    normalQueue: () => DeploymentMockGenerator.generateDeploymentQueue(15),
    pendingApprovals: () => DeploymentMockGenerator.generatePendingDeployments(8),
    criticalDeployments: () => DeploymentMockGenerator.generateCriticalDeployments(3)
  },
  
  // System monitoring presets
  system: {
    healthy: () => SystemMockGenerator.generateHealthySystem(),
    stressed: () => SystemMockGenerator.generateStressedSystem(),
    mixed: () => [
      ...SystemMockGenerator.generateHealthySystem(),
      ...SystemMockGenerator.generateStressedSystem()
    ]
  }
};

/**
 * Test Environment Setup
 */
export class TestEnvironmentSetup {
  static setupSprint3Environment() {
    // Mock environment variables
    process.env.VITE_API_BASE_URL = 'http://localhost:8001';
    process.env.VITE_WS_URL = 'localhost:8001';
    process.env.NODE_ENV = 'test';
    
    // Mock WebSocket globally
    global.WebSocket = vi.fn(() => MockWebSocket.create({ autoConnect: true }));
    
    // Mock performance API
    if (!global.performance) {
      global.performance = {
        now: () => Date.now(),
        mark: vi.fn(),
        measure: vi.fn(),
        clearMarks: vi.fn(),
        clearMeasures: vi.fn(),
        getEntries: vi.fn(() => []),
        getEntriesByName: vi.fn(() => []),
        getEntriesByType: vi.fn(() => [])
      };
    }
    
    // Mock ResizeObserver
    global.ResizeObserver = vi.fn(() => ({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn()
    }));
    
    // Mock IntersectionObserver
    global.IntersectionObserver = vi.fn(() => ({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn(),
      root: null,
      rootMargin: '',
      thresholds: []
    }));
    
    // Mock localStorage
    const localStorageMock = {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn(),
      clear: vi.fn()
    };
    Object.defineProperty(window, 'localStorage', {
      value: localStorageMock
    });
    
    // Mock sessionStorage
    Object.defineProperty(window, 'sessionStorage', {
      value: localStorageMock
    });
  }
  
  static cleanupSprint3Environment() {
    vi.clearAllMocks();
    vi.clearAllTimers();
  }
}

/**
 * Custom Test Matchers
 */
export const customMatchers = {
  toBeHealthyComponent: (component: MockSystemComponent) => {
    return {
      pass: component.status === 'healthy' && component.score > 90,
      message: () => `Expected component to be healthy with score > 90, got ${component.status} with score ${component.score}`
    };
  },
  
  toHaveAcceptablePerformance: (duration: number, threshold: number = 100) => {
    return {
      pass: duration <= threshold,
      message: () => `Expected operation to complete within ${threshold}ms, but took ${duration}ms`
    };
  },
  
  toBeWithinRange: (value: number, min: number, max: number) => {
    return {
      pass: value >= min && value <= max,
      message: () => `Expected ${value} to be between ${min} and ${max}`
    };
  }
};

export default {
  WebSocketMockGenerator,
  RiskMockGenerator,
  StrategyMockGenerator,
  DeploymentMockGenerator,
  SystemMockGenerator,
  PerformanceTestUtils,
  MockWebSocket,
  TestDataPresets,
  TestEnvironmentSetup,
  customMatchers
};