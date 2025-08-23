/**
 * Performance Monitoring Hook
 * Real-time performance tracking and optimization suggestions
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { UI_CONSTANTS } from '../constants/ui';

export interface PerformanceMetrics {
  // Memory metrics
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  memoryUsagePercent: number;
  
  // Timing metrics
  domContentLoaded: number;
  loadComplete: number;
  firstPaint: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  firstInputDelay: number;
  cumulativeLayoutShift: number;
  
  // Component metrics
  renderTime: number;
  updateCount: number;
  reRenderCount: number;
  lastRenderTime: number;
  
  // Network metrics
  connectionType: string;
  effectiveType: string;
  downlink: number;
  rtt: number;
  
  // Frame metrics
  fps: number;
  frameTime: number;
  droppedFrames: number;
}

export interface PerformanceAlert {
  id: string;
  type: 'memory' | 'performance' | 'network' | 'rendering';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  value: number;
  threshold: number;
  suggestion: string;
  timestamp: number;
}

export interface UsePerformanceMonitorOptions {
  /** Enable performance monitoring */
  enabled?: boolean;
  /** Monitoring interval in milliseconds */
  interval?: number;
  /** Memory threshold for alerts (percentage) */
  memoryThreshold?: number;
  /** FPS threshold for alerts */
  fpsThreshold?: number;
  /** Enable Web Vitals monitoring */
  enableWebVitals?: boolean;
  /** Enable component performance tracking */
  trackComponentPerformance?: boolean;
  /** Maximum number of alerts to keep */
  maxAlerts?: number;
  /** Enable automatic optimization suggestions */
  enableOptimizationSuggestions?: boolean;
}

export interface UsePerformanceMonitorReturn {
  /** Current performance metrics */
  metrics: PerformanceMetrics | null;
  /** Performance alerts */
  alerts: PerformanceAlert[];
  /** Loading state */
  loading: boolean;
  /** Overall performance score (0-100) */
  performanceScore: number;
  /** Performance grade (A-F) */
  performanceGrade: string;
  /** Start monitoring */
  startMonitoring: () => void;
  /** Stop monitoring */
  stopMonitoring: () => void;
  /** Clear alerts */
  clearAlerts: () => void;
  /** Mark component render */
  markRender: (componentName?: string, renderTime?: number) => void;
  /** Get optimization suggestions */
  getOptimizationSuggestions: () => string[];
  /** Export performance data */
  exportData: () => any;
}

declare global {
  interface Performance {
    memory?: {
      usedJSHeapSize: number;
      totalJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  }
  
  interface Navigator {
    connection?: {
      effectiveType: string;
      downlink: number;
      rtt: number;
    };
  }
}

export const usePerformanceMonitor = (
  options: UsePerformanceMonitorOptions = {}
): UsePerformanceMonitorReturn => {
  const {
    enabled = true,
    interval = 5000,
    memoryThreshold = 80,
    fpsThreshold = 30,
    enableWebVitals = true,
    trackComponentPerformance = true,
    maxAlerts = 50,
    enableOptimizationSuggestions = true
  } = options;

  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [loading, setLoading] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);

  // Refs for tracking
  const monitoringIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const renderCountRef = useRef(0);
  const componentRenderTimesRef = useRef<Map<string, number[]>>(new Map());
  const performanceObserverRef = useRef<PerformanceObserver | null>(null);

  // Calculate performance score
  const performanceScore = useMemo(() => {
    if (!metrics) return 0;

    let score = 100;
    
    // Memory usage impact
    if (metrics.memoryUsagePercent > 80) score -= 20;
    else if (metrics.memoryUsagePercent > 60) score -= 10;
    
    // FPS impact
    if (metrics.fps < 30) score -= 30;
    else if (metrics.fps < 45) score -= 15;
    
    // Web Vitals impact
    if (metrics.largestContentfulPaint > 4000) score -= 15;
    else if (metrics.largestContentfulPaint > 2500) score -= 8;
    
    if (metrics.firstInputDelay > 300) score -= 15;
    else if (metrics.firstInputDelay > 100) score -= 8;
    
    if (metrics.cumulativeLayoutShift > 0.25) score -= 10;
    else if (metrics.cumulativeLayoutShift > 0.1) score -= 5;

    // Network impact
    if (metrics.effectiveType === 'slow-2g' || metrics.effectiveType === '2g') {
      score -= 10;
    }

    return Math.max(0, Math.min(100, score));
  }, [metrics]);

  // Calculate performance grade
  const performanceGrade = useMemo(() => {
    if (performanceScore >= 90) return 'A';
    if (performanceScore >= 80) return 'B';
    if (performanceScore >= 70) return 'C';
    if (performanceScore >= 60) return 'D';
    return 'F';
  }, [performanceScore]);

  // Get memory info
  const getMemoryInfo = useCallback((): Partial<PerformanceMetrics> => {
    if (performance.memory) {
      const { usedJSHeapSize, totalJSHeapSize, jsHeapSizeLimit } = performance.memory;
      const memoryUsagePercent = (usedJSHeapSize / jsHeapSizeLimit) * 100;
      
      return {
        usedJSHeapSize,
        totalJSHeapSize,
        jsHeapSizeLimit,
        memoryUsagePercent
      };
    }
    return {};
  }, []);

  // Get timing info
  const getTimingInfo = useCallback((): Partial<PerformanceMetrics> => {
    const timing = performance.timing;
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    
    return {
      domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
      loadComplete: timing.loadEventEnd - timing.navigationStart,
      firstPaint: 0, // Will be updated by observer
      firstContentfulPaint: 0, // Will be updated by observer
      largestContentfulPaint: 0, // Will be updated by observer
      firstInputDelay: 0 // Will be updated by observer
    };
  }, []);

  // Get network info
  const getNetworkInfo = useCallback((): Partial<PerformanceMetrics> => {
    const connection = navigator.connection;
    if (connection) {
      return {
        connectionType: connection.effectiveType,
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt
      };
    }
    return {
      connectionType: 'unknown',
      effectiveType: 'unknown',
      downlink: 0,
      rtt: 0
    };
  }, []);

  // Calculate FPS
  const calculateFPS = useCallback(() => {
    const now = performance.now();
    const delta = now - lastFrameTimeRef.current;
    
    if (delta >= 1000) {
      const fps = Math.round((frameCountRef.current * 1000) / delta);
      frameCountRef.current = 0;
      lastFrameTimeRef.current = now;
      return fps;
    }
    
    frameCountRef.current++;
    return 0;
  }, []);

  // Collect all metrics
  const collectMetrics = useCallback(async (): Promise<PerformanceMetrics> => {
    const memory = getMemoryInfo();
    const timing = getTimingInfo();
    const network = getNetworkInfo();
    const fps = calculateFPS();
    
    const now = performance.now();
    
    return {
      // Memory
      usedJSHeapSize: memory.usedJSHeapSize || 0,
      totalJSHeapSize: memory.totalJSHeapSize || 0,
      jsHeapSizeLimit: memory.jsHeapSizeLimit || 0,
      memoryUsagePercent: memory.memoryUsagePercent || 0,
      
      // Timing
      domContentLoaded: timing.domContentLoaded || 0,
      loadComplete: timing.loadComplete || 0,
      firstPaint: timing.firstPaint || 0,
      firstContentfulPaint: timing.firstContentfulPaint || 0,
      largestContentfulPaint: timing.largestContentfulPaint || 0,
      firstInputDelay: timing.firstInputDelay || 0,
      cumulativeLayoutShift: 0, // Will be updated by observer
      
      // Component
      renderTime: 0,
      updateCount: renderCountRef.current,
      reRenderCount: 0,
      lastRenderTime: now,
      
      // Network
      connectionType: network.connectionType || 'unknown',
      effectiveType: network.effectiveType || 'unknown',
      downlink: network.downlink || 0,
      rtt: network.rtt || 0,
      
      // Frame
      fps: fps || 60,
      frameTime: 16.67, // Assuming 60fps target
      droppedFrames: 0
    };
  }, [getMemoryInfo, getTimingInfo, getNetworkInfo, calculateFPS]);

  // Check for performance issues and create alerts
  const checkPerformanceAlerts = useCallback((currentMetrics: PerformanceMetrics) => {
    const newAlerts: PerformanceAlert[] = [];
    
    // Memory alerts
    if (currentMetrics.memoryUsagePercent > memoryThreshold) {
      newAlerts.push({
        id: `memory-${Date.now()}`,
        type: 'memory',
        severity: currentMetrics.memoryUsagePercent > 90 ? 'critical' : 'high',
        message: `High memory usage: ${currentMetrics.memoryUsagePercent.toFixed(1)}%`,
        value: currentMetrics.memoryUsagePercent,
        threshold: memoryThreshold,
        suggestion: 'Consider clearing unnecessary data, optimizing component renders, or implementing virtualization for large lists.',
        timestamp: Date.now()
      });
    }

    // FPS alerts
    if (currentMetrics.fps < fpsThreshold) {
      newAlerts.push({
        id: `fps-${Date.now()}`,
        type: 'performance',
        severity: currentMetrics.fps < 20 ? 'critical' : 'high',
        message: `Low FPS detected: ${currentMetrics.fps}`,
        value: currentMetrics.fps,
        threshold: fpsThreshold,
        suggestion: 'Reduce complex animations, optimize render cycles, or use React.memo for expensive components.',
        timestamp: Date.now()
      });
    }

    // Web Vitals alerts
    if (currentMetrics.largestContentfulPaint > 4000) {
      newAlerts.push({
        id: `lcp-${Date.now()}`,
        type: 'performance',
        severity: 'high',
        message: `Poor LCP: ${currentMetrics.largestContentfulPaint}ms`,
        value: currentMetrics.largestContentfulPaint,
        threshold: 4000,
        suggestion: 'Optimize images, reduce render-blocking resources, or implement code splitting.',
        timestamp: Date.now()
      });
    }

    if (newAlerts.length > 0) {
      setAlerts(prev => [...prev, ...newAlerts].slice(-maxAlerts));
    }
  }, [memoryThreshold, fpsThreshold, maxAlerts]);

  // Start monitoring
  const startMonitoring = useCallback(() => {
    if (!enabled || isMonitoring) return;

    setIsMonitoring(true);
    setLoading(true);

    // Set up performance observer for Web Vitals
    if (enableWebVitals && 'PerformanceObserver' in window) {
      try {
        performanceObserverRef.current = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          
          entries.forEach(entry => {
            setMetrics(prev => {
              if (!prev) return prev;
              
              const updated = { ...prev };
              
              switch (entry.entryType) {
                case 'paint':
                  if (entry.name === 'first-paint') {
                    updated.firstPaint = entry.startTime;
                  } else if (entry.name === 'first-contentful-paint') {
                    updated.firstContentfulPaint = entry.startTime;
                  }
                  break;
                case 'largest-contentful-paint':
                  updated.largestContentfulPaint = entry.startTime;
                  break;
                case 'first-input':
                  updated.firstInputDelay = (entry as any).processingStart - entry.startTime;
                  break;
                case 'layout-shift':
                  if (!(entry as any).hadRecentInput) {
                    updated.cumulativeLayoutShift += (entry as any).value;
                  }
                  break;
              }
              
              return updated;
            });
          });
        });
        
        performanceObserverRef.current.observe({ 
          entryTypes: ['paint', 'largest-contentful-paint', 'first-input', 'layout-shift'] 
        });
      } catch (e) {
        console.warn('PerformanceObserver not fully supported:', e);
      }
    }

    // Set up regular monitoring
    monitoringIntervalRef.current = setInterval(async () => {
      try {
        const newMetrics = await collectMetrics();
        setMetrics(newMetrics);
        checkPerformanceAlerts(newMetrics);
        setLoading(false);
      } catch (error) {
        console.error('Performance monitoring error:', error);
        setLoading(false);
      }
    }, interval);

  }, [enabled, isMonitoring, enableWebVitals, interval, collectMetrics, checkPerformanceAlerts]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    
    if (monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
    }
    
    if (performanceObserverRef.current) {
      performanceObserverRef.current.disconnect();
    }
  }, []);

  // Clear alerts
  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  // Mark component render
  const markRender = useCallback((componentName: string = 'unknown', renderTime?: number) => {
    if (!trackComponentPerformance) return;
    
    renderCountRef.current++;
    
    const time = renderTime || performance.now();
    
    if (!componentRenderTimesRef.current.has(componentName)) {
      componentRenderTimesRef.current.set(componentName, []);
    }
    
    const times = componentRenderTimesRef.current.get(componentName)!;
    times.push(time);
    
    // Keep only last 10 render times
    if (times.length > 10) {
      times.splice(0, times.length - 10);
    }
  }, [trackComponentPerformance]);

  // Get optimization suggestions
  const getOptimizationSuggestions = useCallback((): string[] => {
    if (!enableOptimizationSuggestions || !metrics) return [];
    
    const suggestions: string[] = [];
    
    if (metrics.memoryUsagePercent > 70) {
      suggestions.push('Consider implementing virtualization for large lists');
      suggestions.push('Use React.memo to prevent unnecessary re-renders');
      suggestions.push('Clean up event listeners and subscriptions in useEffect cleanup');
    }
    
    if (metrics.fps < 45) {
      suggestions.push('Optimize expensive calculations with useMemo and useCallback');
      suggestions.push('Reduce the complexity of component render methods');
      suggestions.push('Consider using CSS animations instead of JavaScript animations');
    }
    
    if (metrics.largestContentfulPaint > 2500) {
      suggestions.push('Implement code splitting with React.lazy');
      suggestions.push('Optimize images and use next-gen formats');
      suggestions.push('Preload critical resources');
    }
    
    if (renderCountRef.current > 100) {
      suggestions.push('Review component dependencies to reduce re-renders');
      suggestions.push('Use React DevTools Profiler to identify performance bottlenecks');
    }
    
    return suggestions;
  }, [enableOptimizationSuggestions, metrics]);

  // Export data
  const exportData = useCallback(() => {
    return {
      metrics,
      alerts,
      performanceScore,
      performanceGrade,
      componentRenderTimes: Object.fromEntries(componentRenderTimesRef.current),
      optimizationSuggestions: getOptimizationSuggestions(),
      timestamp: Date.now()
    };
  }, [metrics, alerts, performanceScore, performanceGrade, getOptimizationSuggestions]);

  // Auto-start monitoring when enabled
  useEffect(() => {
    if (enabled) {
      startMonitoring();
    }
    
    return () => {
      stopMonitoring();
    };
  }, [enabled, startMonitoring, stopMonitoring]);

  // FPS calculation loop
  useEffect(() => {
    let animationId: number;
    
    const animate = () => {
      calculateFPS();
      animationId = requestAnimationFrame(animate);
    };
    
    if (isMonitoring) {
      animationId = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isMonitoring, calculateFPS]);

  return {
    metrics,
    alerts,
    loading,
    performanceScore,
    performanceGrade,
    startMonitoring,
    stopMonitoring,
    clearAlerts,
    markRender,
    getOptimizationSuggestions,
    exportData
  };
};