/**
 * Performance Optimization Utilities
 * Sprint 3: Advanced Performance Enhancement and Optimization
 * 
 * Comprehensive performance utilities for React optimization,
 * data caching, memoization, virtual scrolling, and resource management.
 */

import { useRef, useCallback, useMemo, useEffect, useState } from 'react';

// Memoization and Caching Utilities
export class AdvancedCache<K, V> {
  private cache = new Map<string, { value: V; timestamp: number; accessCount: number }>();
  private maxSize: number;
  private ttl: number;
  private strategy: 'LRU' | 'LFU' | 'FIFO';

  constructor(
    maxSize = 1000,
    ttlMs = 300000, // 5 minutes default
    strategy: 'LRU' | 'LFU' | 'FIFO' = 'LRU'
  ) {
    this.maxSize = maxSize;
    this.ttl = ttlMs;
    this.strategy = strategy;
  }

  private getKey(key: K): string {
    return typeof key === 'string' ? key : JSON.stringify(key);
  }

  get(key: K): V | undefined {
    const keyStr = this.getKey(key);
    const item = this.cache.get(keyStr);
    
    if (!item) return undefined;
    
    // Check if expired
    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(keyStr);
      return undefined;
    }
    
    // Update access count for LFU
    if (this.strategy === 'LFU') {
      item.accessCount++;
    }
    
    // Update timestamp for LRU
    if (this.strategy === 'LRU') {
      item.timestamp = Date.now();
    }
    
    return item.value;
  }

  set(key: K, value: V): void {
    const keyStr = this.getKey(key);
    
    // If at capacity, evict according to strategy
    if (this.cache.size >= this.maxSize && !this.cache.has(keyStr)) {
      this.evict();
    }
    
    this.cache.set(keyStr, {
      value,
      timestamp: Date.now(),
      accessCount: 1
    });
  }

  private evict(): void {
    let keyToEvict: string | undefined;
    
    switch (this.strategy) {
      case 'LRU':
        // Remove least recently used (oldest timestamp)
        let oldestTime = Date.now();
        for (const [key, item] of this.cache.entries()) {
          if (item.timestamp < oldestTime) {
            oldestTime = item.timestamp;
            keyToEvict = key;
          }
        }
        break;
        
      case 'LFU':
        // Remove least frequently used (lowest access count)
        let lowestCount = Infinity;
        for (const [key, item] of this.cache.entries()) {
          if (item.accessCount < lowestCount) {
            lowestCount = item.accessCount;
            keyToEvict = key;
          }
        }
        break;
        
      case 'FIFO':
        // Remove first in (first key in the map)
        keyToEvict = this.cache.keys().next().value;
        break;
    }
    
    if (keyToEvict) {
      this.cache.delete(keyToEvict);
    }
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }

  has(key: K): boolean {
    return this.cache.has(this.getKey(key));
  }
}

// Debouncing and Throttling
export class RateLimitedExecutor {
  private timers = new Map<string, NodeJS.Timeout>();
  private lastExecution = new Map<string, number>();

  debounce<T extends (...args: any[]) => any>(
    fn: T,
    delay: number,
    key?: string
  ): (...args: Parameters<T>) => void {
    const executorKey = key || 'default';
    
    return (...args: Parameters<T>) => {
      const timer = this.timers.get(executorKey);
      if (timer) {
        clearTimeout(timer);
      }
      
      const newTimer = setTimeout(() => {
        fn.apply(null, args);
        this.timers.delete(executorKey);
      }, delay);
      
      this.timers.set(executorKey, newTimer);
    };
  }

  throttle<T extends (...args: any[]) => any>(
    fn: T,
    delay: number,
    key?: string
  ): (...args: Parameters<T>) => void {
    const executorKey = key || 'default';
    
    return (...args: Parameters<T>) => {
      const now = Date.now();
      const lastExec = this.lastExecution.get(executorKey) || 0;
      
      if (now - lastExec >= delay) {
        fn.apply(null, args);
        this.lastExecution.set(executorKey, now);
      }
    };
  }

  cleanup(): void {
    this.timers.forEach(timer => clearTimeout(timer));
    this.timers.clear();
    this.lastExecution.clear();
  }
}

// React Performance Hooks
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

export function useThrottle<T>(value: T, delay: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastExecuted = useRef<number>(Date.now());

  useEffect(() => {
    const now = Date.now();
    
    if (now >= lastExecuted.current + delay) {
      lastExecuted.current = now;
      setThrottledValue(value);
    } else {
      const timer = setTimeout(() => {
        lastExecuted.current = Date.now();
        setThrottledValue(value);
      }, delay - (now - lastExecuted.current));

      return () => clearTimeout(timer);
    }
  }, [value, delay]);

  return throttledValue;
}

export function useMemoizedCallback<T extends (...args: any[]) => any>(
  callback: T,
  deps: any[]
): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useCallback(callback, deps);
}

export function useExpensiveCalculation<T>(
  calculate: () => T,
  deps: any[],
  shouldRecalculate?: (prevDeps: any[], currentDeps: any[]) => boolean
): T {
  const prevDepsRef = useRef<any[]>();
  const resultRef = useRef<T>();
  
  const shouldUpdate = shouldRecalculate
    ? shouldRecalculate(prevDepsRef.current || [], deps)
    : JSON.stringify(prevDepsRef.current) !== JSON.stringify(deps);
    
  if (shouldUpdate || resultRef.current === undefined) {
    resultRef.current = calculate();
    prevDepsRef.current = deps;
  }
  
  return resultRef.current!;
}

// Virtual Scrolling Hook
export interface VirtualScrollOptions {
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
  estimatedItemHeight?: number;
  getItemHeight?: (index: number) => number;
}

export interface VirtualScrollResult {
  virtualItems: Array<{
    index: number;
    start: number;
    end: number;
    size: number;
  }>;
  totalHeight: number;
  scrollOffset: number;
  visibleRange: { start: number; end: number };
}

export function useVirtualScroll<T>(
  items: T[],
  options: VirtualScrollOptions
): VirtualScrollResult & {
  scrollElementProps: {
    onScroll: (event: React.UIEvent<HTMLElement>) => void;
    style: React.CSSProperties;
  };
  getItemProps: (index: number) => {
    style: React.CSSProperties;
    'data-index': number;
  };
} {
  const {
    itemHeight,
    containerHeight,
    overscan = 3,
    getItemHeight
  } = options;
  
  const [scrollOffset, setScrollOffset] = useState(0);
  
  const handleScroll = useCallback((event: React.UIEvent<HTMLElement>) => {
    setScrollOffset(event.currentTarget.scrollTop);
  }, []);
  
  const result = useMemo(() => {
    const itemCount = items.length;
    
    // Calculate which items are visible
    const startIndex = Math.max(
      0,
      Math.floor(scrollOffset / itemHeight) - overscan
    );
    const endIndex = Math.min(
      itemCount - 1,
      Math.ceil((scrollOffset + containerHeight) / itemHeight) + overscan
    );
    
    // Create virtual items
    const virtualItems = [];
    for (let i = startIndex; i <= endIndex; i++) {
      const height = getItemHeight ? getItemHeight(i) : itemHeight;
      virtualItems.push({
        index: i,
        start: i * itemHeight,
        end: i * itemHeight + height,
        size: height
      });
    }
    
    const totalHeight = itemCount * itemHeight;
    
    return {
      virtualItems,
      totalHeight,
      scrollOffset,
      visibleRange: { start: startIndex, end: endIndex }
    };
  }, [items.length, scrollOffset, itemHeight, containerHeight, overscan, getItemHeight]);
  
  const scrollElementProps = useMemo(() => ({
    onScroll: handleScroll,
    style: {
      height: containerHeight,
      overflow: 'auto'
    } as React.CSSProperties
  }), [handleScroll, containerHeight]);
  
  const getItemProps = useCallback((index: number) => ({
    style: {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: getItemHeight ? getItemHeight(index) : itemHeight,
      transform: `translateY(${index * itemHeight}px)`
    } as React.CSSProperties,
    'data-index': index
  }), [itemHeight, getItemHeight]);
  
  return {
    ...result,
    scrollElementProps,
    getItemProps
  };
}

// Resource Management
export class ResourcePool<T> {
  private available: T[] = [];
  private inUse = new Set<T>();
  private factory: () => T;
  private destroyer?: (resource: T) => void;
  private maxSize: number;
  private validator?: (resource: T) => boolean;

  constructor(
    factory: () => T,
    options: {
      maxSize?: number;
      destroyer?: (resource: T) => void;
      validator?: (resource: T) => boolean;
      initialSize?: number;
    } = {}
  ) {
    this.factory = factory;
    this.destroyer = options.destroyer;
    this.maxSize = options.maxSize || 10;
    this.validator = options.validator;
    
    // Pre-populate pool
    const initialSize = options.initialSize || 0;
    for (let i = 0; i < initialSize; i++) {
      this.available.push(this.factory());
    }
  }

  acquire(): T {
    let resource: T;
    
    // Try to reuse existing resource
    if (this.available.length > 0) {
      resource = this.available.pop()!;
      
      // Validate resource before use
      if (this.validator && !this.validator(resource)) {
        if (this.destroyer) {
          this.destroyer(resource);
        }
        resource = this.factory();
      }
    } else {
      resource = this.factory();
    }
    
    this.inUse.add(resource);
    return resource;
  }

  release(resource: T): void {
    if (this.inUse.has(resource)) {
      this.inUse.delete(resource);
      
      // Return to pool if not at capacity
      if (this.available.length < this.maxSize) {
        this.available.push(resource);
      } else if (this.destroyer) {
        this.destroyer(resource);
      }
    }
  }

  cleanup(): void {
    // Destroy all resources
    [...this.available, ...this.inUse].forEach(resource => {
      if (this.destroyer) {
        this.destroyer(resource);
      }
    });
    
    this.available = [];
    this.inUse.clear();
  }

  getStats(): { available: number; inUse: number; total: number } {
    return {
      available: this.available.length,
      inUse: this.inUse.size,
      total: this.available.length + this.inUse.size
    };
  }
}

// Batch Processing
export class BatchProcessor<T, R> {
  private queue: Array<{ item: T; resolve: (result: R) => void; reject: (error: Error) => void }> = [];
  private isProcessing = false;
  private batchSize: number;
  private delay: number;
  private processor: (items: T[]) => Promise<R[]>;

  constructor(
    processor: (items: T[]) => Promise<R[]>,
    options: { batchSize?: number; delay?: number } = {}
  ) {
    this.processor = processor;
    this.batchSize = options.batchSize || 10;
    this.delay = options.delay || 100;
  }

  async add(item: T): Promise<R> {
    return new Promise<R>((resolve, reject) => {
      this.queue.push({ item, resolve, reject });
      this.scheduleProcessing();
    });
  }

  private scheduleProcessing(): void {
    if (this.isProcessing) return;
    
    setTimeout(() => {
      this.processBatch();
    }, this.delay);
  }

  private async processBatch(): Promise<void> {
    if (this.isProcessing || this.queue.length === 0) return;
    
    this.isProcessing = true;
    
    try {
      const batch = this.queue.splice(0, this.batchSize);
      const items = batch.map(b => b.item);
      const results = await this.processor(items);
      
      // Resolve all promises in batch
      batch.forEach((batchItem, index) => {
        if (results[index] !== undefined) {
          batchItem.resolve(results[index]);
        } else {
          batchItem.reject(new Error('No result for item'));
        }
      });
    } catch (error) {
      // Reject all promises in current batch
      const batch = this.queue.splice(0, this.batchSize);
      batch.forEach(batchItem => {
        batchItem.reject(error instanceof Error ? error : new Error(String(error)));
      });
    } finally {
      this.isProcessing = false;
      
      // Process next batch if items remaining
      if (this.queue.length > 0) {
        this.scheduleProcessing();
      }
    }
  }
}

// Memory Management
export class MemoryMonitor {
  private static instance: MemoryMonitor;
  private observers: Array<(usage: MemoryInfo) => void> = [];
  private interval?: NodeJS.Timeout;

  static getInstance(): MemoryMonitor {
    if (!MemoryMonitor.instance) {
      MemoryMonitor.instance = new MemoryMonitor();
    }
    return MemoryMonitor.instance;
  }

  startMonitoring(intervalMs = 10000): void {
    if (this.interval) return;
    
    this.interval = setInterval(() => {
      const usage = this.getMemoryUsage();
      if (usage) {
        this.observers.forEach(observer => observer(usage));
      }
    }, intervalMs);
  }

  stopMonitoring(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = undefined;
    }
  }

  onMemoryUsage(callback: (usage: MemoryInfo) => void): () => void {
    this.observers.push(callback);
    
    return () => {
      const index = this.observers.indexOf(callback);
      if (index > -1) {
        this.observers.splice(index, 1);
      }
    };
  }

  getMemoryUsage(): MemoryInfo | null {
    if ('memory' in performance && (performance as any).memory) {
      return (performance as any).memory;
    }
    return null;
  }

  suggestCleanup(thresholdMB = 50): boolean {
    const usage = this.getMemoryUsage();
    if (!usage) return false;
    
    const usedMB = usage.usedJSHeapSize / (1024 * 1024);
    return usedMB > thresholdMB;
  }
}

// Performance Measurement
export class PerformanceProfiler {
  private measurements = new Map<string, number[]>();
  private activeTimers = new Map<string, number>();

  start(label: string): void {
    this.activeTimers.set(label, performance.now());
  }

  end(label: string): number {
    const startTime = this.activeTimers.get(label);
    if (startTime === undefined) {
      console.warn(`No active timer found for label: ${label}`);
      return 0;
    }
    
    const duration = performance.now() - startTime;
    this.activeTimers.delete(label);
    
    if (!this.measurements.has(label)) {
      this.measurements.set(label, []);
    }
    
    this.measurements.get(label)!.push(duration);
    
    return duration;
  }

  measure<T>(label: string, fn: () => T): T {
    this.start(label);
    try {
      return fn();
    } finally {
      this.end(label);
    }
  }

  async measureAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
    this.start(label);
    try {
      return await fn();
    } finally {
      this.end(label);
    }
  }

  getStats(label: string): {
    count: number;
    average: number;
    min: number;
    max: number;
    total: number;
  } | null {
    const measurements = this.measurements.get(label);
    if (!measurements || measurements.length === 0) return null;
    
    const total = measurements.reduce((sum, duration) => sum + duration, 0);
    
    return {
      count: measurements.length,
      average: total / measurements.length,
      min: Math.min(...measurements),
      max: Math.max(...measurements),
      total
    };
  }

  getAllStats(): Record<string, ReturnType<PerformanceProfiler['getStats']>> {
    const stats: Record<string, ReturnType<PerformanceProfiler['getStats']>> = {};
    
    this.measurements.forEach((_, label) => {
      stats[label] = this.getStats(label);
    });
    
    return stats;
  }

  clear(label?: string): void {
    if (label) {
      this.measurements.delete(label);
      this.activeTimers.delete(label);
    } else {
      this.measurements.clear();
      this.activeTimers.clear();
    }
  }
}

// React Hook for Performance Profiling
export function usePerformanceProfiler(): {
  start: (label: string) => void;
  end: (label: string) => number;
  measure: <T>(label: string, fn: () => T) => T;
  measureAsync: <T>(label: string, fn: () => Promise<T>) => Promise<T>;
  getStats: (label: string) => ReturnType<PerformanceProfiler['getStats']>;
  clear: (label?: string) => void;
} {
  const profilerRef = useRef(new PerformanceProfiler());
  const profiler = profilerRef.current;

  useEffect(() => {
    return () => {
      profiler.clear();
    };
  }, [profiler]);

  return {
    start: useCallback((label: string) => profiler.start(label), [profiler]),
    end: useCallback((label: string) => profiler.end(label), [profiler]),
    measure: useCallback(<T>(label: string, fn: () => T) => profiler.measure(label, fn), [profiler]),
    measureAsync: useCallback(<T>(label: string, fn: () => Promise<T>) => profiler.measureAsync(label, fn), [profiler]),
    getStats: useCallback((label: string) => profiler.getStats(label), [profiler]),
    clear: useCallback((label?: string) => profiler.clear(label), [profiler])
  };
}

// Export global instances
export const globalCache = new AdvancedCache(2000, 600000); // 10 minutes TTL
export const globalRateLimiter = new RateLimitedExecutor();
export const globalResourcePool = new ResourcePool(() => ({})); // Generic resource pool
export const globalBatchProcessor = new BatchProcessor(async (items: any[]) => items); // Generic batch processor
export const globalMemoryMonitor = MemoryMonitor.getInstance();
export const globalProfiler = new PerformanceProfiler();

// Utility functions
export const withPerformanceMonitoring = <T extends (...args: any[]) => any>(
  fn: T,
  label: string
): T => {
  return ((...args: Parameters<T>) => {
    return globalProfiler.measure(label, () => fn.apply(null, args));
  }) as T;
};

export const withAsyncPerformanceMonitoring = <T extends (...args: any[]) => Promise<any>>(
  fn: T,
  label: string
): T => {
  return (async (...args: Parameters<T>) => {
    return globalProfiler.measureAsync(label, () => fn.apply(null, args));
  }) as T;
};

export const memoizeWithCache = <K, V>(
  fn: (key: K) => V,
  cache: AdvancedCache<K, V> = globalCache as any
) => {
  return (key: K): V => {
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    
    const result = fn(key);
    cache.set(key, result);
    return result;
  };
};

export const createMemoizedAsyncFunction = <K, V>(
  fn: (key: K) => Promise<V>,
  cache: AdvancedCache<K, Promise<V>> = new AdvancedCache()
) => {
  return (key: K): Promise<V> => {
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    
    const promise = fn(key);
    cache.set(key, promise);
    
    // Remove from cache if promise rejects
    promise.catch(() => {
      cache.set(key, undefined as any);
    });
    
    return promise;
  };
};