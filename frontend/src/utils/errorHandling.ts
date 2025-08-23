/**
 * Error Handling Utilities
 * Sprint 3: Comprehensive Error Management and Recovery
 * 
 * Advanced error handling utilities for WebSocket connections,
 * API calls, data validation, and automatic recovery mechanisms.
 */

// Error Types and Classifications
export enum ErrorCategory {
  NETWORK = 'network',
  WEBSOCKET = 'websocket',
  API = 'api',
  VALIDATION = 'validation',
  AUTHENTICATION = 'authentication',
  RATE_LIMIT = 'rate_limit',
  DATA_PROCESSING = 'data_processing',
  CALCULATION = 'calculation',
  DEPLOYMENT = 'deployment',
  STRATEGY = 'strategy',
  RISK = 'risk',
  PERFORMANCE = 'performance',
  UNKNOWN = 'unknown'
}

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface ErrorContext {
  userId?: string;
  sessionId?: string;
  portfolioId?: string;
  strategyId?: string;
  component?: string;
  action?: string;
  timestamp: string;
  userAgent?: string;
  url?: string;
  additionalData?: Record<string, any>;
}

export interface ErrorDetails {
  id: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  message: string;
  code?: string;
  stack?: string;
  context: ErrorContext;
  retryable: boolean;
  recoverable: boolean;
  correlationId?: string;
}

export interface RetryConfig {
  maxAttempts: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  jitter: boolean;
}

export interface RecoveryAction {
  id: string;
  name: string;
  description: string;
  execute: () => Promise<void>;
  conditions: (error: ErrorDetails) => boolean;
  priority: number;
}

// Enhanced Error Class
export class EnhancedError extends Error {
  public readonly id: string;
  public readonly category: ErrorCategory;
  public readonly severity: ErrorSeverity;
  public readonly context: ErrorContext;
  public readonly retryable: boolean;
  public readonly recoverable: boolean;
  public readonly correlationId?: string;
  public readonly code?: string;

  constructor(
    message: string,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Partial<ErrorContext> = {},
    options: {
      retryable?: boolean;
      recoverable?: boolean;
      correlationId?: string;
      code?: string;
      cause?: Error;
    } = {}
  ) {
    super(message);
    
    this.name = 'EnhancedError';
    this.id = this.generateErrorId();
    this.category = category;
    this.severity = severity;
    this.context = {
      timestamp: new Date().toISOString(),
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
      url: typeof window !== 'undefined' ? window.location.href : undefined,
      ...context
    };
    this.retryable = options.retryable ?? this.isRetryableByCategory(category);
    this.recoverable = options.recoverable ?? this.isRecoverableByCategory(category);
    this.correlationId = options.correlationId;
    this.code = options.code;

    if (options.cause) {
      this.cause = options.cause;
      this.stack = options.cause.stack;
    }
  }

  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private isRetryableByCategory(category: ErrorCategory): boolean {
    return [
      ErrorCategory.NETWORK,
      ErrorCategory.WEBSOCKET,
      ErrorCategory.RATE_LIMIT,
      ErrorCategory.API
    ].includes(category);
  }

  private isRecoverableByCategory(category: ErrorCategory): boolean {
    return [
      ErrorCategory.NETWORK,
      ErrorCategory.WEBSOCKET,
      ErrorCategory.RATE_LIMIT,
      ErrorCategory.DATA_PROCESSING
    ].includes(category);
  }

  toDetails(): ErrorDetails {
    return {
      id: this.id,
      category: this.category,
      severity: this.severity,
      message: this.message,
      code: this.code,
      stack: this.stack,
      context: this.context,
      retryable: this.retryable,
      recoverable: this.recoverable,
      correlationId: this.correlationId
    };
  }
}

// Error Handler Class
export class ErrorHandler {
  private errorLog: ErrorDetails[] = [];
  private recoveryActions: RecoveryAction[] = [];
  private errorCounts: Map<string, number> = new Map();
  private lastErrors: Map<string, number> = new Map();
  private maxLogSize = 1000;

  // Register recovery action
  registerRecoveryAction(action: RecoveryAction): void {
    this.recoveryActions.push(action);
    this.recoveryActions.sort((a, b) => a.priority - b.priority);
  }

  // Handle error with automatic recovery
  async handleError(error: Error | EnhancedError, context?: Partial<ErrorContext>): Promise<void> {
    const enhancedError = this.enhanceError(error, context);
    const errorDetails = enhancedError.toDetails();
    
    // Log error
    this.logError(errorDetails);
    
    // Update error counts
    this.updateErrorCounts(errorDetails);
    
    // Attempt recovery if applicable
    if (errorDetails.recoverable) {
      await this.attemptRecovery(errorDetails);
    }
    
    // Report error to monitoring systems
    this.reportError(errorDetails);
  }

  // Retry mechanism with exponential backoff
  async retryWithBackoff<T>(
    operation: () => Promise<T>,
    config: Partial<RetryConfig> = {},
    context?: Partial<ErrorContext>
  ): Promise<T> {
    const retryConfig: RetryConfig = {
      maxAttempts: 3,
      baseDelay: 1000,
      maxDelay: 30000,
      backoffMultiplier: 2,
      jitter: true,
      ...config
    };

    let lastError: Error | null = null;
    
    for (let attempt = 1; attempt <= retryConfig.maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        const enhancedError = this.enhanceError(lastError, context);
        
        // Don't retry if error is not retryable
        if (!enhancedError.retryable || attempt === retryConfig.maxAttempts) {
          await this.handleError(enhancedError, context);
          throw enhancedError;
        }
        
        // Calculate delay with exponential backoff and jitter
        let delay = Math.min(
          retryConfig.baseDelay * Math.pow(retryConfig.backoffMultiplier, attempt - 1),
          retryConfig.maxDelay
        );
        
        if (retryConfig.jitter) {
          delay = delay * (0.5 + Math.random() * 0.5); // 50-100% of calculated delay
        }
        
        console.warn(`Retry attempt ${attempt}/${retryConfig.maxAttempts} after ${delay}ms:`, enhancedError.message);
        await this.sleep(delay);
      }
    }
    
    throw lastError;
  }

  // Circuit breaker pattern
  createCircuitBreaker<T>(
    operation: () => Promise<T>,
    options: {
      failureThreshold: number;
      resetTimeout: number;
      monitoringPeriod: number;
    }
  ): () => Promise<T> {
    const { failureThreshold, resetTimeout, monitoringPeriod } = options;
    let state: 'closed' | 'open' | 'half-open' = 'closed';
    let failureCount = 0;
    let lastFailureTime = 0;
    let successCount = 0;

    return async (): Promise<T> => {
      const now = Date.now();
      
      // Reset failure count if monitoring period has passed
      if (now - lastFailureTime > monitoringPeriod) {
        failureCount = 0;
      }
      
      // Check if circuit should move from open to half-open
      if (state === 'open' && now - lastFailureTime > resetTimeout) {
        state = 'half-open';
        successCount = 0;
      }
      
      // Fail fast if circuit is open
      if (state === 'open') {
        throw new EnhancedError(
          'Circuit breaker is open',
          ErrorCategory.NETWORK,
          ErrorSeverity.HIGH,
          {},
          { retryable: false }
        );
      }
      
      try {
        const result = await operation();
        
        // Reset on success
        if (state === 'half-open') {
          successCount++;
          if (successCount >= 3) { // Require multiple successes to close
            state = 'closed';
            failureCount = 0;
          }
        } else {
          failureCount = 0;
        }
        
        return result;
      } catch (error) {
        failureCount++;
        lastFailureTime = now;
        
        // Open circuit if failure threshold exceeded
        if (failureCount >= failureThreshold) {
          state = 'open';
        }
        
        throw error;
      }
    };
  }

  // Validate data with detailed error reporting
  validateData<T>(
    data: any,
    schema: ValidationSchema<T>,
    context?: Partial<ErrorContext>
  ): T {
    const errors: string[] = [];
    
    try {
      return this.validateObject(data, schema, '', errors);
    } catch (error) {
      throw new EnhancedError(
        `Validation failed: ${errors.join(', ')}`,
        ErrorCategory.VALIDATION,
        ErrorSeverity.MEDIUM,
        context || {},
        { retryable: false }
      );
    }
  }

  // Get error statistics
  getErrorStatistics(): {
    totalErrors: number;
    errorsByCategory: Record<ErrorCategory, number>;
    errorsBySeverity: Record<ErrorSeverity, number>;
    topErrors: { message: string; count: number }[];
    errorRate: number;
  } {
    const now = Date.now();
    const oneHourAgo = now - 3600000;
    const recentErrors = this.errorLog.filter(error => 
      new Date(error.context.timestamp).getTime() > oneHourAgo
    );
    
    const errorsByCategory = {} as Record<ErrorCategory, number>;
    const errorsBySeverity = {} as Record<ErrorSeverity, number>;
    const errorMessageCounts = new Map<string, number>();
    
    recentErrors.forEach(error => {
      errorsByCategory[error.category] = (errorsByCategory[error.category] || 0) + 1;
      errorsBySeverity[error.severity] = (errorsBySeverity[error.severity] || 0) + 1;
      errorMessageCounts.set(error.message, (errorMessageCounts.get(error.message) || 0) + 1);
    });
    
    const topErrors = Array.from(errorMessageCounts.entries())
      .map(([message, count]) => ({ message, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    
    const errorRate = recentErrors.length / 60; // errors per minute
    
    return {
      totalErrors: recentErrors.length,
      errorsByCategory,
      errorsBySeverity,
      topErrors,
      errorRate
    };
  }

  // Private methods
  private enhanceError(error: Error | EnhancedError, context?: Partial<ErrorContext>): EnhancedError {
    if (error instanceof EnhancedError) {
      return error;
    }
    
    // Try to categorize error based on message and type
    const category = this.categorizeError(error);
    const severity = this.determineSeverity(error, category);
    
    return new EnhancedError(
      error.message,
      category,
      severity,
      context || {},
      {
        cause: error,
        code: (error as any).code
      }
    );
  }

  private categorizeError(error: Error): ErrorCategory {
    const message = error.message.toLowerCase();
    const name = error.name.toLowerCase();
    
    if (message.includes('network') || message.includes('fetch') || name === 'networkerror') {
      return ErrorCategory.NETWORK;
    }
    if (message.includes('websocket') || message.includes('connection')) {
      return ErrorCategory.WEBSOCKET;
    }
    if (message.includes('unauthorized') || message.includes('forbidden')) {
      return ErrorCategory.AUTHENTICATION;
    }
    if (message.includes('rate limit') || message.includes('throttle')) {
      return ErrorCategory.RATE_LIMIT;
    }
    if (message.includes('validation') || message.includes('invalid')) {
      return ErrorCategory.VALIDATION;
    }
    if (name === 'typeerror' || name === 'referenceerror') {
      return ErrorCategory.DATA_PROCESSING;
    }
    
    return ErrorCategory.UNKNOWN;
  }

  private determineSeverity(error: Error, category: ErrorCategory): ErrorSeverity {
    const message = error.message.toLowerCase();
    
    // Critical errors
    if (message.includes('critical') || message.includes('fatal')) {
      return ErrorSeverity.CRITICAL;
    }
    
    // High severity based on category
    if ([ErrorCategory.AUTHENTICATION, ErrorCategory.RISK].includes(category)) {
      return ErrorSeverity.HIGH;
    }
    
    // Medium severity for network and API errors
    if ([ErrorCategory.NETWORK, ErrorCategory.API, ErrorCategory.WEBSOCKET].includes(category)) {
      return ErrorSeverity.MEDIUM;
    }
    
    return ErrorSeverity.LOW;
  }

  private logError(errorDetails: ErrorDetails): void {
    this.errorLog.unshift(errorDetails);
    
    // Trim log to max size
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog = this.errorLog.slice(0, this.maxLogSize);
    }
    
    // Console logging with appropriate level
    switch (errorDetails.severity) {
      case ErrorSeverity.CRITICAL:
        console.error('CRITICAL ERROR:', errorDetails);
        break;
      case ErrorSeverity.HIGH:
        console.error('HIGH SEVERITY ERROR:', errorDetails);
        break;
      case ErrorSeverity.MEDIUM:
        console.warn('MEDIUM SEVERITY ERROR:', errorDetails);
        break;
      default:
        console.log('LOW SEVERITY ERROR:', errorDetails);
    }
  }

  private updateErrorCounts(errorDetails: ErrorDetails): void {
    const key = `${errorDetails.category}-${errorDetails.message}`;
    this.errorCounts.set(key, (this.errorCounts.get(key) || 0) + 1);
    this.lastErrors.set(key, Date.now());
  }

  private async attemptRecovery(errorDetails: ErrorDetails): Promise<void> {
    const applicableActions = this.recoveryActions.filter(action => 
      action.conditions(errorDetails)
    );
    
    for (const action of applicableActions) {
      try {
        console.log(`Attempting recovery action: ${action.name}`);
        await action.execute();
        console.log(`Recovery action successful: ${action.name}`);
        return; // Stop after first successful recovery
      } catch (recoveryError) {
        console.error(`Recovery action failed: ${action.name}`, recoveryError);
      }
    }
  }

  private reportError(errorDetails: ErrorDetails): void {
    // In a real implementation, this would send to monitoring services
    // like Sentry, DataDog, or custom error tracking
    if (errorDetails.severity === ErrorSeverity.CRITICAL) {
      // Send immediate alert
      this.sendCriticalAlert(errorDetails);
    }
  }

  private sendCriticalAlert(errorDetails: ErrorDetails): void {
    // Implementation would integrate with alerting systems
    console.error('CRITICAL ALERT:', {
      id: errorDetails.id,
      message: errorDetails.message,
      category: errorDetails.category,
      context: errorDetails.context
    });
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private validateObject<T>(
    data: any,
    schema: ValidationSchema<T>,
    path: string,
    errors: string[]
  ): T {
    if (schema.required && (data === null || data === undefined)) {
      errors.push(`${path} is required`);
      throw new Error('Validation failed');
    }

    if (data === null || data === undefined) {
      return data;
    }

    if (schema.type) {
      const actualType = Array.isArray(data) ? 'array' : typeof data;
      if (actualType !== schema.type) {
        errors.push(`${path} should be ${schema.type}, got ${actualType}`);
      }
    }

    if (schema.validator && !schema.validator(data)) {
      errors.push(`${path} failed custom validation`);
    }

    if (schema.properties && typeof data === 'object') {
      Object.keys(schema.properties).forEach(key => {
        const propertyPath = path ? `${path}.${key}` : key;
        try {
          this.validateObject(data[key], schema.properties![key], propertyPath, errors);
        } catch {
          // Continue validation to collect all errors
        }
      });
    }

    return data;
  }
}

// Validation Schema Interface
export interface ValidationSchema<T> {
  type?: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required?: boolean;
  validator?: (value: any) => boolean;
  properties?: { [K in keyof T]?: ValidationSchema<T[K]> };
}

// Specialized Error Classes
export class NetworkError extends EnhancedError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, context, {
      retryable: true,
      recoverable: true
    });
  }
}

export class WebSocketError extends EnhancedError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, ErrorCategory.WEBSOCKET, ErrorSeverity.MEDIUM, context, {
      retryable: true,
      recoverable: true
    });
  }
}

export class ValidationError extends EnhancedError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, context, {
      retryable: false,
      recoverable: false
    });
  }
}

export class RiskError extends EnhancedError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, ErrorCategory.RISK, ErrorSeverity.HIGH, context, {
      retryable: false,
      recoverable: true
    });
  }
}

export class CalculationError extends EnhancedError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, ErrorCategory.CALCULATION, ErrorSeverity.MEDIUM, context, {
      retryable: true,
      recoverable: false
    });
  }
}

// Global Error Handler Instance
export const globalErrorHandler = new ErrorHandler();

// Utility Functions
export const withErrorHandling = <T extends any[], R>(
  fn: (...args: T) => Promise<R>,
  context?: Partial<ErrorContext>
) => {
  return async (...args: T): Promise<R> => {
    try {
      return await fn(...args);
    } catch (error) {
      await globalErrorHandler.handleError(error instanceof Error ? error : new Error(String(error)), context);
      throw error;
    }
  };
};

export const createErrorBoundary = (
  fallback: (error: ErrorDetails) => void,
  context?: Partial<ErrorContext>
) => {
  return (error: Error): void => {
    const enhancedError = error instanceof EnhancedError 
      ? error 
      : new EnhancedError(error.message, ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, context);
    
    fallback(enhancedError.toDetails());
  };
};

export const isRetryableError = (error: Error): boolean => {
  if (error instanceof EnhancedError) {
    return error.retryable;
  }
  
  const retryablePatterns = [
    /network/i,
    /timeout/i,
    /connection/i,
    /rate limit/i,
    /throttle/i,
    /502/,
    /503/,
    /504/
  ];
  
  return retryablePatterns.some(pattern => pattern.test(error.message));
};

export const getErrorDisplayMessage = (error: Error): string => {
  if (error instanceof EnhancedError) {
    switch (error.category) {
      case ErrorCategory.NETWORK:
        return 'Network connection issue. Please check your internet connection and try again.';
      case ErrorCategory.WEBSOCKET:
        return 'Real-time connection issue. Attempting to reconnect...';
      case ErrorCategory.AUTHENTICATION:
        return 'Authentication failed. Please log in again.';
      case ErrorCategory.RATE_LIMIT:
        return 'Too many requests. Please wait a moment and try again.';
      case ErrorCategory.VALIDATION:
        return 'Invalid data provided. Please check your input and try again.';
      default:
        return error.message;
    }
  }
  
  return error.message || 'An unexpected error occurred.';
};