/**
 * Persistent API Client with Connection Management
 * Ensures reliable, long-lived connections to the backend with automatic retry logic
 */

interface ConnectionConfig {
  baseUrl: string;
  maxRetries: number;
  retryDelay: number;
  timeout: number;
  keepAlive: boolean;
  heartbeatInterval: number;
}

interface ConnectionState {
  isConnected: boolean;
  lastSuccessfulRequest: Date | null;
  consecutiveFailures: number;
  lastError: string | null;
}

type ConnectionListener = (state: ConnectionState) => void;

class PersistentApiClient {
  private config: ConnectionConfig;
  private state: ConnectionState;
  private listeners: Set<ConnectionListener> = new Set();
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private retryTimer: NodeJS.Timeout | null = null;

  constructor(config?: Partial<ConnectionConfig>) {
    this.config = {
      baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001',
      maxRetries: 5,
      retryDelay: 2000,
      timeout: 30000,
      keepAlive: true,
      heartbeatInterval: 30000,
      ...config
    };

    this.state = {
      isConnected: false,
      lastSuccessfulRequest: null,
      consecutiveFailures: 0,
      lastError: null
    };

    this.startHeartbeat();
  }

  /**
   * Add a listener for connection state changes
   */
  public onConnectionChange(listener: ConnectionListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Get current connection state
   */
  public getConnectionState(): ConnectionState {
    return { ...this.state };
  }

  /**
   * Emit connection state change to all listeners
   */
  private emitConnectionChange() {
    this.listeners.forEach(listener => {
      try {
        listener(this.getConnectionState());
      } catch (error) {
        console.error('Error in connection state listener:', error);
      }
    });
  }

  /**
   * Update connection state
   */
  private updateConnectionState(updates: Partial<ConnectionState>) {
    const previousConnected = this.state.isConnected;
    this.state = { ...this.state, ...updates };
    
    // Only emit if connection status actually changed
    if (previousConnected !== this.state.isConnected) {
      console.log(`🔗 API Client ${this.state.isConnected ? 'connected' : 'disconnected'}`);
      this.emitConnectionChange();
    }
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(async () => {
      try {
        await this.healthCheck();
      } catch (error) {
        // Health check failures are handled by the request method
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat monitoring
   */
  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Perform a health check
   */
  private async healthCheck(): Promise<void> {
    return this.request('/health', {}, false); // Don't retry health checks
  }

  /**
   * Calculate retry delay with exponential backoff
   */
  private getRetryDelay(attempt: number): number {
    return Math.min(this.config.retryDelay * Math.pow(2, attempt), 30000);
  }

  /**
   * Make a request with automatic retry logic
   */
  public async request<T = any>(
    endpoint: string, 
    options: RequestInit = {}, 
    enableRetry: boolean = true
  ): Promise<T> {
    const url = endpoint.startsWith('http') ? endpoint : `${this.config.baseUrl}${endpoint}`;
    
    const requestOptions: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Connection': this.config.keepAlive ? 'keep-alive' : 'close',
        ...options.headers
      },
      signal: AbortSignal.timeout(this.config.timeout)
    };

    let lastError: Error | null = null;
    const maxAttempts = enableRetry ? this.config.maxRetries : 1;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await fetch(url, requestOptions);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Success - update connection state
        this.updateConnectionState({
          isConnected: true,
          lastSuccessfulRequest: new Date(),
          consecutiveFailures: 0,
          lastError: null
        });

        return data;

      } catch (error) {
        lastError = error as Error;
        
        console.warn(`Request attempt ${attempt + 1} failed:`, error);

        // Update failure count
        this.updateConnectionState({
          consecutiveFailures: this.state.consecutiveFailures + 1,
          lastError: lastError.message
        });

        // If we've exceeded maximum failures, mark as disconnected
        if (this.state.consecutiveFailures >= 3) {
          this.updateConnectionState({ isConnected: false });
        }

        // Don't retry on certain errors
        if (error instanceof TypeError || error.message.includes('400')) {
          break;
        }

        // Wait before retry (except on last attempt)
        if (attempt < maxAttempts - 1) {
          const delay = this.getRetryDelay(attempt);
          console.log(`⏳ Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    // All attempts failed
    this.updateConnectionState({ isConnected: false });
    throw lastError || new Error('Request failed after all retries');
  }

  /**
   * GET request with retry logic
   */
  public async get<T = any>(endpoint: string, params?: Record<string, string>): Promise<T> {
    let url = endpoint;
    if (params) {
      const searchParams = new URLSearchParams(params);
      url += (url.includes('?') ? '&' : '?') + searchParams.toString();
    }
    return this.request<T>(url, { method: 'GET' });
  }

  /**
   * POST request with retry logic
   */
  public async post<T = any>(endpoint: string, body?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined
    });
  }

  /**
   * PUT request with retry logic
   */
  public async put<T = any>(endpoint: string, body?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: body ? JSON.stringify(body) : undefined
    });
  }

  /**
   * DELETE request with retry logic
   */
  public async delete<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  /**
   * Force reconnection attempt
   */
  public async reconnect(): Promise<void> {
    console.log('🔄 Attempting to reconnect...');
    try {
      await this.healthCheck();
      console.log('✅ Reconnection successful');
    } catch (error) {
      console.error('❌ Reconnection failed:', error);
      throw error;
    }
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.stopHeartbeat();
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
    }
    this.listeners.clear();
  }
}

// Create singleton instance
export const persistentApiClient = new PersistentApiClient();

// Export connection state hook for React components
import { useState, useEffect } from 'react';

export function useConnectionState() {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    persistentApiClient.getConnectionState()
  );

  useEffect(() => {
    const unsubscribe = persistentApiClient.onConnectionChange(setConnectionState);
    return unsubscribe;
  }, []);

  return {
    ...connectionState,
    reconnect: () => persistentApiClient.reconnect()
  };
}

export { PersistentApiClient };
export type { ConnectionState, ConnectionConfig };