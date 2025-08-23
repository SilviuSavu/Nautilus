/**
 * Nautilus Trading Platform TypeScript SDK
 * Official TypeScript/JavaScript client library for Node.js and browser environments
 */

export { NautilusClient } from './client';
export { WebSocketClient } from './websocket';
export { AuthManager } from './auth';
export * from './types';
export * from './exceptions';

// Re-export for convenience
export { default as Nautilus } from './client';

// SDK Information
export const SDK_VERSION = '3.0.0';
export const SDK_NAME = 'nautilus-typescript-sdk';

// Default configuration
export const DEFAULT_CONFIG = {
  baseUrl: 'http://localhost:8001',
  timeout: 30000,
  maxRetries: 3,
  retryDelay: 1000,
  wsUrl: 'ws://localhost:8001'
};

/**
 * Quick utility functions for common operations
 */
export namespace QuickOps {
  /**
   * Get a quick quote without managing client lifecycle
   */
  export async function getQuote(
    symbol: string,
    config?: { baseUrl?: string; apiKey?: string }
  ) {
    const client = new NautilusClient(config);
    try {
      return await client.marketData.getQuote(symbol);
    } finally {
      await client.close();
    }
  }

  /**
   * Quick health check
   */
  export async function healthCheck(baseUrl?: string) {
    const client = new NautilusClient({ baseUrl });
    try {
      return await client.system.getHealth();
    } finally {
      await client.close();
    }
  }
}