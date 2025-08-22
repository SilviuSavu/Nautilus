/**
 * WebSocket service for connecting to NautilusTrader MessageBus via backend
 */

import { OrderBookMessage } from '../types/orderBook'

export interface WebSocketMessage {
  type: string;
  topic?: string;
  payload?: any;
  timestamp?: number;
  message_type?: string;
  status?: string;
  message?: string;
  // Order book specific fields (for order_book_update messages)
  symbol?: string;
  venue?: string;
  bids?: Array<{ price: number; quantity: number; orderCount?: number }>;
  asks?: Array<{ price: number; quantity: number; orderCount?: number }>;
}

export interface PerformanceMetrics {
  messageLatency: number[];
  averageLatency: number;
  maxLatency: number;
  minLatency: number;
  messagesProcessed: number;
  messagesPerSecond: number;
  lastUpdateTime: number;
}

export interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private isConnecting = false;
  private messageHandlers: Set<(message: WebSocketMessage) => void> = new Set();
  private statusHandlers: Set<(status: 'connecting' | 'connected' | 'disconnected' | 'error') => void> = new Set();
  private performanceMetrics: PerformanceMetrics = {
    messageLatency: [],
    averageLatency: 0,
    maxLatency: 0,
    minLatency: Infinity,
    messagesProcessed: 0,
    messagesPerSecond: 0,
    lastUpdateTime: Date.now()
  };
  private messageTimestamps: Map<string, number> = new Map();

  constructor(config: WebSocketConfig) {
    this.config = config;
  }

  connect(): void {
    // Don't try to connect if not in browser environment
    if (typeof window === 'undefined' || typeof WebSocket === 'undefined') {
      console.warn('WebSocket not available in this environment');
      return;
    }

    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;
    this.notifyStatusHandlers('connecting');

    try {
      this.ws = new WebSocket(this.config.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected to NautilusTrader backend');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.notifyStatusHandlers('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const receiveTime = performance.now();
          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Calculate latency if message has timestamp
          if (message.timestamp) {
            const messageTime = message.timestamp / 1000000; // Convert nanoseconds to milliseconds
            const latency = receiveTime - messageTime;
            this.updatePerformanceMetrics(latency);
          }
          
          this.notifyMessageHandlers(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        this.isConnecting = false;
        this.notifyStatusHandlers('disconnected');
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        this.notifyStatusHandlers('error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.isConnecting = false;
      this.notifyStatusHandlers('error');
      this.handleReconnect();
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.reconnectAttempts = this.config.maxReconnectAttempts; // Prevent auto-reconnect
  }

  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  addMessageHandler(handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.add(handler);
  }

  removeMessageHandler(handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.delete(handler);
  }

  addStatusHandler(handler: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void): void {
    this.statusHandlers.add(handler);
  }

  removeStatusHandler(handler: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void): void {
    this.statusHandlers.delete(handler);
  }

  getConnectionState(): 'connecting' | 'connected' | 'disconnected' | 'error' {
    if (this.isConnecting) return 'connecting';
    if (this.ws?.readyState === WebSocket.OPEN) return 'connected';
    if (this.ws?.readyState === WebSocket.CLOSED) return 'disconnected';
    return 'error';
  }

  getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  private updatePerformanceMetrics(latency: number): void {
    const maxSamples = 100; // Keep last 100 latency measurements
    
    this.performanceMetrics.messageLatency.push(latency);
    if (this.performanceMetrics.messageLatency.length > maxSamples) {
      this.performanceMetrics.messageLatency.shift();
    }
    
    this.performanceMetrics.messagesProcessed++;
    this.performanceMetrics.maxLatency = Math.max(this.performanceMetrics.maxLatency, latency);
    this.performanceMetrics.minLatency = Math.min(this.performanceMetrics.minLatency, latency);
    
    // Calculate average latency
    const sum = this.performanceMetrics.messageLatency.reduce((a, b) => a + b, 0);
    this.performanceMetrics.averageLatency = sum / this.performanceMetrics.messageLatency.length;
    
    // Calculate messages per second
    const currentTime = Date.now();
    const timeDiff = (currentTime - this.performanceMetrics.lastUpdateTime) / 1000;
    if (timeDiff >= 1) {
      this.performanceMetrics.messagesPerSecond = this.performanceMetrics.messagesProcessed / timeDiff;
      this.performanceMetrics.lastUpdateTime = currentTime;
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000); // Exponential backoff, max 30s

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private notifyMessageHandlers(message: WebSocketMessage): void {
    this.messageHandlers.forEach(handler => {
      try {
        handler(message);
      } catch (error) {
        console.error('Error in message handler:', error);
      }
    });
  }

  private notifyStatusHandlers(status: 'connecting' | 'connected' | 'disconnected' | 'error'): void {
    this.statusHandlers.forEach(handler => {
      try {
        handler(status);
      } catch (error) {
        console.error('Error in status handler:', error);
      }
    });
  }
}

// Create singleton instance with safe window access
const getWebSocketConfig = (): WebSocketConfig => {
  try {
    const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
    return {
      url: `ws://${hostname}:8001/ws/realtime`,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    };
  } catch (error) {
    console.warn('Failed to get window.location, using localhost:', error);
    return {
      url: `ws://${import.meta.env.VITE_WS_URL || 'localhost:8001'}/ws/realtime`,
      reconnectInterval: 5000,
      maxReconnectAttempts: 10
    };
  }
};

export const webSocketService = new WebSocketService(getWebSocketConfig());