/**
 * Advanced WebSocket Connection Management System
 * Supports real-time streaming across multiple engines
 * Based on FRONTEND_ENDPOINT_INTEGRATION_GUIDE.md
 */

import { API_CONFIG } from './apiClient';

// WebSocket message types
interface WebSocketMessage {
  type: string;
  timestamp: string;
  data: any;
}

interface VolatilityUpdate extends WebSocketMessage {
  type: 'volatility_update';
  symbol: string;
  data: {
    current_volatility: number;
    forecast_update: any;
    trigger_reason: string;
    confidence: number;
  };
}

interface MarketDataUpdate extends WebSocketMessage {
  type: 'price_update';
  symbol: string;
  data: {
    bid: number;
    ask: number;
    last: number;
    volume: number;
  };
}

interface HealthUpdate extends WebSocketMessage {
  type: 'health_update';
  components: {
    engines: any[];
    databases: any[];
    external_services: any[];
  };
}

interface TradeExecution extends WebSocketMessage {
  type: 'trade_execution';
  trade_id: string;
  data: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    status: 'filled' | 'partial' | 'rejected';
  };
}

interface MessageBusEvent extends WebSocketMessage {
  type: 'messagebus_event';
  source: string;
  event_type: string;
}

// Connection status enum
enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error'
}

// WebSocket connection options
interface WebSocketOptions {
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  timeout?: number;
  onMessage?: (data: WebSocketMessage) => void;
  onStatusChange?: (status: ConnectionStatus) => void;
  onError?: (error: Error) => void;
}

// WebSocket connection manager
class WebSocketManager {
  private connections = new Map<string, WebSocket>();
  private connectionStatus = new Map<string, ConnectionStatus>();
  private reconnectAttempts = new Map<string, number>();
  private heartbeatIntervals = new Map<string, NodeJS.Timeout>();
  private reconnectTimeouts = new Map<string, NodeJS.Timeout>();
  
  private defaultOptions: Required<WebSocketOptions> = {
    reconnectAttempts: 10,
    reconnectInterval: 5000,
    heartbeatInterval: 30000,
    timeout: 10000,
    onMessage: () => {},
    onStatusChange: () => {},
    onError: () => {}
  };

  // Connect to WebSocket endpoint
  connect(endpoint: string, options: WebSocketOptions = {}): Promise<WebSocket> {
    const fullOptions = { ...this.defaultOptions, ...options };
    const wsUrl = `${API_CONFIG.WS_URL}${endpoint}`;
    
    return new Promise((resolve, reject) => {
      try {
        this.setConnectionStatus(endpoint, ConnectionStatus.CONNECTING);
        
        const ws = new WebSocket(wsUrl);
        this.connections.set(endpoint, ws);

        // Connection timeout
        const timeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, fullOptions.timeout);

        ws.onopen = () => {
          clearTimeout(timeout);
          this.setConnectionStatus(endpoint, ConnectionStatus.CONNECTED);
          this.resetReconnectAttempts(endpoint);
          this.startHeartbeat(endpoint, fullOptions);
          fullOptions.onStatusChange(ConnectionStatus.CONNECTED);
          resolve(ws);
        };

        ws.onmessage = (event) => {
          try {
            const data: WebSocketMessage = JSON.parse(event.data);
            fullOptions.onMessage(data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
            fullOptions.onError(error as Error);
          }
        };

        ws.onclose = (event) => {
          clearTimeout(timeout);
          this.stopHeartbeat(endpoint);
          this.setConnectionStatus(endpoint, ConnectionStatus.DISCONNECTED);
          
          if (!event.wasClean && this.shouldReconnect(endpoint, fullOptions)) {
            this.scheduleReconnect(endpoint, fullOptions);
          } else {
            fullOptions.onStatusChange(ConnectionStatus.DISCONNECTED);
          }
        };

        ws.onerror = (error) => {
          clearTimeout(timeout);
          console.error(`WebSocket error on ${endpoint}:`, error);
          this.setConnectionStatus(endpoint, ConnectionStatus.ERROR);
          fullOptions.onError(new Error(`WebSocket connection error`));
          fullOptions.onStatusChange(ConnectionStatus.ERROR);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  // Disconnect from WebSocket endpoint
  disconnect(endpoint: string): void {
    const ws = this.connections.get(endpoint);
    if (ws) {
      this.stopHeartbeat(endpoint);
      this.clearReconnectTimeout(endpoint);
      ws.close();
      this.connections.delete(endpoint);
      this.setConnectionStatus(endpoint, ConnectionStatus.DISCONNECTED);
    }
  }

  // Disconnect from all WebSocket endpoints
  disconnectAll(): void {
    Array.from(this.connections.keys()).forEach(endpoint => {
      this.disconnect(endpoint);
    });
  }

  // Send message to WebSocket endpoint
  send(endpoint: string, data: any): boolean {
    const ws = this.connections.get(endpoint);
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(data));
        return true;
      } catch (error) {
        console.error(`Failed to send message to ${endpoint}:`, error);
        return false;
      }
    }
    return false;
  }

  // Get connection status
  getStatus(endpoint: string): ConnectionStatus {
    return this.connectionStatus.get(endpoint) || ConnectionStatus.DISCONNECTED;
  }

  // Get all connection statuses
  getAllStatuses(): Record<string, ConnectionStatus> {
    const statuses: Record<string, ConnectionStatus> = {};
    this.connectionStatus.forEach((status, endpoint) => {
      statuses[endpoint] = status;
    });
    return statuses;
  }

  // Private methods
  private setConnectionStatus(endpoint: string, status: ConnectionStatus): void {
    this.connectionStatus.set(endpoint, status);
  }

  private shouldReconnect(endpoint: string, options: Required<WebSocketOptions>): boolean {
    const attempts = this.reconnectAttempts.get(endpoint) || 0;
    return attempts < options.reconnectAttempts;
  }

  private scheduleReconnect(endpoint: string, options: Required<WebSocketOptions>): void {
    const attempts = this.reconnectAttempts.get(endpoint) || 0;
    this.reconnectAttempts.set(endpoint, attempts + 1);
    this.setConnectionStatus(endpoint, ConnectionStatus.RECONNECTING);
    options.onStatusChange(ConnectionStatus.RECONNECTING);

    const timeout = setTimeout(() => {
      console.log(`Attempting to reconnect to ${endpoint} (attempt ${attempts + 1}/${options.reconnectAttempts})`);
      this.connect(endpoint, options).catch(() => {
        // Reconnection failed, will be handled by onclose
      });
    }, options.reconnectInterval * Math.pow(2, Math.min(attempts, 5))); // Exponential backoff

    this.reconnectTimeouts.set(endpoint, timeout);
  }

  private resetReconnectAttempts(endpoint: string): void {
    this.reconnectAttempts.set(endpoint, 0);
  }

  private clearReconnectTimeout(endpoint: string): void {
    const timeout = this.reconnectTimeouts.get(endpoint);
    if (timeout) {
      clearTimeout(timeout);
      this.reconnectTimeouts.delete(endpoint);
    }
  }

  private startHeartbeat(endpoint: string, options: Required<WebSocketOptions>): void {
    const interval = setInterval(() => {
      if (!this.send(endpoint, { type: 'ping', timestamp: new Date().toISOString() })) {
        this.stopHeartbeat(endpoint);
      }
    }, options.heartbeatInterval);

    this.heartbeatIntervals.set(endpoint, interval);
  }

  private stopHeartbeat(endpoint: string): void {
    const interval = this.heartbeatIntervals.get(endpoint);
    if (interval) {
      clearInterval(interval);
      this.heartbeatIntervals.delete(endpoint);
    }
  }
}

// Singleton instance
const wsManager = new WebSocketManager();

// Specific WebSocket clients
export class VolatilityWebSocketClient {
  private manager = wsManager;
  
  // Connect to volatility updates for a symbol
  async connectToVolatilityUpdates(
    symbol: string, 
    onUpdate: (update: VolatilityUpdate) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): Promise<WebSocket> {
    const endpoint = `/api/v1/volatility/ws/streaming/${symbol}`;
    
    return this.manager.connect(endpoint, {
      onMessage: (data) => {
        if (data.type === 'volatility_update') {
          onUpdate(data as VolatilityUpdate);
        }
      },
      onStatusChange: onStatusChange || (() => {}),
      onError: (error) => console.error(`Volatility WebSocket error for ${symbol}:`, error)
    });
  }

  // Disconnect from volatility updates
  disconnect(symbol: string): void {
    const endpoint = `/api/v1/volatility/ws/streaming/${symbol}`;
    this.manager.disconnect(endpoint);
  }

  // Get connection status
  getStatus(symbol: string): ConnectionStatus {
    const endpoint = `/api/v1/volatility/ws/streaming/${symbol}`;
    return this.manager.getStatus(endpoint);
  }
}

export class MarketDataWebSocketClient {
  private manager = wsManager;

  // Connect to market data updates for a symbol
  async connectToMarketData(
    symbol: string,
    onUpdate: (update: MarketDataUpdate) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): Promise<WebSocket> {
    const endpoint = `/api/v1/ws/market-data/${symbol}`;
    
    return this.manager.connect(endpoint, {
      onMessage: (data) => {
        if (data.type === 'price_update') {
          onUpdate(data as MarketDataUpdate);
        }
      },
      onStatusChange: onStatusChange || (() => {}),
      onError: (error) => console.error(`Market data WebSocket error for ${symbol}:`, error)
    });
  }

  disconnect(symbol: string): void {
    const endpoint = `/api/v1/ws/market-data/${symbol}`;
    this.manager.disconnect(endpoint);
  }

  getStatus(symbol: string): ConnectionStatus {
    const endpoint = `/api/v1/ws/market-data/${symbol}`;
    return this.manager.getStatus(endpoint);
  }
}

export class SystemHealthWebSocketClient {
  private manager = wsManager;

  // Connect to system health updates
  async connectToSystemHealth(
    onUpdate: (update: HealthUpdate) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): Promise<WebSocket> {
    const endpoint = '/ws/system/health';
    
    return this.manager.connect(endpoint, {
      onMessage: (data) => {
        if (data.type === 'health_update') {
          onUpdate(data as HealthUpdate);
        }
      },
      onStatusChange: onStatusChange || (() => {}),
      onError: (error) => console.error('System health WebSocket error:', error)
    });
  }

  disconnect(): void {
    const endpoint = '/ws/system/health';
    this.manager.disconnect(endpoint);
  }

  getStatus(): ConnectionStatus {
    const endpoint = '/ws/system/health';
    return this.manager.getStatus(endpoint);
  }
}

export class TradeUpdatesWebSocketClient {
  private manager = wsManager;

  // Connect to trade updates
  async connectToTradeUpdates(
    onUpdate: (update: TradeExecution) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): Promise<WebSocket> {
    const endpoint = '/ws/trades/updates';
    
    return this.manager.connect(endpoint, {
      onMessage: (data) => {
        if (data.type === 'trade_execution') {
          onUpdate(data as TradeExecution);
        }
      },
      onStatusChange: onStatusChange || (() => {}),
      onError: (error) => console.error('Trade updates WebSocket error:', error)
    });
  }

  disconnect(): void {
    const endpoint = '/ws/trades/updates';
    this.manager.disconnect(endpoint);
  }

  getStatus(): ConnectionStatus {
    const endpoint = '/ws/trades/updates';
    return this.manager.getStatus(endpoint);
  }
}

export class MessageBusWebSocketClient {
  private manager = wsManager;

  // Connect to MessageBus events
  async connectToMessageBus(
    onEvent: (event: MessageBusEvent) => void,
    onStatusChange?: (status: ConnectionStatus) => void
  ): Promise<WebSocket> {
    const endpoint = '/ws/messagebus';
    
    return this.manager.connect(endpoint, {
      onMessage: (data) => {
        if (data.type === 'messagebus_event') {
          onEvent(data as MessageBusEvent);
        }
      },
      onStatusChange: onStatusChange || (() => {}),
      onError: (error) => console.error('MessageBus WebSocket error:', error)
    });
  }

  disconnect(): void {
    const endpoint = '/ws/messagebus';
    this.manager.disconnect(endpoint);
  }

  getStatus(): ConnectionStatus {
    const endpoint = '/ws/messagebus';
    return this.manager.getStatus(endpoint);
  }
}

// Export instances
export const volatilityWS = new VolatilityWebSocketClient();
export const marketDataWS = new MarketDataWebSocketClient();
export const systemHealthWS = new SystemHealthWebSocketClient();
export const tradeUpdatesWS = new TradeUpdatesWebSocketClient();
export const messageBusWS = new MessageBusWebSocketClient();

// Export WebSocket manager for advanced usage
export { wsManager as webSocketManager };

// Export types
export type {
  WebSocketMessage,
  VolatilityUpdate,
  MarketDataUpdate,
  HealthUpdate,
  TradeExecution,
  MessageBusEvent,
  WebSocketOptions
};

export { ConnectionStatus };