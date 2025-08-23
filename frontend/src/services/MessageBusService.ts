/**
 * Direct Message Bus Service - Bypasses HTTP proxy for maximum performance
 * Connects directly to Redis streams and WebSocket bridge for real-time data
 */

interface MessageBusConfig {
  websocketUrl: string;
  maxReconnectAttempts: number;
  reconnectDelay: number;
  heartbeatInterval: number;
}

interface MessageBusMessage {
  topic: string;
  payload: any;
  timestamp: number;
  messageType: string;
}

interface StreamSubscription {
  id: string;
  topic: string;
  callback: (message: MessageBusMessage) => void;
  active: boolean;
}

export class MessageBusService {
  private ws: WebSocket | null = null;
  private subscriptions: Map<string, StreamSubscription> = new Map();
  private isConnected = false;
  private reconnectAttempts = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private config: MessageBusConfig;

  constructor(config: Partial<MessageBusConfig> = {}) {
    this.config = {
      websocketUrl: `ws://${import.meta.env.VITE_WS_URL || 'localhost:8001'}/ws/messagebus`,
      maxReconnectAttempts: 10,
      reconnectDelay: 1000,
      heartbeatInterval: 30000,
      ...config
    };
  }

  /**
   * Connect to the high-performance message bus
   * Bypasses HTTP proxy entirely for maximum speed
   */
  async connect(): Promise<boolean> {
    try {
      console.log('🚀 Connecting to high-performance message bus:', this.config.websocketUrl);
      
      this.ws = new WebSocket(this.config.websocketUrl);
      
      this.ws.onopen = () => {
        console.log('✅ Direct message bus connection established');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.resubscribeAll();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: MessageBusMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('❌ Error parsing message bus message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('🔌 Message bus connection closed');
        this.isConnected = false;
        this.stopHeartbeat();
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('❌ Message bus WebSocket error:', error);
      };

      return true;
    } catch (error) {
      console.error('❌ Failed to connect to message bus:', error);
      return false;
    }
  }

  /**
   * Subscribe to high-speed data streams
   * Each subscription gets real-time updates without HTTP overhead
   */
  subscribe(topic: string, callback: (message: MessageBusMessage) => void): string {
    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const subscription: StreamSubscription = {
      id: subscriptionId,
      topic,
      callback,
      active: true
    };

    this.subscriptions.set(subscriptionId, subscription);

    // Send subscription message to backend
    if (this.isConnected && this.ws) {
      this.ws.send(JSON.stringify({
        action: 'subscribe',
        topic,
        subscriptionId
      }));
      console.log(`📡 Subscribed to high-speed stream: ${topic}`);
    }

    return subscriptionId;
  }

  /**
   * Unsubscribe from data stream
   */
  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (subscription) {
      subscription.active = false;
      this.subscriptions.delete(subscriptionId);

      if (this.isConnected && this.ws) {
        this.ws.send(JSON.stringify({
          action: 'unsubscribe',
          subscriptionId
        }));
        console.log(`🚫 Unsubscribed from stream: ${subscription.topic}`);
      }
    }
  }

  /**
   * Send command directly to message bus (much faster than HTTP)
   */
  sendCommand(command: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.isConnected || !this.ws) {
        reject(new Error('Not connected to message bus'));
        return;
      }

      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Set up one-time response handler
      const responseHandler = (message: MessageBusMessage) => {
        if (message.payload.requestId === requestId) {
          this.unsubscribe(responseSubscription);
          resolve(message.payload.data);
        }
      };

      const responseSubscription = this.subscribe(`responses.${requestId}`, responseHandler);

      // Send command
      this.ws.send(JSON.stringify({
        action: 'command',
        command,
        data,
        requestId
      }));

      // Timeout after 5 seconds
      setTimeout(() => {
        this.unsubscribe(responseSubscription);
        reject(new Error('Command timeout'));
      }, 5000);
    });
  }

  /**
   * Get real-time portfolio data (bypasses HTTP entirely)
   */
  async getRealtimePortfolio(portfolioId: string): Promise<any> {
    return this.sendCommand('get_realtime_portfolio', { portfolioId });
  }

  /**
   * Get live market data stream (high-frequency updates)
   */
  subscribeToMarketData(symbol: string, callback: (data: any) => void): string {
    return this.subscribe(`market_data.${symbol}`, (message) => {
      if (message.topic === 'quotes' || message.topic === 'trades' || message.topic === 'bars') {
        callback(message.payload);
      }
    });
  }

  /**
   * Get real-time order updates
   */
  subscribeToOrderUpdates(callback: (order: any) => void): string {
    return this.subscribe('order_updates', (message) => {
      callback(message.payload);
    });
  }

  /**
   * Get real-time position updates
   */
  subscribeToPositionUpdates(callback: (position: any) => void): string {
    return this.subscribe('position_updates', (message) => {
      callback(message.payload);
    });
  }

  private handleMessage(message: MessageBusMessage): void {
    // Route message to all matching subscriptions
    this.subscriptions.forEach((subscription) => {
      if (subscription.active && this.topicMatches(subscription.topic, message.topic)) {
        try {
          subscription.callback(message);
        } catch (error) {
          console.error(`❌ Error in subscription callback for ${subscription.topic}:`, error);
        }
      }
    });
  }

  private topicMatches(subscriptionTopic: string, messageTopic: string): boolean {
    // Support wildcard patterns like "market_data.*"
    if (subscriptionTopic.endsWith('*')) {
      const prefix = subscriptionTopic.slice(0, -1);
      return messageTopic.startsWith(prefix);
    }
    return subscriptionTopic === messageTopic;
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach((subscription) => {
      if (subscription.active && this.ws) {
        this.ws.send(JSON.stringify({
          action: 'subscribe',
          topic: subscription.topic,
          subscriptionId: subscription.id
        }));
      }
    });
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.config.maxReconnectAttempts}) in ${delay}ms`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.ws.send(JSON.stringify({ action: 'ping' }));
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  disconnect(): void {
    this.isConnected = false;
    this.stopHeartbeat();
    
    // Clear all subscriptions
    this.subscriptions.clear();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    console.log('🔌 Disconnected from message bus');
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected,
      subscriptions: this.subscriptions.size,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// Global singleton instance for app-wide usage
export const messageBusService = new MessageBusService();