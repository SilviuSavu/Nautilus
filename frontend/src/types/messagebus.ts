/**
 * TypeScript types for NautilusTrader MessageBus integration
 */

export interface MessageBusMessage {
  type: 'messagebus';
  topic: string;
  payload: any;
  timestamp: number;
  message_type: string;
}

export interface ConnectionMessage {
  type: 'connection';
  status: 'connected' | 'disconnected';
  message: string;
}

export interface MessageBusConnectionStatus {
  connection_state: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';
  connected_at?: string;
  last_message_at?: string;
  reconnect_attempts: number;
  error_message?: string;
  messages_received: number;
}

export interface MessageBusState {
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  messages: MessageBusMessage[];
  latestMessage: MessageBusMessage | null;
  connectionInfo: MessageBusConnectionStatus | null;
  messagesReceived: number;
}

export interface MarketDataMessage {
  symbol: string;
  bid?: number;
  ask?: number;
  last?: number;
  volume?: number;
  timestamp: number;
}

export interface TradingEventMessage {
  event_type: string;
  symbol?: string;
  order_id?: string;
  position_id?: string;
  data: any;
  timestamp: number;
}