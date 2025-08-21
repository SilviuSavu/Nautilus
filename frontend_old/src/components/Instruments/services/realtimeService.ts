import { Instrument, VenueInfo, VenueConnectionStatus } from '../types/instrumentTypes'

export interface RealtimePrice {
  instrumentId: string
  symbol: string
  venue: string
  bid: number
  ask: number
  last: number
  change: number
  changePercent: number
  volume: number
  timestamp: string
}

export interface MarketSessionInfo {
  venue: string
  isOpen: boolean
  nextOpen?: string
  nextClose?: string
  timezone: string
  marketHours: {
    open: string
    close: string
    days: string[]
  }
}

export interface VenueStatusUpdate {
  venue: string
  status: VenueConnectionStatus
  lastHeartbeat: string
  errorMessage?: string
  connectedInstruments: number
}

export interface RealtimeUpdate {
  type: 'price' | 'venue_status' | 'market_session' | 'heartbeat'
  data: RealtimePrice | VenueStatusUpdate | MarketSessionInfo | { timestamp: string }
  timestamp: string
}

export type RealtimeEventHandler = (update: RealtimeUpdate) => void

class RealtimeService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private subscribers: Map<string, Set<RealtimeEventHandler>> = new Map()
  private instrumentSubscriptions: Set<string> = new Set()
  private venueSubscriptions: Set<string> = new Set()
  private isConnected = false
  private connectionState: 'disconnected' | 'connecting' | 'connected' | 'error' = 'disconnected'

  constructor() {
    this.connect()
  }

  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = import.meta.env.VITE_WS_URL || `${window.location.hostname}:8001`
    return `${protocol}//${host}/ws/realtime`
  }

  private connect(): void {
    if (this.connectionState === 'connecting') return

    this.connectionState = 'connecting'
    
    try {
      this.ws = new WebSocket(this.getWebSocketUrl())
      
      this.ws.onopen = () => {
        console.log('WebSocket connected for real-time data')
        this.isConnected = true
        this.connectionState = 'connected'
        this.reconnectAttempts = 0
        
        // Resubscribe to instruments and venues
        this.resubscribeAll()
        
        // Notify connection status
        this.notifySubscribers('heartbeat', {
          type: 'heartbeat',
          data: { timestamp: new Date().toISOString() },
          timestamp: new Date().toISOString()
        })
      }

      this.ws.onmessage = (event) => {
        try {
          const update: RealtimeUpdate = JSON.parse(event.data)
          this.handleRealtimeUpdate(update)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        this.isConnected = false
        this.connectionState = 'disconnected'
        this.ws = null
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect()
        } else {
          this.connectionState = 'error'
          console.error('Max reconnection attempts reached')
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.connectionState = 'error'
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      this.connectionState = 'error'
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) // Exponential backoff
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
    
    setTimeout(() => {
      this.connect()
    }, delay)
  }

  private resubscribeAll(): void {
    // Resubscribe to instruments
    if (this.instrumentSubscriptions.size > 0) {
      this.sendMessage({
        type: 'subscribe_instruments',
        instruments: Array.from(this.instrumentSubscriptions)
      })
    }

    // Resubscribe to venues
    if (this.venueSubscriptions.size > 0) {
      this.sendMessage({
        type: 'subscribe_venues',
        venues: Array.from(this.venueSubscriptions)
      })
    }
  }

  private sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket not connected, cannot send message:', message)
    }
  }

  private handleRealtimeUpdate(update: RealtimeUpdate): void {
    // Notify all subscribers of the update type
    this.notifySubscribers(update.type, update)
    
    // Also notify 'all' subscribers
    this.notifySubscribers('all', update)
  }

  private notifySubscribers(eventType: string, update: RealtimeUpdate): void {
    const handlers = this.subscribers.get(eventType)
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(update)
        } catch (error) {
          console.error('Error in realtime event handler:', error)
        }
      })
    }
  }

  // Public API methods
  subscribe(eventType: string, handler: RealtimeEventHandler): () => void {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, new Set())
    }
    
    this.subscribers.get(eventType)!.add(handler)
    
    // Return unsubscribe function
    return () => {
      const handlers = this.subscribers.get(eventType)
      if (handlers) {
        handlers.delete(handler)
        if (handlers.size === 0) {
          this.subscribers.delete(eventType)
        }
      }
    }
  }

  subscribeToInstrument(instrumentId: string): void {
    this.instrumentSubscriptions.add(instrumentId)
    
    if (this.isConnected) {
      this.sendMessage({
        type: 'subscribe_instrument',
        instrumentId
      })
    }
  }

  unsubscribeFromInstrument(instrumentId: string): void {
    this.instrumentSubscriptions.delete(instrumentId)
    
    if (this.isConnected) {
      this.sendMessage({
        type: 'unsubscribe_instrument',
        instrumentId
      })
    }
  }

  subscribeToVenue(venue: string): void {
    this.venueSubscriptions.add(venue)
    
    if (this.isConnected) {
      this.sendMessage({
        type: 'subscribe_venue',
        venue
      })
    }
  }

  unsubscribeFromVenue(venue: string): void {
    this.venueSubscriptions.delete(venue)
    
    if (this.isConnected) {
      this.sendMessage({
        type: 'unsubscribe_venue',
        venue
      })
    }
  }

  requestMarketSession(venue: string): void {
    if (this.isConnected) {
      this.sendMessage({
        type: 'get_market_session',
        venue
      })
    }
  }

  getConnectionStatus(): {
    isConnected: boolean
    state: string
    reconnectAttempts: number
  } {
    return {
      isConnected: this.isConnected,
      state: this.connectionState,
      reconnectAttempts: this.reconnectAttempts
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isConnected = false
    this.connectionState = 'disconnected'
    this.subscribers.clear()
    this.instrumentSubscriptions.clear()
    this.venueSubscriptions.clear()
  }

  // Mock data generators for development/testing
  private mockPriceData(): RealtimePrice {
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    const venues = ['NASDAQ', 'NYSE', 'SMART']
    const symbol = symbols[Math.floor(Math.random() * symbols.length)]
    const venue = venues[Math.floor(Math.random() * venues.length)]
    
    const basePrice = 150 + Math.random() * 300
    const change = (Math.random() - 0.5) * 10
    
    return {
      instrumentId: `${symbol}-STK`,
      symbol,
      venue,
      bid: basePrice - 0.01,
      ask: basePrice + 0.01,
      last: basePrice,
      change,
      changePercent: (change / basePrice) * 100,
      volume: Math.floor(Math.random() * 1000000),
      timestamp: new Date().toISOString()
    }
  }

  // Development method to simulate real-time updates
  startMockUpdates(): void {
    if (import.meta.env.DEV) {
      console.log('Starting mock real-time updates for development')
      
      setInterval(() => {
        if (this.instrumentSubscriptions.size > 0) {
          const mockPrice = this.mockPriceData()
          this.handleRealtimeUpdate({
            type: 'price',
            data: mockPrice,
            timestamp: new Date().toISOString()
          })
        }
      }, 2000)

      setInterval(() => {
        if (this.venueSubscriptions.size > 0) {
          const venues = Array.from(this.venueSubscriptions)
          const randomVenue = venues[Math.floor(Math.random() * venues.length)]
          const statuses: VenueConnectionStatus[] = ['connected', 'connecting', 'disconnected']
          const randomStatus = statuses[Math.floor(Math.random() * statuses.length)]
          
          const mockVenueUpdate: VenueStatusUpdate = {
            venue: randomVenue,
            status: randomStatus,
            lastHeartbeat: new Date().toISOString(),
            connectedInstruments: Math.floor(Math.random() * 100)
          }
          
          this.handleRealtimeUpdate({
            type: 'venue_status',
            data: mockVenueUpdate,
            timestamp: new Date().toISOString()
          })
        }
      }, 5000)
    }
  }
}

export const realtimeService = new RealtimeService()

// Start mock updates in development
if (import.meta.env.DEV) {
  realtimeService.startMockUpdates()
}