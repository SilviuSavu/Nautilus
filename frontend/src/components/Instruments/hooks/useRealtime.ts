import { useState, useEffect, useCallback, useRef } from 'react'
import { 
  realtimeService, 
  RealtimeUpdate, 
  RealtimePrice, 
  VenueStatusUpdate, 
  MarketSessionInfo 
} from '../services/realtimeService'
import { VenueConnectionStatus } from '../types/instrumentTypes'

export interface RealtimeConnectionStatus {
  isConnected: boolean
  state: 'disconnected' | 'connecting' | 'connected' | 'error'
  reconnectAttempts: number
}

// Hook for general real-time connection status
export const useRealtimeConnection = () => {
  const [status, setStatus] = useState<RealtimeConnectionStatus>(() => {
    const serviceStatus = realtimeService.getConnectionStatus()
    return {
      isConnected: serviceStatus.isConnected,
      state: serviceStatus.state as 'disconnected' | 'connecting' | 'connected' | 'error',
      reconnectAttempts: serviceStatus.reconnectAttempts
    }
  })

  useEffect(() => {
    const unsubscribe = realtimeService.subscribe('heartbeat', () => {
      const serviceStatus = realtimeService.getConnectionStatus()
      setStatus({
        isConnected: serviceStatus.isConnected,
        state: serviceStatus.state as 'disconnected' | 'connecting' | 'connected' | 'error',
        reconnectAttempts: serviceStatus.reconnectAttempts
      })
    })

    // Update status every 5 seconds
    const interval = setInterval(() => {
      const serviceStatus = realtimeService.getConnectionStatus()
      setStatus({
        isConnected: serviceStatus.isConnected,
        state: serviceStatus.state as 'disconnected' | 'connecting' | 'connected' | 'error',
        reconnectAttempts: serviceStatus.reconnectAttempts
      })
    }, 5000)

    return () => {
      unsubscribe()
      clearInterval(interval)
    }
  }, [])

  return status
}

// Hook for real-time price data for specific instruments
export const useRealtimePrice = (instrumentId: string | null) => {
  const [price, setPrice] = useState<RealtimePrice | null>(null)
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)

  useEffect(() => {
    if (!instrumentId) {
      setPrice(null)
      setLastUpdate(null)
      return
    }

    // Subscribe to instrument
    realtimeService.subscribeToInstrument(instrumentId)

    const unsubscribe = realtimeService.subscribe('price', (update: RealtimeUpdate) => {
      if (update.type === 'price') {
        const priceData = update.data as RealtimePrice
        if (priceData.instrumentId === instrumentId) {
          setPrice(priceData)
          setLastUpdate(update.timestamp)
        }
      }
    })

    return () => {
      unsubscribe()
      if (instrumentId) {
        realtimeService.unsubscribeFromInstrument(instrumentId)
      }
    }
  }, [instrumentId])

  return { price, lastUpdate }
}

// Hook for real-time price data for multiple instruments
export const useRealtimePrices = (instrumentIds: string[]) => {
  const [prices, setPrices] = useState<Map<string, RealtimePrice>>(new Map())
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)
  const subscribedIds = useRef<Set<string>>(new Set())

  useEffect(() => {
    // Subscribe to new instruments
    instrumentIds.forEach(id => {
      if (!subscribedIds.current.has(id)) {
        realtimeService.subscribeToInstrument(id)
        subscribedIds.current.add(id)
      }
    })

    // Unsubscribe from removed instruments
    subscribedIds.current.forEach(id => {
      if (!instrumentIds.includes(id)) {
        realtimeService.unsubscribeFromInstrument(id)
        subscribedIds.current.delete(id)
        setPrices(prev => {
          const newPrices = new Map(prev)
          newPrices.delete(id)
          return newPrices
        })
      }
    })

    const unsubscribe = realtimeService.subscribe('price', (update: RealtimeUpdate) => {
      if (update.type === 'price') {
        const priceData = update.data as RealtimePrice
        if (instrumentIds.includes(priceData.instrumentId)) {
          setPrices(prev => new Map(prev.set(priceData.instrumentId, priceData)))
          setLastUpdate(update.timestamp)
        }
      }
    })

    return () => {
      unsubscribe()
      // Cleanup subscriptions
      subscribedIds.current.forEach(id => {
        realtimeService.unsubscribeFromInstrument(id)
      })
      subscribedIds.current.clear()
    }
  }, [instrumentIds])

  return { prices, lastUpdate }
}

// Hook for real-time venue status updates
export const useRealtimeVenueStatus = (venues: string[]) => {
  const [venueStatuses, setVenueStatuses] = useState<Map<string, VenueStatusUpdate>>(new Map())
  const [lastUpdate, setLastUpdate] = useState<string | null>(null)
  const subscribedVenues = useRef<Set<string>>(new Set())

  useEffect(() => {
    // Subscribe to new venues
    venues.forEach(venue => {
      if (!subscribedVenues.current.has(venue)) {
        realtimeService.subscribeToVenue(venue)
        subscribedVenues.current.add(venue)
      }
    })

    // Unsubscribe from removed venues
    subscribedVenues.current.forEach(venue => {
      if (!venues.includes(venue)) {
        realtimeService.unsubscribeFromVenue(venue)
        subscribedVenues.current.delete(venue)
        setVenueStatuses(prev => {
          const newStatuses = new Map(prev)
          newStatuses.delete(venue)
          return newStatuses
        })
      }
    })

    const unsubscribe = realtimeService.subscribe('venue_status', (update: RealtimeUpdate) => {
      if (update.type === 'venue_status') {
        const venueData = update.data as VenueStatusUpdate
        if (venues.includes(venueData.venue)) {
          setVenueStatuses(prev => new Map(prev.set(venueData.venue, venueData)))
          setLastUpdate(update.timestamp)
        }
      }
    })

    return () => {
      unsubscribe()
      // Cleanup subscriptions
      subscribedVenues.current.forEach(venue => {
        realtimeService.unsubscribeFromVenue(venue)
      })
      subscribedVenues.current.clear()
    }
  }, [venues])

  return { venueStatuses, lastUpdate }
}

// Hook for market session information
export const useMarketSession = (venue: string | null) => {
  const [sessionInfo, setSessionInfo] = useState<MarketSessionInfo | null>(null)
  const [loading, setLoading] = useState(false)

  const requestSession = useCallback(() => {
    if (venue) {
      setLoading(true)
      realtimeService.requestMarketSession(venue)
    }
  }, [venue])

  useEffect(() => {
    if (!venue) {
      setSessionInfo(null)
      return
    }

    // Request initial session info
    requestSession()

    const unsubscribe = realtimeService.subscribe('market_session', (update: RealtimeUpdate) => {
      if (update.type === 'market_session') {
        const sessionData = update.data as MarketSessionInfo
        if (sessionData.venue === venue) {
          setSessionInfo(sessionData)
          setLoading(false)
        }
      }
    })

    return unsubscribe
  }, [venue, requestSession])

  return { sessionInfo, loading, refetch: requestSession }
}

// Hook for aggregated real-time updates
export const useRealtimeUpdates = (instrumentIds: string[], venues: string[]) => {
  const priceHook = useRealtimePrices(instrumentIds)
  const venueHook = useRealtimeVenueStatus(venues)
  const connectionHook = useRealtimeConnection()

  return {
    prices: priceHook.prices,
    venueStatuses: venueHook.venueStatuses,
    connection: connectionHook,
    lastPriceUpdate: priceHook.lastUpdate,
    lastVenueUpdate: venueHook.lastUpdate
  }
}

// Hook for real-time event streaming (for debugging/monitoring)
export const useRealtimeEventStream = (maxEvents: number = 100) => {
  const [events, setEvents] = useState<RealtimeUpdate[]>([])

  useEffect(() => {
    const unsubscribe = realtimeService.subscribe('all', (update: RealtimeUpdate) => {
      setEvents(prev => {
        const newEvents = [update, ...prev].slice(0, maxEvents)
        return newEvents
      })
    })

    return unsubscribe
  }, [maxEvents])

  const clearEvents = useCallback(() => {
    setEvents([])
  }, [])

  return { events, clearEvents }
}