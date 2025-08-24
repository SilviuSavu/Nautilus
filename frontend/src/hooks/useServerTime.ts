/**
 * Server Time Integration Hook for Trading Operations
 * 
 * Features:
 * - Real-time server time display with nanosecond precision
 * - Trading session awareness (market open/close times)
 * - Time-based trading operation synchronization
 * - Market hours calculation and display
 * - Timezone handling for global markets
 * - Integration with useClockSync for accurate timing
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useClockSync } from './useClockSync';

export interface MarketHours {
  market: string;
  timezone: string;
  open: string; // HH:MM format
  close: string; // HH:MM format
  isOpen: boolean;
  nextOpen?: Date;
  nextClose?: Date;
  sessionType: 'pre-market' | 'regular' | 'after-hours' | 'closed';
}

export interface ServerTimeState {
  serverTime: Date;
  localTime: Date;
  timeDifference: number; // milliseconds
  formattedServerTime: string;
  formattedLocalTime: string;
  marketHours: MarketHours[];
  currentMarketSession: 'pre-market' | 'regular' | 'after-hours' | 'closed';
  tradingTimeRemaining?: number; // milliseconds until market close/open
}

export interface UseServerTimeOptions {
  updateInterval?: number; // Update interval in milliseconds (default: 100ms)
  includeMarkets?: string[]; // Markets to track (default: ['NYSE', 'NASDAQ', 'LSE', 'TSE'])
  timeFormat?: 'iso' | 'local' | 'trading'; // Time display format
  enableTradingAlerts?: boolean; // Enable market open/close alerts
}

export interface UseServerTimeReturn {
  serverTimeState: ServerTimeState;
  isMarketOpen: (market?: string) => boolean;
  getMarketTimeRemaining: (market: string) => number | null;
  formatServerTime: (format: 'iso' | 'local' | 'trading' | 'timestamp') => string;
  getTimestamp: () => number;
  getTimestampNanos: () => bigint;
  scheduleAtServerTime: (targetTime: Date, callback: () => void) => () => void;
}

const DEFAULT_OPTIONS: Required<UseServerTimeOptions> = {
  updateInterval: 100, // 100ms for smooth real-time updates
  includeMarkets: ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'ASX'],
  timeFormat: 'trading',
  enableTradingAlerts: true
};

// Market configurations with timezone and hours
const MARKET_CONFIGS = {
  NYSE: { timezone: 'America/New_York', open: '09:30', close: '16:00', name: 'New York Stock Exchange' },
  NASDAQ: { timezone: 'America/New_York', open: '09:30', close: '16:00', name: 'NASDAQ' },
  LSE: { timezone: 'Europe/London', open: '08:00', close: '16:30', name: 'London Stock Exchange' },
  TSE: { timezone: 'Asia/Tokyo', open: '09:00', close: '15:00', name: 'Tokyo Stock Exchange' },
  ASX: { timezone: 'Australia/Sydney', open: '10:00', close: '16:00', name: 'Australian Securities Exchange' },
  HKEX: { timezone: 'Asia/Hong_Kong', open: '09:30', close: '16:00', name: 'Hong Kong Exchanges' },
  SSE: { timezone: 'Asia/Shanghai', open: '09:30', close: '15:00', name: 'Shanghai Stock Exchange' },
  BSE: { timezone: 'Asia/Kolkata', open: '09:15', close: '15:30', name: 'Bombay Stock Exchange' }
};

export const useServerTime = (options: UseServerTimeOptions = {}): UseServerTimeReturn => {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const { clockState, getServerTime, isClockSynced } = useClockSync();
  
  const [serverTimeState, setServerTimeState] = useState<ServerTimeState>(() => ({
    serverTime: new Date(),
    localTime: new Date(),
    timeDifference: 0,
    formattedServerTime: '',
    formattedLocalTime: '',
    marketHours: [],
    currentMarketSession: 'closed'
  }));

  // Calculate market hours for configured markets
  const calculateMarketHours = useCallback((serverTime: Date): MarketHours[] => {
    return config.includeMarkets.map(market => {
      const config_market = MARKET_CONFIGS[market as keyof typeof MARKET_CONFIGS];
      if (!config_market) {
        return {
          market,
          timezone: 'UTC',
          open: '00:00',
          close: '00:00',
          isOpen: false,
          sessionType: 'closed'
        };
      }

      const { timezone, open, close, name } = config_market;
      
      // Create market time in market timezone
      const marketTime = new Date(serverTime.toLocaleString('en-US', { timeZone: timezone }));
      const [openHour, openMinute] = open.split(':').map(Number);
      const [closeHour, closeMinute] = close.split(':').map(Number);
      
      // Calculate market open/close times for today
      const marketOpen = new Date(marketTime);
      marketOpen.setHours(openHour, openMinute, 0, 0);
      
      const marketClose = new Date(marketTime);
      marketClose.setHours(closeHour, closeMinute, 0, 0);
      
      // Determine if market is currently open
      const currentTime = marketTime.getTime();
      const isOpen = currentTime >= marketOpen.getTime() && currentTime < marketClose.getTime();
      
      // Determine session type
      let sessionType: 'pre-market' | 'regular' | 'after-hours' | 'closed' = 'closed';
      if (isOpen) {
        sessionType = 'regular';
      } else if (currentTime < marketOpen.getTime()) {
        sessionType = 'pre-market';
      } else {
        sessionType = 'after-hours';
      }
      
      // Calculate next open/close times
      let nextOpen: Date | undefined;
      let nextClose: Date | undefined;
      
      if (isOpen) {
        nextClose = marketClose;
      } else if (currentTime < marketOpen.getTime()) {
        nextOpen = marketOpen;
      } else {
        // Market closed for today, next open is tomorrow
        nextOpen = new Date(marketOpen);
        nextOpen.setDate(nextOpen.getDate() + 1);
        
        // Handle weekends (move to Monday)
        while (nextOpen.getDay() === 0 || nextOpen.getDay() === 6) {
          nextOpen.setDate(nextOpen.getDate() + 1);
        }
      }

      return {
        market: name,
        timezone,
        open,
        close,
        isOpen,
        nextOpen,
        nextClose,
        sessionType
      };
    });
  }, [config.includeMarkets]);

  // Format time based on selected format
  const formatTime = useCallback((date: Date, format: 'iso' | 'local' | 'trading' | 'timestamp'): string => {
    switch (format) {
      case 'iso':
        return date.toISOString();
      case 'local':
        return date.toLocaleString();
      case 'trading':
        return date.toLocaleString('en-US', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          fractionalSecondDigits: 3,
          hour12: false
        });
      case 'timestamp':
        return date.getTime().toString();
      default:
        return date.toString();
    }
  }, []);

  // Update server time state
  const updateServerTimeState = useCallback(() => {
    if (!isClockSynced) {
      return;
    }

    const serverTime = new Date(getServerTime());
    const localTime = new Date();
    const timeDifference = serverTime.getTime() - localTime.getTime();
    
    const marketHours = calculateMarketHours(serverTime);
    
    // Determine current market session based on primary market (first in list)
    const primaryMarket = marketHours[0];
    const currentMarketSession = primaryMarket?.sessionType || 'closed';
    
    // Calculate trading time remaining for primary market
    let tradingTimeRemaining: number | undefined;
    if (primaryMarket?.nextClose) {
      tradingTimeRemaining = primaryMarket.nextClose.getTime() - serverTime.getTime();
    } else if (primaryMarket?.nextOpen) {
      tradingTimeRemaining = primaryMarket.nextOpen.getTime() - serverTime.getTime();
    }

    setServerTimeState({
      serverTime,
      localTime,
      timeDifference,
      formattedServerTime: formatTime(serverTime, config.timeFormat),
      formattedLocalTime: formatTime(localTime, config.timeFormat),
      marketHours,
      currentMarketSession,
      tradingTimeRemaining
    });
  }, [isClockSynced, getServerTime, calculateMarketHours, formatTime, config.timeFormat]);

  // Check if specific market is open
  const isMarketOpen = useCallback((market?: string): boolean => {
    if (!market) {
      // Return true if any market is open
      return serverTimeState.marketHours.some(m => m.isOpen);
    }
    
    const marketInfo = serverTimeState.marketHours.find(m => 
      m.market.toLowerCase().includes(market.toLowerCase())
    );
    return marketInfo?.isOpen || false;
  }, [serverTimeState.marketHours]);

  // Get time remaining until market open/close
  const getMarketTimeRemaining = useCallback((market: string): number | null => {
    const marketInfo = serverTimeState.marketHours.find(m => 
      m.market.toLowerCase().includes(market.toLowerCase())
    );
    
    if (!marketInfo) {
      return null;
    }
    
    const now = serverTimeState.serverTime.getTime();
    
    if (marketInfo.isOpen && marketInfo.nextClose) {
      return marketInfo.nextClose.getTime() - now;
    } else if (!marketInfo.isOpen && marketInfo.nextOpen) {
      return marketInfo.nextOpen.getTime() - now;
    }
    
    return null;
  }, [serverTimeState.marketHours, serverTimeState.serverTime]);

  // Format server time with specified format
  const formatServerTime = useCallback((format: 'iso' | 'local' | 'trading' | 'timestamp'): string => {
    return formatTime(serverTimeState.serverTime, format);
  }, [serverTimeState.serverTime, formatTime]);

  // Get current server timestamp
  const getTimestamp = useCallback((): number => {
    return Math.floor(getServerTime());
  }, [getServerTime]);

  // Get nanosecond precision timestamp
  const getTimestampNanos = useCallback((): bigint => {
    const timestamp = getServerTime();
    return BigInt(Math.floor(timestamp * 1000000)); // Convert to nanoseconds
  }, [getServerTime]);

  // Schedule callback at specific server time
  const scheduleAtServerTime = useCallback((targetTime: Date, callback: () => void): (() => void) => {
    const targetTimestamp = targetTime.getTime();
    
    const checkTime = () => {
      const currentServerTime = getServerTime();
      if (currentServerTime >= targetTimestamp) {
        callback();
        return true; // Stop checking
      }
      return false; // Continue checking
    };
    
    // Check immediately
    if (checkTime()) {
      return () => {}; // Return no-op cleanup
    }
    
    // Set up interval to check every 100ms
    const interval = setInterval(() => {
      if (checkTime()) {
        clearInterval(interval);
      }
    }, 100);
    
    // Return cleanup function
    return () => clearInterval(interval);
  }, [getServerTime]);

  // Set up regular updates
  useEffect(() => {
    // Initial update
    updateServerTimeState();
    
    // Set up interval for regular updates
    const interval = setInterval(updateServerTimeState, config.updateInterval);
    
    return () => clearInterval(interval);
  }, [updateServerTimeState, config.updateInterval]);

  // Handle trading alerts
  useEffect(() => {
    if (!config.enableTradingAlerts || !isClockSynced) {
      return;
    }

    // Set up alerts for market open/close events
    const alerts: (() => void)[] = [];
    
    serverTimeState.marketHours.forEach(market => {
      if (market.nextOpen) {
        const cleanup = scheduleAtServerTime(market.nextOpen, () => {
          console.log(`📈 Market Alert: ${market.market} is now open`);
          // Dispatch custom event for market open
          window.dispatchEvent(new CustomEvent('marketOpen', { detail: { market: market.market } }));
        });
        alerts.push(cleanup);
      }
      
      if (market.nextClose) {
        const cleanup = scheduleAtServerTime(market.nextClose, () => {
          console.log(`📉 Market Alert: ${market.market} is now closed`);
          // Dispatch custom event for market close
          window.dispatchEvent(new CustomEvent('marketClose', { detail: { market: market.market } }));
        });
        alerts.push(cleanup);
      }
    });
    
    // Cleanup function
    return () => {
      alerts.forEach(cleanup => cleanup());
    };
  }, [config.enableTradingAlerts, isClockSynced, serverTimeState.marketHours, scheduleAtServerTime]);

  return {
    serverTimeState,
    isMarketOpen,
    getMarketTimeRemaining,
    formatServerTime,
    getTimestamp,
    getTimestampNanos,
    scheduleAtServerTime
  };
};

export default useServerTime;