/**
 * Connection Stability Monitor
 * Tracks connection stability and quality metrics for trading venues
 */

import { ConnectionQuality } from '../../types/monitoring';

export interface ConnectionEvent {
  event_id: string;
  venue_name: string;
  event_type: 'connect' | 'disconnect' | 'reconnect' | 'timeout' | 'error' | 'heartbeat';
  timestamp: Date;
  duration_ms?: number;
  error_message?: string;
  metadata?: Record<string, any>;
}

export interface ConnectionSession {
  session_id: string;
  venue_name: string;
  start_time: Date;
  end_time?: Date;
  duration_seconds?: number;
  disconnection_reason?: string;
  data_quality_score: number;
  average_latency_ms: number;
  total_messages: number;
  error_count: number;
}

export interface StabilityMetrics {
  venue_name: string;
  stability_score: number; // 0-100
  uptime_percent_24h: number;
  connection_count_24h: number;
  disconnection_count_24h: number;
  average_session_duration_minutes: number;
  longest_session_duration_minutes: number;
  mean_time_between_failures_minutes: number;
  reconnection_success_rate: number;
  last_updated: Date;
}

export class ConnectionStabilityMonitor {
  private connectionEvents: Map<string, ConnectionEvent[]> = new Map();
  private activeSessions: Map<string, ConnectionSession> = new Map();
  private completedSessions: Map<string, ConnectionSession[]> = new Map();
  private heartbeatTimers: Map<string, NodeJS.Timeout> = new Map();
  private stabilityMetrics: Map<string, StabilityMetrics> = new Map();
  
  private maxEventsPerVenue: number = 1000;
  private maxSessionsPerVenue: number = 500;
  private heartbeatIntervalMs: number = 30000; // 30 seconds
  private callbacks: {
    onConnect: ((venue: string, sessionId: string) => void)[];
    onDisconnect: ((venue: string, sessionId: string, reason?: string) => void)[];
    onStabilityChange: ((venue: string, metrics: StabilityMetrics) => void)[];
  } = { onConnect: [], onDisconnect: [], onStabilityChange: [] };

  private nextSessionId: number = 1;
  private nextEventId: number = 1;

  /**
   * Start monitoring a connection
   */
  startConnection(venue: string, metadata?: Record<string, any>): string {
    const sessionId = `session_${this.nextSessionId++}`;
    
    // Create connection event
    const connectEvent: ConnectionEvent = {
      event_id: `event_${this.nextEventId++}`,
      venue_name: venue,
      event_type: 'connect',
      timestamp: new Date(),
      metadata
    };

    this.addConnectionEvent(venue, connectEvent);

    // Create active session
    const session: ConnectionSession = {
      session_id: sessionId,
      venue_name: venue,
      start_time: new Date(),
      data_quality_score: 100,
      average_latency_ms: 0,
      total_messages: 0,
      error_count: 0
    };

    this.activeSessions.set(sessionId, session);

    // Start heartbeat monitoring
    this.startHeartbeatMonitoring(venue, sessionId);

    // Update metrics
    this.updateStabilityMetrics(venue);

    // Notify callbacks
    this.callbacks.onConnect.forEach(callback => callback(venue, sessionId));

    return sessionId;
  }

  /**
   * End a connection
   */
  endConnection(sessionId: string, reason?: string): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      console.warn(`ConnectionStabilityMonitor: Session ${sessionId} not found`);
      return;
    }

    const endTime = new Date();
    const durationSeconds = (endTime.getTime() - session.start_time.getTime()) / 1000;

    // Update session
    session.end_time = endTime;
    session.duration_seconds = durationSeconds;
    session.disconnection_reason = reason;

    // Create disconnect event
    const disconnectEvent: ConnectionEvent = {
      event_id: `event_${this.nextEventId++}`,
      venue_name: session.venue_name,
      event_type: 'disconnect',
      timestamp: endTime,
      duration_ms: durationSeconds * 1000,
      error_message: reason
    };

    this.addConnectionEvent(session.venue_name, disconnectEvent);

    // Move to completed sessions
    if (!this.completedSessions.has(session.venue_name)) {
      this.completedSessions.set(session.venue_name, []);
    }
    
    const venueSessions = this.completedSessions.get(session.venue_name)!;
    venueSessions.push(session);
    
    // Maintain session limit
    if (venueSessions.length > this.maxSessionsPerVenue) {
      venueSessions.shift();
    }

    // Remove from active sessions
    this.activeSessions.delete(sessionId);

    // Stop heartbeat monitoring
    this.stopHeartbeatMonitoring(sessionId);

    // Update metrics
    this.updateStabilityMetrics(session.venue_name);

    // Notify callbacks
    this.callbacks.onDisconnect.forEach(callback => callback(session.venue_name, sessionId, reason));
  }

  /**
   * Record a connection error
   */
  recordError(venue: string, errorMessage: string, sessionId?: string): void {
    const errorEvent: ConnectionEvent = {
      event_id: `event_${this.nextEventId++}`,
      venue_name: venue,
      event_type: 'error',
      timestamp: new Date(),
      error_message: errorMessage
    };

    this.addConnectionEvent(venue, errorEvent);

    // Update session error count if session is active
    if (sessionId) {
      const session = this.activeSessions.get(sessionId);
      if (session) {
        session.error_count++;
        session.data_quality_score = Math.max(0, session.data_quality_score - 5); // Reduce quality score
      }
    }

    this.updateStabilityMetrics(venue);
  }

  /**
   * Record message received (for data quality tracking)
   */
  recordMessage(sessionId: string, latencyMs: number): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    session.total_messages++;
    
    // Update running average latency
    if (session.total_messages === 1) {
      session.average_latency_ms = latencyMs;
    } else {
      session.average_latency_ms = (session.average_latency_ms * (session.total_messages - 1) + latencyMs) / session.total_messages;
    }

    // Update data quality score based on latency
    if (latencyMs > 1000) { // > 1 second
      session.data_quality_score = Math.max(0, session.data_quality_score - 2);
    } else if (latencyMs > 500) { // > 500ms
      session.data_quality_score = Math.max(0, session.data_quality_score - 1);
    }
  }

  /**
   * Get current connection quality for venue
   */
  getConnectionQuality(venue: string): ConnectionQuality {
    const activeSession = this.getActiveSession(venue);
    const stabilityMetrics = this.stabilityMetrics.get(venue);
    const recentEvents = this.getRecentEvents(venue, 24 * 60 * 60 * 1000); // Last 24h

    const status = activeSession ? 'connected' : 'disconnected';
    const uptime = stabilityMetrics?.uptime_percent_24h || 0;
    const connectionDuration = activeSession ? 
      (Date.now() - activeSession.start_time.getTime()) / 1000 : 0;

    // Calculate quality score
    let qualityScore = 100;
    if (uptime < 95) qualityScore -= (95 - uptime) * 2;
    if (activeSession && activeSession.error_count > 0) qualityScore -= activeSession.error_count * 5;
    if (activeSession && activeSession.average_latency_ms > 100) qualityScore -= 10;

    const disconnectEvents = recentEvents.filter(e => e.event_type === 'disconnect');
    const lastDisconnect = disconnectEvents.length > 0 ? disconnectEvents[disconnectEvents.length - 1] : null;

    return {
      venue_name: venue,
      status: status as any,
      quality_score: Math.max(0, Math.min(100, Math.round(qualityScore))),
      uptime_percent_24h: Math.round(uptime * 100) / 100,
      connection_duration_seconds: Math.round(connectionDuration),
      last_disconnect_time: lastDisconnect?.timestamp.toISOString(),
      disconnect_count_24h: disconnectEvents.length,
      data_quality: {
        message_rate_per_sec: activeSession ? this.calculateMessageRate(activeSession) : 0,
        duplicate_messages_percent: 0, // Would need more sophisticated tracking
        out_of_sequence_percent: 0, // Would need sequence number tracking
        stale_data_percent: 0 // Would need timestamp analysis
      },
      performance_metrics: {
        response_time_ms: activeSession?.average_latency_ms || 0,
        throughput_mbps: 0, // Would need bandwidth tracking
        error_rate_percent: activeSession ? this.calculateErrorRate(activeSession) : 0
      },
      reconnection_stats: {
        auto_reconnect_enabled: true, // Configurable
        reconnect_attempts_24h: this.countReconnectAttempts(venue),
        avg_reconnect_time_seconds: this.calculateAverageReconnectTime(venue),
        max_reconnect_time_seconds: this.calculateMaxReconnectTime(venue)
      }
    };
  }

  /**
   * Get stability metrics for venue
   */
  getStabilityMetrics(venue: string): StabilityMetrics | null {
    return this.stabilityMetrics.get(venue) || null;
  }

  /**
   * Get all monitored venues
   */
  getMonitoredVenues(): string[] {
    const venues = new Set<string>();
    
    // Add venues with active sessions
    for (const session of this.activeSessions.values()) {
      venues.add(session.venue_name);
    }
    
    // Add venues with completed sessions
    for (const venue of this.completedSessions.keys()) {
      venues.add(venue);
    }
    
    // Add venues with events
    for (const venue of this.connectionEvents.keys()) {
      venues.add(venue);
    }

    return Array.from(venues);
  }

  /**
   * Get connection events for venue
   */
  getConnectionEvents(venue: string, limit: number = 100): ConnectionEvent[] {
    const events = this.connectionEvents.get(venue) || [];
    return events.slice(-limit);
  }

  /**
   * Get connection history summary
   */
  getConnectionHistory(venue: string): {
    total_sessions: number;
    active_sessions: number;
    average_session_duration_minutes: number;
    total_uptime_minutes: number;
    total_downtime_minutes: number;
    reliability_score: number;
  } {
    const completedSessions = this.completedSessions.get(venue) || [];
    const activeSessions = Array.from(this.activeSessions.values()).filter(s => s.venue_name === venue);
    const allSessions = [...completedSessions, ...activeSessions];

    if (allSessions.length === 0) {
      return {
        total_sessions: 0,
        active_sessions: 0,
        average_session_duration_minutes: 0,
        total_uptime_minutes: 0,
        total_downtime_minutes: 0,
        reliability_score: 0
      };
    }

    const totalSessions = allSessions.length;
    const avgSessionDuration = this.calculateAverageSessionDuration(allSessions);
    const totalUptimeMinutes = completedSessions.reduce((sum, session) => {
      return sum + (session.duration_seconds || 0) / 60;
    }, 0);

    // Estimate downtime from disconnect events
    const events = this.connectionEvents.get(venue) || [];
    const disconnectEvents = events.filter(e => e.event_type === 'disconnect');
    const reconnectEvents = events.filter(e => e.event_type === 'reconnect');
    
    let totalDowntimeMinutes = 0;
    for (let i = 0; i < Math.min(disconnectEvents.length, reconnectEvents.length); i++) {
      const downtime = (reconnectEvents[i].timestamp.getTime() - disconnectEvents[i].timestamp.getTime()) / 1000 / 60;
      totalDowntimeMinutes += Math.max(0, downtime);
    }

    const reliabilityScore = totalUptimeMinutes > 0 ? 
      (totalUptimeMinutes / (totalUptimeMinutes + totalDowntimeMinutes)) * 100 : 0;

    return {
      total_sessions: totalSessions,
      active_sessions: activeSessions.length,
      average_session_duration_minutes: Math.round(avgSessionDuration * 100) / 100,
      total_uptime_minutes: Math.round(totalUptimeMinutes * 100) / 100,
      total_downtime_minutes: Math.round(totalDowntimeMinutes * 100) / 100,
      reliability_score: Math.round(reliabilityScore * 100) / 100
    };
  }

  /**
   * Add callbacks
   */
  onConnect(callback: (venue: string, sessionId: string) => void): void {
    this.callbacks.onConnect.push(callback);
  }

  onDisconnect(callback: (venue: string, sessionId: string, reason?: string) => void): void {
    this.callbacks.onDisconnect.push(callback);
  }

  onStabilityChange(callback: (venue: string, metrics: StabilityMetrics) => void): void {
    this.callbacks.onStabilityChange.push(callback);
  }

  /**
   * Remove callbacks
   */
  removeCallback(callback: Function): void {
    let index = this.callbacks.onConnect.indexOf(callback as any);
    if (index > -1) this.callbacks.onConnect.splice(index, 1);

    index = this.callbacks.onDisconnect.indexOf(callback as any);
    if (index > -1) this.callbacks.onDisconnect.splice(index, 1);

    index = this.callbacks.onStabilityChange.indexOf(callback as any);
    if (index > -1) this.callbacks.onStabilityChange.splice(index, 1);
  }

  /**
   * Clear all monitoring data
   */
  clear(): void {
    // Clear all timers
    for (const timer of this.heartbeatTimers.values()) {
      clearInterval(timer);
    }

    this.connectionEvents.clear();
    this.activeSessions.clear();
    this.completedSessions.clear();
    this.heartbeatTimers.clear();
    this.stabilityMetrics.clear();
    this.nextSessionId = 1;
    this.nextEventId = 1;
  }

  // Private methods

  private addConnectionEvent(venue: string, event: ConnectionEvent): void {
    if (!this.connectionEvents.has(venue)) {
      this.connectionEvents.set(venue, []);
    }

    const events = this.connectionEvents.get(venue)!;
    events.push(event);

    // Maintain event limit
    if (events.length > this.maxEventsPerVenue) {
      events.shift();
    }
  }

  private getActiveSession(venue: string): ConnectionSession | null {
    for (const session of this.activeSessions.values()) {
      if (session.venue_name === venue) {
        return session;
      }
    }
    return null;
  }

  private getRecentEvents(venue: string, timeRangeMs: number): ConnectionEvent[] {
    const events = this.connectionEvents.get(venue) || [];
    const cutoff = new Date(Date.now() - timeRangeMs);
    
    return events.filter(event => event.timestamp >= cutoff);
  }

  private startHeartbeatMonitoring(venue: string, sessionId: string): void {
    const timer = setInterval(() => {
      const heartbeatEvent: ConnectionEvent = {
        event_id: `event_${this.nextEventId++}`,
        venue_name: venue,
        event_type: 'heartbeat',
        timestamp: new Date()
      };
      
      this.addConnectionEvent(venue, heartbeatEvent);
    }, this.heartbeatIntervalMs);

    this.heartbeatTimers.set(sessionId, timer);
  }

  private stopHeartbeatMonitoring(sessionId: string): void {
    const timer = this.heartbeatTimers.get(sessionId);
    if (timer) {
      clearInterval(timer);
      this.heartbeatTimers.delete(sessionId);
    }
  }

  private updateStabilityMetrics(venue: string): void {
    const now = new Date();
    const last24h = new Date(now.getTime() - (24 * 60 * 60 * 1000));
    
    const recentEvents = this.getRecentEvents(venue, 24 * 60 * 60 * 1000);
    const completedSessions = (this.completedSessions.get(venue) || [])
      .filter(session => session.start_time >= last24h);
    
    const connectEvents = recentEvents.filter(e => e.event_type === 'connect');
    const disconnectEvents = recentEvents.filter(e => e.event_type === 'disconnect');
    
    // Calculate uptime percentage
    const totalSessionTime = completedSessions.reduce((sum, session) => {
      return sum + (session.duration_seconds || 0);
    }, 0);
    
    const activeSession = this.getActiveSession(venue);
    const activeTime = activeSession ? 
      (now.getTime() - activeSession.start_time.getTime()) / 1000 : 0;
    
    const totalUptime = totalSessionTime + activeTime;
    const uptimePercent = (totalUptime / (24 * 60 * 60)) * 100; // 24 hours in seconds

    // Calculate other metrics
    const avgSessionDuration = this.calculateAverageSessionDuration(completedSessions) / 60; // Convert to minutes
    const longestSession = Math.max(...completedSessions.map(s => s.duration_seconds || 0), activeTime) / 60;
    const mtbf = this.calculateMTBF(venue);
    const reconnectSuccessRate = this.calculateReconnectSuccessRate(venue);

    const stabilityScore = this.calculateStabilityScore(
      uptimePercent,
      disconnectEvents.length,
      avgSessionDuration,
      reconnectSuccessRate
    );

    const metrics: StabilityMetrics = {
      venue_name: venue,
      stability_score: Math.round(stabilityScore),
      uptime_percent_24h: Math.min(100, Math.round(uptimePercent * 100) / 100),
      connection_count_24h: connectEvents.length,
      disconnection_count_24h: disconnectEvents.length,
      average_session_duration_minutes: Math.round(avgSessionDuration * 100) / 100,
      longest_session_duration_minutes: Math.round(longestSession * 100) / 100,
      mean_time_between_failures_minutes: Math.round(mtbf * 100) / 100,
      reconnection_success_rate: Math.round(reconnectSuccessRate * 100) / 100,
      last_updated: now
    };

    this.stabilityMetrics.set(venue, metrics);

    // Notify callbacks
    this.callbacks.onStabilityChange.forEach(callback => callback(venue, metrics));
  }

  private calculateMessageRate(session: ConnectionSession): number {
    const sessionDurationSeconds = (Date.now() - session.start_time.getTime()) / 1000;
    return sessionDurationSeconds > 0 ? session.total_messages / sessionDurationSeconds : 0;
  }

  private calculateErrorRate(session: ConnectionSession): number {
    return session.total_messages > 0 ? (session.error_count / session.total_messages) * 100 : 0;
  }

  private countReconnectAttempts(venue: string): number {
    const recentEvents = this.getRecentEvents(venue, 24 * 60 * 60 * 1000);
    return recentEvents.filter(e => e.event_type === 'reconnect').length;
  }

  private calculateAverageReconnectTime(venue: string): number {
    const events = this.connectionEvents.get(venue) || [];
    const reconnectTimes: number[] = [];
    
    for (let i = 1; i < events.length; i++) {
      if (events[i].event_type === 'reconnect' && events[i-1].event_type === 'disconnect') {
        const reconnectTimeMs = events[i].timestamp.getTime() - events[i-1].timestamp.getTime();
        reconnectTimes.push(reconnectTimeMs / 1000); // Convert to seconds
      }
    }
    
    return reconnectTimes.length > 0 ? 
      reconnectTimes.reduce((sum, time) => sum + time, 0) / reconnectTimes.length : 0;
  }

  private calculateMaxReconnectTime(venue: string): number {
    const events = this.connectionEvents.get(venue) || [];
    let maxReconnectTime = 0;
    
    for (let i = 1; i < events.length; i++) {
      if (events[i].event_type === 'reconnect' && events[i-1].event_type === 'disconnect') {
        const reconnectTimeMs = events[i].timestamp.getTime() - events[i-1].timestamp.getTime();
        maxReconnectTime = Math.max(maxReconnectTime, reconnectTimeMs / 1000);
      }
    }
    
    return maxReconnectTime;
  }

  private calculateAverageSessionDuration(sessions: ConnectionSession[]): number {
    if (sessions.length === 0) return 0;
    
    const totalDuration = sessions.reduce((sum, session) => sum + (session.duration_seconds || 0), 0);
    return totalDuration / sessions.length;
  }

  private calculateMTBF(venue: string): number {
    const completedSessions = this.completedSessions.get(venue) || [];
    if (completedSessions.length < 2) return 0;
    
    const totalOperatingTime = completedSessions.reduce((sum, session) => {
      return sum + (session.duration_seconds || 0);
    }, 0);
    
    const failureCount = completedSessions.filter(session => 
      session.disconnection_reason && session.disconnection_reason !== 'normal_shutdown'
    ).length;
    
    return failureCount > 0 ? (totalOperatingTime / failureCount) / 60 : 0; // Convert to minutes
  }

  private calculateReconnectSuccessRate(venue: string): number {
    const events = this.connectionEvents.get(venue) || [];
    const disconnectEvents = events.filter(e => e.event_type === 'disconnect').length;
    const reconnectEvents = events.filter(e => e.event_type === 'reconnect').length;
    
    return disconnectEvents > 0 ? (reconnectEvents / disconnectEvents) * 100 : 100;
  }

  private calculateStabilityScore(uptime: number, disconnects: number, avgSession: number, reconnectRate: number): number {
    let score = 100;
    
    // Uptime component (40% weight)
    score *= (uptime / 100) * 0.4 + 0.6;
    
    // Disconnect penalty (20% weight)
    const disconnectPenalty = Math.min(50, disconnects * 5); // Max 50% penalty
    score -= disconnectPenalty * 0.2;
    
    // Session duration component (20% weight)
    const sessionBonus = Math.min(20, avgSession / 60 * 10); // Longer sessions = better
    score += sessionBonus * 0.2;
    
    // Reconnect rate component (20% weight)
    score *= (reconnectRate / 100) * 0.2 + 0.8;
    
    return Math.max(0, Math.min(100, score));
  }
}

// Global instance for connection stability monitoring
export const connectionStabilityMonitor = new ConnectionStabilityMonitor();