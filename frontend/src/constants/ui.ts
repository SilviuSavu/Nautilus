/**
 * UI Constants for Sprint 3 Components
 * Centralized configuration for consistent UI behavior
 */

export const UI_CONSTANTS = {
  // Refresh intervals (in milliseconds)
  REFRESH_INTERVALS: {
    FAST: 1000,
    NORMAL: 5000,
    SLOW: 30000,
    VERY_SLOW: 60000
  },

  // Animation durations (in milliseconds)
  ANIMATIONS: {
    FAST: 150,
    NORMAL: 300,
    SLOW: 500
  },

  // Notification durations (in seconds)
  NOTIFICATIONS: {
    SHORT: 2,
    NORMAL: 4,
    LONG: 6,
    PERMANENT: 0
  },

  // Performance thresholds
  PERFORMANCE_THRESHOLDS: {
    LATENCY_WARNING: 100,
    LATENCY_ERROR: 500,
    THROUGHPUT_WARNING: 10,
    THROUGHPUT_ERROR: 5,
    ERROR_RATE_WARNING: 5,
    ERROR_RATE_ERROR: 10,
    CPU_WARNING: 70,
    CPU_ERROR: 85,
    MEMORY_WARNING: 80,
    MEMORY_ERROR: 90
  },

  // Chart colors
  CHART_COLORS: {
    PRIMARY: '#1890ff',
    SUCCESS: '#52c41a',
    WARNING: '#faad14',
    ERROR: '#ff4d4f',
    INFO: '#722ed1',
    SECONDARY: '#d9d9d9'
  },

  // Status colors
  STATUS_COLORS: {
    CONNECTED: '#52c41a',
    CONNECTING: '#1890ff',
    DISCONNECTED: '#d9d9d9',
    ERROR: '#ff4d4f',
    WARNING: '#faad14'
  },

  // Layout breakpoints
  BREAKPOINTS: {
    XS: 480,
    SM: 768,
    MD: 992,
    LG: 1200,
    XL: 1600
  },

  // Component sizes
  COMPONENT_SIZES: {
    CARD_PADDING: 16,
    GRID_GUTTER: [16, 16] as [number, number],
    ICON_SIZE: 24,
    AVATAR_SIZE: 32
  },

  // Data limits
  DATA_LIMITS: {
    MAX_HISTORY_POINTS: 1000,
    MAX_CHART_POINTS: 100,
    MAX_ALERTS: 50,
    MAX_TABLE_ROWS: 100
  }
} as const;

export const ACCESSIBILITY = {
  // ARIA labels
  LABELS: {
    REFRESH_BUTTON: 'Refresh data',
    SETTINGS_BUTTON: 'Open settings',
    FULLSCREEN_BUTTON: 'Toggle fullscreen',
    CLOSE_BUTTON: 'Close',
    LOADING: 'Loading content',
    ERROR: 'Error occurred',
    SUCCESS: 'Operation successful'
  },

  // Keyboard shortcuts
  SHORTCUTS: {
    REFRESH: 'r',
    SETTINGS: 's',
    FULLSCREEN: 'f',
    ESCAPE: 'Escape'
  },

  // Focus management
  FOCUS_DELAY: 100
} as const;

export type UIConstantsType = typeof UI_CONSTANTS;
export type AccessibilityType = typeof ACCESSIBILITY;