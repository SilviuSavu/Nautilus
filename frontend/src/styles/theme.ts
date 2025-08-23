/**
 * Comprehensive Theme Configuration
 * Production-ready theme system with dark mode support and accessibility
 */

import type { ThemeConfig } from 'antd';

// Color palette
export const COLORS = {
  // Primary brand colors
  primary: {
    50: '#e6f7ff',
    100: '#bae7ff',
    200: '#91d5ff',
    300: '#69c0ff',
    400: '#40a9ff',
    500: '#1890ff', // Main primary
    600: '#096dd9',
    700: '#0050b3',
    800: '#003a8c',
    900: '#002766'
  },
  
  // Success colors
  success: {
    50: '#f6ffed',
    100: '#d9f7be',
    200: '#b7eb8f',
    300: '#95de64',
    400: '#73d13d',
    500: '#52c41a', // Main success
    600: '#389e0d',
    700: '#237804',
    800: '#135200',
    900: '#092b00'
  },
  
  // Warning colors
  warning: {
    50: '#fffbe6',
    100: '#fff1b8',
    200: '#ffe58f',
    300: '#ffd666',
    400: '#ffc53d',
    500: '#faad14', // Main warning
    600: '#d48806',
    700: '#ad6800',
    800: '#874d00',
    900: '#613400'
  },
  
  // Error colors
  error: {
    50: '#fff2f0',
    100: '#ffccc7',
    200: '#ffa39e',
    300: '#ff7875',
    400: '#ff4d4f', // Main error
    500: '#f5222d',
    600: '#cf1322',
    700: '#a8071a',
    800: '#820014',
    900: '#5c0011'
  },
  
  // Neutral colors
  neutral: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#f0f0f0',
    300: '#d9d9d9',
    400: '#bfbfbf',
    500: '#8c8c8c',
    600: '#595959',
    700: '#434343',
    800: '#262626',
    900: '#1f1f1f'
  },
  
  // Semantic colors for trading
  trading: {
    bullish: '#52c41a',
    bearish: '#f5222d',
    neutral: '#faad14',
    volume: '#722ed1',
    bid: '#52c41a',
    ask: '#f5222d'
  }
};

// Typography scale
export const TYPOGRAPHY = {
  fontFamily: {
    base: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    mono: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace'
  },
  fontSize: {
    xs: '12px',
    sm: '14px',
    base: '16px',
    lg: '18px',
    xl: '20px',
    '2xl': '24px',
    '3xl': '30px',
    '4xl': '36px'
  },
  fontWeight: {
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700
  },
  lineHeight: {
    tight: 1.25,
    base: 1.5,
    relaxed: 1.75
  }
};

// Spacing scale
export const SPACING = {
  0: '0',
  1: '4px',
  2: '8px',
  3: '12px',
  4: '16px',
  5: '20px',
  6: '24px',
  8: '32px',
  10: '40px',
  12: '48px',
  16: '64px',
  20: '80px',
  24: '96px'
};

// Shadow system
export const SHADOWS = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
};

// Border radius
export const RADIUS = {
  none: '0',
  sm: '4px',
  base: '6px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  '2xl': '24px',
  full: '50%'
};

// Z-index scale
export const Z_INDEX = {
  hide: -1,
  auto: 'auto',
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800
};

// Animation durations
export const DURATIONS = {
  fast: '150ms',
  base: '200ms',
  slow: '300ms',
  slower: '500ms'
};

// Easing functions
export const EASING = {
  easeOut: 'cubic-bezier(0.0, 0.0, 0.2, 1)',
  easeIn: 'cubic-bezier(0.4, 0.0, 1, 1)',
  easeInOut: 'cubic-bezier(0.4, 0.0, 0.2, 1)'
};

// Breakpoints
export const BREAKPOINTS = {
  xs: 480,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1600
};

// Light theme configuration
export const lightTheme: ThemeConfig = {
  token: {
    // Colors
    colorPrimary: COLORS.primary[500],
    colorSuccess: COLORS.success[500],
    colorWarning: COLORS.warning[500],
    colorError: COLORS.error[400],
    colorInfo: COLORS.primary[500],
    colorTextBase: COLORS.neutral[800],
    colorBgBase: '#ffffff',
    colorBgContainer: '#ffffff',
    colorBorder: COLORS.neutral[300],
    colorBorderSecondary: COLORS.neutral[200],
    
    // Typography
    fontFamily: TYPOGRAPHY.fontFamily.base,
    fontSize: 14,
    fontSizeHeading1: 38,
    fontSizeHeading2: 30,
    fontSizeHeading3: 24,
    fontSizeHeading4: 20,
    fontSizeHeading5: 16,
    fontSizeLG: 16,
    fontSizeSM: 12,
    fontSizeXL: 20,
    
    // Spacing
    padding: 16,
    paddingXS: 8,
    paddingSM: 12,
    paddingLG: 24,
    paddingXL: 32,
    margin: 16,
    marginXS: 8,
    marginSM: 12,
    marginLG: 24,
    marginXL: 32,
    
    // Border radius
    borderRadius: 6,
    borderRadiusLG: 8,
    borderRadiusSM: 4,
    borderRadiusXS: 2,
    
    // Box shadow
    boxShadow: SHADOWS.base,
    boxShadowSecondary: SHADOWS.sm,
    boxShadowTertiary: SHADOWS.lg,
    
    // Line height
    lineHeight: 1.5,
    lineHeightHeading1: 1.2,
    lineHeightHeading2: 1.3,
    lineHeightHeading3: 1.3,
    lineHeightHeading4: 1.4,
    lineHeightHeading5: 1.5,
    
    // Z-index
    zIndexBase: 0,
    zIndexPopupBase: 1000,
    
    // Motion
    motionDurationFast: DURATIONS.fast,
    motionDurationMid: DURATIONS.base,
    motionDurationSlow: DURATIONS.slow,
    motionEaseInOut: EASING.easeInOut,
    motionEaseOut: EASING.easeOut,
    motionEaseIn: EASING.easeIn,
    
    // Wireframe (set to false for production)
    wireframe: false,
    
    // Screen sizes
    screenXS: BREAKPOINTS.xs,
    screenSM: BREAKPOINTS.sm,
    screenMD: BREAKPOINTS.md,
    screenLG: BREAKPOINTS.lg,
    screenXL: BREAKPOINTS.xl,
    screenXXL: BREAKPOINTS.xxl
  },
  
  components: {
    // Card styling
    Card: {
      headerBg: 'transparent',
      headerHeight: 56,
      headerHeightSM: 48,
      paddingLG: 24,
      boxShadowTertiary: SHADOWS.base
    },
    
    // Button styling
    Button: {
      borderRadius: 6,
      controlHeight: 32,
      controlHeightLG: 40,
      controlHeightSM: 24,
      paddingInline: 16,
      paddingInlineLG: 24,
      paddingInlineSM: 12
    },
    
    // Table styling
    Table: {
      headerBg: COLORS.neutral[50],
      headerColor: COLORS.neutral[700],
      rowHoverBg: COLORS.primary[50],
      borderColor: COLORS.neutral[200],
      headerSortActiveBg: COLORS.neutral[100],
      headerSortHoverBg: COLORS.neutral[100],
      fixedHeaderSortActiveBg: COLORS.neutral[100]
    },
    
    // Input styling
    Input: {
      borderRadius: 6,
      controlHeight: 32,
      controlHeightLG: 40,
      controlHeightSM: 24,
      paddingInline: 12,
      paddingBlock: 4
    },
    
    // Modal styling
    Modal: {
      borderRadiusLG: 8,
      paddingLG: 24,
      paddingMD: 20,
      paddingContentHorizontalLG: 24,
      marginLG: 24
    },
    
    // Notification styling
    Notification: {
      borderRadiusLG: 8,
      paddingMD: 16,
      paddingLG: 20
    },
    
    // Progress styling
    Progress: {
      remainingColor: COLORS.neutral[200],
      defaultColor: COLORS.primary[500]
    },
    
    // Tag styling
    Tag: {
      borderRadiusSM: 4,
      fontSizeSM: 12,
      lineHeightSM: 1.5
    },
    
    // Badge styling
    Badge: {
      fontSizeSM: 12,
      lineHeight: 1.5
    },
    
    // Alert styling
    Alert: {
      borderRadiusLG: 6,
      paddingMD: 12,
      paddingLG: 16
    },
    
    // Spin styling
    Spin: {
      contentHeight: 400
    },
    
    // Layout styling
    Layout: {
      headerBg: '#ffffff',
      headerHeight: 64,
      headerPadding: '0 24px',
      siderBg: '#ffffff',
      triggerBg: COLORS.neutral[100],
      triggerHeight: 48
    }
  }
};

// Dark theme configuration
export const darkTheme: ThemeConfig = {
  ...lightTheme,
  token: {
    ...lightTheme.token,
    // Dark theme specific colors
    colorTextBase: COLORS.neutral[100],
    colorBgBase: COLORS.neutral[900],
    colorBgContainer: COLORS.neutral[800],
    colorBorder: COLORS.neutral[700],
    colorBorderSecondary: COLORS.neutral[600],
    colorFillSecondary: COLORS.neutral[700],
    colorFillTertiary: COLORS.neutral[600],
    colorFillQuaternary: COLORS.neutral[500]
  },
  
  components: {
    ...lightTheme.components,
    
    // Dark theme component overrides
    Card: {
      ...lightTheme.components?.Card,
      colorBgContainer: COLORS.neutral[800],
      colorBorderSecondary: COLORS.neutral[700]
    },
    
    Table: {
      ...lightTheme.components?.Table,
      headerBg: COLORS.neutral[800],
      headerColor: COLORS.neutral[200],
      rowHoverBg: COLORS.neutral[700],
      borderColor: COLORS.neutral[600]
    },
    
    Layout: {
      ...lightTheme.components?.Layout,
      headerBg: COLORS.neutral[800],
      siderBg: COLORS.neutral[800],
      triggerBg: COLORS.neutral[700]
    }
  }
};

// Theme utilities
export const getTheme = (isDark: boolean): ThemeConfig => {
  return isDark ? darkTheme : lightTheme;
};

// CSS custom properties for dynamic theming
export const getCSSVariables = (isDark: boolean) => {
  const theme = getTheme(isDark);
  return {
    '--color-primary': theme.token?.colorPrimary,
    '--color-success': theme.token?.colorSuccess,
    '--color-warning': theme.token?.colorWarning,
    '--color-error': theme.token?.colorError,
    '--color-text': theme.token?.colorTextBase,
    '--color-bg': theme.token?.colorBgBase,
    '--color-bg-container': theme.token?.colorBgContainer,
    '--color-border': theme.token?.colorBorder,
    '--font-family': theme.token?.fontFamily,
    '--border-radius': `${theme.token?.borderRadius}px`,
    '--box-shadow': theme.token?.boxShadow
  };
};

export default {
  COLORS,
  TYPOGRAPHY,
  SPACING,
  SHADOWS,
  RADIUS,
  Z_INDEX,
  DURATIONS,
  EASING,
  BREAKPOINTS,
  lightTheme,
  darkTheme,
  getTheme,
  getCSSVariables
};