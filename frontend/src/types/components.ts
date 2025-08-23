/**
 * Component Type Definitions
 * Comprehensive type definitions for all Sprint 3 components
 */

import { ReactNode } from 'react';

// Base component props
export interface BaseComponentProps {
  className?: string;
  style?: React.CSSProperties;
  children?: ReactNode;
  testId?: string;
}

// Common props for data components
export interface DataComponentProps extends BaseComponentProps {
  loading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
  refreshInterval?: number;
  autoRefresh?: boolean;
}

// Performance monitoring props
export interface PerformanceProps {
  enablePerformanceMonitoring?: boolean;
  performanceThresholds?: {
    memoryUsage?: number;
    renderTime?: number;
    updateFrequency?: number;
  };
  onPerformanceAlert?: (alert: PerformanceAlert) => void;
}

export interface PerformanceAlert {
  type: 'memory' | 'render' | 'update';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  value: number;
  threshold: number;
  timestamp: number;
}

// Accessibility props
export interface AccessibilityProps {
  ariaLabel?: string;
  ariaLabelledBy?: string;
  ariaDescribedBy?: string;
  role?: string;
  tabIndex?: number;
  focusable?: boolean;
  keyboardShortcuts?: Record<string, () => void>;
}

// Responsive design props
export interface ResponsiveProps {
  breakpoints?: {
    xs?: boolean;
    sm?: boolean;
    md?: boolean;
    lg?: boolean;
    xl?: boolean;
    xxl?: boolean;
  };
  responsiveConfig?: {
    [breakpoint: string]: Partial<any>;
  };
}

// Theme props
export interface ThemeProps {
  theme?: 'light' | 'dark' | 'auto';
  colorScheme?: 'default' | 'compact' | 'comfortable';
  customColors?: Record<string, string>;
}

// WebSocket-related types
export interface WebSocketConnectionState {
  isConnected: boolean;
  connectionId: string;
  lastConnected?: Date;
  reconnectAttempts: number;
  error?: string;
}

export interface WebSocketMessage {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  priority?: 'high' | 'normal' | 'low';
  source?: string;
}

export interface WebSocketSubscription {
  id: string;
  messageType: string;
  callback: (message: WebSocketMessage) => void;
  active: boolean;
  filters?: Record<string, any>;
}

// Chart and visualization types
export interface ChartDataPoint {
  timestamp: number;
  time?: string;
  value: number;
  label?: string;
  category?: string;
  metadata?: Record<string, any>;
}

export interface ChartConfig {
  type: 'line' | 'area' | 'bar' | 'pie' | 'scatter';
  dataKeys: string[];
  colors?: string[];
  animations?: boolean;
  tooltips?: boolean;
  legend?: boolean;
  responsive?: boolean;
}

export interface VisualizationProps extends DataComponentProps {
  data: ChartDataPoint[];
  config: ChartConfig;
  height?: number;
  width?: number;
  interactionMode?: 'none' | 'hover' | 'click' | 'zoom' | 'pan';
  exportOptions?: {
    enabled: boolean;
    formats: ('png' | 'svg' | 'pdf' | 'csv')[];
  };
}

// Table and list types
export interface TableColumn<T = any> {
  key: string;
  title: string;
  dataIndex?: keyof T;
  render?: (value: any, record: T, index: number) => ReactNode;
  sorter?: boolean | ((a: T, b: T) => number);
  filter?: boolean;
  width?: number | string;
  fixed?: 'left' | 'right';
  align?: 'left' | 'center' | 'right';
}

export interface TableProps<T = any> extends DataComponentProps {
  dataSource: T[];
  columns: TableColumn<T>[];
  rowKey: string | ((record: T) => string);
  pagination?: boolean | PaginationConfig;
  selection?: {
    enabled: boolean;
    onChange?: (selectedKeys: string[], selectedRows: T[]) => void;
  };
  virtualization?: {
    enabled: boolean;
    rowHeight: number;
    overscan?: number;
  };
}

export interface PaginationConfig {
  current?: number;
  pageSize?: number;
  total?: number;
  showSizeChanger?: boolean;
  showQuickJumper?: boolean;
  showTotal?: (total: number, range: [number, number]) => string;
  onChange?: (page: number, pageSize?: number) => void;
}

// Form and input types
export interface FormFieldProps extends BaseComponentProps {
  name: string;
  label?: string;
  placeholder?: string;
  required?: boolean;
  disabled?: boolean;
  readOnly?: boolean;
  value?: any;
  defaultValue?: any;
  onChange?: (value: any) => void;
  onBlur?: () => void;
  onFocus?: () => void;
  validation?: {
    rules: ValidationRule[];
    validateTrigger?: 'onChange' | 'onBlur' | 'onSubmit';
  };
  helpText?: string;
  errorMessage?: string;
}

export interface ValidationRule {
  type: 'required' | 'pattern' | 'min' | 'max' | 'custom';
  value?: any;
  message: string;
  validator?: (value: any) => boolean | Promise<boolean>;
}

// Navigation and routing types
export interface NavigationItem {
  key: string;
  label: string;
  icon?: ReactNode;
  path?: string;
  children?: NavigationItem[];
  disabled?: boolean;
  badge?: {
    count: number;
    color?: string;
  };
  permissions?: string[];
}

export interface BreadcrumbItem {
  key: string;
  label: string;
  path?: string;
  icon?: ReactNode;
}

// Modal and overlay types
export interface ModalProps extends BaseComponentProps {
  visible: boolean;
  title?: string;
  content?: ReactNode;
  footer?: ReactNode;
  width?: number | string;
  height?: number | string;
  closable?: boolean;
  maskClosable?: boolean;
  keyboard?: boolean;
  centered?: boolean;
  zIndex?: number;
  onCancel?: () => void;
  onOk?: () => void | Promise<void>;
  confirmLoading?: boolean;
}

// Notification and feedback types
export interface NotificationConfig {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
  placement?: 'topLeft' | 'topRight' | 'bottomLeft' | 'bottomRight';
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
  onClose?: () => void;
}

// Layout types
export interface LayoutProps extends BaseComponentProps {
  header?: ReactNode;
  sidebar?: ReactNode;
  footer?: ReactNode;
  breadcrumb?: BreadcrumbItem[];
  navigation?: NavigationItem[];
  collapsed?: boolean;
  collapsible?: boolean;
  responsive?: boolean;
  theme?: 'light' | 'dark';
  onCollapse?: (collapsed: boolean) => void;
}

// Monitoring and metrics types
export interface MetricDefinition {
  key: string;
  name: string;
  unit?: string;
  format?: 'number' | 'percentage' | 'duration' | 'bytes';
  thresholds?: {
    warning: number;
    critical: number;
  };
  aggregation?: 'sum' | 'average' | 'min' | 'max' | 'count';
}

export interface MetricValue {
  metric: string;
  value: number;
  timestamp: number;
  tags?: Record<string, string>;
}

export interface AlertRule {
  id: string;
  name: string;
  description?: string;
  metric: string;
  condition: 'gt' | 'lt' | 'eq' | 'ne' | 'gte' | 'lte';
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  notifications?: {
    channels: string[];
    template?: string;
  };
}

// State management types
export interface StoreState<T = any> {
  data: T;
  loading: boolean;
  error: string | null;
  lastUpdated?: Date;
  meta?: Record<string, any>;
}

export interface StoreActions<T = any> {
  setData: (data: T) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
  refresh: () => Promise<void>;
}

// Export utility type helpers
export type OptionalExcept<T, K extends keyof T> = Partial<T> & Pick<T, K>;
export type RequiredExcept<T, K extends keyof T> = Required<T> & Partial<Pick<T, K>>;
export type WithRequired<T, K extends keyof T> = T & Required<Pick<T, K>>;
export type Overwrite<T, U> = Pick<T, Exclude<keyof T, keyof U>> & U;

// Component factory types
export type ComponentFactory<P = {}> = (props: P) => ReactNode;
export type HOCFactory<P = {}> = <T extends object>(
  Component: React.ComponentType<T>
) => React.ComponentType<T & P>;

// Generic event handler types
export type EventHandler<T = any> = (event: T) => void;
export type AsyncEventHandler<T = any> = (event: T) => Promise<void>;
export type ValueChangeHandler<T = any> = (value: T) => void;

// Configuration types for production deployment
export interface ProductionConfig {
  apiBaseUrl: string;
  wsBaseUrl: string;
  enableAnalytics: boolean;
  enableErrorReporting: boolean;
  enablePerformanceMonitoring: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  featureFlags: Record<string, boolean>;
  monitoring: {
    prometheus?: {
      enabled: boolean;
      endpoint: string;
    };
    grafana?: {
      enabled: boolean;
      dashboardUrl: string;
    };
  };
  security: {
    enableCSP: boolean;
    trustedDomains: string[];
    apiKeyRequired: boolean;
  };
}