/**
 * Enhanced Loading State Component
 * Production-ready loading states with accessibility and customization
 */

import React, { memo } from 'react';
import { Spin, Typography, Space, Card, Progress } from 'antd';
import {
  LoadingOutlined,
  SyncOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { UI_CONSTANTS, ACCESSIBILITY } from '../../constants/ui';

const { Text } = Typography;

export interface LoadingStateProps {
  /** Loading message to display */
  message?: string;
  /** Size of the loading indicator */
  size?: 'small' | 'default' | 'large';
  /** Type of loading animation */
  type?: 'spin' | 'sync' | 'clock';
  /** Show progress bar */
  showProgress?: boolean;
  /** Progress percentage (0-100) */
  progress?: number;
  /** Minimum height of the loading container */
  minHeight?: number | string;
  /** Whether to show as a card */
  asCard?: boolean;
  /** Custom loading indicator */
  indicator?: React.ReactNode;
  /** Additional CSS class */
  className?: string;
  /** Inline styling */
  style?: React.CSSProperties;
  /** Center the loading state */
  centered?: boolean;
  /** Show estimated time remaining */
  estimatedTime?: string;
  /** Loading state variant */
  variant?: 'default' | 'minimal' | 'detailed';
}

const LoadingState: React.FC<LoadingStateProps> = memo(({
  message = 'Loading...',
  size = 'default',
  type = 'spin',
  showProgress = false,
  progress = 0,
  minHeight = 200,
  asCard = true,
  indicator,
  className,
  style,
  centered = true,
  estimatedTime,
  variant = 'default'
}) => {
  const getLoadingIcon = () => {
    if (indicator) return indicator;
    
    const iconProps = {
      style: { 
        fontSize: size === 'large' ? 24 : size === 'small' ? 14 : 18,
        color: UI_CONSTANTS.CHART_COLORS.PRIMARY
      }
    };

    switch (type) {
      case 'sync':
        return <SyncOutlined spin {...iconProps} />;
      case 'clock':
        return <ClockCircleOutlined {...iconProps} />;
      default:
        return <LoadingOutlined {...iconProps} />;
    }
  };

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: centered ? 'center' : 'flex-start',
    justifyContent: centered ? 'center' : 'flex-start',
    minHeight,
    padding: variant === 'minimal' ? '12px' : '24px',
    ...style
  };

  const renderLoadingContent = () => {
    if (variant === 'minimal') {
      return (
        <Space size="small">
          <Spin 
            indicator={getLoadingIcon()}
            size={size}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {message}
          </Text>
        </Space>
      );
    }

    return (
      <Space direction="vertical" align="center" size="middle">
        <Spin 
          indicator={getLoadingIcon()}
          size={size}
          tip={variant === 'detailed' ? undefined : message}
        />
        
        {variant === 'detailed' && (
          <div style={{ textAlign: 'center' }}>
            <Text strong style={{ fontSize: 16 }}>
              {message}
            </Text>
            
            {estimatedTime && (
              <div style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Estimated time: {estimatedTime}
                </Text>
              </div>
            )}
          </div>
        )}

        {showProgress && (
          <div style={{ width: '100%', maxWidth: 300 }}>
            <Progress 
              percent={progress}
              size="small"
              status={progress === 100 ? 'success' : 'active'}
              strokeColor={UI_CONSTANTS.CHART_COLORS.PRIMARY}
              format={percent => `${percent}%`}
            />
          </div>
        )}
      </Space>
    );
  };

  const content = (
    <div 
      className={className}
      style={containerStyle}
      role="status"
      aria-live="polite"
      aria-label={ACCESSIBILITY.LABELS.LOADING}
    >
      {renderLoadingContent()}
    </div>
  );

  if (asCard && variant !== 'minimal') {
    return (
      <Card 
        style={{ 
          border: 'none',
          boxShadow: 'none',
          background: 'transparent'
        }}
      >
        {content}
      </Card>
    );
  }

  return content;
});

LoadingState.displayName = 'LoadingState';

export default LoadingState;