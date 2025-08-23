/**
 * Connection Status Component
 * Shows real-time API connection status with retry functionality
 */

import React from 'react';
import { Badge, Button, Tooltip, Space } from 'antd';
import { 
  WifiOutlined, 
  DisconnectOutlined, 
  ReloadOutlined, 
  ExclamationCircleOutlined 
} from '@ant-design/icons';
import { useConnectionState } from '../services/persistentApiClient';

interface ConnectionStatusProps {
  showDetails?: boolean;
  size?: 'small' | 'default' | 'large';
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ 
  showDetails = false, 
  size = 'default' 
}) => {
  const { 
    isConnected, 
    lastSuccessfulRequest, 
    consecutiveFailures, 
    lastError, 
    reconnect 
  } = useConnectionState();

  const [reconnecting, setReconnecting] = React.useState(false);

  const handleReconnect = async () => {
    setReconnecting(true);
    try {
      await reconnect();
    } catch (error) {
      console.error('Manual reconnection failed:', error);
    } finally {
      setReconnecting(false);
    }
  };

  const getBadgeStatus = () => {
    if (isConnected) return 'success';
    if (consecutiveFailures > 0) return 'warning';
    return 'error';
  };

  const getStatusText = () => {
    if (isConnected) return 'Connected';
    if (consecutiveFailures > 0) return `Retrying (${consecutiveFailures})`;
    return 'Disconnected';
  };

  const getTooltipContent = () => {
    if (isConnected && lastSuccessfulRequest) {
      return `Connected - Last successful request: ${lastSuccessfulRequest.toLocaleTimeString()}`;
    }
    if (lastError) {
      return `Connection error: ${lastError}`;
    }
    return 'API connection status unknown';
  };

  const StatusIcon = isConnected ? WifiOutlined : DisconnectOutlined;

  if (!showDetails) {
    return (
      <Tooltip title={getTooltipContent()}>
        <Badge status={getBadgeStatus()} text={
          <Space size="small">
            <StatusIcon />
            <span style={{ 
              color: isConnected ? '#52c41a' : '#ff4d4f',
              fontWeight: 500
            }}>
              {getStatusText()}
            </span>
          </Space>
        } />
      </Tooltip>
    );
  }

  return (
    <div style={{ 
      padding: '8px 12px',
      background: isConnected ? '#f6ffed' : '#fff2f0',
      border: `1px solid ${isConnected ? '#b7eb8f' : '#ffccc7'}`,
      borderRadius: '6px',
      fontSize: size === 'small' ? '12px' : '14px'
    }}>
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space>
          <Badge status={getBadgeStatus()} />
          <StatusIcon style={{ color: isConnected ? '#52c41a' : '#ff4d4f' }} />
          <span style={{ 
            fontWeight: 600,
            color: isConnected ? '#52c41a' : '#ff4d4f'
          }}>
            API Client {getStatusText()}
          </span>
          {!isConnected && (
            <Button
              type="link"
              size="small"
              icon={<ReloadOutlined spin={reconnecting} />}
              onClick={handleReconnect}
              disabled={reconnecting}
              style={{ padding: 0, height: 'auto' }}
            >
              Retry
            </Button>
          )}
        </Space>

        {lastSuccessfulRequest && (
          <div style={{ fontSize: '11px', color: '#8c8c8c' }}>
            Last successful: {lastSuccessfulRequest.toLocaleString()}
          </div>
        )}

        {!isConnected && lastError && (
          <div style={{ 
            fontSize: '11px', 
            color: '#ff4d4f',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '4px'
          }}>
            <ExclamationCircleOutlined />
            <span>{lastError}</span>
          </div>
        )}

        {consecutiveFailures > 0 && (
          <div style={{ fontSize: '11px', color: '#faad14' }}>
            {consecutiveFailures} consecutive failure{consecutiveFailures > 1 ? 's' : ''}
          </div>
        )}
      </Space>
    </div>
  );
};