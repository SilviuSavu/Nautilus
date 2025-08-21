/**
 * EngineStatusIndicator - Real-time status display for NautilusTrader engine
 * 
 * Provides visual indicators for engine state, uptime, and configuration
 * with color-coded status and animated indicators.
 */

import React from 'react';
import { Space, Typography, Badge, Statistic, Tag, Tooltip, Progress } from 'antd';
import { 
  CheckCircleOutlined, 
  StopOutlined, 
  LoadingOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  PlayCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface EngineConfig {
  engine_type: string;
  log_level: string;
  instance_id: string;
  trading_mode: 'paper' | 'live';
  max_memory: string;
  max_cpu: string;
  risk_engine_enabled: boolean;
  max_position_size?: number;
  max_order_rate?: number;
}

interface EngineStatus {
  state: 'stopped' | 'starting' | 'running' | 'stopping' | 'error';
  mode?: 'live' | 'paper' | 'backtest';
  started_at?: string;
  uptime_seconds?: number;
  last_error?: string;
  configuration?: EngineConfig;
  config?: EngineConfig;
}

interface EngineStatusIndicatorProps {
  status: EngineStatus;
  config: EngineConfig;
  showDetails?: boolean;
}

export const EngineStatusIndicator: React.FC<EngineStatusIndicatorProps> = ({
  status,
  config,
  showDetails = true
}) => {
  
  const getStatusColor = (state: string): 'success' | 'processing' | 'error' | 'warning' | 'default' => {
    switch (state) {
      case 'running': return 'success';
      case 'starting': case 'stopping': return 'processing';
      case 'error': return 'error';
      case 'stopped': return 'default';
      default: return 'default';
    }
  };

  const getStatusIcon = (state: string) => {
    switch (state) {
      case 'running': 
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'starting': 
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      case 'stopping': 
        return <LoadingOutlined style={{ color: '#faad14' }} />;
      case 'error': 
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'stopped': 
        return <StopOutlined style={{ color: '#8c8c8c' }} />;
      default: 
        return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getStatusText = (state: string): string => {
    switch (state) {
      case 'running': return 'Running';
      case 'starting': return 'Starting...';
      case 'stopping': return 'Stopping...';
      case 'error': return 'Error';
      case 'stopped': return 'Stopped';
      default: return 'Unknown';
    }
  };

  const getUptimeDisplay = (): string => {
    if (!status.uptime_seconds || status.state !== 'running') {
      return 'N/A';
    }

    const seconds = Math.floor(status.uptime_seconds);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      return `${remainingSeconds}s`;
    }
  };

  const getTradingModeTag = () => {
    const mode = status.mode || config.trading_mode || 'paper';
    
    if (mode === 'live') {
      return (
        <Tag color="red" icon={<ExclamationCircleOutlined />}>
          LIVE TRADING
        </Tag>
      );
    } else if (mode === 'paper') {
      return (
        <Tag color="green" icon={<PlayCircleOutlined />}>
          PAPER TRADING
        </Tag>
      );
    } else {
      return (
        <Tag color="blue" icon={<PlayCircleOutlined />}>
          {mode.toUpperCase()}
        </Tag>
      );
    }
  };

  const getStartedAtDisplay = (): string => {
    if (!status.started_at) return 'N/A';
    
    try {
      const startTime = new Date(status.started_at);
      return startTime.toLocaleString();
    } catch (e) {
      return 'Invalid date';
    }
  };

  const renderProgressIndicator = () => {
    if (status.state === 'starting') {
      return (
        <Progress
          type="line"
          status="active"
          showInfo={false}
          strokeColor="#1890ff"
          style={{ marginTop: 8 }}
        />
      );
    } else if (status.state === 'stopping') {
      return (
        <Progress
          type="line"
          status="active"
          showInfo={false}
          strokeColor="#faad14"
          style={{ marginTop: 8 }}
        />
      );
    }
    return null;
  };

  return (
    <div className="engine-status-indicator">
      {/* Main Status Display */}
      <Space direction="vertical" style={{ width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Space size="large">
            <Badge 
              status={getStatusColor(status.state)}
              text={
                <Space>
                  {getStatusIcon(status.state)}
                  <Text strong style={{ fontSize: '16px' }}>
                    {getStatusText(status.state)}
                  </Text>
                </Space>
              }
            />
            {getTradingModeTag()}
          </Space>
          
          <Space>
            {status.configuration?.risk_engine_enabled !== false && config.risk_engine_enabled && (
              <Tooltip title="Risk Engine Enabled">
                <Tag color="orange">RISK ENGINE</Tag>
              </Tooltip>
            )}
            
            <Tooltip title={`Instance: ${config.instance_id}`}>
              <Tag color="blue">{config.instance_id}</Tag>
            </Tooltip>
          </Space>
        </div>

        {renderProgressIndicator()}

        {/* Detailed Status Information */}
        {showDetails && (
          <div style={{ 
            background: '#fafafa', 
            padding: '12px', 
            borderRadius: '4px',
            border: '1px solid #d9d9d9'
          }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {/* Runtime Information */}
              {status.state === 'running' && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Statistic
                    title="Uptime"
                    value={getUptimeDisplay()}
                    prefix={<ClockCircleOutlined />}
                    valueStyle={{ fontSize: '14px' }}
                  />
                  <Statistic
                    title="Started At"
                    value={getStartedAtDisplay()}
                    valueStyle={{ fontSize: '12px' }}
                  />
                </div>
              )}

              {/* Configuration Summary */}
              <div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Engine Type: {config.engine_type} | 
                  Log Level: {config.log_level} | 
                  Memory: {config.max_memory} | 
                  CPU: {config.max_cpu}
                </Text>
              </div>

              {/* Risk Configuration */}
              {config.risk_engine_enabled && (
                <div>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    Risk Limits: 
                    {config.max_position_size && ` Max Position: $${config.max_position_size.toLocaleString()}`}
                    {config.max_order_rate && ` | Max Orders: ${config.max_order_rate}/min`}
                  </Text>
                </div>
              )}

              {/* Error Display */}
              {status.last_error && (
                <div style={{ 
                  background: '#fff2f0', 
                  border: '1px solid #ffccc7',
                  borderRadius: '4px',
                  padding: '8px'
                }}>
                  <Text type="danger" style={{ fontSize: '12px' }}>
                    <ExclamationCircleOutlined /> {status.last_error}
                  </Text>
                </div>
              )}
            </Space>
          </div>
        )}
      </Space>

      <style jsx>{`
        .engine-status-indicator {
          width: 100%;
        }

        .ant-statistic-title {
          font-size: 12px;
          margin-bottom: 4px;
        }

        .ant-statistic-content {
          font-size: 14px;
        }

        .ant-badge .ant-badge-status-text {
          font-weight: 500;
        }

        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }

        .status-processing .anticon {
          animation: pulse 1.5s infinite;
        }
      `}</style>
    </div>
  );
};

export default EngineStatusIndicator;