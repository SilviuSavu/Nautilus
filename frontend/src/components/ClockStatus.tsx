/**
 * Clock Status Visual Indicator Component
 * 
 * Features:
 * - Real-time clock synchronization status display
 * - Visual indicators for sync health, latency, and drift
 * - Interactive controls for manual sync and configuration
 * - Compact and expandable display modes
 * - Integration with trading dashboard and system monitoring
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  Card,
  Badge,
  Button,
  Tooltip,
  Space,
  Typography,
  Progress,
  Statistic,
  Popover,
  Switch,
  InputNumber,
  Divider,
  Alert
} from 'antd';
import {
  ClockCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  WifiOutlined
} from '@ant-design/icons';
import { useClockSync } from '../hooks/useClockSync';
import { useServerTime } from '../hooks/useServerTime';

const { Text } = Typography;

export interface ClockStatusProps {
  size?: 'small' | 'default' | 'large';
  mode?: 'compact' | 'detailed' | 'minimal';
  showControls?: boolean;
  showMetrics?: boolean;
  autoSync?: boolean;
  onStatusChange?: (status: ClockSyncStatus) => void;
}

export interface ClockSyncStatus {
  isHealthy: boolean;
  accuracy: number;
  latency: number;
  drift: number;
  lastSync: number;
}

export const ClockStatus: React.FC<ClockStatusProps> = ({
  size = 'default',
  mode = 'compact',
  showControls = true,
  showMetrics = true,
  autoSync = true,
  onStatusChange
}) => {
  const { 
    clockState, 
    forceSync, 
    isClockSynced, 
    getClockAccuracy 
  } = useClockSync({
    syncInterval: 30000,
    enableDriftCorrection: true,
    retryInterval: 5000,
    maxRetries: 3
  });

  const { 
    serverTimeState, 
    formatServerTime 
  } = useServerTime({
    updateInterval: 1000,
    timeFormat: 'trading'
  });

  const [isSettingsVisible, setIsSettingsVisible] = useState(false);
  const [syncInterval, setSyncInterval] = useState(30);
  const [enableDriftCorrection, setEnableDriftCorrection] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);

  // Calculate comprehensive clock status
  const clockStatus = useMemo(() => {
    const accuracy = getClockAccuracy();
    const isHealthy = isClockSynced && accuracy >= 90;
    const latency = clockState.networkLatency;
    const drift = Math.abs(clockState.clockDrift);
    const lastSync = clockState.lastSyncTimestamp;

    const status: ClockSyncStatus = {
      isHealthy,
      accuracy,
      latency,
      drift,
      lastSync
    };

    // Notify parent component of status changes
    if (onStatusChange) {
      onStatusChange(status);
    }

    return status;
  }, [isClockSynced, getClockAccuracy, clockState, onStatusChange]);

  // Determine status color and icon
  const getStatusIndicator = useCallback(() => {
    if (!isClockSynced || clockState.syncStatus === 'error') {
      return {
        status: 'error' as const,
        color: '#ff4d4f',
        icon: <WarningOutlined style={{ color: '#ff4d4f' }} />,
        text: 'Clock Error'
      };
    }

    if (clockState.syncStatus === 'syncing') {
      return {
        status: 'processing' as const,
        color: '#1890ff',
        icon: <SyncOutlined spin style={{ color: '#1890ff' }} />,
        text: 'Syncing...'
      };
    }

    const accuracy = getClockAccuracy();
    if (accuracy >= 95) {
      return {
        status: 'success' as const,
        color: '#52c41a',
        icon: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
        text: 'Excellent'
      };
    } else if (accuracy >= 80) {
      return {
        status: 'warning' as const,
        color: '#faad14',
        icon: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
        text: 'Good'
      };
    } else {
      return {
        status: 'error' as const,
        color: '#ff4d4f',
        icon: <WarningOutlined style={{ color: '#ff4d4f' }} />,
        text: 'Poor'
      };
    }
  }, [isClockSynced, clockState.syncStatus, getClockAccuracy]);

  // Handle manual sync
  const handleForceSync = useCallback(async () => {
    setIsSyncing(true);
    try {
      await forceSync();
    } catch (error) {
      console.error('Manual sync failed:', error);
    } finally {
      setIsSyncing(false);
    }
  }, [forceSync]);

  // Format time since last sync
  const getTimeSinceLastSync = useCallback(() => {
    if (clockState.lastSyncTimestamp === 0) {
      return 'Never';
    }

    const secondsAgo = Math.floor((Date.now() - clockState.lastSyncTimestamp) / 1000);
    if (secondsAgo < 60) {
      return `${secondsAgo}s ago`;
    } else if (secondsAgo < 3600) {
      return `${Math.floor(secondsAgo / 60)}m ago`;
    } else {
      return `${Math.floor(secondsAgo / 3600)}h ago`;
    }
  }, [clockState.lastSyncTimestamp]);

  // Settings panel content
  const settingsContent = (
    <div style={{ width: 300, padding: '8px 0' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Text strong>Clock Settings</Text>
        
        <Space>
          <Text>Sync Interval:</Text>
          <InputNumber
            value={syncInterval}
            onChange={(value) => setSyncInterval(value || 30)}
            min={10}
            max={300}
            suffix="s"
            size="small"
          />
        </Space>
        
        <Space>
          <Text>Drift Correction:</Text>
          <Switch
            checked={enableDriftCorrection}
            onChange={setEnableDriftCorrection}
            size="small"
          />
        </Space>
        
        <Divider style={{ margin: '8px 0' }} />
        
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text strong>Diagnostics</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Server Time: {formatServerTime('trading')}
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Time Offset: {clockState.serverTimeOffset.toFixed(2)}ms
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Sync Count: {clockState.syncCount} / Errors: {clockState.errorCount}
          </Text>
        </Space>
        
        <Button 
          type="primary" 
          size="small" 
          block 
          onClick={handleForceSync}
          loading={isSyncing}
          icon={<ReloadOutlined />}
        >
          Force Sync Now
        </Button>
      </Space>
    </div>
  );

  const statusIndicator = getStatusIndicator();

  // Render based on mode
  if (mode === 'minimal') {
    return (
      <Badge 
        status={statusIndicator.status} 
        dot 
        title={`Clock Status: ${statusIndicator.text} (${getClockAccuracy().toFixed(1)}%)`}
      />
    );
  }

  if (mode === 'compact') {
    return (
      <Space size="small">
        <Tooltip title={`Clock Status: ${statusIndicator.text}`}>
          <Badge status={statusIndicator.status} />
        </Tooltip>
        
        <Text style={{ fontSize: size === 'small' ? '12px' : '14px' }}>
          {formatServerTime('trading')}
        </Text>
        
        {showControls && (
          <Popover
            content={settingsContent}
            title="Clock Settings"
            trigger="click"
            open={isSettingsVisible}
            onOpenChange={setIsSettingsVisible}
            placement="bottomRight"
          >
            <Button
              type="text"
              icon={<SettingOutlined />}
              size="small"
              style={{ padding: '2px' }}
            />
          </Popover>
        )}
      </Space>
    );
  }

  // Detailed mode
  return (
    <Card 
      size={size}
      title={
        <Space>
          <ClockCircleOutlined />
          <Text>Clock Status</Text>
          <Badge status={statusIndicator.status} text={statusIndicator.text} />
        </Space>
      }
      extra={
        showControls && (
          <Space>
            <Button
              type="text"
              icon={<ReloadOutlined />}
              onClick={handleForceSync}
              loading={isSyncing}
              size="small"
              title="Force Sync"
            />
            <Popover
              content={settingsContent}
              title="Clock Settings"
              trigger="click"
              open={isSettingsVisible}
              onOpenChange={setIsSettingsVisible}
              placement="bottomRight"
            >
              <Button
                type="text"
                icon={<SettingOutlined />}
                size="small"
                title="Settings"
              />
            </Popover>
          </Space>
        )
      }
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Current Time Display */}
        <div style={{ textAlign: 'center', marginBottom: '16px' }}>
          <Text strong style={{ fontSize: '18px' }}>
            {formatServerTime('trading')}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Server Time (Synchronized)
          </Text>
        </div>

        {showMetrics && (
          <>
            {/* Accuracy Progress */}
            <div>
              <Text>Clock Accuracy</Text>
              <Progress
                percent={getClockAccuracy()}
                size="small"
                strokeColor={
                  getClockAccuracy() >= 95 ? '#52c41a' :
                  getClockAccuracy() >= 80 ? '#faad14' : '#ff4d4f'
                }
                format={(percent) => `${(percent || 0).toFixed(1)}%`}
              />
            </div>

            {/* Metrics Grid */}
            <div>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Space split={<Divider type="vertical" />} wrap>
                  <Statistic
                    title="Latency"
                    value={clockState.networkLatency}
                    suffix="ms"
                    valueStyle={{
                      fontSize: '14px',
                      color: clockState.networkLatency < 50 ? '#52c41a' : 
                             clockState.networkLatency < 100 ? '#faad14' : '#ff4d4f'
                    }}
                  />
                  <Statistic
                    title="Drift"
                    value={Math.abs(clockState.clockDrift)}
                    precision={2}
                    suffix="ms/s"
                    valueStyle={{
                      fontSize: '14px',
                      color: Math.abs(clockState.clockDrift) < 1 ? '#52c41a' : '#faad14'
                    }}
                  />
                </Space>

                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Last sync: {getTimeSinceLastSync()}
                </Text>
              </Space>
            </div>
          </>
        )}

        {/* Warnings */}
        {clockState.networkLatency > 100 && (
          <Alert
            message="High Network Latency"
            description="Clock synchronization may be affected by network delays."
            type="warning"
            size="small"
            showIcon
          />
        )}

        {Math.abs(clockState.clockDrift) > 5 && (
          <Alert
            message="High Clock Drift"
            description="System clock is drifting significantly from server time."
            type="warning"
            size="small"
            showIcon
          />
        )}

        {clockState.errorCount > 0 && (
          <Alert
            message="Sync Errors Detected"
            description={`${clockState.errorCount} synchronization errors occurred.`}
            type="warning"
            size="small"
            showIcon
          />
        )}
      </Space>
    </Card>
  );
};

export default ClockStatus;