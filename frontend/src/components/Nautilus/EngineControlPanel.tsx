/**
 * EngineControlPanel - Control panel for NautilusTrader engine operations
 * 
 * Provides start/stop/restart controls with safety confirmations and
 * emergency stop functionality as specified in Story 6.1.
 */

import React, { useState } from 'react';
import { Button, Space, Modal, Alert, Typography, Divider, Popconfirm } from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const { Text, Title } = Typography;

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
}

interface EngineControlPanelProps {
  status: EngineStatus;
  config: EngineConfig;
  onStart: (config: EngineConfig, confirmLive: boolean) => Promise<void>;
  onStop: (force: boolean) => Promise<void>;
  onRestart: () => Promise<void>;
  onEmergencyStop: () => Promise<void>;
  loading: boolean;
}

export const EngineControlPanel: React.FC<EngineControlPanelProps> = ({
  status,
  config,
  onStart,
  onStop,
  onRestart,
  onEmergencyStop,
  loading
}) => {
  const [liveTradingConfirmVisible, setLiveTradingConfirmVisible] = useState(false);
  const [stopConfirmVisible, setStopConfirmVisible] = useState(false);
  const [forceStop, setForceStop] = useState(false);

  const isRunning = status.state === 'running';
  const isStopped = status.state === 'stopped';
  const isTransitioning = status.state === 'starting' || status.state === 'stopping';
  const hasError = status.state === 'error';

  // Handle start engine with safety checks
  const handleStartEngine = () => {
    if (config.trading_mode === 'live') {
      setLiveTradingConfirmVisible(true);
    } else {
      onStart(config, false);
    }
  };

  // Handle live trading confirmation
  const handleLiveTradingConfirm = () => {
    setLiveTradingConfirmVisible(false);
    onStart(config, true);
  };

  // Handle stop engine with confirmation
  const handleStopEngine = (force = false) => {
    setForceStop(force);
    setStopConfirmVisible(true);
  };

  // Handle stop confirmation
  const handleStopConfirm = () => {
    setStopConfirmVisible(false);
    onStop(forceStop);
  };

  // Render live trading confirmation modal
  const renderLiveTradingModal = () => (
    <Modal
      title={
        <Space>
          <WarningOutlined style={{ color: '#ff4d4f' }} />
          <span style={{ color: '#ff4d4f' }}>⚠️ LIVE TRADING MODE CONFIRMATION</span>
        </Space>
      }
      open={liveTradingConfirmVisible}
      onOk={handleLiveTradingConfirm}
      onCancel={() => setLiveTradingConfirmVisible(false)}
      okText="START LIVE TRADING"
      okType="danger"
      cancelText="Cancel"
      width={600}
      maskClosable={false}
    >
      <Alert
        type="error"
        message="CRITICAL WARNING: Real Money Trading"
        description="You are about to start the engine in LIVE TRADING mode with real money."
        style={{ marginBottom: 16 }}
        showIcon
      />
      
      <div style={{ marginBottom: 16 }}>
        <Title level={5} style={{ color: '#ff4d4f' }}>⚠️ RISKS AND CONSEQUENCES:</Title>
        <ul style={{ paddingLeft: 20, marginBottom: 16 }}>
          <li><strong>Real money trading:</strong> All orders will use actual funds</li>
          <li><strong>Automatic execution:</strong> Strategies will execute trades automatically</li>
          <li><strong>Market risk:</strong> You may experience significant financial losses</li>
          <li><strong>No simulation:</strong> This is not a test environment</li>
          <li><strong>Immediate effect:</strong> Orders will be placed on live markets</li>
        </ul>
      </div>

      <Alert
        type="warning"
        message="Configuration Summary"
        description={
          <div>
            <Text strong>Instance:</Text> {config.instance_id}<br />
            <Text strong>Risk Engine:</Text> {config.risk_engine_enabled ? 'Enabled' : 'Disabled'}<br />
            {config.max_position_size && (
              <>
                <Text strong>Max Position:</Text> ${config.max_position_size.toLocaleString()}<br />
              </>
            )}
            {config.max_order_rate && (
              <>
                <Text strong>Max Order Rate:</Text> {config.max_order_rate}/min<br />
              </>
            )}
          </div>
        }
        style={{ marginBottom: 16 }}
      />

      <div style={{ 
        background: '#fff2f0',
        border: '2px solid #ff4d4f', 
        borderRadius: '6px',
        padding: '12px',
        textAlign: 'center'
      }}>
        <Text strong style={{ color: '#ff4d4f', fontSize: '16px' }}>
          ⚠️ ARE YOU ABSOLUTELY CERTAIN YOU WANT TO PROCEED WITH LIVE TRADING? ⚠️
        </Text>
      </div>
    </Modal>
  );

  // Render stop confirmation modal
  const renderStopModal = () => (
    <Modal
      title={
        <Space>
          <StopOutlined style={{ color: forceStop ? '#ff4d4f' : '#faad14' }} />
          {forceStop ? '🚨 FORCE STOP ENGINE' : 'Stop NautilusTrader Engine'}
        </Space>
      }
      open={stopConfirmVisible}
      onOk={handleStopConfirm}
      onCancel={() => setStopConfirmVisible(false)}
      okText={forceStop ? 'FORCE STOP' : 'Stop Engine'}
      okType={forceStop ? 'danger' : 'primary'}
      cancelText="Cancel"
      width={500}
    >
      <Alert
        type={forceStop ? 'error' : 'warning'}
        message={forceStop ? 'Force Stop Warning' : 'Engine Stop Confirmation'}
        description={
          forceStop 
            ? 'This will immediately terminate the engine and all running strategies. Any open positions will remain in the market and may need manual intervention.'
            : 'This will gracefully stop the engine and all running strategies. Open positions will be handled according to strategy settings.'
        }
        style={{ marginBottom: 16 }}
        showIcon
      />

      {status.mode === 'live' && (
        <Alert
          type="warning"
          message="Live Trading Active"
          description="The engine is currently running in live trading mode. Ensure you understand the impact of stopping on your active positions."
          style={{ marginBottom: 16 }}
          showIcon
        />
      )}
    </Modal>
  );

  return (
    <div className="engine-control-panel">
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Main Control Buttons */}
        <Space wrap>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={handleStartEngine}
            disabled={isRunning || isTransitioning || loading}
            loading={loading && status.state === 'starting'}
            style={{ minWidth: 120 }}
          >
            Start Engine
          </Button>

          <Button
            danger
            size="large"
            icon={<StopOutlined />}
            onClick={() => handleStopEngine(false)}
            disabled={isStopped || loading}
            loading={loading && status.state === 'stopping'}
            style={{ minWidth: 120 }}
          >
            Stop Engine
          </Button>

          <Button
            size="large"
            icon={<ReloadOutlined />}
            onClick={onRestart}
            disabled={isStopped || loading}
            loading={loading && status.state === 'stopping'}
            style={{ minWidth: 120 }}
          >
            Restart
          </Button>
        </Space>

        <Divider style={{ margin: '12px 0' }} />

        {/* Advanced Controls */}
        <Space wrap>
          <Button
            danger
            ghost
            icon={<StopOutlined />}
            onClick={() => handleStopEngine(true)}
            disabled={isStopped || loading}
            style={{ minWidth: 120 }}
          >
            Force Stop
          </Button>

          <Popconfirm
            title="Emergency Stop"
            description="This will immediately stop the engine. Are you sure?"
            onConfirm={onEmergencyStop}
            okText="Emergency Stop"
            okType="danger"
            cancelText="Cancel"
            icon={<ExclamationCircleOutlined style={{ color: 'red' }} />}
          >
            <Button
              danger
              type="primary"
              icon={<ThunderboltOutlined />}
              disabled={isStopped}
              style={{ 
                minWidth: 140,
                background: '#ff4d4f',
                borderColor: '#ff4d4f'
              }}
            >
              🚨 Emergency Stop
            </Button>
          </Popconfirm>
        </Space>

        {/* Status-based Messages */}
        {hasError && (
          <Alert
            type="error"
            message="Engine Error"
            description="The engine encountered an error. You may need to restart or check the configuration."
            showIcon
            style={{ marginTop: 12 }}
          />
        )}

        {isTransitioning && (
          <Alert
            type="info"
            message={`Engine ${status.state}...`}
            description="Please wait while the engine state changes. This may take a few moments."
            showIcon
            style={{ marginTop: 12 }}
          />
        )}

        {status.mode === 'live' && isRunning && (
          <Alert
            type="warning"
            message="Live Trading Active"
            description="The engine is currently running in live trading mode with real money. Monitor your positions carefully."
            showIcon
            style={{ marginTop: 12 }}
          />
        )}
      </Space>

      {renderLiveTradingModal()}
      {renderStopModal()}

      <style jsx>{`
        .engine-control-panel {
          width: 100%;
        }

        .ant-btn-danger {
          font-weight: 600;
        }

        .ant-btn-primary {
          font-weight: 600;
        }

        .emergency-stop-btn {
          background: linear-gradient(45deg, #ff4d4f, #ff7875);
          border: none;
          font-weight: bold;
          box-shadow: 0 2px 8px rgba(255, 77, 79, 0.3);
        }

        .emergency-stop-btn:hover {
          background: linear-gradient(45deg, #ff7875, #ffa39e);
          box-shadow: 0 4px 12px rgba(255, 77, 79, 0.4);
        }

        .ant-alert {
          border-radius: 6px;
        }
      `}</style>
    </div>
  );
};

export default EngineControlPanel;