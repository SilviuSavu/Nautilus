import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card,
  Row,
  Col,
  Progress,
  Timeline,
  Alert,
  Button,
  Space,
  Typography,
  Tag,
  Tabs,
  List,
  Spin,
  Badge,
  Tooltip,
  Switch,
  Statistic,
  Divider,
  notification
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  LineChartOutlined,
  BugOutlined,
  RobotOutlined,
  CloudUploadOutlined,
  DeploymentUnitOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  FullscreenExitOutlined
} from '@ant-design/icons';
import { Area, Line } from '@ant-design/charts';
import type {
  PipelineStatusMonitorProps,
  AdvancedDeploymentPipeline,
  DeploymentPipelineStage,
  PipelineStatusResponse
} from './types/deploymentTypes';

const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

interface LogEntry {
  timestamp: Date;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  stage?: string;
  metadata?: Record<string, any>;
}

interface MetricData {
  timestamp: string;
  value: number;
  metric: string;
}

const PipelineStatusMonitor: React.FC<PipelineStatusMonitorProps> = ({
  pipelineId,
  autoRefresh = true,
  showLogs = true,
  onStatusChange
}) => {
  const [pipeline, setPipeline] = useState<AdvancedDeploymentPipeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const intervalRef = useRef<NodeJS.Timeout>();
  const wsRef = useRef<WebSocket>();

  const fetchPipelineStatus = useCallback(async () => {
    if (!pipelineId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/pipeline/${pipelineId}/status`);
      if (!response.ok) throw new Error('Failed to fetch pipeline status');
      
      const data: PipelineStatusResponse = await response.json();
      
      setPipeline(data.pipeline);
      setLastUpdate(new Date());
      
      if (data.logs && showLogs) {
        const logEntries: LogEntry[] = data.logs.map(log => ({
          timestamp: new Date(),
          level: log.includes('ERROR') ? 'error' : 
                 log.includes('WARN') ? 'warn' :
                 log.includes('DEBUG') ? 'debug' : 'info',
          message: log,
          stage: extractStageFromLog(log)
        }));
        setLogs(prev => [...logEntries, ...prev].slice(0, 1000)); // Keep last 1000 logs
      }
      
      onStatusChange?.(data.pipeline.status);
    } catch (error) {
      console.error('Error fetching pipeline status:', error);
      notification.error({
        message: 'Failed to fetch pipeline status',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  }, [pipelineId, showLogs, onStatusChange]);

  const extractStageFromLog = (log: string): string | undefined => {
    const stageMatch = log.match(/\[(\w+)\]/);
    return stageMatch ? stageMatch[1] : undefined;
  };

  const connectWebSocket = useCallback(() => {
    if (!pipelineId) return;
    
    setConnectionStatus('connecting');
    
    const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/pipeline/${pipelineId}/status`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('Pipeline WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'pipeline_update') {
          setPipeline(data.pipeline);
          setLastUpdate(new Date());
          onStatusChange?.(data.pipeline.status);
        } else if (data.type === 'log_entry') {
          const logEntry: LogEntry = {
            timestamp: new Date(data.timestamp),
            level: data.level,
            message: data.message,
            stage: data.stage
          };
          setLogs(prev => [logEntry, ...prev].slice(0, 1000));
        } else if (data.type === 'metrics') {
          setMetrics(prev => [...prev, ...data.metrics].slice(-100)); // Keep last 100 metrics
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('Pipeline WebSocket disconnected');
      
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (autoRefresh) {
          connectWebSocket();
        }
      }, 5000);
    };
    
    ws.onerror = (error) => {
      console.error('Pipeline WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
    
    wsRef.current = ws;
  }, [pipelineId, autoRefresh, onStatusChange]);

  useEffect(() => {
    fetchPipelineStatus();
    
    if (autoRefresh) {
      // Try WebSocket first, fallback to polling
      connectWebSocket();
      
      intervalRef.current = setInterval(fetchPipelineStatus, refreshInterval);
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [fetchPipelineStatus, connectWebSocket, autoRefresh, refreshInterval]);

  const getStageIcon = (stage: DeploymentPipelineStage) => {
    const iconProps = { style: { fontSize: '16px' } };
    
    switch (stage.type) {
      case 'validation': return <BugOutlined {...iconProps} />;
      case 'backtesting': return <LineChartOutlined {...iconProps} />;
      case 'paper_trading': return <RobotOutlined {...iconProps} />;
      case 'staging': return <CloudUploadOutlined {...iconProps} />;
      case 'production': return <DeploymentUnitOutlined {...iconProps} />;
      default: return <ClockCircleOutlined {...iconProps} />;
    }
  };

  const getStageColor = (status: string) => {
    switch (status) {
      case 'completed': return 'green';
      case 'running': return 'blue';
      case 'failed': return 'red';
      case 'pending': return 'gray';
      case 'skipped': return 'orange';
      default: return 'gray';
    }
  };

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'error': return '#ff4d4f';
      case 'warn': return '#faad14';
      case 'info': return '#1890ff';
      case 'debug': return '#8c8c8c';
      default: return '#000000';
    }
  };

  const renderProgressIndicator = () => {
    if (!pipeline) return null;
    
    const { progress } = pipeline;
    const isRunning = pipeline.status === 'running';
    
    return (
      <Card title="Pipeline Progress" className="mb-4">
        <Row gutter={16}>
          <Col span={12}>
            <Progress
              type="circle"
              percent={progress.overall_progress}
              format={() => `${progress.overall_progress}%`}
              status={pipeline.status === 'failed' ? 'exception' : isRunning ? 'active' : 'normal'}
              strokeWidth={8}
              width={120}
            />
          </Col>
          <Col span={12}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="Stages Completed"
                value={progress.stages_completed}
                suffix={`/ ${progress.stages_total}`}
                prefix={<CheckCircleOutlined />}
              />
              <Statistic
                title="Elapsed Time"
                value={progress.elapsed_minutes}
                suffix="minutes"
                prefix={<ClockCircleOutlined />}
              />
              {progress.estimated_completion && (
                <Statistic
                  title="Estimated Completion"
                  value={new Date(progress.estimated_completion).toLocaleTimeString()}
                  prefix={<ClockCircleOutlined />}
                />
              )}
            </Space>
          </Col>
        </Row>
        
        <Divider />
        
        <Progress
          percent={progress.overall_progress}
          status={isRunning ? 'active' : pipeline.status === 'failed' ? 'exception' : 'normal'}
          showInfo={false}
        />
        
        <div className="mt-2">
          <Text type="secondary">
            Current Stage: {pipeline.current_stage || 'None'} • 
            Stage Progress: {progress.current_stage_progress}%
          </Text>
        </div>
      </Card>
    );
  };

  const renderStageTimeline = () => {
    if (!pipeline?.stages) return null;
    
    return (
      <Card title="Stage Timeline" className="mb-4">
        <Timeline mode="left">
          {pipeline.stages.map((stage, index) => (
            <Timeline.Item
              key={stage.id}
              dot={getStageIcon(stage)}
              color={getStageColor(stage.status)}
            >
              <div>
                <Title level={5}>
                  {stage.name}
                  <Tag color={getStageColor(stage.status)} className="ml-2">
                    {stage.status.toUpperCase()}
                  </Tag>
                  {stage.required_approvals && stage.required_approvals.length > 0 && (
                    <Badge count={stage.required_approvals.length} showZero color="blue" />
                  )}
                </Title>
                
                <Paragraph type="secondary">
                  Type: {stage.type.replace('_', ' ')} •
                  Auto Advance: {stage.auto_advance ? 'Yes' : 'No'}
                </Paragraph>
                
                {stage.started_at && (
                  <Paragraph type="secondary">
                    Started: {new Date(stage.started_at).toLocaleString()}
                  </Paragraph>
                )}
                
                {stage.completed_at && (
                  <Paragraph type="secondary">
                    Completed: {new Date(stage.completed_at).toLocaleString()}
                    {stage.duration_ms && (
                      <span> • Duration: {Math.round(stage.duration_ms / 1000 / 60)}min</span>
                    )}
                  </Paragraph>
                )}
                
                {stage.error_message && (
                  <Alert
                    message={stage.error_message}
                    type="error"
                    size="small"
                    className="mt-2"
                  />
                )}
                
                {stage.success_criteria && (
                  <div className="mt-2">
                    <Text strong>Success Criteria:</Text>
                    <ul className="mt-1">
                      {stage.success_criteria.min_trades && (
                        <li>Min Trades: {stage.success_criteria.min_trades}</li>
                      )}
                      {stage.success_criteria.max_drawdown && (
                        <li>Max Drawdown: {(stage.success_criteria.max_drawdown * 100).toFixed(1)}%</li>
                      )}
                      {stage.success_criteria.min_sharpe_ratio && (
                        <li>Min Sharpe Ratio: {stage.success_criteria.min_sharpe_ratio}</li>
                      )}
                      {stage.success_criteria.validation_checks && (
                        <li>Validation Checks: {stage.success_criteria.validation_checks.length} checks</li>
                      )}
                    </ul>
                  </div>
                )}
              </div>
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    );
  };

  const renderLogs = () => {
    if (!showLogs || logs.length === 0) return null;
    
    return (
      <Card 
        title="Real-time Logs" 
        extra={
          <Badge 
            count={logs.length} 
            showZero 
            overflowCount={999}
            style={{ backgroundColor: '#52c41a' }}
          />
        }
      >
        <List
          size="small"
          dataSource={logs.slice(0, 50)} // Show last 50 logs
          renderItem={(log, index) => (
            <List.Item key={index}>
              <div style={{ width: '100%', fontFamily: 'monospace', fontSize: '12px' }}>
                <Space>
                  <Text type="secondary">
                    {log.timestamp.toLocaleTimeString()}
                  </Text>
                  <Tag 
                    color={log.level === 'error' ? 'red' : 
                           log.level === 'warn' ? 'orange' :
                           log.level === 'debug' ? 'gray' : 'blue'}
                    style={{ minWidth: '50px', textAlign: 'center' }}
                  >
                    {log.level.toUpperCase()}
                  </Tag>
                  {log.stage && (
                    <Tag>{log.stage}</Tag>
                  )}
                </Space>
                <div style={{ marginTop: '4px' }}>
                  <Text style={{ color: getLogLevelColor(log.level) }}>
                    {log.message}
                  </Text>
                </div>
              </div>
            </List.Item>
          )}
          style={{ maxHeight: '400px', overflow: 'auto' }}
        />
      </Card>
    );
  };

  const renderMetricsChart = () => {
    if (metrics.length === 0) return null;
    
    const chartData = metrics.map(m => ({
      timestamp: m.timestamp,
      value: m.value,
      metric: m.metric
    }));
    
    const config = {
      data: chartData,
      xField: 'timestamp',
      yField: 'value',
      seriesField: 'metric',
      height: 300,
      smooth: true,
      animation: {
        appear: {
          animation: 'path-in',
          duration: 1000,
        },
      },
    };
    
    return (
      <Card title="Performance Metrics">
        <Line {...config} />
      </Card>
    );
  };

  if (!pipeline) {
    return (
      <div className="flex justify-center items-center h-64">
        <Spin size="large" />
        <Text className="ml-4">Loading pipeline status...</Text>
      </div>
    );
  }

  return (
    <div className={`pipeline-status-monitor ${isFullscreen ? 'fullscreen' : ''}`}>
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <DeploymentUnitOutlined />
              <span>Pipeline Monitor: {pipelineId}</span>
              <Badge 
                status={connectionStatus === 'connected' ? 'success' : 
                       connectionStatus === 'connecting' ? 'processing' : 'error'} 
                text={connectionStatus}
              />
            </div>
            <Space>
              {lastUpdate && (
                <Text type="secondary">
                  Last update: {lastUpdate.toLocaleTimeString()}
                </Text>
              )}
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="Auto"
                unCheckedChildren="Manual"
              />
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchPipelineStatus}
                loading={loading}
              />
              <Button
                icon={isFullscreen ? <FullscreenExitOutlined /> : <FullscreenOutlined />}
                onClick={() => setIsFullscreen(!isFullscreen)}
              />
            </Space>
          </div>
        }
      >
        <Row gutter={16}>
          <Col span={showLogs ? 16 : 24}>
            {renderProgressIndicator()}
            {renderStageTimeline()}
            {renderMetricsChart()}
          </Col>
          {showLogs && (
            <Col span={8}>
              {renderLogs()}
            </Col>
          )}
        </Row>
      </Card>
    </div>
  );
};

export default PipelineStatusMonitor;