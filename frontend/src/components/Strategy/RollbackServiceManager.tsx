import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Alert,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Timeline,
  Progress,
  Table,
  Tag,
  Switch,
  Select,
  InputNumber,
  Form,
  Modal,
  List,
  Badge,
  Tooltip,
  notification,
  Spin,
  Divider
} from 'antd';
import {
  RollbackOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  WarningOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  LineChartOutlined,
  AlertOutlined,
  FireOutlined
} from '@ant-design/icons';
import { Line, Area } from '@ant-design/charts';
import type { ColumnType } from 'antd/es/table';
import type {
  RollbackServiceManagerProps,
  MLRollbackService,
  RollbackEvent,
  MonitoredMetric,
  MLRollbackPredictionRequest,
  MLRollbackPredictionResponse,
  RollbackTrigger,
  RollbackCondition
} from './types/deploymentTypes';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

interface PredictionData {
  timestamp: string;
  confidence: number;
  prediction: string;
  risk_score: number;
}

const RollbackServiceManager: React.FC<RollbackServiceManagerProps> = ({
  strategyId,
  deploymentId,
  enableMLPredictions = true,
  onRollbackTriggered
}) => {
  const [form] = Form.useForm();
  const [mlService, setMlService] = useState<MLRollbackService | null>(null);
  const [rollbackHistory, setRollbackHistory] = useState<RollbackEvent[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<MLRollbackPredictionResponse | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [autoMonitoring, setAutoMonitoring] = useState(false);
  const [triggerThresholds, setTriggerThresholds] = useState<RollbackTrigger[]>([]);

  // Default ML service configuration
  const defaultMLService: MLRollbackService = {
    model_id: 'rollback_predictor_v2',
    model_version: '2.1.0',
    enabled: enableMLPredictions,
    confidence_threshold: 0.75,
    prediction_window_minutes: 30,
    monitored_metrics: [
      { name: 'pnl_1h', weight: 0.3, threshold_type: 'absolute', threshold_value: -500 },
      { name: 'drawdown_percent', weight: 0.25, threshold_type: 'absolute', threshold_value: 0.05 },
      { name: 'sharpe_ratio', weight: 0.2, threshold_type: 'absolute', threshold_value: 0.5 },
      { name: 'win_rate', weight: 0.15, threshold_type: 'absolute', threshold_value: 0.4 },
      { name: 'position_count', weight: 0.1, threshold_type: 'anomaly', anomaly_sensitivity: 2.0 }
    ],
    rollback_history: []
  };

  // Default rollback triggers
  const defaultTriggers: RollbackTrigger[] = [
    {
      id: 'critical_loss',
      type: 'performance_based',
      enabled: true,
      conditions: [
        { metric: 'pnl_daily', operator: 'lt', value: -2000, window_minutes: 60 },
        { metric: 'drawdown_percent', operator: 'gt', value: 0.10, window_minutes: 30 }
      ],
      action: 'rollback'
    },
    {
      id: 'ml_prediction',
      type: 'ml_prediction',
      enabled: enableMLPredictions,
      conditions: [
        { metric: 'prediction_confidence', operator: 'gt', value: 0.85, window_minutes: 15 },
        { metric: 'risk_score', operator: 'gt', value: 0.8, window_minutes: 15 }
      ],
      action: 'alert'
    },
    {
      id: 'rapid_loss',
      type: 'performance_based',
      enabled: true,
      conditions: [
        { metric: 'pnl_5min', operator: 'lt', value: -500, window_minutes: 5 },
        { metric: 'consecutive_losses', operator: 'gt', value: 5, window_minutes: 15 }
      ],
      action: 'pause'
    }
  ];

  useEffect(() => {
    if (!mlService) {
      setMlService(defaultMLService);
      setTriggerThresholds(defaultTriggers);
    }
    
    loadRollbackHistory();
    loadCurrentMetrics();
    
    // Initialize form
    form.setFieldsValue({
      enabled: defaultMLService.enabled,
      confidence_threshold: defaultMLService.confidence_threshold,
      prediction_window: defaultMLService.prediction_window_minutes,
      auto_monitoring: false
    });
  }, [mlService, form, strategyId, deploymentId]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (autoMonitoring && mlService?.enabled) {
      intervalId = setInterval(() => {
        performMLPrediction();
        checkTriggers();
      }, 30000); // Check every 30 seconds
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoMonitoring, mlService]);

  const loadRollbackHistory = useCallback(async () => {
    if (!strategyId) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/rollback/${strategyId}/history`);
      if (response.ok) {
        const data = await response.json();
        setRollbackHistory(data.rollback_events || []);
      }
    } catch (error) {
      console.error('Error loading rollback history:', error);
    }
  }, [strategyId]);

  const loadCurrentMetrics = useCallback(async () => {
    if (!strategyId) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/${strategyId}/metrics/current`);
      if (response.ok) {
        const metrics = await response.json();
        setCurrentMetrics(metrics);
      }
    } catch (error) {
      console.error('Error loading current metrics:', error);
    }
  }, [strategyId]);

  const performMLPrediction = useCallback(async () => {
    if (!mlService?.enabled || !strategyId) return;
    
    setPredicting(true);
    try {
      const request: MLRollbackPredictionRequest = {
        strategy_id: strategyId,
        current_metrics: currentMetrics,
        time_horizon_minutes: mlService.prediction_window_minutes
      };
      
      const response = await fetch(`${API_BASE}/api/v1/strategies/rollback/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (response.ok) {
        const prediction: MLRollbackPredictionResponse = await response.json();
        setLastPrediction(prediction);
        
        // Add to prediction data for charts
        const newDataPoint: PredictionData = {
          timestamp: new Date().toISOString(),
          confidence: prediction.confidence_score,
          prediction: prediction.prediction,
          risk_score: prediction.risk_factors.length * 0.2 // Simple risk scoring
        };
        
        setPredictionData(prev => [...prev.slice(-50), newDataPoint]); // Keep last 50 points
        
        // Check if prediction warrants action
        if (prediction.prediction === 'critical' && prediction.confidence_score >= mlService.confidence_threshold) {
          handleMLRollbackTrigger(prediction);
        }
      }
    } catch (error) {
      console.error('Error performing ML prediction:', error);
    } finally {
      setPredicting(false);
    }
  }, [mlService, strategyId, currentMetrics]);

  const handleMLRollbackTrigger = async (prediction: MLRollbackPredictionResponse) => {
    const rollbackEvent: RollbackEvent = {
      event_id: `ml_${Date.now()}`,
      timestamp: new Date(),
      trigger_type: 'ml_prediction',
      confidence_score: prediction.confidence_score,
      metrics_snapshot: currentMetrics,
      rollback_executed: false,
      outcome: 'successful'
    };
    
    // Show confirmation modal for critical predictions
    Modal.confirm({
      title: 'ML Rollback Prediction Alert',
      content: (
        <div>
          <Alert
            message="Critical Performance Degradation Predicted"
            description={`Confidence: ${(prediction.confidence_score * 100).toFixed(1)}%`}
            type="error"
            showIcon
            className="mb-4"
          />
          <div>
            <Text strong>Risk Factors:</Text>
            <ul>
              {prediction.risk_factors.map((factor, index) => (
                <li key={index}>{factor}</li>
              ))}
            </ul>
          </div>
          <div className="mt-4">
            <Text strong>Recommendations:</Text>
            <ul>
              {prediction.recommendations.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>
          {prediction.time_to_rollback_minutes && (
            <Alert
              message={`Recommended rollback within ${prediction.time_to_rollback_minutes} minutes`}
              type="warning"
              className="mt-4"
            />
          )}
        </div>
      ),
      okText: 'Execute Rollback',
      cancelText: 'Monitor Only',
      onOk: () => executeRollback(rollbackEvent),
      onCancel: () => {
        setRollbackHistory(prev => [rollbackEvent, ...prev]);
        onRollbackTriggered?.('ml_prediction');
      }
    });
  };

  const checkTriggers = useCallback(async () => {
    for (const trigger of triggerThresholds) {
      if (!trigger.enabled) continue;
      
      let conditionsMet = true;
      
      for (const condition of trigger.conditions) {
        const metricValue = currentMetrics[condition.metric];
        if (metricValue === undefined) continue;
        
        let satisfied = false;
        switch (condition.operator) {
          case 'gt': satisfied = metricValue > condition.value; break;
          case 'lt': satisfied = metricValue < condition.value; break;
          case 'gte': satisfied = metricValue >= condition.value; break;
          case 'lte': satisfied = metricValue <= condition.value; break;
          case 'eq': satisfied = metricValue === condition.value; break;
          case 'neq': satisfied = metricValue !== condition.value; break;
        }
        
        if (!satisfied) {
          conditionsMet = false;
          break;
        }
      }
      
      if (conditionsMet) {
        handleTriggerActivation(trigger);
      }
    }
  }, [triggerThresholds, currentMetrics]);

  const handleTriggerActivation = async (trigger: RollbackTrigger) => {
    const rollbackEvent: RollbackEvent = {
      event_id: `trigger_${trigger.id}_${Date.now()}`,
      timestamp: new Date(),
      trigger_type: 'performance',
      metrics_snapshot: currentMetrics,
      rollback_executed: false,
      outcome: 'successful'
    };
    
    switch (trigger.action) {
      case 'rollback':
        await executeRollback(rollbackEvent);
        break;
      case 'pause':
        await pauseStrategy(rollbackEvent);
        break;
      case 'alert':
        showTriggerAlert(trigger, rollbackEvent);
        break;
    }
  };

  const executeRollback = async (event: RollbackEvent) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/rollback/${strategyId}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason: event.trigger_type,
          event_data: event
        })
      });
      
      if (response.ok) {
        event.rollback_executed = true;
        setRollbackHistory(prev => [event, ...prev]);
        
        notification.success({
          message: 'Rollback Executed',
          description: 'Strategy has been rolled back successfully'
        });
        
        onRollbackTriggered?.(event.trigger_type);
      } else {
        throw new Error('Rollback execution failed');
      }
    } catch (error) {
      event.outcome = 'failed';
      setRollbackHistory(prev => [event, ...prev]);
      
      notification.error({
        message: 'Rollback Failed',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  };

  const pauseStrategy = async (event: RollbackEvent) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/${strategyId}/pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: event.trigger_type })
      });
      
      if (response.ok) {
        setRollbackHistory(prev => [event, ...prev]);
        notification.warning({
          message: 'Strategy Paused',
          description: 'Strategy execution has been paused due to trigger activation'
        });
        onRollbackTriggered?.(event.trigger_type);
      }
    } catch (error) {
      console.error('Error pausing strategy:', error);
    }
  };

  const showTriggerAlert = (trigger: RollbackTrigger, event: RollbackEvent) => {
    setRollbackHistory(prev => [event, ...prev]);
    
    notification.warning({
      message: `Rollback Trigger Activated: ${trigger.id}`,
      description: 'Monitoring conditions have been met',
      duration: 10
    });
    
    onRollbackTriggered?.(event.trigger_type);
  };

  const updateMLServiceConfig = (values: any) => {
    if (!mlService) return;
    
    const updatedService: MLRollbackService = {
      ...mlService,
      enabled: values.enabled,
      confidence_threshold: values.confidence_threshold,
      prediction_window_minutes: values.prediction_window
    };
    
    setMlService(updatedService);
    setShowConfigModal(false);
    
    notification.success({
      message: 'Configuration Updated',
      description: 'ML rollback service configuration has been updated'
    });
  };

  const renderPredictionChart = () => {
    if (predictionData.length === 0) return null;
    
    const config = {
      data: predictionData,
      xField: 'timestamp',
      yField: 'confidence',
      height: 250,
      smooth: true,
      line: { color: '#1890ff' },
      point: { size: 3 },
      annotations: [
        {
          type: 'line',
          start: ['start', mlService?.confidence_threshold || 0.75],
          end: ['end', mlService?.confidence_threshold || 0.75],
          style: {
            stroke: '#ff4d4f',
            lineDash: [4, 4]
          }
        }
      ]
    };
    
    return (
      <Card title="ML Prediction Confidence" size="small" className="mb-4">
        <Line {...config} />
      </Card>
    );
  };

  const renderCurrentStatus = () => (
    <Row gutter={16} className="mb-4">
      <Col span={6}>
        <Card size="small">
          <Statistic
            title="ML Service Status"
            value={mlService?.enabled ? 'Active' : 'Disabled'}
            valueStyle={{ color: mlService?.enabled ? '#3f8600' : '#cf1322' }}
            prefix={mlService?.enabled ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
          />
        </Card>
      </Col>
      
      <Col span={6}>
        <Card size="small">
          <Statistic
            title="Last Prediction"
            value={lastPrediction ? lastPrediction.prediction.toUpperCase() : 'None'}
            valueStyle={{ 
              color: 
                lastPrediction?.prediction === 'critical' ? '#cf1322' :
                lastPrediction?.prediction === 'degrading' ? '#faad14' : '#3f8600'
            }}
            prefix={<RobotOutlined />}
          />
        </Card>
      </Col>
      
      <Col span={6}>
        <Card size="small">
          <Statistic
            title="Confidence"
            value={lastPrediction ? (lastPrediction.confidence_score * 100).toFixed(1) : 0}
            suffix="%"
            valueStyle={{ 
              color: lastPrediction && lastPrediction.confidence_score >= (mlService?.confidence_threshold || 0.75) 
                ? '#cf1322' : '#3f8600' 
            }}
            prefix={<LineChartOutlined />}
          />
        </Card>
      </Col>
      
      <Col span={6}>
        <Card size="small">
          <Statistic
            title="Rollback Events"
            value={rollbackHistory.length}
            prefix={<RollbackOutlined />}
            valueStyle={{ color: rollbackHistory.length > 0 ? '#faad14' : '#3f8600' }}
          />
        </Card>
      </Col>
    </Row>
  );

  const eventColumns: ColumnType<RollbackEvent>[] = [
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
      render: (timestamp: Date) => new Date(timestamp).toLocaleString()
    },
    {
      title: 'Trigger Type',
      dataIndex: 'trigger_type',
      key: 'trigger_type',
      width: 120,
      render: (type: string) => (
        <Tag color={
          type === 'ml_prediction' ? 'blue' :
          type === 'performance' ? 'orange' : 'green'
        }>
          {type.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      width: 100,
      render: (score?: number) => score ? `${(score * 100).toFixed(1)}%` : '-'
    },
    {
      title: 'Action',
      key: 'action',
      width: 100,
      render: (_, event) => (
        <Tag color={event.rollback_executed ? 'red' : 'orange'}>
          {event.rollback_executed ? 'ROLLBACK' : 'ALERT'}
        </Tag>
      )
    },
    {
      title: 'Outcome',
      dataIndex: 'outcome',
      key: 'outcome',
      width: 100,
      render: (outcome: string) => (
        <Tag color={outcome === 'successful' ? 'success' : 'error'}>
          {outcome.toUpperCase()}
        </Tag>
      )
    }
  ];

  return (
    <div className="rollback-service-manager">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <RollbackOutlined />
              <span>Rollback Service Manager</span>
              {predicting && <Spin size="small" />}
            </div>
            <Space>
              <Switch
                checked={autoMonitoring}
                onChange={setAutoMonitoring}
                checkedChildren="Auto Monitor"
                unCheckedChildren="Manual"
              />
              <Button
                icon={<SettingOutlined />}
                onClick={() => setShowConfigModal(true)}
              >
                Configure
              </Button>
              <Button
                type="primary"
                icon={<ThunderboltOutlined />}
                onClick={performMLPrediction}
                loading={predicting}
                disabled={!mlService?.enabled}
              >
                Run Prediction
              </Button>
            </Space>
          </div>
        }
      >
        {renderCurrentStatus()}
        
        {lastPrediction && (
          <Alert
            message={`ML Prediction: ${lastPrediction.prediction.toUpperCase()}`}
            description={
              <div>
                <Text>Confidence: {(lastPrediction.confidence_score * 100).toFixed(1)}%</Text>
                {lastPrediction.risk_factors.length > 0 && (
                  <div className="mt-2">
                    <Text strong>Risk Factors: </Text>
                    {lastPrediction.risk_factors.join(', ')}
                  </div>
                )}
                {lastPrediction.time_to_rollback_minutes && (
                  <div className="mt-2">
                    <Text strong>Time to Rollback: </Text>
                    {lastPrediction.time_to_rollback_minutes} minutes
                  </div>
                )}
              </div>
            }
            type={
              lastPrediction.prediction === 'critical' ? 'error' :
              lastPrediction.prediction === 'degrading' ? 'warning' : 'info'
            }
            showIcon
            className="mb-4"
          />
        )}
        
        {renderPredictionChart()}
        
        <Card title="Rollback History" size="small">
          <Table
            columns={eventColumns}
            dataSource={rollbackHistory}
            rowKey="event_id"
            size="small"
            pagination={{ pageSize: 10 }}
          />
        </Card>
      </Card>

      {/* Configuration Modal */}
      <Modal
        title="Rollback Service Configuration"
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={updateMLServiceConfig}
        >
          <Form.Item
            label="Enable ML Predictions"
            name="enabled"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          
          <Form.Item
            label="Confidence Threshold"
            name="confidence_threshold"
            tooltip="Minimum confidence required to trigger rollback"
          >
            <InputNumber
              min={0.5}
              max={1.0}
              step={0.05}
              formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
              parser={value => Number(value!.replace('%', '')) / 100}
            />
          </Form.Item>
          
          <Form.Item
            label="Prediction Window (minutes)"
            name="prediction_window"
            tooltip="Time horizon for predictions"
          >
            <InputNumber min={5} max={120} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default RollbackServiceManager;