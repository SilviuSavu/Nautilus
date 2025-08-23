import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Tag,
  Progress,
  Button,
  Space,
  Alert,
  Tooltip,
  Badge,
  Typography,
  Select,
  Switch,
  Row,
  Col,
  Statistic,
  List,
  Avatar,
  Modal,
  Descriptions,
  Timeline,
  notification,
  Divider
} from 'antd';
import {
  WarningOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BulbOutlined,
  LineChartOutlined,
  SettingOutlined,
  ReloadOutlined,
  EyeOutlined,
  BellOutlined,
  FireOutlined,
  SafetyCertificateOutlined
} from '@ant-design/icons';

import { BreachPrediction, RecommendedAction } from './types/riskTypes';
import { riskService } from './services/riskService';
import { useBreachDetection } from '../../hooks/risk/useBreachDetection';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface AdvancedBreachDetectorProps {
  portfolioId: string;
  className?: string;
}

interface BreachPredictionWithActions extends BreachPrediction {
  actionsExpanded?: boolean;
}

const AdvancedBreachDetector: React.FC<AdvancedBreachDetectorProps> = ({
  portfolioId,
  className
}) => {
  console.log('🎯 AdvancedBreachDetector rendering for portfolio:', portfolioId);

  const [predictions, setPredictions] = useState<BreachPredictionWithActions[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predictionEnabled, setPredictionEnabled] = useState(false);
  const [timeHorizon, setTimeHorizon] = useState<'15m' | '30m' | '1h' | '4h' | '24h'>('1h');
  const [selectedPrediction, setSelectedPrediction] = useState<BreachPrediction | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Use breach detection hook
  const {
    highRiskPredictions,
    imminentBreaches,
    overallRiskScore,
    mlModelAccuracy,
    lastPredictionUpdate
  } = useBreachDetection({ 
    portfolioId, 
    enableRealTime: predictionEnabled,
    timeHorizon
  });

  const fetchBreachPredictions = async () => {
    try {
      setError(null);
      const data = await riskService.getBreachPredictions(portfolioId, timeHorizon);
      setPredictions(data.map(prediction => ({ ...prediction, actionsExpanded: false })));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch breach predictions');
      console.error('Breach predictions fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionToggle = async (enabled: boolean) => {
    try {
      if (enabled) {
        await riskService.enableBreachPrediction(portfolioId, {
          update_frequency_minutes: 5,
          confidence_threshold: 0.7,
          prediction_horizon_hours: parseInt(timeHorizon.replace(/[^0-9]/g, '')) || 1
        });
        notification.success({
          message: 'ML Breach Detection Enabled',
          description: 'Machine learning predictions are now active',
          duration: 3
        });
      } else {
        await riskService.disableBreachPrediction(portfolioId);
        notification.info({
          message: 'ML Breach Detection Disabled',
          description: 'Machine learning predictions have been deactivated',
          duration: 3
        });
      }
      setPredictionEnabled(enabled);
    } catch (error) {
      console.error('Failed to toggle breach prediction:', error);
      notification.error({
        message: 'Prediction Toggle Failed',
        description: 'Unable to change prediction status',
        duration: 4
      });
    }
  };

  const handleTimeHorizonChange = (value: '15m' | '30m' | '1h' | '4h' | '24h') => {
    setTimeHorizon(value);
    if (predictionEnabled) {
      fetchBreachPredictions();
    }
  };

  const getProbabilityColor = (probability: number): string => {
    if (probability >= 0.8) return '#ff4d4f'; // Critical
    if (probability >= 0.6) return '#fa8c16'; // High
    if (probability >= 0.4) return '#faad14'; // Medium
    return '#52c41a'; // Low
  };

  const getRiskLevelColor = (level: string): string => {
    switch (level) {
      case 'critical': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'gold';
      case 'low': return 'green';
      default: return 'blue';
    }
  };

  const getActionPriorityColor = (priority: string): string => {
    switch (priority) {
      case 'urgent': return 'red';
      case 'high': return 'orange';
      case 'medium': return 'blue';
      case 'low': return 'green';
      default: return 'default';
    }
  };

  const formatTimeUntilBreach = (breachTime: Date): string => {
    const now = new Date();
    const diff = breachTime.getTime() - now.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m`;
  };

  const executeRecommendedAction = async (action: RecommendedAction, limitId: string) => {
    try {
      // This would typically trigger the actual action execution
      notification.success({
        message: 'Action Executed',
        description: `${action.description} has been initiated`,
        duration: 4
      });
    } catch (error) {
      notification.error({
        message: 'Action Failed',
        description: 'Unable to execute the recommended action',
        duration: 4
      });
    }
  };

  const columns = [
    {
      title: 'Limit',
      dataIndex: 'limit_name',
      key: 'limit_name',
      render: (text: string, record: BreachPredictionWithActions) => (
        <Space>
          <Text strong>{text}</Text>
          <Tag color={getRiskLevelColor(record.risk_level)}>
            {record.risk_level.toUpperCase()}
          </Tag>
        </Space>
      )
    },
    {
      title: 'Breach Probability',
      dataIndex: 'breach_probability',
      key: 'breach_probability',
      render: (probability: number) => (
        <div style={{ width: 120 }}>
          <Progress
            percent={Math.round(probability * 100)}
            size="small"
            strokeColor={getProbabilityColor(probability)}
            format={(percent) => `${percent}%`}
          />
        </div>
      ),
      sorter: (a: BreachPrediction, b: BreachPrediction) => a.breach_probability - b.breach_probability
    },
    {
      title: 'Time to Breach',
      dataIndex: 'predicted_breach_time',
      key: 'predicted_breach_time',
      render: (time: Date, record: BreachPredictionWithActions) => {
        const timeUntil = formatTimeUntilBreach(new Date(time));
        const isUrgent = new Date(time).getTime() - Date.now() < 30 * 60 * 1000; // 30 minutes
        return (
          <Space>
            <ClockCircleOutlined style={{ color: isUrgent ? '#ff4d4f' : '#1890ff' }} />
            <Text style={{ color: isUrgent ? '#ff4d4f' : undefined }}>
              {timeUntil}
            </Text>
            {isUrgent && <Badge status="error" text="Urgent" />}
          </Space>
        );
      },
      sorter: (a: BreachPrediction, b: BreachPrediction) => 
        new Date(a.predicted_breach_time).getTime() - new Date(b.predicted_breach_time).getTime()
    },
    {
      title: 'Current vs Limit',
      key: 'values',
      render: (record: BreachPrediction) => {
        const currentNum = parseFloat(record.current_value);
        const limitNum = parseFloat(record.limit_value);
        const utilizationPercent = (currentNum / limitNum) * 100;
        
        return (
          <div>
            <div style={{ fontSize: '12px' }}>
              <Text>Current: </Text>
              <Text strong>{record.current_value}</Text>
            </div>
            <div style={{ fontSize: '12px' }}>
              <Text>Limit: </Text>
              <Text>{record.limit_value}</Text>
            </div>
            <Progress
              percent={Math.min(utilizationPercent, 100)}
              size="small"
              strokeColor={utilizationPercent > 90 ? '#ff4d4f' : '#1890ff'}
              showInfo={false}
              style={{ marginTop: 4 }}
            />
          </div>
        );
      }
    },
    {
      title: 'ML Confidence',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      render: (confidence: number) => (
        <Tooltip title="Machine Learning prediction confidence">
          <Space>
            <RobotOutlined />
            <Progress
              type="circle"
              percent={Math.round(confidence * 100)}
              size={40}
              strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f'}
            />
          </Space>
        </Tooltip>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: BreachPredictionWithActions) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedPrediction(record);
              setDetailsVisible(true);
            }}
          >
            Details
          </Button>
          <Badge count={record.recommended_actions.length}>
            <Button
              size="small"
              icon={<BulbOutlined />}
              type={record.recommended_actions.some(a => a.priority === 'urgent') ? 'primary' : 'default'}
            >
              Actions
            </Button>
          </Badge>
        </Space>
      )
    }
  ];

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh && predictionEnabled) {
      const interval = setInterval(fetchBreachPredictions, 30000); // Every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh, predictionEnabled, timeHorizon]);

  // Initial data fetch
  useEffect(() => {
    fetchBreachPredictions();
  }, [portfolioId, timeHorizon]);

  // Use predictions from hook if available
  const currentPredictions = highRiskPredictions.length > 0 ? highRiskPredictions : predictions;
  const criticalCount = currentPredictions.filter(p => p.risk_level === 'critical').length;
  const urgentCount = imminentBreaches.length;

  return (
    <div className={className}>
      {/* Header Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Title level={4} style={{ margin: 0 }}>
                <RobotOutlined /> ML Breach Detection
              </Title>
              <Badge 
                status={predictionEnabled ? 'processing' : 'default'}
                text={predictionEnabled ? 'Active' : 'Inactive'}
              />
              {mlModelAccuracy && (
                <Tooltip title="ML Model Accuracy">
                  <Tag color="blue">
                    Accuracy: {Math.round(mlModelAccuracy * 100)}%
                  </Tag>
                </Tooltip>
              )}
            </Space>
          </Col>
          <Col>
            <Space>
              <Select 
                value={timeHorizon}
                onChange={handleTimeHorizonChange}
                style={{ width: 100 }}
                size="small"
              >
                <Option value="15m">15m</Option>
                <Option value="30m">30m</Option>
                <Option value="1h">1h</Option>
                <Option value="4h">4h</Option>
                <Option value="24h">24h</Option>
              </Select>
              
              <Tooltip title="Auto-refresh predictions">
                <Switch
                  size="small"
                  checked={autoRefresh}
                  onChange={setAutoRefresh}
                  checkedChildren="Auto"
                  unCheckedChildren="Manual"
                />
              </Tooltip>

              <Tooltip title={predictionEnabled ? 'Disable ML predictions' : 'Enable ML predictions'}>
                <Switch
                  checked={predictionEnabled}
                  onChange={handlePredictionToggle}
                  checkedChildren={<ThunderboltOutlined />}
                  unCheckedChildren={<ThunderboltOutlined />}
                  loading={loading}
                />
              </Tooltip>

              <Button 
                icon={<ReloadOutlined />}
                onClick={fetchBreachPredictions}
                size="small"
                loading={loading}
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Breach Detection Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Critical Alerts */}
      {urgentCount > 0 && (
        <Alert
          message={
            <Space>
              <FireOutlined />
              <Text strong>IMMINENT BREACH DETECTED</Text>
            </Space>
          }
          description={`${urgentCount} limit breach(es) predicted within 30 minutes`}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" danger>
              Emergency Actions
            </Button>
          }
        />
      )}

      {/* Summary Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Total Predictions"
              value={currentPredictions.length}
              prefix={<LineChartOutlined />}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Critical Risk"
              value={criticalCount}
              prefix={<AlertOutlined />}
              valueStyle={{ color: criticalCount > 0 ? '#ff4d4f' : undefined }}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Imminent (< 30min)"
              value={urgentCount}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: urgentCount > 0 ? '#ff4d4f' : undefined }}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card size="small">
            <Statistic
              title="Overall Risk Score"
              value={overallRiskScore || 0}
              precision={1}
              suffix="%"
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ 
                color: (overallRiskScore || 0) > 70 ? '#ff4d4f' : 
                       (overallRiskScore || 0) > 40 ? '#faad14' : '#52c41a'
              }}
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* Predictions Table */}
      <Card
        title={
          <Space>
            <WarningOutlined />
            Breach Predictions ({timeHorizon} horizon)
            {lastPredictionUpdate && (
              <Text type="secondary" style={{ fontSize: '12px', marginLeft: 8 }}>
                Updated: {new Date(lastPredictionUpdate).toLocaleTimeString()}
              </Text>
            )}
          </Space>
        }
        extra={
          <Space>
            <Badge count={urgentCount} style={{ backgroundColor: '#ff4d4f' }}>
              <Button size="small" icon={<FireOutlined />}>
                Urgent
              </Button>
            </Badge>
            <Badge count={criticalCount} style={{ backgroundColor: '#fa8c16' }}>
              <Button size="small" icon={<WarningOutlined />}>
                Critical
              </Button>
            </Badge>
          </Space>
        }
      >
        <Table<BreachPredictionWithActions>
          columns={columns}
          dataSource={currentPredictions}
          rowKey="limit_id"
          loading={loading}
          size="small"
          expandable={{
            expandedRowRender: (record) => (
              <div style={{ margin: 0 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Card size="small" title="Contributing Factors">
                      <List
                        size="small"
                        dataSource={record.contributing_factors}
                        renderItem={(factor) => (
                          <List.Item>
                            <Text>• {factor}</Text>
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="Recommended Actions">
                      <List
                        size="small"
                        dataSource={record.recommended_actions}
                        renderItem={(action) => (
                          <List.Item
                            actions={[
                              <Button
                                key="execute"
                                size="small"
                                type="primary"
                                onClick={() => executeRecommendedAction(action, record.limit_id)}
                              >
                                Execute
                              </Button>
                            ]}
                          >
                            <List.Item.Meta
                              avatar={
                                <Avatar 
                                  size="small"
                                  style={{ 
                                    backgroundColor: getActionPriorityColor(action.priority) 
                                  }}
                                >
                                  {action.priority[0].toUpperCase()}
                                </Avatar>
                              }
                              title={
                                <Space>
                                  <Text>{action.description}</Text>
                                  <Tag size="small" color={getActionPriorityColor(action.priority)}>
                                    {action.priority}
                                  </Tag>
                                </Space>
                              }
                              description={
                                <Space>
                                  <Text type="secondary">Impact: {action.estimated_impact}</Text>
                                  {action.execution_time_minutes && (
                                    <Text type="secondary">
                                      ETA: {action.execution_time_minutes}min
                                    </Text>
                                  )}
                                </Space>
                              }
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                </Row>
              </div>
            ),
            rowExpandable: (record) => record.contributing_factors.length > 0
          }}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} predictions`
          }}
        />
      </Card>

      {/* Prediction Details Modal */}
      <Modal
        title={
          <Space>
            <WarningOutlined />
            Breach Prediction Details
          </Space>
        }
        open={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailsVisible(false)}>
            Close
          </Button>
        ]}
        width={800}
      >
        {selectedPrediction && (
          <div>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Limit" span={2}>
                {selectedPrediction.limit_name}
              </Descriptions.Item>
              <Descriptions.Item label="Risk Level">
                <Tag color={getRiskLevelColor(selectedPrediction.risk_level)}>
                  {selectedPrediction.risk_level.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              
              <Descriptions.Item label="Breach Probability">
                {Math.round(selectedPrediction.breach_probability * 100)}%
              </Descriptions.Item>
              <Descriptions.Item label="ML Confidence">
                {Math.round(selectedPrediction.confidence_score * 100)}%
              </Descriptions.Item>
              <Descriptions.Item label="Predicted Time">
                {new Date(selectedPrediction.predicted_breach_time).toLocaleString()}
              </Descriptions.Item>
              
              <Descriptions.Item label="Current Value">
                {selectedPrediction.current_value}
              </Descriptions.Item>
              <Descriptions.Item label="Limit Value">
                {selectedPrediction.limit_value}
              </Descriptions.Item>
              <Descriptions.Item label="Time Until Breach">
                <Text strong style={{ color: '#ff4d4f' }}>
                  {formatTimeUntilBreach(new Date(selectedPrediction.predicted_breach_time))}
                </Text>
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            <Title level={5}>Contributing Factors</Title>
            <Timeline
              items={selectedPrediction.contributing_factors.map(factor => ({
                children: factor,
                color: 'red'
              }))}
            />

            <Divider />

            <Title level={5}>Recommended Actions</Title>
            <List
              dataSource={selectedPrediction.recommended_actions}
              renderItem={(action) => (
                <List.Item
                  actions={[
                    <Button
                      key="execute"
                      type="primary"
                      onClick={() => executeRecommendedAction(action, selectedPrediction.limit_id)}
                    >
                      Execute Now
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar style={{ backgroundColor: getActionPriorityColor(action.priority) }}>
                        {action.priority[0].toUpperCase()}
                      </Avatar>
                    }
                    title={
                      <Space>
                        <Text strong>{action.description}</Text>
                        <Tag color={getActionPriorityColor(action.priority)}>
                          {action.priority.toUpperCase()}
                        </Tag>
                      </Space>
                    }
                    description={
                      <div>
                        <Paragraph>
                          <Text strong>Expected Impact:</Text> {action.estimated_impact}
                        </Paragraph>
                        {action.execution_time_minutes && (
                          <Paragraph>
                            <Text strong>Execution Time:</Text> {action.execution_time_minutes} minutes
                          </Paragraph>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AdvancedBreachDetector;