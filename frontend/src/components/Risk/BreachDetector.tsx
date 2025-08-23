import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Alert,
  Progress,
  Tag,
  Space,
  Button,
  Tooltip,
  Timeline,
  Statistic,
  Table,
  Typography,
  Badge,
  Collapse,
  List,
  Switch,
  Select,
  Divider,
  Modal,
  Descriptions
} from 'antd';
import {
  WarningOutlined,
  ThunderboltOutlined,
  RiseOutlined,
  ClockCircleOutlined,
  BulbOutlined,
  EyeOutlined,
  SettingOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  RocketOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { Line } from '@ant-design/plots';
import { useBreachDetection, BreachPrediction } from '../../hooks/risk/useBreachDetection';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;
const { Option } = Select;

interface BreachDetectorProps {
  portfolioId: string;
  className?: string;
}

const BreachDetector: React.FC<BreachDetectorProps> = ({
  portfolioId,
  className
}) => {
  const {
    predictions,
    patterns,
    activeAlerts,
    recentBreaches,
    mlModelInfo,
    loading,
    error,
    highRiskPredictions,
    imminentBreaches,
    criticalAlerts,
    overallRiskScore,
    fetchPredictions,
    acknowledgeAlert,
    dismissAlert,
    clearError,
    isConnected,
    getPredictionForLimit,
    getPatternForLimitType
  } = useBreachDetection({ portfolioId, enableRealTime: true });

  const [selectedPrediction, setSelectedPrediction] = useState<BreachPrediction | null>(null);
  const [showPredictionModal, setShowPredictionModal] = useState(false);
  const [predictionHorizon, setPredictionHorizon] = useState(60);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Generate chart data for prediction trends
  const generatePredictionTrendData = (prediction: BreachPrediction) => {
    const data = [];
    const currentTime = Date.now();
    
    for (let i = -30; i <= 0; i++) {
      const timestamp = currentTime + (i * 60 * 1000); // Minutes
      const value = prediction.current_value + (Math.random() - 0.5) * prediction.current_value * 0.02;
      data.push({
        time: new Date(timestamp).toLocaleTimeString(),
        value: value,
        threshold: prediction.threshold_value,
        warning: prediction.warning_threshold,
        type: 'historical'
      });
    }

    // Add future projections
    if (prediction.time_to_breach_minutes) {
      for (let i = 1; i <= Math.min(prediction.time_to_breach_minutes, 60); i++) {
        const timestamp = currentTime + (i * 60 * 1000);
        const progressToThreshold = i / prediction.time_to_breach_minutes;
        const projectedValue = prediction.current_value + 
          (prediction.threshold_value - prediction.current_value) * progressToThreshold;
        
        data.push({
          time: new Date(timestamp).toLocaleTimeString(),
          value: projectedValue,
          threshold: prediction.threshold_value,
          warning: prediction.warning_threshold,
          type: 'projected'
        });
      }
    }

    return data;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'high': return '#fa8c16';
      case 'medium': return '#faad14';
      case 'low': return '#52c41a';
      default: return '#d9d9d9';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <WarningOutlined />;
      case 'high': return <WarningOutlined />;
      case 'medium': return <AlertOutlined />;
      case 'low': return <CheckCircleOutlined />;
      default: return <BulbOutlined />;
    }
  };

  const predictionColumns = [
    {
      title: 'Limit',
      key: 'limit',
      width: 200,
      render: (record: BreachPrediction) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.limit_name}</Text>
          <Tag color="blue" size="small">{record.limit_type.toUpperCase()}</Tag>
        </Space>
      )
    },
    {
      title: 'Current vs Threshold',
      key: 'values',
      width: 180,
      render: (record: BreachPrediction) => {
        const percentage = (record.current_value / record.threshold_value) * 100;
        return (
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text style={{ fontSize: '12px' }}>
                {record.current_value.toLocaleString()}
              </Text>
              <Text style={{ fontSize: '12px' }}>
                {record.threshold_value.toLocaleString()}
              </Text>
            </div>
            <Progress
              percent={percentage}
              size="small"
              strokeColor={percentage > 90 ? '#ff4d4f' : percentage > 75 ? '#faad14' : '#1890ff'}
              showInfo={false}
            />
            <Text style={{ fontSize: '11px', color: '#666' }}>
              {percentage.toFixed(1)}% utilized
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Breach Probability',
      dataIndex: 'probability_of_breach',
      key: 'probability',
      width: 120,
      render: (probability: number) => (
        <Space direction="vertical" size={0} align="center">
          <Text strong style={{ color: getSeverityColor(probability > 0.8 ? 'critical' : probability > 0.6 ? 'high' : probability > 0.4 ? 'medium' : 'low') }}>
            {(probability * 100).toFixed(1)}%
          </Text>
          <Progress
            type="circle"
            percent={probability * 100}
            width={40}
            strokeColor={getSeverityColor(probability > 0.8 ? 'critical' : probability > 0.6 ? 'high' : probability > 0.4 ? 'medium' : 'low')}
            format={() => ''}
          />
        </Space>
      )
    },
    {
      title: 'Time to Breach',
      dataIndex: 'time_to_breach_minutes',
      key: 'time_to_breach',
      width: 100,
      render: (minutes?: number) => {
        if (!minutes) return <Text type="secondary">N/A</Text>;
        
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        const isUrgent = minutes < 30;
        
        return (
          <Space>
            <ClockCircleOutlined style={{ color: isUrgent ? '#ff4d4f' : '#666' }} />
            <Text style={{ color: isUrgent ? '#ff4d4f' : undefined }}>
              {hours > 0 ? `${hours}h ${mins}m` : `${mins}m`}
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence_score',
      key: 'confidence',
      width: 100,
      render: (score: number) => (
        <Tooltip title={`ML Model Confidence: ${(score * 100).toFixed(1)}%`}>
          <Badge
            color={score > 0.8 ? 'green' : score > 0.6 ? 'orange' : 'red'}
            text={`${(score * 100).toFixed(0)}%`}
          />
        </Tooltip>
      )
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)} icon={getSeverityIcon(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (record: BreachPrediction) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          size="small"
          onClick={() => {
            setSelectedPrediction(record);
            setShowPredictionModal(true);
          }}
        >
          Details
        </Button>
      )
    }
  ];

  const chartConfig = {
    data: selectedPrediction ? generatePredictionTrendData(selectedPrediction) : [],
    xField: 'time',
    yField: 'value',
    seriesField: 'type',
    color: ['#1890ff', '#ff4d4f'],
    point: {
      size: 3,
      shape: 'circle',
    },
    annotations: selectedPrediction ? [
      {
        type: 'line',
        start: ['min', selectedPrediction.threshold_value],
        end: ['max', selectedPrediction.threshold_value],
        style: {
          stroke: '#ff4d4f',
          lineDash: [4, 4],
        },
        text: {
          content: 'Breach Threshold',
          position: 'end',
          style: { fill: '#ff4d4f' }
        }
      },
      {
        type: 'line',
        start: ['min', selectedPrediction.warning_threshold],
        end: ['max', selectedPrediction.warning_threshold],
        style: {
          stroke: '#faad14',
          lineDash: [2, 2],
        },
        text: {
          content: 'Warning Level',
          position: 'end',
          style: { fill: '#faad14' }
        }
      }
    ] : [],
    legend: {
      position: 'top',
    },
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    }
  };

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Overall Risk Score"
              value={overallRiskScore}
              precision={1}
              suffix="%"
              prefix={<RiseOutlined style={{ color: overallRiskScore > 70 ? '#ff4d4f' : overallRiskScore > 40 ? '#faad14' : '#52c41a' }} />}
              valueStyle={{ color: overallRiskScore > 70 ? '#ff4d4f' : overallRiskScore > 40 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="High Risk Predictions"
              value={highRiskPredictions.length}
              prefix={<WarningOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: highRiskPredictions.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Imminent Breaches"
              value={imminentBreaches.length}
              prefix={<ThunderboltOutlined style={{ color: '#fa8c16' }} />}
              valueStyle={{ color: imminentBreaches.length > 0 ? '#fa8c16' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '14px', color: '#666' }}>ML Model Status</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: isConnected ? '#52c41a' : '#ff4d4f' }}>
                  {isConnected ? 'Active' : 'Offline'}
                </div>
              </div>
              <BulbOutlined style={{ fontSize: '24px', color: isConnected ? '#52c41a' : '#ff4d4f' }} />
            </div>
          </Card>
        </Col>
      </Row>

      {/* Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Text>Prediction Horizon:</Text>
              <Select
                value={predictionHorizon}
                onChange={setPredictionHorizon}
                style={{ width: 120 }}
              >
                <Option value={15}>15 minutes</Option>
                <Option value={30}>30 minutes</Option>
                <Option value={60}>1 hour</Option>
                <Option value={240}>4 hours</Option>
                <Option value={1440}>24 hours</Option>
              </Select>
              
              <Divider type="vertical" />
              
              <Text>Auto Refresh:</Text>
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="ON"
                unCheckedChildren="OFF"
              />
              
              <Divider type="vertical" />
              
              <Tooltip title="Real-time Connection Status">
                <Badge
                  status={isConnected ? 'processing' : 'error'}
                  text={isConnected ? 'Live' : 'Disconnected'}
                />
              </Tooltip>
            </Space>
          </Col>
          <Col>
            <Button icon={<SettingOutlined />}>
              Configure ML Models
            </Button>
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
          onClose={clearError}
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={fetchPredictions}>
              Retry
            </Button>
          }
        />
      )}

      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <Alert
          message={`${criticalAlerts.length} Critical Risk Alert${criticalAlerts.length > 1 ? 's' : ''}`}
          description={
            <List
              size="small"
              dataSource={criticalAlerts}
              renderItem={(alert) => (
                <List.Item
                  actions={[
                    <Button size="small" onClick={() => acknowledgeAlert(alert.id, 'user')}>
                      Acknowledge
                    </Button>,
                    <Button size="small" type="text" onClick={() => dismissAlert(alert.id)}>
                      Dismiss
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    title={alert.alert_type.replace('_', ' ').toUpperCase()}
                    description={alert.message}
                  />
                </List.Item>
              )}
            />
          }
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={16}>
        {/* Predictions Table */}
        <Col span={16}>
          <Card
            title={
              <Space>
                <RocketOutlined />
                <Title level={4} style={{ margin: 0 }}>Breach Predictions</Title>
                <Badge count={predictions.length} style={{ backgroundColor: '#1890ff' }} />
              </Space>
            }
            extra={
              mlModelInfo && (
                <Tooltip title={`Model: v${mlModelInfo.model_version} | Accuracy: ${(mlModelInfo.accuracy_score * 100).toFixed(1)}%`}>
                  <Tag icon={<BulbOutlined />} color="blue">
                    ML Powered
                  </Tag>
                </Tooltip>
              )
            }
          >
            <Table
              dataSource={predictions}
              columns={predictionColumns}
              rowKey="limit_id"
              loading={loading.predictions}
              size="small"
              pagination={{
                pageSize: 10,
                showSizeChanger: false,
                showQuickJumper: true
              }}
              rowClassName={(record) => {
                if (record.severity === 'critical') return 'prediction-critical';
                if (record.severity === 'high') return 'prediction-high';
                return '';
              }}
            />
          </Card>
        </Col>

        {/* Sidebar */}
        <Col span={8}>
          {/* Recent Breaches */}
          <Card
            title="Recent Breaches"
            size="small"
            style={{ marginBottom: 16 }}
            extra={<Badge count={recentBreaches.length} />}
          >
            <Timeline mode="left" size="small">
              {recentBreaches.slice(0, 5).map((breach, index) => (
                <Timeline.Item
                  key={index}
                  color={breach.severity === 'critical' ? 'red' : 'orange'}
                >
                  <div style={{ fontSize: '12px' }}>
                    <div style={{ fontWeight: 'bold' }}>{breach.limit_name}</div>
                    <div style={{ color: '#666' }}>
                      {breach.timestamp.toLocaleTimeString()}
                    </div>
                    {breach.recovery_time_minutes && (
                      <div style={{ color: '#52c41a' }}>
                        Recovered in {breach.recovery_time_minutes}m
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
            {recentBreaches.length === 0 && (
              <Text type="secondary">No recent breaches</Text>
            )}
          </Card>

          {/* Patterns Analysis */}
          <Card
            title="Breach Patterns"
            size="small"
            extra={<LineChartOutlined />}
          >
            <Collapse ghost>
              {patterns.map((pattern, index) => (
                <Panel
                  header={
                    <Space>
                      <Tag color="blue">{pattern.limit_type.toUpperCase()}</Tag>
                      <Text style={{ fontSize: '12px' }}>
                        {(pattern.breach_frequency * 100).toFixed(1)}% frequency
                      </Text>
                    </Space>
                  }
                  key={index}
                >
                  <Descriptions size="small" column={1}>
                    <Descriptions.Item label="Avg Recovery Time">
                      {pattern.average_recovery_time_minutes} minutes
                    </Descriptions.Item>
                    <Descriptions.Item label="Market Correlation">
                      {(pattern.correlation_with_market * 100).toFixed(0)}%
                    </Descriptions.Item>
                    <Descriptions.Item label="Common Triggers">
                      <div>
                        {pattern.common_triggers.map((trigger, i) => (
                          <Tag key={i} size="small">{trigger}</Tag>
                        ))}
                      </div>
                    </Descriptions.Item>
                  </Descriptions>
                </Panel>
              ))}
            </Collapse>
          </Card>
        </Col>
      </Row>

      {/* Prediction Detail Modal */}
      <Modal
        title={
          <Space>
            <LineChartOutlined />
            {selectedPrediction?.limit_name} - Breach Prediction
          </Space>
        }
        open={showPredictionModal}
        onCancel={() => {
          setShowPredictionModal(false);
          setSelectedPrediction(null);
        }}
        width={900}
        footer={null}
      >
        {selectedPrediction && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Statistic
                  title="Breach Probability"
                  value={selectedPrediction.probability_of_breach * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getSeverityColor(selectedPrediction.severity) }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Time to Breach"
                  value={selectedPrediction.time_to_breach_minutes || 0}
                  suffix="minutes"
                  valueStyle={{ color: (selectedPrediction.time_to_breach_minutes || 0) < 30 ? '#ff4d4f' : '#666' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Confidence Score"
                  value={selectedPrediction.confidence_score * 100}
                  precision={1}
                  suffix="%"
                />
              </Col>
            </Row>

            <Divider>Trend Analysis</Divider>
            <div style={{ height: 300 }}>
              <Line {...chartConfig} />
            </div>

            <Divider>Contributing Factors</Divider>
            <List
              size="small"
              dataSource={selectedPrediction.contributing_factors}
              renderItem={(factor) => (
                <List.Item>
                  <BulbOutlined style={{ color: '#faad14', marginRight: 8 }} />
                  {factor}
                </List.Item>
              )}
            />

            <Divider>Recommended Actions</Divider>
            <List
              size="small"
              dataSource={selectedPrediction.recommended_actions}
              renderItem={(action) => (
                <List.Item>
                  <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                  {action}
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>

      <style jsx>{`
        .prediction-critical {
          background-color: #fff2f0 !important;
        }
        .prediction-high {
          background-color: #fff7e6 !important;
        }
      `}</style>
    </div>
  );
};

export default BreachDetector;