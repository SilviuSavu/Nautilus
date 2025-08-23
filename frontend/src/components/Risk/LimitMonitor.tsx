import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Progress,
  Typography,
  Alert,
  Space,
  Tag,
  Button,
  Tooltip,
  Table,
  Statistic,
  Badge,
  Select,
  Switch,
  Timeline,
  List,
  Descriptions,
  Modal,
  Divider
} from 'antd';
import {
  WarningOutlined,
  ThunderboltOutlined,
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  SettingOutlined,
  ReloadOutlined,
  BellOutlined,
  DashboardOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { Line, Gauge } from '@ant-design/plots';
import { useRiskMonitoring, LimitUtilization } from '../../hooks/risk/useRiskMonitoring';

const { Title, Text } = Typography;
const { Option } = Select;

interface LimitMonitorProps {
  portfolioId: string;
  className?: string;
}

const LimitMonitor: React.FC<LimitMonitorProps> = ({
  portfolioId,
  className
}) => {
  const {
    realTimeMetrics,
    limitUtilization,
    systemHealth,
    monitoringConfig,
    loading,
    error,
    connectionStatus,
    criticalAlerts,
    breachedLimits,
    warningLimits,
    avgUtilization,
    riskTrendDirection,
    fetchCurrentMetrics,
    updateConfiguration,
    clearError,
    getLimitUtilization,
    isConnected
  } = useRiskMonitoring({ portfolioId, enableRealTime: true });

  const [selectedLimit, setSelectedLimit] = useState<LimitUtilization | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'table' | 'chart'>('grid');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const getUtilizationColor = (percentage: number) => {
    if (percentage >= 100) return '#ff4d4f';
    if (percentage >= 90) return '#ff7a45';
    if (percentage >= 75) return '#faad14';
    if (percentage >= 50) return '#1890ff';
    return '#52c41a';
  };

  const getStatusIcon = (status: LimitUtilization['status']) => {
    switch (status) {
      case 'breach': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'critical': return <WarningOutlined style={{ color: '#ff7a45' }} />;
      case 'warning': return <BellOutlined style={{ color: '#faad14' }} />;
      case 'normal': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      default: return <MinusOutlined />;
    }
  };

  const getTrendIcon = (direction: LimitUtilization['trend_direction']) => {
    switch (direction) {
      case 'up': return <RiseOutlined style={{ color: '#ff4d4f' }} />;
      case 'down': return <FallOutlined style={{ color: '#52c41a' }} />;
      case 'stable': return <MinusOutlined style={{ color: '#666' }} />;
      default: return <MinusOutlined />;
    }
  };

  const generateTrendData = (limit: LimitUtilization) => {
    // Generate mock trend data - would be replaced with actual historical data
    const data = [];
    const now = Date.now();
    
    for (let i = -60; i <= 0; i += 5) {
      const timestamp = now + (i * 60 * 1000);
      const baseValue = limit.utilization_percentage;
      const variation = (Math.random() - 0.5) * 10;
      const value = Math.max(0, Math.min(120, baseValue + variation));
      
      data.push({
        time: new Date(timestamp).toLocaleTimeString(),
        utilization: value,
        threshold: 100,
        warning: (limit.warning_threshold / limit.threshold_value) * 100
      });
    }
    
    return data;
  };

  const filteredLimits = limitUtilization.filter(limit => {
    if (filterStatus === 'all') return true;
    return limit.status === filterStatus;
  });

  const columns = [
    {
      title: 'Limit',
      key: 'limit',
      width: 200,
      render: (record: LimitUtilization) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.limit_name}</Text>
          <Tag color="blue" size="small">{record.limit_type.toUpperCase()}</Tag>
        </Space>
      )
    },
    {
      title: 'Current Value',
      key: 'current_value',
      width: 120,
      render: (record: LimitUtilization) => (
        <Text>{record.current_value.toLocaleString()}</Text>
      )
    },
    {
      title: 'Utilization',
      key: 'utilization',
      width: 200,
      render: (record: LimitUtilization) => (
        <Space direction="vertical" size={4} style={{ width: '100%' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Text style={{ fontSize: '12px' }}>
              {record.utilization_percentage.toFixed(1)}%
            </Text>
            <Text style={{ fontSize: '12px', color: getUtilizationColor(record.utilization_percentage) }}>
              {record.threshold_value.toLocaleString()}
            </Text>
          </div>
          <Progress
            percent={record.utilization_percentage}
            size="small"
            strokeColor={getUtilizationColor(record.utilization_percentage)}
            showInfo={false}
          />
        </Space>
      )
    },
    {
      title: 'Breach Probability',
      dataIndex: 'breach_probability',
      key: 'breach_probability',
      width: 120,
      render: (probability: number) => (
        <Space>
          <Progress
            type="circle"
            percent={probability * 100}
            width={30}
            strokeColor={getUtilizationColor(probability * 100)}
            format={() => ''}
          />
          <Text>{(probability * 100).toFixed(0)}%</Text>
        </Space>
      )
    },
    {
      title: 'Time to Breach',
      dataIndex: 'time_to_breach_estimate',
      key: 'time_to_breach',
      width: 120,
      render: (minutes?: number) => {
        if (!minutes) return <Text type="secondary">N/A</Text>;
        
        const isUrgent = minutes < 30;
        return (
          <Space>
            <ClockCircleOutlined style={{ color: isUrgent ? '#ff4d4f' : '#666' }} />
            <Text style={{ color: isUrgent ? '#ff4d4f' : undefined }}>
              {minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`}
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: LimitUtilization['status']) => (
        <Tag icon={getStatusIcon(status)} color={
          status === 'breach' ? 'error' :
          status === 'critical' ? 'volcano' :
          status === 'warning' ? 'warning' : 'success'
        }>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Trend',
      dataIndex: 'trend_direction',
      key: 'trend',
      width: 80,
      render: (direction: LimitUtilization['trend_direction']) => (
        <Tooltip title={`Trending ${direction}`}>
          {getTrendIcon(direction)}
        </Tooltip>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (record: LimitUtilization) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          size="small"
          onClick={() => {
            setSelectedLimit(record);
            setShowDetailModal(true);
          }}
        >
          Details
        </Button>
      )
    }
  ];

  const gaugeConfig = {
    percent: avgUtilization / 100,
    range: {
      color: ['#30BF78', '#FAAD14', '#F4664A'],
      width: 12,
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      title: {
        formatter: () => 'Avg Utilization',
        style: {
          fontSize: '14px',
          color: '#666',
        },
      },
      content: {
        formatter: () => `${avgUtilization.toFixed(1)}%`,
        style: {
          fontSize: '24px',
          fontWeight: 'bold',
          color: getUtilizationColor(avgUtilization),
        },
      },
    },
  };

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Limits"
              value={limitUtilization.length}
              prefix={<DashboardOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Breached Limits"
              value={breachedLimits.length}
              prefix={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: breachedLimits.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Warning Limits"
              value={warningLimits.length}
              prefix={<WarningOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: warningLimits.length > 0 ? '#faad14' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '14px', color: '#666' }}>Connection Status</div>
                <div style={{ fontSize: '20px', fontWeight: 'bold', color: isConnected ? '#52c41a' : '#ff4d4f' }}>
                  {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                </div>
              </div>
              <ThunderboltOutlined style={{ fontSize: '24px', color: isConnected ? '#52c41a' : '#ff4d4f' }} />
            </div>
          </Card>
        </Col>
      </Row>

      {/* Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Text>View Mode:</Text>
              <Select value={viewMode} onChange={setViewMode} style={{ width: 100 }}>
                <Option value="grid">Grid</Option>
                <Option value="table">Table</Option>
                <Option value="chart">Chart</Option>
              </Select>
              
              <Divider type="vertical" />
              
              <Text>Filter:</Text>
              <Select value={filterStatus} onChange={setFilterStatus} style={{ width: 120 }}>
                <Option value="all">All Limits</Option>
                <Option value="breach">Breached</Option>
                <Option value="critical">Critical</Option>
                <Option value="warning">Warning</Option>
                <Option value="normal">Normal</Option>
              </Select>
              
              <Divider type="vertical" />
              
              <Tooltip title="Real-time Updates">
                <Badge
                  status={isConnected ? 'processing' : 'error'}
                  text={isConnected ? 'Live' : 'Offline'}
                />
              </Tooltip>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchCurrentMetrics}
                loading={loading.metrics}
              >
                Refresh
              </Button>
              <Button icon={<SettingOutlined />}>
                Configure
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Limit Monitoring Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={clearError}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <Alert
          message={`${criticalAlerts.length} Critical Alert${criticalAlerts.length > 1 ? 's' : ''}`}
          description="Immediate attention required for risk limit breaches"
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" type="primary" danger>
              View Alerts
            </Button>
          }
        />
      )}

      <Row gutter={16}>
        <Col span={18}>
          {/* Main Content */}
          {viewMode === 'grid' && (
            <Row gutter={16}>
              {filteredLimits.map((limit) => (
                <Col span={8} key={limit.limit_id} style={{ marginBottom: 16 }}>
                  <Card
                    size="small"
                    title={
                      <Space>
                        {getStatusIcon(limit.status)}
                        <Text strong style={{ fontSize: '14px' }}>{limit.limit_name}</Text>
                      </Space>
                    }
                    extra={
                      <Space>
                        {getTrendIcon(limit.trend_direction)}
                        <Button
                          type="text"
                          icon={<EyeOutlined />}
                          size="small"
                          onClick={() => {
                            setSelectedLimit(limit);
                            setShowDetailModal(true);
                          }}
                        />
                      </Space>
                    }
                    className={limit.status === 'breach' ? 'limit-breached' : limit.status === 'critical' ? 'limit-critical' : ''}
                  >
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text style={{ fontSize: '12px' }}>
                          {limit.current_value.toLocaleString()}
                        </Text>
                        <Text style={{ fontSize: '12px' }}>
                          {limit.threshold_value.toLocaleString()}
                        </Text>
                      </div>
                      
                      <Progress
                        percent={limit.utilization_percentage}
                        strokeColor={getUtilizationColor(limit.utilization_percentage)}
                        trailColor="#f0f0f0"
                        size="small"
                        format={(percent) => `${percent?.toFixed(1)}%`}
                      />
                      
                      <Row>
                        <Col span={12}>
                          <div style={{ fontSize: '11px', color: '#666' }}>
                            Breach Risk: {(limit.breach_probability * 100).toFixed(0)}%
                          </div>
                        </Col>
                        <Col span={12} style={{ textAlign: 'right' }}>
                          {limit.time_to_breach_estimate && (
                            <div style={{ fontSize: '11px', color: '#666' }}>
                              ETA: {limit.time_to_breach_estimate}m
                            </div>
                          )}
                        </Col>
                      </Row>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          )}

          {viewMode === 'table' && (
            <Card title="Limit Utilization Details">
              <Table
                dataSource={filteredLimits}
                columns={columns}
                rowKey="limit_id"
                loading={loading.metrics}
                size="small"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: false,
                  showQuickJumper: true
                }}
                rowClassName={(record) => {
                  if (record.status === 'breach') return 'limit-breached';
                  if (record.status === 'critical') return 'limit-critical';
                  return '';
                }}
              />
            </Card>
          )}

          {viewMode === 'chart' && selectedLimit && (
            <Card title={`${selectedLimit.limit_name} - Utilization Trend`}>
              <div style={{ height: 400 }}>
                <Line
                  data={generateTrendData(selectedLimit)}
                  xField="time"
                  yField="utilization"
                  smooth={true}
                  color="#1890ff"
                  point={{ size: 3, shape: 'circle' }}
                  annotations={[
                    {
                      type: 'line',
                      start: ['min', 100],
                      end: ['max', 100],
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
                      start: ['min', selectedLimit.warning_threshold / selectedLimit.threshold_value * 100],
                      end: ['max', selectedLimit.warning_threshold / selectedLimit.threshold_value * 100],
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
                  ]}
                />
              </div>
            </Card>
          )}
        </Col>

        <Col span={6}>
          {/* Sidebar */}
          
          {/* Average Utilization Gauge */}
          <Card title="Overall Utilization" size="small" style={{ marginBottom: 16 }}>
            <div style={{ height: 200 }}>
              <Gauge {...gaugeConfig} />
            </div>
          </Card>

          {/* System Health */}
          <Card title="System Health" size="small" style={{ marginBottom: 16 }}>
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="Risk Engine">
                <Tag color={systemHealth.risk_engine_status === 'healthy' ? 'success' : 'error'}>
                  {systemHealth.risk_engine_status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Data Feed">
                <Tag color={systemHealth.data_feed_status === 'healthy' ? 'success' : 'warning'}>
                  {systemHealth.data_feed_status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Latency">
                {systemHealth.calculation_latency_ms}ms
              </Descriptions.Item>
              <Descriptions.Item label="Uptime">
                {systemHealth.uptime_percentage.toFixed(1)}%
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {/* Recent Limit Changes */}
          <Card title="Recent Changes" size="small">
            <Timeline size="small">
              {limitUtilization
                .filter(limit => limit.trend_direction !== 'stable')
                .slice(0, 5)
                .map((limit, index) => (
                  <Timeline.Item
                    key={index}
                    color={limit.trend_direction === 'up' ? 'red' : 'green'}
                  >
                    <div style={{ fontSize: '12px' }}>
                      <div style={{ fontWeight: 'bold' }}>{limit.limit_name}</div>
                      <div style={{ color: '#666' }}>
                        {limit.utilization_percentage.toFixed(1)}% utilized
                      </div>
                      <div style={{ color: limit.trend_direction === 'up' ? '#ff4d4f' : '#52c41a' }}>
                        {limit.trend_direction === 'up' ? 'Increasing' : 'Decreasing'}
                      </div>
                    </div>
                  </Timeline.Item>
                ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      {/* Limit Detail Modal */}
      <Modal
        title={
          <Space>
            <LineChartOutlined />
            {selectedLimit?.limit_name} Details
          </Space>
        }
        open={showDetailModal}
        onCancel={() => {
          setShowDetailModal(false);
          setSelectedLimit(null);
        }}
        width={800}
        footer={null}
      >
        {selectedLimit && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Statistic
                  title="Current Utilization"
                  value={selectedLimit.utilization_percentage}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getUtilizationColor(selectedLimit.utilization_percentage) }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Breach Probability"
                  value={selectedLimit.breach_probability * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: getUtilizationColor(selectedLimit.breach_probability * 100) }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Time to Breach"
                  value={selectedLimit.time_to_breach_estimate || 0}
                  suffix="minutes"
                  valueStyle={{ 
                    color: selectedLimit.time_to_breach_estimate && selectedLimit.time_to_breach_estimate < 30 
                      ? '#ff4d4f' : '#666' 
                  }}
                />
              </Col>
            </Row>

            <Divider>Limit Configuration</Divider>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Limit Type" span={2}>
                {selectedLimit.limit_type.toUpperCase()}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={
                  selectedLimit.status === 'breach' ? 'error' :
                  selectedLimit.status === 'critical' ? 'volcano' :
                  selectedLimit.status === 'warning' ? 'warning' : 'success'
                }>
                  {selectedLimit.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Current Value" span={2}>
                {selectedLimit.current_value.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Threshold">
                {selectedLimit.threshold_value.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Warning Level" span={3}>
                {selectedLimit.warning_threshold.toLocaleString()}
              </Descriptions.Item>
            </Descriptions>

            <Divider>Utilization Trend</Divider>
            <div style={{ height: 300 }}>
              <Line
                data={generateTrendData(selectedLimit)}
                xField="time"
                yField="utilization"
                smooth={true}
                color="#1890ff"
                point={{ size: 3, shape: 'circle' }}
                annotations={[
                  {
                    type: 'line',
                    start: ['min', 100],
                    end: ['max', 100],
                    style: {
                      stroke: '#ff4d4f',
                      lineDash: [4, 4],
                    }
                  },
                  {
                    type: 'line',
                    start: ['min', (selectedLimit.warning_threshold / selectedLimit.threshold_value) * 100],
                    end: ['max', (selectedLimit.warning_threshold / selectedLimit.threshold_value) * 100],
                    style: {
                      stroke: '#faad14',
                      lineDash: [2, 2],
                    }
                  }
                ]}
              />
            </div>
          </div>
        )}
      </Modal>

      <style jsx>{`
        .limit-breached {
          border-left: 4px solid #ff4d4f !important;
          background-color: #fff2f0 !important;
        }
        .limit-critical {
          border-left: 4px solid #fa8c16 !important;
          background-color: #fff7e6 !important;
        }
      `}</style>
    </div>
  );
};

export default LimitMonitor;