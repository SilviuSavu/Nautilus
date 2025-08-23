import React, { useState, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Alert,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Timeline,
  Badge,
  Tooltip,
  Statistic,
  List,
  Avatar,
  Popconfirm,
  Descriptions,
  Divider,
  Tabs,
  Drawer,
  Steps,
  Progress,
  Rate
} from 'antd';
import {
  BellOutlined,
  WarningOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  UserOutlined,
  SendOutlined,
  SettingOutlined,
  FilterOutlined,
  EyeOutlined,
  DeleteOutlined,
  PhoneOutlined,
  MailOutlined,
  MessageOutlined,
  ThunderboltOutlined,
  TeamOutlined,
  RocketOutlined,
  AlertOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import { RiskAlert } from './types/riskTypes';
import { useRiskMonitoring } from '../../hooks/risk/useRiskMonitoring';

dayjs.extend(relativeTime);

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { Step } = Steps;

interface RiskAlertCenterProps {
  portfolioId: string;
  className?: string;
}

interface EscalationLevel {
  level: number;
  name: string;
  description: string;
  contacts: string[];
  auto_escalate_minutes: number;
  notification_methods: ('email' | 'sms' | 'webhook' | 'call')[];
}

const RiskAlertCenter: React.FC<RiskAlertCenterProps> = ({
  portfolioId,
  className
}) => {
  const {
    alerts,
    criticalAlerts,
    monitoringConfig,
    loading,
    error,
    acknowledgeAlert,
    clearError
  } = useRiskMonitoring({ portfolioId });

  const [selectedAlert, setSelectedAlert] = useState<RiskAlert | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showEscalationDrawer, setShowEscalationDrawer] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [escalationForm] = Form.useForm();

  // Mock escalation levels - would be configured in backend
  const escalationLevels: EscalationLevel[] = [
    {
      level: 1,
      name: 'Risk Analyst',
      description: 'First line response team',
      contacts: ['analyst@company.com'],
      auto_escalate_minutes: 10,
      notification_methods: ['email']
    },
    {
      level: 2,
      name: 'Risk Manager',
      description: 'Senior risk management team',
      contacts: ['manager@company.com'],
      auto_escalate_minutes: 20,
      notification_methods: ['email', 'sms']
    },
    {
      level: 3,
      name: 'Chief Risk Officer',
      description: 'Executive level escalation',
      contacts: ['cro@company.com'],
      auto_escalate_minutes: 30,
      notification_methods: ['email', 'sms', 'call']
    },
    {
      level: 4,
      name: 'Emergency Response',
      description: 'Critical system shutdown protocols',
      contacts: ['emergency@company.com'],
      auto_escalate_minutes: 0,
      notification_methods: ['email', 'sms', 'call', 'webhook']
    }
  ];

  const getSeverityColor = (severity: RiskAlert['severity']) => {
    switch (severity) {
      case 'critical': return '#ff4d4f';
      case 'warning': return '#faad14';
      case 'info': return '#1890ff';
      default: return '#52c41a';
    }
  };

  const getSeverityIcon = (severity: RiskAlert['severity']) => {
    switch (severity) {
      case 'critical': return <ExclamationCircleOutlined />;
      case 'warning': return <WarningOutlined />;
      case 'info': return <BellOutlined />;
      default: return <CheckCircleOutlined />;
    }
  };

  const getAlertTypeIcon = (alertType: RiskAlert['alert_type']) => {
    switch (alertType) {
      case 'limit_breach': return <ThunderboltOutlined />;
      case 'concentration_risk': return <AlertOutlined />;
      case 'correlation_spike': return <RocketOutlined />;
      case 'var_exceeded': return <WarningOutlined />;
      default: return <BellOutlined />;
    }
  };

  const handleAcknowledgeAlert = useCallback(async (alertId: string, acknowledgedBy: string) => {
    try {
      await acknowledgeAlert(alertId, acknowledgedBy);
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  }, [acknowledgeAlert]);

  const handleEscalateAlert = useCallback(async (alert: RiskAlert, level: number) => {
    try {
      // Mock escalation API call - would be replaced with actual backend
      console.log(`Escalating alert ${alert.id} to level ${level}`);
      
      // Simulate escalation notification
      const escalationLevel = escalationLevels.find(l => l.level === level);
      if (escalationLevel) {
        // Send notifications to contacts at this level
        escalationLevel.contacts.forEach(contact => {
          console.log(`Notifying ${contact} via ${escalationLevel.notification_methods.join(', ')}`);
        });
      }
    } catch (error) {
      console.error('Error escalating alert:', error);
    }
  }, [escalationLevels]);

  const filteredAlerts = alerts.filter(alert => {
    if (filterSeverity !== 'all' && alert.severity !== filterSeverity) return false;
    if (filterStatus !== 'all') {
      if (filterStatus === 'acknowledged' && !alert.acknowledged) return false;
      if (filterStatus === 'unacknowledged' && alert.acknowledged) return false;
    }
    return true;
  });

  const columns = [
    {
      title: 'Alert',
      key: 'alert',
      width: 250,
      render: (record: RiskAlert) => (
        <Space>
          <Avatar
            icon={getAlertTypeIcon(record.alert_type)}
            style={{ backgroundColor: getSeverityColor(record.severity) }}
          />
          <Space direction="vertical" size={0}>
            <Text strong>{record.alert_type.replace('_', ' ').toUpperCase()}</Text>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Portfolio: {record.portfolio_id}
            </Text>
          </Space>
        </Space>
      )
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      width: 300,
      render: (message: string) => (
        <Paragraph
          ellipsis={{ rows: 2, expandable: true, symbol: 'more' }}
          style={{ margin: 0, fontSize: '13px' }}
        >
          {message}
        </Paragraph>
      )
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: RiskAlert['severity']) => (
        <Tag
          icon={getSeverityIcon(severity)}
          color={
            severity === 'critical' ? 'error' :
            severity === 'warning' ? 'warning' :
            severity === 'info' ? 'processing' : 'default'
          }
        >
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Triggered',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      width: 150,
      render: (date: Date) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: '12px' }}>
            {dayjs(date).format('MMM D, h:mm A')}
          </Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {dayjs(date).fromNow()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Status',
      key: 'status',
      width: 120,
      render: (record: RiskAlert) => (
        <Space direction="vertical" size={0}>
          <Tag color={record.acknowledged ? 'success' : 'warning'}>
            {record.acknowledged ? 'Acknowledged' : 'Pending'}
          </Tag>
          {record.acknowledged && record.acknowledged_by && (
            <Text type="secondary" style={{ fontSize: '11px' }}>
              by {record.acknowledged_by}
            </Text>
          )}
        </Space>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (record: RiskAlert) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              size="small"
              onClick={() => {
                setSelectedAlert(record);
                setShowDetailModal(true);
              }}
            />
          </Tooltip>
          
          {!record.acknowledged && (
            <Tooltip title="Acknowledge">
              <Button
                type="text"
                icon={<CheckCircleOutlined />}
                size="small"
                onClick={() => handleAcknowledgeAlert(record.id, 'current_user')}
              />
            </Tooltip>
          )}
          
          {record.severity === 'critical' && (
            <Tooltip title="Escalate">
              <Button
                type="text"
                icon={<RocketOutlined />}
                size="small"
                danger
                onClick={() => {
                  setSelectedAlert(record);
                  setShowEscalationDrawer(true);
                }}
              />
            </Tooltip>
          )}
          
          <Popconfirm
            title="Are you sure you want to dismiss this alert?"
            onConfirm={() => console.log('Dismiss alert:', record.id)}
          >
            <Tooltip title="Dismiss">
              <Button
                type="text"
                icon={<DeleteOutlined />}
                size="small"
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const getEscalationProgress = (alert: RiskAlert) => {
    if (!alert.triggered_at) return 0;
    
    const minutesElapsed = dayjs().diff(dayjs(alert.triggered_at), 'minute');
    const totalEscalationTime = escalationLevels.reduce((sum, level) => sum + level.auto_escalate_minutes, 0);
    
    return Math.min(100, (minutesElapsed / totalEscalationTime) * 100);
  };

  const getCurrentEscalationLevel = (alert: RiskAlert) => {
    if (!alert.triggered_at) return 0;
    
    const minutesElapsed = dayjs().diff(dayjs(alert.triggered_at), 'minute');
    let cumulativeTime = 0;
    
    for (let i = 0; i < escalationLevels.length; i++) {
      cumulativeTime += escalationLevels[i].auto_escalate_minutes;
      if (minutesElapsed < cumulativeTime) {
        return i;
      }
    }
    
    return escalationLevels.length - 1;
  };

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Total Alerts"
              value={alerts.length}
              prefix={<BellOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Critical Alerts"
              value={criticalAlerts.length}
              prefix={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: criticalAlerts.length > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Unacknowledged"
              value={alerts.filter(a => !a.acknowledged).length}
              prefix={<ClockCircleOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: alerts.filter(a => !a.acknowledged).length > 0 ? '#faad14' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Avg Response Time"
              value={15.3}
              precision={1}
              suffix="min"
              prefix={<UserOutlined style={{ color: '#52c41a' }} />}
            />
          </Card>
        </Col>
      </Row>

      {/* Error Alert */}
      {error && (
        <Alert
          message="Alert System Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={clearError}
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <Alert
          message={`${criticalAlerts.length} Critical Alert${criticalAlerts.length > 1 ? 's' : ''} Require Immediate Attention`}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Space>
              <Button size="small" danger onClick={() => setShowEscalationDrawer(true)}>
                Escalate All
              </Button>
              <Button size="small" type="primary">
                Acknowledge All
              </Button>
            </Space>
          }
        />
      )}

      <Row gutter={16}>
        <Col span={18}>
          {/* Main Alert Table */}
          <Card
            title={
              <Space>
                <BellOutlined />
                <Title level={4} style={{ margin: 0 }}>Risk Alert Center</Title>
                <Badge count={filteredAlerts.filter(a => !a.acknowledged).length} />
              </Space>
            }
            extra={
              <Space>
                <Select
                  value={filterSeverity}
                  onChange={setFilterSeverity}
                  style={{ width: 120 }}
                  placeholder="Severity"
                >
                  <Option value="all">All Severities</Option>
                  <Option value="critical">Critical</Option>
                  <Option value="warning">Warning</Option>
                  <Option value="info">Info</Option>
                </Select>
                
                <Select
                  value={filterStatus}
                  onChange={setFilterStatus}
                  style={{ width: 130 }}
                  placeholder="Status"
                >
                  <Option value="all">All Status</Option>
                  <Option value="acknowledged">Acknowledged</Option>
                  <Option value="unacknowledged">Unacknowledged</Option>
                </Select>
                
                <Button icon={<SettingOutlined />}>
                  Configure
                </Button>
              </Space>
            }
          >
            <Table
              dataSource={filteredAlerts}
              columns={columns}
              rowKey="id"
              loading={loading.alerts}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total) => `Total ${total} alerts`
              }}
              rowClassName={(record) => {
                if (record.severity === 'critical' && !record.acknowledged) return 'alert-critical';
                if (record.severity === 'warning' && !record.acknowledged) return 'alert-warning';
                return '';
              }}
            />
          </Card>
        </Col>

        <Col span={6}>
          {/* Escalation Timeline */}
          <Card title="Escalation Process" size="small" style={{ marginBottom: 16 }}>
            <Steps direction="vertical" size="small" current={-1}>
              {escalationLevels.map((level, index) => (
                <Step
                  key={index}
                  title={level.name}
                  description={`${level.auto_escalate_minutes}min`}
                  icon={
                    index === 0 ? <UserOutlined /> :
                    index === 1 ? <TeamOutlined /> :
                    index === 2 ? <PhoneOutlined /> :
                    <RocketOutlined />
                  }
                />
              ))}
            </Steps>
          </Card>

          {/* Recent Activity */}
          <Card title="Recent Activity" size="small">
            <Timeline size="small">
              {alerts.slice(0, 5).map((alert, index) => (
                <Timeline.Item
                  key={index}
                  color={getSeverityColor(alert.severity)}
                  dot={getSeverityIcon(alert.severity)}
                >
                  <div style={{ fontSize: '12px' }}>
                    <div style={{ fontWeight: 'bold' }}>
                      {alert.alert_type.replace('_', ' ')}
                    </div>
                    <div style={{ color: '#666' }}>
                      {dayjs(alert.triggered_at).fromNow()}
                    </div>
                    <div style={{ color: alert.acknowledged ? '#52c41a' : '#ff4d4f' }}>
                      {alert.acknowledged ? 'Acknowledged' : 'Pending'}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      {/* Alert Detail Modal */}
      <Modal
        title={
          <Space>
            <Avatar
              icon={selectedAlert ? getAlertTypeIcon(selectedAlert.alert_type) : <BellOutlined />}
              style={{ backgroundColor: selectedAlert ? getSeverityColor(selectedAlert.severity) : '#1890ff' }}
            />
            Alert Details
          </Space>
        }
        open={showDetailModal}
        onCancel={() => {
          setShowDetailModal(false);
          setSelectedAlert(null);
        }}
        width={700}
        footer={[
          <Button key="escalate" danger onClick={() => {
            setShowDetailModal(false);
            setShowEscalationDrawer(true);
          }}>
            Escalate
          </Button>,
          <Button key="acknowledge" type="primary" onClick={() => {
            if (selectedAlert) {
              handleAcknowledgeAlert(selectedAlert.id, 'current_user');
              setShowDetailModal(false);
              setSelectedAlert(null);
            }
          }}>
            Acknowledge
          </Button>,
          <Button key="close" onClick={() => {
            setShowDetailModal(false);
            setSelectedAlert(null);
          }}>
            Close
          </Button>
        ]}
      >
        {selectedAlert && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Statistic
                  title="Severity"
                  value={selectedAlert.severity.toUpperCase()}
                  valueStyle={{ color: getSeverityColor(selectedAlert.severity) }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Time Elapsed"
                  value={dayjs().diff(dayjs(selectedAlert.triggered_at), 'minute')}
                  suffix="min"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Escalation Progress"
                  value={getEscalationProgress(selectedAlert)}
                  precision={0}
                  suffix="%"
                />
              </Col>
            </Row>

            <Divider>Alert Information</Divider>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Alert Type" span={2}>
                {selectedAlert.alert_type.replace('_', ' ').toUpperCase()}
              </Descriptions.Item>
              <Descriptions.Item label="Portfolio">
                {selectedAlert.portfolio_id}
              </Descriptions.Item>
              <Descriptions.Item label="Triggered At" span={3}>
                {dayjs(selectedAlert.triggered_at).format('YYYY-MM-DD HH:mm:ss')}
              </Descriptions.Item>
              <Descriptions.Item label="Message" span={3}>
                {selectedAlert.message}
              </Descriptions.Item>
            </Descriptions>

            {selectedAlert.acknowledged && (
              <>
                <Divider>Acknowledgment</Divider>
                <Descriptions bordered size="small">
                  <Descriptions.Item label="Acknowledged By" span={2}>
                    {selectedAlert.acknowledged_by}
                  </Descriptions.Item>
                  <Descriptions.Item label="Time">
                    {selectedAlert.acknowledged_at ? 
                      dayjs(selectedAlert.acknowledged_at).format('YYYY-MM-DD HH:mm:ss') : 'N/A'}
                  </Descriptions.Item>
                </Descriptions>
              </>
            )}

            {selectedAlert.metadata && Object.keys(selectedAlert.metadata).length > 0 && (
              <>
                <Divider>Additional Information</Divider>
                <Descriptions bordered size="small">
                  {Object.entries(selectedAlert.metadata).map(([key, value]) => (
                    <Descriptions.Item key={key} label={key} span={3}>
                      {JSON.stringify(value)}
                    </Descriptions.Item>
                  ))}
                </Descriptions>
              </>
            )}

            <Divider>Escalation Status</Divider>
            <div style={{ marginBottom: 16 }}>
              <Text>Current Level: {getCurrentEscalationLevel(selectedAlert) + 1}</Text>
              <Progress
                percent={getEscalationProgress(selectedAlert)}
                strokeColor={getEscalationProgress(selectedAlert) > 75 ? '#ff4d4f' : '#1890ff'}
              />
            </div>
          </div>
        )}
      </Modal>

      {/* Escalation Drawer */}
      <Drawer
        title="Escalate Alert"
        placement="right"
        width={500}
        onClose={() => setShowEscalationDrawer(false)}
        open={showEscalationDrawer}
      >
        <Form
          form={escalationForm}
          layout="vertical"
          onFinish={(values) => {
            if (selectedAlert) {
              handleEscalateAlert(selectedAlert, values.level);
            }
            setShowEscalationDrawer(false);
            escalationForm.resetFields();
          }}
        >
          <Alert
            message="Alert Escalation"
            description="Select the escalation level and add any additional context for the escalation team."
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Form.Item
            name="level"
            label="Escalation Level"
            rules={[{ required: true, message: 'Please select escalation level' }]}
          >
            <Select placeholder="Select escalation level">
              {escalationLevels.map((level) => (
                <Option key={level.level} value={level.level}>
                  Level {level.level} - {level.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item label="Escalation Details">
            <List
              size="small"
              dataSource={escalationLevels}
              renderItem={(level) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar style={{ backgroundColor: level.level === 4 ? '#ff4d4f' : '#1890ff' }}>
                        {level.level}
                      </Avatar>
                    }
                    title={level.name}
                    description={
                      <div>
                        <div>{level.description}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          Auto-escalate: {level.auto_escalate_minutes}min | 
                          Methods: {level.notification_methods.join(', ')}
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Form.Item>

          <Form.Item
            name="message"
            label="Additional Context"
            rules={[{ required: true, message: 'Please provide escalation context' }]}
          >
            <TextArea
              rows={4}
              placeholder="Describe the urgency and any actions already taken..."
            />
          </Form.Item>

          <Form.Item name="priority" label="Priority Level">
            <Rate />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowEscalationDrawer(false);
                escalationForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit" danger>
                Escalate Alert
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Drawer>

      <style jsx>{`
        .alert-critical {
          background-color: #fff2f0 !important;
          border-left: 4px solid #ff4d4f !important;
        }
        .alert-warning {
          background-color: #fff7e6 !important;
          border-left: 4px solid #faad14 !important;
        }
      `}</style>
    </div>
  );
};

export default RiskAlertCenter;