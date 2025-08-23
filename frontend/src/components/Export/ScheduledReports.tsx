/**
 * Story 5.3: Scheduled Reports Component
 * Management interface for automated report scheduling
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Typography,
  Row,
  Col,
  Switch,
  Modal,
  notification,
  Tooltip,
  Badge,
  Progress,
  Alert,
  Statistic,
  List,
  Avatar,
  Descriptions
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  EditOutlined,
  DeleteOutlined,
  ClockCircleOutlined,
  MailOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  HistoryOutlined,
  CalendarOutlined,
  UserOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  ReportTemplate,
  ReportType,
  ScheduleFrequency,
  ReportSchedule
} from '../../types/export';

const { Title, Text } = Typography;

interface ScheduledReportExecution {
  id: string;
  template_id: string;
  template_name: string;
  scheduled_at: Date;
  executed_at?: Date;
  status: 'pending' | 'running' | 'completed' | 'failed';
  file_size?: string;
  recipients: string[];
  error_message?: string;
}

interface ScheduledReportsProps {
  className?: string;
}

export const ScheduledReports: React.FC<ScheduledReportsProps> = ({
  className
}) => {
  const [scheduledTemplates, setScheduledTemplates] = useState<ReportTemplate[]>([]);
  const [recentExecutions, setRecentExecutions] = useState<ScheduledReportExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyVisible, setHistoryVisible] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null);

  // Mock scheduled templates
  const mockScheduledTemplates: ReportTemplate[] = [
    {
      id: 'template-001',
      name: 'Daily Performance Report',
      description: 'Comprehensive daily trading performance analysis',
      type: ReportType.PERFORMANCE,
      format: 'pdf' as any,
      sections: [],
      parameters: [],
      schedule: {
        frequency: ScheduleFrequency.DAILY,
        time: '08:00',
        timezone: 'UTC',
        recipients: ['trader@example.com', 'manager@example.com']
      },
      created_at: new Date('2024-01-15'),
      updated_at: new Date('2024-01-20')
    },
    {
      id: 'template-002',
      name: 'Weekly Risk Assessment',
      description: 'Weekly risk metrics and exposure analysis',
      type: ReportType.RISK,
      format: 'excel' as any,
      sections: [],
      parameters: [],
      schedule: {
        frequency: ScheduleFrequency.WEEKLY,
        time: '09:00',
        timezone: 'America/New_York',
        recipients: ['risk@example.com']
      },
      created_at: new Date('2024-01-10'),
      updated_at: new Date('2024-01-18')
    },
    {
      id: 'template-003',
      name: 'Monthly Compliance Report',
      description: 'Monthly regulatory compliance and audit report',
      type: ReportType.COMPLIANCE,
      format: 'pdf' as any,
      sections: [],
      parameters: [],
      schedule: {
        frequency: ScheduleFrequency.MONTHLY,
        time: '07:00',
        timezone: 'UTC',
        recipients: ['compliance@example.com', 'legal@example.com']
      },
      created_at: new Date('2024-01-05'),
      updated_at: new Date('2024-01-15')
    }
  ];

  // Mock recent executions
  const mockRecentExecutions: ScheduledReportExecution[] = [
    {
      id: 'exec-001',
      template_id: 'template-001',
      template_name: 'Daily Performance Report',
      scheduled_at: new Date('2024-01-20T08:00:00Z'),
      executed_at: new Date('2024-01-20T08:02:15Z'),
      status: 'completed',
      file_size: '2.3 MB',
      recipients: ['trader@example.com', 'manager@example.com']
    },
    {
      id: 'exec-002',
      template_id: 'template-002',
      template_name: 'Weekly Risk Assessment',
      scheduled_at: new Date('2024-01-19T09:00:00Z'),
      executed_at: new Date('2024-01-19T09:01:45Z'),
      status: 'completed',
      file_size: '1.8 MB',
      recipients: ['risk@example.com']
    },
    {
      id: 'exec-003',
      template_id: 'template-001',
      template_name: 'Daily Performance Report',
      scheduled_at: new Date('2024-01-19T08:00:00Z'),
      status: 'failed',
      recipients: ['trader@example.com', 'manager@example.com'],
      error_message: 'Data source unavailable during scheduled execution'
    },
    {
      id: 'exec-004',
      template_id: 'template-001',
      template_name: 'Daily Performance Report',
      scheduled_at: new Date('2024-01-21T08:00:00Z'),
      status: 'pending',
      recipients: ['trader@example.com', 'manager@example.com']
    },
    {
      id: 'exec-005',
      template_id: 'template-001',
      template_name: 'Daily Performance Report',
      scheduled_at: new Date('2024-01-20T08:00:00Z'),
      status: 'running',
      recipients: ['trader@example.com', 'manager@example.com']
    }
  ];

  useEffect(() => {
    // Initialize with mock data
    setScheduledTemplates(mockScheduledTemplates);
    setRecentExecutions(mockRecentExecutions);
  }, []);

  const toggleSchedule = (templateId: string, enabled: boolean) => {
    // In a real implementation, this would update the schedule status
    notification.info({
      message: 'Schedule Updated',
      description: `Schedule has been ${enabled ? 'enabled' : 'disabled'} for the selected template`,
    });
  };

  const runReportNow = (template: ReportTemplate) => {
    notification.info({
      message: 'Report Generation Started',
      description: `Manual execution started for ${template.name}`,
    });

    // Add a running execution to the list
    const newExecution: ScheduledReportExecution = {
      id: `exec-${Date.now()}`,
      template_id: template.id!,
      template_name: template.name,
      scheduled_at: new Date(),
      status: 'running',
      recipients: template.schedule?.recipients || []
    };
    
    setRecentExecutions(prev => [newExecution, ...prev]);

    // Simulate completion after 3 seconds
    setTimeout(() => {
      setRecentExecutions(prev => 
        prev.map(exec => 
          exec.id === newExecution.id 
            ? { ...exec, status: 'completed' as const, executed_at: new Date(), file_size: '1.5 MB' }
            : exec
        )
      );
    }, 3000);
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'processing';
      case 'pending': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'running': return <PlayCircleOutlined style={{ color: '#1890ff' }} />;
      case 'pending': return <ClockCircleOutlined style={{ color: '#faad14' }} />;
      default: return null;
    }
  };

  const getFrequencyColor = (frequency: ScheduleFrequency): string => {
    const colors = {
      [ScheduleFrequency.DAILY]: 'blue',
      [ScheduleFrequency.WEEKLY]: 'green',
      [ScheduleFrequency.MONTHLY]: 'orange',
      [ScheduleFrequency.QUARTERLY]: 'purple'
    };
    return colors[frequency] || 'default';
  };

  const getNextRun = (schedule: ReportSchedule): Date => {
    const now = dayjs();
    const [hours, minutes] = schedule.time.split(':').map(Number);
    
    let nextRun = now.hour(hours).minute(minutes).second(0);
    
    switch (schedule.frequency) {
      case ScheduleFrequency.DAILY:
        if (nextRun.isBefore(now)) {
          nextRun = nextRun.add(1, 'day');
        }
        break;
      case ScheduleFrequency.WEEKLY:
        nextRun = nextRun.day(1); // Monday
        if (nextRun.isBefore(now)) {
          nextRun = nextRun.add(1, 'week');
        }
        break;
      case ScheduleFrequency.MONTHLY:
        nextRun = nextRun.date(1); // First day of month
        if (nextRun.isBefore(now)) {
          nextRun = nextRun.add(1, 'month');
        }
        break;
      case ScheduleFrequency.QUARTERLY:
        const currentQuarter = Math.floor(now.month() / 3);
        nextRun = nextRun.month(currentQuarter * 3).date(1);
        if (nextRun.isBefore(now)) {
          nextRun = nextRun.add(3, 'month');
        }
        break;
    }
    
    return nextRun.toDate();
  };

  const scheduledColumns = [
    {
      title: 'Report Template',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: ReportTemplate) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Tag color={record.type === ReportType.PERFORMANCE ? 'blue' : 
                     record.type === ReportType.RISK ? 'red' : 
                     record.type === ReportType.COMPLIANCE ? 'orange' : 'purple'}>
            {record.type.toUpperCase()}
          </Tag>
        </div>
      )
    },
    {
      title: 'Schedule',
      key: 'schedule',
      render: (_: any, record: ReportTemplate) => (
        <div>
          <Tag color={getFrequencyColor(record.schedule!.frequency)}>
            {record.schedule!.frequency.toUpperCase()}
          </Tag>
          <br />
          <Text style={{ fontSize: 12 }}>
            <ClockCircleOutlined style={{ marginRight: 4 }} />
            {record.schedule!.time} ({record.schedule!.timezone})
          </Text>
        </div>
      )
    },
    {
      title: 'Recipients',
      key: 'recipients',
      render: (_: any, record: ReportTemplate) => (
        <div>
          <Text style={{ fontSize: 12 }}>
            <MailOutlined style={{ marginRight: 4 }} />
            {record.schedule!.recipients.length} recipients
          </Text>
          <br />
          {record.schedule!.recipients.slice(0, 2).map((email, index) => (
            <Text key={index} style={{ fontSize: 11, color: '#666' }}>
              {email}{index === 0 && record.schedule!.recipients.length > 1 ? ', ' : ''}
            </Text>
          ))}
          {record.schedule!.recipients.length > 2 && (
            <Text style={{ fontSize: 11, color: '#666' }}>
              +{record.schedule!.recipients.length - 2} more
            </Text>
          )}
        </div>
      )
    },
    {
      title: 'Next Run',
      key: 'next_run',
      render: (_: any, record: ReportTemplate) => {
        const nextRun = getNextRun(record.schedule!);
        return (
          <div>
            <Text strong style={{ fontSize: 12 }}>
              {dayjs(nextRun).format('MMM DD, YYYY')}
            </Text>
            <br />
            <Text style={{ fontSize: 11, color: '#666' }}>
              {dayjs(nextRun).format('HH:mm')}
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Status',
      key: 'status',
      render: () => (
        <Badge status="processing" text="Active" />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: ReportTemplate) => (
        <Space>
          <Tooltip title="Run Now">
            <Button
              type="text"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => runReportNow(record)}
            />
          </Tooltip>
          <Tooltip title="View History">
            <Button
              type="text"
              size="small"
              icon={<HistoryOutlined />}
              onClick={() => {
                setSelectedTemplate(record);
                setHistoryVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Disable Schedule">
            <Button
              type="text"
              size="small"
              icon={<PauseCircleOutlined />}
              onClick={() => toggleSchedule(record.id!, false)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const executionColumns = [
    {
      title: 'Template',
      dataIndex: 'template_name',
      key: 'template_name',
      render: (name: string) => <Text strong style={{ fontSize: 12 }}>{name}</Text>
    },
    {
      title: 'Scheduled',
      dataIndex: 'scheduled_at',
      key: 'scheduled_at',
      render: (date: Date) => (
        <Text style={{ fontSize: 11 }}>
          {dayjs(date).format('MMM DD, HH:mm')}
        </Text>
      )
    },
    {
      title: 'Executed',
      dataIndex: 'executed_at',
      key: 'executed_at',
      render: (date?: Date) => (
        <Text style={{ fontSize: 11 }}>
          {date ? dayjs(date).format('MMM DD, HH:mm') : '-'}
        </Text>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: ScheduledReportExecution) => (
        <div>
          <Space>
            {getStatusIcon(status)}
            <Tag color={getStatusColor(status)}>
              {status.toUpperCase()}
            </Tag>
          </Space>
          {status === 'running' && (
            <div style={{ marginTop: 4 }}>
              <Progress percent={65} size="small" showInfo={false} />
            </div>
          )}
          {status === 'failed' && record.error_message && (
            <Tooltip title={record.error_message}>
              <Text type="secondary" style={{ fontSize: 10 }}>
                Click for details
              </Text>
            </Tooltip>
          )}
        </div>
      )
    },
    {
      title: 'File Size',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size?: string) => (
        <Text style={{ fontSize: 11 }}>{size || '-'}</Text>
      )
    }
  ];

  // Statistics
  const completedToday = recentExecutions.filter(
    exec => exec.status === 'completed' && 
    dayjs(exec.executed_at).isSame(dayjs(), 'day')
  ).length;

  const failedToday = recentExecutions.filter(
    exec => exec.status === 'failed' && 
    dayjs(exec.scheduled_at).isSame(dayjs(), 'day')
  ).length;

  const pendingCount = recentExecutions.filter(exec => exec.status === 'pending').length;

  return (
    <div className={`scheduled-reports ${className || ''}`}>
      {/* Statistics Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Active Schedules"
              value={scheduledTemplates.length}
              prefix={<CalendarOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Completed Today"
              value={completedToday}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Failed Today"
              value={failedToday}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card size="small">
            <Statistic
              title="Pending"
              value={pendingCount}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        {/* Scheduled Templates */}
        <Col xs={24} lg={14}>
          <Card 
            title={
              <div>
                <CalendarOutlined style={{ marginRight: 8 }} />
                Scheduled Templates
              </div>
            }
            size="small"
          >
            <Table
              columns={scheduledColumns}
              dataSource={scheduledTemplates}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={false}
              scroll={{ y: 400 }}
            />
          </Card>
        </Col>

        {/* Recent Executions */}
        <Col xs={24} lg={10}>
          <Card 
            title={
              <div>
                <HistoryOutlined style={{ marginRight: 8 }} />
                Recent Executions
              </div>
            }
            size="small"
          >
            <List
              dataSource={recentExecutions.slice(0, 8)}
              renderItem={(execution) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={
                      <Avatar size="small" icon={getStatusIcon(execution.status)} />
                    }
                    title={
                      <Text style={{ fontSize: 12 }}>{execution.template_name}</Text>
                    }
                    description={
                      <div>
                        <Text style={{ fontSize: 10, color: '#666' }}>
                          {dayjs(execution.scheduled_at).format('MMM DD, HH:mm')}
                        </Text>
                        {execution.file_size && (
                          <Tag color="blue" style={{ marginLeft: 8 }}>
                            {execution.file_size}
                          </Tag>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* Execution History Modal */}
      <Modal
        title={
          <div>
            <HistoryOutlined style={{ marginRight: 8 }} />
            Execution History
            {selectedTemplate && ` - ${selectedTemplate.name}`}
          </div>
        }
        open={historyVisible}
        onCancel={() => setHistoryVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setHistoryVisible(false)}>
            Close
          </Button>
        ]}
      >
        {selectedTemplate && (
          <div>
            <Alert
              message="Execution History"
              description={`Showing recent execution history for "${selectedTemplate.name}"`}
              type="info"
              style={{ marginBottom: 16 }}
            />
            
            <Table
              columns={executionColumns}
              dataSource={recentExecutions.filter(exec => exec.template_id === selectedTemplate.id)}
              rowKey="id"
              size="small"
              pagination={{
                pageSize: 10,
                showSizeChanger: false
              }}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ScheduledReports;