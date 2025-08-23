import React, { useState, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Progress,
  Statistic,
  Alert,
  Tabs,
  Select,
  DatePicker,
  Descriptions,
  Badge,
  Timeline,
  List,
  Modal,
  Form,
  Input,
  Tooltip,
  Divider,
  Collapse,
  Switch,
  Rate
} from 'antd';
import {
  FileProtectOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  AuditOutlined,
  SafetyCertificateOutlined,
  DownloadOutlined,
  EyeOutlined,
  SettingOutlined,
  BankOutlined,
  GlobalOutlined,
  FileTextOutlined,
  SendOutlined,
  CalendarOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

const { Title, Text, Paragraph } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { Panel } = Collapse;
const { TextArea } = Input;

interface ComplianceReportingProps {
  portfolioId: string;
  className?: string;
}

interface ComplianceFramework {
  id: string;
  name: string;
  description: string;
  jurisdiction: string;
  requirements: ComplianceRequirement[];
  status: 'compliant' | 'non_compliant' | 'pending' | 'unknown';
  last_assessment: Date;
  next_assessment: Date;
}

interface ComplianceRequirement {
  id: string;
  framework_id: string;
  code: string;
  title: string;
  description: string;
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'compliant' | 'non_compliant' | 'pending' | 'not_applicable';
  last_checked: Date;
  evidence_required: string[];
  remediation_steps: string[];
  deadline?: Date;
}

interface ComplianceReport {
  id: string;
  framework_id: string;
  framework_name: string;
  report_type: 'assessment' | 'gap_analysis' | 'remediation' | 'periodic';
  generated_at: Date;
  period_start: Date;
  period_end: Date;
  status: 'draft' | 'review' | 'approved' | 'submitted';
  compliance_score: number;
  total_requirements: number;
  compliant_requirements: number;
  violations: number;
  file_url?: string;
}

const ComplianceReporting: React.FC<ComplianceReportingProps> = ({
  portfolioId,
  className
}) => {
  const [selectedFramework, setSelectedFramework] = useState<string>('all');
  const [selectedPeriod, setSelectedPeriod] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'month'),
    dayjs()
  ]);
  const [showReportModal, setShowReportModal] = useState(false);
  const [showAssessmentModal, setShowAssessmentModal] = useState(false);
  const [selectedReport, setSelectedReport] = useState<ComplianceReport | null>(null);
  const [reportForm] = Form.useForm();
  const [assessmentForm] = Form.useForm();

  // Mock compliance frameworks
  const frameworks: ComplianceFramework[] = [
    {
      id: 'basel-iii',
      name: 'Basel III',
      description: 'International banking regulation framework',
      jurisdiction: 'Global',
      requirements: [],
      status: 'compliant',
      last_assessment: dayjs().subtract(30, 'day').toDate(),
      next_assessment: dayjs().add(90, 'day').toDate()
    },
    {
      id: 'mifid-ii',
      name: 'MiFID II',
      description: 'Markets in Financial Instruments Directive',
      jurisdiction: 'European Union',
      requirements: [],
      status: 'non_compliant',
      last_assessment: dayjs().subtract(45, 'day').toDate(),
      next_assessment: dayjs().add(60, 'day').toDate()
    },
    {
      id: 'dodd-frank',
      name: 'Dodd-Frank',
      description: 'Wall Street Reform and Consumer Protection Act',
      jurisdiction: 'United States',
      requirements: [],
      status: 'pending',
      last_assessment: dayjs().subtract(60, 'day').toDate(),
      next_assessment: dayjs().add(30, 'day').toDate()
    },
    {
      id: 'crd-iv',
      name: 'CRD IV',
      description: 'Capital Requirements Directive',
      jurisdiction: 'European Union',
      requirements: [],
      status: 'compliant',
      last_assessment: dayjs().subtract(20, 'day').toDate(),
      next_assessment: dayjs().add(100, 'day').toDate()
    }
  ];

  // Mock compliance requirements
  const requirements: ComplianceRequirement[] = [
    {
      id: 'req-001',
      framework_id: 'basel-iii',
      code: 'CET1',
      title: 'Common Equity Tier 1 Capital Ratio',
      description: 'Minimum CET1 capital ratio of 4.5%',
      category: 'Capital Requirements',
      severity: 'critical',
      status: 'compliant',
      last_checked: dayjs().subtract(1, 'day').toDate(),
      evidence_required: ['Capital calculations', 'Regulatory reporting'],
      remediation_steps: []
    },
    {
      id: 'req-002',
      framework_id: 'mifid-ii',
      code: 'ART-25',
      title: 'Best Execution Policy',
      description: 'Implementation of best execution policies',
      category: 'Execution Quality',
      severity: 'high',
      status: 'non_compliant',
      last_checked: dayjs().subtract(3, 'day').toDate(),
      evidence_required: ['Execution policies', 'Transaction analysis'],
      remediation_steps: [
        'Update execution policy documentation',
        'Implement execution quality monitoring',
        'Train trading staff on new requirements'
      ],
      deadline: dayjs().add(30, 'day').toDate()
    },
    {
      id: 'req-003',
      framework_id: 'dodd-frank',
      code: 'VOLCKER',
      title: 'Volcker Rule Compliance',
      description: 'Prohibition on proprietary trading',
      category: 'Trading Restrictions',
      severity: 'critical',
      status: 'pending',
      last_checked: dayjs().subtract(7, 'day').toDate(),
      evidence_required: ['Trading records', 'Risk controls'],
      remediation_steps: []
    }
  ];

  // Mock compliance reports
  const reports: ComplianceReport[] = [
    {
      id: 'report-001',
      framework_id: 'basel-iii',
      framework_name: 'Basel III',
      report_type: 'periodic',
      generated_at: dayjs().subtract(2, 'day').toDate(),
      period_start: dayjs().subtract(1, 'month').toDate(),
      period_end: dayjs().toDate(),
      status: 'approved',
      compliance_score: 92,
      total_requirements: 25,
      compliant_requirements: 23,
      violations: 0,
      file_url: '/api/compliance/reports/report-001.pdf'
    },
    {
      id: 'report-002',
      framework_id: 'mifid-ii',
      framework_name: 'MiFID II',
      report_type: 'gap_analysis',
      generated_at: dayjs().subtract(5, 'day').toDate(),
      period_start: dayjs().subtract(3, 'month').toDate(),
      period_end: dayjs().toDate(),
      status: 'review',
      compliance_score: 78,
      total_requirements: 18,
      compliant_requirements: 14,
      violations: 3,
      file_url: '/api/compliance/reports/report-002.pdf'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'success';
      case 'non_compliant': return 'error';
      case 'pending': return 'warning';
      case 'unknown': return 'default';
      case 'approved': return 'success';
      case 'review': return 'processing';
      case 'draft': return 'default';
      case 'submitted': return 'blue';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant': return <CheckCircleOutlined />;
      case 'non_compliant': return <WarningOutlined />;
      case 'pending': return <ClockCircleOutlined />;
      case 'approved': return <SafetyCertificateOutlined />;
      case 'review': return <AuditOutlined />;
      default: return <FileProtectOutlined />;
    }
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

  const frameworkColumns = [
    {
      title: 'Framework',
      key: 'framework',
      width: 200,
      render: (record: ComplianceFramework) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.name}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.jurisdiction}
          </Text>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Last Assessment',
      dataIndex: 'last_assessment',
      key: 'last_assessment',
      width: 150,
      render: (date: Date) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: '13px' }}>
            {dayjs(date).format('MMM D, YYYY')}
          </Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {dayjs(date).fromNow()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Next Assessment',
      dataIndex: 'next_assessment',
      key: 'next_assessment',
      width: 150,
      render: (date: Date) => {
        const isOverdue = dayjs(date).isBefore(dayjs());
        return (
          <Space direction="vertical" size={0}>
            <Text style={{ fontSize: '13px', color: isOverdue ? '#ff4d4f' : undefined }}>
              {dayjs(date).format('MMM D, YYYY')}
            </Text>
            <Text type="secondary" style={{ fontSize: '11px' }}>
              {dayjs(date).fromNow()}
            </Text>
          </Space>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (record: ComplianceFramework) => (
        <Space>
          <Tooltip title="Generate Report">
            <Button
              type="text"
              icon={<FileTextOutlined />}
              size="small"
              onClick={() => {
                reportForm.setFieldsValue({ framework_id: record.id });
                setShowReportModal(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Run Assessment">
            <Button
              type="text"
              icon={<AuditOutlined />}
              size="small"
              onClick={() => {
                assessmentForm.setFieldsValue({ framework_id: record.id });
                setShowAssessmentModal(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Configure">
            <Button
              type="text"
              icon={<SettingOutlined />}
              size="small"
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const requirementColumns = [
    {
      title: 'Requirement',
      key: 'requirement',
      width: 250,
      render: (record: ComplianceRequirement) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Tag size="small">{record.code}</Tag>
            <Text strong style={{ fontSize: '13px' }}>{record.title}</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.category}
          </Text>
        </Space>
      )
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Badge
          color={getSeverityColor(severity)}
          text={severity.toUpperCase()}
        />
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Last Checked',
      dataIndex: 'last_checked',
      key: 'last_checked',
      width: 120,
      render: (date: Date) => (
        <Text style={{ fontSize: '12px' }}>
          {dayjs(date).fromNow()}
        </Text>
      )
    },
    {
      title: 'Deadline',
      dataIndex: 'deadline',
      key: 'deadline',
      width: 120,
      render: (date?: Date) => {
        if (!date) return <Text type="secondary">N/A</Text>;
        
        const isOverdue = dayjs(date).isBefore(dayjs());
        const isUrgent = dayjs(date).diff(dayjs(), 'day') < 7;
        
        return (
          <Text style={{ 
            fontSize: '12px',
            color: isOverdue ? '#ff4d4f' : isUrgent ? '#faad14' : undefined
          }}>
            {dayjs(date).format('MMM D')}
          </Text>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (record: ComplianceRequirement) => (
        <Button
          type="text"
          icon={<EyeOutlined />}
          size="small"
          onClick={() => {
            Modal.info({
              title: record.title,
              width: 600,
              content: (
                <div>
                  <Descriptions bordered size="small" column={1}>
                    <Descriptions.Item label="Code">{record.code}</Descriptions.Item>
                    <Descriptions.Item label="Description">
                      {record.description}
                    </Descriptions.Item>
                    <Descriptions.Item label="Category">{record.category}</Descriptions.Item>
                    <Descriptions.Item label="Severity">
                      <Badge color={getSeverityColor(record.severity)} text={record.severity} />
                    </Descriptions.Item>
                  </Descriptions>
                  
                  {record.remediation_steps.length > 0 && (
                    <>
                      <Divider>Remediation Steps</Divider>
                      <List
                        size="small"
                        dataSource={record.remediation_steps}
                        renderItem={(step, index) => (
                          <List.Item>
                            {index + 1}. {step}
                          </List.Item>
                        )}
                      />
                    </>
                  )}
                </div>
              )
            });
          }}
        />
      )
    }
  ];

  const reportColumns = [
    {
      title: 'Report',
      key: 'report',
      width: 200,
      render: (record: ComplianceReport) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.framework_name}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.report_type.replace('_', ' ').toUpperCase()}
          </Text>
        </Space>
      )
    },
    {
      title: 'Compliance Score',
      dataIndex: 'compliance_score',
      key: 'compliance_score',
      width: 150,
      render: (score: number) => (
        <Space>
          <Progress
            type="circle"
            percent={score}
            width={40}
            strokeColor={score >= 90 ? '#52c41a' : score >= 70 ? '#faad14' : '#ff4d4f'}
            format={() => `${score}%`}
            size="small"
          />
          <div>
            <div style={{ fontSize: '13px', fontWeight: 'bold' }}>{score}%</div>
          </div>
        </Space>
      )
    },
    {
      title: 'Requirements',
      key: 'requirements',
      width: 150,
      render: (record: ComplianceReport) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: '13px' }}>
            {record.compliant_requirements}/{record.total_requirements} compliant
          </Text>
          {record.violations > 0 && (
            <Text type="danger" style={{ fontSize: '12px' }}>
              {record.violations} violation{record.violations > 1 ? 's' : ''}
            </Text>
          )}
        </Space>
      )
    },
    {
      title: 'Generated',
      dataIndex: 'generated_at',
      key: 'generated_at',
      width: 120,
      render: (date: Date) => (
        <Text style={{ fontSize: '12px' }}>
          {dayjs(date).format('MMM D, YYYY')}
        </Text>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (record: ComplianceReport) => (
        <Space>
          <Tooltip title="Download Report">
            <Button
              type="text"
              icon={<DownloadOutlined />}
              size="small"
              disabled={!record.file_url}
            />
          </Tooltip>
          <Tooltip title="View Details">
            <Button
              type="text"
              icon={<EyeOutlined />}
              size="small"
              onClick={() => setSelectedReport(record)}
            />
          </Tooltip>
          {record.status === 'draft' && (
            <Tooltip title="Submit Report">
              <Button
                type="text"
                icon={<SendOutlined />}
                size="small"
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ];

  const filteredFrameworks = selectedFramework === 'all' 
    ? frameworks 
    : frameworks.filter(f => f.id === selectedFramework);

  const filteredRequirements = selectedFramework === 'all'
    ? requirements
    : requirements.filter(r => r.framework_id === selectedFramework);

  const complianceMetrics = {
    totalFrameworks: frameworks.length,
    compliantFrameworks: frameworks.filter(f => f.status === 'compliant').length,
    nonCompliantFrameworks: frameworks.filter(f => f.status === 'non_compliant').length,
    pendingAssessments: frameworks.filter(f => f.status === 'pending').length,
    overallCompliance: Math.round(
      (frameworks.filter(f => f.status === 'compliant').length / frameworks.length) * 100
    )
  };

  return (
    <div className={className}>
      {/* Header Statistics */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Overall Compliance"
              value={complianceMetrics.overallCompliance}
              precision={0}
              suffix="%"
              prefix={<SafetyCertificateOutlined style={{ color: complianceMetrics.overallCompliance >= 80 ? '#52c41a' : '#faad14' }} />}
              valueStyle={{ color: complianceMetrics.overallCompliance >= 80 ? '#52c41a' : '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Active Frameworks"
              value={complianceMetrics.totalFrameworks}
              prefix={<GlobalOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Non-Compliant"
              value={complianceMetrics.nonCompliantFrameworks}
              prefix={<WarningOutlined style={{ color: '#ff4d4f' }} />}
              valueStyle={{ color: complianceMetrics.nonCompliantFrameworks > 0 ? '#ff4d4f' : undefined }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="Pending Reviews"
              value={complianceMetrics.pendingAssessments}
              prefix={<ClockCircleOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: complianceMetrics.pendingAssessments > 0 ? '#faad14' : undefined }}
            />
          </Card>
        </Col>
      </Row>

      {/* Non-compliance Alert */}
      {complianceMetrics.nonCompliantFrameworks > 0 && (
        <Alert
          message={`${complianceMetrics.nonCompliantFrameworks} Framework${complianceMetrics.nonCompliantFrameworks > 1 ? 's' : ''} Non-Compliant`}
          description="Immediate attention required to address compliance violations and remediation actions."
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" type="primary" danger>
              View Violations
            </Button>
          }
        />
      )}

      {/* Controls */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Text>Framework:</Text>
              <Select
                value={selectedFramework}
                onChange={setSelectedFramework}
                style={{ width: 200 }}
              >
                <Option value="all">All Frameworks</Option>
                {frameworks.map(framework => (
                  <Option key={framework.id} value={framework.id}>
                    {framework.name}
                  </Option>
                ))}
              </Select>
              
              <Text>Period:</Text>
              <RangePicker
                value={selectedPeriod}
                onChange={(dates) => setSelectedPeriod(dates || [dayjs().subtract(1, 'month'), dayjs()])}
                style={{ width: 250 }}
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<FileTextOutlined />}
                onClick={() => setShowReportModal(true)}
              >
                Generate Report
              </Button>
              <Button
                icon={<AuditOutlined />}
                onClick={() => setShowAssessmentModal(true)}
              >
                Run Assessment
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={16}>
        <Col span={18}>
          {/* Main Content Tabs */}
          <Tabs
            defaultActiveKey="frameworks"
            items={[
              {
                key: 'frameworks',
                label: (
                  <Space>
                    <BankOutlined />
                    Frameworks
                    <Badge count={frameworks.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card title="Compliance Frameworks">
                    <Table
                      dataSource={filteredFrameworks}
                      columns={frameworkColumns}
                      rowKey="id"
                      pagination={false}
                      size="small"
                    />
                  </Card>
                )
              },
              {
                key: 'requirements',
                label: (
                  <Space>
                    <FileProtectOutlined />
                    Requirements
                    <Badge count={filteredRequirements.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card title="Compliance Requirements">
                    <Table
                      dataSource={filteredRequirements}
                      columns={requirementColumns}
                      rowKey="id"
                      pagination={{
                        pageSize: 10,
                        showSizeChanger: false
                      }}
                      size="small"
                      rowClassName={(record) => {
                        if (record.status === 'non_compliant') return 'requirement-violation';
                        if (record.deadline && dayjs(record.deadline).diff(dayjs(), 'day') < 7) return 'requirement-urgent';
                        return '';
                      }}
                    />
                  </Card>
                )
              },
              {
                key: 'reports',
                label: (
                  <Space>
                    <FileTextOutlined />
                    Reports
                    <Badge count={reports.length} size="small" />
                  </Space>
                ),
                children: (
                  <Card title="Compliance Reports">
                    <Table
                      dataSource={reports}
                      columns={reportColumns}
                      rowKey="id"
                      pagination={{
                        pageSize: 10,
                        showSizeChanger: false
                      }}
                      size="small"
                    />
                  </Card>
                )
              }
            ]}
          />
        </Col>

        <Col span={6}>
          {/* Sidebar */}
          
          {/* Upcoming Deadlines */}
          <Card title="Upcoming Deadlines" size="small" style={{ marginBottom: 16 }}>
            <Timeline size="small">
              {requirements
                .filter(req => req.deadline)
                .sort((a, b) => dayjs(a.deadline).diff(dayjs(b.deadline)))
                .slice(0, 5)
                .map((req, index) => {
                  const isOverdue = req.deadline && dayjs(req.deadline).isBefore(dayjs());
                  const isUrgent = req.deadline && dayjs(req.deadline).diff(dayjs(), 'day') < 7;
                  
                  return (
                    <Timeline.Item
                      key={index}
                      color={isOverdue ? 'red' : isUrgent ? 'orange' : 'blue'}
                    >
                      <div style={{ fontSize: '12px' }}>
                        <div style={{ fontWeight: 'bold' }}>{req.code}</div>
                        <div style={{ color: '#666' }}>{req.title}</div>
                        <div style={{ 
                          color: isOverdue ? '#ff4d4f' : isUrgent ? '#faad14' : '#666' 
                        }}>
                          {req.deadline && dayjs(req.deadline).format('MMM D, YYYY')}
                        </div>
                      </div>
                    </Timeline.Item>
                  );
                })}
            </Timeline>
          </Card>

          {/* Framework Status Overview */}
          <Card title="Framework Status" size="small" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {frameworks.map((framework) => (
                <div key={framework.id} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text style={{ fontSize: '12px' }}>{framework.name}</Text>
                    <Tag size="small" color={getStatusColor(framework.status)}>
                      {framework.status.replace('_', ' ')}
                    </Tag>
                  </div>
                </div>
              ))}
            </Space>
          </Card>

          {/* Quick Actions */}
          <Card title="Quick Actions" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                block
                icon={<AuditOutlined />}
                onClick={() => setShowAssessmentModal(true)}
              >
                Run Full Assessment
              </Button>
              <Button
                block
                icon={<FileTextOutlined />}
                onClick={() => setShowReportModal(true)}
              >
                Generate Compliance Report
              </Button>
              <Button
                block
                icon={<CalendarOutlined />}
              >
                Schedule Assessment
              </Button>
              <Button
                block
                icon={<SettingOutlined />}
              >
                Configure Frameworks
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Generate Report Modal */}
      <Modal
        title="Generate Compliance Report"
        open={showReportModal}
        onCancel={() => {
          setShowReportModal(false);
          reportForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={reportForm}
          layout="vertical"
          onFinish={(values) => {
            console.log('Generate report:', values);
            setShowReportModal(false);
            reportForm.resetFields();
          }}
        >
          <Form.Item
            name="framework_id"
            label="Framework"
            rules={[{ required: true, message: 'Please select framework' }]}
          >
            <Select placeholder="Select compliance framework">
              {frameworks.map(framework => (
                <Option key={framework.id} value={framework.id}>
                  {framework.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="report_type"
            label="Report Type"
            rules={[{ required: true, message: 'Please select report type' }]}
          >
            <Select placeholder="Select report type">
              <Option value="assessment">Compliance Assessment</Option>
              <Option value="gap_analysis">Gap Analysis</Option>
              <Option value="remediation">Remediation Plan</Option>
              <Option value="periodic">Periodic Review</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="date_range"
            label="Reporting Period"
            rules={[{ required: true, message: 'Please select date range' }]}
          >
            <RangePicker style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item name="include_remediation" label="Include Remediation Plans" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="include_evidence" label="Include Evidence Links" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="comments" label="Additional Comments">
            <TextArea rows={3} placeholder="Any additional context or requirements..." />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowReportModal(false);
                reportForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                Generate Report
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Assessment Modal */}
      <Modal
        title="Run Compliance Assessment"
        open={showAssessmentModal}
        onCancel={() => {
          setShowAssessmentModal(false);
          assessmentForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Alert
          message="Compliance Assessment"
          description="This will run a comprehensive assessment against all selected framework requirements. The process may take several minutes to complete."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Form
          form={assessmentForm}
          layout="vertical"
          onFinish={(values) => {
            console.log('Run assessment:', values);
            setShowAssessmentModal(false);
            assessmentForm.resetFields();
          }}
        >
          <Form.Item
            name="framework_id"
            label="Framework"
            rules={[{ required: true, message: 'Please select framework' }]}
          >
            <Select placeholder="Select compliance framework" mode="multiple">
              {frameworks.map(framework => (
                <Option key={framework.id} value={framework.id}>
                  {framework.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item name="include_evidence_check" label="Include Evidence Verification" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="auto_remediation" label="Generate Automatic Remediation Plans" valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="notification_level" label="Notification Level">
            <Rate count={3} />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setShowAssessmentModal(false);
                assessmentForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                Start Assessment
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <style jsx>{`
        .requirement-violation {
          background-color: #fff2f0 !important;
          border-left: 4px solid #ff4d4f !important;
        }
        .requirement-urgent {
          background-color: #fff7e6 !important;
          border-left: 4px solid #faad14 !important;
        }
      `}</style>
    </div>
  );
};

export default ComplianceReporting;