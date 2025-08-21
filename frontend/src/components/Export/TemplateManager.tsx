/**
 * Story 5.3: Template Manager Component
 * Management interface for report templates
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Typography,
  Row,
  Col,
  Input,
  Select,
  notification,
  Tooltip,
  Badge,
  Popconfirm,
  Descriptions,
  Alert
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  CopyOutlined,
  ClockCircleOutlined,
  MailOutlined,
  FileTextOutlined,
  SearchOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { ReportBuilder } from './ReportBuilder';
import {
  ReportTemplate,
  ReportType,
  ReportFormat,
  ScheduleFrequency,
  ReportGenerationResponse
} from '../../types/export';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

interface TemplateManagerProps {
  onGenerateReport?: (templateId: string, parameters: Record<string, any>) => Promise<ReportGenerationResponse>;
  className?: string;
}

export const TemplateManager: React.FC<TemplateManagerProps> = ({
  onGenerateReport,
  className
}) => {
  const [templates, setTemplates] = useState<ReportTemplate[]>([]);
  const [loading, setLoading] = useState(false);
  const [builderVisible, setBuilderVisible] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | undefined>();
  const [previewVisible, setPreviewVisible] = useState(false);
  const [generateModalVisible, setGenerateModalVisible] = useState(false);
  const [generatingTemplate, setGeneratingTemplate] = useState<ReportTemplate | null>(null);
  const [searchText, setSearchText] = useState('');
  const [filterType, setFilterType] = useState<string>('all');

  // Mock templates data for demonstration
  const mockTemplates: ReportTemplate[] = [
    {
      id: 'template-001',
      name: 'Daily Performance Report',
      description: 'Comprehensive daily trading performance analysis',
      type: ReportType.PERFORMANCE,
      format: ReportFormat.PDF,
      sections: [
        {
          id: 'section-1',
          name: 'Performance Overview',
          type: 'metrics',
          configuration: { metrics: ['total_pnl', 'win_rate', 'sharpe_ratio'] }
        },
        {
          id: 'section-2',
          name: 'Trade Summary',
          type: 'table',
          configuration: { data_source: 'trades', limit: 20 }
        }
      ],
      parameters: [
        {
          name: 'date',
          type: 'date',
          default_value: dayjs().format('YYYY-MM-DD'),
          required: true
        }
      ],
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
      format: ReportFormat.EXCEL,
      sections: [
        {
          id: 'section-1',
          name: 'Risk Metrics',
          type: 'metrics',
          configuration: { metrics: ['var', 'max_drawdown', 'beta'] }
        }
      ],
      parameters: [
        {
          name: 'week_ending',
          type: 'date',
          default_value: dayjs().endOf('week').format('YYYY-MM-DD'),
          required: true
        }
      ],
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
      format: ReportFormat.PDF,
      sections: [
        {
          id: 'section-1',
          name: 'Compliance Summary',
          type: 'summary',
          configuration: { include_violations: true }
        }
      ],
      parameters: [
        {
          name: 'month',
          type: 'text',
          default_value: dayjs().format('YYYY-MM'),
          required: true
        }
      ],
      created_at: new Date('2024-01-05'),
      updated_at: new Date('2024-01-15')
    }
  ];

  useEffect(() => {
    // Initialize with mock data
    setTemplates(mockTemplates);
  }, []);

  const handleCreateTemplate = () => {
    setSelectedTemplate(undefined);
    setBuilderVisible(true);
  };

  const handleEditTemplate = (template: ReportTemplate) => {
    setSelectedTemplate(template);
    setBuilderVisible(true);
  };

  const handleSaveTemplate = async (template: ReportTemplate) => {
    try {
      if (template.id) {
        // Update existing template
        setTemplates(prev => prev.map(t => t.id === template.id ? template : t));
      } else {
        // Create new template
        const newTemplate = { ...template, id: `template-${Date.now()}`, created_at: new Date() };
        setTemplates(prev => [...prev, newTemplate]);
      }
      
      notification.success({
        message: 'Template Saved',
        description: 'Report template has been saved successfully',
      });
    } catch (error: any) {
      notification.error({
        message: 'Save Failed',
        description: error.message || 'Failed to save template',
      });
    }
  };

  const handleDeleteTemplate = (templateId: string) => {
    setTemplates(prev => prev.filter(t => t.id !== templateId));
    notification.success({
      message: 'Template Deleted',
      description: 'Report template has been deleted successfully',
    });
  };

  const handleDuplicateTemplate = (template: ReportTemplate) => {
    const duplicatedTemplate = {
      ...template,
      id: `template-${Date.now()}`,
      name: `${template.name} (Copy)`,
      created_at: new Date(),
      updated_at: new Date(),
      schedule: undefined // Remove schedule from duplicated template
    };
    setTemplates(prev => [...prev, duplicatedTemplate]);
    notification.success({
      message: 'Template Duplicated',
      description: 'Template has been duplicated successfully',
    });
  };

  const handleGenerateReport = (template: ReportTemplate) => {
    setGeneratingTemplate(template);
    setGenerateModalVisible(true);
  };

  const handlePreviewTemplate = (template: ReportTemplate) => {
    setSelectedTemplate(template);
    setPreviewVisible(true);
  };

  const getTypeColor = (type: ReportType): string => {
    const colors = {
      [ReportType.PERFORMANCE]: 'blue',
      [ReportType.COMPLIANCE]: 'orange',
      [ReportType.RISK]: 'red',
      [ReportType.CUSTOM]: 'purple'
    };
    return colors[type] || 'default';
  };

  const getFormatIcon = (format: ReportFormat) => {
    switch (format) {
      case ReportFormat.PDF: return '📄';
      case ReportFormat.EXCEL: return '📊';
      case ReportFormat.HTML: return '🌐';
      default: return '📄';
    }
  };

  // Filter templates based on search and type
  const filteredTemplates = templates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchText.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchText.toLowerCase());
    const matchesType = filterType === 'all' || template.type === filterType;
    return matchesSearch && matchesType;
  });

  const columns = [
    {
      title: 'Template Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: ReportTemplate) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: ReportType) => (
        <Tag color={getTypeColor(type)}>
          {type.replace('_', ' ').toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Format',
      dataIndex: 'format',
      key: 'format',
      render: (format: ReportFormat) => (
        <span>
          {getFormatIcon(format)} {format.toUpperCase()}
        </span>
      )
    },
    {
      title: 'Schedule',
      key: 'schedule',
      render: (_: any, record: ReportTemplate) => (
        record.schedule ? (
          <Tooltip title={`${record.schedule.frequency} at ${record.schedule.time} (${record.schedule.timezone})`}>
            <Badge status="processing" />
            <Text style={{ fontSize: 12 }}>
              <ClockCircleOutlined style={{ marginRight: 4 }} />
              {record.schedule.frequency}
            </Text>
          </Tooltip>
        ) : (
          <Text type="secondary" style={{ fontSize: 12 }}>Manual only</Text>
        )
      )
    },
    {
      title: 'Updated',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (date: Date) => (
        <Text style={{ fontSize: 12 }}>
          {dayjs(date).format('MMM DD, YYYY')}
        </Text>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: ReportTemplate) => (
        <Space>
          <Tooltip title="Generate Report">
            <Button
              type="text"
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleGenerateReport(record)}
            />
          </Tooltip>
          <Tooltip title="Preview">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handlePreviewTemplate(record)}
            />
          </Tooltip>
          <Tooltip title="Edit">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditTemplate(record)}
            />
          </Tooltip>
          <Tooltip title="Duplicate">
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => handleDuplicateTemplate(record)}
            />
          </Tooltip>
          <Popconfirm
            title="Delete Template"
            description="Are you sure you want to delete this template?"
            onConfirm={() => handleDeleteTemplate(record.id!)}
            okText="Delete"
            okType="danger"
          >
            <Tooltip title="Delete">
              <Button
                type="text"
                size="small"
                danger
                icon={<DeleteOutlined />}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <div className={`template-manager ${className || ''}`}>
      <Card>
        <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
          <Col>
            <Title level={3} style={{ margin: 0 }}>
              <FileTextOutlined style={{ marginRight: 8 }} />
              Report Templates
            </Title>
          </Col>
          <Col>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleCreateTemplate}
            >
              Create Template
            </Button>
          </Col>
        </Row>

        {/* Filters */}
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col xs={24} md={12}>
            <Search
              placeholder="Search templates..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              prefix={<SearchOutlined />}
            />
          </Col>
          <Col xs={24} md={8}>
            <Select
              value={filterType}
              onChange={setFilterType}
              style={{ width: '100%' }}
            >
              <Option value="all">All Types</Option>
              <Option value={ReportType.PERFORMANCE}>Performance</Option>
              <Option value={ReportType.COMPLIANCE}>Compliance</Option>
              <Option value={ReportType.RISK}>Risk</Option>
              <Option value={ReportType.CUSTOM}>Custom</Option>
            </Select>
          </Col>
          <Col xs={24} md={4}>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                // Refresh templates
                setTemplates([...mockTemplates]);
                notification.info({ message: 'Templates refreshed' });
              }}
              loading={loading}
            >
              Refresh
            </Button>
          </Col>
        </Row>

        {/* Statistics */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col xs={12} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
                  {templates.length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Total Templates</div>
              </div>
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
                  {templates.filter(t => t.schedule).length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Scheduled</div>
              </div>
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
                  {templates.filter(t => t.type === ReportType.PERFORMANCE).length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Performance</div>
              </div>
            </Card>
          </Col>
          <Col xs={12} sm={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 'bold', color: '#f5222d' }}>
                  {templates.filter(t => t.type === ReportType.RISK).length}
                </div>
                <div style={{ fontSize: 12, color: '#666' }}>Risk</div>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Templates Table */}
        <Table
          columns={columns}
          dataSource={filteredTemplates}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} templates`
          }}
        />
      </Card>

      {/* Report Builder Modal */}
      <ReportBuilder
        visible={builderVisible}
        onCancel={() => setBuilderVisible(false)}
        onSave={handleSaveTemplate}
        initialTemplate={selectedTemplate}
      />

      {/* Template Preview Modal */}
      <Modal
        title="Template Preview"
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setPreviewVisible(false)}>
            Close
          </Button>,
          <Button
            key="edit"
            type="primary"
            onClick={() => {
              setPreviewVisible(false);
              handleEditTemplate(selectedTemplate!);
            }}
          >
            Edit Template
          </Button>
        ]}
      >
        {selectedTemplate && (
          <div>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Name">{selectedTemplate.name}</Descriptions.Item>
              <Descriptions.Item label="Type">
                <Tag color={getTypeColor(selectedTemplate.type)}>
                  {selectedTemplate.type.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Format">
                {getFormatIcon(selectedTemplate.format)} {selectedTemplate.format.toUpperCase()}
              </Descriptions.Item>
              <Descriptions.Item label="Sections" span={3}>
                {selectedTemplate.sections.length} sections configured
              </Descriptions.Item>
              <Descriptions.Item label="Parameters" span={3}>
                {selectedTemplate.parameters.length} parameters defined
              </Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <Title level={5}>Description</Title>
              <Text>{selectedTemplate.description}</Text>
            </div>

            {selectedTemplate.schedule && (
              <div style={{ marginTop: 16 }}>
                <Title level={5}>Schedule</Title>
                <Alert
                  message={`Automated ${selectedTemplate.schedule.frequency} reports`}
                  description={`Generated at ${selectedTemplate.schedule.time} (${selectedTemplate.schedule.timezone}) and sent to ${selectedTemplate.schedule.recipients.length} recipients`}
                  type="info"
                  showIcon
                />
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Generate Report Modal */}
      <Modal
        title="Generate Report"
        open={generateModalVisible}
        onCancel={() => setGenerateModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setGenerateModalVisible(false)}>
            Cancel
          </Button>,
          <Button key="generate" type="primary">
            Generate Report
          </Button>
        ]}
      >
        {generatingTemplate && (
          <div>
            <Text>Generate report from template: <strong>{generatingTemplate.name}</strong></Text>
            <Alert
              message="Feature Implementation"
              description="Report generation functionality will be implemented with parameter collection and API integration."
              type="info"
              style={{ marginTop: 16 }}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default TemplateManager;