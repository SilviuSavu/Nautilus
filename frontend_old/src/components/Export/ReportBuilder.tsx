/**
 * Story 5.3: Report Builder Component
 * Visual report template builder interface
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Button,
  Form,
  Input,
  Select,
  Modal,
  Steps,
  Alert,
  Switch,
  TimePicker,
  Tag,
  Space,
  Divider,
  List,
  Tooltip,
  notification,
  Collapse
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EyeOutlined,
  SaveOutlined,
  SettingOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  MailOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  ReportTemplate,
  ReportType,
  ReportFormat,
  ScheduleFrequency,
  ReportSection,
  ReportParameter,
  ReportSchedule
} from '../../types/export';

const { Title, Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { Step } = Steps;
const { Panel } = Collapse;

interface ReportBuilderProps {
  visible: boolean;
  onCancel: () => void;
  onSave: (template: ReportTemplate) => Promise<void>;
  initialTemplate?: ReportTemplate;
}

export const ReportBuilder: React.FC<ReportBuilderProps> = ({
  visible,
  onCancel,
  onSave,
  initialTemplate
}) => {
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [sections, setSections] = useState<ReportSection[]>([]);
  const [parameters, setParameters] = useState<ReportParameter[]>([]);
  const [schedule, setSchedule] = useState<ReportSchedule | null>(null);
  const [schedulingEnabled, setSchedulingEnabled] = useState(false);

  // Predefined section types
  const sectionTypes = [
    { value: 'metrics', label: 'Performance Metrics', description: 'Key performance indicators and statistics' },
    { value: 'chart', label: 'Chart/Graph', description: 'Visual data representation' },
    { value: 'table', label: 'Data Table', description: 'Tabular data display' },
    { value: 'text', label: 'Text Block', description: 'Formatted text content' },
    { value: 'summary', label: 'Executive Summary', description: 'High-level overview' },
    { value: 'analysis', label: 'Analysis Section', description: 'Detailed analysis and insights' }
  ];

  // Parameter types
  const parameterTypes = [
    { value: 'text', label: 'Text' },
    { value: 'number', label: 'Number' },
    { value: 'date', label: 'Date' },
    { value: 'date_range', label: 'Date Range' },
    { value: 'select', label: 'Selection' },
    { value: 'boolean', label: 'True/False' }
  ];

  useEffect(() => {
    if (initialTemplate) {
      form.setFieldsValue({
        name: initialTemplate.name,
        description: initialTemplate.description,
        type: initialTemplate.type,
        format: initialTemplate.format
      });
      setSections(initialTemplate.sections || []);
      setParameters(initialTemplate.parameters || []);
      if (initialTemplate.schedule) {
        setSchedule(initialTemplate.schedule);
        setSchedulingEnabled(true);
      }
    } else {
      // Reset form for new template
      form.resetFields();
      setSections([]);
      setParameters([]);
      setSchedule(null);
      setSchedulingEnabled(false);
    }
  }, [initialTemplate, form]);

  const addSection = () => {
    const newSection: ReportSection = {
      id: `section-${Date.now()}`,
      name: '',
      type: 'metrics',
      configuration: {}
    };
    setSections([...sections, newSection]);
  };

  const updateSection = (index: number, updatedSection: ReportSection) => {
    const newSections = [...sections];
    newSections[index] = updatedSection;
    setSections(newSections);
  };

  const removeSection = (index: number) => {
    setSections(sections.filter((_, i) => i !== index));
  };

  const addParameter = () => {
    const newParameter: ReportParameter = {
      name: '',
      type: 'text',
      default_value: '',
      required: false
    };
    setParameters([...parameters, newParameter]);
  };

  const updateParameter = (index: number, updatedParameter: ReportParameter) => {
    const newParameters = [...parameters];
    newParameters[index] = updatedParameter;
    setParameters(newParameters);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      
      const values = await form.validateFields();
      
      if (sections.length === 0) {
        notification.error({
          message: 'Validation Error',
          description: 'Please add at least one section to the report template',
        });
        return;
      }

      // Validate sections have names
      const invalidSections = sections.filter(section => !section.name.trim());
      if (invalidSections.length > 0) {
        notification.error({
          message: 'Validation Error',
          description: 'All sections must have a name',
        });
        return;
      }

      const template: ReportTemplate = {
        ...initialTemplate,
        name: values.name,
        description: values.description,
        type: values.type,
        format: values.format,
        sections,
        parameters,
        schedule: schedulingEnabled ? schedule : undefined
      };

      await onSave(template);
      
      notification.success({
        message: 'Template Saved',
        description: 'Report template has been saved successfully',
      });

      onCancel();
    } catch (error: any) {
      notification.error({
        message: 'Save Failed',
        description: error.message || 'Failed to save report template',
      });
    } finally {
      setSaving(false);
    }
  };

  const handleScheduleChange = (scheduleData: Partial<ReportSchedule>) => {
    setSchedule(prev => ({ ...prev, ...scheduleData } as ReportSchedule));
  };

  const renderBasicInfo = () => (
    <Card title="Basic Information">
      <Row gutter={16}>
        <Col xs={24} md={12}>
          <Form.Item
            name="name"
            label="Template Name"
            rules={[{ required: true, message: 'Please enter template name' }]}
          >
            <Input placeholder="e.g., Daily Performance Report" />
          </Form.Item>
        </Col>
        <Col xs={24} md={12}>
          <Form.Item
            name="type"
            label="Report Type"
            rules={[{ required: true, message: 'Please select report type' }]}
          >
            <Select placeholder="Select report type">
              <Option value={ReportType.PERFORMANCE}>Performance Report</Option>
              <Option value={ReportType.COMPLIANCE}>Compliance Report</Option>
              <Option value={ReportType.RISK}>Risk Report</Option>
              <Option value={ReportType.CUSTOM}>Custom Report</Option>
            </Select>
          </Form.Item>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col xs={24} md={12}>
          <Form.Item
            name="format"
            label="Output Format"
            rules={[{ required: true, message: 'Please select output format' }]}
          >
            <Select placeholder="Select output format">
              <Option value={ReportFormat.PDF}>PDF Document</Option>
              <Option value={ReportFormat.EXCEL}>Excel Spreadsheet</Option>
              <Option value={ReportFormat.HTML}>HTML Document</Option>
            </Select>
          </Form.Item>
        </Col>
      </Row>

      <Form.Item
        name="description"
        label="Description"
        rules={[{ required: true, message: 'Please enter description' }]}
      >
        <TextArea 
          rows={3} 
          placeholder="Describe what this report template will generate..."
        />
      </Form.Item>
    </Card>
  );

  const renderSections = () => (
    <Card 
      title="Report Sections" 
      extra={
        <Button type="primary" icon={<PlusOutlined />} onClick={addSection}>
          Add Section
        </Button>
      }
    >
      {sections.length === 0 ? (
        <Alert
          message="No Sections Added"
          description="Add sections to define the content and structure of your report."
          type="info"
          showIcon
        />
      ) : (
        <Collapse defaultActiveKey={['0']}>
          {sections.map((section, index) => (
            <Panel
              key={index}
              header={
                <div>
                  <Text strong>{section.name || `Section ${index + 1}`}</Text>
                  <Tag color="blue" style={{ marginLeft: 8 }}>
                    {sectionTypes.find(t => t.value === section.type)?.label}
                  </Tag>
                </div>
              }
              extra={
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    removeSection(index);
                  }}
                />
              }
            >
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  <Form.Item label="Section Name">
                    <Input
                      value={section.name}
                      placeholder="Enter section name"
                      onChange={(e) => updateSection(index, { ...section, name: e.target.value })}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="Section Type">
                    <Select
                      value={section.type}
                      onChange={(value) => updateSection(index, { ...section, type: value })}
                    >
                      {sectionTypes.map(type => (
                        <Option key={type.value} value={type.value}>
                          <div>
                            <div>{type.label}</div>
                            <Text type="secondary" style={{ fontSize: 12 }}>{type.description}</Text>
                          </div>
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
            </Panel>
          ))}
        </Collapse>
      )}
    </Card>
  );

  const renderParameters = () => (
    <Card 
      title="Template Parameters" 
      extra={
        <Button type="primary" icon={<PlusOutlined />} onClick={addParameter}>
          Add Parameter
        </Button>
      }
    >
      <Alert
        message="Template Parameters"
        description="Parameters allow users to customize the report when generating it. Add parameters for dynamic content like date ranges, filters, etc."
        type="info"
        style={{ marginBottom: 16 }}
      />

      {parameters.length === 0 ? (
        <Text type="secondary">No parameters defined. The report will use default values.</Text>
      ) : (
        <List
          dataSource={parameters}
          renderItem={(parameter, index) => (
            <List.Item
              actions={[
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeParameter(index)}
                />
              ]}
            >
              <Row gutter={16} style={{ width: '100%' }}>
                <Col xs={24} md={6}>
                  <Input
                    value={parameter.name}
                    placeholder="Parameter name"
                    onChange={(e) => updateParameter(index, { ...parameter, name: e.target.value })}
                  />
                </Col>
                <Col xs={24} md={4}>
                  <Select
                    value={parameter.type}
                    onChange={(value) => updateParameter(index, { ...parameter, type: value })}
                    style={{ width: '100%' }}
                  >
                    {parameterTypes.map(type => (
                      <Option key={type.value} value={type.value}>{type.label}</Option>
                    ))}
                  </Select>
                </Col>
                <Col xs={24} md={6}>
                  <Input
                    value={parameter.default_value}
                    placeholder="Default value"
                    onChange={(e) => updateParameter(index, { ...parameter, default_value: e.target.value })}
                  />
                </Col>
                <Col xs={24} md={4}>
                  <Switch
                    checked={parameter.required}
                    onChange={(checked) => updateParameter(index, { ...parameter, required: checked })}
                    checkedChildren="Required"
                    unCheckedChildren="Optional"
                  />
                </Col>
              </Row>
            </List.Item>
          )}
        />
      )}
    </Card>
  );

  const renderScheduling = () => (
    <Card title="Report Scheduling">
      <Form.Item>
        <Switch
          checked={schedulingEnabled}
          onChange={setSchedulingEnabled}
          checkedChildren="Enabled"
          unCheckedChildren="Disabled"
        />
        <Text style={{ marginLeft: 8 }}>Enable automatic report generation</Text>
      </Form.Item>

      {schedulingEnabled && (
        <div>
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item label="Frequency">
                <Select
                  value={schedule?.frequency}
                  placeholder="Select frequency"
                  onChange={(value) => handleScheduleChange({ frequency: value })}
                >
                  <Option value={ScheduleFrequency.DAILY}>Daily</Option>
                  <Option value={ScheduleFrequency.WEEKLY}>Weekly</Option>
                  <Option value={ScheduleFrequency.MONTHLY}>Monthly</Option>
                  <Option value={ScheduleFrequency.QUARTERLY}>Quarterly</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="Time">
                <TimePicker
                  value={schedule?.time ? dayjs(schedule.time, 'HH:mm') : undefined}
                  format="HH:mm"
                  onChange={(time) => handleScheduleChange({ time: time?.format('HH:mm') || '' })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="Timezone">
                <Select
                  value={schedule?.timezone}
                  placeholder="Select timezone"
                  onChange={(value) => handleScheduleChange({ timezone: value })}
                >
                  <Option value="UTC">UTC</Option>
                  <Option value="America/New_York">Eastern Time</Option>
                  <Option value="America/Chicago">Central Time</Option>
                  <Option value="America/Los_Angeles">Pacific Time</Option>
                  <Option value="Europe/London">London Time</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item label="Email Recipients">
            <Select
              mode="tags"
              value={schedule?.recipients}
              placeholder="Enter email addresses"
              onChange={(value) => handleScheduleChange({ recipients: value })}
              style={{ width: '100%' }}
            />
            <Text type="secondary">
              Enter email addresses to receive the automated reports
            </Text>
          </Form.Item>
        </div>
      )}
    </Card>
  );

  const steps = [
    {
      title: 'Basic Info',
      content: renderBasicInfo(),
      icon: <FileTextOutlined />
    },
    {
      title: 'Sections',
      content: renderSections(),
      icon: <SettingOutlined />
    },
    {
      title: 'Parameters',
      content: renderParameters(),
      icon: <SettingOutlined />
    },
    {
      title: 'Scheduling',
      content: renderScheduling(),
      icon: <ClockCircleOutlined />
    }
  ];

  return (
    <Modal
      title={
        <div>
          <FileTextOutlined style={{ marginRight: 8 }} />
          {initialTemplate ? 'Edit Report Template' : 'Create Report Template'}
        </div>
      }
      open={visible}
      onCancel={onCancel}
      width={1000}
      footer={
        <Space>
          <Button onClick={onCancel}>Cancel</Button>
          <Button type="primary" loading={saving} icon={<SaveOutlined />} onClick={handleSave}>
            {saving ? 'Saving...' : 'Save Template'}
          </Button>
        </Space>
      }
    >
      <Form form={form} layout="vertical">
        <Steps 
          current={currentStep} 
          onChange={setCurrentStep}
          style={{ marginBottom: 24 }}
        >
          {steps.map((step, index) => (
            <Step key={index} title={step.title} icon={step.icon} />
          ))}
        </Steps>

        <div style={{ minHeight: 400 }}>
          {steps[currentStep].content}
        </div>

        <Row justify="space-between" style={{ marginTop: 24 }}>
          <Col>
            {currentStep > 0 && (
              <Button onClick={() => setCurrentStep(currentStep - 1)}>
                Previous
              </Button>
            )}
          </Col>
          <Col>
            {currentStep < steps.length - 1 ? (
              <Button type="primary" onClick={() => setCurrentStep(currentStep + 1)}>
                Next
              </Button>
            ) : (
              <Button type="primary" loading={saving} icon={<SaveOutlined />} onClick={handleSave}>
                {saving ? 'Saving...' : 'Save Template'}
              </Button>
            )}
          </Col>
        </Row>
      </Form>
    </Modal>
  );
};

export default ReportBuilder;