import React, { useState, useEffect } from 'react';
import {
  Form,
  Input,
  InputNumber,
  Switch,
  Select,
  Card,
  Row,
  Col,
  Typography,
  Space,
  Alert,
  Tooltip,
  Button,
  Divider,
  Badge
} from 'antd';
import {
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { 
  ParameterDefinition, 
  RiskParameterDefinition, 
  StrategyTemplate,
  ValidationRule 
} from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface ParameterConfigProps {
  template: StrategyTemplate;
  initialValues?: Record<string, any>;
  onChange: (values: Record<string, any>) => void;
  onValidation: (isValid: boolean, errors: string[]) => void;
  className?: string;
}

interface ValidationState {
  [key: string]: {
    status: 'success' | 'warning' | 'error' | '';
    message: string;
  };
}

export const ParameterConfig: React.FC<ParameterConfigProps> = ({
  template,
  initialValues = {},
  onChange,
  onValidation,
  className
}) => {
  const [form] = Form.useForm();
  const [validationState, setValidationState] = useState<ValidationState>({});
  const [availableInstruments, setAvailableInstruments] = useState<string[]>([]);
  const [availableTimeframes, setAvailableTimeframes] = useState<string[]>([]);
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    loadSelectOptions();
    form.setFieldsValue(getInitialFormValues());
  }, [template, initialValues]);

  const loadSelectOptions = async () => {
    try {
      const [instruments, timeframes] = await Promise.all([
        strategyService.getAvailableInstruments().catch(() => []),
        strategyService.getAvailableTimeframes().catch(() => [])
      ]);
      setAvailableInstruments(instruments);
      setAvailableTimeframes(timeframes);
    } catch (error) {
      console.error('Failed to load parameter options:', error);
    }
  };

  const getInitialFormValues = () => {
    const values: Record<string, any> = {};
    
    [...template.parameters, ...template.risk_parameters].forEach(param => {
      if (initialValues[param.name] !== undefined) {
        values[param.name] = initialValues[param.name];
      } else if (param.default_value !== undefined) {
        values[param.name] = param.default_value;
      }
    });
    
    return values;
  };

  const validateParameter = async (parameter: ParameterDefinition, value: any) => {
    const errors: string[] = [];
    
    // Required validation
    if (parameter.required && (value === undefined || value === null || value === '')) {
      errors.push(`${parameter.display_name} is required`);
    }
    
    if (value !== undefined && value !== null && value !== '') {
      // Type validation
      if (parameter.type === 'integer' && (!Number.isInteger(Number(value)) || isNaN(Number(value)))) {
        errors.push(`${parameter.display_name} must be a whole number`);
      }
      
      if (parameter.type === 'decimal' && isNaN(Number(value))) {
        errors.push(`${parameter.display_name} must be a number`);
      }
      
      // Range validation
      const numValue = Number(value);
      if (!isNaN(numValue)) {
        if (parameter.min_value !== undefined && numValue < parameter.min_value) {
          errors.push(`${parameter.display_name} must be at least ${parameter.min_value}`);
        }
        if (parameter.max_value !== undefined && numValue > parameter.max_value) {
          errors.push(`${parameter.display_name} must be at most ${parameter.max_value}`);
        }
      }
      
      // Allowed values validation
      if (parameter.allowed_values && parameter.allowed_values.length > 0) {
        if (!parameter.allowed_values.includes(value)) {
          errors.push(`${parameter.display_name} must be one of: ${parameter.allowed_values.join(', ')}`);
        }
      }
      
      // Custom validation rules
      for (const rule of parameter.validation_rules) {
        if (rule.type === 'regex' && typeof value === 'string') {
          const regex = new RegExp(rule.params.pattern);
          if (!regex.test(value)) {
            errors.push(rule.error_message);
          }
        }
        
        if (rule.type === 'range' && !isNaN(Number(value))) {
          const numVal = Number(value);
          if (rule.params.min !== undefined && numVal < rule.params.min) {
            errors.push(rule.error_message);
          }
          if (rule.params.max !== undefined && numVal > rule.params.max) {
            errors.push(rule.error_message);
          }
        }
      }
    }
    
    return errors;
  };

  const handleFieldChange = async () => {
    const values = form.getFieldsValue();
    const newValidationState: ValidationState = {};
    let hasErrors = false;
    
    // Validate all parameters
    for (const parameter of [...template.parameters, ...template.risk_parameters]) {
      const value = values[parameter.name];
      const fieldErrors = await validateParameter(parameter, value);
      
      if (fieldErrors.length > 0) {
        newValidationState[parameter.name] = {
          status: 'error',
          message: fieldErrors[0] // Show first error
        };
        hasErrors = true;
      } else if (parameter.required && value !== undefined && value !== null && value !== '') {
        newValidationState[parameter.name] = {
          status: 'success',
          message: 'Valid'
        };
      } else {
        newValidationState[parameter.name] = {
          status: '',
          message: ''
        };
      }
    }
    
    setValidationState(newValidationState);
    onChange(values);
    onValidation(!hasErrors, Object.values(newValidationState)
      .filter(v => v.status === 'error')
      .map(v => v.message));
  };

  const renderParameterInput = (parameter: ParameterDefinition) => {
    const validation = validationState[parameter.name];
    const isRisk = 'impact_level' in parameter;
    
    const commonProps = {
      placeholder: parameter.help_text,
      onChange: handleFieldChange,
      status: validation?.status as any
    };

    let inputElement;
    
    switch (parameter.type) {
      case 'boolean':
        inputElement = (
          <Switch 
            onChange={handleFieldChange}
            checkedChildren="Yes"
            unCheckedChildren="No"
          />
        );
        break;
        
      case 'integer':
        inputElement = (
          <InputNumber
            {...commonProps}
            style={{ width: '100%' }}
            min={parameter.min_value}
            max={parameter.max_value}
            precision={0}
          />
        );
        break;
        
      case 'decimal':
      case 'percentage':
        inputElement = (
          <InputNumber
            {...commonProps}
            style={{ width: '100%' }}
            min={parameter.min_value}
            max={parameter.max_value}
            precision={parameter.type === 'percentage' ? 2 : 4}
            addonAfter={parameter.type === 'percentage' ? '%' : undefined}
          />
        );
        break;
        
      case 'instrument_id':
        inputElement = (
          <Select
            {...commonProps}
            showSearch
            style={{ width: '100%' }}
            placeholder="Select an instrument"
            filterOption={(input, option) =>
              (option?.children as string)?.toLowerCase().includes(input.toLowerCase())
            }
          >
            {availableInstruments.map(instrument => (
              <Option key={instrument} value={instrument}>
                {instrument}
              </Option>
            ))}
          </Select>
        );
        break;
        
      case 'timeframe':
        inputElement = (
          <Select
            {...commonProps}
            style={{ width: '100%' }}
            placeholder="Select timeframe"
          >
            {availableTimeframes.map(tf => (
              <Option key={tf} value={tf}>
                {tf}
              </Option>
            ))}
          </Select>
        );
        break;
        
      case 'currency':
        inputElement = (
          <Select
            {...commonProps}
            style={{ width: '100%' }}
            placeholder="Select currency"
          >
            {['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF'].map(curr => (
              <Option key={curr} value={curr}>
                {curr}
              </Option>
            ))}
          </Select>
        );
        break;
        
      default:
        if (parameter.allowed_values && parameter.allowed_values.length > 0) {
          inputElement = (
            <Select {...commonProps} style={{ width: '100%' }}>
              {parameter.allowed_values.map(value => (
                <Option key={value} value={value}>
                  {String(value)}
                </Option>
              ))}
            </Select>
          );
        } else {
          inputElement = parameter.help_text.length > 100 ? (
            <TextArea {...commonProps} rows={2} />
          ) : (
            <Input {...commonProps} />
          );
        }
    }

    const label = (
      <Space>
        {parameter.display_name}
        {parameter.required && <Text type="danger">*</Text>}
        {isRisk && (
          <Badge 
            color={(parameter as RiskParameterDefinition).impact_level === 'critical' ? 'red' : 'orange'}
            text="Risk"
          />
        )}
        <Tooltip title={parameter.help_text}>
          <InfoCircleOutlined style={{ color: '#999' }} />
        </Tooltip>
      </Space>
    );

    return (
      <Form.Item
        key={parameter.name}
        name={parameter.name}
        label={label}
        help={validation?.message}
        validateStatus={validation?.status}
        rules={[
          {
            required: parameter.required,
            message: `${parameter.display_name} is required`
          }
        ]}
      >
        {inputElement}
      </Form.Item>
    );
  };

  const groupedParameters = [...template.parameters, ...template.risk_parameters].reduce(
    (groups, param) => {
      const group = param.group || 'General';
      if (!groups[group]) {
        groups[group] = [];
      }
      groups[group].push(param);
      return groups;
    },
    {} as Record<string, ParameterDefinition[]>
  );

  const hasValidationErrors = Object.values(validationState).some(v => v.status === 'error');

  return (
    <div className={`parameter-config ${className || ''}`}>
      <Card 
        title={
          <Space>
            <Title level={4} style={{ margin: 0 }}>
              Strategy Parameters
            </Title>
            {hasValidationErrors ? (
              <Badge status="error" text="Has Errors" />
            ) : (
              <Badge status="success" text="Valid" />
            )}
          </Space>
        }
        extra={
          <Button
            type="primary"
            ghost
            icon={<CheckCircleOutlined />}
            loading={isValidating}
            onClick={() => {
              setIsValidating(true);
              handleFieldChange().finally(() => setIsValidating(false));
            }}
          >
            Validate All
          </Button>
        }
      >
        <Form
          form={form}
          layout="vertical"
          onValuesChange={handleFieldChange}
          initialValues={getInitialFormValues()}
        >
          {Object.entries(groupedParameters).map(([groupName, parameters]) => (
            <div key={groupName}>
              <Divider orientation="left">
                <Text strong>{groupName}</Text>
              </Divider>
              
              <Row gutter={[16, 0]}>
                {parameters.map(param => (
                  <Col 
                    key={param.name} 
                    xs={24} 
                    sm={12} 
                    md={param.type === 'string' && param.help_text.length > 100 ? 24 : 12}
                  >
                    {renderParameterInput(param)}
                  </Col>
                ))}
              </Row>
            </div>
          ))}
        </Form>
      </Card>
    </div>
  );
};