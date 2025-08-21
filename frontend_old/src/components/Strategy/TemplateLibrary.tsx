import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Input,
  Select,
  Tag,
  Button,
  Space,
  Typography,
  Spin,
  Alert,
  Empty,
  Tooltip,
  Badge
} from 'antd';
import {
  SearchOutlined,
  BookOutlined,
  TrophyOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { StrategyTemplate, TemplateCategory, StrategySearchFilters } from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;
const { Option } = Select;

interface TemplateLibraryProps {
  onTemplateSelect: (template: StrategyTemplate) => void;
  selectedTemplateId?: string;
  className?: string;
}

const CATEGORY_ICONS: Record<string, React.ComponentType<any>> = {
  trend_following: TrophyOutlined,
  mean_reversion: BarChartOutlined,
  arbitrage: ThunderboltOutlined,
  market_making: BookOutlined
};

const CATEGORY_COLORS: Record<string, string> = {
  trend_following: '#1890ff',
  mean_reversion: '#52c41a', 
  arbitrage: '#fa8c16',
  market_making: '#722ed1'
};

export const TemplateLibrary: React.FC<TemplateLibraryProps> = ({
  onTemplateSelect,
  selectedTemplateId,
  className
}) => {
  const [templates, setTemplates] = useState<StrategyTemplate[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<StrategySearchFilters>({});

  useEffect(() => {
    loadTemplates();
  }, [filters]);

  const loadTemplates = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await strategyService.searchTemplates(filters);
      setTemplates(response.templates);
      setCategories(response.categories);
    } catch (err: any) {
      console.error('Failed to load strategy templates:', err);
      setError(err.message || 'Failed to load strategy templates');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (value: string) => {
    setFilters(prev => ({ ...prev, search_text: value || undefined }));
  };

  const handleCategoryFilter = (category: string | undefined) => {
    setFilters(prev => ({ ...prev, category }));
  };

  const getCategoryIcon = (category: string) => {
    const IconComponent = CATEGORY_ICONS[category] || BookOutlined;
    return <IconComponent style={{ color: CATEGORY_COLORS[category] }} />;
  };

  const renderTemplateCard = (template: StrategyTemplate) => {
    const isSelected = template.id === selectedTemplateId;
    const parameterCount = template.parameters.length;
    const riskParameterCount = template.risk_parameters.length;

    return (
      <Card
        key={template.id}
        hoverable
        className={`template-card ${isSelected ? 'selected' : ''}`}
        style={{
          marginBottom: 16,
          border: isSelected ? '2px solid #1890ff' : undefined,
          cursor: 'pointer'
        }}
        onClick={() => onTemplateSelect(template)}
        styles={{ body: { padding: '16px' } }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          <div style={{ fontSize: 24 }}>
            {getCategoryIcon(template.category)}
          </div>
          
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
              <Title level={5} style={{ margin: 0, fontSize: 16 }}>
                {template.name}
              </Title>
              <Tag color={CATEGORY_COLORS[template.category]}>
                {template.category.replace('_', ' ')}
              </Tag>
            </div>

            <Paragraph 
              style={{ margin: '8px 0', fontSize: 14, color: '#666' }}
              ellipsis={{ rows: 2 }}
            >
              {template.description}
            </Paragraph>

            <Space size={16} style={{ marginTop: 12 }}>
              <Badge 
                count={parameterCount} 
                style={{ backgroundColor: '#52c41a' }}
              >
                <Text type="secondary">Parameters</Text>
              </Badge>
              
              <Badge 
                count={riskParameterCount} 
                style={{ backgroundColor: '#fa8c16' }}
              >
                <Text type="secondary">Risk Controls</Text>
              </Badge>

              <Badge 
                count={template.example_configs.length} 
                style={{ backgroundColor: '#722ed1' }}
              >
                <Text type="secondary">Examples</Text>
              </Badge>
            </Space>

            {template.documentation_url && (
              <div style={{ marginTop: 12 }}>
                <Button 
                  type="link" 
                  size="small" 
                  icon={<InfoCircleOutlined />}
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(template.documentation_url, '_blank');
                  }}
                >
                  Documentation
                </Button>
              </div>
            )}

            <div style={{ marginTop: 8, fontSize: 12, color: '#999' }}>
              <Text>Class: </Text>
              <Text code style={{ fontSize: 11 }}>{template.python_class}</Text>
            </div>
          </div>
        </div>
      </Card>
    );
  };

  if (error) {
    return (
      <Alert
        type="error"
        message="Failed to Load Templates"
        description={error}
        showIcon
        action={
          <Button size="small" onClick={loadTemplates}>
            Retry
          </Button>
        }
      />
    );
  }

  return (
    <div className={`template-library ${className || ''}`}>
      <div style={{ marginBottom: 24 }}>
        <Title level={4} style={{ marginBottom: 16 }}>
          Strategy Template Library
        </Title>

        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Search
            placeholder="Search templates by name, description, or class..."
            allowClear
            size="large"
            prefix={<SearchOutlined />}
            onSearch={handleSearch}
            onChange={(e) => {
              if (!e.target.value) {
                handleSearch('');
              }
            }}
            style={{ width: '100%' }}
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <Text strong>Category:</Text>
            <Select
              placeholder="All Categories"
              allowClear
              style={{ minWidth: 160 }}
              value={filters.category}
              onChange={handleCategoryFilter}
            >
              {categories.map(category => (
                <Option key={category} value={category}>
                  <Space>
                    {getCategoryIcon(category)}
                    {category.replace('_', ' ')}
                  </Space>
                </Option>
              ))}
            </Select>

            <Text type="secondary">
              {templates.length} templates found
            </Text>
          </div>
        </Space>
      </div>

      <div style={{ maxHeight: '60vh', overflowY: 'auto', paddingRight: 8 }}>
        <Spin spinning={loading}>
          {templates.length === 0 && !loading ? (
            <Empty
              description="No strategy templates found"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <Button type="primary" onClick={loadTemplates}>
                Reload Templates
              </Button>
            </Empty>
          ) : (
            <div>
              {templates.map(renderTemplateCard)}
            </div>
          )}
        </Spin>
      </div>

      <style jsx>{`
        .template-card.selected {
          box-shadow: 0 4px 12px rgba(24, 144, 255, 0.2);
        }
        .template-card:hover {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
      `}</style>
    </div>
  );
};