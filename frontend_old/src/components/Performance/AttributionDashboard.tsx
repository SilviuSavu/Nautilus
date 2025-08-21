/**
 * Attribution Dashboard Component - Story 5.1
 * Advanced performance attribution analysis with multiple dimensions
 * 
 * Features:
 * - Sector attribution analysis
 * - Factor attribution breakdown  
 * - Security selection vs allocation effects
 * - Interactive attribution charts
 * - Attribution tree map visualization
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Select,
  Space,
  Typography,
  Tag,
  Progress,
  Statistic,
  Tooltip,
  Button,
  Switch,
  Alert,
  Divider
} from 'antd';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ComposedChart,
  Treemap,
  ScatterChart,
  Scatter
} from 'recharts';
import {
  PieChartOutlined,
  BarChartOutlined,
  LineChartOutlined,
  TableOutlined,
  RiseOutlined,
  FallOutlined,
  ExperimentOutlined,
  FilterOutlined
} from '@ant-design/icons';
import { 
  AttributionAnalysisResponse, 
  SectorAttribution, 
  FactorAttribution 
} from '../../types/analytics';

const { Title, Text } = Typography;
const { Option } = Select;

interface Props {
  data: AttributionAnalysisResponse;
  loading?: boolean;
  onAttributionTypeChange?: (type: 'sector' | 'style' | 'security' | 'factor') => void;
  height?: number;
}

const SECTOR_COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', 
  '#fa8c16', '#13c2c2', '#eb2f96', '#a0d911', '#fa541c'
];

const AttributionDashboard: React.FC<Props> = ({ 
  data, 
  loading, 
  onAttributionTypeChange,
  height = 600 
}) => {
  const [viewType, setViewType] = useState<'chart' | 'table'>('chart');
  const [showAllocation, setShowAllocation] = useState(true);
  const [showSelection, setShowSelection] = useState(true);
  const [sortBy, setSortBy] = useState<'total_effect' | 'allocation_effect' | 'selection_effect'>('total_effect');

  const formatNumber = (value: number, precision: number = 2, suffix: string = ''): string => {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    return `${value.toFixed(precision)}${suffix}`;
  };

  const getEffectColor = (value: number): string => {
    return value >= 0 ? '#52c41a' : '#f5222d';
  };

  const getEffectIcon = (value: number) => {
    return value >= 0 ? <RiseOutlined /> : <FallOutlined />;
  };

  // Process sector attribution data for charts
  const sectorChartData = useMemo(() => {
    if (!data.sector_attribution || data.sector_attribution.length === 0) {
      return [];
    }

    return data.sector_attribution
      .map(item => ({
        ...item,
        sector_short: item.sector.length > 15 ? item.sector.substring(0, 15) + '...' : item.sector,
        allocation_effect_abs: Math.abs(item.allocation_effect),
        selection_effect_abs: Math.abs(item.selection_effect),
        total_effect_abs: Math.abs(item.total_effect)
      }))
      .sort((a, b) => Math.abs(b[sortBy]) - Math.abs(a[sortBy]));
  }, [data.sector_attribution, sortBy]);

  // Process factor attribution data
  const factorChartData = useMemo(() => {
    if (!data.factor_attribution || data.factor_attribution.length === 0) {
      return [];
    }

    return data.factor_attribution
      .map(item => ({
        ...item,
        factor_short: item.factor_name.length > 12 ? item.factor_name.substring(0, 12) + '...' : item.factor_name,
        contribution_abs: Math.abs(item.contribution)
      }))
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }, [data.factor_attribution]);

  // Create treemap data for attribution visualization
  const treemapData = useMemo(() => {
    if (!data.sector_attribution || data.sector_attribution.length === 0) {
      return [];
    }

    return data.sector_attribution.map((item, index) => ({
      name: item.sector,
      size: Math.abs(item.total_effect) * 10000, // Scale for visibility
      value: item.total_effect,
      color: item.total_effect >= 0 ? '#52c41a' : '#f5222d',
      fill: SECTOR_COLORS[index % SECTOR_COLORS.length]
    }));
  }, [data.sector_attribution]);

  const renderAttributionSummary = () => (
    <Card title="Attribution Summary" extra={<ExperimentOutlined />}>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Statistic
            title="Total Active Return"
            value={data.total_active_return}
            precision={3}
            suffix="%"
            valueStyle={{ 
              color: getEffectColor(data.total_active_return),
              fontSize: '24px'
            }}
            prefix={getEffectIcon(data.total_active_return)}
          />
        </Col>
        
        <Col xs={24} lg={16}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Security Selection Effect</Text>
              <div>
                <Progress
                  percent={Math.abs(data.attribution_breakdown.security_selection / data.total_active_return * 100) || 0}
                  format={() => formatNumber(data.attribution_breakdown.security_selection, 3, '%')}
                  strokeColor={getEffectColor(data.attribution_breakdown.security_selection)}
                  size="small"
                />
              </div>
            </div>
            
            <div>
              <Text strong>Asset Allocation Effect</Text>
              <div>
                <Progress
                  percent={Math.abs(data.attribution_breakdown.asset_allocation / data.total_active_return * 100) || 0}
                  format={() => formatNumber(data.attribution_breakdown.asset_allocation, 3, '%')}
                  strokeColor={getEffectColor(data.attribution_breakdown.asset_allocation)}
                  size="small"
                />
              </div>
            </div>
            
            <div>
              <Text strong>Interaction Effect</Text>
              <div>
                <Progress
                  percent={Math.abs(data.attribution_breakdown.interaction_effect / data.total_active_return * 100) || 0}
                  format={() => formatNumber(data.attribution_breakdown.interaction_effect, 3, '%')}
                  strokeColor={getEffectColor(data.attribution_breakdown.interaction_effect)}
                  size="small"
                />
              </div>
            </div>

            {data.attribution_breakdown.currency_effect !== undefined && (
              <div>
                <Text strong>Currency Effect</Text>
                <div>
                  <Progress
                    percent={Math.abs(data.attribution_breakdown.currency_effect / data.total_active_return * 100) || 0}
                    format={() => formatNumber(data.attribution_breakdown.currency_effect, 3, '%')}
                    strokeColor={getEffectColor(data.attribution_breakdown.currency_effect)}
                    size="small"
                  />
                </div>
              </div>
            )}
          </Space>
        </Col>
      </Row>
    </Card>
  );

  const renderSectorAttributionChart = () => {
    if (sectorChartData.length === 0) {
      return <Alert message="No sector attribution data available" type="info" />;
    }

    return (
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={sectorChartData} margin={{ top: 20, right: 30, left: 20, bottom: 100 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="sector_short" 
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
          />
          <YAxis />
          <RechartsTooltip 
            formatter={(value: any, name: string, props: any) => [
              formatNumber(value as number, 3, '%'),
              name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())
            ]}
            labelFormatter={(label, payload) => {
              if (payload && payload.length > 0) {
                return payload[0].payload.sector;
              }
              return label;
            }}
          />
          <Legend />
          
          {showAllocation && (
            <Bar 
              dataKey="allocation_effect" 
              fill="#1890ff" 
              name="Allocation Effect"
              radius={[2, 2, 0, 0]}
            />
          )}
          
          {showSelection && (
            <Bar 
              dataKey="selection_effect" 
              fill="#52c41a" 
              name="Selection Effect"
              radius={[2, 2, 0, 0]}
            />
          )}
          
          <Line 
            type="monotone" 
            dataKey="total_effect" 
            stroke="#fa541c" 
            strokeWidth={3}
            name="Total Effect"
            dot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    );
  };

  const renderFactorAttributionChart = () => {
    if (factorChartData.length === 0) {
      return <Alert message="No factor attribution data available" type="info" />;
    }

    return (
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={factorChartData} margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="factor_short" 
            angle={-45}
            textAnchor="end"
            height={70}
          />
          <YAxis />
          <RechartsTooltip 
            formatter={(value: any, name: string, props: any) => [
              formatNumber(value as number, 3, '%'),
              'Contribution'
            ]}
            labelFormatter={(label, payload) => {
              if (payload && payload.length > 0) {
                return payload[0].payload.factor_name;
              }
              return label;
            }}
          />
          <Bar 
            dataKey="contribution" 
            fill={(entry: any) => getEffectColor(entry.contribution)}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderAttributionTreemap = () => {
    if (treemapData.length === 0) {
      return <Alert message="No treemap data available" type="info" />;
    }

    return (
      <ResponsiveContainer width="100%" height={300}>
        <Treemap
          data={treemapData}
          dataKey="size"
          aspectRatio={4/3}
          stroke="#fff"
          fill="#8884d8"
          content={({ x, y, width, height, name, value }: any) => (
            <g>
              <rect
                x={x}
                y={y}
                width={width}
                height={height}
                fill={value >= 0 ? '#52c41a' : '#f5222d'}
                fillOpacity={0.8}
                stroke="#fff"
                strokeWidth={2}
              />
              {width > 60 && height > 30 && (
                <>
                  <text
                    x={x + width / 2}
                    y={y + height / 2 - 6}
                    textAnchor="middle"
                    fill="#fff"
                    fontSize={12}
                    fontWeight="bold"
                  >
                    {name.length > 10 ? name.substring(0, 10) + '...' : name}
                  </text>
                  <text
                    x={x + width / 2}
                    y={y + height / 2 + 8}
                    textAnchor="middle"
                    fill="#fff"
                    fontSize={10}
                  >
                    {formatNumber(value, 2, '%')}
                  </text>
                </>
              )}
            </g>
          )}
        />
      </ResponsiveContainer>
    );
  };

  const renderSectorTable = () => {
    const columns = [
      {
        title: 'Sector',
        dataIndex: 'sector',
        key: 'sector',
        fixed: 'left' as const,
        width: 150,
        render: (text: string) => (
          <Text strong style={{ fontSize: '12px' }}>
            {text}
          </Text>
        )
      },
      {
        title: 'Portfolio Weight',
        dataIndex: 'portfolio_weight',
        key: 'portfolio_weight',
        render: (value: number) => (
          <Progress
            percent={value * 100}
            size="small"
            format={() => formatNumber(value * 100, 1, '%')}
            strokeColor="#1890ff"
          />
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.portfolio_weight - b.portfolio_weight
      },
      {
        title: 'Benchmark Weight',
        dataIndex: 'benchmark_weight',
        key: 'benchmark_weight',
        render: (value: number) => (
          <Progress
            percent={value * 100}
            size="small"
            format={() => formatNumber(value * 100, 1, '%')}
            strokeColor="#52c41a"
          />
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.benchmark_weight - b.benchmark_weight
      },
      {
        title: 'Portfolio Return',
        dataIndex: 'portfolio_return',
        key: 'portfolio_return',
        render: (value: number) => (
          <Text style={{ color: getEffectColor(value) }}>
            {formatNumber(value, 2, '%')}
          </Text>
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.portfolio_return - b.portfolio_return
      },
      {
        title: 'Benchmark Return',
        dataIndex: 'benchmark_return',
        key: 'benchmark_return',
        render: (value: number) => (
          <Text style={{ color: getEffectColor(value) }}>
            {formatNumber(value, 2, '%')}
          </Text>
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.benchmark_return - b.benchmark_return
      },
      {
        title: 'Allocation Effect',
        dataIndex: 'allocation_effect',
        key: 'allocation_effect',
        render: (value: number) => (
          <Space>
            {getEffectIcon(value)}
            <Text strong style={{ color: getEffectColor(value) }}>
              {formatNumber(value, 3, '%')}
            </Text>
          </Space>
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.allocation_effect - b.allocation_effect
      },
      {
        title: 'Selection Effect',
        dataIndex: 'selection_effect',
        key: 'selection_effect',
        render: (value: number) => (
          <Space>
            {getEffectIcon(value)}
            <Text strong style={{ color: getEffectColor(value) }}>
              {formatNumber(value, 3, '%')}
            </Text>
          </Space>
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.selection_effect - b.selection_effect
      },
      {
        title: 'Total Effect',
        dataIndex: 'total_effect',
        key: 'total_effect',
        render: (value: number) => (
          <Space>
            {getEffectIcon(value)}
            <Text strong style={{ 
              color: getEffectColor(value),
              fontSize: '14px'
            }}>
              {formatNumber(value, 3, '%')}
            </Text>
          </Space>
        ),
        sorter: (a: SectorAttribution, b: SectorAttribution) => a.total_effect - b.total_effect,
        defaultSortOrder: 'descend' as const
      }
    ];

    return (
      <Table
        dataSource={data.sector_attribution}
        columns={columns}
        rowKey="sector"
        size="small"
        scroll={{ x: 1000 }}
        pagination={{ pageSize: 10 }}
        rowClassName={(record) => 
          record.total_effect >= 0 ? 'positive-row' : 'negative-row'
        }
      />
    );
  };

  const renderFactorTable = () => {
    const columns = [
      {
        title: 'Factor',
        dataIndex: 'factor_name',
        key: 'factor_name',
        render: (text: string) => (
          <Tag color="blue">{text}</Tag>
        )
      },
      {
        title: 'Factor Exposure',
        dataIndex: 'factor_exposure',
        key: 'factor_exposure',
        render: (value: number) => formatNumber(value, 4),
        sorter: (a: FactorAttribution, b: FactorAttribution) => a.factor_exposure - b.factor_exposure
      },
      {
        title: 'Factor Return',
        dataIndex: 'factor_return',
        key: 'factor_return',
        render: (value: number) => (
          <Text style={{ color: getEffectColor(value) }}>
            {formatNumber(value, 2, '%')}
          </Text>
        ),
        sorter: (a: FactorAttribution, b: FactorAttribution) => a.factor_return - b.factor_return
      },
      {
        title: 'Contribution',
        dataIndex: 'contribution',
        key: 'contribution',
        render: (value: number) => (
          <Space>
            {getEffectIcon(value)}
            <Text strong style={{ color: getEffectColor(value) }}>
              {formatNumber(value, 3, '%')}
            </Text>
          </Space>
        ),
        sorter: (a: FactorAttribution, b: FactorAttribution) => a.contribution - b.contribution,
        defaultSortOrder: 'descend' as const
      }
    ];

    return (
      <Table
        dataSource={data.factor_attribution}
        columns={columns}
        rowKey="factor_name"
        size="small"
        pagination={false}
      />
    );
  };

  return (
    <div style={{ height }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* Attribution Summary */}
        {renderAttributionSummary()}

        {/* Controls */}
        <Card>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Text strong>Attribution Type:</Text>
                <Select
                  value={data.attribution_type}
                  onChange={onAttributionTypeChange}
                  style={{ width: 120 }}
                >
                  <Option value="sector">Sector</Option>
                  <Option value="factor">Factor</Option>
                  <Option value="security">Security</Option>
                  <Option value="style">Style</Option>
                </Select>

                <Text strong>Period:</Text>
                <Text type="secondary">
                  {data.period_start} to {data.period_end}
                </Text>
              </Space>
            </Col>
            
            <Col>
              <Space>
                <Text strong>View:</Text>
                <Button.Group>
                  <Button
                    type={viewType === 'chart' ? 'primary' : 'default'}
                    icon={<BarChartOutlined />}
                    onClick={() => setViewType('chart')}
                  >
                    Chart
                  </Button>
                  <Button
                    type={viewType === 'table' ? 'primary' : 'default'}
                    icon={<TableOutlined />}
                    onClick={() => setViewType('table')}
                  >
                    Table
                  </Button>
                </Button.Group>

                {viewType === 'chart' && data.attribution_type === 'sector' && (
                  <Space>
                    <Switch
                      checked={showAllocation}
                      onChange={setShowAllocation}
                      size="small"
                    />
                    <Text>Allocation</Text>
                    <Switch
                      checked={showSelection}
                      onChange={setShowSelection}
                      size="small"
                    />
                    <Text>Selection</Text>
                  </Space>
                )}
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Main Attribution Analysis */}
        {viewType === 'chart' ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={data.attribution_type === 'sector' ? 16 : 24}>
              <Card 
                title={`${data.attribution_type.charAt(0).toUpperCase() + data.attribution_type.slice(1)} Attribution Chart`}
                extra={<LineChartOutlined />}
              >
                {data.attribution_type === 'sector' ? renderSectorAttributionChart() : renderFactorAttributionChart()}
              </Card>
            </Col>
            
            {data.attribution_type === 'sector' && (
              <Col xs={24} lg={8}>
                <Card title="Attribution Treemap" extra={<PieChartOutlined />}>
                  {renderAttributionTreemap()}
                </Card>
              </Col>
            )}
          </Row>
        ) : (
          <Card 
            title={`${data.attribution_type.charAt(0).toUpperCase() + data.attribution_type.slice(1)} Attribution Table`}
            extra={<TableOutlined />}
          >
            {data.attribution_type === 'sector' ? renderSectorTable() : renderFactorTable()}
          </Card>
        )}
      </Space>
    </div>
  );
};

export default AttributionDashboard;