/**
 * Advanced Volatility Forecasting Dashboard
 * Integrates with Volatility Engine (Port 8001/volatility endpoints)
 * Based on FRONTEND_ENDPOINT_INTEGRATION_GUIDE.md
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Spin,
  Alert,
  Select,
  Button,
  Form,
  InputNumber,
  Tag,
  Progress,
  Tabs,
  Space,
  Typography
} from 'antd';
import {
  ThunderboltOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  CloudSyncOutlined,
  ControlOutlined,
  SyncOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import apiClient from '../../services/apiClient';
import { volatilityWS, ConnectionStatus } from '../../services/websocketClient';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

// Interfaces based on API specifications
interface VolatilityStatus {
  engine_state: string;
  active_symbols: string[];
  models: { total: number; trained: number };
  performance: { avg_response_ms: number };
}

interface VolatilityModels {
  available_models: string[];
  model_capabilities: Record<string, any>;
}

interface VolatilityForecast {
  forecast: {
    ensemble_volatility: number;
    confidence_bounds: Record<string, number>;
    model_contributions: Record<string, number>;
    next_day_prediction: number;
  };
  generated_at: string;
  valid_until: string;
}

interface StreamingStatus {
  messagebus_connected: boolean;
  active_symbols: string[];
  events_processed: number;
  volatility_updates_triggered: number;
  streaming_performance: { events_per_second: number };
}

interface HardwareAcceleration {
  m4_max_available: boolean;
  neural_engine: { available: boolean; utilization: number };
  metal_gpu: { available: boolean; utilization: number };
  optimization_active: boolean;
}

const VolatilityDashboard: React.FC = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [volatilityStatus, setVolatilityStatus] = useState<VolatilityStatus | null>(null);
  const [availableModels, setAvailableModels] = useState<VolatilityModels | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<StreamingStatus | null>(null);
  const [hardwareStatus, setHardwareStatus] = useState<HardwareAcceleration | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL');
  const [forecastData, setForecastData] = useState<VolatilityForecast | null>(null);
  const [realtimeData, setRealtimeData] = useState<any[]>([]);
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [error, setError] = useState<string | null>(null);

  // Form instance
  const [form] = Form.useForm();

  // Load initial data
  useEffect(() => {
    loadVolatilityData();
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (selectedSymbol) {
      connectToRealtimeUpdates();
    }
    
    return () => {
      volatilityWS.disconnect(selectedSymbol);
    };
  }, [selectedSymbol]);

  const loadVolatilityData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const [status, models, streaming, hardware] = await Promise.all([
        apiClient.getVolatilityStatus(),
        apiClient.getVolatilityModels(),
        apiClient.getVolatilityStreamingStatus(),
        apiClient.getHardwareAccelerationStatus()
      ]);

      setVolatilityStatus(status);
      setAvailableModels(models);
      setStreamingStatus(streaming);
      setHardwareStatus(hardware);

    } catch (err) {
      setError(`Failed to load volatility data: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const connectToRealtimeUpdates = async () => {
    try {
      await volatilityWS.connectToVolatilityUpdates(
        selectedSymbol,
        (update) => {
          // Add real-time update to chart data
          setRealtimeData(prev => [
            ...prev.slice(-99), // Keep last 100 points
            {
              timestamp: new Date(update.timestamp).toLocaleTimeString(),
              volatility: update.data.current_volatility,
              confidence: update.data.confidence,
              trigger: update.data.trigger_reason
            }
          ]);
        },
        (status) => {
          setWsStatus(status);
        }
      );
    } catch (err) {
      console.error('Failed to connect to real-time volatility updates:', err);
    }
  };

  const handleAddSymbol = async (values: any) => {
    setLoading(true);
    try {
      await apiClient.addVolatilitySymbol(selectedSymbol, {
        model_types: values.models,
        training_params: values.trainingParams
      });
      
      loadVolatilityData(); // Refresh data
    } catch (err) {
      setError(`Failed to add symbol: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModels = async (values: any) => {
    setLoading(true);
    try {
      await apiClient.trainVolatilityModels(selectedSymbol, {
        models: values.models,
        lookback_days: values.lookbackDays,
        hardware_acceleration: true
      });
      
      loadVolatilityData(); // Refresh data
    } catch (err) {
      setError(`Failed to train models: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateForecast = async (values: any) => {
    setLoading(true);
    try {
      const forecast = await apiClient.forecastVolatility(selectedSymbol, {
        horizon_days: values.horizonDays,
        confidence_levels: values.confidenceLevels
      });
      
      setForecastData(forecast);
    } catch (err) {
      setError(`Failed to generate forecast: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const getConnectionStatusColor = (status: ConnectionStatus) => {
    switch (status) {
      case ConnectionStatus.CONNECTED: return 'green';
      case ConnectionStatus.CONNECTING: return 'orange';
      case ConnectionStatus.RECONNECTING: return 'orange';
      case ConnectionStatus.ERROR: return 'red';
      default: return 'gray';
    }
  };

  const getEngineStatusColor = (state: string) => {
    switch (state) {
      case 'running': return 'green';
      case 'starting': return 'orange';
      case 'stopped': return 'red';
      default: return 'gray';
    }
  };

  if (loading && !volatilityStatus) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
        <Spin size="large" tip="Loading Volatility Engine..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ThunderboltOutlined style={{ marginRight: '8px' }} />
        Advanced Volatility Forecasting Engine
      </Title>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* Status Overview */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Engine Status"
              value={volatilityStatus?.engine_state || 'Unknown'}
              prefix={<ControlOutlined />}
              valueStyle={{ color: getEngineStatusColor(volatilityStatus?.engine_state || '') }}
            />
            <Tag color={getEngineStatusColor(volatilityStatus?.engine_state || '')}>
              {volatilityStatus?.engine_state || 'Unknown'}
            </Tag>
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Symbols"
              value={volatilityStatus?.active_symbols?.length || 0}
              prefix={<LineChartOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Models Trained"
              value={`${volatilityStatus?.models?.trained || 0}/${volatilityStatus?.models?.total || 0}`}
              prefix={<ExperimentOutlined />}
            />
            <Progress 
              percent={volatilityStatus?.models?.total ? 
                (volatilityStatus.models.trained / volatilityStatus.models.total) * 100 : 0
              } 
              size="small" 
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="WebSocket Status"
              value={wsStatus}
              prefix={<CloudSyncOutlined />}
              valueStyle={{ color: getConnectionStatusColor(wsStatus) }}
            />
            <Tag color={getConnectionStatusColor(wsStatus)}>
              {wsStatus}
            </Tag>
          </Card>
        </Col>
      </Row>

      {/* Hardware Acceleration Status */}
      {hardwareStatus && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={24}>
            <Card title="M4 Max Hardware Acceleration Status">
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="M4 Max Available"
                    value={hardwareStatus.m4_max_available ? 'Yes' : 'No'}
                    valueStyle={{ color: hardwareStatus.m4_max_available ? 'green' : 'red' }}
                  />
                </Col>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="Neural Engine"
                    value={`${hardwareStatus.neural_engine?.utilization || 0}%`}
                    suffix="utilization"
                  />
                  <Progress percent={hardwareStatus.neural_engine?.utilization || 0} />
                </Col>
                <Col xs={24} sm={8}>
                  <Statistic
                    title="Metal GPU"
                    value={`${hardwareStatus.metal_gpu?.utilization || 0}%`}
                    suffix="utilization"
                  />
                  <Progress percent={hardwareStatus.metal_gpu?.utilization || 0} />
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultActiveKey="1">
        <TabPane tab="Real-time Forecasting" key="1">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="Real-time Volatility Updates" 
                    extra={
                      <Space>
                        <Select value={selectedSymbol} onChange={setSelectedSymbol} style={{ width: 120 }}>
                          <Option value="AAPL">AAPL</Option>
                          <Option value="GOOGL">GOOGL</Option>
                          <Option value="MSFT">MSFT</Option>
                          <Option value="TSLA">TSLA</Option>
                        </Select>
                        <Button icon={<SyncOutlined />} onClick={loadVolatilityData}>
                          Refresh
                        </Button>
                      </Space>
                    }>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={realtimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="volatility" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="confidence" stroke="#82ca9d" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            
            <Col xs={24} lg={8}>
              <Card title="Current Forecast" style={{ marginBottom: '16px' }}>
                {forecastData ? (
                  <div>
                    <Statistic
                      title="Next Day Prediction"
                      value={forecastData.forecast.next_day_prediction}
                      precision={4}
                      suffix="%"
                    />
                    <Statistic
                      title="Ensemble Volatility"
                      value={forecastData.forecast.ensemble_volatility}
                      precision={4}
                      suffix="%"
                    />
                    <Text type="secondary">
                      Generated: {new Date(forecastData.generated_at).toLocaleString()}
                    </Text>
                  </div>
                ) : (
                  <Text type="secondary">No forecast data available</Text>
                )}
              </Card>

              <Card title="Streaming Statistics">
                <Statistic
                  title="Events Processed"
                  value={streamingStatus?.events_processed || 0}
                />
                <Statistic
                  title="Events/Second"
                  value={streamingStatus?.streaming_performance?.events_per_second || 0}
                />
                <Statistic
                  title="Updates Triggered"
                  value={streamingStatus?.volatility_updates_triggered || 0}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Model Management" key="2">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Add Symbol for Tracking">
                <Form form={form} layout="vertical" onFinish={handleAddSymbol}>
                  <Form.Item name="symbol" label="Symbol" initialValue={selectedSymbol}>
                    <Select onChange={setSelectedSymbol}>
                      <Option value="AAPL">AAPL</Option>
                      <Option value="GOOGL">GOOGL</Option>
                      <Option value="MSFT">MSFT</Option>
                      <Option value="TSLA">TSLA</Option>
                    </Select>
                  </Form.Item>
                  <Form.Item name="models" label="Model Types" initialValue={['garch', 'lstm']}>
                    <Select mode="multiple">
                      {availableModels?.available_models?.map(model => (
                        <Option key={model} value={model}>{model.toUpperCase()}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      Add Symbol
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Train Models">
                <Form layout="vertical" onFinish={handleTrainModels}>
                  <Form.Item name="models" label="Models to Train" initialValue={['garch', 'lstm']}>
                    <Select mode="multiple">
                      {availableModels?.available_models?.map(model => (
                        <Option key={model} value={model}>{model.toUpperCase()}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                  <Form.Item name="lookbackDays" label="Lookback Days" initialValue={252}>
                    <InputNumber min={30} max={1000} style={{ width: '100%' }} />
                  </Form.Item>
                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      Train Models
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="Generate Forecast" key="3">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="Forecast Parameters">
                <Form layout="vertical" onFinish={handleGenerateForecast}>
                  <Form.Item name="horizonDays" label="Forecast Horizon (Days)" initialValue={30}>
                    <InputNumber min={1} max={252} style={{ width: '100%' }} />
                  </Form.Item>
                  <Form.Item name="confidenceLevels" label="Confidence Levels" initialValue={[0.95, 0.99]}>
                    <Select mode="tags" style={{ width: '100%' }}>
                      <Option value={0.90}>90%</Option>
                      <Option value={0.95}>95%</Option>
                      <Option value={0.99}>99%</Option>
                    </Select>
                  </Form.Item>
                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      Generate Forecast
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="Model Contributions">
                {forecastData?.forecast?.model_contributions && (
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={
                      Object.entries(forecastData.forecast.model_contributions).map(([model, contribution]) => ({
                        model: model.toUpperCase(),
                        contribution: contribution as number
                      }))
                    }>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="contribution" stroke="#8884d8" fill="#8884d8" />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default VolatilityDashboard;