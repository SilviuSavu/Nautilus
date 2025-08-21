/**
 * Connection Health Dashboard
 * Real-time visualization of connection quality and health metrics across venues
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Row, Col, Table, Progress, Tag, Statistic, Alert, Badge, Tooltip, Button, Space } from 'antd';
import { 
  LinkIcon, 
  SignalIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  WifiIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';
import { ConnectionQuality } from '../../types/monitoring';
import { connectionQualityScorer, QualityScore } from '../../services/monitoring/ConnectionQualityScorer';

interface ConnectionHealthDashboardProps {
  refreshInterval?: number;
  showDetails?: boolean;
  compactMode?: boolean;
}

interface VenueConnectionData {
  venue_name: string;
  quality: ConnectionQuality;
  score: QualityScore;
  trend_data: {
    timestamp: string;
    quality_score: number;
    response_time: number;
    uptime_percent: number;
  }[];
}

export const ConnectionHealthDashboard: React.FC<ConnectionHealthDashboardProps> = ({
  refreshInterval = 10000,
  showDetails = true,
  compactMode = false
}) => {
  const [venueConnections, setVenueConnections] = useState<VenueConnectionData[]>([]);
  const [overallHealth, setOverallHealth] = useState<{
    total_venues: number;
    connected_venues: number;
    degraded_venues: number;
    overall_score: number;
  }>({ total_venues: 0, connected_venues: 0, degraded_venues: 0, overall_score: 100 });

  // Generate mock connection data for demo
  const generateMockConnectionData = (): VenueConnectionData[] => {
    const venues = ['IB Gateway', 'Market Data Feed', 'Database', 'Redis Cache', 'WebSocket API', 'External API'];
    
    return venues.map((venue, index) => {
      const status = Math.random() > 0.2 ? 'connected' : (Math.random() > 0.5 ? 'degraded' : 'disconnected');
      const baseLatency = 20 + (index * 15) + (Math.random() * 30);
      const uptime = status === 'connected' ? 95 + (Math.random() * 5) : 
                   status === 'degraded' ? 85 + (Math.random() * 10) : 
                   Math.random() * 50;
      
      const quality: ConnectionQuality = {
        venue_name: venue,
        status: status as any,
        quality_score: Math.round((uptime / 100) * 80 + (Math.random() * 20)),
        uptime_percent_24h: Math.round(uptime * 100) / 100,
        connection_duration_seconds: status === 'connected' ? Math.round(Math.random() * 86400) : 0,
        last_disconnect_time: status !== 'connected' ? new Date(Date.now() - Math.random() * 3600000).toISOString() : undefined,
        disconnect_count_24h: Math.round(Math.random() * 5),
        data_quality: {
          message_rate_per_sec: status === 'connected' ? Math.round(Math.random() * 100 + 50) : 0,
          duplicate_messages_percent: Math.round(Math.random() * 2 * 100) / 100,
          out_of_sequence_percent: Math.round(Math.random() * 1 * 100) / 100,
          stale_data_percent: Math.round(Math.random() * 3 * 100) / 100
        },
        performance_metrics: {
          response_time_ms: Math.round(baseLatency * 100) / 100,
          throughput_mbps: Math.round(Math.random() * 50 * 100) / 100,
          error_rate_percent: Math.round(Math.random() * 2 * 100) / 100
        },
        reconnection_stats: {
          auto_reconnect_enabled: true,
          reconnect_attempts_24h: Math.round(Math.random() * 3),
          avg_reconnect_time_seconds: Math.round((Math.random() * 30 + 10) * 100) / 100,
          max_reconnect_time_seconds: Math.round((Math.random() * 60 + 20) * 100) / 100
        }
      };

      const score = connectionQualityScorer.calculateQualityScore(quality);
      
      // Generate trend data
      const trendData = Array.from({ length: 20 }, (_, i) => {
        const timestamp = new Date(Date.now() - (19 - i) * 30000).toISOString();
        const baseScore = score.overall_score;
        const variation = (Math.random() - 0.5) * 20;
        
        return {
          timestamp,
          quality_score: Math.max(0, Math.min(100, baseScore + variation)),
          response_time: baseLatency + (Math.random() - 0.5) * 20,
          uptime_percent: Math.max(80, Math.min(100, uptime + (Math.random() - 0.5) * 10))
        };
      });

      return {
        venue_name: venue,
        quality,
        score,
        trend_data: trendData
      };
    });
  };

  // Calculate overall health metrics
  const calculateOverallHealth = (venues: VenueConnectionData[]) => {
    const totalVenues = venues.length;
    const connectedVenues = venues.filter(v => v.quality.status === 'connected').length;
    const degradedVenues = venues.filter(v => v.quality.status === 'degraded').length;
    const overallScore = totalVenues > 0 ? 
      venues.reduce((sum, v) => sum + v.score.overall_score, 0) / totalVenues : 0;

    return {
      total_venues: totalVenues,
      connected_venues: connectedVenues,
      degraded_venues: degradedVenues,
      overall_score: Math.round(overallScore * 100) / 100
    };
  };

  useEffect(() => {
    const updateConnectionData = () => {
      const mockData = generateMockConnectionData();
      setVenueConnections(mockData);
      setOverallHealth(calculateOverallHealth(mockData));
    };

    // Initial load
    updateConnectionData();

    // Set up refresh interval
    const interval = setInterval(updateConnectionData, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  // Table columns for connection details
  const connectionColumns = [
    {
      title: 'Venue',
      dataIndex: 'venue_name',
      key: 'venue_name',
      render: (venue: string, record: VenueConnectionData) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <LinkIcon className="w-4 h-4" />
          <span style={{ fontWeight: 500 }}>{venue}</span>
          <Badge 
            status={record.quality.status === 'connected' ? 'success' : 
                   record.quality.status === 'degraded' ? 'warning' : 'error'} 
          />
        </div>
      )
    },
    {
      title: 'Quality Score',
      key: 'quality_score',
      render: (record: VenueConnectionData) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Progress
            type="circle"
            size={40}
            percent={record.score.overall_score}
            strokeColor={
              record.score.overall_score >= 90 ? '#52c41a' :
              record.score.overall_score >= 75 ? '#fadb14' :
              record.score.overall_score >= 60 ? '#fa8c16' : '#ff4d4f'
            }
            format={(percent) => percent?.toFixed(0)}
          />
          <div>
            <div style={{ fontWeight: 500 }}>{record.score.overall_score.toFixed(1)}</div>
            <div style={{ fontSize: 12, color: '#666' }}>Grade: {record.score.grade}</div>
          </div>
        </div>
      ),
      sorter: (a: VenueConnectionData, b: VenueConnectionData) => a.score.overall_score - b.score.overall_score
    },
    {
      title: 'Status',
      dataIndex: ['quality', 'status'],
      key: 'status',
      render: (status: string, record: VenueConnectionData) => {
        const colors = {
          connected: 'success',
          degraded: 'warning',
          disconnected: 'error',
          reconnecting: 'processing'
        };
        
        return (
          <Tag color={colors[status as keyof typeof colors]}>
            {status.toUpperCase()}
          </Tag>
        );
      }
    },
    {
      title: 'Uptime',
      dataIndex: ['quality', 'uptime_percent_24h'],
      key: 'uptime',
      render: (uptime: number) => (
        <div style={{ minWidth: 100 }}>
          <Progress
            percent={uptime}
            size="small"
            strokeColor={uptime >= 99 ? '#52c41a' : uptime >= 95 ? '#fadb14' : '#ff4d4f'}
          />
          <div style={{ fontSize: 12, marginTop: 2 }}>{uptime.toFixed(2)}%</div>
        </div>
      ),
      sorter: (a: VenueConnectionData, b: VenueConnectionData) => a.quality.uptime_percent_24h - b.quality.uptime_percent_24h
    },
    {
      title: 'Latency',
      dataIndex: ['quality', 'performance_metrics', 'response_time_ms'],
      key: 'latency',
      render: (latency: number) => {
        const color = latency < 50 ? '#52c41a' : latency < 100 ? '#fadb14' : '#ff4d4f';
        return (
          <span style={{ color, fontWeight: 500 }}>
            {latency.toFixed(1)}ms
          </span>
        );
      },
      sorter: (a: VenueConnectionData, b: VenueConnectionData) => 
        a.quality.performance_metrics.response_time_ms - b.quality.performance_metrics.response_time_ms
    },
    {
      title: 'Error Rate',
      dataIndex: ['quality', 'performance_metrics', 'error_rate_percent'],
      key: 'error_rate',
      render: (errorRate: number) => {
        const color = errorRate < 1 ? '#52c41a' : errorRate < 3 ? '#fadb14' : '#ff4d4f';
        return (
          <span style={{ color }}>
            {errorRate.toFixed(2)}%
          </span>
        );
      },
      sorter: (a: VenueConnectionData, b: VenueConnectionData) => 
        a.quality.performance_metrics.error_rate_percent - b.quality.performance_metrics.error_rate_percent
    }
  ];

  if (!compactMode) {
    connectionColumns.push({
      title: 'Actions',
      key: 'actions',
      render: (record: VenueConnectionData) => (
        <Space>
          <Button size="small" type="link">Details</Button>
          {record.quality.status === 'disconnected' && (
            <Button size="small" type="link">Reconnect</Button>
          )}
        </Space>
      )
    });
  }

  // Critical alerts
  const criticalConnections = venueConnections.filter(v => 
    v.quality.status === 'disconnected' || v.score.overall_score < 60
  );

  return (
    <div className="connection-health-dashboard">
      {/* Critical Alerts */}
      {criticalConnections.length > 0 && (
        <Alert
          type="error"
          showIcon
          icon={<ExclamationTriangleIcon className="w-4 h-4" />}
          message={`${criticalConnections.length} Connection Issues Detected`}
          description={
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {criticalConnections.map((conn, index) => (
                <li key={index}>
                  <strong>{conn.venue_name}</strong>: {conn.quality.status === 'disconnected' ? 'Disconnected' : `Poor quality (${conn.score.overall_score.toFixed(1)})`}
                </li>
              ))}
            </ul>
          }
          style={{ marginBottom: 16 }}
          closable
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Overall Health Summary */}
        <Col xs={24} sm={12} lg={6}>
          <Card title="Overall Health" size={compactMode ? 'small' : 'default'}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Progress
                type="circle"
                percent={overallHealth.overall_score}
                strokeColor={
                  overallHealth.overall_score >= 90 ? '#52c41a' :
                  overallHealth.overall_score >= 75 ? '#fadb14' :
                  overallHealth.overall_score >= 60 ? '#fa8c16' : '#ff4d4f'
                }
                width={compactMode ? 80 : 120}
                format={(percent) => `${percent?.toFixed(0)}`}
              />
            </div>
            
            <Row gutter={8}>
              <Col span={12}>
                <Statistic
                  title="Connected"
                  value={overallHealth.connected_venues}
                  suffix={`/ ${overallHealth.total_venues}`}
                  prefix={<CheckCircleIcon className="w-4 h-4" />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Issues"
                  value={overallHealth.degraded_venues + (overallHealth.total_venues - overallHealth.connected_venues - overallHealth.degraded_venues)}
                  prefix={<ExclamationTriangleIcon className="w-4 h-4" />}
                  valueStyle={{ 
                    color: (overallHealth.degraded_venues + (overallHealth.total_venues - overallHealth.connected_venues - overallHealth.degraded_venues)) > 0 ? '#ff4d4f' : '#52c41a' 
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Top Venue Cards */}
        {!compactMode && venueConnections.slice(0, 3).map((venue, index) => (
          <Col xs={24} sm={12} lg={6} key={venue.venue_name}>
            <Card 
              title={venue.venue_name}
              size="small"
              extra={
                <Tag color={
                  venue.quality.status === 'connected' ? 'success' : 
                  venue.quality.status === 'degraded' ? 'warning' : 'error'
                }>
                  {venue.quality.status.toUpperCase()}
                </Tag>
              }
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: 24, fontWeight: 'bold' }}>
                    {venue.score.overall_score.toFixed(0)}
                  </div>
                  <div style={{ fontSize: 12, color: '#666' }}>
                    Quality Score
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                    <ClockIcon className="w-3 h-3" />
                    <span style={{ fontSize: 12 }}>
                      {venue.quality.performance_metrics.response_time_ms.toFixed(0)}ms
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <SignalIcon className="w-3 h-3" />
                    <span style={{ fontSize: 12 }}>
                      {venue.quality.uptime_percent_24h.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        {/* Quality Score Trends */}
        <Col xs={24} lg={12}>
          <Card title="Quality Score Trends" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp"
                  type="category"
                  scale="point"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis 
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                />
                <RechartsTooltip 
                  formatter={(value: number, name: string) => [`${value.toFixed(1)}`, name]}
                  labelFormatter={(label) => `Time: ${new Date(label).toLocaleTimeString()}`}
                />
                {venueConnections.slice(0, 3).map((venue, index) => (
                  <Line
                    key={venue.venue_name}
                    data={venue.trend_data}
                    type="monotone"
                    dataKey="quality_score"
                    stroke={['#1890ff', '#52c41a', '#fa8c16'][index]}
                    strokeWidth={2}
                    dot={false}
                    name={venue.venue_name}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Response Time vs Quality */}
        <Col xs={24} lg={12}>
          <Card title="Response Time vs Quality Score" size={compactMode ? 'small' : 'default'}>
            <ResponsiveContainer width="100%" height={compactMode ? 200 : 300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="response_time_ms"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Response Time (ms)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="quality_score"
                  type="number"
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Quality Score', angle: -90, position: 'insideLeft' }}
                />
                <RechartsTooltip 
                  formatter={(value: number, name: string) => [
                    name === 'quality_score' ? `${value.toFixed(1)}` : `${value.toFixed(1)}ms`,
                    name === 'quality_score' ? 'Quality Score' : 'Response Time'
                  ]}
                  labelFormatter={() => ''}
                />
                <Scatter
                  data={venueConnections.map(v => ({
                    response_time_ms: v.quality.performance_metrics.response_time_ms,
                    quality_score: v.score.overall_score,
                    venue_name: v.venue_name
                  }))}
                  fill="#1890ff"
                >
                  {venueConnections.map((venue, index) => (
                    <Cell 
                      key={`cell-${index}`}
                      fill={
                        venue.quality.status === 'connected' ? '#52c41a' :
                        venue.quality.status === 'degraded' ? '#fa8c16' : '#ff4d4f'
                      }
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* Detailed Connection Table */}
        {showDetails && (
          <Col xs={24}>
            <Card title="Connection Details" size={compactMode ? 'small' : 'default'}>
              <Table
                columns={connectionColumns}
                dataSource={venueConnections}
                rowKey="venue_name"
                size={compactMode ? 'small' : 'default'}
                pagination={false}
                scroll={{ x: 1000 }}
              />
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default ConnectionHealthDashboard;