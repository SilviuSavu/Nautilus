/**
 * AI-Powered Pattern Recognition Component
 * Advanced machine learning-based pattern detection and analysis
 */

import React, { useState, useEffect, useCallback } from 'react'
import { 
  Card, 
  Button, 
  Space, 
  Progress, 
  Switch, 
  Slider,
  Select,
  Table,
  Tag,
  Alert,
  Tabs,
  Tooltip,
  Modal,
  Spin,
  message,
  Row,
  Col,
  Statistic
} from 'antd'
import { 
  RobotOutlined, 
  BrainOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  InfoCircleOutlined,
  ExperimentOutlined,
  ArrowUpOutlined
} from '@ant-design/icons'
import { Line } from '@ant-design/plots'
import { ChartPattern, OHLCVData } from '../../types/charting'

const { Option } = Select

interface AIModel {
  id: string
  name: string
  type: 'cnn' | 'lstm' | 'transformer' | 'ensemble'
  description: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  trainingData: number
  status: 'training' | 'ready' | 'disabled'
  lastTrained: string
  version: string
}

interface AIDetection {
  id: string
  patternType: string
  confidence: number
  modelId: string
  prediction: 'bullish' | 'bearish' | 'neutral'
  probabilityDistribution: Record<string, number>
  supportingFeatures: string[]
  timeHorizon: number
  targetPrice?: number
  stopLoss?: number
  detectedAt: string
  instrument: string
  timeframe: string
}

interface AIConfiguration {
  enabledModels: string[]
  confidenceThreshold: number
  ensembleVoting: 'majority' | 'weighted' | 'unanimous'
  featureEngineering: {
    technicalIndicators: boolean
    priceAction: boolean
    volumeProfile: boolean
    marketSentiment: boolean
    multiTimeframe: boolean
  }
  realTimeProcessing: boolean
  adaptiveLearning: boolean
}

export const AIPatternRecognition: React.FC = () => {
  const [models, setModels] = useState<AIModel[]>([])
  const [detections, setDetections] = useState<AIDetection[]>([])
  const [config, setConfig] = useState<AIConfiguration>({
    enabledModels: [],
    confidenceThreshold: 0.7,
    ensembleVoting: 'weighted',
    featureEngineering: {
      technicalIndicators: true,
      priceAction: true,
      volumeProfile: false,
      marketSentiment: false,
      multiTimeframe: true
    },
    realTimeProcessing: false,
    adaptiveLearning: false
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedModel, setSelectedModel] = useState<AIModel | null>(null)
  const [isModelModalVisible, setIsModelModalVisible] = useState(false)
  const [processingStats, setProcessingStats] = useState({
    patternsProcessed: 0,
    detectionsFound: 0,
    avgConfidence: 0,
    processingSpeed: 0
  })

  useEffect(() => {
    initializeModels()
    loadDetections()
  }, [])

  const initializeModels = () => {
    const mockModels: AIModel[] = [
      {
        id: 'cnn_v1',
        name: 'Convolutional Neural Network v1.0',
        type: 'cnn',
        description: 'Deep learning model specialized in visual pattern recognition using convolutional layers',
        accuracy: 0.84,
        precision: 0.82,
        recall: 0.79,
        f1Score: 0.80,
        trainingData: 50000,
        status: 'ready',
        lastTrained: '2025-08-15T10:30:00.000Z',
        version: '1.0.2'
      },
      {
        id: 'lstm_v2',
        name: 'Long Short-Term Memory v2.1',
        type: 'lstm',
        description: 'Recurrent neural network for sequential pattern analysis with memory capabilities',
        accuracy: 0.78,
        precision: 0.81,
        recall: 0.75,
        f1Score: 0.78,
        trainingData: 75000,
        status: 'ready',
        lastTrained: '2025-08-18T14:20:00.000Z',
        version: '2.1.0'
      },
      {
        id: 'transformer_beta',
        name: 'Transformer Architecture (Beta)',
        type: 'transformer',
        description: 'State-of-the-art attention-based model for complex pattern relationships',
        accuracy: 0.87,
        precision: 0.85,
        recall: 0.83,
        f1Score: 0.84,
        trainingData: 100000,
        status: 'training',
        lastTrained: '2025-08-20T09:15:00.000Z',
        version: '0.9.5'
      },
      {
        id: 'ensemble_pro',
        name: 'Ensemble Pro Model',
        type: 'ensemble',
        description: 'Advanced ensemble combining multiple models for maximum accuracy',
        accuracy: 0.91,
        precision: 0.89,
        recall: 0.87,
        f1Score: 0.88,
        trainingData: 200000,
        status: 'ready',
        lastTrained: '2025-08-19T16:45:00.000Z',
        version: '3.2.1'
      }
    ]

    setModels(mockModels)
    setConfig(prev => ({
      ...prev,
      enabledModels: mockModels.filter(m => m.status === 'ready').map(m => m.id)
    }))
  }

  const loadDetections = () => {
    // Mock AI detections
    const mockDetections: AIDetection[] = [
      {
        id: 'ai_det_1',
        patternType: 'Head and Shoulders',
        confidence: 0.89,
        modelId: 'ensemble_pro',
        prediction: 'bearish',
        probabilityDistribution: { bullish: 0.15, bearish: 0.75, neutral: 0.10 },
        supportingFeatures: ['volume_divergence', 'momentum_shift', 'resistance_test'],
        timeHorizon: 14,
        targetPrice: 145.30,
        stopLoss: 158.20,
        detectedAt: new Date().toISOString(),
        instrument: 'AAPL',
        timeframe: '1d'
      },
      {
        id: 'ai_det_2',
        patternType: 'Bull Flag',
        confidence: 0.76,
        modelId: 'cnn_v1',
        prediction: 'bullish',
        probabilityDistribution: { bullish: 0.68, bearish: 0.12, neutral: 0.20 },
        supportingFeatures: ['strong_momentum', 'volume_confirmation', 'trend_continuation'],
        timeHorizon: 7,
        targetPrice: 285.50,
        stopLoss: 268.90,
        detectedAt: new Date(Date.now() - 3600000).toISOString(),
        instrument: 'MSFT',
        timeframe: '4h'
      }
    ]

    setDetections(mockDetections)
  }

  const startAIProcessing = useCallback(async () => {
    if (config.enabledModels.length === 0) {
      message.error('Please enable at least one AI model')
      return
    }

    setIsProcessing(true)
    setProcessingStats({
      patternsProcessed: 0,
      detectionsFound: 0,
      avgConfidence: 0,
      processingSpeed: 0
    })

    // Simulate AI processing
    const interval = setInterval(() => {
      setProcessingStats(prev => ({
        patternsProcessed: prev.patternsProcessed + Math.floor(Math.random() * 10) + 5,
        detectionsFound: prev.detectionsFound + (Math.random() > 0.8 ? 1 : 0),
        avgConfidence: 0.7 + Math.random() * 0.2,
        processingSpeed: 150 + Math.random() * 100
      }))
    }, 1000)

    // Stop after 10 seconds
    setTimeout(() => {
      clearInterval(interval)
      setIsProcessing(false)
      message.success('AI pattern processing completed')
      
      // Add some new detections
      const newDetections: AIDetection[] = [
        {
          id: `ai_det_${Date.now()}`,
          patternType: 'Double Bottom',
          confidence: 0.82,
          modelId: 'lstm_v2',
          prediction: 'bullish',
          probabilityDistribution: { bullish: 0.72, bearish: 0.18, neutral: 0.10 },
          supportingFeatures: ['support_bounce', 'volume_spike', 'oversold_conditions'],
          timeHorizon: 10,
          targetPrice: 3250.00,
          stopLoss: 3180.00,
          detectedAt: new Date().toISOString(),
          instrument: 'GOOGL',
          timeframe: '1d'
        }
      ]
      
      setDetections(prev => [...newDetections, ...prev])
    }, 10000)
  }, [config.enabledModels])

  const stopAIProcessing = () => {
    setIsProcessing(false)
    message.info('AI processing stopped')
  }

  const retrainModel = (modelId: string) => {
    setModels(prev => prev.map(model => 
      model.id === modelId 
        ? { ...model, status: 'training', lastTrained: new Date().toISOString() }
        : model
    ))

    // Simulate training completion
    setTimeout(() => {
      setModels(prev => prev.map(model => 
        model.id === modelId 
          ? { 
              ...model, 
              status: 'ready',
              accuracy: Math.min(0.95, model.accuracy + Math.random() * 0.05),
              version: `${model.version.split('.')[0]}.${parseInt(model.version.split('.')[1]) + 1}.0`
            }
          : model
      ))
      message.success('Model retrained successfully')
    }, 5000)
  }

  const modelColumns = [
    {
      title: 'Model',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: AIModel) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            <Space>
              <Tag color="blue">{record.type.toUpperCase()}</Tag>
              <span>v{record.version}</span>
            </Space>
          </div>
        </div>
      )
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (record: AIModel) => (
        <div style={{ minWidth: 120 }}>
          <div style={{ fontSize: '12px', marginBottom: 4 }}>
            Accuracy: {(record.accuracy * 100).toFixed(1)}%
          </div>
          <Progress
            percent={record.accuracy * 100}
            size="small"
            strokeColor={record.accuracy > 0.8 ? '#52c41a' : record.accuracy > 0.7 ? '#faad14' : '#ff4d4f'}
            showInfo={false}
          />
          <div style={{ fontSize: '10px', color: '#999' }}>
            F1: {record.f1Score.toFixed(2)}
          </div>
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: AIModel['status']) => {
        const colors = { ready: 'green', training: 'blue', disabled: 'red' }
        return <Tag color={colors[status]}>{status}</Tag>
      }
    },
    {
      title: 'Enabled',
      key: 'enabled',
      render: (record: AIModel) => (
        <Switch
          size="small"
          checked={config.enabledModels.includes(record.id)}
          disabled={record.status !== 'ready'}
          onChange={(checked) => {
            const newEnabledModels = checked
              ? [...config.enabledModels, record.id]
              : config.enabledModels.filter(id => id !== record.id)
            setConfig({ ...config, enabledModels: newEnabledModels })
          }}
        />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: AIModel) => (
        <Space>
          <Tooltip title="Model details">
            <Button
              type="link"
              size="small"
              icon={<InfoCircleOutlined />}
              onClick={() => {
                setSelectedModel(record)
                setIsModelModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="Retrain model">
            <Button
              type="link"
              size="small"
              icon={<ExperimentOutlined />}
              onClick={() => retrainModel(record.id)}
              disabled={record.status === 'training'}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const detectionColumns = [
    {
      title: 'Pattern',
      dataIndex: 'patternType',
      key: 'patternType',
      render: (text: string, record: AIDetection) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.instrument} • {record.timeframe}
          </div>
        </div>
      )
    },
    {
      title: 'Confidence',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <div>
          <Progress
            percent={confidence * 100}
            size="small"
            strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f'}
          />
          <div style={{ fontSize: '12px', textAlign: 'center', marginTop: 2 }}>
            {(confidence * 100).toFixed(1)}%
          </div>
        </div>
      )
    },
    {
      title: 'Prediction',
      dataIndex: 'prediction',
      key: 'prediction',
      render: (prediction: string, record: AIDetection) => {
        const colors = { bullish: 'green', bearish: 'red', neutral: 'blue' }
        return (
          <div>
            <Tag color={colors[prediction as keyof typeof colors]}>{prediction}</Tag>
            {record.targetPrice && (
              <div style={{ fontSize: '10px', color: '#999' }}>
                Target: ${record.targetPrice.toFixed(2)}
              </div>
            )}
          </div>
        )
      }
    },
    {
      title: 'Model',
      dataIndex: 'modelId',
      key: 'modelId',
      render: (modelId: string) => {
        const model = models.find(m => m.id === modelId)
        return model ? (
          <Tooltip title={model.name}>
            <Tag>{model.type.toUpperCase()}</Tag>
          </Tooltip>
        ) : (
          <Tag>Unknown</Tag>
        )
      }
    },
    {
      title: 'Time Horizon',
      dataIndex: 'timeHorizon',
      key: 'timeHorizon',
      render: (days: number) => `${days} days`
    }
  ]

  const confidenceData = detections.map((det, index) => ({
    index: index + 1,
    confidence: det.confidence * 100,
    pattern: det.patternType
  }))

  return (
    <Card 
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <RobotOutlined />
            <span>AI Pattern Recognition</span>
            {isProcessing && <Spin size="small" />}
          </Space>
          <Space>
            {!isProcessing ? (
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={startAIProcessing}
                disabled={config.enabledModels.length === 0}
              >
                Start AI Analysis
              </Button>
            ) : (
              <Button 
                danger 
                icon={<PauseCircleOutlined />}
                onClick={stopAIProcessing}
              >
                Stop Processing
              </Button>
            )}
          </Space>
        </div>
      }
      size="small"
    >
      {isProcessing && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={
            <div>
              <div>AI models are analyzing market patterns...</div>
              <Row gutter={16} style={{ marginTop: 8 }}>
                <Col span={6}>
                  <Statistic 
                    title="Patterns Processed" 
                    value={processingStats.patternsProcessed} 
                    valueStyle={{ fontSize: 14 }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Detections Found" 
                    value={processingStats.detectionsFound} 
                    valueStyle={{ fontSize: 14 }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Avg Confidence" 
                    value={processingStats.avgConfidence * 100} 
                    precision={1}
                    suffix="%" 
                    valueStyle={{ fontSize: 14 }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic 
                    title="Speed" 
                    value={processingStats.processingSpeed} 
                    suffix="patterns/sec" 
                    valueStyle={{ fontSize: 14 }}
                  />
                </Col>
              </Row>
            </div>
          }
        />
      )}

      <Tabs
        items={[
          {
            key: 'models',
            label: 'AI Models',
            children: (
              <div>
                <div style={{ marginBottom: 16 }}>
                  <Alert
                    type="info"
                    showIcon
                    message="AI Model Status"
                    description="Advanced machine learning models trained on thousands of historical patterns. Enable multiple models for ensemble predictions."
                  />
                </div>

                <Table
                  columns={modelColumns}
                  dataSource={models}
                  rowKey="id"
                  size="small"
                  pagination={false}
                />
              </div>
            )
          },
          {
            key: 'detections',
            label: `Detections (${detections.length})`,
            children: (
              <div>
                {detections.length > 0 && (
                  <Card title="Confidence Distribution" size="small" style={{ marginBottom: 16 }}>
                    <Line
                      data={confidenceData}
                      xField="index"
                      yField="confidence"
                      height={200}
                      point={{ size: 4, shape: 'circle' }}
                      tooltip={{
                        formatter: (datum: any) => ({
                          name: 'Pattern',
                          value: `${datum.pattern}: ${datum.confidence.toFixed(1)}%`
                        })
                      }}
                    />
                  </Card>
                )}

                <Table
                  columns={detectionColumns}
                  dataSource={detections}
                  rowKey="id"
                  size="small"
                  pagination={{
                    pageSize: 10,
                    showTotal: (total) => `${total} AI detections`
                  }}
                />
              </div>
            )
          },
          {
            key: 'configuration',
            label: 'Configuration',
            children: (
              <div style={{ display: 'grid', gap: 16 }}>
                <Card title="Processing Settings" size="small">
                  <div style={{ display: 'grid', gap: 12 }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: 4 }}>
                        Confidence Threshold:
                      </label>
                      <Slider
                        min={0.1}
                        max={0.99}
                        step={0.01}
                        value={config.confidenceThreshold}
                        onChange={(value) => setConfig({ ...config, confidenceThreshold: value })}
                        marks={{
                          0.1: '10%',
                          0.5: '50%',
                          0.7: '70%',
                          0.9: '90%'
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: 4 }}>
                        Ensemble Voting Strategy:
                      </label>
                      <Select
                        value={config.ensembleVoting}
                        onChange={(value) => setConfig({ ...config, ensembleVoting: value })}
                        style={{ width: '100%' }}
                      >
                        <Option value="majority">Majority Voting</Option>
                        <Option value="weighted">Weighted Average</Option>
                        <Option value="unanimous">Unanimous Decision</Option>
                      </Select>
                    </div>
                  </div>
                </Card>

                <Card title="Feature Engineering" size="small">
                  <div style={{ display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
                    <div>
                      <Switch
                        checked={config.featureEngineering.technicalIndicators}
                        onChange={(checked) => 
                          setConfig({
                            ...config,
                            featureEngineering: { ...config.featureEngineering, technicalIndicators: checked }
                          })
                        }
                      />
                      <span style={{ marginLeft: 8 }}>Technical Indicators</span>
                    </div>

                    <div>
                      <Switch
                        checked={config.featureEngineering.priceAction}
                        onChange={(checked) => 
                          setConfig({
                            ...config,
                            featureEngineering: { ...config.featureEngineering, priceAction: checked }
                          })
                        }
                      />
                      <span style={{ marginLeft: 8 }}>Price Action</span>
                    </div>

                    <div>
                      <Switch
                        checked={config.featureEngineering.volumeProfile}
                        onChange={(checked) => 
                          setConfig({
                            ...config,
                            featureEngineering: { ...config.featureEngineering, volumeProfile: checked }
                          })
                        }
                      />
                      <span style={{ marginLeft: 8 }}>Volume Profile</span>
                    </div>

                    <div>
                      <Switch
                        checked={config.featureEngineering.multiTimeframe}
                        onChange={(checked) => 
                          setConfig({
                            ...config,
                            featureEngineering: { ...config.featureEngineering, multiTimeframe: checked }
                          })
                        }
                      />
                      <span style={{ marginLeft: 8 }}>Multi-Timeframe</span>
                    </div>
                  </div>
                </Card>

                <Card title="Advanced Options" size="small">
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div>
                      <Switch
                        checked={config.realTimeProcessing}
                        onChange={(checked) => setConfig({ ...config, realTimeProcessing: checked })}
                      />
                      <span style={{ marginLeft: 8 }}>Real-time Processing</span>
                    </div>

                    <div>
                      <Switch
                        checked={config.adaptiveLearning}
                        onChange={(checked) => setConfig({ ...config, adaptiveLearning: checked })}
                      />
                      <span style={{ marginLeft: 8 }}>Adaptive Learning</span>
                    </div>
                  </div>
                </Card>
              </div>
            )
          }
        ]}
      />

      {/* Model Details Modal */}
      <Modal
        title={selectedModel ? `Model: ${selectedModel.name}` : 'Model Details'}
        open={isModelModalVisible}
        onCancel={() => setIsModelModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsModelModalVisible(false)}>
            Close
          </Button>
        ]}
        width={600}
      >
        {selectedModel && (
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <strong>Description:</strong>
              <p>{selectedModel.description}</p>
            </div>

            <Row gutter={16}>
              <Col span={12}>
                <Card size="small" title="Performance Metrics">
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div>Accuracy: <strong>{(selectedModel.accuracy * 100).toFixed(1)}%</strong></div>
                    <div>Precision: <strong>{(selectedModel.precision * 100).toFixed(1)}%</strong></div>
                    <div>Recall: <strong>{(selectedModel.recall * 100).toFixed(1)}%</strong></div>
                    <div>F1 Score: <strong>{selectedModel.f1Score.toFixed(3)}</strong></div>
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="Training Information">
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div>Training Data: <strong>{selectedModel.trainingData.toLocaleString()} samples</strong></div>
                    <div>Version: <strong>v{selectedModel.version}</strong></div>
                    <div>Last Trained: <strong>{new Date(selectedModel.lastTrained).toLocaleDateString()}</strong></div>
                    <div>Status: <Tag color={selectedModel.status === 'ready' ? 'green' : 'blue'}>{selectedModel.status}</Tag></div>
                  </div>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </Card>
  )
}

export default AIPatternRecognition