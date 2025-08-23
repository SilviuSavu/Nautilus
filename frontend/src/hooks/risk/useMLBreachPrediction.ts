/**
 * useMLBreachPrediction Hook
 * Sprint 3: ML-Based Breach Prediction System
 * 
 * Advanced machine learning models for predicting risk limit breaches
 * with feature engineering, model selection, and real-time scoring.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketStream } from '../useWebSocketStream';

export interface PredictionFeature {
  name: string;
  value: number;
  importance: number;
  category: 'market' | 'portfolio' | 'technical' | 'external';
  description: string;
}

export interface PredictionModel {
  id: string;
  name: string;
  type: 'random_forest' | 'gradient_boosting' | 'neural_network' | 'svm' | 'ensemble';
  version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingDate: string;
  features: string[];
  hyperparameters: Record<string, any>;
  status: 'active' | 'training' | 'deprecated' | 'testing';
}

export interface BreachPrediction {
  id: string;
  limitId: string;
  portfolioId?: string;
  strategyId?: string;
  
  // Prediction details
  timestamp: string;
  predictionHorizon: number; // minutes
  breachProbability: number;
  confidence: number;
  riskScore: number;
  
  // Time estimates
  timeToBreach: number; // minutes
  timeToWarning: number; // minutes
  
  // Contributing factors
  topFeatures: PredictionFeature[];
  marketFactors: {
    volatility: number;
    correlation: number;
    liquidity: number;
    momentum: number;
  };
  
  // Model information
  modelUsed: string;
  modelConfidence: number;
  alternativeModels: {
    modelId: string;
    probability: number;
    confidence: number;
  }[];
  
  // Action recommendations
  recommendations: {
    action: string;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    impact: number;
    description: string;
  }[];
  
  // Historical context
  similarEvents: {
    date: string;
    actualBreach: boolean;
    timeToEvent: number;
    similarity: number;
  }[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  confusionMatrix: number[][];
  calibrationCurve: { predicted: number; actual: number }[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  category: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  description: string;
}

export interface ModelTrainingConfig {
  targetVariable: string;
  features: string[];
  lookbackPeriod: number; // days
  predictionHorizon: number; // minutes
  modelType: PredictionModel['type'];
  hyperparameters: Record<string, any>;
  validationSplit: number;
  crossValidationFolds: number;
}

export interface UseMLBreachPredictionOptions {
  portfolioId?: string;
  enableRealTime?: boolean;
  predictionInterval?: number;
  confidenceThreshold?: number;
  enableEnsemble?: boolean;
  enableExplainability?: boolean;
  maxPredictions?: number;
}

export interface UseMLBreachPredictionReturn {
  // Predictions
  predictions: BreachPrediction[];
  activePredictions: BreachPrediction[];
  
  // Models
  models: PredictionModel[];
  activeModel: PredictionModel | null;
  modelMetrics: ModelMetrics | null;
  featureImportance: FeatureImportance[];
  
  // Status
  isLoading: boolean;
  isPredicting: boolean;
  error: string | null;
  lastPrediction: Date | null;
  
  // Model management
  trainModel: (config: ModelTrainingConfig) => Promise<string>;
  deployModel: (modelId: string) => Promise<void>;
  retireModel: (modelId: string) => Promise<void>;
  evaluateModel: (modelId: string, testData?: any[]) => Promise<ModelMetrics>;
  
  // Predictions
  generatePredictions: (limitIds?: string[]) => Promise<BreachPrediction[]>;
  schedulePredictions: (intervalMinutes: number) => void;
  stopScheduledPredictions: () => void;
  
  // Feature engineering
  calculateFeatures: (limitId: string) => Promise<PredictionFeature[]>;
  updateFeatureImportance: () => Promise<void>;
  
  // Explainability
  explainPrediction: (predictionId: string) => Promise<{
    globalExplanation: { feature: string; contribution: number }[];
    localExplanation: { feature: string; contribution: number }[];
    counterfactuals: { feature: string; currentValue: number; requiredValue: number }[];
  }>;
  
  // Calibration
  calibrateModel: (modelId: string) => Promise<void>;
  getCalibrationCurve: (modelId: string) => Promise<{ predicted: number; actual: number }[]>;
  
  // Backtesting
  backtest: (modelId: string, startDate: string, endDate: string) => Promise<{
    predictions: number;
    accuracy: number;
    profitability: number;
    sharpeRatio: number;
    maxDrawdown: number;
  }>;
  
  // Monitoring
  monitorModelPerformance: () => Promise<{
    drift: boolean;
    degradation: boolean;
    recommendations: string[];
  }>;
  
  // Utilities
  exportPredictions: (format: 'json' | 'csv') => Promise<string>;
  refresh: () => Promise<void>;
  reset: () => void;
}

const DEFAULT_OPTIONS: Required<UseMLBreachPredictionOptions> = {
  portfolioId: '',
  enableRealTime: true,
  predictionInterval: 300000, // 5 minutes
  confidenceThreshold: 0.7,
  enableEnsemble: true,
  enableExplainability: false,
  maxPredictions: 1000
};

export function useMLBreachPrediction(
  options: UseMLBreachPredictionOptions = {}
): UseMLBreachPredictionReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [predictions, setPredictions] = useState<BreachPrediction[]>([]);
  const [models, setModels] = useState<PredictionModel[]>([]);
  const [activeModel, setActiveModel] = useState<PredictionModel | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastPrediction, setLastPrediction] = useState<Date | null>(null);
  
  // Refs
  const predictionIntervalRef = useRef<NodeJS.Timeout>();
  const featureCacheRef = useRef<Map<string, PredictionFeature[]>>(new Map());
  const isMountedRef = useRef(true);
  
  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
  
  // WebSocket stream for real-time prediction updates
  const {
    isActive: isRealTimeActive,
    latestMessage,
    startStream,
    stopStream,
    error: streamError
  } = useWebSocketStream({
    streamId: 'ml_predictions',
    messageType: 'breach_alert',
    bufferSize: 500,
    autoSubscribe: config.enableRealTime
  });
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (predictionIntervalRef.current) {
        clearInterval(predictionIntervalRef.current);
      }
    };
  }, []);
  
  // Process real-time prediction updates
  useEffect(() => {
    if (latestMessage && latestMessage.data && latestMessage.data.type === 'ml_prediction') {
      const predictionData = latestMessage.data;
      
      const newPrediction: BreachPrediction = {
        id: predictionData.prediction_id || generatePredictionId(),
        limitId: predictionData.limit_id || '',
        portfolioId: predictionData.portfolio_id,
        strategyId: predictionData.strategy_id,
        timestamp: predictionData.timestamp || new Date().toISOString(),
        predictionHorizon: predictionData.prediction_horizon || 60,
        breachProbability: predictionData.breach_probability || 0,
        confidence: predictionData.confidence || 0,
        riskScore: predictionData.risk_score || 0,
        timeToBreach: predictionData.time_to_breach || 0,
        timeToWarning: predictionData.time_to_warning || 0,
        topFeatures: predictionData.top_features || [],
        marketFactors: predictionData.market_factors || {
          volatility: 0,
          correlation: 0,
          liquidity: 0,
          momentum: 0
        },
        modelUsed: predictionData.model_used || '',
        modelConfidence: predictionData.model_confidence || 0,
        alternativeModels: predictionData.alternative_models || [],
        recommendations: predictionData.recommendations || [],
        similarEvents: predictionData.similar_events || []
      };
      
      setPredictions(prev => {
        const filtered = prev.filter(p => p.id !== newPrediction.id);
        return [newPrediction, ...filtered].slice(0, config.maxPredictions);
      });
      
      setLastPrediction(new Date());
    }
  }, [latestMessage, config.maxPredictions]);
  
  // Utility functions
  const generatePredictionId = useCallback(() => {
    return `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  const generateModelId = useCallback(() => {
    return `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);
  
  // Feature engineering
  const calculateFeatures = useCallback(async (limitId: string): Promise<PredictionFeature[]> => {
    // Check cache first
    const cached = featureCacheRef.current.get(limitId);
    if (cached && Date.now() - parseInt(cached[0]?.name.split('_')[1] || '0') < 60000) {
      return cached;
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/features/${limitId}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Feature calculation failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      const features: PredictionFeature[] = data.features || [];
      
      // Cache features
      featureCacheRef.current.set(limitId, features);
      
      return features;
    } catch (err) {
      console.error('Failed to calculate features:', err);
      return [];
    }
  }, [API_BASE_URL]);
  
  // Model training
  const trainModel = useCallback(async (config: ModelTrainingConfig): Promise<string> => {
    setIsLoading(true);
    
    try {
      const modelId = generateModelId();
      
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          config,
          portfolio_id: options.portfolioId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Model training failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Add new model to state
      const newModel: PredictionModel = {
        id: modelId,
        name: `${config.modelType}_${new Date().toISOString().split('T')[0]}`,
        type: config.modelType,
        version: '1.0.0',
        accuracy: result.metrics?.accuracy || 0,
        precision: result.metrics?.precision || 0,
        recall: result.metrics?.recall || 0,
        f1Score: result.metrics?.f1_score || 0,
        trainingDate: new Date().toISOString(),
        features: config.features,
        hyperparameters: config.hyperparameters,
        status: 'training'
      };
      
      setModels(prev => [...prev, newModel]);
      
      return modelId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model training failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [generateModelId, API_BASE_URL, options.portfolioId]);
  
  // Model deployment
  const deployModel = useCallback(async (modelId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Model deployment failed: ${response.statusText}`);
      }
      
      // Update model status
      setModels(prev => prev.map(model => ({
        ...model,
        status: model.id === modelId ? 'active' : (model.status === 'active' ? 'deprecated' : model.status)
      })));
      
      // Set as active model
      const model = models.find(m => m.id === modelId);
      if (model) {
        setActiveModel({ ...model, status: 'active' });
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model deployment failed');
      throw err;
    }
  }, [API_BASE_URL, models]);
  
  // Model retirement
  const retireModel = useCallback(async (modelId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/retire`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Model retirement failed: ${response.statusText}`);
      }
      
      setModels(prev => prev.map(model => 
        model.id === modelId ? { ...model, status: 'deprecated' } : model
      ));
      
      if (activeModel?.id === modelId) {
        setActiveModel(null);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model retirement failed');
      throw err;
    }
  }, [API_BASE_URL, activeModel]);
  
  // Model evaluation
  const evaluateModel = useCallback(async (modelId: string, testData?: any[]): Promise<ModelMetrics> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_data: testData })
      });
      
      if (!response.ok) {
        throw new Error(`Model evaluation failed: ${response.statusText}`);
      }
      
      const metrics = await response.json();
      
      const modelMetrics: ModelMetrics = {
        accuracy: metrics.accuracy || 0,
        precision: metrics.precision || 0,
        recall: metrics.recall || 0,
        f1Score: metrics.f1_score || 0,
        auc: metrics.auc || 0,
        falsePositiveRate: metrics.false_positive_rate || 0,
        falseNegativeRate: metrics.false_negative_rate || 0,
        confusionMatrix: metrics.confusion_matrix || [],
        calibrationCurve: metrics.calibration_curve || []
      };
      
      if (modelId === activeModel?.id) {
        setModelMetrics(modelMetrics);
      }
      
      return modelMetrics;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model evaluation failed');
      throw err;
    }
  }, [API_BASE_URL, activeModel]);
  
  // Generate predictions
  const generatePredictions = useCallback(async (limitIds?: string[]): Promise<BreachPrediction[]> => {
    if (!activeModel) {
      throw new Error('No active model available');
    }
    
    setIsPredicting(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/predictions/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: activeModel.id,
          limit_ids: limitIds,
          portfolio_id: config.portfolioId,
          prediction_horizon: 60, // minutes
          confidence_threshold: config.confidenceThreshold
        })
      });
      
      if (!response.ok) {
        throw new Error(`Prediction generation failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      const newPredictions: BreachPrediction[] = result.predictions || [];
      
      setPredictions(prev => {
        const combined = [...newPredictions, ...prev];
        return combined.slice(0, config.maxPredictions);
      });
      
      setLastPrediction(new Date());
      
      return newPredictions;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction generation failed');
      throw err;
    } finally {
      setIsPredicting(false);
    }
  }, [activeModel, API_BASE_URL, config.portfolioId, config.confidenceThreshold, config.maxPredictions]);
  
  // Scheduled predictions
  const schedulePredictions = useCallback((intervalMinutes: number) => {
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
    }
    
    predictionIntervalRef.current = setInterval(() => {
      if (isMountedRef.current && activeModel) {
        generatePredictions();
      }
    }, intervalMinutes * 60 * 1000);
  }, [activeModel, generatePredictions]);
  
  const stopScheduledPredictions = useCallback(() => {
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = undefined;
    }
  }, []);
  
  // Feature importance update
  const updateFeatureImportance = useCallback(async (): Promise<void> => {
    if (!activeModel) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${activeModel.id}/feature-importance`);
      
      if (!response.ok) {
        throw new Error(`Feature importance update failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setFeatureImportance(data.features || []);
    } catch (err) {
      console.error('Failed to update feature importance:', err);
    }
  }, [activeModel, API_BASE_URL]);
  
  // Explainability
  const explainPrediction = useCallback(async (predictionId: string) => {
    if (!config.enableExplainability) {
      throw new Error('Explainability is not enabled');
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/predictions/${predictionId}/explain`);
      
      if (!response.ok) {
        throw new Error(`Prediction explanation failed: ${response.statusText}`);
      }
      
      const explanation = await response.json();
      
      return {
        globalExplanation: explanation.global_explanation || [],
        localExplanation: explanation.local_explanation || [],
        counterfactuals: explanation.counterfactuals || []
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction explanation failed');
      throw err;
    }
  }, [config.enableExplainability, API_BASE_URL]);
  
  // Model calibration
  const calibrateModel = useCallback(async (modelId: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/calibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`Model calibration failed: ${response.statusText}`);
      }
      
      // Update model version
      setModels(prev => prev.map(model => 
        model.id === modelId 
          ? { ...model, version: `${model.version}_calibrated` }
          : model
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model calibration failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Get calibration curve
  const getCalibrationCurve = useCallback(async (modelId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/calibration-curve`);
      
      if (!response.ok) {
        throw new Error(`Calibration curve retrieval failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.calibration_curve || [];
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration curve retrieval failed');
      throw err;
    }
  }, [API_BASE_URL]);
  
  // Backtesting
  const backtest = useCallback(async (modelId: string, startDate: string, endDate: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${modelId}/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          portfolio_id: config.portfolioId
        })
      });
      
      if (!response.ok) {
        throw new Error(`Backtesting failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      return {
        predictions: result.predictions || 0,
        accuracy: result.accuracy || 0,
        profitability: result.profitability || 0,
        sharpeRatio: result.sharpe_ratio || 0,
        maxDrawdown: result.max_drawdown || 0
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtesting failed');
      throw err;
    }
  }, [API_BASE_URL, config.portfolioId]);
  
  // Monitor model performance
  const monitorModelPerformance = useCallback(async () => {
    if (!activeModel) {
      throw new Error('No active model to monitor');
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ml/models/${activeModel.id}/monitor`);
      
      if (!response.ok) {
        throw new Error(`Model monitoring failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      return {
        drift: result.drift || false,
        degradation: result.degradation || false,
        recommendations: result.recommendations || []
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model monitoring failed');
      throw err;
    }
  }, [activeModel, API_BASE_URL]);
  
  // Export predictions
  const exportPredictions = useCallback(async (format: 'json' | 'csv'): Promise<string> => {
    const data = predictions.slice(0, 100); // Export last 100 predictions
    
    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    } else {
      const csvRows = [
        ['Prediction ID', 'Limit ID', 'Timestamp', 'Breach Probability', 'Confidence', 'Time to Breach', 'Risk Score'],
        ...data.map(p => [
          p.id, p.limitId, p.timestamp, p.breachProbability, p.confidence, p.timeToBreach, p.riskScore
        ])
      ];
      return csvRows.map(row => row.join(',')).join('\n');
    }
  }, [predictions]);
  
  // Control functions
  const refresh = useCallback(async () => {
    setIsLoading(true);
    
    try {
      const [modelsResponse, predictionsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/ml/models${config.portfolioId ? `?portfolio_id=${config.portfolioId}` : ''}`),
        fetch(`${API_BASE_URL}/api/v1/ml/predictions${config.portfolioId ? `?portfolio_id=${config.portfolioId}` : ''}`)
      ]);
      
      if (modelsResponse.ok) {
        const modelsData = await modelsResponse.json();
        setModels(modelsData.models || []);
        
        const active = modelsData.models?.find((m: PredictionModel) => m.status === 'active');
        if (active) {
          setActiveModel(active);
        }
      }
      
      if (predictionsResponse.ok) {
        const predictionsData = await predictionsResponse.json();
        setPredictions(predictionsData.predictions || []);
      }
      
      // Update feature importance for active model
      if (activeModel) {
        await updateFeatureImportance();
      }
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE_URL, config.portfolioId, activeModel, updateFeatureImportance]);
  
  const reset = useCallback(() => {
    setPredictions([]);
    setModels([]);
    setActiveModel(null);
    setModelMetrics(null);
    setFeatureImportance([]);
    setError(null);
    setLastPrediction(null);
    featureCacheRef.current.clear();
    stopScheduledPredictions();
  }, [stopScheduledPredictions]);
  
  // Computed values
  const activePredictions = predictions.filter(p => {
    const age = Date.now() - new Date(p.timestamp).getTime();
    return age < p.predictionHorizon * 60 * 1000 && p.breachProbability >= config.confidenceThreshold;
  });
  
  // Initial data load
  useEffect(() => {
    refresh();
  }, [refresh]);
  
  // Auto-schedule predictions if real-time is enabled
  useEffect(() => {
    if (config.enableRealTime && activeModel && config.predictionInterval > 0) {
      schedulePredictions(config.predictionInterval / 60000);
    }
    
    return () => {
      stopScheduledPredictions();
    };
  }, [config.enableRealTime, config.predictionInterval, activeModel, schedulePredictions, stopScheduledPredictions]);
  
  return {
    // Predictions
    predictions,
    activePredictions,
    
    // Models
    models,
    activeModel,
    modelMetrics,
    featureImportance,
    
    // Status
    isLoading,
    isPredicting,
    error: error || streamError,
    lastPrediction,
    
    // Model management
    trainModel,
    deployModel,
    retireModel,
    evaluateModel,
    
    // Predictions
    generatePredictions,
    schedulePredictions,
    stopScheduledPredictions,
    
    // Feature engineering
    calculateFeatures,
    updateFeatureImportance,
    
    // Explainability
    explainPrediction,
    
    // Calibration
    calibrateModel,
    getCalibrationCurve,
    
    // Backtesting
    backtest,
    
    // Monitoring
    monitorModelPerformance,
    
    // Utilities
    exportPredictions,
    refresh,
    reset
  };
}

export default useMLBreachPrediction;