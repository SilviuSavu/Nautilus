"""
Advanced AI/ML API Routes for Phase 6
FastAPI routes for reinforcement learning, LLM integration, computer vision, and model lifecycle
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import asyncio
import numpy as np
import pandas as pd
import torch
import json
import io
from PIL import Image

# Import our advanced AI modules
from .rl_agents import (
    RLAgentManager, DQNAgent, PPOAgent, A3CAgent, 
    MarketState, TradingAction, create_sample_trading_data
)
from .llm_integration import (
    SentimentAnalyzer, LLMMarketAnalyzer, BERTSentimentAnalyzer, 
    GPTMarketAnalyzer, MarketIntelligence, SentimentAnalysis
)
from .computer_vision import (
    CVDataProcessor, ChartPatternDetector, SatelliteImageryAnalyzer,
    ChartPattern, SatelliteInsight, CVAnalysisResult
)
from .neural_networks import (
    MultiModalProcessor, AttentionTradingNet, TransformerPredictor,
    MarketDataSequence, NeuralPrediction
)
from .portfolio_optimizer import (
    PortfolioOptimizerManager, DeepRLPortfolioOptimizer, QuantumInspiredOptimizer,
    PortfolioState, OptimizationResult
)
from .model_lifecycle import (
    AIModelManager, ModelMetadata, ModelPerformance, RetrainingJob
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/advanced-ai", tags=["Advanced AI/ML"])

# Global instances (would be properly initialized in main app)
rl_manager: Optional[RLAgentManager] = None
sentiment_analyzer: Optional[SentimentAnalyzer] = None
cv_processor: Optional[CVDataProcessor] = None
neural_processor: Optional[MultiModalProcessor] = None
portfolio_manager: Optional[PortfolioOptimizerManager] = None
model_manager: Optional[AIModelManager] = None


# Pydantic models for API
class RLTrainingRequest(BaseModel):
    agent_type: str = Field(..., description="Type of RL agent: DQN, PPO, or A3C")
    symbols: List[str] = Field(..., description="Trading symbols")
    training_period: str = Field(default="1y", description="Training data period")
    timesteps: int = Field(default=100000, description="Number of training timesteps")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")


class RLPredictionRequest(BaseModel):
    agent_name: str = Field(..., description="Name of trained agent")
    market_state: Dict[str, Any] = Field(..., description="Current market state data")


class SentimentAnalysisRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to analyze")
    symbols: Optional[List[str]] = Field(None, description="Symbols for market analysis")
    use_news: bool = Field(default=True, description="Include news analysis")
    use_bert: bool = Field(default=True, description="Use BERT model")
    use_gpt: bool = Field(default=False, description="Use GPT model (requires API key)")


class CVAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="Symbols for chart pattern analysis")
    locations: Optional[List[List[float]]] = Field(None, description="GPS coordinates [lat, lon]")
    timeframe: str = Field(default="1y", description="Analysis timeframe")
    include_satellite: bool = Field(default=True, description="Include satellite imagery analysis")


class NeuralNetworkRequest(BaseModel):
    symbols: List[str] = Field(..., description="Trading symbols")
    sequence_length: int = Field(default=60, description="Input sequence length")
    prediction_horizon: int = Field(default=1, description="Prediction horizon")
    include_training: bool = Field(default=False, description="Include model training")


class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., description="Portfolio symbols")
    current_weights: Optional[List[float]] = Field(None, description="Current portfolio weights")
    risk_tolerance: float = Field(default=0.5, description="Risk tolerance (0-1)")
    optimization_method: str = Field(default="ensemble", description="Optimization method")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Portfolio constraints")


class ModelDeploymentRequest(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")


class ModelMonitoringRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    predictions: List[float] = Field(..., description="Model predictions")
    actuals: Optional[List[float]] = Field(None, description="Actual values")
    features: Optional[List[List[float]]] = Field(None, description="Input features")


# Initialize components (would be called during app startup)
async def initialize_advanced_ai():
    """Initialize all advanced AI components"""
    global rl_manager, sentiment_analyzer, cv_processor, neural_processor, portfolio_manager, model_manager
    
    try:
        # Configuration
        config = {
            'rl_config': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'sentiment_config': {
                'bert_config': {'model_name': 'ProsusAI/finbert'},
                'use_bert': True,
                'use_gpt': False
            },
            'cv_config': {
                'chart_config': {},
                'satellite_config': {}
            },
            'neural_config': {
                'attention_config': {
                    'sequence_length': 60,
                    'hidden_dim': 256,
                    'num_heads': 8
                }
            },
            'portfolio_config': {
                'use_deep_rl': True,
                'use_quantum_inspired': True,
                'rl_config': {'num_assets': 10}
            },
            'model_lifecycle_config': {
                'registry_config': {'storage_path': '/app/models'},
                'monitor_config': {'drift_threshold': 0.1}
            }
        }
        
        # Initialize components
        rl_manager = RLAgentManager()
        sentiment_analyzer = SentimentAnalyzer(config['sentiment_config'])
        cv_processor = CVDataProcessor(config['cv_config'])
        neural_processor = MultiModalProcessor(config['neural_config'])
        portfolio_manager = PortfolioOptimizerManager(config['portfolio_config'])
        model_manager = AIModelManager(config['model_lifecycle_config'])
        
        logger.info("Advanced AI components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing advanced AI components: {e}")
        raise


# Reinforcement Learning Routes
@router.post("/rl/train", response_model=Dict[str, Any])
async def train_rl_agent(request: RLTrainingRequest, background_tasks: BackgroundTasks):
    """Train a reinforcement learning agent for trading"""
    try:
        if not rl_manager:
            await initialize_advanced_ai()
        
        # Create sample data (in production, would fetch real data)
        data = create_sample_trading_data(1000)
        
        # Create agent based on type
        agent_configs = {
            'DQN': {'learning_rate': 0.0001, 'buffer_size': 50000},
            'PPO': {'learning_rate': 0.0003, 'n_steps': 2048},
            'A3C': {'learning_rate': 0.0007, 'n_steps': 5}
        }
        
        config = {**agent_configs.get(request.agent_type, {}), **request.config}
        
        if request.agent_type == 'DQN':
            agent = DQNAgent(config)
        elif request.agent_type == 'PPO':
            agent = PPOAgent(config)
        elif request.agent_type == 'A3C':
            agent = A3CAgent(config)
        else:
            raise HTTPException(status_code=400, detail="Invalid agent type")
        
        # Add agent to manager
        rl_manager.add_agent(agent)
        
        # Start training in background
        background_tasks.add_task(
            train_agent_background, agent, data, request.timesteps
        )
        
        return {
            "status": "training_started",
            "agent_name": agent.name,
            "agent_type": request.agent_type,
            "timesteps": request.timesteps,
            "message": "Training started in background"
        }
        
    except Exception as e:
        logger.error(f"Error training RL agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def train_agent_background(agent, data, timesteps):
    """Background task for training RL agent"""
    try:
        from .rl_agents import TradingEnvironment
        env = TradingEnvironment(data)
        await agent.train(env, timesteps)
        logger.info(f"Training completed for agent {agent.name}")
    except Exception as e:
        logger.error(f"Error in background training: {e}")


@router.post("/rl/predict", response_model=Dict[str, Any])
async def predict_with_rl(request: RLPredictionRequest):
    """Make trading predictions using trained RL agent"""
    try:
        if not rl_manager or request.agent_name not in rl_manager.agents:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Convert request to MarketState
        state = MarketState(
            prices=np.array(request.market_state.get('prices', [])),
            volumes=np.array(request.market_state.get('volumes', [])),
            indicators=np.array(request.market_state.get('indicators', [])),
            portfolio=request.market_state.get('portfolio', {}),
            timestamp=datetime.now(),
            market_regime=request.market_state.get('market_regime', 'normal'),
            volatility=request.market_state.get('volatility', 0.02)
        )
        
        # Get prediction
        agent = rl_manager.agents[request.agent_name]
        prediction = agent.predict(state)
        
        return {
            "agent_name": request.agent_name,
            "prediction": {
                "action_type": prediction.action_type,
                "quantity": prediction.quantity,
                "price": prediction.price,
                "confidence": prediction.confidence,
                "metadata": prediction.metadata
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in RL prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl/agents", response_model=List[Dict[str, Any]])
async def list_rl_agents():
    """List all trained RL agents"""
    try:
        if not rl_manager:
            return []
        
        agents_info = []
        for name, agent in rl_manager.agents.items():
            performance_history = rl_manager.performance_history.get(name, [])
            agents_info.append({
                "name": name,
                "type": agent.__class__.__name__,
                "performance_records": len(performance_history),
                "latest_performance": performance_history[-1] if performance_history else None
            })
        
        return agents_info
        
    except Exception as e:
        logger.error(f"Error listing RL agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sentiment Analysis and LLM Routes
@router.post("/sentiment/analyze", response_model=Dict[str, Any])
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze market sentiment using LLMs"""
    try:
        if not sentiment_analyzer:
            await initialize_advanced_ai()
        
        if request.text:
            # Analyze specific text
            if request.use_bert:
                bert_analyzer = BERTSentimentAnalyzer({})
                result = await bert_analyzer.analyze_text(request.text)
                
                return {
                    "text_analysis": {
                        "overall_score": result.overall_score,
                        "confidence": result.confidence,
                        "emotion_scores": result.emotion_scores,
                        "key_phrases": result.key_phrases,
                        "market_impact": result.market_impact,
                        "entity_sentiment": result.entity_sentiment
                    },
                    "timestamp": datetime.now().isoformat()
                }
        
        elif request.symbols:
            # Analyze market sentiment for symbols
            intelligence = await sentiment_analyzer.analyze_market_sentiment(request.symbols)
            
            return {
                "market_analysis": {
                    "overall_sentiment": intelligence.overall_sentiment,
                    "confidence_level": intelligence.confidence_level,
                    "trend_analysis": intelligence.trend_analysis,
                    "sector_analysis": intelligence.sector_analysis,
                    "trading_signals": intelligence.trading_signals,
                    "risk_assessment": intelligence.risk_assessment,
                    "narrative_summary": intelligence.narrative_summary,
                    "data_sources": intelligence.data_sources
                },
                "timestamp": intelligence.timestamp.isoformat()
            }
        
        else:
            raise HTTPException(status_code=400, detail="Either text or symbols must be provided")
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/market-intelligence", response_model=Dict[str, Any])
async def get_market_intelligence():
    """Get comprehensive market intelligence report"""
    try:
        if not sentiment_analyzer:
            await initialize_advanced_ai()
        
        intelligence = await sentiment_analyzer.analyze_market_sentiment()
        
        return {
            "intelligence": {
                "overall_sentiment": intelligence.overall_sentiment,
                "confidence_level": intelligence.confidence_level,
                "trend_analysis": intelligence.trend_analysis,
                "sector_analysis": intelligence.sector_analysis,
                "trading_signals": intelligence.trading_signals,
                "risk_assessment": intelligence.risk_assessment,
                "narrative_summary": intelligence.narrative_summary,
                "data_sources": intelligence.data_sources
            },
            "timestamp": intelligence.timestamp.isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting market intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Computer Vision Routes
@router.post("/cv/analyze", response_model=Dict[str, Any])
async def computer_vision_analysis(request: CVAnalysisRequest):
    """Perform computer vision analysis on charts and satellite imagery"""
    try:
        if not cv_processor:
            await initialize_advanced_ai()
        
        # Convert locations format
        locations = None
        if request.locations:
            locations = [(loc[0], loc[1]) for loc in request.locations]
        
        # Perform comprehensive analysis
        results = await cv_processor.analyze_comprehensive(request.symbols, locations)
        
        # Convert results to JSON-serializable format
        chart_patterns = []
        for pattern in results.chart_patterns:
            chart_patterns.append({
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "coordinates": pattern.coordinates,
                "timeframe": [pattern.timeframe[0].isoformat(), pattern.timeframe[1].isoformat()],
                "price_range": pattern.price_range,
                "breakout_target": pattern.breakout_target,
                "trading_signal": pattern.trading_signal,
                "risk_level": pattern.risk_level,
                "metadata": pattern.metadata
            })
        
        satellite_insights = []
        for insight in results.satellite_insights:
            satellite_insights.append({
                "location": insight.location,
                "timestamp": insight.timestamp.isoformat(),
                "insight_type": insight.insight_type,
                "confidence": insight.confidence,
                "quantitative_measure": insight.quantitative_measure,
                "economic_impact": insight.economic_impact,
                "related_companies": insight.related_companies,
                "metadata": insight.metadata
            })
        
        return {
            "analysis_results": {
                "chart_patterns": chart_patterns,
                "satellite_insights": satellite_insights,
                "alternative_data_signals": results.alternative_data_signals,
                "confidence_score": results.confidence_score,
                "trading_recommendations": results.trading_recommendations,
                "risk_assessment": results.risk_assessment
            },
            "timestamp": results.timestamp.isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in computer vision analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/upload-chart", response_model=Dict[str, Any])
async def upload_chart_for_analysis(file: UploadFile = File(...)):
    """Upload chart image for pattern analysis"""
    try:
        if not cv_processor:
            await initialize_advanced_ai()
        
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Create sample price data for demonstration
        sample_data = create_sample_trading_data(252)
        
        # Detect patterns
        patterns = await cv_processor.chart_detector.detect_patterns(sample_data, "UPLOADED")
        
        # Convert patterns to JSON format
        pattern_results = []
        for pattern in patterns:
            pattern_results.append({
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "trading_signal": pattern.trading_signal,
                "risk_level": pattern.risk_level,
                "metadata": pattern.metadata
            })
        
        return {
            "patterns_detected": pattern_results,
            "image_info": {
                "filename": file.filename,
                "size": image.size,
                "format": image.format
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing uploaded chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Neural Networks Routes
@router.post("/neural/predict", response_model=Dict[str, Any])
async def neural_network_prediction(request: NeuralNetworkRequest):
    """Make predictions using advanced neural networks"""
    try:
        if not neural_processor:
            await initialize_advanced_ai()
        
        # Create sample multi-modal data
        batch_size = 1
        sequence_length = request.sequence_length
        
        sample_sequence = MarketDataSequence(
            prices=torch.randn(batch_size, sequence_length, 5),
            technical=torch.randn(batch_size, sequence_length, 20),
            fundamental=torch.randn(batch_size, sequence_length, 10),
            sentiment=torch.randn(batch_size, sequence_length, 5),
            alternative=torch.randn(batch_size, sequence_length, 8),
            timestamps=[datetime.now()],
            symbols=request.symbols,
            metadata={}
        )
        
        # Process through neural networks
        prediction = await neural_processor.process_multimodal_data(sample_sequence)
        
        return {
            "neural_prediction": {
                "predicted_values": prediction.predicted_values.tolist(),
                "confidence_scores": prediction.confidence_scores.tolist(),
                "feature_importance": prediction.feature_importance,
                "explanation": prediction.explanation,
                "prediction_intervals": [
                    prediction.prediction_intervals[0].tolist() if prediction.prediction_intervals else None,
                    prediction.prediction_intervals[1].tolist() if prediction.prediction_intervals else None
                ] if prediction.prediction_intervals else None
            },
            "model_info": {
                "symbols": request.symbols,
                "sequence_length": request.sequence_length,
                "prediction_horizon": request.prediction_horizon
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in neural network prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Optimization Routes
@router.post("/portfolio/optimize", response_model=Dict[str, Any])
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio using advanced AI methods"""
    try:
        if not portfolio_manager:
            await initialize_advanced_ai()
        
        # Create sample portfolio state
        n_assets = len(request.symbols)
        current_weights = np.array(request.current_weights) if request.current_weights else np.ones(n_assets) / n_assets
        
        # Generate sample market data
        returns = np.random.randn(n_assets) * 0.001
        volatilities = np.random.uniform(0.01, 0.05, n_assets)
        correlation_matrix = np.eye(n_assets)
        
        state = PortfolioState(
            weights=current_weights,
            returns=returns,
            prices=np.ones(n_assets),
            volatilities=volatilities,
            correlations=correlation_matrix,
            market_features=np.array([0.001, 0.02, 0.5, 0.05, 1.0]),
            cash=0.0,
            timestamp=datetime.now()
        )
        
        # Run optimization
        if request.optimization_method == "ensemble":
            results = await portfolio_manager.optimize_portfolio_ensemble(state)
            best_result = portfolio_manager.get_best_optimizer_result(results)
            
            # Convert all results
            all_results = []
            for result in results:
                all_results.append({
                    "optimization_method": result.optimization_method,
                    "optimal_weights": result.optimal_weights.tolist(),
                    "expected_return": result.expected_return,
                    "expected_volatility": result.expected_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "var_95": result.var_95,
                    "confidence_score": result.confidence_score,
                    "rebalancing_frequency": result.rebalancing_frequency,
                    "constraints_satisfied": result.constraints_satisfied,
                    "metadata": result.metadata
                })
            
            return {
                "optimization_results": {
                    "best_result": {
                        "optimization_method": best_result.optimization_method,
                        "optimal_weights": dict(zip(request.symbols, best_result.optimal_weights.tolist())),
                        "expected_return": best_result.expected_return,
                        "expected_volatility": best_result.expected_volatility,
                        "sharpe_ratio": best_result.sharpe_ratio,
                        "max_drawdown": best_result.max_drawdown,
                        "var_95": best_result.var_95,
                        "confidence_score": best_result.confidence_score
                    },
                    "all_methods": all_results,
                    "ensemble_size": len(results)
                },
                "portfolio_info": {
                    "symbols": request.symbols,
                    "current_weights": dict(zip(request.symbols, current_weights.tolist())),
                    "risk_tolerance": request.risk_tolerance
                },
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            # Single method optimization
            if request.optimization_method == "deep_rl" and portfolio_manager.optimizers.get('deep_rl'):
                result = await portfolio_manager.optimizers['deep_rl'].optimize_portfolio(state)
            elif request.optimization_method == "quantum" and portfolio_manager.optimizers.get('quantum'):
                result = await portfolio_manager.optimizers['quantum'].optimize_portfolio(state)
            else:
                raise HTTPException(status_code=400, detail="Invalid optimization method")
            
            return {
                "optimization_result": {
                    "optimization_method": result.optimization_method,
                    "optimal_weights": dict(zip(request.symbols, result.optimal_weights.tolist())),
                    "expected_return": result.expected_return,
                    "expected_volatility": result.expected_volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "var_95": result.var_95,
                    "confidence_score": result.confidence_score,
                    "rebalancing_frequency": result.rebalancing_frequency,
                    "constraints_satisfied": result.constraints_satisfied,
                    "metadata": result.metadata
                },
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Lifecycle Management Routes
@router.post("/models/deploy", response_model=Dict[str, Any])
async def deploy_model(request: ModelDeploymentRequest):
    """Deploy a new AI model"""
    try:
        if not model_manager:
            await initialize_advanced_ai()
        
        # Create dummy model for demonstration
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        dummy_model = DummyModel()
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=f"{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=request.model_name,
            version=request.version,
            model_type=request.model_type,
            architecture="neural_network",
            training_dataset="demo_dataset",
            training_start_time=datetime.now() - timedelta(hours=1),
            training_end_time=datetime.now(),
            performance_metrics={"accuracy": 0.85, "mse": 0.05, "r2": 0.82},
            hyperparameters=request.config,
            feature_importance={},
            data_schema={"input": "float32", "output": "float32"},
            model_size_mb=1.0,
            inference_latency_ms=10.0
        )
        
        # Deploy model
        model_id = await model_manager.deploy_model(dummy_model, metadata)
        
        return {
            "deployment_result": {
                "model_id": model_id,
                "status": "deployed",
                "deployment_time": datetime.now().isoformat(),
                "model_info": {
                    "name": metadata.model_name,
                    "version": metadata.version,
                    "type": metadata.model_type,
                    "performance_metrics": metadata.performance_metrics
                }
            },
            "message": "Model deployed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/monitor", response_model=Dict[str, Any])
async def monitor_model(model_id: str, request: ModelMonitoringRequest):
    """Monitor model performance and detect drift"""
    try:
        if not model_manager:
            await initialize_advanced_ai()
        
        # Convert to numpy arrays
        predictions = np.array(request.predictions)
        actuals = np.array(request.actuals) if request.actuals else None
        features = np.array(request.features) if request.features else None
        
        # Monitor model
        performance = await model_manager.monitor_prediction(
            model_id, predictions, actuals, features
        )
        
        return {
            "monitoring_result": {
                "model_id": model_id,
                "timestamp": performance.timestamp.isoformat(),
                "metrics": performance.metrics,
                "data_quality_score": performance.data_quality_score,
                "prediction_distribution": performance.prediction_distribution,
                "drift_metrics": performance.drift_metrics,
                "recommendation": performance.recommendation,
                "business_impact": performance.business_impact
            },
            "status": "monitoring_complete"
        }
        
    except Exception as e:
        logger.error(f"Error monitoring model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/health", response_model=Dict[str, Any])
async def get_model_health(model_id: str, hours: int = 24):
    """Get model health report"""
    try:
        if not model_manager:
            await initialize_advanced_ai()
        
        time_window = timedelta(hours=hours)
        health = await model_manager.model_monitor.get_model_health(model_id, time_window)
        
        return {
            "model_id": model_id,
            "health_report": health,
            "time_window_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/dashboard", response_model=Dict[str, Any])
async def get_model_dashboard(model_id: str):
    """Get comprehensive model dashboard"""
    try:
        if not model_manager:
            await initialize_advanced_ai()
        
        dashboard = await model_manager.get_model_dashboard(model_id)
        
        return {
            "model_id": model_id,
            "dashboard_data": dashboard,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=Dict[str, Any])
async def list_models(status: Optional[str] = None, model_type: Optional[str] = None):
    """List all models in registry"""
    try:
        if not model_manager:
            await initialize_advanced_ai()
        
        models = await model_manager.model_registry.list_models(status, model_type)
        
        model_list = []
        for model in models:
            model_list.append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "model_type": model.model_type,
                "deployment_status": model.deployment_status,
                "training_end_time": model.training_end_time.isoformat(),
                "performance_metrics": model.performance_metrics,
                "deployment_timestamp": model.deployment_timestamp.isoformat() if model.deployment_timestamp else None
            })
        
        return {
            "models": model_list,
            "count": len(model_list),
            "filters": {"status": status, "model_type": model_type},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health and Status Routes
@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Advanced AI system health check"""
    try:
        component_status = {
            "rl_manager": rl_manager is not None,
            "sentiment_analyzer": sentiment_analyzer is not None,
            "cv_processor": cv_processor is not None,
            "neural_processor": neural_processor is not None,
            "portfolio_manager": portfolio_manager is not None,
            "model_manager": model_manager is not None
        }
        
        all_healthy = all(component_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": component_status,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_capabilities():
    """Get advanced AI system capabilities"""
    return {
        "reinforcement_learning": {
            "algorithms": ["DQN", "PPO", "A3C"],
            "features": ["Trading strategy optimization", "Portfolio management", "Risk-adjusted decisions"]
        },
        "llm_integration": {
            "models": ["BERT (FinBERT)", "GPT (optional)", "Custom sentiment models"],
            "features": ["Market sentiment analysis", "News processing", "Economic impact assessment"]
        },
        "computer_vision": {
            "capabilities": ["Chart pattern recognition", "Satellite imagery analysis", "Alternative data extraction"],
            "patterns": ["Head and shoulders", "Triangles", "Flags", "Cups and handles", "Double tops/bottoms"]
        },
        "neural_networks": {
            "architectures": ["Attention mechanisms", "Transformers", "Multi-modal fusion"],
            "features": ["Time series prediction", "Multi-modal data processing", "Uncertainty quantification"]
        },
        "portfolio_optimization": {
            "methods": ["Deep reinforcement learning", "Quantum-inspired algorithms", "Traditional methods"],
            "features": ["Risk-adjusted returns", "Dynamic rebalancing", "Multi-objective optimization"]
        },
        "model_lifecycle": {
            "features": ["Model versioning", "Performance monitoring", "Drift detection", "Automated retraining"],
            "storage": ["Local registry", "MLflow integration", "S3 support"]
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "frameworks": ["PyTorch", "Transformers", "Stable-Baselines3", "OpenCV", "scikit-learn"]
        }
    }


# Export router for main app integration
__all__ = ["router", "initialize_advanced_ai"]