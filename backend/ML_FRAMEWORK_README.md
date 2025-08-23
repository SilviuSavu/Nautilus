# Advanced Machine Learning Framework for Nautilus Trading Platform

## 🚀 Overview

The Nautilus ML Framework is a comprehensive machine learning system designed for institutional-grade trading platforms. It provides advanced market regime detection, sophisticated feature engineering, automated model lifecycle management, ML-enhanced risk prediction, and real-time inference capabilities.

## 🏗️ Architecture

### Core Components

1. **Market Regime Detection** (`ml/market_regime.py`)
   - Ensemble-based regime classification
   - Real-time regime monitoring
   - Confidence scoring and probability distributions
   - 6 regime types: Bull, Bear, Sideways, Volatile, Crisis, Recovery

2. **Feature Engineering** (`ml/feature_engineering.py`)
   - 50+ technical indicators across multiple categories
   - Multi-asset correlation analysis
   - Real-time feature computation with caching
   - Alternative data integration support

3. **Model Lifecycle Management** (`ml/model_lifecycle.py`)
   - Automated drift detection using statistical tests
   - Model retraining pipelines
   - A/B testing framework for model comparison
   - Performance monitoring and validation

4. **Risk Prediction** (`ml/risk_prediction.py`)
   - ML-enhanced portfolio optimization
   - Multiple optimization methods (Mean-Variance, Black-Litterman, Risk Parity)
   - Monte Carlo stress testing
   - Advanced VaR calculations

5. **Real-time Inference Engine** (`ml/inference_engine.py`)
   - High-performance model serving (<100ms latency)
   - Prediction caching and load balancing
   - Comprehensive monitoring dashboard
   - Batch and streaming inference support

### Integration Layer

The `ml_integration.py` module connects the ML framework with existing Nautilus infrastructure:
- WebSocket streaming for real-time predictions
- Redis pub/sub for event-driven updates
- Integration with existing risk management system
- Background task management for continuous operation

## 📊 Key Features

### Market Regime Detection
```python
# Get current market regime
regime_state = await regime_detector.get_current_regime()
print(f"Current regime: {regime_state.regime.value}")
print(f"Confidence: {regime_state.confidence:.3f}")
print(f"Probabilities: {regime_state.probabilities}")
```

### Feature Engineering
```python
# Compute features for a symbol
features = await feature_engineer.compute_features('AAPL')
print(f"Total features: {len(features.features)}")

# Multi-asset correlation analysis
correlation_result = await feature_engineer.correlation_analyzer.analyze_cross_asset_correlation(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    lookback_days=30
)
```

### Risk Prediction
```python
# Portfolio optimization
portfolio = {'AAPL': 1000, 'GOOGL': 500, 'MSFT': 750}
optimization_result = await risk_predictor.optimize_portfolio(
    holdings=portfolio,
    method='mean_variance'
)

# VaR calculation with ML enhancement
var_result = await risk_predictor.calculate_var(
    portfolio=portfolio,
    confidence_level=0.95,
    method='monte_carlo'
)
```

### Real-time Inference
```python
# Make ML prediction
request = InferenceRequest(
    model_name='regime_detector',
    features={'volatility': 0.25, 'momentum': 0.15}
)
result = await inference_engine.predict(request)
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence}")
print(f"Latency: {result.inference_time_ms}ms")
```

## 🌐 API Endpoints

### Health and Status
- `GET /api/v1/ml/health` - ML system health check
- `GET /api/v1/ml/status` - Comprehensive system status
- `GET /api/v1/ml/monitoring/dashboard` - Monitoring dashboard data

### Market Regime Detection
- `GET /api/v1/ml/regime/current` - Current market regime
- `POST /api/v1/ml/regime/predict` - Predict regime from features
- `GET /api/v1/ml/regime/history` - Historical regime predictions

### Feature Engineering
- `POST /api/v1/ml/features/compute` - Compute features for symbol
- `GET /api/v1/ml/features/correlation` - Multi-asset correlation analysis

### Model Management
- `POST /api/v1/ml/models/retrain` - Trigger model retraining
- `GET /api/v1/ml/models/drift` - Check for model drift
- `GET /api/v1/ml/models/performance` - Model performance metrics

### Risk Prediction
- `POST /api/v1/ml/risk/portfolio/optimize` - Portfolio optimization
- `POST /api/v1/ml/risk/var/calculate` - VaR calculation
- `POST /api/v1/ml/risk/stress-test` - Stress testing

### Real-time Inference
- `POST /api/v1/ml/inference/predict` - Make ML prediction
- `GET /api/v1/ml/inference/models` - List available models
- `GET /api/v1/ml/inference/metrics` - Inference performance metrics

## 🗄️ Database Schema

The ML framework uses TimescaleDB for time-series optimization with the following key tables:

### Core Tables
- `ml.regime_predictions` - Market regime predictions with confidence scores
- `ml.feature_batches` - Computed feature sets for ML models
- `ml.models` - ML model registry with versions and metadata
- `ml.inference_requests` - Real-time ML inference request logs

### Performance Tables
- `ml.model_performance` - Model performance metrics over time
- `ml.model_server_metrics` - Model server performance data
- `ml.drift_detection` - Model drift detection results

### Risk Management Tables
- `ml.portfolio_optimizations` - Portfolio optimization results
- `ml.var_calculations` - VaR calculations with scenarios
- `ml.stress_tests` - Stress testing results

### System Monitoring Tables
- `ml.system_health` - ML system health monitoring
- `ml.alerts` - ML alerts and notifications

## 🚀 Getting Started

### 1. Database Setup
```sql
-- Run the ML database schema
\i ml_database_schema.sql
```

### 2. Environment Configuration
```python
from ml.config import MLConfig

config = MLConfig(
    database_url="postgresql://nautilus:nautilus123@localhost:5432/nautilus",
    redis_url="redis://localhost:6379",
    model_storage_path="/app/models"
)
```

### 3. Initialize ML Components
```python
from ml_integration import MLNautilusIntegrator

# Create integrator
ml_integrator = MLNautilusIntegrator(config)

# Initialize and start background tasks
await ml_integrator.initialize()
await ml_integrator.start_background_tasks()
```

### 4. Test Integration
```bash
# Run integration tests
python test_ml_integration.py
```

## ⚙️ Configuration Options

### Regime Detection Config
```python
regime_config = RegimeDetectionConfig(
    ensemble_size=5,
    confidence_threshold=0.7,
    update_frequency=60,  # seconds
    lookback_window=252   # trading days
)
```

### Feature Engineering Config
```python
feature_config = FeatureEngineeringConfig(
    cache_ttl=300,  # seconds
    max_correlation_features=100,
    alternative_data_enabled=True,
    feature_selection_threshold=0.05
)
```

### Inference Config
```python
inference_config = InferenceConfig(
    max_latency_ms=100,
    cache_size=1000,
    batch_size=32,
    enable_monitoring=True
)
```

## 📈 Performance Metrics

### Latency Targets
- **Inference**: <100ms for real-time predictions
- **Feature Computation**: <500ms for 50+ indicators
- **Regime Detection**: <1s for ensemble prediction
- **Risk Calculation**: <2s for portfolio VaR

### Throughput Capabilities
- **Predictions/second**: 1,000+
- **Feature Updates/minute**: 100+
- **Regime Updates**: Real-time (60s intervals)
- **Model Retraining**: Automated daily/weekly

### Accuracy Targets
- **Regime Classification**: >80% accuracy
- **Risk Prediction**: <5% VaR model error
- **Drift Detection**: 95% sensitivity
- **Feature Selection**: Top 20% by importance

## 🔧 Monitoring and Alerting

### System Health Monitoring
- Component health checks every 5 minutes
- Performance degradation detection
- Resource utilization tracking
- Error rate monitoring

### ML-specific Alerts
- Model drift detection (statistical significance)
- Performance degradation (accuracy drop >10%)
- Inference latency spikes (>200ms)
- Cache hit rate drops (<80%)

### Integration with Existing Systems
- WebSocket broadcasts for real-time updates
- Redis pub/sub for event notifications
- Integration with breach detector for risk alerts
- Prometheus metrics for monitoring

## 🛡️ Risk Management Integration

### ML-Enhanced Breach Detection
The framework integrates with the existing `AdvancedBreachDetector` to provide:
- Regime-aware threshold adjustments
- ML-based portfolio risk scoring
- Volatility spike prediction
- Dynamic risk limit recommendations

### Real-time Risk Monitoring
- Continuous VaR calculations
- Stress test scenario evaluation
- Portfolio optimization alerts
- Regime-based risk adjustments

## 🚦 Production Deployment

### Prerequisites
- PostgreSQL 13+ with TimescaleDB extension
- Redis 6+ for caching and pub/sub
- Python 3.11+ with required ML libraries
- Docker for containerized deployment

### Deployment Steps
1. Apply database schema (`ml_database_schema.sql`)
2. Configure environment variables
3. Start ML services with Docker Compose
4. Run integration tests
5. Monitor system health dashboard

### Scaling Considerations
- Horizontal scaling via Redis clustering
- Model server load balancing
- Feature computation caching
- Database partitioning for time-series data

## 🔍 Troubleshooting

### Common Issues
1. **High inference latency**: Check model server load and cache hit rates
2. **Drift detection false positives**: Adjust statistical thresholds
3. **Feature computation timeouts**: Optimize database queries and add caching
4. **Memory usage spikes**: Configure model server resource limits

### Debugging Tools
- Health check endpoints for component status
- Performance metrics dashboard
- Detailed logging with correlation IDs
- Integration test suite for validation

## 📚 API Documentation

Complete API documentation is available through FastAPI's automatic docs:
- Swagger UI: `http://localhost:8001/docs#/Machine%20Learning`
- ReDoc: `http://localhost:8001/redoc`

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Implement changes with tests
3. Run integration test suite
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Add docstrings for public APIs
- Include comprehensive error handling

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For technical support and questions:
- Create GitHub issues for bugs
- Use discussions for questions
- Check troubleshooting section
- Review integration test output

---

**Built for Nautilus Trading Platform - Enterprise-Grade Machine Learning at Scale** 🚀