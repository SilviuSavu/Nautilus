-- ML Framework Database Schema for Nautilus Trading Platform
-- This schema supports all ML components including regime detection,
-- feature engineering, model lifecycle, risk prediction, and inference

-- Create ML schema
CREATE SCHEMA IF NOT EXISTS ml;

-- Market Regime Detection Tables
CREATE TABLE IF NOT EXISTS ml.regime_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probabilities JSONB NOT NULL,
    features_used JSONB NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_regime_predictions_timestamp ON ml.regime_predictions(timestamp DESC);
CREATE INDEX idx_regime_predictions_regime ON ml.regime_predictions(regime);
CREATE INDEX idx_regime_predictions_confidence ON ml.regime_predictions(confidence DESC);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('ml.regime_predictions', 'timestamp', if_not_exists => TRUE);

-- Feature Engineering Tables
CREATE TABLE IF NOT EXISTS ml.feature_batches (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_group VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    computation_time_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_batches_symbol_timestamp ON ml.feature_batches(symbol, timestamp DESC);
CREATE INDEX idx_feature_batches_feature_group ON ml.feature_batches(feature_group);
CREATE INDEX idx_feature_batches_timestamp ON ml.feature_batches(timestamp DESC);

-- Convert to hypertable
SELECT create_hypertable('ml.feature_batches', 'timestamp', if_not_exists => TRUE);

-- Correlation Analysis Tables
CREATE TABLE IF NOT EXISTS ml.correlation_analysis (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbols TEXT[] NOT NULL,
    correlation_matrix JSONB NOT NULL,
    eigenvalues FLOAT[] NOT NULL,
    clusters JSONB NOT NULL,
    lookback_days INTEGER NOT NULL,
    method VARCHAR(20) NOT NULL DEFAULT 'pearson',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_correlation_analysis_timestamp ON ml.correlation_analysis(timestamp DESC);
CREATE INDEX idx_correlation_analysis_symbols ON ml.correlation_analysis USING GIN(symbols);

-- Model Lifecycle Management Tables
CREATE TABLE IF NOT EXISTS ml.models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    file_path TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    config JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'inactive',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_trained_at TIMESTAMPTZ,
    last_validated_at TIMESTAMPTZ
);

CREATE INDEX idx_models_name ON ml.models(name);
CREATE INDEX idx_models_type ON ml.models(type);
CREATE INDEX idx_models_status ON ml.models(status);

-- Model Performance Tracking
CREATE TABLE IF NOT EXISTS ml.model_performance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES ml.models(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_type VARCHAR(20) NOT NULL, -- train, validation, test
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_performance_model_timestamp ON ml.model_performance(model_id, timestamp DESC);
CREATE INDEX idx_model_performance_metric ON ml.model_performance(metric_name);

-- Convert to hypertable
SELECT create_hypertable('ml.model_performance', 'timestamp', if_not_exists => TRUE);

-- Drift Detection Tables
CREATE TABLE IF NOT EXISTS ml.drift_detection (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES ml.models(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_type VARCHAR(20) NOT NULL, -- data, concept, label, performance
    drift_score FLOAT NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    threshold FLOAT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_drift_detection_model_timestamp ON ml.drift_detection(model_id, timestamp DESC);
CREATE INDEX idx_drift_detection_type ON ml.drift_detection(drift_type);

-- A/B Testing Framework
CREATE TABLE IF NOT EXISTS ml.ab_tests (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    champion_model_id INTEGER NOT NULL REFERENCES ml.models(id),
    challenger_model_id INTEGER NOT NULL REFERENCES ml.models(id),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    traffic_split FLOAT NOT NULL DEFAULT 0.5,
    start_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    success_metric VARCHAR(50) NOT NULL,
    statistical_significance FLOAT,
    winner_model_id INTEGER REFERENCES ml.models(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk Prediction Tables
CREATE TABLE IF NOT EXISTS ml.portfolio_optimizations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    portfolio_id VARCHAR(100) NOT NULL,
    method VARCHAR(50) NOT NULL,
    holdings JSONB NOT NULL,
    optimal_weights JSONB NOT NULL,
    expected_return FLOAT NOT NULL,
    expected_volatility FLOAT NOT NULL,
    sharpe_ratio FLOAT,
    risk_tolerance FLOAT,
    constraints JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_portfolio_optimizations_timestamp ON ml.portfolio_optimizations(timestamp DESC);
CREATE INDEX idx_portfolio_optimizations_portfolio ON ml.portfolio_optimizations(portfolio_id);

-- VaR Calculations
CREATE TABLE IF NOT EXISTS ml.var_calculations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    portfolio_id VARCHAR(100) NOT NULL,
    method VARCHAR(50) NOT NULL,
    confidence_level FLOAT NOT NULL,
    horizon_days INTEGER NOT NULL,
    var_value FLOAT NOT NULL,
    expected_shortfall FLOAT,
    scenario_analysis JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_var_calculations_timestamp ON ml.var_calculations(timestamp DESC);
CREATE INDEX idx_var_calculations_portfolio ON ml.var_calculations(portfolio_id);

-- Convert to hypertable
SELECT create_hypertable('ml.var_calculations', 'timestamp', if_not_exists => TRUE);

-- Stress Testing Results
CREATE TABLE IF NOT EXISTS ml.stress_tests (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    portfolio_id VARCHAR(100) NOT NULL,
    test_name VARCHAR(100) NOT NULL,
    scenarios JSONB NOT NULL,
    results JSONB NOT NULL,
    worst_case_loss FLOAT,
    risk_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_stress_tests_timestamp ON ml.stress_tests(timestamp DESC);
CREATE INDEX idx_stress_tests_portfolio ON ml.stress_tests(portfolio_id);

-- Real-time Inference Tables
CREATE TABLE IF NOT EXISTS ml.inference_requests (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    input_features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    inference_time_ms INTEGER NOT NULL,
    cache_hit BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inference_requests_timestamp ON ml.inference_requests(timestamp DESC);
CREATE INDEX idx_inference_requests_model ON ml.inference_requests(model_name);
CREATE INDEX idx_inference_requests_request_id ON ml.inference_requests(request_id);

-- Convert to hypertable
SELECT create_hypertable('ml.inference_requests', 'timestamp', if_not_exists => TRUE);

-- Model Server Metrics
CREATE TABLE IF NOT EXISTS ml.model_server_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    requests_per_second FLOAT NOT NULL,
    average_latency_ms FLOAT NOT NULL,
    error_rate FLOAT NOT NULL,
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    cache_hit_rate FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_server_metrics_timestamp ON ml.model_server_metrics(timestamp DESC);
CREATE INDEX idx_model_server_metrics_model ON ml.model_server_metrics(model_name);

-- Convert to hypertable
SELECT create_hypertable('ml.model_server_metrics', 'timestamp', if_not_exists => TRUE);

-- Feature Importance Tracking
CREATE TABLE IF NOT EXISTS ml.feature_importance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES ml.models(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_name VARCHAR(100) NOT NULL,
    importance_score FLOAT NOT NULL,
    importance_method VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_feature_importance_model_timestamp ON ml.feature_importance(model_id, timestamp DESC);
CREATE INDEX idx_feature_importance_feature ON ml.feature_importance(feature_name);

-- ML Alerts and Notifications
CREATE TABLE IF NOT EXISTS ml.alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    component VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, acknowledged, resolved
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alerts_timestamp ON ml.alerts(timestamp DESC);
CREATE INDEX idx_alerts_status ON ml.alerts(status);
CREATE INDEX idx_alerts_severity ON ml.alerts(severity);

-- System Health Monitoring
CREATE TABLE IF NOT EXISTS ml.system_health (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- healthy, degraded, unhealthy
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_system_health_timestamp ON ml.system_health(timestamp DESC);
CREATE INDEX idx_system_health_component ON ml.system_health(component);

-- Convert to hypertable
SELECT create_hypertable('ml.system_health', 'timestamp', if_not_exists => TRUE);

-- Create retention policies for time-series data
-- Keep detailed data for 30 days, compressed data for 1 year
SELECT add_retention_policy('ml.regime_predictions', INTERVAL '1 year');
SELECT add_retention_policy('ml.feature_batches', INTERVAL '1 year');
SELECT add_retention_policy('ml.model_performance', INTERVAL '2 years');
SELECT add_retention_policy('ml.inference_requests', INTERVAL '6 months');
SELECT add_retention_policy('ml.model_server_metrics', INTERVAL '1 year');
SELECT add_retention_policy('ml.var_calculations', INTERVAL '2 years');
SELECT add_retention_policy('ml.system_health', INTERVAL '6 months');

-- Create compression policies
SELECT add_compression_policy('ml.regime_predictions', INTERVAL '7 days');
SELECT add_compression_policy('ml.feature_batches', INTERVAL '7 days');
SELECT add_compression_policy('ml.model_performance', INTERVAL '30 days');
SELECT add_compression_policy('ml.inference_requests', INTERVAL '1 day');
SELECT add_compression_policy('ml.model_server_metrics', INTERVAL '7 days');
SELECT add_compression_policy('ml.var_calculations', INTERVAL '30 days');
SELECT add_compression_policy('ml.system_health', INTERVAL '7 days');

-- Create materialized views for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.daily_regime_summary AS
SELECT 
    DATE_TRUNC('day', timestamp) as day,
    regime,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM ml.regime_predictions
WHERE timestamp >= NOW() - INTERVAL '1 year'
GROUP BY DATE_TRUNC('day', timestamp), regime
ORDER BY day DESC, regime;

CREATE MATERIALIZED VIEW IF NOT EXISTS ml.model_performance_summary AS
SELECT 
    m.name,
    m.type,
    m.version,
    AVG(CASE WHEN mp.metric_name = 'accuracy' THEN mp.metric_value END) as avg_accuracy,
    AVG(CASE WHEN mp.metric_name = 'precision' THEN mp.metric_value END) as avg_precision,
    AVG(CASE WHEN mp.metric_name = 'recall' THEN mp.metric_value END) as avg_recall,
    AVG(CASE WHEN mp.metric_name = 'f1_score' THEN mp.metric_value END) as avg_f1_score,
    MAX(mp.timestamp) as last_evaluated
FROM ml.models m
LEFT JOIN ml.model_performance mp ON m.id = mp.model_id
WHERE mp.timestamp >= NOW() - INTERVAL '30 days'
GROUP BY m.id, m.name, m.type, m.version;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION ml.get_latest_regime() 
RETURNS TABLE(regime VARCHAR, confidence FLOAT, probabilities JSONB, timestamp TIMESTAMPTZ) 
AS $$
BEGIN
    RETURN QUERY
    SELECT rp.regime, rp.confidence, rp.probabilities, rp.timestamp
    FROM ml.regime_predictions rp
    ORDER BY rp.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION ml.get_model_health_score(model_name_param VARCHAR)
RETURNS FLOAT AS $$
DECLARE
    health_score FLOAT;
BEGIN
    SELECT 
        CASE 
            WHEN AVG(error_rate) < 0.01 AND AVG(average_latency_ms) < 100 THEN 1.0
            WHEN AVG(error_rate) < 0.05 AND AVG(average_latency_ms) < 500 THEN 0.8
            WHEN AVG(error_rate) < 0.1 AND AVG(average_latency_ms) < 1000 THEN 0.6
            ELSE 0.3
        END INTO health_score
    FROM ml.model_server_metrics
    WHERE model_name = model_name_param
    AND timestamp >= NOW() - INTERVAL '1 hour';
    
    RETURN COALESCE(health_score, 0.0);
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic data quality checks
CREATE OR REPLACE FUNCTION ml.validate_confidence_score()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.confidence < 0 OR NEW.confidence > 1 THEN
        RAISE EXCEPTION 'Confidence score must be between 0 and 1, got %', NEW.confidence;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_regime_confidence
    BEFORE INSERT OR UPDATE ON ml.regime_predictions
    FOR EACH ROW EXECUTE FUNCTION ml.validate_confidence_score();

-- Create indexes for improved query performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_regime_predictions_day 
ON ml.regime_predictions(DATE_TRUNC('day', timestamp), regime);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inference_requests_hour 
ON ml.inference_requests(DATE_TRUNC('hour', timestamp), model_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_metrics_recent 
ON ml.model_server_metrics(model_name, timestamp DESC) 
WHERE timestamp >= NOW() - INTERVAL '24 hours';

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA ml TO nautilus;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO nautilus;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml TO nautilus;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml TO nautilus;

-- Add comments for documentation
COMMENT ON SCHEMA ml IS 'Machine Learning framework schema for Nautilus trading platform';
COMMENT ON TABLE ml.regime_predictions IS 'Market regime predictions with confidence scores';
COMMENT ON TABLE ml.feature_batches IS 'Computed feature sets for ML models';
COMMENT ON TABLE ml.models IS 'ML model registry with versions and metadata';
COMMENT ON TABLE ml.inference_requests IS 'Real-time ML inference request logs';
COMMENT ON TABLE ml.model_server_metrics IS 'Performance metrics for ML model servers';

-- Create initial configuration entries
INSERT INTO ml.models (name, type, version, file_path, status, metadata) VALUES
('regime_detector_v1', 'classification', '1.0.0', '/app/models/regime_detector_v1.pkl', 'inactive', '{"description": "Market regime classification model", "features": 32}'),
('risk_predictor_v1', 'regression', '1.0.0', '/app/models/risk_predictor_v1.pkl', 'inactive', '{"description": "Portfolio risk prediction model", "features": 45}'),
('feature_importance_v1', 'feature_selection', '1.0.0', '/app/models/feature_importance_v1.pkl', 'inactive', '{"description": "Feature importance ranking model", "features": 380000}')
ON CONFLICT (name) DO NOTHING;

-- Refresh materialized views (should be done periodically via cron or background task)
REFRESH MATERIALIZED VIEW ml.daily_regime_summary;
REFRESH MATERIALIZED VIEW ml.model_performance_summary;