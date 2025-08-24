-- Sprint 3: Advanced Trading Infrastructure Schema Extensions
-- Additional tables required for Sprint 3 implementation
-- These complement the existing analytics_tables.sql

-- Performance metrics table for real-time tracking (Sprint 3 requirement)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(strategy_id, metric_name, timestamp)
);

-- Create indexes separately for performance_metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_strategy_timestamp ON performance_metrics(strategy_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_timestamp ON performance_metrics(metric_name, timestamp);

-- Risk monitoring events table (Sprint 3 requirement)
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT NOT NULL,
    portfolio_id VARCHAR(100),
    strategy_id VARCHAR(100),
    instrument_id VARCHAR(100),
    risk_value DECIMAL(20,8),
    threshold_value DECIMAL(20,8),
    breach_magnitude DECIMAL(20,8),
    event_data JSONB DEFAULT '{}',
    resolved_at TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    INDEX(event_type, timestamp),
    INDEX(severity, timestamp),
    INDEX(portfolio_id, timestamp),
    INDEX(strategy_id, timestamp)
);

-- WebSocket connection tracking for monitoring
CREATE TABLE IF NOT EXISTS websocket_connections (
    id SERIAL PRIMARY KEY,
    connection_id VARCHAR(100) NOT NULL UNIQUE,
    user_id VARCHAR(100),
    endpoint_type VARCHAR(50) NOT NULL,
    client_ip INET,
    user_agent TEXT,
    subscriptions JSONB DEFAULT '[]',
    message_count INTEGER DEFAULT 0,
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    connected_at TIMESTAMPTZ DEFAULT NOW(),
    disconnected_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'disconnected', 'stale')),
    INDEX(user_id, connected_at),
    INDEX(endpoint_type, status),
    INDEX(connected_at)
);

-- WebSocket message audit log
CREATE TABLE IF NOT EXISTS websocket_message_log (
    id BIGSERIAL PRIMARY KEY,
    connection_id VARCHAR(100) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    message_size INTEGER NOT NULL,
    processing_time_ms DECIMAL(10,3),
    error_message TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    INDEX(connection_id, timestamp),
    INDEX(message_type, timestamp),
    INDEX(direction, timestamp)
);

-- Strategy deployment history
CREATE TABLE IF NOT EXISTS strategy_deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(100) NOT NULL UNIQUE,
    strategy_id VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    environment VARCHAR(20) NOT NULL CHECK (environment IN ('development', 'staging', 'production')),
    deployment_type VARCHAR(20) NOT NULL CHECK (deployment_type IN ('deploy', 'rollback', 'emergency_stop')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'deploying', 'deployed', 'failed', 'rolled_back')),
    config_snapshot JSONB NOT NULL,
    deployed_by VARCHAR(100) NOT NULL,
    deployment_notes TEXT,
    health_check_url VARCHAR(500),
    performance_baseline JSONB,
    rollback_deployment_id VARCHAR(100),
    deployed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX(strategy_id, deployed_at),
    INDEX(environment, status),
    INDEX(deployed_by, created_at),
    INDEX(status, created_at)
);

-- Strategy deployment validation results
CREATE TABLE IF NOT EXISTS strategy_validation_results (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(100) NOT NULL,
    validation_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pass', 'fail', 'warning', 'skipped')),
    message TEXT,
    details JSONB DEFAULT '{}',
    duration_ms INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    INDEX(deployment_id, validation_type),
    INDEX(status, timestamp),
    FOREIGN KEY (deployment_id) REFERENCES strategy_deployments(deployment_id)
);

-- System monitoring metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    component VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    unit VARCHAR(20),
    threshold_warning DECIMAL(15,6),
    threshold_critical DECIMAL(15,6),
    status VARCHAR(20) DEFAULT 'normal' CHECK (status IN ('normal', 'warning', 'critical')),
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    INDEX(metric_type, component, timestamp),
    INDEX(metric_name, timestamp),
    INDEX(status, timestamp)
);

-- Real-time alerts and notifications
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(100) NOT NULL UNIQUE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    component VARCHAR(50),
    source VARCHAR(100),
    correlation_id VARCHAR(100),
    alert_data JSONB DEFAULT '{}',
    actions_taken JSONB DEFAULT '[]',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX(alert_type, severity, created_at),
    INDEX(component, created_at),
    INDEX(acknowledged, resolved, created_at),
    INDEX(correlation_id)
);

-- Stream subscription management
CREATE TABLE IF NOT EXISTS stream_subscriptions (
    id SERIAL PRIMARY KEY,
    subscription_id VARCHAR(100) NOT NULL UNIQUE,
    connection_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    stream_type VARCHAR(50) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    filters JSONB DEFAULT '{}',
    options JSONB DEFAULT '{}',
    message_count BIGINT DEFAULT 0,
    last_message_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'cancelled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    INDEX(connection_id, stream_type),
    INDEX(user_id, topic),
    INDEX(stream_type, status),
    INDEX(created_at)
);

-- Performance monitoring for analytics calculations
CREATE TABLE IF NOT EXISTS analytics_performance_log (
    id BIGSERIAL PRIMARY KEY,
    calculation_type VARCHAR(50) NOT NULL,
    portfolio_id VARCHAR(100),
    strategy_id VARCHAR(100),
    calculation_start TIMESTAMPTZ NOT NULL,
    calculation_end TIMESTAMPTZ NOT NULL,
    duration_ms INTEGER NOT NULL,
    records_processed INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    memory_usage_mb DECIMAL(10,2),
    cpu_usage_percent DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('completed', 'failed', 'timeout')),
    error_message TEXT,
    result_size_bytes INTEGER,
    INDEX(calculation_type, calculation_start),
    INDEX(portfolio_id, calculation_start),
    INDEX(strategy_id, calculation_start),
    INDEX(duration_ms),
    INDEX(status, calculation_start)
);

-- Create TimescaleDB hypertables for time-series optimization
DO $$
BEGIN
    -- Create hypertables for high-frequency data
    PERFORM create_hypertable('performance_metrics', 'timestamp', 
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    
    PERFORM create_hypertable('risk_events', 'timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    
    PERFORM create_hypertable('websocket_message_log', 'timestamp',
        chunk_time_interval => interval '1 hour', if_not_exists => TRUE);
    
    PERFORM create_hypertable('system_metrics', 'timestamp',
        chunk_time_interval => interval '1 hour', if_not_exists => TRUE);
    
    PERFORM create_hypertable('analytics_performance_log', 'calculation_start',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
        
EXCEPTION WHEN OTHERS THEN
    -- TimescaleDB not available, continue with regular tables
    NULL;
END
$$;

-- Create retention policies for Sprint 3 tables
DO $$
BEGIN
    -- Performance metrics: keep for 1 year
    PERFORM add_retention_policy('performance_metrics', interval '1 year', if_not_exists => TRUE);
    
    -- Risk events: keep for 2 years (compliance requirement)
    PERFORM add_retention_policy('risk_events', interval '2 years', if_not_exists => TRUE);
    
    -- WebSocket logs: keep for 30 days
    PERFORM add_retention_policy('websocket_message_log', interval '30 days', if_not_exists => TRUE);
    
    -- System metrics: keep for 90 days
    PERFORM add_retention_policy('system_metrics', interval '90 days', if_not_exists => TRUE);
    
    -- Analytics performance: keep for 180 days
    PERFORM add_retention_policy('analytics_performance_log', interval '180 days', if_not_exists => TRUE);
    
EXCEPTION WHEN OTHERS THEN
    -- TimescaleDB not available, continue without retention policies
    NULL;
END
$$;

-- Create functions for Sprint 3 analytics

-- Function to calculate real-time Sharpe ratio
CREATE OR REPLACE FUNCTION calculate_realtime_sharpe_ratio(
    p_strategy_id VARCHAR(100),
    p_lookback_days INTEGER DEFAULT 30,
    p_risk_free_rate DECIMAL(5,4) DEFAULT 0.02
)
RETURNS DECIMAL(8,4) AS $$
DECLARE
    avg_return DECIMAL(15,10);
    std_dev DECIMAL(15,10);
    sharpe_ratio DECIMAL(8,4);
BEGIN
    -- Get average daily return and standard deviation
    SELECT 
        AVG(metric_value) as avg_ret,
        STDDEV(metric_value) as std_ret
    INTO avg_return, std_dev
    FROM performance_metrics 
    WHERE strategy_id = p_strategy_id 
      AND metric_name = 'daily_return'
      AND timestamp >= NOW() - INTERVAL '1 day' * p_lookback_days;
    
    -- Calculate annualized Sharpe ratio
    IF std_dev IS NOT NULL AND std_dev > 0 THEN
        sharpe_ratio := (avg_return * 252 - p_risk_free_rate) / (std_dev * SQRT(252));
    ELSE
        sharpe_ratio := 0.0;
    END IF;
    
    RETURN sharpe_ratio;
END;
$$ LANGUAGE plpgsql;

-- Function to detect risk breaches
CREATE OR REPLACE FUNCTION detect_risk_breaches()
RETURNS TABLE(
    portfolio_id VARCHAR(100),
    breach_type VARCHAR(50),
    current_value DECIMAL(20,8),
    threshold_value DECIMAL(20,8),
    severity VARCHAR(20)
) AS $$
BEGIN
    -- This would implement real risk breach detection logic
    -- For now, return empty result set
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old WebSocket connection records
CREATE OR REPLACE FUNCTION cleanup_websocket_connections()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Mark connections as disconnected if no heartbeat in 5 minutes
    UPDATE websocket_connections 
    SET status = 'stale', disconnected_at = NOW()
    WHERE status = 'active' 
      AND last_heartbeat < NOW() - INTERVAL '5 minutes';
    
    -- Delete old disconnected connections (older than 24 hours)
    DELETE FROM websocket_connections 
    WHERE status IN ('disconnected', 'stale') 
      AND COALESCE(disconnected_at, connected_at) < NOW() - INTERVAL '24 hours';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic updates
CREATE TRIGGER update_system_alerts_updated_at 
    BEFORE UPDATE ON system_alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stream_subscriptions_updated_at 
    BEFORE UPDATE ON stream_subscriptions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for development testing
INSERT INTO performance_metrics (strategy_id, metric_name, metric_value) VALUES
('momentum_001', 'total_return', 12.50),
('momentum_001', 'sharpe_ratio', 1.85),
('momentum_001', 'max_drawdown', -5.20),
('mean_reversion_001', 'total_return', 8.75),
('mean_reversion_001', 'sharpe_ratio', 1.45),
('mean_reversion_001', 'max_drawdown', -3.10)
ON CONFLICT DO NOTHING;

INSERT INTO risk_events (event_type, severity, description, portfolio_id) VALUES
('position_limit_breach', 'MEDIUM', 'Position size exceeded 5% of portfolio', 'portfolio_main'),
('var_breach', 'HIGH', '1-day VaR exceeded by 15%', 'portfolio_main'),
('correlation_spike', 'LOW', 'Portfolio correlation increased to 0.85', 'portfolio_main')
ON CONFLICT DO NOTHING;

-- Create scheduled job to cleanup old data (if pg_cron extension is available)
-- This is optional and will only work if pg_cron is installed
DO $$
BEGIN
    -- Schedule cleanup job to run daily at 2 AM
    PERFORM cron.schedule('websocket-cleanup', '0 2 * * *', 'SELECT cleanup_websocket_connections();');
EXCEPTION WHEN OTHERS THEN
    -- pg_cron not available, cleanup must be done manually or via application
    NULL;
END
$$;

COMMENT ON TABLE performance_metrics IS 'Real-time performance metrics for strategies and portfolios (Sprint 3)';
COMMENT ON TABLE risk_events IS 'Risk monitoring events and breach detection log (Sprint 3)';
COMMENT ON TABLE websocket_connections IS 'Active WebSocket connection tracking for monitoring';
COMMENT ON TABLE system_alerts IS 'System-wide alerts and notification management';
COMMENT ON TABLE strategy_deployments IS 'Strategy deployment pipeline tracking';
COMMENT ON TABLE stream_subscriptions IS 'WebSocket stream subscription management';