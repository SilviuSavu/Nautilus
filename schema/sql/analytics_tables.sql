-- Advanced Performance Analytics Database Schema Extensions
-- Sprint 3 Priority 2 - Analytics Engine Tables
-- Extends existing schema with comprehensive analytics capabilities

-- Performance Analytics Tables

-- Performance snapshots for real-time P&L tracking
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    portfolio_id VARCHAR(100) NOT NULL,
    total_pnl DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8) NOT NULL,
    realized_pnl DECIMAL(20, 8) NOT NULL,
    total_return FLOAT NOT NULL,
    sharpe_ratio FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    win_rate FLOAT NOT NULL,
    profit_factor FLOAT NOT NULL,
    volatility FLOAT NOT NULL,
    alpha FLOAT,
    beta FLOAT,
    information_ratio FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(portfolio_id, timestamp)
);

-- Strategy performance metrics
CREATE TABLE IF NOT EXISTS strategy_performance (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    total_return FLOAT NOT NULL,
    annualized_return FLOAT NOT NULL,
    volatility FLOAT NOT NULL,
    sharpe_ratio FLOAT NOT NULL,
    sortino_ratio FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    calmar_ratio FLOAT NOT NULL,
    win_rate FLOAT NOT NULL,
    profit_factor FLOAT NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    avg_trade_duration FLOAT NOT NULL,
    total_pnl DECIMAL(20, 8) NOT NULL,
    alpha FLOAT,
    beta FLOAT,
    information_ratio FLOAT,
    tracking_error FLOAT,
    calculation_timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(strategy_id, start_date, end_date)
);

-- Strategy comparisons
CREATE TABLE IF NOT EXISTS strategy_comparisons (
    id BIGSERIAL PRIMARY KEY,
    comparison_id VARCHAR(100) NOT NULL UNIQUE,
    strategies JSONB NOT NULL,
    benchmark VARCHAR(50),
    comparison_period VARCHAR(10) NOT NULL,
    metrics_comparison JSONB NOT NULL,
    correlation_matrix JSONB NOT NULL,
    ranking JSONB NOT NULL,
    best_performer VARCHAR(100),
    worst_performer VARCHAR(100),
    analysis_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Strategy alpha/beta analysis
CREATE TABLE IF NOT EXISTS strategy_alpha_beta (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL,
    benchmark VARCHAR(50) NOT NULL,
    analysis_period VARCHAR(10) NOT NULL,
    alpha FLOAT NOT NULL,
    beta FLOAT NOT NULL,
    r_squared FLOAT NOT NULL,
    information_ratio FLOAT NOT NULL,
    tracking_error FLOAT NOT NULL,
    up_capture FLOAT NOT NULL,
    down_capture FLOAT NOT NULL,
    correlation FLOAT NOT NULL,
    observations INTEGER NOT NULL,
    calculation_date TIMESTAMP NOT NULL,
    UNIQUE(strategy_id, benchmark, analysis_period)
);

-- Strategy attribution analysis
CREATE TABLE IF NOT EXISTS strategy_attribution (
    id BIGSERIAL PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    instrument_contributions JSONB NOT NULL,
    sector_contributions JSONB NOT NULL,
    asset_class_contributions JSONB NOT NULL,
    top_contributors JSONB NOT NULL,
    top_detractors JSONB NOT NULL,
    concentration_metrics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(strategy_id, timestamp)
);

-- Risk Analytics Tables

-- VaR calculations
CREATE TABLE IF NOT EXISTS risk_var_calculations (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    method VARCHAR(20) NOT NULL,
    confidence_level FLOAT NOT NULL,
    time_horizon INTEGER NOT NULL,
    var_amount DECIMAL(20, 8) NOT NULL,
    expected_shortfall DECIMAL(20, 8) NOT NULL,
    calculation_timestamp TIMESTAMP NOT NULL,
    observations_used INTEGER NOT NULL,
    model_parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Portfolio exposure analysis
CREATE TABLE IF NOT EXISTS risk_exposure_analysis (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    total_exposure DECIMAL(20, 8) NOT NULL,
    net_exposure DECIMAL(20, 8) NOT NULL,
    gross_exposure DECIMAL(20, 8) NOT NULL,
    long_exposure DECIMAL(20, 8) NOT NULL,
    short_exposure DECIMAL(20, 8) NOT NULL,
    exposure_by_asset_class JSONB NOT NULL,
    exposure_by_sector JSONB NOT NULL,
    exposure_by_currency JSONB NOT NULL,
    concentration_metrics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(portfolio_id, timestamp)
);

-- Correlation analysis
CREATE TABLE IF NOT EXISTS risk_correlation_analysis (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    correlation_matrix JSONB NOT NULL,
    asset_ids JSONB NOT NULL,
    average_correlation FLOAT NOT NULL,
    max_correlation FLOAT NOT NULL,
    min_correlation FLOAT NOT NULL,
    eigenvalues JSONB NOT NULL,
    diversification_ratio FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(portfolio_id, timestamp)
);

-- Stress test results
CREATE TABLE IF NOT EXISTS risk_stress_tests (
    id BIGSERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL,
    scenario VARCHAR(50) NOT NULL,
    scenario_name VARCHAR(100) NOT NULL,
    portfolio_impact DECIMAL(20, 8) NOT NULL,
    impact_percentage FLOAT NOT NULL,
    positions_affected INTEGER NOT NULL,
    worst_position_impact DECIMAL(20, 8) NOT NULL,
    var_breach_probability FLOAT NOT NULL,
    recovery_time_estimate INTEGER,
    stress_factors JSONB NOT NULL,
    test_timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Execution Analytics Tables

-- Slippage analysis
CREATE TABLE IF NOT EXISTS execution_slippage_analysis (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL UNIQUE,
    instrument_id VARCHAR(100) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    intended_price DECIMAL(20, 8) NOT NULL,
    executed_price DECIMAL(20, 8) NOT NULL,
    slippage_bps FLOAT NOT NULL,
    slippage_amount DECIMAL(20, 8) NOT NULL,
    market_impact_bps FLOAT NOT NULL,
    execution_timestamp TIMESTAMP NOT NULL,
    market_conditions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Execution metrics aggregation
CREATE TABLE IF NOT EXISTS execution_metrics (
    id BIGSERIAL PRIMARY KEY,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    total_orders INTEGER NOT NULL,
    filled_orders INTEGER NOT NULL,
    fill_rate FLOAT NOT NULL,
    avg_execution_time_ms FLOAT NOT NULL,
    median_execution_time_ms FLOAT NOT NULL,
    avg_slippage_bps FLOAT NOT NULL,
    median_slippage_bps FLOAT NOT NULL,
    total_slippage_cost DECIMAL(20, 8) NOT NULL,
    market_impact_bps FLOAT NOT NULL,
    execution_quality_distribution JSONB NOT NULL,
    best_execution_rate FLOAT NOT NULL,
    worst_execution_rate FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(period_start, period_end)
);

-- Venue execution analysis
CREATE TABLE IF NOT EXISTS execution_venue_analysis (
    id BIGSERIAL PRIMARY KEY,
    venue VARCHAR(50) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    order_count INTEGER NOT NULL,
    fill_rate FLOAT NOT NULL,
    avg_slippage_bps FLOAT NOT NULL,
    avg_execution_time_ms FLOAT NOT NULL,
    market_share_pct FLOAT NOT NULL,
    execution_quality_score FLOAT NOT NULL,
    cost_per_share DECIMAL(20, 8) NOT NULL,
    venue_ranking INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(venue, period_start, period_end)
);

-- Market impact analysis
CREATE TABLE IF NOT EXISTS execution_market_impact (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL UNIQUE,
    instrument_id VARCHAR(100) NOT NULL,
    trade_size DECIMAL(20, 8) NOT NULL,
    order_value DECIMAL(20, 8) NOT NULL,
    adv_percentage FLOAT NOT NULL,
    pre_trade_spread DECIMAL(20, 8) NOT NULL,
    post_trade_spread DECIMAL(20, 8) NOT NULL,
    price_impact_bps FLOAT NOT NULL,
    temporary_impact_bps FLOAT NOT NULL,
    permanent_impact_bps FLOAT NOT NULL,
    impact_cost DECIMAL(20, 8) NOT NULL,
    liquidity_metrics JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Execution timing analysis
CREATE TABLE IF NOT EXISTS execution_timing_analysis (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL UNIQUE,
    instrument_id VARCHAR(100) NOT NULL,
    order_placement_time TIMESTAMP NOT NULL,
    first_fill_time TIMESTAMP,
    last_fill_time TIMESTAMP,
    total_execution_time_ms INTEGER NOT NULL,
    time_to_first_fill_ms INTEGER,
    fill_completion_time_ms INTEGER,
    market_session VARCHAR(20) NOT NULL,
    volatility_regime VARCHAR(20) NOT NULL,
    execution_urgency VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analytics Aggregation Tables

-- Aggregated analytics data
CREATE TABLE IF NOT EXISTS aggregated_analytics (
    id BIGSERIAL PRIMARY KEY,
    data_type VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metrics JSONB NOT NULL,
    record_count INTEGER NOT NULL,
    compression_ratio FLOAT,
    storage_size_bytes INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(data_type, interval, timestamp)
);

-- Aggregation jobs
CREATE TABLE IF NOT EXISTS aggregation_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_id VARCHAR(100) NOT NULL UNIQUE,
    data_type VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    filters JSONB,
    compression_level INTEGER NOT NULL,
    auto_cleanup BOOLEAN NOT NULL,
    retention_days INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    records_processed INTEGER,
    result_summary JSONB,
    error_message TEXT
);

-- Query optimization tracking
CREATE TABLE IF NOT EXISTS query_optimizations (
    id BIGSERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_pattern TEXT NOT NULL,
    execution_time_ms FLOAT NOT NULL,
    rows_examined INTEGER NOT NULL,
    rows_returned INTEGER NOT NULL,
    index_usage JSONB,
    optimization_suggestions JSONB,
    cache_hit BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Supporting Tables

-- Strategies table (if not exists)
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(100) NOT NULL UNIQUE,
    strategy_name VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Portfolios table (if not exists)
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(100) NOT NULL UNIQUE,
    portfolio_name VARCHAR(200) NOT NULL,
    description TEXT,
    base_currency VARCHAR(10) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Positions table (if not exists)
CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    position_id VARCHAR(100) NOT NULL UNIQUE,
    portfolio_id VARCHAR(100) NOT NULL,
    strategy_id VARCHAR(100),
    instrument_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    avg_entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    side VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Orders table (if not exists)
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL UNIQUE,
    portfolio_id VARCHAR(100) NOT NULL,
    strategy_id VARCHAR(100),
    instrument_id VARCHAR(100) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    limit_price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    time_in_force VARCHAR(10) DEFAULT 'DAY',
    venue VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    avg_fill_price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    commission DECIMAL(20, 8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    filled_at TIMESTAMP
);

-- Fills table (if not exists)
CREATE TABLE IF NOT EXISTS fills (
    id BIGSERIAL PRIMARY KEY,
    fill_id VARCHAR(100) NOT NULL UNIQUE,
    order_id VARCHAR(100) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    venue VARCHAR(50),
    commission DECIMAL(20, 8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trades table (if not exists)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(100) NOT NULL UNIQUE,
    portfolio_id VARCHAR(100) NOT NULL,
    strategy_id VARCHAR(100),
    instrument_id VARCHAR(100) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    side VARCHAR(10) NOT NULL,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    commission DECIMAL(20, 8) DEFAULT 0,
    timestamp TIMESTAMP NOT NULL,
    venue VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for optimal performance

-- Performance Analytics Indexes
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_portfolio_time ON performance_snapshots (portfolio_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_timestamp ON performance_snapshots (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_dates ON strategy_performance (strategy_id, start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_calculation_time ON strategy_performance (calculation_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_alpha_beta_strategy ON strategy_alpha_beta (strategy_id, benchmark);
CREATE INDEX IF NOT EXISTS idx_strategy_attribution_strategy_time ON strategy_attribution (strategy_id, timestamp DESC);

-- Risk Analytics Indexes
CREATE INDEX IF NOT EXISTS idx_risk_var_portfolio_time ON risk_var_calculations (portfolio_id, calculation_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_var_method_confidence ON risk_var_calculations (method, confidence_level);

CREATE INDEX IF NOT EXISTS idx_risk_exposure_portfolio_time ON risk_exposure_analysis (portfolio_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_correlation_portfolio_time ON risk_correlation_analysis (portfolio_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_stress_portfolio_time ON risk_stress_tests (portfolio_id, test_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_stress_scenario ON risk_stress_tests (scenario, test_timestamp DESC);

-- Execution Analytics Indexes
CREATE INDEX IF NOT EXISTS idx_execution_slippage_instrument_time ON execution_slippage_analysis (instrument_id, execution_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_execution_slippage_execution_time ON execution_slippage_analysis (execution_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_execution_metrics_period ON execution_metrics (period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_execution_venue_analysis_venue_period ON execution_venue_analysis (venue, period_start, period_end);

CREATE INDEX IF NOT EXISTS idx_execution_market_impact_instrument ON execution_market_impact (instrument_id);
CREATE INDEX IF NOT EXISTS idx_execution_timing_instrument_time ON execution_timing_analysis (instrument_id, order_placement_time DESC);

-- Aggregation Indexes
CREATE INDEX IF NOT EXISTS idx_aggregated_analytics_type_interval_time ON aggregated_analytics (data_type, interval, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_aggregated_analytics_timestamp ON aggregated_analytics (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_aggregation_jobs_status ON aggregation_jobs (status);
CREATE INDEX IF NOT EXISTS idx_aggregation_jobs_created_at ON aggregation_jobs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_query_optimizations_hash ON query_optimizations (query_hash);
CREATE INDEX IF NOT EXISTS idx_query_optimizations_created_at ON query_optimizations (created_at DESC);

-- Supporting table indexes
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies (status);
CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios (status);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_instrument ON positions (portfolio_id, instrument_id);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions (strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_status ON orders (portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders (strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fills_order_timestamp ON fills (order_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_portfolio_timestamp ON trades (portfolio_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_timestamp ON trades (strategy_id, timestamp DESC);

-- Create hypertables for time-series optimization (TimescaleDB)
-- This will work if TimescaleDB is available, otherwise will be ignored
DO $$
BEGIN
    -- Create hypertables for high-frequency analytics data
    -- 1 day chunks for performance snapshots
    PERFORM create_hypertable('performance_snapshots', 'timestamp', 
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    
    -- 1 week chunks for strategy performance (less frequent updates)
    PERFORM create_hypertable('strategy_performance', 'calculation_timestamp',
        chunk_time_interval => interval '1 week', if_not_exists => TRUE);
    
    -- 1 day chunks for risk calculations
    PERFORM create_hypertable('risk_var_calculations', 'calculation_timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    PERFORM create_hypertable('risk_exposure_analysis', 'timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    PERFORM create_hypertable('risk_correlation_analysis', 'timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    PERFORM create_hypertable('risk_stress_tests', 'test_timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    
    -- 1 hour chunks for execution analytics (high frequency)
    PERFORM create_hypertable('execution_slippage_analysis', 'execution_timestamp',
        chunk_time_interval => interval '1 hour', if_not_exists => TRUE);
    PERFORM create_hypertable('execution_timing_analysis', 'order_placement_time',
        chunk_time_interval => interval '1 hour', if_not_exists => TRUE);
    
    -- 1 day chunks for aggregated data
    PERFORM create_hypertable('aggregated_analytics', 'timestamp',
        chunk_time_interval => interval '1 day', if_not_exists => TRUE);
    
EXCEPTION WHEN OTHERS THEN
    -- TimescaleDB not available, continue with regular tables
    NULL;
END
$$;

-- Create data retention policies (TimescaleDB)
-- This will work if TimescaleDB is available, otherwise will be ignored
DO $$
BEGIN
    -- Set retention policies for different data types
    
    -- Keep detailed execution data for 30 days
    PERFORM add_retention_policy('execution_slippage_analysis', interval '30 days', if_not_exists => TRUE);
    PERFORM add_retention_policy('execution_timing_analysis', interval '30 days', if_not_exists => TRUE);
    
    -- Keep performance snapshots for 2 years
    PERFORM add_retention_policy('performance_snapshots', interval '2 years', if_not_exists => TRUE);
    
    -- Keep risk data for 1 year
    PERFORM add_retention_policy('risk_var_calculations', interval '1 year', if_not_exists => TRUE);
    PERFORM add_retention_policy('risk_exposure_analysis', interval '1 year', if_not_exists => TRUE);
    PERFORM add_retention_policy('risk_correlation_analysis', interval '1 year', if_not_exists => TRUE);
    PERFORM add_retention_policy('risk_stress_tests', interval '1 year', if_not_exists => TRUE);
    
    -- Keep aggregated data for 5 years
    PERFORM add_retention_policy('aggregated_analytics', interval '5 years', if_not_exists => TRUE);
    
EXCEPTION WHEN OTHERS THEN
    -- TimescaleDB not available, continue without retention policies
    NULL;
END
$$;

-- Create functions for analytics calculations

-- Function to calculate portfolio beta
CREATE OR REPLACE FUNCTION calculate_portfolio_beta(
    p_portfolio_id VARCHAR(100),
    p_benchmark_symbol VARCHAR(100),
    p_lookback_days INTEGER DEFAULT 252
)
RETURNS TABLE(beta FLOAT, alpha FLOAT, r_squared FLOAT, correlation FLOAT) AS $$
DECLARE
    portfolio_returns FLOAT[];
    benchmark_returns FLOAT[];
    beta_value FLOAT;
    alpha_value FLOAT;
    r_squared_value FLOAT;
    correlation_value FLOAT;
BEGIN
    -- This is a simplified implementation
    -- In practice, you would implement the full statistical calculation
    
    -- Return default values for now
    beta_value := 1.0;
    alpha_value := 0.0;
    r_squared_value := 0.5;
    correlation_value := 0.7;
    
    RETURN QUERY SELECT beta_value, alpha_value, r_squared_value, correlation_value;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old analytics data
CREATE OR REPLACE FUNCTION cleanup_analytics_data(retention_days INTEGER DEFAULT 30)
RETURNS TABLE(deleted_records BIGINT, table_name TEXT) AS $$
DECLARE
    cutoff_timestamp TIMESTAMP;
    deleted_count BIGINT;
    table_record RECORD;
BEGIN
    cutoff_timestamp := NOW() - INTERVAL '1 day' * retention_days;
    
    -- Tables to cleanup with their timestamp columns
    FOR table_record IN 
        SELECT 'execution_slippage_analysis' as tbl, 'execution_timestamp' as ts_col
        UNION ALL
        SELECT 'execution_timing_analysis', 'order_placement_time'
        UNION ALL
        SELECT 'performance_snapshots', 'timestamp'
        UNION ALL
        SELECT 'risk_var_calculations', 'calculation_timestamp'
    LOOP
        EXECUTE format('DELETE FROM %I WHERE %I < $1', table_record.tbl, table_record.ts_col)
        USING cutoff_timestamp;
        
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        
        RETURN QUERY SELECT deleted_count, table_record.tbl;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to relevant tables
CREATE TRIGGER update_strategies_updated_at 
    BEFORE UPDATE ON strategies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at 
    BEFORE UPDATE ON portfolios 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at 
    BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO strategies (strategy_id, strategy_name, description) VALUES
('momentum_001', 'Momentum Strategy V1', 'Trend-following momentum strategy'),
('mean_reversion_001', 'Mean Reversion V1', 'Statistical arbitrage mean reversion'),
('pairs_trading_001', 'Pairs Trading V1', 'Long-short equity pairs trading')
ON CONFLICT (strategy_id) DO NOTHING;

INSERT INTO portfolios (portfolio_id, portfolio_name, description) VALUES
('portfolio_main', 'Main Trading Portfolio', 'Primary algorithmic trading portfolio'),
('portfolio_test', 'Test Portfolio', 'Testing and development portfolio')
ON CONFLICT (portfolio_id) DO NOTHING;

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;