-- PostgreSQL Schema for Live Trading Data System
-- Optimized for real-time data integration and web applications
-- Designed to complement NautilusTrader's research-focused architecture

-- Enable TimescaleDB extension for time-series optimization (optional)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Market Instruments table
CREATE TABLE IF NOT EXISTS instruments (
    id SERIAL PRIMARY KEY,
    venue VARCHAR(50) NOT NULL,
    symbol VARCHAR(100) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL UNIQUE,
    asset_class VARCHAR(50) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    tick_size DECIMAL(20, 8),
    min_quantity DECIMAL(20, 8),
    max_quantity DECIMAL(20, 8),
    lot_size DECIMAL(20, 8),
    multiplier DECIMAL(20, 8) DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(venue, symbol)
);

-- Market Ticks table (nanosecond precision)
CREATE TABLE IF NOT EXISTS market_ticks (
    id BIGSERIAL,
    venue VARCHAR(50) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    timestamp_ns BIGINT NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    side VARCHAR(10),
    trade_id VARCHAR(100),
    sequence_num BIGINT,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Market Quotes table (nanosecond precision)
CREATE TABLE IF NOT EXISTS market_quotes (
    id BIGSERIAL,
    venue VARCHAR(50) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    timestamp_ns BIGINT NOT NULL,
    bid_price DECIMAL(20, 8) NOT NULL,
    ask_price DECIMAL(20, 8) NOT NULL,
    bid_size DECIMAL(20, 8) NOT NULL,
    ask_size DECIMAL(20, 8) NOT NULL,
    spread DECIMAL(20, 8),
    sequence_num BIGINT,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Market Bars table (OHLCV data with nanosecond precision)
CREATE TABLE IF NOT EXISTS market_bars (
    id BIGSERIAL,
    venue VARCHAR(50) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp_ns BIGINT NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    is_final BOOLEAN DEFAULT TRUE,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(venue, instrument_id, timeframe, timestamp_ns)
);

-- Live Trading Sessions table
CREATE TABLE IF NOT EXISTS trading_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    venue VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    configuration JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Real-time Data Subscriptions table
CREATE TABLE IF NOT EXISTS data_subscriptions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    venue VARCHAR(50) NOT NULL,
    instrument_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(20) NOT NULL, -- 'tick', 'quote', 'bar'
    timeframe VARCHAR(10), -- for bars
    status VARCHAR(20) DEFAULT 'active',
    last_update TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES trading_sessions(session_id)
);

-- Create hypertables for time-series optimization (TimescaleDB)
-- This will work if TimescaleDB is available, otherwise will be ignored
DO $$
BEGIN
    -- Create hypertables with appropriate chunk intervals
    -- 1 day chunks for ticks and quotes (high frequency data)
    PERFORM create_hypertable('market_ticks', 'timestamp_ns', 
        chunk_time_interval => 86400000000000, if_not_exists => TRUE);
    PERFORM create_hypertable('market_quotes', 'timestamp_ns',
        chunk_time_interval => 86400000000000, if_not_exists => TRUE);
    -- 1 week chunks for bars (lower frequency data)
    PERFORM create_hypertable('market_bars', 'timestamp_ns',
        chunk_time_interval => 604800000000000, if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    -- TimescaleDB not available, continue with regular tables
    NULL;
END
$$;

-- Create indexes for optimal query performance
-- Instruments indexes
CREATE INDEX IF NOT EXISTS idx_instruments_venue_symbol ON instruments (venue, symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_asset_class ON instruments (asset_class);
CREATE INDEX IF NOT EXISTS idx_instruments_active ON instruments (is_active);

-- Ticks indexes (optimized for time-series queries)
CREATE INDEX IF NOT EXISTS idx_ticks_venue_instrument ON market_ticks (venue, instrument_id, timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_timestamp ON market_ticks (timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_instrument_time ON market_ticks (instrument_id, timestamp_ns DESC);

-- Quotes indexes
CREATE INDEX IF NOT EXISTS idx_quotes_venue_instrument ON market_quotes (venue, instrument_id, timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_quotes_timestamp ON market_quotes (timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_quotes_instrument_time ON market_quotes (instrument_id, timestamp_ns DESC);

-- Bars indexes (includes timeframe for aggregation queries)
CREATE INDEX IF NOT EXISTS idx_bars_venue_instrument_tf ON market_bars (venue, instrument_id, timeframe, timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_bars_timestamp ON market_bars (timestamp_ns DESC);
CREATE INDEX IF NOT EXISTS idx_bars_instrument_timeframe_time ON market_bars (instrument_id, timeframe, timestamp_ns DESC);

-- Sessions and subscriptions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_venue_status ON trading_sessions (venue, status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_session ON data_subscriptions (session_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_venue_instrument ON data_subscriptions (venue, instrument_id);

-- Create views for common queries
CREATE OR REPLACE VIEW latest_quotes AS
SELECT DISTINCT ON (venue, instrument_id)
    venue, instrument_id, timestamp_ns, bid_price, ask_price, 
    bid_size, ask_size, spread, created_at
FROM market_quotes
ORDER BY venue, instrument_id, timestamp_ns DESC;

CREATE OR REPLACE VIEW latest_ticks AS
SELECT DISTINCT ON (venue, instrument_id)
    venue, instrument_id, timestamp_ns, price, size, side, trade_id, created_at
FROM market_ticks
ORDER BY venue, instrument_id, timestamp_ns DESC;

-- Create function for real-time data cleanup
CREATE OR REPLACE FUNCTION cleanup_old_market_data(retention_days INTEGER DEFAULT 7)
RETURNS TABLE(deleted_ticks BIGINT, deleted_quotes BIGINT, deleted_bars BIGINT) AS $$
DECLARE
    cutoff_timestamp BIGINT;
    tick_count BIGINT;
    quote_count BIGINT;
    bar_count BIGINT;
BEGIN
    -- Calculate cutoff timestamp (retention_days ago)
    cutoff_timestamp := EXTRACT(EPOCH FROM NOW() - INTERVAL '1 day' * retention_days) * 1000000000;
    
    -- Delete old ticks
    DELETE FROM market_ticks WHERE timestamp_ns < cutoff_timestamp;
    GET DIAGNOSTICS tick_count = ROW_COUNT;
    
    -- Delete old quotes
    DELETE FROM market_quotes WHERE timestamp_ns < cutoff_timestamp;
    GET DIAGNOSTICS quote_count = ROW_COUNT;
    
    -- Delete old bars (keep longer for analysis)
    DELETE FROM market_bars WHERE timestamp_ns < cutoff_timestamp AND timeframe IN ('1m', '5m');
    GET DIAGNOSTICS bar_count = ROW_COUNT;
    
    RETURN QUERY SELECT tick_count, quote_count, bar_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for data export to Parquet format (for NautilusTrader compatibility)
CREATE OR REPLACE FUNCTION prepare_export_data(
    p_venue VARCHAR(50),
    p_instrument_id VARCHAR(100),
    p_start_time BIGINT,
    p_end_time BIGINT,
    p_data_type VARCHAR(20) DEFAULT 'bar',
    p_timeframe VARCHAR(10) DEFAULT '1h'
)
RETURNS TABLE(export_data JSONB) AS $$
BEGIN
    IF p_data_type = 'tick' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'timestamp_ns', timestamp_ns,
            'price', price,
            'size', size,
            'side', side,
            'trade_id', trade_id
        ) FROM market_ticks
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timestamp_ns >= p_start_time 
        AND timestamp_ns <= p_end_time
        ORDER BY timestamp_ns;
        
    ELSIF p_data_type = 'quote' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'timestamp_ns', timestamp_ns,
            'bid_price', bid_price,
            'ask_price', ask_price,
            'bid_size', bid_size,
            'ask_size', ask_size
        ) FROM market_quotes
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timestamp_ns >= p_start_time 
        AND timestamp_ns <= p_end_time
        ORDER BY timestamp_ns;
        
    ELSIF p_data_type = 'bar' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'timeframe', timeframe,
            'timestamp_ns', timestamp_ns,
            'open', open_price,
            'high', high_price,
            'low', low_price,
            'close', close_price,
            'volume', volume
        ) FROM market_bars
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timeframe = p_timeframe
        AND timestamp_ns >= p_start_time 
        AND timestamp_ns <= p_end_time
        ORDER BY timestamp_ns;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Insert some default instrument configurations
INSERT INTO instruments (venue, symbol, instrument_id, asset_class, currency, tick_size, min_quantity, multiplier) VALUES
('IB', 'AAPL', 'AAPL.NASDAQ', 'STK', 'USD', 0.01, 1, 1),
('IB', 'EURUSD', 'EUR.USD', 'CASH', 'USD', 0.00001, 25000, 1),
('IB', 'ES', 'ES.CME', 'FUT', 'USD', 0.25, 1, 50),
('IB', 'NQ', 'NQ.CME', 'FUT', 'USD', 0.25, 1, 20),
('IB', 'SPY', 'SPY.ARCA', 'STK', 'USD', 0.01, 1, 1),
('IB', 'QQQ', 'QQQ.NASDAQ', 'STK', 'USD', 0.01, 1, 1),
('IB', 'TSLA', 'TSLA.NASDAQ', 'STK', 'USD', 0.01, 1, 1),
('IB', 'AMZN', 'AMZN.NASDAQ', 'STK', 'USD', 0.01, 1, 1),
('IB', 'GOOGL', 'GOOGL.NASDAQ', 'STK', 'USD', 0.01, 1, 1),
('IB', 'MSFT', 'MSFT.NASDAQ', 'STK', 'USD', 0.01, 1, 1)
ON CONFLICT (venue, symbol) DO NOTHING;

-- Create triggers to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_instruments_updated_at 
    BEFORE UPDATE ON instruments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();