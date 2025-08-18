-- PostgreSQL Functions for Live Trading Data System
-- Supporting real-time data processing and NautilusTrader compatibility

-- Function to get market data statistics
CREATE OR REPLACE FUNCTION get_market_data_stats(
    p_venue VARCHAR(50) DEFAULT NULL,
    p_instrument_id VARCHAR(100) DEFAULT NULL,
    p_hours_lookback INTEGER DEFAULT 24
)
RETURNS TABLE(
    data_type VARCHAR(20),
    venue VARCHAR(50),
    instrument_id VARCHAR(100),
    record_count BIGINT,
    first_timestamp BIGINT,
    last_timestamp BIGINT,
    time_range_hours NUMERIC
) AS $$
DECLARE
    cutoff_timestamp BIGINT;
BEGIN
    -- Calculate lookback timestamp
    cutoff_timestamp := EXTRACT(EPOCH FROM NOW() - INTERVAL '1 hour' * p_hours_lookback) * 1000000000;
    
    -- Return tick statistics
    RETURN QUERY
    SELECT 
        'tick'::VARCHAR(20) as data_type,
        t.venue,
        t.instrument_id,
        COUNT(*)::BIGINT as record_count,
        MIN(t.timestamp_ns) as first_timestamp,
        MAX(t.timestamp_ns) as last_timestamp,
        (MAX(t.timestamp_ns) - MIN(t.timestamp_ns))::NUMERIC / 3600000000000 as time_range_hours
    FROM market_ticks t
    WHERE (p_venue IS NULL OR t.venue = p_venue)
    AND (p_instrument_id IS NULL OR t.instrument_id = p_instrument_id)
    AND t.timestamp_ns >= cutoff_timestamp
    GROUP BY t.venue, t.instrument_id
    HAVING COUNT(*) > 0;
    
    -- Return quote statistics
    RETURN QUERY
    SELECT 
        'quote'::VARCHAR(20) as data_type,
        q.venue,
        q.instrument_id,
        COUNT(*)::BIGINT as record_count,
        MIN(q.timestamp_ns) as first_timestamp,
        MAX(q.timestamp_ns) as last_timestamp,
        (MAX(q.timestamp_ns) - MIN(q.timestamp_ns))::NUMERIC / 3600000000000 as time_range_hours
    FROM market_quotes q
    WHERE (p_venue IS NULL OR q.venue = p_venue)
    AND (p_instrument_id IS NULL OR q.instrument_id = p_instrument_id)
    AND q.timestamp_ns >= cutoff_timestamp
    GROUP BY q.venue, q.instrument_id
    HAVING COUNT(*) > 0;
    
    -- Return bar statistics
    RETURN QUERY
    SELECT 
        ('bar_' || b.timeframe)::VARCHAR(20) as data_type,
        b.venue,
        b.instrument_id,
        COUNT(*)::BIGINT as record_count,
        MIN(b.timestamp_ns) as first_timestamp,
        MAX(b.timestamp_ns) as last_timestamp,
        (MAX(b.timestamp_ns) - MIN(b.timestamp_ns))::NUMERIC / 3600000000000 as time_range_hours
    FROM market_bars b
    WHERE (p_venue IS NULL OR b.venue = p_venue)
    AND (p_instrument_id IS NULL OR b.instrument_id = p_instrument_id)
    AND b.timestamp_ns >= cutoff_timestamp
    GROUP BY b.venue, b.instrument_id, b.timeframe
    HAVING COUNT(*) > 0;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest market data for dashboard
CREATE OR REPLACE FUNCTION get_latest_market_data(
    p_venue VARCHAR(50),
    p_instrument_id VARCHAR(100)
)
RETURNS TABLE(
    latest_tick JSONB,
    latest_quote JSONB,
    latest_bars JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT jsonb_build_object(
            'timestamp_ns', timestamp_ns,
            'price', price,
            'size', size,
            'side', side,
            'trade_id', trade_id
        ) FROM market_ticks 
        WHERE venue = p_venue AND instrument_id = p_instrument_id
        ORDER BY timestamp_ns DESC LIMIT 1) as latest_tick,
        
        (SELECT jsonb_build_object(
            'timestamp_ns', timestamp_ns,
            'bid_price', bid_price,
            'ask_price', ask_price,
            'bid_size', bid_size,
            'ask_size', ask_size,
            'spread', spread
        ) FROM market_quotes
        WHERE venue = p_venue AND instrument_id = p_instrument_id
        ORDER BY timestamp_ns DESC LIMIT 1) as latest_quote,
        
        (SELECT jsonb_agg(
            jsonb_build_object(
                'timeframe', timeframe,
                'timestamp_ns', timestamp_ns,
                'open', open_price,
                'high', high_price,
                'low', low_price,
                'close', close_price,
                'volume', volume
            )
        ) FROM (
            SELECT DISTINCT ON (timeframe) *
            FROM market_bars
            WHERE venue = p_venue AND instrument_id = p_instrument_id
            ORDER BY timeframe, timestamp_ns DESC
        ) latest_bars_by_tf) as latest_bars;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate tick data into bars (real-time bar generation)
CREATE OR REPLACE FUNCTION aggregate_ticks_to_bars(
    p_venue VARCHAR(50),
    p_instrument_id VARCHAR(100),
    p_timeframe VARCHAR(10),
    p_start_time BIGINT,
    p_end_time BIGINT
)
RETURNS TABLE(
    timestamp_ns BIGINT,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8),
    volume DECIMAL(20,8)
) AS $$
DECLARE
    timeframe_seconds BIGINT;
    bucket_size_ns BIGINT;
BEGIN
    -- Convert timeframe to nanoseconds
    CASE p_timeframe
        WHEN '1m' THEN bucket_size_ns := 60 * 1000000000;
        WHEN '5m' THEN bucket_size_ns := 300 * 1000000000;
        WHEN '15m' THEN bucket_size_ns := 900 * 1000000000;
        WHEN '1h' THEN bucket_size_ns := 3600 * 1000000000;
        WHEN '4h' THEN bucket_size_ns := 14400 * 1000000000;
        WHEN '1d' THEN bucket_size_ns := 86400 * 1000000000;
        ELSE bucket_size_ns := 3600 * 1000000000; -- Default to 1h
    END CASE;
    
    RETURN QUERY
    SELECT 
        (t.timestamp_ns / bucket_size_ns) * bucket_size_ns as timestamp_ns,
        (array_agg(t.price ORDER BY t.timestamp_ns ASC))[1] as open_price,
        MAX(t.price) as high_price,
        MIN(t.price) as low_price,
        (array_agg(t.price ORDER BY t.timestamp_ns DESC))[1] as close_price,
        SUM(t.size) as volume
    FROM market_ticks t
    WHERE t.venue = p_venue 
    AND t.instrument_id = p_instrument_id
    AND t.timestamp_ns >= p_start_time
    AND t.timestamp_ns <= p_end_time
    GROUP BY (t.timestamp_ns / bucket_size_ns)
    ORDER BY timestamp_ns;
END;
$$ LANGUAGE plpgsql;

-- Function to export data for Parquet format (NautilusTrader compatibility)
CREATE OR REPLACE FUNCTION export_for_nautilus(
    p_venue VARCHAR(50),
    p_instrument_id VARCHAR(100),
    p_start_time TIMESTAMP,
    p_end_time TIMESTAMP,
    p_data_type VARCHAR(20) DEFAULT 'bar',
    p_timeframe VARCHAR(10) DEFAULT '1h'
)
RETURNS TABLE(export_record JSONB) AS $$
DECLARE
    start_ns BIGINT;
    end_ns BIGINT;
BEGIN
    -- Convert timestamps to nanoseconds
    start_ns := EXTRACT(EPOCH FROM p_start_time) * 1000000000;
    end_ns := EXTRACT(EPOCH FROM p_end_time) * 1000000000;
    
    IF p_data_type = 'tick' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'ts_event', timestamp_ns,
            'ts_init', timestamp_ns,
            'price', price,
            'size', size,
            'aggressor_side', CASE WHEN side = 'BUY' THEN 1 WHEN side = 'SELL' THEN 2 ELSE 0 END,
            'trade_id', trade_id
        ) as export_record
        FROM market_ticks
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timestamp_ns >= start_ns 
        AND timestamp_ns <= end_ns
        ORDER BY timestamp_ns;
        
    ELSIF p_data_type = 'quote' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'ts_event', timestamp_ns,
            'ts_init', timestamp_ns,
            'bid_price', bid_price,
            'ask_price', ask_price,
            'bid_size', bid_size,
            'ask_size', ask_size
        ) as export_record
        FROM market_quotes
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timestamp_ns >= start_ns 
        AND timestamp_ns <= end_ns
        ORDER BY timestamp_ns;
        
    ELSIF p_data_type = 'bar' THEN
        RETURN QUERY
        SELECT jsonb_build_object(
            'venue', venue,
            'instrument_id', instrument_id,
            'bar_type', timeframe,
            'ts_event', timestamp_ns,
            'ts_init', timestamp_ns,
            'open', open_price,
            'high', high_price,
            'low', low_price,
            'close', close_price,
            'volume', volume
        ) as export_record
        FROM market_bars
        WHERE venue = p_venue 
        AND instrument_id = p_instrument_id
        AND timeframe = p_timeframe
        AND timestamp_ns >= start_ns 
        AND timestamp_ns <= end_ns
        ORDER BY timestamp_ns;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate trading session statistics
CREATE OR REPLACE FUNCTION get_session_stats(
    p_session_id VARCHAR(100)
)
RETURNS TABLE(
    session_id VARCHAR(100),
    venue VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_hours NUMERIC,
    instruments_traded INTEGER,
    total_ticks BIGINT,
    total_quotes BIGINT,
    total_bars BIGINT,
    avg_spread NUMERIC,
    price_range JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ts.session_id,
        ts.venue,
        ts.start_time,
        ts.end_time,
        EXTRACT(EPOCH FROM (COALESCE(ts.end_time, NOW()) - ts.start_time)) / 3600 as duration_hours,
        (SELECT COUNT(DISTINCT ds.instrument_id) 
         FROM data_subscriptions ds 
         WHERE ds.session_id = ts.session_id)::INTEGER as instruments_traded,
        (SELECT COUNT(*) 
         FROM market_ticks mt 
         JOIN data_subscriptions ds ON ds.venue = mt.venue AND ds.instrument_id = mt.instrument_id
         WHERE ds.session_id = ts.session_id 
         AND mt.timestamp_ns >= EXTRACT(EPOCH FROM ts.start_time) * 1000000000)::BIGINT as total_ticks,
        (SELECT COUNT(*) 
         FROM market_quotes mq 
         JOIN data_subscriptions ds ON ds.venue = mq.venue AND ds.instrument_id = mq.instrument_id
         WHERE ds.session_id = ts.session_id 
         AND mq.timestamp_ns >= EXTRACT(EPOCH FROM ts.start_time) * 1000000000)::BIGINT as total_quotes,
        (SELECT COUNT(*) 
         FROM market_bars mb 
         JOIN data_subscriptions ds ON ds.venue = mb.venue AND ds.instrument_id = mb.instrument_id
         WHERE ds.session_id = ts.session_id 
         AND mb.timestamp_ns >= EXTRACT(EPOCH FROM ts.start_time) * 1000000000)::BIGINT as total_bars,
        (SELECT AVG(spread) 
         FROM market_quotes mq 
         JOIN data_subscriptions ds ON ds.venue = mq.venue AND ds.instrument_id = mq.instrument_id
         WHERE ds.session_id = ts.session_id 
         AND mq.timestamp_ns >= EXTRACT(EPOCH FROM ts.start_time) * 1000000000) as avg_spread,
        (SELECT jsonb_object_agg(
            ds.instrument_id,
            jsonb_build_object(
                'min_price', MIN(mt.price),
                'max_price', MAX(mt.price),
                'first_price', (array_agg(mt.price ORDER BY mt.timestamp_ns))[1],
                'last_price', (array_agg(mt.price ORDER BY mt.timestamp_ns DESC))[1]
            )
         )
         FROM market_ticks mt 
         JOIN data_subscriptions ds ON ds.venue = mt.venue AND ds.instrument_id = mt.instrument_id
         WHERE ds.session_id = ts.session_id 
         AND mt.timestamp_ns >= EXTRACT(EPOCH FROM ts.start_time) * 1000000000
         GROUP BY ds.instrument_id) as price_range
    FROM trading_sessions ts
    WHERE ts.session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Function to manage data retention policies
CREATE OR REPLACE FUNCTION apply_retention_policy()
RETURNS TABLE(
    retention_summary JSONB
) AS $$
DECLARE
    tick_retention_days INTEGER := 3;      -- Keep ticks for 3 days
    quote_retention_days INTEGER := 7;     -- Keep quotes for 1 week  
    bar_retention_days INTEGER := 365;     -- Keep bars for 1 year
    tick_cutoff BIGINT;
    quote_cutoff BIGINT;
    bar_cutoff BIGINT;
    deleted_ticks BIGINT;
    deleted_quotes BIGINT;
    deleted_bars BIGINT;
BEGIN
    -- Calculate cutoff timestamps
    tick_cutoff := EXTRACT(EPOCH FROM NOW() - INTERVAL '1 day' * tick_retention_days) * 1000000000;
    quote_cutoff := EXTRACT(EPOCH FROM NOW() - INTERVAL '1 day' * quote_retention_days) * 1000000000;
    bar_cutoff := EXTRACT(EPOCH FROM NOW() - INTERVAL '1 day' * bar_retention_days) * 1000000000;
    
    -- Delete old ticks
    DELETE FROM market_ticks WHERE timestamp_ns < tick_cutoff;
    GET DIAGNOSTICS deleted_ticks = ROW_COUNT;
    
    -- Delete old quotes
    DELETE FROM market_quotes WHERE timestamp_ns < quote_cutoff;
    GET DIAGNOSTICS deleted_quotes = ROW_COUNT;
    
    -- Delete old high-frequency bars only (keep daily+ bars longer)
    DELETE FROM market_bars 
    WHERE timestamp_ns < bar_cutoff 
    AND timeframe IN ('1m', '5m', '15m');
    GET DIAGNOSTICS deleted_bars = ROW_COUNT;
    
    RETURN QUERY
    SELECT jsonb_build_object(
        'executed_at', NOW(),
        'deleted_records', jsonb_build_object(
            'ticks', deleted_ticks,
            'quotes', deleted_quotes,
            'bars', deleted_bars
        ),
        'retention_policies', jsonb_build_object(
            'tick_retention_days', tick_retention_days,
            'quote_retention_days', quote_retention_days,
            'bar_retention_days', bar_retention_days
        )
    ) as retention_summary;
END;
$$ LANGUAGE plpgsql;

-- Function to get real-time performance metrics
CREATE OR REPLACE FUNCTION get_realtime_performance()
RETURNS TABLE(
    metric_name VARCHAR(50),
    metric_value NUMERIC,
    metric_unit VARCHAR(20),
    timestamp_utc TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM (
        -- Data ingestion rate (records per minute)
        SELECT 
            'tick_ingestion_rate' as metric_name,
            COUNT(*)::NUMERIC / 1.0 as metric_value,
            'records/min' as metric_unit,
            NOW() as timestamp_utc
        FROM market_ticks 
        WHERE timestamp_ns >= EXTRACT(EPOCH FROM NOW() - INTERVAL '1 minute') * 1000000000
        
        UNION ALL
        
        SELECT 
            'quote_ingestion_rate' as metric_name,
            COUNT(*)::NUMERIC / 1.0 as metric_value,
            'records/min' as metric_unit,
            NOW() as timestamp_utc
        FROM market_quotes 
        WHERE timestamp_ns >= EXTRACT(EPOCH FROM NOW() - INTERVAL '1 minute') * 1000000000
        
        UNION ALL
        
        -- Database size metrics
        SELECT 
            'database_size' as metric_name,
            pg_database_size(current_database())::NUMERIC / (1024*1024) as metric_value,
            'MB' as metric_unit,
            NOW() as timestamp_utc
            
        UNION ALL
        
        -- Active instruments count
        SELECT 
            'active_instruments' as metric_name,
            COUNT(DISTINCT instrument_id)::NUMERIC as metric_value,
            'count' as metric_unit,
            NOW() as timestamp_utc
        FROM market_ticks 
        WHERE timestamp_ns >= EXTRACT(EPOCH FROM NOW() - INTERVAL '1 hour') * 1000000000
    ) metrics
    ORDER BY metric_name;
END;
$$ LANGUAGE plpgsql;