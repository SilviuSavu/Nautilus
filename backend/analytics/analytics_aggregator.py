"""
Analytics Data Aggregator for Sprint 3 Priority 2
Time-series data aggregation, historical performance storage, data compression, and query optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncpg
import json
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AggregationInterval(Enum):
    """Data aggregation intervals"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class DataType(Enum):
    """Types of data for aggregation"""
    PERFORMANCE = "performance"
    RISK = "risk"
    EXECUTION = "execution"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"

class CompressionLevel(Enum):
    """Data compression levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9

@dataclass
class AggregationJob:
    """Data aggregation job configuration"""
    job_id: str
    data_type: DataType
    interval: AggregationInterval
    start_date: datetime
    end_date: datetime
    filters: Dict[str, Any]
    compression_level: CompressionLevel
    auto_cleanup: bool
    retention_days: int

@dataclass
class AggregatedMetrics:
    """Aggregated metrics result"""
    data_type: DataType
    interval: AggregationInterval
    timestamp: datetime
    metrics: Dict[str, Any]
    record_count: int
    compression_ratio: Optional[float]
    storage_size_bytes: int

@dataclass
class QueryOptimization:
    """Query optimization statistics"""
    query_hash: str
    query_pattern: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    index_usage: List[str]
    optimization_suggestions: List[str]
    cache_hit: bool

class AnalyticsAggregator:
    """
    Analytics data aggregation and storage optimization engine
    Handles time-series aggregation, compression, and query optimization
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Aggregation configurations
        self.aggregation_configs = {
            DataType.PERFORMANCE: {
                AggregationInterval.MINUTE_5: ['total_pnl', 'unrealized_pnl', 'realized_pnl'],
                AggregationInterval.HOUR_1: ['sharpe_ratio', 'max_drawdown', 'win_rate'],
                AggregationInterval.DAY_1: ['total_return', 'volatility', 'alpha', 'beta']
            },
            DataType.RISK: {
                AggregationInterval.HOUR_1: ['var_amount', 'expected_shortfall'],
                AggregationInterval.DAY_1: ['correlation_matrix', 'exposure_metrics']
            },
            DataType.EXECUTION: {
                AggregationInterval.MINUTE_1: ['slippage_bps', 'execution_time_ms'],
                AggregationInterval.HOUR_1: ['fill_rate', 'avg_slippage_bps']
            }
        }
        
        # Retention policies (in days)
        self.retention_policies = {
            AggregationInterval.MINUTE_1: 7,
            AggregationInterval.MINUTE_5: 30,
            AggregationInterval.MINUTE_15: 90,
            AggregationInterval.HOUR_1: 365,
            AggregationInterval.DAY_1: 1825,  # 5 years
            AggregationInterval.WEEK_1: 3650, # 10 years
            AggregationInterval.MONTH_1: 7300 # 20 years
        }
    
    async def aggregate_performance_data(
        self,
        interval: AggregationInterval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        portfolio_ids: Optional[List[str]] = None
    ) -> List[AggregatedMetrics]:
        """
        Aggregate performance data for specified interval
        
        Args:
            interval: Aggregation interval
            start_date: Start date for aggregation
            end_date: End date for aggregation
            portfolio_ids: Optional list of portfolio IDs to aggregate
            
        Returns:
            List of aggregated metrics
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=1)
        
        async with self.db_pool.acquire() as conn:
            try:
                # Generate time buckets
                time_buckets = self._generate_time_buckets(start_date, end_date, interval)
                
                aggregated_results = []
                
                for bucket_start, bucket_end in time_buckets:
                    # Build query filters
                    where_conditions = ["ps.timestamp >= $1", "ps.timestamp < $2"]
                    params = [bucket_start, bucket_end]
                    
                    if portfolio_ids:
                        placeholders = ",".join(f"${i+3}" for i in range(len(portfolio_ids)))
                        where_conditions.append(f"ps.portfolio_id IN ({placeholders})")
                        params.extend(portfolio_ids)
                    
                    where_clause = " AND ".join(where_conditions)
                    
                    # Aggregate performance metrics
                    agg_query = f"""
                        SELECT 
                            COUNT(*) as record_count,
                            AVG(total_pnl) as avg_total_pnl,
                            SUM(total_pnl) as sum_total_pnl,
                            AVG(unrealized_pnl) as avg_unrealized_pnl,
                            AVG(realized_pnl) as avg_realized_pnl,
                            AVG(total_return) as avg_total_return,
                            AVG(sharpe_ratio) as avg_sharpe_ratio,
                            MIN(max_drawdown) as worst_drawdown,
                            AVG(win_rate) as avg_win_rate,
                            AVG(profit_factor) as avg_profit_factor,
                            AVG(volatility) as avg_volatility,
                            AVG(alpha) as avg_alpha,
                            AVG(beta) as avg_beta
                        FROM performance_snapshots ps
                        WHERE {where_clause}
                    """
                    
                    result = await conn.fetchrow(agg_query, *params)
                    
                    if result and result['record_count'] > 0:
                        metrics = {
                            'avg_total_pnl': float(result['avg_total_pnl'] or 0),
                            'sum_total_pnl': float(result['sum_total_pnl'] or 0),
                            'avg_unrealized_pnl': float(result['avg_unrealized_pnl'] or 0),
                            'avg_realized_pnl': float(result['avg_realized_pnl'] or 0),
                            'avg_total_return': float(result['avg_total_return'] or 0),
                            'avg_sharpe_ratio': float(result['avg_sharpe_ratio'] or 0),
                            'worst_drawdown': float(result['worst_drawdown'] or 0),
                            'avg_win_rate': float(result['avg_win_rate'] or 0),
                            'avg_profit_factor': float(result['avg_profit_factor'] or 0),
                            'avg_volatility': float(result['avg_volatility'] or 0),
                            'avg_alpha': float(result['avg_alpha'] or 0) if result['avg_alpha'] else None,
                            'avg_beta': float(result['avg_beta'] or 0) if result['avg_beta'] else None,
                            'time_bucket_start': bucket_start.isoformat(),
                            'time_bucket_end': bucket_end.isoformat()
                        }
                        
                        # Compress data if needed
                        compressed_metrics, compression_ratio, storage_size = await self._compress_data(
                            metrics, CompressionLevel.MEDIUM
                        )
                        
                        aggregated_metric = AggregatedMetrics(
                            data_type=DataType.PERFORMANCE,
                            interval=interval,
                            timestamp=bucket_start,
                            metrics=compressed_metrics,
                            record_count=result['record_count'],
                            compression_ratio=compression_ratio,
                            storage_size_bytes=storage_size
                        )
                        
                        # Store aggregated data
                        await self._store_aggregated_data(conn, aggregated_metric)
                        aggregated_results.append(aggregated_metric)
                
                return aggregated_results
                
            except Exception as e:
                self.logger.error(f"Error aggregating performance data: {e}")
                raise
    
    async def aggregate_risk_data(
        self,
        interval: AggregationInterval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        portfolio_ids: Optional[List[str]] = None
    ) -> List[AggregatedMetrics]:
        """
        Aggregate risk data for specified interval
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=1)
        
        async with self.db_pool.acquire() as conn:
            try:
                time_buckets = self._generate_time_buckets(start_date, end_date, interval)
                aggregated_results = []
                
                for bucket_start, bucket_end in time_buckets:
                    # Aggregate VaR data
                    var_agg_query = """
                        SELECT 
                            COUNT(*) as record_count,
                            AVG(var_amount) as avg_var_amount,
                            MAX(var_amount) as max_var_amount,
                            AVG(expected_shortfall) as avg_expected_shortfall,
                            AVG(confidence_level) as avg_confidence_level
                        FROM risk_var_calculations
                        WHERE calculation_timestamp >= $1 
                        AND calculation_timestamp < $2
                    """
                    
                    var_result = await conn.fetchrow(var_agg_query, bucket_start, bucket_end)
                    
                    # Aggregate exposure data
                    exposure_agg_query = """
                        SELECT 
                            COUNT(*) as record_count,
                            AVG(total_exposure) as avg_total_exposure,
                            AVG(net_exposure) as avg_net_exposure,
                            AVG(gross_exposure) as avg_gross_exposure,
                            AVG(long_exposure) as avg_long_exposure,
                            AVG(short_exposure) as avg_short_exposure
                        FROM risk_exposure_analysis
                        WHERE timestamp >= $1 
                        AND timestamp < $2
                    """
                    
                    exposure_result = await conn.fetchrow(exposure_agg_query, bucket_start, bucket_end)
                    
                    if (var_result and var_result['record_count'] > 0) or \
                       (exposure_result and exposure_result['record_count'] > 0):
                        
                        metrics = {
                            'var_metrics': {
                                'avg_var_amount': float(var_result['avg_var_amount'] or 0),
                                'max_var_amount': float(var_result['max_var_amount'] or 0),
                                'avg_expected_shortfall': float(var_result['avg_expected_shortfall'] or 0),
                                'avg_confidence_level': float(var_result['avg_confidence_level'] or 0),
                                'record_count': var_result['record_count'] or 0
                            },
                            'exposure_metrics': {
                                'avg_total_exposure': float(exposure_result['avg_total_exposure'] or 0),
                                'avg_net_exposure': float(exposure_result['avg_net_exposure'] or 0),
                                'avg_gross_exposure': float(exposure_result['avg_gross_exposure'] or 0),
                                'avg_long_exposure': float(exposure_result['avg_long_exposure'] or 0),
                                'avg_short_exposure': float(exposure_result['avg_short_exposure'] or 0),
                                'record_count': exposure_result['record_count'] or 0
                            },
                            'time_bucket_start': bucket_start.isoformat(),
                            'time_bucket_end': bucket_end.isoformat()
                        }
                        
                        # Compress and store
                        compressed_metrics, compression_ratio, storage_size = await self._compress_data(
                            metrics, CompressionLevel.MEDIUM
                        )
                        
                        aggregated_metric = AggregatedMetrics(
                            data_type=DataType.RISK,
                            interval=interval,
                            timestamp=bucket_start,
                            metrics=compressed_metrics,
                            record_count=(var_result['record_count'] or 0) + (exposure_result['record_count'] or 0),
                            compression_ratio=compression_ratio,
                            storage_size_bytes=storage_size
                        )
                        
                        await self._store_aggregated_data(conn, aggregated_metric)
                        aggregated_results.append(aggregated_metric)
                
                return aggregated_results
                
            except Exception as e:
                self.logger.error(f"Error aggregating risk data: {e}")
                raise
    
    async def aggregate_execution_data(
        self,
        interval: AggregationInterval,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        venue: Optional[str] = None
    ) -> List[AggregatedMetrics]:
        """
        Aggregate execution quality data
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=1)
        
        async with self.db_pool.acquire() as conn:
            try:
                time_buckets = self._generate_time_buckets(start_date, end_date, interval)
                aggregated_results = []
                
                for bucket_start, bucket_end in time_buckets:
                    # Build query filters
                    where_conditions = ["esa.execution_timestamp >= $1", "esa.execution_timestamp < $2"]
                    params = [bucket_start, bucket_end]
                    
                    if venue:
                        where_conditions.append("o.venue = $3")
                        params.append(venue)
                    
                    where_clause = " AND ".join(where_conditions)
                    
                    # Aggregate execution metrics
                    exec_agg_query = f"""
                        SELECT 
                            COUNT(*) as record_count,
                            AVG(esa.slippage_bps) as avg_slippage_bps,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY esa.slippage_bps) as median_slippage_bps,
                            AVG(esa.market_impact_bps) as avg_market_impact_bps,
                            SUM(esa.slippage_amount) as total_slippage_cost,
                            AVG(EXTRACT(EPOCH FROM (o.filled_at - o.created_at)) * 1000) as avg_execution_time_ms,
                            COUNT(CASE WHEN o.status = 'filled' THEN 1 END) as filled_orders,
                            COUNT(CASE WHEN ABS(esa.slippage_bps) <= 5 THEN 1 END) as good_executions
                        FROM execution_slippage_analysis esa
                        JOIN orders o ON esa.order_id = o.order_id
                        WHERE {where_clause}
                    """
                    
                    result = await conn.fetchrow(exec_agg_query, *params)
                    
                    if result and result['record_count'] > 0:
                        record_count = result['record_count']
                        filled_orders = result['filled_orders'] or 0
                        good_executions = result['good_executions'] or 0
                        
                        metrics = {
                            'avg_slippage_bps': float(result['avg_slippage_bps'] or 0),
                            'median_slippage_bps': float(result['median_slippage_bps'] or 0),
                            'avg_market_impact_bps': float(result['avg_market_impact_bps'] or 0),
                            'total_slippage_cost': float(result['total_slippage_cost'] or 0),
                            'avg_execution_time_ms': float(result['avg_execution_time_ms'] or 0),
                            'fill_rate': filled_orders / record_count if record_count > 0 else 0,
                            'good_execution_rate': good_executions / record_count if record_count > 0 else 0,
                            'total_orders': record_count,
                            'filled_orders': filled_orders,
                            'time_bucket_start': bucket_start.isoformat(),
                            'time_bucket_end': bucket_end.isoformat()
                        }
                        
                        if venue:
                            metrics['venue'] = venue
                        
                        # Compress and store
                        compressed_metrics, compression_ratio, storage_size = await self._compress_data(
                            metrics, CompressionLevel.LOW
                        )
                        
                        aggregated_metric = AggregatedMetrics(
                            data_type=DataType.EXECUTION,
                            interval=interval,
                            timestamp=bucket_start,
                            metrics=compressed_metrics,
                            record_count=record_count,
                            compression_ratio=compression_ratio,
                            storage_size_bytes=storage_size
                        )
                        
                        await self._store_aggregated_data(conn, aggregated_metric)
                        aggregated_results.append(aggregated_metric)
                
                return aggregated_results
                
            except Exception as e:
                self.logger.error(f"Error aggregating execution data: {e}")
                raise
    
    async def create_aggregation_job(
        self,
        job_config: AggregationJob
    ) -> str:
        """
        Create a scheduled aggregation job
        """
        async with self.db_pool.acquire() as conn:
            try:
                job_query = """
                    INSERT INTO aggregation_jobs (
                        job_id, data_type, interval, start_date, end_date,
                        filters, compression_level, auto_cleanup, retention_days,
                        status, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'pending', $10)
                    RETURNING job_id
                """
                
                result = await conn.fetchrow(
                    job_query,
                    job_config.job_id,
                    job_config.data_type.value,
                    job_config.interval.value,
                    job_config.start_date,
                    job_config.end_date,
                    json.dumps(job_config.filters),
                    job_config.compression_level.value,
                    job_config.auto_cleanup,
                    job_config.retention_days,
                    datetime.utcnow()
                )
                
                return result['job_id']
                
            except Exception as e:
                self.logger.error(f"Error creating aggregation job: {e}")
                raise
    
    async def run_aggregation_job(self, job_id: str) -> Dict[str, Any]:
        """
        Execute an aggregation job
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get job details
                job_query = """
                    SELECT * FROM aggregation_jobs
                    WHERE job_id = $1
                """
                
                job = await conn.fetchrow(job_query, job_id)
                if not job:
                    raise ValueError(f"Job {job_id} not found")
                
                # Update job status
                await conn.execute(
                    "UPDATE aggregation_jobs SET status = 'running', started_at = $1 WHERE job_id = $2",
                    datetime.utcnow(), job_id
                )
                
                # Parse job configuration
                data_type = DataType(job['data_type'])
                interval = AggregationInterval(job['interval'])
                filters = json.loads(job['filters']) if job['filters'] else {}
                
                # Run appropriate aggregation
                results = []
                if data_type == DataType.PERFORMANCE:
                    results = await self.aggregate_performance_data(
                        interval, job['start_date'], job['end_date'],
                        filters.get('portfolio_ids')
                    )
                elif data_type == DataType.RISK:
                    results = await self.aggregate_risk_data(
                        interval, job['start_date'], job['end_date'],
                        filters.get('portfolio_ids')
                    )
                elif data_type == DataType.EXECUTION:
                    results = await self.aggregate_execution_data(
                        interval, job['start_date'], job['end_date'],
                        filters.get('venue')
                    )
                
                # Update job completion
                await conn.execute(
                    """UPDATE aggregation_jobs 
                       SET status = 'completed', completed_at = $1, 
                           records_processed = $2, result_summary = $3
                       WHERE job_id = $4""",
                    datetime.utcnow(),
                    len(results),
                    json.dumps({'total_records': len(results)}),
                    job_id
                )
                
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'records_processed': len(results),
                    'results': [
                        {
                            'timestamp': r.timestamp.isoformat(),
                            'record_count': r.record_count,
                            'compression_ratio': r.compression_ratio,
                            'storage_size_bytes': r.storage_size_bytes
                        }
                        for r in results[:10]  # Limit to first 10 for summary
                    ]
                }
                
            except Exception as e:
                # Update job with error status
                await conn.execute(
                    "UPDATE aggregation_jobs SET status = 'failed', error_message = $1 WHERE job_id = $2",
                    str(e), job_id
                )
                self.logger.error(f"Error running aggregation job {job_id}: {e}")
                raise
    
    async def optimize_query_performance(
        self,
        query: str,
        params: List[Any]
    ) -> QueryOptimization:
        """
        Analyze and optimize query performance
        """
        async with self.db_pool.acquire() as conn:
            try:
                start_time = datetime.utcnow()
                
                # Execute query with EXPLAIN ANALYZE
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                explain_result = await conn.fetchrow(explain_query, *params)
                
                end_time = datetime.utcnow()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                # Parse explain plan
                plan_data = explain_result[0][0] if explain_result else {}
                
                # Extract performance metrics
                rows_examined = plan_data.get('Plan', {}).get('Actual Rows', 0)
                rows_returned = rows_examined  # Simplified
                
                # Identify index usage
                index_usage = self._extract_index_usage(plan_data)
                
                # Generate optimization suggestions
                suggestions = self._generate_optimization_suggestions(plan_data, execution_time_ms)
                
                # Generate query hash for caching
                query_hash = str(hash(query + str(params)))
                
                optimization = QueryOptimization(
                    query_hash=query_hash,
                    query_pattern=self._extract_query_pattern(query),
                    execution_time_ms=execution_time_ms,
                    rows_examined=rows_examined,
                    rows_returned=rows_returned,
                    index_usage=index_usage,
                    optimization_suggestions=suggestions,
                    cache_hit=False
                )
                
                # Store optimization data
                await self._store_query_optimization(conn, optimization)
                
                return optimization
                
            except Exception as e:
                self.logger.error(f"Error optimizing query performance: {e}")
                raise
    
    async def cleanup_old_data(
        self,
        data_type: Optional[DataType] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old aggregated data based on retention policies
        """
        async with self.db_pool.acquire() as conn:
            try:
                cleanup_summary = {}
                
                for interval, retention_days in self.retention_policies.items():
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    # Build query conditions
                    where_conditions = ["timestamp < $1"]
                    params = [cutoff_date]
                    
                    if data_type:
                        where_conditions.append("data_type = $2")
                        params.append(data_type.value)
                    
                    where_clause = " AND ".join(where_conditions)
                    
                    if dry_run:
                        # Count records that would be deleted
                        count_query = f"""
                            SELECT COUNT(*) as count
                            FROM aggregated_analytics
                            WHERE {where_clause} AND interval = $3
                        """
                        params.append(interval.value)
                        
                        result = await conn.fetchrow(count_query, *params)
                        cleanup_summary[interval.value] = {
                            'records_to_delete': result['count'],
                            'cutoff_date': cutoff_date.isoformat(),
                            'retention_days': retention_days
                        }
                    else:
                        # Actually delete the records
                        delete_query = f"""
                            DELETE FROM aggregated_analytics
                            WHERE {where_clause} AND interval = $3
                        """
                        params.append(interval.value)
                        
                        result = await conn.execute(delete_query, *params)
                        deleted_count = int(result.split()[-1]) if result else 0
                        
                        cleanup_summary[interval.value] = {
                            'records_deleted': deleted_count,
                            'cutoff_date': cutoff_date.isoformat(),
                            'retention_days': retention_days
                        }
                
                return {
                    'cleanup_type': 'dry_run' if dry_run else 'actual',
                    'timestamp': datetime.utcnow().isoformat(),
                    'summary': cleanup_summary
                }
                
            except Exception as e:
                self.logger.error(f"Error cleaning up old data: {e}")
                raise
    
    # Helper methods
    
    def _generate_time_buckets(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: AggregationInterval
    ) -> List[Tuple[datetime, datetime]]:
        """Generate time buckets for aggregation"""
        buckets = []
        
        # Determine bucket size
        if interval == AggregationInterval.MINUTE_1:
            delta = timedelta(minutes=1)
        elif interval == AggregationInterval.MINUTE_5:
            delta = timedelta(minutes=5)
        elif interval == AggregationInterval.MINUTE_15:
            delta = timedelta(minutes=15)
        elif interval == AggregationInterval.HOUR_1:
            delta = timedelta(hours=1)
        elif interval == AggregationInterval.HOUR_4:
            delta = timedelta(hours=4)
        elif interval == AggregationInterval.DAY_1:
            delta = timedelta(days=1)
        elif interval == AggregationInterval.WEEK_1:
            delta = timedelta(weeks=1)
        elif interval == AggregationInterval.MONTH_1:
            delta = timedelta(days=30)  # Approximate
        else:
            delta = timedelta(hours=1)  # Default
        
        current = start_date
        while current < end_date:
            bucket_end = min(current + delta, end_date)
            buckets.append((current, bucket_end))
            current = bucket_end
        
        return buckets
    
    async def _compress_data(
        self,
        data: Dict[str, Any],
        compression_level: CompressionLevel
    ) -> Tuple[Dict[str, Any], Optional[float], int]:
        """Compress data and return compression metrics"""
        if compression_level == CompressionLevel.NONE:
            data_str = json.dumps(data)
            return data, None, len(data_str.encode('utf-8'))
        
        # Run compression in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            self._compress_data_sync,
            data,
            compression_level
        )
        
        return result
    
    def _compress_data_sync(
        self,
        data: Dict[str, Any],
        compression_level: CompressionLevel
    ) -> Tuple[Dict[str, Any], float, int]:
        """Synchronous data compression"""
        try:
            # Serialize data
            original_data = pickle.dumps(data)
            original_size = len(original_data)
            
            # Compress data
            compressed_data = gzip.compress(original_data, compresslevel=compression_level.value)
            compressed_size = len(compressed_data)
            
            # Calculate compression ratio
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Return compressed data as base64 for JSON storage
            import base64
            compressed_b64 = base64.b64encode(compressed_data).decode('utf-8')
            
            return {
                'compressed': True,
                'compression_level': compression_level.value,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'data': compressed_b64
            }, compression_ratio, compressed_size
            
        except Exception as e:
            self.logger.warning(f"Compression failed, storing uncompressed: {e}")
            data_str = json.dumps(data)
            return data, None, len(data_str.encode('utf-8'))
    
    def _extract_index_usage(self, plan_data: Dict[str, Any]) -> List[str]:
        """Extract index usage from query plan"""
        indexes = []
        
        def traverse_plan(node):
            if isinstance(node, dict):
                if 'Index Name' in node:
                    indexes.append(node['Index Name'])
                if 'Plans' in node:
                    for child in node['Plans']:
                        traverse_plan(child)
        
        if 'Plan' in plan_data:
            traverse_plan(plan_data['Plan'])
        
        return list(set(indexes))
    
    def _generate_optimization_suggestions(
        self,
        plan_data: Dict[str, Any],
        execution_time_ms: float
    ) -> List[str]:
        """Generate query optimization suggestions"""
        suggestions = []
        
        # Analyze execution time
        if execution_time_ms > 1000:  # Slow query
            suggestions.append("Consider adding appropriate indexes")
            suggestions.append("Review WHERE clause conditions")
        
        # Analyze plan for sequential scans
        plan_str = json.dumps(plan_data).lower()
        if 'seq scan' in plan_str:
            suggestions.append("Sequential scan detected - consider adding index")
        
        # Analyze for sorting
        if 'sort' in plan_str and 'memory' in plan_str:
            suggestions.append("Large sort operation - consider increasing work_mem")
        
        # Analyze for hash joins
        if 'hash join' in plan_str:
            suggestions.append("Hash join detected - ensure adequate memory allocation")
        
        return suggestions
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for caching"""
        # Normalize query by removing specific values
        import re
        
        # Remove string literals
        pattern = re.sub(r"'[^']*'", "'?'", query)
        
        # Remove numeric literals
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        
        # Remove extra whitespace
        pattern = re.sub(r'\s+', ' ', pattern).strip()
        
        return pattern.lower()
    
    # Storage methods
    
    async def _store_aggregated_data(
        self,
        conn: asyncpg.Connection,
        aggregated_metric: AggregatedMetrics
    ):
        """Store aggregated metrics data"""
        query = """
            INSERT INTO aggregated_analytics (
                data_type, interval, timestamp, metrics, record_count,
                compression_ratio, storage_size_bytes, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (data_type, interval, timestamp) DO UPDATE SET
                metrics = EXCLUDED.metrics,
                record_count = EXCLUDED.record_count,
                compression_ratio = EXCLUDED.compression_ratio,
                storage_size_bytes = EXCLUDED.storage_size_bytes,
                created_at = EXCLUDED.created_at
        """
        
        await conn.execute(
            query,
            aggregated_metric.data_type.value,
            aggregated_metric.interval.value,
            aggregated_metric.timestamp,
            json.dumps(aggregated_metric.metrics),
            aggregated_metric.record_count,
            aggregated_metric.compression_ratio,
            aggregated_metric.storage_size_bytes,
            datetime.utcnow()
        )
    
    async def _store_query_optimization(
        self,
        conn: asyncpg.Connection,
        optimization: QueryOptimization
    ):
        """Store query optimization data"""
        query = """
            INSERT INTO query_optimizations (
                query_hash, query_pattern, execution_time_ms, rows_examined,
                rows_returned, index_usage, optimization_suggestions,
                cache_hit, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (query_hash) DO UPDATE SET
                execution_time_ms = EXCLUDED.execution_time_ms,
                rows_examined = EXCLUDED.rows_examined,
                rows_returned = EXCLUDED.rows_returned,
                index_usage = EXCLUDED.index_usage,
                optimization_suggestions = EXCLUDED.optimization_suggestions,
                cache_hit = EXCLUDED.cache_hit,
                created_at = EXCLUDED.created_at
        """
        
        await conn.execute(
            query,
            optimization.query_hash,
            optimization.query_pattern,
            optimization.execution_time_ms,
            optimization.rows_examined,
            optimization.rows_returned,
            json.dumps(optimization.index_usage),
            json.dumps(optimization.optimization_suggestions),
            optimization.cache_hit,
            datetime.utcnow()
        )
    
    async def get_aggregated_data(
        self,
        data_type: DataType,
        interval: AggregationInterval,
        start_date: datetime,
        end_date: datetime,
        decompress: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve aggregated data for analysis"""
        async with self.db_pool.acquire() as conn:
            try:
                query = """
                    SELECT timestamp, metrics, record_count, compression_ratio
                    FROM aggregated_analytics
                    WHERE data_type = $1 
                    AND interval = $2
                    AND timestamp >= $3
                    AND timestamp <= $4
                    ORDER BY timestamp
                """
                
                results = await conn.fetch(query, data_type.value, interval.value, start_date, end_date)
                
                aggregated_data = []
                for result in results:
                    metrics_data = json.loads(result['metrics'])
                    
                    # Decompress if needed
                    if decompress and isinstance(metrics_data, dict) and metrics_data.get('compressed'):
                        try:
                            decompressed_data = await self._decompress_data(metrics_data)
                            metrics_data = decompressed_data
                        except Exception as e:
                            self.logger.warning(f"Failed to decompress data: {e}")
                    
                    aggregated_data.append({
                        'timestamp': result['timestamp'],
                        'metrics': metrics_data,
                        'record_count': result['record_count'],
                        'compression_ratio': result['compression_ratio']
                    })
                
                return aggregated_data
                
            except Exception as e:
                self.logger.error(f"Error retrieving aggregated data: {e}")
                raise
    
    async def _decompress_data(self, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress data"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._decompress_data_sync,
            compressed_data
        )
    
    def _decompress_data_sync(self, compressed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous data decompression"""
        import base64
        
        if not compressed_data.get('compressed'):
            return compressed_data
        
        try:
            # Decode base64 data
            compressed_bytes = base64.b64decode(compressed_data['data'])
            
            # Decompress
            decompressed_bytes = gzip.decompress(compressed_bytes)
            
            # Deserialize
            return pickle.loads(decompressed_bytes)
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return compressed_data

# Global instance
analytics_aggregator = None

def get_analytics_aggregator() -> AnalyticsAggregator:
    """Get global analytics aggregator instance"""
    global analytics_aggregator
    if analytics_aggregator is None:
        raise RuntimeError("Analytics aggregator not initialized. Call init_analytics_aggregator() first.")
    return analytics_aggregator

def init_analytics_aggregator(db_pool: asyncpg.Pool) -> AnalyticsAggregator:
    """Initialize global analytics aggregator instance"""
    global analytics_aggregator
    analytics_aggregator = AnalyticsAggregator(db_pool)
    return analytics_aggregator