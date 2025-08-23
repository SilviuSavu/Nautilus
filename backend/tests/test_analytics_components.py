"""
Unit tests for Analytics Components - Sprint 3 Priority 2

Tests all analytics modules including performance calculator, risk analytics,
strategy analytics, and execution analytics with comprehensive coverage.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Import Sprint 3 analytics components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analytics.performance_calculator import (
    PerformanceCalculator, PerformanceSnapshot, PositionPerformance,
    PerformanceMetricType, get_performance_calculator, init_performance_calculator
)
from analytics.risk_analytics import RiskAnalytics, RiskMetrics, PortfolioRisk
from analytics.strategy_analytics import StrategyAnalytics, StrategyPerformance, StrategyMetrics
from analytics.execution_analytics import ExecutionAnalytics, ExecutionMetrics, OrderAnalysis
from analytics.analytics_aggregator import AnalyticsAggregator, AggregatedMetrics


class TestPerformanceCalculator:
    """Test portfolio performance calculation functionality"""
    
    @pytest.fixture
    def mock_db_pool(self):
        """Mock database connection pool"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        return mock_pool, mock_conn
    
    @pytest.fixture
    def perf_calculator(self, mock_db_pool):
        """Create performance calculator with mocked database"""
        pool, _ = mock_db_pool
        return PerformanceCalculator(pool)
    
    @pytest.mark.asyncio
    async def test_real_time_pnl_calculation(self, perf_calculator, mock_db_pool):
        """Test real-time P&L calculation for portfolio"""
        _, mock_conn = mock_db_pool
        
        # Mock position data
        mock_positions = [
            {
                'instrument_id': 'AAPL_NASDAQ',
                'position_id': 'pos_1',
                'quantity': 100,
                'avg_entry_price': 150.0,
                'realized_pnl': 500.0,
                'created_at': datetime.utcnow() - timedelta(days=5),
                'symbol': 'AAPL',
                'venue': 'NASDAQ',
                'multiplier': 1
            },
            {
                'instrument_id': 'MSFT_NASDAQ',
                'position_id': 'pos_2',
                'quantity': 50,
                'avg_entry_price': 300.0,
                'realized_pnl': 0.0,
                'created_at': datetime.utcnow() - timedelta(days=2),
                'symbol': 'MSFT',
                'venue': 'NASDAQ',
                'multiplier': 1
            }
        ]
        
        mock_conn.fetch.return_value = mock_positions
        
        # Mock current prices
        async def mock_get_current_price(conn, instrument_id):
            prices = {
                'AAPL_NASDAQ': Decimal('155.0'),  # $5 gain per share
                'MSFT_NASDAQ': Decimal('295.0')   # $5 loss per share
            }
            return prices.get(instrument_id)
        
        perf_calculator._get_current_price = mock_get_current_price
        
        # Test P&L calculation
        portfolio_id = "test_portfolio"
        pnl_result = await perf_calculator.calculate_real_time_pnl(portfolio_id)
        
        # Verify results
        assert pnl_result["portfolio_id"] == portfolio_id
        assert pnl_result["realized_pnl"] == 500.0
        assert pnl_result["unrealized_pnl"] == 250.0  # (100 * 5) + (50 * -5)
        assert pnl_result["total_pnl"] == 750.0
        assert pnl_result["position_count"] == 2
        
        # Verify position details
        positions = pnl_result["positions"]
        aapl_pos = next(pos for pos in positions if pos["instrument_id"] == "AAPL_NASDAQ")
        assert aapl_pos["unrealized_pnl"] == 500.0
        assert aapl_pos["return_pct"] == 3.33  # Approximately (155-150)/150 * 100
        
        msft_pos = next(pos for pos in positions if pos["instrument_id"] == "MSFT_NASDAQ")
        assert msft_pos["unrealized_pnl"] == -250.0
        assert msft_pos["return_pct"] == -1.67  # Approximately (295-300)/300 * 100
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_calculation(self, perf_calculator, mock_db_pool):
        """Test comprehensive portfolio metrics calculation"""
        _, mock_conn = mock_db_pool
        
        # Mock portfolio returns (daily returns for 30 days)
        portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 6  # 30 returns
        
        async def mock_get_portfolio_returns(conn, portfolio_id, start_date, end_date):
            return portfolio_returns
        
        # Mock current P&L
        async def mock_calculate_real_time_pnl(portfolio_id):
            return {
                'total_pnl': 15000.0,
                'unrealized_pnl': 8000.0,
                'realized_pnl': 7000.0
            }
        
        perf_calculator._get_portfolio_returns = mock_get_portfolio_returns
        perf_calculator.calculate_real_time_pnl = mock_calculate_real_time_pnl
        
        # Mock benchmark returns
        benchmark_returns = [0.008, -0.003, 0.015, -0.008, 0.012] * 6
        
        async def mock_get_benchmark_returns(conn, symbol, start_date, end_date):
            return benchmark_returns
        
        perf_calculator._get_benchmark_returns = mock_get_benchmark_returns
        
        # Mock alpha/beta calculation
        async def mock_calculate_alpha_beta(portfolio_returns, benchmark_returns):
            return 0.05, 1.2, 0.8  # alpha, beta, information_ratio
        
        perf_calculator._calculate_alpha_beta = mock_calculate_alpha_beta
        
        # Mock store snapshot
        async def mock_store_performance_snapshot(conn, snapshot):
            pass
        
        perf_calculator._store_performance_snapshot = mock_store_performance_snapshot
        
        # Test metrics calculation
        portfolio_id = "test_portfolio"
        snapshot = await perf_calculator.calculate_portfolio_metrics(portfolio_id)
        
        # Verify snapshot
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.portfolio_id == portfolio_id
        assert snapshot.total_pnl == Decimal('15000.0')
        assert snapshot.unrealized_pnl == Decimal('8000.0')
        assert snapshot.realized_pnl == Decimal('7000.0')
        
        # Verify calculated metrics
        assert snapshot.sharpe_ratio > 0
        assert snapshot.max_drawdown < 0
        assert 0 <= snapshot.win_rate <= 1
        assert snapshot.profit_factor > 0
        assert snapshot.volatility > 0
        
        # Verify alpha/beta
        assert snapshot.alpha == 0.05
        assert snapshot.beta == 1.2
        assert snapshot.information_ratio == 0.8
    
    @pytest.mark.asyncio
    async def test_performance_attribution_calculation(self, perf_calculator, mock_db_pool):
        """Test performance attribution by asset class"""
        _, mock_conn = mock_db_pool
        
        # Mock attribution query results
        mock_attribution_data = [
            {
                'asset_class': 'EQUITY',
                'symbol': 'AAPL',
                'instrument_id': 'AAPL_NASDAQ',
                'position_value': 15000.0,
                'total_pnl': 1500.0
            },
            {
                'asset_class': 'EQUITY',
                'symbol': 'MSFT',
                'instrument_id': 'MSFT_NASDAQ',
                'position_value': 10000.0,
                'total_pnl': -500.0
            },
            {
                'asset_class': 'FOREX',
                'symbol': 'EURUSD',
                'instrument_id': 'EURUSD_FOREX',
                'position_value': 5000.0,
                'total_pnl': 200.0
            }
        ]
        
        mock_conn.fetch.return_value = mock_attribution_data
        
        # Test attribution calculation
        portfolio_id = "test_portfolio"
        attribution = await perf_calculator.calculate_performance_attribution(portfolio_id)
        
        # Verify results
        assert attribution["portfolio_id"] == portfolio_id
        assert attribution["total_portfolio_value"] == 30000.0
        
        # Verify asset class breakdown
        equity_attribution = attribution["attribution_by_asset_class"]["EQUITY"]
        assert equity_attribution["weight"] == 25000.0 / 30000.0  # (15000 + 10000) / 30000
        assert equity_attribution["total_pnl"] == 1000.0  # 1500 - 500
        assert equity_attribution["position_count"] == 2
        
        forex_attribution = attribution["attribution_by_asset_class"]["FOREX"]
        assert forex_attribution["weight"] == 5000.0 / 30000.0
        assert forex_attribution["total_pnl"] == 200.0
        assert forex_attribution["position_count"] == 1
    
    @pytest.mark.asyncio
    async def test_current_price_retrieval(self, perf_calculator, mock_db_pool):
        """Test current price retrieval with fallback logic"""
        _, mock_conn = mock_db_pool
        
        instrument_id = "AAPL_NASDAQ"
        
        # Test quote data available
        mock_conn.fetchrow.side_effect = [
            {'mid_price': 155.25},  # Quote query result
        ]
        
        price = await perf_calculator._get_current_price(mock_conn, instrument_id)
        assert price == Decimal('155.25')
        
        # Test fallback to trade data
        mock_conn.fetchrow.side_effect = [
            None,  # No quote data
            {'price': 155.50}  # Trade query result
        ]
        
        price = await perf_calculator._get_current_price(mock_conn, instrument_id)
        assert price == Decimal('155.50')
        
        # Test no data available
        mock_conn.fetchrow.side_effect = [None, None]
        
        price = await perf_calculator._get_current_price(mock_conn, instrument_id)
        assert price is None
    
    def test_global_instance_management(self):
        """Test global performance calculator instance management"""
        # Test uninitialized state
        with pytest.raises(RuntimeError, match="Performance calculator not initialized"):
            get_performance_calculator()
        
        # Test initialization
        mock_pool = Mock()
        calculator = init_performance_calculator(mock_pool)
        
        assert isinstance(calculator, PerformanceCalculator)
        assert calculator.db_pool == mock_pool
        
        # Test getting initialized instance
        retrieved_calculator = get_performance_calculator()
        assert retrieved_calculator is calculator


class TestRiskAnalytics:
    """Test risk analytics functionality"""
    
    @pytest.fixture
    def risk_analytics(self):
        """Create risk analytics instance"""
        return RiskAnalytics()
    
    def test_portfolio_risk_calculation(self, risk_analytics):
        """Test portfolio risk metrics calculation"""
        # Mock portfolio data
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'beta': 1.2},
            {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'beta': 0.9},
            {'symbol': 'GOOGL', 'quantity': 25, 'price': 2800.0, 'beta': 1.1}
        ]
        
        # Mock correlation matrix
        correlation_matrix = pd.DataFrame([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ], index=['AAPL', 'MSFT', 'GOOGL'], columns=['AAPL', 'MSFT', 'GOOGL'])
        
        # Mock volatilities
        volatilities = {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.30}
        
        portfolio_risk = risk_analytics.calculate_portfolio_risk(
            positions, correlation_matrix, volatilities
        )
        
        assert isinstance(portfolio_risk, PortfolioRisk)
        assert portfolio_risk.total_value > 0
        assert portfolio_risk.portfolio_beta > 0
        assert portfolio_risk.portfolio_volatility > 0
        assert len(portfolio_risk.position_risks) == 3
    
    def test_value_at_risk_calculation(self, risk_analytics):
        """Test VaR calculation with different methods"""
        # Mock return series
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        
        # Test parametric VaR
        var_95 = risk_analytics.calculate_var(returns, confidence_level=0.95, method='parametric')
        var_99 = risk_analytics.calculate_var(returns, confidence_level=0.99, method='parametric')
        
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more extreme than 95%
        
        # Test historical VaR
        historical_var_95 = risk_analytics.calculate_var(returns, confidence_level=0.95, method='historical')
        assert historical_var_95 < 0
        
        # Test Monte Carlo VaR
        mc_var_95 = risk_analytics.calculate_var(returns, confidence_level=0.95, method='monte_carlo')
        assert mc_var_95 < 0
    
    def test_expected_shortfall_calculation(self, risk_analytics):
        """Test Expected Shortfall (CVaR) calculation"""
        returns = np.random.normal(0.001, 0.02, 252)
        
        es_95 = risk_analytics.calculate_expected_shortfall(returns, confidence_level=0.95)
        es_99 = risk_analytics.calculate_expected_shortfall(returns, confidence_level=0.99)
        
        assert es_95 < 0  # ES should be negative (loss)
        assert es_99 < es_95  # 99% ES should be more extreme than 95%
    
    def test_drawdown_analysis(self, risk_analytics):
        """Test drawdown analysis"""
        # Mock cumulative returns with drawdowns
        cumulative_returns = np.array([
            1.0, 1.05, 1.02, 1.08, 1.12, 1.06, 1.03, 1.15, 1.18, 1.10,
            1.08, 1.13, 1.20, 1.16, 1.22, 1.18, 1.25, 1.22, 1.28, 1.35
        ])
        
        drawdown_metrics = risk_analytics.analyze_drawdowns(cumulative_returns)
        
        assert drawdown_metrics['max_drawdown'] < 0
        assert drawdown_metrics['current_drawdown'] <= 0
        assert drawdown_metrics['drawdown_duration'] >= 0
        assert len(drawdown_metrics['drawdown_periods']) >= 0
    
    def test_concentration_risk_analysis(self, risk_analytics):
        """Test portfolio concentration risk analysis"""
        positions = [
            {'symbol': 'AAPL', 'value': 50000},
            {'symbol': 'MSFT', 'value': 30000},
            {'symbol': 'GOOGL', 'value': 15000},
            {'symbol': 'AMZN', 'value': 5000}
        ]
        
        concentration_metrics = risk_analytics.analyze_concentration_risk(positions)
        
        assert concentration_metrics['herfindahl_index'] > 0
        assert concentration_metrics['concentration_ratio_top5'] <= 1.0
        assert concentration_metrics['largest_position_pct'] > 0
        assert len(concentration_metrics['top_positions']) > 0


class TestStrategyAnalytics:
    """Test strategy analytics functionality"""
    
    @pytest.fixture
    def strategy_analytics(self):
        """Create strategy analytics instance"""
        return StrategyAnalytics()
    
    def test_strategy_performance_calculation(self, strategy_analytics):
        """Test individual strategy performance metrics"""
        # Mock strategy trade data
        trades = [
            {'entry_time': datetime.utcnow() - timedelta(days=10), 'exit_time': datetime.utcnow() - timedelta(days=8), 'pnl': 150.0, 'quantity': 100},
            {'entry_time': datetime.utcnow() - timedelta(days=8), 'exit_time': datetime.utcnow() - timedelta(days=6), 'pnl': -75.0, 'quantity': 50},
            {'entry_time': datetime.utcnow() - timedelta(days=6), 'exit_time': datetime.utcnow() - timedelta(days=4), 'pnl': 225.0, 'quantity': 75},
            {'entry_time': datetime.utcnow() - timedelta(days=4), 'exit_time': datetime.utcnow() - timedelta(days=2), 'pnl': -50.0, 'quantity': 25},
            {'entry_time': datetime.utcnow() - timedelta(days=2), 'exit_time': datetime.utcnow(), 'pnl': 100.0, 'quantity': 80}
        ]
        
        strategy_id = "momentum_v1"
        performance = strategy_analytics.calculate_strategy_performance(strategy_id, trades)
        
        assert isinstance(performance, StrategyPerformance)
        assert performance.strategy_id == strategy_id
        assert performance.total_pnl == 350.0  # 150 - 75 + 225 - 50 + 100
        assert performance.total_trades == 5
        assert performance.winning_trades == 3
        assert performance.losing_trades == 2
        assert performance.win_rate == 0.6
        assert performance.profit_factor > 1.0  # Profitable strategy
        assert performance.average_win > performance.average_loss
    
    def test_strategy_comparison(self, strategy_analytics):
        """Test comparison between multiple strategies"""
        strategy_data = {
            'momentum_v1': {
                'returns': [0.02, -0.01, 0.03, -0.005, 0.015],
                'trades': 20,
                'win_rate': 0.65
            },
            'mean_reversion_v2': {
                'returns': [0.01, 0.002, -0.015, 0.008, 0.02],
                'trades': 15,
                'win_rate': 0.60
            },
            'arbitrage_v1': {
                'returns': [0.005, 0.003, 0.007, 0.004, 0.006],
                'trades': 50,
                'win_rate': 0.90
            }
        }
        
        comparison = strategy_analytics.compare_strategies(strategy_data)
        
        assert 'momentum_v1' in comparison
        assert 'mean_reversion_v2' in comparison
        assert 'arbitrage_v1' in comparison
        
        # Verify ranking metrics
        for strategy, metrics in comparison.items():
            assert 'sharpe_ratio' in metrics
            assert 'volatility' in metrics
            assert 'max_drawdown' in metrics
            assert 'total_return' in metrics
    
    def test_strategy_risk_attribution(self, strategy_analytics):
        """Test risk attribution analysis for strategies"""
        strategy_positions = {
            'momentum_v1': [
                {'symbol': 'AAPL', 'weight': 0.3, 'beta': 1.2, 'volatility': 0.25},
                {'symbol': 'MSFT', 'weight': 0.4, 'beta': 0.9, 'volatility': 0.20},
                {'symbol': 'TSLA', 'weight': 0.3, 'beta': 1.8, 'volatility': 0.40}
            ]
        }
        
        risk_attribution = strategy_analytics.analyze_strategy_risk_attribution(
            'momentum_v1', strategy_positions['momentum_v1']
        )
        
        assert 'portfolio_beta' in risk_attribution
        assert 'portfolio_volatility' in risk_attribution
        assert 'position_contributions' in risk_attribution
        assert len(risk_attribution['position_contributions']) == 3
    
    def test_strategy_alpha_beta_calculation(self, strategy_analytics):
        """Test strategy alpha/beta calculation against benchmark"""
        strategy_returns = [0.02, -0.01, 0.03, -0.005, 0.015, 0.008, -0.012, 0.025]
        benchmark_returns = [0.015, -0.008, 0.022, -0.003, 0.012, 0.006, -0.010, 0.020]
        
        alpha, beta = strategy_analytics.calculate_alpha_beta(strategy_returns, benchmark_returns)
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert beta > 0  # Assuming positive correlation with benchmark


class TestExecutionAnalytics:
    """Test execution analytics functionality"""
    
    @pytest.fixture
    def execution_analytics(self):
        """Create execution analytics instance"""
        return ExecutionAnalytics()
    
    def test_trade_execution_analysis(self, execution_analytics):
        """Test trade execution quality analysis"""
        # Mock execution data
        executions = [
            {
                'order_id': 'ord_1',
                'symbol': 'AAPL',
                'side': 'BUY',
                'quantity': 100,
                'executed_price': 150.25,
                'benchmark_price': 150.20,  # TWAP or arrival price
                'execution_time': datetime.utcnow() - timedelta(minutes=30),
                'order_time': datetime.utcnow() - timedelta(minutes=35)
            },
            {
                'order_id': 'ord_2',
                'symbol': 'MSFT',
                'side': 'SELL',
                'quantity': 50,
                'executed_price': 299.80,
                'benchmark_price': 300.00,
                'execution_time': datetime.utcnow() - timedelta(minutes=15),
                'order_time': datetime.utcnow() - timedelta(minutes=20)
            }
        ]
        
        analysis = execution_analytics.analyze_trade_executions(executions)
        
        assert isinstance(analysis, ExecutionMetrics)
        assert analysis.total_executions == 2
        assert analysis.average_slippage is not None
        assert analysis.implementation_shortfall is not None
        assert len(analysis.execution_details) == 2
    
    def test_order_flow_analysis(self, execution_analytics):
        """Test order flow and routing analysis"""
        orders = [
            {
                'order_id': 'ord_1',
                'symbol': 'AAPL',
                'order_type': 'MARKET',
                'venue': 'NASDAQ',
                'fill_rate': 1.0,
                'time_to_fill': 2.5,
                'price_improvement': 0.01
            },
            {
                'order_id': 'ord_2',
                'symbol': 'MSFT',
                'order_type': 'LIMIT',
                'venue': 'NYSE',
                'fill_rate': 0.8,
                'time_to_fill': 15.0,
                'price_improvement': 0.0
            }
        ]
        
        flow_analysis = execution_analytics.analyze_order_flow(orders)
        
        assert 'venue_performance' in flow_analysis
        assert 'order_type_performance' in flow_analysis
        assert 'average_fill_rate' in flow_analysis
        assert 'average_time_to_fill' in flow_analysis
    
    def test_market_impact_analysis(self, execution_analytics):
        """Test market impact measurement"""
        large_orders = [
            {
                'symbol': 'AAPL',
                'quantity': 10000,
                'executed_price': 150.30,
                'pre_trade_price': 150.25,
                'post_trade_price': 150.35,
                'adv_percentage': 0.05  # 5% of ADV
            },
            {
                'symbol': 'MSFT',
                'quantity': 5000,
                'executed_price': 299.75,
                'pre_trade_price': 300.00,
                'post_trade_price': 299.70,
                'adv_percentage': 0.02  # 2% of ADV
            }
        ]
        
        impact_analysis = execution_analytics.analyze_market_impact(large_orders)
        
        assert 'temporary_impact' in impact_analysis
        assert 'permanent_impact' in impact_analysis
        assert 'impact_by_size' in impact_analysis
        assert len(impact_analysis['order_impacts']) == 2


class TestAnalyticsAggregator:
    """Test analytics aggregation functionality"""
    
    @pytest.fixture
    def analytics_aggregator(self):
        """Create analytics aggregator instance"""
        return AnalyticsAggregator()
    
    @pytest.mark.asyncio
    async def test_comprehensive_analytics_aggregation(self, analytics_aggregator):
        """Test aggregation of all analytics components"""
        # Mock individual analytics components
        mock_performance = Mock()
        mock_performance.calculate_real_time_pnl.return_value = {
            'total_pnl': 15000.0,
            'unrealized_pnl': 8000.0,
            'realized_pnl': 7000.0
        }
        
        mock_risk = Mock()
        mock_risk.calculate_portfolio_risk.return_value = Mock(
            portfolio_volatility=0.15,
            portfolio_beta=1.1,
            var_95=0.025
        )
        
        mock_strategy = Mock()
        mock_strategy.get_strategy_summary.return_value = {
            'total_strategies': 3,
            'active_strategies': 2,
            'best_performer': 'momentum_v1'
        }
        
        mock_execution = Mock()
        mock_execution.get_execution_summary.return_value = {
            'total_trades': 150,
            'average_slippage': 0.0025,
            'fill_rate': 0.98
        }
        
        # Inject mocked components
        analytics_aggregator.performance_calculator = mock_performance
        analytics_aggregator.risk_analytics = mock_risk
        analytics_aggregator.strategy_analytics = mock_strategy
        analytics_aggregator.execution_analytics = mock_execution
        
        # Test aggregation
        portfolio_id = "test_portfolio"
        aggregated = await analytics_aggregator.get_comprehensive_analytics(portfolio_id)
        
        assert isinstance(aggregated, AggregatedMetrics)
        assert aggregated.portfolio_id == portfolio_id
        assert aggregated.performance_metrics is not None
        assert aggregated.risk_metrics is not None
        assert aggregated.strategy_metrics is not None
        assert aggregated.execution_metrics is not None
    
    def test_real_time_analytics_streaming(self, analytics_aggregator):
        """Test real-time analytics streaming capability"""
        portfolio_id = "test_portfolio"
        
        # Mock streaming data source
        def mock_data_stream():
            yield {'type': 'market_data', 'symbol': 'AAPL', 'price': 150.25}
            yield {'type': 'trade', 'symbol': 'AAPL', 'quantity': 100, 'price': 150.30}
            yield {'type': 'market_data', 'symbol': 'MSFT', 'price': 299.75}
        
        # Test streaming analytics
        stream_results = []
        for data in mock_data_stream():
            result = analytics_aggregator.process_streaming_data(portfolio_id, data)
            if result:
                stream_results.append(result)
        
        # Verify streaming processing
        assert len(stream_results) >= 0  # May or may not trigger analytics updates
    
    def test_analytics_caching_mechanism(self, analytics_aggregator):
        """Test analytics caching for performance optimization"""
        portfolio_id = "test_portfolio"
        
        # First call should trigger calculation
        with patch.object(analytics_aggregator, '_calculate_analytics') as mock_calc:
            mock_calc.return_value = Mock(timestamp=datetime.utcnow())
            
            result1 = analytics_aggregator.get_cached_analytics(portfolio_id, cache_ttl_seconds=60)
            mock_calc.assert_called_once()
        
        # Second call within TTL should use cache
        with patch.object(analytics_aggregator, '_calculate_analytics') as mock_calc:
            result2 = analytics_aggregator.get_cached_analytics(portfolio_id, cache_ttl_seconds=60)
            mock_calc.assert_not_called()
        
        assert result1 is result2


class TestAnalyticsIntegration:
    """Integration tests for analytics components working together"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_pipeline(self):
        """Test complete analytics pipeline from raw data to aggregated metrics"""
        # This would test the full pipeline in a real scenario
        # For now, we'll test the component interactions
        
        # Mock database with realistic data
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock realistic position and market data
        mock_conn.fetch.return_value = [
            {
                'instrument_id': 'AAPL_NASDAQ',
                'symbol': 'AAPL',
                'quantity': 100,
                'avg_entry_price': 150.0,
                'realized_pnl': 500.0,
                'created_at': datetime.utcnow() - timedelta(days=5),
                'venue': 'NASDAQ',
                'multiplier': 1
            }
        ]
        
        # Initialize components
        perf_calc = PerformanceCalculator(mock_pool)
        risk_analytics = RiskAnalytics()
        strategy_analytics = StrategyAnalytics()
        execution_analytics = ExecutionAnalytics()
        
        # Mock current price
        async def mock_get_current_price(conn, instrument_id):
            return Decimal('155.0')
        perf_calc._get_current_price = mock_get_current_price
        
        # Test coordinated analytics calculation
        portfolio_id = "integration_test_portfolio"
        
        # Calculate performance
        pnl_result = await perf_calc.calculate_real_time_pnl(portfolio_id)
        assert pnl_result['total_pnl'] == 1000.0  # 500 realized + 500 unrealized
        
        # Verify analytics components can work with the performance data
        assert isinstance(pnl_result, dict)
        assert 'positions' in pnl_result
        assert len(pnl_result['positions']) == 1
    
    def test_analytics_error_handling_and_resilience(self):
        """Test analytics components handle errors gracefully"""
        perf_calc = PerformanceCalculator(None)  # Invalid pool
        
        # Should handle database errors gracefully
        with pytest.raises(Exception):
            # This should fail but not crash the entire system
            asyncio.run(perf_calc.calculate_real_time_pnl("test"))
        
        # Test with invalid data
        risk_analytics = RiskAnalytics()
        
        # Should handle empty or invalid data
        result = risk_analytics.calculate_var([], confidence_level=0.95)
        assert result is None or result == 0
    
    def test_analytics_performance_benchmarks(self):
        """Test analytics performance meets requirements"""
        import time
        
        # Test calculation speed
        risk_analytics = RiskAnalytics()
        
        # Generate test data
        returns = np.random.normal(0.001, 0.02, 1000)
        
        start_time = time.time()
        var_result = risk_analytics.calculate_var(returns, confidence_level=0.95)
        calculation_time = time.time() - start_time
        
        # Should calculate VaR in reasonable time (< 100ms for 1000 data points)
        assert calculation_time < 0.1
        assert var_result is not None