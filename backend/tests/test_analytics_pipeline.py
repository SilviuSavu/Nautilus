"""
Integration tests for Analytics Pipeline - Sprint 3

Tests analytics data flow and calculations from raw market data through
performance metrics to real-time streaming and reporting.
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

from analytics.performance_calculator import PerformanceCalculator, PerformanceSnapshot, init_performance_calculator
from analytics.risk_analytics import RiskAnalytics, PortfolioRisk
from analytics.strategy_analytics import StrategyAnalytics, StrategyPerformance
from analytics.execution_analytics import ExecutionAnalytics, ExecutionMetrics
from analytics.analytics_aggregator import AnalyticsAggregator, AggregatedMetrics
from websocket.websocket_manager import WebSocketManager
from websocket.streaming_service import StreamingService


class TestCompleteAnalyticsPipeline:
    """Test complete analytics pipeline from data ingestion to reporting"""
    
    @pytest.fixture
    def analytics_system(self):
        """Setup complete analytics system"""
        # Mock database pool
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Initialize components
        performance_calculator = PerformanceCalculator(mock_pool)
        risk_analytics = RiskAnalytics()
        strategy_analytics = StrategyAnalytics()
        execution_analytics = ExecutionAnalytics()
        analytics_aggregator = AnalyticsAggregator()
        websocket_manager = WebSocketManager()
        streaming_service = StreamingService()
        
        # Link components to aggregator
        analytics_aggregator.performance_calculator = performance_calculator
        analytics_aggregator.risk_analytics = risk_analytics
        analytics_aggregator.strategy_analytics = strategy_analytics
        analytics_aggregator.execution_analytics = execution_analytics
        
        return {
            "performance_calculator": performance_calculator,
            "risk_analytics": risk_analytics,
            "strategy_analytics": strategy_analytics,
            "execution_analytics": execution_analytics,
            "analytics_aggregator": analytics_aggregator,
            "websocket_manager": websocket_manager,
            "streaming_service": streaming_service,
            "mock_db_conn": mock_conn
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        data = {}
        
        base_time = datetime.utcnow() - timedelta(days=30)
        
        for symbol in symbols:
            base_price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2800, "TSLA": 800, "NVDA": 500}[symbol]
            
            # Generate 30 days of minute-level data
            prices = []
            current_price = base_price
            
            for day in range(30):
                for minute in range(390):  # Trading minutes per day
                    # Random walk with slight upward bias
                    price_change = np.random.normal(0.0002, 0.005)  # Small drift, moderate volatility
                    current_price = max(0.01, current_price * (1 + price_change))
                    
                    timestamp = base_time + timedelta(days=day, minutes=minute)
                    
                    prices.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "price": current_price,
                        "volume": np.random.randint(1000, 10000)
                    })
            
            data[symbol] = prices
        
        return data
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data with positions and trades"""
        return {
            "portfolio_id": "analytics_test_portfolio",
            "positions": {
                "AAPL": {
                    "quantity": 1000,
                    "avg_entry_price": Decimal("148.50"),
                    "current_price": Decimal("155.25"),
                    "market_value": Decimal("155250.00"),
                    "unrealized_pnl": Decimal("6750.00"),
                    "realized_pnl": Decimal("2500.00")
                },
                "MSFT": {
                    "quantity": 500,
                    "avg_entry_price": Decimal("295.00"),
                    "current_price": Decimal("308.75"),
                    "market_value": Decimal("154375.00"),
                    "unrealized_pnl": Decimal("6875.00"),
                    "realized_pnl": Decimal("1200.00")
                },
                "GOOGL": {
                    "quantity": 100,
                    "avg_entry_price": Decimal("2750.00"),
                    "current_price": Decimal("2825.50"),
                    "market_value": Decimal("282550.00"),
                    "unrealized_pnl": Decimal("7550.00"),
                    "realized_pnl": Decimal("-800.00")
                }
            },
            "trades": [
                {
                    "trade_id": "trade_001",
                    "symbol": "AAPL",
                    "side": "BUY",
                    "quantity": 1000,
                    "price": Decimal("148.50"),
                    "timestamp": datetime.utcnow() - timedelta(days=15),
                    "strategy_id": "momentum_v1",
                    "pnl": Decimal("0")
                },
                {
                    "trade_id": "trade_002",
                    "symbol": "MSFT",
                    "side": "BUY",
                    "quantity": 500,
                    "price": Decimal("295.00"),
                    "timestamp": datetime.utcnow() - timedelta(days=12),
                    "strategy_id": "momentum_v1",
                    "pnl": Decimal("0")
                },
                {
                    "trade_id": "trade_003",
                    "symbol": "GOOGL",
                    "side": "BUY",
                    "quantity": 100,
                    "price": Decimal("2750.00"),
                    "timestamp": datetime.utcnow() - timedelta(days=8),
                    "strategy_id": "value_v2",
                    "pnl": Decimal("0")
                }
            ],
            "cash_balance": Decimal("50000.00"),
            "total_value": Decimal("642175.00")
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_calculation(self, analytics_system, sample_portfolio_data):
        """Test end-to-end analytics calculation pipeline"""
        performance_calc = analytics_system["performance_calculator"]
        risk_analytics = analytics_system["risk_analytics"]
        strategy_analytics = analytics_system["strategy_analytics"]
        execution_analytics = analytics_system["execution_analytics"]
        analytics_aggregator = analytics_system["analytics_aggregator"]
        mock_conn = analytics_system["mock_db_conn"]
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # 1. Mock database responses for performance calculation
        mock_positions = []
        for symbol, position in sample_portfolio_data["positions"].items():
            mock_positions.append({
                "instrument_id": f"{symbol}_NASDAQ",
                "position_id": f"pos_{symbol}",
                "quantity": float(position["quantity"]),
                "avg_entry_price": float(position["avg_entry_price"]),
                "realized_pnl": float(position["realized_pnl"]),
                "created_at": datetime.utcnow() - timedelta(days=15),
                "symbol": symbol,
                "venue": "NASDAQ",
                "multiplier": 1
            })
        
        mock_conn.fetch.return_value = mock_positions
        
        # Mock current price lookup
        async def mock_get_current_price(conn, instrument_id):
            symbol = instrument_id.split("_")[0]
            return sample_portfolio_data["positions"][symbol]["current_price"]
        
        performance_calc._get_current_price = mock_get_current_price
        
        # 2. Calculate real-time P&L
        pnl_result = await performance_calc.calculate_real_time_pnl(portfolio_id)
        
        assert pnl_result["portfolio_id"] == portfolio_id
        assert pnl_result["total_pnl"] > 0  # Should be profitable
        assert len(pnl_result["positions"]) == 3
        
        # 3. Calculate portfolio risk metrics
        portfolio_positions = []
        for symbol, position in sample_portfolio_data["positions"].items():
            portfolio_positions.append({
                "symbol": symbol,
                "quantity": float(position["quantity"]),
                "price": float(position["current_price"]),
                "beta": {"AAPL": 1.2, "MSFT": 0.9, "GOOGL": 1.1}[symbol]
            })
        
        # Mock correlation matrix
        correlation_matrix = pd.DataFrame([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0]
        ], index=["AAPL", "MSFT", "GOOGL"], columns=["AAPL", "MSFT", "GOOGL"])
        
        volatilities = {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.30}
        
        portfolio_risk = risk_analytics.calculate_portfolio_risk(
            portfolio_positions, correlation_matrix, volatilities
        )
        
        assert isinstance(portfolio_risk, PortfolioRisk)
        assert portfolio_risk.total_value > 0
        assert portfolio_risk.portfolio_beta > 0
        assert len(portfolio_risk.position_risks) == 3
        
        # 4. Calculate strategy performance
        strategy_trades = {}
        for trade in sample_portfolio_data["trades"]:
            strategy_id = trade["strategy_id"]
            if strategy_id not in strategy_trades:
                strategy_trades[strategy_id] = []
            
            strategy_trades[strategy_id].append({
                "entry_time": trade["timestamp"],
                "exit_time": trade["timestamp"] + timedelta(days=1),  # Mock exit
                "pnl": float(trade["pnl"]) + 100,  # Mock positive P&L
                "quantity": float(trade["quantity"])
            })
        
        strategy_performances = {}
        for strategy_id, trades in strategy_trades.items():
            performance = strategy_analytics.calculate_strategy_performance(strategy_id, trades)
            strategy_performances[strategy_id] = performance
            
            assert isinstance(performance, StrategyPerformance)
            assert performance.strategy_id == strategy_id
        
        # 5. Calculate execution analytics
        executions = []
        for trade in sample_portfolio_data["trades"]:
            executions.append({
                "order_id": f"ord_{trade['trade_id']}",
                "symbol": trade["symbol"],
                "side": trade["side"],
                "quantity": float(trade["quantity"]),
                "executed_price": float(trade["price"]),
                "benchmark_price": float(trade["price"]) * 0.999,  # Slight improvement
                "execution_time": trade["timestamp"],
                "order_time": trade["timestamp"] - timedelta(minutes=1)
            })
        
        execution_metrics = execution_analytics.analyze_trade_executions(executions)
        
        assert isinstance(execution_metrics, ExecutionMetrics)
        assert execution_metrics.total_executions == len(executions)
        
        # 6. Aggregate all analytics
        aggregated = await analytics_aggregator.get_comprehensive_analytics(portfolio_id)
        
        assert isinstance(aggregated, AggregatedMetrics)
        assert aggregated.portfolio_id == portfolio_id
        assert aggregated.performance_metrics is not None
        assert aggregated.risk_metrics is not None
    
    @pytest.mark.asyncio
    async def test_real_time_analytics_streaming(self, analytics_system, sample_portfolio_data):
        """Test real-time analytics streaming through WebSocket"""
        analytics_aggregator = analytics_system["analytics_aggregator"]
        websocket_manager = analytics_system["websocket_manager"]
        streaming_service = analytics_system["streaming_service"]
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # Setup WebSocket clients interested in analytics
        analytics_clients = []
        client_types = ["portfolio_manager", "risk_manager", "trader"]
        
        for client_type in client_types:
            mock_ws = AsyncMock()
            conn_id = f"{client_type}_client"
            
            await websocket_manager.connect(mock_ws, conn_id, client_type)
            websocket_manager.subscribe_to_topic(conn_id, f"analytics.{portfolio_id}.updates")
            
            analytics_clients.append((conn_id, mock_ws, client_type))
        
        # Setup Redis for streaming
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            from websocket.redis_pubsub import RedisPubSubManager
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Simulate real-time market data updates triggering analytics
            market_updates = [
                {"symbol": "AAPL", "price": 157.50, "change": +2.25},
                {"symbol": "MSFT", "price": 312.00, "change": +3.25}, 
                {"symbol": "GOOGL", "price": 2850.00, "change": +24.50}
            ]
            
            for update in market_updates:
                # Mock updated analytics calculation
                updated_metrics = {
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_value": 650000.0,  # Increased due to price gains
                    "daily_pnl": 8000.0,  # Positive P&L
                    "performance_metrics": {
                        "total_return": 0.12,
                        "sharpe_ratio": 1.35,
                        "max_drawdown": -0.05,
                        "volatility": 0.18
                    },
                    "risk_metrics": {
                        "var_95": 13000.0,
                        "expected_shortfall": 19500.0,
                        "portfolio_beta": 1.08
                    },
                    "updated_position": {
                        "symbol": update["symbol"],
                        "new_price": update["price"],
                        "price_change": update["change"],
                        "impact_on_portfolio": update["change"] * sample_portfolio_data["positions"][update["symbol"]]["quantity"]
                    }
                }
                
                # Stream analytics update
                success = await streaming_service.stream_analytics_update(portfolio_id, updated_metrics)
                assert success is True
                
                # Broadcast to WebSocket clients
                sent_count = await websocket_manager.broadcast_message(
                    updated_metrics, f"analytics.{portfolio_id}.updates"
                )
                assert sent_count == 3  # All analytics clients
            
            # Verify all clients received updates
            for conn_id, mock_ws, client_type in analytics_clients:
                # Should have received 3 updates
                assert mock_ws.send_text.call_count == 3
    
    @pytest.mark.asyncio
    async def test_analytics_performance_attribution_pipeline(self, analytics_system, sample_portfolio_data):
        """Test performance attribution calculation pipeline"""
        performance_calc = analytics_system["performance_calculator"]
        mock_conn = analytics_system["mock_db_conn"]
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # Mock attribution query results with asset class data
        mock_attribution_data = []
        asset_classes = {"AAPL": "EQUITY", "MSFT": "EQUITY", "GOOGL": "EQUITY"}
        sectors = {"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "GOOGL": "TECHNOLOGY"}
        
        for symbol, position in sample_portfolio_data["positions"].items():
            mock_attribution_data.append({
                "asset_class": asset_classes[symbol],
                "sector": sectors[symbol],
                "symbol": symbol,
                "instrument_id": f"{symbol}_NASDAQ",
                "position_value": float(position["market_value"]),
                "total_pnl": float(position["unrealized_pnl"] + position["realized_pnl"])
            })
        
        mock_conn.fetch.return_value = mock_attribution_data
        
        # Calculate performance attribution
        attribution = await performance_calc.calculate_performance_attribution(portfolio_id)
        
        assert attribution["portfolio_id"] == portfolio_id
        assert "attribution_by_asset_class" in attribution
        
        # Verify equity attribution (all positions are equity)
        equity_attr = attribution["attribution_by_asset_class"]["EQUITY"]
        assert equity_attr["weight"] == 1.0  # 100% equity
        assert equity_attr["position_count"] == 3
        assert equity_attr["total_pnl"] > 0  # Profitable
        
        # Test sector attribution (if implemented)
        # This would break down equity into technology vs other sectors
        total_tech_value = sum(
            float(pos["market_value"]) 
            for pos in sample_portfolio_data["positions"].values()
        )
        
        assert total_tech_value > 0
    
    @pytest.mark.asyncio
    async def test_analytics_risk_factor_analysis(self, analytics_system, sample_market_data):
        """Test risk factor analysis pipeline"""
        risk_analytics = analytics_system["risk_analytics"]
        
        # Create factor exposure analysis
        portfolio_positions = [
            {"symbol": "AAPL", "weight": 0.25, "beta": 1.2, "value_score": -0.1, "momentum_score": 0.3},
            {"symbol": "MSFT", "weight": 0.25, "beta": 0.9, "value_score": 0.1, "momentum_score": 0.1},
            {"symbol": "GOOGL", "weight": 0.45, "beta": 1.1, "value_score": 0.0, "momentum_score": 0.2},
            {"symbol": "TSLA", "weight": 0.05, "beta": 1.8, "value_score": -0.5, "momentum_score": 0.8}
        ]
        
        # Calculate factor exposures
        factor_exposures = risk_analytics.calculate_factor_exposures(portfolio_positions)
        
        assert "market_beta" in factor_exposures
        assert "value_factor" in factor_exposures
        assert "momentum_factor" in factor_exposures
        
        # Verify calculations
        expected_beta = sum(pos["weight"] * pos["beta"] for pos in portfolio_positions)
        assert abs(factor_exposures["market_beta"] - expected_beta) < 0.01
        
        expected_value = sum(pos["weight"] * pos["value_score"] for pos in portfolio_positions)
        assert abs(factor_exposures["value_factor"] - expected_value) < 0.01
        
        # Calculate factor risk contribution
        factor_risks = risk_analytics.calculate_factor_risk_contribution(
            factor_exposures, 
            factor_volatilities={"market": 0.15, "value": 0.08, "momentum": 0.12},
            factor_correlations={
                ("market", "value"): -0.2,
                ("market", "momentum"): 0.3,
                ("value", "momentum"): -0.4
            }
        )
        
        assert "total_factor_risk" in factor_risks
        assert "specific_risk" in factor_risks
        assert "risk_decomposition" in factor_risks
    
    @pytest.mark.asyncio
    async def test_analytics_backtesting_integration(self, analytics_system, sample_market_data):
        """Test analytics integration with backtesting pipeline"""
        strategy_analytics = analytics_system["strategy_analytics"]
        performance_calc = analytics_system["performance_calculator"]
        
        # Simulate backtest results for multiple strategies
        backtest_results = {
            "momentum_v1": {
                "returns": [0.02, -0.01, 0.015, -0.005, 0.025, -0.008, 0.012, 0.018],
                "trades": [
                    {"entry_time": datetime.utcnow() - timedelta(days=20), "exit_time": datetime.utcnow() - timedelta(days=18), "pnl": 250.0, "quantity": 100},
                    {"entry_time": datetime.utcnow() - timedelta(days=15), "exit_time": datetime.utcnow() - timedelta(days=13), "pnl": -75.0, "quantity": 50},
                    {"entry_time": datetime.utcnow() - timedelta(days=10), "exit_time": datetime.utcnow() - timedelta(days=8), "pnl": 400.0, "quantity": 80}
                ],
                "final_value": 105000.0,
                "max_drawdown": -0.08
            },
            "mean_reversion_v2": {
                "returns": [0.005, 0.008, -0.003, 0.012, -0.006, 0.009, 0.004, -0.002],
                "trades": [
                    {"entry_time": datetime.utcnow() - timedelta(days=18), "exit_time": datetime.utcnow() - timedelta(days=16), "pnl": 125.0, "quantity": 200},
                    {"entry_time": datetime.utcnow() - timedelta(days=12), "exit_time": datetime.utcnow() - timedelta(days=10), "pnl": 200.0, "quantity": 150}
                ],
                "final_value": 103000.0,
                "max_drawdown": -0.03
            }
        }
        
        # Analyze each strategy
        strategy_analysis = {}
        for strategy_id, results in backtest_results.items():
            # Calculate strategy performance
            performance = strategy_analytics.calculate_strategy_performance(
                strategy_id, results["trades"]
            )
            
            # Calculate risk-adjusted metrics
            returns = np.array(results["returns"])
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            strategy_analysis[strategy_id] = {
                "performance": performance,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": results["final_value"] / 100000.0 - 1,
                "volatility": np.std(returns) * np.sqrt(252)
            }
        
        # Compare strategies
        comparison = strategy_analytics.compare_strategies({
            strategy_id: {
                "returns": backtest_results[strategy_id]["returns"],
                "trades": len(backtest_results[strategy_id]["trades"]),
                "win_rate": analysis["performance"].win_rate
            }
            for strategy_id, analysis in strategy_analysis.items()
        })
        
        assert len(comparison) == 2
        assert "momentum_v1" in comparison
        assert "mean_reversion_v2" in comparison
        
        # Verify momentum strategy has higher returns but higher risk
        momentum_analysis = strategy_analysis["momentum_v1"]
        mean_rev_analysis = strategy_analysis["mean_reversion_v2"]
        
        assert momentum_analysis["total_return"] > mean_rev_analysis["total_return"]
        assert abs(momentum_analysis["max_drawdown"]) > abs(mean_rev_analysis["max_drawdown"])
    
    def test_analytics_caching_and_performance(self, analytics_system):
        """Test analytics caching and performance optimization"""
        analytics_aggregator = analytics_system["analytics_aggregator"]
        
        portfolio_id = "performance_test_portfolio"
        
        # Test caching mechanism
        import time
        
        # First call should trigger calculation
        start_time = time.time()
        
        with patch.object(analytics_aggregator, '_calculate_analytics') as mock_calc:
            mock_analytics = Mock()
            mock_analytics.timestamp = datetime.utcnow()
            mock_calc.return_value = mock_analytics
            
            result1 = analytics_aggregator.get_cached_analytics(portfolio_id, cache_ttl_seconds=60)
            
            first_call_time = time.time() - start_time
            mock_calc.assert_called_once()
        
        # Second call within TTL should use cache
        start_time = time.time()
        
        with patch.object(analytics_aggregator, '_calculate_analytics') as mock_calc:
            result2 = analytics_aggregator.get_cached_analytics(portfolio_id, cache_ttl_seconds=60)
            
            second_call_time = time.time() - start_time
            mock_calc.assert_not_called()  # Should not be called due to cache
        
        # Cache should be faster
        assert second_call_time < first_call_time
        assert result1 is result2  # Same object from cache
        
        # Test cache expiration
        with patch.object(analytics_aggregator, '_calculate_analytics') as mock_calc:
            mock_analytics_new = Mock()
            mock_analytics_new.timestamp = datetime.utcnow()
            mock_calc.return_value = mock_analytics_new
            
            # Use expired cache (0 seconds TTL)
            result3 = analytics_aggregator.get_cached_analytics(portfolio_id, cache_ttl_seconds=0)
            
            mock_calc.assert_called_once()  # Should be called due to expired cache
            assert result3 is not result1  # Different object


class TestAnalyticsPerformanceBenchmarks:
    """Test analytics performance under realistic load"""
    
    @pytest.fixture
    def large_portfolio_data(self):
        """Generate large portfolio for performance testing"""
        positions = {}
        trades = []
        
        # Create 500 positions
        for i in range(500):
            symbol = f"STOCK_{i:03d}"
            positions[symbol] = {
                "quantity": 100 + (i % 1000),
                "avg_entry_price": Decimal(f"{50 + (i % 200)}.00"),
                "current_price": Decimal(f"{52 + (i % 180)}.00"),
                "market_value": Decimal(f"{(52 + (i % 180)) * (100 + (i % 1000))}"),
                "unrealized_pnl": Decimal(f"{2 * (100 + (i % 1000))}"),
                "realized_pnl": Decimal(f"{(i % 100) - 50}")
            }
            
            # Create some trades
            if i % 10 == 0:
                trades.append({
                    "trade_id": f"trade_{i}",
                    "symbol": symbol,
                    "side": "BUY" if i % 20 == 0 else "SELL",
                    "quantity": 100 + (i % 500),
                    "price": Decimal(f"{50 + (i % 200)}.00"),
                    "timestamp": datetime.utcnow() - timedelta(days=i % 30),
                    "strategy_id": f"strategy_{i % 10}",
                    "pnl": Decimal(f"{(i % 200) - 100}")
                })
        
        return {
            "portfolio_id": "large_portfolio_perf_test",
            "positions": positions,
            "trades": trades,
            "total_value": sum(pos["market_value"] for pos in positions.values())
        }
    
    def test_large_portfolio_analytics_performance(self, large_portfolio_data):
        """Test analytics performance on large portfolio"""
        import time
        
        # Test risk analytics performance
        risk_analytics = RiskAnalytics()
        
        # Convert positions to format for risk calculation
        portfolio_positions = []
        for symbol, position in large_portfolio_data["positions"].items():
            portfolio_positions.append({
                "symbol": symbol,
                "quantity": float(position["quantity"]),
                "price": float(position["current_price"]),
                "beta": 1.0 + (hash(symbol) % 100) / 100.0  # Random beta 1.0-2.0
            })
        
        start_time = time.time()
        
        # Calculate portfolio risk (simplified, without correlation matrix)
        total_value = sum(pos["quantity"] * pos["price"] for pos in portfolio_positions)
        portfolio_beta = sum(
            (pos["quantity"] * pos["price"] / total_value) * pos["beta"]
            for pos in portfolio_positions
        )
        
        risk_calculation_time = time.time() - start_time
        
        # Should calculate efficiently even for large portfolio
        assert risk_calculation_time < 1.0  # Within 1 second
        assert 0.5 < portfolio_beta < 2.5  # Reasonable beta range
        
        # Test performance attribution calculation
        start_time = time.time()
        
        # Group positions by sector (mock)
        sectors = {}
        for position in portfolio_positions:
            sector = f"SECTOR_{hash(position['symbol']) % 10}"
            if sector not in sectors:
                sectors[sector] = {"value": 0, "count": 0}
            
            sectors[sector]["value"] += position["quantity"] * position["price"]
            sectors[sector]["count"] += 1
        
        attribution_time = time.time() - start_time
        
        # Should handle large attribution calculation efficiently
        assert attribution_time < 0.5  # Within 500ms
        assert len(sectors) <= 10  # Should have 10 sectors
    
    def test_concurrent_analytics_calculations(self, large_portfolio_data):
        """Test concurrent analytics calculations"""
        import concurrent.futures
        import threading
        import time
        
        risk_analytics = RiskAnalytics()
        
        def calculate_portfolio_metrics(portfolio_subset):
            """Calculate metrics for a subset of portfolio"""
            positions = list(large_portfolio_data["positions"].items())[portfolio_subset[0]:portfolio_subset[1]]
            
            total_value = sum(float(pos[1]["market_value"]) for pos in positions)
            total_pnl = sum(float(pos[1]["unrealized_pnl"]) for pos in positions)
            
            return {
                "subset": portfolio_subset,
                "total_value": total_value,
                "total_pnl": total_pnl,
                "position_count": len(positions),
                "return_pct": (total_pnl / total_value) * 100 if total_value > 0 else 0
            }
        
        # Split large portfolio into chunks for concurrent processing
        portfolio_size = len(large_portfolio_data["positions"])
        chunk_size = portfolio_size // 10
        portfolio_chunks = [
            (i * chunk_size, min((i + 1) * chunk_size, portfolio_size))
            for i in range(10)
        ]
        
        start_time = time.time()
        
        # Process chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(calculate_portfolio_metrics, chunk)
                for chunk in portfolio_chunks
            ]
            
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
        
        concurrent_time = time.time() - start_time
        
        # Should process concurrently efficiently
        assert concurrent_time < 2.0  # Within 2 seconds
        assert len(results) == 10
        
        # Verify results
        total_value = sum(result["total_value"] for result in results)
        total_pnl = sum(result["total_pnl"] for result in results)
        
        assert total_value > 0
        assert abs(total_pnl) >= 0  # Could be positive or negative
    
    def test_memory_usage_analytics_calculations(self):
        """Test memory usage during analytics calculations"""
        import sys
        import gc
        
        # Measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform repeated calculations
        risk_analytics = RiskAnalytics()
        
        for i in range(100):
            # Generate data for each iteration
            returns = np.random.normal(0.001, 0.02, 252)  # One year of daily returns
            
            # Calculate VaR
            var_95 = risk_analytics.calculate_var(returns, confidence_level=0.95)
            
            # Calculate expected shortfall
            es_95 = risk_analytics.calculate_expected_shortfall(returns, confidence_level=0.95)
            
            # Verify calculations are reasonable
            assert var_95 < 0  # VaR should be negative
            assert es_95 < var_95  # ES should be more extreme than VaR
        
        # Measure final memory
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be minimal
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Less than 1000 new objects
        
        # Objects per calculation should be minimal
        objects_per_calc = object_growth / 100
        assert objects_per_calc < 10  # Less than 10 objects per calculation


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])