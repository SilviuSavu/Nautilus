"""
Unit tests for trade history service and calculations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from trade_history_service import (
    TradeHistoryService, Trade, TradeSummary, TradeFilter,
    TradeType, TradeStatus
)
from trade_data_processing import TradeDataProcessor


class TestTradeHistoryService:
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing"""
        base_time = datetime.now()
        return [
            Trade(
                trade_id="T001",
                account_id="ACC001",
                venue="IB",
                symbol="AAPL",
                side="BUY",
                quantity=Decimal("100"),
                price=Decimal("150.00"),
                commission=Decimal("1.00"),
                execution_time=base_time,
                strategy="momentum"
            ),
            Trade(
                trade_id="T002",
                account_id="ACC001",
                venue="IB",
                symbol="AAPL",
                side="SELL",
                quantity=Decimal("50"),
                price=Decimal("155.00"),
                commission=Decimal("1.00"),
                execution_time=base_time + timedelta(hours=1),
                strategy="momentum"
            ),
            Trade(
                trade_id="T003",
                account_id="ACC001",
                venue="IB",
                symbol="MSFT",
                side="BUY",
                quantity=Decimal("200"),
                price=Decimal("300.00"),
                commission=Decimal("2.00"),
                execution_time=base_time + timedelta(hours=2),
                strategy="mean_reversion"
            )
        ]
    
    @pytest.fixture
    def trade_service(self):
        """Create a trade history service for testing"""
        service = TradeHistoryService()
        service._connected = True  # Mock connection
        service._pool = MagicMock()  # Mock database pool
        return service
    
    def test_trade_creation(self):
        """Test trade object creation"""
        trade = Trade(
            trade_id="TEST001",
            account_id="ACC001",
            venue="IB",
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            commission=Decimal("1.00"),
            execution_time=datetime.now()
        )
        
        assert trade.trade_id == "TEST001"
        assert trade.symbol == "AAPL"
        assert trade.quantity == Decimal("100")
        assert trade.price == Decimal("150.00")
    
    @pytest.mark.asyncio
    async def test_pnl_calculation(self, trade_service, sample_trades):
        """Test P&L calculation logic"""
        summary = await trade_service.calculate_pnl(sample_trades)
        
        assert isinstance(summary, TradeSummary)
        assert summary.total_trades == 3
        assert summary.total_commission == Decimal("4.00")  # 1 + 1 + 2
        
        # Expected P&L: AAPL position (50 shares sold at 155, bought at 150) = 250
        # MSFT position still open, no realized P&L
        # So realized P&L should be around 250 - commissions
        assert summary.total_pnl > Decimal("0")
    
    def test_trade_filter_creation(self):
        """Test trade filter object creation"""
        filter_obj = TradeFilter(
            symbol="AAPL",
            venue="IB",
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            limit=100
        )
        
        assert filter_obj.symbol == "AAPL"
        assert filter_obj.venue == "IB"
        assert filter_obj.limit == 100
    
    @pytest.mark.asyncio
    async def test_export_csv(self, trade_service, sample_trades):
        """Test CSV export functionality"""
        csv_data = await trade_service.export_trades_csv(sample_trades)
        
        assert isinstance(csv_data, str)
        assert "Trade ID" in csv_data  # Header
        assert "T001" in csv_data  # Trade data
        assert "AAPL" in csv_data
        assert "150.0" in csv_data
    
    @pytest.mark.asyncio
    async def test_export_json(self, trade_service, sample_trades):
        """Test JSON export functionality"""
        json_data = await trade_service.export_trades_json(sample_trades)
        
        assert isinstance(json_data, str)
        assert '"trade_id": "T001"' in json_data
        assert '"symbol": "AAPL"' in json_data
        assert '"quantity": "100"' in json_data


class TestTradeDataProcessor:
    
    @pytest.fixture
    def processor(self):
        """Create a trade data processor for testing"""
        return TradeDataProcessor()
    
    @pytest.fixture
    def position_trades(self):
        """Create trades that form a complete position"""
        base_time = datetime.now()
        return [
            Trade(
                trade_id="P001",
                account_id="ACC001",
                venue="IB",
                symbol="TSLA",
                side="BUY",
                quantity=Decimal("100"),
                price=Decimal("200.00"),
                commission=Decimal("1.00"),
                execution_time=base_time
            ),
            Trade(
                trade_id="P002",
                account_id="ACC001",
                venue="IB",
                symbol="TSLA",
                side="SELL",
                quantity=Decimal("100"),
                price=Decimal("220.00"),
                commission=Decimal("1.00"),
                execution_time=base_time + timedelta(days=1)
            )
        ]
    
    def test_position_tracking(self, processor, position_trades):
        """Test position tracking logic"""
        positions = processor._track_positions(position_trades)
        
        assert len(positions) == 1
        tsla_position = positions["TSLA_IB"]
        
        # Position should be flat (quantity = 0) after buy and sell
        assert tsla_position['quantity'] == Decimal('0')
        
        # Realized P&L should be (220 - 200) * 100 = 2000
        assert tsla_position['realized_pnl'] == Decimal('2000')
    
    def test_win_rate_calculation(self, processor):
        """Test win rate calculation"""
        positions = {
            "POS1": {"realized_pnl": Decimal("100")},  # Win
            "POS2": {"realized_pnl": Decimal("-50")},  # Loss
            "POS3": {"realized_pnl": Decimal("200")},  # Win
            "POS4": {"realized_pnl": Decimal("0")},    # No P&L
        }
        
        win_rate = processor._calculate_win_rate(positions)
        
        # 2 wins out of 3 positions with P&L = 66.67%
        assert win_rate == pytest.approx(66.67, rel=1e-2)
    
    def test_profit_factor_calculation(self, processor):
        """Test profit factor calculation"""
        positions = {
            "POS1": {"realized_pnl": Decimal("100")},  # Win
            "POS2": {"realized_pnl": Decimal("-50")},  # Loss
            "POS3": {"realized_pnl": Decimal("200")},  # Win
        }
        
        profit_factor = processor._calculate_profit_factor(positions)
        
        # Gross profit = 300, Gross loss = 50, PF = 6.0
        assert profit_factor == 6.0
    
    def test_consecutive_trades(self, processor):
        """Test consecutive wins/losses calculation"""
        positions = {
            "POS1": {"realized_pnl": Decimal("100")},   # Win
            "POS2": {"realized_pnl": Decimal("50")},    # Win
            "POS3": {"realized_pnl": Decimal("75")},    # Win
            "POS4": {"realized_pnl": Decimal("-25")},   # Loss
            "POS5": {"realized_pnl": Decimal("-50")},   # Loss
            "POS6": {"realized_pnl": Decimal("100")},   # Win
        }
        
        max_wins, max_losses = processor._calculate_consecutive_trades(positions)
        
        assert max_wins == 3  # Three consecutive wins at start
        assert max_losses == 2  # Two consecutive losses in middle
    
    def test_largest_trades(self, processor):
        """Test largest win/loss calculation"""
        positions = {
            "POS1": {"realized_pnl": Decimal("100")},
            "POS2": {"realized_pnl": Decimal("-150")},
            "POS3": {"realized_pnl": Decimal("250")},
            "POS4": {"realized_pnl": Decimal("-75")},
        }
        
        largest_win, largest_loss = processor._calculate_largest_trades(positions)
        
        assert largest_win == Decimal("250")
        assert largest_loss == Decimal("-150")
    
    def test_sharpe_ratio_calculation(self, processor):
        """Test Sharpe ratio calculation"""
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.025]  # Daily returns
        
        sharpe = processor._calculate_sharpe_ratio(returns)
        
        # Should be a reasonable Sharpe ratio
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_max_drawdown_calculation(self, processor):
        """Test maximum drawdown calculation"""
        returns = [0.01, -0.05, -0.02, 0.03, 0.01, -0.03]
        
        max_dd = processor._calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, Decimal)
        assert max_dd >= Decimal('0')  # Drawdown should be positive
    
    @pytest.mark.asyncio
    async def test_advanced_analytics(self, processor, position_trades):
        """Test advanced analytics calculation"""
        analytics = await processor.calculate_advanced_analytics(position_trades)
        
        assert "TSLA" in analytics
        tsla_analytics = analytics["TSLA"]
        
        assert tsla_analytics.symbol == "TSLA"
        assert tsla_analytics.total_trades == 2
        assert tsla_analytics.total_volume == Decimal("200")  # 100 + 100
        assert tsla_analytics.win_rate >= 0
        assert tsla_analytics.profit_factor >= 0
    
    def test_empty_analytics(self, processor):
        """Test analytics with no trades"""
        analytics = processor._empty_analytics("EMPTY")
        
        assert analytics.symbol == "EMPTY"
        assert analytics.total_trades == 0
        assert analytics.total_volume == Decimal('0')
        assert analytics.total_pnl == Decimal('0')
        assert analytics.win_rate == 0.0


class TestIntegrationScenarios:
    """Integration tests for complex trading scenarios"""
    
    @pytest.fixture
    def complex_trading_scenario(self):
        """Create a complex trading scenario with multiple symbols and strategies"""
        base_time = datetime.now()
        
        return [
            # AAPL momentum strategy
            Trade("T001", "ACC001", "IB", "AAPL", "BUY", Decimal("100"), Decimal("150"), Decimal("1"), base_time, strategy="momentum"),
            Trade("T002", "ACC001", "IB", "AAPL", "SELL", Decimal("100"), Decimal("155"), Decimal("1"), base_time + timedelta(hours=2), strategy="momentum"),
            
            # MSFT mean reversion
            Trade("T003", "ACC001", "IB", "MSFT", "BUY", Decimal("50"), Decimal("300"), Decimal("1"), base_time + timedelta(days=1), strategy="mean_reversion"),
            Trade("T004", "ACC001", "IB", "MSFT", "SELL", Decimal("25"), Decimal("295"), Decimal("1"), base_time + timedelta(days=2), strategy="mean_reversion"),
            Trade("T005", "ACC001", "IB", "MSFT", "SELL", Decimal("25"), Decimal("305"), Decimal("1"), base_time + timedelta(days=3), strategy="mean_reversion"),
            
            # TSLA scalping
            Trade("T006", "ACC001", "IB", "TSLA", "BUY", Decimal("10"), Decimal("200"), Decimal("1"), base_time + timedelta(days=4), strategy="scalping"),
            Trade("T007", "ACC001", "IB", "TSLA", "SELL", Decimal("10"), Decimal("202"), Decimal("1"), base_time + timedelta(days=4, minutes=30), strategy="scalping"),
        ]
    
    @pytest.mark.asyncio
    async def test_multi_symbol_analytics(self, complex_trading_scenario):
        """Test analytics across multiple symbols and strategies"""
        processor = TradeDataProcessor()
        service = TradeHistoryService()
        
        # Test P&L calculation
        summary = await service.calculate_pnl(complex_trading_scenario)
        
        assert summary.total_trades == 7
        
        # Test advanced analytics
        analytics = await processor.calculate_advanced_analytics(complex_trading_scenario)
        
        assert len(analytics) == 3  # AAPL, MSFT, TSLA
        assert "AAPL" in analytics
        assert "MSFT" in analytics
        assert "TSLA" in analytics
        
        # AAPL should show profit
        aapl_analytics = analytics["AAPL"]
        assert aapl_analytics.total_pnl > 0
        
        # TSLA should show small profit from scalping
        tsla_analytics = analytics["TSLA"]
        assert tsla_analytics.total_pnl > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])