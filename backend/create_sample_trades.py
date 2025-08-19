#!/usr/bin/env python3
"""
Create sample trades for testing trade history functionality
This script creates realistic trading scenarios with proper P&L calculations
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from trade_history_service import get_trade_history_service, Trade

async def create_sample_trades():
    """Create sample trades that demonstrate various trading scenarios"""
    
    service = await get_trade_history_service()
    
    if not service.is_connected:
        print("❌ Trade history service not connected")
        return False
    
    print("🔄 Creating sample trades...")
    
    # Base time - start from 1 hour ago
    base_time = datetime.now() - timedelta(hours=1)
    
    # Sample trading scenarios
    trades = [
        # Scenario 1: AAPL - Profitable momentum trade
        Trade(
            trade_id="TEST_001",
            account_id="DU7925702",
            venue="IB",
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("175.50"),
            commission=Decimal("1.00"),
            execution_time=base_time,
            strategy="momentum",
            notes="QA test data - momentum entry"
        ),
        Trade(
            trade_id="TEST_002",
            account_id="DU7925702",
            venue="IB",
            symbol="AAPL",
            side="SELL",
            quantity=Decimal("10"),
            price=Decimal("178.25"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=15),
            strategy="momentum",
            notes="QA test data - momentum exit (+$27.50 profit)"
        ),
        
        # Scenario 2: MSFT - Partial position management
        Trade(
            trade_id="TEST_003",
            account_id="DU7925702",
            venue="IB",
            symbol="MSFT",
            side="BUY",
            quantity=Decimal("20"),
            price=Decimal("410.00"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=5),
            strategy="swing_trade",
            notes="QA test data - initial position"
        ),
        Trade(
            trade_id="TEST_004",
            account_id="DU7925702",
            venue="IB",
            symbol="MSFT",
            side="SELL",
            quantity=Decimal("10"),
            price=Decimal("415.50"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=25),
            strategy="swing_trade",
            notes="QA test data - partial profit taking (+$55 profit on 10 shares)"
        ),
        Trade(
            trade_id="TEST_005",
            account_id="DU7925702",
            venue="IB",
            symbol="MSFT",
            side="SELL",
            quantity=Decimal("10"),
            price=Decimal("407.75"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=45),
            strategy="swing_trade",
            notes="QA test data - remaining position closed (-$22.50 loss on 10 shares)"
        ),
        
        # Scenario 3: TSLA - Small scalping trades
        Trade(
            trade_id="TEST_006",
            account_id="DU7925702",
            venue="IB",
            symbol="TSLA",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("245.80"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=10),
            strategy="scalping",
            notes="QA test data - scalp entry"
        ),
        Trade(
            trade_id="TEST_007",
            account_id="DU7925702",
            venue="IB",
            symbol="TSLA",
            side="SELL",
            quantity=Decimal("5"),
            price=Decimal("246.95"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=12),
            strategy="scalping",
            notes="QA test data - scalp exit (+$5.75 profit)"
        ),
        
        # Scenario 4: GOOGL - Larger position with loss
        Trade(
            trade_id="TEST_008",
            account_id="DU7925702",
            venue="IB",
            symbol="GOOGL",
            side="BUY",
            quantity=Decimal("3"),
            price=Decimal("2825.00"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=20),
            strategy="value_play",
            notes="QA test data - value position"
        ),
        Trade(
            trade_id="TEST_009",
            account_id="DU7925702",
            venue="IB",
            symbol="GOOGL",
            side="SELL",
            quantity=Decimal("3"),
            price=Decimal("2798.50"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=35),
            strategy="value_play",
            notes="QA test data - stop loss (-$79.50 loss)"
        ),
        
        # Scenario 5: NVDA - Recent profitable trade
        Trade(
            trade_id="TEST_010",
            account_id="DU7925702",
            venue="IB",
            symbol="NVDA",
            side="BUY",
            quantity=Decimal("8"),
            price=Decimal("465.25"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=30),
            strategy="tech_momentum",
            notes="QA test data - tech momentum"
        ),
        Trade(
            trade_id="TEST_011",
            account_id="DU7925702",
            venue="IB",
            symbol="NVDA",
            side="SELL",
            quantity=Decimal("8"),
            price=Decimal("472.80"),
            commission=Decimal("1.00"),
            execution_time=base_time + timedelta(minutes=50),
            strategy="tech_momentum",
            notes="QA test data - profit target (+$60.40 profit)"
        )
    ]
    
    # Record all trades
    success_count = 0
    for trade in trades:
        success = await service.record_trade(trade)
        if success:
            success_count += 1
            print(f"✅ Created trade: {trade.symbol} {trade.side} {trade.quantity} @ ${trade.price}")
        else:
            print(f"❌ Failed to create trade: {trade.trade_id}")
    
    print(f"\n📊 Successfully created {success_count}/{len(trades)} sample trades")
    
    # Calculate and display summary
    summary = await service.get_trade_summary(None)  # No filter - get all trades
    
    print(f"\n📈 Trade Summary:")
    print(f"   Total Trades: {summary.total_trades}")
    print(f"   Win Rate: {summary.win_rate:.1f}%")
    print(f"   Total P&L: ${summary.total_pnl}")
    print(f"   Total Commission: ${summary.total_commission}")
    print(f"   Net P&L: ${summary.net_pnl}")
    
    return success_count > 0

if __name__ == "__main__":
    try:
        result = asyncio.run(create_sample_trades())
        if result:
            print("\n✅ Sample trades created successfully!")
            print("🧪 Trade history system ready for testing with real data")
        else:
            print("\n❌ Failed to create sample trades")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error creating sample trades: {e}")
        sys.exit(1)