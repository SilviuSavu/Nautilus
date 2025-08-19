"""
Trade Data Processing Utilities
Advanced processing and analytics for trade data with NautilusTrader integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import pandas as pd
import numpy as np

from trade_history_service import Trade, TradeSummary, TradeFilter
from historical_data_service import historical_data_service


@dataclass
class TradeAnalytics:
    """Advanced trade analytics"""
    symbol: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    total_volume: Decimal
    total_pnl: Decimal
    win_rate: float
    profit_factor: float
    max_drawdown: Decimal
    sharpe_ratio: Optional[float]
    calmar_ratio: Optional[float]
    sortino_ratio: Optional[float]
    average_holding_period: timedelta
    max_consecutive_wins: int
    max_consecutive_losses: int
    largest_win: Decimal
    largest_loss: Decimal
    risk_adjusted_return: float


@dataclass
class MarketImpactAnalysis:
    """Market impact analysis for trades"""
    symbol: str
    trade_id: str
    execution_price: Decimal
    market_price_before: Optional[Decimal]
    market_price_after: Optional[Decimal]
    price_impact: Optional[Decimal]
    price_impact_bps: Optional[float]
    volume_impact: Optional[float]
    execution_quality: str  # 'excellent', 'good', 'fair', 'poor'


class TradeDataProcessor:
    """
    Advanced trade data processing with NautilusTrader integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def calculate_advanced_analytics(self, trades: List[Trade]) -> Dict[str, TradeAnalytics]:
        """Calculate advanced analytics for trades grouped by symbol"""
        symbol_analytics = {}
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in trades:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = []
            trades_by_symbol[trade.symbol].append(trade)
        
        for symbol, symbol_trades in trades_by_symbol.items():
            analytics = await self._calculate_symbol_analytics(symbol, symbol_trades)
            symbol_analytics[symbol] = analytics
        
        return symbol_analytics
    
    async def _calculate_symbol_analytics(self, symbol: str, trades: List[Trade]) -> TradeAnalytics:
        """Calculate analytics for a specific symbol"""
        if not trades:
            return self._empty_analytics(symbol)
        
        # Sort trades by execution time
        sorted_trades = sorted(trades, key=lambda t: t.execution_time)
        
        # Basic metrics
        total_trades = len(trades)
        total_volume = sum(trade.quantity for trade in trades)
        
        # Calculate P&L using position tracking
        positions = self._track_positions(sorted_trades)
        total_pnl = sum(pos['realized_pnl'] for pos in positions.values())
        
        # Performance metrics
        win_rate = self._calculate_win_rate(positions)
        profit_factor = self._calculate_profit_factor(positions)
        
        # Risk metrics
        returns = self._calculate_returns(sorted_trades)
        max_drawdown = self._calculate_max_drawdown(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if returns else None
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown) if returns and max_drawdown > 0 else None
        sortino_ratio = self._calculate_sortino_ratio(returns) if returns else None
        
        # Trade patterns
        avg_holding_period = self._calculate_average_holding_period(sorted_trades)
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(positions)
        largest_win, largest_loss = self._calculate_largest_trades(positions)
        
        # Risk-adjusted return
        risk_adjusted_return = float(total_pnl / max_drawdown) if max_drawdown > 0 else 0.0
        
        return TradeAnalytics(
            symbol=symbol,
            period_start=sorted_trades[0].execution_time,
            period_end=sorted_trades[-1].execution_time,
            total_trades=total_trades,
            total_volume=total_volume,
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            average_holding_period=avg_holding_period,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
            largest_win=largest_win,
            largest_loss=largest_loss,
            risk_adjusted_return=risk_adjusted_return
        )
    
    def _track_positions(self, trades: List[Trade]) -> Dict[str, Dict]:
        """Track positions and calculate P&L"""
        positions = {}
        
        for trade in trades:
            key = f"{trade.symbol}_{trade.venue}"
            if key not in positions:
                positions[key] = {
                    'quantity': Decimal('0'),
                    'cost_basis': Decimal('0'),
                    'realized_pnl': Decimal('0'),
                    'trades': []
                }
            
            pos = positions[key]
            pos['trades'].append(trade)
            
            if trade.side.upper() in ['BUY', 'COVER']:
                # Long position or covering short
                if pos['quantity'] < 0:  # Covering short
                    cover_qty = min(abs(pos['quantity']), trade.quantity)
                    if pos['quantity'] != 0:
                        avg_cost = abs(pos['cost_basis'] / pos['quantity'])
                        pnl = cover_qty * (avg_cost - trade.price)
                        pos['realized_pnl'] += pnl
                    
                    pos['quantity'] += cover_qty
                    remaining = trade.quantity - cover_qty
                    if remaining > 0:
                        pos['quantity'] += remaining
                        pos['cost_basis'] += remaining * trade.price
                else:  # Adding to long
                    pos['quantity'] += trade.quantity
                    pos['cost_basis'] += trade.quantity * trade.price
            
            else:  # SELL or SHORT
                if pos['quantity'] > 0:  # Selling long
                    sell_qty = min(pos['quantity'], trade.quantity)
                    if pos['quantity'] != 0:
                        avg_cost = pos['cost_basis'] / pos['quantity']
                        pnl = sell_qty * (trade.price - avg_cost)
                        pos['realized_pnl'] += pnl
                    
                    pos['quantity'] -= sell_qty
                    remaining = trade.quantity - sell_qty
                    if remaining > 0:
                        pos['quantity'] -= remaining
                        pos['cost_basis'] -= remaining * trade.price
                else:  # Adding to short
                    pos['quantity'] -= trade.quantity
                    pos['cost_basis'] -= trade.quantity * trade.price
        
        return positions
    
    def _calculate_win_rate(self, positions: Dict) -> float:
        """Calculate win rate from positions"""
        winning_positions = sum(1 for pos in positions.values() if pos['realized_pnl'] > 0)
        total_positions = len([pos for pos in positions.values() if pos['realized_pnl'] != 0])
        return (winning_positions / total_positions * 100) if total_positions > 0 else 0.0
    
    def _calculate_profit_factor(self, positions: Dict) -> float:
        """Calculate profit factor"""
        gross_profit = sum(pos['realized_pnl'] for pos in positions.values() if pos['realized_pnl'] > 0)
        gross_loss = abs(sum(pos['realized_pnl'] for pos in positions.values() if pos['realized_pnl'] < 0))
        return float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    def _calculate_returns(self, trades: List[Trade]) -> List[float]:
        """Calculate daily returns"""
        if len(trades) < 2:
            return []
        
        # Group trades by date and calculate daily P&L
        daily_pnl = {}
        positions = self._track_positions(trades)
        
        # This is a simplified calculation - would need more sophisticated approach
        # for accurate daily returns in a real implementation
        total_pnl = sum(pos['realized_pnl'] for pos in positions.values())
        days = (trades[-1].execution_time - trades[0].execution_time).days or 1
        avg_daily_return = float(total_pnl) / days
        
        return [avg_daily_return] * days  # Simplified
    
    def _calculate_max_drawdown(self, returns: List[float]) -> Decimal:
        """Calculate maximum drawdown"""
        if not returns:
            return Decimal('0')
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return Decimal(str(np.max(drawdown)))
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: Decimal) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown <= 0:
            return 0.0
        
        annualized_return = np.mean(returns) * 252  # Annualize
        return float(annualized_return / float(max_drawdown))
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return float(np.mean(excess_returns) / downside_deviation) if downside_deviation > 0 else 0.0
    
    def _calculate_average_holding_period(self, trades: List[Trade]) -> timedelta:
        """Calculate average holding period"""
        if len(trades) < 2:
            return timedelta(0)
        
        # Simplified calculation - would need position tracking for accuracy
        total_duration = trades[-1].execution_time - trades[0].execution_time
        return total_duration / len(trades)
    
    def _calculate_consecutive_trades(self, positions: Dict) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        pnls = [pos['realized_pnl'] for pos in positions.values()]
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def _calculate_largest_trades(self, positions: Dict) -> Tuple[Decimal, Decimal]:
        """Calculate largest win and loss"""
        pnls = [pos['realized_pnl'] for pos in positions.values()]
        
        largest_win = max(pnls) if pnls else Decimal('0')
        largest_loss = min(pnls) if pnls else Decimal('0')
        
        return largest_win, largest_loss
    
    def _empty_analytics(self, symbol: str) -> TradeAnalytics:
        """Return empty analytics for symbol with no trades"""
        now = datetime.now()
        return TradeAnalytics(
            symbol=symbol,
            period_start=now,
            period_end=now,
            total_trades=0,
            total_volume=Decimal('0'),
            total_pnl=Decimal('0'),
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=Decimal('0'),
            sharpe_ratio=None,
            calmar_ratio=None,
            sortino_ratio=None,
            average_holding_period=timedelta(0),
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            largest_win=Decimal('0'),
            largest_loss=Decimal('0'),
            risk_adjusted_return=0.0
        )
    
    async def analyze_market_impact(self, trades: List[Trade]) -> List[MarketImpactAnalysis]:
        """Analyze market impact of trades using historical data"""
        impact_analyses = []
        
        for trade in trades:
            try:
                analysis = await self._analyze_single_trade_impact(trade)
                if analysis:
                    impact_analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing market impact for trade {trade.trade_id}: {e}")
        
        return impact_analyses
    
    async def _analyze_single_trade_impact(self, trade: Trade) -> Optional[MarketImpactAnalysis]:
        """Analyze market impact for a single trade"""
        try:
            # Get market data around trade execution time
            start_time = trade.execution_time - timedelta(minutes=5)
            end_time = trade.execution_time + timedelta(minutes=5)
            
            # This would require integration with the historical data service
            # For now, we'll return a placeholder analysis
            
            # In a full implementation, we would:
            # 1. Query historical tick/quote data around the execution time
            # 2. Calculate price impact based on bid/ask spread and execution price
            # 3. Analyze volume impact based on market depth
            # 4. Determine execution quality based on price improvement/slippage
            
            return MarketImpactAnalysis(
                symbol=trade.symbol,
                trade_id=trade.trade_id,
                execution_price=trade.price,
                market_price_before=None,  # Would be populated from historical data
                market_price_after=None,   # Would be populated from historical data
                price_impact=None,         # Would be calculated
                price_impact_bps=None,     # Would be calculated in basis points
                volume_impact=None,        # Would be calculated
                execution_quality='unknown'  # Would be determined based on analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market impact for trade {trade.trade_id}: {e}")
            return None
    
    async def export_to_nautilus_format(self, trades: List[Trade], output_path: str) -> bool:
        """Export trades to NautilusTrader compatible format"""
        try:
            # Convert trades to NautilusTrader execution format
            executions = []
            
            for trade in trades:
                execution = {
                    'venue': trade.venue,
                    'instrument_id': trade.symbol,
                    'exec_id': trade.execution_id or trade.trade_id,
                    'order_id': trade.order_id,
                    'side': trade.side,
                    'quantity': str(trade.quantity),
                    'price': str(trade.price),
                    'commission': str(trade.commission),
                    'timestamp': int(trade.execution_time.timestamp() * 1_000_000_000),  # nanoseconds
                    'account_id': trade.account_id,
                    'strategy_id': trade.strategy or 'unspecified'
                }
                executions.append(execution)
            
            # Export as JSON for NautilusTrader compatibility
            import json
            with open(output_path, 'w') as f:
                json.dump(executions, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(executions)} trades to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Nautilus format: {e}")
            return False


# Global processor instance
_trade_processor: Optional[TradeDataProcessor] = None

def get_trade_processor() -> TradeDataProcessor:
    """Get or create trade data processor singleton"""
    global _trade_processor
    
    if _trade_processor is None:
        _trade_processor = TradeDataProcessor()
    
    return _trade_processor