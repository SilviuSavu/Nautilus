"""
Strategy-Specific Performance Analytics for Sprint 3 Priority 2
Individual strategy performance metrics, comparison, ranking, and alpha/beta calculations
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
from scipy import stats
import json

logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    """Strategy status types"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    BACKTEST = "backtest"

class PerformancePeriod(Enum):
    """Performance analysis periods"""
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    QUARTERLY = "3m"
    YEARLY = "1y"
    ALL_TIME = "all"

@dataclass
class StrategyPerformance:
    """Individual strategy performance metrics"""
    strategy_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    total_pnl: Decimal
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None

@dataclass
class StrategyComparison:
    """Strategy comparison result"""
    comparison_id: str
    strategies: List[str]
    benchmark: Optional[str]
    comparison_period: PerformancePeriod
    metrics_comparison: Dict[str, Dict[str, float]]
    correlation_matrix: np.ndarray
    ranking: List[Tuple[str, float]]  # (strategy_id, total_score)
    best_performer: str
    worst_performer: str
    analysis_timestamp: datetime

@dataclass
class StrategyAttribution:
    """Strategy performance attribution"""
    strategy_id: str
    timestamp: datetime
    instrument_contributions: Dict[str, float]
    sector_contributions: Dict[str, float]
    asset_class_contributions: Dict[str, float]
    top_contributors: List[Tuple[str, float]]  # (instrument_id, contribution)
    top_detractors: List[Tuple[str, float]]
    concentration_metrics: Dict[str, float]

@dataclass
class AlphaBetaAnalysis:
    """Alpha and Beta analysis for strategy"""
    strategy_id: str
    benchmark: str
    analysis_period: PerformancePeriod
    alpha: float
    beta: float
    r_squared: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float
    correlation: float
    observations: int
    calculation_date: datetime

class StrategyAnalytics:
    """
    Strategy-specific performance analytics engine
    Provides comprehensive analysis of individual trading strategies
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days_per_year = 252
        
        # Performance metrics weights for ranking
        self.ranking_weights = {
            'sharpe_ratio': 0.25,
            'total_return': 0.20,
            'max_drawdown': 0.20,  # Negative weight (lower is better)
            'win_rate': 0.15,
            'profit_factor': 0.10,
            'volatility': 0.10      # Negative weight (lower is better)
        }
    
    async def calculate_strategy_performance(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark: Optional[str] = "SPY"
    ) -> StrategyPerformance:
        """
        Calculate comprehensive performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            start_date: Analysis start date
            end_date: Analysis end date
            benchmark: Benchmark symbol for alpha/beta calculation
            
        Returns:
            StrategyPerformance with all calculated metrics
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            # Default to strategy inception date or 1 year ago
            start_date = await self._get_strategy_start_date(strategy_id) or (end_date - timedelta(days=365))
        
        async with self.db_pool.acquire() as conn:
            try:
                # Get strategy basic info
                strategy_info = await self._get_strategy_info(conn, strategy_id)
                
                # Get strategy trades and returns
                trades = await self._get_strategy_trades(conn, strategy_id, start_date, end_date)
                returns = await self._calculate_strategy_returns(trades)
                
                if len(returns) < 5:
                    raise ValueError(f"Insufficient data for strategy {strategy_id}: {len(returns)} observations")
                
                # Calculate basic performance metrics
                returns_array = np.array(returns)
                
                # Total and annualized return
                total_return = float(np.prod(1 + returns_array) - 1) if len(returns_array) > 0 else 0
                days_elapsed = (end_date - start_date).days
                if days_elapsed > 0:
                    annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
                else:
                    annualized_return = total_return
                
                # Volatility (annualized)
                volatility = np.std(returns_array) * np.sqrt(self.trading_days_per_year)
                
                # Sharpe Ratio
                excess_returns = returns_array - (self.risk_free_rate / self.trading_days_per_year)
                sharpe_ratio = (
                    np.mean(excess_returns) / np.std(returns_array) * np.sqrt(self.trading_days_per_year)
                    if np.std(returns_array) > 0 else 0.0
                )
                
                # Sortino Ratio
                downside_returns = returns_array[returns_array < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns_array)
                sortino_ratio = (
                    np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year)
                    if downside_deviation > 0 else 0.0
                )
                
                # Maximum Drawdown
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = float(np.min(drawdowns))
                
                # Calmar Ratio
                calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # Trade-based metrics
                trade_metrics = await self._calculate_trade_metrics(trades)
                
                # Alpha and Beta analysis
                alpha, beta, info_ratio, tracking_error = None, None, None, None
                if benchmark:
                    try:
                        alpha_beta_result = await self.calculate_alpha_beta(
                            strategy_id, benchmark, start_date, end_date
                        )
                        alpha = alpha_beta_result.alpha
                        beta = alpha_beta_result.beta
                        info_ratio = alpha_beta_result.information_ratio
                        tracking_error = alpha_beta_result.tracking_error
                    except Exception as e:
                        self.logger.warning(f"Could not calculate alpha/beta for {strategy_id}: {e}")
                
                performance = StrategyPerformance(
                    strategy_id=strategy_id,
                    strategy_name=strategy_info['name'],
                    start_date=start_date,
                    end_date=end_date,
                    total_return=total_return,
                    annualized_return=annualized_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    max_drawdown=max_drawdown,
                    calmar_ratio=calmar_ratio,
                    win_rate=trade_metrics['win_rate'],
                    profit_factor=trade_metrics['profit_factor'],
                    total_trades=trade_metrics['total_trades'],
                    winning_trades=trade_metrics['winning_trades'],
                    losing_trades=trade_metrics['losing_trades'],
                    avg_trade_duration=trade_metrics['avg_trade_duration'],
                    total_pnl=trade_metrics['total_pnl'],
                    alpha=alpha,
                    beta=beta,
                    information_ratio=info_ratio,
                    tracking_error=tracking_error
                )
                
                # Store performance metrics
                await self._store_strategy_performance(conn, performance)
                
                return performance
                
            except Exception as e:
                self.logger.error(f"Error calculating strategy performance: {e}")
                raise
    
    async def compare_strategies(
        self,
        strategy_ids: List[str],
        period: PerformancePeriod = PerformancePeriod.QUARTERLY,
        benchmark: Optional[str] = "SPY"
    ) -> StrategyComparison:
        """
        Compare multiple strategies across key performance metrics
        
        Args:
            strategy_ids: List of strategy identifiers
            period: Comparison time period
            benchmark: Benchmark for comparison
            
        Returns:
            StrategyComparison with detailed comparison results
        """
        async with self.db_pool.acquire() as conn:
            try:
                end_date = datetime.utcnow()
                start_date = self._get_period_start_date(end_date, period)
                
                # Calculate performance for each strategy
                strategy_performances = {}
                strategy_returns = {}
                
                for strategy_id in strategy_ids:
                    try:
                        performance = await self.calculate_strategy_performance(
                            strategy_id, start_date, end_date, benchmark
                        )
                        strategy_performances[strategy_id] = performance
                        
                        # Get returns for correlation analysis
                        trades = await self._get_strategy_trades(conn, strategy_id, start_date, end_date)
                        returns = await self._calculate_strategy_returns(trades)
                        strategy_returns[strategy_id] = returns
                        
                    except Exception as e:
                        self.logger.warning(f"Could not calculate performance for {strategy_id}: {e}")
                        continue
                
                if len(strategy_performances) < 2:
                    raise ValueError("Need at least 2 strategies for comparison")
                
                # Build metrics comparison matrix
                metrics_comparison = {}
                metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                          'max_drawdown', 'win_rate', 'profit_factor']
                
                for metric in metrics:
                    metrics_comparison[metric] = {}
                    for strategy_id, perf in strategy_performances.items():
                        metrics_comparison[metric][strategy_id] = getattr(perf, metric)
                
                # Calculate correlation matrix
                correlation_matrix = await self._calculate_strategy_correlations(strategy_returns)
                
                # Rank strategies
                ranking = await self._rank_strategies(strategy_performances)
                
                best_performer = ranking[0][0] if ranking else None
                worst_performer = ranking[-1][0] if ranking else None
                
                comparison = StrategyComparison(
                    comparison_id=f"comp_{int(datetime.utcnow().timestamp())}",
                    strategies=list(strategy_performances.keys()),
                    benchmark=benchmark,
                    comparison_period=period,
                    metrics_comparison=metrics_comparison,
                    correlation_matrix=correlation_matrix,
                    ranking=ranking,
                    best_performer=best_performer,
                    worst_performer=worst_performer,
                    analysis_timestamp=datetime.utcnow()
                )
                
                # Store comparison results
                await self._store_strategy_comparison(conn, comparison)
                
                return comparison
                
            except Exception as e:
                self.logger.error(f"Error comparing strategies: {e}")
                raise
    
    async def calculate_alpha_beta(
        self,
        strategy_id: str,
        benchmark: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AlphaBetaAnalysis:
        """
        Calculate alpha and beta analysis for a strategy against benchmark
        
        Args:
            strategy_id: Strategy identifier
            benchmark: Benchmark symbol
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            AlphaBetaAnalysis with alpha, beta, and related metrics
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=252)  # 1 year
        
        async with self.db_pool.acquire() as conn:
            try:
                # Get strategy returns
                strategy_trades = await self._get_strategy_trades(conn, strategy_id, start_date, end_date)
                strategy_returns = await self._calculate_strategy_returns(strategy_trades)
                
                # Get benchmark returns
                benchmark_returns = await self._get_benchmark_returns(conn, benchmark, start_date, end_date)
                
                if len(strategy_returns) < 30 or len(benchmark_returns) < 30:
                    raise ValueError("Insufficient data for alpha/beta calculation")
                
                # Align return series
                min_length = min(len(strategy_returns), len(benchmark_returns))
                strat_returns = np.array(strategy_returns[-min_length:])
                bench_returns = np.array(benchmark_returns[-min_length:])
                
                # Linear regression for alpha and beta
                slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, strat_returns)
                
                beta = slope
                alpha = intercept * self.trading_days_per_year  # Annualized alpha
                r_squared = r_value ** 2
                correlation = r_value
                
                # Tracking error and information ratio
                excess_returns = strat_returns - bench_returns
                tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
                information_ratio = (
                    np.mean(excess_returns) * self.trading_days_per_year / tracking_error
                    if tracking_error > 0 else 0
                )
                
                # Up/Down capture ratios
                up_periods = bench_returns > 0
                down_periods = bench_returns < 0
                
                up_capture = (
                    np.mean(strat_returns[up_periods]) / np.mean(bench_returns[up_periods])
                    if np.sum(up_periods) > 0 and np.mean(bench_returns[up_periods]) != 0 else 0
                )
                
                down_capture = (
                    np.mean(strat_returns[down_periods]) / np.mean(bench_returns[down_periods])
                    if np.sum(down_periods) > 0 and np.mean(bench_returns[down_periods]) != 0 else 0
                )
                
                analysis = AlphaBetaAnalysis(
                    strategy_id=strategy_id,
                    benchmark=benchmark,
                    analysis_period=PerformancePeriod.QUARTERLY,  # Determine from date range
                    alpha=alpha,
                    beta=beta,
                    r_squared=r_squared,
                    information_ratio=information_ratio,
                    tracking_error=tracking_error,
                    up_capture=up_capture,
                    down_capture=down_capture,
                    correlation=correlation,
                    observations=min_length,
                    calculation_date=datetime.utcnow()
                )
                
                # Store analysis
                await self._store_alpha_beta_analysis(conn, analysis)
                
                return analysis
                
            except Exception as e:
                self.logger.error(f"Error calculating alpha/beta: {e}")
                raise
    
    async def calculate_strategy_attribution(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> StrategyAttribution:
        """
        Calculate performance attribution for strategy holdings
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        async with self.db_pool.acquire() as conn:
            try:
                # Get strategy positions and their performance
                attribution_query = """
                    SELECT 
                        t.instrument_id,
                        i.symbol,
                        i.asset_class,
                        i.metadata::json->>'sector' as sector,
                        SUM(t.realized_pnl) as instrument_pnl,
                        COUNT(*) as trade_count,
                        AVG(t.quantity * t.price) as avg_position_size
                    FROM trades t
                    JOIN instruments i ON t.instrument_id = i.instrument_id
                    WHERE t.strategy_id = $1
                    AND t.timestamp >= $2
                    AND t.timestamp <= $3
                    GROUP BY t.instrument_id, i.symbol, i.asset_class, i.metadata::json->>'sector'
                    ORDER BY instrument_pnl DESC
                """
                
                results = await conn.fetch(attribution_query, strategy_id, start_date, end_date)
                
                instrument_contributions = {}
                sector_contributions = {}
                asset_class_contributions = {}
                
                total_pnl = sum(float(row['instrument_pnl'] or 0) for row in results)
                
                for row in results:
                    instrument_id = row['instrument_id']
                    symbol = row['symbol']
                    asset_class = row['asset_class']
                    sector = row['sector'] or 'Unknown'
                    pnl = float(row['instrument_pnl'] or 0)
                    
                    # Contribution as percentage of total P&L
                    contribution = (pnl / total_pnl * 100) if total_pnl != 0 else 0
                    
                    instrument_contributions[f"{symbol} ({instrument_id})"] = contribution
                    
                    # Aggregate by sector
                    sector_contributions[sector] = sector_contributions.get(sector, 0) + contribution
                    
                    # Aggregate by asset class
                    asset_class_contributions[asset_class] = (
                        asset_class_contributions.get(asset_class, 0) + contribution
                    )
                
                # Top contributors and detractors
                sorted_contributions = sorted(
                    instrument_contributions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                top_contributors = sorted_contributions[:5]
                top_detractors = sorted_contributions[-5:]
                
                # Concentration metrics
                contributions = list(instrument_contributions.values())
                concentration_metrics = self._calculate_concentration_metrics(contributions)
                
                attribution = StrategyAttribution(
                    strategy_id=strategy_id,
                    timestamp=datetime.utcnow(),
                    instrument_contributions=instrument_contributions,
                    sector_contributions=sector_contributions,
                    asset_class_contributions=asset_class_contributions,
                    top_contributors=top_contributors,
                    top_detractors=top_detractors,
                    concentration_metrics=concentration_metrics
                )
                
                # Store attribution
                await self._store_strategy_attribution(conn, attribution)
                
                return attribution
                
            except Exception as e:
                self.logger.error(f"Error calculating strategy attribution: {e}")
                raise
    
    async def get_strategy_ranking(
        self,
        period: PerformancePeriod = PerformancePeriod.QUARTERLY,
        min_trades: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get ranking of all strategies by performance score
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get all active strategies
                strategies_query = """
                    SELECT DISTINCT strategy_id, strategy_name
                    FROM strategies
                    WHERE status = 'active'
                """
                
                strategies = await conn.fetch(strategies_query)
                
                strategy_rankings = []
                end_date = datetime.utcnow()
                start_date = self._get_period_start_date(end_date, period)
                
                for strategy_row in strategies:
                    strategy_id = strategy_row['strategy_id']
                    
                    try:
                        # Check if strategy has enough trades
                        trade_count = await self._get_strategy_trade_count(
                            conn, strategy_id, start_date, end_date
                        )
                        
                        if trade_count < min_trades:
                            continue
                        
                        # Calculate performance
                        performance = await self.calculate_strategy_performance(
                            strategy_id, start_date, end_date
                        )
                        
                        # Calculate composite score
                        score = self._calculate_strategy_score(performance)
                        
                        strategy_rankings.append({
                            'strategy_id': strategy_id,
                            'strategy_name': performance.strategy_name,
                            'score': score,
                            'total_return': performance.total_return,
                            'sharpe_ratio': performance.sharpe_ratio,
                            'max_drawdown': performance.max_drawdown,
                            'win_rate': performance.win_rate,
                            'total_trades': performance.total_trades,
                            'total_pnl': float(performance.total_pnl)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Could not rank strategy {strategy_id}: {e}")
                        continue
                
                # Sort by score (descending)
                strategy_rankings.sort(key=lambda x: x['score'], reverse=True)
                
                # Add rank
                for i, strategy in enumerate(strategy_rankings):
                    strategy['rank'] = i + 1
                
                return strategy_rankings
                
            except Exception as e:
                self.logger.error(f"Error getting strategy ranking: {e}")
                raise
    
    # Helper methods
    
    async def _get_strategy_info(self, conn: asyncpg.Connection, strategy_id: str) -> Dict[str, Any]:
        """Get basic strategy information"""
        query = """
            SELECT strategy_id, strategy_name, created_at, status
            FROM strategies
            WHERE strategy_id = $1
        """
        
        result = await conn.fetchrow(query, strategy_id)
        if not result:
            return {'strategy_id': strategy_id, 'name': strategy_id, 'created_at': datetime.utcnow(), 'status': 'unknown'}
        
        return {
            'strategy_id': result['strategy_id'],
            'name': result['strategy_name'],
            'created_at': result['created_at'],
            'status': result['status']
        }
    
    async def _get_strategy_start_date(self, strategy_id: str) -> Optional[datetime]:
        """Get strategy inception date"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT MIN(timestamp) as start_date
                FROM trades
                WHERE strategy_id = $1
            """
            
            result = await conn.fetchrow(query, strategy_id)
            return result['start_date'] if result else None
    
    async def _get_strategy_trades(
        self,
        conn: asyncpg.Connection,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get strategy trades for analysis"""
        query = """
            SELECT 
                trade_id,
                instrument_id,
                timestamp,
                quantity,
                price,
                side,
                realized_pnl,
                commission,
                strategy_id
            FROM trades
            WHERE strategy_id = $1
            AND timestamp >= $2
            AND timestamp <= $3
            ORDER BY timestamp
        """
        
        results = await conn.fetch(query, strategy_id, start_date, end_date)
        return [dict(row) for row in results]
    
    async def _calculate_strategy_returns(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns from trades"""
        if not trades:
            return []
        
        # Group trades by day and calculate daily P&L
        daily_pnl = {}
        
        for trade in trades:
            trade_date = trade['timestamp'].date()
            pnl = float(trade['realized_pnl'] or 0)
            
            if trade_date not in daily_pnl:
                daily_pnl[trade_date] = 0
            daily_pnl[trade_date] += pnl
        
        # Convert P&L to returns (simplified - assumes constant capital base)
        returns = []
        for date in sorted(daily_pnl.keys()):
            pnl = daily_pnl[date]
            # Normalize by assumed capital base
            returns.append(pnl / 100000)  # Adjust this based on actual capital
        
        return returns
    
    async def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-based performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                'total_pnl': Decimal('0')
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if float(t['realized_pnl'] or 0) > 0)
        losing_trades = sum(1 for t in trades if float(t['realized_pnl'] or 0) < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_wins = sum(float(t['realized_pnl'] or 0) for t in trades if float(t['realized_pnl'] or 0) > 0)
        total_losses = abs(sum(float(t['realized_pnl'] or 0) for t in trades if float(t['realized_pnl'] or 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Average trade duration (simplified)
        avg_trade_duration = 1  # Would need position open/close tracking for accurate calculation
        
        total_pnl = Decimal(str(sum(float(t['realized_pnl'] or 0) for t in trades)))
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_pnl': total_pnl
        }
    
    async def _calculate_strategy_correlations(
        self,
        strategy_returns: Dict[str, List[float]]
    ) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        if len(strategy_returns) < 2:
            return np.array([[1.0]])
        
        # Align return series
        strategy_ids = list(strategy_returns.keys())
        min_length = min(len(returns) for returns in strategy_returns.values())
        
        if min_length < 10:
            # Return identity matrix if insufficient data
            n = len(strategy_ids)
            return np.eye(n)
        
        aligned_returns = []
        for strategy_id in strategy_ids:
            returns = strategy_returns[strategy_id][-min_length:]
            aligned_returns.append(returns)
        
        return np.corrcoef(aligned_returns)
    
    async def _rank_strategies(
        self,
        strategy_performances: Dict[str, StrategyPerformance]
    ) -> List[Tuple[str, float]]:
        """Rank strategies using composite scoring"""
        rankings = []
        
        for strategy_id, performance in strategy_performances.items():
            score = self._calculate_strategy_score(performance)
            rankings.append((strategy_id, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _calculate_strategy_score(self, performance: StrategyPerformance) -> float:
        """Calculate composite performance score"""
        score = 0.0
        
        # Normalize metrics and apply weights
        score += self.ranking_weights['sharpe_ratio'] * min(performance.sharpe_ratio / 2, 1.0)
        score += self.ranking_weights['total_return'] * min(performance.total_return, 1.0)
        score -= self.ranking_weights['max_drawdown'] * abs(performance.max_drawdown)  # Negative impact
        score += self.ranking_weights['win_rate'] * performance.win_rate
        score += self.ranking_weights['profit_factor'] * min(performance.profit_factor / 2, 1.0)
        score -= self.ranking_weights['volatility'] * min(performance.volatility, 1.0)  # Negative impact
        
        return max(score, 0)  # Ensure non-negative score
    
    def _get_period_start_date(self, end_date: datetime, period: PerformancePeriod) -> datetime:
        """Get start date for given period"""
        if period == PerformancePeriod.DAILY:
            return end_date - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            return end_date - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            return end_date - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            return end_date - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            return end_date - timedelta(days=365)
        else:  # ALL_TIME
            return end_date - timedelta(days=365 * 10)  # 10 years max
    
    def _calculate_concentration_metrics(self, contributions: List[float]) -> Dict[str, float]:
        """Calculate concentration metrics for contributions"""
        if not contributions:
            return {}
        
        abs_contributions = [abs(c) for c in contributions]
        total_abs = sum(abs_contributions)
        
        if total_abs == 0:
            return {}
        
        weights = [c / total_abs for c in abs_contributions]
        
        # Herfindahl-Hirschman Index
        hhi = sum(w ** 2 for w in weights)
        
        # Top N concentration
        sorted_weights = sorted(weights, reverse=True)
        top3_concentration = sum(sorted_weights[:3])
        top5_concentration = sum(sorted_weights[:5])
        
        return {
            'herfindahl_index': hhi,
            'top3_concentration': top3_concentration,
            'top5_concentration': top5_concentration,
            'effective_positions': 1 / hhi if hhi > 0 else 0
        }
    
    async def _get_benchmark_returns(
        self,
        conn: asyncpg.Connection,
        benchmark: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get benchmark returns"""
        query = """
            SELECT 
                close_price,
                LAG(close_price) OVER (ORDER BY timestamp_ns) as prev_close
            FROM market_bars
            WHERE instrument_id LIKE $1
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
            AND timeframe = '1d'
            ORDER BY timestamp_ns
        """
        
        start_ns = int(start_date.timestamp() * 1_000_000_000)
        end_ns = int(end_date.timestamp() * 1_000_000_000)
        
        results = await conn.fetch(query, f"%{benchmark}%", start_ns, end_ns)
        
        returns = []
        for result in results:
            if result['prev_close'] and result['close_price']:
                ret = (float(result['close_price']) - float(result['prev_close'])) / float(result['prev_close'])
                returns.append(ret)
        
        return returns
    
    async def _get_strategy_trade_count(
        self,
        conn: asyncpg.Connection,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Get trade count for strategy in period"""
        query = """
            SELECT COUNT(*) as trade_count
            FROM trades
            WHERE strategy_id = $1
            AND timestamp >= $2
            AND timestamp <= $3
        """
        
        result = await conn.fetchrow(query, strategy_id, start_date, end_date)
        return result['trade_count'] if result else 0
    
    # Storage methods
    
    async def _store_strategy_performance(
        self,
        conn: asyncpg.Connection,
        performance: StrategyPerformance
    ):
        """Store strategy performance metrics"""
        query = """
            INSERT INTO strategy_performance (
                strategy_id, start_date, end_date, total_return, annualized_return,
                volatility, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
                win_rate, profit_factor, total_trades, winning_trades, losing_trades,
                avg_trade_duration, total_pnl, alpha, beta, information_ratio,
                tracking_error, calculation_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
            ON CONFLICT (strategy_id, start_date, end_date) DO UPDATE SET
                total_return = EXCLUDED.total_return,
                annualized_return = EXCLUDED.annualized_return,
                volatility = EXCLUDED.volatility,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                sortino_ratio = EXCLUDED.sortino_ratio,
                max_drawdown = EXCLUDED.max_drawdown,
                calmar_ratio = EXCLUDED.calmar_ratio,
                win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                total_trades = EXCLUDED.total_trades,
                winning_trades = EXCLUDED.winning_trades,
                losing_trades = EXCLUDED.losing_trades,
                avg_trade_duration = EXCLUDED.avg_trade_duration,
                total_pnl = EXCLUDED.total_pnl,
                alpha = EXCLUDED.alpha,
                beta = EXCLUDED.beta,
                information_ratio = EXCLUDED.information_ratio,
                tracking_error = EXCLUDED.tracking_error,
                calculation_timestamp = EXCLUDED.calculation_timestamp
        """
        
        await conn.execute(
            query,
            performance.strategy_id,
            performance.start_date,
            performance.end_date,
            performance.total_return,
            performance.annualized_return,
            performance.volatility,
            performance.sharpe_ratio,
            performance.sortino_ratio,
            performance.max_drawdown,
            performance.calmar_ratio,
            performance.win_rate,
            performance.profit_factor,
            performance.total_trades,
            performance.winning_trades,
            performance.losing_trades,
            performance.avg_trade_duration,
            float(performance.total_pnl),
            performance.alpha,
            performance.beta,
            performance.information_ratio,
            performance.tracking_error,
            datetime.utcnow()
        )
    
    async def _store_strategy_comparison(
        self,
        conn: asyncpg.Connection,
        comparison: StrategyComparison
    ):
        """Store strategy comparison results"""
        query = """
            INSERT INTO strategy_comparisons (
                comparison_id, strategies, benchmark, comparison_period,
                metrics_comparison, correlation_matrix, ranking,
                best_performer, worst_performer, analysis_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        await conn.execute(
            query,
            comparison.comparison_id,
            json.dumps(comparison.strategies),
            comparison.benchmark,
            comparison.comparison_period.value,
            json.dumps(comparison.metrics_comparison),
            json.dumps(comparison.correlation_matrix.tolist()),
            json.dumps(comparison.ranking),
            comparison.best_performer,
            comparison.worst_performer,
            comparison.analysis_timestamp
        )
    
    async def _store_alpha_beta_analysis(
        self,
        conn: asyncpg.Connection,
        analysis: AlphaBetaAnalysis
    ):
        """Store alpha/beta analysis results"""
        query = """
            INSERT INTO strategy_alpha_beta (
                strategy_id, benchmark, analysis_period, alpha, beta,
                r_squared, information_ratio, tracking_error, up_capture,
                down_capture, correlation, observations, calculation_date
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (strategy_id, benchmark, analysis_period) DO UPDATE SET
                alpha = EXCLUDED.alpha,
                beta = EXCLUDED.beta,
                r_squared = EXCLUDED.r_squared,
                information_ratio = EXCLUDED.information_ratio,
                tracking_error = EXCLUDED.tracking_error,
                up_capture = EXCLUDED.up_capture,
                down_capture = EXCLUDED.down_capture,
                correlation = EXCLUDED.correlation,
                observations = EXCLUDED.observations,
                calculation_date = EXCLUDED.calculation_date
        """
        
        await conn.execute(
            query,
            analysis.strategy_id,
            analysis.benchmark,
            analysis.analysis_period.value,
            analysis.alpha,
            analysis.beta,
            analysis.r_squared,
            analysis.information_ratio,
            analysis.tracking_error,
            analysis.up_capture,
            analysis.down_capture,
            analysis.correlation,
            analysis.observations,
            analysis.calculation_date
        )
    
    async def _store_strategy_attribution(
        self,
        conn: asyncpg.Connection,
        attribution: StrategyAttribution
    ):
        """Store strategy attribution analysis"""
        query = """
            INSERT INTO strategy_attribution (
                strategy_id, timestamp, instrument_contributions,
                sector_contributions, asset_class_contributions,
                top_contributors, top_detractors, concentration_metrics
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (strategy_id, timestamp) DO UPDATE SET
                instrument_contributions = EXCLUDED.instrument_contributions,
                sector_contributions = EXCLUDED.sector_contributions,
                asset_class_contributions = EXCLUDED.asset_class_contributions,
                top_contributors = EXCLUDED.top_contributors,
                top_detractors = EXCLUDED.top_detractors,
                concentration_metrics = EXCLUDED.concentration_metrics
        """
        
        await conn.execute(
            query,
            attribution.strategy_id,
            attribution.timestamp,
            json.dumps(attribution.instrument_contributions),
            json.dumps(attribution.sector_contributions),
            json.dumps(attribution.asset_class_contributions),
            json.dumps(attribution.top_contributors),
            json.dumps(attribution.top_detractors),
            json.dumps(attribution.concentration_metrics)
        )

# Global instance
strategy_analytics = None

def get_strategy_analytics() -> StrategyAnalytics:
    """Get global strategy analytics instance"""
    global strategy_analytics
    if strategy_analytics is None:
        raise RuntimeError("Strategy analytics not initialized. Call init_strategy_analytics() first.")
    return strategy_analytics

def init_strategy_analytics(db_pool: asyncpg.Pool) -> StrategyAnalytics:
    """Initialize global strategy analytics instance"""
    global strategy_analytics
    strategy_analytics = StrategyAnalytics(db_pool)
    return strategy_analytics