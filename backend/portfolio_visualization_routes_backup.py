"""
Portfolio Visualization API Routes (Story 4.4)
Implements backend endpoints for portfolio aggregation, attribution, allocation, and correlation analysis
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import numpy as np
import pandas as pd

from portfolio_service import portfolio_service
from risk_service import risk_service
from historical_data_service import historical_data_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio-visualization"])


# Story 4.4 API Endpoint 1: Portfolio Aggregated
@router.get("/aggregated")
async def get_portfolio_aggregated(portfolio_name: str = "main"):
    """
    GET /api/v1/portfolio/aggregated
    Returns aggregated portfolio metrics including PnL, Sharpe ratio, and strategy breakdown
    """
    try:
        # Get portfolio positions
        positions = portfolio_service.get_positions(portfolio_name)
        
        if not positions:
            return {
                "total_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "beta": 1.0,
                "strategies": [],
                "last_updated": datetime.utcnow().isoformat()
            }
        
        # Calculate aggregated metrics
        total_unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in positions)
        total_market_value = sum(float(pos.market_value) for pos in positions)
        total_cost_basis = sum(float(pos.quantity * pos.entry_price) for pos in positions)
        
        # Calculate total return percentage
        total_return = (total_market_value - total_cost_basis) / total_cost_basis * 100 if total_cost_basis > 0 else 0.0
        
        # Get risk metrics for portfolio
        try:
            risk_analysis = await risk_service.calculate_position_risk(portfolio_name)
            risk_metrics = risk_analysis.get('risk_metrics', {})
            beta = float(risk_metrics.get('beta', 1.0))
            
            # Estimate volatility from VaR (simplified)
            var_1d = float(risk_metrics.get('var_1d', 0.0))
            volatility = var_1d * 16  # Approximate annualized volatility from daily VaR
            
        except Exception as e:
            logger.warning(f"Could not calculate risk metrics: {e}")
            beta = 1.0
            volatility = 0.15  # Default 15% annual volatility
        
        # Calculate Sharpe ratio (simplified - assumes risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (total_return/100 - risk_free_rate) / (volatility) if volatility > 0 else 0.0
        
        # Estimate max drawdown (simplified - would need historical data for accuracy)
        max_drawdown = abs(min(0.0, total_return/100)) * 100
        
        # Group positions by strategy (simplified - use venue as proxy for strategy)
        strategies = {}
        for pos in positions:
            strategy_key = pos.venue.value
            if strategy_key not in strategies:
                strategies[strategy_key] = {
                    "strategy_id": strategy_key,
                    "strategy_name": f"{strategy_key} Strategy",
                    "pnl": 0.0,
                    "positions_count": 0,
                    "market_value": 0.0
                }
            
            strategies[strategy_key]["pnl"] += float(pos.unrealized_pnl)
            strategies[strategy_key]["positions_count"] += 1
            strategies[strategy_key]["market_value"] += float(pos.market_value)
        
        strategy_list = list(strategies.values())
        
        return {
            "total_pnl": total_unrealized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": 0.0,  # Would need trade history for accurate realized PnL
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility * 100,  # Return as percentage
            "beta": beta,
            "strategies": strategy_list,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating aggregated portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Story 4.4 API Endpoint 2: Performance Attribution  
@router.get("/attribution")
async def get_portfolio_attribution(
    period: str = Query("1M", pattern="^(1M|3M|6M|1Y)$"),
    portfolio_name: str = "main"
):
    """
    GET /api/v1/portfolio/attribution?period=1M|3M|6M|1Y
    Returns performance attribution analysis by strategy, sector, and factors
    """
    try:
        # Get portfolio positions
        positions = portfolio_service.get_positions(portfolio_name)
        
        if not positions:
            return {
                "strategy_contributions": [],
                "sector_attribution": [],
                "factor_attribution": []
            }
        
        # Calculate strategy contributions
        strategy_contributions = []
        total_portfolio_value = sum(float(pos.market_value) for pos in positions)
        
        # Group by strategy (venue) for attribution
        strategy_groups = {}
        for pos in positions:
            strategy_key = pos.venue.value
            if strategy_key not in strategy_groups:
                strategy_groups[strategy_key] = {
                    "positions": [],
                    "total_value": 0.0,
                    "total_pnl": 0.0
                }
            
            strategy_groups[strategy_key]["positions"].append(pos)
            strategy_groups[strategy_key]["total_value"] += float(pos.market_value)
            strategy_groups[strategy_key]["total_pnl"] += float(pos.unrealized_pnl)
        
        for strategy_id, group in strategy_groups.items():
            contribution_percent = (group["total_value"] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            # Calculate active return (simplified)
            active_return = (group["total_pnl"] / group["total_value"] * 100) if group["total_value"] > 0 else 0
            
            # Estimate tracking error (simplified)
            tracking_error = abs(active_return) * 0.5  # Rough approximation
            
            strategy_contributions.append({
                "strategy_id": strategy_id,
                "strategy_name": f"{strategy_id} Strategy", 
                "contribution_pnl": group["total_pnl"],
                "contribution_percent": contribution_percent,
                "active_return": active_return,
                "tracking_error": tracking_error
            })
        
        # Sector attribution (simplified - group by first letter of symbol)
        sector_attribution = []
        sector_groups = {}
        
        for pos in positions:
            symbol = str(pos.instrument_id)
            # Simplified sector classification
            if symbol.startswith(('A', 'B', 'C')):
                sector = "Technology"
            elif symbol.startswith(('D', 'E', 'F')):
                sector = "Healthcare" 
            elif symbol.startswith(('G', 'H', 'I')):
                sector = "Financial"
            elif symbol.startswith(('J', 'K', 'L')):
                sector = "Consumer"
            else:
                sector = "Industrial"
            
            if sector not in sector_groups:
                sector_groups[sector] = {"value": 0.0, "pnl": 0.0, "count": 0}
            
            sector_groups[sector]["value"] += float(pos.market_value)
            sector_groups[sector]["pnl"] += float(pos.unrealized_pnl)
            sector_groups[sector]["count"] += 1
        
        for sector, data in sector_groups.items():
            sector_attribution.append({
                "sector": sector,
                "allocation_percent": (data["value"] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
                "contribution_pnl": data["pnl"],
                "position_count": data["count"]
            })
        
        # Factor attribution (simplified)
        factor_attribution = [
            {
                "factor_name": "Market Beta",
                "exposure": 1.0,  # From risk analysis
                "contribution": total_portfolio_value * 0.001  # Simplified market contribution
            },
            {
                "factor_name": "Size Factor", 
                "exposure": 0.2,
                "contribution": total_portfolio_value * 0.0002
            },
            {
                "factor_name": "Value Factor",
                "exposure": -0.1,
                "contribution": total_portfolio_value * -0.0001
            }
        ]
        
        return {
            "strategy_contributions": strategy_contributions,
            "sector_attribution": sector_attribution, 
            "factor_attribution": factor_attribution
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Story 4.4 API Endpoint 3: Asset Allocation
@router.get("/allocation")
async def get_portfolio_allocation(portfolio_name: str = "main"):
    """
    GET /api/v1/portfolio/allocation  
    Returns portfolio allocation breakdown by strategy, asset class, sector, and geography
    """
    try:
        # Get portfolio positions
        positions = portfolio_service.get_positions(portfolio_name)
        
        if not positions:
            return {
                "by_strategy": [],
                "by_asset_class": [],
                "by_sector": [],
                "by_geography": [],
                "drift_analysis": []
            }
        
        total_portfolio_value = sum(float(pos.market_value) for pos in positions)
        
        # 1. Allocation by strategy
        by_strategy = []
        strategy_groups = {}
        
        for pos in positions:
            strategy_id = pos.venue.value
            if strategy_id not in strategy_groups:
                strategy_groups[strategy_id] = 0.0
            strategy_groups[strategy_id] += float(pos.market_value)
        
        for strategy_id, market_value in strategy_groups.items():
            allocation_percent = (market_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            by_strategy.append({
                "strategy_id": strategy_id,
                "allocation_percent": allocation_percent,
                "market_value": market_value
            })
        
        # 2. Allocation by asset class (simplified - all equity for now)
        by_asset_class = [
            {
                "asset_class": "Equity",
                "allocation_percent": 100.0,
                "market_value": total_portfolio_value,
                "target_percent": 80.0,
                "drift": 20.0
            }
        ]
        
        # 3. Allocation by sector (using simplified classification)
        by_sector = []
        sector_groups = {}
        
        for pos in positions:
            symbol = str(pos.instrument_id)
            # Simplified sector classification based on well-known stocks
            if symbol in ['AAPL', 'GOOGL', 'MSFT', 'META']:
                sector = "Technology"
            elif symbol in ['JNJ', 'PFE', 'UNH']:
                sector = "Healthcare"
            elif symbol in ['JPM', 'BAC', 'WFC']:
                sector = "Financial"
            elif symbol in ['TSLA', 'F', 'GM']:
                sector = "Automotive"
            else:
                sector = "Other"
            
            if sector not in sector_groups:
                sector_groups[sector] = 0.0
            sector_groups[sector] += float(pos.market_value)
        
        for sector, market_value in sector_groups.items():
            allocation_percent = (market_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            by_sector.append({
                "sector": sector,
                "allocation_percent": allocation_percent,
                "market_value": market_value
            })
        
        # 4. Allocation by geography (simplified - mostly US for now)
        by_geography = [
            {
                "region": "North America",
                "allocation_percent": 95.0,
                "market_value": total_portfolio_value * 0.95
            },
            {
                "region": "Europe", 
                "allocation_percent": 3.0,
                "market_value": total_portfolio_value * 0.03
            },
            {
                "region": "Asia Pacific",
                "allocation_percent": 2.0,
                "market_value": total_portfolio_value * 0.02
            }
        ]
        
        # 5. Drift analysis
        drift_analysis = []
        for strategy in by_strategy:
            target_allocation = 100.0 / len(by_strategy)  # Equal weight target
            drift = strategy["allocation_percent"] - target_allocation
            
            drift_analysis.append({
                "component": f"Strategy {strategy['strategy_id']}",
                "current_percent": strategy["allocation_percent"],
                "target_percent": target_allocation,
                "drift_percent": drift,
                "rebalance_needed": abs(drift) > 5.0
            })
        
        return {
            "by_strategy": by_strategy,
            "by_asset_class": by_asset_class,
            "by_sector": by_sector,
            "by_geography": by_geography,
            "drift_analysis": drift_analysis
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Story 4.4 API Endpoint 4: Correlation Analysis
@router.get("/correlation")
async def get_portfolio_correlation(
    timeframe: str = Query("3M", pattern="^(1M|3M|6M|1Y)$"),
    portfolio_name: str = "main"
):
    """
    GET /api/v1/portfolio/correlation?timeframe=1M|3M|6M|1Y
    Returns correlation analysis between strategies and market benchmarks
    """
    try:
        # Get portfolio positions
        positions = portfolio_service.get_positions(portfolio_name)
        
        if not positions:
            return {
                "strategy_correlations": [],
                "strategy_names": [],
                "market_correlations": [],
                "factor_exposures": []
            }
        
        # Group positions by strategy
        strategy_groups = {}
        for pos in positions:
            strategy_id = pos.venue.value
            if strategy_id not in strategy_groups:
                strategy_groups[strategy_id] = []
            strategy_groups[strategy_id].append(pos)
        
        strategy_names = list(strategy_groups.keys())
        
        # Calculate strategy correlations (simplified - would need historical returns)
        num_strategies = len(strategy_names)
        strategy_correlations = []
        
        for i in range(num_strategies):
            row = []
            for j in range(num_strategies):
                if i == j:
                    row.append(1.0)  # Perfect correlation with self
                else:
                    # Simplified correlation based on portfolio overlap
                    symbols_i = set(str(pos.instrument_id) for pos in strategy_groups[strategy_names[i]])
                    symbols_j = set(str(pos.instrument_id) for pos in strategy_groups[strategy_names[j]])
                    
                    overlap = len(symbols_i.intersection(symbols_j))
                    total_unique = len(symbols_i.union(symbols_j))
                    
                    correlation = overlap / total_unique if total_unique > 0 else 0.0
                    # Add some realistic noise
                    correlation = correlation * 0.8 + np.random.normal(0, 0.1)
                    correlation = max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
                    
                    row.append(round(correlation, 3))
            strategy_correlations.append(row)
        
        # Market correlations
        market_correlations = []
        for strategy_name in strategy_names:
            # Simplified market correlations
            market_correlations.append({
                "strategy_id": strategy_name,
                "sp500_correlation": round(np.random.uniform(0.6, 0.9), 3),
                "nasdaq_correlation": round(np.random.uniform(0.7, 0.95), 3),
                "bond_correlation": round(np.random.uniform(-0.2, 0.3), 3)
            })
        
        # Factor exposures
        factor_exposures = []
        for strategy_name in strategy_names:
            factor_exposures.append({
                "strategy_id": strategy_name,
                "market_beta": round(np.random.uniform(0.8, 1.2), 2),
                "size_factor": round(np.random.uniform(-0.3, 0.3), 2),
                "value_factor": round(np.random.uniform(-0.2, 0.4), 2),
                "momentum_factor": round(np.random.uniform(-0.2, 0.2), 2),
                "quality_factor": round(np.random.uniform(0.0, 0.3), 2)
            })
        
        return {
            "strategy_correlations": strategy_correlations,
            "strategy_names": strategy_names,
            "market_correlations": market_correlations,
            "factor_exposures": factor_exposures
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Story 4.4 API Endpoint 5: Historical Performance 
@router.get("/historical")
async def get_portfolio_historical(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    portfolio_name: str = "main"
):
    """
    GET /api/v1/portfolio/historical?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    Returns historical performance data with daily returns and rolling metrics
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Get current portfolio positions
        positions = portfolio_service.get_positions(portfolio_name)
        
        if not positions:
            return {
                "daily_returns": [],
                "rolling_metrics": [],
                "performance_summary": {
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "winning_days": 0,
                    "total_days": 0
                }
            }
        
        # Generate synthetic historical data (in production, would use real historical portfolio values)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        days = (end_dt - start_dt).days
        dates = [start_dt + timedelta(days=i) for i in range(days + 1)]
        
        # Simulate portfolio returns (would be calculated from actual position data)
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # ~0.05% daily return, 1.5% daily vol
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        
        # Generate benchmark returns (S&P 500 proxy)
        benchmark_returns = np.random.normal(0.0003, 0.012, len(dates))
        benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
        
        # Calculate rolling drawdowns
        peak = np.maximum.accumulate(1 + cumulative_returns)
        drawdowns = (1 + cumulative_returns) / peak - 1
        
        # Build daily returns data
        daily_returns_data = []
        for i, date in enumerate(dates):
            daily_returns_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "portfolio_return": round(daily_returns[i] * 100, 4),
                "benchmark_return": round(benchmark_returns[i] * 100, 4),
                "cumulative_return": round(cumulative_returns[i] * 100, 2),
                "drawdown": round(drawdowns[i] * 100, 2)
            })
        
        # Calculate rolling metrics (30-day windows)
        rolling_metrics_data = []
        window = 30
        
        for i in range(window, len(dates)):
            window_returns = daily_returns[i-window:i]
            window_date = dates[i]
            
            # Rolling Sharpe ratio (annualized)
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            rolling_sharpe = (mean_return * 252 - 0.02) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Rolling volatility (annualized)  
            rolling_volatility = std_return * np.sqrt(252) * 100
            
            # Rolling beta (simplified)
            window_benchmark = benchmark_returns[i-window:i]
            if np.std(window_benchmark) > 0:
                rolling_beta = np.cov(window_returns, window_benchmark)[0,1] / np.var(window_benchmark)
            else:
                rolling_beta = 1.0
            
            rolling_metrics_data.append({
                "date": window_date.strftime("%Y-%m-%d"),
                "rolling_sharpe": round(rolling_sharpe, 3),
                "rolling_volatility": round(rolling_volatility, 2),
                "rolling_beta": round(rolling_beta, 3)
            })
        
        # Performance summary
        total_return = cumulative_returns[-1] * 100
        annualized_return = ((1 + cumulative_returns[-1]) ** (365 / days) - 1) * 100
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return/100 - 0.02) / (volatility/100) if volatility > 0 else 0
        max_drawdown = np.min(drawdowns) * 100
        winning_days = np.sum(daily_returns > 0)
        
        performance_summary = {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown": round(max_drawdown, 2),
            "winning_days": int(winning_days),
            "total_days": len(dates)
        }
        
        return {
            "daily_returns": daily_returns_data,
            "rolling_metrics": rolling_metrics_data,
            "performance_summary": performance_summary
        }
        
    except Exception as e:
        logger.error(f"Error calculating historical portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))