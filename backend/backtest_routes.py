"""
Enhanced Backtest Management API Routes
Extends the nautilus engine routes with specialized backtest endpoints
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import asyncio
import json

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field, validator
import uuid

from auth.middleware import get_current_user
from nautilus_engine_service import get_nautilus_engine_manager, BacktestConfig
from monitoring_service import get_monitoring_service
from rate_limiter import RateLimiter
from backtest_database import get_backtest_database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/nautilus/backtest", tags=["backtest"])

# Rate limiter for backtest operations
backtest_limiter = RateLimiter(max_requests=10, window_seconds=300)  # 10 backtests per 5 minutes

# Extended Models for Enhanced Backtest Functionality

class BacktestResultsRequest(BaseModel):
    """Request model for retrieving backtest results"""
    include_trades: bool = Field(True, description="Include individual trade data")
    include_equity_curve: bool = Field(True, description="Include equity curve data")
    include_metrics: bool = Field(True, description="Include performance metrics")

class BacktestCompareRequest(BaseModel):
    """Request model for comparing multiple backtests"""
    backtest_ids: List[str] = Field(..., min_items=2, max_items=10, description="List of backtest IDs to compare")
    metrics_only: bool = Field(False, description="Compare only performance metrics")

class BacktestExportRequest(BaseModel):
    """Request model for exporting backtest results"""
    format: str = Field(..., regex="^(pdf|excel|csv|json)$", description="Export format")
    include_charts: bool = Field(True, description="Include charts in export")
    include_trades: bool = Field(True, description="Include trade details")

class BacktestFilterRequest(BaseModel):
    """Request model for filtering backtests"""
    status: Optional[str] = Field(None, regex="^(queued|running|completed|failed|cancelled)$")
    strategy_class: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_return: Optional[float] = None
    max_return: Optional[float] = None

# Enhanced Backtest Management Endpoints

@router.post("/start", response_model=Dict[str, Any])
async def start_backtest(
    config: BacktestConfig,
    current_user=Depends(get_current_user)
):
    """
    Start a new backtest with enhanced configuration and validation
    """
    try:
        # Apply rate limiting
        user_id = current_user.get("user_id", "anonymous")
        if not await backtest_limiter.check_rate_limit(user_id):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Maximum 10 backtests per 5 minutes."
            )
        
        engine_manager = get_nautilus_engine_manager()
        monitoring = get_monitoring_service()
        
        # Generate unique backtest ID
        backtest_id = f"backtest-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Enhanced validation
        validation_result = await validate_backtest_config(config)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {', '.join(validation_result['errors'])}"
            )
        
        # Log backtest start
        await monitoring.log_event(
            "enhanced_backtest_started",
            {
                "user_id": user_id,
                "backtest_id": backtest_id,
                "strategy_class": config.strategy_class,
                "instruments": config.instruments,
                "date_range": f"{config.start_date} to {config.end_date}",
                "initial_balance": config.initial_balance
            }
        )
        
        # Start backtest
        result = await engine_manager.run_backtest(backtest_id, config)
        
        return {
            "success": True,
            "backtest_id": backtest_id,
            "status": "queued",
            "message": f"Backtest {backtest_id} started successfully",
            "estimated_completion": await estimate_completion_time(config)
        }
        
    except Exception as e:
        logger.error(f"Error starting enhanced backtest: {str(e)}")
        await monitoring.log_error(
            "enhanced_backtest_start_error",
            str(e),
            {"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@router.get("/results/{backtest_id}")
async def get_backtest_results(
    backtest_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$"),
    include_trades: bool = Query(True),
    include_equity_curve: bool = Query(True),
    include_metrics: bool = Query(True),
    current_user=Depends(get_current_user)
):
    """
    Get comprehensive backtest results with optional data filtering
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        
        # Get basic backtest status
        backtest_status = await engine_manager.get_backtest_status(backtest_id)
        
        if not backtest_status["success"]:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        backtest_data = backtest_status["backtest"]
        
        if backtest_data["status"] != "completed":
            return {
                "backtest_id": backtest_id,
                "status": backtest_data["status"],
                "progress": backtest_data.get("progress", 0),
                "error": backtest_data.get("error"),
                "message": "Backtest not completed yet"
            }
        
        # Build comprehensive results
        results = {
            "backtest_id": backtest_id,
            "status": backtest_data["status"],
            "config": backtest_data["config"],
            "started_at": backtest_data["started_at"],
            "completed_at": backtest_data.get("completed_at"),
            "execution_time": calculate_execution_time(backtest_data)
        }
        
        # Add metrics if requested
        if include_metrics:
            results["metrics"] = await calculate_enhanced_metrics(backtest_id)
        
        # Add trades if requested
        if include_trades:
            results["trades"] = await get_backtest_trades(backtest_id)
        
        # Add equity curve if requested
        if include_equity_curve:
            results["equity_curve"] = await get_equity_curve(backtest_id)
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@router.post("/compare")
async def compare_backtests(
    request: BacktestCompareRequest,
    current_user=Depends(get_current_user)
):
    """
    Compare multiple backtests side-by-side
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        
        comparison_results = {
            "comparison_id": str(uuid.uuid4()),
            "backtests": [],
            "summary": {},
            "relative_performance": {}
        }
        
        # Get results for each backtest
        for backtest_id in request.backtest_ids:
            backtest_status = await engine_manager.get_backtest_status(backtest_id)
            
            if backtest_status["success"] and backtest_status["backtest"]["status"] == "completed":
                backtest_data = {
                    "backtest_id": backtest_id,
                    "metrics": await calculate_enhanced_metrics(backtest_id),
                    "config": backtest_status["backtest"]["config"]
                }
                
                if not request.metrics_only:
                    backtest_data["equity_curve"] = await get_equity_curve(backtest_id)
                
                comparison_results["backtests"].append(backtest_data)
        
        # Calculate comparative statistics
        if len(comparison_results["backtests"]) >= 2:
            comparison_results["summary"] = calculate_comparison_summary(comparison_results["backtests"])
            comparison_results["relative_performance"] = calculate_relative_performance(comparison_results["backtests"])
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Error comparing backtests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare backtests: {str(e)}")

@router.post("/export/{backtest_id}")
async def export_backtest(
    backtest_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$"),
    request: BacktestExportRequest = None,
    current_user=Depends(get_current_user)
):
    """
    Export backtest results in various formats
    """
    try:
        # Get full results
        results = await get_backtest_results(
            backtest_id, 
            include_trades=request.include_trades if request else True,
            include_equity_curve=request.include_charts if request else True,
            include_metrics=True,
            current_user=current_user
        )
        
        export_format = request.format if request else "json"
        
        # Generate export based on format
        if export_format == "json":
            return results
        elif export_format == "csv":
            return await export_to_csv(results)
        elif export_format == "excel":
            return await export_to_excel(results)
        elif export_format == "pdf":
            return await export_to_pdf(results)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
    except Exception as e:
        logger.error(f"Error exporting backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export backtest: {str(e)}")

@router.get("/list")
async def list_backtests(
    status: Optional[str] = Query(None),
    strategy_class: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_user)
):
    """
    List backtests with filtering and pagination
    """
    try:
        db = get_backtest_database()
        user_id = current_user.get("user_id")
        
        # Get backtests from database with filtering
        result = db.list_backtests(
            user_id=user_id,
            status=status,
            strategy_class=strategy_class,
            limit=limit,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing backtests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")

@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: str = Path(..., regex="^[a-zA-Z0-9_-]+$"),
    current_user=Depends(get_current_user)
):
    """
    Delete a backtest and all associated data
    """
    try:
        engine_manager = get_nautilus_engine_manager()
        monitoring = get_monitoring_service()
        
        # Check if backtest exists and is not running
        backtest_status = await engine_manager.get_backtest_status(backtest_id)
        
        if not backtest_status["success"]:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        if backtest_status["backtest"]["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot delete running backtest. Cancel it first.")
        
        # Delete backtest data
        success = await delete_backtest_data(backtest_id)
        
        if success:
            await monitoring.log_event(
                "backtest_deleted",
                {
                    "user_id": current_user.get("user_id"),
                    "backtest_id": backtest_id
                }
            )
            
            return {
                "success": True,
                "message": f"Backtest {backtest_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete backtest data")
        
    except Exception as e:
        logger.error(f"Error deleting backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")

# Helper Functions

async def validate_backtest_config(config: BacktestConfig) -> Dict[str, Any]:
    """Enhanced configuration validation"""
    errors = []
    warnings = []
    
    # Date validation
    try:
        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)
        
        if start_date >= end_date:
            errors.append("Start date must be before end date")
        
        if start_date > datetime.now():
            errors.append("Start date cannot be in the future")
        
        duration_days = (end_date - start_date).days
        if duration_days < 1:
            errors.append("Minimum backtest duration is 1 day")
        elif duration_days > 1825:  # 5 years
            errors.append("Maximum backtest duration is 5 years")
        elif duration_days > 365:  # 1 year
            warnings.append("Long backtests (>1 year) may take significant time to complete")
            
    except ValueError:
        errors.append("Invalid date format. Use YYYY-MM-DD")
    
    # Instrument validation
    if not config.instruments or len(config.instruments) == 0:
        errors.append("At least one instrument must be specified")
    elif len(config.instruments) > 50:
        errors.append("Maximum 50 instruments allowed per backtest")
    
    # Capital validation
    if config.initial_balance < 1000:
        errors.append("Minimum initial balance is $1,000")
    elif config.initial_balance > 100_000_000:
        errors.append("Maximum initial balance is $100,000,000")
    
    # Strategy validation
    if not config.strategy_class:
        errors.append("Strategy class is required")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

async def estimate_completion_time(config: BacktestConfig) -> str:
    """Estimate backtest completion time based on configuration"""
    try:
        start_date = datetime.fromisoformat(config.start_date)
        end_date = datetime.fromisoformat(config.end_date)
        duration_days = (end_date - start_date).days
        
        # Base time calculation (rough estimates)
        base_time_per_day = 2  # seconds per day of data
        instrument_multiplier = len(config.instruments) * 0.5
        
        estimated_seconds = duration_days * base_time_per_day * instrument_multiplier
        
        if estimated_seconds < 60:
            return f"~{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"~{int(estimated_seconds / 60)} minutes"
        else:
            return f"~{int(estimated_seconds / 3600)} hours"
            
    except:
        return "Unknown"

async def calculate_enhanced_metrics(backtest_id: str) -> Dict[str, float]:
    """Calculate comprehensive performance metrics from actual backtest results"""
    try:
        engine_manager = get_nautilus_engine_manager()
        backtest_status = await engine_manager.get_backtest_status(backtest_id)
        
        if not backtest_status["success"] or backtest_status["backtest"]["status"] != "completed":
            logger.warning(f"Backtest {backtest_id} not completed, returning default metrics")
            return _get_default_metrics()
        
        # Get actual results from NautilusTrader
        backtest_results = backtest_status["backtest"].get("results")
        if not backtest_results:
            logger.warning(f"No results found for backtest {backtest_id}, returning default metrics")
            return _get_default_metrics()
        
        # Extract metrics from NautilusTrader results
        metrics = _parse_nautilus_results(backtest_results)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics for backtest {backtest_id}: {e}")
        return _get_default_metrics()


def _get_default_metrics() -> Dict[str, float]:
    """Return default metrics when real data unavailable"""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "volatility": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0
    }


def _parse_nautilus_results(results: Dict[str, Any]) -> Dict[str, float]:
    """Parse NautilusTrader backtest results into metrics"""
    try:
        # Extract portfolio statistics if available
        portfolio_stats = results.get("portfolio_statistics", {})
        
        # Extract trade statistics
        trades = results.get("trades", [])
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            wins = [trade["pnl"] for trade in trades if trade.get("pnl", 0) > 0]
            losses = [trade["pnl"] for trade in trades if trade.get("pnl", 0) < 0]
            
            average_win = sum(wins) / len(wins) if wins else 0
            average_loss = sum(losses) / len(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Calculate profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1  # Avoid division by zero
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = average_win = average_loss = largest_win = largest_loss = profit_factor = 0
        
        # Extract key metrics from portfolio statistics
        total_return = portfolio_stats.get("total_return", 0.0)
        max_drawdown = portfolio_stats.get("max_drawdown", 0.0)
        sharpe_ratio = portfolio_stats.get("sharpe_ratio", 0.0)
        volatility = portfolio_stats.get("volatility", 0.0)
        
        # Calculate additional metrics
        annualized_return = portfolio_stats.get("annualized_return", total_return)
        sortino_ratio = portfolio_stats.get("sortino_ratio", 0.0)
        calmar_ratio = portfolio_stats.get("calmar_ratio", 0.0)
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "alpha": float(portfolio_stats.get("alpha", 0.0)),
            "beta": float(portfolio_stats.get("beta", 0.0)),
            "information_ratio": float(portfolio_stats.get("information_ratio", 0.0)),
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "average_win": float(average_win),
            "average_loss": float(average_loss),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss)
        }
        
    except Exception as e:
        logger.error(f"Error parsing NautilusTrader results: {e}")
        return _get_default_metrics()

async def get_backtest_trades(backtest_id: str) -> List[Dict[str, Any]]:
    """Get individual trade data from actual backtest results"""
    try:
        engine_manager = get_nautilus_engine_manager()
        backtest_status = await engine_manager.get_backtest_status(backtest_id)
        
        if not backtest_status["success"] or backtest_status["backtest"]["status"] != "completed":
            logger.warning(f"Backtest {backtest_id} not completed, returning empty trades")
            return []
        
        # Get actual results from NautilusTrader
        backtest_results = backtest_status["backtest"].get("results")
        if not backtest_results:
            logger.warning(f"No results found for backtest {backtest_id}, returning empty trades")
            return []
        
        # Extract trades from NautilusTrader results
        raw_trades = backtest_results.get("trades", [])
        
        formatted_trades = []
        for i, trade in enumerate(raw_trades):
            try:
                formatted_trade = {
                    "trade_id": trade.get("id", f"trade-{i}"),
                    "instrument_id": trade.get("instrument", "UNKNOWN"),
                    "side": "buy" if trade.get("side", "").lower() in ["buy", "long", "1"] else "sell",
                    "quantity": float(trade.get("quantity", 0)),
                    "entry_price": float(trade.get("entry_price", 0)),
                    "exit_price": float(trade.get("exit_price", 0)),
                    "entry_time": trade.get("entry_time", ""),
                    "exit_time": trade.get("exit_time", ""),
                    "pnl": float(trade.get("pnl", 0)),
                    "commission": float(trade.get("commission", 0)),
                    "duration": _calculate_trade_duration(trade.get("entry_time"), trade.get("exit_time"))
                }
                formatted_trades.append(formatted_trade)
            except Exception as e:
                logger.warning(f"Error formatting trade {i} for backtest {backtest_id}: {e}")
                continue
        
        return formatted_trades
        
    except Exception as e:
        logger.error(f"Error getting trades for backtest {backtest_id}: {e}")
        return []


def _calculate_trade_duration(entry_time: str, exit_time: str) -> int:
    """Calculate trade duration in minutes"""
    try:
        from datetime import datetime
        entry = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        exit = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        duration = (exit - entry).total_seconds() / 60
        return int(duration)
    except:
        return 0

async def get_equity_curve(backtest_id: str) -> List[Dict[str, Any]]:
    """Get equity curve data from actual backtest results"""
    try:
        engine_manager = get_nautilus_engine_manager()
        backtest_status = await engine_manager.get_backtest_status(backtest_id)
        
        if not backtest_status["success"] or backtest_status["backtest"]["status"] != "completed":
            logger.warning(f"Backtest {backtest_id} not completed, returning empty equity curve")
            return []
        
        # Get actual results from NautilusTrader
        backtest_results = backtest_status["backtest"].get("results")
        if not backtest_results:
            logger.warning(f"No results found for backtest {backtest_id}, returning empty equity curve")
            return []
        
        # Extract equity curve from NautilusTrader results
        raw_equity_data = backtest_results.get("equity_curve", [])
        
        if not raw_equity_data:
            # If no direct equity curve, try to construct from portfolio snapshots
            portfolio_snapshots = backtest_results.get("portfolio_snapshots", [])
            if portfolio_snapshots:
                raw_equity_data = portfolio_snapshots
            else:
                logger.warning(f"No equity data found for backtest {backtest_id}")
                return []
        
        formatted_equity_data = []
        initial_balance = None
        
        for point in raw_equity_data:
            try:
                equity = float(point.get("total_value", point.get("equity", 0)))
                
                if initial_balance is None:
                    initial_balance = equity
                
                # Calculate drawdown as percentage from peak
                drawdown_pct = float(point.get("drawdown_pct", 0))
                if drawdown_pct == 0 and initial_balance > 0:
                    # Calculate drawdown if not provided
                    drawdown_pct = min(0, ((equity - initial_balance) / initial_balance) * 100)
                
                formatted_point = {
                    "timestamp": point.get("timestamp", point.get("time", "")),
                    "equity": round(equity, 2),
                    "balance": round(float(point.get("cash_balance", equity)), 2),
                    "drawdown": round(drawdown_pct, 2),
                    "unrealized_pnl": round(float(point.get("unrealized_pnl", 0)), 2)
                }
                formatted_equity_data.append(formatted_point)
                
            except Exception as e:
                logger.warning(f"Error formatting equity point for backtest {backtest_id}: {e}")
                continue
        
        return formatted_equity_data
        
    except Exception as e:
        logger.error(f"Error getting equity curve for backtest {backtest_id}: {e}")
        return []

def calculate_execution_time(backtest_data: Dict[str, Any]) -> float:
    """Calculate backtest execution time in seconds"""
    try:
        start_time = datetime.fromisoformat(backtest_data["started_at"])
        end_time = datetime.fromisoformat(backtest_data.get("completed_at", datetime.now().isoformat()))
        return (end_time - start_time).total_seconds()
    except:
        return 0.0

def calculate_comparison_summary(backtests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics for backtest comparison"""
    if not backtests:
        return {}
    
    metrics = [bt["metrics"] for bt in backtests]
    
    return {
        "best_return": max(m["total_return"] for m in metrics),
        "worst_return": min(m["total_return"] for m in metrics),
        "best_sharpe": max(m["sharpe_ratio"] for m in metrics),
        "lowest_drawdown": max(m["max_drawdown"] for m in metrics),  # Least negative
        "highest_win_rate": max(m["win_rate"] for m in metrics),
        "average_return": sum(m["total_return"] for m in metrics) / len(metrics)
    }

def calculate_relative_performance(backtests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate relative performance metrics between backtests"""
    if len(backtests) < 2:
        return {}
    
    # For simplicity, compare against the first backtest
    baseline = backtests[0]["metrics"]
    relatives = {}
    
    for i, bt in enumerate(backtests[1:], 1):
        metrics = bt["metrics"]
        relatives[f"backtest_{i}_vs_baseline"] = {
            "return_difference": metrics["total_return"] - baseline["total_return"],
            "sharpe_difference": metrics["sharpe_ratio"] - baseline["sharpe_ratio"],
            "drawdown_difference": metrics["max_drawdown"] - baseline["max_drawdown"]
        }
    
    return relatives

async def export_to_csv(results: Dict[str, Any]) -> Dict[str, str]:
    """Export results to CSV format"""
    import csv
    import tempfile
    import os
    from pathlib import Path
    
    backtest_id = results.get("backtest_id")
    
    try:
        # Create temporary directory for exports
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        
        csv_file_path = export_dir / f"backtest_{backtest_id}.csv"
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Export summary metrics
            if 'metrics' in results:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Value"])
                
                metrics = results['metrics']
                for key, value in metrics.items():
                    writer.writerow([key.replace('_', ' ').title(), value])
                
                writer.writerow([])  # Empty row separator
            
            # Export trades if available
            if 'trades' in results and results['trades']:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'trade_id', 'instrument_id', 'side', 'quantity', 
                    'entry_price', 'exit_price', 'entry_time', 'exit_time', 
                    'pnl', 'commission', 'duration'
                ])
                writer.writeheader()
                writer.writerows(results['trades'])
        
        return {
            "download_url": f"/api/v1/downloads/backtest_{backtest_id}.csv",
            "file_path": str(csv_file_path),
            "file_size": os.path.getsize(csv_file_path)
        }
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

async def export_to_excel(results: Dict[str, Any]) -> Dict[str, str]:
    """Export results to Excel format"""
    try:
        import pandas as pd
        from pathlib import Path
        import os
        
        backtest_id = results.get("backtest_id")
        
        # Create temporary directory for exports
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        
        excel_file_path = export_dir / f"backtest_{backtest_id}.xlsx"
        
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            # Export summary metrics
            if 'metrics' in results:
                metrics_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), "Value": v} 
                    for k, v in results['metrics'].items()
                ])
                metrics_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Export trades
            if 'trades' in results and results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_excel(writer, sheet_name='Trades', index=False)
            
            # Export equity curve
            if 'equity_curve' in results and results['equity_curve']:
                equity_df = pd.DataFrame(results['equity_curve'])
                equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
            
            # Export backtest configuration
            if 'config' in results:
                config_data = []
                for key, value in results['config'].items():
                    config_data.append({"Setting": key, "Value": str(value)})
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        return {
            "download_url": f"/api/v1/downloads/backtest_{backtest_id}.xlsx",
            "file_path": str(excel_file_path),
            "file_size": os.path.getsize(excel_file_path)
        }
        
    except ImportError:
        logger.error("pandas or openpyxl not available for Excel export")
        raise HTTPException(status_code=500, detail="Excel export not available - missing dependencies")
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

async def export_to_pdf(results: Dict[str, Any]) -> Dict[str, str]:
    """Export results to PDF format"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from pathlib import Path
        import os
        
        backtest_id = results.get("backtest_id")
        
        # Create temporary directory for exports
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        
        pdf_file_path = export_dir / f"backtest_{backtest_id}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_file_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.navy
        )
        story.append(Paragraph(f"Backtest Report: {backtest_id}", title_style))
        story.append(Spacer(1, 12))
        
        # Summary Metrics
        if 'metrics' in results:
            story.append(Paragraph("Performance Metrics", styles['Heading2']))
            
            metrics_data = [['Metric', 'Value']]
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                metrics_data.append([key.replace('_', ' ').title(), formatted_value])
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 12))
        
        # Configuration Summary
        if 'config' in results:
            story.append(Paragraph("Configuration", styles['Heading2']))
            config_text = ""
            for key, value in results['config'].items():
                config_text += f"<b>{key.replace('_', ' ').title()}:</b> {value}<br/>"
            story.append(Paragraph(config_text, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Trade Summary
        if 'trades' in results and results['trades']:
            story.append(Paragraph(f"Trade Summary ({len(results['trades'])} trades)", styles['Heading2']))
            
            # Show first 10 trades
            trade_data = [['Symbol', 'Side', 'Quantity', 'Entry', 'Exit', 'PnL']]
            for trade in results['trades'][:10]:
                trade_data.append([
                    trade.get('instrument_id', ''),
                    trade.get('side', ''),
                    str(trade.get('quantity', '')),
                    f"{trade.get('entry_price', 0):.2f}",
                    f"{trade.get('exit_price', 0):.2f}",
                    f"{trade.get('pnl', 0):.2f}"
                ])
            
            trade_table = Table(trade_data)
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(trade_table)
            
            if len(results['trades']) > 10:
                story.append(Paragraph(f"... and {len(results['trades']) - 10} more trades", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return {
            "download_url": f"/api/v1/downloads/backtest_{backtest_id}.pdf",
            "file_path": str(pdf_file_path),
            "file_size": os.path.getsize(pdf_file_path)
        }
        
    except ImportError:
        logger.error("reportlab not available for PDF export")
        raise HTTPException(status_code=500, detail="PDF export not available - missing dependencies")
    except Exception as e:
        logger.error(f"Error exporting to PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

# Export download endpoint
@router.get("/downloads/{filename}")
async def download_export_file(filename: str):
    """Download exported backtest files"""
    from fastapi.responses import FileResponse
    from pathlib import Path
    import os
    
    export_dir = Path("./exports")
    file_path = export_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on extension
    if filename.endswith('.csv'):
        media_type = 'text/csv'
    elif filename.endswith('.xlsx'):
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filename.endswith('.pdf'):
        media_type = 'application/pdf'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )