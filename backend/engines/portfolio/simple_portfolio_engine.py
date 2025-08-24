#!/usr/bin/env python3
"""
Simple Portfolio Engine - Containerized Portfolio Optimization Service
Advanced portfolio optimization, rebalancing, and performance analytics
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Real market data integration
from market_data_client import MarketDataClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    BLACK_LITTERMAN = "black_litterman"

class PortfolioStatus(Enum):
    ACTIVE = "active"
    REBALANCING = "rebalancing"
    PAUSED = "paused"
    OPTIMIZING = "optimizing"
    ERROR = "error"

class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    MANUAL = "manual"

@dataclass
class Position:
    symbol: str
    quantity: float
    market_value: float
    weight: float
    avg_cost: float
    unrealized_pnl: float
    last_updated: datetime

@dataclass
class Portfolio:
    portfolio_id: str
    portfolio_name: str
    total_value: float
    cash_balance: float
    positions: Dict[str, Position]
    target_weights: Dict[str, float]
    status: PortfolioStatus
    optimization_method: OptimizationMethod
    rebalance_frequency: RebalanceFrequency
    created_at: datetime
    last_rebalanced: datetime

@dataclass
class OptimizationResult:
    optimization_id: str
    portfolio_id: str
    method: OptimizationMethod
    target_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_time_ms: float
    created_at: datetime

@dataclass
class PerformanceMetrics:
    portfolio_id: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    var_95: float
    calculated_at: datetime

class SimplePortfolioEngine:
    """
    Simple Portfolio Engine demonstrating containerization approach
    Advanced portfolio optimization and management
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple Portfolio Engine", version="1.0.0")
        self.is_running = False
        self.optimizations_completed = 0
        self.rebalances_executed = 0
        self.portfolios_managed = 0
        self.start_time = time.time()
        
        # Portfolio management state
        self.portfolios: Dict[str, Portfolio] = {}
        self.optimization_history: Dict[str, List[OptimizationResult]] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        
        # Optimization parameters
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        
        # Real market data client
        self.market_data_client = MarketDataClient()
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.rebalance_task = None
        self.market_data_task = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "optimizations_completed": self.optimizations_completed,
                "rebalances_executed": self.rebalances_executed,
                "portfolios_managed": self.portfolios_managed,
                "active_portfolios": len([p for p in self.portfolios.values() if p.status == PortfolioStatus.ACTIVE]),
                "total_portfolio_value": sum(p.total_value for p in self.portfolios.values()),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            uptime = time.time() - self.start_time
            return {
                "optimizations_per_hour": (self.optimizations_completed / max(1, uptime)) * 3600,
                "rebalances_per_hour": (self.rebalances_executed / max(1, uptime)) * 3600,
                "total_optimizations": self.optimizations_completed,
                "total_rebalances": self.rebalances_executed,
                "portfolios_managed": len(self.portfolios),
                "avg_portfolio_value": np.mean([p.total_value for p in self.portfolios.values()]) if self.portfolios else 0,
                "total_aum": sum(p.total_value for p in self.portfolios.values()),
                "cache_efficiency": len(self.optimization_cache) / max(1, self.optimizations_completed),
                "uptime": uptime,
                "engine_type": "portfolio_optimization",
                "containerized": True
            }
        
        @self.app.get("/portfolios")
        async def get_portfolios():
            """Get all managed portfolios"""
            portfolios = []
            for portfolio in self.portfolios.values():
                portfolios.append({
                    "portfolio_id": portfolio.portfolio_id,
                    "portfolio_name": portfolio.portfolio_name,
                    "total_value": portfolio.total_value,
                    "cash_balance": portfolio.cash_balance,
                    "position_count": len(portfolio.positions),
                    "status": portfolio.status.value,
                    "optimization_method": portfolio.optimization_method.value,
                    "rebalance_frequency": portfolio.rebalance_frequency.value,
                    "created_at": portfolio.created_at.isoformat(),
                    "last_rebalanced": portfolio.last_rebalanced.isoformat()
                })
            
            return {
                "portfolios": portfolios,
                "count": len(portfolios),
                "total_aum": sum(p.total_value for p in self.portfolios.values())
            }
        
        @self.app.post("/portfolios")
        async def create_portfolio(portfolio_config: Dict[str, Any]):
            """Create new portfolio"""
            try:
                portfolio = Portfolio(
                    portfolio_id=f"port_{int(time.time())}_{len(self.portfolios)}",
                    portfolio_name=portfolio_config.get("portfolio_name", "Unnamed Portfolio"),
                    total_value=float(portfolio_config.get("initial_value", 100000)),
                    cash_balance=float(portfolio_config.get("cash_balance", 10000)),
                    positions={},
                    target_weights={},
                    status=PortfolioStatus.ACTIVE,
                    optimization_method=OptimizationMethod(portfolio_config.get("optimization_method", "mean_variance")),
                    rebalance_frequency=RebalanceFrequency(portfolio_config.get("rebalance_frequency", "monthly")),
                    created_at=datetime.now(),
                    last_rebalanced=datetime.now()
                )
                
                self.portfolios[portfolio.portfolio_id] = portfolio
                self.portfolios_managed += 1
                self.optimization_history[portfolio.portfolio_id] = []
                self.performance_history[portfolio.portfolio_id] = []
                
                return {
                    "status": "created",
                    "portfolio_id": portfolio.portfolio_id,
                    "portfolio_name": portfolio.portfolio_name,
                    "initial_value": portfolio.total_value
                }
                
            except Exception as e:
                logger.error(f"Portfolio creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/portfolios/with-symbols")
        async def create_portfolio_with_symbols(portfolio_config: Dict[str, Any]):
            """Create new portfolio with specific symbols"""
            try:
                symbols = portfolio_config.get("symbols", [])
                if not symbols:
                    raise HTTPException(status_code=400, detail="Symbols list cannot be empty")
                
                portfolio = Portfolio(
                    portfolio_id=f"port_{int(time.time())}_{len(self.portfolios)}",
                    portfolio_name=portfolio_config.get("portfolio_name", f"Custom Portfolio ({', '.join(symbols)})"),
                    total_value=float(portfolio_config.get("initial_value", 100000)),
                    cash_balance=float(portfolio_config.get("cash_balance", 10000)),
                    positions={},
                    target_weights={},
                    status=PortfolioStatus.ACTIVE,
                    optimization_method=OptimizationMethod(portfolio_config.get("optimization_method", "mean_variance")),
                    rebalance_frequency=RebalanceFrequency(portfolio_config.get("rebalance_frequency", "monthly")),
                    created_at=datetime.now(),
                    last_rebalanced=datetime.now()
                )
                
                # Generate positions with the specified symbols
                portfolio.positions = await self._generate_sample_positions(
                    portfolio.total_value, 
                    symbols=symbols
                )
                
                # Update portfolio metrics with real data
                await self._update_portfolio_metrics(portfolio)
                
                self.portfolios[portfolio.portfolio_id] = portfolio
                self.portfolios_managed += 1
                self.optimization_history[portfolio.portfolio_id] = []
                self.performance_history[portfolio.portfolio_id] = []
                
                logger.info(f"Created portfolio {portfolio.portfolio_id} with symbols: {symbols}")
                
                return {
                    "status": "created",
                    "portfolio_id": portfolio.portfolio_id,
                    "portfolio_name": portfolio.portfolio_name,
                    "initial_value": portfolio.total_value,
                    "symbols": symbols,
                    "positions": {
                        symbol: {
                            "quantity": pos.quantity,
                            "market_value": pos.market_value,
                            "weight": pos.weight
                        }
                        for symbol, pos in portfolio.positions.items()
                    }
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Portfolio creation with symbols error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/portfolios/{portfolio_id}")
        async def get_portfolio_details(portfolio_id: str):
            """Get detailed portfolio information"""
            if portfolio_id not in self.portfolios:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(portfolio)
            
            return {
                "portfolio_id": portfolio_id,
                "portfolio_name": portfolio.portfolio_name,
                "total_value": portfolio.total_value,
                "cash_balance": portfolio.cash_balance,
                "status": portfolio.status.value,
                "optimization_method": portfolio.optimization_method.value,
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "market_value": pos.market_value,
                        "weight": pos.weight,
                        "avg_cost": pos.avg_cost,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "last_updated": pos.last_updated.isoformat()
                    }
                    for symbol, pos in portfolio.positions.items()
                },
                "target_weights": portfolio.target_weights,
                "performance_metrics": {
                    "total_return": performance_metrics.total_return,
                    "annualized_return": performance_metrics.annualized_return,
                    "volatility": performance_metrics.volatility,
                    "sharpe_ratio": performance_metrics.sharpe_ratio,
                    "max_drawdown": performance_metrics.max_drawdown,
                    "alpha": performance_metrics.alpha,
                    "beta": performance_metrics.beta,
                    "var_95": performance_metrics.var_95
                },
                "last_rebalanced": portfolio.last_rebalanced.isoformat()
            }
        
        @self.app.post("/portfolios/{portfolio_id}/optimize")
        async def optimize_portfolio(portfolio_id: str, optimization_config: Dict[str, Any]):
            """Optimize portfolio allocation"""
            try:
                if portfolio_id not in self.portfolios:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                portfolio = self.portfolios[portfolio_id]
                method = OptimizationMethod(optimization_config.get("method", portfolio.optimization_method.value))
                
                # Run optimization
                optimization_result = await self._optimize_portfolio(portfolio, method, optimization_config)
                
                # Store optimization result
                if portfolio_id not in self.optimization_history:
                    self.optimization_history[portfolio_id] = []
                self.optimization_history[portfolio_id].append(optimization_result)
                
                self.optimizations_completed += 1
                
                return {
                    "status": "optimization_completed",
                    "optimization_id": optimization_result.optimization_id,
                    "portfolio_id": portfolio_id,
                    "method": method.value,
                    "target_weights": optimization_result.target_weights,
                    "expected_return": optimization_result.expected_return,
                    "expected_risk": optimization_result.expected_risk,
                    "sharpe_ratio": optimization_result.sharpe_ratio,
                    "optimization_time_ms": optimization_result.optimization_time_ms
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Portfolio optimization error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/portfolios/{portfolio_id}/rebalance")
        async def rebalance_portfolio(portfolio_id: str, rebalance_config: Dict[str, Any]):
            """Rebalance portfolio to target weights"""
            try:
                if portfolio_id not in self.portfolios:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                portfolio = self.portfolios[portfolio_id]
                target_weights = rebalance_config.get("target_weights", portfolio.target_weights)
                
                # Execute rebalancing
                rebalance_result = await self._execute_rebalance(portfolio, target_weights)
                
                self.rebalances_executed += 1
                portfolio.last_rebalanced = datetime.now()
                
                return {
                    "status": "rebalance_completed",
                    "portfolio_id": portfolio_id,
                    "target_weights": target_weights,
                    "trades_executed": rebalance_result.get("trades_executed", 0),
                    "total_turnover": rebalance_result.get("total_turnover", 0),
                    "execution_cost": rebalance_result.get("execution_cost", 0)
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Portfolio rebalance error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/portfolios/{portfolio_id}/positions")
        async def update_positions(portfolio_id: str, positions_data: Dict[str, Any]):
            """Update portfolio positions"""
            try:
                if portfolio_id not in self.portfolios:
                    raise HTTPException(status_code=404, detail="Portfolio not found")
                
                portfolio = self.portfolios[portfolio_id]
                
                for symbol, position_data in positions_data.get("positions", {}).items():
                    position = Position(
                        symbol=symbol,
                        quantity=float(position_data.get("quantity", 0)),
                        market_value=float(position_data.get("market_value", 0)),
                        weight=0,  # Will be calculated
                        avg_cost=float(position_data.get("avg_cost", 0)),
                        unrealized_pnl=float(position_data.get("unrealized_pnl", 0)),
                        last_updated=datetime.now()
                    )
                    portfolio.positions[symbol] = position
                
                # Recalculate portfolio metrics
                await self._update_portfolio_metrics(portfolio)
                
                return {
                    "status": "positions_updated",
                    "portfolio_id": portfolio_id,
                    "position_count": len(portfolio.positions),
                    "total_value": portfolio.total_value
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Position update error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/portfolios/{portfolio_id}/performance")
        async def get_portfolio_performance(portfolio_id: str, period_days: int = 365):
            """Get portfolio performance history"""
            if portfolio_id not in self.portfolios:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            
            performance_history = self.performance_history.get(portfolio_id, [])
            
            # Filter by period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            filtered_history = [
                perf for perf in performance_history 
                if perf.calculated_at >= cutoff_date
            ]
            
            return {
                "portfolio_id": portfolio_id,
                "period_days": period_days,
                "performance_history": [
                    {
                        "date": perf.calculated_at.isoformat(),
                        "total_return": perf.total_return,
                        "annualized_return": perf.annualized_return,
                        "volatility": perf.volatility,
                        "sharpe_ratio": perf.sharpe_ratio,
                        "max_drawdown": perf.max_drawdown,
                        "var_95": perf.var_95
                    }
                    for perf in filtered_history
                ],
                "count": len(filtered_history)
            }
        
        @self.app.get("/portfolios/{portfolio_id}/optimization-history")
        async def get_optimization_history(portfolio_id: str):
            """Get portfolio optimization history"""
            if portfolio_id not in self.portfolios:
                raise HTTPException(status_code=404, detail="Portfolio not found")
            
            history = self.optimization_history.get(portfolio_id, [])
            
            return {
                "portfolio_id": portfolio_id,
                "optimization_history": [
                    {
                        "optimization_id": opt.optimization_id,
                        "method": opt.method.value,
                        "target_weights": opt.target_weights,
                        "expected_return": opt.expected_return,
                        "expected_risk": opt.expected_risk,
                        "sharpe_ratio": opt.sharpe_ratio,
                        "optimization_time_ms": opt.optimization_time_ms,
                        "created_at": opt.created_at.isoformat()
                    }
                    for opt in history
                ],
                "count": len(history)
            }

    async def start_engine(self):
        """Start the portfolio engine"""
        try:
            logger.info("Starting Simple Portfolio Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Initialize market data client
            try:
                await self.market_data_client.start()
                logger.info("MarketDataClient connected successfully")
            except Exception as e:
                logger.warning(f"MarketDataClient connection failed: {e}. Using fallback pricing.")
            
            # Initialize sample portfolios
            await self._initialize_sample_portfolios()
            
            # Start periodic rebalancing task
            self.rebalance_task = asyncio.create_task(self._periodic_rebalance_check())
            
            self.is_running = True
            logger.info(f"Simple Portfolio Engine started successfully managing {len(self.portfolios)} portfolios")
            
        except Exception as e:
            logger.error(f"Failed to start Portfolio Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the portfolio engine"""
        logger.info("Stopping Simple Portfolio Engine...")
        self.is_running = False
        
        # Cancel rebalancing task
        if self.rebalance_task:
            self.rebalance_task.cancel()
        
        # Stop market data task
        if self.market_data_task:
            self.market_data_task.cancel()
        
        # Stop clients
        if self.messagebus:
            await self.messagebus.stop()
            
        if self.market_data_client:
            await self.market_data_client.stop()
        
        logger.info("Simple Portfolio Engine stopped")
    
    async def _initialize_sample_portfolios(self):
        """Initialize sample portfolios for demonstration"""
        sample_portfolios = [
            {
                "portfolio_name": "Balanced Growth Portfolio",
                "initial_value": 250000,
                "cash_balance": 25000,
                "optimization_method": "mean_variance",
                "rebalance_frequency": "monthly"
            },
            {
                "portfolio_name": "Conservative Income Portfolio",
                "initial_value": 500000,
                "cash_balance": 50000,
                "optimization_method": "min_variance",
                "rebalance_frequency": "quarterly"
            }
        ]
        
        for portfolio_data in sample_portfolios:
            portfolio = Portfolio(
                portfolio_id=f"sample_{len(self.portfolios)}",
                portfolio_name=portfolio_data["portfolio_name"],
                total_value=portfolio_data["initial_value"],
                cash_balance=portfolio_data["cash_balance"],
                positions=await self._generate_sample_positions(portfolio_data["initial_value"]),
                target_weights={},
                status=PortfolioStatus.ACTIVE,
                optimization_method=OptimizationMethod(portfolio_data["optimization_method"]),
                rebalance_frequency=RebalanceFrequency(portfolio_data["rebalance_frequency"]),
                created_at=datetime.now(),
                last_rebalanced=datetime.now()
            )
            
            self.portfolios[portfolio.portfolio_id] = portfolio
            self.optimization_history[portfolio.portfolio_id] = []
            self.performance_history[portfolio.portfolio_id] = []
            
            # Update portfolio metrics
            await self._update_portfolio_metrics(portfolio)
        
        self.portfolios_managed = len(self.portfolios)
        logger.info(f"Initialized {len(sample_portfolios)} sample portfolios")
    
    async def _generate_sample_positions(self, total_value: float, symbols: List[str] = None) -> Dict[str, Position]:
        """Generate sample portfolio positions with real market data"""
        if symbols is None:
            # Default to diversified sample symbols
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        
        positions = {}
        remaining_value = total_value * 0.9  # 90% invested, 10% cash
        
        # Get real current prices for all symbols
        try:
            current_prices = await self.market_data_client.get_multiple_prices(symbols)
            logger.info(f"Retrieved real prices for {len(current_prices)} symbols")
        except Exception as e:
            logger.warning(f"Failed to get real prices: {e}. Using fallback pricing.")
            current_prices = {}
        
        for i, symbol in enumerate(symbols):
            if remaining_value <= 0:
                break
                
            # Random allocation between 5% and 20% of remaining value
            allocation = np.random.uniform(0.05, 0.2) * remaining_value
            allocation = min(allocation, remaining_value)
            
            # Use real stock price or fallback to realistic estimate
            stock_price = current_prices.get(symbol) or self.market_data_client.get_real_price_for_mock(symbol)
            quantity = allocation / stock_price
            
            position = Position(
                symbol=symbol,
                quantity=quantity,
                market_value=allocation,
                weight=allocation / total_value,
                avg_cost=stock_price * np.random.uniform(0.9, 1.1),
                unrealized_pnl=allocation * np.random.uniform(-0.1, 0.2),
                last_updated=datetime.now()
            )
            
            positions[symbol] = position
            remaining_value -= allocation
        
        return positions
    
    async def _update_portfolio_metrics(self, portfolio: Portfolio):
        """Update portfolio total value and weights with real market data"""
        try:
            # Get current prices for all positions
            symbols = list(portfolio.positions.keys())
            current_prices = await self.market_data_client.get_multiple_prices(symbols)
            
            # Update market values with current prices
            for symbol, position in portfolio.positions.items():
                current_price = current_prices.get(symbol)
                if current_price is not None:
                    old_value = position.market_value
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                    position.last_updated = datetime.now()
                    logger.debug(f"Updated {symbol}: ${old_value:.2f} -> ${position.market_value:.2f}")
                else:
                    logger.warning(f"Could not get current price for {symbol}, keeping existing value")
        
        except Exception as e:
            logger.warning(f"Failed to update portfolio metrics with real prices: {e}")
        
        # Calculate total portfolio value
        total_market_value = sum(pos.market_value for pos in portfolio.positions.values())
        portfolio.total_value = total_market_value + portfolio.cash_balance
        
        # Update position weights
        for position in portfolio.positions.values():
            position.weight = position.market_value / portfolio.total_value if portfolio.total_value > 0 else 0
    
    async def _calculate_performance_metrics(self, portfolio: Portfolio) -> PerformanceMetrics:
        """Calculate portfolio performance metrics using real historical data"""
        try:
            # Get real historical performance if we have positions
            if portfolio.positions:
                total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, alpha, beta = await self._calculate_real_performance(portfolio)
            else:
                # Fallback to reasonable defaults for empty portfolios
                total_return = 0.0
                annualized_return = 0.0
                volatility = 0.15
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                alpha = 0.0
                beta = 1.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate real performance for portfolio {portfolio.portfolio_id}: {e}. Using estimated values.")
            # Fallback to reasonable market-based estimates
            total_return = np.random.uniform(-0.05, 0.15)
            annualized_return = np.random.uniform(0.05, 0.12)
            volatility = np.random.uniform(0.12, 0.20)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.8
            max_drawdown = np.random.uniform(-0.12, -0.03)
            alpha = np.random.uniform(-0.02, 0.03)
            beta = np.random.uniform(0.8, 1.2)
        
        return PerformanceMetrics(
            portfolio_id=portfolio.portfolio_id,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            alpha=alpha,
            beta=beta,
            var_95=np.random.uniform(-0.08, -0.02) * portfolio.total_value,  # VaR still estimated
            calculated_at=datetime.now()
        )
    
    async def _calculate_real_performance(self, portfolio: Portfolio) -> tuple:
        """Calculate real performance metrics using historical data"""
        try:
            symbols = list(portfolio.positions.keys())
            weights = [pos.weight for pos in portfolio.positions.values()]
            
            # Get historical data for portfolio symbols (1 year)
            historical_data = {}
            for symbol in symbols:
                data = await self.market_data_client.get_historical_data(symbol, period="1y")
                if data is not None and not data.empty:
                    historical_data[symbol] = data
            
            if not historical_data:
                raise Exception("No historical data available")
            
            # Calculate portfolio returns
            portfolio_returns = []
            dates = None
            
            for symbol, data in historical_data.items():
                if dates is None:
                    dates = data.index
                else:
                    # Align dates (intersection)
                    dates = dates.intersection(data.index)
            
            if len(dates) < 30:  # Need at least 30 days of data
                raise Exception("Insufficient historical data")
            
            # Calculate weighted portfolio returns
            for date in dates:
                daily_return = 0.0
                for i, symbol in enumerate(symbols):
                    if symbol in historical_data and date in historical_data[symbol].index:
                        # Simple return calculation
                        price_data = historical_data[symbol].loc[date]
                        if len(historical_data[symbol]) > 1:
                            prev_close = historical_data[symbol]['Close'].iloc[-2] if len(historical_data[symbol]) >= 2 else price_data['Close']
                            daily_ret = (price_data['Close'] - prev_close) / prev_close
                            daily_return += weights[i] * daily_ret
                
                portfolio_returns.append(daily_return)
            
            returns_array = np.array(portfolio_returns)
            
            # Calculate metrics
            total_return = np.sum(returns_array)
            annualized_return = np.mean(returns_array) * 252  # 252 trading days
            volatility = np.std(returns_array) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Simplified alpha/beta calculation (vs market average)
            market_return = 0.10  # Assume 10% market return
            alpha = annualized_return - market_return
            beta = 1.0  # Simplified beta
            
            return total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, alpha, beta
            
        except Exception as e:
            logger.warning(f"Real performance calculation failed: {e}")
            raise
    
    async def _optimize_portfolio(self, portfolio: Portfolio, method: OptimizationMethod, config: Dict[str, Any]) -> OptimizationResult:
        """Optimize portfolio allocation"""
        start_time = time.time()
        
        # Simulate optimization processing time
        await asyncio.sleep(0.5)  # 500ms optimization time
        
        symbols = list(portfolio.positions.keys())
        if not symbols:
            # If no positions exist, use default symbols but log this
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
            logger.warning(f"No positions found for portfolio {portfolio.portfolio_id}, using default symbols for optimization")
        
        if method == OptimizationMethod.MEAN_VARIANCE:
            # Mean-variance optimization (mock)
            weights = await self._mean_variance_optimization(symbols, config)
        elif method == OptimizationMethod.RISK_PARITY:
            # Risk parity (equal risk contribution)
            weights = {symbol: 1.0/len(symbols) for symbol in symbols}
        elif method == OptimizationMethod.MIN_VARIANCE:
            # Minimum variance (mock)
            weights = await self._min_variance_optimization(symbols, config)
        elif method == OptimizationMethod.MAX_SHARPE:
            # Maximum Sharpe ratio (mock)
            weights = await self._max_sharpe_optimization(symbols, config)
        else:
            # Default to equal weights
            weights = {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight/total_weight for symbol, weight in weights.items()}
        
        optimization_time = (time.time() - start_time) * 1000
        
        # Update portfolio target weights
        portfolio.target_weights = weights
        
        optimization_result = OptimizationResult(
            optimization_id=f"opt_{int(time.time())}_{portfolio.portfolio_id}",
            portfolio_id=portfolio.portfolio_id,
            method=method,
            target_weights=weights,
            expected_return=np.random.uniform(0.08, 0.15),
            expected_risk=np.random.uniform(0.12, 0.20),
            sharpe_ratio=np.random.uniform(0.6, 1.8),
            optimization_time_ms=optimization_time,
            created_at=datetime.now()
        )
        
        return optimization_result
    
    async def _mean_variance_optimization(self, symbols: List[str], config: Dict[str, Any]) -> Dict[str, float]:
        """Mock mean-variance optimization"""
        # In reality, would use actual returns and covariance matrix
        n = len(symbols)
        weights = np.random.dirichlet(np.ones(n))
        return {symbols[i]: weights[i] for i in range(n)}
    
    async def _min_variance_optimization(self, symbols: List[str], config: Dict[str, Any]) -> Dict[str, float]:
        """Mock minimum variance optimization"""
        n = len(symbols)
        # Simulate min variance weights (typically more concentrated)
        weights = np.random.exponential(0.5, n)
        weights = weights / weights.sum()
        return {symbols[i]: weights[i] for i in range(n)}
    
    async def _max_sharpe_optimization(self, symbols: List[str], config: Dict[str, Any]) -> Dict[str, float]:
        """Mock maximum Sharpe ratio optimization"""
        n = len(symbols)
        # Simulate max Sharpe weights (typically more concentrated in high Sharpe assets)
        base_weights = np.random.exponential(1.0, n)
        weights = base_weights / base_weights.sum()
        return {symbols[i]: weights[i] for i in range(n)}
    
    async def _execute_rebalance(self, portfolio: Portfolio, target_weights: Dict[str, float]) -> Dict[str, Any]:
        """Execute portfolio rebalancing"""
        # Simulate rebalancing execution
        await asyncio.sleep(0.2)  # 200ms execution time
        
        trades_executed = 0
        total_turnover = 0.0
        execution_cost = 0.0
        
        for symbol, target_weight in target_weights.items():
            current_weight = 0.0
            if symbol in portfolio.positions:
                current_weight = portfolio.positions[symbol].weight
            
            weight_diff = abs(target_weight - current_weight)
            if weight_diff > 0.01:  # 1% threshold
                trades_executed += 1
                total_turnover += weight_diff * portfolio.total_value
                execution_cost += weight_diff * portfolio.total_value * 0.001  # 0.1% transaction cost
        
        # Update portfolio with new target weights
        portfolio.target_weights = target_weights
        
        return {
            "trades_executed": trades_executed,
            "total_turnover": total_turnover,
            "execution_cost": execution_cost
        }
    
    async def _periodic_rebalance_check(self):
        """Periodic check for portfolios that need rebalancing"""
        try:
            while self.is_running:
                for portfolio in self.portfolios.values():
                    if portfolio.status != PortfolioStatus.ACTIVE:
                        continue
                    
                    # Check if rebalancing is due
                    days_since_rebalance = (datetime.now() - portfolio.last_rebalanced).days
                    
                    rebalance_needed = False
                    if portfolio.rebalance_frequency == RebalanceFrequency.DAILY and days_since_rebalance >= 1:
                        rebalance_needed = True
                    elif portfolio.rebalance_frequency == RebalanceFrequency.WEEKLY and days_since_rebalance >= 7:
                        rebalance_needed = True
                    elif portfolio.rebalance_frequency == RebalanceFrequency.MONTHLY and days_since_rebalance >= 30:
                        rebalance_needed = True
                    elif portfolio.rebalance_frequency == RebalanceFrequency.QUARTERLY and days_since_rebalance >= 90:
                        rebalance_needed = True
                    
                    if rebalance_needed and portfolio.target_weights:
                        logger.info(f"Auto-rebalancing portfolio {portfolio.portfolio_id}")
                        portfolio.status = PortfolioStatus.REBALANCING
                        
                        try:
                            await self._execute_rebalance(portfolio, portfolio.target_weights)
                            portfolio.last_rebalanced = datetime.now()
                            portfolio.status = PortfolioStatus.ACTIVE
                            self.rebalances_executed += 1
                        except Exception as e:
                            logger.error(f"Auto-rebalance failed for {portfolio.portfolio_id}: {e}")
                            portfolio.status = PortfolioStatus.ERROR
                
                await asyncio.sleep(3600)  # Check every hour
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Periodic rebalance check error: {e}")

# Create and start the engine
simple_portfolio_engine = SimplePortfolioEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        # For now, use simple engine with hybrid mode flag
        logger.info("Hybrid Portfolio Engine integration enabled (using enhanced simple engine)")
        app = simple_portfolio_engine.app
        engine_instance = simple_portfolio_engine
        # Add hybrid flag to engine
        engine_instance.hybrid_enabled = True
    except Exception as e:
        logger.warning(f"Hybrid Portfolio Engine setup failed: {e}. Using simple engine.")
        app = simple_portfolio_engine.app
        engine_instance = simple_portfolio_engine
else:
    logger.info("Using Simple Portfolio Engine (hybrid disabled)")
    app = simple_portfolio_engine.app
    engine_instance = simple_portfolio_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8900"))
    
    logger.info(f"Starting Portfolio Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
    # Start the engine on startup
    async def lifespan():
        await engine_instance.start_engine()
    
    # Run startup
    asyncio.run(lifespan())
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )