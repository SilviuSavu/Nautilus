"""
Backtest Database Management
In-memory database for backtest data with persistence capabilities
In production, this would be replaced with PostgreSQL
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BacktestDatabase:
    """In-memory database for backtest management"""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        
        # Core data structures
        self._backtests: Dict[str, Dict[str, Any]] = {}
        self._backtest_results: Dict[str, Dict[str, Any]] = {}
        self._backtest_trades: Dict[str, List[Dict[str, Any]]] = {}
        self._backtest_equity_curves: Dict[str, List[Dict[str, Any]]] = {}
        self._backtest_metrics: Dict[str, Dict[str, float]] = {}
        
        # Indexes for efficient querying
        self._status_index: Dict[str, List[str]] = {
            'queued': [],
            'running': [],
            'completed': [],
            'failed': [],
            'cancelled': []
        }
        self._strategy_index: Dict[str, List[str]] = {}
        self._user_index: Dict[str, List[str]] = {}
        
        # Load from persistence if available
        if persist_path:
            self._load_from_disk()
    
    def create_backtest(self, backtest_id: str, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Create a new backtest record"""
        now = datetime.now(timezone.utc)
        
        backtest_data = {
            "backtest_id": backtest_id,
            "user_id": user_id,
            "config": config,
            "status": "queued",
            "created_at": now.isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": 0.0,
            "error_message": None,
            "execution_time": None,
            "resource_usage": {},
            "tags": []
        }
        
        self._backtests[backtest_id] = backtest_data
        
        # Update indexes
        self._status_index["queued"].append(backtest_id)
        
        strategy_class = config.get("strategy_class", "unknown")
        if strategy_class not in self._strategy_index:
            self._strategy_index[strategy_class] = []
        self._strategy_index[strategy_class].append(backtest_id)
        
        if user_id not in self._user_index:
            self._user_index[user_id] = []
        self._user_index[user_id].append(backtest_id)
        
        self._persist_to_disk()
        
        return backtest_data.copy()
    
    def get_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get backtest by ID"""
        return self._backtests.get(backtest_id, {}).copy() if backtest_id in self._backtests else None
    
    def update_backtest_status(self, backtest_id: str, status: str, **kwargs) -> bool:
        """Update backtest status and related fields"""
        if backtest_id not in self._backtests:
            return False
        
        backtest = self._backtests[backtest_id]
        old_status = backtest.get("status")
        
        # Update status
        backtest["status"] = status
        
        # Update timestamps
        now = datetime.now(timezone.utc).isoformat()
        if status == "running" and not backtest.get("started_at"):
            backtest["started_at"] = now
        elif status in ["completed", "failed", "cancelled"]:
            backtest["completed_at"] = now
            
            # Calculate execution time
            if backtest.get("started_at"):
                start_time = datetime.fromisoformat(backtest["started_at"])
                end_time = datetime.fromisoformat(now)
                backtest["execution_time"] = (end_time - start_time).total_seconds()
        
        # Update additional fields
        for key, value in kwargs.items():
            backtest[key] = value
        
        # Update status index
        if old_status and old_status in self._status_index:
            if backtest_id in self._status_index[old_status]:
                self._status_index[old_status].remove(backtest_id)
        
        if status in self._status_index:
            if backtest_id not in self._status_index[status]:
                self._status_index[status].append(backtest_id)
        
        self._persist_to_disk()
        
        return True
    
    def store_backtest_results(self, backtest_id: str, results: Dict[str, Any]) -> bool:
        """Store comprehensive backtest results"""
        if backtest_id not in self._backtests:
            return False
        
        self._backtest_results[backtest_id] = {
            "backtest_id": backtest_id,
            "results": results,
            "stored_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._persist_to_disk()
        return True
    
    def store_backtest_trades(self, backtest_id: str, trades: List[Dict[str, Any]]) -> bool:
        """Store individual trade data"""
        if backtest_id not in self._backtests:
            return False
        
        # Add metadata to each trade
        processed_trades = []
        for trade in trades:
            trade_with_meta = trade.copy()
            trade_with_meta["backtest_id"] = backtest_id
            trade_with_meta["stored_at"] = datetime.now(timezone.utc).isoformat()
            processed_trades.append(trade_with_meta)
        
        self._backtest_trades[backtest_id] = processed_trades
        
        self._persist_to_disk()
        return True
    
    def store_equity_curve(self, backtest_id: str, equity_data: List[Dict[str, Any]]) -> bool:
        """Store equity curve data for visualization"""
        if backtest_id not in self._backtests:
            return False
        
        self._backtest_equity_curves[backtest_id] = equity_data
        
        self._persist_to_disk()
        return True
    
    def store_performance_metrics(self, backtest_id: str, metrics: Dict[str, float]) -> bool:
        """Store calculated performance metrics"""
        if backtest_id not in self._backtests:
            return False
        
        self._backtest_metrics[backtest_id] = {
            **metrics,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._persist_to_disk()
        return True
    
    def get_backtest_results(self, backtest_id: str, include_trades: bool = True, 
                           include_equity_curve: bool = True) -> Optional[Dict[str, Any]]:
        """Get comprehensive backtest results"""
        backtest = self.get_backtest(backtest_id)
        if not backtest:
            return None
        
        results = {
            "backtest": backtest,
            "metrics": self._backtest_metrics.get(backtest_id, {}),
            "results": self._backtest_results.get(backtest_id, {})
        }
        
        if include_trades:
            results["trades"] = self._backtest_trades.get(backtest_id, [])
        
        if include_equity_curve:
            results["equity_curve"] = self._backtest_equity_curves.get(backtest_id, [])
        
        return results
    
    def list_backtests(self, user_id: Optional[str] = None, status: Optional[str] = None,
                      strategy_class: Optional[str] = None, limit: int = 50, 
                      offset: int = 0) -> Dict[str, Any]:
        """List backtests with filtering and pagination"""
        
        # Start with all backtest IDs
        backtest_ids = list(self._backtests.keys())
        
        # Apply filters
        if user_id:
            user_backtests = self._user_index.get(user_id, [])
            backtest_ids = [bid for bid in backtest_ids if bid in user_backtests]
        
        if status:
            status_backtests = self._status_index.get(status, [])
            backtest_ids = [bid for bid in backtest_ids if bid in status_backtests]
        
        if strategy_class:
            strategy_backtests = self._strategy_index.get(strategy_class, [])
            backtest_ids = [bid for bid in backtest_ids if bid in strategy_backtests]
        
        # Sort by creation date (newest first)
        backtest_ids.sort(key=lambda bid: self._backtests[bid]["created_at"], reverse=True)
        
        # Apply pagination
        total_count = len(backtest_ids)
        paginated_ids = backtest_ids[offset:offset + limit]
        
        # Get backtest data
        backtests = []
        for bid in paginated_ids:
            backtest_data = self._backtests[bid].copy()
            
            # Add summary metrics if available
            if bid in self._backtest_metrics:
                metrics = self._backtest_metrics[bid]
                backtest_data["summary_metrics"] = {
                    "total_return": metrics.get("total_return"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "win_rate": metrics.get("win_rate")
                }
            
            backtests.append(backtest_data)
        
        return {
            "backtests": backtests,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    
    def delete_backtest(self, backtest_id: str) -> bool:
        """Delete backtest and all associated data"""
        if backtest_id not in self._backtests:
            return False
        
        backtest = self._backtests[backtest_id]
        
        # Remove from main storage
        del self._backtests[backtest_id]
        
        # Remove associated data
        self._backtest_results.pop(backtest_id, None)
        self._backtest_trades.pop(backtest_id, None)
        self._backtest_equity_curves.pop(backtest_id, None)
        self._backtest_metrics.pop(backtest_id, None)
        
        # Remove from indexes
        status = backtest.get("status")
        if status and status in self._status_index:
            if backtest_id in self._status_index[status]:
                self._status_index[status].remove(backtest_id)
        
        strategy_class = backtest.get("config", {}).get("strategy_class")
        if strategy_class and strategy_class in self._strategy_index:
            if backtest_id in self._strategy_index[strategy_class]:
                self._strategy_index[strategy_class].remove(backtest_id)
        
        user_id = backtest.get("user_id")
        if user_id and user_id in self._user_index:
            if backtest_id in self._user_index[user_id]:
                self._user_index[user_id].remove(backtest_id)
        
        self._persist_to_disk()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_backtests = len(self._backtests)
        
        status_counts = {}
        for status, backtest_ids in self._status_index.items():
            status_counts[status] = len(backtest_ids)
        
        strategy_counts = {}
        for strategy, backtest_ids in self._strategy_index.items():
            strategy_counts[strategy] = len(backtest_ids)
        
        return {
            "total_backtests": total_backtests,
            "status_counts": status_counts,
            "strategy_counts": strategy_counts,
            "storage_sizes": {
                "backtests": len(self._backtests),
                "results": len(self._backtest_results),
                "trades": len(self._backtest_trades),
                "equity_curves": len(self._backtest_equity_curves),
                "metrics": len(self._backtest_metrics)
            }
        }
    
    def _persist_to_disk(self):
        """Persist data to disk if path is configured"""
        if not self.persist_path:
            return
        
        try:
            persist_dir = Path(self.persist_path).parent
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                "backtests": self._backtests,
                "results": self._backtest_results,
                "trades": self._backtest_trades,
                "equity_curves": self._backtest_equity_curves,
                "metrics": self._backtest_metrics,
                "indexes": {
                    "status": self._status_index,
                    "strategy": self._strategy_index,
                    "user": self._user_index
                },
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug(f"Backtest data persisted to {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to persist backtest data: {e}")
    
    def _load_from_disk(self):
        """Load data from disk if available"""
        if not self.persist_path or not Path(self.persist_path).exists():
            logger.info("No persisted backtest data found, starting fresh")
            return
        
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            self._backtests = data.get("backtests", {})
            self._backtest_results = data.get("results", {})
            self._backtest_trades = data.get("trades", {})
            self._backtest_equity_curves = data.get("equity_curves", {})
            self._backtest_metrics = data.get("metrics", {})
            
            # Load indexes
            indexes = data.get("indexes", {})
            self._status_index = indexes.get("status", {
                'queued': [], 'running': [], 'completed': [], 'failed': [], 'cancelled': []
            })
            self._strategy_index = indexes.get("strategy", {})
            self._user_index = indexes.get("user", {})
            
            logger.info(f"Loaded {len(self._backtests)} backtests from {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to load backtest data from disk: {e}")
            logger.info("Starting with fresh database")


# Global database instance
backtest_db = BacktestDatabase(persist_path="./data/backtests.json")


def get_backtest_database() -> BacktestDatabase:
    """Get the global backtest database instance"""
    return backtest_db


# Database schema for reference (if using PostgreSQL in production)
BACKTEST_SCHEMA_SQL = """
-- Backtest management tables for PostgreSQL
-- This would be used instead of the in-memory database in production

CREATE TABLE IF NOT EXISTS backtests (
    backtest_id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL,
    strategy_class VARCHAR(255) NOT NULL,
    configuration JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    progress DECIMAL(5,2) DEFAULT 0.0,
    execution_time DECIMAL(10,2),
    error_message TEXT,
    resource_usage JSONB,
    tags TEXT[],
    
    CONSTRAINT valid_status CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_progress CHECK (progress >= 0.0 AND progress <= 100.0)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id UUID REFERENCES backtests(backtest_id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL, -- 'summary', 'detailed', 'metrics'
    result_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(backtest_id, result_type)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id UUID REFERENCES backtests(backtest_id) ON DELETE CASCADE,
    instrument_id VARCHAR(255) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8) NOT NULL,
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE NOT NULL,
    pnl DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0.0,
    duration_seconds INTEGER GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (exit_time - entry_time))
    ) STORED,
    tags JSONB,
    
    CONSTRAINT valid_quantity CHECK (quantity > 0),
    CONSTRAINT valid_prices CHECK (entry_price > 0 AND exit_price > 0),
    CONSTRAINT valid_times CHECK (exit_time > entry_time)
);

CREATE TABLE IF NOT EXISTS backtest_equity_curves (
    point_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id UUID REFERENCES backtests(backtest_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity DECIMAL(20,8) NOT NULL,
    balance DECIMAL(20,8) NOT NULL,
    drawdown DECIMAL(8,4) NOT NULL,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0.0,
    
    CONSTRAINT valid_equity CHECK (equity >= 0),
    CONSTRAINT valid_drawdown CHECK (drawdown <= 0)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_id UUID REFERENCES backtests(backtest_id) ON DELETE CASCADE,
    total_return DECIMAL(8,4),
    annualized_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(8,4),
    alpha DECIMAL(8,4),
    beta DECIMAL(8,4),
    information_ratio DECIMAL(8,4),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    average_win DECIMAL(20,8),
    average_loss DECIMAL(20,8),
    largest_win DECIMAL(20,8),
    largest_loss DECIMAL(20,8),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(backtest_id),
    CONSTRAINT valid_trades CHECK (total_trades = winning_trades + losing_trades),
    CONSTRAINT valid_win_rate CHECK (win_rate >= 0 AND win_rate <= 100)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_backtests_user_id ON backtests(user_id);
CREATE INDEX IF NOT EXISTS idx_backtests_status ON backtests(status);
CREATE INDEX IF NOT EXISTS idx_backtests_strategy_class ON backtests(strategy_class);
CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON backtests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_backtest_id ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_instrument_id ON backtest_trades(instrument_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_entry_time ON backtest_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_backtest_equity_curves_backtest_id ON backtest_equity_curves(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_equity_curves_timestamp ON backtest_equity_curves(timestamp);

-- Views for common queries
CREATE OR REPLACE VIEW backtest_summary AS
SELECT 
    b.backtest_id,
    b.user_id,
    b.strategy_class,
    b.status,
    b.created_at,
    b.started_at,
    b.completed_at,
    b.execution_time,
    m.total_return,
    m.sharpe_ratio,
    m.max_drawdown,
    m.total_trades,
    COUNT(t.trade_id) as actual_trade_count
FROM backtests b
LEFT JOIN backtest_metrics m ON b.backtest_id = m.backtest_id
LEFT JOIN backtest_trades t ON b.backtest_id = t.backtest_id
GROUP BY 
    b.backtest_id, b.user_id, b.strategy_class, b.status, 
    b.created_at, b.started_at, b.completed_at, b.execution_time,
    m.total_return, m.sharpe_ratio, m.max_drawdown, m.total_trades;
"""


def init_postgresql_schema(connection_string: str):
    """Initialize PostgreSQL schema for production use"""
    try:
        import psycopg2
        
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        
        # Execute schema creation
        cur.execute(BACKTEST_SCHEMA_SQL)
        conn.commit()
        
        cur.close()
        conn.close()
        
        logger.info("PostgreSQL backtest schema initialized successfully")
        
    except ImportError:
        logger.warning("psycopg2 not available, skipping PostgreSQL schema initialization")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL schema: {e}")
        raise