"""
Deployment Database Schema and Management
Handles persistence for strategy deployment pipeline
In production, this would use PostgreSQL with SQLAlchemy
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import uuid
import json

class DeploymentStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class StrategyState(str, Enum):
    DEPLOYING = "deploying"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY_STOPPED = "emergency_stopped"

class EventType(str, Enum):
    DEPLOYMENT_CREATED = "deployment_created"
    DEPLOYMENT_APPROVED = "deployment_approved"
    DEPLOYMENT_REJECTED = "deployment_rejected"
    STRATEGY_DEPLOYED = "strategy_deployed"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_RESUMED = "strategy_resumed"
    STRATEGY_STOPPED = "strategy_stopped"
    EMERGENCY_STOP = "emergency_stop"
    PHASE_ADVANCED = "phase_advanced"
    ROLLBACK_INITIATED = "rollback_initiated"


class DeploymentDatabase:
    """In-memory database for deployment management"""
    
    def __init__(self):
        # Strategy deployments table
        self._strategy_deployments: Dict[str, Dict] = {}
        
        # Live strategies table
        self._live_strategies: Dict[str, Dict] = {}
        
        # Deployment approvals table
        self._deployment_approvals: Dict[str, Dict] = {}
        
        # Strategy events table
        self._strategy_events: List[Dict] = []
        
        # Indexes for efficient queries
        self._deployments_by_strategy: Dict[str, List[str]] = {}
        self._approvals_by_deployment: Dict[str, List[str]] = {}
        self._events_by_strategy: Dict[str, List[str]] = {}
    
    # Strategy Deployments
    
    def create_deployment(
        self,
        strategy_id: str,
        version: str,
        deployment_config: Dict[str, Any],
        rollout_plan: Dict[str, Any],
        created_by: str,
        backtest_id: Optional[str] = None
    ) -> str:
        """Create a new strategy deployment"""
        
        deployment_id = str(uuid.uuid4())
        
        deployment = {
            "deployment_id": deployment_id,
            "strategy_id": strategy_id,
            "version": version,
            "backtest_id": backtest_id,
            "deployment_config": deployment_config,
            "rollout_plan": rollout_plan,
            "status": DeploymentStatus.PENDING_APPROVAL,
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc),
            "approved_by": None,
            "approved_at": None,
            "deployed_at": None,
            "stopped_at": None
        }
        
        self._strategy_deployments[deployment_id] = deployment
        
        # Update indexes
        if strategy_id not in self._deployments_by_strategy:
            self._deployments_by_strategy[strategy_id] = []
        self._deployments_by_strategy[strategy_id].append(deployment_id)
        
        # Log event
        self._log_event(
            None, EventType.DEPLOYMENT_CREATED,
            {"deployment_id": deployment_id, "strategy_id": strategy_id}
        )
        
        return deployment_id
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict]:
        """Get deployment by ID"""
        return self._strategy_deployments.get(deployment_id)
    
    def get_deployments_by_strategy(self, strategy_id: str) -> List[Dict]:
        """Get all deployments for a strategy"""
        deployment_ids = self._deployments_by_strategy.get(strategy_id, [])
        return [self._strategy_deployments[dep_id] for dep_id in deployment_ids if dep_id in self._strategy_deployments]
    
    def get_all_deployments(self) -> List[Dict]:
        """Get all deployments"""
        return list(self._strategy_deployments.values())
    
    def update_deployment_status(
        self,
        deployment_id: str,
        status: DeploymentStatus,
        **kwargs
    ) -> bool:
        """Update deployment status"""
        
        deployment = self._strategy_deployments.get(deployment_id)
        if not deployment:
            return False
        
        deployment["status"] = status
        
        # Update specific fields based on status
        if status == DeploymentStatus.APPROVED:
            deployment["approved_by"] = kwargs.get("approved_by")
            deployment["approved_at"] = datetime.now(timezone.utc)
        elif status == DeploymentStatus.DEPLOYED:
            deployment["deployed_at"] = datetime.now(timezone.utc)
        elif status in [DeploymentStatus.STOPPED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]:
            deployment["stopped_at"] = datetime.now(timezone.utc)
        
        return True
    
    # Live Strategies
    
    def create_live_strategy(
        self,
        strategy_instance_id: str,
        deployment_id: str,
        strategy_id: str,
        initial_state: Dict[str, Any]
    ) -> bool:
        """Create a live strategy instance"""
        
        strategy = {
            "strategy_instance_id": strategy_instance_id,
            "deployment_id": deployment_id,
            "strategy_id": strategy_id,
            "state": initial_state.get("state", StrategyState.DEPLOYING),
            "current_position": initial_state.get("current_position", {}),
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "last_heartbeat": datetime.now(timezone.utc),
            "risk_metrics": initial_state.get("risk_metrics", {}),
            "performance_metrics": initial_state.get("performance_metrics", {}),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        self._live_strategies[strategy_instance_id] = strategy
        
        # Log event
        self._log_event(
            strategy_instance_id, EventType.STRATEGY_DEPLOYED,
            {"deployment_id": deployment_id}
        )
        
        return True
    
    def get_live_strategy(self, strategy_instance_id: str) -> Optional[Dict]:
        """Get live strategy by instance ID"""
        return self._live_strategies.get(strategy_instance_id)
    
    def get_all_live_strategies(self) -> List[Dict]:
        """Get all live strategies"""
        return list(self._live_strategies.values())
    
    def update_live_strategy(
        self,
        strategy_instance_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update live strategy"""
        
        strategy = self._live_strategies.get(strategy_instance_id)
        if not strategy:
            return False
        
        # Track state changes for events
        old_state = strategy.get("state")
        
        for key, value in updates.items():
            strategy[key] = value
        
        strategy["updated_at"] = datetime.now(timezone.utc)
        
        # Log state change events
        new_state = strategy.get("state")
        if old_state != new_state:
            event_type = None
            if new_state == StrategyState.PAUSED:
                event_type = EventType.STRATEGY_PAUSED
            elif new_state == StrategyState.RUNNING:
                event_type = EventType.STRATEGY_RESUMED
            elif new_state == StrategyState.STOPPED:
                event_type = EventType.STRATEGY_STOPPED
            elif new_state == StrategyState.EMERGENCY_STOPPED:
                event_type = EventType.EMERGENCY_STOP
            
            if event_type:
                self._log_event(
                    strategy_instance_id, event_type,
                    {"old_state": old_state, "new_state": new_state}
                )
        
        return True
    
    def delete_live_strategy(self, strategy_instance_id: str) -> bool:
        """Delete live strategy"""
        return self._live_strategies.pop(strategy_instance_id, None) is not None
    
    # Deployment Approvals
    
    def create_approval(
        self,
        deployment_id: str,
        approver_id: str,
        approver_name: str,
        approval_level: int,
        required_role: str
    ) -> str:
        """Create deployment approval record"""
        
        approval_id = str(uuid.uuid4())
        
        approval = {
            "approval_id": approval_id,
            "deployment_id": deployment_id,
            "approver_id": approver_id,
            "approver_name": approver_name,
            "approval_level": approval_level,
            "required_role": required_role,
            "status": ApprovalStatus.PENDING,
            "comments": None,
            "approved_at": None,
            "created_at": datetime.now(timezone.utc)
        }
        
        self._deployment_approvals[approval_id] = approval
        
        # Update index
        if deployment_id not in self._approvals_by_deployment:
            self._approvals_by_deployment[deployment_id] = []
        self._approvals_by_deployment[deployment_id].append(approval_id)
        
        return approval_id
    
    def get_approval(self, approval_id: str) -> Optional[Dict]:
        """Get approval by ID"""
        return self._deployment_approvals.get(approval_id)
    
    def get_approvals_by_deployment(self, deployment_id: str) -> List[Dict]:
        """Get all approvals for a deployment"""
        approval_ids = self._approvals_by_deployment.get(deployment_id, [])
        return [self._deployment_approvals[app_id] for app_id in approval_ids if app_id in self._deployment_approvals]
    
    def update_approval(
        self,
        approval_id: str,
        status: ApprovalStatus,
        comments: Optional[str] = None
    ) -> bool:
        """Update approval status"""
        
        approval = self._deployment_approvals.get(approval_id)
        if not approval:
            return False
        
        approval["status"] = status
        approval["comments"] = comments
        
        if status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]:
            approval["approved_at"] = datetime.now(timezone.utc)
        
        # Log event
        event_type = EventType.DEPLOYMENT_APPROVED if status == ApprovalStatus.APPROVED else EventType.DEPLOYMENT_REJECTED
        self._log_event(
            None, event_type,
            {
                "deployment_id": approval["deployment_id"],
                "approver_id": approval["approver_id"],
                "comments": comments
            }
        )
        
        return True
    
    # Strategy Events
    
    def _log_event(
        self,
        strategy_instance_id: Optional[str],
        event_type: EventType,
        event_data: Dict[str, Any]
    ) -> str:
        """Log a strategy event"""
        
        event_id = str(uuid.uuid4())
        
        event = {
            "event_id": event_id,
            "strategy_instance_id": strategy_instance_id,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self._strategy_events.append(event)
        
        # Update index
        if strategy_instance_id:
            if strategy_instance_id not in self._events_by_strategy:
                self._events_by_strategy[strategy_instance_id] = []
            self._events_by_strategy[strategy_instance_id].append(event_id)
        
        return event_id
    
    def get_events_by_strategy(
        self,
        strategy_instance_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get events for a strategy"""
        
        event_ids = self._events_by_strategy.get(strategy_instance_id, [])
        events = []
        
        for event in reversed(self._strategy_events):
            if event["event_id"] in event_ids:
                events.append(event)
                if len(events) >= limit:
                    break
        
        return events
    
    def get_all_events(self, limit: int = 1000) -> List[Dict]:
        """Get all events"""
        return list(reversed(self._strategy_events))[-limit:]
    
    # Utility methods
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary statistics"""
        
        deployments = list(self._strategy_deployments.values())
        live_strategies = list(self._live_strategies.values())
        
        status_counts = {}
        for deployment in deployments:
            status = deployment["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        state_counts = {}
        for strategy in live_strategies:
            state = strategy["state"]
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            "total_deployments": len(deployments),
            "total_live_strategies": len(live_strategies),
            "deployment_status_counts": status_counts,
            "strategy_state_counts": state_counts,
            "pending_approvals": len([
                d for d in deployments 
                if d["status"] == DeploymentStatus.PENDING_APPROVAL
            ])
        }
    
    def cleanup_old_events(self, days: int = 30):
        """Clean up old events"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Remove old events
        self._strategy_events = [
            event for event in self._strategy_events
            if event["timestamp"] > cutoff_time
        ]
        
        # Rebuild index
        self._events_by_strategy.clear()
        for event in self._strategy_events:
            strategy_id = event.get("strategy_instance_id")
            if strategy_id:
                if strategy_id not in self._events_by_strategy:
                    self._events_by_strategy[strategy_id] = []
                self._events_by_strategy[strategy_id].append(event["event_id"])


# Global database instance
deployment_db = DeploymentDatabase()


# Production PostgreSQL Schema
DEPLOYMENT_SQL_SCHEMA = """
-- Strategy deployment management schema for PostgreSQL

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Strategy deployments table
CREATE TABLE strategy_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL,
    version VARCHAR(50) NOT NULL,
    backtest_id UUID,
    deployment_config JSONB NOT NULL,
    rollout_plan JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending_approval',
    created_by UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    approved_by UUID,
    approved_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_status CHECK (
        status IN ('draft', 'pending_approval', 'approved', 'deploying', 
                  'deployed', 'running', 'paused', 'stopped', 'failed', 'rolled_back')
    )
);

-- Indexes for strategy_deployments
CREATE INDEX idx_strategy_deployments_strategy_id ON strategy_deployments(strategy_id);
CREATE INDEX idx_strategy_deployments_status ON strategy_deployments(status);
CREATE INDEX idx_strategy_deployments_created_at ON strategy_deployments(created_at);

-- Live strategies table
CREATE TABLE live_strategies (
    strategy_instance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deployment_id UUID NOT NULL REFERENCES strategy_deployments(deployment_id),
    strategy_id UUID NOT NULL,
    state VARCHAR(50) NOT NULL DEFAULT 'deploying',
    current_position JSONB,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    risk_metrics JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_state CHECK (
        state IN ('deploying', 'running', 'paused', 'stopped', 'error', 'emergency_stopped')
    )
);

-- Indexes for live_strategies
CREATE INDEX idx_live_strategies_deployment_id ON live_strategies(deployment_id);
CREATE INDEX idx_live_strategies_strategy_id ON live_strategies(strategy_id);
CREATE INDEX idx_live_strategies_state ON live_strategies(state);
CREATE INDEX idx_live_strategies_last_heartbeat ON live_strategies(last_heartbeat);

-- Deployment approvals table
CREATE TABLE deployment_approvals (
    approval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deployment_id UUID NOT NULL REFERENCES strategy_deployments(deployment_id),
    approver_id UUID NOT NULL,
    approver_name VARCHAR(255) NOT NULL,
    approval_level INTEGER NOT NULL,
    required_role VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    comments TEXT,
    approved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_approval_status CHECK (
        status IN ('pending', 'approved', 'rejected')
    ),
    CONSTRAINT valid_approval_level CHECK (approval_level > 0)
);

-- Indexes for deployment_approvals
CREATE INDEX idx_deployment_approvals_deployment_id ON deployment_approvals(deployment_id);
CREATE INDEX idx_deployment_approvals_approver_id ON deployment_approvals(approver_id);
CREATE INDEX idx_deployment_approvals_status ON deployment_approvals(status);

-- Strategy events table
CREATE TABLE strategy_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_instance_id UUID REFERENCES live_strategies(strategy_instance_id),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for strategy_events
CREATE INDEX idx_strategy_events_strategy_instance_id ON strategy_events(strategy_instance_id);
CREATE INDEX idx_strategy_events_event_type ON strategy_events(event_type);
CREATE INDEX idx_strategy_events_timestamp ON strategy_events(timestamp);

-- Rollout monitoring table
CREATE TABLE rollout_monitors (
    monitor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deployment_id UUID NOT NULL REFERENCES strategy_deployments(deployment_id),
    strategy_instance_id UUID REFERENCES live_strategies(strategy_instance_id),
    current_phase INTEGER NOT NULL DEFAULT 0,
    phase_start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    auto_advance BOOLEAN DEFAULT true,
    phase_history JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for rollout_monitors
CREATE INDEX idx_rollout_monitors_deployment_id ON rollout_monitors(deployment_id);
CREATE INDEX idx_rollout_monitors_strategy_instance_id ON rollout_monitors(strategy_instance_id);

-- Emergency actions table
CREATE TABLE emergency_actions (
    action_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_instance_id UUID NOT NULL REFERENCES live_strategies(strategy_instance_id),
    action_type VARCHAR(50) NOT NULL,
    reason TEXT NOT NULL,
    initiated_by UUID NOT NULL,
    initiated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    confirmation_required BOOLEAN DEFAULT true,
    second_confirmation_required BOOLEAN DEFAULT false,
    
    CONSTRAINT valid_action_type CHECK (
        action_type IN ('pause', 'resume', 'stop', 'emergency_stop', 'reduce_size', 'close_positions')
    ),
    CONSTRAINT valid_action_status CHECK (
        status IN ('pending', 'executing', 'completed', 'failed')
    )
);

-- Indexes for emergency_actions
CREATE INDEX idx_emergency_actions_strategy_instance_id ON emergency_actions(strategy_instance_id);
CREATE INDEX idx_emergency_actions_status ON emergency_actions(status);
CREATE INDEX idx_emergency_actions_initiated_at ON emergency_actions(initiated_at);

-- Views for common queries

-- Deployment overview view
CREATE VIEW deployment_overview AS
SELECT 
    d.deployment_id,
    d.strategy_id,
    d.version,
    d.status as deployment_status,
    d.created_at,
    d.deployed_at,
    ls.strategy_instance_id,
    ls.state as strategy_state,
    ls.realized_pnl,
    ls.unrealized_pnl,
    ls.last_heartbeat,
    (
        SELECT COUNT(*) 
        FROM deployment_approvals da 
        WHERE da.deployment_id = d.deployment_id AND da.status = 'pending'
    ) as pending_approvals
FROM strategy_deployments d
LEFT JOIN live_strategies ls ON d.deployment_id = ls.deployment_id;

-- Active strategies view
CREATE VIEW active_strategies AS
SELECT 
    ls.*,
    d.version,
    d.rollout_plan,
    rm.current_phase,
    rm.auto_advance
FROM live_strategies ls
JOIN strategy_deployments d ON ls.deployment_id = d.deployment_id
LEFT JOIN rollout_monitors rm ON d.deployment_id = rm.deployment_id
WHERE ls.state IN ('running', 'paused');

-- Functions for common operations

-- Function to update strategy heartbeat
CREATE OR REPLACE FUNCTION update_strategy_heartbeat(p_strategy_instance_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE live_strategies 
    SET last_heartbeat = NOW(), updated_at = NOW()
    WHERE strategy_instance_id = p_strategy_instance_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check stale strategies
CREATE OR REPLACE FUNCTION get_stale_strategies(p_minutes INTEGER DEFAULT 5)
RETURNS TABLE(strategy_instance_id UUID, strategy_id UUID, last_heartbeat TIMESTAMP WITH TIME ZONE) AS $$
BEGIN
    RETURN QUERY
    SELECT ls.strategy_instance_id, ls.strategy_id, ls.last_heartbeat
    FROM live_strategies ls
    WHERE ls.state = 'running' 
    AND ls.last_heartbeat < NOW() - INTERVAL '1 minute' * p_minutes;
END;
$$ LANGUAGE plpgsql;

-- Function to log strategy event
CREATE OR REPLACE FUNCTION log_strategy_event(
    p_strategy_instance_id UUID,
    p_event_type VARCHAR(100),
    p_event_data JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    event_id UUID;
BEGIN
    INSERT INTO strategy_events (strategy_instance_id, event_type, event_data)
    VALUES (p_strategy_instance_id, p_event_type, p_event_data)
    RETURNING strategy_events.event_id INTO event_id;
    
    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic updates

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to live_strategies
CREATE TRIGGER update_live_strategies_updated_at
    BEFORE UPDATE ON live_strategies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply update trigger to rollout_monitors
CREATE TRIGGER update_rollout_monitors_updated_at
    BEFORE UPDATE ON rollout_monitors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Event logging trigger
CREATE OR REPLACE FUNCTION log_strategy_state_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.state IS DISTINCT FROM NEW.state THEN
        INSERT INTO strategy_events (strategy_instance_id, event_type, event_data)
        VALUES (
            NEW.strategy_instance_id,
            'state_change',
            jsonb_build_object(
                'old_state', OLD.state,
                'new_state', NEW.state,
                'changed_at', NOW()
            )
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply state change trigger
CREATE TRIGGER log_strategy_state_changes
    AFTER UPDATE ON live_strategies
    FOR EACH ROW
    EXECUTE FUNCTION log_strategy_state_change();

-- Cleanup functions

-- Clean up old events
CREATE OR REPLACE FUNCTION cleanup_old_events(p_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM strategy_events 
    WHERE timestamp < NOW() - INTERVAL '1 day' * p_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Materialized view for deployment statistics
CREATE MATERIALIZED VIEW deployment_stats AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_deployments,
    COUNT(*) FILTER (WHERE status = 'deployed') as successful_deployments,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_deployments,
    COUNT(*) FILTER (WHERE status = 'pending_approval') as pending_deployments,
    AVG(EXTRACT(EPOCH FROM (deployed_at - created_at))/60) FILTER (WHERE deployed_at IS NOT NULL) as avg_deployment_time_minutes
FROM strategy_deployments
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date;

-- Index for deployment stats
CREATE INDEX idx_deployment_stats_date ON deployment_stats(date);

-- Refresh function for stats
CREATE OR REPLACE FUNCTION refresh_deployment_stats()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY deployment_stats;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE strategy_deployments IS 'Main deployment tracking table';
COMMENT ON TABLE live_strategies IS 'Active strategy instances with real-time metrics';
COMMENT ON TABLE deployment_approvals IS 'Multi-level approval workflow tracking';
COMMENT ON TABLE strategy_events IS 'Audit trail of all strategy-related events';
COMMENT ON TABLE rollout_monitors IS 'Gradual rollout phase tracking';
COMMENT ON TABLE emergency_actions IS 'Emergency control actions history';
"""