"""
Tests for Strategy Deployment Pipeline API endpoints
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app and dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_routes import router
from deployment_service import deployment_service, DeploymentStatus, StrategyState
from deployment_database import deployment_db
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestDeploymentRoutes:
    """Test suite for deployment API routes"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear in-memory database
        deployment_service.deployments.clear()
        deployment_service.live_strategies.clear()
        deployment_service.approval_workflows.clear()
        deployment_service.rollout_monitors.clear()
    
    def test_create_deployment_request_success(self):
        """Test successful deployment request creation"""
        
        request_data = {
            "strategy_id": "test-strategy-123",
            "version": "2.1.0",
            "backtest_results": {
                "total_return": 0.15,
                "sharpe_ratio": 1.25,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "avg_trade": 0.002,
                "total_trades": 150
            },
            "proposed_config": {
                "strategy_id": "test-strategy-123",
                "risk_engine": {
                    "enabled": True,
                    "max_order_size": 100000,
                    "max_notional_per_order": 50000,
                    "max_daily_loss": 1000,
                    "position_limits": {
                        "max_positions": 5,
                        "max_position_size": 25000
                    }
                },
                "venues": [{
                    "name": "INTERACTIVE_BROKERS",
                    "venue_type": "ECN",
                    "account_id": "DU123456",
                    "routing": "SMART",
                    "client_id": "1",
                    "gateway_host": "localhost",
                    "gateway_port": 7497
                }],
                "data_engine": {
                    "time_bars_timestamp_on_close": True,
                    "validate_data_sequence": True,
                    "buffer_deltas": True
                },
                "exec_engine": {
                    "reconciliation": True,
                    "inflight_check_interval_ms": 5000,
                    "snapshot_orders": True,
                    "snapshot_positions": True
                },
                "environment": {
                    "container_name": "nautilus-backend",
                    "database_url": "${DATABASE_URL}",
                    "redis_url": "${REDIS_URL}",
                    "deployment_id": "deployment_uuid_here",
                    "monitoring_enabled": True,
                    "logging_level": "INFO"
                }
            },
            "rollout_plan": {
                "phases": [
                    {
                        "name": "validation",
                        "position_size_percent": 25,
                        "duration": 7200,
                        "success_criteria": {
                            "min_trades": 5,
                            "max_drawdown": 0.03,
                            "pnl_threshold": -0.01
                        }
                    }
                ],
                "current_phase": 0,
                "escalation_criteria": {
                    "max_loss_percentage": 0.05,
                    "consecutive_losses": 5,
                    "correlation_threshold": 0.8
                }
            }
        }
        
        response = client.post("/create", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "deployment_id" in data
        assert data["status"] == "pending_approval"
        assert data["validation_result"]["valid"] is True
        assert len(data["validation_result"]["errors"]) == 0
    
    def test_create_deployment_request_validation_errors(self):
        """Test deployment request creation with validation errors"""
        
        request_data = {
            "strategy_id": "",  # Missing strategy ID
            "version": "2.1.0",
            "proposed_config": {
                "strategy_id": "",
                "risk_engine": {},  # Missing required fields
                "venues": [],
                "data_engine": {},
                "exec_engine": {},
                "environment": {}
            },
            "rollout_plan": {
                "phases": [],  # No phases
                "current_phase": 0
            }
        }
        
        response = client.post("/create", json=request_data)
        
        assert response.status_code == 400
        assert "errors" in response.json()["detail"]
    
    def test_approve_deployment_success(self):
        """Test successful deployment approval"""
        
        # First create a deployment
        deployment_id = self._create_test_deployment()
        
        approval_data = {
            "deployment_id": deployment_id,
            "comments": "Looks good to deploy"
        }
        
        response = client.post("/approve", json=approval_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "status" in data
    
    def test_approve_deployment_not_found(self):
        """Test approval of non-existent deployment"""
        
        approval_data = {
            "deployment_id": "non-existent-id",
            "comments": "Test comment"
        }
        
        response = client.post("/approve", json=approval_data)
        
        assert response.status_code == 404
        assert "Deployment not found" in response.json()["detail"]
    
    def test_reject_deployment_success(self):
        """Test successful deployment rejection"""
        
        deployment_id = self._create_test_deployment()
        
        rejection_data = {
            "deployment_id": deployment_id,
            "comments": "Risk too high"
        }
        
        response = client.post("/reject", json=rejection_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["status"] == "rejected"
    
    def test_deploy_strategy_success(self):
        """Test successful strategy deployment"""
        
        # Create and approve deployment
        deployment_id = self._create_test_deployment()
        self._approve_test_deployment(deployment_id)
        
        deploy_data = {
            "deployment_id": deployment_id,
            "force_restart": False
        }
        
        response = client.post("/deploy", json=deploy_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "strategy_instance_id" in data
        assert "estimated_start_time" in data
    
    def test_deploy_strategy_not_approved(self):
        """Test deployment of non-approved strategy"""
        
        deployment_id = self._create_test_deployment()
        
        deploy_data = {
            "deployment_id": deployment_id,
            "force_restart": False
        }
        
        response = client.post("/deploy", json=deploy_data)
        
        assert response.status_code == 400
        assert "must be approved" in response.json()["detail"]
    
    def test_get_deployment_status(self):
        """Test getting deployment status"""
        
        deployment_id = self._create_test_deployment()
        
        response = client.get(f"/status/{deployment_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["deployment_id"] == deployment_id
        assert data["status"] == "pending_approval"
        assert "created_at" in data
    
    def test_get_strategy_deployments(self):
        """Test getting all deployments for a strategy"""
        
        strategy_id = "test-strategy-123"
        deployment_id = self._create_test_deployment()
        
        response = client.get(f"/strategy/{strategy_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["deployment_id"] == deployment_id
    
    def test_pause_strategy_success(self):
        """Test successful strategy pause"""
        
        strategy_instance_id = self._create_live_strategy()
        
        control_data = {
            "action": "pause",
            "reason": "Manual pause for testing",
            "force": False
        }
        
        response = client.post(f"/pause/{strategy_instance_id}", json=control_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_state"] == "paused"
    
    def test_resume_strategy_success(self):
        """Test successful strategy resume"""
        
        strategy_instance_id = self._create_live_strategy()
        
        # First pause the strategy
        client.post(f"/pause/{strategy_instance_id}", json={
            "action": "pause",
            "reason": "Test pause",
            "force": False
        })
        
        # Then resume it
        control_data = {
            "action": "resume",
            "reason": "Test resume",
            "force": False
        }
        
        response = client.post(f"/resume/{strategy_instance_id}", json=control_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_state"] == "running"
    
    def test_stop_strategy_success(self):
        """Test successful strategy stop"""
        
        strategy_instance_id = self._create_live_strategy()
        
        control_data = {
            "action": "stop",
            "reason": "Manual stop for testing",
            "force": False
        }
        
        response = client.post(f"/stop/{strategy_instance_id}", json=control_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_state"] == "stopped"
    
    def test_emergency_stop_strategy(self):
        """Test emergency stop strategy"""
        
        strategy_instance_id = self._create_live_strategy()
        
        control_data = {
            "action": "emergency_stop",
            "reason": "Emergency test",
            "force": True
        }
        
        response = client.post(f"/stop/{strategy_instance_id}", json=control_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_state"] == "emergency_stopped"
    
    def test_rollback_deployment(self):
        """Test deployment rollback"""
        
        deployment_id = self._create_test_deployment()
        
        rollback_data = {
            "deployment_id": deployment_id,
            "target_version": "2.0.0",
            "reason": "Performance issues",
            "immediate": False
        }
        
        response = client.post("/rollback", json=rollback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rollback_id" in data
        assert "estimated_duration" in data
        assert "rollback_plan" in data
        assert len(data["rollback_plan"]) > 0
    
    def test_get_live_strategies(self):
        """Test getting all live strategies"""
        
        # Create a live strategy
        strategy_instance_id = self._create_live_strategy()
        
        response = client.get("/strategies/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy_instance_id"] == strategy_instance_id
    
    def test_get_live_strategy_details(self):
        """Test getting specific live strategy details"""
        
        strategy_instance_id = self._create_live_strategy()
        
        response = client.get(f"/strategies/live/{strategy_instance_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["strategy_instance_id"] == strategy_instance_id
        assert "performance_metrics" in data
        assert "risk_metrics" in data
        assert "health_status" in data
    
    def test_get_strategy_metrics(self):
        """Test getting real-time strategy metrics"""
        
        strategy_instance_id = self._create_live_strategy()
        
        response = client.get(f"/strategies/live/{strategy_instance_id}/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "performanceMetrics" in data
        assert "riskMetrics" in data
        assert "positions" in data
        assert "alerts" in data
        assert "timestamp" in data
        assert "healthStatus" in data
    
    def test_control_live_strategy(self):
        """Test general strategy control endpoint"""
        
        strategy_instance_id = self._create_live_strategy()
        
        control_data = {
            "action": "pause",
            "reason": "Test control",
            "force": False
        }
        
        response = client.post(f"/strategies/live/{strategy_instance_id}/control", json=control_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["new_state"] == "paused"
    
    def test_risk_assessment(self):
        """Test risk assessment endpoint"""
        
        assessment_data = {
            "strategyId": "test-strategy-123",
            "proposedConfig": {
                "maxPositions": 5,
                "maxDailyLoss": 1000
            },
            "backtestResults": {
                "maxDrawdown": 0.12,
                "sharpeRatio": 1.1
            }
        }
        
        response = client.post("/risk-assessment", json=assessment_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "risk_level" in data
        assert "portfolio_impact" in data
        assert "max_drawdown_estimate" in data
        assert "warnings" in data
        assert "recommendations" in data
    
    def test_control_strategy_invalid_action(self):
        """Test strategy control with invalid action"""
        
        strategy_instance_id = self._create_live_strategy()
        
        control_data = {
            "action": "invalid_action",
            "reason": "Test",
            "force": False
        }
        
        response = client.post(f"/strategies/live/{strategy_instance_id}/control", json=control_data)
        
        assert response.status_code == 400
        assert "Invalid action" in response.json()["detail"]
    
    def test_strategy_not_found_errors(self):
        """Test various endpoints with non-existent strategy"""
        
        non_existent_id = "non-existent-strategy"
        
        # Test pause
        response = client.post(f"/pause/{non_existent_id}", json={
            "action": "pause",
            "reason": "Test",
            "force": False
        })
        assert response.status_code == 404
        
        # Test get strategy
        response = client.get(f"/strategies/live/{non_existent_id}")
        assert response.status_code == 404
        
        # Test get metrics
        response = client.get(f"/strategies/live/{non_existent_id}/metrics")
        assert response.status_code == 404
    
    # Helper methods
    
    def _create_test_deployment(self) -> str:
        """Create a test deployment and return its ID"""
        
        request_data = {
            "strategy_id": "test-strategy-123",
            "version": "2.1.0",
            "proposed_config": {
                "strategy_id": "test-strategy-123",
                "risk_engine": {
                    "enabled": True,
                    "max_daily_loss": 1000,
                    "position_limits": {"max_positions": 5}
                },
                "venues": [{"name": "INTERACTIVE_BROKERS"}],
                "data_engine": {},
                "exec_engine": {},
                "environment": {"container_name": "nautilus-backend"}
            },
            "rollout_plan": {
                "phases": [{
                    "name": "validation",
                    "position_size_percent": 25,
                    "duration": 7200,
                    "success_criteria": {}
                }],
                "current_phase": 0
            }
        }
        
        response = client.post("/create", json=request_data)
        return response.json()["deployment_id"]
    
    def _approve_test_deployment(self, deployment_id: str):
        """Approve a test deployment"""
        
        # Approve all pending approvals
        deployment = deployment_service.deployments[deployment_id]
        for approval in deployment["approval_chain"]:
            if approval["status"] == "pending":
                approval["status"] = "approved"
                approval["approved_at"] = datetime.utcnow()
        
        deployment["status"] = DeploymentStatus.APPROVED
        deployment["approved_at"] = datetime.utcnow()
    
    def _create_live_strategy(self) -> str:
        """Create a live strategy and return its instance ID"""
        
        deployment_id = self._create_test_deployment()
        self._approve_test_deployment(deployment_id)
        
        deploy_data = {
            "deployment_id": deployment_id,
            "force_restart": False
        }
        
        response = client.post("/deploy", json=deploy_data)
        return response.json()["strategy_instance_id"]


class TestDeploymentService:
    """Test suite for deployment service business logic"""
    
    def setup_method(self):
        """Setup for each test method"""
        deployment_service.deployments.clear()
        deployment_service.live_strategies.clear()
        deployment_service.approval_workflows.clear()
        deployment_service.rollout_monitors.clear()
    
    @pytest.mark.asyncio
    async def test_create_deployment_request_validation(self):
        """Test deployment request validation"""
        
        # Test successful creation
        deployment_id = await deployment_service.create_deployment_request(
            strategy_id="test-strategy",
            version="1.0.0",
            proposed_config={
                "risk_engine": {"max_daily_loss": 1000},
                "venues": [{"name": "IB"}]
            },
            rollout_plan={"phases": [{"name": "test"}]}
        )
        
        assert deployment_id is not None
        assert deployment_id in deployment_service.deployments
        
        # Test validation errors
        with pytest.raises(ValueError):
            await deployment_service.create_deployment_request(
                strategy_id="",  # Empty strategy ID
                version="1.0.0",
                proposed_config={},
                rollout_plan={}
            )
    
    @pytest.mark.asyncio
    async def test_approval_workflow(self):
        """Test approval workflow logic"""
        
        deployment_id = await deployment_service.create_deployment_request(
            strategy_id="test-strategy",
            version="1.0.0",
            proposed_config={
                "risk_engine": {"max_daily_loss": 1000},
                "venues": [{"name": "IB"}]
            },
            rollout_plan={"phases": [{"name": "test"}]}
        )
        
        # Test approval
        result = await deployment_service.approve_deployment(
            deployment_id, "senior_trader", True, "Looks good"
        )
        
        assert result is True
        
        deployment = deployment_service.deployments[deployment_id]
        approvals = deployment_service.approval_workflows[deployment_id]
        
        # Check that the first approval was processed
        senior_approval = next(a for a in approvals if a["approver_id"] == "senior_trader")
        assert senior_approval["status"] == "approved"
        assert senior_approval["comments"] == "Looks good"
    
    @pytest.mark.asyncio
    async def test_strategy_deployment(self):
        """Test strategy deployment process"""
        
        # Create and approve deployment
        deployment_id = await deployment_service.create_deployment_request(
            strategy_id="test-strategy",
            version="1.0.0",
            proposed_config={
                "risk_engine": {"max_daily_loss": 1000},
                "venues": [{"name": "IB"}]
            },
            rollout_plan={"phases": [{"name": "test"}]}
        )
        
        # Manually approve (simulate all approvals)
        deployment = deployment_service.deployments[deployment_id]
        deployment["status"] = DeploymentStatus.APPROVED
        
        # Deploy strategy
        strategy_instance_id = await deployment_service.deploy_strategy(deployment_id)
        
        assert strategy_instance_id is not None
        assert strategy_instance_id in deployment_service.live_strategies
        
        strategy = deployment_service.live_strategies[strategy_instance_id]
        assert strategy["state"] == StrategyState.RUNNING
        assert strategy["deployment_id"] == deployment_id
    
    @pytest.mark.asyncio
    async def test_strategy_control_actions(self):
        """Test strategy control actions"""
        
        # Create and deploy strategy
        deployment_id = await deployment_service.create_deployment_request(
            strategy_id="test-strategy",
            version="1.0.0",
            proposed_config={
                "risk_engine": {"max_daily_loss": 1000},
                "venues": [{"name": "IB"}]
            },
            rollout_plan={"phases": [{"name": "test"}]}
        )
        
        deployment = deployment_service.deployments[deployment_id]
        deployment["status"] = DeploymentStatus.APPROVED
        
        strategy_instance_id = await deployment_service.deploy_strategy(deployment_id)
        
        # Test pause
        result = await deployment_service.control_strategy(
            strategy_instance_id, "pause", "Test pause"
        )
        
        assert result["success"] is True
        assert result["new_state"] == StrategyState.PAUSED
        
        strategy = deployment_service.live_strategies[strategy_instance_id]
        assert strategy["state"] == StrategyState.PAUSED
        
        # Test resume
        result = await deployment_service.control_strategy(
            strategy_instance_id, "resume", "Test resume"
        )
        
        assert result["success"] is True
        assert result["new_state"] == StrategyState.RUNNING
        
        # Test emergency stop
        result = await deployment_service.control_strategy(
            strategy_instance_id, "emergency_stop", "Test emergency"
        )
        
        assert result["success"] is True
        assert result["new_state"] == StrategyState.EMERGENCY_STOPPED
    
    @pytest.mark.asyncio
    async def test_rollback_process(self):
        """Test deployment rollback"""
        
        # Create and deploy strategy
        deployment_id = await deployment_service.create_deployment_request(
            strategy_id="test-strategy",
            version="2.0.0",
            proposed_config={
                "risk_engine": {"max_daily_loss": 1000},
                "venues": [{"name": "IB"}]
            },
            rollout_plan={"phases": [{"name": "test"}]}
        )
        
        deployment = deployment_service.deployments[deployment_id]
        deployment["status"] = DeploymentStatus.DEPLOYED
        
        # Test rollback
        rollback_id = await deployment_service.rollback_deployment(
            deployment_id, "1.0.0", "Performance issues"
        )
        
        assert rollback_id is not None
        assert deployment["status"] == DeploymentStatus.ROLLED_BACK
        assert deployment["rollback_reason"] == "Performance issues"


if __name__ == "__main__":
    pytest.main([__file__])