"""
Tests for NautilusTrader Engine Management Service

Tests Docker-based engine management functionality as specified in Story 6.1.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from nautilus_engine_service import (
    NautilusEngineManager,
    EngineState,
    EngineConfig,
    get_nautilus_engine_manager
)


@pytest.fixture
def engine_service():
    """Create engine service instance for testing"""
    return NautilusEngineManager()


@pytest.fixture
def sample_config():
    """Sample engine configuration for testing"""
    return EngineConfig(
        engine_type="live",
        log_level="INFO",
        instance_id="test-001",
        trading_mode="paper",
        max_memory="2g",
        max_cpu="2.0",
        risk_engine_enabled=True,
        max_position_size=100000,
        max_order_rate=100
    )


class TestNautilusEngineService:
    """Test suite for NautilusEngineService"""

    @pytest.mark.asyncio
    async def test_initial_state(self, engine_service):
        """Test engine service initial state"""
        assert engine_service.current_state == EngineState.STOPPED
        assert engine_service.current_config is None
        assert engine_service.session_id is None
        assert engine_service.started_at is None

    @pytest.mark.asyncio
    async def test_start_engine_success(self, engine_service, sample_config):
        """Test successful engine start"""
        with patch.object(engine_service, '_check_container_status') as mock_status, \
             patch.object(engine_service, '_write_engine_config') as mock_write, \
             patch.object(engine_service, '_execute_engine_command') as mock_execute:
            
            # Mock container exists and running
            mock_status.return_value = {"exists": True, "running": True, "status": "running"}
            mock_execute.return_value = {"success": True, "output": "Engine started"}

            result = await engine_service.start_engine(sample_config)

            assert result.success is True
            assert result.state == EngineState.RUNNING
            assert "Engine started in paper mode" in result.message
            assert engine_service.current_state == EngineState.RUNNING
            assert engine_service.current_config == sample_config
            assert engine_service.session_id is not None
            assert engine_service.started_at is not None

    @pytest.mark.asyncio
    async def test_start_engine_container_not_found(self, engine_service, sample_config):
        """Test engine start when container doesn't exist"""
        with patch.object(engine_service, '_check_container_status') as mock_status:
            mock_status.return_value = {"exists": False, "running": False, "status": "not_found"}

            result = await engine_service.start_engine(sample_config)

            assert result.success is False
            assert result.state == EngineState.ERROR
            assert "Container 'test-nautilus-engine' does not exist" in result.error_details
            assert engine_service.current_state == EngineState.ERROR

    @pytest.mark.asyncio
    async def test_start_engine_command_failure(self, engine_service, sample_config):
        """Test engine start when Docker command fails"""
        with patch.object(engine_service, '_check_container_status') as mock_status, \
             patch.object(engine_service, '_write_engine_config') as mock_write, \
             patch.object(engine_service, '_execute_engine_command') as mock_execute:
            
            mock_status.return_value = {"exists": True, "running": True, "status": "running"}
            mock_execute.return_value = {"success": False, "error": "Engine failed to start"}

            result = await engine_service.start_engine(sample_config)

            assert result.success is False
            assert result.state == EngineState.ERROR
            assert "Engine failed to start" in result.error_details
            assert engine_service.current_state == EngineState.ERROR

    @pytest.mark.asyncio
    async def test_stop_engine_success(self, engine_service):
        """Test successful engine stop"""
        # Set up running state
        engine_service.current_state = EngineState.RUNNING
        engine_service.started_at = datetime.utcnow()
        engine_service.session_id = "test_session"

        with patch.object(engine_service, '_execute_engine_command') as mock_execute:
            mock_execute.return_value = {"success": True, "output": "Engine stopped"}

            result = await engine_service.stop_engine()

            assert result.success is True
            assert result.state == EngineState.STOPPED
            assert "Engine stopped successfully" in result.message
            assert engine_service.current_state == EngineState.STOPPED
            assert engine_service.started_at is None
            assert engine_service.session_id is None

    @pytest.mark.asyncio
    async def test_stop_engine_force(self, engine_service):
        """Test force stop engine"""
        engine_service.current_state = EngineState.RUNNING

        with patch.object(engine_service, '_execute_engine_command') as mock_execute:
            mock_execute.return_value = {"success": True, "output": "Engine force stopped"}

            result = await engine_service.stop_engine(force=True)

            assert result.success is True
            assert result.state == EngineState.STOPPED
            mock_execute.assert_called_with("stop_force")

    @pytest.mark.asyncio
    async def test_restart_engine_success(self, engine_service, sample_config):
        """Test successful engine restart"""
        engine_service.current_config = sample_config

        with patch.object(engine_service, 'stop_engine') as mock_stop, \
             patch.object(engine_service, 'start_engine') as mock_start:
            
            mock_stop.return_value = EngineResult(
                success=True, message="Stopped", state=EngineState.STOPPED
            )
            mock_start.return_value = EngineResult(
                success=True, message="Started", state=EngineState.RUNNING
            )

            result = await engine_service.restart_engine()

            assert result.success is True
            mock_stop.assert_called_once()
            mock_start.assert_called_once_with(sample_config)

    @pytest.mark.asyncio
    async def test_restart_engine_no_config(self, engine_service):
        """Test restart engine when no config available"""
        result = await engine_service.restart_engine()

        assert result.success is False
        assert result.state == EngineState.ERROR
        assert "No configuration available for restart" in result.message

    @pytest.mark.asyncio
    async def test_get_engine_status(self, engine_service, sample_config):
        """Test get engine status"""
        # Set up engine state
        engine_service.current_state = EngineState.RUNNING
        engine_service.current_config = sample_config
        engine_service.started_at = datetime.utcnow()

        with patch.object(engine_service, '_get_resource_metrics') as mock_metrics:
            mock_metrics.return_value = ResourceMetrics(
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1200.0,
                network_rx_mb=100.0,
                network_tx_mb=50.0,
                timestamp=datetime.utcnow()
            )

            status = await engine_service.get_engine_status()

            assert isinstance(status, EngineStatus)
            assert status.state == EngineState.RUNNING
            assert status.mode == EngineMode.PAPER
            assert status.configuration == sample_config
            assert status.started_at is not None
            assert status.uptime_seconds is not None
            assert status.resource_metrics is not None

    @pytest.mark.asyncio
    async def test_get_engine_status_stopped(self, engine_service):
        """Test get engine status when stopped"""
        status = await engine_service.get_engine_status()

        assert status.state == EngineState.STOPPED
        assert status.mode is None
        assert status.started_at is None
        assert status.uptime_seconds is None
        assert status.resource_metrics is None

    @pytest.mark.asyncio
    async def test_update_config(self, engine_service, sample_config):
        """Test update engine configuration"""
        result = await engine_service.update_config(sample_config)

        assert result.success is True
        assert engine_service.current_config == sample_config

    @pytest.mark.asyncio
    async def test_update_config_running_engine(self, engine_service, sample_config):
        """Test update config when engine is running"""
        engine_service.current_state = EngineState.RUNNING

        result = await engine_service.update_config(sample_config)

        assert result.success is True
        assert "Restart required for changes to take effect" in result.message

    @pytest.mark.asyncio
    async def test_get_engine_logs(self, engine_service):
        """Test get engine logs"""
        mock_logs = "2023-01-01 INFO: Engine started\n2023-01-01 INFO: Processing data"
        
        with patch.object(engine_service, '_run_command') as mock_run:
            mock_run.return_value = {"success": True, "output": mock_logs}

            logs = await engine_service.get_engine_logs(lines=50)

            assert isinstance(logs, list)
            assert len(logs) == 2
            assert "Engine started" in logs[0]
            assert "Processing data" in logs[1]

    @pytest.mark.asyncio
    async def test_get_engine_logs_failure(self, engine_service):
        """Test get engine logs when command fails"""
        with patch.object(engine_service, '_run_command') as mock_run:
            mock_run.return_value = {"success": False, "error": "Container not found"}

            logs = await engine_service.get_engine_logs()

            assert isinstance(logs, list)
            assert len(logs) == 1
            assert "Error retrieving logs: Container not found" in logs[0]

    @pytest.mark.asyncio
    async def test_check_container_status_exists(self, engine_service):
        """Test check container status when container exists"""
        mock_inspect_data = [{
            "State": {"Running": True, "Status": "running"},
            "Config": {"Image": "nautilus:latest"},
            "Platform": "linux"
        }]

        with patch.object(engine_service, '_run_command') as mock_run:
            mock_run.return_value = {
                "success": True, 
                "output": '[{"State": {"Running": true, "Status": "running"}}]'
            }

            result = await engine_service._check_container_status()

            assert result["exists"] is True
            assert result["running"] is True
            assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_check_container_status_not_exists(self, engine_service):
        """Test check container status when container doesn't exist"""
        with patch.object(engine_service, '_run_command') as mock_run:
            mock_run.return_value = {"success": False, "error": "No such container"}

            result = await engine_service._check_container_status()

            assert result["exists"] is False
            assert result["running"] is False
            assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_resource_metrics(self, engine_service):
        """Test get resource metrics"""
        mock_stats = "45.67%,1.5GiB / 4GiB,2.5GiB / 8GiB,1.2kB / 500B"

        with patch.object(engine_service, '_run_command') as mock_run:
            mock_run.return_value = {"success": True, "output": mock_stats}

            metrics = await engine_service._get_resource_metrics()

            assert isinstance(metrics, ResourceMetrics)
            assert metrics.cpu_percent == 45.67
            assert metrics.timestamp is not None

    @pytest.mark.asyncio
    async def test_parse_memory_size(self, engine_service):
        """Test memory size parsing"""
        assert engine_service._parse_memory_size("1.5GiB") == 1536.0  # 1.5 * 1024
        assert engine_service._parse_memory_size("500MiB") == 500.0
        assert engine_service._parse_memory_size("1024KiB") == 1.0
        assert engine_service._parse_memory_size("1048576B") == 1.0
        assert engine_service._parse_memory_size("invalid") == 0.0

    @pytest.mark.asyncio
    async def test_run_command_success(self, engine_service):
        """Test successful command execution"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"success output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await engine_service._run_command(["echo", "test"])

            assert result["success"] is True
            assert result["output"] == "success output"
            assert result["returncode"] == 0

    @pytest.mark.asyncio
    async def test_run_command_failure(self, engine_service):
        """Test failed command execution"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"", b"error output")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            result = await engine_service._run_command(["false"])

            assert result["success"] is False
            assert result["error"] == "error output"
            assert result["returncode"] == 1

    @pytest.mark.asyncio
    async def test_run_command_exception(self, engine_service):
        """Test command execution with exception"""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_subprocess.side_effect = Exception("Command failed")

            result = await engine_service._run_command(["invalid"])

            assert result["success"] is False
            assert "Command failed" in result["error"]
            assert result["returncode"] == -1

    def test_engine_state_enum(self):
        """Test EngineState enum values"""
        assert EngineState.STOPPED.value == "stopped"
        assert EngineState.STARTING.value == "starting"
        assert EngineState.RUNNING.value == "running"
        assert EngineState.STOPPING.value == "stopping"
        assert EngineState.ERROR.value == "error"

    def test_engine_mode_enum(self):
        """Test EngineMode enum values"""
        assert EngineMode.LIVE.value == "live"
        assert EngineMode.PAPER.value == "paper"
        assert EngineMode.BACKTEST.value == "backtest"

    def test_engine_config_defaults(self):
        """Test EngineConfig default values"""
        config = EngineConfig(mode=EngineMode.PAPER)
        
        assert config.mode == EngineMode.PAPER
        assert config.memory_limit == "2g"
        assert config.cpu_limit == 2.0
        assert config.data_catalog_path == "/app/data"
        assert config.cache_enabled is True
        assert config.risk_engine_enabled is True
        assert config.position_limits == {"max_position_size": 100000}
        assert config.venues == {"interactive_brokers": {"enabled": True, "client_id": 1}}

    def test_resource_metrics_dataclass(self):
        """Test ResourceMetrics dataclass"""
        timestamp = datetime.utcnow()
        metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_percent=75.0,
            memory_used_mb=1500.0,
            network_rx_mb=100.0,
            network_tx_mb=50.0,
            timestamp=timestamp
        )

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 75.0
        assert metrics.memory_used_mb == 1500.0
        assert metrics.network_rx_mb == 100.0
        assert metrics.network_tx_mb == 50.0
        assert metrics.timestamp == timestamp

    def test_engine_result_dataclass(self):
        """Test EngineResult dataclass"""
        result = EngineResult(
            success=True,
            message="Operation successful",
            state=EngineState.RUNNING,
            error_details=None
        )

        assert result.success is True
        assert result.message == "Operation successful"
        assert result.state == EngineState.RUNNING
        assert result.error_details is None


@pytest.mark.asyncio
async def test_engine_service_integration():
    """Integration test for engine service lifecycle"""
    service = NautilusEngineService(container_name="test-integration")
    config = EngineConfig(mode=EngineMode.PAPER)

    # Mock all Docker interactions
    with patch.object(service, '_check_container_status') as mock_status, \
         patch.object(service, '_write_engine_config') as mock_write, \
         patch.object(service, '_execute_engine_command') as mock_execute, \
         patch.object(service, '_get_resource_metrics') as mock_metrics:
        
        # Setup mocks
        mock_status.return_value = {"exists": True, "running": True, "status": "running"}
        mock_execute.return_value = {"success": True, "output": "Success"}
        mock_metrics.return_value = ResourceMetrics(
            cpu_percent=25.0, memory_percent=30.0, memory_used_mb=600.0,
            network_rx_mb=10.0, network_tx_mb=5.0, timestamp=datetime.utcnow()
        )

        # Test full lifecycle
        # 1. Start engine
        start_result = await service.start_engine(config)
        assert start_result.success is True
        assert service.current_state == EngineState.RUNNING

        # 2. Get status
        status = await service.get_engine_status()
        assert status.state == EngineState.RUNNING
        assert status.configuration == config

        # 3. Update config
        update_result = await service.update_config(config)
        assert update_result.success is True

        # 4. Stop engine
        stop_result = await service.stop_engine()
        assert stop_result.success is True
        assert service.current_state == EngineState.STOPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])