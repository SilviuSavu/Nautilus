"""
Comprehensive Test Suite for Nautilus Hybrid Architecture
Tests all components: Circuit Breakers, Health Monitor, Enhanced Gateway, 
Hybrid Router, and Integration scenarios.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import components to test
from ..circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenException,
    circuit_breaker_registry, CircuitState
)
from ..health_monitor import (
    EngineHealthChecker, EngineStatus, EngineMetrics, health_monitor
)
from ..enhanced_gateway import (
    EnhancedAPIGateway, RequestPriority, CacheStrategy, enhanced_gateway
)
from ..hybrid_router import (
    IntelligentHybridRouter, RoutingStrategy, OperationCategory,
    RoutingDecision, hybrid_router
)


class TestCircuitBreaker:
    """Test Circuit Breaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5,
            success_threshold=2,
            timeout=1.0
        )
        return CircuitBreaker("test-engine", config)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, circuit_breaker):
        """Test successful circuit breaker operation"""
        async def successful_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self, circuit_breaker):
        """Test circuit breaker failure handling"""
        async def failing_operation():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.metrics.failed_requests == 1
        assert circuit_breaker.metrics.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_after_failures(self, circuit_breaker):
        """Test circuit breaker opens after threshold failures"""
        async def failing_operation():
            raise Exception("Test failure")
        
        # Cause failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self, circuit_breaker):
        """Test circuit breaker timeout handling"""
        async def slow_operation():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_operation)
        
        assert circuit_breaker.metrics.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_context_manager(self, circuit_breaker):
        """Test circuit breaker context manager usage"""
        async with circuit_breaker as cb:
            # Simulate successful operation
            await asyncio.sleep(0.1)
        
        assert circuit_breaker.metrics.successful_requests == 1
        
        # Test failure in context manager
        try:
            async with circuit_breaker as cb:
                raise Exception("Test failure")
        except Exception:
            pass
        
        assert circuit_breaker.metrics.failed_requests == 1


class TestEngineHealthChecker:
    """Test Engine Health Checker functionality"""
    
    @pytest.fixture
    def health_checker(self):
        return EngineHealthChecker(check_interval=1)  # 1 second for testing
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self, health_checker):
        """Test health checker initialization"""
        assert len(health_checker.engines) == 9
        assert "strategy" in health_checker.engines
        assert "risk" in health_checker.engines
        assert not health_checker.running
    
    @pytest.mark.asyncio
    async def test_engine_health_check_success(self, health_checker):
        """Test successful engine health check"""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock successful health check response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "uptime_seconds": 3600,
                "cpu_percent": 25.0
            }
            mock_get.return_value = mock_response
            
            await health_checker._check_engine_health("strategy")
            
            engine = health_checker.engines["strategy"]
            assert engine.status in [EngineStatus.HEALTHY, EngineStatus.DEGRADED]
            assert engine.total_requests == 1
            assert engine.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_engine_health_check_failure(self, health_checker):
        """Test failed engine health check"""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock connection error
            mock_get.side_effect = Exception("Connection refused")
            
            await health_checker._check_engine_health("strategy")
            
            engine = health_checker.engines["strategy"]
            assert engine.status == EngineStatus.UNHEALTHY
            assert engine.total_requests == 1
            assert engine.failed_requests == 1
            assert "Connection refused" in engine.last_error
    
    @pytest.mark.asyncio
    async def test_get_healthy_engines(self, health_checker):
        """Test getting list of healthy engines"""
        # Set some engines as healthy
        health_checker.engines["strategy"].status = EngineStatus.HEALTHY
        health_checker.engines["risk"].status = EngineStatus.HEALTHY
        health_checker.engines["analytics"].status = EngineStatus.UNHEALTHY
        
        healthy = await health_checker.get_healthy_engines()
        assert "strategy" in healthy
        assert "risk" in healthy
        assert "analytics" not in healthy
    
    @pytest.mark.asyncio
    async def test_system_health_summary(self, health_checker):
        """Test system health summary generation"""
        # Set various engine states
        health_checker.engines["strategy"].status = EngineStatus.HEALTHY
        health_checker.engines["risk"].status = EngineStatus.HEALTHY
        health_checker.engines["analytics"].status = EngineStatus.DEGRADED
        health_checker.engines["ml"].status = EngineStatus.UNHEALTHY
        
        summary = await health_checker.get_system_health_summary()
        
        assert summary["total_engines"] == 9
        assert summary["healthy_engines"] == 2
        assert summary["degraded_engines"] == 1
        assert summary["unhealthy_engines"] == 1
        assert summary["overall_status"] in ["healthy", "degraded", "unhealthy"]


class TestEnhancedGateway:
    """Test Enhanced Gateway functionality"""
    
    @pytest.fixture
    def gateway(self):
        return EnhancedAPIGateway()
    
    @pytest.mark.asyncio
    async def test_gateway_initialization(self, gateway):
        """Test gateway initialization"""
        assert len(gateway.routing_configs) == 9
        assert "strategy" in gateway.routing_configs
        assert gateway.routing_configs["strategy"].priority == RequestPriority.CRITICAL
        assert gateway.metrics["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_gateway_request_routing(self, gateway):
        """Test gateway request routing"""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock successful engine response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_get.return_value = mock_response
            
            # Mock connection manager
            gateway.connection_manager.get_client = AsyncMock(return_value=Mock())
            
            # Mock health monitor
            with patch('backend.hybrid_architecture.health_monitor.health_monitor.get_engine_health') as mock_health:
                mock_health.return_value = Mock(status=EngineStatus.HEALTHY)
                
                result = await gateway.route_request("strategy", "/execute", "POST", {"order": "buy"})
                
                assert result["success"] == True
                assert "data" in result
                assert "metadata" in result
                assert gateway.metrics["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_gateway_circuit_breaker_integration(self, gateway):
        """Test gateway integration with circuit breakers"""
        with patch('backend.hybrid_architecture.circuit_breaker.circuit_breaker_registry.get_or_create') as mock_breaker:
            mock_cb = Mock()
            mock_cb.call = AsyncMock(side_effect=Exception("Service unavailable"))
            mock_breaker.return_value = mock_cb
            
            result = await gateway.route_request("strategy", "/execute", "POST")
            
            assert result["success"] == False
            assert "error" in result
            assert gateway.metrics["failed_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_gateway_caching(self, gateway):
        """Test gateway caching functionality"""
        # Mock cache get/set
        gateway.cache.get = AsyncMock(return_value=None)  # Cache miss
        gateway.cache.set = AsyncMock()
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "cached_data"}
            mock_get.return_value = mock_response
            
            gateway.connection_manager.get_client = AsyncMock(return_value=Mock())
            
            with patch('backend.hybrid_architecture.health_monitor.health_monitor.get_engine_health') as mock_health:
                mock_health.return_value = Mock(status=EngineStatus.HEALTHY)
                
                # First request - should cache
                result1 = await gateway.route_request("analytics", "/data", "GET")
                
                # Verify cache.set was called
                gateway.cache.set.assert_called_once()
                
                # Second request - should use cache
                gateway.cache.get = AsyncMock(return_value={"result": "cached_data"})
                result2 = await gateway.route_request("analytics", "/data", "GET")
                
                assert result2["metadata"]["cached"] == True


class TestHybridRouter:
    """Test Hybrid Router functionality"""
    
    @pytest.fixture
    def router(self):
        return IntelligentHybridRouter()
    
    @pytest.mark.asyncio
    async def test_router_initialization(self, router):
        """Test router initialization"""
        assert len(router.routing_rules) > 0
        assert len(router.performance_profiles) == 9
        assert router.enabled == True
        assert "strategy" in router.performance_profiles
    
    @pytest.mark.asyncio
    async def test_routing_decision_critical_trading(self, router):
        """Test routing decision for critical trading operations"""
        # Mock healthy strategy engine
        router.performance_profiles["strategy"].health_status = EngineStatus.HEALTHY
        router.performance_profiles["strategy"].direct_avg_latency_ms = 25.0
        router.performance_profiles["strategy"].current_load = 30.0
        
        decision = await router.make_routing_decision(
            engine="strategy",
            endpoint="/execute",
            method="POST",
            priority=RequestPriority.CRITICAL
        )
        
        assert decision.target_engine == "strategy"
        assert decision.expected_latency_ms <= 100  # Should be fast
        assert decision.confidence > 0.5
        assert "critical" in decision.reasoning.lower() or "strategy" in decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_routing_decision_background_processing(self, router):
        """Test routing decision for background processing"""
        decision = await router.make_routing_decision(
            engine="factor",
            endpoint="/calculate",
            method="GET",
            priority=RequestPriority.NORMAL
        )
        
        # Background processing should prefer gateway
        assert decision.strategy in [RoutingStrategy.GATEWAY_ONLY, RoutingStrategy.HYBRID_INTELLIGENT]
        assert decision.expected_latency_ms <= 2000
    
    @pytest.mark.asyncio
    async def test_routing_decision_unhealthy_engine(self, router):
        """Test routing decision with unhealthy engine"""
        # Set engine as unhealthy
        router.performance_profiles["strategy"].health_status = EngineStatus.UNHEALTHY
        router.performance_profiles["strategy"].current_load = 95.0
        
        decision = await router.make_routing_decision(
            engine="strategy",
            endpoint="/execute",
            method="POST"
        )
        
        # Should prefer gateway or have low confidence
        assert decision.confidence < 0.8 or not decision.use_direct_access
    
    @pytest.mark.asyncio
    async def test_routing_outcome_recording(self, router):
        """Test recording routing outcomes for learning"""
        decision = RoutingDecision(
            strategy=RoutingStrategy.HYBRID_PERFORMANCE,
            target_engine="strategy",
            use_direct_access=True,
            use_gateway=False,
            enable_fallback=True,
            expected_latency_ms=50,
            confidence=0.9,
            reasoning="Test decision"
        )
        
        # Record successful outcome
        router.record_routing_outcome(decision, 45.0, True)
        
        # Check metrics were updated
        assert router.metrics.strategy_counts[RoutingStrategy.HYBRID_PERFORMANCE] == 1
        assert len(router.metrics.latency_by_strategy[RoutingStrategy.HYBRID_PERFORMANCE]) == 1
        assert router.metrics.success_rates[RoutingStrategy.HYBRID_PERFORMANCE]["success"] == 1
    
    @pytest.mark.asyncio
    async def test_intelligent_score_calculation(self, router):
        """Test intelligent routing score calculation"""
        factors = {
            "health_status": "healthy",
            "current_load": 40.0,
            "direct_score": 80.0,
            "gateway_score": 60.0,
            "priority": "critical",
            "max_latency_ms": 50,
            "time_of_day": 14,  # 2 PM - trading hours
            "context": {}
        }
        
        score = router._calculate_intelligent_score(factors)
        
        # Should prefer direct access for healthy, critical, trading hours
        assert 0.0 <= score <= 1.0
        assert score > 0.6  # Should be biased toward direct access
    
    @pytest.mark.asyncio
    async def test_routing_rule_matching(self, router):
        """Test routing rule matching logic"""
        # Test exact match
        rule = router._match_routing_rule("strategy", "/execute")
        assert rule is not None
        assert rule.category == OperationCategory.CRITICAL_TRADING
        
        # Test wildcard match
        rule = router._match_routing_rule("factor", "/anything")
        assert rule is not None
        assert rule.category == OperationCategory.BACKGROUND_PROCESSING
        
        # Test no match
        rule = router._match_routing_rule("nonexistent", "/test")
        assert rule is None


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_operation(self):
        """Test complete trading operation flow"""
        # Mock all external dependencies
        with patch('httpx.AsyncClient.get') as mock_get, \
             patch('httpx.AsyncClient.post') as mock_post:
            
            # Mock health check response
            mock_health_response = Mock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {"status": "healthy", "uptime_seconds": 3600}
            mock_get.return_value = mock_health_response
            
            # Mock trading operation response
            mock_trade_response = Mock()
            mock_trade_response.status_code = 200
            mock_trade_response.json.return_value = {"order_id": "12345", "status": "executed"}
            mock_post.return_value = mock_trade_response
            
            # Test the full flow
            # 1. Health check should succeed
            await health_monitor.force_health_check("strategy")
            
            # 2. Routing decision should prefer direct access for critical trading
            decision = await hybrid_router.make_routing_decision(
                engine="strategy",
                endpoint="/execute",
                method="POST",
                priority=RequestPriority.CRITICAL
            )
            
            assert decision.target_engine == "strategy"
            
            # 3. Gateway should route the request appropriately
            enhanced_gateway.connection_manager.get_client = AsyncMock(return_value=Mock())
            
            result = await enhanced_gateway.route_request(
                engine="strategy",
                endpoint="/execute",
                method="POST",
                data={"symbol": "AAPL", "quantity": 100},
                priority=RequestPriority.CRITICAL
            )
            
            assert result["success"] == True
            assert "order_id" in result["data"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_scenario(self):
        """Test circuit breaker opening and fallback behavior"""
        # Create circuit breaker that fails quickly
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker("test-engine", config)
        
        async def failing_operation():
            raise Exception("Service unavailable")
        
        # Trigger circuit breaker opening
        for i in range(3):
            try:
                await cb.call(failing_operation)
            except:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Test that subsequent calls raise CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            await cb.call(failing_operation)
    
    @pytest.mark.asyncio
    async def test_performance_degradation_routing(self):
        """Test routing behavior under performance degradation"""
        # Simulate high load conditions
        hybrid_router.performance_profiles["strategy"].current_load = 90.0
        hybrid_router.performance_profiles["strategy"].direct_avg_latency_ms = 150.0
        hybrid_router.performance_profiles["strategy"].health_status = EngineStatus.DEGRADED
        
        decision = await hybrid_router.make_routing_decision(
            engine="strategy",
            endpoint="/execute",
            method="POST",
            priority=RequestPriority.CRITICAL
        )
        
        # Under high load, should either use gateway or have low confidence
        assert decision.confidence < 0.9 or not decision.use_direct_access
    
    @pytest.mark.asyncio
    async def test_load_balancing_scenario(self):
        """Test load balancing across multiple engine instances"""
        load_balancer = hybrid_router.load_balancer
        
        # Set up different load levels
        load_balancer.update_engine_load("strategy-1", 30.0, 40.0)
        load_balancer.update_engine_load("strategy-2", 70.0, 60.0)
        load_balancer.update_engine_load("strategy-3", 50.0, 45.0)
        
        engines = ["strategy-1", "strategy-2", "strategy-3"]
        
        # Test performance-based selection
        selected = load_balancer.select_engine_performance_based(engines)
        
        # Should select strategy-1 (lowest load)
        assert selected == "strategy-1"
        
        # Test round-robin selection
        selections = []
        for i in range(6):
            selected = load_balancer.select_engine_round_robin(engines)
            selections.append(selected)
        
        # Should cycle through all engines
        assert len(set(selections)) == 3
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness_scenario(self):
        """Test caching effectiveness in real scenarios"""
        cache = enhanced_gateway.cache
        
        # Test cache miss then hit
        result = await cache.get("analytics", "/data", {"param": "value"})
        assert result is None  # Cache miss
        
        # Set cache
        await cache.set("analytics", "/data", {"result": "cached"}, CacheStrategy.MEDIUM, {"param": "value"})
        
        # Get from cache
        result = await cache.get("analytics", "/data", {"param": "value"})
        assert result == {"result": "cached"}  # Cache hit


# Test Configuration and Fixtures

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def reset_components():
    """Reset all components before each test"""
    # Reset circuit breakers
    await circuit_breaker_registry.reset_all()
    
    # Reset metrics
    enhanced_gateway.metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "circuit_breaker_opens": 0,
        "fallback_used": 0,
        "average_response_time": 0.0,
        "response_times": []
    }
    
    # Reset routing metrics
    hybrid_router.metrics = hybrid_router.metrics.__class__()
    
    yield
    
    # Cleanup after test
    pass


# Performance Benchmarks

class TestPerformanceBenchmarks:
    """Performance benchmarks for hybrid architecture"""
    
    @pytest.mark.asyncio
    async def test_routing_decision_performance(self):
        """Test routing decision performance"""
        start_time = time.time()
        
        # Make 1000 routing decisions
        for i in range(1000):
            await hybrid_router.make_routing_decision(
                engine="strategy",
                endpoint="/execute",
                method="POST"
            )
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / 1000) * 1000
        
        # Should be able to make routing decisions in < 1ms on average
        assert avg_time_ms < 1.0, f"Routing decision too slow: {avg_time_ms:.2f}ms average"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance overhead"""
        cb = CircuitBreaker("perf-test")
        
        async def fast_operation():
            return "success"
        
        start_time = time.time()
        
        # Execute 10000 operations through circuit breaker
        for i in range(10000):
            await cb.call(fast_operation)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / 10000) * 1000
        
        # Circuit breaker overhead should be minimal (< 0.1ms)
        assert avg_time_ms < 0.1, f"Circuit breaker overhead too high: {avg_time_ms:.3f}ms average"
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance"""
        health_checker = EngineHealthChecker()
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            start_time = time.time()
            
            # Check all engines
            await health_checker._check_all_engines()
            
            total_time = time.time() - start_time
            
            # Health check of all 9 engines should complete in < 1 second
            assert total_time < 1.0, f"Health check too slow: {total_time:.2f}s for 9 engines"


if __name__ == "__main__":
    # Run tests with pytest
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])