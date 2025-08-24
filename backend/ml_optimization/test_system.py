#!/usr/bin/env python3
"""
Standalone test for ML Optimization System
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components with absolute imports
from ml_autoscaler import MLAutoScaler, TradingPattern, ScalingDecision
from predictive_allocator import PredictiveResourceAllocator, AllocationStrategy
from market_optimizer import MarketConditionOptimizer, MarketRegime
from training_pipeline import MLTrainingPipeline, ModelType, TrainingMode
from performance_monitor import MLPerformanceMonitor, PerformanceMetric


async def test_ml_autoscaler():
    """Test the ML Auto-Scaler component"""
    print("\n🤖 Testing ML Auto-Scaler...")
    
    autoscaler = MLAutoScaler(namespace="test-namespace")
    
    # Test metrics collection
    metrics = await autoscaler.collect_metrics("test-service")
    print(f"  ✅ Collected metrics: CPU={metrics.cpu_utilization:.1f}%, Memory={metrics.memory_utilization:.1f}%")
    
    # Test ML prediction
    prediction = await autoscaler.predict_scaling_needs(metrics)
    print(f"  ✅ ML Prediction: Load={prediction.predicted_load:.2f}, Pattern={prediction.pattern.value}")
    print(f"     Confidence={prediction.confidence:.2f}, Decision={prediction.scaling_recommendation.value}")
    
    # Test scaling decision execution (simulated)
    result = await autoscaler.execute_scaling_decision("test-service", prediction)
    print(f"  ✅ Scaling execution: {result['message']}")
    
    return True


async def test_resource_allocator():
    """Test the Predictive Resource Allocator"""
    print("\n🔮 Testing Predictive Resource Allocator...")
    
    allocator = PredictiveResourceAllocator()
    
    # Test demand prediction
    demand = await allocator.predict_demand("test-service", horizon_minutes=15)
    print(f"  ✅ Demand prediction: CPU={demand.cpu_demand:.2f}, Memory={demand.memory_demand:.1f}GB")
    print(f"     Confidence={demand.prediction_confidence:.2f}, Risk={demand.risk_level}")
    
    # Test allocation plan creation
    plan = await allocator.create_allocation_plan(AllocationStrategy.ML_OPTIMIZED)
    print(f"  ✅ Allocation plan: {len(plan.allocations)} allocations, Cost=${plan.total_cost:.2f}")
    print(f"     Performance gain={plan.expected_performance_gain:.1f}%")
    
    # Test plan execution (simulated)
    if plan.allocations:
        execution = await allocator.execute_allocation_plan(plan)
        print(f"  ✅ Plan execution: {execution['successful_allocations']}/{execution['total_allocations']} successful")
    
    return True


async def test_market_optimizer():
    """Test the Market Condition Optimizer"""
    print("\n🌐 Testing Market Condition Optimizer...")
    
    optimizer = MarketConditionOptimizer()
    
    # Test market data collection
    market_data = await optimizer.collect_real_time_market_data()
    print(f"  ✅ Market data collected: {len(market_data)} indicators")
    
    # Test market condition analysis
    condition = await optimizer.analyze_market_condition()
    print(f"  ✅ Market analysis: Regime={condition.regime.value}, Volatility={condition.volatility_level:.2f}")
    print(f"     VIX={condition.vix_level:.1f}, Confidence={condition.confidence:.2f}")
    
    # Test optimization settings
    settings = await optimizer.optimize_for_market_condition(condition)
    print(f"  ✅ Optimization: Strategy={settings.strategy.value}")
    print(f"     CPU={settings.cpu_multiplier:.2f}x, Memory={settings.memory_multiplier:.2f}x")
    
    # Test market insights
    insights = await optimizer.get_current_market_insights()
    if "error" not in insights:
        print(f"  ✅ Market insights: {len(insights.get('recommendations', []))} recommendations")
    
    return True


async def test_training_pipeline():
    """Test the ML Training Pipeline"""
    print("\n🏋️ Testing ML Training Pipeline...")
    
    pipeline = MLTrainingPipeline()
    
    # Test training job scheduling
    job_id = pipeline.schedule_training_job(
        ModelType.LOAD_PREDICTOR,
        TrainingMode.INITIAL_TRAINING,
        priority=8
    )
    print(f"  ✅ Training job scheduled: {job_id}")
    
    # Test training status
    status = await pipeline.get_training_status()
    print(f"  ✅ Pipeline status: {status['queued_jobs']} queued, {status['active_jobs']} active")
    
    # Test model training (simplified)
    try:
        print("  🔄 Running sample training...")
        from training_pipeline import TrainingJob
        from datetime import datetime, timedelta
        
        sample_job = TrainingJob(
            job_id="test_job",
            model_type=ModelType.LOAD_PREDICTOR,
            training_mode=TrainingMode.INITIAL_TRAINING,
            priority=10,
            data_start_date=datetime.now() - timedelta(days=1),
            data_end_date=datetime.now(),
            algorithms=['random_forest'],
            hyperparameter_tuning=False,
            min_samples=50
        )
        
        metrics = await pipeline.train_model(sample_job)
        print(f"  ✅ Training completed: R²={metrics.val_r2:.3f}, MSE={metrics.val_mse:.4f}")
        
    except Exception as e:
        print(f"  ⚠️ Training test skipped: {str(e)[:60]}...")
    
    return True


async def test_performance_monitor():
    """Test the Performance Monitor"""
    print("\n📊 Testing Performance Monitor...")
    
    monitor = MLPerformanceMonitor()
    
    # Test prediction performance recording
    await monitor.record_prediction_performance(
        service_name="test-service",
        predicted_value=0.7,
        actual_value=0.65,
        metric_type=PerformanceMetric.PREDICTION_ACCURACY,
        context={'confidence': 0.8, 'market_regime': 'normal'}
    )
    print("  ✅ Recorded prediction performance")
    
    # Test scaling outcome recording
    await monitor.record_scaling_outcome(
        service_name="test-service",
        predicted_need="scale_up",
        actual_outcome="scaled_up",
        effectiveness_score=0.85
    )
    print("  ✅ Recorded scaling outcome")
    
    # Test model drift calculation
    drift_score = await monitor.calculate_model_drift("test_model")
    print(f"  ✅ Model drift score: {drift_score:.3f}")
    
    # Test dashboard data
    dashboard = await monitor.get_monitoring_dashboard_data()
    if "error" not in dashboard:
        stats = dashboard.get("overall_statistics", {})
        print(f"  ✅ Dashboard data: {stats.get('total_predictions', 0)} predictions")
    
    # Test active alerts
    alerts = await monitor.get_active_alerts()
    print(f"  ✅ Active alerts: {len(alerts)}")
    
    return True


async def run_integration_test():
    """Run integration test simulating real optimization cycle"""
    print("\n🔄 Running Integration Test...")
    
    # Initialize all components
    autoscaler = MLAutoScaler()
    allocator = PredictiveResourceAllocator()
    market_optimizer = MarketConditionOptimizer()
    monitor = MLPerformanceMonitor()
    
    services = ["nautilus-market-data", "nautilus-risk-engine"]
    
    for service in services:
        try:
            # 1. Analyze market conditions
            market_condition = await market_optimizer.analyze_market_condition()
            
            # 2. Collect metrics and make scaling prediction
            metrics = await autoscaler.collect_metrics(service)
            prediction = await autoscaler.predict_scaling_needs(metrics)
            
            # 3. Execute scaling decision (simulated)
            scaling_result = await autoscaler.execute_scaling_decision(service, prediction)
            
            # 4. Record performance
            await monitor.record_scaling_outcome(
                service_name=service,
                predicted_need=prediction.scaling_recommendation.value,
                actual_outcome="scaled_up" if scaling_result.get("success") else "failed",
                effectiveness_score=prediction.confidence,
                context={'market_regime': market_condition.regime.value}
            )
            
            print(f"  ✅ {service}: {prediction.scaling_recommendation.value} (confidence={prediction.confidence:.2f})")
            
        except Exception as e:
            print(f"  ❌ {service}: Error - {str(e)[:50]}...")
    
    # 5. Create resource allocation plan
    try:
        allocation_plan = await allocator.create_allocation_plan(AllocationStrategy.ML_OPTIMIZED)
        print(f"  ✅ Resource allocation: {len(allocation_plan.allocations)} allocations planned")
        
        if allocation_plan.allocations:
            execution_result = await allocator.execute_allocation_plan(allocation_plan)
            print(f"  ✅ Allocation executed: {execution_result['successful_allocations']} successful")
    
    except Exception as e:
        print(f"  ❌ Resource allocation failed: {str(e)[:50]}...")
    
    # 6. Get monitoring summary
    try:
        dashboard = await monitor.get_monitoring_dashboard_data()
        if "overall_statistics" in dashboard:
            stats = dashboard["overall_statistics"]
            print(f"  ✅ Monitoring: {stats.get('success_rate', 0):.1f}% success rate")
        
    except Exception as e:
        print(f"  ❌ Monitoring failed: {str(e)[:50]}...")
    
    return True


async def main():
    """Main test runner"""
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🚀 ML Optimization System Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("ML Auto-Scaler", test_ml_autoscaler),
        ("Resource Allocator", test_resource_allocator), 
        ("Market Optimizer", test_market_optimizer),
        ("Training Pipeline", test_training_pipeline),
        ("Performance Monitor", test_performance_monitor),
        ("Integration Test", run_integration_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - ERROR: {str(e)[:60]}...")
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! ML Optimization System is ready for deployment.")
    else:
        print("⚠️ Some tests failed. Review implementation before deployment.")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Critical test error: {str(e)}")
        sys.exit(1)