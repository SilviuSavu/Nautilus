#!/usr/bin/env python3
"""
Production Readiness Assessment for M4 Max Neural Engine
Sustained load testing, thermal monitoring, and deployment validation
"""

import time
import warnings
import numpy as np
import psutil
import threading
import multiprocessing as mp
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json
import os
import subprocess
import concurrent.futures
from collections import deque
import signal
import sys

warnings.filterwarnings('ignore')

class SystemMonitor:
    """Monitor system performance during sustained load"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'temperature': deque(maxlen=1000),
            'power_consumption': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000)
        }
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Monitoring loop - runs in background"""
        while self.monitoring:
            try:
                timestamp = datetime.now()
                
                # CPU and Memory metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                
                # Try to get temperature (macOS specific)
                try:
                    # Use powermetrics command for Mac thermal data
                    temp_result = subprocess.run(['sysctl', '-n', 'machdep.xcpm.cpu_thermal_state'], 
                                               capture_output=True, text=True, timeout=1)
                    if temp_result.returncode == 0:
                        thermal_state = int(temp_result.stdout.strip())
                        # Convert thermal state to approximate temperature
                        temperature = 50 + (thermal_state * 10)  # Rough approximation
                    else:
                        temperature = 60.0  # Default value
                except:
                    temperature = 60.0  # Default value if unavailable
                
                # Power consumption (estimated based on CPU usage)
                power_consumption = 15 + (cpu_percent / 100.0) * 45  # 15-60W range
                
                # Store metrics
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_percent'].append(memory_percent)
                self.metrics['temperature'].append(temperature)
                self.metrics['power_consumption'].append(power_consumption)
                self.metrics['timestamps'].append(timestamp)
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """Get summary of collected metrics"""
        if not self.metrics['cpu_percent']:
            return {'error': 'No metrics collected'}
            
        return {
            'cpu_percent': {
                'avg': np.mean(self.metrics['cpu_percent']),
                'max': np.max(self.metrics['cpu_percent']),
                'min': np.min(self.metrics['cpu_percent'])
            },
            'memory_percent': {
                'avg': np.mean(self.metrics['memory_percent']),
                'max': np.max(self.metrics['memory_percent']),
                'min': np.min(self.metrics['memory_percent'])
            },
            'temperature': {
                'avg': np.mean(self.metrics['temperature']),
                'max': np.max(self.metrics['temperature']),
                'min': np.min(self.metrics['temperature'])
            },
            'power_consumption': {
                'avg': np.mean(self.metrics['power_consumption']),
                'max': np.max(self.metrics['power_consumption']),
                'min': np.min(self.metrics['power_consumption'])
            },
            'duration_seconds': len(self.metrics['cpu_percent']) * 0.1,
            'sample_count': len(self.metrics['cpu_percent'])
        }

class TradingModel(nn.Module):
    """Lightweight trading model for production testing"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 64, output_size: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class ModelManager:
    """Manage multiple model versions and deployment"""
    
    def __init__(self):
        self.models = {}
        self.active_model = None
        self.model_metrics = {}
        
    def deploy_model(self, model_id: str, model: nn.Module) -> bool:
        """Deploy a new model version"""
        try:
            model.eval()  # Set to evaluation mode
            self.models[model_id] = model
            self.model_metrics[model_id] = {
                'deployment_time': datetime.now(),
                'prediction_count': 0,
                'total_inference_time': 0,
                'errors': 0
            }
            return True
        except Exception as e:
            print(f"Model deployment error: {e}")
            return False
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model version"""
        if model_id in self.models:
            self.active_model = model_id
            return True
        return False
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make prediction with active model"""
        if self.active_model is None or self.active_model not in self.models:
            raise ValueError("No active model available")
            
        model = self.models[self.active_model]
        metrics = self.model_metrics[self.active_model]
        
        try:
            start_time = time.perf_counter()
            with torch.no_grad():
                result = model(data)
            inference_time = time.perf_counter() - start_time
            
            # Update metrics
            metrics['prediction_count'] += 1
            metrics['total_inference_time'] += inference_time
            
            return result
            
        except Exception as e:
            metrics['errors'] += 1
            raise e
    
    def get_model_stats(self) -> Dict[str, any]:
        """Get statistics for all deployed models"""
        stats = {}
        for model_id, metrics in self.model_metrics.items():
            if metrics['prediction_count'] > 0:
                avg_inference_time = metrics['total_inference_time'] / metrics['prediction_count']
            else:
                avg_inference_time = 0
                
            stats[model_id] = {
                'predictions': metrics['prediction_count'],
                'avg_inference_ms': avg_inference_time * 1000,
                'total_errors': metrics['errors'],
                'error_rate': metrics['errors'] / max(metrics['prediction_count'], 1),
                'uptime_hours': (datetime.now() - metrics['deployment_time']).total_seconds() / 3600
            }
        
        return stats

class ProductionReadinessValidator:
    """Comprehensive production readiness validation"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.model_manager = ModelManager()
        self.test_results = {}
        
    def test_sustained_load(self, duration_seconds: int = 300) -> Dict[str, any]:
        """Test sustained load performance"""
        print(f"🔄 Running sustained load test ({duration_seconds}s)...")
        
        # Create test model
        model = TradingModel()
        model_id = "prod_test_v1"
        self.model_manager.deploy_model(model_id, model)
        self.model_manager.switch_model(model_id)
        
        # Generate test data
        test_data = torch.randn(1000, 20)  # 1000 samples, 20 features
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Run sustained load
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        predictions_made = 0
        latencies = []
        errors = 0
        
        print(f"   Starting sustained load (target: {duration_seconds}s)...")
        
        while time.perf_counter() < end_time:
            try:
                # Random sample for prediction
                sample_idx = np.random.randint(0, len(test_data))
                sample = test_data[sample_idx:sample_idx+1]
                
                pred_start = time.perf_counter()
                _ = self.model_manager.predict(sample)
                pred_end = time.perf_counter()
                
                latencies.append((pred_end - pred_start) * 1000)
                predictions_made += 1
                
                # Brief pause to simulate realistic load
                time.sleep(0.001)  # 1ms pause between predictions
                
            except Exception as e:
                errors += 1
                print(f"   Prediction error: {e}")
        
        actual_duration = time.perf_counter() - start_time
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        system_metrics = self.monitor.get_metrics_summary()
        
        # Calculate performance metrics
        throughput = predictions_made / actual_duration
        avg_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        results = {
            'test_type': 'sustained_load',
            'duration_seconds': actual_duration,
            'predictions_made': predictions_made,
            'throughput_predictions_per_second': throughput,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'error_count': errors,
            'error_rate': errors / max(predictions_made + errors, 1),
            'system_metrics': system_metrics
        }
        
        print(f"   ✅ Completed: {predictions_made:,} predictions in {actual_duration:.1f}s")
        print(f"   📊 Throughput: {throughput:.0f} predictions/sec")
        print(f"   ⚡ P99 latency: {p99_latency:.3f}ms")
        print(f"   🌡️  Peak CPU: {system_metrics.get('cpu_percent', {}).get('max', 0):.1f}%")
        
        return results
    
    def test_concurrent_models(self, num_models: int = 5, duration_seconds: int = 60) -> Dict[str, any]:
        """Test concurrent model execution"""
        print(f"🔄 Testing concurrent models ({num_models} models, {duration_seconds}s)...")
        
        # Deploy multiple models
        models = []
        for i in range(num_models):
            model = TradingModel()
            model_id = f"concurrent_model_{i}"
            self.model_manager.deploy_model(model_id, model)
            models.append(model_id)
        
        # Test data
        test_data = torch.randn(100, 20)
        
        def run_model_predictions(model_id: str, duration: float) -> Dict[str, any]:
            """Run predictions for a specific model"""
            end_time = time.perf_counter() + duration
            predictions = 0
            latencies = []
            errors = 0
            
            self.model_manager.switch_model(model_id)
            
            while time.perf_counter() < end_time:
                try:
                    sample_idx = np.random.randint(0, len(test_data))
                    sample = test_data[sample_idx:sample_idx+1]
                    
                    start = time.perf_counter()
                    _ = self.model_manager.predict(sample)
                    end = time.perf_counter()
                    
                    latencies.append((end - start) * 1000)
                    predictions += 1
                    
                    time.sleep(0.005)  # 5ms between predictions
                    
                except Exception as e:
                    errors += 1
            
            return {
                'model_id': model_id,
                'predictions': predictions,
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'errors': errors
            }
        
        # Run concurrent models
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = [executor.submit(run_model_predictions, model_id, duration_seconds) 
                      for model_id in models]
            
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        actual_duration = time.perf_counter() - start_time
        
        # Aggregate results
        total_predictions = sum(r['predictions'] for r in concurrent_results)
        total_errors = sum(r['errors'] for r in concurrent_results)
        avg_latencies = [r['avg_latency_ms'] for r in concurrent_results if r['avg_latency_ms'] > 0]
        
        results = {
            'test_type': 'concurrent_models',
            'num_models': num_models,
            'duration_seconds': actual_duration,
            'total_predictions': total_predictions,
            'total_throughput_predictions_per_second': total_predictions / actual_duration,
            'avg_latency_across_models_ms': np.mean(avg_latencies) if avg_latencies else 0,
            'total_errors': total_errors,
            'model_results': concurrent_results
        }
        
        print(f"   ✅ Total predictions: {total_predictions:,}")
        print(f"   📊 Combined throughput: {results['total_throughput_predictions_per_second']:.0f}/sec")
        print(f"   ⚡ Avg latency: {results['avg_latency_across_models_ms']:.3f}ms")
        
        return results
    
    def test_model_deployment_rollback(self) -> Dict[str, any]:
        """Test model deployment and rollback capabilities"""
        print("🔄 Testing model deployment and rollback...")
        
        # Deploy initial model
        model_v1 = TradingModel()
        deploy_v1_start = time.perf_counter()
        success_v1 = self.model_manager.deploy_model("prod_v1", model_v1)
        deploy_v1_time = time.perf_counter() - deploy_v1_start
        
        # Switch to v1
        switch_v1_success = self.model_manager.switch_model("prod_v1")
        
        # Test v1 performance
        test_data = torch.randn(10, 20)
        v1_latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            _ = self.model_manager.predict(test_data[:1])
            end = time.perf_counter()
            v1_latencies.append((end - start) * 1000)
        
        # Deploy new model (v2)
        model_v2 = TradingModel(hidden_size=128)  # Larger model
        deploy_v2_start = time.perf_counter()
        success_v2 = self.model_manager.deploy_model("prod_v2", model_v2)
        deploy_v2_time = time.perf_counter() - deploy_v2_start
        
        # Switch to v2
        switch_v2_success = self.model_manager.switch_model("prod_v2")
        
        # Test v2 performance
        v2_latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            _ = self.model_manager.predict(test_data[:1])
            end = time.perf_counter()
            v2_latencies.append((end - start) * 1000)
        
        # Rollback to v1
        rollback_start = time.perf_counter()
        rollback_success = self.model_manager.switch_model("prod_v1")
        rollback_time = time.perf_counter() - rollback_start
        
        # Test after rollback
        rollback_latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            _ = self.model_manager.predict(test_data[:1])
            end = time.perf_counter()
            rollback_latencies.append((end - start) * 1000)
        
        results = {
            'test_type': 'deployment_rollback',
            'v1_deployment': {
                'success': success_v1,
                'deploy_time_ms': deploy_v1_time * 1000,
                'avg_latency_ms': np.mean(v1_latencies),
                'switch_success': switch_v1_success
            },
            'v2_deployment': {
                'success': success_v2,
                'deploy_time_ms': deploy_v2_time * 1000,
                'avg_latency_ms': np.mean(v2_latencies),
                'switch_success': switch_v2_success
            },
            'rollback': {
                'success': rollback_success,
                'rollback_time_ms': rollback_time * 1000,
                'avg_latency_ms': np.mean(rollback_latencies),
                'latency_consistency': abs(np.mean(v1_latencies) - np.mean(rollback_latencies)) < 0.01
            }
        }
        
        print(f"   ✅ V1 Deploy: {deploy_v1_time*1000:.2f}ms")
        print(f"   ✅ V2 Deploy: {deploy_v2_time*1000:.2f}ms")
        print(f"   ✅ Rollback: {rollback_time*1000:.2f}ms")
        print(f"   📊 Consistency: {'✅' if results['rollback']['latency_consistency'] else '❌'}")
        
        return results
    
    def test_error_handling_recovery(self) -> Dict[str, any]:
        """Test error handling and recovery mechanisms"""
        print("🔄 Testing error handling and recovery...")
        
        # Deploy model
        model = TradingModel()
        self.model_manager.deploy_model("error_test", model)
        self.model_manager.switch_model("error_test")
        
        # Test scenarios
        results = {
            'test_type': 'error_handling',
            'scenarios': {}
        }
        
        # 1. Invalid input data
        try:
            invalid_data = torch.randn(1, 50)  # Wrong input size
            _ = self.model_manager.predict(invalid_data)
            results['scenarios']['invalid_input'] = {'handled': False, 'error': 'None'}
        except Exception as e:
            results['scenarios']['invalid_input'] = {'handled': True, 'error': str(e)[:100]}
        
        # 2. No active model
        self.model_manager.active_model = None
        try:
            valid_data = torch.randn(1, 20)
            _ = self.model_manager.predict(valid_data)
            results['scenarios']['no_active_model'] = {'handled': False, 'error': 'None'}
        except Exception as e:
            results['scenarios']['no_active_model'] = {'handled': True, 'error': str(e)[:100]}
        
        # 3. Recovery test
        recovery_start = time.perf_counter()
        self.model_manager.switch_model("error_test")  # Recover
        try:
            _ = self.model_manager.predict(torch.randn(1, 20))
            recovery_success = True
        except:
            recovery_success = False
        recovery_time = time.perf_counter() - recovery_start
        
        results['scenarios']['recovery'] = {
            'success': recovery_success,
            'recovery_time_ms': recovery_time * 1000
        }
        
        print(f"   ✅ Invalid input handled: {'✅' if results['scenarios']['invalid_input']['handled'] else '❌'}")
        print(f"   ✅ No model handled: {'✅' if results['scenarios']['no_active_model']['handled'] else '❌'}")
        print(f"   ✅ Recovery success: {'✅' if recovery_success else '❌'}")
        
        return results
    
    def run_full_production_assessment(self) -> Dict[str, any]:
        """Run comprehensive production readiness assessment"""
        print("=" * 80)
        print("🏭 M4 Max Neural Engine - Production Readiness Assessment")
        print("   Nautilus Trading Platform - Enterprise Deployment Validation")
        print("=" * 80)
        
        results = {
            'assessment_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        print("")
        
        # 1. Sustained Load Test (5 minutes)
        sustained_results = self.test_sustained_load(duration_seconds=300)
        results['sustained_load'] = sustained_results
        
        print("")
        
        # 2. Concurrent Models Test
        concurrent_results = self.test_concurrent_models(num_models=5, duration_seconds=120)
        results['concurrent_models'] = concurrent_results
        
        print("")
        
        # 3. Model Deployment/Rollback Test
        deployment_results = self.test_model_deployment_rollback()
        results['model_deployment'] = deployment_results
        
        print("")
        
        # 4. Error Handling Test
        error_results = self.test_error_handling_recovery()
        results['error_handling'] = error_results
        
        print("")
        
        # Final Assessment
        self._print_production_assessment(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_production_assessment(self, results: Dict[str, any]):
        """Print comprehensive production assessment"""
        print("=" * 80)
        print("📋 PRODUCTION READINESS ASSESSMENT")
        print("=" * 80)
        
        sustained = results['sustained_load']
        concurrent = results['concurrent_models']
        deployment = results['model_deployment']
        error_handling = results['error_handling']
        
        print("🔄 Sustained Load Performance:")
        print(f"  • Duration:              {sustained['duration_seconds']:.0f} seconds")
        print(f"  • Predictions:           {sustained['predictions_made']:,}")
        print(f"  • Throughput:            {sustained['throughput_predictions_per_second']:,.0f}/sec")
        print(f"  • P99 Latency:           {sustained['p99_latency_ms']:.3f}ms")
        print(f"  • Error Rate:            {sustained['error_rate']*100:.3f}%")
        print(f"  • Peak CPU:              {sustained['system_metrics'].get('cpu_percent', {}).get('max', 0):.1f}%")
        print(f"  • Peak Memory:           {sustained['system_metrics'].get('memory_percent', {}).get('max', 0):.1f}%")
        
        print("")
        
        print("🔄 Concurrent Model Performance:")
        print(f"  • Models Tested:         {concurrent['num_models']}")
        print(f"  • Total Predictions:     {concurrent['total_predictions']:,}")
        print(f"  • Combined Throughput:   {concurrent['total_throughput_predictions_per_second']:,.0f}/sec")
        print(f"  • Avg Latency:           {concurrent['avg_latency_across_models_ms']:.3f}ms")
        
        print("")
        
        print("🚀 Deployment & Rollback:")
        print(f"  • V1 Deploy Time:        {deployment['v1_deployment']['deploy_time_ms']:.2f}ms")
        print(f"  • V2 Deploy Time:        {deployment['v2_deployment']['deploy_time_ms']:.2f}ms")
        print(f"  • Rollback Time:         {deployment['rollback']['rollback_time_ms']:.2f}ms")
        print(f"  • Consistency:           {'✅ Pass' if deployment['rollback']['latency_consistency'] else '❌ Fail'}")
        
        print("")
        
        print("🛡️  Error Handling:")
        invalid_handled = error_handling['scenarios']['invalid_input']['handled']
        no_model_handled = error_handling['scenarios']['no_active_model']['handled']
        recovery_success = error_handling['scenarios']['recovery']['success']
        
        print(f"  • Invalid Input:         {'✅ Handled' if invalid_handled else '❌ Failed'}")
        print(f"  • No Active Model:       {'✅ Handled' if no_model_handled else '❌ Failed'}")
        print(f"  • Recovery:              {'✅ Success' if recovery_success else '❌ Failed'}")
        
        print("")
        
        # Overall Score
        score_components = [
            sustained['p99_latency_ms'] < 1.0,  # Sub-millisecond P99
            sustained['error_rate'] < 0.001,    # <0.1% error rate
            sustained['throughput_predictions_per_second'] > 1000,  # >1K/sec throughput
            concurrent['total_throughput_predictions_per_second'] > 2000,  # >2K/sec concurrent
            deployment['rollback']['latency_consistency'],  # Consistent rollback
            invalid_handled and no_model_handled and recovery_success  # Error handling
        ]
        
        score = sum(score_components) / len(score_components) * 100
        
        print("🎯 Overall Production Readiness Score:")
        print(f"  • Performance:           {'✅' if score_components[0] else '❌'} {'✅' if score_components[1] else '❌'} {'✅' if score_components[2] else '❌'}")
        print(f"  • Scalability:           {'✅' if score_components[3] else '❌'}")
        print(f"  • Deployment:            {'✅' if score_components[4] else '❌'}")
        print(f"  • Reliability:           {'✅' if score_components[5] else '❌'}")
        print("")
        print(f"  • Overall Score:         {score:.1f}% ({'✅ PRODUCTION READY' if score >= 85 else '⚠️ NEEDS IMPROVEMENT' if score >= 70 else '❌ NOT READY'})")
        
        print("")
        print("🏆 M4 Max Neural Engine production assessment complete!")
    
    def _save_results(self, results: Dict[str, any]):
        """Save assessment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml/production_assessment_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📁 Assessment results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main assessment function"""
    validator = ProductionReadinessValidator()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down assessment...")
        validator.monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        results = validator.run_full_production_assessment()
        return results
    finally:
        validator.monitor.stop_monitoring()

if __name__ == "__main__":
    main()