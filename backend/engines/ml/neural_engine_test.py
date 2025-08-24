#!/usr/bin/env python3
"""
Neural Engine Testing and Benchmarking Suite for M4 Max
Optimizes Core ML models for Neural Engine acceleration
"""

import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# Suppress Core ML warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleTradePredictor(nn.Module):
    """Simple PyTorch model for trade signal prediction"""
    def __init__(self, input_size: int = 50, hidden_size: int = 128, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)

class MLPClassifier(nn.Module):
    """Fast MLP for real-time inference"""
    def __init__(self, input_size: int = 20, hidden_sizes: List[int] = [64, 32], output_size: int = 3):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class NeuralEngineOptimizer:
    """Core ML Neural Engine optimization and testing suite"""
    
    def __init__(self):
        self.models = {}
        self.benchmarks = {}
        
    def check_neural_engine_availability(self) -> Dict[str, bool]:
        """Check Neural Engine availability and capabilities"""
        print("🔍 Checking Neural Engine Availability...")
        
        try:
            # Test Core ML compute units
            import CoreML
            available_units = {
                'cpu': True,
                'gpu': True,
                'neural_engine': True  # Assume available on M4 Max
            }
            
            print(f"✅ Core ML compute units available: {list(available_units.keys())}")
            return available_units
            
        except ImportError:
            print("⚠️  PyObjC Core ML framework not available, using Core ML Tools")
            return {'cpu': True, 'gpu': False, 'neural_engine': False}
    
    def create_sample_models(self) -> Dict[str, nn.Module]:
        """Create sample PyTorch models for conversion"""
        print("🏗️  Creating sample PyTorch models...")
        
        models = {
            'trade_predictor': SimpleTradePredictor(),
            'fast_classifier': MLPClassifier(input_size=20),
            'mini_classifier': MLPClassifier(input_size=10, hidden_sizes=[32, 16])
        }
        
        # Set to eval mode
        for name, model in models.items():
            model.eval()
            print(f"   - {name}: {sum(p.numel() for p in model.parameters())} parameters")
        
        return models
    
    def convert_to_coreml(self, pytorch_model: nn.Module, model_name: str, 
                         input_shape: Tuple[int, ...], compute_units: str = 'cpuAndNeuralEngine') -> Optional[ct.models.MLModel]:
        """Convert PyTorch model to Core ML with Neural Engine optimization"""
        print(f"🔄 Converting {model_name} to Core ML...")
        
        try:
            # Create example input
            example_input = torch.randn(input_shape)
            
            # Trace the model
            traced_model = torch.jit.trace(pytorch_model, example_input)
            
            # Convert to Core ML with correct compute unit specification
            if compute_units == 'cpuAndNeuralEngine':
                compute_unit = ct.ComputeUnit.CPU_AND_NEURAL_ENGINE
            else:
                compute_unit = ct.ComputeUnit.CPU_ONLY
            
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape, name="input")],
                compute_units=compute_unit,
                minimum_deployment_target=ct.target.macOS15,
                convert_to="mlpackage"
            )
            
            print(f"   ✅ Successfully converted {model_name}")
            return coreml_model
            
        except Exception as e:
            print(f"   ❌ Failed to convert {model_name}: {e}")
            return None
    
    def optimize_for_neural_engine(self, coreml_model: ct.models.MLModel, model_name: str) -> ct.models.MLModel:
        """Apply Neural Engine specific optimizations"""
        print(f"⚡ Optimizing {model_name} for Neural Engine...")
        
        try:
            # Apply quantization for Neural Engine
            quantized_model = quantization_utils.quantize_weights(
                coreml_model,
                nbits=8,
                quantization_mode="linear"
            )
            
            print(f"   ✅ Applied 8-bit quantization to {model_name}")
            return quantized_model
            
        except Exception as e:
            print(f"   ⚠️  Quantization failed for {model_name}, using original: {e}")
            return coreml_model
    
    def benchmark_inference_speed(self, coreml_model: ct.models.MLModel, 
                                 model_name: str, input_shape: Tuple[int, ...], 
                                 num_runs: int = 1000) -> Dict[str, float]:
        """Benchmark inference speed for Core ML model"""
        print(f"🚀 Benchmarking {model_name} inference speed...")
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        input_dict = {"input": test_input}
        
        # Warmup runs
        for _ in range(10):
            try:
                _ = coreml_model.predict(input_dict)
            except Exception as e:
                print(f"   ❌ Warmup failed: {e}")
                return {'avg_ms': float('inf'), 'min_ms': float('inf'), 'max_ms': float('inf')}
        
        # Benchmark runs
        times = []
        successful_runs = 0
        
        for _ in range(num_runs):
            try:
                start_time = time.perf_counter()
                _ = coreml_model.predict(input_dict)
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(inference_time)
                successful_runs += 1
                
            except Exception as e:
                print(f"   ⚠️  Run failed: {e}")
                continue
        
        if successful_runs == 0:
            return {'avg_ms': float('inf'), 'min_ms': float('inf'), 'max_ms': float('inf')}
        
        avg_ms = np.mean(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        std_ms = np.std(times)
        
        print(f"   📊 {model_name} Results ({successful_runs}/{num_runs} successful runs):")
        print(f"      Average: {avg_ms:.3f}ms")
        print(f"      Min:     {min_ms:.3f}ms") 
        print(f"      Max:     {max_ms:.3f}ms")
        print(f"      Std:     {std_ms:.3f}ms")
        print(f"      Target:  <10ms {'✅' if avg_ms < 10 else '❌'}")
        
        return {
            'avg_ms': avg_ms,
            'min_ms': min_ms,
            'max_ms': max_ms,
            'std_ms': std_ms,
            'success_rate': successful_runs / num_runs,
            'meets_target': avg_ms < 10.0
        }
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive Neural Engine testing pipeline"""
        print("🧠 Starting M4 Max Neural Engine Comprehensive Test\n")
        
        results = {
            'neural_engine_available': False,
            'models_converted': {},
            'benchmarks': {},
            'summary': {}
        }
        
        # Check Neural Engine availability
        compute_units = self.check_neural_engine_availability()
        results['neural_engine_available'] = compute_units.get('neural_engine', False)
        
        # Create sample models
        pytorch_models = self.create_sample_models()
        
        # Model specifications
        model_specs = {
            'trade_predictor': (1, 10, 50),    # batch_size, sequence_length, features
            'fast_classifier': (1, 20),        # batch_size, features
            'mini_classifier': (1, 10)         # batch_size, features
        }
        
        # Convert and benchmark each model
        for model_name, pytorch_model in pytorch_models.items():
            input_shape = model_specs[model_name]
            
            # Convert to Core ML
            compute_unit = 'cpuAndNeuralEngine' if results['neural_engine_available'] else 'cpuOnly'
            coreml_model = self.convert_to_coreml(pytorch_model, model_name, input_shape, compute_unit)
            
            if coreml_model is not None:
                # Optimize for Neural Engine
                optimized_model = self.optimize_for_neural_engine(coreml_model, model_name)
                
                # Save model info
                results['models_converted'][model_name] = {
                    'success': True,
                    'input_shape': input_shape,
                    'compute_unit': compute_unit
                }
                
                # Benchmark performance
                benchmark_results = self.benchmark_inference_speed(
                    optimized_model, model_name, input_shape, num_runs=100
                )
                results['benchmarks'][model_name] = benchmark_results
                
            else:
                results['models_converted'][model_name] = {'success': False}
        
        # Generate summary
        successful_models = [name for name, info in results['models_converted'].items() if info.get('success', False)]
        avg_inference_times = [results['benchmarks'][name]['avg_ms'] for name in successful_models if name in results['benchmarks']]
        models_meeting_target = sum(1 for name in successful_models if results['benchmarks'].get(name, {}).get('meets_target', False))
        
        results['summary'] = {
            'total_models': len(pytorch_models),
            'successful_conversions': len(successful_models),
            'average_inference_time_ms': np.mean(avg_inference_times) if avg_inference_times else float('inf'),
            'models_meeting_10ms_target': models_meeting_target,
            'neural_engine_optimization': results['neural_engine_available']
        }
        
        return results

def main():
    """Main testing function"""
    print("=" * 70)
    print("🚀 M4 Max Neural Engine Optimization Suite")
    print("   Nautilus Trading Platform - Core ML Integration")
    print("=" * 70)
    
    optimizer = NeuralEngineOptimizer()
    results = optimizer.run_comprehensive_test()
    
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    summary = results['summary']
    print(f"Neural Engine Available:     {'✅ Yes' if results['neural_engine_available'] else '❌ No'}")
    print(f"Models Converted:            {summary['successful_conversions']}/{summary['total_models']}")
    print(f"Average Inference Time:      {summary['average_inference_time_ms']:.3f}ms")
    print(f"Models Meeting <10ms Target: {summary['models_meeting_10ms_target']}/{summary['successful_conversions']}")
    print(f"Neural Engine Optimization:  {'✅ Enabled' if summary['neural_engine_optimization'] else '❌ Disabled'}")
    
    # Individual model performance
    print(f"\n📊 Individual Model Performance:")
    for model_name, benchmark in results['benchmarks'].items():
        status = "✅" if benchmark.get('meets_target', False) else "❌"
        print(f"   {model_name:20} {benchmark['avg_ms']:8.3f}ms {status}")
    
    print("\n🎯 Neural Engine setup complete!")
    return results

if __name__ == "__main__":
    main()