"""
Nautilus Phase 6: Quantum-Neuromorphic Computing Integration

This package implements breakthrough quantum computing and neuromorphic computing
capabilities for ultra-efficient trading platform operations. It provides:

1. Neuromorphic Computing Framework:
   - Spike-based neural networks for real-time processing
   - Neuromorphic hardware integration (Intel Loihi, SpiNNaker)
   - Event-driven computation with ultra-low power consumption
   - Bio-inspired learning algorithms

2. Quantum Computing Portfolio Optimization:
   - VQE (Variational Quantum Eigensolver) algorithms
   - QAOA (Quantum Approximate Optimization Algorithm)
   - Quantum machine learning models
   - Hybrid quantum-classical optimization

3. Quantum-Classical Hybrid Systems:
   - Seamless integration between quantum and classical processing
   - Optimal workload distribution
   - Performance benchmarking and comparison
   - Fault-tolerant quantum error correction

4. Advanced Pattern Recognition:
   - Quantum Support Vector Machines (QSVM)
   - Quantum Neural Networks (QNN)
   - Quantum feature mapping and kernel methods
   - Real-time market pattern detection

This implementation represents the next frontier in financial technology,
combining quantum supremacy with brain-inspired computing for unprecedented
trading performance and efficiency.

Author: Nautilus Quantum-Neuromorphic Computing Team
Version: 1.0.0
License: MIT
"""

from .neuromorphic_framework import NeuromorphicFramework, SpikingNeuralNetwork
from .quantum_portfolio_optimizer import QuantumPortfolioOptimizer, VQEOptimizer, QAOAOptimizer
from .quantum_machine_learning import QuantumMLFramework, QSVM, QuantumNeuralNetwork
from .hybrid_computing_system import HybridComputingSystem, QuantumClassicalOrchestrator
from .neuromorphic_hardware import NeuromorphicHardwareManager, LoihiInterface, SpiNNakerInterface
from .quantum_neuromorphic_routes import router
from .benchmarking_system import QuantumNeuromorphicBenchmarks
from .performance_optimizer import PerformanceOptimizer

__version__ = "1.0.0"
__author__ = "Nautilus Quantum-Neuromorphic Computing Team"

__all__ = [
    # Core Components
    "NeuromorphicFramework",
    "SpikingNeuralNetwork", 
    "QuantumPortfolioOptimizer",
    "VQEOptimizer",
    "QAOAOptimizer",
    "QuantumMLFramework",
    "QSVM",
    "QuantumNeuralNetwork",
    
    # System Integration
    "HybridComputingSystem",
    "QuantumClassicalOrchestrator",
    "NeuromorphicHardwareManager",
    "LoihiInterface",
    "SpiNNakerInterface",
    
    # API and Performance
    "router",
    "QuantumNeuromorphicBenchmarks", 
    "PerformanceOptimizer",
]

# Module-level configuration
QUANTUM_BACKENDS = {
    "ibm": "qiskit",
    "google": "cirq", 
    "rigetti": "pyquil",
    "aws": "braket",
    "pennylane": "default.qubit"
}

NEUROMORPHIC_BACKENDS = {
    "loihi": "intel_loihi",
    "spinnaker": "spynnaker",
    "brainscales": "nest",
    "simulation": "nengo"
}

# Performance configuration
DEFAULT_QUANTUM_CONFIG = {
    "max_qubits": 30,
    "shot_count": 1024,
    "optimization_level": 3,
    "error_mitigation": True,
    "noise_model": "ibm_cairo"
}

DEFAULT_NEUROMORPHIC_CONFIG = {
    "timestep": 0.1,  # ms
    "simulation_time": 1000.0,  # ms
    "neuron_model": "LIF",  # Leaky Integrate-and-Fire
    "plasticity": "STDP",  # Spike-Timing Dependent Plasticity
    "hardware_acceleration": True
}

# Trading-specific configuration
TRADING_CONFIG = {
    "portfolio_update_frequency": 100,  # Hz
    "risk_calculation_interval": 10,  # ms
    "pattern_recognition_window": 1000,  # ms
    "quantum_advantage_threshold": 2.0  # 2x speedup required
}

def get_version() -> str:
    """Get the current version of the quantum-neuromorphic package."""
    return __version__

def get_quantum_backends() -> dict:
    """Get available quantum computing backends."""
    return QUANTUM_BACKENDS.copy()

def get_neuromorphic_backends() -> dict:
    """Get available neuromorphic computing backends."""
    return NEUROMORPHIC_BACKENDS.copy()

def get_default_config() -> dict:
    """Get default configuration for quantum-neuromorphic systems."""
    return {
        "quantum": DEFAULT_QUANTUM_CONFIG.copy(),
        "neuromorphic": DEFAULT_NEUROMORPHIC_CONFIG.copy(),
        "trading": TRADING_CONFIG.copy()
    }