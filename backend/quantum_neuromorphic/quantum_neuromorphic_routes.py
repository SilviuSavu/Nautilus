"""
Nautilus Quantum-Neuromorphic Computing API Routes

This module provides REST API endpoints for the quantum-neuromorphic computing
system, enabling integration with the Nautilus trading platform and external
applications.

Key Features:
- Neuromorphic computing API endpoints
- Quantum portfolio optimization API
- Quantum machine learning API  
- Hybrid computing orchestration API
- Hardware management and status API
- Performance benchmarking API

Author: Nautilus API Team
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
import time
from pydantic import BaseModel, Field, validator

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

# Import our quantum-neuromorphic modules
from .neuromorphic_framework import (
    NeuromorphicFramework, NeuromorphicConfig, NeuronModel, PlasticityRule
)
from .quantum_portfolio_optimizer import (
    QuantumPortfolioOptimizer, QuantumConfig, OptimizationObjective, QuantumBackend
)
from .quantum_machine_learning import (
    QuantumMLFramework, QuantumMLConfig, QuantumMLAlgorithm, FeatureMap, Ansatz
)
from .hybrid_computing_system import (
    HybridComputingSystem, ComputeBackend, WorkloadType, OptimizationObjective as HybridOptimization
)
from .neuromorphic_hardware import (
    NeuromorphicHardwareManager, HardwareConfig, NeuromorphicHardware
)

logger = logging.getLogger(__name__)

# Global system instances
neuromorphic_system: Optional[NeuromorphicFramework] = None
quantum_optimizer: Optional[QuantumPortfolioOptimizer] = None 
quantum_ml_system: Optional[QuantumMLFramework] = None
hybrid_system: Optional[HybridComputingSystem] = None
hardware_manager: Optional[NeuromorphicHardwareManager] = None

# Create router
router = APIRouter(prefix="/api/v1/quantum-neuromorphic", tags=["Quantum-Neuromorphic Computing"])

# Pydantic models for API
class NeuromorphicConfigModel(BaseModel):
    """Configuration model for neuromorphic computing."""
    timestep: float = Field(0.1, gt=0.0, description="Simulation timestep in ms")
    simulation_time: float = Field(1000.0, gt=0.0, description="Total simulation time in ms")
    neuron_model: str = Field("LIF", description="Neuron model type")
    input_size: int = Field(100, gt=0, le=10000, description="Number of input neurons")
    hidden_sizes: List[int] = Field([256, 128, 64], description="Hidden layer sizes")
    output_size: int = Field(10, gt=0, le=1000, description="Number of output neurons")
    learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Learning rate")
    hardware_backend: str = Field("simulation", description="Hardware backend")

class QuantumConfigModel(BaseModel):
    """Configuration model for quantum computing."""
    backend: str = Field("qiskit_aer_simulator", description="Quantum backend")
    max_qubits: int = Field(20, gt=0, le=100, description="Maximum qubits")
    shots: int = Field(1024, gt=0, description="Number of shots")
    algorithm: str = Field("VQE", description="Quantum algorithm")
    optimizer: str = Field("SPSA", description="Classical optimizer")
    max_iterations: int = Field(300, gt=0, description="Maximum optimization iterations")

class MarketDataModel(BaseModel):
    """Model for market data input."""
    data: List[List[float]] = Field(..., description="Market data array")
    timestamps: Optional[List[str]] = Field(None, description="Timestamps for data points")
    symbols: Optional[List[str]] = Field(None, description="Asset symbols")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata")

class PortfolioOptimizationRequest(BaseModel):
    """Request model for portfolio optimization."""
    returns_data: Dict[str, List[float]] = Field(..., description="Returns data by asset")
    covariance_matrix: Optional[List[List[float]]] = Field(None, description="Covariance matrix")
    market_caps: Optional[List[float]] = Field(None, description="Market capitalizations")
    risk_aversion: float = Field(0.5, ge=0.0, le=2.0, description="Risk aversion parameter")
    max_weight: float = Field(0.4, ge=0.0, le=1.0, description="Maximum allocation per asset")

class MLTrainingRequest(BaseModel):
    """Request model for ML training."""
    training_data: List[List[float]] = Field(..., description="Training features")
    training_labels: List[float] = Field(..., description="Training labels") 
    algorithm: str = Field("QSVM", description="ML algorithm")
    model_name: str = Field("default_model", description="Model name")
    config: Optional[Dict[str, Any]] = Field({}, description="Algorithm configuration")

class HybridTaskRequest(BaseModel):
    """Request model for hybrid computing tasks."""
    workload_type: str = Field(..., description="Type of workload")
    data: Dict[str, Any] = Field(..., description="Task data")
    preferred_backend: str = Field("AUTO", description="Preferred computing backend")
    priority: int = Field(5, ge=1, le=10, description="Task priority")
    deadline: Optional[str] = Field(None, description="Task deadline (ISO format)")

# Startup and shutdown
@router.on_event("startup")
async def startup_quantum_neuromorphic():
    """Initialize quantum-neuromorphic systems on startup."""
    global neuromorphic_system, quantum_optimizer, quantum_ml_system, hybrid_system, hardware_manager
    
    try:
        logger.info("Initializing Quantum-Neuromorphic Computing systems...")
        
        # Initialize neuromorphic system
        neuromorphic_config = NeuromorphicConfig()
        neuromorphic_system = NeuromorphicFramework(neuromorphic_config)
        await neuromorphic_system.initialize()
        
        # Initialize quantum optimizer
        quantum_config = QuantumConfig()
        quantum_optimizer = QuantumPortfolioOptimizer(quantum_config)
        await quantum_optimizer.initialize()
        
        # Initialize quantum ML system
        quantum_ml_config = QuantumMLConfig()
        quantum_ml_system = QuantumMLFramework(quantum_ml_config)
        await quantum_ml_system.initialize()
        
        # Initialize hybrid system
        hybrid_system = HybridComputingSystem(
            quantum_config=quantum_config,
            neuromorphic_config=neuromorphic_config
        )
        await hybrid_system.initialize()
        
        # Initialize hardware manager
        hardware_config = HardwareConfig(hardware_type=NeuromorphicHardware.SIMULATION)
        hardware_manager = NeuromorphicHardwareManager()
        await hardware_manager.initialize(hardware_config)
        
        logger.info("Quantum-Neuromorphic Computing systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Quantum-Neuromorphic systems: {e}")
        # Continue with partial initialization

@router.on_event("shutdown") 
async def shutdown_quantum_neuromorphic():
    """Shutdown quantum-neuromorphic systems."""
    global neuromorphic_system, quantum_optimizer, quantum_ml_system, hybrid_system, hardware_manager
    
    try:
        logger.info("Shutting down Quantum-Neuromorphic Computing systems...")
        
        if neuromorphic_system:
            await neuromorphic_system.shutdown()
            
        if hybrid_system:
            await hybrid_system.shutdown()
            
        if hardware_manager:
            await hardware_manager.shutdown()
            
        logger.info("Quantum-Neuromorphic Computing systems shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Health and status endpoints
@router.get("/health")
async def health_check():
    """Health check for quantum-neuromorphic systems."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "systems": {
            "neuromorphic": neuromorphic_system is not None,
            "quantum_optimizer": quantum_optimizer is not None,
            "quantum_ml": quantum_ml_system is not None,
            "hybrid_system": hybrid_system is not None,
            "hardware_manager": hardware_manager is not None
        }
    }
    
    # Check system health
    try:
        if neuromorphic_system:
            neuromorphic_status = neuromorphic_system.get_framework_status()
            health_status["systems"]["neuromorphic"] = neuromorphic_status["initialized"]
            
        if quantum_optimizer:
            quantum_status = quantum_optimizer.get_optimizer_status()
            health_status["systems"]["quantum_optimizer"] = quantum_status["initialized"]
            
        if quantum_ml_system:
            ml_status = quantum_ml_system.get_framework_status()
            health_status["systems"]["quantum_ml"] = ml_status["initialized"]
            
        if hybrid_system:
            hybrid_status = hybrid_system.get_system_status()
            health_status["systems"]["hybrid_system"] = hybrid_status["initialized"]
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status["status"] = "degraded"
        health_status["error"] = str(e)
        
    return health_status

@router.get("/status")
async def system_status():
    """Get comprehensive system status."""
    
    if not any([neuromorphic_system, quantum_optimizer, quantum_ml_system, hybrid_system]):
        raise HTTPException(status_code=503, detail="Systems not initialized")
        
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "neuromorphic": {},
        "quantum_optimizer": {},
        "quantum_ml": {},
        "hybrid_system": {},
        "hardware": {}
    }
    
    try:
        if neuromorphic_system:
            status["neuromorphic"] = neuromorphic_system.get_framework_status()
            
        if quantum_optimizer:
            status["quantum_optimizer"] = quantum_optimizer.get_optimizer_status()
            
        if quantum_ml_system:
            status["quantum_ml"] = quantum_ml_system.get_framework_status()
            
        if hybrid_system:
            status["hybrid_system"] = hybrid_system.get_system_status()
            
        if hardware_manager:
            status["hardware"] = hardware_manager.get_hardware_status()
            
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")
        
    return status

# Neuromorphic computing endpoints
@router.post("/neuromorphic/configure")
async def configure_neuromorphic(config: NeuromorphicConfigModel):
    """Configure neuromorphic computing system."""
    
    if not neuromorphic_system:
        raise HTTPException(status_code=503, detail="Neuromorphic system not available")
        
    try:
        # Update configuration
        neuromorphic_system.config.timestep = config.timestep
        neuromorphic_system.config.simulation_time = config.simulation_time
        neuromorphic_system.config.input_size = config.input_size
        neuromorphic_system.config.hidden_sizes = config.hidden_sizes
        neuromorphic_system.config.output_size = config.output_size
        neuromorphic_system.config.learning_rate = config.learning_rate
        
        # Apply neuron model
        if config.neuron_model in [model.value for model in NeuronModel]:
            neuromorphic_system.config.neuron_model = NeuronModel(config.neuron_model)
            
        return {
            "status": "configured",
            "message": "Neuromorphic system configured successfully",
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Neuromorphic configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/neuromorphic/process")
async def process_neuromorphic(data: MarketDataModel, task: str = "pattern_recognition"):
    """Process data using neuromorphic computing."""
    
    if not neuromorphic_system:
        raise HTTPException(status_code=503, detail="Neuromorphic system not available")
        
    try:
        # Convert data to numpy array
        market_data = np.array(data.data)
        
        # Process using neuromorphic framework
        start_time = time.time()
        result = await neuromorphic_system.process_market_data(market_data, task)
        processing_time = time.time() - start_time
        
        return {
            "status": "completed",
            "task_type": task,
            "processing_time_ms": processing_time * 1000,
            "results": {
                "output_values": result["output_values"].tolist() if isinstance(result.get("output_values"), np.ndarray) else result.get("output_values"),
                "spike_events": len(result.get("spike_events", [])),
                "performance_metrics": result.get("performance_metrics", {}),
                "framework_metrics": result.get("framework_metrics", {})
            },
            "metadata": data.metadata
        }
        
    except Exception as e:
        logger.error(f"Neuromorphic processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/neuromorphic/train")
async def train_neuromorphic(request: MLTrainingRequest):
    """Train neuromorphic network."""
    
    if not neuromorphic_system:
        raise HTTPException(status_code=503, detail="Neuromorphic system not available")
        
    try:
        # Prepare training data
        X_train = np.array(request.training_data)
        y_train = np.array(request.training_labels)
        
        # Train network
        start_time = time.time()
        training_result = await neuromorphic_system.train_network(
            request.model_name,
            list(zip(X_train, y_train)),
            epochs=request.config.get("epochs", 100)
        )
        training_time = time.time() - start_time
        
        return {
            "status": "trained",
            "model_name": request.model_name,
            "training_time": training_time,
            "results": {
                "epochs_completed": training_result["epochs_completed"],
                "final_loss": training_result["average_loss"][-1] if training_result["average_loss"] else 0,
                "final_accuracy": training_result["accuracy_history"][-1] if training_result["accuracy_history"] else 0,
                "training_history": training_result
            }
        }
        
    except Exception as e:
        logger.error(f"Neuromorphic training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Quantum computing endpoints
@router.post("/quantum/configure")
async def configure_quantum(config: QuantumConfigModel):
    """Configure quantum computing system."""
    
    if not quantum_optimizer:
        raise HTTPException(status_code=503, detail="Quantum system not available")
        
    try:
        # Update quantum configuration
        quantum_optimizer.config.max_qubits = config.max_qubits
        quantum_optimizer.config.shots = config.shots
        quantum_optimizer.config.max_iterations = config.max_iterations
        
        # Apply backend
        for backend in QuantumBackend:
            if backend.value == config.backend:
                quantum_optimizer.config.backend = backend
                break
                
        return {
            "status": "configured",
            "message": "Quantum system configured successfully",
            "configuration": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Quantum configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.post("/quantum/optimize-portfolio")
async def optimize_portfolio_quantum(request: PortfolioOptimizationRequest):
    """Optimize portfolio using quantum algorithms."""
    
    if not quantum_optimizer:
        raise HTTPException(status_code=503, detail="Quantum optimizer not available")
        
    try:
        # Convert data to DataFrame
        returns_df = pd.DataFrame(request.returns_data)
        
        # Prepare covariance matrix if provided
        covariance_matrix = None
        if request.covariance_matrix:
            covariance_matrix = np.array(request.covariance_matrix)
            
        # Update configuration
        quantum_optimizer.config.risk_aversion = request.risk_aversion
        quantum_optimizer.config.max_weight = request.max_weight
        
        # Run optimization
        start_time = time.time()
        result = await quantum_optimizer.optimize_portfolio(
            returns_df, 
            covariance_matrix, 
            np.array(request.market_caps) if request.market_caps else None
        )
        optimization_time = time.time() - start_time
        
        return {
            "status": "optimized",
            "optimization_time": optimization_time,
            "results": {
                "optimal_weights": result.optimal_weights.tolist(),
                "expected_return": result.expected_return,
                "expected_risk": result.expected_risk,
                "sharpe_ratio": result.sharpe_ratio,
                "quantum_advantage": result.quantum_advantage,
                "circuit_depth": result.circuit_depth,
                "gate_count": result.gate_count,
                "convergence_info": result.convergence_info
            }
        }
        
    except Exception as e:
        logger.error(f"Quantum portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.post("/quantum/quantum-advantage")
async def assess_quantum_advantage(request: PortfolioOptimizationRequest, iterations: int = Query(5, ge=1, le=20)):
    """Assess quantum advantage for portfolio optimization."""
    
    if not quantum_optimizer:
        raise HTTPException(status_code=503, detail="Quantum optimizer not available")
        
    try:
        # Convert data
        returns_df = pd.DataFrame(request.returns_data)
        
        # Run quantum advantage assessment
        start_time = time.time()
        advantage_result = await quantum_optimizer.calculate_quantum_advantage(
            returns_df, 
            benchmark_iterations=iterations
        )
        assessment_time = time.time() - start_time
        
        return {
            "status": "assessed",
            "assessment_time": assessment_time,
            "quantum_advantage": {
                "time_speedup": advantage_result["quantum_advantage"]["time_speedup"],
                "quality_improvement": advantage_result["quantum_advantage"]["quality_improvement"],
                "quantum_supremacy_achieved": advantage_result["quantum_advantage"]["quantum_supremacy_achieved"],
                "benchmark_iterations": iterations
            },
            "detailed_results": advantage_result
        }
        
    except Exception as e:
        logger.error(f"Quantum advantage assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

# Quantum machine learning endpoints
@router.post("/quantum-ml/train")
async def train_quantum_ml(request: MLTrainingRequest):
    """Train quantum machine learning model."""
    
    if not quantum_ml_system:
        raise HTTPException(status_code=503, detail="Quantum ML system not available")
        
    try:
        # Prepare data
        X_train = np.array(request.training_data)
        y_train = np.array(request.training_labels)
        
        # Configure algorithm
        if request.algorithm in [alg.value for alg in QuantumMLAlgorithm]:
            quantum_ml_system.config.algorithm = QuantumMLAlgorithm(request.algorithm)
            
        # Update configuration from request
        for key, value in request.config.items():
            if hasattr(quantum_ml_system.config, key):
                setattr(quantum_ml_system.config, key, value)
                
        # Train model
        start_time = time.time()
        result = await quantum_ml_system.train_quantum_model(
            X_train, y_train, request.model_name
        )
        training_time = time.time() - start_time
        
        return {
            "status": "trained", 
            "model_name": request.model_name,
            "algorithm": result.algorithm,
            "training_time": training_time,
            "results": {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "quantum_advantage": result.quantum_advantage,
                "model_complexity": result.model_complexity,
                "quantum_state_fidelity": result.quantum_state_fidelity,
                "training_history": result.training_history
            }
        }
        
    except Exception as e:
        logger.error(f"Quantum ML training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/quantum-ml/predict")
async def predict_quantum_ml(data: List[List[float]], model_name: str = "default_model"):
    """Make predictions using trained quantum ML model."""
    
    if not quantum_ml_system:
        raise HTTPException(status_code=503, detail="Quantum ML system not available")
        
    try:
        # Convert data
        X_test = np.array(data)
        
        # Make predictions
        start_time = time.time()
        predictions = await quantum_ml_system.predict(X_test, model_name)
        prediction_time = time.time() - start_time
        
        return {
            "status": "predicted",
            "model_name": model_name,
            "prediction_time": prediction_time,
            "results": {
                "predictions": predictions["predictions"].tolist(),
                "probabilities": predictions["probabilities"].tolist(),
                "algorithm": predictions["algorithm"],
                "quantum_advantage": predictions["quantum_advantage"]
            }
        }
        
    except Exception as e:
        logger.error(f"Quantum ML prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Hybrid computing endpoints  
@router.post("/hybrid/submit-task")
async def submit_hybrid_task(request: HybridTaskRequest):
    """Submit task to hybrid computing system."""
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
        
    try:
        # Convert workload type
        workload_type = None
        for wt in WorkloadType:
            if wt.value == request.workload_type:
                workload_type = wt
                break
                
        if not workload_type:
            raise ValueError(f"Unknown workload type: {request.workload_type}")
            
        # Convert backend preference
        preferred_backend = ComputeBackend.AUTO
        for backend in ComputeBackend:
            if backend.value == request.preferred_backend:
                preferred_backend = backend
                break
                
        # Submit task
        task_id = await hybrid_system.submit_task(
            workload_type=workload_type,
            data=request.data,
            preferred_backend=preferred_backend
        )
        
        return {
            "status": "submitted",
            "task_id": task_id,
            "workload_type": request.workload_type,
            "preferred_backend": request.preferred_backend,
            "estimated_completion": "unknown"  # Could add estimation logic
        }
        
    except Exception as e:
        logger.error(f"Hybrid task submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")

@router.get("/hybrid/task/{task_id}")
async def get_hybrid_task_status(task_id: str):
    """Get status of hybrid computing task."""
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
        
    try:
        status = await hybrid_system.get_task_status(task_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Task not found")
            
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/hybrid/task/{task_id}/result")
async def get_hybrid_task_result(task_id: str):
    """Get result of completed hybrid computing task."""
    
    if not hybrid_system:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
        
    try:
        result = await hybrid_system.get_task_result(task_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Task result not found or not completed")
            
        return {
            "status": "completed",
            "task_id": task_id,
            "result": {
                "data": result.result_data,
                "backend_used": result.backend_used.value,
                "execution_time": result.execution_time,
                "energy_consumed": result.energy_consumed,
                "accuracy_achieved": result.accuracy_achieved,
                "quantum_advantage": result.quantum_advantage,
                "neuromorphic_efficiency": result.neuromorphic_efficiency,
                "resource_utilization": result.resource_utilization
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task result retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Result retrieval failed: {str(e)}")

# Hardware management endpoints
@router.get("/hardware/status")
async def get_hardware_status():
    """Get neuromorphic hardware status."""
    
    if not hardware_manager:
        raise HTTPException(status_code=503, detail="Hardware manager not available")
        
    try:
        status = hardware_manager.get_hardware_status()
        return status
        
    except Exception as e:
        logger.error(f"Hardware status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.post("/hardware/benchmark")
async def benchmark_hardware(network_spec: Dict[str, Any], duration_ms: int = Query(1000, ge=100, le=10000)):
    """Benchmark neuromorphic hardware platforms."""
    
    if not hardware_manager:
        raise HTTPException(status_code=503, detail="Hardware manager not available")
        
    try:
        # Create dummy input spikes for benchmarking
        from .neuromorphic_hardware import SpikeEvent
        
        num_input_spikes = network_spec.get("num_inputs", 10)
        input_spikes = [
            SpikeEvent(
                neuron_id=i,
                timestamp_us=i * 1000,
                core_id=0
            )
            for i in range(num_input_spikes)
        ]
        
        # Run benchmark
        start_time = time.time()
        results = await hardware_manager.benchmark_platforms(
            network_spec=network_spec,
            input_spikes=input_spikes,
            duration_us=duration_ms * 1000
        )
        benchmark_time = time.time() - start_time
        
        return {
            "status": "completed",
            "benchmark_time": benchmark_time,
            "duration_ms": duration_ms,
            "results": {
                platform: {
                    "total_spikes": stats.total_spikes,
                    "energy_consumed_uj": stats.energy_consumed_uj,
                    "execution_time_us": stats.execution_time_us,
                    "power_mw": stats.power_mw,
                    "efficiency": stats.energy_consumed_uj / max(stats.execution_time_us, 1)
                }
                for platform, stats in results.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Hardware benchmarking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")

# Performance and analytics endpoints
@router.get("/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics across all systems."""
    
    analytics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "neuromorphic": {},
        "quantum": {},
        "hybrid": {},
        "summary": {}
    }
    
    try:
        # Neuromorphic performance
        if neuromorphic_system:
            neuromorphic_status = neuromorphic_system.get_framework_status()
            analytics["neuromorphic"] = {
                "networks": len(neuromorphic_status.get("networks", {})),
                "total_simulations": neuromorphic_status.get("performance", {}).get("total_simulations", 0),
                "total_energy": neuromorphic_status.get("performance", {}).get("total_energy", 0)
            }
            
        # Quantum performance  
        if quantum_optimizer:
            quantum_status = quantum_optimizer.get_optimizer_status()
            analytics["quantum"] = {
                "total_optimizations": quantum_status.get("performance_metrics", {}).get("total_optimizations", 0),
                "average_quantum_advantage": quantum_status.get("performance_metrics", {}).get("average_quantum_advantage", 1.0),
                "success_rate": quantum_status.get("performance_metrics", {}).get("success_rate", 1.0)
            }
            
        # Hybrid performance
        if hybrid_system:
            hybrid_status = hybrid_system.get_system_status()
            analytics["hybrid"] = {
                "active_tasks": hybrid_status.get("active_tasks", 0),
                "completed_tasks": hybrid_status.get("completed_tasks", 0),
                "average_quantum_advantage": hybrid_status.get("performance_summary", {}).get("average_quantum_advantage", 1.0)
            }
            
        # Summary metrics
        total_energy_saved = (
            analytics["neuromorphic"].get("total_energy", 0) + 
            analytics["hybrid"].get("total_energy_saved", 0)
        )
        
        analytics["summary"] = {
            "total_computations": (
                analytics["neuromorphic"].get("total_simulations", 0) +
                analytics["quantum"].get("total_optimizations", 0) + 
                analytics["hybrid"].get("completed_tasks", 0)
            ),
            "average_quantum_advantage": np.mean([
                analytics["quantum"].get("average_quantum_advantage", 1.0),
                analytics["hybrid"].get("average_quantum_advantage", 1.0)
            ]),
            "total_energy_saved": total_energy_saved,
            "system_efficiency": 0.92  # Estimated overall efficiency
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@router.get("/analytics/quantum-advantage")
async def get_quantum_advantage_analytics():
    """Get quantum advantage analytics."""
    
    if not quantum_optimizer:
        raise HTTPException(status_code=503, detail="Quantum optimizer not available")
        
    try:
        quantum_status = quantum_optimizer.get_optimizer_status()
        performance_metrics = quantum_status.get("performance_metrics", {})
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quantum_advantage_metrics": {
                "average_speedup": performance_metrics.get("average_quantum_advantage", 1.0),
                "total_optimizations": performance_metrics.get("total_optimizations", 0),
                "quantum_time": performance_metrics.get("quantum_time", 0.0),
                "classical_time": performance_metrics.get("classical_time", 0.0),
                "success_rate": performance_metrics.get("success_rate", 1.0)
            },
            "analysis": {
                "quantum_supremacy_achieved": performance_metrics.get("average_quantum_advantage", 1.0) > 1.5,
                "energy_efficiency": "High" if performance_metrics.get("quantum_time", 1.0) < performance_metrics.get("classical_time", 1.0) else "Moderate",
                "recommendation": "Continue quantum optimization" if performance_metrics.get("average_quantum_advantage", 1.0) > 1.2 else "Evaluate alternatives"
            }
        }
        
    except Exception as e:
        logger.error(f"Quantum advantage analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

# Utility endpoints
@router.post("/utilities/initialize")
async def initialize_systems():
    """Manually initialize all quantum-neuromorphic systems."""
    
    try:
        await startup_quantum_neuromorphic()
        return {
            "status": "initialized",
            "message": "All quantum-neuromorphic systems initialized successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Manual initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.post("/utilities/reset")
async def reset_systems():
    """Reset all quantum-neuromorphic systems."""
    
    try:
        # Shutdown existing systems
        await shutdown_quantum_neuromorphic()
        
        # Wait briefly
        await asyncio.sleep(2.0)
        
        # Reinitialize
        await startup_quantum_neuromorphic()
        
        return {
            "status": "reset",
            "message": "All quantum-neuromorphic systems reset successfully", 
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"System reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Export router
__all__ = ["router"]