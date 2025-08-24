"""
Core ML Neural Engine Acceleration Package for M4 Max

Comprehensive Core ML Neural Engine integration for ultra-fast ML inference in trading:
- Hardware detection and optimization for M4 Max Neural Engine (38 TOPS)
- Financial ML model architectures (LSTM, Transformer, CNN) optimized for trading
- Sub-10ms inference latency with batch processing optimization
- Model lifecycle management with A/B testing and automated retraining
- Real-time streaming inference with thermal management

Key Features:
- 16-core Neural Engine optimization for M4 Max (38 TOPS performance)
- Trading-specific models: price prediction, sentiment analysis, pattern recognition
- Advanced model management with versioning and deployment strategies
- Performance monitoring and automated optimization
- Production-ready inference pipeline with fallback mechanisms

Usage:
    from backend.acceleration import (
        initialize_neural_engine,
        initialize_inference_engine,
        create_price_prediction_model,
        predict,
        predict_batch
    )
    
    # Initialize Neural Engine and inference pipeline
    neural_status = initialize_neural_engine()
    inference_ready = await initialize_inference_engine()
    
    # Create and deploy trading models
    model, model_id = await create_price_prediction_model(config)
    result = await predict(model_path, market_data)
"""

import logging
from typing import Dict, Any, Optional, List

# Import Core ML Neural Engine components
from .neural_engine_config import (
    initialize_neural_engine,
    get_neural_engine_status,
    get_optimization_config,
    is_m4_max_detected,
    get_neural_engine_specs,
    cleanup_neural_engine,
    neural_performance_context
)

from .coreml_pipeline import (
    convert_model_to_coreml,
    get_pipeline_status,
    ModelType,
    OptimizationLevel,
    model_converter,
    version_manager
)

from .trading_models import (
    create_price_prediction_model,
    create_pattern_recognition_model,
    create_sentiment_model,
    get_default_model_config,
    get_trading_models_status,
    ModelArchitecture,
    DataFrequency,
    model_builder,
    model_trainer,
    data_preprocessor
)

from .neural_inference import (
    initialize_inference_engine,
    shutdown_inference_engine,
    predict,
    predict_batch,
    hft_predict,
    risk_predict,
    get_inference_status,
    inference_engine,
    PriorityLevel,
    InferenceMode
)

from .model_manager import (
    initialize_model_management,
    register_model,
    deploy_model,
    get_model_management_status,
    model_manager,
    ModelStatus,
    DeploymentStrategy
)

# Legacy Metal components (maintaining backward compatibility)
try:
    from .metal_config import (
        metal_device_manager,
        is_metal_available,
        get_metal_capabilities,
        optimize_for_financial_computing,
        metal_performance_context as metal_perf_context
    )
    
    from .metal_compute import (
        price_option_metal,
        calculate_rsi_metal,
        calculate_macd_metal,
        calculate_bollinger_bands_metal
    )
    
    from .pytorch_metal import (
        create_metal_model_wrapper,
        create_financial_lstm,
        create_financial_transformer,
        metal_inference_mode
    )
    
    from .gpu_memory_pool import (
        allocate_gpu_tensor,
        get_memory_pool_stats,
        clear_gpu_memory_cache,
        optimize_gpu_memory_layout
    )
    
    METAL_LEGACY_AVAILABLE = True
except ImportError:
    METAL_LEGACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Package version
__version__ = "1.0.0"

# Export public API
__all__ = [
    # Core Neural Engine configuration
    'initialize_neural_engine',
    'get_neural_engine_status',
    'get_optimization_config',
    'is_m4_max_detected',
    'get_neural_engine_specs',
    'cleanup_neural_engine',
    'neural_performance_context',
    
    # Core ML Pipeline
    'convert_model_to_coreml',
    'get_pipeline_status',
    'ModelType',
    'OptimizationLevel',
    
    # Trading Models
    'create_price_prediction_model',
    'create_pattern_recognition_model', 
    'create_sentiment_model',
    'get_default_model_config',
    'get_trading_models_status',
    'ModelArchitecture',
    'DataFrequency',
    
    # Neural Inference Engine
    'initialize_inference_engine',
    'shutdown_inference_engine',
    'predict',
    'predict_batch',
    'hft_predict',
    'risk_predict',
    'get_inference_status',
    'PriorityLevel',
    'InferenceMode',
    
    # Model Management
    'initialize_model_management',
    'register_model',
    'deploy_model',
    'get_model_management_status',
    'ModelStatus',
    'DeploymentStrategy',
    
    # Legacy Metal GPU support (backward compatibility)
    'initialize_metal_acceleration',
    'is_metal_available',
    'get_metal_capabilities',
    'price_option_metal',
    'calculate_rsi_metal',
    'calculate_macd_metal',
    'calculate_bollinger_bands_metal',
    'create_metal_model_wrapper',
    'create_financial_lstm',
    'create_financial_transformer',
    'allocate_gpu_tensor',
    'get_memory_pool_stats',
    'clear_gpu_memory_cache',
    'optimize_gpu_memory_layout',
    'optimize_for_financial_computing'
]

async def initialize_coreml_acceleration(enable_logging: bool = True) -> Dict[str, Any]:
    """
    Initialize Core ML Neural Engine acceleration for the Nautilus trading platform
    
    Args:
        enable_logging: Enable detailed logging for initialization process
        
    Returns:
        Dictionary containing initialization status and configuration details
    """
    if enable_logging:
        logging.getLogger(__name__).setLevel(logging.INFO)
        
    logger.info("Initializing Core ML Neural Engine acceleration for Nautilus trading platform")
    
    initialization_results = {
        "package_version": __version__,
        "neural_engine_available": False,
        "coreml_available": False,
        "inference_engine_ready": False,
        "model_management_ready": False,
        "m4_max_detected": False,
        "initialization_time_ms": 0,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    import time
    start_time = time.time()
    
    try:
        # Initialize Neural Engine configuration
        logger.info("Initializing Neural Engine configuration...")
        neural_status = initialize_neural_engine(enable_logging)
        m4_max_detected = is_m4_max_detected()
        
        initialization_results.update({
            "neural_engine_available": neural_status.get('success', False),
            "m4_max_detected": m4_max_detected,
            "neural_engine_config": neural_status
        })
        
        if neural_status.get('success', False):
            logger.info("Neural Engine configuration successful")
            if 'specs' in neural_status and neural_status['specs']:
                specs = neural_status['specs']
                initialization_results["neural_engine_cores"] = specs.get('cores', 0)
                initialization_results["tops_performance"] = specs.get('tops_performance', 0)
                initialization_results["memory_bandwidth_gbps"] = specs.get('memory_bandwidth_gb_s', 0)
        else:
            initialization_results["warnings"].append("Neural Engine not available - using CPU fallback")
            initialization_results["errors"].append(neural_status.get('message', 'Neural Engine initialization failed'))
            
        # Initialize Core ML Pipeline
        logger.info("Initializing Core ML Pipeline...")
        try:
            pipeline_status = get_pipeline_status()
            initialization_results["coreml_available"] = pipeline_status.get('coreml_available', False)
            initialization_results["frameworks_available"] = pipeline_status.get('frameworks_available', {})
            
            if initialization_results["coreml_available"]:
                logger.info("Core ML Pipeline ready")
            else:
                initialization_results["warnings"].append("Core ML not available - install coremltools")
                
        except Exception as e:
            logger.error(f"Core ML Pipeline initialization failed: {e}")
            initialization_results["errors"].append(f"Core ML Pipeline error: {str(e)}")
            
        # Initialize Inference Engine
        logger.info("Initializing Neural Inference Engine...")
        try:
            inference_ready = await initialize_inference_engine()
            initialization_results["inference_engine_ready"] = inference_ready
            
            if inference_ready:
                logger.info("Neural Inference Engine ready")
                inference_status = get_inference_status()
                initialization_results["inference_workers"] = inference_status.get('max_workers', 0)
                initialization_results["inference_config"] = {
                    "streaming_enabled": inference_status.get('streaming_enabled', False),
                    "cache_size": inference_status.get('cached_models', 0)
                }
            else:
                initialization_results["warnings"].append("Neural Inference Engine failed to start")
                
        except Exception as e:
            logger.error(f"Neural Inference Engine initialization failed: {e}")
            initialization_results["errors"].append(f"Inference Engine error: {str(e)}")
            
        # Initialize Model Management
        logger.info("Initializing Model Management System...")
        try:
            management_ready = await initialize_model_management()
            initialization_results["model_management_ready"] = management_ready
            
            if management_ready:
                logger.info("Model Management System ready")
                management_status = get_model_management_status()
                initialization_results["model_registry"] = {
                    "total_models": management_status.get('registry_stats', {}).get('total_models', 0),
                    "active_models": management_status.get('registry_stats', {}).get('active_models', 0)
                }
            else:
                initialization_results["warnings"].append("Model Management System failed to start")
                
        except Exception as e:
            logger.error(f"Model Management System initialization failed: {e}")
            initialization_results["errors"].append(f"Model Management error: {str(e)}")
            
        # Generate recommendations
        recommendations = []
        
        if not initialization_results["neural_engine_available"]:
            recommendations.append("Install Core ML tools: pip install coremltools")
            recommendations.append("Ensure you're running on Apple Silicon with macOS 13.0+")
            
        if not m4_max_detected and initialization_results["neural_engine_available"]:
            recommendations.append("For optimal 38 TOPS performance, consider upgrading to M4 Max hardware")
            
        if initialization_results["neural_engine_available"] and not initialization_results["coreml_available"]:
            recommendations.append("Core ML pipeline failed - check coremltools installation")
            
        if not initialization_results["inference_engine_ready"]:
            recommendations.append("Neural Inference Engine failed to start - check system resources")
            
        if not initialization_results["model_management_ready"]:
            recommendations.append("Model Management System failed - check storage permissions")
            
        initialization_results["recommendations"] = recommendations
        
        # Calculate initialization time
        initialization_time = (time.time() - start_time) * 1000
        initialization_results["initialization_time_ms"] = initialization_time
        
        # Log summary
        if initialization_results["errors"]:
            logger.error(f"Core ML Neural Engine initialization completed with {len(initialization_results['errors'])} errors")
        elif initialization_results["warnings"]:
            logger.warning(f"Core ML Neural Engine initialization completed with {len(initialization_results['warnings'])} warnings")
        else:
            logger.info(f"Core ML Neural Engine initialization successful in {initialization_time:.2f}ms")
            
        if initialization_results["neural_engine_available"] and m4_max_detected:
            logger.info("🚀 M4 Max Neural Engine (38 TOPS) fully operational for trading!")
            
    except Exception as e:
        logger.error(f"Critical error during Core ML Neural Engine initialization: {e}")
        initialization_results["errors"].append(f"Critical initialization error: {str(e)}")
        initialization_results["initialization_time_ms"] = (time.time() - start_time) * 1000
        
    return initialization_results

def initialize_metal_acceleration(enable_logging: bool = True) -> Dict[str, Any]:
    """
    Legacy Metal GPU acceleration initialization (backward compatibility)
    
    Args:
        enable_logging: Enable detailed logging for initialization process
        
    Returns:
        Dictionary containing initialization status and configuration details
    """
    if not METAL_LEGACY_AVAILABLE:
        return {
            "success": False,
            "message": "Legacy Metal GPU acceleration not available",
            "recommendation": "Use initialize_coreml_acceleration() for Neural Engine support"
        }
    
    try:
        # Call legacy Metal initialization if available
        from .metal_config import metal_device_manager
        from .pytorch_metal import initialize_metal_pytorch
        
        logger.info("Initializing legacy Metal GPU acceleration...")
        
        metal_available = is_metal_available()
        m4_max_detected = is_m4_max_detected()
        
        result = {
            "metal_available": metal_available,
            "m4_max_detected": m4_max_detected,
            "pytorch_metal_available": False,
            "legacy_mode": True
        }
        
        if metal_available:
            pytorch_status = initialize_metal_pytorch()
            result["pytorch_metal_available"] = not pytorch_status.get("error", False)
            
        return result
        
    except Exception as e:
        logger.error(f"Legacy Metal initialization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "recommendation": "Use initialize_coreml_acceleration() for Neural Engine support"
        }

def get_acceleration_status() -> Dict[str, Any]:
    """
    Get current Core ML Neural Engine acceleration status and performance metrics
    
    Returns:
        Dictionary containing current status, capabilities, and performance data
    """
    status = {
        "package_version": __version__,
        "timestamp": time.time(),
        "neural_engine_available": False,
        "m4_max_detected": is_m4_max_detected(),
        "neural_engine_specs": None,
        "inference_engine_status": None,
        "model_management_status": None,
        "coreml_pipeline_status": None,
        "performance_recommendations": []
    }
    
    try:
        # Get Neural Engine status
        neural_engine_status = get_neural_engine_status()
        status["neural_engine_available"] = neural_engine_status.get('neural_engine_available', False)
        
        if status["neural_engine_available"]:
            specs = neural_engine_status.get('specs')
            if specs:
                status["neural_engine_specs"] = {
                    "cores": specs.get('cores', 0),
                    "tops_performance": specs.get('tops_performance', 0),
                    "memory_bandwidth_gbps": specs.get('memory_bandwidth_gb_s', 0),
                    "unified_memory_gb": specs.get('unified_memory_gb', 0),
                    "optimal_batch_size": specs.get('max_batch_size', 0)
                }
        
        # Get Core ML Pipeline status
        try:
            pipeline_status = get_pipeline_status()
            status["coreml_pipeline_status"] = {
                "coreml_available": pipeline_status.get('coreml_available', False),
                "frameworks_available": pipeline_status.get('frameworks_available', {}),
                "conversion_stats": pipeline_status.get('conversion_stats', {}),
                "active_experiments": pipeline_status.get('active_experiments', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            
        # Get Inference Engine status
        try:
            inference_status = get_inference_status()
            status["inference_engine_status"] = {
                "is_running": inference_status.get('is_running', False),
                "max_workers": inference_status.get('max_workers', 0),
                "active_workers": inference_status.get('active_workers', 0),
                "queue_size": inference_status.get('queue_size', 0),
                "cached_models": inference_status.get('cached_models', 0),
                "performance_stats": inference_status.get('performance_stats', {})
            }
        except Exception as e:
            logger.error(f"Failed to get inference status: {e}")
            
        # Get Model Management status
        try:
            management_status = get_model_management_status()
            status["model_management_status"] = {
                "total_models": management_status.get('registry_stats', {}).get('total_models', 0),
                "active_models": management_status.get('registry_stats', {}).get('active_models', 0),
                "active_experiments": management_status.get('experiments', {}).get('active_experiments', 0),
                "pending_retraining_jobs": management_status.get('retraining', {}).get('pending_jobs', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get model management status: {e}")
            
        # Generate performance recommendations
        recommendations = []
        
        if not status["neural_engine_available"]:
            recommendations.append("Install Core ML tools for Neural Engine acceleration")
            
        if status["neural_engine_available"] and not status["m4_max_detected"]:
            recommendations.append("Upgrade to M4 Max for optimal 38 TOPS performance")
            
        if status.get("inference_engine_status", {}).get("queue_size", 0) > 1000:
            recommendations.append("High inference queue - consider increasing worker count")
            
        status["performance_recommendations"] = recommendations
        
    except Exception as e:
        logger.error(f"Failed to get acceleration status: {e}")
        status["error"] = str(e)
        
    return status

# Auto-initialize if imported
try:
    import time
    _init_start = time.time()
    
    # Perform lightweight initialization check
    _init_status = {
        "m4_max_detected": is_m4_max_detected(),
        "neural_engine_ready": False
    }
    
    # Check Neural Engine availability (lightweight check)
    try:
        neural_status = get_neural_engine_status()
        _init_status["neural_engine_ready"] = neural_status.get('neural_engine_available', False)
    except:
        _init_status["neural_engine_ready"] = False
    
    _init_time = (time.time() - _init_start) * 1000
    
    logger.info(f"Core ML Neural Engine package loaded in {_init_time:.2f}ms")
    logger.info(f"Neural Engine ready: {_init_status['neural_engine_ready']}, M4 Max detected: {_init_status['m4_max_detected']}")
    
    if _init_status["neural_engine_ready"] and _init_status["m4_max_detected"]:
        logger.info("🚀 M4 Max Neural Engine (38 TOPS) ready for trading!")
    elif _init_status["neural_engine_ready"]:
        logger.info("⚡ Neural Engine acceleration available for trading")
    else:
        logger.info("💻 CPU fallback mode - Neural Engine not available")
        
except Exception as e:
    logger.error(f"Package initialization warning: {e}")