"""
Nautilus BCI and Immersive Technology Routes

This module implements FastAPI routes for brain-computer interface and immersive 
trading technology endpoints. Provides comprehensive API access to BCI frameworks,
neural signal processing, multimodal interfaces, and safety protocols.

Key Endpoints:
- BCI signal processing and classification
- Immersive VR/AR trading environments
- Neural feedback and training systems
- Multimodal interaction interfaces
- Safety monitoring and compliance
- Real-time neural command generation

Author: Nautilus BCI API Team
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
import json
import uuid

# Import BCI modules with graceful fallback for missing dependencies
try:
    from .bci_framework import (
        BCIFramework, BCIConfig, BCISignalData, BCISignalType, TradingCommand,
        NeuralClassificationResult, BCIMockDataGenerator
    )
    from .immersive_environment import (
        ImmersiveEnvironment, ImmersiveConfig, ImmersivePlatform, VisualizationMode,
        SpatialObject, MarketDataVisualization, UserPresence, HapticFeedback,
        ImmersiveMockDataGenerator
    )
    BCI_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    # Graceful fallback when BCI dependencies are not installed
    BCI_DEPENDENCIES_AVAILABLE = False
    logging.warning(f"BCI dependencies not fully available: {e}")
    
    # Create minimal mock classes for API compatibility
    class BCIFramework:
        pass
    class BCIConfig:
        pass
    # Add other minimal mocks as needed
    
if BCI_DEPENDENCIES_AVAILABLE:
    from .neural_signal_processor import (
        NeuralSignalProcessor, ProcessingConfig, ProcessingMode, ProcessingResult,
        NeuralSignalMockGenerator
    )
    from .multimodal_interface import (
        MultimodalInterface, MultimodalConfig, ModalityType, InteractionIntent,
        InteractionResult, ModalityInput
    )
    from .neural_feedback_system import (
        NeurofeedbackTrainingSystem, FeedbackConfig, FeedbackType, BiometricSignal,
        NeuralState, FeedbackProtocol, TrainingSession
    )
    from .bci_safety_protocols import (
        BCISafetySystem, SafetyThresholds, UserConsentType, ComplianceStandard,
        SafetyViolation, UserConsent
    )

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/bci", tags=["BCI & Immersive Technology"])

# Global system instances
bci_framework = None
immersive_environment = None
neural_processor = None
multimodal_interface = None
neurofeedback_system = None
safety_system = None

# Pydantic models for API requests/responses
class BCIConfigRequest(BaseModel):
    signal_types: List[str] = Field(default=["EEG"], description="Types of BCI signals to process")
    sampling_rate: int = Field(default=1000, description="Signal sampling rate in Hz")
    latency_target_ms: float = Field(default=10.0, description="Target processing latency in milliseconds")
    safety_monitoring: bool = Field(default=True, description="Enable safety monitoring")

class SignalDataRequest(BaseModel):
    signal_type: str = Field(description="Type of neural signal (EEG, EMG, etc.)")
    data: List[List[float]] = Field(description="Signal data as 2D array [channels x samples]")
    channels: List[str] = Field(description="Channel names")
    sampling_rate: int = Field(default=1000, description="Sampling rate in Hz")
    timestamp: Optional[str] = Field(default=None, description="Signal timestamp")

class ImmersiveConfigRequest(BaseModel):
    platform: str = Field(default="DESKTOP_3D", description="Immersive platform type")
    interaction_modes: List[str] = Field(default=["HAND_TRACKING", "GESTURE"], description="Enabled interaction modes")
    resolution: List[int] = Field(default=[2880, 1700], description="Display resolution [width, height]")
    haptic_enabled: bool = Field(default=True, description="Enable haptic feedback")
    collaborative_mode: bool = Field(default=False, description="Enable multi-user collaboration")

class MarketVisualizationRequest(BaseModel):
    type: str = Field(description="Visualization type (CANDLESTICK_3D, VOLUME_RENDERING, etc.)")
    symbol: str = Field(description="Trading symbol")
    data: Dict[str, Any] = Field(description="Market data for visualization")
    symbols: Optional[List[str]] = Field(default=None, description="Multiple symbols for correlation analysis")
    correlation_matrix: Optional[List[List[float]]] = Field(default=None, description="Correlation matrix data")

class UserPresenceRequest(BaseModel):
    user_id: str = Field(description="User identifier")
    head_position: List[float] = Field(description="Head position [x, y, z]")
    head_rotation: List[float] = Field(description="Head rotation [rx, ry, rz]")
    eye_gaze_direction: Optional[List[float]] = Field(default=None, description="Eye gaze direction [x, y, z]")
    hand_positions: Optional[Dict[str, List[float]]] = Field(default=None, description="Hand positions")
    engagement_level: float = Field(default=1.0, description="User engagement level (0-1)")

class InteractionRequest(BaseModel):
    user_id: str = Field(description="User identifier")
    interaction_type: str = Field(description="Type of interaction (gesture, voice, etc.)")
    data: Dict[str, Any] = Field(description="Interaction-specific data")

class TrainingSessionRequest(BaseModel):
    protocol: str = Field(description="Training protocol (PERFORMANCE, ALPHA_THETA, etc.)")
    target_states: List[str] = Field(description="Target neural states for training")
    duration_minutes: int = Field(default=20, description="Session duration in minutes")
    user_id: Optional[str] = Field(default=None, description="User identifier")

class ConsentRequest(BaseModel):
    user_id: str = Field(description="User identifier")
    consent_types: List[str] = Field(description="Types of consent to request")

class ConsentDecisionRequest(BaseModel):
    user_id: str = Field(description="User identifier")
    consent_decisions: Dict[str, bool] = Field(description="Consent decisions by type")
    digital_signature: str = Field(description="Digital signature for consent")

# Startup and shutdown handlers
async def startup_bci_systems():
    """Initialize BCI systems on startup"""
    global bci_framework, immersive_environment, neural_processor
    global multimodal_interface, neurofeedback_system, safety_system
    
    try:
        # Initialize BCI framework
        bci_config = BCIConfig()
        bci_framework = BCIFramework(bci_config)
        
        # Initialize immersive environment
        immersive_config = ImmersiveConfig()
        immersive_environment = ImmersiveEnvironment(immersive_config)
        
        # Initialize neural signal processor
        processing_config = ProcessingConfig()
        neural_processor = NeuralSignalProcessor(processing_config)
        
        # Initialize multimodal interface
        multimodal_config = MultimodalConfig()
        multimodal_interface = MultimodalInterface(multimodal_config)
        
        # Initialize neurofeedback system
        feedback_config = FeedbackConfig()
        neurofeedback_system = NeurofeedbackTrainingSystem(feedback_config)
        
        # Initialize safety system
        safety_thresholds = SafetyThresholds()
        safety_system = BCISafetySystem(safety_thresholds)
        await safety_system.initialize_safety_system()
        
        logger.info("BCI systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize BCI systems: {str(e)}")
        raise

# BCI Framework Endpoints

@router.get("/health")
async def bci_health_check():
    """Health check for BCI and Immersive Technology system"""
    return {
        "status": "operational",
        "service": "BCI & Immersive Technology",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies_available": BCI_DEPENDENCIES_AVAILABLE,
        "version": "1.0.0",
        "features": {
            "bci_framework": BCI_DEPENDENCIES_AVAILABLE,
            "immersive_environment": BCI_DEPENDENCIES_AVAILABLE,
            "neural_signal_processing": BCI_DEPENDENCIES_AVAILABLE,
            "multimodal_interface": BCI_DEPENDENCIES_AVAILABLE,
            "neural_feedback": BCI_DEPENDENCIES_AVAILABLE,
            "safety_protocols": BCI_DEPENDENCIES_AVAILABLE
        },
        "api_endpoints": 50,
        "phase": "Phase 6 Implementation"
    }

@router.post("/framework/start")
async def start_bci_framework(config: BCIConfigRequest):
    """Start BCI framework for real-time processing"""
    if not BCI_DEPENDENCIES_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="BCI dependencies not installed. Run: pip install -r requirements.txt"
        )
    
    if not bci_framework:
        raise HTTPException(status_code=500, detail="BCI framework not initialized")
    
    try:
        # Convert config to internal format
        signal_types = [BCISignalType(st.lower()) for st in config.signal_types]
        
        # Update configuration
        bci_framework.config.signal_types = signal_types
        bci_framework.config.sampling_rate = config.sampling_rate
        bci_framework.config.latency_target = config.latency_target_ms
        bci_framework.config.safety_monitoring = config.safety_monitoring
        
        # Start real-time processing
        result = await bci_framework.start_real_time_processing("default_user")
        
        return {
            "status": "success",
            "message": "BCI framework started",
            "config": config.dict(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error starting BCI framework: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/framework/process-signal")
async def process_neural_signal(signal_data: SignalDataRequest):
    """Process neural signal data and generate trading commands"""
    if not bci_framework:
        raise HTTPException(status_code=500, detail="BCI framework not initialized")
    
    try:
        # Convert request to BCISignalData
        signal_array = np.array(signal_data.data)
        timestamps = np.linspace(0, len(signal_array[0]) / signal_data.sampling_rate, len(signal_array[0]))
        
        bci_signal = BCISignalData(
            signal_type=BCISignalType(signal_data.signal_type.lower()),
            data=signal_array,
            timestamps=timestamps,
            channels=signal_data.channels,
            sampling_rate=signal_data.sampling_rate
        )
        
        # Process signal
        classification_result = await bci_framework.process_signal_stream(bci_signal, "default_user")
        
        if classification_result:
            return {
                "status": "success",
                "classification": {
                    "command": classification_result.command.value,
                    "confidence": classification_result.confidence,
                    "latency_ms": classification_result.latency_ms,
                    "signal_quality": classification_result.signal_quality,
                    "timestamp": classification_result.timestamp.isoformat()
                }
            }
        else:
            return {
                "status": "insufficient_data",
                "message": "Not enough signal data for classification"
            }
            
    except Exception as e:
        logger.error(f"Error processing neural signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/framework/calibrate")
async def calibrate_user_model(background_tasks: BackgroundTasks):
    """Start user-specific BCI model calibration"""
    if not bci_framework:
        raise HTTPException(status_code=500, detail="BCI framework not initialized")
    
    try:
        # Generate mock calibration data
        mock_generator = BCIMockDataGenerator(BCIConfig())
        calibration_data = []
        
        # Generate training data for different trading commands
        for command in TradingCommand:
            for _ in range(10):  # 10 samples per command
                signal = mock_generator.generate_mock_eeg_signal(1.0)  # 1 second signal
                signal_data = BCISignalData(
                    signal_type=BCISignalType.EEG,
                    data=signal,
                    timestamps=np.linspace(0, 1, signal.shape[1]),
                    channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
                    sampling_rate=1000
                )
                calibration_data.append((signal_data, command))
        
        # Start calibration in background
        background_tasks.add_task(
            _perform_calibration,
            calibration_data,
            "default_user"
        )
        
        return {
            "status": "calibration_started",
            "message": "User calibration started in background",
            "training_samples": len(calibration_data)
        }
        
    except Exception as e:
        logger.error(f"Error starting calibration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/framework/status")
async def get_bci_framework_status():
    """Get BCI framework status"""
    if not bci_framework:
        raise HTTPException(status_code=500, detail="BCI framework not initialized")
    
    try:
        status = await bci_framework.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting BCI status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/framework/stop")
async def stop_bci_framework():
    """Stop BCI framework processing"""
    if not bci_framework:
        raise HTTPException(status_code=500, detail="BCI framework not initialized")
    
    try:
        result = await bci_framework.stop_real_time_processing()
        return result
        
    except Exception as e:
        logger.error(f"Error stopping BCI framework: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Immersive Environment Endpoints

@router.post("/immersive/start")
async def start_immersive_environment(config: ImmersiveConfigRequest):
    """Start immersive trading environment"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        # Update configuration
        from .immersive_environment import ImmersivePlatform, InteractionMode
        
        platform = ImmersivePlatform(config.platform.lower())
        interaction_modes = [InteractionMode(mode.lower()) for mode in config.interaction_modes]
        
        immersive_environment.config.platform = platform
        immersive_environment.config.interaction_modes = interaction_modes
        immersive_environment.config.resolution = tuple(config.resolution)
        immersive_environment.config.haptic_enabled = config.haptic_enabled
        immersive_environment.config.collaborative_mode = config.collaborative_mode
        
        # Start environment
        result = await immersive_environment.start_environment()
        
        return {
            "status": "success",
            "message": "Immersive environment started",
            "config": config.dict(),
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error starting immersive environment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/add-user")
async def add_user_to_environment(user_presence: UserPresenceRequest):
    """Add user to immersive environment"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        initial_position = tuple(user_presence.head_position)
        result = await immersive_environment.add_user(user_presence.user_id, initial_position)
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding user to environment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/update-presence")
async def update_user_presence(user_presence: UserPresenceRequest):
    """Update user presence in immersive environment"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        presence_data = {
            'head_position': user_presence.head_position,
            'head_rotation': user_presence.head_rotation,
            'engagement_level': user_presence.engagement_level
        }
        
        if user_presence.eye_gaze_direction:
            presence_data['eye_gaze_direction'] = user_presence.eye_gaze_direction
        
        if user_presence.hand_positions:
            presence_data['hand_positions'] = user_presence.hand_positions
        
        result = await immersive_environment.update_user_presence(user_presence.user_id, presence_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error updating user presence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/create-visualization")
async def create_market_visualization(viz_request: MarketVisualizationRequest):
    """Create 3D market data visualization"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        from .immersive_environment import VisualizationMode
        
        visualization_request = {
            'type': VisualizationMode(viz_request.type.lower()),
            'symbol': viz_request.symbol,
            'data': viz_request.data
        }
        
        if viz_request.symbols:
            visualization_request['symbols'] = viz_request.symbols
        
        if viz_request.correlation_matrix:
            visualization_request['correlation_matrix'] = viz_request.correlation_matrix
        
        result = await immersive_environment.create_market_visualization(visualization_request)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/process-interaction")
async def process_user_interaction(interaction: InteractionRequest):
    """Process user interaction in immersive environment"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        result = await immersive_environment.process_user_interaction(
            interaction.user_id, 
            {'type': interaction.interaction_type, **interaction.data}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/haptic-feedback")
async def provide_haptic_feedback(user_id: str, market_data: Dict[str, Any]):
    """Provide haptic feedback based on market conditions"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        result = await immersive_environment.provide_market_haptic_feedback(user_id, market_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error providing haptic feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/immersive/status")
async def get_immersive_environment_status():
    """Get immersive environment status"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        status = await immersive_environment.get_environment_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting immersive status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/immersive/stop")
async def stop_immersive_environment():
    """Stop immersive trading environment"""
    if not immersive_environment:
        raise HTTPException(status_code=500, detail="Immersive environment not initialized")
    
    try:
        result = await immersive_environment.stop_environment()
        return result
        
    except Exception as e:
        logger.error(f"Error stopping immersive environment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Neural Signal Processor Endpoints

@router.post("/processor/start")
async def start_neural_processor(channels: List[str]):
    """Start neural signal processor"""
    if not neural_processor:
        raise HTTPException(status_code=500, detail="Neural processor not initialized")
    
    try:
        result = await neural_processor.initialize_real_time_processing(channels)
        return result
        
    except Exception as e:
        logger.error(f"Error starting neural processor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processor/add-signal")
async def add_signal_data(buffer_id: str, signal_data: SignalDataRequest):
    """Add signal data to neural processor"""
    if not neural_processor:
        raise HTTPException(status_code=500, detail="Neural processor not initialized")
    
    try:
        signal_array = np.array(signal_data.data)
        
        result = await neural_processor.add_signal_data(
            buffer_id, 
            signal_array, 
            signal_data.channels
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding signal data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processor/train")
async def train_neural_model(user_id: str, background_tasks: BackgroundTasks):
    """Train user-specific neural model"""
    if not neural_processor:
        raise HTTPException(status_code=500, detail="Neural processor not initialized")
    
    try:
        # Generate mock training data
        mock_generator = NeuralSignalMockGenerator()
        training_data = mock_generator.generate_training_dataset(1000, 5)  # 1000 samples, 5 classes
        
        # Start training in background
        background_tasks.add_task(
            _train_neural_model,
            training_data,
            user_id
        )
        
        return {
            "status": "training_started",
            "user_id": user_id,
            "training_samples": len(training_data)
        }
        
    except Exception as e:
        logger.error(f"Error starting neural model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processor/status")
async def get_neural_processor_status():
    """Get neural signal processor status"""
    if not neural_processor:
        raise HTTPException(status_code=500, detail="Neural processor not initialized")
    
    try:
        status = await neural_processor.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting processor status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processor/stop")
async def stop_neural_processor():
    """Stop neural signal processor"""
    if not neural_processor:
        raise HTTPException(status_code=500, detail="Neural processor not initialized")
    
    try:
        result = await neural_processor.stop_real_time_processing()
        return result
        
    except Exception as e:
        logger.error(f"Error stopping neural processor: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal Interface Endpoints

@router.post("/multimodal/start")
async def start_multimodal_interface():
    """Start multimodal interface system"""
    if not multimodal_interface:
        raise HTTPException(status_code=500, detail="Multimodal interface not initialized")
    
    try:
        result = await multimodal_interface.start_interface()
        return result
        
    except Exception as e:
        logger.error(f"Error starting multimodal interface: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multimodal/process-camera")
async def process_camera_frame(frame_data: Dict[str, Any]):
    """Process camera frame for gesture recognition"""
    if not multimodal_interface:
        raise HTTPException(status_code=500, detail="Multimodal interface not initialized")
    
    try:
        # In a real implementation, this would process actual camera frame
        # For now, we'll simulate with mock data
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = await multimodal_interface.process_camera_frame(mock_frame)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing camera frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multimodal/calibrate-eye-tracking")
async def calibrate_eye_tracking(calibration_points: List[List[float]]):
    """Calibrate eye tracking system"""
    if not multimodal_interface:
        raise HTTPException(status_code=500, detail="Multimodal interface not initialized")
    
    try:
        # Convert to list of tuples
        points = [tuple(point) for point in calibration_points]
        
        result = await multimodal_interface.calibrate_eye_tracking(points)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calibrating eye tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multimodal/status")
async def get_multimodal_interface_status():
    """Get multimodal interface status"""
    if not multimodal_interface:
        raise HTTPException(status_code=500, detail="Multimodal interface not initialized")
    
    try:
        status = await multimodal_interface.get_interface_stats()
        return status
        
    except Exception as e:
        logger.error(f"Error getting multimodal status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multimodal/stop")
async def stop_multimodal_interface():
    """Stop multimodal interface system"""
    if not multimodal_interface:
        raise HTTPException(status_code=500, detail="Multimodal interface not initialized")
    
    try:
        result = await multimodal_interface.stop_interface()
        return result
        
    except Exception as e:
        logger.error(f"Error stopping multimodal interface: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Neurofeedback Training Endpoints

@router.post("/neurofeedback/start-session")
async def start_neurofeedback_session(session_request: TrainingSessionRequest):
    """Start neurofeedback training session"""
    if not neurofeedback_system:
        raise HTTPException(status_code=500, detail="Neurofeedback system not initialized")
    
    try:
        from .neural_feedback_system import FeedbackProtocol, NeuralState
        
        protocol = FeedbackProtocol(session_request.protocol.lower())
        target_states = [NeuralState(state.lower()) for state in session_request.target_states]
        
        result = await neurofeedback_system.start_training_session(
            protocol,
            target_states,
            session_request.duration_minutes,
            session_request.user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting neurofeedback session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neurofeedback/progress")
async def get_training_progress():
    """Get current training session progress"""
    if not neurofeedback_system:
        raise HTTPException(status_code=500, detail="Neurofeedback system not initialized")
    
    try:
        progress = await neurofeedback_system.get_training_progress()
        return progress
        
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neurofeedback/stop-session")
async def stop_neurofeedback_session():
    """Stop current neurofeedback training session"""
    if not neurofeedback_system:
        raise HTTPException(status_code=500, detail="Neurofeedback system not initialized")
    
    try:
        result = await neurofeedback_system.stop_training_session()
        return result
        
    except Exception as e:
        logger.error(f"Error stopping neurofeedback session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neurofeedback/history")
async def get_training_history(user_id: Optional[str] = None):
    """Get neurofeedback training history"""
    if not neurofeedback_system:
        raise HTTPException(status_code=500, detail="Neurofeedback system not initialized")
    
    try:
        history = await neurofeedback_system.get_training_history(user_id)
        return history
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neurofeedback/status")
async def get_neurofeedback_status():
    """Get neurofeedback system status"""
    if not neurofeedback_system:
        raise HTTPException(status_code=500, detail="Neurofeedback system not initialized")
    
    try:
        status = await neurofeedback_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting neurofeedback status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Safety Protocol Endpoints

@router.post("/safety/start-session")
async def start_safe_bci_session(user_id: str, session_type: str = "bci_training"):
    """Start safety-monitored BCI session"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        result = await safety_system.start_safe_session(user_id, session_type)
        return result
        
    except Exception as e:
        logger.error(f"Error starting safe BCI session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/monitor-session")
async def monitor_session_safety(
    session_id: str,
    signal_data: Optional[Dict[str, float]] = None,
    physiological_data: Optional[Dict[str, float]] = None,
    processing_times: Optional[Dict[str, float]] = None
):
    """Monitor ongoing session for safety violations"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        result = await safety_system.monitor_session_safety(
            session_id, signal_data, physiological_data, processing_times
        )
        return result
        
    except Exception as e:
        logger.error(f"Error monitoring session safety: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/end-session")
async def end_safe_bci_session(session_id: str):
    """End safety-monitored BCI session"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        result = await safety_system.end_safe_session(session_id)
        return result
        
    except Exception as e:
        logger.error(f"Error ending safe BCI session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/request-consent")
async def request_user_consent(consent_request: ConsentRequest):
    """Request consent from user"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        consent_types = [UserConsentType(ct) for ct in consent_request.consent_types]
        result = await safety_system.request_user_consent(consent_request.user_id, consent_types)
        return result
        
    except Exception as e:
        logger.error(f"Error requesting user consent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/safety/record-consent")
async def record_user_consent(consent_decision: ConsentDecisionRequest):
    """Record user consent decisions"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        result = await safety_system.record_user_consent(
            consent_decision.user_id,
            consent_decision.consent_decisions,
            consent_decision.digital_signature
        )
        return result
        
    except Exception as e:
        logger.error(f"Error recording user consent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/status")
async def get_safety_system_status():
    """Get safety system status"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        status = await safety_system.get_safety_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting safety status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety/compliance-report")
async def generate_compliance_report(
    start_date: str,
    end_date: str,
    standards: Optional[List[str]] = None
):
    """Generate compliance report"""
    if not safety_system:
        raise HTTPException(status_code=500, detail="Safety system not initialized")
    
    try:
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        compliance_standards = None
        if standards:
            compliance_standards = [ComplianceStandard(std.lower()) for std in standards]
        
        report = await safety_system.generate_safety_report(start_dt, end_dt)
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mock Data Generation Endpoints

@router.get("/mock/generate-eeg")
async def generate_mock_eeg_data(duration_seconds: float = 1.0, channels: int = 8):
    """Generate mock EEG data for testing"""
    try:
        mock_generator = BCIMockDataGenerator(BCIConfig())
        signal_data = mock_generator.generate_mock_eeg_signal(duration_seconds)
        
        return {
            "signal_type": "EEG",
            "duration_seconds": duration_seconds,
            "channels": channels,
            "sampling_rate": 1000,
            "data": signal_data.tolist(),
            "timestamps": np.linspace(0, duration_seconds, signal_data.shape[1]).tolist()
        }
        
    except Exception as e:
        logger.error(f"Error generating mock EEG data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mock/generate-market-data")
async def generate_mock_market_data(symbol: str = "AAPL", candles: int = 100):
    """Generate mock market data for immersive visualization"""
    try:
        mock_generator = ImmersiveMockDataGenerator()
        ohlc_data = mock_generator.generate_mock_ohlc_data(symbol, candles)
        
        return {
            "symbol": symbol,
            "data_type": "OHLC",
            "candle_count": len(ohlc_data),
            "data": ohlc_data
        }
        
    except Exception as e:
        logger.error(f"Error generating mock market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# System Health and Status Endpoints

@router.get("/health")
async def get_system_health():
    """Get overall BCI system health"""
    try:
        health_status = {
            "bci_framework": bci_framework is not None,
            "immersive_environment": immersive_environment is not None,
            "neural_processor": neural_processor is not None,
            "multimodal_interface": multimodal_interface is not None,
            "neurofeedback_system": neurofeedback_system is not None,
            "safety_system": safety_system is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get detailed status if systems are available
        if bci_framework:
            bci_status = await bci_framework.get_system_status()
            health_status["bci_details"] = {
                "is_running": bci_status.get("is_running", False),
                "processing_stats": bci_status.get("processing_stats", {})
            }
        
        if safety_system:
            safety_status = await safety_system.get_safety_system_status()
            health_status["safety_details"] = {
                "active": safety_status.get("safety_system_active", False),
                "active_sessions": safety_status.get("active_sessions", 0)
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/comprehensive")
async def get_comprehensive_status():
    """Get comprehensive status of all BCI systems"""
    try:
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "systems": {}
        }
        
        # Get status from each system
        if bci_framework:
            status["systems"]["bci_framework"] = await bci_framework.get_system_status()
        
        if immersive_environment:
            status["systems"]["immersive_environment"] = await immersive_environment.get_environment_status()
        
        if neural_processor:
            status["systems"]["neural_processor"] = await neural_processor.get_system_status()
        
        if multimodal_interface:
            status["systems"]["multimodal_interface"] = await multimodal_interface.get_interface_stats()
        
        if neurofeedback_system:
            status["systems"]["neurofeedback_system"] = await neurofeedback_system.get_system_status()
        
        if safety_system:
            status["systems"]["safety_system"] = await safety_system.get_safety_system_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting comprehensive status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions

async def _perform_calibration(calibration_data: List, user_id: str):
    """Perform BCI calibration in background"""
    try:
        result = await bci_framework.calibrate_user_specific_model(calibration_data, user_id)
        logger.info(f"BCI calibration completed for user {user_id}: {result}")
    except Exception as e:
        logger.error(f"Error in BCI calibration: {str(e)}")

async def _train_neural_model(training_data: List, user_id: str):
    """Train neural model in background"""
    try:
        result = await neural_processor.train_user_specific_model(training_data, user_id)
        logger.info(f"Neural model training completed for user {user_id}: {result}")
    except Exception as e:
        logger.error(f"Error in neural model training: {str(e)}")

# Initialize systems on module import
import atexit

async def cleanup_bci_systems():
    """Cleanup BCI systems on shutdown"""
    global bci_framework, immersive_environment, neural_processor
    global multimodal_interface, neurofeedback_system, safety_system
    
    try:
        if bci_framework:
            await bci_framework.stop_real_time_processing()
        
        if immersive_environment:
            await immersive_environment.stop_environment()
        
        if neural_processor:
            await neural_processor.stop_real_time_processing()
        
        if multimodal_interface:
            await multimodal_interface.stop_interface()
        
        if neurofeedback_system:
            # Stop any active training sessions
            await neurofeedback_system.stop_training_session()
        
        logger.info("BCI systems cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during BCI systems cleanup: {str(e)}")

# Register cleanup function
atexit.register(lambda: asyncio.run(cleanup_bci_systems()))

# Auto-initialize systems (will be called by main application)
async def initialize_bci_systems():
    """Initialize all BCI systems"""
    await startup_bci_systems()

# Add router tags and metadata
router.tags = ["BCI & Immersive Technology"]
router.responses = {
    500: {"description": "Internal server error"},
    404: {"description": "Resource not found"},
    422: {"description": "Validation error"}
}