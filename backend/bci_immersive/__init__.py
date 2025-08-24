"""
Nautilus Phase 6: Brain-Computer Interface and Immersive Trading Technologies

This module implements revolutionary human-computer interaction paradigms for 
intuitive trading control through brain-computer interfaces, immersive VR/AR 
environments, and multimodal interaction systems.

Key Components:
- Brain-computer interface framework with EEG/fNIRS signal processing
- Immersive VR/AR trading environments with haptic feedback
- Real-time neural signal processing for trading decisions
- Multi-modal interaction (gesture, eye-tracking, voice, neural)
- Neural feedback and biometric optimization systems
- Medical device safety protocols and regulatory compliance

Author: Nautilus BCI & Immersive Technology Team
Version: 1.0.0
"""

# Import modules with graceful fallback for missing dependencies
import logging

logger = logging.getLogger(__name__)

try:
    from .bci_framework import *
    from .immersive_environment import *
    from .neural_signal_processor import *
    from .multimodal_interface import *
    from .neural_feedback_system import *
    from .bci_safety_protocols import *
    BCI_MODULES_AVAILABLE = True
    logger.info("✅ BCI modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ BCI dependencies not fully available: {e}")
    BCI_MODULES_AVAILABLE = False

# Always import routes (they handle missing dependencies internally)
from .bci_routes import *

__version__ = "1.0.0"
__author__ = "Nautilus BCI & Immersive Technology Team"

# Module configuration
BCI_CONFIG = {
    "version": "1.0.0",
    "safety_standards": ["ISO 14155", "FDA 21 CFR 820", "IEC 60601"],
    "supported_devices": ["EEG", "fNIRS", "EMG", "EOG", "ECG"],
    "latency_target": 10,  # milliseconds
    "sampling_rate": 1000,  # Hz
    "immersive_platforms": ["VR", "AR", "MR", "Haptic"]
}