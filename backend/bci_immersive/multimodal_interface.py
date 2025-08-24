"""
Nautilus Multimodal Interface System

This module implements advanced multimodal interaction combining gesture recognition,
eye tracking, voice commands, neural signals, and spatial computing for intuitive 
trading control.

Key Features:
- Multi-modal input fusion (gesture, eye, voice, neural, haptic)
- Real-time gesture and pose recognition
- Advanced eye tracking with attention analysis
- Voice command processing with natural language understanding
- Neural signal integration with BCI framework
- Spatial interaction in 3D environments
- Intelligent interaction prioritization and conflict resolution

Author: Nautilus Multimodal Interface Team
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timezone
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
from collections import defaultdict, deque

# Computer vision and pose estimation
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    warnings.warn("MediaPipe not available - gesture recognition limited")
    MEDIAPIPE_AVAILABLE = False

# Speech recognition and processing
try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    warnings.warn("Speech libraries not available - voice commands disabled")
    SPEECH_AVAILABLE = False

# Natural language processing
try:
    import spacy
    import nltk
    NLP_AVAILABLE = True
except ImportError:
    warnings.warn("NLP libraries not available - using simple command parsing")
    NLP_AVAILABLE = False

# Eye tracking simulation
try:
    import tobii_research as tr
    TOBII_AVAILABLE = True
except ImportError:
    warnings.warn("Tobii eye tracker not available - using simulation")
    TOBII_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of interaction modalities"""
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"
    VOICE = "voice"
    NEURAL = "neural"
    HAPTIC = "haptic"
    TOUCH = "touch"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"

class InteractionIntent(Enum):
    """High-level interaction intents"""
    SELECT = "select"
    NAVIGATE = "navigate"
    MANIPULATE = "manipulate"
    COMMAND = "command"
    QUERY = "query"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    FOCUS = "focus"

class InteractionPriority(Enum):
    """Priority levels for multimodal interactions"""
    CRITICAL = 1    # Emergency commands, safety
    HIGH = 2        # Direct trading commands  
    MEDIUM = 3      # Navigation, selection
    LOW = 4         # Ambient feedback, non-essential

@dataclass
class MultimodalConfig:
    """Configuration for multimodal interface system"""
    enabled_modalities: List[ModalityType] = field(default_factory=lambda: [
        ModalityType.GESTURE, ModalityType.EYE_TRACKING, ModalityType.VOICE
    ])
    
    # Gesture recognition
    gesture_confidence_threshold: float = 0.7
    gesture_smoothing_window: int = 5
    hand_tracking_enabled: bool = True
    pose_tracking_enabled: bool = True
    
    # Eye tracking
    eye_tracking_frequency: int = 60  # Hz
    fixation_threshold_ms: int = 300
    saccade_threshold_degrees: float = 2.0
    gaze_smoothing_window: int = 3
    
    # Voice recognition
    voice_activation_threshold: float = 0.6
    continuous_listening: bool = True
    wake_word: str = "nautilus"
    language: str = "en-US"
    
    # Neural interface
    neural_confidence_threshold: float = 0.65
    neural_signal_timeout_ms: int = 2000
    
    # Fusion settings
    modality_weights: Dict[ModalityType, float] = field(default_factory=lambda: {
        ModalityType.NEURAL: 1.0,      # Highest priority
        ModalityType.VOICE: 0.9,
        ModalityType.GESTURE: 0.8,
        ModalityType.EYE_TRACKING: 0.7,
        ModalityType.HAPTIC: 0.6
    })
    
    fusion_timeout_ms: int = 500  # Time window for modality fusion
    conflict_resolution: str = "weighted_average"  # Options: "priority", "confidence", "weighted_average"
    
    # Performance
    max_concurrent_processing: int = 4
    input_buffer_size: int = 100
    processing_timeout_ms: int = 200

@dataclass
class ModalityInput:
    """Input from a specific modality"""
    modality: ModalityType
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InteractionResult:
    """Result of multimodal interaction processing"""
    intent: InteractionIntent
    confidence: float
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    contributing_modalities: List[ModalityType] = field(default_factory=list)
    fusion_method: str = ""
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class GestureRecognitionSystem:
    """Advanced gesture and pose recognition using MediaPipe"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.gesture_history = deque(maxlen=config.gesture_smoothing_window)
        self.pose_history = deque(maxlen=config.gesture_smoothing_window)
        
        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe components
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize hand tracking
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Initialize pose estimation
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Gesture templates for recognition
        self._initialize_gesture_templates()
        
        logger.info("Gesture recognition system initialized")
    
    def _initialize_gesture_templates(self):
        """Initialize gesture recognition templates"""
        self.gesture_templates = {
            # Trading gestures
            'buy_signal': {
                'hand_shape': 'thumbs_up',
                'motion': 'upward',
                'confidence_threshold': 0.8
            },
            'sell_signal': {
                'hand_shape': 'thumbs_down', 
                'motion': 'downward',
                'confidence_threshold': 0.8
            },
            'hold_signal': {
                'hand_shape': 'flat_hand',
                'motion': 'stationary',
                'confidence_threshold': 0.7
            },
            'stop_loss': {
                'hand_shape': 'closed_fist',
                'motion': 'sharp_downward',
                'confidence_threshold': 0.9
            },
            
            # Navigation gestures
            'point_select': {
                'hand_shape': 'index_extended',
                'motion': 'pointing',
                'confidence_threshold': 0.7
            },
            'swipe_left': {
                'hand_shape': 'open_hand',
                'motion': 'left_swipe',
                'confidence_threshold': 0.6
            },
            'swipe_right': {
                'hand_shape': 'open_hand',
                'motion': 'right_swipe', 
                'confidence_threshold': 0.6
            },
            'zoom_in': {
                'hand_shape': 'pinch_spread',
                'motion': 'expansion',
                'confidence_threshold': 0.7
            },
            'zoom_out': {
                'hand_shape': 'spread_pinch',
                'motion': 'contraction',
                'confidence_threshold': 0.7
            },
            
            # Control gestures
            'confirm': {
                'hand_shape': 'ok_sign',
                'motion': 'stationary',
                'confidence_threshold': 0.8
            },
            'cancel': {
                'hand_shape': 'cross_hands',
                'motion': 'crossing',
                'confidence_threshold': 0.8
            },
            'menu_open': {
                'hand_shape': 'open_palm',
                'motion': 'upward_palm',
                'confidence_threshold': 0.7
            }
        }
    
    async def process_camera_frame(self, frame: np.ndarray) -> ModalityInput:
        """Process camera frame for gesture recognition"""
        start_time = time.time()
        
        if not MEDIAPIPE_AVAILABLE:
            return ModalityInput(
                modality=ModalityType.GESTURE,
                data={'error': 'MediaPipe not available'},
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=0.0
            )
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Extract hand landmarks and gestures
        hand_data = await self._extract_hand_features(hand_results)
        
        # Extract pose landmarks and gestures  
        pose_data = await self._extract_pose_features(pose_results)
        
        # Recognize gestures
        recognized_gestures = await self._recognize_gestures(hand_data, pose_data)
        
        # Calculate overall confidence
        confidence = max([gesture.get('confidence', 0) for gesture in recognized_gestures] + [0])
        
        processing_time = (time.time() - start_time) * 1000
        
        return ModalityInput(
            modality=ModalityType.GESTURE,
            data={
                'hand_landmarks': hand_data,
                'pose_landmarks': pose_data,
                'recognized_gestures': recognized_gestures,
                'frame_shape': frame.shape
            },
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time
        )
    
    async def _extract_hand_features(self, hand_results) -> Dict[str, Any]:
        """Extract features from hand tracking results"""
        if not hand_results.multi_hand_landmarks:
            return {'hands_detected': False}
        
        hand_features = {
            'hands_detected': True,
            'num_hands': len(hand_results.multi_hand_landmarks),
            'hands': []
        }
        
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            # Calculate hand features
            hand_info = {
                'hand_id': idx,
                'landmarks': landmarks,
                'handedness': hand_results.multi_handedness[idx].classification[0].label,
                'confidence': hand_results.multi_handedness[idx].classification[0].score,
                'features': await self._calculate_hand_features(landmarks)
            }
            
            hand_features['hands'].append(hand_info)
        
        return hand_features
    
    async def _calculate_hand_features(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate geometric features from hand landmarks"""
        if len(landmarks) < 21:  # MediaPipe hand has 21 landmarks
            return {}
        
        # Key landmark indices (MediaPipe hand model)
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        
        # Calculate finger extensions
        finger_extensions = {
            'thumb': self._is_finger_extended(landmarks, [2, 3, 4]),
            'index': self._is_finger_extended(landmarks, [6, 7, 8]),
            'middle': self._is_finger_extended(landmarks, [10, 11, 12]),
            'ring': self._is_finger_extended(landmarks, [14, 15, 16]),
            'pinky': self._is_finger_extended(landmarks, [18, 19, 20])
        }
        
        # Calculate hand shape indicators
        hand_shape = self._classify_hand_shape(finger_extensions, landmarks)
        
        # Calculate palm center and orientation
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Palm base landmarks
        palm_center = {
            'x': np.mean([landmarks[i]['x'] for i in palm_landmarks]),
            'y': np.mean([landmarks[i]['y'] for i in palm_landmarks]),
            'z': np.mean([landmarks[i]['z'] for i in palm_landmarks])
        }
        
        return {
            'finger_extensions': finger_extensions,
            'hand_shape': hand_shape,
            'palm_center': palm_center,
            'extended_fingers_count': sum(finger_extensions.values()),
            'pinch_distance': self._calculate_pinch_distance(landmarks),
            'hand_openness': self._calculate_hand_openness(landmarks)
        }
    
    def _is_finger_extended(self, landmarks: List[Dict[str, float]], finger_indices: List[int]) -> bool:
        """Check if a finger is extended based on joint angles"""
        if len(finger_indices) < 3:
            return False
        
        # Simple heuristic: finger is extended if tip is farther from palm than middle joint
        base_idx, middle_idx, tip_idx = finger_indices
        
        base_pos = np.array([landmarks[base_idx]['x'], landmarks[base_idx]['y']])
        middle_pos = np.array([landmarks[middle_idx]['x'], landmarks[middle_idx]['y']])
        tip_pos = np.array([landmarks[tip_idx]['x'], landmarks[tip_idx]['y']])
        
        # Distance from base to middle vs base to tip
        base_to_middle = np.linalg.norm(middle_pos - base_pos)
        base_to_tip = np.linalg.norm(tip_pos - base_pos)
        
        return base_to_tip > base_to_middle * 1.2  # Threshold for extension
    
    def _classify_hand_shape(self, finger_extensions: Dict[str, bool], landmarks: List[Dict[str, float]]) -> str:
        """Classify hand shape based on finger extensions"""
        extended_count = sum(finger_extensions.values())
        
        if extended_count == 0:
            return 'closed_fist'
        elif extended_count == 5:
            return 'open_hand'
        elif extended_count == 1 and finger_extensions['index']:
            return 'index_extended'
        elif extended_count == 1 and finger_extensions['thumb']:
            # Check if thumb is up or down
            thumb_tip_y = landmarks[4]['y']
            palm_center_y = np.mean([landmarks[i]['y'] for i in [0, 5, 17]])
            if thumb_tip_y < palm_center_y:  # Thumb up
                return 'thumbs_up'
            else:
                return 'thumbs_down'
        elif extended_count == 2 and finger_extensions['thumb'] and finger_extensions['index']:
            # Check pinch configuration
            pinch_distance = self._calculate_pinch_distance(landmarks)
            if pinch_distance < 0.05:  # Threshold for pinch
                return 'pinch'
            else:
                return 'two_fingers'
        elif finger_extensions['thumb'] and finger_extensions['index'] and finger_extensions['middle']:
            return 'ok_sign' if self._is_ok_sign(landmarks) else 'three_fingers'
        else:
            return f'{extended_count}_fingers'
    
    def _calculate_pinch_distance(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate distance between thumb and index finger tips"""
        if len(landmarks) < 21:
            return 1.0  # Max distance
        
        thumb_tip = np.array([landmarks[4]['x'], landmarks[4]['y'], landmarks[4]['z']])
        index_tip = np.array([landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']])
        
        return np.linalg.norm(thumb_tip - index_tip)
    
    def _calculate_hand_openness(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate how open the hand is (0 = closed, 1 = fully open)"""
        if len(landmarks) < 21:
            return 0.0
        
        # Calculate average distance of fingertips from palm center
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        palm_landmarks = [0, 1, 5, 9, 13, 17]
        
        palm_center = np.array([
            np.mean([landmarks[i]['x'] for i in palm_landmarks]),
            np.mean([landmarks[i]['y'] for i in palm_landmarks]),
            np.mean([landmarks[i]['z'] for i in palm_landmarks])
        ])
        
        fingertip_distances = []
        for tip_idx in fingertips:
            tip_pos = np.array([landmarks[tip_idx]['x'], landmarks[tip_idx]['y'], landmarks[tip_idx]['z']])
            distance = np.linalg.norm(tip_pos - palm_center)
            fingertip_distances.append(distance)
        
        # Normalize by average hand size (approximate)
        avg_distance = np.mean(fingertip_distances)
        normalized_openness = min(1.0, avg_distance / 0.15)  # 0.15 is approximate max distance
        
        return normalized_openness
    
    def _is_ok_sign(self, landmarks: List[Dict[str, float]]) -> bool:
        """Check if hand configuration represents OK sign"""
        if len(landmarks) < 21:
            return False
        
        # OK sign: thumb and index finger form circle, other fingers extended
        pinch_distance = self._calculate_pinch_distance(landmarks)
        
        # Check if thumb and index form small circle
        if pinch_distance > 0.05:
            return False
        
        # Check if other fingers are extended
        middle_extended = self._is_finger_extended(landmarks, [10, 11, 12])
        ring_extended = self._is_finger_extended(landmarks, [14, 15, 16])
        pinky_extended = self._is_finger_extended(landmarks, [18, 19, 20])
        
        return middle_extended and ring_extended and pinky_extended
    
    async def _extract_pose_features(self, pose_results) -> Dict[str, Any]:
        """Extract features from pose estimation results"""
        if not pose_results.pose_landmarks:
            return {'pose_detected': False}
        
        landmarks = []
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # Calculate pose features
        pose_features = await self._calculate_pose_features(landmarks)
        
        return {
            'pose_detected': True,
            'landmarks': landmarks,
            'features': pose_features
        }
    
    async def _calculate_pose_features(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pose-based features"""
        if len(landmarks) < 33:  # MediaPipe pose has 33 landmarks
            return {}
        
        # Key pose landmarks (MediaPipe indices)
        nose = 0
        left_shoulder = 11
        right_shoulder = 12
        left_wrist = 15
        right_wrist = 16
        
        # Body orientation
        shoulder_center = {
            'x': (landmarks[left_shoulder]['x'] + landmarks[right_shoulder]['x']) / 2,
            'y': (landmarks[left_shoulder]['y'] + landmarks[right_shoulder]['y']) / 2
        }
        
        # Head orientation
        head_tilt = self._calculate_head_tilt(landmarks)
        
        # Arm positions
        left_arm_raised = landmarks[left_wrist]['y'] < landmarks[left_shoulder]['y']
        right_arm_raised = landmarks[right_wrist]['y'] < landmarks[right_shoulder]['y']
        
        return {
            'shoulder_center': shoulder_center,
            'head_tilt': head_tilt,
            'left_arm_raised': left_arm_raised,
            'right_arm_raised': right_arm_raised,
            'both_arms_raised': left_arm_raised and right_arm_raised,
            'body_lean': self._calculate_body_lean(landmarks)
        }
    
    def _calculate_head_tilt(self, landmarks: List[Dict[str, Any]]) -> float:
        """Calculate head tilt angle"""
        if len(landmarks) < 33:
            return 0.0
        
        # Use nose and shoulder landmarks to estimate head tilt
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Calculate shoulder line angle
        shoulder_vector = np.array([
            right_shoulder['x'] - left_shoulder['x'],
            right_shoulder['y'] - left_shoulder['y']
        ])
        
        # Calculate head position relative to shoulders
        shoulder_center = np.array([
            (left_shoulder['x'] + right_shoulder['x']) / 2,
            (left_shoulder['y'] + right_shoulder['y']) / 2
        ])
        
        head_vector = np.array([nose['x'] - shoulder_center[0], nose['y'] - shoulder_center[1]])
        
        # Calculate tilt angle
        shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        head_angle = np.arctan2(head_vector[1], head_vector[0])
        
        tilt_angle = head_angle - shoulder_angle
        return float(np.degrees(tilt_angle))
    
    def _calculate_body_lean(self, landmarks: List[Dict[str, Any]]) -> float:
        """Calculate body lean angle"""
        if len(landmarks) < 33:
            return 0.0
        
        # Use hip and shoulder landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate torso center line
        shoulder_center = np.array([
            (left_shoulder['x'] + right_shoulder['x']) / 2,
            (left_shoulder['y'] + right_shoulder['y']) / 2
        ])
        
        hip_center = np.array([
            (left_hip['x'] + right_hip['x']) / 2,
            (left_hip['y'] + right_hip['y']) / 2
        ])
        
        # Calculate lean angle from vertical
        torso_vector = shoulder_center - hip_center
        lean_angle = np.arctan2(torso_vector[0], torso_vector[1])
        
        return float(np.degrees(lean_angle))
    
    async def _recognize_gestures(self, hand_data: Dict[str, Any], pose_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize gestures from hand and pose data"""
        recognized_gestures = []
        
        if not hand_data.get('hands_detected', False):
            return recognized_gestures
        
        # Process each detected hand
        for hand_info in hand_data.get('hands', []):
            hand_features = hand_info.get('features', {})
            hand_shape = hand_features.get('hand_shape', '')
            
            # Match against gesture templates
            for gesture_name, template in self.gesture_templates.items():
                confidence = await self._match_gesture_template(
                    gesture_name, template, hand_features, pose_data
                )
                
                if confidence >= template['confidence_threshold']:
                    recognized_gestures.append({
                        'gesture': gesture_name,
                        'confidence': confidence,
                        'hand_id': hand_info['hand_id'],
                        'handedness': hand_info['handedness'],
                        'hand_shape': hand_shape,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
        
        # Add gesture history for smoothing
        self.gesture_history.append(recognized_gestures)
        
        # Apply temporal smoothing
        smoothed_gestures = await self._apply_gesture_smoothing()
        
        return smoothed_gestures
    
    async def _match_gesture_template(self, gesture_name: str, template: Dict[str, Any], 
                                   hand_features: Dict[str, Any], pose_data: Dict[str, Any]) -> float:
        """Match hand features against gesture template"""
        confidence = 0.0
        
        # Check hand shape match
        expected_shape = template.get('hand_shape', '')
        actual_shape = hand_features.get('hand_shape', '')
        
        if expected_shape == actual_shape:
            confidence += 0.6  # Base confidence for shape match
        elif self._are_similar_shapes(expected_shape, actual_shape):
            confidence += 0.3  # Partial confidence for similar shapes
        
        # Check motion requirements (simplified)
        expected_motion = template.get('motion', '')
        
        # This would analyze motion over time using gesture history
        # For now, use simplified motion detection
        motion_confidence = await self._analyze_motion_pattern(expected_motion, hand_features)
        confidence += 0.4 * motion_confidence
        
        return min(1.0, confidence)
    
    def _are_similar_shapes(self, shape1: str, shape2: str) -> bool:
        """Check if two hand shapes are similar"""
        similar_groups = [
            ['thumbs_up', 'thumbs_down'],
            ['open_hand', '5_fingers'],
            ['closed_fist', '0_fingers'],
            ['index_extended', '1_fingers'],
            ['pinch', 'ok_sign']
        ]
        
        for group in similar_groups:
            if shape1 in group and shape2 in group:
                return True
        
        return False
    
    async def _analyze_motion_pattern(self, expected_motion: str, hand_features: Dict[str, Any]) -> float:
        """Analyze motion pattern from gesture history"""
        if len(self.gesture_history) < 2:
            return 0.5  # Neutral confidence without history
        
        # Simplified motion analysis
        # In a full implementation, this would analyze velocity, acceleration, direction
        
        motion_patterns = {
            'upward': 0.7,
            'downward': 0.7,
            'left_swipe': 0.8,
            'right_swipe': 0.8,
            'stationary': 0.9,
            'pointing': 0.6,
            'expansion': 0.5,
            'contraction': 0.5,
            'crossing': 0.4,
            'upward_palm': 0.6
        }
        
        return motion_patterns.get(expected_motion, 0.3)
    
    async def _apply_gesture_smoothing(self) -> List[Dict[str, Any]]:
        """Apply temporal smoothing to gesture recognition"""
        if not self.gesture_history:
            return []
        
        # Count gesture occurrences across history window
        gesture_counts = defaultdict(int)
        gesture_confidences = defaultdict(list)
        latest_gesture_info = {}
        
        for gesture_frame in self.gesture_history:
            for gesture in gesture_frame:
                gesture_name = gesture['gesture']
                gesture_counts[gesture_name] += 1
                gesture_confidences[gesture_name].append(gesture['confidence'])
                latest_gesture_info[gesture_name] = gesture
        
        # Filter gestures that appear consistently
        smoothed_gestures = []
        min_occurrences = max(1, len(self.gesture_history) // 2)  # Must appear in at least half the frames
        
        for gesture_name, count in gesture_counts.items():
            if count >= min_occurrences:
                avg_confidence = np.mean(gesture_confidences[gesture_name])
                
                # Use latest gesture info but with smoothed confidence
                gesture_info = latest_gesture_info[gesture_name].copy()
                gesture_info['confidence'] = avg_confidence
                gesture_info['consistency_score'] = count / len(self.gesture_history)
                
                smoothed_gestures.append(gesture_info)
        
        return smoothed_gestures

class EyeTrackingSystem:
    """Advanced eye tracking system for gaze-based interaction"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.gaze_history = deque(maxlen=config.gaze_smoothing_window)
        self.fixation_detector = FixationDetector(config)
        self.attention_analyzer = AttentionAnalyzer(config)
        
        # Eye tracker initialization
        self.eye_tracker = None
        if TOBII_AVAILABLE:
            self._initialize_tobii_tracker()
        else:
            logger.warning("Using simulated eye tracking")
        
        # Calibration data
        self.calibration_points = []
        self.is_calibrated = False
        
        logger.info("Eye tracking system initialized")
    
    def _initialize_tobii_tracker(self):
        """Initialize Tobii eye tracker"""
        try:
            # Find connected eye trackers
            found_eyetrackers = tr.find_all_eyetrackers()
            
            if found_eyetrackers:
                self.eye_tracker = found_eyetrackers[0]
                logger.info(f"Connected to Tobii eye tracker: {self.eye_tracker.device_name}")
            else:
                logger.warning("No Tobii eye trackers found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Tobii eye tracker: {str(e)}")
    
    async def start_eye_tracking(self) -> Dict[str, Any]:
        """Start eye tracking data collection"""
        if self.eye_tracker:
            try:
                self.eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._gaze_data_callback)
                logger.info("Eye tracking started")
                return {'status': 'started', 'tracker': self.eye_tracker.device_name}
            except Exception as e:
                logger.error(f"Failed to start eye tracking: {str(e)}")
                return {'status': 'error', 'message': str(e)}
        else:
            # Use simulated eye tracking
            asyncio.create_task(self._simulate_eye_tracking())
            return {'status': 'started', 'tracker': 'simulated'}
    
    def _gaze_data_callback(self, gaze_data):
        """Callback for receiving gaze data from Tobii tracker"""
        # Process gaze data in real-time
        asyncio.create_task(self._process_gaze_data(gaze_data))
    
    async def _simulate_eye_tracking(self):
        """Simulate eye tracking data for testing"""
        while True:
            # Generate simulated gaze data
            simulated_gaze = {
                'left_gaze_point_on_display_area': (
                    np.random.uniform(0, 1), 
                    np.random.uniform(0, 1)
                ),
                'right_gaze_point_on_display_area': (
                    np.random.uniform(0, 1), 
                    np.random.uniform(0, 1)
                ),
                'left_gaze_origin_in_user_coordinate_system': (
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50),
                    np.random.uniform(500, 700)
                ),
                'right_gaze_origin_in_user_coordinate_system': (
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50), 
                    np.random.uniform(500, 700)
                ),
                'device_time_stamp': int(time.time() * 1000000),
                'system_time_stamp': int(time.time() * 1000000)
            }
            
            await self._process_gaze_data(simulated_gaze)
            await asyncio.sleep(1.0 / self.config.eye_tracking_frequency)
    
    async def _process_gaze_data(self, gaze_data) -> ModalityInput:
        """Process raw gaze data into meaningful features"""
        start_time = time.time()
        
        # Extract gaze coordinates
        left_gaze = gaze_data.get('left_gaze_point_on_display_area', (0.5, 0.5))
        right_gaze = gaze_data.get('right_gaze_point_on_display_area', (0.5, 0.5))
        
        # Calculate average gaze point
        avg_gaze = (
            (left_gaze[0] + right_gaze[0]) / 2,
            (left_gaze[1] + right_gaze[1]) / 2
        )
        
        # Smooth gaze data
        self.gaze_history.append(avg_gaze)
        smoothed_gaze = self._smooth_gaze_data()
        
        # Detect fixations and saccades
        fixation_data = await self.fixation_detector.process_gaze_point(smoothed_gaze)
        
        # Analyze attention patterns
        attention_data = await self.attention_analyzer.analyze_gaze_pattern(
            smoothed_gaze, fixation_data
        )
        
        # Calculate gaze quality
        gaze_quality = self._calculate_gaze_quality(left_gaze, right_gaze)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ModalityInput(
            modality=ModalityType.EYE_TRACKING,
            data={
                'raw_gaze': avg_gaze,
                'smoothed_gaze': smoothed_gaze,
                'left_gaze': left_gaze,
                'right_gaze': right_gaze,
                'fixation_data': fixation_data,
                'attention_data': attention_data,
                'gaze_quality': gaze_quality
            },
            confidence=gaze_quality,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time
        )
    
    def _smooth_gaze_data(self) -> Tuple[float, float]:
        """Apply smoothing to gaze coordinates"""
        if not self.gaze_history:
            return (0.5, 0.5)
        
        # Simple moving average
        x_coords = [point[0] for point in self.gaze_history]
        y_coords = [point[1] for point in self.gaze_history]
        
        smoothed_x = np.mean(x_coords)
        smoothed_y = np.mean(y_coords)
        
        return (float(smoothed_x), float(smoothed_y))
    
    def _calculate_gaze_quality(self, left_gaze: Tuple[float, float], 
                              right_gaze: Tuple[float, float]) -> float:
        """Calculate gaze data quality"""
        # Check if gaze points are reasonable
        left_valid = 0 <= left_gaze[0] <= 1 and 0 <= left_gaze[1] <= 1
        right_valid = 0 <= right_gaze[0] <= 1 and 0 <= right_gaze[1] <= 1
        
        if not (left_valid and right_valid):
            return 0.0
        
        # Calculate binocular disparity (should be small for good tracking)
        disparity = np.sqrt((left_gaze[0] - right_gaze[0])**2 + (left_gaze[1] - right_gaze[1])**2)
        
        # Good tracking typically has disparity < 0.05
        quality = max(0.0, 1.0 - (disparity / 0.1))
        
        return quality
    
    async def calibrate(self, calibration_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Perform eye tracker calibration"""
        if not self.eye_tracker:
            return {'status': 'error', 'message': 'No eye tracker available'}
        
        try:
            # Start calibration process
            calibration = tr.ScreenBasedCalibration(self.eye_tracker)
            calibration.enter_calibration_mode()
            
            calibration_results = []
            
            for point in calibration_points:
                # Collect calibration data for this point
                calibration.collect_data(point[0], point[1])
                
                # Verify calibration quality
                result = calibration.compute_and_apply()
                calibration_results.append({
                    'point': point,
                    'status': result.status.value,
                    'quality': len([p for p in result.calibration_points if p.calibration_sample])
                })
            
            calibration.leave_calibration_mode()
            self.is_calibrated = True
            
            return {
                'status': 'completed',
                'calibration_points': len(calibration_points),
                'results': calibration_results
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def stop_eye_tracking(self) -> Dict[str, Any]:
        """Stop eye tracking"""
        if self.eye_tracker:
            try:
                self.eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA)
                return {'status': 'stopped'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        
        return {'status': 'stopped', 'tracker': 'simulated'}

class FixationDetector:
    """Detect eye fixations and saccades"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.gaze_buffer = deque(maxlen=50)  # Buffer for fixation detection
        self.current_fixation = None
        self.fixations = []
    
    async def process_gaze_point(self, gaze_point: Tuple[float, float]) -> Dict[str, Any]:
        """Process gaze point and detect fixations"""
        timestamp = time.time()
        self.gaze_buffer.append((gaze_point, timestamp))
        
        # Detect fixations using velocity-based algorithm
        if len(self.gaze_buffer) >= 3:
            is_fixating = await self._is_fixating()
            
            if is_fixating:
                if self.current_fixation is None:
                    # Start new fixation
                    self.current_fixation = {
                        'start_time': timestamp,
                        'start_point': gaze_point,
                        'points': [gaze_point],
                        'duration': 0
                    }
                else:
                    # Continue current fixation
                    self.current_fixation['points'].append(gaze_point)
                    self.current_fixation['duration'] = timestamp - self.current_fixation['start_time']
            else:
                # End current fixation if it exists
                if self.current_fixation is not None:
                    if self.current_fixation['duration'] * 1000 >= self.config.fixation_threshold_ms:
                        # Valid fixation - add to history
                        fixation_center = self._calculate_fixation_center(self.current_fixation['points'])
                        self.current_fixation['center'] = fixation_center
                        self.fixations.append(self.current_fixation.copy())
                    
                    self.current_fixation = None
        
        return {
            'is_fixating': self.current_fixation is not None,
            'current_fixation': self.current_fixation,
            'recent_fixations': self.fixations[-5:],  # Last 5 fixations
            'total_fixations': len(self.fixations)
        }
    
    async def _is_fixating(self) -> bool:
        """Determine if eyes are currently fixating"""
        if len(self.gaze_buffer) < 3:
            return False
        
        # Calculate gaze velocity over recent points
        recent_points = list(self.gaze_buffer)[-3:]
        
        velocities = []
        for i in range(1, len(recent_points)):
            prev_point, prev_time = recent_points[i-1]
            curr_point, curr_time = recent_points[i]
            
            # Calculate angular velocity (simplified)
            distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
            time_diff = curr_time - prev_time
            
            if time_diff > 0:
                velocity = distance / time_diff
                velocities.append(velocity)
        
        if not velocities:
            return False
        
        # Low velocity indicates fixation
        avg_velocity = np.mean(velocities)
        velocity_threshold = self.config.saccade_threshold_degrees / 1000  # Convert to screen coordinates
        
        return avg_velocity < velocity_threshold
    
    def _calculate_fixation_center(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate the center of a fixation"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        return (np.mean(x_coords), np.mean(y_coords))

class AttentionAnalyzer:
    """Analyze attention patterns from eye tracking data"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.attention_history = deque(maxlen=100)
        self.regions_of_interest = {}
    
    async def analyze_gaze_pattern(self, gaze_point: Tuple[float, float], 
                                 fixation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gaze patterns for attention insights"""
        # Determine which ROI the gaze is in
        current_roi = self._get_roi_for_gaze(gaze_point)
        
        # Calculate attention score
        attention_score = await self._calculate_attention_score(gaze_point, fixation_data)
        
        # Track attention over time
        attention_entry = {
            'timestamp': time.time(),
            'gaze_point': gaze_point,
            'roi': current_roi,
            'attention_score': attention_score,
            'is_fixating': fixation_data.get('is_fixating', False)
        }
        
        self.attention_history.append(attention_entry)
        
        # Analyze attention patterns
        attention_patterns = await self._analyze_attention_patterns()
        
        return {
            'current_roi': current_roi,
            'attention_score': attention_score,
            'attention_patterns': attention_patterns,
            'gaze_stability': self._calculate_gaze_stability(),
            'focus_duration': self._calculate_focus_duration(current_roi)
        }
    
    def _get_roi_for_gaze(self, gaze_point: Tuple[float, float]) -> str:
        """Determine which region of interest the gaze point is in"""
        x, y = gaze_point
        
        # Define default ROIs (can be customized)
        default_rois = {
            'top_left': (0.0, 0.0, 0.33, 0.33),
            'top_center': (0.33, 0.0, 0.67, 0.33),
            'top_right': (0.67, 0.0, 1.0, 0.33),
            'center_left': (0.0, 0.33, 0.33, 0.67),
            'center': (0.33, 0.33, 0.67, 0.67),
            'center_right': (0.67, 0.33, 1.0, 0.67),
            'bottom_left': (0.0, 0.67, 0.33, 1.0),
            'bottom_center': (0.33, 0.67, 0.67, 1.0),
            'bottom_right': (0.67, 0.67, 1.0, 1.0)
        }
        
        # Check custom ROIs first
        for roi_name, (x1, y1, x2, y2) in self.regions_of_interest.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return roi_name
        
        # Check default ROIs
        for roi_name, (x1, y1, x2, y2) in default_rois.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return roi_name
        
        return 'unknown'
    
    async def _calculate_attention_score(self, gaze_point: Tuple[float, float], 
                                       fixation_data: Dict[str, Any]) -> float:
        """Calculate attention score based on gaze behavior"""
        score = 0.5  # Base score
        
        # Boost score for fixations
        if fixation_data.get('is_fixating', False):
            score += 0.3
            
            # Additional boost for longer fixations
            current_fixation = fixation_data.get('current_fixation')
            if current_fixation and current_fixation.get('duration', 0) > 0.5:  # > 500ms
                score += 0.2
        
        # Consider gaze stability
        stability = self._calculate_gaze_stability()
        score += stability * 0.3
        
        return min(1.0, score)
    
    def _calculate_gaze_stability(self) -> float:
        """Calculate how stable the gaze has been recently"""
        if len(self.attention_history) < 5:
            return 0.5
        
        recent_points = [entry['gaze_point'] for entry in list(self.attention_history)[-10:]]
        
        # Calculate variance in gaze positions
        x_coords = [p[0] for p in recent_points]
        y_coords = [p[1] for p in recent_points]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # Lower variance = higher stability
        stability = 1.0 - min(1.0, (x_var + y_var) * 10)  # Scale factor
        
        return max(0.0, stability)
    
    def _calculate_focus_duration(self, current_roi: str) -> float:
        """Calculate how long user has been focused on current ROI"""
        if not self.attention_history:
            return 0.0
        
        # Find continuous duration in current ROI
        duration = 0.0
        
        for entry in reversed(self.attention_history):
            if entry['roi'] == current_roi:
                duration = time.time() - entry['timestamp']
            else:
                break
        
        return duration
    
    async def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze overall attention patterns"""
        if not self.attention_history:
            return {}
        
        # ROI visit frequency
        roi_visits = defaultdict(int)
        roi_durations = defaultdict(float)
        
        for entry in self.attention_history:
            roi_visits[entry['roi']] += 1
            if entry['is_fixating']:
                roi_durations[entry['roi']] += 0.1  # Approximate duration
        
        # Most attended ROI
        most_attended_roi = max(roi_visits, key=roi_visits.get) if roi_visits else 'unknown'
        
        # Attention distribution
        total_visits = sum(roi_visits.values())
        attention_distribution = {roi: count/total_visits for roi, count in roi_visits.items()}
        
        return {
            'most_attended_roi': most_attended_roi,
            'attention_distribution': attention_distribution,
            'total_rois_visited': len(roi_visits),
            'average_attention_score': np.mean([entry['attention_score'] for entry in self.attention_history])
        }
    
    def define_roi(self, name: str, x1: float, y1: float, x2: float, y2: float):
        """Define a custom region of interest"""
        self.regions_of_interest[name] = (x1, y1, x2, y2)

class VoiceRecognitionSystem:
    """Advanced voice recognition and natural language processing"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        
        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Configure recognizer
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            self.recognizer.energy_threshold = 300  # Minimum audio energy
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio
        
        # Initialize NLP components
        self.nlp_processor = None
        if NLP_AVAILABLE:
            try:
                self.nlp_processor = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found, using simplified NLP")
        
        # Command templates
        self._initialize_voice_commands()
        
        logger.info("Voice recognition system initialized")
    
    def _initialize_voice_commands(self):
        """Initialize voice command templates"""
        self.voice_commands = {
            # Trading commands
            'buy': {
                'patterns': ['buy', 'purchase', 'acquire', 'long'],
                'intent': InteractionIntent.COMMAND,
                'parameters': ['symbol', 'quantity', 'price']
            },
            'sell': {
                'patterns': ['sell', 'short', 'dispose', 'liquidate'],
                'intent': InteractionIntent.COMMAND,
                'parameters': ['symbol', 'quantity', 'price']
            },
            'hold': {
                'patterns': ['hold', 'wait', 'pause', 'maintain'],
                'intent': InteractionIntent.COMMAND,
                'parameters': ['symbol']
            },
            'close': {
                'patterns': ['close', 'exit', 'terminate', 'end position'],
                'intent': InteractionIntent.COMMAND,
                'parameters': ['symbol', 'all']
            },
            
            # Navigation commands
            'show': {
                'patterns': ['show', 'display', 'view', 'open'],
                'intent': InteractionIntent.NAVIGATE,
                'parameters': ['target', 'chart', 'portfolio', 'news']
            },
            'hide': {
                'patterns': ['hide', 'close', 'minimize', 'remove'],
                'intent': InteractionIntent.NAVIGATE,
                'parameters': ['target']
            },
            'zoom': {
                'patterns': ['zoom in', 'zoom out', 'magnify', 'scale'],
                'intent': InteractionIntent.MANIPULATE,
                'parameters': ['direction', 'level']
            },
            
            # Query commands
            'what': {
                'patterns': ['what is', 'what are', 'tell me', 'information'],
                'intent': InteractionIntent.QUERY,
                'parameters': ['topic', 'symbol']
            },
            'how': {
                'patterns': ['how much', 'how many', 'what price'],
                'intent': InteractionIntent.QUERY,
                'parameters': ['metric', 'symbol']
            },
            
            # Control commands
            'confirm': {
                'patterns': ['yes', 'confirm', 'proceed', 'execute'],
                'intent': InteractionIntent.CONFIRM,
                'parameters': []
            },
            'cancel': {
                'patterns': ['no', 'cancel', 'abort', 'stop'],
                'intent': InteractionIntent.CANCEL,
                'parameters': []
            }
        }
    
    async def start_listening(self, continuous: bool = True) -> Dict[str, Any]:
        """Start voice recognition"""
        if not SPEECH_AVAILABLE:
            return {'status': 'error', 'message': 'Speech recognition not available'}
        
        self.is_listening = True
        
        if continuous:
            # Start continuous listening in background
            asyncio.create_task(self._continuous_listening_loop())
            return {'status': 'started', 'mode': 'continuous'}
        else:
            # Single command recognition
            result = await self._recognize_single_command()
            return {'status': 'completed', 'result': result}
    
    async def _continuous_listening_loop(self):
        """Continuous listening loop for voice commands"""
        while self.is_listening:
            try:
                # Listen for wake word if configured
                if self.config.wake_word:
                    await self._listen_for_wake_word()
                
                # Recognize command
                command_result = await self._recognize_single_command()
                
                if command_result and command_result.get('confidence', 0) > self.config.voice_activation_threshold:
                    # Process recognized command
                    await self._process_voice_command(command_result)
                
            except Exception as e:
                logger.error(f"Error in continuous listening: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _listen_for_wake_word(self):
        """Listen for wake word activation"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=self.config.language).lower()
            
            if self.config.wake_word.lower() in text:
                logger.info(f"Wake word '{self.config.wake_word}' detected")
                return True
                
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            pass  # No speech or recognition error
        
        return False
    
    async def _recognize_single_command(self) -> Optional[Dict[str, Any]]:
        """Recognize a single voice command"""
        try:
            start_time = time.time()
            
            with self.microphone as source:
                logger.info("Listening for voice command...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech using Google's service
            text = self.recognizer.recognize_google(audio, language=self.config.language)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Recognized: '{text}'")
            
            # Process the recognized text
            command_data = await self._parse_voice_command(text)
            
            return {
                'recognized_text': text,
                'command_data': command_data,
                'confidence': command_data.get('confidence', 0.0),
                'processing_time_ms': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {str(e)}")
            return None
        except sr.WaitTimeoutError:
            logger.debug("Listening timeout")
            return None
    
    async def _parse_voice_command(self, text: str) -> Dict[str, Any]:
        """Parse voice command using NLP"""
        text_lower = text.lower()
        
        # Find matching command
        best_match = None
        best_confidence = 0.0
        
        for command_name, command_info in self.voice_commands.items():
            for pattern in command_info['patterns']:
                if pattern in text_lower:
                    # Calculate confidence based on pattern match
                    confidence = len(pattern) / len(text_lower)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            'command': command_name,
                            'intent': command_info['intent'],
                            'confidence': confidence,
                            'raw_text': text
                        }
        
        if best_match:
            # Extract parameters using NLP if available
            parameters = await self._extract_parameters(text, best_match['command'])
            best_match['parameters'] = parameters
        
        return best_match or {'command': 'unknown', 'confidence': 0.0, 'raw_text': text}
    
    async def _extract_parameters(self, text: str, command: str) -> Dict[str, Any]:
        """Extract parameters from voice command"""
        parameters = {}
        
        if self.nlp_processor:
            # Use spaCy for advanced parameter extraction
            doc = self.nlp_processor(text)
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    parameters['price'] = ent.text
                elif ent.label_ == "CARDINAL":
                    parameters['quantity'] = ent.text
                elif ent.label_ == "ORG":
                    parameters['symbol'] = ent.text
            
            # Extract numbers
            numbers = [token.text for token in doc if token.like_num]
            if numbers:
                parameters['numbers'] = numbers
        
        else:
            # Simple parameter extraction
            import re
            
            # Extract potential stock symbols (3-5 uppercase letters)
            symbols = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
            if symbols:
                parameters['symbol'] = symbols[0]
            
            # Extract numbers
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            if numbers:
                parameters['numbers'] = numbers
        
        return parameters
    
    async def _process_voice_command(self, command_result: Dict[str, Any]):
        """Process recognized voice command"""
        logger.info(f"Processing voice command: {command_result}")
        
        # This would trigger the appropriate action based on the command
        # For now, just log the processed command
        command_data = command_result.get('command_data', {})
        
        if command_data.get('intent') == InteractionIntent.COMMAND:
            logger.info(f"Trading command detected: {command_data.get('command')}")
        elif command_data.get('intent') == InteractionIntent.NAVIGATE:
            logger.info(f"Navigation command detected: {command_data.get('command')}")
        elif command_data.get('intent') == InteractionIntent.QUERY:
            logger.info(f"Query command detected: {command_data.get('command')}")
    
    async def stop_listening(self) -> Dict[str, Any]:
        """Stop voice recognition"""
        self.is_listening = False
        return {'status': 'stopped'}

class MultimodalFusionEngine:
    """Engine for fusing multiple interaction modalities"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.modality_inputs = deque(maxlen=config.input_buffer_size)
        self.fusion_history = deque(maxlen=100)
        
        # Initialize conflict resolution
        self.conflict_resolver = ConflictResolver(config)
        
        logger.info("Multimodal fusion engine initialized")
    
    async def add_modality_input(self, modality_input: ModalityInput) -> Dict[str, Any]:
        """Add input from a specific modality"""
        self.modality_inputs.append(modality_input)
        
        # Check if we have inputs to fuse
        fusion_result = await self._attempt_fusion()
        
        return {
            'input_added': True,
            'modality': modality_input.modality.value,
            'confidence': modality_input.confidence,
            'fusion_result': fusion_result
        }
    
    async def _attempt_fusion(self) -> Optional[InteractionResult]:
        """Attempt to fuse recent modality inputs"""
        if not self.modality_inputs:
            return None
        
        # Get recent inputs within fusion window
        current_time = datetime.now(timezone.utc)
        fusion_window_ms = self.config.fusion_timeout_ms
        
        recent_inputs = []
        for modality_input in reversed(self.modality_inputs):
            time_diff = (current_time - modality_input.timestamp).total_seconds() * 1000
            if time_diff <= fusion_window_ms:
                recent_inputs.append(modality_input)
            else:
                break
        
        if len(recent_inputs) < 2:
            # Need at least 2 modalities for fusion
            return None
        
        # Perform fusion
        fusion_result = await self._fuse_modality_inputs(recent_inputs)
        
        if fusion_result:
            self.fusion_history.append(fusion_result)
        
        return fusion_result
    
    async def _fuse_modality_inputs(self, inputs: List[ModalityInput]) -> Optional[InteractionResult]:
        """Fuse multiple modality inputs into single interaction result"""
        start_time = time.time()
        
        # Extract intents from each modality
        modality_intents = []
        
        for modality_input in inputs:
            intent = await self._extract_intent_from_modality(modality_input)
            if intent:
                modality_intents.append({
                    'modality': modality_input.modality,
                    'intent': intent['intent'],
                    'confidence': modality_input.confidence * intent['confidence'],
                    'parameters': intent.get('parameters', {}),
                    'target': intent.get('target')
                })
        
        if not modality_intents:
            return None
        
        # Resolve conflicts between modalities
        resolved_intent = await self.conflict_resolver.resolve_conflicts(modality_intents)
        
        if not resolved_intent:
            return None
        
        processing_time = (time.time() - start_time) * 1000
        
        return InteractionResult(
            intent=resolved_intent['intent'],
            confidence=resolved_intent['confidence'],
            target_object=resolved_intent.get('target'),
            parameters=resolved_intent.get('parameters', {}),
            contributing_modalities=[intent['modality'] for intent in modality_intents],
            fusion_method=self.config.conflict_resolution,
            processing_time_ms=processing_time
        )
    
    async def _extract_intent_from_modality(self, modality_input: ModalityInput) -> Optional[Dict[str, Any]]:
        """Extract interaction intent from modality-specific data"""
        modality = modality_input.modality
        data = modality_input.data
        
        if modality == ModalityType.GESTURE:
            return await self._extract_gesture_intent(data)
        elif modality == ModalityType.EYE_TRACKING:
            return await self._extract_gaze_intent(data)
        elif modality == ModalityType.VOICE:
            return await self._extract_voice_intent(data)
        elif modality == ModalityType.NEURAL:
            return await self._extract_neural_intent(data)
        else:
            return None
    
    async def _extract_gesture_intent(self, gesture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract intent from gesture data"""
        recognized_gestures = gesture_data.get('recognized_gestures', [])
        
        if not recognized_gestures:
            return None
        
        # Use the highest confidence gesture
        best_gesture = max(recognized_gestures, key=lambda g: g.get('confidence', 0))
        
        # Map gesture to intent
        gesture_intent_map = {
            'buy_signal': {'intent': InteractionIntent.COMMAND, 'action': 'buy'},
            'sell_signal': {'intent': InteractionIntent.COMMAND, 'action': 'sell'},
            'point_select': {'intent': InteractionIntent.SELECT, 'action': 'select'},
            'swipe_left': {'intent': InteractionIntent.NAVIGATE, 'action': 'navigate_left'},
            'swipe_right': {'intent': InteractionIntent.NAVIGATE, 'action': 'navigate_right'},
            'zoom_in': {'intent': InteractionIntent.MANIPULATE, 'action': 'zoom_in'},
            'zoom_out': {'intent': InteractionIntent.MANIPULATE, 'action': 'zoom_out'},
            'confirm': {'intent': InteractionIntent.CONFIRM, 'action': 'confirm'},
            'cancel': {'intent': InteractionIntent.CANCEL, 'action': 'cancel'}
        }
        
        gesture_name = best_gesture.get('gesture')
        if gesture_name in gesture_intent_map:
            intent_info = gesture_intent_map[gesture_name]
            return {
                'intent': intent_info['intent'],
                'confidence': best_gesture.get('confidence', 0),
                'parameters': {'action': intent_info['action'], 'gesture': gesture_name}
            }
        
        return None
    
    async def _extract_gaze_intent(self, gaze_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract intent from eye tracking data"""
        attention_data = gaze_data.get('attention_data', {})
        fixation_data = gaze_data.get('fixation_data', {})
        
        # Determine intent based on gaze behavior
        if fixation_data.get('is_fixating', False):
            current_roi = attention_data.get('current_roi', 'unknown')
            focus_duration = attention_data.get('focus_duration', 0)
            
            # Long fixation indicates selection intent
            if focus_duration > 1.0:  # 1 second threshold
                return {
                    'intent': InteractionIntent.SELECT,
                    'confidence': min(1.0, focus_duration / 2.0),  # Scale confidence
                    'target': current_roi,
                    'parameters': {'focus_duration': focus_duration}
                }
            else:
                # Shorter fixation indicates focus/attention
                return {
                    'intent': InteractionIntent.FOCUS,
                    'confidence': 0.6,
                    'target': current_roi,
                    'parameters': {'focus_duration': focus_duration}
                }
        
        return None
    
    async def _extract_voice_intent(self, voice_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract intent from voice command data"""
        command_data = voice_data.get('command_data', {})
        
        if not command_data or command_data.get('command') == 'unknown':
            return None
        
        return {
            'intent': command_data.get('intent', InteractionIntent.COMMAND),
            'confidence': command_data.get('confidence', 0),
            'parameters': command_data.get('parameters', {}),
            'raw_command': command_data.get('command')
        }
    
    async def _extract_neural_intent(self, neural_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract intent from neural signal data"""
        # This would integrate with the BCI framework
        classification_result = neural_data.get('classification_result')
        
        if not classification_result:
            return None
        
        # Map neural classification to interaction intent
        neural_intent_map = {
            'buy': {'intent': InteractionIntent.COMMAND, 'action': 'buy'},
            'sell': {'intent': InteractionIntent.COMMAND, 'action': 'sell'},
            'hold': {'intent': InteractionIntent.COMMAND, 'action': 'hold'},
            'select': {'intent': InteractionIntent.SELECT, 'action': 'select'},
            'focus': {'intent': InteractionIntent.FOCUS, 'action': 'focus'}
        }
        
        predicted_command = classification_result.get('predicted_class')
        if predicted_command in neural_intent_map:
            intent_info = neural_intent_map[predicted_command]
            return {
                'intent': intent_info['intent'],
                'confidence': classification_result.get('confidence', 0),
                'parameters': {'action': intent_info['action'], 'neural_command': predicted_command}
            }
        
        return None
    
    async def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        if not self.fusion_history:
            return {
                'total_fusions': 0,
                'average_confidence': 0.0,
                'most_common_intent': None
            }
        
        # Calculate statistics
        total_fusions = len(self.fusion_history)
        confidences = [result.confidence for result in self.fusion_history]
        average_confidence = np.mean(confidences)
        
        # Count intent frequencies
        intent_counts = defaultdict(int)
        for result in self.fusion_history:
            intent_counts[result.intent.value] += 1
        
        most_common_intent = max(intent_counts, key=intent_counts.get) if intent_counts else None
        
        # Modality participation
        modality_participation = defaultdict(int)
        for result in self.fusion_history:
            for modality in result.contributing_modalities:
                modality_participation[modality.value] += 1
        
        return {
            'total_fusions': total_fusions,
            'average_confidence': average_confidence,
            'confidence_distribution': {
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            },
            'most_common_intent': most_common_intent,
            'intent_distribution': dict(intent_counts),
            'modality_participation': dict(modality_participation),
            'average_processing_time_ms': np.mean([r.processing_time_ms for r in self.fusion_history])
        }

class ConflictResolver:
    """Resolve conflicts between multiple modality inputs"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
    
    async def resolve_conflicts(self, modality_intents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Resolve conflicts between multiple modality intents"""
        if not modality_intents:
            return None
        
        if len(modality_intents) == 1:
            return modality_intents[0]
        
        resolution_method = self.config.conflict_resolution
        
        if resolution_method == "priority":
            return await self._resolve_by_priority(modality_intents)
        elif resolution_method == "confidence":
            return await self._resolve_by_confidence(modality_intents)
        elif resolution_method == "weighted_average":
            return await self._resolve_by_weighted_average(modality_intents)
        else:
            # Default to confidence-based resolution
            return await self._resolve_by_confidence(modality_intents)
    
    async def _resolve_by_priority(self, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts using modality priority"""
        priority_map = {modality: priority.value for modality, priority in [
            (ModalityType.NEURAL, InteractionPriority.CRITICAL),
            (ModalityType.VOICE, InteractionPriority.HIGH),
            (ModalityType.GESTURE, InteractionPriority.MEDIUM),
            (ModalityType.EYE_TRACKING, InteractionPriority.LOW),
            (ModalityType.HAPTIC, InteractionPriority.LOW)
        ]}
        
        # Find highest priority intent
        highest_priority = None
        highest_priority_value = float('inf')
        
        for intent in intents:
            modality = intent['modality']
            priority_value = priority_map.get(modality, InteractionPriority.LOW.value)
            
            if priority_value < highest_priority_value:
                highest_priority_value = priority_value
                highest_priority = intent
        
        return highest_priority
    
    async def _resolve_by_confidence(self, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts using confidence scores"""
        return max(intents, key=lambda x: x.get('confidence', 0))
    
    async def _resolve_by_weighted_average(self, intents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts using weighted averaging"""
        # Group intents by type
        intent_groups = defaultdict(list)
        for intent in intents:
            intent_groups[intent['intent']].append(intent)
        
        # Find the intent type with highest weighted confidence
        best_intent_type = None
        best_weighted_confidence = 0.0
        
        for intent_type, intent_list in intent_groups.items():
            # Calculate weighted confidence for this intent type
            total_weight = 0.0
            weighted_confidence_sum = 0.0
            
            for intent in intent_list:
                modality = intent['modality']
                weight = self.config.modality_weights.get(modality, 0.5)
                confidence = intent.get('confidence', 0)
                
                total_weight += weight
                weighted_confidence_sum += weight * confidence
            
            if total_weight > 0:
                weighted_confidence = weighted_confidence_sum / total_weight
                
                if weighted_confidence > best_weighted_confidence:
                    best_weighted_confidence = weighted_confidence
                    best_intent_type = intent_type
        
        if best_intent_type:
            # Return the highest confidence intent of the best type
            best_intents = intent_groups[best_intent_type]
            best_intent = max(best_intents, key=lambda x: x.get('confidence', 0))
            
            # Update confidence with weighted score
            best_intent['confidence'] = best_weighted_confidence
            
            return best_intent
        
        return None

class MultimodalInterface:
    """Main multimodal interface system orchestrating all interaction modalities"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        
        # Initialize subsystems
        self.gesture_system = GestureRecognitionSystem(config) if ModalityType.GESTURE in config.enabled_modalities else None
        self.eye_tracking_system = EyeTrackingSystem(config) if ModalityType.EYE_TRACKING in config.enabled_modalities else None
        self.voice_system = VoiceRecognitionSystem(config) if ModalityType.VOICE in config.enabled_modalities else None
        self.fusion_engine = MultimodalFusionEngine(config)
        
        # System state
        self.is_active = False
        self.interaction_history = deque(maxlen=1000)
        
        # Performance monitoring
        self.performance_stats = {
            'interactions_processed': 0,
            'average_processing_time_ms': 0.0,
            'modality_usage': defaultdict(int),
            'successful_fusions': 0
        }
        
        logger.info("Multimodal interface system initialized")
    
    async def start_interface(self) -> Dict[str, Any]:
        """Start the multimodal interface"""
        if self.is_active:
            return {'status': 'already_active'}
        
        self.is_active = True
        
        # Start subsystems
        startup_results = {}
        
        if self.gesture_system:
            # Gesture system is activated when processing camera frames
            startup_results['gesture'] = {'status': 'ready'}
        
        if self.eye_tracking_system:
            startup_results['eye_tracking'] = await self.eye_tracking_system.start_eye_tracking()
        
        if self.voice_system:
            startup_results['voice'] = await self.voice_system.start_listening(continuous=True)
        
        logger.info("Multimodal interface started")
        
        return {
            'status': 'started',
            'enabled_modalities': [modality.value for modality in self.config.enabled_modalities],
            'subsystem_status': startup_results
        }
    
    async def process_camera_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process camera frame for gesture recognition"""
        if not self.gesture_system or not self.is_active:
            return {'status': 'gesture_system_not_available'}
        
        start_time = time.time()
        
        # Process gesture recognition
        gesture_input = await self.gesture_system.process_camera_frame(frame)
        
        # Add to fusion engine
        fusion_result = await self.fusion_engine.add_modality_input(gesture_input)
        
        # Update statistics
        self.performance_stats['modality_usage'][ModalityType.GESTURE.value] += 1
        
        processing_time = (time.time() - start_time) * 1000
        self._update_performance_stats(processing_time)
        
        return {
            'modality': ModalityType.GESTURE.value,
            'gesture_data': gesture_input.data,
            'confidence': gesture_input.confidence,
            'fusion_result': fusion_result.get('fusion_result'),
            'processing_time_ms': processing_time
        }
    
    async def process_interaction_result(self, interaction_result: InteractionResult) -> Dict[str, Any]:
        """Process a completed multimodal interaction"""
        if not self.is_active:
            return {'status': 'interface_not_active'}
        
        # Store in history
        self.interaction_history.append(interaction_result)
        
        # Update statistics
        self.performance_stats['interactions_processed'] += 1
        if len(interaction_result.contributing_modalities) > 1:
            self.performance_stats['successful_fusions'] += 1
        
        # Log significant interactions
        if interaction_result.confidence > 0.8:
            logger.info(f"High-confidence multimodal interaction: {interaction_result.intent.value} "
                       f"(confidence: {interaction_result.confidence:.2f}, "
                       f"modalities: {[m.value for m in interaction_result.contributing_modalities]})")
        
        return {
            'interaction_processed': True,
            'intent': interaction_result.intent.value,
            'confidence': interaction_result.confidence,
            'contributing_modalities': [m.value for m in interaction_result.contributing_modalities],
            'target_object': interaction_result.target_object,
            'parameters': interaction_result.parameters
        }
    
    async def calibrate_eye_tracking(self, calibration_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calibrate eye tracking system"""
        if not self.eye_tracking_system:
            return {'status': 'eye_tracking_not_available'}
        
        return await self.eye_tracking_system.calibrate(calibration_points)
    
    async def stop_interface(self) -> Dict[str, Any]:
        """Stop the multimodal interface"""
        if not self.is_active:
            return {'status': 'not_active'}
        
        self.is_active = False
        
        # Stop subsystems
        stop_results = {}
        
        if self.eye_tracking_system:
            stop_results['eye_tracking'] = await self.eye_tracking_system.stop_eye_tracking()
        
        if self.voice_system:
            stop_results['voice'] = await self.voice_system.stop_listening()
        
        # Get final statistics
        final_stats = await self.get_interface_stats()
        
        logger.info("Multimodal interface stopped")
        
        return {
            'status': 'stopped',
            'subsystem_status': stop_results,
            'final_statistics': final_stats
        }
    
    def _update_performance_stats(self, processing_time_ms: float):
        """Update performance statistics"""
        current_avg = self.performance_stats['average_processing_time_ms']
        interactions_count = self.performance_stats['interactions_processed']
        
        if interactions_count > 0:
            self.performance_stats['average_processing_time_ms'] = (
                (current_avg * interactions_count + processing_time_ms) / (interactions_count + 1)
            )
        else:
            self.performance_stats['average_processing_time_ms'] = processing_time_ms
    
    async def get_interface_stats(self) -> Dict[str, Any]:
        """Get comprehensive interface statistics"""
        # Get fusion engine stats
        fusion_stats = await self.fusion_engine.get_fusion_stats()
        
        # Calculate interaction success rate
        total_interactions = len(self.interaction_history)
        high_confidence_interactions = len([
            interaction for interaction in self.interaction_history 
            if interaction.confidence > 0.7
        ])
        
        success_rate = (high_confidence_interactions / total_interactions * 100) if total_interactions > 0 else 0
        
        # Most common intents
        intent_counts = defaultdict(int)
        for interaction in self.interaction_history:
            intent_counts[interaction.intent.value] += 1
        
        return {
            'is_active': self.is_active,
            'enabled_modalities': [modality.value for modality in self.config.enabled_modalities],
            'performance_stats': self.performance_stats,
            'fusion_stats': fusion_stats,
            'interaction_stats': {
                'total_interactions': total_interactions,
                'high_confidence_interactions': high_confidence_interactions,
                'success_rate_percentage': success_rate,
                'intent_distribution': dict(intent_counts)
            },
            'system_health': {
                'gesture_system': self.gesture_system is not None,
                'eye_tracking_system': self.eye_tracking_system is not None,
                'voice_system': self.voice_system is not None,
                'fusion_engine': True
            }
        }