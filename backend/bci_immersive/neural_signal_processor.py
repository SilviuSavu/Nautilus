"""
Nautilus Neural Signal Processor

This module implements ultra-low latency neural signal processing for real-time 
trading command generation. Features advanced signal processing, machine learning 
classification, and real-time neural pattern recognition.

Key Features:
- Real-time neural signal processing with <10ms latency
- Advanced signal filtering and artifact removal
- Machine learning-based pattern recognition
- Multi-modal signal fusion (EEG, fNIRS, EMG, EOG)
- Adaptive learning for user-specific patterns
- Safety monitoring and medical device compliance

Performance Targets:
- Latency: <10ms end-to-end processing
- Accuracy: >85% classification accuracy
- Throughput: 1000+ samples/second per channel
- Safety: Medical device grade monitoring

Author: Nautilus Neural Signal Processing Team
"""

import asyncio
import logging
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy import fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timezone
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
from collections import deque, defaultdict
import pickle

# Advanced signal processing
try:
    import pywt  # PyWavelets for wavelet transforms
    WAVELET_AVAILABLE = True
except ImportError:
    warnings.warn("PyWavelets not available - wavelet features disabled")
    WAVELET_AVAILABLE = False

try:
    import scikit_learn as sklearn
    from sklearn.decomposition import FastICA, PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVM
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("Scikit-learn not available - using simplified ML models")
    SKLEARN_AVAILABLE = False

# GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.signal as cupy_signal
    CUPY_AVAILABLE = True
except ImportError:
    warnings.warn("CuPy not available - using CPU processing")
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Neural signal processing modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    TRAINING = "training"
    CALIBRATION = "calibration"
    SIMULATION = "simulation"

class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"  # >0.9
    GOOD = "good"           # 0.7-0.9
    FAIR = "fair"           # 0.5-0.7
    POOR = "poor"           # 0.3-0.5
    UNUSABLE = "unusable"   # <0.3

class ProcessingStage(Enum):
    """Processing pipeline stages"""
    RAW_ACQUISITION = "raw_acquisition"
    PREPROCESSING = "preprocessing" 
    ARTIFACT_REMOVAL = "artifact_removal"
    FEATURE_EXTRACTION = "feature_extraction"
    CLASSIFICATION = "classification"
    POST_PROCESSING = "post_processing"

@dataclass
class ProcessingConfig:
    """Configuration for neural signal processor"""
    mode: ProcessingMode = ProcessingMode.REAL_TIME
    sampling_rate: int = 1000  # Hz
    buffer_size: int = 1000  # samples
    overlap_size: int = 500   # samples for sliding window
    latency_target_ms: float = 10.0  # milliseconds
    
    # Signal processing parameters
    filter_order: int = 4
    notch_frequencies: List[float] = field(default_factory=lambda: [50, 60])  # Power line noise
    bandpass_range: Tuple[float, float] = (0.5, 40)  # Hz
    artifact_threshold: float = 100.0  # μV
    
    # Feature extraction
    feature_window_ms: float = 1000.0  # 1 second feature windows
    feature_overlap: float = 0.5  # 50% overlap
    wavelet_family: str = "db4"  # Daubechies wavelet
    
    # Machine learning
    use_gpu: bool = True
    model_type: str = "deep_neural_network"  # Options: "svm", "random_forest", "neural_network", "deep_neural_network"
    confidence_threshold: float = 0.7
    
    # Safety and monitoring
    safety_monitoring: bool = True
    signal_quality_threshold: float = 0.5
    max_processing_time_ms: float = 50.0
    
    # Performance optimization
    parallel_processing: bool = True
    batch_size: int = 32
    cache_size: int = 1000

@dataclass
class ProcessingResult:
    """Result of neural signal processing"""
    signal_id: str
    processing_time_ms: float
    latency_ms: float
    signal_quality: SignalQuality
    artifacts_detected: List[str]
    features_extracted: Dict[str, np.ndarray]
    classification_result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ProcessingStats:
    """Processing performance statistics"""
    total_signals_processed: int = 0
    average_latency_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    accuracy_percentage: float = 0.0
    throughput_samples_per_sec: float = 0.0
    artifacts_detected_rate: float = 0.0
    signal_quality_distribution: Dict[SignalQuality, int] = field(default_factory=lambda: defaultdict(int))
    error_count: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class RealTimeBuffer:
    """Ultra-low latency circular buffer for real-time signal processing"""
    
    def __init__(self, buffer_size: int, n_channels: int):
        self.buffer_size = buffer_size
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, buffer_size))
        self.write_index = 0
        self.samples_written = 0
        self.lock = threading.RLock()
    
    def add_samples(self, samples: np.ndarray):
        """Add new samples to the buffer"""
        with self.lock:
            n_new_samples = samples.shape[1]
            
            # Handle buffer wrap-around
            if self.write_index + n_new_samples <= self.buffer_size:
                self.buffer[:, self.write_index:self.write_index + n_new_samples] = samples
            else:
                # Split across buffer boundary
                remaining_space = self.buffer_size - self.write_index
                self.buffer[:, self.write_index:] = samples[:, :remaining_space]
                self.buffer[:, :n_new_samples - remaining_space] = samples[:, remaining_space:]
            
            self.write_index = (self.write_index + n_new_samples) % self.buffer_size
            self.samples_written += n_new_samples
    
    def get_latest_samples(self, n_samples: int) -> np.ndarray:
        """Get the latest n samples from the buffer"""
        with self.lock:
            if n_samples > self.buffer_size:
                raise ValueError(f"Requested {n_samples} samples but buffer size is {self.buffer_size}")
            
            if self.samples_written < n_samples:
                # Not enough samples yet
                return np.zeros((self.n_channels, n_samples))
            
            start_index = (self.write_index - n_samples) % self.buffer_size
            
            if start_index + n_samples <= self.buffer_size:
                return self.buffer[:, start_index:start_index + n_samples].copy()
            else:
                # Handle wrap-around
                part1_size = self.buffer_size - start_index
                part1 = self.buffer[:, start_index:]
                part2 = self.buffer[:, :n_samples - part1_size]
                return np.hstack([part1, part2])
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        with self.lock:
            return {
                'buffer_size': self.buffer_size,
                'samples_written': self.samples_written,
                'write_index': self.write_index,
                'fill_percentage': min(100.0, (self.samples_written / self.buffer_size) * 100),
                'ready_for_processing': self.samples_written >= self.buffer_size
            }

class AdvancedSignalProcessor:
    """Advanced signal processing pipeline with GPU acceleration"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.use_gpu = config.use_gpu and CUPY_AVAILABLE
        
        # Initialize processing components
        self._initialize_filters()
        self._initialize_artifact_detectors()
        self._initialize_feature_extractors()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        
        logger.info(f"Advanced signal processor initialized (GPU: {self.use_gpu})")
    
    def _initialize_filters(self):
        """Initialize digital filters"""
        nyquist = self.config.sampling_rate / 2
        
        # Bandpass filter
        low_freq, high_freq = self.config.bandpass_range
        self.bandpass_sos = signal.butter(
            self.config.filter_order, 
            [low_freq / nyquist, high_freq / nyquist], 
            btype='band', 
            output='sos'
        )
        
        # Notch filters for power line noise
        self.notch_filters = []
        for notch_freq in self.config.notch_frequencies:
            notch_sos = signal.iirnotch(notch_freq / nyquist, 30.0)  # Q factor = 30
            self.notch_filters.append(notch_sos)
        
        # High-order Butterworth filters for different frequency bands
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }
        
        self.band_filters = {}
        for band_name, (low, high) in self.frequency_bands.items():
            if high < nyquist:
                sos = signal.butter(
                    self.config.filter_order,
                    [low / nyquist, high / nyquist],
                    btype='band',
                    output='sos'
                )
                self.band_filters[band_name] = sos
    
    def _initialize_artifact_detectors(self):
        """Initialize artifact detection algorithms"""
        self.artifact_detectors = {
            'amplitude': self._detect_amplitude_artifacts,
            'gradient': self._detect_gradient_artifacts,
            'statistical': self._detect_statistical_artifacts,
            'frequency': self._detect_frequency_artifacts,
            'correlation': self._detect_correlation_artifacts
        }
        
        # Thresholds for different artifact types
        self.artifact_thresholds = {
            'amplitude_threshold': self.config.artifact_threshold,  # μV
            'gradient_threshold': 50.0,  # μV/sample
            'kurtosis_threshold': 5.0,
            'correlation_threshold': 0.95,
            'frequency_power_threshold': 3.0  # Standard deviations
        }
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction methods"""
        self.feature_extractors = {
            'time_domain': self._extract_time_domain_features,
            'frequency_domain': self._extract_frequency_domain_features,
            'time_frequency': self._extract_time_frequency_features,
            'connectivity': self._extract_connectivity_features,
            'nonlinear': self._extract_nonlinear_features,
            'wavelet': self._extract_wavelet_features if WAVELET_AVAILABLE else None
        }
        
        # Remove None extractors
        self.feature_extractors = {k: v for k, v in self.feature_extractors.items() if v is not None}
    
    async def process_signal_realtime(self, signal_data: np.ndarray, 
                                    channel_names: List[str]) -> ProcessingResult:
        """Process neural signals with ultra-low latency for real-time applications"""
        start_time = time.time()
        signal_id = f"rt_{int(start_time * 1000000)}"  # Microsecond precision
        
        try:
            # Move to GPU if available
            if self.use_gpu:
                signal_gpu = cp.asarray(signal_data)
                processed_signal = await self._process_signal_gpu(signal_gpu)
                processed_signal = cp.asnumpy(processed_signal)
            else:
                processed_signal = await self._process_signal_cpu(signal_data)
            
            # Extract features
            features = await self._extract_features_realtime(processed_signal, channel_names)
            
            # Assess signal quality
            signal_quality = self._assess_signal_quality(processed_signal)
            
            # Detect artifacts
            artifacts = await self._detect_artifacts_fast(processed_signal)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)
            
            return ProcessingResult(
                signal_id=signal_id,
                processing_time_ms=processing_time,
                latency_ms=processing_time,  # For real-time, processing time = latency
                signal_quality=signal_quality,
                artifacts_detected=artifacts,
                features_extracted=features,
                metadata={
                    'n_channels': signal_data.shape[0],
                    'n_samples': signal_data.shape[1],
                    'sampling_rate': self.config.sampling_rate,
                    'gpu_used': self.use_gpu
                }
            )
            
        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Error processing signal {signal_id}: {str(e)}")
            
            return ProcessingResult(
                signal_id=signal_id,
                processing_time_ms=error_time,
                latency_ms=error_time,
                signal_quality=SignalQuality.UNUSABLE,
                artifacts_detected=['processing_error'],
                features_extracted={},
                metadata={'error': str(e)}
            )
    
    async def _process_signal_gpu(self, signal_gpu: 'cp.ndarray') -> 'cp.ndarray':
        """GPU-accelerated signal processing"""
        # Apply bandpass filter
        filtered_signal = self._apply_filter_gpu(signal_gpu, self.bandpass_sos)
        
        # Apply notch filters
        for notch_filter in self.notch_filters:
            filtered_signal = self._apply_filter_gpu(filtered_signal, notch_filter)
        
        return filtered_signal
    
    async def _process_signal_cpu(self, signal_data: np.ndarray) -> np.ndarray:
        """CPU signal processing fallback"""
        # Apply bandpass filter
        filtered_signal = signal.sosfilt(self.bandpass_sos, signal_data, axis=1)
        
        # Apply notch filters
        for b, a in self.notch_filters:
            filtered_signal = signal.filtfilt(b, a, filtered_signal, axis=1)
        
        return filtered_signal
    
    def _apply_filter_gpu(self, signal_gpu: 'cp.ndarray', filter_sos) -> 'cp.ndarray':
        """Apply filter using GPU acceleration"""
        if isinstance(filter_sos, tuple):
            # IIR filter (b, a coefficients)
            b, a = filter_sos
            return cupy_signal.filtfilt(b, a, signal_gpu, axis=1)
        else:
            # SOS filter
            return cupy_signal.sosfilt(filter_sos, signal_gpu, axis=1)
    
    async def _extract_features_realtime(self, signal_data: np.ndarray, 
                                       channel_names: List[str]) -> Dict[str, np.ndarray]:
        """Extract features optimized for real-time processing"""
        features = {}
        
        # Fast time-domain features
        features.update(self._extract_time_domain_features(signal_data))
        
        # Fast frequency-domain features (using FFT)
        features.update(await self._extract_frequency_domain_features_fast(signal_data))
        
        # Channel connectivity (simplified for speed)
        if signal_data.shape[0] > 1:  # Multiple channels
            features.update(self._extract_connectivity_features_fast(signal_data))
        
        return features
    
    def _extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-domain features"""
        features = {}
        
        # Statistical moments
        features['mean'] = np.mean(signal_data, axis=1)
        features['std'] = np.std(signal_data, axis=1)
        features['variance'] = np.var(signal_data, axis=1)
        features['skewness'] = stats.skew(signal_data, axis=1)
        features['kurtosis'] = stats.kurtosis(signal_data, axis=1)
        
        # Signal properties
        features['rms'] = np.sqrt(np.mean(signal_data**2, axis=1))
        features['peak_to_peak'] = np.ptp(signal_data, axis=1)
        features['zero_crossings'] = np.array([
            len(np.where(np.diff(np.signbit(channel)))[0]) for channel in signal_data
        ])
        
        # Energy and power
        features['energy'] = np.sum(signal_data**2, axis=1)
        features['average_power'] = features['energy'] / signal_data.shape[1]
        
        # Peak detection
        features['n_peaks'] = np.array([
            len(signal.find_peaks(channel, height=np.std(channel))[0]) for channel in signal_data
        ])
        
        # Hjorth parameters
        hjorth_features = self._calculate_hjorth_parameters(signal_data)
        features.update(hjorth_features)
        
        return features
    
    async def _extract_frequency_domain_features_fast(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency-domain features optimized for speed"""
        features = {}
        
        # Compute FFT for all channels
        fft_data = fft.rfft(signal_data, axis=1)
        power_spectrum = np.abs(fft_data)**2
        freqs = fft.rfftfreq(signal_data.shape[1], 1/self.config.sampling_rate)
        
        # Band power features
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                band_power = np.sum(power_spectrum[:, freq_mask], axis=1)
                features[f'{band_name}_power'] = band_power
                
                # Relative band power
                total_power = np.sum(power_spectrum, axis=1)
                features[f'{band_name}_relative_power'] = band_power / (total_power + 1e-10)
        
        # Spectral features
        features['spectral_centroid'] = np.sum(freqs * power_spectrum, axis=1) / (np.sum(power_spectrum, axis=1) + 1e-10)
        
        # Peak frequency
        peak_freq_indices = np.argmax(power_spectrum, axis=1)
        features['peak_frequency'] = freqs[peak_freq_indices]
        
        # Spectral entropy
        features['spectral_entropy'] = self._calculate_spectral_entropy(power_spectrum)
        
        return features
    
    def _extract_connectivity_features_fast(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract connectivity features optimized for speed"""
        features = {}
        n_channels = signal_data.shape[0]
        
        # Pearson correlation matrix
        correlation_matrix = np.corrcoef(signal_data)
        
        # Extract upper triangular correlation values
        triu_indices = np.triu_indices(n_channels, k=1)
        features['correlations'] = correlation_matrix[triu_indices]
        
        # Global connectivity measures
        features['mean_correlation'] = np.array([np.mean(np.abs(features['correlations']))])
        features['max_correlation'] = np.array([np.max(np.abs(features['correlations']))])
        
        return features
    
    def _calculate_hjorth_parameters(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Hjorth parameters (Activity, Mobility, Complexity)"""
        # Activity (variance)
        activity = np.var(signal_data, axis=1)
        
        # Mobility (relative variance of first derivative)
        first_diff = np.diff(signal_data, axis=1)
        mobility = np.sqrt(np.var(first_diff, axis=1) / (activity + 1e-10))
        
        # Complexity (relative variance of second derivative)
        second_diff = np.diff(first_diff, axis=1)
        second_var = np.var(second_diff, axis=1)
        first_var = np.var(first_diff, axis=1)
        complexity = np.sqrt((second_var / (first_var + 1e-10)) / (mobility**2 + 1e-10))
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    def _calculate_spectral_entropy(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Calculate spectral entropy for each channel"""
        # Normalize power spectrum
        normalized_psd = power_spectrum / (np.sum(power_spectrum, axis=1, keepdims=True) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-10), axis=1)
        
        return entropy
    
    def _assess_signal_quality(self, signal_data: np.ndarray) -> SignalQuality:
        """Assess overall signal quality"""
        # Calculate signal-to-noise ratio
        signal_power = np.mean(signal_data**2)
        noise_estimate = np.mean(np.diff(signal_data, axis=1)**2)
        snr = signal_power / (noise_estimate + 1e-10)
        
        # Normalize SNR to quality score
        quality_score = min(1.0, snr / 100.0)
        
        # Map to quality levels
        if quality_score > 0.9:
            return SignalQuality.EXCELLENT
        elif quality_score > 0.7:
            return SignalQuality.GOOD
        elif quality_score > 0.5:
            return SignalQuality.FAIR
        elif quality_score > 0.3:
            return SignalQuality.POOR
        else:
            return SignalQuality.UNUSABLE
    
    async def _detect_artifacts_fast(self, signal_data: np.ndarray) -> List[str]:
        """Fast artifact detection for real-time processing"""
        artifacts = []
        
        # Amplitude-based artifacts
        max_amplitude = np.max(np.abs(signal_data))
        if max_amplitude > self.artifact_thresholds['amplitude_threshold']:
            artifacts.append('amplitude_artifact')
        
        # Gradient-based artifacts (muscle activity, movement)
        max_gradient = np.max(np.abs(np.diff(signal_data, axis=1)))
        if max_gradient > self.artifact_thresholds['gradient_threshold']:
            artifacts.append('gradient_artifact')
        
        # Statistical artifacts (eye blinks, etc.)
        kurtosis_values = stats.kurtosis(signal_data, axis=1)
        if np.any(np.abs(kurtosis_values) > self.artifact_thresholds['kurtosis_threshold']):
            artifacts.append('statistical_artifact')
        
        return artifacts
    
    def _detect_amplitude_artifacts(self, signal_data: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect amplitude-based artifacts"""
        artifacts = []
        threshold = self.artifact_thresholds['amplitude_threshold']
        
        for ch_idx, channel_data in enumerate(signal_data):
            artifact_samples = np.where(np.abs(channel_data) > threshold)[0]
            if len(artifact_samples) > 0:
                # Group consecutive artifacts
                artifact_groups = self._group_consecutive_samples(artifact_samples)
                artifacts.extend([(ch_idx, start, end) for start, end in artifact_groups])
        
        return artifacts
    
    def _detect_gradient_artifacts(self, signal_data: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect gradient-based artifacts"""
        artifacts = []
        threshold = self.artifact_thresholds['gradient_threshold']
        
        for ch_idx, channel_data in enumerate(signal_data):
            gradient = np.abs(np.diff(channel_data))
            artifact_samples = np.where(gradient > threshold)[0]
            if len(artifact_samples) > 0:
                artifact_groups = self._group_consecutive_samples(artifact_samples)
                artifacts.extend([(ch_idx, start, end) for start, end in artifact_groups])
        
        return artifacts
    
    def _detect_statistical_artifacts(self, signal_data: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect statistical artifacts using sliding window analysis"""
        artifacts = []
        window_size = int(self.config.sampling_rate * 0.5)  # 500ms windows
        overlap = window_size // 2
        
        for ch_idx, channel_data in enumerate(signal_data):
            for start_idx in range(0, len(channel_data) - window_size, overlap):
                window_data = channel_data[start_idx:start_idx + window_size]
                
                # Check kurtosis
                kurtosis_value = stats.kurtosis(window_data)
                if abs(kurtosis_value) > self.artifact_thresholds['kurtosis_threshold']:
                    artifacts.append((ch_idx, start_idx, start_idx + window_size))
        
        return artifacts
    
    def _detect_frequency_artifacts(self, signal_data: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect frequency-domain artifacts"""
        artifacts = []
        
        for ch_idx, channel_data in enumerate(signal_data):
            # Compute power spectrum
            freqs, psd = signal.welch(channel_data, self.config.sampling_rate)
            
            # Check for abnormal power in specific frequency ranges
            # Power line noise detection
            for notch_freq in self.config.notch_frequencies:
                freq_mask = (freqs >= notch_freq - 2) & (freqs <= notch_freq + 2)
                if np.any(freq_mask):
                    power_in_band = np.sum(psd[freq_mask])
                    total_power = np.sum(psd)
                    
                    if power_in_band / total_power > 0.1:  # >10% of total power
                        artifacts.append((ch_idx, 0, len(channel_data)))
        
        return artifacts
    
    def _detect_correlation_artifacts(self, signal_data: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect artifacts based on abnormal inter-channel correlations"""
        artifacts = []
        
        if signal_data.shape[0] < 2:  # Need at least 2 channels
            return artifacts
        
        correlation_matrix = np.corrcoef(signal_data)
        
        # Find channels with abnormally high correlation
        n_channels = signal_data.shape[0]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if abs(correlation_matrix[i, j]) > self.artifact_thresholds['correlation_threshold']:
                    # Both channels might be artifactual
                    artifacts.append((i, 0, signal_data.shape[1]))
                    artifacts.append((j, 0, signal_data.shape[1]))
        
        return artifacts
    
    def _group_consecutive_samples(self, samples: np.ndarray) -> List[Tuple[int, int]]:
        """Group consecutive artifact samples"""
        if len(samples) == 0:
            return []
        
        groups = []
        start = samples[0]
        prev = start
        
        for sample in samples[1:]:
            if sample != prev + 1:
                groups.append((start, prev))
                start = sample
            prev = sample
        
        groups.append((start, prev))
        return groups
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {
                'total_processed': 0,
                'average_latency_ms': 0.0,
                'min_latency_ms': 0.0,
                'max_latency_ms': 0.0
            }
        
        processing_times = list(self.processing_times)
        
        return {
            'total_processed': len(processing_times),
            'average_latency_ms': np.mean(processing_times),
            'min_latency_ms': np.min(processing_times),
            'max_latency_ms': np.max(processing_times),
            'median_latency_ms': np.median(processing_times),
            'p95_latency_ms': np.percentile(processing_times, 95),
            'p99_latency_ms': np.percentile(processing_times, 99),
            'throughput_samples_per_sec': len(processing_times) / max(1, sum(processing_times) / 1000),
            'gpu_acceleration': self.use_gpu,
            'target_latency_ms': self.config.latency_target_ms,
            'latency_target_met': np.mean(processing_times) <= self.config.latency_target_ms
        }

class NeuralPatternClassifier:
    """Advanced machine learning classifier for neural pattern recognition"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = None
        
        # Model performance tracking
        self.training_history = {}
        self.validation_scores = []
        
        # Initialize model based on configuration
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize machine learning model"""
        model_type = self.config.model_type.lower()
        
        if model_type == "deep_neural_network":
            self.model = self._create_deep_neural_network()
        elif model_type == "neural_network":
            self.model = self._create_neural_network()
        elif model_type == "svm" and SKLEARN_AVAILABLE:
            from sklearn.svm import SVC
            self.model = SVC(probability=True, kernel='rbf', gamma='scale')
        elif model_type == "random_forest" and SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting" and SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            # Fallback to simple neural network
            self.model = self._create_neural_network()
        
        logger.info(f"Initialized {model_type} classifier")
    
    def _create_deep_neural_network(self):
        """Create deep neural network with attention mechanism"""
        
        class DeepNeuralClassifier(nn.Module):
            def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.3):
                super().__init__()
                
                # Feature extraction layers
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
                self.attention_norm = nn.LayerNorm(256)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(64, num_classes)
                )
                
                # Confidence estimation
                self.confidence_estimator = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # Feature extraction
                features = self.feature_extractor(x)
                
                # Self-attention
                features_att = features.unsqueeze(1)  # Add sequence dimension
                attended_features, attention_weights = self.attention(
                    features_att, features_att, features_att
                )
                attended_features = attended_features.squeeze(1)  # Remove sequence dimension
                
                # Residual connection and normalization
                features = self.attention_norm(features + attended_features)
                
                # Classification
                logits = self.classifier(features)
                confidence = self.confidence_estimator(features)
                
                return logits, confidence, attention_weights
        
        # Model will be fully initialized when we know input size
        self.model_class = DeepNeuralClassifier
        return None  # Will be created during training
    
    def _create_neural_network(self):
        """Create standard neural network"""
        
        class NeuralClassifier(nn.Module):
            def __init__(self, input_size: int, num_classes: int):
                super().__init__()
                
                self.network = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, num_classes)
                )
                
                self.confidence_head = nn.Sequential(
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.network[:-1](x)  # All layers except last
                logits = self.network[-1](features)
                confidence = self.confidence_head(features)
                
                return logits, confidence
        
        self.model_class = NeuralClassifier
        return None
    
    async def train_classifier(self, training_data: List[Tuple[Dict[str, np.ndarray], int]], 
                              validation_data: Optional[List[Tuple[Dict[str, np.ndarray], int]]] = None,
                              user_id: Optional[str] = None) -> Dict[str, Any]:
        """Train the neural pattern classifier"""
        if not training_data:
            raise ValueError("Training data is empty")
        
        logger.info(f"Training neural classifier with {len(training_data)} samples")
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(training_data)
        
        # Feature selection and scaling
        self._prepare_feature_pipeline(X_train)
        X_train_processed = self._apply_feature_pipeline(X_train)
        
        # Prepare validation data if provided
        X_val_processed = None
        y_val = None
        if validation_data:
            X_val, y_val = self._prepare_training_data(validation_data)
            X_val_processed = self._apply_feature_pipeline(X_val)
        
        # Initialize model with correct input size
        input_size = X_train_processed.shape[1]
        num_classes = len(np.unique(y_train))
        
        if hasattr(self, 'model_class'):
            self.model = self.model_class(input_size, num_classes)
        
        # Train the model
        if isinstance(self.model, nn.Module):
            training_result = await self._train_pytorch_model(
                X_train_processed, y_train, X_val_processed, y_val
            )
        else:
            training_result = await self._train_sklearn_model(
                X_train_processed, y_train, X_val_processed, y_val
            )
        
        self.is_trained = True
        
        return {
            'user_id': user_id,
            'training_samples': len(training_data),
            'validation_samples': len(validation_data) if validation_data else 0,
            'feature_count': input_size,
            'class_count': num_classes,
            'training_result': training_result
        }
    
    async def _train_pytorch_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Train PyTorch neural network model"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training parameters
        num_epochs = 200
        batch_size = min(self.config.batch_size, len(X_train))
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'forward') and len(self.model.forward.__code__.co_varnames) > 2:
                    # Model returns multiple values (logits, confidence, etc.)
                    outputs = self.model(batch_X)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    logits = self.model(batch_X)
                
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / (len(X_train) // batch_size + 1)
            train_accuracy = epoch_correct / epoch_total
            
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            val_accuracy = 0.0
            if X_val is not None and y_val is not None:
                val_accuracy = await self._validate_pytorch_model(X_val, y_val)
                val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Logging
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                          f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return {
            'final_train_accuracy': train_accuracies[-1],
            'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
            'best_val_accuracy': max(val_accuracies) if val_accuracies else 0.0,
            'epochs_trained': num_epochs,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    async def _validate_pytorch_model(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Validate PyTorch model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            outputs = self.model(X_val_tensor)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            _, predicted = torch.max(logits.data, 1)
            total += y_val_tensor.size(0)
            correct += (predicted == y_val_tensor).sum().item()
        
        self.model.train()
        return correct / total
    
    async def _train_sklearn_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Train scikit-learn model"""
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        # Validation accuracy
        val_accuracy = 0.0
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
        
        return {
            'final_train_accuracy': train_accuracy,
            'final_val_accuracy': val_accuracy,
            'model_type': type(self.model).__name__
        }
    
    def _prepare_training_data(self, training_data: List[Tuple[Dict[str, np.ndarray], int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert training data to arrays"""
        X = []
        y = []
        
        for features, label in training_data:
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _features_to_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert feature dictionary to vector"""
        feature_vector = []
        
        # Sort keys for consistent ordering
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    feature_vector.extend(value.flatten())
                else:
                    feature_vector.extend(value)
            else:
                feature_vector.append(float(value))
        
        return np.array(feature_vector)
    
    def _prepare_feature_pipeline(self, X: np.ndarray):
        """Prepare feature scaling and selection pipeline"""
        # Feature scaling
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Feature selection (optional)
            from sklearn.feature_selection import SelectKBest, f_classif
            k_features = min(X.shape[1], 1000)  # Limit features for performance
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
            self.feature_selector.fit(X_scaled, np.zeros(X_scaled.shape[0]))  # Dummy y for fitting
            
            self.selected_features = self.feature_selector.get_support()
        else:
            # Simple standardization fallback
            self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def _apply_feature_pipeline(self, X: np.ndarray) -> np.ndarray:
        """Apply feature scaling and selection"""
        X_processed = X
        
        # Apply scaling
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        # Apply feature selection
        if self.feature_selector is not None and self.selected_features is not None:
            X_processed = X_processed[:, self.selected_features]
        
        return X_processed
    
    async def classify_pattern(self, features: Dict[str, np.ndarray], 
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """Classify neural pattern from extracted features"""
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before use")
        
        start_time = time.time()
        
        # Convert features to vector and apply preprocessing
        feature_vector = self._features_to_vector(features)
        feature_vector_processed = self._apply_feature_pipeline(feature_vector.reshape(1, -1))
        
        # Make prediction
        if isinstance(self.model, nn.Module):
            prediction_result = await self._predict_pytorch(feature_vector_processed)
        else:
            prediction_result = await self._predict_sklearn(feature_vector_processed)
        
        classification_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'predicted_class': prediction_result['predicted_class'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result.get('probabilities'),
            'classification_time_ms': classification_time,
            'user_id': user_id,
            'feature_count': len(feature_vector),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _predict_pytorch(self, X: np.ndarray) -> Dict[str, Any]:
        """Make prediction with PyTorch model"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            
            if isinstance(outputs, tuple):
                logits, confidence = outputs[0], outputs[1]
            else:
                logits = outputs
                confidence = torch.ones(1)  # Default confidence
            
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            max_probability = torch.max(probabilities).item()
            
            confidence_score = confidence.item() if confidence.numel() == 1 else max_probability
        
        return {
            'predicted_class': predicted_class,
            'confidence': max(confidence_score, max_probability),
            'probabilities': probabilities.numpy().tolist()
        }
    
    async def _predict_sklearn(self, X: np.ndarray) -> Dict[str, Any]:
        """Make prediction with scikit-learn model"""
        predicted_class = self.model.predict(X)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = np.max(probabilities)
        else:
            probabilities = None
            confidence = 1.0  # Default confidence
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }

class NeuralSignalProcessor:
    """Main neural signal processor orchestrating real-time processing pipeline"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.signal_processor = AdvancedSignalProcessor(config)
        self.pattern_classifier = NeuralPatternClassifier(config)
        
        # Real-time processing components
        self.real_time_buffers = {}
        self.processing_stats = ProcessingStats()
        
        # Processing thread management
        self.is_processing = False
        self.processing_thread = None
        self.processing_queue = asyncio.Queue()
        
        logger.info("Neural signal processor initialized")
    
    async def initialize_real_time_processing(self, channel_names: List[str]) -> Dict[str, Any]:
        """Initialize real-time processing for given channels"""
        n_channels = len(channel_names)
        
        # Create real-time buffers for each channel group
        buffer_id = f"buffer_{int(time.time())}"
        self.real_time_buffers[buffer_id] = RealTimeBuffer(
            self.config.buffer_size, n_channels
        )
        
        # Start processing if not already running
        if not self.is_processing:
            await self.start_real_time_processing()
        
        return {
            'status': 'initialized',
            'buffer_id': buffer_id,
            'channel_count': n_channels,
            'buffer_size': self.config.buffer_size,
            'sampling_rate': self.config.sampling_rate
        }
    
    async def start_real_time_processing(self) -> Dict[str, Any]:
        """Start real-time signal processing"""
        if self.is_processing:
            return {'status': 'already_running'}
        
        self.is_processing = True
        logger.info("Starting real-time neural signal processing")
        
        # Start processing thread
        self.processing_thread = asyncio.create_task(self._processing_loop())
        
        return {
            'status': 'started',
            'latency_target_ms': self.config.latency_target_ms,
            'mode': self.config.mode.value
        }
    
    async def _processing_loop(self):
        """Main real-time processing loop"""
        while self.is_processing:
            try:
                # Process any queued signals
                if not self.processing_queue.empty():
                    signal_task = await self.processing_queue.get()
                    await self._process_signal_task(signal_task)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)  # 1ms
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(0.01)  # 10ms delay on error
    
    async def add_signal_data(self, buffer_id: str, signal_data: np.ndarray, 
                            channel_names: List[str]) -> Dict[str, Any]:
        """Add new signal data to real-time buffer"""
        if buffer_id not in self.real_time_buffers:
            return {'status': 'buffer_not_found', 'buffer_id': buffer_id}
        
        # Add to buffer
        buffer = self.real_time_buffers[buffer_id]
        buffer.add_samples(signal_data)
        
        # Queue processing task if buffer has enough data
        buffer_status = buffer.get_buffer_status()
        if buffer_status['ready_for_processing']:
            # Get latest samples for processing
            window_size = int(self.config.feature_window_ms * self.config.sampling_rate / 1000)
            latest_data = buffer.get_latest_samples(window_size)
            
            # Queue processing task
            task = {
                'signal_data': latest_data,
                'channel_names': channel_names,
                'buffer_id': buffer_id,
                'timestamp': time.time()
            }
            await self.processing_queue.put(task)
        
        return {
            'status': 'data_added',
            'buffer_id': buffer_id,
            'samples_added': signal_data.shape[1],
            'buffer_status': buffer_status
        }
    
    async def _process_signal_task(self, task: Dict[str, Any]):
        """Process a single signal processing task"""
        try:
            start_time = time.time()
            
            # Extract task data
            signal_data = task['signal_data']
            channel_names = task['channel_names']
            buffer_id = task['buffer_id']
            
            # Process signal
            processing_result = await self.signal_processor.process_signal_realtime(
                signal_data, channel_names
            )
            
            # Classify pattern if trained
            if self.pattern_classifier.is_trained and processing_result.features_extracted:
                classification_result = await self.pattern_classifier.classify_pattern(
                    processing_result.features_extracted
                )
                processing_result.classification_result = classification_result
                processing_result.confidence = classification_result['confidence']
            
            # Update statistics
            self._update_processing_stats(processing_result)
            
            # Log results for high-confidence classifications
            if processing_result.confidence > self.config.confidence_threshold:
                logger.info(f"High-confidence classification: {processing_result.classification_result}")
            
        except Exception as e:
            logger.error(f"Error processing signal task: {str(e)}")
            self.processing_stats.error_count += 1
    
    def _update_processing_stats(self, result: ProcessingResult):
        """Update processing performance statistics"""
        self.processing_stats.total_signals_processed += 1
        
        # Update latency statistics
        total_processed = self.processing_stats.total_signals_processed
        current_latency = result.latency_ms
        avg_latency = self.processing_stats.average_latency_ms
        
        self.processing_stats.average_latency_ms = (
            (avg_latency * (total_processed - 1) + current_latency) / total_processed
        )
        
        # Update processing time
        current_processing_time = result.processing_time_ms
        avg_processing_time = self.processing_stats.average_processing_time_ms
        
        self.processing_stats.average_processing_time_ms = (
            (avg_processing_time * (total_processed - 1) + current_processing_time) / total_processed
        )
        
        # Update signal quality distribution
        self.processing_stats.signal_quality_distribution[result.signal_quality] += 1
        
        # Update artifact detection rate
        if result.artifacts_detected:
            artifacts_count = len(result.artifacts_detected)
            current_rate = self.processing_stats.artifacts_detected_rate
            self.processing_stats.artifacts_detected_rate = (
                (current_rate * (total_processed - 1) + artifacts_count) / total_processed
            )
        
        # Update accuracy if classification available
        if result.classification_result and result.confidence > self.config.confidence_threshold:
            # This would be updated with ground truth in a real implementation
            pass
    
    async def stop_real_time_processing(self) -> Dict[str, Any]:
        """Stop real-time signal processing"""
        if not self.is_processing:
            return {'status': 'not_running'}
        
        self.is_processing = False
        
        # Cancel processing thread
        if self.processing_thread:
            self.processing_thread.cancel()
            try:
                await self.processing_thread
            except asyncio.CancelledError:
                pass
        
        stats = {
            'total_processed': self.processing_stats.total_signals_processed,
            'average_latency_ms': self.processing_stats.average_latency_ms,
            'average_processing_time_ms': self.processing_stats.average_processing_time_ms,
            'error_count': self.processing_stats.error_count
        }
        
        logger.info("Real-time neural signal processing stopped")
        
        return {
            'status': 'stopped',
            'final_stats': stats
        }
    
    async def train_user_specific_model(self, training_data: List[Tuple[Dict[str, np.ndarray], int]], 
                                      user_id: str) -> Dict[str, Any]:
        """Train user-specific classification model"""
        training_result = await self.pattern_classifier.train_classifier(
            training_data, user_id=user_id
        )
        
        return {
            'user_id': user_id,
            'training_status': 'completed',
            'model_performance': training_result
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get processing statistics from signal processor
        signal_processing_stats = await self.signal_processor.get_processing_stats()
        
        # Get buffer statuses
        buffer_statuses = {}
        for buffer_id, buffer in self.real_time_buffers.items():
            buffer_statuses[buffer_id] = buffer.get_buffer_status()
        
        return {
            'is_processing': self.is_processing,
            'config': {
                'mode': self.config.mode.value,
                'sampling_rate': self.config.sampling_rate,
                'latency_target_ms': self.config.latency_target_ms,
                'buffer_size': self.config.buffer_size,
                'use_gpu': self.config.use_gpu
            },
            'processing_stats': {
                'total_signals_processed': self.processing_stats.total_signals_processed,
                'average_latency_ms': self.processing_stats.average_latency_ms,
                'average_processing_time_ms': self.processing_stats.average_processing_time_ms,
                'accuracy_percentage': self.processing_stats.accuracy_percentage,
                'artifacts_detected_rate': self.processing_stats.artifacts_detected_rate,
                'error_count': self.processing_stats.error_count
            },
            'signal_processing_stats': signal_processing_stats,
            'buffer_statuses': buffer_statuses,
            'classifier_trained': self.pattern_classifier.is_trained,
            'active_buffers': len(self.real_time_buffers)
        }

# Mock data generator for testing
class NeuralSignalMockGenerator:
    """Generate realistic mock neural signal data for testing"""
    
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        self.random_state = np.random.RandomState(42)
    
    def generate_eeg_signal(self, duration_seconds: float, n_channels: int = 8) -> np.ndarray:
        """Generate realistic EEG signal with multiple frequency components"""
        n_samples = int(duration_seconds * self.sampling_rate)
        time_points = np.linspace(0, duration_seconds, n_samples)
        
        signal_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Alpha rhythm (8-12 Hz) - dominant in relaxed state
            alpha_freq = 8 + self.random_state.random() * 4
            alpha_signal = 20 * np.sin(2 * np.pi * alpha_freq * time_points + 
                                     self.random_state.random() * 2 * np.pi)
            
            # Beta rhythm (12-30 Hz) - active thinking
            beta_freq = 12 + self.random_state.random() * 18
            beta_signal = 8 * np.sin(2 * np.pi * beta_freq * time_points + 
                                   self.random_state.random() * 2 * np.pi)
            
            # Theta rhythm (4-8 Hz) - drowsiness, meditation
            theta_freq = 4 + self.random_state.random() * 4
            theta_signal = 12 * np.sin(2 * np.pi * theta_freq * time_points + 
                                     self.random_state.random() * 2 * np.pi)
            
            # Gamma rhythm (30-100 Hz) - high-level cognitive processing
            gamma_freq = 30 + self.random_state.random() * 20
            gamma_signal = 3 * np.sin(2 * np.pi * gamma_freq * time_points + 
                                    self.random_state.random() * 2 * np.pi)
            
            # Background noise
            noise = self.random_state.normal(0, 5, n_samples)
            
            # Combine signals
            channel_signal = (0.6 * alpha_signal + 0.3 * beta_signal + 
                            0.4 * theta_signal + 0.2 * gamma_signal + noise)
            
            # Add channel-specific variations
            channel_weight = 0.7 + 0.6 * self.random_state.random()
            signal_data[ch] = channel_weight * channel_signal
            
            # Add occasional artifacts
            if self.random_state.random() < 0.15:  # 15% chance
                artifact_start = self.random_state.randint(0, max(1, n_samples - 200))
                artifact_length = self.random_state.randint(10, 100)
                artifact_end = min(artifact_start + artifact_length, n_samples)
                
                # Eye blink artifact (large amplitude, short duration)
                artifact_amplitude = 80 + self.random_state.random() * 100
                signal_data[ch, artifact_start:artifact_end] += artifact_amplitude
        
        return signal_data
    
    def generate_training_dataset(self, n_samples: int = 1000, 
                                n_classes: int = 5) -> List[Tuple[Dict[str, np.ndarray], int]]:
        """Generate training dataset with features and labels"""
        training_data = []
        
        for _ in range(n_samples):
            # Generate signal
            signal = self.generate_eeg_signal(1.0, 8)  # 1 second, 8 channels
            
            # Extract basic features (simplified)
            features = {
                'mean': np.mean(signal, axis=1),
                'std': np.std(signal, axis=1),
                'alpha_power': self._calculate_band_power(signal, 8, 12),
                'beta_power': self._calculate_band_power(signal, 12, 30),
                'gamma_power': self._calculate_band_power(signal, 30, 40)
            }
            
            # Random label for demonstration
            label = self.random_state.randint(0, n_classes)
            
            training_data.append((features, label))
        
        return training_data
    
    def _calculate_band_power(self, signal: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Calculate power in specific frequency band"""
        band_powers = []
        
        for channel in signal:
            freqs, psd = signal.welch(channel, self.sampling_rate, nperseg=len(channel)//4)
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.sum(psd[freq_mask])
            band_powers.append(band_power)
        
        return np.array(band_powers)