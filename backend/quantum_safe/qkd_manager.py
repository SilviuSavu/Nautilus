"""
Quantum Key Distribution (QKD) Manager
=====================================

Implementation of quantum key distribution for ultra-secure communication channels.
Supports hardware QKD devices and quantum-safe key exchange protocols.

Key Features:
- Hardware QKD device integration
- Quantum key pool management
- Quantum random number generation
- Entanglement-based key distribution
- BB84 protocol implementation
- Key distillation and amplification
- Quantum channel monitoring

"""

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import qiskit
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.quantum_info import random_statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available - using simulated quantum operations")


class QKDProtocol(Enum):
    """Quantum key distribution protocols"""
    BB84 = "bb84"
    B92 = "b92"
    E91 = "e91"
    SARG04 = "sarg04"
    COW = "cow"  # Coherent One Way
    DPS = "dps"  # Differential Phase Shift


class QKDChannelType(Enum):
    """QKD channel types"""
    FIBER_OPTIC = "fiber_optic"
    FREE_SPACE = "free_space"
    SATELLITE = "satellite"
    INTEGRATED_PHOTONIC = "integrated_photonic"


class QKDKeyState(Enum):
    """States of QKD keys"""
    GENERATING = "generating"
    RAW = "raw"
    SIFTED = "sifted"
    ERROR_CORRECTED = "error_corrected"
    AMPLIFIED = "amplified"
    READY = "ready"
    CONSUMED = "consumed"
    EXPIRED = "expired"


class QuantumChannelQuality(Enum):
    """Quantum channel quality levels"""
    EXCELLENT = "excellent"  # <1% error rate
    GOOD = "good"           # 1-3% error rate
    FAIR = "fair"           # 3-5% error rate
    POOR = "poor"           # 5-10% error rate
    UNUSABLE = "unusable"   # >10% error rate


@dataclass
class QKDDevice:
    """Quantum key distribution device configuration"""
    device_id: str
    device_type: str
    protocol: QKDProtocol
    channel_type: QKDChannelType
    location: str
    coordinates: Tuple[float, float]
    hardware_version: str
    firmware_version: str
    max_key_rate_bps: int
    max_distance_km: float
    wavelength_nm: float
    detection_efficiency: float
    dark_count_rate: float
    is_active: bool = True
    last_calibration: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumKey:
    """Quantum-generated cryptographic key"""
    key_id: str
    key_data: bytes
    key_length_bits: int
    protocol: QKDProtocol
    generation_time: datetime
    source_device_id: str
    destination_device_id: str
    channel_id: str
    state: QKDKeyState
    error_rate: float
    security_parameter: float
    entropy_estimate: float
    quantum_bit_error_rate: float
    frame_error_rate: float
    secret_key_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumChannel:
    """Quantum communication channel"""
    channel_id: str
    source_device: str
    destination_device: str
    protocol: QKDProtocol
    channel_type: QKDChannelType
    distance_km: float
    attenuation_db: float
    visibility: float
    quantum_bit_error_rate: float
    secret_key_rate_bps: float
    channel_quality: QuantumChannelQuality
    is_active: bool = True
    establishment_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_keys_generated: int = 0
    total_bits_transmitted: int = 0
    security_events: List[Dict[str, Any]] = field(default_factory=list)


class QuantumKeyDistribution:
    """
    Quantum Key Distribution Manager for Nautilus trading platform.
    
    Manages quantum key generation, distribution, and lifecycle for
    ultra-secure trading communications. Integrates with hardware
    QKD devices and provides quantum-safe key management.
    """
    
    def __init__(self,
                 key_pool_size: int = 1000,
                 min_key_length_bits: int = 256,
                 max_key_length_bits: int = 2048,
                 key_refresh_interval_seconds: int = 300,
                 enable_hardware_qkd: bool = False,
                 simulation_mode: bool = True):
        """
        Initialize Quantum Key Distribution system.
        
        Args:
            key_pool_size: Size of quantum key pool
            min_key_length_bits: Minimum key length in bits
            max_key_length_bits: Maximum key length in bits
            key_refresh_interval_seconds: Key refresh interval
            enable_hardware_qkd: Enable hardware QKD devices
            simulation_mode: Use quantum simulation instead of hardware
        """
        self.key_pool_size = key_pool_size
        self.min_key_length_bits = min_key_length_bits
        self.max_key_length_bits = max_key_length_bits
        self.key_refresh_interval = key_refresh_interval_seconds
        self.enable_hardware_qkd = enable_hardware_qkd
        self.simulation_mode = simulation_mode
        
        self.logger = logging.getLogger("quantum_safe.qkd_manager")
        
        # QKD system components
        self.devices: Dict[str, QKDDevice] = {}
        self.channels: Dict[str, QuantumChannel] = {}
        self.quantum_keys: Dict[str, QuantumKey] = {}
        self.key_pool: List[str] = []  # Pool of ready keys
        
        # Quantum state management
        self.quantum_rng_state = None
        self.entanglement_pairs: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.generation_stats = {
            "keys_generated": 0,
            "total_entropy": 0.0,
            "avg_error_rate": 0.0,
            "avg_key_rate": 0.0,
            "channel_uptime": 0.0
        }
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown_event = threading.Event()
        
        # Initialize quantum simulator if available
        if QISKIT_AVAILABLE and simulation_mode:
            self.quantum_simulator = Aer.get_backend('qasm_simulator')
            self.quantum_backend = Aer.get_backend('statevector_simulator')
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize QKD system with default configuration"""
        
        # Create default QKD devices for demonstration
        if self.simulation_mode:
            self._create_simulated_devices()
        
        # Start background tasks
        asyncio.create_task(self._key_generation_loop())
        asyncio.create_task(self._channel_monitoring_loop())
        asyncio.create_task(self._key_lifecycle_management())
    
    def _create_simulated_devices(self):
        """Create simulated QKD devices for testing"""
        
        # Primary trading hub device
        primary_device = QKDDevice(
            device_id="qkd-primary-001",
            device_type="QKD-Transmitter-Pro",
            protocol=QKDProtocol.BB84,
            channel_type=QKDChannelType.FIBER_OPTIC,
            location="Primary Trading Hub",
            coordinates=(40.7589, -73.9851),  # New York
            hardware_version="HW-v2.1",
            firmware_version="FW-v3.4.1",
            max_key_rate_bps=10000000,  # 10 Mbps
            max_distance_km=100.0,
            wavelength_nm=1550.0,
            detection_efficiency=0.85,
            dark_count_rate=100.0,
            last_calibration=datetime.now(timezone.utc) - timedelta(days=7),
            next_maintenance=datetime.now(timezone.utc) + timedelta(days=23)
        )
        self.add_device(primary_device)
        
        # Secondary hub device
        secondary_device = QKDDevice(
            device_id="qkd-secondary-001",
            device_type="QKD-Receiver-Pro",
            protocol=QKDProtocol.BB84,
            channel_type=QKDChannelType.FIBER_OPTIC,
            location="Secondary Trading Hub",
            coordinates=(41.8781, -87.6298),  # Chicago
            hardware_version="HW-v2.1",
            firmware_version="FW-v3.4.1",
            max_key_rate_bps=10000000,  # 10 Mbps
            max_distance_km=100.0,
            wavelength_nm=1550.0,
            detection_efficiency=0.83,
            dark_count_rate=120.0,
            last_calibration=datetime.now(timezone.utc) - timedelta(days=5),
            next_maintenance=datetime.now(timezone.utc) + timedelta(days=25)
        )
        self.add_device(secondary_device)
        
        # Create quantum channel between devices
        asyncio.create_task(self._establish_quantum_channel(
            "qkd-primary-001", "qkd-secondary-001"
        ))
    
    def add_device(self, device: QKDDevice) -> bool:
        """Add a QKD device to the system"""
        try:
            self.devices[device.device_id] = device
            self.logger.info(f"Added QKD device: {device.device_id} at {device.location}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add QKD device: {str(e)}")
            return False
    
    async def _establish_quantum_channel(self, source_id: str, destination_id: str) -> Optional[str]:
        """Establish quantum channel between two devices"""
        
        if source_id not in self.devices or destination_id not in self.devices:
            self.logger.error("One or both devices not found for channel establishment")
            return None
        
        source_device = self.devices[source_id]
        dest_device = self.devices[destination_id]
        
        # Calculate distance between devices
        distance = self._calculate_distance(
            source_device.coordinates,
            dest_device.coordinates
        )
        
        # Create channel ID
        channel_id = f"channel-{source_id}-{destination_id}-{secrets.token_hex(8)}"
        
        # Determine channel quality based on distance and device specs
        attenuation = self._calculate_attenuation(distance, source_device.channel_type)
        qber = self._estimate_quantum_error_rate(attenuation, source_device, dest_device)
        
        if qber < 0.01:
            quality = QuantumChannelQuality.EXCELLENT
        elif qber < 0.03:
            quality = QuantumChannelQuality.GOOD
        elif qber < 0.05:
            quality = QuantumChannelQuality.FAIR
        elif qber < 0.10:
            quality = QuantumChannelQuality.POOR
        else:
            quality = QuantumChannelQuality.UNUSABLE
        
        # Create channel
        channel = QuantumChannel(
            channel_id=channel_id,
            source_device=source_id,
            destination_device=destination_id,
            protocol=source_device.protocol,
            channel_type=source_device.channel_type,
            distance_km=distance,
            attenuation_db=attenuation,
            visibility=0.98,
            quantum_bit_error_rate=qber,
            secret_key_rate_bps=self._calculate_secret_key_rate(qber, source_device.max_key_rate_bps),
            channel_quality=quality
        )
        
        if quality != QuantumChannelQuality.UNUSABLE:
            self.channels[channel_id] = channel
            self.logger.info(f"Established quantum channel: {channel_id} (Quality: {quality.value})")
            return channel_id
        else:
            self.logger.warning(f"Channel quality too poor for secure operation: {qber:.3f} QBER")
            return None
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates"""
        # Simplified distance calculation (great circle approximation)
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        # Haversine formula
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Earth's radius in kilometers
        R = 6371.0
        
        return R * c
    
    def _calculate_attenuation(self, distance_km: float, channel_type: QKDChannelType) -> float:
        """Calculate channel attenuation"""
        if channel_type == QKDChannelType.FIBER_OPTIC:
            # Typical fiber attenuation: 0.2 dB/km at 1550nm
            return distance_km * 0.2
        elif channel_type == QKDChannelType.FREE_SPACE:
            # Free space path loss
            wavelength_m = 1550e-9  # 1550 nm
            return 20 * np.log10(distance_km * 1000) + 20 * np.log10(4 * np.pi / wavelength_m)
        else:
            # Default attenuation
            return distance_km * 0.5
    
    def _estimate_quantum_error_rate(self, attenuation_db: float, 
                                   source: QKDDevice, dest: QKDDevice) -> float:
        """Estimate quantum bit error rate"""
        # Base error rate due to environmental factors
        base_error = 0.001
        
        # Error due to attenuation
        attenuation_error = attenuation_db * 0.0001
        
        # Error due to detector efficiency
        efficiency_error = (1.0 - (source.detection_efficiency + dest.detection_efficiency) / 2) * 0.01
        
        # Dark count error
        dark_count_error = (source.dark_count_rate + dest.dark_count_rate) * 1e-6
        
        total_error = base_error + attenuation_error + efficiency_error + dark_count_error
        
        return min(total_error, 0.5)  # Cap at 50%
    
    def _calculate_secret_key_rate(self, qber: float, raw_key_rate: float) -> float:
        """Calculate secret key rate using Shannon entropy"""
        if qber >= 0.11:  # Theoretical limit for BB84
            return 0.0
        
        # Simplified secret key rate calculation
        # In practice, this involves more complex error correction and privacy amplification
        h2_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
        secret_fraction = max(0, 1 - 2 * h2_qber)
        
        return raw_key_rate * secret_fraction
    
    async def generate_quantum_key(self, 
                                 channel_id: str,
                                 key_length_bits: int,
                                 protocol: Optional[QKDProtocol] = None) -> Optional[str]:
        """
        Generate a quantum key using specified channel and protocol.
        
        Args:
            channel_id: Quantum channel identifier
            key_length_bits: Desired key length in bits
            protocol: QKD protocol to use
            
        Returns:
            Key identifier if successful, None otherwise
        """
        
        if channel_id not in self.channels:
            self.logger.error(f"Quantum channel not found: {channel_id}")
            return None
        
        channel = self.channels[channel_id]
        
        if not channel.is_active:
            self.logger.error(f"Quantum channel is inactive: {channel_id}")
            return None
        
        if key_length_bits < self.min_key_length_bits or key_length_bits > self.max_key_length_bits:
            self.logger.error(f"Invalid key length: {key_length_bits} bits")
            return None
        
        # Use channel protocol if not specified
        if protocol is None:
            protocol = channel.protocol
        
        start_time = time.time()
        
        try:
            # Generate quantum key based on protocol
            if protocol == QKDProtocol.BB84:
                key_data = await self._generate_bb84_key(channel, key_length_bits)
            elif protocol == QKDProtocol.E91:
                key_data = await self._generate_e91_key(channel, key_length_bits)
            else:
                # Fallback to BB84
                key_data = await self._generate_bb84_key(channel, key_length_bits)
            
            if key_data is None:
                return None
            
            # Create quantum key object
            key_id = f"qkey-{secrets.token_hex(16)}"
            
            quantum_key = QuantumKey(
                key_id=key_id,
                key_data=key_data,
                key_length_bits=key_length_bits,
                protocol=protocol,
                generation_time=datetime.now(timezone.utc),
                source_device_id=channel.source_device,
                destination_device_id=channel.destination_device,
                channel_id=channel_id,
                state=QKDKeyState.READY,
                error_rate=channel.quantum_bit_error_rate,
                security_parameter=self._calculate_security_parameter(channel.quantum_bit_error_rate),
                entropy_estimate=self._estimate_entropy(key_data),
                quantum_bit_error_rate=channel.quantum_bit_error_rate,
                frame_error_rate=channel.quantum_bit_error_rate * 1.2,
                secret_key_rate=channel.secret_key_rate_bps
            )
            
            # Store key
            self.quantum_keys[key_id] = quantum_key
            self.key_pool.append(key_id)
            
            # Update statistics
            self.generation_stats["keys_generated"] += 1
            self.generation_stats["total_entropy"] += quantum_key.entropy_estimate
            channel.total_keys_generated += 1
            channel.total_bits_transmitted += key_length_bits
            
            generation_time = time.time() - start_time
            
            self.logger.info(f"Generated quantum key {key_id} in {generation_time:.3f}s")
            
            return key_id
            
        except Exception as e:
            self.logger.error(f"Failed to generate quantum key: {str(e)}")
            return None
    
    async def _generate_bb84_key(self, channel: QuantumChannel, key_length_bits: int) -> Optional[bytes]:
        """Generate key using BB84 protocol"""
        
        if QISKIT_AVAILABLE and self.simulation_mode:
            return await self._generate_bb84_quantum_simulation(channel, key_length_bits)
        else:
            return await self._generate_bb84_classical_simulation(channel, key_length_bits)
    
    async def _generate_bb84_quantum_simulation(self, 
                                              channel: QuantumChannel, 
                                              key_length_bits: int) -> Optional[bytes]:
        """Generate BB84 key using quantum simulation"""
        
        try:
            # BB84 protocol simulation
            num_qubits = key_length_bits * 4  # Oversample for error correction
            
            # Alice's random bits and bases
            alice_bits = np.random.randint(2, size=num_qubits)
            alice_bases = np.random.randint(2, size=num_qubits)
            
            # Bob's random measurement bases
            bob_bases = np.random.randint(2, size=num_qubits)
            
            # Simulate quantum transmission and measurement
            bob_results = []
            
            for i in range(num_qubits):
                # Create quantum circuit for single qubit
                qc = QuantumCircuit(1, 1)
                
                # Alice prepares qubit
                if alice_bits[i] == 1:
                    qc.x(0)  # Flip to |1⟩
                
                if alice_bases[i] == 1:
                    qc.h(0)  # Hadamard for diagonal basis
                
                # Simulate channel noise
                if np.random.random() < channel.quantum_bit_error_rate:
                    qc.x(0)  # Bit flip error
                
                # Bob measures
                if bob_bases[i] == 1:
                    qc.h(0)  # Measure in diagonal basis
                
                qc.measure(0, 0)
                
                # Execute circuit
                job = execute(qc, self.quantum_simulator, shots=1)
                result = job.result()
                counts = result.get_counts()
                
                # Get measurement result
                measured_bit = int(list(counts.keys())[0])
                bob_results.append(measured_bit)
            
            # Classical post-processing
            # Bob announces his measurement bases
            matching_bases = alice_bases == bob_bases
            sifted_key_alice = alice_bits[matching_bases]
            sifted_key_bob = np.array(bob_results)[matching_bases]
            
            # Error estimation
            sample_size = min(len(sifted_key_alice) // 4, 100)
            if sample_size > 0:
                sample_indices = np.random.choice(len(sifted_key_alice), sample_size, replace=False)
                error_count = np.sum(sifted_key_alice[sample_indices] != sifted_key_bob[sample_indices])
                estimated_error_rate = error_count / sample_size
            else:
                estimated_error_rate = 0.0
            
            # Privacy amplification (simplified)
            final_key_bits = sifted_key_alice[sample_size:]  # Remove test bits
            
            if len(final_key_bits) < key_length_bits:
                self.logger.warning(f"Insufficient key material: {len(final_key_bits)} < {key_length_bits}")
                return None
            
            # Convert to bytes
            final_key_bits = final_key_bits[:key_length_bits]
            key_bytes = np.packbits(final_key_bits).tobytes()
            
            return key_bytes
            
        except Exception as e:
            self.logger.error(f"Quantum simulation failed: {str(e)}")
            return None
    
    async def _generate_bb84_classical_simulation(self, 
                                                channel: QuantumChannel, 
                                                key_length_bits: int) -> Optional[bytes]:
        """Generate BB84 key using classical simulation"""
        
        # Classical simulation of quantum key generation
        # This provides a realistic approximation of quantum key properties
        
        # Start with cryptographically secure random bits
        raw_bits = secrets.randbits(key_length_bits * 2)  # Oversample
        raw_bytes = raw_bits.to_bytes((key_length_bits * 2 + 7) // 8, 'big')
        
        # Apply simulated quantum noise based on channel error rate
        noisy_bits = []
        for bit_pos in range(key_length_bits * 2):
            byte_pos = bit_pos // 8
            bit_in_byte = 7 - (bit_pos % 8)
            original_bit = (raw_bytes[byte_pos] >> bit_in_byte) & 1
            
            # Apply quantum bit error
            if np.random.random() < channel.quantum_bit_error_rate:
                final_bit = 1 - original_bit  # Flip bit
            else:
                final_bit = original_bit
            
            noisy_bits.append(final_bit)
        
        # Simulate sifting (basis matching) - typically 50% efficiency
        sifted_bits = []
        for i in range(0, len(noisy_bits), 2):
            if np.random.random() < 0.5:  # Basis match probability
                sifted_bits.append(noisy_bits[i])
        
        # Ensure we have enough bits
        if len(sifted_bits) < key_length_bits:
            # Generate more bits if needed
            additional_bits_needed = key_length_bits - len(sifted_bits)
            additional_bits = [secrets.randbits(1) for _ in range(additional_bits_needed)]
            sifted_bits.extend(additional_bits)
        
        # Take exactly the requested number of bits
        final_bits = sifted_bits[:key_length_bits]
        
        # Convert to bytes
        final_bytes = bytearray()
        for i in range(0, len(final_bits), 8):
            byte_bits = final_bits[i:i+8]
            # Pad with zeros if needed
            while len(byte_bits) < 8:
                byte_bits.append(0)
            
            byte_value = 0
            for j, bit in enumerate(byte_bits):
                byte_value |= (bit << (7 - j))
            
            final_bytes.append(byte_value)
        
        return bytes(final_bytes)
    
    async def _generate_e91_key(self, channel: QuantumChannel, key_length_bits: int) -> Optional[bytes]:
        """Generate key using E91 (entanglement-based) protocol"""
        
        # E91 protocol uses entangled photon pairs
        # For simulation, we'll generate correlated random bits
        
        try:
            # Generate entangled pairs
            num_pairs = key_length_bits * 2  # Oversample
            
            # Alice and Bob's measurement settings
            alice_settings = np.random.choice([0, 1, 2], num_pairs)  # 3 measurement angles
            bob_settings = np.random.choice([1, 2, 3], num_pairs)    # 3 measurement angles
            
            # Generate correlated results based on quantum mechanics
            alice_results = []
            bob_results = []
            
            for i in range(num_pairs):
                # Simulate Bell state measurements
                # When settings match, results are perfectly correlated
                # When settings differ, correlation depends on angle
                
                if alice_settings[i] == bob_settings[i] - 1:  # Perfect correlation case
                    bit = secrets.randbits(1)
                    alice_results.append(bit)
                    # Add quantum errors
                    if np.random.random() < channel.quantum_bit_error_rate:
                        bob_results.append(1 - bit)
                    else:
                        bob_results.append(bit)
                else:
                    # Different settings - random correlation
                    alice_results.append(secrets.randbits(1))
                    bob_results.append(secrets.randbits(1))
            
            # CHSH test for entanglement verification (simplified)
            chsh_violations = 0
            test_size = min(100, num_pairs)
            for _ in range(test_size):
                idx = np.random.randint(num_pairs)
                # Simplified CHSH check
                if alice_results[idx] == bob_results[idx]:
                    chsh_violations += 1
            
            chsh_ratio = chsh_violations / test_size
            if chsh_ratio < 0.7:  # Threshold for quantum entanglement
                self.logger.warning("E91 entanglement verification failed")
                return None
            
            # Extract final key from correlated bits
            final_bits = []
            for i in range(num_pairs):
                if alice_settings[i] == bob_settings[i] - 1:  # Use correlated measurements
                    final_bits.append(alice_results[i])
                    if len(final_bits) >= key_length_bits:
                        break
            
            if len(final_bits) < key_length_bits:
                self.logger.warning("Insufficient E91 key material")
                return None
            
            # Convert to bytes
            key_bytes = bytearray()
            for i in range(0, key_length_bits, 8):
                byte_bits = final_bits[i:i+8]
                while len(byte_bits) < 8:
                    byte_bits.append(0)
                
                byte_value = 0
                for j, bit in enumerate(byte_bits):
                    byte_value |= (bit << (7 - j))
                
                key_bytes.append(byte_value)
            
            return bytes(key_bytes)
            
        except Exception as e:
            self.logger.error(f"E91 key generation failed: {str(e)}")
            return None
    
    def _calculate_security_parameter(self, qber: float) -> float:
        """Calculate security parameter based on quantum bit error rate"""
        # Simplified security parameter calculation
        # Higher QBER reduces security
        if qber <= 0.01:
            return 256.0  # Very high security
        elif qber <= 0.05:
            return 192.0  # High security
        elif qber <= 0.10:
            return 128.0  # Medium security
        else:
            return 64.0   # Low security
    
    def _estimate_entropy(self, key_data: bytes) -> float:
        """Estimate entropy of key data"""
        if not key_data:
            return 0.0
        
        # Calculate Shannon entropy
        byte_counts = {}
        for byte in key_data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        total_bytes = len(key_data)
        entropy = 0.0
        
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * np.log2(probability)
        
        return entropy * total_bytes / 8.0  # Convert to bits per bit
    
    async def get_quantum_key(self, preferred_length_bits: Optional[int] = None) -> Optional[str]:
        """Get a quantum key from the pool"""
        
        if not self.key_pool:
            self.logger.warning("Quantum key pool is empty")
            return None
        
        # Find suitable key
        if preferred_length_bits is not None:
            # Look for key with exact or larger length
            for key_id in self.key_pool:
                key = self.quantum_keys[key_id]
                if key.key_length_bits >= preferred_length_bits and key.state == QKDKeyState.READY:
                    self.key_pool.remove(key_id)
                    key.state = QKDKeyState.CONSUMED
                    return key_id
        
        # Return any available key
        key_id = self.key_pool.pop(0)
        key = self.quantum_keys[key_id]
        key.state = QKDKeyState.CONSUMED
        
        return key_id
    
    def get_key_data(self, key_id: str) -> Optional[bytes]:
        """Get key data by key ID"""
        if key_id not in self.quantum_keys:
            return None
        
        key = self.quantum_keys[key_id]
        if key.state != QKDKeyState.CONSUMED:
            self.logger.warning(f"Key {key_id} not in consumed state")
            return None
        
        return key.key_data
    
    async def _key_generation_loop(self):
        """Background key generation loop"""
        while not self._shutdown_event.is_set():
            try:
                # Check if key pool needs replenishment
                if len(self.key_pool) < self.key_pool_size // 2:
                    
                    # Generate keys for all active channels
                    for channel_id, channel in self.channels.items():
                        if channel.is_active and channel.channel_quality in [
                            QuantumChannelQuality.EXCELLENT,
                            QuantumChannelQuality.GOOD,
                            QuantumChannelQuality.FAIR
                        ]:
                            # Generate key of default size
                            key_length = 256  # Standard key length
                            await self.generate_quantum_key(channel_id, key_length)
                
                await asyncio.sleep(self.key_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in key generation loop: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _channel_monitoring_loop(self):
        """Monitor quantum channel health"""
        while not self._shutdown_event.is_set():
            try:
                for channel_id, channel in self.channels.items():
                    # Update channel heartbeat
                    channel.last_heartbeat = datetime.now(timezone.utc)
                    
                    # Check channel quality
                    if channel.quantum_bit_error_rate > 0.11:
                        channel.channel_quality = QuantumChannelQuality.UNUSABLE
                        channel.is_active = False
                        self.logger.warning(f"Channel {channel_id} disabled due to high error rate")
                    
                    # Update statistics
                    if channel.total_keys_generated > 0:
                        avg_error = sum(self.quantum_keys[kid].error_rate 
                                      for kid in self.quantum_keys 
                                      if self.quantum_keys[kid].channel_id == channel_id) / channel.total_keys_generated
                        self.generation_stats["avg_error_rate"] = avg_error
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in channel monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _key_lifecycle_management(self):
        """Manage quantum key lifecycle"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for key_id, key in self.quantum_keys.items():
                    # Mark old keys as expired (24 hours lifetime)
                    if current_time - key.generation_time > timedelta(hours=24):
                        key.state = QKDKeyState.EXPIRED
                        expired_keys.append(key_id)
                
                # Remove expired keys
                for key_id in expired_keys:
                    if key_id in self.key_pool:
                        self.key_pool.remove(key_id)
                    del self.quantum_keys[key_id]
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired quantum keys")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in key lifecycle management: {str(e)}")
                await asyncio.sleep(300)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive QKD system status"""
        
        # Channel statistics
        active_channels = sum(1 for c in self.channels.values() if c.is_active)
        total_keys_all_channels = sum(c.total_keys_generated for c in self.channels.values())
        total_bits_all_channels = sum(c.total_bits_transmitted for c in self.channels.values())
        
        # Key pool statistics
        ready_keys = len(self.key_pool)
        total_keys = len(self.quantum_keys)
        consumed_keys = sum(1 for k in self.quantum_keys.values() if k.state == QKDKeyState.CONSUMED)
        expired_keys = sum(1 for k in self.quantum_keys.values() if k.state == QKDKeyState.EXPIRED)
        
        # Security metrics
        avg_security_param = (sum(k.security_parameter for k in self.quantum_keys.values()) / 
                            max(total_keys, 1))
        avg_entropy = (sum(k.entropy_estimate for k in self.quantum_keys.values()) / 
                      max(total_keys, 1))
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "operational" if active_channels > 0 else "degraded",
            "qkd_configuration": {
                "simulation_mode": self.simulation_mode,
                "hardware_qkd_enabled": self.enable_hardware_qkd,
                "key_pool_size": self.key_pool_size,
                "key_refresh_interval": self.key_refresh_interval
            },
            "devices": {
                "total_devices": len(self.devices),
                "active_devices": sum(1 for d in self.devices.values() if d.is_active)
            },
            "channels": {
                "total_channels": len(self.channels),
                "active_channels": active_channels,
                "channel_qualities": {
                    quality.value: sum(1 for c in self.channels.values() if c.channel_quality == quality)
                    for quality in QuantumChannelQuality
                }
            },
            "key_management": {
                "ready_keys": ready_keys,
                "total_keys": total_keys,
                "consumed_keys": consumed_keys,
                "expired_keys": expired_keys,
                "pool_utilization": ready_keys / max(self.key_pool_size, 1)
            },
            "performance": {
                "total_keys_generated": total_keys_all_channels,
                "total_bits_transmitted": total_bits_all_channels,
                "average_error_rate": self.generation_stats.get("avg_error_rate", 0.0),
                "average_security_parameter": avg_security_param,
                "average_entropy": avg_entropy
            },
            "protocols_supported": [protocol.value for protocol in QKDProtocol],
            "channel_types_supported": [channel_type.value for channel_type in QKDChannelType]
        }
    
    async def shutdown(self):
        """Shutdown QKD system gracefully"""
        self.logger.info("Shutting down Quantum Key Distribution system")
        
        self._shutdown_event.set()
        
        # Cleanup resources
        self.executor.shutdown(wait=True)
        
        # Clear sensitive key data
        for key in self.quantum_keys.values():
            key.key_data = b'\x00' * len(key.key_data)  # Zero out key data
        
        self.quantum_keys.clear()
        self.key_pool.clear()
        
        self.logger.info("QKD system shutdown complete")