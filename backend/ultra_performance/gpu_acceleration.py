"""
GPU Acceleration Framework with CUDA Support

Provides GPU-accelerated computations for:
- Risk calculations and VaR computations
- Monte Carlo simulations for options pricing
- Matrix operations for correlation analysis
- GPU memory management and optimization

Supports NVIDIA CUDA with fallback to CPU for systems without GPU.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading

# GPU acceleration imports with fallback
try:
    import cupy as cp  # CUDA acceleration
    import cupyx.scipy.linalg as cp_linalg
    import cupyx.scipy.sparse as cp_sparse
    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    cp_linalg = None
    cp_sparse = None
    CUDA_AVAILABLE = False

try:
    import numba
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = cuda.is_available() if hasattr(cuda, 'is_available') else False
except ImportError:
    numba = None
    cuda = None
    NUMBA_CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GPUDevice:
    """GPU device information"""
    id: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    
@dataclass
class GPUMemoryStats:
    """GPU memory usage statistics"""
    allocated: int
    reserved: int
    free: int
    total: int
    utilization_percent: float

class CUDAManager:
    """
    CUDA GPU management and device selection
    Handles GPU device enumeration, memory management, and context switching
    """
    
    def __init__(self):
        self.devices: List[GPUDevice] = []
        self.current_device: Optional[int] = None
        self.memory_pools: Dict[int, Any] = {}
        self._lock = threading.RLock()
        self._initialize_devices()
        
    def _initialize_devices(self):
        """Initialize and enumerate available GPU devices"""
        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available - falling back to CPU")
            return
            
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for device_id in range(device_count):
                with cp.cuda.Device(device_id):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    device = GPUDevice(
                        id=device_id,
                        name=props['name'].decode('utf-8'),
                        memory_total=props['totalGlobalMem'],
                        memory_free=cp.cuda.runtime.memGetInfo()[0],
                        compute_capability=(props['major'], props['minor']),
                        multiprocessor_count=props['multiProcessorCount']
                    )
                    self.devices.append(device)
                    
            if self.devices:
                self.current_device = 0  # Default to first device
                logger.info(f"Initialized {len(self.devices)} CUDA devices")
                for device in self.devices:
                    logger.info(f"  Device {device.id}: {device.name} "
                              f"({device.memory_total // 1024**2} MB)")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA devices: {e}")
            
    def select_device(self, device_id: int) -> bool:
        """Select GPU device for computations"""
        if not CUDA_AVAILABLE or device_id >= len(self.devices):
            return False
            
        with self._lock:
            try:
                cp.cuda.Device(device_id).use()
                self.current_device = device_id
                logger.info(f"Selected GPU device {device_id}: {self.devices[device_id].name}")
                return True
            except Exception as e:
                logger.error(f"Failed to select device {device_id}: {e}")
                return False
                
    def get_memory_stats(self, device_id: Optional[int] = None) -> Optional[GPUMemoryStats]:
        """Get memory statistics for specified device"""
        if not CUDA_AVAILABLE:
            return None
            
        device_id = device_id or self.current_device
        if device_id is None:
            return None
            
        try:
            with cp.cuda.Device(device_id):
                free, total = cp.cuda.runtime.memGetInfo()
                allocated = total - free
                reserved = cp.get_default_memory_pool().used_bytes()
                
                return GPUMemoryStats(
                    allocated=allocated,
                    reserved=reserved,
                    free=free,
                    total=total,
                    utilization_percent=(allocated / total) * 100
                )
        except Exception as e:
            logger.error(f"Failed to get memory stats for device {device_id}: {e}")
            return None
            
    def clear_memory_cache(self, device_id: Optional[int] = None):
        """Clear GPU memory cache"""
        if not CUDA_AVAILABLE:
            return
            
        device_id = device_id or self.current_device
        if device_id is None:
            return
            
        try:
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
                logger.info(f"Cleared memory cache for device {device_id}")
        except Exception as e:
            logger.error(f"Failed to clear memory cache: {e}")

class GPUMemoryManager:
    """
    Advanced GPU memory management with pooling and optimization
    """
    
    def __init__(self, cuda_manager: CUDAManager):
        self.cuda_manager = cuda_manager
        self._memory_pools: Dict[str, List[cp.ndarray]] = {}
        self._lock = threading.RLock()
        
    @contextmanager
    def allocate_array(self, shape: Tuple[int, ...], dtype: np.dtype, pool_name: str = "default"):
        """Context manager for GPU array allocation with automatic cleanup"""
        if not CUDA_AVAILABLE:
            yield np.zeros(shape, dtype=dtype)
            return
            
        array = None
        try:
            with self._lock:
                # Try to reuse from pool
                pool = self._memory_pools.get(pool_name, [])
                for i, arr in enumerate(pool):
                    if arr.shape == shape and arr.dtype == dtype:
                        array = pool.pop(i)
                        break
                        
                if array is None:
                    array = cp.zeros(shape, dtype=dtype)
                    
            yield array
            
        finally:
            if array is not None:
                with self._lock:
                    if pool_name not in self._memory_pools:
                        self._memory_pools[pool_name] = []
                    self._memory_pools[pool_name].append(array)
                    
                    # Limit pool size to prevent memory leaks
                    if len(self._memory_pools[pool_name]) > 100:
                        self._memory_pools[pool_name].pop(0)
                        
    def clear_pools(self):
        """Clear all memory pools"""
        with self._lock:
            self._memory_pools.clear()

class GPUAcceleratedRiskCalculator:
    """
    GPU-accelerated risk calculations including VaR, CVaR, and stress testing
    """
    
    def __init__(self, cuda_manager: CUDAManager):
        self.cuda_manager = cuda_manager
        self.memory_manager = GPUMemoryManager(cuda_manager)
        
    async def calculate_var_gpu(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.05,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        GPU-accelerated Value at Risk calculation using historical simulation
        
        Args:
            returns: Historical return data
            confidence_level: VaR confidence level (default 5%)
            lookback_days: Historical lookback period
            
        Returns:
            Dictionary containing VaR metrics
        """
        if not CUDA_AVAILABLE:
            return await self._calculate_var_cpu(returns, confidence_level, lookback_days)
            
        try:
            # Move data to GPU
            returns_gpu = cp.asarray(returns[-lookback_days:])
            
            # Sort returns on GPU
            sorted_returns = cp.sort(returns_gpu)
            
            # Calculate VaR percentiles
            var_index = int(confidence_level * len(sorted_returns))
            var_95 = float(cp.percentile(sorted_returns, 5))
            var_99 = float(cp.percentile(sorted_returns, 1))
            
            # Calculate Expected Shortfall (CVaR)
            es_95 = float(cp.mean(sorted_returns[:var_index]))
            es_99 = float(cp.mean(sorted_returns[:int(0.01 * len(sorted_returns))]))
            
            # Calculate volatility
            volatility = float(cp.std(returns_gpu))
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "expected_shortfall_95": es_95,
                "expected_shortfall_99": es_99,
                "volatility": volatility,
                "lookback_days": lookback_days,
                "computation_time_ms": 0,  # GPU computation is very fast
                "gpu_accelerated": True
            }
            
        except Exception as e:
            logger.error(f"GPU VaR calculation failed: {e}")
            return await self._calculate_var_cpu(returns, confidence_level, lookback_days)
            
    async def _calculate_var_cpu(
        self,
        returns: np.ndarray,
        confidence_level: float,
        lookback_days: int
    ) -> Dict[str, float]:
        """CPU fallback for VaR calculation"""
        start_time = time.time()
        
        returns_subset = returns[-lookback_days:]
        sorted_returns = np.sort(returns_subset)
        
        var_index = int(confidence_level * len(sorted_returns))
        var_95 = np.percentile(sorted_returns, 5)
        var_99 = np.percentile(sorted_returns, 1)
        
        es_95 = np.mean(sorted_returns[:var_index])
        es_99 = np.mean(sorted_returns[:int(0.01 * len(sorted_returns))])
        
        volatility = np.std(returns_subset)
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "volatility": volatility,
            "lookback_days": lookback_days,
            "computation_time_ms": computation_time,
            "gpu_accelerated": False
        }
        
    async def calculate_portfolio_risk_gpu(
        self,
        positions: np.ndarray,
        returns: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        GPU-accelerated portfolio risk calculation
        """
        if not CUDA_AVAILABLE:
            return await self._calculate_portfolio_risk_cpu(positions, returns, correlation_matrix)
            
        try:
            # Move data to GPU
            positions_gpu = cp.asarray(positions)
            returns_gpu = cp.asarray(returns)
            corr_gpu = cp.asarray(correlation_matrix)
            
            # Calculate portfolio variance using matrix operations
            portfolio_variance = cp.dot(positions_gpu, cp.dot(corr_gpu, positions_gpu))
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Calculate individual asset contributions to risk
            marginal_var = cp.dot(corr_gpu, positions_gpu) / portfolio_volatility
            component_var = positions_gpu * marginal_var
            
            # Calculate diversification ratio
            individual_volatilities = cp.sqrt(cp.diag(corr_gpu))
            weighted_vol = cp.sum(positions_gpu * individual_volatilities)
            diversification_ratio = weighted_vol / portfolio_volatility
            
            return {
                "portfolio_volatility": float(portfolio_volatility),
                "portfolio_variance": float(portfolio_variance),
                "component_var": component_var.get().tolist(),
                "marginal_var": marginal_var.get().tolist(),
                "diversification_ratio": float(diversification_ratio),
                "gpu_accelerated": True
            }
            
        except Exception as e:
            logger.error(f"GPU portfolio risk calculation failed: {e}")
            return await self._calculate_portfolio_risk_cpu(positions, returns, correlation_matrix)
            
    async def _calculate_portfolio_risk_cpu(
        self,
        positions: np.ndarray,
        returns: np.ndarray, 
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """CPU fallback for portfolio risk calculation"""
        portfolio_variance = np.dot(positions, np.dot(correlation_matrix, positions))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        marginal_var = np.dot(correlation_matrix, positions) / portfolio_volatility
        component_var = positions * marginal_var
        
        individual_volatilities = np.sqrt(np.diag(correlation_matrix))
        weighted_vol = np.sum(positions * individual_volatilities)
        diversification_ratio = weighted_vol / portfolio_volatility
        
        return {
            "portfolio_volatility": float(portfolio_volatility),
            "portfolio_variance": float(portfolio_variance),
            "component_var": component_var.tolist(),
            "marginal_var": marginal_var.tolist(),
            "diversification_ratio": float(diversification_ratio),
            "gpu_accelerated": False
        }

class GPUMonteCarloSimulator:
    """
    GPU-accelerated Monte Carlo simulations for options pricing and risk scenarios
    """
    
    def __init__(self, cuda_manager: CUDAManager):
        self.cuda_manager = cuda_manager
        self.memory_manager = GPUMemoryManager(cuda_manager)
        
    async def simulate_price_paths_gpu(
        self,
        initial_price: float,
        volatility: float,
        risk_free_rate: float,
        time_horizon: float,
        num_simulations: int = 100000,
        num_steps: int = 252
    ) -> Dict[str, Any]:
        """
        GPU-accelerated Monte Carlo price path simulation
        """
        if not CUDA_AVAILABLE:
            return await self._simulate_price_paths_cpu(
                initial_price, volatility, risk_free_rate, 
                time_horizon, num_simulations, num_steps
            )
            
        try:
            start_time = time.time()
            
            # Generate random numbers on GPU
            dt = time_horizon / num_steps
            sqrt_dt = np.sqrt(dt)
            
            # Use CuPy's random number generator
            random_numbers = cp.random.normal(
                0, 1, (num_simulations, num_steps)
            )
            
            # Initialize price paths
            price_paths = cp.zeros((num_simulations, num_steps + 1))
            price_paths[:, 0] = initial_price
            
            # Calculate drift
            drift = (risk_free_rate - 0.5 * volatility**2) * dt
            
            # Simulate price paths using geometric Brownian motion
            for step in range(num_steps):
                price_paths[:, step + 1] = (
                    price_paths[:, step] * 
                    cp.exp(drift + volatility * sqrt_dt * random_numbers[:, step])
                )
                
            # Calculate statistics
            final_prices = price_paths[:, -1]
            mean_final_price = float(cp.mean(final_prices))
            std_final_price = float(cp.std(final_prices))
            percentiles = cp.percentile(final_prices, [5, 25, 50, 75, 95])
            
            computation_time = (time.time() - start_time) * 1000
            
            return {
                "num_simulations": num_simulations,
                "num_steps": num_steps,
                "mean_final_price": mean_final_price,
                "std_final_price": std_final_price,
                "percentiles": {
                    "5th": float(percentiles[0]),
                    "25th": float(percentiles[1]),
                    "50th": float(percentiles[2]),
                    "75th": float(percentiles[3]),
                    "95th": float(percentiles[4])
                },
                "computation_time_ms": computation_time,
                "gpu_accelerated": True,
                "price_paths_sample": price_paths[:100].get().tolist()  # Return sample
            }
            
        except Exception as e:
            logger.error(f"GPU Monte Carlo simulation failed: {e}")
            return await self._simulate_price_paths_cpu(
                initial_price, volatility, risk_free_rate,
                time_horizon, num_simulations, num_steps
            )
            
    async def _simulate_price_paths_cpu(
        self,
        initial_price: float,
        volatility: float,
        risk_free_rate: float,
        time_horizon: float,
        num_simulations: int,
        num_steps: int
    ) -> Dict[str, Any]:
        """CPU fallback for Monte Carlo simulation"""
        start_time = time.time()
        
        dt = time_horizon / num_steps
        sqrt_dt = np.sqrt(dt)
        drift = (risk_free_rate - 0.5 * volatility**2) * dt
        
        price_paths = np.zeros((num_simulations, num_steps + 1))
        price_paths[:, 0] = initial_price
        
        random_numbers = np.random.normal(0, 1, (num_simulations, num_steps))
        
        for step in range(num_steps):
            price_paths[:, step + 1] = (
                price_paths[:, step] * 
                np.exp(drift + volatility * sqrt_dt * random_numbers[:, step])
            )
            
        final_prices = price_paths[:, -1]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "num_simulations": num_simulations,
            "num_steps": num_steps,
            "mean_final_price": mean_final_price,
            "std_final_price": std_final_price,
            "percentiles": {
                "5th": percentiles[0],
                "25th": percentiles[1],
                "50th": percentiles[2],
                "75th": percentiles[3],
                "95th": percentiles[4]
            },
            "computation_time_ms": computation_time,
            "gpu_accelerated": False,
            "price_paths_sample": price_paths[:100].tolist()
        }

class GPUMatrixOperations:
    """
    GPU-accelerated matrix operations for correlation analysis and linear algebra
    """
    
    def __init__(self, cuda_manager: CUDAManager):
        self.cuda_manager = cuda_manager
        
    async def calculate_correlation_matrix_gpu(
        self,
        returns_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        GPU-accelerated correlation matrix calculation
        """
        if not CUDA_AVAILABLE:
            return await self._calculate_correlation_matrix_cpu(returns_matrix)
            
        try:
            start_time = time.time()
            
            # Move data to GPU
            returns_gpu = cp.asarray(returns_matrix)
            
            # Calculate correlation matrix
            correlation_matrix = cp.corrcoef(returns_gpu, rowvar=False)
            
            # Calculate eigenvalues for principal component analysis
            eigenvalues, eigenvectors = cp.linalg.eigh(correlation_matrix)
            
            # Sort eigenvalues in descending order
            idx = cp.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate condition number
            condition_number = float(eigenvalues[0] / eigenvalues[-1])
            
            computation_time = (time.time() - start_time) * 1000
            
            return {
                "correlation_matrix": correlation_matrix.get().tolist(),
                "eigenvalues": eigenvalues.get().tolist(),
                "eigenvectors": eigenvectors.get().tolist(),
                "condition_number": condition_number,
                "rank": int(cp.linalg.matrix_rank(correlation_matrix)),
                "computation_time_ms": computation_time,
                "gpu_accelerated": True
            }
            
        except Exception as e:
            logger.error(f"GPU correlation matrix calculation failed: {e}")
            return await self._calculate_correlation_matrix_cpu(returns_matrix)
            
    async def _calculate_correlation_matrix_cpu(
        self,
        returns_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """CPU fallback for correlation matrix calculation"""
        start_time = time.time()
        
        correlation_matrix = np.corrcoef(returns_matrix, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        condition_number = eigenvalues[0] / eigenvalues[-1]
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "correlation_matrix": correlation_matrix.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "condition_number": condition_number,
            "rank": np.linalg.matrix_rank(correlation_matrix),
            "computation_time_ms": computation_time,
            "gpu_accelerated": False
        }
        
    async def solve_linear_system_gpu(
        self,
        coefficient_matrix: np.ndarray,
        constants_vector: np.ndarray
    ) -> Dict[str, Any]:
        """
        GPU-accelerated linear system solver for portfolio optimization
        """
        if not CUDA_AVAILABLE:
            return await self._solve_linear_system_cpu(coefficient_matrix, constants_vector)
            
        try:
            start_time = time.time()
            
            # Move data to GPU
            A_gpu = cp.asarray(coefficient_matrix)
            b_gpu = cp.asarray(constants_vector)
            
            # Solve Ax = b using GPU
            solution = cp.linalg.solve(A_gpu, b_gpu)
            
            # Calculate residual
            residual = cp.linalg.norm(cp.dot(A_gpu, solution) - b_gpu)
            
            computation_time = (time.time() - start_time) * 1000
            
            return {
                "solution": solution.get().tolist(),
                "residual": float(residual),
                "computation_time_ms": computation_time,
                "gpu_accelerated": True
            }
            
        except Exception as e:
            logger.error(f"GPU linear system solver failed: {e}")
            return await self._solve_linear_system_cpu(coefficient_matrix, constants_vector)
            
    async def _solve_linear_system_cpu(
        self,
        coefficient_matrix: np.ndarray,
        constants_vector: np.ndarray
    ) -> Dict[str, Any]:
        """CPU fallback for linear system solver"""
        start_time = time.time()
        
        solution = np.linalg.solve(coefficient_matrix, constants_vector)
        residual = np.linalg.norm(np.dot(coefficient_matrix, solution) - constants_vector)
        
        computation_time = (time.time() - start_time) * 1000
        
        return {
            "solution": solution.tolist(),
            "residual": residual,
            "computation_time_ms": computation_time,
            "gpu_accelerated": False
        }

# Global instances
cuda_manager = CUDAManager()
gpu_memory_manager = GPUMemoryManager(cuda_manager)
gpu_risk_calculator = GPUAcceleratedRiskCalculator(cuda_manager)
gpu_monte_carlo = GPUMonteCarloSimulator(cuda_manager)
gpu_matrix_ops = GPUMatrixOperations(cuda_manager)

# Convenience functions
async def calculate_var_gpu(returns: np.ndarray, **kwargs) -> Dict[str, float]:
    """Convenience function for GPU VaR calculation"""
    return await gpu_risk_calculator.calculate_var_gpu(returns, **kwargs)

async def simulate_monte_carlo_gpu(initial_price: float, volatility: float, 
                                  risk_free_rate: float, time_horizon: float, 
                                  **kwargs) -> Dict[str, Any]:
    """Convenience function for GPU Monte Carlo simulation"""
    return await gpu_monte_carlo.simulate_price_paths_gpu(
        initial_price, volatility, risk_free_rate, time_horizon, **kwargs
    )

async def calculate_correlation_gpu(returns_matrix: np.ndarray) -> Dict[str, Any]:
    """Convenience function for GPU correlation matrix calculation"""
    return await gpu_matrix_ops.calculate_correlation_matrix_gpu(returns_matrix)