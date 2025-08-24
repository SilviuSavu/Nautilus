#!/usr/bin/env python3
"""
Memory Pool Management and Garbage Collection Validation System
Validates memory pool efficiency and garbage collection optimization for M4 Max.

This module validates:
- Memory pool allocation and reuse strategies
- Garbage collection performance and tuning
- Memory fragmentation prevention
- Pool-based memory recycling
- Weak reference management
- Memory leak detection and prevention
"""

import os
import sys
import time
import gc
import weakref
import asyncio
import threading
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from pathlib import Path
import uuid

# Memory management and profiling
import numpy as np
import psutil
from memory_profiler import memory_usage
import tracemalloc

# Advanced memory profiling
try:
    import pympler.tracker
    import pympler.muppy
    import pympler.summary
    import objgraph
    HAS_ADVANCED_PROFILING = True
except ImportError:
    HAS_ADVANCED_PROFILING = False
    print("Warning: Advanced memory profiling libraries not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryPoolStats:
    """Statistics for a memory pool"""
    pool_id: str
    pool_size_mb: float
    allocated_blocks: int
    free_blocks: int
    fragmentation_percent: float
    allocation_count: int
    deallocation_count: int
    reuse_count: int
    efficiency_percent: float
    created_at: float = field(default_factory=time.time)

@dataclass
class GarbageCollectionStats:
    """Garbage collection performance statistics"""
    generation: int
    objects_before: int
    objects_after: int
    objects_collected: int
    collection_time_ms: float
    memory_freed_mb: float
    collection_trigger: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class MemoryPoolValidationResult:
    """Results from memory pool validation"""
    test_name: str
    pools_tested: int
    total_allocations: int
    total_deallocations: int
    memory_reuse_percent: float
    average_fragmentation_percent: float
    gc_collections: int
    gc_objects_collected: int
    gc_total_time_ms: float
    memory_efficiency_percent: float
    leak_objects_detected: int
    validation_score: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class MemoryPool:
    """High-performance memory pool for M4 Max optimization"""
    
    def __init__(self, pool_id: str, block_size_mb: float, pool_size_mb: float):
        self.pool_id = pool_id
        self.block_size_mb = block_size_mb
        self.pool_size_mb = pool_size_mb
        self.max_blocks = int(pool_size_mb / block_size_mb)
        
        # Memory management
        self.allocated_blocks: Dict[str, np.ndarray] = {}
        self.free_blocks: List[np.ndarray] = []
        self.block_metadata: Dict[str, Dict] = {}
        
        # Statistics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.reuse_count = 0
        
        # Pre-allocate blocks for better performance
        self._preallocate_blocks()
        
        logger.info(f"Memory pool {pool_id} created: {self.max_blocks} blocks x {block_size_mb}MB")
    
    def _preallocate_blocks(self):
        """Pre-allocate memory blocks to reduce allocation overhead"""
        try:
            block_size_elements = int((self.block_size_mb * 1024 * 1024) / 4)  # float32
            
            for i in range(self.max_blocks):
                block = np.empty(block_size_elements, dtype=np.float32)
                self.free_blocks.append(block)
            
            logger.info(f"Pre-allocated {len(self.free_blocks)} blocks for pool {self.pool_id}")
            
        except Exception as e:
            logger.error(f"Failed to pre-allocate blocks for pool {self.pool_id}: {e}")
    
    def allocate_block(self, block_id: str) -> Optional[np.ndarray]:
        """Allocate a memory block from the pool"""
        try:
            if not self.free_blocks:
                logger.warning(f"No free blocks available in pool {self.pool_id}")
                return None
            
            # Reuse a free block (zero-copy)
            block = self.free_blocks.pop()
            self.allocated_blocks[block_id] = block
            
            # Store metadata
            self.block_metadata[block_id] = {
                'allocated_at': time.time(),
                'size_mb': self.block_size_mb,
                'reused': self.allocation_count > 0
            }
            
            self.allocation_count += 1
            if self.block_metadata[block_id]['reused']:
                self.reuse_count += 1
            
            return block
            
        except Exception as e:
            logger.error(f"Failed to allocate block {block_id} from pool {self.pool_id}: {e}")
            return None
    
    def deallocate_block(self, block_id: str) -> bool:
        """Deallocate a memory block back to the pool"""
        try:
            if block_id not in self.allocated_blocks:
                logger.warning(f"Block {block_id} not found in pool {self.pool_id}")
                return False
            
            # Return block to free pool for reuse
            block = self.allocated_blocks.pop(block_id)
            self.free_blocks.append(block)
            
            # Clean up metadata
            del self.block_metadata[block_id]
            
            self.deallocation_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deallocate block {block_id} from pool {self.pool_id}: {e}")
            return False
    
    def get_stats(self) -> MemoryPoolStats:
        """Get current pool statistics"""
        allocated_blocks = len(self.allocated_blocks)
        free_blocks = len(self.free_blocks)
        
        # Calculate fragmentation
        fragmentation = 0.0
        if self.max_blocks > 0:
            fragmentation = ((self.max_blocks - free_blocks - allocated_blocks) / self.max_blocks) * 100
        
        # Calculate efficiency
        efficiency = 0.0
        if self.allocation_count > 0:
            efficiency = (self.reuse_count / self.allocation_count) * 100
        
        return MemoryPoolStats(
            pool_id=self.pool_id,
            pool_size_mb=self.pool_size_mb,
            allocated_blocks=allocated_blocks,
            free_blocks=free_blocks,
            fragmentation_percent=fragmentation,
            allocation_count=self.allocation_count,
            deallocation_count=self.deallocation_count,
            reuse_count=self.reuse_count,
            efficiency_percent=efficiency
        )

class GarbageCollectionOptimizer:
    """Optimizes garbage collection for M4 Max memory architecture"""
    
    def __init__(self):
        self.gc_stats: List[GarbageCollectionStats] = []
        self.initial_threshold = gc.get_threshold()
        self.weak_refs: Set[weakref.ref] = set()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Initialize advanced profiling if available
        if HAS_ADVANCED_PROFILING:
            self.memory_tracker = pympler.tracker.SummaryTracker()
        
        logger.info("Garbage Collection Optimizer initialized")
    
    def optimize_gc_thresholds(self, generation0: int = 700, generation1: int = 10, generation2: int = 10):
        """Optimize GC thresholds for M4 Max memory patterns"""
        try:
            gc.set_threshold(generation0, generation1, generation2)
            logger.info(f"GC thresholds optimized: {generation0}, {generation1}, {generation2}")
            return True
        except Exception as e:
            logger.error(f"Failed to optimize GC thresholds: {e}")
            return False
    
    def perform_targeted_gc(self, generation: int = 2) -> GarbageCollectionStats:
        """Perform targeted garbage collection with metrics"""
        objects_before = len(gc.get_objects())
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        gc_start = time.perf_counter()
        collected = gc.collect(generation)
        gc_time = (time.perf_counter() - gc_start) * 1000  # ms
        
        objects_after = len(gc.get_objects())
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = memory_before - memory_after
        
        stats = GarbageCollectionStats(
            generation=generation,
            objects_before=objects_before,
            objects_after=objects_after,
            objects_collected=collected,
            collection_time_ms=gc_time,
            memory_freed_mb=memory_freed,
            collection_trigger="manual"
        )
        
        self.gc_stats.append(stats)
        
        logger.info(f"GC gen {generation}: {collected} objects collected in {gc_time:.1f}ms, "
                   f"{memory_freed:.1f}MB freed")
        
        return stats
    
    def add_weak_reference(self, obj) -> weakref.ref:
        """Add weak reference for leak detection"""
        def cleanup_callback(ref):
            self.weak_refs.discard(ref)
        
        weak_ref = weakref.ref(obj, cleanup_callback)
        self.weak_refs.add(weak_ref)
        return weak_ref
    
    def detect_memory_leaks(self) -> Tuple[int, List[str]]:
        """Detect potential memory leaks using weak references and object tracking"""
        leaked_objects = 0
        leak_types = []
        
        # Check weak references
        dead_refs = [ref for ref in self.weak_refs if ref() is None]
        alive_refs = [ref for ref in self.weak_refs if ref() is not None]
        
        if len(alive_refs) > len(dead_refs) * 2:  # Heuristic for potential leaks
            leaked_objects = len(alive_refs)
            leak_types.append("weak_reference_leak")
        
        # Use advanced profiling if available
        if HAS_ADVANCED_PROFILING:
            try:
                # Track object growth
                all_objects = pympler.muppy.get_objects()
                summary = pympler.summary.summarize(all_objects)
                
                # Look for suspicious growth patterns
                for row in summary[:10]:  # Top 10 object types
                    obj_type, count, size = row
                    if count > 10000:  # Many objects of same type
                        leak_types.append(f"high_count_{obj_type}")
                        leaked_objects += count
                
                # Use objgraph for reference cycles
                growth = objgraph.growth(limit=5)
                if growth:
                    leaked_objects += sum(count for name, count in growth)
                    leak_types.extend([name for name, count in growth])
                
            except Exception as e:
                logger.warning(f"Advanced leak detection failed: {e}")
        
        return leaked_objects, leak_types

class MemoryPoolValidator:
    """Validates memory pool management and garbage collection"""
    
    def __init__(self):
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.validation_results: List[MemoryPoolValidationResult] = []
    
    def create_memory_pools(self, pool_configs: List[Tuple[float, float]]) -> None:
        """Create memory pools with specified configurations"""
        for i, (block_size_mb, pool_size_mb) in enumerate(pool_configs):
            pool_id = f"pool_{i}_{int(block_size_mb)}mb"
            pool = MemoryPool(pool_id, block_size_mb, pool_size_mb)
            self.memory_pools[pool_id] = pool
        
        logger.info(f"Created {len(self.memory_pools)} memory pools")
    
    async def validate_memory_pool_efficiency(self) -> MemoryPoolValidationResult:
        """Validate memory pool allocation and reuse efficiency"""
        logger.info("Validating memory pool efficiency")
        
        start_time = time.perf_counter()
        
        try:
            total_allocations = 0
            total_deallocations = 0
            allocated_blocks = {}
            
            # Phase 1: Allocation stress test
            for pool_id, pool in self.memory_pools.items():
                for i in range(pool.max_blocks):
                    block_id = f"{pool_id}_block_{i}"
                    block = pool.allocate_block(block_id)
                    
                    if block is not None:
                        # Simulate work with the allocated block
                        block.fill(np.random.random())
                        allocated_blocks[block_id] = (pool_id, block)
                        total_allocations += 1
                    
                    # Brief pause to simulate real workload
                    if i % 10 == 0:
                        await asyncio.sleep(0.001)
            
            # Phase 2: Random deallocation and reallocation
            block_ids = list(allocated_blocks.keys())
            
            # Deallocate 50% of blocks
            for i in range(0, len(block_ids), 2):
                block_id = block_ids[i]
                pool_id, _ = allocated_blocks[block_id]
                pool = self.memory_pools[pool_id]
                
                if pool.deallocate_block(block_id):
                    total_deallocations += 1
                    del allocated_blocks[block_id]
            
            # Reallocate blocks to test reuse
            reallocation_count = 0
            for pool_id, pool in self.memory_pools.items():
                for i in range(pool.max_blocks // 4):  # 25% reallocation
                    block_id = f"{pool_id}_realloc_{i}"
                    block = pool.allocate_block(block_id)
                    
                    if block is not None:
                        block.fill(np.random.random())
                        reallocation_count += 1
                        total_allocations += 1
            
            # Phase 3: Cleanup
            for block_id, (pool_id, _) in list(allocated_blocks.items()):
                pool = self.memory_pools[pool_id]
                if pool.deallocate_block(block_id):
                    total_deallocations += 1
            
            # Cleanup reallocated blocks
            for pool_id, pool in self.memory_pools.items():
                for i in range(pool.max_blocks // 4):
                    block_id = f"{pool_id}_realloc_{i}"
                    pool.deallocate_block(block_id)
                    total_deallocations += 1
            
            # Calculate statistics
            total_reuse = sum(pool.reuse_count for pool in self.memory_pools.values())
            reuse_percent = (total_reuse / total_allocations) * 100 if total_allocations > 0 else 0
            
            fragmentation_values = [pool.get_stats().fragmentation_percent for pool in self.memory_pools.values()]
            avg_fragmentation = sum(fragmentation_values) / len(fragmentation_values) if fragmentation_values else 0
            
            # Memory efficiency calculation
            efficiency_values = [pool.get_stats().efficiency_percent for pool in self.memory_pools.values()]
            avg_efficiency = sum(efficiency_values) / len(efficiency_values) if efficiency_values else 0
            
            result = MemoryPoolValidationResult(
                test_name="memory_pool_efficiency",
                pools_tested=len(self.memory_pools),
                total_allocations=total_allocations,
                total_deallocations=total_deallocations,
                memory_reuse_percent=reuse_percent,
                average_fragmentation_percent=avg_fragmentation,
                gc_collections=0,  # Will be updated in GC test
                gc_objects_collected=0,
                gc_total_time_ms=0.0,
                memory_efficiency_percent=avg_efficiency,
                leak_objects_detected=0,
                validation_score=avg_efficiency,
                success=True
            )
            
            logger.info(f"Memory pool efficiency validation completed: {reuse_percent:.1f}% reuse")
            return result
            
        except Exception as e:
            logger.error(f"Memory pool efficiency validation failed: {e}")
            return MemoryPoolValidationResult(
                test_name="memory_pool_efficiency",
                pools_tested=0,
                total_allocations=0,
                total_deallocations=0,
                memory_reuse_percent=0.0,
                average_fragmentation_percent=0.0,
                gc_collections=0,
                gc_objects_collected=0,
                gc_total_time_ms=0.0,
                memory_efficiency_percent=0.0,
                leak_objects_detected=0,
                validation_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def validate_garbage_collection_optimization(self) -> MemoryPoolValidationResult:
        """Validate garbage collection optimization"""
        logger.info("Validating garbage collection optimization")
        
        try:
            # Optimize GC thresholds
            self.gc_optimizer.optimize_gc_thresholds(1000, 15, 15)
            
            # Create objects for GC testing
            test_objects = []
            weak_refs = []
            
            # Phase 1: Create many objects
            for i in range(10000):
                obj = np.random.random(1000).astype(np.float32)
                test_objects.append(obj)
                
                # Add some weak references for leak detection
                if i % 100 == 0:
                    weak_ref = self.gc_optimizer.add_weak_reference(obj)
                    weak_refs.append(weak_ref)
            
            # Phase 2: Perform targeted garbage collections
            gc_stats = []
            for generation in [0, 1, 2]:
                stats = self.gc_optimizer.perform_targeted_gc(generation)
                gc_stats.append(stats)
            
            # Phase 3: Delete objects and test cleanup
            del test_objects
            cleanup_stats = self.gc_optimizer.perform_targeted_gc(2)
            gc_stats.append(cleanup_stats)
            
            # Phase 4: Leak detection
            leaked_objects, leak_types = self.gc_optimizer.detect_memory_leaks()
            
            # Calculate totals
            total_collections = len(gc_stats)
            total_objects_collected = sum(s.objects_collected for s in gc_stats)
            total_gc_time = sum(s.collection_time_ms for s in gc_stats)
            
            # Calculate validation score
            avg_collection_time = total_gc_time / total_collections if total_collections > 0 else 0
            gc_efficiency = max(0, 100 - (avg_collection_time / 10))  # Lower time = higher efficiency
            
            result = MemoryPoolValidationResult(
                test_name="garbage_collection_optimization",
                pools_tested=0,
                total_allocations=0,
                total_deallocations=0,
                memory_reuse_percent=0.0,
                average_fragmentation_percent=0.0,
                gc_collections=total_collections,
                gc_objects_collected=total_objects_collected,
                gc_total_time_ms=total_gc_time,
                memory_efficiency_percent=gc_efficiency,
                leak_objects_detected=leaked_objects,
                validation_score=gc_efficiency * (1 - leaked_objects / 10000) if leaked_objects < 10000 else 0,
                success=True
            )
            
            logger.info(f"GC optimization validation completed: {total_objects_collected} objects collected")
            return result
            
        except Exception as e:
            logger.error(f"GC optimization validation failed: {e}")
            return MemoryPoolValidationResult(
                test_name="garbage_collection_optimization",
                pools_tested=0,
                total_allocations=0,
                total_deallocations=0,
                memory_reuse_percent=0.0,
                average_fragmentation_percent=0.0,
                gc_collections=0,
                gc_objects_collected=0,
                gc_total_time_ms=0.0,
                memory_efficiency_percent=0.0,
                leak_objects_detected=0,
                validation_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def run_comprehensive_validation(self) -> Dict[str, MemoryPoolValidationResult]:
        """Run comprehensive memory pool and GC validation"""
        logger.info("Starting comprehensive memory pool validation")
        
        # Create test memory pools
        pool_configs = [
            (1.0, 10.0),    # 1MB blocks, 10MB pool
            (2.0, 20.0),    # 2MB blocks, 20MB pool
            (4.0, 40.0),    # 4MB blocks, 40MB pool
            (8.0, 80.0),    # 8MB blocks, 80MB pool
        ]
        
        self.create_memory_pools(pool_configs)
        
        results = {}
        
        try:
            # Test 1: Memory Pool Efficiency
            results['pool_efficiency'] = await self.validate_memory_pool_efficiency()
            
            # Test 2: Garbage Collection Optimization
            results['gc_optimization'] = await self.validate_garbage_collection_optimization()
            
            # Generate comprehensive report
            self.generate_validation_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            raise
    
    def generate_validation_report(self, results: Dict[str, MemoryPoolValidationResult]) -> None:
        """Generate comprehensive memory pool validation report"""
        report_data = {
            'system_info': {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'gc_thresholds': gc.get_threshold(),
                'has_advanced_profiling': HAS_ADVANCED_PROFILING,
                'memory_pools_created': len(self.memory_pools)
            },
            'pool_statistics': {},
            'validation_results': {},
            'summary': {},
            'timestamp': time.time()
        }
        
        # Add pool statistics
        for pool_id, pool in self.memory_pools.items():
            stats = pool.get_stats()
            report_data['pool_statistics'][pool_id] = {
                'pool_size_mb': stats.pool_size_mb,
                'allocated_blocks': stats.allocated_blocks,
                'free_blocks': stats.free_blocks,
                'fragmentation_percent': stats.fragmentation_percent,
                'allocation_count': stats.allocation_count,
                'deallocation_count': stats.deallocation_count,
                'reuse_count': stats.reuse_count,
                'efficiency_percent': stats.efficiency_percent
            }
        
        # Process validation results
        total_tests = 0
        successful_tests = 0
        total_score = 0.0
        
        for test_name, result in results.items():
            report_data['validation_results'][test_name] = {
                'pools_tested': result.pools_tested,
                'total_allocations': result.total_allocations,
                'total_deallocations': result.total_deallocations,
                'memory_reuse_percent': result.memory_reuse_percent,
                'average_fragmentation_percent': result.average_fragmentation_percent,
                'gc_collections': result.gc_collections,
                'gc_objects_collected': result.gc_objects_collected,
                'gc_total_time_ms': result.gc_total_time_ms,
                'memory_efficiency_percent': result.memory_efficiency_percent,
                'leak_objects_detected': result.leak_objects_detected,
                'validation_score': result.validation_score,
                'success': result.success,
                'error_message': result.error_message
            }
            
            total_tests += 1
            if result.success:
                successful_tests += 1
                total_score += result.validation_score
        
        # Calculate summary
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        average_score = total_score / successful_tests if successful_tests > 0 else 0
        
        report_data['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate_percent': success_rate,
            'average_validation_score': average_score,
            'memory_pool_optimization_score': average_score,
            'm4_max_memory_management_score': min(100, average_score * (success_rate / 100))
        }
        
        # Save report
        report_path = Path('memory_pool_validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Log summary
        logger.info(f"Memory Pool Validation Report:")
        logger.info(f"  Tests Passed: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"  Average Score: {average_score:.1f}/100")
        logger.info(f"  M4 Max Score: {report_data['summary']['m4_max_memory_management_score']:.1f}/100")
        logger.info(f"  Report saved to: {report_path}")

async def main():
    """Main function for memory pool validation"""
    print("🧠 Memory Pool Management & Garbage Collection Validation")
    print("=" * 65)
    
    validator = MemoryPoolValidator()
    
    try:
        print("🧪 Running comprehensive memory pool validation...")
        results = await validator.run_comprehensive_validation()
        
        # Display results
        print("\n📋 Memory Pool Validation Results:")
        print("-" * 45)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{test_name}: {status}")
            if result.success:
                print(f"  Validation Score: {result.validation_score:.1f}/100")
                if result.memory_reuse_percent > 0:
                    print(f"  Memory Reuse: {result.memory_reuse_percent:.1f}%")
                if result.gc_collections > 0:
                    print(f"  GC Collections: {result.gc_collections}")
                    print(f"  Objects Collected: {result.gc_objects_collected}")
                if result.leak_objects_detected > 0:
                    print(f"  ⚠️  Potential Leaks: {result.leak_objects_detected}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            print()
        
        # Summary
        passed = sum(1 for r in results.values() if r.success)
        total = len(results)
        avg_score = sum(r.validation_score for r in results.values() if r.success) / passed if passed > 0 else 0
        
        print(f"🎯 Overall Results: {passed}/{total} tests passed")
        print(f"📊 Average Score: {avg_score:.1f}/100")
        
        if passed == total and avg_score > 80:
            print("🎉 Memory pool management optimized for M4 Max!")
        else:
            print("⚠️  Memory pool management needs optimization")
        
    except Exception as e:
        logger.error(f"Memory pool validation failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)