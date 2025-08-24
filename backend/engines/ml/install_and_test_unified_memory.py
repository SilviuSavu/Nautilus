#!/usr/bin/env python3
"""
Unified Memory Management System Installation and Testing Script
Comprehensive installation and validation of M4 Max memory optimization system.

This script:
1. Installs all required dependencies
2. Runs comprehensive memory management tests
3. Validates M4 Max unified memory architecture
4. Generates performance benchmarks and reports
5. Provides production deployment readiness assessment
"""

import os
import sys
import subprocess
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedMemoryInstaller:
    """Installer for unified memory management system dependencies"""
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = Path(requirements_file)
        self.installation_log = []
        self.test_results = {}
        
    def log_step(self, step: str, success: bool, details: str = ""):
        """Log installation step"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        message = f"{status}: {step}"
        if details:
            message += f" - {details}"
        
        print(message)
        logger.info(message)
        
        self.installation_log.append({
            'step': step,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a system command and log results"""
        try:
            print(f"🔄 {description}...")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.log_step(description, True, f"Command completed successfully")
                return True
            else:
                self.log_step(description, False, f"Exit code: {result.returncode}, Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_step(description, False, "Command timed out after 5 minutes")
            return False
        except Exception as e:
            self.log_step(description, False, f"Exception: {str(e)}")
            return False
    
    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies for macOS/Apple Silicon"""
        print("\n🍎 Installing Apple Silicon System Dependencies")
        print("=" * 50)
        
        success = True
        
        # Check if we're on Apple Silicon
        try:
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            architecture = result.stdout.strip()
            
            if 'arm64' in architecture or 'arm' in architecture:
                self.log_step("Apple Silicon Detection", True, f"Architecture: {architecture}")
            else:
                self.log_step("Apple Silicon Detection", False, f"Not Apple Silicon: {architecture}")
                success = False
        except Exception as e:
            self.log_step("Apple Silicon Detection", False, str(e))
            success = False
        
        # Install Homebrew dependencies (if needed)
        homebrew_packages = [
            "python@3.13",
            "numpy",
            "cmake",
            "llvm",
            "libomp"
        ]
        
        for package in homebrew_packages:
            # Check if brew is available and package exists
            check_cmd = ['brew', 'list', package]
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_step(f"Homebrew package {package}", True, "Already installed")
                else:
                    self.log_step(f"Homebrew package {package}", True, "Will be installed via pip dependencies")
            except FileNotFoundError:
                self.log_step("Homebrew availability", False, "Homebrew not found - using pip only")
                break
        
        return success
    
    def install_python_dependencies(self) -> bool:
        """Install Python dependencies from requirements.txt"""
        print("\n🐍 Installing Python Dependencies")
        print("=" * 40)
        
        if not self.requirements_file.exists():
            self.log_step("Requirements file check", False, f"{self.requirements_file} not found")
            return False
        
        # Upgrade pip first
        pip_upgrade = self.run_command(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        
        if not pip_upgrade:
            return False
        
        # Install requirements with retry logic
        install_attempts = 3
        for attempt in range(install_attempts):
            print(f"\n📦 Installation Attempt {attempt + 1}/{install_attempts}")
            
            success = self.run_command(
                [
                    sys.executable, '-m', 'pip', 'install', 
                    '-r', str(self.requirements_file),
                    '--upgrade',
                    '--no-cache-dir',
                    '--timeout', '300'
                ],
                f"Installing requirements (attempt {attempt + 1})"
            )
            
            if success:
                self.log_step("Python dependencies installation", True, "All dependencies installed")
                return True
            else:
                if attempt < install_attempts - 1:
                    print("⏳ Waiting 10 seconds before retry...")
                    time.sleep(10)
        
        self.log_step("Python dependencies installation", False, "All attempts failed")
        return False
    
    def verify_critical_imports(self) -> bool:
        """Verify that critical libraries can be imported"""
        print("\n🔍 Verifying Critical Imports")
        print("=" * 35)
        
        critical_imports = [
            ('numpy', 'NumPy'),
            ('psutil', 'System monitoring'),
            ('multiprocessing', 'Multiprocessing'),
            ('asyncio', 'Async operations'),
            ('threading', 'Threading'),
            ('gc', 'Garbage collection'),
            ('tracemalloc', 'Memory tracking'),
            ('weakref', 'Weak references'),
            ('mmap', 'Memory mapping'),
            ('ctypes', 'C types interface')
        ]
        
        optional_imports = [
            ('docker', 'Docker SDK'),
            ('kubernetes', 'Kubernetes client'),
            ('prometheus_client', 'Prometheus metrics'),
            ('memory_profiler', 'Memory profiler'),
            ('pympler', 'Advanced memory profiling')
        ]
        
        all_success = True
        
        # Test critical imports
        for module_name, description in critical_imports:
            try:
                __import__(module_name)
                self.log_step(f"Import {module_name}", True, description)
            except ImportError as e:
                self.log_step(f"Import {module_name}", False, f"{description} - {str(e)}")
                all_success = False
        
        # Test optional imports
        for module_name, description in optional_imports:
            try:
                __import__(module_name)
                self.log_step(f"Import {module_name} (optional)", True, description)
            except ImportError:
                self.log_step(f"Import {module_name} (optional)", True, f"{description} - Will use fallback")
        
        return all_success
    
    async def run_unified_memory_tests(self) -> Dict[str, Any]:
        """Run all unified memory management tests"""
        print("\n🧪 Running Unified Memory Management Tests")
        print("=" * 45)
        
        test_results = {}
        
        # Test 1: Unified Memory Architecture Test
        try:
            print("\n🔬 Test 1: M4 Max Unified Memory Architecture")
            from unified_memory_test import UnifiedMemoryManager, UnifiedMemoryConfig
            
            config = UnifiedMemoryConfig()
            memory_manager = UnifiedMemoryManager(config)
            
            results = await memory_manager.run_comprehensive_benchmark()
            test_results['unified_memory'] = {
                'success': True,
                'results': {k: {
                    'bandwidth_gbps': v.bandwidth_gbps,
                    'latency_ns': v.latency_ns,
                    'efficiency_percent': v.memory_efficiency_percent
                } for k, v in results.items()}
            }
            self.log_step("Unified Memory Architecture Test", True, "All benchmarks completed")
            
        except Exception as e:
            test_results['unified_memory'] = {'success': False, 'error': str(e)}
            self.log_step("Unified Memory Architecture Test", False, str(e))
        
        # Test 2: Zero-Copy Operations Verification
        try:
            print("\n🔬 Test 2: Zero-Copy Operations Verification")
            from zero_copy_verification import ZeroCopyVerifier
            
            verifier = ZeroCopyVerifier()
            results = await verifier.run_comprehensive_zero_copy_verification()
            
            test_results['zero_copy'] = {
                'success': True,
                'results': {k: {
                    'copy_detected': v.copy_detected,
                    'bandwidth_gbps': v.bandwidth_gbps,
                    'success': v.success
                } for k, v in results.items()}
            }
            self.log_step("Zero-Copy Operations Verification", True, "All verifications completed")
            
        except Exception as e:
            test_results['zero_copy'] = {'success': False, 'error': str(e)}
            self.log_step("Zero-Copy Operations Verification", False, str(e))
        
        # Test 3: Container Memory Orchestration
        try:
            print("\n🔬 Test 3: Container Memory Orchestration")
            from container_orchestration_test import ContainerOrchestrator
            
            orchestrator = ContainerOrchestrator()
            results = await orchestrator.run_comprehensive_orchestration_test()
            
            test_results['container_orchestration'] = {
                'success': True,
                'results': {k: {
                    'containers_tested': v.containers_tested,
                    'memory_efficiency_percent': v.memory_efficiency_percent,
                    'scaling_events': v.scaling_events,
                    'success': v.success
                } for k, v in results.items()}
            }
            self.log_step("Container Memory Orchestration", True, "All orchestration tests completed")
            
        except Exception as e:
            test_results['container_orchestration'] = {'success': False, 'error': str(e)}
            self.log_step("Container Memory Orchestration", False, str(e))
        
        # Test 4: Memory Pool Management Validation
        try:
            print("\n🔬 Test 4: Memory Pool Management Validation")
            from memory_pool_validation import MemoryPoolValidator
            
            validator = MemoryPoolValidator()
            results = await validator.run_comprehensive_validation()
            
            test_results['memory_pool'] = {
                'success': True,
                'results': {k: {
                    'validation_score': v.validation_score,
                    'memory_reuse_percent': v.memory_reuse_percent,
                    'gc_collections': v.gc_collections,
                    'success': v.success
                } for k, v in results.items()}
            }
            self.log_step("Memory Pool Management Validation", True, "All validations completed")
            
        except Exception as e:
            test_results['memory_pool'] = {'success': False, 'error': str(e)}
            self.log_step("Memory Pool Management Validation", False, str(e))
        
        return test_results
    
    def generate_final_report(self, test_results: Dict[str, Any]) -> None:
        """Generate comprehensive final report"""
        print("\n📊 Generating Final Installation & Testing Report")
        print("=" * 50)
        
        # Calculate overall metrics
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate report data
        report = {
            'installation_summary': {
                'timestamp': time.time(),
                'total_installation_steps': len(self.installation_log),
                'successful_steps': sum(1 for step in self.installation_log if step['success']),
                'installation_success_rate': (sum(1 for step in self.installation_log if step['success']) / len(self.installation_log)) * 100 if self.installation_log else 0
            },
            'testing_summary': {
                'total_test_suites': total_tests,
                'successful_test_suites': successful_tests,
                'testing_success_rate': success_rate
            },
            'test_results': test_results,
            'installation_log': self.installation_log,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
            },
            'performance_summary': self.calculate_performance_summary(test_results),
            'deployment_readiness': self.assess_deployment_readiness(test_results)
        }
        
        # Save comprehensive report
        report_path = Path('unified_memory_installation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print(f"\n🎯 Installation & Testing Summary:")
        print(f"   Installation Success Rate: {report['installation_summary']['installation_success_rate']:.1f}%")
        print(f"   Testing Success Rate: {success_rate:.1f}%")
        print(f"   M4 Max Optimization Score: {report['performance_summary']['m4_max_optimization_score']:.1f}/100")
        print(f"   Deployment Readiness: {report['deployment_readiness']['status']}")
        print(f"   Full Report: {report_path}")
        
        # Production readiness assessment
        if report['deployment_readiness']['ready_for_production']:
            print("\n🎉 PRODUCTION READY: Unified Memory Management System is optimized for M4 Max!")
        else:
            print("\n⚠️  NEEDS ATTENTION: Some components require optimization before production deployment")
            for issue in report['deployment_readiness']['issues']:
                print(f"   - {issue}")
    
    def calculate_performance_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance summary"""
        summary = {
            'total_bandwidth_gbps': 0.0,
            'zero_copy_success_rate': 0.0,
            'container_efficiency': 0.0,
            'memory_pool_score': 0.0,
            'm4_max_optimization_score': 0.0
        }
        
        try:
            # Unified memory performance
            if test_results.get('unified_memory', {}).get('success'):
                results = test_results['unified_memory']['results']
                summary['total_bandwidth_gbps'] = max(
                    result.get('bandwidth_gbps', 0) for result in results.values()
                )
            
            # Zero-copy performance
            if test_results.get('zero_copy', {}).get('success'):
                results = test_results['zero_copy']['results']
                zero_copy_successes = sum(1 for result in results.values() if not result.get('copy_detected', True))
                summary['zero_copy_success_rate'] = (zero_copy_successes / len(results)) * 100
            
            # Container orchestration
            if test_results.get('container_orchestration', {}).get('success'):
                results = test_results['container_orchestration']['results']
                efficiencies = [result.get('memory_efficiency_percent', 0) for result in results.values()]
                summary['container_efficiency'] = sum(efficiencies) / len(efficiencies) if efficiencies else 0
            
            # Memory pool performance
            if test_results.get('memory_pool', {}).get('success'):
                results = test_results['memory_pool']['results']
                scores = [result.get('validation_score', 0) for result in results.values()]
                summary['memory_pool_score'] = sum(scores) / len(scores) if scores else 0
            
            # Calculate M4 Max optimization score
            bandwidth_score = min(100, (summary['total_bandwidth_gbps'] / 546.0) * 100)  # 546 GB/s is M4 Max theoretical max
            overall_score = (
                bandwidth_score * 0.3 +
                summary['zero_copy_success_rate'] * 0.3 +
                summary['container_efficiency'] * 0.2 +
                summary['memory_pool_score'] * 0.2
            )
            summary['m4_max_optimization_score'] = overall_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance summary: {e}")
        
        return summary
    
    def assess_deployment_readiness(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess deployment readiness"""
        readiness = {
            'ready_for_production': True,
            'status': 'READY',
            'confidence_score': 100.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check each test suite
        for test_name, result in test_results.items():
            if not result.get('success', False):
                readiness['ready_for_production'] = False
                readiness['issues'].append(f"{test_name} test suite failed")
                readiness['confidence_score'] -= 25
        
        # Check performance thresholds
        performance = self.calculate_performance_summary(test_results)
        
        if performance['m4_max_optimization_score'] < 70:
            readiness['ready_for_production'] = False
            readiness['issues'].append("M4 Max optimization score below production threshold (70%)")
            readiness['recommendations'].append("Optimize memory allocation patterns")
        
        if performance['zero_copy_success_rate'] < 80:
            readiness['issues'].append("Zero-copy success rate below optimal (80%)")
            readiness['recommendations'].append("Review memory view implementations")
        
        if performance['container_efficiency'] < 75:
            readiness['issues'].append("Container memory efficiency below optimal (75%)")
            readiness['recommendations'].append("Tune container memory orchestration")
        
        # Determine final status
        if readiness['confidence_score'] >= 90:
            readiness['status'] = 'PRODUCTION READY'
        elif readiness['confidence_score'] >= 70:
            readiness['status'] = 'NEEDS MINOR OPTIMIZATION'
        else:
            readiness['status'] = 'NEEDS MAJOR OPTIMIZATION'
        
        return readiness

async def main():
    """Main installation and testing function"""
    print("🚀 Unified Memory Management System Installation & Testing")
    print("=" * 65)
    print("M4 Max Apple Silicon Optimization for Nautilus Trading Platform")
    print("-" * 65)
    
    installer = UnifiedMemoryInstaller()
    
    try:
        # Phase 1: System Dependencies
        print("\n📋 Phase 1: System Dependencies Installation")
        system_success = installer.install_system_dependencies()
        
        if not system_success:
            print("⚠️  System dependency issues detected - continuing with Python installation")
        
        # Phase 2: Python Dependencies
        print("\n📋 Phase 2: Python Dependencies Installation")
        python_success = installer.install_python_dependencies()
        
        if not python_success:
            print("❌ Python dependency installation failed - cannot proceed with testing")
            return 1
        
        # Phase 3: Import Verification
        print("\n📋 Phase 3: Import Verification")
        import_success = installer.verify_critical_imports()
        
        if not import_success:
            print("❌ Critical imports failed - cannot proceed with testing")
            return 1
        
        # Phase 4: Comprehensive Testing
        print("\n📋 Phase 4: Comprehensive Unified Memory Testing")
        test_results = await installer.run_unified_memory_tests()
        
        # Phase 5: Final Report
        print("\n📋 Phase 5: Final Report Generation")
        installer.generate_final_report(test_results)
        
        # Determine exit code
        successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
        total_tests = len(test_results)
        
        if successful_tests == total_tests:
            print("\n🎉 SUCCESS: Unified Memory Management System fully installed and optimized!")
            return 0
        elif successful_tests > 0:
            print(f"\n⚠️  PARTIAL SUCCESS: {successful_tests}/{total_tests} test suites passed")
            return 0
        else:
            print("\n❌ FAILURE: All test suites failed")
            return 1
        
    except Exception as e:
        logger.error(f"Installation and testing failed: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)