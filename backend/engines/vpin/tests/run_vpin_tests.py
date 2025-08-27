#!/usr/bin/env python3
"""
VPIN Test Protocol Execution Script
Comprehensive test execution for VPIN Market Microstructure Engine
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))


class VPINTestRunner:
    """VPIN Test Protocol Execution Manager"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {
            "test_session": {
                "start_time": datetime.now().isoformat(),
                "test_protocol_version": "1.0.0",
                "environment": "development"
            },
            "test_suites": {},
            "summary": {}
        }
        
    def print_header(self, title: str):
        """Print formatted test section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
        
    def print_subheader(self, title: str):
        """Print formatted test subsection header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
        
    def run_test_suite(self, test_file: str, suite_name: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a specific test suite and capture results"""
        self.print_subheader(f"Running {suite_name}")
        
        test_path = self.test_dir / test_file
        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"Test file not found: {test_file}",
                "execution_time": 0
            }
            
        start_time = time.time()
        
        try:
            # Run pytest on the specific test file
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_path),
                "-v",
                "--tb=short",
                "-x",  # Stop on first failure for quicker feedback
                f"--timeout={timeout}"
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.test_dir.parent),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = "passed"
                print(f"✅ {suite_name} - All tests passed")
            elif result.returncode == 1:
                status = "failed"
                print(f"❌ {suite_name} - Some tests failed")
            else:
                status = "error"
                print(f"🔥 {suite_name} - Test execution error")
                
            # Extract test information from output
            output_lines = result.stdout.split('\n')
            test_count = 0
            passed_count = 0
            failed_count = 0
            
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Parse pytest summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed_count = int(parts[i-1])
                        elif part == "failed":
                            failed_count = int(parts[i-1])
                    test_count = passed_count + failed_count
                    
            return {
                "status": status,
                "execution_time": execution_time,
                "test_count": test_count,
                "passed": passed_count,
                "failed": failed_count,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"⏱️  {suite_name} - Test suite timed out after {timeout}s")
            return {
                "status": "timeout",
                "execution_time": execution_time,
                "reason": f"Tests timed out after {timeout} seconds"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"🔥 {suite_name} - Test execution failed: {e}")
            return {
                "status": "error",
                "execution_time": execution_time,
                "error": str(e)
            }
            
    def check_test_environment(self) -> Dict[str, Any]:
        """Check test environment and dependencies"""
        self.print_subheader("Environment Check")
        
        env_results = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "working_directory": str(Path.cwd()),
            "test_directory": str(self.test_dir),
            "dependencies": {},
            "test_files": {}
        }
        
        # Check Python version
        print(f"Python Version: {env_results['python_version']}")
        
        # Check required dependencies
        required_deps = [
            "pytest", "pytest-asyncio", "fastapi", "httpx", 
            "websockets", "numpy", "pandas", "asyncio"
        ]
        
        for dep in required_deps:
            try:
                __import__(dep.replace("-", "_"))
                env_results["dependencies"][dep] = "available"
                print(f"✅ {dep}")
            except ImportError:
                env_results["dependencies"][dep] = "missing"
                print(f"❌ {dep} - Missing")
                
        # Check test files
        test_files = [
            "test_vpin_unit.py",
            "test_vpin_api.py", 
            "test_vpin_performance.py",
            "test_vpin_e2e.py"
        ]
        
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                env_results["test_files"][test_file] = "found"
                print(f"✅ {test_file}")
            else:
                env_results["test_files"][test_file] = "missing"
                print(f"❌ {test_file} - Not found")
                
        return env_results
        
    def check_docker_environment(self) -> Dict[str, Any]:
        """Check Docker environment for integration tests"""
        self.print_subheader("Docker Environment Check")
        
        docker_results = {
            "docker_available": False,
            "vpin_container_running": False,
            "vpin_container_healthy": False,
            "api_accessible": False
        }
        
        # Check Docker availability
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_results["docker_available"] = True
                print(f"✅ Docker available: {result.stdout.strip()}")
            else:
                print("❌ Docker not available")
                return docker_results
        except Exception as e:
            print(f"❌ Docker check failed: {e}")
            return docker_results
            
        # Check VPIN container
        try:
            result = subprocess.run([
                "docker", "ps", "--filter", "name=nautilus-vpin-engine", 
                "--format", "{{.Names}}"
            ], capture_output=True, text=True, timeout=10)
            
            if "nautilus-vpin-engine" in result.stdout:
                docker_results["vpin_container_running"] = True
                print("✅ VPIN container is running")
                
                # Check container health
                import requests
                try:
                    response = requests.get("http://localhost:10000/health", timeout=5)
                    if response.status_code == 200:
                        docker_results["vpin_container_healthy"] = True
                        docker_results["api_accessible"] = True
                        print("✅ VPIN container is healthy and API is accessible")
                    else:
                        print(f"⚠️  VPIN container API returned: {response.status_code}")
                except Exception as e:
                    print(f"⚠️  VPIN container API not accessible: {e}")
            else:
                print("⚠️  VPIN container not running")
                
        except Exception as e:
            print(f"❌ Container check failed: {e}")
            
        return docker_results
        
    def run_full_test_protocol(self) -> Dict[str, Any]:
        """Execute the complete VPIN test protocol"""
        self.print_header("VPIN ENGINE COMPREHENSIVE TEST PROTOCOL")
        
        print(f"Test Protocol Version: 1.0.0")
        print(f"Test Start Time: {self.results['test_session']['start_time']}")
        print(f"Test Environment: {self.results['test_session']['environment']}")
        
        # Phase 1: Environment Setup
        self.print_header("PHASE 1: ENVIRONMENT VALIDATION")
        env_check = self.check_test_environment()
        docker_check = self.check_docker_environment()
        
        self.results["environment_check"] = env_check
        self.results["docker_check"] = docker_check
        
        # Phase 2: Unit Testing
        self.print_header("PHASE 2: UNIT TESTING")
        unit_results = self.run_test_suite("test_vpin_unit.py", "Unit Tests", timeout=180)
        self.results["test_suites"]["unit_tests"] = unit_results
        
        # Phase 3: API Testing
        self.print_header("PHASE 3: API ENDPOINT TESTING") 
        api_results = self.run_test_suite("test_vpin_api.py", "API Tests", timeout=120)
        self.results["test_suites"]["api_tests"] = api_results
        
        # Phase 4: Performance Testing
        self.print_header("PHASE 4: PERFORMANCE & HARDWARE ACCELERATION")
        perf_results = self.run_test_suite("test_vpin_performance.py", "Performance Tests", timeout=300)
        self.results["test_suites"]["performance_tests"] = perf_results
        
        # Phase 5: End-to-End Testing
        self.print_header("PHASE 5: END-TO-END INTEGRATION")
        e2e_results = self.run_test_suite("test_vpin_e2e.py", "End-to-End Tests", timeout=240)
        self.results["test_suites"]["e2e_tests"] = e2e_results
        
        # Generate summary
        self.generate_test_summary()
        
        # Final report
        self.print_final_report()
        
        return self.results
        
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        summary = {
            "total_suites": len(self.results["test_suites"]),
            "passed_suites": 0,
            "failed_suites": 0,
            "error_suites": 0,
            "skipped_suites": 0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "total_execution_time": 0,
            "overall_grade": "F"
        }
        
        for suite_name, suite_results in self.results["test_suites"].items():
            status = suite_results.get("status", "unknown")
            
            if status == "passed":
                summary["passed_suites"] += 1
            elif status == "failed":
                summary["failed_suites"] += 1
            elif status == "error" or status == "timeout":
                summary["error_suites"] += 1
            elif status == "skipped":
                summary["skipped_suites"] += 1
                
            # Aggregate test counts
            summary["total_tests"] += suite_results.get("test_count", 0)
            summary["passed_tests"] += suite_results.get("passed", 0)
            summary["failed_tests"] += suite_results.get("failed", 0)
            summary["total_execution_time"] += suite_results.get("execution_time", 0)
            
        # Calculate overall grade
        if summary["total_suites"] > 0:
            success_rate = summary["passed_suites"] / summary["total_suites"]
            if success_rate >= 0.9:
                summary["overall_grade"] = "A+"
            elif success_rate >= 0.8:
                summary["overall_grade"] = "A"
            elif success_rate >= 0.7:
                summary["overall_grade"] = "B+"
            elif success_rate >= 0.6:
                summary["overall_grade"] = "B"
            elif success_rate >= 0.5:
                summary["overall_grade"] = "C"
            else:
                summary["overall_grade"] = "F"
                
        self.results["summary"] = summary
        
    def print_final_report(self):
        """Print comprehensive final test report"""
        self.print_header("FINAL TEST REPORT")
        
        summary = self.results["summary"]
        
        print(f"📊 TEST EXECUTION SUMMARY")
        print(f"   Total Test Suites: {summary['total_suites']}")
        print(f"   Passed Suites: {summary['passed_suites']} ✅")
        print(f"   Failed Suites: {summary['failed_suites']} ❌")
        print(f"   Error Suites: {summary['error_suites']} 🔥")
        print(f"   Skipped Suites: {summary['skipped_suites']} ⏭️")
        print(f"")
        print(f"   Total Individual Tests: {summary['total_tests']}")
        print(f"   Passed Tests: {summary['passed_tests']} ✅")
        print(f"   Failed Tests: {summary['failed_tests']} ❌")
        print(f"")
        print(f"   Total Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"   Overall Grade: {summary['overall_grade']}")
        
        # Individual suite results
        print(f"\n📋 INDIVIDUAL SUITE RESULTS")
        for suite_name, suite_results in self.results["test_suites"].items():
            status = suite_results.get("status", "unknown")
            exec_time = suite_results.get("execution_time", 0)
            test_count = suite_results.get("test_count", 0)
            
            status_emoji = {
                "passed": "✅",
                "failed": "❌", 
                "error": "🔥",
                "timeout": "⏱️",
                "skipped": "⏭️"
            }.get(status, "❓")
            
            print(f"   {status_emoji} {suite_name}: {status.upper()} ({test_count} tests, {exec_time:.1f}s)")
            
        # Environment status
        print(f"\n🔧 ENVIRONMENT STATUS")
        env = self.results.get("environment_check", {})
        docker = self.results.get("docker_check", {})
        
        deps_available = sum(1 for dep in env.get("dependencies", {}).values() if dep == "available")
        total_deps = len(env.get("dependencies", {}))
        
        print(f"   Python Version: {env.get('python_version', 'unknown')}")
        print(f"   Dependencies: {deps_available}/{total_deps} available")
        print(f"   Docker Available: {'✅' if docker.get('docker_available') else '❌'}")
        print(f"   VPIN Container: {'✅' if docker.get('vpin_container_running') else '❌'}")
        print(f"   API Accessible: {'✅' if docker.get('api_accessible') else '❌'}")
        
        # Final assessment
        grade = summary['overall_grade']
        if grade in ['A+', 'A']:
            print(f"\n🎉 EXCELLENT! VPIN engine testing completed with grade {grade}")
            print("   All critical functionality is working correctly.")
        elif grade in ['B+', 'B']:
            print(f"\n👍 GOOD! VPIN engine testing completed with grade {grade}")
            print("   Most functionality is working, minor issues detected.")
        elif grade == 'C':
            print(f"\n⚠️  ACCEPTABLE! VPIN engine testing completed with grade {grade}")
            print("   Core functionality working but significant issues found.")
        else:
            print(f"\n❌ NEEDS WORK! VPIN engine testing completed with grade {grade}")
            print("   Major issues detected that need to be addressed.")
            
        print("\n" + "="*80)
        
    def save_results(self, output_file: Optional[str] = None):
        """Save test results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.test_dir / f"vpin_test_results_{timestamp}.json"
            
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 Test results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main test execution function"""
    runner = VPINTestRunner()
    
    try:
        results = runner.run_full_test_protocol()
        runner.save_results()
        
        # Exit with appropriate code
        summary = results.get("summary", {})
        if summary.get("failed_suites", 0) > 0 or summary.get("error_suites", 0) > 0:
            sys.exit(1)  # Indicate test failures
        else:
            sys.exit(0)  # All tests passed
            
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(2)
        
    except Exception as e:
        print(f"\n🔥 Test execution failed with error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()