"""
AI-Powered Testing Framework - Phase 8 Autonomous Operations
===========================================================

Provides intelligent test generation, automatic test case creation, and
AI-driven quality assurance with adaptive testing strategies.

Key Features:
- AI-powered test case generation from code analysis
- Intelligent mutation testing and edge case discovery
- Automated regression detection and test maintenance
- Smart test execution prioritization and parallelization
- Performance testing with ML-driven load patterns
- Visual testing with computer vision validation
"""

import asyncio
import json
import logging
import ast
import inspect
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import requests
import subprocess
from pathlib import Path
import coverage
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UI = "ui"
    API = "api"
    MUTATION = "mutation"
    PROPERTY = "property"
    VISUAL = "visual"

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

class TestPriority(Enum):
    """Test execution priority"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MAINTENANCE = 5

class CodeAnalysisResult(BaseModel):
    """Code analysis result for test generation"""
    file_path: str
    functions: List[Dict[str, Any]] = Field(default_factory=list)
    classes: List[Dict[str, Any]] = Field(default_factory=list)
    complexity_score: float = 0.0
    dependencies: List[str] = Field(default_factory=list)
    test_coverage: float = 0.0
    risk_areas: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_test_types: List[TestType] = Field(default_factory=list)

@dataclass
class GeneratedTestCase:
    """Generated test case definition"""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.UNIT
    target_function: str = ""
    test_code: str = ""
    test_data: List[Dict[str, Any]] = field(default_factory=list)
    assertions: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    priority: TestPriority = TestPriority.MEDIUM
    estimated_runtime: float = 1.0
    confidence_score: float = 0.5
    generated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class TestExecution:
    """Test execution result"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_id: str = ""
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    coverage_data: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

class AITestGenerator:
    """AI-powered test case generator"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai
        
        self.test_patterns = self._load_test_patterns()
        self.code_analyzer = CodeAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.test_templates = self._load_test_templates()
        
    def _load_test_patterns(self) -> Dict[str, List[str]]:
        """Load common test patterns for different scenarios"""
        return {
            'edge_cases': [
                'empty_input', 'null_input', 'negative_numbers', 
                'large_numbers', 'special_characters', 'unicode_characters'
            ],
            'boundary_conditions': [
                'min_value', 'max_value', 'zero_value', 'one_off_errors'
            ],
            'error_conditions': [
                'invalid_input', 'missing_parameters', 'type_errors',
                'permission_errors', 'network_errors'
            ],
            'performance_scenarios': [
                'large_datasets', 'concurrent_access', 'memory_pressure',
                'cpu_intensive_operations'
            ]
        }
    
    def _load_test_templates(self) -> Dict[TestType, str]:
        """Load test code templates for different test types"""
        return {
            TestType.UNIT: '''
import pytest
from unittest.mock import Mock, patch
from {module_path} import {function_name}

class Test{class_name}:
    
    def test_{test_name}_valid_input(self):
        """Test {function_name} with valid input"""
        # Arrange
        {setup_code}
        
        # Act
        result = {function_name}({test_parameters})
        
        # Assert
        {assertions}
    
    def test_{test_name}_edge_cases(self):
        """Test {function_name} edge cases"""
        {edge_case_tests}
    
    def test_{test_name}_error_conditions(self):
        """Test {function_name} error handling"""
        {error_tests}
''',
            TestType.INTEGRATION: '''
import pytest
import asyncio
from {module_path} import {class_name}

class Test{class_name}Integration:
    
    @pytest.fixture
    async def setup_integration_environment(self):
        """Setup integration test environment"""
        {integration_setup}
        yield
        {integration_teardown}
    
    async def test_{test_name}_integration_flow(self, setup_integration_environment):
        """Test complete integration flow"""
        {integration_test_code}
''',
            TestType.API: '''
import pytest
import requests
from fastapi.testclient import TestClient
from {module_path} import app

class Test{class_name}API:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_{test_name}_api_endpoint(self, client):
        """Test API endpoint {endpoint_path}"""
        response = client.{http_method}("{endpoint_path}", {request_data})
        
        assert response.status_code == {expected_status}
        {response_assertions}
'''
        }
    
    async def analyze_code_for_testing(self, file_path: str) -> CodeAnalysisResult:
        """Analyze code file to determine testing requirements"""
        try:
            analysis = CodeAnalysisResult(file_path=file_path)
            
            # Read and parse code
            with open(file_path, 'r') as f:
                code_content = f.read()
            
            try:
                tree = ast.parse(code_content)
            except SyntaxError as e:
                logger.error(f"Syntax error in {file_path}: {e}")
                return analysis
            
            # Analyze functions and classes
            analysis.functions = self._extract_functions(tree, code_content)
            analysis.classes = self._extract_classes(tree, code_content)
            
            # Calculate complexity
            analysis.complexity_score = self._calculate_complexity(tree)
            
            # Extract dependencies
            analysis.dependencies = self._extract_dependencies(tree)
            
            # Identify risk areas
            analysis.risk_areas = self._identify_risk_areas(tree, code_content)
            
            # Suggest test types
            analysis.suggested_test_types = self._suggest_test_types(analysis)
            
            # Get current test coverage if available
            analysis.test_coverage = await self._get_test_coverage(file_path)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing code {file_path}: {e}")
            return CodeAnalysisResult(file_path=file_path)
    
    def _extract_functions(self, tree: ast.AST, code_content: str) -> List[Dict[str, Any]]:
        """Extract function definitions from AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'complexity': self._calculate_function_complexity(node),
                    'returns': self._analyze_return_statements(node)
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST, code_content: str) -> List[Dict[str, Any]]:
        """Extract class definitions from AST"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                
                class_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'docstring': ast.get_docstring(node),
                    'methods': [method.name for method in methods],
                    'base_classes': [self._get_base_class_name(base) for base in node.bases],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'method_count': len(methods)
                }
                classes.append(class_info)
        
        return classes
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity of the code"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        # Normalize by number of functions
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        return complexity / max(function_count, 1)
    
    def _identify_risk_areas(self, tree: ast.AST, code_content: str) -> List[Dict[str, Any]]:
        """Identify high-risk areas that need thorough testing"""
        risk_areas = []
        
        # Look for error-prone patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                risk_areas.append({
                    'type': 'exception_handling',
                    'line': node.lineno,
                    'description': 'Exception handling code needs thorough testing',
                    'severity': 'medium'
                })
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Network calls, file operations, database operations
                    if any(risky in str(node.func.attr).lower() 
                          for risky in ['request', 'open', 'execute', 'query', 'connect']):
                        risk_areas.append({
                            'type': 'external_dependency',
                            'line': node.lineno,
                            'description': f'External dependency call: {node.func.attr}',
                            'severity': 'high'
                        })
        
        # Check for complex logic patterns
        nested_depth = self._calculate_nesting_depth(tree)
        if nested_depth > 3:
            risk_areas.append({
                'type': 'complex_logic',
                'line': 0,
                'description': f'High nesting depth ({nested_depth}) indicates complex logic',
                'severity': 'medium'
            })
        
        return risk_areas
    
    async def generate_test_cases(self, analysis: CodeAnalysisResult, 
                                test_types: Optional[List[TestType]] = None) -> List[GeneratedTestCase]:
        """Generate test cases based on code analysis"""
        test_cases = []
        
        target_types = test_types or analysis.suggested_test_types
        
        for test_type in target_types:
            if test_type == TestType.UNIT:
                test_cases.extend(await self._generate_unit_tests(analysis))
            elif test_type == TestType.INTEGRATION:
                test_cases.extend(await self._generate_integration_tests(analysis))
            elif test_type == TestType.API:
                test_cases.extend(await self._generate_api_tests(analysis))
            elif test_type == TestType.PERFORMANCE:
                test_cases.extend(await self._generate_performance_tests(analysis))
            elif test_type == TestType.MUTATION:
                test_cases.extend(await self._generate_mutation_tests(analysis))
        
        # Use AI to enhance test cases if available
        if self.openai_client:
            test_cases = await self._enhance_tests_with_ai(analysis, test_cases)
        
        return test_cases
    
    async def _generate_unit_tests(self, analysis: CodeAnalysisResult) -> List[GeneratedTestCase]:
        """Generate unit test cases"""
        test_cases = []
        
        for func in analysis.functions:
            # Skip private methods and special methods
            if func['name'].startswith('_') and not func['name'].startswith('__'):
                continue
            
            # Generate test for normal case
            normal_test = GeneratedTestCase(
                name=f"test_{func['name']}_normal_case",
                description=f"Test {func['name']} with valid input",
                test_type=TestType.UNIT,
                target_function=func['name'],
                priority=TestPriority.HIGH if func['complexity'] > 3 else TestPriority.MEDIUM
            )
            
            # Generate test code based on function signature
            normal_test.test_code = self._generate_unit_test_code(func, analysis)
            normal_test.confidence_score = 0.8
            
            test_cases.append(normal_test)
            
            # Generate edge case tests
            edge_test = GeneratedTestCase(
                name=f"test_{func['name']}_edge_cases",
                description=f"Test {func['name']} edge cases and boundary conditions",
                test_type=TestType.UNIT,
                target_function=func['name'],
                priority=TestPriority.HIGH
            )
            
            edge_test.test_code = self._generate_edge_case_tests(func, analysis)
            edge_test.confidence_score = 0.7
            
            test_cases.append(edge_test)
            
            # Generate error condition tests
            if func['complexity'] > 2:
                error_test = GeneratedTestCase(
                    name=f"test_{func['name']}_error_conditions",
                    description=f"Test {func['name']} error handling",
                    test_type=TestType.UNIT,
                    target_function=func['name'],
                    priority=TestPriority.MEDIUM
                )
                
                error_test.test_code = self._generate_error_condition_tests(func, analysis)
                error_test.confidence_score = 0.6
                
                test_cases.append(error_test)
        
        return test_cases
    
    def _generate_unit_test_code(self, func: Dict[str, Any], analysis: CodeAnalysisResult) -> str:
        """Generate unit test code for a function"""
        template = self.test_templates[TestType.UNIT]
        
        # Extract module information
        module_path = analysis.file_path.replace('/', '.').replace('.py', '')
        class_name = func['name'].title().replace('_', '')
        
        # Generate test parameters based on function arguments
        test_parameters = self._generate_test_parameters(func['args'])
        
        # Generate assertions based on expected behavior
        assertions = self._generate_assertions(func)
        
        # Generate setup code if needed
        setup_code = self._generate_setup_code(func, analysis)
        
        return template.format(
            module_path=module_path,
            function_name=func['name'],
            class_name=class_name,
            test_name=func['name'],
            test_parameters=test_parameters,
            assertions=assertions,
            setup_code=setup_code,
            edge_case_tests=self._generate_edge_case_code(func),
            error_tests=self._generate_error_test_code(func)
        )
    
    def _generate_test_parameters(self, args: List[str]) -> str:
        """Generate test parameters based on function arguments"""
        if not args:
            return ""
        
        # Remove 'self' or 'cls' for methods
        filtered_args = [arg for arg in args if arg not in ['self', 'cls']]
        
        # Generate sample values based on argument names
        param_values = []
        for arg in filtered_args:
            if 'id' in arg.lower():
                param_values.append(f"{arg}=1")
            elif 'name' in arg.lower():
                param_values.append(f'{arg}="test_name"')
            elif 'count' in arg.lower() or 'size' in arg.lower():
                param_values.append(f"{arg}=10")
            elif 'url' in arg.lower():
                param_values.append(f'{arg}="https://example.com"')
            elif 'email' in arg.lower():
                param_values.append(f'{arg}="test@example.com"')
            else:
                param_values.append(f'{arg}="test_value"')
        
        return ", ".join(param_values)
    
    async def _enhance_tests_with_ai(self, analysis: CodeAnalysisResult, 
                                   test_cases: List[GeneratedTestCase]) -> List[GeneratedTestCase]:
        """Enhance generated tests using AI/LLM"""
        if not self.openai_client:
            return test_cases
        
        enhanced_tests = []
        
        for test_case in test_cases:
            try:
                prompt = f"""
Improve this test case for better coverage and robustness:

Function: {test_case.target_function}
Current test code:
{test_case.test_code}

Code context:
- File: {analysis.file_path}
- Complexity: {analysis.complexity_score}
- Risk areas: {analysis.risk_areas}

Please provide:
1. Improved test code with better assertions
2. Additional edge cases to test
3. Mock strategies for dependencies
4. Performance considerations

Return only the improved test code.
"""
                
                response = await self.openai_client.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                enhanced_code = response.choices[0].message.content
                test_case.test_code = enhanced_code
                test_case.confidence_score = min(test_case.confidence_score + 0.2, 1.0)
                
            except Exception as e:
                logger.warning(f"Could not enhance test with AI: {e}")
            
            enhanced_tests.append(test_case)
        
        return enhanced_tests
    
    async def _generate_performance_tests(self, analysis: CodeAnalysisResult) -> List[GeneratedTestCase]:
        """Generate performance test cases"""
        test_cases = []
        
        # Look for functions that might have performance implications
        for func in analysis.functions:
            if any(keyword in func['name'].lower() 
                  for keyword in ['process', 'calculate', 'analyze', 'compute', 'search']):
                
                perf_test = GeneratedTestCase(
                    name=f"test_{func['name']}_performance",
                    description=f"Test {func['name']} performance characteristics",
                    test_type=TestType.PERFORMANCE,
                    target_function=func['name'],
                    priority=TestPriority.MEDIUM
                )
                
                perf_test.test_code = self._generate_performance_test_code(func, analysis)
                perf_test.confidence_score = 0.6
                
                test_cases.append(perf_test)
        
        return test_cases
    
    def _generate_performance_test_code(self, func: Dict[str, Any], analysis: CodeAnalysisResult) -> str:
        """Generate performance test code"""
        return f'''
import pytest
import time
from {analysis.file_path.replace('/', '.').replace('.py', '')} import {func['name']}

def test_{func['name']}_performance():
    """Test {func['name']} performance"""
    # Arrange
    test_data = [i for i in range(1000)]  # Large dataset
    
    # Act
    start_time = time.time()
    result = {func['name']}(test_data)
    end_time = time.time()
    
    # Assert
    execution_time = end_time - start_time
    assert execution_time < 1.0, f"Function took too long: {{execution_time:.3f}}s"
    assert result is not None

def test_{func['name']}_memory_usage():
    """Test {func['name']} memory usage"""
    import tracemalloc
    
    tracemalloc.start()
    
    # Act
    result = {func['name']}("test_input")
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert reasonable memory usage (< 10MB)
    assert peak < 10 * 1024 * 1024, f"Memory usage too high: {{peak / 1024 / 1024:.2f}}MB"
'''

class CodeAnalyzer:
    """Analyzes code for testing requirements"""
    
    def __init__(self):
        self.complexity_threshold = 5
        self.risk_patterns = [
            r'except\s*:',  # Bare except
            r'eval\(',      # eval usage
            r'exec\(',      # exec usage
            r'__import__\(',  # Dynamic imports
        ]
    
    def calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node: ast.AST, current_depth: int = 0) -> int:
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)

class IntelligentTestExecutor:
    """Intelligent test execution with prioritization and parallelization"""
    
    def __init__(self, max_parallel_tests: int = 4):
        self.max_parallel_tests = max_parallel_tests
        self.test_history: Dict[str, List[TestExecution]] = {}
        self.coverage_data = coverage.Coverage()
        
    async def execute_test_suite(self, test_cases: List[GeneratedTestCase]) -> List[TestExecution]:
        """Execute test suite with intelligent prioritization"""
        # Prioritize tests
        prioritized_tests = self._prioritize_tests(test_cases)
        
        # Group tests for parallel execution
        test_groups = self._group_tests_for_execution(prioritized_tests)
        
        # Execute test groups
        all_executions = []
        
        for group in test_groups:
            group_executions = await self._execute_test_group(group)
            all_executions.extend(group_executions)
            
            # Stop on critical failures if needed
            if self._should_stop_execution(group_executions):
                logger.warning("Stopping test execution due to critical failures")
                break
        
        return all_executions
    
    def _prioritize_tests(self, test_cases: List[GeneratedTestCase]) -> List[GeneratedTestCase]:
        """Prioritize tests based on various factors"""
        def priority_score(test_case: GeneratedTestCase) -> Tuple[int, float, float]:
            # Primary: Priority enum value (lower is higher priority)
            # Secondary: Confidence score (higher is better)
            # Tertiary: Historical failure rate (higher failure rate = higher priority)
            
            historical_failures = self._get_historical_failure_rate(test_case.test_id)
            
            return (
                test_case.priority.value,
                -test_case.confidence_score,  # Negative for reverse sort
                -historical_failures  # Negative for reverse sort
            )
        
        return sorted(test_cases, key=priority_score)
    
    def _group_tests_for_execution(self, test_cases: List[GeneratedTestCase]) -> List[List[GeneratedTestCase]]:
        """Group tests for optimal parallel execution"""
        groups = []
        current_group = []
        current_group_runtime = 0.0
        max_group_runtime = 60.0  # Max 60 seconds per group
        
        for test_case in test_cases:
            estimated_runtime = test_case.estimated_runtime
            
            # Start new group if current would exceed limits
            if (len(current_group) >= self.max_parallel_tests or 
                current_group_runtime + estimated_runtime > max_group_runtime):
                
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_group_runtime = 0.0
            
            current_group.append(test_case)
            current_group_runtime += estimated_runtime
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _execute_test_group(self, test_group: List[GeneratedTestCase]) -> List[TestExecution]:
        """Execute a group of tests in parallel"""
        tasks = []
        
        for test_case in test_group:
            task = asyncio.create_task(self._execute_single_test(test_case))
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _execute_single_test(self, test_case: GeneratedTestCase) -> TestExecution:
        """Execute a single test case"""
        execution = TestExecution(
            test_id=test_case.test_id,
            start_time=datetime.now()
        )
        
        try:
            execution.status = TestStatus.RUNNING
            
            # Write test to temporary file
            test_file = await self._write_test_to_file(test_case)
            
            # Start coverage monitoring
            self.coverage_data.start()
            
            # Execute test using pytest
            result = await self._run_pytest(test_file)
            
            # Stop coverage monitoring
            self.coverage_data.stop()
            execution.coverage_data = self._get_coverage_data()
            
            # Process results
            if result.returncode == 0:
                execution.status = TestStatus.PASSED
                execution.result = {"success": True}
            else:
                execution.status = TestStatus.FAILED
                execution.error_message = result.stderr.decode() if result.stderr else "Test failed"
                execution.result = {"success": False, "output": result.stdout.decode()}
            
        except asyncio.TimeoutError:
            execution.status = TestStatus.TIMEOUT
            execution.error_message = "Test execution timed out"
            
        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_message = str(e)
            
        finally:
            execution.end_time = datetime.now()
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Store execution history
            if test_case.test_id not in self.test_history:
                self.test_history[test_case.test_id] = []
            self.test_history[test_case.test_id].append(execution)
            
            # Cleanup
            await self._cleanup_test_files(test_case.test_id)
        
        return execution
    
    async def _write_test_to_file(self, test_case: GeneratedTestCase) -> str:
        """Write test case to a temporary file"""
        test_dir = Path(f"/tmp/ai_tests/{test_case.test_id}")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / "test_generated.py"
        
        with open(test_file, 'w') as f:
            f.write(test_case.setup_code + "\n\n")
            f.write(test_case.test_code + "\n\n")
            f.write(test_case.teardown_code)
        
        return str(test_file)
    
    async def _run_pytest(self, test_file: str) -> subprocess.CompletedProcess:
        """Run pytest on the test file"""
        cmd = [
            "python", "-m", "pytest", 
            test_file,
            "-v", 
            "--tb=short",
            f"--timeout={30}"  # 30 second timeout per test
        ]
        
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await result.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    def _get_historical_failure_rate(self, test_id: str) -> float:
        """Get historical failure rate for a test"""
        history = self.test_history.get(test_id, [])
        if not history:
            return 0.0
        
        failures = len([exec for exec in history if exec.status == TestStatus.FAILED])
        return failures / len(history)
    
    def _should_stop_execution(self, executions: List[TestExecution]) -> bool:
        """Determine if test execution should be stopped due to failures"""
        critical_failures = [
            exec for exec in executions 
            if exec.status == TestStatus.FAILED and 'critical' in exec.test_id.lower()
        ]
        
        return len(critical_failures) > 0
    
    def generate_test_report(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Generate comprehensive test execution report"""
        total_tests = len(executions)
        passed_tests = len([e for e in executions if e.status == TestStatus.PASSED])
        failed_tests = len([e for e in executions if e.status == TestStatus.FAILED])
        
        # Calculate metrics
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        total_runtime = sum(e.duration for e in executions if e.duration)
        avg_runtime = total_runtime / total_tests if total_tests > 0 else 0
        
        # Coverage analysis
        coverage_summary = self._analyze_coverage(executions)
        
        return {
            'execution_summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'pass_rate': f"{pass_rate:.1f}%",
                'total_runtime': f"{total_runtime:.2f}s",
                'average_runtime': f"{avg_runtime:.2f}s"
            },
            'coverage_summary': coverage_summary,
            'failed_tests': [
                {
                    'test_id': e.test_id,
                    'error': e.error_message,
                    'duration': e.duration
                }
                for e in executions if e.status == TestStatus.FAILED
            ],
            'performance_metrics': {
                'slowest_tests': sorted(
                    [(e.test_id, e.duration) for e in executions if e.duration],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        }

class VisualTestingFramework:
    """AI-powered visual regression testing"""
    
    def __init__(self):
        self.driver = None
        self.baseline_images: Dict[str, np.ndarray] = {}
        self.similarity_threshold = 0.95
        
    async def setup_browser(self, headless: bool = True):
        """Setup Selenium WebDriver"""
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1920, 1080)
    
    async def capture_baseline(self, url: str, test_name: str) -> str:
        """Capture baseline screenshot for visual comparison"""
        if not self.driver:
            await self.setup_browser()
        
        self.driver.get(url)
        await asyncio.sleep(2)  # Wait for page load
        
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        
        # Store baseline
        baseline_path = f"/tmp/visual_baselines/{test_name}_baseline.png"
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(baseline_path)
        
        # Store in memory for comparison
        self.baseline_images[test_name] = np.array(image)
        
        return baseline_path
    
    async def compare_visual(self, url: str, test_name: str) -> Dict[str, Any]:
        """Compare current page with baseline"""
        if not self.driver:
            await self.setup_browser()
        
        self.driver.get(url)
        await asyncio.sleep(2)
        
        # Capture current screenshot
        screenshot = self.driver.get_screenshot_as_png()
        current_image = np.array(Image.open(io.BytesIO(screenshot)))
        
        # Get baseline
        baseline = self.baseline_images.get(test_name)
        if baseline is None:
            return {"error": "No baseline found", "similarity": 0.0}
        
        # Calculate similarity
        similarity = self._calculate_image_similarity(baseline, current_image)
        
        # Generate diff image if different
        diff_image = None
        if similarity < self.similarity_threshold:
            diff_image = self._generate_diff_image(baseline, current_image)
        
        return {
            "similarity": similarity,
            "passed": similarity >= self.similarity_threshold,
            "diff_image": diff_image,
            "threshold": self.similarity_threshold
        }
    
    def _calculate_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate similarity between two images"""
        # Ensure same dimensions
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        # Calculate mean squared error
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Convert to similarity score (0-1)
        max_pixel_value = 255.0
        max_mse = max_pixel_value ** 2
        similarity = 1 - (mse / max_mse)
        
        return similarity
    
    def _generate_diff_image(self, baseline: np.ndarray, current: np.ndarray) -> str:
        """Generate visual diff highlighting differences"""
        # Calculate absolute difference
        diff = np.abs(baseline.astype(float) - current.astype(float))
        
        # Highlight significant differences in red
        diff_highlighted = current.copy()
        threshold = 50  # Pixel difference threshold
        
        mask = np.any(diff > threshold, axis=2)
        diff_highlighted[mask] = [255, 0, 0]  # Red for differences
        
        # Save diff image
        diff_path = f"/tmp/visual_diffs/diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Path(diff_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(diff_highlighted).save(diff_path)
        
        return diff_path

# Main AI-Powered Testing Framework
class AIPoweredTestingFramework:
    """Main AI-powered testing framework orchestrator"""
    
    def __init__(self):
        self.test_generator = AITestGenerator()
        self.test_executor = IntelligentTestExecutor()
        self.visual_tester = VisualTestingFramework()
        self.generated_tests: Dict[str, List[GeneratedTestCase]] = {}
        self.test_reports: List[Dict[str, Any]] = []
        
    async def analyze_and_test_codebase(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze entire codebase and generate comprehensive test suite"""
        logger.info(f"Analyzing codebase: {codebase_path}")
        
        # Find all Python files
        python_files = list(Path(codebase_path).rglob("*.py"))
        
        # Analyze each file
        analysis_results = []
        for file_path in python_files:
            if 'test_' in str(file_path) or '__pycache__' in str(file_path):
                continue  # Skip existing tests and cache
            
            analysis = await self.test_generator.analyze_code_for_testing(str(file_path))
            analysis_results.append(analysis)
        
        # Generate tests for all analyzed code
        all_test_cases = []
        for analysis in analysis_results:
            test_cases = await self.test_generator.generate_test_cases(analysis)
            all_test_cases.extend(test_cases)
            self.generated_tests[analysis.file_path] = test_cases
        
        # Execute generated tests
        test_executions = await self.test_executor.execute_test_suite(all_test_cases)
        
        # Generate comprehensive report
        report = self.test_executor.generate_test_report(test_executions)
        report['analysis_summary'] = {
            'files_analyzed': len(analysis_results),
            'total_functions': sum(len(a.functions) for a in analysis_results),
            'total_classes': sum(len(a.classes) for a in analysis_results),
            'average_complexity': np.mean([a.complexity_score for a in analysis_results]),
            'high_risk_areas': sum(len(a.risk_areas) for a in analysis_results)
        }
        
        self.test_reports.append(report)
        
        return report
    
    async def continuous_testing_mode(self, codebase_path: str, watch_interval: int = 300):
        """Run in continuous testing mode, monitoring for code changes"""
        logger.info("Starting continuous testing mode")
        
        last_modified_times = {}
        
        while True:
            try:
                # Check for modified files
                python_files = list(Path(codebase_path).rglob("*.py"))
                modified_files = []
                
                for file_path in python_files:
                    if 'test_' in str(file_path):
                        continue
                    
                    mtime = file_path.stat().st_mtime
                    if str(file_path) not in last_modified_times or mtime > last_modified_times[str(file_path)]:
                        modified_files.append(file_path)
                        last_modified_times[str(file_path)] = mtime
                
                # Test modified files
                if modified_files:
                    logger.info(f"Testing {len(modified_files)} modified files")
                    
                    for file_path in modified_files:
                        analysis = await self.test_generator.analyze_code_for_testing(str(file_path))
                        test_cases = await self.test_generator.generate_test_cases(analysis)
                        executions = await self.test_executor.execute_test_suite(test_cases)
                        
                        # Log results
                        passed = len([e for e in executions if e.status == TestStatus.PASSED])
                        total = len(executions)
                        logger.info(f"File {file_path}: {passed}/{total} tests passed")
                
                await asyncio.sleep(watch_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous testing: {e}")
                await asyncio.sleep(60)

# Example usage
async def example_usage():
    """Example usage of AI-Powered Testing Framework"""
    framework = AIPoweredTestingFramework()
    
    # Analyze and test a specific file
    analysis = await framework.test_generator.analyze_code_for_testing("example_module.py")
    print(f"Analysis: {len(analysis.functions)} functions, complexity: {analysis.complexity_score}")
    
    # Generate tests
    test_cases = await framework.test_generator.generate_test_cases(analysis)
    print(f"Generated {len(test_cases)} test cases")
    
    # Execute tests
    executions = await framework.test_executor.execute_test_suite(test_cases)
    
    # Generate report
    report = framework.test_executor.generate_test_report(executions)
    print(f"Test results: {report['execution_summary']}")

if __name__ == "__main__":
    asyncio.run(example_usage())