#!/usr/bin/env python3
"""
Nautilus Trading Platform - Comprehensive System Performance Assessment
M4 Max Integration Validation & Production Deployment Recommendation

Consolidates all benchmark results to provide:
- Overall system performance assessment
- Bottleneck identification
- Production deployment recommendation
- M4 Max optimization validation
"""

import json
import glob
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveSystemAssessment:
    """Comprehensive system assessment analyzer"""
    
    def __init__(self, results_directory: str):
        self.results_directory = results_directory
        self.benchmark_results = {}
        self.performance_grades = {}
        self.bottlenecks = []
        self.optimization_status = {}
        
    def load_all_benchmark_results(self):
        """Load all benchmark result files"""
        logger.info("Loading all benchmark results...")
        
        # Pattern matching for result files
        result_patterns = [
            "*system_integration_results_*.json",
            "*container_performance_results_*.json", 
            "*production_workload_results_*.json",
            "*reliability_stress_results_*.json",
            "*neural_engine_optimization_results_*.json",
            "*metal_gpu_benchmark_results_*.json",
            "*trading_benchmark_results_*.json",
            "*production_assessment_*.json"
        ]
        
        for pattern in result_patterns:
            files = glob.glob(os.path.join(self.results_directory, pattern))
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract test type from filename
                    filename = os.path.basename(file_path)
                    if 'system_integration' in filename:
                        self.benchmark_results['system_integration'] = data
                    elif 'container_performance' in filename:
                        self.benchmark_results['container_architecture'] = data
                    elif 'production_workload' in filename:
                        self.benchmark_results['production_workload'] = data
                    elif 'reliability_stress' in filename:
                        self.benchmark_results['reliability_stress'] = data
                    elif 'neural_engine_optimization' in filename:
                        self.benchmark_results['neural_engine'] = data
                    elif 'metal_gpu_benchmark' in filename:
                        self.benchmark_results['metal_gpu'] = data
                    elif 'trading_benchmark' in filename:
                        self.benchmark_results['trading_models'] = data
                    elif 'production_assessment' in filename:
                        self.benchmark_results['production_readiness'] = data
                        
                    logger.info(f"Loaded: {filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    
        logger.info(f"Loaded {len(self.benchmark_results)} benchmark result sets")
        
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance"""
        logger.info("Analyzing overall system performance...")
        
        analysis = {
            'performance_overview': self._analyze_performance_overview(),
            'resource_utilization': self._analyze_resource_utilization(), 
            'container_architecture': self._analyze_container_architecture(),
            'm4_max_optimization': self._analyze_m4_max_optimization(),
            'production_readiness': self._analyze_production_readiness(),
            'bottlenecks_and_recommendations': self._identify_bottlenecks_and_recommendations()
        }
        
        return analysis
        
    def _analyze_performance_overview(self) -> Dict[str, Any]:
        """Analyze overall performance overview"""
        grades = []
        success_rates = []
        
        # Collect grades from all tests
        for test_name, results in self.benchmark_results.items():
            if isinstance(results, dict):
                # Extract grades from various result structures
                if 'summary' in results and 'overall_performance' in results['summary']:
                    grade = results['summary']['overall_performance'].get('performance_grade')
                    if grade:
                        grades.append(grade)
                        
                if 'overall_performance' in results:
                    grade = results['overall_performance'].get('overall_grade')
                    if grade:
                        grades.append(grade)
                        
                # Extract success rates
                if 'production_readiness' in results:
                    success = results['production_readiness'].get('ready_for_production') or \
                             results['production_readiness'].get('reliability_validated')
                    if success is not None:
                        success_rates.append(1.0 if success else 0.0)
                        
                if 'summary' in results and 'overall_success' in results['summary']:
                    success_rates.append(1.0 if results['summary']['overall_success'] else 0.0)
                    
        # Calculate performance metrics
        grade_values = {'A+': 100, 'A': 95, 'B+': 85, 'B': 80, 'C': 70, 'D': 60}
        numeric_grades = [grade_values.get(g, 60) for g in grades if g in grade_values]
        
        avg_performance = np.mean(numeric_grades) if numeric_grades else 60
        avg_success_rate = np.mean(success_rates) if success_rates else 0.0
        
        # Overall grade
        if avg_performance >= 95:
            overall_grade = "A+"
        elif avg_performance >= 90:
            overall_grade = "A"  
        elif avg_performance >= 85:
            overall_grade = "B+"
        elif avg_performance >= 80:
            overall_grade = "B"
        elif avg_performance >= 70:
            overall_grade = "C"
        else:
            overall_grade = "D"
            
        return {
            'overall_grade': overall_grade,
            'average_performance_score': avg_performance,
            'overall_success_rate': avg_success_rate,
            'individual_grades': grades,
            'tests_analyzed': len(self.benchmark_results),
            'performance_consistency': np.std(numeric_grades) if len(numeric_grades) > 1 else 0
        }
        
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze system resource utilization"""
        cpu_usage = []
        memory_usage = []
        throughput_metrics = []
        
        # Extract resource metrics from all tests
        for test_name, results in self.benchmark_results.items():
            if isinstance(results, dict):
                # System integration results
                if 'benchmark_results' in results:
                    for result in results['benchmark_results']:
                        if result.get('cpu_usage'):
                            cpu_usage.append(result['cpu_usage'])
                        if result.get('memory_usage'):
                            memory_usage.append(result['memory_usage'])
                        if result.get('throughput'):
                            throughput_metrics.append(result['throughput'])
                            
                # Production workload results
                if 'individual_results' in results:
                    for test_result in results['individual_results'].values():
                        if isinstance(test_result, dict) and 'resource_usage' in test_result:
                            res_usage = test_result['resource_usage']
                            if 'avg_cpu_percent' in res_usage:
                                cpu_usage.append(res_usage['avg_cpu_percent'])
                            if 'avg_memory_percent' in res_usage:
                                memory_usage.append(res_usage['avg_memory_percent'])
                                
                # Other resource metrics
                if 'system_utilization' in results:
                    sys_util = results['system_utilization']
                    if 'avg_cpu_usage' in sys_util:
                        cpu_usage.append(sys_util['avg_cpu_usage'])
                    if 'avg_memory_usage' in sys_util:
                        memory_usage.append(sys_util['avg_memory_usage'])
                    if 'avg_throughput' in sys_util:
                        throughput_metrics.append(sys_util['avg_throughput'])
                        
        # Calculate resource statistics
        resource_stats = {
            'cpu_utilization': {
                'average': np.mean(cpu_usage) if cpu_usage else 0,
                'maximum': max(cpu_usage) if cpu_usage else 0,
                'minimum': min(cpu_usage) if cpu_usage else 0,
                'efficiency': 'High' if cpu_usage and np.mean(cpu_usage) > 50 and max(cpu_usage) < 90 else 'Moderate'
            },
            'memory_utilization': {
                'average': np.mean(memory_usage) if memory_usage else 0,
                'maximum': max(memory_usage) if memory_usage else 0,
                'minimum': min(memory_usage) if memory_usage else 0,
                'efficiency': 'High' if memory_usage and np.mean(memory_usage) > 30 and max(memory_usage) < 85 else 'Moderate'
            },
            'throughput_analysis': {
                'average': np.mean(throughput_metrics) if throughput_metrics else 0,
                'maximum': max(throughput_metrics) if throughput_metrics else 0,
                'samples_analyzed': len(throughput_metrics)
            },
            'resource_optimization_status': self._assess_resource_optimization(cpu_usage, memory_usage)
        }
        
        return resource_stats
        
    def _analyze_container_architecture(self) -> Dict[str, Any]:
        """Analyze container architecture performance"""
        container_analysis = {
            'architecture_validated': False,
            'container_health_rate': 0.0,
            'startup_performance': 'Unknown',
            'inter_container_communication': 'Unknown',
            'scaling_capability': 'Unknown'
        }
        
        if 'container_architecture' in self.benchmark_results:
            container_results = self.benchmark_results['container_architecture']
            
            if 'summary' in container_results:
                summary = container_results['summary']
                container_analysis.update({
                    'architecture_validated': summary.get('overall_success', False),
                    'container_health_rate': summary.get('container_health_rate', 0.0),
                    'performance_grade': summary.get('performance_grade', 'D'),
                    'total_containers': summary.get('total_containers', 0),
                    'running_containers': summary.get('running_containers', 0)
                })
                
            # Analyze specific performance aspects
            if 'startup_performance' in container_results:
                startup = container_results['startup_performance']
                if isinstance(startup, dict) and not startup.get('error'):
                    total_restart_time = sum(
                        result.get('total_restart_time', 0) 
                        for result in startup.values() 
                        if isinstance(result, dict)
                    )
                    container_analysis['startup_performance'] = (
                        'Excellent' if total_restart_time < 30 else
                        'Good' if total_restart_time < 60 else
                        'Needs Improvement'
                    )
                    
            if 'inter_container_communication' in container_results:
                comm = container_results['inter_container_communication']
                if isinstance(comm, dict):
                    basic_success = all(
                        test.get('success', False) 
                        for test in comm.get('basic_communication', [])
                    )
                    engine_success = all(
                        test.get('success', False) 
                        for test in comm.get('engine_communication', [])
                    )
                    container_analysis['inter_container_communication'] = (
                        'Excellent' if basic_success and engine_success else
                        'Good' if basic_success or engine_success else
                        'Needs Improvement'
                    )
                    
        return container_analysis
        
    def _analyze_m4_max_optimization(self) -> Dict[str, Any]:
        """Analyze M4 Max optimization validation"""
        m4_optimization = {
            'cpu_optimization_validated': False,
            'memory_bandwidth_optimized': False,
            'gpu_utilization_validated': False,
            'neural_engine_validated': False,
            'unified_memory_validated': False,
            'overall_m4_optimization_grade': 'D'
        }
        
        # Check system integration results
        if 'system_integration' in self.benchmark_results:
            sys_results = self.benchmark_results['system_integration']
            if 'summary' in sys_results and 'm4_max_optimization_validation' in sys_results['summary']:
                m4_validation = sys_results['summary']['m4_max_optimization_validation']
                m4_optimization.update({
                    'cpu_optimization_validated': m4_validation.get('cpu_optimization_validated', False),
                    'memory_bandwidth_optimized': m4_validation.get('memory_bandwidth_validated', False),
                    'gpu_utilization_validated': m4_validation.get('gpu_utilization_validated', False),
                    'neural_engine_validated': m4_validation.get('neural_engine_validated', False),
                    'unified_memory_validated': m4_validation.get('integrated_performance_validated', False)
                })
                
        # Check individual optimization results
        optimization_components = 0
        optimized_components = 0
        
        if 'neural_engine' in self.benchmark_results:
            optimization_components += 1
            neural_results = self.benchmark_results['neural_engine']
            if neural_results.get('overall_performance', {}).get('performance_grade') in ['A+', 'A', 'B+']:
                optimized_components += 1
                m4_optimization['neural_engine_validated'] = True
                
        if 'metal_gpu' in self.benchmark_results:
            optimization_components += 1
            gpu_results = self.benchmark_results['metal_gpu']
            if gpu_results.get('overall_performance', {}).get('performance_grade') in ['A+', 'A', 'B+']:
                optimized_components += 1
                m4_optimization['gpu_utilization_validated'] = True
                
        if 'trading_models' in self.benchmark_results:
            optimization_components += 1
            trading_results = self.benchmark_results['trading_models']
            if trading_results.get('overall_performance', {}).get('performance_grade') in ['A+', 'A', 'B+']:
                optimized_components += 1
                
        # Calculate overall M4 optimization grade
        optimization_rate = optimized_components / optimization_components if optimization_components > 0 else 0
        
        if optimization_rate >= 0.9:
            m4_optimization['overall_m4_optimization_grade'] = 'A+'
        elif optimization_rate >= 0.8:
            m4_optimization['overall_m4_optimization_grade'] = 'A'
        elif optimization_rate >= 0.7:
            m4_optimization['overall_m4_optimization_grade'] = 'B+'
        elif optimization_rate >= 0.6:
            m4_optimization['overall_m4_optimization_grade'] = 'B'
        elif optimization_rate >= 0.5:
            m4_optimization['overall_m4_optimization_grade'] = 'C'
            
        m4_optimization['optimization_components_tested'] = optimization_components
        m4_optimization['optimized_components'] = optimized_components
        m4_optimization['optimization_rate'] = optimization_rate
        
        return m4_optimization
        
    def _analyze_production_readiness(self) -> Dict[str, Any]:
        """Analyze production readiness"""
        readiness_criteria = {
            'system_stability': False,
            'performance_meets_requirements': False,
            'error_handling_validated': False,
            'scalability_validated': False,
            'reliability_validated': False,
            'overall_production_ready': False
        }
        
        readiness_scores = []
        
        # Check reliability and stress test results
        if 'reliability_stress' in self.benchmark_results:
            reliability = self.benchmark_results['reliability_stress']
            if 'production_readiness' in reliability:
                prod_ready = reliability['production_readiness']
                readiness_criteria['reliability_validated'] = prod_ready.get('reliability_validated', False)
                readiness_criteria['system_stability'] = reliability.get('overall_performance', {}).get('system_stability', False)
                
                if prod_ready.get('reliability_validated'):
                    readiness_scores.append(1.0)
                else:
                    readiness_scores.append(0.0)
                    
        # Check production workload results
        if 'production_workload' in self.benchmark_results:
            workload = self.benchmark_results['production_workload']
            if 'production_readiness' in workload:
                prod_ready = workload['production_readiness']
                readiness_criteria['performance_meets_requirements'] = any([
                    prod_ready.get('trading_throughput_validated', False),
                    prod_ready.get('market_data_processing_validated', False),
                    prod_ready.get('ml_pipeline_validated', False)
                ])
                
                if readiness_criteria['performance_meets_requirements']:
                    readiness_scores.append(1.0)
                else:
                    readiness_scores.append(0.5)
                    
        # Check container architecture
        if 'container_architecture' in self.benchmark_results:
            container = self.benchmark_results['container_architecture']
            if 'summary' in container:
                readiness_criteria['scalability_validated'] = container['summary'].get('overall_success', False)
                
                if readiness_criteria['scalability_validated']:
                    readiness_scores.append(1.0)
                else:
                    readiness_scores.append(0.0)
                    
        # Overall production readiness
        avg_readiness = np.mean(readiness_scores) if readiness_scores else 0.0
        readiness_criteria['overall_production_ready'] = avg_readiness >= 0.8
        readiness_criteria['readiness_score'] = avg_readiness
        
        # Production recommendation
        if avg_readiness >= 0.9:
            recommendation = "APPROVED - Excellent production readiness"
            confidence = "High"
        elif avg_readiness >= 0.8:
            recommendation = "APPROVED - Good production readiness with minor monitoring"
            confidence = "High"  
        elif avg_readiness >= 0.7:
            recommendation = "CONDITIONAL APPROVAL - Address identified issues before full deployment"
            confidence = "Medium"
        else:
            recommendation = "NOT APPROVED - Significant improvements required"
            confidence = "Low"
            
        readiness_criteria['recommendation'] = recommendation
        readiness_criteria['confidence_level'] = confidence
        
        return readiness_criteria
        
    def _identify_bottlenecks_and_recommendations(self) -> Dict[str, Any]:
        """Identify system bottlenecks and provide recommendations"""
        bottlenecks = []
        recommendations = []
        
        # Analyze performance grades to identify weak areas
        performance_areas = {}
        
        for test_name, results in self.benchmark_results.items():
            if isinstance(results, dict):
                # Extract performance information
                grade = None
                
                if 'summary' in results and 'overall_performance' in results['summary']:
                    grade = results['summary']['overall_performance'].get('performance_grade')
                elif 'overall_performance' in results:
                    grade = results['overall_performance'].get('overall_grade')
                elif 'production_readiness' in results:
                    grade = results['production_readiness'].get('overall_grade')
                    
                if grade:
                    performance_areas[test_name] = grade
                    
        # Identify bottlenecks based on poor grades
        grade_values = {'A+': 100, 'A': 95, 'B+': 85, 'B': 80, 'C': 70, 'D': 60}
        
        for area, grade in performance_areas.items():
            grade_value = grade_values.get(grade, 60)
            
            if grade_value < 80:  # B grade or below
                bottlenecks.append({
                    'area': area,
                    'grade': grade,
                    'severity': 'High' if grade_value < 70 else 'Medium',
                    'description': f"{area.replace('_', ' ').title()} performance needs improvement"
                })
                
                # Generate specific recommendations
                if 'production_workload' in area:
                    recommendations.append("Optimize trading algorithms and increase concurrent processing capacity")
                elif 'container' in area:
                    recommendations.append("Review container resource allocation and scaling policies")
                elif 'neural_engine' in area:
                    recommendations.append("Optimize ML models for Neural Engine acceleration")
                elif 'metal_gpu' in area:
                    recommendations.append("Implement Metal Performance Shaders for GPU-accelerated operations")
                elif 'reliability' in area:
                    recommendations.append("Strengthen error handling and implement additional failover mechanisms")
                    
        # Resource utilization bottlenecks
        resource_analysis = self._analyze_resource_utilization()
        
        if resource_analysis['cpu_utilization']['maximum'] > 90:
            bottlenecks.append({
                'area': 'cpu_utilization',
                'grade': 'C',
                'severity': 'Medium',
                'description': 'High CPU utilization detected during peak load'
            })
            recommendations.append("Consider implementing CPU load balancing and process optimization")
            
        if resource_analysis['memory_utilization']['maximum'] > 90:
            bottlenecks.append({
                'area': 'memory_utilization', 
                'grade': 'C',
                'severity': 'Medium',
                'description': 'High memory utilization detected during testing'
            })
            recommendations.append("Optimize memory usage patterns and implement memory pooling")
            
        # General optimization recommendations
        if not bottlenecks:
            recommendations.extend([
                "System performance is excellent - consider advanced optimizations for edge cases",
                "Implement comprehensive monitoring and alerting for production deployment",
                "Plan for horizontal scaling as transaction volume grows"
            ])
        else:
            recommendations.extend([
                "Prioritize addressing high-severity bottlenecks before production deployment",
                "Implement comprehensive performance monitoring to track improvements",
                "Consider staged rollout to production with careful monitoring"
            ])
            
        return {
            'identified_bottlenecks': bottlenecks,
            'optimization_recommendations': recommendations,
            'bottleneck_count': len(bottlenecks),
            'high_severity_bottlenecks': len([b for b in bottlenecks if b['severity'] == 'High']),
            'overall_bottleneck_risk': (
                'High' if any(b['severity'] == 'High' for b in bottlenecks) else
                'Medium' if bottlenecks else
                'Low'
            )
        }
        
    def _assess_resource_optimization(self, cpu_usage: List[float], memory_usage: List[float]) -> str:
        """Assess resource optimization status"""
        if not cpu_usage or not memory_usage:
            return "Insufficient Data"
            
        avg_cpu = np.mean(cpu_usage)
        avg_memory = np.mean(memory_usage)
        max_cpu = max(cpu_usage)
        max_memory = max(memory_usage)
        
        # Optimal range: 50-80% average utilization, <90% peak
        cpu_optimal = 50 <= avg_cpu <= 80 and max_cpu < 90
        memory_optimal = 40 <= avg_memory <= 75 and max_memory < 90
        
        if cpu_optimal and memory_optimal:
            return "Excellent"
        elif cpu_optimal or memory_optimal:
            return "Good"
        else:
            return "Needs Optimization"
            
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system assessment report"""
        logger.info("Generating comprehensive system assessment report...")
        
        # Load all benchmark results
        self.load_all_benchmark_results()
        
        # Analyze system performance
        analysis = self.analyze_system_performance()
        
        # Generate final report
        timestamp = datetime.now(timezone.utc)
        
        comprehensive_report = {
            'assessment_metadata': {
                'timestamp': timestamp.isoformat(),
                'assessment_version': '1.0.0',
                'benchmarks_analyzed': len(self.benchmark_results),
                'assessment_type': 'M4 Max Integration Validation & Production Readiness'
            },
            'executive_summary': {
                'overall_grade': analysis['performance_overview']['overall_grade'],
                'production_ready': analysis['production_readiness']['overall_production_ready'],
                'recommendation': analysis['production_readiness']['recommendation'],
                'confidence_level': analysis['production_readiness']['confidence_level'],
                'm4_max_optimization_success': analysis['m4_max_optimization']['optimization_rate'] >= 0.8
            },
            'detailed_analysis': analysis,
            'benchmark_data_sources': list(self.benchmark_results.keys()),
            'deployment_decision': self._make_deployment_decision(analysis)
        }
        
        return comprehensive_report
        
    def _make_deployment_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make final deployment decision"""
        
        # Decision criteria
        performance_grade = analysis['performance_overview']['overall_grade']
        production_ready = analysis['production_readiness']['overall_production_ready']
        m4_optimization = analysis['m4_max_optimization']['optimization_rate'] >= 0.7
        system_stable = analysis['production_readiness']['system_stability']
        bottleneck_risk = analysis['bottlenecks_and_recommendations']['overall_bottleneck_risk']
        
        # Decision matrix
        grade_values = {'A+': 100, 'A': 95, 'B+': 85, 'B': 80, 'C': 70, 'D': 60}
        performance_score = grade_values.get(performance_grade, 60)
        
        decision_score = 0
        max_score = 100
        
        # Performance score (40 points)
        decision_score += (performance_score / 100) * 40
        
        # Production readiness (25 points)
        if production_ready:
            decision_score += 25
            
        # M4 Max optimization (20 points)  
        if m4_optimization:
            decision_score += 20
            
        # System stability (10 points)
        if system_stable:
            decision_score += 10
            
        # Bottleneck risk (-5 to 5 points)
        if bottleneck_risk == 'Low':
            decision_score += 5
        elif bottleneck_risk == 'High':
            decision_score -= 5
            
        # Final decision
        decision_percentage = (decision_score / max_score) * 100
        
        if decision_percentage >= 90:
            decision = "APPROVED - Full Production Deployment"
            deployment_phase = "Production"
            risk_level = "Low"
        elif decision_percentage >= 80:
            decision = "APPROVED - Staged Production Deployment"
            deployment_phase = "Staged Production"
            risk_level = "Low"
        elif decision_percentage >= 70:
            decision = "CONDITIONAL - Pre-Production Testing Required"
            deployment_phase = "Pre-Production"
            risk_level = "Medium"
        elif decision_percentage >= 60:
            decision = "NOT APPROVED - Optimization Required"
            deployment_phase = "Development"
            risk_level = "High"
        else:
            decision = "NOT APPROVED - Significant Issues Found"
            deployment_phase = "Development"
            risk_level = "Critical"
            
        return {
            'decision': decision,
            'decision_score': decision_percentage,
            'deployment_phase': deployment_phase,
            'risk_level': risk_level,
            'key_factors': {
                'performance_grade': performance_grade,
                'production_ready': production_ready,
                'm4_optimization_successful': m4_optimization,
                'system_stable': system_stable,
                'bottleneck_risk': bottleneck_risk
            },
            'next_steps': self._generate_next_steps(deployment_phase, analysis)
        }
        
    def _generate_next_steps(self, deployment_phase: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate next steps based on deployment decision"""
        
        if deployment_phase == "Production":
            return [
                "Deploy to production environment with full monitoring",
                "Implement production-grade logging and alerting",
                "Plan capacity scaling based on actual usage patterns",
                "Schedule regular performance reviews"
            ]
        elif deployment_phase == "Staged Production":
            return [
                "Deploy to production with gradual traffic ramp-up",
                "Monitor key performance indicators closely",
                "Implement automated rollback procedures", 
                "Plan for full deployment after validation period"
            ]
        elif deployment_phase == "Pre-Production":
            return [
                "Address identified bottlenecks and performance issues",
                "Conduct additional stress testing",
                "Implement performance optimizations",
                "Re-run comprehensive benchmarks after improvements"
            ]
        else:
            bottlenecks = analysis['bottlenecks_and_recommendations']['identified_bottlenecks']
            steps = [
                "Address all high-severity bottlenecks identified",
                "Implement optimization recommendations",
                "Re-run failed benchmark tests"
            ]
            
            # Add specific steps based on bottlenecks
            for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
                if 'production_workload' in bottleneck['area']:
                    steps.append("Optimize trading algorithms and increase processing capacity")
                elif 'container' in bottleneck['area']:
                    steps.append("Review and optimize container architecture")
                elif 'neural_engine' in bottleneck['area']:
                    steps.append("Implement Neural Engine optimizations for ML models")
                    
            steps.append("Re-assess system readiness after addressing issues")
            return steps

async def main():
    """Main assessment execution"""
    results_dir = "/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml"
    
    assessor = ComprehensiveSystemAssessment(results_dir)
    comprehensive_report = assessor.generate_comprehensive_report()
    
    # Save comprehensive assessment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assessment_file = f"{results_dir}/comprehensive_system_assessment_{timestamp}.json"
    
    with open(assessment_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    logger.info(f"Comprehensive system assessment saved to: {assessment_file}")
    
    # Print Executive Summary
    logger.info("=" * 80)
    logger.info("NAUTILUS TRADING PLATFORM - M4 MAX COMPREHENSIVE SYSTEM ASSESSMENT")
    logger.info("=" * 80)
    
    exec_summary = comprehensive_report['executive_summary']
    deployment = comprehensive_report['deployment_decision']
    
    logger.info(f"Overall Performance Grade: {exec_summary['overall_grade']}")
    logger.info(f"Production Ready: {'YES' if exec_summary['production_ready'] else 'NO'}")
    logger.info(f"M4 Max Optimization Success: {'YES' if exec_summary['m4_max_optimization_success'] else 'NO'}")
    logger.info(f"")
    logger.info(f"DEPLOYMENT DECISION: {deployment['decision']}")
    logger.info(f"Decision Score: {deployment['decision_score']:.1f}/100")
    logger.info(f"Risk Level: {deployment['risk_level']}")
    logger.info(f"")
    logger.info("KEY METRICS:")
    
    detailed = comprehensive_report['detailed_analysis']
    logger.info(f"  • Container Architecture: {detailed['container_architecture'].get('performance_grade', 'N/A')}")
    logger.info(f"  • Resource Utilization: {detailed['resource_utilization']['cpu_utilization']['efficiency']} CPU, {detailed['resource_utilization']['memory_utilization']['efficiency']} Memory")
    logger.info(f"  • System Stability: {'STABLE' if detailed['production_readiness']['system_stability'] else 'UNSTABLE'}")
    logger.info(f"  • Bottleneck Risk: {detailed['bottlenecks_and_recommendations']['overall_bottleneck_risk']}")
    
    logger.info(f"")
    logger.info("NEXT STEPS:")
    for i, step in enumerate(deployment['next_steps'], 1):
        logger.info(f"  {i}. {step}")
    
    logger.info("=" * 80)
    
    return comprehensive_report

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())