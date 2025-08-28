#!/usr/bin/env python3
"""
Nautilus Stress Test Results Analyzer
Comprehensive analysis of stress test results with optimization recommendations.
Uses real data from completed stress tests, monitoring, and data downloads.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NautilusResultsAnalyzer:
    """Analyzes stress test results and generates optimization recommendations"""
    
    def __init__(self):
        self.stress_test_results = None
        self.download_results = None
        self.monitoring_data = None
        
        # Performance thresholds
        self.thresholds = {
            "excellent_latency": 2.0,  # ms
            "good_latency": 5.0,       # ms
            "acceptable_latency": 10.0, # ms
            "excellent_rps": 50000,     # requests per second
            "good_rps": 25000,
            "acceptable_rps": 10000,
            "max_error_rate": 0.01,    # 1%
            "min_success_rate": 0.95   # 95%
        }
        
        # System specifications
        self.system_specs = {
            "engines_count": 18,
            "redis_buses": 4,
            "data_sources": 8,
            "hardware": "M4 Max",
            "neural_engine_tops": 38.0,
            "gpu_cores": 40,
            "unified_memory_gb": 64
        }

    def load_stress_test_results(self, filename: str = None) -> Dict[str, Any]:
        """Load and parse stress test results"""
        if filename is None:
            # Find the most recent stress test results
            result_files = list(Path(".").glob("stress_test_results_*.json"))
            if not result_files:
                logger.error("No stress test results found")
                return {}
            filename = max(result_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(filename, 'r') as f:
                self.stress_test_results = json.load(f)
            logger.info(f"✅ Loaded stress test results from {filename}")
            return self.stress_test_results
        except Exception as e:
            logger.error(f"❌ Failed to load stress test results: {e}")
            return {}

    def load_download_results(self, filename: str = "massive_datasets/download_summary.json") -> Dict[str, Any]:
        """Load data download results"""
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    self.download_results = json.load(f)
                logger.info(f"✅ Loaded download results from {filename}")
                return self.download_results
            else:
                logger.warning(f"⚠️ Download results not found at {filename}")
                return {}
        except Exception as e:
            logger.error(f"❌ Failed to load download results: {e}")
            return {}

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics from stress test"""
        if not self.stress_test_results:
            return {}
        
        perf_summary = self.stress_test_results.get("performance_summary", {})
        hardware_util = self.stress_test_results.get("hardware_utilization", {})
        test_results = self.stress_test_results.get("test_results", [])
        
        # Calculate performance grades
        analysis = {
            "latency_analysis": self._analyze_latency(perf_summary.get("final_latency_ms", 0)),
            "throughput_analysis": self._analyze_throughput(test_results),
            "error_analysis": self._analyze_errors(perf_summary),
            "hardware_analysis": self._analyze_hardware_utilization(hardware_util),
            "engine_health": self._analyze_engine_health(perf_summary.get("healthy_engines", "0/0")),
            "overall_grade": perf_summary.get("system_grade", "N/A")
        }
        
        return analysis

    def _analyze_latency(self, latency_ms: float) -> Dict[str, Any]:
        """Analyze latency performance"""
        if latency_ms <= self.thresholds["excellent_latency"]:
            grade = "Excellent"
            color = "green"
            improvement = 0
        elif latency_ms <= self.thresholds["good_latency"]:
            grade = "Good"
            color = "yellow"
            improvement = (latency_ms - self.thresholds["excellent_latency"]) / latency_ms * 100
        elif latency_ms <= self.thresholds["acceptable_latency"]:
            grade = "Acceptable"
            color = "orange"
            improvement = (latency_ms - self.thresholds["good_latency"]) / latency_ms * 100
        else:
            grade = "Poor"
            color = "red"
            improvement = (latency_ms - self.thresholds["acceptable_latency"]) / latency_ms * 100
        
        return {
            "latency_ms": latency_ms,
            "grade": grade,
            "color": color,
            "improvement_needed_pct": max(0, improvement),
            "target_latency": self.thresholds["excellent_latency"],
            "recommendations": self._get_latency_recommendations(latency_ms)
        }

    def _analyze_throughput(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze throughput performance"""
        load_test = next((t for t in test_results if t.get("test") == "extreme_data_load"), {})
        actual_rps = load_test.get("actual_rps", 0)
        target_rps = load_test.get("target_rps", 50000)
        
        achievement_pct = (actual_rps / target_rps) * 100 if target_rps > 0 else 0
        
        if achievement_pct >= 90:
            grade = "Excellent"
            color = "green"
        elif achievement_pct >= 70:
            grade = "Good"
            color = "yellow"
        elif achievement_pct >= 50:
            grade = "Acceptable"
            color = "orange"
        else:
            grade = "Poor"
            color = "red"
        
        return {
            "actual_rps": actual_rps,
            "target_rps": target_rps,
            "achievement_pct": achievement_pct,
            "grade": grade,
            "color": color,
            "recommendations": self._get_throughput_recommendations(achievement_pct)
        }

    def _analyze_errors(self, perf_summary: Dict) -> Dict[str, Any]:
        """Analyze error rates"""
        error_rate = perf_summary.get("overall_error_rate", 0)
        total_errors = perf_summary.get("total_errors", 0)
        
        if error_rate <= 0.001:  # 0.1%
            grade = "Excellent"
            color = "green"
        elif error_rate <= 0.01:  # 1%
            grade = "Good"
            color = "yellow"
        elif error_rate <= 0.05:  # 5%
            grade = "Acceptable"
            color = "orange"
        else:
            grade = "Poor"
            color = "red"
        
        return {
            "error_rate": error_rate,
            "error_rate_pct": error_rate * 100,
            "total_errors": total_errors,
            "grade": grade,
            "color": color,
            "recommendations": self._get_error_recommendations(error_rate)
        }

    def _analyze_hardware_utilization(self, hardware_util: Dict) -> Dict[str, Any]:
        """Analyze hardware utilization"""
        cpu_usage = hardware_util.get("cpu_usage", 0)
        memory_usage = hardware_util.get("memory_usage", 0)
        neural_engine = hardware_util.get("neural_engine_usage", 0)
        gpu_usage = hardware_util.get("gpu_usage", 0)
        
        # Calculate overall efficiency score
        efficiency_score = np.mean([
            min(100, neural_engine) / 100,
            min(100, gpu_usage) / 100,
            min(100, cpu_usage) / 100 * 0.5,  # CPU weight lower since we want specialized hardware
        ])
        
        if efficiency_score >= 0.85:
            grade = "Excellent"
            color = "green"
        elif efficiency_score >= 0.70:
            grade = "Good"
            color = "yellow"
        elif efficiency_score >= 0.55:
            grade = "Acceptable"
            color = "orange"
        else:
            grade = "Poor"
            color = "red"
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "neural_engine_usage": neural_engine,
            "gpu_usage": gpu_usage,
            "efficiency_score": efficiency_score * 100,
            "grade": grade,
            "color": color,
            "recommendations": self._get_hardware_recommendations(hardware_util)
        }

    def _analyze_engine_health(self, healthy_engines_str: str) -> Dict[str, Any]:
        """Analyze engine health status"""
        try:
            healthy, total = map(int, healthy_engines_str.split('/'))
            health_pct = (healthy / total) * 100 if total > 0 else 0
            
            if health_pct >= 95:
                grade = "Excellent"
                color = "green"
            elif health_pct >= 90:
                grade = "Good"
                color = "yellow"
            elif health_pct >= 80:
                grade = "Acceptable"
                color = "orange"
            else:
                grade = "Poor"
                color = "red"
            
            return {
                "healthy_engines": healthy,
                "total_engines": total,
                "health_pct": health_pct,
                "grade": grade,
                "color": color,
                "recommendations": self._get_engine_recommendations(health_pct)
            }
        except:
            return {"healthy_engines": 0, "total_engines": 0, "health_pct": 0, "grade": "Unknown", "color": "gray", "recommendations": []}

    def _get_latency_recommendations(self, latency_ms: float) -> List[str]:
        """Generate latency optimization recommendations"""
        recommendations = []
        
        if latency_ms > 5.0:
            recommendations.extend([
                "Increase Redis connection pool sizes",
                "Implement connection pooling for database",
                "Enable Redis pipelining for batch operations",
                "Optimize database queries and add indexes"
            ])
        
        if latency_ms > 2.0:
            recommendations.extend([
                "Implement request response caching",
                "Optimize serialization/deserialization",
                "Consider using faster serialization formats (e.g., MessagePack)"
            ])
        
        return recommendations

    def _get_throughput_recommendations(self, achievement_pct: float) -> List[str]:
        """Generate throughput optimization recommendations"""
        recommendations = []
        
        if achievement_pct < 70:
            recommendations.extend([
                "Scale engines horizontally (add more instances)",
                "Implement load balancing across Redis buses",
                "Optimize message routing algorithms",
                "Consider async processing for heavy operations"
            ])
        
        if achievement_pct < 90:
            recommendations.extend([
                "Increase worker thread pools",
                "Optimize hot code paths with profiling",
                "Implement request batching where possible"
            ])
        
        return recommendations

    def _get_error_recommendations(self, error_rate: float) -> List[str]:
        """Generate error handling recommendations"""
        recommendations = []
        
        if error_rate > 0.01:
            recommendations.extend([
                "Implement circuit breaker patterns",
                "Add comprehensive input validation",
                "Improve error handling and recovery mechanisms",
                "Add retry logic with exponential backoff"
            ])
        
        if error_rate > 0.001:
            recommendations.extend([
                "Enhance logging and monitoring",
                "Implement health checks for all services",
                "Add graceful degradation for non-critical services"
            ])
        
        return recommendations

    def _get_hardware_recommendations(self, hardware_util: Dict) -> List[str]:
        """Generate hardware optimization recommendations"""
        recommendations = []
        
        neural_engine = hardware_util.get("neural_engine_usage", 0)
        gpu_usage = hardware_util.get("gpu_usage", 0)
        memory_usage = hardware_util.get("memory_usage", 0)
        
        if neural_engine < 80:
            recommendations.append("Increase Neural Engine utilization for ML workloads")
        
        if gpu_usage < 85:
            recommendations.append("Optimize GPU usage for parallel computations")
        
        if memory_usage > 80:
            recommendations.append("Implement memory optimization and garbage collection tuning")
        
        return recommendations

    def _get_engine_recommendations(self, health_pct: float) -> List[str]:
        """Generate engine health recommendations"""
        recommendations = []
        
        if health_pct < 95:
            recommendations.extend([
                "Investigate unhealthy engines and fix underlying issues",
                "Implement automatic engine restart mechanisms",
                "Add comprehensive engine health monitoring"
            ])
        
        if health_pct < 90:
            recommendations.extend([
                "Consider engine redundancy and failover",
                "Implement graceful shutdown and startup procedures"
            ])
        
        return recommendations

    def analyze_data_download_performance(self) -> Dict[str, Any]:
        """Analyze data download performance"""
        if not self.download_results:
            return {}
        
        download_summary = self.download_results.get("download_summary", {})
        
        success_rate = download_summary.get("success_rate", 0) * 100
        data_points = download_summary.get("total_data_points", 0)
        download_rate = download_summary.get("download_rate", 0)
        sources_used = download_summary.get("data_sources_used", 0)
        
        return {
            "success_rate_pct": success_rate,
            "total_data_points": data_points,
            "download_rate_per_sec": download_rate,
            "data_sources_active": sources_used,
            "data_sources_total": self.system_specs["data_sources"],
            "grade": "Excellent" if success_rate >= 80 else "Good" if success_rate >= 60 else "Poor",
            "recommendations": self._get_download_recommendations(success_rate, sources_used)
        }

    def _get_download_recommendations(self, success_rate: float, sources_used: int) -> List[str]:
        """Generate data download optimization recommendations"""
        recommendations = []
        
        if success_rate < 80:
            recommendations.extend([
                "Implement retry mechanisms for failed downloads",
                "Add exponential backoff for rate-limited APIs",
                "Implement parallel download streams"
            ])
        
        if sources_used < self.system_specs["data_sources"]:
            recommendations.extend([
                "Fix SSL certificate issues for external APIs",
                "Implement API key management and rotation",
                "Add fallback data sources for redundancy"
            ])
        
        return recommendations

    def generate_comprehensive_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive optimization recommendations"""
        
        performance_analysis = self.analyze_performance_metrics()
        download_analysis = self.analyze_data_download_performance()
        
        # Prioritize recommendations by impact
        high_impact_recs = []
        medium_impact_recs = []
        low_impact_recs = []
        
        # Extract all recommendations
        all_recs = []
        for analysis in performance_analysis.values():
            if isinstance(analysis, dict) and "recommendations" in analysis:
                all_recs.extend(analysis["recommendations"])
        
        all_recs.extend(download_analysis.get("recommendations", []))
        
        # Prioritize based on keywords
        for rec in set(all_recs):  # Remove duplicates
            if any(keyword in rec.lower() for keyword in ["redis", "database", "connection", "pool"]):
                high_impact_recs.append(rec)
            elif any(keyword in rec.lower() for keyword in ["scale", "horizontal", "load"]):
                medium_impact_recs.append(rec)
            else:
                low_impact_recs.append(rec)
        
        # Calculate overall system score
        scores = []
        if performance_analysis.get("latency_analysis", {}).get("latency_ms"):
            scores.append(min(100, 2000 / performance_analysis["latency_analysis"]["latency_ms"]))
        if performance_analysis.get("hardware_analysis", {}).get("efficiency_score"):
            scores.append(performance_analysis["hardware_analysis"]["efficiency_score"])
        if performance_analysis.get("engine_health", {}).get("health_pct"):
            scores.append(performance_analysis["engine_health"]["health_pct"])
        
        overall_score = np.mean(scores) if scores else 0
        
        return {
            "overall_system_score": overall_score,
            "system_grade": "A+" if overall_score >= 95 else "A" if overall_score >= 85 else "B+" if overall_score >= 75 else "B",
            "performance_analysis": performance_analysis,
            "download_analysis": download_analysis,
            "prioritized_recommendations": {
                "high_impact": high_impact_recs,
                "medium_impact": medium_impact_recs,
                "low_impact": low_impact_recs
            },
            "implementation_roadmap": self._create_implementation_roadmap(high_impact_recs, medium_impact_recs),
            "expected_improvements": self._calculate_expected_improvements(performance_analysis)
        }

    def _create_implementation_roadmap(self, high_impact: List[str], medium_impact: List[str]) -> List[Dict[str, Any]]:
        """Create implementation roadmap for recommendations"""
        roadmap = []
        
        # Phase 1: High Impact (Week 1-2)
        if high_impact:
            roadmap.append({
                "phase": 1,
                "timeline": "Week 1-2",
                "priority": "High Impact",
                "tasks": high_impact[:3],  # Top 3 high impact
                "expected_improvement": "20-40%"
            })
        
        # Phase 2: Medium Impact (Week 3-4)
        if medium_impact:
            roadmap.append({
                "phase": 2,
                "timeline": "Week 3-4", 
                "priority": "Medium Impact",
                "tasks": medium_impact[:3],  # Top 3 medium impact
                "expected_improvement": "10-20%"
            })
        
        return roadmap

    def _calculate_expected_improvements(self, performance_analysis: Dict) -> Dict[str, str]:
        """Calculate expected performance improvements"""
        improvements = {}
        
        latency_analysis = performance_analysis.get("latency_analysis", {})
        if latency_analysis.get("improvement_needed_pct", 0) > 0:
            improvements["latency"] = f"{min(50, latency_analysis['improvement_needed_pct']):.0f}% reduction possible"
        
        throughput_analysis = performance_analysis.get("throughput_analysis", {})
        if throughput_analysis.get("achievement_pct", 100) < 90:
            improvements["throughput"] = f"{100 - throughput_analysis['achievement_pct']:.0f}% increase possible"
        
        hardware_analysis = performance_analysis.get("hardware_analysis", {})
        if hardware_analysis.get("efficiency_score", 100) < 85:
            improvements["hardware"] = f"{85 - hardware_analysis['efficiency_score']:.0f}% efficiency gain possible"
        
        return improvements

    def create_performance_report(self) -> str:
        """Create comprehensive performance report"""
        
        comprehensive_analysis = self.generate_comprehensive_recommendations()
        
        report = f"""
# 🚀 Nautilus System Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 Overall System Assessment
- **System Grade**: {comprehensive_analysis['system_grade']}
- **Overall Score**: {comprehensive_analysis['overall_system_score']:.1f}/100
- **Status**: {'🟢 Excellent' if comprehensive_analysis['overall_system_score'] >= 85 else '🟡 Good' if comprehensive_analysis['overall_system_score'] >= 70 else '🔴 Needs Improvement'}

## 📊 Performance Breakdown

### ⚡ Latency Analysis
{self._format_analysis_section(comprehensive_analysis['performance_analysis'].get('latency_analysis', {}))}

### 🚀 Throughput Analysis  
{self._format_analysis_section(comprehensive_analysis['performance_analysis'].get('throughput_analysis', {}))}

### ❌ Error Analysis
{self._format_analysis_section(comprehensive_analysis['performance_analysis'].get('error_analysis', {}))}

### 🖥️ Hardware Utilization
{self._format_analysis_section(comprehensive_analysis['performance_analysis'].get('hardware_analysis', {}))}

### 🏥 Engine Health
{self._format_analysis_section(comprehensive_analysis['performance_analysis'].get('engine_health', {}))}

## 📥 Data Download Performance
{self._format_analysis_section(comprehensive_analysis.get('download_analysis', {}))}

## 🎯 Optimization Recommendations

### 🔥 High Impact (Immediate Action)
{self._format_recommendations_list(comprehensive_analysis['prioritized_recommendations']['high_impact'])}

### ⚡ Medium Impact (Next Steps)
{self._format_recommendations_list(comprehensive_analysis['prioritized_recommendations']['medium_impact'])}

### 💡 Low Impact (Future Enhancements)
{self._format_recommendations_list(comprehensive_analysis['prioritized_recommendations']['low_impact'])}

## 🗓️ Implementation Roadmap
{self._format_roadmap(comprehensive_analysis.get('implementation_roadmap', []))}

## 📈 Expected Improvements
{self._format_improvements(comprehensive_analysis.get('expected_improvements', {}))}

---
*Report generated by Nautilus Adaptive Learning System*
        """
        
        return report.strip()

    def _format_analysis_section(self, analysis: Dict[str, Any]) -> str:
        """Format analysis section for report"""
        if not analysis:
            return "- No data available"
        
        lines = []
        grade = analysis.get('grade', 'N/A')
        color_map = {'Excellent': '🟢', 'Good': '🟡', 'Acceptable': '🟠', 'Poor': '🔴'}
        icon = color_map.get(grade, '⚪')
        
        lines.append(f"- **Grade**: {icon} {grade}")
        
        # Add specific metrics based on analysis type
        if 'latency_ms' in analysis:
            lines.append(f"- **Latency**: {analysis['latency_ms']:.2f}ms")
        if 'actual_rps' in analysis:
            lines.append(f"- **Throughput**: {analysis['actual_rps']:,.0f} RPS")
        if 'error_rate_pct' in analysis:
            lines.append(f"- **Error Rate**: {analysis['error_rate_pct']:.3f}%")
        if 'efficiency_score' in analysis:
            lines.append(f"- **Efficiency**: {analysis['efficiency_score']:.1f}%")
        if 'health_pct' in analysis:
            lines.append(f"- **Health**: {analysis['health_pct']:.1f}% ({analysis.get('healthy_engines', 0)}/{analysis.get('total_engines', 0)})")
        if 'success_rate_pct' in analysis:
            lines.append(f"- **Success Rate**: {analysis['success_rate_pct']:.1f}%")
            lines.append(f"- **Data Points**: {analysis.get('total_data_points', 0):,}")
        
        return '\n'.join(lines)

    def _format_recommendations_list(self, recommendations: List[str]) -> str:
        """Format recommendations list"""
        if not recommendations:
            return "- None identified"
        return '\n'.join([f"- {rec}" for rec in recommendations])

    def _format_roadmap(self, roadmap: List[Dict[str, Any]]) -> str:
        """Format implementation roadmap"""
        if not roadmap:
            return "- No roadmap generated"
        
        formatted = []
        for phase in roadmap:
            formatted.append(f"""
### Phase {phase['phase']}: {phase['priority']} ({phase['timeline']})
- **Expected Improvement**: {phase['expected_improvement']}
- **Tasks**:
{self._format_recommendations_list(phase['tasks'])}
            """.strip())
        
        return '\n\n'.join(formatted)

    def _format_improvements(self, improvements: Dict[str, str]) -> str:
        """Format expected improvements"""
        if not improvements:
            return "- System performing optimally"
        
        return '\n'.join([f"- **{key.title()}**: {value}" for key, value in improvements.items()])

    def save_analysis_results(self, analysis: Dict[str, Any], report: str):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON analysis
        analysis_file = f"nautilus_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save markdown report
        report_file = f"nautilus_performance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Analysis saved to {analysis_file}")
        logger.info(f"📋 Report saved to {report_file}")
        
        return analysis_file, report_file

def main():
    """Main analysis execution"""
    analyzer = NautilusResultsAnalyzer()
    
    # Load results
    stress_results = analyzer.load_stress_test_results()
    download_results = analyzer.load_download_results()
    
    if not stress_results:
        logger.error("❌ No stress test results to analyze")
        return
    
    # Generate comprehensive analysis
    comprehensive_analysis = analyzer.generate_comprehensive_recommendations()
    
    # Create performance report
    performance_report = analyzer.create_performance_report()
    
    # Save results
    analysis_file, report_file = analyzer.save_analysis_results(comprehensive_analysis, performance_report)
    
    # Print summary
    print(f"""
🎯 NAUTILUS PERFORMANCE ANALYSIS COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 OVERALL SYSTEM GRADE: {comprehensive_analysis['system_grade']}
📊 SYSTEM SCORE: {comprehensive_analysis['overall_system_score']:.1f}/100

📋 KEY FINDINGS:
• Latency: {stress_results.get('performance_summary', {}).get('final_latency_ms', 0):.2f}ms
• Engines: {stress_results.get('performance_summary', {}).get('healthy_engines', 'N/A')} healthy
• Error Rate: {stress_results.get('performance_summary', {}).get('overall_error_rate', 0):.3%}

🎯 RECOMMENDATIONS GENERATED:
• High Impact: {len(comprehensive_analysis['prioritized_recommendations']['high_impact'])}
• Medium Impact: {len(comprehensive_analysis['prioritized_recommendations']['medium_impact'])}  
• Low Impact: {len(comprehensive_analysis['prioritized_recommendations']['low_impact'])}

📁 FILES CREATED:
• Analysis: {analysis_file}
• Report: {report_file}

🚀 Next Steps: Review recommendations and implement Phase 1 optimizations!
    """)

if __name__ == "__main__":
    print("📊 Nautilus Results Analyzer")
    print("=" * 40)
    main()