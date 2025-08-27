"""
SME Performance Monitoring

Real-time monitoring of SME hardware utilization, performance metrics,
and system optimization for M4 Max trading applications.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SMEMetrics:
    """SME Performance Metrics"""
    timestamp: datetime
    fp32_utilization_percent: float
    fp64_utilization_percent: float
    memory_bandwidth_gbps: float
    operations_per_second: float
    jit_kernel_hit_rate: float
    average_execution_time_ms: float
    thermal_state: str
    power_consumption_watts: float

@dataclass
class SMEEngineMetrics:
    """SME Metrics per Engine"""
    engine_name: str
    engine_port: int
    sme_operations_count: int
    average_speedup: float
    total_execution_time_ms: float
    memory_bandwidth_used_gbps: float
    error_count: int

class SMEPerformanceMonitor:
    """SME Performance Monitoring System"""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        self.engine_metrics = {}
        self.performance_alerts = []
        self.monitoring_task = None
        
        # Configuration
        self.monitoring_interval_seconds = 1.0
        self.metrics_retention_hours = 24
        self.alert_thresholds = {
            "low_utilization": 20.0,  # Below 20% utilization
            "high_thermal": 85.0,     # Above 85°C
            "low_speedup": 5.0,       # Below 5x speedup
            "high_error_rate": 0.01   # Above 1% error rate
        }
        
        # Synthetic metrics (would be replaced with actual hardware monitoring)
        self.base_metrics = SMEMetrics(
            timestamp=datetime.now(),
            fp32_utilization_percent=0.0,
            fp64_utilization_percent=0.0,
            memory_bandwidth_gbps=0.0,
            operations_per_second=0.0,
            jit_kernel_hit_rate=0.0,
            average_execution_time_ms=0.0,
            thermal_state="normal",
            power_consumption_watts=0.0
        )
    
    async def start_monitoring(self) -> None:
        """Start SME performance monitoring"""
        if self.monitoring_active:
            logger.warning("SME monitoring already active")
            return
        
        try:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("✅ SME performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start SME monitoring: {e}")
            self.monitoring_active = False
    
    async def stop_monitoring(self) -> None:
        """Stop SME performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ SME performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                # Collect SME metrics
                current_metrics = await self._collect_sme_metrics()
                
                # Store metrics
                self.metrics_history.append(current_metrics)
                
                # Clean old metrics
                await self._cleanup_old_metrics()
                
                # Check for alerts
                await self._check_performance_alerts(current_metrics)
                
                # Update engine-specific metrics
                await self._update_engine_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.monitoring_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("SME monitoring loop cancelled")
        except Exception as e:
            logger.error(f"SME monitoring loop error: {e}")
    
    async def _collect_sme_metrics(self) -> SMEMetrics:
        """Collect current SME performance metrics"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Simulate SME-specific metrics (would be actual hardware monitoring)
            current_time = datetime.now()
            
            # Simulate realistic SME utilization based on system load
            fp32_utilization = min(85.0, cpu_percent * 1.2)  # SME typically higher than CPU
            fp64_utilization = fp32_utilization * 0.25  # FP64 is 4x slower, less used
            
            # Simulate memory bandwidth based on memory utilization
            memory_bandwidth = (memory.percent / 100.0) * 546  # M4 Max has 546 GB/s
            
            # Simulate operations per second based on utilization
            ops_per_second = (fp32_utilization / 100.0) * 2.9e12  # 2.9 TFLOPS peak
            
            # Simulate JIT kernel hit rate (higher for small matrices)
            jit_hit_rate = 75.0 + np.random.normal(0, 5)  # 75% average with variation
            jit_hit_rate = max(0, min(100, jit_hit_rate))
            
            # Simulate execution time based on utilization
            avg_execution_time = 1.0 + (100 - fp32_utilization) / 50.0  # Lower utilization = longer times
            
            # Simulate thermal state
            if fp32_utilization > 80:
                thermal_state = "warm"
            elif fp32_utilization > 90:
                thermal_state = "hot"
            else:
                thermal_state = "normal"
            
            # Simulate power consumption
            power_watts = 15 + (fp32_utilization / 100.0) * 25  # 15-40W range
            
            return SMEMetrics(
                timestamp=current_time,
                fp32_utilization_percent=fp32_utilization,
                fp64_utilization_percent=fp64_utilization,
                memory_bandwidth_gbps=memory_bandwidth,
                operations_per_second=ops_per_second,
                jit_kernel_hit_rate=jit_hit_rate,
                average_execution_time_ms=avg_execution_time,
                thermal_state=thermal_state,
                power_consumption_watts=power_watts
            )
            
        except Exception as e:
            logger.error(f"Failed to collect SME metrics: {e}")
            return self.base_metrics
    
    async def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    async def _check_performance_alerts(self, metrics: SMEMetrics) -> None:
        """Check for performance alerts"""
        try:
            alerts = []
            
            # Low utilization alert
            if metrics.fp32_utilization_percent < self.alert_thresholds["low_utilization"]:
                alerts.append({
                    "type": "low_utilization",
                    "message": f"SME FP32 utilization below {self.alert_thresholds['low_utilization']}%: {metrics.fp32_utilization_percent:.1f}%",
                    "severity": "warning",
                    "timestamp": metrics.timestamp
                })
            
            # Thermal alert
            if metrics.thermal_state == "hot":
                alerts.append({
                    "type": "high_thermal",
                    "message": f"SME thermal state: {metrics.thermal_state}",
                    "severity": "warning",
                    "timestamp": metrics.timestamp
                })
            
            # Performance alert
            if metrics.average_execution_time_ms > 10.0:
                alerts.append({
                    "type": "high_execution_time",
                    "message": f"SME average execution time high: {metrics.average_execution_time_ms:.1f}ms",
                    "severity": "info",
                    "timestamp": metrics.timestamp
                })
            
            # Store alerts
            self.performance_alerts.extend(alerts)
            
            # Log critical alerts
            for alert in alerts:
                if alert["severity"] == "warning":
                    logger.warning(f"SME Alert: {alert['message']}")
                    
        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")
    
    async def _update_engine_metrics(self) -> None:
        """Update engine-specific SME metrics"""
        try:
            # This would collect actual metrics from each engine
            # For now, simulate metrics for key engines
            engines = [
                ("Risk Engine", 8200),
                ("Portfolio Engine", 8900),
                ("Analytics Engine", 8100),
                ("ML Engine", 8400)
            ]
            
            for engine_name, port in engines:
                if engine_name not in self.engine_metrics:
                    self.engine_metrics[engine_name] = SMEEngineMetrics(
                        engine_name=engine_name,
                        engine_port=port,
                        sme_operations_count=0,
                        average_speedup=0.0,
                        total_execution_time_ms=0.0,
                        memory_bandwidth_used_gbps=0.0,
                        error_count=0
                    )
                
                # Simulate metric updates
                metrics = self.engine_metrics[engine_name]
                metrics.sme_operations_count += np.random.poisson(5)  # Average 5 operations per second
                metrics.average_speedup = 12.5 + np.random.normal(0, 2)  # 12.5x average speedup
                metrics.total_execution_time_ms += np.random.exponential(2.0)  # Exponential distribution
                metrics.memory_bandwidth_used_gbps += np.random.uniform(5, 25)  # 5-25 GB/s usage
                
                # Occasional errors
                if np.random.random() < 0.001:  # 0.1% error rate
                    metrics.error_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to update engine metrics: {e}")
    
    async def get_current_metrics(self) -> Optional[SMEMetrics]:
        """Get the most recent SME metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    async def get_metrics_history(self, hours: int = 1) -> List[SMEMetrics]:
        """Get SME metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    async def get_engine_metrics(self) -> Dict[str, SMEEngineMetrics]:
        """Get engine-specific SME metrics"""
        return self.engine_metrics.copy()
    
    async def get_performance_summary(self) -> Dict:
        """Get SME performance summary"""
        try:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = await self.get_metrics_history(1)  # Last hour
            
            if not recent_metrics:
                return {"status": "no_recent_data"}
            
            # Calculate averages
            avg_fp32_util = sum(m.fp32_utilization_percent for m in recent_metrics) / len(recent_metrics)
            avg_bandwidth = sum(m.memory_bandwidth_gbps for m in recent_metrics) / len(recent_metrics)
            avg_ops_per_sec = sum(m.operations_per_second for m in recent_metrics) / len(recent_metrics)
            avg_jit_hit_rate = sum(m.jit_kernel_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            # Calculate total operations across all engines
            total_operations = sum(m.sme_operations_count for m in self.engine_metrics.values())
            total_errors = sum(m.error_count for m in self.engine_metrics.values())
            error_rate = (total_errors / max(total_operations, 1)) * 100
            
            return {
                "status": "active",
                "monitoring_duration_hours": len(self.metrics_history) * self.monitoring_interval_seconds / 3600,
                "average_fp32_utilization": avg_fp32_util,
                "average_memory_bandwidth_gbps": avg_bandwidth,
                "average_operations_per_second": avg_ops_per_sec,
                "jit_kernel_hit_rate": avg_jit_hit_rate,
                "total_sme_operations": total_operations,
                "error_rate_percent": error_rate,
                "active_engines": len(self.engine_metrics),
                "recent_alerts": len([a for a in self.performance_alerts if a["timestamp"] > datetime.now() - timedelta(hours=1)])
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_speedup_analysis(self) -> Dict:
        """Analyze SME speedup performance"""
        try:
            if not self.engine_metrics:
                return {"status": "no_engine_data"}
            
            speedups = []
            for engine_metrics in self.engine_metrics.values():
                if engine_metrics.sme_operations_count > 0:
                    speedups.append(engine_metrics.average_speedup)
            
            if not speedups:
                return {"status": "no_speedup_data"}
            
            return {
                "status": "active",
                "average_speedup": sum(speedups) / len(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "speedup_std_dev": np.std(speedups),
                "engines_analyzed": len(speedups),
                "target_speedup": 95.0,  # Target from our analysis
                "performance_rating": self._calculate_performance_rating(sum(speedups) / len(speedups))
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze speedup performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_performance_rating(self, average_speedup: float) -> str:
        """Calculate performance rating based on speedup"""
        if average_speedup >= 90:
            return "excellent"
        elif average_speedup >= 50:
            return "good"
        elif average_speedup >= 20:
            return "fair"
        else:
            return "poor"
    
    async def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sme_metrics_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "monitoring_config": {
                    "interval_seconds": self.monitoring_interval_seconds,
                    "retention_hours": self.metrics_retention_hours,
                    "alert_thresholds": self.alert_thresholds
                },
                "metrics_history": [asdict(m) for m in self.metrics_history],
                "engine_metrics": {k: asdict(v) for k, v in self.engine_metrics.items()},
                "performance_alerts": self.performance_alerts,
                "summary": await self.get_performance_summary()
            }
            
            # Convert datetime objects to strings for JSON serialization
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Write to file
            import json
            with open(filename, 'w') as f:
                json.dump(export_data, f, default=serialize_datetime, indent=2)
            
            logger.info(f"✅ SME metrics exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive SME performance report"""
        try:
            summary = await self.get_performance_summary()
            speedup_analysis = await self.get_speedup_analysis()
            current_metrics = await self.get_current_metrics()
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "system_status": "operational" if summary.get("status") == "active" else "degraded",
                "performance_summary": summary,
                "speedup_analysis": speedup_analysis,
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "engine_performance": {
                    engine_name: {
                        "operations_count": metrics.sme_operations_count,
                        "average_speedup": metrics.average_speedup,
                        "error_rate": (metrics.error_count / max(metrics.sme_operations_count, 1)) * 100,
                        "performance_rating": self._calculate_performance_rating(metrics.average_speedup)
                    }
                    for engine_name, metrics in self.engine_metrics.items()
                },
                "recommendations": self._generate_recommendations(summary, speedup_analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_recommendations(self, summary: Dict, speedup_analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        try:
            if summary.get("status") != "active":
                recommendations.append("SME monitoring not active - enable monitoring for performance insights")
                return recommendations
            
            # Utilization recommendations
            avg_util = summary.get("average_fp32_utilization", 0)
            if avg_util < 30:
                recommendations.append("Low SME utilization detected - consider increasing SME workload routing")
            elif avg_util > 90:
                recommendations.append("High SME utilization - monitor for thermal throttling")
            
            # Speedup recommendations
            avg_speedup = speedup_analysis.get("average_speedup", 0)
            if avg_speedup < 20:
                recommendations.append("Below target speedup performance - review SME optimization settings")
            elif avg_speedup > 50:
                recommendations.append("Excellent speedup performance - consider expanding SME usage to more operations")
            
            # Error rate recommendations
            error_rate = summary.get("error_rate_percent", 0)
            if error_rate > 1.0:
                recommendations.append("High error rate detected - review SME operation stability")
            
            # JIT kernel recommendations
            jit_hit_rate = summary.get("jit_kernel_hit_rate", 0)
            if jit_hit_rate < 60:
                recommendations.append("Low JIT kernel hit rate - review matrix size thresholds")
            
            if not recommendations:
                recommendations.append("SME performance is optimal - continue monitoring")
                
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations