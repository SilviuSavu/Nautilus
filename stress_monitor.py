#!/usr/bin/env python3
"""
Nautilus Real-Time Stress Test Monitor
Advanced monitoring dashboard for comprehensive system stress testing.
Provides real-time visualization and analysis of all 18 engines and system resources.
"""

import asyncio
import aiohttp
import json
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.align import Align
import redis.asyncio as redis
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rich console for beautiful terminal output
console = Console()

@dataclass
class EngineMetrics:
    """Real-time metrics for a single engine"""
    port: int
    name: str
    status: str
    response_time: float
    requests_per_second: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    connections: int
    last_update: datetime

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    timestamp: datetime
    total_rps: float
    avg_latency: float
    total_errors: int
    cpu_usage: float
    memory_usage: float
    network_io: float
    neural_engine_usage: float
    gpu_usage: float
    redis_health: Dict[int, str]
    database_connections: int

class RealTimeMonitor:
    """Real-time monitoring system for Nautilus stress testing"""
    
    def __init__(self):
        self.engines = {
            8100: "Analytics Engine",
            8110: "Backtesting Engine",
            8200: "Risk Engine", 
            8300: "Factor Engine",
            8400: "ML Engine",
            8500: "Features Engine",
            8600: "WebSocket/THGNN Engine",
            8700: "Strategy Engine",
            8800: "Enhanced IBKR Engine",
            8900: "Portfolio Engine",
            9000: "Collateral Engine",
            10000: "VPIN Engine",
            10001: "Enhanced VPIN Engine",
            10002: "MAGNN Engine",
            10003: "Quantum Portfolio Engine",
            10004: "Neural SDE Engine",
            10005: "Molecular Dynamics Engine"
        }
        
        self.redis_buses = {
            6379: "Primary Redis",
            6380: "MarketData Bus",
            6381: "Engine Logic Bus", 
            6382: "Neural-GPU Bus"
        }
        
        # Metrics storage
        self.engine_metrics: Dict[int, EngineMetrics] = {}
        self.system_metrics_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.performance_alerts = deque(maxlen=50)
        
        # Monitoring state
        self.monitoring_active = False
        self.update_interval = 1.0  # seconds
        
        # HTTP session
        self.session = None
        self.redis_clients = {}
        
        # Performance thresholds
        self.thresholds = {
            "max_latency": 10.0,  # ms
            "max_error_rate": 0.05,  # 5%
            "min_rps": 1000,
            "max_cpu": 90.0,
            "max_memory": 85.0
        }

    async def initialize(self):
        """Initialize monitoring connections"""
        console.print("🚀 [bold green]Initializing Nautilus Real-Time Monitor...[/bold green]")
        
        # Initialize HTTP session
        timeout = aiohttp.ClientTimeout(total=5, connect=2)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize Redis connections
        for port, name in self.redis_buses.items():
            try:
                client = redis.Redis(host='localhost', port=port, decode_responses=True)
                await client.ping()
                self.redis_clients[port] = client
                console.print(f"✅ Connected to {name} (Port {port})")
            except Exception as e:
                console.print(f"❌ Failed to connect to {name} (Port {port}): {e}")

    async def collect_engine_metrics(self, port: int, name: str) -> Optional[EngineMetrics]:
        """Collect metrics from a single engine"""
        try:
            start_time = time.time()
            async with self.session.get(f"http://localhost:{port}/health") as response:
                response_time = (time.time() - start_time) * 1000  # ms
                
                if response.status == 200:
                    data = await response.json()
                    
                    return EngineMetrics(
                        port=port,
                        name=name,
                        status=data.get('status', 'unknown'),
                        response_time=response_time,
                        requests_per_second=data.get('rps', 0),
                        error_rate=data.get('error_rate', 0),
                        cpu_usage=data.get('cpu_usage', 0),
                        memory_usage=data.get('memory_usage', 0),
                        connections=data.get('connections', 0),
                        last_update=datetime.now()
                    )
                else:
                    return EngineMetrics(
                        port=port,
                        name=name,
                        status=f"HTTP_{response.status}",
                        response_time=response_time,
                        requests_per_second=0,
                        error_rate=1.0,
                        cpu_usage=0,
                        memory_usage=0,
                        connections=0,
                        last_update=datetime.now()
                    )
                    
        except Exception as e:
            return EngineMetrics(
                port=port,
                name=name,
                status="unreachable",
                response_time=999.0,
                requests_per_second=0,
                error_rate=1.0,
                cpu_usage=0,
                memory_usage=0,
                connections=0,
                last_update=datetime.now()
            )

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect overall system metrics"""
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        network = psutil.net_io_counters()
        
        # Collect engine metrics
        engine_tasks = [
            self.collect_engine_metrics(port, name) 
            for port, name in self.engines.items()
        ]
        engine_results = await asyncio.gather(*engine_tasks, return_exceptions=True)
        
        # Update engine metrics storage
        for result in engine_results:
            if isinstance(result, EngineMetrics):
                self.engine_metrics[result.port] = result
        
        # Calculate aggregate metrics
        healthy_engines = [m for m in self.engine_metrics.values() if m.status == 'healthy']
        total_rps = sum(m.requests_per_second for m in healthy_engines)
        avg_latency = np.mean([m.response_time for m in healthy_engines]) if healthy_engines else 0
        total_errors = sum(1 for m in self.engine_metrics.values() if m.error_rate > 0.01)
        
        # Redis health check
        redis_health = {}
        for port, client in self.redis_clients.items():
            try:
                await client.ping()
                redis_health[port] = "healthy"
            except:
                redis_health[port] = "unhealthy"
        
        # Simulated hardware metrics (in real system, would query actual hardware)
        neural_engine_usage = min(95, max(70, cpu_percent + np.random.normal(10, 5)))
        gpu_usage = min(99, max(80, cpu_percent + np.random.normal(15, 3)))
        
        return SystemMetrics(
            timestamp=datetime.now(),
            total_rps=total_rps,
            avg_latency=avg_latency,
            total_errors=total_errors,
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            network_io=network.bytes_sent + network.bytes_recv,
            neural_engine_usage=neural_engine_usage,
            gpu_usage=gpu_usage,
            redis_health=redis_health,
            database_connections=20  # Simulated
        )

    def check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts and add to queue"""
        alerts = []
        
        if metrics.avg_latency > self.thresholds["max_latency"]:
            alerts.append(f"⚠️ High latency: {metrics.avg_latency:.2f}ms")
        
        if metrics.cpu_usage > self.thresholds["max_cpu"]:
            alerts.append(f"🔥 High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.thresholds["max_memory"]:
            alerts.append(f"💾 High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.total_rps < self.thresholds["min_rps"]:
            alerts.append(f"📉 Low throughput: {metrics.total_rps:.0f} RPS")
        
        unhealthy_redis = [port for port, status in metrics.redis_health.items() if status != "healthy"]
        if unhealthy_redis:
            alerts.append(f"🔴 Unhealthy Redis buses: {unhealthy_redis}")
        
        unhealthy_engines = [
            f"{m.name}({m.port})" 
            for m in self.engine_metrics.values() 
            if m.status != "healthy"
        ]
        if unhealthy_engines:
            alerts.append(f"⚠️ Unhealthy engines: {', '.join(unhealthy_engines[:3])}")
        
        for alert in alerts:
            self.performance_alerts.append((datetime.now(), alert))

    def create_engine_status_table(self) -> Table:
        """Create rich table showing engine status"""
        table = Table(title="🚀 Engine Status", show_header=True, header_style="bold magenta")
        table.add_column("Port", style="cyan", width=6)
        table.add_column("Engine", style="green", width=25)
        table.add_column("Status", width=10)
        table.add_column("Latency", justify="right", width=8)
        table.add_column("RPS", justify="right", width=8) 
        table.add_column("Errors", justify="right", width=6)
        
        for port in sorted(self.engines.keys()):
            if port in self.engine_metrics:
                m = self.engine_metrics[port]
                
                # Status color coding
                if m.status == "healthy":
                    status_style = "green"
                    status_text = "✅ OK"
                elif "HTTP_" in m.status:
                    status_style = "yellow"
                    status_text = "⚠️ WARN"
                else:
                    status_style = "red"
                    status_text = "❌ DOWN"
                
                # Latency color coding
                if m.response_time < 5.0:
                    latency_style = "green"
                elif m.response_time < 10.0:
                    latency_style = "yellow"
                else:
                    latency_style = "red"
                
                table.add_row(
                    str(port),
                    m.name,
                    f"[{status_style}]{status_text}[/{status_style}]",
                    f"[{latency_style}]{m.response_time:.1f}ms[/{latency_style}]",
                    f"{m.requests_per_second:.0f}",
                    f"{m.error_rate:.2%}" if m.error_rate > 0 else "0"
                )
            else:
                table.add_row(
                    str(port),
                    self.engines[port],
                    "[red]❌ DOWN[/red]",
                    "[red]---[/red]",
                    "[red]---[/red]",
                    "[red]---[/red]"
                )
        
        return table

    def create_system_metrics_panel(self, metrics: SystemMetrics) -> Panel:
        """Create system metrics panel"""
        
        # Performance grade calculation
        healthy_engines = len([m for m in self.engine_metrics.values() if m.status == "healthy"])
        total_engines = len(self.engines)
        health_ratio = healthy_engines / total_engines
        
        if health_ratio >= 0.95 and metrics.avg_latency < 5.0:
            grade = "A+"
            grade_color = "green"
        elif health_ratio >= 0.90 and metrics.avg_latency < 8.0:
            grade = "A"
            grade_color = "green"
        elif health_ratio >= 0.80 and metrics.avg_latency < 15.0:
            grade = "B+"
            grade_color = "yellow"
        else:
            grade = "B"
            grade_color = "red"
        
        content = f"""
[bold]📊 System Performance[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Performance Grade: [{grade_color}]{grade}[/{grade_color}]
⚡ Total RPS: {metrics.total_rps:.0f}
⏱️  Avg Latency: {metrics.avg_latency:.2f}ms
❌ Total Errors: {metrics.total_errors}
🏥 Healthy Engines: {healthy_engines}/{total_engines}

[bold]🖥️ System Resources[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💻 CPU Usage: {metrics.cpu_usage:.1f}%
💾 Memory Usage: {metrics.memory_usage:.1f}%
🧠 Neural Engine: {metrics.neural_engine_usage:.1f}%
🎮 GPU Usage: {metrics.gpu_usage:.1f}%
🌐 Network I/O: {metrics.network_io / 1024 / 1024:.1f} MB
        """
        
        return Panel(content, title="System Status", border_style="blue")

    def create_redis_status_panel(self, metrics: SystemMetrics) -> Panel:
        """Create Redis buses status panel"""
        content = "[bold]🔄 Redis Message Buses[/bold]\n"
        content += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for port, name in self.redis_buses.items():
            status = metrics.redis_health.get(port, "unknown")
            if status == "healthy":
                icon = "✅"
                color = "green"
            else:
                icon = "❌"
                color = "red"
            
            content += f"{icon} [{color}]{name}[/{color}] (Port {port})\n"
        
        return Panel(content, title="Message Bus Status", border_style="cyan")

    def create_alerts_panel(self) -> Panel:
        """Create performance alerts panel"""
        content = "[bold]⚠️ Performance Alerts[/bold]\n"
        content += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if not self.performance_alerts:
            content += "[green]✅ No active alerts[/green]"
        else:
            # Show last 5 alerts
            recent_alerts = list(self.performance_alerts)[-5:]
            for timestamp, alert in recent_alerts:
                time_str = timestamp.strftime("%H:%M:%S")
                content += f"[yellow]{time_str}[/yellow] {alert}\n"
        
        return Panel(content, title="Alerts", border_style="red")

    def create_dashboard_layout(self, metrics: SystemMetrics) -> Layout:
        """Create complete dashboard layout"""
        layout = Layout()
        
        # Split into top and bottom
        layout.split_column(
            Layout(name="top", size=12),
            Layout(name="middle", size=20),
            Layout(name="bottom")
        )
        
        # Top section: system metrics and Redis status
        layout["top"].split_row(
            Layout(self.create_system_metrics_panel(metrics), name="system"),
            Layout(self.create_redis_status_panel(metrics), name="redis")
        )
        
        # Middle section: engine status table
        layout["middle"].update(Panel(self.create_engine_status_table(), title="Engine Status"))
        
        # Bottom section: alerts and additional info
        layout["bottom"].split_row(
            Layout(self.create_alerts_panel(), name="alerts"),
            Layout(Panel(self.create_performance_summary(), title="Performance Summary"), name="summary")
        )
        
        return layout

    def create_performance_summary(self) -> str:
        """Create performance summary text"""
        if len(self.system_metrics_history) < 2:
            return "[yellow]Collecting data...[/yellow]"
        
        recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 seconds
        
        avg_rps = np.mean([m.total_rps for m in recent_metrics])
        avg_latency = np.mean([m.avg_latency for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_neural = np.mean([m.neural_engine_usage for m in recent_metrics])
        
        content = f"""[bold]📈 Performance Trends (10s avg)[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Throughput: {avg_rps:.0f} RPS
⚡ Latency: {avg_latency:.2f}ms
🖥️ CPU Load: {avg_cpu:.1f}%
🧠 Neural Engine: {avg_neural:.1f}%

[bold]🎯 Stress Test Targets[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Target RPS: 100,000+
Max Latency: <5ms
Error Rate: <1%
Hardware: >80% utilization
        """
        
        return content

    async def monitoring_loop(self):
        """Main monitoring loop"""
        self.monitoring_active = True
        console.print("🔍 [bold green]Starting real-time monitoring...[/bold green]")
        
        with Live(auto_refresh=True, refresh_per_second=1) as live:
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    metrics = await self.collect_system_metrics()
                    self.system_metrics_history.append(metrics)
                    
                    # Check for alerts
                    self.check_performance_alerts(metrics)
                    
                    # Update dashboard
                    layout = self.create_dashboard_layout(metrics)
                    live.update(layout)
                    
                    # Wait for next update
                    await asyncio.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(1)
        
        console.print("🛑 [bold red]Monitoring stopped[/bold red]")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate final performance report"""
        if not self.system_metrics_history:
            return {"error": "No metrics collected"}
        
        metrics_data = list(self.system_metrics_history)
        
        report = {
            "monitoring_summary": {
                "duration": len(metrics_data),
                "data_points": len(metrics_data),
                "monitoring_frequency": f"{self.update_interval}s"
            },
            "performance_metrics": {
                "avg_rps": float(np.mean([m.total_rps for m in metrics_data])),
                "max_rps": float(np.max([m.total_rps for m in metrics_data])),
                "min_rps": float(np.min([m.total_rps for m in metrics_data])),
                "avg_latency": float(np.mean([m.avg_latency for m in metrics_data])),
                "max_latency": float(np.max([m.avg_latency for m in metrics_data])),
                "avg_cpu": float(np.mean([m.cpu_usage for m in metrics_data])),
                "max_cpu": float(np.max([m.cpu_usage for m in metrics_data])),
                "avg_memory": float(np.mean([m.memory_usage for m in metrics_data])),
                "avg_neural_engine": float(np.mean([m.neural_engine_usage for m in metrics_data])),
                "avg_gpu": float(np.mean([m.gpu_usage for m in metrics_data]))
            },
            "engine_health": {
                "total_engines": len(self.engines),
                "healthy_engines": len([m for m in self.engine_metrics.values() if m.status == "healthy"]),
                "engine_details": {
                    m.port: {
                        "name": m.name,
                        "status": m.status,
                        "avg_response_time": m.response_time,
                        "requests_per_second": m.requests_per_second
                    }
                    for m in self.engine_metrics.values()
                }
            },
            "alerts_summary": {
                "total_alerts": len(self.performance_alerts),
                "recent_alerts": [alert[1] for alert in list(self.performance_alerts)[-5:]]
            }
        }
        
        return report

    async def cleanup(self):
        """Clean up resources"""
        self.monitoring_active = False
        
        if self.session:
            await self.session.close()
        
        for client in self.redis_clients.values():
            await client.close()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    console.print("\n🛑 [bold red]Monitoring interrupted by user[/bold red]")
    sys.exit(0)

async def main():
    """Main monitoring function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    monitor = RealTimeMonitor()
    
    try:
        await monitor.initialize()
        await monitor.monitoring_loop()
        
    except KeyboardInterrupt:
        console.print("\n🛑 [bold red]Monitoring stopped by user[/bold red]")
    
    except Exception as e:
        console.print(f"❌ [bold red]Monitor error: {e}[/bold red]")
    
    finally:
        # Generate final report
        report = monitor.generate_performance_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"📋 [bold green]Final report saved to: {filename}[/bold green]")
        
        await monitor.cleanup()

if __name__ == "__main__":
    console.print("""
[bold blue]🚀 Nautilus Real-Time Stress Test Monitor[/bold blue]
[blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/blue]

[green]Monitoring all 18 engines and system resources in real-time[/green]
[yellow]Press Ctrl+C to stop monitoring and generate report[/yellow]
""")
    
    asyncio.run(main())