#!/usr/bin/env python3
"""
Nautilus Prometheus Data Collector
Advanced data collection from Prometheus and Grafana containers for machine learning models.
Pulls real metrics from containerized monitoring stack.
"""

import asyncio
import aiohttp
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import sqlite3
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrometheusMetric:
    """Prometheus metric data point"""
    metric_name: str
    labels: Dict[str, str]
    value: float
    timestamp: datetime

@dataclass
class ContainerMetrics:
    """Container-specific metrics"""
    container_name: str
    cpu_usage: float
    memory_usage: float
    memory_limit: float
    network_rx: float
    network_tx: float
    disk_read: float
    disk_write: float
    status: str

class PrometheusDataCollector:
    """Collects real-time data from Prometheus for ML model training"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", 
                 grafana_url: str = "http://localhost:3002"):
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.session = None
        
        # Key metrics to collect for ML model training
        self.core_metrics = {
            # System metrics
            "system_cpu": 'rate(node_cpu_seconds_total[1m])',
            "system_memory": 'node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes',
            "system_load": 'node_load1',
            "network_rx": 'rate(node_network_receive_bytes_total[1m])',
            "network_tx": 'rate(node_network_transmit_bytes_total[1m])',
            
            # Container metrics  
            "container_cpu": 'rate(container_cpu_usage_seconds_total[1m])',
            "container_memory": 'container_memory_usage_bytes / container_spec_memory_limit_bytes',
            "container_network_rx": 'rate(container_network_receive_bytes_total[1m])',
            "container_network_tx": 'rate(container_network_transmit_bytes_total[1m])',
            
            # Redis metrics
            "redis_connected_clients": 'redis_connected_clients',
            "redis_used_memory": 'redis_memory_used_bytes',
            "redis_commands_per_sec": 'rate(redis_commands_processed_total[1m])',
            "redis_keyspace_hits": 'rate(redis_keyspace_hits_total[1m])',
            "redis_keyspace_misses": 'rate(redis_keyspace_misses_total[1m])',
            
            # PostgreSQL metrics
            "postgres_connections": 'pg_stat_database_numbackends',
            "postgres_transactions": 'rate(pg_stat_database_xact_commit[1m]) + rate(pg_stat_database_xact_rollback[1m])',
            "postgres_queries": 'rate(pg_stat_database_tup_fetched[1m])',
            "postgres_locks": 'pg_locks_count',
            
            # HTTP metrics (from engines)
            "http_requests": 'rate(prometheus_http_requests_total[1m])',
            "http_duration": 'prometheus_http_request_duration_seconds',
            "http_errors": 'rate(prometheus_http_requests_total{code!~"2.."}[1m])',
            
            # Custom Nautilus engine metrics (if exposed)
            "engine_latency": 'nautilus_engine_response_time_seconds',
            "engine_throughput": 'rate(nautilus_engine_requests_total[1m])',
            "engine_errors": 'rate(nautilus_engine_errors_total[1m])'
        }
        
        # Container names we're monitoring
        self.monitored_containers = [
            "nautilus-postgres",
            "nautilus-redis", 
            "nautilus-marketdata-bus",
            "nautilus-neural-gpu-bus",
            "nautilus-prometheus",
            "nautilus-grafana",
            "nautilus-backend",
            "nautilus-frontend"
        ]

    async def initialize(self):
        """Initialize HTTP session and verify connections"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Test Prometheus connection
        try:
            async with self.session.get(f"{self.prometheus_url}/api/v1/query?query=up") as resp:
                if resp.status == 200:
                    logger.info("✅ Connected to Prometheus")
                else:
                    logger.error(f"❌ Prometheus connection failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"❌ Cannot reach Prometheus: {e}")
        
        # Test Grafana connection  
        try:
            async with self.session.get(f"{self.grafana_url}/api/health") as resp:
                if resp.status == 200:
                    logger.info("✅ Connected to Grafana")
                else:
                    logger.error(f"❌ Grafana connection failed: HTTP {resp.status}")
        except Exception as e:
            logger.error(f"❌ Cannot reach Grafana: {e}")

    async def query_prometheus(self, query: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[PrometheusMetric]:
        """Query Prometheus for metrics"""
        
        # Build query parameters
        params = {"query": query}
        
        if time_range:
            start_time, end_time = time_range
            params.update({
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "step": "15s"  # 15 second resolution
            })
            endpoint = "/api/v1/query_range"
        else:
            endpoint = "/api/v1/query"
        
        try:
            url = f"{self.prometheus_url}{endpoint}"
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Prometheus query failed: {resp.status} for query: {query}")
                    return []
                
                data = await resp.json()
                
                if data.get("status") != "success":
                    logger.warning(f"Prometheus query unsuccessful: {data.get('error', 'unknown error')}")
                    return []
                
                metrics = []
                result = data.get("data", {}).get("result", [])
                
                for item in result:
                    metric_name = item.get("metric", {}).get("__name__", "unknown")
                    labels = {k: v for k, v in item.get("metric", {}).items() if k != "__name__"}
                    
                    if "values" in item:  # Time range query
                        for timestamp, value in item["values"]:
                            try:
                                metrics.append(PrometheusMetric(
                                    metric_name=metric_name,
                                    labels=labels,
                                    value=float(value),
                                    timestamp=datetime.fromtimestamp(float(timestamp))
                                ))
                            except (ValueError, TypeError):
                                continue
                    else:  # Instant query
                        timestamp, value = item.get("value", [time.time(), "0"])
                        try:
                            metrics.append(PrometheusMetric(
                                metric_name=metric_name,
                                labels=labels,
                                value=float(value),
                                timestamp=datetime.fromtimestamp(float(timestamp))
                            ))
                        except (ValueError, TypeError):
                            continue
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return []

    async def collect_container_metrics(self) -> List[ContainerMetrics]:
        """Collect detailed container metrics"""
        container_metrics = []
        
        for container_name in self.monitored_containers:
            try:
                # CPU usage
                cpu_query = f'rate(container_cpu_usage_seconds_total{{name="{container_name}"}}[1m])'
                cpu_metrics = await self.query_prometheus(cpu_query)
                cpu_usage = cpu_metrics[0].value if cpu_metrics else 0.0
                
                # Memory usage
                memory_query = f'container_memory_usage_bytes{{name="{container_name}"}}'
                memory_metrics = await self.query_prometheus(memory_query)
                memory_usage = memory_metrics[0].value if memory_metrics else 0.0
                
                # Memory limit
                memory_limit_query = f'container_spec_memory_limit_bytes{{name="{container_name}"}}'
                memory_limit_metrics = await self.query_prometheus(memory_limit_query)
                memory_limit = memory_limit_metrics[0].value if memory_limit_metrics else 1.0
                
                # Network metrics
                network_rx_query = f'rate(container_network_receive_bytes_total{{name="{container_name}"}}[1m])'
                network_rx_metrics = await self.query_prometheus(network_rx_query)
                network_rx = network_rx_metrics[0].value if network_rx_metrics else 0.0
                
                network_tx_query = f'rate(container_network_transmit_bytes_total{{name="{container_name}"}}[1m])'
                network_tx_metrics = await self.query_prometheus(network_tx_query)
                network_tx = network_tx_metrics[0].value if network_tx_metrics else 0.0
                
                # Disk I/O
                disk_read_query = f'rate(container_fs_reads_bytes_total{{name="{container_name}"}}[1m])'
                disk_read_metrics = await self.query_prometheus(disk_read_query)
                disk_read = disk_read_metrics[0].value if disk_read_metrics else 0.0
                
                disk_write_query = f'rate(container_fs_writes_bytes_total{{name="{container_name}"}}[1m])'
                disk_write_metrics = await self.query_prometheus(disk_write_query)
                disk_write = disk_write_metrics[0].value if disk_write_metrics else 0.0
                
                # Container status
                status_query = f'container_last_seen{{name="{container_name}"}}'
                status_metrics = await self.query_prometheus(status_query)
                status = "running" if status_metrics else "unknown"
                
                container_metrics.append(ContainerMetrics(
                    container_name=container_name,
                    cpu_usage=cpu_usage * 100,  # Convert to percentage
                    memory_usage=memory_usage,
                    memory_limit=memory_limit,
                    network_rx=network_rx,
                    network_tx=network_tx,
                    disk_read=disk_read,
                    disk_write=disk_write,
                    status=status
                ))
                
            except Exception as e:
                logger.error(f"Error collecting metrics for {container_name}: {e}")
                # Add placeholder metrics
                container_metrics.append(ContainerMetrics(
                    container_name=container_name,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    memory_limit=1.0,
                    network_rx=0.0,
                    network_tx=0.0,
                    disk_read=0.0,
                    disk_write=0.0,
                    status="error"
                ))
        
        return container_metrics

    async def collect_comprehensive_metrics(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Collect comprehensive metrics over a time period"""
        logger.info(f"🔍 Collecting comprehensive metrics for {duration_minutes} minutes...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)
        time_range = (start_time, end_time)
        
        all_metrics = {}
        
        # Collect each metric type
        for metric_name, query in self.core_metrics.items():
            logger.info(f"📊 Collecting {metric_name}...")
            metrics = await self.query_prometheus(query, time_range)
            
            if metrics:
                # Convert to DataFrame for easier analysis
                data = []
                for metric in metrics:
                    data.append({
                        "timestamp": metric.timestamp,
                        "value": metric.value,
                        "labels": json.dumps(metric.labels)
                    })
                
                df = pd.DataFrame(data)
                all_metrics[metric_name] = {
                    "raw_data": data,
                    "summary": {
                        "count": len(data),
                        "mean": df["value"].mean() if len(data) > 0 else 0,
                        "median": df["value"].median() if len(data) > 0 else 0,
                        "std": df["value"].std() if len(data) > 0 else 0,
                        "min": df["value"].min() if len(data) > 0 else 0,
                        "max": df["value"].max() if len(data) > 0 else 0,
                        "p95": df["value"].quantile(0.95) if len(data) > 0 else 0,
                        "p99": df["value"].quantile(0.99) if len(data) > 0 else 0
                    }
                }
            else:
                all_metrics[metric_name] = {
                    "raw_data": [],
                    "summary": {
                        "count": 0, "mean": 0, "median": 0, "std": 0,
                        "min": 0, "max": 0, "p95": 0, "p99": 0
                    }
                }
            
            # Small delay to avoid overwhelming Prometheus
            await asyncio.sleep(0.1)
        
        # Collect container metrics
        container_metrics = await self.collect_container_metrics()
        
        return {
            "collection_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes
            },
            "system_metrics": all_metrics,
            "container_metrics": {
                container.container_name: asdict(container)
                for container in container_metrics
            },
            "collection_summary": {
                "total_metrics": len(all_metrics),
                "successful_metrics": len([m for m in all_metrics.values() if m["summary"]["count"] > 0]),
                "total_data_points": sum(m["summary"]["count"] for m in all_metrics.values()),
                "containers_monitored": len(container_metrics)
            }
        }

    async def create_ml_training_dataset(self, hours_back: int = 24) -> pd.DataFrame:
        """Create ML training dataset from Prometheus data"""
        logger.info(f"🧠 Creating ML training dataset from last {hours_back} hours...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        time_range = (start_time, end_time)
        
        # Collect key metrics for ML training
        ml_metrics = {
            "cpu_usage": 'rate(node_cpu_seconds_total{mode="user"}[1m])',
            "memory_usage": '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
            "network_io": 'rate(node_network_receive_bytes_total[1m]) + rate(node_network_transmit_bytes_total[1m])',
            "redis_ops": 'rate(redis_commands_processed_total[1m])',
            "postgres_connections": 'pg_stat_database_numbackends',
            "http_requests": 'rate(prometheus_http_requests_total[1m])',
            "response_time": 'prometheus_http_request_duration_seconds',
            "error_rate": 'rate(prometheus_http_requests_total{code!~"2.."}[1m]) / rate(prometheus_http_requests_total[1m])'
        }
        
        dataset = []
        
        for metric_name, query in ml_metrics.items():
            metrics = await self.query_prometheus(query, time_range)
            
            for metric in metrics:
                dataset.append({
                    "timestamp": metric.timestamp,
                    "metric_name": metric_name,
                    "value": metric.value,
                    "hour": metric.timestamp.hour,
                    "day_of_week": metric.timestamp.weekday(),
                    "labels": json.dumps(metric.labels)
                })
        
        df = pd.DataFrame(dataset)
        
        if len(df) > 0:
            # Pivot to get features as columns
            df_pivot = df.pivot_table(
                index=['timestamp', 'hour', 'day_of_week'], 
                columns='metric_name', 
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Fill NaN values
            df_pivot = df_pivot.fillna(0)
            
            # Add derived features
            if 'cpu_usage' in df_pivot.columns and 'memory_usage' in df_pivot.columns:
                df_pivot['system_load'] = df_pivot['cpu_usage'] + (df_pivot['memory_usage'] / 100)
            
            if 'http_requests' in df_pivot.columns and 'response_time' in df_pivot.columns:
                df_pivot['throughput_quality'] = df_pivot['http_requests'] / (df_pivot['response_time'] + 0.001)
            
            logger.info(f"✅ Created ML dataset with {len(df_pivot)} rows and {len(df_pivot.columns)} features")
            return df_pivot
        else:
            logger.warning("⚠️ No data available for ML dataset creation")
            return pd.DataFrame()

    async def save_metrics_to_database(self, metrics: Dict[str, Any], db_path: str = "prometheus_metrics.db"):
        """Save collected metrics to SQLite database"""
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prometheus_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time DATETIME,
                metric_name TEXT,
                metric_value REAL,
                labels TEXT,
                summary_stats TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS container_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time DATETIME,
                container_name TEXT,
                cpu_usage REAL,
                memory_usage REAL,
                memory_limit REAL,
                network_rx REAL,
                network_tx REAL,
                disk_read REAL,
                disk_write REAL,
                status TEXT
            )
        ''')
        
        collection_time = datetime.now()
        
        # Insert system metrics
        for metric_name, data in metrics.get("system_metrics", {}).items():
            cursor.execute('''
                INSERT INTO prometheus_metrics 
                (collection_time, metric_name, metric_value, labels, summary_stats)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                collection_time,
                metric_name,
                data["summary"]["mean"],
                "",  # We could store labels here if needed
                json.dumps(data["summary"])
            ))
        
        # Insert container metrics
        for container_name, container_data in metrics.get("container_metrics", {}).items():
            cursor.execute('''
                INSERT INTO container_metrics
                (collection_time, container_name, cpu_usage, memory_usage, memory_limit,
                 network_rx, network_tx, disk_read, disk_write, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                collection_time,
                container_name,
                container_data["cpu_usage"],
                container_data["memory_usage"],
                container_data["memory_limit"],
                container_data["network_rx"],
                container_data["network_tx"],
                container_data["disk_read"],
                container_data["disk_write"],
                container_data["status"]
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Saved metrics to {db_path}")

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

async def main():
    """Test the Prometheus data collector"""
    collector = PrometheusDataCollector()
    
    try:
        await collector.initialize()
        
        # Collect 5 minutes of comprehensive metrics
        metrics = await collector.collect_comprehensive_metrics(duration_minutes=5)
        
        # Save to database
        await collector.save_metrics_to_database(metrics)
        
        # Create ML dataset
        ml_dataset = await collector.create_ml_training_dataset(hours_back=1)
        
        if not ml_dataset.empty:
            # Save dataset to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"ml_training_data_{timestamp}.csv"
            ml_dataset.to_csv(csv_filename, index=False)
            logger.info(f"📈 ML dataset saved to {csv_filename}")
        
        # Save comprehensive metrics to JSON
        json_filename = f"prometheus_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive metrics saved to {json_filename}")
        
        # Print summary
        summary = metrics.get("collection_summary", {})
        print(f"""
🎯 Data Collection Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Total Metrics Collected: {summary.get('total_metrics', 0)}
📊 Successful Metrics: {summary.get('successful_metrics', 0)}
📈 Total Data Points: {summary.get('total_data_points', 0)}
🐳 Containers Monitored: {summary.get('containers_monitored', 0)}
🧠 ML Dataset Shape: {ml_dataset.shape if not ml_dataset.empty else 'No data'}
        """)
        
    except Exception as e:
        logger.error(f"❌ Error in data collection: {e}")
    
    finally:
        await collector.cleanup()

if __name__ == "__main__":
    print("📊 Nautilus Prometheus Data Collector")
    print("=" * 40)
    asyncio.run(main())