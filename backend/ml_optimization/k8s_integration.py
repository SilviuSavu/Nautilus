"""
Kubernetes Integration for ML-Powered Auto-Scaling

This module provides integration between the ML optimization system
and Kubernetes HPA, enabling intelligent scaling decisions to be
applied to the actual infrastructure.
"""

import asyncio
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import redis

# Kubernetes imports
try:
    from kubernetes import client, config, watch
    from kubernetes.client import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logging.warning("Kubernetes client not available - using simulation mode")

from .ml_autoscaler import MLAutoScaler, ScalingDecision, TradingPattern
from .predictive_allocator import PredictiveResourceAllocator, AllocationStrategy
from .market_optimizer import MarketConditionOptimizer, OptimizationStrategy


@dataclass
class K8sResource:
    """Kubernetes resource configuration"""
    name: str
    namespace: str
    resource_type: str  # Deployment, StatefulSet, etc.
    current_replicas: int
    min_replicas: int
    max_replicas: int
    current_cpu: str
    current_memory: str
    hpa_enabled: bool = True


class K8sMLIntegrator:
    """
    Kubernetes integration for ML-powered optimization.
    
    This class bridges the ML optimization components with actual
    Kubernetes infrastructure, applying intelligent scaling and
    resource allocation decisions to real workloads.
    """
    
    def __init__(self, namespace: str = "nautilus-trading", redis_url: str = "redis://localhost:6379"):
        self.namespace = namespace
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # ML Components
        self.ml_autoscaler = MLAutoScaler(namespace, redis_url)
        self.resource_allocator = PredictiveResourceAllocator(redis_url)
        self.market_optimizer = MarketConditionOptimizer(redis_url)
        
        # Kubernetes clients
        self.k8s_apps_v1 = None
        self.k8s_autoscaling_v2 = None
        self.k8s_core_v1 = None
        
        # Configuration
        self.managed_services = [
            "nautilus-market-data",
            "nautilus-strategy-engine", 
            "nautilus-risk-engine",
            "nautilus-order-manager",
            "nautilus-position-keeper"
        ]
        
        self.scaling_cooldown = 300  # seconds
        self.last_scaling_actions = {}
        
        # Initialize Kubernetes connection
        self._initialize_k8s_clients()
    
    def _initialize_k8s_clients(self):
        """Initialize Kubernetes API clients"""
        if not KUBERNETES_AVAILABLE:
            self.logger.warning("Kubernetes not available - running in simulation mode")
            return
        
        try:
            # Try in-cluster config first
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster Kubernetes config")
        except:
            try:
                # Fall back to local config
                config.load_kube_config()
                self.logger.info("Loaded local Kubernetes config")
            except Exception as e:
                self.logger.error(f"Failed to load Kubernetes config: {str(e)}")
                return
        
        # Initialize API clients
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
        self.k8s_core_v1 = client.CoreV1Api()
        
        self.logger.info("Kubernetes clients initialized successfully")
    
    async def get_current_resources(self) -> Dict[str, K8sResource]:
        """Get current Kubernetes resource configurations"""
        resources = {}
        
        if not KUBERNETES_AVAILABLE or self.k8s_apps_v1 is None:
            # Return simulated resources
            return self._get_simulated_resources()
        
        try:
            for service_name in self.managed_services:
                try:
                    # Get deployment
                    deployment = self.k8s_apps_v1.read_namespaced_deployment(
                        name=service_name,
                        namespace=self.namespace
                    )
                    
                    # Get HPA if exists
                    hpa = None
                    hpa_enabled = False
                    min_replicas = 1
                    max_replicas = 10
                    
                    try:
                        hpa = self.k8s_autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                            name=f"{service_name}-hpa",
                            namespace=self.namespace
                        )
                        hpa_enabled = True
                        min_replicas = hpa.spec.min_replicas
                        max_replicas = hpa.spec.max_replicas
                    except ApiException:
                        pass  # HPA doesn't exist
                    
                    # Get resource limits
                    container = deployment.spec.template.spec.containers[0]
                    current_cpu = container.resources.limits.get('cpu', '1000m')
                    current_memory = container.resources.limits.get('memory', '2Gi')
                    
                    resources[service_name] = K8sResource(
                        name=service_name,
                        namespace=self.namespace,
                        resource_type="Deployment",
                        current_replicas=deployment.spec.replicas,
                        min_replicas=min_replicas,
                        max_replicas=max_replicas,
                        current_cpu=current_cpu,
                        current_memory=current_memory,
                        hpa_enabled=hpa_enabled
                    )
                    
                except ApiException as e:
                    self.logger.error(f"Error getting resource {service_name}: {str(e)}")
                    continue
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error getting current resources: {str(e)}")
            return self._get_simulated_resources()
    
    def _get_simulated_resources(self) -> Dict[str, K8sResource]:
        """Get simulated resource configurations for testing"""
        resources = {}
        
        resource_configs = {
            "nautilus-market-data": {"cpu": "2000m", "memory": "4Gi", "replicas": 3},
            "nautilus-strategy-engine": {"cpu": "3000m", "memory": "6Gi", "replicas": 2},
            "nautilus-risk-engine": {"cpu": "2500m", "memory": "3Gi", "replicas": 2},
            "nautilus-order-manager": {"cpu": "2000m", "memory": "2Gi", "replicas": 3},
            "nautilus-position-keeper": {"cpu": "1500m", "memory": "2Gi", "replicas": 2}
        }
        
        for service_name, config in resource_configs.items():
            resources[service_name] = K8sResource(
                name=service_name,
                namespace=self.namespace,
                resource_type="Deployment",
                current_replicas=config["replicas"],
                min_replicas=1,
                max_replicas=10,
                current_cpu=config["cpu"],
                current_memory=config["memory"],
                hpa_enabled=True
            )
        
        return resources
    
    async def apply_ml_scaling_decision(self, service_name: str) -> Dict[str, Any]:
        """Apply ML-powered scaling decision to a Kubernetes service"""
        try:
            # Check scaling cooldown
            last_action_time = self.last_scaling_actions.get(service_name)
            if last_action_time:
                time_since_last = (datetime.now() - last_action_time).total_seconds()
                if time_since_last < self.scaling_cooldown:
                    return {
                        "service": service_name,
                        "action": "skipped",
                        "reason": f"Cooldown period ({self.scaling_cooldown - time_since_last:.0f}s remaining)"
                    }
            
            # Get current resource configuration
            resources = await self.get_current_resources()
            if service_name not in resources:
                return {
                    "service": service_name,
                    "action": "error",
                    "reason": "Service not found in managed resources"
                }
            
            current_resource = resources[service_name]
            
            # Get ML scaling recommendation
            metrics = await self.ml_autoscaler.collect_metrics(service_name)
            prediction = await self.ml_autoscaler.predict_scaling_needs(metrics)
            
            # Get market condition context
            market_condition = await self.market_optimizer.analyze_market_condition()
            optimization_settings = await self.market_optimizer.optimize_for_market_condition(market_condition)
            
            # Apply market-aware adjustments to ML recommendations
            adjusted_prediction = self._apply_market_adjustments(prediction, optimization_settings)
            
            # Execute scaling decision
            if adjusted_prediction.scaling_recommendation != ScalingDecision.MAINTAIN:
                result = await self._execute_k8s_scaling(service_name, current_resource, adjusted_prediction)
                
                if result["success"]:
                    self.last_scaling_actions[service_name] = datetime.now()
                
                # Store decision history
                await self._store_scaling_decision(service_name, adjusted_prediction, result, market_condition)
                
                return result
            else:
                return {
                    "service": service_name,
                    "action": "maintain",
                    "current_replicas": current_resource.current_replicas,
                    "reason": "ML recommendation suggests no scaling needed"
                }
                
        except Exception as e:
            self.logger.error(f"Error applying ML scaling decision for {service_name}: {str(e)}")
            return {
                "service": service_name,
                "action": "error",
                "reason": str(e)
            }
    
    def _apply_market_adjustments(self, prediction, optimization_settings):
        """Apply market condition adjustments to ML predictions"""
        # Create adjusted prediction based on market conditions
        adjusted_prediction = prediction
        
        # Adjust based on optimization strategy
        if optimization_settings.strategy == OptimizationStrategy.RELIABILITY_FOCUSED:
            # Be more conservative, scale up more aggressively
            if prediction.scaling_recommendation in [ScalingDecision.SCALE_UP_MODERATE]:
                adjusted_prediction.scaling_recommendation = ScalingDecision.SCALE_UP_AGGRESSIVE
                adjusted_prediction.recommended_replicas += 1
        
        elif optimization_settings.strategy == OptimizationStrategy.COST_OPTIMIZED:
            # Be more aggressive about scaling down
            if prediction.scaling_recommendation == ScalingDecision.MAINTAIN:
                if prediction.predicted_load < 0.4:
                    adjusted_prediction.scaling_recommendation = ScalingDecision.SCALE_DOWN_MODERATE
                    adjusted_prediction.recommended_replicas = max(1, adjusted_prediction.recommended_replicas - 1)
        
        elif optimization_settings.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            # Scale up more readily for performance
            if prediction.predicted_load > 0.5:
                adjusted_prediction.recommended_replicas += 1
        
        return adjusted_prediction
    
    async def _execute_k8s_scaling(self, service_name: str, current_resource: K8sResource, prediction) -> Dict[str, Any]:
        """Execute scaling decision on Kubernetes"""
        try:
            if not KUBERNETES_AVAILABLE or self.k8s_autoscaling_v2 is None:
                # Simulate scaling action
                return {
                    "service": service_name,
                    "action": "simulated_scaling",
                    "success": True,
                    "current_replicas": current_resource.current_replicas,
                    "target_replicas": prediction.recommended_replicas,
                    "scaling_decision": prediction.scaling_recommendation.value,
                    "confidence": prediction.confidence
                }
            
            target_replicas = max(
                current_resource.min_replicas,
                min(current_resource.max_replicas, prediction.recommended_replicas)
            )
            
            if current_resource.hpa_enabled:
                # Update HPA min/max replicas to influence scaling
                hpa = self.k8s_autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                    name=f"{service_name}-hpa",
                    namespace=self.namespace
                )
                
                # Temporarily adjust HPA bounds to achieve target scaling
                if prediction.scaling_recommendation in [ScalingDecision.SCALE_UP_AGGRESSIVE, ScalingDecision.SCALE_UP_MODERATE]:
                    hpa.spec.min_replicas = min(target_replicas, hpa.spec.max_replicas)
                elif prediction.scaling_recommendation in [ScalingDecision.SCALE_DOWN_MODERATE, ScalingDecision.SCALE_DOWN_AGGRESSIVE]:
                    hpa.spec.max_replicas = max(target_replicas, hpa.spec.min_replicas)
                
                # Update HPA
                self.k8s_autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                    name=f"{service_name}-hpa",
                    namespace=self.namespace,
                    body=hpa
                )
                
                scaling_method = "hpa_adjustment"
                
            else:
                # Direct deployment scaling
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace=self.namespace
                )
                
                deployment.spec.replicas = target_replicas
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=self.namespace,
                    body=deployment
                )
                
                scaling_method = "direct_scaling"
            
            return {
                "service": service_name,
                "action": "scaled",
                "success": True,
                "method": scaling_method,
                "current_replicas": current_resource.current_replicas,
                "target_replicas": target_replicas,
                "scaling_decision": prediction.scaling_recommendation.value,
                "confidence": prediction.confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except ApiException as e:
            return {
                "service": service_name,
                "action": "scaling_failed",
                "success": False,
                "error": f"Kubernetes API error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "service": service_name,
                "action": "scaling_failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def apply_resource_allocation_plan(self) -> Dict[str, Any]:
        """Apply predictive resource allocation plan to Kubernetes"""
        try:
            # Create allocation plan
            allocation_plan = await self.resource_allocator.create_allocation_plan(
                strategy=AllocationStrategy.ML_OPTIMIZED
            )
            
            if not allocation_plan.allocations:
                return {
                    "status": "no_changes",
                    "message": "No resource allocation changes recommended",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Apply allocations
            results = []
            successful_allocations = 0
            failed_allocations = 0
            
            for allocation in allocation_plan.allocations:
                try:
                    result = await self._apply_single_allocation(allocation)
                    results.append(result)
                    
                    if result.get("success", False):
                        successful_allocations += 1
                    else:
                        failed_allocations += 1
                        
                except Exception as e:
                    results.append({
                        "service": allocation.service_name,
                        "resource_type": allocation.resource_type.value,
                        "success": False,
                        "error": str(e)
                    })
                    failed_allocations += 1
            
            # Store allocation plan execution results
            await self._store_allocation_results(allocation_plan, results)
            
            return {
                "status": "completed",
                "plan_id": allocation_plan.plan_id,
                "total_allocations": len(allocation_plan.allocations),
                "successful": successful_allocations,
                "failed": failed_allocations,
                "total_cost_impact": allocation_plan.total_cost,
                "expected_performance_gain": allocation_plan.expected_performance_gain,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error applying resource allocation plan: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _apply_single_allocation(self, allocation) -> Dict[str, Any]:
        """Apply a single resource allocation change"""
        try:
            if not KUBERNETES_AVAILABLE or self.k8s_apps_v1 is None:
                # Simulate allocation
                return {
                    "service": allocation.service_name,
                    "resource_type": allocation.resource_type.value,
                    "success": True,
                    "action": "simulated",
                    "change": allocation.allocation_change,
                    "new_allocation": allocation.recommended_allocation
                }
            
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=allocation.service_name,
                namespace=self.namespace
            )
            
            # Update resource limits
            container = deployment.spec.template.spec.containers[0]
            
            if allocation.resource_type.value == "cpu":
                new_cpu = f"{allocation.recommended_allocation * 1000:.0f}m"
                container.resources.limits['cpu'] = new_cpu
                container.resources.requests['cpu'] = new_cpu
                
            elif allocation.resource_type.value == "memory":
                new_memory = f"{allocation.recommended_allocation:.1f}Gi"
                container.resources.limits['memory'] = new_memory
                container.resources.requests['memory'] = new_memory
            
            # Apply changes
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=allocation.service_name,
                namespace=self.namespace,
                body=deployment
            )
            
            return {
                "service": allocation.service_name,
                "resource_type": allocation.resource_type.value,
                "success": True,
                "action": "updated",
                "change": allocation.allocation_change,
                "new_allocation": allocation.recommended_allocation,
                "justification": allocation.justification
            }
            
        except ApiException as e:
            return {
                "service": allocation.service_name,
                "resource_type": allocation.resource_type.value,
                "success": False,
                "error": f"Kubernetes API error: {str(e)}"
            }
        
        except Exception as e:
            return {
                "service": allocation.service_name,
                "resource_type": allocation.resource_type.value,
                "success": False,
                "error": str(e)
            }
    
    async def create_enhanced_hpa(self, service_name: str) -> Dict[str, Any]:
        """Create enhanced HPA with ML-powered custom metrics"""
        try:
            # Get current resource configuration
            resources = await self.get_current_resources()
            if service_name not in resources:
                raise ValueError(f"Service {service_name} not found")
            
            resource = resources[service_name]
            
            # Create enhanced HPA configuration
            hpa_config = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{service_name}-ml-hpa",
                    "namespace": self.namespace,
                    "labels": {
                        "app": service_name,
                        "component": "ml-autoscaling",
                        "managed-by": "nautilus-ml-optimizer"
                    }
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": service_name
                    },
                    "minReplicas": resource.min_replicas,
                    "maxReplicas": resource.max_replicas,
                    "metrics": [
                        # Traditional CPU metric
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": 70
                                }
                            }
                        },
                        # Traditional memory metric
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "memory",
                                "target": {
                                    "type": "Utilization",
                                    "averageUtilization": 80
                                }
                            }
                        },
                        # ML-powered custom metrics
                        {
                            "type": "Pods",
                            "pods": {
                                "metric": {
                                    "name": "ml_predicted_load"
                                },
                                "target": {
                                    "type": "AverageValue",
                                    "averageValue": "0.7"
                                }
                            }
                        },
                        {
                            "type": "Pods",
                            "pods": {
                                "metric": {
                                    "name": "trading_pattern_score"
                                },
                                "target": {
                                    "type": "AverageValue",
                                    "averageValue": "0.5"
                                }
                            }
                        },
                        {
                            "type": "External",
                            "external": {
                                "metric": {
                                    "name": "market_volatility_index"
                                },
                                "target": {
                                    "type": "Value",
                                    "value": "25"
                                }
                            }
                        }
                    ],
                    "behavior": {
                        "scaleUp": {
                            "stabilizationWindowSeconds": 60,
                            "policies": [
                                {
                                    "type": "Percent",
                                    "value": 100,
                                    "periodSeconds": 60
                                },
                                {
                                    "type": "Pods",
                                    "value": 2,
                                    "periodSeconds": 60
                                }
                            ]
                        },
                        "scaleDown": {
                            "stabilizationWindowSeconds": 300,
                            "policies": [
                                {
                                    "type": "Percent",
                                    "value": 50,
                                    "periodSeconds": 300
                                },
                                {
                                    "type": "Pods",
                                    "value": 1,
                                    "periodSeconds": 300
                                }
                            ]
                        }
                    }
                }
            }
            
            if KUBERNETES_AVAILABLE and self.k8s_autoscaling_v2:
                # Apply HPA to Kubernetes
                try:
                    # Delete existing HPA if exists
                    try:
                        self.k8s_autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                            name=f"{service_name}-ml-hpa",
                            namespace=self.namespace
                        )
                        await asyncio.sleep(2)  # Wait for deletion
                    except ApiException:
                        pass  # HPA doesn't exist
                    
                    # Create new HPA
                    hpa_body = client.V2HorizontalPodAutoscaler(
                        api_version=hpa_config["apiVersion"],
                        kind=hpa_config["kind"],
                        metadata=client.V1ObjectMeta(**hpa_config["metadata"]),
                        spec=client.V2HorizontalPodAutoscalerSpec(**hpa_config["spec"])
                    )
                    
                    created_hpa = self.k8s_autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                        namespace=self.namespace,
                        body=hpa_body
                    )
                    
                    return {
                        "service": service_name,
                        "action": "hpa_created",
                        "success": True,
                        "hpa_name": created_hpa.metadata.name,
                        "metrics_count": len(hpa_config["spec"]["metrics"]),
                        "config": hpa_config
                    }
                    
                except ApiException as e:
                    return {
                        "service": service_name,
                        "action": "hpa_creation_failed",
                        "success": False,
                        "error": str(e),
                        "config": hpa_config
                    }
            else:
                # Return configuration for manual application
                return {
                    "service": service_name,
                    "action": "hpa_config_generated",
                    "success": True,
                    "message": "Kubernetes not available - configuration generated for manual application",
                    "config": hpa_config
                }
                
        except Exception as e:
            return {
                "service": service_name,
                "action": "hpa_creation_error",
                "success": False,
                "error": str(e)
            }
    
    async def _store_scaling_decision(self, service_name: str, prediction, result: Dict[str, Any], market_condition):
        """Store scaling decision for analysis"""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "ml_prediction": {
                "predicted_load": prediction.predicted_load,
                "confidence": prediction.confidence,
                "pattern": prediction.pattern.value,
                "scaling_decision": prediction.scaling_recommendation.value,
                "recommended_replicas": prediction.recommended_replicas
            },
            "market_context": {
                "regime": market_condition.regime.value,
                "volatility_level": market_condition.volatility_level,
                "volume_profile": market_condition.volume_profile
            },
            "execution_result": result
        }
        
        # Store in Redis
        self.redis_client.lpush(
            f"k8s:scaling:history:{service_name}",
            json.dumps(decision_record)
        )
        self.redis_client.ltrim(f"k8s:scaling:history:{service_name}", 0, 499)
    
    async def _store_allocation_results(self, allocation_plan, results: List[Dict[str, Any]]):
        """Store allocation plan results"""
        plan_record = {
            "timestamp": datetime.now().isoformat(),
            "plan_id": allocation_plan.plan_id,
            "strategy": allocation_plan.strategy.value,
            "total_cost": allocation_plan.total_cost,
            "expected_performance_gain": allocation_plan.expected_performance_gain,
            "successful_allocations": len([r for r in results if r.get("success", False)]),
            "total_allocations": len(results),
            "results": results
        }
        
        # Store in Redis
        self.redis_client.lpush("k8s:allocation:history", json.dumps(plan_record))
        self.redis_client.ltrim("k8s:allocation:history", 0, 99)  # Keep last 100
    
    async def run_continuous_optimization(self):
        """Run continuous ML-powered optimization"""
        self.logger.info("Starting continuous ML-powered Kubernetes optimization")
        
        while True:
            try:
                optimization_start = datetime.now()
                
                # Apply ML scaling decisions for all managed services
                scaling_results = []
                for service in self.managed_services:
                    result = await self.apply_ml_scaling_decision(service)
                    scaling_results.append(result)
                
                # Apply resource allocation plan (less frequently)
                allocation_result = None
                if datetime.now().minute % 15 == 0:  # Every 15 minutes
                    allocation_result = await self.apply_resource_allocation_plan()
                
                # Log optimization results
                successful_scalings = len([r for r in scaling_results if r.get("success", False)])
                self.logger.info(
                    f"Optimization cycle: {successful_scalings}/{len(scaling_results)} successful scalings"
                )
                
                if allocation_result:
                    self.logger.info(
                        f"Resource allocation: {allocation_result['status']}, "
                        f"Cost impact: ${allocation_result.get('total_cost_impact', 0):.2f}"
                    )
                
                # Calculate cycle time
                cycle_time = (datetime.now() - optimization_start).total_seconds()
                
                # Wait for next cycle (5 minutes)
                await asyncio.sleep(max(0, 300 - cycle_time))
                
            except Exception as e:
                self.logger.error(f"Error in optimization cycle: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and metrics"""
        try:
            # Get current resources
            resources = await self.get_current_resources()
            
            # Get recent scaling decisions
            recent_decisions = []
            for service in self.managed_services:
                history_key = f"k8s:scaling:history:{service}"
                recent_history = self.redis_client.lrange(history_key, 0, 4)  # Last 5
                
                for record_str in recent_history:
                    try:
                        record = json.loads(record_str)
                        recent_decisions.append(record)
                    except:
                        continue
            
            # Sort by timestamp
            recent_decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Get allocation history
            allocation_history = []
            history_data = self.redis_client.lrange("k8s:allocation:history", 0, 4)
            for record_str in history_data:
                try:
                    record = json.loads(record_str)
                    allocation_history.append(record)
                except:
                    continue
            
            # Calculate summary statistics
            total_services = len(self.managed_services)
            hpa_enabled_services = len([r for r in resources.values() if r.hpa_enabled])
            
            recent_successful_scalings = len([
                d for d in recent_decisions[-20:] 
                if d.get('execution_result', {}).get('success', False)
            ])
            
            return {
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "managed_services": total_services,
                    "hpa_enabled": hpa_enabled_services,
                    "recent_successful_scalings": recent_successful_scalings,
                    "kubernetes_available": KUBERNETES_AVAILABLE
                },
                "current_resources": {
                    name: {
                        "replicas": resource.current_replicas,
                        "cpu": resource.current_cpu,
                        "memory": resource.current_memory,
                        "hpa_enabled": resource.hpa_enabled
                    }
                    for name, resource in resources.items()
                },
                "recent_decisions": recent_decisions[:10],
                "allocation_history": allocation_history[:5],
                "cooldown_status": {
                    service: {
                        "last_action": last_action.isoformat(),
                        "seconds_until_next": max(0, self.scaling_cooldown - 
                                                 (datetime.now() - last_action).total_seconds())
                    }
                    for service, last_action in self.last_scaling_actions.items()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test the Kubernetes ML Integration"""
    logging.basicConfig(level=logging.INFO)
    
    integrator = K8sMLIntegrator()
    
    print("🔧 Testing Kubernetes ML Integration")
    print("=" * 45)
    
    # Test getting current resources
    print("\n📊 Getting Current Resources...")
    resources = await integrator.get_current_resources()
    
    print(f"Found {len(resources)} managed services:")
    for name, resource in resources.items():
        print(f"  {name}: {resource.current_replicas} replicas, "
              f"{resource.current_cpu} CPU, {resource.current_memory} memory")
    
    # Test ML scaling decision for one service
    test_service = "nautilus-market-data"
    print(f"\n🤖 Testing ML Scaling Decision for {test_service}...")
    
    scaling_result = await integrator.apply_ml_scaling_decision(test_service)
    print(f"Scaling Result: {scaling_result['action']}")
    if 'target_replicas' in scaling_result:
        print(f"  Current: {scaling_result.get('current_replicas')} → Target: {scaling_result.get('target_replicas')}")
        print(f"  Confidence: {scaling_result.get('confidence', 0):.2f}")
    if 'reason' in scaling_result:
        print(f"  Reason: {scaling_result['reason']}")
    
    # Test resource allocation plan
    print(f"\n📈 Testing Resource Allocation Plan...")
    allocation_result = await integrator.apply_resource_allocation_plan()
    
    print(f"Allocation Status: {allocation_result['status']}")
    if allocation_result['status'] == 'completed':
        print(f"  Total Allocations: {allocation_result['total_allocations']}")
        print(f"  Successful: {allocation_result['successful']}")
        print(f"  Failed: {allocation_result['failed']}")
        print(f"  Cost Impact: ${allocation_result['total_cost_impact']:.2f}/hour")
        print(f"  Performance Gain: {allocation_result['expected_performance_gain']:.1f}%")
    
    # Test creating enhanced HPA
    print(f"\n⚡ Creating Enhanced HPA for {test_service}...")
    hpa_result = await integrator.create_enhanced_hpa(test_service)
    
    print(f"HPA Creation: {hpa_result['action']}")
    if hpa_result['success']:
        print(f"  HPA Name: {hpa_result.get('hpa_name', 'N/A')}")
        print(f"  Metrics Count: {hpa_result.get('metrics_count', 0)}")
    else:
        print(f"  Error: {hpa_result.get('error', 'Unknown error')}")
    
    # Get optimization status
    print(f"\n📋 Optimization Status...")
    status = await integrator.get_optimization_status()
    
    print(f"Status: {status['status']}")
    if status['status'] != 'error':
        summary = status['summary']
        print(f"  Managed Services: {summary['managed_services']}")
        print(f"  HPA Enabled: {summary['hpa_enabled']}")
        print(f"  Recent Successful Scalings: {summary['recent_successful_scalings']}")
        print(f"  Kubernetes Available: {summary['kubernetes_available']}")
        
        if status['recent_decisions']:
            print("  Recent Decisions:")
            for decision in status['recent_decisions'][:3]:
                service = decision.get('service', 'unknown')
                ml_pred = decision.get('ml_prediction', {})
                result = decision.get('execution_result', {})
                print(f"    {service}: {ml_pred.get('scaling_decision', 'N/A')} "
                      f"({result.get('action', 'N/A')})")
    else:
        print(f"  Error: {status['error']}")
    
    print("\n✅ Kubernetes ML Integration test completed!")
    print("\n💡 To run continuous optimization, call:")
    print("   await integrator.run_continuous_optimization()")


if __name__ == "__main__":
    asyncio.run(main())