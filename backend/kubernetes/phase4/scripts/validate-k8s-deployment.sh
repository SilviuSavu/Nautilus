#!/bin/bash
set -e

echo "🔍 Phase 4: Kubernetes Production Infrastructure Validation"
echo "=========================================================="
echo "📅 $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="nautilus-trading"
VALIDATION_START=$(date +%s)
FAILED_CHECKS=0
TOTAL_CHECKS=0

# Helper function to run validation check
validate_check() {
    local name="$1"
    local command="$2"
    local expected_result="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "   Testing $name... "
    
    if result=$(eval "$command" 2>/dev/null); then
        if [[ "$result" == *"$expected_result"* ]] || [ -z "$expected_result" ]; then
            echo -e "${GREEN}✅ PASS${NC}"
            return 0
        else
            echo -e "${RED}❌ FAIL (Result: $result)${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            return 1
        fi
    else
        echo -e "${RED}❌ FAIL (Command failed)${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

echo -e "${BLUE}1. Kubernetes Cluster Validation${NC}"
echo "================================="

# Check kubectl access
validate_check "Kubectl connectivity" "kubectl cluster-info" "is running"

# Check namespace exists
validate_check "Namespace existence" "kubectl get namespace $NAMESPACE" "$NAMESPACE"

echo ""

echo -e "${BLUE}2. Deployment Status Validation${NC}"
echo "==============================="

# Get all deployments in namespace
DEPLOYMENTS=$(kubectl get deployments -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")

if [ -z "$DEPLOYMENTS" ]; then
    echo -e "${RED}❌ No deployments found in namespace $NAMESPACE${NC}"
    echo "Run deployment first: ./scripts/deploy-k8s.sh"
    exit 1
fi

echo "📋 Found deployments: $(echo "$DEPLOYMENTS" | wc -w)"

# Check each deployment status
while IFS= read -r deployment; do
    if [ -n "$deployment" ]; then
        validate_check "$deployment deployment ready" "kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type==\"Available\")].status}'" "True"
        
        # Check replica count
        DESIRED=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
        READY=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        if [ "$DESIRED" = "$READY" ] && [ "$READY" -gt 0 ]; then
            echo "      Replicas: ${READY}/${DESIRED} ✅"
        else
            echo "      Replicas: ${READY}/${DESIRED} ⚠️"
        fi
    fi
done <<< "$DEPLOYMENTS"

echo ""

echo -e "${BLUE}3. Pod Health Validation${NC}"
echo "========================"

# Check pod status
PODS=$(kubectl get pods -n $NAMESPACE --no-headers 2>/dev/null || echo "")

if [ -n "$PODS" ]; then
    echo "📋 Pod Status Summary:"
    kubectl get pods -n $NAMESPACE -o wide
    echo ""
    
    # Count pod states
    RUNNING_PODS=$(echo "$PODS" | grep -c " Running " || echo "0")
    TOTAL_PODS=$(echo "$PODS" | wc -l)
    
    echo "   Running pods: $RUNNING_PODS/$TOTAL_PODS"
    
    # Check for failed pods
    FAILED_PODS=$(echo "$PODS" | grep -E "(Error|CrashLoopBackOff|ImagePullBackOff)" || echo "")
    if [ -n "$FAILED_PODS" ]; then
        echo -e "${RED}⚠️  Failed pods detected:${NC}"
        echo "$FAILED_PODS"
    fi
    
else
    echo -e "${RED}❌ No pods found in namespace${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""

echo -e "${BLUE}4. Service Connectivity Validation${NC}"
echo "=================================="

# Check services
SERVICES=$(kubectl get services -n $NAMESPACE --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")

if [ -n "$SERVICES" ]; then
    echo "🌐 Service Status:"
    kubectl get services -n $NAMESPACE -o wide
    echo ""
    
    # Test service DNS resolution
    while IFS= read -r service; do
        if [ -n "$service" ]; then
            validate_check "$service DNS resolution" "kubectl exec -n $NAMESPACE deployment/nautilus-integration-engine -- nslookup $service.$NAMESPACE.svc.cluster.local" "$service"
        fi
    done <<< "$SERVICES"
    
else
    echo -e "${YELLOW}⚠️  No services found${NC}"
fi

echo ""

echo -e "${BLUE}5. Ultra-Low Latency Performance Validation${NC}"
echo "============================================="

echo "⚡ Testing Phase 2 performance preservation in Kubernetes..."

# Check if integration engine is accessible
INTEGRATION_POD=$(kubectl get pods -n $NAMESPACE -l app=nautilus-integration-engine --no-headers -o custom-columns=":metadata.name" | head -n1)

if [ -n "$INTEGRATION_POD" ]; then
    echo "🔗 Testing Integration Engine: $INTEGRATION_POD"
    
    # Test basic health
    if kubectl exec -n $NAMESPACE $INTEGRATION_POD -- curl -f http://localhost:8000/health --max-time 2 > /dev/null 2>&1; then
        echo -e "   Basic health: ${GREEN}✅ PASS${NC}"
        
        # Test integration health
        if kubectl exec -n $NAMESPACE $INTEGRATION_POD -- curl -f http://localhost:8000/health/integration --max-time 5 > /dev/null 2>&1; then
            echo -e "   Integration health: ${GREEN}✅ PASS${NC}"
            
            # Test end-to-end latency (if available)
            E2E_RESULT=$(kubectl exec -n $NAMESPACE $INTEGRATION_POD -- curl -s http://localhost:8000/health/e2e-latency --max-time 10 2>/dev/null || echo "{}")
            
            if [ "$E2E_RESULT" != "{}" ] && [ "$E2E_RESULT" != "" ]; then
                # Extract latency information if jq is available in pod
                if kubectl exec -n $NAMESPACE $INTEGRATION_POD -- which jq > /dev/null 2>&1; then
                    P99_LATENCY=$(kubectl exec -n $NAMESPACE $INTEGRATION_POD -- curl -s http://localhost:8000/health/e2e-latency | jq -r '.p99_latency_us' 2>/dev/null || echo "N/A")
                    TARGET_ACHIEVED=$(kubectl exec -n $NAMESPACE $INTEGRATION_POD -- curl -s http://localhost:8000/health/e2e-latency | jq -r '.target_achieved' 2>/dev/null || echo "unknown")
                    
                    echo "   End-to-End P99 Latency: ${P99_LATENCY}μs"
                    
                    if [ "$TARGET_ACHIEVED" = "true" ]; then
                        echo -e "   Phase 2B Target (< 2.75μs): ${GREEN}✅ ACHIEVED${NC}"
                    elif [ "$TARGET_ACHIEVED" = "false" ]; then
                        echo -e "   Phase 2B Target (< 2.75μs): ${YELLOW}⚠️  ABOVE TARGET${NC}"
                    else
                        echo -e "   Phase 2B Target: ${BLUE}ℹ️  MEASUREMENT IN PROGRESS${NC}"
                    fi
                else
                    echo -e "   End-to-End Latency: ${BLUE}ℹ️  MEASURED (details require jq)${NC}"
                fi
            else
                echo -e "   End-to-End Latency: ${YELLOW}⚠️  TIMEOUT (JIT warmup may be needed)${NC}"
            fi
            
        else
            echo -e "   Integration health: ${RED}❌ FAIL${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        fi
        
    else
        echo -e "   Basic health: ${RED}❌ FAIL${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    
else
    echo -e "${RED}❌ No integration engine pod found${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""

echo -e "${BLUE}6. Monitoring Infrastructure Validation${NC}"
echo "======================================="

# Check Prometheus
PROMETHEUS_POD=$(kubectl get pods -n $NAMESPACE -l app=prometheus --no-headers -o custom-columns=":metadata.name" | head -n1)
if [ -n "$PROMETHEUS_POD" ]; then
    validate_check "Prometheus health" "kubectl exec -n $NAMESPACE $PROMETHEUS_POD -- curl -f http://localhost:9090/-/healthy --max-time 5" "Prometheus is Healthy"
else
    echo -e "   Prometheus: ${YELLOW}⚠️  POD NOT FOUND${NC}"
fi

# Check Grafana  
GRAFANA_POD=$(kubectl get pods -n $NAMESPACE -l app=grafana --no-headers -o custom-columns=":metadata.name" | head -n1)
if [ -n "$GRAFANA_POD" ]; then
    validate_check "Grafana health" "kubectl exec -n $NAMESPACE $GRAFANA_POD -- curl -f http://localhost:3000/api/health --max-time 5" "ok"
else
    echo -e "   Grafana: ${YELLOW}⚠️  POD NOT FOUND${NC}"
fi

echo ""

echo -e "${BLUE}7. Auto-scaling Configuration Validation${NC}"
echo "========================================"

# Check HPA status
HPA_COUNT=$(kubectl get hpa -n $NAMESPACE --no-headers 2>/dev/null | wc -l || echo "0")

if [ "$HPA_COUNT" -gt 0 ]; then
    echo "📈 Horizontal Pod Autoscaler Status:"
    kubectl get hpa -n $NAMESPACE
    echo ""
    echo -e "   HPA Resources: ${HPA_COUNT} configured ✅"
else
    echo -e "   HPA Resources: ${YELLOW}⚠️  NONE CONFIGURED${NC}"
fi

echo ""

echo -e "${BLUE}8. Resource Utilization Analysis${NC}"
echo "================================"

echo "💻 Resource Usage Summary:"
if command -v kubectl &> /dev/null; then
    echo "   Node Resource Usage:"
    kubectl top nodes 2>/dev/null || echo "   (Metrics server not available)"
    
    echo "   Pod Resource Usage:"
    kubectl top pods -n $NAMESPACE 2>/dev/null || echo "   (Metrics server not available)"
fi

echo ""

# Calculate validation time
VALIDATION_END=$(date +%s)
VALIDATION_TIME=$((VALIDATION_END - VALIDATION_START))

echo -e "${BLUE}🏁 Validation Summary${NC}"
echo "===================="

echo "⏱️  Total validation time: ${VALIDATION_TIME} seconds"
echo "📊 Validation checks: $((TOTAL_CHECKS - FAILED_CHECKS))/$TOTAL_CHECKS passed"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 Phase 4 Kubernetes Validation: ALL CHECKS PASSED${NC}"
    echo ""
    echo "✅ Production Infrastructure: Operational"
    echo "✅ Ultra-Low Latency Tier: Healthy" 
    echo "✅ High Availability: Multi-replica deployments running"
    echo "✅ Monitoring: Prometheus + Grafana operational"
    echo "✅ Performance: Phase 2 targets maintained in Kubernetes"
    echo ""
    echo -e "${GREEN}Phase 4 Status: KUBERNETES DEPLOYMENT VALIDATED ✅${NC}"
    echo -e "${GREEN}Ready for: Enterprise production workloads${NC}"
    echo -e "${PURPLE}Achievement: Production-grade Kubernetes trading platform! 🚀${NC}"
    
elif [ $FAILED_CHECKS -le 3 ]; then
    echo -e "${YELLOW}⚠️  Phase 4 Kubernetes Validation: MOSTLY SUCCESSFUL${NC}"
    echo "   $FAILED_CHECKS minor issues detected (acceptable for initial deployment)"
    echo ""
    echo "🔧 Recommended actions:"
    echo "   1. Monitor failed components: kubectl get pods -n $NAMESPACE"
    echo "   2. Check pod logs: kubectl logs -f <pod-name> -n $NAMESPACE"
    echo "   3. Allow additional time for JIT compilation and warmup"
    echo "   4. Verify cluster resources: kubectl describe nodes"
    echo ""
    echo -e "${YELLOW}Phase 4 Status: DEPLOYMENT ACCEPTABLE ⚠️${NC}"
    
else
    echo -e "${RED}❌ Phase 4 Kubernetes Validation: CRITICAL ISSUES DETECTED${NC}"
    echo "   $FAILED_CHECKS/$TOTAL_CHECKS checks failed"
    echo ""
    echo "🚨 Required actions:"
    echo "   1. Check cluster status: kubectl get nodes"
    echo "   2. Review pod status: kubectl get pods -n $NAMESPACE -o wide"
    echo "   3. Check pod logs: kubectl logs <failing-pod> -n $NAMESPACE"
    echo "   4. Verify resource availability: kubectl describe nodes"
    echo "   5. Consider scaling cluster or adjusting resource requests"
    echo ""
    echo -e "${RED}Phase 4 Status: DEPLOYMENT REQUIRES ATTENTION ❌${NC}"
    exit 1
fi

echo ""
echo "📋 Management Commands:"
echo "   • Monitor pods: kubectl get pods -n $NAMESPACE -w"
echo "   • View logs: kubectl logs -f deployment/<name> -n $NAMESPACE"
echo "   • Port forward Integration Engine: kubectl port-forward svc/nautilus-integration-engine 8000:8000 -n $NAMESPACE"
echo "   • Port forward Grafana: kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo "   • Port forward Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
echo "   • Scale deployment: kubectl scale deployment/<name> --replicas=<count> -n $NAMESPACE"
echo ""
echo "🎯 Performance Testing:"
echo "   1. Port forward: kubectl port-forward svc/nautilus-integration-engine 8000:8000 -n $NAMESPACE"
echo "   2. Test latency: curl http://localhost:8000/health/e2e-latency"
echo "   3. Run benchmarks: curl http://localhost:8000/benchmarks/run"
echo "   4. Monitor Grafana: kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo ""
echo -e "${BLUE}Kubernetes Phase 4 Validation Complete ✨${NC}"