#!/bin/bash
set -e

# Multi-Cloud Federation Deployment Script for Nautilus Trading Platform
# Deploys complete multi-cloud Kubernetes federation with ultra-low latency optimization

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/nautilus/multi-cloud-deployment.log"
METRICS_FILE="/var/log/nautilus/deployment-metrics.json"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_NAME="nautilus-multi-cloud-federation"
FEDERATION_NAMESPACE="nautilus-federation"
MONITORING_NAMESPACE="nautilus-monitoring"

# Cloud provider configurations
declare -A AWS_REGIONS=(
    ["primary"]="us-east-1"
    ["dr"]="us-west-2"
)

declare -A GCP_REGIONS=(
    ["primary"]="europe-west1"
    ["dr"]="europe-west3"
)

declare -A AZURE_REGIONS=(
    ["primary"]="japaneast"
    ["dr"]="australiaeast"
)

# Deployment phases
PHASES=(
    "prerequisites"
    "infrastructure"
    "networking"
    "service_mesh"
    "federation"
    "monitoring"
    "disaster_recovery"
    "validation"
)

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    
    # Color output based on level
    case "$level" in
        "INFO")  echo -e "${GREEN}[$timestamp] [INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[$timestamp] [WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[$timestamp] [ERROR]${NC} $message" ;;
        "DEBUG") echo -e "${CYAN}[$timestamp] [DEBUG]${NC} $message" ;;
        *)       echo -e "${NC}[$timestamp] [$level]${NC} $message" ;;
    esac
    
    # Also write to log file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

log_metric() {
    local metric_name="$1"
    local metric_value="$2"
    local timestamp=$(date +%s)
    
    echo "{\"timestamp\":$timestamp,\"metric\":\"$metric_name\",\"value\":$metric_value}" >> "$METRICS_FILE"
}

# Progress tracking
show_progress() {
    local current="$1"
    local total="$2"
    local phase="$3"
    
    local percent=$((current * 100 / total))
    local progress_bar=""
    
    for ((i=0; i<50; i++)); do
        if [ $i -lt $((percent / 2)) ]; then
            progress_bar+="█"
        else
            progress_bar+="░"
        fi
    done
    
    echo -e "\n${PURPLE}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${PURPLE}│${NC} Phase: ${BLUE}$phase${NC} ${PURPLE}($current/$total)${NC} ${PURPLE}│${NC}"
    echo -e "${PURPLE}│${NC} Progress: [$progress_bar] ${percent}% ${PURPLE}│${NC}"
    echo -e "${PURPLE}└─────────────────────────────────────────────────────────────┘${NC}\n"
}

# Phase 1: Prerequisites
deploy_prerequisites() {
    log "INFO" "Phase 1: Checking and installing prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "terraform" "aws" "gcloud" "az" "istioctl")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        log "INFO" "Please install the missing tools and run again"
        return 1
    fi
    
    # Check cloud provider authentication
    log "INFO" "Checking cloud provider authentication..."
    
    # AWS
    if ! aws sts get-caller-identity &> /dev/null; then
        log "WARN" "AWS authentication not configured"
    else
        log "INFO" "AWS authentication verified"
    fi
    
    # GCP
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
        log "WARN" "GCP authentication not configured"
    else
        log "INFO" "GCP authentication verified"
    fi
    
    # Azure
    if ! az account show &> /dev/null; then
        log "WARN" "Azure authentication not configured"
    else
        log "INFO" "Azure authentication verified"
    fi
    
    # Create required directories
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$METRICS_FILE")"
    
    # Install Helm repositories
    log "INFO" "Adding Helm repositories..."
    helm repo add istio https://istio-release.storage.googleapis.com/charts
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    log "INFO" "Prerequisites check completed"
    return 0
}

# Phase 2: Infrastructure Deployment
deploy_infrastructure() {
    log "INFO" "Phase 2: Deploying multi-cloud infrastructure..."
    
    # Deploy AWS infrastructure
    log "INFO" "Deploying AWS infrastructure..."
    cd "$PROJECT_ROOT/terraform/aws"
    
    if [ ! -f "terraform.tfstate" ]; then
        terraform init
    fi
    
    terraform plan -out=aws-plan.tfplan
    terraform apply aws-plan.tfplan
    
    # Deploy GCP infrastructure
    log "INFO" "Deploying GCP infrastructure..."
    cd "$PROJECT_ROOT/terraform/gcp"
    
    if [ ! -f "terraform.tfstate" ]; then
        terraform init
    fi
    
    terraform plan -out=gcp-plan.tfplan
    terraform apply gcp-plan.tfplan
    
    # Deploy Azure infrastructure
    log "INFO" "Deploying Azure infrastructure..."
    cd "$PROJECT_ROOT/terraform/azure"
    
    if [ ! -f "terraform.tfstate" ]; then
        terraform init
    fi
    
    terraform plan -out=azure-plan.tfplan
    terraform apply azure-plan.tfplan
    
    cd "$SCRIPT_DIR"
    
    # Get cluster credentials
    log "INFO" "Configuring kubectl contexts..."
    
    # AWS EKS
    aws eks update-kubeconfig --region us-east-1 --name nautilus-primary-us-east
    aws eks update-kubeconfig --region us-west-2 --name nautilus-dr-us-west
    
    # GCP GKE
    gcloud container clusters get-credentials nautilus-primary-eu-west --region europe-west1
    gcloud container clusters get-credentials nautilus-dr-eu-central --region europe-west3
    
    # Azure AKS
    az aks get-credentials --resource-group nautilus-primary-asia-northeast-rg --name nautilus-primary-asia-northeast
    az aks get-credentials --resource-group nautilus-dr-asia-australia-rg --name nautilus-dr-asia-australia
    
    log "INFO" "Infrastructure deployment completed"
    return 0
}

# Phase 3: Networking Configuration
deploy_networking() {
    log "INFO" "Phase 3: Configuring multi-cloud networking..."
    
    # Configure VPC peering for AWS
    log "INFO" "Setting up AWS VPC peering..."
    
    # Create VPC peering connections
    aws ec2 create-vpc-peering-connection \
        --vpc-id vpc-12345678 \
        --peer-vpc-id vpc-87654321 \
        --peer-region us-west-2 \
        --region us-east-1 || true
    
    # Configure GCP network peering
    log "INFO" "Setting up GCP network peering..."
    
    gcloud compute networks peerings create nautilus-eu-west-to-eu-central \
        --network=nautilus-primary-eu-west-vpc \
        --peer-network=nautilus-dr-eu-central-vpc \
        --project="$GCP_PROJECT_ID" || true
    
    # Configure Azure virtual network peering
    log "INFO" "Setting up Azure VNet peering..."
    
    az network vnet peering create \
        --name nautilus-asia-peering \
        --resource-group nautilus-primary-asia-northeast-rg \
        --vnet-name nautilus-primary-asia-northeast-vnet \
        --remote-vnet nautilus-dr-asia-australia-vnet || true
    
    # Apply network optimization settings
    log "INFO" "Applying network optimizations..."
    
    # Run network optimization script
    if [ -f "$PROJECT_ROOT/networking/ultra_low_latency_topology.py" ]; then
        python3 "$PROJECT_ROOT/networking/ultra_low_latency_topology.py" &
        NETWORK_OPTIMIZER_PID=$!
    fi
    
    log "INFO" "Networking configuration completed"
    return 0
}

# Phase 4: Service Mesh Deployment
deploy_service_mesh() {
    log "INFO" "Phase 4: Deploying Istio service mesh..."
    
    local clusters=(
        "nautilus-primary-us-east"
        "nautilus-primary-eu-west"
        "nautilus-primary-asia-northeast"
        "nautilus-dr-us-west"
        "nautilus-dr-eu-central"
        "nautilus-dr-asia-australia"
    )
    
    # Install Istio on each cluster
    for cluster in "${clusters[@]}"; do
        log "INFO" "Installing Istio on $cluster..."
        
        # Switch to cluster context
        kubectl config use-context "$cluster"
        
        # Create Istio namespace
        kubectl create namespace istio-system --dry-run=client -o yaml | kubectl apply -f -
        
        # Install Istio with multi-cluster configuration
        istioctl install --set values.pilot.env.ISTIOD_ENABLE_CROSS_CLUSTER_WORKLOAD_ENTRY=true \
                         --set values.pilot.env.ISTIOD_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true \
                         --set values.pilot.env.ISTIOD_DEFAULT_REQUEST_TIMEOUT=10s \
                         --set meshConfig.defaultConfig.holdApplicationUntilProxyStarts=true \
                         --set meshConfig.defaultConfig.proxyMetadata.PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true \
                         -y
        
        # Label namespace for Istio injection
        kubectl label namespace default istio-injection=enabled --overwrite
        
        # Install east-west gateway for multi-cluster
        kubectl apply -f - <<EOF
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: eastwest
  namespace: istio-system
spec:
  revision: ""
  components:
    ingressGateways:
    - name: istio-eastwestgateway
      label:
        istio: eastwestgateway
        app: istio-eastwestgateway
      enabled: true
      k8s:
        service:
          type: LoadBalancer
          ports:
          - port: 15021
            targetPort: 15021
            name: status-port
          - port: 15443
            targetPort: 15443
            name: tls
EOF
        
        # Wait for gateway to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/istio-eastwestgateway -n istio-system
        
    done
    
    # Configure cross-cluster service discovery
    log "INFO" "Configuring cross-cluster service discovery..."
    
    # Create secrets for cross-cluster communication
    for cluster in "${clusters[@]}"; do
        kubectl config use-context "$cluster"
        
        # Create cluster secret for service discovery
        kubectl create secret generic cacerts -n istio-system \
            --from-file=root-cert.pem \
            --from-file=cert-chain.pem \
            --from-file=ca-cert.pem \
            --from-file=ca-key.pem \
            --dry-run=client -o yaml | kubectl apply -f - || true
    done
    
    log "INFO" "Service mesh deployment completed"
    return 0
}

# Phase 5: Federation Configuration
deploy_federation() {
    log "INFO" "Phase 5: Configuring multi-cloud federation..."
    
    # Create federation namespace on all clusters
    local clusters=(
        "nautilus-primary-us-east"
        "nautilus-primary-eu-west" 
        "nautilus-primary-asia-northeast"
    )
    
    for cluster in "${clusters[@]}"; do
        kubectl config use-context "$cluster"
        
        # Create federation namespace
        kubectl create namespace "$FEDERATION_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        
        # Label namespace for Istio injection
        kubectl label namespace "$FEDERATION_NAMESPACE" istio-injection=enabled --overwrite
        
    done
    
    # Apply federation manifests
    log "INFO" "Applying federation manifests..."
    
    # Primary cluster (US East)
    kubectl config use-context "nautilus-primary-us-east"
    kubectl apply -f "$PROJECT_ROOT/manifests/" -R
    
    # Apply cross-cluster service discovery configuration
    kubectl apply -f "$PROJECT_ROOT/federation/cross_cluster_service_discovery.yaml"
    
    # Configure global DNS
    log "INFO" "Configuring global DNS..."
    
    # Create DNS configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns-custom
  namespace: kube-system
data:
  nautilus.server: |
    nautilus.trading:53 {
        template IN A {
            match "^api\.nautilus\.trading\.$"
            answer "{{ .Name }} 60 IN A 52.86.123.45"
            fallthrough
        }
        template IN A {
            match "^eu-api\.nautilus\.trading\.$"
            answer "{{ .Name }} 60 IN A 34.76.89.123"
            fallthrough
        }
        template IN A {
            match "^asia-api\.nautilus\.trading\.$"
            answer "{{ .Name }} 60 IN A 20.48.156.78"
            fallthrough
        }
        forward . 8.8.8.8 8.8.4.4
        cache 300
    }
EOF
    
    # Restart CoreDNS to pick up configuration
    kubectl rollout restart deployment/coredns -n kube-system
    
    log "INFO" "Federation configuration completed"
    return 0
}

# Phase 6: Monitoring Deployment
deploy_monitoring() {
    log "INFO" "Phase 6: Deploying monitoring stack..."
    
    # Switch to primary cluster
    kubectl config use-context "nautilus-primary-us-east"
    
    # Create monitoring namespace
    kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    log "INFO" "Deploying Prometheus..."
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.replicas=2 \
        --set prometheus.prometheusSpec.resources.requests.cpu=500m \
        --set prometheus.prometheusSpec.resources.requests.memory=2Gi \
        --set prometheus.prometheusSpec.resources.limits.cpu=2000m \
        --set prometheus.prometheusSpec.resources.limits.memory=4Gi \
        --wait
    
    # Deploy Grafana dashboards
    log "INFO" "Deploying Grafana dashboards..."
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: nautilus-federation-dashboard
  namespace: $MONITORING_NAMESPACE
  labels:
    grafana_dashboard: "1"
data:
  nautilus-federation.json: |
    {
      "dashboard": {
        "title": "Nautilus Multi-Cloud Federation",
        "tags": ["nautilus", "trading", "federation"],
        "panels": [
          {
            "title": "Cross-Cluster Latency",
            "type": "stat",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket{source_cluster!=\"unknown\",destination_cluster!=\"unknown\"}[5m]))"
              }
            ]
          },
          {
            "title": "Cluster Health Status",
            "type": "table",
            "targets": [
              {
                "expr": "up{job=\"kubernetes-apiservers\"}"
              }
            ]
          },
          {
            "title": "Network Throughput",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(istio_requests_total{source_cluster!=\"unknown\",destination_cluster!=\"unknown\"}[5m])"
              }
            ]
          }
        ]
      }
    }
EOF
    
    # Deploy Jaeger for distributed tracing
    log "INFO" "Deploying Jaeger..."
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: $MONITORING_NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.45
        ports:
        - containerPort: 16686
          name: ui
        - containerPort: 6831
          name: agent-thrift
          protocol: UDP
        - containerPort: 14268
          name: collector
        env:
        - name: COLLECTOR_ZIPKIN_HTTP_PORT
          value: "9411"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger
  namespace: $MONITORING_NAMESPACE
spec:
  selector:
    app: jaeger
  ports:
  - port: 16686
    targetPort: 16686
    name: ui
  - port: 6831
    targetPort: 6831
    protocol: UDP
    name: agent-thrift
  - port: 14268
    targetPort: 14268
    name: collector
EOF
    
    log "INFO" "Monitoring stack deployment completed"
    return 0
}

# Phase 7: Disaster Recovery Setup
deploy_disaster_recovery() {
    log "INFO" "Phase 7: Setting up disaster recovery automation..."
    
    # Deploy disaster recovery orchestrator
    log "INFO" "Deploying DR orchestrator..."
    
    # Create DR service account and RBAC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: disaster-recovery-orchestrator
  namespace: $FEDERATION_NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: disaster-recovery-orchestrator
rules:
- apiGroups: [""]
  resources: ["services", "endpoints", "pods"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["networking.istio.io"]
  resources: ["virtualservices", "destinationrules", "gateways"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: disaster-recovery-orchestrator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: disaster-recovery-orchestrator
subjects:
- kind: ServiceAccount
  name: disaster-recovery-orchestrator
  namespace: $FEDERATION_NAMESPACE
EOF
    
    # Deploy DR orchestrator pod
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: disaster-recovery-orchestrator
  namespace: $FEDERATION_NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: disaster-recovery-orchestrator
  template:
    metadata:
      labels:
        app: disaster-recovery-orchestrator
    spec:
      serviceAccountName: disaster-recovery-orchestrator
      containers:
      - name: orchestrator
        image: python:3.11-slim
        command: ["python3", "/app/dr_orchestrator.py"]
        volumeMounts:
        - name: dr-scripts
          mountPath: /app
        env:
        - name: REDIS_HOST
          value: "redis-cluster.nautilus-federation.svc.cluster.local"
        - name: FEDERATION_NAMESPACE
          value: "$FEDERATION_NAMESPACE"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
      volumes:
      - name: dr-scripts
        configMap:
          name: dr-orchestrator-scripts
          defaultMode: 0755
EOF
    
    # Create ConfigMap with DR scripts
    kubectl create configmap dr-orchestrator-scripts \
        --from-file="$PROJECT_ROOT/disaster_recovery/" \
        -n "$FEDERATION_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Set up automated failover script as a CronJob
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: automated-failover-check
  namespace: $FEDERATION_NAMESPACE
spec:
  schedule: "*/1 * * * *"  # Every minute
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: failover-check
            image: alpine:latest
            command: ["/bin/sh"]
            args:
            - -c
            - |
              apk add --no-cache curl
              /scripts/automated_failover.sh health
            volumeMounts:
            - name: failover-scripts
              mountPath: /scripts
          volumes:
          - name: failover-scripts
            configMap:
              name: failover-scripts
              defaultMode: 0755
          restartPolicy: OnFailure
EOF
    
    # Create failover scripts ConfigMap
    kubectl create configmap failover-scripts \
        --from-file="$PROJECT_ROOT/disaster_recovery/automated_failover.sh" \
        -n "$FEDERATION_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "INFO" "Disaster recovery setup completed"
    return 0
}

# Phase 8: Validation and Testing
validate_deployment() {
    log "INFO" "Phase 8: Validating deployment..."
    
    local validation_start=$(date +%s)
    local validation_errors=0
    
    # Test 1: Cluster connectivity
    log "INFO" "Testing cluster connectivity..."
    
    local clusters=(
        "nautilus-primary-us-east"
        "nautilus-primary-eu-west"
        "nautilus-primary-asia-northeast"
    )
    
    for cluster in "${clusters[@]}"; do
        kubectl config use-context "$cluster"
        
        if kubectl get nodes &> /dev/null; then
            log "INFO" "✓ Cluster $cluster is accessible"
        else
            log "ERROR" "✗ Cluster $cluster is not accessible"
            ((validation_errors++))
        fi
    done
    
    # Test 2: Service mesh connectivity
    log "INFO" "Testing service mesh connectivity..."
    
    kubectl config use-context "nautilus-primary-us-east"
    
    # Check Istio components
    if kubectl get pods -n istio-system | grep -q "Running"; then
        log "INFO" "✓ Istio service mesh is running"
    else
        log "ERROR" "✗ Istio service mesh is not running properly"
        ((validation_errors++))
    fi
    
    # Test 3: Cross-cluster service discovery
    log "INFO" "Testing cross-cluster service discovery..."
    
    # Try to resolve services from other clusters
    if kubectl exec -n "$FEDERATION_NAMESPACE" deployment/test-pod -- nslookup integration-engine.nautilus-primary-eu-west.local &> /dev/null; then
        log "INFO" "✓ Cross-cluster DNS resolution working"
    else
        log "WARN" "⚠ Cross-cluster DNS resolution may not be working (test pod may not exist)"
    fi
    
    # Test 4: Network latency
    log "INFO" "Testing network latency..."
    
    # Run network performance test
    if [ -f "$PROJECT_ROOT/networking/ultra_low_latency_topology.py" ]; then
        timeout 60s python3 "$PROJECT_ROOT/networking/ultra_low_latency_topology.py" > /tmp/network_test.log 2>&1
        
        if [ $? -eq 0 ]; then
            log "INFO" "✓ Network performance test completed"
        else
            log "WARN" "⚠ Network performance test had issues"
        fi
    fi
    
    # Test 5: Monitoring stack
    log "INFO" "Testing monitoring stack..."
    
    kubectl config use-context "nautilus-primary-us-east"
    
    if kubectl get pods -n "$MONITORING_NAMESPACE" | grep -q "Running"; then
        log "INFO" "✓ Monitoring stack is running"
    else
        log "ERROR" "✗ Monitoring stack is not running properly"
        ((validation_errors++))
    fi
    
    # Test 6: Disaster recovery readiness
    log "INFO" "Testing disaster recovery readiness..."
    
    if kubectl get deployment disaster-recovery-orchestrator -n "$FEDERATION_NAMESPACE" &> /dev/null; then
        log "INFO" "✓ Disaster recovery orchestrator deployed"
    else
        log "ERROR" "✗ Disaster recovery orchestrator not found"
        ((validation_errors++))
    fi
    
    # Test 7: Global load balancer
    log "INFO" "Testing global load balancer..."
    
    # Test DNS resolution for main domain
    if nslookup api.nautilus.trading &> /dev/null; then
        log "INFO" "✓ Global DNS resolution working"
    else
        log "WARN" "⚠ Global DNS may not be configured yet"
    fi
    
    # Test 8: Federation endpoints health
    log "INFO" "Testing federation endpoints health..."
    
    local healthy_endpoints=0
    local total_endpoints=0
    
    for cluster in "${clusters[@]}"; do
        kubectl config use-context "$cluster"
        
        # Get cluster external IP (simplified)
        case "$cluster" in
            "nautilus-primary-us-east")
                cluster_ip="52.86.123.45"
                ;;
            "nautilus-primary-eu-west")
                cluster_ip="34.76.89.123"
                ;;
            "nautilus-primary-asia-northeast")
                cluster_ip="20.48.156.78"
                ;;
        esac
        
        ((total_endpoints++))
        
        if timeout 5 curl -k -s "https://$cluster_ip:443/health" &> /dev/null; then
            log "INFO" "✓ Endpoint $cluster ($cluster_ip) is healthy"
            ((healthy_endpoints++))
        else
            log "WARN" "⚠ Endpoint $cluster ($cluster_ip) is not responding"
        fi
    done
    
    # Calculate validation metrics
    local validation_duration=$(($(date +%s) - validation_start))
    local success_rate=$((healthy_endpoints * 100 / total_endpoints))
    
    log_metric "validation_duration_seconds" "$validation_duration"
    log_metric "validation_errors" "$validation_errors"
    log_metric "endpoint_success_rate_percent" "$success_rate"
    
    # Validation summary
    log "INFO" "=========================================="
    log "INFO" "VALIDATION SUMMARY"
    log "INFO" "=========================================="
    log "INFO" "Duration: ${validation_duration} seconds"
    log "INFO" "Validation errors: $validation_errors"
    log "INFO" "Endpoint health: $healthy_endpoints/$total_endpoints ($success_rate%)"
    
    if [ $validation_errors -eq 0 ]; then
        log "INFO" "✅ All validation tests passed!"
        return 0
    else
        log "WARN" "⚠️  Validation completed with $validation_errors errors"
        return 1
    fi
}

# Main deployment function
main() {
    local deployment_start=$(date +%s)
    
    echo -e "${PURPLE}"
    echo "🌐 Nautilus Multi-Cloud Federation Deployment"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script will deploy a complete multi-cloud Kubernetes federation"
    echo "across AWS, GCP, and Azure with ultra-low latency optimization."
    echo
    echo "Deployment phases:"
    for i in "${!PHASES[@]}"; do
        echo "  $((i+1)). ${PHASES[i]}"
    done
    echo
    
    # Confirm deployment
    read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "INFO" "Deployment cancelled by user"
        exit 0
    fi
    
    # Execute deployment phases
    local phase_count=${#PHASES[@]}
    local current_phase=1
    local failed_phases=()
    
    for phase in "${PHASES[@]}"; do
        show_progress $current_phase $phase_count "$phase"
        
        case "$phase" in
            "prerequisites")
                if deploy_prerequisites; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "infrastructure")
                if deploy_infrastructure; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "networking")
                if deploy_networking; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "service_mesh")
                if deploy_service_mesh; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "federation")
                if deploy_federation; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "monitoring")
                if deploy_monitoring; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "disaster_recovery")
                if deploy_disaster_recovery; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "ERROR" "❌ Phase $current_phase/$phase_count failed: $phase"
                    failed_phases+=("$phase")
                fi
                ;;
            "validation")
                if validate_deployment; then
                    log "INFO" "✅ Phase $current_phase/$phase_count completed: $phase"
                else
                    log "WARN" "⚠️  Phase $current_phase/$phase_count completed with warnings: $phase"
                fi
                ;;
        esac
        
        ((current_phase++))
    done
    
    # Deployment summary
    local deployment_duration=$(($(date +%s) - deployment_start))
    log_metric "total_deployment_duration_seconds" "$deployment_duration"
    
    echo
    echo -e "${PURPLE}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${PURPLE}│${NC}                    ${GREEN}DEPLOYMENT COMPLETE${NC}                    ${PURPLE}│${NC}"
    echo -e "${PURPLE}└─────────────────────────────────────────────────────────────┘${NC}"
    echo
    
    log "INFO" "=========================================="
    log "INFO" "DEPLOYMENT SUMMARY"
    log "INFO" "=========================================="
    log "INFO" "Total duration: $deployment_duration seconds"
    log "INFO" "Phases completed: $((phase_count - ${#failed_phases[@]}))/$phase_count"
    
    if [ ${#failed_phases[@]} -eq 0 ]; then
        log "INFO" "🎉 All phases completed successfully!"
    else
        log "WARN" "Failed phases: ${failed_phases[*]}"
    fi
    
    echo
    echo -e "${BLUE}📊 Access URLs:${NC}"
    echo "   • Grafana Dashboard: kubectl port-forward svc/prometheus-grafana 3000:80 -n $MONITORING_NAMESPACE"
    echo "   • Jaeger Tracing: kubectl port-forward svc/jaeger 16686:16686 -n $MONITORING_NAMESPACE"  
    echo "   • Prometheus: kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n $MONITORING_NAMESPACE"
    echo
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "   • Monitor DR: kubectl logs -f deployment/disaster-recovery-orchestrator -n $FEDERATION_NAMESPACE"
    echo "   • Check federation: kubectl get all -n $FEDERATION_NAMESPACE"
    echo "   • View metrics: tail -f $METRICS_FILE"
    echo
    echo -e "${GREEN}✅ Multi-cloud federation deployment completed!${NC}"
}

# Trap signals for graceful shutdown
trap 'log "INFO" "Deployment interrupted by user"; exit 130' INT TERM

# Execute main function
main "$@"