#!/bin/bash
set -e

# DNS Failover Script for Nautilus Multi-Cloud Federation
# Updates DNS records to point to healthy clusters

ZONE_ID="${ROUTE53_ZONE_ID:-Z1234567890}"
DOMAIN="${TRADING_DOMAIN:-api.nautilus.trading}"
TTL=60

# Function to update Route53 record
update_dns_record() {
    local target_ip="$1"
    local record_type="${2:-A}"
    
    echo "Updating DNS record ${DOMAIN} to point to ${target_ip}"
    
    # Create change batch JSON
    cat > /tmp/dns_change.json <<EOF
{
    "Comment": "Automated failover - $(date)",
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "${DOMAIN}",
            "Type": "${record_type}",
            "TTL": ${TTL},
            "ResourceRecords": [{
                "Value": "${target_ip}"
            }]
        }
    }]
}
EOF
    
    # Apply DNS change
    aws route53 change-resource-record-sets         --hosted-zone-id "${ZONE_ID}"         --change-batch file:///tmp/dns_change.json
    
    # Wait for change to propagate
    echo "Waiting for DNS propagation..."
    sleep 30
    
    # Verify change
    dig +short "${DOMAIN}" | head -1
    echo "DNS failover completed to ${target_ip}"
}

# Function to get cluster IP
get_cluster_ip() {
    local cluster_name="$1"
    
    case "$cluster_name" in
        "nautilus-primary-us-east")
            echo "52.86.123.45"
            ;;
        "nautilus-primary-eu-west") 
            echo "34.76.89.123"
            ;;
        "nautilus-primary-asia-northeast")
            echo "20.48.156.78"
            ;;
        "nautilus-dr-us-west")
            echo "54.183.45.67"
            ;;
        *)
            echo "Unknown cluster: $cluster_name" >&2
            exit 1
            ;;
    esac
}

# Main execution
if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_cluster>"
    exit 1
fi

TARGET_CLUSTER="$1"
TARGET_IP=$(get_cluster_ip "$TARGET_CLUSTER")

if [ -z "$TARGET_IP" ]; then
    echo "Could not determine IP for cluster: $TARGET_CLUSTER"
    exit 1
fi

update_dns_record "$TARGET_IP"
