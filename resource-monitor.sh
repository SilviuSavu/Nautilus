#!/bin/bash
while true; do
    # Memory usage alert
    MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" | sed 's/%//' | sort -nr | head -1)
    if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
        echo "[$(date +'%H:%M:%S')] WARNING: High memory usage detected: ${MEMORY_USAGE}%"
    fi
    
    # CPU usage alert
    CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | sort -nr | head -1)
    if (( $(echo "$CPU_USAGE > 85" | bc -l) )); then
        echo "[$(date +'%H:%M:%S')] WARNING: High CPU usage detected: ${CPU_USAGE}%"
    fi
    
    sleep 60
done >> resource-alerts.log 2>&1
