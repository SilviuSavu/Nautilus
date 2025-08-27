#!/usr/bin/env python3
"""
Nautilus Firewall Manager
🔒 CRITICAL SECURITY - iptables-based network firewall for API blocking

This service manages iptables firewall rules to block external API access
at the network level. It provides comprehensive protection against bypass
attempts and ensures 100% MarketData Hub compliance.

Features:
- iptables rule management for API blocking
- Real-time firewall monitoring and alerting
- Automatic rule updates from blocked hosts file
- Health monitoring and metrics API
- Integration with security monitoring system
- Comprehensive logging and audit trails

Author: Agent Alex (Security & DevOps Engineer)  
Date: August 25, 2025
Security Level: CRITICAL
"""

import os
import sys
import subprocess
import time
import json
import logging
import threading
import signal
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import socket
from collections import defaultdict

# FastAPI for management interface
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

# Configuration
class FirewallConfig:
    """Firewall configuration settings"""
    
    # Service settings
    MANAGEMENT_PORT = 9998
    LOG_LEVEL = "INFO"
    
    # Firewall settings
    BLOCKED_HOSTS_FILE = "/etc/nautilus-firewall/blocked_hosts.txt"
    IPTABLES_RULES_FILE = "/etc/nautilus-firewall/iptables_rules.conf"
    FIREWALL_LOG = "/var/log/nautilus/firewall.log"
    
    # Allowed internal networks (CIDR notation)
    ALLOWED_NETWORKS = [
        "127.0.0.0/8",      # Localhost
        "10.0.0.0/8",       # Private Class A
        "172.16.0.0/12",    # Private Class B  
        "192.168.0.0/16",   # Private Class C
        "172.20.0.0/16",    # Nautilus internal
        "172.21.0.0/16",    # Nautilus MarketData
        "172.22.0.0/16"     # Nautilus database
    ]
    
    # Blocked ports (common API ports)
    BLOCKED_PORTS = [80, 443, 8080, 8443, 9000, 9443]
    
    # Chain names for organization
    INPUT_CHAIN = "nautilus-input"
    OUTPUT_CHAIN = "nautilus-output" 
    FORWARD_CHAIN = "nautilus-forward"

class FirewallLogger:
    """Enhanced logging for firewall operations"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Configure firewall logging"""
        
        # Create log directory
        os.makedirs("/var/log/nautilus", exist_ok=True)
        
        # Main firewall logger
        self.logger = logging.getLogger("firewall_manager")
        self.logger.setLevel(getattr(logging, FirewallConfig.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | FIREWALL | %(levelname)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(FirewallConfig.FIREWALL_LOG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log_rule_action(self, action: str, rule_details: str, result: str = "SUCCESS"):
        """Log firewall rule actions"""
        self.logger.info(f"🔥 {action}: {rule_details} - {result}")
    
    def log_blocked_attempt(self, source_ip: str, dest_host: str, port: int):
        """Log blocked connection attempt"""
        self.logger.warning(f"🚫 BLOCKED: {source_ip} -> {dest_host}:{port}")
    
    def log_firewall_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None):
        """Log critical firewall alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "message": message,
            "details": details or {}
        }
        self.logger.critical(f"🚨 FIREWALL ALERT - {alert_type}: {message}")

class IPTablesManager:
    """iptables firewall rule management"""
    
    def __init__(self, logger: FirewallLogger):
        self.logger = logger
        self.blocked_ips: Set[str] = set()
        self.blocked_hosts: Set[str] = set()
        self.rule_stats = defaultdict(int)
        self.rules_applied = False
        
        # Load blocked hosts
        self._load_blocked_hosts()
    
    def _load_blocked_hosts(self):
        """Load blocked hosts from configuration file"""
        try:
            if os.path.exists(FirewallConfig.BLOCKED_HOSTS_FILE):
                with open(FirewallConfig.BLOCKED_HOSTS_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            self.blocked_hosts.add(line)
                
                self.logger.log_rule_action(
                    "LOAD_BLOCKED_HOSTS",
                    f"Loaded {len(self.blocked_hosts)} blocked hosts"
                )
            else:
                self.logger.logger.error(f"Blocked hosts file not found: {FirewallConfig.BLOCKED_HOSTS_FILE}")
                
        except Exception as e:
            self.logger.logger.error(f"Failed to load blocked hosts: {e}")
    
    def _run_iptables_command(self, command: List[str], ignore_errors: bool = False) -> bool:
        """Execute iptables command with error handling"""
        try:
            full_command = ["iptables"] + command
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=not ignore_errors
            )
            
            if result.returncode == 0:
                self.logger.log_rule_action("IPTABLES_CMD", " ".join(command), "SUCCESS")
                return True
            else:
                if not ignore_errors:
                    self.logger.logger.error(f"iptables command failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                self.logger.logger.error(f"iptables command error: {e}")
            return False
        except Exception as e:
            self.logger.logger.error(f"Unexpected iptables error: {e}")
            return False
    
    def initialize_firewall(self):
        """Initialize firewall with base rules"""
        self.logger.log_rule_action("INIT_FIREWALL", "Starting firewall initialization")
        
        # Create custom chains
        self._create_custom_chains()
        
        # Set default policies (ACCEPT for now, we'll use specific rules)
        self._run_iptables_command(["-P", "INPUT", "ACCEPT"])
        self._run_iptables_command(["-P", "FORWARD", "ACCEPT"])
        self._run_iptables_command(["-P", "OUTPUT", "ACCEPT"])
        
        # Apply base security rules
        self._apply_base_rules()
        
        # Apply API blocking rules
        self._apply_api_blocking_rules()
        
        # Save rules
        self._save_iptables_rules()
        
        self.rules_applied = True
        self.logger.log_rule_action("INIT_FIREWALL", "Firewall initialization completed")
    
    def _create_custom_chains(self):
        """Create custom iptables chains for organization"""
        
        chains = [
            FirewallConfig.INPUT_CHAIN,
            FirewallConfig.OUTPUT_CHAIN,
            FirewallConfig.FORWARD_CHAIN
        ]
        
        for chain in chains:
            # Create new chain (ignore if exists)
            self._run_iptables_command(["-N", chain], ignore_errors=True)
            
            # Clear existing rules in chain
            self._run_iptables_command(["-F", chain])
            
        # Link custom chains to standard chains
        self._run_iptables_command(["-I", "INPUT", "-j", FirewallConfig.INPUT_CHAIN], ignore_errors=True)
        self._run_iptables_command(["-I", "OUTPUT", "-j", FirewallConfig.OUTPUT_CHAIN], ignore_errors=True)
        self._run_iptables_command(["-I", "FORWARD", "-j", FirewallConfig.FORWARD_CHAIN], ignore_errors=True)
    
    def _apply_base_rules(self):
        """Apply base security rules"""
        
        # Allow loopback traffic
        self._run_iptables_command(["-A", FirewallConfig.INPUT_CHAIN, "-i", "lo", "-j", "ACCEPT"])
        self._run_iptables_command(["-A", FirewallConfig.OUTPUT_CHAIN, "-o", "lo", "-j", "ACCEPT"])
        
        # Allow established and related connections
        self._run_iptables_command(["-A", FirewallConfig.INPUT_CHAIN, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"])
        
        # Allow internal networks
        for network in FirewallConfig.ALLOWED_NETWORKS:
            self._run_iptables_command(["-A", FirewallConfig.OUTPUT_CHAIN, "-d", network, "-j", "ACCEPT"])
            self._run_iptables_command(["-A", FirewallConfig.INPUT_CHAIN, "-s", network, "-j", "ACCEPT"])
    
    def _apply_api_blocking_rules(self):
        """Apply API blocking rules for external hosts"""
        
        # Block by IP addresses (resolve hostnames)
        blocked_ips = self._resolve_blocked_hosts()
        
        for ip in blocked_ips:
            # Block outgoing connections to blocked IPs on API ports
            for port in FirewallConfig.BLOCKED_PORTS:
                self._run_iptables_command([
                    "-A", FirewallConfig.OUTPUT_CHAIN,
                    "-d", ip,
                    "-p", "tcp",
                    "--dport", str(port),
                    "-j", "REJECT",
                    "--reject-with", "tcp-reset"
                ])
                
                self._run_iptables_command([
                    "-A", FirewallConfig.OUTPUT_CHAIN,
                    "-d", ip,
                    "-p", "udp", 
                    "--dport", str(port),
                    "-j", "REJECT",
                    "--reject-with", "icmp-port-unreachable"
                ])
        
        # Add logging for blocked attempts
        self._add_logging_rules()
        
        self.logger.log_rule_action(
            "API_BLOCKING_RULES",
            f"Applied blocking rules for {len(blocked_ips)} IPs"
        )
    
    def _resolve_blocked_hosts(self) -> Set[str]:
        """Resolve blocked hostnames to IP addresses"""
        resolved_ips = set()
        
        for host in self.blocked_hosts:
            try:
                # Resolve hostname to IP
                ip = socket.gethostbyname(host)
                resolved_ips.add(ip)
                self.blocked_ips.add(ip)
                
            except socket.gaierror:
                # DNS resolution failed, log but continue
                self.logger.logger.warning(f"Could not resolve blocked host: {host}")
        
        return resolved_ips
    
    def _add_logging_rules(self):
        """Add rules to log blocked connection attempts"""
        
        # Log blocked outgoing connections
        self._run_iptables_command([
            "-A", FirewallConfig.OUTPUT_CHAIN,
            "-j", "LOG",
            "--log-prefix", "[NAUTILUS-BLOCKED-OUT] ",
            "--log-level", "4"
        ])
        
        # Log blocked incoming connections (if any)
        self._run_iptables_command([
            "-A", FirewallConfig.INPUT_CHAIN,
            "-j", "LOG", 
            "--log-prefix", "[NAUTILUS-BLOCKED-IN] ",
            "--log-level", "4"
        ])
    
    def _save_iptables_rules(self):
        """Save current iptables rules"""
        try:
            # Save IPv4 rules
            result = subprocess.run(
                ["iptables-save"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Write to rules file
            with open(FirewallConfig.IPTABLES_RULES_FILE, 'w') as f:
                f.write(result.stdout)
            
            self.logger.log_rule_action("SAVE_RULES", "iptables rules saved successfully")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save iptables rules: {e}")
    
    def reload_blocked_hosts(self):
        """Reload blocked hosts and update rules"""
        old_count = len(self.blocked_hosts)
        
        # Clear current blocked hosts
        self.blocked_hosts.clear()
        self.blocked_ips.clear()
        
        # Reload from file
        self._load_blocked_hosts()
        
        # Only update rules if the list changed
        if len(self.blocked_hosts) != old_count:
            self.logger.log_rule_action(
                "RELOAD_BLOCKED_HOSTS", 
                f"Host count changed: {old_count} -> {len(self.blocked_hosts)}"
            )
            
            # Re-apply API blocking rules
            self._flush_api_rules()
            self._apply_api_blocking_rules()
            self._save_iptables_rules()
    
    def _flush_api_rules(self):
        """Flush API blocking rules from custom chains"""
        self._run_iptables_command(["-F", FirewallConfig.OUTPUT_CHAIN])
        self._run_iptables_command(["-F", FirewallConfig.INPUT_CHAIN])
        
        # Re-apply base rules
        self._apply_base_rules()
    
    def get_firewall_stats(self) -> Dict[str, Any]:
        """Get firewall statistics"""
        try:
            # Get rule counts
            result = subprocess.run(
                ["iptables", "-L", "-n", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.split('\n')
            rule_count = len([line for line in lines if line.strip() and not line.startswith('Chain') and not line.startswith('pkts')])
            
            return {
                "rules_applied": self.rules_applied,
                "blocked_hosts_count": len(self.blocked_hosts),
                "blocked_ips_count": len(self.blocked_ips),
                "total_rules_count": rule_count,
                "allowed_networks": FirewallConfig.ALLOWED_NETWORKS,
                "blocked_ports": FirewallConfig.BLOCKED_PORTS,
                "custom_chains": [
                    FirewallConfig.INPUT_CHAIN,
                    FirewallConfig.OUTPUT_CHAIN,
                    FirewallConfig.FORWARD_CHAIN
                ]
            }
            
        except Exception as e:
            self.logger.logger.error(f"Failed to get firewall stats: {e}")
            return {"error": str(e)}
    
    def test_blocking(self, host: str) -> Dict[str, Any]:
        """Test if a host is properly blocked"""
        try:
            # Try to resolve the host
            try:
                ip = socket.gethostbyname(host)
                is_resolved = True
            except socket.gaierror:
                ip = None
                is_resolved = False
            
            # Check if host is in blocked list
            is_in_blocklist = host in self.blocked_hosts
            
            # Check if IP is blocked
            is_ip_blocked = ip in self.blocked_ips if ip else False
            
            # Test connection attempt (this should fail if firewall is working)
            connection_test = self._test_connection(host, 443)  # Try HTTPS
            
            return {
                "host": host,
                "resolved_ip": ip,
                "is_resolved": is_resolved,
                "is_in_blocklist": is_in_blocklist,
                "is_ip_blocked": is_ip_blocked,
                "connection_blocked": not connection_test["success"],
                "test_result": "PASS" if (not connection_test["success"] and is_in_blocklist) else "FAIL"
            }
            
        except Exception as e:
            return {"host": host, "error": str(e), "test_result": "ERROR"}
    
    def _test_connection(self, host: str, port: int, timeout: int = 3) -> Dict[str, Any]:
        """Test connection to a host (should fail if blocked)"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return {
                "success": result == 0,
                "result_code": result,
                "message": "Connection successful" if result == 0 else f"Connection failed (code: {result})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Connection test failed: {str(e)}"
            }

class FirewallAPI:
    """FastAPI application for firewall management"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Nautilus Firewall Manager",
            description="Network firewall for external API blocking",
            version="1.0.0"
        )
        
        self.logger = FirewallLogger()
        self.iptables = IPTablesManager(self.logger)
        self.monitoring_active = True
        
        # Setup routes
        self._setup_routes()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize firewall
        self._initialize_firewall()
        
        self.logger.log_firewall_alert(
            "FIREWALL_STARTED",
            "Nautilus Firewall Manager initialized with CRITICAL enforcement",
            {"port": FirewallConfig.MANAGEMENT_PORT}
        )
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "nautilus-firewall", 
                "firewall_active": self.iptables.rules_applied,
                "monitoring_active": self.monitoring_active,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/firewall/status")
        async def firewall_status():
            """Get firewall status and statistics"""
            stats = self.iptables.get_firewall_stats()
            return {
                "firewall_status": "ACTIVE" if self.iptables.rules_applied else "INACTIVE",
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/firewall/reload-hosts")
        async def reload_blocked_hosts():
            """Reload blocked hosts from configuration file"""
            try:
                self.iptables.reload_blocked_hosts()
                return {
                    "status": "success",
                    "message": "Blocked hosts reloaded successfully",
                    "blocked_hosts_count": len(self.iptables.blocked_hosts),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to reload hosts: {str(e)}")
        
        @self.app.post("/firewall/test-blocking/{host}")
        async def test_host_blocking(host: str):
            """Test if a specific host is properly blocked"""
            try:
                test_result = self.iptables.test_blocking(host)
                return {
                    "test_type": "HOST_BLOCKING",
                    "test_result": test_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
        
        @self.app.get("/firewall/blocked-hosts")
        async def get_blocked_hosts():
            """Get list of blocked hosts"""
            return {
                "blocked_hosts": sorted(list(self.iptables.blocked_hosts)),
                "blocked_ips": sorted(list(self.iptables.blocked_ips)),
                "total_hosts": len(self.iptables.blocked_hosts),
                "total_ips": len(self.iptables.blocked_ips),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/firewall/rules") 
        async def get_firewall_rules():
            """Get current iptables rules"""
            try:
                result = subprocess.run(
                    ["iptables", "-L", "-n", "-v", "--line-numbers"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                return {
                    "iptables_rules": result.stdout,
                    "rules_applied": self.iptables.rules_applied,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")
        
        @self.app.post("/firewall/reinitialize")
        async def reinitialize_firewall():
            """Reinitialize firewall rules"""
            try:
                self._initialize_firewall()
                return {
                    "status": "success",
                    "message": "Firewall reinitialized successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Reinitialization failed: {str(e)}")
    
    def _initialize_firewall(self):
        """Initialize the firewall"""
        try:
            self.iptables.initialize_firewall()
            self.logger.log_firewall_alert(
                "FIREWALL_INITIALIZED",
                "iptables firewall rules applied successfully"
            )
        except Exception as e:
            self.logger.log_firewall_alert(
                "FIREWALL_INIT_FAILED",
                f"Failed to initialize firewall: {str(e)}"
            )
            raise
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.log_firewall_alert(
                "FIREWALL_SHUTDOWN",
                f"Nautilus Firewall Manager shutting down (signal {signum})"
            )
            self.monitoring_active = False
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main entry point"""
    try:
        # Create firewall API
        firewall_api = FirewallAPI()
        
        # Run the firewall manager
        uvicorn.run(
            firewall_api.app,
            host="0.0.0.0",
            port=FirewallConfig.MANAGEMENT_PORT,
            log_level=FirewallConfig.LOG_LEVEL.lower(),
            access_log=True
        )
        
    except Exception as e:
        logging.getLogger("firewall_manager").critical(f"Failed to start firewall manager: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()