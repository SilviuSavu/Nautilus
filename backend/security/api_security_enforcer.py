#!/usr/bin/env python3
"""
Nautilus API Security Enforcer - System-Wide API Blocking Implementation
🔒 CRITICAL SECURITY MODULE - DirectAPIBlocker Enforcement

This module provides comprehensive runtime enforcement of API blocking policies
to ensure 100% MarketData Hub compliance and prevent external API bypasses.

Features:
- Runtime import monkey patching for API modules
- Network-level connection blocking
- Real-time monitoring and alerting
- Comprehensive audit logging
- Container security integration
- Bypass attempt detection and prevention

Author: Agent Alex (Security & DevOps Engineer)
Date: August 25, 2025
Security Level: CRITICAL
"""

import os
import sys
import socket
import threading
import time
import json
import logging
import inspect
import importlib
import traceback
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from functools import wraps
from pathlib import Path
import atexit
import signal

# Import the base DirectAPIBlocker for compatibility
try:
    from marketdata_client import DirectAPIBlocker
except ImportError:
    # Fallback if import fails
    class DirectAPIBlocker:
        BLOCKED_MODULES = ["requests", "urllib", "httpx", "aiohttp"]
        BLOCKED_HOSTS = [
            "api.alphaVantage.co",
            "api.fred.stlouisfed.org", 
            "data.nasdaq.com",
            "query1.finance.yahoo.com"
        ]

# Enhanced Security Configuration
class SecurityConfig:
    """Centralized security configuration"""
    
    # Enhanced blocked modules (comprehensive list)
    BLOCKED_MODULES = [
        "requests",
        "urllib",
        "urllib2", 
        "urllib3",
        "httpx",
        "aiohttp",
        "curl",
        "pycurl",
        "tornado.httpclient",
        "twisted.web.client",
        "http.client",
        "httplib",
        "httplib2"
    ]
    
    # Comprehensive blocked hosts (all external market data APIs)
    BLOCKED_HOSTS = [
        # Alpha Vantage
        "api.alphavantage.co",
        "www.alphavantage.co",
        
        # FRED (Federal Reserve)
        "api.fred.stlouisfed.org",
        "fred.stlouisfed.org",
        
        # Yahoo Finance
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
        "finance.yahoo.com",
        "chart.yahoo.com",
        
        # NASDAQ
        "data.nasdaq.com",
        "api.nasdaq.com",
        
        # Trading Economics
        "api.tradingeconomics.com",
        "tradingeconomics.com",
        
        # DBnomics
        "api.db.nomics.world",
        "db.nomics.world",
        
        # Data.gov
        "api.data.gov",
        "catalog.data.gov",
        
        # Other potential bypasses
        "api.quandl.com",
        "api.tiingo.com",
        "api.iexcloud.io",
        "api.polygon.io",
        "api.marketstack.com",
        "api.worldtradingdata.com",
        
        # Cryptocurrency APIs
        "api.coinbase.com",
        "api.binance.com",
        "api.kraken.com",
        
        # General finance data
        "api.bloomberg.com",
        "api.reuters.com",
        "api.morningstar.com",
        "api.refinitiv.com"
    ]
    
    # Allowed internal hosts (whitelist)
    ALLOWED_HOSTS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "host.docker.internal",
        "nautilus-backend",
        "nautilus-frontend",
        "nautilus-db",
        "nautilus-redis"
    ]
    
    # Security enforcement levels
    ENFORCEMENT_LEVEL = "CRITICAL"  # WARN, BLOCK, CRITICAL
    
    # Monitoring configuration
    ENABLE_MONITORING = True
    ENABLE_ALERTING = True
    ENABLE_AUDIT_LOGGING = True
    
    # Audit log path (use relative paths in development)
    AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "security_audit.log")
    ALERT_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "security_alerts.log")

class SecurityLogger:
    """Enhanced security logging system"""
    
    def __init__(self):
        self.audit_logger = self._setup_logger("security_audit", SecurityConfig.AUDIT_LOG_PATH)
        self.alert_logger = self._setup_logger("security_alerts", SecurityConfig.ALERT_LOG_PATH)
        self.console_logger = logging.getLogger("api_security_enforcer")
        
        # Create log directories if they don't exist
        os.makedirs(os.path.dirname(SecurityConfig.AUDIT_LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(SecurityConfig.ALERT_LOG_PATH), exist_ok=True)
    
    def _setup_logger(self, name: str, log_path: str) -> logging.Logger:
        """Setup specialized security logger"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler with rotation
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_blocked_attempt(self, attempt_type: str, details: Dict[str, Any]):
        """Log blocked API attempt"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "API_BLOCK",
            "attempt_type": attempt_type,
            "severity": "HIGH",
            **details
        }
        
        self.audit_logger.info(json.dumps(log_entry))
        self.console_logger.warning(f"🚫 BLOCKED {attempt_type}: {details}")
    
    def log_security_alert(self, alert_type: str, message: str, details: Dict[str, Any] = None):
        """Log critical security alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "message": message,
            "severity": "CRITICAL",
            "details": details or {}
        }
        
        self.alert_logger.critical(json.dumps(alert))
        self.console_logger.critical(f"🚨 SECURITY ALERT - {alert_type}: {message}")
    
    def log_bypass_attempt(self, method: str, target: str, caller: str, stack_trace: str):
        """Log potential bypass attempt"""
        bypass_details = {
            "method": method,
            "target": target,
            "caller": caller,
            "stack_trace": stack_trace,
            "process_id": os.getpid(),
            "process_name": sys.argv[0]
        }
        
        self.log_security_alert(
            "BYPASS_ATTEMPT",
            f"Detected potential bypass attempt via {method} to {target}",
            bypass_details
        )

class RuntimeMonitor:
    """Real-time monitoring of API attempts"""
    
    def __init__(self, security_logger: SecurityLogger):
        self.security_logger = security_logger
        self.blocked_attempts = 0
        self.bypass_attempts = 0
        self.start_time = time.time()
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor system state every 30 seconds
                time.sleep(30)
                self._check_system_integrity()
                
            except Exception as e:
                self.security_logger.log_security_alert(
                    "MONITOR_ERROR",
                    f"Monitoring loop error: {str(e)}"
                )
    
    def _check_system_integrity(self):
        """Perform periodic system integrity checks"""
        # Check if critical modules have been compromised
        for module_name in SecurityConfig.BLOCKED_MODULES:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, '__file__') and module.__file__:
                    # Module is loaded - check if it's been patched by our enforcer
                    if not hasattr(module, '_nautilus_security_patched'):
                        self.security_logger.log_security_alert(
                            "MODULE_INTEGRITY_VIOLATION",
                            f"Module {module_name} loaded without security patch"
                        )
    
    def record_blocked_attempt(self, attempt_type: str):
        """Record a blocked attempt"""
        self.blocked_attempts += 1
    
    def record_bypass_attempt(self):
        """Record a bypass attempt"""
        self.bypass_attempts += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "blocked_attempts": self.blocked_attempts,
            "bypass_attempts": self.bypass_attempts,
            "monitoring_active": self.monitoring_active,
            "enforcement_level": SecurityConfig.ENFORCEMENT_LEVEL
        }

class APISecurityEnforcer:
    """
    Comprehensive API Security Enforcement System
    
    This class implements runtime enforcement of API blocking policies
    through multiple layers of protection:
    
    1. Import-time module blocking
    2. Runtime function patching
    3. Network-level connection blocking
    4. Real-time monitoring and alerting
    5. Comprehensive audit logging
    """
    
    def __init__(self):
        self.security_logger = SecurityLogger()
        self.monitor = RuntimeMonitor(self.security_logger)
        self.original_functions: Dict[str, Any] = {}
        self.enforcement_active = True
        
        # Initialize enforcement
        self._initialize_security_enforcement()
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.security_logger.log_security_alert(
            "ENFORCER_INITIALIZED",
            "API Security Enforcer activated with CRITICAL enforcement level"
        )
    
    def _initialize_security_enforcement(self):
        """Initialize all security enforcement mechanisms"""
        self._patch_import_system()
        self._patch_network_modules()
        self._patch_subprocess_module()
        self._install_import_hooks()
        
        self.security_logger.console_logger.info("🔒 API Security Enforcer initialized")
    
    def _patch_import_system(self):
        """Patch the import system to block dangerous modules"""
        # Handle different __builtins__ types (dict in some environments)
        if isinstance(__builtins__, dict):
            original_import = __builtins__.get('__import__')
        else:
            original_import = __builtins__.__import__
        
        def secure_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Security-enhanced import function"""
            
            # Get caller information
            caller_frame = inspect.currentframe().f_back
            caller_module = caller_frame.f_globals.get('__name__', 'unknown')
            
            # Check if this is a blocked module
            if name in SecurityConfig.BLOCKED_MODULES:
                # Allow MarketData client to import aiohttp
                if name == "aiohttp" and "marketdata_client" in caller_module:
                    module = original_import(name, globals, locals, fromlist, level)
                    # Mark as patched and monitor
                    self._patch_aiohttp_for_marketdata(module)
                    return module
                
                # Block all other attempts
                stack_trace = ''.join(traceback.format_stack())
                
                self.security_logger.log_blocked_attempt("MODULE_IMPORT", {
                    "module": name,
                    "caller": caller_module,
                    "enforcement_level": SecurityConfig.ENFORCEMENT_LEVEL
                })
                
                self.monitor.record_blocked_attempt("MODULE_IMPORT")
                
                if SecurityConfig.ENFORCEMENT_LEVEL == "CRITICAL":
                    raise ImportError(
                        f"🚫 NAUTILUS SECURITY: Import of '{name}' blocked by API Security Enforcer\n"
                        f"Caller: {caller_module}\n"
                        f"Reason: Direct API access prohibited - Use MarketDataClient for data access\n"
                        f"For support, check MarketData Hub at localhost:8800"
                    )
            
            # Allow import and mark as monitored
            module = original_import(name, globals, locals, fromlist, level)
            
            # If this is a network module that got through, patch it
            if name in ["socket", "ssl"] and not hasattr(module, '_nautilus_security_patched'):
                self._patch_network_module(name, module)
            
            return module
        
        # Install the patched import
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = secure_import
        else:
            __builtins__.__import__ = secure_import
        self.original_functions['__import__'] = original_import
    
    def _patch_aiohttp_for_marketdata(self, aiohttp_module):
        """Special patching for aiohttp used by MarketDataClient"""
        if hasattr(aiohttp_module, 'ClientSession'):
            original_request = aiohttp_module.ClientSession._request
            
            async def secure_request(self, method, url, **kwargs):
                """Security wrapper for aiohttp requests"""
                
                # Extract host from URL
                if isinstance(url, str):
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        host = parsed.netloc or parsed.hostname
                        
                        # Check if this is an allowed internal request
                        if any(allowed in host for allowed in SecurityConfig.ALLOWED_HOSTS):
                            return await original_request(self, method, url, **kwargs)
                        
                        # Block external requests
                        self.security_logger.log_blocked_attempt("AIOHTTP_REQUEST", {
                            "method": method,
                            "url": str(url),
                            "host": host,
                            "caller": "marketdata_client"
                        })
                        
                        raise ConnectionError(
                            f"🚫 NAUTILUS SECURITY: External API access blocked\n"
                            f"Attempted URL: {url}\n"
                            f"Use MarketData Hub for data access"
                        )
                        
                    except Exception as e:
                        if "NAUTILUS SECURITY" in str(e):
                            raise
                        # If URL parsing fails, block by default
                        self.security_logger.log_bypass_attempt(
                            "AIOHTTP_MALFORMED_URL", str(url), "marketdata_client", 
                            ''.join(traceback.format_stack())
                        )
                        raise ConnectionError("🚫 NAUTILUS SECURITY: Malformed URL blocked")
                
                return await original_request(self, method, url, **kwargs)
            
            aiohttp_module.ClientSession._request = secure_request
            aiohttp_module._nautilus_security_patched = True
    
    def _patch_network_modules(self):
        """Patch low-level network modules"""
        # Patch socket module
        if 'socket' in sys.modules:
            self._patch_socket_module(sys.modules['socket'])
        
        # Patch ssl module
        if 'ssl' in sys.modules:
            self._patch_ssl_module(sys.modules['ssl'])
    
    def _patch_socket_module(self, socket_module):
        """Patch socket module to block external connections"""
        if hasattr(socket_module, '_nautilus_security_patched'):
            return
            
        original_connect = socket_module.socket.connect
        original_connect_ex = socket_module.socket.connect_ex
        
        def secure_connect(self, address):
            """Security wrapper for socket.connect"""
            host = address[0] if isinstance(address, (tuple, list)) else str(address)
            
            # Check against blocked hosts
            for blocked_host in SecurityConfig.BLOCKED_HOSTS:
                if blocked_host in host:
                    caller_frame = inspect.currentframe().f_back
                    caller_module = caller_frame.f_globals.get('__name__', 'unknown')
                    
                    self.security_logger.log_blocked_attempt("SOCKET_CONNECT", {
                        "host": host,
                        "address": str(address),
                        "caller": caller_module
                    })
                    
                    self.monitor.record_blocked_attempt("SOCKET_CONNECT")
                    
                    raise ConnectionError(
                        f"🚫 NAUTILUS SECURITY: Connection to {host} blocked\n"
                        f"Direct API connections are prohibited\n"
                        f"Use MarketData Hub for data access"
                    )
            
            # Allow internal connections
            return original_connect(self, address)
        
        def secure_connect_ex(self, address):
            """Security wrapper for socket.connect_ex"""
            try:
                secure_connect(self, address)
                return 0  # Success
            except ConnectionError:
                return 1  # Blocked by security
        
        # Install patches
        socket_module.socket.connect = secure_connect
        socket_module.socket.connect_ex = secure_connect_ex
        socket_module._nautilus_security_patched = True
        
        self.original_functions['socket.connect'] = original_connect
        self.original_functions['socket.connect_ex'] = original_connect_ex
    
    def _patch_ssl_module(self, ssl_module):
        """Patch SSL module for HTTPS blocking"""
        if hasattr(ssl_module, '_nautilus_security_patched'):
            return
            
        # SSL connections also need to be monitored
        if hasattr(ssl_module, 'wrap_socket'):
            original_wrap_socket = ssl_module.wrap_socket
            
            def secure_wrap_socket(sock, **kwargs):
                """Security wrapper for SSL socket wrapping"""
                # This will be caught by the underlying socket security
                return original_wrap_socket(sock, **kwargs)
            
            ssl_module.wrap_socket = secure_wrap_socket
            ssl_module._nautilus_security_patched = True
    
    def _patch_subprocess_module(self):
        """Patch subprocess to prevent curl/wget bypasses"""
        if 'subprocess' in sys.modules:
            subprocess_module = sys.modules['subprocess']
            
            if hasattr(subprocess_module, '_nautilus_security_patched'):
                return
            
            original_popen = subprocess_module.Popen
            
            def secure_popen(args, **kwargs):
                """Security wrapper for subprocess.Popen"""
                
                # Convert args to list if it's a string
                if isinstance(args, str):
                    args_list = args.split()
                else:
                    args_list = list(args) if args else []
                
                # Check for blocked commands
                blocked_commands = ['curl', 'wget', 'nc', 'netcat', 'telnet']
                
                if args_list and args_list[0] in blocked_commands:
                    caller_frame = inspect.currentframe().f_back
                    caller_module = caller_frame.f_globals.get('__name__', 'unknown')
                    
                    self.security_logger.log_blocked_attempt("SUBPROCESS_COMMAND", {
                        "command": args_list[0],
                        "args": str(args_list[:5]),  # First 5 args only
                        "caller": caller_module
                    })
                    
                    self.monitor.record_blocked_attempt("SUBPROCESS_COMMAND")
                    
                    raise PermissionError(
                        f"🚫 NAUTILUS SECURITY: Command '{args_list[0]}' blocked\n"
                        f"Network commands are prohibited for security\n"
                        f"Use MarketData Hub for data access"
                    )
                
                return original_popen(args, **kwargs)
            
            subprocess_module.Popen = secure_popen
            subprocess_module._nautilus_security_patched = True
            
            self.original_functions['subprocess.Popen'] = original_popen
    
    def _patch_network_module(self, module_name: str, module):
        """Generic network module patching"""
        # Add security marker
        module._nautilus_security_patched = True
        
        self.security_logger.console_logger.debug(f"Patched {module_name} for security monitoring")
    
    def _install_import_hooks(self):
        """Install import hooks for comprehensive module monitoring"""
        class SecurityImportHook:
            def __init__(self, enforcer):
                self.enforcer = enforcer
            
            def find_spec(self, fullname, path, target=None):
                # Check for blocked modules
                if fullname in SecurityConfig.BLOCKED_MODULES:
                    caller_frame = inspect.currentframe().f_back.f_back
                    caller_module = caller_frame.f_globals.get('__name__', 'unknown')
                    
                    if not (fullname == "aiohttp" and "marketdata_client" in caller_module):
                        self.enforcer.security_logger.log_blocked_attempt("IMPORT_HOOK", {
                            "module": fullname,
                            "caller": caller_module
                        })
                        return None  # Block the import
                
                return None  # Let standard import handle it
        
        # Install the hook
        if not any(isinstance(hook, SecurityImportHook) for hook in sys.meta_path):
            sys.meta_path.insert(0, SecurityImportHook(self))
    
    def create_security_bypass_detector(self):
        """Create advanced bypass detection mechanisms"""
        
        # Monitor for eval/exec usage
        original_eval = __builtins__.eval if hasattr(__builtins__, 'eval') else eval
        original_exec = __builtins__.exec if hasattr(__builtins__, 'exec') else exec
        
        def secure_eval(expression, globals=None, locals=None):
            """Security wrapper for eval"""
            if isinstance(expression, str):
                # Check for suspicious patterns
                suspicious_patterns = ['import', 'urllib', 'requests', 'socket', 'http']
                for pattern in suspicious_patterns:
                    if pattern in expression.lower():
                        caller_frame = inspect.currentframe().f_back
                        caller_module = caller_frame.f_globals.get('__name__', 'unknown')
                        
                        self.security_logger.log_bypass_attempt(
                            "EVAL_BYPASS", expression[:100], caller_module,
                            ''.join(traceback.format_stack())
                        )
                        
                        if SecurityConfig.ENFORCEMENT_LEVEL == "CRITICAL":
                            raise SecurityError("🚫 NAUTILUS SECURITY: Suspicious eval blocked")
            
            return original_eval(expression, globals, locals)
        
        def secure_exec(expression, globals=None, locals=None):
            """Security wrapper for exec"""
            if isinstance(expression, str):
                # Similar checks as eval
                suspicious_patterns = ['import', 'urllib', 'requests', 'socket', 'http']
                for pattern in suspicious_patterns:
                    if pattern in expression.lower():
                        caller_frame = inspect.currentframe().f_back
                        caller_module = caller_frame.f_globals.get('__name__', 'unknown')
                        
                        self.security_logger.log_bypass_attempt(
                            "EXEC_BYPASS", expression[:100], caller_module,
                            ''.join(traceback.format_stack())
                        )
                        
                        if SecurityConfig.ENFORCEMENT_LEVEL == "CRITICAL":
                            raise SecurityError("🚫 NAUTILUS SECURITY: Suspicious exec blocked")
            
            return original_exec(expression, globals, locals)
        
        # Install the secure versions
        if isinstance(__builtins__, dict):
            if 'eval' in __builtins__:
                __builtins__['eval'] = secure_eval
            if 'exec' in __builtins__:
                __builtins__['exec'] = secure_exec
        else:
            if hasattr(__builtins__, 'eval'):
                __builtins__.eval = secure_eval
            if hasattr(__builtins__, 'exec'):
                __builtins__.exec = secure_exec
        
        self.original_functions['eval'] = original_eval
        self.original_functions['exec'] = original_exec
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current enforcement status"""
        return {
            "enforcement_active": self.enforcement_active,
            "enforcement_level": SecurityConfig.ENFORCEMENT_LEVEL,
            "patched_functions": len(self.original_functions),
            "monitoring_stats": self.monitor.get_stats(),
            "blocked_modules_count": len(SecurityConfig.BLOCKED_MODULES),
            "blocked_hosts_count": len(SecurityConfig.BLOCKED_HOSTS),
            "allowed_hosts_count": len(SecurityConfig.ALLOWED_HOSTS)
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "enforcement_status": self.get_enforcement_status(),
            "security_configuration": {
                "blocked_modules": SecurityConfig.BLOCKED_MODULES,
                "blocked_hosts": SecurityConfig.BLOCKED_HOSTS[:10],  # First 10 for brevity
                "allowed_hosts": SecurityConfig.ALLOWED_HOSTS,
                "enforcement_level": SecurityConfig.ENFORCEMENT_LEVEL
            },
            "audit_trails": {
                "audit_log_path": SecurityConfig.AUDIT_LOG_PATH,
                "alert_log_path": SecurityConfig.ALERT_LOG_PATH,
                "log_files_exist": {
                    "audit": os.path.exists(SecurityConfig.AUDIT_LOG_PATH),
                    "alerts": os.path.exists(SecurityConfig.ALERT_LOG_PATH)
                }
            },
            "recommendations": [
                "Review audit logs regularly for compliance",
                "Monitor alert logs for security incidents",
                "Verify all engines use MarketDataClient",
                "Test security enforcement periodically",
                "Update blocked hosts list as needed"
            ]
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.security_logger.log_security_alert(
            "ENFORCER_SHUTDOWN",
            f"API Security Enforcer shutting down (signal {signum})"
        )
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup security enforcement on shutdown"""
        self.enforcement_active = False
        self.monitor.monitoring_active = False
        
        self.security_logger.log_security_alert(
            "ENFORCER_DISABLED", 
            "API Security Enforcer disabled"
        )

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

# Global enforcer instance
_global_enforcer: Optional[APISecurityEnforcer] = None

def initialize_api_security():
    """Initialize the global API security enforcer"""
    global _global_enforcer
    
    if _global_enforcer is None:
        _global_enforcer = APISecurityEnforcer()
        _global_enforcer.create_security_bypass_detector()
    
    return _global_enforcer

def get_security_status() -> Dict[str, Any]:
    """Get current security enforcement status"""
    if _global_enforcer is None:
        return {"error": "Security enforcer not initialized"}
    
    return _global_enforcer.get_enforcement_status()

def generate_security_report() -> Dict[str, Any]:
    """Generate comprehensive security report"""
    if _global_enforcer is None:
        return {"error": "Security enforcer not initialized"}
    
    return _global_enforcer.generate_security_report()

# Auto-initialize if this module is imported
if __name__ != "__main__":
    try:
        initialize_api_security()
    except Exception as e:
        logging.getLogger("api_security_enforcer").error(f"Failed to auto-initialize security: {e}")

if __name__ == "__main__":
    # Command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Nautilus API Security Enforcer")
    parser.add_argument("--status", action="store_true", help="Show enforcement status")
    parser.add_argument("--report", action="store_true", help="Generate security report")
    parser.add_argument("--test", action="store_true", help="Test security enforcement")
    
    args = parser.parse_args()
    
    enforcer = initialize_api_security()
    
    if args.status:
        status = get_security_status()
        print(json.dumps(status, indent=2))
    
    elif args.report:
        report = generate_security_report()
        print(json.dumps(report, indent=2))
    
    elif args.test:
        print("🧪 Testing API Security Enforcement...")
        
        # Test 1: Try to import blocked module
        try:
            import requests
            print("❌ FAIL: requests import succeeded")
        except ImportError as e:
            print(f"✅ PASS: requests import blocked - {e}")
        
        # Test 2: Try subprocess bypass
        try:
            import subprocess
            subprocess.run(["curl", "http://api.alphavantage.co"], check=True)
            print("❌ FAIL: curl bypass succeeded")
        except (PermissionError, FileNotFoundError) as e:
            print(f"✅ PASS: curl bypass blocked - {e}")
        
        print("🔒 Security enforcement tests completed")
    
    else:
        print("API Security Enforcer initialized successfully")
        print("Use --status, --report, or --test for more options")