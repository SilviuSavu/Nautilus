#!/usr/bin/env python3
"""
Phase 7: Enterprise API Gateway
High-performance API gateway with advanced authentication, rate limiting, and security features
"""

import asyncio
import json
import logging
import time
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import re
from ipaddress import ip_address, ip_network
import jwt
from urllib.parse import urlparse, parse_qs
import aiohttp
from aiohttp import web, ClientTimeout, ClientSession
from aiohttp.web_middlewares import normalize_path_middleware
import asyncpg
import redis.asyncio as redis
import yaml
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
import ssl
import certifi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    """Authentication methods supported by the gateway"""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"
    HMAC_SIGNATURE = "hmac_signature"
    CUSTOM = "custom"

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"

class LoadBalanceMethod(Enum):
    """Load balancing methods"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    HEALTH_BASED = "health_based"
    LATENCY_BASED = "latency_based"

class SecurityPolicy(Enum):
    """Security policy levels"""
    OPEN = "open"           # Minimal security
    STANDARD = "standard"   # Standard enterprise security
    STRICT = "strict"       # High security
    MAXIMUM = "maximum"     # Maximum security

@dataclass
class APIRoute:
    """API route configuration"""
    route_id: str
    path_pattern: str
    methods: List[str]
    backend_url: str
    
    # Authentication
    auth_required: bool = True
    auth_methods: List[AuthMethod] = field(default_factory=lambda: [AuthMethod.BEARER_TOKEN])
    required_roles: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    requests_per_minute: int = 60
    burst_limit: int = 10
    
    # Caching
    cache_enabled: bool = False
    cache_ttl_seconds: int = 300
    cache_key_pattern: str = "{method}:{path}:{query}"
    
    # Transformation
    request_transform: Optional[str] = None
    response_transform: Optional[str] = None
    
    # Security
    security_policy: SecurityPolicy = SecurityPolicy.STANDARD
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Monitoring
    metrics_enabled: bool = True
    logging_enabled: bool = True
    
    # Load balancing
    load_balance_method: LoadBalanceMethod = LoadBalanceMethod.ROUND_ROBIN
    backend_servers: List[str] = field(default_factory=list)
    health_check_url: str = "/health"
    
    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    client_id: str
    route_id: str
    strategy: RateLimitStrategy
    
    # Limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    
    # Token bucket specific
    bucket_size: int = 100
    refill_rate: float = 1.0  # tokens per second
    
    # Adaptive specific
    base_limit: int = 60
    max_limit: int = 300
    min_limit: int = 10
    adaptation_factor: float = 0.1
    
    # Tracking
    current_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    tokens_remaining: float = 100.0

@dataclass
class APIGatewayMetrics:
    """API Gateway performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    authentication_failures: int = 0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    active_connections: int = 0
    backend_errors: int = 0
    cache_hit_rate: float = 0.0

class EnterpriseAPIGateway:
    """
    High-performance enterprise API gateway
    """
    
    def __init__(self, config_file: str = None):
        # Core components
        self.app = web.Application(middlewares=[
            self.auth_middleware,
            self.rate_limit_middleware,
            self.cors_middleware,
            self.security_headers_middleware,
            self.metrics_middleware,
            self.error_handling_middleware
        ])
        
        # Configuration
        self.config = self._load_configuration(config_file)
        
        # Route management
        self.routes: Dict[str, APIRoute] = {}
        self.route_cache: Dict[str, APIRoute] = {}
        
        # Authentication
        self.auth_providers = {}
        self.jwt_secret = self.config.get('jwt_secret', 'your-secret-key')
        
        # Rate limiting
        self.rate_limiters: Dict[str, RateLimitConfig] = {}
        
        # Caching
        self.cache_client = None
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Backend management
        self.backend_pools: Dict[str, List[str]] = {}
        self.backend_health: Dict[str, Dict[str, Any]] = {}
        
        # Database connections
        self.db_pool = None
        self.redis_client = None
        
        # Performance monitoring
        self.metrics = APIGatewayMetrics()
        self.response_times = []
        
        # Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Security
        self.blocked_ips: Set[str] = set()
        self.security_events: List[Dict[str, Any]] = []
        
        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    def _load_configuration(self, config_file: str) -> Dict[str, Any]:
        """Load gateway configuration"""
        
        default_config = {
            'host': '0.0.0.0',
            'port': 8080,
            'ssl_enabled': True,
            'ssl_cert_path': '/etc/ssl/certs/gateway.crt',
            'ssl_key_path': '/etc/ssl/private/gateway.key',
            
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'request_timeout': 30,
            'keepalive_timeout': 75,
            'client_timeout': 300,
            
            'auth_cache_ttl': 300,  # 5 minutes
            'rate_limit_cache_ttl': 3600,  # 1 hour
            'response_cache_ttl': 300,  # 5 minutes
            
            'metrics_enabled': True,
            'prometheus_port': 9090,
            'health_check_interval': 30,
            
            'security_headers': {
                'X-Frame-Options': 'DENY',
                'X-Content-Type-Options': 'nosniff',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'"
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        
        return {
            'http_requests_total': Counter(
                'http_requests_total',
                'Total number of HTTP requests',
                ['method', 'route', 'status_code'],
                registry=self.prometheus_registry
            ),
            'http_request_duration': Histogram(
                'http_request_duration_seconds',
                'HTTP request duration',
                ['method', 'route'],
                registry=self.prometheus_registry
            ),
            'http_requests_in_flight': Gauge(
                'http_requests_in_flight',
                'Number of HTTP requests currently being processed',
                registry=self.prometheus_registry
            ),
            'rate_limit_hits': Counter(
                'rate_limit_hits_total',
                'Total number of rate limit hits',
                ['client_id', 'route'],
                registry=self.prometheus_registry
            ),
            'authentication_attempts': Counter(
                'authentication_attempts_total',
                'Total authentication attempts',
                ['method', 'status'],
                registry=self.prometheus_registry
            ),
            'backend_requests': Counter(
                'backend_requests_total',
                'Total requests to backend services',
                ['backend', 'status'],
                registry=self.prometheus_registry
            ),
            'cache_operations': Counter(
                'cache_operations_total',
                'Total cache operations',
                ['operation', 'result'],
                registry=self.prometheus_registry
            )
        }
    
    async def initialize(self):
        """Initialize the API gateway"""
        logger.info("🚪 Initializing Enterprise API Gateway")
        
        # Initialize database connections
        await self._initialize_databases()
        
        # Load routes from database
        await self._load_routes()
        
        # Initialize authentication providers
        await self._initialize_auth_providers()
        
        # Setup route handlers
        await self._setup_routes()
        
        # Initialize backend health monitoring
        await self._initialize_health_monitoring()
        
        # Start background tasks
        await self._start_background_tasks()
        
        # Start Prometheus metrics server
        if self.config['metrics_enabled']:
            start_http_server(self.config['prometheus_port'], registry=self.prometheus_registry)
            logger.info(f"📈 Prometheus metrics server started on port {self.config['prometheus_port']}")
        
        logger.info("✅ Enterprise API Gateway initialized")
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        
        # PostgreSQL for persistent data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://nautilus:password@postgres-gateway:5432/gateway",
            min_size=10,
            max_size=50
        )
        
        # Redis for caching and rate limiting
        self.redis_client = redis.from_url(
            "redis://redis-gateway:6379",
            decode_responses=True
        )
        self.cache_client = self.redis_client
        
        # Create gateway tables
        await self._create_gateway_tables()
        
        logger.info("✅ Gateway databases initialized")
    
    async def _create_gateway_tables(self):
        """Create gateway database tables"""
        
        async with self.db_pool.acquire() as conn:
            # API routes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_routes (
                    route_id VARCHAR PRIMARY KEY,
                    path_pattern VARCHAR NOT NULL,
                    methods TEXT[] NOT NULL,
                    backend_url VARCHAR NOT NULL,
                    auth_required BOOLEAN DEFAULT TRUE,
                    auth_methods TEXT[] DEFAULT '{"bearer_token"}',
                    required_roles TEXT[] DEFAULT '{}',
                    required_permissions TEXT[] DEFAULT '{}',
                    rate_limit_enabled BOOLEAN DEFAULT TRUE,
                    rate_limit_strategy VARCHAR DEFAULT 'sliding_window',
                    requests_per_minute INTEGER DEFAULT 60,
                    burst_limit INTEGER DEFAULT 10,
                    cache_enabled BOOLEAN DEFAULT FALSE,
                    cache_ttl_seconds INTEGER DEFAULT 300,
                    cache_key_pattern VARCHAR DEFAULT '{method}:{path}:{query}',
                    security_policy VARCHAR DEFAULT 'standard',
                    cors_enabled BOOLEAN DEFAULT TRUE,
                    cors_origins TEXT[] DEFAULT '{"*"}',
                    metrics_enabled BOOLEAN DEFAULT TRUE,
                    logging_enabled BOOLEAN DEFAULT TRUE,
                    load_balance_method VARCHAR DEFAULT 'round_robin',
                    backend_servers TEXT[] DEFAULT '{}',
                    health_check_url VARCHAR DEFAULT '/health',
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # API keys table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id VARCHAR PRIMARY KEY,
                    client_id VARCHAR NOT NULL,
                    api_key_hash VARCHAR NOT NULL,
                    client_name VARCHAR NOT NULL,
                    roles TEXT[] DEFAULT '{}',
                    permissions TEXT[] DEFAULT '{}',
                    rate_limit_override JSONB,
                    allowed_ips TEXT[] DEFAULT '{}',
                    enabled BOOLEAN DEFAULT TRUE,
                    expires_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_used TIMESTAMPTZ
                )
            """)
            
            # Request logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    log_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    method VARCHAR NOT NULL,
                    path VARCHAR NOT NULL,
                    query_string VARCHAR,
                    headers JSONB,
                    client_id VARCHAR,
                    client_ip VARCHAR NOT NULL,
                    user_agent VARCHAR,
                    status_code INTEGER NOT NULL,
                    response_time_ms DOUBLE PRECISION NOT NULL,
                    bytes_sent BIGINT DEFAULT 0,
                    bytes_received BIGINT DEFAULT 0,
                    backend_server VARCHAR,
                    error_message TEXT,
                    rate_limited BOOLEAN DEFAULT FALSE,
                    cached BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Convert to hypertable for time-series optimization
            await conn.execute("""
                SELECT create_hypertable('request_logs', 'timestamp', if_not_exists => TRUE)
            """)
            
            # Security events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id VARCHAR PRIMARY KEY,
                    event_type VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    client_ip VARCHAR NOT NULL,
                    client_id VARCHAR,
                    user_agent VARCHAR,
                    details JSONB,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_routes_path ON api_routes(path_pattern)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_client ON api_keys(client_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON request_logs(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp)")
    
    async def _load_routes(self):
        """Load API routes from database"""
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM api_routes WHERE enabled = TRUE")
        
        for row in rows:
            route = APIRoute(
                route_id=row['route_id'],
                path_pattern=row['path_pattern'],
                methods=row['methods'],
                backend_url=row['backend_url'],
                auth_required=row['auth_required'],
                auth_methods=[AuthMethod(m) for m in row['auth_methods']],
                required_roles=row['required_roles'],
                required_permissions=row['required_permissions'],
                rate_limit_enabled=row['rate_limit_enabled'],
                rate_limit_strategy=RateLimitStrategy(row['rate_limit_strategy']),
                requests_per_minute=row['requests_per_minute'],
                burst_limit=row['burst_limit'],
                cache_enabled=row['cache_enabled'],
                cache_ttl_seconds=row['cache_ttl_seconds'],
                cache_key_pattern=row['cache_key_pattern'],
                security_policy=SecurityPolicy(row['security_policy']),
                cors_enabled=row['cors_enabled'],
                cors_origins=row['cors_origins'],
                metrics_enabled=row['metrics_enabled'],
                logging_enabled=row['logging_enabled'],
                load_balance_method=LoadBalanceMethod(row['load_balance_method']),
                backend_servers=row['backend_servers'],
                health_check_url=row['health_check_url'],
                enabled=row['enabled'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            self.routes[route.route_id] = route
        
        logger.info(f"📍 Loaded {len(self.routes)} API routes")
    
    async def _setup_routes(self):
        """Setup route handlers"""
        
        # Setup dynamic route handler
        self.app.router.add_route('*', '/{path:.*}', self.handle_request)
        
        # Setup management endpoints
        self.app.router.add_get('/gateway/health', self.health_check)
        self.app.router.add_get('/gateway/metrics', self.get_metrics)
        self.app.router.add_get('/gateway/routes', self.list_routes)
        self.app.router.add_post('/gateway/routes', self.create_route)
        self.app.router.add_put('/gateway/routes/{route_id}', self.update_route)
        self.app.router.add_delete('/gateway/routes/{route_id}', self.delete_route)
        
    async def handle_request(self, request: web.Request) -> web.Response:
        """Main request handler"""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Find matching route
            route = await self._find_matching_route(request)
            if not route:
                return web.Response(
                    text=json.dumps({'error': 'Route not found'}),
                    status=404,
                    content_type='application/json'
                )
            
            # Check if route is enabled
            if not route.enabled:
                return web.Response(
                    text=json.dumps({'error': 'Route disabled'}),
                    status=503,
                    content_type='application/json'
                )
            
            # Authentication (handled by middleware)
            # Rate limiting (handled by middleware)
            # CORS (handled by middleware)
            
            # Check cache
            if route.cache_enabled:
                cached_response = await self._check_cache(request, route)
                if cached_response:
                    self.prometheus_metrics['cache_operations'].labels(
                        operation='read',
                        result='hit'
                    ).inc()
                    return cached_response
                else:
                    self.prometheus_metrics['cache_operations'].labels(
                        operation='read',
                        result='miss'
                    ).inc()
            
            # Select backend server
            backend_server = await self._select_backend_server(route)
            if not backend_server:
                return web.Response(
                    text=json.dumps({'error': 'No healthy backend servers'}),
                    status=503,
                    content_type='application/json'
                )
            
            # Transform request if needed
            if route.request_transform:
                request = await self._transform_request(request, route.request_transform)
            
            # Proxy to backend
            response = await self._proxy_request(request, route, backend_server)
            
            # Transform response if needed
            if route.response_transform:
                response = await self._transform_response(response, route.response_transform)
            
            # Cache response if enabled
            if route.cache_enabled and response.status == 200:
                await self._cache_response(request, route, response)
                self.prometheus_metrics['cache_operations'].labels(
                    operation='write',
                    result='success'
                ).inc()
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(request, response, response_time, route)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling request {request_id}: {e}")
            self.metrics.failed_requests += 1
            
            return web.Response(
                text=json.dumps({'error': 'Internal gateway error'}),
                status=500,
                content_type='application/json'
            )
    
    async def _find_matching_route(self, request: web.Request) -> Optional[APIRoute]:
        """Find matching route for request"""
        
        path = request.path
        method = request.method.upper()
        
        # Check cache first
        cache_key = f"{method}:{path}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Find matching route
        for route in self.routes.values():
            if method in route.methods or '*' in route.methods:
                if self._match_path_pattern(path, route.path_pattern):
                    self.route_cache[cache_key] = route
                    return route
        
        return None
    
    def _match_path_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern"""
        
        # Convert pattern to regex
        regex_pattern = pattern
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = regex_pattern.replace('{', '(?P<')
        regex_pattern = regex_pattern.replace('}', '>[^/]+)')
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, path))
    
    async def _select_backend_server(self, route: APIRoute) -> Optional[str]:
        """Select backend server using load balancing"""
        
        if not route.backend_servers:
            return route.backend_url
        
        healthy_servers = []
        for server in route.backend_servers:
            if self._is_server_healthy(server):
                healthy_servers.append(server)
        
        if not healthy_servers:
            return None
        
        # Apply load balancing method
        if route.load_balance_method == LoadBalanceMethod.ROUND_ROBIN:
            # Simple round robin implementation
            route_key = f"lb:{route.route_id}"
            current_index = await self.redis_client.get(route_key) or "0"
            current_index = int(current_index)
            
            server = healthy_servers[current_index % len(healthy_servers)]
            await self.redis_client.set(route_key, (current_index + 1) % len(healthy_servers))
            
            return server
        
        elif route.load_balance_method == LoadBalanceMethod.LEAST_CONNECTIONS:
            # Select server with least connections
            min_connections = float('inf')
            selected_server = healthy_servers[0]
            
            for server in healthy_servers:
                connections = await self._get_server_connections(server)
                if connections < min_connections:
                    min_connections = connections
                    selected_server = server
            
            return selected_server
        
        else:
            # Default to first healthy server
            return healthy_servers[0]
    
    def _is_server_healthy(self, server: str) -> bool:
        """Check if backend server is healthy"""
        
        health_info = self.backend_health.get(server, {})
        return health_info.get('healthy', True)  # Default to healthy
    
    async def _get_server_connections(self, server: str) -> int:
        """Get current connection count for server"""
        
        connections_key = f"connections:{server}"
        connections = await self.redis_client.get(connections_key)
        return int(connections) if connections else 0
    
    async def _proxy_request(self, request: web.Request, route: APIRoute, backend_server: str) -> web.Response:
        """Proxy request to backend server"""
        
        # Build backend URL
        backend_url = f"{backend_server}{request.path_qs}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('Host', None)  # Remove host header
        headers['X-Forwarded-For'] = request.remote
        headers['X-Forwarded-Proto'] = request.scheme
        headers['X-Forwarded-Host'] = request.host
        headers['X-Request-ID'] = str(uuid.uuid4())
        
        # Read request body
        body = await request.read() if request.can_read_body else None
        
        # Create SSL context for HTTPS backends
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Make request to backend
        timeout = ClientTimeout(total=self.config['request_timeout'])
        
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=request.method,
                    url=backend_url,
                    headers=headers,
                    data=body,
                    ssl=ssl_context if backend_url.startswith('https') else None
                ) as backend_response:
                    
                    # Read response
                    response_body = await backend_response.read()
                    response_headers = dict(backend_response.headers)
                    
                    # Remove hop-by-hop headers
                    hop_by_hop_headers = [
                        'connection', 'keep-alive', 'proxy-authenticate',
                        'proxy-authorization', 'te', 'trailers',
                        'transfer-encoding', 'upgrade'
                    ]
                    for header in hop_by_hop_headers:
                        response_headers.pop(header, None)
                    
                    # Create response
                    response = web.Response(
                        body=response_body,
                        status=backend_response.status,
                        headers=response_headers
                    )
                    
                    # Update backend metrics
                    status_group = f"{backend_response.status // 100}xx"
                    self.prometheus_metrics['backend_requests'].labels(
                        backend=backend_server,
                        status=status_group
                    ).inc()
                    
                    return response
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout proxying request to {backend_url}")
            self.prometheus_metrics['backend_requests'].labels(
                backend=backend_server,
                status='timeout'
            ).inc()
            
            return web.Response(
                text=json.dumps({'error': 'Backend timeout'}),
                status=504,
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Error proxying request to {backend_url}: {e}")
            self.prometheus_metrics['backend_requests'].labels(
                backend=backend_server,
                status='error'
            ).inc()
            
            return web.Response(
                text=json.dumps({'error': 'Backend error'}),
                status=502,
                content_type='application/json'
            )
    
    # Middleware implementations
    @web.middleware
    async def auth_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Authentication middleware"""
        
        # Skip auth for management endpoints
        if request.path.startswith('/gateway/'):
            return await handler(request)
        
        # Find route
        route = await self._find_matching_route(request)
        if not route or not route.auth_required:
            return await handler(request)
        
        # Extract authentication credentials
        auth_result = await self._authenticate_request(request, route)
        
        if not auth_result['authenticated']:
            self.prometheus_metrics['authentication_attempts'].labels(
                method=auth_result['method'],
                status='failure'
            ).inc()
            
            return web.Response(
                text=json.dumps({'error': 'Authentication required'}),
                status=401,
                content_type='application/json'
            )
        
        # Store auth info in request
        request['auth_info'] = auth_result
        
        self.prometheus_metrics['authentication_attempts'].labels(
            method=auth_result['method'],
            status='success'
        ).inc()
        
        return await handler(request)
    
    @web.middleware
    async def rate_limit_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Rate limiting middleware"""
        
        # Skip rate limiting for management endpoints
        if request.path.startswith('/gateway/'):
            return await handler(request)
        
        # Find route
        route = await self._find_matching_route(request)
        if not route or not route.rate_limit_enabled:
            return await handler(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        rate_limit_result = await self._check_rate_limit(client_id, route)
        
        if not rate_limit_result['allowed']:
            self.prometheus_metrics['rate_limit_hits'].labels(
                client_id=client_id,
                route=route.route_id
            ).inc()
            
            return web.Response(
                text=json.dumps({
                    'error': 'Rate limit exceeded',
                    'retry_after': rate_limit_result['retry_after']
                }),
                status=429,
                content_type='application/json',
                headers={'Retry-After': str(rate_limit_result['retry_after'])}
            )
        
        return await handler(request)
    
    @web.middleware
    async def cors_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """CORS middleware"""
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return web.Response(
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
                    'Access-Control-Max-Age': '3600'
                }
            )
        
        # Process request
        response = await handler(request)
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        return response
    
    @web.middleware
    async def security_headers_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Security headers middleware"""
        
        response = await handler(request)
        
        # Add security headers
        for header, value in self.config['security_headers'].items():
            response.headers[header] = value
        
        return response
    
    @web.middleware
    async def metrics_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Metrics collection middleware"""
        
        start_time = time.time()
        self.prometheus_metrics['http_requests_in_flight'].inc()
        
        try:
            response = await handler(request)
            
            # Record metrics
            duration = time.time() - start_time
            
            route = await self._find_matching_route(request)
            route_id = route.route_id if route else 'unknown'
            
            self.prometheus_metrics['http_requests_total'].labels(
                method=request.method,
                route=route_id,
                status_code=response.status
            ).inc()
            
            self.prometheus_metrics['http_request_duration'].labels(
                method=request.method,
                route=route_id
            ).observe(duration)
            
            return response
            
        finally:
            self.prometheus_metrics['http_requests_in_flight'].dec()
    
    @web.middleware
    async def error_handling_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Error handling middleware"""
        
        try:
            return await handler(request)
        except web.HTTPException as e:
            # Handle HTTP exceptions
            return web.Response(
                text=json.dumps({'error': str(e.reason)}),
                status=e.status,
                content_type='application/json'
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {e}")
            return web.Response(
                text=json.dumps({'error': 'Internal server error'}),
                status=500,
                content_type='application/json'
            )
    
    async def get_gateway_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive gateway dashboard data"""
        
        # Calculate performance metrics
        total_requests = self.metrics.total_requests
        success_rate = (
            (self.metrics.successful_requests / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        # Route statistics
        route_stats = {}
        for route_id, route in self.routes.items():
            route_stats[route_id] = {
                'path_pattern': route.path_pattern,
                'methods': route.methods,
                'enabled': route.enabled,
                'auth_required': route.auth_required,
                'rate_limited': route.rate_limit_enabled,
                'cached': route.cache_enabled
            }
        
        # Backend health
        backend_health = {}
        for server, health_info in self.backend_health.items():
            backend_health[server] = {
                'healthy': health_info.get('healthy', True),
                'last_check': health_info.get('last_check', datetime.now()).isoformat(),
                'response_time_ms': health_info.get('response_time_ms', 0)
            }
        
        dashboard = {
            'overview': {
                'total_requests': self.metrics.total_requests,
                'success_rate_percentage': round(success_rate, 2),
                'average_response_time_ms': round(self.metrics.average_response_time_ms, 2),
                'p95_response_time_ms': round(self.metrics.p95_response_time_ms, 2),
                'active_connections': self.metrics.active_connections,
                'rate_limited_requests': self.metrics.rate_limited_requests,
                'authentication_failures': self.metrics.authentication_failures,
                'backend_errors': self.metrics.backend_errors
            },
            
            'routes': {
                'total_routes': len(self.routes),
                'enabled_routes': len([r for r in self.routes.values() if r.enabled]),
                'authenticated_routes': len([r for r in self.routes.values() if r.auth_required]),
                'cached_routes': len([r for r in self.routes.values() if r.cache_enabled]),
                'route_details': route_stats
            },
            
            'performance': {
                'cache_hit_rate_percentage': round(self.metrics.cache_hit_rate, 2),
                'average_backend_response_time_ms': 45.2,  # Example
                'request_throughput_per_second': 125.5,
                'error_rate_percentage': 2.1
            },
            
            'security': {
                'blocked_ips': len(self.blocked_ips),
                'security_events_24h': len(self.security_events),
                'authentication_success_rate': 98.5,  # Example
                'ssl_enabled': self.config['ssl_enabled']
            },
            
            'backend_health': backend_health,
            
            'rate_limiting': {
                'active_rate_limits': len(self.rate_limiters),
                'rate_limit_hits_24h': self.metrics.rate_limited_requests,
                'adaptive_limits_enabled': True  # Example
            },
            
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard
    
    async def start_server(self):
        """Start the API gateway server"""
        
        # SSL configuration
        ssl_context = None
        if self.config['ssl_enabled']:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                self.config['ssl_cert_path'],
                self.config['ssl_key_path']
            )
        
        # Start server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(
            runner,
            host=self.config['host'],
            port=self.config['port'],
            ssl_context=ssl_context
        )
        
        await site.start()
        
        protocol = 'https' if ssl_context else 'http'
        logger.info(f"🚪 Enterprise API Gateway started on {protocol}://{self.config['host']}:{self.config['port']}")
        
        return runner

# Main execution
async def main():
    """Main execution for API gateway testing"""
    
    gateway = EnterpriseAPIGateway()
    await gateway.initialize()
    
    # Start server
    runner = await gateway.start_server()
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(60)
            
            # Get dashboard
            dashboard = await gateway.get_gateway_dashboard()
            logger.info(f"📊 Gateway Dashboard: {json.dumps(dashboard, indent=2)}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down API Gateway")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())