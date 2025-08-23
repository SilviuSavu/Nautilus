# Sprint 3 Integration Guide

## Overview

This comprehensive integration guide covers connecting external systems, building custom clients, and integrating with all Sprint 3 services including WebSocket infrastructure, analytics APIs, risk management systems, and strategy deployment frameworks.

## Table of Contents

1. [Authentication & Security](#authentication--security)
2. [WebSocket Integration](#websocket-integration)
3. [Analytics API Integration](#analytics-api-integration)
4. [Risk Management Integration](#risk-management-integration)
5. [Strategy Management Integration](#strategy-management-integration)
6. [Real-time Data Streaming](#real-time-data-streaming)
7. [Client SDK Examples](#client-sdk-examples)
8. [Third-party Platform Integration](#third-party-platform-integration)
9. [Monitoring & Health Checks](#monitoring--health-checks)
10. [Best Practices](#best-practices)

---

## Authentication & Security

### API Key Authentication

```bash
# Generate API key
curl -X POST http://localhost:8001/api/v1/auth/api-keys \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Trading System Integration",
    "permissions": ["analytics:read", "risk:read", "websocket:connect"],
    "expires_in": 2592000
  }'
```

### JWT Token Authentication

```python
# Python authentication example
import requests
import jwt
from datetime import datetime, timedelta

class NautilusAuth:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
        self.refresh_token = None
    
    def login(self, username: str, password: str) -> bool:
        """Login and obtain JWT tokens."""
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            return True
        return False
    
    def get_headers(self) -> dict:
        """Get authentication headers for API requests."""
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.refresh_token:
            return False
        
        response = requests.post(
            f"{self.base_url}/api/v1/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            return True
        return False

# Usage
auth = NautilusAuth("http://localhost:8001")
auth.login("your_username", "your_password")
headers = auth.get_headers()
```

### API Key Management

```javascript
// JavaScript API key authentication
class NautilusApiClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey
    };
  }

  async makeRequest(endpoint, method = 'GET', body = null) {
    const url = `${this.baseUrl}${endpoint}`;
    const options = {
      method,
      headers: this.defaultHeaders,
    };

    if (body && method !== 'GET') {
      options.body = JSON.stringify(body);
    }

    try {
      const response = await fetch(url, options);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }
}

// Usage
const client = new NautilusApiClient('http://localhost:8001', 'your-api-key');
```

---

## WebSocket Integration

### Real-time Connection Management

```python
# Advanced WebSocket client with reconnection
import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Callable, Optional
import backoff

class NautilusWebSocketClient:
    """Advanced WebSocket client with automatic reconnection."""
    
    def __init__(self, url: str, auth_token: str):
        self.url = url.replace("http", "ws")
        self.auth_token = auth_token
        self.websocket = None
        self.subscriptions = {}
        self.message_handlers = {}
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
    
    async def connect(self) -> bool:
        """Connect to WebSocket with authentication."""
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            self.websocket = await websockets.connect(
                f"{self.url}/ws/sprint3",
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            
            # Start message listening
            asyncio.create_task(self.listen_for_messages())
            
            # Send initial connection message
            await self.send_message({
                "type": "connection",
                "client_info": {
                    "client_type": "python_integration",
                    "version": "1.0.0"
                }
            })
            
            logging.info("Connected to Nautilus WebSocket")
            return True
            
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False
    
    @backoff.on_exception(
        backoff.expo,
        (websockets.exceptions.ConnectionClosed, ConnectionError),
        max_tries=10,
        max_time=300
    )
    async def reconnect(self):
        """Reconnect with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logging.error("Max reconnection attempts reached")
            return False
        
        self.reconnect_attempts += 1
        logging.info(f"Reconnecting... Attempt {self.reconnect_attempts}")
        
        await asyncio.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
        return await self.connect()
    
    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(data)
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logging.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            logging.warning("WebSocket connection closed")
            await self.reconnect()
        except Exception as e:
            logging.error(f"WebSocket listener error: {e}")
            self.connected = False
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        
        # Handle specific message types
        if message_type == "analytics_update":
            await self.handle_analytics_update(data)
        elif message_type == "risk_alert":
            await self.handle_risk_alert(data)
        elif message_type == "strategy_update":
            await self.handle_strategy_update(data)
        elif message_type in self.message_handlers:
            await self.message_handlers[message_type](data)
        else:
            logging.debug(f"Unhandled message type: {message_type}")
    
    async def subscribe(self, topic: str, filters: Dict[str, Any] = None, callback: Callable = None):
        """Subscribe to WebSocket topic."""
        subscription_id = f"{topic}_{len(self.subscriptions)}"
        
        subscription_message = {
            "type": "subscribe",
            "subscription_id": subscription_id,
            "topic": topic,
            "filters": filters or {}
        }
        
        await self.send_message(subscription_message)
        
        self.subscriptions[subscription_id] = {
            "topic": topic,
            "filters": filters,
            "callback": callback
        }
        
        return subscription_id
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if not self.connected or not self.websocket:
            raise ConnectionError("WebSocket not connected")
        
        await self.websocket.send(json.dumps(message))

# Usage example
async def main():
    client = NautilusWebSocketClient("http://localhost:8001", "your-jwt-token")
    
    # Connect
    await client.connect()
    
    # Subscribe to portfolio updates
    await client.subscribe(
        "portfolio.updates",
        {"portfolio_id": "PORTFOLIO_001"},
        callback=lambda data: print(f"Portfolio update: {data}")
    )
    
    # Subscribe to risk alerts
    await client.subscribe(
        "risk.alerts",
        {"severity": "critical"},
        callback=lambda data: print(f"Risk alert: {data}")
    )
    
    # Keep connection alive
    while True:
        await asyncio.sleep(1)

# Run the client
asyncio.run(main())
```

### JavaScript WebSocket Integration

```javascript
// JavaScript WebSocket client
class NautilusWebSocketClient {
  constructor(url, authToken) {
    this.url = url.replace('http', 'ws');
    this.authToken = authToken;
    this.ws = null;
    this.subscriptions = new Map();
    this.messageHandlers = new Map();
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${this.url}/ws/sprint3`);
        
        this.ws.onopen = () => {
          this.connected = true;
          this.reconnectAttempts = 0;
          
          // Send authentication
          this.send({
            type: 'auth',
            token: this.authToken
          });
          
          console.log('Connected to Nautilus WebSocket');
          resolve(true);
        };
        
        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Invalid JSON received:', event.data);
          }
        };
        
        this.ws.onclose = () => {
          this.connected = false;
          console.warn('WebSocket connection closed');
          this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
        
      } catch (error) {
        reject(error);
      }
    });
  }

  async attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return false;
    }

    this.reconnectAttempts++;
    console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);

    // Exponential backoff
    const delay = Math.pow(2, this.reconnectAttempts) * 1000;
    
    setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  handleMessage(data) {
    const messageType = data.type;
    
    if (this.messageHandlers.has(messageType)) {
      this.messageHandlers.get(messageType)(data);
    }
    
    // Handle subscription messages
    const subscriptionId = data.subscription_id;
    if (subscriptionId && this.subscriptions.has(subscriptionId)) {
      const subscription = this.subscriptions.get(subscriptionId);
      if (subscription.callback) {
        subscription.callback(data);
      }
    }
  }

  send(message) {
    if (!this.connected || !this.ws) {
      throw new Error('WebSocket not connected');
    }
    
    this.ws.send(JSON.stringify(message));
  }

  subscribe(topic, filters = {}, callback = null) {
    const subscriptionId = `${topic}_${this.subscriptions.size}`;
    
    const subscriptionMessage = {
      type: 'subscribe',
      subscription_id: subscriptionId,
      topic: topic,
      filters: filters
    };
    
    this.send(subscriptionMessage);
    
    this.subscriptions.set(subscriptionId, {
      topic,
      filters,
      callback
    });
    
    return subscriptionId;
  }

  on(messageType, handler) {
    this.messageHandlers.set(messageType, handler);
  }
}

// Usage
const client = new NautilusWebSocketClient('http://localhost:8001', 'your-jwt-token');

// Connect
client.connect().then(() => {
  // Subscribe to analytics updates
  client.subscribe('analytics.updates', 
    { portfolio_id: 'PORTFOLIO_001' },
    (data) => console.log('Analytics update:', data)
  );
  
  // Handle custom messages
  client.on('system_alert', (data) => {
    console.log('System alert:', data);
  });
});
```

---

## Analytics API Integration

### Performance Analytics Integration

```python
# Advanced analytics client
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class NautilusAnalyticsClient:
    """Comprehensive analytics API client."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_portfolio_analytics(
        self,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime,
        benchmark: str = "SPY",
        include_attribution: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive portfolio analytics."""
        
        payload = {
            "portfolio_id": portfolio_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "benchmark": benchmark,
            "include_attribution": include_attribution,
            "metrics": [
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "var_95",
                "beta",
                "alpha",
                "tracking_error",
                "information_ratio"
            ]
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/sprint3/analytics/performance/analyze",
            headers=self.headers,
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_real_time_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get real-time portfolio metrics."""
        async with self.session.get(
            f"{self.base_url}/api/v1/sprint3/analytics/portfolio/{portfolio_id}/summary",
            headers=self.headers,
            params={"include_real_time": True}
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def calculate_risk_analytics(
        self,
        portfolio_id: str,
        confidence_levels: List[float] = [0.95, 0.99],
        methods: List[str] = ["parametric", "historical", "monte_carlo"]
    ) -> Dict[str, Any]:
        """Calculate advanced risk analytics."""
        
        payload = {
            "portfolio_id": portfolio_id,
            "risk_models": methods,
            "confidence_levels": confidence_levels,
            "holding_period": 1,
            "include_correlations": True,
            "stress_scenarios": [
                {
                    "name": "market_crash",
                    "shocks": {
                        "SPY": -0.20,
                        "TLT": 0.05,
                        "VIX": 2.0
                    }
                }
            ]
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/sprint3/analytics/risk/analyze",
            headers=self.headers,
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def stream_analytics(self, portfolio_id: str, callback_func):
        """Stream real-time analytics updates."""
        # Implementation would use WebSocket streaming
        # This is a simplified example
        while True:
            try:
                metrics = await self.get_real_time_metrics(portfolio_id)
                await callback_func(metrics)
                await asyncio.sleep(1)  # 1-second updates
            except Exception as e:
                print(f"Streaming error: {e}")
                await asyncio.sleep(5)  # Wait before retry

# Usage example
async def analytics_example():
    async with NautilusAnalyticsClient("http://localhost:8001", "your-api-key") as client:
        # Get portfolio analytics
        analytics = await client.get_portfolio_analytics(
            portfolio_id="PORTFOLIO_001",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            benchmark="SPY"
        )
        
        print("Portfolio Performance:")
        print(f"Sharpe Ratio: {analytics['data']['performance_metrics']['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {analytics['data']['performance_metrics']['max_drawdown']:.3f}")
        
        # Get risk analytics
        risk_analytics = await client.calculate_risk_analytics("PORTFOLIO_001")
        
        print("Risk Analytics:")
        var_95 = risk_analytics['data']['var_calculations']['model_average']['var_95']
        print(f"VaR 95%: {var_95:.2f}")

asyncio.run(analytics_example())
```

### C# Analytics Integration

```csharp
// C# analytics client
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Collections.Generic;

public class NautilusAnalyticsClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public NautilusAnalyticsClient(string baseUrl, string apiKey)
    {
        _baseUrl = baseUrl;
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("X-API-Key", apiKey);
    }

    public async Task<PortfolioAnalytics> GetPortfolioAnalyticsAsync(
        string portfolioId, 
        DateTime startDate, 
        DateTime endDate,
        string benchmark = "SPY")
    {
        var request = new AnalyticsRequest
        {
            PortfolioId = portfolioId,
            StartDate = startDate,
            EndDate = endDate,
            Benchmark = benchmark,
            IncludeAttribution = true,
            Metrics = new[] 
            { 
                "sharpe_ratio", 
                "sortino_ratio", 
                "max_drawdown", 
                "var_95" 
            }
        };

        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync(
            $"{_baseUrl}/api/v1/sprint3/analytics/performance/analyze",
            content
        );

        response.EnsureSuccessStatusCode();
        var responseContent = await response.Content.ReadAsStringAsync();
        
        return JsonSerializer.Deserialize<PortfolioAnalytics>(responseContent);
    }

    public async Task<RealTimeMetrics> GetRealTimeMetricsAsync(string portfolioId)
    {
        var response = await _httpClient.GetAsync(
            $"{_baseUrl}/api/v1/sprint3/analytics/portfolio/{portfolioId}/summary?include_real_time=true"
        );

        response.EnsureSuccessStatusCode();
        var content = await response.Content.ReadAsStringAsync();
        
        return JsonSerializer.Deserialize<RealTimeMetrics>(content);
    }

    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}

// Data models
public class AnalyticsRequest
{
    public string PortfolioId { get; set; }
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public string Benchmark { get; set; }
    public bool IncludeAttribution { get; set; }
    public string[] Metrics { get; set; }
}

public class PortfolioAnalytics
{
    public bool Success { get; set; }
    public AnalyticsData Data { get; set; }
}

public class AnalyticsData
{
    public string PortfolioId { get; set; }
    public PerformanceMetrics PerformanceMetrics { get; set; }
    public AttributionData Attribution { get; set; }
}

public class PerformanceMetrics
{
    public double TotalReturn { get; set; }
    public double AnnualizedReturn { get; set; }
    public double Volatility { get; set; }
    public double SharpeRatio { get; set; }
    public double SortinoRatio { get; set; }
    public double MaxDrawdown { get; set; }
    public double Var95 { get; set; }
    public double Beta { get; set; }
    public double Alpha { get; set; }
}

// Usage
var client = new NautilusAnalyticsClient("http://localhost:8001", "your-api-key");

var analytics = await client.GetPortfolioAnalyticsAsync(
    "PORTFOLIO_001",
    DateTime.Now.AddDays(-30),
    DateTime.Now
);

Console.WriteLine($"Sharpe Ratio: {analytics.Data.PerformanceMetrics.SharpeRatio:F3}");
Console.WriteLine($"Max Drawdown: {analytics.Data.PerformanceMetrics.MaxDrawdown:F3}");

client.Dispose();
```

---

## Risk Management Integration

### Dynamic Risk Limits

```python
# Risk management client
class NautilusRiskClient:
    """Advanced risk management client."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def create_risk_limit(
        self,
        portfolio_id: str,
        limit_type: str,
        threshold_value: float,
        warning_threshold: float,
        action: str = "warn",
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new risk limit."""
        
        payload = {
            "name": f"{limit_type.title()} Limit - {portfolio_id}",
            "portfolio_id": portfolio_id,
            "limit_type": limit_type,
            "threshold_value": threshold_value,
            "warning_threshold": warning_threshold,
            "action": action,
            "parameters": parameters or {},
            "schedule": {
                "active_hours": {
                    "start": "09:30",
                    "end": "16:00",
                    "timezone": "America/New_York"
                },
                "active_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/risk/limits",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def get_real_time_risk(self, portfolio_id: str) -> Dict[str, Any]:
        """Get real-time risk metrics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/sprint3/risk/realtime/{portfolio_id}",
                headers=self.headers,
                params={
                    "metrics": "var_95,var_99,expected_shortfall,beta,tracking_error",
                    "include_positions": True
                }
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def start_risk_monitoring(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Start real-time risk monitoring."""
        payload = {
            "portfolio_ids": portfolio_ids,
            "monitoring_interval": 5,  # 5 seconds
            "alert_channels": ["email", "websocket"],
            "include_predictive_alerts": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/risk/monitoring/start",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def check_pre_trade_risk(
        self,
        portfolio_id: str,
        trade_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a trade would breach risk limits."""
        payload = {
            "portfolio_id": portfolio_id,
            "trade": trade_details,
            "check_all_limits": True,
            "include_impact_analysis": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/risk/pre-trade-check",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

# Usage example
async def risk_management_example():
    client = NautilusRiskClient("http://localhost:8001", "your-api-key")
    
    # Create VaR limit
    var_limit = await client.create_risk_limit(
        portfolio_id="PORTFOLIO_001",
        limit_type="var",
        threshold_value=50000.0,
        warning_threshold=40000.0,
        action="warn",
        parameters={"confidence_level": 0.95, "holding_period": 1}
    )
    
    print(f"Created VaR limit: {var_limit['data']['limit_id']}")
    
    # Get real-time risk metrics
    risk_metrics = await client.get_real_time_risk("PORTFOLIO_001")
    
    print("Current Risk Metrics:")
    for metric, value in risk_metrics['data']['risk_metrics'].items():
        print(f"{metric}: {value}")
    
    # Check pre-trade risk
    trade_check = await client.check_pre_trade_risk(
        portfolio_id="PORTFOLIO_001",
        trade_details={
            "symbol": "AAPL",
            "quantity": 1000,
            "side": "buy",
            "order_type": "market"
        }
    )
    
    if trade_check['data']['risk_approved']:
        print("Trade approved - no risk limits breached")
    else:
        print("Trade blocked - risk limits would be breached:")
        for breach in trade_check['data']['potential_breaches']:
            print(f"- {breach['limit_name']}: {breach['reason']}")

asyncio.run(risk_management_example())
```

---

## Strategy Management Integration

### Strategy Deployment Client

```python
# Strategy management client
class NautilusStrategyClient:
    """Strategy management and deployment client."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def create_strategy_version(
        self,
        strategy_id: str,
        version: str,
        strategy_code: str,
        strategy_config: Dict[str, Any],
        description: str = "",
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new strategy version."""
        
        payload = {
            "strategy_id": strategy_id,
            "version": version,
            "strategy_code": strategy_code,
            "strategy_config": strategy_config,
            "description": description,
            "tags": tags or []
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/strategy/versions",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def deploy_strategy(
        self,
        strategy_id: str,
        version: str,
        target_environment: str = "staging",
        deployment_strategy: str = "blue_green",
        auto_rollback: bool = True,
        rollback_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Deploy strategy with advanced options."""
        
        payload = {
            "strategy_id": strategy_id,
            "version": version,
            "target_environment": target_environment,
            "deployment_strategy": deployment_strategy,
            "auto_rollback": auto_rollback,
            "rollback_threshold": rollback_threshold,
            "canary_percentage": 10.0 if deployment_strategy == "canary" else None,
            "approval_required": target_environment == "production",
            "notifications": {
                "on_success": ["email:trading@company.com"],
                "on_failure": ["slack:#trading-alerts", "email:risk@company.com"],
                "on_rollback": ["email:risk@company.com", "sms:+1234567890"]
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/strategy/deploy",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status and details."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/sprint3/strategy/deployments/{deployment_id}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def rollback_deployment(
        self,
        deployment_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Rollback a deployment."""
        payload = {
            "deployment_id": deployment_id,
            "force": force,
            "reason": "Manual rollback requested"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/sprint3/strategy/rollback",
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def monitor_deployment_progress(
        self, 
        deployment_id: str,
        callback: Callable[[Dict[str, Any]], None] = None
    ):
        """Monitor deployment progress in real-time."""
        while True:
            try:
                status = await self.get_deployment_status(deployment_id)
                
                if callback:
                    callback(status)
                
                deployment_status = status['data']['status']
                
                if deployment_status in ['completed', 'failed', 'rolled_back']:
                    break
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error monitoring deployment: {e}")
                await asyncio.sleep(30)

# Usage example
async def strategy_deployment_example():
    client = NautilusStrategyClient("http://localhost:8001", "your-api-key")
    
    # Create strategy version
    strategy_code = """
    class MomentumStrategy(Strategy):
        def __init__(self):
            self.fast_period = 10
            self.slow_period = 20
            
        def on_bar(self, bar):
            fast_ma = self.calculate_sma(bar, self.fast_period)
            slow_ma = self.calculate_sma(bar, self.slow_period)
            
            if fast_ma > slow_ma and not self.position:
                self.buy(100)
            elif fast_ma < slow_ma and self.position:
                self.sell(self.position.quantity)
    """
    
    version_result = await client.create_strategy_version(
        strategy_id="MOMENTUM_V1",
        version="1.2.0",
        strategy_code=strategy_code,
        strategy_config={
            "risk_budget": 100000,
            "max_position_size": 50000,
            "stop_loss": 0.02,
            "take_profit": 0.05
        },
        description="Enhanced momentum strategy with risk controls",
        tags=["momentum", "trend-following", "v1.2"]
    )
    
    print(f"Created strategy version: {version_result['data']['version_id']}")
    
    # Deploy to staging
    deployment = await client.deploy_strategy(
        strategy_id="MOMENTUM_V1",
        version="1.2.0",
        target_environment="staging",
        deployment_strategy="blue_green",
        auto_rollback=True
    )
    
    deployment_id = deployment['data']['deployment_id']
    print(f"Started deployment: {deployment_id}")
    
    # Monitor deployment progress
    def progress_callback(status_data):
        status = status_data['data']['status']
        progress = status_data['data'].get('progress', 0)
        print(f"Deployment {deployment_id}: {status} ({progress}%)")
    
    await client.monitor_deployment_progress(deployment_id, progress_callback)
    
    print("Deployment monitoring completed")

asyncio.run(strategy_deployment_example())
```

---

## Real-time Data Streaming

### Multi-source Data Stream

```python
# Real-time data streaming client
import asyncio
import json
from typing import Dict, Any, Callable, List

class NautilusDataStreamer:
    """Real-time data streaming client."""
    
    def __init__(self, websocket_client: NautilusWebSocketClient):
        self.ws_client = websocket_client
        self.active_streams = {}
        self.data_handlers = {}
    
    async def stream_portfolio_analytics(
        self,
        portfolio_id: str,
        metrics: List[str] = None,
        update_interval: int = 1000,  # milliseconds
        callback: Callable[[Dict[str, Any]], None] = None
    ) -> str:
        """Stream real-time portfolio analytics."""
        
        stream_id = f"portfolio_analytics_{portfolio_id}"
        
        subscription_id = await self.ws_client.subscribe(
            "analytics.portfolio.updates",
            {
                "portfolio_id": portfolio_id,
                "metrics": metrics or [
                    "unrealized_pnl",
                    "realized_pnl",
                    "total_value",
                    "daily_return",
                    "sharpe_ratio",
                    "var_95"
                ],
                "update_interval": update_interval
            },
            callback=self._handle_portfolio_analytics
        )
        
        self.active_streams[stream_id] = {
            "subscription_id": subscription_id,
            "type": "portfolio_analytics",
            "callback": callback
        }
        
        return stream_id
    
    async def stream_risk_alerts(
        self,
        portfolio_id: str = None,
        severity_levels: List[str] = None,
        callback: Callable[[Dict[str, Any]], None] = None
    ) -> str:
        """Stream real-time risk alerts."""
        
        stream_id = f"risk_alerts_{portfolio_id or 'all'}"
        
        subscription_id = await self.ws_client.subscribe(
            "risk.alerts",
            {
                "portfolio_id": portfolio_id,
                "severity_levels": severity_levels or ["warning", "critical"],
                "alert_types": ["limit_breach", "var_exceeded", "concentration_risk"]
            },
            callback=self._handle_risk_alerts
        )
        
        self.active_streams[stream_id] = {
            "subscription_id": subscription_id,
            "type": "risk_alerts",
            "callback": callback
        }
        
        return stream_id
    
    async def stream_strategy_updates(
        self,
        strategy_id: str = None,
        environment: str = None,
        callback: Callable[[Dict[str, Any]], None] = None
    ) -> str:
        """Stream strategy deployment and performance updates."""
        
        stream_id = f"strategy_updates_{strategy_id or 'all'}"
        
        subscription_id = await self.ws_client.subscribe(
            "strategy.updates",
            {
                "strategy_id": strategy_id,
                "environment": environment,
                "update_types": [
                    "deployment_status",
                    "performance_metrics",
                    "error_events",
                    "position_changes"
                ]
            },
            callback=self._handle_strategy_updates
        )
        
        self.active_streams[stream_id] = {
            "subscription_id": subscription_id,
            "type": "strategy_updates",
            "callback": callback
        }
        
        return stream_id
    
    async def stream_market_data(
        self,
        symbols: List[str],
        data_types: List[str] = None,
        callback: Callable[[Dict[str, Any]], None] = None
    ) -> str:
        """Stream real-time market data."""
        
        stream_id = f"market_data_{'_'.join(symbols)}"
        
        subscription_id = await self.ws_client.subscribe(
            "market.data.updates",
            {
                "symbols": symbols,
                "data_types": data_types or ["price", "volume", "bid_ask"],
                "aggregation_interval": 100  # 100ms aggregation
            },
            callback=self._handle_market_data
        )
        
        self.active_streams[stream_id] = {
            "subscription_id": subscription_id,
            "type": "market_data",
            "callback": callback
        }
        
        return stream_id
    
    async def _handle_portfolio_analytics(self, data: Dict[str, Any]):
        """Handle portfolio analytics updates."""
        stream_id = f"portfolio_analytics_{data.get('portfolio_id')}"
        
        if stream_id in self.active_streams:
            callback = self.active_streams[stream_id].get('callback')
            if callback:
                await callback(data)
    
    async def _handle_risk_alerts(self, data: Dict[str, Any]):
        """Handle risk alert updates."""
        portfolio_id = data.get('portfolio_id', 'all')
        stream_id = f"risk_alerts_{portfolio_id}"
        
        if stream_id in self.active_streams:
            callback = self.active_streams[stream_id].get('callback')
            if callback:
                await callback(data)
    
    async def _handle_strategy_updates(self, data: Dict[str, Any]):
        """Handle strategy update messages."""
        strategy_id = data.get('strategy_id', 'all')
        stream_id = f"strategy_updates_{strategy_id}"
        
        if stream_id in self.active_streams:
            callback = self.active_streams[stream_id].get('callback')
            if callback:
                await callback(data)
    
    async def _handle_market_data(self, data: Dict[str, Any]):
        """Handle market data updates."""
        symbols = data.get('symbols', [])
        stream_id = f"market_data_{'_'.join(symbols)}"
        
        if stream_id in self.active_streams:
            callback = self.active_streams[stream_id].get('callback')
            if callback:
                await callback(data)
    
    async def stop_stream(self, stream_id: str):
        """Stop a specific data stream."""
        if stream_id in self.active_streams:
            subscription_id = self.active_streams[stream_id]['subscription_id']
            # Unsubscribe from WebSocket
            await self.ws_client.send_message({
                "type": "unsubscribe",
                "subscription_id": subscription_id
            })
            
            del self.active_streams[stream_id]

# Usage example
async def streaming_example():
    # Initialize WebSocket client
    ws_client = NautilusWebSocketClient("http://localhost:8001", "your-jwt-token")
    await ws_client.connect()
    
    # Initialize data streamer
    streamer = NautilusDataStreamer(ws_client)
    
    # Stream portfolio analytics
    async def handle_analytics(data):
        metrics = data.get('analytics', {})
        print(f"Portfolio Analytics Update:")
        print(f"  P&L: ${metrics.get('unrealized_pnl', 0):,.2f}")
        print(f"  Daily Return: {metrics.get('daily_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    
    analytics_stream = await streamer.stream_portfolio_analytics(
        portfolio_id="PORTFOLIO_001",
        callback=handle_analytics
    )
    
    # Stream risk alerts
    async def handle_risk_alerts(data):
        alert = data.get('alert', {})
        print(f"Risk Alert: {alert.get('message')}")
        print(f"  Severity: {alert.get('severity')}")
        print(f"  Portfolio: {alert.get('portfolio_id')}")
    
    risk_stream = await streamer.stream_risk_alerts(
        portfolio_id="PORTFOLIO_001",
        callback=handle_risk_alerts
    )
    
    # Stream market data
    async def handle_market_data(data):
        for symbol, quote in data.get('quotes', {}).items():
            print(f"{symbol}: ${quote.get('price'):.2f} ({quote.get('change'):+.2f})")
    
    market_stream = await streamer.stream_market_data(
        symbols=["AAPL", "MSFT", "GOOGL"],
        callback=handle_market_data
    )
    
    # Keep streams running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping streams...")
        await streamer.stop_stream(analytics_stream)
        await streamer.stop_stream(risk_stream)
        await streamer.stop_stream(market_stream)

asyncio.run(streaming_example())
```

This comprehensive integration guide provides detailed examples for connecting to all Sprint 3 services with production-ready code samples in multiple programming languages. The examples include proper error handling, authentication, and real-world usage patterns for enterprise trading systems.