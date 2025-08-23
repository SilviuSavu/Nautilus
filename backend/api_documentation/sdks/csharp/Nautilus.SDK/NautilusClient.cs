/*
 * Nautilus Trading Platform C# SDK
 * Official .NET client library for enterprise integration
 */

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Nautilus.SDK
{
    /// <summary>
    /// Main client for interacting with the Nautilus Trading Platform API
    /// Provides comprehensive API access with full .NET integration
    /// </summary>
    public class NautilusClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<NautilusClient> _logger;
        private readonly NautilusConfig _config;
        private readonly AuthenticationManager _authManager;
        private readonly WebSocketManager _webSocketManager;
        
        // API modules
        public IMarketDataAPI MarketData { get; }
        public IRiskManagementAPI Risk { get; }
        public IStrategyAPI Strategies { get; }
        public IAnalyticsAPI Analytics { get; }
        public ISystemAPI System { get; }

        /// <summary>
        /// Initialize new Nautilus client
        /// </summary>
        /// <param name="config">Client configuration</param>
        /// <param name="logger">Optional logger instance</param>
        public NautilusClient(NautilusConfig config, ILogger<NautilusClient> logger = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<NautilusClient>.Instance;

            // Initialize HTTP client with retry policy
            var handler = new HttpClientHandler();
            _httpClient = new HttpClient(handler)
            {
                BaseAddress = new Uri(_config.BaseUrl),
                Timeout = TimeSpan.FromMilliseconds(_config.TimeoutMs)
            };

            _httpClient.DefaultRequestHeaders.Add("User-Agent", "Nautilus-CSharp-SDK/3.0.0");
            
            // Initialize auth manager
            _authManager = new AuthenticationManager(_httpClient, _config, _logger);
            
            // Initialize WebSocket manager
            _webSocketManager = new WebSocketManager(_config, _authManager, _logger);

            // Initialize API modules
            MarketData = new MarketDataAPI(this, _logger);
            Risk = new RiskManagementAPI(this, _logger);
            Strategies = new StrategyAPI(this, _logger);
            Analytics = new AnalyticsAPI(this, _logger);
            System = new SystemAPI(this, _logger);

            _logger.LogInformation("Nautilus client initialized with base URL: {BaseUrl}", _config.BaseUrl);
        }

        /// <summary>
        /// Make authenticated HTTP request
        /// </summary>
        /// <typeparam name="T">Response type</typeparam>
        /// <param name="method">HTTP method</param>
        /// <param name="endpoint">API endpoint</param>
        /// <param name="data">Request data</param>
        /// <param name="queryParams">Query parameters</param>
        /// <param name="requireAuth">Whether authentication is required</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>API response</returns>
        public async Task<ApiResponse<T>> RequestAsync<T>(
            HttpMethod method,
            string endpoint,
            object data = null,
            Dictionary<string, string> queryParams = null,
            bool requireAuth = true,
            CancellationToken cancellationToken = default)
        {
            var requestUri = endpoint;
            
            // Add query parameters
            if (queryParams?.Count > 0)
            {
                var queryString = string.Join("&", 
                    queryParams.Select(kv => $"{Uri.EscapeDataString(kv.Key)}={Uri.EscapeDataString(kv.Value)}"));
                requestUri += $"?{queryString}";
            }

            var request = new HttpRequestMessage(method, requestUri);
            
            // Add authentication headers
            if (requireAuth)
            {
                var authHeaders = await _authManager.GetAuthHeadersAsync(cancellationToken);
                foreach (var header in authHeaders)
                {
                    request.Headers.Add(header.Key, header.Value);
                }
            }

            // Add request body
            if (data != null && (method == HttpMethod.Post || method == HttpMethod.Put))
            {
                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                request.Content = new StringContent(json, Encoding.UTF8, "application/json");
            }

            // Execute request with retry logic
            return await ExecuteWithRetryAsync<T>(request, cancellationToken);
        }

        /// <summary>
        /// Execute HTTP request with retry logic
        /// </summary>
        private async Task<ApiResponse<T>> ExecuteWithRetryAsync<T>(
            HttpRequestMessage request, 
            CancellationToken cancellationToken)
        {
            Exception lastException = null;
            
            for (int attempt = 0; attempt <= _config.MaxRetries; attempt++)
            {
                try
                {
                    var response = await _httpClient.SendAsync(request, cancellationToken);
                    
                    // Handle authentication errors
                    if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized && attempt == 0)
                    {
                        await _authManager.RefreshTokenAsync(cancellationToken);
                        
                        // Update auth headers and retry
                        var authHeaders = await _authManager.GetAuthHeadersAsync(cancellationToken);
                        foreach (var header in authHeaders)
                        {
                            request.Headers.Remove(header.Key);
                            request.Headers.Add(header.Key, header.Value);
                        }
                        continue;
                    }

                    // Handle rate limiting
                    if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                    {
                        var retryAfter = response.Headers.RetryAfter?.Delta ?? TimeSpan.FromSeconds(60);
                        throw new RateLimitException($"Rate limit exceeded. Retry after {retryAfter.TotalSeconds}s")
                        {
                            RetryAfter = retryAfter
                        };
                    }

                    // Handle other HTTP errors
                    if (!response.IsSuccessStatusCode)
                    {
                        var errorContent = await response.Content.ReadAsStringAsync();
                        var errorData = string.IsNullOrEmpty(errorContent) 
                            ? null 
                            : JsonSerializer.Deserialize<ErrorResponse>(errorContent);
                            
                        throw new NautilusException(
                            errorData?.Message ?? $"HTTP {(int)response.StatusCode} error",
                            (int)response.StatusCode,
                            errorData
                        );
                    }

                    // Parse successful response
                    var content = await response.Content.ReadAsStringAsync();
                    var responseData = string.IsNullOrEmpty(content) 
                        ? default(T)
                        : JsonSerializer.Deserialize<T>(content, new JsonSerializerOptions
                        {
                            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                            PropertyNameCaseInsensitive = true
                        });

                    return new ApiResponse<T>
                    {
                        Data = responseData,
                        StatusCode = (int)response.StatusCode,
                        Headers = response.Headers.ToDictionary(h => h.Key, h => string.Join(",", h.Value))
                    };
                }
                catch (HttpRequestException ex) when (attempt < _config.MaxRetries)
                {
                    lastException = ex;
                    var delay = TimeSpan.FromMilliseconds(_config.RetryDelayMs * (attempt + 1));
                    _logger.LogWarning("Request failed (attempt {Attempt}), retrying in {Delay}ms: {Error}", 
                        attempt + 1, delay.TotalMilliseconds, ex.Message);
                    await Task.Delay(delay, cancellationToken);
                }
                catch (TaskCanceledException ex) when (ex.InnerException is TimeoutException && attempt < _config.MaxRetries)
                {
                    lastException = ex;
                    var delay = TimeSpan.FromMilliseconds(_config.RetryDelayMs * (attempt + 1));
                    _logger.LogWarning("Request timeout (attempt {Attempt}), retrying in {Delay}ms", 
                        attempt + 1, delay.TotalMilliseconds);
                    await Task.Delay(delay, cancellationToken);
                }
            }

            throw new NautilusException($"Request failed after {_config.MaxRetries} retries", 0, lastException);
        }

        /// <summary>
        /// Authenticate user and get access token
        /// </summary>
        /// <param name="username">Username or email</param>
        /// <param name="password">Password</param>
        /// <param name="rememberMe">Use long-lived token</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Login response</returns>
        public async Task<LoginResponse> LoginAsync(
            string username, 
            string password, 
            bool rememberMe = false,
            CancellationToken cancellationToken = default)
        {
            var loginData = new LoginRequest
            {
                Username = username,
                Password = password,
                RememberMe = rememberMe
            };

            var response = await RequestAsync<LoginResponse>(
                HttpMethod.Post,
                "/api/v1/auth/login",
                loginData,
                requireAuth: false,
                cancellationToken: cancellationToken
            );

            // Store tokens
            await _authManager.SetTokensAsync(
                response.Data.AccessToken,
                response.Data.RefreshToken,
                cancellationToken
            );

            _logger.LogInformation("User authenticated successfully");
            return response.Data;
        }

        /// <summary>
        /// Refresh access token
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>New token response</returns>
        public async Task<TokenRefreshResponse> RefreshTokenAsync(CancellationToken cancellationToken = default)
        {
            return await _authManager.RefreshTokenAsync(cancellationToken);
        }

        /// <summary>
        /// Logout and clear stored tokens
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task LogoutAsync(CancellationToken cancellationToken = default)
        {
            await _authManager.ClearTokensAsync(cancellationToken);
            _logger.LogInformation("User logged out");
        }

        /// <summary>
        /// Check if user is currently authenticated
        /// </summary>
        /// <returns>True if authenticated</returns>
        public bool IsAuthenticated()
        {
            return _authManager.IsAuthenticated();
        }

        /// <summary>
        /// Get current access token
        /// </summary>
        /// <returns>Access token or null</returns>
        public string GetAccessToken()
        {
            return _authManager.GetAccessToken();
        }

        /// <summary>
        /// Connect to WebSocket for real-time data
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task ConnectWebSocketAsync(CancellationToken cancellationToken = default)
        {
            await _webSocketManager.ConnectAsync(cancellationToken);
        }

        /// <summary>
        /// Disconnect from WebSocket
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task DisconnectWebSocketAsync(CancellationToken cancellationToken = default)
        {
            await _webSocketManager.DisconnectAsync(cancellationToken);
        }

        /// <summary>
        /// Subscribe to real-time market data
        /// </summary>
        /// <param name="symbols">Symbols to subscribe to</param>
        /// <param name="callback">Callback for received data</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task SubscribeToMarketDataAsync(
            IEnumerable<string> symbols,
            Action<MarketDataUpdate> callback,
            CancellationToken cancellationToken = default)
        {
            await _webSocketManager.SubscribeAsync(
                "market_data",
                new { symbols = symbols.ToList() },
                callback,
                cancellationToken
            );
        }

        /// <summary>
        /// Subscribe to trade updates
        /// </summary>
        /// <param name="callback">Callback for trade updates</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task SubscribeToTradeUpdatesAsync(
            Action<TradeUpdate> callback,
            CancellationToken cancellationToken = default)
        {
            await _webSocketManager.SubscribeAsync(
                "trade_updates", 
                new { },
                callback, 
                cancellationToken
            );
        }

        /// <summary>
        /// Subscribe to risk alerts
        /// </summary>
        /// <param name="callback">Callback for risk alerts</param>
        /// <param name="cancellationToken">Cancellation token</param>
        public async Task SubscribeToRiskAlertsAsync(
            Action<RiskAlert> callback,
            CancellationToken cancellationToken = default)
        {
            await _webSocketManager.SubscribeAsync(
                "risk_alerts",
                new { },
                callback,
                cancellationToken
            );
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            _webSocketManager?.Dispose();
            _httpClient?.Dispose();
            _logger.LogInformation("Nautilus client disposed");
        }
    }

    /// <summary>
    /// Configuration for Nautilus client
    /// </summary>
    public class NautilusConfig
    {
        /// <summary>
        /// Base URL for the API
        /// </summary>
        public string BaseUrl { get; set; } = "http://localhost:8001";

        /// <summary>
        /// WebSocket URL
        /// </summary>
        public string WebSocketUrl { get; set; } = "ws://localhost:8001";

        /// <summary>
        /// Request timeout in milliseconds
        /// </summary>
        public int TimeoutMs { get; set; } = 30000;

        /// <summary>
        /// Maximum number of retries for failed requests
        /// </summary>
        public int MaxRetries { get; set; } = 3;

        /// <summary>
        /// Delay between retries in milliseconds
        /// </summary>
        public int RetryDelayMs { get; set; } = 1000;

        /// <summary>
        /// API key for authentication (optional)
        /// </summary>
        public string ApiKey { get; set; }

        /// <summary>
        /// Enable debug logging
        /// </summary>
        public bool EnableDebugLogging { get; set; } = false;
    }

    /// <summary>
    /// Generic API response wrapper
    /// </summary>
    /// <typeparam name="T">Response data type</typeparam>
    public class ApiResponse<T>
    {
        public T Data { get; set; }
        public int StatusCode { get; set; }
        public Dictionary<string, string> Headers { get; set; } = new();
    }

    /// <summary>
    /// Standard error response
    /// </summary>
    public class ErrorResponse
    {
        public string Error { get; set; }
        public string Message { get; set; }
        public object Details { get; set; }
        public DateTime Timestamp { get; set; }
        public string RequestId { get; set; }
    }
}

/* Usage Example:

// Configuration
var config = new NautilusConfig
{
    BaseUrl = "http://localhost:8001",
    TimeoutMs = 30000,
    MaxRetries = 3
};

// Create client
using var client = new NautilusClient(config);

// Authentication
await client.LoginAsync("trader@nautilus.com", "password");

// Get market data
var quote = await client.MarketData.GetQuoteAsync("AAPL");
Console.WriteLine($"AAPL: ${quote.Price}");

// Create risk limit
var riskLimit = await client.Risk.CreateLimitAsync(new CreateRiskLimitRequest
{
    Type = RiskLimitType.PositionLimit,
    Value = 1000000,
    Symbol = "AAPL",
    WarningThreshold = 0.8m
});

// Deploy strategy
var deployment = await client.Strategies.DeployAsync(new DeployStrategyRequest
{
    Name = "EMA_Cross",
    Version = "1.0.0",
    Description = "EMA crossover strategy",
    Parameters = new Dictionary<string, object>
    {
        {"fast_ema", 12},
        {"slow_ema", 26}
    }
});

// Real-time market data
await client.SubscribeToMarketDataAsync(
    new[] {"AAPL", "GOOGL"}, 
    data => Console.WriteLine($"Market update: {data}")
);

// Keep connection alive
await Task.Delay(-1);

*/