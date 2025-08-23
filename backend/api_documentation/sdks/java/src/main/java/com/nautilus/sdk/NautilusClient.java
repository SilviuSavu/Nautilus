/**
 * Nautilus Trading Platform Java SDK
 * Official Java client library for enterprise integration
 */
package com.nautilus.sdk;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.nautilus.sdk.auth.AuthenticationManager;
import com.nautilus.sdk.client.*;
import com.nautilus.sdk.config.NautilusConfig;
import com.nautilus.sdk.exceptions.*;
import com.nautilus.sdk.models.*;
import com.nautilus.sdk.websocket.WebSocketManager;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Main client for interacting with the Nautilus Trading Platform API
 * Provides comprehensive API access with full Java enterprise integration
 */
public class NautilusClient implements AutoCloseable {
    
    private static final Logger logger = LoggerFactory.getLogger(NautilusClient.class);
    private static final String USER_AGENT = "Nautilus-Java-SDK/3.0.0";
    
    private final OkHttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final NautilusConfig config;
    private final AuthenticationManager authManager;
    private final WebSocketManager webSocketManager;
    
    // API clients
    public final MarketDataClient marketData;
    public final RiskManagementClient risk;
    public final StrategyClient strategies;
    public final AnalyticsClient analytics;
    public final SystemClient system;

    /**
     * Create new Nautilus client with configuration
     * 
     * @param config Client configuration
     */
    public NautilusClient(NautilusConfig config) {
        this.config = config;
        
        // Initialize HTTP client with timeouts and retry logic
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(Duration.ofMilliseconds(config.getConnectTimeoutMs()))
            .readTimeout(Duration.ofMilliseconds(config.getReadTimeoutMs()))
            .writeTimeout(Duration.ofMilliseconds(config.getWriteTimeoutMs()))
            .addInterceptor(new AuthenticationInterceptor())
            .addInterceptor(new RetryInterceptor(config.getMaxRetries(), config.getRetryDelayMs()))
            .addInterceptor(new LoggingInterceptor(config.isEnableDebugLogging()))
            .build();
            
        // Initialize JSON mapper
        this.objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            
        // Initialize managers
        this.authManager = new AuthenticationManager(httpClient, objectMapper, config);
        this.webSocketManager = new WebSocketManager(config, authManager, objectMapper);
        
        // Initialize API clients
        this.marketData = new MarketDataClient(this);
        this.risk = new RiskManagementClient(this);
        this.strategies = new StrategyClient(this);
        this.analytics = new AnalyticsClient(this);
        this.system = new SystemClient(this);
        
        logger.info("Nautilus client initialized with base URL: {}", config.getBaseUrl());
    }

    /**
     * Make authenticated HTTP request
     * 
     * @param method HTTP method
     * @param endpoint API endpoint
     * @param requestBody Request body (optional)
     * @param queryParams Query parameters (optional)
     * @param requireAuth Whether authentication is required
     * @param responseClass Response class type
     * @return API response
     */
    public <T> CompletableFuture<ApiResponse<T>> requestAsync(
            HttpMethod method,
            String endpoint,
            Object requestBody,
            Map<String, String> queryParams,
            boolean requireAuth,
            Class<T> responseClass) {
            
        return CompletableFuture.supplyAsync(() -> {
            try {
                // Build request URL
                HttpUrl.Builder urlBuilder = HttpUrl.parse(config.getBaseUrl() + endpoint).newBuilder();
                if (queryParams != null) {
                    queryParams.forEach(urlBuilder::addQueryParameter);
                }
                
                // Build request
                Request.Builder requestBuilder = new Request.Builder()
                    .url(urlBuilder.build())
                    .header("User-Agent", USER_AGENT);
                
                // Add authentication headers
                if (requireAuth) {
                    Map<String, String> authHeaders = authManager.getAuthHeaders();
                    authHeaders.forEach(requestBuilder::header);
                }
                
                // Add request body
                RequestBody body = null;
                if (requestBody != null && (method == HttpMethod.POST || method == HttpMethod.PUT)) {
                    String json = objectMapper.writeValueAsString(requestBody);
                    body = RequestBody.create(json, MediaType.get("application/json"));
                }
                
                // Set HTTP method
                switch (method) {
                    case GET:
                        requestBuilder.get();
                        break;
                    case POST:
                        requestBuilder.post(body != null ? body : RequestBody.create("", null));
                        break;
                    case PUT:
                        requestBuilder.put(body != null ? body : RequestBody.create("", null));
                        break;
                    case DELETE:
                        requestBuilder.delete();
                        break;
                }
                
                // Execute request
                Request request = requestBuilder.build();
                try (Response response = httpClient.newCall(request).execute()) {
                    return handleResponse(response, responseClass);
                }
                
            } catch (Exception e) {
                logger.error("Request failed: {}", e.getMessage(), e);
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Handle HTTP response and parse data
     */
    private <T> ApiResponse<T> handleResponse(Response response, Class<T> responseClass) throws IOException {
        int statusCode = response.code();
        String responseBody = response.body() != null ? response.body().string() : "";
        
        // Handle error responses
        if (!response.isSuccessful()) {
            handleErrorResponse(statusCode, responseBody);
        }
        
        // Parse successful response
        T data = null;
        if (!responseBody.isEmpty() && responseClass != Void.class) {
            data = objectMapper.readValue(responseBody, responseClass);
        }
        
        return new ApiResponse<>(data, statusCode, response.headers().toMultimap());
    }

    /**
     * Handle error responses and throw appropriate exceptions
     */
    private void handleErrorResponse(int statusCode, String responseBody) {
        try {
            ErrorResponse errorResponse = objectMapper.readValue(responseBody, ErrorResponse.class);
            String message = errorResponse.getMessage() != null ? errorResponse.getMessage() : "HTTP " + statusCode + " error";
            
            switch (statusCode) {
                case 401:
                    throw new AuthenticationException(message);
                case 403:
                    throw new AuthorizationException(message);
                case 429:
                    throw new RateLimitException(message);
                case 422:
                    throw new ValidationException(message, errorResponse.getDetails());
                default:
                    throw new NautilusException(message, statusCode, errorResponse);
            }
        } catch (IOException e) {
            throw new NautilusException("HTTP " + statusCode + " error", statusCode, null);
        }
    }

    /**
     * Authenticate user and get access token
     * 
     * @param username Username or email
     * @param password Password
     * @param rememberMe Use long-lived token
     * @return Login response future
     */
    public CompletableFuture<LoginResponse> loginAsync(String username, String password, boolean rememberMe) {
        LoginRequest loginRequest = new LoginRequest(username, password, rememberMe);
        
        return requestAsync(
            HttpMethod.POST,
            "/api/v1/auth/login",
            loginRequest,
            null,
            false,
            LoginResponse.class
        ).thenApply(response -> {
            // Store tokens
            authManager.setTokens(
                response.getData().getAccessToken(),
                response.getData().getRefreshToken()
            );
            
            logger.info("User authenticated successfully");
            return response.getData();
        });
    }

    /**
     * Refresh access token
     * 
     * @return Token refresh response future
     */
    public CompletableFuture<TokenRefreshResponse> refreshTokenAsync() {
        return authManager.refreshTokenAsync();
    }

    /**
     * Logout and clear stored tokens
     * 
     * @return Completion future
     */
    public CompletableFuture<Void> logoutAsync() {
        return CompletableFuture.runAsync(() -> {
            authManager.clearTokens();
            logger.info("User logged out");
        });
    }

    /**
     * Check if user is currently authenticated
     * 
     * @return True if authenticated
     */
    public boolean isAuthenticated() {
        return authManager.isAuthenticated();
    }

    /**
     * Get current access token
     * 
     * @return Access token or null
     */
    public String getAccessToken() {
        return authManager.getAccessToken();
    }

    /**
     * Connect to WebSocket for real-time data
     * 
     * @return Connection future
     */
    public CompletableFuture<Void> connectWebSocketAsync() {
        return webSocketManager.connectAsync();
    }

    /**
     * Disconnect from WebSocket
     * 
     * @return Disconnection future
     */
    public CompletableFuture<Void> disconnectWebSocketAsync() {
        return webSocketManager.disconnectAsync();
    }

    /**
     * Subscribe to real-time market data
     * 
     * @param symbols List of symbols to subscribe to
     * @param callback Callback for received data
     * @return Subscription future
     */
    public CompletableFuture<Void> subscribeToMarketDataAsync(
            List<String> symbols,
            Consumer<MarketDataUpdate> callback) {
        return webSocketManager.subscribeAsync(
            "market_data",
            Map.of("symbols", symbols),
            callback,
            MarketDataUpdate.class
        );
    }

    /**
     * Subscribe to trade updates
     * 
     * @param callback Callback for trade updates
     * @return Subscription future
     */
    public CompletableFuture<Void> subscribeToTradeUpdatesAsync(Consumer<TradeUpdate> callback) {
        return webSocketManager.subscribeAsync(
            "trade_updates",
            Map.of(),
            callback,
            TradeUpdate.class
        );
    }

    /**
     * Subscribe to risk alerts
     * 
     * @param callback Callback for risk alerts
     * @return Subscription future
     */
    public CompletableFuture<Void> subscribeToRiskAlertsAsync(Consumer<RiskAlert> callback) {
        return webSocketManager.subscribeAsync(
            "risk_alerts",
            Map.of(),
            callback,
            RiskAlert.class
        );
    }

    /**
     * Get configuration
     * 
     * @return Client configuration
     */
    public NautilusConfig getConfig() {
        return config;
    }

    /**
     * Get object mapper for JSON operations
     * 
     * @return Jackson ObjectMapper
     */
    public ObjectMapper getObjectMapper() {
        return objectMapper;
    }

    /**
     * Close client and cleanup resources
     */
    @Override
    public void close() {
        try {
            webSocketManager.close();
            httpClient.dispatcher().executorService().shutdown();
            httpClient.connectionPool().evictAll();
            logger.info("Nautilus client closed");
        } catch (Exception e) {
            logger.error("Error closing client", e);
        }
    }

    /**
     * Authentication interceptor for HTTP requests
     */
    private class AuthenticationInterceptor implements Interceptor {
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            
            // Skip auth for login endpoints
            if (request.url().encodedPath().contains("/auth/login")) {
                return chain.proceed(request);
            }
            
            // Add authentication headers
            if (authManager.isAuthenticated()) {
                Map<String, String> authHeaders = authManager.getAuthHeaders();
                Request.Builder builder = request.newBuilder();
                authHeaders.forEach(builder::header);
                request = builder.build();
            }
            
            Response response = chain.proceed(request);
            
            // Handle token refresh on 401
            if (response.code() == 401 && authManager.canRefreshToken()) {
                response.close();
                
                try {
                    authManager.refreshTokenAsync().get(5, TimeUnit.SECONDS);
                    
                    // Retry with new token
                    Map<String, String> authHeaders = authManager.getAuthHeaders();
                    Request.Builder builder = request.newBuilder();
                    authHeaders.forEach(builder::header);
                    return chain.proceed(builder.build());
                    
                } catch (Exception e) {
                    logger.error("Token refresh failed", e);
                    throw new IOException("Authentication failed", e);
                }
            }
            
            return response;
        }
    }

    /**
     * Retry interceptor for failed requests
     */
    private class RetryInterceptor implements Interceptor {
        private final int maxRetries;
        private final int retryDelayMs;
        
        public RetryInterceptor(int maxRetries, int retryDelayMs) {
            this.maxRetries = maxRetries;
            this.retryDelayMs = retryDelayMs;
        }
        
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            IOException lastException = null;
            
            for (int attempt = 0; attempt <= maxRetries; attempt++) {
                try {
                    Response response = chain.proceed(request);
                    
                    // Don't retry on successful responses or client errors
                    if (response.isSuccessful() || response.code() < 500) {
                        return response;
                    }
                    
                    response.close();
                    
                    if (attempt < maxRetries) {
                        try {
                            Thread.sleep(retryDelayMs * (attempt + 1));
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                    
                } catch (IOException e) {
                    lastException = e;
                    
                    if (attempt < maxRetries) {
                        logger.warn("Request failed (attempt {}), retrying: {}", attempt + 1, e.getMessage());
                        try {
                            Thread.sleep(retryDelayMs * (attempt + 1));
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            break;
                        }
                    }
                }
            }
            
            throw new IOException("Request failed after " + maxRetries + " retries", lastException);
        }
    }

    /**
     * Logging interceptor for debug purposes
     */
    private class LoggingInterceptor implements Interceptor {
        private final boolean enabled;
        
        public LoggingInterceptor(boolean enabled) {
            this.enabled = enabled;
        }
        
        @Override
        public Response intercept(Chain chain) throws IOException {
            Request request = chain.request();
            
            if (enabled) {
                logger.debug("→ {} {}", request.method(), request.url());
            }
            
            long startTime = System.currentTimeMillis();
            Response response = chain.proceed(request);
            long duration = System.currentTimeMillis() - startTime;
            
            if (enabled) {
                logger.debug("← {} {} ({}ms)", response.code(), request.url(), duration);
            }
            
            return response;
        }
    }

    /**
     * HTTP methods enum
     */
    public enum HttpMethod {
        GET, POST, PUT, DELETE
    }

    /**
     * Builder for creating Nautilus client instances
     */
    public static class Builder {
        private NautilusConfig config = new NautilusConfig();
        
        public Builder baseUrl(String baseUrl) {
            config.setBaseUrl(baseUrl);
            return this;
        }
        
        public Builder webSocketUrl(String webSocketUrl) {
            config.setWebSocketUrl(webSocketUrl);
            return this;
        }
        
        public Builder timeouts(int connectTimeoutMs, int readTimeoutMs, int writeTimeoutMs) {
            config.setConnectTimeoutMs(connectTimeoutMs);
            config.setReadTimeoutMs(readTimeoutMs);
            config.setWriteTimeoutMs(writeTimeoutMs);
            return this;
        }
        
        public Builder retryPolicy(int maxRetries, int retryDelayMs) {
            config.setMaxRetries(maxRetries);
            config.setRetryDelayMs(retryDelayMs);
            return this;
        }
        
        public Builder apiKey(String apiKey) {
            config.setApiKey(apiKey);
            return this;
        }
        
        public Builder enableDebugLogging(boolean enabled) {
            config.setEnableDebugLogging(enabled);
            return this;
        }
        
        public NautilusClient build() {
            return new NautilusClient(config);
        }
    }
}

/* Usage Example:

// Create client
NautilusClient client = new NautilusClient.Builder()
    .baseUrl("http://localhost:8001")
    .timeouts(5000, 30000, 30000)
    .retryPolicy(3, 1000)
    .enableDebugLogging(true)
    .build();

try {
    // Authentication
    LoginResponse loginResponse = client.loginAsync("trader@nautilus.com", "password", false)
        .get(10, TimeUnit.SECONDS);
    System.out.println("Authenticated: " + loginResponse.getAccessToken().substring(0, 20) + "...");
    
    // Get market data
    MarketData quote = client.marketData.getQuoteAsync("AAPL")
        .get(5, TimeUnit.SECONDS);
    System.out.println("AAPL: $" + quote.getPrice());
    
    // Create risk limit
    CreateRiskLimitRequest riskLimitRequest = new CreateRiskLimitRequest()
        .setType(RiskLimitType.POSITION_LIMIT)
        .setValue(1000000.0)
        .setSymbol("AAPL")
        .setWarningThreshold(0.8);
        
    RiskLimit riskLimit = client.risk.createLimitAsync(riskLimitRequest)
        .get(5, TimeUnit.SECONDS);
    System.out.println("Risk limit created: " + riskLimit.getId());
    
    // Deploy strategy
    DeployStrategyRequest strategyRequest = new DeployStrategyRequest()
        .setName("EMA_Cross")
        .setVersion("1.0.0")
        .setDescription("EMA crossover strategy")
        .setParameters(Map.of("fast_ema", 12, "slow_ema", 26));
        
    DeployStrategyResponse deployment = client.strategies.deployAsync(strategyRequest)
        .get(10, TimeUnit.SECONDS);
    System.out.println("Strategy deployed: " + deployment.getDeploymentId());
    
    // Real-time market data
    client.subscribeToMarketDataAsync(
        List.of("AAPL", "GOOGL"),
        data -> System.out.println("Market update: " + data)
    ).get(5, TimeUnit.SECONDS);
    
    // Keep running
    Thread.sleep(60000);
    
} catch (Exception e) {
    e.printStackTrace();
} finally {
    client.close();
}

*/