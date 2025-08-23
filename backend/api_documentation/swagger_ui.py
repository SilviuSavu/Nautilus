"""
Enhanced Swagger UI Integration with Interactive Features
Provides comprehensive API documentation with live testing capabilities
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import json
from typing import Dict, Any, Optional
from openapi_spec import OpenAPIGenerator


class SwaggerUIEnhanced:
    """Enhanced Swagger UI with interactive features and customizations"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.openapi_generator = OpenAPIGenerator()
        self.setup_swagger_routes()
        
    def setup_swagger_routes(self):
        """Setup enhanced Swagger UI routes"""
        
        @self.app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
        async def custom_swagger_ui():
            """Enhanced Swagger UI with custom features"""
            return self.generate_enhanced_swagger_html()
        
        @self.app.get("/docs/openapi.json", include_in_schema=False)
        async def get_openapi_spec():
            """Get the complete OpenAPI specification"""
            return JSONResponse(self.openapi_generator.generate_complete_spec())
        
        @self.app.get("/docs/redoc", response_class=HTMLResponse, include_in_schema=False)
        async def redoc():
            """Alternative ReDoc documentation"""
            return self.generate_redoc_html()
        
        @self.app.get("/docs/playground", response_class=HTMLResponse, include_in_schema=False)
        async def api_playground():
            """Interactive API playground"""
            return self.generate_playground_html()
        
        @self.app.get("/docs/examples", include_in_schema=False)
        async def get_code_examples():
            """Get comprehensive code examples for all endpoints"""
            return JSONResponse(self.generate_code_examples())
        
        @self.app.get("/docs/postman", include_in_schema=False)
        async def get_postman_collection():
            """Generate Postman collection for API testing"""
            return JSONResponse(self.generate_postman_collection())
    
    def generate_enhanced_swagger_html(self) -> str:
        """Generate enhanced Swagger UI HTML with custom features"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nautilus Trading Platform API - Enhanced Documentation</title>
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui.css" />
            <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5.10.5/favicon-32x32.png" sizes="32x32" />
            <style>
                .swagger-ui .topbar {{ display: none }}
                .swagger-ui .info {{ margin: 50px 0 }}
                .swagger-ui .info .title {{ 
                    font-size: 36px;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .swagger-ui .info .description {{
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .custom-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 8px;
                }}
                .feature-badges {{
                    display: flex;
                    gap: 10px;
                    margin: 20px 0;
                    flex-wrap: wrap;
                }}
                .badge {{
                    background: #27ae60;
                    color: white;
                    padding: 5px 12px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                .quick-links {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .quick-links h3 {{
                    margin-top: 0;
                    color: #2c3e50;
                }}
                .quick-links a {{
                    display: inline-block;
                    margin: 5px 10px 5px 0;
                    padding: 8px 16px;
                    background: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    font-size: 14px;
                }}
                .quick-links a:hover {{
                    background: #2980b9;
                }}
                .auth-info {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    padding: 15px;
                    border-radius: 6px;
                    margin: 20px 0;
                }}
                .performance-metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div id="custom-header" class="custom-header">
                <h1>🚀 Nautilus Trading Platform API</h1>
                <p>Enterprise-grade trading platform with 8-source data integration, real-time streaming, and advanced risk management</p>
                
                <div class="feature-badges">
                    <span class="badge">8 Data Sources</span>
                    <span class="badge">380K+ Factors</span>
                    <span class="badge">Real-time WebSocket</span>
                    <span class="badge">ML Risk Management</span>
                    <span class="badge">Strategy Deployment</span>
                    <span class="badge">50+ Endpoints</span>
                </div>
            </div>
            
            <div class="quick-links">
                <h3>📚 Quick Start & Resources</h3>
                <a href="/docs/playground">🎮 API Playground</a>
                <a href="/docs/examples">💻 Code Examples</a>
                <a href="/docs/postman">📬 Postman Collection</a>
                <a href="/docs/redoc">📖 ReDoc View</a>
                <a href="/ws-test">🔌 WebSocket Tester</a>
                <a href="/docs/sdk">📦 SDKs</a>
            </div>
            
            <div class="auth-info">
                <h4>🔐 Authentication Quick Start</h4>
                <p><strong>1.</strong> Get your API key from <code>POST /api/v1/auth/login</code></p>
                <p><strong>2.</strong> Add to requests: <code>Authorization: Bearer YOUR_TOKEN</code></p>
                <p><strong>3.</strong> Use the "Authorize" button below to test authenticated endpoints</p>
            </div>
            
            <div class="performance-metrics">
                <div class="metric">
                    <div class="metric-value">1000+</div>
                    <div class="metric-label">WebSocket Connections</div>
                </div>
                <div class="metric">
                    <div class="metric-value">50K+</div>
                    <div class="metric-label">Messages/Second</div>
                </div>
                <div class="metric">
                    <div class="metric-value">5ms</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">99.9%</div>
                    <div class="metric-label">Uptime SLA</div>
                </div>
            </div>
            
            <div id="swagger-ui"></div>
            
            <script src="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui-bundle.js"></script>
            <script src="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui-standalone-preset.js"></script>
            <script>
                window.onload = function() {{
                    const ui = SwaggerUIBundle({{
                        url: '/docs/openapi.json',
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset
                        ],
                        plugins: [
                            SwaggerUIBundle.plugins.DownloadUrl
                        ],
                        layout: "StandaloneLayout",
                        defaultModelsExpandDepth: 1,
                        defaultModelExpandDepth: 1,
                        displayOperationId: true,
                        displayRequestDuration: true,
                        filter: true,
                        showExtensions: true,
                        showCommonExtensions: true,
                        tryItOutEnabled: true,
                        requestSnippetsEnabled: true,
                        requestSnippets: {{
                            generators: {{
                                "curl_bash": {{
                                    title: "cURL (bash)",
                                    syntax: "bash"
                                }},
                                "curl_powershell": {{
                                    title: "cURL (PowerShell)",
                                    syntax: "powershell"
                                }},
                                "curl_cmd": {{
                                    title: "cURL (CMD)",
                                    syntax: "bash"
                                }}
                            }},
                            defaultExpanded: true,
                            languages: null
                        }},
                        onComplete: function(swaggerApi, swaggerUi) {{
                            console.log("Nautilus API Documentation loaded");
                            
                            // Add custom features
                            addPerformanceMetrics();
                            addQuickTestButtons();
                            highlightNewFeatures();
                        }}
                    }});
                    
                    function addPerformanceMetrics() {{
                        // Add real-time performance metrics
                        setInterval(() => {{
                            fetch('/api/v1/system/metrics')
                                .then(response => response.json())
                                .then(data => {{
                                    // Update performance metrics display
                                    updateMetrics(data);
                                }})
                                .catch(err => console.log('Metrics update failed:', err));
                        }}, 30000); // Update every 30 seconds
                    }}
                    
                    function addQuickTestButtons() {{
                        // Add quick test buttons for common endpoints
                        const operations = document.querySelectorAll('.opblock');
                        operations.forEach(op => {{
                            const quickTestBtn = document.createElement('button');
                            quickTestBtn.textContent = '⚡ Quick Test';
                            quickTestBtn.className = 'btn quick-test-btn';
                            quickTestBtn.style.cssText = 'margin-left: 10px; background: #27ae60; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;';
                            
                            const summary = op.querySelector('.opblock-summary');
                            if (summary) {{
                                summary.appendChild(quickTestBtn);
                                
                                quickTestBtn.addEventListener('click', (e) => {{
                                    e.stopPropagation();
                                    // Auto-populate common test data
                                    populateTestData(op);
                                }});
                            }}
                        }});
                    }}
                    
                    function highlightNewFeatures() {{
                        // Highlight Sprint 3 features
                        const sprint3Tags = ['WebSocket Streaming', 'Risk Management', 'Strategy Management'];
                        sprint3Tags.forEach(tag => {{
                            const tagElements = document.querySelectorAll(`[data-tag="${{tag}}"]`);
                            tagElements.forEach(el => {{
                                el.style.background = 'linear-gradient(45deg, #ff6b6b, #4ecdc4)';
                                el.style.color = 'white';
                                
                                // Add "NEW" badge
                                const newBadge = document.createElement('span');
                                newBadge.textContent = '🆕 NEW';
                                newBadge.style.cssText = 'background: #e74c3c; color: white; font-size: 10px; padding: 2px 6px; border-radius: 10px; margin-left: 8px;';
                                el.appendChild(newBadge);
                            }});
                        }});
                    }}
                    
                    function populateTestData(operation) {{
                        // Auto-populate with sample test data
                        const pathElement = operation.querySelector('.opblock-summary-path');
                        if (pathElement) {{
                            const path = pathElement.textContent;
                            
                            if (path.includes('/market-data/quote/')) {{
                                // Populate with AAPL for market data endpoints
                                const symbolInput = operation.querySelector('input[placeholder*="symbol"]');
                                if (symbolInput) symbolInput.value = 'AAPL';
                            }} else if (path.includes('/risk/')) {{
                                // Populate risk management test data
                                const limitInput = operation.querySelector('input[name*="limit"]');
                                if (limitInput) limitInput.value = '1000000';
                            }}
                        }}
                    }}
                    
                    function updateMetrics(data) {{
                        if (data.websocket_connections) {{
                            const wsMetric = document.querySelector('.metric:nth-child(1) .metric-value');
                            if (wsMetric) wsMetric.textContent = data.websocket_connections + '+';
                        }}
                    }}
                }};
            </script>
        </body>
        </html>
        """
    
    def generate_redoc_html(self) -> str:
        """Generate ReDoc documentation HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nautilus Trading Platform API - ReDoc</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; }}
                redoc {{
                    --redoc-primary-color: #2c3e50;
                    --redoc-primary-color-dark: #1a252f;
                }}
            </style>
        </head>
        <body>
            <redoc spec-url='/docs/openapi.json' expand-responses="200,201"></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """
    
    def generate_playground_html(self) -> str:
        """Generate interactive API playground HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nautilus API Playground</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0; padding: 20px; background: #f8f9fa;
                }}
                .playground-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                .playground-container {{
                    display: grid; grid-template-columns: 1fr 1fr; gap: 30px;
                    max-width: 1400px; margin: 0 auto;
                }}
                .panel {{
                    background: white; border-radius: 12px; padding: 25px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                }}
                .panel h3 {{ margin-top: 0; color: #2c3e50; }}
                .form-group {{ margin-bottom: 20px; }}
                .form-group label {{ 
                    display: block; margin-bottom: 8px; font-weight: 600;
                    color: #34495e;
                }}
                .form-group input, .form-group select, .form-group textarea {{
                    width: 100%; padding: 12px; border: 2px solid #e1e8ed;
                    border-radius: 6px; font-size: 14px;
                    transition: border-color 0.2s ease;
                }}
                .form-group input:focus, .form-group select:focus, .form-group textarea:focus {{
                    outline: none; border-color: #3498db;
                }}
                .btn {{
                    background: #3498db; color: white; border: none;
                    padding: 12px 24px; border-radius: 6px; cursor: pointer;
                    font-size: 14px; font-weight: 600;
                    transition: background-color 0.2s ease;
                }}
                .btn:hover {{ background: #2980b9; }}
                .btn-success {{ background: #27ae60; }}
                .btn-success:hover {{ background: #229954; }}
                .response-container {{
                    background: #2c3e50; color: #ecf0f1; padding: 20px;
                    border-radius: 8px; margin-top: 20px; font-family: 'Monaco', monospace;
                    max-height: 400px; overflow-y: auto;
                }}
                .quick-tests {{
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px; margin-bottom: 30px;
                }}
                .quick-test {{
                    background: white; padding: 15px; border-radius: 8px;
                    text-align: center; cursor: pointer;
                    transition: transform 0.2s ease;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}
                .quick-test:hover {{ transform: translateY(-2px); }}
                .websocket-panel {{
                    grid-column: 1 / -1; margin-top: 20px;
                }}
                .ws-messages {{
                    background: #2c3e50; color: #ecf0f1; padding: 15px;
                    border-radius: 8px; height: 200px; overflow-y: auto;
                    font-family: 'Monaco', monospace; font-size: 12px;
                }}
                .status-indicator {{
                    display: inline-block; width: 12px; height: 12px;
                    border-radius: 50%; margin-right: 8px;
                }}
                .status-connected {{ background: #27ae60; }}
                .status-disconnected {{ background: #e74c3c; }}
                .status-connecting {{ background: #f39c12; }}
            </style>
        </head>
        <body>
            <div class="playground-header">
                <h1>🎮 Nautilus API Playground</h1>
                <p>Interactive testing environment for all Nautilus Trading Platform endpoints</p>
            </div>
            
            <div class="quick-tests">
                <div class="quick-test" onclick="loadQuickTest('health')">
                    <h4>🏥 Health Check</h4>
                    <p>Test system status</p>
                </div>
                <div class="quick-test" onclick="loadQuickTest('auth')">
                    <h4>🔐 Authentication</h4>
                    <p>Get access token</p>
                </div>
                <div class="quick-test" onclick="loadQuickTest('market_data')">
                    <h4>📈 Market Data</h4>
                    <p>Real-time quotes</p>
                </div>
                <div class="quick-test" onclick="loadQuickTest('websocket')">
                    <h4>🔌 WebSocket</h4>
                    <p>Live streaming</p>
                </div>
            </div>
            
            <div class="playground-container">
                <div class="panel">
                    <h3>🔧 Request Builder</h3>
                    
                    <div class="form-group">
                        <label>HTTP Method</label>
                        <select id="method">
                            <option value="GET">GET</option>
                            <option value="POST">POST</option>
                            <option value="PUT">PUT</option>
                            <option value="DELETE">DELETE</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Endpoint URL</label>
                        <input type="text" id="url" placeholder="/api/v1/health" value="/health">
                    </div>
                    
                    <div class="form-group">
                        <label>Headers (JSON format)</label>
                        <textarea id="headers" rows="3" placeholder='{{"Authorization": "Bearer YOUR_TOKEN"}}'>{{}}</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>Request Body (JSON format)</label>
                        <textarea id="body" rows="4" placeholder='{{"symbol": "AAPL"}}'></textarea>
                    </div>
                    
                    <button class="btn" onclick="sendRequest()">🚀 Send Request</button>
                    <button class="btn btn-success" onclick="generateCode()">💻 Generate Code</button>
                </div>
                
                <div class="panel">
                    <h3>📋 Response</h3>
                    <div id="response-status"></div>
                    <div class="response-container">
                        <pre id="response-body">Click "Send Request" to see response...</pre>
                    </div>
                </div>
                
                <div class="panel websocket-panel">
                    <h3>🔌 WebSocket Tester</h3>
                    
                    <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 15px;">
                        <div class="form-group" style="flex: 1; margin-bottom: 0;">
                            <input type="text" id="ws-url" placeholder="ws://localhost:8001/ws/market-data/AAPL" 
                                   value="ws://localhost:8001/ws/market-data/AAPL">
                        </div>
                        <button class="btn" onclick="connectWebSocket()">
                            <span class="status-indicator status-disconnected" id="ws-status"></span>
                            Connect
                        </button>
                    </div>
                    
                    <div class="form-group">
                        <input type="text" id="ws-message" placeholder="Enter message to send (JSON format)"
                               value='{{"type": "subscribe", "symbol": "AAPL"}}'>
                        <button class="btn" onclick="sendWebSocketMessage()" style="margin-top: 10px;">Send Message</button>
                    </div>
                    
                    <div class="ws-messages" id="ws-messages">
                        WebSocket messages will appear here...
                    </div>
                </div>
            </div>
            
            <script>
                let websocket = null;
                let messageCount = 0;
                
                // Quick test templates
                const quickTests = {{
                    health: {{
                        method: 'GET',
                        url: '/health',
                        headers: {{}},
                        body: ''
                    }},
                    auth: {{
                        method: 'POST',
                        url: '/api/v1/auth/login',
                        headers: {{"Content-Type": "application/json"}},
                        body: JSON.stringify({{
                            "username": "demo@nautilus.com",
                            "password": "demo123"
                        }}, null, 2)
                    }},
                    market_data: {{
                        method: 'GET',
                        url: '/api/v1/market-data/quote/AAPL',
                        headers: {{"Authorization": "Bearer YOUR_TOKEN"}},
                        body: ''
                    }},
                    websocket: {{
                        method: 'WebSocket',
                        url: 'ws://localhost:8001/ws/market-data/AAPL',
                        headers: {{"Authorization": "Bearer YOUR_TOKEN"}},
                        body: ''
                    }}
                }};
                
                function loadQuickTest(testName) {{
                    const test = quickTests[testName];
                    if (test) {{
                        document.getElementById('method').value = test.method;
                        document.getElementById('url').value = test.url;
                        document.getElementById('headers').value = JSON.stringify(test.headers, null, 2);
                        document.getElementById('body').value = test.body;
                    }}
                }}
                
                async function sendRequest() {{
                    const method = document.getElementById('method').value;
                    const url = document.getElementById('url').value;
                    const headersText = document.getElementById('headers').value;
                    const bodyText = document.getElementById('body').value;
                    
                    try {{
                        const headers = JSON.parse(headersText || '{{}}');
                        const options = {{
                            method: method,
                            headers: headers
                        }};
                        
                        if (bodyText && method !== 'GET') {{
                            options.body = bodyText;
                        }}
                        
                        const startTime = Date.now();
                        const response = await fetch('http://localhost:8001' + url, options);
                        const endTime = Date.now();
                        
                        const responseData = await response.json();
                        
                        // Update response display
                        document.getElementById('response-status').innerHTML = `
                            <div style="background: ${{response.ok ? '#d4edda' : '#f8d7da'}}; 
                                        color: ${{response.ok ? '#155724' : '#721c24'}}; 
                                        padding: 10px; border-radius: 6px; margin-bottom: 15px;">
                                Status: ${{response.status}} ${{response.statusText}} 
                                (Response time: ${{endTime - startTime}}ms)
                            </div>
                        `;
                        
                        document.getElementById('response-body').textContent = 
                            JSON.stringify(responseData, null, 2);
                        
                    }} catch (error) {{
                        document.getElementById('response-status').innerHTML = `
                            <div style="background: #f8d7da; color: #721c24; padding: 10px; border-radius: 6px; margin-bottom: 15px;">
                                Error: ${{error.message}}
                            </div>
                        `;
                        document.getElementById('response-body').textContent = error.toString();
                    }}
                }}
                
                function connectWebSocket() {{
                    const wsUrl = document.getElementById('ws-url').value;
                    const statusIndicator = document.getElementById('ws-status');
                    
                    if (websocket) {{
                        websocket.close();
                        websocket = null;
                        statusIndicator.className = 'status-indicator status-disconnected';
                        return;
                    }}
                    
                    statusIndicator.className = 'status-indicator status-connecting';
                    
                    websocket = new WebSocket(wsUrl);
                    
                    websocket.onopen = function(event) {{
                        statusIndicator.className = 'status-indicator status-connected';
                        addWebSocketMessage('Connected to ' + wsUrl, 'system');
                    }};
                    
                    websocket.onmessage = function(event) {{
                        addWebSocketMessage(event.data, 'received');
                    }};
                    
                    websocket.onclose = function(event) {{
                        statusIndicator.className = 'status-indicator status-disconnected';
                        addWebSocketMessage('Connection closed', 'system');
                        websocket = null;
                    }};
                    
                    websocket.onerror = function(error) {{
                        addWebSocketMessage('Error: ' + error, 'error');
                    }};
                }}
                
                function sendWebSocketMessage() {{
                    if (!websocket) {{
                        alert('Please connect to WebSocket first');
                        return;
                    }}
                    
                    const message = document.getElementById('ws-message').value;
                    websocket.send(message);
                    addWebSocketMessage(message, 'sent');
                }}
                
                function addWebSocketMessage(message, type) {{
                    const messagesDiv = document.getElementById('ws-messages');
                    const timestamp = new Date().toLocaleTimeString();
                    const color = type === 'sent' ? '#3498db' : 
                                 type === 'received' ? '#27ae60' : 
                                 type === 'error' ? '#e74c3c' : '#95a5a6';
                    
                    messagesDiv.innerHTML += `
                        <div style="color: ${{color}}; margin-bottom: 5px;">
                            [${{timestamp}}] ${{type.toUpperCase()}}: ${{message}}
                        </div>
                    `;
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }}
                
                function generateCode() {{
                    const method = document.getElementById('method').value;
                    const url = document.getElementById('url').value;
                    const headersText = document.getElementById('headers').value;
                    const bodyText = document.getElementById('body').value;
                    
                    // Generate code examples
                    const examples = generateCodeExamples(method, url, headersText, bodyText);
                    
                    // Show in a popup or new window
                    const newWindow = window.open('', '_blank', 'width=800,height=600');
                    newWindow.document.write(`
                        <html>
                        <head><title>Generated Code Examples</title></head>
                        <body style="font-family: monospace; padding: 20px;">
                            <h2>Generated Code Examples</h2>
                            ${{examples}}
                        </body>
                        </html>
                    `);
                }}
                
                function generateCodeExamples(method, url, headers, body) {{
                    const fullUrl = 'http://localhost:8001' + url;
                    
                    return `
                        <h3>cURL</h3>
                        <pre style="background: #f4f4f4; padding: 15px; border-radius: 6px;">
curl -X ${{method}} "${{fullUrl}}" \\
  -H "Content-Type: application/json" \\
  ${{headers ? `-H '${{headers}}'` : ''}} \\
  ${{body ? `-d '${{body}}'` : ''}}
                        </pre>
                        
                        <h3>Python (requests)</h3>
                        <pre style="background: #f4f4f4; padding: 15px; border-radius: 6px;">
import requests

headers = ${{headers || '{}'}}
${{body ? `data = ${{body}}` : ''}}

response = requests.${{method.toLowerCase()}}(
    "${{fullUrl}}",
    headers=headers${{body ? ',\\n    json=data' : ''}}
)

print(response.json())
                        </pre>
                        
                        <h3>JavaScript (fetch)</h3>
                        <pre style="background: #f4f4f4; padding: 15px; border-radius: 6px;">
const response = await fetch("${{fullUrl}}", {{
    method: "${{method}}",
    headers: ${{headers || '{}'}}, ${{body ? `,
    body: JSON.stringify(${{body}})` : ''}}
}});

const data = await response.json();
console.log(data);
                        </pre>
                    `;
                }}
            </script>
        </body>
        </html>
        """
    
    def generate_code_examples(self) -> Dict[str, Any]:
        """Generate comprehensive code examples for all endpoints"""
        return {
            "authentication": {
                "curl": [
                    {
                        "title": "Login and get access token",
                        "code": '''curl -X POST "http://localhost:8001/api/v1/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "trader@nautilus.com", "password": "your_password"}'
'''
                    }
                ],
                "python": [
                    {
                        "title": "Authentication with requests",
                        "code": '''import requests

# Login to get access token
login_data = {
    "username": "trader@nautilus.com",
    "password": "your_password"
}

response = requests.post(
    "http://localhost:8001/api/v1/auth/login",
    json=login_data
)

token_data = response.json()
access_token = token_data["access_token"]

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {access_token}"}

# Example authenticated request
market_data = requests.get(
    "http://localhost:8001/api/v1/market-data/quote/AAPL",
    headers=headers
)

print(market_data.json())
'''
                    }
                ],
                "javascript": [
                    {
                        "title": "Authentication with fetch",
                        "code": '''// Login and get access token
const loginResponse = await fetch("http://localhost:8001/api/v1/auth/login", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        username: "trader@nautilus.com",
        password: "your_password"
    })
});

const tokenData = await loginResponse.json();
const accessToken = tokenData.access_token;

// Use token for authenticated requests
const marketData = await fetch("http://localhost:8001/api/v1/market-data/quote/AAPL", {
    headers: {"Authorization": `Bearer ${accessToken}`}
});

const data = await marketData.json();
console.log(data);
'''
                    }
                ]
            },
            "websocket": {
                "javascript": [
                    {
                        "title": "WebSocket real-time market data",
                        "code": '''// Connect to WebSocket with authentication
const ws = new WebSocket("ws://localhost:8001/ws/market-data/AAPL", [], {
    headers: {
        "Authorization": "Bearer " + accessToken
    }
});

ws.onopen = function(event) {
    console.log("Connected to market data stream");
    
    // Subscribe to specific symbols
    ws.send(JSON.stringify({
        type: "subscribe",
        symbols: ["AAPL", "GOOGL", "MSFT"],
        data_types: ["quotes", "trades"]
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case "market_data":
            console.log("Market update:", message.data);
            updatePriceDisplay(message.data);
            break;
            
        case "trade_update":
            console.log("Trade executed:", message.data);
            updateTradeHistory(message.data);
            break;
            
        case "heartbeat":
            console.log("Connection alive");
            break;
    }
};

ws.onerror = function(error) {
    console.error("WebSocket error:", error);
};

ws.onclose = function(event) {
    console.log("Connection closed:", event.code);
    // Implement reconnection logic
    setTimeout(connectWebSocket, 5000);
};
'''
                    }
                ],
                "python": [
                    {
                        "title": "WebSocket with asyncio",
                        "code": '''import asyncio
import websockets
import json

async def market_data_stream():
    uri = "ws://localhost:8001/ws/market-data/AAPL"
    headers = {"Authorization": "Bearer " + access_token}
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # Subscribe to market data
        await websocket.send(json.dumps({
            "type": "subscribe",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "data_types": ["quotes", "trades"]
        }))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "market_data":
                print(f"Market update: {data['data']}")
                # Process market data
                
            elif data["type"] == "trade_update":
                print(f"Trade executed: {data['data']}")
                # Process trade update
                
            elif data["type"] == "heartbeat":
                print("Connection alive")

# Run the WebSocket client
asyncio.run(market_data_stream())
'''
                    }
                ]
            },
            "risk_management": {
                "python": [
                    {
                        "title": "Create and monitor risk limits",
                        "code": '''import requests

headers = {"Authorization": f"Bearer {access_token}"}

# Create a position limit
limit_data = {
    "type": "position_limit",
    "symbol": "AAPL",
    "value": 1000000,  # $1M position limit
    "warning_threshold": 0.8,  # 80% warning
    "auto_adjust": True
}

response = requests.post(
    "http://localhost:8001/api/v1/risk/limits",
    headers=headers,
    json=limit_data
)

limit_id = response.json()["id"]

# Monitor risk in real-time
def monitor_risk():
    while True:
        risk_status = requests.get(
            f"http://localhost:8001/api/v1/risk/limits/{limit_id}/check",
            headers=headers
        )
        
        status_data = risk_status.json()
        
        if status_data["status"] == "breached":
            print(f"⚠️ RISK BREACH: {status_data}")
            # Implement breach response
            
        elif status_data["utilization"] > 0.8:
            print(f"⚠️ High utilization: {status_data['utilization']:.2%}")
            
        time.sleep(5)  # Check every 5 seconds

monitor_risk()
'''
                    }
                ]
            },
            "strategy_deployment": {
                "python": [
                    {
                        "title": "Deploy trading strategy",
                        "code": '''import requests

headers = {"Authorization": f"Bearer {access_token}"}

# Deploy a new strategy
strategy_config = {
    "name": "EMA_Cross_Strategy",
    "version": "1.2.0",
    "description": "Exponential Moving Average crossover strategy",
    "parameters": {
        "fast_ema": 12,
        "slow_ema": 26,
        "signal_ema": 9,
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "position_size": 0.02  # 2% of portfolio
    },
    "risk_limits": {
        "max_drawdown": 0.05,  # 5% max drawdown
        "position_limit": 1000000,
        "daily_loss_limit": 50000
    },
    "deployment_config": {
        "environment": "paper",  # Start with paper trading
        "auto_rollback": True,
        "rollback_threshold": 0.02  # Rollback if 2% loss
    }
}

# Deploy strategy
deployment = requests.post(
    "http://localhost:8001/api/v1/strategies/deploy",
    headers=headers,
    json=strategy_config
)

deployment_id = deployment.json()["deployment_id"]
print(f"Strategy deployed: {deployment_id}")

# Monitor deployment status
while True:
    status = requests.get(
        f"http://localhost:8001/api/v1/strategies/pipeline/{deployment_id}/status",
        headers=headers
    )
    
    status_data = status.json()
    print(f"Status: {status_data['status']}")
    
    if status_data["status"] == "deployed":
        print("✅ Strategy successfully deployed!")
        break
    elif status_data["status"] == "failed":
        print(f"❌ Deployment failed: {status_data['error']}")
        break
        
    time.sleep(10)
'''
                    }
                ]
            }
        }
    
    def generate_postman_collection(self) -> Dict[str, Any]:
        """Generate Postman collection for API testing"""
        return {
            "info": {
                "name": "Nautilus Trading Platform API",
                "description": "Complete API collection for Nautilus Trading Platform",
                "version": "3.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{access_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "http://localhost:8001",
                    "type": "string"
                },
                {
                    "key": "access_token",
                    "value": "",
                    "type": "string"
                }
            ],
            "item": [
                {
                    "name": "Authentication",
                    "item": [
                        {
                            "name": "Login",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": '{\n  "username": "trader@nautilus.com",\n  "password": "your_password"\n}'
                                },
                                "url": "{{base_url}}/api/v1/auth/login"
                            },
                            "event": [
                                {
                                    "listen": "test",
                                    "script": {
                                        "exec": [
                                            "if (pm.response.code === 200) {",
                                            "    const response = pm.response.json();",
                                            "    pm.collectionVariables.set('access_token', response.access_token);",
                                            "    console.log('Access token saved:', response.access_token);",
                                            "}"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "Market Data",
                    "item": [
                        {
                            "name": "Get Quote",
                            "request": {
                                "method": "GET",
                                "url": {
                                    "raw": "{{base_url}}/api/v1/market-data/quote/AAPL?source=IBKR",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "v1", "market-data", "quote", "AAPL"],
                                    "query": [
                                        {
                                            "key": "source",
                                            "value": "IBKR"
                                        }
                                    ]
                                }
                            }
                        },
                        {
                            "name": "Get Historical Data",
                            "request": {
                                "method": "GET",
                                "url": {
                                    "raw": "{{base_url}}/api/v1/market-data/historical/AAPL?interval=1day&limit=100",
                                    "host": ["{{base_url}}"],
                                    "path": ["api", "v1", "market-data", "historical", "AAPL"],
                                    "query": [
                                        {
                                            "key": "interval",
                                            "value": "1day"
                                        },
                                        {
                                            "key": "limit",
                                            "value": "100"
                                        }
                                    ]
                                }
                            }
                        }
                    ]
                },
                {
                    "name": "Risk Management",
                    "item": [
                        {
                            "name": "Create Risk Limit",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {
                                        "key": "Content-Type",
                                        "value": "application/json"
                                    }
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": '{\n  "type": "position_limit",\n  "symbol": "AAPL",\n  "value": 1000000,\n  "warning_threshold": 0.8,\n  "auto_adjust": true\n}'
                                },
                                "url": "{{base_url}}/api/v1/risk/limits"
                            }
                        },
                        {
                            "name": "Check Risk Limits",
                            "request": {
                                "method": "GET",
                                "url": "{{base_url}}/api/v1/risk/limits"
                            }
                        }
                    ]
                },
                {
                    "name": "System Monitoring",
                    "item": [
                        {
                            "name": "Health Check",
                            "request": {
                                "method": "GET",
                                "url": "{{base_url}}/health"
                            }
                        },
                        {
                            "name": "System Metrics",
                            "request": {
                                "method": "GET",
                                "url": "{{base_url}}/api/v1/system/metrics"
                            }
                        }
                    ]
                }
            ]
        }


def integrate_swagger_ui(app: FastAPI):
    """Integrate enhanced Swagger UI with FastAPI application"""
    swagger_ui = SwaggerUIEnhanced(app)
    return swagger_ui